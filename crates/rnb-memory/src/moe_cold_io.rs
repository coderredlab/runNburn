//! O_DIRECT-backed cold MoE expert reader.
//!
//! This module owns the cold-tier IO path after policy has selected a cold
//! `.rnb` file. It keeps cold expert reads out of the kernel pagecache and
//! exposes per-expert reads to the model runtime without owning router
//! decisions or layer execution.
//!
//! Session 64 axis G.
//!
//! When a hot/cold split `.rnb` exists (`<output>.cold.rnb`), the cold tail is
//! never `mmap`ed — instead this module opens it with `O_DIRECT` and `pread`s
//! per-expert bytes on demand into a per-call aligned buffer. The cold IO
//! bypasses the kernel pagecache entirely, so the hot `mmap` (a few GB) keeps
//! its pagecache pressure isolated and the LRU never evicts hot pages on cold
//! access.
//!
//! See `docs/superpowers/plans/2026-04-21-session64-summary.md` axis G.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::unix::fs::OpenOptionsExt;

/// Cold-tier MoE reader. Holds an `O_DIRECT` file descriptor and per-tensor
/// `(file_offset, per_expert_bytes, n_cold_experts)` for every cold MoE
/// tensor.
pub struct ColdReader {
    file: std::fs::File,
    tensor_index: HashMap<String, (u64, usize, usize)>,
    /// 4 KB-aligned scratch buffer pool — one per worker thread to avoid
    /// contention on concurrent `read_expert` calls.
    buffers: Vec<Mutex<AlignedBuf>>,
    /// Session 68 Phase 2 hot-off experiment: optional second file handle
    /// pointing at the hot `.rnb` (opened with `O_DIRECT` at attach time).
    /// When set, `read_expert_unified` can route the hot range (rank
    /// `0..n_hot`) through pread on this file instead of relying on the
    /// mmap hot path. Gated by `ColdReader::attach_hot_file`; unset for
    /// the default runtime path.
    hot_file: Option<std::fs::File>,
    /// Per-tensor `(base_off, per_expert_bytes, n_hot_experts)` for every
    /// MoE tensor in the hot `.rnb`. Parallel to `tensor_index` but
    /// referring to the hot-file stream.
    hot_tensor_index: HashMap<String, (u64, usize, usize)>,
    /// Cumulative diagnostics (atomic so they're cheap on the hot path).
    /// Read via `stats()`; reset via `reset_stats()`.
    pub stats_n_calls: AtomicU64,
    pub stats_total_ns: AtomicU64,
    pub stats_total_bytes: AtomicU64,
}

/// 4 KB-aligned heap buffer.
pub struct AlignedBuf {
    raw: Vec<u8>,
    aligned_offset: usize,
    capacity: usize,
}

impl AlignedBuf {
    fn with_capacity(min_capacity: usize) -> Self {
        let aligned_cap = (min_capacity + 4095) & !4095;
        let raw = vec![0u8; aligned_cap + 4096];
        let raw_addr = raw.as_ptr() as usize;
        let aligned_offset = (4096 - (raw_addr & 4095)) & 4095;
        Self {
            raw,
            aligned_offset,
            capacity: aligned_cap,
        }
    }

    fn as_aligned_mut(&mut self, len: usize) -> &mut [u8] {
        assert!(
            len <= self.capacity,
            "len {} > buf capacity {}",
            len,
            self.capacity
        );
        let ptr = unsafe { self.raw.as_mut_ptr().add(self.aligned_offset) };
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

fn open_direct_read_file(path: &Path) -> io::Result<std::fs::File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        opts.custom_flags(libc::O_DIRECT);
    }
    opts.open(path)
}

#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[derive(Default)]
pub struct DirectFileReaderCache {
    files: HashMap<PathBuf, std::fs::File>,
}

impl DirectFileReaderCache {
    pub fn read_aligned(
        &mut self,
        path: &Path,
        file_offset: u64,
        destination: &mut [u8],
        required_len: usize,
    ) -> io::Result<usize> {
        const ALIGNMENT: usize = 4096;
        if !file_offset.is_multiple_of(ALIGNMENT as u64)
            || !(destination.as_ptr() as usize).is_multiple_of(ALIGNMENT)
            || !destination.len().is_multiple_of(ALIGNMENT)
            || required_len > destination.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct file read requires aligned offset, buffer, and length",
            ));
        }
        let file = if let Some(file) = self.files.get(path) {
            file
        } else {
            let file = open_direct_read_file(path)?;
            self.files.insert(path.to_path_buf(), file);
            self.files
                .get(path)
                .expect("direct file handle inserted before lookup")
        };
        loop {
            match read_at(file, destination, file_offset) {
                Ok(read) if read >= required_len => return Ok(read),
                Ok(read) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "short direct read: got {read} bytes, need {required_len} from {} at {file_offset}",
                            path.display()
                        ),
                    ))
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn cold_io_mode_label() -> &'static str {
    "O_DIRECT"
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn cold_io_mode_label() -> &'static str {
    "positioned read"
}

impl ColdReader {
    /// Open the cold `.rnb` with `O_DIRECT`, parse its tensor table directly
    /// (without mmap), and allocate `n_buffers` 4 KB-aligned scratch buffers
    /// (one per worker thread).
    pub fn open(path: &Path, max_expert_bytes: usize, n_buffers: usize) -> io::Result<Self> {
        let file = open_direct_read_file(path)?;

        // Build the tensor index by reading the header + tensor table from a
        // SECOND, regular (non-O_DIRECT) handle so we don't have to worry
        // about alignment for the small one-time parse.
        let tensor_index = parse_rnb_tensor_index(path)?;

        let buffers = (0..n_buffers.max(1))
            .map(|_| Mutex::new(AlignedBuf::with_capacity(max_expert_bytes)))
            .collect();

        Ok(Self {
            file,
            tensor_index,
            buffers,
            hot_file: None,
            hot_tensor_index: HashMap::new(),
            stats_n_calls: AtomicU64::new(0),
            stats_total_ns: AtomicU64::new(0),
            stats_total_bytes: AtomicU64::new(0),
        })
    }

    /// Attach the hot `.rnb` so `read_expert_unified` can pread its range
    /// through `O_DIRECT` instead of relying on the mmap hot path. Called
    /// once from the engine after the cold reader is constructed.
    pub fn attach_hot_file(&mut self, hot_path: &Path) -> io::Result<()> {
        let hf = open_direct_read_file(hot_path)?;
        let hot_idx = parse_rnb_tensor_index(hot_path)?;
        self.hot_file = Some(hf);
        self.hot_tensor_index = hot_idx;
        Ok(())
    }

    /// Read a single MoE expert by **global rank** (0..n_expert). Hot range
    /// (rank < n_hot) routes through the hot `.rnb` pread; cold range routes
    /// through the existing cold file. Both use `O_DIRECT` so the pagecache
    /// is never populated — the "hot vs cold" distinction collapses to a
    /// single pread path, at the cost of losing the mmap hit when the
    /// pagecache happened to already contain the hot bytes.
    ///
    /// Session 68 hot-off experiment.
    pub fn read_expert_unified(&self, name: &str, rank_idx: usize) -> io::Result<Vec<u8>> {
        if let Some(hf) = self.hot_file.as_ref() {
            if let Some(&(base_off, per_expert, n_hot)) = self.hot_tensor_index.get(name) {
                if rank_idx < n_hot {
                    return self.pread_into_vec(hf, base_off, per_expert, rank_idx);
                }
                // Fall through: this rank lives in the cold tail.
                let cold_idx = rank_idx - n_hot;
                return self.read_expert(name, cold_idx);
            }
        }
        // No hot file attached → delegate to the legacy cold_idx-based path.
        // Caller is expected to have subtracted `hot_count` already.
        self.read_expert(name, rank_idx)
    }

    /// Shared pread-to-Vec helper used by both the cold and the attached-hot
    /// streams. Picks a scratch buf from the pool, reads an `O_DIRECT`-aligned
    /// chunk, trims head padding, and hands back a fresh owned `Vec<u8>` of
    /// length `per_expert`.
    fn pread_into_vec(
        &self,
        file: &std::fs::File,
        base_off: u64,
        per_expert: usize,
        idx: usize,
    ) -> io::Result<Vec<u8>> {
        let logical_off = base_off + (idx * per_expert) as u64;
        let aligned_off = logical_off & !4095;
        let head_drop = (logical_off - aligned_off) as usize;
        let aligned_len = (head_drop + per_expert + 4095) & !4095;

        let slot = rayon::current_thread_index().unwrap_or(0) % self.buffers.len();
        let mut guard = self.buffers[slot]
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "cold buffer mutex poisoned"))?;
        let buf = guard.as_aligned_mut(aligned_len);
        let t0 = std::time::Instant::now();
        let n = read_at(file, buf, aligned_off)?;
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        self.stats_n_calls.fetch_add(1, Ordering::Relaxed);
        self.stats_total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.stats_total_bytes
            .fetch_add(aligned_len as u64, Ordering::Relaxed);
        if n < head_drop + per_expert {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "short pread: got {} bytes, need {} (head_drop={}, per_expert={})",
                    n,
                    head_drop + per_expert,
                    head_drop,
                    per_expert
                ),
            ));
        }
        Ok(buf[head_drop..head_drop + per_expert].to_vec())
    }

    pub fn reset_stats(&self) {
        self.stats_n_calls.store(0, Ordering::Relaxed);
        self.stats_total_ns.store(0, Ordering::Relaxed);
        self.stats_total_bytes.store(0, Ordering::Relaxed);
    }

    /// Returns `(n_calls, total_us, total_bytes)`.
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.stats_n_calls.load(Ordering::Relaxed),
            self.stats_total_ns.load(Ordering::Relaxed) / 1_000,
            self.stats_total_bytes.load(Ordering::Relaxed),
        )
    }

    /// Read one cold expert's bytes (rank `cold_idx` within the cold tail) for
    /// the given tensor name. Returns a fresh `Vec<u8>` of length
    /// `per_expert_bytes`.
    pub fn read_expert(&self, name: &str, cold_idx: usize) -> io::Result<Vec<u8>> {
        let &(base_off, per_expert, n_cold) = self.tensor_index.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("cold tensor '{}' not in index", name),
            )
        })?;
        if cold_idx >= n_cold {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("cold_idx {} ≥ n_cold {} for {}", cold_idx, n_cold, name),
            ));
        }
        let logical_off = base_off + (cold_idx * per_expert) as u64;
        let aligned_off = logical_off & !4095;
        let head_drop = (logical_off - aligned_off) as usize;
        let aligned_len = (head_drop + per_expert + 4095) & !4095;

        let slot = rayon::current_thread_index().unwrap_or(0) % self.buffers.len();
        let mut guard = self.buffers[slot]
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "cold buffer mutex poisoned"))?;
        let buf = guard.as_aligned_mut(aligned_len);
        let t0 = std::time::Instant::now();
        let n = read_at(&self.file, buf, aligned_off)?;
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        self.stats_n_calls.fetch_add(1, Ordering::Relaxed);
        self.stats_total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.stats_total_bytes
            .fetch_add(aligned_len as u64, Ordering::Relaxed);
        if n < head_drop + per_expert {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "short pread on {}: got {} bytes, need {} (head_drop={}, per_expert={})",
                    name,
                    n,
                    head_drop + per_expert,
                    head_drop,
                    per_expert
                ),
            ));
        }
        Ok(buf[head_drop..head_drop + per_expert].to_vec())
    }

    pub fn per_expert_bytes(&self, name: &str) -> Option<usize> {
        self.tensor_index.get(name).map(|&(_, pe, _)| pe)
    }

    pub fn n_cold_experts(&self, name: &str) -> Option<usize> {
        self.tensor_index.get(name).map(|&(_, _, n)| n)
    }
}

/// `ColdByteSource` impl that pread's a single cold rank from a `ColdReader`'s
/// O_DIRECT file. Owned `Vec<u8>` per call (cannot borrow into the file).
pub struct ColdReaderColdSource {
    pub cr: Arc<ColdReader>,
    pub name: String,
}

impl crate::moe_residency::ColdByteSource for ColdReaderColdSource {
    fn cold(&self, cold_rank: usize, _per_expert: usize) -> std::borrow::Cow<'_, [u8]> {
        let bytes = self
            .cr
            .read_expert(&self.name, cold_rank)
            .expect("cold pread failed");
        std::borrow::Cow::Owned(bytes)
    }
}

/// `HotByteSource` impl that pread's *every* rank (hot included) through the
/// unified hot+cold pread path. Used in Session 68 axis G "hot-off" mode where
/// the hot mmap is bypassed and the hot `.rnb` is opened with O_DIRECT instead.
pub struct ColdReaderUnifiedHotSource {
    pub cr: Arc<ColdReader>,
    pub name: String,
}

impl crate::moe_residency::HotByteSource for ColdReaderUnifiedHotSource {
    fn hot(&self, rank: usize, _per_expert: usize) -> std::borrow::Cow<'_, [u8]> {
        let bytes = self
            .cr
            .read_expert_unified(&self.name, rank)
            .expect("unified hot pread failed");
        std::borrow::Cow::Owned(bytes)
    }
}

pub fn open_configured_cold_reader(
    hot_rnb_path: &Path,
    max_expert_bytes: usize,
    n_buffers: usize,
) -> Option<Arc<ColdReader>> {
    let cold_path = crate::moe_policy::cold_rnb_path_for(hot_rnb_path);
    if !cold_path.exists() {
        return None;
    }
    match ColdReader::open(&cold_path, max_expert_bytes, n_buffers.max(1)) {
        Ok(mut reader) => {
            eprintln!(
                "[INFO] MoE cold .rnb opened ({}): {} ({} thread buffers, max {} MiB)",
                cold_io_mode_label(),
                cold_path.display(),
                n_buffers.max(1),
                max_expert_bytes / (1024 * 1024)
            );
            if crate::moe_policy::unified_cold_enabled() && hot_rnb_path.exists() {
                match reader.attach_hot_file(hot_rnb_path) {
                    Ok(()) => eprintln!(
                        "[INFO] MoE hot .rnb attached ({}): {}",
                        cold_io_mode_label(),
                        hot_rnb_path.display()
                    ),
                    Err(e) => eprintln!("[WARN] hot .rnb attach failed: {}", e),
                }
            }
            Some(Arc::new(reader))
        }
        Err(e) => {
            eprintln!(
                "[WARN] MoE cold .rnb open failed: {} ({})",
                cold_path.display(),
                e
            );
            None
        }
    }
}

/// Parse the `.rnb` header + tensor table to build `name → (file_offset,
/// per_expert_bytes, n_experts)`. Uses standard buffered IO (not O_DIRECT) —
/// the tensor table is small (a few KB) so pagecache pressure is negligible.
fn parse_rnb_tensor_index(path: &Path) -> io::Result<HashMap<String, (u64, usize, usize)>> {
    let mut f = std::fs::File::open(path)?;
    // Header is 64 bytes. We only need:
    //   num_tensors (u32 @ offset 8)
    //   tensor_table_offset (u64 @ offset 16)
    //   data_offset (u64 @ offset 24)
    let mut header = [0u8; 64];
    f.read_exact(&mut header)?;
    let magic = &header[0..4];
    if magic != b"RNBD" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad RNB magic"));
    }
    let num_tensors = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let tensor_table_offset = u64::from_le_bytes([
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ]);
    let data_offset = u64::from_le_bytes([
        header[24], header[25], header[26], header[27], header[28], header[29], header[30],
        header[31],
    ]);

    f.seek(SeekFrom::Start(tensor_table_offset))?;
    let mut entries_buf = vec![0u8; num_tensors * 96];
    f.read_exact(&mut entries_buf)?;

    let mut idx = HashMap::with_capacity(num_tensors);
    for i in 0..num_tensors {
        let off = i * 96;
        let entry = &entries_buf[off..off + 96];
        let name_end = entry[..64].iter().position(|&b| b == 0).unwrap_or(64);
        let name = String::from_utf8_lossy(&entry[..name_end]).into_owned();
        let rows = u32::from_le_bytes([entry[68], entry[69], entry[70], entry[71]]) as usize;
        // entry[80..88] = data_offset (u64, relative to data section start)
        let rel_data_off = u64::from_le_bytes([
            entry[80], entry[81], entry[82], entry[83], entry[84], entry[85], entry[86], entry[87],
        ]);
        let data_len = u64::from_le_bytes([
            entry[88], entry[89], entry[90], entry[91], entry[92], entry[93], entry[94], entry[95],
        ]) as usize;
        let abs_off = data_offset + rel_data_off;
        let per_expert = if rows > 0 { data_len / rows } else { 0 };
        idx.insert(name, (abs_off, per_expert, rows));
    }
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buf_returns_4kb_aligned_pointer() {
        let mut buf = AlignedBuf::with_capacity(8 * 1024);
        let s = buf.as_aligned_mut(4096);
        assert_eq!(s.as_ptr() as usize & 4095, 0);
        assert_eq!(s.len(), 4096);
    }

    #[test]
    fn aligned_buf_capacity_rounds_up() {
        let buf = AlignedBuf::with_capacity(100);
        assert!(buf.capacity >= 4096);
    }

    #[test]
    fn direct_file_reader_rejects_unaligned_offsets_before_open() {
        let mut reader = DirectFileReaderCache::default();
        let mut buffer = AlignedBuf::with_capacity(4096);
        let error = reader
            .read_aligned(
                Path::new("/path/need/not/exist"),
                1,
                buffer.as_aligned_mut(4096),
                4096,
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
