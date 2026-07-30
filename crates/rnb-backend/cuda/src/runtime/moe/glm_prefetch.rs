//! Cross-layer expert-region read-ahead for the direct-file stream.
//!
//! The direct-file prefill stream reads each layer's selected-expert bytes
//! with O_DIRECT on the submission thread, which serializes NVMe read time
//! into every layer wall. When the engine registers the per-layer expert
//! region sequence at load time, this module prefetches the *next* layer's
//! whole gate/up/down regions on a background thread while the current
//! layer computes. The stream then serves its staging copies from the
//! prefetched buffers instead of the disk.
//!
//! The stream entry gates (route-slot reuse for Hy3, constrained direct-file
//! prefill for GLM) only engage on long prompts where the selected-expert
//! union covers essentially the whole layer, so whole-region read-ahead adds
//! no meaningful read amplification over the plan-driven reads it replaces.
//! Prefetched bytes are identical file bytes; staging layout, kernel inputs,
//! and outputs are unchanged.

use rnb_memory::moe_cold_io::{AlignedBuf, DirectFileReaderCache};
use std::path::PathBuf;
use std::sync::Mutex;
use std::thread::JoinHandle;

const REGION_COUNT: usize = 3;
const DIRECT_FILE_ALIGNMENT: u64 = 4096;
/// Chunk size for background region reads. Large enough to reach device
/// sequential bandwidth, small enough to keep short-read recovery bounded.
const PREFETCH_READ_CHUNK: usize = 64 << 20;

#[derive(Clone, PartialEq, Eq, Debug)]
pub(in crate::runtime) struct StreamRegionKey {
    pub(in crate::runtime) path: PathBuf,
    pub(in crate::runtime) file_offset: u64,
    pub(in crate::runtime) len: usize,
}

static REGION_SEQUENCE: Mutex<Option<Vec<[StreamRegionKey; REGION_COUNT]>>> = Mutex::new(None);

/// Registers the per-layer `[gate, up, down]` expert region sequence in
/// forward layer order. Called once by the engine after weight loading.
pub fn register_stream_region_sequence(sequence: Vec<[(PathBuf, u64, usize); REGION_COUNT]>) {
    let sequence = sequence
        .into_iter()
        .map(|regions| {
            regions.map(|(path, file_offset, len)| StreamRegionKey {
                path,
                file_offset,
                len,
            })
        })
        .collect::<Vec<_>>();
    let mut guard = REGION_SEQUENCE
        .lock()
        .expect("GLM stream region sequence lock poisoned");
    *guard = if sequence.is_empty() {
        None
    } else {
        Some(sequence)
    };
}

fn next_regions_after(
    current: &[StreamRegionKey; REGION_COUNT],
) -> Option<[StreamRegionKey; REGION_COUNT]> {
    let guard = REGION_SEQUENCE
        .lock()
        .expect("GLM stream region sequence lock poisoned");
    let sequence = guard.as_ref()?;
    let index = sequence.iter().position(|entry| entry == current)?;
    sequence.get(index + 1).cloned()
}

pub(in crate::runtime) struct PrefetchedRegions {
    keys: [StreamRegionKey; REGION_COUNT],
    bufs: [AlignedBuf; REGION_COUNT],
    aligned_starts: [u64; REGION_COUNT],
    valid_lens: [usize; REGION_COUNT],
}

impl PrefetchedRegions {
    pub(in crate::runtime) fn matches(&self, keys: &[StreamRegionKey; REGION_COUNT]) -> bool {
        &self.keys == keys
    }

    /// Copies one staging run from the prefetched region bytes. Returns
    /// `false` when the run is not fully covered, in which case the caller
    /// falls back to the disk read for the whole batch.
    pub(in crate::runtime) fn copy_run(
        &mut self,
        region_index: usize,
        file_offset: u64,
        destination: &mut [u8],
        required_len: usize,
    ) -> bool {
        if region_index >= REGION_COUNT || file_offset < self.aligned_starts[region_index] {
            return false;
        }
        let Ok(relative) = usize::try_from(file_offset - self.aligned_starts[region_index]) else {
            return false;
        };
        let valid_len = self.valid_lens[region_index];
        let Some(available) = valid_len.checked_sub(relative) else {
            return false;
        };
        if available < required_len {
            return false;
        }
        let copy_len = destination.len().min(available);
        let source = self.bufs[region_index].as_aligned_mut(valid_len);
        destination[..copy_len].copy_from_slice(&source[relative..relative + copy_len]);
        true
    }
}

fn aligned_read_window(key: &StreamRegionKey) -> Option<(u64, usize, usize)> {
    let aligned_start = key.file_offset & !(DIRECT_FILE_ALIGNMENT - 1);
    let head = usize::try_from(key.file_offset - aligned_start).ok()?;
    let required = head.checked_add(key.len)?;
    let buffer_len = required.checked_add(DIRECT_FILE_ALIGNMENT as usize - 1)?
        / DIRECT_FILE_ALIGNMENT as usize
        * DIRECT_FILE_ALIGNMENT as usize;
    Some((aligned_start, required, buffer_len))
}

fn read_regions(
    keys: [StreamRegionKey; REGION_COUNT],
    mut bufs: [AlignedBuf; REGION_COUNT],
) -> Result<PrefetchedRegions, String> {
    let mut reader = DirectFileReaderCache::default();
    let mut aligned_starts = [0u64; REGION_COUNT];
    let mut valid_lens = [0usize; REGION_COUNT];
    for region_index in 0..REGION_COUNT {
        let key = &keys[region_index];
        let (aligned_start, required, buffer_len) = aligned_read_window(key)
            .ok_or_else(|| "GLM stream prefetch region window overflow".to_string())?;
        aligned_starts[region_index] = aligned_start;
        valid_lens[region_index] = buffer_len;
        let buf = bufs[region_index].as_aligned_mut(buffer_len);
        let mut done = 0usize;
        while done < buffer_len {
            let chunk_len = PREFETCH_READ_CHUNK.min(buffer_len - done);
            let chunk_required = required.saturating_sub(done).min(chunk_len);
            reader
                .read_aligned(
                    &key.path,
                    aligned_start + done as u64,
                    &mut buf[done..done + chunk_len],
                    chunk_required,
                )
                .map_err(|error| {
                    format!(
                        "GLM stream prefetch read failed for {} at {}: {error}",
                        key.path.display(),
                        aligned_start + done as u64
                    )
                })?;
            done += chunk_len;
        }
    }
    Ok(PrefetchedRegions {
        keys,
        bufs,
        aligned_starts,
        valid_lens,
    })
}

#[derive(Default)]
pub(in crate::runtime) struct GlmStreamPrefetch {
    pending: Option<JoinHandle<Result<PrefetchedRegions, String>>>,
    spare_bufs: Vec<[AlignedBuf; REGION_COUNT]>,
    failure_logged: bool,
}

impl GlmStreamPrefetch {
    /// Joins the in-flight prefetch when it targets `current`; otherwise the
    /// stale prefetch is drained and recycled.
    pub(in crate::runtime) fn take_matching(
        &mut self,
        current: &[StreamRegionKey; REGION_COUNT],
    ) -> Option<PrefetchedRegions> {
        let handle = self.pending.take()?;
        match handle.join() {
            Ok(Ok(prefetched)) => {
                if prefetched.matches(current) {
                    return Some(prefetched);
                }
                self.recycle(prefetched);
                None
            }
            Ok(Err(error)) => {
                if !self.failure_logged {
                    self.failure_logged = true;
                    eprintln!("[WARN] GLM stream prefetch disabled after read failure: {error}");
                }
                None
            }
            Err(_) => {
                if !self.failure_logged {
                    self.failure_logged = true;
                    eprintln!(
                        "[WARN] GLM stream prefetch thread panicked; falling back to direct reads"
                    );
                }
                None
            }
        }
    }

    /// Spawns a background read of the layer following `current` in the
    /// registered sequence. No-op when a prefetch is already pending, the
    /// sequence is unknown, or a prior prefetch failed.
    pub(in crate::runtime) fn spawn_after(&mut self, current: &[StreamRegionKey; REGION_COUNT]) {
        if self.pending.is_some() || self.failure_logged {
            return;
        }
        let Some(next) = next_regions_after(current) else {
            return;
        };
        let mut required = [0usize; REGION_COUNT];
        for (slot, key) in next.iter().enumerate() {
            let Some((_, _, buffer_len)) = aligned_read_window(key) else {
                return;
            };
            required[slot] = buffer_len;
        }
        let bufs = match self.spare_bufs.pop() {
            Some(bufs)
                if bufs
                    .iter()
                    .zip(required.iter())
                    .all(|(buf, &len)| buf.capacity() >= len) =>
            {
                bufs
            }
            _ => [
                AlignedBuf::with_capacity(required[0]),
                AlignedBuf::with_capacity(required[1]),
                AlignedBuf::with_capacity(required[2]),
            ],
        };
        let handle = std::thread::Builder::new()
            .name("rnb-glm-prefetch".to_string())
            .spawn(move || read_regions(next, bufs));
        match handle {
            Ok(handle) => self.pending = Some(handle),
            Err(error) => {
                if !self.failure_logged {
                    self.failure_logged = true;
                    eprintln!("[WARN] GLM stream prefetch thread spawn failed: {error}");
                }
            }
        }
    }

    pub(in crate::runtime) fn recycle(&mut self, prefetched: PrefetchedRegions) {
        if self.spare_bufs.len() < 2 {
            self.spare_bufs.push(prefetched.bufs);
        }
    }
}

pub(in crate::runtime) fn stream_region_keys(
    regions: &[rnb_core::tensor::FileBackedRegion; REGION_COUNT],
) -> [StreamRegionKey; REGION_COUNT] {
    [0, 1, 2].map(|index| StreamRegionKey {
        path: regions[index].path().to_path_buf(),
        file_offset: regions[index].file_offset(),
        len: regions[index].len(),
    })
}
