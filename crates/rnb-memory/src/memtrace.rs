//! Inline memory-behavior probe. No-op unless `RNB_MEMTRACE_FILE` env is set.
//!
//! Outputs TSV rows with two shapes:
//!   "step\t<ts_ns>\t<step_idx>\t<phase:start|end>\n"
//!   "expert\t<ts_ns>\t<step_idx>\t<layer>\t<expert>\t<rank>\t<resident_pages>\t<total_pages>\t<region:gu|dn>\t<expert_original_id>\t<tier:hot|cold_mmap|cold_pread>\n"
//!
//! The expert row is **11 columns**. `expert_original_id` remains an explicit
//! identity column for stable analysis, and `tier` records the backing tier.
//! Direct GGUF execution currently reports the runtime expert id unchanged and
//! uses the `hot` tier.
//!
//! Legacy 9-column rows (Session 66/67 data) remain parseable by accepting
//! both widths in `scripts/memtrace_analyze.py`.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use once_cell::sync::Lazy;

/// Program-wide monotonic epoch captured at process start.
static EPOCH: Lazy<Instant> = Lazy::new(Instant::now);

/// Lock-free gate for hot callers (`MoeLayerView::forward_with_logits`).
/// Set to mirror `WRITER.is_some()` at init and after `reset_for_tests`.
/// Using a plain atomic avoids the per-call `Mutex::lock()` that was
/// serializing the par_iter fan-out across 240 expert calls per decode step.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// TSV writer. `None` if `RNB_MEMTRACE_FILE` is unset at module init time.
static WRITER: Lazy<Mutex<Option<BufWriter<std::fs::File>>>> = Lazy::new(|| {
    let w = open_writer();
    ACTIVE.store(w.is_some(), Ordering::Relaxed);
    Mutex::new(w)
});

fn open_writer() -> Option<BufWriter<std::fs::File>> {
    let path = std::env::var("RNB_MEMTRACE_FILE").ok()?;
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()?;
    Some(BufWriter::new(f))
}

#[cfg(test)]
pub fn reset_for_tests() {
    // Force the lazy to be initialised, then swap in a fresh writer.
    let _ = *EPOCH;
    let mut guard = WRITER.lock().unwrap();
    let w = open_writer();
    ACTIVE.store(w.is_some(), Ordering::Relaxed);
    *guard = w;
}

#[cfg(test)]
pub fn flush_for_tests() {
    if let Some(w) = WRITER.lock().unwrap().as_mut() {
        let _ = w.flush();
    }
}

#[inline]
fn now_ns() -> u64 {
    Instant::now().saturating_duration_since(*EPOCH).as_nanos() as u64
}

/// Returns true when a TSV writer is configured. Lock-free: mirrors
/// `WRITER.is_some()` via a plain `AtomicBool` so parallel gate checks
/// (e.g. 240 expert calls/decode step) don't serialize on the writer mutex.
#[inline]
pub fn is_active() -> bool {
    // Touch WRITER once to guarantee the Lazy init has fired and set ACTIVE.
    // After the first call, ACTIVE alone is authoritative.
    Lazy::force(&WRITER);
    ACTIVE.load(Ordering::Relaxed)
}

pub fn record_step_start(step_idx: usize) {
    let mut guard = match WRITER.try_lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(w) = guard.as_mut() {
        let _ = writeln!(w, "step\t{}\t{}\tstart", now_ns(), step_idx);
    }
}

pub fn record_step_end(step_idx: usize) {
    let mut guard = match WRITER.try_lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(w) = guard.as_mut() {
        let _ = writeln!(w, "step\t{}\t{}\tend", now_ns(), step_idx);
        let _ = w.flush();
    }
}

pub fn record_expert_call(
    step_idx: usize,
    layer: usize,
    expert: usize,
    expert_original_id: u32,
    rank: usize,
    ptr: *const u8,
    len: usize,
    region: &str,
    tier: &str,
) {
    // Call mincore BEFORE acquiring the writer lock — the syscall is the
    // expensive part, and we want to minimize time spent holding the lock.
    let (resident, total) = mincore_resident(ptr, len);

    let mut guard = match WRITER.try_lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(w) = guard.as_mut() {
        let _ = writeln!(
            w,
            "expert\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            now_ns(),
            step_idx,
            layer,
            expert,
            rank,
            resident,
            total,
            region,
            expert_original_id,
            tier,
        );
    }
}

/// Call `mincore(2)` on `[ptr, ptr+len)` and count pages with the resident bit set.
/// Returns `(resident_pages, total_pages)`. On syscall failure returns `(0, total_pages)`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn mincore_resident(ptr: *const u8, len: usize) -> (usize, usize) {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    if page_size == 0 || len == 0 {
        return (0, 0);
    }
    let total = (len + page_size - 1) / page_size;
    let mut buf = vec![0u8; total];
    let rc = unsafe { libc::mincore(ptr as *mut _, len, buf.as_mut_ptr()) };
    if rc != 0 {
        return (0, total);
    }
    let resident = buf.iter().filter(|b| (*b & 1) == 1).count();
    (resident, total)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn mincore_resident(_ptr: *const u8, len: usize) -> (usize, usize) {
    let page_size = 4096;
    let total = (len + page_size - 1) / page_size;
    (0, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::NamedTempFile;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn no_op_when_env_unset() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("RNB_MEMTRACE_FILE");
        reset_for_tests();
        // Should not panic and should not create any file
        record_step_start(0);
        record_step_end(0);
    }

    #[test]
    fn writes_step_rows_when_env_set() {
        let _guard = env_lock().lock().unwrap();
        let tf = NamedTempFile::new().unwrap();
        std::env::set_var("RNB_MEMTRACE_FILE", tf.path());
        reset_for_tests();

        record_step_start(3);
        record_step_end(3);

        flush_for_tests();
        let content = fs::read_to_string(tf.path()).unwrap();
        assert!(
            content.contains("step\t"),
            "expected 'step\\t' row: {}",
            content
        );
        assert!(
            content.contains("\t3\tstart\n"),
            "expected start row: {}",
            content
        );
        assert!(
            content.contains("\t3\tend\n"),
            "expected end row: {}",
            content
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn mincore_resident_counts_anonymous_pages() {
        // Anonymous private mapping, touched -> all resident
        let page_size = 4096;
        let total_bytes = page_size * 10;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0x11, total_bytes) };

        let (resident, total) = mincore_resident(ptr as *const u8, total_bytes);
        assert_eq!(total, 10);
        assert_eq!(resident, 10);

        unsafe { libc::munmap(ptr, total_bytes) };
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn expert_call_row_has_correct_shape() {
        let _guard = env_lock().lock().unwrap();
        let tf = NamedTempFile::new().unwrap();
        std::env::set_var("RNB_MEMTRACE_FILE", tf.path());
        reset_for_tests();

        // Create a small mmapped region so we have a real pointer
        let page_size = 4096;
        let total = page_size * 4;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0xAA, total) };

        record_expert_call(5, 12, 47, 93, 2, ptr as *const u8, total, "gu", "hot");

        flush_for_tests();
        let content = fs::read_to_string(tf.path()).unwrap();

        unsafe { libc::munmap(ptr, total) };

        // Expect one "expert" row with 11 tab-separated fields
        let row = content
            .lines()
            .find(|l| l.starts_with("expert\t"))
            .expect("no expert row");
        let fields: Vec<&str> = row.split('\t').collect();
        assert_eq!(fields.len(), 11, "expected 11 cols, got: {:?}", fields);
        assert_eq!(fields[0], "expert");
        // fields[1] = ts_ns (non-zero monotonic)
        assert!(
            fields[1].parse::<u64>().is_ok(),
            "ts not numeric: {}",
            fields[1]
        );
        assert_eq!(fields[2], "5"); // step_idx
        assert_eq!(fields[3], "12"); // layer
        assert_eq!(fields[4], "47"); // expert (rank-space id)
        assert_eq!(fields[5], "2"); // rank (top-k order)
        assert_eq!(fields[7], "4"); // total_pages = 4
        assert_eq!(fields[6], "4"); // all touched -> all resident
        assert_eq!(fields[8], "gu"); // region
        assert_eq!(fields[9], "93"); // expert_original_id (post inverse-permutation)
        assert_eq!(fields[10], "hot"); // tier
    }
}
