//! LRU eviction policy for the runNburn cache directory.
//!
//! Keeps total disk usage of cached `.rnb` sidecars under a configurable cap.
//! When the cap is exceeded, the oldest (lowest mtime) `.rnb` files are
//! deleted until the total fits. Files that are not `.rnb` (e.g. accidental
//! user files in the cache directory) are ignored — both for accounting and
//! for eviction.
//!
//! Per spec, this lives in `rnb-memory`: cache eviction policy is a memory
//! concern. The `rnb-loader::cache` module owns lookup and key calculation;
//! this module owns the budget and eviction loop. They cooperate via the
//! filesystem layout (the cache directory).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Default mobile cache cap. PCs use much larger caps or none — caller chooses.
pub const MOBILE_DEFAULT_CAP_BYTES: u64 = 5 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
struct CacheEntry {
    path: PathBuf,
    size: u64,
    mtime: SystemTime,
}

/// Walk a cache directory and apply an LRU cap, deleting oldest `.rnb`
/// entries until total size is `<= cap_bytes`. Returns the number of bytes
/// freed and the number of files deleted.
pub fn apply_lru_cap(cache_dir: &Path, cap_bytes: u64) -> Result<EvictionReport, String> {
    let entries = collect_rnb_entries(cache_dir)?;
    let total: u64 = entries.iter().map(|e| e.size).sum();
    if total <= cap_bytes {
        return Ok(EvictionReport {
            total_before: total,
            total_after: total,
            files_deleted: 0,
            bytes_freed: 0,
        });
    }

    let mut sorted = entries;
    sorted.sort_by_key(|e| e.mtime);

    let mut current = total;
    let mut bytes_freed: u64 = 0;
    let mut files_deleted: usize = 0;
    for e in sorted {
        if current <= cap_bytes {
            break;
        }
        match std::fs::remove_file(&e.path) {
            Ok(()) => {
                current = current.saturating_sub(e.size);
                bytes_freed += e.size;
                files_deleted += 1;
            }
            Err(err) => {
                return Err(format!("evict {}: {err}", e.path.display()));
            }
        }
    }
    Ok(EvictionReport {
        total_before: total,
        total_after: current,
        files_deleted,
        bytes_freed,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionReport {
    pub total_before: u64,
    pub total_after: u64,
    pub files_deleted: usize,
    pub bytes_freed: u64,
}

fn collect_rnb_entries(cache_dir: &Path) -> Result<Vec<CacheEntry>, String> {
    let read_dir = match std::fs::read_dir(cache_dir) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("read {}: {e}", cache_dir.display())),
    };
    let mut out: Vec<CacheEntry> = Vec::new();
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rnb") {
            continue;
        }
        let meta = entry
            .metadata()
            .map_err(|e| format!("metadata {}: {e}", path.display()))?;
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        out.push(CacheEntry {
            path,
            size: meta.len(),
            mtime,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    fn write_rnb(dir: &Path, name: &str, bytes: usize) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, vec![0u8; bytes]).unwrap();
        path
    }

    #[test]
    fn under_cap_no_eviction() {
        let dir = tempfile::tempdir().unwrap();
        write_rnb(dir.path(), "a.rnb", 100);
        write_rnb(dir.path(), "b.rnb", 200);
        let report = apply_lru_cap(dir.path(), 1000).unwrap();
        assert_eq!(report.files_deleted, 0);
        assert_eq!(report.bytes_freed, 0);
    }

    #[test]
    fn over_cap_evicts_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_rnb(dir.path(), "a.rnb", 600);
        sleep(Duration::from_millis(10));
        let b = write_rnb(dir.path(), "b.rnb", 600);
        // total 1200, cap 1000 → must evict 1 file. Oldest is `a.rnb`.
        let report = apply_lru_cap(dir.path(), 1000).unwrap();
        assert_eq!(report.files_deleted, 1);
        assert_eq!(report.bytes_freed, 600);
        assert!(!a.exists(), "oldest must be deleted");
        assert!(b.exists(), "newest must survive");
    }

    #[test]
    fn ignores_non_rnb_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"unrelated").unwrap();
        write_rnb(dir.path(), "a.rnb", 100);
        let report = apply_lru_cap(dir.path(), 50).unwrap();
        assert_eq!(report.files_deleted, 1);
        assert!(
            dir.path().join("readme.txt").exists(),
            "non-.rnb must not be touched"
        );
    }

    #[test]
    fn missing_dir_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("does/not/exist");
        let report = apply_lru_cap(&nested, 1000).unwrap();
        assert_eq!(report.files_deleted, 0);
    }
}
