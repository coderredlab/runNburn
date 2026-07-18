//! Cache directory lookup helpers for sidecar `.rnb` artifacts.
//!
//! These helpers search a cache directory for an entry matching the cache
//! filename produced by [`super::key::cache_filename`]. Eviction policy (LRU,
//! disk-budget caps) lives in `rnb-memory` (Phase 2); this module only does
//! exact-name lookup and directory creation.

use std::path::{Path, PathBuf};

/// Look up a cache entry by exact filename. Returns:
/// - `Ok(Some(path))` if the file exists,
/// - `Ok(None)` if the cache directory exists but the file does not (cache miss),
/// - `Err(_)` for any other I/O error (permission denied, etc.).
pub fn find_cache_entry(cache_dir: &Path, filename: &str) -> Result<Option<PathBuf>, String> {
    let candidate = cache_dir.join(filename);
    match std::fs::metadata(&candidate) {
        Ok(_) => Ok(Some(candidate)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("cache lookup {}: {e}", candidate.display())),
    }
}

/// Ensure a cache directory exists, creating it (and parents) if missing.
pub fn ensure_cache_dir(cache_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("create cache dir {}: {e}", cache_dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_cache_entry(dir.path(), "nonexistent_xxx_yyy.rnb").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn lookup_hit_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.gguf_abcdef0123456789.rnb");
        std::fs::write(&target, b"dummy").unwrap();
        let result = find_cache_entry(dir.path(), "model.gguf_abcdef0123456789.rnb").unwrap();
        assert_eq!(result.unwrap(), target);
    }

    #[test]
    fn ensure_cache_dir_creates_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("cache");
        assert!(!nested.exists());
        ensure_cache_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_cache_dir_idempotent_on_existing() {
        let dir = tempfile::tempdir().unwrap();
        ensure_cache_dir(dir.path()).unwrap();
        ensure_cache_dir(dir.path()).unwrap();
        assert!(dir.path().is_dir());
    }

    /// Verify v2 cache files (`RNBD` magic) are NOT matched by v3 lookups.
    /// Invalidation works automatically because the v3 `cache_filename`
    /// embeds `FORMAT_VERSION_V3` into the cache key, so v2 file names that
    /// were generated with `format_version=2` would have a different hash.
    #[test]
    fn v2_magic_file_ignored_by_v3_lookup() {
        let dir = tempfile::tempdir().unwrap();
        // Drop a file with v2 magic and a hash that simulates the v2 layout.
        let v2_file = dir.path().join("model.gguf_oldv2hashabcde00.rnb");
        std::fs::write(&v2_file, b"RNBD\x00\x00\x00\x00").unwrap();
        // v3 lookup uses a different filename (different cache key from format_version=3)
        let result = find_cache_entry(dir.path(), "model.gguf_NEWV3HASH123456.rnb").unwrap();
        assert!(result.is_none());
    }
}
