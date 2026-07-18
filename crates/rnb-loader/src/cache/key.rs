//! Cache key calculation for sidecar `.rnb` artifacts.
//!
//! Key inputs (all bound together by SHA-256, truncated to 16 hex chars):
//! - Partial file hash: first 4 MB + last 4 MB of the GGUF (full hash is too
//!   expensive on multi-GB models; head+tail catches most accidental collisions)
//! - File mtime (nanoseconds)
//! - File size (bytes)
//! - Sidecar format version
//! - Convert options hash (Phase 1 fixed)

use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Sidecar format version. Phase 1 introduces v3 (RNBC magic, packed Q4_K
/// row-pair + Q6_K + MoE Q4_K hot/cold). v2 (RNBD) is read-only legacy until
/// Task 18 removes it; v2 cache files are never matched by v3 lookups because
/// the filename embeds the format version into the key.
pub const FORMAT_VERSION_V3: u32 = 3;

/// Phase 1 convert options are a single fixed bundle (option B in spec §3.3).
/// When Phase 2 introduces toggles (Q5_K, output Q8 tile8, …), the options
/// struct will be hashed into this value to invalidate cache on policy change.
pub const OPTIONS_HASH_PHASE1: u64 = 0x1;

const PARTIAL_BLOCK: u64 = 4 * 1024 * 1024;
const KEY_HEX_CHARS: usize = 16;

/// Compute a 16-hex-char cache key for the given GGUF file.
pub fn compute_cache_key(
    gguf_path: &Path,
    format_version: u32,
    options_hash: u64,
) -> Result<String, String> {
    let metadata = std::fs::metadata(gguf_path).map_err(|e| format!("metadata: {e}"))?;
    let size = metadata.len();
    let mtime_ns = metadata
        .modified()
        .map_err(|e| format!("mtime: {e}"))?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("mtime epoch: {e}"))?
        .as_nanos() as u64;
    let partial = partial_file_hash(gguf_path, size)?;

    let mut hasher = Sha256::new();
    hasher.update(partial);
    hasher.update(mtime_ns.to_le_bytes());
    hasher.update(size.to_le_bytes());
    hasher.update(format_version.to_le_bytes());
    hasher.update(options_hash.to_le_bytes());
    let digest = hasher.finalize();
    Ok(hex::encode(&digest[..KEY_HEX_CHARS / 2]))
}

/// Hash the head and (if file is large enough) the tail of a file.
fn partial_file_hash(path: &Path, size: u64) -> Result<[u8; 32], String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut hasher = Sha256::new();
    let head_len = size.min(PARTIAL_BLOCK) as usize;
    if head_len > 0 {
        let mut buf = vec![0u8; head_len];
        f.read_exact(&mut buf)
            .map_err(|e| format!("read head: {e}"))?;
        hasher.update(&buf);
    }
    if size > PARTIAL_BLOCK * 2 {
        let mut buf = vec![0u8; PARTIAL_BLOCK as usize];
        f.seek(SeekFrom::Start(size - PARTIAL_BLOCK))
            .map_err(|e| format!("seek tail: {e}"))?;
        f.read_exact(&mut buf)
            .map_err(|e| format!("read tail: {e}"))?;
        hasher.update(&buf);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(hasher.finalize().as_slice());
    Ok(out)
}

/// Filename layout: `{gguf_basename}_{key16}.rnb`.
pub fn cache_filename(gguf_basename: &str, key: &str) -> String {
    debug_assert_eq!(key.len(), KEY_HEX_CHARS, "cache key must be 16 hex chars");
    format!("{gguf_basename}_{key}.rnb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_file_yields_same_key() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"hello world repeated content").unwrap();
        let k1 = compute_cache_key(f.path(), FORMAT_VERSION_V3, OPTIONS_HASH_PHASE1).unwrap();
        let k2 = compute_cache_key(f.path(), FORMAT_VERSION_V3, OPTIONS_HASH_PHASE1).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn different_format_version_yields_different_key() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"x").unwrap();
        let k1 = compute_cache_key(f.path(), 2, OPTIONS_HASH_PHASE1).unwrap();
        let k2 = compute_cache_key(f.path(), 3, OPTIONS_HASH_PHASE1).unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn different_options_hash_yields_different_key() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"x").unwrap();
        let k1 = compute_cache_key(f.path(), FORMAT_VERSION_V3, 0x1).unwrap();
        let k2 = compute_cache_key(f.path(), FORMAT_VERSION_V3, 0x2).unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_filename_format() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"x").unwrap();
        let key = compute_cache_key(f.path(), FORMAT_VERSION_V3, OPTIONS_HASH_PHASE1).unwrap();
        let filename = cache_filename("gemma-e2b.gguf", &key);
        assert!(filename.starts_with("gemma-e2b.gguf_"));
        assert!(filename.ends_with(".rnb"));
        assert_eq!(key.len(), 16);
        // total = "gemma-e2b.gguf_" (15) + 16 hex chars + ".rnb" (4)
        assert_eq!(filename.len(), 15 + 16 + 4);
    }

    #[test]
    fn small_file_hashes_correctly() {
        // File smaller than 4MB should still hash (no tail seek panic)
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"tiny").unwrap();
        let key = compute_cache_key(f.path(), FORMAT_VERSION_V3, OPTIONS_HASH_PHASE1).unwrap();
        assert_eq!(key.len(), 16);
    }
}
