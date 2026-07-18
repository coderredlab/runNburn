//! MoE hot-expert pool allocator.
//!
//! This module owns the residency mechanism for bytes that the MoE policy
//! has classified as hot. It does not decide expert popularity itself; it
//! consumes the effective hot count from `moe_policy` and copies only the
//! required hot prefix into the selected backing store.
//!
//! Session 65 axis N.
//!
//! Two backing stores:
//!   - `Heap(Vec<u8>)`: anonymous private mapping. Session 64 axis E path.
//!     On Android this is a zRAM target — Samsung OneUI / AOSP aggressively
//!     compress anonymous memory into swap, causing ~7× regression when the
//!     hot tier is 4 GB (Session 64 bench).
//!   - `Memfd { fd, ptr, len }`: `memfd_create(2)` + `ftruncate` + `mmap
//!     (MAP_SHARED)`. The memfd is tmpfs-backed, and Android's zRAM policy
//!     only targets anonymous mappings — tmpfs pages stay out of zRAM.
//!     Session 65 probe APK confirmed: 4 GB memfd touch gave `VmSwap delta
//!     -24 kB` and `VmRSS +4.1 GB`, i.e. fully RAM-resident. Reread ran at
//!     159 GB/s (L2/L3 cache speed), so hot-expert access should not pay
//!     any decompression tax.
//!
//! Both variants expose `&[u8]` through `as_slice`, so the rest of the
//! engine does not care which allocator produced the bytes.

use std::io;

/// Selects which backing store `HotPool::alloc_and_fill` should use.
#[derive(Clone, Copy, Debug)]
pub enum AllocatorKind {
    /// `Vec<u8>` on the anonymous private heap. zRAM compresses this on
    /// Android. Kept for A/B comparison against axis N.
    Heap,
    /// `memfd_create` + `mmap(MAP_SHARED)`. zRAM-exempt on Android.
    Memfd,
}

pub enum ConfiguredHotPoolPlan {
    Disabled,
    MissingHotCount {
        allocator_kind: AllocatorKind,
    },
    Enabled {
        allocator_kind: AllocatorKind,
        hot_count: usize,
    },
}

pub enum HotPool {
    Heap(Vec<u8>),
    Memfd { fd: i32, ptr: *mut u8, len: usize },
}

// `HotPool::Memfd` holds a raw pointer into a process-wide mmap, and
// callers may wrap the pool in `Arc` and share read-only slices across
// rayon workers during MoE forward. Writes only happen during
// `alloc_and_fill` before the pool is handed out, so shared reads are
// safe.
unsafe impl Send for HotPool {}
unsafe impl Sync for HotPool {}

impl HotPool {
    /// Allocate a pool of `src.len()` bytes with the chosen backing store
    /// and copy `src` into it.
    pub fn alloc_and_fill(kind: AllocatorKind, src: &[u8]) -> io::Result<Self> {
        match kind {
            AllocatorKind::Heap => Ok(HotPool::Heap(src.to_vec())),
            AllocatorKind::Memfd => {
                let mut pool = Self::new_memfd(src.len())?;
                pool.write_bytes(src);
                Ok(pool)
            }
        }
    }

    /// Allocate a `memfd_create + ftruncate + mmap(MAP_SHARED)` pool of
    /// exactly `size` bytes. Pool content is uninitialized (zero on first
    /// access — tmpfs pages are zero-filled).
    pub fn new_memfd(size: usize) -> io::Result<Self> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use std::ffi::CString;
            let name = CString::new("rnb-moe-hot").unwrap();
            let fd = unsafe { libc::memfd_create(name.as_ptr(), 0) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
                let err = io::Error::last_os_error();
                unsafe { libc::close(fd) };
                return Err(err);
            }
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                let err = io::Error::last_os_error();
                unsafe { libc::close(fd) };
                return Err(err);
            }
            Ok(HotPool::Memfd {
                fd,
                ptr: ptr as *mut u8,
                len: size,
            })
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = size;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "memfd_create is Linux-only",
            ))
        }
    }

    fn write_bytes(&mut self, src: &[u8]) {
        match self {
            HotPool::Heap(v) => {
                v.clear();
                v.extend_from_slice(src);
            }
            HotPool::Memfd { ptr, len, .. } => {
                assert!(
                    src.len() <= *len,
                    "memfd pool too small: src {} > pool {}",
                    src.len(),
                    *len
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), *ptr, src.len());
                }
            }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            HotPool::Heap(v) => v.as_slice(),
            HotPool::Memfd { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HotPool::Heap(v) => v.len(),
            HotPool::Memfd { len, .. } => *len,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            HotPool::Heap(_) => "heap",
            HotPool::Memfd { .. } => "memfd",
        }
    }
}

/// `HotByteSource` impl backed by a `HotPool` (heap or memfd). Zero-copy:
/// `Cow::Borrowed` over the pool's contiguous bytes.
pub struct HotPoolSource {
    pub pool: std::sync::Arc<HotPool>,
}

impl crate::moe_residency::HotByteSource for HotPoolSource {
    fn hot(&self, rank: usize, per_expert: usize) -> std::borrow::Cow<'_, [u8]> {
        let s = self.pool.as_slice();
        std::borrow::Cow::Borrowed(&s[rank * per_expert..(rank + 1) * per_expert])
    }
}

impl Drop for HotPool {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let HotPool::Memfd { fd, ptr, len } = *self {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, len);
                libc::close(fd);
            }
        }
    }
}

/// Parse the user-facing `RNB_MOE_HOT_POOL` env. `"heap"` = legacy Vec,
/// `"memfd"` = axis N memfd pool. Any other value (or unset) → `None`.
///
/// Also accepts the Session 64 `RNB_MOE_HEAP=1` as a deprecated alias for
/// `heap`, so old benchmark scripts keep working.
pub fn parse_env_allocator_kind() -> Option<AllocatorKind> {
    if let Ok(v) = std::env::var("RNB_MOE_HOT_POOL") {
        return match v.to_ascii_lowercase().as_str() {
            "heap" => Some(AllocatorKind::Heap),
            "memfd" => Some(AllocatorKind::Memfd),
            _ => None,
        };
    }
    if std::env::var("RNB_MOE_HEAP").is_ok() {
        return Some(AllocatorKind::Heap);
    }
    None
}

pub fn configured_hot_pool_plan(metadata: &[u8]) -> ConfiguredHotPoolPlan {
    let Some(allocator_kind) = parse_env_allocator_kind() else {
        return ConfiguredHotPoolPlan::Disabled;
    };
    match crate::moe_policy::effective_hot_count(crate::moe_preheat::parse_hot_count(metadata)) {
        Some(hot_count) => ConfiguredHotPoolPlan::Enabled {
            allocator_kind,
            hot_count,
        },
        None => ConfiguredHotPoolPlan::MissingHotCount { allocator_kind },
    }
}

pub fn alloc_required_prefix(
    allocator_kind: AllocatorKind,
    label: impl std::fmt::Display,
    src: &[u8],
    hot_bytes: usize,
) -> Result<HotPool, String> {
    if hot_bytes > src.len() {
        return Err(format!(
            "{label}: hot prefix needs {hot_bytes} bytes but source has {}",
            src.len()
        ));
    }
    HotPool::alloc_and_fill(allocator_kind, &src[..hot_bytes]).map_err(|e| format!("{label}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_roundtrip() {
        let src = (0u8..128).collect::<Vec<_>>();
        let pool = HotPool::alloc_and_fill(AllocatorKind::Heap, &src).unwrap();
        assert_eq!(pool.kind(), "heap");
        assert_eq!(pool.len(), 128);
        assert_eq!(pool.as_slice(), src.as_slice());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn memfd_roundtrip() {
        let src = (0u8..200).collect::<Vec<_>>();
        let pool = HotPool::alloc_and_fill(AllocatorKind::Memfd, &src).unwrap();
        assert_eq!(pool.kind(), "memfd");
        assert_eq!(pool.len(), 200);
        assert_eq!(pool.as_slice(), src.as_slice());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn memfd_64mb_allocates_and_reads_back() {
        // Exercise a larger pool to be sure ftruncate + mmap works.
        let size = 64 * 1024 * 1024;
        let mut src = vec![0u8; size];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31);
        }
        let pool = HotPool::alloc_and_fill(AllocatorKind::Memfd, &src).unwrap();
        assert_eq!(pool.len(), size);
        assert_eq!(&pool.as_slice()[..1024], &src[..1024]);
        assert_eq!(&pool.as_slice()[size - 1024..], &src[size - 1024..]);
    }
}
