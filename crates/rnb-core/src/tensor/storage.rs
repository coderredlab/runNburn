use std::alloc::{self, Layout};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock, Weak};

pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn alloc(size: usize, align: usize) -> Self {
        if size == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: Layout::from_size_align(0, align).unwrap(),
            };
        }
        let layout = Layout::from_size_align(size, align).expect("invalid layout");
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        Self {
            ptr,
            len: size,
            layout,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Take ownership of a Vec's allocation (zero-copy).
    /// The Vec must not be empty.
    pub fn from_vec<T>(mut v: Vec<T>) -> Self {
        let byte_len = v.len() * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = v.as_mut_ptr() as *mut u8;
        let layout = Layout::from_size_align(v.capacity() * std::mem::size_of::<T>(), align)
            .expect("invalid layout");
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        std::mem::forget(v); // prevent Vec from deallocating
        Self {
            ptr,
            len: byte_len,
            layout,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

pub struct FileMmapStorage {
    mmap: memmap2::Mmap,
    path: Arc<PathBuf>,
}

impl FileMmapStorage {
    pub fn new(mmap: memmap2::Mmap, path: &Path) -> Self {
        Self {
            mmap,
            path: Arc::new(path.to_path_buf()),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.mmap.as_ref()
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    pub fn path_arc(&self) -> Arc<PathBuf> {
        Arc::clone(&self.path)
    }

    #[cfg(unix)]
    pub unsafe fn unchecked_advise_range(
        &self,
        advice: memmap2::UncheckedAdvice,
        offset: usize,
        len: usize,
    ) -> std::io::Result<()> {
        unsafe { self.mmap.unchecked_advise_range(advice, offset, len) }
    }
}

#[derive(Clone, Debug)]
pub struct FileBackedRegion {
    path: Arc<PathBuf>,
    file_offset: u64,
    len: usize,
    host_address: usize,
}

impl FileBackedRegion {
    pub(crate) fn new(
        path: Arc<PathBuf>,
        file_offset: u64,
        len: usize,
        host_address: usize,
    ) -> Self {
        Self {
            path,
            file_offset,
            len,
            host_address,
        }
    }

    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }

    /// mmap 상 region 시작 주소 — slice 포인터에서 파일 오프셋 역산용.
    pub fn host_address(&self) -> usize {
        self.host_address
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn resolve_subslice(&self, bytes: &[u8]) -> Option<(u64, usize)> {
        let start = bytes.as_ptr() as usize;
        let end = start.checked_add(bytes.len())?;
        let region_end = self.host_address.checked_add(self.len)?;
        if start < self.host_address || end > region_end {
            return None;
        }
        let relative = start - self.host_address;
        let file_offset = self.file_offset.checked_add(relative as u64)?;
        Some((file_offset, bytes.len()))
    }
}

pub struct DeviceBuffer {
    pub id: u64,
    pub size: usize,
}

/// Stable identity for an immutable byte range within a registered host storage.
///
/// `allocation_id` changes when a storage allocation is recreated, even if the
/// allocator reuses the same virtual address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostStorageIdentity {
    pub allocation_id: u64,
    pub byte_offset: usize,
    pub len: usize,
}
/// Strong lease for a registered host storage allocation.
///
/// Keeping this value alive keeps the allocation behind a resolved byte range
/// alive, so external zero-copy views cannot outlive their source memory.
#[derive(Clone)]
pub struct HostStorageLease {
    identity: HostStorageIdentity,
    _storage: Arc<Storage>,
}

impl std::fmt::Debug for HostStorageLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostStorageLease")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

struct HostStorageRegistration {
    allocation_id: u64,
    storage: Weak<Storage>,
    base: usize,
    len: usize,
}

static NEXT_HOST_STORAGE_ID: AtomicU64 = AtomicU64::new(1);
static HOST_STORAGE_REGISTRY: LazyLock<RwLock<Vec<HostStorageRegistration>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

/// Registers model storage whose byte ranges need allocation-stable identities.
///
/// Registrations hold only a `Weak` reference and therefore do not extend the
/// storage lifetime.
pub(crate) fn register_host_storage(storage: &Arc<Storage>) {
    let Some(bytes) = storage.as_slice() else {
        return;
    };
    let base = bytes.as_ptr() as usize;
    let len = bytes.len();
    let weak = Arc::downgrade(storage);
    let mut registrations = HOST_STORAGE_REGISTRY
        .write()
        .expect("host storage registry lock poisoned");
    registrations.retain(|entry| entry.storage.strong_count() > 0);
    if registrations
        .iter()
        .any(|entry| Weak::ptr_eq(&entry.storage, &weak))
    {
        return;
    }
    registrations.push(HostStorageRegistration {
        allocation_id: NEXT_HOST_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
        storage: weak,
        base,
        len,
    });
}

/// Resolves a byte slice to its registered storage allocation and relative range.
///
/// Returns `None` for ordinary transient buffers and after their storage dies.
pub fn host_storage_identity(bytes: &[u8]) -> Option<HostStorageIdentity> {
    let start = bytes.as_ptr() as usize;
    let end = start.checked_add(bytes.len())?;
    let registrations = HOST_STORAGE_REGISTRY.read().ok()?;
    registrations.iter().find_map(|entry| {
        entry.storage.upgrade()?;
        let storage_end = entry.base.checked_add(entry.len)?;
        if start < entry.base || end > storage_end {
            return None;
        }
        Some(HostStorageIdentity {
            allocation_id: entry.allocation_id,
            byte_offset: start - entry.base,
            len: bytes.len(),
        })
    })
}

/// Resolves a byte slice and retains its registered storage allocation.
///
/// Unlike [`host_storage_identity`], the returned lease is an owning lifetime
/// link suitable for caches of external zero-copy views.
pub fn host_storage_lease(bytes: &[u8]) -> Option<HostStorageLease> {
    let start = bytes.as_ptr() as usize;
    let end = start.checked_add(bytes.len())?;
    let registrations = HOST_STORAGE_REGISTRY.read().ok()?;
    registrations.iter().find_map(|entry| {
        let storage = entry.storage.upgrade()?;
        let storage_end = entry.base.checked_add(entry.len)?;
        if start < entry.base || end > storage_end {
            return None;
        }
        Some(HostStorageLease {
            identity: HostStorageIdentity {
                allocation_id: entry.allocation_id,
                byte_offset: start - entry.base,
                len: bytes.len(),
            },
            _storage: storage,
        })
    })
}

pub enum Storage {
    Owned(Buffer),
    Mmap(memmap2::Mmap),
    /// Immutable file-backed mapping. Unlike `Mmap`, ranges may be reclaimed
    /// with `MADV_DONTNEED` because no anonymous/COW contents can be lost.
    FileMmap(FileMmapStorage),
    Device(DeviceBuffer),
}

impl Storage {
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Storage::Owned(buf) => Some(buf.as_slice()),
            Storage::Mmap(mmap) => Some(mmap.as_ref()),
            Storage::FileMmap(mmap) => Some(mmap.as_slice()),
            Storage::Device(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Owned(buf) => buf.len(),
            Storage::Mmap(mmap) => mmap.len(),
            Storage::FileMmap(mmap) => mmap.len(),
            Storage::Device(dev) => dev.size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_alloc_and_access() {
        let mut buf = Buffer::alloc(16, 8);
        assert_eq!(buf.len(), 16);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    fn test_buffer_zero_size() {
        let buf = Buffer::alloc(0, 8);
        assert_eq!(buf.len(), 0);
        assert!(buf.as_slice().is_empty());
    }

    #[test]
    fn test_storage_owned_as_slice() {
        let buf = Buffer::alloc(32, 8);
        let storage = Storage::Owned(buf);
        assert_eq!(storage.as_slice().unwrap().len(), 32);
    }

    #[test]
    fn host_storage_lease_keeps_registered_allocation_alive() {
        let storage = Arc::new(Storage::Owned(Buffer::from_vec(vec![0u8; 32])));
        register_host_storage(&storage);
        let weak_storage = Arc::downgrade(&storage);
        let lease = host_storage_lease(storage.as_slice().expect("host storage"))
            .expect("registered storage lease");

        drop(storage);
        assert!(
            weak_storage.upgrade().is_some(),
            "lease must retain the registered allocation"
        );

        drop(lease);
        assert!(
            weak_storage.upgrade().is_none(),
            "allocation must be released with its last lease"
        );
    }
}
