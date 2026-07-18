use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::error::{Result, RnbError};
use crate::tensor::storage::Buffer;

use super::allocator::{AllocStats, Allocator};

pub struct ArenaAllocator {
    capacity: usize,
    offset: AtomicUsize,
    allocation_count: AtomicU64,
    freed_count: AtomicU64,
}

impl ArenaAllocator {
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            capacity,
            offset: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            freed_count: AtomicU64::new(0),
        })
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
        self.freed_count.store(
            self.allocation_count.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }
}

impl Allocator for ArenaAllocator {
    fn alloc(&self, size: usize, align: usize) -> Result<Buffer> {
        // align 고려해서 offset 계산
        let current = self.offset.load(Ordering::SeqCst);
        let aligned = (current + align - 1) & !(align - 1);
        let next = aligned + size;

        if next > self.capacity {
            return Err(RnbError::OutOfMemory {
                requested: size,
                available: self.capacity.saturating_sub(current),
            });
        }

        // CAS로 atomic하게 offset 업데이트
        match self
            .offset
            .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => {
                self.allocation_count.fetch_add(1, Ordering::SeqCst);
                Ok(Buffer::alloc(size, align))
            }
            Err(actual) => {
                // 다른 스레드가 먼저 바꿨으면 재시도 - 단순하게 현재 값 기준으로 체크
                let next2 = actual + size;
                if next2 > self.capacity {
                    return Err(RnbError::OutOfMemory {
                        requested: size,
                        available: self.capacity.saturating_sub(actual),
                    });
                }
                self.offset.fetch_add(size, Ordering::SeqCst);
                self.allocation_count.fetch_add(1, Ordering::SeqCst);
                Ok(Buffer::alloc(size, align))
            }
        }
    }

    fn dealloc(&self, _buf: Buffer) {
        // Arena는 개별 dealloc 안 함 — reset()으로 전체 해제
        // Buffer는 drop될 때 자동으로 메모리 해제됨
    }

    fn stats(&self) -> AllocStats {
        let offset = self.offset.load(Ordering::SeqCst);
        let count = self.allocation_count.load(Ordering::SeqCst);
        let freed = self.freed_count.load(Ordering::SeqCst);
        AllocStats {
            allocated_bytes: offset,
            peak_bytes: self.capacity,
            allocation_count: count.saturating_sub(freed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let arena = ArenaAllocator::new(1024).unwrap();
        let buf = arena.alloc(64, 8).unwrap();
        assert_eq!(buf.len(), 64);
        assert_eq!(arena.stats().allocation_count, 1);
    }

    #[test]
    fn test_arena_oom() {
        let arena = ArenaAllocator::new(64).unwrap();
        let result = arena.alloc(128, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_arena_reset() {
        let arena = ArenaAllocator::new(256).unwrap();
        let _ = arena.alloc(128, 8).unwrap();
        assert!(arena.stats().allocated_bytes >= 128);
        arena.reset();
        assert_eq!(arena.stats().allocated_bytes, 0);
        let buf = arena.alloc(128, 8).unwrap();
        assert_eq!(buf.len(), 128);
    }
}
