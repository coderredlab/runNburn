use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::Result;
use crate::tensor::storage::Buffer;

use super::allocator::{AllocStats, Allocator};

pub struct PoolAllocator {
    pool: Mutex<HashMap<usize, Vec<Buffer>>>,
    max_pool_size: usize,
    stats: Mutex<PoolStats>,
}

#[derive(Default)]
struct PoolStats {
    allocated_bytes: usize,
    peak_bytes: usize,
    allocation_count: u64,
}

impl PoolAllocator {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            max_pool_size,
            stats: Mutex::new(PoolStats::default()),
        }
    }
}

impl Allocator for PoolAllocator {
    fn alloc(&self, size: usize, align: usize) -> Result<Buffer> {
        // 풀에서 같은 사이즈 Buffer가 있으면 재사용
        let reused = {
            let mut pool = self.pool.lock().unwrap();
            pool.get_mut(&size).and_then(|v| v.pop())
        };

        let buf = match reused {
            Some(b) => b,
            None => Buffer::alloc(size, align),
        };

        let mut s = self.stats.lock().unwrap();
        s.allocated_bytes += size;
        s.allocation_count += 1;
        if s.allocated_bytes > s.peak_bytes {
            s.peak_bytes = s.allocated_bytes;
        }

        Ok(buf)
    }

    fn dealloc(&self, buf: Buffer) {
        let size = buf.len();
        let mut s = self.stats.lock().unwrap();
        s.allocated_bytes = s.allocated_bytes.saturating_sub(size);
        drop(s);

        // max_pool_size 미만이면 풀에 반환
        let mut pool = self.pool.lock().unwrap();
        let entry = pool.entry(size).or_default();
        if entry.len() < self.max_pool_size {
            entry.push(buf);
        }
        // 초과면 그냥 drop (Buffer의 Drop impl이 메모리 해제)
    }

    fn stats(&self) -> AllocStats {
        let s = self.stats.lock().unwrap();
        AllocStats {
            allocated_bytes: s.allocated_bytes,
            peak_bytes: s.peak_bytes,
            allocation_count: s.allocation_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_alloc_and_reuse() {
        let pool = PoolAllocator::new(16);
        let buf1 = pool.alloc(64, 8).unwrap();
        let ptr1 = buf1.as_ptr();
        pool.dealloc(buf1);
        let buf2 = pool.alloc(64, 8).unwrap();
        assert_eq!(buf2.as_ptr(), ptr1);
    }

    #[test]
    fn test_pool_different_sizes() {
        let pool = PoolAllocator::new(16);
        let buf1 = pool.alloc(64, 8).unwrap();
        let buf2 = pool.alloc(128, 8).unwrap();
        assert_ne!(buf1.len(), buf2.len());
    }
}
