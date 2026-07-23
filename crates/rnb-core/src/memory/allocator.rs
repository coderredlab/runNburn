use crate::error::Result;
use crate::tensor::storage::Buffer;

#[derive(Debug, Default)]
pub struct AllocStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub allocation_count: u64,
}

pub trait Allocator: Send + Sync {
    fn alloc(&self, size: usize, align: usize) -> Result<Buffer>;
    fn dealloc(&self, buf: Buffer);
    fn stats(&self) -> AllocStats;
}
