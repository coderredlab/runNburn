//! Memory-tier primitives and runtime memory policies.
//!
//! This crate is the destination for policy that decides where model bytes
//! live and how they are touched: budgets, cache tiers, page residency,
//! MoE hot/cold split decisions, preheat, and direct cold IO. It intentionally
//! does not own model execution, routing math, or session wiring.

pub mod byte_lru;
pub use byte_lru::ByteLruPolicy;
pub mod device_residency;
pub use device_residency::{default_device_dynamic_reserve_bytes, DeviceResidencyPlan};
pub mod memtrace;
pub mod mlock;
pub mod moe_cold_io;
pub mod residency;
pub mod sparse_expert_cache;

pub use sparse_expert_cache::{
    evaluate_expert_bundle_admission, CurrentLookupTransfer, ExpertBundleAdmissionCandidate,
    ExpertBundleAdmissionCost, ExpertBundleAdmissionDecision, ExpertBundleCacheStats,
    ExpertBundleFootprint, ExpertBundleObservationReceipt, ExpertBundleResidency,
    ExpertBundleReuseHistory, SparseExpertCacheKey,
};

pub const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;
pub const GIB: u64 = 1024 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    Ram,
    Vram,
    Npu,
    Ufs,
    File,
    Mmap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    tier: MemoryTier,
    total_bytes: u64,
    reserved_bytes: u64,
}

impl MemoryBudget {
    pub const fn new(tier: MemoryTier, total_bytes: u64, reserved_bytes: u64) -> Self {
        Self {
            tier,
            total_bytes,
            reserved_bytes,
        }
    }

    pub const fn tier(self) -> MemoryTier {
        self.tier
    }

    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    pub const fn reserved_bytes(self) -> u64 {
        self.reserved_bytes
    }

    pub const fn available_bytes(self) -> u64 {
        self.total_bytes.saturating_sub(self.reserved_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicy {
    None,
    LeastRecentlyUsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePolicy {
    tier: MemoryTier,
    eviction: EvictionPolicy,
}

impl CachePolicy {
    pub const fn new(tier: MemoryTier, eviction: EvictionPolicy) -> Self {
        Self { tier, eviction }
    }

    pub const fn tier(self) -> MemoryTier {
        self.tier
    }

    pub const fn eviction(self) -> EvictionPolicy {
        self.eviction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_reports_available_bytes_after_reserve() {
        let budget = MemoryBudget::new(MemoryTier::Vram, 12 * GIB, 2 * GIB);
        assert_eq!(budget.available_bytes(), 10 * GIB);
    }

    #[test]
    fn cache_policy_records_eviction_strategy() {
        let policy = CachePolicy::new(MemoryTier::Vram, EvictionPolicy::LeastRecentlyUsed);
        assert_eq!(policy.tier(), MemoryTier::Vram);
        assert_eq!(policy.eviction(), EvictionPolicy::LeastRecentlyUsed);
    }

    #[test]
    fn npu_memory_budget_uses_distinct_tier() {
        let budget = MemoryBudget::new(MemoryTier::Npu, 512 * MIB, 64 * MIB);

        assert_eq!(budget.tier(), MemoryTier::Npu);
        assert_eq!(budget.available_bytes(), 448 * MIB);
    }
}
