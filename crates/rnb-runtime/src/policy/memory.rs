use rnb_memory::{MemoryBudget, MemoryTier};

const HOST_RUNTIME_RESERVE_DIVISOR: u64 = 4;

/// Runtime policy derived from application-provided or detected host RAM.
/// The budget limits engine-owned host residency and cache decisions. File-backed
/// GGUF mappings remain reclaimable kernel page-cache pages, so this is not an OS
/// RSS hard limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryPlan {
    ram_budget: Option<MemoryBudget>,
    mapped_weight_bytes: u64,
    constrained: bool,
    prefer_sparse_moe_cuda: bool,
    automatic_budget: bool,
}

impl HostMemoryPlan {
    /// Builds a deterministic plan from an explicit application budget.
    pub fn new(
        budget: Option<MemoryBudget>,
        mapped_weight_bytes: u64,
        cuda_available: bool,
    ) -> Self {
        let ram_budget = budget.filter(|budget| budget.tier() == MemoryTier::Ram);
        let constrained =
            ram_budget.is_some_and(|budget| budget.available_bytes() < mapped_weight_bytes);
        Self {
            ram_budget,
            mapped_weight_bytes,
            constrained,
            prefer_sparse_moe_cuda: constrained && cuda_available,
            automatic_budget: false,
        }
    }

    /// Builds a plan from an explicit budget, or detected host memory when none was supplied.
    ///
    /// Automatic policy reserves one quarter of both physical RAM capacity and
    /// currently available RAM. The lower bound limits engine-owned file-backed
    /// sparse-expert residency without forcing active applications into swap.
    pub fn automatic(
        budget: Option<MemoryBudget>,
        mapped_weight_bytes: u64,
        cuda_available: bool,
    ) -> Self {
        Self::from_capacity(
            budget,
            mapped_weight_bytes,
            cuda_available,
            rnb_platform::host_physical_memory_bytes(),
            rnb_platform::host_available_memory_bytes(),
        )
    }

    fn from_capacity(
        budget: Option<MemoryBudget>,
        mapped_weight_bytes: u64,
        cuda_available: bool,
        physical_ram_bytes: Option<u64>,
        available_ram_bytes: Option<u64>,
    ) -> Self {
        if budget.is_some() {
            return Self::new(budget, mapped_weight_bytes, cuda_available);
        }
        let Some(total_bytes) = physical_ram_bytes.filter(|&bytes| bytes > 0) else {
            return Self::new(None, mapped_weight_bytes, cuda_available);
        };
        let capacity_budget_bytes =
            total_bytes.saturating_sub(total_bytes / HOST_RUNTIME_RESERVE_DIVISOR);
        let pressure_budget_bytes = available_ram_bytes
            .filter(|&bytes| bytes > 0)
            .map(|bytes| bytes.min(total_bytes))
            .map(|bytes| bytes.saturating_sub(bytes / HOST_RUNTIME_RESERVE_DIVISOR))
            .unwrap_or(capacity_budget_bytes);
        let budget_bytes = capacity_budget_bytes.min(pressure_budget_bytes);
        let reserved_bytes = total_bytes.saturating_sub(budget_bytes);
        let mut plan = Self::new(
            Some(MemoryBudget::new(
                MemoryTier::Ram,
                total_bytes,
                reserved_bytes,
            )),
            mapped_weight_bytes,
            cuda_available,
        );
        plan.automatic_budget = true;
        plan
    }

    pub const fn ram_budget(self) -> Option<MemoryBudget> {
        self.ram_budget
    }

    pub const fn mapped_weight_bytes(self) -> u64 {
        self.mapped_weight_bytes
    }

    pub const fn is_constrained(self) -> bool {
        self.constrained
    }

    pub const fn prefer_sparse_moe_cuda(self) -> bool {
        self.prefer_sparse_moe_cuda
    }

    pub const fn uses_automatic_budget(self) -> bool {
        self.automatic_budget
    }
}

impl Default for HostMemoryPlan {
    fn default() -> Self {
        Self::new(None, 0, false)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn constrained_ram_budget_prefers_sparse_cuda_when_available() {
        let budget = MemoryBudget::new(MemoryTier::Ram, 32 * GIB, 0);
        let plan = HostMemoryPlan::new(Some(budget), 100 * GIB, true);

        assert!(plan.is_constrained());
        assert!(plan.prefer_sparse_moe_cuda());
        assert_eq!(plan.ram_budget(), Some(budget));
        assert_eq!(plan.mapped_weight_bytes(), 100 * GIB);
    }

    #[test]
    fn constrained_ram_budget_does_not_claim_unavailable_cuda() {
        let budget = MemoryBudget::new(MemoryTier::Ram, 16 * GIB, 0);
        let plan = HostMemoryPlan::new(Some(budget), 100 * GIB, false);

        assert!(plan.is_constrained());
        assert!(!plan.prefer_sparse_moe_cuda());
    }

    #[test]
    fn fitting_or_non_ram_budget_does_not_change_host_execution() {
        let fitting = MemoryBudget::new(MemoryTier::Ram, 128 * GIB, 8 * GIB);
        let vram = MemoryBudget::new(MemoryTier::Vram, 16 * GIB, 0);

        assert!(!HostMemoryPlan::new(Some(fitting), 100 * GIB, true).is_constrained());
        assert!(!HostMemoryPlan::new(Some(vram), 100 * GIB, true).is_constrained());
    }

    #[test]
    fn automatic_budget_reserves_quarter_of_available_ram() {
        let plan =
            HostMemoryPlan::from_capacity(None, 100 * GIB, true, Some(16 * GIB), Some(16 * GIB));
        let budget = plan.ram_budget().expect("automatic RAM budget");

        assert_eq!(budget.total_bytes(), 16 * GIB);
        assert_eq!(budget.reserved_bytes(), 4 * GIB);
        assert_eq!(budget.available_bytes(), 12 * GIB);
        assert!(plan.uses_automatic_budget());
        assert!(plan.is_constrained());
        assert!(plan.prefer_sparse_moe_cuda());
    }

    #[test]
    fn automatic_budget_shrinks_under_current_memory_pressure() {
        let plan =
            HostMemoryPlan::from_capacity(None, 100 * GIB, true, Some(16 * GIB), Some(8 * GIB));
        let budget = plan.ram_budget().expect("automatic RAM budget");

        assert_eq!(budget.total_bytes(), 16 * GIB);
        assert_eq!(budget.reserved_bytes(), 10 * GIB);
        assert_eq!(budget.available_bytes(), 6 * GIB);
        assert!(plan.uses_automatic_budget());
    }

    #[test]
    fn explicit_budget_overrides_detected_physical_ram() {
        let explicit = MemoryBudget::new(MemoryTier::Ram, 10 * GIB, 0);
        let plan = HostMemoryPlan::from_capacity(
            Some(explicit),
            100 * GIB,
            true,
            Some(16 * GIB),
            Some(GIB),
        );

        assert_eq!(plan.ram_budget(), Some(explicit));
        assert!(!plan.uses_automatic_budget());
    }

    #[test]
    fn unavailable_physical_ram_keeps_automatic_budget_disabled() {
        let plan = HostMemoryPlan::from_capacity(None, 100 * GIB, true, None, None);

        assert_eq!(plan.ram_budget(), None);
        assert!(!plan.uses_automatic_budget());
        assert!(!plan.is_constrained());
    }
}
