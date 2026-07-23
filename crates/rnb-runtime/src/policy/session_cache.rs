use super::HostMemoryPlan;

const RESPONSE_SESSION_BUDGET_DIVISOR: u64 = 16;

/// Derives a host-memory cap for stored response history and compact sequence
/// snapshots. An explicit application value wins. The automatic cap scales down
/// with available RAM and never exceeds the mapped model size.
pub fn response_session_cache_budget_bytes(
    plan: HostMemoryPlan,
    explicit_bytes: Option<u64>,
) -> u64 {
    if let Some(bytes) = explicit_bytes {
        return bytes;
    }
    let Some(ram_budget) = plan.ram_budget() else {
        return 0;
    };
    let proportional = ram_budget.available_bytes() / RESPONSE_SESSION_BUDGET_DIVISOR;
    let model_bound = plan.mapped_weight_bytes();
    if model_bound == 0 {
        proportional
    } else {
        proportional.min(model_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_memory::{MemoryBudget, MemoryTier};

    #[test]
    fn explicit_budget_wins() {
        let plan = HostMemoryPlan::new(None, 4_000, false);
        assert_eq!(response_session_cache_budget_bytes(plan, Some(123)), 123);
    }

    #[test]
    fn automatic_budget_scales_with_ram_and_model() {
        let small = HostMemoryPlan::new(
            Some(MemoryBudget::new(MemoryTier::Ram, 1_600, 0)),
            1_000,
            false,
        );
        let large = HostMemoryPlan::new(
            Some(MemoryBudget::new(MemoryTier::Ram, 32_000, 0)),
            1_000,
            false,
        );
        assert_eq!(response_session_cache_budget_bytes(small, None), 100);
        assert_eq!(response_session_cache_budget_bytes(large, None), 1_000);
    }
}
