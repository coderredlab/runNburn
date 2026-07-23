//! Backend-neutral device residency budget policy.
//!
//! A single plan caps all optional resident weight representations together.
//! Backend-local cache limits remain class-specific safety ceilings, but they
//! may not collectively exceed this plan. Runtime scratch, activations, KV,
//! and allocator fragmentation are protected by the dynamic reserve.

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;
const RESERVE_ALIGNMENT: usize = 256 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceResidencyPlan {
    pub total_bytes: usize,
    pub initial_free_bytes: usize,
    pub dynamic_reserve_bytes: usize,
    pub resident_limit_bytes: usize,
}

impl DeviceResidencyPlan {
    pub fn from_snapshot(
        total_bytes: usize,
        free_bytes: usize,
        dynamic_reserve_bytes: usize,
    ) -> Self {
        let initial_free_bytes = free_bytes.min(total_bytes);
        let dynamic_reserve_bytes = dynamic_reserve_bytes.min(total_bytes);
        let resident_limit_bytes = initial_free_bytes.saturating_sub(dynamic_reserve_bytes);
        Self {
            total_bytes,
            initial_free_bytes,
            dynamic_reserve_bytes,
            resident_limit_bytes,
        }
    }

    pub fn allows_resident_admission(
        self,
        resident_bytes: usize,
        incoming_bytes: usize,
        current_free_bytes: usize,
    ) -> bool {
        resident_bytes.saturating_add(incoming_bytes) <= self.resident_limit_bytes
            && incoming_bytes.saturating_add(self.dynamic_reserve_bytes) <= current_free_bytes
    }

    pub fn resident_limit_for_class(
        self,
        class_bytes: usize,
        other_resident_bytes: usize,
    ) -> usize {
        class_bytes.saturating_add(
            self.resident_limit_bytes
                .saturating_sub(class_bytes.saturating_add(other_resident_bytes)),
        )
    }

    pub fn transient_reclaim_bytes(
        self,
        current_free_bytes: usize,
        requested_bytes: usize,
    ) -> usize {
        requested_bytes
            .saturating_add(self.dynamic_reserve_bytes)
            .saturating_sub(current_free_bytes)
    }
}

pub fn default_device_dynamic_reserve_bytes(total_bytes: usize) -> usize {
    let proportional = align_up(total_bytes / 16, RESERVE_ALIGNMENT);
    proportional.clamp((512 * MIB).min(total_bytes), (2 * GIB).min(total_bytes))
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    value
        .saturating_add(alignment - 1)
        .checked_div(alignment)
        .unwrap_or(usize::MAX / alignment)
        .saturating_mul(alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_scales_down_and_caps_on_large_devices() {
        assert_eq!(default_device_dynamic_reserve_bytes(4 * GIB), 512 * MIB);
        assert_eq!(default_device_dynamic_reserve_bytes(24 * GIB), 1536 * MIB);
        assert_eq!(default_device_dynamic_reserve_bytes(80 * GIB), 2 * GIB);
    }

    #[test]
    fn one_resident_limit_protects_dynamic_headroom() {
        let plan = DeviceResidencyPlan::from_snapshot(24 * GIB, 23 * GIB, 4 * GIB);
        assert_eq!(plan.resident_limit_bytes, 19 * GIB);
        assert!(plan.allows_resident_admission(10 * GIB, GIB, 13 * GIB));
        assert!(!plan.allows_resident_admission(18 * GIB, 2 * GIB, 5 * GIB));
        assert!(!plan.allows_resident_admission(10 * GIB, 2 * GIB, 5 * GIB));
    }

    #[test]
    fn class_limit_uses_only_unclaimed_global_bytes() {
        let plan = DeviceResidencyPlan::from_snapshot(24 * GIB, 23 * GIB, 4 * GIB);
        assert_eq!(plan.resident_limit_for_class(8 * GIB, 7 * GIB), 12 * GIB);
        assert_eq!(plan.resident_limit_for_class(8 * GIB, 15 * GIB), 8 * GIB);
    }

    #[test]
    fn transient_reclaim_tracks_sequence_dependent_request() {
        let plan = DeviceResidencyPlan::from_snapshot(24 * GIB, 23 * GIB, 4 * GIB);
        assert_eq!(plan.transient_reclaim_bytes(6 * GIB, GIB), 0);
        assert_eq!(plan.transient_reclaim_bytes(4 * GIB, 2 * GIB), 2 * GIB);
    }
}
