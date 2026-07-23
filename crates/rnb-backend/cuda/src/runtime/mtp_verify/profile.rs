#[derive(Debug, Clone, Default, PartialEq)]
pub(in crate::runtime) struct MtpVerifyExpertResidencyProfile {
    pub(in crate::runtime) layer_index: usize,
    pub(in crate::runtime) slots: usize,
    pub(in crate::runtime) predicted_hits: usize,
    pub(in crate::runtime) resident_hits_before: usize,
    pub(in crate::runtime) resident_misses_before: usize,
    pub(in crate::runtime) resident_hits_after: usize,
    pub(in crate::runtime) resident_misses_after: usize,
    pub(in crate::runtime) temp_h2d_bytes: usize,
    pub(in crate::runtime) promoted_bytes: usize,
    pub(in crate::runtime) weight_ptr_ms: f64,
    pub(in crate::runtime) setup_h2d_ms: f64,
    pub(in crate::runtime) kernels_ms: f64,
}

impl MtpVerifyExpertResidencyProfile {
    pub(in crate::runtime) fn predicted_hit_rate(&self) -> f64 {
        if self.slots == 0 {
            0.0
        } else {
            (self.predicted_hits as f64 * 100.0) / self.slots as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MtpVerifyExpertResidencyProfile;

    #[test]
    fn profile_predicted_hit_rate_handles_empty_and_non_empty_slots() {
        let empty = MtpVerifyExpertResidencyProfile {
            slots: 0,
            predicted_hits: 3,
            ..Default::default()
        };
        assert_eq!(empty.predicted_hit_rate(), 0.0);

        let half = MtpVerifyExpertResidencyProfile {
            slots: 8,
            predicted_hits: 4,
            ..Default::default()
        };
        assert_eq!(half.predicted_hit_rate(), 50.0);
    }
}
