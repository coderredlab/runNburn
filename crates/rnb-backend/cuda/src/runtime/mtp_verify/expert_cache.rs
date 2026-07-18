use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::runtime) struct MtpExpertCandidate {
    pub(in crate::runtime) expert_id: u32,
    pub(in crate::runtime) first_slot: usize,
    pub(in crate::runtime) slot_count: usize,
    pub(in crate::runtime) route_score: f32,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::runtime) struct MtpExpertHotScore {
    pub(in crate::runtime) hits: u32,
    pub(in crate::runtime) misses: u32,
    pub(in crate::runtime) route_score: f32,
    pub(in crate::runtime) upload_bytes: usize,
}

#[cfg(test)]
impl MtpExpertHotScore {
    pub(in crate::runtime) fn score(&self) -> f32 {
        let hit_term = self.hits as f32 * 4.0;
        let miss_penalty = self.misses as f32 * 1.5;
        let upload_penalty = (self.upload_bytes as f32) / (16.0 * 1024.0 * 1024.0);
        hit_term + self.route_score - miss_penalty - upload_penalty
    }
}

pub(in crate::runtime) fn extra_expert_candidates(
    expert_ids: &[u32],
    route_weights: &[f32],
    previous: Option<&HashSet<u32>>,
) -> Vec<MtpExpertCandidate> {
    let mut by_expert = HashMap::new();
    for (slot, &expert_id) in expert_ids.iter().enumerate() {
        if previous.is_some_and(|prev| prev.contains(&expert_id)) {
            continue;
        }
        let route_weight = route_weights
            .get(slot)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        let entry = by_expert.entry(expert_id).or_insert(MtpExpertCandidate {
            expert_id,
            first_slot: slot,
            slot_count: 0,
            route_score: 0.0,
        });
        entry.slot_count += 1;
        entry.route_score += route_weight;
    }
    let mut candidates = by_expert.into_values().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .slot_count
            .cmp(&left.slot_count)
            .then_with(|| {
                right
                    .route_score
                    .partial_cmp(&left.route_score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.first_slot.cmp(&right.first_slot))
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });
    candidates
}

#[cfg(test)]
mod mtp_expert_cache_tests {
    use super::*;

    #[test]
    fn candidates_prioritize_reuse_then_route_weight() {
        let previous = HashSet::from([1_u32]);
        let expert_ids = [1_u32, 2, 3, 2, 4, 3, 3];
        let route_weights = [0.9_f32, 0.2, 0.1, 0.8, 0.7, 0.2, 0.2];

        let candidates = extra_expert_candidates(&expert_ids, &route_weights, Some(&previous));
        let ordered = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.expert_id,
                    candidate.slot_count,
                    candidate.first_slot,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(ordered, vec![(3, 3, 2), (2, 2, 1), (4, 1, 4)]);
        assert!((candidates[0].route_score - 0.5).abs() < 1.0e-6);
        assert!((candidates[1].route_score - 1.0).abs() < 1.0e-6);
        assert!((candidates[2].route_score - 0.7).abs() < 1.0e-6);
    }

    #[test]
    fn hot_score_rewards_hits_and_penalizes_uploads() {
        let hot = MtpExpertHotScore {
            hits: 4,
            misses: 0,
            route_score: 1.0,
            upload_bytes: 4 * 1024 * 1024,
        };
        let cold = MtpExpertHotScore {
            hits: 1,
            misses: 3,
            route_score: 1.0,
            upload_bytes: 32 * 1024 * 1024,
        };
        assert!(hot.score() > cold.score());
    }
}
