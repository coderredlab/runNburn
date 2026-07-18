#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyCandidate {
    id: String,
    bytes: u64,
    reuse_weight: u32,
}

impl ResidencyCandidate {
    pub fn new(id: impl Into<String>, bytes: u64, reuse_weight: u32) -> Self {
        Self {
            id: id.into(),
            bytes,
            reuse_weight,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn reuse_weight(&self) -> u32 {
        self.reuse_weight
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
    selected: Vec<ResidencyCandidate>,
    spilled: Vec<ResidencyCandidate>,
}

impl ResidencyPlan {
    pub fn selected(&self) -> &[ResidencyCandidate] {
        &self.selected
    }

    pub fn spilled(&self) -> &[ResidencyCandidate] {
        &self.spilled
    }

    pub fn selected_ids(&self) -> Vec<&str> {
        self.selected
            .iter()
            .map(|candidate| candidate.id())
            .collect()
    }

    pub fn selected_bytes(&self) -> u64 {
        self.selected
            .iter()
            .map(|candidate| candidate.bytes())
            .sum()
    }

    pub fn spill_bytes(&self) -> u64 {
        self.spilled.iter().map(|candidate| candidate.bytes()).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyPlanner {
    budget_bytes: u64,
}

impl ResidencyPlanner {
    pub const fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes }
    }

    pub fn plan(&self, candidates: &[ResidencyCandidate]) -> ResidencyPlan {
        let mut ranked = candidates
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<(usize, ResidencyCandidate)>>();
        ranked.sort_by(|(left_idx, left), (right_idx, right)| {
            right
                .reuse_weight
                .cmp(&left.reuse_weight)
                .then_with(|| left.bytes.cmp(&right.bytes))
                .then_with(|| left_idx.cmp(right_idx))
        });

        let mut remaining = self.budget_bytes;
        let mut selected_ranked = Vec::new();
        let mut spilled_ranked = Vec::new();
        for (idx, candidate) in ranked {
            if candidate.bytes <= remaining {
                remaining -= candidate.bytes;
                selected_ranked.push((idx, candidate));
            } else {
                spilled_ranked.push((idx, candidate));
            }
        }

        selected_ranked.sort_by_key(|(idx, _)| *idx);
        spilled_ranked.sort_by_key(|(idx, _)| *idx);
        ResidencyPlan {
            selected: selected_ranked
                .into_iter()
                .map(|(_, candidate)| candidate)
                .collect(),
            spilled: spilled_ranked
                .into_iter()
                .map(|(_, candidate)| candidate)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidencyTransferStats {
    pub resident_hit_bytes: u64,
    pub resident_upload_bytes: u64,
    pub temp_upload_bytes: u64,
    pub spill_bytes: u64,
}

impl ResidencyTransferStats {
    pub const fn total_h2d_bytes(self) -> u64 {
        self.resident_upload_bytes + self.temp_upload_bytes
    }

    pub const fn total_referenced_bytes(self) -> u64 {
        self.resident_hit_bytes + self.resident_upload_bytes + self.temp_upload_bytes
    }

    pub fn record_resident_hit(&mut self, bytes: u64) {
        self.resident_hit_bytes = self.resident_hit_bytes.saturating_add(bytes);
    }

    pub fn record_resident_upload(&mut self, bytes: u64) {
        self.resident_upload_bytes = self.resident_upload_bytes.saturating_add(bytes);
    }

    pub fn record_temp_upload(&mut self, bytes: u64) {
        self.temp_upload_bytes = self.temp_upload_bytes.saturating_add(bytes);
    }

    pub fn record_spill(&mut self, bytes: u64) {
        self.spill_bytes = self.spill_bytes.saturating_add(bytes);
    }
}

pub fn unique_candidate_bytes(candidates: &[ResidencyCandidate]) -> u64 {
    candidates.iter().map(|candidate| candidate.bytes()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_selects_highest_reuse_candidates_within_budget() {
        let candidates = [
            ResidencyCandidate::new("layer0.shared", 64, 1),
            ResidencyCandidate::new("layer1.expert.hot", 96, 6),
            ResidencyCandidate::new("layer2.expert.warm", 64, 3),
        ];

        let plan = ResidencyPlanner::new(128).plan(&candidates);

        assert_eq!(plan.selected_ids(), vec!["layer1.expert.hot"]);
        assert_eq!(plan.selected_bytes(), 96);
        assert_eq!(plan.spill_bytes(), 128);
    }

    #[test]
    fn planner_keeps_input_order_for_equal_priority() {
        let candidates = [
            ResidencyCandidate::new("a", 64, 2),
            ResidencyCandidate::new("b", 64, 2),
            ResidencyCandidate::new("c", 64, 2),
        ];

        let plan = ResidencyPlanner::new(128).plan(&candidates);

        assert_eq!(plan.selected_ids(), vec!["a", "b"]);
        assert_eq!(plan.selected_bytes(), 128);
        assert_eq!(plan.spill_bytes(), 64);
    }

    #[test]
    fn transfer_stats_separates_resident_hits_from_h2d_uploads() {
        let mut stats = ResidencyTransferStats::default();

        stats.record_resident_hit(64);
        stats.record_resident_upload(32);
        stats.record_temp_upload(96);
        stats.record_spill(16);

        assert_eq!(stats.total_h2d_bytes(), 128);
        assert_eq!(stats.total_referenced_bytes(), 192);
        assert_eq!(stats.spill_bytes, 16);
    }
}
