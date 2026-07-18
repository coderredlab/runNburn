use crate::byte_lru::ByteLruPolicy;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SparseExpertCacheKey {
    layer_index: u32,
    expert_index: u32,
}

impl SparseExpertCacheKey {
    pub const fn new(layer_index: usize, expert_index: usize) -> Self {
        Self {
            layer_index: layer_index as u32,
            expert_index: expert_index as u32,
        }
    }

    pub const fn layer_index(self) -> usize {
        self.layer_index as usize
    }

    pub const fn expert_index(self) -> usize {
        self.expert_index as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBundleFootprint {
    key: SparseExpertCacheKey,
    gate_bytes: u64,
    up_bytes: u64,
    down_bytes: u64,
}

impl ExpertBundleFootprint {
    pub const fn new(
        key: SparseExpertCacheKey,
        gate_bytes: u64,
        up_bytes: u64,
        down_bytes: u64,
    ) -> Self {
        Self {
            key,
            gate_bytes,
            up_bytes,
            down_bytes,
        }
    }

    pub const fn key(self) -> SparseExpertCacheKey {
        self.key
    }

    pub const fn gate_bytes(self) -> u64 {
        self.gate_bytes
    }

    pub const fn up_bytes(self) -> u64 {
        self.up_bytes
    }

    pub const fn down_bytes(self) -> u64 {
        self.down_bytes
    }

    /// Returns the exact bundle payload size, or `None` when the sum exceeds
    /// `u64`.
    pub const fn checked_total_bytes(self) -> Option<u64> {
        let Some(gate_and_up) = self.gate_bytes.checked_add(self.up_bytes) else {
            return None;
        };
        gate_and_up.checked_add(self.down_bytes)
    }

    /// Returns the bundle payload size, saturating at `u64::MAX` on overflow.
    pub const fn total_bytes(self) -> u64 {
        self.gate_bytes
            .saturating_add(self.up_bytes)
            .saturating_add(self.down_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertBundleResidency {
    Full,
    Partial,
    Miss,
}

impl ExpertBundleResidency {
    pub const fn from_roles(gate_resident: bool, up_resident: bool, down_resident: bool) -> Self {
        match (gate_resident, up_resident, down_resident) {
            (true, true, true) => Self::Full,
            (false, false, false) => Self::Miss,
            _ => Self::Partial,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBundleAdmissionCandidate {
    footprint: ExpertBundleFootprint,
    gate_resident: bool,
    up_resident: bool,
    down_resident: bool,
}

impl ExpertBundleAdmissionCandidate {
    pub const fn new(
        footprint: ExpertBundleFootprint,
        gate_resident: bool,
        up_resident: bool,
        down_resident: bool,
    ) -> Self {
        Self {
            footprint,
            gate_resident,
            up_resident,
            down_resident,
        }
    }

    pub const fn footprint(self) -> ExpertBundleFootprint {
        self.footprint
    }

    pub const fn gate_resident(self) -> bool {
        self.gate_resident
    }

    pub const fn up_resident(self) -> bool {
        self.up_resident
    }

    pub const fn down_resident(self) -> bool {
        self.down_resident
    }

    pub const fn residency(self) -> ExpertBundleResidency {
        ExpertBundleResidency::from_roles(self.gate_resident, self.up_resident, self.down_resident)
    }

    pub const fn is_fully_resident(self) -> bool {
        self.gate_resident && self.up_resident && self.down_resident
    }

    /// Returns only the bytes for roles that are not already resident.
    ///
    /// The sum saturates at `u64::MAX`, matching
    /// [`ExpertBundleFootprint::total_bytes`].
    pub const fn missing_admission_bytes(self) -> u64 {
        let gate_bytes = if self.gate_resident {
            0
        } else {
            self.footprint.gate_bytes
        };
        let up_bytes = if self.up_resident {
            0
        } else {
            self.footprint.up_bytes
        };
        let down_bytes = if self.down_resident {
            0
        } else {
            self.footprint.down_bytes
        };
        gate_bytes
            .saturating_add(up_bytes)
            .saturating_add(down_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentLookupTransfer {
    ReplacesTempUpload,
    NoUpload,
}

impl CurrentLookupTransfer {
    const fn saved_transfer_count(self) -> u128 {
        match self {
            Self::ReplacesTempUpload => 1,
            Self::NoUpload => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBundleAdmissionCost {
    pub missing_admission_bytes: u64,
    pub eviction_reload_bytes: u64,
    pub predicted_future_reuses: u64,
    pub predicted_saved_bytes: u128,
    pub net_saved_bytes: i128,
    pub profitable: bool,
}

/// Result of evaluating whether missing roles should become resident.
///
/// A fully resident bundle has `cost.profitable == true` but `admit == false`:
/// it is already in the desired state and needs no additional admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBundleAdmissionDecision {
    /// Whether the caller should admit the candidate's missing roles.
    pub admit: bool,
    /// The complete byte-cost calculation, including the profitability result.
    pub cost: ExpertBundleAdmissionCost,
}

/// Evaluates one expert bundle without knowing how or where its bytes live.
///
/// `predicted_future_reuses` excludes the current lookup. A fused GPU caller
/// can use `ReplacesTempUpload` when admission replaces the temporary upload
/// required by the current lookup. A mixed CPU/GPU planner must use `NoUpload`
/// for CPU misses, because those bytes would not otherwise cross H2D now.
///
/// The admission upload costs `missing_admission_bytes`, plus any
/// `eviction_reload_bytes`. Profitability is strict. Fully resident bundles
/// are profitable by definition, while returning `admit == false` because no
/// work remains.
pub fn evaluate_expert_bundle_admission(
    candidate: ExpertBundleAdmissionCandidate,
    predicted_future_reuses: u64,
    eviction_reload_bytes: u64,
    current_lookup_transfer: CurrentLookupTransfer,
) -> ExpertBundleAdmissionDecision {
    let missing_admission_bytes = candidate.missing_admission_bytes();
    let predicted_saved_bytes = u128::from(missing_admission_bytes)
        * (u128::from(predicted_future_reuses) + current_lookup_transfer.saved_transfer_count());
    let total_cost_bytes = u128::from(missing_admission_bytes) + u128::from(eviction_reload_bytes);
    let net_saved_bytes = saturating_signed_difference(predicted_saved_bytes, total_cost_bytes);
    let profitable = candidate.is_fully_resident() || predicted_saved_bytes > total_cost_bytes;

    ExpertBundleAdmissionDecision {
        admit: !candidate.is_fully_resident() && profitable,
        cost: ExpertBundleAdmissionCost {
            missing_admission_bytes,
            eviction_reload_bytes,
            predicted_future_reuses,
            predicted_saved_bytes,
            net_saved_bytes,
            profitable,
        },
    }
}

fn saturating_signed_difference(left: u128, right: u128) -> i128 {
    if left >= right {
        let difference = left - right;
        difference.min(i128::MAX as u128) as i128
    } else {
        let difference = right - left;
        if difference > i128::MAX as u128 {
            i128::MIN
        } else {
            -(difference as i128)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpertBundleReuseEntry {
    frequency: u64,
    last_seen_epoch: u64,
}

/// Bounded frequency-and-recency history for expert bundle reuse.
///
/// Frequency halves once per `decay_interval` epochs spent unobserved. The
/// default interval is the caller-provided capacity, so the aging horizon
/// scales with the amount of history the caller chose to retain.
pub struct ExpertBundleReuseHistory {
    capacity: usize,
    decay_interval: u64,
    epoch: u64,
    entries: BTreeMap<SparseExpertCacheKey, ExpertBundleReuseEntry>,
}

impl ExpertBundleReuseHistory {
    pub fn new(capacity: usize) -> Self {
        Self::with_decay_interval(capacity, Self::capacity_decay_interval(capacity))
    }

    pub fn with_decay_interval(capacity: usize, decay_interval: u64) -> Self {
        Self {
            capacity,
            decay_interval: decay_interval.max(1),
            epoch: 0,
            entries: BTreeMap::new(),
        }
    }

    /// Resizes the bounded history without discarding retained hot entries.
    ///
    /// Shrinking immediately applies the same frequency, recency, and key
    /// victim order used during observation. The decay interval returns to
    /// the capacity-proportional default; callers that need another interval
    /// can construct a replacement with [`Self::with_decay_interval`].
    pub fn set_capacity(&mut self, capacity: usize) {
        while self.entries.len() > capacity {
            self.evict_lowest_priority();
        }
        self.capacity = capacity;
        self.decay_interval = Self::capacity_decay_interval(capacity);
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn decay_interval(&self) -> u64 {
        self.decay_interval
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: SparseExpertCacheKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Records one lookup and returns its decayed reuse frequency before this
    /// observation is included.
    pub fn observe(&mut self, key: SparseExpertCacheKey) -> u64 {
        self.epoch = self.epoch.saturating_add(1);
        if self.capacity == 0 {
            return 0;
        }

        let prior_reuse_score = self
            .entries
            .get(&key)
            .map(|entry| self.effective_frequency(*entry))
            .unwrap_or(0);
        self.entries.insert(
            key,
            ExpertBundleReuseEntry {
                frequency: prior_reuse_score.saturating_add(1),
                last_seen_epoch: self.epoch,
            },
        );

        if self.entries.len() > self.capacity {
            self.evict_lowest_priority();
        }

        prior_reuse_score
    }

    fn evict_lowest_priority(&mut self) {
        let victim = self
            .entries
            .iter()
            .map(|(&candidate_key, &entry)| {
                (
                    self.effective_frequency(entry),
                    entry.last_seen_epoch,
                    candidate_key,
                )
            })
            .min()
            .map(|(_, _, candidate_key)| candidate_key)
            .expect("cannot evict from an empty reuse history");
        self.entries.remove(&victim);
    }

    fn capacity_decay_interval(capacity: usize) -> u64 {
        u64::try_from(capacity).unwrap_or(u64::MAX).max(1)
    }

    fn effective_frequency(&self, entry: ExpertBundleReuseEntry) -> u64 {
        let unobserved_epochs = self
            .epoch
            .saturating_sub(entry.last_seen_epoch)
            .saturating_sub(1);
        let decay_periods = unobserved_epochs / self.decay_interval;
        if decay_periods >= u64::from(u64::BITS) {
            0
        } else {
            entry.frequency >> decay_periods
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertBundleCacheStats {
    pub bundle_lookups: u64,
    pub bundle_hits: u64,
    pub bundle_partial_hits: u64,
    pub bundle_misses: u64,
    pub bundle_admissions: u64,
    pub bundle_evictions: u64,
    pub admitted_bytes: u64,
    pub evicted_bytes: u64,
    pub h2d_bytes: u64,
    pub temp_h2d_bytes: u64,
}

impl ExpertBundleCacheStats {
    pub fn record_lookup(&mut self, residency: ExpertBundleResidency) {
        self.bundle_lookups = self.bundle_lookups.saturating_add(1);
        match residency {
            ExpertBundleResidency::Full => {
                self.bundle_hits = self.bundle_hits.saturating_add(1);
            }
            ExpertBundleResidency::Partial => {
                self.bundle_partial_hits = self.bundle_partial_hits.saturating_add(1);
            }
            ExpertBundleResidency::Miss => {
                self.bundle_misses = self.bundle_misses.saturating_add(1);
            }
        }
    }

    pub const fn delta(self, before: Self) -> Self {
        Self {
            bundle_lookups: self.bundle_lookups.saturating_sub(before.bundle_lookups),
            bundle_hits: self.bundle_hits.saturating_sub(before.bundle_hits),
            bundle_partial_hits: self
                .bundle_partial_hits
                .saturating_sub(before.bundle_partial_hits),
            bundle_misses: self.bundle_misses.saturating_sub(before.bundle_misses),
            bundle_admissions: self
                .bundle_admissions
                .saturating_sub(before.bundle_admissions),
            bundle_evictions: self
                .bundle_evictions
                .saturating_sub(before.bundle_evictions),
            admitted_bytes: self.admitted_bytes.saturating_sub(before.admitted_bytes),
            evicted_bytes: self.evicted_bytes.saturating_sub(before.evicted_bytes),
            h2d_bytes: self.h2d_bytes.saturating_sub(before.h2d_bytes),
            temp_h2d_bytes: self.temp_h2d_bytes.saturating_sub(before.temp_h2d_bytes),
        }
    }

    pub fn hit_rate(self) -> f64 {
        if self.bundle_lookups == 0 {
            0.0
        } else {
            self.bundle_hits as f64 / self.bundle_lookups as f64
        }
    }

    pub fn h2d_bytes_per_token(self, token_count: u64) -> f64 {
        if token_count == 0 {
            0.0
        } else {
            self.h2d_bytes as f64 / token_count as f64
        }
    }
    pub fn saturating_add_assign(&mut self, delta: Self) {
        self.bundle_lookups = self.bundle_lookups.saturating_add(delta.bundle_lookups);
        self.bundle_hits = self.bundle_hits.saturating_add(delta.bundle_hits);
        self.bundle_partial_hits = self
            .bundle_partial_hits
            .saturating_add(delta.bundle_partial_hits);
        self.bundle_misses = self.bundle_misses.saturating_add(delta.bundle_misses);
        self.bundle_admissions = self
            .bundle_admissions
            .saturating_add(delta.bundle_admissions);
        self.bundle_evictions = self.bundle_evictions.saturating_add(delta.bundle_evictions);
        self.admitted_bytes = self.admitted_bytes.saturating_add(delta.admitted_bytes);
        self.evicted_bytes = self.evicted_bytes.saturating_add(delta.evicted_bytes);
        self.h2d_bytes = self.h2d_bytes.saturating_add(delta.h2d_bytes);
        self.temp_h2d_bytes = self.temp_h2d_bytes.saturating_add(delta.temp_h2d_bytes);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertBundleObservationReceipt {
    consumed: bool,
    pending_stats: ExpertBundleCacheStats,
}

impl ExpertBundleObservationReceipt {
    pub const fn consumed(self) -> bool {
        self.consumed
    }

    pub fn mark_consumed(&mut self) {
        self.consumed = true;
    }

    pub const fn pending_stats(self) -> ExpertBundleCacheStats {
        self.pending_stats
    }

    pub fn record_stats(&mut self, stats: ExpertBundleCacheStats) {
        self.pending_stats.saturating_add_assign(stats);
    }

    pub fn clear_stats(&mut self) {
        self.pending_stats = ExpertBundleCacheStats::default();
    }
}

/// Byte-budgeted LRU policy for file-backed sparse experts.
///
/// It owns only accounting. The caller owns mappings and reclaims the keys
/// returned by `touch` after releasing this policy from its lock.
pub struct SparseExpertCacheBudget {
    policy: ByteLruPolicy<SparseExpertCacheKey>,
}

impl SparseExpertCacheBudget {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            policy: ByteLruPolicy::new(max_bytes),
        }
    }

    pub const fn max_bytes(&self) -> u64 {
        self.policy.max_bytes()
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.policy.resident_bytes()
    }

    pub fn touch(&mut self, key: SparseExpertCacheKey, bytes: u64) -> Vec<SparseExpertCacheKey> {
        self.policy.touch(key, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_and_records_one_bundle_outcome_per_lookup() {
        assert_eq!(
            ExpertBundleResidency::from_roles(true, true, true),
            ExpertBundleResidency::Full
        );
        assert_eq!(
            ExpertBundleResidency::from_roles(false, false, false),
            ExpertBundleResidency::Miss
        );
        for roles in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, true, false),
            (true, false, true),
            (false, true, true),
        ] {
            assert_eq!(
                ExpertBundleResidency::from_roles(roles.0, roles.1, roles.2),
                ExpertBundleResidency::Partial
            );
        }

        let mut stats = ExpertBundleCacheStats::default();
        stats.record_lookup(ExpertBundleResidency::Full);
        stats.record_lookup(ExpertBundleResidency::Partial);
        stats.record_lookup(ExpertBundleResidency::Miss);

        assert_eq!(stats.bundle_lookups, 3);
        assert_eq!(stats.bundle_hits, 1);
        assert_eq!(stats.bundle_partial_hits, 1);
        assert_eq!(stats.bundle_misses, 1);
    }

    #[test]
    fn bundle_footprint_total_never_wraps() {
        let key = SparseExpertCacheKey::new(2, 5);
        let exact = ExpertBundleFootprint::new(key, 10, 20, 30);
        assert_eq!(exact.key(), key);
        assert_eq!(exact.gate_bytes(), 10);
        assert_eq!(exact.up_bytes(), 20);
        assert_eq!(exact.down_bytes(), 30);
        assert_eq!(exact.checked_total_bytes(), Some(60));
        assert_eq!(exact.total_bytes(), 60);

        let overflow = ExpertBundleFootprint::new(key, u64::MAX, 1, 1);
        assert_eq!(overflow.checked_total_bytes(), None);
        assert_eq!(overflow.total_bytes(), u64::MAX);
    }

    #[test]
    fn bundle_stats_delta_saturates_every_field() {
        let before = ExpertBundleCacheStats {
            bundle_lookups: 1,
            bundle_hits: 2,
            bundle_partial_hits: 3,
            bundle_misses: 4,
            bundle_admissions: 5,
            bundle_evictions: 6,
            admitted_bytes: 8,
            evicted_bytes: 9,
            h2d_bytes: 10,
            temp_h2d_bytes: 11,
        };
        let after = ExpertBundleCacheStats {
            bundle_lookups: 11,
            bundle_hits: 1,
            bundle_partial_hits: 13,
            bundle_misses: 2,
            bundle_admissions: 15,
            bundle_evictions: 3,
            admitted_bytes: 4,
            evicted_bytes: 19,
            h2d_bytes: 5,
            temp_h2d_bytes: 21,
        };

        assert_eq!(
            after.delta(before),
            ExpertBundleCacheStats {
                bundle_lookups: 10,
                bundle_hits: 0,
                bundle_partial_hits: 10,
                bundle_misses: 0,
                bundle_admissions: 10,
                bundle_evictions: 0,
                admitted_bytes: 0,
                evicted_bytes: 10,
                h2d_bytes: 0,
                temp_h2d_bytes: 10,
            }
        );
    }

    #[test]
    fn bundle_stats_rates_are_zero_without_a_denominator() {
        let stats = ExpertBundleCacheStats {
            bundle_lookups: 4,
            bundle_hits: 2,
            h2d_bytes: 100,
            ..ExpertBundleCacheStats::default()
        };

        assert_eq!(ExpertBundleCacheStats::default().hit_rate(), 0.0);
        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.h2d_bytes_per_token(0), 0.0);
        assert_eq!(stats.h2d_bytes_per_token(4), 25.0);
    }

    #[test]
    fn bundle_observation_receipt_accumulates_and_clears_stats_without_reopening() {
        let mut receipt = ExpertBundleObservationReceipt::default();
        assert!(!receipt.consumed());
        assert_eq!(receipt.pending_stats(), ExpertBundleCacheStats::default());

        receipt.record_stats(ExpertBundleCacheStats {
            bundle_lookups: 1,
            bundle_partial_hits: 1,
            h2d_bytes: u64::MAX,
            ..ExpertBundleCacheStats::default()
        });
        receipt.record_stats(ExpertBundleCacheStats {
            bundle_lookups: 2,
            bundle_admissions: 1,
            h2d_bytes: 1,
            ..ExpertBundleCacheStats::default()
        });
        receipt.mark_consumed();

        assert!(receipt.consumed());
        assert_eq!(
            receipt.pending_stats(),
            ExpertBundleCacheStats {
                bundle_lookups: 3,
                bundle_partial_hits: 1,
                bundle_admissions: 1,
                h2d_bytes: u64::MAX,
                ..ExpertBundleCacheStats::default()
            }
        );

        receipt.clear_stats();
        assert!(receipt.consumed());
        assert_eq!(receipt.pending_stats(), ExpertBundleCacheStats::default());
    }

    #[test]
    fn cold_seen_once_and_reused_follow_current_transfer_break_even() {
        let key = SparseExpertCacheKey::new(3, 7);
        let footprint = ExpertBundleFootprint::new(key, 20, 30, 50);
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, false, false, false);
        let mut history = ExpertBundleReuseHistory::new(4);

        let cold = evaluate_expert_bundle_admission(
            candidate,
            history.observe(key),
            0,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(cold.cost.predicted_future_reuses, 0);
        assert_eq!(cold.cost.predicted_saved_bytes, 100);
        assert_eq!(cold.cost.net_saved_bytes, 0);
        assert!(!cold.admit);

        let seen_once = evaluate_expert_bundle_admission(
            candidate,
            history.observe(key),
            0,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(seen_once.cost.predicted_future_reuses, 1);
        assert_eq!(seen_once.cost.predicted_saved_bytes, 200);
        assert_eq!(seen_once.cost.net_saved_bytes, 100);
        assert!(seen_once.admit);

        let reused = evaluate_expert_bundle_admission(
            candidate,
            history.observe(key),
            0,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(reused.cost.predicted_future_reuses, 2);
        assert_eq!(reused.cost.predicted_saved_bytes, 300);
        assert_eq!(reused.cost.net_saved_bytes, 200);
        assert!(reused.admit);
    }

    #[test]
    fn no_current_upload_requires_more_than_one_future_reuse() {
        let footprint = ExpertBundleFootprint::new(SparseExpertCacheKey::new(3, 8), 20, 30, 50);
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, false, false, false);

        let first_reuse =
            evaluate_expert_bundle_admission(candidate, 1, 0, CurrentLookupTransfer::NoUpload);
        assert_eq!(first_reuse.cost.predicted_saved_bytes, 100);
        assert_eq!(first_reuse.cost.net_saved_bytes, 0);
        assert!(!first_reuse.admit);

        let second_reuse =
            evaluate_expert_bundle_admission(candidate, 2, 0, CurrentLookupTransfer::NoUpload);
        assert_eq!(second_reuse.cost.predicted_saved_bytes, 200);
        assert_eq!(second_reuse.cost.net_saved_bytes, 100);
        assert!(second_reuse.admit);
    }

    #[test]
    fn eviction_reload_cost_uses_strict_break_even() {
        let footprint = ExpertBundleFootprint::new(SparseExpertCacheKey::new(0, 1), 25, 25, 50);
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, false, false, false);

        let just_below = evaluate_expert_bundle_admission(
            candidate,
            3,
            299,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(just_below.cost.predicted_saved_bytes, 400);
        assert_eq!(just_below.cost.net_saved_bytes, 1);
        assert!(just_below.admit);

        let exact_boundary = evaluate_expert_bundle_admission(
            candidate,
            3,
            300,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(exact_boundary.cost.predicted_saved_bytes, 400);
        assert_eq!(exact_boundary.cost.net_saved_bytes, 0);
        assert!(!exact_boundary.admit);

        let just_above = evaluate_expert_bundle_admission(
            candidate,
            3,
            301,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(just_above.cost.net_saved_bytes, -1);
        assert!(!just_above.admit);
    }

    #[test]
    fn partial_bundle_charges_and_saves_only_missing_roles() {
        let footprint = ExpertBundleFootprint::new(SparseExpertCacheKey::new(1, 2), 10, 20, 30);
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, true, false, true);

        assert_eq!(candidate.residency(), ExpertBundleResidency::Partial);
        assert_eq!(candidate.missing_admission_bytes(), 20);

        let cold = evaluate_expert_bundle_admission(
            candidate,
            0,
            0,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(cold.cost.missing_admission_bytes, 20);
        assert_eq!(cold.cost.predicted_saved_bytes, 20);
        assert_eq!(cold.cost.net_saved_bytes, 0);
        assert!(!cold.admit);

        let first_reuse = evaluate_expert_bundle_admission(
            candidate,
            1,
            0,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(first_reuse.cost.predicted_saved_bytes, 40);
        assert_eq!(first_reuse.cost.net_saved_bytes, 20);
        assert!(first_reuse.admit);

        let exact_boundary = evaluate_expert_bundle_admission(
            candidate,
            1,
            20,
            CurrentLookupTransfer::ReplacesTempUpload,
        );
        assert_eq!(exact_boundary.cost.predicted_saved_bytes, 40);
        assert_eq!(exact_boundary.cost.net_saved_bytes, 0);
        assert!(!exact_boundary.admit);
    }

    #[test]
    fn already_full_bundle_is_profitable_without_admission() {
        let footprint = ExpertBundleFootprint::new(SparseExpertCacheKey::new(2, 4), 10, 20, 30);
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, true, true, true);
        let decision = evaluate_expert_bundle_admission(
            candidate,
            u64::MAX,
            500,
            CurrentLookupTransfer::NoUpload,
        );

        assert_eq!(candidate.residency(), ExpertBundleResidency::Full);
        assert_eq!(decision.cost.missing_admission_bytes, 0);
        assert_eq!(decision.cost.predicted_saved_bytes, 0);
        assert!(decision.cost.profitable);
        assert!(!decision.admit);
    }

    #[test]
    fn admission_cost_uses_wide_and_saturating_arithmetic() {
        let footprint = ExpertBundleFootprint::new(
            SparseExpertCacheKey::new(5, 6),
            u64::MAX,
            u64::MAX,
            u64::MAX,
        );
        let candidate = ExpertBundleAdmissionCandidate::new(footprint, false, false, false);
        let decision = evaluate_expert_bundle_admission(
            candidate,
            u64::MAX,
            u64::MAX,
            CurrentLookupTransfer::ReplacesTempUpload,
        );

        assert_eq!(candidate.missing_admission_bytes(), u64::MAX);
        assert_eq!(decision.cost.predicted_future_reuses, u64::MAX);
        assert_eq!(
            decision.cost.predicted_saved_bytes,
            u128::from(u64::MAX) * (u128::from(u64::MAX) + 1)
        );
        assert_eq!(decision.cost.net_saved_bytes, i128::MAX);
        assert!(decision.admit);
    }

    #[test]
    fn zero_capacity_history_never_retains_or_predicts_reuse() {
        let key = SparseExpertCacheKey::new(7, 8);
        let mut history = ExpertBundleReuseHistory::new(0);

        assert_eq!(history.observe(key), 0);
        assert_eq!(history.observe(key), 0);
        assert_eq!(history.capacity(), 0);
        assert_eq!(history.len(), 0);
        assert!(!history.contains(key));
        assert_eq!(history.epoch(), 2);
    }

    #[test]
    fn resizing_history_preserves_hot_entries_and_reapplies_capacity_decay() {
        let hot = SparseExpertCacheKey::new(0, 1);
        let colder = SparseExpertCacheKey::new(0, 2);
        let newest = SparseExpertCacheKey::new(0, 3);
        let mut history = ExpertBundleReuseHistory::with_decay_interval(3, 100);
        history.observe(hot);
        history.observe(hot);
        history.observe(colder);
        history.observe(newest);

        history.set_capacity(2);
        assert_eq!(history.capacity(), 2);
        assert_eq!(history.decay_interval(), 2);
        assert!(history.contains(hot));
        assert!(!history.contains(colder));
        assert!(history.contains(newest));

        history.set_capacity(1);
        assert!(history.contains(hot));
        assert!(!history.contains(newest));

        history.set_capacity(4);
        assert_eq!(history.observe(hot), 2);
        assert_eq!(history.capacity(), 4);
        assert_eq!(history.decay_interval(), 4);

        history.set_capacity(0);
        assert!(history.is_empty());
        assert_eq!(history.decay_interval(), 1);
    }

    #[test]
    fn history_eviction_uses_frequency_then_recency() {
        let a = SparseExpertCacheKey::new(0, 1);
        let b = SparseExpertCacheKey::new(0, 2);
        let c = SparseExpertCacheKey::new(0, 3);

        let mut by_recency = ExpertBundleReuseHistory::with_decay_interval(2, 100);
        by_recency.observe(a);
        by_recency.observe(b);
        by_recency.observe(c);
        assert!(!by_recency.contains(a));
        assert!(by_recency.contains(b));
        assert!(by_recency.contains(c));

        let mut by_frequency = ExpertBundleReuseHistory::with_decay_interval(2, 100);
        by_frequency.observe(a);
        by_frequency.observe(a);
        by_frequency.observe(b);
        by_frequency.observe(c);
        assert!(by_frequency.contains(a));
        assert!(!by_frequency.contains(b));
        assert!(by_frequency.contains(c));
    }

    #[test]
    fn history_key_tie_break_is_deterministic_at_saturated_epoch() {
        let lower = SparseExpertCacheKey::new(0, 1);
        let higher = SparseExpertCacheKey::new(0, 2);

        let mut first = ExpertBundleReuseHistory::with_decay_interval(1, 100);
        first.epoch = u64::MAX;
        first.observe(lower);
        first.observe(higher);

        let mut second = ExpertBundleReuseHistory::with_decay_interval(1, 100);
        second.epoch = u64::MAX;
        second.observe(higher);
        second.observe(lower);

        assert!(!first.contains(lower));
        assert!(first.contains(higher));
        assert!(!second.contains(lower));
        assert!(second.contains(higher));
    }

    #[test]
    fn evicts_least_recently_used_expert_within_byte_budget() {
        let a = SparseExpertCacheKey::new(0, 1);
        let b = SparseExpertCacheKey::new(0, 2);
        let c = SparseExpertCacheKey::new(1, 3);
        let mut cache = SparseExpertCacheBudget::new(20);

        assert!(cache.touch(a, 10).is_empty());
        assert!(cache.touch(b, 10).is_empty());
        assert!(cache.touch(a, 10).is_empty());
        assert_eq!(cache.touch(c, 10), vec![b]);
        assert_eq!(cache.resident_bytes(), 20);
    }

    #[test]
    fn zero_budget_reclaims_touched_expert_immediately() {
        let key = SparseExpertCacheKey::new(4, 7);
        let mut cache = SparseExpertCacheBudget::new(0);

        assert_eq!(cache.touch(key, 42), vec![key]);
        assert_eq!(cache.resident_bytes(), 0);
    }
}
