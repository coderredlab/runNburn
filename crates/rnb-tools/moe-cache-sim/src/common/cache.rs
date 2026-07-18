use std::collections::{HashMap, HashSet, VecDeque};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    Lru,
    StaticHot,
    StaticHotLru,
    StaticHotAdaptive,
    LayerQuotaStatic,
    RankWeightedStatic,
    GateUpStatic,
    LeastStale,
    RouterCurrentJit,
    PrevStepPrefetch,
    OnlineLayerHotPrefetch,
    OnlineLayerHotStaged,
    AdaptiveLfuLru,
    WindowLfuLru,
}

impl FromStr for CachePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lru" => Ok(Self::Lru),
            "static-hot" | "static_hot" => Ok(Self::StaticHot),
            "static-hot-lru" | "static_hot_lru" | "lru-static-hot" | "lru_static_hot" => {
                Ok(Self::StaticHotLru)
            }
            "static-hot-adaptive"
            | "static_hot_adaptive"
            | "adaptive-static-hot"
            | "adaptive_static_hot" => Ok(Self::StaticHotAdaptive),
            "layer-quota-static" | "layer_quota_static" => Ok(Self::LayerQuotaStatic),
            "rank-weighted-static" | "rank_weighted_static" => Ok(Self::RankWeightedStatic),
            "gate-up-static" | "gate_up_static" => Ok(Self::GateUpStatic),
            "least-stale" | "least_stale" => Ok(Self::LeastStale),
            "router-current-jit" | "router_current_jit" => Ok(Self::RouterCurrentJit),
            "prev-step-prefetch" | "prev_step_prefetch" => Ok(Self::PrevStepPrefetch),
            "online-layer-hot-prefetch" | "online_layer_hot_prefetch" => {
                Ok(Self::OnlineLayerHotPrefetch)
            }
            "online-layer-hot-staged" | "online_layer_hot_staged" => Ok(Self::OnlineLayerHotStaged),
            "adaptive-lfu-lru" | "adaptive_lfu_lru" => Ok(Self::AdaptiveLfuLru),
            "window-lfu-lru" | "window_lfu_lru" => Ok(Self::WindowLfuLru),
            other => Err(format!("unknown cache policy: {other}")),
        }
    }
}

pub struct ExpertCache {
    capacity: u64,
    used_dynamic: u64,
    dynamic: HashMap<ExpertKey, u64>,
    lru: VecDeque<ExpertKey>,
    static_hot: HashMap<ExpertKey, u64>,
    static_used: u64,
}

impl ExpertCache {
    pub fn new_lru(capacity: u64) -> Self {
        Self {
            capacity,
            used_dynamic: 0,
            dynamic: HashMap::new(),
            lru: VecDeque::new(),
            static_hot: HashMap::new(),
            static_used: 0,
        }
    }

    pub fn new_static_hot(capacity: u64, static_hot: HashMap<ExpertKey, u64>) -> Self {
        let static_used = static_hot.values().copied().sum();
        Self {
            capacity,
            used_dynamic: 0,
            dynamic: HashMap::new(),
            lru: VecDeque::new(),
            static_hot,
            static_used,
        }
    }

    pub fn new_static_hot_lru(capacity: u64, static_hot: HashMap<ExpertKey, u64>) -> Self {
        Self::new_static_hot(capacity, static_hot)
    }

    pub fn access(&mut self, key: ExpertKey, bytes: u64) -> bool {
        if self.static_hot.contains_key(&key) {
            return true;
        }
        if self.capacity <= self.static_used {
            return false;
        }
        if self.dynamic.contains_key(&key) {
            self.touch(key);
            return true;
        }
        let dynamic_capacity = self.capacity - self.static_used;
        if bytes > dynamic_capacity {
            return false;
        }
        while self.used_dynamic.saturating_add(bytes) > dynamic_capacity {
            let Some(old) = self.lru.pop_front() else {
                break;
            };
            if let Some(old_bytes) = self.dynamic.remove(&old) {
                self.used_dynamic = self.used_dynamic.saturating_sub(old_bytes);
            }
        }
        self.dynamic.insert(key, bytes);
        self.lru.push_back(key);
        self.used_dynamic = self.used_dynamic.saturating_add(bytes);
        false
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.static_hot.contains_key(&key) || self.dynamic.contains_key(&key)
    }

    pub fn remove_dynamic(&mut self, key: ExpertKey) -> Option<u64> {
        let removed = self.dynamic.remove(&key)?;
        self.used_dynamic = self.used_dynamic.saturating_sub(removed);
        self.lru.retain(|candidate| *candidate != key);
        Some(removed)
    }

    pub fn resident_entries(&self) -> usize {
        self.static_hot.len() + self.dynamic.len()
    }

    fn touch(&mut self, key: ExpertKey) {
        self.lru.retain(|candidate| *candidate != key);
        self.lru.push_back(key);
    }
}

pub struct LeastStaleExpertCache {
    capacity: u64,
    used: u64,
    current_step: usize,
    current_layer: usize,
    clock: u64,
    resident: HashMap<ExpertKey, LeastStaleEntry>,
}

#[derive(Debug, Clone, Copy)]
struct LeastStaleEntry {
    bytes: u64,
    last_step: usize,
    last_layer: usize,
    last_seen: u64,
}

impl LeastStaleExpertCache {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            current_step: 0,
            current_layer: 0,
            clock: 0,
            resident: HashMap::new(),
        }
    }

    pub fn begin_layer(&mut self, step: usize, layer: usize, selected: &[ExpertKey]) {
        self.current_step = step;
        self.current_layer = layer;
        for key in selected {
            if let Some(entry) = self.resident.get_mut(key) {
                entry.last_step = step;
                entry.last_layer = layer;
            }
        }
    }

    pub fn access(&mut self, key: ExpertKey, bytes: u64) -> bool {
        self.clock = self.clock.saturating_add(1);
        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_step = self.current_step;
            entry.last_layer = self.current_layer;
            entry.last_seen = self.clock;
            return true;
        }

        if bytes > self.capacity {
            return false;
        }

        while self.used.saturating_add(bytes) > self.capacity {
            let Some((victim, victim_entry)) = self.highest_eviction_priority() else {
                break;
            };
            self.resident.remove(&victim);
            self.used = self.used.saturating_sub(victim_entry.bytes);
        }

        if self.used.saturating_add(bytes) > self.capacity {
            return false;
        }

        self.resident.insert(
            key,
            LeastStaleEntry {
                bytes,
                last_step: self.current_step,
                last_layer: self.current_layer,
                last_seen: self.clock,
            },
        );
        self.used = self.used.saturating_add(bytes);
        false
    }

    pub fn prefetch(&mut self, key: ExpertKey, bytes: u64, step: usize, layer: usize) {
        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_step = step;
            entry.last_layer = layer;
            return;
        }

        if bytes > self.capacity {
            return;
        }

        while self.used.saturating_add(bytes) > self.capacity {
            let Some((victim, victim_entry)) = self.highest_eviction_priority() else {
                break;
            };
            self.resident.remove(&victim);
            self.used = self.used.saturating_sub(victim_entry.bytes);
        }

        if self.used.saturating_add(bytes) > self.capacity {
            return;
        }

        self.resident.insert(
            key,
            LeastStaleEntry {
                bytes,
                last_step: step,
                last_layer: layer,
                last_seen: self.clock,
            },
        );
        self.used = self.used.saturating_add(bytes);
    }

    pub fn prefetch_if_space(&mut self, key: ExpertKey, bytes: u64, step: usize, layer: usize) {
        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_step = step;
            entry.last_layer = layer;
            return;
        }

        if bytes > self.capacity || self.used.saturating_add(bytes) > self.capacity {
            return;
        }

        self.resident.insert(
            key,
            LeastStaleEntry {
                bytes,
                last_step: step,
                last_layer: layer,
                last_seen: self.clock,
            },
        );
        self.used = self.used.saturating_add(bytes);
    }

    pub fn resident_entries(&self) -> usize {
        self.resident.len()
    }

    fn highest_eviction_priority(&self) -> Option<(ExpertKey, LeastStaleEntry)> {
        self.resident
            .iter()
            .map(|(&key, &entry)| (key, entry))
            .max_by(|(left_key, left), (right_key, right)| {
                self.eviction_priority(*left_key, *left)
                    .cmp(&self.eviction_priority(*right_key, *right))
                    .then_with(|| right.last_seen.cmp(&left.last_seen))
                    .then_with(|| right_key.layer.cmp(&left_key.layer))
                    .then_with(|| right_key.expert_id.cmp(&left_key.expert_id))
            })
    }

    fn eviction_priority(&self, key: ExpertKey, entry: LeastStaleEntry) -> (u8, usize) {
        let is_current = entry.last_step == self.current_step;
        let is_left = key.layer < self.current_layer;
        match (is_current, is_left) {
            (false, true) => (3, self.current_layer.saturating_sub(key.layer)),
            (false, false) => (2, key.layer.saturating_sub(self.current_layer)),
            (true, true) => (1, self.current_layer.saturating_sub(key.layer)),
            (true, false) => (0, key.layer.saturating_sub(self.current_layer)),
        }
    }
}

pub struct AdaptiveExpertCache {
    capacity: u64,
    used: u64,
    clock: u64,
    counts: HashMap<ExpertKey, u64>,
    resident: HashMap<ExpertKey, AdaptiveEntry>,
}

#[derive(Debug, Clone, Copy)]
struct AdaptiveEntry {
    bytes: u64,
    count: u64,
    last_seen: u64,
}

impl AdaptiveExpertCache {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            clock: 0,
            counts: HashMap::new(),
            resident: HashMap::new(),
        }
    }

    pub fn access(&mut self, key: ExpertKey, bytes: u64) -> bool {
        self.clock = self.clock.saturating_add(1);
        let count = self.counts.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        let count = *count;

        if let Some(entry) = self.resident.get_mut(&key) {
            entry.count = count;
            entry.last_seen = self.clock;
            return true;
        }

        if bytes > self.capacity {
            return false;
        }

        let candidate = AdaptiveEntry {
            bytes,
            count,
            last_seen: self.clock,
        };
        while self.used.saturating_add(bytes) > self.capacity {
            let Some((victim, victim_entry)) = self.lowest_scored_entry() else {
                break;
            };
            if adaptive_score(victim_entry) > adaptive_score(candidate) {
                return false;
            }
            self.resident.remove(&victim);
            self.used = self.used.saturating_sub(victim_entry.bytes);
        }

        self.resident.insert(key, candidate);
        self.used = self.used.saturating_add(bytes);
        false
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.resident.contains_key(&key)
    }

    pub fn resident_entries(&self) -> usize {
        self.resident.len()
    }

    fn lowest_scored_entry(&self) -> Option<(ExpertKey, AdaptiveEntry)> {
        self.resident
            .iter()
            .map(|(&key, &entry)| (key, entry))
            .min_by(|(left_key, left), (right_key, right)| {
                adaptive_score(*left)
                    .cmp(&adaptive_score(*right))
                    .then_with(|| right.last_seen.cmp(&left.last_seen))
                    .then_with(|| left_key.layer.cmp(&right_key.layer))
                    .then_with(|| left_key.expert_id.cmp(&right_key.expert_id))
            })
    }
}

fn adaptive_score(entry: AdaptiveEntry) -> u64 {
    entry
        .count
        .saturating_mul(1_000_000)
        .saturating_add(entry.last_seen)
}

pub struct WindowExpertCache {
    capacity: u64,
    used: u64,
    clock: u64,
    window_len: usize,
    history: VecDeque<ExpertKey>,
    window_counts: HashMap<ExpertKey, u64>,
    resident: HashMap<ExpertKey, WindowEntry>,
}

#[derive(Debug, Clone, Copy)]
struct WindowEntry {
    bytes: u64,
    last_seen: u64,
}

impl WindowExpertCache {
    pub fn new(capacity: u64, window_len: usize) -> Self {
        Self {
            capacity,
            used: 0,
            clock: 0,
            window_len: window_len.max(1),
            history: VecDeque::new(),
            window_counts: HashMap::new(),
            resident: HashMap::new(),
        }
    }

    pub fn access(&mut self, key: ExpertKey, bytes: u64) -> bool {
        self.clock = self.clock.saturating_add(1);
        self.observe(key);

        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_seen = self.clock;
            return true;
        }

        if bytes > self.capacity {
            return false;
        }

        let candidate = WindowEntry {
            bytes,
            last_seen: self.clock,
        };
        while self.used.saturating_add(bytes) > self.capacity {
            let Some((victim, victim_entry)) = self.lowest_scored_entry() else {
                break;
            };
            if self.score(victim, victim_entry) > self.score(key, candidate) {
                return false;
            }
            self.resident.remove(&victim);
            self.used = self.used.saturating_sub(victim_entry.bytes);
        }

        self.resident.insert(key, candidate);
        self.used = self.used.saturating_add(bytes);
        false
    }

    pub fn resident_entries(&self) -> usize {
        self.resident.len()
    }

    fn observe(&mut self, key: ExpertKey) {
        self.history.push_back(key);
        *self.window_counts.entry(key).or_insert(0) += 1;
        while self.history.len() > self.window_len {
            let Some(old) = self.history.pop_front() else {
                break;
            };
            if let Some(count) = self.window_counts.get_mut(&old) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.window_counts.remove(&old);
                }
            }
        }
    }

    fn lowest_scored_entry(&self) -> Option<(ExpertKey, WindowEntry)> {
        self.resident
            .iter()
            .map(|(&key, &entry)| (key, entry))
            .min_by(|(left_key, left), (right_key, right)| {
                self.score(*left_key, *left)
                    .cmp(&self.score(*right_key, *right))
                    .then_with(|| right.last_seen.cmp(&left.last_seen))
                    .then_with(|| left_key.layer.cmp(&right_key.layer))
                    .then_with(|| left_key.expert_id.cmp(&right_key.expert_id))
            })
    }

    fn score(&self, key: ExpertKey, entry: WindowEntry) -> u64 {
        self.window_counts
            .get(&key)
            .copied()
            .unwrap_or(0)
            .saturating_mul(1_000_000)
            .saturating_add(entry.last_seen)
    }
}

pub fn select_static_hot<F>(
    events: &[ExpertKey],
    mut size_of: F,
    capacity: u64,
) -> HashMap<ExpertKey, u64>
where
    F: FnMut(ExpertKey) -> u64,
{
    let mut counts = HashMap::<ExpertKey, u64>::new();
    for &event in events {
        *counts.entry(event).or_insert(0) += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(ka, ca), (kb, cb)| {
        cb.cmp(ca)
            .then_with(|| ka.layer.cmp(&kb.layer))
            .then_with(|| ka.expert_id.cmp(&kb.expert_id))
    });

    let mut out = HashMap::new();
    let mut used = 0u64;
    let mut seen = HashSet::new();
    for (key, _count) in ranked {
        if !seen.insert(key) {
            continue;
        }
        let bytes = size_of(key);
        if used.saturating_add(bytes) <= capacity {
            out.insert(key, bytes);
            used += bytes;
        }
    }
    out
}

pub fn select_layer_quota_hot<F>(
    events: &[ExpertKey],
    mut size_of: F,
    capacity: u64,
    num_layers: usize,
) -> HashMap<ExpertKey, u64>
where
    F: FnMut(ExpertKey) -> u64,
{
    if num_layers == 0 {
        return HashMap::new();
    }

    let mut counts_by_layer = HashMap::<usize, HashMap<ExpertKey, u64>>::new();
    for &event in events {
        *counts_by_layer
            .entry(event.layer)
            .or_default()
            .entry(event)
            .or_insert(0) += 1;
    }

    let layer_budget = capacity / num_layers as u64;
    let mut out = HashMap::new();
    let mut used = 0u64;

    for layer in 0..num_layers {
        let Some(counts) = counts_by_layer.remove(&layer) else {
            continue;
        };
        let mut ranked = counts.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|(ka, ca), (kb, cb)| {
            cb.cmp(ca)
                .then_with(|| ka.expert_id.cmp(&kb.expert_id))
                .then_with(|| ka.layer.cmp(&kb.layer))
        });

        let mut layer_used = 0u64;
        for (key, _count) in ranked {
            let bytes = size_of(key);
            if layer_used.saturating_add(bytes) <= layer_budget
                && used.saturating_add(bytes) <= capacity
            {
                out.insert(key, bytes);
                layer_used = layer_used.saturating_add(bytes);
                used = used.saturating_add(bytes);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(layer: usize, expert_id: usize) -> ExpertKey {
        ExpertKey { layer, expert_id }
    }

    #[test]
    fn lru_evicts_oldest_entry_when_capacity_is_exceeded() {
        let mut cache = ExpertCache::new_lru(100);

        assert!(!cache.access(key(0, 0), 60));
        assert!(!cache.access(key(0, 1), 60));

        assert!(!cache.contains(key(0, 0)));
        assert!(cache.contains(key(0, 1)));
        assert_eq!(cache.resident_entries(), 1);
    }

    #[test]
    fn static_hot_selects_most_frequent_entries_within_capacity() {
        let events = vec![key(0, 1), key(0, 1), key(0, 2), key(0, 3)];
        let sizes = |k: ExpertKey| if k.expert_id == 1 { 70 } else { 40 };

        let hot = select_static_hot(&events, sizes, 100);

        assert!(hot.contains_key(&key(0, 1)));
        assert!(!hot.contains_key(&key(0, 2)));
        assert!(!hot.contains_key(&key(0, 3)));
    }

    #[test]
    fn layer_quota_static_spreads_capacity_across_layers() {
        let events = vec![
            key(0, 1),
            key(0, 1),
            key(0, 2),
            key(1, 3),
            key(1, 3),
            key(1, 0),
        ];

        let hot = select_layer_quota_hot(&events, |_| 40, 80, 2);

        assert!(hot.contains_key(&key(0, 1)));
        assert!(hot.contains_key(&key(1, 3)));
        assert_eq!(hot.len(), 2);
    }

    #[test]
    fn static_hot_lru_keeps_static_entries_and_uses_lru_for_rest() {
        let static_hot = [(key(0, 1), 70)].into_iter().collect();
        let mut cache = ExpertCache::new_static_hot_lru(100, static_hot);

        assert!(cache.access(key(0, 1), 70));
        assert!(!cache.access(key(0, 2), 20));
        assert!(!cache.access(key(0, 3), 20));

        assert!(cache.contains(key(0, 1)));
        assert!(!cache.contains(key(0, 2)));
        assert!(cache.contains(key(0, 3)));
    }

    #[test]
    fn adaptive_cache_keeps_frequent_entry_over_one_off_miss() {
        let mut cache = AdaptiveExpertCache::new(100);

        assert!(!cache.access(key(0, 0), 50));
        assert!(cache.access(key(0, 0), 50));
        assert!(!cache.access(key(0, 1), 50));
        assert!(!cache.access(key(0, 2), 50));

        assert!(cache.contains(key(0, 0)));
        assert!(!cache.contains(key(0, 1)));
        assert!(cache.contains(key(0, 2)));
        assert_eq!(cache.resident_entries(), 2);
    }

    #[test]
    fn window_cache_forgets_old_frequency() {
        let mut cache = WindowExpertCache::new(100, 3);

        assert!(!cache.access(key(0, 0), 50));
        assert!(cache.access(key(0, 0), 50));
        assert!(cache.access(key(0, 0), 50));
        assert!(!cache.access(key(0, 1), 50));
        assert!(!cache.access(key(0, 2), 50));

        assert_eq!(cache.resident_entries(), 2);
    }
}
