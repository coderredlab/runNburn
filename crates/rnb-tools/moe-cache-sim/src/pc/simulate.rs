use crate::common::cache::{
    select_layer_quota_hot, select_static_hot, AdaptiveExpertCache, CachePolicy, ExpertCache,
    ExpertKey, LeastStaleExpertCache, WindowExpertCache,
};
use crate::common::metrics::SimulationMetrics;
use crate::common::model::ModelMeta;
use crate::common::trace::TraceEvent;
use crate::pc::hardware::HardwareMeta;
use rnb_memory::{MemoryBudget, MemoryTier};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, serde::Serialize)]
pub struct PcSimulationRow {
    pub cache_bytes: u64,
    pub usable_cache_bytes: u64,
    pub warmup_steps: usize,
    pub eval_start_step: usize,
    pub static_hot_pct: u8,
    pub window_steps: usize,
    pub predictor_recall_pct: u8,
    pub predictor_extra_ratio: u16,
    pub evaluated_events: u64,
    pub evaluated_steps: usize,
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub miss_bytes_per_token: f64,
    pub copy_ms_per_token: f64,
    pub resident_entries: usize,
    pub break_even_hit_rate: Option<f64>,
}

pub fn simulate_pc_cache(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    cache_bytes: u64,
) -> Result<PcSimulationRow, String> {
    simulate_pc_cache_with_options(events, model, hardware, policy, cache_bytes, 0, 0, 0)
}

pub fn simulate_pc_cache_with_memory_budget(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    budget: MemoryBudget,
) -> Result<PcSimulationRow, String> {
    if budget.tier() != MemoryTier::Vram {
        return Err(format!(
            "pc moe cache budget must target VRAM, got {:?}",
            budget.tier()
        ));
    }

    simulate_pc_cache(events, model, hardware, policy, budget.available_bytes())
}

pub fn simulate_pc_cache_after_warmup(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    cache_bytes: u64,
    warmup_steps: usize,
) -> Result<PcSimulationRow, String> {
    simulate_pc_cache_with_options(
        events,
        model,
        hardware,
        policy,
        cache_bytes,
        warmup_steps,
        0,
        warmup_steps,
    )
}

pub fn simulate_pc_cache_with_options(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    cache_bytes: u64,
    warmup_steps: usize,
    window_steps: usize,
    eval_start_step: usize,
) -> Result<PcSimulationRow, String> {
    simulate_pc_cache_with_static_options(
        events,
        model,
        hardware,
        policy,
        cache_bytes,
        warmup_steps,
        window_steps,
        eval_start_step,
        50,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_pc_cache_with_static_options(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    cache_bytes: u64,
    warmup_steps: usize,
    window_steps: usize,
    eval_start_step: usize,
    static_hot_pct: u8,
) -> Result<PcSimulationRow, String> {
    simulate_pc_cache_with_predictor_options(
        events,
        model,
        hardware,
        policy,
        cache_bytes,
        warmup_steps,
        window_steps,
        eval_start_step,
        static_hot_pct,
        100,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_pc_cache_with_predictor_options(
    events: &[TraceEvent],
    model: &ModelMeta,
    hardware: &HardwareMeta,
    policy: CachePolicy,
    cache_bytes: u64,
    warmup_steps: usize,
    window_steps: usize,
    eval_start_step: usize,
    static_hot_pct: u8,
    predictor_recall_pct: u8,
    predictor_extra_ratio: u16,
) -> Result<PcSimulationRow, String> {
    if predictor_recall_pct > 100 {
        return Err(format!(
            "predictor recall must be 0..100, got {predictor_recall_pct}"
        ));
    }
    if static_hot_pct > 100 {
        return Err(format!(
            "static hot pct must be 0..100, got {static_hot_pct}"
        ));
    }

    let reserved = model.reserved_bytes();
    let usable_cache_bytes = cache_bytes.saturating_sub(reserved);
    let ordered = ordered_events(events);
    let warmup_events = ordered
        .iter()
        .copied()
        .filter(|event| event.step < warmup_steps)
        .collect::<Vec<_>>();
    let evaluated_events = ordered
        .iter()
        .copied()
        .filter(|event| event.step >= eval_start_step)
        .collect::<Vec<_>>();
    let hot_source = if warmup_steps == 0 {
        events.iter().collect::<Vec<_>>()
    } else {
        warmup_events.clone()
    };
    let hot_keys = hot_source
        .iter()
        .map(|event| ExpertKey {
            layer: event.layer,
            expert_id: event.expert_id,
        })
        .collect::<Vec<_>>();
    let mut cache = match policy {
        CachePolicy::Lru | CachePolicy::RouterCurrentJit => {
            ExpertCache::new_lru(usable_cache_bytes)
        }
        CachePolicy::StaticHot => {
            let static_hot = select_static_hot(
                &hot_keys,
                |key| model.expert_bytes(key.layer, key.expert_id),
                usable_cache_bytes,
            );
            ExpertCache::new_static_hot(usable_cache_bytes, static_hot)
        }
        CachePolicy::StaticHotLru => {
            let static_budget = pct_bytes(usable_cache_bytes, static_hot_pct);
            let static_hot = select_static_hot(
                &hot_keys,
                |key| model.expert_bytes(key.layer, key.expert_id),
                static_budget,
            );
            ExpertCache::new_static_hot_lru(usable_cache_bytes, static_hot)
        }
        CachePolicy::StaticHotAdaptive => {
            let static_budget = pct_bytes(usable_cache_bytes, static_hot_pct);
            let static_hot = select_static_hot(
                &hot_keys,
                |key| model.expert_bytes(key.layer, key.expert_id),
                static_budget,
            );
            ExpertCache::new_static_hot(static_budget, static_hot)
        }
        CachePolicy::LayerQuotaStatic => {
            let static_hot = select_layer_quota_hot(
                &hot_keys,
                |key| model.expert_bytes(key.layer, key.expert_id),
                usable_cache_bytes,
                model.num_layers,
            );
            ExpertCache::new_static_hot(usable_cache_bytes, static_hot)
        }
        CachePolicy::RankWeightedStatic => {
            let rank_weighted_keys = rank_weighted_hot_keys(&hot_source, model.experts_per_token);
            let static_hot = select_static_hot(
                &rank_weighted_keys,
                |key| model.expert_bytes(key.layer, key.expert_id),
                usable_cache_bytes,
            );
            ExpertCache::new_static_hot(usable_cache_bytes, static_hot)
        }
        CachePolicy::GateUpStatic => {
            let static_hot = select_static_hot(
                &hot_keys,
                |key| model.expert_gate_up_bytes(key.layer, key.expert_id),
                usable_cache_bytes,
            );
            ExpertCache::new_static_hot(usable_cache_bytes, static_hot)
        }
        CachePolicy::AdaptiveLfuLru => ExpertCache::new_lru(0),
        CachePolicy::LeastStale
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch => ExpertCache::new_lru(0),
        CachePolicy::OnlineLayerHotStaged => ExpertCache::new_lru(
            usable_cache_bytes.saturating_sub(pct_bytes(usable_cache_bytes, static_hot_pct)),
        ),
        CachePolicy::WindowLfuLru => ExpertCache::new_lru(0),
    };
    let mut staged_prefetch_cache = match policy {
        CachePolicy::OnlineLayerHotStaged => Some(ExpertCache::new_lru(pct_bytes(
            usable_cache_bytes,
            static_hot_pct,
        ))),
        CachePolicy::Lru
        | CachePolicy::RouterCurrentJit
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::GateUpStatic
        | CachePolicy::LeastStale
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::AdaptiveLfuLru
        | CachePolicy::WindowLfuLru => None,
    };
    let mut adaptive_cache = match policy {
        CachePolicy::AdaptiveLfuLru => Some(AdaptiveExpertCache::new(usable_cache_bytes)),
        CachePolicy::StaticHotAdaptive => Some(AdaptiveExpertCache::new(
            usable_cache_bytes.saturating_sub(pct_bytes(usable_cache_bytes, static_hot_pct)),
        )),
        CachePolicy::Lru
        | CachePolicy::RouterCurrentJit
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::GateUpStatic
        | CachePolicy::LeastStale
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::OnlineLayerHotStaged
        | CachePolicy::WindowLfuLru => None,
    };
    let window_len = model
        .num_layers
        .saturating_mul(model.experts_per_token)
        .saturating_mul(window_steps.max(1));
    let mut window_cache = match policy {
        CachePolicy::WindowLfuLru => Some(WindowExpertCache::new(usable_cache_bytes, window_len)),
        CachePolicy::Lru
        | CachePolicy::RouterCurrentJit
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::GateUpStatic
        | CachePolicy::LeastStale
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::OnlineLayerHotStaged
        | CachePolicy::AdaptiveLfuLru => None,
    };
    let mut least_stale_cache = match policy {
        CachePolicy::LeastStale
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch => {
            Some(LeastStaleExpertCache::new(usable_cache_bytes))
        }
        CachePolicy::OnlineLayerHotStaged
        | CachePolicy::RouterCurrentJit
        | CachePolicy::Lru
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::GateUpStatic
        | CachePolicy::AdaptiveLfuLru
        | CachePolicy::WindowLfuLru => None,
    };
    let previous_step_groups = previous_step_group_map(&ordered);
    let mut online_hot_counts = Vec::<Vec<u32>>::new();

    for group in layer_groups(warmup_events) {
        if let Some(cache) = least_stale_cache.as_mut() {
            cache.begin_layer(group.step, group.layer, &group.keys);
        }
        for &key in &group.keys {
            let bytes = cache_entry_bytes(model, policy, key);
            if policy == CachePolicy::StaticHotAdaptive && cache.contains(key) {
                cache.access(key, bytes);
            } else if let Some(cache) = adaptive_cache.as_mut() {
                cache.access(key, bytes);
            } else if let Some(cache) = window_cache.as_mut() {
                cache.access(key, bytes);
            } else if let Some(cache) = least_stale_cache.as_mut() {
                cache.access(key, bytes);
            } else {
                cache.access(key, bytes);
            }
        }
        if matches!(
            policy,
            CachePolicy::OnlineLayerHotPrefetch | CachePolicy::OnlineLayerHotStaged
        ) {
            record_online_layer_hot_counts(&mut online_hot_counts, &group);
        }
    }

    let mut metrics = SimulationMetrics {
        events: 0,
        hits: 0,
        misses: 0,
        miss_bytes: 0,
        resident_entries: 0,
    };
    let mut jit_critical_copy_ms = 0.0;
    let evaluated_groups = layer_groups(evaluated_events);
    for (idx, group) in evaluated_groups.iter().enumerate() {
        if let Some(cache) = least_stale_cache.as_mut() {
            cache.begin_layer(group.step, group.layer, &group.keys);
            if policy == CachePolicy::PrevStepPrefetch {
                prefetch_previous_step(cache, model, group, &previous_step_groups);
            } else if policy == CachePolicy::OnlineLayerHotPrefetch {
                prefetch_online_layer_hot(cache, model, group, &online_hot_counts, window_steps);
            } else {
                prefetch_lookahead(
                    cache,
                    model,
                    policy,
                    &evaluated_groups,
                    idx,
                    window_steps,
                    predictor_recall_pct,
                    predictor_extra_ratio,
                );
            }
        }
        if policy == CachePolicy::OnlineLayerHotStaged {
            prefetch_online_layer_hot_staged(
                staged_prefetch_cache
                    .as_mut()
                    .expect("staged policy must have a prefetch cache"),
                &cache,
                model,
                group,
                &online_hot_counts,
                window_steps,
            );
        }
        let mut group_jit_overlap_ms = 0.0;
        for &key in &group.keys {
            let bytes = cache_entry_bytes(model, policy, key);
            let tail_bytes = uncached_tail_bytes(model, policy, key);
            let hit = if policy == CachePolicy::OnlineLayerHotStaged {
                access_staged_cache(
                    &mut cache,
                    staged_prefetch_cache
                        .as_mut()
                        .expect("staged policy must have a prefetch cache"),
                    key,
                    bytes,
                )
            } else if policy == CachePolicy::StaticHotAdaptive && cache.contains(key) {
                cache.access(key, bytes)
            } else if let Some(cache) = adaptive_cache.as_mut() {
                cache.access(key, bytes)
            } else if let Some(cache) = window_cache.as_mut() {
                cache.access(key, bytes)
            } else if let Some(cache) = least_stale_cache.as_mut() {
                cache.access(key, bytes)
            } else {
                cache.access(key, bytes)
            };
            metrics.events += 1;
            if hit {
                metrics.hits += 1;
                metrics.miss_bytes = metrics.miss_bytes.saturating_add(tail_bytes);
            } else {
                metrics.misses += 1;
                let missed_bytes = bytes.saturating_add(tail_bytes);
                metrics.miss_bytes = metrics.miss_bytes.saturating_add(missed_bytes);
                if policy == CachePolicy::RouterCurrentJit {
                    let copy_ms = hardware.copy_ms_for_bytes(missed_bytes as f64, 1);
                    jit_critical_copy_ms += (copy_ms - group_jit_overlap_ms).max(0.0);
                    group_jit_overlap_ms = (group_jit_overlap_ms - copy_ms).max(0.0);
                }
            }
            if policy == CachePolicy::RouterCurrentJit {
                group_jit_overlap_ms += hardware.gpu_expert_compute_us / 1000.0;
            }
        }
        if matches!(
            policy,
            CachePolicy::OnlineLayerHotPrefetch | CachePolicy::OnlineLayerHotStaged
        ) {
            record_online_layer_hot_counts(&mut online_hot_counts, group);
        }
    }
    metrics.resident_entries = if policy == CachePolicy::StaticHotAdaptive {
        cache.resident_entries()
            + adaptive_cache
                .as_ref()
                .map(AdaptiveExpertCache::resident_entries)
                .unwrap_or(0)
    } else {
        adaptive_cache
            .as_ref()
            .map(AdaptiveExpertCache::resident_entries)
            .or_else(|| {
                window_cache
                    .as_ref()
                    .map(WindowExpertCache::resident_entries)
            })
            .or_else(|| {
                least_stale_cache
                    .as_ref()
                    .map(LeastStaleExpertCache::resident_entries)
            })
            .or_else(|| {
                staged_prefetch_cache.as_ref().map(|staged| {
                    cache
                        .resident_entries()
                        .saturating_add(staged.resident_entries())
                })
            })
            .unwrap_or_else(|| cache.resident_entries())
    };

    let steps = trace_step_count_after(events, eval_start_step);
    let miss_bytes_per_token = metrics.miss_bytes_per_token(steps);
    let copy_ms_per_token = if steps == 0 {
        0.0
    } else if policy == CachePolicy::RouterCurrentJit {
        jit_critical_copy_ms / steps as f64
    } else {
        let transfers = copy_transfer_count(&metrics, policy);
        hardware.copy_ms_for_bytes(metrics.miss_bytes as f64, transfers) / steps as f64
    };
    let transfer_count = copy_transfer_count(&metrics, policy);
    let avg_miss_bytes = if transfer_count == 0 {
        0.0
    } else {
        metrics.miss_bytes as f64 / transfer_count as f64
    };
    let avg_copy_us = hardware.copy_ms_for_bytes(avg_miss_bytes, 1) * 1000.0;

    Ok(PcSimulationRow {
        cache_bytes,
        usable_cache_bytes,
        warmup_steps,
        eval_start_step,
        static_hot_pct,
        window_steps,
        predictor_recall_pct,
        predictor_extra_ratio,
        evaluated_events: metrics.events,
        evaluated_steps: steps,
        hit_rate: metrics.hit_rate(),
        miss_rate: metrics.miss_rate(),
        miss_bytes_per_token,
        copy_ms_per_token,
        resident_entries: metrics.resident_entries,
        break_even_hit_rate: hardware.break_even_hit_rate(avg_copy_us),
    })
}

fn pct_bytes(bytes: u64, pct: u8) -> u64 {
    bytes.saturating_mul(pct as u64) / 100
}

fn prefetch_previous_step(
    cache: &mut LeastStaleExpertCache,
    model: &ModelMeta,
    group: &LayerGroup,
    previous_step_groups: &HashMap<(usize, usize), Vec<ExpertKey>>,
) {
    let Some(previous_step) = group.step.checked_sub(1) else {
        return;
    };
    let Some(keys) = previous_step_groups.get(&(previous_step, group.layer)) else {
        return;
    };
    for &key in keys {
        let bytes = cache_entry_bytes(model, CachePolicy::PrevStepPrefetch, key);
        cache.prefetch(key, bytes, group.step, group.layer);
    }
}

fn prefetch_online_layer_hot(
    cache: &mut LeastStaleExpertCache,
    model: &ModelMeta,
    group: &LayerGroup,
    counts_by_layer: &[Vec<u32>],
    top_n: usize,
) {
    if top_n == 0 {
        return;
    }
    let Some(counts) = counts_by_layer.get(group.layer) else {
        return;
    };
    let mut ranked = counts
        .iter()
        .enumerate()
        .filter_map(|(expert_id, &count)| {
            (count > 0).then_some(ExpertKey {
                layer: group.layer,
                expert_id,
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        counts[right.expert_id]
            .cmp(&counts[left.expert_id])
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });
    for key in ranked.into_iter().take(top_n) {
        let bytes = cache_entry_bytes(model, CachePolicy::OnlineLayerHotPrefetch, key);
        cache.prefetch_if_space(key, bytes, group.step, group.layer);
    }
}

fn prefetch_online_layer_hot_staged(
    staging_cache: &mut ExpertCache,
    main_cache: &ExpertCache,
    model: &ModelMeta,
    group: &LayerGroup,
    counts_by_layer: &[Vec<u32>],
    top_n: usize,
) {
    if top_n == 0 {
        return;
    }
    let Some(counts) = counts_by_layer.get(group.layer) else {
        return;
    };
    let mut ranked = counts
        .iter()
        .enumerate()
        .filter_map(|(expert_id, &count)| {
            (count > 0).then_some(ExpertKey {
                layer: group.layer,
                expert_id,
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        counts[right.expert_id]
            .cmp(&counts[left.expert_id])
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });
    for key in ranked.into_iter().take(top_n) {
        if main_cache.contains(key) {
            continue;
        }
        let bytes = cache_entry_bytes(model, CachePolicy::OnlineLayerHotStaged, key);
        staging_cache.access(key, bytes);
    }
}

fn access_staged_cache(
    main_cache: &mut ExpertCache,
    staging_cache: &mut ExpertCache,
    key: ExpertKey,
    bytes: u64,
) -> bool {
    if main_cache.contains(key) {
        main_cache.access(key, bytes);
        return true;
    }
    if staging_cache.remove_dynamic(key).is_some() {
        main_cache.access(key, bytes);
        return true;
    }
    main_cache.access(key, bytes);
    false
}

fn record_online_layer_hot_counts(counts_by_layer: &mut Vec<Vec<u32>>, group: &LayerGroup) {
    if counts_by_layer.len() <= group.layer {
        counts_by_layer.resize_with(group.layer + 1, Vec::new);
    }
    let counts = &mut counts_by_layer[group.layer];
    for key in &group.keys {
        if counts.len() <= key.expert_id {
            counts.resize(key.expert_id + 1, 0);
        }
        counts[key.expert_id] = counts[key.expert_id].saturating_add(1);
    }
}

fn prefetch_lookahead(
    cache: &mut LeastStaleExpertCache,
    model: &ModelMeta,
    policy: CachePolicy,
    groups: &[LayerGroup],
    current_idx: usize,
    lookahead_groups: usize,
    predictor_recall_pct: u8,
    predictor_extra_ratio: u16,
) {
    if policy != CachePolicy::LeastStale || lookahead_groups == 0 {
        return;
    }
    let end = current_idx
        .saturating_add(1)
        .saturating_add(lookahead_groups)
        .min(groups.len());
    for group in groups.iter().take(end).skip(current_idx + 1) {
        for key in
            predicted_prefetch_keys(model, group, predictor_recall_pct, predictor_extra_ratio)
        {
            let bytes = cache_entry_bytes(model, policy, key);
            cache.prefetch(key, bytes, group.step, group.layer);
        }
    }
}

fn predicted_prefetch_keys(
    model: &ModelMeta,
    group: &LayerGroup,
    predictor_recall_pct: u8,
    predictor_extra_ratio: u16,
) -> Vec<ExpertKey> {
    let keep = group
        .keys
        .len()
        .saturating_mul(predictor_recall_pct as usize)
        .div_ceil(100);
    let mut out = group.keys.iter().take(keep).copied().collect::<Vec<_>>();
    if model.num_experts == 0 || out.is_empty() || predictor_extra_ratio == 0 {
        return out;
    }

    let extra_count = out
        .len()
        .saturating_mul(predictor_extra_ratio as usize)
        .div_ceil(100);
    let target_len = out.len().saturating_add(extra_count);
    let mut seen = out.iter().copied().collect::<HashSet<_>>();
    let seeds = out.clone();
    let mut offset = 1usize;
    while seen.len() < target_len && seen.len() < model.num_experts {
        let seed = seeds[(offset - 1) % seeds.len()];
        let key = ExpertKey {
            layer: seed.layer,
            expert_id: (seed.expert_id + offset) % model.num_experts,
        };
        if seen.insert(key) {
            out.push(key);
        }
        offset = offset.saturating_add(1);
    }
    out
}

fn trace_step_count_after(events: &[TraceEvent], warmup_steps: usize) -> usize {
    let steps = events
        .iter()
        .filter(|event| event.step >= warmup_steps)
        .map(|event| event.step)
        .collect::<HashSet<_>>();
    steps.len()
}

struct LayerGroup {
    step: usize,
    layer: usize,
    keys: Vec<ExpertKey>,
}

fn layer_groups(events: Vec<&TraceEvent>) -> Vec<LayerGroup> {
    let mut groups = Vec::new();
    let mut current = None::<LayerGroup>;
    for event in events {
        let key = ExpertKey {
            layer: event.layer,
            expert_id: event.expert_id,
        };
        match current.as_mut() {
            Some(group) if group.step == event.step && group.layer == event.layer => {
                group.keys.push(key);
            }
            Some(_) => {
                groups.push(current.take().expect("group exists"));
                current = Some(LayerGroup {
                    step: event.step,
                    layer: event.layer,
                    keys: vec![key],
                });
            }
            None => {
                current = Some(LayerGroup {
                    step: event.step,
                    layer: event.layer,
                    keys: vec![key],
                });
            }
        }
    }
    if let Some(group) = current {
        groups.push(group);
    }
    groups
}

fn previous_step_group_map(events: &[&TraceEvent]) -> HashMap<(usize, usize), Vec<ExpertKey>> {
    let mut out = HashMap::<(usize, usize), Vec<ExpertKey>>::new();
    for group in layer_groups(events.to_vec()) {
        out.insert((group.step, group.layer), group.keys);
    }
    out
}

fn ordered_events(events: &[TraceEvent]) -> Vec<&TraceEvent> {
    let mut out = events.iter().collect::<Vec<_>>();
    out.sort_by(|left, right| {
        left.step
            .cmp(&right.step)
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| {
                left.rank
                    .unwrap_or(usize::MAX)
                    .cmp(&right.rank.unwrap_or(usize::MAX))
            })
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });
    out
}

fn rank_weighted_hot_keys(events: &[&TraceEvent], experts_per_token: usize) -> Vec<ExpertKey> {
    let mut out = Vec::new();
    for event in events {
        let key = ExpertKey {
            layer: event.layer,
            expert_id: event.expert_id,
        };
        let weight = event
            .rank
            .map(|rank| experts_per_token.saturating_sub(rank))
            .unwrap_or(1)
            .max(1);
        out.extend(std::iter::repeat_n(key, weight));
    }
    out
}

fn cache_entry_bytes(model: &ModelMeta, policy: CachePolicy, key: ExpertKey) -> u64 {
    match policy {
        CachePolicy::GateUpStatic => model.expert_gate_up_bytes(key.layer, key.expert_id),
        CachePolicy::Lru
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::LeastStale
        | CachePolicy::RouterCurrentJit
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::OnlineLayerHotStaged
        | CachePolicy::AdaptiveLfuLru
        | CachePolicy::WindowLfuLru => model.expert_bytes(key.layer, key.expert_id),
    }
}

fn uncached_tail_bytes(model: &ModelMeta, policy: CachePolicy, key: ExpertKey) -> u64 {
    match policy {
        CachePolicy::GateUpStatic => model.expert_down_bytes(key.layer, key.expert_id),
        CachePolicy::Lru
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::LeastStale
        | CachePolicy::RouterCurrentJit
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::OnlineLayerHotStaged
        | CachePolicy::AdaptiveLfuLru
        | CachePolicy::WindowLfuLru => 0,
    }
}

fn copy_transfer_count(metrics: &SimulationMetrics, policy: CachePolicy) -> u64 {
    match policy {
        CachePolicy::GateUpStatic => metrics.events,
        CachePolicy::Lru
        | CachePolicy::StaticHot
        | CachePolicy::StaticHotLru
        | CachePolicy::StaticHotAdaptive
        | CachePolicy::LayerQuotaStatic
        | CachePolicy::RankWeightedStatic
        | CachePolicy::LeastStale
        | CachePolicy::RouterCurrentJit
        | CachePolicy::PrevStepPrefetch
        | CachePolicy::OnlineLayerHotPrefetch
        | CachePolicy::OnlineLayerHotStaged
        | CachePolicy::AdaptiveLfuLru
        | CachePolicy::WindowLfuLru => metrics.misses,
    }
}
