use rnb_memory::{MemoryBudget, MemoryTier};
use rnb_moe_cache_sim::common::cache::CachePolicy;
use rnb_moe_cache_sim::common::model::ModelMeta;
use rnb_moe_cache_sim::common::trace::TraceEvent;
use rnb_moe_cache_sim::pc::hardware::HardwareMeta;
use rnb_moe_cache_sim::pc::simulate::{
    simulate_pc_cache, simulate_pc_cache_after_warmup, simulate_pc_cache_with_memory_budget,
    simulate_pc_cache_with_options, simulate_pc_cache_with_predictor_options,
    simulate_pc_cache_with_static_options,
};

fn model() -> ModelMeta {
    ModelMeta {
        name: "synthetic".to_string(),
        num_layers: 2,
        num_experts: 4,
        experts_per_token: 2,
        default_expert_bytes: 100,
        default_gate_up_bytes: 60,
        default_down_bytes: 40,
        dense_resident_bytes: 0,
        kv_cache_bytes: 0,
        entries: Vec::new(),
    }
}

fn hardware() -> HardwareMeta {
    HardwareMeta {
        name: "pcie".to_string(),
        h2d_bandwidth_gbps: 10.0,
        h2d_latency_us: 10.0,
        gpu_expert_compute_us: 100.0,
        cpu_expert_compute_us: 500.0,
    }
}

fn event(step: usize, layer: usize, expert_id: usize) -> TraceEvent {
    TraceEvent {
        step,
        layer,
        expert_id,
        rank: None,
        score: None,
    }
}

fn ranked_event(step: usize, layer: usize, expert_id: usize, rank: usize) -> TraceEvent {
    TraceEvent {
        step,
        layer,
        expert_id,
        rank: Some(rank),
        score: None,
    }
}

#[test]
fn larger_cache_never_reduces_hit_rate_for_lru() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 0),
        event(1, 0, 1),
    ];

    let small = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 100).unwrap();
    let large = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 200).unwrap();

    assert!(large.hit_rate >= small.hit_rate);
    assert!(large.miss_bytes_per_token <= small.miss_bytes_per_token);
}

#[test]
fn pc_cache_uses_vram_memory_budget_available_bytes() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 0),
        event(1, 0, 1),
    ];
    let budget = MemoryBudget::new(MemoryTier::Vram, 250, 50);

    let row = simulate_pc_cache_with_memory_budget(
        &events,
        &model(),
        &hardware(),
        CachePolicy::Lru,
        budget,
    )
    .unwrap();

    assert_eq!(row.cache_bytes, 200);
    assert_eq!(row.usable_cache_bytes, 200);
    assert_eq!(row.resident_entries, 2);
}

#[test]
fn pc_cache_rejects_non_vram_memory_budget() {
    let events = vec![event(0, 0, 0)];
    let budget = MemoryBudget::new(MemoryTier::Ram, 200, 0);

    let error = simulate_pc_cache_with_memory_budget(
        &events,
        &model(),
        &hardware(),
        CachePolicy::Lru,
        budget,
    )
    .unwrap_err();

    assert!(error.contains("VRAM"));
}

#[test]
fn computes_copy_ms_and_break_even_hit_rate() {
    let events = vec![event(0, 0, 0), event(0, 0, 1), event(1, 0, 0)];

    let row = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 100).unwrap();

    assert!(row.copy_ms_per_token > 0.0);
    assert!(row.break_even_hit_rate.is_some());
}

#[test]
fn warmup_static_hot_uses_prefix_and_evaluates_suffix() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 0),
        event(1, 0, 0),
        event(1, 0, 1),
        event(2, 0, 0),
        event(2, 0, 1),
    ];

    let row = simulate_pc_cache_after_warmup(
        &events,
        &model(),
        &hardware(),
        CachePolicy::StaticHot,
        100,
        1,
    )
    .unwrap();

    assert_eq!(row.warmup_steps, 1);
    assert_eq!(row.evaluated_steps, 2);
    assert_eq!(row.evaluated_events, 4);
    assert_eq!(row.hit_rate, 0.5);
}

#[test]
fn static_hot_can_use_prefill_source_and_evaluate_later_decode() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 0),
        event(1, 0, 0),
        event(1, 0, 0),
        event(2, 0, 1),
        event(2, 0, 1),
        event(3, 0, 0),
        event(3, 0, 1),
    ];

    let row = simulate_pc_cache_with_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::StaticHot,
        100,
        2,
        0,
        3,
    )
    .unwrap();

    assert_eq!(row.warmup_steps, 2);
    assert_eq!(row.eval_start_step, 3);
    assert_eq!(row.evaluated_steps, 1);
    assert_eq!(row.evaluated_events, 2);
    assert_eq!(row.hit_rate, 0.5);
}

#[test]
fn adaptive_policy_learns_during_evaluation() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 0),
        event(1, 0, 2),
        event(2, 0, 0),
        event(2, 0, 3),
        event(3, 0, 0),
        event(3, 0, 1),
    ];

    let row = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::AdaptiveLfuLru,
        200,
    )
    .unwrap();

    assert_eq!(row.evaluated_events, 8);
    assert!(row.hit_rate > 0.0);
    assert_eq!(row.resident_entries, 2);
}

#[test]
fn window_policy_records_window_steps() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 0),
        event(1, 0, 2),
        event(2, 0, 2),
        event(2, 0, 3),
    ];

    let row = simulate_pc_cache_with_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::WindowLfuLru,
        200,
        0,
        2,
        0,
    )
    .unwrap();

    assert_eq!(row.window_steps, 2);
    assert_eq!(row.evaluated_events, 6);
    assert_eq!(row.resident_entries, 2);
}

#[test]
fn layer_quota_static_is_available_in_simulator() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 0),
        event(0, 1, 1),
        event(0, 1, 1),
        event(1, 0, 0),
        event(1, 1, 1),
    ];

    let row = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::LayerQuotaStatic,
        200,
    )
    .unwrap();

    assert_eq!(row.resident_entries, 2);
    assert!(row.hit_rate > 0.0);
}

#[test]
fn rank_weighted_static_is_available_in_simulator() {
    let events = vec![
        ranked_event(0, 0, 0, 0),
        ranked_event(0, 0, 1, 1),
        ranked_event(1, 0, 0, 0),
        ranked_event(1, 0, 2, 1),
        ranked_event(2, 0, 0, 0),
        ranked_event(2, 0, 3, 1),
    ];

    let row = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::RankWeightedStatic,
        100,
    )
    .unwrap();

    assert_eq!(row.resident_entries, 1);
    assert!(row.hit_rate >= 0.5);
}

#[test]
fn gate_up_static_counts_down_bytes_on_hits() {
    let events = vec![event(0, 0, 0), event(0, 0, 0), event(1, 0, 0)];

    let row = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::GateUpStatic,
        60,
    )
    .unwrap();

    assert_eq!(row.resident_entries, 1);
    assert!(row.hit_rate > 0.0);
    assert!(row.miss_bytes_per_token > 40.0);
    assert!(row.miss_bytes_per_token < 200.0);
}

#[test]
fn least_stale_keeps_selected_current_layer_experts() {
    let events = vec![
        event(0, 0, 0),
        event(0, 1, 1),
        event(1, 0, 0),
        ranked_event(1, 1, 2, 0),
        ranked_event(1, 1, 1, 1),
    ];

    let lru = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 200).unwrap();
    let least_stale =
        simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::LeastStale, 200).unwrap();

    assert!(least_stale.hit_rate > lru.hit_rate);
    assert_eq!(least_stale.resident_entries, 2);
}

#[test]
fn least_stale_lookahead_prefetch_hides_future_layer_misses() {
    let events = vec![
        event(0, 0, 0),
        event(0, 1, 1),
        event(1, 0, 0),
        event(1, 1, 1),
    ];

    let no_prefetch =
        simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::LeastStale, 200).unwrap();
    let lookahead = simulate_pc_cache_with_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::LeastStale,
        200,
        0,
        2,
        0,
    )
    .unwrap();

    assert!(lookahead.hit_rate > no_prefetch.hit_rate);
    assert!(lookahead.copy_ms_per_token < no_prefetch.copy_ms_per_token);
}

#[test]
fn partial_lookahead_prefetch_loses_to_oracle_lookahead() {
    let events = vec![
        ranked_event(0, 0, 0, 0),
        ranked_event(0, 1, 1, 0),
        ranked_event(0, 1, 2, 1),
        ranked_event(1, 0, 0, 0),
        ranked_event(1, 1, 1, 0),
        ranked_event(1, 1, 2, 1),
    ];

    let partial = simulate_pc_cache_with_predictor_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::LeastStale,
        300,
        0,
        1,
        0,
        50,
        50,
        0,
    )
    .unwrap();
    let oracle = simulate_pc_cache_with_predictor_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::LeastStale,
        300,
        0,
        1,
        0,
        50,
        100,
        0,
    )
    .unwrap();

    assert!(oracle.hit_rate > partial.hit_rate);
    assert!(oracle.copy_ms_per_token < partial.copy_ms_per_token);
}

#[test]
fn previous_step_prefetch_uses_same_layer_route_from_previous_token() {
    let events = vec![
        event(0, 0, 0),
        event(0, 1, 1),
        event(1, 0, 0),
        event(1, 1, 1),
    ];

    let lru = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 100).unwrap();
    let prev_step = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::PrevStepPrefetch,
        100,
    )
    .unwrap();

    assert!(prev_step.hit_rate > lru.hit_rate);
    assert!(prev_step.copy_ms_per_token < lru.copy_ms_per_token);
}

#[test]
fn router_current_jit_keeps_lru_hits_but_hides_copy_on_critical_path() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 2),
        event(1, 0, 3),
    ];

    let lru = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 100).unwrap();
    let jit = simulate_pc_cache(
        &events,
        &model(),
        &hardware(),
        CachePolicy::RouterCurrentJit,
        100,
    )
    .unwrap();

    assert_eq!(jit.hit_rate, lru.hit_rate);
    assert_eq!(jit.miss_bytes_per_token, lru.miss_bytes_per_token);
    assert!(jit.copy_ms_per_token < lru.copy_ms_per_token);
}

#[test]
fn online_layer_hot_prefetch_uses_layer_frequency_history() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 1),
        event(1, 0, 0),
        event(1, 0, 2),
        event(2, 0, 0),
        event(2, 0, 3),
    ];

    let lru = simulate_pc_cache(&events, &model(), &hardware(), CachePolicy::Lru, 100).unwrap();
    let online_hot = simulate_pc_cache_with_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::OnlineLayerHotPrefetch,
        100,
        0,
        1,
        0,
    )
    .unwrap();

    assert!(online_hot.hit_rate >= lru.hit_rate);
    assert!(online_hot.copy_ms_per_token <= lru.copy_ms_per_token);
}

#[test]
fn online_layer_hot_staged_prefetch_hits_without_polluting_main_cache() {
    let events = vec![event(0, 0, 0), event(1, 0, 1), event(2, 0, 0)];

    let staged = simulate_pc_cache_with_static_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::OnlineLayerHotStaged,
        200,
        1,
        1,
        1,
        50,
    )
    .unwrap();

    assert_eq!(staged.hit_rate, 0.5);
    assert_eq!(staged.resident_entries, 1);
}

#[test]
fn static_hot_adaptive_uses_static_prefix_and_adaptive_tail() {
    let events = vec![
        event(0, 0, 0),
        event(0, 0, 0),
        event(1, 0, 0),
        event(1, 0, 1),
        event(2, 0, 0),
        event(2, 0, 1),
        event(3, 0, 0),
        event(3, 0, 2),
    ];

    let row = simulate_pc_cache_with_options(
        &events,
        &model(),
        &hardware(),
        CachePolicy::StaticHotAdaptive,
        200,
        1,
        0,
        1,
    )
    .unwrap();

    assert_eq!(row.resident_entries, 2);
    assert!(row.hit_rate > 0.0);
}
