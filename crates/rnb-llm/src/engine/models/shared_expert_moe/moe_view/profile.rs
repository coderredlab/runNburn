//! Qwen MoE decode profile aggregation.

use super::*;

pub(super) fn record_decode_fanout_profile(
    per_expert: &[ExpertProfileAcc],
    fanout_us: u128,
    profile_enabled: bool,
) {
    if !profile_enabled {
        return;
    }

    let high_us: u128 = per_expert.iter().map(|x| x.high_us).sum();
    let high_gate_up_us: u128 = per_expert.iter().map(|x| x.high_gate_up_us).sum();
    let high_down_us: u128 = per_expert.iter().map(|x| x.high_down_us).sum();
    let low_us: u128 = per_expert.iter().map(|x| x.low_us).sum();
    let low_gate_up_us: u128 = per_expert.iter().map(|x| x.low_gate_up_us).sum();
    let low_gate_up_row_us: u128 = per_expert.iter().map(|x| x.low_gate_up_row_us).sum();
    let low_gate_up_tile_us: u128 = per_expert.iter().map(|x| x.low_gate_up_tile_us).sum();
    let low_gate_up_post_us: u128 = per_expert.iter().map(|x| x.low_gate_up_post_us).sum();
    let low_shadow_down_us: u128 = per_expert.iter().map(|x| x.low_shadow_down_us).sum();
    let low_base_down_us: u128 = per_expert.iter().map(|x| x.low_base_down_us).sum();
    let high_count: u64 = per_expert.iter().map(|x| x.high).sum();
    let low_count: u64 = per_expert.iter().map(|x| x.low).sum();
    let skip_count: u64 = per_expert.iter().map(|x| x.skip).sum();
    let max_expert_us: u128 = per_expert.iter().map(|x| x.wall_us).max().unwrap_or(0);
    let dispatch_us = fanout_us.saturating_sub(max_expert_us);

    record_moe_profile(
        "qwen35moe:decode:dispatch",
        std::time::Duration::from_micros(dispatch_us.min(u64::MAX as u128) as u64),
    );
    record_if_nonzero("qwen35moe:decode:high_compute", high_us);
    record_if_nonzero("qwen35moe:decode:high_gate_up_compute", high_gate_up_us);
    record_if_nonzero("qwen35moe:decode:high_down_compute", high_down_us);
    record_if_nonzero("qwen35moe:decode:low_compute", low_us);
    record_if_nonzero("qwen35moe:decode:low_gate_up_compute", low_gate_up_us);
    record_if_nonzero(
        "qwen35moe:decode:low_gate_up_row_compute",
        low_gate_up_row_us,
    );
    record_if_nonzero(
        "qwen35moe:decode:low_gate_up_tile_compute",
        low_gate_up_tile_us,
    );
    record_if_nonzero("qwen35moe:decode:low_gate_up_post", low_gate_up_post_us);
    record_if_nonzero(
        "qwen35moe:decode:low_shadow_down_compute",
        low_shadow_down_us,
    );
    record_if_nonzero("qwen35moe:decode:low_base_down_compute", low_base_down_us);
    record_moe_counts("qwen35moe:decode", high_count, low_count, skip_count);
}

fn record_if_nonzero(label: &'static str, elapsed_us: u128) {
    if elapsed_us > 0 {
        record_moe_profile(
            label,
            std::time::Duration::from_micros(elapsed_us.min(u64::MAX as u128) as u64),
        );
    }
}
