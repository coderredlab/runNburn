//! Parallel execution of independent Qwen sparse expert groups on AArch64.

use super::prefill_cpu::{QwenExpertGroupOutput, QwenRouteSlot};
use crate::engine::norm::apply_model_gate_mul_inplace;
use crate::engine::quantized_dispatch::{
    prefill_raw_dual_q4k_q8k, prefill_raw_quantized_batch, QuantizedQ8KBlock,
};
use rayon::prelude::*;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};
use std::time::Instant;

fn expert_group_ranges(slots: &[QwenRouteSlot]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < slots.len() {
        let expert = slots[start].expert;
        let mut end = start + 1;
        while end < slots.len() && slots[end].expert == expert {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_expert_groups(
    slots: &[QwenRouteSlot],
    normalized_q8k: &[QuantizedQ8KBlock],
    gate_exps_bytes: &[u8],
    up_exps_bytes: &[u8],
    down_exps_bytes: &[u8],
    per_gate_up: usize,
    gate_up_bytes_per_row: usize,
    per_down: usize,
    down_bytes_per_row: usize,
    n_ff: usize,
    hidden_dim: usize,
    down_quant: GGMLType,
    architecture: ModelArchitecture,
    profile_enabled: bool,
) -> Vec<QwenExpertGroupOutput> {
    expert_group_ranges(slots)
        .par_iter()
        .map(|&(start, end)| {
            compute_expert_group(
                slots,
                start,
                end,
                normalized_q8k,
                gate_exps_bytes,
                up_exps_bytes,
                down_exps_bytes,
                per_gate_up,
                gate_up_bytes_per_row,
                per_down,
                down_bytes_per_row,
                n_ff,
                hidden_dim,
                down_quant,
                architecture,
                profile_enabled,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_expert_group(
    slots: &[QwenRouteSlot],
    start: usize,
    end: usize,
    normalized_q8k: &[QuantizedQ8KBlock],
    gate_exps_bytes: &[u8],
    up_exps_bytes: &[u8],
    down_exps_bytes: &[u8],
    per_gate_up: usize,
    gate_up_bytes_per_row: usize,
    per_down: usize,
    down_bytes_per_row: usize,
    n_ff: usize,
    hidden_dim: usize,
    down_quant: GGMLType,
    architecture: ModelArchitecture,
    profile_enabled: bool,
) -> QwenExpertGroupOutput {
    let group_slots = &slots[start..end];
    let group_len = group_slots.len();
    let expert = group_slots[0].expert;
    let q8k_blocks_per_row = hidden_dim / 256;

    let gather_start = profile_enabled.then(Instant::now);
    let mut expert_q8k = vec![QuantizedQ8KBlock::default(); group_len * q8k_blocks_per_row];
    for (group_row, slot) in group_slots.iter().enumerate() {
        let source_start = slot.token * q8k_blocks_per_row;
        let destination_start = group_row * q8k_blocks_per_row;
        expert_q8k[destination_start..destination_start + q8k_blocks_per_row]
            .copy_from_slice(&normalized_q8k[source_start..source_start + q8k_blocks_per_row]);
    }
    let gather_elapsed = gather_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let gate_up_start = profile_enabled.then(Instant::now);
    let gate_bytes = &gate_exps_bytes[expert * per_gate_up..(expert + 1) * per_gate_up];
    let up_bytes = &up_exps_bytes[expert * per_gate_up..(expert + 1) * per_gate_up];
    let mut gate_output = vec![0.0f32; group_len * n_ff];
    let mut up_output = vec![0.0f32; group_len * n_ff];
    prefill_raw_dual_q4k_q8k(
        gate_bytes,
        up_bytes,
        &expert_q8k,
        &mut gate_output,
        &mut up_output,
        n_ff,
        hidden_dim,
        group_len,
        gate_up_bytes_per_row,
    );
    let gate_up_elapsed = gate_up_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let activation_start = profile_enabled.then(Instant::now);
    apply_model_gate_mul_inplace(&mut gate_output, &up_output, architecture);
    let activation_elapsed = activation_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let down_start = profile_enabled.then(Instant::now);
    let down_bytes = &down_exps_bytes[expert * per_down..(expert + 1) * per_down];
    let mut output = vec![0.0f32; group_len * hidden_dim];
    prefill_raw_quantized_batch(
        down_bytes,
        &gate_output,
        &mut output,
        hidden_dim,
        n_ff,
        group_len,
        down_bytes_per_row,
        down_quant,
    );
    let down_elapsed = down_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    QwenExpertGroupOutput {
        start,
        end,
        output,
        gather_elapsed,
        gate_up_elapsed,
        activation_elapsed,
        down_elapsed,
    }
}
