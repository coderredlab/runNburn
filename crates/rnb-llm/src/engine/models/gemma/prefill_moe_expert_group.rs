//! Parallel execution of independent Gemma4 MoE expert groups on AArch64.

use super::prefill_moe_expert_major::GemmaRouteSlot;
use crate::engine::norm::apply_model_gate_mul_inplace;
use crate::engine::quantized_dispatch::{
    prefill_raw_quantized_batch, prefill_raw_split_q4k_q8k, QuantizedQ8KBlock,
};
use rayon::prelude::*;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};
use std::time::{Duration, Instant};

pub(super) struct GemmaExpertGroupOutput {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) output: Vec<f32>,
    pub(super) gather_elapsed: Duration,
    pub(super) gate_up_elapsed: Duration,
    pub(super) activation_elapsed: Duration,
    pub(super) down_elapsed: Duration,
}

fn expert_group_ranges(slots: &[GemmaRouteSlot]) -> Vec<(usize, usize)> {
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
    slots: &[GemmaRouteSlot],
    moe_input_q8k: &[QuantizedQ8KBlock],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    per_gate_up: usize,
    gate_up_bytes_per_row: usize,
    per_down: usize,
    down_bytes_per_row: usize,
    n_ff: usize,
    hidden_dim: usize,
    down_quant: GGMLType,
    profile_enabled: bool,
) -> crate::error::Result<Vec<GemmaExpertGroupOutput>> {
    expert_group_ranges(slots)
        .par_iter()
        .map(|&(start, end)| {
            compute_expert_group(
                slots,
                start,
                end,
                moe_input_q8k,
                gate_up_bytes,
                down_bytes,
                per_gate_up,
                gate_up_bytes_per_row,
                per_down,
                down_bytes_per_row,
                n_ff,
                hidden_dim,
                down_quant,
                profile_enabled,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_expert_group(
    slots: &[GemmaRouteSlot],
    start: usize,
    end: usize,
    moe_input_q8k: &[QuantizedQ8KBlock],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    per_gate_up: usize,
    gate_up_bytes_per_row: usize,
    per_down: usize,
    down_bytes_per_row: usize,
    n_ff: usize,
    hidden_dim: usize,
    down_quant: GGMLType,
    profile_enabled: bool,
) -> crate::error::Result<GemmaExpertGroupOutput> {
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
            .copy_from_slice(&moe_input_q8k[source_start..source_start + q8k_blocks_per_row]);
    }
    let gather_elapsed = gather_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let gate_up_start = profile_enabled.then(Instant::now);
    let gate_up_slice = &gate_up_bytes[expert * per_gate_up..(expert + 1) * per_gate_up];
    let mut expert_mid = vec![0.0f32; group_len * n_ff];
    let mut expert_up = vec![0.0f32; group_len * n_ff];
    prefill_raw_split_q4k_q8k(
        gate_up_slice,
        &expert_q8k,
        &mut expert_mid,
        &mut expert_up,
        n_ff,
        hidden_dim,
        group_len,
        gate_up_bytes_per_row,
    );
    let gate_up_elapsed = gate_up_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let activation_start = profile_enabled.then(Instant::now);
    apply_model_gate_mul_inplace(&mut expert_mid, &expert_up, ModelArchitecture::Gemma4);
    let activation_elapsed = activation_start
        .map(|started| started.elapsed())
        .unwrap_or_default();

    let down_start = profile_enabled.then(Instant::now);
    let down_slice = &down_bytes[expert * per_down..(expert + 1) * per_down];
    let mut output = vec![0.0f32; group_len * hidden_dim];
    prefill_raw_quantized_batch(
        down_slice,
        &expert_mid,
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

    Ok(GemmaExpertGroupOutput {
        start,
        end,
        output,
        gather_elapsed,
        gate_up_elapsed,
        activation_elapsed,
        down_elapsed,
    })
}
