//! Expert-major batched Gemma4 26B-A4B prefill.

use super::moe_view::select_experts_from_logits;
#[cfg(target_arch = "aarch64")]
use super::prefill_moe_expert_group::compute_expert_groups;
use super::select_ffn_pre_norm_weight;
use crate::engine::cpu_runtime::kernels;
use crate::engine::dense_dispatch;
use crate::engine::layer_weights::{AttentionLayerWeights, MoeLayerWeights};
use crate::engine::moe_profile::{
    is_enabled as moe_profile_enabled, record_moe_profile, record_moe_profile_by_layer,
};
use crate::engine::norm::{apply_model_gate_mul_inplace, apply_model_norm_into};
use crate::engine::policy;
use crate::engine::quantized_dispatch::{prefill_gate_up_vectors, prefill_raw_quantized_batch};
#[cfg(target_arch = "aarch64")]
use crate::engine::quantized_dispatch::{quantize_raw_q8k, QuantizedQ8KBlock};
use rnb_core::tensor::Tensor;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GemmaRouteSlot {
    pub(super) expert: usize,
    pub(super) token: usize,
    pub(super) rank: usize,
    pub(super) weight: f32,
}

pub(super) fn gemma4_moe_expert_major_enabled(moe_w: &MoeLayerWeights, seq_len: usize) -> bool {
    policy::gemma4_moe_expert_major_enabled()
        && seq_len > 1
        && !policy::moe_mixed_precision_requested()
        && moe_w.packed_model.is_none()
        && moe_w.gate_up_residency.is_none()
        && moe_w.down_residency.is_none()
}

fn gemma_expert_major_slots(
    logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
    layer_idx: Option<usize>,
) -> Vec<GemmaRouteSlot> {
    let mut slots = Vec::with_capacity(seq_len * n_expert_used);
    for token in 0..seq_len {
        let token_logits = &logits[token * n_expert..(token + 1) * n_expert];
        let (experts, weights) = select_experts_from_logits(token_logits, n_expert_used);
        if let Some(layer) = layer_idx {
            crate::engine::moe_trace::record_selection(layer, &experts);
        }
        for (rank, (&expert, &weight)) in experts.iter().zip(weights.iter()).enumerate() {
            slots.push(GemmaRouteSlot {
                expert,
                token,
                rank,
                weight,
            });
        }
    }
    slots.sort_unstable_by_key(|slot| (slot.expert, slot.token));
    slots
}

fn gemma_expert_group_shape(slots: &[GemmaRouteSlot]) -> (usize, usize) {
    let mut groups = 0usize;
    let mut max_group = 0usize;
    let mut start = 0usize;
    while start < slots.len() {
        let expert = slots[start].expert;
        let mut end = start + 1;
        while end < slots.len() && slots[end].expert == expert {
            end += 1;
        }
        groups += 1;
        max_group = max_group.max(end - start);
        start = end;
    }
    (groups, max_group)
}

fn scatter_weighted_expert_group(
    slots: &[GemmaRouteSlot],
    expert_output: &[f32],
    down_scale: f32,
    n_expert_used: usize,
    hidden_dim: usize,
    ranked_output: &mut [f32],
) {
    for (group_row, slot) in slots.iter().enumerate() {
        let source = &expert_output[group_row * hidden_dim..(group_row + 1) * hidden_dim];
        let route = slot.token * n_expert_used + slot.rank;
        let destination = &mut ranked_output[route * hidden_dim..(route + 1) * hidden_dim];
        let weight = slot.weight * down_scale;
        for (dst, &value) in destination.iter_mut().zip(source.iter()) {
            *dst = value * weight;
        }
    }
}

fn reduce_ranked_expert_output_into(
    ranked_output: &[f32],
    seq_len: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    reduced: &mut [f32],
) {
    reduced.fill(0.0);
    for token in 0..seq_len {
        let destination = &mut reduced[token * hidden_dim..(token + 1) * hidden_dim];
        for rank in 0..n_expert_used {
            let route = token * n_expert_used + rank;
            let source = &ranked_output[route * hidden_dim..(route + 1) * hidden_dim];
            for (dst, &value) in destination.iter_mut().zip(source.iter()) {
                *dst += value;
            }
        }
    }
}

#[inline]
fn record_expert_major_duration(
    key: &'static str,
    layer_idx: usize,
    stage: &'static str,
    elapsed: Duration,
) {
    record_moe_profile(key, elapsed);
    record_moe_profile_by_layer(
        "gemma4:prefill:expert_major",
        Some(layer_idx),
        stage,
        elapsed,
    );
}

#[inline]
fn finish_expert_major_stage(
    key: &'static str,
    layer_idx: usize,
    stage: &'static str,
    started: Option<Instant>,
) {
    if let Some(started) = started {
        record_expert_major_duration(key, layer_idx, stage, started.elapsed());
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_ffn_gemma4_moe_expert_major(
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    moe_w: &MoeLayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<Tensor> {
    let profile_enabled = moe_profile_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let setup_start = profile_enabled.then(Instant::now);
    let mut attn_out_data: Vec<f32> = kernels::tensor_as_f32_slice(&hidden).to_vec();
    let ffn_norm_data = kernels::tensor_as_f32_slice(select_ffn_pre_norm_weight(w, architecture));
    let post_norm_1_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_1);
    let pre_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.pre_ffw_norm_2);
    let post_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_2);
    let router_scale_data = kernels::tensor_as_f32_slice(&moe_w.router_scale);
    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("router_f32 missing".into()))?;
    let gate_up_bytes = moe_w
        .gate_up_exps
        .as_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("gate_up bytes missing".into()))?;
    let down_bytes = moe_w
        .down_exps
        .as_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("down bytes missing".into()))?;
    let down_scale = kernels::tensor_as_f32_slice(&moe_w.down_scale);
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:setup",
        layer_idx,
        "setup",
        setup_start,
    );
    let norms_start = profile_enabled.then(Instant::now);

    let mut shared_norm = vec![0.0f32; seq_len * hidden_dim];
    let mut moe_input = vec![0.0f32; seq_len * hidden_dim];
    let mut router_input = vec![0.0f32; seq_len * hidden_dim];
    let inv_sqrt_n = 1.0 / (hidden_dim as f32).sqrt();
    for token in 0..seq_len {
        let attn_out = &attn_out_data[token * hidden_dim..(token + 1) * hidden_dim];
        apply_model_norm_into(
            attn_out,
            ffn_norm_data,
            norm_eps,
            &mut shared_norm[token * hidden_dim..(token + 1) * hidden_dim],
            architecture,
        );
        apply_model_norm_into(
            attn_out,
            pre_norm_2_data,
            norm_eps,
            &mut moe_input[token * hidden_dim..(token + 1) * hidden_dim],
            architecture,
        );

        let sum_sq: f32 = attn_out.iter().map(|value| value * value).sum();
        let rrms = 1.0 / (sum_sq / hidden_dim as f32 + norm_eps).sqrt();
        let router_row = &mut router_input[token * hidden_dim..(token + 1) * hidden_dim];
        for i in 0..hidden_dim {
            router_row[i] = attn_out[i] * rrms * inv_sqrt_n * router_scale_data[i];
        }
    }
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:norms",
        layer_idx,
        "norms",
        norms_start,
    );
    #[cfg(target_arch = "aarch64")]
    let moe_input_q8k = quantize_raw_q8k(&moe_input);
    #[cfg(target_arch = "aarch64")]
    drop(moe_input);
    let shared_start = profile_enabled.then(Instant::now);

    let (mut shared_gate, shared_up) = prefill_gate_up_vectors(
        &w.ffn_gate_weight,
        &w.ffn_up_weight,
        w.ffn_gate_up_fused.as_ref(),
        &shared_norm,
        seq_len,
    )?;
    apply_model_gate_mul_inplace(&mut shared_gate, &shared_up, architecture);
    let shared_down = w.ffn_down_weight.gemv_vec(&shared_gate)?;
    let mut shared_output = vec![0.0f32; seq_len * hidden_dim];
    for token in 0..seq_len {
        apply_model_norm_into(
            &shared_down[token * hidden_dim..(token + 1) * hidden_dim],
            post_norm_1_data,
            norm_eps,
            &mut shared_output[token * hidden_dim..(token + 1) * hidden_dim],
            architecture,
        );
    }
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:shared",
        layer_idx,
        "shared",
        shared_start,
    );
    let routing_start = profile_enabled.then(Instant::now);

    let mut router_logits = vec![0.0f32; seq_len * moe_w.n_expert];
    dense_dispatch::gemv_f32(
        router_w_data,
        &router_input,
        &mut router_logits,
        moe_w.n_expert,
        hidden_dim,
        seq_len,
    );
    let slots = gemma_expert_major_slots(
        &router_logits,
        seq_len,
        moe_w.n_expert,
        moe_w.n_expert_used,
        Some(layer_idx),
    );
    let (group_count, max_group) = gemma_expert_group_shape(&slots);
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:routing",
        layer_idx,
        "routing",
        routing_start,
    );
    let scratch_start = profile_enabled.then(Instant::now);

    let gate_up_rows = moe_w.n_ff * 2;
    let per_gate_up = gate_up_bytes.len() / moe_w.n_expert;
    let gate_up_bytes_per_row = per_gate_up / gate_up_rows;
    let per_down = down_bytes.len() / moe_w.n_expert;
    let down_bytes_per_row = per_down / hidden_dim;

    #[cfg(not(target_arch = "aarch64"))]
    let mut expert_input = vec![0.0f32; max_group * hidden_dim];
    #[cfg(not(target_arch = "aarch64"))]
    let mut gate_up_output = vec![0.0f32; max_group * gate_up_rows];
    #[cfg(not(target_arch = "aarch64"))]
    let mut expert_mid = vec![0.0f32; max_group * moe_w.n_ff];
    #[cfg(not(target_arch = "aarch64"))]
    let mut expert_output = vec![0.0f32; max_group * hidden_dim];
    let mut ranked_output = vec![0.0f32; slots.len() * hidden_dim];
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:scratch",
        layer_idx,
        "scratch",
        scratch_start,
    );
    let mut gather_elapsed = Duration::ZERO;
    let mut gate_up_elapsed = Duration::ZERO;
    let mut activation_elapsed = Duration::ZERO;
    let mut down_elapsed = Duration::ZERO;
    let mut scatter_elapsed = Duration::ZERO;

    #[cfg(target_arch = "aarch64")]
    {
        let group_results = compute_expert_groups(
            &slots,
            &moe_input_q8k,
            gate_up_bytes,
            down_bytes,
            per_gate_up,
            gate_up_bytes_per_row,
            per_down,
            down_bytes_per_row,
            moe_w.n_ff,
            hidden_dim,
            moe_w.down_quant,
            profile_enabled,
        )?;
        for group in group_results {
            gather_elapsed += group.gather_elapsed;
            gate_up_elapsed += group.gate_up_elapsed;
            activation_elapsed += group.activation_elapsed;
            down_elapsed += group.down_elapsed;

            let scatter_start = profile_enabled.then(Instant::now);
            let expert = slots[group.start].expert;
            scatter_weighted_expert_group(
                &slots[group.start..group.end],
                &group.output,
                down_scale[expert],
                moe_w.n_expert_used,
                hidden_dim,
                &mut ranked_output,
            );
            if let Some(scatter_start) = scatter_start {
                scatter_elapsed += scatter_start.elapsed();
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut start = 0usize;
        while start < slots.len() {
            let expert = slots[start].expert;
            let mut end = start + 1;
            while end < slots.len() && slots[end].expert == expert {
                end += 1;
            }
            let group_slots = &slots[start..end];
            let group_len = group_slots.len();

            let gather_start = profile_enabled.then(Instant::now);
            for (group_row, slot) in group_slots.iter().enumerate() {
                expert_input[group_row * hidden_dim..(group_row + 1) * hidden_dim].copy_from_slice(
                    &moe_input[slot.token * hidden_dim..(slot.token + 1) * hidden_dim],
                );
            }
            if let Some(gather_start) = gather_start {
                gather_elapsed += gather_start.elapsed();
            }

            let gate_up_start = profile_enabled.then(Instant::now);
            let gate_up_slice = &gate_up_bytes[expert * per_gate_up..(expert + 1) * per_gate_up];
            prefill_raw_quantized_batch(
                gate_up_slice,
                &expert_input[..group_len * hidden_dim],
                &mut gate_up_output[..group_len * gate_up_rows],
                gate_up_rows,
                hidden_dim,
                group_len,
                gate_up_bytes_per_row,
                GGMLType::Q4_K,
            );
            if let Some(gate_up_start) = gate_up_start {
                gate_up_elapsed += gate_up_start.elapsed();
            }
            let activation_start = profile_enabled.then(Instant::now);
            for group_row in 0..group_len {
                let gate_up_row =
                    &mut gate_up_output[group_row * gate_up_rows..(group_row + 1) * gate_up_rows];
                let (gate, up) = gate_up_row.split_at_mut(moe_w.n_ff);
                apply_model_gate_mul_inplace(gate, up, ModelArchitecture::Gemma4);
                expert_mid[group_row * moe_w.n_ff..(group_row + 1) * moe_w.n_ff]
                    .copy_from_slice(gate);
            }
            if let Some(activation_start) = activation_start {
                activation_elapsed += activation_start.elapsed();
            }

            let down_start = profile_enabled.then(Instant::now);
            let down_slice = &down_bytes[expert * per_down..(expert + 1) * per_down];
            prefill_raw_quantized_batch(
                down_slice,
                &expert_mid[..group_len * moe_w.n_ff],
                &mut expert_output[..group_len * hidden_dim],
                hidden_dim,
                moe_w.n_ff,
                group_len,
                down_bytes_per_row,
                moe_w.down_quant,
            );
            if let Some(down_start) = down_start {
                down_elapsed += down_start.elapsed();
            }
            let scatter_start = profile_enabled.then(Instant::now);
            scatter_weighted_expert_group(
                group_slots,
                &expert_output[..group_len * hidden_dim],
                down_scale[expert],
                moe_w.n_expert_used,
                hidden_dim,
                &mut ranked_output,
            );
            if let Some(scatter_start) = scatter_start {
                scatter_elapsed += scatter_start.elapsed();
            }
            start = end;
        }
    }
    if profile_enabled {
        record_expert_major_duration(
            "gemma4:prefill:expert_major:gather",
            layer_idx,
            "gather",
            gather_elapsed,
        );
        record_expert_major_duration(
            "gemma4:prefill:expert_major:gate_up",
            layer_idx,
            "gate_up",
            gate_up_elapsed,
        );
        record_expert_major_duration(
            "gemma4:prefill:expert_major:activation",
            layer_idx,
            "activation",
            activation_elapsed,
        );
        record_expert_major_duration(
            "gemma4:prefill:expert_major:down",
            layer_idx,
            "down",
            down_elapsed,
        );
        record_expert_major_duration(
            "gemma4:prefill:expert_major:scatter",
            layer_idx,
            "scatter",
            scatter_elapsed,
        );
    }

    if policy::profiling_enabled() {
        #[cfg(not(target_arch = "aarch64"))]
        let scratch_bytes = expert_input.len() * std::mem::size_of::<f32>()
            + gate_up_output.len() * std::mem::size_of::<f32>()
            + (expert_mid.len() + expert_output.len() + ranked_output.len())
                * std::mem::size_of::<f32>();
        #[cfg(target_arch = "aarch64")]
        let scratch_bytes =
            max_group * (hidden_dim / 256) * std::mem::size_of::<QuantizedQ8KBlock>()
                + max_group * (moe_w.n_ff * 2 + hidden_dim) * std::mem::size_of::<f32>()
                + ranked_output.len() * std::mem::size_of::<f32>();
        eprintln!(
            "  [GEMMA L{layer_idx}] moe_expert_major routes={} groups={} max_group={} scratch_mib={:.1}",
            slots.len(),
            group_count,
            max_group,
            scratch_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    let finalize_start = profile_enabled.then(Instant::now);
    reduce_ranked_expert_output_into(
        &ranked_output,
        seq_len,
        moe_w.n_expert_used,
        hidden_dim,
        &mut router_input,
    );
    let post_ffw_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    for token in 0..seq_len {
        let token_range = token * hidden_dim..(token + 1) * hidden_dim;
        apply_model_norm_into(
            &router_input[token_range.clone()],
            post_norm_2_data,
            norm_eps,
            &mut shared_norm[token_range.clone()],
            architecture,
        );
        kernels::elementwise::add_inplace(
            &mut shared_output[token_range.clone()],
            &shared_norm[token_range.clone()],
        );
        if let Some(post_norm) = post_ffw_norm {
            apply_model_norm_into(
                &shared_output[token_range.clone()],
                post_norm,
                norm_eps,
                &mut router_input[token_range.clone()],
                architecture,
            );
            kernels::elementwise::add_inplace(
                &mut attn_out_data[token_range.clone()],
                &router_input[token_range],
            );
        } else {
            kernels::elementwise::add_inplace(
                &mut attn_out_data[token_range.clone()],
                &shared_output[token_range],
            );
        }
    }

    let result = Ok(Tensor::from_vec(attn_out_data, &[seq_len, hidden_dim]));
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:finalize",
        layer_idx,
        "finalize",
        finalize_start,
    );
    finish_expert_major_stage(
        "gemma4:prefill:expert_major:total",
        layer_idx,
        "total",
        total_start,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_major_slots_preserve_token_rank_and_softmax() {
        let logits = [5.0, 1.0, 4.0, 3.0, 0.0, 6.0, 5.0, 2.0];
        let slots = gemma_expert_major_slots(&logits, 2, 4, 3, None);
        assert_eq!(slots.len(), 6);
        assert!(slots
            .windows(2)
            .all(|pair| { (pair[0].expert, pair[0].token) <= (pair[1].expert, pair[1].token) }));

        for token in 0..2 {
            let (expected_experts, expected_weights) =
                select_experts_from_logits(&logits[token * 4..(token + 1) * 4], 3);
            let mut token_slots: Vec<_> = slots.iter().filter(|slot| slot.token == token).collect();
            token_slots.sort_unstable_by_key(|slot| slot.rank);
            for rank in 0..3 {
                assert_eq!(token_slots[rank].expert, expected_experts[rank]);
                assert_eq!(token_slots[rank].weight, expected_weights[rank]);
            }
            let weight_sum: f32 = token_slots.iter().map(|slot| slot.weight).sum();
            assert!((weight_sum - 1.0).abs() <= f32::EPSILON * 4.0);
        }
    }

    #[test]
    fn expert_major_scatter_reduces_in_original_rank_order() {
        let expert_sorted_slots = [
            GemmaRouteSlot {
                expert: 0,
                token: 0,
                rank: 2,
                weight: 1.0,
            },
            GemmaRouteSlot {
                expert: 1,
                token: 0,
                rank: 0,
                weight: 1.0,
            },
            GemmaRouteSlot {
                expert: 2,
                token: 0,
                rank: 1,
                weight: 1.0,
            },
        ];
        let mut ranked = vec![0.0f32; 3];
        scatter_weighted_expert_group(&expert_sorted_slots[0..1], &[3.0], 1.0, 3, 1, &mut ranked);
        scatter_weighted_expert_group(
            &expert_sorted_slots[1..2],
            &[1.0e20],
            1.0,
            3,
            1,
            &mut ranked,
        );
        scatter_weighted_expert_group(
            &expert_sorted_slots[2..3],
            &[-1.0e20],
            1.0,
            3,
            1,
            &mut ranked,
        );

        assert_eq!(ranked, [1.0e20, -1.0e20, 3.0]);
        let mut reduced = vec![0.0f32; 1];
        reduce_ranked_expert_output_into(&ranked, 1, 3, 1, &mut reduced);
        assert_eq!(reduced, [3.0]);

        let second_ranked = [1.0, 2.0, 3.0];
        reduce_ranked_expert_output_into(&second_ranked, 1, 3, 1, &mut reduced);
        assert_eq!(reduced, [6.0]);
    }
}
