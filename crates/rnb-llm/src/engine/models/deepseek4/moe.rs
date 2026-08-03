use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::dequant::dequantize_bytes_to_f32;
use crate::engine::scalar_gemv::{
    gemv_host_quantized, gemv_host_quantized_batch, host_quant_gemv_supported,
};
#[cfg(feature = "cuda")]
use crate::error::LlmError;
use crate::error::Result;
use rayon::prelude::*;
use rnb_loader::convert::ggml_quant_params;
use rnb_loader::GGMLType;

use super::math::tensor_f32;
use super::weights::{DeepSeek4Config, DeepSeek4MoeWeights};
fn metal_decode_route_supported(expert_slots: usize, route_weight_slots: usize) -> bool {
    (1..=8).contains(&expert_slots) && expert_slots == route_weight_slots
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MoeBatchMode {
    /// Large-batch prefill: grouped CUDA path with temp uploads.
    Prefill,
    /// Speculative verify: deterministic resident CUDA path, host fallback.
    Verify,
    /// DSpark draft trunk: resident CUDA path, then prefill-style fallback.
    Draft,
}

pub(super) fn forward_moe(
    input: &[f32],
    token_id: u32,
    layer_idx: usize,
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let (experts, route_weights) = route(input, token_id, weights, config);
    crate::engine::moe_trace::record_selection(layer_idx, &experts);
    if let Some(output) =
        forward_moe_metal_decode(input, &experts, &route_weights, weights, config)?
    {
        return Ok(output);
    }
    #[cfg(feature = "cuda")]
    if routed_cuda_supported(weights, config) {
        let routes = [(experts.clone(), route_weights.clone())];
        if let Some(sparse) = compute_sparse_experts_cuda_resident(input, &routes, weights, config)?
        {
            let shared = &weights.weights;
            let mut shared_gate = shared.shared_gate.gemv_vec(input)?;
            let mut shared_up = shared.shared_up.gemv_vec(input)?;
            swiglu_clamped(&mut shared_gate, &mut shared_up, weights.shared_clamp);
            let mut output = shared.shared_down.gemv_vec(&shared_gate)?;
            for (dst, value) in output.iter_mut().zip(sparse) {
                *dst += value;
            }
            return Ok(output);
        }
    }
    let sparse_outputs: Vec<Vec<f32>> = experts
        .par_iter()
        .zip(route_weights.par_iter())
        .map(|(&expert, &route_weight)| {
            compute_sparse_expert(input, expert, route_weight, weights, config)
        })
        .collect();

    let shared = &weights.weights;
    let mut shared_gate = shared.shared_gate.gemv_vec(input)?;
    let mut shared_up = shared.shared_up.gemv_vec(input)?;
    swiglu_clamped(&mut shared_gate, &mut shared_up, weights.shared_clamp);
    let shared_output = shared.shared_down.gemv_vec(&shared_gate)?;

    let mut output = shared_output;
    for expert_output in sparse_outputs {
        for (dst, value) in output.iter_mut().zip(expert_output) {
            *dst += value;
        }
    }
    Ok(output)
}

fn forward_moe_metal_decode(
    input: &[f32],
    experts: &[usize],
    route_weights: &[f32],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Option<Vec<f32>>> {
    if !crate::engine::backend_runtime::metal_deepseek4_moe_decode_requested() {
        return Ok(None);
    }

    let moe = &weights.weights;
    let supported = metal_decode_route_supported(experts.len(), route_weights.len())
        && moe.gate_quant == moe.up_quant
        && matches!(
            moe.gate_quant,
            GGMLType::IQ2_XXS | GGMLType::IQ2_S | GGMLType::IQ3_XXS
        )
        && matches!(moe.down_quant, GGMLType::IQ3_XXS | GGMLType::IQ4_XS)
        && moe.shared_gate.ggml_type == moe.shared_up.ggml_type
        && matches!(
            moe.shared_gate.ggml_type,
            GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_0
        )
        && matches!(moe.shared_down.ggml_type, GGMLType::Q6_K | GGMLType::Q8_0)
        && moe.shared_gate.rows == config.expert_ffn_dim
        && moe.shared_up.rows == config.expert_ffn_dim
        && moe.shared_down.rows == config.hidden_dim
        && moe.shared_gate.cols == config.hidden_dim
        && moe.shared_up.cols == config.hidden_dim
        && moe.shared_down.cols == config.expert_ffn_dim
        && config.hidden_dim % 256 == 0
        && config.expert_ffn_dim % 256 == 0;
    if !supported {
        return Ok(None);
    }

    let Some(gate_bytes) = moe.gate_exps_bytes() else {
        return Ok(None);
    };
    let Some(up_bytes) = moe.up_exps_bytes() else {
        return Ok(None);
    };
    let Some(down_bytes) = moe.down_exps_bytes() else {
        return Ok(None);
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe.shared_gate.data.as_bytes(),
        moe.shared_up.data.as_bytes(),
        moe.shared_down.data.as_bytes(),
    ) else {
        return Ok(None);
    };

    let gate_expert_bytes =
        config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_expert_bytes = config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_expert_bytes =
        config.hidden_dim * bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let mut gate_slots = Vec::with_capacity(experts.len());
    let mut up_slots = Vec::with_capacity(experts.len());
    let mut down_slots = Vec::with_capacity(experts.len());
    for &expert in experts {
        let gate_start = expert * gate_expert_bytes;
        let up_start = expert * up_expert_bytes;
        let down_start = expert * down_expert_bytes;
        gate_slots.push(&gate_bytes[gate_start..gate_start + gate_expert_bytes]);
        up_slots.push(&up_bytes[up_start..up_start + up_expert_bytes]);
        down_slots.push(&down_bytes[down_start..down_start + down_expert_bytes]);
    }

    let mut activation_limits = vec![weights.routed_clamp; experts.len()];
    activation_limits.push(weights.shared_clamp);
    let mut output = vec![0.0f32; config.hidden_dim];
    let used = crate::engine::backend_runtime::glm_moe_decode_iq2xxs_iq3xxs_into(
        &gate_slots,
        &up_slots,
        &down_slots,
        route_weights,
        shared_gate,
        shared_up,
        shared_down,
        1.0,
        config.expert_ffn_dim,
        config.hidden_dim,
        input,
        &mut output,
        moe.gate_quant == GGMLType::IQ2_S,
        moe.down_quant == GGMLType::IQ4_XS,
        moe.shared_gate.ggml_type == GGMLType::Q6_K,
        moe.shared_down.ggml_type == GGMLType::Q8_0,
        moe.gate_quant == GGMLType::IQ3_XXS,
        moe.shared_gate.ggml_type == GGMLType::Q8_0,
        Some(&activation_limits),
        true,
    )
    .map_err(crate::error::LlmError::Forward)?;
    Ok(used.then_some(output))
}
pub(super) fn forward_moe_batch(
    inputs: &[f32],
    token_ids: &[u32],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    forward_moe_batch_impl(inputs, token_ids, weights, config, MoeBatchMode::Prefill)
}

pub(super) fn forward_moe_verify_batch(
    inputs: &[f32],
    token_ids: &[u32],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    // Verify must match tokenwise decode bit-for-bit. Both share the
    // deterministic resident CUDA path (per-token slot-ordered kernels);
    // when the resident cache is unavailable both fall back to host
    // arithmetic with the same per-token reduction order.
    forward_moe_batch_impl(inputs, token_ids, weights, config, MoeBatchMode::Verify)
}

pub(super) fn forward_moe_draft_batch(
    inputs: &[f32],
    token_ids: &[u32],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    forward_moe_batch_impl(inputs, token_ids, weights, config, MoeBatchMode::Draft)
}

fn forward_moe_batch_impl(
    inputs: &[f32],
    token_ids: &[u32],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
    mode: MoeBatchMode,
) -> Result<Vec<f32>> {
    #[cfg(not(feature = "cuda"))]
    let _ = mode;
    let seq_len = token_ids.len();
    debug_assert_eq!(inputs.len(), seq_len * config.hidden_dim);
    let routes: Vec<(Vec<usize>, Vec<f32>)> = inputs
        .par_chunks_exact(config.hidden_dim)
        .zip(token_ids.par_iter())
        .map(|(input, &token_id)| route(input, token_id, weights, config))
        .collect();

    #[cfg(feature = "cuda")]
    if routed_cuda_supported(weights, config) {
        let sparse = match mode {
            MoeBatchMode::Verify | MoeBatchMode::Draft => {
                compute_sparse_experts_cuda_resident(inputs, &routes, weights, config)?
            }
            MoeBatchMode::Prefill => None,
        };
        let sparse = match (sparse, mode) {
            (Some(sparse), _) => Some(sparse),
            // Verify must stay on the same arithmetic as tokenwise decode;
            // without the resident cache both drop to the host path below.
            (None, MoeBatchMode::Verify) => None,
            (None, MoeBatchMode::Prefill | MoeBatchMode::Draft) => Some(
                compute_sparse_experts_cuda_batch(inputs, &routes, weights, config)?,
            ),
        };
        if let Some(sparse) = sparse {
            let shared = &weights.weights;
            let mut shared_gate = shared.shared_gate.gemv_vec(inputs)?;
            let mut shared_up = shared.shared_up.gemv_vec(inputs)?;
            swiglu_clamped(&mut shared_gate, &mut shared_up, weights.shared_clamp);
            let mut output = shared.shared_down.gemv_vec(&shared_gate)?;
            for (dst, sparse) in output.iter_mut().zip(sparse) {
                *dst += sparse;
            }
            return Ok(output);
        }
    }

    if let Some(output) = forward_moe_metal_batch(inputs, &routes, weights, config)? {
        return Ok(output);
    }

    let mut assignments = vec![Vec::<(usize, usize, f32)>::new(); config.expert_count];
    for (token, (experts, route_weights)) in routes.iter().enumerate() {
        for (slot, (&expert, &route_weight)) in experts.iter().zip(route_weights).enumerate() {
            assignments[expert].push((token, slot, route_weight));
        }
    }

    let sparse_groups: Vec<(Vec<(usize, usize, f32)>, Vec<f32>)> = assignments
        .into_par_iter()
        .enumerate()
        .filter(|(_, assignments)| !assignments.is_empty())
        .map(|(expert, assignments)| {
            let group_len = assignments.len();
            let mut expert_inputs = Vec::with_capacity(group_len * config.hidden_dim);
            for &(token, _, _) in &assignments {
                let start = token * config.hidden_dim;
                expert_inputs.extend_from_slice(&inputs[start..start + config.hidden_dim]);
            }
            let expert_output =
                compute_sparse_expert_batch(&expert_inputs, expert, group_len, weights, config);
            (assignments, expert_output)
        })
        .collect();

    let shared = &weights.weights;
    let mut shared_gate = shared.shared_gate.gemv_vec(inputs)?;
    let mut shared_up = shared.shared_up.gemv_vec(inputs)?;
    swiglu_clamped(&mut shared_gate, &mut shared_up, weights.shared_clamp);
    let mut output = shared.shared_down.gemv_vec(&shared_gate)?;

    let slot_stride = config.expert_used_count * config.hidden_dim;
    let mut sparse_by_slot = vec![0.0f32; seq_len * slot_stride];
    for (assignments, expert_output) in sparse_groups {
        for (group_row, &(token, slot, route_weight)) in assignments.iter().enumerate() {
            let src_start = group_row * config.hidden_dim;
            let dst_start = token * slot_stride + slot * config.hidden_dim;
            for (dst, &value) in sparse_by_slot[dst_start..dst_start + config.hidden_dim]
                .iter_mut()
                .zip(&expert_output[src_start..src_start + config.hidden_dim])
            {
                *dst = value * route_weight;
            }
        }
    }
    for token in 0..seq_len {
        let output_row = &mut output[token * config.hidden_dim..(token + 1) * config.hidden_dim];
        for slot in 0..config.expert_used_count {
            let start = token * slot_stride + slot * config.hidden_dim;
            for (dst, &value) in output_row
                .iter_mut()
                .zip(&sparse_by_slot[start..start + config.hidden_dim])
            {
                *dst += value;
            }
        }
    }
    Ok(output)
}

fn forward_moe_metal_batch(
    inputs: &[f32],
    routes: &[(Vec<usize>, Vec<f32>)],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Option<Vec<f32>>> {
    if !crate::engine::backend_runtime::metal_deepseek4_moe_prefill_batch_requested() {
        return Ok(None);
    }

    let moe = &weights.weights;
    let sparse_gate_up_supported = moe.gate_quant == moe.up_quant
        && matches!(
            moe.gate_quant,
            GGMLType::IQ2_XXS | GGMLType::IQ2_S | GGMLType::IQ3_XXS
        );
    let sparse_down_supported = matches!(moe.down_quant, GGMLType::IQ3_XXS | GGMLType::IQ4_XS);
    let shared_gate_up_supported = moe.shared_gate.ggml_type == moe.shared_up.ggml_type
        && matches!(
            moe.shared_gate.ggml_type,
            GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_0
        );
    let shared_down_supported =
        matches!(moe.shared_down.ggml_type, GGMLType::Q6_K | GGMLType::Q8_0);
    let shared_shape_matches = moe.shared_gate.rows == config.expert_ffn_dim
        && moe.shared_up.rows == config.expert_ffn_dim
        && moe.shared_down.rows == config.hidden_dim
        && moe.shared_gate.cols == config.hidden_dim
        && moe.shared_up.cols == config.hidden_dim
        && moe.shared_down.cols == config.expert_ffn_dim;
    if !sparse_gate_up_supported
        || !sparse_down_supported
        || !shared_gate_up_supported
        || !shared_down_supported
        || !shared_shape_matches
        || config.hidden_dim % 256 != 0
        || config.expert_ffn_dim % 256 != 0
    {
        return Ok(None);
    }

    let Some(gate_bytes) = moe.gate_exps_bytes() else {
        return Ok(None);
    };
    let Some(up_bytes) = moe.up_exps_bytes() else {
        return Ok(None);
    };
    let Some(down_bytes) = moe.down_exps_bytes() else {
        return Ok(None);
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe.shared_gate.data.as_bytes(),
        moe.shared_up.data.as_bytes(),
        moe.shared_down.data.as_bytes(),
    ) else {
        return Ok(None);
    };

    let sparse_slots = config.expert_used_count;
    let slots = sparse_slots + 1;
    let gate_expert_bytes =
        config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_expert_bytes = config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_expert_bytes =
        config.hidden_dim * bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let mut gate_slots = Vec::with_capacity(routes.len() * slots);
    let mut up_slots = Vec::with_capacity(routes.len() * slots);
    let mut down_slots = Vec::with_capacity(routes.len() * slots);
    let mut route_weights = Vec::with_capacity(routes.len() * slots);
    for (experts, weights) in routes {
        for (&expert, &route_weight) in experts.iter().zip(weights) {
            let gate_start = expert * gate_expert_bytes;
            let up_start = expert * up_expert_bytes;
            let down_start = expert * down_expert_bytes;
            gate_slots.push(&gate_bytes[gate_start..gate_start + gate_expert_bytes]);
            up_slots.push(&up_bytes[up_start..up_start + up_expert_bytes]);
            down_slots.push(&down_bytes[down_start..down_start + down_expert_bytes]);
            route_weights.push(route_weight);
        }
        gate_slots.push(shared_gate);
        up_slots.push(shared_up);
        down_slots.push(shared_down);
        route_weights.push(1.0);
    }

    let mut activation_limits = vec![weights.routed_clamp; sparse_slots];
    activation_limits.push(weights.shared_clamp);
    let mut output = vec![0.0f32; routes.len() * config.hidden_dim];
    let file_regions = moe.sparse_expert_file_regions();
    let used = crate::engine::backend_runtime::glm_moe_prefill_iq_batch_into(
        &gate_slots,
        &up_slots,
        &down_slots,
        &route_weights,
        routes.len(),
        sparse_slots,
        config.expert_ffn_dim,
        config.hidden_dim,
        inputs,
        &mut output,
        moe.gate_quant == GGMLType::IQ2_S,
        moe.down_quant == GGMLType::IQ4_XS,
        moe.shared_gate.ggml_type == GGMLType::Q6_K,
        moe.shared_down.ggml_type == GGMLType::Q8_0,
        moe.gate_quant == GGMLType::IQ3_XXS,
        moe.shared_gate.ggml_type == GGMLType::Q8_0,
        Some(&activation_limits),
        true,
        file_regions.as_ref(),
    )
    .map_err(crate::error::LlmError::Forward)?;
    Ok(used.then_some(output))
}

fn route(
    input: &[f32],
    token_id: u32,
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> (Vec<usize>, Vec<f32>) {
    let mut logits = vec![0.0f32; config.expert_count];
    gemv_f32(
        tensor_f32(&weights.weights.router_w),
        input,
        &mut logits,
        config.expert_count,
        config.hidden_dim,
        1,
    );
    let scores: Vec<f32> = logits
        .into_iter()
        .map(|logit| softplus(logit).sqrt())
        .collect();

    let experts = if let Some(hash_routes) = &weights.hash_routes {
        let start = token_id as usize * config.expert_used_count;
        hash_routes[start..start + config.expert_used_count]
            .iter()
            .map(|&expert| expert as usize)
            .collect::<Vec<_>>()
    } else {
        let bias = tensor_f32(
            weights
                .weights
                .router_selection_bias
                .as_ref()
                .expect("DeepSeek4 score router requires selection bias"),
        );
        let mut candidates: Vec<usize> = (0..config.expert_count).collect();
        candidates.select_nth_unstable_by(config.expert_used_count, |&left, &right| {
            (scores[right] + bias[right])
                .partial_cmp(&(scores[left] + bias[left]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(config.expert_used_count);
        candidates.sort_unstable_by(|&left, &right| {
            (scores[right] + bias[right])
                .partial_cmp(&(scores[left] + bias[left]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    };
    let selected_sum = experts.iter().map(|&expert| scores[expert]).sum::<f32>();
    let route_weights = experts
        .iter()
        .map(|&expert| scores[expert] / selected_sum * config.expert_scale)
        .collect();
    (experts, route_weights)
}

#[inline]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(feature = "cuda")]
fn routed_cuda_supported(weights: &DeepSeek4MoeWeights, config: &DeepSeek4Config) -> bool {
    let moe = &weights.weights;
    routed_cuda_layout_supported(
        weights.prefer_sparse_moe_cuda,
        moe.gate_quant,
        moe.up_quant,
        moe.down_quant,
        config.hidden_dim,
        config.expert_ffn_dim,
    )
}

#[cfg(feature = "cuda")]
fn routed_cuda_layout_supported(
    prefer_sparse_moe_cuda: bool,
    gate_quant: GGMLType,
    up_quant: GGMLType,
    down_quant: GGMLType,
    hidden_dim: usize,
    expert_ffn_dim: usize,
) -> bool {
    prefer_sparse_moe_cuda
        && ((gate_quant == GGMLType::IQ2_XXS
            && up_quant == GGMLType::IQ2_XXS
            && down_quant == GGMLType::IQ3_XXS)
            || (gate_quant == GGMLType::MXFP4
                && up_quant == GGMLType::MXFP4
                && down_quant == GGMLType::MXFP4))
        && hidden_dim % 256 == 0
        && expert_ffn_dim % 256 == 0
}

#[cfg(feature = "cuda")]
fn compute_sparse_experts_cuda_batch(
    inputs: &[f32],
    routes: &[(Vec<usize>, Vec<f32>)],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let moe = &weights.weights;
    let gate_expert_bytes =
        config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_expert_bytes = config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_expert_bytes =
        config.hidden_dim * bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let gate_bytes = moe.gate_exps_bytes().expect("DeepSeek4 gate expert bytes");
    let up_bytes = moe.up_exps_bytes().expect("DeepSeek4 up expert bytes");
    let down_bytes = moe.down_exps_bytes().expect("DeepSeek4 down expert bytes");
    let slot_count = routes.len() * config.expert_used_count;
    let mut gate_slots = Vec::with_capacity(slot_count);
    let mut up_slots = Vec::with_capacity(slot_count);
    let mut down_slots = Vec::with_capacity(slot_count);
    let mut route_weights = Vec::with_capacity(slot_count);
    let mut token_ids = Vec::with_capacity(slot_count);
    for (token, (experts, weights)) in routes.iter().enumerate() {
        for (&expert, &route_weight) in experts.iter().zip(weights) {
            let gate_start = expert * gate_expert_bytes;
            let up_start = expert * up_expert_bytes;
            let down_start = expert * down_expert_bytes;
            gate_slots.push(&gate_bytes[gate_start..gate_start + gate_expert_bytes]);
            up_slots.push(&up_bytes[up_start..up_start + up_expert_bytes]);
            down_slots.push(&down_bytes[down_start..down_start + down_expert_bytes]);
            route_weights.push(route_weight);
            token_ids.push(token as u32);
        }
    }
    let output = if moe.gate_quant == GGMLType::MXFP4 {
        crate::engine::backend_runtime::mxfp4_sparse_experts_by_token_clamped_swiglu(
            &gate_slots,
            &up_slots,
            &down_slots,
            &route_weights,
            &token_ids,
            routes.len(),
            config.expert_ffn_dim,
            config.hidden_dim,
            inputs,
            weights.routed_clamp,
        )
    } else {
        crate::engine::backend_runtime::moe_prefill_sparse_experts_iq2xxs_iq3xxs_clamped_swiglu(
            &gate_slots,
            &up_slots,
            &down_slots,
            &route_weights,
            &token_ids,
            routes.len(),
            config.expert_ffn_dim,
            config.hidden_dim,
            inputs,
            weights.routed_clamp,
        )
    };
    output.map_err(LlmError::Forward)
}

/// Deterministic selected-expert forward through the CUDA resident slice
/// cache. `Ok(None)` means the cache is unavailable; callers must fall back
/// consistently for decode and verify so both stay on identical arithmetic.
#[cfg(feature = "cuda")]
fn compute_sparse_experts_cuda_resident(
    inputs: &[f32],
    routes: &[(Vec<usize>, Vec<f32>)],
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Option<Vec<f32>>> {
    let moe = &weights.weights;
    let gate_expert_bytes =
        config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_expert_bytes = config.expert_ffn_dim * bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_expert_bytes =
        config.hidden_dim * bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let gate_bytes = moe.gate_exps_bytes().expect("DeepSeek4 gate expert bytes");
    let up_bytes = moe.up_exps_bytes().expect("DeepSeek4 up expert bytes");
    let down_bytes = moe.down_exps_bytes().expect("DeepSeek4 down expert bytes");
    let slot_count = routes.len() * config.expert_used_count;
    let mut gate_slots = Vec::with_capacity(slot_count);
    let mut up_slots = Vec::with_capacity(slot_count);
    let mut down_slots = Vec::with_capacity(slot_count);
    let mut route_weights = Vec::with_capacity(slot_count);
    let mut token_ids = Vec::with_capacity(slot_count);
    for (token, (experts, weights)) in routes.iter().enumerate() {
        if experts.len() != config.expert_used_count {
            return Ok(None);
        }
        for (&expert, &route_weight) in experts.iter().zip(weights) {
            let gate_start = expert * gate_expert_bytes;
            let up_start = expert * up_expert_bytes;
            let down_start = expert * down_expert_bytes;
            gate_slots.push(&gate_bytes[gate_start..gate_start + gate_expert_bytes]);
            up_slots.push(&up_bytes[up_start..up_start + up_expert_bytes]);
            down_slots.push(&down_bytes[down_start..down_start + down_expert_bytes]);
            route_weights.push(route_weight);
            token_ids.push(token as u32);
        }
    }
    crate::engine::backend_runtime::sparse_experts_by_token_clamped_swiglu_resident(
        &gate_slots,
        &up_slots,
        &down_slots,
        moe.gate_quant,
        moe.down_quant,
        &route_weights,
        &token_ids,
        routes.len(),
        config.expert_ffn_dim,
        config.hidden_dim,
        inputs,
        weights.routed_clamp,
    )
    .map_err(LlmError::Forward)
}

fn compute_sparse_expert(
    input: &[f32],
    expert: usize,
    route_weight: f32,
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Vec<f32> {
    let moe = &weights.weights;
    let gate_bpr = bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_bpr = bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_bpr = bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let gate_expert_bytes = config.expert_ffn_dim * gate_bpr;
    let up_expert_bytes = config.expert_ffn_dim * up_bpr;
    let down_expert_bytes = config.hidden_dim * down_bpr;
    let gate_bytes = moe.gate_exps_bytes().expect("DeepSeek4 gate expert bytes");
    let up_bytes = moe.up_exps_bytes().expect("DeepSeek4 up expert bytes");
    let down_bytes = moe.down_exps_bytes().expect("DeepSeek4 down expert bytes");
    let gate_slice = &gate_bytes[expert * gate_expert_bytes..(expert + 1) * gate_expert_bytes];
    let up_slice = &up_bytes[expert * up_expert_bytes..(expert + 1) * up_expert_bytes];
    let down_slice = &down_bytes[expert * down_expert_bytes..(expert + 1) * down_expert_bytes];

    let mut gate = vec![0.0f32; config.expert_ffn_dim];
    let mut up = vec![0.0f32; config.expert_ffn_dim];
    host_gemv(
        gate_slice,
        input,
        &mut gate,
        config.expert_ffn_dim,
        config.hidden_dim,
        gate_bpr,
        moe.gate_quant,
    );
    host_gemv(
        up_slice,
        input,
        &mut up,
        config.expert_ffn_dim,
        config.hidden_dim,
        up_bpr,
        moe.up_quant,
    );
    swiglu_clamped(&mut gate, &mut up, weights.routed_clamp);
    let mut output = vec![0.0f32; config.hidden_dim];
    host_gemv(
        down_slice,
        &gate,
        &mut output,
        config.hidden_dim,
        config.expert_ffn_dim,
        down_bpr,
        moe.down_quant,
    );
    for value in &mut output {
        *value *= route_weight;
    }
    output
}
fn compute_sparse_expert_batch(
    inputs: &[f32],
    expert: usize,
    seq_len: usize,
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Vec<f32> {
    let moe = &weights.weights;
    let gate_bpr = bytes_per_row(config.hidden_dim, moe.gate_quant);
    let up_bpr = bytes_per_row(config.hidden_dim, moe.up_quant);
    let down_bpr = bytes_per_row(config.expert_ffn_dim, moe.down_quant);
    let gate_expert_bytes = config.expert_ffn_dim * gate_bpr;
    let up_expert_bytes = config.expert_ffn_dim * up_bpr;
    let down_expert_bytes = config.hidden_dim * down_bpr;
    let gate_bytes = moe.gate_exps_bytes().expect("DeepSeek4 gate expert bytes");
    let up_bytes = moe.up_exps_bytes().expect("DeepSeek4 up expert bytes");
    let down_bytes = moe.down_exps_bytes().expect("DeepSeek4 down expert bytes");
    let gate_slice = &gate_bytes[expert * gate_expert_bytes..(expert + 1) * gate_expert_bytes];
    let up_slice = &up_bytes[expert * up_expert_bytes..(expert + 1) * up_expert_bytes];
    let down_slice = &down_bytes[expert * down_expert_bytes..(expert + 1) * down_expert_bytes];

    let mut gate = vec![0.0f32; seq_len * config.expert_ffn_dim];
    let mut up = vec![0.0f32; seq_len * config.expert_ffn_dim];
    host_gemv_batch(
        gate_slice,
        inputs,
        &mut gate,
        config.expert_ffn_dim,
        config.hidden_dim,
        seq_len,
        gate_bpr,
        moe.gate_quant,
    );
    host_gemv_batch(
        up_slice,
        inputs,
        &mut up,
        config.expert_ffn_dim,
        config.hidden_dim,
        seq_len,
        up_bpr,
        moe.up_quant,
    );
    swiglu_clamped(&mut gate, &mut up, weights.routed_clamp);
    let mut output = vec![0.0f32; seq_len * config.hidden_dim];
    host_gemv_batch(
        down_slice,
        &gate,
        &mut output,
        config.hidden_dim,
        config.expert_ffn_dim,
        seq_len,
        down_bpr,
        moe.down_quant,
    );
    output
}

fn host_gemv(
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    bytes_per_row: usize,
    quant: GGMLType,
) {
    if host_quant_gemv_supported(quant) {
        gemv_host_quantized(bytes, input, output, rows, cols, bytes_per_row, quant);
        return;
    }
    output.par_iter_mut().enumerate().for_each(|(row, dst)| {
        let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
        let values = dequantize_bytes_to_f32(row_bytes, quant);
        *dst = values
            .iter()
            .take(cols)
            .zip(input)
            .map(|(&left, &right)| left * right)
            .sum();
    });
}
#[allow(clippy::too_many_arguments)]
fn host_gemv_batch(
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    quant: GGMLType,
) {
    if host_quant_gemv_supported(quant) {
        gemv_host_quantized_batch(
            bytes,
            input,
            output,
            rows,
            cols,
            seq_len,
            bytes_per_row,
            quant,
        );
        return;
    }
    let mut row_major = vec![0.0f32; rows * seq_len];
    row_major
        .par_chunks_mut(seq_len)
        .enumerate()
        .for_each(|(row, values)| {
            let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let dequantized = dequantize_bytes_to_f32(row_bytes, quant);
            for (token, dst) in values.iter_mut().enumerate() {
                *dst = dequantized
                    .iter()
                    .take(cols)
                    .zip(&input[token * cols..(token + 1) * cols])
                    .map(|(&left, &right)| left * right)
                    .sum();
            }
        });
    for row in 0..rows {
        for token in 0..seq_len {
            output[token * rows + row] = row_major[row * seq_len + token];
        }
    }
}

fn swiglu_clamped(gate: &mut [f32], up: &mut [f32], limit: f32) {
    for (gate, up) in gate.iter_mut().zip(up) {
        let gate_value = gate.min(limit);
        let up_value = up.clamp(-limit, limit);
        *gate = gate_value / (1.0 + (-gate_value).exp()) * up_value;
    }
}

fn bytes_per_row(cols: usize, quant: GGMLType) -> usize {
    let (block_elements, block_bytes) = ggml_quant_params(quant);
    cols.div_ceil(block_elements) * block_bytes
}

#[cfg(test)]
mod tests {
    use super::metal_decode_route_supported;

    #[test]
    fn metal_decode_route_admission_matches_backend_slot_limit() {
        assert!(metal_decode_route_supported(1, 1));
        assert!(metal_decode_route_supported(8, 8));
        assert!(!metal_decode_route_supported(0, 0));
        assert!(!metal_decode_route_supported(9, 9));
        assert!(!metal_decode_route_supported(8, 7));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn routed_cuda_layout_respects_sparse_moe_policy() {
        use super::routed_cuda_layout_supported;
        use rnb_loader::GGMLType;

        assert!(!routed_cuda_layout_supported(
            false,
            GGMLType::IQ2_XXS,
            GGMLType::IQ2_XXS,
            GGMLType::IQ3_XXS,
            7_168,
            2_048,
        ));
        assert!(routed_cuda_layout_supported(
            true,
            GGMLType::IQ2_XXS,
            GGMLType::IQ2_XXS,
            GGMLType::IQ3_XXS,
            7_168,
            2_048,
        ));
    }
}
