use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::dequant::dequantize_bytes_to_f32;
use crate::engine::scalar_gemv::{gemv_host_quantized, host_quant_gemv_supported};
use crate::error::Result;
use rayon::prelude::*;
use rnb_loader::convert::ggml_quant_params;
use rnb_loader::GGMLType;

use super::math::tensor_f32;
use super::weights::{DeepSeek4Config, DeepSeek4MoeWeights};

pub(super) fn forward_moe(
    input: &[f32],
    token_id: u32,
    weights: &DeepSeek4MoeWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let (experts, route_weights) = route(input, token_id, weights, config);
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
