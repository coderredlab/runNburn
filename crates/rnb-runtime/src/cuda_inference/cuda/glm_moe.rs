use super::*;

#[allow(clippy::too_many_arguments)]
pub fn glm_moe_decode_sparse_experts_iq2xxs_iq3xxs(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_sparse_experts_iq2xxs_iq3xxs(gate, up, down, route_weights, n_ff, n_embd, input)
        .map_err(|err| format!("CUDA GLM sparse IQ2_XXS/IQ3_XXS MoE failed: {err}"))
}

pub fn glm_moe_direct_file_prefill_enabled(auto_enabled: bool) -> bool {
    rnb_backend_cuda::tuning::glm_direct_file_prefill_enabled(auto_enabled)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_moe_prefill_sparse_experts_iq_by_token(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    gate_quant: GGMLType,
    down_quant: GGMLType,
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
    direct_file: bool,
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_sparse_experts_iq_by_token(
        gate,
        up,
        down,
        gate_quant as u32,
        down_quant as u32,
        file_regions,
        direct_file,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA GLM batched sparse IQ MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_prefill_sparse_experts_iq2xxs_iq3xxs_clamped_swiglu(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    activation_limit: f32,
) -> Result<Vec<f32>> {
    backend::sparse_experts_iq2xxs_iq3xxs_by_token_clamped_swiglu(
        gate,
        up,
        down,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
        activation_limit,
    )
    .map_err(|err| format!("CUDA batched sparse clamped SwiGLU MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn mxfp4_sparse_experts_by_token_clamped_swiglu(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    activation_limit: f32,
) -> Result<Vec<f32>> {
    backend::mxfp4_sparse_experts_by_token_clamped_swiglu(
        gate,
        up,
        down,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
        activation_limit,
    )
    .map_err(|error| format!("CUDA batched sparse MXFP4 MoE failed: {error}"))
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_experts_by_token_clamped_swiglu_resident(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    gate_quant: u32,
    down_quant: u32,
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    activation_limit: f32,
) -> Result<Option<Vec<f32>>> {
    backend::sparse_experts_by_token_clamped_swiglu_resident(
        gate,
        up,
        down,
        gate_quant,
        down_quant,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
        activation_limit,
    )
    .map_err(|err| format!("CUDA resident sparse clamped SwiGLU MoE failed: {err}"))
}

pub fn clear_moe_expert_slice_cache() -> Result<()> {
    backend::clear_moe_expert_slice_cache()
        .map_err(|err| format!("clearing CUDA MoE expert slice cache failed: {err}"))
}

pub fn glm_moe_decode_shared_expert_q5k_q6k(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_shared_expert_iq(gate, up, down, 13, 14, n_ff, n_embd, input)
        .map_err(|err| format!("CUDA GLM shared Q5_K/Q6_K expert failed: {err}"))
}

pub fn glm_moe_prefill_shared_expert_iq(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    gate_quant: GGMLType,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_shared_expert_iq(
        gate,
        up,
        down,
        gate_quant as u32,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA GLM batched shared IQ expert failed: {err}"))
}

/// Registers per-layer sparse expert regions for stream read-ahead.
pub fn glm_register_stream_region_sequence(sequence: &[[rnb_core::tensor::FileBackedRegion; 3]]) {
    let sequence = sequence
        .iter()
        .map(|regions| {
            [0usize, 1, 2].map(|index| {
                (
                    regions[index].path().to_path_buf(),
                    regions[index].file_offset(),
                    regions[index].len(),
                )
            })
        })
        .collect();
    backend::glm_register_stream_region_sequence(sequence);
}
