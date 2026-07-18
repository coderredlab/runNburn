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
pub fn glm_moe_prefill_sparse_experts_iq2xxs_iq3xxs_by_token(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
    direct_file: bool,
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_sparse_experts_iq2xxs_iq3xxs_by_token(
        gate,
        up,
        down,
        file_regions,
        direct_file,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA GLM batched sparse IQ2_XXS/IQ3_XXS MoE failed: {err}"))
}

pub fn glm_moe_decode_shared_expert_q5k_q6k(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::glm_shared_expert_q5k_q6k(gate, up, down, n_ff, n_embd, input)
        .map_err(|err| format!("CUDA GLM shared Q5_K/Q6_K expert failed: {err}"))
}
