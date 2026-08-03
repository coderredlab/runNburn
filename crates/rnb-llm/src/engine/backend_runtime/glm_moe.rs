#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "cuda")]
use rnb_loader::GGMLType;

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn glm_moe_decode_sparse_experts_iq2xxs_iq3xxs(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::glm_moe_decode_sparse_experts_iq2xxs_iq3xxs(
            gate,
            up,
            down,
            route_weights,
            n_ff,
            n_embd,
            input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn glm_moe_prefill_sparse_experts_iq_by_token(
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
) -> std::result::Result<Vec<f32>, String> {
    cuda_runtime::glm_moe_prefill_sparse_experts_iq_by_token(
        gate,
        up,
        down,
        gate_quant,
        down_quant,
        file_regions,
        direct_file,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn moe_prefill_sparse_experts_iq2xxs_iq3xxs_clamped_swiglu(
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
) -> std::result::Result<Vec<f32>, String> {
    cuda_runtime::moe_prefill_sparse_experts_iq2xxs_iq3xxs_clamped_swiglu(
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
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn mxfp4_sparse_experts_by_token_clamped_swiglu(
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
) -> std::result::Result<Vec<f32>, String> {
    cuda_runtime::mxfp4_sparse_experts_by_token_clamped_swiglu(
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
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn sparse_experts_by_token_clamped_swiglu_resident(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    gate_quant: rnb_loader::GGMLType,
    down_quant: rnb_loader::GGMLType,
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    activation_limit: f32,
) -> std::result::Result<Option<Vec<f32>>, String> {
    cuda_runtime::sparse_experts_by_token_clamped_swiglu_resident(
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
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn glm_moe_decode_shared_expert_q5k_q6k(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::glm_moe_decode_shared_expert_q5k_q6k(
            gate, up, down, n_ff, n_embd, input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
}
