#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;

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
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn glm_moe_prefill_sparse_experts_iq2xxs_iq3xxs_by_token(
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
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::glm_moe_prefill_sparse_experts_iq2xxs_iq3xxs_by_token(
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
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
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
