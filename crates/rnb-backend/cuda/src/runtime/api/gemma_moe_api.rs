use crate::runtime::{CudaState, DEFAULT_CUDA_COMPUTE};
use std::sync::Mutex;

#[allow(clippy::too_many_arguments)]
pub fn gemma4_moe_gelu_selected(
    gate_up_experts: &[u8],
    down_experts: &[u8],
    down_quant: u32,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    expert_ids: &[u32],
    token_ids: &[u32],
    route_weights: &[f32],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .gemma4_moe_gelu_selected(
            gate_up_experts,
            down_experts,
            down_quant,
            n_expert,
            n_ff,
            n_embd,
            seq_len,
            expert_ids,
            token_ids,
            route_weights,
            input,
        )
}
