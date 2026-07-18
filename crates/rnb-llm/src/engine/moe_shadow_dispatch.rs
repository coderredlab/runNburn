pub(super) fn q2k_gate_up_tile_bytes_per_expert(n_ff: usize, n_embd: usize) -> usize {
    super::gemm_runtime::shadow_tile_q2k::q2k_gate_up_tile_bytes_per_expert(n_ff, n_embd)
}

pub(super) fn gemv_q2k_gate_up_tile(
    tile: &[u8],
    input: &[f32],
    n_ff: usize,
    n_embd: usize,
) -> (Vec<f32>, Vec<f32>) {
    super::gemm_runtime::q2k_gate_up::gemv_q2k_gate_up_tile(tile, input, n_ff, n_embd)
}

#[cfg(test)]
pub(super) fn pack_q2k_gate_up_tile(gate: &[u8], up: &[u8], n_ff: usize, n_embd: usize) -> Vec<u8> {
    super::gemm_runtime::shadow_tile_q2k::pack_q2k_gate_up_tile(gate, up, n_ff, n_embd)
}
