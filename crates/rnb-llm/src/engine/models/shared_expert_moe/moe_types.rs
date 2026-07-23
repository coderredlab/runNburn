use rnb_loader::GGMLType;

use rnb_loader::convert::ggml_quant_params;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in crate::engine) struct SparseExpertBytes {
    pub(in crate::engine) gate: usize,
    pub(in crate::engine) up: usize,
    pub(in crate::engine) down: usize,
}

#[inline]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in crate::engine) fn sparse_expert_bytes(
    n_embd: usize,
    n_ff: usize,
    gate_quant: GGMLType,
    up_quant: GGMLType,
    down_quant: GGMLType,
) -> Option<SparseExpertBytes> {
    let gate = n_ff * expert_bytes_per_row(n_embd, gate_quant, "gate_exps");
    let up = n_ff * expert_bytes_per_row(n_embd, up_quant, "up_exps");
    let down_bpr = expert_bytes_per_row(n_ff, down_quant, "down_exps");
    Some(SparseExpertBytes {
        gate,
        up,
        down: n_embd * down_bpr,
    })
}

/// Byte width for one sparse-expert down-projection row.
#[inline]
pub(in crate::engine) fn down_bytes_per_row(cols: usize, quant: GGMLType) -> usize {
    quant_bytes_per_row(cols, quant)
}

#[inline]
pub(in crate::engine) fn expert_bytes_per_row(cols: usize, quant: GGMLType, _label: &str) -> usize {
    quant_bytes_per_row(cols, quant)
}

#[inline]
fn quant_bytes_per_row(cols: usize, quant: GGMLType) -> usize {
    let (elements_per_block, bytes_per_block) = ggml_quant_params(quant);
    cols.div_ceil(elements_per_block) * bytes_per_block
}

/// Borrowed view over one split sparse-expert MoE layer and its shared expert.
pub struct SharedExpertMoEView<'a> {
    /// F32 `[n_expert, n_embd]` router projection (`ffn_gate_inp`).
    pub router_w: &'a [f32],
    /// Optional correction added to sigmoid probabilities only for top-k selection.
    pub router_selection_bias: Option<&'a [f32]>,
    pub expert_gating_func: u32,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    /// Quantized `[n_expert, n_ff, n_embd]` sparse-expert gate projection.
    pub gate_exps_bytes: &'a [u8],
    pub gate_quant: GGMLType,
    /// Quantized `[n_expert, n_ff, n_embd]` sparse-expert up projection.
    pub up_exps_bytes: &'a [u8],
    pub up_quant: GGMLType,
    /// Quantized `[n_expert, n_embd, n_ff]` sparse-expert down projection.
    pub down_exps_bytes: &'a [u8],
    pub down_quant: GGMLType,
    /// Optional shared-expert scalar-gate projection. Used only when
    /// `shared_expert_gated` is true.
    pub shared_input_scale: &'a [f32],
    pub shared_expert_gated: bool,
    /// Quantized `[n_ff, n_embd]` shared-expert gate.
    pub shared_gate_bytes: &'a [u8],
    pub shared_gate_quant: GGMLType,
    /// Quantized `[n_ff, n_embd]` shared-expert up.
    pub shared_up_bytes: &'a [u8],
    pub shared_up_quant: GGMLType,
    /// Quantized `[n_embd, n_ff]` shared-expert down.
    pub shared_down_bytes: &'a [u8],
    pub shared_down_quant: GGMLType,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub layer_idx: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_cuda_gemv_quant_rows_use_canonical_ggml_sizes() {
        let cols = 256;
        for (quant, expected) in [
            (GGMLType::F32, 1024),
            (GGMLType::F16, 512),
            (GGMLType::BF16, 512),
            (GGMLType::Q4_0, 144),
            (GGMLType::Q4_1, 160),
            (GGMLType::Q5_0, 176),
            (GGMLType::Q5_1, 192),
            (GGMLType::Q8_0, 272),
        ] {
            assert_eq!(expert_bytes_per_row(cols, quant, "test"), expected);
            assert_eq!(down_bytes_per_row(cols, quant), expected);
        }
    }
}
