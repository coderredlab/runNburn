use crate::engine::moe_types::{q4k_bytes_per_row, q5_1_bytes_per_row, q8_0_bytes_per_row};
use rnb_loader::GGMLType;

/// Bytes per expert for the `gate_up_exps` Q4_K tensor (shape
/// `[n_expert, n_ff*2, n_embd]`).
#[inline]
pub fn per_expert_gate_up_bytes(n_embd: usize, n_ff: usize) -> usize {
    (n_ff * 2) * q4k_bytes_per_row(n_embd)
}

/// Bytes per row for the `down_exps` tensor. Gemma4 MoE GGUFs can mix
/// Q5_0 and Q8_0 down experts by layer, so callers must use the real tensor
/// quant type instead of assuming one fixed layout.
#[inline]
pub fn down_bytes_per_row(cols: usize, quant: GGMLType) -> usize {
    match quant {
        GGMLType::Q5_0 => {
            debug_assert!(cols % 32 == 0, "Q5_0 requires cols divisible by 32");
            (cols / 32) * 22
        }
        GGMLType::Q5_1 => q5_1_bytes_per_row(cols),
        GGMLType::Q8_0 => q8_0_bytes_per_row(cols),
        other => panic!("unsupported Gemma MoE down quant {other:?}"),
    }
}

#[inline]
pub fn per_expert_down_bytes(n_embd: usize, n_ff: usize, quant: GGMLType) -> usize {
    n_embd * down_bytes_per_row(n_ff, quant)
}

/// View of one Gemma4 MoE FFN layer's GGUF-backed weights.
pub struct MoeLayerView<'a> {
    /// Router F32 weights, row-major `[n_expert, n_embd]`.
    pub router_w: &'a [f32],
    /// Flat Q4_K bytes for the full `gate_up_exps` expert range.
    pub gate_up_bytes: &'a [u8],
    pub down_bytes: &'a [u8],
    pub down_scale: &'a [f32],
    pub down_quant: GGMLType,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub layer_idx: Option<usize>,
}
