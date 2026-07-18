use crate::engine::moe_types::{
    q2k_bytes_per_row, q4k_bytes_per_row, q5_1_bytes_per_row, q8_0_bytes_per_row,
};
use rnb_loader::GGMLType;

/// Bytes per expert for the Q2_K shadow `gate_up_exps` tensor
/// (shape `[n_expert, n_ff*2, n_embd]`, same as base but Q2_K encoded).
#[inline]
pub fn per_expert_shadow_gate_up_bytes(n_embd: usize, n_ff: usize) -> usize {
    (n_ff * 2) * q2k_bytes_per_row(n_embd)
}

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

/// View of one gemma4-MoE FFN layer's weights.
///
/// in4 cleanup: legacy 5-way `ExpertResidencySource` (hot mmap / runtime hot
/// pool / cold mmap / cold pread / unified cold pread) is gone. Hot/cold
/// split now lives behind a single `MoeExpertResidencyView` trait dispatch
/// (`rnb-memory::moe_residency`). When the engine has a `.rnb` sidecar with
/// `metadata_v3` (or any other residency-aware loader) wired, it sets
/// `gate_up_residency` / `down_residency` and forward routes per-rank bytes
/// through the trait. When neither is present, the engine falls back to the
/// flat GGUF mmap (`gate_up_bytes` / `down_bytes`) for an "all hot, no
/// split" path.
pub struct MoeLayerView<'a> {
    /// Router F32 weights, row-major `[n_expert, n_embd]`.
    pub router_w: &'a [f32],
    /// Flat Q4_K bytes for `gate_up_exps`. Holds the full `[n_expert]`
    /// expert range when `gate_up_residency == None`. When residency is set
    /// the trait owns byte routing and these slices are unused.
    pub gate_up_bytes: &'a [u8],
    pub down_bytes: &'a [u8],
    pub down_scale: &'a [f32],
    pub down_quant: GGMLType,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub layer_idx: Option<usize>,
    /// rank → original_expert_id map (memtrace only; does not affect math).
    pub rank_to_original: Option<&'a [u32]>,
    /// Session 71 MoE mixed precision: Q2_K shadow gate_up_exps bytes.
    pub shadow_gate_up_bytes: Option<&'a [u8]>,
    /// Residency view for `gate_up_exps`: hot/cold dispatch + popularity
    /// order. `None` falls back to flat `gate_up_bytes` (all-hot).
    pub gate_up_residency: Option<&'a (dyn rnb_loader::MoeExpertResidencyView + 'a)>,
    pub down_residency: Option<&'a (dyn rnb_loader::MoeExpertResidencyView + 'a)>,
}
