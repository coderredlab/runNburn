use rnb_loader::GGMLType;

use crate::engine::moe_types::{
    iq4_xs_bytes_per_row, q2k_bytes_per_row, q3k_bytes_per_row, q4k_bytes_per_row,
    q5k_bytes_per_row, q6k_bytes_per_row, q8_0_bytes_per_row,
};

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
    match quant {
        GGMLType::Q2_K => q2k_bytes_per_row(cols),
        GGMLType::Q3_K => q3k_bytes_per_row(cols),
        GGMLType::Q5_K => q5k_bytes_per_row(cols),
        GGMLType::Q6_K => q6k_bytes_per_row(cols),
        GGMLType::Q4_K => q4k_bytes_per_row(cols),
        GGMLType::IQ4_XS => iq4_xs_bytes_per_row(cols),
        GGMLType::IQ2_XXS => cols.div_ceil(256) * 66,
        GGMLType::IQ3_XXS => cols.div_ceil(256) * 98,
        GGMLType::IQ2_S => cols.div_ceil(256) * 82,
        other => panic!("shared-expert MoE down_exps unsupported quant {other:?}"),
    }
}

#[inline]
pub(in crate::engine) fn expert_bytes_per_row(cols: usize, quant: GGMLType, label: &str) -> usize {
    match quant {
        GGMLType::Q2_K => q2k_bytes_per_row(cols),
        GGMLType::Q3_K => q3k_bytes_per_row(cols),
        GGMLType::Q4_K => q4k_bytes_per_row(cols),
        GGMLType::Q5_K => q5k_bytes_per_row(cols),
        GGMLType::Q6_K => q6k_bytes_per_row(cols),
        GGMLType::Q8_0 => q8_0_bytes_per_row(cols),
        GGMLType::IQ4_XS => iq4_xs_bytes_per_row(cols),
        GGMLType::IQ2_XXS => cols.div_ceil(256) * 66,
        GGMLType::IQ3_XXS => cols.div_ceil(256) * 98,
        GGMLType::IQ2_S => cols.div_ceil(256) * 82,
        other => panic!("shared-expert MoE {label} unsupported quant {other:?}"),
    }
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

    /// `None` 이면 mixed precision off (shadow `.rnb` 없거나 `RNB_HOBBIT` env 없음).
    /// `Some` 이면 Q2_K shadow gate — `[n_expert, n_ff, n_embd]`.
    pub shadow_gate_bytes: Option<&'a [u8]>,
    /// Q2_K shadow up — `[n_expert, n_ff, n_embd]`.
    pub shadow_up_bytes: Option<&'a [u8]>,
    /// Optional fused tile-major gate/up shadow sidecar.
    pub shadow_gate_up_tile_bytes: Option<&'a [u8]>,
    /// Optional lower-precision shadow for the down projection.
    pub shadow_down_bytes: Option<&'a [u8]>,

    /// Optional `.rnb` MoE section using the `MOE_DECODE_SECTION` layout.
    /// On aarch64, `forward` may dispatch into
    /// `SharedExpertMoEView::forward_moe_section_sdot`; `None` keeps the
    /// direct tensor path.
    ///
    /// Field is crate-private because `MoeSectionDecodeLayer` itself is
    /// `pub(crate)` (engine-owned layout, no external API). Callers outside
    /// the crate cannot construct or borrow one anyway.
    ///
    /// `dead_code` allow: on x86 dev builds the dispatch + sdot method are
    /// both `#[cfg(target_arch = "aarch64")]`, so this field has no reader
    /// outside of the host-arch reachability check.
    #[allow(dead_code)]
    pub(crate) moe_section_decode: Option<&'a crate::engine::MoeSectionDecodeLayer>,
    /// V3 sidecar residency views for sparse experts. When present, sparse
    /// fanout resolves each expert through `expert_bytes(rank)` instead of
    /// indexing flat tensor bytes. Shadow and tile paths continue to use their
    /// dedicated byte slices.
    pub gate_residency: Option<&'a (dyn rnb_loader::MoeExpertResidencyView + 'a)>,
    pub up_residency: Option<&'a (dyn rnb_loader::MoeExpertResidencyView + 'a)>,
    pub down_residency: Option<&'a (dyn rnb_loader::MoeExpertResidencyView + 'a)>,
}
