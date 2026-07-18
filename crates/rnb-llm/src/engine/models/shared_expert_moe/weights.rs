use super::page_cache::SparseExpertPageCache;
use crate::engine::cpu_runtime::kernels;
use crate::engine::moe_section::MoeSectionDecodeLayer;
use crate::engine::packed_runtime::PackedModel;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::{GGMLType, MoeExpertResidencyView};
use std::sync::Arc;

/// Per-layer weights for split sparse experts plus an always-on shared expert.
///
/// Qwen3.5 MoE and Hy3 both use separate `ffn_gate_exps`,
/// `ffn_up_exps`, and `ffn_down_exps` tensors. Routing behavior and all
/// quantization types are loaded from GGUF metadata rather than inferred from
/// the model name.
pub(in crate::engine) struct SharedExpertMoELayerWeights {
    /// F32 `[n_expert, n_embd]` — `ffn_gate_inp.weight` router projection.
    pub(in crate::engine) router_w: Tensor,
    /// Optional F32 `[n_expert]` correction added only when selecting top-k experts.
    pub(in crate::engine) router_selection_bias: Option<Tensor>,
    pub(in crate::engine) expert_gating_func: u32,
    pub(in crate::engine) expert_weights_norm: bool,
    pub(in crate::engine) expert_weights_scale: f32,
    /// Quantized `[n_expert, n_ff, n_embd]` — `ffn_gate_exps.weight`.
    pub(in crate::engine) gate_exps: Tensor,
    pub(in crate::engine) gate_quant: GGMLType,
    /// Quantized `[n_expert, n_ff, n_embd]` — `ffn_up_exps.weight`.
    pub(in crate::engine) up_exps: Tensor,
    pub(in crate::engine) up_quant: GGMLType,
    /// Quantized bytes `[n_expert, n_embd, n_ff]` — `ffn_down_exps.weight`.
    /// Quantization type is layer-local and model-defined.
    pub(in crate::engine) down_exps: Tensor,
    pub(in crate::engine) down_quant: GGMLType,
    /// F32 `[n_embd]` — `ffn_gate_inp_shexp.weight` (shared-expert pre-norm /
    /// per-feature scaling applied to the post-attention hidden before the
    /// shared expert FFN).
    pub(in crate::engine) shared_input_scale: Tensor,
    /// Whether the shared expert output is multiplied by
    /// `sigmoid(ffn_gate_inp_shexp · h)`.
    pub(in crate::engine) shared_expert_gated: bool,
    /// Quantized `[n_ff, n_embd]` — `ffn_gate_shexp.weight`.
    pub(in crate::engine) shared_gate: QuantizedWeight,
    /// Quantized `[n_ff, n_embd]` — `ffn_up_shexp.weight`.
    pub(in crate::engine) shared_up: QuantizedWeight,
    /// Quantized `[n_embd, n_ff]` — `ffn_down_shexp.weight`.
    pub(in crate::engine) shared_down: QuantizedWeight,
    pub(in crate::engine) n_embd: usize,
    pub(in crate::engine) n_ff: usize,
    pub(in crate::engine) n_expert: usize,
    pub(in crate::engine) n_expert_used: usize,
    /// Resolved engine-load policy for Q2_K/Q3_K sparse CUDA execution,
    /// including application RAM policy and the diagnostic environment override.
    pub(in crate::engine) prefer_sparse_moe_cuda: bool,
    pub(in crate::engine) sparse_page_cache: Option<Arc<SparseExpertPageCache>>,

    /// `Arc` clone of Engine `packed_model`; `None` when `.rnb` not loaded.
    pub(in crate::engine) packed_model: Option<Arc<PackedModel>>,
    /// Tensor name in the `.rnb` for `gate_exps` (`blk.{i}.ffn_gate_exps.weight`).
    pub(in crate::engine) gate_exps_rnb_name: Option<String>,
    pub(in crate::engine) up_exps_rnb_name: Option<String>,
    pub(in crate::engine) down_exps_rnb_name: Option<String>,
    pub(in crate::engine) router_rnb_name: Option<String>,
    pub(in crate::engine) shared_gate_rnb_name: Option<String>,
    pub(in crate::engine) shared_up_rnb_name: Option<String>,
    pub(in crate::engine) shared_down_rnb_name: Option<String>,
    pub(in crate::engine) shared_scale_rnb_name: Option<String>,
    /// `.rnb` hot-sort permutation (rank -> original_expert_id).
    pub(in crate::engine) rank_to_original: Option<Vec<usize>>,

    pub(in crate::engine) shadow_model: Option<Arc<PackedModel>>,
    pub(in crate::engine) shadow_gate_up_tile_rnb_name: Option<String>,
    pub(in crate::engine) shadow_gate_rnb_name: Option<String>,
    pub(in crate::engine) shadow_up_rnb_name: Option<String>,
    /// Optional lower-precision shadow for the down projection.
    pub(in crate::engine) shadow_down_rnb_name: Option<String>,

    /// Optional `.rnb` MoE section using the `MOE_DECODE_SECTION` layout.
    /// `Some(_)` when the engine was loaded with a MoE section `.rnb` companion file
    /// carrying a parsed MoE decode section; `None` for GGUF-only loads or
    /// when diagnostic `RNB_MOE_DECODE=1` is not set.
    pub(in crate::engine) moe_section_decode: Option<MoeSectionDecodeLayer>,
    /// V3 sidecar residency views for the gate, up, and down sparse-expert
    /// tensors. Sparse fanout resolves expert bytes through these views when
    /// present; `None` keeps direct flat-tensor indexing.
    /// Trait owner lives in `rnb-memory`.
    pub(in crate::engine) gate_residency: Option<Arc<dyn MoeExpertResidencyView>>,
    pub(in crate::engine) up_residency: Option<Arc<dyn MoeExpertResidencyView>>,
    pub(in crate::engine) down_residency: Option<Arc<dyn MoeExpertResidencyView>>,
}

impl SharedExpertMoELayerWeights {
    /// Returns the bytes for `gate_exps`, preferring the `.rnb` override when
    /// available.
    #[inline]
    pub fn gate_exps_bytes(&self) -> Option<&[u8]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.gate_exps_rnb_name) {
            return pm.get_weight(name).map(|w| w.data());
        }
        self.gate_exps.as_bytes()
    }

    #[inline]
    pub fn up_exps_bytes(&self) -> Option<&[u8]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.up_exps_rnb_name) {
            return pm.get_weight(name).map(|w| w.data());
        }
        self.up_exps.as_bytes()
    }

    #[inline]
    pub fn down_exps_bytes(&self) -> Option<&[u8]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.down_exps_rnb_name) {
            return pm.get_weight(name).map(|w| w.data());
        }
        self.down_exps.as_bytes()
    }

    pub fn sparse_expert_file_regions(&self) -> Option<[rnb_core::tensor::FileBackedRegion; 3]> {
        if self.gate_residency.is_some()
            || self.up_residency.is_some()
            || self.down_residency.is_some()
            || (self.packed_model.is_some()
                && (self.gate_exps_rnb_name.is_some()
                    || self.up_exps_rnb_name.is_some()
                    || self.down_exps_rnb_name.is_some()))
        {
            return None;
        }
        Some([
            self.gate_exps.file_backed_region()?,
            self.up_exps.file_backed_region()?,
            self.down_exps.file_backed_region()?,
        ])
    }

    /// Router weights as F32 slice, preferring `.rnb` override.
    #[inline]
    pub fn router_f32(&self) -> Option<&[f32]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.router_rnb_name) {
            if let Some(w) = pm.get_weight(name) {
                let bytes = w.data();
                if bytes.len() % 4 != 0 {
                    return None;
                }
                // SAFETY: PackedModel mmap owned by Arc, 4096-aligned tensor data,
                // F32 is 4B so alignment is guaranteed.
                let f32s = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                };
                return Some(f32s);
            }
        }
        Some(kernels::tensor_as_f32_slice(&self.router_w))
    }

    // Session 73 MoE mixed-precision shadow accessors.
    #[inline]
    pub fn shadow_gate_up_tile_bytes(&self) -> Option<&[u8]> {
        let pm = self.shadow_model.as_ref()?;
        let name = self.shadow_gate_up_tile_rnb_name.as_deref()?;
        pm.get_weight(name).map(|w| w.data())
    }

    #[inline]
    pub fn shadow_gate_bytes(&self) -> Option<&[u8]> {
        let pm = self.shadow_model.as_ref()?;
        let name = self.shadow_gate_rnb_name.as_deref()?;
        pm.get_weight(name).map(|w| w.data())
    }

    #[inline]
    pub fn shadow_up_bytes(&self) -> Option<&[u8]> {
        let pm = self.shadow_model.as_ref()?;
        let name = self.shadow_up_rnb_name.as_deref()?;
        pm.get_weight(name).map(|w| w.data())
    }

    #[inline]
    pub fn shadow_down_bytes(&self) -> Option<&[u8]> {
        let pm = self.shadow_model.as_ref()?;
        let name = self.shadow_down_rnb_name.as_deref()?;
        pm.get_weight(name).map(|w| w.data())
    }

    /// Borrows the attached MoE decode section, when present, for construction
    /// of a `SharedExpertMoEView`.
    #[inline]
    pub fn moe_section_decode_view(&self) -> Option<&MoeSectionDecodeLayer> {
        self.moe_section_decode.as_ref()
    }
}
