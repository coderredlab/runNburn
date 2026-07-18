use crate::engine::cpu_runtime::kernels;
use crate::engine::packed_runtime::PackedModel;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::{GGMLType, MoeExpertResidencyView};
use std::sync::Arc;

/// Gemma4 26B-A4B per-layer MoE weights.
///
/// in4 cleanup: legacy 5-way hot/cold split fields (`gate_up_hot_heap`,
/// `down_hot_heap`, `hot_count`, `runtime_hot_count`, `cold_reader`,
/// `gate_up_cold_name`, `down_cold_name`) are replaced by a single
/// `gate_up_residency` / `down_residency` pair (trait
/// `rnb-memory::moe_residency::MoeExpertResidencyView`). `packed_wiring`
/// composes the right `HotByteSource` + `ColdByteSource` per layer based on
/// the loaded `.rnb` shape (v3 sidecar / legacy hot+cold split / GGUF flat).
#[derive(Clone)]
pub(in crate::engine) struct MoeLayerWeights {
    /// F32 `[n_expert, n_embd]` — `ffn_gate_inp.weight` (router projection).
    pub(in crate::engine) router_w: Tensor,
    /// F32 `[n_embd]` — `ffn_gate_inp.scale` (learned router pre-projection scale).
    pub(in crate::engine) router_scale: Tensor,
    /// Q4_K bytes `[n_expert, n_ff*2, n_embd]` — `ffn_gate_up_exps.weight`.
    pub(in crate::engine) gate_up_exps: Tensor,
    /// Quantized bytes `[n_expert, n_embd, n_ff]` — `ffn_down_exps.weight`.
    pub(in crate::engine) down_exps: Tensor,
    /// F32 `[n_expert]` — per-expert scale applied to `down_exps`.
    pub(in crate::engine) down_scale: Tensor,
    pub(in crate::engine) down_quant: GGMLType,
    pub(in crate::engine) post_ffw_norm_1: Tensor,
    pub(in crate::engine) post_ffw_norm_2: Tensor,
    pub(in crate::engine) pre_ffw_norm_2: Tensor,
    pub(in crate::engine) n_embd: usize,
    pub(in crate::engine) n_ff: usize,
    pub(in crate::engine) n_expert: usize,
    pub(in crate::engine) n_expert_used: usize,
    /// When `Some`, MoE bytes come from this PackedModel instead of GGUF.
    /// Shared across all layers via `Arc`.
    pub(in crate::engine) packed_model: Option<Arc<PackedModel>>,
    /// Tensor name in the `.rnb` for `gate_up_exps`. `None` -> use GGUF.
    pub(in crate::engine) gate_up_rnb_name: Option<String>,
    pub(in crate::engine) down_rnb_name: Option<String>,
    pub(in crate::engine) router_rnb_name: Option<String>,
    /// rank -> original_expert_id map (memtrace only; does not affect math).
    pub(in crate::engine) rank_to_original: Option<Vec<u32>>,
    /// Session 71 MoE mixed precision: Q2_K shadow `.rnb` handle.
    pub(in crate::engine) shadow_model: Option<Arc<PackedModel>>,
    pub(in crate::engine) shadow_gate_up_rnb_name: Option<String>,
    /// Single residency view (composed in `packed_wiring`) for the per-rank
    /// hot/cold byte dispatch in `MoeLayerView::forward`. `None` falls back
    /// to the GGUF-flat indexing of `gate_up_exps` / `down_exps`.
    pub(in crate::engine) gate_up_residency: Option<Arc<dyn MoeExpertResidencyView>>,
    pub(in crate::engine) down_residency: Option<Arc<dyn MoeExpertResidencyView>>,
}

impl MoeLayerWeights {
    /// Returns the bytes for `gate_up_exps`, preferring the `.rnb` (hot-sorted)
    /// override when available.
    #[inline]
    pub fn gate_up_bytes(&self) -> Option<&[u8]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.gate_up_rnb_name) {
            return pm.get_weight(name).map(|w| w.data());
        }
        self.gate_up_exps.as_bytes()
    }

    /// Same as `gate_up_bytes` for `down_exps`.
    #[inline]
    pub fn down_bytes(&self) -> Option<&[u8]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.down_rnb_name) {
            return pm.get_weight(name).map(|w| w.data());
        }
        self.down_exps.as_bytes()
    }

    /// Session 71 MoE mixed precision: Q2_K shadow bytes for `gate_up_exps`. `None` when
    /// no shadow `.rnb` is wired. Same lifetime as the shadow Arc mmap.
    #[inline]
    pub fn shadow_gate_up_bytes(&self) -> Option<&[u8]> {
        let pm = self.shadow_model.as_ref()?;
        let name = self.shadow_gate_up_rnb_name.as_deref()?;
        pm.get_weight(name).map(|w| w.data())
    }

    /// Router weights as F32 slice, preferring `.rnb` (hot-sorted) override.
    #[inline]
    pub fn router_f32(&self) -> Option<&[f32]> {
        if let (Some(pm), Some(name)) = (&self.packed_model, &self.router_rnb_name) {
            if let Some(w) = pm.get_weight(name) {
                let bytes = w.data();
                if bytes.len() % 4 != 0 {
                    return None;
                }
                // SAFETY: PackedModel mmap is owned by Arc inside the Engine
                // (and cloned into this struct). Bytes are properly aligned
                // because `.rnb` writer aligns tensor data to 4096 and F32 is 4B.
                let f32s = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                };
                return Some(f32s);
            }
        }
        Some(kernels::tensor_as_f32_slice(&self.router_w))
    }
}

pub(in crate::engine) struct GemmaPerLayerLayerWeights {
    pub(in crate::engine) inp_gate: QuantizedWeight,
    pub(in crate::engine) proj: QuantizedWeight,
    pub(in crate::engine) post_norm: Tensor,
}

pub(in crate::engine) struct GemmaPerLayerWeights {
    pub(in crate::engine) token_embd: QuantizedWeight,
    pub(in crate::engine) model_proj: QuantizedWeight,
    pub(in crate::engine) proj_norm: Tensor,
    pub(in crate::engine) layers: Vec<GemmaPerLayerLayerWeights>,
}

pub(in crate::engine) struct GemmaPerLayerBase {
    pub(in crate::engine) mixed: Vec<f32>,
    pub(in crate::engine) token: Vec<f32>,
    pub(in crate::engine) model: Vec<f32>,
}
