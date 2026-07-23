use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::GGMLType;

/// Gemma4 26B-A4B per-layer MoE weights backed directly by the GGUF mapping.
///
/// Expert tensors retain their source quantization and are indexed in their
/// original order.
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
}

impl MoeLayerWeights {
    /// Returns the GGUF bytes for `gate_up_exps`.
    #[inline]
    pub fn gate_up_bytes(&self) -> Option<&[u8]> {
        self.gate_up_exps.as_bytes()
    }

    /// Same as `gate_up_bytes` for `down_exps`.
    #[inline]
    pub fn down_bytes(&self) -> Option<&[u8]> {
        self.down_exps.as_bytes()
    }

    /// Router weights as a direct F32 slice.
    #[inline]
    pub fn router_f32(&self) -> Option<&[f32]> {
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
