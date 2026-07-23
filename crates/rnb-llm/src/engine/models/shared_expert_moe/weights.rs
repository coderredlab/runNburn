use super::page_cache::SparseExpertPageCache;
use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::GGMLType;
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
    /// Quantized `[n_expert, n_ff, n_embd]` sparse expert projections.
    pub(in crate::engine) gate_exps: Tensor,
    pub(in crate::engine) gate_quant: GGMLType,
    pub(in crate::engine) up_exps: Tensor,
    pub(in crate::engine) up_quant: GGMLType,
    /// Quantized `[n_expert, n_embd, n_ff]` sparse expert down projection.
    pub(in crate::engine) down_exps: Tensor,
    pub(in crate::engine) down_quant: GGMLType,
    pub(in crate::engine) shared_input_scale: Tensor,
    pub(in crate::engine) shared_expert_gated: bool,
    pub(in crate::engine) shared_gate: QuantizedWeight,
    pub(in crate::engine) shared_up: QuantizedWeight,
    pub(in crate::engine) shared_down: QuantizedWeight,
    pub(in crate::engine) n_embd: usize,
    pub(in crate::engine) n_ff: usize,
    pub(in crate::engine) n_expert: usize,
    pub(in crate::engine) n_expert_used: usize,
    /// Resolved engine-load policy for Q2_K/Q3_K sparse CUDA execution.
    pub(in crate::engine) prefer_sparse_moe_cuda: bool,
    pub(in crate::engine) sparse_page_cache: Option<Arc<SparseExpertPageCache>>,
}

impl SharedExpertMoELayerWeights {
    #[inline]
    pub fn gate_exps_bytes(&self) -> Option<&[u8]> {
        self.gate_exps.as_bytes()
    }

    #[inline]
    pub fn up_exps_bytes(&self) -> Option<&[u8]> {
        self.up_exps.as_bytes()
    }

    #[inline]
    pub fn down_exps_bytes(&self) -> Option<&[u8]> {
        self.down_exps.as_bytes()
    }

    pub fn sparse_expert_file_regions(&self) -> Option<[rnb_core::tensor::FileBackedRegion; 3]> {
        Some([
            self.gate_exps.file_backed_region()?,
            self.up_exps.file_backed_region()?,
            self.down_exps.file_backed_region()?,
        ])
    }

    #[inline]
    pub fn router_f32(&self) -> Option<&[f32]> {
        Some(kernels::tensor_as_f32_slice(&self.router_w))
    }
}
