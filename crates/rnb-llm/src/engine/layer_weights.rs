use super::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;

#[allow(unused_imports)]
pub(in crate::engine) use super::models::gemma::{
    GemmaPerLayerBase, GemmaPerLayerLayerWeights, GemmaPerLayerWeights, MoeLayerWeights,
};
pub(in crate::engine) use super::models::nemotron::{
    mamba::NemotronMamba2LayerWeights, moe::NemotronMoELayerWeights,
};
#[allow(unused_imports)]
pub(in crate::engine) use super::models::shared_expert_moe::SharedExpertMoELayerWeights;

pub(super) struct AttentionLayerWeights {
    pub(super) attn_norm: Tensor, // F32 [hidden]
    pub(super) q_weight: QuantizedWeight,
    pub(super) k_weight: QuantizedWeight,
    pub(super) v_weight: QuantizedWeight,
    pub(super) o_weight: QuantizedWeight,
    pub(super) q_bias: Option<Tensor>, // Qwen2 등 bias 있는 모델용
    pub(super) k_bias: Option<Tensor>,
    pub(super) v_bias: Option<Tensor>,
    // Qwen3.5 attention: Q/K norm + gated attention
    pub(super) q_norm: Option<Tensor>,         // F32 [head_dim]
    pub(super) k_norm: Option<Tensor>,         // F32 [head_dim]
    pub(super) post_attn_norm: Option<Tensor>, // F32 [hidden] (Qwen3.5 post_attention_norm)
    pub(super) out_scale: Option<Tensor>,      // F32 [1] (Gemma4 layer_output_scale)
    pub(super) ffn_norm: Tensor,               // F32 [hidden]
    pub(super) post_ffw_norm: Option<Tensor>,
    pub(super) ffn_gate_weight: QuantizedWeight,
    pub(super) ffn_up_weight: QuantizedWeight,
    pub(super) ffn_down_weight: QuantizedWeight,
    pub(super) ffn_gate_up_fused: Option<QuantizedWeight>,
    // Gemma4 26B-A4B hybrid MoE (None = dense-only layer)
    pub(super) moe: Option<MoeLayerWeights>,
    /// Split sparse experts plus an always-on shared expert, used by Qwen3.5
    /// MoE and Hy3. Mutually exclusive with `moe`. When set, the dense FFN
    /// weights are placeholders and this shared-expert path owns the FFN.
    pub(super) shared_expert_moe: Option<SharedExpertMoELayerWeights>,
    /// True iff `blk.{i}.attn_v.weight` was absent (gemma4 26B-A4B full-attn layers
    /// 5/11/17/23/29). When true, V is aliased from K (matches llama.cpp
    /// `gemma4-iswa.cpp:83` — `Vcur = wv ? wv*cur : Kcur`).
    pub(super) v_proj_missing: bool,
}

pub(super) struct GdnLayerWeights {
    pub(super) attn_norm: Tensor,            // F32 [hidden]
    pub(super) post_attn_norm: Tensor,       // F32 [hidden]
    pub(super) qkv_weight: QuantizedWeight,  // [conv_channels, hidden]
    pub(super) gate_weight: QuantizedWeight, // [d_inner, hidden] (z gate)
    pub(super) ssm_a: Tensor,                // F32 [num_heads] (A_log, negative)
    pub(super) ssm_alpha: QuantizedWeight,   // [num_heads, hidden]
    pub(super) ssm_beta: QuantizedWeight,    // [num_heads, hidden]
    pub(super) ssm_conv1d: Tensor,           // F32 [conv_kernel, conv_channels]
    pub(super) ssm_dt_bias: Tensor,          // F32 [num_heads]
    pub(super) ssm_norm: Tensor,             // F32 [head_v_dim]
    pub(super) ssm_out: QuantizedWeight,     // [hidden, d_inner]
    pub(super) ffn_gate_weight: QuantizedWeight,
    pub(super) ffn_up_weight: QuantizedWeight,
    pub(super) ffn_down_weight: QuantizedWeight,
    pub(super) ffn_gate_up_fused: Option<QuantizedWeight>,
    /// Qwen3.5 MoE shared-expert FFN on GDN layers. When set, the dense FFN
    /// weights are placeholders and this shared-expert path owns the FFN.
    pub(super) shared_expert_moe: Option<SharedExpertMoELayerWeights>,
}

pub(super) enum LayerType {
    Attention(AttentionLayerWeights),
    GatedDeltaNet(GdnLayerWeights),
    NemotronMamba2(NemotronMamba2LayerWeights),
    NemotronMoE(NemotronMoELayerWeights),
}

pub(super) struct ModelWeights {
    pub(super) token_embd: QuantizedWeight, // 양자화 상태 유지 (gather 시 on-the-fly dequant)
    pub(super) output_norm: Tensor,         // F32 [hidden]
    pub(super) output: QuantizedWeight,     // [vocab, hidden]
    pub(super) layers: Vec<LayerType>,
    pub(super) gemma_per_layer: Option<GemmaPerLayerWeights>,
    pub(super) glm_dsa_attention: Option<Vec<super::models::glm_dsa::GlmDsaAttentionLayerWeights>>,
    pub(super) rope_freqs: Option<Tensor>,
}

pub(super) struct MtpLayerWeights {
    pub(super) layer_index: usize,
    pub(super) eh_proj: QuantizedWeight,
    pub(super) enorm: Tensor,
    pub(super) hnorm: Tensor,
    pub(super) shared_head_norm: Tensor,
    pub(super) embed_tokens: Option<QuantizedWeight>,
    pub(super) shared_head_head: Option<QuantizedWeight>,
    pub(super) block: AttentionLayerWeights,
    pub(super) glm_dsa_attention: Option<super::models::glm_dsa::GlmDsaAttentionLayerWeights>,
}
