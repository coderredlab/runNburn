//! Shared-expert MoE decode helpers.

use crate::engine::*;

/// Decodes a split sparse-expert MoE FFN plus its shared expert. Writes the
/// FFN output into `scratch.ffn_down[..hidden_dim]` so the caller's residual
/// addition remains identical to the other FFN paths.
///
/// Input: `scratch.hidden[..hidden_dim]` holds attn_out (post o-proj + residual).
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn decode_shared_expert_moe(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    ffn_norm_weight: &Tensor,
    moe_w: &SharedExpertMoELayerWeights,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<()> {
    let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
    apply_model_norm_into(
        &scratch.hidden[..hidden_dim],
        ffn_norm_data,
        norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
        architecture,
    );
    emit_mtp_finite_trace(
        "decode-qwen-moe",
        layer_idx,
        "ffn_norm",
        &scratch.norm_buf[..hidden_dim],
    );

    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE router_f32 failed".into()))?;
    let gate_exps_bytes = moe_w
        .gate_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE gate_exps_bytes failed".into()))?;
    let up_exps_bytes = moe_w
        .up_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE up_exps_bytes failed".into()))?;
    let down_exps_bytes = moe_w
        .down_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE down_exps_bytes failed".into()))?;
    let shared_input_scale = kernels::tensor_as_f32_slice(&moe_w.shared_input_scale);
    let shared_gate_bytes =
        moe_w.shared_gate.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_gate as_bytes failed".into())
        })?;
    let shared_up_bytes =
        moe_w.shared_up.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_up as_bytes failed".into())
        })?;
    let shared_down_bytes =
        moe_w.shared_down.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_down as_bytes failed".into())
        })?;

    let view = crate::engine::moe::SharedExpertMoEView {
        router_w: router_w_data,
        router_selection_bias: moe_w
            .router_selection_bias
            .as_ref()
            .map(kernels::tensor_as_f32_slice),
        expert_gating_func: moe_w.expert_gating_func,
        expert_weights_norm: moe_w.expert_weights_norm,
        expert_weights_scale: moe_w.expert_weights_scale,
        gate_exps_bytes,
        gate_quant: moe_w.gate_quant,
        up_exps_bytes,
        up_quant: moe_w.up_quant,
        down_exps_bytes,
        down_quant: moe_w.down_quant,
        shared_input_scale,
        shared_expert_gated: moe_w.shared_expert_gated,
        shared_gate_bytes,
        shared_gate_quant: moe_w.shared_gate.ggml_type,
        shared_up_bytes,
        shared_up_quant: moe_w.shared_up.ggml_type,
        shared_down_bytes,
        shared_down_quant: moe_w.shared_down.ggml_type,
        n_embd: moe_w.n_embd,
        n_ff: moe_w.n_ff,
        n_expert: moe_w.n_expert,
        n_expert_used: moe_w.n_expert_used,
        layer_idx: Some(layer_idx),
    };

    if view.forward_add_residual_with_policy(
        &scratch.norm_buf[..hidden_dim],
        &mut scratch.ffn_down[..hidden_dim],
        &mut scratch.hidden[..hidden_dim],
        moe_w.prefer_sparse_moe_cuda,
        moe_w.sparse_page_cache.as_deref(),
    ) {
        emit_mtp_finite_trace(
            "decode-qwen-moe",
            layer_idx,
            "after_moe_residual",
            &scratch.hidden[..hidden_dim],
        );
        return Ok(());
    }
    emit_mtp_finite_trace(
        "decode-qwen-moe",
        layer_idx,
        "moe_out",
        &scratch.ffn_down[..hidden_dim],
    );

    add_f32_inplace(
        &mut scratch.hidden[..hidden_dim],
        &scratch.ffn_down[..hidden_dim],
    );
    emit_mtp_finite_trace(
        "decode-qwen-moe",
        layer_idx,
        "after_moe_residual",
        &scratch.hidden[..hidden_dim],
    );
    Ok(())
}
