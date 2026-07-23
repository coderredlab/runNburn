//! Prefill MoE forward helpers.

use super::prefill_moe_expert_major::{
    forward_ffn_gemma4_moe_expert_major, gemma4_moe_expert_major_enabled,
};
use super::select_ffn_pre_norm_weight;
use crate::engine::cpu_runtime::kernels;
use crate::engine::dense_dispatch;
use crate::engine::layer_weights::{AttentionLayerWeights, MoeLayerWeights};
use crate::engine::norm::{add_tensors, apply_model_gate_mul_inplace, apply_model_norm_into};
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;

/// Gemma4 26B-A4B hybrid FFN (shared dense MLP + sparse MoE), prefill path (seq_len ≥ 1).
///
/// Uses expert-major batched execution for GGUF-direct multi-token prefill and the
/// token-major path for single-token, mixed-precision, and residency-backed execution.
/// Layout per token (matches llama.cpp `llm_build_gemma4_iswa` MoE branch):
/// ```text
///   shared_mlp = ffn_norm(attn_out_t) → gate/up GELU-tanh → down → post_ffw_norm_1
///   moe_input  = pre_ffw_norm_2(attn_out_t)
///   router_tmp = rms_norm(attn_out_t) * (1/sqrt(n_embd)) * ffn_gate_inp_s
///   logits     = router_w @ router_tmp
///   moe_out    = MoeLayerView::forward_with_logits(moe_input, logits) → post_ffw_norm_2
///   combined   = shared_mlp + moe_out
///   ffn_out_t  = post_ffw_norm(combined)             // 공통 post_ffw_norm
///   hidden_t  += ffn_out_t                            // residual (caller side)
/// ```
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn forward_ffn_gemma4_moe_hybrid(
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    moe_w: &MoeLayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<Tensor> {
    if gemma4_moe_expert_major_enabled(moe_w, seq_len) {
        return forward_ffn_gemma4_moe_expert_major(
            architecture,
            hidden,
            w,
            moe_w,
            seq_len,
            hidden_dim,
            norm_eps,
            layer_idx,
        );
    }
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());

    let attn_out_data: Vec<f32> = kernels::tensor_as_f32_slice(&hidden).to_vec();

    let ffn_norm_w = select_ffn_pre_norm_weight(w, architecture);
    let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm_w);
    let post_norm_1_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_1);
    let pre_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.pre_ffw_norm_2);
    let post_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_2);
    let router_scale_data = kernels::tensor_as_f32_slice(&moe_w.router_scale);
    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("router_f32 missing".into()))?;
    let gate_up_bytes = moe_w
        .gate_up_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("gate_up bytes missing".into()))?;
    let down_bytes = moe_w
        .down_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("down bytes missing".into()))?;

    let view = crate::engine::moe::MoeLayerView {
        router_w: router_w_data, // unused by forward_with_logits
        gate_up_bytes,
        down_bytes,
        down_scale: kernels::tensor_as_f32_slice(&moe_w.down_scale),
        down_quant: moe_w.down_quant,
        n_embd: moe_w.n_embd,
        n_ff: moe_w.n_ff,
        n_expert: moe_w.n_expert,
        n_expert_used: moe_w.n_expert_used,
        layer_idx: Some(layer_idx),
    };

    // Per-token reusable buffers
    let mut norm_buf = vec![0f32; hidden_dim];
    let mut ffn_gate = vec![0f32; w.ffn_gate_weight.rows];
    let mut ffn_up = vec![0f32; w.ffn_up_weight.rows];
    let mut ffn_down_t = vec![0f32; hidden_dim];
    let mut shared_out = vec![0f32; hidden_dim];
    let mut moe_input = vec![0f32; hidden_dim];
    let mut router_tmp = vec![0f32; hidden_dim];
    let mut logits = vec![0f32; moe_w.n_expert];
    let mut moe_out_pre = vec![0f32; hidden_dim];
    let mut moe_out = vec![0f32; hidden_dim];

    let mut down_buf = vec![0f32; seq_len * hidden_dim];
    let inv_sqrt_n = 1.0 / (hidden_dim as f32).sqrt();

    for t in 0..seq_len {
        let attn_out_t = &attn_out_data[t * hidden_dim..(t + 1) * hidden_dim];

        // --- Shared MLP ---
        apply_model_norm_into(
            attn_out_t,
            ffn_norm_data,
            norm_eps,
            &mut norm_buf,
            architecture,
        );
        w.ffn_gate_weight.gemv_into(&norm_buf, &mut ffn_gate)?;
        w.ffn_up_weight.gemv_into(&norm_buf, &mut ffn_up)?;
        apply_model_gate_mul_inplace(&mut ffn_gate, &ffn_up, architecture);
        w.ffn_down_weight.gemv_into(&ffn_gate, &mut ffn_down_t)?;
        apply_model_norm_into(
            &ffn_down_t,
            post_norm_1_data,
            norm_eps,
            &mut shared_out,
            architecture,
        );

        // --- MoE branch ---
        apply_model_norm_into(
            attn_out_t,
            pre_norm_2_data,
            norm_eps,
            &mut moe_input,
            architecture,
        );
        let sum_sq: f32 = attn_out_t.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / (hidden_dim as f32);
        let rrms = 1.0 / (mean_sq + norm_eps).sqrt();
        for i in 0..hidden_dim {
            router_tmp[i] = attn_out_t[i] * rrms * inv_sqrt_n * router_scale_data[i];
        }
        dense_dispatch::gemv_f32(
            router_w_data,
            &router_tmp,
            &mut logits,
            moe_w.n_expert,
            hidden_dim,
            1,
        );
        view.forward_with_logits(&moe_input, &logits, &mut moe_out_pre);
        apply_model_norm_into(
            &moe_out_pre,
            post_norm_2_data,
            norm_eps,
            &mut moe_out,
            architecture,
        );

        // --- Combine ---
        for i in 0..hidden_dim {
            shared_out[i] += moe_out[i];
        }

        let dst = &mut down_buf[t * hidden_dim..(t + 1) * hidden_dim];
        if let Some(post_ffw_norm) = &w.post_ffw_norm {
            let pn_data = kernels::tensor_as_f32_slice(post_ffw_norm);
            apply_model_norm_into(&shared_out, pn_data, norm_eps, dst, architecture);
        } else {
            dst.copy_from_slice(&shared_out);
        }
    }

    let down = Tensor::from_vec(down_buf, &[seq_len, hidden_dim]);
    let hidden = add_tensors(&hidden, &down).map_err(fwd)?;
    Ok(hidden)
}
