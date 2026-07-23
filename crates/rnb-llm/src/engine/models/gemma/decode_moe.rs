//! Decode-path MoE FFN helpers without backend-specific dispatch.

#![allow(unused_imports)]

use crate::engine::*;

/// Gemma4 26B-A4B hybrid FFN (shared dense MLP + sparse MoE), decode path (seq_len=1).
///
/// Mirrors llama.cpp `llm_build_gemma4_iswa` MoE-layer branch:
/// ```text
///   cur_mlp = ffn_norm(attn_out) -> GELU_PAR ffn(gate/up/down) -> post_ffw_norm_1
///   cur_moe = pre_ffw_norm_2(attn_out)
///   tmp     = rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp_s
///   logits  = ffn_gate_inp @ tmp                       // [n_expert]
///   cur_moe = moe_ffn(cur_moe, logits) -> post_ffw_norm_2
///   cur     = cur_mlp + cur_moe
///   cur     = post_ffw_norm(cur)                        // common post_ffw_norm
///   hidden  = hidden + cur                              // residual
/// ```
///
/// Scalar path only; reuses `QuantizedWeight::gemv_into` and `MoeLayerView::forward_with_logits`.
#[allow(unused_variables)]
pub(in crate::engine) fn decode_ffn_gemma4_moe_hybrid(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    ffn_norm_weight: &Tensor,
    post_ffw_norm_weight: &Option<Tensor>,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    moe_w: &MoeLayerWeights,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<()> {
    let prof_level = super::policy::profiling_level();
    let profiling = prof_level >= 1;
    let verbose = prof_level >= 2;
    macro_rules! prof {
        ($label:expr, $t:expr) => {
            if profiling && (verbose || layer_idx == 0) {
                eprintln!(
                    "  [DEC-MOE L{}] {:20} {:.1}ms",
                    layer_idx,
                    $label,
                    $t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
    }

    // attn_out = scratch.hidden (before residual add). Keep a copy for router input.
    let attn_out: Vec<f32> = scratch.hidden[..hidden_dim].to_vec();

    // ----- SHARED MLP path -----
    let t_mlp = std::time::Instant::now();
    let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
    apply_model_norm_into(
        &attn_out,
        ffn_norm_data,
        norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
        architecture,
    );
    let norm_data = scratch.norm_buf[..hidden_dim].to_vec();
    ffn_gate_weight.gemv_into(&norm_data, &mut scratch.ffn_gate)?;
    ffn_up_weight.gemv_into(&norm_data, &mut scratch.ffn_up)?;
    apply_model_gate_mul_inplace(&mut scratch.ffn_gate, &scratch.ffn_up, architecture);
    ffn_down_weight.gemv_into(&scratch.ffn_gate, &mut scratch.ffn_down)?;
    // post_ffw_norm_1 -> shared_out (save to a local Vec; scratch.ffn_down reused by MoE path)
    let post_norm_1_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_1);
    let mut shared_out = vec![0f32; hidden_dim];
    apply_model_norm_into(
        &scratch.ffn_down[..hidden_dim],
        post_norm_1_data,
        norm_eps,
        &mut shared_out,
        architecture,
    );
    prof!("shared_mlp", t_mlp);

    // ----- MoE branch -----
    let t_moe_in = std::time::Instant::now();
    // pre_ffw_norm_2(attn_out) -> moe_input
    let pre_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.pre_ffw_norm_2);
    let mut moe_input = vec![0f32; hidden_dim];
    apply_model_norm_into(
        &attn_out,
        pre_norm_2_data,
        norm_eps,
        &mut moe_input,
        architecture,
    );
    // Router tmp = rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp_s
    let router_scale_data = kernels::tensor_as_f32_slice(&moe_w.router_scale);
    let mut router_tmp = vec![0f32; hidden_dim];
    {
        let n = hidden_dim as f32;
        let sum_sq: f32 = attn_out.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / n;
        let rrms = 1.0 / (mean_sq + norm_eps).sqrt();
        let inv_sqrt_n = 1.0 / n.sqrt();
        for i in 0..hidden_dim {
            router_tmp[i] = attn_out[i] * rrms * inv_sqrt_n * router_scale_data[i];
        }
    }
    prof!("moe_prenorm+router_tmp", t_moe_in);

    // Router logits [n_expert] = router_w @ tmp  (router_w is F32, shape [n_expert, n_embd])
    let t_router = std::time::Instant::now();
    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("router_f32 missing".into()))?;
    let mut logits = vec![0f32; moe_w.n_expert];
    super::dense_dispatch::gemv_f32(
        router_w_data,
        &router_tmp,
        &mut logits,
        moe_w.n_expert,
        hidden_dim,
        1,
    );
    prof!("router_gemv", t_router);

    // MoeLayerView::forward_with_logits -> moe_out_pre
    let t_experts = std::time::Instant::now();
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
    let mut moe_out_pre = vec![0f32; hidden_dim];
    view.forward_with_logits(&moe_input, &logits, &mut moe_out_pre);
    prof!("moe_experts", t_experts);

    // post_ffw_norm_2 -> moe_out
    let post_norm_2_data = kernels::tensor_as_f32_slice(&moe_w.post_ffw_norm_2);
    let mut moe_out = vec![0f32; hidden_dim];
    apply_model_norm_into(
        &moe_out_pre,
        post_norm_2_data,
        norm_eps,
        &mut moe_out,
        architecture,
    );

    // ----- COMBINE -----
    for i in 0..hidden_dim {
        shared_out[i] += moe_out[i];
    }

    // Common post_ffw_norm (if present)
    if let Some(post_ffw_norm) = post_ffw_norm_weight {
        let pn_data = kernels::tensor_as_f32_slice(post_ffw_norm);
        apply_model_norm_into(
            &shared_out,
            pn_data,
            norm_eps,
            &mut scratch.ffn_down[..hidden_dim],
            architecture,
        );
    } else {
        scratch.ffn_down[..hidden_dim].copy_from_slice(&shared_out);
    }

    // Residual add: hidden += ffn_down (scratch.hidden still holds attn_out)
    add_f32_inplace(
        &mut scratch.hidden[..hidden_dim],
        &scratch.ffn_down[..hidden_dim],
    );

    Ok(())
}
