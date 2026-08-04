#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(feature = "metal")]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_ffn_chain_into_if_supported(
    norm_weight: &[f32],
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    hidden: &mut [f32],
    hidden_dim: usize,
    ffn_dim: usize,
    norm_eps: f32,
) -> crate::error::Result<bool> {
    let (Some(gate_v), Some(up_v), Some(down_v)) = (
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_ffn_chain_into_if_supported(
            backend_ggml_type(gate_v.quant()),
            backend_ggml_type(up_v.quant()),
            backend_ggml_type(down_v.quant()),
            gate_v.raw(),
            up_v.raw(),
            down_v.raw(),
            norm_weight,
            hidden,
            hidden_dim,
            ffn_dim,
            norm_eps,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (gate_v, up_v, down_v);
        Ok(false)
    }
}

/// pm33: prefill FFN batch GEMM chain seam. `metal_ffn_chain_into_if_supported`(decode)의
/// M>1 아날로그. norm 은 caller(normed 입력), residual 도 caller(out = down 결과, residual 전).
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_prefill_ffn_chain_into_if_supported(
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    normed: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
) -> crate::error::Result<bool> {
    let (Some(gate_v), Some(up_v), Some(down_v)) = (
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    // gate weight = [ffn_dim, hidden_dim] → rows = ffn_dim. backend view 가 shape source.
    let ffn_dim = gate_v.rows();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_prefill_ffn_chain_into_if_supported(
            backend_ggml_type(gate_v.quant()),
            backend_ggml_type(up_v.quant()),
            backend_ggml_type(down_v.quant()),
            gate_v.raw(),
            up_v.raw(),
            down_v.raw(),
            normed,
            out,
            seq_len,
            hidden_dim,
            ffn_dim,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (gate_v, up_v, down_v, ffn_dim);
        Ok(false)
    }
}

/// pm35 M2: prefill GDN proj(in_proj/gate) single batch GEMM seam. FFN prefill chain 의 single
/// GEMM 아날로그. n_out=view.rows()(=conv_ch(in_proj) 또는 d_inner(gate)). 성공 시 Some(out[seq*n_out]).
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct MetalProjTrace {
    pub(in crate::engine) role: &'static str,
    pub(in crate::engine) layer_idx: usize,
    pub(in crate::engine) timing_enabled: bool,
}

pub(in crate::engine) fn metal_deepseek4_attention_prefill_batch_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_attention_prefill_batch_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}

pub(in crate::engine) fn metal_deepseek4_attention_prefill_output_batch_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_attention_prefill_output_batch_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}
pub(in crate::engine) fn metal_deepseek4_attention_prefill_compressor_fused_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_attention_prefill_compressor_fused_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}

pub(in crate::engine) fn metal_deepseek4_attention_prefill_index_batch_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_attention_prefill_index_batch_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}

pub(in crate::engine) fn metal_deepseek4_q_front_if_supported(
    q_a: &QuantizedWeight,
    q_norm: &Tensor,
    output_weights: &[&QuantizedWeight],
    input: &[f32],
    eps: f32,
) -> crate::error::Result<Option<(Vec<f32>, Vec<Vec<f32>>)>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(q_a_raw) = q_a.data.as_bytes() else {
            return Ok(None);
        };
        let Some(q_norm_raw) = q_norm.as_bytes() else {
            return Ok(None);
        };
        let Some(output_raw) = output_weights
            .iter()
            .map(|weight| weight.data.as_bytes())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let mut quants = Vec::with_capacity(output_weights.len() + 1);
        quants.push(q_a.ggml_type);
        quants.extend(output_weights.iter().map(|weight| weight.ggml_type));
        let mut raw = Vec::with_capacity(output_weights.len() + 1);
        raw.push(q_a_raw);
        raw.extend(output_raw);
        let mut layout = Vec::with_capacity(output_weights.len() + 1);
        layout.push((q_a.rows, q_a.cols));
        layout.extend(
            output_weights
                .iter()
                .map(|weight| (weight.rows, weight.cols)),
        );
        return metal_runtime::metal_deepseek4_q_front_if_supported(
            &quants, &raw, q_norm_raw, input, &layout, eps,
        )
        .map_err(|err| crate::error::LlmError::Forward(err.to_string()));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (q_a, q_norm, output_weights, input, eps);
        Ok(None)
    }
}

pub(in crate::engine) fn metal_deepseek4_q8_multi_gemv_if_supported(
    weights: &[&QuantizedWeight],
    inputs: &[&[f32]],
) -> crate::error::Result<Option<Vec<Vec<f32>>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(raw) = weights
            .iter()
            .map(|weight| weight.data.as_bytes())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let quants = weights
            .iter()
            .map(|weight| weight.ggml_type)
            .collect::<Vec<_>>();
        let layout = weights
            .iter()
            .map(|weight| (weight.rows, weight.cols))
            .collect::<Vec<_>>();
        return metal_runtime::metal_deepseek4_q8_multi_gemv_if_supported(
            &quants, &raw, inputs, &layout,
        )
        .map_err(|err| crate::error::LlmError::Forward(err.to_string()));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (weights, inputs);
        Ok(None)
    }
}

pub(in crate::engine) fn metal_deepseek4_q8_output_chain_if_supported(
    projection_weights: &[&QuantizedWeight],
    inputs: &[&[f32]],
    final_weight: &QuantizedWeight,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(projection_raw) = projection_weights
            .iter()
            .map(|weight| weight.data.as_bytes())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let Some(final_raw) = final_weight.data.as_bytes() else {
            return Ok(None);
        };
        let projection_quants = projection_weights
            .iter()
            .map(|weight| weight.ggml_type)
            .collect::<Vec<_>>();
        let projection_layout = projection_weights
            .iter()
            .map(|weight| (weight.rows, weight.cols))
            .collect::<Vec<_>>();
        return metal_runtime::metal_deepseek4_q8_output_chain_if_supported(
            &projection_quants,
            &projection_raw,
            inputs,
            &projection_layout,
            final_weight.ggml_type,
            final_raw,
            (final_weight.rows, final_weight.cols),
        )
        .map_err(|err| crate::error::LlmError::Forward(err.to_string()));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (projection_weights, inputs, final_weight);
        Ok(None)
    }
}
pub(in crate::engine) fn metal_deepseek4_prefill_q8_multi_gemm_if_supported(
    weights: &[&QuantizedWeight],
    input: &[f32],
    seq_len: usize,
) -> crate::error::Result<Option<Vec<Vec<f32>>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(raw) = weights
            .iter()
            .map(|weight| weight.data.as_bytes())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let quants = weights
            .iter()
            .map(|weight| weight.ggml_type)
            .collect::<Vec<_>>();
        let layout = weights
            .iter()
            .map(|weight| (weight.rows, weight.cols))
            .collect::<Vec<_>>();
        return metal_runtime::metal_deepseek4_prefill_q8_multi_gemm_if_supported(
            &quants, &raw, input, seq_len, &layout,
        )
        .map_err(|err| crate::error::LlmError::Forward(err.to_string()));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (weights, input, seq_len);
        Ok(None)
    }
}

pub(in crate::engine) fn metal_deepseek4_moe_prefill_batch_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_moe_prefill_batch_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}

pub(in crate::engine) fn metal_deepseek4_moe_decode_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_moe_decode_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        false
    }
}

pub(in crate::engine) fn metal_deepseek4_attention_prefill_batch_tokens(
    seq_len: usize,
    scratch_bytes_per_token: usize,
) -> usize {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_deepseek4_attention_prefill_batch_tokens(
            seq_len,
            scratch_bytes_per_token,
        );
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = scratch_bytes_per_token;
        seq_len.min(1)
    }
}

pub(in crate::engine) fn metal_prefill_gdn_proj_into_if_supported(
    weight: &QuantizedWeight,
    normed: &[f32],
    seq_len: usize,
    hidden_dim: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_prefill_gdn_proj_into_if_supported_with_trace(
            weight, normed, seq_len, hidden_dim, None,
        );
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (weight, normed, seq_len, hidden_dim);
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_gdn_proj_into_if_supported_with_trace(
    weight: &QuantizedWeight,
    normed: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    trace: Option<MetalProjTrace>,
) -> crate::error::Result<Option<Vec<f32>>> {
    let Some(view) = weight.backend_view() else {
        return Ok(None);
    };
    // weight = [n_out, hidden] → rows = n_out. backend view 가 shape source (weight.rows() 아님).
    let n_out = view.rows();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let mut out = vec![0f32; seq_len * n_out];
        let runtime_trace = trace.map(|trace| metal_runtime::MetalPrefillProjTrace {
            role: trace.role,
            layer_idx: trace.layer_idx,
            timing_enabled: trace.timing_enabled,
        });
        let used = metal_runtime::metal_prefill_gdn_proj_into_if_supported_with_trace(
            backend_ggml_type(view.quant()),
            view.raw(),
            normed,
            &mut out,
            seq_len,
            hidden_dim,
            n_out,
            runtime_trace,
        )
        .map_err(|e| crate::error::LlmError::Forward(e))?;
        return Ok(if used { Some(out) } else { None });
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (view, n_out, normed, seq_len, hidden_dim);
        Ok(None)
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_gdn_f32_dual_proj_if_supported(
    left: &QuantizedWeight,
    right: &QuantizedWeight,
    normed: &[f32],
    seq_len: usize,
    hidden_dim: usize,
) -> crate::error::Result<Option<(Vec<f32>, Vec<f32>)>> {
    let (Some(left_view), Some(right_view)) = (left.backend_view(), right.backend_view()) else {
        return Ok(None);
    };
    if left_view.rows() != right_view.rows() || left_view.cols() != right_view.cols() {
        return Ok(None);
    }
    let n_out = left_view.rows();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_prefill_gdn_f32_dual_proj_if_supported(
            backend_ggml_type(left_view.quant()),
            backend_ggml_type(right_view.quant()),
            left_view.raw(),
            right_view.raw(),
            normed,
            seq_len,
            hidden_dim,
            n_out,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (left_view, right_view, n_out, normed, seq_len, hidden_dim);
        Ok(None)
    }
}

/// pm39 M3: prefill GDN delta scan(순차 recurrence)을 Metal GPU chunkwise parallel scan 으로.
/// `state` in-place hand-off, 성공 시 Some(output[seq_len*num_heads*head_v_dim]). GQA 는 caller 가
/// q/k 를 num_heads(=num_v_heads) 로 repeat 푼 뒤 넘긴다. opt-in RNB_METAL_PREFILL_GDN_SCAN=1.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_delta_net_scan_into_if_supported(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let mut out = vec![0f32; seq_len * num_heads * head_v_dim];
        let used = metal_runtime::metal_prefill_delta_net_scan_into_if_supported(
            q, k, v, gate, beta, state, &mut out, seq_len, num_heads, head_k_dim, head_v_dim,
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(if used { Some(out) } else { None });
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            q, k, v, gate, beta, state, seq_len, num_heads, head_k_dim, head_v_dim,
        );
        Ok(None)
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attention_o_chain_into_if_supported(
    attn_out: &[f32],
    o_weight: &QuantizedWeight,
    hidden: &mut [f32],
    hidden_dim: usize,
    q_dim: usize,
) -> crate::error::Result<bool> {
    let Some(o_v) = o_weight.backend_view() else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attention_o_chain_into_if_supported(
            backend_ggml_type(o_v.quant()),
            attn_out,
            o_v.raw(),
            hidden,
            hidden_dim,
            q_dim,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = o_v;
        Ok(false)
    }
}
