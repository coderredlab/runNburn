#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(feature = "metal")]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attn_decode_kv_resident_into_if_supported(
    layer: usize,
    q: &[f32],
    k_all: &[u16],
    v_all: &[u16],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    capacity: usize,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> crate::error::Result<bool> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_decode_kv_resident_into_if_supported(
            layer,
            q,
            k_all,
            v_all,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            capacity,
            sliding_window,
            has_softcap,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            layer,
            q,
            k_all,
            v_all,
            &attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            capacity,
            sliding_window,
            has_softcap,
        );
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_rope_mrope_into_if_supported(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    mrope_dim: usize,
    theta: f32,
    pos: usize,
    apply_k: bool,
) -> crate::error::Result<bool> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_rope_mrope_into_if_supported(
            q, k, head_dim, q_dim, kv_dim, mrope_dim, theta, pos, apply_k,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            &q, &k, head_dim, q_dim, kv_dim, mrope_dim, theta, pos, apply_k,
        );
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_inproj_chain_into_if_supported(
    norm_input: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    qkv_out: &mut [f32],
    gate_out: &mut [f32],
    hidden_dim: usize,
    qkv_dim: usize,
    gate_dim: usize,
) -> crate::error::Result<bool> {
    let (Some(qkv_v), Some(gate_v)) = (qkv_weight.backend_view(), gate_weight.backend_view())
    else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_inproj_chain_into_if_supported(
            backend_ggml_type(qkv_v.quant()),
            backend_ggml_type(gate_v.quant()),
            norm_input,
            qkv_v.raw(),
            gate_v.raw(),
            qkv_out,
            gate_out,
            hidden_dim,
            qkv_dim,
            gate_dim,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (qkv_v, gate_v);
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attention_qkv_chain_into_if_supported(
    norm_input: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    hidden_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
    enabled_by_default: bool,
    v_from_k: bool,
) -> crate::error::Result<bool> {
    let (Some(q_v), Some(k_v), Some(v_v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attention_qkv_chain_into_if_supported(
            backend_ggml_type(q_v.quant()),
            backend_ggml_type(k_v.quant()),
            backend_ggml_type(v_v.quant()),
            norm_input,
            q_v.raw(),
            k_v.raw(),
            v_v.raw(),
            q_out,
            k_out,
            v_out,
            hidden_dim,
            q_out_dim,
            kv_dim,
            enabled_by_default,
            v_from_k,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (q_v, k_v, v_v);
        Ok(false)
    }
}
#[cfg(feature = "metal")]
pub(in crate::engine) fn metal_decode_attention_kvarn_into_if_supported(
    layer_index: usize,
    q: &[f32],
    cache: crate::engine::cpu_runtime::quantize::kvarn::KvarnKvView<'_>,
    output: &mut [f32],
    num_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<bool> {
    let request = metal_runtime::KvarnDecodeRequest::new(
        layer_index,
        q,
        cache.device_blocks,
        cache.sink_key,
        cache.sink_value,
        cache.tail_key,
        cache.tail_value,
        cache.len,
        cache.tail_start,
        num_heads,
        cache.num_kv_heads,
        cache.head_dim,
        cache.config.key_bits,
        cache.config.value_bits,
        cache.config.group,
        cache.config.sink_tokens,
        cache.device_layout.block_bytes,
        scale,
        sliding_window,
        softcap,
    );
    metal_runtime::metal_kvarn_attention_decode_into_if_supported(request, output)
        .map_err(crate::error::LlmError::Forward)
}
