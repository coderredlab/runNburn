#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::runtime::QuantFormat;

#[cfg(feature = "cuda")]
fn cuda_error(err: String) -> crate::error::LlmError {
    crate::error::LlmError::Forward(err)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_f16kv_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_attention_f16kv_if_supported(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            has_sliding_window,
            has_softcap,
        )
        .map_err(cuda_error);
    }
    // pm48 ①: Metal flash attention prefill seam(dense causal GQA, head_dim==256, host 입출력).
    // CUDA 는 head_dim==512 만 이 seam 진입(다른 head_dim → None) 인데, Metal 은 simdgroup
    // matmul2d 커널이 head_dim==256 컴파일타임 고정이라 지원 head_dim 이 backend 별로 다르다
    // (CUDA 512 / Metal 256 — 비대칭은 각 backend 의 kernel shape 제약 때문, 의도된 분기).
    // None 반환 시(non-M5 / gate OFF / shape 미충족) caller 가 f16 NEON CPU 로 fallback.
    #[cfg(feature = "metal")]
    {
        return Ok(metal_runtime::metal_prefill_attention_flash_if_supported(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            has_sliding_window,
            has_softcap,
        ));
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Ok(None)
}
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_f16kv_window_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_attention_f16kv_window_if_supported(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            has_softcap,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_f16kv_dense_chain_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<bool> {
    let (Some(o), Some(gate), Some(up), Some(down)) = (
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    if o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_attention_f16kv_dense_chain_if_supported(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            has_sliding_window,
            has_softcap,
            o.raw(),
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_f16kv_window_dense_chain_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<bool> {
    let (Some(o), Some(gate), Some(up), Some(down)) = (
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    if o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_attention_f16kv_window_dense_chain_if_supported(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            has_softcap,
            o.raw(),
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn decode_gemv_into_if_supported(
    weight: &QuantizedWeight,
    input: &[f32],
    output: &mut [f32],
    label: &str,
    rms_used_cuda: bool,
) -> crate::error::Result<bool> {
    let Some(view) = weight.backend_view() else {
        return Ok(false);
    };
    #[cfg(feature = "cuda")]
    {
        // cu42 step 9 + cu45 step 23 + cu57: env opt-in + rms_used_cuda (caller
        // 가 norm_buf_carrier 가 fresh 함을 보장) 일 때만 device-input variant.
        // rms_used_cuda=false 는 cu56 step 63 fix 와 동등 — host scratch.norm_buf
        // 가 input source 임. cu42 path 비활성, cuda fallback gemv 그대로.
        if crate::engine::policy::cuda_decode_device_chain_enabled() && rms_used_cuda {
            let bytes = std::mem::size_of_val(input);
            match view.quant() {
                QuantFormat::Q4K => {
                    if let Ok(carrier) = cuda_runtime::acquire_decode_norm_buf_carrier(bytes) {
                        cuda_runtime::q4k_gemv_with_device_input(
                            view.raw(),
                            view.rows(),
                            view.cols(),
                            carrier,
                            output,
                        )
                        .map_err(cuda_error)?;
                        return Ok(true);
                    }
                }
                QuantFormat::Q6K => {
                    if let Ok(carrier) = cuda_runtime::acquire_decode_norm_buf_carrier(bytes) {
                        cuda_runtime::q6k_gemv_with_device_input(
                            view.raw(),
                            view.rows(),
                            view.cols(),
                            carrier,
                            output,
                        )
                        .map_err(cuda_error)?;
                        return Ok(true);
                    }
                }
                _ => {}
            }
        }
        return cuda_runtime::decode_gemv_into_if_supported(
            backend_ggml_type(view.quant()),
            view.raw(),
            view.rows(),
            view.cols(),
            input,
            output,
            label,
        )
        .map_err(cuda_error);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::decode_gemv_into_if_supported(
            backend_ggml_type(view.quant()),
            view.raw(),
            view.rows(),
            view.cols(),
            input,
            output,
            label,
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Ok(false)
}
