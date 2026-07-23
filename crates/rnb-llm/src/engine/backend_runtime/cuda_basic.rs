#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
// `metal_decode_chain_run` 시그니처(attn/gdn 혼합 enum)에서 항상 참조(facade 는 비-metal
// 빌드에서도 dead_code stub 으로 컴파일되므로 import 도 무조건 필요).
#[allow(unused_imports)]
use crate::engine::layer_weights::GdnLayerWeights;
#[allow(unused_imports)]
use crate::engine::layer_weights::LayerType;
use crate::engine::layer_weights::ModelWeights;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::runtime::QuantFormat;
#[cfg(any(feature = "cuda", feature = "metal"))]
use rnb_loader::GGMLType;

#[cfg(feature = "cuda")]
fn cuda_error(err: String) -> crate::error::LlmError {
    crate::error::LlmError::Forward(err)
}

#[cfg(feature = "cuda")]
fn gemma4_ple_fusion_trace_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_GEMMA_PLE_FUSION_TRACE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn trace_gemma4_ple_fusion_backend(message: impl std::fmt::Display) {
    if gemma4_ple_fusion_trace_enabled() {
        eprintln!("[gemma4-ple-fusion-backend] {message}");
    }
}

#[cfg(feature = "cuda")]
type Q4PackedGateUpPrewarm<'a> = (&'a [u8], &'a [u8], usize, usize);
#[cfg(feature = "cuda")]
type Q4PackedSinglePrewarm<'a> = (&'a [u8], usize, usize);
#[cfg(feature = "cuda")]
type Q4RawQuantPrewarm<'a> = &'a [u8];
#[cfg(feature = "cuda")]
type Q6PackedDownPrewarm<'a> = (&'a [u8], usize, usize);

#[cfg(all(test, feature = "cuda"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaProductPrewarmRequestKind {
    Q4PackedGateUp,
    Q4PackedSingle,
    Q4RawQuant,
    Q6PackedDown,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
struct CudaProductPrewarmRequests<'a> {
    q4_gate_up: Vec<Q4PackedGateUpPrewarm<'a>>,
    q4_single: Vec<Q4PackedSinglePrewarm<'a>>,
    q4_raw: Vec<Q4RawQuantPrewarm<'a>>,
    q6_down: Vec<Q6PackedDownPrewarm<'a>>,
}

#[cfg(all(test, feature = "cuda"))]
impl CudaProductPrewarmRequests<'_> {
    fn kinds(&self) -> Vec<CudaProductPrewarmRequestKind> {
        let mut kinds = Vec::new();
        if !self.q4_gate_up.is_empty() {
            kinds.push(CudaProductPrewarmRequestKind::Q4PackedGateUp);
        }
        if !self.q4_single.is_empty() {
            kinds.push(CudaProductPrewarmRequestKind::Q4PackedSingle);
        }
        if !self.q4_raw.is_empty() {
            kinds.push(CudaProductPrewarmRequestKind::Q4RawQuant);
        }
        if !self.q6_down.is_empty() {
            kinds.push(CudaProductPrewarmRequestKind::Q6PackedDown);
        }
        kinds
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum CudaProductPrewarmSelection {
    Q4Dense,
    Q6Dense,
}

fn ple_dense_chain_quant_supported(gate: QuantFormat, proj: QuantFormat) -> bool {
    matches!(
        (gate, proj),
        (QuantFormat::Q4K, QuantFormat::Q4K) | (QuantFormat::F32, QuantFormat::F32)
    )
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_output_logits_cuda(
    weights: &ModelWeights,
    normed_data: &[f32],
) -> Option<Vec<f32>> {
    weights.output.data.as_bytes().and_then(|raw| {
        cuda_runtime::prefill_output_logits(
            weights.output.ggml_type,
            weights.output.rows,
            weights.output.cols,
            raw,
            normed_data,
        )
    })
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_output_argmax_token_cuda(
    weights: &ModelWeights,
    normed_data: &[f32],
) -> Option<u32> {
    weights.output.data.as_bytes().and_then(|raw| {
        cuda_runtime::try_output_argmax_token(
            weights.output.ggml_type,
            weights.output.rows,
            weights.output.cols,
            raw,
            normed_data,
        )
    })
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn mtp_output_argmax_token_cuda(
    head: &QuantizedWeight,
    normed_data: &[f32],
) -> Option<u32> {
    head.data.as_bytes().and_then(|raw| {
        cuda_runtime::try_output_argmax_token(
            head.ggml_type,
            head.rows,
            head.cols,
            raw,
            normed_data,
        )
    })
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_gemv_cuda(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    seq_len: usize,
) -> Option<crate::error::Result<Vec<f32>>> {
    cuda_runtime::prefill_gemv(
        weight.ggml_type,
        bytes,
        weight.rows,
        weight.cols,
        input,
        seq_len,
    )
    .map(|result| result.map_err(cuda_error))
}

#[allow(clippy::type_complexity)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
) -> crate::error::Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if seq_len <= 1 {
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    if q.cols() != k.cols() || q.cols() != v.cols() || k.rows() != v.rows() {
        return Ok(None);
    }
    if input.len() != seq_len.saturating_mul(q.cols()) {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_qkv_gemm(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            input,
            seq_len,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_attention_hd512_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> crate::error::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>> {
    if seq_len <= 1 || q_norm.len() != 512 || k_norm.len() != 512 {
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(512)
        || k.rows() != num_kv_heads.saturating_mul(512)
        || input.len() != seq_len.saturating_mul(q.cols())
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_qkv_attention_hd512(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            input,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<(Vec<u16>, Vec<u16>)>> {
    if seq_len <= 1
        || pos_start != 0
        || q_norm.len() != 512
        || k_norm.len() != 512
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(512)
        || k.rows() != num_kv_heads.saturating_mul(512)
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_qkv_attention_hd512_dense_chain(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            hidden_input,
            attn_norm_weight,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
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
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_device_output_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    hidden_input: &[f32],
    hidden_input_device: Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<(Vec<u16>, Vec<u16>, NemotronDeviceLayerOutput)>> {
    if seq_len <= 1
        || pos_start != 0
        || q_norm.len() != 512
        || k_norm.len() != 512
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd512-device reject=shape seq_len={} q_norm={} k_norm={} hidden={} expected_hidden={}",
            seq_len,
            q_norm.len(),
            k_norm.len(),
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        trace_gemma4_ple_fusion_backend("qkv-hd512-device reject=missing_backend_view");
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd512-device reject=quant q={:?} k={:?} v={:?} o={:?} gate={:?} up={:?}",
            q.quant(),
            k.quant(),
            v.quant(),
            o.quant(),
            gate.quant(),
            up.quant()
        ));
        return Ok(None);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            trace_gemma4_ple_fusion_backend("qkv-hd512-device reject=partial_ple");
            return Ok(None);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            trace_gemma4_ple_fusion_backend(format!(
                "qkv-hd512-device reject=ple_quant gate={:?} proj={:?}",
                ple_gate.quant(),
                ple_proj.quant()
            ));
            return Ok(None);
        }
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(512)
        || k.rows() != num_kv_heads.saturating_mul(512)
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd512-device reject=qkv_shape q={}x{} k={}x{} v={}x{} input={} attn_norm={} o_cols={}",
            q.rows(),
            q.cols(),
            k.rows(),
            k.cols(),
            v.rows(),
            v.cols(),
            hidden_input.len(),
            attn_norm_weight.len(),
            o_cols
        ));
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device.as_ref() {
        if desc.rows() != seq_len
            || desc.cols() != q.cols()
            || desc.dtype() != cuda_runtime::ScalarType::F32
        {
            trace_gemma4_ple_fusion_backend(format!(
                "qkv-hd512-device reject=input_device rows={} cols={} dtype={:?} expected_rows={} expected_cols={}",
                desc.rows(),
                desc.cols(),
                desc.dtype(),
                seq_len,
                q.cols()
            ));
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(q.cols()) {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd512-device reject=input_len input={} expected={}",
            hidden_input.len(),
            seq_len.saturating_mul(q.cols())
        ));
        return Ok(None);
    }
    cuda_runtime::prefill_q4k_f16_qkv_attention_hd512_dense_chain_device_output(
        q.raw(),
        k.raw(),
        v.raw(),
        q.rows(),
        k.rows(),
        q.cols(),
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        k_norm,
        freq_factors,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
        o.raw(),
        gate.raw(),
        up.raw(),
        down.raw(),
        backend_ggml_type(down.quant()),
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate.as_ref().map(|view| view.raw()),
        ple_proj.as_ref().map(|view| view.raw()),
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output| {
        output.map(|output| {
            (
                output.k_bits,
                output.v_bits,
                NemotronDeviceLayerOutput {
                    output_id: output.output_id,
                    output_desc: output.output_desc,
                },
            )
        })
    })
    .map_err(cuda_error)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_if_supported(
    q_weight: &QuantizedWeight,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<bool> {
    if seq_len <= 1 || q_norm.len() != 512 || hidden.len() != seq_len.saturating_mul(n_embd) {
        return Ok(false);
    }
    let (Some(q), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    if q.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(false);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            return Ok(false);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            return Ok(false);
        }
    }
    if q.rows() != num_heads.saturating_mul(512)
        || hidden_input.len() != seq_len.saturating_mul(q.cols())
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain(
            q.raw(),
            q.rows(),
            q.cols(),
            hidden_input,
            attn_norm_weight,
            q_norm,
            freq_factors,
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            o.raw(),
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            ple_gate.as_ref().map(|view| view.raw()),
            ple_proj.as_ref().map(|view| view.raw()),
            ple_post_norm_weight,
            ple_input,
            ple_dim,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            unit_offset_attn_norm,
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
#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output_if_supported(
    q_weight: &QuantizedWeight,
    hidden_input: &[f32],
    hidden_input_device: Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<NemotronDeviceLayerOutput>> {
    if seq_len <= 1 || q_norm.len() != 512 || hidden.len() != seq_len.saturating_mul(n_embd) {
        trace_gemma4_ple_fusion_backend(format!(
            "hd512-device reject=shape seq_len={} q_norm={} hidden={} expected_hidden={}",
            seq_len,
            q_norm.len(),
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
        return Ok(None);
    }
    let (Some(q), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        trace_gemma4_ple_fusion_backend("hd512-device reject=missing_backend_view");
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        trace_gemma4_ple_fusion_backend(format!(
            "hd512-device reject=quant q={:?} o={:?} gate={:?} up={:?}",
            q.quant(),
            o.quant(),
            gate.quant(),
            up.quant()
        ));
        return Ok(None);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            trace_gemma4_ple_fusion_backend("hd512-device reject=partial_ple");
            return Ok(None);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            trace_gemma4_ple_fusion_backend(format!(
                "hd512-device reject=ple_quant gate={:?} proj={:?}",
                ple_gate.quant(),
                ple_proj.quant()
            ));
            return Ok(None);
        }
    }
    if q.rows() != num_heads.saturating_mul(512)
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        trace_gemma4_ple_fusion_backend(format!(
            "hd512-device reject=q_shape q_rows={} expected_q_rows={} attn_norm={} q_cols={} o_cols={}",
            q.rows(),
            num_heads.saturating_mul(512),
            attn_norm_weight.len(),
            q.cols(),
            o_cols
        ));
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len
            || desc.cols() != q.cols()
            || desc.dtype() != cuda_runtime::ScalarType::F32
        {
            trace_gemma4_ple_fusion_backend(format!(
                "hd512-device reject=device_input_desc rows={} cols={} dtype={:?} expected_rows={} expected_cols={}",
                desc.rows(),
                desc.cols(),
                desc.dtype(),
                seq_len,
                q.cols()
            ));
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(q.cols()) {
        trace_gemma4_ple_fusion_backend(format!(
            "hd512-device reject=host_input_len got={} expected={}",
            hidden_input.len(),
            seq_len.saturating_mul(q.cols())
        ));
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        let output =
            cuda_runtime::prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output(
                q.raw(),
                q.rows(),
                q.cols(),
                hidden_input,
                hidden_input_device,
                attn_norm_weight,
                q_norm,
                freq_factors,
                cached_k_f16,
                cached_v_f16,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
                rope_theta,
                pos_start,
                norm_eps,
                q_unit_offset,
                o.raw(),
                gate.raw(),
                up.raw(),
                down.raw(),
                backend_ggml_type(down.quant()),
                post_attn_norm_weight,
                ffn_norm_weight,
                post_ffn_norm_weight,
                ple_gate.as_ref().map(|view| view.raw()),
                ple_proj.as_ref().map(|view| view.raw()),
                ple_post_norm_weight,
                ple_input,
                ple_dim,
                o_cols,
                n_ff,
                n_embd,
                hidden,
                layer_out_scale,
                unit_offset_attn_norm,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )
            .map(|output| {
                output.map(|output| NemotronDeviceLayerOutput {
                    output_id: output.output_id,
                    output_desc: output.output_desc,
                })
            })
            .map_err(cuda_error)?;
        if output.is_some() {
            trace_gemma4_ple_fusion_backend("hd512-device ok=device_output");
        } else {
            trace_gemma4_ple_fusion_backend("hd512-device reject=runtime_none");
        }
        return Ok(output);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_if_supported(
    q_weight: &QuantizedWeight,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    window: usize,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<bool> {
    if seq_len <= 1
        || window == 0
        || q_norm.len() != 256
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        return Ok(false);
    }
    let (Some(q), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    if q.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(false);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            return Ok(false);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            return Ok(false);
        }
    }
    if q.rows() != num_heads.saturating_mul(256)
        || hidden_input.len() != seq_len.saturating_mul(q.cols())
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain(
            q.raw(),
            q.rows(),
            q.cols(),
            hidden_input,
            attn_norm_weight,
            q_norm,
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            window,
            o.raw(),
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            ple_gate.as_ref().map(|view| view.raw()),
            ple_proj.as_ref().map(|view| view.raw()),
            ple_post_norm_weight,
            ple_input,
            ple_dim,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            unit_offset_attn_norm,
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
#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output_if_supported(
    q_weight: &QuantizedWeight,
    hidden_input: &[f32],
    hidden_input_device: Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    window: usize,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<NemotronDeviceLayerOutput>> {
    if seq_len <= 1
        || window == 0
        || q_norm.len() != 256
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        return Ok(None);
    }
    let (Some(q), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            return Ok(None);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            return Ok(None);
        }
    }
    if q.rows() != num_heads.saturating_mul(256)
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
    {
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len
            || desc.cols() != q.cols()
            || desc.dtype() != cuda_runtime::ScalarType::F32
        {
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(q.cols()) {
        return Ok(None);
    }
    cuda_runtime::prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output(
        q.raw(),
        q.rows(),
        q.cols(),
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        cached_k_f16,
        cached_v_f16,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        window,
        o.raw(),
        gate.raw(),
        up.raw(),
        down.raw(),
        backend_ggml_type(down.quant()),
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate.as_ref().map(|view| view.raw()),
        ple_proj.as_ref().map(|view| view.raw()),
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output| {
        output.map(|output| NemotronDeviceLayerOutput {
            output_id: output.output_id,
            output_desc: output.output_desc,
        })
    })
    .map_err(cuda_error)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_postprocess_hd256_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> crate::error::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>> {
    if seq_len <= 1 || q_norm.len() != 256 || k_norm.len() != 256 {
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(256)
        || k.rows() != num_kv_heads.saturating_mul(256)
        || input.len() != seq_len.saturating_mul(q.cols())
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_qkv_postprocess_hd256(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            input,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    hidden_input: &[f32],
    hidden_input_device: Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    window: usize,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<(Vec<u16>, Vec<u16>, NemotronDeviceLayerOutput)>> {
    if seq_len <= 1 || pos_start != 0 || q_norm.len() != 256 || k_norm.len() != 256 || window == 0 {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd256-device reject=shape seq_len={} pos_start={} q_norm={} k_norm={} window={}",
            seq_len,
            pos_start,
            q_norm.len(),
            k_norm.len(),
            window
        ));
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        trace_gemma4_ple_fusion_backend("qkv-hd256-device reject=missing_backend_view");
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || !matches!(v.quant(), QuantFormat::Q4K | QuantFormat::Q6K)
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd256-device reject=quant q={:?} k={:?} v={:?} o={:?} gate={:?} up={:?}",
            q.quant(),
            k.quant(),
            v.quant(),
            o.quant(),
            gate.quant(),
            up.quant()
        ));
        return Ok(None);
    }
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some()
        || ple_proj_weight.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
    {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            trace_gemma4_ple_fusion_backend("qkv-hd256-device reject=partial_ple");
            return Ok(None);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            trace_gemma4_ple_fusion_backend(format!(
                "qkv-hd256-device reject=ple_quant gate={:?} proj={:?}",
                ple_gate.quant(),
                ple_proj.quant()
            ));
            return Ok(None);
        }
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(256)
        || k.rows() != num_kv_heads.saturating_mul(256)
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd256-device reject=qkv_shape q={}x{} k={}x{} v={}x{} input={} attn_norm={} o_cols={} hidden={} expected_hidden={}",
            q.rows(),
            q.cols(),
            k.rows(),
            k.cols(),
            v.rows(),
            v.cols(),
            hidden_input.len(),
            attn_norm_weight.len(),
            o_cols,
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device.as_ref() {
        if desc.rows() != seq_len
            || desc.cols() != q.cols()
            || desc.dtype() != cuda_runtime::ScalarType::F32
        {
            trace_gemma4_ple_fusion_backend(format!(
                "qkv-hd256-device reject=input_device rows={} cols={} dtype={:?} expected_rows={} expected_cols={}",
                desc.rows(),
                desc.cols(),
                desc.dtype(),
                seq_len,
                q.cols()
            ));
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(q.cols()) {
        trace_gemma4_ple_fusion_backend(format!(
            "qkv-hd256-device reject=input_len input={} expected={}",
            hidden_input.len(),
            seq_len.saturating_mul(q.cols())
        ));
        return Ok(None);
    }
    cuda_runtime::prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output(
        q.raw(),
        k.raw(),
        v.raw(),
        backend_ggml_type(v.quant()),
        q.rows(),
        k.rows(),
        q.cols(),
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        k_norm,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
        window,
        o.raw(),
        gate.raw(),
        up.raw(),
        down.raw(),
        backend_ggml_type(down.quant()),
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate.as_ref().map(|view| view.raw()),
        ple_proj.as_ref().map(|view| view.raw()),
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output| {
        output.map(|output| {
            (
                output.k_bits,
                output.v_bits,
                NemotronDeviceLayerOutput {
                    output_id: output.output_id,
                    output_desc: output.output_desc,
                },
            )
        })
    })
    .map_err(cuda_error)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    window: usize,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<Option<(Vec<u16>, Vec<u16>)>> {
    if seq_len <= 1 || pos_start != 0 || q_norm.len() != 256 || k_norm.len() != 256 || window == 0 {
        return Ok(None);
    }
    let (Some(q), Some(k), Some(v), Some(o), Some(gate), Some(up), Some(down)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || !matches!(v.quant(), QuantFormat::Q4K | QuantFormat::Q6K)
        || o.quant() != QuantFormat::Q4K
        || gate.quant() != QuantFormat::Q4K
        || up.quant() != QuantFormat::Q4K
    {
        return Ok(None);
    }
    if q.cols() != k.cols()
        || q.cols() != v.cols()
        || k.rows() != v.rows()
        || q.rows() != num_heads.saturating_mul(256)
        || k.rows() != num_kv_heads.saturating_mul(256)
        || hidden_input.len() != seq_len.saturating_mul(q.cols())
        || attn_norm_weight.len() != q.cols()
        || o_cols != q.rows()
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            q.raw(),
            k.raw(),
            v.raw(),
            backend_ggml_type(v.quant()),
            q.rows(),
            k.rows(),
            q.cols(),
            hidden_input,
            attn_norm_weight,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
            window,
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
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn decode_gemv_cuda(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
) -> Option<crate::error::Result<Vec<f32>>> {
    if !decode_gemv_cuda_supports_type(weight.ggml_type) {
        return None;
    }
    cuda_runtime::decode_gemv(weight.ggml_type, bytes, weight.rows, weight.cols, input)
        .map(|result| result.map_err(cuda_error))
}

#[cfg(feature = "cuda")]
fn decode_gemv_cuda_supports_type(ggml_type: GGMLType) -> bool {
    !matches!(ggml_type, GGMLType::I32)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_decode_gemv_supports_every_engine_weight_type() {
        let supported = [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::BF16,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q5_0,
            GGMLType::Q5_1,
            GGMLType::Q8_0,
            GGMLType::Q8_1,
            GGMLType::Q2_K,
            GGMLType::Q3_K,
            GGMLType::Q4_K,
            GGMLType::Q5_K,
            GGMLType::Q6_K,
            GGMLType::IQ2_XXS,
            GGMLType::IQ2_S,
            GGMLType::IQ3_XXS,
            GGMLType::IQ4_XS,
        ];
        assert!(supported.into_iter().all(decode_gemv_cuda_supports_type));
        assert!(!decode_gemv_cuda_supports_type(GGMLType::I32));
    }

    #[test]
    fn nemotron_device_mamba_trace_bytes_separate_hidden_and_state() {
        let trace = super::NemotronMamba2DeviceTrace {
            boundary_d2h_bytes: 0,
            hidden_d2h_bytes: 0,
            conv_state_d2h_bytes: 96,
            delta_state_d2h_bytes: 128,
        };

        assert_eq!(trace.mamba2_state_d2h_bytes(), 224);
    }

    #[test]
    fn nemotron_device_route_pack_wrapper_reorders_experts() {
        let seq_len = 2usize;
        let n_expert = 4usize;
        let expert_used = 2usize;
        let logits = [0.5_f32, -0.25, 1.0, 0.125, -0.5, 0.75, 0.25, 1.25];
        let bias = [0.0_f32, 0.1, -0.2, 0.0];
        let logits_desc = cuda_runtime::DeviceTensorDesc::new(
            seq_len,
            n_expert,
            cuda_runtime::ScalarType::F32,
            cuda_runtime::DeviceTensorRole::RouterLogits,
        );
        let logits_id =
            cuda_runtime::upload_device_tensor_f32(logits_desc, &logits).expect("upload logits");

        let route = super::nemotron_device_route_pack_from_logits(
            logits_id,
            logits_desc,
            Some(&bias),
            seq_len,
            n_expert,
            expert_used,
            1.0,
        )
        .expect("route pack");

        assert_eq!(route.slots(), seq_len * expert_used);
        let experts =
            super::nemotron_device_route_pack_expert_ids(&route).expect("download expert ids");
        let order = [1_u32, 0, 3, 2];
        let sorted =
            super::nemotron_reorder_device_route_pack(&route, &order).expect("reorder route");
        let sorted_experts = super::nemotron_device_route_pack_expert_ids(&sorted)
            .expect("download sorted expert ids");
        assert_eq!(
            sorted_experts,
            order
                .iter()
                .map(|&idx| experts[idx as usize])
                .collect::<Vec<_>>()
        );

        super::release_nemotron_device_route_pack(sorted).expect("release sorted route");
        super::release_nemotron_device_route_pack(route).expect("release route");
        assert!(cuda_runtime::release_device_tensor(logits_id).expect("release logits"));
    }

    #[test]
    fn cuda_cleanup_failure_message_preserves_primary_and_cleanup_state() {
        assert_eq!(
            super::cuda_cleanup_failure_message("input release failed", "output", Ok(true)),
            "input release failed"
        );

        let missing =
            super::cuda_cleanup_failure_message("input release failed", "output", Ok(false));
        assert!(missing.contains("input release failed"));
        assert!(missing.contains("output cleanup missing"));

        let failed = super::cuda_cleanup_failure_message(
            "input release failed",
            "output",
            Err("boom".to_string()),
        );
        assert!(failed.contains("input release failed"));
        assert!(failed.contains("output cleanup failed: boom"));
    }

    #[test]
    fn release_mamba2_input_or_cleanup_output_skips_output_when_input_released() {
        let mut output_cleanup_called = false;
        let result = super::release_mamba2_input_or_cleanup_output(
            || Ok(true),
            || {
                output_cleanup_called = true;
                Ok(true)
            },
        );

        assert_eq!(result, Ok(()));
        assert!(!output_cleanup_called);
    }

    #[test]
    fn release_mamba2_input_or_cleanup_output_cleans_output_when_input_missing() {
        let mut output_cleanup_called = false;
        let result = super::release_mamba2_input_or_cleanup_output(
            || Ok(false),
            || {
                output_cleanup_called = true;
                Ok(false)
            },
        );

        let err = result.unwrap_err();
        assert!(output_cleanup_called);
        assert!(err.contains("already missing"));
        assert!(err.contains("output cleanup missing"));
    }

    #[test]
    fn release_mamba2_input_or_cleanup_output_cleans_output_when_input_errors() {
        let mut output_cleanup_called = false;
        let result = super::release_mamba2_input_or_cleanup_output(
            || Err("input boom".to_string()),
            || {
                output_cleanup_called = true;
                Ok(true)
            },
        );

        let err = result.unwrap_err();
        assert!(output_cleanup_called);
        assert!(err.contains("input boom"));
    }

    #[test]
    fn release_mamba2_input_or_cleanup_output_reports_output_cleanup_error() {
        let result = super::release_mamba2_input_or_cleanup_output(
            || Ok(false),
            || Err("output boom".to_string()),
        );

        let err = result.unwrap_err();
        assert!(err.contains("output cleanup failed: output boom"));
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn gdn_prefill_quantized_projection(
    weight: &QuantizedWeight,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    let Some(view) = weight.backend_view() else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::gdn_prefill_quantized_projection(
            backend_ggml_type(view.quant()),
            view.raw(),
            view.rows(),
            view.cols(),
            input,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn gdn_prefill_quantized_projection_q(
    weight: &QuantizedWeight,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    let Some(view) = weight.backend_view() else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::gdn_prefill_quantized_projection_q(
            backend_ggml_type(view.quant()),
            view.raw(),
            view.rows(),
            view.cols(),
            input,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prefill_attention_hd256_if_supported(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::prefill_attention_hd256_if_supported(
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
            softcap,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
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

/// pm48 ②: Metal prefill attention 2차 device-resident chain seam(rope/qk_norm→cast→flash 단일
/// command buffer). 입력(host): q_proj(gate split 후, norm 전), k_proj(norm 전), v(f32),
/// q_norm/k_norm weight. 반환 `(attn_out, k_f16, v_f16)`. Metal 전용(CUDA 미지원 → None).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_attn_chain_if_supported(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    eps: f32,
    pos_start: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
    layer_idx: usize,
    timing_enabled: bool,
) -> Option<(Vec<f32>, Vec<u16>, Vec<u16>)> {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_attn_chain_if_supported(
            q_in,
            k_in,
            v_in,
            q_norm_w,
            k_norm_w,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            n_rot,
            theta,
            eps,
            pos_start,
            scale,
            has_sliding_window,
            has_softcap,
            layer_idx,
            timing_enabled,
        );
    }
    #[cfg(not(feature = "metal"))]
    None
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_timing_enabled() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_full_timing_enabled();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_expected_dense_layer() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_expected_dense_layer();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_core_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_core_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_full_layer_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_full_layer_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_skip() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_skip();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_adapter_reject() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_adapter_reject();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_backend_err() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_backend_err();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_full_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_timing_enabled() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_o_tail_timing_enabled();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_requested() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_o_tail_requested();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_expected_dense_layer() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_expected_dense_layer();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_skip() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_skip();
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_adapter_reject() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_backend_err() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_backend_err();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_legacy_carrier_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().legacy_carrier_enabled();
    }
    #[cfg(not(feature = "metal"))]
    true
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_legacy_attn_layer_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().legacy_attn_layer_enabled;
    }
    #[cfg(not(feature = "metal"))]
    true
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_qwen_moe_decode_chain_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().qwen_moe_decode_chain_enabled;
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_decode_kv_int8_requires_carrier_error(
    carrier_chain_enabled: bool,
    attn_layer_enabled: bool,
) -> Option<&'static str> {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy()
            .kv_int8_requires_carrier_error(carrier_chain_enabled, attn_layer_enabled);
    }
    #[cfg(not(feature = "metal"))]
    None
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_parity_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_parity_record_expected_token() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_record_expected_token();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_decode_parity_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct MetalPrefillAtnCoreShape {
    pub(in crate::engine) seq_len: usize,
    pub(in crate::engine) num_heads: usize,
    pub(in crate::engine) num_kv_heads: usize,
    pub(in crate::engine) head_dim: usize,
    pub(in crate::engine) hidden_dim: usize,
    pub(in crate::engine) q_dim: usize,
    pub(in crate::engine) kv_dim: usize,
    pub(in crate::engine) n_rot: usize,
    pub(in crate::engine) rope_theta: f32,
    pub(in crate::engine) scale: f32,
    pub(in crate::engine) norm_eps: f32,
    pub(in crate::engine) pos_start: usize,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnCoreAdapterOut {
    pub(in crate::engine) attn_out: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnFullLayerAdapterOut {
    pub(in crate::engine) hidden: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnOTailAdapterOut {
    pub(in crate::engine) hidden: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_core_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnCoreAdapterOut>> {
    let (Some(q_view), Some(k_view), Some(v_view)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        metal_prefill_atn_full_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_core_if_supported(
            metal_runtime::MetalPrefillAtnCoreRequest {
                hidden,
                attn_norm_w,
                q_norm_w,
                k_norm_w,
                q_weight_ggml: backend_ggml_type(q_view.quant()),
                q_weight_raw: q_view.raw(),
                q_weight_rows: q_view.rows(),
                q_weight_cols: q_view.cols(),
                k_weight_ggml: backend_ggml_type(k_view.quant()),
                k_weight_raw: k_view.raw(),
                k_weight_rows: k_view.rows(),
                k_weight_cols: k_view.cols(),
                v_weight_ggml: backend_ggml_type(v_view.quant()),
                v_weight_raw: v_view.raw(),
                v_weight_rows: v_view.rows(),
                v_weight_cols: v_view.cols(),
                seq_len: shape.seq_len,
                num_heads: shape.num_heads,
                num_kv_heads: shape.num_kv_heads,
                head_dim: shape.head_dim,
                hidden_dim: shape.hidden_dim,
                q_dim: shape.q_dim,
                kv_dim: shape.kv_dim,
                n_rot: shape.n_rot,
                rope_theta: shape.rope_theta,
                scale: shape.scale,
                norm_eps: shape.norm_eps,
                pos_start: shape.pos_start,
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnCoreAdapterOut {
            attn_out: out.attn_out,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            shape,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnOTailAdapterOut>> {
    let (Some(q_view), Some(k_view), Some(v_view), Some(o_view)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
    ) else {
        metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_o_tail_if_supported(
            metal_runtime::MetalPrefillAtnOTailRequest {
                core: metal_runtime::MetalPrefillAtnCoreRequest {
                    hidden,
                    attn_norm_w,
                    q_norm_w,
                    k_norm_w,
                    q_weight_ggml: backend_ggml_type(q_view.quant()),
                    q_weight_raw: q_view.raw(),
                    q_weight_rows: q_view.rows(),
                    q_weight_cols: q_view.cols(),
                    k_weight_ggml: backend_ggml_type(k_view.quant()),
                    k_weight_raw: k_view.raw(),
                    k_weight_rows: k_view.rows(),
                    k_weight_cols: k_view.cols(),
                    v_weight_ggml: backend_ggml_type(v_view.quant()),
                    v_weight_raw: v_view.raw(),
                    v_weight_rows: v_view.rows(),
                    v_weight_cols: v_view.cols(),
                    seq_len: shape.seq_len,
                    num_heads: shape.num_heads,
                    num_kv_heads: shape.num_kv_heads,
                    head_dim: shape.head_dim,
                    hidden_dim: shape.hidden_dim,
                    q_dim: shape.q_dim,
                    kv_dim: shape.kv_dim,
                    n_rot: shape.n_rot,
                    rope_theta: shape.rope_theta,
                    scale: shape.scale,
                    norm_eps: shape.norm_eps,
                    pos_start: shape.pos_start,
                },
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnOTailAdapterOut {
            hidden: out.hidden,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            o_view,
            shape,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_full_layer_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    ffn_norm_w: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnFullLayerAdapterOut>> {
    let (
        Some(q_view),
        Some(k_view),
        Some(v_view),
        Some(o_view),
        Some(ffn_gate_view),
        Some(ffn_up_view),
        Some(ffn_down_view),
    ) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    )
    else {
        metal_prefill_atn_full_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_full_layer_if_supported(
            metal_runtime::MetalPrefillAtnFullLayerRequest {
                core: metal_runtime::MetalPrefillAtnCoreRequest {
                    hidden,
                    attn_norm_w,
                    q_norm_w,
                    k_norm_w,
                    q_weight_ggml: backend_ggml_type(q_view.quant()),
                    q_weight_raw: q_view.raw(),
                    q_weight_rows: q_view.rows(),
                    q_weight_cols: q_view.cols(),
                    k_weight_ggml: backend_ggml_type(k_view.quant()),
                    k_weight_raw: k_view.raw(),
                    k_weight_rows: k_view.rows(),
                    k_weight_cols: k_view.cols(),
                    v_weight_ggml: backend_ggml_type(v_view.quant()),
                    v_weight_raw: v_view.raw(),
                    v_weight_rows: v_view.rows(),
                    v_weight_cols: v_view.cols(),
                    seq_len: shape.seq_len,
                    num_heads: shape.num_heads,
                    num_kv_heads: shape.num_kv_heads,
                    head_dim: shape.head_dim,
                    hidden_dim: shape.hidden_dim,
                    q_dim: shape.q_dim,
                    kv_dim: shape.kv_dim,
                    n_rot: shape.n_rot,
                    rope_theta: shape.rope_theta,
                    scale: shape.scale,
                    norm_eps: shape.norm_eps,
                    pos_start: shape.pos_start,
                },
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
                ffn_norm_w,
                ffn_gate_weight_ggml: backend_ggml_type(ffn_gate_view.quant()),
                ffn_gate_weight_raw: ffn_gate_view.raw(),
                ffn_gate_weight_rows: ffn_gate_view.rows(),
                ffn_gate_weight_cols: ffn_gate_view.cols(),
                ffn_up_weight_ggml: backend_ggml_type(ffn_up_view.quant()),
                ffn_up_weight_raw: ffn_up_view.raw(),
                ffn_up_weight_rows: ffn_up_view.rows(),
                ffn_up_weight_cols: ffn_up_view.cols(),
                ffn_down_weight_ggml: backend_ggml_type(ffn_down_view.quant()),
                ffn_down_weight_raw: ffn_down_view.raw(),
                ffn_down_weight_rows: ffn_down_view.rows(),
                ffn_down_weight_cols: ffn_down_view.cols(),
                ffn_dim: ffn_gate_view.rows(),
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnFullLayerAdapterOut {
            hidden: out.hidden,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            o_view,
            ffn_norm_w,
            ffn_gate_view,
            ffn_up_view,
            ffn_down_view,
            shape,
        );
        Ok(None)
    }
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

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attn_decode_into_if_supported(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> crate::error::Result<bool> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_decode_into_if_supported(
            q,
            k_cache,
            v_cache,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            sliding_window,
            has_softcap,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            q,
            k_cache,
            v_cache,
            &attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            sliding_window,
            has_softcap,
        );
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    o_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    prior_k: &[u16],
    prior_v: &[u16],
    pos: usize,
    hidden_dim: usize,
    q_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    n_rot: usize,
    capacity: usize,
    ffn_dim: usize,
    eps: f32,
    theta: f32,
    scale: f32,
) -> crate::error::Result<bool> {
    let (Some(q_v), Some(k_v), Some(v_v), Some(o_v), Some(fg_v), Some(fu_v), Some(fd_v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_layer_into_if_supported(
            layer,
            hidden,
            norm_weight,
            backend_ggml_type(q_v.quant()),
            backend_ggml_type(k_v.quant()),
            backend_ggml_type(v_v.quant()),
            backend_ggml_type(o_v.quant()),
            q_v.raw(),
            k_v.raw(),
            v_v.raw(),
            q_norm_weight,
            k_norm_weight,
            o_v.raw(),
            ffn_norm_weight,
            backend_ggml_type(fg_v.quant()),
            fg_v.raw(),
            backend_ggml_type(fu_v.quant()),
            fu_v.raw(),
            backend_ggml_type(fd_v.quant()),
            fd_v.raw(),
            prior_k,
            prior_v,
            pos,
            hidden_dim,
            q_dim,
            q_out_dim,
            kv_dim,
            head_dim,
            num_heads,
            num_kv_heads,
            n_rot,
            capacity,
            ffn_dim,
            eps,
            theta,
            scale,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (q_v, k_v, v_v, o_v, fg_v, fu_v, fd_v);
        Ok(false)
    }
}

#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_attn_carrier_kv_filled(layer: usize) -> Option<usize> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_carrier_kv_filled(layer);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        None
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    ffn_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (
        Some(qkv_v),
        Some(gate_v),
        Some(alpha_v),
        Some(beta_v),
        Some(ssm_out_v),
        Some(fg_v),
        Some(fu_v),
        Some(fd_v),
    ) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    )
    else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_layer_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            ffn_norm_weight,
            backend_ggml_type(fg_v.quant()),
            fg_v.raw(),
            backend_ggml_type(fu_v.quant()),
            fu_v.raw(),
            backend_ggml_type(fd_v.quant()),
            fd_v.raw(),
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            ffn_dim,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (qkv_v, gate_v, alpha_v, beta_v, ssm_out_v, fg_v, fu_v, fd_v);
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_core_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_core_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (qkv_v, gate_v, alpha_v, beta_v, ssm_out_v);
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_moe_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    moe_w: &crate::engine::layer_weights::SharedExpertMoELayerWeights,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    let Some(router_w) = moe_w.router_f32() else {
        return Ok(false);
    };
    let (Some(gate_exps), Some(up_exps), Some(down_exps)) = (
        moe_w.gate_exps_bytes(),
        moe_w.up_exps_bytes(),
        moe_w.down_exps_bytes(),
    ) else {
        return Ok(false);
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe_w.shared_gate.data.as_bytes(),
        moe_w.shared_up.data.as_bytes(),
        moe_w.shared_down.data.as_bytes(),
    ) else {
        return Ok(false);
    };
    let shared_input_scale = crate::engine::kernels::tensor_as_f32_slice(&moe_w.shared_input_scale);
    let expert_bytes = crate::engine::models::shared_expert_moe::moe_types::sparse_expert_bytes(
        moe_w.n_embd,
        moe_w.n_ff,
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
    )
    .expect("qwen35 sparse expert bytes");
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_moe_layer_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            ffn_norm_weight,
            router_w,
            moe_w.gate_quant,
            gate_exps,
            expert_bytes.gate,
            moe_w.up_quant,
            up_exps,
            expert_bytes.up,
            moe_w.down_quant,
            down_exps,
            expert_bytes.down,
            shared_input_scale,
            moe_w.shared_gate.ggml_type,
            shared_gate,
            moe_w.shared_up.ggml_type,
            shared_up,
            moe_w.shared_down.ggml_type,
            shared_down,
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            moe_w.n_ff,
            moe_w.n_expert,
            moe_w.n_expert_used,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            qkv_v,
            gate_v,
            alpha_v,
            beta_v,
            ssm_out_v,
            router_w,
            gate_exps,
            up_exps,
            down_exps,
            shared_gate,
            shared_up,
            shared_down,
            shared_input_scale,
            expert_bytes,
        );
        Ok(false)
    }
}

/// chain run 1개 layer 의 입력 state(borrowed). backend 호출이 끝난 뒤 이 borrow 를
/// 명시적으로 해제하고 GDN state 를 kv_cache 에 반영한다.
/// - Attn: 첫 token KV init 용 prior KV(host f16). filled!=0 이면 빈 slice.
/// - Gdn: 직전 token conv/delta state.
#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) enum ChainLayerInput<'a> {
    Attn {
        prior_k: &'a [u16],
        prior_v: &'a [u16],
        kvarn: Option<crate::runtime::KvarnChainView<'a>>,
    },
    Gdn {
        conv_state: &'a [f32],
        delta_state: &'a [f32],
    },
}

/// chain run 1개 attention layer 의 shape(decode loop 이 `resolve_attention_layout` +
/// `resolve_rope_params` 로 1회 계산해 넘긴다). carrier entry 생성에 그대로 사용.
#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) struct ChainAttnShape {
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub pos: usize,
    pub theta: f32,
    pub scale: f32,
}

#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) struct MetalDecodeOutputArgmax<'a> {
    pub norm_weight: &'a [f32],
    pub output_weight: &'a QuantizedWeight,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
struct QwenMoeChainMaterials<'a> {
    router_w: &'a [f32],
    gate_exps: &'a [u8],
    gate_expert_bytes: usize,
    up_exps: &'a [u8],
    up_expert_bytes: usize,
    down_exps: &'a [u8],
    down_expert_bytes: usize,
    shared_input_scale: &'a [f32],
    shared_gate: &'a [u8],
    shared_up: &'a [u8],
    shared_down: &'a [u8],
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn qwen_moe_chain_materials(
    moe_w: &crate::engine::layer_weights::SharedExpertMoELayerWeights,
) -> std::result::Result<QwenMoeChainMaterials<'_>, &'static str> {
    let shared_q8_0 = moe_w.shared_gate.ggml_type == GGMLType::Q8_0
        && moe_w.shared_up.ggml_type == GGMLType::Q8_0
        && moe_w.shared_down.ggml_type == GGMLType::Q8_0;
    let shared_q4 = moe_w.shared_gate.ggml_type == GGMLType::Q4_K
        && moe_w.shared_up.ggml_type == GGMLType::Q4_K
        && moe_w.shared_down.ggml_type == moe_w.down_quant;
    if moe_w.gate_quant != GGMLType::Q4_K
        || moe_w.up_quant != GGMLType::Q4_K
        || !matches!(
            moe_w.down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        || !(shared_q8_0 || shared_q4)
    {
        return Err("unsupported qwen moe quant");
    }
    let Some(router_w) = moe_w.router_f32() else {
        return Err("missing qwen moe router");
    };
    let (Some(gate_exps), Some(up_exps), Some(down_exps)) = (
        moe_w.gate_exps_bytes(),
        moe_w.up_exps_bytes(),
        moe_w.down_exps_bytes(),
    ) else {
        return Err("missing qwen moe expert raw");
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe_w.shared_gate.data.as_bytes(),
        moe_w.shared_up.data.as_bytes(),
        moe_w.shared_down.data.as_bytes(),
    ) else {
        return Err("missing qwen moe shared raw");
    };
    let Some(expert_bytes) =
        crate::engine::models::shared_expert_moe::moe_types::sparse_expert_bytes(
            moe_w.n_embd,
            moe_w.n_ff,
            moe_w.gate_quant,
            moe_w.up_quant,
            moe_w.down_quant,
        )
    else {
        return Err("unsupported qwen moe expert shape");
    };
    Ok(QwenMoeChainMaterials {
        router_w,
        gate_exps,
        gate_expert_bytes: expert_bytes.gate,
        up_exps,
        up_expert_bytes: expert_bytes.up,
        down_exps,
        down_expert_bytes: expert_bytes.down,
        shared_input_scale: crate::engine::kernels::tensor_as_f32_slice(&moe_w.shared_input_scale),
        shared_gate,
        shared_up,
        shared_down,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::engine) struct MetalDecodeChainRunReport {
    pub did_run: bool,
    pub qwen_moe_layers: usize,
    pub fallback_reason: Option<&'static str>,
    pub output_argmax_token: Option<u32>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_decode_chain_runtime_result(
    result: std::result::Result<metal_runtime::MetalDecodeChainReport, String>,
) -> crate::error::Result<MetalDecodeChainRunReport> {
    result
        .map(|report| MetalDecodeChainRunReport {
            did_run: report.did_run,
            qwen_moe_layers: report.qwen_moe_layers,
            fallback_reason: report.fallback_reason,
            output_argmax_token: report.output_argmax.token_id,
        })
        .map_err(crate::error::LlmError::Forward)
}

/// 연속된 attn carrier + GDN carrier layer N개를 단일 command buffer 로 실행하는 facade(2단계).
/// 1단계 GDN-only(`metal_gdn_decode_chain_run`)를 일반화 — `layers` 가 `&LayerType`(attn/gdn
/// 혼합)이라 9B 의 32 layer 를 전부 하나의 command buffer 로 묶는다. 각 layer 에서 quant raw
/// view + f32 weight 를 추출해 seam `MetalChainLayer` enum 으로 변환 → `metal_decode_chain_run`.
///
/// borrow 안전: `inputs`(attn prior KV / gdn conv·delta)는 caller 가 owned clone 으로 넘겨
/// kv_cache 가변 borrow 와 충돌 안 함. attn 은 KV device 소유라 out_states 가 `None`,
/// gdn 만 `Some((conv_new, delta_new))` → caller 가 ssm_state 에 write.
/// GDN shape 는 모든 GDN layer 동일(caller 가 metadata 에서 1회 계산), attn shape 는
/// `attn_shapes`(layer 별, 9B 는 동일하지만 self-contained).
///
/// 모든 layer 가 backend_view + quant code 지원이어야 한다. 하나라도 미충족이면 Ok(false)
/// → caller 가 per-layer 경로로 fallback.
/// milestone 5(MTP): 단일/배치 공용 구현. `out_attn_kv=None` → 단일(batch=1) `decode_chain_run`,
/// `Some` → `decode_chain_run_batched_collect_attn_kv`(B lane; attn window K/V 를 host commit 용
/// 으로 반환). lane 별 report(Vec) 반환 — 단일 wrapper 는 [0] 만 추출한다.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
fn metal_decode_chain_run_impl(
    hidden: &mut [f32],
    batch: usize,
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    out_attn_kv: Option<&mut Vec<Option<(Vec<u16>, Vec<u16>)>>>,
    out_gdn_prefix: Option<&mut Vec<Vec<Option<(Vec<f32>, Vec<f32>)>>>>,
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use crate::engine::kernels;
        let fallback = |reason: &'static str| {
            vec![
                MetalDecodeChainRunReport {
                    fallback_reason: Some(reason),
                    ..MetalDecodeChainRunReport::default()
                };
                batch
            ]
        };
        assert_eq!(layers.len(), inputs.len());
        assert_eq!(layers.len(), attn_shapes.len());
        assert_eq!(layers.len(), out_states.len());
        if layers.is_empty() {
            return Ok(vec![
                MetalDecodeChainRunReport {
                    did_run: true,
                    ..MetalDecodeChainRunReport::default()
                };
                batch
            ]);
        }
        // 각 layer 의 quant backend_view 를 먼저 모은다(raw borrow 수명을 specs 와 묶기 위해
        // 별도 Vec 에 보관). 하나라도 view 가 없거나 layer 종류 미지원이면 chain 포기.
        // Dense Attn은 q/k/v/o + 3개 FFN, Attention+MoE는 q/k/v/o만, Gdn은
        // qkv/gate/alpha/beta/ssm_out + dense일 때만 3개 FFN view를 보관한다.
        let mut views: Vec<Vec<_>> = Vec::with_capacity(layers.len());
        for (_, lt) in layers {
            match lt {
                LayerType::Attention(w) => {
                    let (Some(q_v), Some(k_v), Some(v_v), Some(o_v)) = (
                        w.q_weight.backend_view(),
                        w.k_weight.backend_view(),
                        w.v_weight.backend_view(),
                        w.o_weight.backend_view(),
                    ) else {
                        return Ok(fallback("missing attention backend view"));
                    };
                    if w.shared_expert_moe.is_some() {
                        views.push(vec![q_v, k_v, v_v, o_v]);
                    } else {
                        let (Some(fg_v), Some(fu_v), Some(fd_v)) = (
                            w.ffn_gate_weight.backend_view(),
                            w.ffn_up_weight.backend_view(),
                            w.ffn_down_weight.backend_view(),
                        ) else {
                            return Ok(fallback("missing attention ffn backend view"));
                        };
                        views.push(vec![q_v, k_v, v_v, o_v, fg_v, fu_v, fd_v]);
                    }
                }
                LayerType::GatedDeltaNet(w) => {
                    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
                        w.qkv_weight.backend_view(),
                        w.gate_weight.backend_view(),
                        w.ssm_alpha.backend_view(),
                        w.ssm_beta.backend_view(),
                        w.ssm_out.backend_view(),
                    ) else {
                        return Ok(fallback("missing gdn backend view"));
                    };
                    if w.shared_expert_moe.is_some() {
                        views.push(vec![qkv_v, gate_v, alpha_v, beta_v, ssm_out_v]);
                    } else {
                        let (Some(fg_v), Some(fu_v), Some(fd_v)) = (
                            w.ffn_gate_weight.backend_view(),
                            w.ffn_up_weight.backend_view(),
                            w.ffn_down_weight.backend_view(),
                        ) else {
                            return Ok(fallback("missing dense gdn ffn backend view"));
                        };
                        views.push(vec![
                            qkv_v, gate_v, alpha_v, beta_v, ssm_out_v, fg_v, fu_v, fd_v,
                        ]);
                    }
                }
                _ => return Ok(fallback("unsupported chain layer")),
            }
        }
        // views 가 살아있는 동안 raw()/quant() borrow 로 seam spec 구성.
        let mut specs = Vec::with_capacity(layers.len());
        for (i, (layer_idx, lt)) in layers.iter().enumerate() {
            let v = &views[i];
            match lt {
                LayerType::Attention(w) => {
                    let (
                        ChainLayerInput::Attn {
                            prior_k,
                            prior_v,
                            kvarn,
                        },
                        Some(shape),
                    ) = (&inputs[i], &attn_shapes[i])
                    else {
                        return Ok(fallback("attention chain input mismatch"));
                    };
                    let (Some(q_norm), Some(k_norm)) = (w.q_norm.as_ref(), w.k_norm.as_ref())
                    else {
                        return Ok(fallback("missing attention qk norm"));
                    };
                    let q_ggml = backend_ggml_type(v[0].quant());
                    let k_ggml = backend_ggml_type(v[1].quant());
                    let v_ggml = backend_ggml_type(v[2].quant());
                    let o_ggml = backend_ggml_type(v[3].quant());
                    if let Some(moe_w) = w.shared_expert_moe.as_ref() {
                        // q/k/v/o decode GEMV 지원 집합. 실제 code 변환/검증은 runtime
                        // attn_moe_qwen_chain_ref_if_supported 가 수행(단일 소스). 여기선
                        // 미지원 quant 를 조기 fallback 시켜 이유를 명확히 남긴다.
                        let attn_quant_ok = |t: GGMLType| {
                            matches!(
                                t,
                                GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_0
                            )
                        };
                        if !attn_quant_ok(q_ggml)
                            || !attn_quant_ok(k_ggml)
                            || !attn_quant_ok(v_ggml)
                            || !attn_quant_ok(o_ggml)
                        {
                            return Ok(fallback("unsupported qwen attention quant"));
                        }
                        let moe = match qwen_moe_chain_materials(moe_w) {
                            Ok(materials) => materials,
                            Err(reason) => return Ok(fallback(reason)),
                        };
                        specs.push(metal_runtime::MetalChainLayer::AttnMoeQwen(
                            metal_runtime::MetalAttnMoeQwenChainLayer {
                                layer: *layer_idx,
                                norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                q_norm_weight: kernels::tensor_as_f32_slice(q_norm),
                                k_norm_weight: kernels::tensor_as_f32_slice(k_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.ffn_norm),
                                q_ggml,
                                q_raw: v[0].raw(),
                                k_ggml,
                                k_raw: v[1].raw(),
                                v_ggml,
                                v_raw: v[2].raw(),
                                o_ggml,
                                o_raw: v[3].raw(),
                                router_w: moe.router_w,
                                gate_exps_ggml: moe_w.gate_quant,
                                gate_exps_raw: moe.gate_exps,
                                gate_expert_bytes: moe.gate_expert_bytes,
                                up_exps_ggml: moe_w.up_quant,
                                up_exps_raw: moe.up_exps,
                                up_expert_bytes: moe.up_expert_bytes,
                                down_exps_ggml: moe_w.down_quant,
                                down_exps_raw: moe.down_exps,
                                down_expert_bytes: moe.down_expert_bytes,
                                shared_input_scale: moe.shared_input_scale,
                                shared_gate_ggml: moe_w.shared_gate.ggml_type,
                                shared_gate_raw: moe.shared_gate,
                                shared_up_ggml: moe_w.shared_up.ggml_type,
                                shared_up_raw: moe.shared_up,
                                shared_down_ggml: moe_w.shared_down.ggml_type,
                                shared_down_raw: moe.shared_down,
                                prior_k,
                                prior_v,
                                kvarn: *kvarn,
                                pos: shape.pos,
                                hidden_dim,
                                q_dim: shape.q_dim,
                                q_out_dim: shape.q_out_dim,
                                kv_dim: shape.kv_dim,
                                head_dim: shape.head_dim,
                                num_heads: shape.num_heads,
                                num_kv_heads: shape.num_kv_heads,
                                n_rot: shape.n_rot,
                                capacity,
                                n_ff: moe_w.n_ff,
                                n_expert: moe_w.n_expert,
                                n_expert_used: moe_w.n_expert_used,
                                eps,
                                theta: shape.theta,
                                scale: shape.scale,
                            },
                        ));
                    } else {
                        specs.push(metal_runtime::MetalChainLayer::Attn(
                            metal_runtime::MetalAttnChainLayer {
                                layer: *layer_idx,
                                norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                q_norm_weight: kernels::tensor_as_f32_slice(q_norm),
                                k_norm_weight: kernels::tensor_as_f32_slice(k_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.ffn_norm),
                                q_ggml,
                                q_raw: v[0].raw(),
                                k_ggml,
                                k_raw: v[1].raw(),
                                v_ggml,
                                v_raw: v[2].raw(),
                                o_ggml,
                                o_raw: v[3].raw(),
                                ffn_gate_ggml: backend_ggml_type(v[4].quant()),
                                ffn_gate_raw: v[4].raw(),
                                ffn_up_ggml: backend_ggml_type(v[5].quant()),
                                ffn_up_raw: v[5].raw(),
                                ffn_down_ggml: backend_ggml_type(v[6].quant()),
                                ffn_down_raw: v[6].raw(),
                                prior_k,
                                prior_v,
                                pos: shape.pos,
                                hidden_dim,
                                q_dim: shape.q_dim,
                                q_out_dim: shape.q_out_dim,
                                kv_dim: shape.kv_dim,
                                head_dim: shape.head_dim,
                                num_heads: shape.num_heads,
                                num_kv_heads: shape.num_kv_heads,
                                n_rot: shape.n_rot,
                                capacity,
                                ffn_dim: w.ffn_gate_weight.rows,
                                eps,
                                theta: shape.theta,
                                scale: shape.scale,
                            },
                        ));
                    }
                }
                LayerType::GatedDeltaNet(w) => {
                    let ChainLayerInput::Gdn {
                        conv_state,
                        delta_state,
                    } = &inputs[i]
                    else {
                        return Ok(fallback("gdn chain input mismatch"));
                    };
                    if let Some(moe_w) = w.shared_expert_moe.as_ref() {
                        let moe = match qwen_moe_chain_materials(moe_w) {
                            Ok(materials) => materials,
                            Err(reason) => return Ok(fallback(reason)),
                        };
                        specs.push(metal_runtime::MetalChainLayer::GdnMoeQwen(
                            metal_runtime::MetalGdnMoeQwenChainLayer {
                                layer: *layer_idx,
                                conv_state,
                                delta_state,
                                attn_norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                dt_bias_weight: kernels::tensor_as_f32_slice(&w.ssm_dt_bias),
                                ssm_a_weight: kernels::tensor_as_f32_slice(&w.ssm_a),
                                conv1d_weight: kernels::tensor_as_f32_slice(&w.ssm_conv1d),
                                ssm_norm_weight: kernels::tensor_as_f32_slice(&w.ssm_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.post_attn_norm),
                                qkv_ggml: backend_ggml_type(v[0].quant()),
                                qkv_raw: v[0].raw(),
                                gate_ggml: backend_ggml_type(v[1].quant()),
                                gate_raw: v[1].raw(),
                                alpha_ggml: backend_ggml_type(v[2].quant()),
                                alpha_raw: v[2].raw(),
                                beta_ggml: backend_ggml_type(v[3].quant()),
                                beta_raw: v[3].raw(),
                                ssm_out_ggml: backend_ggml_type(v[4].quant()),
                                ssm_out_raw: v[4].raw(),
                                router_w: moe.router_w,
                                gate_exps_ggml: moe_w.gate_quant,
                                gate_exps_raw: moe.gate_exps,
                                gate_expert_bytes: moe.gate_expert_bytes,
                                up_exps_ggml: moe_w.up_quant,
                                up_exps_raw: moe.up_exps,
                                up_expert_bytes: moe.up_expert_bytes,
                                down_exps_ggml: moe_w.down_quant,
                                down_exps_raw: moe.down_exps,
                                down_expert_bytes: moe.down_expert_bytes,
                                shared_input_scale: moe.shared_input_scale,
                                shared_gate_ggml: moe_w.shared_gate.ggml_type,
                                shared_gate_raw: moe.shared_gate,
                                shared_up_ggml: moe_w.shared_up.ggml_type,
                                shared_up_raw: moe.shared_up,
                                shared_down_ggml: moe_w.shared_down.ggml_type,
                                shared_down_raw: moe.shared_down,
                                hidden_dim,
                                conv_channels,
                                conv_kernel,
                                z_dim,
                                num_v_heads,
                                num_k_heads,
                                head_k_dim,
                                head_v_dim,
                                n_ff: moe_w.n_ff,
                                n_expert: moe_w.n_expert,
                                n_expert_used: moe_w.n_expert_used,
                                eps,
                            },
                        ));
                    } else {
                        specs.push(metal_runtime::MetalChainLayer::Gdn(
                            metal_runtime::MetalGdnChainLayer {
                                layer: *layer_idx,
                                conv_state,
                                delta_state,
                                attn_norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                dt_bias_weight: kernels::tensor_as_f32_slice(&w.ssm_dt_bias),
                                ssm_a_weight: kernels::tensor_as_f32_slice(&w.ssm_a),
                                conv1d_weight: kernels::tensor_as_f32_slice(&w.ssm_conv1d),
                                ssm_norm_weight: kernels::tensor_as_f32_slice(&w.ssm_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.post_attn_norm),
                                qkv_ggml: backend_ggml_type(v[0].quant()),
                                qkv_raw: v[0].raw(),
                                gate_ggml: backend_ggml_type(v[1].quant()),
                                gate_raw: v[1].raw(),
                                alpha_ggml: backend_ggml_type(v[2].quant()),
                                alpha_raw: v[2].raw(),
                                beta_ggml: backend_ggml_type(v[3].quant()),
                                beta_raw: v[3].raw(),
                                ssm_out_ggml: backend_ggml_type(v[4].quant()),
                                ssm_out_raw: v[4].raw(),
                                ffn_gate_ggml: backend_ggml_type(v[5].quant()),
                                ffn_gate_raw: v[5].raw(),
                                ffn_up_ggml: backend_ggml_type(v[6].quant()),
                                ffn_up_raw: v[6].raw(),
                                ffn_down_ggml: backend_ggml_type(v[7].quant()),
                                ffn_down_raw: v[7].raw(),
                                hidden_dim,
                                conv_channels,
                                conv_kernel,
                                z_dim,
                                num_v_heads,
                                num_k_heads,
                                head_k_dim,
                                head_v_dim,
                                ffn_dim: w.ffn_gate_weight.rows,
                                eps,
                            },
                        ));
                    }
                }
                _ => return Ok(fallback("unsupported chain layer")),
            }
        }
        let output_view = output_argmax
            .as_ref()
            .and_then(|tail| tail.output_weight.backend_view());
        let output_tail = match (output_argmax.as_ref(), output_view.as_ref()) {
            (Some(tail), Some(view)) => Some(metal_runtime::MetalDecodeOutputArgmax {
                norm_weight: tail.norm_weight,
                output_ggml: backend_ggml_type(view.quant()),
                output_raw: view.raw(),
                rows: tail.rows,
                cols: tail.cols,
                eps: tail.eps,
            }),
            (Some(_), None) => {
                return Ok(fallback("missing output backend view"));
            }
            (None, _) => None,
        };
        return match out_attn_kv {
            Some(out_kv) => {
                let out_prefix =
                    out_gdn_prefix.expect("batched decode chain requires out_gdn_prefix buffer");
                metal_decode_chain_runtime_result_batched(
                    metal_runtime::metal_decode_chain_run_batched_collect_attn_kv(
                        hidden,
                        batch,
                        &specs,
                        out_states,
                        out_kv,
                        out_prefix,
                        output_tail,
                    ),
                )
            }
            None => metal_decode_chain_runtime_result(metal_runtime::metal_decode_chain_run(
                hidden,
                &specs,
                out_states,
                output_tail,
            ))
            .map(|report| vec![report]),
        };
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            batch,
            layers,
            inputs,
            attn_shapes,
            out_states,
            out_attn_kv,
            out_gdn_prefix,
            capacity,
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            eps,
            output_argmax,
        );
        Ok(vec![MetalDecodeChainRunReport::default(); batch])
    }
}

/// 단일-토큰 decode chain facade(기존 시그니처 유지). `metal_decode_chain_run_impl` 에 batch=1,
/// out_attn_kv=None 으로 위임하고 lane 0 report 를 반환한다.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_chain_run(
    hidden: &mut [f32],
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<MetalDecodeChainRunReport> {
    metal_decode_chain_run_impl(
        hidden,
        1,
        layers,
        inputs,
        attn_shapes,
        out_states,
        None,
        None,
        capacity,
        hidden_dim,
        conv_channels,
        conv_kernel,
        z_dim,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        eps,
        output_argmax,
    )
    .map(|reports| reports.into_iter().next().unwrap_or_default())
}

/// milestone 5(MTP): 배치(B-lane) decode chain facade. 단일과 동일한 spec 빌드를 쓰되 `hidden`
/// 은 `[batch*hidden_dim]`, `out_attn_kv` 에 attn layer 별 window post-rope f16 K/V 를 채워
/// 반환한다(엔진이 accept-n 커밋에서 host kv_cache 에 append). 반환: lane 별 report.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_chain_run_batched(
    hidden: &mut [f32],
    batch: usize,
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    out_attn_kv: &mut Vec<Option<(Vec<u16>, Vec<u16>)>>,
    out_gdn_prefix: &mut Vec<Vec<Option<(Vec<f32>, Vec<f32>)>>>,
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    metal_decode_chain_run_impl(
        hidden,
        batch,
        layers,
        inputs,
        attn_shapes,
        out_states,
        Some(out_attn_kv),
        Some(out_gdn_prefix),
        capacity,
        hidden_dim,
        conv_channels,
        conv_kernel,
        z_dim,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        eps,
        output_argmax,
    )
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_decode_chain_runtime_result_batched(
    result: std::result::Result<Vec<metal_runtime::MetalDecodeChainReport>, String>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    result
        .map(|reports| {
            reports
                .into_iter()
                .map(|report| MetalDecodeChainRunReport {
                    did_run: report.did_run,
                    qwen_moe_layers: report.qwen_moe_layers,
                    fallback_reason: report.fallback_reason,
                    output_argmax_token: report.output_argmax.token_id,
                })
                .collect()
        })
        .map_err(crate::error::LlmError::Forward)
}

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
        )
        .map_err(|e| crate::error::LlmError::Forward(e));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (q_v, k_v, v_v);
        Ok(false)
    }
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

// cu49 step 38: K/V projection 의 device output variant. weight (Q4K/Q6K) +
// device input (norm_buf carrier) → device output (K/V carrier). D2H 없음.
// caller 가 nullable host_output 받음 — Some 시 D2H + sync 도 함 (stepping
// stone, host kv_cache.append 호환). None 시 device-only (cu50 step 에서
// host kv_cache append 도 skip 가능 시).
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn dense_q4k_gelu_ffn_if_supported(
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    let (Some(gate), Some(up), Some(down)) = (
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if gate.quant() != QuantFormat::Q4K || up.quant() != QuantFormat::Q4K {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::dense_q4k_gelu_ffn(
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            gate.rows(),
            gate.cols(),
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn dense_q4k_gelu_ffn_batch_if_supported(
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    if seq_len <= 1 {
        return Ok(None);
    }
    let (Some(gate), Some(up), Some(down)) = (
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if gate.quant() != QuantFormat::Q4K || up.quant() != QuantFormat::Q4K {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::dense_q4k_gelu_ffn_batch(
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            gate.rows(),
            gate.cols(),
            seq_len,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn dense_q4k_gelu_ffn_norm_residual_if_supported(
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    norm_weight: &[f32],
    post_norm_weight: Option<&[f32]>,
    hidden: &[f32],
    norm_eps: f32,
    unit_offset_norm: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    let (Some(gate), Some(up), Some(down)) = (
        gate_weight.backend_view(),
        up_weight.backend_view(),
        down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    if gate.quant() != QuantFormat::Q4K || up.quant() != QuantFormat::Q4K {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::dense_q4k_gelu_ffn_norm_residual(
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            norm_weight,
            post_norm_weight,
            gate.rows(),
            gate.cols(),
            hidden,
            norm_eps,
            unit_offset_norm,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn dense_q4k_attention_output_gelu_ffn_norm_residual_if_supported(
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate_weight: Option<&QuantizedWeight>,
    ple_proj_weight: Option<&QuantizedWeight>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_input_device_offset: Option<usize>,
    ple_dim: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_ple_norm: bool,
    hidden_carrier_dev: Option<u64>,
    skip_h2d_hidden: bool,
    skip_d2h_hidden: bool,
    layer_output_scale: Option<f32>,
    attn_out_dev_carrier: Option<u64>,
    ffn_uses_gelu: bool,
    dense_chain_graph_allowed: bool,
    #[cfg(feature = "cuda")] layer_segment_graph_context: Option<
        cuda_runtime::Cu71LayerSegmentGraphRuntimeContext,
    >,
    #[cfg(not(feature = "cuda"))] _layer_segment_graph_context: Option<()>,
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
    let ple_gate = ple_gate_weight.and_then(|weight| weight.backend_view());
    let ple_proj = ple_proj_weight.and_then(|weight| weight.backend_view());
    if ple_gate_weight.is_some() || ple_proj_weight.is_some() {
        let (Some(ple_gate), Some(ple_proj), Some(_), Some(_)) = (
            ple_gate.as_ref(),
            ple_proj.as_ref(),
            ple_post_norm_weight,
            ple_input,
        ) else {
            return Ok(false);
        };
        if !ple_dense_chain_quant_supported(ple_gate.quant(), ple_proj.quant()) {
            return Ok(false);
        }
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::dense_q4k_attention_output_gelu_ffn_norm_residual(
            o.raw(),
            gate.raw(),
            up.raw(),
            down.raw(),
            backend_ggml_type(down.quant()),
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            ple_gate.as_ref().map(|view| view.raw()),
            ple_proj.as_ref().map(|view| view.raw()),
            ple_post_norm_weight,
            ple_input,
            ple_input_device_offset,
            ple_dim,
            o.cols(),
            gate.rows(),
            gate.cols(),
            hidden,
            attn_out,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_ple_norm,
            hidden_carrier_dev,
            skip_h2d_hidden,
            skip_d2h_hidden,
            layer_output_scale,
            attn_out_dev_carrier,
            ffn_uses_gelu,
            dense_chain_graph_allowed,
            layer_segment_graph_context,
        )
        .map(|()| true)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn dense_q4k_attention_output_gelu_ffn_batch_norm_residual_if_supported(
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
    seq_len: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> crate::error::Result<bool> {
    if seq_len <= 1 {
        return Ok(false);
    }
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
        return cuda_runtime::dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
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
            seq_len,
            hidden,
            attn_out,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map(|()| true)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn gemma4_ple_q4k_batch_norm_residual_if_supported(
    gate_weight: &QuantizedWeight,
    proj_weight: &QuantizedWeight,
    post_norm_weight: &[f32],
    out_scale: Option<&[f32]>,
    ple_input: &[f32],
    ple_dim: usize,
    n_embd: usize,
    seq_len: usize,
    hidden: &mut [f32],
    norm_eps: f32,
) -> crate::error::Result<bool> {
    if seq_len <= 1 {
        return Ok(false);
    }
    let (Some(gate), Some(proj)) = (gate_weight.backend_view(), proj_weight.backend_view()) else {
        return Ok(false);
    };
    if gate.quant() != QuantFormat::Q4K
        || proj.quant() != QuantFormat::Q4K
        || gate.rows() != ple_dim
        || gate.cols() != n_embd
        || proj.rows() != n_embd
        || proj.cols() != ple_dim
        || post_norm_weight.len() != n_embd
        || ple_input.len() != seq_len.saturating_mul(ple_dim)
        || hidden.len() != seq_len.saturating_mul(n_embd)
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::gemma4_ple_q4k_batch_norm_residual(
            gate.raw(),
            proj.raw(),
            post_norm_weight,
            out_scale,
            ple_input,
            ple_dim,
            n_embd,
            seq_len,
            hidden,
            norm_eps,
        )
        .map(|()| true)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn upload_gemma_ple_base(data: &[f32]) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        cuda_runtime::upload_gemma_ple_base(data).map_err(cuda_error)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
// cu29 Phase 2: Llama / Mistral hd=128 fused Q4K QKV + GPU RoPE + f16 K/V
// pack. Host RoPE round-trip 제거. opt-in via RNB_CUDA_HD128_FUSED_QKV_ROPE.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn dense_q4k_attention_qkv_rope_hd128_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    input: &[f32],
    q_rope: &mut [f32],
    k_bits: &mut [u16],
    v_bits: &mut [u16],
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        // cu30: decode 전용 env (master env 또는 decode-only).
        let master = crate::engine::policy::env_string("RNB_CUDA_HD128_FUSED_QKV_ROPE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let decode_only = crate::engine::policy::env_string("RNB_CUDA_HD128_FUSED_QKV_ROPE_DECODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !master && !decode_only {
            return Ok(false);
        }
        let (Some(q), Some(k), Some(v)) = (
            q_weight.backend_view(),
            k_weight.backend_view(),
            v_weight.backend_view(),
        ) else {
            return Ok(false);
        };
        if q.quant() != QuantFormat::Q4K
            || k.quant() != QuantFormat::Q4K
            || v.quant() != QuantFormat::Q4K
        {
            return Ok(false);
        }
        if q.rows() != num_heads * 128 || k.rows() != num_kv_heads * 128 {
            return Ok(false);
        }
        cuda_runtime::dense_q4k_attention_qkv_rope_hd128_decode(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            input,
            q_rope,
            k_bits,
            v_bits,
        )
        .map_err(cuda_error)?;
        return Ok(true);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

// cu30 Phase 2c: multi-token (prefill) 변형. seq_len 만큼 한 번에 처리.
// 결과 — q_vec (RoPE 적용 f32), k_bits/v_bits (f16 packed). caller 가
// PrefillAttentionProjection.cached_kv_f16 에 그대로 채우면 forward.rs 의
// apply_prefill_rope + host f32→f16 conversion 자동 skip.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn dense_q4k_attention_qkv_rope_hd128_prefill_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    seq_len: usize,
    input: &[f32],
) -> crate::error::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>> {
    #[cfg(feature = "cuda")]
    {
        let master = crate::engine::policy::env_string("RNB_CUDA_HD128_FUSED_QKV_ROPE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let prefill_only =
            crate::engine::policy::env_string("RNB_CUDA_HD128_FUSED_QKV_ROPE_PREFILL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        if !master && !prefill_only {
            return Ok(None);
        }
        let (Some(q), Some(k), Some(v)) = (
            q_weight.backend_view(),
            k_weight.backend_view(),
            v_weight.backend_view(),
        ) else {
            return Ok(None);
        };
        if q.quant() != QuantFormat::Q4K
            || k.quant() != QuantFormat::Q4K
            || v.quant() != QuantFormat::Q4K
        {
            return Ok(None);
        }
        if q.rows() != num_heads * 128 || k.rows() != num_kv_heads * 128 {
            return Ok(None);
        }
        let q_rows = q.rows();
        let kv_rows = k.rows();
        let mut q_vec = vec![0.0f32; seq_len * q_rows];
        let mut k_bits = vec![0u16; seq_len * kv_rows];
        let mut v_bits = vec![0u16; seq_len * kv_rows];
        cuda_runtime::dense_q4k_attention_qkv_rope_hd128_prefill(
            q.raw(),
            k.raw(),
            v.raw(),
            q_rows,
            kv_rows,
            q.cols(),
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            seq_len,
            input,
            &mut q_vec,
            &mut k_bits,
            &mut v_bits,
        )
        .map_err(cuda_error)?;
        return Ok(Some((q_vec, k_bits, v_bits)));
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    unit_offset: bool,
) -> crate::error::Result<()> {
    if weight.is_empty() || input.len() != output.len() || input.len() % weight.len() != 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "CUDA RMS norm shape mismatch: input={} weight={} output={}",
            input.len(),
            weight.len(),
            output.len()
        )));
    }
    cuda_runtime::rms_norm_rows_f32(input, weight, output, eps, unit_offset).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_activation_mul_inplace(
    gate: &mut [f32],
    up: &[f32],
    gelu: bool,
) -> crate::error::Result<()> {
    cuda_runtime::activation_mul_f32_inplace(gate, up, gelu).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_add_f32_inplace(
    dst: &mut [f32],
    src: &[f32],
) -> crate::error::Result<()> {
    cuda_runtime::add_f32_inplace(dst, src).map_err(cuda_error)
}
#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_add_rows_f32_inplace(
    dst: &mut [f32],
    src: &[f32],
) -> crate::error::Result<()> {
    cuda_runtime::add_rows_f32_inplace(dst, src).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_mul_rows_f32_inplace(
    dst: &mut [f32],
    src: &[f32],
) -> crate::error::Result<()> {
    cuda_runtime::mul_rows_f32_inplace(dst, src).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_scale_f32_inplace(
    values: &mut [f32],
    scale: f32,
) -> crate::error::Result<()> {
    cuda_runtime::scale_f32_inplace(values, scale).map_err(cuda_error)
}

#[cfg(all(feature = "cuda", test))]
pub(in crate::engine) fn cuda_sigmoid_f32_inplace(values: &mut [f32]) -> crate::error::Result<()> {
    cuda_runtime::sigmoid_f32_inplace(values).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_gdn_prepare_delta_gate_beta_f32(
    alpha: &mut [f32],
    beta: &mut [f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    num_heads: usize,
) -> crate::error::Result<()> {
    cuda_runtime::gdn_prepare_delta_gate_beta_f32(alpha, beta, dt_bias, ssm_a, num_heads)
        .map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_l2_norm_rows(
    input: &[f32],
    output: &mut [f32],
    row_width: usize,
    eps: f32,
) -> crate::error::Result<()> {
    cuda_runtime::l2_norm_rows_f32(input, output, row_width, eps).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_axpby_f32_inplace(
    dst: &mut [f32],
    src: &[f32],
    alpha: f32,
    beta: f32,
) -> crate::error::Result<()> {
    cuda_runtime::axpby_f32_inplace(dst, src, alpha, beta).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_sigmoid_mul_f32_inplace(
    values: &mut [f32],
    gate: &[f32],
) -> crate::error::Result<()> {
    cuda_runtime::sigmoid_mul_f32_inplace(values, gate).map_err(cuda_error)
}

#[cfg(all(feature = "cuda", test))]
pub(in crate::engine) fn cuda_relu_sqr_f32_inplace(values: &mut [f32]) -> crate::error::Result<()> {
    cuda_runtime::relu_sqr_f32_inplace(values).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_hadamard_f32_inplace(
    values: &mut [f32],
    chunk_len: usize,
) -> crate::error::Result<()> {
    cuda_runtime::hadamard_f32_inplace(values, chunk_len).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum CudaForwardRopeMode {
    Adjacent,
    Neox,
    PartialAdjacent,
    PartialProportional,
    MropeText,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn cuda_rope_f32_inplace(
    values: &mut [f32],
    dim: usize,
    head_dim: usize,
    n_rot: usize,
    pos_start: usize,
    theta: f32,
    mode: CudaForwardRopeMode,
    factors: Option<&[f32]>,
) -> crate::error::Result<()> {
    let mode = match mode {
        CudaForwardRopeMode::Adjacent => cuda_runtime::CudaRopeMode::Adjacent,
        CudaForwardRopeMode::Neox => cuda_runtime::CudaRopeMode::Neox,
        CudaForwardRopeMode::PartialAdjacent => cuda_runtime::CudaRopeMode::PartialAdjacent,
        CudaForwardRopeMode::PartialProportional => cuda_runtime::CudaRopeMode::PartialProportional,
        CudaForwardRopeMode::MropeText => cuda_runtime::CudaRopeMode::MropeText,
    };
    cuda_runtime::rope_f32_inplace(
        values, dim, head_dim, n_rot, pos_start, theta, mode, factors,
    )
    .map_err(cuda_error)
}

// cu41 Phase 1 step 5: cuda RMS norm into decode hidden carrier + D2H + sync
// (fallback path 호환). env opt-in `RNB_CUDA_DECODE_DEVICE_CHAIN=1` 시에만 호출.
//
// step 7/8 revert: D2H + sync 제거 시 chat correctness 깨짐 (EOS step 5 vs
// baseline 9). dedicated input buffer (step 8) 도 효과 없음 — race 원인 다른 곳.
// 추가 정찰 필요 (stream/kernel/cache 의 어디서 ordering 깨지는지).
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn try_rms_norm_into_decode_carrier_if_supported(
    layer_idx: usize,
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    unit_offset: bool,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if !crate::engine::policy::cuda_decode_device_chain_enabled() {
            return Ok(false);
        }
        if input.len() != weight.len() || input.len() != output.len() {
            return Ok(false);
        }
        let bytes = std::mem::size_of_val(input);
        let Ok(norm_carrier) = cuda_runtime::acquire_decode_norm_buf_carrier(bytes) else {
            return Ok(false);
        };
        // cu42 step 14: layer_idx > 0 시 chain function 의 hidden_carrier output
        // (이전 layer 결과) 을 device input 으로 사용. layer 0 은 host input
        // (embed lookup 결과) 그대로.
        if layer_idx > 0 {
            let Ok(hidden_carrier) = cuda_runtime::acquire_decode_hidden_carrier(bytes) else {
                return Ok(false);
            };
            // cu44 diag: try_rms_norm 직전 시점 carrier 의 device 값을 read 해서
            // host input (= 이전 layer chain end 의 host scratch.hidden) 과 비교.
            if crate::engine::policy::env_string("RNB_CU44_DIAG_PRE_RMS").is_some() {
                let mut dbg = vec![0.0f32; input.len()];
                let dl =
                    cuda_runtime::download_from_decode_hidden_carrier(hidden_carrier, &mut dbg)
                        .and_then(|_| cuda_runtime::sync_decode_stream());
                if let Err(e) = dl {
                    eprintln!("[cu44 diag pre_rms] download err={}", e);
                } else {
                    let n = input.len().min(4);
                    let max_diff = input
                        .iter()
                        .zip(dbg.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "[cu44 diag pre_rms layer={}] host[..{}]={:?} carrier[..{}]={:?} max_diff={}",
                        layer_idx,
                        n,
                        &input[..n],
                        n,
                        &dbg[..n],
                        max_diff
                    );
                }
            }
            cuda_runtime::rms_norm_f32_dev_input_to_carrier(
                hidden_carrier,
                weight,
                norm_carrier,
                input.len(),
                eps,
                unit_offset,
            )
            .map_err(cuda_error)?;
        } else {
            cuda_runtime::rms_norm_f32_to_carrier(input, weight, norm_carrier, eps, unit_offset)
                .map_err(cuda_error)?;
        }
        // cu44 step 22 (D2H + sync) revert — cu45 step 25.
        // cu45 step 23 의 Q6K device input wire 로 V=Q6K layer 의 individual q/k/v
        // path 도 device input 사용. host scratch.norm_buf 미필요 → D2H 제거.
        // 진짜 lever 도달: norm_carrier → host D2H 자체 제거.
        return Ok(true);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn dense_q4k_attention_qkv_if_supported(
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    input: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    rms_used_cuda: bool,
) -> crate::error::Result<bool> {
    let (Some(q), Some(k), Some(v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    if q.quant() != QuantFormat::Q4K
        || k.quant() != QuantFormat::Q4K
        || v.quant() != QuantFormat::Q4K
    {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        // cu41 Phase 1 step 6 + cu57 step 67b: env opt-in + rms_used_cuda (caller
        // 가 norm_buf_carrier 가 fresh 함을 보장) 일 때만 device-input variant.
        // chain function 미호출 arch (Nemotron 등) 가 effective_rms_layer=0
        // 강제로 carrier 에 layer 0 norm 만 들고 있을 때 attention QKV input
        // 부정확 → garbage. rms_used_cuda=false 시 host input fallback 으로 회피.
        if crate::engine::policy::cuda_decode_device_chain_enabled() && rms_used_cuda {
            let bytes = std::mem::size_of_val(input);
            if let Ok(carrier) = cuda_runtime::acquire_decode_norm_buf_carrier(bytes) {
                cuda_runtime::dense_q4k_attention_qkv_with_device_input(
                    q.raw(),
                    k.raw(),
                    v.raw(),
                    q.rows(),
                    k.rows(),
                    q.cols(),
                    carrier,
                    q_out,
                    k_out,
                    v_out,
                )
                .map_err(cuda_error)?;
                return Ok(true);
            }
        }
        cuda_runtime::dense_q4k_attention_qkv(
            q.raw(),
            k.raw(),
            v.raw(),
            q.rows(),
            k.rows(),
            q.cols(),
            input,
            q_out,
            k_out,
            v_out,
        )
        .map_err(cuda_error)?;
        return Ok(true);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn decode_attention_hd256_if_supported(
    layer_index: Option<usize>,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> Option<crate::error::Result<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::decode_attention_hd256_if_supported(
            layer_index,
            q,
            k,
            v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            has_softcap,
        )
        .map(|result| result.map_err(cuda_error));
    }
    #[cfg(not(feature = "cuda"))]
    None
}

// cu47 step 32: decode_attention 의 device output variant. caller 가 carrier
// 제공 → attention compute 결과 device 에 유지. host scratch.attn_out 의 D2H 제거.
// 조건: cached path 사용 (kv_cache_enabled + head_dim ∈ {128,256,512}). 그 외 None.
// host post-processing (gated_attn, gemma4_rot) 적용 필요 시 caller 는 carrier
// 사용 안 함 (None 받음). 적용 안 되는 layer + env opt-in 시만 device chain.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn decode_attention_cached_to_device_if_supported(
    layer_index: Option<usize>,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
    use_device_len: bool,
) -> Option<crate::error::Result<()>> {
    if has_softcap
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || !matches!(head_dim, 128 | 256 | 512)
        || sliding_window.is_some()
    {
        // sliding window 또는 hd unsupported 시 caller 가 host return path 로 fallback.
        return None;
    }
    if use_device_len && head_dim != 512 {
        return Some(Err(cuda_error(format!(
            "cu68 device-length attention requires head_dim=512, got {head_dim}"
        ))));
    }
    let layer_idx = layer_index?;
    #[cfg(feature = "cuda")]
    {
        if use_device_len {
            return Some(
                cuda_runtime::attention_decode_cached_to_device_len_device_graph(
                    layer_idx,
                    q,
                    k,
                    v,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    output_dev_target,
                    last_token_k_dev,
                    last_token_v_dev,
                    q_dev_override,
                )
                .map_err(cuda_error),
            );
        }
        Some(
            cuda_runtime::attention_decode_cached_to_device(
                layer_idx,
                q,
                k,
                v,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                output_dev_target,
                last_token_k_dev,
                last_token_v_dev,
                q_dev_override,
            )
            .map_err(cuda_error),
        )
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn decode_attention_kvarn_to_device_if_supported(
    layer_index: usize,
    q: &[f32],
    cache: crate::engine::cpu_runtime::quantize::kvarn::KvarnKvView<'_>,
    num_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
    output_dev_target: u64,
) -> Option<crate::error::Result<()>> {
    if !cuda_runtime::kvarn_attention_enabled(true, cache.len, num_heads, cache.head_dim) {
        return None;
    }
    if cache.num_kv_heads == 0
        || num_heads % cache.num_kv_heads != 0
        || !matches!(cache.head_dim, 128 | 256 | 512)
    {
        return None;
    }
    let request = cuda_runtime::KvarnDecodeRequest::new(
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
    Some(
        cuda_runtime::attention_decode_kvarn_to_device(request, output_dev_target)
            .map_err(cuda_error),
    )
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn decode_attention_kvarn_into_if_supported(
    layer_index: usize,
    q: &[f32],
    cache: crate::engine::cpu_runtime::quantize::kvarn::KvarnKvView<'_>,
    output: &mut [f32],
    num_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<bool> {
    if !cuda_runtime::kvarn_attention_enabled(true, cache.len, num_heads, cache.head_dim)
        || cache.num_kv_heads == 0
        || num_heads % cache.num_kv_heads != 0
        || !matches!(cache.head_dim, 128 | 256 | 512)
    {
        return Ok(false);
    }
    let request = cuda_runtime::KvarnDecodeRequest::new(
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
    let result = cuda_runtime::attention_decode_kvarn(request).map_err(cuda_error)?;
    output.copy_from_slice(&result);
    Ok(true)
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

#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn q5_basic_gemv(
    weight: &QuantizedWeight,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    let Some(raw) = weight.data.as_bytes() else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    {
        let result = match weight.ggml_type {
            rnb_loader::GGMLType::Q5_0 => Some(cuda_runtime::q5_0_gemv(
                raw,
                weight.rows,
                weight.cols,
                input,
            )),
            rnb_loader::GGMLType::Q5_1 => Some(cuda_runtime::q5_1_gemv(
                raw,
                weight.rows,
                weight.cols,
                input,
            )),
            _ => None,
        };
        return result.transpose().map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn q5_basic_gemv_raw(
    ggml_type: rnb_loader::GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        let result = match ggml_type {
            rnb_loader::GGMLType::Q5_0 => Some(cuda_runtime::q5_0_gemv(raw, rows, cols, input)),
            rnb_loader::GGMLType::Q5_1 => Some(cuda_runtime::q5_1_gemv(raw, rows, cols, input)),
            _ => None,
        };
        return result.transpose().map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_sparse_relu_sqr_by_token(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_decode_moe_shared_sparse(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_decode_moe_shared_sparse(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            n_ff,
            n_embd,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_prefill(
    shared_up: &[u8],
    shared_down: &[u8],
    shared_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_prefill(
            shared_up,
            shared_down,
            shared_ff,
            n_embd,
            token_count,
            input,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_q5_sparse_prefill_moe(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input,
            residual,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(feature = "cuda")]
fn release_device_tensor_ids_for_probe(ids: &[cuda_runtime::DeviceTensorId]) -> Result<(), String> {
    let mut errors = Vec::new();
    for &id in ids {
        match cuda_runtime::release_device_tensor(id) {
            Ok(true) => {}
            Ok(false) => errors.push(format!("missing CUDA device tensor id {}", id.raw())),
            Err(err) => errors.push(err),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

#[cfg(feature = "cuda")]
fn cuda_error_with_cleanup(err: String, cleanup: Result<(), String>) -> crate::error::LlmError {
    match cleanup {
        Ok(()) => cuda_error(err),
        Err(cleanup_err) => cuda_error(format!("{err}; cleanup failed: {cleanup_err}")),
    }
}

#[cfg(feature = "cuda")]
fn cuda_cleanup_failure_message(
    primary: &str,
    cleanup_label: &str,
    cleanup: Result<bool, String>,
) -> String {
    match cleanup {
        Ok(true) => primary.to_string(),
        Ok(false) => format!("{primary}; {cleanup_label} cleanup missing"),
        Err(cleanup_err) => {
            format!("{primary}; {cleanup_label} cleanup failed: {cleanup_err}")
        }
    }
}

#[cfg(feature = "cuda")]
fn release_mamba2_input_or_cleanup_output<ReleaseInput, ReleaseOutput>(
    release_input: ReleaseInput,
    release_output: ReleaseOutput,
) -> Result<(), String>
where
    ReleaseInput: FnOnce() -> Result<bool, String>,
    ReleaseOutput: FnOnce() -> Result<bool, String>,
{
    match release_input() {
        Ok(true) => Ok(()),
        Ok(false) => Err(cuda_cleanup_failure_message(
            "CUDA Mamba2 input device tensor was already missing",
            "output",
            release_output(),
        )),
        Err(err) => Err(cuda_cleanup_failure_message(
            &err,
            "output",
            release_output(),
        )),
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NemotronDevicePrefillTraceBytes {
    h2d_bytes: usize,
    d2h_bytes: usize,
}

#[cfg(feature = "cuda")]
fn nemotron_device_prefill_trace_bytes(
    input_bytes: usize,
    residual_bytes: usize,
    output_download_bytes: Option<usize>,
) -> crate::error::Result<NemotronDevicePrefillTraceBytes> {
    let h2d_bytes = input_bytes.checked_add(residual_bytes).ok_or_else(|| {
        cuda_error("CUDA device prefill trace H2D byte length overflow".to_string())
    })?;
    Ok(NemotronDevicePrefillTraceBytes {
        h2d_bytes,
        d2h_bytes: output_download_bytes.unwrap_or(0),
    })
}

#[cfg(feature = "cuda")]
fn emit_nemotron_device_prefill_trace(
    input_desc: cuda_runtime::DeviceTensorDesc,
    residual_desc: cuda_runtime::DeviceTensorDesc,
    output_desc: cuda_runtime::DeviceTensorDesc,
    output_downloaded: bool,
) -> crate::error::Result<()> {
    if !crate::engine::policy::cuda_device_prefill_trace_enabled() {
        return Ok(());
    }

    let input_bytes = input_desc.byte_len().ok_or_else(|| {
        cuda_error("CUDA device prefill trace input byte length overflow".to_string())
    })?;
    let residual_bytes = residual_desc.byte_len().ok_or_else(|| {
        cuda_error("CUDA device prefill trace residual byte length overflow".to_string())
    })?;
    let output_bytes = output_desc.byte_len().ok_or_else(|| {
        cuda_error("CUDA device prefill trace D2H byte length overflow".to_string())
    })?;
    let trace = nemotron_device_prefill_trace_bytes(
        input_bytes,
        residual_bytes,
        output_downloaded.then_some(output_bytes),
    )?;

    eprintln!(
        "[cuda:device-prefill] op=nemotron_moe tokens={} hidden={} h2d_bytes={} d2h_bytes={} output_device=1",
        output_desc.rows(),
        output_desc.cols(),
        trace.h2d_bytes,
        trace.d2h_bytes
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
#[allow(dead_code)]
pub(in crate::engine) struct NemotronDeviceLayerOutput {
    pub output_id: cuda_runtime::DeviceTensorId,
    pub output_desc: cuda_runtime::DeviceTensorDesc,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(in crate::engine) struct QwenAttentionDeviceInputOutput {
    pub normalized: NemotronDeviceLayerOutput,
    pub residual: NemotronDeviceLayerOutput,
    pub attention_kv: cuda_runtime::MtpDeviceVerifyAttentionKvState,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
#[allow(dead_code)]
pub(in crate::engine) struct NemotronDeviceRouterLogitsOutput {
    pub normalized_id: cuda_runtime::DeviceTensorId,
    pub normalized_desc: cuda_runtime::DeviceTensorDesc,
    pub router_logits_id: cuda_runtime::DeviceTensorId,
    pub router_logits_desc: cuda_runtime::DeviceTensorDesc,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
#[allow(dead_code)]
pub(in crate::engine) struct NemotronDeviceRoutePack {
    inner: cuda_runtime::NemotronDeviceRoutePack,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl NemotronDeviceRoutePack {
    pub(in crate::engine) fn slots(&self) -> usize {
        self.inner.slots()
    }

    pub(in crate::engine) fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    pub(in crate::engine) fn expert_used(&self) -> usize {
        self.inner.expert_used()
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl NemotronDeviceLayerOutput {
    pub(in crate::engine) fn release(self) -> crate::error::Result<bool> {
        cuda_runtime::release_device_tensor(self.output_id).map_err(cuda_error)
    }
}
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn qwen35_prefill_attention_device_input(
    input: &NemotronDeviceLayerOutput,
    layer: &cuda_runtime::MtpDeviceVerifyAttentionMoeLayer<'_>,
    seq_len: usize,
    hidden_dim: usize,
    rope_dim: usize,
    rope_neox: bool,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    mirror_attention_kv_to_host: bool,
) -> crate::error::Result<QwenAttentionDeviceInputOutput> {
    let output = cuda_runtime::qwen35_prefill_attention_device_input(
        input.output_id,
        input.output_desc,
        layer,
        seq_len,
        hidden_dim,
        rope_dim,
        rope_neox,
        rope_theta,
        pos_start,
        norm_eps,
        mirror_attention_kv_to_host,
    )
    .map_err(cuda_error)?;
    Ok(QwenAttentionDeviceInputOutput {
        normalized: NemotronDeviceLayerOutput {
            output_id: output.normalized_id,
            output_desc: output.normalized_desc,
        },
        residual: NemotronDeviceLayerOutput {
            output_id: output.residual_id,
            output_desc: output.residual_desc,
        },
        attention_kv: output.attention_kv,
    })
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn upload_hidden_device_output_f32(
    values: &[f32],
    rows: usize,
    cols: usize,
) -> crate::error::Result<NemotronDeviceLayerOutput> {
    let desc = cuda_runtime::DeviceTensorDesc::new(
        rows,
        cols,
        cuda_runtime::ScalarType::F32,
        cuda_runtime::DeviceTensorRole::Hidden,
    );
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "CUDA hidden upload element count overflow: rows={rows} cols={cols}"
        ))
    })?;
    if values.len() != expected {
        return Err(crate::error::LlmError::Forward(format!(
            "CUDA hidden upload length mismatch: got {}, expected {expected}",
            values.len()
        )));
    }
    let output_id = cuda_runtime::upload_device_tensor_f32(desc, values).map_err(cuda_error)?;
    Ok(NemotronDeviceLayerOutput {
        output_id,
        output_desc: desc,
    })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl NemotronDeviceRouterLogitsOutput {
    pub(in crate::engine) fn release_normalized(&self) -> crate::error::Result<bool> {
        cuda_runtime::release_device_tensor(self.normalized_id).map_err(cuda_error)
    }

    pub(in crate::engine) fn release_router_logits(&self) -> crate::error::Result<bool> {
        cuda_runtime::release_device_tensor(self.router_logits_id).map_err(cuda_error)
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(in crate::engine) struct NemotronDevicePrefillHandoff {
    pub input_id: cuda_runtime::DeviceTensorId,
    pub residual_id: cuda_runtime::DeviceTensorId,
    pub output_id: cuda_runtime::DeviceTensorId,
    pub input_desc: cuda_runtime::DeviceTensorDesc,
    pub residual_desc: cuda_runtime::DeviceTensorDesc,
    pub output_desc: cuda_runtime::DeviceTensorDesc,
}

#[cfg(feature = "cuda")]
impl NemotronDevicePrefillHandoff {
    fn release_all(self) -> Result<(), String> {
        release_device_tensor_ids_for_probe(&[self.input_id, self.residual_id, self.output_id])
    }

    #[allow(dead_code)]
    pub(in crate::engine) fn into_layer_output(
        self,
    ) -> crate::error::Result<NemotronDeviceLayerOutput> {
        let output = NemotronDeviceLayerOutput {
            output_id: self.output_id,
            output_desc: self.output_desc,
        };
        match release_device_tensor_ids_for_probe(&[self.input_id, self.residual_id]) {
            Ok(()) => Ok(output),
            Err(err) => {
                let output_cleanup = cuda_runtime::release_device_tensor(output.output_id);
                Err(cuda_error(cuda_cleanup_failure_message(
                    &format!("CUDA device handoff input cleanup failed: {err}"),
                    "output",
                    output_cleanup,
                )))
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(in crate::engine) struct NemotronMamba2DeviceTrace {
    pub boundary_d2h_bytes: usize,
    pub hidden_d2h_bytes: usize,
    pub conv_state_d2h_bytes: usize,
    pub delta_state_d2h_bytes: usize,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl NemotronMamba2DeviceTrace {
    pub(in crate::engine) fn mamba2_state_d2h_bytes(self) -> usize {
        self.conv_state_d2h_bytes
            .saturating_add(self.delta_state_d2h_bytes)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_router_logits_from_device_f32(
    input_id: cuda_runtime::DeviceTensorId,
    input_desc: cuda_runtime::DeviceTensorDesc,
    norm_weight: &[f32],
    router_weight_f32: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    n_expert: usize,
    norm_eps: f32,
) -> crate::error::Result<NemotronDeviceRouterLogitsOutput> {
    let output = cuda_runtime::nemotron_router_logits_from_device_f32(
        input_id,
        input_desc,
        norm_weight,
        router_weight_f32,
        seq_len,
        hidden_dim,
        n_expert,
        norm_eps,
    )
    .map_err(cuda_error)?;
    Ok(NemotronDeviceRouterLogitsOutput {
        normalized_id: output.normalized_id,
        normalized_desc: output.normalized_desc,
        router_logits_id: output.router_logits_id,
        router_logits_desc: output.router_logits_desc,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_device_route_pack_from_logits(
    router_logits_id: cuda_runtime::DeviceTensorId,
    router_logits_desc: cuda_runtime::DeviceTensorDesc,
    bias: Option<&[f32]>,
    seq_len: usize,
    n_expert: usize,
    expert_used: usize,
    expert_weight_scale: f32,
) -> crate::error::Result<NemotronDeviceRoutePack> {
    let inner = cuda_runtime::nemotron_device_route_pack_from_logits(
        router_logits_id,
        router_logits_desc,
        bias,
        seq_len,
        n_expert,
        expert_used,
        expert_weight_scale,
    )
    .map_err(cuda_error)?;
    Ok(NemotronDeviceRoutePack { inner })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_device_route_pack_expert_ids(
    route: &NemotronDeviceRoutePack,
) -> crate::error::Result<Vec<u32>> {
    cuda_runtime::nemotron_device_route_pack_expert_ids(&route.inner).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_reorder_device_route_pack(
    route: &NemotronDeviceRoutePack,
    order_indices: &[u32],
) -> crate::error::Result<NemotronDeviceRoutePack> {
    let inner = cuda_runtime::nemotron_reorder_device_route_pack(&route.inner, order_indices)
        .map_err(cuda_error)?;
    Ok(NemotronDeviceRoutePack { inner })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn release_nemotron_device_route_pack(
    route: NemotronDeviceRoutePack,
) -> crate::error::Result<()> {
    cuda_runtime::release_nemotron_device_route_pack(route.inner).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack_ids(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_pack: &NemotronDeviceRoutePack,
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    normalized_id: cuda_runtime::DeviceTensorId,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
) -> crate::error::Result<Option<NemotronDeviceLayerOutput>> {
    let output_id = match cuda_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        &route_pack.inner,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        normalized_id,
        residual_id,
        residual_desc,
    ) {
        Ok(Some(output_id)) => output_id,
        Ok(None) => {
            release_device_tensor_ids_for_probe(&[normalized_id]).map_err(|err| {
                cuda_error(format!(
                    "CUDA device route-pack MoE normalized input cleanup failed: {err}"
                ))
            })?;
            return Ok(None);
        }
        Err(err) => {
            let cleanup = release_device_tensor_ids_for_probe(&[normalized_id]);
            return Err(cuda_error_with_cleanup(err, cleanup));
        }
    };
    let output = NemotronDeviceLayerOutput {
        output_id,
        output_desc: cuda_runtime::DeviceTensorDesc::new(
            token_count,
            n_embd,
            cuda_runtime::ScalarType::F32,
            cuda_runtime::DeviceTensorRole::MoeOutput,
        ),
    };
    match release_device_tensor_ids_for_probe(&[normalized_id, residual_id]) {
        Ok(()) => Ok(Some(output)),
        Err(err) => {
            let output_cleanup = cuda_runtime::release_device_tensor(output.output_id);
            Err(cuda_error(cuda_cleanup_failure_message(
                &format!("CUDA device route-pack MoE input cleanup failed: {err}"),
                "output",
                output_cleanup,
            )))
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn download_cuda_device_tensor_f32(
    id: cuda_runtime::DeviceTensorId,
) -> crate::error::Result<Vec<f32>> {
    cuda_runtime::download_device_tensor_f32(id).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn download_cuda_device_tensor_f32_row(
    id: cuda_runtime::DeviceTensorId,
    row: usize,
) -> crate::error::Result<Vec<f32>> {
    cuda_runtime::download_device_tensor_f32_row(id, row).map_err(cuda_error)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_mamba2_prefill_device(
    input: NemotronDeviceLayerOutput,
    ssm_in_quant: rnb_loader::GGMLType,
    ssm_in: &[u8],
    ssm_in_rows: usize,
    ssm_in_cols: usize,
    ssm_out_quant: rnb_loader::GGMLType,
    ssm_out: &[u8],
    ssm_out_rows: usize,
    ssm_out_cols: usize,
    input_norm: &[f32],
    conv_kernel: &[f32],
    conv_bias: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_d: &[f32],
    ssm_norm: &[f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    d_inner: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    n_group: usize,
    d_state: usize,
    conv_kernel_size: usize,
    norm_eps: f32,
) -> crate::error::Result<(NemotronDeviceLayerOutput, NemotronMamba2DeviceTrace)> {
    let result = match cuda_runtime::nemotron_mamba2_prefill_device(
        input.output_id,
        input.output_desc,
        ssm_in_quant as u32,
        ssm_in,
        ssm_in_rows,
        ssm_in_cols,
        ssm_out_quant as u32,
        ssm_out,
        ssm_out_rows,
        ssm_out_cols,
        input_norm,
        conv_kernel,
        conv_bias,
        dt_bias,
        ssm_a,
        ssm_d,
        ssm_norm,
        conv_state,
        delta_state,
        seq_len,
        hidden_dim,
        d_inner,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        n_group,
        d_state,
        conv_kernel_size,
        norm_eps,
    ) {
        Ok(result) => result,
        Err(err) => {
            let cleanup = cuda_runtime::release_device_tensor(input.output_id);
            return match cleanup {
                Ok(_) => Err(cuda_error(err)),
                Err(cleanup_err) => {
                    Err(cuda_error(format!("{err}; cleanup failed: {cleanup_err}")))
                }
            };
        }
    };
    release_mamba2_input_or_cleanup_output(
        || cuda_runtime::release_device_tensor(input.output_id),
        || cuda_runtime::release_device_tensor(result.output_id),
    )
    .map_err(cuda_error)?;
    Ok((
        NemotronDeviceLayerOutput {
            output_id: result.output_id,
            output_desc: result.output_desc,
        },
        NemotronMamba2DeviceTrace {
            boundary_d2h_bytes: 0,
            hidden_d2h_bytes: 0,
            conv_state_d2h_bytes: result.conv_state_d2h_bytes,
            delta_state_d2h_bytes: result.delta_state_d2h_bytes,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn create_nemotron_q8_shared_q5_sparse_prefill_moe_handoff(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<NemotronDevicePrefillHandoff>> {
    let input_desc = cuda_runtime::DeviceTensorDesc::new(
        token_count,
        n_embd,
        cuda_runtime::ScalarType::F32,
        cuda_runtime::DeviceTensorRole::Normalized,
    );
    let residual_desc = cuda_runtime::DeviceTensorDesc::new(
        token_count,
        n_embd,
        cuda_runtime::ScalarType::F32,
        cuda_runtime::DeviceTensorRole::Residual,
    );
    let output_desc = cuda_runtime::DeviceTensorDesc::new(
        token_count,
        n_embd,
        cuda_runtime::ScalarType::F32,
        cuda_runtime::DeviceTensorRole::MoeOutput,
    );
    let input_id = cuda_runtime::upload_device_tensor_f32(input_desc, input).map_err(cuda_error)?;
    let residual_id = match cuda_runtime::upload_device_tensor_f32(residual_desc, residual) {
        Ok(id) => id,
        Err(err) => {
            let cleanup = release_device_tensor_ids_for_probe(&[input_id]);
            return Err(cuda_error_with_cleanup(err, cleanup));
        }
    };
    let output_id = match cuda_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_output(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
    ) {
        Ok(Some(id)) => id,
        Ok(None) => {
            release_device_tensor_ids_for_probe(&[input_id, residual_id])
                .map_err(|err| cuda_error(format!("CUDA device probe cleanup failed: {err}")))?;
            return Ok(None);
        }
        Err(err) => {
            let cleanup = release_device_tensor_ids_for_probe(&[input_id, residual_id]);
            return Err(cuda_error_with_cleanup(err, cleanup));
        }
    };
    Ok(Some(NemotronDevicePrefillHandoff {
        input_id,
        residual_id,
        output_id,
        input_desc,
        residual_desc,
        output_desc,
    }))
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_probe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        let Some(handoff) = create_nemotron_q8_shared_q5_sparse_prefill_moe_handoff(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input,
            residual,
        )?
        else {
            return Ok(None);
        };
        let input_desc = handoff.input_desc;
        let residual_desc = handoff.residual_desc;
        let output_desc = handoff.output_desc;
        let output = download_nemotron_device_prefill_handoff(handoff)?;
        emit_nemotron_device_prefill_trace(input_desc, residual_desc, output_desc, true)?;
        return Ok(Some(output));
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_output(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<NemotronDevicePrefillHandoff>> {
    let Some(handoff) = create_nemotron_q8_shared_q5_sparse_prefill_moe_handoff(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input,
        residual,
    )?
    else {
        return Ok(None);
    };
    let trace = emit_nemotron_device_prefill_trace(
        handoff.input_desc,
        handoff.residual_desc,
        handoff.output_desc,
        false,
    );
    if let Err(err) = trace {
        let cleanup = handoff.release_all();
        return match cleanup {
            Ok(()) => Err(err),
            Err(cleanup_err) => match err {
                crate::error::LlmError::Forward(msg) => {
                    Err(cuda_error(format!("{msg}; cleanup failed: {cleanup_err}")))
                }
                err => Err(cuda_error(format!("{err}; cleanup failed: {cleanup_err}"))),
            },
        };
    }
    Ok(Some(handoff))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_ids(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    normalized_id: cuda_runtime::DeviceTensorId,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
) -> crate::error::Result<Option<NemotronDeviceLayerOutput>> {
    let output_id =
        match cuda_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            normalized_id,
            residual_id,
            residual_desc,
        ) {
            Ok(Some(output_id)) => output_id,
            Ok(None) => {
                release_device_tensor_ids_for_probe(&[normalized_id]).map_err(|err| {
                    cuda_error(format!(
                        "CUDA device MoE normalized input cleanup failed: {err}"
                    ))
                })?;
                return Ok(None);
            }
            Err(err) => {
                let cleanup = release_device_tensor_ids_for_probe(&[normalized_id]);
                return Err(cuda_error_with_cleanup(err, cleanup));
            }
        };
    let output = NemotronDeviceLayerOutput {
        output_id,
        output_desc: cuda_runtime::DeviceTensorDesc::new(
            token_count,
            n_embd,
            cuda_runtime::ScalarType::F32,
            cuda_runtime::DeviceTensorRole::MoeOutput,
        ),
    };
    match release_device_tensor_ids_for_probe(&[normalized_id, residual_id]) {
        Ok(()) => Ok(Some(output)),
        Err(err) => {
            let output_cleanup = cuda_runtime::release_device_tensor(output.output_id);
            Err(cuda_error(cuda_cleanup_failure_message(
                &format!("CUDA device MoE input cleanup failed: {err}"),
                "output",
                output_cleanup,
            )))
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn download_nemotron_device_prefill_handoff(
    handoff: NemotronDevicePrefillHandoff,
) -> crate::error::Result<Vec<f32>> {
    let download = cuda_runtime::download_device_tensor_f32(handoff.output_id);
    let cleanup = handoff.release_all();
    match (download, cleanup) {
        (Ok(output), Ok(())) => Ok(output),
        (Ok(_), Err(cleanup_err)) => Err(cuda_error(format!(
            "CUDA device handoff cleanup failed: {cleanup_err}"
        ))),
        (Err(err), Ok(())) => Err(cuda_error(err)),
        (Err(err), Err(cleanup_err)) => {
            Err(cuda_error(format!("{err}; cleanup failed: {cleanup_err}")))
        }
    }
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn download_nemotron_device_layer_output(
    output: NemotronDeviceLayerOutput,
) -> crate::error::Result<Vec<f32>> {
    let download = cuda_runtime::download_device_tensor_f32(output.output_id);
    let cleanup = cuda_runtime::release_device_tensor(output.output_id);
    match (download, cleanup) {
        (Ok(values), Ok(true)) => Ok(values),
        (Ok(_), Ok(false)) => Err(cuda_error(
            "CUDA device layer output was already missing".to_string(),
        )),
        (Ok(_), Err(cleanup_err)) => Err(cuda_error(format!(
            "CUDA device layer output cleanup failed: {cleanup_err}"
        ))),
        (Err(err), Ok(_)) => Err(cuda_error(err)),
        (Err(err), Err(cleanup_err)) => {
            Err(cuda_error(format!("{err}; cleanup failed: {cleanup_err}")))
        }
    }
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn download_nemotron_device_layer_output_row(
    output: NemotronDeviceLayerOutput,
    row: usize,
) -> crate::error::Result<Vec<f32>> {
    let download = cuda_runtime::download_device_tensor_f32_row(output.output_id, row);
    let cleanup = cuda_runtime::release_device_tensor(output.output_id);
    match (download, cleanup) {
        (Ok(values), Ok(true)) => Ok(values),
        (Ok(_), Ok(false)) => Err(cuda_error(
            "CUDA device layer output was already missing".to_string(),
        )),
        (Ok(_), Err(cleanup_err)) => Err(cuda_error(format!(
            "CUDA device layer output cleanup failed: {cleanup_err}"
        ))),
        (Err(err), Ok(_)) => Err(cuda_error(err)),
        (Err(err), Err(cleanup_err)) => {
            Err(cuda_error(format!("{err}; cleanup failed: {cleanup_err}")))
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
            shared_up,
            shared_down,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            shared_ff,
            n_expert,
            n_ff,
            n_embd,
            token_count,
            input,
            residual,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_decode_enabled() -> bool {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_q5_sparse_decode_enabled();
    }
    #[cfg(not(feature = "cuda"))]
    false
}

#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_q4k_weight_slices(weights: &[&[u8]]) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        cuda_runtime::prewarm_q4k_weight_slices(weights).map_err(cuda_error)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn cuda_cache_snapshot() -> cuda_runtime::CudaCacheSnapshot {
    cuda_runtime::cuda_cache_snapshot()
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_dense_q4_packed_gate_up_weights(
    weights: &ModelWeights,
) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        let requests = collect_cuda_product_prewarm_requests(weights);
        execute_cuda_product_prewarm_requests(&requests, CudaProductPrewarmSelection::Q4Dense)?;
    }
    Ok(())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_prefill_q4_f32_projection_weights(
    _weights: &ModelWeights,
) -> crate::error::Result<()> {
    Ok(())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_dense_q6_packed_down_weights(
    weights: &ModelWeights,
) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        let requests = collect_cuda_product_prewarm_requests(weights);
        execute_cuda_product_prewarm_requests(&requests, CudaProductPrewarmSelection::Q6Dense)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn push_q4_raw_candidate<'a>(
    requests: &mut CudaProductPrewarmRequests<'a>,
    seen: &mut std::collections::HashSet<(usize, usize)>,
    weight: &'a QuantizedWeight,
) {
    let Some(view) = weight.backend_view() else {
        return;
    };
    if view.quant() != QuantFormat::Q4K {
        return;
    }
    let raw = view.raw();
    if raw.is_empty() {
        return;
    }
    if seen.insert((raw.as_ptr() as usize, raw.len())) {
        requests.q4_raw.push(raw);
    }
}

#[cfg(feature = "cuda")]
fn push_dense_product_packed_candidates<'a>(
    requests: &mut CudaProductPrewarmRequests<'a>,
    gate: &'a QuantizedWeight,
    up: &'a QuantizedWeight,
    down: &'a QuantizedWeight,
) {
    if down.ggml_type == GGMLType::Q6_K {
        if let Some(down_raw) = down.data.as_bytes() {
            requests.q6_down.push((down_raw, down.rows, down.cols));
        }
    }

    if gate.ggml_type != GGMLType::Q4_K || up.ggml_type != GGMLType::Q4_K {
        return;
    }
    if gate.rows != up.rows || gate.cols != up.cols {
        return;
    }
    let (Some(gate_raw), Some(up_raw)) = (gate.data.as_bytes(), up.data.as_bytes()) else {
        return;
    };

    requests
        .q4_gate_up
        .push((gate_raw, up_raw, gate.rows, gate.cols));
    if down.ggml_type == GGMLType::Q4_K {
        if let Some(down_raw) = down.data.as_bytes() {
            requests.q4_single.push((down_raw, down.rows, down.cols));
        }
    }
}

#[cfg(feature = "cuda")]
fn collect_cuda_product_prewarm_requests(weights: &ModelWeights) -> CudaProductPrewarmRequests<'_> {
    let mut requests = CudaProductPrewarmRequests::default();
    let mut seen_q4_raw = std::collections::HashSet::new();
    for layer in &weights.layers {
        match layer {
            LayerType::Attention(layer)
                if layer.moe.is_none() && layer.shared_expert_moe.is_none() =>
            {
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.o_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_gate_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_up_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_down_weight);
                push_dense_product_packed_candidates(
                    &mut requests,
                    &layer.ffn_gate_weight,
                    &layer.ffn_up_weight,
                    &layer.ffn_down_weight,
                );
            }
            LayerType::GatedDeltaNet(layer) if layer.shared_expert_moe.is_none() => {
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.qkv_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.gate_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ssm_out);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_gate_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_up_weight);
                push_q4_raw_candidate(&mut requests, &mut seen_q4_raw, &layer.ffn_down_weight);
                push_dense_product_packed_candidates(
                    &mut requests,
                    &layer.ffn_gate_weight,
                    &layer.ffn_up_weight,
                    &layer.ffn_down_weight,
                );
            }
            _ => {}
        }
    }
    requests
}

#[cfg(feature = "cuda")]
fn execute_cuda_product_prewarm_requests(
    requests: &CudaProductPrewarmRequests<'_>,
    selection: CudaProductPrewarmSelection,
) -> crate::error::Result<()> {
    match selection {
        CudaProductPrewarmSelection::Q4Dense => {
            cuda_runtime::prewarm_q4k_packed_gate_up_weights(&requests.q4_gate_up)
                .map_err(cuda_error)?;
            cuda_runtime::prewarm_q4k_packed_weights(&requests.q4_single).map_err(cuda_error)?;
            let raw_warmed = cuda_runtime::prewarm_quant_resident_q4k_weights(&requests.q4_raw)
                .map_err(cuda_error)?;
            if raw_warmed > 0 {
                eprintln!("[INFO] CUDA Q4_K raw quant resident weights prewarmed: {raw_warmed}");
            }
        }
        CudaProductPrewarmSelection::Q6Dense => {
            cuda_runtime::prewarm_q6k_packed_weights(&requests.q6_down).map_err(cuda_error)?;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum CudaProductPrewarmRequestKindForTest {
    Q4PackedGateUp,
    Q4PackedSingle,
    Q4RawQuant,
    Q6PackedDown,
}

#[cfg(all(test, feature = "cuda"))]
impl From<CudaProductPrewarmRequestKind> for CudaProductPrewarmRequestKindForTest {
    fn from(kind: CudaProductPrewarmRequestKind) -> Self {
        match kind {
            CudaProductPrewarmRequestKind::Q4PackedGateUp => Self::Q4PackedGateUp,
            CudaProductPrewarmRequestKind::Q4PackedSingle => Self::Q4PackedSingle,
            CudaProductPrewarmRequestKind::Q4RawQuant => Self::Q4RawQuant,
            CudaProductPrewarmRequestKind::Q6PackedDown => Self::Q6PackedDown,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(in crate::engine) fn cuda_product_prewarm_request_kinds_for_test(
    weights: &ModelWeights,
) -> Vec<CudaProductPrewarmRequestKindForTest> {
    collect_cuda_product_prewarm_requests(weights)
        .kinds()
        .into_iter()
        .map(Into::into)
        .collect()
}

#[cfg(all(test, feature = "cuda"))]
pub(in crate::engine) fn cuda_product_prewarm_q4_raw_count_for_test(
    weights: &ModelWeights,
) -> usize {
    collect_cuda_product_prewarm_requests(weights).q4_raw.len()
}

#[cfg(all(test, feature = "cuda"))]
pub(in crate::engine) fn cuda_product_prewarm_quant_resident_executor_missing_for_test() -> bool {
    let source = include_str!("cuda_basic.rs");
    let Some(body) = cuda_product_prewarm_function_body_for_test(
        source,
        "execute_cuda_product_prewarm_requests",
    ) else {
        return true;
    };
    !body.contains("prewarm_quant_resident_q4k_weights")
}

#[cfg(all(test, feature = "cuda"))]
pub(in crate::engine) fn cuda_product_prewarm_forbidden_expanded_calls_for_test(
) -> Vec<&'static str> {
    let source = include_str!("cuda_basic.rs");
    let product_functions = [
        "prewarm_dense_q4_packed_gate_up_weights",
        "prewarm_prefill_q4_f32_projection_weights",
        "prewarm_dense_q6_packed_down_weights",
        "execute_cuda_product_prewarm_requests",
    ];
    let forbidden_calls = [
        "prewarm_q4k_f16",
        "prewarm_q4k_f32",
        "prewarm_q4k_prefill_f16",
        "prewarm_q4k_prefill_f32",
        "prewarm_q6k_f16",
        "prewarm_q6k_f32",
    ];

    let mut hits = Vec::new();
    for function in product_functions {
        let Some(body) = cuda_product_prewarm_function_body_for_test(source, function) else {
            hits.push(function);
            continue;
        };
        for call in forbidden_calls {
            if body.contains(call) && !hits.contains(&call) {
                hits.push(call);
            }
        }
    }
    hits
}

#[cfg(all(test, feature = "cuda"))]
pub(in crate::engine) fn cuda_product_prewarm_wrapper_executor_violations_for_test(
) -> Vec<&'static str> {
    let source = include_str!("cuda_basic.rs");
    let contracts = [
        (
            "prewarm_dense_q4_packed_gate_up_weights",
            "execute_cuda_product_prewarm_requests",
            "CudaProductPrewarmSelection::Q4Dense",
        ),
        (
            "prewarm_dense_q6_packed_down_weights",
            "execute_cuda_product_prewarm_requests",
            "CudaProductPrewarmSelection::Q6Dense",
        ),
    ];

    let mut violations = Vec::new();
    for (function, executor, selection) in contracts {
        let Some(body) = cuda_product_prewarm_function_body_for_test(source, function) else {
            violations.push(function);
            continue;
        };
        if !body.contains(executor) || !body.contains(selection) {
            violations.push(function);
        }
    }
    violations
}

#[cfg(all(test, feature = "cuda"))]
fn cuda_product_prewarm_function_body_for_test<'a>(
    source: &'a str,
    function: &str,
) -> Option<&'a str> {
    let signature = format!("fn {function}");
    let fn_pos = source.find(&signature)?;
    let body_start = source[fn_pos..].find('{')? + fn_pos;
    let mut depth = 0usize;
    for (offset, ch) in source[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let body_end = body_start + offset + ch.len_utf8();
                    return Some(&source[body_start..body_end]);
                }
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_decode_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_q5_sparse_decode_moe(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            shared_ff,
            n_ff,
            n_embd,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
            shared_up,
            shared_down,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            shared_ff,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_q8_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_q8_sparse_relu_sqr_by_token(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_prefill_sparse_copy_prefetch(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    n_ff: usize,
    n_embd: usize,
    down_q8: bool,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_prefill_sparse_copy_prefetch(
            up_weights,
            down_weights,
            n_ff,
            n_embd,
            down_q8,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(not(feature = "cuda"), test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_sparse_relu_sqr_full_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        if token_count <= 1 {
            return Ok(None);
        }
        if cuda_runtime::nemotron_q5_layer_cache_enabled() {
            if let Some(output) = cuda_runtime::nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
                up_all,
                down_all,
                expert_ids,
                route_weights,
                token_ids,
                token_count,
                n_expert,
                n_ff,
                n_embd,
                input,
            )
            .map_err(cuda_error)?
            {
                return Ok(Some(output));
            }
        }
        if !cuda_runtime::nemotron_q5_full_layer_enabled() {
            return Ok(None);
        }
        return cuda_runtime::nemotron_q5_sparse_relu_sqr_full_layer_by_token(
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
        .map(Some)
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_register_layer(up_all, down_all, n_expert, n_ff, n_embd)
            .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn nemotron_q5_q8_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_q5_q8_register_layer(
            up_all, down_all, n_expert, n_ff, n_embd,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn try_delta_step_if_supported(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Option<std::result::Result<Vec<f32>, String>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::try_delta_step_resident_if_supported(
            state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim,
        );
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn try_delta_restore_step_if_supported(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Option<std::result::Result<(), String>> {
    #[cfg(feature = "cuda")]
    {
        return try_delta_step_if_supported(
            state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim,
        )
        .map(|result| result.map(drop));
    }
    #[cfg(not(feature = "cuda"))]
    {
        crate::engine::cpu_runtime::kernels::delta_net::delta_net_step_state_only(
            k, v, gate, beta, state, num_heads, head_k_dim, head_v_dim,
        );
        Some(Ok(()))
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod delta_restore_tests {
    use super::*;

    #[test]
    fn cpu_delta_output_step_contract_stays_unsupported() {
        let mut state = vec![0.25, -0.5, 0.75, 1.0];
        let initial_state = state.clone();
        let q = [0.5, -0.25];
        let k = [0.2, 0.8];
        let v = [1.5, -0.5];
        let gate = [-0.1];
        let beta = [0.6];

        let result = try_delta_step_if_supported(&mut state, &q, &k, &v, &gate, &beta, 1, 2, 2);

        assert!(result.is_none());
        assert_eq!(state, initial_state);
    }

    #[test]
    fn cpu_delta_restore_step_matches_one_token_scan_state() {
        let q = [0.5, -0.25];
        let k = [0.2, 0.8];
        let v = [1.5, -0.5];
        let gate = [-0.1];
        let beta = [0.6];
        let mut expected_state = vec![0.25, -0.5, 0.75, 1.0];
        let mut output = [0.0; 2];
        crate::engine::cpu_runtime::kernels::delta_net::delta_net_scan_into(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut expected_state,
            &mut output,
            1,
            1,
            2,
            2,
        );
        let mut restored_state = vec![0.25, -0.5, 0.75, 1.0];

        try_delta_restore_step_if_supported(&mut restored_state, &q, &k, &v, &gate, &beta, 1, 2, 2)
            .expect("CPU delta restore must be supported")
            .expect("CPU delta restore must succeed");

        assert_eq!(restored_state, expected_state);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod device_prefill_trace_tests {
    use super::*;

    #[test]
    fn device_prefill_trace_counts_download_probe_bytes() {
        let bytes = nemotron_device_prefill_trace_bytes(128, 64, Some(32)).expect("trace bytes");

        assert_eq!(bytes.h2d_bytes, 192);
        assert_eq!(bytes.d2h_bytes, 32);
    }

    #[test]
    fn device_prefill_trace_counts_handoff_as_zero_d2h() {
        let bytes = nemotron_device_prefill_trace_bytes(128, 64, None).expect("trace bytes");

        assert_eq!(bytes.h2d_bytes, 192);
        assert_eq!(bytes.d2h_bytes, 0);
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod metal_decode_policy_facade_tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_decode_env() {
        for key in [
            "RNB_METAL_DECODE_CHAIN",
            "RNB_METAL_GDN_LAYER",
            "RNB_METAL_ATTN_LAYER",
            "RNB_METAL_QWEN35_MOE_DECODE_CHAIN",
            "RNB_METAL_KV_INT8",
            "RNB_METAL_DECODE_PARITY_TIME",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn metal_decode_policy_facade_reads_runtime_policy() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_decode_env();
        assert!(metal_decode_legacy_carrier_enabled_by_policy());
        assert!(metal_decode_legacy_attn_layer_enabled_by_policy());
        assert!(metal_qwen_moe_decode_chain_enabled_by_policy());
        assert_eq!(
            metal_decode_kv_int8_requires_carrier_error(true, true),
            None
        );

        std::env::set_var("RNB_METAL_DECODE_CHAIN", "0");
        std::env::set_var("RNB_METAL_GDN_LAYER", "0");
        std::env::set_var("RNB_METAL_ATTN_LAYER", "0");
        std::env::set_var("RNB_METAL_KV_INT8", "1");
        std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", "0");

        assert!(!metal_decode_legacy_carrier_enabled_by_policy());
        assert!(!metal_decode_legacy_attn_layer_enabled_by_policy());
        assert!(!metal_qwen_moe_decode_chain_enabled_by_policy());
        assert_eq!(
            metal_decode_kv_int8_requires_carrier_error(false, false),
            Some("RNB_METAL_KV_INT8=1 requires Metal carrier chain and attention layer")
        );
        std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", "1");
        assert!(metal_qwen_moe_decode_chain_enabled_by_policy());
        clear_decode_env();
    }

    #[test]
    fn chain_layer_input_borrows_attention_and_gdn_state() {
        let prior_k = [1, 2];
        let prior_v = [3, 4];
        let conv_state = [5.0];
        let delta_state = [6.0];
        let attention = ChainLayerInput::Attn {
            prior_k: &prior_k,
            prior_v: &prior_v,
            kvarn: None,
        };
        let gdn = ChainLayerInput::Gdn {
            conv_state: &conv_state,
            delta_state: &delta_state,
        };

        match attention {
            ChainLayerInput::Attn {
                prior_k, prior_v, ..
            } => {
                assert_eq!(prior_k, [1, 2]);
                assert_eq!(prior_v, [3, 4]);
            }
            ChainLayerInput::Gdn { .. } => panic!("expected borrowed attention input"),
        }
        match gdn {
            ChainLayerInput::Gdn {
                conv_state,
                delta_state,
            } => {
                assert_eq!(conv_state, [5.0]);
                assert_eq!(delta_state, [6.0]);
            }
            ChainLayerInput::Attn { .. } => panic!("expected borrowed GDN input"),
        }
    }

    #[test]
    fn metal_decode_parity_counter_facade_is_callable() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_decode_env();
        std::env::set_var("RNB_METAL_DECODE_PARITY_TIME", "1");

        metal_decode_parity_counters_reset();
        metal_decode_parity_record_expected_token();
        metal_decode_parity_counters_report("facade-test");

        clear_decode_env();
    }
}
