//! cu46: drafter forward 의 CUDA port (Phase 1 = FFN).
//!
//! mc78 wiring infra (host CPU drafter) 의 production speedup. drafter forward
//! 의 매 layer FFN 위치에 env opt-in (`RNB_MTP_DRAFTER_CUDA=1`) + cuda feature
//! 활성 시 GPU kernel 호출. host fallback 은 `Ok(false)` 시 caller 가 사용.
//!
//! Phase 1 step 2 (cu46) = `rnb_runtime::cuda_inference::cuda::dense_q4k_gelu_ffn`
//! 호출. Phase 2 (cu47+) = attention CUDA port.

use super::types::TensorView;

/// Phase 1 의 drafter FFN GPU forward.
///
/// `Ok(true)` 반환 = GPU 가 처리 완료, `output` 에 결과 write.
/// `Ok(false)` 반환 = caller 가 host fallback 호출 (cuda feature 비활성 또는
/// weight quant 호환 안 됨).
/// `Err(_)` 반환 = GPU 호출 자체 실패 (panic 안 함, host fallback 권장).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_ffn_cuda(
    gate_weight: &TensorView,
    up_weight: &TensorView,
    down_weight: &TensorView,
    input: &[f32],
    output: &mut [f32],
    n_embd: usize,
    n_ff: usize,
) -> Result<bool, String> {
    use rnb_loader::GGMLType;

    // dense_q4k_gelu_ffn 의 hot path = gate/up Q4_K + down Q4_K 또는 Q6_K.
    // 다른 quant (F32/F16 등) 은 host fallback.
    if gate_weight.ggml_type != GGMLType::Q4_K || up_weight.ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    if !matches!(down_weight.ggml_type, GGMLType::Q4_K | GGMLType::Q6_K) {
        return Ok(false);
    }
    if input.len() < n_embd || output.len() < n_embd {
        return Err(format!(
            "drafter_ffn_cuda buffer size mismatch: input={} output={} n_embd={n_embd}",
            input.len(),
            output.len()
        ));
    }

    // drafter 는 SwiGLU (silu) activation. seq_len=1 의 batch variant 호출.
    let result = rnb_runtime::cuda_inference::cuda::dense_q4k_silu_ffn_batch(
        gate_weight.as_bytes(),
        up_weight.as_bytes(),
        down_weight.as_bytes(),
        down_weight.ggml_type,
        n_ff,
        n_embd,
        1,
        &input[..n_embd],
    )?;
    if result.len() < n_embd {
        return Err(format!(
            "drafter_ffn_cuda output length mismatch: got {}, expected {n_embd}",
            result.len()
        ));
    }
    output[..n_embd].copy_from_slice(&result[..n_embd]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_ffn_cuda(
    _gate_weight: &TensorView,
    _up_weight: &TensorView,
    _down_weight: &TensorView,
    _input: &[f32],
    _output: &mut [f32],
    _n_embd: usize,
    _n_ff: usize,
) -> Result<bool, String> {
    Ok(false)
}

/// cu47 Phase 1 step 1: drafter weight 의 device cache prewarm.
///
/// drafter attach 시점 (mc78 `Engine::attach_external_drafter`) 에서 호출.
/// 각 layer 의 Q4_K weight (gate/up/down + attention q/o + cross_attn 의 일부)
/// 를 device 에 prewarm. 매 forward call 의 weight reupload 제거.
///
/// `Ok(n)` = device cache 에 등록된 weight 수. `Err(_)` 시 caller 가 host
/// path 만 사용 (drafter forward 가 자동 fallback).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn drafter_prewarm_weights_cuda(
    layers: &[super::types::DrafterLayer],
) -> Result<usize, String> {
    drafter_prewarm_weights_cuda_full(layers, None, None)
}

/// cu61: drafter prewarm 의 full variant — layer weight + 글로벌 weight
/// (pre_projection / post_projection 등) 의 합산 prewarm.
#[cfg(feature = "cuda")]
pub fn drafter_prewarm_weights_cuda_full(
    layers: &[super::types::DrafterLayer],
    pre_projection: Option<&TensorView>,
    post_projection: Option<&TensorView>,
) -> Result<usize, String> {
    use rnb_loader::GGMLType;

    let mut q4k_slices: Vec<&[u8]> = Vec::new();
    for layer in layers {
        for tensor in [&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down] {
            if tensor.ggml_type == GGMLType::Q4_K {
                q4k_slices.push(tensor.as_bytes());
            }
        }
        for tensor in [&layer.attn_q, &layer.attn_output] {
            if tensor.ggml_type == GGMLType::Q4_K {
                q4k_slices.push(tensor.as_bytes());
            }
        }
    }
    for tensor in [pre_projection, post_projection].into_iter().flatten() {
        if tensor.ggml_type == GGMLType::Q4_K {
            q4k_slices.push(tensor.as_bytes());
        }
    }
    if q4k_slices.is_empty() {
        return Ok(0);
    }
    rnb_runtime::cuda_inference::cuda::prewarm_q4k_weight_slices(&q4k_slices)
        .map_err(|e| format!("drafter prewarm Q4_K failed: {e}"))
}

#[cfg(not(feature = "cuda"))]
pub fn drafter_prewarm_weights_cuda_full(
    _layers: &[super::types::DrafterLayer],
    _pre_projection: Option<&TensorView>,
    _post_projection: Option<&TensorView>,
) -> Result<usize, String> {
    Ok(0)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub fn drafter_prewarm_weights_cuda(
    _layers: &[super::types::DrafterLayer],
) -> Result<usize, String> {
    Ok(0)
}

/// cu59 Phase 3 step 1: drafter q_proj cuda port.
///
/// drafter attention forward 의 q_proj GEMV (Q4_K × hidden=256 → q_dim) 의
/// cuda 호출. drafter_ffn_cuda 와 같은 패턴 — env opt-in + Q4_K 만 + cuda
/// feature gate.
///
/// `Ok(true)` = cuda 처리, output 채워짐.
/// `Ok(false)` = caller host fallback.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_attn_q_cuda(
    attn_q_weight: &TensorView,
    hidden: &[f32],
    output: &mut [f32],
    q_dim: usize,
    hidden_size: usize,
) -> Result<bool, String> {
    use rnb_loader::GGMLType;

    if attn_q_weight.ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    if hidden.len() < hidden_size || output.len() < q_dim {
        return Err(format!(
            "drafter_attn_q_cuda size mismatch: hidden={} output={} q_dim={q_dim} hidden_size={hidden_size}",
            hidden.len(), output.len()
        ));
    }

    let result = rnb_runtime::cuda_inference::cuda::q4k_gemv(
        attn_q_weight.as_bytes(),
        q_dim,
        hidden_size,
        &hidden[..hidden_size],
    )?;
    if result.len() < q_dim {
        return Err(format!(
            "drafter_attn_q_cuda output len {} < q_dim {}",
            result.len(),
            q_dim
        ));
    }
    output[..q_dim].copy_from_slice(&result[..q_dim]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_attn_q_cuda(
    _attn_q_weight: &TensorView,
    _hidden: &[f32],
    _output: &mut [f32],
    _q_dim: usize,
    _hidden_size: usize,
) -> Result<bool, String> {
    Ok(false)
}

/// cu60: drafter o_proj cuda port. attn_output GEMV (Q4_K × q_dim → hidden).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_attn_o_cuda(
    attn_output_weight: &TensorView,
    attn_out: &[f32],
    output: &mut [f32],
    hidden_size: usize,
    q_dim: usize,
) -> Result<bool, String> {
    use rnb_loader::GGMLType;
    if attn_output_weight.ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    if attn_out.len() < q_dim || output.len() < hidden_size {
        return Err(format!(
            "drafter_attn_o_cuda size mismatch: attn_out={} output={} q_dim={q_dim} hidden_size={hidden_size}",
            attn_out.len(), output.len()
        ));
    }
    let result = rnb_runtime::cuda_inference::cuda::q4k_gemv(
        attn_output_weight.as_bytes(),
        hidden_size,
        q_dim,
        &attn_out[..q_dim],
    )?;
    if result.len() < hidden_size {
        return Err(format!(
            "drafter_attn_o_cuda output len {} < hidden_size {}",
            result.len(),
            hidden_size
        ));
    }
    output[..hidden_size].copy_from_slice(&result[..hidden_size]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_attn_o_cuda(
    _attn_output_weight: &TensorView,
    _attn_out: &[f32],
    _output: &mut [f32],
    _hidden_size: usize,
    _q_dim: usize,
) -> Result<bool, String> {
    Ok(false)
}

/// cu62: drafter pre_projection / post_projection 의 cuda port (q4k_gemv).
/// 매 token decode 의 forward entry/exit 의 single matvec.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_projection_cuda(
    projection_weight: &TensorView,
    input: &[f32],
    output: &mut [f32],
    out_rows: usize,
    in_cols: usize,
) -> Result<bool, String> {
    use rnb_loader::GGMLType;
    if projection_weight.ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    if input.len() < in_cols || output.len() < out_rows {
        return Err(format!(
            "drafter_projection_cuda size mismatch: input={} output={} out_rows={out_rows} in_cols={in_cols}",
            input.len(), output.len()
        ));
    }
    let result = rnb_runtime::cuda_inference::cuda::q4k_gemv(
        projection_weight.as_bytes(),
        out_rows,
        in_cols,
        &input[..in_cols],
    )?;
    if result.len() < out_rows {
        return Err(format!(
            "drafter_projection_cuda result len {} < out_rows {}",
            result.len(),
            out_rows
        ));
    }
    output[..out_rows].copy_from_slice(&result[..out_rows]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_projection_cuda(
    _projection_weight: &TensorView,
    _input: &[f32],
    _output: &mut [f32],
    _out_rows: usize,
    _in_cols: usize,
) -> Result<bool, String> {
    Ok(false)
}

/// cu63: drafter cross_attention_gqa cuda port. cu29 의 decode_attention
/// _hd256_if_supported reuse. drafter 의 K/V (host f32) → f16 변환 후 호출.
///
/// scale = 1.0 (drafter spec §4 verbatim, q_norm 가 magnitude 정규화).
///
/// `Ok(true)` = cuda 처리. `Ok(false)` = caller host fallback.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_cross_attention_cuda(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<bool, String> {
    if seq_len == 0 {
        out.fill(0.0);
        return Ok(true);
    }
    // cu29 의 decode_attention_hd256_if_supported 가 hd=256 만. drafter
    // 의 head_dim 가 256 아니면 host fallback.
    if head_dim != 256 {
        return Ok(false);
    }
    // K/V 의 host f32 → host u16 (f16 bits) 변환. small overhead but matches
    // cu29 signature.
    let kv_len_kv = n_kv_heads * seq_len * head_dim;
    if k.len() < kv_len_kv || v.len() < kv_len_kv {
        return Err(format!(
            "drafter_cross_attention_cuda K/V len mismatch: k={} v={} expected={}",
            k.len(),
            v.len(),
            kv_len_kv
        ));
    }
    let k_f16: Vec<u16> = k[..kv_len_kv]
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    let v_f16: Vec<u16> = v[..kv_len_kv]
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();

    // cu63: cuda decode_attention_hd256 signature = (q, k, v, kv_len,
    // num_heads, num_kv_heads, scale). head_dim 인자 없음 (hd=256 hardcoded).
    let result = rnb_runtime::compute::attention_decode_hd256(
        q, &k_f16, &v_f16, seq_len, n_heads, n_kv_heads,
        1.0f32, // drafter scale (spec §4 verbatim, q_norm 가 magnitude)
    )?;
    if result.len() < n_heads * head_dim {
        return Err(format!(
            "drafter_cross_attention_cuda result len {} < {}",
            result.len(),
            n_heads * head_dim
        ));
    }
    out[..n_heads * head_dim].copy_from_slice(&result[..n_heads * head_dim]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_cross_attention_cuda(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _seq_len: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _out: &mut [f32],
) -> Result<bool, String> {
    Ok(false)
}

/// cu66 Phase 4 step 1: drafter forward-scoped K/V f16 conversion cache.
///
/// shared_kv 의 host f32 K/V 를 forward call 의 entry 에 한 번 f16 변환.
/// 매 layer iteration 의 cross_attention call 의 매번 변환 제거.
///
/// cu63 의 saturation 의 main cause (host f32 → f16 변환 overhead) fix 의
/// entry point.
#[allow(dead_code)]
pub(crate) fn drafter_kv_f16_convert(
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (Vec<u16>, Vec<u16>) {
    let kv_len = n_kv_heads * seq_len * head_dim;
    let k_safe_len = kv_len.min(k.len());
    let v_safe_len = kv_len.min(v.len());
    let k_f16: Vec<u16> = k[..k_safe_len]
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    let v_f16: Vec<u16> = v[..v_safe_len]
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    (k_f16, v_f16)
}

/// cu66 Phase 4 step 1: cross_attention with cached f16 K/V (변환 안 함).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_cross_attention_cuda_cached(
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<bool, String> {
    if seq_len == 0 {
        out.fill(0.0);
        return Ok(true);
    }
    if head_dim != 256 {
        return Ok(false);
    }
    let kv_len_kv = n_kv_heads * seq_len * head_dim;
    if k_f16.len() < kv_len_kv || v_f16.len() < kv_len_kv {
        return Err(format!(
            "drafter_cross_attention_cuda_cached K/V f16 len mismatch: k={} v={} expected={}",
            k_f16.len(),
            v_f16.len(),
            kv_len_kv
        ));
    }
    let result = rnb_runtime::compute::attention_decode_hd256(
        q, k_f16, v_f16, seq_len, n_heads, n_kv_heads, 1.0f32,
    )?;
    if result.len() < n_heads * head_dim {
        return Err(format!(
            "drafter_cross_attention_cuda_cached result len {} < {}",
            result.len(),
            n_heads * head_dim
        ));
    }
    out[..n_heads * head_dim].copy_from_slice(&result[..n_heads * head_dim]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_cross_attention_cuda_cached(
    _q: &[f32],
    _k_f16: &[u16],
    _v_f16: &[u16],
    _seq_len: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _out: &mut [f32],
) -> Result<bool, String> {
    Ok(false)
}

/// cu67: drafter vq_head centroids GEMV cuda port (Q4_K × hidden → n_centroids).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn drafter_vq_head_cuda(
    centroids_weight: &TensorView,
    x_norm: &[f32],
    cluster_logits: &mut [f32],
    n_centroids: usize,
    centroid_dim: usize,
) -> Result<bool, String> {
    use rnb_loader::GGMLType;
    if centroids_weight.ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    if x_norm.len() < centroid_dim || cluster_logits.len() < n_centroids {
        return Err(format!(
            "drafter_vq_head_cuda size mismatch: x_norm={} cluster_logits={} n_centroids={n_centroids} centroid_dim={centroid_dim}",
            x_norm.len(), cluster_logits.len()
        ));
    }
    let result = rnb_runtime::cuda_inference::cuda::q4k_gemv(
        centroids_weight.as_bytes(),
        n_centroids,
        centroid_dim,
        &x_norm[..centroid_dim],
    )?;
    if result.len() < n_centroids {
        return Err(format!(
            "drafter_vq_head_cuda result len {} < n_centroids {}",
            result.len(),
            n_centroids
        ));
    }
    cluster_logits[..n_centroids].copy_from_slice(&result[..n_centroids]);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn drafter_vq_head_cuda(
    _centroids_weight: &TensorView,
    _x_norm: &[f32],
    _cluster_logits: &mut [f32],
    _n_centroids: usize,
    _centroid_dim: usize,
) -> Result<bool, String> {
    Ok(false)
}
