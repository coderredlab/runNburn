//! Prefill attention Q/K/V projection helpers.

use super::super::*;

pub(super) struct PrefillAttentionProjection {
    pub(super) q: Tensor,
    pub(super) k: Option<Tensor>,
    pub(super) v: Option<Tensor>,
    pub(super) attn_gate: Option<Tensor>,
    pub(super) cached_kv_f16: Option<(Vec<u16>, Vec<u16>)>,
}

pub(super) struct PrefillFusedAttention {
    pub(super) attn_out: Tensor,
    pub(super) k_bits: Vec<u16>,
    pub(super) v_bits: Vec<u16>,
}

pub(super) struct PrefillFullAttentionLayer {
    pub(super) hidden: Tensor,
    pub(super) k_bits: Vec<u16>,
    pub(super) v_bits: Vec<u16>,
}

/// pm37: ATN q/k/v/o projection prefill seam. ssm_out(pm36) 과 동일한 generic
/// `gdn_proj` 경로(Q4_K tensorops batch GEMM)를 재사용 — 신규 커널/가드 0. callsite
/// micro-gate `RNB_METAL_PREFILL_ATN_PROJ` — pm37 부터 default ON(27B token-identical +
/// prefill −6.9% 검증 후 metal prefill dense projection GPU 승격), opt-out=`=0`(GDN 과
/// 독립 롤백). 실제 GPU 진입은 runtime seam 의 `RNB_METAL_PREFILL_GDN_INPROJ` 를 공유한다.
/// RoPE/qk_norm/bias/residual 은 gemv 후처리라 seam(gemv 만 대체) 밖에서 그대로 CPU 유지.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine::forward) fn atn_proj_metal(
    role: &'static str,
    layer_idx: usize,
    weight: &QuantizedWeight,
    input: &Tensor,
    seq_len: usize,
) -> crate::error::Result<Option<Tensor>> {
    if crate::engine::policy::env_string("RNB_METAL_PREFILL_ATN_PROJ").as_deref() == Some("0") {
        return Ok(None);
    }
    let slice = kernels::tensor_as_f32_slice(input);
    // K = input feature dim (= weight.cols). N(n_out) 은 runtime wrapper 가 backend
    // view.rows() 로 추론하므로 호출자는 K 만 전달하면 된다.
    let k = slice.len() / seq_len;
    let trace = backend_runtime::MetalProjTrace {
        role,
        layer_idx,
        timing_enabled: backend_runtime::metal_prefill_atn_full_timing_enabled(),
    };
    match backend_runtime::metal_prefill_gdn_proj_into_if_supported_with_trace(
        weight,
        slice,
        seq_len,
        k,
        Some(trace),
    )? {
        Some(out) => {
            let n_out = out.len() / seq_len;
            Ok(Some(Tensor::from_vec(out, &[seq_len, n_out])))
        }
        None => Ok(None),
    }
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(in crate::engine::forward) fn atn_proj_metal(
    _role: &'static str,
    _layer_idx: usize,
    _weight: &QuantizedWeight,
    _input: &Tensor,
    _seq_len: usize,
) -> crate::error::Result<Option<Tensor>> {
    Ok(None)
}

/// pm70: dense Qwen gated attention prefill full-layer Metal carrier.
/// attn core의 gated attention output을 host로 읽지 않고 o_proj+FFN까지 device에서 이어 간다.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_full_layer_metal(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<PrefillFullAttentionLayer>> {
    if crate::engine::policy::env_string("RNB_METAL_PREFILL_ATN_FULL_LAYER_TAIL").as_deref()
        == Some("0")
    {
        return Ok(None);
    }
    if gemma4_reuse_q_only
        || use_gemma_block_semantics(architecture)
        || w.v_proj_missing
        || !layout.has_gated_attn
        || layout.head_dim != 256
        || pos_start != 0
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || w.ffn_gate_up_fused.is_some()
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim == 0 || rope_dim >= layout.head_dim {
        return Ok(None);
    }
    if qwen_text_mrope_dim(metadata, architecture, rope_dim, layout.head_dim).is_some() {
        return Ok(None);
    }
    if gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    )
    .is_some()
    {
        return Ok(None);
    }
    if gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim) {
        return Ok(None);
    }

    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;
    let hidden_dim = kernels::tensor_as_f32_slice(hidden).len() / seq_len;
    let scale = resolve_attention_scale(metadata, architecture);
    let n_rot = rope_dim.min(head_dim);
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);

    let Some(out) = backend_runtime::metal_prefill_atn_full_layer_if_supported(
        kernels::tensor_as_f32_slice(hidden),
        kernels::tensor_as_f32_slice(&w.attn_norm),
        kernels::tensor_as_f32_slice(q_norm),
        kernels::tensor_as_f32_slice(k_norm),
        &w.q_weight,
        &w.k_weight,
        &w.v_weight,
        &w.o_weight,
        kernels::tensor_as_f32_slice(ffn_norm),
        &w.ffn_gate_weight,
        &w.ffn_up_weight,
        &w.ffn_down_weight,
        backend_runtime::MetalPrefillAtnCoreShape {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        },
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(out.hidden.len(), seq_len * hidden_dim, "pm70 hidden len");
    debug_assert_eq!(out.k_bits.len(), seq_len * kv_dim, "pm70 k_bits len");
    debug_assert_eq!(out.v_bits.len(), seq_len * kv_dim, "pm70 v_bits len");
    Ok(Some(PrefillFullAttentionLayer {
        hidden: Tensor::from_vec(out.hidden, &[seq_len, hidden_dim]),
        k_bits: out.k_bits,
        v_bits: out.v_bits,
    }))
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_full_layer_metal(
    _metadata: &ModelMetadata,
    _architecture: ModelArchitecture,
    _hidden: &Tensor,
    _w: &AttentionLayerWeights,
    _rope_freqs: Option<&Tensor>,
    _layout: AttentionLayout,
    _gemma4_reuse_q_only: bool,
    _layer_idx: usize,
    _seq_len: usize,
    _pos_start: usize,
    _norm_eps: f32,
) -> crate::error::Result<Option<PrefillFullAttentionLayer>> {
    Ok(None)
}

/// pm108: dense Qwen gated attention prefill o-tail Metal carrier.
/// ATN core의 gated attention output을 host로 읽지 않고 o_proj+residual까지 device에서 잇고,
/// Qwen MoE FFN은 기존 상위 prefill flow가 계속 처리한다.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_o_tail_metal(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<PrefillFullAttentionLayer>> {
    if !backend_runtime::metal_prefill_atn_o_tail_requested() {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }
    if gemma4_reuse_q_only
        || use_gemma_block_semantics(architecture)
        || w.v_proj_missing
        || !layout.has_gated_attn
        || layout.head_dim != 256
        || pos_start != 0
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || w.moe.is_some()
        || w.shared_expert_moe.is_none()
        || w.ffn_gate_up_fused.is_some()
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
    {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim == 0 || rope_dim >= layout.head_dim {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }
    if qwen_text_mrope_dim(metadata, architecture, rope_dim, layout.head_dim).is_some() {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }
    if gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    )
    .is_some()
    {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }
    if gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim) {
        backend_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    }

    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;
    let hidden_dim = kernels::tensor_as_f32_slice(hidden).len() / seq_len;
    let scale = resolve_attention_scale(metadata, architecture);
    let n_rot = rope_dim.min(head_dim);

    let Some(out) = backend_runtime::metal_prefill_atn_o_tail_if_supported(
        kernels::tensor_as_f32_slice(hidden),
        kernels::tensor_as_f32_slice(&w.attn_norm),
        kernels::tensor_as_f32_slice(q_norm),
        kernels::tensor_as_f32_slice(k_norm),
        &w.q_weight,
        &w.k_weight,
        &w.v_weight,
        &w.o_weight,
        backend_runtime::MetalPrefillAtnCoreShape {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        },
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(out.hidden.len(), seq_len * hidden_dim, "pm108 hidden len");
    debug_assert_eq!(out.k_bits.len(), seq_len * kv_dim, "pm108 k_bits len");
    debug_assert_eq!(out.v_bits.len(), seq_len * kv_dim, "pm108 v_bits len");
    Ok(Some(PrefillFullAttentionLayer {
        hidden: Tensor::from_vec(out.hidden, &[seq_len, hidden_dim]),
        k_bits: out.k_bits,
        v_bits: out.v_bits,
    }))
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_o_tail_metal(
    _metadata: &ModelMetadata,
    _architecture: ModelArchitecture,
    _hidden: &Tensor,
    _w: &AttentionLayerWeights,
    _rope_freqs: Option<&Tensor>,
    _layout: AttentionLayout,
    _gemma4_reuse_q_only: bool,
    _layer_idx: usize,
    _seq_len: usize,
    _pos_start: usize,
    _norm_eps: f32,
) -> crate::error::Result<Option<PrefillFullAttentionLayer>> {
    Ok(None)
}

/// pm50 M1: dense Qwen gated attention prefill core carrier.
/// 기존 `PrefillFusedAttention` seam과 같이 gated `attn_out` + f16 KV bits를 반환한다.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_core_metal(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<PrefillFusedAttention>> {
    if gemma4_reuse_q_only
        || w.v_proj_missing
        || !layout.has_gated_attn
        || layout.head_dim != 256
        || pos_start != 0
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || use_gemma_block_semantics(architecture)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim == 0 || rope_dim >= layout.head_dim {
        return Ok(None);
    }
    if qwen_text_mrope_dim(metadata, architecture, rope_dim, layout.head_dim).is_some() {
        return Ok(None);
    }
    if gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    )
    .is_some()
    {
        return Ok(None);
    }
    if gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim) {
        return Ok(None);
    }

    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;
    let hidden_dim = kernels::tensor_as_f32_slice(hidden).len() / seq_len;
    let scale = resolve_attention_scale(metadata, architecture);
    let n_rot = rope_dim.min(head_dim);

    backend_runtime::metal_prefill_atn_full_expected_dense_layer();
    let Some(out) = backend_runtime::metal_prefill_atn_core_if_supported(
        kernels::tensor_as_f32_slice(hidden),
        kernels::tensor_as_f32_slice(&w.attn_norm),
        kernels::tensor_as_f32_slice(q_norm),
        kernels::tensor_as_f32_slice(k_norm),
        &w.q_weight,
        &w.k_weight,
        &w.v_weight,
        backend_runtime::MetalPrefillAtnCoreShape {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        },
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(out.attn_out.len(), seq_len * q_dim, "pm50 attn_out len");
    debug_assert_eq!(out.k_bits.len(), seq_len * kv_dim, "pm50 k_bits len");
    debug_assert_eq!(out.v_bits.len(), seq_len * kv_dim, "pm50 v_bits len");
    Ok(Some(PrefillFusedAttention {
        attn_out: Tensor::from_vec(out.attn_out, &[seq_len, q_dim]),
        k_bits: out.k_bits,
        v_bits: out.v_bits,
    }))
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_atn_core_metal(
    _metadata: &ModelMetadata,
    _architecture: ModelArchitecture,
    _hidden: &Tensor,
    _w: &AttentionLayerWeights,
    _rope_freqs: Option<&Tensor>,
    _layout: AttentionLayout,
    _gemma4_reuse_q_only: bool,
    _layer_idx: usize,
    _seq_len: usize,
    _pos_start: usize,
    _norm_eps: f32,
) -> crate::error::Result<Option<PrefillFusedAttention>> {
    Ok(None)
}

/// pm48 ②: dense Qwen prefill attention 2차 device-resident chain 진입 시도.
/// attn_norm → q/k/v proj(GPU) → gate split → (device chain: rope/qk_norm(q,k) → cast → flash)
/// → gate apply(sigmoid) 을 묶는다. RoPE/qk_norm/flash 가 전부 chain 안(단일 command buffer)이라
/// 1차(host 입출력)의 layer 당 CPU rope/qk_norm + q 재upload 가 제거된다. gate `RNB_METAL_PREFILL_ATTN_CHAIN`.
///
/// 진입 가드(dense Qwen 전용): gated attn(q_out=2*q_dim) + q_norm/k_norm 존재 + head_dim==256 +
/// rope_partial(adjacent-pair, n_rot<head_dim, non-iMRoPE, non-proportional) + sliding window/softcap
/// 없음 + pos_start==0 + bias 없음 + dump/trace 비활성. 미충족 시 None(기존 CPU 경로).
/// 반환 `(attn_out[gate 적용], k_f16, v_f16)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_attn_chain_metal(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<PrefillFusedAttention>> {
    // pm48 ②: default ON(27B/9B token-identical + 27B ABAB ≥10%). opt-out=`=0`.
    if crate::engine::policy::env_string("RNB_METAL_PREFILL_ATTN_CHAIN").as_deref() == Some("0") {
        return Ok(None);
    }
    // dense Qwen gated attention 전용 가드.
    if gemma4_reuse_q_only
        || w.v_proj_missing
        || !layout.has_gated_attn
        || layout.head_dim != 256
        || pos_start != 0
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || use_gemma_block_semantics(architecture)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    // rope: rope_partial(adjacent-pair, n_rot<head_dim, non-iMRoPE, non-proportional, no freq_factors).
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim == 0 || rope_dim >= layout.head_dim {
        return Ok(None);
    }
    if qwen_text_mrope_dim(metadata, architecture, rope_dim, layout.head_dim).is_some() {
        return Ok(None); // iMRoPE(split-half) 는 이 chain 의 adjacent-pair 커널과 불일치
    }
    if gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    )
    .is_some()
    {
        return Ok(None);
    }
    // v 후처리(norm/rotation) 가 필요한 layer 는 chain 미지원(v 를 그대로 f16 cast).
    if gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim) {
        return Ok(None);
    }

    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;

    let timing_enabled = backend_runtime::metal_prefill_atn_full_timing_enabled();
    let t_norm0 = std::time::Instant::now();
    let normed = apply_model_norm(hidden, &w.attn_norm, norm_eps, architecture).map_err(fwd)?;
    if timing_enabled {
        eprintln!(
            "[prefill-atn-stage-time] layer={layer_idx} stage=attn_norm_cpu m={seq_len} ms={:.3}",
            t_norm0.elapsed().as_secs_f64() * 1000.0
        );
    }

    // q/k/v projection(GPU seam 우선, host fallback). q_full 은 gated [query|gate] 인터리브.
    let q_full =
        if let Some(t) = atn_proj_metal("q_proj", layer_idx, &w.q_weight, &normed, seq_len)? {
            t
        } else {
            w.q_weight.gemv(&normed)?
        };
    let k = if let Some(t) = atn_proj_metal("k_proj", layer_idx, &w.k_weight, &normed, seq_len)? {
        t
    } else {
        w.k_weight.gemv(&normed)?
    };
    let v = if let Some(t) = atn_proj_metal("v_proj", layer_idx, &w.v_weight, &normed, seq_len)? {
        t
    } else {
        w.v_weight.gemv(&normed)?
    };

    // gated split: q_full [seq, q_dim*2] → query [seq, q_dim] + gate [seq, q_dim] (head 별 인터리브).
    let q_data = kernels::tensor_as_f32_slice(&q_full);
    let q_out_dim = q_full.shape().last().copied().unwrap_or(0);
    let mut q_vec = vec![0.0f32; seq_len * q_dim];
    let mut gate_vec = vec![0.0f32; seq_len * q_dim];
    let t_split0 = std::time::Instant::now();
    for t in 0..seq_len {
        for h in 0..num_heads {
            let src_off = t * q_out_dim + h * head_dim * 2;
            let dst = t * q_dim + h * head_dim;
            q_vec[dst..dst + head_dim].copy_from_slice(&q_data[src_off..src_off + head_dim]);
            gate_vec[dst..dst + head_dim]
                .copy_from_slice(&q_data[src_off + head_dim..src_off + head_dim * 2]);
        }
    }
    if timing_enabled {
        eprintln!(
            "[prefill-atn-stage-time] layer={layer_idx} stage=q_gate_split_cpu m={seq_len} ms={:.3}",
            t_split0.elapsed().as_secs_f64() * 1000.0
        );
    }

    let k_data = kernels::tensor_as_f32_slice(&k).to_vec();
    let v_data = kernels::tensor_as_f32_slice(&v).to_vec();
    let q_norm_data = kernels::tensor_as_f32_slice(q_norm);
    let k_norm_data = kernels::tensor_as_f32_slice(k_norm);
    let scale = resolve_attention_scale(metadata, architecture);
    let n_rot = rope_dim.min(head_dim);

    let Some((mut attn_out, k_bits, v_bits)) =
        backend_runtime::metal_prefill_attn_chain_if_supported(
            &q_vec,
            &k_data,
            &v_data,
            q_norm_data,
            k_norm_data,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            n_rot,
            rope_theta,
            norm_eps,
            pos_start,
            scale,
            false,
            false,
            layer_idx,
            timing_enabled,
        )
    else {
        return Ok(None);
    };
    debug_assert_eq!(attn_out.len(), seq_len * q_dim, "attn_out len");
    debug_assert_eq!(k_bits.len(), seq_len * kv_dim, "k_bits len");
    debug_assert_eq!(v_bits.len(), seq_len * kv_dim, "v_bits len");

    let t_gate0 = std::time::Instant::now();
    sigmoid_mul_f32_inplace(&mut attn_out, &gate_vec);
    if timing_enabled {
        eprintln!(
            "[prefill-atn-stage-time] layer={layer_idx} stage=gate_sigmoid m={seq_len} ms={:.3}",
            t_gate0.elapsed().as_secs_f64() * 1000.0
        );
    }

    Ok(Some(PrefillFusedAttention {
        attn_out: Tensor::from_vec(attn_out, &[seq_len, q_dim]),
        k_bits,
        v_bits,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_attention_q4k_f16_qkv_attention_hd512(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<PrefillFusedAttention>> {
    if !matches!(architecture, ModelArchitecture::Gemma4)
        || !super::super::policy::q4k_fused_prefill_attention_requested()
        || gemma4_reuse_q_only
        || w.v_proj_missing
        || layout.has_gated_attn
        || layout.head_dim != 512
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim != layout.head_dim {
        return Ok(None);
    }
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let normed = if use_gemma_block_semantics(architecture)
        && policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
    {
        apply_model_norm_unit_offset(hidden, &w.attn_norm, norm_eps).map_err(fwd)?
    } else {
        apply_model_norm(hidden, &w.attn_norm, norm_eps, architecture).map_err(fwd)?
    };
    let q_norm_data = kernels::tensor_as_f32_slice(q_norm);
    let k_norm_data = kernels::tensor_as_f32_slice(k_norm);
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    let qk_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let Some((attn_out, k_bits, v_bits)) =
        backend_runtime::prefill_attention_q4k_f16_qkv_attention_hd512_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            kernels::tensor_as_f32_slice(&normed),
            q_norm_data,
            k_norm_data,
            freq_factors,
            seq_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            qk_unit_offset,
            qk_unit_offset,
            use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
        )?
    else {
        return Ok(None);
    };

    Ok(Some(PrefillFusedAttention {
        attn_out: Tensor::from_vec(attn_out, &[seq_len, layout.q_dim]),
        k_bits,
        v_bits,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_prefill_attention(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<PrefillAttentionProjection> {
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let kv_dim = layout.kv_dim;

    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "layer_in",
            kernels::tensor_as_f32_slice(hidden),
        );
    }
    let normed = if use_gemma_block_semantics(architecture)
        && policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
    {
        apply_model_norm_unit_offset(hidden, &w.attn_norm, norm_eps).map_err(fwd)?
    } else {
        apply_model_norm(hidden, &w.attn_norm, norm_eps, architecture).map_err(fwd)?
    };
    // mv38 D: baseline hidden + normed (layer 3, t=last 첫 8 elem).
    if crate::engine::policy::env_string("RNB_VULKAN_LAYER_TRACE")
        .map(|v| v != "0")
        .unwrap_or(false)
        && layer_idx == 3
    {
        let hd = kernels::tensor_as_f32_slice(hidden);
        let nd = kernels::tensor_as_f32_slice(&normed);
        let last_off = (seq_len - 1) * metadata.hidden_dim;
        let preview_n = metadata.hidden_dim.min(8);
        let h_preview: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", hd[last_off + d]))
            .collect();
        let n_preview: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", nd[last_off + d]))
            .collect();
        eprintln!(
            "[mv38:hidden_in] baseline layer={} hidden[0..{}] = [{}]",
            layer_idx,
            preview_n,
            h_preview.join(", ")
        );
        eprintln!(
            "[mv38:normed] baseline layer={} normed[0..{}] = [{}]",
            layer_idx,
            preview_n,
            n_preview.join(", ")
        );
    }
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "attn_norm",
            kernels::tensor_as_f32_slice(&normed),
        );
    }
    if layer_idx == 0 && attn_trace_enabled() {
        let normed_data = kernels::tensor_as_f32_slice(&normed);
        let first_n = normed_data.len().min(metadata.hidden_dim).min(8);
        eprintln!(
            "[attn-trace][prefill-pre-qkv] layer={} token0_attn_norm first{}={:?}",
            layer_idx,
            first_n,
            &normed_data[..first_n]
        );
        if seq_len > 1 {
            let start = metadata.hidden_dim;
            let end = start + first_n.min(metadata.hidden_dim);
            eprintln!(
                "[attn-trace][prefill-pre-qkv] layer={} token1_attn_norm first{}={:?}",
                layer_idx,
                first_n,
                &normed_data[start..end]
            );
        }
    }

    // cu30 Phase 2c-2: Llama / Mistral hd=128 fused QKV + GPU RoPE + f16 K/V
    // pack prefill path. opt-in via RNB_CUDA_HD128_FUSED_QKV_ROPE. forward.rs
    // 의 cached_kv_f16 path 가 자동으로 apply_prefill_rope + host f32→f16
    // 변환 + KvCache append 모두 처리.
    if !gemma4_reuse_q_only
        && !w.v_proj_missing
        && layout.head_dim == 128
        && !layout.has_gated_attn
        && w.q_norm.is_none()
        && w.k_norm.is_none()
        && w.q_bias.is_none()
        && w.k_bias.is_none()
        && w.v_bias.is_none()
        && dump_bin_dir().is_none()
        && !attn_trace_enabled()
        && !targeted_attn_trace_enabled(layer_idx)
    {
        let (rope_dim, rope_theta, proportional_rope) =
            resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
        let freq_factors = gemma_rope_freq_factors(
            rope_freqs,
            metadata,
            architecture,
            layer_idx,
            layout.head_dim,
        );
        if !proportional_rope && rope_dim == layout.head_dim && freq_factors.is_none() {
            if let Some((q_vec, k_bits, v_bits)) =
                backend_runtime::dense_q4k_attention_qkv_rope_hd128_prefill_if_supported(
                    &w.q_weight,
                    &w.k_weight,
                    &w.v_weight,
                    layout.num_heads,
                    layout.num_kv_heads,
                    rope_theta,
                    pos_start,
                    seq_len,
                    kernels::tensor_as_f32_slice(&normed),
                )?
            {
                return Ok(PrefillAttentionProjection {
                    q: Tensor::from_vec(q_vec, &[seq_len, layout.q_dim]),
                    k: None,
                    v: None,
                    attn_gate: None,
                    cached_kv_f16: Some((k_bits, v_bits)),
                });
            }
        }
    }

    if matches!(architecture, ModelArchitecture::Gemma4)
        && !gemma4_reuse_q_only
        && !w.v_proj_missing
        && layout.head_dim == 256
        && !layout.has_gated_attn
        && active_sliding_window(metadata, architecture, layer_idx).is_some()
        && super::super::policy::gemma_neox_rope_enabled()
        && !super::super::policy::gemma_qk_norm_disabled()
        && dump_bin_dir().is_none()
        && !attn_trace_enabled()
        && !targeted_attn_trace_enabled(layer_idx)
        && w.q_bias.is_none()
        && w.k_bias.is_none()
        && w.v_bias.is_none()
        && !gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        && !gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
    {
        if let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) {
            let (rope_dim, rope_theta, proportional_rope) =
                resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
            let freq_factors = gemma_rope_freq_factors(
                rope_freqs,
                metadata,
                architecture,
                layer_idx,
                layout.head_dim,
            );
            if !proportional_rope && rope_dim == layout.head_dim && freq_factors.is_none() {
                let qk_unit_offset = use_gemma_block_semantics(architecture)
                    && super::super::policy::gemma_unit_offset_norm_enabled();
                let q_norm_data = kernels::tensor_as_f32_slice(q_norm);
                let k_norm_data = kernels::tensor_as_f32_slice(k_norm);
                if let Some((q_vec, k_bits, v_bits)) =
                    backend_runtime::prefill_attention_q4k_f16_qkv_postprocess_hd256_if_supported(
                        &w.q_weight,
                        &w.k_weight,
                        &w.v_weight,
                        kernels::tensor_as_f32_slice(&normed),
                        q_norm_data,
                        k_norm_data,
                        seq_len,
                        layout.num_heads,
                        layout.num_kv_heads,
                        rope_theta,
                        pos_start,
                        norm_eps,
                        qk_unit_offset,
                        qk_unit_offset,
                        use_gemma_block_semantics(architecture)
                            && super::super::policy::gemma_v_norm_enabled(),
                    )?
                {
                    return Ok(PrefillAttentionProjection {
                        q: Tensor::from_vec(q_vec, &[seq_len, layout.q_dim]),
                        k: None,
                        v: None,
                        attn_gate: None,
                        cached_kv_f16: Some((k_bits, v_bits)),
                    });
                }
            }
        }
    }

    let fused_qkv = if !gemma4_reuse_q_only && !w.v_proj_missing {
        backend_runtime::prefill_attention_q4k_f16_qkv_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            kernels::tensor_as_f32_slice(&normed),
            seq_len,
        )?
    } else {
        None
    };
    let (mut q_full, mut k, mut v) = if let Some((q_vec, k_vec, v_vec)) = fused_qkv {
        (
            Tensor::from_vec(q_vec, &[seq_len, w.q_weight.rows]),
            Some(Tensor::from_vec(k_vec, &[seq_len, w.k_weight.rows])),
            Some(Tensor::from_vec(v_vec, &[seq_len, w.v_weight.rows])),
        )
    } else {
        // mt94: Q/K projection mixed-precision ablation. When
        // `RNB_QK_F32_OVERRIDE=1` is set, q_proj and k_proj GEMV use per-row
        // `dequantize_bytes_to_f32` + pure f32×f32 dot, bypassing both the
        // production NEON Q4×Q8K integer kernel and the `quant_gemv` scalar
        // quant-aware fallback. V/O/MLP/output keep production dispatch.
        let qk_f32_override = crate::engine::policy::env_string("RNB_QK_F32_OVERRIDE")
            .map(|v| v != "0")
            .unwrap_or(false);
        let q_full = if qk_f32_override {
            w.q_weight.gemv_full_dequant_f32(&normed)?
        } else if let Some(t) = atn_proj_metal("q_proj", layer_idx, &w.q_weight, &normed, seq_len)?
        {
            t
        } else {
            w.q_weight.gemv(&normed)?
        };
        let k = if gemma4_reuse_q_only {
            None
        } else if qk_f32_override {
            Some(w.k_weight.gemv_full_dequant_f32(&normed)?)
        } else if let Some(t) = atn_proj_metal("k_proj", layer_idx, &w.k_weight, &normed, seq_len)?
        {
            Some(t)
        } else {
            Some(w.k_weight.gemv(&normed)?)
        };
        let v = if gemma4_reuse_q_only {
            None
        } else if w.v_proj_missing {
            // Gemma4 full-attn layers may omit V projection, matching K reuse semantics.
            k.clone()
        } else if let Some(t) = atn_proj_metal("v_proj", layer_idx, &w.v_weight, &normed, seq_len)?
        {
            Some(t)
        } else {
            Some(w.v_weight.gemv(&normed)?)
        };
        (q_full, k, v)
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "q_proj",
            kernels::tensor_as_f32_slice(&q_full),
        );
    }
    if let Some(ref k_tensor) = k {
        if dump_bin_dir().is_some() {
            dump_bin(
                "prefill",
                layer_idx,
                "k_proj",
                kernels::tensor_as_f32_slice(k_tensor),
            );
        }
    }
    if let Some(ref v_tensor) = v {
        if dump_bin_dir().is_some() {
            dump_bin(
                "prefill",
                layer_idx,
                "v_proj",
                kernels::tensor_as_f32_slice(v_tensor),
            );
        }
    }

    if let Some(bias) = &w.q_bias {
        q_full = add_tensors(&q_full, bias).map_err(fwd)?;
    }
    if let (Some(k_tensor), Some(bias)) = (&mut k, &w.k_bias) {
        *k_tensor = add_tensors(k_tensor, bias).map_err(fwd)?;
    }
    if let (Some(v_tensor), Some(bias)) = (&mut v, &w.v_bias) {
        *v_tensor = add_tensors(v_tensor, bias).map_err(fwd)?;
    }

    let q_out_dim = q_full.shape().last().copied().unwrap_or(0);
    let (q, attn_gate) = if layout.has_gated_attn {
        let q_data = kernels::tensor_as_f32_slice(&q_full);
        let half_dim = q_out_dim / 2;
        let mut q_vec = vec![0.0f32; seq_len * half_dim];
        let mut gate_vec = vec![0.0f32; seq_len * half_dim];
        for t in 0..seq_len {
            for h in 0..num_heads {
                let src_off = t * q_out_dim + h * head_dim * 2;
                let dst_off = t * half_dim + h * head_dim;
                q_vec[dst_off..dst_off + head_dim]
                    .copy_from_slice(&q_data[src_off..src_off + head_dim]);
                gate_vec[dst_off..dst_off + head_dim]
                    .copy_from_slice(&q_data[src_off + head_dim..src_off + head_dim * 2]);
            }
        }
        (
            Tensor::from_vec(q_vec, &[seq_len, half_dim]),
            Some(Tensor::from_vec(gate_vec, &[seq_len, half_dim])),
        )
    } else {
        (q_full, None)
    };

    // mv38 D / mv39: baseline q before q_norm (after gemv). sum/max_abs 강화.
    if crate::engine::policy::env_string("RNB_VULKAN_LAYER_TRACE")
        .map(|v| v != "0")
        .unwrap_or(false)
        && layer_idx == 3
    {
        let q_data_pre = kernels::tensor_as_f32_slice(&q);
        let q_dim_actual = q_data_pre.len() / seq_len;
        let q_last_off = (seq_len - 1) * q_dim_actual;
        let preview_n = q_dim_actual.min(8);
        let q_t0: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data_pre[d]))
            .collect();
        let q_last: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data_pre[q_last_off + d]))
            .collect();
        let q_sum: f64 = q_data_pre.iter().map(|&v| v as f64).sum();
        let q_max_abs = q_data_pre.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let q_xor: u64 = q_data_pre
            .iter()
            .map(|&v| v.to_bits() as u64)
            .reduce(|a, b| a ^ b)
            .unwrap_or(0);
        eprintln!(
            "[mv39:q_pre_norm] baseline layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
            layer_idx, preview_n, q_t0.join(", "), preview_n, q_last.join(", "),
            q_sum, q_max_abs, q_xor, q_data_pre.len(),
        );
    }
    let q = if let Some(q_norm) = &w.q_norm {
        let q_data = kernels::tensor_as_f32_slice(&q);
        let total_heads = seq_len * num_heads;
        let q_3d = Tensor::from_slice(q_data, &[total_heads, head_dim]);
        let normed = apply_model_qk_norm(&q_3d, q_norm, norm_eps, architecture).map_err(fwd)?;
        let normed_data = kernels::tensor_as_f32_slice(&normed);
        Tensor::from_slice(normed_data, &[seq_len, layout.q_dim])
    } else {
        q
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "q_normed",
            kernels::tensor_as_f32_slice(&q),
        );
    }

    let k = if let Some(k_tensor) = k {
        if let Some(k_norm) = &w.k_norm {
            let k_data = kernels::tensor_as_f32_slice(&k_tensor);
            let total_heads = seq_len * num_kv_heads;
            let k_3d = Tensor::from_slice(k_data, &[total_heads, head_dim]);
            let normed = apply_model_qk_norm(&k_3d, k_norm, norm_eps, architecture).map_err(fwd)?;
            let normed_data = kernels::tensor_as_f32_slice(&normed);
            Some(Tensor::from_slice(normed_data, &[seq_len, kv_dim]))
        } else {
            Some(k_tensor)
        }
    } else {
        None
    };
    if let Some(ref k_tensor) = k {
        if dump_bin_dir().is_some() {
            dump_bin(
                "prefill",
                layer_idx,
                "k_normed",
                kernels::tensor_as_f32_slice(k_tensor),
            );
        }
    }

    Ok(PrefillAttentionProjection {
        q,
        k,
        v,
        attn_gate,
        cached_kv_f16: None,
    })
}
