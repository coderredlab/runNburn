#[cfg(feature = "vulkan")]
use crate::engine::layer_weights::AttentionLayerWeights;
#[cfg(feature = "vulkan")]
use crate::engine::quantized_weight_types::QuantizedWeight;

#[cfg(feature = "vulkan")]
use super::materialize::{
    append_attention_kv_f32_for_layer, attention_decode_window_grouped_from_mirror_for_layer,
};
#[cfg(feature = "vulkan")]
use crate::engine::cpu_runtime::kernels;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;
#[cfg(feature = "vulkan")]
use rnb_core::tensor::Tensor;

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn quantized_weight_bytes_supported(weight: &QuantizedWeight) -> bool {
    gpu::quantized_bytes_supported(weight.ggml_type, weight.data.as_bytes())
}

/// mv33: q_norm/k_norm hybrid path 활성 env. default OFF — q_norm/k_norm 이
/// 있는 모델 (Qwen3+) 이 자동으로 partial path 에 진입하면 raw mode token
/// sequence 가 CPU strict 와 다르게 흐른다 (Adreno/Mali fp drift, mv31 정책 근거).
/// `RNB_MOBILE_VULKAN_QK_NORM=1` 로만 hybrid path 진입을 허용한다.
#[cfg(feature = "vulkan")]
fn qk_norm_hybrid_enabled() -> bool {
    crate::engine::policy::env_string("RNB_MOBILE_VULKAN_QK_NORM")
        .map(|v| v != "0")
        .unwrap_or(false)
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn attention_window_fast_path_supported(
    weights: &AttentionLayerWeights,
) -> bool {
    // mv33: q_norm/k_norm guard 는 hybrid env 가 켜지면 풀린다. 기본은 그대로
    // false 라 default behavior 회귀 없음.
    let qk_norm_ok =
        (weights.q_norm.is_none() && weights.k_norm.is_none()) || qk_norm_hybrid_enabled();
    qk_norm_ok
        && quantized_weight_bytes_supported(&weights.q_weight)
        && quantized_weight_bytes_supported(&weights.k_weight)
        && quantized_weight_bytes_supported(&weights.v_weight)
        && quantized_weight_bytes_supported(&weights.o_weight)
        && quantized_weight_bytes_supported(&weights.ffn_gate_weight)
        && quantized_weight_bytes_supported(&weights.ffn_up_weight)
        && quantized_weight_bytes_supported(&weights.ffn_down_weight)
        && weights.q_bias.is_none()
        && weights.k_bias.is_none()
        && weights.v_bias.is_none()
}

/// mv33: q_norm/k_norm 이 있는 모델 (Qwen3+) 이 hybrid path 로 진입하는지 검사.
/// 이 케이스는 gpu_executor 의 post_ffw_norm-like path 를 따라가 CPU 에서 q/k
/// rmsnorm 을 적용 후 vulkan attention 으로 dispatch 한다. env 미설정 시
/// `attention_window_fast_path_supported` 가 false 라 진입 자체가 차단되니 이
/// 함수는 hybrid path 진입 후 분기 결정에만 쓰인다.
#[cfg(feature = "vulkan")]
pub(in crate::engine) fn attention_window_needs_qk_norm_path(
    weights: &AttentionLayerWeights,
) -> bool {
    weights.q_norm.is_some() || weights.k_norm.is_some()
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_attention_block_window_for_layer(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    normed_input: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    q_dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_data: &[f32],
    ffn_norm: &[f32],
    norm_eps: f32,
    gemma_needs_post_ffw_norm: bool,
    output: &mut [f32],
) -> Result<bool, String> {
    if gemma_needs_post_ffw_norm {
        return Ok(false);
    }
    let Some(q_quant) = gpu::ggml_to_quant(q_weight.ggml_type) else {
        return Ok(false);
    };
    let Some(k_quant) = gpu::ggml_to_quant(k_weight.ggml_type) else {
        return Ok(false);
    };
    let Some(v_quant) = gpu::ggml_to_quant(v_weight.ggml_type) else {
        return Ok(false);
    };
    let Some(q_raw) = q_weight.data.as_bytes() else {
        return Ok(false);
    };
    let Some(k_raw) = k_weight.data.as_bytes() else {
        return Ok(false);
    };
    let Some(v_raw) = v_weight.data.as_bytes() else {
        return Ok(false);
    };
    let o_quant = gpu::quant_or_error(o_weight.ggml_type, "o_proj")?;
    let gate_quant = gpu::quant_or_error(gate_weight.ggml_type, "gate")?;
    let up_quant = gpu::quant_or_error(up_weight.ggml_type, "up")?;
    let down_quant = gpu::quant_or_error(down_weight.ggml_type, "down")?;
    let o_raw = o_weight
        .data
        .as_bytes()
        .ok_or_else(|| "o_proj weight bytes missing".to_string())?;
    let gate_raw = gate_weight
        .data
        .as_bytes()
        .ok_or_else(|| "ffn gate weight bytes missing".to_string())?;
    let up_raw = up_weight
        .data
        .as_bytes()
        .ok_or_else(|| "ffn up weight bytes missing".to_string())?;
    let down_raw = down_weight
        .data
        .as_bytes()
        .ok_or_else(|| "ffn down weight bytes missing".to_string())?;

    gpu::attention_block_window_for_layer(
        runtime,
        layer_idx,
        normed_input,
        q_weight.cols,
        q_raw,
        q_dim,
        q_quant,
        k_raw,
        kv_dim,
        k_quant,
        v_raw,
        kv_dim,
        v_quant,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        o_raw,
        hidden_dim,
        o_weight.cols,
        o_quant,
        hidden_data,
        ffn_norm,
        norm_eps,
        gate_raw,
        gate_weight.rows,
        gate_weight.cols,
        gate_quant,
        up_raw,
        up_weight.rows,
        up_weight.cols,
        up_quant,
        down_raw,
        down_weight.rows,
        down_weight.cols,
        down_quant,
        output,
    )
    .map(|()| true)
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_attention_ffn_window_fast_path(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    hidden_data: &[f32],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
    seq_len: usize,
    q_dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
    output: &mut [f32],
) -> crate::error::Result<bool> {
    runtime
        .rms_norm_window(hidden_data, attn_norm, norm_eps, hidden_dim, output)
        .map_err(|err| crate::error::LlmError::Forward(err.to_string()))?;
    let normed_input = output[..seq_len * hidden_dim].to_vec();
    output.fill(0.0);
    try_attention_block_window_for_layer(
        runtime,
        layer_idx,
        &normed_input,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        q_dim,
        kv_dim,
        hidden_dim,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_data,
        ffn_norm,
        norm_eps,
        false,
        output,
    )
    .map_err(crate::error::LlmError::Forward)
}

/// mv36: hybrid path 의 attention dispatch 만 vulkan, o_proj/FFN 은 caller 가
/// CPU strict 로 처리하도록 분리. mv35 측정에서 single attention layer 자체는
/// 1-ULP 영역 임을 확인 → drift 의 진짜 origin 은 hybrid path 의 vulkan
/// o_proj/FFN 단계로 의심. attention dispatch 만 keep 해 token-identical
/// 시도.
#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn run_attention_window_attention_only_for_layer(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
) -> crate::error::Result<Vec<f32>> {
    let q_dim = num_heads * head_dim;
    for t in 0..seq_len {
        for kvh in 0..num_kv_heads {
            let start = t * kv_dim + kvh * head_dim;
            let end = start + head_dim;
            append_attention_kv_f32_for_layer(
                runtime,
                layer_idx * 100 + kvh,
                pos_start + t,
                &k_all[start..end],
                &v_all[start..end],
            )?;
        }
    }
    // mv39: opt-in CPU strict NEON attention swap. Mali (Lenovo) 에선 mv39 dot
    // 8-acc + 8-step + pairwise tree fix 로 vulkan attention 도 token-identical
    // 도달, 그러나 Adreno (Flip4) 는 vendor driver fp ops pattern 다름 → token
    // diff. universal token-identical 보장 위해 CPU strict swap. perf 손실 (Mali
    // submits 감소) 인정. KV cache prefix 가 있으면 (pos_start>0) vulkan path
    // fallback (KV materialize 복잡, 별도 작업).
    let cpu_attn_swap = pos_start == 0
        && crate::engine::policy::env_string("RNB_VULKAN_PARTIAL_ATTN_CPU")
            .map(|v| v != "0")
            .unwrap_or(false);
    let mut attention_output = vec![0.0f32; seq_len * q_dim];
    if cpu_attn_swap {
        let q_tensor = Tensor::from_slice(q_all, &[seq_len, q_dim]);
        let k_tensor = Tensor::from_slice(k_all, &[seq_len, kv_dim]);
        let v_tensor = Tensor::from_slice(v_all, &[seq_len, kv_dim]);
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        match kernels::attention::attention_with_scale_window_and_softcap(
            &q_tensor,
            &k_tensor,
            &v_tensor,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            None,
        ) {
            Ok(t) => {
                attention_output.copy_from_slice(kernels::tensor_as_f32_slice(&t));
            }
            Err(_) => {
                attention_decode_window_grouped_from_mirror_for_layer(
                    runtime,
                    layer_idx,
                    q_all,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    pos_start,
                    &mut attention_output,
                )?;
            }
        }
    } else {
        attention_decode_window_grouped_from_mirror_for_layer(
            runtime,
            layer_idx,
            q_all,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            pos_start,
            &mut attention_output,
        )?;
    }
    if pos_start == 0 && seq_len > 1 && attn_debug_enabled() {
        debug_compare_first_layer_attention(
            layer_idx,
            q_all,
            k_all,
            v_all,
            &attention_output,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
        );
    }
    Ok(attention_output)
}

/// mv35: vulkan attention intermediate diff infra activation env.
/// `RNB_VULKAN_ATTN_DEBUG=1` 설정 시 layer 0 의 마지막 prefill token 의
/// attention output (head 0) 를 vulkan vs CPU naive scalar f32 ref 비교 후
/// stdout 출력. drift 의 정확한 axis 를 isolate 하기 위한 측정 인프라.
#[cfg(feature = "vulkan")]
fn attn_debug_enabled() -> bool {
    crate::engine::policy::env_string("RNB_VULKAN_ATTN_DEBUG")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// mv35: ULP-distance helper. Returns `i64::MAX` for opposite-sign / NaN cases.
/// Bit-distance (Steele/Goldberg) — 같은 부호 + 같은 magnitude 영역에선 두 fp 값
/// 사이 representable f32 개수와 동일하다.
#[cfg(feature = "vulkan")]
fn ulp_diff_f32(a: f32, b: f32) -> i64 {
    if a.is_nan() || b.is_nan() {
        return i64::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    // Bias-flip negatives so consecutive bit patterns map to monotone integers.
    let map = |x: i32| -> i64 {
        if x < 0 {
            (i32::MIN as i64) - (x as i64)
        } else {
            x as i64
        }
    };
    (map(ai) - map(bi)).abs()
}

/// mv35 Phase 1: vulkan attention 의 layer 0 last-token, head 0 출력을 CPU naive
/// scalar f32 reference 와 elementwise 비교. drift 가 첫 layer 부터 큰지 / 누적인지
/// 확인용. self-attention 단일 prefill chunk (pos_start=0) 가정.
#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
fn debug_compare_first_layer_attention(
    layer_idx: usize,
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    vulkan_out: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
) {
    let q_dim = num_heads * head_dim;
    let group_size = num_heads / num_kv_heads.max(1);
    let last_t = seq_len - 1;
    let h = 0usize;
    let g = h / group_size.max(1);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_off = last_t * q_dim + h * head_dim;
    let q_vec = &q_all[q_off..q_off + head_dim];
    // mv38: NEON 4-lane indep acc + Fma + horizontal sum reference.
    // ARM CPU strict path 의 NEON `vfmaq_f32 + vaddvq_f32` 와 비트 일치 가까움.
    // mv35 의 sequential scalar reference 로는 vulkan 의 4-lane Fma 패턴이 ULP
    // 후퇴로 보였음 (CPU naive scalar baseline mismatch).
    let mut scores = vec![0.0f32; last_t + 1];
    let mut max_score = f32::NEG_INFINITY;
    for i in 0..=last_t {
        let k_off = i * kv_dim + g * head_dim;
        let k_vec = &k_all[k_off..k_off + head_dim];
        let mut acc = [0.0f32; 4];
        let chunks = head_dim / 4;
        for chunk in 0..chunks {
            let base = chunk * 4;
            for lane in 0..4 {
                acc[lane] = q_vec[base + lane].mul_add(k_vec[base + lane], acc[lane]);
            }
        }
        let dot = ((acc[0] + acc[1]) + acc[2]) + acc[3];
        let s = dot * scale;
        scores[i] = s;
        if s > max_score {
            max_score = s;
        }
    }
    // Pass 2 — softmax weighted V sum.
    let mut weights = vec![0.0f32; last_t + 1];
    let mut den = 0.0f32;
    for i in 0..=last_t {
        let w = (scores[i] - max_score).exp();
        weights[i] = w;
        den += w;
    }
    let mut cpu_out = vec![0.0f32; head_dim];
    for i in 0..=last_t {
        let v_off = i * kv_dim + g * head_dim;
        let v_vec = &v_all[v_off..v_off + head_dim];
        let w = weights[i];
        for d in 0..head_dim {
            cpu_out[d] += w * v_vec[d];
        }
    }
    for d in 0..head_dim {
        cpu_out[d] /= den;
    }
    let vk_off = last_t * q_dim + h * head_dim;
    let vk_out = &vulkan_out[vk_off..vk_off + head_dim];
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_ulp: i64 = 0;
    for d in 0..head_dim {
        let abs_d = (vk_out[d] - cpu_out[d]).abs();
        let denom = cpu_out[d].abs().max(1e-30);
        let rel_d = abs_d / denom;
        let ulp_d = ulp_diff_f32(vk_out[d], cpu_out[d]);
        if abs_d > max_abs {
            max_abs = abs_d;
        }
        if rel_d > max_rel {
            max_rel = rel_d;
        }
        if ulp_d > max_ulp {
            max_ulp = ulp_d;
        }
    }
    println!(
        "[mv35:attn_debug] layer={} t={} h=0 group=0 head_dim={} max_score={:.6} den={:.6} max_abs={:.4e} max_rel={:.4e} max_ulp={}",
        layer_idx, last_t, head_dim, max_score, den, max_abs, max_rel, max_ulp
    );
    let preview = head_dim.min(8);
    let vk_str: Vec<String> = (0..preview).map(|d| format!("{:.6}", vk_out[d])).collect();
    let cpu_str: Vec<String> = (0..preview).map(|d| format!("{:.6}", cpu_out[d])).collect();
    println!(
        "[mv35:attn_debug] vulkan[0..{}] = [{}]",
        preview,
        vk_str.join(", ")
    );
    println!(
        "[mv35:attn_debug] cpu_ref[0..{}] = [{}]",
        preview,
        cpu_str.join(", ")
    );
}
