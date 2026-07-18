//! Prefill attention compute helpers.

use super::super::*;

// pm48 진단용 임시 계측: prefill attention compute(QK^T/softmax/AV) CPU 시간.
// `RNB_DIAG_ATTN_COMPUTE_TIME=1` 일 때만 호출당 누적 ms 를 stderr 로 출력.
static ATTN_COMPUTE_DIAG_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ATTN_COMPUTE_DIAG_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct AttnComputeDiagTimer(std::time::Instant);
impl Drop for AttnComputeDiagTimer {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        let ns = self.0.elapsed().as_nanos() as u64;
        let total = ATTN_COMPUTE_DIAG_NS.fetch_add(ns, Ordering::Relaxed) + ns;
        let calls = ATTN_COMPUTE_DIAG_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "[prefill-attn-compute] call#{calls} this={:.3}ms cumulative={:.1}ms",
            ns as f64 / 1e6,
            total as f64 / 1e6
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_prefill_attention(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    w: &AttentionLayerWeights,
    layout: AttentionLayout,
    q: &Tensor,
    cached_k_tensor: Option<&Tensor>,
    cached_v_tensor: Option<&Tensor>,
    cached_kv_f16: Option<(&[u16], &[u16])>,
    attn_gate: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    kv_len: usize,
) -> crate::error::Result<Tensor> {
    let _attn_diag = if std::env::var("RNB_DIAG_ATTN_COMPUTE_TIME").as_deref() == Ok("1") {
        Some(AttnComputeDiagTimer(std::time::Instant::now()))
    } else {
        None
    };
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let kv_dim = layout.kv_dim;
    let force_tokenwise = force_tokenwise_prefill_attn_layer(layer_idx);
    let sliding_window = active_sliding_window(metadata, architecture, layer_idx);
    let has_sliding_window = sliding_window.is_some();
    let has_softcap = resolve_attention_softcap(architecture).is_some();
    let f16_attention_out = if !force_tokenwise {
        if let Some((cached_k_f16, cached_v_f16)) = cached_kv_f16 {
            if has_sliding_window {
                backend_runtime::prefill_attention_f16kv_window_if_supported(
                    kernels::tensor_as_f32_slice(q),
                    cached_k_f16,
                    cached_v_f16,
                    seq_len,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    resolve_attention_scale(metadata, architecture),
                    sliding_window,
                    has_softcap,
                )?
            } else {
                backend_runtime::prefill_attention_f16kv_if_supported(
                    kernels::tensor_as_f32_slice(q),
                    cached_k_f16,
                    cached_v_f16,
                    seq_len,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    resolve_attention_scale(metadata, architecture),
                    has_sliding_window,
                    has_softcap,
                )?
            }
        } else {
            None
        }
    } else {
        None
    };

    let needs_f32_kv = force_tokenwise || f16_attention_out.is_none();
    let cached_f32_storage =
        if needs_f32_kv && !(cached_k_tensor.is_some() && cached_v_tensor.is_some()) {
            if let Some((cached_k_f16, cached_v_f16)) = cached_kv_f16 {
                let cached_k_f32 = cached_k_f16
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect::<Vec<_>>();
                let cached_v_f32 = cached_v_f16
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect::<Vec<_>>();
                Some((
                    Tensor::from_vec(cached_k_f32, &[kv_len, kv_dim]),
                    Tensor::from_vec(cached_v_f32, &[kv_len, kv_dim]),
                ))
            } else {
                None
            }
        } else {
            None
        };
    let (cached_k_tensor, cached_v_tensor) = if needs_f32_kv {
        if let (Some(k_tensor), Some(v_tensor)) = (cached_k_tensor, cached_v_tensor) {
            (Some(k_tensor), Some(v_tensor))
        } else if let Some((cached_k_f32, cached_v_f32)) = cached_f32_storage.as_ref() {
            (Some(cached_k_f32), Some(cached_v_f32))
        } else {
            return Err(crate::error::LlmError::Forward(
                "prefill attention missing cached K/V tensors".to_string(),
            ));
        }
    } else {
        (None, None)
    };

    // mc73 prefill SIMD axis unification.
    //
    // `RNB_ATTN_F16_NEON=1` enables a native fp16 SIMD prefill path. Two
    // implementations live behind it:
    //
    //   1. **Batch BR×BC tile (mc73 default for SIMD ON)** — calls
    //      `kernels::attention::attention_batch_f16`. Keeps the f32 batch path's
    //      cache locality (BR=32 query rows × BC=32 KV cols) while running Q×K,
    //      P@V, online-softmax rescale on the same native fp16 SIMD ops as the
    //      decode `process_head_f16_acc` (vmulq_f16 / vfmaq_f16 /
    //      neon_vec_dot_f16_f16). KV is read straight from `cached_kv_f16`
    //      (zero-copy); Q is rounded once per call. acc/row_max/row_sum stay in
    //      f32 (GGML F16_VEC attention pattern).
    //   2. **Token-wise dispatch (mc72 axis-isolation, opt-in via
    //      `RNB_ATTN_F16_PREFILL_TOKENWISE=1`)** — runs `attention_decode_flash`
    //      per token. Matches the vulkan attention path
    //      (`run_attention_window_post_ffw_norm_path`): per-token KV append +
    //      decode kernel reuse. Useful for verifying CPU↔Vulkan axis
    //      consistency, but O(seq_len²) cumulative KV traverse on CPU-only.
    //
    // The SIMD path requires aarch64 + FEAT_FP16. On other targets the
    // default falls back to the mc71 fp16-fallback batch path below.
    // pm48: Apple Silicon(macOS) default ON. M5 Pro 측정상 batch f16 prefill 이
    // f32 대비 attention compute −38%(1102→685ms) / 전체 prefill −5.1%, raw
    // token-identical(5tok 한국어 + 1139tok 中/英) + decode 무영향 + chat 정상.
    // mobile ARM(Android)은 mc73에서 decode −25% / prefill tie 라 default OFF 유지
    // (fp16 SIMD 유닛 성능 차이). 미설정 시 macOS=ON / 그 외=OFF, env 로 override.
    let f16_neon_prefill = std::env::var("RNB_ATTN_F16_NEON")
        .map(|v| v != "0")
        .unwrap_or(cfg!(target_os = "macos"));
    let f16_prefill_tokenwise = std::env::var("RNB_ATTN_F16_PREFILL_TOKENWISE")
        .map(|v| v != "0")
        .unwrap_or(false);
    #[cfg(target_arch = "aarch64")]
    let has_fp16 = std::arch::is_aarch64_feature_detected!("fp16");
    #[cfg(not(target_arch = "aarch64"))]
    let has_fp16 = false;
    // pm48 ①: f16kv seam(예: Metal flash attention prefill GPU)이 성공하면 그 결과를 우선한다.
    // CPU↔GPU attention 은 상호배타 경로 — flash Some 이면 f16 NEON CPU(simd_prefill) 를 건너뛰어
    // 불필요한 f16 KV materialize 도 회피한다. flash None(non-M5/gate OFF/shape 미충족)이면
    // 기존 f16 NEON CPU batch path 로 fallback.
    let simd_prefill_active = f16_neon_prefill && has_fp16 && f16_attention_out.is_none();
    // mc73: materialize K/V as f16 stored bits — either from cached_kv_f16
    // directly (zero-copy) or by converting cached_k_tensor/v_tensor (one-time
    // f32→f16 alloc per layer per prefill).
    let kv_f16_owned: Option<(Vec<u16>, Vec<u16>)> = if simd_prefill_active {
        if cached_kv_f16.is_some() {
            None
        } else if let (Some(k_t), Some(v_t)) = (cached_k_tensor, cached_v_tensor) {
            let k_data = kernels::tensor_as_f32_slice(k_t);
            let v_data = kernels::tensor_as_f32_slice(v_t);
            let k_v: Vec<u16> = k_data
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            let v_v: Vec<u16> = v_data
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            Some((k_v, v_v))
        } else {
            None
        }
    } else {
        None
    };
    let kv_f16_for_simd: Option<(&[u16], &[u16])> = if simd_prefill_active {
        if let Some((k, v)) = cached_kv_f16 {
            Some((k, v))
        } else {
            kv_f16_owned
                .as_ref()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
        }
    } else {
        None
    };
    let mut attn_out = if let Some((k_f16, v_f16)) = kv_f16_for_simd {
        let q_data = kernels::tensor_as_f32_slice(q);
        let scale = resolve_attention_scale(metadata, architecture);
        let softcap = resolve_attention_softcap(architecture);
        if f16_prefill_tokenwise {
            // mc72 axis-isolation path: per-token attention_decode_flash dispatch.
            let mut out = vec![0.0f32; seq_len * layout.q_dim];
            for t in 0..seq_len {
                let kv_prefix_len = pos_start + t + 1;
                let q_off = t * layout.q_dim;
                let q_token = &q_data[q_off..q_off + layout.q_dim];
                let k_prefix = &k_f16[..kv_prefix_len * kv_dim];
                let v_prefix = &v_f16[..kv_prefix_len * kv_dim];
                let out_slice = &mut out[q_off..q_off + layout.q_dim];
                kernels::attention::attention_decode_flash(
                    q_token,
                    k_prefix,
                    v_prefix,
                    out_slice,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_prefix_len,
                    scale,
                    sliding_window,
                    softcap,
                );
            }
            Tensor::from_vec(out, &[seq_len, layout.q_dim])
        } else {
            // mc73 batch BR×BC f16 path (default for SIMD ON).
            #[cfg(target_arch = "aarch64")]
            {
                let out = kernels::attention::attention_batch_f16(
                    q_data,
                    k_f16,
                    v_f16,
                    seq_len,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    sliding_window,
                    softcap,
                );
                Tensor::from_vec(out, &[seq_len, layout.q_dim])
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("simd_prefill_active requires aarch64");
            }
        }
    } else if force_tokenwise {
        let cached_k_tensor = cached_k_tensor.expect("cached K tensor materialized");
        let cached_v_tensor = cached_v_tensor.expect("cached V tensor materialized");
        let q_data = kernels::tensor_as_f32_slice(q);
        let k_data = kernels::tensor_as_f32_slice(cached_k_tensor);
        let v_data = kernels::tensor_as_f32_slice(cached_v_tensor);
        let mut out = vec![0.0f32; seq_len * layout.q_dim];
        let scale = resolve_attention_scale(metadata, architecture);
        let softcap = resolve_attention_softcap(architecture);
        for t in 0..seq_len {
            let kv_prefix_len = pos_start + t + 1;
            let q_token = Tensor::from_slice(
                &q_data[t * layout.q_dim..(t + 1) * layout.q_dim],
                &[1, layout.q_dim],
            );
            let k_prefix =
                Tensor::from_slice(&k_data[..kv_prefix_len * kv_dim], &[kv_prefix_len, kv_dim]);
            let v_prefix =
                Tensor::from_slice(&v_data[..kv_prefix_len * kv_dim], &[kv_prefix_len, kv_dim]);
            let token_out = kernels::attention::attention_with_scale_window_and_softcap(
                &q_token,
                &k_prefix,
                &v_prefix,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                sliding_window,
                softcap,
            )
            .map_err(fwd)?;
            out[t * layout.q_dim..(t + 1) * layout.q_dim]
                .copy_from_slice(kernels::tensor_as_f32_slice(&token_out));
        }
        Tensor::from_vec(out, &[seq_len, layout.q_dim])
    } else if let Some(out) = f16_attention_out {
        Tensor::from_vec(out, &[seq_len, layout.q_dim])
    } else if let Some(out) = backend_runtime::prefill_attention_hd256_if_supported(
        kernels::tensor_as_f32_slice(q),
        kernels::tensor_as_f32_slice(cached_k_tensor.expect("cached K tensor materialized")),
        kernels::tensor_as_f32_slice(cached_v_tensor.expect("cached V tensor materialized")),
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        resolve_attention_scale(metadata, architecture),
        has_sliding_window,
        has_softcap,
    )? {
        Tensor::from_vec(out, &[seq_len, layout.q_dim])
    } else {
        kernels::attention::attention_with_scale_window_and_softcap(
            q,
            cached_k_tensor.expect("cached K tensor materialized"),
            cached_v_tensor.expect("cached V tensor materialized"),
            num_heads,
            num_kv_heads,
            head_dim,
            resolve_attention_scale(metadata, architecture),
            active_sliding_window(metadata, architecture, layer_idx),
            resolve_attention_softcap(architecture),
        )
        .map_err(fwd)?
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "attn_out",
            kernels::tensor_as_f32_slice(&attn_out),
        );
        // mt93 step1: dump reference attention softmax_input / softmax_output
        // for the last query token. This is observation-only — the live
        // attention compute path is untouched. The reference computation
        // mirrors `eager_attention_forward` (PyTorch reference): QK^T * scale,
        // then optional softcap (tanh), then causal mask (+ optional sliding
        // window), then softmax over kv axis. Output shape is
        // `[num_heads, kv_len]` (last query token only), matching the
        // mt90 Python last-row dump pattern.
        if super::super::policy::dump_bin_layer_enabled(layer_idx) {
            dump_reference_attn_softmax(
                metadata,
                architecture,
                layout,
                q,
                cached_k_tensor,
                cached_v_tensor,
                cached_kv_f16,
                kv_f16_for_simd,
                layer_idx,
                seq_len,
                kv_len,
                pos_start,
                sliding_window,
            );
        }
    }
    if layer_idx == 0 && attn_trace_enabled() {
        let attn_data = kernels::tensor_as_f32_slice(&attn_out);
        let attn_last = &attn_data[(seq_len - 1) * layout.q_dim..seq_len * layout.q_dim];
        emit_vec_trace("prefill", layer_idx, "attn_out", attn_last);
    }
    if targeted_attn_trace_enabled(layer_idx) {
        let attn_data = kernels::tensor_as_f32_slice(&attn_out);
        let attn_last = &attn_data[(seq_len - 1) * layout.q_dim..seq_len * layout.q_dim];
        emit_vec_trace("prefill-l34", layer_idx, "attn_out", attn_last);
    }

    if let Some(gate) = attn_gate {
        let gate_sig = kernels::activation::sigmoid(gate).map_err(fwd)?;
        attn_out = kernels::elementwise::mul(&attn_out, &gate_sig).map_err(fwd)?;
    }

    if gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, head_dim) {
        let mut rotated = kernels::tensor_as_f32_slice(&attn_out).to_vec();
        gemma4_apply_attn_rot_inplace(&mut rotated, head_dim, layout.q_dim, 64);
        attn_out = Tensor::from_vec(rotated, attn_out.shape());
    }

    Ok(attn_out)
}

/// mt93 step1 — observation-only reference softmax dump.
///
/// Reconstructs `softmax_input` (Q·K^T·scale + softcap + causal/sliding-window
/// mask) and `softmax_output` (softmax over kv axis) for the **last query
/// token** of the prefill batch, then writes them as flat f32 buffers via
/// `dump_bin`. Shape is `[num_q_heads, kv_len]` (heads-major). This mirrors
/// the PyTorch eager attention path so the dump can be compared 1:1 with
/// `layer_NNN_attn_softmax_input.bin` / `_output.bin` produced by the
/// `mt93_attention_softmax_dump` Python hook.
#[allow(clippy::too_many_arguments)]
fn dump_reference_attn_softmax(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layout: AttentionLayout,
    q: &Tensor,
    cached_k_tensor: Option<&Tensor>,
    cached_v_tensor: Option<&Tensor>,
    cached_kv_f16: Option<(&[u16], &[u16])>,
    kv_f16_for_simd: Option<(&[u16], &[u16])>,
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    pos_start: usize,
    sliding_window: Option<usize>,
) {
    let num_q_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let kv_dim = layout.kv_dim;
    let scale = resolve_attention_scale(metadata, architecture);
    let softcap = resolve_attention_softcap(architecture);

    if seq_len == 0 || kv_len == 0 || head_dim == 0 || num_q_heads == 0 || num_kv_heads == 0 {
        return;
    }
    let groups = num_q_heads / num_kv_heads.max(1);

    // Materialize K and V as contiguous f32 of shape [kv_len, kv_dim].
    let (k_f32, v_f32): (Vec<f32>, Vec<f32>) =
        if let (Some(k_t), Some(v_t)) = (cached_k_tensor, cached_v_tensor) {
            (
                kernels::tensor_as_f32_slice(k_t).to_vec(),
                kernels::tensor_as_f32_slice(v_t).to_vec(),
            )
        } else if let Some((k_bits, v_bits)) = cached_kv_f16.or(kv_f16_for_simd) {
            let kf: Vec<f32> = k_bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let vf: Vec<f32> = v_bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            (kf, vf)
        } else {
            // No K/V available — can't dump reference. Skip silently.
            return;
        };

    if k_f32.len() < kv_len * kv_dim || v_f32.len() < kv_len * kv_dim {
        return;
    }

    let q_data = kernels::tensor_as_f32_slice(q);
    let last_t = seq_len - 1;
    let pos_q = pos_start + last_t; // absolute query position in kv axis
    let q_row = &q_data[last_t * layout.q_dim..(last_t + 1) * layout.q_dim];

    let mut softmax_input = vec![0.0f32; num_q_heads * kv_len];
    let mut softmax_output = vec![0.0f32; num_q_heads * kv_len];

    for h in 0..num_q_heads {
        let kv_h = h / groups.max(1);
        let q_off = h * head_dim;
        let q_h = &q_row[q_off..q_off + head_dim];

        // Compute scaled logits over kv axis.
        let row_start = h * kv_len;
        for t in 0..kv_len {
            let k_row = &k_f32[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q_h[d] * k_row[d];
            }
            let mut logit = acc * scale;
            if let Some(cap) = softcap {
                if cap != 0.0 {
                    logit = (logit / cap).tanh() * cap;
                }
            }
            // Causal mask + optional sliding-window mask.
            let masked = if t > pos_q {
                f32::NEG_INFINITY
            } else if let Some(w) = sliding_window {
                if pos_q >= w && t + w <= pos_q {
                    f32::NEG_INFINITY
                } else {
                    logit
                }
            } else {
                logit
            };
            softmax_input[row_start + t] = masked;
        }

        // Softmax over kv axis for this head.
        let head_slice = &softmax_input[row_start..row_start + kv_len];
        let max_v = head_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; kv_len];
        if max_v.is_finite() {
            for (i, &v) in head_slice.iter().enumerate() {
                let e = if v.is_finite() {
                    (v - max_v).exp()
                } else {
                    0.0
                };
                probs[i] = e;
                sum += e;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }
        softmax_output[row_start..row_start + kv_len].copy_from_slice(&probs);
    }

    dump_bin("prefill", layer_idx, "attn_softmax_input", &softmax_input);
    dump_bin("prefill", layer_idx, "attn_softmax_output", &softmax_output);
}
