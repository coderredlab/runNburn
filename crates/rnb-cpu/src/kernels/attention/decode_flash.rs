//! Single-token decode attention path. Includes `attention_decode_flash`
//! (FlashAttention with online softmax) and the four `attention_decode_into_*`
//! API entries used by the engine.
//!
//! mc72 added `process_head_f16_acc` (closure inside `attention_decode_flash`)
//! with native fp16 SIMD when `RNB_ATTN_F16_NEON=1` and FEAT_FP16 is detected.
//! Split out from `attention/mod.rs` in mc74 cleanup.

use super::dispatch::*;
#[cfg(target_arch = "aarch64")]
use super::neon_helpers::*;

/// Single-query attention for decode (seq_len=1). Writes to pre-allocated output.
/// q: [num_heads * head_dim], k_cache/v_cache: [kv_len * num_kv_heads * head_dim] (F16 as u16 bits)
/// output: [num_heads * head_dim]
pub fn attention_decode_into(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) {
    attention_decode_into_with_scale(
        q,
        k_cache,
        v_cache,
        output,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        1.0 / (head_dim as f32).sqrt(),
    );
}

pub fn attention_decode_into_with_scale(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) {
    attention_decode_into_with_scale_and_window(
        q,
        k_cache,
        v_cache,
        output,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
        None,
    );
}

pub fn attention_decode_into_with_scale_and_window(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
) {
    attention_decode_into_with_scale_window_and_softcap(
        q,
        k_cache,
        v_cache,
        output,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
        sliding_window,
        None,
    )
}

pub fn attention_decode_into_with_scale_window_and_softcap(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) {
    let prof_level = std::env::var("RNB_PROFILE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    let profiling = prof_level >= 3;
    let heads_per_group = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;

    let t_alloc = std::time::Instant::now();

    const STACK_MAX: usize = 8192;
    let mut stack_scores = [0.0f32; STACK_MAX];
    let use_stack = kv_len <= STACK_MAX;
    let mut heap_scores: Vec<f32> = if use_stack {
        Vec::new()
    } else {
        vec![0.0f32; kv_len]
    };
    let scores = if use_stack {
        &mut stack_scores[..kv_len]
    } else {
        &mut heap_scores[..]
    };
    let alloc_us = t_alloc.elapsed().as_micros();

    let mut dot_us = 0u128;
    let mut softmax_us = 0u128;
    let mut zero_us = 0u128;
    let mut accum_us = 0u128;

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        for j in 0..kv_len {
            if sliding_window
                .map(|window| j + window <= kv_len - 1)
                .unwrap_or(false)
            {
                scores[j] = f32::NEG_INFINITY;
                continue;
            }
            let k_off = j * kv_dim + kv_h * head_dim;
            let t_dot = std::time::Instant::now();
            let dot = dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim);
            scores[j] = dot * scale;
            if let Some(cap) = softcap {
                scores[j] = cap * (scores[j] / cap).tanh();
            }
            if profiling {
                dot_us += t_dot.elapsed().as_micros();
            }
        }

        let t_softmax = std::time::Instant::now();
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..kv_len {
            if scores[j] > max_score {
                max_score = scores[j];
            }
        }
        let mut sum = 0.0f32;
        for j in 0..kv_len {
            scores[j] = (scores[j] - max_score).exp();
            sum += scores[j];
        }
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        if profiling {
            softmax_us += t_softmax.elapsed().as_micros();
        }

        let out_off = h * head_dim;
        let t_zero = std::time::Instant::now();
        for d in 0..head_dim {
            output[out_off + d] = 0.0;
        }
        if profiling {
            zero_us += t_zero.elapsed().as_micros();
        }

        let t_accum = std::time::Instant::now();
        for j in 0..kv_len {
            let v_off = j * kv_dim + kv_h * head_dim;
            scaled_add_f16(
                &mut output[out_off..out_off + head_dim],
                &v_cache[v_off..v_off + head_dim],
                scores[j] * inv_sum,
            );
        }
        if profiling {
            accum_us += t_accum.elapsed().as_micros();
        }
    }

    if profiling {
        eprintln!(
            "    [ATTN-DEC-INNER] alloc={:.3}ms dot={:.3}ms softmax={:.3}ms zero={:.3}ms accum={:.3}ms heads={} kv_len={} head_dim={}",
            alloc_us as f64 / 1000.0,
            dot_us as f64 / 1000.0,
            softmax_us as f64 / 1000.0,
            zero_us as f64 / 1000.0,
            accum_us as f64 / 1000.0,
            num_heads,
            kv_len,
            head_dim,
        );
    }
}

/// Flash-style fused online-softmax decode attention (seq_len=1, F16 KV cache).
///
/// 기존 `attention_decode_into_with_scale_window_and_softcap` 의 4-pass 구조
/// (dot → softmax max/exp/sum → zero → accum) 를 **single j-loop** 으로 fuse.
///
/// Running (m, s) = (max, sum_exp) 를 j 순서대로 업데이트하고,
/// max 가 바뀔 때마다 누적 output 과 s 를 rescale 한다.
/// 이점:
/// 1. scores[] 버퍼 제거 (stack 8192 / heap)
/// 2. K, V j-loop 통합 (cache locality)
/// 3. Head 별 par_iter 로 병렬화 (현재 단일 thread)
///
/// 정확도: 순수 fp32 online softmax — 수학적으로 기존 4-pass 와 동치.
/// F16 round-trip 은 기존과 동일.
pub fn attention_decode_flash(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) {
    let heads_per_group = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Per-head kernel — closure 로 추출해서 par/seq 양쪽 재사용.
    let process_head = |h: usize, out_head: &mut [f32]| {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        // Running state
        let mut m = f32::NEG_INFINITY;
        let mut s = 0.0f32;
        // Zero-init output for this head
        for x in out_head.iter_mut() {
            *x = 0.0;
        }

        for j in 0..kv_len {
            if sliding_window
                .map(|window| j + window <= kv_len - 1)
                .unwrap_or(false)
            {
                continue;
            }

            let k_off = j * kv_dim + kv_h * head_dim;
            let mut x = dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim) * scale;
            if let Some(cap) = softcap {
                x = cap * (x / cap).tanh();
            }

            let v_off = j * kv_dim + kv_h * head_dim;
            if x > m {
                // Max moved: rescale running state (skipped on the very first valid j)
                if m > f32::NEG_INFINITY {
                    let alpha = (m - x).exp();
                    scale_f32(out_head, alpha);
                    s *= alpha;
                }
                // p = exp(x - new_m) = exp(0) = 1
                scaled_add_f16(out_head, &v_cache[v_off..v_off + head_dim], 1.0);
                s += 1.0;
                m = x;
            } else {
                // x <= m: standard exp(x - m), no rescale of accumulator
                let p = (x - m).exp();
                scaled_add_f16(out_head, &v_cache[v_off..v_off + head_dim], p);
                s += p;
            }
        }

        // Final normalize
        if s > 0.0 {
            let inv_s = 1.0 / s;
            scale_f32(out_head, inv_s);
        }
    };

    // f64-accumulator path — opt-in via RNB_ATTN_F64=1. Used to test whether
    // the 24-layer decode hidden-state drift (mc70 finding: top-1 vs top-2
    // logit margin 0.1-0.5 gets flipped by f32 accumulation) is rooted in the
    // attention V-accumulator + running softmax sum. Cost: extra f64 buffer
    // per head, so only the slow scalar path is taken when this is set.
    let f64_path = std::env::var("RNB_ATTN_F64").is_ok();
    // Kahan compensated-sum path — opt-in via RNB_ATTN_KAHAN=1. Stays in f32
    // memory, but every running-sum add is done as Kahan summation so the
    // ULP rounding error stops accumulating linearly with kv_len (and across
    // the 24 layers). Cost is roughly 2-3x more f32 ops on the running sum
    // and the V-accumulator vector, but no extra memory bandwidth.
    let kahan_path = std::env::var("RNB_ATTN_KAHAN").is_ok();
    let process_head_kahan = |h: usize, out_head: &mut [f32]| {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        let mut m = f32::NEG_INFINITY;
        let mut s: f32 = 0.0;
        let mut s_c: f32 = 0.0; // Kahan compensation for running softmax sum
                                // Use stack-friendly Vec for accumulator + compensation. head_dim is
                                // typically 64-256 so allocation cost is small versus the kv_len loop.
        let mut acc: Vec<f32> = vec![0.0; head_dim];
        let mut acc_c: Vec<f32> = vec![0.0; head_dim];

        for j in 0..kv_len {
            if sliding_window
                .map(|window| j + window <= kv_len - 1)
                .unwrap_or(false)
            {
                continue;
            }

            let k_off = j * kv_dim + kv_h * head_dim;
            let mut x = dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim) * scale;
            if let Some(cap) = softcap {
                x = cap * (x / cap).tanh();
            }

            let v_off = j * kv_dim + kv_h * head_dim;
            if x > m {
                if m > f32::NEG_INFINITY {
                    let alpha = (m - x).exp();
                    // Multiplicative rescale — compensation must scale too,
                    // otherwise the next Kahan add would over-correct.
                    for d in 0..head_dim {
                        acc[d] *= alpha;
                        acc_c[d] *= alpha;
                    }
                    s *= alpha;
                    s_c *= alpha;
                }
                // p = exp(x - new_m) = 1
                for d in 0..head_dim {
                    let v = half::f16::from_bits(v_cache[v_off + d]).to_f32();
                    let y = v - acc_c[d];
                    let t = acc[d] + y;
                    acc_c[d] = (t - acc[d]) - y;
                    acc[d] = t;
                }
                let y = 1.0 - s_c;
                let t = s + y;
                s_c = (t - s) - y;
                s = t;
                m = x;
            } else {
                let p = (x - m).exp();
                for d in 0..head_dim {
                    let v = half::f16::from_bits(v_cache[v_off + d]).to_f32();
                    let pv = p * v;
                    let y = pv - acc_c[d];
                    let t = acc[d] + y;
                    acc_c[d] = (t - acc[d]) - y;
                    acc[d] = t;
                }
                let y = p - s_c;
                let t = s + y;
                s_c = (t - s) - y;
                s = t;
            }
        }

        if s > 0.0 {
            let inv_s = 1.0 / s;
            for d in 0..head_dim {
                out_head[d] = acc[d] * inv_s;
            }
        } else {
            for x in out_head.iter_mut() {
                *x = 0.0;
            }
        }
    };
    let process_head_f64 = |h: usize, out_head: &mut [f32]| {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        let mut m = f32::NEG_INFINITY;
        let mut s: f64 = 0.0;
        let mut acc: Vec<f64> = vec![0.0; head_dim];

        for j in 0..kv_len {
            if sliding_window
                .map(|window| j + window <= kv_len - 1)
                .unwrap_or(false)
            {
                continue;
            }

            let k_off = j * kv_dim + kv_h * head_dim;
            let mut x = dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim) * scale;
            if let Some(cap) = softcap {
                x = cap * (x / cap).tanh();
            }

            let v_off = j * kv_dim + kv_h * head_dim;
            if x > m {
                if m > f32::NEG_INFINITY {
                    let alpha = ((m - x) as f64).exp();
                    for a in acc.iter_mut() {
                        *a *= alpha;
                    }
                    s *= alpha;
                }
                for d in 0..head_dim {
                    acc[d] += half::f16::from_bits(v_cache[v_off + d]).to_f32() as f64;
                }
                s += 1.0;
                m = x;
            } else {
                let p = ((x - m) as f64).exp();
                for d in 0..head_dim {
                    acc[d] += p * half::f16::from_bits(v_cache[v_off + d]).to_f32() as f64;
                }
                s += p;
            }
        }

        if s > 0.0 {
            let inv_s = 1.0 / s;
            for d in 0..head_dim {
                out_head[d] = (acc[d] * inv_s) as f32;
            }
        } else {
            for x in out_head.iter_mut() {
                *x = 0.0;
            }
        }
    };

    // f16 V-accumulator path — default ON. mc71 finding: GGML's flash
    // attention keeps the V accumulator in f16 (`VKQ16`), using native fp16
    // arithmetic (vmulq_f16, vfmaq_f16) to scale and mad. Q is rounded to
    // f16 before the K-dot (matching GGML's `q_to_vec_dot`). NEON native
    // fp16 vector path is taken when target supports ARMv8.2 fp16 vector
    // arithmetic; falls back to scalar emulation otherwise.
    let f16_acc_path = std::env::var("RNB_ATTN_F16_ACC")
        .map(|v| v != "0")
        .unwrap_or(true);
    // mc72 SIMD path policy (production default OFF).
    //
    // Native ARMv8.2-A FEAT_FP16 path (vmulq_f16 / vfmaq_f16 single-rounding +
    // GGML mul_add control flow + optional native f16×f16 dot + token-wise
    // prefill dispatch via attention_decode_flash). When fully enabled, raw
    // mode reaches step 0-8 token-identical with llama.cpp ARM (native fp16
    // path), but in CPU-only environments (vulkan inactive, `submits=0`) the
    // token-wise prefill incurs O(seq_len²) cost per layer because each
    // attention call re-traverses cumulative KV — overwhelming the SIMD ops
    // win on long prompts (chat mode prefill ~23 tokens: 952ms vs batch 438ms).
    //
    // The token-wise dispatch matches the vulkan attention path
    // (`run_attention_window_post_ffw_norm_path` in
    // `engine/backend_runtime/vulkan_attention_prefill.rs`), which iterates
    // `for t in 0..seq_len { append_kv → decode_attention }` because GPU
    // attention prefers single-Q × multi-KV kernels. So the CPU SIMD path is
    // architecturally aligned with vulkan-active environments and serves as
    // research/benchmark infra; CPU-only production should stay on the mc71
    // fp16-fallback batch path.
    //
    // `RNB_ATTN_F16_NEON=1` opts in (research, vulkan-active alignment check).
    // mc73 will convert the batch prefill kernel
    // (`attention_with_scale_window_and_softcap`) to a SIMD path so CPU-only
    // gets the SIMD ops win without the token-wise quadratic cost; that's
    // the precondition for promoting this default to ON.
    let f16_neon_enabled = std::env::var("RNB_ATTN_F16_NEON")
        .map(|v| v != "0")
        .unwrap_or(false);
    #[cfg(target_arch = "aarch64")]
    let has_fp16 = f16_neon_enabled && std::arch::is_aarch64_feature_detected!("fp16");
    #[cfg(not(target_arch = "aarch64"))]
    #[allow(unused_variables)]
    let has_fp16 = false;
    #[cfg(not(target_arch = "aarch64"))]
    let _ = f16_neon_enabled;
    // Debug-only axis probe: skip the Q -> f16 -> f32 round so we can A/B
    // diagnose whether Q rounding is the quality axis. Production never sets
    // this; if you find yourself reaching for it, log the smoke result first.
    let no_q_round = std::env::var("RNB_ATTN_NO_Q_ROUND").is_ok();
    // mc73 cleanup: native f16×f16 dot is bundled into `RNB_ATTN_F16_NEON`.
    // GGML simd-mappings.h `GGML_F16_STEP=32, ARR=4, EPR=8` native fp16 path
    // (`neon_vec_dot_f16_f16`) used whenever the SIMD path is active.
    let process_head_f16_acc = |h: usize, out_head: &mut [f32]| {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        // Round Q to f16 (matches GGML's `q_to_vec_dot(pq, Q_q, DK)`)
        let q_head_f32 = &q[q_off..q_off + head_dim];
        // f32 storage (f16-rounded *value*) — used by fp16-fallback dot path.
        let q_head_f16: Vec<f32> = if no_q_round {
            q_head_f32.to_vec()
        } else {
            q_head_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect()
        };
        let q_head = &q_head_f16[..];
        // mc72 axis 4: q in raw u16 f16 storage — used by native f16×f16 dot
        // (`neon_vec_dot_f16_f16`). Matches GGML's `q_to_vec_dot` followed by
        // `vec_dot_f16` on the native fp16 cpu path.
        #[cfg(target_arch = "aarch64")]
        let q_head_u16: Vec<u16> = if has_fp16 && !no_q_round {
            q_head_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect()
        } else {
            Vec::new()
        };

        let mut m = f32::NEG_INFINITY;
        let mut s: f32 = 0.0;
        // f16 accumulator stored as raw u16 bits, matching ggml_fp16_t.
        let mut acc_f16: Vec<u16> = vec![0u16; head_dim];

        for j in 0..kv_len {
            if sliding_window
                .map(|window| j + window <= kv_len - 1)
                .unwrap_or(false)
            {
                continue;
            }

            let k_off = j * kv_dim + kv_h * head_dim;
            #[cfg(target_arch = "aarch64")]
            let dot = if has_fp16 && !no_q_round {
                unsafe {
                    neon_vec_dot_f16_f16(q_head_u16.as_ptr(), k_cache[k_off..].as_ptr(), head_dim)
                }
            } else {
                dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dot = dot_f32_f16(q_head, &k_cache[k_off..k_off + head_dim], head_dim);
            let mut x = dot * scale;
            if let Some(cap) = softcap {
                x = cap * (x / cap).tanh();
            }

            let v_off = j * kv_dim + kv_h * head_dim;
            let v_slice = &v_cache[v_off..v_off + head_dim];
            // mc72 axis 5: SIMD ON path uses GGML's `S = S*ms + vs` mul_add
            // form + always-call vec_mad_f16 to match `flash_attn_ext_f16`
            // exactly. SIMD OFF path keeps mc71's branched `s *= alpha; s +=
            // 1.0` form (the close-call alignment that produces natural
            // Korean generation). Mixed dispatch — has_fp16 picks the form.
            if has_fp16 {
                #[cfg(target_arch = "aarch64")]
                {
                    let m_old = m;
                    let mut ms = 1.0_f32;
                    let mut vs = 1.0_f32;
                    if x > m {
                        m = x;
                        ms = (m_old - m).exp();
                        if m_old > f32::NEG_INFINITY {
                            let ms_bits = half::f16::from_f32(ms).to_bits();
                            unsafe {
                                neon_vec_scale_f16(acc_f16.as_mut_ptr(), ms_bits, head_dim);
                            }
                        }
                    } else {
                        vs = (x - m).exp();
                    }
                    let vs_bits = half::f16::from_f32(vs).to_bits();
                    unsafe {
                        neon_vec_mad_f16(acc_f16.as_mut_ptr(), v_slice.as_ptr(), vs_bits, head_dim);
                    }
                    s = s.mul_add(ms, vs);
                }
            } else if x > m {
                if m > f32::NEG_INFINITY {
                    let alpha = (m - x).exp();
                    for d in 0..head_dim {
                        let v = half::f16::from_bits(acc_f16[d]).to_f32() * alpha;
                        acc_f16[d] = half::f16::from_f32(v).to_bits();
                    }
                    s *= alpha;
                }
                for d in 0..head_dim {
                    let acc_v = half::f16::from_bits(acc_f16[d]).to_f32();
                    let v_v = half::f16::from_bits(v_slice[d]).to_f32();
                    let new = acc_v + v_v;
                    acc_f16[d] = half::f16::from_f32(new).to_bits();
                }
                s += 1.0;
                m = x;
            } else {
                let p = (x - m).exp();
                for d in 0..head_dim {
                    let acc_v = half::f16::from_bits(acc_f16[d]).to_f32();
                    let v_v = half::f16::from_bits(v_slice[d]).to_f32();
                    let new = acc_v + v_v * p;
                    acc_f16[d] = half::f16::from_f32(new).to_bits();
                }
                s += p;
            }
        }

        // Final f16 -> f32 + normalize (matches GGML's
        // `VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]); ... /= S` pattern).
        if s > 0.0 {
            let inv_s = 1.0 / s;
            for d in 0..head_dim {
                out_head[d] = half::f16::from_bits(acc_f16[d]).to_f32() * inv_s;
            }
        } else {
            for x in out_head.iter_mut() {
                *x = 0.0;
            }
        }
    };

    let parallel = std::env::var("RNB_FLASH_DECODE_SERIAL").is_err() && num_heads >= 4;
    if parallel {
        use rayon::prelude::*;
        // Output 을 head 별로 분할해서 par_chunks_mut 로 안전하게 나눠줌
        output
            .par_chunks_mut(head_dim)
            .enumerate()
            .take(num_heads)
            .for_each(|(h, out_head)| {
                if f16_acc_path {
                    process_head_f16_acc(h, out_head);
                } else if f64_path {
                    process_head_f64(h, out_head);
                } else if kahan_path {
                    process_head_kahan(h, out_head);
                } else {
                    process_head(h, out_head);
                }
            });
    } else {
        for h in 0..num_heads {
            let out_off = h * head_dim;
            let out_head = &mut output[out_off..out_off + head_dim];
            if f16_acc_path {
                process_head_f16_acc(h, out_head);
            } else if f64_path {
                process_head_f64(h, out_head);
            } else if kahan_path {
                process_head_kahan(h, out_head);
            } else {
                process_head(h, out_head);
            }
        }
    }
}
