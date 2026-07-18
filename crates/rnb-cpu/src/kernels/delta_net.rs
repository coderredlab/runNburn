/// Gated Delta Net autoregressive scan.
///
/// Delta Net recurrence per head per token:
///   state_decay = exp(gate) * state
///   d = (v - state_decay^T @ k) * beta
///   state = state_decay + outer(k, d)
///   output = state^T @ q
///
/// q: [seq_len * num_heads * head_k_dim]  (L2-normalized, interleaved by head)
/// k: [seq_len * num_heads * head_k_dim]  (L2-normalized)
/// v: [seq_len * num_heads * head_v_dim]
/// gate: [seq_len * num_heads]  (raw gate values, will be exp'd)
/// beta: [seq_len * num_heads]  (already sigmoid'd)
/// state: [num_heads * head_v_dim * head_k_dim]  (mutable, updated in-place)
///
/// Returns: [seq_len * num_heads * head_v_dim]
pub fn delta_net_scan(
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
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * num_heads * head_v_dim];
    delta_net_scan_into(
        q,
        k,
        v,
        gate,
        beta,
        state,
        &mut output,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    output
}

/// delta_net_scan zero-alloc wrapper. Writes to pre-allocated output buffer.
pub fn delta_net_scan_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if head_k_dim % 4 == 0 {
        unsafe {
            delta_net_scan_neon_into(
                q, k, v, gate, beta, state, output, seq_len, num_heads, head_k_dim, head_v_dim,
            )
        };
        return;
    }
    delta_net_scan_scalar_into(
        q, k, v, gate, beta, state, output, seq_len, num_heads, head_k_dim, head_v_dim,
    );
}
/// Applies one Gated Delta Net token to `state` without computing an output.
///
/// `k`, `v`, `gate`, and `beta` contain exactly one token, interleaved by head.
/// The state layout matches [`delta_net_scan_into`].
pub fn delta_net_step_state_only(
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if head_k_dim % 4 == 0 {
        let state_size = head_v_dim * head_k_dim;
        unsafe {
            for h in 0..num_heads {
                delta_net_step_neon_head::<false>(
                    k.as_ptr().add(h * head_k_dim),
                    &v[h * head_v_dim..(h + 1) * head_v_dim],
                    gate[h].exp(),
                    beta[h],
                    state.as_mut_ptr().add(h * state_size),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    head_k_dim,
                    head_v_dim,
                );
            }
        }
        return;
    }

    let state_size = head_v_dim * head_k_dim;
    for h in 0..num_heads {
        delta_net_step_scalar_head(
            &k[h * head_k_dim..(h + 1) * head_k_dim],
            &v[h * head_v_dim..(h + 1) * head_v_dim],
            gate[h].exp(),
            beta[h],
            &mut state[h * state_size..(h + 1) * state_size],
            head_k_dim,
            head_v_dim,
        );
    }
}

/// pm38 M1: chunkwise parallel form of [`delta_net_scan`] (WY representation + UT transform).
///
/// 수학적으로 sequential recurrence 와 **exact 동등**(근사 아님). sequence 를 `chunk_size`
/// 단위로 쪼개 intra-chunk 는 dense GEMM 으로 병렬, inter-chunk 만 순차 state hand-off.
/// gated delta rule: `S_t = decay_t·S_{t-1}·(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t·q_t`.
///
/// Metal GPU 커널(향후)의 correctness oracle. 같은 precision(f32)이면 sequential 과
/// token-identical. 인자/레이아웃은 [`delta_net_scan`] 과 동일 + `chunk_size`.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_scan_chunkwise(
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
    chunk_size: usize,
) -> Vec<f32> {
    // pm38 M1: WY representation + UT transform (flash-linear-attention chunk gated delta rule).
    // gated delta rule `S_t = decay_t·S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t q_t`
    // 를 chunk 단위로 재배열 — intra-chunk 는 dense 연산, inter-chunk 만 순차 state hand-off.
    // 수학적으로 sequential 과 exact 동등(chunk_size=1 에서 per-token recurrence 와 대수 동일).
    // decay 는 chunk-local log-cumsum G 로 들고 항상 상대형 exp(G_i−G_j), i≥j(≤1, underflow 안전).
    // 모든 누적 f32, reduction j 오름차순(sequential 과 roundoff 정합). Metal GPU 커널 oracle.
    let chunk_size = chunk_size.max(1);
    let state_size = head_v_dim * head_k_dim;
    let mut output = vec![0.0f32; seq_len * num_heads * head_v_dim];

    // 각 head 독립. state[h*state_size + vi*head_k_dim + ki] = S_h[vi,ki].
    for h in 0..num_heads {
        let s_base = h * state_size;
        let mut t0 = 0;
        while t0 < seq_len {
            let c = chunk_size.min(seq_len - t0);
            let qk_idx = |r: usize, d: usize| ((t0 + r) * num_heads + h) * head_k_dim + d;
            let v_idx = |r: usize, d: usize| ((t0 + r) * num_heads + h) * head_v_dim + d;
            let gb = |r: usize| (t0 + r) * num_heads + h;

            // STEP 1: chunk-local 누적 log-decay G[r] = Σ_{m≤r} gate_m (f32).
            let mut g_cum = vec![0.0f32; c];
            let mut acc_g = 0.0f32;
            for r in 0..c {
                acc_g += gate[gb(r)];
                g_cum[r] = acc_g;
            }

            // STEP 2: WY forward substitution + S_init 보정을 한 recurrence 로 (필수 — 분리하면
            // 교차항이 raw u 를 써서 multi-token chunk drift). u_corr[r] = d_r (exact):
            //   d_r = β_r·(v_r − γ_r·(S·k_r) − Σ_{i<r} s_kk[i]·d_i),  s_kk[i]=(k_i·k_r)·exp(G_r−G_i).
            // 교차항에 보정된 d_i(=u_corr[i]) 사용. (chunk_size=1 은 Σ 없어 자명, ≥2 부터 결정적.)
            let mut u_corr = vec![0.0f32; c * head_v_dim];
            for r in 0..c {
                let br = beta[gb(r)];
                let gr = g_cum[r].exp();
                // s_kk[i] = (k_i·k_r)·exp(G_r−G_i), i<r (상대 decay ≤1)
                let mut s_kk = vec![0.0f32; r];
                for (i, skk) in s_kk.iter_mut().enumerate() {
                    let mut kk = 0.0f32;
                    for d in 0..head_k_dim {
                        kk += k[qk_idx(i, d)] * k[qk_idx(r, d)];
                    }
                    *skk = kk * (g_cum[r] - g_cum[i]).exp();
                }
                for vi in 0..head_v_dim {
                    // γ_r·(S_init·k_r)[vi]
                    let mut pred = 0.0f32;
                    for ki in 0..head_k_dim {
                        pred += state[s_base + vi * head_k_dim + ki] * k[qk_idx(r, ki)];
                    }
                    let mut a = v[v_idx(r, vi)] - gr * pred;
                    for i in 0..r {
                        a -= s_kk[i] * u_corr[i * head_v_dim + vi];
                    }
                    u_corr[r * head_v_dim + vi] = br * a;
                }
            }

            // STEP 4: output o_r = γ_r·(S·q_r) + Σ_{j≤r} (q_r·k_j)·exp(G_r−G_j)·u'_j.
            for r in 0..c {
                let gr = g_cum[r].exp();
                for vi in 0..head_v_dim {
                    let mut inter = 0.0f32;
                    for ki in 0..head_k_dim {
                        inter += state[s_base + vi * head_k_dim + ki] * q[qk_idx(r, ki)];
                    }
                    let mut o = inter * gr;
                    for j in 0..=r {
                        let mut qk = 0.0f32;
                        for d in 0..head_k_dim {
                            qk += q[qk_idx(r, d)] * k[qk_idx(j, d)];
                        }
                        o += qk * (g_cum[r] - g_cum[j]).exp() * u_corr[j * head_v_dim + vi];
                    }
                    output[v_idx(r, vi)] = o;
                }
            }

            // STEP 5: state hand-off S ← γ_C·S + Σ_j exp(G_last−G_j)·u'_j·k_jᵀ (output 계산 후!).
            let g_last = g_cum[c - 1];
            let gc = g_last.exp();
            for vi in 0..head_v_dim {
                for ki in 0..head_k_dim {
                    let mut a = gc * state[s_base + vi * head_k_dim + ki];
                    for j in 0..c {
                        let rescale = (g_last - g_cum[j]).exp();
                        a += rescale * u_corr[j * head_v_dim + vi] * k[qk_idx(j, ki)];
                    }
                    state[s_base + vi * head_k_dim + ki] = a;
                }
            }

            t0 += c;
        }
    }
    output
}

fn delta_net_step_scalar_head(
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    state: &mut [f32],
    head_k_dim: usize,
    head_v_dim: usize,
) {
    let mut d = [0.0f32; 256]; // stack buffer; max head_v_dim is 128
    for vi in 0..head_v_dim {
        let mut sk = 0.0f32;
        for ki in 0..head_k_dim {
            sk += decay * state[vi * head_k_dim + ki] * k[ki];
        }
        d[vi] = (v[vi] - sk) * beta;
    }

    for vi in 0..head_v_dim {
        for ki in 0..head_k_dim {
            let idx = vi * head_k_dim + ki;
            state[idx] = decay * state[idx] + k[ki] * d[vi];
        }
    }
}

fn delta_net_scan_scalar_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    let state_size = head_v_dim * head_k_dim;
    for t in 0..seq_len {
        for h in 0..num_heads {
            let q_off = (t * num_heads + h) * head_k_dim;
            let k_off = (t * num_heads + h) * head_k_dim;
            let v_off = (t * num_heads + h) * head_v_dim;
            let gb_off = t * num_heads + h;
            let s_off = h * state_size;
            let o_off = (t * num_heads + h) * head_v_dim;

            let q_h = &q[q_off..q_off + head_k_dim];
            let state_h = &mut state[s_off..s_off + state_size];
            delta_net_step_scalar_head(
                &k[k_off..k_off + head_k_dim],
                &v[v_off..v_off + head_v_dim],
                gate[gb_off].exp(),
                beta[gb_off],
                state_h,
                head_k_dim,
                head_v_dim,
            );

            for vi in 0..head_v_dim {
                let mut sq = 0.0f32;
                for ki in 0..head_k_dim {
                    sq += state_h[vi * head_k_dim + ki] * q_h[ki];
                }
                output[o_off + vi] = sq;
            }
        }
    }
}

/// Applies one NEON-vectorized recurrence to one head.
///
/// `WRITE_OUTPUT = false` omits all q loads, output stores, and output dot products.
#[cfg(target_arch = "aarch64")]
unsafe fn delta_net_step_neon_head<const WRITE_OUTPUT: bool>(
    k_ptr: *const f32,
    v: &[f32],
    decay: f32,
    beta: f32,
    state_ptr: *mut f32,
    q_ptr: *const f32,
    output_ptr: *mut f32,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    use std::arch::aarch64::*;

    let k_steps = head_k_dim / 4;
    let decay_v = vdupq_n_f32(decay);

    let mut d = [0.0f32; 256];
    for vi in 0..head_v_dim {
        let state_row = state_ptr.add(vi * head_k_dim);
        let mut acc = vdupq_n_f32(0.0);
        for j in 0..k_steps {
            let state_v = vmulq_f32(decay_v, vld1q_f32(state_row.add(j * 4)));
            acc = vfmaq_f32(acc, state_v, vld1q_f32(k_ptr.add(j * 4)));
        }
        d[vi] = (v[vi] - vaddvq_f32(acc)) * beta;
    }

    for vi in 0..head_v_dim {
        let state_row = state_ptr.add(vi * head_k_dim);
        let d_vi = vdupq_n_f32(d[vi]);
        let mut output_acc = vdupq_n_f32(0.0);
        for j in 0..k_steps {
            let off = j * 4;
            let state_v = vld1q_f32(state_row.add(off));
            let k_v = vld1q_f32(k_ptr.add(off));
            let result = vfmaq_f32(vmulq_f32(decay_v, state_v), k_v, d_vi);
            vst1q_f32(state_row.add(off), result);
            if WRITE_OUTPUT {
                output_acc = vfmaq_f32(output_acc, result, vld1q_f32(q_ptr.add(off)));
            }
        }
        if WRITE_OUTPUT {
            *output_ptr.add(vi) = vaddvq_f32(output_acc);
        }
    }
}

/// NEON-vectorized delta_net_scan: 4-wide f32 SIMD for all matrix ops.
/// For prefill (seq_len > 1) with 4+ heads, heads are processed in parallel via rayon.
/// For decode (seq_len == 1), sequential to avoid rayon overhead (~10μs).
#[cfg(target_arch = "aarch64")]
unsafe fn delta_net_scan_neon_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    use rayon::prelude::*;

    let state_size = head_v_dim * head_k_dim;
    let output_ptr = output.as_mut_ptr() as usize;
    let state_ptr = state.as_mut_ptr() as usize;

    for t in 0..seq_len {
        let head_fn = |h: usize| {
            let q_off = (t * num_heads + h) * head_k_dim;
            let k_off = (t * num_heads + h) * head_k_dim;
            let v_off = (t * num_heads + h) * head_v_dim;
            let gb_off = t * num_heads + h;
            let s_off = h * state_size;
            let o_off = (t * num_heads + h) * head_v_dim;

            delta_net_step_neon_head::<true>(
                k.as_ptr().add(k_off),
                &v[v_off..v_off + head_v_dim],
                gate[gb_off].exp(),
                beta[gb_off],
                (state_ptr as *mut f32).add(s_off),
                q.as_ptr().add(q_off),
                (output_ptr as *mut f32).add(o_off),
                head_k_dim,
                head_v_dim,
            );
        };

        if seq_len > 1 && num_heads >= 4 {
            (0..num_heads).into_par_iter().for_each(|h| head_fn(h));
        } else {
            for h in 0..num_heads {
                head_fn(h);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct CountingAlloc;

    thread_local! {
        static COUNT_ALLOC_ENABLED: Cell<bool> = const { Cell::new(false) };
        static COUNT_ALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNT_ALLOC_ENABLED.with(|enabled| {
                if enabled.get() {
                    COUNT_ALLOC_CALLS.with(|calls| calls.set(calls.get() + 1));
                }
            });
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAlloc = CountingAlloc;

    #[test]
    fn test_delta_net_single_token_single_head() {
        let num_heads = 1;
        let head_k_dim = 2;
        let head_v_dim = 2;

        // q = [1, 0], k = [1, 0], v = [3, 7]
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![3.0, 7.0];
        let gate = vec![0.0]; // exp(0) = 1 (no decay)
        let beta = vec![1.0]; // full learning rate

        let mut state = vec![0.0f32; num_heads * head_v_dim * head_k_dim];

        let output = delta_net_scan(
            &q, &k, &v, &gate, &beta, &mut state, 1, num_heads, head_k_dim, head_v_dim,
        );

        // d = (v - 0) * 1 = [3, 7]
        // state = 0 + outer(k=[1,0], d=[3,7]) = [[3,0],[7,0]]
        // output = state^T @ q = [[3,7],[0,0]] @ [1,0] = [3, 7]
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_delta_net_decay() {
        let num_heads = 1;
        let hd = 1;

        // Two tokens: first writes v=10, second reads with decay
        let q = vec![1.0, 1.0]; // 2 tokens, 1 head, 1 dim
        let k = vec![1.0, 0.0]; // second token k=0 (no write)
        let v = vec![10.0, 0.0];
        let gate = vec![0.0, -1.0]; // first: decay=1, second: decay=exp(-1)≈0.368
        let beta = vec![1.0, 1.0];

        let mut state = vec![0.0f32];

        let output = delta_net_scan(&q, &k, &v, &gate, &beta, &mut state, 2, num_heads, hd, hd);

        // Token 0: d=(10-0)*1=10, state=1*0+1*10=10, out=10*1=10
        assert!((output[0] - 10.0).abs() < 1e-4);
        // Token 1: d=(0-state*0)*1=0, state=exp(-1)*10+0=3.679, out=3.679*1
        assert!((output[1] - 10.0 * (-1.0f32).exp()).abs() < 1e-3);
    }

    #[test]
    fn test_delta_net_delta_uses_decayed_state() {
        let num_heads = 1;
        let hd = 1;

        let q = vec![1.0];
        let k = vec![1.0];
        let v = vec![10.0];
        let gate = vec![(0.5f32).ln()];
        let beta = vec![1.0];
        let mut state = vec![8.0f32];

        let output = delta_net_scan(&q, &k, &v, &gate, &beta, &mut state, 1, num_heads, hd, hd);

        assert!((state[0] - 10.0).abs() < 1e-5);
        assert!((output[0] - 10.0).abs() < 1e-5);
    }

    /// 결정적 pseudo-random 입력 (sin/cos 기반). gate 는 음수(decay<1 안정), beta∈(0,1).
    fn make_deterministic_inputs(
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_qk = seq_len * num_heads * head_k_dim;
        let n_v = seq_len * num_heads * head_v_dim;
        let n_gb = seq_len * num_heads;
        let q: Vec<f32> = (0..n_qk).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let k: Vec<f32> = (0..n_qk)
            .map(|i| (i as f32 * 0.53 + 1.0).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..n_v).map(|i| (i as f32 * 0.29).sin()).collect();
        // gate 음수 → decay = exp(gate) ∈ (0,1)
        let gate: Vec<f32> = (0..n_gb)
            .map(|i| -0.1 - 0.2 * (i as f32 * 0.11).sin().abs())
            .collect();
        // beta ∈ (0,1)
        let beta: Vec<f32> = (0..n_gb)
            .map(|i| 0.3 + 0.4 * (i as f32 * 0.17).cos().abs())
            .collect();
        (q, k, v, gate, beta)
    }

    /// pm38 M1: chunkwise 가 sequential 과 token-identical (exact 알고리즘).
    #[test]
    fn chunkwise_matches_sequential_multihead() {
        let (seq_len, num_heads, head_k_dim, head_v_dim) = (12, 2, 4, 4);
        let (q, k, v, gate, beta) =
            make_deterministic_inputs(seq_len, num_heads, head_k_dim, head_v_dim);

        let mut state_seq = vec![0.0f32; num_heads * head_v_dim * head_k_dim];
        let out_seq = delta_net_scan(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut state_seq,
            seq_len,
            num_heads,
            head_k_dim,
            head_v_dim,
        );

        for cs in [1usize, 2, 3, 4, 12] {
            let mut state_chunk = vec![0.0f32; num_heads * head_v_dim * head_k_dim];
            let out_chunk = delta_net_scan_chunkwise(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                &mut state_chunk,
                seq_len,
                num_heads,
                head_k_dim,
                head_v_dim,
                cs,
            );
            let mut max_o = 0.0f32;
            let mut max_s = 0.0f32;
            for i in 0..out_seq.len() {
                max_o = max_o.max((out_seq[i] - out_chunk[i]).abs());
            }
            for i in 0..state_seq.len() {
                max_s = max_s.max((state_seq[i] - state_chunk[i]).abs());
            }
            assert!(
                max_o < 1e-4 && max_s < 1e-4,
                "chunk_size={cs}: max_o_diff={max_o:.6} max_s_diff={max_s:.6}"
            );
        }
    }

    /// pm38 M1: 큰 입력 + non-zero initial state + remainder chunk (37 = 16+16+5).
    #[test]
    fn chunkwise_matches_sequential_large_nonzero_state() {
        let (seq_len, num_heads, head_k_dim, head_v_dim) = (37, 3, 16, 8);
        let (q, k, v, gate, beta) =
            make_deterministic_inputs(seq_len, num_heads, head_k_dim, head_v_dim);
        // non-zero 초기 state (decode-후-prefill 같은 carry 상황)
        let init: Vec<f32> = (0..num_heads * head_v_dim * head_k_dim)
            .map(|i| (i as f32 * 0.13).sin() * 0.2)
            .collect();

        let mut state_seq = init.clone();
        let out_seq = delta_net_scan(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut state_seq,
            seq_len,
            num_heads,
            head_k_dim,
            head_v_dim,
        );

        for cs in [8usize, 16, 37] {
            let mut state_chunk = init.clone();
            let out_chunk = delta_net_scan_chunkwise(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                &mut state_chunk,
                seq_len,
                num_heads,
                head_k_dim,
                head_v_dim,
                cs,
            );
            let max_o = out_seq
                .iter()
                .zip(&out_chunk)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_s = state_seq
                .iter()
                .zip(&state_chunk)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_o < 2e-4 && max_s < 2e-4,
                "chunk_size={cs}: max_o_diff={max_o:.6} max_s_diff={max_s:.6}"
            );
        }
    }

    #[test]
    fn delta_net_state_only_matches_single_token_scalar_scan() {
        let (num_heads, head_k_dim, head_v_dim) = (3, 5, 4);
        let (q, k, v, gate, beta) = make_deterministic_inputs(1, num_heads, head_k_dim, head_v_dim);
        let initial_state: Vec<f32> = (0..num_heads * head_v_dim * head_k_dim)
            .map(|i| (i as f32 * 0.19 + 0.3).cos() * 0.25)
            .collect();
        let mut scan_state = initial_state.clone();
        let mut state_only = initial_state;
        let mut output = vec![0.0f32; num_heads * head_v_dim];

        delta_net_scan_scalar_into(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut scan_state,
            &mut output,
            1,
            num_heads,
            head_k_dim,
            head_v_dim,
        );
        delta_net_step_state_only(
            &k,
            &v,
            &gate,
            &beta,
            &mut state_only,
            num_heads,
            head_k_dim,
            head_v_dim,
        );

        assert_eq!(state_only, scan_state);
    }

    #[test]
    fn delta_net_state_only_avoids_heap_allocation() {
        let num_heads = 2;
        let head_k_dim = 4;
        let head_v_dim = 4;
        let k = vec![0.2; num_heads * head_k_dim];
        let v = vec![0.3; num_heads * head_v_dim];
        let gate = vec![0.0; num_heads];
        let beta = vec![1.0; num_heads];
        let mut state = vec![0.0f32; num_heads * head_v_dim * head_k_dim];

        COUNT_ALLOC_CALLS.with(|calls| calls.set(0));
        COUNT_ALLOC_ENABLED.with(|enabled| enabled.set(true));
        delta_net_step_state_only(
            &k, &v, &gate, &beta, &mut state, num_heads, head_k_dim, head_v_dim,
        );
        COUNT_ALLOC_ENABLED.with(|enabled| enabled.set(false));

        assert_eq!(
            COUNT_ALLOC_CALLS.with(|calls| calls.get()),
            0,
            "delta_net_step_state_only should not allocate"
        );
    }

    #[test]
    fn test_delta_net_scan_into_avoids_heap_allocation() {
        let num_heads = 2;
        let head_k_dim = 4;
        let head_v_dim = 4;
        let seq_len = 3;
        let q = vec![0.1; seq_len * num_heads * head_k_dim];
        let k = vec![0.2; seq_len * num_heads * head_k_dim];
        let v = vec![0.3; seq_len * num_heads * head_v_dim];
        let gate = vec![0.0; seq_len * num_heads];
        let beta = vec![1.0; seq_len * num_heads];
        let mut state = vec![0.0f32; num_heads * head_v_dim * head_k_dim];
        let mut output = vec![0.0f32; seq_len * num_heads * head_v_dim];

        COUNT_ALLOC_CALLS.with(|calls| calls.set(0));
        COUNT_ALLOC_ENABLED.with(|enabled| enabled.set(true));
        delta_net_scan_into(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut state,
            &mut output,
            seq_len,
            num_heads,
            head_k_dim,
            head_v_dim,
        );
        COUNT_ALLOC_ENABLED.with(|enabled| enabled.set(false));

        assert_eq!(
            COUNT_ALLOC_CALLS.with(|calls| calls.get()),
            0,
            "delta_net_scan_into should not allocate"
        );
    }
}
