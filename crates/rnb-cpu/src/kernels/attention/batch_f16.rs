//! mc73 batch FlashAttention with f16-stored KV + f16 accumulator (native
//! fp16 SIMD path). Opt-in via `RNB_ATTN_F16_NEON=1` after caller verifies
//! `is_aarch64_feature_detected!("fp16")`.
//!
//! Split out from `attention/mod.rs` in mc74 cleanup.

#![cfg(target_arch = "aarch64")]

use super::neon_helpers::*;

/// Batch FlashAttention with f16-stored KV + f16 accumulator — native fp16 SIMD path.
///
/// mc73 0순위. Replaces mc72 `RNB_ATTN_F16_NEON` 의 token-wise prefill dispatch
/// (O(seq_len²) cumulative KV traverse) with a batch BR×BC tile path that keeps
/// cache locality, while running Q×K, P@V, online-softmax rescale on the same
/// native fp16 SIMD ops as the decode `process_head_f16_acc`. Online-softmax
/// max/sum stay in f32 (GGML F16_VEC attention pattern). KV is read straight
/// from `cached_kv_f16` (zero-copy); Q is rounded f32→f16 once per call.
///
/// Caller responsibility: verify `is_aarch64_feature_detected!("fp16")` first.
/// On non-aarch64 or no-FEAT_FP16 cpu, fall back to the f32 batch path.
#[cfg(target_arch = "aarch64")]
pub fn attention_batch_f16(
    q_f32: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> Vec<f32> {
    use rayon::prelude::*;

    debug_assert_eq!(q_f32.len(), seq_len * num_heads * head_dim);
    debug_assert_eq!(k_f16.len(), kv_len * num_kv_heads * head_dim);
    debug_assert_eq!(v_f16.len(), kv_len * num_kv_heads * head_dim);
    debug_assert!(num_kv_heads > 0 && num_heads % num_kv_heads == 0);
    let heads_per_group = num_heads / num_kv_heads;

    // Q is rounded to f16 once per layer/prefill so the dot stays native f16×f16.
    let q_f16: Vec<u16> = q_f32
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    const BR: usize = 32;
    const BC: usize = 32;
    let zero_bits = half::f16::from_f32(0.0).to_bits();

    let flash_head = |h: usize| -> Vec<f32> {
        let kv_h = h / heads_per_group;
        let mut head_out = vec![0.0f32; seq_len * head_dim];

        let mut row_max_buf = [f32::NEG_INFINITY; BR];
        let mut row_sum_buf = [0.0f32; BR];
        let mut acc_buf = vec![zero_bits; BR * head_dim];

        let num_q_tiles = (seq_len + BR - 1) / BR;
        for qi in 0..num_q_tiles {
            let i_start = qi * BR;
            let i_end = (i_start + BR).min(seq_len);
            let tile_rows = i_end - i_start;

            let row_max = &mut row_max_buf[..tile_rows];
            row_max.fill(f32::NEG_INFINITY);
            let row_sum = &mut row_sum_buf[..tile_rows];
            row_sum.fill(0.0);
            let acc = &mut acc_buf[..tile_rows * head_dim];
            acc.fill(zero_bits);

            let earliest_global_pos = (kv_len - seq_len) + i_start;
            let latest_global_pos = (kv_len - seq_len) + (i_end - 1);

            let num_kv_tiles = (kv_len + BC - 1) / BC;
            for kj in 0..num_kv_tiles {
                let j_start = kj * BC;
                let j_end = (j_start + BC).min(kv_len);

                if j_start > latest_global_pos {
                    break;
                }
                let tile_fully_unmasked = j_end <= earliest_global_pos + 1;

                for br in 0..tile_rows {
                    let i = i_start + br;
                    let global_pos = (kv_len - seq_len) + i;
                    let q_off = i * num_heads * head_dim + h * head_dim;
                    let q_row_f16 = &q_f16[q_off..q_off + head_dim];

                    let mut tile_max = f32::NEG_INFINITY;
                    let bc_len = j_end - j_start;
                    let mut scores = [0.0f32; BC];

                    if tile_fully_unmasked && sliding_window.is_none() {
                        for bc in 0..bc_len {
                            let j = j_start + bc;
                            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                            let k_row_f16 = &k_f16[k_off..k_off + head_dim];
                            let dot = unsafe {
                                neon_vec_dot_f16_f16(
                                    q_row_f16.as_ptr(),
                                    k_row_f16.as_ptr(),
                                    head_dim,
                                )
                            };
                            let mut s = dot * scale;
                            if let Some(cap) = softcap {
                                s = cap * (s / cap).tanh();
                            }
                            scores[bc] = s;
                            if s > tile_max {
                                tile_max = s;
                            }
                        }
                    } else {
                        for bc in 0..bc_len {
                            let j = j_start + bc;
                            let masked_by_window = sliding_window
                                .map(|window| j + window <= global_pos)
                                .unwrap_or(false);
                            if j > global_pos || masked_by_window {
                                scores[bc] = f32::NEG_INFINITY;
                            } else {
                                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                                let k_row_f16 = &k_f16[k_off..k_off + head_dim];
                                let dot = unsafe {
                                    neon_vec_dot_f16_f16(
                                        q_row_f16.as_ptr(),
                                        k_row_f16.as_ptr(),
                                        head_dim,
                                    )
                                };
                                let mut s = dot * scale;
                                if let Some(cap) = softcap {
                                    s = cap * (s / cap).tanh();
                                }
                                scores[bc] = s;
                            }
                            if scores[bc] > tile_max {
                                tile_max = scores[bc];
                            }
                        }
                    }

                    if tile_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let old_max = row_max[br];
                    let new_max = if old_max > tile_max {
                        old_max
                    } else {
                        tile_max
                    };
                    let rescale = if old_max == f32::NEG_INFINITY {
                        0.0f32
                    } else {
                        (old_max - new_max).exp()
                    };

                    row_sum[br] *= rescale;
                    let acc_off = br * head_dim;
                    let rescale_bits = half::f16::from_f32(rescale).to_bits();
                    unsafe {
                        neon_vec_scale_f16(
                            acc[acc_off..acc_off + head_dim].as_mut_ptr(),
                            rescale_bits,
                            head_dim,
                        );
                    }

                    let mut tile_sum = 0.0f32;
                    for bc in 0..bc_len {
                        if scores[bc] == f32::NEG_INFINITY {
                            continue;
                        }
                        let p = (scores[bc] - new_max).exp();
                        tile_sum += p;

                        let j = j_start + bc;
                        let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                        let v_row_f16 = &v_f16[v_off..v_off + head_dim];
                        let p_bits = half::f16::from_f32(p).to_bits();
                        unsafe {
                            neon_vec_mad_f16(
                                acc[acc_off..acc_off + head_dim].as_mut_ptr(),
                                v_row_f16.as_ptr(),
                                p_bits,
                                head_dim,
                            );
                        }
                    }

                    row_max[br] = new_max;
                    row_sum[br] += tile_sum;
                }
            }

            // Final normalization: O = acc_f16 / row_sum (f32 lane).
            for br in 0..tile_rows {
                let i = i_start + br;
                let ho_off = i * head_dim;
                let acc_off = br * head_dim;
                let sum = row_sum[br];
                if sum > 0.0 {
                    let inv = 1.0_f32 / sum;
                    for d in 0..head_dim {
                        let acc_f32 = half::f16::from_bits(acc[acc_off + d]).to_f32();
                        head_out[ho_off + d] = acc_f32 * inv;
                    }
                }
            }
        }
        head_out
    };

    let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
    if seq_len > 1 && num_heads >= 4 {
        let head_outs: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| flash_head(h))
            .collect();
        for h in 0..num_heads {
            for i in 0..seq_len {
                let src = &head_outs[h][i * head_dim..(i + 1) * head_dim];
                let dst_off = i * num_heads * head_dim + h * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }
    } else {
        for h in 0..num_heads {
            let head_out = flash_head(h);
            for i in 0..seq_len {
                let src = &head_out[i * head_dim..(i + 1) * head_dim];
                let dst_off = i * num_heads * head_dim + h * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }
    }
    out
}
