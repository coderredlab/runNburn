//! Legacy fp16-fallback batch FlashAttention (BR×BC tile, f32 accumulator).
//!
//! Default path when `RNB_ATTN_F16_NEON` is OFF (production default in
//! mc72/mc73 policy). Split out from `attention/mod.rs` in mc74 cleanup.

use rnb_core::error::{Result, RnbError};
use rnb_core::tensor::Tensor;

use super::super::tensor_as_f32_slice;
use super::dispatch::*;

/// Scaled dot-product attention (CPU, F32, single-batch).
/// Scalar FlashAttention with tiled online softmax.
///
/// q: [seq_len, num_heads * head_dim]
/// k: [kv_len, num_kv_heads * head_dim]
/// v: [kv_len, num_kv_heads * head_dim]
/// 반환: [seq_len, num_heads * head_dim]
///
/// GQA(Grouped Query Attention) 지원:
///   num_heads >= num_kv_heads이면 kv_head를 반복 사용한다 (heads_per_kv_group).
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    attention_with_scale(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_dim,
        1.0_f32 / (head_dim as f32).sqrt(),
    )
}

pub fn attention_with_scale(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Tensor> {
    attention_with_scale_and_window(q, k, v, num_heads, num_kv_heads, head_dim, scale, None)
}

pub fn attention_with_scale_and_window(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    attention_with_scale_window_and_softcap(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window,
        None,
    )
}

pub fn attention_with_scale_window_and_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
        return Err(RnbError::InvalidGraph(
            "attention: q/k/v 텐서는 최소 2D여야 함".to_string(),
        ));
    }

    let seq_len = q_shape[q_shape.len() - 2];
    let kv_len = k_shape[k_shape.len() - 2];

    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(RnbError::InvalidGraph(format!(
            "attention: num_heads({num_heads})는 num_kv_heads({num_kv_heads})의 배수여야 함"
        )));
    }
    let heads_per_group = num_heads / num_kv_heads;

    let q_data = tensor_as_f32_slice(q);
    let k_data = tensor_as_f32_slice(k);
    let v_data = tensor_as_f32_slice(v);

    const BR: usize = 32; // Q tile rows
    const BC: usize = 32; // KV tile cols

    // Per-head FlashAttention kernel — returns [seq_len * head_dim] for one head
    let flash_head = |h: usize| -> Vec<f32> {
        let kv_h = h / heads_per_group;
        let mut head_out = vec![0.0f32; seq_len * head_dim];

        // Stack buffers local to this head
        let mut row_max_buf = [f32::NEG_INFINITY; BR];
        let mut row_sum_buf = [0.0f32; BR];
        let mut acc_buf = vec![0.0f32; BR * head_dim];

        // Process Q in tiles of BR rows
        let num_q_tiles = (seq_len + BR - 1) / BR;
        for qi in 0..num_q_tiles {
            let i_start = qi * BR;
            let i_end = (i_start + BR).min(seq_len);

            // Per-row online softmax state — reset from stack buffers
            let tile_rows = i_end - i_start;
            let row_max = &mut row_max_buf[..tile_rows];
            row_max.fill(f32::NEG_INFINITY);
            let row_sum = &mut row_sum_buf[..tile_rows];
            row_sum.fill(0.0);
            let acc = &mut acc_buf[..tile_rows * head_dim];
            acc.fill(0.0);

            // Process KV in tiles of BC cols
            let num_kv_tiles = (kv_len + BC - 1) / BC;
            // Causal: earliest query in this Q tile can attend up to this position
            let earliest_global_pos = (kv_len - seq_len) + i_start;
            let latest_global_pos = (kv_len - seq_len) + (i_end - 1);

            for kj in 0..num_kv_tiles {
                let j_start = kj * BC;
                let j_end = (j_start + BC).min(kv_len);

                // Tile-level causal skip: if all keys are future for all queries
                if j_start > latest_global_pos {
                    break;
                }
                // Check if tile is fully unmasked (no causal boundary)
                let tile_fully_unmasked = j_end <= earliest_global_pos + 1;

                // For each row in Q tile
                for br in 0..tile_rows {
                    let i = i_start + br;
                    let global_pos = (kv_len - seq_len) + i;
                    let q_off = i * num_heads * head_dim + h * head_dim;
                    let q_row = &q_data[q_off..q_off + head_dim];

                    // Compute scores for this KV tile
                    let mut tile_max = f32::NEG_INFINITY;
                    let bc_len = j_end - j_start;
                    let mut scores = [0.0f32; BC];

                    if tile_fully_unmasked && sliding_window.is_none() {
                        // No causal masking needed — all keys are visible
                        for bc in 0..bc_len {
                            let j = j_start + bc;
                            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                            let k_row = &k_data[k_off..k_off + head_dim];
                            scores[bc] = dot_f32(q_row, k_row, head_dim) * scale;
                            if let Some(cap) = softcap {
                                scores[bc] = cap * (scores[bc] / cap).tanh();
                            }
                            if scores[bc] > tile_max {
                                tile_max = scores[bc];
                            }
                        }
                    } else {
                        // Boundary tile — per-element causal check
                        for bc in 0..bc_len {
                            let j = j_start + bc;
                            let masked_by_window = sliding_window
                                .map(|window| j + window <= global_pos)
                                .unwrap_or(false);
                            if j > global_pos || masked_by_window {
                                scores[bc] = f32::NEG_INFINITY;
                            } else {
                                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                                let k_row = &k_data[k_off..k_off + head_dim];
                                scores[bc] = dot_f32(q_row, k_row, head_dim) * scale;
                                if let Some(cap) = softcap {
                                    scores[bc] = cap * (scores[bc] / cap).tanh();
                                }
                            }
                            if scores[bc] > tile_max {
                                tile_max = scores[bc];
                            }
                        }
                    }

                    // Skip if entire tile is masked
                    if tile_max == f32::NEG_INFINITY {
                        continue;
                    }

                    // Online softmax update
                    let old_max = row_max[br];
                    let new_max = if old_max > tile_max {
                        old_max
                    } else {
                        tile_max
                    };

                    // Rescale old accumulator
                    let rescale = if old_max == f32::NEG_INFINITY {
                        0.0f32
                    } else {
                        (old_max - new_max).exp()
                    };

                    row_sum[br] *= rescale;
                    let acc_off = br * head_dim;
                    scale_f32(&mut acc[acc_off..acc_off + head_dim], rescale);

                    // Compute exp(scores - new_max) and accumulate P @ V
                    let mut tile_sum = 0.0f32;
                    for bc in 0..bc_len {
                        if scores[bc] == f32::NEG_INFINITY {
                            continue;
                        }
                        let p = (scores[bc] - new_max).exp();
                        tile_sum += p;

                        let j = j_start + bc;
                        let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                        let v_row = &v_data[v_off..v_off + head_dim];
                        let acc_off = br * head_dim;
                        scaled_add_f32(&mut acc[acc_off..acc_off + head_dim], v_row, p);
                    }

                    row_max[br] = new_max;
                    row_sum[br] += tile_sum;
                }
            }

            // Final normalization: O = acc / row_sum
            for br in 0..tile_rows {
                let i = i_start + br;
                let ho_off = i * head_dim;
                let acc_off = br * head_dim;
                let sum = row_sum[br];
                if sum > 0.0 {
                    for d in 0..head_dim {
                        head_out[ho_off + d] = acc[acc_off + d] / sum;
                    }
                }
            }
        }
        head_out
    };

    // Dispatch: parallel for prefill (seq_len > 1), sequential for decode
    let out = if seq_len > 1 && num_heads >= 4 {
        use rayon::prelude::*;
        let head_outs: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| flash_head(h))
            .collect();
        // Interleave: head_outs[h][i * head_dim + d] → out[i * num_heads * head_dim + h * head_dim + d]
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        for h in 0..num_heads {
            for i in 0..seq_len {
                let src = &head_outs[h][i * head_dim..(i + 1) * head_dim];
                let dst_off = i * num_heads * head_dim + h * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }
        out
    } else {
        // Sequential
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        for h in 0..num_heads {
            let head_out = flash_head(h);
            for i in 0..seq_len {
                let src = &head_out[i * head_dim..(i + 1) * head_dim];
                let dst_off = i * num_heads * head_dim + h * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }
        out
    };

    Ok(Tensor::from_vec(out, &[seq_len, num_heads * head_dim]))
}
