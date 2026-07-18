//! Q5_K packed GEMM 구현 (row-pair interleaved layout)
//!
//! `gemm_q5k_packed`: packed weight × Q8K input → f32 output
//!
//! 새 레이아웃: unsigned 5-bit (0..31) + raw u8 scales + integer-domain accumulation
//!
//! GEMM 공식:
//!   sumi = Σ(sb=0..7) sc_raw[sb] * dot_i32(w_unsigned[sb*32..+32], x_qs[sb*32..+32])
//!   summ = Σ(sb=0..7) mn_raw[sb] * bsums[sb]
//!   output += x_d * (d * sumi_f32 - dmin * summ_f32)
//!
//! 디스패치 순서:
//!   aarch64 + i8mm    → gemm_q5k_packed_i8mm  (smmla, row-pair interleaved weight 직접 사용)
//!   aarch64 + dotprod → gemm_q5k_packed_neon   (vdotq_s32, deinterleave to stack)
//!   fallback          → gemm_q5k_packed_scalar

use crate::gemm::pack_q5k::{
    Q5K_DMIN_OFF, Q5K_D_OFF, Q5K_MN_RAW_OFF, Q5K_PACKED_BLOCK_BYTES, Q5K_QS_OFF, Q5K_SC_RAW_OFF,
};

/// Q5_K packed GEMM: output = input × weight^T
///
/// # 인자
/// - `packed`: `pack_q5k()`로 생성된 packed weight bytes
/// - `input_qs`: Q8K quantized input, flat `[seq_len * cols * 256]` i8
/// - `input_d`: Q8K scale per block, `[seq_len * cols]` f32
/// - `input_bsums`: Q8K block sums, `[seq_len * cols * 8]` i16
/// - `output`: `[seq_len * rows]` f32, column-major: `output[s * rows + row]`
/// - `rows`: output dimension (weight rows)
/// - `cols`: number of super-blocks (= input_dim / 256)
/// - `seq_len`: number of input tokens
pub fn gemm_q5k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon+dotprod
            unsafe {
                return gemm_q5k_packed_i8mm(
                    packed,
                    input_qs,
                    input_d,
                    input_bsums,
                    output,
                    rows,
                    cols,
                    seq_len,
                );
            }
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod implies neon
            unsafe {
                return gemm_q5k_packed_neon(
                    packed,
                    input_qs,
                    input_d,
                    input_bsums,
                    output,
                    rows,
                    cols,
                    seq_len,
                );
            }
        }
    }

    gemm_q5k_packed_scalar(
        packed,
        input_qs,
        input_d,
        input_bsums,
        output,
        rows,
        cols,
        seq_len,
    );
}

/// Q5_K packed GEMV for decode (seq_len=1).
/// Same packed weight layout, delegates to gemm with seq_len=1.
pub fn gemv_q5k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon.
            unsafe {
                return gemv_q5k_packed_i8mm(
                    packed,
                    input_qs,
                    input_d,
                    input_bsums,
                    output,
                    rows,
                    cols,
                );
            }
        }
    }

    gemm_q5k_packed(
        packed,
        input_qs,
        input_d,
        input_bsums,
        output,
        rows,
        cols,
        1,
    );
}

// ─── 스칼라 레퍼런스 ──────────────────────────────────────────────────────────

fn gemm_q5k_packed_scalar(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    let row_groups = rows.div_ceil(8);

    for s in 0..seq_len {
        for rg in 0..row_groups {
            for bi in 0..cols {
                let blk_off = (rg * cols + bi) * Q5K_PACKED_BLOCK_BYTES;
                let blk = &packed[blk_off..blk_off + Q5K_PACKED_BLOCK_BYTES];

                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d = input_d[s * cols + bi];
                let bsum_base = (s * cols + bi) * 8;
                let x_bsums = &input_bsums[bsum_base..bsum_base + 8];

                for nr in 0..8 {
                    let row = rg * 8 + nr;
                    if row >= rows {
                        break;
                    }

                    // Read row's weight data from interleaved layout
                    let pair = nr / 2;
                    let is_odd = nr % 2;

                    // Read d, dmin, sc_raw, mn_raw
                    let d = f32::from_le_bytes(
                        blk[Q5K_D_OFF + nr * 4..Q5K_D_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let dmin = f32::from_le_bytes(
                        blk[Q5K_DMIN_OFF + nr * 4..Q5K_DMIN_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );

                    let sc_base = Q5K_SC_RAW_OFF + nr * 8;
                    let mn_base = Q5K_MN_RAW_OFF + nr * 8;

                    let mut sumi = 0i32;
                    let mut summ = 0i32;

                    for sb in 0..8usize {
                        let sc_raw = blk[sc_base + sb] as i32;
                        let mn_raw = blk[mn_base + sb] as i32;

                        // Dot product from interleaved qs
                        // 32 elements for this sub-block
                        let mut dot = 0i32;
                        for k in 0..32usize {
                            let elem_idx = sb * 32 + k;
                            let chunk = elem_idx / 8;
                            let within = elem_idx % 8;
                            let qs_off = Q5K_QS_OFF + pair * 512 + chunk * 16 + is_odd * 8 + within;
                            let w = blk[qs_off] as i32;
                            dot += w * (x_qs[elem_idx] as i32);
                        }

                        sumi += dot * sc_raw;
                        summ += mn_raw * (x_bsums[sb] as i32);
                    }

                    output[s * rows + row] += x_d * (d * sumi as f32 - dmin * summ as f32);
                }
            }
        }
    }
}

// ─── NEON vdotq_s32 커널 ─────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q5k_packed_neon(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    let row_groups = rows.div_ceil(8);
    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();

    (0..row_groups).into_par_iter().for_each(|rg| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };

        let rows_in_group = (rows - rg * 8).min(8);

        // Per-token accumulator for 8 rows
        let mut acc = vec![0.0f32; seq_len * 8];

        for bi in 0..cols {
            let blk_off = (rg * cols + bi) * Q5K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q5K_PACKED_BLOCK_BYTES];

            // --- Phase 1: Deinterleave weight to stack [8][256] + load raw scales ---
            let mut w_qs = [[0u8; 256]; 8];
            let mut sc = [[0u8; 8]; 8];
            let mut mn = [[0u8; 8]; 8];
            let mut d = [0.0f32; 8];
            let mut dmin = [0.0f32; 8];

            for nr in 0..8 {
                // Deinterleave qs
                let pair = nr / 2;
                let is_odd = nr % 2;
                let pair_base = Q5K_QS_OFF + pair * 512;
                for k in 0..32usize {
                    let src_off = pair_base + k * 16 + is_odd * 8;
                    w_qs[nr][k * 8..k * 8 + 8].copy_from_slice(&blk[src_off..src_off + 8]);
                }

                // Copy sc_raw, mn_raw
                let sc_off = Q5K_SC_RAW_OFF + nr * 8;
                let mn_off = Q5K_MN_RAW_OFF + nr * 8;
                sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                d[nr] = f32::from_le_bytes(
                    blk[Q5K_D_OFF + nr * 4..Q5K_D_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                dmin[nr] = f32::from_le_bytes(
                    blk[Q5K_DMIN_OFF + nr * 4..Q5K_DMIN_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
            }

            // --- Phase 2: For each token, compute 8 rows ---
            for s in 0..seq_len {
                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d = input_d[s * cols + bi];
                let bsum_base = (s * cols + bi) * 8;
                let x_bsums = &input_bsums[bsum_base..bsum_base + 8];

                for r in 0..rows_in_group {
                    let w = w_qs[r].as_ptr() as *const i8;
                    let vzero = vdupq_n_s32(0);
                    let mut acc_v = vzero;
                    let mut summ = 0i32;

                    // 8 sub-blocks × 32 elements, integer-domain scale
                    for sb in 0..8usize {
                        let off = sb * 32;
                        // 32 elements: 2×16B vdotq
                        let w0 = vld1q_s8(w.add(off));
                        let w1 = vld1q_s8(w.add(off + 16));
                        let x0 = vld1q_s8(x_qs[off..].as_ptr());
                        let x1 = vld1q_s8(x_qs[off + 16..].as_ptr());

                        let dot4 = vdotq_s32(vdotq_s32(vzero, w0, x0), w1, x1);
                        // multiply by raw scale in integer domain
                        acc_v = vmlaq_n_s32(acc_v, dot4, sc[r][sb] as i32);

                        summ += mn[r][sb] as i32 * x_bsums[sb] as i32;
                    }

                    let sumi = vaddvq_s32(acc_v);
                    acc[s * 8 + r] += x_d * (d[r] * sumi as f32 - dmin[r] * summ as f32);
                }
            }
        }

        // Write output
        for s in 0..seq_len {
            for r in 0..rows_in_group {
                out_slice[s * rows + rg * 8 + r] += acc[s * 8 + r];
            }
        }
    });
}

// ─── i8mm smmla 커널 ─────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_q5k_packed_i8mm(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    let row_groups = rows.div_ceil(8);
    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();

    (0..row_groups).into_par_iter().for_each(|rg| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };

        let rows_in_group = (rows - rg * 8).min(8);
        let row_pairs = (rows_in_group + 1) / 2;

        // Per-token accumulator for 8 rows
        let mut acc = vec![0.0f32; seq_len * 8];

        for bi in 0..cols {
            let blk_off = (rg * cols + bi) * Q5K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q5K_PACKED_BLOCK_BYTES];
            let blk_ptr = blk.as_ptr() as *const i8;

            // --- Phase 1: Load raw scales/mins/d/dmin to stack ---
            let mut sc = [[0u8; 8]; 8];
            let mut mn = [[0u8; 8]; 8];
            let mut d = [0.0f32; 8];
            let mut dmin = [0.0f32; 8];

            for nr in 0..8 {
                let sc_off = Q5K_SC_RAW_OFF + nr * 8;
                let mn_off = Q5K_MN_RAW_OFF + nr * 8;
                sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                d[nr] = f32::from_le_bytes(
                    blk[Q5K_D_OFF + nr * 4..Q5K_D_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                dmin[nr] = f32::from_le_bytes(
                    blk[Q5K_DMIN_OFF + nr * 4..Q5K_DMIN_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
            }

            // --- Phase 2: Tiered token batching ---
            // 8-token → 2-token → 1-token fallback.
            let mut s = 0usize;

            // ── 8-token path: process 4 token-pairs sharing weight loads ──
            while s + 7 < seq_len {
                let mut t_qs: [&[i8]; 8] = [&[]; 8];
                let mut t_d = [0.0f32; 8];
                let mut t_bsums: [&[i16]; 8] = [&[]; 8];
                for t in 0..8 {
                    let st = s + t;
                    let inp_base = (st * cols + bi) * 256;
                    t_qs[t] = &input_qs[inp_base..inp_base + 256];
                    t_d[t] = input_d[st * cols + bi];
                    let bsum_base = (st * cols + bi) * 8;
                    t_bsums[t] = &input_bsums[bsum_base..bsum_base + 8];
                }

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    let mut dots = [[[0i32; 8]; 2]; 8];

                    let w_pair_base = Q5K_QS_OFF + rp * 512;

                    for sb in 0..8usize {
                        let k_off = sb * 32;

                        let mut acc0 = vdupq_n_s32(0);
                        let mut acc1 = vdupq_n_s32(0);
                        let mut acc2 = vdupq_n_s32(0);
                        let mut acc3 = vdupq_n_s32(0);

                        for ki in 0..4usize {
                            let elem_off = k_off + ki * 8;
                            let chunk = elem_off / 8;
                            let w_off = w_pair_base + chunk * 16;

                            let w = vld1q_s8(blk_ptr.add(w_off));

                            let x01 = vcombine_s8(
                                vld1_s8(t_qs[0][elem_off..].as_ptr()),
                                vld1_s8(t_qs[1][elem_off..].as_ptr()),
                            );
                            let x23 = vcombine_s8(
                                vld1_s8(t_qs[2][elem_off..].as_ptr()),
                                vld1_s8(t_qs[3][elem_off..].as_ptr()),
                            );
                            let x45 = vcombine_s8(
                                vld1_s8(t_qs[4][elem_off..].as_ptr()),
                                vld1_s8(t_qs[5][elem_off..].as_ptr()),
                            );
                            let x67 = vcombine_s8(
                                vld1_s8(t_qs[6][elem_off..].as_ptr()),
                                vld1_s8(t_qs[7][elem_off..].as_ptr()),
                            );

                            acc0 = vmmlaq_s32(acc0, w, x01);
                            acc1 = vmmlaq_s32(acc1, w, x23);
                            acc2 = vmmlaq_s32(acc2, w, x45);
                            acc3 = vmmlaq_s32(acc3, w, x67);
                        }

                        dots[0][0][sb] = vgetq_lane_s32::<0>(acc0);
                        dots[1][0][sb] = vgetq_lane_s32::<1>(acc0);
                        dots[2][0][sb] = vgetq_lane_s32::<0>(acc1);
                        dots[3][0][sb] = vgetq_lane_s32::<1>(acc1);
                        dots[4][0][sb] = vgetq_lane_s32::<0>(acc2);
                        dots[5][0][sb] = vgetq_lane_s32::<1>(acc2);
                        dots[6][0][sb] = vgetq_lane_s32::<0>(acc3);
                        dots[7][0][sb] = vgetq_lane_s32::<1>(acc3);

                        if has_r1 {
                            dots[0][1][sb] = vgetq_lane_s32::<2>(acc0);
                            dots[1][1][sb] = vgetq_lane_s32::<3>(acc0);
                            dots[2][1][sb] = vgetq_lane_s32::<2>(acc1);
                            dots[3][1][sb] = vgetq_lane_s32::<3>(acc1);
                            dots[4][1][sb] = vgetq_lane_s32::<2>(acc2);
                            dots[5][1][sb] = vgetq_lane_s32::<3>(acc2);
                            dots[6][1][sb] = vgetq_lane_s32::<2>(acc3);
                            dots[7][1][sb] = vgetq_lane_s32::<3>(acc3);
                        }
                    }

                    let sc_r0_u8 = vld1_u8(sc[r0].as_ptr());
                    let sc_r0_u16 = vmovl_u8(sc_r0_u8);
                    let sc_r0_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc_r0_u16)));
                    let sc_r0_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc_r0_u16)));

                    let mn_r0_u8 = vld1_u8(mn[r0].as_ptr());
                    let mn_r0_u16 = vmovl_u8(mn_r0_u8);
                    let mn_r0_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mn_r0_u16)));
                    let mn_r0_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mn_r0_u16)));

                    for t in 0..8 {
                        let d_lo = vld1q_s32(dots[t][0][0..4].as_ptr());
                        let d_hi = vld1q_s32(dots[t][0][4..8].as_ptr());
                        let sumi_val = vaddvq_s32(vaddq_s32(
                            vmulq_s32(d_lo, sc_r0_lo),
                            vmulq_s32(d_hi, sc_r0_hi),
                        ));

                        let bs = vld1q_s16(t_bsums[t].as_ptr());
                        let bs_lo = vmovl_s16(vget_low_s16(bs));
                        let bs_hi = vmovl_s16(vget_high_s16(bs));
                        let summ_val = vaddvq_s32(vaddq_s32(
                            vmulq_s32(mn_r0_lo, bs_lo),
                            vmulq_s32(mn_r0_hi, bs_hi),
                        ));

                        acc[(s + t) * 8 + r0] +=
                            t_d[t] * (d[r0] * sumi_val as f32 - dmin[r0] * summ_val as f32);
                    }

                    if has_r1 {
                        let sc_r1_u8 = vld1_u8(sc[r1].as_ptr());
                        let sc_r1_u16 = vmovl_u8(sc_r1_u8);
                        let sc_r1_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc_r1_u16)));
                        let sc_r1_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc_r1_u16)));

                        let mn_r1_u8 = vld1_u8(mn[r1].as_ptr());
                        let mn_r1_u16 = vmovl_u8(mn_r1_u8);
                        let mn_r1_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mn_r1_u16)));
                        let mn_r1_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mn_r1_u16)));

                        for t in 0..8 {
                            let d_lo = vld1q_s32(dots[t][1][0..4].as_ptr());
                            let d_hi = vld1q_s32(dots[t][1][4..8].as_ptr());
                            let sumi_val = vaddvq_s32(vaddq_s32(
                                vmulq_s32(d_lo, sc_r1_lo),
                                vmulq_s32(d_hi, sc_r1_hi),
                            ));

                            let bs = vld1q_s16(t_bsums[t].as_ptr());
                            let bs_lo = vmovl_s16(vget_low_s16(bs));
                            let bs_hi = vmovl_s16(vget_high_s16(bs));
                            let summ_val = vaddvq_s32(vaddq_s32(
                                vmulq_s32(mn_r1_lo, bs_lo),
                                vmulq_s32(mn_r1_hi, bs_hi),
                            ));

                            acc[(s + t) * 8 + r1] +=
                                t_d[t] * (d[r1] * sumi_val as f32 - dmin[r1] * summ_val as f32);
                        }
                    }
                }

                s += 8;
            }

            // ── 2-token path: remainder after 8-token batching ──
            while s + 1 < seq_len {
                let inp_base0 = (s * cols + bi) * 256;
                let inp_base1 = ((s + 1) * cols + bi) * 256;
                let x_qs0 = &input_qs[inp_base0..inp_base0 + 256];
                let x_qs1 = &input_qs[inp_base1..inp_base1 + 256];
                let x_d0 = input_d[s * cols + bi];
                let x_d1 = input_d[(s + 1) * cols + bi];
                let bsum_base0 = (s * cols + bi) * 8;
                let bsum_base1 = ((s + 1) * cols + bi) * 8;
                let x_bsums0 = &input_bsums[bsum_base0..bsum_base0 + 8];
                let x_bsums1 = &input_bsums[bsum_base1..bsum_base1 + 8];

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    let mut sumi_r0_s0 = 0i32;
                    let mut sumi_r0_s1 = 0i32;
                    let mut sumi_r1_s0 = 0i32;
                    let mut sumi_r1_s1 = 0i32;
                    let mut summ_r0_s0 = 0i32;
                    let mut summ_r0_s1 = 0i32;
                    let mut summ_r1_s0 = 0i32;
                    let mut summ_r1_s1 = 0i32;

                    let w_pair_base = Q5K_QS_OFF + rp * 512;

                    for sb in 0..8usize {
                        let k_off = sb * 32;

                        let mut smmla_acc = vdupq_n_s32(0);

                        for ki in 0..4usize {
                            let elem_off = k_off + ki * 8;
                            let chunk = elem_off / 8;
                            let w_off = w_pair_base + chunk * 16;

                            let w = vld1q_s8(blk_ptr.add(w_off));

                            let xs0 = vld1_s8(x_qs0[elem_off..].as_ptr());
                            let xs1 = vld1_s8(x_qs1[elem_off..].as_ptr());
                            let x_pair = vcombine_s8(xs0, xs1);

                            smmla_acc = vmmlaq_s32(smmla_acc, w, x_pair);
                        }

                        let dot_r0_s0 = vgetq_lane_s32::<0>(smmla_acc);
                        let dot_r0_s1 = vgetq_lane_s32::<1>(smmla_acc);
                        let dot_r1_s0 = vgetq_lane_s32::<2>(smmla_acc);
                        let dot_r1_s1 = vgetq_lane_s32::<3>(smmla_acc);

                        let sc_r0 = sc[r0][sb] as i32;
                        sumi_r0_s0 += dot_r0_s0 * sc_r0;
                        sumi_r0_s1 += dot_r0_s1 * sc_r0;
                        summ_r0_s0 += mn[r0][sb] as i32 * x_bsums0[sb] as i32;
                        summ_r0_s1 += mn[r0][sb] as i32 * x_bsums1[sb] as i32;

                        if has_r1 {
                            let sc_r1 = sc[r1][sb] as i32;
                            sumi_r1_s0 += dot_r1_s0 * sc_r1;
                            sumi_r1_s1 += dot_r1_s1 * sc_r1;
                            summ_r1_s0 += mn[r1][sb] as i32 * x_bsums0[sb] as i32;
                            summ_r1_s1 += mn[r1][sb] as i32 * x_bsums1[sb] as i32;
                        }
                    }

                    acc[s * 8 + r0] +=
                        x_d0 * (d[r0] * sumi_r0_s0 as f32 - dmin[r0] * summ_r0_s0 as f32);
                    acc[(s + 1) * 8 + r0] +=
                        x_d1 * (d[r0] * sumi_r0_s1 as f32 - dmin[r0] * summ_r0_s1 as f32);

                    if has_r1 {
                        acc[s * 8 + r1] +=
                            x_d0 * (d[r1] * sumi_r1_s0 as f32 - dmin[r1] * summ_r1_s0 as f32);
                        acc[(s + 1) * 8 + r1] +=
                            x_d1 * (d[r1] * sumi_r1_s1 as f32 - dmin[r1] * summ_r1_s1 as f32);
                    }
                }

                s += 2;
            }

            // ── 1-token remainder (odd seq_len) ──
            if s < seq_len {
                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d_val = input_d[s * cols + bi];
                let bsum_base = (s * cols + bi) * 8;
                let x_bsums = &input_bsums[bsum_base..bsum_base + 8];

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    let mut sumi_r0 = 0i32;
                    let mut sumi_r1 = 0i32;
                    let mut summ_r0 = 0i32;
                    let mut summ_r1 = 0i32;

                    let w_pair_base = Q5K_QS_OFF + rp * 512;

                    for sb in 0..8usize {
                        let k_off = sb * 32;
                        let mut smmla_acc = vdupq_n_s32(0);

                        for ki in 0..4usize {
                            let elem_off = k_off + ki * 8;
                            let chunk = elem_off / 8;
                            let w_off = w_pair_base + chunk * 16;

                            let w = vld1q_s8(blk_ptr.add(w_off));
                            let xs = vld1_s8(x_qs[elem_off..].as_ptr());
                            let x_padded = vcombine_s8(xs, vdup_n_s8(0));

                            smmla_acc = vmmlaq_s32(smmla_acc, w, x_padded);
                        }

                        let dot_r0 = vgetq_lane_s32::<0>(smmla_acc);
                        let dot_r1 = vgetq_lane_s32::<2>(smmla_acc);

                        sumi_r0 += dot_r0 * sc[r0][sb] as i32;
                        summ_r0 += mn[r0][sb] as i32 * x_bsums[sb] as i32;

                        if has_r1 {
                            sumi_r1 += dot_r1 * sc[r1][sb] as i32;
                            summ_r1 += mn[r1][sb] as i32 * x_bsums[sb] as i32;
                        }
                    }

                    acc[s * 8 + r0] +=
                        x_d_val * (d[r0] * sumi_r0 as f32 - dmin[r0] * summ_r0 as f32);

                    if has_r1 {
                        acc[s * 8 + r1] +=
                            x_d_val * (d[r1] * sumi_r1 as f32 - dmin[r1] * summ_r1 as f32);
                    }
                }
            }
        }

        // Write output
        for s in 0..seq_len {
            for r in 0..rows_in_group {
                out_slice[s * rows + rg * 8 + r] += acc[s * 8 + r];
            }
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemv_q5k_packed_i8mm(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    const ROW_GROUPS_PER_TASK: usize = 16;

    let row_groups = rows.div_ceil(8);
    let output = &mut output[..rows];

    output
        .par_chunks_mut(ROW_GROUPS_PER_TASK * 8)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let rg_start = chunk_idx * ROW_GROUPS_PER_TASK;
            let rg_end = (rg_start + ROW_GROUPS_PER_TASK).min(row_groups);

            for rg in rg_start..rg_end {
                let local_rg = rg - rg_start;
                let local_base = local_rg * 8;
                let rows_in_group = (rows - rg * 8).min(8);
                let row_pairs = (rows_in_group + 1) / 2;

                let mut acc = [0.0f32; 8];

                for bi in 0..cols {
                    let blk_off = (rg * cols + bi) * Q5K_PACKED_BLOCK_BYTES;
                    let blk = &packed[blk_off..blk_off + Q5K_PACKED_BLOCK_BYTES];
                    let blk_ptr = blk.as_ptr() as *const i8;

                    let mut sc = [[0u8; 8]; 8];
                    let mut mn = [[0u8; 8]; 8];
                    let mut d = [0.0f32; 8];
                    let mut dmin = [0.0f32; 8];

                    for nr in 0..rows_in_group {
                        let sc_off = Q5K_SC_RAW_OFF + nr * 8;
                        let mn_off = Q5K_MN_RAW_OFF + nr * 8;
                        sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                        mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                        d[nr] = f32::from_le_bytes(
                            blk[Q5K_D_OFF + nr * 4..Q5K_D_OFF + nr * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        dmin[nr] = f32::from_le_bytes(
                            blk[Q5K_DMIN_OFF + nr * 4..Q5K_DMIN_OFF + nr * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                    }

                    let x_qs = &input_qs[bi * 256..bi * 256 + 256];
                    let x_d_val = input_d[bi];
                    let x_bsums = &input_bsums[bi * 8..bi * 8 + 8];

                    for rp in 0..row_pairs {
                        let r0 = rp * 2;
                        let r1 = r0 + 1;
                        let has_r1 = r1 < rows_in_group;

                        let mut sumi_r0 = 0i32;
                        let mut sumi_r1 = 0i32;
                        let mut summ_r0 = 0i32;
                        let mut summ_r1 = 0i32;

                        let w_pair_base = Q5K_QS_OFF + rp * 512;

                        for sb in 0..8usize {
                            let k_off = sb * 32;
                            let mut smmla_acc = vdupq_n_s32(0);

                            for ki in 0..4usize {
                                let elem_off = k_off + ki * 8;
                                let chunk = elem_off / 8;
                                let w_off = w_pair_base + chunk * 16;

                                let w = vld1q_s8(blk_ptr.add(w_off));
                                let xs = vld1_s8(x_qs[elem_off..].as_ptr());
                                let x_padded = vcombine_s8(xs, vdup_n_s8(0));

                                smmla_acc = vmmlaq_s32(smmla_acc, w, x_padded);
                            }

                            let dot_r0 = vgetq_lane_s32::<0>(smmla_acc);
                            let dot_r1 = vgetq_lane_s32::<2>(smmla_acc);

                            sumi_r0 += dot_r0 * sc[r0][sb] as i32;
                            summ_r0 += mn[r0][sb] as i32 * x_bsums[sb] as i32;

                            if has_r1 {
                                sumi_r1 += dot_r1 * sc[r1][sb] as i32;
                                summ_r1 += mn[r1][sb] as i32 * x_bsums[sb] as i32;
                            }
                        }

                        acc[r0] += x_d_val * (d[r0] * sumi_r0 as f32 - dmin[r0] * summ_r0 as f32);

                        if has_r1 {
                            acc[r1] +=
                                x_d_val * (d[r1] * sumi_r1 as f32 - dmin[r1] * summ_r1 as f32);
                        }
                    }
                }

                for r in 0..rows_in_group {
                    out_chunk[local_base + r] += acc[r];
                }
            }
        });
}

// ─── 테스트 ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::pack_q4k::decode_q4k_scales;
    use crate::gemm::pack_q5k::pack_q5k;
    use half::f16;

    // ─── 테스트 헬퍼 ─────────────────────────────────────────────

    /// Q5_K 더미 블록 생성 (176 bytes)
    fn make_q5k_block(
        d_val: f32,
        dmin_val: f32,
        scales: [u8; 12],
        qh: [u8; 32],
        qs: [u8; 128],
    ) -> Vec<u8> {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..48].copy_from_slice(&qh);
        block[48..176].copy_from_slice(&qs);
        block
    }

    /// Q8K 수동 양자화: f32[256] → (qs: i8[256], d: f32, bsums: i16[8])
    fn quantize_q8k(x: &[f32; 256]) -> ([i8; 256], f32, [i16; 8]) {
        let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv_d = 1.0 / d;

        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 8];

        for (i, &v) in x.iter().enumerate() {
            let q = (v * inv_d).round().clamp(-128.0, 127.0) as i8;
            qs[i] = q;
            bsums[i / 32] += q as i16;
        }

        (qs, d, bsums)
    }

    /// Q5_K 블록을 f32[256]으로 dequant
    ///
    /// val = scale[sb] * val_unsigned - min[sb]
    fn dequant_q5k_block(block: &[u8]) -> [f32; 256] {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales_raw: &[u8; 12] = block[4..16].try_into().unwrap();
        let qh_raw: &[u8; 32] = block[16..48].try_into().unwrap();
        let qs_raw: &[u8; 128] = block[48..176].try_into().unwrap();

        let (sc_f32, mn_f32) = decode_q4k_scales(scales_raw, d, dmin);

        let mut out = [0.0f32; 256];
        for g in 0..4usize {
            let sb0 = g * 2;
            let sb1 = g * 2 + 1;
            let group_out = g * 64;
            let group_qs = g * 32;

            // l < 32: high bit at qh[l] bit (2*g) per GGML spec
            for l in 0..32usize {
                let low = (qs_raw[group_qs + l] & 0x0F) as u32;
                let high_bit = ((qh_raw[l] >> (2 * g)) & 1) as u32;
                let val_u = low | (high_bit << 4);
                out[group_out + l] = sc_f32[sb0] * val_u as f32 - mn_f32[sb0];
            }

            // l >= 32: high bit at qh[l-32] bit (2*g+1) per GGML spec
            for l in 32..64usize {
                let low = (qs_raw[group_qs + (l - 32)] >> 4) as u32;
                let high_bit = ((qh_raw[l - 32] >> (2 * g + 1)) & 1) as u32;
                let val_u = low | (high_bit << 4);
                out[group_out + l] = sc_f32[sb1] * val_u as f32 - mn_f32[sb1];
            }
        }
        out
    }

    fn q5k_q8k_exact_ref(
        src: &[u8],
        input_qs: &[i8],
        input_d: &[f32],
        input_bsums: &[i16],
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * rows];

        for s in 0..seq_len {
            for row in 0..rows {
                let mut acc = 0.0f32;
                for bi in 0..cols {
                    let src_off = (row * cols + bi) * 176;
                    let block = &src[src_off..src_off + 176];
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                    let qh_raw: &[u8; 32] = block[16..48].try_into().unwrap();
                    let qs_raw: &[u8; 128] = block[48..176].try_into().unwrap();
                    let (sc_raw, mn_raw) = crate::gemm::pack_q4k::decode_q4k_scales_raw(scales_12);

                    let mut w_unsigned = [0u8; 256];
                    crate::gemm::pack_q5k::unpack_q5k_bits_unsigned(
                        qs_raw,
                        qh_raw,
                        &mut w_unsigned,
                    );

                    let x_off = (s * cols + bi) * 256;
                    let x_qs = &input_qs[x_off..x_off + 256];
                    let x_d = input_d[s * cols + bi];
                    let bs_off = (s * cols + bi) * 8;
                    let x_bsums = &input_bsums[bs_off..bs_off + 8];

                    let mut sumi = 0i32;
                    let mut summ = 0i32;
                    for sb in 0..8usize {
                        let mut dot = 0i32;
                        for k in 0..32usize {
                            let idx = sb * 32 + k;
                            dot += w_unsigned[idx] as i32 * x_qs[idx] as i32;
                        }
                        sumi += sc_raw[sb] as i32 * dot;
                        summ += mn_raw[sb] as i32 * x_bsums[sb] as i32;
                    }

                    acc += x_d * (d * sumi as f32 - dmin * summ as f32);
                }
                out[s * rows + row] = acc;
            }
        }

        out
    }

    // ─── 테스트 1: 제로 입력 ─────────────────────────────────────

    #[test]
    fn test_gemm_zero_input() {
        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let block = make_q5k_block(1.0, 0.5, [10u8; 12], [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q5k(&src, rows, cols);

        let input_qs = vec![0i8; seq_len * cols * 256];
        let input_d = vec![1.0f32; seq_len * cols];
        let input_bsums = vec![0i16; seq_len * cols * 8];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q5k_packed(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        // qs=0이면 dot=0이고 bsums=0이면 min 보정도 0 → output 전부 0
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] = {v}, expected 0");
        }
    }

    // ─── 테스트 2: bsums만 있는 경우 (min bias) ──────────────────

    #[test]
    fn test_gemm_min_bias_only() {
        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        // scales=0 → mn_raw=0, sc_raw=0
        let block = make_q5k_block(1.0, 1.0, [0u8; 12], [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q5k(&src, rows, cols);

        let input_qs = vec![0i8; seq_len * cols * 256];
        let input_d = vec![1.0f32; seq_len * cols];
        let input_bsums = vec![100i16; seq_len * cols * 8];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q5k_packed(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        // sc_raw=0이면 sumi=0, mn_raw=0이면 summ=0 → output 전부 0
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] = {v}, expected 0 (mn=0)");
        }
    }

    // ─── 테스트 3: 수동 계산 크로스체크 ─────────────────────────

    #[test]
    fn test_gemm_single_nonzero() {
        // rows=8, cols=1
        // qs=0x88 (unsigned 8, qh=0), d=1.0, dmin=0.0
        // scales[0]=10 → sc_raw[0]=10, mn_raw[0]=0
        //
        // input_qs: sub-block 0 원소 모두 1 (32개), 나머지 0
        //   bsums[0] = 32, bsums[1..] = 0
        //   x_d = 1.0
        //
        // 새 수식 (unsigned):
        //   sumi = sc_raw[0] * dot(w_unsigned=8, x=1, 32개) = 10 * (8*32) = 2560
        //   summ = mn_raw[0] * bsum[0] = 0 * 32 = 0
        //   out = x_d * (d * sumi - dmin * summ) = 1.0 * (1.0 * 2560 - 0) = 2560

        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        let block = make_q5k_block(1.0, 0.0, scales_raw, [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q5k(&src, rows, cols);

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        for k in 0..32 {
            input_qs[k] = 1;
        }
        let input_d = vec![1.0f32; seq_len * cols];
        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        input_bsums[0] = 32;

        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q5k_packed(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        let expected = 2560.0f32;
        for row in 0..rows {
            let v = output[row];
            assert!(
                (v - expected).abs() < 0.1,
                "output[{row}] = {v}, expected {expected}"
            );
        }
    }

    // ─── 테스트 4: f32 레퍼런스 크로스체크 ───────────────────────

    #[test]
    fn test_gemm_crosscheck_f32_ref() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 5;
        scales_raw[4] = 3;

        let mut qs_data = [0x88u8; 128];
        for i in 0..32usize {
            let nibble = (i % 9) as u8;
            qs_data[i] = (qs_data[i] & 0xF0) | nibble;
        }

        // qh에 절반만 set
        let mut qh_data = [0x00u8; 32];
        for i in 0..16usize {
            qh_data[i] = 0xAA; // 비트 1,3,5,7 set
        }

        let block = make_q5k_block(0.01, 0.005, scales_raw, qh_data, qs_data);
        let src: Vec<u8> = block.repeat(rows * cols);
        let packed = pack_q5k(&src, rows, cols);

        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 64.0).sin() * 0.5;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];
        let mut input_bsums_flat = vec![0i16; total_tokens * 8];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d, bsums) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
            input_bsums_flat[t * 8..(t + 1) * 8].copy_from_slice(&bsums);
        }

        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q5k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        // f32 reference — Q8K 양자화된 입력 사용 (GEMM과 동등 조건)
        let mut output_ref = vec![0.0f32; seq_len * rows];
        for s in 0..seq_len {
            for row in 0..rows {
                let mut total = 0.0f32;
                for bi in 0..cols {
                    let src_off = (row * cols + bi) * 176;
                    let w_blk = dequant_q5k_block(&src[src_off..src_off + 176]);

                    let inp_base = (s * cols + bi) * 256;
                    let x_qs = &input_qs_flat[inp_base..inp_base + 256];
                    let x_d = input_d_flat[s * cols + bi];

                    let mut dot = 0.0f32;
                    for k in 0..256usize {
                        dot += w_blk[k] * (x_qs[k] as f32 * x_d);
                    }
                    total += dot;
                }
                output_ref[s * rows + row] = total;
            }
        }

        for s in 0..seq_len {
            for row in 0..rows {
                let got = output_gemm[s * rows + row];
                let exp = output_ref[s * rows + row];
                let abs_err = (got - exp).abs();
                let rel_err = if exp.abs() > 1e-6 {
                    abs_err / exp.abs()
                } else {
                    abs_err
                };
                assert!(
                    rel_err < 0.15 || abs_err < 0.1,
                    "s={s} row={row}: got={got:.6}, ref={exp:.6}, abs_err={abs_err:.6}, rel_err={rel_err:.4}"
                );
            }
        }
    }

    // ─── 테스트 5: seq_len > 1 독립성 ───────────────────────────

    #[test]
    fn test_gemm_seq_len_independence() {
        let rows = 8;
        let cols = 1;
        let seq_len = 2;

        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 5;
        // qs=0x88 (unsigned 8), qh=0, d=1.0, dmin=0.0
        let block = make_q5k_block(1.0, 0.0, scales_raw, [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q5k(&src, rows, cols);

        // token 0: sub-block 0에 x_qs=1, token 1: sub-block 0에 x_qs=-1
        let mut input_qs = vec![0i8; seq_len * cols * 256];
        for k in 0..32 {
            input_qs[k] = 1; // token 0
            input_qs[256 + k] = -1; // token 1
        }
        let input_d = vec![1.0f32; seq_len * cols];

        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        input_bsums[0] = 32; // token 0, sub-block 0
        input_bsums[8] = -32; // token 1, sub-block 0

        let mut output = vec![0.0f32; seq_len * rows];
        gemm_q5k_packed(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        // sumi token 0: sc_raw=5, dot=8*32=256, sumi=5*256=1280
        // out = 1.0 * (1.0 * 1280 - 0) = 1280
        // sumi token 1: sc_raw=5, dot=8*(-32)=-256, sumi=5*(-256)=-1280
        // out = 1.0 * (1.0 * (-1280) - 0) = -1280
        let expected0 = 1280.0f32;

        for row in 0..rows {
            let v0 = output[row];
            let v1 = output[rows + row];
            assert!(
                (v0 + v1).abs() < 1e-4,
                "row={row}: v0={v0}, v1={v1} should be negatives of each other"
            );
            assert!(
                (v0 - expected0).abs() < 0.1,
                "row={row}: v0={v0}, expected {expected0}"
            );
        }
    }

    // ─── 테스트 6: unique rows (행마다 다른 weight) ──────────────

    #[test]
    fn test_gemm_unique_rows() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

        // 각 row, 각 col마다 DIFFERENT Q5_K 블록 생성
        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let seed = (row * cols + col) as u8;
                let d_val = 0.005 + 0.002 * seed as f32;
                let dmin_val = 0.001 + 0.001 * seed as f32;

                // 각 row마다 다른 scales
                let mut scales = [0u8; 12];
                for i in 0..12 {
                    scales[i] = ((seed as u16 * 7 + i as u16 * 3) % 63) as u8;
                }

                // 각 row마다 다른 qh (5번째 비트)
                let mut qh = [0u8; 32];
                for i in 0..32 {
                    qh[i] = ((seed as u16 * 11 + i as u16 * 13) % 256) as u8;
                }

                // 각 row마다 다른 qs nibbles (0..15 unsigned)
                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }

                let block = make_q5k_block(d_val, dmin_val, scales, qh, qs);
                src.extend_from_slice(&block);
            }
        }

        let packed = pack_q5k(&src, rows, cols);

        // 다양한 입력 생성
        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 37.0).sin() * 0.8 + ((i as f32) * PI / 97.0).cos() * 0.3;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];
        let mut input_bsums_flat = vec![0i16; total_tokens * 8];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d, bsums) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
            input_bsums_flat[t * 8..(t + 1) * 8].copy_from_slice(&bsums);
        }

        // GEMM 실행
        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q5k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        // f32 reference — Q8K 양자화된 입력 사용 (GEMM과 동등 조건)
        // dequant weight를 Q8K 양자화된 입력과 곱해야 공정한 비교가 됨
        let mut output_ref = vec![0.0f32; seq_len * rows];
        for s in 0..seq_len {
            for row in 0..rows {
                let mut total = 0.0f32;
                for bi in 0..cols {
                    let src_off = (row * cols + bi) * 176;
                    let w_blk = dequant_q5k_block(&src[src_off..src_off + 176]);

                    // Q8K 양자화된 입력으로 dot product
                    let inp_base = (s * cols + bi) * 256;
                    let x_qs = &input_qs_flat[inp_base..inp_base + 256];
                    let x_d = input_d_flat[s * cols + bi];

                    let mut dot = 0.0f32;
                    for k in 0..256usize {
                        // dequant된 x = x_qs[k] * x_d
                        dot += w_blk[k] * (x_qs[k] as f32 * x_d);
                    }
                    total += dot;
                }
                output_ref[s * rows + row] = total;
            }
        }

        // 비교: 행마다 다른 weight이므로 interleaving 버그가 있으면 여기서 잡힘
        let mut max_rel_err = 0.0f32;
        let mut fail_count = 0;
        for s in 0..seq_len {
            for row in 0..rows {
                let got = output_gemm[s * rows + row];
                let exp = output_ref[s * rows + row];
                let abs_err = (got - exp).abs();
                let rel_err = if exp.abs() > 1e-6 {
                    abs_err / exp.abs()
                } else {
                    abs_err
                };
                if rel_err > max_rel_err {
                    max_rel_err = rel_err;
                }
                if rel_err >= 0.10 && abs_err >= 0.05 {
                    eprintln!(
                        "  HIGH ERR s={s} row={row}: got={got:.6}, ref={exp:.6}, \
                         abs_err={abs_err:.6}, rel_err={rel_err:.4}"
                    );
                    fail_count += 1;
                }
                assert!(
                    rel_err < 0.20 || abs_err < 0.15,
                    "UNIQUE ROW BUG: s={s} row={row}: got={got:.6}, ref={exp:.6}, \
                     abs_err={abs_err:.6}, rel_err={rel_err:.4}"
                );
            }
        }

        // 각 row의 결과가 서로 다른지도 확인 (같으면 interleaving이 무시되는 버그)
        for s in 0..seq_len {
            let mut all_same = true;
            let first = output_gemm[s * rows];
            for row in 1..rows {
                if (output_gemm[s * rows + row] - first).abs() > 0.01 {
                    all_same = false;
                    break;
                }
            }
            assert!(
                !all_same,
                "s={s}: all rows produced the same output — interleaving is broken"
            );
        }

        eprintln!(
            "test_gemm_unique_rows: max_rel_err = {max_rel_err:.6}, high_err_count = {fail_count}"
        );
    }

    #[test]
    fn test_q5k_packed_matches_exact_q8_reference() {
        use std::f32::consts::PI;

        let rows = 24;
        let cols = 3;
        let seq_len = 5;

        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let seed = (row * cols + col) as u16;
                let d_val = 0.003 + 0.0005 * seed as f32;
                let dmin_val = 0.001 + 0.0003 * seed as f32;

                let mut scales = [0u8; 12];
                for (i, scale) in scales.iter_mut().enumerate() {
                    *scale = ((seed + i as u16 * 7 + 11) % 256) as u8;
                }

                let mut qh = [0u8; 32];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = ((seed * 3 + i as u16 * 5 + 17) % 256) as u8;
                }

                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
                }

                src.extend_from_slice(&make_q5k_block(d_val, dmin_val, scales, qh, qs));
            }
        }

        let packed = pack_q5k(&src, rows, cols);

        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 29.0).sin() * 0.65 + ((i as f32) * PI / 83.0).cos() * 0.35;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];
        let mut input_bsums_flat = vec![0i16; total_tokens * 8];
        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d, bsums) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
            input_bsums_flat[t * 8..(t + 1) * 8].copy_from_slice(&bsums);
        }

        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q5k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        let output_ref = q5k_q8k_exact_ref(
            &src,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            rows,
            cols,
            seq_len,
        );

        let mut max_abs_err = 0.0f32;
        for s in 0..seq_len {
            for row in 0..rows {
                let got = output_gemm[s * rows + row];
                let exp = output_ref[s * rows + row];
                let abs_err = (got - exp).abs();
                max_abs_err = max_abs_err.max(abs_err);
                assert!(
                    abs_err < 1e-4,
                    "packed-vs-exact mismatch: s={s} row={row} got={got:.8} ref={exp:.8} abs_err={abs_err:.8}"
                );
            }
        }

        eprintln!("test_q5k_packed_matches_exact_q8_reference: max_abs_err = {max_abs_err:.8}");
    }

    // ─── 테스트: seq_len=3 (2-token pair + 1-token fallback) ────

    #[test]
    fn test_q5k_packed_seq3_odd_fallback() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 3;

        // 각 row/col마다 다른 weight 블록 생성
        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let seed = (row * cols + col) as u8;
                let d_val = 0.005 + 0.002 * seed as f32;
                let dmin_val = 0.001 + 0.001 * seed as f32;

                let mut scales = [0u8; 12];
                for i in 0..12 {
                    scales[i] = ((seed as u16 * 7 + i as u16 * 3) % 63) as u8;
                }

                let mut qh = [0u8; 32];
                for i in 0..32 {
                    qh[i] = ((seed as u16 * 11 + i as u16 * 13) % 256) as u8;
                }

                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }

                src.extend_from_slice(&make_q5k_block(d_val, dmin_val, scales, qh, qs));
            }
        }

        let packed = pack_q5k(&src, rows, cols);

        // 3개 토큰에 대해 각각 다른 입력 생성
        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 41.0).sin() * 0.7 + ((i as f32) * PI / 89.0).cos() * 0.4;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];
        let mut input_bsums_flat = vec![0i16; total_tokens * 8];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d, bsums) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
            input_bsums_flat[t * 8..(t + 1) * 8].copy_from_slice(&bsums);
        }

        // seq_len=3 GEMM 실행
        let mut output_seq3 = vec![0.0f32; seq_len * rows];
        gemm_q5k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_seq3,
            rows,
            cols,
            seq_len,
        );

        // 레퍼런스: 각 토큰을 seq_len=1로 개별 실행
        let mut output_ref = vec![0.0f32; seq_len * rows];
        for s in 0..seq_len {
            let tok_qs = &input_qs_flat[s * cols * 256..(s + 1) * cols * 256];
            let tok_d = &input_d_flat[s * cols..(s + 1) * cols];
            let tok_bsums = &input_bsums_flat[s * cols * 8..(s + 1) * cols * 8];
            let mut tok_out = vec![0.0f32; rows];

            gemm_q5k_packed(
                &packed,
                tok_qs,
                tok_d,
                tok_bsums,
                &mut tok_out,
                rows,
                cols,
                1,
            );

            output_ref[s * rows..(s + 1) * rows].copy_from_slice(&tok_out);
        }

        // 비교: seq_len=3 결과 vs 3× seq_len=1 결과
        let mut max_rel_err = 0.0f32;
        for s in 0..seq_len {
            for row in 0..rows {
                let got = output_seq3[s * rows + row];
                let exp = output_ref[s * rows + row];
                let abs_err = (got - exp).abs();
                let rel_err = if exp.abs() > 1e-6 {
                    abs_err / exp.abs()
                } else {
                    abs_err
                };
                if rel_err > max_rel_err {
                    max_rel_err = rel_err;
                }
                assert!(
                    rel_err < 1e-5 || abs_err < 1e-5,
                    "seq3 vs 3×seq1 MISMATCH: s={s} row={row}: \
                     got={got:.8}, ref={exp:.8}, abs_err={abs_err:.8}, rel_err={rel_err:.8}"
                );
            }
        }

        eprintln!("test_q5k_packed_seq3_odd_fallback: max_rel_err = {max_rel_err:.8}");
    }
}
