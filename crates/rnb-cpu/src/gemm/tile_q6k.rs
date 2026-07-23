//! Q6_K packed GEMM 구현 (row-pair interleaved layout)
//!
//! `gemm_q6k_packed`: packed weight × Q8K input → f32 output
//!
//! 새 레이아웃: signed i8 (-32..31) + raw i8 scales + integer-domain accumulation
//!
//! GEMM 공식:
//!   sumi = Σ(sb=0..15) sc_raw[sb] * dot_i32(w_signed[sb*16..(sb+1)*16], x_qs[sb*16..(sb+1)*16])
//!   output += x_d * d * sumi_f32
//!
//! Q6_K는 dmin 없음, bias correction 불필요. input_bsums 인자 없음.
//!
//! 디스패치 순서:
//!   aarch64 + i8mm    → gemm_q6k_packed_i8mm  (smmla, row-pair interleaved weight 직접 사용)
//!   aarch64 + dotprod → gemm_q6k_packed_neon   (vdotq_s32, deinterleave to stack)
//!   fallback          → gemm_q6k_packed_scalar

use crate::gemm::pack_q6k::{Q6K_D_OFF, Q6K_PACKED_BLOCK_BYTES, Q6K_QS_OFF, Q6K_SC_RAW_OFF};

/// Q6_K packed GEMM: output = input × weight^T
///
/// # 인자
/// - `packed`: `pack_q6k()`로 생성된 packed weight bytes
/// - `input_qs`: Q8K quantized input, flat `[seq_len * cols * 256]` i8
/// - `input_d`: Q8K scale per block, `[seq_len * cols]` f32
/// - `output`: `[seq_len * rows]` f32, column-major: `output[s * rows + row]`
/// - `rows`: output dimension (weight rows)
/// - `cols`: number of super-blocks (= input_dim / 256)
/// - `seq_len`: number of input tokens
///
/// Q6_K는 bias correction이 없으므로 input_bsums 인자 없음.
pub fn gemm_q6k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    if let Ok(mode) = std::env::var("RNB_Q6K_PACKED_BACKEND") {
        match mode.trim().to_ascii_lowercase().as_str() {
            "scalar" => {
                gemm_q6k_packed_scalar(packed, input_qs, input_d, output, rows, cols, seq_len);
                return;
            }
            #[cfg(target_arch = "aarch64")]
            "neon" if std::arch::is_aarch64_feature_detected!("dotprod") => unsafe {
                gemm_q6k_packed_neon(packed, input_qs, input_d, output, rows, cols, seq_len);
                return;
            },
            #[cfg(target_arch = "aarch64")]
            "i8mm" if std::arch::is_aarch64_feature_detected!("i8mm") => unsafe {
                gemm_q6k_packed_i8mm(packed, input_qs, input_d, output, rows, cols, seq_len);
                return;
            },
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            unsafe {
                return gemm_q6k_packed_i8mm(
                    packed, input_qs, input_d, output, rows, cols, seq_len,
                );
            }
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                return gemm_q6k_packed_neon(
                    packed, input_qs, input_d, output, rows, cols, seq_len,
                );
            }
        }
    }

    gemm_q6k_packed_scalar(packed, input_qs, input_d, output, rows, cols, seq_len);
}

/// Q6_K packed GEMV for decode (seq_len=1).
pub fn gemv_q6k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon.
            unsafe {
                return gemv_q6k_packed_i8mm(packed, input_qs, input_d, output, rows, cols);
            }
        }
    }

    gemm_q6k_packed(packed, input_qs, input_d, output, rows, cols, 1);
}

// ─── 스칼라 레퍼런스 ──────────────────────────────────────────────────────────

fn gemm_q6k_packed_scalar(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    let row_groups = rows.div_ceil(8);

    for s in 0..seq_len {
        for rg in 0..row_groups {
            for bi in 0..cols {
                let blk_off = (rg * cols + bi) * Q6K_PACKED_BLOCK_BYTES;
                let blk = &packed[blk_off..blk_off + Q6K_PACKED_BLOCK_BYTES];

                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d = input_d[s * cols + bi];

                for nr in 0..8 {
                    let row = rg * 8 + nr;
                    if row >= rows {
                        break;
                    }

                    // Read row's weight data from interleaved layout
                    let pair = nr / 2;
                    let is_odd = nr % 2;

                    // Read d, sc_raw
                    let d = f32::from_le_bytes(
                        blk[Q6K_D_OFF + nr * 4..Q6K_D_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );

                    let sc_base = Q6K_SC_RAW_OFF + nr * 16;

                    let mut sumi = 0i32;

                    for sb in 0..16usize {
                        let sc_raw = blk[sc_base + sb] as i8 as i32;

                        // Dot product from interleaved qs
                        // 16 elements for this sub-block
                        let mut dot = 0i32;
                        for k in 0..16usize {
                            let elem_idx = sb * 16 + k;
                            let chunk = elem_idx / 8;
                            let within = elem_idx % 8;
                            let qs_off = Q6K_QS_OFF + pair * 512 + chunk * 16 + is_odd * 8 + within;
                            let w = blk[qs_off] as i8 as i32; // signed i8
                            dot += w * (x_qs[elem_idx] as i32);
                        }

                        sumi += dot * sc_raw;
                    }

                    output[s * rows + row] += x_d * d * sumi as f32;
                }
            }
        }
    }
}

// ─── NEON vdotq_s32 커널 ─────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q6k_packed_neon(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
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
            let blk_off = (rg * cols + bi) * Q6K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q6K_PACKED_BLOCK_BYTES];

            // --- Phase 1: Deinterleave weight to stack [8][256] + load raw scales ---
            let mut w_qs = [[0u8; 256]; 8];
            let mut sc = [[0i8; 16]; 8];
            let mut d = [0.0f32; 8];

            for nr in 0..8 {
                // Deinterleave qs
                let pair = nr / 2;
                let is_odd = nr % 2;
                let pair_base = Q6K_QS_OFF + pair * 512;
                for k in 0..32usize {
                    let src_off = pair_base + k * 16 + is_odd * 8;
                    w_qs[nr][k * 8..k * 8 + 8].copy_from_slice(&blk[src_off..src_off + 8]);
                }

                // Copy sc_raw
                let sc_off = Q6K_SC_RAW_OFF + nr * 16;
                for i in 0..16 {
                    sc[nr][i] = blk[sc_off + i] as i8;
                }

                d[nr] = f32::from_le_bytes(
                    blk[Q6K_D_OFF + nr * 4..Q6K_D_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
            }

            // --- Phase 2: For each token, compute 8 rows ---
            for s in 0..seq_len {
                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d = input_d[s * cols + bi];

                for r in 0..rows_in_group {
                    let w = w_qs[r].as_ptr() as *const i8;
                    let vzero = vdupq_n_s32(0);
                    let mut acc_v = vzero;

                    // 16 sub-blocks × 16 elements, integer-domain scale
                    for sb in 0..16usize {
                        let off = sb * 16;
                        // 16 elements: 1×16B vdotq
                        let w0 = vld1q_s8(w.add(off));
                        let x0 = vld1q_s8(x_qs[off..].as_ptr());

                        let dot4 = vdotq_s32(vzero, w0, x0);
                        let dot = vaddvq_s32(dot4);

                        // multiply by raw scale in integer domain
                        // sc is i8, can be negative
                        acc_v = vsetq_lane_s32::<0>(
                            vgetq_lane_s32::<0>(acc_v) + dot * sc[r][sb] as i32,
                            acc_v,
                        );
                    }

                    let sumi = vgetq_lane_s32::<0>(acc_v);
                    acc[s * 8 + r] += x_d * d[r] * sumi as f32;
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
//
// Q6_K는 16 sub-blocks × 16 elements, bias/mins 없음.
// raw i8 scales + integer-domain accumulation.
// 16 elements / 8 per smmla = 2 iterations per sub-block.
// Weight is row-pair interleaved → vld1q_s8 loads [r_even:8|r_odd:8].

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_q6k_packed_i8mm(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
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
            let blk_off = (rg * cols + bi) * Q6K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q6K_PACKED_BLOCK_BYTES];
            let blk_ptr = blk.as_ptr() as *const i8;

            // --- Phase 1: Load raw scales/d to stack ---
            let mut sc = [[0i8; 16]; 8];
            let mut d = [0.0f32; 8];

            for nr in 0..8 {
                let sc_off = Q6K_SC_RAW_OFF + nr * 16;
                for i in 0..16 {
                    sc[nr][i] = blk[sc_off + i] as i8;
                }

                d[nr] = f32::from_le_bytes(
                    blk[Q6K_D_OFF + nr * 4..Q6K_D_OFF + nr * 4 + 4]
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
                for t in 0..8 {
                    let st = s + t;
                    let inp_base = (st * cols + bi) * 256;
                    t_qs[t] = &input_qs[inp_base..inp_base + 256];
                    t_d[t] = input_d[st * cols + bi];
                }

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    // Collect per-sub-block dots for vectorized scale multiply
                    let mut dots = [[[0i32; 16]; 2]; 8]; // [token][row_in_pair][sub_block]

                    let w_pair_base = Q6K_QS_OFF + rp * 512;

                    for sb in 0..16usize {
                        let k_off = sb * 16;

                        let mut acc0 = vdupq_n_s32(0);
                        let mut acc1 = vdupq_n_s32(0);
                        let mut acc2 = vdupq_n_s32(0);
                        let mut acc3 = vdupq_n_s32(0);

                        for ki in 0..2usize {
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

                    // Vectorized sumi: dot_product(dots[0..16], sc[0..16])
                    // Q6_K sc is i8[16] → widen to i32, multiply in 4 chunks of 4
                    let sc_r0_i8 = vld1q_s8(sc[r0].as_ptr());
                    let sc_r0_i16_lo = vmovl_s8(vget_low_s8(sc_r0_i8));
                    let sc_r0_i16_hi = vmovl_s8(vget_high_s8(sc_r0_i8));
                    let sc_r0_0 = vmovl_s16(vget_low_s16(sc_r0_i16_lo));
                    let sc_r0_1 = vmovl_s16(vget_high_s16(sc_r0_i16_lo));
                    let sc_r0_2 = vmovl_s16(vget_low_s16(sc_r0_i16_hi));
                    let sc_r0_3 = vmovl_s16(vget_high_s16(sc_r0_i16_hi));

                    for t in 0..8 {
                        let d0 = vld1q_s32(dots[t][0][0..4].as_ptr());
                        let d1 = vld1q_s32(dots[t][0][4..8].as_ptr());
                        let d2 = vld1q_s32(dots[t][0][8..12].as_ptr());
                        let d3 = vld1q_s32(dots[t][0][12..16].as_ptr());
                        let sum01 = vaddq_s32(vmulq_s32(d0, sc_r0_0), vmulq_s32(d1, sc_r0_1));
                        let sum23 = vaddq_s32(vmulq_s32(d2, sc_r0_2), vmulq_s32(d3, sc_r0_3));
                        let sumi_val = vaddvq_s32(vaddq_s32(sum01, sum23));

                        acc[(s + t) * 8 + r0] += t_d[t] * d[r0] * sumi_val as f32;
                    }

                    if has_r1 {
                        let sc_r1_i8 = vld1q_s8(sc[r1].as_ptr());
                        let sc_r1_i16_lo = vmovl_s8(vget_low_s8(sc_r1_i8));
                        let sc_r1_i16_hi = vmovl_s8(vget_high_s8(sc_r1_i8));
                        let sc_r1_0 = vmovl_s16(vget_low_s16(sc_r1_i16_lo));
                        let sc_r1_1 = vmovl_s16(vget_high_s16(sc_r1_i16_lo));
                        let sc_r1_2 = vmovl_s16(vget_low_s16(sc_r1_i16_hi));
                        let sc_r1_3 = vmovl_s16(vget_high_s16(sc_r1_i16_hi));

                        for t in 0..8 {
                            let d0 = vld1q_s32(dots[t][1][0..4].as_ptr());
                            let d1 = vld1q_s32(dots[t][1][4..8].as_ptr());
                            let d2 = vld1q_s32(dots[t][1][8..12].as_ptr());
                            let d3 = vld1q_s32(dots[t][1][12..16].as_ptr());
                            let sum01 = vaddq_s32(vmulq_s32(d0, sc_r1_0), vmulq_s32(d1, sc_r1_1));
                            let sum23 = vaddq_s32(vmulq_s32(d2, sc_r1_2), vmulq_s32(d3, sc_r1_3));
                            let sumi_val = vaddvq_s32(vaddq_s32(sum01, sum23));

                            acc[(s + t) * 8 + r1] += t_d[t] * d[r1] * sumi_val as f32;
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

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    let mut sumi_r0_s0 = 0i32;
                    let mut sumi_r0_s1 = 0i32;
                    let mut sumi_r1_s0 = 0i32;
                    let mut sumi_r1_s1 = 0i32;

                    let w_pair_base = Q6K_QS_OFF + rp * 512;

                    for sb in 0..16usize {
                        let k_off = sb * 16;

                        let mut smmla_acc = vdupq_n_s32(0);

                        for ki in 0..2usize {
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

                        if has_r1 {
                            let sc_r1 = sc[r1][sb] as i32;
                            sumi_r1_s0 += dot_r1_s0 * sc_r1;
                            sumi_r1_s1 += dot_r1_s1 * sc_r1;
                        }
                    }

                    acc[s * 8 + r0] += x_d0 * d[r0] * sumi_r0_s0 as f32;
                    acc[(s + 1) * 8 + r0] += x_d1 * d[r0] * sumi_r0_s1 as f32;

                    if has_r1 {
                        acc[s * 8 + r1] += x_d0 * d[r1] * sumi_r1_s0 as f32;
                        acc[(s + 1) * 8 + r1] += x_d1 * d[r1] * sumi_r1_s1 as f32;
                    }
                }

                s += 2;
            }

            // ── 1-token remainder (odd seq_len) ──
            if s < seq_len {
                let inp_base = (s * cols + bi) * 256;
                let x_qs = &input_qs[inp_base..inp_base + 256];
                let x_d_val = input_d[s * cols + bi];

                for rp in 0..row_pairs {
                    let r0 = rp * 2;
                    let r1 = r0 + 1;
                    let has_r1 = r1 < rows_in_group;

                    let mut sumi_r0 = 0i32;
                    let mut sumi_r1 = 0i32;

                    let w_pair_base = Q6K_QS_OFF + rp * 512;

                    for sb in 0..16usize {
                        let k_off = sb * 16;
                        let mut smmla_acc = vdupq_n_s32(0);

                        for ki in 0..2usize {
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

                        if has_r1 {
                            sumi_r1 += dot_r1 * sc[r1][sb] as i32;
                        }
                    }

                    acc[s * 8 + r0] += x_d_val * d[r0] * sumi_r0 as f32;

                    if has_r1 {
                        acc[s * 8 + r1] += x_d_val * d[r1] * sumi_r1 as f32;
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
unsafe fn gemv_q6k_packed_i8mm(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
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
                    let blk_off = (rg * cols + bi) * Q6K_PACKED_BLOCK_BYTES;
                    let blk = &packed[blk_off..blk_off + Q6K_PACKED_BLOCK_BYTES];
                    let blk_ptr = blk.as_ptr() as *const i8;

                    let mut sc = [[0i8; 16]; 8];
                    let mut d = [0.0f32; 8];

                    for nr in 0..rows_in_group {
                        let sc_off = Q6K_SC_RAW_OFF + nr * 16;
                        for i in 0..16 {
                            sc[nr][i] = blk[sc_off + i] as i8;
                        }

                        d[nr] = f32::from_le_bytes(
                            blk[Q6K_D_OFF + nr * 4..Q6K_D_OFF + nr * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                    }

                    let x_qs = &input_qs[bi * 256..bi * 256 + 256];
                    let x_d_val = input_d[bi];

                    for rp in 0..row_pairs {
                        let r0 = rp * 2;
                        let r1 = r0 + 1;
                        let has_r1 = r1 < rows_in_group;

                        let mut sumi_r0 = 0i32;
                        let mut sumi_r1 = 0i32;

                        let w_pair_base = Q6K_QS_OFF + rp * 512;

                        for sb in 0..16usize {
                            let k_off = sb * 16;
                            let mut smmla_acc = vdupq_n_s32(0);

                            for ki in 0..2usize {
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

                            if has_r1 {
                                sumi_r1 += dot_r1 * sc[r1][sb] as i32;
                            }
                        }

                        acc[r0] += x_d_val * d[r0] * sumi_r0 as f32;

                        if has_r1 {
                            acc[r1] += x_d_val * d[r1] * sumi_r1 as f32;
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
    use crate::gemm::pack_q6k::{pack_q6k, unpack_q6k};
    use half::f16;

    /// Q6_K 더미 블록 생성 (210 bytes)
    fn make_q6k_block(d_val: f32, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        block[0..128].copy_from_slice(&ql);
        block[128..192].copy_from_slice(&qh);
        for (i, &s) in scales.iter().enumerate() {
            block[192 + i] = s as u8;
        }
        block[208..210].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block
    }

    /// Q8K 수동 양자화: f32[256] → (qs: i8[256], d: f32)
    fn quantize_q8k(x: &[f32; 256]) -> ([i8; 256], f32) {
        let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv_d = 1.0 / d;

        let mut qs = [0i8; 256];
        for (i, &v) in x.iter().enumerate() {
            qs[i] = (v * inv_d).round().clamp(-128.0, 127.0) as i8;
        }

        (qs, d)
    }

    /// Q6_K 블록을 f32[256]으로 dequant
    fn dequant_q6k_block(block: &[u8]) -> [f32; 256] {
        let ql: &[u8; 128] = block[0..128].try_into().unwrap();
        let qh: &[u8; 64] = block[128..192].try_into().unwrap();
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut unpacked = [0i8; 256];
        unpack_q6k(ql, qh, &mut unpacked);

        let mut out = [0.0f32; 256];
        for i in 0..256 {
            let scale_raw = block[192 + i / 16] as i8;
            out[i] = d * scale_raw as f32 * unpacked[i] as f32;
        }
        out
    }

    fn q6k_q8k_exact_ref(
        src: &[u8],
        input_qs: &[i8],
        input_d: &[f32],
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * rows];

        for s in 0..seq_len {
            for row in 0..rows {
                let mut acc = 0.0f32;
                for bi in 0..cols {
                    let src_off = (row * cols + bi) * 210;
                    let block = &src[src_off..src_off + 210];
                    let ql: &[u8; 128] = block[0..128].try_into().unwrap();
                    let qh: &[u8; 64] = block[128..192].try_into().unwrap();
                    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

                    let mut w_signed = [0i8; 256];
                    unpack_q6k(ql, qh, &mut w_signed);

                    let x_off = (s * cols + bi) * 256;
                    let x_qs = &input_qs[x_off..x_off + 256];
                    let x_d = input_d[s * cols + bi];

                    let mut sumi = 0i32;
                    for sb in 0..16usize {
                        let sc_raw = block[192 + sb] as i8 as i32;
                        let mut dot = 0i32;
                        for k in 0..16usize {
                            let idx = sb * 16 + k;
                            dot += w_signed[idx] as i32 * x_qs[idx] as i32;
                        }
                        sumi += sc_raw * dot;
                    }

                    acc += x_d * d * sumi as f32;
                }
                out[s * rows + row] = acc;
            }
        }

        out
    }

    // ─── 테스트 1: 제로 입력 ─────────────────────────────────────

    #[test]
    fn test_tile_q6k_gemm_zero_input() {
        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let block = make_q6k_block(1.0, [5i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q6k(&src, rows, cols);

        let input_qs = vec![0i8; seq_len * cols * 256];
        let input_d = vec![1.0f32; seq_len * cols];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q6k_packed(
            &packed,
            &input_qs,
            &input_d,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0.0, "output[{i}] = {v}, expected 0");
        }
    }

    // ─── 테스트 2: 단일 비제로 수동 계산 크로스체크 ──────────────

    #[test]
    fn test_tile_q6k_gemm_single_nonzero() {
        // ql=0, qh=0 → unsigned 6-bit = 0
        // w_unsigned[0..16] = 0
        // input_qs[0..16] = 1
        // d = 1.0, sc_raw[0] = 5
        //
        // sumi = 5 * dot(w=0, x=1, 16개) = 5 * 0 = 0
        // output = 1.0 * 1.0 * 0 = 0
        //
        // ql=0xFF, qh=0xFF → unsigned = 63
        // dot = 63 * 1 * 16 = 1008
        // sumi = 5 * 1008 = 5040
        // output = 1.0 * 1.0 * 5040 = 5040

        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let block = make_q6k_block(1.0, [5i8; 16], [0xFFu8; 128], [0xFFu8; 64]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q6k(&src, rows, cols);

        // sub-block 0 (16원소) = 1
        let mut input_qs = vec![0i8; seq_len * cols * 256];
        for k in 0..16 {
            input_qs[k] = 1;
        }
        let input_d = vec![1.0f32; seq_len * cols];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q6k_packed(
            &packed,
            &input_qs,
            &input_d,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        // w_signed = 31, dot = 31 * 16 = 496, sumi = 5 * 496 = 2480
        let expected = 2480.0f32;
        for row in 0..rows {
            let v = output[row];
            assert!(
                (v - expected).abs() < 0.5,
                "output[{row}] = {v}, expected {expected}"
            );
        }
    }

    // ─── 테스트 3: f32 레퍼런스 크로스체크 ───────────────────────

    #[test]
    fn test_tile_q6k_gemm_crosscheck_f32_ref() {
        use std::f32::consts::PI;

        let rows = 16; // 2 row groups
        let cols = 2; // 2 super-blocks
        let seq_len = 2; // 2 tokens

        // 다양한 ql 패턴
        let mut ql = [0u8; 128];
        for i in 0..128usize {
            ql[i] = ((i * 3 + 7) % 256) as u8;
        }
        let mut qh = [0u8; 64];
        for i in 0..64usize {
            qh[i] = ((i * 5 + 11) % 256) as u8;
        }

        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i as i8) - 8).clamp(-8, 7);
        }

        let block = make_q6k_block(0.01, scales, ql, qh);
        let src: Vec<u8> = block.repeat(rows * cols);
        let packed = pack_q6k(&src, rows, cols);

        // f32 입력 생성
        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 64.0).sin() * 0.5;
        }

        // Q8K 양자화
        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
        }

        // packed GEMM
        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q6k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        // f32 레퍼런스: dequant weight × f32 input
        let mut weight_f32 = vec![0.0f32; rows * cols * 256];
        for row in 0..rows {
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 210;
                let w_blk = dequant_q6k_block(&src[src_off..src_off + 210]);
                let w_off = (row * cols + bi) * 256;
                weight_f32[w_off..w_off + 256].copy_from_slice(&w_blk);
            }
        }

        let mut output_ref = vec![0.0f32; seq_len * rows];
        for s in 0..seq_len {
            for row in 0..rows {
                let mut dot = 0.0f32;
                for bi in 0..cols {
                    let x_off = (s * cols + bi) * 256;
                    let w_off = (row * cols + bi) * 256;
                    for k in 0..256usize {
                        dot += weight_f32[w_off + k] * x_f32_all[x_off + k];
                    }
                }
                output_ref[s * rows + row] = dot;
            }
        }

        // 비교 (relative < 0.15 or absolute < 0.1)
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

    // ─── 테스트 4: seq_len > 1 독립성 ───────────────────────────

    #[test]
    fn test_tile_q6k_gemm_seq_len_independence() {
        // ql=0xFF, qh=0xFF → signed=31
        // token 0: x_qs all +1, token 1: x_qs all -1
        // d=1.0, sc_raw=1
        //
        // dot per sb = 31 * (±1) * 16 = ±496
        // sumi per sb = 1 * ±496 = ±496
        // total sumi = 16 * ±496 = ±7936
        // output = 1.0 * 1.0 * ±7936 = ±7936
        let rows = 8;
        let cols = 1;
        let seq_len = 2;

        let block = make_q6k_block(1.0, [1i8; 16], [0xFFu8; 128], [0xFFu8; 64]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q6k(&src, rows, cols);

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        // token 0: all +1
        for k in 0..256 {
            input_qs[k] = 1;
        }
        // token 1: all -1
        for k in 0..256 {
            input_qs[256 + k] = -1;
        }
        let input_d = vec![1.0f32; seq_len * cols];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q6k_packed(
            &packed,
            &input_qs,
            &input_d,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        let expected = 7936.0f32;
        for row in 0..rows {
            let v0 = output[row];
            let v1 = output[rows + row];
            assert!(
                (v0 + v1).abs() < 1e-3,
                "row={row}: v0={v0}, v1={v1} should be negatives"
            );
            assert!(
                (v0 - expected).abs() < 1.0,
                "row={row}: v0={v0}, expected {expected}"
            );
        }
    }

    // ─── 테스트 5: unique rows (행마다 다른 weight) ──────────────

    #[test]
    fn test_tile_q6k_gemm_unique_rows() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

        // 각 row, 각 col마다 DIFFERENT Q6_K 블록 생성
        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let seed = (row * cols + col) as u8;
                let d_val = 0.005 + 0.002 * seed as f32;

                // 각 row마다 다른 scales (signed i8, -32..31)
                let mut scales = [0i8; 16];
                for i in 0..16 {
                    scales[i] = (((seed as i16 * 7 + i as i16 * 3) % 63) - 31) as i8;
                }

                // 각 row마다 다른 ql (lower 4+2 bits packed)
                let mut ql = [0u8; 128];
                for i in 0..128 {
                    ql[i] = ((seed as u16 + i as u16 * 5 + 3) % 256) as u8;
                }

                // 각 row마다 다른 qh (upper 2 bits packed)
                let mut qh = [0u8; 64];
                for i in 0..64 {
                    qh[i] = ((seed as u16 * 11 + i as u16 * 13 + 7) % 256) as u8;
                }

                let block = make_q6k_block(d_val, scales, ql, qh);
                src.extend_from_slice(&block);
            }
        }

        let packed = pack_q6k(&src, rows, cols);

        // 다양한 입력 생성
        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 37.0).sin() * 0.8 + ((i as f32) * PI / 97.0).cos() * 0.3;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
        }

        // GEMM 실행
        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q6k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        let output_ref =
            q6k_q8k_exact_ref(&src, &input_qs_flat, &input_d_flat, rows, cols, seq_len);

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
                if rel_err >= 1e-5 && abs_err >= 1e-5 {
                    eprintln!(
                        "  HIGH ERR s={s} row={row}: got={got:.6}, ref={exp:.6}, \
                         abs_err={abs_err:.6}, rel_err={rel_err:.4}"
                    );
                    fail_count += 1;
                }
                assert!(
                    rel_err < 1e-5 || abs_err < 1e-5,
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

        eprintln!("test_tile_q6k_gemm_unique_rows: max_rel_err = {max_rel_err:.6}, high_err_count = {fail_count}");
    }

    #[test]
    fn test_tile_q6k_packed_matches_exact_q8_reference() {
        use std::f32::consts::PI;

        let rows = 25;
        let cols = 3;
        let seq_len = 15;

        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let seed = (row * cols + col) as u16;
                let d_val = 0.0025 + 0.0007 * seed as f32;

                let mut scales = [0i8; 16];
                for (i, scale) in scales.iter_mut().enumerate() {
                    *scale = ((((seed as i32) * 11 + i as i32 * 7) % 63) - 31) as i8;
                }

                let mut ql = [0u8; 128];
                for (i, q) in ql.iter_mut().enumerate() {
                    *q = ((seed + i as u16 * 9 + 5) % 256) as u8;
                }

                let mut qh = [0u8; 64];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = ((seed * 13 + i as u16 * 3 + 17) % 256) as u8;
                }

                src.extend_from_slice(&make_q6k_block(d_val, scales, ql, qh));
            }
        }

        let packed = pack_q6k(&src, rows, cols);

        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 29.0).sin() * 0.65 + ((i as f32) * PI / 83.0).cos() * 0.35;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];
        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
        }

        let mut output_gemm = vec![0.0f32; seq_len * rows];
        gemm_q6k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        let output_ref =
            q6k_q8k_exact_ref(&src, &input_qs_flat, &input_d_flat, rows, cols, seq_len);

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

        eprintln!(
            "test_tile_q6k_packed_matches_exact_q8_reference: max_abs_err = {max_abs_err:.8}"
        );
    }

    // ─── 테스트 6: seq_len=3 (2-token pair + 1-token fallback) ──

    #[test]
    fn test_q6k_packed_seq3_odd_fallback() {
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

                let mut scales = [0i8; 16];
                for i in 0..16 {
                    scales[i] = (((seed as i16 * 7 + i as i16 * 3) % 63) - 31) as i8;
                }

                let mut ql = [0u8; 128];
                for i in 0..128 {
                    ql[i] = ((seed as u16 + i as u16 * 5 + 3) % 256) as u8;
                }

                let mut qh = [0u8; 64];
                for i in 0..64 {
                    qh[i] = ((seed as u16 * 11 + i as u16 * 13 + 7) % 256) as u8;
                }

                let block = make_q6k_block(d_val, scales, ql, qh);
                src.extend_from_slice(&block);
            }
        }

        let packed = pack_q6k(&src, rows, cols);

        // 3개 토큰에 대해 각각 다른 입력 생성
        let total_tokens = seq_len * cols;
        let mut x_f32_all = vec![0.0f32; total_tokens * 256];
        for (i, v) in x_f32_all.iter_mut().enumerate() {
            *v = ((i as f32) * PI / 41.0).sin() * 0.7 + ((i as f32) * PI / 89.0).cos() * 0.4;
        }

        let mut input_qs_flat = vec![0i8; total_tokens * 256];
        let mut input_d_flat = vec![0.0f32; total_tokens];

        for t in 0..total_tokens {
            let x_slice: &[f32; 256] = x_f32_all[t * 256..(t + 1) * 256].try_into().unwrap();
            let (qs, d) = quantize_q8k(x_slice);
            input_qs_flat[t * 256..(t + 1) * 256].copy_from_slice(&qs);
            input_d_flat[t] = d;
        }

        // seq_len=3 GEMM 실행
        let mut output_seq3 = vec![0.0f32; seq_len * rows];
        gemm_q6k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
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
            let mut tok_out = vec![0.0f32; rows];

            gemm_q6k_packed(&packed, tok_qs, tok_d, &mut tok_out, rows, cols, 1);

            output_ref[s * rows..(s + 1) * rows].copy_from_slice(&tok_out);
        }

        // 비교: seq_len=3 결과 vs 3x seq_len=1 결과
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
                    "seq3 vs 3x seq1 MISMATCH: s={s} row={row}: \
                     got={got:.8}, ref={exp:.8}, abs_err={abs_err:.8}, rel_err={rel_err:.8}"
                );
            }
        }

        eprintln!("test_q6k_packed_seq3_odd_fallback: max_rel_err = {max_rel_err:.8}");
    }
}
