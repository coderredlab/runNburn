//! Q4_K packed GEMM 구현 (row-pair interleaved layout)
//!
//! `gemm_q4k_packed`: packed weight × Q8K input → f32 output
//!
//! 새 레이아웃: unsigned nibble (0..15) + raw u8 scales + integer-domain accumulation
//!
//! GEMM 공식:
//!   sumi = Σ(sb=0..7) sc_raw[sb] * dot_i32(w_unsigned[sb*32..+32], x_qs[sb*32..+32])
//!   summ = Σ(sb=0..7) mn_raw[sb] * bsums[sb]
//!   output += x_d * (d * sumi_f32 - dmin * summ_f32)
//!
//! 디스패치 순서:
//!   aarch64 + i8mm    → gemm_q4k_packed_i8mm  (smmla, row-pair interleaved weight 직접 사용)
//!   aarch64 + dotprod → gemm_q4k_packed_neon   (vdotq_s32, deinterleave to stack)
//!   fallback          → gemm_q4k_packed_scalar

use crate::gemm::pack_q4k::{
    Q4K_COMPACT_BLOCK_BYTES, Q4K_COMPACT_DMIN_OFF, Q4K_COMPACT_D_OFF, Q4K_COMPACT_MN_RAW_OFF,
    Q4K_COMPACT_QS_OFF, Q4K_COMPACT_SC_RAW_OFF, Q4K_DMIN_OFF, Q4K_D_OFF, Q4K_MN_RAW_OFF,
    Q4K_PACKED_BLOCK_BYTES, Q4K_QS_OFF, Q4K_RAW_META_BLOCK_BYTES, Q4K_RAW_META_QS_BYTES,
    Q4K_SC_RAW_OFF,
};

/// Q4_K packed GEMM: output = input × weight^T
///
/// # 인자
/// - `packed`: `pack_q4k()`로 생성된 packed weight bytes
/// - `input_qs`: Q8K quantized input, flat `[seq_len * cols * 256]` i8
/// - `input_d`: Q8K scale per block, `[seq_len * cols]` f32
/// - `input_bsums`: Q8K block sums, `[seq_len * cols * 8]` i16
/// - `output`: `[seq_len * rows]` f32, column-major: `output[s * rows + row]`
/// - `rows`: output dimension (weight rows)
/// - `cols`: number of super-blocks (= input_dim / 256)
/// - `seq_len`: number of input tokens
pub fn gemm_q4k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    if is_q4k_compact_len(packed.len(), rows, cols) {
        return gemm_q4k_compact(
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

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon+dotprod
            unsafe {
                return gemm_q4k_packed_i8mm(
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
                return gemm_q4k_packed_neon(
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

    gemm_q4k_packed_scalar(
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

/// Q4_K packed GEMV for decode (seq_len=1).
/// Same packed weight layout, delegates to gemm with seq_len=1.
pub fn gemv_q4k_packed(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    if is_q4k_compact_len(packed.len(), rows, cols) {
        return gemv_q4k_compact(packed, input_qs, input_d, input_bsums, output, rows, cols);
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon.
            unsafe {
                return gemv_q4k_packed_i8mm(
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

    gemm_q4k_packed(
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

/// Q4_K rawmeta GEMM: output = input × weight^T.
///
/// Layout is row-major Q4_K blocks where each block is `qs[128] + sc[8] + mn[8] + d + dmin`.
pub fn gemm_q4k_raw_meta(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    assert_eq!(packed.len(), rows * cols * Q4K_RAW_META_BLOCK_BYTES);
    assert!(input_qs.len() >= seq_len * cols * 256);
    assert!(input_d.len() >= seq_len * cols);
    assert!(input_bsums.len() >= seq_len * cols * 8);
    assert!(output.len() >= seq_len * rows);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            // SAFETY: i8mm implies neon+dotprod.
            unsafe {
                return gemm_q4k_raw_meta_i8mm(
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
            // SAFETY: dotprod implies neon.
            unsafe {
                return gemm_q4k_raw_meta_neon(
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

    gemm_q4k_raw_meta_scalar(
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

fn gemm_q4k_raw_meta_scalar(
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

    output[..seq_len * rows].fill(0.0);

    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();
    (0..rows).into_par_iter().for_each(|row| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };
        for s in 0..seq_len {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let block_off = (row * cols + bi) * Q4K_RAW_META_BLOCK_BYTES;
                let block = &packed[block_off..block_off + Q4K_RAW_META_BLOCK_BYTES];
                let input_off = (s * cols + bi) * 256;
                let bsum_off = (s * cols + bi) * 8;
                acc += q4k_raw_meta_dot(
                    block,
                    &input_qs[input_off..input_off + 256],
                    input_d[s * cols + bi],
                    &input_bsums[bsum_off..bsum_off + 8],
                );
            }
            out_slice[s * rows + row] = acc;
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_q4k_raw_meta_i8mm(
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

    output[..seq_len * rows].fill(0.0);

    let row_pairs = rows.div_ceil(2);
    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();
    (0..row_pairs).into_par_iter().for_each(|rp| {
        let r0 = rp * 2;
        let r1 = r0 + 1;
        let has_r1 = r1 < rows;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };

        let mut s = 0usize;
        while has_r1 && s + 7 < seq_len {
            let mut acc = [[0.0f32; 2]; 8];
            for bi in 0..cols {
                let b0 = &packed[(r0 * cols + bi) * Q4K_RAW_META_BLOCK_BYTES
                    ..(r0 * cols + bi + 1) * Q4K_RAW_META_BLOCK_BYTES];
                let b1 = &packed[(r1 * cols + bi) * Q4K_RAW_META_BLOCK_BYTES
                    ..(r1 * cols + bi + 1) * Q4K_RAW_META_BLOCK_BYTES];
                unsafe {
                    q4k_raw_meta_accum_8_i8mm(
                        b0,
                        b1,
                        input_qs,
                        input_d,
                        input_bsums,
                        &mut acc,
                        cols,
                        s,
                        bi,
                    );
                }
            }
            for t in 0..8usize {
                out_slice[(s + t) * rows + r0] = acc[t][0];
                out_slice[(s + t) * rows + r1] = acc[t][1];
            }
            s += 8;
        }

        while s < seq_len {
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            for bi in 0..cols {
                let b0 = &packed[(r0 * cols + bi) * Q4K_RAW_META_BLOCK_BYTES
                    ..(r0 * cols + bi + 1) * Q4K_RAW_META_BLOCK_BYTES];
                let input_off = (s * cols + bi) * 256;
                let bsum_off = (s * cols + bi) * 8;
                acc0 += q4k_raw_meta_dot(
                    b0,
                    &input_qs[input_off..input_off + 256],
                    input_d[s * cols + bi],
                    &input_bsums[bsum_off..bsum_off + 8],
                );
                if has_r1 {
                    let b1 = &packed[(r1 * cols + bi) * Q4K_RAW_META_BLOCK_BYTES
                        ..(r1 * cols + bi + 1) * Q4K_RAW_META_BLOCK_BYTES];
                    acc1 += q4k_raw_meta_dot(
                        b1,
                        &input_qs[input_off..input_off + 256],
                        input_d[s * cols + bi],
                        &input_bsums[bsum_off..bsum_off + 8],
                    );
                }
            }
            out_slice[s * rows + r0] = acc0;
            if has_r1 {
                out_slice[s * rows + r1] = acc1;
            }
            s += 1;
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn q4k_raw_meta_accum_8_i8mm(
    block0: &[u8],
    block1: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    acc: &mut [[f32; 2]; 8],
    cols: usize,
    seq_start: usize,
    bi: usize,
) {
    use std::arch::aarch64::*;

    let qs0 = block0.as_ptr();
    let qs1 = block1.as_ptr();
    let meta0 = block0.as_ptr().add(Q4K_RAW_META_QS_BYTES);
    let meta1 = block1.as_ptr().add(Q4K_RAW_META_QS_BYTES);
    let sc0 = std::slice::from_raw_parts(meta0, 8);
    let mn0 = std::slice::from_raw_parts(meta0.add(8), 8);
    let d0 = f32::from_bits((meta0.add(16) as *const u32).read_unaligned());
    let dmin0 = f32::from_bits((meta0.add(20) as *const u32).read_unaligned());
    let sc1 = std::slice::from_raw_parts(meta1, 8);
    let mn1 = std::slice::from_raw_parts(meta1.add(8), 8);
    let d1 = f32::from_bits((meta1.add(16) as *const u32).read_unaligned());
    let dmin1 = f32::from_bits((meta1.add(20) as *const u32).read_unaligned());

    let mut acc_v = [vdupq_n_s32(0); 4];
    let mut summ = [[0i32; 2]; 8];
    let mask_low = vdup_n_u8(0x0f);

    for group in 0..4usize {
        let sb0 = group * 2;
        let sb1 = sb0 + 1;
        let sc_vec0 = [
            sc0[sb0] as i32,
            sc0[sb0] as i32,
            sc1[sb0] as i32,
            sc1[sb0] as i32,
        ];
        let sc_vec1 = [
            sc0[sb1] as i32,
            sc0[sb1] as i32,
            sc1[sb1] as i32,
            sc1[sb1] as i32,
        ];
        let scv0 = vld1q_s32(sc_vec0.as_ptr());
        let scv1 = vld1q_s32(sc_vec1.as_ptr());
        let mn0_0 = mn0[sb0] as i32;
        let mn0_1 = mn0[sb1] as i32;
        let mn1_0 = mn1[sb0] as i32;
        let mn1_1 = mn1[sb1] as i32;

        for chunk in 0..4usize {
            let elem0 = sb0 * 32 + chunk * 8;
            let elem1 = sb1 * 32 + chunk * 8;
            let qoff = group * 32 + chunk * 8;

            let r0b = vld1_u8(qs0.add(qoff));
            let r1b = vld1_u8(qs1.add(qoff));
            let w0 =
                vreinterpretq_s8_u8(vcombine_u8(vand_u8(r0b, mask_low), vand_u8(r1b, mask_low)));
            let w1 = vreinterpretq_s8_u8(vcombine_u8(vshr_n_u8::<4>(r0b), vshr_n_u8::<4>(r1b)));

            for tp in 0..4usize {
                let t0 = tp * 2;
                let t1 = t0 + 1;
                let x0 = input_qs.as_ptr().add(((seq_start + t0) * cols + bi) * 256);
                let x1 = input_qs.as_ptr().add(((seq_start + t1) * cols + bi) * 256);
                let xv0 = vcombine_s8(vld1_s8(x0.add(elem0)), vld1_s8(x1.add(elem0)));
                let xv1 = vcombine_s8(vld1_s8(x0.add(elem1)), vld1_s8(x1.add(elem1)));
                let dot0 = vmmlaq_s32(vdupq_n_s32(0), w0, xv0);
                let dot1 = vmmlaq_s32(vdupq_n_s32(0), w1, xv1);
                acc_v[tp] = vaddq_s32(acc_v[tp], vmulq_s32(dot0, scv0));
                acc_v[tp] = vaddq_s32(acc_v[tp], vmulq_s32(dot1, scv1));
            }
        }

        for t in 0..8usize {
            let bs = input_bsums.as_ptr().add(((seq_start + t) * cols + bi) * 8);
            summ[t][0] += mn0_0 * *bs.add(sb0) as i32 + mn0_1 * *bs.add(sb1) as i32;
            summ[t][1] += mn1_0 * *bs.add(sb0) as i32 + mn1_1 * *bs.add(sb1) as i32;
        }
    }

    for tp in 0..4usize {
        let t0 = tp * 2;
        let t1 = t0 + 1;
        let idx0 = (seq_start + t0) * cols + bi;
        let idx1 = (seq_start + t1) * cols + bi;
        let sumi_r0_t0 = vgetq_lane_s32::<0>(acc_v[tp]);
        let sumi_r0_t1 = vgetq_lane_s32::<1>(acc_v[tp]);
        let sumi_r1_t0 = vgetq_lane_s32::<2>(acc_v[tp]);
        let sumi_r1_t1 = vgetq_lane_s32::<3>(acc_v[tp]);
        acc[t0][0] += input_d[idx0] * (d0 * sumi_r0_t0 as f32 - dmin0 * summ[t0][0] as f32);
        acc[t1][0] += input_d[idx1] * (d0 * sumi_r0_t1 as f32 - dmin0 * summ[t1][0] as f32);
        acc[t0][1] += input_d[idx0] * (d1 * sumi_r1_t0 as f32 - dmin1 * summ[t0][1] as f32);
        acc[t1][1] += input_d[idx1] * (d1 * sumi_r1_t1 as f32 - dmin1 * summ[t1][1] as f32);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q4k_raw_meta_neon(
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

    output[..seq_len * rows].fill(0.0);

    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();
    (0..rows).into_par_iter().for_each(|row| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };
        let mut s = 0usize;
        while s + 7 < seq_len {
            let mut acc = [0.0f32; 8];
            for bi in 0..cols {
                let block_off = (row * cols + bi) * Q4K_RAW_META_BLOCK_BYTES;
                let block = &packed[block_off..block_off + Q4K_RAW_META_BLOCK_BYTES];
                unsafe {
                    q4k_raw_meta_accum_8_neon(
                        block,
                        input_qs,
                        input_d,
                        input_bsums,
                        &mut acc,
                        cols,
                        s,
                        bi,
                    );
                }
            }
            for t in 0..8usize {
                out_slice[(s + t) * rows + row] = acc[t];
            }
            s += 8;
        }

        while s < seq_len {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let block_off = (row * cols + bi) * Q4K_RAW_META_BLOCK_BYTES;
                let block = &packed[block_off..block_off + Q4K_RAW_META_BLOCK_BYTES];
                let input_off = (s * cols + bi) * 256;
                let bsum_off = (s * cols + bi) * 8;
                acc += q4k_raw_meta_dot(
                    block,
                    &input_qs[input_off..input_off + 256],
                    input_d[s * cols + bi],
                    &input_bsums[bsum_off..bsum_off + 8],
                );
            }
            out_slice[s * rows + row] = acc;
            s += 1;
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k_raw_meta_accum_8_neon(
    block: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    acc: &mut [f32; 8],
    cols: usize,
    seq_start: usize,
    bi: usize,
) {
    use std::arch::aarch64::*;

    let qs = block.as_ptr();
    let meta = block.as_ptr().add(Q4K_RAW_META_QS_BYTES);
    let sc = std::slice::from_raw_parts(meta, 8);
    let mn = std::slice::from_raw_parts(meta.add(8), 8);
    let d = f32::from_bits((meta.add(16) as *const u32).read_unaligned());
    let dmin = f32::from_bits((meta.add(20) as *const u32).read_unaligned());
    let mask_low = vdupq_n_u8(0x0f);
    let zero = vdupq_n_s32(0);
    let mut acc_v = [zero; 8];
    let mut summ = [0i32; 8];

    for group in 0..4usize {
        let sb0 = group * 2;
        let sb1 = sb0 + 1;
        let sc0 = sc[sb0] as i32;
        let sc1 = sc[sb1] as i32;
        let mn0 = mn[sb0] as i32;
        let mn1 = mn[sb1] as i32;
        let qbytes = vld1q_u8(qs.add(group * 32));
        let qbytes_hi = vld1q_u8(qs.add(group * 32 + 16));
        let w0_a = vreinterpretq_s8_u8(vandq_u8(qbytes, mask_low));
        let w0_b = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
        let w1_a = vreinterpretq_s8_u8(vshrq_n_u8(qbytes, 4));
        let w1_b = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
        let x_elem0 = sb0 * 32;
        let x_elem1 = sb1 * 32;

        for t in 0..8usize {
            let input_off = ((seq_start + t) * cols + bi) * 256;
            let x = input_qs.as_ptr().add(input_off);

            let x0_a = vld1q_s8(x.add(x_elem0));
            let x0_b = vld1q_s8(x.add(x_elem0 + 16));
            let dot0 = vdotq_s32(vdotq_s32(zero, w0_a, x0_a), w0_b, x0_b);
            acc_v[t] = vmlaq_n_s32(acc_v[t], dot0, sc0);

            let x1_a = vld1q_s8(x.add(x_elem1));
            let x1_b = vld1q_s8(x.add(x_elem1 + 16));
            let dot1 = vdotq_s32(vdotq_s32(zero, w1_a, x1_a), w1_b, x1_b);
            acc_v[t] = vmlaq_n_s32(acc_v[t], dot1, sc1);

            let bs = input_bsums.as_ptr().add(((seq_start + t) * cols + bi) * 8);
            summ[t] += mn0 * *bs.add(sb0) as i32 + mn1 * *bs.add(sb1) as i32;
        }
    }

    for t in 0..8usize {
        let idx = (seq_start + t) * cols + bi;
        let sumi = vaddvq_s32(acc_v[t]);
        acc[t] += input_d[idx] * (d * sumi as f32 - dmin * summ[t] as f32);
    }
}

fn q4k_raw_meta_dot(block: &[u8], x_qs: &[i8], x_d: f32, x_bsums: &[i16]) -> f32 {
    let qs = &block[..Q4K_RAW_META_QS_BYTES];
    let meta = &block[Q4K_RAW_META_QS_BYTES..Q4K_RAW_META_BLOCK_BYTES];
    let sc = &meta[..8];
    let mn = &meta[8..16];
    let d = f32::from_le_bytes(meta[16..20].try_into().unwrap());
    let dmin = f32::from_le_bytes(meta[20..24].try_into().unwrap());

    let mut sumi = 0i32;
    let mut summ = 0i32;
    for sb in 0..8usize {
        let mut dot = 0i32;
        let q_group = sb / 2;
        let q_off = q_group * 32;
        let elem_off = sb * 32;
        if sb % 2 == 0 {
            for k in 0..32usize {
                dot += (qs[q_off + k] & 0x0f) as i32 * x_qs[elem_off + k] as i32;
            }
        } else {
            for k in 0..32usize {
                dot += (qs[q_off + k] >> 4) as i32 * x_qs[elem_off + k] as i32;
            }
        }
        sumi += sc[sb] as i32 * dot;
        summ += mn[sb] as i32 * x_bsums[sb] as i32;
    }

    x_d * (d * sumi as f32 - dmin * summ as f32)
}

#[inline]
fn is_q4k_compact_len(len: usize, rows: usize, cols: usize) -> bool {
    len == rows.div_ceil(8) * cols * Q4K_COMPACT_BLOCK_BYTES
}

// ─── 스칼라 레퍼런스 ──────────────────────────────────────────────────────────

fn gemm_q4k_packed_scalar(
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
                let blk_off = (rg * cols + bi) * Q4K_PACKED_BLOCK_BYTES;
                let blk = &packed[blk_off..blk_off + Q4K_PACKED_BLOCK_BYTES];

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
                        blk[Q4K_D_OFF + nr * 4..Q4K_D_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let dmin = f32::from_le_bytes(
                        blk[Q4K_DMIN_OFF + nr * 4..Q4K_DMIN_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );

                    let sc_base = Q4K_SC_RAW_OFF + nr * 8;
                    let mn_base = Q4K_MN_RAW_OFF + nr * 8;

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
                            let qs_off = Q4K_QS_OFF + pair * 512 + chunk * 16 + is_odd * 8 + within;
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

fn gemm_q4k_compact(
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

    let out_addr = output.as_mut_ptr() as usize;
    let output_len = output.len();

    (0..rows).into_par_iter().for_each(|row| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, output_len) };
        for s in 0..seq_len {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let x_base = (s * cols + bi) * 256;
                let bsum_base = (s * cols + bi) * 8;
                let rg = row / 8;
                let nr = row % 8;
                let blk_off = (rg * cols + bi) * Q4K_COMPACT_BLOCK_BYTES;
                let blk = &packed[blk_off..blk_off + Q4K_COMPACT_BLOCK_BYTES];
                acc += q4k_compact_dot(
                    blk,
                    nr,
                    &input_qs[x_base..x_base + 256],
                    input_d[s * cols + bi],
                    &input_bsums[bsum_base..bsum_base + 8],
                );
            }
            out_slice[s * rows + row] += acc;
        }
    });
}

fn gemv_q4k_compact(
    packed: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    use rayon::prelude::*;

    const ROWS_PER_TASK: usize = 512;

    output[..rows]
        .par_chunks_mut(ROWS_PER_TASK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * ROWS_PER_TASK;
            for (local, out) in out_chunk.iter_mut().enumerate() {
                let row = row_start + local;
                let rg = row / 8;
                let nr = row % 8;
                let mut acc = 0.0f32;
                for bi in 0..cols {
                    let blk_off = (rg * cols + bi) * Q4K_COMPACT_BLOCK_BYTES;
                    let blk = &packed[blk_off..blk_off + Q4K_COMPACT_BLOCK_BYTES];
                    acc += q4k_compact_dot(
                        blk,
                        nr,
                        &input_qs[bi * 256..bi * 256 + 256],
                        input_d[bi],
                        &input_bsums[bi * 8..bi * 8 + 8],
                    );
                }
                *out += acc;
            }
        });
}

#[inline(always)]
fn q4k_compact_dot(blk: &[u8], nr: usize, x_qs: &[i8], x_d: f32, x_bsums: &[i16]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { q4k_compact_dot_neon(blk, nr, x_qs, x_d, x_bsums) };
        }
    }

    q4k_compact_dot_scalar(blk, nr, x_qs, x_d, x_bsums)
}

fn q4k_compact_dot_scalar(blk: &[u8], nr: usize, x_qs: &[i8], x_d: f32, x_bsums: &[i16]) -> f32 {
    let d_off = Q4K_COMPACT_D_OFF + nr * 4;
    let d = f32::from_le_bytes(blk[d_off..d_off + 4].try_into().unwrap());
    let dmin_off = Q4K_COMPACT_DMIN_OFF + nr * 4;
    let dmin = f32::from_le_bytes(blk[dmin_off..dmin_off + 4].try_into().unwrap());
    let sc = &blk[Q4K_COMPACT_SC_RAW_OFF + nr * 8..Q4K_COMPACT_SC_RAW_OFF + nr * 8 + 8];
    let mn = &blk[Q4K_COMPACT_MN_RAW_OFF + nr * 8..Q4K_COMPACT_MN_RAW_OFF + nr * 8 + 8];
    let qs = &blk[Q4K_COMPACT_QS_OFF + nr * 128..Q4K_COMPACT_QS_OFF + nr * 128 + 128];

    let mut sumi = 0i32;
    let mut summ = 0i32;
    for g in 0..4usize {
        let q_off = g * 32;
        let x_off = g * 64;
        let sb0 = g * 2;
        let sb1 = sb0 + 1;
        let mut dot0 = 0i32;
        let mut dot1 = 0i32;
        for k in 0..32usize {
            let q = qs[q_off + k];
            dot0 += (q & 0x0f) as i32 * x_qs[x_off + k] as i32;
            dot1 += (q >> 4) as i32 * x_qs[x_off + 32 + k] as i32;
        }
        sumi += dot0 * sc[sb0] as i32 + dot1 * sc[sb1] as i32;
        summ += mn[sb0] as i32 * x_bsums[sb0] as i32 + mn[sb1] as i32 * x_bsums[sb1] as i32;
    }

    x_d * (d * sumi as f32 - dmin * summ as f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k_compact_dot_neon(
    blk: &[u8],
    nr: usize,
    x_qs: &[i8],
    x_d: f32,
    x_bsums: &[i16],
) -> f32 {
    use std::arch::aarch64::*;

    let bp = blk.as_ptr();
    let qs = bp.add(Q4K_COMPACT_QS_OFF + nr * 128);
    let sc = bp.add(Q4K_COMPACT_SC_RAW_OFF + nr * 8);
    let mn = bp.add(Q4K_COMPACT_MN_RAW_OFF + nr * 8);
    let d = f32::from_bits((bp.add(Q4K_COMPACT_D_OFF + nr * 4) as *const u32).read_unaligned());
    let dmin =
        f32::from_bits((bp.add(Q4K_COMPACT_DMIN_OFF + nr * 4) as *const u32).read_unaligned());

    let mask_low = vdupq_n_u8(0x0f);
    let vzero = vdupq_n_s32(0);
    let mut acc_v = vzero;
    let mut summ = 0i32;

    for g in 0..4usize {
        let q_off = g * 32;
        let x_off = g * 64;
        let sb0 = g * 2;
        let sb1 = sb0 + 1;

        let q0 = vld1q_u8(qs.add(q_off));
        let q1 = vld1q_u8(qs.add(q_off + 16));
        let w_lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask_low));
        let w_lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask_low));
        let w_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
        let w_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

        let x_lo0 = vld1q_s8(x_qs.as_ptr().add(x_off));
        let x_lo1 = vld1q_s8(x_qs.as_ptr().add(x_off + 16));
        let x_hi0 = vld1q_s8(x_qs.as_ptr().add(x_off + 32));
        let x_hi1 = vld1q_s8(x_qs.as_ptr().add(x_off + 48));

        let mut p0 = vzero;
        p0 = vdotq_s32(p0, w_lo0, x_lo0);
        p0 = vdotq_s32(p0, w_lo1, x_lo1);
        let mut p1 = vzero;
        p1 = vdotq_s32(p1, w_hi0, x_hi0);
        p1 = vdotq_s32(p1, w_hi1, x_hi1);

        acc_v = vmlaq_n_s32(acc_v, p0, *sc.add(sb0) as i32);
        acc_v = vmlaq_n_s32(acc_v, p1, *sc.add(sb1) as i32);
        summ +=
            *mn.add(sb0) as i32 * x_bsums[sb0] as i32 + *mn.add(sb1) as i32 * x_bsums[sb1] as i32;
    }

    let sumi = vaddvq_s32(acc_v);
    x_d * (d * sumi as f32 - dmin * summ as f32)
}

// ─── NEON vdotq_s32 커널 ─────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q4k_packed_neon(
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
            let blk_off = (rg * cols + bi) * Q4K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q4K_PACKED_BLOCK_BYTES];

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
                let pair_base = Q4K_QS_OFF + pair * 512;
                for k in 0..32usize {
                    let src_off = pair_base + k * 16 + is_odd * 8;
                    w_qs[nr][k * 8..k * 8 + 8].copy_from_slice(&blk[src_off..src_off + 8]);
                }

                // Copy sc_raw, mn_raw
                let sc_off = Q4K_SC_RAW_OFF + nr * 8;
                let mn_off = Q4K_MN_RAW_OFF + nr * 8;
                sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                d[nr] = f32::from_le_bytes(
                    blk[Q4K_D_OFF + nr * 4..Q4K_D_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                dmin[nr] = f32::from_le_bytes(
                    blk[Q4K_DMIN_OFF + nr * 4..Q4K_DMIN_OFF + nr * 4 + 4]
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
//
// vmmlaq_s32(acc, a, b): signed 8-bit matrix multiply-accumulate
//   a = [row0: 8 elems | row1: 8 elems] (int8x16)
//   b = [col0: 8 elems | col1: 8 elems] (int8x16)
//   acc[0] += dot(row0, col0)
//   acc[1] += dot(row0, col1)
//   acc[2] += dot(row1, col0)
//   acc[3] += dot(row1, col1)
//
// Weight is already row-pair interleaved → vld1q_s8 loads [r_even:8|r_odd:8].
// For single-token: input = [x:8|zeros:8] for smmla.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_q4k_packed_i8mm(
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
            let blk_off = (rg * cols + bi) * Q4K_PACKED_BLOCK_BYTES;
            let blk = &packed[blk_off..blk_off + Q4K_PACKED_BLOCK_BYTES];
            let blk_ptr = blk.as_ptr() as *const i8;

            // --- Phase 1: Load raw scales/mins/d/dmin to stack ---
            let mut sc = [[0u8; 8]; 8];
            let mut mn = [[0u8; 8]; 8];
            let mut d = [0.0f32; 8];
            let mut dmin = [0.0f32; 8];

            for nr in 0..8 {
                let sc_off = Q4K_SC_RAW_OFF + nr * 8;
                let mn_off = Q4K_MN_RAW_OFF + nr * 8;
                sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                d[nr] = f32::from_le_bytes(
                    blk[Q4K_D_OFF + nr * 4..Q4K_D_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                dmin[nr] = f32::from_le_bytes(
                    blk[Q4K_DMIN_OFF + nr * 4..Q4K_DMIN_OFF + nr * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
            }

            // --- Phase 2: Tiered token batching ---
            // 8-token → 2-token → 1-token fallback.
            // smmla computes 2×2 tile (2 rows × 2 tokens).
            // 8-token: load weight once, 4× smmla (4 token-pairs share weight).
            // This reduces weight memory bandwidth by 4× vs 2-token.
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
                    {
                        let mut dots = [[[0i32; 8]; 2]; 8];
                        let w_pair_base = Q4K_QS_OFF + rp * 512;

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

                        let sc_r0_u16 = vmovl_u8(vld1_u8(sc[r0].as_ptr()));
                        let sc_r0_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc_r0_u16)));
                        let sc_r0_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc_r0_u16)));
                        let mn_r0_u16 = vmovl_u8(vld1_u8(mn[r0].as_ptr()));
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
                            let sc_r1_u16 = vmovl_u8(vld1_u8(sc[r1].as_ptr()));
                            let sc_r1_lo =
                                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc_r1_u16)));
                            let sc_r1_hi =
                                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc_r1_u16)));
                            let mn_r1_u16 = vmovl_u8(vld1_u8(mn[r1].as_ptr()));
                            let mn_r1_lo =
                                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mn_r1_u16)));
                            let mn_r1_hi =
                                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mn_r1_u16)));
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

                    let w_pair_base = Q4K_QS_OFF + rp * 512;

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

                    let w_pair_base = Q4K_QS_OFF + rp * 512;

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
#[inline(always)]
unsafe fn q4k_pair_dup_i32x4(a: i32, b: i32) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let mut v = vdupq_n_s32(a);
    v = vsetq_lane_s32::<1>(a, v);
    v = vsetq_lane_s32::<2>(b, v);
    v = vsetq_lane_s32::<3>(b, v);
    v
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn q4k_pair_dup_f32x4(a: f32, b: f32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let mut v = vdupq_n_f32(a);
    v = vsetq_lane_f32::<1>(a, v);
    v = vsetq_lane_f32::<2>(b, v);
    v = vsetq_lane_f32::<3>(b, v);
    v
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemv_q4k_packed_i8mm(
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

    const ROW_GROUPS_PER_TASK: usize = 64;

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
                    let blk_off = (rg * cols + bi) * Q4K_PACKED_BLOCK_BYTES;
                    let blk = &packed[blk_off..blk_off + Q4K_PACKED_BLOCK_BYTES];
                    let blk_ptr = blk.as_ptr() as *const i8;

                    let mut sc = [[0u8; 8]; 8];
                    let mut mn = [[0u8; 8]; 8];
                    let mut d = [0.0f32; 8];
                    let mut dmin = [0.0f32; 8];

                    for nr in 0..rows_in_group {
                        let sc_off = Q4K_SC_RAW_OFF + nr * 8;
                        let mn_off = Q4K_MN_RAW_OFF + nr * 8;
                        sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                        mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);

                        d[nr] = f32::from_le_bytes(
                            blk[Q4K_D_OFF + nr * 4..Q4K_D_OFF + nr * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        dmin[nr] = f32::from_le_bytes(
                            blk[Q4K_DMIN_OFF + nr * 4..Q4K_DMIN_OFF + nr * 4 + 4]
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

                        let mut visum = vdupq_n_s32(0);
                        let mut vbias = vdupq_n_s32(0);

                        let w_pair_base = Q4K_QS_OFF + rp * 512;

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

                            let sc_pair = q4k_pair_dup_i32x4(
                                sc[r0][sb] as i32,
                                if has_r1 { sc[r1][sb] as i32 } else { 0 },
                            );
                            visum = vmlaq_s32(visum, smmla_acc, sc_pair);

                            let bsum = x_bsums[sb] as i32;
                            let mn_pair = q4k_pair_dup_i32x4(
                                mn[r0][sb] as i32 * bsum,
                                if has_r1 { mn[r1][sb] as i32 * bsum } else { 0 },
                            );
                            vbias = vaddq_s32(vbias, mn_pair);
                        }

                        let vscaled = vmlsq_f32(
                            vmulq_f32(
                                vcvtq_f32_s32(visum),
                                q4k_pair_dup_f32x4(
                                    d[r0] * x_d_val,
                                    if has_r1 { d[r1] * x_d_val } else { 0.0 },
                                ),
                            ),
                            vcvtq_f32_s32(vbias),
                            q4k_pair_dup_f32x4(
                                dmin[r0] * x_d_val,
                                if has_r1 { dmin[r1] * x_d_val } else { 0.0 },
                            ),
                        );

                        acc[r0] += vgetq_lane_f32::<0>(vscaled);
                        if has_r1 {
                            acc[r1] += vgetq_lane_f32::<2>(vscaled);
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
    use crate::gemm::pack_q4k::{
        pack_q4k, pack_q4k_compact, pack_q4k_from_raw_meta, pack_q4k_raw_meta,
    };
    use half::f16;

    // ─── 테스트 헬퍼 ─────────────────────────────────────────────

    /// Q4_K 더미 블록 생성 (144 bytes)
    fn make_q4k_block(d_val: f32, dmin_val: f32, scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..144].copy_from_slice(&qs);
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

    /// Q4_K 블록을 f32[256]으로 dequant
    ///
    /// val = scale[sb] * nibble_unsigned - min[sb]
    fn dequant_q4k_block(block: &[u8]) -> [f32; 256] {
        use crate::gemm::pack_q4k::decode_q4k_scales;
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales_raw: &[u8; 12] = block[4..16].try_into().unwrap();
        let qs_raw: &[u8; 128] = block[16..144].try_into().unwrap();

        let (sc_f32, mn_f32) = decode_q4k_scales(scales_raw, d, dmin);

        let mut out = [0.0f32; 256];
        let mut q_off = 0usize;
        let mut y_off = 0usize;
        for group in 0..4usize {
            let sb0 = group * 2;
            let sb1 = group * 2 + 1;
            for l in 0..32usize {
                let lo = (qs_raw[q_off + l] & 0x0F) as u32;
                let hi = (qs_raw[q_off + l] >> 4) as u32;
                out[y_off + l] = sc_f32[sb0] * lo as f32 - mn_f32[sb0];
                out[y_off + 32 + l] = sc_f32[sb1] * hi as f32 - mn_f32[sb1];
            }
            q_off += 32;
            y_off += 64;
        }
        out
    }

    fn unpack_q4k_unsigned(qs: &[u8; 128], out: &mut [u8; 256]) {
        let mut q_off = 0usize;
        let mut y_off = 0usize;

        for _ in 0..4 {
            for l in 0..32 {
                out[y_off + l] = qs[q_off + l] & 0x0F;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = qs[q_off + l] >> 4;
            }
            q_off += 32;
            y_off += 64;
        }
    }

    fn q4k_q8k_exact_ref(
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
                    let src_off = (row * cols + bi) * 144;
                    let block = &src[src_off..src_off + 144];
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                    let qs_raw: &[u8; 128] = block[16..144].try_into().unwrap();
                    let (sc_raw, mn_raw) = crate::gemm::pack_q4k::decode_q4k_scales_raw(scales_12);

                    let mut w_unsigned = [0u8; 256];
                    unpack_q4k_unsigned(qs_raw, &mut w_unsigned);

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
    fn test_q4k_compact_matches_exact_q8_reference() {
        let rows = 11;
        let cols = 3;
        let seq_len = 2;

        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let d = 0.01 * (row + 1) as f32;
                let dmin = 0.004 * (col + 1) as f32;
                let mut scales = [0u8; 12];
                for (i, v) in scales.iter_mut().enumerate() {
                    *v = ((row * 7 + col * 11 + i * 3 + 5) & 0x3f) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, v) in qs.iter_mut().enumerate() {
                    let lo = ((row * 13 + col * 17 + i * 5 + 1) & 0x0f) as u8;
                    let hi = ((row * 19 + col * 23 + i * 7 + 9) & 0x0f) as u8;
                    *v = lo | (hi << 4);
                }
                src.extend_from_slice(&make_q4k_block(d, dmin, scales, qs));
            }
        }

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        let mut input_d = vec![0.0f32; seq_len * cols];
        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        for s in 0..seq_len {
            for col in 0..cols {
                let mut x = [0.0f32; 256];
                for (i, v) in x.iter_mut().enumerate() {
                    *v = (((s * 31 + col * 17 + i * 3) % 41) as f32 - 20.0) / 9.0;
                }
                let (qs, d, bsums) = quantize_q8k(&x);
                let off = (s * cols + col) * 256;
                input_qs[off..off + 256].copy_from_slice(&qs);
                input_d[s * cols + col] = d;
                let bs_off = (s * cols + col) * 8;
                input_bsums[bs_off..bs_off + 8].copy_from_slice(&bsums);
            }
        }

        let packed = pack_q4k_compact(&src, rows, cols);
        let mut output = vec![0.0f32; seq_len * rows];
        gemm_q4k_packed(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        let expected =
            q4k_q8k_exact_ref(&src, &input_qs, &input_d, &input_bsums, rows, cols, seq_len);
        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff <= 1e-4,
                "output[{i}] got {got}, want {want}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_q4k_raw_meta_gemm_matches_exact_q8_reference_for_prefill() {
        let rows = 11;
        let cols = 3;
        let seq_len = 5;

        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let d = 0.006 * (row + 2) as f32;
                let dmin = 0.003 * (col + 1) as f32;
                let mut scales = [0u8; 12];
                for (i, v) in scales.iter_mut().enumerate() {
                    *v = ((row * 5 + col * 13 + i * 7 + 3) & 0x3f) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, v) in qs.iter_mut().enumerate() {
                    let lo = ((row * 11 + col * 3 + i * 9 + 2) & 0x0f) as u8;
                    let hi = ((row * 7 + col * 19 + i * 5 + 10) & 0x0f) as u8;
                    *v = lo | (hi << 4);
                }
                src.extend_from_slice(&make_q4k_block(d, dmin, scales, qs));
            }
        }

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        let mut input_d = vec![0.0f32; seq_len * cols];
        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        for s in 0..seq_len {
            for col in 0..cols {
                let mut x = [0.0f32; 256];
                for (i, v) in x.iter_mut().enumerate() {
                    *v = (((s * 29 + col * 31 + i * 11) % 53) as f32 - 26.0) / 11.0;
                }
                let (qs, d, bsums) = quantize_q8k(&x);
                let off = (s * cols + col) * 256;
                input_qs[off..off + 256].copy_from_slice(&qs);
                input_d[s * cols + col] = d;
                let bs_off = (s * cols + col) * 8;
                input_bsums[bs_off..bs_off + 8].copy_from_slice(&bsums);
            }
        }

        let packed = pack_q4k_raw_meta(&src, rows, cols);
        let mut output = vec![0.0f32; seq_len * rows];
        gemm_q4k_raw_meta(
            &packed,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut output,
            rows,
            cols,
            seq_len,
        );

        let expected =
            q4k_q8k_exact_ref(&src, &input_qs, &input_d, &input_bsums, rows, cols, seq_len);
        for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff <= 1e-4,
                "output[{i}] got {got}, want {want}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_q4k_from_raw_meta_matches_direct_packed_gemm() {
        let rows = 13;
        let cols = 3;
        let seq_len = 5;

        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let d = 0.004 * (row + 3) as f32;
                let dmin = 0.002 * (col + 2) as f32;
                let mut scales = [0u8; 12];
                for (i, v) in scales.iter_mut().enumerate() {
                    *v = ((row * 17 + col * 7 + i * 5 + 11) & 0x3f) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, v) in qs.iter_mut().enumerate() {
                    let lo = ((row * 3 + col * 29 + i * 7 + 4) & 0x0f) as u8;
                    let hi = ((row * 23 + col * 5 + i * 11 + 1) & 0x0f) as u8;
                    *v = lo | (hi << 4);
                }
                src.extend_from_slice(&make_q4k_block(d, dmin, scales, qs));
            }
        }

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        let mut input_d = vec![0.0f32; seq_len * cols];
        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        for s in 0..seq_len {
            for col in 0..cols {
                let mut x = [0.0f32; 256];
                for (i, v) in x.iter_mut().enumerate() {
                    *v = (((s * 13 + col * 19 + i * 17) % 67) as f32 - 33.0) / 13.0;
                }
                let (qs, d, bsums) = quantize_q8k(&x);
                let off = (s * cols + col) * 256;
                input_qs[off..off + 256].copy_from_slice(&qs);
                input_d[s * cols + col] = d;
                let bs_off = (s * cols + col) * 8;
                input_bsums[bs_off..bs_off + 8].copy_from_slice(&bsums);
            }
        }

        let raw_meta = pack_q4k_raw_meta(&src, rows, cols);
        let from_raw_meta = pack_q4k_from_raw_meta(&raw_meta, rows, cols);
        let direct = pack_q4k(&src, rows, cols);
        assert_eq!(from_raw_meta, direct);

        let mut raw_meta_output = vec![0.0f32; seq_len * rows];
        gemm_q4k_raw_meta(
            &raw_meta,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut raw_meta_output,
            rows,
            cols,
            seq_len,
        );

        let mut packed_output = vec![0.0f32; seq_len * rows];
        gemm_q4k_packed(
            &from_raw_meta,
            &input_qs,
            &input_d,
            &input_bsums,
            &mut packed_output,
            rows,
            cols,
            seq_len,
        );

        for (i, (&got, &want)) in packed_output.iter().zip(raw_meta_output.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff <= 1e-4,
                "output[{i}] got {got}, want {want}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_gemm_zero_input() {
        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let block = make_q4k_block(1.0, 0.5, [10u8; 12], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q4k(&src, rows, cols);

        let input_qs = vec![0i8; seq_len * cols * 256];
        let input_d = vec![1.0f32; seq_len * cols];
        let input_bsums = vec![0i16; seq_len * cols * 8];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q4k_packed(
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
        let block = make_q4k_block(1.0, 1.0, [0u8; 12], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q4k(&src, rows, cols);

        let input_qs = vec![0i8; seq_len * cols * 256];
        let input_d = vec![1.0f32; seq_len * cols];
        let input_bsums = vec![100i16; seq_len * cols * 8];
        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q4k_packed(
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
        // qs 전부 0x88 (→ unsigned 8), d=1.0, dmin=0.0
        // scales[0]=10 → sc_raw[0]=10, mn_raw[0]=0
        //
        // input_qs: sub-block 0 원소 모두 1 (32개), 나머지 0
        //   bsums[0] = 32, bsums[1..] = 0
        //   x_d = 1.0
        //
        // 새 수식 (unsigned nibble):
        //   sumi = sc_raw[0] * dot(w_unsigned=8, x=1, 32개) = 10 * (8*32) = 2560
        //   summ = mn_raw[0] * bsum[0] = 0 * 32 = 0
        //   out = x_d * (d * sumi - dmin * summ) = 1.0 * (1.0 * 2560 - 0) = 2560
        //
        // 실제 dequant: d*sc_raw*unsigned - dmin*mn_raw = 1.0*10*8 - 0 = 80.0 per elem
        // f32 내적: sum(80.0 * 1.0/d_q8k for 32 elements)
        // 하지만 Q8K에서 x_d=1.0이고 x_qs=1이므로 실제 x_f32 ≈ 1/127 ≈ 0.00787...
        // 실제 내적 = 80 * 32 * (1/127) * ... 아 여기서 x_d는 Q8K의 d인데,
        // input_qs=1, input_d=1.0이면 x_f32_approx ≈ 1.0 * 1 = 1.0? 아니, Q8K에서
        // 실제 값 = d * qs 이므로 x_f32 = 1.0 * 1 = 1.0.
        //
        // 그러므로 실제 GEMM = Σ(w_dequant * x_f32) = Σ(80 * 1.0) = 80*32 = 2560 ✓

        let rows = 8;
        let cols = 1;
        let seq_len = 1;

        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        let block = make_q4k_block(1.0, 0.0, scales_raw, [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q4k(&src, rows, cols);

        let mut input_qs = vec![0i8; seq_len * cols * 256];
        for k in 0..32 {
            input_qs[k] = 1;
        }
        let input_d = vec![1.0f32; seq_len * cols];
        let mut input_bsums = vec![0i16; seq_len * cols * 8];
        input_bsums[0] = 32;

        let mut output = vec![0.0f32; seq_len * rows];

        gemm_q4k_packed(
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

        let block = make_q4k_block(0.01, 0.005, scales_raw, qs_data);
        let src: Vec<u8> = block.repeat(rows * cols);
        let packed = pack_q4k(&src, rows, cols);

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
        gemm_q4k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        // f32 레퍼런스
        let mut output_ref = vec![0.0f32; seq_len * rows];

        let mut weight_f32 = vec![0.0f32; rows * cols * 256];
        for row in 0..rows {
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 144;
                let w_blk = dequant_q4k_block(&src[src_off..src_off + 144]);
                let w_off = (row * cols + bi) * 256;
                weight_f32[w_off..w_off + 256].copy_from_slice(&w_blk);
            }
        }

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

    // ─── 테스트 NEW: 행마다 다른 weight로 interleaving 검증 ────

    #[test]
    fn test_gemm_unique_rows() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

        // 각 row, 각 col마다 DIFFERENT Q4_K 블록 생성
        // row_idx와 col_idx를 seed로 사용해서 d, dmin, scales, qs를 다르게 만듦
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

                // 각 row마다 다른 qs nibbles
                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }

                let block = make_q4k_block(d_val, dmin_val, scales, qs);
                src.extend_from_slice(&block);
            }
        }

        let packed = pack_q4k(&src, rows, cols);

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
        gemm_q4k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        // f32 레퍼런스: dequant → dot product
        let mut weight_f32 = vec![0.0f32; rows * cols * 256];
        for row in 0..rows {
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 144;
                let w_blk = dequant_q4k_block(&src[src_off..src_off + 144]);
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

        // 비교: 행마다 다른 weight이므로 interleaving 버그가 있으면 여기서 잡힘
        let mut max_rel_err = 0.0f32;
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
                assert!(
                    rel_err < 0.15 || abs_err < 0.1,
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

        eprintln!("test_gemm_unique_rows: max_rel_err = {max_rel_err:.6}");
    }

    #[test]
    fn test_q4k_packed_matches_exact_q8_reference() {
        use std::f32::consts::PI;

        let rows = 24;
        let cols = 3;
        let seq_len = 15;

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

                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
                }

                src.extend_from_slice(&make_q4k_block(d_val, dmin_val, scales, qs));
            }
        }

        let packed = pack_q4k(&src, rows, cols);
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
        gemm_q4k_packed(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_gemm,
            rows,
            cols,
            seq_len,
        );

        let output_ref = q4k_q8k_exact_ref(
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

        eprintln!("test_q4k_packed_matches_exact_q8_reference: max_abs_err = {max_abs_err:.8}");
    }

    // ─── 테스트 NEW2: i8mm 커널 로직 scalar 시뮬레이션 ─────────

    /// vmmlaq_s32 scalar 시뮬레이션
    /// a = [a_row0:8 | a_row1:8], b = [b_col0:8 | b_col1:8]
    /// acc[0] += dot(a_row0, b_col0)
    /// acc[1] += dot(a_row0, b_col1)
    /// acc[2] += dot(a_row1, b_col0)
    /// acc[3] += dot(a_row1, b_col1)
    fn smmla_scalar(acc: &mut [i32; 4], a: &[i8; 16], b: &[i8; 16]) {
        for i in 0..8 {
            acc[0] += a[i] as i32 * b[i] as i32;
            acc[1] += a[i] as i32 * b[8 + i] as i32;
            acc[2] += a[8 + i] as i32 * b[i] as i32;
            acc[3] += a[8 + i] as i32 * b[8 + i] as i32;
        }
    }

    /// i8mm 커널 로직을 순수 scalar로 시뮬레이션
    fn gemm_q4k_simulate_i8mm(
        packed: &[u8],
        input_qs: &[i8],
        input_d: &[f32],
        input_bsums: &[i16],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) {
        use crate::gemm::pack_q4k::*;
        let row_groups = rows.div_ceil(8);

        for rg in 0..row_groups {
            let rows_in_group = (rows - rg * 8).min(8);
            let row_pairs = (rows_in_group + 1) / 2;
            let mut acc = vec![0.0f32; seq_len * 8];

            for bi in 0..cols {
                let blk_off = (rg * cols + bi) * Q4K_PACKED_BLOCK_BYTES;
                let blk = &packed[blk_off..blk_off + Q4K_PACKED_BLOCK_BYTES];

                let mut sc = [[0u8; 8]; 8];
                let mut mn = [[0u8; 8]; 8];
                let mut d = [0.0f32; 8];
                let mut dmin = [0.0f32; 8];

                for nr in 0..8 {
                    let sc_off = Q4K_SC_RAW_OFF + nr * 8;
                    let mn_off = Q4K_MN_RAW_OFF + nr * 8;
                    sc[nr].copy_from_slice(&blk[sc_off..sc_off + 8]);
                    mn[nr].copy_from_slice(&blk[mn_off..mn_off + 8]);
                    d[nr] = f32::from_le_bytes(
                        blk[Q4K_D_OFF + nr * 4..Q4K_D_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    dmin[nr] = f32::from_le_bytes(
                        blk[Q4K_DMIN_OFF + nr * 4..Q4K_DMIN_OFF + nr * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                }

                // 2-token batching (mirrors the real i8mm kernel)
                let mut s = 0usize;

                // ── 2-token path ──
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

                        let w_pair_base = Q4K_QS_OFF + rp * 512;

                        for sb in 0..8usize {
                            let k_off = sb * 32;
                            let mut smmla_acc = [0i32; 4];

                            for ki in 0..4usize {
                                let elem_off = k_off + ki * 8;
                                let chunk = elem_off / 8;
                                let w_off = w_pair_base + chunk * 16;

                                let mut w_bytes = [0i8; 16];
                                for i in 0..16 {
                                    w_bytes[i] = blk[w_off + i] as i8;
                                }

                                // Pack 2 tokens: [token_s:8 | token_s+1:8]
                                let mut x_pair = [0i8; 16];
                                for i in 0..8 {
                                    x_pair[i] = x_qs0[elem_off + i];
                                    x_pair[8 + i] = x_qs1[elem_off + i];
                                }

                                smmla_scalar(&mut smmla_acc, &w_bytes, &x_pair);
                            }

                            let dot_r0_s0 = smmla_acc[0];
                            let dot_r0_s1 = smmla_acc[1];
                            let dot_r1_s0 = smmla_acc[2];
                            let dot_r1_s1 = smmla_acc[3];

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

                // ── 1-token remainder ──
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

                        let w_pair_base = Q4K_QS_OFF + rp * 512;

                        for sb in 0..8usize {
                            let k_off = sb * 32;
                            let mut smmla_acc = [0i32; 4];

                            for ki in 0..4usize {
                                let elem_off = k_off + ki * 8;
                                let chunk = elem_off / 8;
                                let w_off = w_pair_base + chunk * 16;

                                let mut w_bytes = [0i8; 16];
                                for i in 0..16 {
                                    w_bytes[i] = blk[w_off + i] as i8;
                                }

                                let mut x_padded = [0i8; 16];
                                for i in 0..8 {
                                    x_padded[i] = x_qs[elem_off + i];
                                }

                                smmla_scalar(&mut smmla_acc, &w_bytes, &x_padded);
                            }

                            let dot_r0 = smmla_acc[0];
                            let dot_r1 = smmla_acc[2];

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

            for s in 0..seq_len {
                for r in 0..rows_in_group {
                    output[s * rows + rg * 8 + r] += acc[s * 8 + r];
                }
            }
        }
    }

    #[test]
    fn test_gemm_i8mm_sim_unique_rows() {
        use std::f32::consts::PI;

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

        // 각 row, 각 col마다 DIFFERENT Q4_K 블록 생성
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

                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }

                let block = make_q4k_block(d_val, dmin_val, scales, qs);
                src.extend_from_slice(&block);
            }
        }

        let packed = pack_q4k(&src, rows, cols);

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

        // scalar 커널
        let mut output_scalar = vec![0.0f32; seq_len * rows];
        gemm_q4k_packed_scalar(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_scalar,
            rows,
            cols,
            seq_len,
        );

        // i8mm 시뮬레이션
        let mut output_i8mm = vec![0.0f32; seq_len * rows];
        gemm_q4k_simulate_i8mm(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_i8mm,
            rows,
            cols,
            seq_len,
        );

        // f32 레퍼런스
        let mut weight_f32 = vec![0.0f32; rows * cols * 256];
        for row in 0..rows {
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 144;
                let w_blk = dequant_q4k_block(&src[src_off..src_off + 144]);
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

        // i8mm sim vs scalar 비교 (정확히 같아야 함, 단 i8mm은 smmla arg order 이슈)
        let mut i8mm_vs_scalar_ok = true;
        for s in 0..seq_len {
            for row in 0..rows {
                let i8mm_val = output_i8mm[s * rows + row];
                let scalar_val = output_scalar[s * rows + row];
                let diff = (i8mm_val - scalar_val).abs();
                if diff > 0.01 {
                    eprintln!(
                        "i8mm_sim vs scalar MISMATCH: s={s} row={row}: \
                         i8mm={i8mm_val:.6}, scalar={scalar_val:.6}, diff={diff:.6}"
                    );
                    i8mm_vs_scalar_ok = false;
                }
            }
        }

        // i8mm sim vs f32 ref 비교
        let mut max_rel_err_i8mm = 0.0f32;
        for s in 0..seq_len {
            for row in 0..rows {
                let got = output_i8mm[s * rows + row];
                let exp = output_ref[s * rows + row];
                let abs_err = (got - exp).abs();
                let rel_err = if exp.abs() > 1e-6 {
                    abs_err / exp.abs()
                } else {
                    abs_err
                };
                if rel_err > max_rel_err_i8mm {
                    max_rel_err_i8mm = rel_err;
                }
                assert!(
                    rel_err < 0.15 || abs_err < 0.1,
                    "i8mm_sim vs ref: s={s} row={row}: got={got:.6}, ref={exp:.6}, \
                     abs_err={abs_err:.6}, rel_err={rel_err:.4}"
                );
            }
        }

        assert!(
            i8mm_vs_scalar_ok,
            "i8mm simulation produced different results from scalar kernel — \
             smmla addressing bug detected!"
        );

        eprintln!("test_gemm_i8mm_sim_unique_rows: max_rel_err={max_rel_err_i8mm:.6}");
    }

    // ─── 테스트 NEW3: 직접 i8mm/neon 커널 호출 + unique rows ───

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_i8mm_direct_unique_rows() {
        use std::f32::consts::PI;

        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            eprintln!("SKIP: i8mm not available on this CPU");
            return;
        }
        eprintln!("i8mm detected — testing i8mm kernel directly");

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

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
                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }
                src.extend_from_slice(&make_q4k_block(d_val, dmin_val, scales, qs));
            }
        }

        let packed = pack_q4k(&src, rows, cols);

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

        // i8mm 커널 직접 호출
        let mut output_i8mm = vec![0.0f32; seq_len * rows];
        unsafe {
            gemm_q4k_packed_i8mm(
                &packed,
                &input_qs_flat,
                &input_d_flat,
                &input_bsums_flat,
                &mut output_i8mm,
                rows,
                cols,
                seq_len,
            );
        }

        // scalar 레퍼런스
        let mut output_scalar = vec![0.0f32; seq_len * rows];
        gemm_q4k_packed_scalar(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_scalar,
            rows,
            cols,
            seq_len,
        );

        // 비교
        for s in 0..seq_len {
            for row in 0..rows {
                let i8mm_val = output_i8mm[s * rows + row];
                let scalar_val = output_scalar[s * rows + row];
                let diff = (i8mm_val - scalar_val).abs();
                let rel = if scalar_val.abs() > 1e-6 {
                    diff / scalar_val.abs()
                } else {
                    diff
                };
                assert!(
                    diff < 0.01 || rel < 0.001,
                    "i8mm vs scalar MISMATCH: s={s} row={row}: \
                     i8mm={i8mm_val:.6}, scalar={scalar_val:.6}, diff={diff:.6}"
                );
            }
        }

        // 각 row의 결과가 다른지 확인
        for s in 0..seq_len {
            let mut all_same = true;
            let first = output_i8mm[s * rows];
            for row in 1..rows {
                if (output_i8mm[s * rows + row] - first).abs() > 0.01 {
                    all_same = false;
                    break;
                }
            }
            assert!(!all_same, "i8mm s={s}: all rows same — interleaving broken");
        }

        eprintln!("test_gemm_i8mm_direct_unique_rows: PASSED");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_direct_unique_rows() {
        use std::f32::consts::PI;

        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            eprintln!("SKIP: dotprod not available on this CPU");
            return;
        }
        eprintln!("dotprod detected — testing neon kernel directly");

        let rows = 16;
        let cols = 2;
        let seq_len = 2;

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
                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }
                src.extend_from_slice(&make_q4k_block(d_val, dmin_val, scales, qs));
            }
        }

        let packed = pack_q4k(&src, rows, cols);

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

        // neon 커널 직접 호출
        let mut output_neon = vec![0.0f32; seq_len * rows];
        unsafe {
            gemm_q4k_packed_neon(
                &packed,
                &input_qs_flat,
                &input_d_flat,
                &input_bsums_flat,
                &mut output_neon,
                rows,
                cols,
                seq_len,
            );
        }

        // scalar 레퍼런스
        let mut output_scalar = vec![0.0f32; seq_len * rows];
        gemm_q4k_packed_scalar(
            &packed,
            &input_qs_flat,
            &input_d_flat,
            &input_bsums_flat,
            &mut output_scalar,
            rows,
            cols,
            seq_len,
        );

        for s in 0..seq_len {
            for row in 0..rows {
                let neon_val = output_neon[s * rows + row];
                let scalar_val = output_scalar[s * rows + row];
                let diff = (neon_val - scalar_val).abs();
                let rel = if scalar_val.abs() > 1e-6 {
                    diff / scalar_val.abs()
                } else {
                    diff
                };
                assert!(
                    diff < 0.01 || rel < 0.001,
                    "neon vs scalar MISMATCH: s={s} row={row}: \
                     neon={neon_val:.6}, scalar={scalar_val:.6}, diff={diff:.6}"
                );
            }
        }

        eprintln!("test_gemm_neon_direct_unique_rows: PASSED");
    }

    // ─── 테스트 5: seq_len > 1 독립성 ───────────────────────────

    #[test]
    fn test_gemm_seq_len_independence() {
        // 2개 토큰이 서로 독립적으로 계산되는지 확인
        // weight qs=0x88 (unsigned 8), d=1.0, dmin=0.0
        // sc_raw[0]=5, mn_raw[0]=0
        // token 0: bsums[0]=+10 → sumi = 5 * 8*10 (unsigned nibble=8, x_qs=0) + ...
        //   wait: input_qs = 0 → dot = 0
        //   sumi = 5 * 0 = 0
        //   summ = 0 * bsum = 0
        //   output = x_d * (d * 0 - dmin * 0) = 0... 이건 0이네.
        //
        // 기존 테스트는 (8*sc - mn) * bsum 보정에 의존했는데,
        // 새 수식에서는 sc_raw/mn_raw가 0이면 전부 0이됨.
        //
        // 대신: 모든 원소가 같은 unsigned nibble일 때,
        // dot(unsigned=8, x_qs=1, 32개) = 256
        // sumi = 5 * 256 = 1280
        // out = 1.0 * (1.0 * 1280) = 1280
        //
        // 하지만 bsums와의 관계도 봐야 해.
        // x_qs=1, 32개 → bsums[0]=32
        // summ = mn_raw[0] * 32 = 0
        //
        // 부호 반전 테스트: x_qs=-1, bsums=-32
        // dot = 8 * (-1) * 32 = -256
        // sumi = 5 * (-256) = -1280
        // out = -1280

        let rows = 8;
        let cols = 1;
        let seq_len = 2;

        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 5;
        let block = make_q4k_block(1.0, 0.0, scales_raw, [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(rows);
        let packed = pack_q4k(&src, rows, cols);

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
        gemm_q4k_packed(
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
        // sumi token 1: sc_raw=5, dot=8*(-32)=-256, sumi=5*(-256)=-1280
        let expected0 = 1280.0f32;
        let _expected1 = -1280.0f32;

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

    // ─── 테스트: seq_len=3 (2-token pair + 1-token fallback) ────

    #[test]
    fn test_gemm_q4k_packed_seq3() {
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

                let mut qs = [0u8; 128];
                for i in 0..128 {
                    let lo = ((seed as u16 + i as u16 * 5) % 16) as u8;
                    let hi = ((seed as u16 + i as u16 * 3 + 7) % 16) as u8;
                    qs[i] = lo | (hi << 4);
                }

                src.extend_from_slice(&make_q4k_block(d_val, dmin_val, scales, qs));
            }
        }

        let packed = pack_q4k(&src, rows, cols);

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
        gemm_q4k_packed(
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

            gemm_q4k_packed(
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

        eprintln!("test_gemm_q4k_packed_seq3: max_rel_err = {max_rel_err:.8}");
    }
}
