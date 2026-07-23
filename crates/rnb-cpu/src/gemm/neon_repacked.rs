use super::Q8KBlock;
use crate::gemm::repack::{
    REPACKED_BLOCK_BYTES, RPK_DMIN_OFF, RPK_D_OFF, RPK_MN_OFF, RPK_QS_OFF, RPK_SC_OFF,
    TPK_DMIN_OFF, TPK_D_OFF, TPK_MN_OFF, TPK_QS_OFF, TPK_SC_OFF, TWIN_Q4K_BLOCK_BYTES,
};
use rayon::prelude::*;
use std::arch::aarch64::*;

/// Repacked Q4_K GEMM: 8-row groups, block-by-block weight unpack, seq_len reuse.
///
/// For prefill (seq_len > 1): unpack each block's 8 rows once, dot with all tokens.
/// Weight data is read once per block and reused seq_len times.
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q4k_8rows(
    repacked: &[u8],  // this group's repacked data
    q8k: &[Q8KBlock], // [seq_len * n_blocks]
    n_blocks: usize,
    seq_len: usize,
    out: *mut f32, // column-major: out[s * total_rows + row]
    group_row_start: usize,
    total_rows: usize,
) {
    let mask_low = vdupq_n_u8(0x0F);
    let rp = repacked.as_ptr();
    let vzero = vdupq_n_s32(0);

    // seq_len × 8 accumulators
    let mut acc = vec![0.0f32; seq_len * 8];

    for bi in 0..n_blocks {
        let boff = bi * REPACKED_BLOCK_BYTES;

        // --- Phase 1: Unpack 8 rows' weights for this block (2048B on stack) ---
        // unpacked[r][256] = nibble-split i8 values
        let mut unpacked = [[0i8; 256]; 8];
        // pre-extracted scales (already in repacked format)
        let mut sc = [[0u8; 8]; 8];
        let mut mn = [[0u8; 8]; 8];
        let mut d = [0.0f32; 8];
        let mut dmin = [0.0f32; 8];

        for r in 0..8 {
            let qs = rp.add(boff + RPK_QS_OFF + r * 128);
            let out_ptr = unpacked[r].as_mut_ptr();

            // NEON vectorized nibble unpack: 4 groups of 64 elements
            for g in 0..4 {
                let q_off = g * 32;
                let o_off = g * 64;
                let qlo = vld1q_u8(qs.add(q_off));
                let qhi = vld1q_u8(qs.add(q_off + 16));
                vst1q_s8(
                    out_ptr.add(o_off),
                    vreinterpretq_s8_u8(vandq_u8(qlo, mask_low)),
                );
                vst1q_s8(
                    out_ptr.add(o_off + 16),
                    vreinterpretq_s8_u8(vandq_u8(qhi, mask_low)),
                );
                vst1q_s8(
                    out_ptr.add(o_off + 32),
                    vreinterpretq_s8_u8(vshrq_n_u8(qlo, 4)),
                );
                vst1q_s8(
                    out_ptr.add(o_off + 48),
                    vreinterpretq_s8_u8(vshrq_n_u8(qhi, 4)),
                );
            }

            // Copy pre-extracted sc/mn
            let sc_ptr = rp.add(boff + RPK_SC_OFF + r * 8);
            let mn_ptr = rp.add(boff + RPK_MN_OFF + r * 8);
            for j in 0..8 {
                sc[r][j] = *sc_ptr.add(j);
                mn[r][j] = *mn_ptr.add(j);
            }
            d[r] =
                f32::from_bits((rp.add(boff + RPK_D_OFF + r * 4) as *const u32).read_unaligned());
            dmin[r] = f32::from_bits(
                (rp.add(boff + RPK_DMIN_OFF + r * 4) as *const u32).read_unaligned(),
            );
        }

        // --- Phase 2: Dot product with all tokens using unpacked weights ---
        for s in 0..seq_len {
            let q8b = q8k.get_unchecked(s * n_blocks + bi);

            for r in 0..8 {
                let w = unpacked[r].as_ptr();
                let mut acc_v = vzero;
                let mut summ = 0i32;

                // Group 0
                {
                    let p0 = vdotq_s32(
                        vdotq_s32(vzero, vld1q_s8(w), vld1q_s8(q8b.qs.as_ptr())),
                        vld1q_s8(w.add(16)),
                        vld1q_s8(q8b.qs.as_ptr().add(16)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(32)),
                            vld1q_s8(q8b.qs.as_ptr().add(32)),
                        ),
                        vld1q_s8(w.add(48)),
                        vld1q_s8(q8b.qs.as_ptr().add(48)),
                    );
                    acc_v = vmlaq_n_s32(acc_v, p0, sc[r][0] as i32);
                    acc_v = vmlaq_n_s32(acc_v, p1, sc[r][1] as i32);
                    summ += mn[r][0] as i32 * q8b.bsum32(0) as i32
                        + mn[r][1] as i32 * q8b.bsum32(1) as i32;
                }
                // Group 1
                {
                    let p0 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(64)),
                            vld1q_s8(q8b.qs.as_ptr().add(64)),
                        ),
                        vld1q_s8(w.add(80)),
                        vld1q_s8(q8b.qs.as_ptr().add(80)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(96)),
                            vld1q_s8(q8b.qs.as_ptr().add(96)),
                        ),
                        vld1q_s8(w.add(112)),
                        vld1q_s8(q8b.qs.as_ptr().add(112)),
                    );
                    acc_v = vmlaq_n_s32(acc_v, p0, sc[r][2] as i32);
                    acc_v = vmlaq_n_s32(acc_v, p1, sc[r][3] as i32);
                    summ += mn[r][2] as i32 * q8b.bsum32(2) as i32
                        + mn[r][3] as i32 * q8b.bsum32(3) as i32;
                }
                // Group 2
                {
                    let p0 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(128)),
                            vld1q_s8(q8b.qs.as_ptr().add(128)),
                        ),
                        vld1q_s8(w.add(144)),
                        vld1q_s8(q8b.qs.as_ptr().add(144)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(160)),
                            vld1q_s8(q8b.qs.as_ptr().add(160)),
                        ),
                        vld1q_s8(w.add(176)),
                        vld1q_s8(q8b.qs.as_ptr().add(176)),
                    );
                    acc_v = vmlaq_n_s32(acc_v, p0, sc[r][4] as i32);
                    acc_v = vmlaq_n_s32(acc_v, p1, sc[r][5] as i32);
                    summ += mn[r][4] as i32 * q8b.bsum32(4) as i32
                        + mn[r][5] as i32 * q8b.bsum32(5) as i32;
                }
                // Group 3
                {
                    let p0 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(192)),
                            vld1q_s8(q8b.qs.as_ptr().add(192)),
                        ),
                        vld1q_s8(w.add(208)),
                        vld1q_s8(q8b.qs.as_ptr().add(208)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w.add(224)),
                            vld1q_s8(q8b.qs.as_ptr().add(224)),
                        ),
                        vld1q_s8(w.add(240)),
                        vld1q_s8(q8b.qs.as_ptr().add(240)),
                    );
                    acc_v = vmlaq_n_s32(acc_v, p0, sc[r][6] as i32);
                    acc_v = vmlaq_n_s32(acc_v, p1, sc[r][7] as i32);
                    summ += mn[r][6] as i32 * q8b.bsum32(6) as i32
                        + mn[r][7] as i32 * q8b.bsum32(7) as i32;
                }

                let sumi = vaddvq_s32(acc_v);
                acc[s * 8 + r] += q8b.d * (d[r] * sumi as f32 - dmin[r] * summ as f32);
            }
        }
    }

    // Write output (column-major)
    for s in 0..seq_len {
        for r in 0..8 {
            *out.add(s * total_rows + group_row_start + r) = acc[s * 8 + r];
        }
    }
}

/// Repacked Q4_K GEMM: 8-row groups in parallel for prefill.
/// Unpack weight once per block, reuse for all seq_len tokens.
pub fn gemm_q4k_repacked(
    repacked: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    original_bytes: &[u8],
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let groups = rows / 8;
    let remainder = rows % 8;
    let group_bytes = n_blocks * REPACKED_BLOCK_BYTES;

    let out_addr = output.as_mut_ptr() as usize;

    if groups > 0 {
        (0..groups).into_par_iter().for_each(|g| {
            let out_ptr = out_addr as *mut f32;
            let group_data = &repacked[g * group_bytes..(g + 1) * group_bytes];
            unsafe {
                gemm_q4k_8rows(group_data, q8k, n_blocks, seq_len, out_ptr, g * 8, rows);
            }
        });
    }

    if remainder > 0 {
        let start = groups * 8;
        for row in start..start + remainder {
            let rb = &original_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let out_ptr = out_addr as *mut f32;
            unsafe {
                super::neon_dot::gemm_q4_k_row(rb, q8k, n_blocks, seq_len, out_ptr, row, rows);
            }
        }
    }
}

/// Repacked Q4_K GEMV for decode (seq_len == 1).
/// Uses original non-repacked kernel — repack has no decode advantage.
/// This function is kept for API compatibility but just delegates to the original.
pub fn gemv_q4k_repacked(
    _repacked: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    original_bytes: &[u8],
    bytes_per_row: usize,
) {
    // Decode: use original kernel (better cache pattern for single-row access)
    super::neon_dot::gemv_q4_k_int8(
        original_bytes,
        q8k,
        output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
    );
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4k_compact_row(
    repacked: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
    group_base: usize,
    r: usize,
) -> f32 {
    let rp = repacked.as_ptr();
    let mask_low = vdupq_n_u8(0x0f);
    let vzero = vdupq_n_s32(0);
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = group_base + bi * REPACKED_BLOCK_BYTES;
        let q8b = q8k.get_unchecked(bi);
        let qs = rp.add(boff + RPK_QS_OFF + r * 128);
        let sc = rp.add(boff + RPK_SC_OFF + r * 8);
        let mn = rp.add(boff + RPK_MN_OFF + r * 8);
        let d = f32::from_bits((rp.add(boff + RPK_D_OFF + r * 4) as *const u32).read_unaligned());
        let dmin =
            f32::from_bits((rp.add(boff + RPK_DMIN_OFF + r * 4) as *const u32).read_unaligned());

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

            let x_lo0 = vld1q_s8(q8b.qs.as_ptr().add(x_off));
            let x_lo1 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 16));
            let x_hi0 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 32));
            let x_hi1 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 48));

            let mut p0 = vzero;
            p0 = vdotq_s32(p0, w_lo0, x_lo0);
            p0 = vdotq_s32(p0, w_lo1, x_lo1);
            let mut p1 = vzero;
            p1 = vdotq_s32(p1, w_hi0, x_hi0);
            p1 = vdotq_s32(p1, w_hi1, x_hi1);

            acc_v = vmlaq_n_s32(acc_v, p0, *sc.add(sb0) as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, *sc.add(sb1) as i32);
            summ += *mn.add(sb0) as i32 * q8b.bsum32(sb0) as i32
                + *mn.add(sb1) as i32 * q8b.bsum32(sb1) as i32;
        }

        let sumi = vaddvq_s32(acc_v);
        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }

    acc
}

pub fn gemv_q4k_compact(
    repacked: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let n_blocks = cols / 256;
    let groups = rows.div_ceil(8);
    let group_bytes = n_blocks * REPACKED_BLOCK_BYTES;

    output[..rows]
        .par_chunks_mut(512)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * 512;
            for (local, out) in out_chunk.iter_mut().enumerate() {
                let row = row_start + local;
                let g = row / 8;
                let r = row % 8;
                if g >= groups {
                    continue;
                }
                *out += unsafe { dot_q4k_compact_row(repacked, q8k, n_blocks, g * group_bytes, r) };
            }
        });
}

#[inline(always)]
unsafe fn pair_dup_i32x4(a: i32, b: i32) -> int32x4_t {
    let mut v = vdupq_n_s32(a);
    v = vsetq_lane_s32(a, v, 1);
    v = vsetq_lane_s32(b, v, 2);
    v = vsetq_lane_s32(b, v, 3);
    v
}

#[inline(always)]
unsafe fn pair_dup_f32x4(a: f32, b: f32) -> float32x4_t {
    let mut v = vdupq_n_f32(a);
    v = vsetq_lane_f32(a, v, 1);
    v = vsetq_lane_f32(b, v, 2);
    v = vsetq_lane_f32(b, v, 3);
    v
}

#[target_feature(enable = "neon,i8mm")]
unsafe fn dot_q4k_twin_pair(repacked: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> [f32; 2] {
    let rp = repacked.as_ptr();
    let rp_i8 = rp as *const i8;
    let mut vfsum = vdupq_n_f32(0.0);

    for bi in 0..n_blocks {
        let boff = bi * TWIN_Q4K_BLOCK_BYTES;
        let q8b = q8k.get_unchecked(bi);
        let q8d = q8b.d;
        let sc0 = std::slice::from_raw_parts(rp.add(boff + TPK_SC_OFF), 8);
        let sc1 = std::slice::from_raw_parts(rp.add(boff + TPK_SC_OFF + 8), 8);
        let mn0 = std::slice::from_raw_parts(rp.add(boff + TPK_MN_OFF), 8);
        let mn1 = std::slice::from_raw_parts(rp.add(boff + TPK_MN_OFF + 8), 8);
        let d0 = f32::from_bits((rp.add(boff + TPK_D_OFF) as *const u32).read_unaligned());
        let d1 = f32::from_bits((rp.add(boff + TPK_D_OFF + 4) as *const u32).read_unaligned());
        let dmin0 = f32::from_bits((rp.add(boff + TPK_DMIN_OFF) as *const u32).read_unaligned());
        let dmin1 =
            f32::from_bits((rp.add(boff + TPK_DMIN_OFF + 4) as *const u32).read_unaligned());

        let mut visum = vdupq_n_s32(0);
        for group in 0..4 {
            let x_off = group * 64;
            let q_off = boff + TPK_QS_OFF + group * 128;
            let vy0 = vld1q_s8(q8b.qs.as_ptr().add(x_off));
            let vy1 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 16));
            let vy2 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 32));
            let vy3 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 48));
            let vy_l0 = vcombine_s8(vget_low_s8(vy0), vget_low_s8(vy0));
            let vy_h0 = vcombine_s8(vget_high_s8(vy0), vget_high_s8(vy0));
            let vy_l1 = vcombine_s8(vget_low_s8(vy1), vget_low_s8(vy1));
            let vy_h1 = vcombine_s8(vget_high_s8(vy1), vget_high_s8(vy1));
            let vy_l2 = vcombine_s8(vget_low_s8(vy2), vget_low_s8(vy2));
            let vy_h2 = vcombine_s8(vget_high_s8(vy2), vget_high_s8(vy2));
            let vy_l3 = vcombine_s8(vget_low_s8(vy3), vget_low_s8(vy3));
            let vy_h3 = vcombine_s8(vget_high_s8(vy3), vget_high_s8(vy3));
            let vy_l = [vy_l0, vy_l1, vy_l2, vy_l3];
            let vy_h = [vy_h0, vy_h1, vy_h2, vy_h3];

            let lo0 = q_off;
            let hi0 = q_off + 32;
            let lo1 = q_off + 64;
            let hi1 = q_off + 96;

            let mut vr0 = vdupq_n_s32(0);
            vr0 = vmmlaq_s32(vr0, vld1q_s8(rp_i8.add(lo0)), vy_l[0]);
            vr0 = vmmlaq_s32(vr0, vld1q_s8(rp_i8.add(lo0 + 16)), vy_h[0]);
            vr0 = vmmlaq_s32(vr0, vld1q_s8(rp_i8.add(lo1)), vy_l[1]);
            vr0 = vmmlaq_s32(vr0, vld1q_s8(rp_i8.add(lo1 + 16)), vy_h[1]);
            visum = vmlaq_s32(
                visum,
                vr0,
                pair_dup_i32x4(sc0[group * 2] as i32, sc1[group * 2] as i32),
            );

            let mut vr1 = vdupq_n_s32(0);
            vr1 = vmmlaq_s32(vr1, vld1q_s8(rp_i8.add(hi0)), vy_l[2]);
            vr1 = vmmlaq_s32(vr1, vld1q_s8(rp_i8.add(hi0 + 16)), vy_h[2]);
            vr1 = vmmlaq_s32(vr1, vld1q_s8(rp_i8.add(hi1)), vy_l[3]);
            vr1 = vmmlaq_s32(vr1, vld1q_s8(rp_i8.add(hi1 + 16)), vy_h[3]);
            visum = vmlaq_s32(
                visum,
                vr1,
                pair_dup_i32x4(sc0[group * 2 + 1] as i32, sc1[group * 2 + 1] as i32),
            );
        }

        let bias0 = mn0[0] as i32 * q8b.bsum32(0) as i32
            + mn0[1] as i32 * q8b.bsum32(1) as i32
            + mn0[2] as i32 * q8b.bsum32(2) as i32
            + mn0[3] as i32 * q8b.bsum32(3) as i32
            + mn0[4] as i32 * q8b.bsum32(4) as i32
            + mn0[5] as i32 * q8b.bsum32(5) as i32
            + mn0[6] as i32 * q8b.bsum32(6) as i32
            + mn0[7] as i32 * q8b.bsum32(7) as i32;
        let bias1 = mn1[0] as i32 * q8b.bsum32(0) as i32
            + mn1[1] as i32 * q8b.bsum32(1) as i32
            + mn1[2] as i32 * q8b.bsum32(2) as i32
            + mn1[3] as i32 * q8b.bsum32(3) as i32
            + mn1[4] as i32 * q8b.bsum32(4) as i32
            + mn1[5] as i32 * q8b.bsum32(5) as i32
            + mn1[6] as i32 * q8b.bsum32(6) as i32
            + mn1[7] as i32 * q8b.bsum32(7) as i32;

        vfsum = vmlsq_f32(
            vfsum,
            vcvtq_f32_s32(pair_dup_i32x4(bias0, bias1)),
            pair_dup_f32x4(dmin0 * q8d, dmin1 * q8d),
        );
        vfsum = vmlaq_f32(
            vfsum,
            vcvtq_f32_s32(visum),
            pair_dup_f32x4(d0 * q8d, d1 * q8d),
        );
    }

    [vgetq_lane_f32(vfsum, 0), vgetq_lane_f32(vfsum, 2)]
}

pub fn gemv_q4k_twin_repacked(
    repacked: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    original_bytes: &[u8],
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let pairs = rows / 2;

    for p in 0..pairs {
        let pair_data = &repacked
            [p * n_blocks * TWIN_Q4K_BLOCK_BYTES..(p + 1) * n_blocks * TWIN_Q4K_BLOCK_BYTES];
        let pair = unsafe { dot_q4k_twin_pair(pair_data, q8k, n_blocks) };
        output[p * 2] = pair[0];
        output[p * 2 + 1] = pair[1];
    }

    if rows % 2 != 0 {
        let row = rows - 1;
        let rb = &original_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
        output[row] = unsafe { super::neon_dot::dot_q4_k_q8k_neon(rb, q8k, n_blocks) };
    }
}
