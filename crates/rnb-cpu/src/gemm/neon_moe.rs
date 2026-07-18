//! MoE section NEON sdot kernels for MoE/ATTN/GDN decode paths.
//!
//! Each kernel takes IntScale weight blocks + Q8K activation blocks and returns
//! i32 integer accumulate (caller multiplies by row_multiplier × activation d
//! for the final f32 result). This keeps the hot loop in integer arithmetic
//! and defers float conversion to row boundary.
//!
//! # Q4_K GGUF block layout recap
//!
//! Each 256-elem super-block has 8 sub-blocks of 32 elems. GGUF packs the
//! nibbles as **4 groups of 32 bytes** (`qs[128]`), where group `g` covers
//! 64 output elements:
//!   - `qs[g*32 + l] & 0x0F` (low nibble)  → elem `g*64 + l`       (sub-block `g*2`)
//!   - `qs[g*32 + l] >> 4`  (high nibble) → elem `g*64 + 32 + l`  (sub-block `g*2 + 1`)
//!
//! The sub-block 6-bit scales/mins live in `sub_scales[12]` and are unpacked
//! via `get_scale_min_k4(j, &sub_scales)` for `j ∈ 0..8`.
//!
//! The `Q8KBlock.bsums[j]` field already contains the per-sub-block i16 sum of
//! the activation quants, so the integer accumulate splits cleanly into:
//!
//! ```text
//!   block_acc =   scale_int * Σⱼ sc[j] * <qs[j], x[j]>
//!               - min_int   * Σⱼ  m[j] * bsums[j]
//! ```
//!
//! The caller folds in `row_mul_weight * x.d` at row boundary to recover f32.

use crate::gemm::Q8KBlock;
use crate::quantize::moe_blocks::{
    GUPairQ4K, GUPairQ4KScaleMin, GUPairQ4KUnpackedScales, Q5KIntScale, Q80IntScale,
    SharedGUQ8KUnit,
};
use crate::quantize::quant::get_scale_min_k4;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(any(target_arch = "aarch64", test))]
#[inline(always)]
fn unpack_k4_scales(q: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let sc = [
        q[0] & 63,
        q[1] & 63,
        q[2] & 63,
        q[3] & 63,
        (q[8] & 0x0F) | ((q[0] >> 6) << 4),
        (q[9] & 0x0F) | ((q[1] >> 6) << 4),
        (q[10] & 0x0F) | ((q[2] >> 6) << 4),
        (q[11] & 0x0F) | ((q[3] >> 6) << 4),
    ];
    let m = [
        q[4] & 63,
        q[5] & 63,
        q[6] & 63,
        q[7] & 63,
        (q[8] >> 4) | ((q[4] >> 6) << 4),
        (q[9] >> 4) | ((q[5] >> 6) << 4),
        (q[10] >> 4) | ((q[6] >> 6) << 4),
        (q[11] >> 4) | ((q[7] >> 6) << 4),
    ];
    (sc, m)
}

/// Scalar oracle for [`sdot_q4k_gu_neon`]. Computes block-interleaved gate/up
/// dot, honouring Q4_K 6-bit packed sub-block scales.
///
/// Returns `(gate_i64, up_i64)`. The caller multiplies each by
/// `row_mul_weight * x.d` to recover the f32 dot product (where
/// `row_mul_weight` is the row-level f32 scale and `x.d` is the Q8K activation
/// scalar).
///
/// Mirrors `dequantize_q4k_intscale` nibble indexing exactly so the dequant
/// and dot-product paths agree by construction.
///
/// Returns i64 (not i32) because of adversarial overflow bounds: per Q4_K
/// block, `scaled_dot ≤ 3.07e7` and `scale_int ∈ ±127`, so
/// `scale_int × scaled_dot` can reach ±3.9e9 — larger than i32 range (±2.14e9).
/// Widening the running accumulator to i64 keeps the kernel safe across
/// pathological weight/activation combinations. `scaled_dot` itself stays i32
/// (fits with margin), and the i32→i64 promotion happens right before the
/// `scale_int × scaled_dot` multiply.
pub fn sdot_q4k_gu_scalar(row: &[GUPairQ4K], h_q8k: &[Q8KBlock]) -> (i64, i64) {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;

    for (pair, x) in row.iter().zip(h_q8k.iter()) {
        // Unpack 8 sub-block (sc, m) pairs from the 6-bit packed sub_scales.
        let mut sc_g = [0u8; 8];
        let mut m_g = [0u8; 8];
        let mut sc_u = [0u8; 8];
        let mut m_u = [0u8; 8];
        for j in 0..8 {
            let (s, mm) = get_scale_min_k4(j, &pair.gate.sub_scales);
            sc_g[j] = s;
            m_g[j] = mm;
            let (s, mm) = get_scale_min_k4(j, &pair.up.sub_scales);
            sc_u[j] = s;
            m_u[j] = mm;
        }

        // Accumulate Σⱼ sc[j] * <qs[j], x[j]> and Σⱼ m[j] * bsums[j] per pair.
        let mut g_scaled_dot: i32 = 0;
        let mut g_scaled_bsum: i32 = 0;
        let mut u_scaled_dot: i32 = 0;
        let mut u_scaled_bsum: i32 = 0;

        // 4 groups × 2 sub-blocks (low/high nibble of each byte in the 32-byte group).
        for g in 0..4 {
            let q_off = g * 32;
            let x_off = g * 64;

            let mut sub_dot_g_lo: i32 = 0;
            let mut sub_dot_g_hi: i32 = 0;
            let mut sub_dot_u_lo: i32 = 0;
            let mut sub_dot_u_hi: i32 = 0;
            for l in 0..32 {
                let qg = pair.gate.qs[q_off + l];
                let qu = pair.up.qs[q_off + l];
                let lo_g = (qg & 0x0F) as i32;
                let hi_g = ((qg >> 4) & 0x0F) as i32;
                let lo_u = (qu & 0x0F) as i32;
                let hi_u = ((qu >> 4) & 0x0F) as i32;
                let xlo = x.qs[x_off + l] as i32;
                let xhi = x.qs[x_off + 32 + l] as i32;
                sub_dot_g_lo += lo_g * xlo;
                sub_dot_g_hi += hi_g * xhi;
                sub_dot_u_lo += lo_u * xlo;
                sub_dot_u_hi += hi_u * xhi;
            }

            let s_lo = g * 2;
            let s_hi = g * 2 + 1;
            g_scaled_dot += (sc_g[s_lo] as i32) * sub_dot_g_lo + (sc_g[s_hi] as i32) * sub_dot_g_hi;
            u_scaled_dot += (sc_u[s_lo] as i32) * sub_dot_u_lo + (sc_u[s_hi] as i32) * sub_dot_u_hi;
            g_scaled_bsum += (m_g[s_lo] as i32) * (x.bsum32(s_lo) as i32)
                + (m_g[s_hi] as i32) * (x.bsum32(s_hi) as i32);
            u_scaled_bsum += (m_u[s_lo] as i32) * (x.bsum32(s_lo) as i32)
                + (m_u[s_hi] as i32) * (x.bsum32(s_hi) as i32);
        }

        gate_acc += (pair.gate.scale_int as i64) * (g_scaled_dot as i64)
            - (pair.gate.min_int as i64) * (g_scaled_bsum as i64);
        up_acc += (pair.up.scale_int as i64) * (u_scaled_dot as i64)
            - (pair.up.min_int as i64) * (u_scaled_bsum as i64);
    }

    (gate_acc, up_acc)
}

/// NEON variant of [`sdot_q4k_gu_scalar`]. Uses `vdotq_s32` for the per-sub-block
/// `<qs, x>` inner products; the scale/min weighting stays in scalar i32 (its
/// cost is negligible vs the sdot pipeline).
///
/// Returns `(i64, i64)` to match the scalar oracle — see its doc comment for
/// the adversarial overflow bound that forces i64.
///
/// # Safety
///
/// - Requires the ARMv8.2 `dotprod` extension (`vdotq_s32`). The caller must
///   ensure the target CPU has `neon` + `dotprod` available; the
///   `#[target_feature(enable = "neon,dotprod")]` attribute below gates the
///   intrinsic but does not probe the CPU at runtime.
/// - All `vld1q_u8` / `vld1q_s8` loads are in-bounds by construction: the
///   weight fields `GUPairQ4K.{gate,up}.qs: [u8; 128]` and the activation
///   `Q8KBlock.qs: [i8; 256]` are fixed-size arrays, and the outer loop
///   `gidx ∈ 0..4` combined with the per-group offsets (`q_off = gidx*32`,
///   `x_off = gidx*64`) means the largest byte offset touched is
///   `qs.add(q_off + 16) = qs.add(112)` (plus 16-byte load) = byte 127 and
///   `x.qs.add(x_off + 48) = x.qs.add(240)` (plus 16 bytes) = byte 255.
///   Both stay within their fixed-size arrays, so every 16-byte load is valid.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q4k_gu_block_neon(pair: &GUPairQ4K, x: &Q8KBlock) -> (i64, i64) {
    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;
    let mask_low = vdupq_n_u8(0x0F);

    let (sc_g, m_g) = unpack_k4_scales(&pair.gate.sub_scales);
    let (sc_u, m_u) = unpack_k4_scales(&pair.up.sub_scales);

    let qs_g = pair.gate.qs.as_ptr();
    let qs_u = pair.up.qs.as_ptr();
    let qs_x = x.qs.as_ptr();

    let mut g_scaled_dot: i32 = 0;
    let mut u_scaled_dot: i32 = 0;
    let mut g_scaled_bsum: i32 = 0;
    let mut u_scaled_bsum: i32 = 0;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;

        let g_bytes_lo = vld1q_u8(qs_g.add(q_off));
        let g_bytes_hi = vld1q_u8(qs_g.add(q_off + 16));
        let u_bytes_lo = vld1q_u8(qs_u.add(q_off));
        let u_bytes_hi = vld1q_u8(qs_u.add(q_off + 16));

        let w_g_lo_0 = vreinterpretq_s8_u8(vandq_u8(g_bytes_lo, mask_low));
        let w_g_lo_1 = vreinterpretq_s8_u8(vandq_u8(g_bytes_hi, mask_low));
        let w_u_lo_0 = vreinterpretq_s8_u8(vandq_u8(u_bytes_lo, mask_low));
        let w_u_lo_1 = vreinterpretq_s8_u8(vandq_u8(u_bytes_hi, mask_low));

        let w_g_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_lo, 4));
        let w_g_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_hi, 4));
        let w_u_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_lo, 4));
        let w_u_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_hi, 4));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut g_lo = vdupq_n_s32(0);
        g_lo = vdotq_s32(g_lo, w_g_lo_0, x_lo_0);
        g_lo = vdotq_s32(g_lo, w_g_lo_1, x_lo_1);
        let mut g_hi = vdupq_n_s32(0);
        g_hi = vdotq_s32(g_hi, w_g_hi_0, x_hi_0);
        g_hi = vdotq_s32(g_hi, w_g_hi_1, x_hi_1);

        let mut u_lo = vdupq_n_s32(0);
        u_lo = vdotq_s32(u_lo, w_u_lo_0, x_lo_0);
        u_lo = vdotq_s32(u_lo, w_u_lo_1, x_lo_1);
        let mut u_hi = vdupq_n_s32(0);
        u_hi = vdotq_s32(u_hi, w_u_hi_0, x_hi_0);
        u_hi = vdotq_s32(u_hi, w_u_hi_1, x_hi_1);

        let dot_g_lo = vaddvq_s32(g_lo);
        let dot_g_hi = vaddvq_s32(g_hi);
        let dot_u_lo = vaddvq_s32(u_lo);
        let dot_u_hi = vaddvq_s32(u_hi);

        let s_lo = gidx * 2;
        let s_hi = gidx * 2 + 1;
        g_scaled_dot += (sc_g[s_lo] as i32) * dot_g_lo + (sc_g[s_hi] as i32) * dot_g_hi;
        u_scaled_dot += (sc_u[s_lo] as i32) * dot_u_lo + (sc_u[s_hi] as i32) * dot_u_hi;

        let bsum_lo = x.bsum32(s_lo) as i32;
        let bsum_hi = x.bsum32(s_hi) as i32;
        g_scaled_bsum += (m_g[s_lo] as i32) * bsum_lo + (m_g[s_hi] as i32) * bsum_hi;
        u_scaled_bsum += (m_u[s_lo] as i32) * bsum_lo + (m_u[s_hi] as i32) * bsum_hi;
    }

    gate_acc += (pair.gate.scale_int as i64) * (g_scaled_dot as i64)
        - (pair.gate.min_int as i64) * (g_scaled_bsum as i64);
    up_acc += (pair.up.scale_int as i64) * (u_scaled_dot as i64)
        - (pair.up.min_int as i64) * (u_scaled_bsum as i64);

    (gate_acc, up_acc)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q4k_gu_block_unpacked_scales_neon(
    unpacked: &GUPairQ4KUnpackedScales,
    x: &Q8KBlock,
) -> (i64, i64) {
    let pair = &unpacked.pair;
    sdot_q4k_gu_block_scale_min_neon(pair, unpacked.as_scale_min(), x)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q4k_gu_block_scale_min_neon(
    pair: &GUPairQ4K,
    scale: &GUPairQ4KScaleMin,
    x: &Q8KBlock,
) -> (i64, i64) {
    let mask_low = vdupq_n_u8(0x0F);

    let qs_g = pair.gate.qs.as_ptr();
    let qs_u = pair.up.qs.as_ptr();
    let qs_x = x.qs.as_ptr();

    let mut g_scaled_dot: i32 = 0;
    let mut u_scaled_dot: i32 = 0;
    let mut g_scaled_bsum: i32 = 0;
    let mut u_scaled_bsum: i32 = 0;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;

        let g_bytes_lo = vld1q_u8(qs_g.add(q_off));
        let g_bytes_hi = vld1q_u8(qs_g.add(q_off + 16));
        let u_bytes_lo = vld1q_u8(qs_u.add(q_off));
        let u_bytes_hi = vld1q_u8(qs_u.add(q_off + 16));

        let w_g_lo_0 = vreinterpretq_s8_u8(vandq_u8(g_bytes_lo, mask_low));
        let w_g_lo_1 = vreinterpretq_s8_u8(vandq_u8(g_bytes_hi, mask_low));
        let w_u_lo_0 = vreinterpretq_s8_u8(vandq_u8(u_bytes_lo, mask_low));
        let w_u_lo_1 = vreinterpretq_s8_u8(vandq_u8(u_bytes_hi, mask_low));

        let w_g_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_lo, 4));
        let w_g_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_hi, 4));
        let w_u_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_lo, 4));
        let w_u_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_hi, 4));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut g_lo = vdupq_n_s32(0);
        g_lo = vdotq_s32(g_lo, w_g_lo_0, x_lo_0);
        g_lo = vdotq_s32(g_lo, w_g_lo_1, x_lo_1);
        let mut g_hi = vdupq_n_s32(0);
        g_hi = vdotq_s32(g_hi, w_g_hi_0, x_hi_0);
        g_hi = vdotq_s32(g_hi, w_g_hi_1, x_hi_1);

        let mut u_lo = vdupq_n_s32(0);
        u_lo = vdotq_s32(u_lo, w_u_lo_0, x_lo_0);
        u_lo = vdotq_s32(u_lo, w_u_lo_1, x_lo_1);
        let mut u_hi = vdupq_n_s32(0);
        u_hi = vdotq_s32(u_hi, w_u_hi_0, x_hi_0);
        u_hi = vdotq_s32(u_hi, w_u_hi_1, x_hi_1);

        let dot_g_lo = vaddvq_s32(g_lo);
        let dot_g_hi = vaddvq_s32(g_hi);
        let dot_u_lo = vaddvq_s32(u_lo);
        let dot_u_hi = vaddvq_s32(u_hi);

        let s_lo = gidx * 2;
        let s_hi = gidx * 2 + 1;
        g_scaled_dot +=
            (scale.gate_sc[s_lo] as i32) * dot_g_lo + (scale.gate_sc[s_hi] as i32) * dot_g_hi;
        u_scaled_dot +=
            (scale.up_sc[s_lo] as i32) * dot_u_lo + (scale.up_sc[s_hi] as i32) * dot_u_hi;

        let bsum_lo = x.bsum32(s_lo) as i32;
        let bsum_hi = x.bsum32(s_hi) as i32;
        g_scaled_bsum +=
            (scale.gate_m[s_lo] as i32) * bsum_lo + (scale.gate_m[s_hi] as i32) * bsum_hi;
        u_scaled_bsum += (scale.up_m[s_lo] as i32) * bsum_lo + (scale.up_m[s_hi] as i32) * bsum_hi;
    }

    let gate_acc = (pair.gate.scale_int as i64) * (g_scaled_dot as i64)
        - (pair.gate.min_int as i64) * (g_scaled_bsum as i64);
    let up_acc = (pair.up.scale_int as i64) * (u_scaled_dot as i64)
        - (pair.up.min_int as i64) * (u_scaled_bsum as i64);

    (gate_acc, up_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn q4k_gu_block_unpack_checksum(pair: &GUPairQ4K) -> i64 {
    let (sc_g, m_g) = unpack_k4_scales(&pair.gate.sub_scales);
    let (sc_u, m_u) = unpack_k4_scales(&pair.up.sub_scales);
    let mut acc = 0i64;
    for i in 0..8 {
        acc = acc.wrapping_add(sc_g[i] as i64);
        acc = acc.wrapping_add(m_g[i] as i64);
        acc = acc.wrapping_add(sc_u[i] as i64);
        acc = acc.wrapping_add(m_u[i] as i64);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn q4k_gu_block_min_bsum_only(pair: &GUPairQ4K, x: &Q8KBlock) -> (i64, i64) {
    let (_sc_g, m_g) = unpack_k4_scales(&pair.gate.sub_scales);
    let (_sc_u, m_u) = unpack_k4_scales(&pair.up.sub_scales);
    let mut gate_bsum = 0i32;
    let mut up_bsum = 0i32;
    for j in 0..8 {
        let bsum = x.bsum32(j) as i32;
        gate_bsum += (m_g[j] as i32) * bsum;
        up_bsum += (m_u[j] as i32) * bsum;
    }
    (
        -((pair.gate.min_int as i64) * (gate_bsum as i64)),
        -((pair.up.min_int as i64) * (up_bsum as i64)),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4k_gate_block_neon(pair: &GUPairQ4K, x: &Q8KBlock) -> i64 {
    let mask_low = vdupq_n_u8(0x0F);
    let (sc_g, m_g) = unpack_k4_scales(&pair.gate.sub_scales);
    let qs_g = pair.gate.qs.as_ptr();
    let qs_x = x.qs.as_ptr();
    let mut g_scaled_dot: i32 = 0;
    let mut g_scaled_bsum: i32 = 0;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;

        let g_bytes_lo = vld1q_u8(qs_g.add(q_off));
        let g_bytes_hi = vld1q_u8(qs_g.add(q_off + 16));
        let w_g_lo_0 = vreinterpretq_s8_u8(vandq_u8(g_bytes_lo, mask_low));
        let w_g_lo_1 = vreinterpretq_s8_u8(vandq_u8(g_bytes_hi, mask_low));
        let w_g_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_lo, 4));
        let w_g_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_hi, 4));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut g_lo = vdupq_n_s32(0);
        g_lo = vdotq_s32(g_lo, w_g_lo_0, x_lo_0);
        g_lo = vdotq_s32(g_lo, w_g_lo_1, x_lo_1);
        let mut g_hi = vdupq_n_s32(0);
        g_hi = vdotq_s32(g_hi, w_g_hi_0, x_hi_0);
        g_hi = vdotq_s32(g_hi, w_g_hi_1, x_hi_1);

        let dot_g_lo = vaddvq_s32(g_lo);
        let dot_g_hi = vaddvq_s32(g_hi);

        let s_lo = gidx * 2;
        let s_hi = gidx * 2 + 1;
        g_scaled_dot += (sc_g[s_lo] as i32) * dot_g_lo + (sc_g[s_hi] as i32) * dot_g_hi;

        let bsum_lo = x.bsum32(s_lo) as i32;
        let bsum_hi = x.bsum32(s_hi) as i32;
        g_scaled_bsum += (m_g[s_lo] as i32) * bsum_lo + (m_g[s_hi] as i32) * bsum_hi;
    }

    (pair.gate.scale_int as i64) * (g_scaled_dot as i64)
        - (pair.gate.min_int as i64) * (g_scaled_bsum as i64)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4k_up_block_neon(pair: &GUPairQ4K, x: &Q8KBlock) -> i64 {
    let mask_low = vdupq_n_u8(0x0F);
    let (sc_u, m_u) = unpack_k4_scales(&pair.up.sub_scales);
    let qs_u = pair.up.qs.as_ptr();
    let qs_x = x.qs.as_ptr();
    let mut u_scaled_dot: i32 = 0;
    let mut u_scaled_bsum: i32 = 0;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;

        let u_bytes_lo = vld1q_u8(qs_u.add(q_off));
        let u_bytes_hi = vld1q_u8(qs_u.add(q_off + 16));
        let w_u_lo_0 = vreinterpretq_s8_u8(vandq_u8(u_bytes_lo, mask_low));
        let w_u_lo_1 = vreinterpretq_s8_u8(vandq_u8(u_bytes_hi, mask_low));
        let w_u_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_lo, 4));
        let w_u_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_hi, 4));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut u_lo = vdupq_n_s32(0);
        u_lo = vdotq_s32(u_lo, w_u_lo_0, x_lo_0);
        u_lo = vdotq_s32(u_lo, w_u_lo_1, x_lo_1);
        let mut u_hi = vdupq_n_s32(0);
        u_hi = vdotq_s32(u_hi, w_u_hi_0, x_hi_0);
        u_hi = vdotq_s32(u_hi, w_u_hi_1, x_hi_1);

        let dot_u_lo = vaddvq_s32(u_lo);
        let dot_u_hi = vaddvq_s32(u_hi);

        let s_lo = gidx * 2;
        let s_hi = gidx * 2 + 1;
        u_scaled_dot += (sc_u[s_lo] as i32) * dot_u_lo + (sc_u[s_hi] as i32) * dot_u_hi;

        let bsum_lo = x.bsum32(s_lo) as i32;
        let bsum_hi = x.bsum32(s_hi) as i32;
        u_scaled_bsum += (m_u[s_lo] as i32) * bsum_lo + (m_u[s_hi] as i32) * bsum_hi;
    }

    (pair.up.scale_int as i64) * (u_scaled_dot as i64)
        - (pair.up.min_int as i64) * (u_scaled_bsum as i64)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4k_gu_block_nibble_dot_only_neon(pair: &GUPairQ4K, x: &Q8KBlock) -> (i32, i32) {
    let mask_low = vdupq_n_u8(0x0F);
    let qs_g = pair.gate.qs.as_ptr();
    let qs_u = pair.up.qs.as_ptr();
    let qs_x = x.qs.as_ptr();
    let mut gate_dot = 0i32;
    let mut up_dot = 0i32;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;

        let g_bytes_lo = vld1q_u8(qs_g.add(q_off));
        let g_bytes_hi = vld1q_u8(qs_g.add(q_off + 16));
        let u_bytes_lo = vld1q_u8(qs_u.add(q_off));
        let u_bytes_hi = vld1q_u8(qs_u.add(q_off + 16));

        let w_g_lo_0 = vreinterpretq_s8_u8(vandq_u8(g_bytes_lo, mask_low));
        let w_g_lo_1 = vreinterpretq_s8_u8(vandq_u8(g_bytes_hi, mask_low));
        let w_u_lo_0 = vreinterpretq_s8_u8(vandq_u8(u_bytes_lo, mask_low));
        let w_u_lo_1 = vreinterpretq_s8_u8(vandq_u8(u_bytes_hi, mask_low));
        let w_g_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_lo, 4));
        let w_g_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(g_bytes_hi, 4));
        let w_u_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_lo, 4));
        let w_u_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(u_bytes_hi, 4));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut g_lo = vdupq_n_s32(0);
        g_lo = vdotq_s32(g_lo, w_g_lo_0, x_lo_0);
        g_lo = vdotq_s32(g_lo, w_g_lo_1, x_lo_1);
        let mut g_hi = vdupq_n_s32(0);
        g_hi = vdotq_s32(g_hi, w_g_hi_0, x_hi_0);
        g_hi = vdotq_s32(g_hi, w_g_hi_1, x_hi_1);

        let mut u_lo = vdupq_n_s32(0);
        u_lo = vdotq_s32(u_lo, w_u_lo_0, x_lo_0);
        u_lo = vdotq_s32(u_lo, w_u_lo_1, x_lo_1);
        let mut u_hi = vdupq_n_s32(0);
        u_hi = vdotq_s32(u_hi, w_u_hi_0, x_hi_0);
        u_hi = vdotq_s32(u_hi, w_u_hi_1, x_hi_1);

        gate_dot += vaddvq_s32(g_lo) + vaddvq_s32(g_hi);
        up_dot += vaddvq_s32(u_lo) + vaddvq_s32(u_hi);
    }

    (gate_dot, up_dot)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q4k_gu_neon(row: &[GUPairQ4K], h_q8k: &[Q8KBlock]) -> (i64, i64) {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;
    for (pair, x) in row.iter().zip(h_q8k.iter()) {
        let (g, u) = sdot_q4k_gu_block_neon(pair, x);
        gate_acc += g;
        up_acc += u;
    }

    (gate_acc, up_acc)
}

// ---------------------------------------------------------------------------
// Q5_K row kernel (MoE down_exps path — single tensor per expert, no gate/up).
// ---------------------------------------------------------------------------
//
// Q5_K 256-elem super-block layout:
//   - 8 sub-blocks × 32 elems (6-bit sub-block scales in `sub_scales[12]`)
//   - `qs_low[128]`: 4 LSB per elem, 2 elems per byte, same 4-group nibble
//     traversal as Q4_K (group g covers 64 elems: low nibbles → elems 0..32,
//     high nibbles → elems 32..64).
//   - `qs_high[32]`: 1 MSB per elem. Per group g ∈ 0..4, bit `(g*2)` of
//     `qs_high[l]` provides the high bit for elems `g*64 + l` (low-nibble
//     sub-block, `l ∈ 0..32`), and bit `(g*2 + 1)` provides the high bit for
//     elems `g*64 + 32 + l` (high-nibble sub-block).
//
// Block-level accumulate (identical structure to Q4_K, q replaced by 5-bit):
//
//   block_acc =   scale_int * Σⱼ sc[j] * <q5[j], x[j]>
//               - min_int   * Σⱼ  m[j] * bsums[j]
//
// Caller folds in `row_mul_weight * x.d` at row boundary.

/// Scalar oracle for [`sdot_q5k_row_neon`]. Mirrors `dequantize_q5k_intscale`
/// nibble + high-bit indexing exactly (same group/sub-block traversal as the
/// reference dequant, so any divergence is a bug in one place only).
///
/// Returns i64 (not i32) because of adversarial overflow bounds: per Q5_K
/// block, `scaled_dot ≤ 6.35e7` and `scale_int ∈ ±127`, so
/// `scale_int × scaled_dot` can reach ±8.06e9 — well outside i32 range
/// (±2.14e9). Widening the running accumulator to i64 keeps the kernel safe
/// across pathological weight/activation combinations. `scaled_dot` itself
/// stays i32 (fits by margin), and the i32→i64 promotion happens at the
/// `scale_int × scaled_dot` multiply site.
pub fn sdot_q5k_row_scalar(row: &[Q5KIntScale], h_q8k: &[Q8KBlock]) -> i64 {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut acc: i64 = 0;

    for (b, x) in row.iter().zip(h_q8k.iter()) {
        // Unpack 8 sub-block (sc, m) pairs from the 6-bit packed sub_scales.
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            let (s, mm) = get_scale_min_k4(j, &b.sub_scales);
            sc[j] = s;
            m[j] = mm;
        }

        let mut scaled_dot: i32 = 0;
        let mut scaled_bsum: i32 = 0;

        // 4 groups × 2 sub-blocks. Match dequantize_q5k_intscale u1/u2 masks.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for g in 0..4 {
            let q_off = g * 32;
            let x_off = g * 64;

            let mut sub_dot_lo: i32 = 0;
            let mut sub_dot_hi: i32 = 0;
            for l in 0..32 {
                let q_byte = b.qs_low[q_off + l];
                let h_byte = b.qs_high[l];

                let hi_bit_lo: u8 = if h_byte & u1 != 0 { 16 } else { 0 };
                let hi_bit_hi: u8 = if h_byte & u2 != 0 { 16 } else { 0 };

                let w_lo = ((q_byte & 0x0F) + hi_bit_lo) as i32;
                let w_hi = ((q_byte >> 4) + hi_bit_hi) as i32;

                sub_dot_lo += w_lo * (x.qs[x_off + l] as i32);
                sub_dot_hi += w_hi * (x.qs[x_off + 32 + l] as i32);
            }

            let s_lo = g * 2;
            let s_hi = g * 2 + 1;
            scaled_dot += (sc[s_lo] as i32) * sub_dot_lo + (sc[s_hi] as i32) * sub_dot_hi;
            scaled_bsum += (m[s_lo] as i32) * (x.bsum32(s_lo) as i32)
                + (m[s_hi] as i32) * (x.bsum32(s_hi) as i32);

            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }

        acc +=
            (b.scale_int as i64) * (scaled_dot as i64) - (b.min_int as i64) * (scaled_bsum as i64);
    }
    acc
}

/// NEON variant of [`sdot_q5k_row_scalar`]. Follows the high-bit expansion
/// technique from `dot_q5_k_q8k_neon` in `neon_dot.rs`:
///   - Load `qs_high[0..32]` as two 16-byte registers (`qh0`, `qh1`).
///   - Per group, `vandq_u8(qh, mone)` isolates bit (g*2), shifted left by 4
///     gives 0 or 16 per byte. `vandq_u8(qh, mtwo)` isolates bit (g*2+1),
///     shifted left by 3 gives 0 or 16 per byte.
///   - After each group, `vshrq_n_u8(qh, 2)` advances to the next group's bits.
///   - Combine `low_nibble | high_bit` via `vorrq_u8` → 5-bit unsigned (0..31)
///     fits in `i8` positive range for `vdotq_s32`.
///   - Scale/min weighting stays in scalar i32, matching the Q4_K GU kernel.
///
/// Returns i64 to match the scalar oracle — see its doc comment for the
/// adversarial overflow bound that forces i64.
///
/// # Safety
///
/// - Requires the ARMv8.2 `dotprod` extension (`vdotq_s32`). The caller must
///   ensure the target CPU has `neon` + `dotprod` available; the
///   `#[target_feature(enable = "neon,dotprod")]` attribute below gates the
///   intrinsic but does not probe the CPU at runtime.
/// - All `vld1q_u8` / `vld1q_s8` loads are in-bounds by construction: the
///   weight fields `Q5KIntScale.qs_low: [u8; 128]` and
///   `Q5KIntScale.qs_high: [u8; 32]`, and the activation
///   `Q8KBlock.qs: [i8; 256]`, are fixed-size arrays. The `qs_high` pre-loop
///   loads `qs_high.add(0)` and `qs_high.add(16)` (valid for a 32-byte array).
///   Inside the loop `gidx ∈ 0..4` with `q_off = gidx*32` and `x_off = gidx*64`
///   means the largest offsets touched are `qs_low.add(112)` (plus 16 bytes =
///   byte 127) and `x.qs.add(240)` (plus 16 bytes = byte 255). Every 16-byte
///   load stays within its fixed-size array.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q5k_row_block_neon(b: &Q5KIntScale, x: &Q8KBlock) -> i64 {
    let mask_low = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x01);
    let mtwo = vdupq_n_u8(0x02);

    let (sc, m) = unpack_k4_scales(&b.sub_scales);

    let qs_low_ptr = b.qs_low.as_ptr();
    let qs_high_ptr = b.qs_high.as_ptr();
    let qs_x = x.qs.as_ptr();
    let mut qh0 = vld1q_u8(qs_high_ptr);
    let mut qh1 = vld1q_u8(qs_high_ptr.add(16));

    let mut scaled_dot: i32 = 0;
    let mut scaled_bsum: i32 = 0;

    for gidx in 0..4 {
        let q_off = gidx * 32;
        let x_off = gidx * 64;
        let ql0 = vld1q_u8(qs_low_ptr.add(q_off));
        let ql1 = vld1q_u8(qs_low_ptr.add(q_off + 16));
        let h_lo_0 = vshlq_n_u8(vandq_u8(qh0, mone), 4);
        let h_lo_1 = vshlq_n_u8(vandq_u8(qh1, mone), 4);
        let h_hi_0 = vshlq_n_u8(vandq_u8(qh0, mtwo), 3);
        let h_hi_1 = vshlq_n_u8(vandq_u8(qh1, mtwo), 3);
        qh0 = vshrq_n_u8(qh0, 2);
        qh1 = vshrq_n_u8(qh1, 2);

        let w_lo_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_low), h_lo_0));
        let w_lo_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_low), h_lo_1));
        let w_hi_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), h_hi_0));
        let w_hi_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), h_hi_1));

        let x_lo_0 = vld1q_s8(qs_x.add(x_off));
        let x_lo_1 = vld1q_s8(qs_x.add(x_off + 16));
        let x_hi_0 = vld1q_s8(qs_x.add(x_off + 32));
        let x_hi_1 = vld1q_s8(qs_x.add(x_off + 48));

        let mut acc_lo = vdupq_n_s32(0);
        acc_lo = vdotq_s32(acc_lo, w_lo_0, x_lo_0);
        acc_lo = vdotq_s32(acc_lo, w_lo_1, x_lo_1);
        let mut acc_hi = vdupq_n_s32(0);
        acc_hi = vdotq_s32(acc_hi, w_hi_0, x_hi_0);
        acc_hi = vdotq_s32(acc_hi, w_hi_1, x_hi_1);

        let dot_lo = vaddvq_s32(acc_lo);
        let dot_hi = vaddvq_s32(acc_hi);
        let s_lo = gidx * 2;
        let s_hi = gidx * 2 + 1;
        scaled_dot += (sc[s_lo] as i32) * dot_lo + (sc[s_hi] as i32) * dot_hi;

        let bsum_lo = x.bsum32(s_lo) as i32;
        let bsum_hi = x.bsum32(s_hi) as i32;
        scaled_bsum += (m[s_lo] as i32) * bsum_lo + (m[s_hi] as i32) * bsum_hi;
    }

    (b.scale_int as i64) * (scaled_dot as i64) - (b.min_int as i64) * (scaled_bsum as i64)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q5k_row_neon(row: &[Q5KIntScale], h_q8k: &[Q8KBlock]) -> i64 {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut acc: i64 = 0;
    for (b, x) in row.iter().zip(h_q8k.iter()) {
        acc += sdot_q5k_row_block_neon(b, x);
    }
    acc
}

// ---------------------------------------------------------------------------
// Q8_0 GU kernel (shared expert gate/up — Qwen3.6 only).
// ---------------------------------------------------------------------------
//
// Shared-expert layout (one `SharedGUQ8KUnit` covers 256 elems = 1 Q8K block):
//   - `gate_q8_0[0..8]`: 8 Q8_0 blocks, each with i16 `scale_int` + `[i8; 32]` qs
//   - `up_q8_0[0..8]`:   same layout for up-proj
//
// Q8_0 is simpler than Q4_K/Q5_K: no 6-bit packed sub-scales, no nibble unpack,
// no high-bit expansion. Per sub-block the accumulate is just
//
//   sub_dot = Σⱼ qs[j] * x[j]    (j ∈ 0..32)
//   unit_acc += scale_int * sub_dot
//
// Caller multiplies by `row_mul * x.d` at row boundary to recover f32.

/// Scalar oracle for [`sdot_q80_gu_neon`]. Computes block-interleaved gate/up
/// dot for shared-expert Q8_0 weight layout.
///
/// Returns `(gate_i64, up_i64)`. The caller multiplies each by
/// `row_mul_gate * x.d` / `row_mul_up * x.d` to recover the f32 dot product.
///
/// Note: returned type is `i64` because Q8_0 `scale_int` is `i16` (±32767) and
/// the per-sub-block i32 dot product (`qs_w * qs_x` summed over 32 elems, each
/// ±127) can reach ±5e5, so `scale_int * dot` alone is up to ±1.6e10, which
/// overflows i32 once accumulated across 8 sub-blocks per unit. Q4_K/Q5_K GU
/// kernels use `i32` because their `scale_int` is `i8` (±127), so the
/// accumulate fits comfortably in i32.
pub fn sdot_q80_gu_scalar(row: &[SharedGUQ8KUnit], h_q8k: &[Q8KBlock]) -> (i64, i64) {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;
    for (unit, x) in row.iter().zip(h_q8k.iter()) {
        for sub in 0..8 {
            let x_base = sub * 32;
            let g = &unit.gate_q8_0[sub];
            let u = &unit.up_q8_0[sub];
            let mut dot_g: i32 = 0;
            let mut dot_u: i32 = 0;
            for j in 0..32 {
                let xq = x.qs[x_base + j] as i32;
                dot_g += (g.qs[j] as i32) * xq;
                dot_u += (u.qs[j] as i32) * xq;
            }
            gate_acc += (g.scale_int as i64) * (dot_g as i64);
            up_acc += (u.scale_int as i64) * (dot_u as i64);
        }
    }
    (gate_acc, up_acc)
}

/// NEON variant of [`sdot_q80_gu_scalar`]. Uses `vdotq_s32` for the per-sub-block
/// `<qs, x>` inner products. Scale weighting stays in scalar i32 (cost negligible).
///
/// # Safety
///
/// - Requires the ARMv8.2 `dotprod` extension (`vdotq_s32`). The caller must
///   ensure the target CPU has `neon` + `dotprod` available; the
///   `#[target_feature(enable = "neon,dotprod")]` attribute below gates the
///   intrinsic but does not probe the CPU at runtime.
/// - All `vld1q_s8` loads are in-bounds by construction: the weight fields
///   `Q80IntScale.qs: [i8; 32]` (inside each `SharedGUQ8KUnit.gate_q8_0[sub]`
///   and `.up_q8_0[sub]` for `sub ∈ 0..8`) and the activation
///   `Q8KBlock.qs: [i8; 256]` are fixed-size arrays. Per iteration the kernel
///   loads `qs.as_ptr()` and `qs.as_ptr().add(16)` of each weight block
///   (bytes 0..32 of a 32-byte array) and `x.qs.as_ptr().add(x_base)` plus
///   `.add(x_base + 16)` where `x_base = sub*32 ∈ {0, 32, …, 224}` (last
///   16-byte load lands at byte 240..256, still within the 256-byte array).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q80_gu_block_neon(unit: &SharedGUQ8KUnit, x: &Q8KBlock) -> (i64, i64) {
    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;

    for sub in 0..8 {
        let x_base = sub * 32;
        let g = &unit.gate_q8_0[sub];
        let u = &unit.up_q8_0[sub];

        let g_lo = vld1q_s8(g.qs.as_ptr());
        let g_hi = vld1q_s8(g.qs.as_ptr().add(16));
        let u_lo = vld1q_s8(u.qs.as_ptr());
        let u_hi = vld1q_s8(u.qs.as_ptr().add(16));
        let x_lo = vld1q_s8(x.qs.as_ptr().add(x_base));
        let x_hi = vld1q_s8(x.qs.as_ptr().add(x_base + 16));

        let mut dot_g = vdupq_n_s32(0);
        let mut dot_u = vdupq_n_s32(0);
        dot_g = vdotq_s32(dot_g, g_lo, x_lo);
        dot_g = vdotq_s32(dot_g, g_hi, x_hi);
        dot_u = vdotq_s32(dot_u, u_lo, x_lo);
        dot_u = vdotq_s32(dot_u, u_hi, x_hi);

        gate_acc += (g.scale_int as i64) * (vaddvq_s32(dot_g) as i64);
        up_acc += (u.scale_int as i64) * (vaddvq_s32(dot_u) as i64);
    }

    (gate_acc, up_acc)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q80_gu_neon(row: &[SharedGUQ8KUnit], h_q8k: &[Q8KBlock]) -> (i64, i64) {
    assert_eq!(
        row.len(),
        h_q8k.len(),
        "row.len()={} must match h_q8k.len()={}",
        row.len(),
        h_q8k.len()
    );

    let mut gate_acc: i64 = 0;
    let mut up_acc: i64 = 0;
    for (unit, x) in row.iter().zip(h_q8k.iter()) {
        let (g, u) = sdot_q80_gu_block_neon(unit, x);
        gate_acc += g;
        up_acc += u;
    }

    (gate_acc, up_acc)
}

// ---------------------------------------------------------------------------
// Q8_0 row kernel (shared expert down — Qwen3.6 only).
// ---------------------------------------------------------------------------
//
// Shared-expert down row is a flat `&[Q80IntScale]` where every 8 consecutive
// Q8_0 blocks cover the 256 elems of one Q8K activation block:
//   row.len() == h_q8k.len() * 8
//
// Per sub-block (index `sub ∈ 0..8`) within Q8K block `x_idx`:
//   blk = row[x_idx * 8 + sub]
//   sub_dot = Σⱼ blk.qs[j] * x.qs[sub*32 + j]    (j ∈ 0..32)
//   row_acc += scale_int * sub_dot
//
// Caller multiplies by `row_mul × x.d` at row boundary to recover f32.
//
// Q8_0 `scale_int` is `i16` (±32767); per-sub-block i32 dot ranges ±5e5, and
// accumulating `scale_int * sub_dot` across multiple Q8K blocks overflows i32
// (observed in Task 6). Return `i64` instead.

/// Scalar oracle for [`sdot_q80_row_neon`].
/// Input: `row.len() == h_q8k.len() * 8` (8 Q8_0 blocks per Q8K activation block).
/// Returns i64 accumulate; caller multiplies by `row_mul × x.d` for f32.
/// Uses i64 because Q8_0 `scale_int` is i16 — i32 accumulator can overflow
/// across multi-block rows (discovered in Task 6).
pub fn sdot_q80_row_scalar(row: &[Q80IntScale], h_q8k: &[Q8KBlock]) -> i64 {
    assert_eq!(
        row.len(),
        h_q8k.len() * 8,
        "row.len()={} must equal h_q8k.len() * 8 (={}*8={})",
        row.len(),
        h_q8k.len(),
        h_q8k.len() * 8
    );

    let mut acc: i64 = 0;
    for (x_idx, x) in h_q8k.iter().enumerate() {
        for sub in 0..8 {
            let blk = &row[x_idx * 8 + sub];
            let x_base = sub * 32;
            let mut sub_dot: i32 = 0;
            for j in 0..32 {
                sub_dot += (blk.qs[j] as i32) * (x.qs[x_base + j] as i32);
            }
            acc += (blk.scale_int as i64) * (sub_dot as i64);
        }
    }
    acc
}

/// NEON variant of [`sdot_q80_row_scalar`]. Uses `vdotq_s32` for the
/// per-sub-block `<qs, x>` inner products; scale weighting stays in scalar i64
/// (cost negligible).
///
/// # Safety
///
/// - Requires the ARMv8.2 `dotprod` extension (`vdotq_s32`). The caller must
///   ensure the target CPU has `neon` + `dotprod` available; the
///   `#[target_feature(enable = "neon,dotprod")]` attribute below gates the
///   intrinsic but does not probe the CPU at runtime.
/// - All `vld1q_s8` loads are in-bounds by construction: weight blocks have
///   `Q80IntScale.qs: [i8; 32]` (loads touch bytes 0..32) and the activation
///   `Q8KBlock.qs: [i8; 256]`. The `sub ∈ 0..8` loop uses `x_base = sub*32`,
///   so `x.qs.as_ptr().add(x_base + 16)` has a largest value of
///   `x.qs.add(240)` with a 16-byte load ending at byte 256 — exactly at the
///   array boundary, still valid.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q80_row_block_neon(row8: &[Q80IntScale], x: &Q8KBlock) -> i64 {
    debug_assert_eq!(row8.len(), 8);
    let mut acc: i64 = 0;
    for (sub, blk) in row8.iter().enumerate() {
        let x_base = sub * 32;
        let w_lo = vld1q_s8(blk.qs.as_ptr());
        let w_hi = vld1q_s8(blk.qs.as_ptr().add(16));
        let x_lo = vld1q_s8(x.qs.as_ptr().add(x_base));
        let x_hi = vld1q_s8(x.qs.as_ptr().add(x_base + 16));
        let mut dot = vdupq_n_s32(0);
        dot = vdotq_s32(dot, w_lo, x_lo);
        dot = vdotq_s32(dot, w_hi, x_hi);
        acc += (blk.scale_int as i64) * (vaddvq_s32(dot) as i64);
    }
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn sdot_q80_row_neon(row: &[Q80IntScale], h_q8k: &[Q8KBlock]) -> i64 {
    assert_eq!(
        row.len(),
        h_q8k.len() * 8,
        "row.len()={} must equal h_q8k.len() * 8 (={}*8={})",
        row.len(),
        h_q8k.len(),
        h_q8k.len() * 8
    );

    let mut acc: i64 = 0;
    for (x_idx, x) in h_q8k.iter().enumerate() {
        acc += sdot_q80_row_block_neon(&row[x_idx * 8..(x_idx + 1) * 8], x);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::quantize_input_q8k;
    use crate::quantize::blocks::{BlockQ4_K, BlockQ5_K, BlockQ8_0};
    use crate::quantize::moe_blocks::Q4KIntScale;
    use crate::quantize::moe_convert::{
        dequantize_q4k_intscale, dequantize_q5k_intscale, dequantize_q80_intscale,
        row_q4k_to_intscale, row_q5k_to_intscale, row_q80_to_intscale,
    };
    use half::f16;

    /// Pack per-sub-block `sc[0..8]` / `m[0..8]` (each 6-bit) into GGUF
    /// `scales: [u8; 12]` layout. Inverse of `get_scale_min_k4`. Copied from
    /// moe_convert tests — needed here to build fixtures with non-uniform
    /// sub-block scales so a broken kernel can't pass by ignoring `sub_scales`.
    fn pack_q4k_sub_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (sc[j] & 0x3F) | (((sc[j + 4] >> 4) & 0x03) << 6);
            scales[j + 4] = (m[j] & 0x3F) | (((m[j + 4] >> 4) & 0x03) << 6);
            scales[j + 8] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
        }
        scales
    }

    #[test]
    fn unpack_k4_scales_matches_canonical_decoder() {
        let sc_in = [1u8, 7, 12, 31, 33, 42, 55, 63];
        let m_in = [2u8, 9, 14, 29, 34, 43, 56, 62];
        let packed = pack_q4k_sub_scales(&sc_in, &m_in);
        let (sc, m) = super::unpack_k4_scales(&packed);

        for j in 0..8 {
            let (s, mm) = get_scale_min_k4(j, &packed);
            assert_eq!(sc[j], s, "scale mismatch at sub-block {j}");
            assert_eq!(m[j], mm, "min mismatch at sub-block {j}");
        }
    }

    /// Non-uniform synthetic Q4_K block — sc/m differ by sub-block index so
    /// any kernel that ignores `sub_scales` will diverge measurably.
    fn synth_q4k_block(idx: usize, offset: u8) -> BlockQ4_K {
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = ((i as u8).wrapping_add(offset)).wrapping_mul(17);
        }
        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m = [9u8, 10, 11, 12, 13, 14, 15, 16];
        let scales = pack_q4k_sub_scales(&sc, &m);
        BlockQ4_K {
            d: f16::from_f32(0.05 * (1.0 + 0.1 * idx as f32)),
            dmin: f16::from_f32(0.02 * (1.0 + 0.05 * idx as f32)),
            scales,
            qs,
        }
    }

    /// Build a GUPairQ4K row (n_blocks blocks) where gate and up come from
    /// different synthetic patterns. Returns (row, row_mul_gate, row_mul_up).
    fn build_gu_row(n_blocks: usize) -> (Vec<GUPairQ4K>, f32, f32) {
        let gate_blocks: Vec<BlockQ4_K> = (0..n_blocks).map(|i| synth_q4k_block(i, 0)).collect();
        let up_blocks: Vec<BlockQ4_K> = (0..n_blocks).map(|i| synth_q4k_block(i, 53)).collect();

        let (gate_is, row_mul_g) = row_q4k_to_intscale(&gate_blocks);
        let (up_is, row_mul_u) = row_q4k_to_intscale(&up_blocks);

        let pairs: Vec<GUPairQ4K> = gate_is
            .into_iter()
            .zip(up_is.into_iter())
            .map(|(g, u)| GUPairQ4K { gate: g, up: u })
            .collect();
        (pairs, row_mul_g, row_mul_u)
    }

    /// Build a deterministic f32 activation vector (n_blocks * 256 elems).
    fn build_activation(n_blocks: usize) -> Vec<f32> {
        let n = n_blocks * 256;
        (0..n)
            .map(|i| {
                let t = i as f32 * 0.0137;
                t.sin() + 0.4 * (t * 1.7).cos() - 0.3 * (t * 0.3).sin()
            })
            .collect()
    }

    /// Reference f32 dot <dequant(row), x>.
    fn f32_reference_dot(row_is: &[Q4KIntScale], row_mul: f32, x: &[f32]) -> f32 {
        assert_eq!(row_is.len() * 256, x.len());
        let mut acc = 0.0f64;
        for (bi, b) in row_is.iter().enumerate() {
            let dq = dequantize_q4k_intscale(b, row_mul);
            for l in 0..256 {
                acc += (dq[l] as f64) * (x[bi * 256 + l] as f64);
            }
        }
        acc as f32
    }

    #[test]
    fn q4k_gu_scalar_matches_float_reference() {
        let n_blocks = 4;
        let (row, row_mul_g, row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        // Dequantized f32 reference for each path.
        let gate_row: Vec<Q4KIntScale> = row.iter().map(|p| p.gate).collect();
        let up_row: Vec<Q4KIntScale> = row.iter().map(|p| p.up).collect();
        let ref_gate = f32_reference_dot(&gate_row, row_mul_g, &x_f32);
        let ref_up = f32_reference_dot(&up_row, row_mul_u, &x_f32);

        // Scalar kernel over Q8K-quantized activation.
        let (gate_i, up_i) = sdot_q4k_gu_scalar(&row, &h_q8k);

        // Recombine: each block contributes `x.d` per-block, so we can't
        // simply multiply by a single x.d — instead the kernel's i64 is a
        // weighted sum. We fold block-by-block to recover the f32 value:
        // here the kernel already summed across all blocks with equal
        // integer weights (the `x.d` differs per block but the integer acc
        // alone is only meaningful under constant d). To get a fair f32
        // comparison, recompute the scalar kernel per-block with its own
        // `x.d` fold.
        //
        // Simpler: repeat sdot block-by-block and fold.
        let mut gate_f = 0.0f64;
        let mut up_f = 0.0f64;
        for bi in 0..n_blocks {
            let pair_slice = &row[bi..bi + 1];
            let q8k_slice = &h_q8k[bi..bi + 1];
            let (gi, ui) = sdot_q4k_gu_scalar(pair_slice, q8k_slice);
            gate_f += (gi as f64) * (row_mul_g as f64) * (h_q8k[bi].d as f64);
            up_f += (ui as f64) * (row_mul_u as f64) * (h_q8k[bi].d as f64);
        }
        let gate_f = gate_f as f32;
        let up_f = up_f as f32;

        let rel_gate = ((gate_f - ref_gate) / ref_gate).abs();
        let rel_up = ((up_f - ref_up) / ref_up).abs();
        eprintln!(
            "q4k_gu scalar: gate_f={gate_f:.4} ref={ref_gate:.4} rel={rel_gate:.5} | up_f={up_f:.4} ref={ref_up:.4} rel={rel_up:.5}"
        );
        assert!(rel_gate < 0.01, "gate rel err too high: {rel_gate}");
        assert!(rel_up < 0.01, "up rel err too high: {rel_up}");

        // Also: gate_i / up_i should be non-trivial (not both zero) so we
        // actually exercised the accumulate path.
        assert!(
            gate_i != 0,
            "gate_i unexpectedly zero — test fixture degenerate"
        );
        assert!(
            up_i != 0,
            "up_i unexpectedly zero — test fixture degenerate"
        );
    }

    /// NEON kernel must exactly match the scalar oracle (i64 == i64).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_gu_neon_matches_scalar() {
        let n_blocks = 4;
        let (row, _row_mul_g, _row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        let (scalar_g, scalar_u) = sdot_q4k_gu_scalar(&row, &h_q8k);
        let (neon_g, neon_u) = unsafe { sdot_q4k_gu_neon(&row, &h_q8k) };

        assert_eq!(
            scalar_g, neon_g,
            "gate i64 mismatch: scalar={scalar_g} neon={neon_g}"
        );
        assert_eq!(
            scalar_u, neon_u,
            "up i64 mismatch: scalar={scalar_u} neon={neon_u}"
        );
    }

    /// Microbench attribution helpers must be non-degenerate and preserve the
    /// full gate result when measuring the gate-only path.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_gu_microbench_helpers_are_componentized() {
        let n_blocks = 4;
        let (row, _row_mul_g, _row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);
        let pair = &row[0];
        let x = &h_q8k[0];

        let (full_g, full_u) = unsafe { sdot_q4k_gu_block_neon(pair, x) };
        let gate_only = unsafe { q4k_gate_block_neon(pair, x) };
        let up_only = unsafe { q4k_up_block_neon(pair, x) };
        let (dot_g, dot_u) = unsafe { q4k_gu_block_nibble_dot_only_neon(pair, x) };
        let (min_g, min_u) = q4k_gu_block_min_bsum_only(pair, x);
        let unpack = q4k_gu_block_unpack_checksum(pair);

        assert_eq!(gate_only, full_g);
        assert_eq!(up_only, full_u);
        assert_ne!(full_g, 0, "full gate path is degenerate");
        assert_ne!(full_u, 0, "full up path is degenerate");
        assert_ne!(dot_g, 0, "gate dot-only path is degenerate");
        assert_ne!(dot_u, 0, "up dot-only path is degenerate");
        assert_ne!(min_g, 0, "gate min/bsum path is degenerate");
        assert_ne!(min_u, 0, "up min/bsum path is degenerate");
        assert_ne!(unpack, 0, "unpack-only path is degenerate");
    }

    /// An unpacked-scale block keeps the same math as the regular kernel but
    /// moves Q4_K 6-bit scale/min unpacking out of the hot call.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_gu_unpacked_scales_kernel_matches_regular_neon() {
        let n_blocks = 4;
        let (row, _row_mul_g, _row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);
        let pair = &row[0];
        let x = &h_q8k[0];

        let unpacked = GUPairQ4KUnpackedScales::from_pair(*pair);
        let regular = unsafe { sdot_q4k_gu_block_neon(pair, x) };
        let candidate = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&unpacked, x) };

        assert_eq!(regular, candidate);
    }

    /// A separate scale/min side plane keeps the same math as the regular
    /// kernel while leaving the Q4_K pair payload in regular layout.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_gu_scale_plane_kernel_matches_regular_neon() {
        let n_blocks = 4;
        let (row, _row_mul_g, _row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);
        let pair = &row[0];
        let x = &h_q8k[0];

        let scale = GUPairQ4KScaleMin::from_pair(*pair);
        let regular = unsafe { sdot_q4k_gu_block_neon(pair, x) };
        let candidate = unsafe { sdot_q4k_gu_block_scale_min_neon(pair, &scale, x) };

        assert_eq!(regular, candidate);
    }

    /// Full-path f32 round-trip: NEON integer × row_mul × x.d should match
    /// the f32 reference dot within < 1% relative error.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_gu_neon_full_path_matches_f32_reference() {
        let n_blocks = 4;
        let (row, row_mul_g, row_mul_u) = build_gu_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        let gate_row: Vec<Q4KIntScale> = row.iter().map(|p| p.gate).collect();
        let up_row: Vec<Q4KIntScale> = row.iter().map(|p| p.up).collect();
        let ref_gate = f32_reference_dot(&gate_row, row_mul_g, &x_f32);
        let ref_up = f32_reference_dot(&up_row, row_mul_u, &x_f32);

        // Per-block NEON fold (same shape as the caller in Task 13).
        let mut gate_f = 0.0f64;
        let mut up_f = 0.0f64;
        for bi in 0..n_blocks {
            let (gi, ui) = unsafe { sdot_q4k_gu_neon(&row[bi..bi + 1], &h_q8k[bi..bi + 1]) };
            gate_f += (gi as f64) * (row_mul_g as f64) * (h_q8k[bi].d as f64);
            up_f += (ui as f64) * (row_mul_u as f64) * (h_q8k[bi].d as f64);
        }
        let gate_f = gate_f as f32;
        let up_f = up_f as f32;

        let rel_gate = ((gate_f - ref_gate) / ref_gate).abs();
        let rel_up = ((up_f - ref_up) / ref_up).abs();
        eprintln!(
            "q4k_gu neon full: gate={gate_f:.4} ref={ref_gate:.4} rel={rel_gate:.5} | up={up_f:.4} ref={ref_up:.4} rel={rel_up:.5}"
        );
        assert!(rel_gate < 0.01, "gate rel err too high: {rel_gate}");
        assert!(rel_up < 0.01, "up rel err too high: {rel_up}");
    }

    // ----------------------------------------------------------------------
    // Q5_K row kernel tests
    // ----------------------------------------------------------------------

    /// Pack per-sub-block sc/m into GGUF 6-bit `scales: [u8; 12]` layout.
    /// Same layout as Q4_K (shared `get_scale_min_k4` decoder).
    fn pack_q5k_sub_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (sc[j] & 0x3F) | (((sc[j + 4] >> 4) & 0x03) << 6);
            scales[j + 4] = (m[j] & 0x3F) | (((m[j + 4] >> 4) & 0x03) << 6);
            scales[j + 8] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
        }
        scales
    }

    /// Non-uniform synthetic Q5_K block — sc/m differ by sub-block index and
    /// qs/qh vary by block index so any kernel that ignores sub_scales or
    /// high-bits will diverge measurably.
    fn synth_q5k_block(idx: usize, offset: u8) -> BlockQ5_K {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for i in 0..128 {
            qs[i] = ((i as u8).wrapping_add(offset)).wrapping_mul(19);
        }
        for i in 0..32 {
            // Vary high-bit pattern per block: different byte values mean every
            // bit position (g*2, g*2+1 for g ∈ 0..4) is genuinely exercised.
            qh[i] = ((i as u8).wrapping_add(offset).wrapping_add(idx as u8 * 7)).wrapping_mul(29);
        }
        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m = [9u8, 10, 11, 12, 13, 14, 15, 16];
        let scales = pack_q5k_sub_scales(&sc, &m);
        BlockQ5_K {
            d: f16::from_f32(0.05 * (1.0 + 0.1 * idx as f32)),
            dmin: f16::from_f32(0.02 * (1.0 + 0.05 * idx as f32)),
            scales,
            qh,
            qs,
        }
    }

    /// Build a Q5_K row (n_blocks blocks). Returns (row, row_mul).
    fn build_q5k_row(n_blocks: usize) -> (Vec<Q5KIntScale>, f32) {
        let blocks: Vec<BlockQ5_K> = (0..n_blocks).map(|i| synth_q5k_block(i, 0)).collect();
        row_q5k_to_intscale(&blocks)
    }

    /// f32 reference dot: dequantize(row) · x, same shape as Q4_K reference.
    fn f32_reference_dot_q5k(row_is: &[Q5KIntScale], row_mul: f32, x: &[f32]) -> f32 {
        assert_eq!(row_is.len() * 256, x.len());
        let mut acc = 0.0f64;
        for (bi, b) in row_is.iter().enumerate() {
            let dq = dequantize_q5k_intscale(b, row_mul);
            for l in 0..256 {
                acc += (dq[l] as f64) * (x[bi * 256 + l] as f64);
            }
        }
        acc as f32
    }

    #[test]
    fn q5k_row_scalar_matches_float_reference_rel_err_under_1pct() {
        let n_blocks = 4;
        let (row, row_mul) = build_q5k_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        let ref_dot = f32_reference_dot_q5k(&row, row_mul, &x_f32);

        // Per-block fold (mirrors Task 13 caller: each block has its own x.d).
        let mut f_acc = 0.0f64;
        for bi in 0..n_blocks {
            let i = sdot_q5k_row_scalar(&row[bi..bi + 1], &h_q8k[bi..bi + 1]);
            f_acc += (i as f64) * (row_mul as f64) * (h_q8k[bi].d as f64);
        }
        let f = f_acc as f32;

        // Sanity: single-call accumulate must be non-zero (fixture not degenerate).
        let i_all = sdot_q5k_row_scalar(&row, &h_q8k);
        assert!(
            i_all != 0,
            "i_all unexpectedly zero — test fixture degenerate"
        );

        let rel = ((f - ref_dot) / ref_dot).abs();
        eprintln!("q5k_row scalar: f={f:.4} ref={ref_dot:.4} rel={rel:.5} i_all={i_all}");
        assert!(rel < 0.01, "q5k_row scalar rel err too high: {rel}");
    }

    /// NEON kernel must exactly match the scalar oracle (i64 == i64).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q5k_row_neon_matches_scalar() {
        let n_blocks = 4;
        let (row, _row_mul) = build_q5k_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        let scalar_i = sdot_q5k_row_scalar(&row, &h_q8k);
        let neon_i = unsafe { sdot_q5k_row_neon(&row, &h_q8k) };

        assert_eq!(
            scalar_i, neon_i,
            "q5k_row i64 mismatch: scalar={scalar_i} neon={neon_i}"
        );

        // Also verify per-block equality — the caller calls it per-block in
        // Task 13 to fold x.d correctly.
        for bi in 0..n_blocks {
            let s = sdot_q5k_row_scalar(&row[bi..bi + 1], &h_q8k[bi..bi + 1]);
            let n = unsafe { sdot_q5k_row_neon(&row[bi..bi + 1], &h_q8k[bi..bi + 1]) };
            assert_eq!(
                s, n,
                "q5k_row per-block mismatch at bi={bi}: scalar={s} neon={n}"
            );
        }
    }

    /// Full-path f32 round-trip: NEON integer × row_mul × x.d should match
    /// the f32 reference dot within < 1% relative error.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q5k_row_neon_full_path_matches_f32_reference() {
        let n_blocks = 4;
        let (row, row_mul) = build_q5k_row(n_blocks);
        let x_f32 = build_activation(n_blocks);
        let h_q8k = quantize_input_q8k(&x_f32);

        let ref_dot = f32_reference_dot_q5k(&row, row_mul, &x_f32);

        // Per-block NEON fold (same shape as the caller in Task 13).
        let mut f_acc = 0.0f64;
        for bi in 0..n_blocks {
            let i = unsafe { sdot_q5k_row_neon(&row[bi..bi + 1], &h_q8k[bi..bi + 1]) };
            f_acc += (i as f64) * (row_mul as f64) * (h_q8k[bi].d as f64);
        }
        let f = f_acc as f32;

        let rel = ((f - ref_dot) / ref_dot).abs();
        eprintln!("q5k_row neon full: f={f:.4} ref={ref_dot:.4} rel={rel:.5}");
        assert!(rel < 0.01, "q5k_row neon full-path rel err too high: {rel}");
    }

    // ----------------------------------------------------------------------
    // Q8_0 GU kernel tests (shared expert gate/up)
    // ----------------------------------------------------------------------

    /// Synthetic BlockQ8_0 with non-trivial qs + varying d per block index.
    /// A kernel that ignores `scale_int` or reads the wrong sub-block will
    /// diverge measurably.
    fn synth_q80_block(idx: usize) -> BlockQ8_0 {
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (((i + idx * 7) as i32) % 127 - 63) as i8;
        }
        BlockQ8_0 {
            d: f16::from_f32(0.04 * (1.0 + 0.13 * idx as f32)),
            qs,
        }
    }

    /// Build a SharedGUQ8KUnit row covering `n_units * 256` elems.
    /// Gate and up come from different synthetic patterns so divergence is
    /// observable. Returns (row, row_mul_gate, row_mul_up).
    fn synth_shared_gu_row(n_units: usize) -> (Vec<SharedGUQ8KUnit>, f32, f32) {
        let total_blocks = 8 * n_units;
        let gate_blocks: Vec<BlockQ8_0> = (0..total_blocks).map(|i| synth_q80_block(i)).collect();
        let up_blocks: Vec<BlockQ8_0> = (0..total_blocks)
            .map(|i| synth_q80_block(i + 1000))
            .collect();
        let (gate_is, gate_mul) = row_q80_to_intscale(&gate_blocks);
        let (up_is, up_mul) = row_q80_to_intscale(&up_blocks);

        let mut row = Vec::with_capacity(n_units);
        for u in 0..n_units {
            let base = u * 8;
            let mut gate_arr: [Q80IntScale; 8] = [Q80IntScale {
                scale_int: 0,
                qs: [0; 32],
            }; 8];
            let mut up_arr: [Q80IntScale; 8] = [Q80IntScale {
                scale_int: 0,
                qs: [0; 32],
            }; 8];
            for k in 0..8 {
                gate_arr[k] = gate_is[base + k];
                up_arr[k] = up_is[base + k];
            }
            row.push(SharedGUQ8KUnit {
                gate_q8_0: gate_arr,
                up_q8_0: up_arr,
            });
        }
        (row, gate_mul, up_mul)
    }

    /// f32 reference dot over dequantized Q8_0 weights.
    /// Row covers n_units * 256 elems: 8 Q8_0 blocks per unit, 32 elems each.
    fn f32_reference_dot_q80(row: &[SharedGUQ8KUnit], row_mul: f32, gate: bool, x: &[f32]) -> f32 {
        assert_eq!(row.len() * 256, x.len());
        let mut acc = 0.0f64;
        for (ui, unit) in row.iter().enumerate() {
            for sub in 0..8 {
                let b = if gate {
                    &unit.gate_q8_0[sub]
                } else {
                    &unit.up_q8_0[sub]
                };
                let dq = dequantize_q80_intscale(b, row_mul);
                let x_base = ui * 256 + sub * 32;
                for l in 0..32 {
                    acc += (dq[l] as f64) * (x[x_base + l] as f64);
                }
            }
        }
        acc as f32
    }

    #[test]
    fn q80_gu_scalar_matches_float_reference_rel_err_under_1pct() {
        let n_units = 4;
        let (row, gate_mul, up_mul) = synth_shared_gu_row(n_units);
        let x_f32 = build_activation(n_units);
        let h_q8k = quantize_input_q8k(&x_f32);

        let ref_gate = f32_reference_dot_q80(&row, gate_mul, true, &x_f32);
        let ref_up = f32_reference_dot_q80(&row, up_mul, false, &x_f32);

        // Per-unit fold (mirrors caller shape — each unit has its own x.d).
        let mut gate_f = 0.0f64;
        let mut up_f = 0.0f64;
        for u in 0..n_units {
            let (gi, ui) = sdot_q80_gu_scalar(&row[u..u + 1], &h_q8k[u..u + 1]);
            gate_f += (gi as f64) * (gate_mul as f64) * (h_q8k[u].d as f64);
            up_f += (ui as f64) * (up_mul as f64) * (h_q8k[u].d as f64);
        }
        let gate_f = gate_f as f32;
        let up_f = up_f as f32;

        // Sanity: single-call accumulate must be non-zero.
        let (g_all, u_all) = sdot_q80_gu_scalar(&row, &h_q8k);
        assert!(g_all != 0, "gate accumulate unexpectedly zero");
        assert!(u_all != 0, "up accumulate unexpectedly zero");

        let rel_gate = ((gate_f - ref_gate) / ref_gate).abs();
        let rel_up = ((up_f - ref_up) / ref_up).abs();
        eprintln!(
            "q80_gu scalar: gate_f={gate_f:.4} ref={ref_gate:.4} rel={rel_gate:.5} | up_f={up_f:.4} ref={ref_up:.4} rel={rel_up:.5}"
        );
        assert!(rel_gate < 0.01, "gate rel err too high: {rel_gate}");
        assert!(rel_up < 0.01, "up rel err too high: {rel_up}");
    }

    /// NEON kernel must exactly match scalar oracle (i32 == i32).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q80_gu_neon_matches_scalar_exact() {
        let n_units = 4;
        let (row, _, _) = synth_shared_gu_row(n_units);
        let x_f32 = build_activation(n_units);
        let h_q8k = quantize_input_q8k(&x_f32);

        let scalar_res = sdot_q80_gu_scalar(&row, &h_q8k);
        let neon_res = unsafe { sdot_q80_gu_neon(&row, &h_q8k) };
        assert_eq!(
            scalar_res, neon_res,
            "q80_gu i32 mismatch: scalar={scalar_res:?} neon={neon_res:?}"
        );

        // Also verify per-unit equality — the caller folds x.d per-unit.
        for u in 0..n_units {
            let s = sdot_q80_gu_scalar(&row[u..u + 1], &h_q8k[u..u + 1]);
            let n = unsafe { sdot_q80_gu_neon(&row[u..u + 1], &h_q8k[u..u + 1]) };
            assert_eq!(
                s, n,
                "q80_gu per-unit mismatch at u={u}: scalar={s:?} neon={n:?}"
            );
        }
    }

    // ----------------------------------------------------------------------
    // Q8_0 row kernel tests (shared expert down)
    // ----------------------------------------------------------------------

    /// Build a flat Q80IntScale row covering `n_q8k_blocks * 256` elems
    /// (8 Q8_0 blocks per Q8K activation block). Returns (row, row_mul).
    fn synth_q80_down_row(n_q8k_blocks: usize) -> (Vec<Q80IntScale>, f32) {
        let total = n_q8k_blocks * 8;
        let blocks: Vec<BlockQ8_0> = (0..total).map(|i| synth_q80_block(i + 2000)).collect();
        row_q80_to_intscale(&blocks)
    }

    #[test]
    fn q80_row_scalar_matches_float_reference_rel_err_under_1pct() {
        let n_q8k = 4;
        let (row, row_mul) = synth_q80_down_row(n_q8k);

        let x_f: Vec<f32> = (0..n_q8k * 256)
            .map(|i| (i as f32 * 0.011).sin() * 0.3 + (i as f32 * 0.023).cos() * 0.1)
            .collect();
        let x_q8k = quantize_input_q8k(&x_f);

        // f32 reference: dequantize each Q8_0 block and dot with x_f.
        let mut w_f: Vec<f32> = Vec::with_capacity(x_f.len());
        for b in &row {
            let tmp = dequantize_q80_intscale(b, row_mul);
            w_f.extend_from_slice(&tmp);
        }
        let expected: f32 = w_f.iter().zip(&x_f).map(|(a, b)| a * b).sum();

        // Per-Q8K-block fold: x.d is per Q8K block.
        let mut got: f64 = 0.0;
        for u in 0..n_q8k {
            let row_1 = &row[u * 8..(u + 1) * 8];
            let x_1 = std::slice::from_ref(&x_q8k[u]);
            let acc_i64 = sdot_q80_row_scalar(row_1, x_1);
            got += (acc_i64 as f64) * (row_mul as f64) * (x_q8k[u].d as f64);
        }

        // Sanity: whole-row accumulate non-zero (fixture not degenerate).
        let acc_all = sdot_q80_row_scalar(&row, &x_q8k);
        assert!(acc_all != 0, "q80_row acc unexpectedly zero");

        let rel = ((got as f32 - expected) / expected).abs();
        eprintln!("q80_row scalar: got={got} ref={expected} rel={rel} acc_all={acc_all}");
        assert!(rel < 0.01, "q80_row rel err {rel} >= 1%");
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q80_row_neon_matches_scalar_exact() {
        let n_q8k = 4;
        let (row, _) = synth_q80_down_row(n_q8k);
        let x_f: Vec<f32> = (0..n_q8k * 256)
            .map(|i| (i as f32 * 0.011).sin() * 0.3 + (i as f32 * 0.023).cos() * 0.1)
            .collect();
        let x_q8k = quantize_input_q8k(&x_f);

        let scalar = sdot_q80_row_scalar(&row, &x_q8k);
        let neon = unsafe { sdot_q80_row_neon(&row, &x_q8k) };
        assert_eq!(
            scalar, neon,
            "q80_row i64 mismatch: scalar={scalar} neon={neon}"
        );

        // Per-Q8K-block equality too (caller folds x.d per block).
        for u in 0..n_q8k {
            let s = sdot_q80_row_scalar(&row[u * 8..(u + 1) * 8], std::slice::from_ref(&x_q8k[u]));
            let n = unsafe {
                sdot_q80_row_neon(&row[u * 8..(u + 1) * 8], std::slice::from_ref(&x_q8k[u]))
            };
            assert_eq!(
                s, n,
                "q80_row per-block mismatch at u={u}: scalar={s} neon={n}"
            );
        }
    }
}
