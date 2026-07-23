//! f32 → quantized block converters (reverse of `dequant.rs`).
//!
//! Session 71 HOBBIT pivot: we need to materialize **Q2_K shadow weights** for
//! MoE experts by requantizing already-Q4_K-dequantized tensors to Q2_K. The
//! shadow variant is consumed at runtime only when the token-level dispatcher
//! decides a given expert is non-critical enough to use a lower-precision copy.
//!
//! The Q2_K format is a clean-room re-derivation from the block layout +
//! dequantization formula (see `dequantize_q2_k`). The algorithm here is the
//! **naive per-sub-block range fit** — not iterative MSE-optimal like ggml's
//! `make_qkx2_quants`. Accuracy is acceptable for PoC; a refined variant can
//! replace this later without changing the on-disk format.
//!
//! Layout recap (84 bytes per super-block of 256 elements):
//!   scales[16] (4-bit scale | 4-bit min per sub-block of 16)
//!   qs[64]     (four 32-element groups interleaved by 2-bit plane)
//!   d     f16  (super-block scale of sub-scales)
//!   dmin  f16  (super-block scale of sub-mins)
//!
//! Dequant formula (from dequant.rs):
//!   out[idx] = q * (d * sc4) - (dmin * mn4)
//!   where sc4 = scales[j] & 0x0F, mn4 = scales[j] >> 4, q ∈ {0,1,2,3}

use half::f16;

/// Quantize a contiguous run of f32 samples into Q2_K blocks.
///
/// - `input.len()` must be a multiple of 256.
/// - `output.len()` must be `input.len() / 256 * 84`.
///
/// The algorithm per 256-element super-block:
///   1. For each of 16 sub-blocks (16 elements each):
///       * Compute `x_min`, `x_max` of the sub-block.
///       * If `x_min <= 0`: `scale_sub = (x_max - x_min) / 3`, `min_sub = -x_min`.
///         Both are non-negative → storable as 4-bit unsigned after super-scale.
///       * Else (all values positive): `scale_sub = x_max / 3`, `min_sub = 0`.
///         (2-bit precision is coarse; this branch loses minor accuracy vs a
///         shifted representation, but keeps `min_sub >= 0` strictly.)
///       * Quantize each element: `q = round((x[l] - x_min) / scale_sub).clamp(0, 3)`.
///   2. Super-block scales `d = max(scale_sub) / 15`, `dmin = max(min_sub) / 15`.
///      Per-sub-block 4-bit codes: `sc4 = round(scale_sub / d)`, `mn4 = round(min_sub / dmin)`.
///   3. Pack: `scales[j] = (mn4 << 4) | sc4`; each 128-element half interleaves
///      four 32-element quant groups in the canonical GGUF Q2_K layout.
pub fn quantize_row_q2_k(input: &[f32], output: &mut [u8]) {
    assert_eq!(
        input.len() % 256,
        0,
        "Q2_K quantize: input len must be multiple of 256 (got {})",
        input.len()
    );
    let n_super = input.len() / 256;
    assert_eq!(
        output.len(),
        n_super * 84,
        "Q2_K quantize: output len mismatch"
    );

    const EPS: f32 = 1e-10;

    for sb in 0..n_super {
        let x = &input[sb * 256..(sb + 1) * 256];
        let out = &mut output[sb * 84..(sb + 1) * 84];

        let mut sub_scales = [0f32; 16];
        let mut sub_mins = [0f32; 16];
        let mut q_values = [0u8; 256];

        // Step 1 + 2: sub-block range fit → 2-bit quants + real scale/min.
        for j in 0..16 {
            let xs = &x[j * 16..j * 16 + 16];
            let mut x_min = xs[0];
            let mut x_max = xs[0];
            for &v in xs.iter().skip(1) {
                if v < x_min {
                    x_min = v;
                }
                if v > x_max {
                    x_max = v;
                }
            }

            let (scale_sub, min_sub) = if x_min <= 0.0 {
                let range = x_max - x_min;
                let scale = if range > EPS { range / 3.0 } else { 0.0 };
                (scale, -x_min)
            } else {
                // All positive. Use pure-scale representation (min = 0).
                let scale = if x_max > EPS { x_max / 3.0 } else { 0.0 };
                (scale, 0.0)
            };

            sub_scales[j] = scale_sub;
            sub_mins[j] = min_sub;

            if scale_sub > EPS {
                // q = (x - x_min) / scale_sub  (when x_min <= 0, x_min = -min_sub)
                // If x_min > 0: we shifted min_sub=0 so x_min effectively becomes 0.
                let effective_min = if x_min <= 0.0 { x_min } else { 0.0 };
                for l in 0..16 {
                    let raw = (xs[l] - effective_min) / scale_sub;
                    let qv = raw.round().clamp(0.0, 3.0) as u8;
                    q_values[j * 16 + l] = qv;
                }
            }
            // else: all q_values already 0, which is what we want.
        }

        // Step 3: super-block scale / dmin quantize (4-bit per sub-block).
        let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

        let d = if max_scale > EPS {
            max_scale / 15.0
        } else {
            0.0
        };
        let dmin = if max_min > EPS { max_min / 15.0 } else { 0.0 };

        let mut scales_packed = [0u8; 16];
        for j in 0..16 {
            let sc4 = if d > EPS {
                (sub_scales[j] / d).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            let mn4 = if dmin > EPS {
                (sub_mins[j] / dmin).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            scales_packed[j] = (mn4 << 4) | (sc4 & 0x0F);
        }

        // Canonical GGUF Q2_K packing. Each byte stores the same lane from
        // four 32-element groups in one 128-element half.
        let mut qs_packed = [0u8; 64];
        for idx in 0..256 {
            let half = idx / 128;
            let within_half = idx % 128;
            let byte_idx = half * 32 + within_half % 32;
            let shift = (within_half / 32) * 2;
            qs_packed[byte_idx] |= (q_values[idx] & 3) << shift;
        }

        // Emit bytes in BlockQ2_K layout: scales(16) + qs(64) + d(f16) + dmin(f16)
        out[0..16].copy_from_slice(&scales_packed);
        out[16..80].copy_from_slice(&qs_packed);
        let d_f16 = f16::from_f32(d);
        let dmin_f16 = f16::from_f32(dmin);
        out[80..82].copy_from_slice(&d_f16.to_le_bytes());
        out[82..84].copy_from_slice(&dmin_f16.to_le_bytes());
    }
}

/// Convenience: quantize into a freshly-allocated `Vec<u8>`.
pub fn quantize_q2_k_vec(input: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; input.len() / 256 * 84];
    quantize_row_q2_k(input, &mut out);
    out
}

// =============================================================================
// Q5_K quantization — ported from llama.cpp `quantize_row_q5_K_ref`
// (ggml/src/ggml-quants.c ~line 1582). The "ref" variant runs the standard
// per-sub-block `make_qkx2_quants` fit without external quant-weights, which
// is exactly what we need for offline repack (e.g. Q5_1 → f32 → Q5_K).
// =============================================================================

const QK_K: usize = 256;

/// `nearest_int` (bit-twiddling round-to-nearest; matches llama.cpp exactly).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let bits = val.to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Port of llama.cpp `make_qkx2_quants` — min-max-and-refine optimizer used by
/// the reference Q4_K/Q5_K quantizers. Returns the found scale and writes the
/// quantized codes into `L[0..n]` and `-min` into `the_min`.
///
/// Notes:
/// - `nmax` is the maximum quant value (31 for Q5_K, 15 for Q4_K).
/// - `weights[i]` are per-element weights (set to `av_x + |x[i]|` in Q5_K-ref).
/// - `use_mad = false` for Q5_K-ref (L2 error).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        let li = li.max(0).min(nmax) as u8;
        l[i] = li;
        let mut diff = scale * li as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min));
            let li = li.max(0).min(nmax) as u8;
            laux[i] = li;
            let w = weights[i];
            sum_l += w * li as f32;
            sum_l2 += w * (li as f32) * (li as f32);
            sum_xl += w * (li as f32) * x[i];
        }
        let d_det = sum_w * sum_l2 - sum_l * sum_l;
        if d_det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Quantize a contiguous run of f32 samples into Q4_K blocks.
///
/// - `input.len()` must be a multiple of 256 (QK_K).
/// - `output.len()` must be `input.len() / 256 * size_of::<BlockQ4_K>()` (= 144/super-block).
///
/// This mirrors the Q5_K reference quantizer below, but uses 4-bit per-element
/// quants and omits the Q5 high-bit plane.
pub fn quantize_row_q4_k(input: &[f32], output: &mut [u8]) {
    use crate::quantize::blocks::BlockQ4_K;
    use std::mem::size_of;

    assert_eq!(
        input.len() % QK_K,
        0,
        "Q4_K quantize: input len must be multiple of 256 (got {})",
        input.len()
    );
    let nb = input.len() / QK_K;
    assert_eq!(
        output.len(),
        nb * size_of::<BlockQ4_K>(),
        "Q4_K quantize: output len mismatch"
    );

    let mut el = [0u8; QK_K];
    let mut mins = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];
    let mut weights = [0f32; 32];
    let mut laux = [0u8; 32];

    for i in 0..nb {
        let x_base = i * QK_K;
        let x = &input[x_base..x_base + QK_K];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..(QK_K / 32) {
            let xs = &x[32 * j..32 * j + 32];

            let mut sum_x2 = 0.0f32;
            for l in 0..32 {
                sum_x2 += xs[l] * xs[l];
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 {
                weights[l] = av_x + xs[l].abs();
            }

            let sub_l_slice = &mut el[32 * j..32 * j + 32];
            let mut sub_min = 0.0f32;
            let s = make_qkx2_quants(
                32,
                15,
                xs,
                &weights,
                sub_l_slice,
                &mut sub_min,
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            scales[j] = s;
            mins[j] = sub_min;
            if s > max_scale {
                max_scale = s;
            }
            if sub_min > max_min {
                max_min = sub_min;
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        let mut y_scales = [0u8; 12];
        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).max(0).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).max(0).min(63) as u8;
            if j < 4 {
                y_scales[j] = ls;
                y_scales[j + 4] = lm;
            } else {
                y_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y_scales[j - 4] |= (ls >> 4) << 6;
                y_scales[j] |= (lm >> 4) << 6;
            }
        }

        let y_d = max_scale / 63.0;
        let y_dmin = max_min / 63.0;
        let y_d_f16 = f16::from_f32(y_d);
        let y_dmin_f16 = f16::from_f32(y_dmin);
        let y_d_round = y_d_f16.to_f32();
        let y_dmin_round = y_dmin_f16.to_f32();

        for j in 0..(QK_K / 32) {
            let (sc, mn) = get_scale_min_k4(j, &y_scales);
            let d_sub = y_d_round * sc as f32;
            if d_sub == 0.0 {
                continue;
            }
            let dm_sub = y_dmin_round * mn as f32;
            for ii in 0..32 {
                let li = nearest_int((x[32 * j + ii] + dm_sub) / d_sub);
                el[32 * j + ii] = li.max(0).min(15) as u8;
            }
        }

        let mut qs = [0u8; QK_K / 2];
        let mut ql_off = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let l1 = el[n + j] & 0x0F;
                let l2 = el[n + j + 32] & 0x0F;
                qs[ql_off + j] = l1 | (l2 << 4);
            }
            ql_off += 32;
        }

        let out = &mut output[i * size_of::<BlockQ4_K>()..(i + 1) * size_of::<BlockQ4_K>()];
        out[0..2].copy_from_slice(&y_d_f16.to_le_bytes());
        out[2..4].copy_from_slice(&y_dmin_f16.to_le_bytes());
        out[4..16].copy_from_slice(&y_scales);
        out[16..144].copy_from_slice(&qs);
    }
}

/// Convenience: quantize into a freshly-allocated `Vec<u8>`.
pub fn quantize_q4_k_vec(input: &[f32]) -> Vec<u8> {
    use crate::quantize::blocks::BlockQ4_K;
    use std::mem::size_of;
    let mut out = vec![0u8; input.len() / 256 * size_of::<BlockQ4_K>()];
    quantize_row_q4_k(input, &mut out);
    out
}

/// Quantize a contiguous run of f32 samples into Q5_K blocks.
///
/// - `input.len()` must be a multiple of 256 (QK_K).
/// - `output.len()` must be `input.len() / 256 * size_of::<BlockQ5_K>()` (= 176/super-block).
///
/// Port of llama.cpp `quantize_row_q5_K_ref`: 8 sub-blocks × 32 elems each,
/// per-sub-block `make_qkx2_quants(nmax=31, rmin=-0.5, rdelta=0.1, nstep=15)`,
/// 6-bit packed sub-block scales/mins, 5-bit per-element quants split into
/// `qs` (low 4 bits) + `qh` (high 1 bit).
pub fn quantize_row_q5_k(input: &[f32], output: &mut [u8]) {
    use crate::quantize::blocks::BlockQ5_K;
    use std::mem::size_of;

    assert_eq!(
        input.len() % QK_K,
        0,
        "Q5_K quantize: input len must be multiple of 256 (got {})",
        input.len()
    );
    let nb = input.len() / QK_K;
    assert_eq!(
        output.len(),
        nb * size_of::<BlockQ5_K>(),
        "Q5_K quantize: output len mismatch"
    );

    // Scratch buffers reused per super-block.
    let mut el = [0u8; QK_K]; // per-element 5-bit codes
    let mut mins = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];
    let mut weights = [0f32; 32];
    let mut laux = [0u8; 32];

    for i in 0..nb {
        let x_base = i * QK_K;
        let x = &input[x_base..x_base + QK_K];

        // Per-sub-block fit.
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..(QK_K / 32) {
            let xs = &x[32 * j..32 * j + 32];

            let mut sum_x2 = 0.0f32;
            for l in 0..32 {
                sum_x2 += xs[l] * xs[l];
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 {
                weights[l] = av_x + xs[l].abs();
            }

            let sub_l_slice = &mut el[32 * j..32 * j + 32];
            let mut sub_min = 0.0f32;
            let s = make_qkx2_quants(
                32,
                31,
                xs,
                &weights,
                sub_l_slice,
                &mut sub_min,
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            scales[j] = s;
            mins[j] = sub_min;
            if s > max_scale {
                max_scale = s;
            }
            if sub_min > max_min {
                max_min = sub_min;
            }
        }

        // Pack sub-block 6-bit codes into `scales[12]`.
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        let mut y_scales = [0u8; 12];
        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).max(0).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).max(0).min(63) as u8;
            if j < 4 {
                y_scales[j] = ls;
                y_scales[j + 4] = lm;
            } else {
                y_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y_scales[j - 4] |= (ls >> 4) << 6;
                y_scales[j] |= (lm >> 4) << 6;
            }
        }

        let y_d = max_scale / 63.0;
        let y_dmin = max_min / 63.0;

        // Requantize per-element using packed sub-block scales.
        // Mirrors llama.cpp: get_scale_min_k4 → d = fp16(d) * sc, dm = fp16(dmin) * m.
        // We apply the same f16 round-trip to match dequant exactly.
        let y_d_f16 = f16::from_f32(y_d);
        let y_dmin_f16 = f16::from_f32(y_dmin);
        let y_d_round = y_d_f16.to_f32();
        let y_dmin_round = y_dmin_f16.to_f32();

        for j in 0..(QK_K / 32) {
            let (sc, mn) = get_scale_min_k4(j, &y_scales);
            let d_sub = y_d_round * sc as f32;
            if d_sub == 0.0 {
                continue;
            }
            let dm_sub = y_dmin_round * mn as f32;
            for ii in 0..32 {
                let li = nearest_int((x[32 * j + ii] + dm_sub) / d_sub);
                let li = li.max(0).min(31) as u8;
                el[32 * j + ii] = li;
            }
        }

        // Pack qs (low 4 bits) + qh (bit 4).
        let mut qh = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 2];
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        let mut ql_off = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = el[n + j] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = el[n + j + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[ql_off + j] = (l1 as u8) | ((l2 as u8) << 4);
            }
            m1 = m1.wrapping_shl(2);
            m2 = m2.wrapping_shl(2);
            ql_off += 32;
        }

        // Emit BlockQ5_K: d, dmin, scales[12], qh[32], qs[128].
        let out = &mut output[i * size_of::<BlockQ5_K>()..(i + 1) * size_of::<BlockQ5_K>()];
        out[0..2].copy_from_slice(&y_d_f16.to_le_bytes());
        out[2..4].copy_from_slice(&y_dmin_f16.to_le_bytes());
        out[4..16].copy_from_slice(&y_scales);
        out[16..48].copy_from_slice(&qh);
        out[48..176].copy_from_slice(&qs);
    }
}

/// Unpack a single sub-block 6-bit scale/min pair (llama.cpp `get_scale_min_k4`).
///
/// Shared with `moe_convert.rs` so Q4_K / Q5_K IntScale dequant paths can reuse
/// the same canonical sub-block scale unpack logic. Also used by rnb-llm NEON
/// MoE decode kernels (`sdot_q4k_gu_*`) to unpack sub-block scales directly from
/// `Q4KIntScale.sub_scales` — the canonical k-quant 6-bit unpacker, no hidden
/// invariants to guard.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Convenience: quantize into a freshly-allocated `Vec<u8>`.
pub fn quantize_q5_k_vec(input: &[f32]) -> Vec<u8> {
    use crate::quantize::blocks::BlockQ5_K;
    use std::mem::size_of;
    let mut out = vec![0u8; input.len() / 256 * size_of::<BlockQ5_K>()];
    quantize_row_q5_k(input, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::super::blocks::BlockQ2_K;
    use super::super::dequant::dequantize_q2_k;
    use super::*;
    use std::mem::size_of;

    fn cos_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32
    }

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y) as f64;
            s += d * d;
        }
        (s / a.len() as f64) as f32
    }

    #[test]
    fn test_q2k_output_size() {
        let input = vec![0.0f32; 256];
        let out = quantize_q2_k_vec(&input);
        assert_eq!(out.len(), 84);
    }

    #[test]
    fn test_q2k_block_struct_size_matches() {
        assert_eq!(size_of::<BlockQ2_K>(), 84);
    }

    #[test]
    fn test_q2k_roundtrip_zero() {
        let input = vec![0.0f32; 256];
        let mut bytes = vec![0u8; 84];
        quantize_row_q2_k(&input, &mut bytes);
        // SAFETY: bytes is 84 aligned to BlockQ2_K layout
        let block = unsafe { &*(bytes.as_ptr() as *const BlockQ2_K) };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(block, &mut out);
        for &v in out.iter() {
            assert!(v.abs() < 1e-6, "zero input should dequant to ~0, got {v}");
        }
    }

    #[test]
    fn test_q2k_roundtrip_constant_positive() {
        let c = 0.7f32;
        let input = vec![c; 256];
        let mut bytes = vec![0u8; 84];
        quantize_row_q2_k(&input, &mut bytes);
        let block = unsafe { &*(bytes.as_ptr() as *const BlockQ2_K) };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(block, &mut out);
        // Constant positive → x_min = x_max = c > 0, scale = c/3, min = 0.
        // q = round(c / (c/3)) = 3, dequant = 3 * c/3 = c (approx).
        let err: f32 = out.iter().map(|&v| (v - c).abs()).sum::<f32>() / 256.0;
        assert!(err < 0.02, "avg err too large: {err}");
    }

    #[test]
    fn test_q2k_roundtrip_random_normal() {
        // pseudo-random f32 samples (reproducible)
        let mut state: u64 = 0xC0FFEE12345;
        let mut input = vec![0.0f32; 256];
        for v in input.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map top 24 bits → [-1, 1)
            let u = (state >> 40) as u32;
            *v = (u as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        }

        let mut bytes = vec![0u8; 84];
        quantize_row_q2_k(&input, &mut bytes);
        let block = unsafe { &*(bytes.as_ptr() as *const BlockQ2_K) };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(block, &mut out);

        let cos = cos_similarity(&input, &out);
        let err = mse(&input, &out);
        // Q2_K is very coarse (2 bits per element). Expect cos ~0.9+ for
        // uniformly-distributed data and low MSE relative to variance (≈1/3).
        assert!(cos > 0.90, "cos similarity too low: {cos}, mse={err}");
    }

    #[test]
    fn test_q2k_roundtrip_gaussian_like() {
        // Sum of 4 uniforms → rough normal, concentrated around 0.
        let mut state: u64 = 0xDEADBEEFCAFEFEED;
        let mut input = vec![0.0f32; 256];
        for v in input.iter_mut() {
            let mut s = 0.0f32;
            for _ in 0..4 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (state >> 40) as u32;
                s += u as f32 / (1u32 << 24) as f32;
            }
            *v = (s - 2.0) * 0.5; // zero-centered, small variance
        }

        let mut bytes = vec![0u8; 84];
        quantize_row_q2_k(&input, &mut bytes);
        let block = unsafe { &*(bytes.as_ptr() as *const BlockQ2_K) };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(block, &mut out);

        let cos = cos_similarity(&input, &out);
        assert!(cos > 0.90, "gaussian-like cos too low: {cos}");
    }

    #[test]
    fn q5k_quantize_row_roundtrip() {
        use crate::quantize::blocks::BlockQ5_K;
        use crate::quantize::dequant::dequantize_q5_k;

        // 1024-elem (4 super-blocks) of varied magnitudes.
        let input: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 * 0.019;
                t.sin() * (1.0 + 0.3 * (t * 0.7).cos())
            })
            .collect();

        let mut bytes = vec![0u8; (input.len() / 256) * std::mem::size_of::<BlockQ5_K>()];
        quantize_row_q5_k(&input, &mut bytes);

        let n_blocks = input.len() / 256;
        let mut reconstructed = Vec::with_capacity(input.len());
        for bi in 0..n_blocks {
            let offset = bi * std::mem::size_of::<BlockQ5_K>();
            let block: BlockQ5_K =
                unsafe { std::ptr::read(bytes.as_ptr().add(offset) as *const BlockQ5_K) };
            let mut tmp = [0.0f32; 256];
            dequantize_q5_k(&block, &mut tmp);
            reconstructed.extend_from_slice(&tmp);
        }

        let dot: f32 = input.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let na: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb + 1e-10);
        eprintln!("Q5_K quantize_row roundtrip cos = {}", cos);
        assert!(cos >= 0.98, "Q5_K quant roundtrip cos too low: {}", cos);
    }

    #[test]
    fn q4k_quantize_row_roundtrip() {
        use crate::quantize::blocks::BlockQ4_K;
        use crate::quantize::dequant::dequantize_q4_k;

        let input: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 * 0.017;
                t.sin() * (1.0 + 0.25 * (t * 0.9).cos())
            })
            .collect();

        let mut bytes = vec![0u8; (input.len() / 256) * std::mem::size_of::<BlockQ4_K>()];
        quantize_row_q4_k(&input, &mut bytes);

        let n_blocks = input.len() / 256;
        let mut reconstructed = Vec::with_capacity(input.len());
        for bi in 0..n_blocks {
            let offset = bi * std::mem::size_of::<BlockQ4_K>();
            let block: BlockQ4_K =
                unsafe { std::ptr::read(bytes.as_ptr().add(offset) as *const BlockQ4_K) };
            let mut tmp = [0.0f32; 256];
            dequantize_q4_k(&block, &mut tmp);
            reconstructed.extend_from_slice(&tmp);
        }

        let dot: f32 = input.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let na: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb + 1e-10);
        eprintln!("Q4_K quantize_row roundtrip cos = {}", cos);
        assert!(cos >= 0.97, "Q4_K quant roundtrip cos too low: {}", cos);
    }

    #[test]
    fn test_q2k_multi_block() {
        let n_super = 5;
        let mut input = vec![0.0f32; n_super * 256];
        let mut state: u64 = 0xABCDEF0123456789;
        for v in input.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 40) as u32;
            *v = (u as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        }

        let bytes = quantize_q2_k_vec(&input);
        assert_eq!(bytes.len(), n_super * 84);

        let mut out = vec![0.0f32; n_super * 256];
        for sb in 0..n_super {
            let block = unsafe { &*(bytes.as_ptr().add(sb * 84) as *const BlockQ2_K) };
            let mut tmp = [0.0f32; 256];
            dequantize_q2_k(block, &mut tmp);
            out[sb * 256..(sb + 1) * 256].copy_from_slice(&tmp);
        }

        let cos = cos_similarity(&input, &out);
        assert!(cos > 0.90, "multi-block cos too low: {cos}");
    }
}
