use super::blocks::*;

// =============================================================================
// Task 3: Basic Dequantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
// =============================================================================

pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    for i in 0..16 {
        let byte = block.qs[i];
        output[i] = ((byte & 0x0F) as f32 - 8.0) * d;
        output[i + 16] = ((byte >> 4) as f32 - 8.0) * d;
    }
}

pub fn dequantize_q4_1(block: &BlockQ4_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    for i in 0..16 {
        let byte = block.qs[i];
        output[i] = (byte & 0x0F) as f32 * d + m;
        output[i + 16] = (byte >> 4) as f32 * d + m;
    }
}

pub fn dequantize_q5_0(block: &BlockQ5_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let qh = u32::from_le_bytes(block.qh);
    for i in 0..16 {
        let byte = block.qs[i];
        let xh_0 = ((qh >> i) & 1) as u8;
        let xh_1 = ((qh >> (i + 16)) & 1) as u8;
        output[i] = ((byte & 0x0F) | (xh_0 << 4)) as f32 - 16.0;
        output[i + 16] = ((byte >> 4) | (xh_1 << 4)) as f32 - 16.0;
        output[i] *= d;
        output[i + 16] *= d;
    }
}

pub fn dequantize_q5_1(block: &BlockQ5_1, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let qh = u32::from_le_bytes(block.qh);
    for i in 0..16 {
        let byte = block.qs[i];
        let xh_0 = ((qh >> i) & 1) as u8;
        let xh_1 = ((qh >> (i + 16)) & 1) as u8;
        let q0 = ((byte & 0x0F) | (xh_0 << 4)) as f32;
        let q1 = ((byte >> 4) | (xh_1 << 4)) as f32;
        output[i] = q0 * d + m;
        output[i + 16] = q1 * d + m;
    }
}

pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32; 32]) {
    let d = block.d.to_f32();
    for (i, out) in output.iter_mut().enumerate().take(32) {
        *out = block.qs[i] as f32 * d;
    }
}

// =============================================================================
// Task 4: K-Quant Dequantization (Q2_K ~ Q6_K)
// =============================================================================

pub fn dequantize_q2_k(block: &BlockQ2_K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // GGUF Q2_K interleaves four 32-element groups in each 128-element half:
    // qs[l] = q[l] | q[l + 32] << 2 | q[l + 64] << 4 | q[l + 96] << 6.
    for j in 0..16 {
        let sc = block.scales[j];
        let scale = d * (sc & 0x0F) as f32;
        let min = dmin * (sc >> 4) as f32;
        let half = j / 8;
        let group = j % 8;
        let q_base = half * 32 + (group % 2) * 16;
        let shift = (group / 2) * 2;

        for l in 0..16 {
            let q = ((block.qs[q_base + l] >> shift) & 3) as f32;
            output[j * 16 + l] = q * scale - min;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_q2k_quant_f32(qs: *const u8, shift: usize) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;

    let packed = vld1q_u8(qs);
    let shifted = vshlq_u8(packed, vdupq_n_s8(-(shift as i8)));
    let quant = vandq_u8(shifted, vdupq_n_u8(3));
    let low = vmovl_u8(vget_low_u8(quant));
    let high = vmovl_u8(vget_high_u8(quant));
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(low))),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(high))),
    ]
}

/// Fused Q2_K dequant + f32 dot. Processes one canonical GGUF Q2_K
/// 256-element super-block against 256 f32 inputs.
///
/// Each 128-element half stores four 32-element quant groups interleaved by
/// bit plane. A 16-element scale group therefore reads 16 consecutive bytes
/// with one shared shift rather than unpacking four consecutive values from
/// each byte.
///
/// # Safety
/// `x` must point to at least 256 valid f32 elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_q2k_fused_neon(block: &BlockQ2_K, x: *const f32) -> f32 {
    use std::arch::aarch64::*;

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for j in 0..16 {
        let sc = block.scales[j];
        let scale = d * (sc & 0x0F) as f32;
        let min = dmin * (sc >> 4) as f32;
        let half = j / 8;
        let group = j % 8;
        let q_base = half * 32 + (group % 2) * 16;
        let shift = (group / 2) * 2;

        let q = load_q2k_quant_f32(block.qs.as_ptr().add(q_base), shift);
        let scale_v = vdupq_n_f32(scale);
        let neg_min_v = vdupq_n_f32(-min);
        let w0 = vfmaq_f32(neg_min_v, q[0], scale_v);
        let w1 = vfmaq_f32(neg_min_v, q[1], scale_v);
        let w2 = vfmaq_f32(neg_min_v, q[2], scale_v);
        let w3 = vfmaq_f32(neg_min_v, q[3], scale_v);

        let xp = x.add(j * 16);
        let x0 = vld1q_f32(xp);
        let x1 = vld1q_f32(xp.add(4));
        let x2 = vld1q_f32(xp.add(8));
        let x3 = vld1q_f32(xp.add(12));

        // Two-way split matching `dot_q4k_fused_neon` horizontal-sum style.
        acc0 = vfmaq_f32(acc0, w0, x0);
        acc1 = vfmaq_f32(acc1, w1, x1);
        acc0 = vfmaq_f32(acc0, w2, x2);
        acc1 = vfmaq_f32(acc1, w3, x3);
    }

    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// Paired Q2_K fused dequant + f32 dot for one 256-element block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_q2k_fused_neon_pair(
    block0: &BlockQ2_K,
    block1: &BlockQ2_K,
    x: *const f32,
) -> [f32; 2] {
    use std::arch::aarch64::*;

    let d0 = block0.d.to_f32();
    let dmin0 = block0.dmin.to_f32();
    let d1 = block1.d.to_f32();
    let dmin1 = block1.dmin.to_f32();

    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    for j in 0..16 {
        let sc0 = block0.scales[j];
        let scale0_v = vdupq_n_f32(d0 * (sc0 & 0x0F) as f32);
        let neg_min0_v = vdupq_n_f32(-(dmin0 * (sc0 >> 4) as f32));

        let sc1 = block1.scales[j];
        let scale1_v = vdupq_n_f32(d1 * (sc1 & 0x0F) as f32);
        let neg_min1_v = vdupq_n_f32(-(dmin1 * (sc1 >> 4) as f32));

        let half = j / 8;
        let group = j % 8;
        let q_base = half * 32 + (group % 2) * 16;
        let shift = (group / 2) * 2;
        let q0 = load_q2k_quant_f32(block0.qs.as_ptr().add(q_base), shift);
        let q1 = load_q2k_quant_f32(block1.qs.as_ptr().add(q_base), shift);

        let w0_0 = vfmaq_f32(neg_min0_v, q0[0], scale0_v);
        let w0_1 = vfmaq_f32(neg_min0_v, q0[1], scale0_v);
        let w0_2 = vfmaq_f32(neg_min0_v, q0[2], scale0_v);
        let w0_3 = vfmaq_f32(neg_min0_v, q0[3], scale0_v);
        let w1_0 = vfmaq_f32(neg_min1_v, q1[0], scale1_v);
        let w1_1 = vfmaq_f32(neg_min1_v, q1[1], scale1_v);
        let w1_2 = vfmaq_f32(neg_min1_v, q1[2], scale1_v);
        let w1_3 = vfmaq_f32(neg_min1_v, q1[3], scale1_v);

        let xp = x.add(j * 16);
        let x0 = vld1q_f32(xp);
        let x1 = vld1q_f32(xp.add(4));
        let x2 = vld1q_f32(xp.add(8));
        let x3 = vld1q_f32(xp.add(12));

        acc00 = vfmaq_f32(acc00, w0_0, x0);
        acc01 = vfmaq_f32(acc01, w0_1, x1);
        acc00 = vfmaq_f32(acc00, w0_2, x2);
        acc01 = vfmaq_f32(acc01, w0_3, x3);

        acc10 = vfmaq_f32(acc10, w1_0, x0);
        acc11 = vfmaq_f32(acc11, w1_1, x1);
        acc10 = vfmaq_f32(acc10, w1_2, x2);
        acc11 = vfmaq_f32(acc11, w1_3, x3);
    }

    [
        vaddvq_f32(vaddq_f32(acc00, acc01)),
        vaddvq_f32(vaddq_f32(acc10, acc11)),
    ]
}

/// Q3_K dequantization — matches llama.cpp dequantize_row_q3_K exactly.
///
/// 256 elements, 16 sub-blocks of 16 elements each.
/// Scales: 16 six-bit values packed in 12 bytes via uint32 bit manipulation.
/// qs: 2 bits per element, 4 elements per byte, shift increases per j iteration.
/// hmask: 1 high bit per element, bit position m advances per j iteration.
pub fn dequantize_q3_k(block: &BlockQ3_K, output: &mut [f32; 256]) {
    let d_all = block.d.to_f32();

    // Extract 16 six-bit scales from 12 bytes (llama.cpp bit layout)
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;

    let sb = &block.scales;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
    aux[1] = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
    aux[2] = u32::from_le_bytes([sb[8], sb[9], sb[10], sb[11]]);

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    // Reinterpret aux as 16 i8 scales (little-endian byte order)
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        let bytes = a.to_le_bytes();
        for (j, &b) in bytes.iter().enumerate() {
            scales[i * 4 + j] = b as i8;
        }
    }

    let q = &block.qs;
    let hm = &block.hmask;
    let mut is = 0usize;
    let mut m: u8 = 1;
    let mut y_idx = 0;
    let mut q_off = 0;

    for _ in 0..2 {
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16usize {
                let qv = (q[q_off + l] >> shift) & 3;
                let hv: i32 = if hm[l] & m != 0 { 0 } else { 4 };
                output[y_idx] = dl * (qv as i32 - hv) as f32;
                y_idx += 1;
            }

            let dl = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16usize {
                let qv = (q[q_off + l + 16] >> shift) & 3;
                let hv: i32 = if hm[l + 16] & m != 0 { 0 } else { 4 };
                output[y_idx] = dl * (qv as i32 - hv) as f32;
                y_idx += 1;
            }

            shift += 2;
            m = m.wrapping_shl(1);
        }
        q_off += 32;
    }
}

// Q4_K: 4-bit quants, 256 elements, 8 sub-blocks of 32
// scales[12]: 6-bit packed sub-block scales and mins
fn extract_k_quant_scales(scales_bytes: &[u8; 12]) -> ([f32; 8], [f32; 8]) {
    // 8 sub-block scales and 8 sub-block mins, each 6 bits
    // Packing layout (from llama.cpp):
    //   byte 0: scale0[5:0]
    //   byte 1: scale1[5:0]
    //   byte 2: scale2[5:0]
    //   byte 3: scale3[5:0]
    //   byte 4: scale4[5:0]
    //   byte 5: scale5[5:0]
    //   byte 6: scale6[5:0]
    //   byte 7: scale7[5:0]
    //   byte 8: min0[5:0]   (lower 6 bits)
    //   byte 9: min1[5:0]
    //  byte 10: min2[5:0]
    //  byte 11: min3[5:0]
    // But actually ggml uses a more compact scheme — let's use the correct layout:
    // scales[0..3]: lower 6 bits = scale0..3, upper 2 bits = high bits of min0..3
    // scales[4..7]: lower 6 bits = scale4..7, upper 2 bits = high bits of min4..7
    // scales[8..11]: bits packed for min values
    //
    // From ggml-quants.c dequantize_row_q4_K:
    //   uint8_t sc, m;
    //   get_scale_min_k4(j, x->scales, &sc, &m);
    // get_scale_min_k4:
    //   if (j < 4) { d = x[j] & 63; m = x[j+4] & 63; }
    //   else { d = (x[j+4] & 0xF) | ((x[j-4] >> 6) << 4);
    //           m = (x[j+4] >> 4) | ((x[j-0] >> 6) << 4); }

    let mut sc_out = [0f32; 8];
    let mut min_out = [0f32; 8];

    for j in 0usize..8 {
        let (sc, m) = if j < 4 {
            (scales_bytes[j] & 63, scales_bytes[j + 4] & 63)
        } else {
            let sc = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
            let m = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
            (sc, m)
        };
        sc_out[j] = sc as f32;
        min_out[j] = m as f32;
    }

    (sc_out, min_out)
}

/// Q4_K dequantization — matches llama.cpp dequantize_row_q4_K exactly.
///
/// 256 elements processed in 4 groups of 64. Each group uses 32 bytes of qs:
///   first 32 elements = low nibbles (scale is+0)
///   next 32 elements = high nibbles (scale is+1)
pub fn dequantize_q4_k(block: &BlockQ4_K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let (scales, mins) = extract_k_quant_scales(&block.scales);
    let q = &block.qs;

    let mut is = 0;
    let mut q_off = 0;
    let mut y_off = 0;

    for _ in 0..4 {
        let d1 = d * scales[is];
        let m1 = dmin * mins[is];
        let d2 = d * scales[is + 1];
        let m2 = dmin * mins[is + 1];

        for l in 0..32 {
            output[y_off + l] = d1 * (q[q_off + l] & 0xF) as f32 - m1;
        }
        for l in 0..32 {
            output[y_off + 32 + l] = d2 * (q[q_off + l] >> 4) as f32 - m2;
        }

        q_off += 32;
        is += 2;
        y_off += 64;
    }
}

/// Q4_K dequant NEON variant — bit-identical to `dequantize_q4_k`.
///
/// Same IEEE op order (u8→f32, mul, sub) as the scalar loop. No FMA; the MoE
/// router softmax argmax is sensitive to float-add non-associativity and C v1
/// regression in Session 60 showed FMA reorders diverge the sampled token.
/// Keeping two separate rounding steps (vmulq_f32 then vsubq_f32) preserves
/// bit equality with the scalar reference in `dequantize_q4_k`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dequantize_q4_k_neon(block: &BlockQ4_K, output: &mut [f32; 256]) {
    use std::arch::aarch64::*;

    // Inline the f32x16 store helper by macro to avoid nested-fn call overhead.
    // A nested `unsafe fn` does *not* inherit the parent's `#[target_feature]`
    // and `#[inline(always)]` on it isn't enough on all rustc versions — with
    // 3.7M block calls per decode step the overhead dwarfs the compute.
    macro_rules! store_dequant_x16 {
        ($dst:expr, $nibbles:expr, $dv:expr, $mv:expr) => {{
            let nibbles: uint8x16_t = $nibbles;
            let dv: float32x4_t = $dv;
            let mv: float32x4_t = $mv;
            let u16_lo = vmovl_u8(vget_low_u8(nibbles));
            let u16_hi = vmovl_u8(vget_high_u8(nibbles));

            let u32_0 = vmovl_u16(vget_low_u16(u16_lo));
            let u32_1 = vmovl_u16(vget_high_u16(u16_lo));
            let u32_2 = vmovl_u16(vget_low_u16(u16_hi));
            let u32_3 = vmovl_u16(vget_high_u16(u16_hi));

            let f0 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_0), dv), mv);
            let f1 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_1), dv), mv);
            let f2 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_2), dv), mv);
            let f3 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_3), dv), mv);

            let dst: *mut f32 = $dst;
            vst1q_f32(dst, f0);
            vst1q_f32(dst.add(4), f1);
            vst1q_f32(dst.add(8), f2);
            vst1q_f32(dst.add(12), f3);
        }};
    }

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();
    let (scales, mins) = extract_k_quant_scales(&block.scales);
    let q = &block.qs;

    let mask_lo = vdupq_n_u8(0x0F);
    let q_base = q.as_ptr();
    let y_base = output.as_mut_ptr();

    for g in 0..4 {
        let is = g * 2;
        let d1 = vdupq_n_f32(d * scales[is]);
        let m1 = vdupq_n_f32(dmin * mins[is]);
        let d2 = vdupq_n_f32(d * scales[is + 1]);
        let m2 = vdupq_n_f32(dmin * mins[is + 1]);

        let q_ptr = q_base.add(g * 32);
        let q_v0 = vld1q_u8(q_ptr);
        let q_v1 = vld1q_u8(q_ptr.add(16));

        let n_lo0 = vandq_u8(q_v0, mask_lo);
        let n_lo1 = vandq_u8(q_v1, mask_lo);
        let n_hi0 = vshrq_n_u8::<4>(q_v0);
        let n_hi1 = vshrq_n_u8::<4>(q_v1);

        let y_ptr = y_base.add(g * 64);
        store_dequant_x16!(y_ptr, n_lo0, d1, m1);
        store_dequant_x16!(y_ptr.add(16), n_lo1, d1, m1);
        store_dequant_x16!(y_ptr.add(32), n_hi0, d2, m2);
        store_dequant_x16!(y_ptr.add(48), n_hi1, d2, m2);
    }
}

/// Q4_K fused dequant-dot NEON — 한 BlockQ4_K (256 elements) 을 dequantize 하면서
/// 바로 같은 위치의 `x[256]` 와 dot 을 계산.
///
/// Stack buffer `[f32; 256]` write + read 제거. Dequant 결과 `float32x4_t` 를
/// 바로 `vfmaq_f32` 의 weight 로 사용해 register→register 로 accumulate.
///
/// # Bit-exactness
///
/// 이 함수의 결과는 `dequantize_q4_k_neon` 후 아래 구조와 동일한 순서의 dot 을
/// 수행한 값과 **bit-identical** 이어야 함:
///
/// ```ignore
///   let mut acc0 = vdupq_n_f32(0.0);
///   let mut acc1 = vdupq_n_f32(0.0);
///   // i = 0, 8, 16, ..., 248 (256 / 8 = 32 steps)
///   for i in (0..256).step_by(8) {
///       acc0 = vfmaq_f32(acc0, vld1q_f32(tmp.add(i)),   vld1q_f32(x.add(i)));
///       acc1 = vfmaq_f32(acc1, vld1q_f32(tmp.add(i+4)), vld1q_f32(x.add(i+4)));
///   }
///   vaddvq_f32(vaddq_f32(acc0, acc1))
/// ```
///
/// 즉 `rnb-llm::engine::dot_f32_neon(tmp.as_ptr(), x, 256)` 와 동일.
///
/// Dequant 순서 (g=0..4, 16-element 단위로 `store_dequant_x16` 와 동일):
///   y[g*64    ..g*64+16]  ← n_lo0 × d1 - m1   (f0, f1, f2, f3)
///   y[g*64+16 ..g*64+32]  ← n_lo1 × d1 - m1
///   y[g*64+32 ..g*64+48]  ← n_hi0 × d2 - m2
///   y[g*64+48 ..g*64+64]  ← n_hi1 × d2 - m2
///
/// 64-element 그룹 안에서 `tmp` 의 index [0..7],[8..15],... 은 이어지는 8-element
/// pair 로 acc0/acc1 에 fma. Fused 경로도 동일하게 두 개의 f32x4 dequant lane
/// (f0, f1) 이 하나의 8-element chunk 를 구성하도록 순서를 유지함.
///
/// MoE router softmax argmax 가 float add 순서에 민감 (Session 60 C v1 regression
/// 참고) → 이 순서 맞추지 않으면 같은 수치라도 argmax 가 뒤집힘.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_q4k_fused_neon(block: &BlockQ4_K, x: *const f32) -> f32 {
    use std::arch::aarch64::*;

    /// 16-element dequant + fma. nibbles(u8x16) → 4× f32x4 dequant → 대응되는
    /// x 의 f32x4 4개와 vfmaq_f32. acc0 가 [0..3],[8..11] (lane 0,2),
    /// acc1 가 [4..7],[12..15] (lane 1,3) 의 부분 합을 담음.
    /// `dot_f32_neon` 의 2-way unroll 패턴과 bit-identical.
    macro_rules! fmla_dequant_x16 {
        ($x_ptr:expr, $nibbles:expr, $dv:expr, $mv:expr, $acc0:expr, $acc1:expr) => {{
            let nibbles: uint8x16_t = $nibbles;
            let dv: float32x4_t = $dv;
            let mv: float32x4_t = $mv;
            let u16_lo = vmovl_u8(vget_low_u8(nibbles));
            let u16_hi = vmovl_u8(vget_high_u8(nibbles));

            let u32_0 = vmovl_u16(vget_low_u16(u16_lo));
            let u32_1 = vmovl_u16(vget_high_u16(u16_lo));
            let u32_2 = vmovl_u16(vget_low_u16(u16_hi));
            let u32_3 = vmovl_u16(vget_high_u16(u16_hi));

            // Bit-exact with `dequantize_q4_k_neon`: vmulq then vsubq (no FMA).
            // Scalar reference: `d * nibble - m`.
            let f0 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_0), dv), mv);
            let f1 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_1), dv), mv);
            let f2 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_2), dv), mv);
            let f3 = vsubq_f32(vmulq_f32(vcvtq_f32_u32(u32_3), dv), mv);

            let xp: *const f32 = $x_ptr;
            let x0 = vld1q_f32(xp);
            let x1 = vld1q_f32(xp.add(4));
            let x2 = vld1q_f32(xp.add(8));
            let x3 = vld1q_f32(xp.add(12));

            // 8-elem pair 매핑 — dot_f32_neon 의 acc0/acc1 순서와 동일.
            $acc0 = vfmaq_f32($acc0, f0, x0);
            $acc1 = vfmaq_f32($acc1, f1, x1);
            $acc0 = vfmaq_f32($acc0, f2, x2);
            $acc1 = vfmaq_f32($acc1, f3, x3);
        }};
    }

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();
    let (scales, mins) = extract_k_quant_scales(&block.scales);
    let q = &block.qs;

    let mask_lo = vdupq_n_u8(0x0F);
    let q_base = q.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for g in 0..4 {
        let is = g * 2;
        let d1 = vdupq_n_f32(d * scales[is]);
        let m1 = vdupq_n_f32(dmin * mins[is]);
        let d2 = vdupq_n_f32(d * scales[is + 1]);
        let m2 = vdupq_n_f32(dmin * mins[is + 1]);

        let q_ptr = q_base.add(g * 32);
        let q_v0 = vld1q_u8(q_ptr);
        let q_v1 = vld1q_u8(q_ptr.add(16));

        let n_lo0 = vandq_u8(q_v0, mask_lo);
        let n_lo1 = vandq_u8(q_v1, mask_lo);
        let n_hi0 = vshrq_n_u8::<4>(q_v0);
        let n_hi1 = vshrq_n_u8::<4>(q_v1);

        let x_ptr = x.add(g * 64);
        fmla_dequant_x16!(x_ptr, n_lo0, d1, m1, acc0, acc1);
        fmla_dequant_x16!(x_ptr.add(16), n_lo1, d1, m1, acc0, acc1);
        fmla_dequant_x16!(x_ptr.add(32), n_hi0, d2, m2, acc0, acc1);
        fmla_dequant_x16!(x_ptr.add(48), n_hi1, d2, m2, acc0, acc1);
    }

    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// Chunked-scalar Q5_1 dot product. **Same accumulation order as
/// `dot_q5_1_fused_neon`** (acc0 takes 4-wide chunks 0,2,4,6;
/// acc1 takes 1,3,5,7), then horizontal sum. This is the host
/// reference for bit-exact NEON validation and the x86 production path.
///
/// Differs from `dequantize_q5_1` + sequential scalar dot (which is what
/// `dot_basic_blocks_scalar` does) — that path's 32-elem sequential add
/// chain is float-non-associative-different from this chunked version.
/// The chunked layout is what makes NEON bit-exact possible.
pub fn dot_q5_1_chunked_scalar(block: &BlockQ5_1, x: &[f32; 32]) -> f32 {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let qh = u32::from_le_bytes(block.qh);
    let mut tmp = [0f32; 32];
    for i in 0..16 {
        let byte = block.qs[i];
        let xh_0 = ((qh >> i) & 1) as u8;
        let xh_1 = ((qh >> (i + 16)) & 1) as u8;
        let q0 = ((byte & 0x0F) | (xh_0 << 4)) as f32;
        let q1 = ((byte >> 4) | (xh_1 << 4)) as f32;
        tmp[i] = q0 * d + m;
        tmp[i + 16] = q1 * d + m;
    }
    let mut acc0 = [0f32; 4];
    let mut acc1 = [0f32; 4];
    // 8 chunks of 4 elements each: 0,2,4,6 → acc0; 1,3,5,7 → acc1.
    for chunk in 0..8 {
        let base = chunk * 4;
        let acc = if chunk % 2 == 0 { &mut acc0 } else { &mut acc1 };
        for j in 0..4 {
            acc[j] += tmp[base + j] * x[base + j];
        }
    }
    let s0 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
    let s1 = acc1[0] + acc1[1] + acc1[2] + acc1[3];
    // vaddq_f32(acc0, acc1) lane-wise sum → vaddvq_f32 horizontal sum.
    // Expressed as scalars: (acc0[0]+acc1[0]) + (acc0[1]+acc1[1]) + ...
    // But NEON's vaddvq_f32 may pairwise-reduce as
    //   ((s0+s1) + (s2+s3)) where s_i = acc0[i]+acc1[i].
    // To match, do the equivalent step here:
    let p0 = acc0[0] + acc1[0];
    let p1 = acc0[1] + acc1[1];
    let p2 = acc0[2] + acc1[2];
    let p3 = acc0[3] + acc1[3];
    let _ = (s0, s1); // unused, kept for clarity above
    (p0 + p1) + (p2 + p3)
}

/// Bit-exact NEON Q5_1 fused dequant + dot.
///
/// 32 quantized weights (5-bit unsigned, 0..31) are dequantized in-register
/// (`q * d + m`) and dotted against the 32-element f32 input `x` via two
/// f32x4 accumulators (chunks 0,2,4,6 → acc0; 1,3,5,7 → acc1) — the same
/// 2-way unroll pattern as `dot_f32_neon`. The companion
/// `dot_q5_1_chunked_scalar` reproduces the identical add order on
/// host/scalar paths so the two are bit-identical.
///
/// Skips the `[f32; 32]` stack buffer that `dot_basic_blocks_scalar` uses —
/// dequantized lanes flow straight into the FMA pipe.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_q5_1_fused_neon(block: &BlockQ5_1, x: *const f32) -> f32 {
    use std::arch::aarch64::*;

    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let qh = u32::from_le_bytes(block.qh);
    let dv = vdupq_n_f32(d);
    let mv = vdupq_n_f32(m);

    // Build a u8x16 mask of high-bits for low-nibble lane:
    //   byte i = 0x10 if (qh >> i) & 1 else 0.
    // Same for the high-nibble lane, using bits 16..31 of qh.
    // Cheap to materialize via small stack arrays — 16 ops, no NEON branch.
    let mut qh_lo_bytes = [0u8; 16];
    let mut qh_hi_bytes = [0u8; 16];
    for i in 0..16 {
        if (qh >> i) & 1 != 0 {
            qh_lo_bytes[i] = 0x10;
        }
        if (qh >> (i + 16)) & 1 != 0 {
            qh_hi_bytes[i] = 0x10;
        }
    }
    let qh_lo_v = vld1q_u8(qh_lo_bytes.as_ptr());
    let qh_hi_v = vld1q_u8(qh_hi_bytes.as_ptr());

    let qs = vld1q_u8(block.qs.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);
    let lo_nib = vandq_u8(qs, mask_lo);
    let hi_nib = vshrq_n_u8::<4>(qs);

    // q5 values 0..31 in u8 lanes.
    let q5_lo = vorrq_u8(lo_nib, qh_lo_v);
    let q5_hi = vorrq_u8(hi_nib, qh_hi_v);

    // Widen u8x16 → u16x8 × 2 → u32x4 × 4 for each of lo / hi.
    let u16_lo_l = vmovl_u8(vget_low_u8(q5_lo));
    let u16_lo_h = vmovl_u8(vget_high_u8(q5_lo));
    let u16_hi_l = vmovl_u8(vget_low_u8(q5_hi));
    let u16_hi_h = vmovl_u8(vget_high_u8(q5_hi));

    let lo0 = vmovl_u16(vget_low_u16(u16_lo_l));
    let lo1 = vmovl_u16(vget_high_u16(u16_lo_l));
    let lo2 = vmovl_u16(vget_low_u16(u16_lo_h));
    let lo3 = vmovl_u16(vget_high_u16(u16_lo_h));
    let hi0 = vmovl_u16(vget_low_u16(u16_hi_l));
    let hi1 = vmovl_u16(vget_high_u16(u16_hi_l));
    let hi2 = vmovl_u16(vget_low_u16(u16_hi_h));
    let hi3 = vmovl_u16(vget_high_u16(u16_hi_h));

    // f = q*d + m  — bit-exact with scalar `q0 * d + m`. Use separate
    // multiply then add (vaddq), NOT vfmaq, because FMA collapses two
    // rounds into one and produces a different f32 than scalar
    // `(q*d) + m`. Same reason `dot_q4k_fused_neon` uses vmulq+vsubq
    // instead of vfmaq for the dequant stage.
    let f_lo0 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(lo0), dv), mv);
    let f_lo1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(lo1), dv), mv);
    let f_lo2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(lo2), dv), mv);
    let f_lo3 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(lo3), dv), mv);
    let f_hi0 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(hi0), dv), mv);
    let f_hi1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(hi1), dv), mv);
    let f_hi2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(hi2), dv), mv);
    let f_hi3 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(hi3), dv), mv);

    let x_l0 = vld1q_f32(x);
    let x_l1 = vld1q_f32(x.add(4));
    let x_l2 = vld1q_f32(x.add(8));
    let x_l3 = vld1q_f32(x.add(12));
    let x_h0 = vld1q_f32(x.add(16));
    let x_h1 = vld1q_f32(x.add(20));
    let x_h2 = vld1q_f32(x.add(24));
    let x_h3 = vld1q_f32(x.add(28));

    // 8 chunks of 4 lanes; 2-way acc split (even/odd chunk).
    // Use separate vmulq + vaddq (NOT vfmaq) so the mul/add round trip
    // matches the scalar reference's `+= a * b` (mul, then add — two rounds).
    // FMA collapses to one round and produces a different ULP.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    acc0 = vaddq_f32(acc0, vmulq_f32(f_lo0, x_l0)); // chunk 0
    acc1 = vaddq_f32(acc1, vmulq_f32(f_lo1, x_l1)); // chunk 1
    acc0 = vaddq_f32(acc0, vmulq_f32(f_lo2, x_l2)); // chunk 2
    acc1 = vaddq_f32(acc1, vmulq_f32(f_lo3, x_l3)); // chunk 3
    acc0 = vaddq_f32(acc0, vmulq_f32(f_hi0, x_h0)); // chunk 4
    acc1 = vaddq_f32(acc1, vmulq_f32(f_hi1, x_h1)); // chunk 5
    acc0 = vaddq_f32(acc0, vmulq_f32(f_hi2, x_h2)); // chunk 6
    acc1 = vaddq_f32(acc1, vmulq_f32(f_hi3, x_h3)); // chunk 7

    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// Q5_K dequantization — matches llama.cpp dequantize_row_q5_K exactly.
///
/// 256 elements processed in 4 groups of 64. Each group uses 32 bytes of ql:
///   first 32 elements = low nibbles + qh bit (scale is+0)
///   next 32 elements = high nibbles + qh bit (scale is+1)
/// qh bits advance by 2 positions per group (u1 and u2 masks).
pub fn dequantize_q5_k(block: &BlockQ5_K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let (scales, mins) = extract_k_quant_scales(&block.scales);
    let ql = &block.qs;
    let qh = &block.qh;

    let mut is = 0;
    let mut ql_off = 0;
    let mut y_off = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for _ in 0..4 {
        let d1 = d * scales[is];
        let m1 = dmin * mins[is];
        let d2 = d * scales[is + 1];
        let m2 = dmin * mins[is + 1];

        for l in 0..32 {
            let high: u8 = if qh[l] & u1 != 0 { 16 } else { 0 };
            output[y_off + l] = d1 * ((ql[ql_off + l] & 0xF) + high) as f32 - m1;
        }
        for l in 0..32 {
            let high: u8 = if qh[l] & u2 != 0 { 16 } else { 0 };
            output[y_off + 32 + l] = d2 * ((ql[ql_off + l] >> 4) + high) as f32 - m2;
        }

        ql_off += 32;
        is += 2;
        u1 = u1.wrapping_shl(2);
        u2 = u2.wrapping_shl(2);
        y_off += 64;
    }
}

/// Q6_K dequantization — llama.cpp dequantize_row_q6_K と完全一致する実装.
///
/// 256 elements を 2つの128-element グループで処理.
/// 各グループ内で 32 elements を interleaved pattern で展開:
///   y[l+0]  = low4(ql[l])    | high2(qh[l], bits 0-1)
///   y[l+32] = low4(ql[l+32]) | high2(qh[l], bits 2-3)
///   y[l+64] = high4(ql[l])   | high2(qh[l], bits 4-5)
///   y[l+96] = high4(ql[l+32])| high2(qh[l], bits 6-7)
pub fn dequantize_q6_k(block: &BlockQ6_K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let ql = &block.ql;
    let qh = &block.qh;
    let sc = &block.scales;

    // 2 groups of 128 elements
    for n in 0..2 {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let sc_base = n * 8;
        let y_base = n * 128;

        for l in 0..32 {
            let is = l / 16; // 0 for first 16, 1 for next 16

            let q1 = (ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4);
            let q2 = (ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4);
            let q3 = (ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4);
            let q4 = (ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4);

            output[y_base + l] = d * sc[sc_base + is] as f32 * (q1 as i32 - 32) as f32;
            output[y_base + l + 32] = d * sc[sc_base + is + 2] as f32 * (q2 as i32 - 32) as f32;
            output[y_base + l + 64] = d * sc[sc_base + is + 4] as f32 * (q3 as i32 - 32) as f32;
            output[y_base + l + 96] = d * sc[sc_base + is + 6] as f32 * (q4 as i32 - 32) as f32;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    // --- Task 3 tests ---

    #[test]
    fn test_dequant_q4_0_zero_scale() {
        let block = BlockQ4_0 {
            d: f16::ZERO,
            qs: [0x88; 16],
        };
        let mut out = [0.0f32; 32];
        dequantize_q4_0(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q4_0_known_values() {
        // All nibbles = 8 → value = (8-8)*scale = 0
        let block = BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88; 16],
        };
        let mut out = [0.0f32; 32];
        dequantize_q4_0(&block, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_dequant_q8_0() {
        let mut block = BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: [0i8; 32],
        };
        block.qs[0] = 5;
        block.qs[1] = -3;
        let mut out = [0.0f32; 32];
        dequantize_q8_0(&block, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-3);
        assert!((out[1] - (-3.0)).abs() < 1e-3);
        assert!(out[2].abs() < 1e-5);
    }

    #[test]
    fn test_dequant_q4_1() {
        // nibble=0, scale=2.0, min=1.0 → value = 0*2+1 = 1.0
        let block = BlockQ4_1 {
            d: f16::from_f32(2.0),
            m: f16::from_f32(1.0),
            qs: [0; 16],
        };
        let mut out = [0.0f32; 32];
        dequantize_q4_1(&block, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-3));
    }

    #[test]
    fn test_dequant_q5_0_zero() {
        let block = BlockQ5_0 {
            d: f16::ZERO,
            qh: [0; 4],
            qs: [0; 16],
        };
        let mut out = [0.0f32; 32];
        dequantize_q5_0(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q5_1_zero_min() {
        // All nibbles=0, qh=0, scale=1.0, min=0.0 → all values = 0
        let block = BlockQ5_1 {
            d: f16::from_f32(1.0),
            m: f16::ZERO,
            qh: [0; 4],
            qs: [0; 16],
        };
        let mut out = [0.0f32; 32];
        dequantize_q5_1(&block, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-5));
    }

    // --- Task 4 tests ---

    #[test]
    fn test_dequant_q2_k_zero() {
        let block = BlockQ2_K {
            scales: [0; 16],
            qs: [0; 64],
            d: f16::ZERO,
            dmin: f16::ZERO,
        };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(&block, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_dequant_q2_k_uses_canonical_gguf_interleave() {
        let mut qs = [0u8; 64];
        qs[1] = 3;
        let block = BlockQ2_K {
            scales: [1; 16],
            qs,
            d: f16::ONE,
            dmin: f16::ZERO,
        };
        let mut out = [0.0f32; 256];
        dequantize_q2_k(&block, &mut out);

        assert_eq!(out[1], 3.0);
        assert_eq!(out[4], 0.0);
        assert_eq!(out[33], 0.0);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dot_q2k_fused_neon_close_to_scalar_reference() {
        fn lcg(seed: &mut u32) -> u32 {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (*seed >> 16) & 0xFFFF
        }

        for trial in 0..256u32 {
            let mut seed = 0xA5A5_u32.wrapping_add(trial.wrapping_mul(0x9E37));
            let d_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let dmin_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let mut scales = [0u8; 16];
            for s in scales.iter_mut() {
                *s = lcg(&mut seed) as u8;
            }
            let mut qs = [0u8; 64];
            for q in qs.iter_mut() {
                *q = lcg(&mut seed) as u8;
            }
            let block = BlockQ2_K {
                scales,
                qs,
                d: f16::from_bits(d_bits),
                dmin: f16::from_bits(dmin_bits),
            };

            let d_f = block.d.to_f32();
            let dmin_f = block.dmin.to_f32();
            if !d_f.is_finite() || !dmin_f.is_finite() {
                continue;
            }

            let mut x = [0.0f32; 256];
            for (i, v) in x.iter_mut().enumerate() {
                let s = (lcg(&mut seed) & 0xFFFF) as i32 - 0x8000;
                *v = (s as f32) / 65536.0;
                if i & 1 == 0 {
                    *v = -*v;
                }
            }

            let mut tmp = [0.0f32; 256];
            dequantize_q2_k(&block, &mut tmp);
            let scalar = tmp.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum::<f32>();
            let neon = unsafe { dot_q2k_fused_neon(&block, x.as_ptr()) };

            if !scalar.is_finite() || !neon.is_finite() {
                continue;
            }

            let diff = (scalar - neon).abs();
            let tol = 1e-4_f32 * scalar.abs().max(1.0);
            assert!(
                diff <= tol,
                "trial {}: scalar={} neon={} diff={} tol={}",
                trial,
                scalar,
                neon,
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_dequant_q6_k_zero() {
        let block = BlockQ6_K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f16::ZERO,
        };
        let mut out = [0.0f32; 256];
        dequantize_q6_k(&block, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_dequant_q3_k_zero() {
        let block = BlockQ3_K {
            hmask: [0; 32],
            qs: [0; 64],
            scales: [0; 12],
            d: f16::ZERO,
        };
        let mut out = [0.0f32; 256];
        dequantize_q3_k(&block, &mut out);
        // With d=0, all outputs should be 0
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q4_k_zero() {
        let block = BlockQ4_K {
            d: f16::ZERO,
            dmin: f16::ZERO,
            scales: [0; 12],
            qs: [0; 128],
        };
        let mut out = [0.0f32; 256];
        dequantize_q4_k(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q5_k_zero() {
        let block = BlockQ5_K {
            d: f16::ZERO,
            dmin: f16::ZERO,
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        let mut out = [0.0f32; 256];
        dequantize_q5_k(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Reference dot: `dequantize_q4_k_neon` 를 stack buffer 에 쓴 뒤
    /// `dot_f32_neon` 패턴 (2-way acc0/acc1 + vfmaq_f32, 8-elem pair unroll)
    /// 으로 accumulate. `dot_q4k_fused_neon` 과 이 함수 결과가 bit-identical
    /// 이어야 함 — MoE router softmax argmax tie-sensitivity 방지.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn reference_dot_q4k_neon(block: &BlockQ4_K, x: &[f32; 256]) -> f32 {
        use std::arch::aarch64::*;
        let mut tmp = [0.0f32; 256];
        dequantize_q4_k_neon(block, &mut tmp);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let a = tmp.as_ptr();
        let b = x.as_ptr();
        let mut i = 0usize;
        while i + 8 <= 256 {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
            i += 8;
        }
        vaddvq_f32(vaddq_f32(acc0, acc1))
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dot_q4k_fused_neon_bit_exact() {
        fn lcg(seed: &mut u32) -> u32 {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (*seed >> 16) & 0xFFFF
        }

        for trial in 0..128u32 {
            let mut seed = 0xFEED_u32.wrapping_add(trial.wrapping_mul(0x9E37));
            let d_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let dmin_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let mut scales = [0u8; 12];
            for s in scales.iter_mut() {
                *s = lcg(&mut seed) as u8;
            }
            let mut qs = [0u8; 128];
            for q in qs.iter_mut() {
                *q = lcg(&mut seed) as u8;
            }
            let block = BlockQ4_K {
                d: f16::from_bits(d_bits),
                dmin: f16::from_bits(dmin_bits),
                scales,
                qs,
            };
            let d_f = block.d.to_f32();
            let dmin_f = block.dmin.to_f32();
            if !d_f.is_finite() || !dmin_f.is_finite() {
                continue;
            }

            // Small-magnitude f32 x to keep dot in finite range regardless of super-scale.
            let mut x = [0.0f32; 256];
            for (i, v) in x.iter_mut().enumerate() {
                let s = (lcg(&mut seed) & 0xFFFF) as i32 - 0x8000;
                *v = (s as f32) / 65536.0;
                if i & 1 == 0 {
                    *v = -*v;
                }
            }

            let ref_sum = unsafe { reference_dot_q4k_neon(&block, &x) };
            let fused = unsafe { dot_q4k_fused_neon(&block, x.as_ptr()) };

            if !ref_sum.is_finite() || !fused.is_finite() {
                // Skip super-scale combos that overflow to inf/NaN — bit-pattern of
                // NaN can differ; outside what we're validating.
                continue;
            }
            assert_eq!(
                ref_sum.to_bits(),
                fused.to_bits(),
                "trial {}: ref={} fused={}",
                trial,
                ref_sum,
                fused
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dot_q5_1_fused_neon_bit_exact() {
        fn lcg(seed: &mut u32) -> u32 {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (*seed >> 16) & 0xFFFF
        }

        for trial in 0..256u32 {
            let mut seed = 0xCAFE_u32.wrapping_add(trial.wrapping_mul(0x9E37));
            let d_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let m_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let mut qh = [0u8; 4];
            for q in qh.iter_mut() {
                *q = lcg(&mut seed) as u8;
            }
            let mut qs = [0u8; 16];
            for q in qs.iter_mut() {
                *q = lcg(&mut seed) as u8;
            }
            let block = BlockQ5_1 {
                d: f16::from_bits(d_bits),
                m: f16::from_bits(m_bits),
                qh,
                qs,
            };
            let d_f = block.d.to_f32();
            let m_f = block.m.to_f32();
            if !d_f.is_finite() || !m_f.is_finite() {
                continue;
            }

            let mut x = [0.0f32; 32];
            for (i, v) in x.iter_mut().enumerate() {
                let s = (lcg(&mut seed) & 0xFFFF) as i32 - 0x8000;
                *v = (s as f32) / 65536.0;
                if i & 1 == 0 {
                    *v = -*v;
                }
            }

            let scalar = dot_q5_1_chunked_scalar(&block, &x);
            let neon = unsafe { dot_q5_1_fused_neon(&block, x.as_ptr()) };
            if !scalar.is_finite() || !neon.is_finite() {
                continue;
            }
            assert_eq!(
                scalar.to_bits(),
                neon.to_bits(),
                "trial {}: scalar={} neon={}",
                trial,
                scalar,
                neon
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dequant_q4_k_neon_bit_exact() {
        // Pseudo-random BlockQ4_K values cover the full (d, dmin, scales, qs) space.
        // NEON path must be bit-identical to the scalar reference — MoE router softmax
        // argmax is tie-sensitive (Session 60 C v1: FMA reorders flipped tokens).
        fn lcg(seed: &mut u32) -> u32 {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (*seed >> 16) & 0xFFFF
        }

        for trial in 0..128u32 {
            let mut seed = 0xC0FFEE_u32.wrapping_add(trial.wrapping_mul(0x9E37));
            let d_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let dmin_bits = (lcg(&mut seed) & 0xFFFF) as u16;
            let mut scales = [0u8; 12];
            for s in scales.iter_mut() {
                *s = lcg(&mut seed) as u8;
            }
            let mut qs = [0u8; 128];
            for q in qs.iter_mut() {
                *q = lcg(&mut seed) as u8;
            }
            let block = BlockQ4_K {
                d: f16::from_bits(d_bits),
                dmin: f16::from_bits(dmin_bits),
                scales,
                qs,
            };

            // Skip NaN/Inf super-scales — scalar and NEON both produce NaN/Inf but
            // bit-patterns of NaNs can differ; not what we're validating here.
            let d_f = block.d.to_f32();
            let dmin_f = block.dmin.to_f32();
            if !d_f.is_finite() || !dmin_f.is_finite() {
                continue;
            }

            let mut out_scalar = [0.0f32; 256];
            let mut out_neon = [0.0f32; 256];
            dequantize_q4_k(&block, &mut out_scalar);
            unsafe { dequantize_q4_k_neon(&block, &mut out_neon) };

            for i in 0..256 {
                assert_eq!(
                    out_scalar[i].to_bits(),
                    out_neon[i].to_bits(),
                    "trial {} idx {}: scalar={} neon={}",
                    trial,
                    i,
                    out_scalar[i],
                    out_neon[i]
                );
            }
        }
    }
}
