use super::{Q8Block, Q8KBlock};
use crate::gemm::repack::{META_Q4K_BLOCK_BYTES, MPK_DMIN_OFF, MPK_D_OFF, MPK_MN_OFF, MPK_SC_OFF};
use rayon::prelude::*;
use std::arch::aarch64::*;

/// NEON-accelerated f32 dot product (4-wide FMA, 2x unrolled).
#[inline]
pub unsafe fn dot_f32_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
        i += 8;
    }
    if i + 4 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

fn parallel_chunk_len(rows: usize, n_threads: usize) -> usize {
    if rows <= 64 {
        rows.max(1)
    } else {
        ((rows + (n_threads * 4) - 1) / (n_threads * 4)).max(1)
    }
}

/// Hardware f16→f32 conversion using NEON FCVTL instruction.
/// ARMv8.0-A compatible (no FP16 extension needed).
/// DUP + FCVTL = 2 instructions vs ~15 for software `half::f16::to_f32()`.
#[inline(always)]
#[allow(asm_sub_register)]
unsafe fn f16_to_f32(bits: u16) -> f32 {
    let result: f32;
    core::arch::asm!(
        "dup {v}.4h, {bits:w}",
        "fcvtl {v}.4s, {v}.4h",
        bits = in(reg) bits as u32,
        v = lateout(vreg) result,
    );
    result
}

/// Load two adjacent f16 values and convert both to f32 in one shot.
/// `ldr s + fcvtl` = 2 instructions vs 6 for two separate f16_to_f32 calls.
#[inline(always)]
#[allow(asm_sub_register)]
unsafe fn f16_pair_to_f32(ptr: *const u8) -> (f32, f32) {
    let v: float32x4_t;
    core::arch::asm!(
        "ldr {v:s}, [{ptr}]",
        "fcvtl {v}.4s, {v}.4h",
        ptr = in(reg) ptr,
        v = out(vreg) v,
    );
    (vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1))
}

/// Prefetch data into L1 cache (read, keep in all cache levels).
#[inline(always)]
pub unsafe fn prefetch_l1(ptr: *const u8) {
    core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ptr, options(nostack, preserves_flags));
}

/// Read 2 little-endian bytes as u16 (f16 bits) from a byte slice.
#[inline(always)]
unsafe fn read_f16_le(ptr: *const u8) -> u16 {
    (ptr.read() as u16) | ((ptr.add(1).read() as u16) << 8)
}

/// Q4_0 x Q8 integer dot product with NEON vdotq_s32.
/// Processes 16 int8 multiply-accumulates per vdotq_s32 instruction.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_0_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mask_low = vdupq_n_u8(0x0F);
    let sub8 = vdupq_n_s8(8);

    let mut bi = 0;
    while bi + 2 <= n_blocks {
        // Prefetch next pair of blocks (2 × 18 = 36 bytes)
        if bi + 4 <= n_blocks {
            prefetch_l1(row_bytes.as_ptr().add((bi + 2) * 18));
        }
        // Block 0
        let boff0 = bi * 18;
        let d0 = f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff0)));
        let qbytes0 = vld1q_u8(row_bytes.as_ptr().add(boff0 + 2));
        let v0_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qbytes0, mask_low)), sub8);
        let v0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qbytes0, 4)), sub8);
        let x0_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x0_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let mut p0 = vdupq_n_s32(0);
        p0 = vdotq_s32(p0, v0_lo, x0_lo);
        p0 = vdotq_s32(p0, v0_hi, x0_hi);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d0 * q8[bi].d);

        // Block 1
        let boff1 = (bi + 1) * 18;
        let d1 = f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff1)));
        let qbytes1 = vld1q_u8(row_bytes.as_ptr().add(boff1 + 2));
        let v1_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qbytes1, mask_low)), sub8);
        let v1_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qbytes1, 4)), sub8);
        let x1_lo = vld1q_s8(q8[bi + 1].qs.as_ptr());
        let x1_hi = vld1q_s8(q8[bi + 1].qs.as_ptr().add(16));
        let mut p1 = vdupq_n_s32(0);
        p1 = vdotq_s32(p1, v1_lo, x1_lo);
        p1 = vdotq_s32(p1, v1_hi, x1_hi);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d1 * q8[bi + 1].d);

        bi += 2;
    }
    if bi < n_blocks {
        let boff = bi * 18;
        let d = f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff)));
        let qbytes = vld1q_u8(row_bytes.as_ptr().add(boff + 2));
        let v_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qbytes, mask_low)), sub8);
        let v_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qbytes, 4)), sub8);
        let x_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let mut p = vdupq_n_s32(0);
        p = vdotq_s32(p, v_lo, x_lo);
        p = vdotq_s32(p, v_hi, x_hi);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d * q8[bi].d);
    }
    vaddvq_f32(vaddq_f32(sumv0, sumv1))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn unpack_q5_0_block(
    block: *const u8,
    bit_masks: uint8x16_t,
    mask_low: uint8x16_t,
    high_bit: uint8x16_t,
    sub16: int8x16_t,
) -> (int8x16_t, int8x16_t) {
    let qh_lo_source = vcombine_u8(vdup_n_u8(*block.add(2)), vdup_n_u8(*block.add(3)));
    let qh_hi_source = vcombine_u8(vdup_n_u8(*block.add(4)), vdup_n_u8(*block.add(5)));
    let qh_lo = vandq_u8(vtstq_u8(qh_lo_source, bit_masks), high_bit);
    let qh_hi = vandq_u8(vtstq_u8(qh_hi_source, bit_masks), high_bit);
    let packed = vld1q_u8(block.add(6));
    let low = vorrq_u8(vandq_u8(packed, mask_low), qh_lo);
    let high = vorrq_u8(vshrq_n_u8(packed, 4), qh_hi);
    (
        vsubq_s8(vreinterpretq_s8_u8(low), sub16),
        vsubq_s8(vreinterpretq_s8_u8(high), sub16),
    )
}

/// Q5_0 x Q8 integer dot product with NEON vdotq_s32 (2-block unrolled).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q5_0_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    const BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    let bits = vld1_u8(BIT_MASKS.as_ptr());
    let bit_masks = vcombine_u8(bits, bits);
    let mask_low = vdupq_n_u8(0x0f);
    let high_bit = vdupq_n_u8(0x10);
    let sub16 = vdupq_n_s8(16);
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let ptr = row_bytes.as_ptr();
    let mut bi = 0usize;

    while bi + 2 <= n_blocks {
        if bi + 4 <= n_blocks {
            prefetch_l1(ptr.add((bi + 2) * 22));
        }

        let block0 = ptr.add(bi * 22);
        let (q0_lo, q0_hi) = unpack_q5_0_block(block0, bit_masks, mask_low, high_bit, sub16);
        let mut p0 = vdupq_n_s32(0);
        p0 = vdotq_s32(p0, q0_lo, vld1q_s8(q8[bi].qs.as_ptr()));
        p0 = vdotq_s32(p0, q0_hi, vld1q_s8(q8[bi].qs.as_ptr().add(16)));
        let d0 = f16_to_f32(read_f16_le(block0)) * q8[bi].d;
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d0);

        let block1 = ptr.add((bi + 1) * 22);
        let (q1_lo, q1_hi) = unpack_q5_0_block(block1, bit_masks, mask_low, high_bit, sub16);
        let mut p1 = vdupq_n_s32(0);
        p1 = vdotq_s32(p1, q1_lo, vld1q_s8(q8[bi + 1].qs.as_ptr()));
        p1 = vdotq_s32(p1, q1_hi, vld1q_s8(q8[bi + 1].qs.as_ptr().add(16)));
        let d1 = f16_to_f32(read_f16_le(block1)) * q8[bi + 1].d;
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d1);

        bi += 2;
    }

    if bi < n_blocks {
        let block = ptr.add(bi * 22);
        let (q_lo, q_hi) = unpack_q5_0_block(block, bit_masks, mask_low, high_bit, sub16);
        let mut p = vdupq_n_s32(0);
        p = vdotq_s32(p, q_lo, vld1q_s8(q8[bi].qs.as_ptr()));
        p = vdotq_s32(p, q_hi, vld1q_s8(q8[bi].qs.as_ptr().add(16)));
        let d = f16_to_f32(read_f16_le(block)) * q8[bi].d;
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d);
    }

    vaddvq_f32(vaddq_f32(sumv0, sumv1))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn accumulate_q5_0_block(
    sum: float32x4_t,
    block: *const u8,
    x_lo: int8x16_t,
    x_hi: int8x16_t,
    x_scale: f32,
    bit_masks: uint8x16_t,
    mask_low: uint8x16_t,
    high_bit: uint8x16_t,
    sub16: int8x16_t,
) -> float32x4_t {
    let (w_lo, w_hi) = unpack_q5_0_block(block, bit_masks, mask_low, high_bit, sub16);
    let mut dot = vdupq_n_s32(0);
    dot = vdotq_s32(dot, w_lo, x_lo);
    dot = vdotq_s32(dot, w_hi, x_hi);
    let scale = f16_to_f32(read_f16_le(block)) * x_scale;
    vmlaq_n_f32(sum, vcvtq_f32_s32(dot), scale)
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q5_0_q8_neon_rows4(rows: [&[u8]; 4], q8: &[Q8Block], n_blocks: usize) -> [f32; 4] {
    const BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    let bits = vld1_u8(BIT_MASKS.as_ptr());
    let bit_masks = vcombine_u8(bits, bits);
    let mask_low = vdupq_n_u8(0x0f);
    let high_bit = vdupq_n_u8(0x10);
    let sub16 = vdupq_n_s8(16);
    let ptr0 = rows[0].as_ptr();
    let ptr1 = rows[1].as_ptr();
    let ptr2 = rows[2].as_ptr();
    let ptr3 = rows[3].as_ptr();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for (bi, x) in q8.iter().take(n_blocks).enumerate() {
        let x_lo = vld1q_s8(x.qs.as_ptr());
        let x_hi = vld1q_s8(x.qs.as_ptr().add(16));
        let boff = bi * 22;
        sum0 = accumulate_q5_0_block(
            sum0,
            ptr0.add(boff),
            x_lo,
            x_hi,
            x.d,
            bit_masks,
            mask_low,
            high_bit,
            sub16,
        );
        sum1 = accumulate_q5_0_block(
            sum1,
            ptr1.add(boff),
            x_lo,
            x_hi,
            x.d,
            bit_masks,
            mask_low,
            high_bit,
            sub16,
        );
        sum2 = accumulate_q5_0_block(
            sum2,
            ptr2.add(boff),
            x_lo,
            x_hi,
            x.d,
            bit_masks,
            mask_low,
            high_bit,
            sub16,
        );
        sum3 = accumulate_q5_0_block(
            sum3,
            ptr3.add(boff),
            x_lo,
            x_hi,
            x.d,
            bit_masks,
            mask_low,
            high_bit,
            sub16,
        );
    }

    [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ]
}

#[target_feature(enable = "neon,dotprod,i8mm")]
unsafe fn dot_q5_0_q8_i8mm_2x2(
    row0: &[u8],
    row1: &[u8],
    x0: &[Q8Block],
    x1: &[Q8Block],
    n_blocks: usize,
) -> [f32; 4] {
    const BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    let bits = vld1_u8(BIT_MASKS.as_ptr());
    let bit_masks = vcombine_u8(bits, bits);
    let mask_low = vdupq_n_u8(0x0f);
    let high_bit = vdupq_n_u8(0x10);
    let sub16 = vdupq_n_s8(16);
    let ptr0 = row0.as_ptr();
    let ptr1 = row1.as_ptr();
    let mut sum = vdupq_n_f32(0.0);

    for bi in 0..n_blocks {
        let boff = bi * 22;
        let block0 = ptr0.add(boff);
        let block1 = ptr1.add(boff);
        let (w0_lo, w0_hi) = unpack_q5_0_block(block0, bit_masks, mask_low, high_bit, sub16);
        let (w1_lo, w1_hi) = unpack_q5_0_block(block1, bit_masks, mask_low, high_bit, sub16);
        let x0_lo = vld1q_s8(x0[bi].qs.as_ptr());
        let x0_hi = vld1q_s8(x0[bi].qs.as_ptr().add(16));
        let x1_lo = vld1q_s8(x1[bi].qs.as_ptr());
        let x1_hi = vld1q_s8(x1[bi].qs.as_ptr().add(16));
        let mut dot = vdupq_n_s32(0);
        dot = vmmlaq_s32(
            dot,
            vcombine_s8(vget_low_s8(w0_lo), vget_low_s8(w1_lo)),
            vcombine_s8(vget_low_s8(x0_lo), vget_low_s8(x1_lo)),
        );
        dot = vmmlaq_s32(
            dot,
            vcombine_s8(vget_high_s8(w0_lo), vget_high_s8(w1_lo)),
            vcombine_s8(vget_high_s8(x0_lo), vget_high_s8(x1_lo)),
        );
        dot = vmmlaq_s32(
            dot,
            vcombine_s8(vget_low_s8(w0_hi), vget_low_s8(w1_hi)),
            vcombine_s8(vget_low_s8(x0_hi), vget_low_s8(x1_hi)),
        );
        dot = vmmlaq_s32(
            dot,
            vcombine_s8(vget_high_s8(w0_hi), vget_high_s8(w1_hi)),
            vcombine_s8(vget_high_s8(x0_hi), vget_high_s8(x1_hi)),
        );
        let d0 = f16_to_f32(read_f16_le(block0));
        let d1 = f16_to_f32(read_f16_le(block1));
        let scales = [d0 * x0[bi].d, d0 * x1[bi].d, d1 * x0[bi].d, d1 * x1[bi].d];
        sum = vfmaq_f32(sum, vcvtq_f32_s32(dot), vld1q_f32(scales.as_ptr()));
    }

    [
        vgetq_lane_f32::<0>(sum),
        vgetq_lane_f32::<1>(sum),
        vgetq_lane_f32::<2>(sum),
        vgetq_lane_f32::<3>(sum),
    ]
}

/// Q8_0 x Q8 integer dot product with NEON vdotq_s32 (2-block unrolled)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_0_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let pairs = n_blocks / 2;
    let ptr = row_bytes.as_ptr();

    // 2-block unrolled loop: overlap vdotq latency between blocks
    for i in 0..pairs {
        let bi = i * 2;
        let boff0 = bi * 34;
        let boff1 = boff0 + 34;

        // Prefetch 2 blocks ahead
        if bi + 3 < n_blocks {
            prefetch_l1(ptr.add(boff0 + 68));
            prefetch_l1(ptr.add(boff0 + 68 + 34));
        }

        let d0 = f16_to_f32(read_f16_le(ptr.add(boff0)));
        let d1 = f16_to_f32(read_f16_le(ptr.add(boff1)));

        // Interleave loads and dots between block 0 and block 1
        let w0_lo = vld1q_s8(ptr.add(boff0 + 2) as *const i8);
        let w1_lo = vld1q_s8(ptr.add(boff1 + 2) as *const i8);
        let x0_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x1_lo = vld1q_s8(q8[bi + 1].qs.as_ptr());

        let mut p0 = vdupq_n_s32(0);
        p0 = vdotq_s32(p0, w0_lo, x0_lo);
        let mut p1 = vdupq_n_s32(0);
        p1 = vdotq_s32(p1, w1_lo, x1_lo);

        let w0_hi = vld1q_s8(ptr.add(boff0 + 18) as *const i8);
        let w1_hi = vld1q_s8(ptr.add(boff1 + 18) as *const i8);
        let x0_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let x1_hi = vld1q_s8(q8[bi + 1].qs.as_ptr().add(16));

        p0 = vdotq_s32(p0, w0_hi, x0_hi);
        p1 = vdotq_s32(p1, w1_hi, x1_hi);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d0 * q8[bi].d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d1 * q8[bi + 1].d);
    }

    // Handle odd remainder block
    if n_blocks & 1 != 0 {
        let bi = n_blocks - 1;
        let boff = bi * 34;
        let d = f16_to_f32(read_f16_le(ptr.add(boff)));
        let w_lo = vld1q_s8(ptr.add(boff + 2) as *const i8);
        let w_hi = vld1q_s8(ptr.add(boff + 18) as *const i8);
        let x_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let mut p = vdupq_n_s32(0);
        p = vdotq_s32(p, w_lo, x_lo);
        p = vdotq_s32(p, w_hi, x_hi);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d * q8[bi].d);
    }

    vaddvq_f32(vaddq_f32(sumv0, sumv1))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn accumulate_q8_0_block(
    sum: float32x4_t,
    block: *const u8,
    x_lo: int8x16_t,
    x_hi: int8x16_t,
    x_scale: f32,
) -> float32x4_t {
    let mut dot = vdupq_n_s32(0);
    dot = vdotq_s32(dot, vld1q_s8(block.add(2) as *const i8), x_lo);
    dot = vdotq_s32(dot, vld1q_s8(block.add(18) as *const i8), x_hi);
    let scale = f16_to_f32(read_f16_le(block)) * x_scale;
    vmlaq_n_f32(sum, vcvtq_f32_s32(dot), scale)
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q8_0_q8_neon_rows4(rows: [&[u8]; 4], q8: &[Q8Block], n_blocks: usize) -> [f32; 4] {
    let ptr0 = rows[0].as_ptr();
    let ptr1 = rows[1].as_ptr();
    let ptr2 = rows[2].as_ptr();
    let ptr3 = rows[3].as_ptr();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for (bi, x) in q8.iter().take(n_blocks).enumerate() {
        let x_lo = vld1q_s8(x.qs.as_ptr());
        let x_hi = vld1q_s8(x.qs.as_ptr().add(16));
        let boff = bi * 34;
        sum0 = accumulate_q8_0_block(sum0, ptr0.add(boff), x_lo, x_hi, x.d);
        sum1 = accumulate_q8_0_block(sum1, ptr1.add(boff), x_lo, x_hi, x.d);
        sum2 = accumulate_q8_0_block(sum2, ptr2.add(boff), x_lo, x_hi, x.d);
        sum3 = accumulate_q8_0_block(sum3, ptr3.add(boff), x_lo, x_hi, x.d);
    }

    [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ]
}

#[target_feature(enable = "neon,i8mm")]
unsafe fn dot_q8_0_q8_i8mm_4x2(
    rows: [&[u8]; 4],
    x0: &[Q8Block],
    x1: &[Q8Block],
    n_blocks: usize,
) -> [f32; 8] {
    let ptr0 = rows[0].as_ptr();
    let ptr1 = rows[1].as_ptr();
    let ptr2 = rows[2].as_ptr();
    let ptr3 = rows[3].as_ptr();
    let mut sum01 = vdupq_n_f32(0.0);
    let mut sum23 = vdupq_n_f32(0.0);

    for bi in 0..n_blocks {
        let boff = bi * 34;
        let blocks = [
            ptr0.add(boff),
            ptr1.add(boff),
            ptr2.add(boff),
            ptr3.add(boff),
        ];
        let mut dot01 = vdupq_n_s32(0);
        let mut dot23 = vdupq_n_s32(0);
        for ki in 0..4usize {
            let off = ki * 8;
            let activations = vcombine_s8(
                vld1_s8(x0[bi].qs.as_ptr().add(off)),
                vld1_s8(x1[bi].qs.as_ptr().add(off)),
            );
            dot01 = vmmlaq_s32(
                dot01,
                vcombine_s8(
                    vld1_s8(blocks[0].add(2 + off) as *const i8),
                    vld1_s8(blocks[1].add(2 + off) as *const i8),
                ),
                activations,
            );
            dot23 = vmmlaq_s32(
                dot23,
                vcombine_s8(
                    vld1_s8(blocks[2].add(2 + off) as *const i8),
                    vld1_s8(blocks[3].add(2 + off) as *const i8),
                ),
                activations,
            );
        }
        let d = [
            f16_to_f32(read_f16_le(blocks[0])),
            f16_to_f32(read_f16_le(blocks[1])),
            f16_to_f32(read_f16_le(blocks[2])),
            f16_to_f32(read_f16_le(blocks[3])),
        ];
        let scales01 = [
            d[0] * x0[bi].d,
            d[0] * x1[bi].d,
            d[1] * x0[bi].d,
            d[1] * x1[bi].d,
        ];
        let scales23 = [
            d[2] * x0[bi].d,
            d[2] * x1[bi].d,
            d[3] * x0[bi].d,
            d[3] * x1[bi].d,
        ];
        sum01 = vfmaq_f32(sum01, vcvtq_f32_s32(dot01), vld1q_f32(scales01.as_ptr()));
        sum23 = vfmaq_f32(sum23, vcvtq_f32_s32(dot23), vld1q_f32(scales23.as_ptr()));
    }

    [
        vgetq_lane_f32::<0>(sum01),
        vgetq_lane_f32::<1>(sum01),
        vgetq_lane_f32::<2>(sum01),
        vgetq_lane_f32::<3>(sum01),
        vgetq_lane_f32::<0>(sum23),
        vgetq_lane_f32::<1>(sum23),
        vgetq_lane_f32::<2>(sum23),
        vgetq_lane_f32::<3>(sum23),
    ]
}

/// Q8_0 x Q8 dot using pre-expanded f32 scales for the weight row.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_0_q8_neon_f32_scales(
    row_bytes: &[u8],
    row_scales: &[f32],
    q8: &[Q8Block],
    n_blocks: usize,
) -> f32 {
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let pairs = n_blocks / 2;
    let ptr = row_bytes.as_ptr();

    for i in 0..pairs {
        let bi = i * 2;
        let boff0 = bi * 34;
        let boff1 = boff0 + 34;

        if bi + 3 < n_blocks {
            prefetch_l1(ptr.add(boff0 + 68));
            prefetch_l1(ptr.add(boff0 + 68 + 34));
        }

        let w0_lo = vld1q_s8(ptr.add(boff0 + 2) as *const i8);
        let w1_lo = vld1q_s8(ptr.add(boff1 + 2) as *const i8);
        let x0_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x1_lo = vld1q_s8(q8[bi + 1].qs.as_ptr());

        let mut p0 = vdupq_n_s32(0);
        p0 = vdotq_s32(p0, w0_lo, x0_lo);
        let mut p1 = vdupq_n_s32(0);
        p1 = vdotq_s32(p1, w1_lo, x1_lo);

        let w0_hi = vld1q_s8(ptr.add(boff0 + 18) as *const i8);
        let w1_hi = vld1q_s8(ptr.add(boff1 + 18) as *const i8);
        let x0_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let x1_hi = vld1q_s8(q8[bi + 1].qs.as_ptr().add(16));

        p0 = vdotq_s32(p0, w0_hi, x0_hi);
        p1 = vdotq_s32(p1, w1_hi, x1_hi);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), row_scales[bi] * q8[bi].d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), row_scales[bi + 1] * q8[bi + 1].d);
    }

    if n_blocks & 1 != 0 {
        let bi = n_blocks - 1;
        let boff = bi * 34;
        let w_lo = vld1q_s8(ptr.add(boff + 2) as *const i8);
        let w_hi = vld1q_s8(ptr.add(boff + 18) as *const i8);
        let x_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let mut p = vdupq_n_s32(0);
        p = vdotq_s32(p, w_lo, x_lo);
        p = vdotq_s32(p, w_hi, x_hi);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), row_scales[bi] * q8[bi].d);
    }

    vaddvq_f32(vaddq_f32(sumv0, sumv1))
}

/// Q8_0 row-pair x Q8 using i8mm.
///
/// Computes two independent row dots while loading each activation chunk once.
#[target_feature(enable = "neon,i8mm")]
unsafe fn dot_q8_0_q8_i8mm_pair(
    row0: &[u8],
    row1: &[u8],
    q8: &[Q8Block],
    n_blocks: usize,
) -> [f32; 2] {
    let ptr0 = row0.as_ptr();
    let ptr1 = row1.as_ptr();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 34;
        if bi + 2 < n_blocks {
            prefetch_l1(ptr0.add(boff + 68));
            prefetch_l1(ptr1.add(boff + 68));
        }

        let d0 = f16_to_f32(read_f16_le(ptr0.add(boff)));
        let d1 = f16_to_f32(read_f16_le(ptr1.add(boff)));
        let x = &q8[bi];
        let mut acc = vdupq_n_s32(0);

        for ki in 0..4usize {
            let off = ki * 8;
            let w = vcombine_s8(
                vld1_s8(ptr0.add(boff + 2 + off) as *const i8),
                vld1_s8(ptr1.add(boff + 2 + off) as *const i8),
            );
            let xs = vcombine_s8(vld1_s8(x.qs[off..].as_ptr()), vdup_n_s8(0));
            acc = vmmlaq_s32(acc, w, xs);
        }

        sum0 += vgetq_lane_s32::<0>(acc) as f32 * d0 * x.d;
        sum1 += vgetq_lane_s32::<2>(acc) as f32 * d1 * x.d;
    }

    [sum0, sum1]
}

pub fn pack_q8_0_row_pairs(bytes: &[u8], rows: usize, bytes_per_row: usize) -> Vec<u8> {
    let row_pairs = rows.div_ceil(2);
    let n_blocks = bytes_per_row / 34;
    let mut packed = vec![0u8; row_pairs * n_blocks * 68];

    for rp in 0..row_pairs {
        let row0 = rp * 2;
        let row1 = row0 + 1;
        for bi in 0..n_blocks {
            let dst = (rp * n_blocks + bi) * 68;
            let src0 = row0 * bytes_per_row + bi * 34;
            packed[dst..dst + 2].copy_from_slice(&bytes[src0..src0 + 2]);
            if row1 < rows {
                let src1 = row1 * bytes_per_row + bi * 34;
                packed[dst + 2..dst + 4].copy_from_slice(&bytes[src1..src1 + 2]);
                for ki in 0..4usize {
                    let off = ki * 8;
                    packed[dst + 4 + ki * 16..dst + 4 + ki * 16 + 8]
                        .copy_from_slice(&bytes[src0 + 2 + off..src0 + 2 + off + 8]);
                    packed[dst + 4 + ki * 16 + 8..dst + 4 + ki * 16 + 16]
                        .copy_from_slice(&bytes[src1 + 2 + off..src1 + 2 + off + 8]);
                }
            } else {
                for ki in 0..4usize {
                    let off = ki * 8;
                    packed[dst + 4 + ki * 16..dst + 4 + ki * 16 + 8]
                        .copy_from_slice(&bytes[src0 + 2 + off..src0 + 2 + off + 8]);
                }
            }
        }
    }

    packed
}

#[target_feature(enable = "neon,i8mm")]
unsafe fn dot_q8_0_packed_i8mm_pair(
    pair_blocks: &[u8],
    q8: &[Q8Block],
    n_blocks: usize,
) -> [f32; 2] {
    let ptr = pair_blocks.as_ptr();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 68;
        if bi + 2 < n_blocks {
            prefetch_l1(ptr.add(boff + 136));
        }
        let d0 = f16_to_f32(read_f16_le(ptr.add(boff)));
        let d1 = f16_to_f32(read_f16_le(ptr.add(boff + 2)));
        let x = &q8[bi];
        let mut acc = vdupq_n_s32(0);
        for ki in 0..4usize {
            let off = ki * 8;
            let w = vld1q_s8(ptr.add(boff + 4 + ki * 16) as *const i8);
            let xs = vcombine_s8(vld1_s8(x.qs[off..].as_ptr()), vdup_n_s8(0));
            acc = vmmlaq_s32(acc, w, xs);
        }
        sum0 += vgetq_lane_s32::<0>(acc) as f32 * d0 * x.d;
        sum1 += vgetq_lane_s32::<2>(acc) as f32 * d1 * x.d;
    }

    [sum0, sum1]
}

pub fn gemv_q8_0_packed_i8mm(
    packed: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let n_blocks = cols / 32;
    let row_pairs = rows.div_ceil(2);
    let n_threads = rayon::current_num_threads().max(1);
    let pair_chunk = parallel_chunk_len(row_pairs, n_threads);

    output[..rows]
        .par_chunks_mut(pair_chunk * 2)
        .enumerate()
        .for_each(|(ci, out)| {
            let pair_start = ci * pair_chunk;
            let pair_count = out.len().div_ceil(2);
            for local_pair in 0..pair_count {
                let rp = pair_start + local_pair;
                if rp >= row_pairs {
                    break;
                }
                let pair_bytes = &packed[rp * n_blocks * 68..(rp + 1) * n_blocks * 68];
                let pair = unsafe { dot_q8_0_packed_i8mm_pair(pair_bytes, q8, n_blocks) };
                let out_base = local_pair * 2;
                out[out_base] = pair[0];
                if out_base + 1 < out.len() && rp * 2 + 1 < rows {
                    out[out_base + 1] = pair[1];
                }
            }
        });
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q8_0_tile8_neon(
    tile_blocks: &[u8],
    q8: &[Q8Block],
    n_blocks: usize,
    out: &mut [f32; 8],
) {
    const TILE_ROWS: usize = 8;
    const TILE_BLOCK_BYTES: usize = 272;
    let ptr = tile_blocks.as_ptr();
    let mut sum = [0.0f32; TILE_ROWS];

    for bi in 0..n_blocks {
        let boff = bi * TILE_BLOCK_BYTES;
        if bi + 2 < n_blocks {
            prefetch_l1(ptr.add(boff + TILE_BLOCK_BYTES * 2));
        }
        let x_lo = vld1q_s8(q8[bi].qs.as_ptr());
        let x_hi = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let xd = q8[bi].d;
        let qptr = ptr.add(boff + TILE_ROWS * 2);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);

        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr as *const i8), x_lo, 0);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(16) as *const i8), x_lo, 0);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(32) as *const i8), x_lo, 1);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(48) as *const i8), x_lo, 1);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(64) as *const i8), x_lo, 2);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(80) as *const i8), x_lo, 2);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(96) as *const i8), x_lo, 3);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(112) as *const i8), x_lo, 3);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(128) as *const i8), x_hi, 0);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(144) as *const i8), x_hi, 0);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(160) as *const i8), x_hi, 1);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(176) as *const i8), x_hi, 1);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(192) as *const i8), x_hi, 2);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(208) as *const i8), x_hi, 2);
        acc0 = vdotq_laneq_s32(acc0, vld1q_s8(qptr.add(224) as *const i8), x_hi, 3);
        acc1 = vdotq_laneq_s32(acc1, vld1q_s8(qptr.add(240) as *const i8), x_hi, 3);

        sum[0] += vgetq_lane_s32::<0>(acc0) as f32 * f16_to_f32(read_f16_le(ptr.add(boff))) * xd;
        sum[1] +=
            vgetq_lane_s32::<1>(acc0) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 2))) * xd;
        sum[2] +=
            vgetq_lane_s32::<2>(acc0) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 4))) * xd;
        sum[3] +=
            vgetq_lane_s32::<3>(acc0) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 6))) * xd;
        sum[4] +=
            vgetq_lane_s32::<0>(acc1) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 8))) * xd;
        sum[5] +=
            vgetq_lane_s32::<1>(acc1) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 10))) * xd;
        sum[6] +=
            vgetq_lane_s32::<2>(acc1) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 12))) * xd;
        sum[7] +=
            vgetq_lane_s32::<3>(acc1) as f32 * f16_to_f32(read_f16_le(ptr.add(boff + 14))) * xd;
    }

    out.copy_from_slice(&sum);
}

pub fn gemv_q8_0_tile8_neon(
    packed: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const TILE_ROWS: usize = 8;
    const TILE_BLOCK_BYTES: usize = 272;
    let n_blocks = cols / 32;
    let row_tiles = rows.div_ceil(TILE_ROWS);
    let n_threads = rayon::current_num_threads().max(1);
    let tile_chunk = parallel_chunk_len(row_tiles, n_threads);

    output[..rows]
        .par_chunks_mut(tile_chunk * TILE_ROWS)
        .enumerate()
        .for_each(|(ci, out)| {
            let tile_start = ci * tile_chunk;
            let tile_count = out.len().div_ceil(TILE_ROWS);
            for local_tile in 0..tile_count {
                let tile = tile_start + local_tile;
                if tile >= row_tiles {
                    break;
                }
                let tile_bytes = &packed
                    [tile * n_blocks * TILE_BLOCK_BYTES..(tile + 1) * n_blocks * TILE_BLOCK_BYTES];
                let mut vals = [0.0f32; TILE_ROWS];
                unsafe { dot_q8_0_tile8_neon(tile_bytes, q8, n_blocks, &mut vals) };
                let out_base = local_tile * TILE_ROWS;
                let copy = (out.len() - out_base)
                    .min(TILE_ROWS)
                    .min(rows - tile * TILE_ROWS);
                out[out_base..out_base + copy].copy_from_slice(&vals[..copy]);
            }
        });
}

#[inline(always)]
fn extract_q4_k_scales_mins(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales_bytes[j] & 63;
        mn[j] = scales_bytes[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
        mn[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
    }
    (sc, mn)
}

pub fn dot_q4_k_q8k_scalar(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;

    for (bi, q8b) in q8k.iter().take(n_blocks).enumerate() {
        let boff = bi * 144;
        let d = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff))) };
        let dmin = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 2))) };
        let (sc, mn) = extract_q4_k_scales_mins(&row_bytes[boff + 4..boff + 16]);
        let qs = &row_bytes[boff + 16..boff + 144];

        let mut sumi = 0i32;
        let mut summ = 0i32;
        for group in 0..4 {
            let q_off = group * 32;
            let x_off = group * 64;
            let is = group * 2;
            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            for l in 0..32 {
                let lo = (qs[q_off + l] & 0x0F) as i8;
                let hi = (qs[q_off + l] >> 4) as i8;
                isum0 += lo as i32 * q8b.qs[x_off + l] as i32;
                isum1 += hi as i32 * q8b.qs[x_off + 32 + l] as i32;
            }
            sumi += sc[is] as i32 * isum0 + sc[is + 1] as i32 * isum1;
            summ += mn[is] as i32 * q8b.bsum32(group * 2) as i32
                + mn[is + 1] as i32 * q8b.bsum32(group * 2 + 1) as i32;
        }
        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }

    acc
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

/// Q4_K x Q8K dot product, NEON DOTPROD path aligned with GGML's
/// `ggml_vec_dot_q4_K_q8_K` (`ggml-cpu/arch/arm/quants.c`).
///
/// Differences from `dot_q4_k_q8k_neon` (the original optimized path):
/// 1. **Scalar i32 sumi1/sumi2** per sub-block reduce (matches GGML), not
///    vector i32x4 accumulator.
/// 2. **dmin term added separately** (`sumf -= dmin * summ`) instead of
///    folded into `q8b.d * (d * sumi - dmin * summ)`.
/// 3. **per-block d scaling**: `sumf += d * (sumi1 + sumi2)` where
///    `d = y[i].d * x[i].d`, matching GGML order of f32 ops.
///
/// Used to recover token-identical behavior with llama.cpp ARM NEON output
/// at the cost of slightly more cross-lane reduces. Opt-in via
/// `RNB_DOT_Q4K_GGML_ALIGN=1`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8k_neon_ggml_align(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
) -> f32 {
    let mut sumf = 0.0f32;
    let mask_low = vdupq_n_u8(0x0F);
    for bi in 0..n_blocks {
        let boff = bi * 144;
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 144);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
        }
        let base = row_bytes.as_ptr().add(boff);
        let (d_x, dmin_x) = f16_pair_to_f32(base);
        let sb = base.add(4);
        let qs = base.add(16);

        let q8b = q8k.get_unchecked(bi);
        let d = q8b.d * d_x;
        let dmin = q8b.d * dmin_x;

        // sub-block scales / mins (GGML utmp packing pattern)
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..4 {
            sc[j] = *sb.add(j) & 63;
            mn[j] = *sb.add(j + 4) & 63;
        }
        for j in 4..8 {
            sc[j] = (*sb.add(j + 4) & 0x0F) | ((*sb.add(j - 4) >> 6) << 4);
            mn[j] = (*sb.add(j + 4) >> 4) | ((*sb.add(j) >> 6) << 4);
        }

        // dmin term — q8sums × mins (separate accumulation, matches GGML)
        let mut summ: i32 = 0;
        for j in 0..8 {
            summ += mn[j] as i32 * q8b.bsum32(j) as i32;
        }
        sumf -= dmin * summ as f32;

        // Main term — scalar sumi1 (lower 4-bit) + sumi2 (upper 4-bit)
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        for g in 0..4 {
            let qbytes_lo = vld1q_u8(qs.add(g * 32));
            let qbytes_hi = vld1q_u8(qs.add(g * 32 + 16));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 48));

            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_lo_0, x_lo_0);
            p1 = vdotq_s32(p1, w_lo_1, x_lo_1);
            sumi1 += vaddvq_s32(p1) * sc[2 * g] as i32;

            let mut p2 = vdupq_n_s32(0);
            p2 = vdotq_s32(p2, w_hi_0, x_hi_0);
            p2 = vdotq_s32(p2, w_hi_1, x_hi_1);
            sumi2 += vaddvq_s32(p2) * sc[2 * g + 1] as i32;
        }

        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

/// Two output rows × one Q8K activation. The pair keeps the weight streams
/// close enough for mobile DRAM prefetch while halving repeated Q8K loads.
/// Each row retains GGML's integer and floating-point reduction order.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4_k_q8k_neon_ggml_align_rows2(
    row_bytes: [&[u8]; 2],
    q8k: &[Q8KBlock],
    n_blocks: usize,
) -> [f32; 2] {
    let mut sumf = [0.0f32; 2];
    let mask_low = vdupq_n_u8(0x0F);

    for bi in 0..n_blocks {
        let boff = bi * 144;
        let q8b = q8k.get_unchecked(bi);
        let mut scales = [[0u8; 8]; 2];
        let mut d = [0.0f32; 2];

        for row in 0..2 {
            if bi + 1 < n_blocks {
                let next = row_bytes[row].as_ptr().add(boff + 144);
                prefetch_l1(next);
                prefetch_l1(next.add(64));
                prefetch_l1(next.add(128));
            }

            let base = row_bytes[row].as_ptr().add(boff);
            let (d_x, dmin_x) = f16_pair_to_f32(base);
            let sb = base.add(4);
            let mut mins = [0u8; 8];
            for j in 0..4 {
                scales[row][j] = *sb.add(j) & 63;
                mins[j] = *sb.add(j + 4) & 63;
            }
            for j in 4..8 {
                scales[row][j] = (*sb.add(j + 4) & 0x0F) | ((*sb.add(j - 4) >> 6) << 4);
                mins[j] = (*sb.add(j + 4) >> 4) | ((*sb.add(j) >> 6) << 4);
            }

            let mut summ = 0i32;
            for j in 0..8 {
                summ += mins[j] as i32 * q8b.bsum32(j) as i32;
            }
            d[row] = q8b.d * d_x;
            sumf[row] -= q8b.d * dmin_x * summ as f32;
        }

        let mut sumi1 = [0i32; 2];
        let mut sumi2 = [0i32; 2];
        for g in 0..4 {
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 48));

            for row in 0..2 {
                let qs = row_bytes[row].as_ptr().add(boff + 16);
                let qbytes_lo = vld1q_u8(qs.add(g * 32));
                let qbytes_hi = vld1q_u8(qs.add(g * 32 + 16));
                let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
                let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
                let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
                let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));

                let mut p1 = vdupq_n_s32(0);
                p1 = vdotq_s32(p1, w_lo_0, x_lo_0);
                p1 = vdotq_s32(p1, w_lo_1, x_lo_1);
                sumi1[row] += vaddvq_s32(p1) * scales[row][2 * g] as i32;

                let mut p2 = vdupq_n_s32(0);
                p2 = vdotq_s32(p2, w_hi_0, x_hi_0);
                p2 = vdotq_s32(p2, w_hi_1, x_hi_1);
                sumi2[row] += vaddvq_s32(p2) * scales[row][2 * g + 1] as i32;
            }
        }

        for row in 0..2 {
            sumf[row] += d[row] * (sumi1[row] + sumi2[row]) as f32;
        }
    }

    sumf
}

#[inline]
fn q4k_decode_rows2_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("RNB_AARCH64_Q4K_DECODE_ROWS2")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    });
    *ENABLED
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q4_k_decode_rows2_chunk(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    out: &mut [f32],
    start: usize,
    bytes_per_row: usize,
    n_blocks: usize,
) {
    let mut i = 0usize;
    while i + 1 < out.len() {
        let row = start + i;
        let row_bytes: [&[u8]; 2] = std::array::from_fn(|offset| {
            let begin = (row + offset) * bytes_per_row;
            &bytes[begin..begin + bytes_per_row]
        });
        let values = dot_q4_k_q8k_neon_ggml_align_rows2(row_bytes, q8k, n_blocks);
        out[i..i + 2].copy_from_slice(&values);
        i += 2;
    }

    if i < out.len() {
        let row = start + i;
        let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
        out[i] = dot_q4_k_q8k_neon_ggml_align(row_bytes, q8k, n_blocks);
    }
}

#[inline]
fn dot_q4k_use_ggml_align() -> bool {
    // Default ON in this prototype branch — `RNB_DOT_Q4K_GGML_ALIGN=0` (or
    // explicit empty value via env in some shells) opts back into the
    // legacy vector-acc path. mc71 finding: production uses GGML reduction
    // tree alignment by default to make token-identical with llama.cpp the
    // baseline assumption.
    static GGML_ALIGN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GGML_ALIGN.get_or_init(|| {
        std::env::var("RNB_DOT_Q4K_GGML_ALIGN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Q4_K x Q8K integer dot product with NEON vdotq_s32.
/// Each Q4_K block = 256 elements with 8 sub-blocks (32 elements each).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    if dot_q4k_use_ggml_align() {
        return dot_q4_k_q8k_neon_ggml_align(row_bytes, q8k, n_blocks);
    }
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 144;

        // Prefetch next weight block (144 bytes = 3 cache lines)
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 144);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
        }

        let base = row_bytes.as_ptr().add(boff);
        let (d, dmin) = f16_pair_to_f32(base);
        let sb = base.add(4); // scales_bytes ptr (12 bytes)
        let qs = base.add(16); // quantized nibbles ptr (128 bytes)

        // Extract 8 sub-block scales and mins (6-bit packed) — raw ptr, no bounds check
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..4 {
            sc[j] = *sb.add(j) & 63;
            mn[j] = *sb.add(j + 4) & 63;
        }
        for j in 4..8 {
            sc[j] = (*sb.add(j + 4) & 0x0F) | ((*sb.add(j - 4) >> 6) << 4);
            mn[j] = (*sb.add(j + 4) >> 4) | ((*sb.add(j) >> 6) << 4);
        }

        let q8b = q8k.get_unchecked(bi);
        let mask_low = vdupq_n_u8(0x0F);
        // Vector accumulator: avoid cross-lane vaddvq_s32 per group (8→1 reduces)
        let mut acc_v = vdupq_n_s32(0);
        let mut summ = 0i32;

        // 4 groups of 64 elements (2 sub-blocks each) — fully unrolled
        // Group 0
        {
            let qbytes_lo = vld1q_u8(qs);
            let qbytes_hi = vld1q_u8(qs.add(16));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr());
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(48));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[0] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[1] as i32);
            summ += mn[0] as i32 * q8b.bsum32(0) as i32 + mn[1] as i32 * q8b.bsum32(1) as i32;
        }
        // Group 1
        {
            let qbytes_lo = vld1q_u8(qs.add(32));
            let qbytes_hi = vld1q_u8(qs.add(48));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(80));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(96));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(112));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[2] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[3] as i32);
            summ += mn[2] as i32 * q8b.bsum32(2) as i32 + mn[3] as i32 * q8b.bsum32(3) as i32;
        }
        // Group 2
        {
            let qbytes_lo = vld1q_u8(qs.add(64));
            let qbytes_hi = vld1q_u8(qs.add(80));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(128));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(144));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(160));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(176));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[4] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[5] as i32);
            summ += mn[4] as i32 * q8b.bsum32(4) as i32 + mn[5] as i32 * q8b.bsum32(5) as i32;
        }
        // Group 3
        {
            let qbytes_lo = vld1q_u8(qs.add(96));
            let qbytes_hi = vld1q_u8(qs.add(112));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(192));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(208));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(224));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(240));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[6] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[7] as i32);
            summ += mn[6] as i32 * q8b.bsum32(6) as i32 + mn[7] as i32 * q8b.bsum32(7) as i32;
        }

        let sumi = vaddvq_s32(acc_v);
        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }
    acc
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8k_neon_meta(
    meta_bytes: &[u8],
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
) -> f32 {
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 144;
        let moff = bi * META_Q4K_BLOCK_BYTES;

        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 144);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
        }

        let qs = row_bytes.as_ptr().add(boff + 16);
        let meta = meta_bytes.as_ptr().add(moff);
        let d = f32::from_bits((meta.add(MPK_D_OFF) as *const u32).read_unaligned());
        let dmin = f32::from_bits((meta.add(MPK_DMIN_OFF) as *const u32).read_unaligned());
        let sc = std::slice::from_raw_parts(meta.add(MPK_SC_OFF), 8);
        let mn = std::slice::from_raw_parts(meta.add(MPK_MN_OFF), 8);

        let q8b = q8k.get_unchecked(bi);
        let mask_low = vdupq_n_u8(0x0F);
        let mut acc_v = vdupq_n_s32(0);
        let mut summ = 0i32;

        {
            let qbytes_lo = vld1q_u8(qs);
            let qbytes_hi = vld1q_u8(qs.add(16));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr());
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(48));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[0] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[1] as i32);
            summ += mn[0] as i32 * q8b.bsum32(0) as i32 + mn[1] as i32 * q8b.bsum32(1) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(32));
            let qbytes_hi = vld1q_u8(qs.add(48));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(80));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(96));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(112));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[2] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[3] as i32);
            summ += mn[2] as i32 * q8b.bsum32(2) as i32 + mn[3] as i32 * q8b.bsum32(3) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(64));
            let qbytes_hi = vld1q_u8(qs.add(80));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(128));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(144));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(160));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(176));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[4] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[5] as i32);
            summ += mn[4] as i32 * q8b.bsum32(4) as i32 + mn[5] as i32 * q8b.bsum32(5) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(96));
            let qbytes_hi = vld1q_u8(qs.add(112));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(192));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(208));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(224));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(240));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[6] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[7] as i32);
            summ += mn[6] as i32 * q8b.bsum32(6) as i32 + mn[7] as i32 * q8b.bsum32(7) as i32;
        }

        let sumi = vaddvq_s32(acc_v);
        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }

    acc
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8k_neon_raw_meta_interleaved(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
) -> f32 {
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * crate::gemm::pack_q4k::Q4K_RAW_META_BLOCK_BYTES;

        if bi + 1 < n_blocks {
            let next = row_bytes
                .as_ptr()
                .add(boff + crate::gemm::pack_q4k::Q4K_RAW_META_BLOCK_BYTES);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
        }

        let base = row_bytes.as_ptr().add(boff);
        let qs = base;
        let meta = base.add(crate::gemm::pack_q4k::Q4K_RAW_META_QS_BYTES);
        let d = f32::from_bits((meta.add(MPK_D_OFF) as *const u32).read_unaligned());
        let dmin = f32::from_bits((meta.add(MPK_DMIN_OFF) as *const u32).read_unaligned());
        let sc = std::slice::from_raw_parts(meta.add(MPK_SC_OFF), 8);
        let mn = std::slice::from_raw_parts(meta.add(MPK_MN_OFF), 8);

        let q8b = q8k.get_unchecked(bi);
        let mask_low = vdupq_n_u8(0x0F);
        let mut acc_v = vdupq_n_s32(0);
        let mut summ = 0i32;

        {
            let qbytes_lo = vld1q_u8(qs);
            let qbytes_hi = vld1q_u8(qs.add(16));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr());
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(48));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[0] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[1] as i32);
            summ += mn[0] as i32 * q8b.bsum32(0) as i32 + mn[1] as i32 * q8b.bsum32(1) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(32));
            let qbytes_hi = vld1q_u8(qs.add(48));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(80));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(96));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(112));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[2] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[3] as i32);
            summ += mn[2] as i32 * q8b.bsum32(2) as i32 + mn[3] as i32 * q8b.bsum32(3) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(64));
            let qbytes_hi = vld1q_u8(qs.add(80));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(128));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(144));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(160));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(176));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[4] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[5] as i32);
            summ += mn[4] as i32 * q8b.bsum32(4) as i32 + mn[5] as i32 * q8b.bsum32(5) as i32;
        }
        {
            let qbytes_lo = vld1q_u8(qs.add(96));
            let qbytes_hi = vld1q_u8(qs.add(112));
            let w_lo_0 = vreinterpretq_s8_u8(vandq_u8(qbytes_lo, mask_low));
            let w_lo_1 = vreinterpretq_s8_u8(vandq_u8(qbytes_hi, mask_low));
            let w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_lo, 4));
            let w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(qbytes_hi, 4));
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(192));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(208));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(224));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(240));
            let mut p0 = vdupq_n_s32(0);
            p0 = vdotq_s32(p0, w_lo_0, x_lo_0);
            p0 = vdotq_s32(p0, w_lo_1, x_lo_1);
            let mut p1 = vdupq_n_s32(0);
            p1 = vdotq_s32(p1, w_hi_0, x_hi_0);
            p1 = vdotq_s32(p1, w_hi_1, x_hi_1);
            acc_v = vmlaq_n_s32(acc_v, p0, sc[6] as i32);
            acc_v = vmlaq_n_s32(acc_v, p1, sc[7] as i32);
            summ += mn[6] as i32 * q8b.bsum32(6) as i32 + mn[7] as i32 * q8b.bsum32(7) as i32;
        }

        let sumi = vaddvq_s32(acc_v);
        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }

    acc
}

pub fn gemv_q4_k_int8_raw_meta_interleaved(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    let n_blocks = cols / 256;
    let bytes_per_row = n_blocks * crate::gemm::pack_q4k::Q4K_RAW_META_BLOCK_BYTES;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);

    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                let bptr = bytes.as_ptr();
                for i in 0..out.len() {
                    let row = start + i;
                    if i + 2 < out.len() {
                        let far = bptr.wrapping_add((row + 2) * bytes_per_row);
                        unsafe {
                            let mut off = 0;
                            while off < bytes_per_row {
                                prefetch_l1(far.add(off));
                                off += 64;
                            }
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = unsafe { dot_q4_k_q8k_neon_raw_meta_interleaved(rb, q8k, n_blocks) };
                }
            });
    } else {
        let out_addr = output.as_mut_ptr() as usize;
        (0..rows).into_par_iter().for_each(|row| {
            let out_ptr = out_addr as *mut f32;
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..seq_len {
                let q8k_s = &q8k[s * n_blocks..(s + 1) * n_blocks];
                unsafe {
                    *out_ptr.add(s * rows + row) =
                        dot_q4_k_q8k_neon_raw_meta_interleaved(rb, q8k_s, n_blocks);
                }
            }
        });
    }
}

pub fn gemv_q4_k_int8_meta(
    meta_bytes: &[u8],
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let meta_bytes_per_row = n_blocks * META_Q4K_BLOCK_BYTES;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);

    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                let bptr = bytes.as_ptr();
                for i in 0..out.len() {
                    let row = start + i;
                    if i + 2 < out.len() {
                        let far = bptr.wrapping_add((row + 2) * bytes_per_row);
                        unsafe {
                            let mut off = 0;
                            while off < bytes_per_row {
                                prefetch_l1(far.add(off));
                                off += 64;
                            }
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    let mb = &meta_bytes[row * meta_bytes_per_row..(row + 1) * meta_bytes_per_row];
                    out[i] = unsafe { dot_q4_k_q8k_neon_meta(mb, rb, q8k, n_blocks) };
                }
            });
    } else {
        gemv_q4_k_int8(bytes, q8k, output, rows, cols, seq_len, bytes_per_row);
    }
}

/// Q5_K x Q8K: fully NEON vectorized 5-bit extraction + vdotq_s32 dot product.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q5_k_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    let m4b = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x01);
    let mtwo = vdupq_n_u8(0x02);
    let vzero = vdupq_n_s32(0);

    for bi in 0..n_blocks {
        let boff = bi * 176;

        // Prefetch next Q5_K block (176 bytes = 3 cache lines)
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 176);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
        }

        let base = row_bytes.as_ptr().add(boff);
        let (d, dmin) = f16_pair_to_f32(base);
        let scales_bytes = &row_bytes[boff + 4..boff + 16];
        let qh_base = base.add(16);
        let qs_base = base.add(48);

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..4 {
            sc[j] = scales_bytes[j] & 63;
            mn[j] = scales_bytes[j + 4] & 63;
        }
        for j in 4..8 {
            sc[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
            mn[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
        }

        let q8b = &q8k[bi];
        let mut sumi = 0i32;
        let mut summ = 0i32;

        // Load 32 bytes qh (256 high bits, consumed 2 bits per iteration)
        let mut qh0 = vld1q_u8(qh_base);
        let mut qh1 = vld1q_u8(qh_base.add(16));

        for group in 0..4 {
            let q_off = group * 32;
            let x_off = group * 64;
            let is = group * 2;

            // Load 32 bytes qs (64 nibbles)
            let ql0 = vld1q_u8(qs_base.add(q_off));
            let ql1 = vld1q_u8(qs_base.add(q_off + 16));

            // Extract high bits: bit0 → position 4, bit1 → position 4
            let h0 = vshlq_n_u8(vandq_u8(qh0, mone), 4);
            let h1 = vshlq_n_u8(vandq_u8(qh1, mone), 4);
            let h2 = vshlq_n_u8(vandq_u8(qh0, mtwo), 3);
            let h3 = vshlq_n_u8(vandq_u8(qh1, mtwo), 3);
            qh0 = vshrq_n_u8(qh0, 2);
            qh1 = vshrq_n_u8(qh1, 2);

            // Low nibble | high bit = 5-bit unsigned (0-31)
            let v0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h0));
            let v1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h1));
            // High nibble | high bit
            let v2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), h2));
            let v3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), h3));

            // Load Q8K input
            let x0 = vld1q_s8(q8b.qs.as_ptr().add(x_off));
            let x1 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 16));
            let x2 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 32));
            let x3 = vld1q_s8(q8b.qs.as_ptr().add(x_off + 48));

            // Chained dot: 32 elements per result
            let p0 = vaddvq_s32(vdotq_s32(vdotq_s32(vzero, v0, x0), v1, x1));
            let p1 = vaddvq_s32(vdotq_s32(vdotq_s32(vzero, v2, x2), v3, x3));

            sumi += sc[is] as i32 * p0 + sc[is + 1] as i32 * p1;
            summ += mn[is] as i32 * q8b.bsum32(group * 2) as i32
                + mn[is + 1] as i32 * q8b.bsum32(group * 2 + 1) as i32;
        }

        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }
    acc
}

/// Q5_K x Q8K scalar fallback.
pub fn dot_q5_k_q8k_scalar(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 176;
        let d = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff))) };
        let dmin = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 2))) };
        let scales_bytes = &row_bytes[boff + 4..boff + 16];
        let qh = &row_bytes[boff + 16..boff + 48];
        let qs = &row_bytes[boff + 48..boff + 176];

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..4 {
            sc[j] = scales_bytes[j] & 63;
            mn[j] = scales_bytes[j + 4] & 63;
        }
        for j in 4..8 {
            sc[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
            mn[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
        }

        let q8b = &q8k[bi];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut w_buf = [0i8; 64];

        for group in 0..4 {
            let q_off = group * 32;
            let x_off = group * 64;
            let is = group * 2;

            for l in 0..32 {
                let h1: u8 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let h2: u8 = if qh[l] & u2 != 0 { 16 } else { 0 };
                w_buf[l] = ((qs[q_off + l] & 0xF) + h1) as i8;
                w_buf[32 + l] = ((qs[q_off + l] >> 4) + h2) as i8;
            }

            let mut isum_lo = 0i32;
            let mut isum_hi = 0i32;
            for l in 0..32 {
                isum_lo += w_buf[l] as i32 * q8b.qs[x_off + l] as i32;
                isum_hi += w_buf[32 + l] as i32 * q8b.qs[x_off + 32 + l] as i32;
            }

            sumi += sc[is] as i32 * isum_lo + sc[is + 1] as i32 * isum_hi;
            summ += mn[is] as i32 * q8b.bsum32(group * 2) as i32
                + mn[is + 1] as i32 * q8b.bsum32(group * 2 + 1) as i32;

            u1 <<= 2;
            u2 <<= 2;
        }

        acc += q8b.d * (d * sumi as f32 - dmin * summ as f32);
    }
    acc
}

/// Unpack Q5_K row + dot with multiple Q8K inputs (GEMM inner kernel).
/// Q5_K: 5-bit values = low 4-bit (ql) + high 1-bit (qh), 6-bit packed scales/mins.
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q5_k_row(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
    seq_len: usize,
    out_ptr: *mut f32,
    row: usize,
    rows: usize,
) {
    // Stack buffers sized for ONE chunk (≤24 blocks = 6144 cols). Rows whose K
    // exceeds 24 blocks (e.g. GLM MLA o_proj: K=32768 = 128 blocks) are
    // processed in chunks of 24: unpack each chunk's weights once, dot against
    // every token, and accumulate the chunk's partial result into out_ptr. With
    // n_blocks ≤ 24 there is exactly one chunk and the arithmetic is identical
    // to the pre-chunking path.
    const CHUNK_BLOCKS: usize = 24;
    let mut unpacked = [0i8; CHUNK_BLOCKS * 256];
    let mut d_arr = [0.0f32; CHUNK_BLOCKS];
    let mut dmin_arr = [0.0f32; CHUNK_BLOCKS];
    let mut sc_arr = [[0u8; 8]; CHUNK_BLOCKS];
    let mut mn_arr = [[0u8; 8]; CHUNK_BLOCKS];

    let m4b = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x01);
    let mtwo = vdupq_n_u8(0x02);
    let vzero = vdupq_n_s32(0);

    let mut blk_start = 0usize;
    while blk_start < n_blocks {
        let chunk_n = (n_blocks - blk_start).min(CHUNK_BLOCKS);
        let is_first_chunk = blk_start == 0;

        // 1) Unpack this chunk's blocks ONCE
        for ci in 0..chunk_n {
            let bi = blk_start + ci;
            let boff = bi * 176;

            // Prefetch next Q5_K weight block
            if bi + 1 < n_blocks {
                let next = row_bytes.as_ptr().add(boff + 176);
                prefetch_l1(next);
                prefetch_l1(next.add(64));
                prefetch_l1(next.add(128));
            }

            let base5 = row_bytes.as_ptr().add(boff);
            let (d5, dmin5) = f16_pair_to_f32(base5);
            d_arr[ci] = d5;
            dmin_arr[ci] = dmin5;
            let scales_bytes = &row_bytes[boff + 4..boff + 16];
            let qh_base = base5.add(16);
            let qs_base = base5.add(48);

            for j in 0..4 {
                sc_arr[ci][j] = scales_bytes[j] & 63;
                mn_arr[ci][j] = scales_bytes[j + 4] & 63;
            }
            for j in 4..8 {
                sc_arr[ci][j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
                mn_arr[ci][j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
            }

            let uout_ptr = unpacked.as_mut_ptr().add(ci * 256);
            let mut qh0 = vld1q_u8(qh_base);
            let mut qh1 = vld1q_u8(qh_base.add(16));

            for group in 0..4 {
                let q_off = group * 32;
                let o_off = group * 64;
                let ql0 = vld1q_u8(qs_base.add(q_off));
                let ql1 = vld1q_u8(qs_base.add(q_off + 16));

                // 5-bit = low 4 bits | high 1 bit << 4
                let h0 = vshlq_n_u8(vandq_u8(qh0, mone), 4);
                let h1 = vshlq_n_u8(vandq_u8(qh1, mone), 4);
                let h2 = vshlq_n_u8(vandq_u8(qh0, mtwo), 3);
                let h3 = vshlq_n_u8(vandq_u8(qh1, mtwo), 3);
                qh0 = vshrq_n_u8(qh0, 2);
                qh1 = vshrq_n_u8(qh1, 2);

                // Low nibble | high bit (unsigned 0-31)
                vst1q_s8(
                    uout_ptr.add(o_off),
                    vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h0)),
                );
                vst1q_s8(
                    uout_ptr.add(o_off + 16),
                    vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h1)),
                );
                // High nibble | high bit
                vst1q_s8(
                    uout_ptr.add(o_off + 32),
                    vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), h2)),
                );
                vst1q_s8(
                    uout_ptr.add(o_off + 48),
                    vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), h3)),
                );
            }
        }

        // 2) Dot this chunk with each token, accumulating into out_ptr
        for s in 0..seq_len {
            let q8k_s = &q8k[s * n_blocks..];
            // Prefetch next token's first Q8K block of this chunk
            if s + 1 < seq_len {
                let next_q8k = &q8k[(s + 1) * n_blocks + blk_start];
                prefetch_l1(next_q8k.qs.as_ptr() as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(64) as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(128) as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(192) as *const u8);
            }
            let mut acc = 0.0f32;
            for ci in 0..chunk_n {
                let bi = blk_start + ci;
                let q8b = &q8k_s[bi];
                let w_ptr = unpacked.as_ptr().add(ci * 256);
                let mut sumi = 0i32;
                let mut summ = 0i32;
                for group in 0..4 {
                    let w_off = group * 64;
                    let x_off = group * 64;
                    let is = group * 2;
                    let p0 = vaddvq_s32(vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w_ptr.add(w_off)),
                            vld1q_s8(q8b.qs.as_ptr().add(x_off)),
                        ),
                        vld1q_s8(w_ptr.add(w_off + 16)),
                        vld1q_s8(q8b.qs.as_ptr().add(x_off + 16)),
                    ));
                    let p1 = vaddvq_s32(vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w_ptr.add(w_off + 32)),
                            vld1q_s8(q8b.qs.as_ptr().add(x_off + 32)),
                        ),
                        vld1q_s8(w_ptr.add(w_off + 48)),
                        vld1q_s8(q8b.qs.as_ptr().add(x_off + 48)),
                    ));
                    sumi += sc_arr[ci][is] as i32 * p0 + sc_arr[ci][is + 1] as i32 * p1;
                    summ += mn_arr[ci][is] as i32 * q8b.bsum32(group * 2) as i32
                        + mn_arr[ci][is + 1] as i32 * q8b.bsum32(group * 2 + 1) as i32;
                }
                acc += q8b.d * (d_arr[ci] * sumi as f32 - dmin_arr[ci] * summ as f32);
            }
            if is_first_chunk {
                *out_ptr.add(s * rows + row) = acc;
            } else {
                *out_ptr.add(s * rows + row) += acc;
            }
        }

        blk_start += CHUNK_BLOCKS;
    }
}

pub fn gemv_q5_k_int8(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let use_neon = std::arch::is_aarch64_feature_detected!("dotprod");
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);
    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    if i + 1 < out.len() {
                        let next_ptr = bytes.as_ptr().wrapping_add((row + 1) * bytes_per_row);
                        unsafe {
                            prefetch_l1(next_ptr);
                            prefetch_l1(next_ptr.add(64));
                            prefetch_l1(next_ptr.add(128));
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = if use_neon {
                        unsafe { dot_q5_k_q8k_neon(rb, q8k, n_blocks) }
                    } else {
                        dot_q5_k_q8k_scalar(rb, q8k, n_blocks)
                    };
                }
            });
    } else {
        let out_addr = output.as_mut_ptr() as usize;
        let use_neon_local = use_neon;
        (0..rows).into_par_iter().for_each(move |row| {
            let out_ptr = out_addr as *mut f32;
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            if use_neon_local {
                unsafe { gemm_q5_k_row(rb, q8k, n_blocks, seq_len, out_ptr, row, rows) };
            } else {
                for s in 0..seq_len {
                    let q8k_s = &q8k[s * n_blocks..(s + 1) * n_blocks];
                    unsafe {
                        *out_ptr.add(s * rows + row) = dot_q5_k_q8k_scalar(rb, q8k_s, n_blocks)
                    };
                }
            }
        });
    }
}

#[inline]
fn dot_q6k_use_ggml_align() -> bool {
    static GGML_ALIGN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GGML_ALIGN.get_or_init(|| {
        std::env::var("RNB_DOT_Q6K_GGML_ALIGN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Q6_K × Q8K dot product, NEON DOTPROD path aligned with GGML
/// `ggml_vec_dot_q6_K_q8_K` (`ggml-cpu/arch/arm/quants.c`).
///
/// mc71 found a layout mismatch: our old `Q8KBlock` had 8 per-32-element
/// bsums while GGML Q8_K has 16 per-16-element. mc72 widened our bsums to
/// 16 entries (`activation_q8.rs`), so this function can now reconstruct
/// `isum_mins = sum_{j=0..16} bsums[j] * scales[j]` exactly the way GGML
/// does — no more best-effort halving.
///
/// Difference from `dot_q6_k_q8k_neon`: GGML does **not** subtract m32s
/// per-byte inside the NEON path; instead it accumulates the unsigned
/// `isum` and `isum_mins` (q8sums × q6scales) and folds the offset at the
/// end as `sum += d * q8d * (isum - 32 * isum_mins)`. Opt-in via
/// `RNB_DOT_Q6K_GGML_ALIGN=1`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k_q8k_neon_ggml_align(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let m4b = vdupq_n_u8(0x0F);
    let m2b = vdupq_n_u8(0x03);
    let vzero = vdupq_n_s32(0);

    for bi in 0..n_blocks {
        let boff = bi * 210;
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 210);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
            prefetch_l1(next.add(192));
        }
        let d_all = f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 208)));
        let q8b = &q8k[bi];
        let mut ql_ptr = row_bytes.as_ptr().add(boff);
        let mut qh_ptr = row_bytes.as_ptr().add(boff + 128);
        let mut sc_ptr = row_bytes.as_ptr().add(boff + 192);
        let mut q8_ptr = q8b.qs.as_ptr();

        // mc72 — Q8KBlock now has per-16-element bsums (16 entries), matching
        // the GGML Q8_K layout. We can compute `isum_mins` exactly the way
        // ggml_vec_dot_q6_K_q8_K does: per-sub-block bsum × Q6_K scale.
        let mut isum_mins: i32 = 0;
        for j in 0..16 {
            let s = *sc_ptr.add(j) as i8 as i32;
            isum_mins += q8b.bsums[j] as i32 * s;
        }

        let mut isum: i32 = 0;
        for _ in 0..2 {
            let qh0 = vld1q_u8(qh_ptr);
            let qh1 = vld1q_u8(qh_ptr.add(16));
            qh_ptr = qh_ptr.add(32);
            let ql0 = vld1q_u8(ql_ptr);
            let ql1 = vld1q_u8(ql_ptr.add(16));
            let ql2 = vld1q_u8(ql_ptr.add(32));
            let ql3 = vld1q_u8(ql_ptr.add(48));
            ql_ptr = ql_ptr.add(64);

            // Unsigned q6 bytes (NO sub32 — GGML defers to outer fold)
            let h0 = vshlq_n_u8(vandq_u8(qh0, m2b), 4);
            let h1 = vshlq_n_u8(vandq_u8(qh1, m2b), 4);
            let h2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), m2b), 4);
            let h3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), m2b), 4);

            let v0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h0));
            let v1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h1));
            let v2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, m4b), h2));
            let v3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, m4b), h3));

            isum +=
                vaddvq_s32(vdotq_s32(vzero, v0, vld1q_s8(q8_ptr))) * *sc_ptr.add(0) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v1, vld1q_s8(q8_ptr.add(16))))
                * *sc_ptr.add(1) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v2, vld1q_s8(q8_ptr.add(32))))
                * *sc_ptr.add(2) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v3, vld1q_s8(q8_ptr.add(48))))
                * *sc_ptr.add(3) as i8 as i32;
            q8_ptr = q8_ptr.add(64);
            sc_ptr = sc_ptr.add(4);

            let h0 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), m2b), 4);
            let h1 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), m2b), 4);
            let h2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 6), m2b), 4);
            let h3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 6), m2b), 4);
            let v0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), h0));
            let v1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), h1));
            let v2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), h2));
            let v3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), h3));

            isum +=
                vaddvq_s32(vdotq_s32(vzero, v0, vld1q_s8(q8_ptr))) * *sc_ptr.add(0) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v1, vld1q_s8(q8_ptr.add(16))))
                * *sc_ptr.add(1) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v2, vld1q_s8(q8_ptr.add(32))))
                * *sc_ptr.add(2) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v3, vld1q_s8(q8_ptr.add(48))))
                * *sc_ptr.add(3) as i8 as i32;
            q8_ptr = q8_ptr.add(64);
            sc_ptr = sc_ptr.add(4);
        }

        // GGML-style fold: sum += d * q8d * (isum - 32 * isum_mins)
        sum += d_all * q8b.d * (isum - 32 * isum_mins) as f32;
    }
    sum
}

/// Q6_K × Q8K dot product: fully NEON vectorized 6-bit extraction + vdotq_s32.
/// Extracts 6-bit values using NEON bit manipulation (no scalar extraction buffer).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    if dot_q6k_use_ggml_align() {
        return dot_q6_k_q8k_neon_ggml_align(row_bytes, q8k, n_blocks);
    }
    let mut acc = 0.0f32;
    let m4b = vdupq_n_u8(0x0F);
    let m2b = vdupq_n_u8(0x03);
    let sub32 = vdupq_n_s8(32);
    let vzero = vdupq_n_s32(0);

    for bi in 0..n_blocks {
        let boff = bi * 210;

        // Prefetch next Q6_K block (210 bytes = 4 cache lines)
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 210);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
            prefetch_l1(next.add(192));
        }

        let d = f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 208)));
        let q8b = &q8k[bi];
        let mut ql_ptr = row_bytes.as_ptr().add(boff);
        let mut qh_ptr = row_bytes.as_ptr().add(boff + 128);
        let mut sc_ptr = row_bytes.as_ptr().add(boff + 192);
        let mut q8_ptr = q8b.qs.as_ptr();
        let mut isum = 0i32;

        // 2 iterations × 128 elements = 256 total
        for _ in 0..2 {
            // Load 32 bytes qh (high 2-bit parts)
            let qh0 = vld1q_u8(qh_ptr);
            let qh1 = vld1q_u8(qh_ptr.add(16));
            qh_ptr = qh_ptr.add(32);

            // Load 64 bytes ql (low 4-bit nibbles)
            let ql0 = vld1q_u8(ql_ptr);
            let ql1 = vld1q_u8(ql_ptr.add(16));
            let ql2 = vld1q_u8(ql_ptr.add(32));
            let ql3 = vld1q_u8(ql_ptr.add(48));
            ql_ptr = ql_ptr.add(64);

            // --- First 64 elements: ql low nibbles + qh bits[1:0],[3:2] ---
            let h0 = vshlq_n_u8(vandq_u8(qh0, m2b), 4);
            let h1 = vshlq_n_u8(vandq_u8(qh1, m2b), 4);
            let h2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), m2b), 4);
            let h3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), m2b), 4);

            let v0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h0)), sub32);
            let v1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h1)), sub32);
            let v2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, m4b), h2)), sub32);
            let v3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, m4b), h3)), sub32);

            isum +=
                vaddvq_s32(vdotq_s32(vzero, v0, vld1q_s8(q8_ptr))) * *sc_ptr.add(0) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v1, vld1q_s8(q8_ptr.add(16))))
                * *sc_ptr.add(1) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v2, vld1q_s8(q8_ptr.add(32))))
                * *sc_ptr.add(2) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v3, vld1q_s8(q8_ptr.add(48))))
                * *sc_ptr.add(3) as i8 as i32;
            q8_ptr = q8_ptr.add(64);
            sc_ptr = sc_ptr.add(4);

            // --- Second 64 elements: ql high nibbles + qh bits[5:4],[7:6] ---
            let h0 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), m2b), 4);
            let h1 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), m2b), 4);
            let h2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 6), m2b), 4);
            let h3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 6), m2b), 4);

            let v0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), h0)), sub32);
            let v1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), h1)), sub32);
            let v2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), h2)), sub32);
            let v3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), h3)), sub32);

            isum +=
                vaddvq_s32(vdotq_s32(vzero, v0, vld1q_s8(q8_ptr))) * *sc_ptr.add(0) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v1, vld1q_s8(q8_ptr.add(16))))
                * *sc_ptr.add(1) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v2, vld1q_s8(q8_ptr.add(32))))
                * *sc_ptr.add(2) as i8 as i32;
            isum += vaddvq_s32(vdotq_s32(vzero, v3, vld1q_s8(q8_ptr.add(48))))
                * *sc_ptr.add(3) as i8 as i32;
            q8_ptr = q8_ptr.add(64);
            sc_ptr = sc_ptr.add(4);
        }

        acc += d * q8b.d * isum as f32;
    }
    acc
}

/// Q6_K x Q8K scalar integer dot product (fallback).
/// Q6_K has per-16-element scales so vdotq_s32 is not directly applicable.
pub fn dot_q6_k_q8k_scalar(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;

    for bi in 0..n_blocks {
        let boff = bi * 210;
        let ql = &row_bytes[boff..boff + 128];
        let qh = &row_bytes[boff + 128..boff + 192];
        let scales = &row_bytes[boff + 192..boff + 208];
        let d = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 208))) };
        let q8b = &q8k[bi];

        let mut sumi = 0i32;

        // 2 halves x 4 sub-blocks = 8 sub-blocks of 32 elements
        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let x_base = n * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc1 = scales[sc_base + is] as i8 as i32;
                let sc2 = scales[sc_base + is + 2] as i8 as i32;
                let sc3 = scales[sc_base + is + 4] as i8 as i32;
                let sc4 = scales[sc_base + is + 6] as i8 as i32;

                // For Q6_K, the scales vary per 16-element group, so we can't easily
                // use pure vdotq_s32. Instead, accumulate scaled products directly.
                sumi += sc1 * q1 * q8b.qs[x_base + l] as i32;
                sumi += sc2 * q2 * q8b.qs[x_base + l + 32] as i32;
                sumi += sc3 * q3 * q8b.qs[x_base + l + 64] as i32;
                sumi += sc4 * q4 * q8b.qs[x_base + l + 96] as i32;
            }
        }

        acc += d * q8b.d * sumi as f32;
    }
    acc
}

fn gemm_q6_k_row_scalar(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
    seq_len: usize,
    out_ptr: *mut f32,
    row: usize,
    rows: usize,
) {
    let mut unpacked = vec![0i8; n_blocks * 256];
    let mut d_arr = vec![0.0f32; n_blocks];
    let mut sc_arr = vec![[0i8; 16]; n_blocks];

    for bi in 0..n_blocks {
        let boff = bi * 210;
        let ql = &row_bytes[boff..boff + 128];
        let qh = &row_bytes[boff + 128..boff + 192];
        let scales = &row_bytes[boff + 192..boff + 208];
        d_arr[bi] = unsafe { f16_to_f32(read_f16_le(row_bytes.as_ptr().add(boff + 208))) };
        for i in 0..16 {
            sc_arr[bi][i] = scales[i] as i8;
        }

        let ql_arr: &[u8; 128] = ql.try_into().unwrap();
        let qh_arr: &[u8; 64] = qh.try_into().unwrap();
        let mut block = [0i8; 256];
        crate::gemm::pack_q6k::unpack_q6k(ql_arr, qh_arr, &mut block);
        unpacked[bi * 256..(bi + 1) * 256].copy_from_slice(&block);
    }

    for s in 0..seq_len {
        let q8k_s = &q8k[s * n_blocks..(s + 1) * n_blocks];
        let mut acc = 0.0f32;
        for bi in 0..n_blocks {
            let w = &unpacked[bi * 256..(bi + 1) * 256];
            let q8b = &q8k_s[bi];
            let mut isum = 0i32;
            for sb in 0..16usize {
                let mut dot = 0i32;
                for k in 0..16usize {
                    let idx = sb * 16 + k;
                    dot += w[idx] as i32 * q8b.qs[idx] as i32;
                }
                isum += sc_arr[bi][sb] as i32 * dot;
            }
            acc += d_arr[bi] * q8b.d * isum as f32;
        }
        unsafe {
            *out_ptr.add(s * rows + row) = acc;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q6_k_block_neon(block: *const u8, output: *mut i8) {
    let ql_base = block;
    let qh_base = block.add(128);
    let mask_low = vdupq_n_u8(0x0f);
    let mask_high = vdupq_n_u8(0x03);
    let sub32 = vdupq_n_s8(32);

    for half in 0..2 {
        let qh0 = vld1q_u8(qh_base.add(half * 32));
        let qh1 = vld1q_u8(qh_base.add(half * 32 + 16));
        let ql0 = vld1q_u8(ql_base.add(half * 64));
        let ql1 = vld1q_u8(ql_base.add(half * 64 + 16));
        let ql2 = vld1q_u8(ql_base.add(half * 64 + 32));
        let ql3 = vld1q_u8(ql_base.add(half * 64 + 48));
        let base = half * 128;

        let high0 = vshlq_n_u8(vandq_u8(qh0, mask_high), 4);
        let high1 = vshlq_n_u8(vandq_u8(qh1, mask_high), 4);
        let high2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_high), 4);
        let high3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_high), 4);
        vst1q_s8(
            output.add(base),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_low), high0)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 16),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_low), high1)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 32),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, mask_low), high2)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 48),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, mask_low), high3)),
                sub32,
            ),
        );

        let high0 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_high), 4);
        let high1 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_high), 4);
        let high2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 6), mask_high), 4);
        let high3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 6), mask_high), 4);
        vst1q_s8(
            output.add(base + 64),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), high0)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 80),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), high1)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 96),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), high2)),
                sub32,
            ),
        );
        vst1q_s8(
            output.add(base + 112),
            vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), high3)),
                sub32,
            ),
        );
    }
}

/// Unpack Q6_K row + dot with multiple Q8K inputs (GEMM inner kernel).
/// Q6_K: 6-bit values = low 4-bit (ql) + high 2-bit (qh), per-16-elem signed scales.
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_q6_k_row(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
    seq_len: usize,
    out_ptr: *mut f32,
    row: usize,
    rows: usize,
) {
    // Stack buffers: max 48 blocks (12288 cols — Gemma4 ffn_down.weight)
    let mut unpacked = [0i8; 48 * 256];
    let mut d_arr = [0.0f32; 48];
    let mut sc_arr = [[0i8; 16]; 48]; // Q6_K has 16 signed scales per block

    // 1) Unpack all blocks
    for bi in 0..n_blocks {
        let boff = bi * 210;
        if bi + 1 < n_blocks {
            let next = row_bytes.as_ptr().add(boff + 210);
            prefetch_l1(next);
            prefetch_l1(next.add(64));
            prefetch_l1(next.add(128));
            prefetch_l1(next.add(192));
        }

        let block = row_bytes.as_ptr().add(boff);
        d_arr[bi] = f16_to_f32(read_f16_le(block.add(208)));
        let sc_base = block.add(192);
        for i in 0..16 {
            sc_arr[bi][i] = *sc_base.add(i) as i8;
        }
        unpack_q6_k_block_neon(block, unpacked.as_mut_ptr().add(bi * 256));
    }

    // 2) Dot with each token
    let vzero = vdupq_n_s32(0);
    for s in 0..seq_len {
        let q8k_s = &q8k[s * n_blocks..];
        // Prefetch next token's Q8K data
        if s + 1 < seq_len {
            let next_q8k = &q8k[(s + 1) * n_blocks];
            prefetch_l1(next_q8k.qs.as_ptr() as *const u8);
            prefetch_l1(next_q8k.qs.as_ptr().add(64) as *const u8);
            prefetch_l1(next_q8k.qs.as_ptr().add(128) as *const u8);
            prefetch_l1(next_q8k.qs.as_ptr().add(192) as *const u8);
        }
        let mut acc = 0.0f32;
        for bi in 0..n_blocks {
            let q8b = &q8k_s[bi];
            let w_ptr = unpacked.as_ptr().add(bi * 256);
            let mut isum = 0i32;
            // 16 sub-blocks of 16 elements, each with its own scale
            for sb in 0..16 {
                let w_off = sb * 16;
                let p = vaddvq_s32(vdotq_s32(
                    vzero,
                    vld1q_s8(w_ptr.add(w_off)),
                    vld1q_s8(q8b.qs.as_ptr().add(w_off)),
                ));
                isum += sc_arr[bi][sb] as i32 * p;
            }
            acc += d_arr[bi] * q8b.d * isum as f32;
        }
        *out_ptr.add(s * rows + row) = acc;
    }
}

pub fn gemv_q6_k_int8(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let use_neon = std::arch::is_aarch64_feature_detected!("dotprod");
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);
    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for (i, outv) in out.iter_mut().enumerate() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    *outv = if use_neon {
                        unsafe { dot_q6_k_q8k_neon(rb, q8k, n_blocks) }
                    } else {
                        dot_q6_k_q8k_scalar(rb, q8k, n_blocks)
                    };
                }
            });
    } else {
        let out_addr = output.as_mut_ptr() as usize;
        let use_neon_local = use_neon;
        // Default: unpack-once NEON GEMM kernel (fits up to 48 blocks / 12288 cols on stack).
        // `RNB_Q6K_GEMM_SCALAR=1` forces the scalar fallback for debugging / rollback.
        let use_neon_gemm =
            use_neon_local && n_blocks <= 48 && std::env::var("RNB_Q6K_GEMM_SCALAR").is_err();
        (0..rows).into_par_iter().for_each(move |row| {
            let out_ptr = out_addr as *mut f32;
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            if use_neon_local && use_neon_gemm {
                unsafe { gemm_q6_k_row(rb, q8k, n_blocks, seq_len, out_ptr, row, rows) };
            } else {
                gemm_q6_k_row_scalar(rb, q8k, n_blocks, seq_len, out_ptr, row, rows);
            }
        });
    }
}

pub fn gemv_q4_0_int8(
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 32;
    // Chunk size: divide work evenly across cores, min 1 row per chunk
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);

    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    if i + 1 < out.len() {
                        let next_ptr = bytes.as_ptr().wrapping_add((row + 1) * bytes_per_row);
                        unsafe {
                            prefetch_l1(next_ptr);
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = unsafe { dot_q4_0_q8_neon(rb, q8, n_blocks) };
                }
            });
    } else {
        let out_addr = output.as_mut_ptr() as usize;
        (0..rows).into_par_iter().for_each(move |row| {
            let out_ptr = out_addr as *mut f32;
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..seq_len {
                let q8_s = &q8[s * n_blocks..(s + 1) * n_blocks];
                unsafe { *out_ptr.add(s * rows + row) = dot_q4_0_q8_neon(rb, q8_s, n_blocks) };
            }
        });
    }
}

pub fn gemv_q5_0_int8(
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 32;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);

    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                let out_len = out.len();
                for (i, value) in out.iter_mut().enumerate() {
                    let row = start + i;
                    if i + 1 < out_len {
                        let next_ptr = bytes.as_ptr().wrapping_add((row + 1) * bytes_per_row);
                        unsafe {
                            prefetch_l1(next_ptr);
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    *value = unsafe { dot_q5_0_q8_neon(rb, q8, n_blocks) };
                }
            });
    } else {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            let out_addr = output.as_mut_ptr() as usize;
            let row_pairs = rows / 2;
            (0..row_pairs).into_par_iter().for_each(move |pair| {
                let row = pair * 2;
                let out_ptr = out_addr as *mut f32;
                let row0 = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let row1 = &bytes[(row + 1) * bytes_per_row..(row + 2) * bytes_per_row];
                let mut s = 0usize;
                while s + 1 < seq_len {
                    let x0 = &q8[s * n_blocks..(s + 1) * n_blocks];
                    let x1 = &q8[(s + 1) * n_blocks..(s + 2) * n_blocks];
                    let values = unsafe { dot_q5_0_q8_i8mm_2x2(row0, row1, x0, x1, n_blocks) };
                    unsafe {
                        *out_ptr.add(s * rows + row) = values[0];
                        *out_ptr.add((s + 1) * rows + row) = values[1];
                        *out_ptr.add(s * rows + row + 1) = values[2];
                        *out_ptr.add((s + 1) * rows + row + 1) = values[3];
                    }
                    s += 2;
                }
                if s < seq_len {
                    let x = &q8[s * n_blocks..(s + 1) * n_blocks];
                    unsafe {
                        *out_ptr.add(s * rows + row) = dot_q5_0_q8_neon(row0, x, n_blocks);
                        *out_ptr.add(s * rows + row + 1) = dot_q5_0_q8_neon(row1, x, n_blocks);
                    }
                }
            });
            if rows & 1 != 0 {
                let row = rows - 1;
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    let x = &q8[s * n_blocks..(s + 1) * n_blocks];
                    unsafe {
                        *(out_addr as *mut f32).add(s * rows + row) =
                            dot_q5_0_q8_neon(row_bytes, x, n_blocks);
                    }
                }
            }
            return;
        }
        let out_addr = output.as_mut_ptr() as usize;
        let row_tiles = rows / 4;
        (0..row_tiles).into_par_iter().for_each(move |tile| {
            let row = tile * 4;
            let out_ptr = out_addr as *mut f32;
            let row0 = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let row1 = &bytes[(row + 1) * bytes_per_row..(row + 2) * bytes_per_row];
            let row2 = &bytes[(row + 2) * bytes_per_row..(row + 3) * bytes_per_row];
            let row3 = &bytes[(row + 3) * bytes_per_row..(row + 4) * bytes_per_row];
            for s in 0..seq_len {
                let q8_s = &q8[s * n_blocks..(s + 1) * n_blocks];
                let values =
                    unsafe { dot_q5_0_q8_neon_rows4([row0, row1, row2, row3], q8_s, n_blocks) };
                unsafe {
                    *out_ptr.add(s * rows + row) = values[0];
                    *out_ptr.add(s * rows + row + 1) = values[1];
                    *out_ptr.add(s * rows + row + 2) = values[2];
                    *out_ptr.add(s * rows + row + 3) = values[3];
                }
            }
        });
        ((row_tiles * 4)..rows).into_par_iter().for_each(|row| {
            let out_ptr = out_addr as *mut f32;
            let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..seq_len {
                let q8_s = &q8[s * n_blocks..(s + 1) * n_blocks];
                unsafe {
                    *out_ptr.add(s * rows + row) = dot_q5_0_q8_neon(row_bytes, q8_s, n_blocks);
                }
            }
        });
    }
}

pub fn gemv_q8_0_int8(
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 32;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);
    if seq_len == 1 {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("i8mm") && std::env::var("RNB_Q80_I8MM").is_ok()
        {
            output[..rows]
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(ci, out)| {
                    let start = ci * chunk;
                    let mut i = 0usize;
                    while i + 1 < out.len() {
                        let row = start + i;
                        let rb0 = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                        let rb1 = &bytes[(row + 1) * bytes_per_row..(row + 2) * bytes_per_row];
                        let pair = unsafe { dot_q8_0_q8_i8mm_pair(rb0, rb1, q8, n_blocks) };
                        out[i] = pair[0];
                        out[i + 1] = pair[1];
                        i += 2;
                    }
                    if i < out.len() {
                        let row = start + i;
                        let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                        out[i] = unsafe { dot_q8_0_q8_neon(rb, q8, n_blocks) };
                    }
                });
            return;
        }

        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    if i + 1 < out.len() {
                        let next_ptr = bytes.as_ptr().wrapping_add((row + 1) * bytes_per_row);
                        unsafe {
                            prefetch_l1(next_ptr);
                        }
                    }
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = unsafe { dot_q8_0_q8_neon(rb, q8, n_blocks) };
                }
            });
    } else {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            let out_addr = output.as_mut_ptr() as usize;
            let row_tiles = rows / 4;
            (0..row_tiles).into_par_iter().for_each(move |tile| {
                let row = tile * 4;
                let out_ptr = out_addr as *mut f32;
                let tile_rows = [
                    &bytes[row * bytes_per_row..(row + 1) * bytes_per_row],
                    &bytes[(row + 1) * bytes_per_row..(row + 2) * bytes_per_row],
                    &bytes[(row + 2) * bytes_per_row..(row + 3) * bytes_per_row],
                    &bytes[(row + 3) * bytes_per_row..(row + 4) * bytes_per_row],
                ];
                let mut s = 0usize;
                while s + 1 < seq_len {
                    let x0 = &q8[s * n_blocks..(s + 1) * n_blocks];
                    let x1 = &q8[(s + 1) * n_blocks..(s + 2) * n_blocks];
                    let values = unsafe { dot_q8_0_q8_i8mm_4x2(tile_rows, x0, x1, n_blocks) };
                    unsafe {
                        for tile_row in 0..4 {
                            *out_ptr.add(s * rows + row + tile_row) = values[tile_row * 2];
                            *out_ptr.add((s + 1) * rows + row + tile_row) =
                                values[tile_row * 2 + 1];
                        }
                    }
                    s += 2;
                }
                if s < seq_len {
                    let x = &q8[s * n_blocks..(s + 1) * n_blocks];
                    let values = unsafe { dot_q8_0_q8_neon_rows4(tile_rows, x, n_blocks) };
                    unsafe {
                        for tile_row in 0..4 {
                            *out_ptr.add(s * rows + row + tile_row) = values[tile_row];
                        }
                    }
                }
            });
            ((row_tiles * 4)..rows).into_par_iter().for_each(|row| {
                let out_ptr = out_addr as *mut f32;
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    let x = &q8[s * n_blocks..(s + 1) * n_blocks];
                    unsafe {
                        *out_ptr.add(s * rows + row) = dot_q8_0_q8_neon(row_bytes, x, n_blocks);
                    }
                }
            });
            return;
        }
        let out_addr = output.as_mut_ptr() as usize;
        let row_tiles = rows / 4;
        (0..row_tiles).into_par_iter().for_each(move |tile| {
            let row = tile * 4;
            let out_ptr = out_addr as *mut f32;
            let row0 = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let row1 = &bytes[(row + 1) * bytes_per_row..(row + 2) * bytes_per_row];
            let row2 = &bytes[(row + 2) * bytes_per_row..(row + 3) * bytes_per_row];
            let row3 = &bytes[(row + 3) * bytes_per_row..(row + 4) * bytes_per_row];
            for s in 0..seq_len {
                let q8_s = &q8[s * n_blocks..(s + 1) * n_blocks];
                let values =
                    unsafe { dot_q8_0_q8_neon_rows4([row0, row1, row2, row3], q8_s, n_blocks) };
                unsafe {
                    *out_ptr.add(s * rows + row) = values[0];
                    *out_ptr.add(s * rows + row + 1) = values[1];
                    *out_ptr.add(s * rows + row + 2) = values[2];
                    *out_ptr.add(s * rows + row + 3) = values[3];
                }
            }
        });
        ((row_tiles * 4)..rows).into_par_iter().for_each(|row| {
            let out_ptr = out_addr as *mut f32;
            let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..seq_len {
                let q8_s = &q8[s * n_blocks..(s + 1) * n_blocks];
                unsafe {
                    *out_ptr.add(s * rows + row) = dot_q8_0_q8_neon(row_bytes, q8_s, n_blocks);
                }
            }
        });
    }
}

pub fn gemv_q8_0_int8_f32_scales(
    bytes: &[u8],
    scales: &[f32],
    q8: &[Q8Block],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 32;
    assert!(
        scales.len() >= rows * n_blocks,
        "Q8_0 scale sidecar too small"
    );
    if seq_len != 1 {
        gemv_q8_0_int8(bytes, q8, output, rows, cols, seq_len, bytes_per_row);
        return;
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk = parallel_chunk_len(rows, n_threads);
    output[..rows]
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, out)| {
            let start = ci * chunk;
            for i in 0..out.len() {
                let row = start + i;
                if i + 1 < out.len() {
                    let next_ptr = bytes.as_ptr().wrapping_add((row + 1) * bytes_per_row);
                    unsafe {
                        prefetch_l1(next_ptr);
                    }
                }
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let row_scales = &scales[row * n_blocks..(row + 1) * n_blocks];
                out[i] = unsafe { dot_q8_0_q8_neon_f32_scales(rb, row_scales, q8, n_blocks) };
            }
        });
}

/// Unpack Q4_K row + dot with multiple Q8K inputs (GEMM inner kernel).
/// Unpacks weight nibbles once, then dots with seq_len input vectors.
/// Writes directly to column-major output: out_ptr[s * rows + row] for each s.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_q4_k_row(
    row_bytes: &[u8],
    q8k: &[Q8KBlock],
    n_blocks: usize,
    seq_len: usize,
    out_ptr: *mut f32,
    row: usize,
    rows: usize,
) {
    // Stack buffers sized for ONE chunk (≤48 blocks = 12288 cols). Rows whose
    // K exceeds 48 blocks (e.g. 27B ffn_gate/up: feed_forward_length=17408 →
    // K=17408 = 68 blocks) are processed in chunks of 48: unpack each chunk's
    // weights once, dot against every token, and accumulate the chunk's partial
    // result into out_ptr. With n_blocks ≤ 48 there is exactly one chunk and the
    // arithmetic is identical to the pre-chunking path (token-identical for 9B).
    const CHUNK_BLOCKS: usize = 48;
    let mut unpacked = [0i8; CHUNK_BLOCKS * 256];
    let mut d_arr = [0.0f32; CHUNK_BLOCKS];
    let mut dmin_arr = [0.0f32; CHUNK_BLOCKS];
    let mut sc_arr = [[0u8; 8]; CHUNK_BLOCKS];
    let mut mn_arr = [[0u8; 8]; CHUNK_BLOCKS];

    let mask_low = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut blk_start = 0usize;
    while blk_start < n_blocks {
        let chunk_n = (n_blocks - blk_start).min(CHUNK_BLOCKS);
        let is_first_chunk = blk_start == 0;

        // 1) Unpack this chunk's blocks ONCE (NEON vectorized) into [0, chunk_n)
        for ci in 0..chunk_n {
            let bi = blk_start + ci;
            let boff = bi * 144;

            // Prefetch next weight block during unpack
            if bi + 1 < n_blocks {
                let next = row_bytes.as_ptr().add(boff + 144);
                prefetch_l1(next);
                prefetch_l1(next.add(64));
                prefetch_l1(next.add(128));
            }

            let base4 = row_bytes.as_ptr().add(boff);
            let (d4, dmin4) = f16_pair_to_f32(base4);
            d_arr[ci] = d4;
            dmin_arr[ci] = dmin4;
            let scales_bytes = &row_bytes[boff + 4..boff + 16];
            let qs_ptr = base4.add(16);

            // Extract scales and mins (scalar, tiny cost)
            for j in 0..4 {
                sc_arr[ci][j] = scales_bytes[j] & 63;
                mn_arr[ci][j] = scales_bytes[j + 4] & 63;
            }
            for j in 4..8 {
                sc_arr[ci][j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
                mn_arr[ci][j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
            }

            // NEON vectorized nibble unpack: 4 groups of 64 elements
            let uout_ptr = unpacked.as_mut_ptr().add(ci * 256);
            for group in 0..4 {
                let q_off = group * 32;
                let o_off = group * 64;
                let qb_lo = vld1q_u8(qs_ptr.add(q_off));
                let qb_hi = vld1q_u8(qs_ptr.add(q_off + 16));
                // Low nibbles → first 32 elements
                vst1q_s8(
                    uout_ptr.add(o_off),
                    vreinterpretq_s8_u8(vandq_u8(qb_lo, mask_low)),
                );
                vst1q_s8(
                    uout_ptr.add(o_off + 16),
                    vreinterpretq_s8_u8(vandq_u8(qb_hi, mask_low)),
                );
                // High nibbles → next 32 elements
                vst1q_s8(
                    uout_ptr.add(o_off + 32),
                    vreinterpretq_s8_u8(vshrq_n_u8(qb_lo, 4)),
                );
                vst1q_s8(
                    uout_ptr.add(o_off + 48),
                    vreinterpretq_s8_u8(vshrq_n_u8(qb_hi, 4)),
                );
            }
        }

        // 2) Dot this chunk with each token, accumulating into out_ptr
        for s in 0..seq_len {
            let q8k_s = &q8k[s * n_blocks..];
            // Prefetch next token's first Q8K block of this chunk
            if s + 1 < seq_len {
                let next_q8k = &q8k[(s + 1) * n_blocks + blk_start];
                prefetch_l1(next_q8k.qs.as_ptr() as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(64) as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(128) as *const u8);
                prefetch_l1(next_q8k.qs.as_ptr().add(192) as *const u8);
            }
            let mut acc = 0.0f32;
            for ci in 0..chunk_n {
                let bi = blk_start + ci;
                let q8b = &q8k_s[bi];
                let w_ptr = unpacked.as_ptr().add(ci * 256);
                let mut sumi = 0i32;
                let mut summ = 0i32;
                for group in 0..4 {
                    let w_off = group * 64;
                    let x_off = group * 64;
                    let is = group * 2;
                    // Sub-block 0: vdotq_s32 (32 elements)
                    let p0 = vaddvq_s32(vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w_ptr.add(w_off)),
                            vld1q_s8(q8b.qs.as_ptr().add(x_off)),
                        ),
                        vld1q_s8(w_ptr.add(w_off + 16)),
                        vld1q_s8(q8b.qs.as_ptr().add(x_off + 16)),
                    ));
                    // Sub-block 1: vdotq_s32 (32 elements)
                    let p1 = vaddvq_s32(vdotq_s32(
                        vdotq_s32(
                            vzero,
                            vld1q_s8(w_ptr.add(w_off + 32)),
                            vld1q_s8(q8b.qs.as_ptr().add(x_off + 32)),
                        ),
                        vld1q_s8(w_ptr.add(w_off + 48)),
                        vld1q_s8(q8b.qs.as_ptr().add(x_off + 48)),
                    ));
                    sumi += sc_arr[ci][is] as i32 * p0 + sc_arr[ci][is + 1] as i32 * p1;
                    summ += mn_arr[ci][is] as i32 * q8b.bsum32(group * 2) as i32
                        + mn_arr[ci][is + 1] as i32 * q8b.bsum32(group * 2 + 1) as i32;
                }
                acc += q8b.d * (d_arr[ci] * sumi as f32 - dmin_arr[ci] * summ as f32);
            }
            if is_first_chunk {
                *out_ptr.add(s * rows + row) = acc;
            } else {
                *out_ptr.add(s * rows + row) += acc;
            }
        }

        blk_start += CHUNK_BLOCKS;
    }
}

struct Q8KI8mmTokenPairs {
    tiles: Vec<i8>,
    bsums: Vec<i32>,
    d: Vec<f32>,
}

/// Reorders Q8K token pairs into native `smmla` right-hand operand tiles and
/// precomputes the token-pair metadata reused by every output-row pair.
#[target_feature(enable = "neon")]
unsafe fn pack_q8k_i8mm_token_pairs(
    q8k: &[Q8KBlock],
    n_blocks: usize,
    seq_len: usize,
) -> Q8KI8mmTokenPairs {
    const PAIR_BLOCK_BYTES: usize = 512;
    let token_pairs = seq_len.div_ceil(2);
    let pair_blocks = token_pairs * n_blocks;
    let mut tiles = vec![0i8; pair_blocks * PAIR_BLOCK_BYTES];
    let mut bsums = vec![0i32; pair_blocks * 8 * 4];
    let mut d = vec![0.0f32; pair_blocks * 4];
    for pair in 0..token_pairs {
        let token0 = pair * 2;
        let token1 = (token0 + 1).min(seq_len - 1);
        for bi in 0..n_blocks {
            let x0 = q8k.get_unchecked(token0 * n_blocks + bi);
            let x1 = q8k.get_unchecked(token1 * n_blocks + bi);
            let pair_block = pair * n_blocks + bi;
            let tile_dst = tiles.as_mut_ptr().add(pair_block * PAIR_BLOCK_BYTES);
            for chunk in 0..32 {
                let off = chunk * 8;
                vst1q_s8(
                    tile_dst.add(chunk * 16),
                    vcombine_s8(
                        vld1_s8(x0.qs.as_ptr().add(off)),
                        vld1_s8(x1.qs.as_ptr().add(off)),
                    ),
                );
            }
            d[pair_block * 4..pair_block * 4 + 4].copy_from_slice(&[x0.d, x1.d, x0.d, x1.d]);
            for sub in 0..8 {
                let bsum0 = x0.bsum32(sub) as i32;
                let bsum1 = x1.bsum32(sub) as i32;
                let offset = (pair_block * 8 + sub) * 4;
                bsums[offset..offset + 4].copy_from_slice(&[bsum0, bsum1, bsum0, bsum1]);
            }
        }
    }
    Q8KI8mmTokenPairs { tiles, bsums, d }
}

/// Sixteen output rows × two input tokens Q4_K GEMM kernel for Arm i8mm.
///
/// Both operands and their scale/min metadata are arranged before the hot
/// token loop. Each inner tile is therefore two aligned loads plus `smmla`,
/// followed by vector scale and min accumulation.
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_q4_k_rows16_i8mm(
    row_bytes: [&[u8]; 16],
    packed_q8k: &Q8KI8mmTokenPairs,
    n_blocks: usize,
    seq_len: usize,
    out_ptr: *mut f32,
    row: usize,
    rows: usize,
) {
    const CHUNK_BLOCKS: usize = 8;
    const PAIR_BLOCK_BYTES: usize = 512;
    const ROW_PAIRS: usize = 8;
    let mut unpacked = [0i8; CHUNK_BLOCKS * PAIR_BLOCK_BYTES * ROW_PAIRS];
    let mut d_arr = [vdupq_n_f32(0.0); CHUNK_BLOCKS * ROW_PAIRS];
    let mut dmin_arr = [vdupq_n_f32(0.0); CHUNK_BLOCKS * ROW_PAIRS];
    let mut scale_arr = [vdupq_n_s32(0); CHUNK_BLOCKS * 8 * ROW_PAIRS];
    let mut min_arr = [vdupq_n_s32(0); CHUNK_BLOCKS * 8 * ROW_PAIRS];
    let mask_low = vdupq_n_u8(0x0F);

    let mut blk_start = 0usize;
    while blk_start < n_blocks {
        let chunk_n = (n_blocks - blk_start).min(CHUNK_BLOCKS);
        let is_first_chunk = blk_start == 0;

        for pair in 0..ROW_PAIRS {
            let pair_row0 = row_bytes[pair * 2];
            let pair_row1 = row_bytes[pair * 2 + 1];
            for ci in 0..chunk_n {
                let bi = blk_start + ci;
                let boff = bi * 144;
                let base0 = pair_row0.as_ptr().add(boff);
                let base1 = pair_row1.as_ptr().add(boff);
                let meta_index = pair * CHUNK_BLOCKS + ci;
                let (d0, dmin0) = f16_pair_to_f32(base0);
                let (d1, dmin1) = f16_pair_to_f32(base1);
                d_arr[meta_index] = pair_dup_f32x4(d0, d1);
                dmin_arr[meta_index] = pair_dup_f32x4(dmin0, dmin1);
                let (sc0, mn0) =
                    extract_q4_k_scales_mins(std::slice::from_raw_parts(base0.add(4), 12));
                let (sc1, mn1) =
                    extract_q4_k_scales_mins(std::slice::from_raw_parts(base1.add(4), 12));
                for sub in 0..8 {
                    let scale_index = meta_index * 8 + sub;
                    scale_arr[scale_index] = pair_dup_i32x4(sc0[sub] as i32, sc1[sub] as i32);
                    min_arr[scale_index] = pair_dup_i32x4(mn0[sub] as i32, mn1[sub] as i32);
                }

                let qx0 = base0.add(16);
                let qx1 = base1.add(16);
                let unpacked_block = unpacked.as_mut_ptr().add(meta_index * PAIR_BLOCK_BYTES);
                for group in 0..4 {
                    let q_off = group * 32;
                    let packed00 = vld1q_u8(qx0.add(q_off));
                    let packed01 = vld1q_u8(qx0.add(q_off + 16));
                    let packed10 = vld1q_u8(qx1.add(q_off));
                    let packed11 = vld1q_u8(qx1.add(q_off + 16));

                    let low00 = vreinterpretq_s8_u8(vandq_u8(packed00, mask_low));
                    let low01 = vreinterpretq_s8_u8(vandq_u8(packed01, mask_low));
                    let low10 = vreinterpretq_s8_u8(vandq_u8(packed10, mask_low));
                    let low11 = vreinterpretq_s8_u8(vandq_u8(packed11, mask_low));
                    let high00 = vreinterpretq_s8_u8(vshrq_n_u8(packed00, 4));
                    let high01 = vreinterpretq_s8_u8(vshrq_n_u8(packed01, 4));
                    let high10 = vreinterpretq_s8_u8(vshrq_n_u8(packed10, 4));
                    let high11 = vreinterpretq_s8_u8(vshrq_n_u8(packed11, 4));

                    let low_dst = unpacked_block.add((group * 2) * 64);
                    vst1q_s8(low_dst, vcombine_s8(vget_low_s8(low00), vget_low_s8(low10)));
                    vst1q_s8(
                        low_dst.add(16),
                        vcombine_s8(vget_high_s8(low00), vget_high_s8(low10)),
                    );
                    vst1q_s8(
                        low_dst.add(32),
                        vcombine_s8(vget_low_s8(low01), vget_low_s8(low11)),
                    );
                    vst1q_s8(
                        low_dst.add(48),
                        vcombine_s8(vget_high_s8(low01), vget_high_s8(low11)),
                    );

                    let high_dst = low_dst.add(64);
                    vst1q_s8(
                        high_dst,
                        vcombine_s8(vget_low_s8(high00), vget_low_s8(high10)),
                    );
                    vst1q_s8(
                        high_dst.add(16),
                        vcombine_s8(vget_high_s8(high00), vget_high_s8(high10)),
                    );
                    vst1q_s8(
                        high_dst.add(32),
                        vcombine_s8(vget_low_s8(high01), vget_low_s8(high11)),
                    );
                    vst1q_s8(
                        high_dst.add(48),
                        vcombine_s8(vget_high_s8(high01), vget_high_s8(high11)),
                    );
                }
            }
        }

        let mut token = 0usize;
        while token < seq_len {
            let token1 = (token + 1).min(seq_len - 1);
            let token_pair = token / 2;
            let mut acc = [vdupq_n_f32(0.0); ROW_PAIRS];

            for ci in 0..chunk_n {
                let bi = blk_start + ci;
                let pair_block = token_pair * n_blocks + bi;
                let activations = packed_q8k.tiles.as_ptr().add(pair_block * PAIR_BLOCK_BYTES);
                let x_d = vld1q_f32(packed_q8k.d.as_ptr().add(pair_block * 4));
                let mut sumi = [vdupq_n_s32(0); ROW_PAIRS];
                let mut summ = [vdupq_n_s32(0); ROW_PAIRS];

                for sub in 0..8 {
                    let off = sub * 64;
                    let activation0 = vld1q_s8(activations.add(off));
                    let activation1 = vld1q_s8(activations.add(off + 16));
                    let activation2 = vld1q_s8(activations.add(off + 32));
                    let activation3 = vld1q_s8(activations.add(off + 48));
                    let bsum = vld1q_s32(packed_q8k.bsums.as_ptr().add((pair_block * 8 + sub) * 4));

                    for pair in 0..ROW_PAIRS {
                        let meta_index = pair * CHUNK_BLOCKS + ci;
                        let weights = unpacked.as_ptr().add(meta_index * PAIR_BLOCK_BYTES + off);
                        let mut dot = vdupq_n_s32(0);
                        dot = vmmlaq_s32(dot, vld1q_s8(weights), activation0);
                        dot = vmmlaq_s32(dot, vld1q_s8(weights.add(16)), activation1);
                        dot = vmmlaq_s32(dot, vld1q_s8(weights.add(32)), activation2);
                        dot = vmmlaq_s32(dot, vld1q_s8(weights.add(48)), activation3);
                        let scale_index = meta_index * 8 + sub;
                        sumi[pair] = vmlaq_s32(sumi[pair], dot, scale_arr[scale_index]);
                        summ[pair] = vmlaq_s32(summ[pair], bsum, min_arr[scale_index]);
                    }
                }

                for pair in 0..ROW_PAIRS {
                    let meta_index = pair * CHUNK_BLOCKS + ci;
                    acc[pair] = vmlaq_f32(
                        acc[pair],
                        vcvtq_f32_s32(sumi[pair]),
                        vmulq_f32(d_arr[meta_index], x_d),
                    );
                    acc[pair] = vmlsq_f32(
                        acc[pair],
                        vcvtq_f32_s32(summ[pair]),
                        vmulq_f32(dmin_arr[meta_index], x_d),
                    );
                }
            }

            for (pair, pair_acc) in acc.into_iter().enumerate() {
                let pair_row = row + pair * 2;
                let values = [
                    vgetq_lane_f32(pair_acc, 0),
                    vgetq_lane_f32(pair_acc, 1),
                    vgetq_lane_f32(pair_acc, 2),
                    vgetq_lane_f32(pair_acc, 3),
                ];
                for (lane, (output_token, output_row)) in [
                    (token, pair_row),
                    (token1, pair_row),
                    (token, pair_row + 1),
                    (token1, pair_row + 1),
                ]
                .into_iter()
                .enumerate()
                {
                    if output_token != token || lane == 0 || lane == 2 {
                        let dst = out_ptr.add(output_token * rows + output_row);
                        if is_first_chunk {
                            *dst = values[lane];
                        } else {
                            *dst += values[lane];
                        }
                    }
                }
            }
            token += 2;
        }

        blk_start += CHUNK_BLOCKS;
    }
}

pub fn gemv_q4_k_int8(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let n_blocks = cols / 256;
    let n_threads = rayon::current_num_threads().max(1);
    let use_neon = std::arch::is_aarch64_feature_detected!("dotprod");
    // Skip rayon for tiny matrices (overhead > compute)
    let chunk = parallel_chunk_len(rows, n_threads);
    let use_rows2 = use_neon && dot_q4k_use_ggml_align() && q4k_decode_rows2_enabled();

    if seq_len == 1 {
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                if use_rows2 {
                    unsafe {
                        gemv_q4_k_decode_rows2_chunk(
                            bytes,
                            q8k,
                            out,
                            start,
                            bytes_per_row,
                            n_blocks,
                        )
                    };
                    return;
                }
                for (i, outv) in out.iter_mut().enumerate() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    *outv = if use_neon {
                        unsafe { dot_q4_k_q8k_neon(rb, q8k, n_blocks) }
                    } else {
                        dot_q4_k_q8k_scalar(rb, q8k, n_blocks)
                    };
                }
            });
    } else if std::arch::is_aarch64_feature_detected!("i8mm") && rows >= 16 {
        let packed_q8k = unsafe { pack_q8k_i8mm_token_pairs(q8k, n_blocks, seq_len) };
        let out_addr = output.as_mut_ptr() as usize;
        let row_tiles = rows / 16;
        (0..row_tiles).into_par_iter().for_each(move |tile| {
            let row = tile * 16;
            let out_ptr = out_addr as *mut f32;
            let row_bytes: [&[u8]; 16] = std::array::from_fn(|offset| {
                let start = (row + offset) * bytes_per_row;
                &bytes[start..start + bytes_per_row]
            });
            unsafe {
                gemm_q4_k_rows16_i8mm(
                    row_bytes,
                    &packed_q8k,
                    n_blocks,
                    seq_len,
                    out_ptr,
                    row,
                    rows,
                )
            };
        });
        for row in row_tiles * 16..rows {
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            unsafe { gemm_q4_k_row(rb, q8k, n_blocks, seq_len, output.as_mut_ptr(), row, rows) };
        }
    } else {
        // True GEMM: unpack weight once per row, write directly column-major (no transpose)
        let out_addr = output.as_mut_ptr() as usize;
        (0..rows).into_par_iter().for_each(move |row| {
            let out_ptr = out_addr as *mut f32;
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            if use_neon {
                unsafe { gemm_q4_k_row(rb, q8k, n_blocks, seq_len, out_ptr, row, rows) };
            } else {
                for s in 0..seq_len {
                    let q8k_s = &q8k[s * n_blocks..(s + 1) * n_blocks];
                    unsafe {
                        *out_ptr.add(s * rows + row) = dot_q4_k_q8k_scalar(rb, q8k_s, n_blocks)
                    };
                }
            }
        });
    }
}

/// Runs two Q4_K projections over the same Q8K activation batch.
///
/// The i8mm path packs each token pair once, then shares that packed operand
/// across both weight matrices.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_k_int8_dual(
    left_bytes: &[u8],
    right_bytes: &[u8],
    q8k: &[Q8KBlock],
    left_output: &mut [f32],
    right_output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    if seq_len > 1 && std::arch::is_aarch64_feature_detected!("i8mm") && rows >= 16 {
        let n_blocks = cols / 256;
        let packed_q8k = unsafe { pack_q8k_i8mm_token_pairs(q8k, n_blocks, seq_len) };
        let left_out_addr = left_output.as_mut_ptr() as usize;
        let right_out_addr = right_output.as_mut_ptr() as usize;
        let row_tiles = rows / 16;
        (0..row_tiles * 2).into_par_iter().for_each(move |work| {
            let right = work >= row_tiles;
            let tile = if right { work - row_tiles } else { work };
            let row = tile * 16;
            let bytes = if right { right_bytes } else { left_bytes };
            let out_ptr = if right {
                right_out_addr as *mut f32
            } else {
                left_out_addr as *mut f32
            };
            let row_bytes: [&[u8]; 16] = std::array::from_fn(|offset| {
                let start = (row + offset) * bytes_per_row;
                &bytes[start..start + bytes_per_row]
            });
            unsafe {
                gemm_q4_k_rows16_i8mm(
                    row_bytes,
                    &packed_q8k,
                    n_blocks,
                    seq_len,
                    out_ptr,
                    row,
                    rows,
                )
            };
        });
        for row in row_tiles * 16..rows {
            let left_row = &left_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let right_row = &right_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            unsafe {
                gemm_q4_k_row(
                    left_row,
                    q8k,
                    n_blocks,
                    seq_len,
                    left_output.as_mut_ptr(),
                    row,
                    rows,
                );
                gemm_q4_k_row(
                    right_row,
                    q8k,
                    n_blocks,
                    seq_len,
                    right_output.as_mut_ptr(),
                    row,
                    rows,
                );
            }
        }
        return;
    }

    gemv_q4_k_int8(
        left_bytes,
        q8k,
        left_output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
    );
    gemv_q4_k_int8(
        right_bytes,
        q8k,
        right_output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn encode_q4_k_scales_mins(sc: [u8; 8], mn: [u8; 8]) -> [u8; 12] {
        let mut out = [0u8; 12];
        for j in 0..4 {
            out[j] = sc[j] & 63;
            out[j + 4] = mn[j] & 63;
        }
        for j in 4..8 {
            out[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
            out[j - 4] |= ((sc[j] >> 4) & 0x03) << 6;
            out[j] |= ((mn[j] >> 4) & 0x03) << 6;
        }
        out
    }

    fn make_q4_k_row(blocks: usize) -> Vec<u8> {
        let mut row = vec![0u8; blocks * 144];
        for bi in 0..blocks {
            let boff = bi * 144;
            let d = f16::from_f32(0.03125 * (bi as f32 + 1.0));
            let dmin = f16::from_f32(0.015625 * (bi as f32 + 1.0));
            row[boff..boff + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            row[boff + 2..boff + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                sc[j] = ((bi * 9 + j * 5 + 3) % 64) as u8;
                mn[j] = ((bi * 7 + j * 11 + 1) % 64) as u8;
            }
            row[boff + 4..boff + 16].copy_from_slice(&encode_q4_k_scales_mins(sc, mn));
            for i in 0..128 {
                let lo = ((bi * 13 + i * 3 + 5) % 16) as u8;
                let hi = ((bi * 17 + i * 7 + 9) % 16) as u8;
                row[boff + 16 + i] = lo | (hi << 4);
            }
        }
        row
    }

    fn make_q5_k_row(blocks: usize) -> Vec<u8> {
        let mut row = vec![0u8; blocks * 176];
        for bi in 0..blocks {
            let boff = bi * 176;
            let d = f16::from_f32(0.03125 * (bi as f32 + 1.0));
            let dmin = f16::from_f32(0.015625 * (bi as f32 + 1.0));
            row[boff..boff + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            row[boff + 2..boff + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                sc[j] = ((bi * 9 + j * 5 + 3) % 64) as u8;
                mn[j] = ((bi * 7 + j * 11 + 1) % 64) as u8;
            }
            row[boff + 4..boff + 16].copy_from_slice(&encode_q4_k_scales_mins(sc, mn));
            for i in 0..32 {
                row[boff + 16 + i] = ((bi * 19 + i * 3 + 7) % 256) as u8;
            }
            for i in 0..128 {
                let lo = ((bi * 13 + i * 3 + 5) % 16) as u8;
                let hi = ((bi * 17 + i * 7 + 9) % 16) as u8;
                row[boff + 48 + i] = lo | (hi << 4);
            }
        }
        row
    }

    fn make_q6_k_row(blocks: usize) -> Vec<u8> {
        let mut row = vec![0u8; blocks * 210];
        for bi in 0..blocks {
            let boff = bi * 210;
            for i in 0..128 {
                row[boff + i] = ((bi * 13 + i * 7 + 5) % 256) as u8;
            }
            for i in 0..64 {
                row[boff + 128 + i] = ((bi * 11 + i * 5 + 17) % 256) as u8;
            }
            for i in 0..16 {
                row[boff + 192 + i] = ((((bi as i32) * 9 + i as i32 * 5) % 63) - 31) as i8 as u8;
            }
            let d = f16::from_f32(0.015625 * (bi as f32 + 1.0));
            row[boff + 208..boff + 210].copy_from_slice(&d.to_bits().to_le_bytes());
        }
        row
    }

    fn exact_q6_k_q8k_dot(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
        let mut acc = 0.0f32;
        for bi in 0..n_blocks {
            let boff = bi * 210;
            let ql = &row_bytes[boff..boff + 128];
            let qh = &row_bytes[boff + 128..boff + 192];
            let scales = &row_bytes[boff + 192..boff + 208];
            let d = f16::from_bits(u16::from_le_bytes([
                row_bytes[boff + 208],
                row_bytes[boff + 209],
            ]))
            .to_f32();
            let q8b = &q8k[bi];

            let mut w = [0i8; 256];
            for n in 0..2usize {
                let ql_base = n * 64;
                let qh_base = n * 32;
                let out_base = n * 128;
                for l in 0..32usize {
                    w[out_base + l] =
                        ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 0x03) << 4)) as i8
                            - 32;
                    w[out_base + l + 32] = ((ql[ql_base + l + 32] & 0x0F)
                        | (((qh[qh_base + l] >> 2) & 0x03) << 4))
                        as i8
                        - 32;
                    w[out_base + l + 64] =
                        ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) as i8
                            - 32;
                    w[out_base + l + 96] = ((ql[ql_base + l + 32] >> 4)
                        | (((qh[qh_base + l] >> 6) & 0x03) << 4))
                        as i8
                        - 32;
                }
            }

            let mut sumi = 0i32;
            for sb in 0..16usize {
                let sc = scales[sb] as i8 as i32;
                let mut dot = 0i32;
                for k in 0..16usize {
                    let idx = sb * 16 + k;
                    dot += w[idx] as i32 * q8b.qs[idx] as i32;
                }
                sumi += sc * dot;
            }
            acc += d * q8b.d * sumi as f32;
        }
        acc
    }

    #[test]
    fn test_parallel_chunk_len_keeps_small_matrices_single_chunk() {
        assert_eq!(parallel_chunk_len(32, 4), 32);
    }

    #[test]
    fn test_parallel_chunk_len_creates_finer_grained_tasks() {
        assert_eq!(parallel_chunk_len(1024, 4), 64);
    }

    fn make_q8k_blocks(blocks: usize) -> Vec<Q8KBlock> {
        let mut out = Vec::with_capacity(blocks);
        for bi in 0..blocks {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for i in 0..256 {
                let q = ((bi as i32 * 19 + i as i32 * 23 + 17) % 255 - 127) as i8;
                qs[i] = q;
                bsums[i / 16] += q as i16;
            }
            out.push(Q8KBlock {
                d: 0.0078125 * (bi as f32 + 1.0),
                qs,
                bsums,
            });
        }
        out
    }

    fn make_q8_blocks(blocks: usize) -> Vec<Q8Block> {
        let mut out = Vec::with_capacity(blocks);
        for bi in 0..blocks {
            let mut qs = [0i8; 32];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((bi as i32 * 19 + i as i32 * 23 + 17) % 255 - 127) as i8;
            }
            out.push(Q8Block {
                d: 0.0078125 * (bi as f32 + 1.0),
                qs,
            });
        }
        out
    }

    fn make_q8_0_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut row = vec![0u8; blocks * 34];
        for bi in 0..blocks {
            let boff = bi * 34;
            let d = f16::from_f32(0.015625 * (seed as f32 + bi as f32 + 1.0));
            row[boff..boff + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            for i in 0..32 {
                row[boff + 2 + i] =
                    ((seed as i32 * 29 + bi as i32 * 17 + i as i32 * 7) % 255 - 127) as i8 as u8;
            }
        }
        row
    }

    fn make_q5_0_row(blocks: usize, seed: usize) -> Vec<u8> {
        let mut row = vec![0u8; blocks * 22];
        for bi in 0..blocks {
            let boff = bi * 22;
            let d = f16::from_f32(0.015625 * (seed as f32 + bi as f32 + 1.0));
            row[boff..boff + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            let qh = (seed as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((bi % 31) as u32);
            row[boff + 2..boff + 6].copy_from_slice(&qh.to_le_bytes());
            for i in 0..16 {
                row[boff + 6 + i] =
                    (seed as u8).wrapping_mul(29) ^ (bi as u8).wrapping_mul(17) ^ (i as u8 * 11);
            }
        }
        row
    }

    fn dot_q5_0_q8_scalar(row: &[u8], q8: &[Q8Block], blocks: usize) -> f32 {
        let mut result = 0.0f32;
        for (bi, q8_block) in q8.iter().take(blocks).enumerate() {
            let boff = bi * 22;
            let d = f16::from_bits(u16::from_le_bytes([row[boff], row[boff + 1]])).to_f32();
            let qh =
                u32::from_le_bytes([row[boff + 2], row[boff + 3], row[boff + 4], row[boff + 5]]);
            let mut sum = 0i32;
            for i in 0..16 {
                let packed = row[boff + 6 + i];
                let low = ((packed & 0x0f) | ((((qh >> i) & 1) as u8) << 4)) as i32 - 16;
                let high = ((packed >> 4) | ((((qh >> (i + 16)) & 1) as u8) << 4)) as i32 - 16;
                sum += low * q8_block.qs[i] as i32;
                sum += high * q8_block.qs[i + 16] as i32;
            }
            result += d * q8_block.d * sum as f32;
        }
        result
    }

    #[test]
    fn q5_0_dotprod_matches_scalar_quantized_input() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let blocks = 22;
        let row = make_q5_0_row(blocks, 7);
        let q8 = make_q8_blocks(blocks);
        let expected = dot_q5_0_q8_scalar(&row, &q8, blocks);
        let actual = unsafe { dot_q5_0_q8_neon(&row, &q8, blocks) };
        let tolerance = expected.abs() * 1.0e-5 + 1.0e-3;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    #[test]
    fn q5_0_batch_writes_seq_major_output() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let rows = 5;
        let blocks = 22;
        let cols = blocks * 32;
        let seq_len = 3;
        let bytes_per_row = blocks * 22;
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            bytes.extend_from_slice(&make_q5_0_row(blocks, row + 3));
        }
        let mut q8 = Vec::with_capacity(seq_len * blocks);
        for seq in 0..seq_len {
            let mut seq_blocks = make_q8_blocks(blocks);
            for block in &mut seq_blocks {
                block.d *= seq as f32 + 1.0;
            }
            q8.extend(seq_blocks);
        }
        let mut actual = vec![0.0f32; seq_len * rows];
        gemv_q5_0_int8(&bytes, &q8, &mut actual, rows, cols, seq_len, bytes_per_row);
        for seq in 0..seq_len {
            let q8_seq = &q8[seq * blocks..(seq + 1) * blocks];
            for row in 0..rows {
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let expected = dot_q5_0_q8_scalar(row_bytes, q8_seq, blocks);
                let tolerance = expected.abs() * 1.0e-5 + 1.0e-3;
                assert!(
                    (actual[seq * rows + row] - expected).abs() <= tolerance,
                    "seq={seq} row={row} actual={} expected={expected} tolerance={tolerance}",
                    actual[seq * rows + row]
                );
            }
        }
    }

    #[test]
    fn q8_0_batch_writes_seq_major_output() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let rows = 5;
        let blocks = 7;
        let cols = blocks * 32;
        let seq_len = 3;
        let bytes_per_row = blocks * 34;
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            bytes.extend_from_slice(&make_q8_0_row(blocks, row + 5));
        }
        let mut q8 = Vec::with_capacity(seq_len * blocks);
        for seq in 0..seq_len {
            let mut seq_blocks = make_q8_blocks(blocks);
            for block in &mut seq_blocks {
                block.d *= seq as f32 + 1.0;
            }
            q8.extend(seq_blocks);
        }
        let mut actual = vec![0.0f32; seq_len * rows];
        gemv_q8_0_int8(&bytes, &q8, &mut actual, rows, cols, seq_len, bytes_per_row);
        for seq in 0..seq_len {
            let q8_seq = &q8[seq * blocks..(seq + 1) * blocks];
            for row in 0..rows {
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let expected = unsafe { dot_q8_0_q8_neon(row_bytes, q8_seq, blocks) };
                let tolerance = expected.abs() * 1.0e-5 + 1.0e-3;
                assert!(
                    (actual[seq * rows + row] - expected).abs() <= tolerance,
                    "seq={seq} row={row} actual={} expected={expected} tolerance={tolerance}",
                    actual[seq * rows + row]
                );
            }
        }
    }

    #[test]
    fn gemm_q4_k_row_handles_k_beyond_48_blocks() {
        // 27B(qwen35 dense)의 ffn_gate/up 은 feed_forward_length=17408 → K=17408
        // = 68 blocks > 기존 48-block 스택 버퍼 한계. d_arr[48] out-of-bounds 재현.
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let n_blocks = 68; // K = 17408
        let seq_len = 3;
        let row = make_q4_k_row(n_blocks);
        let mut q8k: Vec<Q8KBlock> = Vec::new();
        let mut refs: Vec<f32> = Vec::new();
        for _s in 0..seq_len {
            let blk = make_q8k_blocks(n_blocks);
            refs.push(dot_q4_k_q8k_scalar(&row, &blk, n_blocks));
            q8k.extend(blk);
        }
        let rows = 1usize;
        let mut out = vec![0.0f32; seq_len * rows];
        unsafe {
            gemm_q4_k_row(&row, &q8k, n_blocks, seq_len, out.as_mut_ptr(), 0, rows);
        }
        for s in 0..seq_len {
            let diff = (out[s * rows] - refs[s]).abs();
            let tol = refs[s].abs() * 1e-3 + 1e-2;
            assert!(
                diff <= tol,
                "s={s}: gemm={} scalar={} diff={} tol={}",
                out[s * rows],
                refs[s],
                diff,
                tol
            );
        }
    }

    #[test]
    fn gemm_q5_k_row_handles_k_beyond_24_blocks() {
        // GLM-5.2 MLA o_proj 는 K=32768 = 128 blocks > 기존 24-block 스택 버퍼
        // 한계. d_arr[24] out-of-bounds 재현 (batch prefill seq_len>1 경로).
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let n_blocks = 128; // K = 32768
        let seq_len = 3;
        let row = make_q5_k_row(n_blocks);
        let mut q8k: Vec<Q8KBlock> = Vec::new();
        let mut refs: Vec<f32> = Vec::new();
        for _s in 0..seq_len {
            let blk = make_q8k_blocks(n_blocks);
            refs.push(dot_q5_k_q8k_scalar(&row, &blk, n_blocks));
            q8k.extend(blk);
        }
        let rows = 1usize;
        let mut out = vec![0.0f32; seq_len * rows];
        unsafe {
            gemm_q5_k_row(&row, &q8k, n_blocks, seq_len, out.as_mut_ptr(), 0, rows);
        }
        for s in 0..seq_len {
            let diff = (out[s * rows] - refs[s]).abs();
            let tol = refs[s].abs() * 1e-3 + 1e-2;
            assert!(
                diff <= tol,
                "s={s}: gemm={} scalar={} diff={} tol={}",
                out[s * rows],
                refs[s],
                diff,
                tol
            );
        }
    }

    #[test]
    fn q8_0_i8mm_pair_matches_dotprod_rows() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let blocks = 4;
        let q8 = make_q8_blocks(blocks);
        let row0 = make_q8_0_row(blocks, 3);
        let row1 = make_q8_0_row(blocks, 11);

        let expected0 = unsafe { dot_q8_0_q8_neon(&row0, &q8, blocks) };
        let expected1 = unsafe { dot_q8_0_q8_neon(&row1, &q8, blocks) };
        let actual = unsafe { dot_q8_0_q8_i8mm_pair(&row0, &row1, &q8, blocks) };

        assert!(
            (actual[0] - expected0).abs() < 1e-4,
            "row0 actual={} expected={expected0}",
            actual[0]
        );
        assert!(
            (actual[1] - expected1).abs() < 1e-4,
            "row1 actual={} expected={expected1}",
            actual[1]
        );
    }

    #[test]
    fn q8_0_packed_i8mm_gemv_matches_dotprod_rows() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let rows = 5;
        let blocks = 4;
        let cols = blocks * 32;
        let bytes_per_row = blocks * 34;
        let q8 = make_q8_blocks(blocks);
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            bytes.extend_from_slice(&make_q8_0_row(blocks, row + 3));
        }
        let packed = pack_q8_0_row_pairs(&bytes, rows, bytes_per_row);

        let mut actual = vec![0.0f32; rows];
        gemv_q8_0_packed_i8mm(&packed, &q8, &mut actual, rows, cols);

        for row in 0..rows {
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let expected = unsafe { dot_q8_0_q8_neon(rb, &q8, blocks) };
            assert!(
                (actual[row] - expected).abs() < 1e-4,
                "row {row}: actual={} expected={expected}",
                actual[row]
            );
        }
    }

    #[test]
    fn q8_0_tile8_gemv_matches_dotprod_rows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let rows = 11;
        let blocks = 4;
        let cols = blocks * 32;
        let bytes_per_row = blocks * 34;
        let q8 = make_q8_blocks(blocks);
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            bytes.extend_from_slice(&make_q8_0_row(blocks, row + 5));
        }
        let packed = crate::gemm::pack_q8_0_pair::pack_q8_0_tile8(&bytes, rows, cols);

        let mut actual = vec![0.0f32; rows];
        gemv_q8_0_tile8_neon(&packed, &q8, &mut actual, rows, cols);

        for row in 0..rows {
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let expected = unsafe { dot_q8_0_q8_neon(rb, &q8, blocks) };
            assert!(
                (actual[row] - expected).abs() < 1e-4,
                "row {row}: actual={} expected={expected}",
                actual[row]
            );
        }
    }

    #[test]
    fn q4_k_scalar_oracle_matches_decode_wrapper_for_seq1() {
        let rows = 3;
        let cols = 512;
        let blocks = cols / 256;
        let bytes_per_row = blocks * 144;
        let mut bytes = vec![0u8; rows * bytes_per_row];
        for row in 0..rows {
            let row_bytes = make_q4_k_row(blocks);
            bytes[row * bytes_per_row..(row + 1) * bytes_per_row].copy_from_slice(&row_bytes);
        }
        let q8k = make_q8k_blocks(blocks);
        let mut out = vec![0.0f32; rows];

        gemv_q4_k_int8(&bytes, &q8k, &mut out, rows, cols, 1, bytes_per_row);

        for row in 0..rows {
            let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let expected = dot_q4_k_q8k_scalar(rb, &q8k, blocks);
            assert!(
                (out[row] - expected).abs() < 1e-4,
                "row {row}: wrapper={} scalar={expected}",
                out[row]
            );
        }
    }

    #[test]
    fn q4_k_neon_matches_scalar_oracle() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let blocks = 2;
        let row = make_q4_k_row(blocks);
        let q8k = make_q8k_blocks(blocks);
        let scalar = dot_q4_k_q8k_scalar(&row, &q8k, blocks);
        let neon = unsafe { dot_q4_k_q8k_neon(&row, &q8k, blocks) };
        assert!((scalar - neon).abs() < 1e-4, "scalar={scalar} neon={neon}");
    }
    #[test]
    fn q4_k_decode_row_tiles_match_single_row_reduction() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let rows = 5;
        let blocks = 3;
        let cols = blocks * 256;
        let bytes_per_row = blocks * 144;
        let mut left_bytes = Vec::with_capacity(rows * bytes_per_row);
        let mut right_bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            let mut left = make_q4_k_row(blocks);
            let mut right = make_q4_k_row(blocks);
            left[16 + row] ^= (row as u8 + 1) * 3;
            right[48 + row] ^= (row as u8 + 1) * 5;
            left_bytes.extend_from_slice(&left);
            right_bytes.extend_from_slice(&right);
        }
        let q8k = make_q8k_blocks(blocks);
        let expected_left: Vec<f32> = (0..rows)
            .map(|row| {
                let row_bytes = &left_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                unsafe { dot_q4_k_q8k_neon_ggml_align(row_bytes, &q8k, blocks) }
            })
            .collect();
        let expected_right: Vec<f32> = (0..rows)
            .map(|row| {
                let row_bytes = &right_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                unsafe { dot_q4_k_q8k_neon_ggml_align(row_bytes, &q8k, blocks) }
            })
            .collect();

        let row_tile: [&[u8]; 2] =
            std::array::from_fn(|row| &left_bytes[row * bytes_per_row..(row + 1) * bytes_per_row]);
        let tiled = unsafe { dot_q4_k_q8k_neon_ggml_align_rows2(row_tile, &q8k, blocks) };
        for row in 0..2 {
            assert_eq!(
                tiled[row].to_bits(),
                expected_left[row].to_bits(),
                "direct row tile {row}"
            );
        }

        let mut single = vec![0.0f32; rows];
        gemv_q4_k_int8(&left_bytes, &q8k, &mut single, rows, cols, 1, bytes_per_row);
        let mut dual_left = vec![0.0f32; rows];
        let mut dual_right = vec![0.0f32; rows];
        gemv_q4_k_int8_dual(
            &left_bytes,
            &right_bytes,
            &q8k,
            &mut dual_left,
            &mut dual_right,
            rows,
            cols,
            1,
            bytes_per_row,
        );

        for row in 0..rows {
            assert_eq!(
                single[row].to_bits(),
                expected_left[row].to_bits(),
                "single projection row {row}"
            );
            assert_eq!(
                dual_left[row].to_bits(),
                expected_left[row].to_bits(),
                "dual left row {row}"
            );
            assert_eq!(
                dual_right[row].to_bits(),
                expected_right[row].to_bits(),
                "dual right row {row}"
            );
        }
    }

    #[test]
    fn q4_k_i8mm_batch_matches_scalar_oracle() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let rows = 17;
        let seq_len = 3;
        let blocks = 10;
        let cols = blocks * 256;
        let bytes_per_row = blocks * 144;
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            let mut row_bytes = make_q4_k_row(blocks);
            row_bytes[16 + row] ^= (row as u8 + 1) * 3;
            bytes.extend_from_slice(&row_bytes);
        }
        let token_blocks = make_q8k_blocks(blocks);
        let mut q8k = Vec::with_capacity(seq_len * blocks);
        for _ in 0..seq_len {
            q8k.extend_from_slice(&token_blocks);
        }
        let mut actual = vec![0.0f32; rows * seq_len];

        gemv_q4_k_int8(
            &bytes,
            &q8k,
            &mut actual,
            rows,
            cols,
            seq_len,
            bytes_per_row,
        );

        for token in 0..seq_len {
            let q8k_token = &q8k[token * blocks..(token + 1) * blocks];
            for row in 0..rows {
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let expected = dot_q4_k_q8k_scalar(row_bytes, q8k_token, blocks);
                let got = actual[token * rows + row];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "token {token} row {row}: actual={got} expected={expected}",
                );
            }
        }
    }

    #[test]
    fn q4_k_i8mm_dual_batch_matches_scalar_oracle() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let rows = 17;
        let seq_len = 3;
        let blocks = 10;
        let cols = blocks * 256;
        let bytes_per_row = blocks * 144;
        let mut left_bytes = Vec::with_capacity(rows * bytes_per_row);
        let mut right_bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            let mut left_row = make_q4_k_row(blocks);
            let mut right_row = make_q4_k_row(blocks);
            left_row[16 + row] ^= (row as u8 + 1) * 3;
            right_row[48 + row] ^= (row as u8 + 1) * 5;
            left_bytes.extend_from_slice(&left_row);
            right_bytes.extend_from_slice(&right_row);
        }
        let token_blocks = make_q8k_blocks(blocks);
        let mut q8k = Vec::with_capacity(seq_len * blocks);
        for _ in 0..seq_len {
            q8k.extend_from_slice(&token_blocks);
        }
        let mut left_actual = vec![0.0f32; rows * seq_len];
        let mut right_actual = vec![0.0f32; rows * seq_len];

        gemv_q4_k_int8_dual(
            &left_bytes,
            &right_bytes,
            &q8k,
            &mut left_actual,
            &mut right_actual,
            rows,
            cols,
            seq_len,
            bytes_per_row,
        );

        for token in 0..seq_len {
            let q8k_token = &q8k[token * blocks..(token + 1) * blocks];
            for row in 0..rows {
                let left_row = &left_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let right_row = &right_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let expected_left = dot_q4_k_q8k_scalar(left_row, q8k_token, blocks);
                let expected_right = dot_q4_k_q8k_scalar(right_row, q8k_token, blocks);
                let index = token * rows + row;
                assert!(
                    (left_actual[index] - expected_left).abs() < 1e-4,
                    "left token {token} row {row}: actual={} expected={expected_left}",
                    left_actual[index]
                );
                assert!(
                    (right_actual[index] - expected_right).abs() < 1e-4,
                    "right token {token} row {row}: actual={} expected={expected_right}",
                    right_actual[index]
                );
            }
        }
    }

    #[test]
    fn q6_k_batch_matches_exact_oracle() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let rows = 17;
        let seq_len = 3;
        let blocks = 2;
        let cols = blocks * 256;
        let bytes_per_row = blocks * 210;
        let mut bytes = Vec::with_capacity(rows * bytes_per_row);
        for _ in 0..rows {
            bytes.extend_from_slice(&make_q6_k_row(blocks));
        }
        let mut q8k = Vec::with_capacity(seq_len * blocks);
        for token in 0..seq_len {
            let mut token_blocks = make_q8k_blocks(blocks);
            for block in &mut token_blocks {
                block.d *= token as f32 + 1.0;
            }
            q8k.extend_from_slice(&token_blocks);
        }
        let mut actual = vec![0.0f32; rows * seq_len];

        gemv_q6_k_int8(
            &bytes,
            &q8k,
            &mut actual,
            rows,
            cols,
            seq_len,
            bytes_per_row,
        );

        for token in 0..seq_len {
            let q8k_token = &q8k[token * blocks..(token + 1) * blocks];
            for row in 0..rows {
                let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let expected = exact_q6_k_q8k_dot(row_bytes, q8k_token, blocks);
                let got = actual[token * rows + row];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "token {token} row {row}: actual={got} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q6_k_scalar_matches_exact_oracle() {
        let blocks = 2;
        let row = make_q6_k_row(blocks);
        let q8k = make_q8k_blocks(blocks);

        let scalar = dot_q6_k_q8k_scalar(&row, &q8k, blocks);
        let exact = exact_q6_k_q8k_dot(&row, &q8k, blocks);

        assert!(
            (scalar - exact).abs() < 1e-4,
            "scalar={scalar} exact={exact} diff={}",
            (scalar - exact).abs()
        );
    }
}
