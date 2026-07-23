//! AArch64 DOTPROD kernels for quant formats outside the main Q4_K/Q5_K/Q6_K path.

use crate::gemm::activation_q8::{Q8Block, Q8KBlock};
use crate::quantize::iq::{IQ2S_GRID, IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL};
use crate::quantize::BlockQ3_K;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn dot_i8x16(a: int8x16_t, b: int8x16_t) -> i32 {
    vaddvq_s32(vdotq_s32(vdupq_n_s32(0), a, b))
}

#[inline(always)]
unsafe fn sum_i8x16(values: int8x16_t) -> i32 {
    vaddlvq_s8(values) as i32
}

#[inline(always)]
fn f16_le(bytes: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

#[inline]
fn q3k_scales(block: &BlockQ3_K) -> [i8; 16] {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let sb = &block.scales;
    let a0 = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
    let a1 = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
    let a2 = u32::from_le_bytes([sb[8], sb[9], sb[10], sb[11]]);
    let unpacked = [
        (a0 & KMASK2) | (((a2 >> 0) & KMASK1) << 4),
        (a1 & KMASK2) | (((a2 >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((a2 >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((a2 >> 6) & KMASK1) << 4),
    ];

    let mut scales = [0i8; 16];
    for (chunk, value) in scales.chunks_exact_mut(4).zip(unpacked) {
        for (dst, byte) in chunk.iter_mut().zip(value.to_le_bytes()) {
            *dst = byte as i8 - 32;
        }
    }
    scales
}

/// Q3_K × Q8_K direct integer dot product.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q3_k_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 110);
    debug_assert_eq!(q8k.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = &*(row_bytes.as_ptr().add(bi * 110) as *const BlockQ3_K);
        let activation = &q8k[bi];
        let scales = q3k_scales(block);
        let mut quant_lanes = vdupq_n_s32(0);

        for sub in 0..16 {
            let half = sub / 8;
            let local = sub % 8;
            let q_base = half * 32 + (local % 2) * 16;
            let shift = (local / 2) * 2;
            let high_base = (local % 2) * 16;
            let high_mask = 1u8 << (sub / 2);

            let packed = vld1q_u8(block.qs.as_ptr().add(q_base));
            let low = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(shift as i8))), vdupq_n_u8(3));
            let high = vld1q_u8(block.hmask.as_ptr().add(high_base));
            let missing_high = vceqq_u8(vandq_u8(high, vdupq_n_u8(high_mask)), vdupq_n_u8(0));
            let correction = vandq_u8(missing_high, vdupq_n_u8(4));
            let quant = vsubq_s8(vreinterpretq_s8_u8(low), vreinterpretq_s8_u8(correction));
            let input = vld1q_s8(activation.qs.as_ptr().add(sub * 16));
            let dot_lanes = vdotq_s32(vdupq_n_s32(0), quant, input);
            quant_lanes = vmlaq_n_s32(quant_lanes, dot_lanes, scales[sub] as i32);
        }

        let quant_sum = vaddvq_s32(quant_lanes);
        sum += block.d.to_f32() * activation.d * quant_sum as f32;
    }
    sum
}

/// Q4_1 × Q8_0 direct integer dot product.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_1_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 20);
    debug_assert_eq!(q8.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = row_bytes.as_ptr().add(bi * 20);
        let packed = vld1q_u8(block.add(4));
        let low = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f)));
        let high = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
        let input_low = vld1q_s8(q8[bi].qs.as_ptr());
        let input_high = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let quant_dot = dot_i8x16(low, input_low) + dot_i8x16(high, input_high);
        let input_sum = sum_i8x16(input_low) + sum_i8x16(input_high);
        let d = f16_le(std::slice::from_raw_parts(block, 2));
        let min = f16_le(std::slice::from_raw_parts(block.add(2), 2));
        sum += q8[bi].d * (d * quant_dot as f32 + min * input_sum as f32);
    }
    sum
}

/// Q5_1 × Q8_0 direct integer dot product.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q5_1_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    const BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    debug_assert_eq!(row_bytes.len(), n_blocks * 24);
    debug_assert_eq!(q8.len(), n_blocks);

    let bits = vld1_u8(BIT_MASKS.as_ptr());
    let bit_masks = vcombine_u8(bits, bits);
    let high_bit = vdupq_n_u8(0x10);
    let mut sum = 0.0f32;

    for bi in 0..n_blocks {
        let block = row_bytes.as_ptr().add(bi * 24);
        let qh_low_source = vcombine_u8(vdup_n_u8(*block.add(4)), vdup_n_u8(*block.add(5)));
        let qh_high_source = vcombine_u8(vdup_n_u8(*block.add(6)), vdup_n_u8(*block.add(7)));
        let qh_low = vandq_u8(vtstq_u8(qh_low_source, bit_masks), high_bit);
        let qh_high = vandq_u8(vtstq_u8(qh_high_source, bit_masks), high_bit);
        let packed = vld1q_u8(block.add(8));
        let low = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)), qh_low));
        let high = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(packed, 4), qh_high));
        let input_low = vld1q_s8(q8[bi].qs.as_ptr());
        let input_high = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let quant_dot = dot_i8x16(low, input_low) + dot_i8x16(high, input_high);
        let input_sum = sum_i8x16(input_low) + sum_i8x16(input_high);
        let d = f16_le(std::slice::from_raw_parts(block, 2));
        let min = f16_le(std::slice::from_raw_parts(block.add(2), 2));
        sum += q8[bi].d * (d * quant_dot as f32 + min * input_sum as f32);
    }
    sum
}

/// Q8_1 × Q8_0 direct integer dot product. The stored Q8_1 sum is not needed
/// because the weight has no additive minimum term.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_1_q8_neon(row_bytes: &[u8], q8: &[Q8Block], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 36);
    debug_assert_eq!(q8.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = row_bytes.as_ptr().add(bi * 36);
        let input_low = vld1q_s8(q8[bi].qs.as_ptr());
        let input_high = vld1q_s8(q8[bi].qs.as_ptr().add(16));
        let quant_dot = dot_i8x16(vld1q_s8(block.add(4) as *const i8), input_low)
            + dot_i8x16(vld1q_s8(block.add(20) as *const i8), input_high);
        sum += f16_le(std::slice::from_raw_parts(block, 2)) * q8[bi].d * quant_dot as f32;
    }
    sum
}

const SIGN_BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

#[inline(always)]
unsafe fn signed_grid_u64(grid: u64, signs: u8) -> int8x8_t {
    let values = vcreate_u8(grid);
    let negative = vtst_u8(vdup_n_u8(signs), vld1_u8(SIGN_BIT_MASKS.as_ptr()));
    let negated = vreinterpret_u8_s8(vneg_s8(vreinterpret_s8_u8(values)));
    vreinterpret_s8_u8(vbsl_u8(negative, negated, values))
}

#[inline(always)]
unsafe fn signed_grid16(first: u64, first_signs: u8, second: u64, second_signs: u8) -> int8x16_t {
    vcombine_s8(
        signed_grid_u64(first, first_signs),
        signed_grid_u64(second, second_signs),
    )
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_iq2_xxs_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 66);
    debug_assert_eq!(q8k.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = &row_bytes[bi * 66..(bi + 1) * 66];
        let mut block_sum = 0.0f32;
        for group in 0..8 {
            let packed = &block[2 + group * 8..2 + (group + 1) * 8];
            let indices = [packed[0], packed[1], packed[2], packed[3]];
            let scales_signs = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
            let sign = |index: usize| KSIGNS_IQ2XS[((scales_signs >> (7 * index)) & 127) as usize];
            let grid = |index: usize| IQ2XXS_GRID[indices[index] as usize];
            let values0 = signed_grid16(grid(0), sign(0), grid(1), sign(1));
            let values1 = signed_grid16(grid(2), sign(2), grid(3), sign(3));
            let input = q8k[bi].qs.as_ptr().add(group * 32);
            let dot =
                dot_i8x16(values0, vld1q_s8(input)) + dot_i8x16(values1, vld1q_s8(input.add(16)));
            block_sum += (0.5 + (scales_signs >> 28) as f32) * dot as f32;
        }
        sum += f16_le(block) * q8k[bi].d * 0.25 * block_sum;
    }
    sum
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_iq2_s_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 82);
    debug_assert_eq!(q8k.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = &row_bytes[bi * 82..(bi + 1) * 82];
        let qs = &block[2..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        let mut block_sum = 0i32;

        for group in 0..8 {
            let grid = |part: usize| {
                let high = ((qh[group] as usize) << (8 - 2 * part)) & 0x300;
                IQ2S_GRID[qs[group * 4 + part] as usize | high]
            };
            let signs = |part: usize| qs[32 + group * 4 + part];
            let values0 = signed_grid16(grid(0), signs(0), grid(1), signs(1));
            let values1 = signed_grid16(grid(2), signs(2), grid(3), signs(3));
            let input = q8k[bi].qs.as_ptr().add(group * 32);
            let scale = scales[group];
            block_sum += dot_i8x16(values0, vld1q_s8(input)) * (1 + 2 * (scale & 0x0f) as i32);
            block_sum +=
                dot_i8x16(values1, vld1q_s8(input.add(16))) * (1 + 2 * (scale >> 4) as i32);
        }
        sum += f16_le(block) * q8k[bi].d * 0.125 * block_sum as f32;
    }
    sum
}

#[inline(always)]
unsafe fn signed_iq3_grid8(first: u32, second: u32, signs: u8) -> int8x8_t {
    signed_grid_u64(first as u64 | ((second as u64) << 32), signs)
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_iq3_xxs_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 98);
    debug_assert_eq!(q8k.len(), n_blocks);

    let mut sum = 0.0f32;
    for bi in 0..n_blocks {
        let block = &row_bytes[bi * 98..(bi + 1) * 98];
        let qs = &block[2..66];
        let scales_signs = &block[66..98];
        let mut block_sum = 0.0f32;

        for group in 0..8 {
            let packed = u32::from_le_bytes([
                scales_signs[group * 4],
                scales_signs[group * 4 + 1],
                scales_signs[group * 4 + 2],
                scales_signs[group * 4 + 3],
            ]);
            let signs = |part: usize| KSIGNS_IQ2XS[((packed >> (7 * part)) & 127) as usize];
            let grid = |part: usize| {
                signed_iq3_grid8(
                    IQ3XXS_GRID[qs[group * 8 + 2 * part] as usize],
                    IQ3XXS_GRID[qs[group * 8 + 2 * part + 1] as usize],
                    signs(part),
                )
            };
            let values0 = vcombine_s8(grid(0), grid(1));
            let values1 = vcombine_s8(grid(2), grid(3));
            let input = q8k[bi].qs.as_ptr().add(group * 32);
            let dot =
                dot_i8x16(values0, vld1q_s8(input)) + dot_i8x16(values1, vld1q_s8(input.add(16)));
            block_sum += (0.5 + (packed >> 28) as f32) * dot as f32;
        }
        sum += f16_le(block) * q8k[bi].d * 0.5 * block_sum;
    }
    sum
}

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_iq4_xs_q8k_neon(row_bytes: &[u8], q8k: &[Q8KBlock], n_blocks: usize) -> f32 {
    debug_assert_eq!(row_bytes.len(), n_blocks * 136);
    debug_assert_eq!(q8k.len(), n_blocks);

    let values = vld1q_s8(KVALUES_IQ4NL.as_ptr());
    let mask = vdupq_n_u8(0x0f);
    let mut sum = 0.0f32;

    for bi in 0..n_blocks {
        let block = &row_bytes[bi * 136..(bi + 1) * 136];
        let mut high_scales = u16::from_le_bytes([block[2], block[3]]);
        let low_scales = &block[4..8];
        let quants = &block[8..136];
        let mut block_sum = 0i32;

        for group in 0..4 {
            let packed0 = vld1q_u8(quants.as_ptr().add(group * 32));
            let packed1 = vld1q_u8(quants.as_ptr().add(group * 32 + 16));
            let q0 = vqtbl1q_s8(values, vandq_u8(packed0, mask));
            let q1 = vqtbl1q_s8(values, vshrq_n_u8(packed0, 4));
            let q2 = vqtbl1q_s8(values, vandq_u8(packed1, mask));
            let q3 = vqtbl1q_s8(values, vshrq_n_u8(packed1, 4));
            let input = q8k[bi].qs.as_ptr().add(group * 64);
            let dot0 = dot_i8x16(q0, vld1q_s8(input)) + dot_i8x16(q1, vld1q_s8(input.add(16)));
            let dot1 =
                dot_i8x16(q2, vld1q_s8(input.add(32))) + dot_i8x16(q3, vld1q_s8(input.add(48)));
            let scale0 =
                ((low_scales[group] & 0x0f) | ((high_scales << 4) as u8 & 0x30)) as i32 - 32;
            let scale1 = ((low_scales[group] >> 4) | ((high_scales << 2) as u8 & 0x30)) as i32 - 32;
            high_scales >>= 4;
            block_sum += dot0 * scale0 + dot1 * scale1;
        }
        sum += f16_le(block) * q8k[bi].d * block_sum as f32;
    }
    sum
}
