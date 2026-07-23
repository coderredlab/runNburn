//! AVX2 Q4_K/Q5_K/Q6_K × Q8_K integer dot product kernels.
//!
//! Port of llama.cpp `ggml-cpu/arch/x86/quants.c` (AVX2 path) for use in
//! `rnb-cpu`. Input is pre-quantized to `Q8KBlock` (see `activation_q8.rs`)
//! so each weight row reuses the same Q8K vector across `seq_len` columns
//! during prefill — exactly the row-reuse pattern that makes mobile NEON
//! fast on this same workload.
//!
//! Block layouts match `rnb-cpu/quantize/blocks.rs` which mirrors
//! `llama.cpp/ggml-common.h`.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use crate::gemm::activation_q8::Q8KBlock;

const BLOCK_Q2K_SIZE: usize = 84;
const BLOCK_Q3K_SIZE: usize = 110;
const BLOCK_Q4K_SIZE: usize = 144;
const BLOCK_Q5K_SIZE: usize = 176;
const BLOCK_Q6K_SIZE: usize = 210;

#[inline(always)]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// 256-bit shuffle masks for Q4/Q5_K scale broadcast (k4 variant: each scale
/// pair byte broadcasts to 32 lanes).
const SCALE_SHUFFLE_K4: [[u8; 32]; 8] = [
    [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1,
    ],
    [
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3,
    ],
    [
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5,
    ],
    [
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7,
    ],
    [
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9,
    ],
    [
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11,
    ],
    [
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 12, 13, 12, 13, 12, 13,
    ],
    [
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14,
        15, 14, 15, 14, 15, 14, 15, 14, 15,
    ],
];

/// Scale-pair broadcasts used by Q2_K/Q3_K. Each 128-bit half selects one
/// adjacent pair from the 16 expanded sub-block scales.
const SCALE_SHUFFLE_Q3: [[u8; 32]; 4] = [
    [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3,
    ],
    [
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7,
    ],
    [
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11,
    ],
    [
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14,
        15, 14, 15, 14, 15, 14, 15, 14, 15,
    ],
];

/// 128-bit shuffle masks for Q6_K scale broadcast (each scale byte broadcasts
/// to 8 lanes within a 128-bit register).
const SCALE_SHUFFLE_Q6: [[u8; 16]; 16] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9],
    [
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    ],
    [
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    ],
    [
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ],
    // unused entries 8..16; only 0..8 needed for QK_K=256 (16 scales × 8-wide)
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
    [0; 16],
];

#[inline(always)]
unsafe fn scale_shuffle_k4(i: usize) -> __m256i {
    _mm256_loadu_si256(SCALE_SHUFFLE_K4[i].as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn scale_shuffle_q3(i: usize) -> __m256i {
    _mm256_loadu_si256(SCALE_SHUFFLE_Q3[i].as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn scale_shuffle_q6(i: usize) -> __m128i {
    _mm_loadu_si128(SCALE_SHUFFLE_Q6[i].as_ptr() as *const __m128i)
}

/// Decode Q4_K / Q5_K packed 12-byte `scales` field into 8 sub-block scales
/// and 8 sub-block mins (each 6-bit, expanded to u8). Layout matches the
/// `kmask1/kmask2/kmask3` unpack in `ggml-cpu/arch/x86/quants.c`.
#[inline(always)]
unsafe fn unpack_q4k_scales(scales12: &[u8; 12]) -> [u32; 4] {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut utmp = [0u32; 4];
    // memcpy 12 bytes -> 3 u32 (utmp[0..3])
    let src = scales12.as_ptr() as *const u32;
    utmp[0] = src.read_unaligned();
    utmp[1] = src.add(1).read_unaligned();
    utmp[2] = src.add(2).read_unaligned();
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    utmp
}

/// Q2_K row (n_blocks × 84 bytes) × pre-quantized Q8_K input vector.
/// Caller must ensure the exact row length and AVX2+FMA availability.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q2k_q8k_avx2(row_bytes: &[u8], y: &[Q8KBlock]) -> f32 {
    let nb = y.len();
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q2K_SIZE);

    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q2K_SIZE);
        let scales_ptr = block;
        let q2_ptr = block.add(16);
        let d_x = f16_to_f32((block.add(80) as *const u16).read_unaligned());
        let dmin_x = f16_to_f32((block.add(82) as *const u16).read_unaligned());
        let d = y[i].d * d_x;
        let dmin = -y[i].d * dmin_x;

        let mins_and_scales = _mm_loadu_si128(scales_ptr as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let prod = _mm256_madd_epi16(
            mins,
            _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i),
        );
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(low_scales, low_scales),
            mm256_set_m128i(high_scales, high_scales),
        ];

        let mut sumi = _mm256_setzero_si256();
        let mut q2 = q2_ptr;
        let mut q8 = y[i].qs.as_ptr();
        for scale_group in &scales {
            let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
            q2 = q2.add(32);
            let q2_0 = _mm256_and_si256(q2bits, m3);
            let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            let mut p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            let mut p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            let mut p3 = _mm256_maddubs_epi16(q2_3, q8_3);
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(3)), p3);
            sumi = _mm256_add_epi32(
                sumi,
                _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)),
            );
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

/// Q3_K row (n_blocks × 110 bytes) × pre-quantized Q8_K input vector.
/// Caller must ensure the exact row length and AVX2+FMA availability.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q3k_q8k_avx2(row_bytes: &[u8], y: &[Q8KBlock]) -> f32 {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let nb = y.len();
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q3K_SIZE);

    let m3 = _mm256_set1_epi8(3);
    let mone = _mm256_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q3K_SIZE);
        let hmask_ptr = block;
        let q3_ptr = block.add(32);
        let scales_ptr = block.add(96);
        let d_x = f16_to_f32((block.add(108) as *const u16).read_unaligned());
        let d = y[i].d * d_x;

        let aux0 = (scales_ptr as *const u32).read_unaligned();
        let aux1 = (scales_ptr.add(4) as *const u32).read_unaligned();
        let aux2 = (scales_ptr.add(8) as *const u32).read_unaligned();
        let mut scales128 = _mm_set_epi32(
            (((aux1 >> 4) & KMASK2) | (((aux2 >> 6) & KMASK1) << 4)) as i32,
            (((aux0 >> 4) & KMASK2) | (((aux2 >> 4) & KMASK1) << 4)) as i32,
            ((aux1 & KMASK2) | (((aux2 >> 2) & KMASK1) << 4)) as i32,
            ((aux0 & KMASK2) | ((aux2 & KMASK1) << 4)) as i32,
        );
        scales128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(low_scales, low_scales),
            mm256_set_m128i(high_scales, high_scales),
        ];

        let hbits = _mm256_loadu_si256(hmask_ptr as *const __m256i);
        let mut sumi = _mm256_setzero_si256();
        let mut bit = 0i32;
        let mut q3 = q3_ptr;
        let mut q8 = y[i].qs.as_ptr();

        for scale_group in &scales {
            let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
            q3 = q3.add(32);

            let q3_0 = _mm256_and_si256(q3bits, m3);
            let shift0 = _mm_cvtsi32_si128(bit);
            let high0 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift0)),
                    shift0,
                ),
                2,
            );
            bit += 1;
            let q3_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            let shift1 = _mm_cvtsi32_si128(bit);
            let high1 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift1)),
                    shift1,
                ),
                2,
            );
            bit += 1;
            let q3_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            let shift2 = _mm_cvtsi32_si128(bit);
            let high2 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift2)),
                    shift2,
                ),
                2,
            );
            bit += 1;
            let q3_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            let shift3 = _mm_cvtsi32_si128(bit);
            let high3 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift3)),
                    shift3,
                ),
                2,
            );
            bit += 1;

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p0 = _mm256_sub_epi16(
                _mm256_maddubs_epi16(q3_0, q8_0),
                _mm256_maddubs_epi16(high0, q8_0),
            );
            let mut p1 = _mm256_sub_epi16(
                _mm256_maddubs_epi16(q3_1, q8_1),
                _mm256_maddubs_epi16(high1, q8_1),
            );
            let mut p2 = _mm256_sub_epi16(
                _mm256_maddubs_epi16(q3_2, q8_2),
                _mm256_maddubs_epi16(high2, q8_2),
            );
            let mut p3 = _mm256_sub_epi16(
                _mm256_maddubs_epi16(q3_3, q8_3),
                _mm256_maddubs_epi16(high3, q8_3),
            );
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(3)), p3);
            sumi = _mm256_add_epi32(
                sumi,
                _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)),
            );
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

/// Four Q2_K rows × Q8_K activations sharing one packed weight decode.
///
/// Each output preserves the single-vector kernel's block and reduction order.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q2k_q8k_avx2_x4(row_bytes: &[u8], y: [&[Q8KBlock]; 4]) -> [f32; 4] {
    let nb = y[0].len();
    debug_assert!(y.iter().all(|input| input.len() == nb));
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q2K_SIZE);

    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0f);
    let mut acc = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q2K_SIZE);
        let scales_ptr = block;
        let q2_ptr = block.add(16);
        let d_x = f16_to_f32((block.add(80) as *const u16).read_unaligned());
        let dmin_x = f16_to_f32((block.add(82) as *const u16).read_unaligned());

        let mins_and_scales = _mm_loadu_si128(scales_ptr as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let mut d = [0.0f32; 4];
        for token in 0..4 {
            let input = &y[token][i];
            d[token] = input.d * d_x;
            let dmin = -input.d * dmin_x;
            let prod = _mm256_madd_epi16(
                mins,
                _mm256_loadu_si256(input.bsums.as_ptr() as *const __m256i),
            );
            acc[token] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&dmin),
                _mm256_cvtepi32_ps(prod),
                acc[token],
            );
        }

        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(low_scales, low_scales),
            mm256_set_m128i(high_scales, high_scales),
        ];
        let mut sumi = [_mm256_setzero_si256(); 4];
        for (scale_idx, scale_group) in scales.iter().enumerate() {
            let q2bits = _mm256_loadu_si256(q2_ptr.add(scale_idx * 32) as *const __m256i);
            let q2_0 = _mm256_and_si256(q2bits, m3);
            let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
            let scale0 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(0));
            let scale1 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(1));
            let scale2 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(2));
            let scale3 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(3));

            for token in 0..4 {
                let q8 = y[token][i].qs.as_ptr().add(scale_idx * 128);
                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                let q8_1 = _mm256_loadu_si256(q8.add(32) as *const __m256i);
                let q8_2 = _mm256_loadu_si256(q8.add(64) as *const __m256i);
                let q8_3 = _mm256_loadu_si256(q8.add(96) as *const __m256i);

                let mut p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                let mut p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                let mut p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                let mut p3 = _mm256_maddubs_epi16(q2_3, q8_3);
                p0 = _mm256_madd_epi16(scale0, p0);
                p1 = _mm256_madd_epi16(scale1, p1);
                p2 = _mm256_madd_epi16(scale2, p2);
                p3 = _mm256_madd_epi16(scale3, p3);
                sumi[token] = _mm256_add_epi32(
                    sumi[token],
                    _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)),
                );
            }
        }

        for token in 0..4 {
            acc[token] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&d[token]),
                _mm256_cvtepi32_ps(sumi[token]),
                acc[token],
            );
        }
    }

    [
        hsum_float_8(acc[0]),
        hsum_float_8(acc[1]),
        hsum_float_8(acc[2]),
        hsum_float_8(acc[3]),
    ]
}

/// Four Q3_K rows × Q8_K activations sharing one packed weight decode.
///
/// Each output preserves the single-vector kernel's block and reduction order.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q3k_q8k_avx2_x4(row_bytes: &[u8], y: [&[Q8KBlock]; 4]) -> [f32; 4] {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let nb = y[0].len();
    debug_assert!(y.iter().all(|input| input.len() == nb));
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q3K_SIZE);

    let m3 = _mm256_set1_epi8(3);
    let mone = _mm256_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);
    let mut acc = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q3K_SIZE);
        let hmask_ptr = block;
        let q3_ptr = block.add(32);
        let scales_ptr = block.add(96);
        let d_x = f16_to_f32((block.add(108) as *const u16).read_unaligned());
        let mut d = [0.0f32; 4];
        for token in 0..4 {
            d[token] = y[token][i].d * d_x;
        }

        let aux0 = (scales_ptr as *const u32).read_unaligned();
        let aux1 = (scales_ptr.add(4) as *const u32).read_unaligned();
        let aux2 = (scales_ptr.add(8) as *const u32).read_unaligned();
        let mut scales128 = _mm_set_epi32(
            (((aux1 >> 4) & KMASK2) | (((aux2 >> 6) & KMASK1) << 4)) as i32,
            (((aux0 >> 4) & KMASK2) | (((aux2 >> 4) & KMASK1) << 4)) as i32,
            ((aux1 & KMASK2) | (((aux2 >> 2) & KMASK1) << 4)) as i32,
            ((aux0 & KMASK2) | ((aux2 & KMASK1) << 4)) as i32,
        );
        scales128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(low_scales, low_scales),
            mm256_set_m128i(high_scales, high_scales),
        ];

        let hbits = _mm256_loadu_si256(hmask_ptr as *const __m256i);
        let mut sumi = [_mm256_setzero_si256(); 4];
        for (scale_idx, scale_group) in scales.iter().enumerate() {
            let q3bits = _mm256_loadu_si256(q3_ptr.add(scale_idx * 32) as *const __m256i);
            let bit = (scale_idx * 4) as i32;

            let q3_0 = _mm256_and_si256(q3bits, m3);
            let shift0 = _mm_cvtsi32_si128(bit);
            let high0 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift0)),
                    shift0,
                ),
                2,
            );
            let q3_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            let shift1 = _mm_cvtsi32_si128(bit + 1);
            let high1 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift1)),
                    shift1,
                ),
                2,
            );
            let q3_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            let shift2 = _mm_cvtsi32_si128(bit + 2);
            let high2 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift2)),
                    shift2,
                ),
                2,
            );
            let q3_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            let shift3 = _mm_cvtsi32_si128(bit + 3);
            let high3 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, shift3)),
                    shift3,
                ),
                2,
            );
            let scale0 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(0));
            let scale1 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(1));
            let scale2 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(2));
            let scale3 = _mm256_shuffle_epi8(*scale_group, scale_shuffle_q3(3));

            for token in 0..4 {
                let q8 = y[token][i].qs.as_ptr().add(scale_idx * 128);
                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                let q8_1 = _mm256_loadu_si256(q8.add(32) as *const __m256i);
                let q8_2 = _mm256_loadu_si256(q8.add(64) as *const __m256i);
                let q8_3 = _mm256_loadu_si256(q8.add(96) as *const __m256i);

                let mut p0 = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3_0, q8_0),
                    _mm256_maddubs_epi16(high0, q8_0),
                );
                let mut p1 = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3_1, q8_1),
                    _mm256_maddubs_epi16(high1, q8_1),
                );
                let mut p2 = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3_2, q8_2),
                    _mm256_maddubs_epi16(high2, q8_2),
                );
                let mut p3 = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3_3, q8_3),
                    _mm256_maddubs_epi16(high3, q8_3),
                );
                p0 = _mm256_madd_epi16(scale0, p0);
                p1 = _mm256_madd_epi16(scale1, p1);
                p2 = _mm256_madd_epi16(scale2, p2);
                p3 = _mm256_madd_epi16(scale3, p3);
                sumi[token] = _mm256_add_epi32(
                    sumi[token],
                    _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)),
                );
            }
        }

        for token in 0..4 {
            acc[token] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&d[token]),
                _mm256_cvtepi32_ps(sumi[token]),
                acc[token],
            );
        }
    }

    [
        hsum_float_8(acc[0]),
        hsum_float_8(acc[1]),
        hsum_float_8(acc[2]),
        hsum_float_8(acc[3]),
    ]
}

/// Q4_K row (n_blocks × 144 bytes) × pre-quantized Q8_K input vector.
/// Caller must ensure `row_bytes.len() == y.len() * 144` and the CPU has AVX2+FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4k_q8k_avx2(row_bytes: &[u8], y: &[Q8KBlock]) -> f32 {
    let nb = y.len();
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q4K_SIZE);

    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q4K_SIZE);
        let d_x = f16_to_f32((block as *const u16).read_unaligned());
        let dmin_x = f16_to_f32((block as *const u16).add(1).read_unaligned());
        let d = y[i].d * d_x;
        let dmin = -y[i].d * dmin_x;

        let scales12 = &*(block.add(4) as *const [u8; 12]);
        let utmp = unpack_q4k_scales(scales12);

        let qs = block.add(16); // 128 bytes Q4 quants

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        // bsums: hadd 16 i16 -> 8 i16 (pairs), multiply by mins, scale by dmin
        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        let prod = _mm_madd_epi16(mins128, q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        let mut sumi = _mm256_setzero_si256();
        let mut q4 = qs;
        let mut q8 = y[i].qs.as_ptr();

        // 4 inner iterations: QK_K / 64 = 256 / 64
        for j in 0..4 {
            let scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            let scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    let acc_m_hi = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    let acc_m_final = _mm_add_ss(acc_m_hi, _mm_movehdup_ps(acc_m_hi));
    hsum_float_8(acc) + _mm_cvtss_f32(acc_m_final)
}

/// Q5_K row × Q8_K input. Same scale layout as Q4_K (12-byte packed scales)
/// plus an extra `qh` byte plane (32 bytes per block, providing the 5th bit).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q5k_q8k_avx2(row_bytes: &[u8], y: &[Q8KBlock]) -> f32 {
    let nb = y.len();
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q5K_SIZE);

    let m4 = _mm256_set1_epi8(0xF);
    let mone = _mm256_set1_epi8(1);

    let mut acc = _mm256_setzero_ps();
    let mut summs = 0.0f32;

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q5K_SIZE);
        let d_x = f16_to_f32((block as *const u16).read_unaligned());
        let dmin_x = f16_to_f32((block as *const u16).add(1).read_unaligned());
        let d = y[i].d * d_x;
        let dmin = -y[i].d * dmin_x;

        let scales12 = &*(block.add(4) as *const [u8; 12]);
        let utmp = unpack_q4k_scales(scales12);

        let qh_ptr = block.add(16); // 32 bytes high bits
        let qs_ptr = block.add(48); // 128 bytes low 4 bits

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        let prod = _mm_madd_epi16(mins128, q8s);
        let hsum = _mm_hadd_epi32(
            _mm_hadd_epi32(prod, _mm_setzero_si128()),
            _mm_setzero_si128(),
        );
        summs += dmin * (_mm_extract_epi32(hsum, 0) as f32);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        let hbits = _mm256_loadu_si256(qh_ptr as *const __m256i);
        let mut hmask = mone;
        let mut sumi = _mm256_setzero_si256();
        let mut bit = 0i32;
        let mut q5 = qs_ptr;
        let mut q8 = y[i].qs.as_ptr();

        for j in 0..4 {
            let scale_0 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            let scale_1 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));

            let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
            q5 = q5.add(32);

            let q5l_0 = _mm256_and_si256(q5bits, m4);
            let shift0 = _mm_cvtsi32_si128(bit);
            let q5h_0 =
                _mm256_slli_epi16(_mm256_srl_epi16(_mm256_and_si256(hbits, hmask), shift0), 4);
            bit += 1;
            let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            let shift1 = _mm_cvtsi32_si128(bit);
            let q5h_1 =
                _mm256_slli_epi16(_mm256_srl_epi16(_mm256_and_si256(hbits, hmask), shift1), 4);
            bit += 1;
            let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc) + summs
}

/// Q6_K row × Q8_K input. 6-bit weight (ql low-4 + qh high-2 bits), 16
/// signed i8 sub-block scales, single block-level d (f16). The per-sub-block
/// bias of -32 is absorbed into `q8sclsub` (scaled by 32 via `slli_epi32 5`).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6k_q8k_avx2(row_bytes: &[u8], y: &[Q8KBlock]) -> f32 {
    let nb = y.len();
    debug_assert_eq!(row_bytes.len(), nb * BLOCK_Q6K_SIZE);

    let m3 = _mm256_set1_epi8(3);
    let m15 = _mm256_set1_epi8(15);
    let m12 = _mm256_set1_epi8(12);
    let m48 = _mm256_set1_epi8(48);
    let m_top2 = _mm256_set1_epi8(-64); // 0xC0 sign-extended

    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let block = row_bytes.as_ptr().add(i * BLOCK_Q6K_SIZE);
        let ql_ptr = block; // 128 bytes
        let qh_ptr = block.add(128); // 64 bytes
        let scales_ptr = block.add(192); // 16 i8
        let d_x = f16_to_f32((block.add(208) as *const u16).read_unaligned());
        let d = y[i].d * d_x;

        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let scales = _mm_loadu_si128(scales_ptr as *const __m128i);
        let scales_16 = _mm256_cvtepi8_epi16(scales);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        let mut sumi = _mm256_setzero_si256();
        let mut q4 = ql_ptr;
        let mut qh = qh_ptr;
        let mut q8 = y[i].qs.as_ptr();
        let mut is = 0usize;

        // QK_K / 128 = 2 outer iterations
        for _j in 0..2 {
            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bitsh = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m12), 2);
            let q4h_2 = _mm256_and_si256(q4bitsh, m48);
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsh, m_top2), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            let scale_0 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 0));
            let scale_1 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 1));
            let scale_2 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 2));
            let scale_3 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

#[cfg(test)]
mod tests {
    use super::{dot_q2k_q8k_avx2, dot_q2k_q8k_avx2_x4, dot_q3k_q8k_avx2, dot_q3k_q8k_avx2_x4};
    use crate::gemm::activation_q8::quantize_input_q8k;
    use crate::gemm::quant_gemv::{dot_quantized_row, QuantGemvType};

    fn q2k_row(n_blocks: usize) -> Vec<u8> {
        let mut row = vec![0u8; n_blocks * 84];
        for block_idx in 0..n_blocks {
            let block = &mut row[block_idx * 84..(block_idx + 1) * 84];
            for (i, scale) in block[..16].iter_mut().enumerate() {
                let low = (1 + i * 7 + block_idx * 3) & 0x0f;
                let high = (15 + block_idx - i % 16) & 0x0f;
                *scale = (low | (high << 4)) as u8;
            }
            for (i, q) in block[16..80].iter_mut().enumerate() {
                *q = (i * 37 + block_idx * 19 + 11) as u8;
            }
            block[80..82].copy_from_slice(
                &half::f16::from_f32(0.03125 * (block_idx + 1) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
            block[82..84].copy_from_slice(
                &half::f16::from_f32(0.015625 * (block_idx + 1) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        row
    }

    fn q3k_row(n_blocks: usize) -> Vec<u8> {
        let mut row = vec![0u8; n_blocks * 110];
        for block_idx in 0..n_blocks {
            let block = &mut row[block_idx * 110..(block_idx + 1) * 110];
            for (i, hmask) in block[..32].iter_mut().enumerate() {
                *hmask = ((i * 29 + block_idx * 13) as u8) ^ 0xa5;
            }
            for (i, q) in block[32..96].iter_mut().enumerate() {
                *q = (i * 41 + block_idx * 23 + 7) as u8;
            }
            for (i, scale) in block[96..108].iter_mut().enumerate() {
                *scale = (i * 53 + block_idx * 17 + 3) as u8;
            }
            block[108..110].copy_from_slice(
                &half::f16::from_f32(0.0234375 * (block_idx + 1) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        row
    }

    fn assert_close(got: f32, expected: f32) {
        let tolerance = 1.0e-3 + 2.0e-4 * expected.abs();
        assert!(
            (got - expected).abs() <= tolerance,
            "got={got} expected={expected} tolerance={tolerance}"
        );
    }

    #[test]
    fn q2q3_avx2_dots_match_scalar_on_quantized_input() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }

        let input: Vec<f32> = (0..512)
            .map(|i| ((i * 97 % 251) as f32 - 125.0) * 0.03125)
            .collect();
        let q8k = quantize_input_q8k(&input);
        let reconstructed: Vec<f32> = q8k
            .iter()
            .flat_map(|block| block.qs.iter().map(|&q| block.d * q as f32))
            .collect();

        let q2 = q2k_row(2);
        let q2_expected = dot_quantized_row(&q2, &reconstructed, 512, QuantGemvType::Q2K);
        assert_close(unsafe { dot_q2k_q8k_avx2(&q2, &q8k) }, q2_expected);
        let mut q2_misaligned = vec![0u8];
        q2_misaligned.extend_from_slice(&q2);
        assert_close(
            unsafe { dot_q2k_q8k_avx2(&q2_misaligned[1..], &q8k) },
            q2_expected,
        );

        let q3 = q3k_row(2);
        let q3_expected = dot_quantized_row(&q3, &reconstructed, 512, QuantGemvType::Q3K);
        assert_close(unsafe { dot_q3k_q8k_avx2(&q3, &q8k) }, q3_expected);
        let mut q3_misaligned = vec![0u8];
        q3_misaligned.extend_from_slice(&q3);
        assert_close(
            unsafe { dot_q3k_q8k_avx2(&q3_misaligned[1..], &q8k) },
            q3_expected,
        );
    }

    #[test]
    fn q2q3_avx2_x4_matches_single_vector_reduction() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }

        let inputs = (0..4)
            .map(|token| {
                let input: Vec<f32> = (0..512)
                    .map(|i| {
                        (((i + token * 43) * 97 % 251) as f32 - 125.0)
                            * (0.015625 + token as f32 * 0.00390625)
                    })
                    .collect();
                quantize_input_q8k(&input)
            })
            .collect::<Vec<_>>();
        let input_refs = [
            inputs[0].as_slice(),
            inputs[1].as_slice(),
            inputs[2].as_slice(),
            inputs[3].as_slice(),
        ];

        let q2 = q2k_row(2);
        let q2_batch = unsafe { dot_q2k_q8k_avx2_x4(&q2, input_refs) };
        for token in 0..4 {
            assert_eq!(q2_batch[token], unsafe {
                dot_q2k_q8k_avx2(&q2, &inputs[token])
            });
        }

        let q3 = q3k_row(2);
        let q3_batch = unsafe { dot_q3k_q8k_avx2_x4(&q3, input_refs) };
        for token in 0..4 {
            assert_eq!(q3_batch[token], unsafe {
                dot_q3k_q8k_avx2(&q3, &inputs[token])
            });
        }
    }
}
