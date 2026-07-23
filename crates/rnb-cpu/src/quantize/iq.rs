//! Scalar decode helpers and GGML-compatible codebooks for importance quants.

pub(crate) use rnb_core::quant_codebooks::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
};

const QK_K: usize = 256;
const KMASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[inline]
fn f16(bytes: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

#[inline]
fn u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

pub(crate) fn dequantize_iq2_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 66);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..];
    for ib32 in 0..QK_K / 32 {
        let packed = &qs[ib32 * 8..ib32 * 8 + 8];
        let indices = u32_le(packed).to_le_bytes();
        let scales_and_signs = u32_le(&packed[4..]);
        let db = d * (0.5 + (scales_and_signs >> 28) as f32) * 0.25;
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let grid = IQ2XXS_GRID[indices[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((scales_and_signs >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let sign = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                out[l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

pub(crate) fn dequantize_iq2_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 82);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    let signs = &qs[32..];
    for ib32 in 0..QK_K / 32 {
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let scale = if l < 2 {
                scales[ib32] & 0x0f
            } else {
                scales[ib32] >> 4
            };
            let db = d * (0.5 + scale as f32) * 0.25;
            let high = ((qh[ib32] as usize) << (8 - 2 * l)) & 0x300;
            let grid = IQ2S_GRID[qs[4 * ib32 + l] as usize | high].to_le_bytes();
            let sign_bits = signs[4 * ib32 + l];
            for j in 0..8 {
                let sign = if sign_bits & KMASK[j] != 0 { -1.0 } else { 1.0 };
                out[l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

pub(crate) fn dequantize_iq3_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 98);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..66];
    let scales_and_signs = &block[66..98];
    for ib32 in 0..QK_K / 32 {
        let packed = u32_le(&scales_and_signs[ib32 * 4..]);
        let db = d * (0.5 + (packed >> 28) as f32) * 0.5;
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((packed >> (7 * l)) & 127) as usize];
            let grid1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize].to_le_bytes();
            let grid2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                let sign1 = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                let sign2 = if signs & KMASK[j + 4] != 0 { -1.0 } else { 1.0 };
                out[l * 8 + j] = db * grid1[j] as f32 * sign1;
                out[l * 8 + j + 4] = db * grid2[j] as f32 * sign2;
            }
        }
    }
}

pub(crate) fn dequantize_iq4_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 136);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..136];

    for ib in 0..8 {
        let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
        let high = (((scales_h >> (2 * ib)) & 0x03) as u8) << 4;
        let dl = d * ((low | high) as f32 - 32.0);
        let q = &qs[ib * 16..(ib + 1) * 16];
        let out = &mut output[ib * 32..(ib + 1) * 32];
        for j in 0..16 {
            out[j] = dl * KVALUES_IQ4NL[(q[j] & 0x0f) as usize] as f32;
            out[j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

pub(crate) fn dequantize_iq2_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 74);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..66];
    let scales = &block[66..74];
    for ib32 in 0..QK_K / 32 {
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let packed = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
            let scale_code = if l < 2 {
                scales[ib32] & 0x0f
            } else {
                scales[ib32] >> 4
            };
            let db = d * (0.5 + scale_code as f32) * 0.25;
            let grid = IQ2XS_GRID[(packed & 0x01ff) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(packed >> 9) as usize];
            for j in 0..8 {
                let sign = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                out[l * 8 + j] = db * grid[j] as f32 * sign;
            }
        }
    }
}

pub(crate) fn dequantize_iq3_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 110);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for ib32 in 0..QK_K / 32 {
        let scale_byte = scales[ib32 / 2];
        let scale_code = if ib32 % 2 == 0 {
            scale_byte & 0x0f
        } else {
            scale_byte >> 4
        };
        let db = d * (1 + 2 * scale_code) as f32;
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let high = qh[ib32] as usize;
            let index1 = qs[8 * ib32 + 2 * l] as usize | ((high << (8 - 2 * l)) & 0x100);
            let index2 = qs[8 * ib32 + 2 * l + 1] as usize | ((high << (7 - 2 * l)) & 0x100);
            let grid1 = IQ3S_GRID[index1].to_le_bytes();
            let grid2 = IQ3S_GRID[index2].to_le_bytes();
            let sign_bits = signs[4 * ib32 + l];
            for j in 0..4 {
                let sign1 = if sign_bits & KMASK[j] != 0 { -1.0 } else { 1.0 };
                let sign2 = if sign_bits & KMASK[j + 4] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[l * 8 + j] = db * grid1[j] as f32 * sign1;
                out[l * 8 + j + 4] = db * grid2[j] as f32 * sign2;
            }
        }
    }
}

pub(crate) fn dequantize_iq1_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 50);
    debug_assert_eq!(output.len(), QK_K);
    let d = f16(block);
    let qs = &block[2..34];
    for ib32 in 0..QK_K / 32 {
        let qh_offset = 34 + 2 * ib32;
        let qh = u16::from_le_bytes([block[qh_offset], block[qh_offset + 1]]);
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let index = qs[4 * ib32 + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
            let grid = IQ1S_GRID[index].to_le_bytes();
            for j in 0..8 {
                out[l * 8 + j] = dl * (grid[j] as i8 as f32 + delta);
            }
        }
    }
}

pub(crate) fn dequantize_iq1_m_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), 56);
    debug_assert_eq!(output.len(), QK_K);
    let qs = &block[..32];
    let qh = &block[32..48];
    let scales = &block[48..56];
    let scale_words = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];
    let d_bits = (scale_words[0] >> 12)
        | ((scale_words[1] >> 8) & 0x00f0)
        | ((scale_words[2] >> 4) & 0x0f00)
        | (scale_words[3] & 0xf000);
    let d = half::f16::from_bits(d_bits).to_f32();
    for ib32 in 0..QK_K / 32 {
        let scale_word = scale_words[ib32 / 2];
        let scale_shift = 6 * (ib32 % 2);
        let dl1 = d * (2 * ((scale_word >> scale_shift) & 7) + 1) as f32;
        let dl2 = d * (2 * ((scale_word >> (scale_shift + 3)) & 7) + 1) as f32;
        let qh0 = qh[2 * ib32];
        let qh1 = qh[2 * ib32 + 1];
        let indices = [
            qs[4 * ib32] as usize | (((qh0 as usize) << 8) & 0x700),
            qs[4 * ib32 + 1] as usize | (((qh0 as usize) << 4) & 0x700),
            qs[4 * ib32 + 2] as usize | (((qh1 as usize) << 8) & 0x700),
            qs[4 * ib32 + 3] as usize | (((qh1 as usize) << 4) & 0x700),
        ];
        let deltas = [
            if qh0 & 0x08 != 0 { -0.125 } else { 0.125 },
            if qh0 & 0x80 != 0 { -0.125 } else { 0.125 },
            if qh1 & 0x08 != 0 { -0.125 } else { 0.125 },
            if qh1 & 0x80 != 0 { -0.125 } else { 0.125 },
        ];
        let out = &mut output[ib32 * 32..(ib32 + 1) * 32];
        for l in 0..4 {
            let grid = IQ1S_GRID[indices[l]].to_le_bytes();
            let dl = if l < 2 { dl1 } else { dl2 };
            for j in 0..8 {
                out[l * 8 + j] = dl * (grid[j] as i8 as f32 + deltas[l]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_f16() -> [u8; 2] {
        half::f16::from_f32(1.0).to_bits().to_le_bytes()
    }

    fn ggml_reference_hash<const N: usize>(
        decoder: fn(&[u8], &mut [f32]),
        has_leading_f16: bool,
    ) -> u64 {
        let mut block = [0; N];
        for (index, byte) in block.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(73).wrapping_add(19) & 0xff) as u8;
        }
        if has_leading_f16 {
            block[..2].copy_from_slice(&one_f16());
        }

        let mut output = [0.0; QK_K];
        decoder(&block, &mut output);
        output
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .fold(0xcbf29ce484222325, |hash, byte| {
                (hash ^ byte as u64).wrapping_mul(0x100000001b3)
            })
    }

    #[test]
    fn zero_codes_decode_to_positive_ones() {
        let mut output = [0.0; QK_K];

        let mut iq2_xxs = [0; 66];
        iq2_xxs[..2].copy_from_slice(&one_f16());
        dequantize_iq2_xxs_block(&iq2_xxs, &mut output);
        assert_eq!(output, [1.0; QK_K]);

        let mut iq2_s = [0; 82];
        iq2_s[..2].copy_from_slice(&one_f16());
        dequantize_iq2_s_block(&iq2_s, &mut output);
        assert_eq!(output, [1.0; QK_K]);

        let mut iq3_xxs = [0; 98];
        iq3_xxs[..2].copy_from_slice(&one_f16());
        dequantize_iq3_xxs_block(&iq3_xxs, &mut output);
        assert_eq!(output, [1.0; QK_K]);
    }
    #[test]
    fn deterministic_blocks_match_ggml_reference() {
        assert_eq!(
            ggml_reference_hash::<66>(dequantize_iq2_xxs_block, true),
            0x0ead89a48d8fee11
        );
        assert_eq!(
            ggml_reference_hash::<74>(dequantize_iq2_xs_block, true),
            0x9f2439c2f96127cd
        );
        assert_eq!(
            ggml_reference_hash::<82>(dequantize_iq2_s_block, true),
            0x69e6a32989d193d1
        );
        assert_eq!(
            ggml_reference_hash::<98>(dequantize_iq3_xxs_block, true),
            0xbcf1592321ede577
        );
        assert_eq!(
            ggml_reference_hash::<110>(dequantize_iq3_s_block, true),
            0x498fd7c1619a5e4c
        );
        assert_eq!(
            ggml_reference_hash::<50>(dequantize_iq1_s_block, true),
            0xb19ca22a960875f1
        );
        assert_eq!(
            ggml_reference_hash::<56>(dequantize_iq1_m_block, false),
            0x0ed87fc9b145001e
        );
        assert_eq!(
            ggml_reference_hash::<136>(dequantize_iq4_xs_block, true),
            0x0c822de33756fd50
        );
    }
}
