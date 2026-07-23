//! Quantized row dequantization helpers.

use crate::quantize as q;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ3XXS,
    IQ4XS,
    IQ1M,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_0,
}

pub fn dequantize_bytes_to_f32(bytes: &[u8], dequant_type: DequantType) -> Vec<f32> {
    let byte_count = bytes.len();
    if byte_count == 0 {
        return vec![];
    }

    match dequant_type {
        DequantType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DequantType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        DequantType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let bf16 = u16::from_le_bytes([c[0], c[1]]) as u32;
                f32::from_bits(bf16 << 16)
            })
            .collect(),
        DequantType::Q4_0 => {
            let n_blocks = byte_count / 18;
            let mut out = vec![0.0f32; n_blocks * 32];
            for (bi, chunk) in bytes.chunks_exact(18).enumerate() {
                let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                let qs = &chunk[2..18];
                let base = bi * 32;
                for i in 0..16 {
                    out[base + i] = ((qs[i] & 0x0F) as f32 - 8.0) * d;
                    out[base + i + 16] = ((qs[i] >> 4) as f32 - 8.0) * d;
                }
            }
            out
        }
        DequantType::Q8_0 => {
            let n_blocks = byte_count / 34;
            let mut out = vec![0.0f32; n_blocks * 32];
            for (bi, chunk) in bytes.chunks_exact(34).enumerate() {
                let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                for i in 0..32 {
                    out[bi * 32 + i] = chunk[2 + i] as i8 as f32 * d;
                }
            }
            out
        }
        DequantType::Q4_1 => {
            dequant_basic_blocks::<q::BlockQ4_1>(bytes, 20, 32, q::dequantize_q4_1)
        }
        DequantType::Q5_0 => {
            dequant_basic_blocks::<q::BlockQ5_0>(bytes, 22, 32, q::dequantize_q5_0)
        }
        DequantType::Q5_1 => {
            dequant_basic_blocks::<q::BlockQ5_1>(bytes, 24, 32, q::dequantize_q5_1)
        }
        DequantType::Q8_1 => {
            let n_blocks = byte_count / 36;
            let mut out = vec![0.0f32; n_blocks * 32];
            for (bi, chunk) in bytes.chunks_exact(36).enumerate() {
                let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                for i in 0..32 {
                    out[bi * 32 + i] = chunk[4 + i] as i8 as f32 * d;
                }
            }
            out
        }
        DequantType::Q4K => dequant_k_blocks::<q::BlockQ4_K>(bytes, 144, 256, q::dequantize_q4_k),
        DequantType::Q6K => dequant_k_blocks::<q::BlockQ6_K>(bytes, 210, 256, q::dequantize_q6_k),
        DequantType::Q5K => dequant_k_blocks::<q::BlockQ5_K>(bytes, 176, 256, q::dequantize_q5_k),
        DequantType::Q3K => dequant_k_blocks::<q::BlockQ3_K>(bytes, 110, 256, q::dequantize_q3_k),
        DequantType::Q2K => dequant_k_blocks::<q::BlockQ2_K>(bytes, 84, 256, q::dequantize_q2_k),
        DequantType::IQ2XXS => dequant_iq_blocks(bytes, 66, q::iq::dequantize_iq2_xxs_block),
        DequantType::IQ2S => dequant_iq_blocks(bytes, 82, q::iq::dequantize_iq2_s_block),
        DequantType::IQ3XXS => dequant_iq_blocks(bytes, 98, q::iq::dequantize_iq3_xxs_block),
        DequantType::IQ4XS => dequant_iq_blocks(bytes, 136, q::iq::dequantize_iq4_xs_block),
        DequantType::Q8K => dequant_q8_k(bytes),
        DequantType::IQ4NL => dequant_iq4_nl(bytes),
        DequantType::TQ1_0 => dequant_tq1_0(bytes),
        DequantType::TQ2_0 => dequant_tq2_0(bytes),
        DequantType::MXFP4 => dequant_mxfp4(bytes),
        DequantType::NVFP4 => dequant_nvfp4(bytes),
        DequantType::Q1_0 => dequant_q1_0(bytes),
        DequantType::Q2_0 => dequant_q2_0(bytes),
        DequantType::IQ2XS => dequant_iq_blocks(bytes, 74, q::iq::dequantize_iq2_xs_block),
        DequantType::IQ1S => dequant_iq_blocks(bytes, 50, q::iq::dequantize_iq1_s_block),
        DequantType::IQ3S => dequant_iq_blocks(bytes, 110, q::iq::dequantize_iq3_s_block),
        DequantType::IQ1M => dequant_iq_blocks(bytes, 56, q::iq::dequantize_iq1_m_block),
    }
}

pub fn dequantize_row_to_slice_if_supported(
    bytes: &[u8],
    dequant_type: DequantType,
    output: &mut [f32],
) -> bool {
    match dequant_type {
        DequantType::F32 => {
            if bytes.len() != output.len() * 4 {
                return false;
            }
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            true
        }
        DequantType::F16 => {
            if bytes.len() != output.len() * 2 {
                return false;
            }
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                output[i] = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
            true
        }
        DequantType::Q6K => {
            if bytes.len() % 210 != 0 || output.len() != (bytes.len() / 210) * 256 {
                return false;
            }
            let mut tmp = [0.0f32; 256];
            for (bi, chunk) in bytes.chunks_exact(210).enumerate() {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ6_K) };
                q::dequantize_q6_k(block, &mut tmp);
                output[bi * 256..(bi + 1) * 256].copy_from_slice(&tmp);
            }
            true
        }
        DequantType::IQ2XXS => {
            dequant_iq_blocks_to_slice(bytes, output, 66, q::iq::dequantize_iq2_xxs_block)
        }
        DequantType::IQ2S => {
            dequant_iq_blocks_to_slice(bytes, output, 82, q::iq::dequantize_iq2_s_block)
        }
        DequantType::IQ3XXS => {
            dequant_iq_blocks_to_slice(bytes, output, 98, q::iq::dequantize_iq3_xxs_block)
        }
        DequantType::IQ4XS => {
            dequant_iq_blocks_to_slice(bytes, output, 136, q::iq::dequantize_iq4_xs_block)
        }
        _ => false,
    }
}

const KVALUES_IQ4_NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];
const KVALUES_FP4: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

fn f16_at(bytes: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn dequant_q8_k(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 292 * 256);
    for block in bytes.chunks_exact(292) {
        let d = f32::from_le_bytes(block[0..4].try_into().unwrap());
        output.extend(block[4..260].iter().map(|&q| q as i8 as f32 * d));
    }
    output
}

fn dequant_iq4_nl(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 18 * 32);
    for block in bytes.chunks_exact(18) {
        let d = f16_at(block, 0);
        for &packed in &block[2..18] {
            output.push(d * KVALUES_IQ4_NL[(packed & 0x0f) as usize]);
        }
        for &packed in &block[2..18] {
            output.push(d * KVALUES_IQ4_NL[(packed >> 4) as usize]);
        }
    }
    output
}

fn dequant_tq1_0(bytes: &[u8]) -> Vec<f32> {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let mut output = Vec::with_capacity(bytes.len() / 54 * 256);
    for block in bytes.chunks_exact(54) {
        let d = f16_at(block, 52);
        for base in [0usize] {
            for &power in &POW3 {
                for &packed in &block[base..base + 32] {
                    let q = packed.wrapping_mul(power);
                    output.push((((q as u16 * 3) >> 8) as i16 - 1) as f32 * d);
                }
            }
        }
        for &power in &POW3 {
            for &packed in &block[32..48] {
                let q = packed.wrapping_mul(power);
                output.push((((q as u16 * 3) >> 8) as i16 - 1) as f32 * d);
            }
        }
        for power in POW3.iter().take(4) {
            for &packed in &block[48..52] {
                let q = packed.wrapping_mul(*power);
                output.push((((q as u16 * 3) >> 8) as i16 - 1) as f32 * d);
            }
        }
    }
    output
}

fn dequant_tq2_0(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 66 * 256);
    for block in bytes.chunks_exact(66) {
        let d = f16_at(block, 64);
        for packed_group in block[..64].chunks_exact(32) {
            for plane in 0..4 {
                output.extend(
                    packed_group
                        .iter()
                        .map(|&packed| (((packed >> (2 * plane)) & 3) as f32 - 1.0) * d),
                );
            }
        }
    }
    output
}

fn e8m0_half_to_f32(encoded: u8) -> f32 {
    let bits = if encoded < 2 {
        0x0020_0000u32 << encoded
    } else {
        ((encoded as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

fn ue4m3_to_f32(encoded: u8) -> f32 {
    if encoded == 0 || encoded == 0x7f {
        return 0.0;
    }
    let exponent = (encoded >> 3) & 0x0f;
    let mantissa = encoded & 7;
    if exponent == 0 {
        mantissa as f32 * 2f32.powi(-10)
    } else {
        f32::from_bits(((119 + exponent as u32) << 23) | ((mantissa as u32) << 20))
    }
}

fn dequant_mxfp4(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 17 * 32);
    for block in bytes.chunks_exact(17) {
        let d = e8m0_half_to_f32(block[0]);
        for &packed in &block[1..17] {
            output.push(d * KVALUES_FP4[(packed & 0x0f) as usize]);
        }
        for &packed in &block[1..17] {
            output.push(d * KVALUES_FP4[(packed >> 4) as usize]);
        }
    }
    output
}

fn dequant_nvfp4(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 36 * 64);
    for block in bytes.chunks_exact(36) {
        for subblock in 0..4 {
            let d = ue4m3_to_f32(block[subblock]);
            let packed = &block[4 + subblock * 8..12 + subblock * 8];
            for &q in packed {
                output.push(d * KVALUES_FP4[(q & 0x0f) as usize]);
            }
            for &q in packed {
                output.push(d * KVALUES_FP4[(q >> 4) as usize]);
            }
        }
    }
    output
}

fn dequant_q1_0(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 18 * 128);
    for block in bytes.chunks_exact(18) {
        let d = f16_at(block, 0);
        for index in 0..128 {
            let bit = (block[2 + index / 8] >> (index % 8)) & 1;
            output.push(if bit == 0 { -d } else { d });
        }
    }
    output
}

fn dequant_q2_0(bytes: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(bytes.len() / 18 * 64);
    for block in bytes.chunks_exact(18) {
        let d = f16_at(block, 0);
        for index in 0..64 {
            let q = (block[2 + index / 4] >> (2 * (index % 4))) & 3;
            output.push((q as f32 - 1.0) * d);
        }
    }
    output
}

fn dequant_iq_blocks(
    bytes: &[u8],
    block_bytes: usize,
    dequant_fn: fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    let mut output = vec![0.0; bytes.len() / block_bytes * 256];
    dequant_iq_blocks_to_slice(bytes, &mut output, block_bytes, dequant_fn);
    output
}

fn dequant_iq_blocks_to_slice(
    bytes: &[u8],
    output: &mut [f32],
    block_bytes: usize,
    dequant_fn: fn(&[u8], &mut [f32]),
) -> bool {
    if bytes.len() % block_bytes != 0 || output.len() != bytes.len() / block_bytes * 256 {
        return false;
    }
    bytes
        .par_chunks_exact(block_bytes)
        .zip(output.par_chunks_mut(256))
        .for_each(|(block, out)| dequant_fn(block, out));
    true
}

fn dequant_k_blocks<T>(
    bytes: &[u8],
    block_bytes: usize,
    elems_per_block: usize,
    dequant_fn: fn(&T, &mut [f32; 256]),
) -> Vec<f32> {
    let n_blocks = bytes.len() / block_bytes;
    let mut out = vec![0.0f32; n_blocks * elems_per_block];
    bytes
        .par_chunks_exact(block_bytes)
        .zip(out.par_chunks_mut(elems_per_block))
        .for_each(|(chunk, dst)| {
            let block = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const T) };
            let mut tmp = [0.0f32; 256];
            dequant_fn(&block, &mut tmp);
            dst.copy_from_slice(&tmp);
        });
    out
}

fn dequant_basic_blocks<T>(
    bytes: &[u8],
    block_bytes: usize,
    elems_per_block: usize,
    dequant_fn: fn(&T, &mut [f32; 32]),
) -> Vec<f32> {
    let n_blocks = bytes.len() / block_bytes;
    let mut out = vec![0.0f32; n_blocks * elems_per_block];
    for (bi, chunk) in bytes.chunks_exact(block_bytes).enumerate() {
        let block = unsafe { &*(chunk.as_ptr() as *const T) };
        let mut tmp = [0.0f32; 32];
        dequant_fn(block, &mut tmp);
        out[bi * elems_per_block..(bi + 1) * elems_per_block].copy_from_slice(&tmp);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{dequantize_bytes_to_f32, dequantize_row_to_slice_if_supported, DequantType};

    #[test]
    fn dequant_f32_roundtrips_bytes() {
        let bytes = [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat();

        assert_eq!(
            dequantize_bytes_to_f32(&bytes, DequantType::F32),
            vec![1.5, -2.0]
        );
    }

    #[test]
    fn dequant_bf16_reads_high_bits() {
        let bytes = [
            ((1.0f32.to_bits() >> 16) as u16).to_le_bytes(),
            (((-4.0f32).to_bits() >> 16) as u16).to_le_bytes(),
        ]
        .concat();

        assert_eq!(
            dequantize_bytes_to_f32(&bytes, DequantType::BF16),
            vec![1.0, -4.0]
        );
    }

    #[test]
    fn dequant_row_to_slice_f32_validates_shape() {
        let bytes = [3.0f32.to_le_bytes(), 4.0f32.to_le_bytes()].concat();
        let mut output = [0.0f32; 2];

        assert!(dequantize_row_to_slice_if_supported(
            &bytes,
            DequantType::F32,
            &mut output
        ));
        assert_eq!(output, [3.0, 4.0]);
    }

    #[test]
    fn dequant_iq4_xs_reads_scales_and_non_linear_values() {
        let mut bytes = vec![0u8; 136];
        bytes[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        bytes[2..4].copy_from_slice(&0xaaaau16.to_le_bytes());
        bytes[4..8].copy_from_slice(&[0x11, 0x11, 0x11, 0x11]);
        bytes[8..].fill(0xf0);

        let out = dequantize_bytes_to_f32(&bytes, DequantType::IQ4XS);

        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -127.0);
        assert_eq!(out[16], 113.0);
        assert_eq!(out[32], -127.0);
    }

    #[test]
    fn ternary_blocks_match_ggml_reference() {
        fn hash(values: &[f32]) -> u64 {
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .fold(0xcbf29ce484222325, |hash, byte| {
                    (hash ^ byte as u64).wrapping_mul(0x100000001b3)
                })
        }

        let mut tq1 = (0..54)
            .map(|index| (index * 73 + 19) as u8)
            .collect::<Vec<_>>();
        tq1[52..54].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        assert_eq!(
            hash(&dequantize_bytes_to_f32(&tq1, DequantType::TQ1_0)),
            0x07250703215a1458
        );

        let mut tq2 = (0..66)
            .map(|index| (index * 73 + 19) as u8)
            .collect::<Vec<_>>();
        tq2[64..66].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        assert_eq!(
            hash(&dequantize_bytes_to_f32(&tq2, DequantType::TQ2_0)),
            0x8f13210451dcf918
        );
    }
}
