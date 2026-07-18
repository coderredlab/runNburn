//! GGUF block → IntScale conversion (Phase 1 Task 3 + 3b + 3c).
//!
//! Converts GGUF Q4_K / Q5_K / Q8_0 blocks into row-major IntScale form with
//! a row-level `row_multiplier: f32`. Designed for offline use in `rnb-convert`.
//!
//! **Sub-block fidelity**: `Q4KIntScale` / `Q5KIntScale` preserve the GGUF
//! 6-bit packed `scales: [u8; 12]` verbatim in `sub_scales`, so the dequant
//! functions reconstruct per-sub-block `sc` / `m` identically to their GGUF
//! counterparts. Only the super-block `d` / `dmin` are approximated as
//! `scale_int * row_mul` / `min_int * row_mul` (row-level i8 quantization).
//! Expected round-trip cosine similarity ≥ 0.99.
//!
//! **Q5_1 → Q5_K IntScale (Task 3c)**: Qwen3.6's `down_exps` come as Q5_1
//! 32-elem blocks. `row_q51_to_q5k_intscale` dequantizes Q5_1 → f32, then
//! re-quantizes f32 → Q5_K (via `quantize_row_q5_k`) → IntScale. Double
//! re-quantization, so threshold ≥ 0.97 applies against the original Q5_1
//! dequant output.

use crate::quantize::blocks::{BlockQ4_K, BlockQ5_1, BlockQ5_K, BlockQ6_K, BlockQ8_0};
use crate::quantize::dequant::{dequantize_q5_1, dequantize_q6_k};
use crate::quantize::moe_blocks::{Q4KIntScale, Q5KIntScale, Q80IntScale};
use crate::quantize::quant::get_scale_min_k4;

/// Convert a row of Q4_K blocks into Q4KIntScale form with a shared row multiplier.
///
/// The super-block `d` / `dmin` f16 values are row-normalized into i8 via the
/// max-abs row multiplier. The GGUF 6-bit packed `scales: [u8; 12]` and the
/// raw `qs: [u8; 128]` are preserved verbatim so sub-block scale fidelity is
/// maintained on round-trip.
pub fn row_q4k_to_intscale(row: &[BlockQ4_K]) -> (Vec<Q4KIntScale>, f32) {
    let mut max_abs = 0.0f32;
    for b in row {
        max_abs = max_abs.max(b.d.to_f32().abs()).max(b.dmin.to_f32().abs());
    }
    if max_abs == 0.0 {
        let mut out = Vec::with_capacity(row.len());
        for _ in 0..row.len() {
            out.push(Q4KIntScale {
                scale_int: 0,
                min_int: 0,
                sub_scales: [0u8; 12],
                qs: [0u8; 128],
            });
        }
        return (out, 0.0);
    }
    let row_mul = max_abs / 127.0;
    let inv_mul = 1.0 / row_mul;

    let mut out = Vec::with_capacity(row.len());
    for b in row {
        let d = b.d.to_f32();
        let dmin = b.dmin.to_f32();
        let scale_int = (d * inv_mul).round().clamp(-127.0, 127.0) as i8;
        let min_int = (dmin * inv_mul).round().clamp(-127.0, 127.0) as i8;
        out.push(Q4KIntScale {
            scale_int,
            min_int,
            sub_scales: b.scales, // preserve GGUF 6-bit packed scales verbatim
            qs: b.qs,
        });
    }
    (out, row_mul)
}

/// Dequantize a Q4KIntScale block back to f32 for round-trip verification.
/// NOT hot-path; for tests and tooling.
///
/// Mirrors `dequantize_q4_k` exactly, including sub-block 6-bit `sc` / `m`
/// scaling:
///   d_s = (scale_int * row_mul) * sc[s]
///   m_s = (min_int   * row_mul) * m[s]
/// qs layout: 4 groups of 64 elements, each group consumes 32 bytes of qs
/// (first 32 outputs use low nibbles, next 32 outputs use high nibbles).
pub fn dequantize_q4k_intscale(b: &Q4KIntScale, row_mul: f32) -> [f32; 256] {
    let d = b.scale_int as f32 * row_mul;
    let dmin = b.min_int as f32 * row_mul;
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        let (s, mm) = get_scale_min_k4(j, &b.sub_scales);
        sc[j] = s;
        m[j] = mm;
    }
    let mut out = [0.0f32; 256];

    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut y_off = 0usize;
    for _ in 0..4 {
        let d1 = d * sc[is] as f32;
        let m1 = dmin * m[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * m[is + 1] as f32;

        for l in 0..32 {
            out[y_off + l] = d1 * ((b.qs[q_off + l] & 0x0F) as f32) - m1;
        }
        for l in 0..32 {
            out[y_off + 32 + l] = d2 * ((b.qs[q_off + l] >> 4) as f32) - m2;
        }
        q_off += 32;
        is += 2;
        y_off += 64;
    }
    out
}

/// Convert a row of Q5_K blocks into Q5KIntScale form with a shared row multiplier.
///
/// Mirrors `row_q4k_to_intscale` exactly, but operates on `BlockQ5_K`
/// (which has an extra `qh: [u8; 32]` high-bit field). The GGUF 6-bit packed
/// `scales: [u8; 12]`, `qs: [u8; 128]`, and `qh: [u8; 32]` are all preserved
/// verbatim so sub-block scale + 5-bit element fidelity is maintained.
pub fn row_q5k_to_intscale(row: &[BlockQ5_K]) -> (Vec<Q5KIntScale>, f32) {
    let mut max_abs = 0.0f32;
    for b in row {
        max_abs = max_abs.max(b.d.to_f32().abs()).max(b.dmin.to_f32().abs());
    }
    if max_abs == 0.0 {
        let zero = Q5KIntScale {
            scale_int: 0,
            min_int: 0,
            sub_scales: [0u8; 12],
            qs_low: [0u8; 128],
            qs_high: [0u8; 32],
        };
        return (vec![zero; row.len()], 0.0);
    }
    let row_mul = max_abs / 127.0;
    let inv_mul = 1.0 / row_mul;

    let mut out = Vec::with_capacity(row.len());
    for b in row {
        let d = b.d.to_f32();
        let dmin = b.dmin.to_f32();
        let scale_int = (d * inv_mul).round().clamp(-127.0, 127.0) as i8;
        let min_int = (dmin * inv_mul).round().clamp(-127.0, 127.0) as i8;
        out.push(Q5KIntScale {
            scale_int,
            min_int,
            sub_scales: b.scales,
            qs_low: b.qs,
            qs_high: b.qh,
        });
    }
    (out, row_mul)
}

/// Dequantize a Q5KIntScale block back to f32 for round-trip verification.
/// NOT hot-path; for tests and tooling.
///
/// Mirrors `dequantize_q5_k` exactly:
///   d = scale_int * row_mul
///   dmin = min_int   * row_mul
///   for j in 0..4:
///     d1 = d * sc[is], m1 = dmin * m[is]   (elems 0..32 of group)
///     d2 = d * sc[is+1], m2 = dmin * m[is+1] (elems 32..64 of group)
///     out[..] = d_x * ((qs_low & 0x0F) + qh_bit*16) - m_x
/// qh bit advances by 2 positions per group (u1, u2 masks).
pub fn dequantize_q5k_intscale(b: &Q5KIntScale, row_mul: f32) -> [f32; 256] {
    let d = b.scale_int as f32 * row_mul;
    let dmin = b.min_int as f32 * row_mul;
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        let (s, mm) = get_scale_min_k4(j, &b.sub_scales);
        sc[j] = s;
        m[j] = mm;
    }
    let mut out = [0.0f32; 256];

    let mut is = 0usize;
    let mut ql_off = 0usize;
    let mut y_off = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for _ in 0..4 {
        let d1 = d * sc[is] as f32;
        let m1 = dmin * m[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * m[is + 1] as f32;

        for l in 0..32 {
            let high: u8 = if b.qs_high[l] & u1 != 0 { 16 } else { 0 };
            out[y_off + l] = d1 * ((b.qs_low[ql_off + l] & 0x0F) + high) as f32 - m1;
        }
        for l in 0..32 {
            let high: u8 = if b.qs_high[l] & u2 != 0 { 16 } else { 0 };
            out[y_off + 32 + l] = d2 * ((b.qs_low[ql_off + l] >> 4) + high) as f32 - m2;
        }

        ql_off += 32;
        is += 2;
        u1 = u1.wrapping_shl(2);
        u2 = u2.wrapping_shl(2);
        y_off += 64;
    }
    out
}

/// Convert a row of Q5_1 blocks (32-elem each) into Q5KIntScale form via
/// dequantize → f32 → quantize-to-Q5_K → IntScale.
///
/// Used for Qwen3.6's `down_exps`: GGUF stores them as Q5_1, but the MoE
/// decode layout expects Q5_K super-blocks (256 elems) to match the token
/// activation Q8K layout. Two re-quantizations (Q5_1 → f32 → Q5_K → IntScale)
/// so expected round-trip cos ≥ 0.97 against the original Q5_1 dequant.
///
/// Requires `row_q51.len() * 32` is a multiple of 256 (i.e. `row_q51.len() % 8 == 0`).
pub fn row_q51_to_q5k_intscale(row_q51: &[BlockQ5_1]) -> (Vec<Q5KIntScale>, f32) {
    use std::mem::size_of;
    assert!(
        (row_q51.len() * 32) % 256 == 0,
        "Q5_1 row must cover a multiple of 256 elems for Q5_K re-pack: got {} elems",
        row_q51.len() * 32
    );

    // Step A: dequantize Q5_1 → f32.
    let mut floats = Vec::with_capacity(row_q51.len() * 32);
    for b in row_q51 {
        let mut tmp = [0.0f32; 32];
        dequantize_q5_1(b, &mut tmp);
        floats.extend_from_slice(&tmp);
    }

    // Step B: quantize f32 → Q5_K bytes.
    let n_q5k_blocks = floats.len() / 256;
    let mut bytes = vec![0u8; n_q5k_blocks * size_of::<BlockQ5_K>()];
    crate::quantize::quant::quantize_row_q5_k(&floats, &mut bytes);

    // Step C: reinterpret bytes as BlockQ5_K.
    let mut q5k_blocks = Vec::with_capacity(n_q5k_blocks);
    for bi in 0..n_q5k_blocks {
        let offset = bi * size_of::<BlockQ5_K>();
        // SAFETY:
        //   * `BlockQ5_K` is `#[repr(C)]` with POD fields (`half::f16`, `[u8; _]`)
        //     and no padding, so any bit pattern of the correct size is a valid
        //     instance — byte-wise reinterpretation is well-defined.
        //   * `bytes` was sized exactly as
        //     `n_q5k_blocks * size_of::<BlockQ5_K>()` above, so
        //     `offset + size_of::<BlockQ5_K>() <= bytes.len()` holds for every
        //     `bi in 0..n_q5k_blocks`.
        //   * `Vec<u8>` only guarantees 1-byte alignment, but `BlockQ5_K`
        //     contains `half::f16` (2-byte align). Therefore we use
        //     `read_unaligned`, which tolerates any alignment.
        let block: BlockQ5_K =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const BlockQ5_K) };
        q5k_blocks.push(block);
    }

    // Step D: convert to IntScale.
    row_q5k_to_intscale(&q5k_blocks)
}

/// Convert a row of Q6_K blocks (256-elem each) into Q5KIntScale form via
/// dequantize → f32 → quantize-to-Q5_K → IntScale.
///
/// Used for Gemma4 26B MoE's `down_exps` when stored as Q6_K: the MoE
/// decode layout expects Q5_K super-blocks (256 elems) to match the token
/// activation Q8K layout. Two re-quantizations (Q6_K → f32 → Q5_K → IntScale)
/// so expected round-trip cos ≥ 0.97 against the original Q6_K dequant.
pub fn row_q6k_to_q5k_intscale(row_q6k: &[BlockQ6_K]) -> (Vec<Q5KIntScale>, f32) {
    use std::mem::size_of;

    // Step A: dequantize Q6_K → f32.
    let n_elems = row_q6k.len() * 256;
    let mut floats = vec![0.0f32; n_elems];
    for (bi, blk) in row_q6k.iter().enumerate() {
        let mut tmp = [0.0f32; 256];
        dequantize_q6_k(blk, &mut tmp);
        floats[bi * 256..(bi + 1) * 256].copy_from_slice(&tmp);
    }

    // Step B: quantize f32 → Q5_K bytes.
    let n_q5k_blocks = n_elems / 256;
    let mut bytes = vec![0u8; n_q5k_blocks * size_of::<BlockQ5_K>()];
    crate::quantize::quant::quantize_row_q5_k(&floats, &mut bytes);

    // Step C: reinterpret bytes as BlockQ5_K.
    let mut q5k_blocks = Vec::with_capacity(n_q5k_blocks);
    for bi in 0..n_q5k_blocks {
        let offset = bi * size_of::<BlockQ5_K>();
        // SAFETY:
        //   * `BlockQ5_K` is `#[repr(C)]` with POD fields (`half::f16`,
        //     `[u8; _]`) and no padding, so any bit pattern of the correct
        //     size is a valid instance — byte-wise reinterpretation is
        //     well-defined.
        //   * `bytes` was sized as `n_q5k_blocks * size_of::<BlockQ5_K>()`
        //     above, so `offset + size_of::<BlockQ5_K>() <= bytes.len()`
        //     holds for every `bi in 0..n_q5k_blocks`.
        //   * `Vec<u8>` only guarantees 1-byte alignment, but `BlockQ5_K`
        //     contains `half::f16` (2-byte align). Therefore we use
        //     `read_unaligned`, which tolerates any alignment.
        let block: BlockQ5_K =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const BlockQ5_K) };
        q5k_blocks.push(block);
    }

    // Step D: convert to IntScale.
    row_q5k_to_intscale(&q5k_blocks)
}

/// Convert a row of Q8_0 blocks (32-elem each) into Q80IntScale form.
pub fn row_q80_to_intscale(row: &[BlockQ8_0]) -> (Vec<Q80IntScale>, f32) {
    let mut max_abs = 0.0f32;
    for b in row {
        max_abs = max_abs.max(b.d.to_f32().abs());
    }
    if max_abs == 0.0 {
        let mut out = Vec::with_capacity(row.len());
        for _ in 0..row.len() {
            out.push(Q80IntScale {
                scale_int: 0,
                qs: [0i8; 32],
            });
        }
        return (out, 0.0);
    }
    let row_mul = max_abs / 32767.0;
    let inv_mul = 1.0 / row_mul;

    let mut out = Vec::with_capacity(row.len());
    for b in row {
        let d = b.d.to_f32();
        let scale_int = (d * inv_mul).round().clamp(-32767.0, 32767.0) as i16;
        out.push(Q80IntScale {
            scale_int,
            qs: b.qs,
        });
    }
    (out, row_mul)
}

/// Dequantize a Q80IntScale block to f32.
pub fn dequantize_q80_intscale(b: &Q80IntScale, row_mul: f32) -> [f32; 32] {
    let d = b.scale_int as f32 * row_mul;
    let mut out = [0.0f32; 32];
    for i in 0..32 {
        out[i] = d * (b.qs[i] as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::blocks::BlockQ5_K;
    use crate::quantize::dequant::{
        dequantize_q4_k, dequantize_q5_1, dequantize_q5_k, dequantize_q8_0,
    };
    use half::f16;

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-10)
    }

    /// Pack per-sub-block `sc[0..8]` / `m[0..8]` (each 6-bit) into the GGUF
    /// `scales: [u8; 12]` layout. Inverse of `get_scale_min_k4` /
    /// `extract_k_quant_scales`. Used to hand-craft test fixtures with
    /// non-uniform sub-block scales.
    ///
    /// Layout:
    ///   scales[0..4]:  bits 0-5 = sc[0..4] (6 bits), bits 6-7 = sc[4..8][4..6]
    ///   scales[4..8]:  bits 0-5 = m[0..4]  (6 bits), bits 6-7 = m[4..8][4..6]
    ///   scales[8..12]: bits 0-3 = sc[4..8][0..4],    bits 4-7 = m[4..8][0..4]
    fn pack_q4k_sub_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (sc[j] & 0x3F) | (((sc[j + 4] >> 4) & 0x03) << 6);
            scales[j + 4] = (m[j] & 0x3F) | (((m[j + 4] >> 4) & 0x03) << 6);
            scales[j + 8] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
        }
        scales
    }

    /// Build a synthetic BlockQ4_K with **non-uniform** sub-block scales.
    ///
    /// Target sub-block values: sc = [1,2,3,4,5,6,7,8], m = [9,10,11,12,13,14,15,16].
    /// These are packed via `pack_q4k_sub_scales` to produce a fixture that
    /// exercises every sub-block slot independently, so the round-trip test
    /// actually detects sub-block scale loss.
    fn synth_q4k_block(idx: usize) -> BlockQ4_K {
        // Pattern: d and dmin vary slightly by index so rows with multiple blocks
        // exercise the row-level max-abs path.
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = (i as u8).wrapping_mul(17);
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

    fn synth_q80_block(idx: usize) -> BlockQ8_0 {
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (((i + idx) as i8).wrapping_mul(5)).wrapping_sub(10);
        }
        BlockQ8_0 {
            d: f16::from_f32(0.04 * (1.0 + 0.1 * idx as f32)),
            qs,
        }
    }

    fn zero_q4k_block() -> BlockQ4_K {
        BlockQ4_K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0; 12],
            qs: [0; 128],
        }
    }

    fn zero_q80_block() -> BlockQ8_0 {
        BlockQ8_0 {
            d: f16::from_f32(0.0),
            qs: [0; 32],
        }
    }

    #[test]
    fn pack_unpack_q4k_sub_scales_roundtrip() {
        // Sanity: the hand-crafted packer must invert get_scale_min_k4.
        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m = [9u8, 10, 11, 12, 13, 14, 15, 16];
        let scales = pack_q4k_sub_scales(&sc, &m);
        let mut sc_back = [0u8; 8];
        let mut m_back = [0u8; 8];
        for j in 0..8 {
            let (s, mm) = get_scale_min_k4(j, &scales);
            sc_back[j] = s;
            m_back[j] = mm;
        }
        assert_eq!(sc_back, sc, "sc roundtrip mismatch, scales={:?}", scales);
        assert_eq!(m_back, m, "m roundtrip mismatch, scales={:?}", scales);
    }

    #[test]
    fn q4k_row_to_intscale_roundtrip_cos_above_099() {
        // Fixture uses non-uniform sub-block scales sc=[1..8], m=[9..16].
        // Dropping sub_scales would tank cos well below 0.99, so this test
        // actually enforces sub-block fidelity.
        let blocks: Vec<BlockQ4_K> = (0..4).map(synth_q4k_block).collect();
        let (intscale, row_mul) = row_q4k_to_intscale(&blocks);

        let mut original = Vec::with_capacity(blocks.len() * 256);
        for b in &blocks {
            let mut tmp = [0.0f32; 256];
            dequantize_q4_k(b, &mut tmp);
            original.extend_from_slice(&tmp);
        }

        let mut reconstructed = Vec::with_capacity(intscale.len() * 256);
        for b in &intscale {
            let tmp = dequantize_q4k_intscale(b, row_mul);
            reconstructed.extend_from_slice(&tmp);
        }

        assert_eq!(original.len(), reconstructed.len());
        let cos = cos_sim(&original, &reconstructed);
        eprintln!("Q4_K IntScale roundtrip cos = {}", cos);
        assert!(cos >= 0.99, "Q4_K IntScale roundtrip cos too low: {}", cos);
    }

    #[test]
    fn q80_row_to_intscale_roundtrip_cos_above_099() {
        let blocks: Vec<BlockQ8_0> = (0..4).map(synth_q80_block).collect();
        let (intscale, row_mul) = row_q80_to_intscale(&blocks);

        let mut original = Vec::with_capacity(blocks.len() * 32);
        for b in &blocks {
            let mut tmp = [0.0f32; 32];
            dequantize_q8_0(b, &mut tmp);
            original.extend_from_slice(&tmp);
        }

        let mut reconstructed = Vec::with_capacity(intscale.len() * 32);
        for b in &intscale {
            let tmp = dequantize_q80_intscale(b, row_mul);
            reconstructed.extend_from_slice(&tmp);
        }

        let cos = cos_sim(&original, &reconstructed);
        assert!(cos >= 0.99, "Q8_0 IntScale roundtrip cos too low: {}", cos);
    }

    #[test]
    fn q4k_zero_row_returns_zero_mul() {
        let blocks: Vec<BlockQ4_K> = (0..2).map(|_| zero_q4k_block()).collect();
        let (_intscale, row_mul) = row_q4k_to_intscale(&blocks);
        assert_eq!(row_mul, 0.0);
    }

    #[test]
    fn q80_zero_row_returns_zero_mul() {
        let blocks: Vec<BlockQ8_0> = (0..2).map(|_| zero_q80_block()).collect();
        let (_intscale, row_mul) = row_q80_to_intscale(&blocks);
        assert_eq!(row_mul, 0.0);
    }

    #[test]
    fn q5k_row_to_intscale_roundtrip_cos_above_099() {
        // 1024-elem (4 super-blocks) of varied magnitudes → quantize to Q5_K
        // → row_q5k_to_intscale → dequantize both paths → compare cos.
        let input: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 * 0.019;
                t.sin() * (1.0 + 0.3 * (t * 0.7).cos())
            })
            .collect();

        let mut bytes = vec![0u8; (input.len() / 256) * std::mem::size_of::<BlockQ5_K>()];
        crate::quantize::quant::quantize_row_q5_k(&input, &mut bytes);

        let n_blocks = input.len() / 256;
        let mut q5k_blocks = Vec::with_capacity(n_blocks);
        for bi in 0..n_blocks {
            let offset = bi * std::mem::size_of::<BlockQ5_K>();
            // SAFETY:
            //   * `BlockQ5_K` is `#[repr(C)]` POD (half::f16 + fixed-size u8
            //     arrays, no padding), so any correctly-sized bit pattern is
            //     a valid value.
            //   * `bytes` was sized as
            //     `(input.len() / 256) * size_of::<BlockQ5_K>()` above, so
            //     every `offset..offset+size_of::<BlockQ5_K>()` window is
            //     in-bounds.
            //   * `Vec<u8>` provides only 1-byte alignment while `BlockQ5_K`
            //     contains `half::f16` (2-byte align), so we must use
            //     `read_unaligned` rather than `read`.
            q5k_blocks.push(unsafe {
                std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const BlockQ5_K)
            });
        }

        // Original Q5_K dequant (reference).
        let mut original = Vec::with_capacity(input.len());
        for b in &q5k_blocks {
            let mut tmp = [0.0f32; 256];
            dequantize_q5_k(b, &mut tmp);
            original.extend_from_slice(&tmp);
        }

        // IntScale round-trip.
        let (intscale, row_mul) = row_q5k_to_intscale(&q5k_blocks);
        let mut reconstructed = Vec::with_capacity(input.len());
        for b in &intscale {
            let tmp = dequantize_q5k_intscale(b, row_mul);
            reconstructed.extend_from_slice(&tmp);
        }

        let cos = cos_sim(&original, &reconstructed);
        eprintln!("q5k_intscale roundtrip cos = {}", cos);
        assert!(cos >= 0.99, "Q5_K IntScale roundtrip cos too low: {}", cos);
    }

    #[test]
    fn q5k_zero_row_returns_zero_mul() {
        // Zero-d blocks → row_mul should be 0.0 and scale_int/min_int zero.
        let zero = BlockQ5_K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        let blocks: Vec<BlockQ5_K> = (0..2)
            .map(|_| BlockQ5_K {
                d: zero.d,
                dmin: zero.dmin,
                scales: zero.scales,
                qh: zero.qh,
                qs: zero.qs,
            })
            .collect();
        let (_intscale, row_mul) = row_q5k_to_intscale(&blocks);
        assert_eq!(row_mul, 0.0);
    }

    #[test]
    fn q6k_to_q5k_intscale_path_cos_above_097() {
        // Build 1 Q6_K super-block (256 elems) from a hand-crafted pattern
        // with varied scales/qh/ql, dequantize through the original Q6_K path
        // vs the new row_q6k_to_q5k_intscale → dequantize_q5k_intscale chain,
        // compare cosine similarity. Threshold 0.97 because the chain is
        // Q6_K → f32 → Q5_K → IntScale (two re-quantizations).
        use crate::quantize::blocks::BlockQ6_K;
        use crate::quantize::dequant::dequantize_q6_k;

        // Hand-crafted Q6_K block with non-zero scales to exercise every
        // sub-block slot. d = 0.02, scales vary across 16 sub-blocks.
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for i in 0..128 {
            ql[i] = (i as u8).wrapping_mul(23).wrapping_add(7);
        }
        for i in 0..64 {
            qh[i] = (i as u8).wrapping_mul(31).wrapping_add(3);
        }
        for i in 0..16 {
            // Keep within i8 range; make them non-uniform.
            scales[i] = (((i as i32) * 7 - 40) as i8).clamp(-60, 60);
        }
        let blocks = vec![BlockQ6_K {
            ql,
            qh,
            scales,
            d: f16::from_f32(0.02),
        }];

        // Original Q6_K dequant.
        let mut original = vec![0.0f32; 256];
        {
            let mut tmp = [0.0f32; 256];
            dequantize_q6_k(&blocks[0], &mut tmp);
            original.copy_from_slice(&tmp);
        }

        // Q6_K → Q5_K IntScale → dequant.
        let (intscale, row_mul) = row_q6k_to_q5k_intscale(&blocks);
        let mut reconstructed = Vec::with_capacity(256);
        for b in &intscale {
            let tmp = dequantize_q5k_intscale(b, row_mul);
            reconstructed.extend_from_slice(&tmp);
        }

        let cos = cos_sim(&original, &reconstructed);
        eprintln!("q6k_to_q5k_intscale roundtrip cos = {}", cos);
        assert!(cos >= 0.97, "Q6_K → Q5_K IntScale cos too low: {}", cos);
    }

    #[test]
    fn q51_to_q5k_intscale_path_cos_above_097() {
        // Build 8 Q5_1 blocks (256 elems total) from synthetic f32, convert to
        // Q5_K IntScale, dequantize, compare to the original Q5_1 dequant.
        // Threshold 0.97 because Q5_1 → f32 → Q5_K → IntScale is two re-quantizations.
        let mut q51_blocks: Vec<BlockQ5_1> = Vec::with_capacity(8);
        for bi in 0..8 {
            let mut tmp = [0.0f32; 32];
            for i in 0..32 {
                tmp[i] = ((bi * 32 + i) as f32 * 0.017).sin();
            }
            let max = tmp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = tmp.iter().cloned().fold(f32::INFINITY, f32::min);
            let d = if max > min { (max - min) / 31.0 } else { 1.0 };
            let m = min;
            let mut qs = [0u8; 16];
            let mut qh_bytes = [0u8; 4];
            for i in 0..32 {
                let q = ((tmp[i] - m) / d).round().clamp(0.0, 31.0) as u32;
                let qs_idx = i / 2;
                if i % 2 == 0 {
                    qs[qs_idx] |= (q & 0x0F) as u8;
                } else {
                    qs[qs_idx] |= ((q & 0x0F) << 4) as u8;
                }
                if (q & 0x10) != 0 {
                    qh_bytes[i / 8] |= 1 << (i % 8);
                }
            }
            q51_blocks.push(BlockQ5_1 {
                d: f16::from_f32(d),
                m: f16::from_f32(m),
                qh: qh_bytes,
                qs,
            });
        }

        // Original Q5_1 dequant.
        let mut original = Vec::with_capacity(256);
        for b in &q51_blocks {
            let mut tmp = [0.0f32; 32];
            dequantize_q5_1(b, &mut tmp);
            original.extend_from_slice(&tmp);
        }

        // Q5_1 → Q5_K IntScale → dequant.
        let (intscale, row_mul) = row_q51_to_q5k_intscale(&q51_blocks);
        let mut reconstructed = Vec::with_capacity(256);
        for b in &intscale {
            let tmp = dequantize_q5k_intscale(b, row_mul);
            reconstructed.extend_from_slice(&tmp);
        }

        let cos = cos_sim(&original, &reconstructed);
        eprintln!("q51_to_q5k_intscale roundtrip cos = {}", cos);
        assert!(cos >= 0.97, "Q5_1 → Q5_K IntScale cos too low: {}", cos);
    }
}
