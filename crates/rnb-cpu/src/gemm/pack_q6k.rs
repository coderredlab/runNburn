//! Q6_K → i8mm packed layout 변환
//!
//! # Packed 블록 레이아웃 (NR=8 rows per group, super-block 1개 기준)
//!
//! ```text
//! qs:      [4 pairs × 32 chunks × 16B]          = 2048B
//!          pair p, chunk k: [row(p*2)[k*8..(k+1)*8] | row(p*2+1)[k*8..(k+1)*8]]
//!          signed i8 values (-32..31), stored as u8 (bit pattern preserved)
//! sc_raw:  [8][16] i8 (raw signed scale values)  = 128B
//! d:       [8] f32                                = 32B
//! ```
//!
//! 총 Q6K_PACKED_BLOCK_BYTES = 2048 + 128 + 32 = 2208 bytes per 8-row group
//!
//! Q6_K는 dmin 없음, bias correction 불필요.
//! Q4_K/Q5_K 대비: 16 sub-blocks × 16 elements, scales는 signed i8.

use half::f16;

// ─── 오프셋 상수 ─────────────────────────────────────────────────

/// qs 시작 오프셋: 0
/// 4 pairs × 32 chunks × 16B = 2048B
pub const Q6K_QS_OFF: usize = 0;

/// sc_raw 시작 오프셋: 2048
/// [8 rows][16 sub-blocks] i8 = 128B
pub const Q6K_SC_RAW_OFF: usize = 2048;

/// d 시작 오프셋: 2048 + 128 = 2176
/// [8] f32 = 32B
pub const Q6K_D_OFF: usize = Q6K_SC_RAW_OFF + 128;

/// packed 블록 전체 바이트: 2176 + 32 = 2208
pub const Q6K_PACKED_BLOCK_BYTES: usize = Q6K_D_OFF + 32;

// ─── Q6_K 블록 크기 ──────────────────────────────────────────────

/// Q6_K 원본 블록 바이트 (ql:128 + qh:64 + scales:16 + d:2 = 210)
const Q6K_BLOCK_BYTES: usize = 210;

// ─── 6-bit 언팩 → signed i8 변환 ─────────────────────────────────

/// Q6_K 블록의 6-bit 값을 256개 signed i8로 언팩.
///
/// Q6_K 레이아웃:
/// - ql[0..128]: low 4 bits
/// - qh[0..64]:  high 2 bits
///
/// 실제 배치는 llama.cpp / `crate::quantize::dequantize_q6_k`와 동일하게
/// 2개의 128원소 그룹 안에서 32원소 인터리브 순서로 펼친다:
/// - `out[base + l]`
/// - `out[base + l + 32]`
/// - `out[base + l + 64]`
/// - `out[base + l + 96]`
pub fn unpack_q6k(ql: &[u8; 128], qh: &[u8; 64], out: &mut [i8; 256]) {
    for n in 0..2usize {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let out_base = n * 128;

        for l in 0..32usize {
            out[out_base + l] =
                ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i8 - 32;
            out[out_base + l + 32] =
                ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i8 - 32;
            out[out_base + l + 64] =
                ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i8 - 32;
            out[out_base + l + 96] =
                ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}

/// Q6_K 블록의 6-bit 값을 256개 unsigned u8로 언팩 (0..63).
///
/// `unpack_q6k`와 동일하되 -32 하지 않음.
#[cfg(test)]
fn unpack_q6k_unsigned(ql: &[u8; 128], qh: &[u8; 64], out: &mut [u8; 256]) {
    for n in 0..2usize {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let out_base = n * 128;

        for l in 0..32usize {
            out[out_base + l] = (ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4);
            out[out_base + l + 32] =
                (ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4);
            out[out_base + l + 64] = (ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4);
            out[out_base + l + 96] =
                (ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4);
        }
    }
}

// ─── 메인 pack 함수 ───────────────────────────────────────────────

/// Q6_K 가중치 → row-pair interleaved packed layout 변환.
///
/// # 인자
/// - `src_bytes`: GGUF 원본 Q6_K 바이트 스트림 (rows × blocks_per_row × 210 bytes)
/// - `rows`: 가중치 행 수 (out_features)
/// - `cols`: 슈퍼블록 수 (cols_in_blocks = in_features / 256)
///
/// # 반환
/// NR=8 행 그룹 단위로 packed된 Vec<u8>
/// 크기: ceil(rows/8) × cols × Q6K_PACKED_BLOCK_BYTES
///
/// # 나머지 행 처리
/// rows % 8 != 0이면 마지막 그룹에서 모자란 행은 0으로 패딩됨.
///
/// # qs 인터리빙 구조
/// pair p (=0..3): even row = p*2, odd row = p*2+1
/// 256 elements = 32 chunks of 8 bytes each
/// chunk k의 16B = [even_row[k*8..k*8+8] | odd_row[k*8..k*8+8]]
/// → smmla용 vld1q_s8로 두 row 동시 로드 가능
pub fn pack_q6k(src_bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let row_groups = rows.div_ceil(8);
    let total_packed_bytes = row_groups * cols * Q6K_PACKED_BLOCK_BYTES;
    let mut out = vec![0u8; total_packed_bytes];

    // 임시 버퍼: 각 row의 signed 6-bit i8 [8][256]
    let mut unpacked_rows = [[0i8; 256]; 8];

    for rg in 0..row_groups {
        let base_row = rg * 8;

        for col in 0..cols {
            let out_off = (rg * cols + col) * Q6K_PACKED_BLOCK_BYTES;
            let packed = &mut out[out_off..out_off + Q6K_PACKED_BLOCK_BYTES];

            let mut sc_raw_all = [[0i8; 16]; 8];
            let mut d_all = [0.0f32; 8];

            // 임시 버퍼 초기화
            for r in 0..8 {
                unpacked_rows[r] = [0i8; 256];
            }

            for nr in 0..8 {
                let row = base_row + nr;
                if row >= rows {
                    continue;
                }

                let src_off = (row * cols + col) * Q6K_BLOCK_BYTES;
                let block = &src_bytes[src_off..src_off + Q6K_BLOCK_BYTES];

                let ql: &[u8; 128] = block[0..128].try_into().unwrap();
                let qh: &[u8; 64] = block[128..192].try_into().unwrap();
                let scales_raw: &[i8; 16] =
                    unsafe { &*(block[192..208].as_ptr() as *const [i8; 16]) };
                let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

                // signed 6-bit unpack (-32..31)
                unpack_q6k(ql, qh, &mut unpacked_rows[nr]);

                sc_raw_all[nr] = *scales_raw;
                d_all[nr] = d;
            }

            // ─── qs 인터리빙: pair-interleaved at 8-byte granularity ───
            // pair p (0..3): even = p*2, odd = p*2+1
            // 32 chunks × 16B per pair = 512B per pair × 4 pairs = 2048B
            for p in 0..4usize {
                let even = p * 2;
                let odd = p * 2 + 1;
                for k in 0..32usize {
                    let dst_off = Q6K_QS_OFF + p * 512 + k * 16;
                    // even row's 8 bytes (i8 → u8 bit pattern)
                    for i in 0..8 {
                        packed[dst_off + i] = unpacked_rows[even][k * 8 + i] as u8;
                    }
                    // odd row's 8 bytes
                    for i in 0..8 {
                        packed[dst_off + 8 + i] = unpacked_rows[odd][k * 8 + i] as u8;
                    }
                }
            }

            // ─── sc_raw: [8][16] i8 ───
            for nr in 0..8 {
                let off = Q6K_SC_RAW_OFF + nr * 16;
                for i in 0..16 {
                    packed[off + i] = sc_raw_all[nr][i] as u8;
                }
            }

            // ─── d: [8] f32 ───
            for nr in 0..8 {
                let off = Q6K_D_OFF + nr * 4;
                packed[off..off + 4].copy_from_slice(&d_all[nr].to_le_bytes());
            }
        }
    }

    out
}

// ─── 헬퍼: packed 블록 읽기 ──────────────────────────────────────

/// packed 블록에서 row nr의 qs[256] u8 읽기 (deinterleave)
///
/// 인터리빙된 레이아웃에서 특정 row의 256개 원소를 추출함.
pub fn read_packed_qs(packed: &[u8], nr: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    let pair = nr / 2;
    let is_odd = nr % 2;
    let pair_base = Q6K_QS_OFF + pair * 512;

    for k in 0..32usize {
        let chunk_off = pair_base + k * 16 + is_odd * 8;
        out[k * 8..k * 8 + 8].copy_from_slice(&packed[chunk_off..chunk_off + 8]);
    }

    out
}

/// packed 블록에서 row nr의 sc_raw[16] i8 읽기
pub fn read_packed_sc_raw(packed: &[u8], nr: usize) -> [i8; 16] {
    let base = Q6K_SC_RAW_OFF + nr * 16;
    let mut out = [0i8; 16];
    for i in 0..16 {
        out[i] = packed[base + i] as i8;
    }
    out
}

/// packed 블록에서 row nr의 d f32 읽기
pub fn read_packed_d(packed: &[u8], nr: usize) -> f32 {
    let base = Q6K_D_OFF + nr * 4;
    f32::from_le_bytes(packed[base..base + 4].try_into().unwrap())
}

// ─── 테스트 ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::{dequantize_q6_k, BlockQ6_K};
    use half::f16;

    /// 테스트용 Q6_K 더미 블록 생성 (210 bytes)
    fn make_q6k_block(d_val: f32, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        block[0..128].copy_from_slice(&ql);
        block[128..192].copy_from_slice(&qh);
        // i8 → u8 transmute (same bit pattern)
        for (i, &s) in scales.iter().enumerate() {
            block[192 + i] = s as u8;
        }
        block[208..210].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block
    }

    #[test]
    fn test_unpack_q6k_matches_rnb_cpu_dequant_layout() {
        let mut ql = [0u8; 128];
        for (i, q) in ql.iter_mut().enumerate() {
            *q = ((i * 7 + 3) % 256) as u8;
        }
        let mut qh = [0u8; 64];
        for (i, q) in qh.iter_mut().enumerate() {
            *q = ((i * 11 + 5) % 256) as u8;
        }
        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (((i as i32 * 5) % 63) - 31) as i8;
        }

        let block_bytes = make_q6k_block(0.125, scales, ql, qh);
        let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ6_K) };

        let mut unpacked = [0i8; 256];
        unpack_q6k(&ql, &qh, &mut unpacked);

        let mut dequant = [0.0f32; 256];
        dequantize_q6_k(block, &mut dequant);

        for i in 0..256 {
            let scale = scales[i / 16] as f32;
            let expected = if scale == 0.0 {
                0.0
            } else {
                dequant[i] / (0.125 * scale)
            };
            assert!(
                (unpacked[i] as f32 - expected).abs() < 1e-4,
                "idx={i} unpacked={} expected={expected}",
                unpacked[i]
            );
        }
    }

    // ─── 오프셋 상수 확인 ────────────────────────────────────────

    #[test]
    fn test_pack_q6k_offset_constants() {
        assert_eq!(Q6K_QS_OFF, 0);
        assert_eq!(Q6K_SC_RAW_OFF, 2048);
        assert_eq!(Q6K_D_OFF, 2176);
        assert_eq!(Q6K_PACKED_BLOCK_BYTES, 2208);
    }

    // ─── shape 정확성 ────────────────────────────────────────────

    #[test]
    fn test_pack_q6k_shape_exact() {
        let block = make_q6k_block(1.0, [1i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q6k(&src, 8, 1);
        assert_eq!(packed.len(), 1 * 1 * Q6K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_q6k_shape_multi_col() {
        let block = make_q6k_block(1.0, [1i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(8 * 4);
        let packed = pack_q6k(&src, 8, 4);
        assert_eq!(packed.len(), 1 * 4 * Q6K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_q6k_shape_multi_rowgroup() {
        let block = make_q6k_block(1.0, [1i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(16 * 2);
        let packed = pack_q6k(&src, 16, 2);
        assert_eq!(packed.len(), 2 * 2 * Q6K_PACKED_BLOCK_BYTES);
    }

    // ─── 6-bit 언팩 → signed i8 변환 ─────────────────────────────

    #[test]
    fn test_pack_q6k_unpack_all_zero() {
        let ql = [0u8; 128];
        let qh = [0u8; 64];
        let mut out = [0i8; 256];
        unpack_q6k(&ql, &qh, &mut out);
        assert!(out.iter().all(|&x| x == -32), "all-zero should be -32");
    }

    #[test]
    fn test_pack_q6k_unpack_all_max() {
        let ql = [0xFFu8; 128];
        let qh = [0xFFu8; 64];
        let mut out = [0i8; 256];
        unpack_q6k(&ql, &qh, &mut out);
        assert!(out.iter().all(|&x| x == 31), "all-max should be 31");
    }

    #[test]
    fn test_pack_q6k_unpack_unsigned_all_zero() {
        let ql = [0u8; 128];
        let qh = [0u8; 64];
        let mut out = [0u8; 256];
        unpack_q6k_unsigned(&ql, &qh, &mut out);
        assert!(out.iter().all(|&x| x == 0), "unsigned all-zero should be 0");
    }

    #[test]
    fn test_pack_q6k_unpack_unsigned_all_max() {
        let ql = [0xFFu8; 128];
        let qh = [0xFFu8; 64];
        let mut out = [0u8; 256];
        unpack_q6k_unsigned(&ql, &qh, &mut out);
        assert!(
            out.iter().all(|&x| x == 63),
            "unsigned all-max should be 63"
        );
    }

    #[test]
    fn test_pack_q6k_unpack_unsigned_range() {
        let ql = [0x5Au8; 128];
        let qh = [0xA5u8; 64];
        let mut out = [0u8; 256];
        unpack_q6k_unsigned(&ql, &qh, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(v <= 63, "elem {i}: value {v} out of range 0..63");
        }
    }

    #[test]
    fn test_pack_q6k_unpack_signed_unsigned_consistency() {
        // unsigned = signed + 32
        let ql = [0x5Au8; 128];
        let qh = [0xA5u8; 64];
        let mut signed = [0i8; 256];
        let mut unsigned = [0u8; 256];
        unpack_q6k(&ql, &qh, &mut signed);
        unpack_q6k_unsigned(&ql, &qh, &mut unsigned);

        for i in 0..256 {
            let expected = (signed[i] as i32 + 32) as u8;
            assert_eq!(
                unsigned[i], expected,
                "elem {i}: unsigned={} != signed+32={}",
                unsigned[i], expected
            );
        }
    }

    // ─── 나머지 행 패딩 ──────────────────────────────────────────

    #[test]
    fn test_pack_q6k_remainder_rows_padding() {
        let block = make_q6k_block(2.0, [5i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(5);
        let packed = pack_q6k(&src, 5, 1);

        assert_eq!(packed.len(), Q6K_PACKED_BLOCK_BYTES);

        // row 0..4 → d가 2.0이어야 함
        for nr in 0..5 {
            let d = read_packed_d(&packed, nr);
            assert!((d - 2.0).abs() < 0.01, "row {nr}: d={d} (expected ~2.0)");
        }

        // row 5..7 → 패딩이라 d=0
        for nr in 5..8 {
            let d = read_packed_d(&packed, nr);
            assert_eq!(d, 0.0, "padding row {nr}: d should be 0.0");
        }
    }

    #[test]
    fn test_pack_q6k_remainder_rows_qs_padding() {
        let block = make_q6k_block(1.0, [1i8; 16], [0xFFu8; 128], [0xFFu8; 64]);
        let src: Vec<u8> = block.repeat(3);
        let packed = pack_q6k(&src, 3, 1);

        // row 0..2: ql=0xFF, qh=0xFF → signed=31
        for nr in 0..3 {
            let qs = read_packed_qs(&packed, nr);
            assert!(qs.iter().all(|&b| b as i8 == 31), "row {nr} qs must be 31");
        }

        // row 3..7 qs는 0 (패딩)
        for nr in 3..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(qs.iter().all(|&b| b == 0), "padding row {nr} qs must be 0");
        }
    }

    // ─── pack 후 값 무결성 검증 ──────────────────────────────────

    #[test]
    fn test_pack_q6k_qs_value_correctness() {
        // ql=0xFF, qh=0xFF → 6-bit=63 → signed=31 → as u8 = 31
        let block = make_q6k_block(1.0, [1i8; 16], [0xFFu8; 128], [0xFFu8; 64]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q6k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(
                qs.iter().all(|&b| b as i8 == 31),
                "row {nr}: qs should all be 31 (signed max)"
            );
        }
    }

    #[test]
    fn test_pack_q6k_qs_zero_value() {
        // ql=0, qh=0 → 6-bit=0 → signed=-32 → as u8 = 0xE0 = 224
        let block = make_q6k_block(1.0, [1i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q6k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(
                qs.iter().all(|&b| b as i8 == -32),
                "row {nr}: qs should all be -32 (signed min)"
            );
        }
    }

    #[test]
    fn test_pack_q6k_d_value_correctness() {
        let d_val = 0.125f32;
        let block = make_q6k_block(d_val, [1i8; 16], [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q6k(&src, 8, 1);

        for nr in 0..8 {
            let d = read_packed_d(&packed, nr);
            assert!((d - d_val).abs() < 1e-4, "row {nr}: d={d} expected {d_val}");
        }
    }

    #[test]
    fn test_pack_q6k_sc_raw_correctness() {
        let mut scales = [0i8; 16];
        scales[0] = 10;
        scales[5] = -8;
        scales[15] = 127;
        let block = make_q6k_block(2.0, scales, [0u8; 128], [0u8; 64]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q6k(&src, 8, 1);

        for nr in 0..8 {
            let sc = read_packed_sc_raw(&packed, nr);
            assert_eq!(sc[0], 10, "row {nr} sc_raw[0] expected 10");
            assert_eq!(sc[5], -8, "row {nr} sc_raw[5] expected -8");
            assert_eq!(sc[15], 127, "row {nr} sc_raw[15] expected 127");
        }
    }

    // ─── 인터리빙 정확성 확인 ────────────────────────────────────

    #[test]
    fn test_pack_q6k_interleaving() {
        // 각 row마다 다른 ql 패턴으로 인터리빙 확인
        let mut src = Vec::new();
        for row in 0..8u8 {
            // row별 고유 패턴: ql byte = row+1 → low nibble = row+1
            let ql_byte = (row + 1) & 0x0F;
            let block = make_q6k_block(1.0, [1i8; 16], [ql_byte; 128], [0u8; 64]);
            src.extend_from_slice(&block);
        }
        let packed = pack_q6k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            // low nibble = (nr+1), qh=0 → 6-bit = (nr+1) → signed = (nr+1) - 32
            let expected_signed = ((nr as i8 + 1) & 0x0F) - 32;
            // 첫 64개 원소 (elem 0..63): low nibble of ql
            for i in 0..64 {
                assert_eq!(
                    qs[i] as i8, expected_signed,
                    "row {nr} elem {i}: qs={}, expected {expected_signed}",
                    qs[i] as i8
                );
            }
        }
    }

    // ─── 다중 col 인터리빙 확인 ──────────────────────────────────

    #[test]
    fn test_pack_q6k_multi_col_interleave() {
        let block0 = make_q6k_block(1.0, [2i8; 16], [0u8; 128], [0u8; 64]);
        let block1 = make_q6k_block(2.0, [3i8; 16], [0u8; 128], [0u8; 64]);

        let mut src = Vec::new();
        for _ in 0..8 {
            src.extend_from_slice(&block0);
            src.extend_from_slice(&block1);
        }

        let packed = pack_q6k(&src, 8, 2);
        assert_eq!(packed.len(), 1 * 2 * Q6K_PACKED_BLOCK_BYTES);

        let pb0 = &packed[0..Q6K_PACKED_BLOCK_BYTES];
        let pb1 = &packed[Q6K_PACKED_BLOCK_BYTES..2 * Q6K_PACKED_BLOCK_BYTES];

        let d0 = read_packed_d(pb0, 0);
        assert!((d0 - 1.0).abs() < 0.01, "col0 d={d0}");

        let d1 = read_packed_d(pb1, 0);
        assert!((d1 - 2.0).abs() < 0.01, "col1 d={d1}");

        let sc0 = read_packed_sc_raw(pb0, 0);
        assert_eq!(sc0[0], 2, "col0 sc_raw[0]=2");

        let sc1 = read_packed_sc_raw(pb1, 0);
        assert_eq!(sc1[0], 3, "col1 sc_raw[0]=3");
    }
}
