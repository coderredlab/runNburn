//! Q5_K → i8mm packed layout 변환
//!
//! # Packed 블록 레이아웃 (NR=8 rows per group, super-block 1개 기준)
//!
//! Q4_K와 동일한 구조, 값 범위만 다름 (unsigned 0..31 vs 0..15):
//!
//! ```text
//! qs:      [4 pairs × 32 chunks × 16B]          = 2048B
//!          pair p, chunk k: [row(p*2)[k*8..(k+1)*8] | row(p*2+1)[k*8..(k+1)*8]]
//! sc_raw:  [8][8] u8 (raw 6-bit decoded values)  = 64B
//! mn_raw:  [8][8] u8 (raw 6-bit decoded values)  = 64B
//! d:       [8] f32                                = 32B
//! dmin:    [8] f32                                = 32B
//! ```
//!
//! 총 Q5K_PACKED_BLOCK_BYTES = 2240 bytes per 8-row group

use crate::gemm::pack_q4k::decode_q4k_scales_raw;
use half::f16;

// ─── 오프셋 상수 ─────────────────────────────────────────────────

/// qs 시작 오프셋: 0
/// 4 pairs × 32 chunks × 16B = 2048B
pub const Q5K_QS_OFF: usize = 0;

/// sc_raw 시작 오프셋: 2048
/// [8 rows][8 sub-blocks] u8 = 64B
pub const Q5K_SC_RAW_OFF: usize = 2048;

/// mn_raw 시작 오프셋: 2048 + 64 = 2112
/// [8 rows][8 sub-blocks] u8 = 64B
pub const Q5K_MN_RAW_OFF: usize = Q5K_SC_RAW_OFF + 64;

/// d 시작 오프셋: 2112 + 64 = 2176
/// [8] f32 = 32B
pub const Q5K_D_OFF: usize = Q5K_MN_RAW_OFF + 64;

/// dmin 시작 오프셋: 2176 + 32 = 2208
/// [8] f32 = 32B
pub const Q5K_DMIN_OFF: usize = Q5K_D_OFF + 32;

/// packed 블록 전체 바이트: 2208 + 32 = 2240
pub const Q5K_PACKED_BLOCK_BYTES: usize = Q5K_DMIN_OFF + 32;

// ─── Q5_K 블록 크기 ──────────────────────────────────────────────

/// Q5_K 원본 블록 바이트 (d:2 + dmin:2 + scales:12 + qh:32 + qs:128 = 176)
const Q5K_BLOCK_BYTES: usize = 176;

// ─── 5-bit → unsigned u8 변환 ────────────────────────────────────

/// Q5_K 블록의 5-bit값을 256개 unsigned u8로 언팩 (0..31).
///
/// Q5_K 레이아웃: 4그룹 × 64개
///   group g (0..3), within group index l (0..63):
///   - l < 32:  low = qs[g*32 + l] & 0x0F,  high_bit = (qh[l] >> g) & 1
///   - l >= 32: low = qs[g*32 + (l-32)] >> 4, high_bit = (qh[l-32] >> (g+4)) & 1
///   val_unsigned = low | (high_bit << 4)  // 0..31
pub(crate) fn unpack_q5k_bits_unsigned(qs: &[u8; 128], qh: &[u8; 32], out: &mut [u8; 256]) {
    for g in 0..4usize {
        let group_out_base = g * 64;
        let group_qs_base = g * 32;

        // l < 32: low nibble — high bit at qh[l] bit (2*g)
        for l in 0..32usize {
            let low = (qs[group_qs_base + l] & 0x0F) as u8;
            let high_bit = (qh[l] >> (2 * g)) & 1;
            out[group_out_base + l] = low | (high_bit << 4);
        }

        // l >= 32: high nibble — high bit at qh[l-32] bit (2*g+1)
        for l in 32..64usize {
            let low = qs[group_qs_base + (l - 32)] >> 4;
            let high_bit = (qh[l - 32] >> (2 * g + 1)) & 1;
            out[group_out_base + l] = low | (high_bit << 4);
        }
    }
}

// ─── 메인 pack 함수 ───────────────────────────────────────────────

/// Q5_K 가중치 → row-pair interleaved packed layout 변환.
///
/// # 인자
/// - `src_bytes`: GGUF 원본 Q5_K 바이트 스트림 (rows × blocks_per_row × 176 bytes)
/// - `rows`: 가중치 행 수 (out_features)
/// - `cols`: 슈퍼블록 수 (cols_in_blocks = in_features / 256)
///
/// # 반환
/// NR=8 행 그룹 단위로 packed된 Vec<u8>
/// 크기: ceil(rows/8) × cols × Q5K_PACKED_BLOCK_BYTES
///
/// # 나머지 행 처리
/// rows % 8 != 0이면 마지막 그룹에서 모자란 행은 0으로 패딩됨.
///
/// # qs 인터리빙 구조
/// pair p (=0..3): even row = p*2, odd row = p*2+1
/// 256 elements = 32 chunks of 8 bytes each
/// chunk k의 16B = [even_row[k*8..k*8+8] | odd_row[k*8..k*8+8]]
/// → smmla용 vld1q_s8로 두 row 동시 로드 가능
pub fn pack_q5k(src_bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let row_groups = rows.div_ceil(8);
    let total_packed_bytes = row_groups * cols * Q5K_PACKED_BLOCK_BYTES;
    let mut out = vec![0u8; total_packed_bytes];

    // 임시 버퍼: 각 row의 unsigned 5-bit [8][256]
    let mut unpacked_rows = [[0u8; 256]; 8];

    for rg in 0..row_groups {
        let base_row = rg * 8;

        for col in 0..cols {
            let out_off = (rg * cols + col) * Q5K_PACKED_BLOCK_BYTES;
            let packed = &mut out[out_off..out_off + Q5K_PACKED_BLOCK_BYTES];

            // 8 rows의 unpack + scale/d/dmin 추출
            let mut sc_raw_all = [[0u8; 8]; 8];
            let mut mn_raw_all = [[0u8; 8]; 8];
            let mut d_all = [0.0f32; 8];
            let mut dmin_all = [0.0f32; 8];

            // 임시 버퍼 초기화
            for r in 0..8 {
                unpacked_rows[r] = [0u8; 256];
            }

            for nr in 0..8 {
                let row = base_row + nr;
                if row >= rows {
                    continue;
                }

                let src_off = (row * cols + col) * Q5K_BLOCK_BYTES;
                let block = &src_bytes[src_off..src_off + Q5K_BLOCK_BYTES];

                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                let qh_raw: &[u8; 32] = block[16..48].try_into().unwrap();
                let qs_raw: &[u8; 128] = block[48..176].try_into().unwrap();

                // unsigned 5-bit unpack (0..31)
                unpack_q5k_bits_unsigned(qs_raw, qh_raw, &mut unpacked_rows[nr]);

                // raw scale/min 디코딩
                let (sc, mn) = decode_q4k_scales_raw(scales_12);
                sc_raw_all[nr] = sc;
                mn_raw_all[nr] = mn;
                d_all[nr] = d;
                dmin_all[nr] = dmin;
            }

            // ─── qs 인터리빙: pair-interleaved at 8-byte granularity ───
            // pair p (0..3): even = p*2, odd = p*2+1
            // 32 chunks × 16B per pair = 512B per pair × 4 pairs = 2048B
            for p in 0..4usize {
                let even = p * 2;
                let odd = p * 2 + 1;
                for k in 0..32usize {
                    let dst_off = Q5K_QS_OFF + p * 512 + k * 16;
                    // even row's 8 bytes
                    packed[dst_off..dst_off + 8]
                        .copy_from_slice(&unpacked_rows[even][k * 8..k * 8 + 8]);
                    // odd row's 8 bytes
                    packed[dst_off + 8..dst_off + 16]
                        .copy_from_slice(&unpacked_rows[odd][k * 8..k * 8 + 8]);
                }
            }

            // ─── sc_raw: [8][8] u8 ───
            for nr in 0..8 {
                let off = Q5K_SC_RAW_OFF + nr * 8;
                packed[off..off + 8].copy_from_slice(&sc_raw_all[nr]);
            }

            // ─── mn_raw: [8][8] u8 ───
            for nr in 0..8 {
                let off = Q5K_MN_RAW_OFF + nr * 8;
                packed[off..off + 8].copy_from_slice(&mn_raw_all[nr]);
            }

            // ─── d: [8] f32 ───
            for nr in 0..8 {
                let off = Q5K_D_OFF + nr * 4;
                packed[off..off + 4].copy_from_slice(&d_all[nr].to_le_bytes());
            }

            // ─── dmin: [8] f32 ───
            for nr in 0..8 {
                let off = Q5K_DMIN_OFF + nr * 4;
                packed[off..off + 4].copy_from_slice(&dmin_all[nr].to_le_bytes());
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
    let pair_base = Q5K_QS_OFF + pair * 512;

    for k in 0..32usize {
        let chunk_off = pair_base + k * 16 + is_odd * 8;
        out[k * 8..k * 8 + 8].copy_from_slice(&packed[chunk_off..chunk_off + 8]);
    }

    out
}

/// packed 블록에서 row nr의 sc_raw[8] u8 읽기
pub fn read_packed_sc_raw(packed: &[u8], nr: usize) -> [u8; 8] {
    let base = Q5K_SC_RAW_OFF + nr * 8;
    let mut out = [0u8; 8];
    out.copy_from_slice(&packed[base..base + 8]);
    out
}

/// packed 블록에서 row nr의 mn_raw[8] u8 읽기
pub fn read_packed_mn_raw(packed: &[u8], nr: usize) -> [u8; 8] {
    let base = Q5K_MN_RAW_OFF + nr * 8;
    let mut out = [0u8; 8];
    out.copy_from_slice(&packed[base..base + 8]);
    out
}

/// packed 블록에서 row nr의 d f32 읽기
pub fn read_packed_d(packed: &[u8], nr: usize) -> f32 {
    let base = Q5K_D_OFF + nr * 4;
    f32::from_le_bytes(packed[base..base + 4].try_into().unwrap())
}

/// packed 블록에서 row nr의 dmin f32 읽기
pub fn read_packed_dmin(packed: &[u8], nr: usize) -> f32 {
    let base = Q5K_DMIN_OFF + nr * 4;
    f32::from_le_bytes(packed[base..base + 4].try_into().unwrap())
}

// ─── 테스트 ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// 테스트용 Q5_K 더미 블록 생성 (176 bytes)
    fn make_q5k_block(
        d_val: f32,
        dmin_val: f32,
        scales: [u8; 12],
        qh: [u8; 32],
        qs: [u8; 128],
    ) -> Vec<u8> {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..48].copy_from_slice(&qh);
        block[48..176].copy_from_slice(&qs);
        block
    }

    // ─── 오프셋 상수 확인 ────────────────────────────────────────

    #[test]
    fn test_offset_constants() {
        assert_eq!(Q5K_QS_OFF, 0);
        assert_eq!(Q5K_SC_RAW_OFF, 2048);
        assert_eq!(Q5K_MN_RAW_OFF, 2112);
        assert_eq!(Q5K_D_OFF, 2176);
        assert_eq!(Q5K_DMIN_OFF, 2208);
        assert_eq!(Q5K_PACKED_BLOCK_BYTES, 2240);
    }

    // ─── shape 정확성 ────────────────────────────────────────────

    #[test]
    fn test_pack_shape_exact() {
        let block = make_q5k_block(1.0, 0.5, [1u8; 12], [0u8; 32], [0x77u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q5k(&src, 8, 1);
        assert_eq!(packed.len(), 1 * 1 * Q5K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_shape_multi_col() {
        let block = make_q5k_block(1.0, 0.5, [1u8; 12], [0u8; 32], [0xAAu8; 128]);
        let src: Vec<u8> = block.repeat(8 * 4);
        let packed = pack_q5k(&src, 8, 4);
        assert_eq!(packed.len(), 1 * 4 * Q5K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_shape_multi_rowgroup() {
        let block = make_q5k_block(1.0, 0.5, [0u8; 12], [0u8; 32], [0x55u8; 128]);
        let src: Vec<u8> = block.repeat(16 * 2);
        let packed = pack_q5k(&src, 16, 2);
        assert_eq!(packed.len(), 2 * 2 * Q5K_PACKED_BLOCK_BYTES);
    }

    // ─── 5-bit → unsigned u8 변환 ────────────────────────────────

    #[test]
    fn test_unpack_all_zero() {
        // qs=0, qh=0 → val_unsigned=0
        let qs = [0x00u8; 128];
        let qh = [0x00u8; 32];
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);
        assert!(out.iter().all(|&x| x == 0), "all-zero → should all be 0");
    }

    #[test]
    fn test_unpack_all_ff_qh() {
        // qs=0xFF (low=15, high=15), qh=0xFF (all high bits=1)
        // val = 15|(1<<4) = 31
        let qs = [0xFFu8; 128];
        let qh = [0xFFu8; 32];
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);
        assert!(
            out.iter().all(|&x| x == 31),
            "all-FF qs/qh → should all be 31"
        );
    }

    #[test]
    fn test_unpack_mid_value() {
        // qs=0x88 (low=8, high=8), qh=0x00 → val_unsigned = 8
        let qs = [0x88u8; 128];
        let qh = [0x00u8; 32];
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);
        assert!(out.iter().all(|&x| x == 8), "0x88/0x00 → should all be 8");
    }

    #[test]
    fn test_unpack_high_bit_effect() {
        // qs=0x00, qh[0]=0x01 (bit 0 set) → group 0, l=0: high_bit = 1
        // val = 0 | (1<<4) = 16
        let qs = [0x00u8; 128];
        let mut qh = [0x00u8; 32];
        qh[0] = 0x01;
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);

        assert_eq!(out[0], 16, "group 0, l=0 high_bit set → unsigned 16");
        assert_eq!(out[1], 0, "group 0, l=1 no high_bit → 0");
        assert_eq!(out[64], 0, "group 1, l=0 no high_bit → 0");
    }

    #[test]
    fn test_unpack_group1_high_bit() {
        // group 1, l<32: high_bit = (qh[l] >> (2*1)) & 1 = bit 2
        // qh[0] = 0x04 → bit 2 set → group 1, l=0의 high_bit = 1
        let qs = [0x00u8; 128];
        let mut qh = [0x00u8; 32];
        qh[0] = 0x04;
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);

        assert_eq!(out[64], 16, "group 1, l=0 high_bit set → unsigned 16");
        assert_eq!(out[0], 0, "group 0, l=0 no high_bit → 0");
    }

    #[test]
    fn test_unpack_l_ge32_high_nibble() {
        // l>=32: qs[0] = 0xF0 → high nibble = 15
        // group 0, l=32: low = 15, qh high_bit = 0 → val = 15
        let mut qs = [0x00u8; 128];
        qs[0] = 0xF0;
        let qh = [0x00u8; 32];
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);

        assert_eq!(out[32], 15, "high nibble 15, no high_bit → 15");
        assert_eq!(out[0], 0, "low nibble 0 → 0");
    }

    #[test]
    fn test_unpack_l_ge32_high_bit() {
        // l>=32, group 0: high_bit = (qh[l-32] >> (2*0+1)) & 1 = bit 1
        // qh[0] = 0x02 → bit 1 set → l=32의 high_bit = 1
        // val = 0 | (1<<4) = 16
        let qs = [0x00u8; 128];
        let mut qh = [0x00u8; 32];
        qh[0] = 0x02;
        let mut out = [0u8; 256];
        unpack_q5k_bits_unsigned(&qs, &qh, &mut out);

        assert_eq!(out[32], 16, "l>=32 high_bit → unsigned 16");
    }

    // ─── 나머지 행 패딩 ──────────────────────────────────────────

    #[test]
    fn test_remainder_rows_padding() {
        let block = make_q5k_block(2.0, 1.0, [5u8; 12], [0u8; 32], [0xAAu8; 128]);
        let src: Vec<u8> = block.repeat(5);
        let packed = pack_q5k(&src, 5, 1);

        assert_eq!(packed.len(), Q5K_PACKED_BLOCK_BYTES);

        for nr in 0..5 {
            let d = read_packed_d(&packed, nr);
            assert!((d - 2.0).abs() < 0.01, "row {nr}: d={d} (expected ~2.0)");
        }

        for nr in 5..8 {
            let d = read_packed_d(&packed, nr);
            assert_eq!(d, 0.0, "padding row {nr}: d should be 0.0");
        }
    }

    #[test]
    fn test_remainder_rows_qs_padding() {
        let block = make_q5k_block(1.0, 0.5, [0u8; 12], [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(3);
        let packed = pack_q5k(&src, 3, 1);

        // row 3..7 qs는 0 (패딩)
        for nr in 3..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(qs.iter().all(|&b| b == 0), "padding row {nr} qs must be 0");
        }
    }

    // ─── pack 후 값 무결성 검증 ──────────────────────────────────

    #[test]
    fn test_pack_qs_value_correctness() {
        // qs=0x88 (low=8, high=8), qh=0 → unsigned 8
        let block = make_q5k_block(1.0, 0.0, [0u8; 12], [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q5k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(
                qs.iter().all(|&b| b == 8u8),
                "row {nr}: qs should all be 8 (unsigned)"
            );
        }
    }

    #[test]
    fn test_pack_qs_with_high_bit() {
        // qs=0x88 (low=8, high=8), qh=0xFF → val = 8|(1<<4) = 24
        let block = make_q5k_block(1.0, 0.0, [0u8; 12], [0xFFu8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q5k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(
                qs.iter().all(|&b| b == 24u8),
                "row {nr}: qs should all be 24 (8 + high_bit<<4)"
            );
        }
    }

    #[test]
    fn test_pack_d_value_correctness() {
        let d_val = 0.125f32;
        let block = make_q5k_block(d_val, 0.0, [0u8; 12], [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q5k(&src, 8, 1);

        for nr in 0..8 {
            let d = read_packed_d(&packed, nr);
            assert!((d - d_val).abs() < 1e-4, "row {nr}: d={d} expected {d_val}");
        }
    }

    #[test]
    fn test_pack_scales_mins_raw_correctness() {
        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        scales_raw[4] = 20;
        let block = make_q5k_block(2.0, 3.0, scales_raw, [0u8; 32], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q5k(&src, 8, 1);

        for nr in 0..8 {
            let sc = read_packed_sc_raw(&packed, nr);
            let mn = read_packed_mn_raw(&packed, nr);
            assert_eq!(sc[0], 10, "row {nr} sc_raw[0] expected 10");
            assert_eq!(mn[0], 20, "row {nr} mn_raw[0] expected 20");

            let dmin = read_packed_dmin(&packed, nr);
            assert!(
                (dmin - 3.0).abs() < 0.01,
                "row {nr}: dmin={dmin} expected 3.0"
            );
        }
    }

    // ─── 인터리빙 정확성 확인 ────────────────────────────────────

    #[test]
    fn test_pack_interleaving() {
        // 각 row마다 다른 qs 패턴으로 인터리빙 확인
        let mut src = Vec::new();
        for row in 0..8u8 {
            let nibble = (row + 1) % 16;
            let qs_byte = nibble | (nibble << 4);
            // qh=0 → val = nibble (low) or nibble (high)
            let block = make_q5k_block(1.0, 0.0, [0u8; 12], [0u8; 32], [qs_byte; 128]);
            src.extend_from_slice(&block);
        }
        let packed = pack_q5k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            let expected = ((nr as u8 + 1) % 16) as u8;
            assert!(
                qs.iter().all(|&b| b == expected),
                "row {nr}: qs should all be {expected}, got {:?}",
                &qs[0..8]
            );
        }
    }

    // ─── 다중 col 인터리빙 확인 ──────────────────────────────────

    #[test]
    fn test_pack_multi_col_interleave() {
        let block0 = make_q5k_block(1.0, 0.0, [0u8; 12], [0u8; 32], [0x11u8; 128]);
        let block1 = make_q5k_block(2.0, 0.0, [0u8; 12], [0u8; 32], [0x22u8; 128]);

        let mut src = Vec::new();
        for _ in 0..8 {
            src.extend_from_slice(&block0);
            src.extend_from_slice(&block1);
        }

        let packed = pack_q5k(&src, 8, 2);
        assert_eq!(packed.len(), 1 * 2 * Q5K_PACKED_BLOCK_BYTES);

        let pb0 = &packed[0..Q5K_PACKED_BLOCK_BYTES];
        let pb1 = &packed[Q5K_PACKED_BLOCK_BYTES..2 * Q5K_PACKED_BLOCK_BYTES];

        let d0 = read_packed_d(pb0, 0);
        assert!((d0 - 1.0).abs() < 0.01, "col0 d={d0}");

        let d1 = read_packed_d(pb1, 0);
        assert!((d1 - 2.0).abs() < 0.01, "col1 d={d1}");
    }
}
