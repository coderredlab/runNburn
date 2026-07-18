//! Q4_K → i8mm packed layout 변환
//!
//! # Packed 블록 레이아웃 (NR=8 rows per group, super-block 1개 기준)
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
//! 총 Q4K_PACKED_BLOCK_BYTES = 2240 bytes per 8-row group

use half::f16;

// ─── 오프셋 상수 ─────────────────────────────────────────────────

/// qs 시작 오프셋: 0
/// 4 pairs × 32 chunks × 16B = 2048B
pub const Q4K_QS_OFF: usize = 0;

/// sc_raw 시작 오프셋: 2048
/// [8 rows][8 sub-blocks] u8 = 64B
pub const Q4K_SC_RAW_OFF: usize = 2048;

/// mn_raw 시작 오프셋: 2048 + 64 = 2112
/// [8 rows][8 sub-blocks] u8 = 64B
pub const Q4K_MN_RAW_OFF: usize = Q4K_SC_RAW_OFF + 64;

/// d 시작 오프셋: 2112 + 64 = 2176
/// [8] f32 = 32B
pub const Q4K_D_OFF: usize = Q4K_MN_RAW_OFF + 64;

/// dmin 시작 오프셋: 2176 + 32 = 2208
/// [8] f32 = 32B
pub const Q4K_DMIN_OFF: usize = Q4K_D_OFF + 32;

/// packed 블록 전체 바이트: 2208 + 32 = 2240
pub const Q4K_PACKED_BLOCK_BYTES: usize = Q4K_DMIN_OFF + 32;

/// Compact Q4_K layout keeps GGUF nibbles compressed and stores decoded metadata.
/// Per 8-row group and super-block:
///   qs:   [8][128] raw Q4_K nibble bytes = 1024B
///   sc:   [8][8] raw decoded scales       = 64B
///   mn:   [8][8] raw decoded mins         = 64B
///   d:    [8] f32                         = 32B
///   dmin: [8] f32                         = 32B
pub const Q4K_COMPACT_QS_OFF: usize = 0;
pub const Q4K_COMPACT_SC_RAW_OFF: usize = 1024;
pub const Q4K_COMPACT_MN_RAW_OFF: usize = Q4K_COMPACT_SC_RAW_OFF + 64;
pub const Q4K_COMPACT_D_OFF: usize = Q4K_COMPACT_MN_RAW_OFF + 64;
pub const Q4K_COMPACT_DMIN_OFF: usize = Q4K_COMPACT_D_OFF + 32;
pub const Q4K_COMPACT_BLOCK_BYTES: usize = Q4K_COMPACT_DMIN_OFF + 32;

// ─── Q4_K 블록 크기 ──────────────────────────────────────────────

/// Q4_K 원본 블록 바이트 (d:2 + dmin:2 + scales:12 + qs:128 = 144)
pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_RAW_META_QS_BYTES: usize = 128;
pub const Q4K_RAW_META_META_BYTES: usize = 24;
pub const Q4K_RAW_META_BLOCK_BYTES: usize = Q4K_RAW_META_QS_BYTES + Q4K_RAW_META_META_BYTES;

// ─── 스케일 디코딩 ────────────────────────────────────────────────

/// Q4_K 6-bit packed scales 디코딩 (f32 pre-multiplied).
///
/// `scales_raw[12]` → (sc_f32[8], mn_f32[8])
///
/// 공식 (llama.cpp `get_scale_min_k4`):
/// - j < 4:  sc = scales[j] & 63,           mn = scales[j+4] & 63
/// - j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
///            mn = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
///
/// 최종 f32: sc_f32[i] = d * sc_raw[i], mn_f32[i] = dmin * mn_raw[i]
pub fn decode_q4k_scales(scales_raw: &[u8; 12], d: f32, dmin: f32) -> ([f32; 8], [f32; 8]) {
    let mut sc_out = [0f32; 8];
    let mut mn_out = [0f32; 8];

    for j in 0usize..8 {
        let (sc, mn) = if j < 4 {
            (scales_raw[j] & 63, scales_raw[j + 4] & 63)
        } else {
            let sc = (scales_raw[j + 4] & 0x0F) | ((scales_raw[j - 4] >> 6) << 4);
            let mn = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
            (sc, mn)
        };
        sc_out[j] = d * sc as f32;
        mn_out[j] = dmin * mn as f32;
    }

    (sc_out, mn_out)
}

/// Q4_K 6-bit packed scales 디코딩 (raw u8 값).
///
/// `scales_raw[12]` → (sc_u8[8], mn_u8[8])
///
/// d/dmin를 곱하지 않고 raw 6-bit 값만 반환.
pub fn decode_q4k_scales_raw(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc_out = [0u8; 8];
    let mut mn_out = [0u8; 8];

    for j in 0usize..8 {
        let (sc, mn) = if j < 4 {
            (scales_raw[j] & 63, scales_raw[j + 4] & 63)
        } else {
            let sc = (scales_raw[j + 4] & 0x0F) | ((scales_raw[j - 4] >> 6) << 4);
            let mn = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
            (sc, mn)
        };
        sc_out[j] = sc;
        mn_out[j] = mn;
    }

    (sc_out, mn_out)
}

// ─── nibble → unsigned u8 변환 ───────────────────────────────────

/// Q4_K 블록의 nibble을 256개 unsigned u8로 언팩 (0..15).
///
/// Q4_K qs 레이아웃: 4그룹×64개, 각 그룹에서
///   - qs[0..32]: low nibble → 원소 0..32 (sub-block is+0)
///   - qs[0..32]: high nibble → 원소 32..64 (sub-block is+1)
fn unpack_nibbles_unsigned(qs: &[u8; 128], out: &mut [u8; 256]) {
    let mut q_off = 0usize;
    let mut y_off = 0usize;

    for _ in 0..4 {
        // low nibble: 원소 0..32
        for l in 0..32 {
            out[y_off + l] = qs[q_off + l] & 0x0F;
        }
        // high nibble: 원소 32..64
        for l in 0..32 {
            out[y_off + 32 + l] = qs[q_off + l] >> 4;
        }
        q_off += 32;
        y_off += 64;
    }
}

// ─── 메인 pack 함수 ───────────────────────────────────────────────

/// Q4_K 가중치 → row-pair interleaved packed layout 변환.
///
/// # 인자
/// - `src_bytes`: GGUF 원본 Q4_K 바이트 스트림 (rows × blocks_per_row × 144 bytes)
/// - `rows`: 가중치 행 수 (out_features)
/// - `cols`: 슈퍼블록 수 (cols_in_blocks = in_features / 256)
///
/// # 반환
/// NR=8 행 그룹 단위로 packed된 Vec<u8>
/// 크기: ceil(rows/8) × cols × Q4K_PACKED_BLOCK_BYTES
///
/// # 나머지 행 처리
/// rows % 8 != 0이면 마지막 그룹에서 모자란 행은 0으로 패딩됨.
///
/// # qs 인터리빙 구조
/// pair p (=0..3): even row = p*2, odd row = p*2+1
/// 256 elements = 32 chunks of 8 bytes each
/// chunk k의 16B = [even_row[k*8..k*8+8] | odd_row[k*8..k*8+8]]
/// → smmla용 vld1q_s8로 두 row 동시 로드 가능
pub fn pack_q4k(src_bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let row_groups = rows.div_ceil(8);
    let total_packed_bytes = row_groups * cols * Q4K_PACKED_BLOCK_BYTES;
    let mut out = vec![0u8; total_packed_bytes];

    // 임시 버퍼: 각 row의 unsigned nibble [8][256]
    let mut unpacked_rows = [[0u8; 256]; 8];

    for rg in 0..row_groups {
        let base_row = rg * 8;

        for col in 0..cols {
            let out_off = (rg * cols + col) * Q4K_PACKED_BLOCK_BYTES;
            let packed = &mut out[out_off..out_off + Q4K_PACKED_BLOCK_BYTES];

            // 8 rows의 nibble unpack + scale/d/dmin 추출
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

                let src_off = (row * cols + col) * Q4K_BLOCK_BYTES;
                let block = &src_bytes[src_off..src_off + Q4K_BLOCK_BYTES];

                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                let qs_raw: &[u8; 128] = block[16..144].try_into().unwrap();

                // unsigned nibble unpack
                unpack_nibbles_unsigned(qs_raw, &mut unpacked_rows[nr]);

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
                    let dst_off = Q4K_QS_OFF + p * 512 + k * 16;
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
                let off = Q4K_SC_RAW_OFF + nr * 8;
                packed[off..off + 8].copy_from_slice(&sc_raw_all[nr]);
            }

            // ─── mn_raw: [8][8] u8 ───
            for nr in 0..8 {
                let off = Q4K_MN_RAW_OFF + nr * 8;
                packed[off..off + 8].copy_from_slice(&mn_raw_all[nr]);
            }

            // ─── d: [8] f32 ───
            for nr in 0..8 {
                let off = Q4K_D_OFF + nr * 4;
                packed[off..off + 4].copy_from_slice(&d_all[nr].to_le_bytes());
            }

            // ─── dmin: [8] f32 ───
            for nr in 0..8 {
                let off = Q4K_DMIN_OFF + nr * 4;
                packed[off..off + 4].copy_from_slice(&dmin_all[nr].to_le_bytes());
            }
        }
    }

    out
}

/// Q4_K 가중치 → compact 8-row group layout 변환.
///
/// 이 레이아웃은 qs를 byte-unpack하지 않아 기존 i8mm packed(2240B)보다 작고,
/// scale/min/d/dmin만 미리 풀어 decode에서 GGUF raw보다 메타데이터 처리 비용을 줄인다.
pub fn pack_q4k_compact(src_bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let row_groups = rows.div_ceil(8);
    let total_packed_bytes = row_groups * cols * Q4K_COMPACT_BLOCK_BYTES;
    let mut out = vec![0u8; total_packed_bytes];

    for rg in 0..row_groups {
        let base_row = rg * 8;

        for col in 0..cols {
            let out_off = (rg * cols + col) * Q4K_COMPACT_BLOCK_BYTES;
            let packed = &mut out[out_off..out_off + Q4K_COMPACT_BLOCK_BYTES];

            for nr in 0..8 {
                let row = base_row + nr;
                if row >= rows {
                    continue;
                }

                let src_off = (row * cols + col) * Q4K_BLOCK_BYTES;
                let block = &src_bytes[src_off..src_off + Q4K_BLOCK_BYTES];
                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                let (sc, mn) = decode_q4k_scales_raw(scales_12);

                let qs_off = Q4K_COMPACT_QS_OFF + nr * 128;
                packed[qs_off..qs_off + 128].copy_from_slice(&block[16..144]);

                let sc_off = Q4K_COMPACT_SC_RAW_OFF + nr * 8;
                packed[sc_off..sc_off + 8].copy_from_slice(&sc);

                let mn_off = Q4K_COMPACT_MN_RAW_OFF + nr * 8;
                packed[mn_off..mn_off + 8].copy_from_slice(&mn);

                let d_off = Q4K_COMPACT_D_OFF + nr * 4;
                packed[d_off..d_off + 4].copy_from_slice(&d.to_le_bytes());

                let dmin_off = Q4K_COMPACT_DMIN_OFF + nr * 4;
                packed[dmin_off..dmin_off + 4].copy_from_slice(&dmin.to_le_bytes());
            }
        }
    }

    out
}

/// Q4_K raw row-major bytes plus metadata interleaved per block.
///
/// Layout:
///   rows * cols blocks, each block = Q4_K qs 128B + meta 24B
///   meta = sc[8] + mn[8] + d(f32) + dmin(f32)
pub fn pack_q4k_raw_meta(src_bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let raw_len = rows * cols * Q4K_BLOCK_BYTES;
    assert_eq!(src_bytes.len(), raw_len);
    let mut out = vec![0u8; rows * cols * Q4K_RAW_META_BLOCK_BYTES];

    for row in 0..rows {
        for col in 0..cols {
            let src = (row * cols + col) * Q4K_BLOCK_BYTES;
            let dst = (row * cols + col) * Q4K_RAW_META_BLOCK_BYTES;
            out[dst..dst + Q4K_RAW_META_QS_BYTES]
                .copy_from_slice(&src_bytes[src + 16..src + Q4K_BLOCK_BYTES]);

            let meta = dst + Q4K_RAW_META_QS_BYTES;
            let scales_12: &[u8; 12] = src_bytes[src + 4..src + 16].try_into().unwrap();
            let (sc, mn) = decode_q4k_scales_raw(scales_12);
            out[meta..meta + 8].copy_from_slice(&sc);
            out[meta + 8..meta + 16].copy_from_slice(&mn);

            let d = f16::from_le_bytes([src_bytes[src], src_bytes[src + 1]]).to_f32();
            let dmin = f16::from_le_bytes([src_bytes[src + 2], src_bytes[src + 3]]).to_f32();
            out[meta + 16..meta + 20].copy_from_slice(&d.to_le_bytes());
            out[meta + 20..meta + 24].copy_from_slice(&dmin.to_le_bytes());
        }
    }

    out
}

/// Q4KRawMeta row-major layout → row-pair interleaved packed layout 변환.
///
/// 입력 rawmeta block = `qs[128] + sc[8] + mn[8] + d(f32) + dmin(f32)`.
/// 출력은 `pack_q4k()`와 같은 `Q4K_PACKED_BLOCK_BYTES` 레이아웃이다.
pub fn pack_q4k_from_raw_meta(raw_meta: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let raw_len = rows * cols * Q4K_RAW_META_BLOCK_BYTES;
    assert_eq!(raw_meta.len(), raw_len);

    let row_groups = rows.div_ceil(8);
    let total_packed_bytes = row_groups * cols * Q4K_PACKED_BLOCK_BYTES;
    let mut out = vec![0u8; total_packed_bytes];
    let mut unpacked_rows = [[0u8; 256]; 8];

    for rg in 0..row_groups {
        let base_row = rg * 8;

        for col in 0..cols {
            let out_off = (rg * cols + col) * Q4K_PACKED_BLOCK_BYTES;
            let packed = &mut out[out_off..out_off + Q4K_PACKED_BLOCK_BYTES];

            for row_buf in &mut unpacked_rows {
                *row_buf = [0u8; 256];
            }

            for nr in 0..8usize {
                let row = base_row + nr;
                if row >= rows {
                    continue;
                }

                let src = (row * cols + col) * Q4K_RAW_META_BLOCK_BYTES;
                let qs_raw: &[u8; 128] = raw_meta[src..src + Q4K_RAW_META_QS_BYTES]
                    .try_into()
                    .unwrap();
                unpack_nibbles_unsigned(qs_raw, &mut unpacked_rows[nr]);

                let meta = src + Q4K_RAW_META_QS_BYTES;
                let sc_off = Q4K_SC_RAW_OFF + nr * 8;
                packed[sc_off..sc_off + 8].copy_from_slice(&raw_meta[meta..meta + 8]);

                let mn_off = Q4K_MN_RAW_OFF + nr * 8;
                packed[mn_off..mn_off + 8].copy_from_slice(&raw_meta[meta + 8..meta + 16]);

                let d_off = Q4K_D_OFF + nr * 4;
                packed[d_off..d_off + 4].copy_from_slice(&raw_meta[meta + 16..meta + 20]);

                let dmin_off = Q4K_DMIN_OFF + nr * 4;
                packed[dmin_off..dmin_off + 4].copy_from_slice(&raw_meta[meta + 20..meta + 24]);
            }

            for p in 0..4usize {
                let even = p * 2;
                let odd = p * 2 + 1;
                for k in 0..32usize {
                    let dst_off = Q4K_QS_OFF + p * 512 + k * 16;
                    packed[dst_off..dst_off + 8]
                        .copy_from_slice(&unpacked_rows[even][k * 8..k * 8 + 8]);
                    packed[dst_off + 8..dst_off + 16]
                        .copy_from_slice(&unpacked_rows[odd][k * 8..k * 8 + 8]);
                }
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
    let pair_base = Q4K_QS_OFF + pair * 512;

    for k in 0..32usize {
        let chunk_off = pair_base + k * 16 + is_odd * 8;
        out[k * 8..k * 8 + 8].copy_from_slice(&packed[chunk_off..chunk_off + 8]);
    }

    out
}

/// packed 블록에서 row nr의 sc_raw[8] u8 읽기
pub fn read_packed_sc_raw(packed: &[u8], nr: usize) -> [u8; 8] {
    let base = Q4K_SC_RAW_OFF + nr * 8;
    let mut out = [0u8; 8];
    out.copy_from_slice(&packed[base..base + 8]);
    out
}

/// packed 블록에서 row nr의 mn_raw[8] u8 읽기
pub fn read_packed_mn_raw(packed: &[u8], nr: usize) -> [u8; 8] {
    let base = Q4K_MN_RAW_OFF + nr * 8;
    let mut out = [0u8; 8];
    out.copy_from_slice(&packed[base..base + 8]);
    out
}

/// packed 블록에서 row nr의 d f32 읽기
pub fn read_packed_d(packed: &[u8], nr: usize) -> f32 {
    let base = Q4K_D_OFF + nr * 4;
    f32::from_le_bytes(packed[base..base + 4].try_into().unwrap())
}

/// packed 블록에서 row nr의 dmin f32 읽기
pub fn read_packed_dmin(packed: &[u8], nr: usize) -> f32 {
    let base = Q4K_DMIN_OFF + nr * 4;
    f32::from_le_bytes(packed[base..base + 4].try_into().unwrap())
}

// ─── 테스트 ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// 테스트용 Q4_K 더미 블록 생성 (144 bytes)
    fn make_q4k_block(d_val: f32, dmin_val: f32, scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..144].copy_from_slice(&qs);
        block
    }

    // ─── 오프셋 상수 확인 ────────────────────────────────────────

    #[test]
    fn test_offset_constants() {
        assert_eq!(Q4K_QS_OFF, 0);
        assert_eq!(Q4K_SC_RAW_OFF, 2048);
        assert_eq!(Q4K_MN_RAW_OFF, 2112);
        assert_eq!(Q4K_D_OFF, 2176);
        assert_eq!(Q4K_DMIN_OFF, 2208);
        assert_eq!(Q4K_PACKED_BLOCK_BYTES, 2240);
    }

    #[test]
    fn test_raw_meta_stores_qs_and_meta_per_q4k_block() {
        let block0 = make_q4k_block(
            0.25,
            0.125,
            [1, 2, 3, 4, 5, 6, 7, 8, 0x91, 0xa2, 0xb3, 0xc4],
            [0x12u8; 128],
        );
        let block1 = make_q4k_block(
            0.5,
            0.25,
            [9, 10, 11, 12, 13, 14, 15, 16, 0x55, 0x66, 0x77, 0x88],
            [0x34u8; 128],
        );
        let mut src = Vec::new();
        src.extend_from_slice(&block0);
        src.extend_from_slice(&block1);

        let packed = pack_q4k_raw_meta(&src, 1, 2);
        assert_eq!(packed.len(), 2 * Q4K_RAW_META_BLOCK_BYTES);

        assert_eq!(&packed[0..128], &block0[16..144]);
        assert_eq!(
            &packed[Q4K_RAW_META_BLOCK_BYTES..Q4K_RAW_META_BLOCK_BYTES + 128],
            &block1[16..144]
        );

        let meta0 = &packed[128..Q4K_RAW_META_BLOCK_BYTES];
        let (sc0, mn0) = decode_q4k_scales_raw(block0[4..16].try_into().unwrap());
        assert_eq!(&meta0[0..8], &sc0);
        assert_eq!(&meta0[8..16], &mn0);
        assert_eq!(
            f32::from_le_bytes(meta0[16..20].try_into().unwrap()),
            f16::from_f32(0.25).to_f32()
        );
        assert_eq!(
            f32::from_le_bytes(meta0[20..24].try_into().unwrap()),
            f16::from_f32(0.125).to_f32()
        );
    }

    #[test]
    fn test_compact_and_raw_meta_have_same_len_for_one_rowgroup() {
        let mut src = Vec::new();
        for row in 0..8u8 {
            let block =
                make_q4k_block(0.25 + row as f32, 0.125, [row; 12], [row | (row << 4); 128]);
            src.extend_from_slice(&block);
        }

        let compact = pack_q4k_compact(&src, 8, 1);
        let raw_meta = pack_q4k_raw_meta(&src, 8, 1);

        assert_eq!(compact.len(), Q4K_COMPACT_BLOCK_BYTES);
        assert_eq!(raw_meta.len(), 8 * Q4K_RAW_META_BLOCK_BYTES);
        assert_eq!(compact.len(), raw_meta.len());
    }

    // ─── shape 정확성 ────────────────────────────────────────────

    #[test]
    fn test_pack_shape_exact() {
        let block = make_q4k_block(1.0, 0.5, [1u8; 12], [0x77u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q4k(&src, 8, 1);
        assert_eq!(packed.len(), 1 * 1 * Q4K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_shape_multi_col() {
        let block = make_q4k_block(1.0, 0.5, [1u8; 12], [0xAAu8; 128]);
        let src: Vec<u8> = block.repeat(8 * 4);
        let packed = pack_q4k(&src, 8, 4);
        assert_eq!(packed.len(), 1 * 4 * Q4K_PACKED_BLOCK_BYTES);
    }

    #[test]
    fn test_pack_shape_multi_rowgroup() {
        let block = make_q4k_block(1.0, 0.5, [0u8; 12], [0x55u8; 128]);
        let src: Vec<u8> = block.repeat(16 * 2);
        let packed = pack_q4k(&src, 16, 2);
        assert_eq!(packed.len(), 2 * 2 * Q4K_PACKED_BLOCK_BYTES);
    }

    // ─── nibble → unsigned u8 변환 ───────────────────────────────

    #[test]
    fn test_nibble_to_unsigned() {
        let qs = [0x00u8; 128];
        let mut out = [0u8; 256];
        unpack_nibbles_unsigned(&qs, &mut out);
        assert!(out.iter().all(|&x| x == 0), "0x00 nibble은 0이어야 함");
    }

    #[test]
    fn test_nibble_to_unsigned_ff() {
        let qs = [0xFFu8; 128];
        let mut out = [0u8; 256];
        unpack_nibbles_unsigned(&qs, &mut out);
        assert!(out.iter().all(|&x| x == 15), "0xFF nibble은 15여야 함");
    }

    #[test]
    fn test_nibble_to_unsigned_88() {
        let qs = [0x88u8; 128];
        let mut out = [0u8; 256];
        unpack_nibbles_unsigned(&qs, &mut out);
        assert!(out.iter().all(|&x| x == 8), "0x88 nibble은 8이어야 함");
    }

    #[test]
    fn test_nibble_unsigned_mixed() {
        let mut qs = [0x88u8; 128];
        qs[0] = 0x3C;
        let mut out = [0u8; 256];
        unpack_nibbles_unsigned(&qs, &mut out);
        // 원소 0 = low nibble of qs[0] = 0x0C = 12
        assert_eq!(out[0], 12, "low nibble 12");
        // 원소 32 = high nibble of qs[0] = 0x03 = 3
        assert_eq!(out[32], 3, "high nibble 3");
        // 나머지는 8 (0x88)
        assert_eq!(out[1], 8);
        assert_eq!(out[33], 8);
    }

    // ─── 스케일 디코딩 (f32) ──────────────────────────────────────

    #[test]
    fn test_decode_scales_zero() {
        let scales_raw = [0u8; 12];
        let (sc, mn) = decode_q4k_scales(&scales_raw, 1.0, 1.0);
        assert!(sc.iter().all(|&x| x == 0.0));
        assert!(mn.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_decode_scales_j_lt4() {
        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        scales_raw[4] = 20;
        let (sc, mn) = decode_q4k_scales(&scales_raw, 2.0, 3.0);
        assert_eq!(sc[0], 2.0 * 10.0);
        assert_eq!(mn[0], 3.0 * 20.0);
    }

    #[test]
    fn test_decode_scales_j_ge4() {
        let mut scales_raw = [0u8; 12];
        scales_raw[8] = 0x0F;
        scales_raw[0] = 0x00;
        scales_raw[4] = 0x00;
        let (sc, mn) = decode_q4k_scales(&scales_raw, 1.0, 1.0);
        assert_eq!(sc[4], 15.0);
        assert_eq!(mn[4], 0.0);
    }

    #[test]
    fn test_decode_scales_roundtrip() {
        let mut scales_raw = [0u8; 12];
        scales_raw[2] = 63;
        scales_raw[10] = 0xF;
        let (sc, _) = decode_q4k_scales(&scales_raw, 1.0, 1.0);
        assert_eq!(sc[2], 63.0);
        assert_eq!(sc[6], 15.0);
    }

    // ─── 스케일 디코딩 (raw u8) ───────────────────────────────────

    #[test]
    fn test_decode_scales_raw_zero() {
        let scales_raw = [0u8; 12];
        let (sc, mn) = decode_q4k_scales_raw(&scales_raw);
        assert!(sc.iter().all(|&x| x == 0));
        assert!(mn.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_decode_scales_raw_values() {
        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        scales_raw[4] = 20;
        let (sc, mn) = decode_q4k_scales_raw(&scales_raw);
        assert_eq!(sc[0], 10);
        assert_eq!(mn[0], 20);
    }

    // ─── 나머지 행 패딩 ──────────────────────────────────────────

    #[test]
    fn test_remainder_rows_padding() {
        let block = make_q4k_block(2.0, 1.0, [5u8; 12], [0xAAu8; 128]);
        let src: Vec<u8> = block.repeat(5);
        let packed = pack_q4k(&src, 5, 1);

        assert_eq!(packed.len(), Q4K_PACKED_BLOCK_BYTES);

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
    fn test_remainder_rows_qs_padding() {
        let block = make_q4k_block(1.0, 0.5, [0u8; 12], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(3);
        let packed = pack_q4k(&src, 3, 1);

        // row 3..7 qs는 0 (패딩)
        for nr in 3..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(qs.iter().all(|&b| b == 0), "padding row {nr} qs must be 0");
        }
    }

    // ─── pack 후 값 무결성 검증 ──────────────────────────────────

    #[test]
    fn test_pack_qs_value_correctness() {
        // qs 전부 0x88 → unsigned 8
        let block = make_q4k_block(1.0, 0.0, [0u8; 12], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q4k(&src, 8, 1);

        for nr in 0..8 {
            let qs = read_packed_qs(&packed, nr);
            assert!(
                qs.iter().all(|&b| b == 8u8),
                "row {nr}: qs should all be 8 (unsigned)"
            );
        }
    }

    #[test]
    fn test_pack_d_value_correctness() {
        let d_val = 0.125f32;
        let block = make_q4k_block(d_val, 0.0, [0u8; 12], [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q4k(&src, 8, 1);

        for nr in 0..8 {
            let d = read_packed_d(&packed, nr);
            assert!((d - d_val).abs() < 1e-4, "row {nr}: d={d} expected {d_val}");
        }
    }

    #[test]
    fn test_pack_scales_mins_raw_correctness() {
        // scales[0]=10, scales[4]=20
        let mut scales_raw = [0u8; 12];
        scales_raw[0] = 10;
        scales_raw[4] = 20;
        let block = make_q4k_block(2.0, 3.0, scales_raw, [0x88u8; 128]);
        let src: Vec<u8> = block.repeat(8);
        let packed = pack_q4k(&src, 8, 1);

        for nr in 0..8 {
            let sc = read_packed_sc_raw(&packed, nr);
            let mn = read_packed_mn_raw(&packed, nr);
            // j=0: sc_raw = 10, mn_raw = 20
            assert_eq!(sc[0], 10, "row {nr} sc_raw[0] expected 10");
            assert_eq!(mn[0], 20, "row {nr} mn_raw[0] expected 20");

            // dmin 확인
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
        // row 0: qs low nibble = 1, row 1: qs low nibble = 2, ...
        let mut src = Vec::new();
        for row in 0..8u8 {
            let nibble = (row + 1) % 16;
            let qs_byte = nibble | (nibble << 4); // low=nibble, high=nibble
            let block = make_q4k_block(1.0, 0.0, [0u8; 12], [qs_byte; 128]);
            src.extend_from_slice(&block);
        }
        let packed = pack_q4k(&src, 8, 1);

        // read_packed_qs로 deinterleave 후 값 확인
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
        let block0 = make_q4k_block(1.0, 0.0, [0u8; 12], [0x11u8; 128]);
        let block1 = make_q4k_block(2.0, 0.0, [0u8; 12], [0x22u8; 128]);

        let mut src = Vec::new();
        for _ in 0..8 {
            src.extend_from_slice(&block0);
            src.extend_from_slice(&block1);
        }

        let packed = pack_q4k(&src, 8, 2);
        assert_eq!(packed.len(), 1 * 2 * Q4K_PACKED_BLOCK_BYTES);

        let pb0 = &packed[0..Q4K_PACKED_BLOCK_BYTES];
        let pb1 = &packed[Q4K_PACKED_BLOCK_BYTES..2 * Q4K_PACKED_BLOCK_BYTES];

        let d0 = read_packed_d(pb0, 0);
        assert!((d0 - 1.0).abs() < 0.01, "col0 d={d0}");

        let d1 = read_packed_d(pb1, 0);
        assert!((d1 - 2.0).abs() < 0.01, "col1 d={d1}");
    }
}
