/// Q4_K weight repacking for 8-row simultaneous NEON dot products.
///
/// Q4_K block structure (144 bytes, 256 elements):
///   offset 0-1:   d (f16)
///   offset 2-3:   dmin (f16)
///   offset 4-15:  scales (12 bytes, 6-bit packed → 8 sc + 8 mn)
///   offset 16-143: qs (128 bytes, 4-bit nibbles)
///
/// Repacked layout per 8-row group, per block (1216 bytes):
///   qs[0..7]:   1024 bytes — row-contiguous: row r's 128 qs bytes at r*128
///     Each row has 4 groups × 32 bytes = 128 bytes contiguous
///   sc[0..7]:     64 bytes — row-major: sc[r][j] = sc[r*8+j]
///   mn[0..7]:     64 bytes — row-major: mn[r][j] = mn[r*8+j]
///   d[0..7]:      32 bytes — d[r] as f32
///   dmin[0..7]:   32 bytes — dmin[r] as f32

const Q4K_BLOCK_BYTES: usize = 144;
pub const REPACKED_BLOCK_BYTES: usize = 1024 + 64 + 64 + 32 + 32; // 1216
pub const TWIN_Q4K_BLOCK_BYTES: usize = 512 + 16 + 16 + 8 + 8;

pub const RPK_QS_OFF: usize = 0;
pub const RPK_SC_OFF: usize = 1024;
pub const RPK_MN_OFF: usize = 1088;
pub const RPK_D_OFF: usize = 1152;
pub const RPK_DMIN_OFF: usize = 1184;
pub const TPK_QS_OFF: usize = 0;
pub const TPK_SC_OFF: usize = 512;
pub const TPK_MN_OFF: usize = 528;
pub const TPK_D_OFF: usize = 544;
pub const TPK_DMIN_OFF: usize = 552;
pub const META_Q4K_BLOCK_BYTES: usize = 8 + 8 + 4 + 4;
pub const MPK_SC_OFF: usize = 0;
pub const MPK_MN_OFF: usize = 8;
pub const MPK_D_OFF: usize = 16;
pub const MPK_DMIN_OFF: usize = 20;

pub fn repacked_groups(rows: usize) -> usize {
    rows / 8
}
pub fn repacked_remainder(rows: usize) -> usize {
    rows % 8
}

pub fn twin_packed_pairs(rows: usize) -> usize {
    rows / 2
}

/// Extract Q4_K 6-bit packed scales → sc[8] + mn[8]
fn extract_scales(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
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

/// Repack Q4_K weights for 8-row simultaneous NEON dot products.
///
/// qs: row-contiguous — row r's 128 bytes at offset r*128
/// sc/mn: row-major — sc[r][j] = sc[r*8+j]
pub fn repack_q4k(bytes: &[u8], rows: usize, cols: usize) -> memmap2::Mmap {
    assert!(cols % 256 == 0);
    let n_blocks = cols / 256;
    let groups = repacked_groups(rows);
    let bytes_per_row = n_blocks * Q4K_BLOCK_BYTES;
    let total_bytes = groups * n_blocks * REPACKED_BLOCK_BYTES;

    let tmp_file = tempfile::tempfile().expect("failed to create tempfile for repack");
    tmp_file
        .set_len(total_bytes as u64)
        .expect("failed to set tempfile size");
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&tmp_file).expect("mmap failed") };
    let out = &mut mmap[..];

    for g in 0..groups {
        for bi in 0..n_blocks {
            let dst = (g * n_blocks + bi) * REPACKED_BLOCK_BYTES;

            // [1] qs: row-contiguous — row r gets 128 bytes at r*128
            for r in 0..8 {
                let row = g * 8 + r;
                let src = row * bytes_per_row + bi * Q4K_BLOCK_BYTES + 16;
                let dst_off = dst + RPK_QS_OFF + r * 128;
                out[dst_off..dst_off + 128].copy_from_slice(&bytes[src..src + 128]);
            }

            // [2] sc row-major: sc[r*8+j]
            for r in 0..8 {
                let row = g * 8 + r;
                let boff = row * bytes_per_row + bi * Q4K_BLOCK_BYTES;
                let (sc, mn) = extract_scales(&bytes[boff + 4..boff + 16]);
                for j in 0..8 {
                    out[dst + RPK_SC_OFF + r * 8 + j] = sc[j];
                }
                // [3] mn row-major: mn[r*8+j]
                for j in 0..8 {
                    out[dst + RPK_MN_OFF + r * 8 + j] = mn[j];
                }
            }

            // [4] d as f32
            for r in 0..8 {
                let row = g * 8 + r;
                let boff = row * bytes_per_row + bi * Q4K_BLOCK_BYTES;
                let d = half::f16::from_bits(u16::from_le_bytes([bytes[boff], bytes[boff + 1]]))
                    .to_f32();
                out[dst + RPK_D_OFF + r * 4..dst + RPK_D_OFF + r * 4 + 4]
                    .copy_from_slice(&d.to_le_bytes());
            }

            // [5] dmin as f32
            for r in 0..8 {
                let row = g * 8 + r;
                let boff = row * bytes_per_row + bi * Q4K_BLOCK_BYTES;
                let dmin =
                    half::f16::from_bits(u16::from_le_bytes([bytes[boff + 2], bytes[boff + 3]]))
                        .to_f32();
                out[dst + RPK_DMIN_OFF + r * 4..dst + RPK_DMIN_OFF + r * 4 + 4]
                    .copy_from_slice(&dmin.to_le_bytes());
            }
        }
    }

    mmap.make_read_only()
        .expect("failed to make repack mmap read-only")
}

pub fn repack_q4k_twin(bytes: &[u8], rows: usize, cols: usize) -> memmap2::Mmap {
    assert!(cols % 256 == 0);
    let n_blocks = cols / 256;
    let pairs = twin_packed_pairs(rows);
    let bytes_per_row = n_blocks * Q4K_BLOCK_BYTES;
    let total_bytes = pairs * n_blocks * TWIN_Q4K_BLOCK_BYTES;

    let tmp_file = tempfile::tempfile().expect("failed to create tempfile for twin repack");
    tmp_file
        .set_len(total_bytes as u64)
        .expect("failed to set twin repack tempfile size");
    let mut mmap =
        unsafe { memmap2::MmapMut::map_mut(&tmp_file).expect("twin repack mmap failed") };
    let out = &mut mmap[..];

    for p in 0..pairs {
        let row0 = p * 2;
        let row1 = row0 + 1;
        for bi in 0..n_blocks {
            let dst = (p * n_blocks + bi) * TWIN_Q4K_BLOCK_BYTES;
            let boff0 = row0 * bytes_per_row + bi * Q4K_BLOCK_BYTES;
            let boff1 = row1 * bytes_per_row + bi * Q4K_BLOCK_BYTES;
            let qs0 = &bytes[boff0 + 16..boff0 + 144];
            let qs1 = &bytes[boff1 + 16..boff1 + 144];

            for group in 0..4 {
                let q_off = group * 32;
                let gdst = dst + TPK_QS_OFF + group * 128;
                for part in 0..2 {
                    let base = q_off + part * 16;
                    let lo_dst = gdst + (part * 2) * 32;
                    let hi_dst = gdst + (part * 2 + 1) * 32;
                    let q0 = &qs0[base..base + 16];
                    let q1 = &qs1[base..base + 16];
                    for i in 0..8 {
                        out[lo_dst + i] = q0[i] & 0x0F;
                        out[lo_dst + 8 + i] = q1[i] & 0x0F;
                        out[lo_dst + 16 + i] = q0[8 + i] & 0x0F;
                        out[lo_dst + 24 + i] = q1[8 + i] & 0x0F;
                        out[hi_dst + i] = q0[i] >> 4;
                        out[hi_dst + 8 + i] = q1[i] >> 4;
                        out[hi_dst + 16 + i] = q0[8 + i] >> 4;
                        out[hi_dst + 24 + i] = q1[8 + i] >> 4;
                    }
                }
            }

            let (sc0, mn0) = extract_scales(&bytes[boff0 + 4..boff0 + 16]);
            let (sc1, mn1) = extract_scales(&bytes[boff1 + 4..boff1 + 16]);
            out[dst + TPK_SC_OFF..dst + TPK_SC_OFF + 8].copy_from_slice(&sc0);
            out[dst + TPK_SC_OFF + 8..dst + TPK_SC_OFF + 16].copy_from_slice(&sc1);
            out[dst + TPK_MN_OFF..dst + TPK_MN_OFF + 8].copy_from_slice(&mn0);
            out[dst + TPK_MN_OFF + 8..dst + TPK_MN_OFF + 16].copy_from_slice(&mn1);

            let d0 =
                half::f16::from_bits(u16::from_le_bytes([bytes[boff0], bytes[boff0 + 1]])).to_f32();
            let d1 =
                half::f16::from_bits(u16::from_le_bytes([bytes[boff1], bytes[boff1 + 1]])).to_f32();
            let dmin0 =
                half::f16::from_bits(u16::from_le_bytes([bytes[boff0 + 2], bytes[boff0 + 3]]))
                    .to_f32();
            let dmin1 =
                half::f16::from_bits(u16::from_le_bytes([bytes[boff1 + 2], bytes[boff1 + 3]]))
                    .to_f32();
            out[dst + TPK_D_OFF..dst + TPK_D_OFF + 4].copy_from_slice(&d0.to_le_bytes());
            out[dst + TPK_D_OFF + 4..dst + TPK_D_OFF + 8].copy_from_slice(&d1.to_le_bytes());
            out[dst + TPK_DMIN_OFF..dst + TPK_DMIN_OFF + 4].copy_from_slice(&dmin0.to_le_bytes());
            out[dst + TPK_DMIN_OFF + 4..dst + TPK_DMIN_OFF + 8]
                .copy_from_slice(&dmin1.to_le_bytes());
        }
    }

    mmap.make_read_only()
        .expect("failed to make twin repack mmap read-only")
}

pub fn repack_q4k_meta(bytes: &[u8], rows: usize, cols: usize) -> memmap2::Mmap {
    assert!(cols % 256 == 0);
    let n_blocks = cols / 256;
    let bytes_per_row = n_blocks * Q4K_BLOCK_BYTES;
    let total_bytes = rows * n_blocks * META_Q4K_BLOCK_BYTES;

    let tmp_file = tempfile::tempfile().expect("failed to create tempfile for meta repack");
    tmp_file
        .set_len(total_bytes as u64)
        .expect("failed to set meta repack tempfile size");
    let mut mmap =
        unsafe { memmap2::MmapMut::map_mut(&tmp_file).expect("meta repack mmap failed") };
    let out = &mut mmap[..];

    for row in 0..rows {
        for bi in 0..n_blocks {
            let src = row * bytes_per_row + bi * Q4K_BLOCK_BYTES;
            let dst = (row * n_blocks + bi) * META_Q4K_BLOCK_BYTES;
            let (sc, mn) = extract_scales(&bytes[src + 4..src + 16]);
            out[dst + MPK_SC_OFF..dst + MPK_SC_OFF + 8].copy_from_slice(&sc);
            out[dst + MPK_MN_OFF..dst + MPK_MN_OFF + 8].copy_from_slice(&mn);
            let d = half::f16::from_bits(u16::from_le_bytes([bytes[src], bytes[src + 1]])).to_f32();
            let dmin =
                half::f16::from_bits(u16::from_le_bytes([bytes[src + 2], bytes[src + 3]])).to_f32();
            out[dst + MPK_D_OFF..dst + MPK_D_OFF + 4].copy_from_slice(&d.to_le_bytes());
            out[dst + MPK_DMIN_OFF..dst + MPK_DMIN_OFF + 4].copy_from_slice(&dmin.to_le_bytes());
        }
    }

    mmap.make_read_only()
        .expect("failed to make meta repack mmap read-only")
}

pub struct Q4KRepackArtifacts {
    pub repacked: memmap2::Mmap,
    pub twin_repacked: memmap2::Mmap,
    pub meta_repacked: memmap2::Mmap,
}

pub fn repack_q4k_artifacts(bytes: &[u8], rows: usize, cols: usize) -> Q4KRepackArtifacts {
    Q4KRepackArtifacts {
        repacked: repack_q4k(bytes, rows, cols),
        twin_repacked: repack_q4k_twin(bytes, rows, cols),
        meta_repacked: repack_q4k_meta(bytes, rows, cols),
    }
}

#[cfg(test)]
fn repacked_qs_byte(repacked: &[u8], row: usize, group: usize, byte_pos: usize) -> u8 {
    repacked[RPK_QS_OFF + row * 128 + group * 32 + byte_pos]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repack_q4k_interleaved() {
        let rows = 8;
        let cols = 256;
        let mut input = vec![0u8; rows * Q4K_BLOCK_BYTES];

        for r in 0..8 {
            let off = r * Q4K_BLOCK_BYTES;
            let d = half::f16::from_f32(0.1 * (r as f32 + 1.0));
            input[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            let dmin = half::f16::from_f32(0.05 * (r as f32 + 1.0));
            input[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for s in 0..12 {
                input[off + 4 + s] = (r * 3 + s) as u8;
            }
            for q in 0..128 {
                input[off + 16 + q] = (r * 7 + q) as u8;
            }
        }

        let repacked = repack_q4k(&input, rows, cols);
        assert_eq!(repacked.len(), REPACKED_BLOCK_BYTES);

        // Verify qs interleaving: repacked_qs_byte should match original
        for r in 0..8 {
            for group in 0..4 {
                for byte_pos in 0..32 {
                    let original_byte = (r * 7 + group * 32 + byte_pos) as u8;
                    let got = repacked_qs_byte(&repacked, r, group, byte_pos);
                    assert_eq!(
                        got, original_byte,
                        "qs mismatch r={r} group={group} byte={byte_pos}"
                    );
                }
            }
        }

        // Verify sc row-major: sc[r][j] at offset RPK_SC_OFF + r*8 + j
        // Original sc for each row is extract_scales(row's scale bytes)
        for r in 0..8 {
            let boff = r * Q4K_BLOCK_BYTES;
            let (sc, _) = extract_scales(&input[boff + 4..boff + 16]);
            for j in 0..8 {
                assert_eq!(
                    repacked[RPK_SC_OFF + r * 8 + j],
                    sc[j],
                    "sc row-major mismatch r={r} j={j}"
                );
            }
        }

        // Verify d as f32
        for r in 0..8 {
            let d_bytes = &repacked[RPK_D_OFF + r * 4..RPK_D_OFF + (r + 1) * 4];
            let d = f32::from_le_bytes([d_bytes[0], d_bytes[1], d_bytes[2], d_bytes[3]]);
            let expected = half::f16::from_f32(0.1 * (r as f32 + 1.0)).to_f32();
            assert!((d - expected).abs() < 1e-6, "d[{r}]: {d} != {expected}");
        }
    }

    #[test]
    fn test_repack_remainder() {
        assert_eq!(repacked_groups(24), 3);
        assert_eq!(repacked_remainder(24), 0);
        assert_eq!(repacked_groups(25), 3);
        assert_eq!(repacked_remainder(25), 1);
    }

    /// Scalar reference: Q4_K dot product using ORIGINAL byte layout.
    fn dot_q4k_original_scalar(
        row_bytes: &[u8],
        q8k_d: f32,
        q8k_qs: &[i8; 256],
        q8k_bsums: &[i16; 8],
    ) -> f32 {
        let d = half::f16::from_bits(u16::from_le_bytes([row_bytes[0], row_bytes[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([row_bytes[2], row_bytes[3]])).to_f32();
        let scales_bytes = &row_bytes[4..16];
        let qs = &row_bytes[16..144];

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
                isum0 += lo as i32 * q8k_qs[x_off + l] as i32;
                isum1 += hi as i32 * q8k_qs[x_off + 32 + l] as i32;
            }
            sumi += sc[is] as i32 * isum0 + sc[is + 1] as i32 * isum1;
            summ += mn[is] as i32 * q8k_bsums[group * 2] as i32
                + mn[is + 1] as i32 * q8k_bsums[group * 2 + 1] as i32;
        }
        d * q8k_d * sumi as f32 - dmin * q8k_d * summ as f32
    }

    /// Scalar reference: Q4_K dot using REPACKED interleaved layout.
    fn dot_q4k_repacked_scalar(
        repacked: &[u8],
        row: usize,
        q8k_d: f32,
        q8k_qs: &[i8; 256],
        q8k_bsums: &[i16; 8],
    ) -> f32 {
        let r = row;
        let d = f32::from_le_bytes([
            repacked[RPK_D_OFF + r * 4],
            repacked[RPK_D_OFF + r * 4 + 1],
            repacked[RPK_D_OFF + r * 4 + 2],
            repacked[RPK_D_OFF + r * 4 + 3],
        ]);
        let dmin = f32::from_le_bytes([
            repacked[RPK_DMIN_OFF + r * 4],
            repacked[RPK_DMIN_OFF + r * 4 + 1],
            repacked[RPK_DMIN_OFF + r * 4 + 2],
            repacked[RPK_DMIN_OFF + r * 4 + 3],
        ]);

        let mut sumi = 0i32;
        let mut summ = 0i32;
        for group in 0..4 {
            let is = group * 2;
            let x_off = group * 64;

            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            for l in 0..32 {
                let byte = repacked_qs_byte(repacked, r, group, l);
                let lo = (byte & 0x0F) as i8;
                let hi = (byte >> 4) as i8;
                isum0 += lo as i32 * q8k_qs[x_off + l] as i32;
                isum1 += hi as i32 * q8k_qs[x_off + 32 + l] as i32;
            }

            // Row-major sc: sc[r][is] at RPK_SC_OFF + r*8 + is
            let sc0 = repacked[RPK_SC_OFF + r * 8 + is] as i32;
            let sc1 = repacked[RPK_SC_OFF + r * 8 + is + 1] as i32;
            let mn0 = repacked[RPK_MN_OFF + r * 8 + is] as i32;
            let mn1 = repacked[RPK_MN_OFF + r * 8 + is + 1] as i32;

            sumi += sc0 * isum0 + sc1 * isum1;
            summ += mn0 * q8k_bsums[group * 2] as i32 + mn1 * q8k_bsums[group * 2 + 1] as i32;
        }
        d * q8k_d * sumi as f32 - dmin * q8k_d * summ as f32
    }

    fn dot_q4k_twin_scalar(
        repacked: &[u8],
        row_in_pair: usize,
        q8k_d: f32,
        q8k_qs: &[i8; 256],
        q8k_bsums: &[i16; 8],
    ) -> f32 {
        let sc_base = TPK_SC_OFF + row_in_pair * 8;
        let mn_base = TPK_MN_OFF + row_in_pair * 8;
        let d_off = TPK_D_OFF + row_in_pair * 4;
        let dmin_off = TPK_DMIN_OFF + row_in_pair * 4;
        let d = f32::from_le_bytes([
            repacked[d_off],
            repacked[d_off + 1],
            repacked[d_off + 2],
            repacked[d_off + 3],
        ]);
        let dmin = f32::from_le_bytes([
            repacked[dmin_off],
            repacked[dmin_off + 1],
            repacked[dmin_off + 2],
            repacked[dmin_off + 3],
        ]);

        let mut sumi = 0i32;
        let mut summ = 0i32;
        for group in 0..4 {
            let is = group * 2;
            let x_off = group * 64;
            let group_off = TPK_QS_OFF + group * 128;
            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            for part in 0..2 {
                let lo = &repacked[group_off + part * 64..group_off + part * 64 + 32];
                let hi = &repacked[group_off + part * 64 + 32..group_off + part * 64 + 64];
                for i in 0..8 {
                    let base = part * 16 + i;
                    let q_lo = if row_in_pair == 0 { lo[i] } else { lo[8 + i] } as i32;
                    let q_hi = if row_in_pair == 0 { hi[i] } else { hi[8 + i] } as i32;
                    isum0 += q_lo * q8k_qs[x_off + base] as i32;
                    isum1 += q_hi * q8k_qs[x_off + 32 + base] as i32;
                    let q_lo_hi = if row_in_pair == 0 {
                        lo[16 + i]
                    } else {
                        lo[24 + i]
                    } as i32;
                    let q_hi_hi = if row_in_pair == 0 {
                        hi[16 + i]
                    } else {
                        hi[24 + i]
                    } as i32;
                    isum0 += q_lo_hi * q8k_qs[x_off + base + 8] as i32;
                    isum1 += q_hi_hi * q8k_qs[x_off + 32 + base + 8] as i32;
                }
            }
            let sc0 = repacked[sc_base + is] as i32;
            let sc1 = repacked[sc_base + is + 1] as i32;
            let mn0 = repacked[mn_base + is] as i32;
            let mn1 = repacked[mn_base + is + 1] as i32;
            sumi += sc0 * isum0 + sc1 * isum1;
            summ += mn0 * q8k_bsums[group * 2] as i32 + mn1 * q8k_bsums[group * 2 + 1] as i32;
        }
        d * q8k_d * sumi as f32 - dmin * q8k_d * summ as f32
    }

    fn dot_q4k_meta_scalar(
        meta: &[u8],
        row_bytes: &[u8],
        q8k_d: f32,
        q8k_qs: &[i8; 256],
        q8k_bsums: &[i16; 8],
    ) -> f32 {
        let d = f32::from_le_bytes([
            meta[MPK_D_OFF],
            meta[MPK_D_OFF + 1],
            meta[MPK_D_OFF + 2],
            meta[MPK_D_OFF + 3],
        ]);
        let dmin = f32::from_le_bytes([
            meta[MPK_DMIN_OFF],
            meta[MPK_DMIN_OFF + 1],
            meta[MPK_DMIN_OFF + 2],
            meta[MPK_DMIN_OFF + 3],
        ]);
        let qs = &row_bytes[16..144];
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
                isum0 += lo as i32 * q8k_qs[x_off + l] as i32;
                isum1 += hi as i32 * q8k_qs[x_off + 32 + l] as i32;
            }
            let sc0 = meta[MPK_SC_OFF + is] as i32;
            let sc1 = meta[MPK_SC_OFF + is + 1] as i32;
            let mn0 = meta[MPK_MN_OFF + is] as i32;
            let mn1 = meta[MPK_MN_OFF + is + 1] as i32;
            sumi += sc0 * isum0 + sc1 * isum1;
            summ += mn0 * q8k_bsums[group * 2] as i32 + mn1 * q8k_bsums[group * 2 + 1] as i32;
        }
        d * q8k_d * sumi as f32 - dmin * q8k_d * summ as f32
    }

    #[test]
    fn test_repacked_dot_matches_original() {
        let rows = 8;
        let cols = 256;
        let mut original = vec![0u8; rows * Q4K_BLOCK_BYTES];

        for r in 0..rows {
            let off = r * Q4K_BLOCK_BYTES;
            let d = half::f16::from_f32(0.02 * (r as f32 + 1.0));
            original[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            let dmin = half::f16::from_f32(0.01 * (r as f32 + 1.0));
            original[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for s in 0..12 {
                original[off + 4 + s] = ((r * 17 + s * 7 + 3) % 64) as u8;
            }
            for q in 0..128 {
                original[off + 16 + q] = ((r * 13 + q * 11 + 5) % 256) as u8;
            }
        }

        let q8k_d = 0.05f32;
        let mut q8k_qs = [0i8; 256];
        let mut q8k_bsums = [0i16; 8];
        for i in 0..256 {
            q8k_qs[i] = ((i as i32 * 7 + 3) % 255 - 127) as i8;
            q8k_bsums[i / 32] += q8k_qs[i] as i16;
        }

        let repacked = repack_q4k(&original, rows, cols);

        for r in 0..rows {
            let row_bytes = &original[r * Q4K_BLOCK_BYTES..(r + 1) * Q4K_BLOCK_BYTES];
            let expected = dot_q4k_original_scalar(row_bytes, q8k_d, &q8k_qs, &q8k_bsums);
            let actual = dot_q4k_repacked_scalar(&repacked, r, q8k_d, &q8k_qs, &q8k_bsums);
            assert!(
                (expected - actual).abs() < 1e-6,
                "row {r}: original={expected}, repacked={actual}, diff={}",
                (expected - actual).abs()
            );
        }
    }

    #[test]
    fn test_twin_repacked_dot_matches_original() {
        let rows = 2;
        let cols = 256;
        let mut original = vec![0u8; rows * Q4K_BLOCK_BYTES];

        for r in 0..rows {
            let off = r * Q4K_BLOCK_BYTES;
            let d = half::f16::from_f32(0.02 * (r as f32 + 1.0));
            original[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            let dmin = half::f16::from_f32(0.01 * (r as f32 + 1.0));
            original[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for s in 0..12 {
                original[off + 4 + s] = ((r * 17 + s * 7 + 3) % 64) as u8;
            }
            for q in 0..128 {
                original[off + 16 + q] = ((r * 13 + q * 11 + 5) % 256) as u8;
            }
        }

        let q8k_d = 0.05f32;
        let mut q8k_qs = [0i8; 256];
        let mut q8k_bsums = [0i16; 8];
        for i in 0..256 {
            q8k_qs[i] = ((i as i32 * 7 + 3) % 255 - 127) as i8;
            q8k_bsums[i / 32] += q8k_qs[i] as i16;
        }

        let twin = repack_q4k_twin(&original, rows, cols);

        for r in 0..rows {
            let row_bytes = &original[r * Q4K_BLOCK_BYTES..(r + 1) * Q4K_BLOCK_BYTES];
            let expected = dot_q4k_original_scalar(row_bytes, q8k_d, &q8k_qs, &q8k_bsums);
            let actual = dot_q4k_twin_scalar(&twin, r, q8k_d, &q8k_qs, &q8k_bsums);
            assert!(
                (expected - actual).abs() < 1e-6,
                "row {r}: original={expected}, twin={actual}, diff={}",
                (expected - actual).abs()
            );
        }
    }

    #[test]
    fn test_meta_repacked_dot_matches_original() {
        let rows = 2;
        let cols = 256;
        let mut original = vec![0u8; rows * Q4K_BLOCK_BYTES];

        for r in 0..rows {
            let off = r * Q4K_BLOCK_BYTES;
            let d = half::f16::from_f32(0.02 * (r as f32 + 1.0));
            original[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            let dmin = half::f16::from_f32(0.01 * (r as f32 + 1.0));
            original[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for s in 0..12 {
                original[off + 4 + s] = ((r * 17 + s * 7 + 3) % 64) as u8;
            }
            for q in 0..128 {
                original[off + 16 + q] = ((r * 13 + q * 11 + 5) % 256) as u8;
            }
        }

        let q8k_d = 0.05f32;
        let mut q8k_qs = [0i8; 256];
        let mut q8k_bsums = [0i16; 8];
        for i in 0..256 {
            q8k_qs[i] = ((i as i32 * 7 + 3) % 255 - 127) as i8;
            q8k_bsums[i / 32] += q8k_qs[i] as i16;
        }

        let meta = repack_q4k_meta(&original, rows, cols);

        for r in 0..rows {
            let row_bytes = &original[r * Q4K_BLOCK_BYTES..(r + 1) * Q4K_BLOCK_BYTES];
            let expected = dot_q4k_original_scalar(row_bytes, q8k_d, &q8k_qs, &q8k_bsums);
            let actual = dot_q4k_meta_scalar(
                &meta[r * META_Q4K_BLOCK_BYTES..(r + 1) * META_Q4K_BLOCK_BYTES],
                row_bytes,
                q8k_d,
                &q8k_qs,
                &q8k_bsums,
            );
            assert!(
                (expected - actual).abs() < 1e-6,
                "row {r}: original={expected}, meta={actual}, diff={}",
                (expected - actual).abs()
            );
        }
    }
}
