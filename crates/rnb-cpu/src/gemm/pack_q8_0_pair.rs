pub fn pack_q8_0_row_pairs(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_ELEMS: usize = 32;
    const PAIR_BLOCK_BYTES: usize = 68;

    assert!(
        cols % Q8_0_BLOCK_ELEMS == 0,
        "Q8_0 pair pack requires cols divisible by {Q8_0_BLOCK_ELEMS}"
    );
    let bytes_per_row = (cols / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    assert_eq!(
        bytes.len(),
        rows * bytes_per_row,
        "unexpected Q8_0 source byte length"
    );

    let row_pairs = rows.div_ceil(2);
    let n_blocks = bytes_per_row / Q8_0_BLOCK_BYTES;
    let mut packed = vec![0u8; row_pairs * n_blocks * PAIR_BLOCK_BYTES];

    for rp in 0..row_pairs {
        let row0 = rp * 2;
        let row1 = row0 + 1;
        for bi in 0..n_blocks {
            let dst = (rp * n_blocks + bi) * PAIR_BLOCK_BYTES;
            let src0 = row0 * bytes_per_row + bi * Q8_0_BLOCK_BYTES;
            packed[dst..dst + 2].copy_from_slice(&bytes[src0..src0 + 2]);
            if row1 < rows {
                let src1 = row1 * bytes_per_row + bi * Q8_0_BLOCK_BYTES;
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

pub fn pack_q8_0_tile8(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_ELEMS: usize = 32;
    const TILE_ROWS: usize = 8;
    const TILE_BLOCK_BYTES: usize = TILE_ROWS * 2 + TILE_ROWS * Q8_0_BLOCK_ELEMS;

    assert!(
        cols % Q8_0_BLOCK_ELEMS == 0,
        "Q8_0 tile8 pack requires cols divisible by {Q8_0_BLOCK_ELEMS}"
    );
    let blocks = cols / Q8_0_BLOCK_ELEMS;
    let bytes_per_row = blocks * Q8_0_BLOCK_BYTES;
    assert_eq!(
        bytes.len(),
        rows * bytes_per_row,
        "unexpected Q8_0 source byte length"
    );

    let row_tiles = rows.div_ceil(TILE_ROWS);
    let mut packed = vec![0u8; row_tiles * blocks * TILE_BLOCK_BYTES];
    for tile in 0..row_tiles {
        for bi in 0..blocks {
            let dst = (tile * blocks + bi) * TILE_BLOCK_BYTES;
            for tr in 0..TILE_ROWS {
                let row = tile * TILE_ROWS + tr;
                if row < rows {
                    let src = row * bytes_per_row + bi * Q8_0_BLOCK_BYTES;
                    packed[dst + tr * 2..dst + tr * 2 + 2].copy_from_slice(&bytes[src..src + 2]);
                    for chunk in 0..8usize {
                        let src_q = src + 2 + chunk * 4;
                        let dst_q = dst + TILE_ROWS * 2 + chunk * TILE_ROWS * 4 + tr * 4;
                        packed[dst_q..dst_q + 4].copy_from_slice(&bytes[src_q..src_q + 4]);
                    }
                }
            }
        }
    }

    packed
}

#[cfg(test)]
mod tests {
    use super::{pack_q8_0_row_pairs, pack_q8_0_tile8};

    #[test]
    fn test_pack_q8_0_row_pairs_shape() {
        let rows = 3;
        let cols = 64;
        let src = vec![0u8; rows * (cols / 32) * 34];
        let packed = pack_q8_0_row_pairs(&src, rows, cols);
        assert_eq!(packed.len(), rows.div_ceil(2) * (cols / 32) * 68);
    }

    #[test]
    fn test_pack_q8_0_row_pairs_interleaves_two_rows() {
        let rows = 2;
        let cols = 32;
        let mut src = vec![0u8; rows * 34];
        src[0..2].copy_from_slice(&[1, 2]);
        src[2..34].copy_from_slice(&[10; 32]);
        src[34..36].copy_from_slice(&[3, 4]);
        src[36..68].copy_from_slice(&[20; 32]);

        let packed = pack_q8_0_row_pairs(&src, rows, cols);
        assert_eq!(&packed[0..4], &[1, 2, 3, 4]);
        for ki in 0..4 {
            let base = 4 + ki * 16;
            assert_eq!(&packed[base..base + 8], &[10; 8]);
            assert_eq!(&packed[base + 8..base + 16], &[20; 8]);
        }
    }

    #[test]
    fn test_pack_q8_0_tile8_shape() {
        let rows = 9;
        let cols = 64;
        let src = vec![0u8; rows * (cols / 32) * 34];
        let packed = pack_q8_0_tile8(&src, rows, cols);
        assert_eq!(packed.len(), rows.div_ceil(8) * (cols / 32) * 272);
    }

    #[test]
    fn test_pack_q8_0_tile8_scales_then_rows() {
        let rows = 2;
        let cols = 32;
        let mut src = vec![0u8; rows * 34];
        src[0..2].copy_from_slice(&[1, 2]);
        src[2..34].copy_from_slice(&[10; 32]);
        src[34..36].copy_from_slice(&[3, 4]);
        src[36..68].copy_from_slice(&[20; 32]);

        let packed = pack_q8_0_tile8(&src, rows, cols);
        assert_eq!(&packed[0..4], &[1, 2, 3, 4]);
        assert_eq!(&packed[4..16], &[0; 12]);
        for chunk in 0..8 {
            let base = 16 + chunk * 32;
            assert_eq!(&packed[base..base + 4], &[10; 4]);
            assert_eq!(&packed[base + 4..base + 8], &[20; 4]);
            assert_eq!(&packed[base + 8..base + 32], &[0; 24]);
        }
    }
}
