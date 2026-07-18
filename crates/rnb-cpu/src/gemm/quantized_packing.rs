#[cfg(target_arch = "aarch64")]
use super::Q8KBlock;

pub fn build_q80_f32_scales(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    total_bytes: usize,
) -> Vec<f32> {
    let n_blocks = cols / 32;
    let bytes_per_row = total_bytes / rows;
    let mut scales = vec![0.0f32; rows * n_blocks];
    for row in 0..rows {
        let row_off = row * bytes_per_row;
        for bi in 0..n_blocks {
            let off = row_off + bi * 34;
            let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            scales[row * n_blocks + bi] = half::f16::from_bits(bits).to_f32();
        }
    }
    scales
}

#[cfg(target_arch = "aarch64")]
pub fn flatten_q8k_blocks(q8k: &[Q8KBlock]) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
    let total_blocks = q8k.len();
    let mut qs_flat = vec![0i8; total_blocks * 256];
    let mut d_flat = vec![0.0f32; total_blocks];
    // mc72: Q8KBlock.bsums is now 16 per-16-element entries, but the legacy
    // tile-GEMM kernels (tile_q4k / tile_q5k) still expect 8 per-32-element
    // bsums. Pair-add adjacent 16-element halves to recover the legacy
    // 32-element layout for these flat outputs; new ggml-aligned dot kernels
    // (`dot_q*_q8k_neon_ggml_align`) read `q8k.bsums` directly (16 entries).
    let mut bsums_flat = vec![0i16; total_blocks * 8];
    for (i, blk) in q8k.iter().enumerate() {
        qs_flat[i * 256..(i + 1) * 256].copy_from_slice(&blk.qs);
        d_flat[i] = blk.d;
        for k in 0..8 {
            bsums_flat[i * 8 + k] = blk.bsums[2 * k] + blk.bsums[2 * k + 1];
        }
    }
    (qs_flat, d_flat, bsums_flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_q80_f32_scales_reads_row_major_f16_scales() {
        let rows = 2;
        let cols = 64;
        let total_bytes = rows * (cols / 32) * 34;
        let mut bytes = vec![0u8; total_bytes];

        let scales = [1.25f32, -0.5, 3.0, 0.0];
        for (i, scale) in scales.iter().enumerate() {
            let row = i / 2;
            let block = i % 2;
            let off = row * 68 + block * 34;
            bytes[off..off + 2]
                .copy_from_slice(&half::f16::from_f32(*scale).to_bits().to_le_bytes());
        }

        assert_eq!(
            build_q80_f32_scales(&bytes, rows, cols, total_bytes),
            scales
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn flatten_q8k_blocks_preserves_block_order() {
        let mut first = Q8KBlock::default();
        first.d = 0.25;
        first.qs[0] = -7;
        first.qs[255] = 9;
        first.bsums[0] = -12;
        first.bsums[15] = 34;

        let mut second = Q8KBlock::default();
        second.d = 1.5;
        second.qs[0] = 11;
        second.qs[255] = -13;
        second.bsums[0] = 56;
        second.bsums[15] = -78;

        let (qs_flat, d_flat, bsums_flat) = flatten_q8k_blocks(&[first, second]);

        assert_eq!(qs_flat.len(), 512);
        assert_eq!(qs_flat[0], -7);
        assert_eq!(qs_flat[255], 9);
        assert_eq!(qs_flat[256], 11);
        assert_eq!(qs_flat[511], -13);
        assert_eq!(d_flat, [0.25, 1.5]);
        assert_eq!(bsums_flat.len(), 16);
        assert_eq!(bsums_flat[0], -12);
        assert_eq!(bsums_flat[7], 34);
        assert_eq!(bsums_flat[8], 56);
        assert_eq!(bsums_flat[15], -78);
    }
}
