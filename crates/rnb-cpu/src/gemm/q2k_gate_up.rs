//! Q2_K gate/up fused tile GEMV kernels.

#[cfg(any(test, not(target_arch = "aarch64")))]
use crate::quantize::dequantize_q2_k;
use crate::quantize::BlockQ2_K;

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q2k_gate_up_block_pair(g_chunk: &[u8], u_chunk: &[u8], x: *const f32) -> [f32; 2] {
    let g_block = &*(g_chunk.as_ptr() as *const BlockQ2_K);
    let u_block = &*(u_chunk.as_ptr() as *const BlockQ2_K);
    crate::quantize::dot_q2k_fused_neon_pair(g_block, u_block, x)
}

#[inline]
pub fn gemv_q2k_gate_up_tile(
    packed: &[u8],
    x: &[f32],
    n_ff: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let tile_rows = crate::gemm::shadow_tile_q2k::SHADOW_Q2K_TILE_ROWS;
    let block_size = crate::gemm::shadow_tile_q2k::SHADOW_Q2K_BLOCK_ELEMS;
    let block_bytes = crate::gemm::shadow_tile_q2k::SHADOW_Q2K_BLOCK_BYTES;
    let row_blocks = cols / block_size;
    assert_eq!(cols % block_size, 0);
    assert_eq!(n_ff % tile_rows, 0);
    assert_eq!(
        packed.len(),
        crate::gemm::shadow_tile_q2k::q2k_gate_up_tile_bytes_per_expert(n_ff, cols)
    );

    let tile_stride = tile_rows * row_blocks * 2 * block_bytes;
    let mut gate_out = vec![0.0f32; n_ff];
    let mut up_out = vec![0.0f32; n_ff];

    for tile_start in (0..n_ff).step_by(tile_rows) {
        let tile_off = (tile_start / tile_rows) * tile_stride;
        for bi in 0..row_blocks {
            let block_off = tile_off + bi * tile_rows * 2 * block_bytes;
            let xb = &x[bi * block_size..bi * block_size + block_size];
            for lr in 0..tile_rows {
                let row = tile_start + lr;
                let src = block_off + lr * 2 * block_bytes;
                let g_chunk = &packed[src..src + block_bytes];
                let u_chunk = &packed[src + block_bytes..src + 2 * block_bytes];

                #[cfg(target_arch = "aarch64")]
                {
                    let pair = unsafe { dot_q2k_gate_up_block_pair(g_chunk, u_chunk, xb.as_ptr()) };
                    gate_out[row] += pair[0];
                    up_out[row] += pair[1];
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    let g_block = unsafe { &*(g_chunk.as_ptr() as *const BlockQ2_K) };
                    let u_block = unsafe { &*(u_chunk.as_ptr() as *const BlockQ2_K) };
                    let mut tmp_g = [0.0f32; 256];
                    let mut tmp_u = [0.0f32; 256];
                    dequantize_q2_k(g_block, &mut tmp_g);
                    dequantize_q2_k(u_block, &mut tmp_u);
                    gate_out[row] += tmp_g
                        .iter()
                        .zip(xb.iter())
                        .map(|(w, xv)| w * xv)
                        .sum::<f32>();
                    up_out[row] += tmp_u
                        .iter()
                        .zip(xb.iter())
                        .map(|(w, xv)| w * xv)
                        .sum::<f32>();
                }
            }
        }
    }

    (gate_out, up_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::shadow_tile_q2k::pack_q2k_gate_up_tile;
    use half::f16;

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed >> 16) & 0xFFFF
    }

    fn fill_q2k_row(row: &mut [u8], seed: &mut u32) {
        for block_bytes in row.chunks_exact_mut(84) {
            let block = unsafe { &mut *(block_bytes.as_mut_ptr() as *mut BlockQ2_K) };
            block.d = f16::from_bits((lcg(seed) & 0xFFFF) as u16);
            block.dmin = f16::from_bits((lcg(seed) & 0xFFFF) as u16);
            for s in block.scales.iter_mut() {
                *s = lcg(seed) as u8;
            }
            for q in block.qs.iter_mut() {
                *q = lcg(seed) as u8;
            }
        }
    }

    fn dot_q2k_row_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut tmp = [0.0f32; 256];
        for (bi, chunk) in row_bytes.chunks_exact(84).enumerate() {
            let block = unsafe { &*(chunk.as_ptr() as *const BlockQ2_K) };
            dequantize_q2_k(block, &mut tmp);
            let xb = &x[bi * 256..bi * 256 + 256];
            for i in 0..256 {
                acc += tmp[i] * xb[i];
            }
        }
        acc
    }

    #[test]
    fn gemv_q2k_gate_up_tile_matches_row_major_oracle() {
        let n_ff = 8usize;
        let cols = 512usize;
        let row_bytes = cols / 256 * 84;
        let mut seed = 0xACDC_u32;
        let mut gate = vec![0u8; n_ff * row_bytes];
        let mut up = vec![0u8; n_ff * row_bytes];

        fill_q2k_row(&mut gate, &mut seed);
        fill_q2k_row(&mut up, &mut seed);

        let tile = pack_q2k_gate_up_tile(&gate, &up, n_ff, cols);
        let mut x = vec![0.0f32; cols];
        for (i, v) in x.iter_mut().enumerate() {
            let s = (lcg(&mut seed) & 0xFFFF) as i32 - 0x8000;
            *v = (s as f32) / 65536.0;
            if i & 1 == 0 {
                *v = -*v;
            }
        }

        let mut gate_oracle = vec![0.0f32; n_ff];
        let mut up_oracle = vec![0.0f32; n_ff];
        for r in 0..n_ff {
            gate_oracle[r] = dot_q2k_row_scalar(&gate[r * row_bytes..(r + 1) * row_bytes], &x);
            up_oracle[r] = dot_q2k_row_scalar(&up[r * row_bytes..(r + 1) * row_bytes], &x);
        }

        let (gate_out, up_out) = gemv_q2k_gate_up_tile(&tile, &x, n_ff, cols);
        assert_eq!(gate_out.len(), n_ff);
        assert_eq!(up_out.len(), n_ff);
        for r in 0..n_ff {
            if !gate_oracle[r].is_finite()
                || !up_oracle[r].is_finite()
                || !gate_out[r].is_finite()
                || !up_out[r].is_finite()
            {
                continue;
            }
            let dg = (gate_oracle[r] - gate_out[r]).abs();
            let du = (up_oracle[r] - up_out[r]).abs();
            let tg = 1e-4_f32 * gate_oracle[r].abs().max(1.0);
            let tu = 1e-4_f32 * up_oracle[r].abs().max(1.0);
            assert!(dg <= tg, "row {r} gate diff={dg} tol={tg}");
            assert!(du <= tu, "row {r} up diff={du} tol={tu}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q2k_gate_up_block_pair_matches_dual_single_row_dots() {
        let mut seed = 0xBEEF_u32;
        let mut gate_chunk = [0u8; 84];
        let mut up_chunk = [0u8; 84];
        fill_q2k_row(&mut gate_chunk, &mut seed);
        fill_q2k_row(&mut up_chunk, &mut seed);

        let gate_block = unsafe { &*(gate_chunk.as_ptr() as *const BlockQ2_K) };
        let up_block = unsafe { &*(up_chunk.as_ptr() as *const BlockQ2_K) };

        let mut x = [0.0f32; 256];
        for (i, v) in x.iter_mut().enumerate() {
            let s = (lcg(&mut seed) & 0xFFFF) as i32 - 0x8000;
            *v = (s as f32) / 65536.0;
            if i & 1 == 0 {
                *v = -*v;
            }
        }

        let gate_single = unsafe { crate::quantize::dot_q2k_fused_neon(gate_block, x.as_ptr()) };
        let up_single = unsafe { crate::quantize::dot_q2k_fused_neon(up_block, x.as_ptr()) };
        let pair = unsafe { dot_q2k_gate_up_block_pair(&gate_chunk, &up_chunk, x.as_ptr()) };

        if gate_single.is_finite() && pair[0].is_finite() {
            let diff = (gate_single - pair[0]).abs();
            let tol = 1e-4_f32 * gate_single.abs().max(1.0);
            assert!(diff <= tol, "gate diff={diff} tol={tol}");
        }
        if up_single.is_finite() && pair[1].is_finite() {
            let diff = (up_single - pair[1]).abs();
            let tol = 1e-4_f32 * up_single.abs().max(1.0);
            assert!(diff <= tol, "up diff={diff} tol={tol}");
        }
    }
}
