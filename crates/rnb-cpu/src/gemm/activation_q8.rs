#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use rayon::prelude::*;

/// Pre-quantized input block (32 elements -> int8 + scale).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Q8Block {
    pub d: f32,
    pub qs: [i8; 32],
}

/// Pre-quantize f32 input vector to Q8 blocks (32 elements each).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn quantize_input_q8(input: &[f32]) -> Vec<Q8Block> {
    let n_blocks = input.len() / 32;
    let mut blocks = Vec::with_capacity(n_blocks);
    for bi in 0..n_blocks {
        let chunk = &input[bi * 32..(bi + 1) * 32];
        let mut amax = 0.0f32;
        for &x in chunk {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
        }
        blocks.push(Q8Block { d, qs });
    }
    blocks
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn quantize_input_q8_into(input: &[f32], output: &mut [Q8Block]) {
    let n_blocks = input.len() / 32;
    assert_eq!(output.len(), n_blocks);
    for bi in 0..n_blocks {
        let chunk = &input[bi * 32..(bi + 1) * 32];
        let mut amax = 0.0f32;
        for &x in chunk {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
        }
        output[bi] = Q8Block { d, qs };
    }
}

/// Pre-quantized input block for K-quants (256 elements -> int8 + scale + bsums).
///
/// `repr(C)` keeps the layout stable for assembly kernels:
/// `d(f32)` at 0, `qs([i8; 256])` at 4, `bsums([i16; 16])` at 260.
///
/// mc71/mc72 — `bsums` widened from 8×i16 (per-32-element) to 16×i16
/// (per-16-element) to match GGML Q8_K layout, which lets us reconstruct
/// GGML's `isum_mins` exactly in `dot_q6_k_q8k_neon_ggml_align`. Q4_K dot
/// kernels can still treat `(bsums[2k] + bsums[2k+1])` as the old 32-element
/// bsum (matching GGML's `vpaddq_s16`), so they remain mathematically
/// equivalent without code changes — only the bookkeeping doubled.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8KBlock {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl Q8KBlock {
    /// Backward-compat helper: sum of two adjacent 16-element bsums to get
    /// the legacy 32-element bsum. Use this in dot kernels that operate on
    /// 8 per-32-element sub-blocks (Q4_K, Q5_K mins, etc.).
    #[inline(always)]
    pub fn bsum32(&self, k: usize) -> i16 {
        self.bsums[2 * k] + self.bsums[2 * k + 1]
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl Default for Q8KBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }
    }
}

/// Pre-quantize f32 input to Q8K blocks (256 elements each).
///
/// mc71 — aligned with GGML's `quantize_row_q8_K_ref`:
/// - `iscale = -127.f / max` (signed max, not amax) so q sign is inverted
///   relative to a positive-d convention.
/// - `d = 1 / iscale` (signed) is stored on the block; downstream dot
///   kernels see the same `d × q` product as GGML, but the per-element bit
///   pattern of q matches GGML exactly.
/// - upper clamp only (`MIN(127, v)`); the inverse-sign trick ensures the
///   range stays in `[-128, 127]` without an explicit lower clamp.
/// - `nearest_int` semantics (round-half-to-even on ties) match the
///   reference `iround` GGML uses inside `nearest_int`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn quantize_input_q8k(input: &[f32]) -> Vec<Q8KBlock> {
    let n_blocks = input.len() / 256;
    if n_blocks >= rayon::current_num_threads().max(1) * 4 {
        input
            .par_chunks_exact(256)
            .map(quantize_q8k_block)
            .collect()
    } else {
        input.chunks_exact(256).map(quantize_q8k_block).collect()
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn quantize_input_q8k_into(input: &[f32], output: &mut [Q8KBlock]) {
    let n_blocks = input.len() / 256;
    assert_eq!(output.len(), n_blocks);
    let quantize = |(chunk, output): (&[f32], &mut Q8KBlock)| {
        *output = quantize_q8k_block(chunk);
    };
    if n_blocks >= rayon::current_num_threads().max(1) * 4 {
        input
            .par_chunks_exact(256)
            .zip(output.par_iter_mut())
            .for_each(quantize);
    } else {
        input
            .chunks_exact(256)
            .zip(output.iter_mut())
            .for_each(quantize);
    }
}

#[inline]
fn quantize_q8k_block(chunk: &[f32]) -> Q8KBlock {
    let (d, qs, bsums) = quantize_q8k_block_ggml(chunk);
    Q8KBlock { d, qs, bsums }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
pub(super) fn quantize_q8k_block_ggml(chunk: &[f32]) -> (f32, [i8; 256], [i16; 16]) {
    let mut amax = 0.0f32;
    let mut max_val = 0.0f32;
    for &x in chunk {
        let ax = x.abs();
        if ax > amax {
            amax = ax;
            max_val = x;
        }
    }
    if amax == 0.0 {
        return (0.0, [0i8; 256], [0i16; 16]);
    }
    let iscale = -127.0 / max_val; // signed: matches GGML quantize_row_q8_K_ref
    let d = 1.0 / iscale;
    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];
    for i in 0..256 {
        let v = (iscale * chunk[i]).round_ties_even() as i32;
        let q = v.min(127) as i8; // upper clamp only, GGML convention
        qs[i] = q;
        bsums[i / 16] += q as i16;
    }
    (d, qs, bsums)
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
mod tests {
    use super::{quantize_input_q8k, quantize_input_q8k_into, quantize_q8k_block_ggml, Q8KBlock};

    #[test]
    fn q8k_quantization_matches_ggml_ties_and_bsums() {
        let zero = [0.0f32; 256];
        assert_eq!(
            quantize_q8k_block_ggml(&zero),
            (0.0, [0i8; 256], [0i16; 16])
        );

        let mut ties = [0.0f32; 256];
        ties[0] = 127.0;
        ties[1..7].copy_from_slice(&[-0.5, -1.5, -2.5, 0.5, 1.5, 2.5]);
        let (d, qs, _) = quantize_q8k_block_ggml(&ties);
        assert_eq!(d, -1.0);
        assert_eq!(&qs[1..7], &[0, 2, 2, 0, -2, -2]);

        let mut first_positive = [0.0f32; 256];
        first_positive[0] = 127.0;
        first_positive[1] = -127.0;
        let (d, qs, _) = quantize_q8k_block_ggml(&first_positive);
        assert_eq!(d, -1.0);
        assert_eq!(&qs[..2], &[-127, 127]);

        let mut first_negative = [0.0f32; 256];
        first_negative[0] = -127.0;
        first_negative[1] = 127.0;
        let (d, qs, _) = quantize_q8k_block_ggml(&first_negative);
        assert_eq!(d, 1.0);
        assert_eq!(&qs[..2], &[-127, 127]);

        let mut boundaries = [0.0f32; 256];
        boundaries[15] = -1.0;
        boundaries[16] = -2.0;
        boundaries[255] = 127.0;
        let (_, _, bsums) = quantize_q8k_block_ggml(&boundaries);
        assert_eq!(bsums[0], 1);
        assert_eq!(bsums[1], 2);
        assert!(bsums[2..15].iter().all(|&sum| sum == 0));
        assert_eq!(bsums[15], -127);
    }

    #[test]
    fn q8k_parallel_batch_matches_block_oracle() {
        const BLOCKS: usize = 16;
        let input: Vec<f32> = (0..BLOCKS * 256)
            .map(|index| ((index * 73 + 19) % 2001) as f32 - 1000.0)
            .collect();
        let expected: Vec<Q8KBlock> = input
            .chunks_exact(256)
            .map(|chunk| {
                let (d, qs, bsums) = quantize_q8k_block_ggml(chunk);
                Q8KBlock { d, qs, bsums }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let mut output = vec![Q8KBlock::default(); BLOCKS];
        pool.install(|| quantize_input_q8k_into(&input, &mut output));
        assert_eq!(output, expected);
        assert_eq!(pool.install(|| quantize_input_q8k(&input)), expected);
    }
}
