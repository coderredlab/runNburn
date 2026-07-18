//! Scalar quantized GEMV kernels.

use crate::quantize as q;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGemvType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2XXS,
    IQ2S,
    IQ3XXS,
}

impl QuantGemvType {
    #[inline]
    fn block_elems(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::IQ2XXS
            | Self::IQ2S
            | Self::IQ3XXS => 256,
        }
    }

    #[inline]
    fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::IQ2XXS => 66,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
        }
    }
}

#[inline]
fn avx2_eligible(quant: QuantGemvType, cols: usize, avx2: bool, fma: bool) -> bool {
    matches!(
        quant,
        QuantGemvType::Q2K
            | QuantGemvType::Q3K
            | QuantGemvType::Q4K
            | QuantGemvType::Q5K
            | QuantGemvType::Q6K
    ) && cols % 256 == 0
        && avx2
        && fma
}

#[cfg(target_arch = "x86_64")]
fn q2q3_batch_x4_enabled() -> bool {
    std::env::var("RNB_CPU_Q2Q3_BATCH_X4")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[cfg(target_arch = "x86_64")]
type Avx2DotFn = unsafe fn(&[u8], &[crate::gemm::activation_q8::Q8KBlock]) -> f32;

#[cfg(target_arch = "x86_64")]
fn avx2_dot_fn(quant: QuantGemvType) -> Option<Avx2DotFn> {
    use crate::gemm::avx2_dot::{
        dot_q2k_q8k_avx2, dot_q3k_q8k_avx2, dot_q4k_q8k_avx2, dot_q5k_q8k_avx2, dot_q6k_q8k_avx2,
    };

    Some(match quant {
        QuantGemvType::Q2K => dot_q2k_q8k_avx2,
        QuantGemvType::Q3K => dot_q3k_q8k_avx2,
        QuantGemvType::Q4K => dot_q4k_q8k_avx2,
        QuantGemvType::Q5K => dot_q5k_q8k_avx2,
        QuantGemvType::Q6K => dot_q6k_q8k_avx2,
        _ => return None,
    })
}

#[inline]
pub fn dot_q4_0_row(row_bytes: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for bi in 0..blocks_per_row {
        let boff = bi * 18;
        let d = half::f16::from_bits(u16::from_le_bytes([row_bytes[boff], row_bytes[boff + 1]]))
            .to_f32();
        let qs = &row_bytes[boff + 2..boff + 18];
        let xb = &x[bi * 32..];
        let mut block_sum = 0.0f32;
        for i in 0..16 {
            block_sum += ((qs[i] & 0x0F) as f32 - 8.0) * xb[i];
            block_sum += ((qs[i] >> 4) as f32 - 8.0) * xb[i + 16];
        }
        acc += d * block_sum;
    }
    acc
}

#[inline]
pub fn dot_q8_0_row(row_bytes: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for bi in 0..blocks_per_row {
        let boff = bi * 34;
        let d = half::f16::from_bits(u16::from_le_bytes([row_bytes[boff], row_bytes[boff + 1]]))
            .to_f32();
        let xb = &x[bi * 32..];
        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += row_bytes[boff + 2 + i] as i8 as f32 * xb[i];
        }
        acc += d * block_sum;
    }
    acc
}

/// Basic-block row dot for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0.
///
/// This keeps the sequential accumulation order used by the full-row fallback.
/// MoE router softmax argmax is tie-sensitive, so changing f32 accumulation
/// order can change generated tokens.
#[inline]
fn dot_basic_blocks_scalar(row_bytes: &[u8], x: &[f32], cols: usize, quant: QuantGemvType) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if matches!(quant, QuantGemvType::Q5_1) {
        let n_blocks = cols / 32;
        let mut acc = 0.0f32;
        for bi in 0..n_blocks {
            let block = unsafe { &*(row_bytes.as_ptr().add(bi * 24) as *const q::BlockQ5_1) };
            let xb_ptr = unsafe { x.as_ptr().add(bi * 32) };
            acc += unsafe { q::dot_q5_1_fused_neon(block, xb_ptr) };
        }
        return acc;
    }

    if matches!(quant, QuantGemvType::Q5_1) {
        let n_blocks = cols / 32;
        let mut acc = 0.0f32;
        for bi in 0..n_blocks {
            let block = unsafe { &*(row_bytes.as_ptr().add(bi * 24) as *const q::BlockQ5_1) };
            let xb_ptr = unsafe { x.as_ptr().add(bi * 32) };
            let xb_arr: &[f32; 32] = unsafe { &*(xb_ptr as *const [f32; 32]) };
            acc += q::dot_q5_1_chunked_scalar(block, xb_arr);
        }
        return acc;
    }

    let block_bytes = quant.block_bytes();
    let n_blocks = cols / 32;
    let mut acc = 0.0f32;
    let mut tmp = [0.0f32; 32];
    for bi in 0..n_blocks {
        let chunk = &row_bytes[bi * block_bytes..(bi + 1) * block_bytes];
        match quant {
            QuantGemvType::Q4_0 => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ4_0) };
                q::dequantize_q4_0(block, &mut tmp);
            }
            QuantGemvType::Q4_1 => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ4_1) };
                q::dequantize_q4_1(block, &mut tmp);
            }
            QuantGemvType::Q5_0 => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ5_0) };
                q::dequantize_q5_0(block, &mut tmp);
            }
            QuantGemvType::Q5_1 => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ5_1) };
                q::dequantize_q5_1(block, &mut tmp);
            }
            QuantGemvType::Q8_0 => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ8_0) };
                q::dequantize_q8_0(block, &mut tmp);
            }
            _ => unreachable!("basic quant expected"),
        }
        let xb = &x[bi * 32..];
        for i in 0..32 {
            acc += tmp[i] * xb[i];
        }
    }
    acc
}

#[inline]
pub fn dot_quantized_row(row_bytes: &[u8], x: &[f32], cols: usize, quant: QuantGemvType) -> f32 {
    match quant {
        QuantGemvType::Q4_0
        | QuantGemvType::Q4_1
        | QuantGemvType::Q5_0
        | QuantGemvType::Q5_1
        | QuantGemvType::Q8_0 => return dot_basic_blocks_scalar(row_bytes, x, cols, quant),
        _ => {}
    }

    let block_size = quant.block_elems();
    let block_bytes = quant.block_bytes();
    let n_blocks = cols / block_size;
    let mut acc = 0.0f32;
    let mut tmp = [0.0f32; 256];

    for bi in 0..n_blocks {
        let bstart = bi * block_bytes;
        let bend = bstart + block_bytes;
        let chunk = &row_bytes[bstart..bend];
        let xb = &x[bi * block_size..];

        #[cfg(target_arch = "aarch64")]
        if matches!(quant, QuantGemvType::Q4K) {
            let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ4_K) };
            acc += unsafe { q::dot_q4k_fused_neon(block, xb.as_ptr()) };
            continue;
        }

        #[cfg(target_arch = "aarch64")]
        if matches!(quant, QuantGemvType::Q2K) {
            let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ2_K) };
            acc += unsafe { q::dot_q2k_fused_neon(block, xb.as_ptr()) };
            continue;
        }

        match quant {
            QuantGemvType::Q4K => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ4_K) };
                q::dequantize_q4_k(block, &mut tmp);
            }
            QuantGemvType::Q5K => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ5_K) };
                q::dequantize_q5_k(block, &mut tmp);
            }
            QuantGemvType::Q6K => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ6_K) };
                q::dequantize_q6_k(block, &mut tmp);
            }
            QuantGemvType::Q3K => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ3_K) };
                q::dequantize_q3_k(block, &mut tmp);
            }
            QuantGemvType::Q2K => {
                let block = unsafe { &*(chunk.as_ptr() as *const q::BlockQ2_K) };
                q::dequantize_q2_k(block, &mut tmp);
            }
            QuantGemvType::IQ2XXS => q::iq::dequantize_iq2_xxs_block(chunk, &mut tmp),
            QuantGemvType::IQ2S => q::iq::dequantize_iq2_s_block(chunk, &mut tmp),
            QuantGemvType::IQ3XXS => q::iq::dequantize_iq3_xxs_block(chunk, &mut tmp),
            _ => unreachable!("k quant expected"),
        }

        #[cfg(target_arch = "aarch64")]
        {
            acc += unsafe {
                crate::gemm::neon_dot::dot_f32_neon(tmp.as_ptr(), xb.as_ptr(), block_size)
            };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut block_acc = 0.0f32;
            for i in 0..block_size {
                block_acc += tmp[i] * xb[i];
            }
            acc += block_acc;
        }
    }
    acc
}

pub fn gemv_q4_0(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let blocks_per_row = cols / 32;
    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            let target_tasks = n_threads * 4;
            ((rows + target_tasks - 1) / target_tasks).max(1024)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = dot_q4_0_row(rb, x_data, blocks_per_row);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    out[s] = dot_q4_0_row(rb, &x_data[s * cols..], blocks_per_row);
                }
            });
        transpose_row_major_to_seq_major(&row_major, output, rows, seq_len);
    }
}

pub fn gemv_q8_0(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let blocks_per_row = cols / 32;
    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = dot_q8_0_row(rb, x_data, blocks_per_row);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    out[s] = dot_q8_0_row(rb, &x_data[s * cols..], blocks_per_row);
                }
            });
        transpose_row_major_to_seq_major(&row_major, output, rows, seq_len);
    }
}

pub fn gemv_quantized(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    quant: QuantGemvType,
) {
    #[cfg(target_arch = "aarch64")]
    if aarch64_kquant_q8k_gemv_enabled()
        && matches!(
            quant,
            QuantGemvType::Q4K | QuantGemvType::Q5K | QuantGemvType::Q6K
        )
        && cols % 256 == 0
        && std::arch::is_aarch64_feature_detected!("dotprod")
    {
        gemv_quantized_aarch64_q8k(
            bytes,
            x_data,
            output,
            rows,
            cols,
            seq_len,
            bytes_per_row,
            quant,
        );
        return;
    }

    // x86_64 AVX2 fast path for K-quants. Input is pre-quantized to Q8K once
    // and reused across all output rows × seq_len positions.
    #[cfg(target_arch = "x86_64")]
    if avx2_eligible(
        quant,
        cols,
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
    ) {
        gemv_quantized_avx2(
            bytes,
            x_data,
            output,
            rows,
            cols,
            seq_len,
            bytes_per_row,
            quant,
        );
        return;
    }

    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = dot_quantized_row(rb, x_data, cols, quant);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    out[s] = dot_quantized_row(rb, &x_data[s * cols..], cols, quant);
                }
            });
        transpose_row_major_to_seq_major(&row_major, output, rows, seq_len);
    }
}

#[cfg(target_arch = "aarch64")]
fn aarch64_kquant_q8k_gemv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(aarch64_kquant_q8k_gemv_enabled_from_env)
}

#[cfg(target_arch = "aarch64")]
fn aarch64_kquant_q8k_gemv_enabled_from_env() -> bool {
    std::env::var("RNB_AARCH64_KQUANT_Q8K_GEMV")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(cfg!(target_vendor = "apple"))
}

/// Apple Silicon Q4_K/Q5_K/Q6_K × Q8_K GEMV. Input is Q8K-quantized once per
/// seq position and reused across rows, matching the llama.cpp K-quant pattern.
#[cfg(target_arch = "aarch64")]
fn gemv_quantized_aarch64_q8k(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    quant: QuantGemvType,
) {
    use crate::gemm::activation_q8::{quantize_input_q8k_into, Q8KBlock};
    use crate::gemm::neon_dot::{dot_q4_k_q8k_neon, dot_q5_k_q8k_neon, dot_q6_k_q8k_neon};

    let n_blocks = cols / 256;
    let mut input_q8k = vec![Q8KBlock::default(); seq_len * n_blocks];
    for s in 0..seq_len {
        quantize_input_q8k_into(
            &x_data[s * cols..(s + 1) * cols],
            &mut input_q8k[s * n_blocks..(s + 1) * n_blocks],
        );
    }

    let dot_fn: unsafe fn(&[u8], &[Q8KBlock], usize) -> f32 = match quant {
        QuantGemvType::Q4K => dot_q4_k_q8k_neon,
        QuantGemvType::Q5K => dot_q5_k_q8k_neon,
        QuantGemvType::Q6K => dot_q6_k_q8k_neon,
        _ => unreachable!("aarch64 Q8K path only for K-quants"),
    };

    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = unsafe { dot_fn(rb, &input_q8k, n_blocks) };
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                for s in 0..seq_len {
                    let q8k_slice = &input_q8k[s * n_blocks..(s + 1) * n_blocks];
                    out[s] = unsafe { dot_fn(rb, q8k_slice, n_blocks) };
                }
            });
        transpose_row_major_to_seq_major(&row_major, output, rows, seq_len);
    }
}

/// AArch64 paired Q4_K gate/up GEMV for MoE decode. Quantize the shared
/// activation once, then choose the weight traversal with better locality.
#[cfg(target_arch = "aarch64")]
pub fn gemv_q4k_pair_aarch64_q8k(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    x_data: &[f32],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
) -> bool {
    use crate::gemm::activation_q8::{quantize_input_q8k_into, Q8KBlock};

    if cols % 256 != 0 || !std::arch::is_aarch64_feature_detected!("dotprod") {
        return false;
    }

    let n_blocks = cols / 256;
    let mut input_q8k = vec![Q8KBlock::default(); n_blocks];
    quantize_input_q8k_into(x_data, &mut input_q8k);
    gemv_q4k_pair_aarch64_q8k_prequantized(
        gate_bytes,
        up_bytes,
        &input_q8k,
        gate_output,
        up_output,
        rows,
        cols,
        gate_bytes_per_row,
        up_bytes_per_row,
    )
}

#[cfg(target_arch = "aarch64")]
pub fn gemv_q4k_pair_aarch64_q8k_prequantized(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    input_q8k: &[crate::gemm::activation_q8::Q8KBlock],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
) -> bool {
    gemv_q4k_pair_aarch64_q8k_prequantized_impl(
        gate_bytes,
        up_bytes,
        input_q8k,
        gate_output,
        up_output,
        rows,
        cols,
        gate_bytes_per_row,
        up_bytes_per_row,
        true,
    )
}

#[cfg(target_arch = "aarch64")]
pub fn gemv_q4k_pair_aarch64_q8k_prequantized_serial(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    input_q8k: &[crate::gemm::activation_q8::Q8KBlock],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
) -> bool {
    gemv_q4k_pair_aarch64_q8k_prequantized_impl(
        gate_bytes,
        up_bytes,
        input_q8k,
        gate_output,
        up_output,
        rows,
        cols,
        gate_bytes_per_row,
        up_bytes_per_row,
        false,
    )
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gemv_q4k_pair_aarch64_q8k_prequantized_impl(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    input_q8k: &[crate::gemm::activation_q8::Q8KBlock],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
    parallel_rows: bool,
) -> bool {
    if cols % 256 != 0 || !std::arch::is_aarch64_feature_detected!("dotprod") {
        return false;
    }
    if gate_output.len() < rows || up_output.len() < rows {
        return false;
    }
    let n_blocks = cols / 256;
    if input_q8k.len() != n_blocks {
        return false;
    }

    let fused_raw_pair = qwen35_moe_decode_pair_fused_raw_enabled();
    let compute_row = |row: usize| {
        let gate_row = &gate_bytes[row * gate_bytes_per_row..(row + 1) * gate_bytes_per_row];
        let up_row = &up_bytes[row * up_bytes_per_row..(row + 1) * up_bytes_per_row];
        if fused_raw_pair {
            unsafe { dot_q4k_pair_q8k_neon_ggml_align(gate_row, up_row, input_q8k) }
        } else {
            use crate::gemm::neon_dot::dot_q4_k_q8k_neon;
            (
                unsafe { dot_q4_k_q8k_neon(gate_row, input_q8k, n_blocks) },
                unsafe { dot_q4_k_q8k_neon(up_row, input_q8k, n_blocks) },
            )
        }
    };

    if parallel_rows {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        gate_output[..rows]
            .par_chunks_mut(chunk)
            .zip(up_output[..rows].par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(ci, (gate_out, up_out))| {
                let start = ci * chunk;
                for i in 0..gate_out.len() {
                    let (gate, up) = compute_row(start + i);
                    gate_out[i] = gate;
                    up_out[i] = up;
                }
            });
    } else {
        for row in 0..rows {
            let (gate, up) = compute_row(row);
            gate_output[row] = gate;
            up_output[row] = up;
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
fn qwen35_moe_decode_pair_fused_raw_enabled() -> bool {
    std::env::var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4k_pair_q8k_neon_ggml_align(
    gate_row: &[u8],
    up_row: &[u8],
    q8k: &[crate::gemm::activation_q8::Q8KBlock],
) -> (f32, f32) {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn f16_pair_to_f32(base: *const u8) -> (f32, f32) {
        let d_bits = std::ptr::read_unaligned(base as *const u16);
        let dmin_bits = std::ptr::read_unaligned(base.add(2) as *const u16);
        (
            half::f16::from_bits(d_bits).to_f32(),
            half::f16::from_bits(dmin_bits).to_f32(),
        )
    }

    #[inline(always)]
    unsafe fn unpack_scales_mins(sb: *const u8) -> ([u8; 8], [u8; 8]) {
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..4 {
            sc[j] = *sb.add(j) & 63;
            mn[j] = *sb.add(j + 4) & 63;
        }
        for j in 4..8 {
            sc[j] = (*sb.add(j + 4) & 0x0F) | ((*sb.add(j - 4) >> 6) << 4);
            mn[j] = (*sb.add(j + 4) >> 4) | ((*sb.add(j) >> 6) << 4);
        }
        (sc, mn)
    }

    let mut gate_sumf = 0.0f32;
    let mut up_sumf = 0.0f32;
    let mask_low = vdupq_n_u8(0x0F);
    let n_blocks = q8k.len();

    for bi in 0..n_blocks {
        let boff = bi * 144;
        let gate_base = gate_row.as_ptr().add(boff);
        let up_base = up_row.as_ptr().add(boff);
        let (gate_dx, gate_dminx) = f16_pair_to_f32(gate_base);
        let (up_dx, up_dminx) = f16_pair_to_f32(up_base);
        let (gate_sc, gate_mn) = unpack_scales_mins(gate_base.add(4));
        let (up_sc, up_mn) = unpack_scales_mins(up_base.add(4));
        let gate_qs = gate_base.add(16);
        let up_qs = up_base.add(16);

        let q8b = q8k.get_unchecked(bi);
        let gate_d = q8b.d * gate_dx;
        let gate_dmin = q8b.d * gate_dminx;
        let up_d = q8b.d * up_dx;
        let up_dmin = q8b.d * up_dminx;

        let mut gate_summ: i32 = 0;
        let mut up_summ: i32 = 0;
        for j in 0..8 {
            let bsum = q8b.bsum32(j) as i32;
            gate_summ += gate_mn[j] as i32 * bsum;
            up_summ += up_mn[j] as i32 * bsum;
        }
        gate_sumf -= gate_dmin * gate_summ as f32;
        up_sumf -= up_dmin * up_summ as f32;

        let mut gate_sumi1: i32 = 0;
        let mut gate_sumi2: i32 = 0;
        let mut up_sumi1: i32 = 0;
        let mut up_sumi2: i32 = 0;
        for g in 0..4 {
            let x_lo_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64));
            let x_lo_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 16));
            let x_hi_0 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 32));
            let x_hi_1 = vld1q_s8(q8b.qs.as_ptr().add(g * 64 + 48));

            let gate_qbytes_lo = vld1q_u8(gate_qs.add(g * 32));
            let gate_qbytes_hi = vld1q_u8(gate_qs.add(g * 32 + 16));
            let gate_w_lo_0 = vreinterpretq_s8_u8(vandq_u8(gate_qbytes_lo, mask_low));
            let gate_w_lo_1 = vreinterpretq_s8_u8(vandq_u8(gate_qbytes_hi, mask_low));
            let gate_w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(gate_qbytes_lo, 4));
            let gate_w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(gate_qbytes_hi, 4));

            let mut gate_p1 = vdupq_n_s32(0);
            gate_p1 = vdotq_s32(gate_p1, gate_w_lo_0, x_lo_0);
            gate_p1 = vdotq_s32(gate_p1, gate_w_lo_1, x_lo_1);
            gate_sumi1 += vaddvq_s32(gate_p1) * gate_sc[2 * g] as i32;

            let mut gate_p2 = vdupq_n_s32(0);
            gate_p2 = vdotq_s32(gate_p2, gate_w_hi_0, x_hi_0);
            gate_p2 = vdotq_s32(gate_p2, gate_w_hi_1, x_hi_1);
            gate_sumi2 += vaddvq_s32(gate_p2) * gate_sc[2 * g + 1] as i32;

            let up_qbytes_lo = vld1q_u8(up_qs.add(g * 32));
            let up_qbytes_hi = vld1q_u8(up_qs.add(g * 32 + 16));
            let up_w_lo_0 = vreinterpretq_s8_u8(vandq_u8(up_qbytes_lo, mask_low));
            let up_w_lo_1 = vreinterpretq_s8_u8(vandq_u8(up_qbytes_hi, mask_low));
            let up_w_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(up_qbytes_lo, 4));
            let up_w_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(up_qbytes_hi, 4));

            let mut up_p1 = vdupq_n_s32(0);
            up_p1 = vdotq_s32(up_p1, up_w_lo_0, x_lo_0);
            up_p1 = vdotq_s32(up_p1, up_w_lo_1, x_lo_1);
            up_sumi1 += vaddvq_s32(up_p1) * up_sc[2 * g] as i32;

            let mut up_p2 = vdupq_n_s32(0);
            up_p2 = vdotq_s32(up_p2, up_w_hi_0, x_hi_0);
            up_p2 = vdotq_s32(up_p2, up_w_hi_1, x_hi_1);
            up_sumi2 += vaddvq_s32(up_p2) * up_sc[2 * g + 1] as i32;
        }

        let gate_sumi = gate_sumi1 + gate_sumi2;
        let up_sumi = up_sumi1 + up_sumi2;
        gate_sumf += gate_d * gate_sumi as f32;
        up_sumf += up_d * up_sumi as f32;
    }

    (gate_sumf, up_sumf)
}

/// AVX2 Q4_K/Q5_K/Q6_K × Q8_K dot. Input is Q8K-quantized once per token and
/// reused across rows. Outputs are written in seq-major layout.
#[cfg(target_arch = "x86_64")]
fn gemv_quantized_avx2(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    quant: QuantGemvType,
) {
    use crate::gemm::activation_q8::{quantize_input_q8k_into, Q8KBlock};

    let n_blocks = cols / 256;
    let mut input_q8k = vec![Q8KBlock::default(); seq_len * n_blocks];
    for s in 0..seq_len {
        quantize_input_q8k_into(
            &x_data[s * cols..(s + 1) * cols],
            &mut input_q8k[s * n_blocks..(s + 1) * n_blocks],
        );
    }

    let dot_fn = avx2_dot_fn(quant).expect("eligible K-quant must have an AVX2 dot");

    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    out[i] = unsafe { dot_fn(rb, &input_q8k) };
                }
            });
    } else {
        let use_q2q3_x4 = seq_len >= 4
            && matches!(quant, QuantGemvType::Q2K | QuantGemvType::Q3K)
            && q2q3_batch_x4_enabled();
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                let mut s = 0;
                if use_q2q3_x4 {
                    use crate::gemm::avx2_dot::{dot_q2k_q8k_avx2_x4, dot_q3k_q8k_avx2_x4};
                    while s + 4 <= seq_len {
                        let inputs = [
                            &input_q8k[s * n_blocks..(s + 1) * n_blocks],
                            &input_q8k[(s + 1) * n_blocks..(s + 2) * n_blocks],
                            &input_q8k[(s + 2) * n_blocks..(s + 3) * n_blocks],
                            &input_q8k[(s + 3) * n_blocks..(s + 4) * n_blocks],
                        ];
                        let values = unsafe {
                            match quant {
                                QuantGemvType::Q2K => dot_q2k_q8k_avx2_x4(rb, inputs),
                                QuantGemvType::Q3K => dot_q3k_q8k_avx2_x4(rb, inputs),
                                _ => unreachable!(),
                            }
                        };
                        out[s..s + 4].copy_from_slice(&values);
                        s += 4;
                    }
                }
                for token in s..seq_len {
                    let q8k_slice = &input_q8k[token * n_blocks..(token + 1) * n_blocks];
                    out[token] = unsafe { dot_fn(rb, q8k_slice) };
                }
            });
        transpose_row_major_to_seq_major(&row_major, output, rows, seq_len);
    }
}

fn transpose_row_major_to_seq_major(
    row_major: &[f32],
    output: &mut [f32],
    rows: usize,
    seq_len: usize,
) {
    for row in 0..rows {
        for s in 0..seq_len {
            output[s * rows + row] = row_major[row * seq_len + s];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        avx2_eligible, dot_q4_0_row, dot_q8_0_row, dot_quantized_row, gemv_q4_0, gemv_q8_0,
        gemv_quantized, QuantGemvType,
    };
    use half::f16;

    #[test]
    fn iq_rows_use_stack_dequantization_without_losing_values() {
        let x = [1.0f32; 256];
        for (quant, block_bytes) in [
            (QuantGemvType::IQ2XXS, 66),
            (QuantGemvType::IQ2S, 82),
            (QuantGemvType::IQ3XXS, 98),
        ] {
            let mut row = vec![0u8; block_bytes];
            row[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
            assert_eq!(dot_quantized_row(&row, &x, 256, quant), 256.0);
        }
    }

    #[test]
    fn avx2_dispatch_accepts_q2k_q3k() {
        for quant in [
            QuantGemvType::Q2K,
            QuantGemvType::Q3K,
            QuantGemvType::Q4K,
            QuantGemvType::Q5K,
            QuantGemvType::Q6K,
        ] {
            assert!(avx2_eligible(quant, 256, true, true));
        }
        assert!(!avx2_eligible(QuantGemvType::Q2K, 255, true, true));
        assert!(!avx2_eligible(QuantGemvType::Q3K, 256, false, true));
        assert!(!avx2_eligible(QuantGemvType::Q3K, 256, true, false));
        assert!(!avx2_eligible(QuantGemvType::Q8_0, 256, true, true));

        #[cfg(target_arch = "x86_64")]
        {
            assert!(super::avx2_dot_fn(QuantGemvType::Q2K).is_some());
            assert!(super::avx2_dot_fn(QuantGemvType::Q3K).is_some());
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn packed_rows(quant: QuantGemvType, rows: usize, n_blocks: usize) -> Vec<u8> {
        let bytes_per_block = quant.block_bytes();
        let bytes_per_row = bytes_per_block * n_blocks;
        let mut bytes = vec![0u8; rows * bytes_per_row];
        for row in 0..rows {
            for block_idx in 0..n_blocks {
                let start = row * bytes_per_row + block_idx * bytes_per_block;
                let block = &mut bytes[start..start + bytes_per_block];
                match quant {
                    QuantGemvType::Q2K => {
                        for (i, scale) in block[..16].iter_mut().enumerate() {
                            let low = (1 + i * 7 + block_idx * 3 + row) & 0x0f;
                            let high = (15 + block_idx + row - i % 16) & 0x0f;
                            *scale = (low | (high << 4)) as u8;
                        }
                        for (i, q) in block[16..80].iter_mut().enumerate() {
                            *q = (i * 37 + block_idx * 19 + row * 11 + 7) as u8;
                        }
                        block[80..82].copy_from_slice(
                            &half::f16::from_f32(0.015625 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                        block[82..84].copy_from_slice(
                            &half::f16::from_f32(0.0078125 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                    QuantGemvType::Q3K => {
                        for (i, hmask) in block[..32].iter_mut().enumerate() {
                            *hmask = ((i * 29 + block_idx * 13 + row * 5) as u8) ^ 0xa5;
                        }
                        for (i, q) in block[32..96].iter_mut().enumerate() {
                            *q = (i * 41 + block_idx * 23 + row * 17 + 3) as u8;
                        }
                        for (i, scale) in block[96..108].iter_mut().enumerate() {
                            *scale = (i * 53 + block_idx * 17 + row * 7 + 3) as u8;
                        }
                        block[108..110].copy_from_slice(
                            &half::f16::from_f32(0.01171875 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        bytes
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q2q3_avx2_matrix_rows_and_seq_major() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }

        use crate::gemm::activation_q8::quantize_input_q8k;

        let rows = 65;
        let cols = 512;
        let seq_len = 2;
        let n_blocks = cols / 256;
        let input: Vec<f32> = (0..seq_len * cols)
            .map(|i| ((i * 97 % 251) as f32 - 125.0) * 0.03125)
            .collect();

        for quant in [QuantGemvType::Q2K, QuantGemvType::Q3K] {
            let bytes_per_row = quant.block_bytes() * n_blocks;
            let bytes = packed_rows(quant, rows, n_blocks);
            let mut output = vec![0.0f32; rows * seq_len];
            gemv_quantized(
                &bytes,
                &input,
                &mut output,
                rows,
                cols,
                seq_len,
                bytes_per_row,
                quant,
            );
            let dot = super::avx2_dot_fn(quant).expect("Q2/Q3 AVX2 mapping");
            for s in 0..seq_len {
                let q8k = quantize_input_q8k(&input[s * cols..(s + 1) * cols]);
                for row in 0..rows {
                    let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                    assert_eq!(output[s * rows + row], unsafe { dot(rb, &q8k) });
                }
            }
        }
    }

    #[test]
    fn q4_0_row_dot_matches_manual_decode() {
        let row = [
            0x00, 0x3c, 0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91, 0xa2, 0xb3,
            0xc4, 0xd5, 0xe6, 0xf7,
        ];
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.125 - 1.5).collect();
        let got = dot_q4_0_row(&row, &x, 1);

        let mut expected = 0.0f32;
        for i in 0..16 {
            expected += ((row[2 + i] & 0x0f) as f32 - 8.0) * x[i];
            expected += ((row[2 + i] >> 4) as f32 - 8.0) * x[i + 16];
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn q8_0_gemv_writes_seq_major_output() {
        let bytes = [
            0x00, 0x3c, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0x00, 0x3c, 32, 31, 30, 29, 28, 27, 26, 25,
            24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        ];
        let input = [1.0f32; 64];
        let mut output = [0.0f32; 4];

        gemv_q8_0(&bytes, &input, &mut output, 2, 32, 2, 34);

        assert_eq!(output[0], dot_q8_0_row(&bytes[..34], &input[..32], 1));
        assert_eq!(output[1], dot_q8_0_row(&bytes[34..], &input[..32], 1));
        assert_eq!(output[2], dot_q8_0_row(&bytes[..34], &input[32..], 1));
        assert_eq!(output[3], dot_q8_0_row(&bytes[34..], &input[32..], 1));
    }

    #[test]
    fn q4_0_gemv_smoke() {
        let row = [0u8; 18];
        let bytes = [row, row].concat();
        let input = [1.0f32; 32];
        let mut output = [1.0f32; 2];

        gemv_q4_0(&bytes, &input, &mut output, 2, 32, 1, 18);

        assert_eq!(output, [0.0, 0.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_kquant_q8k_gemv_defaults_by_platform_with_env_override() {
        let key = "RNB_AARCH64_KQUANT_Q8K_GEMV";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert_eq!(
            super::aarch64_kquant_q8k_gemv_enabled_from_env(),
            cfg!(target_vendor = "apple")
        );

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !super::aarch64_kquant_q8k_gemv_enabled_from_env(),
                "{value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(super::aarch64_kquant_q8k_gemv_enabled_from_env());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_pair_gemv_matches_two_single_gemvs() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let rows = 3;
        let cols = 256;
        let bpr = 144;
        let mut gate = vec![0u8; rows * bpr];
        let mut up = vec![0u8; rows * bpr];
        for row in 0..rows {
            let gate_row = &mut gate[row * bpr..(row + 1) * bpr];
            let up_row = &mut up[row * bpr..(row + 1) * bpr];
            gate_row[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            up_row[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            for i in 4..16 {
                gate_row[i] = 1 + ((row + i) % 31) as u8;
                up_row[i] = 3 + ((row + i * 2) % 29) as u8;
            }
            for i in 16..bpr {
                gate_row[i] = (row as u8).wrapping_mul(17).wrapping_add(i as u8);
                up_row[i] = (row as u8).wrapping_mul(23).wrapping_add((i * 3) as u8);
            }
        }
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.013).sin() * 2.0) - 0.5)
            .collect();
        let mut expected_gate = vec![0.0f32; rows];
        let mut expected_up = vec![0.0f32; rows];
        let mut got_gate = vec![0.0f32; rows];
        let mut got_up = vec![0.0f32; rows];

        super::gemv_quantized_aarch64_q8k(
            &gate,
            &input,
            &mut expected_gate,
            rows,
            cols,
            1,
            bpr,
            super::QuantGemvType::Q4K,
        );
        super::gemv_quantized_aarch64_q8k(
            &up,
            &input,
            &mut expected_up,
            rows,
            cols,
            1,
            bpr,
            super::QuantGemvType::Q4K,
        );
        let prev = std::env::var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW").ok();
        std::env::set_var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW", "1");
        assert!(super::gemv_q4k_pair_aarch64_q8k(
            &gate,
            &up,
            &input,
            &mut got_gate,
            &mut got_up,
            rows,
            cols,
            bpr,
            bpr,
        ));

        assert_eq!(got_gate, expected_gate);
        assert_eq!(got_up, expected_up);
        let input_q8k = crate::gemm::activation_q8::quantize_input_q8k(&input);
        got_gate.fill(0.0);
        got_up.fill(0.0);
        assert!(super::gemv_q4k_pair_aarch64_q8k_prequantized_serial(
            &gate,
            &up,
            &input_q8k,
            &mut got_gate,
            &mut got_up,
            rows,
            cols,
            bpr,
            bpr,
        ));
        assert_eq!(got_gate, expected_gate);
        assert_eq!(got_up, expected_up);
        std::env::set_var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW", "0");
        got_gate.fill(0.0);
        got_up.fill(0.0);
        assert!(super::gemv_q4k_pair_aarch64_q8k(
            &gate,
            &up,
            &input,
            &mut got_gate,
            &mut got_up,
            rows,
            cols,
            bpr,
            bpr,
        ));
        match prev {
            Some(value) => std::env::set_var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW", value),
            None => std::env::remove_var("RNB_QWEN35_MOE_DECODE_PAIR_FUSED_RAW"),
        }

        assert_eq!(got_gate, expected_gate);
        assert_eq!(got_up, expected_up);
    }
}
