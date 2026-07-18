//! LLM-facing quantized GEMV adapters.
//!
//! The kernels live in `rnb-cpu::gemm`; this module only maps loader quant identity
//! to runtime GEMV quant identity and keeps the legacy full-row dequant fallback
//! for unsupported GGML types.

use rayon::prelude::*;
use rnb_loader::GGMLType;

use super::dequant::dequantize_bytes_to_f32;
use super::gemm_runtime::quant_gemv::{self, QuantGemvType};

#[inline]
fn quant_gemv_type(ggml_type: GGMLType) -> Option<QuantGemvType> {
    match ggml_type {
        GGMLType::Q4_0 => Some(QuantGemvType::Q4_0),
        GGMLType::Q4_1 => Some(QuantGemvType::Q4_1),
        GGMLType::Q5_0 => Some(QuantGemvType::Q5_0),
        GGMLType::Q5_1 => Some(QuantGemvType::Q5_1),
        GGMLType::Q8_0 => Some(QuantGemvType::Q8_0),
        GGMLType::Q2_K => Some(QuantGemvType::Q2K),
        GGMLType::Q3_K => Some(QuantGemvType::Q3K),
        GGMLType::Q4_K => Some(QuantGemvType::Q4K),
        GGMLType::Q5_K => Some(QuantGemvType::Q5K),
        GGMLType::Q6_K => Some(QuantGemvType::Q6K),
        GGMLType::IQ2_XXS => Some(QuantGemvType::IQ2XXS),
        GGMLType::IQ2_S => Some(QuantGemvType::IQ2S),
        GGMLType::IQ3_XXS => Some(QuantGemvType::IQ3XXS),
        _ => None,
    }
}

pub(super) fn gemv_q4_0(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    quant_gemv::gemv_q4_0(bytes, x_data, output, rows, cols, seq_len, bytes_per_row);
}

pub(super) fn gemv_q8_0(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    quant_gemv::gemv_q8_0(bytes, x_data, output, rows, cols, seq_len, bytes_per_row);
}

#[inline]
pub(super) fn dot_k_block_row(
    row_bytes: &[u8],
    x: &[f32],
    cols: usize,
    _bytes_per_row: usize,
    ggml_type: GGMLType,
) -> f32 {
    if let Some(quant) = quant_gemv_type(ggml_type) {
        return quant_gemv::dot_quantized_row(row_bytes, x, cols, quant);
    }

    let row_f32 = dequantize_bytes_to_f32(row_bytes, ggml_type);
    let mut acc = 0.0f32;
    for i in 0..cols {
        acc += row_f32[i] * x[i];
    }
    acc
}

/// Output-projection GEMV with `f64` accumulator.
///
/// Used only for the final `hidden → vocab` projection in decode (and only
/// when `RNB_OUTPUT_F64_LOGIT=1` is set). The standard `gemv_generic` /
/// `dot_quantized_row` path keeps its f32 accumulator; that is fine for
/// hidden-state propagation but lossy at the output stage where ranking
/// margins between top-1 and top-2 logits can be 0.1-0.5 — small enough
/// that f32 accumulation drift across hidden_dim can flip the argmax.
///
/// Cost: each call dequantizes the entire output table (vocab × hidden).
/// This is one call per decode step, so it's bounded — the goal here is
/// correctness recovery, not throughput.
pub(super) fn gemv_output_f64_logit(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    bytes_per_row: usize,
    ggml_type: GGMLType,
) {
    output[..rows]
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_idx, out)| {
            let rb = &bytes[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
            let row_f32 = dequantize_bytes_to_f32(rb, ggml_type);
            let mut acc: f64 = 0.0;
            for i in 0..cols {
                acc += row_f32[i] as f64 * x_data[i] as f64;
            }
            *out = acc as f32;
        });
}

/// mt94 axis — force per-row `dequantize_bytes_to_f32` + pure f32×f32 dot,
/// bypassing the production Q8K-quantized integer kernel for the entire
/// row reduction. Used by Q/K projection mixed-precision ablation only;
/// other operators keep production dispatch.
///
/// Output layout matches `gemv_quantized` (seq-major, `output[s * rows + row]`).
/// For `seq_len = 1` this reduces to the same shape as `gemv_output_f64_logit`
/// but with f32 accumulator (so the only axis changed vs production is the
/// kernel reduction precision, not the activation type).
pub(super) fn gemv_full_dequant_f32(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    ggml_type: GGMLType,
) {
    if seq_len == 1 {
        output[..rows]
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let rb = &bytes[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
                let row_f32 = dequantize_bytes_to_f32(rb, ggml_type);
                let mut acc: f32 = 0.0;
                for i in 0..cols {
                    acc += row_f32[i] * x_data[i];
                }
                *out = acc;
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row_idx, out)| {
                let rb = &bytes[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
                let row_f32 = dequantize_bytes_to_f32(rb, ggml_type);
                for s in 0..seq_len {
                    let x = &x_data[s * cols..(s + 1) * cols];
                    let mut acc: f32 = 0.0;
                    for i in 0..cols {
                        acc += row_f32[i] * x[i];
                    }
                    out[s] = acc;
                }
            });
        for row in 0..rows {
            for s in 0..seq_len {
                output[s * rows + row] = row_major[row * seq_len + s];
            }
        }
    }
}

/// Generic fallback gemv. Supported quantized rows dispatch to `rnb-cpu::gemm`;
/// unsupported GGML types keep the old full-row dequant path here.
pub(super) fn gemv_generic(
    bytes: &[u8],
    x_data: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    ggml_type: GGMLType,
) {
    if let Some(quant) = quant_gemv_type(ggml_type) {
        quant_gemv::gemv_quantized(
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
                    out[i] = dot_k_block_row(rb, x_data, cols, bytes_per_row, ggml_type);
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
                    out[s] =
                        dot_k_block_row(rb, &x_data[s * cols..], cols, bytes_per_row, ggml_type);
                }
            });
        for row in 0..rows {
            for s in 0..seq_len {
                output[s * rows + row] = row_major[row * seq_len + s];
            }
        }
    }
}
