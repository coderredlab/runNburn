//! AArch64 quantized GEMV dispatch paths.

use super::*;
use crate::engine::policy;

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q80_pair_packed_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> bool {
    if weight.ggml_type != GGMLType::Q8_0 || seq_len != 1 {
        return false;
    }
    if !q80_pair_i8mm_supported(weight.cols) {
        return false;
    }

    if bytes_per_row != (weight.cols / 32) * 34 {
        return false;
    }
    let Some(packed) = weight
        .arch
        .q80_pair_packed
        .get_or_init(|| Some(pack_q8_0_row_pairs(bytes, weight.rows, bytes_per_row)))
        .as_ref()
    else {
        return false;
    };
    gemv_q8_0_packed_i8mm(packed, q8, output, weight.rows, weight.cols);
    true
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q80_f32_scale_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> bool {
    let Some(scales) = q80_f32_scales(weight, bytes, bytes_per_row) else {
        return false;
    };
    gemv_q8_0_int8_f32_scales(
        bytes,
        scales,
        q8,
        output,
        weight.rows,
        weight.cols,
        seq_len,
        bytes_per_row,
    );
    true
}

#[cfg(target_arch = "aarch64")]
fn q80_f32_scales<'a>(
    weight: &'a QuantizedWeight,
    bytes: &[u8],
    bytes_per_row: usize,
) -> Option<&'a [f32]> {
    if weight.ggml_type != GGMLType::Q8_0 || bytes_per_row != (weight.cols / 32) * 34 {
        return None;
    }
    weight
        .arch
        .q80_f32_scales
        .get_or_init(|| {
            if !q80_f32_scales_requested() {
                return None;
            }
            let n_blocks = weight.cols / 32;
            let mut scales = vec![0.0f32; weight.rows * n_blocks];
            for row in 0..weight.rows {
                let row_off = row * bytes_per_row;
                for bi in 0..n_blocks {
                    let off = row_off + bi * 34;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    scales[row * n_blocks + bi] = half::f16::from_bits(bits).to_f32();
                }
            }
            Some(scales)
        })
        .as_deref()
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_q8k_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<()> {
    if weight.ggml_type == GGMLType::Q4_K {
        let q4k_backend = q4k_kernel_backend_from_env(policy::q4k_kernel_backend().as_deref());
        if let Some(backend) = q4k_backend.filter(|_| has_full_raw_quantized_bytes(weight)) {
            dispatch_q4k_kernel_backend(
                backend,
                bytes,
                weight.arch.repacked.as_ref(),
                q8k,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            return Ok(());
        }
    }

    match weight.ggml_type {
        GGMLType::Q2_K
        | GGMLType::Q3_K
        | GGMLType::IQ2_XXS
        | GGMLType::IQ2_S
        | GGMLType::IQ3_XXS
        | GGMLType::IQ4_XS => {
            let quant = match weight.ggml_type {
                GGMLType::Q2_K => gemm_runtime::quant_gemv::QuantGemvType::Q2K,
                GGMLType::Q3_K => gemm_runtime::quant_gemv::QuantGemvType::Q3K,
                GGMLType::IQ2_XXS => gemm_runtime::quant_gemv::QuantGemvType::IQ2XXS,
                GGMLType::IQ2_S => gemm_runtime::quant_gemv::QuantGemvType::IQ2S,
                GGMLType::IQ3_XXS => gemm_runtime::quant_gemv::QuantGemvType::IQ3XXS,
                GGMLType::IQ4_XS => gemm_runtime::quant_gemv::QuantGemvType::IQ4XS,
                _ => unreachable!("Q8K general quant expected"),
            };
            gemm_runtime::quant_gemv::gemv_aarch64_q8k_prequantized(
                bytes,
                q8k,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
                quant,
            );
        }
        GGMLType::Q4_K => {
            if let Some(ref rpk) = weight.arch.repacked {
                gemm_runtime::neon_repacked::gemv_q4k_repacked(
                    rpk,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                    bytes,
                    bytes_per_row,
                );
            } else {
                gemv_q4_k_int8(
                    bytes,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                    bytes_per_row,
                );
            }
        }
        GGMLType::Q5_K => gemv_q5_k_int8(
            bytes,
            q8k,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            bytes_per_row,
        ),
        GGMLType::Q6_K => gemv_q6_k_int8(
            bytes,
            q8k,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            bytes_per_row,
        ),
        _ => {
            return Err(crate::error::LlmError::Forward(format!(
                "dispatch_q8k_gemv: unsupported type {:?}",
                weight.ggml_type
            )))
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_q8_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<()> {
    if let Some(ref q4_0) = weight.q4_0_data {
        let q4_0_bytes_per_row = (weight.cols / 32) * 18;
        gemv_q4_0_int8(
            q4_0,
            q8,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            q4_0_bytes_per_row,
        );
        return Ok(());
    }

    match weight.ggml_type {
        GGMLType::Q4_1 | GGMLType::Q5_1 | GGMLType::Q8_1 => {
            let quant = match weight.ggml_type {
                GGMLType::Q4_1 => gemm_runtime::quant_gemv::QuantGemvType::Q4_1,
                GGMLType::Q5_1 => gemm_runtime::quant_gemv::QuantGemvType::Q5_1,
                GGMLType::Q8_1 => gemm_runtime::quant_gemv::QuantGemvType::Q8_1,
                _ => unreachable!("Q8 general quant expected"),
            };
            gemm_runtime::quant_gemv::gemv_aarch64_q8_prequantized(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
                quant,
            );
            Ok(())
        }
        GGMLType::Q4_0 => {
            gemv_q4_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        GGMLType::Q5_0 => {
            gemv_q5_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        GGMLType::Q8_0 => {
            if dispatch_q80_pair_packed_gemv(weight, bytes, q8, output, seq_len, bytes_per_row) {
                return Ok(());
            }
            if dispatch_q80_f32_scale_gemv(weight, bytes, q8, output, seq_len, bytes_per_row) {
                return Ok(());
            }
            gemv_q8_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        _ => Err(crate::error::LlmError::Forward(format!(
            "dispatch_q8_gemv: unsupported type {:?}",
            weight.ggml_type
        ))),
    }
}

/// Raw-bytes Q8_0 GEMV (fast int8 dotprod) for callers that hold `&[u8]` weight
/// without a `QuantizedWeight` (e.g. shared-expert view). Quantizes the f32 input
/// to Q8 blocks then runs `gemv_q8_0_int8`. `cols % 32 == 0` required.
#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn gemv_q8_0_raw(
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    bytes_per_row: usize,
) {
    let q8 = quantize_input_q8(input);
    gemv_q8_0_int8(bytes, &q8, output, rows, cols, 1, bytes_per_row);
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_vec_fast_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if weight.ggml_type == GGMLType::Q4_K {
        let q4k_backend = q4k_kernel_backend_from_env(policy::q4k_kernel_backend().as_deref());
        if let Some(backend) = q4k_backend.filter(|_| has_full_raw_quantized_bytes(weight)) {
            let q8k = quantize_input_q8k(input);
            dispatch_q4k_kernel_backend(
                backend,
                bytes,
                weight.arch.repacked.as_ref(),
                &q8k,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            return Ok(true);
        }
    }

    if weight.q4_0_data.is_some() && fast_dotprod_enabled() {
        let q8 = quantize_input_q8(input);
        dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
        return Ok(true);
    }

    dispatch_dotprod_f32_input_gemv(weight, bytes, input, output, seq_len, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_into_fast_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if weight.q4_0_data.is_some() && fast_dotprod_enabled() {
        let q8 = quantize_input_q8(input);
        dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
        return Ok(true);
    }

    dispatch_dotprod_f32_input_gemv(weight, bytes, input, output, seq_len, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_dotprod_f32_input_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if !fast_dotprod_enabled() {
        return Ok(false);
    }

    match weight.ggml_type {
        GGMLType::Q2_K
        | GGMLType::Q3_K
        | GGMLType::Q4_K
        | GGMLType::Q5_K
        | GGMLType::Q6_K
        | GGMLType::IQ2_XXS
        | GGMLType::IQ2_S
        | GGMLType::IQ3_XXS
        | GGMLType::IQ4_XS => {
            let q8k = quantize_input_q8k(input);
            dispatch_q8k_gemv(weight, bytes, &q8k, output, seq_len, bytes_per_row)?;
            Ok(true)
        }
        GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q5_0
        | GGMLType::Q5_1
        | GGMLType::Q8_0
        | GGMLType::Q8_1 => {
            let q8 = quantize_input_q8(input);
            dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn has_full_raw_quantized_bytes(weight: &QuantizedWeight) -> bool {
    weight.data.as_bytes().is_some_and(|bytes| {
        expected_quantized_byte_len(weight.ggml_type, weight.rows, weight.cols) == Some(bytes.len())
    })
}
#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q4k_kernel_backend(
    backend: Q4KKernelBackend,
    bytes: &[u8],
    repacked: Option<&memmap2::Mmap>,
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    match backend {
        Q4KKernelBackend::Builtin => {
            if seq_len > 1 {
                if let Some(rpk) = repacked {
                    gemm_runtime::neon_repacked::gemm_q4k_repacked(
                        rpk,
                        q8k,
                        output,
                        rows,
                        cols,
                        seq_len,
                        bytes,
                        bytes_per_row,
                    );
                } else {
                    gemv_q4_k_int8(bytes, q8k, output, rows, cols, seq_len, bytes_per_row);
                }
            } else {
                gemv_q4_k_int8(bytes, q8k, output, rows, cols, seq_len, bytes_per_row);
            }
        }
    }
}
