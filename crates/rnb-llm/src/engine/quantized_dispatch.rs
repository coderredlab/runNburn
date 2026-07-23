#[cfg(any(not(feature = "cuda"), target_arch = "aarch64", test))]
use super::gemm_runtime;
#[cfg(target_arch = "aarch64")]
use super::gemm_runtime::neon_dot::{
    gemv_q4_0_int8, gemv_q4_k_int8, gemv_q4_k_int8_dual, gemv_q5_0_int8, gemv_q5_k_int8,
    gemv_q6_k_int8, gemv_q8_0_int8,
};
#[cfg(target_arch = "aarch64")]
use super::gemm_runtime::neon_dot::{
    gemv_q8_0_int8_f32_scales, gemv_q8_0_packed_i8mm, pack_q8_0_row_pairs,
};
#[cfg(any(target_arch = "aarch64", test))]
pub(super) use super::gemm_runtime::policy::Q4KKernelBackend;
#[cfg(target_arch = "aarch64")]
use super::gemm_runtime::{quantize_input_q8, quantize_input_q8k, Q8Block, Q8KBlock};
#[cfg(target_arch = "aarch64")]
pub(super) use super::gemm_runtime::{Q8Block as QuantizedQ8Block, Q8KBlock as QuantizedQ8KBlock};
#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use super::quantized_packing::quantize_q8_for_test;
#[cfg(target_arch = "aarch64")]
pub(super) use super::quantized_packing::{
    build_q80_f32_scales, pack_q80_row_pairs, q80_prepack_load_enabled, repack_q4k_artifacts,
};
use super::quantized_weight_types::QuantizedWeight;
use super::types::ScratchBuffers;
use crate::engine::norm::apply_model_gate_mul_inplace;
use rnb_loader::Architecture as ModelArchitecture;
use rnb_loader::GGMLType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
mod aarch64_gemv;
mod decode;
mod prefill;
#[cfg(target_arch = "aarch64")]
pub(super) use aarch64_gemv::{
    dispatch_into_fast_gemv, dispatch_q8_gemv, dispatch_q8k_gemv, dispatch_vec_fast_gemv,
    gemv_q8_0_raw,
};
#[cfg(feature = "vulkan")]
pub(super) use decode::decode_ffn_up_cpu_best_effort;
pub(super) use decode::{
    decode_attention_qkv_cpu_into, decode_ffn_gate_up_cpu_into, decode_gdn_qkv_gate_cpu_into,
};
pub(super) use prefill::{
    prefill_dual_gemv_q8_or_f32, prefill_gate_up_vectors, prefill_gemv_vec,
    prefill_quantized_input_for_weight, prefill_raw_quantized_batch,
};
#[cfg(target_arch = "aarch64")]
pub(super) use prefill::{prefill_raw_dual_q4k_q8k, prefill_raw_split_q4k_q8k, quantize_raw_q8k};

#[cfg(target_arch = "aarch64")]
#[derive(Clone)]
pub(super) struct ArchScratchBuffers {
    pub(super) ffn_combined: Vec<f32>,
    pub(super) q8_scratch: Vec<Q8Block>,
    pub(super) q8k_scratch: Vec<Q8KBlock>,
}

#[cfg(not(target_arch = "aarch64"))]
#[derive(Clone)]
pub(super) struct ArchScratchBuffers;

impl ArchScratchBuffers {
    pub(super) fn new(hidden_dim: usize, ffn_inner_dim: usize) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                ffn_combined: vec![0.0; ffn_inner_dim * 2],
                q8_scratch: vec![Q8Block::default(); hidden_dim / 32],
                q8k_scratch: vec![Q8KBlock::default(); hidden_dim / 256],
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = (hidden_dim, ffn_inner_dim);
            Self
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub(super) fn force_generic_gemv(rows: usize, cols: usize) -> bool {
    gemm_runtime::policy::force_generic_gemv_for_shape(rows, cols)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn aarch64_dotprod_available() -> bool {
    gemm_runtime::policy::aarch64_dotprod_available()
}
#[cfg(not(feature = "cuda"))]
pub(super) fn dispatch_f32_gemv(
    weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    gemm_runtime::f32_gemv::gemv_f32(weights, input, output, rows, cols, batch);
}

#[cfg(target_arch = "aarch64")]
pub(super) fn quantize_q8k_input_into(input: &[f32], output: &mut [QuantizedQ8KBlock]) {
    gemm_runtime::quantize_input_q8k_into(input, output);
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q6k_q8k_gemv(
    weights: &[u8],
    input: &[QuantizedQ8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    batch: usize,
    bytes_per_row: usize,
) {
    gemm_runtime::neon_dot::gemv_q6_k_int8(
        weights,
        input,
        output,
        rows,
        cols,
        batch,
        bytes_per_row,
    );
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dot_q4k_q8k(weight_row: &[u8], input: &[QuantizedQ8KBlock], blocks: usize) -> f32 {
    unsafe { gemm_runtime::neon_dot::dot_q4_k_q8k_neon(weight_row, input, blocks) }
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn dot_q6k_q8k(weight_row: &[u8], input: &[QuantizedQ8KBlock], blocks: usize) -> f32 {
    unsafe { gemm_runtime::neon_dot::dot_q6_k_q8k_neon(weight_row, input, blocks) }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_q4k_pair_q8k_prequantized(
    gate: &[u8],
    up: &[u8],
    input: &[QuantizedQ8KBlock],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
    serial: bool,
) -> bool {
    if serial {
        gemm_runtime::quant_gemv::gemv_q4k_pair_aarch64_q8k_prequantized_serial(
            gate,
            up,
            input,
            gate_output,
            up_output,
            rows,
            cols,
            gate_bytes_per_row,
            up_bytes_per_row,
        )
    } else {
        gemm_runtime::quant_gemv::gemv_q4k_pair_aarch64_q8k_prequantized(
            gate,
            up,
            input,
            gate_output,
            up_output,
            rows,
            cols,
            gate_bytes_per_row,
            up_bytes_per_row,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn dispatch_q4k_pair_q8k(
    gate: &[u8],
    up: &[u8],
    input: &[f32],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows: usize,
    cols: usize,
    gate_bytes_per_row: usize,
    up_bytes_per_row: usize,
) -> bool {
    gemm_runtime::quant_gemv::gemv_q4k_pair_aarch64_q8k(
        gate,
        up,
        input,
        gate_output,
        up_output,
        rows,
        cols,
        gate_bytes_per_row,
        up_bytes_per_row,
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) fn q80_pair_i8mm_supported(cols: usize) -> bool {
    gemm_runtime::policy::q80_pair_i8mm_supported(cols)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn q80_f32_scales_requested() -> bool {
    gemm_runtime::policy::q80_f32_scales_requested()
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn expected_quantized_byte_len(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
) -> Option<usize> {
    match ggml_type {
        GGMLType::F32 => Some(rows * cols * 4),
        GGMLType::F16 | GGMLType::BF16 => Some(rows * cols * 2),
        GGMLType::Q4_0 if cols % 32 == 0 => Some(rows * (cols / 32) * 18),
        GGMLType::Q4_1 if cols % 32 == 0 => Some(rows * (cols / 32) * 20),
        GGMLType::Q5_0 if cols % 32 == 0 => Some(rows * (cols / 32) * 22),
        GGMLType::Q5_1 if cols % 32 == 0 => Some(rows * (cols / 32) * 24),
        GGMLType::Q8_0 if cols % 32 == 0 => Some(rows * (cols / 32) * 34),
        GGMLType::Q8_1 if cols % 32 == 0 => Some(rows * (cols / 32) * 36),
        GGMLType::Q2_K if cols % 256 == 0 => Some(rows * (cols / 256) * 84),
        GGMLType::Q3_K if cols % 256 == 0 => Some(rows * (cols / 256) * 110),
        GGMLType::Q4_K if cols % 256 == 0 => Some(rows * (cols / 256) * 144),
        GGMLType::Q5_K if cols % 256 == 0 => Some(rows * (cols / 256) * 176),
        GGMLType::Q6_K if cols % 256 == 0 => Some(rows * (cols / 256) * 210),
        GGMLType::IQ2_XXS if cols % 256 == 0 => Some(rows * (cols / 256) * 66),
        GGMLType::IQ2_S if cols % 256 == 0 => Some(rows * (cols / 256) * 82),
        GGMLType::IQ3_XXS if cols % 256 == 0 => Some(rows * (cols / 256) * 98),
        GGMLType::IQ4_XS if cols % 256 == 0 => Some(rows * (cols / 256) * 136),
        _ => None,
    }
}

#[cfg(any(target_arch = "aarch64", test))]
pub(super) fn q4k_kernel_backend_from_env(explicit: Option<&str>) -> Option<Q4KKernelBackend> {
    gemm_runtime::policy::q4k_kernel_backend_from_env(explicit)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn fast_dotprod_enabled() -> bool {
    gemm_runtime::policy::fast_dotprod_enabled()
}

#[cfg(target_arch = "aarch64")]
fn k_quant_q8k_candidate(weight: &QuantizedWeight) -> bool {
    weight.q4_0_data.is_none()
        && weight.cols % 256 == 0
        && matches!(
            weight.ggml_type,
            GGMLType::Q3_K
                | GGMLType::Q4_K
                | GGMLType::Q5_K
                | GGMLType::Q6_K
                | GGMLType::IQ2_XXS
                | GGMLType::IQ2_S
                | GGMLType::IQ3_XXS
                | GGMLType::IQ4_XS
        )
}

#[cfg(target_arch = "aarch64")]
fn decode_q8k_candidate(
    weight: &QuantizedWeight,
    input_len: usize,
    require_fast_gemv: bool,
) -> bool {
    let dotprod = if require_fast_gemv {
        fast_dotprod_enabled()
    } else {
        aarch64_dotprod_available()
    };
    dotprod
        && input_len % 256 == 0
        && matches!(
            weight.ggml_type,
            GGMLType::Q3_K
                | GGMLType::Q4_K
                | GGMLType::Q5_K
                | GGMLType::Q6_K
                | GGMLType::IQ2_XXS
                | GGMLType::IQ2_S
                | GGMLType::IQ3_XXS
                | GGMLType::IQ4_XS
        )
}
