//! LLM-facing dequantization adapters.

use rnb_loader::GGMLType;

use super::gemm_runtime::dequant::{self, DequantType};

#[inline]
fn dequant_type(ggml_type: GGMLType) -> DequantType {
    match ggml_type {
        GGMLType::F32 => DequantType::F32,
        GGMLType::F16 => DequantType::F16,
        GGMLType::BF16 => DequantType::BF16,
        GGMLType::Q4_0 => DequantType::Q4_0,
        GGMLType::Q4_1 => DequantType::Q4_1,
        GGMLType::Q5_0 => DequantType::Q5_0,
        GGMLType::Q5_1 => DequantType::Q5_1,
        GGMLType::Q8_0 => DequantType::Q8_0,
        GGMLType::Q8_1 => DequantType::Q8_1,
        GGMLType::Q2_K => DequantType::Q2K,
        GGMLType::Q3_K => DequantType::Q3K,
        GGMLType::Q4_K => DequantType::Q4K,
        GGMLType::Q5_K => DequantType::Q5K,
        GGMLType::Q6_K => DequantType::Q6K,
        GGMLType::Q8_K => DequantType::Q8K,
        GGMLType::IQ2_XXS => DequantType::IQ2XXS,
        GGMLType::IQ2_XS => DequantType::IQ2XS,
        GGMLType::IQ1_S => DequantType::IQ1S,
        GGMLType::IQ4_NL => DequantType::IQ4NL,
        GGMLType::IQ3_S => DequantType::IQ3S,
        GGMLType::IQ3_XXS => DequantType::IQ3XXS,
        GGMLType::IQ2_S => DequantType::IQ2S,
        GGMLType::IQ4_XS => DequantType::IQ4XS,
        GGMLType::IQ1_M => DequantType::IQ1M,
        GGMLType::TQ1_0 => DequantType::TQ1_0,
        GGMLType::TQ2_0 => DequantType::TQ2_0,
        GGMLType::MXFP4 => DequantType::MXFP4,
        GGMLType::NVFP4 => DequantType::NVFP4,
        GGMLType::Q1_0 => DequantType::Q1_0,
        GGMLType::Q2_0 => DequantType::Q2_0,
        GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 | GGMLType::F64 => {
            panic!("{ggml_type:?} GGUF tensors cannot be dequantized as model weights")
        }
    }
}

pub(super) fn dequantize_bytes_to_f32(bytes: &[u8], ggml_type: GGMLType) -> Vec<f32> {
    dequant::dequantize_bytes_to_f32(bytes, dequant_type(ggml_type))
}

#[cfg(test)]
pub(super) fn dequantize_row_to_slice_if_supported(
    bytes: &[u8],
    ggml_type: GGMLType,
    output: &mut [f32],
) -> bool {
    dequant::dequantize_row_to_slice_if_supported(bytes, dequant_type(ggml_type), output)
}
