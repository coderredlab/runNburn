use crate::engine::quantized_weight_types::QuantizedWeight;

#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;
#[cfg(feature = "vulkan")]
use rnb_loader::GGMLType;

pub(in crate::engine) use crate::runtime::DecodeWeightKind as DecodeProjectionKind;

#[cfg(feature = "vulkan")]
fn gpu_decode_weight<'a>(
    kind: DecodeProjectionKind,
    weight: &'a QuantizedWeight,
    rows: usize,
    ggml_type: GGMLType,
) -> gpu::DecodeGemvWeight<'a> {
    gpu::DecodeGemvWeight {
        kind,
        raw: weight.data.as_bytes().unwrap_or(&[]),
        rows,
        cols: weight.cols,
        ggml_type,
    }
}

mod attention;
mod ffn;
mod projection;

pub(in crate::engine) use attention::try_decode_attention_single_head_if_supported;
#[cfg(feature = "vulkan")]
pub(in crate::engine) use ffn::wait_decode_async_if_supported;
pub(in crate::engine) use ffn::{
    try_decode_ffn_chain_if_supported, try_decode_ffn_gate_async_if_supported,
};
pub(in crate::engine) use projection::{
    try_decode_attention_qkv_if_supported, try_decode_gdn_alpha_beta_if_supported,
    try_decode_gdn_qkv_gate_if_supported, try_decode_gemv_if_supported,
};
