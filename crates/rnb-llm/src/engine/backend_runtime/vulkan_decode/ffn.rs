//! Vulkan decode FFN chain and async FFN helpers.

use super::*;

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_ffn_chain(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    norm_eps: f32,
    hidden_dim: usize,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
) -> Result<bool, String> {
    gpu::decode_ffn_chain(
        runtime,
        layer_idx,
        hidden,
        norm_weight,
        norm_eps,
        hidden_dim,
        gate_weight.data.as_bytes().unwrap_or(&[]),
        gate_weight.rows,
        gate_weight.cols,
        gate_weight.ggml_type,
        up_weight.data.as_bytes().unwrap_or(&[]),
        up_weight.rows,
        up_weight.cols,
        up_weight.ggml_type,
        down_weight.data.as_bytes().unwrap_or(&[]),
        down_weight.rows,
        down_weight.cols,
        down_weight.ggml_type,
    )
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_ffn_chain_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    norm_eps: f32,
    hidden_dim: usize,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_ffn_chain(
            runtime,
            layer_idx,
            hidden,
            norm_weight,
            norm_eps,
            hidden_dim,
            gate_weight,
            up_weight,
            down_weight,
        ) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] FFN chain failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn try_decode_ffn_gate_async(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    norm_data: &[f32],
    gate_weight: &QuantizedWeight,
    gate_rows: usize,
) -> Result<bool, String> {
    let weight = gpu_decode_weight(
        DecodeProjectionKind::FfnGate,
        gate_weight,
        gate_rows,
        gate_weight.ggml_type,
    );
    gpu::decode_gemv_multi_async(runtime, layer_idx, norm_data, &[weight])
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn try_decode_ffn_gate_async_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    norm_data: &[f32],
    gate_weight: &QuantizedWeight,
    gate_rows: usize,
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_ffn_gate_async(runtime, layer_idx, norm_data, gate_weight, gate_rows) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] FFN gate async failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn wait_decode_async(
    runtime: &mut gpu::Runtime,
    outputs: &mut [&mut [f32]],
) -> Result<(), String> {
    gpu::decode_wait_async(runtime, outputs)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn wait_decode_async_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    outputs: &mut [&mut [f32]],
    label: &str,
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match wait_decode_async(runtime, outputs) {
            Ok(()) => return true,
            Err(err) => {
                eprintln!("[gpu] {label} wait failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}
