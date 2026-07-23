use crate::engine::quantized_weight_types::QuantizedWeight;

#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn gdn_qkv_conv_prefill_enabled() -> bool {
    crate::engine::policy::env_os_string("RNB_VULKAN_GDN_QKV_CONV_PREFILL").is_some()
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_gdn_qkv_conv_prefill_window(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    qkv_weight: &QuantizedWeight,
    kernel: &[f32],
    input_all: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    conv_channels: usize,
    kernel_size: usize,
    conv_state: &mut [f32],
    defer_state_materialization: bool,
    profiling: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    if seq_len <= 1 {
        return Ok(None);
    }
    if !gdn_qkv_conv_prefill_enabled() {
        return Ok(None);
    }
    if profiling && layer_idx == 0 {
        eprintln!(
            "  [GDN L0] qkv_gpu_quant      {:?} (ggml={:?})",
            gpu::layer_quant_debug(qkv_weight.ggml_type),
            qkv_weight.ggml_type
        );
    }
    if !gpu::layer_quant_supported(qkv_weight.ggml_type) {
        return Ok(None);
    }

    let qkv_raw = qkv_weight.data.as_bytes().unwrap_or(&[]);
    let mut conv_out = vec![0.0f32; seq_len * conv_channels];
    if let Err(e) = gpu::write_gdn_conv_state_f32(runtime, layer_idx, conv_state) {
        eprintln!("[gpu] GDN conv_state seed failed, CPU fallback: {}", e);
        return Ok(None);
    }

    let window_chunk = gpu::gdn_qkv_prefill_window_chunk(hidden_dim);
    for chunk_start in (0..seq_len).step_by(window_chunk) {
        let chunk_len = (seq_len - chunk_start).min(window_chunk);
        let input_chunk =
            &input_all[chunk_start * hidden_dim..(chunk_start + chunk_len) * hidden_dim];
        let out_chunk =
            &mut conv_out[chunk_start * conv_channels..(chunk_start + chunk_len) * conv_channels];
        match gpu::gdn_qkv_conv_window_from_resident_state(
            runtime,
            layer_idx,
            qkv_weight.ggml_type,
            qkv_raw,
            conv_channels,
            qkv_weight.cols,
            kernel,
            input_chunk,
            chunk_len,
            hidden_dim,
            conv_channels,
            kernel_size,
            out_chunk,
        ) {
            Ok(true) => {}
            Ok(false) => return Ok(None),
            Err(e) => {
                eprintln!("[gpu] GDN qkv->conv window failed, CPU fallback: {}", e);
                return Ok(None);
            }
        }
    }

    if !defer_state_materialization {
        if let Err(e) = gpu::materialize_gdn_conv_state_f32(runtime, layer_idx, conv_state) {
            eprintln!(
                "[gpu] GDN conv_state materialize failed, CPU fallback: {}",
                e
            );
            return Ok(None);
        }
    }

    Ok(Some(conv_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_gdn_qkv_conv_prefill_is_opt_in() {
        std::env::remove_var("RNB_VULKAN_GDN_QKV_CONV_PREFILL");

        assert!(!gdn_qkv_conv_prefill_enabled());
    }
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_gdn_prefill_ffn_chain_window(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    hidden_data: &mut [f32],
    hidden_dim: usize,
    norm_weight: &[f32],
    norm_eps: f32,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
) -> Result<bool, String> {
    if !gpu::layer_quant_supported(gate_weight.ggml_type)
        || !gpu::layer_quant_supported(up_weight.ggml_type)
        || !gpu::layer_quant_supported(down_weight.ggml_type)
    {
        return Ok(false);
    }
    gpu::gdn_prefill_ffn_chain_window(
        runtime,
        layer_idx,
        hidden_data,
        hidden_dim,
        norm_weight,
        norm_eps,
        gate_weight.ggml_type,
        gate_weight.data.as_bytes().unwrap_or(&[]),
        gate_weight.rows,
        gate_weight.cols,
        up_weight.ggml_type,
        up_weight.data.as_bytes().unwrap_or(&[]),
        up_weight.rows,
        up_weight.cols,
        down_weight.ggml_type,
        down_weight.data.as_bytes().unwrap_or(&[]),
        down_weight.rows,
        down_weight.cols,
    )
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_gdn_prefill_ffn_chain_window_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    hidden_data: &mut [f32],
    hidden_dim: usize,
    norm_weight: &[f32],
    norm_eps: f32,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    down_weight: &QuantizedWeight,
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_gdn_prefill_ffn_chain_window(
            runtime,
            layer_idx,
            hidden_data,
            hidden_dim,
            norm_weight,
            norm_eps,
            gate_weight,
            up_weight,
            down_weight,
        ) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] GDN prefill FFN chain failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}
