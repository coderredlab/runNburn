//! Nemotron-H Mamba2 blocks.
//!
//! This module will own recurrent state and Mamba2 tensor wiring for
//! `nemotron_h_moe`. Backend kernels stay behind `backend_runtime`.

#[cfg(any(not(feature = "cuda"), test))]
use super::policy::{prefill_projection_path, PrefillProjectionPath};
#[cfg(any(not(feature = "cuda"), test))]
use crate::engine::norm::{
    add_tensors, apply_plain_rms_norm, apply_plain_rms_norm_into, sigmoid_mul_f32_inplace,
};
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::engine::types::ModelMetadata;
use crate::engine::types::ScratchBuffers;
use crate::engine::{backend_runtime, cpu_runtime::kernels};
use crate::kv_cache::KVCache;
#[cfg(any(not(feature = "cuda"), test))]
use rayon::prelude::*;
use rnb_core::tensor::Tensor;
use rnb_loader::LoadedModel;
use std::time::Instant;

#[allow(dead_code)]
pub(in crate::engine) struct NemotronMamba2LayerWeights {
    pub(in crate::engine) norm: Tensor,
    pub(in crate::engine) ssm_in: QuantizedWeight,
    pub(in crate::engine) ssm_conv1d: Tensor,
    pub(in crate::engine) ssm_conv1d_bias: Tensor,
    pub(in crate::engine) ssm_dt_bias: Tensor,
    pub(in crate::engine) ssm_a: Tensor,
    pub(in crate::engine) ssm_d: Tensor,
    pub(in crate::engine) ssm_norm: Tensor,
    pub(in crate::engine) ssm_out: QuantizedWeight,
}

pub(in crate::engine) fn load_mamba2_layer(
    model: &LoadedModel,
    layer_idx: usize,
    load_f32_weight: fn(&LoadedModel, &str) -> Tensor,
    load_quantized_weight: fn(&LoadedModel, &str) -> QuantizedWeight,
) -> NemotronMamba2LayerWeights {
    NemotronMamba2LayerWeights {
        norm: load_f32_weight(model, &format!("blk.{layer_idx}.attn_norm.weight")),
        ssm_in: load_quantized_weight(model, &format!("blk.{layer_idx}.ssm_in.weight")),
        ssm_conv1d: load_mamba_conv1d(model, layer_idx, load_f32_weight),
        ssm_conv1d_bias: load_f32_weight(model, &format!("blk.{layer_idx}.ssm_conv1d.bias")),
        ssm_dt_bias: load_f32_weight(model, &format!("blk.{layer_idx}.ssm_dt.bias")),
        ssm_a: load_f32_weight(model, &format!("blk.{layer_idx}.ssm_a")),
        ssm_d: load_f32_weight(model, &format!("blk.{layer_idx}.ssm_d")),
        ssm_norm: load_f32_weight(model, &format!("blk.{layer_idx}.ssm_norm.weight")),
        ssm_out: load_quantized_weight(model, &format!("blk.{layer_idx}.ssm_out.weight")),
    }
}

pub(in crate::engine) fn forward_mamba2_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &NemotronMamba2LayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    #[cfg(all(feature = "cuda", not(test)))]
    {
        let hidden_dim = metadata.hidden_dim;
        let input = backend_runtime::upload_hidden_device_output_f32(
            kernels::tensor_as_f32_slice(&hidden),
            seq_len,
            hidden_dim,
        )?;
        return match forward_mamba2_layer_from_device(
            kv_cache, metadata, input, w, layer_idx, seq_len, norm_eps,
        )? {
            NemotronMamba2DeviceAttempt::Done(output, _) => {
                let values = backend_runtime::download_nemotron_device_layer_output(output)?;
                Ok(Tensor::from_vec(values, &[seq_len, hidden_dim]))
            }
            NemotronMamba2DeviceAttempt::Fallback(input) => {
                let _ = input.release()?;
                Err(crate::error::LlmError::Forward(format!(
                    "CUDA Mamba2 execution is unavailable for layer {layer_idx}; CPU fallback is disabled"
                )))
            }
        };
    }
    #[cfg(any(not(feature = "cuda"), test))]
    {
        let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
        let hidden_dim = metadata.hidden_dim;
        let d_inner = metadata.ssm_d_inner;
        let d_state = metadata.ssm_d_state;
        let n_group = metadata.ssm_n_group.max(1);
        let num_heads = metadata.ssm_dt_rank.max(1);
        let conv_kernel = metadata.ssm_conv_kernel.max(1);
        let head_dim = d_inner / num_heads;
        let bc_dim = n_group * d_state;
        let conv_channels = d_inner + 2 * bc_dim;

        let stage_start = (seq_len > 1).then(Instant::now);
        let normed = apply_plain_rms_norm(&hidden, &w.norm, norm_eps).map_err(fwd)?;
        record_prefill_mamba_stage("norm", stage_start);
        let normed_data = kernels::tensor_as_f32_slice(&normed);
        let stage_start = (seq_len > 1).then(Instant::now);
        let projected = quantized_project(&w.ssm_in, normed_data, seq_len, "nemotron_ssm_in")?;
        record_prefill_mamba_stage("ssm_in", stage_start);
        let expected_rows = d_inner + conv_channels + num_heads;
        if projected.len() != seq_len * expected_rows {
            return Err(crate::error::LlmError::Forward(format!(
                "Nemotron-H Mamba2 ssm_in rows {} != expected {}",
                projected.len() / seq_len,
                expected_rows
            )));
        }

        let stage_start = (seq_len > 1).then(Instant::now);
        let mut z_data = vec![0.0f32; seq_len * d_inner];
        let mut conv_seed = vec![0.0f32; seq_len * conv_channels];
        let mut dt_data = vec![0.0f32; seq_len * num_heads];
        let dt_bias = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
        for t in 0..seq_len {
            let src = &projected[t * expected_rows..(t + 1) * expected_rows];
            z_data[t * d_inner..(t + 1) * d_inner].copy_from_slice(&src[..d_inner]);
            conv_seed[t * conv_channels..(t + 1) * conv_channels]
                .copy_from_slice(&src[d_inner..d_inner + conv_channels]);
            for h in 0..num_heads {
                dt_data[t * num_heads + h] =
                    src[d_inner + conv_channels + h] + dt_bias[h.min(dt_bias.len() - 1)];
            }
        }
        record_prefill_mamba_stage("split", stage_start);

        let state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "Nemotron-H Mamba2 state not initialized for layer {layer_idx}"
            ))
        })?;
        let stage_start = (seq_len > 1).then(Instant::now);
        let conv_input = build_conv_input_and_advance_state(
            &mut state.conv_state,
            &conv_seed,
            seq_len,
            conv_channels,
            conv_kernel,
        );
        record_prefill_mamba_stage("conv_state", stage_start);
        let stage_start = (seq_len > 1).then(Instant::now);
        let total_conv_len = (conv_kernel - 1) + seq_len;
        let conv_input_tensor = Tensor::from_vec(conv_input, &[total_conv_len, conv_channels]);
        let conv_out = kernels::conv::ssm_conv1d(&conv_input_tensor, &w.ssm_conv1d).map_err(fwd)?;
        record_prefill_mamba_stage("conv", stage_start);
        let conv_data = kernels::tensor_as_f32_slice(&conv_out);
        let conv_bias = kernels::tensor_as_f32_slice(&w.ssm_conv1d_bias);
        let stage_start = (seq_len > 1).then(Instant::now);
        let mut conv_activated = vec![0.0f32; seq_len * conv_channels];
        for t in 0..seq_len {
            for c in 0..conv_channels {
                let idx = t * conv_channels + c;
                let x = conv_data[idx] + conv_bias[c.min(conv_bias.len() - 1)];
                conv_activated[idx] = x / (1.0 + (-x).exp());
            }
        }
        record_prefill_mamba_stage("conv_act", stage_start);
        let a_data = kernels::tensor_as_f32_slice(&w.ssm_a);
        let d_data = kernels::tensor_as_f32_slice(&w.ssm_d);

        let stage_start = (seq_len > 1).then(Instant::now);
        let scan_out = if seq_len > 1
            && crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_MAMBA_PREFILL_SCAN").as_deref()
                != Some("0")
        {
            if let Some(output) = backend_runtime::nemotron_mamba2_prefill_scan(
                &mut state.delta_state,
                &conv_activated,
                &dt_data,
                a_data,
                d_data,
                seq_len,
                d_inner,
                conv_channels,
                bc_dim,
                num_heads,
                head_dim,
                n_group,
                d_state,
            )? {
                output
            } else {
                scan_mamba2_cpu(
                    &mut state.delta_state,
                    &conv_activated,
                    &dt_data,
                    a_data,
                    d_data,
                    seq_len,
                    d_inner,
                    conv_channels,
                    bc_dim,
                    num_heads,
                    head_dim,
                    n_group,
                    d_state,
                )
            }
        } else if seq_len == 1
            && crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_MAMBA_SCAN").as_deref()
                == Some("1")
        {
            let token = &conv_activated[..conv_channels];
            let x = &token[..d_inner];
            let b = &token[d_inner..d_inner + bc_dim];
            let c = &token[d_inner + bc_dim..d_inner + 2 * bc_dim];
            let dt = dt_data[..num_heads]
                .iter()
                .map(|&value| softplus(value))
                .collect::<Vec<_>>();
            if let Some(output) = backend_runtime::nemotron_mamba2_decode_scan(
                &mut state.delta_state,
                x,
                b,
                c,
                &dt,
                a_data,
                d_data,
                num_heads,
                head_dim,
                d_state,
                n_group,
            )? {
                output
            } else {
                scan_mamba2_cpu(
                    &mut state.delta_state,
                    &conv_activated,
                    &dt_data,
                    a_data,
                    d_data,
                    seq_len,
                    d_inner,
                    conv_channels,
                    bc_dim,
                    num_heads,
                    head_dim,
                    n_group,
                    d_state,
                )
            }
        } else {
            scan_mamba2_cpu(
                &mut state.delta_state,
                &conv_activated,
                &dt_data,
                a_data,
                d_data,
                seq_len,
                d_inner,
                conv_channels,
                bc_dim,
                num_heads,
                head_dim,
                n_group,
                d_state,
            )
        };
        record_prefill_mamba_stage("scan", stage_start);

        let stage_start = (seq_len > 1).then(Instant::now);
        let mut gated = scan_out;
        sigmoid_mul_f32_inplace(&mut gated, &z_data);
        record_prefill_mamba_stage("gate", stage_start);
        let ssm_norm_data = kernels::tensor_as_f32_slice(&w.ssm_norm);
        let group_dim = d_inner / n_group;
        let stage_start = (seq_len > 1).then(Instant::now);
        let mut normed = vec![0.0f32; seq_len * d_inner];
        for t in 0..seq_len {
            for group in 0..n_group {
                let off = t * d_inner + group * group_dim;
                let weight_off = group * group_dim;
                apply_plain_rms_norm_into(
                    &gated[off..off + group_dim],
                    &ssm_norm_data[weight_off..weight_off + group_dim],
                    norm_eps,
                    &mut normed[off..off + group_dim],
                );
            }
        }
        record_prefill_mamba_stage("ssm_norm", stage_start);

        let stage_start = (seq_len > 1).then(Instant::now);
        let proj = quantized_project(&w.ssm_out, &normed, seq_len, "nemotron_ssm_out")?;
        record_prefill_mamba_stage("ssm_out", stage_start);
        let stage_start = (seq_len > 1).then(Instant::now);
        let proj_tensor = Tensor::from_vec(proj, &[seq_len, hidden_dim]);
        let output = add_tensors(&hidden, &proj_tensor).map_err(fwd);
        record_prefill_mamba_stage("residual", stage_start);
        output
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
#[allow(dead_code)]
pub(in crate::engine) enum NemotronMamba2DeviceAttempt {
    Done(
        backend_runtime::NemotronDeviceLayerOutput,
        backend_runtime::NemotronMamba2DeviceTrace,
    ),
    Fallback(backend_runtime::NemotronDeviceLayerOutput),
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn mamba2_device_fallback(
    input: backend_runtime::NemotronDeviceLayerOutput,
) -> NemotronMamba2DeviceAttempt {
    NemotronMamba2DeviceAttempt::Fallback(input)
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct Mamba2DeviceShape {
    hidden_dim: usize,
    d_inner: usize,
    d_state: usize,
    n_group: usize,
    num_heads: usize,
    conv_kernel: usize,
    head_dim: usize,
    bc_dim: usize,
    conv_channels: usize,
}

#[cfg(feature = "cuda")]
fn mamba2_device_shape(
    metadata: &ModelMetadata,
    seq_len: usize,
) -> Result<Mamba2DeviceShape, &'static str> {
    let hidden_dim = metadata.hidden_dim;
    let d_inner = metadata.ssm_d_inner;
    let d_state = metadata.ssm_d_state;
    let n_group = metadata.ssm_n_group.max(1);
    let num_heads = metadata.ssm_dt_rank.max(1);
    let conv_kernel = metadata.ssm_conv_kernel.max(1);
    if seq_len == 0 || hidden_dim == 0 || d_inner == 0 || d_state == 0 {
        return Err("invalid_zero_dim");
    }
    if d_state > 256 {
        return Err("unsupported_state_dim");
    }
    if conv_kernel < 2 {
        return Err("unsupported_conv_kernel");
    }
    if num_heads % n_group != 0 || d_inner % n_group != 0 {
        return Err("invalid_ssm_groups");
    }
    if d_inner % num_heads != 0 {
        return Err("invalid_head_dim");
    }
    let head_dim = d_inner / num_heads;
    if head_dim == 0 {
        return Err("invalid_head_dim");
    }
    let bc_dim = n_group.checked_mul(d_state).ok_or("invalid_bc_dim")?;
    let double_bc_dim = bc_dim.checked_mul(2).ok_or("invalid_conv_channels")?;
    let conv_channels = d_inner
        .checked_add(double_bc_dim)
        .ok_or("invalid_conv_channels")?;
    Ok(Mamba2DeviceShape {
        hidden_dim,
        d_inner,
        d_state,
        n_group,
        num_heads,
        conv_kernel,
        head_dim,
        bc_dim,
        conv_channels,
    })
}

#[cfg(feature = "cuda")]
fn mamba2_state_error_after_input_release(
    err: String,
    input_release_result: crate::error::Result<bool>,
) -> crate::error::LlmError {
    match input_release_result {
        Ok(true) => crate::error::LlmError::Forward(err),
        Ok(false) => crate::error::LlmError::Forward(format!("{err}; input cleanup missing")),
        Err(cleanup_err) => {
            crate::error::LlmError::Forward(format!("{err}; cleanup failed: {cleanup_err}"))
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn forward_mamba2_layer_from_device(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    input: backend_runtime::NemotronDeviceLayerOutput,
    w: &NemotronMamba2LayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<NemotronMamba2DeviceAttempt> {
    if !device_prefill_mamba2_requested_for_chain() {
        emit_mamba2_device_fallback_trace(layer_idx, "cuda_feature_disabled");
        return Ok(mamba2_device_fallback(input));
    }
    let Some(ssm_in_bytes) = w.ssm_in.data.as_bytes() else {
        emit_mamba2_device_fallback_trace(layer_idx, "unsupported_ssm_in_quant");
        return Ok(mamba2_device_fallback(input));
    };
    let Some(ssm_out_bytes) = w.ssm_out.data.as_bytes() else {
        emit_mamba2_device_fallback_trace(layer_idx, "unsupported_ssm_out_quant");
        return Ok(mamba2_device_fallback(input));
    };
    if w.ssm_in.ggml_type != rnb_loader::GGMLType::Q8_0 {
        emit_mamba2_device_fallback_trace(layer_idx, "unsupported_ssm_in_quant");
        return Ok(mamba2_device_fallback(input));
    }
    if w.ssm_out.ggml_type != rnb_loader::GGMLType::Q8_0 {
        emit_mamba2_device_fallback_trace(layer_idx, "unsupported_ssm_out_quant");
        return Ok(mamba2_device_fallback(input));
    }
    let shape = match mamba2_device_shape(metadata, seq_len) {
        Ok(shape) => shape,
        Err(reason) => {
            emit_mamba2_device_fallback_trace(layer_idx, reason);
            return Ok(mamba2_device_fallback(input));
        }
    };
    let state = match kv_cache.get_ssm_state_mut(layer_idx) {
        Some(state) => state,
        None => {
            let err = format!("Nemotron-H Mamba2 state not initialized for layer {layer_idx}");
            return Err(mamba2_state_error_after_input_release(err, input.release()));
        }
    };
    let input_norm = kernels::tensor_as_f32_slice(&w.norm);
    let conv_kernel_data = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
    let conv_bias = kernels::tensor_as_f32_slice(&w.ssm_conv1d_bias);
    let dt_bias = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
    let ssm_a = kernels::tensor_as_f32_slice(&w.ssm_a);
    let ssm_d = kernels::tensor_as_f32_slice(&w.ssm_d);
    let ssm_norm = kernels::tensor_as_f32_slice(&w.ssm_norm);
    backend_runtime::nemotron_mamba2_prefill_device(
        input,
        w.ssm_in.ggml_type,
        ssm_in_bytes,
        w.ssm_in.rows,
        w.ssm_in.cols,
        w.ssm_out.ggml_type,
        ssm_out_bytes,
        w.ssm_out.rows,
        w.ssm_out.cols,
        input_norm,
        conv_kernel_data,
        conv_bias,
        dt_bias,
        ssm_a,
        ssm_d,
        ssm_norm,
        &mut state.conv_state,
        &mut state.delta_state,
        seq_len,
        shape.hidden_dim,
        shape.d_inner,
        shape.conv_channels,
        shape.bc_dim,
        shape.num_heads,
        shape.head_dim,
        shape.n_group,
        shape.d_state,
        shape.conv_kernel,
        norm_eps,
    )
    .map(|(output, trace)| NemotronMamba2DeviceAttempt::Done(output, trace))
}

#[allow(dead_code)]
pub(in crate::engine) fn device_prefill_mamba2_enabled() -> bool {
    true
}

#[allow(dead_code)]
pub(in crate::engine) fn device_prefill_mamba2_requested_for_chain() -> bool {
    true
}

#[allow(dead_code)]
pub(in crate::engine) fn device_prefill_trace_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL_TRACE").as_deref() == Some("1")
}

#[allow(dead_code)]
fn emit_mamba2_device_fallback_trace(layer_idx: usize, reason: &str) {
    if !device_prefill_trace_enabled() {
        return;
    }
    eprintln!(
        "[cuda:device-prefill-chain] op=nemotron_mamba2_device layer={} chain_supported=0 reason={}",
        layer_idx, reason
    );
}

#[cfg(any(not(feature = "cuda"), test))]
fn record_prefill_mamba_stage(stage: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        let key = match stage {
            "norm" => "nemotron:prefill:mamba:norm",
            "ssm_in" => "nemotron:prefill:mamba:ssm_in",
            "split" => "nemotron:prefill:mamba:split",
            "conv_state" => "nemotron:prefill:mamba:conv_state",
            "conv" => "nemotron:prefill:mamba:conv",
            "conv_act" => "nemotron:prefill:mamba:conv_act",
            "scan" => "nemotron:prefill:mamba:scan",
            "gate" => "nemotron:prefill:mamba:gate",
            "ssm_norm" => "nemotron:prefill:mamba:ssm_norm",
            "ssm_out" => "nemotron:prefill:mamba:ssm_out",
            "residual" => "nemotron:prefill:mamba:residual",
            _ => return,
        };
        crate::engine::moe_profile::record_moe_profile(key, start.elapsed());
    }
}

#[cfg(any(not(feature = "cuda"), test))]
fn quantized_project(
    weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
    label: &str,
) -> crate::error::Result<Vec<f32>> {
    if seq_len > 1 {
        match prefill_projection_path(seq_len, "RNB_CUDA_NEMOTRON_MAMBA_PREFILL_Q_GEMV") {
            PrefillProjectionPath::F32Gemm => {
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection(weight, input)?
                {
                    return Ok(output);
                }
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection_q(weight, input)?
                {
                    return Ok(output);
                }
                return weight.gemv_vec(input);
            }
            PrefillProjectionPath::QuantizedGemv => {
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection_q(weight, input)?
                {
                    return Ok(output);
                }
                return weight.gemv_vec(input);
            }
        }
    } else {
        let mut output = vec![0.0f32; weight.rows];
        if backend_runtime::decode_gemv_into_if_supported(weight, input, &mut output, label, false)?
        {
            return Ok(output);
        }
    }
    weight.gemv_vec(input)
}

pub(in crate::engine) fn decode_mamba2_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    w: &NemotronMamba2LayerWeights,
    layer_idx: usize,
    norm_eps: f32,
) -> crate::error::Result<()> {
    let hidden_dim = metadata.hidden_dim;
    let input = Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]);
    let profile_start = crate::engine::moe_profile::is_enabled().then(Instant::now);
    let output = forward_mamba2_layer(kv_cache, metadata, input, w, layer_idx, 1, norm_eps)?;
    if let Some(start) = profile_start {
        crate::engine::moe_profile::record_moe_profile(
            "nemotron:decode:mamba:total",
            start.elapsed(),
        );
    }
    let output_data = kernels::tensor_as_f32_slice(&output);
    scratch.hidden[..hidden_dim].copy_from_slice(&output_data[..hidden_dim]);
    Ok(())
}

fn load_mamba_conv1d(
    model: &LoadedModel,
    layer_idx: usize,
    load_f32_weight: fn(&LoadedModel, &str) -> Tensor,
) -> Tensor {
    let conv_raw = load_f32_weight(model, &format!("blk.{layer_idx}.ssm_conv1d.weight"));
    let conv_shape = conv_raw.shape();
    if conv_shape.len() == 2 && conv_shape[0] > conv_shape[1] {
        let data = kernels::tensor_as_f32_slice(&conv_raw);
        let channels = conv_shape[0];
        let ksize = conv_shape[1];
        let mut transposed = vec![0.0f32; channels * ksize];
        for c in 0..channels {
            for k in 0..ksize {
                transposed[k * channels + c] = data[c * ksize + k];
            }
        }
        Tensor::from_slice(&transposed, &[ksize, channels])
    } else {
        conv_raw
    }
}

#[cfg(any(not(feature = "cuda"), test))]
fn build_conv_input_and_advance_state(
    state: &mut [f32],
    current: &[f32],
    seq_len: usize,
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let state_len = (kernel - 1) * channels;
    let mut out = vec![0.0f32; state_len + seq_len * channels];
    out[..state_len].copy_from_slice(state);
    out[state_len..].copy_from_slice(current);
    if state_len > 0 {
        let suffix_start = out.len() - state_len;
        state.copy_from_slice(&out[suffix_start..]);
    }
    out
}

#[cfg(any(not(feature = "cuda"), test))]
#[allow(clippy::too_many_arguments)]
fn scan_mamba2_cpu(
    state: &mut [f32],
    conv_activated: &[f32],
    dt_data: &[f32],
    a_data: &[f32],
    d_data: &[f32],
    seq_len: usize,
    d_inner: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    n_group: usize,
    d_state: usize,
) -> Vec<f32> {
    let mut scan_out = vec![0.0f32; seq_len * d_inner];
    let head_state_len = head_dim * d_state;
    let heads_per_group = (num_heads / n_group).max(1);
    for t in 0..seq_len {
        let token = &conv_activated[t * conv_channels..(t + 1) * conv_channels];
        let x = &token[..d_inner];
        let b = &token[d_inner..d_inner + bc_dim];
        let c = &token[d_inner + bc_dim..d_inner + 2 * bc_dim];
        let out_token = &mut scan_out[t * d_inner..(t + 1) * d_inner];
        state
            .par_chunks_mut(head_state_len)
            .zip(out_token.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(h, (state_head, out_head))| {
                let dt = softplus(dt_data[t * num_heads + h]);
                let a = a_data[h.min(a_data.len().saturating_sub(1))];
                let d = d_data[h.min(d_data.len().saturating_sub(1))];
                let group = h / heads_per_group;
                let decay = (dt * a).exp();
                for p in 0..head_dim {
                    let x_idx = h * head_dim + p;
                    let mut y = d * x[x_idx];
                    let x_dt = x[x_idx] * dt;
                    for s in 0..d_state {
                        let bc_idx = group * d_state + s;
                        let state_idx = p * d_state + s;
                        state_head[state_idx] = state_head[state_idx] * decay + b[bc_idx] * x_dt;
                        y += state_head[state_idx] * c[bc_idx];
                    }
                    out_head[p] = y;
                }
            });
    }
    scan_out
}

#[cfg(any(not(feature = "cuda"), test))]
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_loader::GGMLType;

    #[cfg(feature = "cuda")]
    #[test]
    fn mamba2_device_fallback_returns_input_owner() {
        let input = dummy_device_layer_output(17);
        let output_id = input.output_id;
        let output_desc = input.output_desc;

        match super::mamba2_device_fallback(input) {
            super::NemotronMamba2DeviceAttempt::Fallback(input) => {
                assert_eq!(input.output_id, output_id);
                assert_eq!(input.output_desc, output_desc);
            }
            super::NemotronMamba2DeviceAttempt::Done(_, _) => {
                panic!("fallback helper returned done")
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mamba2_device_shape_guard_rejects_backend_fallback_shapes() {
        let mut metadata = mamba2_device_test_metadata();
        metadata.ssm_conv_kernel = 1;
        assert_eq!(
            super::mamba2_device_shape(&metadata, 2).unwrap_err(),
            "unsupported_conv_kernel"
        );

        metadata = mamba2_device_test_metadata();
        metadata.ssm_dt_rank = 5;
        metadata.ssm_n_group = 2;
        assert_eq!(
            super::mamba2_device_shape(&metadata, 2).unwrap_err(),
            "invalid_ssm_groups"
        );

        metadata = mamba2_device_test_metadata();
        metadata.ssm_dt_rank = 3;
        assert_eq!(
            super::mamba2_device_shape(&metadata, 2).unwrap_err(),
            "invalid_head_dim"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mamba2_device_shape_rejects_state_dim_above_cuda_block_width() {
        let mut metadata = mamba2_device_test_metadata();
        metadata.ssm_d_state = 257;

        assert_eq!(
            super::mamba2_device_shape(&metadata, 2).unwrap_err(),
            "unsupported_state_dim"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mamba2_state_error_cleanup_message_reports_missing_or_failed_cleanup() {
        let primary = "Nemotron-H Mamba2 state not initialized for layer 7";

        assert_eq!(
            forward_error_message(super::mamba2_state_error_after_input_release(
                primary.to_string(),
                Ok(true),
            )),
            primary
        );

        let missing = forward_error_message(super::mamba2_state_error_after_input_release(
            primary.to_string(),
            Ok(false),
        ));
        assert!(missing.contains(primary));
        assert!(missing.contains("input cleanup missing"));

        let failed = forward_error_message(super::mamba2_state_error_after_input_release(
            primary.to_string(),
            Err(crate::error::LlmError::Forward(
                "release failed".to_string(),
            )),
        ));
        assert!(failed.contains(primary));
        assert!(failed.contains("cleanup failed"));
        assert!(failed.contains("release failed"));
    }

    #[test]
    fn mamba2_forward_zero_projection_preserves_residual() {
        let metadata = ModelMetadata {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size: 1,
            max_seq_len: 1,
            hidden_dim: 2,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 1.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            head_count_kv_per_layer: None,
            embedding_length_per_layer_input: 0,
            expert_used_count: 0,
            expert_weights_scale: 1.0,
            ssm_d_inner: 2,
            ssm_d_state: 1,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ssm_conv_kernel: 1,
            full_attention_interval: 0,
        };
        let weights = NemotronMamba2LayerWeights {
            norm: Tensor::from_slice(&[1.0, 1.0], &[2]),
            ssm_in: qweight(&[0.0; 14], 7, 2),
            ssm_conv1d: Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[1, 4]),
            ssm_conv1d_bias: Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[4]),
            ssm_dt_bias: Tensor::from_slice(&[0.0], &[1]),
            ssm_a: Tensor::from_slice(&[-1.0], &[1]),
            ssm_d: Tensor::from_slice(&[0.0, 0.0], &[2]),
            ssm_norm: Tensor::from_slice(&[1.0, 1.0], &[2]),
            ssm_out: qweight(&[0.0, 0.0, 0.0, 0.0], 2, 2),
        };
        let mut kv_cache = KVCache::new_per_layer(1, &[0], &[2]);
        kv_cache.init_ssm_state(0, 1, 4, 1, 2, 1);

        let out = forward_mamba2_layer(
            &mut kv_cache,
            &metadata,
            Tensor::from_slice(&[1.0, -2.0], &[1, 2]),
            &weights,
            0,
            1,
            1e-5,
        )
        .unwrap();

        assert_eq!(kernels::tensor_as_f32_slice(&out), &[1.0, -2.0]);
    }

    fn qweight(data: &[f32], rows: usize, cols: usize) -> QuantizedWeight {
        QuantizedWeight::new(
            Tensor::from_slice(data, &[rows, cols]),
            GGMLType::F32,
            rows,
            cols,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_error_message(err: crate::error::LlmError) -> String {
        match err {
            crate::error::LlmError::Forward(message) => message,
            other => panic!("expected forward error, got {other:?}"),
        }
    }

    #[cfg(feature = "cuda")]
    fn dummy_device_layer_output(raw: u64) -> backend_runtime::NemotronDeviceLayerOutput {
        use crate::engine::cuda_runtime::{
            DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, ScalarType,
        };
        use crate::runtime::BackendKind;

        backend_runtime::NemotronDeviceLayerOutput {
            output_id: DeviceTensorId::new(BackendKind::Cuda, raw),
            output_desc: DeviceTensorDesc::new(2, 2, ScalarType::F32, DeviceTensorRole::Hidden),
        }
    }

    #[cfg(feature = "cuda")]
    fn mamba2_device_test_metadata() -> ModelMetadata {
        ModelMetadata {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size: 1,
            max_seq_len: 2,
            hidden_dim: 2,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 1.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            head_count_kv_per_layer: None,
            embedding_length_per_layer_input: 0,
            expert_used_count: 0,
            expert_weights_scale: 1.0,
            ssm_d_inner: 10,
            ssm_d_state: 1,
            ssm_n_group: 1,
            ssm_dt_rank: 5,
            ssm_conv_kernel: 2,
            full_attention_interval: 0,
        }
    }

    #[cfg(feature = "cuda")]
    fn mamba2_device_test_weights() -> NemotronMamba2LayerWeights {
        NemotronMamba2LayerWeights {
            norm: Tensor::from_slice(&[1.0, 1.0], &[2]),
            ssm_in: qweight(&[0.0], 1, 1),
            ssm_conv1d: Tensor::from_slice(&[1.0], &[1, 1]),
            ssm_conv1d_bias: Tensor::from_slice(&[0.0], &[1]),
            ssm_dt_bias: Tensor::from_slice(&[0.0], &[1]),
            ssm_a: Tensor::from_slice(&[-1.0], &[1]),
            ssm_d: Tensor::from_slice(&[0.0], &[1]),
            ssm_norm: Tensor::from_slice(&[1.0], &[1]),
            ssm_out: qweight(&[0.0], 1, 1),
        }
    }
}
