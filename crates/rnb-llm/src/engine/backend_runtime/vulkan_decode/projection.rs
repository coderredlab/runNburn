//! Vulkan decode projection GEMV helpers.

use super::*;

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn try_decode_gemv(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    kind: DecodeProjectionKind,
    weight: &QuantizedWeight,
    rows: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<bool, String> {
    let weight = gpu_decode_weight(kind, weight, rows, weight.ggml_type);
    gpu::decode_gemv(runtime, layer_idx, weight, input, output)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn try_decode_gemv_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    kind: DecodeProjectionKind,
    weight: &QuantizedWeight,
    rows: usize,
    input: &[f32],
    output: &mut [f32],
    label: &str,
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_gemv(runtime, layer_idx, kind, weight, rows, input, output) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] {label} GEMV failed, CPU fallback: {err}");
                return false;
            }
        }
    }

    #[cfg(feature = "cuda")]
    match super::super::cuda_basic::decode_gemv_into_if_supported(
        weight, input, output, label, false,
    ) {
        Ok(true) => {
            if output.len() > rows {
                output[rows..].fill(0.0);
            }
            return true;
        }
        Ok(false) => {}
        Err(err) => {
            eprintln!("[cuda] {label} GEMV failed, CPU fallback: {err}");
            return false;
        }
    }

    false
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_attention_qkv(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    input: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_rows: usize,
    kv_rows: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<bool, String> {
    let weights = [
        gpu_decode_weight(
            DecodeProjectionKind::QProj,
            q_weight,
            q_rows,
            q_weight.ggml_type,
        ),
        gpu_decode_weight(
            DecodeProjectionKind::KProj,
            k_weight,
            kv_rows,
            k_weight.ggml_type,
        ),
        gpu_decode_weight(
            DecodeProjectionKind::VProj,
            v_weight,
            kv_rows,
            v_weight.ggml_type,
        ),
    ];
    let mut outputs: [&mut [f32]; 3] = [q_out, k_out, v_out];
    gpu::decode_gemv_multi(runtime, layer_idx, input, &weights, &mut outputs)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_attention_qkv_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    input: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_rows: usize,
    kv_rows: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_attention_qkv(
            runtime, layer_idx, input, q_weight, k_weight, v_weight, q_rows, kv_rows, q_out, k_out,
            v_out,
        ) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] QKV batch GEMV failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_gdn_qkv_gate(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    input: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    qkv_rows: usize,
    gate_rows: usize,
    qkv_out: &mut [f32],
    gate_out: &mut [f32],
) -> Result<bool, String> {
    let weights = [
        gpu_decode_weight(
            DecodeProjectionKind::GdnQkv,
            qkv_weight,
            qkv_rows,
            qkv_weight.ggml_type,
        ),
        gpu_decode_weight(
            DecodeProjectionKind::GdnGate,
            gate_weight,
            gate_rows,
            qkv_weight.ggml_type,
        ),
    ];
    let mut outputs: [&mut [f32]; 2] = [qkv_out, gate_out];
    gpu::decode_gemv_multi(runtime, layer_idx, input, &weights, &mut outputs)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_gdn_qkv_gate_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    input: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    qkv_rows: usize,
    gate_rows: usize,
    qkv_out: &mut [f32],
    gate_out: &mut [f32],
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_gdn_qkv_gate(
            runtime,
            layer_idx,
            input,
            qkv_weight,
            gate_weight,
            qkv_rows,
            gate_rows,
            qkv_out,
            gate_out,
        ) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] GDN qkv+gate GEMV failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn try_decode_gdn_alpha_beta(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    input: &[f32],
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    rows: usize,
    alpha_out: &mut [f32],
    beta_out: &mut [f32],
) -> Result<bool, String> {
    let weights = [
        gpu_decode_weight(
            DecodeProjectionKind::GdnAlpha,
            alpha_weight,
            rows,
            alpha_weight.ggml_type,
        ),
        gpu_decode_weight(
            DecodeProjectionKind::GdnBeta,
            beta_weight,
            rows,
            beta_weight.ggml_type,
        ),
    ];
    let mut outputs: [&mut [f32]; 2] = [alpha_out, beta_out];
    gpu::decode_gemv_multi_same_quant(runtime, layer_idx, input, &weights, &mut outputs)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn try_decode_gdn_alpha_beta_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    input: &[f32],
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    rows: usize,
    alpha_out: &mut [f32],
    beta_out: &mut [f32],
) -> bool {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        match try_decode_gdn_alpha_beta(
            runtime,
            layer_idx,
            input,
            alpha_weight,
            beta_weight,
            rows,
            alpha_out,
            beta_out,
        ) {
            Ok(done) => return done,
            Err(err) => {
                eprintln!("[gpu] GDN alpha+beta GEMV failed, CPU fallback: {err}");
                return false;
            }
        }
    }
    false
}
