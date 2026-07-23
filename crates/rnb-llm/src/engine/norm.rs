//! Gemma-aware wrappers around `super::cpu_runtime::kernels::norm` / activation.

#[cfg(feature = "cuda")]
use super::backend_runtime;
use super::cpu_runtime::kernels;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;

use super::use_gemma_block_semantics;

pub(super) fn apply_model_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    architecture: ModelArchitecture,
) -> rnb_core::error::Result<Tensor> {
    let unit_offset = use_gemma_block_semantics(architecture)
        && (super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled());
    #[cfg(feature = "cuda")]
    {
        let input_data = kernels::tensor_as_f32_slice(input);
        let weight_data = kernels::tensor_as_f32_slice(weight);
        let mut output = vec![0.0f32; input_data.len()];
        backend_runtime::cuda_rms_norm_rows(input_data, weight_data, eps, &mut output, unit_offset)
            .unwrap_or_else(|err| panic!("CUDA model RMS norm failed: {err}"));
        return Ok(Tensor::from_vec(output, input.shape()));
    }
    #[cfg(not(feature = "cuda"))]
    if unit_offset {
        kernels::norm::rms_norm_unit_offset(input, weight, eps)
    } else {
        kernels::norm::rms_norm(input, weight, eps)
    }
}

pub(super) fn apply_model_norm_unit_offset(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> rnb_core::error::Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let input_data = kernels::tensor_as_f32_slice(input);
        let weight_data = kernels::tensor_as_f32_slice(weight);
        let mut output = vec![0.0f32; input_data.len()];
        backend_runtime::cuda_rms_norm_rows(input_data, weight_data, eps, &mut output, true)
            .unwrap_or_else(|err| panic!("CUDA unit-offset RMS norm failed: {err}"));
        return Ok(Tensor::from_vec(output, input.shape()));
    }
    #[cfg(not(feature = "cuda"))]
    kernels::norm::rms_norm_unit_offset(input, weight, eps)
}

pub(super) fn apply_model_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    architecture: ModelArchitecture,
) {
    let unit_offset = use_gemma_block_semantics(architecture)
        && (super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled());
    #[cfg(feature = "cuda")]
    {
        backend_runtime::cuda_rms_norm_rows(input, weight, eps, output, unit_offset)
            .unwrap_or_else(|err| panic!("CUDA model RMS norm failed: {err}"));
    }
    #[cfg(not(feature = "cuda"))]
    if unit_offset {
        kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
    } else {
        kernels::norm::rms_norm_into(input, weight, eps, output);
    }
}

pub(super) fn apply_model_norm_unit_offset_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_rms_norm_rows(input, weight, eps, output, true)
        .unwrap_or_else(|err| panic!("CUDA unit-offset RMS norm failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
}

pub(super) fn apply_plain_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> rnb_core::error::Result<Tensor> {
    let input_data = kernels::tensor_as_f32_slice(input);
    let weight_data = kernels::tensor_as_f32_slice(weight);
    let mut output = vec![0.0f32; input_data.len()];
    apply_plain_rms_norm_into(input_data, weight_data, eps, &mut output);
    Ok(Tensor::from_vec(output, input.shape()))
}

pub(super) fn apply_plain_rms_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_rms_norm_rows(input, weight, eps, output, false)
        .unwrap_or_else(|err| panic!("CUDA plain RMS norm failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    kernels::norm::rms_norm_into(input, weight, eps, output);
}

pub(super) fn apply_l2_norm_into(input: &[f32], eps: f32, output: &mut [f32], row_width: usize) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_l2_norm_rows(input, output, row_width, eps)
        .unwrap_or_else(|err| panic!("CUDA L2 norm failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    kernels::norm::l2_norm_into(input, eps, output, row_width);
}

pub(super) fn apply_l2_norm(input: &Tensor, eps: f32) -> rnb_core::error::Result<Tensor> {
    let row_width = input.shape().last().copied().unwrap_or(0);
    let input_data = kernels::tensor_as_f32_slice(input);
    let mut output = vec![0.0f32; input_data.len()];
    apply_l2_norm_into(input_data, eps, &mut output, row_width);
    Ok(Tensor::from_vec(output, input.shape()))
}

pub(super) fn apply_model_gate_mul_inplace(
    gate: &mut [f32],
    up: &[f32],
    architecture: ModelArchitecture,
) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_activation_mul_inplace(gate, up, use_gemma_block_semantics(architecture))
        .unwrap_or_else(|err| panic!("CUDA gated activation failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    if use_gemma_block_semantics(architecture) {
        kernels::activation::fused_gelu_mul_inplace(gate, up);
    } else {
        kernels::activation::fused_silu_mul_inplace(gate, up);
    }
}

pub(super) fn apply_model_qk_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    architecture: ModelArchitecture,
) -> rnb_core::error::Result<Tensor> {
    if use_gemma_block_semantics(architecture) && super::policy::gemma_qk_norm_disabled() {
        return Ok(input.clone());
    }
    let input_data = kernels::tensor_as_f32_slice(input);
    let weight_data = kernels::tensor_as_f32_slice(weight);
    let unit_offset =
        use_gemma_block_semantics(architecture) && super::policy::gemma_unit_offset_norm_enabled();
    #[cfg(feature = "cuda")]
    {
        let mut output = vec![0.0f32; input_data.len()];
        backend_runtime::cuda_rms_norm_rows(input_data, weight_data, eps, &mut output, unit_offset)
            .unwrap_or_else(|err| panic!("CUDA QK RMS norm failed: {err}"));
        return Ok(Tensor::from_vec(output, input.shape()));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut output = vec![0.0f32; input_data.len()];
        for (src, dst) in input_data
            .chunks_exact(weight_data.len())
            .zip(output.chunks_exact_mut(weight_data.len()))
        {
            if unit_offset {
                kernels::norm::rms_norm_unit_offset_into(src, weight_data, eps, dst);
            } else {
                kernels::norm::rms_norm_into(src, weight_data, eps, dst);
            }
        }
        Ok(Tensor::from_vec(output, input.shape()))
    }
}

pub(super) fn apply_model_qk_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    architecture: ModelArchitecture,
) {
    if use_gemma_block_semantics(architecture) && super::policy::gemma_qk_norm_disabled() {
        output.copy_from_slice(input);
        return;
    }
    let unit_offset =
        use_gemma_block_semantics(architecture) && super::policy::gemma_unit_offset_norm_enabled();
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_rms_norm_rows(input, weight, eps, output, unit_offset)
        .unwrap_or_else(|err| panic!("CUDA QK RMS norm failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    if unit_offset {
        kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
    } else {
        kernels::norm::rms_norm_into(input, weight, eps, output);
    }
}

pub(super) fn apply_rms_norm_no_scale_into(input: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(feature = "cuda")]
    {
        let weight = vec![1.0f32; input.len()];
        backend_runtime::cuda_rms_norm_rows(input, &weight, eps, output, false)
            .unwrap_or_else(|err| panic!("CUDA unscaled RMS norm failed: {err}"));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
        let inv_rms = (mean_sq + eps).powf(-0.5);
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = *src * inv_rms;
        }
    }
}

pub(super) fn hadamard_inplace(block: &mut [f32]) {
    let n = block.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "hadamard block must be power-of-two");
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_hadamard_f32_inplace(block, n)
        .unwrap_or_else(|err| panic!("CUDA Hadamard transform failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    {
        let mut stride = 1usize;
        while stride < n {
            let step = stride * 2;
            let mut i = 0usize;
            while i < n {
                for j in 0..stride {
                    let a = block[i + j];
                    let b = block[i + j + stride];
                    block[i + j] = a + b;
                    block[i + j + stride] = a - b;
                }
                i += step;
            }
            stride = step;
        }
        let scale = (n as f32).sqrt().recip();
        for v in block.iter_mut() {
            *v *= scale;
        }
    }
}

pub(super) fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_add_f32_inplace(dst, src)
        .unwrap_or_else(|err| panic!("CUDA residual add failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        *dst += *src;
    }
}

pub(super) fn add_tensors(lhs: &Tensor, rhs: &Tensor) -> rnb_core::error::Result<Tensor> {
    let mut output = kernels::tensor_as_f32_slice(lhs).to_vec();
    let rhs_shape = rhs.shape().to_vec();
    let rhs = kernels::tensor_as_f32_slice(rhs);
    if rhs.is_empty() || output.len() % rhs.len() != 0 {
        return Err(rnb_core::error::RnbError::ShapeMismatch {
            expected: lhs.shape().to_vec(),
            got: rhs_shape,
        });
    }
    add_rows_f32_inplace(&mut output, rhs);
    Ok(Tensor::from_vec(output, lhs.shape()))
}

pub(super) fn add_rows_f32_inplace(dst: &mut [f32], src: &[f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_add_rows_f32_inplace(dst, src)
        .unwrap_or_else(|err| panic!("CUDA broadcast add failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for (index, dst) in dst.iter_mut().enumerate() {
        *dst += src[index % src.len()];
    }
}

pub(super) fn mul_rows_f32_inplace(dst: &mut [f32], src: &[f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_mul_rows_f32_inplace(dst, src)
        .unwrap_or_else(|err| panic!("CUDA broadcast multiply failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for (index, dst) in dst.iter_mut().enumerate() {
        *dst *= src[index % src.len()];
    }
}

pub(super) fn scale_f32_inplace(values: &mut [f32], scale: f32) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_scale_f32_inplace(values, scale)
        .unwrap_or_else(|err| panic!("CUDA scale failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for value in values {
        *value *= scale;
    }
}

#[cfg(any(not(feature = "cuda"), test))]
pub(super) fn sigmoid_f32_inplace(values: &mut [f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_sigmoid_f32_inplace(values)
        .unwrap_or_else(|err| panic!("CUDA sigmoid failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    kernels::activation::sigmoid_inplace(values);
}

pub(super) fn axpby_f32_inplace(dst: &mut [f32], src: &[f32], alpha: f32, beta: f32) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_axpby_f32_inplace(dst, src, alpha, beta)
        .unwrap_or_else(|err| panic!("CUDA axpby failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        *dst = alpha * *src + beta * *dst;
    }
}

pub(super) fn sigmoid_mul_f32_inplace(values: &mut [f32], gate: &[f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_sigmoid_mul_f32_inplace(values, gate)
        .unwrap_or_else(|err| panic!("CUDA sigmoid multiply failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for (value, gate) in values.iter_mut().zip(gate.iter()) {
        *value *= 1.0 / (1.0 + (-*gate).exp());
    }
}

#[cfg(any(not(feature = "cuda"), test))]
pub(super) fn relu_sqr_f32_inplace(values: &mut [f32]) {
    #[cfg(feature = "cuda")]
    backend_runtime::cuda_relu_sqr_f32_inplace(values)
        .unwrap_or_else(|err| panic!("CUDA relu-squared failed: {err}"));
    #[cfg(not(feature = "cuda"))]
    for value in values {
        *value = rnb_model_nemotron::relu_sqr(*value);
    }
}
