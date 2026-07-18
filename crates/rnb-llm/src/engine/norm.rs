//! Gemma-aware wrappers around `super::cpu_runtime::kernels::norm` / activation.

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
    if use_gemma_block_semantics(architecture)
        && super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
    {
        return kernels::norm::rms_norm_unit_offset(input, weight, eps);
    }
    if use_gemma_block_semantics(architecture)
        && (super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled())
    {
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
    kernels::norm::rms_norm_unit_offset(input, weight, eps)
}

pub(super) fn apply_model_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    architecture: ModelArchitecture,
) {
    if use_gemma_block_semantics(architecture)
        && super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
    {
        kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
        return;
    }
    if use_gemma_block_semantics(architecture)
        && (super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled())
    {
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
    kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
}

pub(super) fn apply_model_gate_mul_inplace(
    gate: &mut [f32],
    up: &[f32],
    architecture: ModelArchitecture,
) {
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
    if use_gemma_block_semantics(architecture) && super::policy::gemma_unit_offset_norm_enabled() {
        kernels::norm::rms_norm_unit_offset(input, weight, eps)
    } else {
        // mv39: per-head rms_norm_into 호출 (mc71 production fix conjunction의 f64
        // accumulator path). hybrid path의 apply_optional_qk_norm_per_head와 같은
        // 함수로 통일 → vulkan partial path q_norm token-identical.
        let input_data = kernels::tensor_as_f32_slice(input);
        let weight_data = kernels::tensor_as_f32_slice(weight);
        let head_dim = weight_data.len();
        let total = input_data.len();
        debug_assert_eq!(total % head_dim, 0);
        let mut out = vec![0.0f32; total];
        for h_off in (0..total).step_by(head_dim) {
            kernels::norm::rms_norm_into(
                &input_data[h_off..h_off + head_dim],
                weight_data,
                eps,
                &mut out[h_off..h_off + head_dim],
            );
        }
        Ok(Tensor::from_vec(out, input.shape()))
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
    if use_gemma_block_semantics(architecture) && super::policy::gemma_unit_offset_norm_enabled() {
        kernels::norm::rms_norm_unit_offset_into(input, weight, eps, output);
    } else {
        kernels::norm::rms_norm_into(input, weight, eps, output);
    }
}

pub(super) fn apply_rms_norm_no_scale_into(input: &[f32], eps: f32, output: &mut [f32]) {
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_sq + eps).powf(-0.5);
    for (dst, src) in output.iter_mut().zip(input.iter()) {
        *dst = *src * inv_rms;
    }
}

pub(super) fn hadamard_inplace(block: &mut [f32]) {
    let n = block.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "hadamard block must be power-of-two");
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
