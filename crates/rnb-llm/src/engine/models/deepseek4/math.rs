use crate::engine::cpu_runtime::kernels;
use crate::engine::dense_dispatch::gemv_f32;
use rnb_core::tensor::Tensor;

use super::weights::{DeepSeek4Config, HyperConnectionWeights};

#[inline]
pub(super) fn tensor_f32(tensor: &Tensor) -> &[f32] {
    kernels::tensor_as_f32_slice(tensor)
}

pub(super) fn rms_norm(input: &[f32], weight: &Tensor, eps: f32) -> Vec<f32> {
    let weight = tensor_f32(weight);
    debug_assert_eq!(input.len(), weight.len());
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(&value, &gain)| value * scale * gain)
        .collect()
}

pub(super) fn rms_unit_inplace(values: &mut [f32], eps: f32) {
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(super) struct HyperConnectionMix {
    pub(super) branch: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

pub(super) fn hyper_pre(
    hidden: &[f32],
    weights: &HyperConnectionWeights,
    config: &DeepSeek4Config,
) -> HyperConnectionMix {
    let hc = config.hc_count;
    let hidden_dim = config.hidden_dim;
    debug_assert_eq!(hidden.len(), hc * hidden_dim);
    let rms = (hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32
        + config.norm_eps)
        .sqrt()
        .recip();
    let mut mixes = vec![0.0f32; hc * (hc + 2)];
    let mix_count = mixes.len();
    gemv_f32(
        tensor_f32(&weights.function),
        hidden,
        &mut mixes,
        mix_count,
        hidden.len(),
        1,
    );
    for value in &mut mixes {
        *value *= rms;
    }

    let scale = tensor_f32(&weights.scale);
    let base = tensor_f32(&weights.base);
    let pre: Vec<f32> = (0..hc)
        .map(|index| sigmoid(mixes[index] * scale[0] + base[index]) + config.hc_eps)
        .collect();
    let post: Vec<f32> = (0..hc)
        .map(|index| 2.0 * sigmoid(mixes[hc + index] * scale[1] + base[hc + index]))
        .collect();

    let mut comb = vec![0.0f32; hc * hc];
    for row in 0..hc {
        let start = 2 * hc + row * hc;
        let max_value = (0..hc)
            .map(|col| mixes[start + col] * scale[2] + base[start + col])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for col in 0..hc {
            let value = (mixes[start + col] * scale[2] + base[start + col] - max_value).exp();
            comb[row * hc + col] = value;
            sum += value;
        }
        for col in 0..hc {
            comb[row * hc + col] = comb[row * hc + col] / sum + config.hc_eps;
        }
    }
    normalize_columns(&mut comb, hc, config.hc_eps);
    for _ in 1..config.sinkhorn_iterations {
        normalize_rows(&mut comb, hc, config.hc_eps);
        normalize_columns(&mut comb, hc, config.hc_eps);
    }

    let mut branch = vec![0.0f32; hidden_dim];
    for copy in 0..hc {
        let row = &hidden[copy * hidden_dim..(copy + 1) * hidden_dim];
        for (dst, &value) in branch.iter_mut().zip(row) {
            *dst += pre[copy] * value;
        }
    }
    HyperConnectionMix { branch, post, comb }
}

fn normalize_rows(matrix: &mut [f32], size: usize, eps: f32) {
    for row in 0..size {
        let start = row * size;
        let sum = matrix[start..start + size].iter().sum::<f32>();
        for value in &mut matrix[start..start + size] {
            *value /= sum + eps;
        }
    }
}

fn normalize_columns(matrix: &mut [f32], size: usize, eps: f32) {
    for col in 0..size {
        let sum = (0..size).map(|row| matrix[row * size + col]).sum::<f32>();
        for row in 0..size {
            matrix[row * size + col] /= sum + eps;
        }
    }
}

pub(super) fn hyper_post(
    branch_output: &[f32],
    residual: &[f32],
    mix: HyperConnectionMix,
    config: &DeepSeek4Config,
) -> Vec<f32> {
    let hc = config.hc_count;
    let dim = config.hidden_dim;
    let mut output = vec![0.0f32; hc * dim];
    for row in 0..hc {
        for feature in 0..dim {
            let mut value = mix.post[row] * branch_output[feature];
            for col in 0..hc {
                value += mix.comb[col * hc + row] * residual[col * dim + feature];
            }
            output[row * dim + feature] = value;
        }
    }
    output
}

pub(super) fn hyper_head(
    hidden: &[f32],
    function: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    config: &DeepSeek4Config,
) -> Vec<f32> {
    let hc = config.hc_count;
    let dim = config.hidden_dim;
    let rms = (hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32
        + config.norm_eps)
        .sqrt()
        .recip();
    let mut mixes = vec![0.0f32; hc];
    gemv_f32(
        tensor_f32(function),
        hidden,
        &mut mixes,
        hc,
        hidden.len(),
        1,
    );
    let scale = tensor_f32(scale)[0];
    let base = tensor_f32(base);
    let pre: Vec<f32> = mixes
        .iter()
        .zip(base)
        .map(|(&mix, &bias)| sigmoid(mix * rms * scale + bias) + config.hc_eps)
        .collect();
    let mut output = vec![0.0f32; dim];
    for copy in 0..hc {
        for feature in 0..dim {
            output[feature] += pre[copy] * hidden[copy * dim + feature];
        }
    }
    output
}

pub(super) fn apply_rope(
    values: &mut [f32],
    position: usize,
    config: &DeepSeek4Config,
    compressed_layer: bool,
    inverse: bool,
) {
    let rope_dim = config.rope_dim;
    debug_assert_eq!(values.len(), rope_dim);
    let base = if compressed_layer {
        config.compress_rope_theta
    } else {
        config.rope_theta
    };
    let use_yarn = compressed_layer && config.rope_original_context_length > 0;
    let correction = if use_yarn {
        let correction_dim = |rotations: f32| {
            rope_dim as f32
                * (config.rope_original_context_length as f32
                    / (rotations * 2.0 * std::f32::consts::PI))
                    .ln()
                / (2.0 * base.ln())
        };
        let low = correction_dim(config.rope_yarn_beta_fast).floor().max(0.0);
        let high = correction_dim(config.rope_yarn_beta_slow)
            .ceil()
            .min((rope_dim - 1) as f32);
        Some((low, if low == high { high + 0.001 } else { high }))
    } else {
        None
    };
    for pair in 0..rope_dim / 2 {
        let exponent = (2 * pair) as f32 / rope_dim as f32;
        let base_freq = 1.0 / base.powf(exponent);
        let freq = if let Some((low, high)) = correction {
            let ramp = ((pair as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            base_freq / config.rope_factor * (1.0 - smooth) + base_freq * smooth
        } else {
            base_freq
        };
        let angle = position as f32 * freq * if inverse { -1.0 } else { 1.0 };
        let (sin, cos) = angle.sin_cos();
        let real = values[2 * pair];
        let imag = values[2 * pair + 1];
        values[2 * pair] = real * cos - imag * sin;
        values[2 * pair + 1] = real * sin + imag * cos;
    }
}

pub(super) fn hadamard_inplace(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut width = 1;
    while width < values.len() {
        for start in (0..values.len()).step_by(width * 2) {
            for offset in 0..width {
                let a = values[start + offset];
                let b = values[start + width + offset];
                values[start + offset] = a + b;
                values[start + width + offset] = a - b;
            }
        }
        width *= 2;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}

pub(super) fn fp8_quantize_inplace(values: &mut [f32], block_size: usize) {
    for block in values.chunks_mut(block_size) {
        let amax = block
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(1e-4);
        let scale = 2.0f32.powf((amax / 448.0).log2().ceil());
        for value in block {
            *value = nearest_fp8(*value / scale) * scale;
        }
    }
}

fn nearest_fp8(value: f32) -> f32 {
    let sign = value.signum();
    let target = value.abs().min(448.0);
    let mut best = 0.0f32;
    let mut best_distance = target;
    for mantissa in 1..=7 {
        let candidate = mantissa as f32 * 2.0f32.powi(-9);
        let distance = (target - candidate).abs();
        if distance < best_distance {
            best = candidate;
            best_distance = distance;
        }
    }
    for exponent in 1..=15 {
        let max_mantissa = if exponent == 15 { 6 } else { 7 };
        for mantissa in 0..=max_mantissa {
            let candidate = (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent - 7);
            let distance = (target - candidate).abs();
            if distance < best_distance {
                best = candidate;
                best_distance = distance;
            }
        }
    }
    sign * best
}

pub(super) fn fp4_quantize_inplace(values: &mut [f32], block_size: usize) {
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for block in values.chunks_mut(block_size) {
        let amax = block
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(6.0 * 2.0f32.powi(-126));
        let scale = 2.0f32.powf((amax / 6.0).log2().ceil());
        for value in block {
            let sign = value.signum();
            let target = (value.abs() / scale).min(6.0);
            let nearest = LEVELS
                .iter()
                .copied()
                .min_by(|left, right| {
                    (target - *left)
                        .abs()
                        .partial_cmp(&(target - *right).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            *value = sign * nearest * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{hyper_post, DeepSeek4Config, HyperConnectionMix};

    #[test]
    fn hyper_post_reads_comb_as_source_by_destination() {
        let config = DeepSeek4Config {
            hidden_dim: 1,
            num_heads: 1,
            head_dim: 1,
            rope_dim: 0,
            output_groups: 1,
            output_lora_rank: 1,
            window_size: 1,
            index_heads: 1,
            index_head_dim: 1,
            index_topk: 1,
            hc_count: 2,
            sinkhorn_iterations: 1,
            hc_eps: 0.0,
            norm_eps: 0.0,
            expert_count: 1,
            expert_used_count: 1,
            expert_ffn_dim: 1,
            expert_scale: 1.0,
            rope_theta: 1.0,
            compress_rope_theta: 1.0,
            rope_factor: 1.0,
            rope_original_context_length: 0,
            rope_yarn_beta_fast: 1.0,
            rope_yarn_beta_slow: 1.0,
        };
        let mix = HyperConnectionMix {
            branch: Vec::new(),
            post: vec![0.0, 0.0],
            // Flattened as comb[source * hc + destination].
            comb: vec![1.0, 2.0, 3.0, 4.0],
        };

        assert_eq!(
            hyper_post(&[0.0], &[10.0, 20.0], mix, &config),
            vec![70.0, 100.0]
        );
    }
}
