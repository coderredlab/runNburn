use rayon::prelude::*;
use rnb_cpu::gemm::f32_gemv::dot_f32_row;

use super::vision::{Gemma4VisionError, GEMMA4_VISION_ROPE_THETA};

pub(crate) fn layer_norm_affine(
    input: &[f32],
    width: usize,
    epsilon: f32,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, Gemma4VisionError> {
    if width == 0 || input.len() % width != 0 || weight.len() != width || bias.len() != width {
        return Err(error("LayerNorm input, weight, or bias shape is invalid"));
    }
    let mut output = vec![0.0f32; input.len()];
    input
        .par_chunks(width)
        .zip(output.par_chunks_mut(width))
        .for_each(|(source, target)| {
            let mean = source.iter().map(|value| *value as f64).sum::<f64>() / width as f64;
            let variance = source
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / width as f64;
            let scale = 1.0 / (variance as f32 + epsilon).sqrt();
            for index in 0..width {
                target[index] = (source[index] - mean as f32) * scale * weight[index] + bias[index];
            }
        });
    Ok(output)
}

pub(crate) fn rms_norm_affine(
    input: &[f32],
    width: usize,
    epsilon: f32,
    weight: &[f32],
) -> Result<Vec<f32>, Gemma4VisionError> {
    if width == 0 || input.len() % width != 0 || weight.len() != width {
        return Err(error("RMSNorm input or weight shape is invalid"));
    }
    let mut output = vec![0.0f32; input.len()];
    input
        .par_chunks(width)
        .zip(output.par_chunks_mut(width))
        .for_each(|(source, target)| {
            let mean_square = source
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                / width as f64;
            let scale = 1.0 / (mean_square as f32 + epsilon).sqrt();
            for index in 0..width {
                target[index] = source[index] * scale * weight[index];
            }
        });
    Ok(output)
}

pub(crate) fn rms_norm(
    input: &[f32],
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if width == 0 || input.len() % width != 0 {
        return Err(error("RMSNorm input shape is invalid"));
    }
    let mut output = vec![0.0f32; input.len()];
    input
        .par_chunks(width)
        .zip(output.par_chunks_mut(width))
        .for_each(|(source, target)| {
            let mean_square = source
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                / width as f64;
            let scale = 1.0 / (mean_square as f32 + epsilon).sqrt();
            for index in 0..width {
                target[index] = source[index] * scale;
            }
        });
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_and_rotate_qkv(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    grid_width: usize,
    grid_height: usize,
    embedding_length: usize,
    head_count: usize,
    epsilon: f32,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
) -> Result<(), Gemma4VisionError> {
    if head_count == 0 || embedding_length % head_count != 0 {
        return Err(error("vision attention head shape is invalid"));
    }
    let head_dim = embedding_length / head_count;
    if head_dim % 4 != 0 || q_norm_weight.len() != head_dim || k_norm_weight.len() != head_dim {
        return Err(error("Gemma 4 vision Q/K normalization shape is invalid"));
    }
    let token_count = grid_width
        .checked_mul(grid_height)
        .ok_or_else(|| error("vision token count overflows usize"))?;
    let expected = token_count
        .checked_mul(embedding_length)
        .ok_or_else(|| error("vision Q/K/V size overflows usize"))?;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(error("vision Q/K/V buffer shape mismatch"));
    }

    q.par_chunks_mut(embedding_length)
        .zip(k.par_chunks_mut(embedding_length))
        .zip(v.par_chunks_mut(embedding_length))
        .enumerate()
        .for_each(|(token, ((q_row, k_row), v_row))| {
            let x = token % grid_width;
            let y = token / grid_width;
            for head in 0..head_count {
                let start = head * head_dim;
                let end = start + head_dim;
                rms_norm_row_in_place(&mut q_row[start..end], epsilon, Some(q_norm_weight));
                rms_norm_row_in_place(&mut k_row[start..end], epsilon, Some(k_norm_weight));
                rms_norm_row_in_place(&mut v_row[start..end], epsilon, None);
                apply_2d_neox_rope(&mut q_row[start..end], x, y);
                apply_2d_neox_rope(&mut k_row[start..end], x, y);
            }
        });
    Ok(())
}

fn rms_norm_row_in_place(row: &mut [f32], epsilon: f32, weight: Option<&[f32]>) {
    let mean_square = row
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        / row.len() as f64;
    let scale = 1.0 / (mean_square as f32 + epsilon).sqrt();
    match weight {
        Some(weight) => row
            .iter_mut()
            .zip(weight.iter())
            .for_each(|(value, weight)| *value *= scale * *weight),
        None => row.iter_mut().for_each(|value| *value *= scale),
    }
}

fn apply_2d_neox_rope(row: &mut [f32], x: usize, y: usize) {
    let half = row.len() / 2;
    apply_neox_rope(&mut row[..half], x);
    apply_neox_rope(&mut row[half..], y);
}

fn apply_neox_rope(values: &mut [f32], position: usize) {
    let half = values.len() / 2;
    for pair in 0..half {
        let frequency = GEMMA4_VISION_ROPE_THETA.powf(-2.0 * pair as f32 / values.len() as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let first = values[pair];
        let second = values[pair + half];
        values[pair] = first * cos - second * sin;
        values[pair + half] = first * sin + second * cos;
    }
}

pub(crate) fn full_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    embedding_length: usize,
    head_count: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if head_count == 0 || embedding_length % head_count != 0 {
        return Err(error("vision attention head shape is invalid"));
    }
    let expected = sequence_length
        .checked_mul(embedding_length)
        .ok_or_else(|| error("vision attention size overflows usize"))?;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(error("vision attention Q/K/V shape mismatch"));
    }
    let head_dim = embedding_length / head_count;
    let heads = (0..head_count)
        .into_par_iter()
        .map(|head| {
            let mut output = vec![0.0f32; sequence_length * head_dim];
            let mut scores = vec![0.0f32; sequence_length];
            for query in 0..sequence_length {
                let query_start = query * embedding_length + head * head_dim;
                let query_values = &q[query_start..query_start + head_dim];
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..sequence_length {
                    let key_start = key * embedding_length + head * head_dim;
                    let score =
                        dot_f32_row(query_values, &k[key_start..key_start + head_dim], head_dim);
                    scores[key] = score;
                    maximum = maximum.max(score);
                }
                let mut sum = 0.0f64;
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                    sum += *score as f64;
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(error("vision attention softmax normalization is invalid"));
                }
                let inverse_sum = (1.0 / sum) as f32;
                let target = &mut output[query * head_dim..(query + 1) * head_dim];
                for key in 0..sequence_length {
                    let probability = scores[key] * inverse_sum;
                    let value_start = key * embedding_length + head * head_dim;
                    let value = &v[value_start..value_start + head_dim];
                    for dimension in 0..head_dim {
                        target[dimension] += probability * value[dimension];
                    }
                }
            }
            Ok(output)
        })
        .collect::<Result<Vec<_>, Gemma4VisionError>>()?;

    let mut output = vec![0.0f32; expected];
    for (head, values) in heads.iter().enumerate() {
        for token in 0..sequence_length {
            let source = &values[token * head_dim..(token + 1) * head_dim];
            let target_start = token * embedding_length + head * head_dim;
            output[target_start..target_start + head_dim].copy_from_slice(source);
        }
    }
    Ok(output)
}

pub(crate) fn geglu_quick_in_place(gate: &mut [f32], up: &[f32]) -> Result<(), Gemma4VisionError> {
    if gate.len() != up.len() {
        return Err(error("Gemma 4 vision GeGLU shape mismatch"));
    }
    gate.par_iter_mut()
        .zip(up.par_iter())
        .for_each(|(gate, up)| {
            let activated = *gate / (1.0 + (-1.702 * *gate).exp());
            *gate = activated * *up;
        });
    Ok(())
}

pub(crate) fn add_in_place(target: &mut [f32], source: &[f32]) -> Result<(), Gemma4VisionError> {
    if target.len() != source.len() {
        return Err(error("vision residual shape mismatch"));
    }
    target
        .par_iter_mut()
        .zip(source.par_iter())
        .for_each(|(target, source)| *target += *source);
    Ok(())
}

fn error(message: impl Into<String>) -> Gemma4VisionError {
    Gemma4VisionError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_dimensional_rope_uses_x_then_y_halves() {
        let mut row = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        apply_2d_neox_rope(&mut row, 1, 0);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((row[0] - (cos - 2.0 * sin)).abs() < 1e-6);
        assert!((row[2] - (sin + 2.0 * cos)).abs() < 1e-6);
        assert_eq!(&row[4..], &[3.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn quick_geglu_multiplies_activated_gate_by_up() {
        let mut gate = [-1.0, 0.0, 1.0];
        geglu_quick_in_place(&mut gate, &[2.0, 3.0, 4.0]).unwrap();
        assert!((gate[0] - -0.308_409).abs() < 1e-5);
        assert_eq!(gate[1], 0.0);
        assert!((gate[2] - 3.383_182).abs() < 1e-5);
    }
}
