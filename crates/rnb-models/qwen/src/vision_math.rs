use rayon::prelude::*;
use rnb_cpu::gemm::f32_gemv::dot_f32_row;

use super::vision::Qwen36VisionError;

const ROPE_FREQUENCY_BASE: f32 = 10_000.0;
const GELU_COEFFICIENT: f32 = 0.044_715;
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

pub(super) fn layer_norm_affine(
    input: &[f32],
    width: usize,
    epsilon: f32,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, Qwen36VisionError> {
    if width == 0 || input.len() % width != 0 {
        return Err(error("LayerNorm input shape is invalid"));
    }
    if weight.len() != width || bias.len() != width {
        return Err(error("LayerNorm affine parameter width mismatch"));
    }
    let mut output = vec![0.0f32; input.len()];
    input
        .par_chunks(width)
        .zip(output.par_chunks_mut(width))
        .for_each(|(source, target)| {
            let sum = source.iter().map(|value| *value as f64).sum::<f64>();
            let mean = (sum / width as f64) as f32;
            let variance = (source
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    (centered * centered) as f64
                })
                .sum::<f64>()
                / width as f64) as f32;
            let scale = 1.0 / (variance + epsilon).sqrt();
            for index in 0..width {
                target[index] = (source[index] - mean) * scale * weight[index] + bias[index];
            }
        });
    Ok(output)
}

pub(super) fn apply_vision_mrope(
    qkv: &mut [f32],
    patch_grid_width: usize,
    patch_grid_height: usize,
    spatial_merge_size: usize,
    embedding_length: usize,
    head_count: usize,
) -> Result<(), Qwen36VisionError> {
    if head_count == 0 || embedding_length % head_count != 0 {
        return Err(error("vision attention head shape is invalid"));
    }
    let head_dim = embedding_length / head_count;
    if head_dim % 4 != 0 {
        return Err(error(
            "vision attention head width must be divisible by four",
        ));
    }
    let patch_count = patch_grid_width
        .checked_mul(patch_grid_height)
        .ok_or_else(|| error("vision patch count overflows usize"))?;
    let qkv_width = embedding_length
        .checked_mul(3)
        .ok_or_else(|| error("vision QKV width overflows usize"))?;
    if qkv.len() != patch_count * qkv_width {
        return Err(error("vision QKV buffer shape mismatch"));
    }
    let rotary_dims = head_dim / 2;
    let section_pairs = head_dim / 4;
    let theta_scale = ROPE_FREQUENCY_BASE.powf(-2.0 / rotary_dims as f32);

    qkv.par_chunks_mut(qkv_width)
        .enumerate()
        .for_each(|(patch_index, row)| {
            let (position_y, position_x) =
                patch_position(patch_index, patch_grid_width, spatial_merge_size);
            for head in 0..head_count {
                let q_offset = head * head_dim;
                let k_offset = embedding_length + head * head_dim;
                for pair in 0..rotary_dims {
                    let (position, exponent) = if pair < section_pairs {
                        (position_y, pair)
                    } else {
                        (position_x, pair - section_pairs)
                    };
                    let angle = position as f32 * theta_scale.powi(exponent as i32);
                    let (sin, cos) = angle.sin_cos();
                    rotate_pair(
                        row,
                        q_offset + pair,
                        q_offset + pair + rotary_dims,
                        sin,
                        cos,
                    );
                    rotate_pair(
                        row,
                        k_offset + pair,
                        k_offset + pair + rotary_dims,
                        sin,
                        cos,
                    );
                }
            }
        });
    Ok(())
}

fn patch_position(
    patch_index: usize,
    patch_grid_width: usize,
    spatial_merge_size: usize,
) -> (usize, usize) {
    let merge_area = spatial_merge_size * spatial_merge_size;
    let tiles_per_row = patch_grid_width / spatial_merge_size;
    let tile = patch_index / merge_area;
    let within_tile = patch_index % merge_area;
    let tile_y = tile / tiles_per_row;
    let tile_x = tile % tiles_per_row;
    let dy = within_tile / spatial_merge_size;
    let dx = within_tile % spatial_merge_size;
    (
        tile_y * spatial_merge_size + dy,
        tile_x * spatial_merge_size + dx,
    )
}

fn rotate_pair(values: &mut [f32], first: usize, second: usize, sin: f32, cos: f32) {
    let x0 = values[first];
    let x1 = values[second];
    values[first] = x0 * cos - x1 * sin;
    values[second] = x0 * sin + x1 * cos;
}

pub(super) fn full_attention(
    qkv: &[f32],
    embedding_length: usize,
    head_count: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Qwen36VisionError> {
    if head_count == 0 || embedding_length % head_count != 0 {
        return Err(error("vision attention head shape is invalid"));
    }
    let head_dim = embedding_length / head_count;
    let qkv_width = embedding_length
        .checked_mul(3)
        .ok_or_else(|| error("vision QKV width overflows usize"))?;
    if qkv.len() != sequence_length * qkv_width {
        return Err(error("vision attention QKV shape mismatch"));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads = (0..head_count)
        .into_par_iter()
        .map(|head| {
            let mut output = vec![0.0f32; sequence_length * head_dim];
            let mut scores = vec![0.0f32; sequence_length];
            for query in 0..sequence_length {
                let query_start = query * qkv_width + head * head_dim;
                let query_values = &qkv[query_start..query_start + head_dim];
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..sequence_length {
                    let key_start = key * qkv_width + embedding_length + head * head_dim;
                    let score = dot_f32_row(
                        query_values,
                        &qkv[key_start..key_start + head_dim],
                        head_dim,
                    ) * scale;
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
                    let value_start = key * qkv_width + 2 * embedding_length + head * head_dim;
                    let value = &qkv[value_start..value_start + head_dim];
                    for dimension in 0..head_dim {
                        target[dimension] += probability * value[dimension];
                    }
                }
            }
            Ok(output)
        })
        .collect::<Result<Vec<_>, Qwen36VisionError>>()?;

    let mut output = vec![0.0f32; sequence_length * embedding_length];
    for (head, values) in heads.iter().enumerate() {
        for token in 0..sequence_length {
            let source = &values[token * head_dim..(token + 1) * head_dim];
            let target_start = token * embedding_length + head * head_dim;
            output[target_start..target_start + head_dim].copy_from_slice(source);
        }
    }
    Ok(output)
}

pub(super) fn gelu_in_place(values: &mut [f32]) {
    values.par_iter_mut().for_each(|value| {
        let x = *value;
        *value = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEFFICIENT * x * x)).tanh());
    });
}

pub(super) fn add_in_place(target: &mut [f32], source: &[f32]) -> Result<(), Qwen36VisionError> {
    if target.len() != source.len() {
        return Err(error("vision residual shape mismatch"));
    }
    target
        .par_iter_mut()
        .zip(source.par_iter())
        .for_each(|(target, source)| *target += *source);
    Ok(())
}

pub(super) fn ensure_finite(values: &[f32], label: &str) -> Result<(), Qwen36VisionError> {
    if values.par_iter().any(|value| !value.is_finite()) {
        return Err(error(format!("{label} contains a non-finite value")));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> Qwen36VisionError {
    Qwen36VisionError::new(message)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_vision_mrope, full_attention, gelu_in_place, layer_norm_affine, patch_position,
    };

    #[test]
    fn patch_positions_follow_spatial_merge_order() {
        let positions: Vec<_> = (0..8).map(|index| patch_position(index, 4, 2)).collect();
        assert_eq!(
            positions,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }

    #[test]
    fn layer_norm_applies_affine_parameters_per_row() {
        let output =
            layer_norm_affine(&[1.0, 2.0, 3.0, 4.0], 2, 0.0, &[2.0, 3.0], &[0.5, -0.5]).unwrap();
        assert_eq!(output, [-1.5, 2.5, -1.5, 2.5]);
    }

    #[test]
    fn vision_mrope_uses_height_then_width_sections() {
        let hidden = 8;
        let mut qkv = vec![0.0f32; 4 * hidden * 3];
        let row = &mut qkv[2 * hidden * 3..3 * hidden * 3];
        row[0] = 1.0;
        row[4] = 2.0;
        row[hidden] = 3.0;
        row[hidden + 4] = 4.0;
        row[2] = 5.0;
        row[6] = 6.0;

        apply_vision_mrope(&mut qkv, 2, 2, 2, hidden, 1).unwrap();

        let row = &qkv[2 * hidden * 3..3 * hidden * 3];
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((row[0] - (cos - 2.0 * sin)).abs() < 1e-6);
        assert!((row[4] - (sin + 2.0 * cos)).abs() < 1e-6);
        assert!((row[hidden] - (3.0 * cos - 4.0 * sin)).abs() < 1e-6);
        assert!((row[hidden + 4] - (3.0 * sin + 4.0 * cos)).abs() < 1e-6);
        assert_eq!(row[2], 5.0);
        assert_eq!(row[6], 6.0);
    }

    #[test]
    fn attention_softmax_is_stable_for_large_logits() {
        let qkv = [
            1000.0, 0.0, 1000.0, 0.0, 1.0, 2.0, 0.0, 1000.0, 0.0, 1000.0, 3.0, 4.0,
        ];
        let output = full_attention(&qkv, 2, 1, 2).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gelu_matches_ggml_tanh_formula() {
        let mut values = [-1.0f32, 0.0, 1.0];
        gelu_in_place(&mut values);
        assert!((values[0] - -0.158_808).abs() < 1e-6);
        assert_eq!(values[1], 0.0);
        assert!((values[2] - 0.841_192).abs() < 1e-6);
    }
}
