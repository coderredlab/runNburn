use std::mem;

use rnb_cpu::gemm::f32_gemv::gemv_bf16;
use rnb_loader::{GGMLType, LoadedVisionProjector};

use super::vision::{inspect_gemma4_vision_projector, Gemma4VisionError, GEMMA4_VISION_MERGE_SIZE};
use super::vision_math::{
    add_in_place, full_attention, geglu_quick_in_place, normalize_and_rotate_qkv, rms_norm,
    rms_norm_affine,
};
use super::vision_preprocess::{
    ensure_finite, tensor_f32, tensor_stats, Gemma4TensorStats, Gemma4VisionIntermediate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionLayerSummary {
    pub layer_index: usize,
    pub stats: Gemma4TensorStats,
    pub first_values: [f32; 8],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionOutput {
    pub target_width: usize,
    pub target_height: usize,
    pub patch_grid_width: usize,
    pub patch_grid_height: usize,
    pub pooled_grid_width: usize,
    pub pooled_grid_height: usize,
    pub projection_dim: usize,
    pub layer_summaries: Vec<Gemma4VisionLayerSummary>,
    pub pooled_stats: Gemma4TensorStats,
    pub embedding_stats: Gemma4TensorStats,
    pub embeddings: Vec<f32>,
}

pub fn encode_gemma4_vision_intermediate(
    projector: &LoadedVisionProjector,
    intermediate: Gemma4VisionIntermediate,
) -> Result<Gemma4VisionOutput, Gemma4VisionError> {
    let capability = inspect_gemma4_vision_projector(&projector.descriptor)?;
    validate_intermediate(&intermediate, capability.embedding_length)?;
    let token_count = intermediate.patch_grid_width * intermediate.patch_grid_height;
    let mut hidden = intermediate.patch_embeddings;
    let mut layer_summaries = Vec::with_capacity(capability.block_count);

    for layer in 0..capability.block_count {
        let prefix = format!("v.blk.{layer}");
        let ln1 = tensor_f32(projector, &format!("{prefix}.ln1.weight"))?;
        let normalized = rms_norm_affine(
            &hidden,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &ln1,
        )?;
        let mut q = linear_bf16(
            projector,
            &format!("{prefix}.attn_q.weight"),
            &normalized,
            capability.embedding_length,
            capability.embedding_length,
            token_count,
        )?;
        let mut k = linear_bf16(
            projector,
            &format!("{prefix}.attn_k.weight"),
            &normalized,
            capability.embedding_length,
            capability.embedding_length,
            token_count,
        )?;
        let mut v = linear_bf16(
            projector,
            &format!("{prefix}.attn_v.weight"),
            &normalized,
            capability.embedding_length,
            capability.embedding_length,
            token_count,
        )?;
        let q_norm = tensor_f32(projector, &format!("{prefix}.attn_q_norm.weight"))?;
        let k_norm = tensor_f32(projector, &format!("{prefix}.attn_k_norm.weight"))?;
        normalize_and_rotate_qkv(
            &mut q,
            &mut k,
            &mut v,
            intermediate.patch_grid_width,
            intermediate.patch_grid_height,
            capability.embedding_length,
            capability.head_count,
            capability.layer_norm_epsilon,
            &q_norm,
            &k_norm,
        )?;
        let attended = full_attention(
            &q,
            &k,
            &v,
            capability.embedding_length,
            capability.head_count,
            token_count,
        )?;
        let attention = linear_bf16(
            projector,
            &format!("{prefix}.attn_out.weight"),
            &attended,
            capability.embedding_length,
            capability.embedding_length,
            token_count,
        )?;
        let post_attention = tensor_f32(projector, &format!("{prefix}.attn_post_norm.weight"))?;
        let attention = rms_norm_affine(
            &attention,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &post_attention,
        )?;
        add_in_place(&mut hidden, &attention)?;

        let ln2 = tensor_f32(projector, &format!("{prefix}.ln2.weight"))?;
        let normalized = rms_norm_affine(
            &hidden,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &ln2,
        )?;
        let up = linear_bf16(
            projector,
            &format!("{prefix}.ffn_up.weight"),
            &normalized,
            capability.feed_forward_length,
            capability.embedding_length,
            token_count,
        )?;
        let mut gate = linear_bf16(
            projector,
            &format!("{prefix}.ffn_gate.weight"),
            &normalized,
            capability.feed_forward_length,
            capability.embedding_length,
            token_count,
        )?;
        geglu_quick_in_place(&mut gate, &up)?;
        let feed_forward = linear_bf16(
            projector,
            &format!("{prefix}.ffn_down.weight"),
            &gate,
            capability.embedding_length,
            capability.feed_forward_length,
            token_count,
        )?;
        let post_ffn = tensor_f32(projector, &format!("{prefix}.ffn_post_norm.weight"))?;
        let feed_forward = rms_norm_affine(
            &feed_forward,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &post_ffn,
        )?;
        add_in_place(&mut hidden, &feed_forward)?;
        ensure_finite(&hidden, &format!("Gemma 4 vision block {layer} output"))?;

        let mut first_values = [0.0f32; 8];
        first_values.copy_from_slice(&hidden[..8]);
        layer_summaries.push(Gemma4VisionLayerSummary {
            layer_index: layer,
            stats: tensor_stats(&hidden)?,
            first_values,
        });
    }

    let mut pooled = average_pool_2d(
        &hidden,
        intermediate.patch_grid_width,
        intermediate.patch_grid_height,
        capability.embedding_length,
        GEMMA4_VISION_MERGE_SIZE,
    )?;
    let pool_scale = (capability.embedding_length as f32).sqrt();
    pooled.iter_mut().for_each(|value| *value *= pool_scale);
    if projector.descriptor.tensors.contains_key("v.std_bias") {
        let std_bias = tensor_f32(projector, "v.std_bias")?;
        let std_scale = tensor_f32(projector, "v.std_scale")?;
        for row in pooled.chunks_mut(capability.embedding_length) {
            for index in 0..capability.embedding_length {
                row[index] = (row[index] - std_bias[index]) * std_scale[index];
            }
        }
    }
    let pooled = rms_norm(
        &pooled,
        capability.embedding_length,
        capability.layer_norm_epsilon,
    )?;
    let pooled_stats = tensor_stats(&pooled)?;
    let pooled_count = intermediate.pooled_grid_width * intermediate.pooled_grid_height;
    let embeddings = linear_bf16(
        projector,
        "mm.input_projection.weight",
        &pooled,
        capability.projection_dim,
        capability.embedding_length,
        pooled_count,
    )?;
    ensure_finite(&embeddings, "Gemma 4 vision projection")?;
    let embedding_stats = tensor_stats(&embeddings)?;

    Ok(Gemma4VisionOutput {
        target_width: intermediate.target_width,
        target_height: intermediate.target_height,
        patch_grid_width: intermediate.patch_grid_width,
        patch_grid_height: intermediate.patch_grid_height,
        pooled_grid_width: intermediate.pooled_grid_width,
        pooled_grid_height: intermediate.pooled_grid_height,
        projection_dim: capability.projection_dim,
        layer_summaries,
        pooled_stats,
        embedding_stats,
        embeddings,
    })
}

fn validate_intermediate(
    intermediate: &Gemma4VisionIntermediate,
    embedding_length: usize,
) -> Result<(), Gemma4VisionError> {
    if intermediate.embedding_length != embedding_length {
        return Err(error(format!(
            "Gemma 4 vision intermediate width is {}, expected {embedding_length}",
            intermediate.embedding_length
        )));
    }
    if intermediate.patch_grid_width % GEMMA4_VISION_MERGE_SIZE != 0
        || intermediate.patch_grid_height % GEMMA4_VISION_MERGE_SIZE != 0
    {
        return Err(error(
            "Gemma 4 vision patch grid is not divisible by pool size",
        ));
    }
    let expected = intermediate
        .patch_grid_width
        .checked_mul(intermediate.patch_grid_height)
        .and_then(|count| count.checked_mul(embedding_length))
        .ok_or_else(|| error("Gemma 4 vision intermediate size overflows usize"))?;
    if intermediate.patch_embeddings.len() != expected {
        return Err(error(format!(
            "Gemma 4 vision intermediate has {} values, expected {expected}",
            intermediate.patch_embeddings.len()
        )));
    }
    Ok(())
}

fn average_pool_2d(
    input: &[f32],
    grid_width: usize,
    grid_height: usize,
    width: usize,
    kernel: usize,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if grid_width % kernel != 0 || grid_height % kernel != 0 {
        return Err(error(
            "Gemma 4 vision pooling grid is not divisible by kernel",
        ));
    }
    let output_width = grid_width / kernel;
    let output_height = grid_height / kernel;
    let mut output = vec![0.0f32; output_width * output_height * width];
    let inverse_area = 1.0 / (kernel * kernel) as f32;
    for output_y in 0..output_height {
        for output_x in 0..output_width {
            let target = &mut output[(output_y * output_width + output_x) * width
                ..(output_y * output_width + output_x + 1) * width];
            for dy in 0..kernel {
                for dx in 0..kernel {
                    let source_y = output_y * kernel + dy;
                    let source_x = output_x * kernel + dx;
                    let source = &input[(source_y * grid_width + source_x) * width
                        ..(source_y * grid_width + source_x + 1) * width];
                    for index in 0..width {
                        target[index] += source[index] * inverse_area;
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn linear_bf16(
    projector: &LoadedVisionProjector,
    weight_name: &str,
    input: &[f32],
    rows: usize,
    cols: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if input.len() != cols * sequence_length {
        return Err(error(format!(
            "tensor '{weight_name}' input shape mismatch"
        )));
    }
    let clamp = clippable_bounds(projector, weight_name)?;
    let clamped_input = clamp.map(|(input_min, input_max, _, _)| {
        input
            .iter()
            .map(|value| value.clamp(input_min, input_max))
            .collect::<Vec<_>>()
    });
    let gemv_input = clamped_input.as_deref().unwrap_or(input);
    let weight = tensor_bf16_words(projector, weight_name, rows, cols)?;
    let mut output = vec![0.0f32; rows * sequence_length];
    gemv_bf16(weight, gemv_input, &mut output, rows, cols, sequence_length);
    if let Some((_, _, output_min, output_max)) = clamp {
        output
            .iter_mut()
            .for_each(|value| *value = value.clamp(output_min, output_max));
    }
    Ok(output)
}

fn clippable_bounds(
    projector: &LoadedVisionProjector,
    weight_name: &str,
) -> Result<Option<(f32, f32, f32, f32)>, Gemma4VisionError> {
    let prefix = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| error(format!("invalid weight tensor name '{weight_name}'")))?;
    let input_min_name = format!("{prefix}.input_min");
    if !projector.descriptor.tensors.contains_key(&input_min_name) {
        return Ok(None);
    }
    let input_max_name = format!("{prefix}.input_max");
    let output_min_name = format!("{prefix}.output_min");
    let output_max_name = format!("{prefix}.output_max");
    let input_min = tensor_f32(projector, &input_min_name)?[0];
    let input_max = tensor_f32(projector, &input_max_name)?[0];
    let output_min = tensor_f32(projector, &output_min_name)?[0];
    let output_max = tensor_f32(projector, &output_max_name)?[0];
    if [input_min, input_max, output_min, output_max]
        .iter()
        .any(|value| !value.is_finite())
        || input_min > input_max
        || output_min > output_max
    {
        return Err(error(format!(
            "tensor '{weight_name}' clamp bounds must be finite and ordered"
        )));
    }
    Ok(Some((input_min, input_max, output_min, output_max)))
}

fn tensor_bf16_words<'a>(
    projector: &'a LoadedVisionProjector,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<&'a [u16], Gemma4VisionError> {
    if cfg!(target_endian = "big") {
        return Err(error(
            "BF16 vision execution requires a little-endian target",
        ));
    }
    let descriptor = projector
        .descriptor
        .tensors
        .get(name)
        .ok_or_else(|| error(format!("missing projector tensor '{name}'")))?;
    if descriptor.ggml_type != GGMLType::BF16 {
        return Err(error(format!("tensor '{name}' must be BF16")));
    }
    let expected_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| error(format!("tensor '{name}' element count overflows usize")))?;
    let tensor = projector
        .weights
        .get(name)
        .ok_or_else(|| error(format!("missing mapped projector tensor '{name}'")))?;
    let bytes = tensor
        .as_bytes()
        .ok_or_else(|| error(format!("tensor '{name}' has no contiguous host bytes")))?;
    let expected_bytes = expected_elements
        .checked_mul(mem::size_of::<u16>())
        .ok_or_else(|| error(format!("tensor '{name}' byte count overflows usize")))?;
    if bytes.len() != expected_bytes {
        return Err(error(format!(
            "tensor '{name}' has {} mapped bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }
    if bytes.as_ptr().align_offset(mem::align_of::<u16>()) != 0 {
        return Err(error(format!(
            "tensor '{name}' BF16 bytes are not u16-aligned"
        )));
    }
    // SAFETY: the mapped bytes have exact u16 size and alignment, checked above.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), expected_elements) })
}

fn error(message: impl Into<String>) -> Gemma4VisionError {
    Gemma4VisionError::new(message)
}

#[cfg(test)]
mod tests {
    use super::average_pool_2d;

    #[test]
    fn pooling_averages_non_overlapping_three_by_three_windows() {
        let input = (0..18).map(|value| value as f32).collect::<Vec<_>>();
        let pooled = average_pool_2d(&input, 6, 3, 1, 3).unwrap();
        assert!((pooled[0] - 7.0).abs() < 1e-6);
        assert!((pooled[1] - 10.0).abs() < 2e-6);
    }
}
