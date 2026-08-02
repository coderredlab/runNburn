use rnb_core::image::RgbImage;
use rnb_cpu::gemm::f32_gemv::gemv_f32;
use rnb_loader::{GGMLType, LoadedVisionProjector};

use super::vision::{inspect_gemma4_vision_projector, Gemma4VisionError, GEMMA4_VISION_MERGE_SIZE};

pub const GEMMA4_MIN_IMAGE_TOKENS: usize = 40;
pub const GEMMA4_MAX_IMAGE_TOKENS: usize = 280;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4TensorStats {
    pub count: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionIntermediate {
    pub target_width: usize,
    pub target_height: usize,
    pub patch_grid_width: usize,
    pub patch_grid_height: usize,
    pub pooled_grid_width: usize,
    pub pooled_grid_height: usize,
    pub embedding_length: usize,
    pub patch_stats: Gemma4TensorStats,
    pub position_stats: Gemma4TensorStats,
    pub intermediate_stats: Gemma4TensorStats,
    pub patch_embeddings: Vec<f32>,
}

pub fn gemma4_smart_resize(
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<(usize, usize), Gemma4VisionError> {
    smart_resize_with_limits(
        width,
        height,
        patch_size,
        GEMMA4_VISION_MERGE_SIZE,
        GEMMA4_MIN_IMAGE_TOKENS,
        GEMMA4_MAX_IMAGE_TOKENS,
    )
}

pub fn prepare_gemma4_vision_intermediate(
    projector: &LoadedVisionProjector,
    image: &RgbImage,
) -> Result<Gemma4VisionIntermediate, Gemma4VisionError> {
    let capability = inspect_gemma4_vision_projector(&projector.descriptor)?;
    let (target_width, target_height) =
        gemma4_smart_resize(image.width(), image.height(), capability.patch_size)?;
    let resized = resize_bilinear_align_corners(image, target_width, target_height)?;
    let normalized = normalize_rgb(&resized, capability.image_mean, capability.image_std);
    let scaled = normalized
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect::<Vec<_>>();

    let patch_grid_width = target_width / capability.patch_size;
    let patch_grid_height = target_height / capability.patch_size;
    if patch_grid_width > capability.position_table_size
        || patch_grid_height > capability.position_table_size
    {
        return Err(error(format!(
            "Gemma 4 vision grid {patch_grid_width}x{patch_grid_height} exceeds position table {}",
            capability.position_table_size
        )));
    }
    let patch_rows =
        collect_patch_rows(&scaled, target_width, target_height, capability.patch_size)?;
    let patch_width = capability
        .patch_size
        .checked_mul(capability.patch_size)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| error("patch input width overflows usize"))?;
    let patch_count = patch_grid_width
        .checked_mul(patch_grid_height)
        .ok_or_else(|| error("patch count overflows usize"))?;
    let kernel = tensor_f32(projector, "v.patch_embd.weight")?;
    let mut patch_embeddings = vec![0.0f32; patch_count * capability.embedding_length];
    gemv_f32(
        &kernel,
        &patch_rows,
        &mut patch_embeddings,
        capability.embedding_length,
        patch_width,
        patch_count,
    );
    let patch_stats = tensor_stats(&patch_embeddings)?;

    let positions = tensor_f32(projector, "v.position_embd.weight")?;
    add_learned_positions(
        &mut patch_embeddings,
        &positions,
        patch_grid_width,
        patch_grid_height,
        capability.position_table_size,
        capability.embedding_length,
    )?;
    let position_stats = tensor_stats(&positions)?;
    ensure_finite(&patch_embeddings, "Gemma 4 vision intermediate")?;
    let intermediate_stats = tensor_stats(&patch_embeddings)?;

    Ok(Gemma4VisionIntermediate {
        target_width,
        target_height,
        patch_grid_width,
        patch_grid_height,
        pooled_grid_width: patch_grid_width / GEMMA4_VISION_MERGE_SIZE,
        pooled_grid_height: patch_grid_height / GEMMA4_VISION_MERGE_SIZE,
        embedding_length: capability.embedding_length,
        patch_stats,
        position_stats,
        intermediate_stats,
        patch_embeddings,
    })
}

#[allow(clippy::too_many_arguments)]
fn smart_resize_with_limits(
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(usize, usize), Gemma4VisionError> {
    if width == 0 || height == 0 {
        return Err(error("image dimensions must be positive"));
    }
    if patch_size == 0 || merge_size == 0 || min_tokens == 0 || max_tokens < min_tokens {
        return Err(error("invalid Gemma 4 image sizing parameters"));
    }
    let factor = patch_size
        .checked_mul(merge_size)
        .ok_or_else(|| error("image alignment factor overflows usize"))?;
    let area = factor
        .checked_mul(factor)
        .ok_or_else(|| error("image token area overflows usize"))?;
    let min_pixels = min_tokens
        .checked_mul(area)
        .ok_or_else(|| error("minimum image pixel count overflows usize"))?;
    let max_pixels = max_tokens
        .checked_mul(area)
        .ok_or_else(|| error("maximum image pixel count overflows usize"))?;
    let source_pixels = width
        .checked_mul(height)
        .ok_or_else(|| error("source image pixel count overflows usize"))?;

    let round_by = |value: f64| ((value / factor as f64).round() as usize) * factor;
    let ceil_by = |value: f64| ((value / factor as f64).ceil() as usize) * factor;
    let floor_by = |value: f64| ((value / factor as f64).floor() as usize) * factor;
    let mut target_height = round_by(height as f64).max(factor);
    let mut target_width = round_by(width as f64).max(factor);
    let aligned_pixels = target_width
        .checked_mul(target_height)
        .ok_or_else(|| error("aligned image pixel count overflows usize"))?;
    if aligned_pixels > max_pixels {
        let beta = (source_pixels as f64 / max_pixels as f64).sqrt();
        target_height = floor_by(height as f64 / beta).max(factor);
        target_width = floor_by(width as f64 / beta).max(factor);
    } else if aligned_pixels < min_pixels {
        let beta = (min_pixels as f64 / source_pixels as f64).sqrt();
        target_height = ceil_by(height as f64 * beta).max(factor);
        target_width = ceil_by(width as f64 * beta).max(factor);
    }
    Ok((target_width, target_height))
}

fn resize_bilinear_align_corners(
    image: &RgbImage,
    target_width: usize,
    target_height: usize,
) -> Result<RgbImage, Gemma4VisionError> {
    if image.width() == target_width && image.height() == target_height {
        return Ok(image.clone());
    }
    let mut output = vec![0u8; target_width * target_height * 3];
    let x_ratio = if target_width > 1 {
        (image.width() - 1) as f32 / (target_width - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if target_height > 1 {
        (image.height() - 1) as f32 / (target_height - 1) as f32
    } else {
        0.0
    };
    for y in 0..target_height {
        let source_y = y as f32 * y_ratio;
        let y0 = (source_y as usize).min(image.height() - 1);
        let y1 = (y0 + 1).min(image.height() - 1);
        let yf = source_y - y0 as f32;
        for x in 0..target_width {
            let source_x = x as f32 * x_ratio;
            let x0 = (source_x as usize).min(image.width() - 1);
            let x1 = (x0 + 1).min(image.width() - 1);
            let xf = source_x - x0 as f32;
            for channel in 0..3 {
                let p00 = image.pixels()[(y0 * image.width() + x0) * 3 + channel] as f32;
                let p10 = image.pixels()[(y0 * image.width() + x1) * 3 + channel] as f32;
                let p01 = image.pixels()[(y1 * image.width() + x0) * 3 + channel] as f32;
                let p11 = image.pixels()[(y1 * image.width() + x1) * 3 + channel] as f32;
                let top = p00 + (p10 - p00) * xf;
                let bottom = p01 + (p11 - p01) * xf;
                output[(y * target_width + x) * 3 + channel] =
                    (top + (bottom - top) * yf).clamp(0.0, 255.0) as u8;
            }
        }
    }
    RgbImage::new(target_width, target_height, output)
        .map_err(|error| self::error(error.to_string()))
}

fn normalize_rgb(image: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    image
        .pixels()
        .chunks_exact(3)
        .flat_map(|pixel| {
            [
                (pixel[0] as f32 / 255.0 - mean[0]) / std[0],
                (pixel[1] as f32 / 255.0 - mean[1]) / std[1],
                (pixel[2] as f32 / 255.0 - mean[2]) / std[2],
            ]
        })
        .collect()
}

fn collect_patch_rows(
    pixels: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if width % patch_size != 0 || height % patch_size != 0 {
        return Err(error("resized image is not aligned to patch size"));
    }
    let grid_width = width / patch_size;
    let grid_height = height / patch_size;
    let patch_width = patch_size * patch_size * 3;
    let mut rows = vec![0.0f32; grid_width * grid_height * patch_width];
    for patch_y in 0..grid_height {
        for patch_x in 0..grid_width {
            let patch = patch_y * grid_width + patch_x;
            let row = &mut rows[patch * patch_width..(patch + 1) * patch_width];
            let mut destination = 0;
            for channel in 0..3 {
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let image_y = patch_y * patch_size + y;
                        let image_x = patch_x * patch_size + x;
                        row[destination] = pixels[(image_y * width + image_x) * 3 + channel];
                        destination += 1;
                    }
                }
            }
        }
    }
    Ok(rows)
}

fn add_learned_positions(
    hidden: &mut [f32],
    positions: &[f32],
    grid_width: usize,
    grid_height: usize,
    table_size: usize,
    width: usize,
) -> Result<(), Gemma4VisionError> {
    if hidden.len() != grid_width * grid_height * width || positions.len() != 2 * table_size * width
    {
        return Err(error("Gemma 4 learned position buffer shape mismatch"));
    }
    for y in 0..grid_height {
        for x in 0..grid_width {
            let target =
                &mut hidden[(y * grid_width + x) * width..(y * grid_width + x + 1) * width];
            let x_row = &positions[x * width..(x + 1) * width];
            let y_offset = (table_size + y) * width;
            let y_row = &positions[y_offset..y_offset + width];
            for index in 0..width {
                target[index] += x_row[index] + y_row[index];
            }
        }
    }
    Ok(())
}

pub(crate) fn tensor_f32(
    projector: &LoadedVisionProjector,
    name: &str,
) -> Result<Vec<f32>, Gemma4VisionError> {
    let descriptor = projector
        .descriptor
        .tensors
        .get(name)
        .ok_or_else(|| error(format!("missing projector tensor '{name}'")))?;
    if descriptor.ggml_type != GGMLType::F32 {
        return Err(error(format!("tensor '{name}' must be F32")));
    }
    let tensor = projector
        .weights
        .get(name)
        .ok_or_else(|| error(format!("missing mapped projector tensor '{name}'")))?;
    let bytes = tensor
        .as_bytes()
        .ok_or_else(|| error(format!("tensor '{name}' has no contiguous host bytes")))?;
    let expected = descriptor
        .shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| error(format!("tensor '{name}' byte count overflows usize")))?;
    if bytes.len() != expected {
        return Err(error(format!(
            "tensor '{name}' has {} mapped bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 chunk")))
        .collect())
}

pub(crate) fn tensor_stats(values: &[f32]) -> Result<Gemma4TensorStats, Gemma4VisionError> {
    if values.is_empty() {
        return Err(error("cannot compute statistics for an empty tensor"));
    }
    let mut sum = 0.0f64;
    let mut sum_squares = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in values {
        sum += value as f64;
        sum_squares += value as f64 * value as f64;
        min = min.min(value);
        max = max.max(value);
    }
    let mean = sum / values.len() as f64;
    let variance = (sum_squares / values.len() as f64 - mean * mean).max(0.0);
    Ok(Gemma4TensorStats {
        count: values.len(),
        mean,
        stddev: variance.sqrt(),
        min,
        max,
    })
}

pub(crate) fn ensure_finite(values: &[f32], label: &str) -> Result<(), Gemma4VisionError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(error(format!("{label} contains a non-finite value")));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> Gemma4VisionError {
    Gemma4VisionError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_gemma4_alignment_and_limits() {
        assert_eq!(gemma4_smart_resize(224, 224, 16).unwrap(), (336, 336));
        assert_eq!(gemma4_smart_resize(1920, 1080, 16).unwrap(), (1056, 576));
    }

    #[test]
    fn learned_positions_use_x_then_y_tables() {
        let mut hidden = vec![0.0; 4];
        let positions = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        add_learned_positions(&mut hidden, &positions, 2, 1, 2, 2).unwrap();
        assert_eq!(hidden, [11.0, 22.0, 13.0, 24.0]);
    }
}
