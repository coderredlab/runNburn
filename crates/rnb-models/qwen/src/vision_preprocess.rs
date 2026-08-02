use rnb_core::image::RgbImage;
use rnb_cpu::gemm::f32_gemv::gemv_f32;
use rnb_loader::{GGMLType, LoadedVisionProjector};

use super::vision::{inspect_qwen36_vision_projector, Qwen36VisionError};

pub const QWEN36_MIN_IMAGE_TOKENS: usize = 8;
pub const QWEN36_MAX_IMAGE_TOKENS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen36TensorStats {
    pub count: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36VisionIntermediate {
    pub target_width: usize,
    pub target_height: usize,
    pub patch_grid_width: usize,
    pub patch_grid_height: usize,
    pub merged_grid_width: usize,
    pub merged_grid_height: usize,
    pub embedding_length: usize,
    pub normalized_stats: Qwen36TensorStats,
    pub temporal_patch_stats: Qwen36TensorStats,
    pub position_stats: Qwen36TensorStats,
    pub intermediate_stats: Qwen36TensorStats,
    pub patch_embeddings: Vec<f32>,
}

pub fn qwen36_smart_resize(
    width: usize,
    height: usize,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(usize, usize), Qwen36VisionError> {
    smart_resize_with_limits(
        width,
        height,
        patch_size,
        spatial_merge_size,
        QWEN36_MIN_IMAGE_TOKENS,
        QWEN36_MAX_IMAGE_TOKENS,
    )
}

pub fn prepare_qwen36_vision_intermediate(
    projector: &LoadedVisionProjector,
    image: &RgbImage,
) -> Result<Qwen36VisionIntermediate, Qwen36VisionError> {
    let capability = inspect_qwen36_vision_projector(&projector.descriptor)?;
    let (target_width, target_height) = qwen36_smart_resize(
        image.width(),
        image.height(),
        capability.patch_size,
        capability.spatial_merge_size,
    )?;
    let resized = resize_bilinear_align_corners(image, target_width, target_height)?;
    let normalized = normalize_rgb(&resized, capability.image_mean, capability.image_std);
    let normalized_stats = tensor_stats(&normalized)?;

    let patch_grid_width = target_width / capability.patch_size;
    let patch_grid_height = target_height / capability.patch_size;
    let patch_rows = collect_patch_rows(
        &normalized,
        target_width,
        target_height,
        capability.patch_size,
        capability.spatial_merge_size,
    )?;
    let patch_width = capability
        .patch_size
        .checked_mul(capability.patch_size)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| error("patch input width overflows usize"))?;
    let patch_count = patch_grid_width
        .checked_mul(patch_grid_height)
        .ok_or_else(|| error("patch count overflows usize"))?;

    let kernel_0 = tensor_f32(projector, "v.patch_embd.weight")?;
    let kernel_1 = tensor_f32(projector, "v.patch_embd.weight.1")?;
    let bias = tensor_f32(projector, "v.patch_embd.bias")?;
    let mut patch_embeddings = run_temporal_patch_embedding(
        &kernel_0,
        &kernel_1,
        &bias,
        &patch_rows,
        capability.embedding_length,
        patch_width,
        patch_count,
    )?;
    let temporal_patch_stats = tensor_stats(&patch_embeddings)?;

    let source_positions = tensor_f32(projector, "v.position_embd.weight")?;
    let source_grid = capability.image_size / capability.patch_size;
    let position_stats = add_interpolated_positions(
        &mut patch_embeddings,
        &source_positions,
        source_grid,
        source_grid,
        patch_grid_width,
        patch_grid_height,
        capability.spatial_merge_size,
        capability.embedding_length,
    )?;
    if patch_embeddings.iter().any(|value| !value.is_finite()) {
        return Err(error("vision intermediate contains a non-finite value"));
    }
    let intermediate_stats = tensor_stats(&patch_embeddings)?;

    Ok(Qwen36VisionIntermediate {
        target_width,
        target_height,
        patch_grid_width,
        patch_grid_height,
        merged_grid_width: patch_grid_width / capability.spatial_merge_size,
        merged_grid_height: patch_grid_height / capability.spatial_merge_size,
        embedding_length: capability.embedding_length,
        normalized_stats,
        temporal_patch_stats,
        position_stats,
        intermediate_stats,
        patch_embeddings,
    })
}

fn smart_resize_with_limits(
    width: usize,
    height: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(usize, usize), Qwen36VisionError> {
    if width == 0 || height == 0 {
        return Err(error("image dimensions must be positive"));
    }
    if width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(error("image dimensions must fit u32"));
    }
    if patch_size == 0 || spatial_merge_size == 0 {
        return Err(error("patch and spatial merge sizes must be positive"));
    }
    if min_tokens == 0 || max_tokens < min_tokens {
        return Err(error("invalid image token limits"));
    }

    let factor = patch_size
        .checked_mul(spatial_merge_size)
        .ok_or_else(|| error("image alignment factor overflows usize"))?;
    let pixels_per_token = factor
        .checked_mul(factor)
        .ok_or_else(|| error("merged image token area overflows usize"))?;
    let min_pixels = min_tokens
        .checked_mul(pixels_per_token)
        .ok_or_else(|| error("minimum image pixel count overflows usize"))?;
    let max_pixels = max_tokens
        .checked_mul(pixels_per_token)
        .ok_or_else(|| error("maximum image pixel count overflows usize"))?;
    let source_pixels = width
        .checked_mul(height)
        .ok_or_else(|| error("source image pixel count overflows usize"))?;

    let round_by_factor = |value: f64| ((value / factor as f64).round() as usize) * factor;
    let ceil_by_factor = |value: f64| ((value / factor as f64).ceil() as usize) * factor;
    let floor_by_factor = |value: f64| ((value / factor as f64).floor() as usize) * factor;

    let mut target_height = round_by_factor(height as f64).max(factor);
    let mut target_width = round_by_factor(width as f64).max(factor);
    let aligned_pixels = target_width
        .checked_mul(target_height)
        .ok_or_else(|| error("aligned image pixel count overflows usize"))?;

    if aligned_pixels > max_pixels {
        let beta = (source_pixels as f64 / max_pixels as f64).sqrt();
        target_height = floor_by_factor(height as f64 / beta).max(factor);
        target_width = floor_by_factor(width as f64 / beta).max(factor);
    } else if aligned_pixels < min_pixels {
        let beta = (min_pixels as f64 / source_pixels as f64).sqrt();
        target_height = ceil_by_factor(height as f64 * beta).max(factor);
        target_width = ceil_by_factor(width as f64 * beta).max(factor);
    }

    let target_pixels = target_width
        .checked_mul(target_height)
        .ok_or_else(|| error("target image pixel count overflows usize"))?;
    if target_pixels < min_pixels || target_pixels > max_pixels {
        return Err(error(format!(
            "smart resize produced {target_width}x{target_height} outside pixel limits {min_pixels}..={max_pixels}"
        )));
    }
    Ok((target_width, target_height))
}

fn resize_bilinear_align_corners(
    image: &RgbImage,
    target_width: usize,
    target_height: usize,
) -> Result<RgbImage, Qwen36VisionError> {
    if image.width() == target_width && image.height() == target_height {
        return Ok(image.clone());
    }
    let byte_count = target_width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| error("resized RGB byte count overflows usize"))?;
    let mut output = vec![0u8; byte_count];
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
        let y_fraction = source_y - y0 as f32;
        for x in 0..target_width {
            let source_x = x as f32 * x_ratio;
            let x0 = (source_x as usize).min(image.width() - 1);
            let x1 = (x0 + 1).min(image.width() - 1);
            let x_fraction = source_x - x0 as f32;
            for channel in 0..3 {
                let p00 = image.pixels()[(y0 * image.width() + x0) * 3 + channel] as f32;
                let p10 = image.pixels()[(y0 * image.width() + x1) * 3 + channel] as f32;
                let p01 = image.pixels()[(y1 * image.width() + x0) * 3 + channel] as f32;
                let p11 = image.pixels()[(y1 * image.width() + x1) * 3 + channel] as f32;
                let top = p00 + (p10 - p00) * x_fraction;
                let bottom = p01 + (p11 - p01) * x_fraction;
                let value = top + (bottom - top) * y_fraction;
                output[(y * target_width + x) * 3 + channel] = value.clamp(0.0, 255.0) as u8;
            }
        }
    }

    RgbImage::new(target_width, target_height, output)
        .map_err(|image_error| error(image_error.to_string()))
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
    normalized: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<Vec<f32>, Qwen36VisionError> {
    if width % (patch_size * spatial_merge_size) != 0
        || height % (patch_size * spatial_merge_size) != 0
    {
        return Err(error("resized image is not aligned to merged patch size"));
    }
    let patch_grid_width = width / patch_size;
    let patch_grid_height = height / patch_size;
    let patch_width = patch_size * patch_size * 3;
    let patch_count = patch_grid_width * patch_grid_height;
    let mut rows = vec![0.0f32; patch_count * patch_width];
    let mut output_row = 0;

    for merged_y in (0..patch_grid_height).step_by(spatial_merge_size) {
        for merged_x in (0..patch_grid_width).step_by(spatial_merge_size) {
            for dy in 0..spatial_merge_size {
                for dx in 0..spatial_merge_size {
                    let patch_y = merged_y + dy;
                    let patch_x = merged_x + dx;
                    let row = &mut rows[output_row * patch_width..(output_row + 1) * patch_width];
                    let mut destination = 0;
                    for channel in 0..3 {
                        for y in 0..patch_size {
                            for x in 0..patch_size {
                                let image_y = patch_y * patch_size + y;
                                let image_x = patch_x * patch_size + x;
                                row[destination] =
                                    normalized[(image_y * width + image_x) * 3 + channel];
                                destination += 1;
                            }
                        }
                    }
                    output_row += 1;
                }
            }
        }
    }
    Ok(rows)
}

fn run_temporal_patch_embedding(
    kernel_0: &[f32],
    kernel_1: &[f32],
    bias: &[f32],
    patch_rows: &[f32],
    embedding_length: usize,
    patch_width: usize,
    patch_count: usize,
) -> Result<Vec<f32>, Qwen36VisionError> {
    let kernel_len = embedding_length
        .checked_mul(patch_width)
        .ok_or_else(|| error("patch kernel element count overflows usize"))?;
    let output_len = embedding_length
        .checked_mul(patch_count)
        .ok_or_else(|| error("patch output element count overflows usize"))?;
    if kernel_0.len() != kernel_len || kernel_1.len() != kernel_len {
        return Err(error("patch kernel length does not match capability"));
    }
    if bias.len() != embedding_length {
        return Err(error("patch bias length does not match embedding length"));
    }
    if patch_rows.len() != patch_count * patch_width {
        return Err(error("patch row buffer length does not match patch grid"));
    }

    let mut output = vec![0.0f32; output_len];
    let mut temporal = vec![0.0f32; output_len];
    gemv_f32(
        kernel_0,
        patch_rows,
        &mut output,
        embedding_length,
        patch_width,
        patch_count,
    );
    gemv_f32(
        kernel_1,
        patch_rows,
        &mut temporal,
        embedding_length,
        patch_width,
        patch_count,
    );
    for (row, values) in output.chunks_exact_mut(embedding_length).enumerate() {
        let temporal_row = &temporal[row * embedding_length..(row + 1) * embedding_length];
        for index in 0..embedding_length {
            values[index] += temporal_row[index] + bias[index];
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn add_interpolated_positions(
    embeddings: &mut [f32],
    source: &[f32],
    source_width: usize,
    source_height: usize,
    target_width: usize,
    target_height: usize,
    spatial_merge_size: usize,
    embedding_length: usize,
) -> Result<Qwen36TensorStats, Qwen36VisionError> {
    let source_len = source_width
        .checked_mul(source_height)
        .and_then(|positions| positions.checked_mul(embedding_length))
        .ok_or_else(|| error("source position element count overflows usize"))?;
    let target_len = target_width
        .checked_mul(target_height)
        .and_then(|positions| positions.checked_mul(embedding_length))
        .ok_or_else(|| error("target position element count overflows usize"))?;
    if source.len() != source_len || embeddings.len() != target_len {
        return Err(error("position embedding buffer length mismatch"));
    }
    if target_width % spatial_merge_size != 0 || target_height % spatial_merge_size != 0 {
        return Err(error("position grid is not aligned to spatial merge size"));
    }

    let x_ratio = if target_width > 1 {
        (source_width - 1) as f32 / (target_width - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if target_height > 1 {
        (source_height - 1) as f32 / (target_height - 1) as f32
    } else {
        0.0
    };
    let mut stats = OnlineStats::new();
    let mut output_row = 0;

    for merged_y in (0..target_height).step_by(spatial_merge_size) {
        for merged_x in (0..target_width).step_by(spatial_merge_size) {
            for dy in 0..spatial_merge_size {
                for dx in 0..spatial_merge_size {
                    let target_y = merged_y + dy;
                    let target_x = merged_x + dx;
                    let source_y = target_y as f32 * y_ratio;
                    let source_x = target_x as f32 * x_ratio;
                    let y0 = (source_y as usize).min(source_height - 1);
                    let y1 = (y0 + 1).min(source_height - 1);
                    let x0 = (source_x as usize).min(source_width - 1);
                    let x1 = (x0 + 1).min(source_width - 1);
                    let yf = source_y - y0 as f32;
                    let xf = source_x - x0 as f32;
                    let destination = &mut embeddings
                        [output_row * embedding_length..(output_row + 1) * embedding_length];
                    for index in 0..embedding_length {
                        let p00 = source[(y0 * source_width + x0) * embedding_length + index];
                        let p10 = source[(y0 * source_width + x1) * embedding_length + index];
                        let p01 = source[(y1 * source_width + x0) * embedding_length + index];
                        let p11 = source[(y1 * source_width + x1) * embedding_length + index];
                        let top = p00 + (p10 - p00) * xf;
                        let bottom = p01 + (p11 - p01) * xf;
                        let position = top + (bottom - top) * yf;
                        destination[index] += position;
                        stats.push(position);
                    }
                    output_row += 1;
                }
            }
        }
    }
    stats.finish()
}

pub(super) fn tensor_f32(
    projector: &LoadedVisionProjector,
    name: &str,
) -> Result<Vec<f32>, Qwen36VisionError> {
    let descriptor = projector
        .descriptor
        .tensors
        .get(name)
        .ok_or_else(|| error(format!("missing projector tensor '{name}'")))?;
    if descriptor.ggml_type != GGMLType::F32 {
        return Err(error(format!(
            "tensor '{name}' must be F32 for CPU vision preprocessing"
        )));
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

pub(super) fn tensor_stats(values: &[f32]) -> Result<Qwen36TensorStats, Qwen36VisionError> {
    let mut stats = OnlineStats::new();
    for value in values {
        stats.push(*value);
    }
    stats.finish()
}

struct OnlineStats {
    count: usize,
    sum: f64,
    sum_squares: f64,
    min: f32,
    max: f32,
}

impl OnlineStats {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    fn push(&mut self, value: f32) {
        self.count += 1;
        self.sum += value as f64;
        self.sum_squares += value as f64 * value as f64;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn finish(self) -> Result<Qwen36TensorStats, Qwen36VisionError> {
        if self.count == 0 {
            return Err(error("cannot compute statistics for an empty tensor"));
        }
        let mean = self.sum / self.count as f64;
        let variance = (self.sum_squares / self.count as f64 - mean * mean).max(0.0);
        Ok(Qwen36TensorStats {
            count: self.count,
            mean,
            stddev: variance.sqrt(),
            min: self.min,
            max: self.max,
        })
    }
}

fn error(message: impl Into<String>) -> Qwen36VisionError {
    Qwen36VisionError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_qwen_reference_limits_and_alignment() {
        assert_eq!(qwen36_smart_resize(768, 768, 16, 2).unwrap(), (768, 768));
        assert_eq!(qwen36_smart_resize(100, 50, 16, 2).unwrap(), (128, 64));
        assert_eq!(qwen36_smart_resize(50, 100, 16, 2).unwrap(), (64, 128));
        assert_eq!(
            qwen36_smart_resize(4000, 4000, 16, 2).unwrap(),
            (2048, 2048)
        );
    }

    #[test]
    fn bilinear_resize_uses_align_corners_then_normalizes_rgb() {
        let image =
            RgbImage::new(2, 2, vec![0, 10, 20, 100, 10, 20, 200, 10, 20, 255, 10, 20]).unwrap();
        let resized = resize_bilinear_align_corners(&image, 3, 3).unwrap();
        assert_eq!(resized.pixels()[(1 * 3 + 1) * 3], 138);

        let normalized = normalize_rgb(&resized, [0.5; 3], [0.5; 3]);
        let center = normalized[(1 * 3 + 1) * 3];
        assert!((center - (138.0 / 255.0 - 0.5) / 0.5).abs() < 1e-7);
    }

    #[test]
    fn patch_rows_follow_spatial_merge_group_order() {
        let normalized = (0..8)
            .flat_map(|value| [value as f32, 0.0, 0.0])
            .collect::<Vec<_>>();
        let rows = collect_patch_rows(&normalized, 4, 2, 1, 2).unwrap();
        let red = rows.chunks_exact(3).map(|row| row[0]).collect::<Vec<_>>();

        assert_eq!(red, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn still_image_temporal_kernels_are_summed_before_bias() {
        let output = run_temporal_patch_embedding(
            &[1.0, 1.0, 1.0],
            &[2.0, 0.0, 0.0],
            &[0.5],
            &[1.0, 2.0, 3.0],
            1,
            3,
            1,
        )
        .unwrap();
        assert_eq!(output, vec![8.5]);
    }

    #[test]
    fn learned_positions_use_bilinear_align_corners() {
        let source = vec![0.0, 10.0, 20.0, 30.0];
        let mut target = vec![0.0; 9];
        let stats = add_interpolated_positions(&mut target, &source, 2, 2, 3, 3, 1, 1).unwrap();

        assert_eq!(target[0], 0.0);
        assert_eq!(target[4], 15.0);
        assert_eq!(target[8], 30.0);
        assert_eq!(stats.count, 9);
        assert_eq!(stats.mean, 15.0);
    }

    #[test]
    fn rgb_constructor_rejects_dimension_and_byte_count_errors() {
        assert!(RgbImage::new(0, 1, Vec::new()).is_err());
        assert!(RgbImage::new(2, 2, vec![0; 11]).is_err());
    }
}
