use rnb_cpu::gemm::f32_gemv::gemv_f32;
use rnb_loader::convert::compute_tensor_size;
use rnb_loader::gguf::metadata::get_f32;
use rnb_loader::{GGMLType, LoadedVisionProjector, VisionProjectorDescriptor};

use super::vision::{
    invalid, metadata_rgb, metadata_usize, require_tensor, Gemma4VisionError,
    GEMMA4_VISION_MERGE_SIZE,
};
use super::vision_encoder::{linear_bf16, Gemma4VisionOutput};
use super::vision_math::{layer_norm_affine, rms_norm};
use super::vision_preprocess::{
    add_learned_positions, collect_patch_rows, ensure_finite, gemma4_smart_resize, normalize_rgb,
    resize_bilinear_align_corners, tensor_f32, tensor_stats,
};
use super::RgbImage;

pub const GEMMA4_UNIFIED_PROJECTOR_TYPE: &str = "gemma4uv";
const PYTORCH_LAYER_NORM_EPSILON: f32 = 1.0e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4UnifiedVisionCapability {
    pub projector_type: String,
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub effective_patch_size: usize,
    pub embedding_length: usize,
    pub layer_norm_epsilon: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub position_table_size: usize,
    pub tensor_count: usize,
    pub tensor_bytes: usize,
}

pub fn inspect_gemma4_unified_vision_projector(
    projector: &VisionProjectorDescriptor,
) -> Result<Gemma4UnifiedVisionCapability, Gemma4VisionError> {
    if projector.envelope.projector_type != GEMMA4_UNIFIED_PROJECTOR_TYPE {
        return invalid(format!(
            "clip.vision.projector_type must be '{GEMMA4_UNIFIED_PROJECTOR_TYPE}', got '{}'",
            projector.envelope.projector_type
        ));
    }

    let projection_dim = metadata_usize(projector, "clip.vision.projection_dim")?;
    let image_size = metadata_usize(projector, "clip.vision.image_size")?;
    let patch_size = metadata_usize(projector, "clip.vision.patch_size")?;
    let embedding_length = metadata_usize(projector, "clip.vision.embedding_length")?;
    let feed_forward_length = metadata_usize(projector, "clip.vision.feed_forward_length")?;
    let block_count = metadata_usize(projector, "clip.vision.block_count")?;
    let head_count = metadata_usize(projector, "clip.vision.attention.head_count")?;
    for (key, value) in [
        ("clip.vision.projection_dim", projection_dim),
        ("clip.vision.image_size", image_size),
        ("clip.vision.patch_size", patch_size),
        ("clip.vision.embedding_length", embedding_length),
        ("clip.vision.attention.head_count", head_count),
    ] {
        if value == 0 {
            return invalid(format!("{key} must be positive"));
        }
    }
    if feed_forward_length != 0 || block_count != 0 {
        return invalid(
            "Gemma 4 unified vision projector must not contain transformer blocks or feed-forward layers",
        );
    }

    let layer_norm_epsilon = get_f32(
        &projector.metadata,
        "clip.vision.attention.layer_norm_epsilon",
    )
    .map_err(|error| Gemma4VisionError::new(error.to_string()))?;
    if !layer_norm_epsilon.is_finite() || layer_norm_epsilon <= 0.0 {
        return invalid("clip.vision.attention.layer_norm_epsilon must be positive and finite");
    }
    let image_mean = metadata_rgb(projector, "clip.vision.image_mean")?;
    let image_std = metadata_rgb(projector, "clip.vision.image_std")?;
    if image_mean.iter().any(|value| !value.is_finite()) {
        return invalid("clip.vision.image_mean values must be finite");
    }
    if image_std
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return invalid("clip.vision.image_std values must be positive and finite");
    }

    let effective_patch_size = patch_size
        .checked_mul(GEMMA4_VISION_MERGE_SIZE)
        .ok_or_else(|| Gemma4VisionError::new("Gemma 4 unified patch size overflows usize"))?;
    let patch_width = effective_patch_size
        .checked_mul(effective_patch_size)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| Gemma4VisionError::new("Gemma 4 unified patch width overflows usize"))?;

    let position = projector
        .tensors
        .get("v.position_embd.weight")
        .ok_or_else(|| {
            Gemma4VisionError::new("missing projector tensor 'v.position_embd.weight'")
        })?;
    if position.shape.len() != 3 || position.shape[0] != 2 || position.shape[2] != embedding_length
    {
        return invalid(format!(
            "tensor 'v.position_embd.weight' has shape {:?}, expected [2, positions, {embedding_length}]",
            position.shape
        ));
    }
    let position_table_size = position.shape[1];
    if position_table_size == 0 {
        return invalid("Gemma 4 unified vision position table must not be empty");
    }

    for (name, shape, ggml_type) in [
        (
            "v.patch_embd.weight",
            vec![embedding_length, patch_width],
            GGMLType::F32,
        ),
        ("v.patch_embd.bias", vec![embedding_length], GGMLType::F32),
        ("v.patch_norm.1.weight", vec![patch_width], GGMLType::F32),
        ("v.patch_norm.1.bias", vec![patch_width], GGMLType::F32),
        (
            "v.patch_norm.2.weight",
            vec![embedding_length],
            GGMLType::F32,
        ),
        ("v.patch_norm.2.bias", vec![embedding_length], GGMLType::F32),
        (
            "v.patch_norm.3.weight",
            vec![embedding_length],
            GGMLType::F32,
        ),
        ("v.patch_norm.3.bias", vec![embedding_length], GGMLType::F32),
        (
            "v.position_embd.weight",
            vec![2, position_table_size, embedding_length],
            GGMLType::F32,
        ),
        (
            "mm.input_projection.weight",
            vec![projection_dim, embedding_length],
            GGMLType::BF16,
        ),
    ] {
        require_tensor(projector, name, &shape, ggml_type)?;
    }

    let tensor_bytes = projector
        .tensors
        .values()
        .try_fold(0usize, |total, tensor| {
            total.checked_add(compute_tensor_size(&tensor.shape, tensor.ggml_type))
        })
        .ok_or_else(|| Gemma4VisionError::new("projector tensor byte count overflows usize"))?;

    Ok(Gemma4UnifiedVisionCapability {
        projector_type: projector.envelope.projector_type.clone(),
        projection_dim,
        image_size,
        patch_size,
        effective_patch_size,
        embedding_length,
        layer_norm_epsilon,
        image_mean,
        image_std,
        position_table_size,
        tensor_count: projector.tensors.len(),
        tensor_bytes,
    })
}

pub fn encode_gemma4_unified_vision(
    projector: &LoadedVisionProjector,
    image: &RgbImage,
) -> Result<Gemma4VisionOutput, Gemma4VisionError> {
    let capability = inspect_gemma4_unified_vision_projector(&projector.descriptor)?;
    let (target_width, target_height) =
        gemma4_smart_resize(image.width(), image.height(), capability.patch_size)?;
    let resized = resize_bilinear_align_corners(image, target_width, target_height)?;
    let normalized = normalize_rgb(&resized, capability.image_mean, capability.image_std);
    let patch_rows = collect_patch_rows(
        &normalized,
        target_width,
        target_height,
        capability.effective_patch_size,
    )?;
    let patch_grid_width = target_width / capability.effective_patch_size;
    let patch_grid_height = target_height / capability.effective_patch_size;
    if patch_grid_width > capability.position_table_size
        || patch_grid_height > capability.position_table_size
    {
        return Err(Gemma4VisionError::new(format!(
            "Gemma 4 unified vision grid {patch_grid_width}x{patch_grid_height} exceeds position table {}",
            capability.position_table_size
        )));
    }
    let patch_count = patch_grid_width
        .checked_mul(patch_grid_height)
        .ok_or_else(|| Gemma4VisionError::new("Gemma 4 unified patch count overflows usize"))?;
    let patch_width = capability
        .effective_patch_size
        .checked_mul(capability.effective_patch_size)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| Gemma4VisionError::new("Gemma 4 unified patch width overflows usize"))?;

    let patch_norm_1 = layer_norm_affine(
        &patch_rows,
        patch_width,
        PYTORCH_LAYER_NORM_EPSILON,
        &tensor_f32(projector, "v.patch_norm.1.weight")?,
        &tensor_f32(projector, "v.patch_norm.1.bias")?,
    )?;
    let mut hidden = linear_f32_with_bias(
        projector,
        "v.patch_embd.weight",
        "v.patch_embd.bias",
        &patch_norm_1,
        capability.embedding_length,
        patch_width,
        patch_count,
    )?;
    hidden = layer_norm_affine(
        &hidden,
        capability.embedding_length,
        PYTORCH_LAYER_NORM_EPSILON,
        &tensor_f32(projector, "v.patch_norm.2.weight")?,
        &tensor_f32(projector, "v.patch_norm.2.bias")?,
    )?;
    let positions = tensor_f32(projector, "v.position_embd.weight")?;
    add_learned_positions(
        &mut hidden,
        &positions,
        patch_grid_width,
        patch_grid_height,
        capability.position_table_size,
        capability.embedding_length,
    )?;
    hidden = layer_norm_affine(
        &hidden,
        capability.embedding_length,
        PYTORCH_LAYER_NORM_EPSILON,
        &tensor_f32(projector, "v.patch_norm.3.weight")?,
        &tensor_f32(projector, "v.patch_norm.3.bias")?,
    )?;
    let projected_input = rms_norm(
        &hidden,
        capability.embedding_length,
        capability.layer_norm_epsilon,
    )?;
    let pooled_stats = tensor_stats(&projected_input)?;
    let embeddings = linear_bf16(
        projector,
        "mm.input_projection.weight",
        &projected_input,
        capability.projection_dim,
        capability.embedding_length,
        patch_count,
    )?;
    ensure_finite(&embeddings, "Gemma 4 unified vision projection")?;
    let embedding_stats = tensor_stats(&embeddings)?;

    Ok(Gemma4VisionOutput {
        target_width,
        target_height,
        patch_grid_width,
        patch_grid_height,
        pooled_grid_width: patch_grid_width,
        pooled_grid_height: patch_grid_height,
        projection_dim: capability.projection_dim,
        layer_summaries: Vec::new(),
        pooled_stats,
        embedding_stats,
        embeddings,
    })
}

#[allow(clippy::too_many_arguments)]
fn linear_f32_with_bias(
    projector: &LoadedVisionProjector,
    weight_name: &str,
    bias_name: &str,
    input: &[f32],
    rows: usize,
    cols: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Gemma4VisionError> {
    if input.len() != cols * sequence_length {
        return Err(Gemma4VisionError::new(format!(
            "tensor '{weight_name}' input shape mismatch"
        )));
    }
    let weight = tensor_f32(projector, weight_name)?;
    let bias = tensor_f32(projector, bias_name)?;
    let mut output = vec![0.0f32; rows * sequence_length];
    gemv_f32(&weight, input, &mut output, rows, cols, sequence_length);
    for row in output.chunks_mut(rows) {
        row.iter_mut()
            .zip(&bias)
            .for_each(|(value, bias)| *value += *bias);
    }
    Ok(output)
}
