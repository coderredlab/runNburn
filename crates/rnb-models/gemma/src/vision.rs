use std::fmt;

use rnb_loader::convert::compute_tensor_size;
use rnb_loader::gguf::metadata::{get_f32, get_f32_array, get_u32};
use rnb_loader::{GGMLType, VisionProjectorDescriptor};

const PROJECTOR_TYPE: &str = "gemma4v";
pub const GEMMA4_VISION_MERGE_SIZE: usize = 3;
pub const GEMMA4_VISION_ROPE_THETA: f32 = 100.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionCapability {
    pub projector_type: String,
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub layer_norm_epsilon: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub position_table_size: usize,
    pub tensor_count: usize,
    pub tensor_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4VisionError(String);

impl Gemma4VisionError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Gemma4VisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Gemma4VisionError {}

pub fn inspect_gemma4_vision_projector(
    projector: &VisionProjectorDescriptor,
) -> Result<Gemma4VisionCapability, Gemma4VisionError> {
    if projector.envelope.projector_type != PROJECTOR_TYPE {
        return invalid(format!(
            "clip.vision.projector_type must be '{PROJECTOR_TYPE}', got '{}'",
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
    let layer_norm_epsilon = get_f32(
        &projector.metadata,
        "clip.vision.attention.layer_norm_epsilon",
    )
    .map_err(metadata_error)?;

    for (key, value) in [
        ("clip.vision.projection_dim", projection_dim),
        ("clip.vision.image_size", image_size),
        ("clip.vision.patch_size", patch_size),
        ("clip.vision.embedding_length", embedding_length),
        ("clip.vision.feed_forward_length", feed_forward_length),
        ("clip.vision.block_count", block_count),
        ("clip.vision.attention.head_count", head_count),
    ] {
        if value == 0 {
            return invalid(format!("{key} must be positive"));
        }
    }
    if !layer_norm_epsilon.is_finite() || layer_norm_epsilon <= 0.0 {
        return invalid("clip.vision.attention.layer_norm_epsilon must be positive and finite");
    }
    if embedding_length % head_count != 0 {
        return invalid(
            "clip.vision.embedding_length must be divisible by clip.vision.attention.head_count",
        );
    }
    let head_dim = embedding_length / head_count;
    if head_dim % 4 != 0 {
        return invalid("Gemma 4 vision attention head width must be divisible by four");
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

    let position = projector
        .tensors
        .get("v.position_embd.weight")
        .ok_or_else(|| {
            Gemma4VisionError("missing projector tensor 'v.position_embd.weight'".into())
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
        return invalid("Gemma 4 vision position table must not be empty");
    }

    validate_tensors(
        projector,
        projection_dim,
        patch_size,
        embedding_length,
        feed_forward_length,
        block_count,
        head_dim,
        position_table_size,
    )?;

    let tensor_bytes = projector
        .tensors
        .values()
        .try_fold(0usize, |total, tensor| {
            total.checked_add(compute_tensor_size(&tensor.shape, tensor.ggml_type))
        })
        .ok_or_else(|| Gemma4VisionError("projector tensor byte count overflows usize".into()))?;

    Ok(Gemma4VisionCapability {
        projector_type: projector.envelope.projector_type.clone(),
        projection_dim,
        image_size,
        patch_size,
        embedding_length,
        feed_forward_length,
        block_count,
        head_count,
        layer_norm_epsilon,
        image_mean,
        image_std,
        position_table_size,
        tensor_count: projector.tensors.len(),
        tensor_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_tensors(
    projector: &VisionProjectorDescriptor,
    projection_dim: usize,
    patch_size: usize,
    embedding_length: usize,
    feed_forward_length: usize,
    block_count: usize,
    head_dim: usize,
    position_table_size: usize,
) -> Result<(), Gemma4VisionError> {
    require_tensor(
        projector,
        "v.patch_embd.weight",
        &[embedding_length, 3, patch_size, patch_size],
        GGMLType::F32,
    )?;
    require_tensor(
        projector,
        "v.position_embd.weight",
        &[2, position_table_size, embedding_length],
        GGMLType::F32,
    )?;
    for name in ["v.std_bias", "v.std_scale"] {
        require_tensor(projector, name, &[embedding_length], GGMLType::F32)?;
    }
    require_tensor(
        projector,
        "mm.input_projection.weight",
        &[projection_dim, embedding_length],
        GGMLType::BF16,
    )?;

    for layer in 0..block_count {
        let prefix = format!("v.blk.{layer}");
        for suffix in [
            "ln1.weight",
            "ln2.weight",
            "attn_post_norm.weight",
            "ffn_post_norm.weight",
        ] {
            require_tensor(
                projector,
                &format!("{prefix}.{suffix}"),
                &[embedding_length],
                GGMLType::F32,
            )?;
        }
        for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            require_tensor(
                projector,
                &format!("{prefix}.{suffix}"),
                &[head_dim],
                GGMLType::F32,
            )?;
        }
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_out.weight",
        ] {
            require_tensor(
                projector,
                &format!("{prefix}.{suffix}"),
                &[embedding_length, embedding_length],
                GGMLType::BF16,
            )?;
        }
        for suffix in ["ffn_gate.weight", "ffn_up.weight"] {
            require_tensor(
                projector,
                &format!("{prefix}.{suffix}"),
                &[feed_forward_length, embedding_length],
                GGMLType::BF16,
            )?;
        }
        require_tensor(
            projector,
            &format!("{prefix}.ffn_down.weight"),
            &[embedding_length, feed_forward_length],
            GGMLType::BF16,
        )?;
    }
    Ok(())
}

fn require_tensor(
    projector: &VisionProjectorDescriptor,
    name: &str,
    shape: &[usize],
    ggml_type: GGMLType,
) -> Result<(), Gemma4VisionError> {
    let tensor = projector
        .tensors
        .get(name)
        .ok_or_else(|| Gemma4VisionError(format!("missing projector tensor '{name}'")))?;
    if tensor.shape != shape {
        return invalid(format!(
            "tensor '{name}' has shape {:?}, expected {shape:?}",
            tensor.shape
        ));
    }
    if tensor.ggml_type != ggml_type {
        return invalid(format!(
            "tensor '{name}' has type {:?}, expected {ggml_type:?}",
            tensor.ggml_type
        ));
    }
    Ok(())
}

fn metadata_usize(
    projector: &VisionProjectorDescriptor,
    key: &str,
) -> Result<usize, Gemma4VisionError> {
    get_u32(&projector.metadata, key)
        .map(|value| value as usize)
        .map_err(metadata_error)
}

fn metadata_rgb(
    projector: &VisionProjectorDescriptor,
    key: &str,
) -> Result<[f32; 3], Gemma4VisionError> {
    let values = get_f32_array(&projector.metadata, key).map_err(metadata_error)?;
    values.try_into().map_err(|values: Vec<f32>| {
        Gemma4VisionError(format!(
            "{key} must contain three values, got {}",
            values.len()
        ))
    })
}

fn metadata_error(error: rnb_loader::LoaderError) -> Gemma4VisionError {
    Gemma4VisionError(error.to_string())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, Gemma4VisionError> {
    Err(Gemma4VisionError(message.into()))
}
