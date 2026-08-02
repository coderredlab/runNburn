use std::fmt;

use rnb_loader::convert::compute_tensor_size;
use rnb_loader::gguf::metadata::{get_bool, get_bool_array, get_f32, get_f32_array, get_u32};
use rnb_loader::{GGMLType, VisionProjectorDescriptor};

const PROJECTOR_TYPE: &str = "qwen3vl_merger";

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36VisionCapability {
    pub projector_type: String,
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub layer_norm_epsilon: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub tensor_count: usize,
    pub tensor_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen36VisionError(String);

impl Qwen36VisionError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Qwen36VisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Qwen36VisionError {}

pub fn inspect_qwen36_vision_projector(
    projector: &VisionProjectorDescriptor,
) -> Result<Qwen36VisionCapability, Qwen36VisionError> {
    if projector.envelope.projector_type != PROJECTOR_TYPE {
        return invalid(format!(
            "clip.projector_type must be '{PROJECTOR_TYPE}', got '{}'",
            projector.envelope.projector_type
        ));
    }

    let projection_dim = metadata_usize(projector, "clip.vision.projection_dim")?;
    let image_size = metadata_usize(projector, "clip.vision.image_size")?;
    let patch_size = metadata_usize(projector, "clip.vision.patch_size")?;
    let spatial_merge_size = metadata_usize(projector, "clip.vision.spatial_merge_size")?;
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
        ("clip.vision.spatial_merge_size", spatial_merge_size),
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
    if image_size % patch_size != 0 {
        return invalid("clip.vision.image_size must be divisible by clip.vision.patch_size");
    }
    if embedding_length % head_count != 0 {
        return invalid(
            "clip.vision.embedding_length must be divisible by clip.vision.attention.head_count",
        );
    }

    let use_gelu = get_bool(&projector.metadata, "clip.use_gelu").map_err(metadata_error)?;
    if !use_gelu {
        return invalid("clip.use_gelu must be true");
    }

    let deepstack = get_bool_array(&projector.metadata, "clip.vision.is_deepstack_layers")
        .map_err(metadata_error)?;
    if deepstack.len() != block_count {
        return invalid(format!(
            "clip.vision.is_deepstack_layers has {} entries, expected {block_count}",
            deepstack.len()
        ));
    }
    if let Some(index) = deepstack.iter().position(|enabled| *enabled) {
        return invalid(format!(
            "deepstack vision layer {index} is not supported by the first Qwen3.6 projector contract"
        ));
    }

    let image_mean = metadata_rgb(projector, "clip.vision.image_mean")?;
    let image_std = metadata_rgb(projector, "clip.vision.image_std")?;
    if image_std
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return invalid("clip.vision.image_std values must be positive and finite");
    }
    if image_mean.iter().any(|value| !value.is_finite()) {
        return invalid("clip.vision.image_mean values must be finite");
    }

    validate_tensors(
        projector,
        projection_dim,
        image_size,
        patch_size,
        spatial_merge_size,
        embedding_length,
        feed_forward_length,
        block_count,
    )?;

    let tensor_bytes = projector
        .tensors
        .values()
        .try_fold(0usize, |total, tensor| {
            total.checked_add(compute_tensor_size(&tensor.shape, tensor.ggml_type))
        })
        .ok_or_else(|| Qwen36VisionError("projector tensor byte count overflows usize".into()))?;

    Ok(Qwen36VisionCapability {
        projector_type: projector.envelope.projector_type.clone(),
        projection_dim,
        image_size,
        patch_size,
        spatial_merge_size,
        embedding_length,
        feed_forward_length,
        block_count,
        head_count,
        layer_norm_epsilon,
        image_mean,
        image_std,
        tensor_count: projector.tensors.len(),
        tensor_bytes,
    })
}

fn validate_tensors(
    projector: &VisionProjectorDescriptor,
    projection_dim: usize,
    image_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    embedding_length: usize,
    feed_forward_length: usize,
    block_count: usize,
) -> Result<(), Qwen36VisionError> {
    let patch_grid = image_size / patch_size;
    let position_count = patch_grid
        .checked_mul(patch_grid)
        .ok_or_else(|| Qwen36VisionError("vision position count overflows usize".into()))?;
    let merge_area = spatial_merge_size
        .checked_mul(spatial_merge_size)
        .ok_or_else(|| Qwen36VisionError("vision merge area overflows usize".into()))?;
    let merger_input = embedding_length
        .checked_mul(merge_area)
        .ok_or_else(|| Qwen36VisionError("vision merger input width overflows usize".into()))?;
    let qkv_width = embedding_length
        .checked_mul(3)
        .ok_or_else(|| Qwen36VisionError("vision QKV width overflows usize".into()))?;

    require_tensor(
        projector,
        "v.patch_embd.weight",
        &[embedding_length, 3, patch_size, patch_size],
        GGMLType::F32,
    )?;
    require_tensor(
        projector,
        "v.patch_embd.weight.1",
        &[embedding_length, 3, patch_size, patch_size],
        GGMLType::F32,
    )?;
    require_tensor(
        projector,
        "v.patch_embd.bias",
        &[embedding_length],
        GGMLType::F32,
    )?;
    require_tensor(
        projector,
        "v.position_embd.weight",
        &[position_count, embedding_length],
        GGMLType::F32,
    )?;
    for name in ["v.post_ln.weight", "v.post_ln.bias"] {
        require_tensor(projector, name, &[embedding_length], GGMLType::F32)?;
    }
    require_tensor(
        projector,
        "mm.0.weight",
        &[merger_input, merger_input],
        GGMLType::BF16,
    )?;
    require_tensor(projector, "mm.0.bias", &[merger_input], GGMLType::F32)?;
    require_tensor(
        projector,
        "mm.2.weight",
        &[projection_dim, merger_input],
        GGMLType::BF16,
    )?;
    require_tensor(projector, "mm.2.bias", &[projection_dim], GGMLType::F32)?;

    for layer in 0..block_count {
        let prefix = format!("v.blk.{layer}");
        require_tensor(
            projector,
            &format!("{prefix}.attn_qkv.weight"),
            &[qkv_width, embedding_length],
            GGMLType::BF16,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.attn_qkv.bias"),
            &[qkv_width],
            GGMLType::F32,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.attn_out.weight"),
            &[embedding_length, embedding_length],
            GGMLType::BF16,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.attn_out.bias"),
            &[embedding_length],
            GGMLType::F32,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.ffn_up.weight"),
            &[feed_forward_length, embedding_length],
            GGMLType::BF16,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.ffn_up.bias"),
            &[feed_forward_length],
            GGMLType::F32,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.ffn_down.weight"),
            &[embedding_length, feed_forward_length],
            GGMLType::BF16,
        )?;
        require_tensor(
            projector,
            &format!("{prefix}.ffn_down.bias"),
            &[embedding_length],
            GGMLType::F32,
        )?;
        for suffix in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
            require_tensor(
                projector,
                &format!("{prefix}.{suffix}"),
                &[embedding_length],
                GGMLType::F32,
            )?;
        }
    }

    let expected_count = block_count
        .checked_mul(12)
        .and_then(|count| count.checked_add(10))
        .ok_or_else(|| Qwen36VisionError("vision tensor count overflows usize".into()))?;
    if projector.tensors.len() != expected_count {
        return invalid(format!(
            "projector has {} tensors, expected exactly {expected_count}",
            projector.tensors.len()
        ));
    }

    Ok(())
}

fn require_tensor(
    projector: &VisionProjectorDescriptor,
    name: &str,
    expected_shape: &[usize],
    expected_type: GGMLType,
) -> Result<(), Qwen36VisionError> {
    let tensor = projector
        .tensors
        .get(name)
        .ok_or_else(|| Qwen36VisionError(format!("missing projector tensor '{name}'")))?;
    if tensor.shape != expected_shape {
        return invalid(format!(
            "tensor '{name}' has shape {:?}, expected {expected_shape:?}",
            tensor.shape
        ));
    }
    if tensor.ggml_type != expected_type {
        return invalid(format!(
            "tensor '{name}' has type {:?}, expected {expected_type:?}",
            tensor.ggml_type
        ));
    }
    Ok(())
}

fn metadata_usize(
    projector: &VisionProjectorDescriptor,
    key: &str,
) -> Result<usize, Qwen36VisionError> {
    usize::try_from(get_u32(&projector.metadata, key).map_err(metadata_error)?)
        .map_err(|_| Qwen36VisionError(format!("metadata key '{key}' does not fit usize")))
}

fn metadata_rgb(
    projector: &VisionProjectorDescriptor,
    key: &str,
) -> Result<[f32; 3], Qwen36VisionError> {
    let values = get_f32_array(&projector.metadata, key).map_err(metadata_error)?;
    values.try_into().map_err(|values: Vec<f32>| {
        Qwen36VisionError(format!(
            "metadata key '{key}' has {} entries, expected 3",
            values.len()
        ))
    })
}

fn metadata_error(error: rnb_loader::LoaderError) -> Qwen36VisionError {
    Qwen36VisionError(error.to_string())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, Qwen36VisionError> {
    Err(Qwen36VisionError(message.into()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rnb_loader::{VisionProjectorEnvelope, VisionProjectorTensor};

    use super::*;

    const EMBEDDING: usize = 1152;
    const FEED_FORWARD: usize = 4304;
    const BLOCKS: usize = 27;
    const PROJECTION: usize = 2048;
    const PATCH: usize = 16;
    const IMAGE: usize = 768;
    const MERGE: usize = 2;

    fn kv(key: &str, value: rnb_loader::gguf::GGUFValue) -> (String, rnb_loader::gguf::GGUFValue) {
        (key.to_string(), value)
    }

    fn tensor(shape: &[usize], ggml_type: GGMLType) -> VisionProjectorTensor {
        VisionProjectorTensor {
            shape: shape.to_vec(),
            ggml_type,
            file_offset: 0,
        }
    }

    fn descriptor() -> VisionProjectorDescriptor {
        use rnb_loader::gguf::GGUFValue::{Array, Bool, F32, U32};

        let metadata = vec![
            kv("clip.vision.projection_dim", U32(PROJECTION as u32)),
            kv("clip.vision.image_size", U32(IMAGE as u32)),
            kv("clip.vision.patch_size", U32(PATCH as u32)),
            kv("clip.vision.embedding_length", U32(EMBEDDING as u32)),
            kv("clip.vision.feed_forward_length", U32(FEED_FORWARD as u32)),
            kv("clip.vision.block_count", U32(BLOCKS as u32)),
            kv("clip.vision.attention.head_count", U32(16)),
            kv("clip.vision.spatial_merge_size", U32(MERGE as u32)),
            kv("clip.vision.attention.layer_norm_epsilon", F32(1e-6)),
            kv("clip.use_gelu", Bool(true)),
            kv(
                "clip.vision.is_deepstack_layers",
                Array((0..BLOCKS).map(|_| Bool(false)).collect()),
            ),
            kv(
                "clip.vision.image_mean",
                Array([0.48, 0.46, 0.41].into_iter().map(F32).collect()),
            ),
            kv(
                "clip.vision.image_std",
                Array([0.27, 0.26, 0.28].into_iter().map(F32).collect()),
            ),
        ];

        let merger_input = EMBEDDING * MERGE * MERGE;
        let mut tensors = HashMap::new();
        tensors.insert(
            "v.patch_embd.weight".into(),
            tensor(&[EMBEDDING, 3, PATCH, PATCH], GGMLType::F32),
        );
        tensors.insert(
            "v.patch_embd.weight.1".into(),
            tensor(&[EMBEDDING, 3, PATCH, PATCH], GGMLType::F32),
        );
        tensors.insert(
            "v.patch_embd.bias".into(),
            tensor(&[EMBEDDING], GGMLType::F32),
        );
        tensors.insert(
            "v.position_embd.weight".into(),
            tensor(&[(IMAGE / PATCH).pow(2), EMBEDDING], GGMLType::F32),
        );
        for name in ["v.post_ln.weight", "v.post_ln.bias"] {
            tensors.insert(name.into(), tensor(&[EMBEDDING], GGMLType::F32));
        }
        tensors.insert(
            "mm.0.weight".into(),
            tensor(&[merger_input, merger_input], GGMLType::BF16),
        );
        tensors.insert("mm.0.bias".into(), tensor(&[merger_input], GGMLType::F32));
        tensors.insert(
            "mm.2.weight".into(),
            tensor(&[PROJECTION, merger_input], GGMLType::BF16),
        );
        tensors.insert("mm.2.bias".into(), tensor(&[PROJECTION], GGMLType::F32));

        for layer in 0..BLOCKS {
            let prefix = format!("v.blk.{layer}");
            tensors.insert(
                format!("{prefix}.attn_qkv.weight"),
                tensor(&[EMBEDDING * 3, EMBEDDING], GGMLType::BF16),
            );
            tensors.insert(
                format!("{prefix}.attn_qkv.bias"),
                tensor(&[EMBEDDING * 3], GGMLType::F32),
            );
            tensors.insert(
                format!("{prefix}.attn_out.weight"),
                tensor(&[EMBEDDING, EMBEDDING], GGMLType::BF16),
            );
            tensors.insert(
                format!("{prefix}.attn_out.bias"),
                tensor(&[EMBEDDING], GGMLType::F32),
            );
            tensors.insert(
                format!("{prefix}.ffn_up.weight"),
                tensor(&[FEED_FORWARD, EMBEDDING], GGMLType::BF16),
            );
            tensors.insert(
                format!("{prefix}.ffn_up.bias"),
                tensor(&[FEED_FORWARD], GGMLType::F32),
            );
            tensors.insert(
                format!("{prefix}.ffn_down.weight"),
                tensor(&[EMBEDDING, FEED_FORWARD], GGMLType::BF16),
            );
            tensors.insert(
                format!("{prefix}.ffn_down.bias"),
                tensor(&[EMBEDDING], GGMLType::F32),
            );
            for suffix in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
                tensors.insert(
                    format!("{prefix}.{suffix}"),
                    tensor(&[EMBEDDING], GGMLType::F32),
                );
            }
        }

        VisionProjectorDescriptor {
            envelope: VisionProjectorEnvelope {
                architecture: "clip".into(),
                kind: "mmproj".into(),
                projector_type: PROJECTOR_TYPE.into(),
                has_vision_encoder: true,
            },
            metadata,
            tensors,
        }
    }

    #[test]
    fn accepts_observed_qwen36_projector_contract() {
        let capability = inspect_qwen36_vision_projector(&descriptor()).unwrap();

        assert_eq!(capability.tensor_count, 334);
        assert_eq!(capability.tensor_bytes, 902_802_368);
        assert_eq!(capability.projection_dim, PROJECTION);
        assert_eq!(capability.block_count, BLOCKS);
    }

    #[test]
    fn rejects_wrong_merger_projection_shape() {
        let mut projector = descriptor();
        projector.tensors.get_mut("mm.2.weight").unwrap().shape[0] += 1;

        let error = inspect_qwen36_vision_projector(&projector).unwrap_err();
        assert!(error.to_string().contains("mm.2.weight"));
        assert!(error.to_string().contains("expected"));
    }

    #[test]
    fn rejects_deepstack_before_execution_support_exists() {
        let mut projector = descriptor();
        let (_, value) = projector
            .metadata
            .iter_mut()
            .find(|(key, _)| key == "clip.vision.is_deepstack_layers")
            .unwrap();
        let rnb_loader::gguf::GGUFValue::Array(values) = value else {
            unreachable!();
        };
        values[3] = rnb_loader::gguf::GGUFValue::Bool(true);

        let error = inspect_qwen36_vision_projector(&projector).unwrap_err();
        assert!(error.to_string().contains("deepstack vision layer 3"));
    }
}
