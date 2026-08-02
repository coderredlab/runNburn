use std::collections::HashMap;
use std::path::Path;

use rnb_core::tensor::Tensor;

use crate::error::LoaderError;
use crate::gguf::metadata::{get_bool, get_string};
use crate::gguf::types::{GGMLType, GGUFValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionProjectorEnvelope {
    pub architecture: String,
    pub kind: String,
    pub projector_type: String,
    pub has_vision_encoder: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionProjectorTensor {
    pub shape: Vec<usize>,
    pub ggml_type: GGMLType,
    pub file_offset: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionProjectorDescriptor {
    pub envelope: VisionProjectorEnvelope,
    pub metadata: Vec<(String, GGUFValue)>,
    pub tensors: HashMap<String, VisionProjectorTensor>,
}

pub struct LoadedVisionProjector {
    pub descriptor: VisionProjectorDescriptor,
    pub weights: HashMap<String, Tensor>,
}

pub fn load_vision_projector(path: &Path) -> Result<LoadedVisionProjector, LoaderError> {
    let mapped = crate::gguf::sharded::load_mapped_gguf(path)?;
    let envelope = parse_envelope(&mapped.metadata)?;

    let mut tensors = HashMap::with_capacity(mapped.weights.len());
    for (name, tensor) in &mapped.weights {
        let ggml_type = mapped.tensor_ggml_types.get(name).copied().ok_or_else(|| {
            LoaderError::InvalidVisionProjector(format!("tensor '{name}' is missing its GGML type"))
        })?;
        let file_offset = mapped
            .tensor_file_offsets
            .get(name)
            .copied()
            .ok_or_else(|| {
                LoaderError::InvalidVisionProjector(format!(
                    "tensor '{name}' is missing its file offset"
                ))
            })?;
        let shape = mapped
            .float_shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| tensor.shape().to_vec());
        tensors.insert(
            name.clone(),
            VisionProjectorTensor {
                shape,
                ggml_type,
                file_offset,
            },
        );
    }

    Ok(LoadedVisionProjector {
        descriptor: VisionProjectorDescriptor {
            envelope,
            metadata: mapped.metadata,
            tensors,
        },
        weights: mapped.weights,
    })
}

fn parse_envelope(
    metadata: &[(String, GGUFValue)],
) -> Result<VisionProjectorEnvelope, LoaderError> {
    let architecture = get_string(metadata, "general.architecture")?;
    if architecture != "clip" {
        return Err(LoaderError::InvalidVisionProjector(format!(
            "general.architecture must be 'clip', got '{architecture}'"
        )));
    }

    let kind = get_string(metadata, "general.type")?;
    if kind != "mmproj" {
        return Err(LoaderError::InvalidVisionProjector(format!(
            "general.type must be 'mmproj', got '{kind}'"
        )));
    }

    let has_vision_encoder = get_bool(metadata, "clip.has_vision_encoder")?;
    if !has_vision_encoder {
        return Err(LoaderError::InvalidVisionProjector(
            "clip.has_vision_encoder must be true".to_string(),
        ));
    }

    Ok(VisionProjectorEnvelope {
        architecture: architecture.to_string(),
        kind: kind.to_string(),
        projector_type: get_string(metadata, "clip.projector_type")?.to_string(),
        has_vision_encoder,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(
        architecture: &str,
        kind: &str,
        has_vision_encoder: bool,
    ) -> Vec<(String, GGUFValue)> {
        vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String(architecture.to_string()),
            ),
            (
                "general.type".to_string(),
                GGUFValue::String(kind.to_string()),
            ),
            (
                "clip.has_vision_encoder".to_string(),
                GGUFValue::Bool(has_vision_encoder),
            ),
            (
                "clip.projector_type".to_string(),
                GGUFValue::String("qwen3vl_merger".to_string()),
            ),
        ]
    }

    #[test]
    fn accepts_mmproj_vision_envelope() {
        let envelope = parse_envelope(&metadata("clip", "mmproj", true)).unwrap();

        assert_eq!(envelope.architecture, "clip");
        assert_eq!(envelope.kind, "mmproj");
        assert_eq!(envelope.projector_type, "qwen3vl_merger");
        assert!(envelope.has_vision_encoder);
    }

    #[test]
    fn rejects_text_model_or_non_projector_role() {
        let text_model = parse_envelope(&metadata("qwen3moe", "model", true));
        assert!(matches!(
            text_model,
            Err(LoaderError::InvalidVisionProjector(message))
                if message.contains("general.architecture")
        ));

        let wrong_role = parse_envelope(&metadata("clip", "model", true));
        assert!(matches!(
            wrong_role,
            Err(LoaderError::InvalidVisionProjector(message))
                if message.contains("general.type")
        ));
    }

    #[test]
    fn rejects_projector_without_vision_encoder() {
        let result = parse_envelope(&metadata("clip", "mmproj", false));

        assert!(matches!(
            result,
            Err(LoaderError::InvalidVisionProjector(message))
                if message.contains("clip.has_vision_encoder")
        ));
    }
}
