use super::super::dequant::dequantize_bytes_to_f32;
use rnb_core::tensor::{DType, Tensor};
use rnb_loader::{GGMLType, LoadedModel};

fn register_model_host_storage(tensor: Tensor) -> Tensor {
    #[cfg(feature = "metal")]
    tensor.register_host_storage();
    tensor
}

/// F32로 dequant해야 하는 small weight (norm 등)
pub(super) fn load_f32_weight(model: &LoadedModel, name: &str) -> Tensor {
    let tensor = match model.weights.get(name) {
        Some(t) => t,
        None => {
            eprintln!("[WARN] weight '{}' not found, using zeros", name);
            return register_model_host_storage(Tensor::from_slice(&[0.0f32], &[1]));
        }
    };

    if matches!(tensor.dtype(), DType::U8 | DType::I8) {
        if let (Some(bytes), Some(float_shape)) = (tensor.as_bytes(), model.float_shapes.get(name))
        {
            let ggml_type = model
                .tensor_ggml_types
                .get(name)
                .copied()
                .unwrap_or(GGMLType::Q4_0);
            let f32_data = dequantize_bytes_to_f32(bytes, ggml_type);
            let target_numel: usize = float_shape.iter().product();
            if f32_data.len() == target_numel {
                return register_model_host_storage(Tensor::from_slice(&f32_data, float_shape));
            }
            return register_model_host_storage(Tensor::from_slice(&f32_data, &[f32_data.len()]));
        }
    }

    if tensor.dtype() == DType::F16 {
        if let Some(bytes) = tensor.as_bytes() {
            let ggml_type = model
                .tensor_ggml_types
                .get(name)
                .copied()
                .unwrap_or(GGMLType::F16);
            let f32_data: Vec<f32> = match ggml_type {
                GGMLType::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let bf16 = u16::from_le_bytes([c[0], c[1]]) as u32;
                        f32::from_bits(bf16 << 16)
                    })
                    .collect(),
                _ => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
            };
            if let Some(float_shape) = model.float_shapes.get(name) {
                return register_model_host_storage(Tensor::from_slice(&f32_data, float_shape));
            }
            return register_model_host_storage(Tensor::from_slice(&f32_data, &[f32_data.len()]));
        }
    }

    register_model_host_storage(tensor.clone())
}

pub(super) fn load_optional_f32_weight(model: &LoadedModel, name: &str) -> Option<Tensor> {
    model
        .weights
        .contains_key(name)
        .then(|| load_f32_weight(model, name))
}
