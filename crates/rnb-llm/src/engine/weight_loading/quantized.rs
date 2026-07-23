use super::super::quantized_dispatch::expected_quantized_byte_len;
use super::super::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::{GGMLType, LoadedModel};

fn is_gather_only_embedding_weight(name: &str) -> bool {
    matches!(name, "token_embd.weight" | "per_layer_token_embd.weight")
}

fn q4k_load_repack_enabled() -> bool {
    let env = crate::engine::policy::env_string("RNB_Q4K_LOAD_REPACK");
    q4k_load_repack_enabled_with_feature(env.as_deref(), cfg!(feature = "metal"))
}

fn q4k_load_repack_enabled_with_feature(raw: Option<&str>, metal_feature: bool) -> bool {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("0" | "false" | "off" | "no") => false,
        Some("1" | "true" | "on" | "yes") => true,
        Some(_) => true,
        None => !metal_feature,
    }
}

/// 양자화 상태 유지 weight (matmul용 2D weight)
pub(super) fn load_quantized_weight(model: &LoadedModel, name: &str) -> QuantizedWeight {
    let tensor = match model.weights.get(name) {
        Some(t) => t.clone(),
        None => {
            eprintln!("[WARN] weight '{}' not found, using zeros", name);
            return dummy_quantized_weight();
        }
    };

    let ggml_type = model
        .tensor_ggml_types
        .get(name)
        .copied()
        .unwrap_or(GGMLType::F32);
    let (rows, cols) = if let Some(float_shape) = model.float_shapes.get(name) {
        (
            float_shape[0],
            if float_shape.len() > 1 {
                float_shape[1]
            } else {
                1
            },
        )
    } else {
        let shape = tensor.shape();
        (shape[0], if shape.len() > 1 { shape[1] } else { 1 })
    };

    let has_full_raw_bytes = tensor.as_bytes().is_some_and(|bytes| {
        expected_quantized_byte_len(ggml_type, rows, cols) == Some(bytes.len())
    });
    let should_repack_q4k = ggml_type == GGMLType::Q4_K
        && has_full_raw_bytes
        && !is_gather_only_embedding_weight(name)
        && q4k_load_repack_enabled();

    QuantizedWeight::new_with_q4k_repack(
        tensor.clone(),
        ggml_type,
        rows,
        cols,
        tensor.as_bytes(),
        should_repack_q4k,
    )
}

pub(super) fn dummy_quantized_weight() -> QuantizedWeight {
    QuantizedWeight::new(Tensor::from_slice(&[0.0f32], &[1]), GGMLType::F32, 1, 1)
}

#[cfg(test)]
fn q4k_load_repack_enabled_for_test() -> bool {
    q4k_load_repack_enabled()
}

#[cfg(test)]
fn q4k_load_repack_enabled_with_feature_for_test(raw: Option<&str>, metal_feature: bool) -> bool {
    q4k_load_repack_enabled_with_feature(raw, metal_feature)
}

#[cfg(test)]
mod tests {
    #[test]
    fn q4k_load_repack_policy_keeps_cpu_default_and_skips_metal_default() {
        assert!(super::q4k_load_repack_enabled_with_feature_for_test(
            None, false
        ));
        assert!(!super::q4k_load_repack_enabled_with_feature_for_test(
            None, true
        ));
    }

    #[test]
    fn q4k_load_repack_can_be_disabled_for_metal_load_probe() {
        let _guard = crate::engine::moe::tests::env_lock()
            .lock()
            .expect("env lock poisoned");
        let prev = crate::engine::policy::env_string("RNB_Q4K_LOAD_REPACK");
        unsafe {
            std::env::set_var("RNB_Q4K_LOAD_REPACK", "0");
        }

        assert!(!super::q4k_load_repack_enabled_for_test());

        unsafe {
            if let Some(value) = prev {
                std::env::set_var("RNB_Q4K_LOAD_REPACK", value);
            } else {
                std::env::remove_var("RNB_Q4K_LOAD_REPACK");
            }
        }
    }

    #[test]
    fn q4k_load_repack_explicit_enable_overrides_metal_default() {
        assert!(super::q4k_load_repack_enabled_with_feature_for_test(
            Some("1"),
            true
        ));
        assert!(super::q4k_load_repack_enabled_with_feature_for_test(
            Some("true"),
            true
        ));
    }
}

/// Fuse gate + up weights into a single QuantizedWeight for combined GEMV.
pub(super) fn fuse_gate_up(
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<QuantizedWeight> {
    if gate.cols != up.cols {
        return None;
    }
    let (gate_q4, up_q4) = match (&gate.q4_0_data, &up.q4_0_data) {
        (Some(g), Some(u)) => (g, u),
        _ => return None,
    };
    let mut combined = Vec::with_capacity(gate_q4.len() + up_q4.len());
    combined.extend_from_slice(gate_q4);
    combined.extend_from_slice(up_q4);
    let mut weight = QuantizedWeight::new(
        gate.data.clone(),
        gate.ggml_type,
        gate.rows + up.rows,
        gate.cols,
    );
    weight.q4_0_data = Some(combined);
    Some(weight)
}
