use super::types::TensorView;
use rnb_loader::gguf::types::GGMLType;

#[cfg(feature = "metal")]
fn q8_0_multi_gemv(
    weights: &[&TensorView],
    inputs: &[&[f32]],
) -> Result<Option<Vec<Vec<f32>>>, String> {
    if weights.len() != inputs.len() || weights.is_empty() {
        return Ok(None);
    }
    let mut raw = Vec::with_capacity(weights.len());
    let mut layout = Vec::with_capacity(weights.len());
    for (weight, input) in weights.iter().zip(inputs) {
        if weight.ggml_type != GGMLType::Q8_0 || weight.shape.len() != 2 {
            return Ok(None);
        }
        let rows = weight.shape[0];
        let cols = weight.shape[1];
        if input.len() != cols {
            return Ok(None);
        }
        raw.push(weight.as_bytes());
        layout.push((rows, cols));
    }
    rnb_runtime::metal_inference::metal_drafter_q8_0_multi_gemv_if_supported(&raw, inputs, &layout)
}

#[cfg(not(feature = "metal"))]
fn q8_0_multi_gemv(
    _weights: &[&TensorView],
    _inputs: &[&[f32]],
) -> Result<Option<Vec<Vec<f32>>>, String> {
    Ok(None)
}

pub(crate) fn drafter_q8_0_gemv(
    weight: &TensorView,
    input: &[f32],
    output: &mut [f32],
) -> Result<bool, String> {
    let Some(mut outputs) = q8_0_multi_gemv(&[weight], &[input])? else {
        return Ok(false);
    };
    let values = outputs.pop().expect("single Metal drafter GEMV output");
    if values.len() != output.len() {
        return Err(format!(
            "Metal drafter GEMV output length {} != {}",
            values.len(),
            output.len()
        ));
    }
    output.copy_from_slice(&values);
    Ok(true)
}

pub(crate) fn drafter_q8_0_dual_gemv(
    left_weight: &TensorView,
    right_weight: &TensorView,
    input: &[f32],
    left_output: &mut [f32],
    right_output: &mut [f32],
) -> Result<bool, String> {
    let Some(outputs) = q8_0_multi_gemv(&[left_weight, right_weight], &[input, input])? else {
        return Ok(false);
    };
    let [left, right]: [Vec<f32>; 2] = outputs.try_into().map_err(|outputs: Vec<Vec<f32>>| {
        format!("Metal drafter dual GEMV returned {} outputs", outputs.len())
    })?;
    if left.len() != left_output.len() || right.len() != right_output.len() {
        return Err("Metal drafter dual GEMV output length mismatch".to_string());
    }
    left_output.copy_from_slice(&left);
    right_output.copy_from_slice(&right);
    Ok(true)
}

pub(crate) fn drafter_q8_0_argmax(
    weight: &TensorView,
    input: &[f32],
) -> Result<Option<(usize, f32)>, String> {
    let mut logits = vec![0.0f32; weight.shape.first().copied().unwrap_or(0)];
    if !drafter_q8_0_gemv(weight, input, &mut logits)? {
        return Ok(None);
    }
    let mut best = (0usize, f32::NEG_INFINITY);
    for (token, &value) in logits.iter().enumerate() {
        if value > best.1 {
            best = (token, value);
        }
    }
    Ok(Some(best))
}
