//! Quantized prefill dispatch helpers.

use super::*;
#[cfg(target_arch = "aarch64")]
use std::cell::OnceCell;

pub(in crate::engine) struct PrefillQuantizedInput<'a> {
    #[cfg(target_arch = "aarch64")]
    input: &'a [f32],
    #[cfg(target_arch = "aarch64")]
    q8k_enabled: bool,
    #[cfg(target_arch = "aarch64")]
    q8_enabled: bool,
    #[cfg(target_arch = "aarch64")]
    q8k: OnceCell<Vec<Q8KBlock>>,
    #[cfg(target_arch = "aarch64")]
    q8: OnceCell<Vec<Q8Block>>,
    #[cfg(not(target_arch = "aarch64"))]
    _marker: std::marker::PhantomData<&'a [f32]>,
}

#[cfg(target_arch = "aarch64")]
impl PrefillQuantizedInput<'_> {
    fn q8k(&self) -> Option<&[Q8KBlock]> {
        if !self.q8k_enabled {
            return None;
        }
        Some(
            self.q8k
                .get_or_init(|| gemm_runtime::quantize_input_q8k(self.input))
                .as_slice(),
        )
    }

    fn q8(&self) -> Option<&[Q8Block]> {
        if !self.q8_enabled {
            return None;
        }
        Some(
            self.q8
                .get_or_init(|| gemm_runtime::quantize_input_q8(self.input))
                .as_slice(),
        )
    }
}

pub(in crate::engine) fn prefill_quantized_input_for_weight<'a>(
    weight: &QuantizedWeight,
    input: &'a [f32],
) -> PrefillQuantizedInput<'a> {
    #[cfg(target_arch = "aarch64")]
    {
        let has_dotprod = fast_dotprod_enabled();
        PrefillQuantizedInput {
            input,
            q8k_enabled: has_dotprod && k_quant_q8k_candidate(weight),
            q8_enabled: has_dotprod,
            q8k: OnceCell::new(),
            q8: OnceCell::new(),
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (weight, input);
        PrefillQuantizedInput {
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn q8_direct_candidate(weight: &QuantizedWeight) -> bool {
    weight.q4_0_data.is_some() || matches!(weight.ggml_type, GGMLType::Q4_0 | GGMLType::Q8_0)
}

pub(in crate::engine) fn prefill_gemv_vec(
    weight: &QuantizedWeight,
    input: &[f32],
    quantized: &PrefillQuantizedInput<'_>,
) -> crate::error::Result<Vec<f32>> {
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(q8k) = quantized.q8k() {
            return weight.gemv_vec_q8k(q8k);
        }
        if q8_direct_candidate(weight) {
            if let Some(q8) = quantized.q8() {
                return weight.gemv_vec_q8(q8);
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    let _ = quantized;

    weight.gemv_vec(input)
}

pub(in crate::engine) fn prefill_dual_gemv_q8_or_f32(
    left: &QuantizedWeight,
    right: &QuantizedWeight,
    input: &[f32],
    quantized: &PrefillQuantizedInput<'_>,
) -> crate::error::Result<(Vec<f32>, Vec<f32>)> {
    #[cfg(target_arch = "aarch64")]
    {
        if q8_direct_candidate(left) && q8_direct_candidate(right) {
            if let Some(q8) = quantized.q8() {
                return Ok((left.gemv_vec_q8(q8)?, right.gemv_vec_q8(q8)?));
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    let _ = quantized;

    Ok((left.gemv_vec(input)?, right.gemv_vec(input)?))
}

pub(in crate::engine) fn prefill_gate_up_vectors(
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    fused_gate_up: Option<&QuantizedWeight>,
    input: &[f32],
    seq_len: usize,
) -> crate::error::Result<(Vec<f32>, Vec<f32>)> {
    #[cfg(not(target_arch = "aarch64"))]
    let _ = (fused_gate_up, seq_len);

    #[cfg(target_arch = "aarch64")]
    {
        if fast_dotprod_enabled() {
            if k_quant_q8k_candidate(gate_weight) {
                let q8k = gemm_runtime::quantize_input_q8k(input);
                if gate_weight.ggml_type == GGMLType::Q4_K
                    && up_weight.ggml_type == GGMLType::Q4_K
                    && gate_weight.rows == up_weight.rows
                    && gate_weight.cols == up_weight.cols
                {
                    if let (Some(gate_bytes), Some(up_bytes)) =
                        (gate_weight.data.as_bytes(), up_weight.data.as_bytes())
                    {
                        let bytes_per_row = gate_bytes.len() / gate_weight.rows;
                        if up_bytes.len() / up_weight.rows == bytes_per_row {
                            let mut gate = vec![0.0f32; seq_len * gate_weight.rows];
                            let mut up = vec![0.0f32; seq_len * up_weight.rows];
                            prefill_raw_dual_q4k_q8k(
                                gate_bytes,
                                up_bytes,
                                &q8k,
                                &mut gate,
                                &mut up,
                                gate_weight.rows,
                                gate_weight.cols,
                                seq_len,
                                bytes_per_row,
                            );
                            return Ok((gate, up));
                        }
                    }
                }
                return Ok((
                    gate_weight.gemv_vec_q8k(&q8k)?,
                    up_weight.gemv_vec_q8k(&q8k)?,
                ));
            }

            let q8 = gemm_runtime::quantize_input_q8(input);
            if let Some(fused) = fused_gate_up {
                let combined = fused.gemv_vec_q8(&q8)?;
                let gate_rows = gate_weight.rows;
                let up_rows = up_weight.rows;
                let total_rows = gate_rows + up_rows;
                let mut gate = Vec::with_capacity(seq_len * gate_rows);
                let mut up = Vec::with_capacity(seq_len * up_rows);
                for s in 0..seq_len {
                    let base = s * total_rows;
                    for i in 0..gate_rows {
                        gate.push(combined[base + i]);
                    }
                    for i in 0..up_rows {
                        up.push(combined[base + gate_rows + i]);
                    }
                }
                return Ok((gate, up));
            }

            return Ok((gate_weight.gemv_vec_q8(&q8)?, up_weight.gemv_vec_q8(&q8)?));
        }
    }

    Ok((gate_weight.gemv_vec(input)?, up_weight.gemv_vec(input)?))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn prefill_raw_quantized_batch(
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
    ggml_type: GGMLType,
) {
    #[cfg(target_arch = "aarch64")]
    if seq_len > 1
        && cols % 256 == 0
        && matches!(ggml_type, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K)
    {
        let q8k = gemm_runtime::quantize_input_q8k(input);
        match ggml_type {
            GGMLType::Q4_K => {
                gemv_q4_k_int8(bytes, &q8k, output, rows, cols, seq_len, bytes_per_row)
            }
            GGMLType::Q5_K => {
                gemv_q5_k_int8(bytes, &q8k, output, rows, cols, seq_len, bytes_per_row)
            }
            GGMLType::Q6_K => {
                gemv_q6_k_int8(bytes, &q8k, output, rows, cols, seq_len, bytes_per_row)
            }
            _ => unreachable!(),
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if ggml_type == GGMLType::Q5_0 && seq_len > 1 && cols % 32 == 0 {
        let q8 = gemm_runtime::quantize_input_q8(input);
        gemv_q5_0_int8(bytes, &q8, output, rows, cols, seq_len, bytes_per_row);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if ggml_type == GGMLType::Q8_0 && seq_len > 1 && cols % 32 == 0 {
        let q8 = gemm_runtime::quantize_input_q8(input);
        gemv_q8_0_int8(bytes, &q8, output, rows, cols, seq_len, bytes_per_row);
        return;
    }

    crate::engine::scalar_gemv::gemv_generic(
        bytes,
        input,
        output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
        ggml_type,
    );
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn quantize_raw_q8k(input: &[f32]) -> Vec<Q8KBlock> {
    gemm_runtime::quantize_input_q8k(input)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn prefill_raw_dual_q4k_q8k(
    left_bytes: &[u8],
    right_bytes: &[u8],
    q8k: &[Q8KBlock],
    left_output: &mut [f32],
    right_output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    gemv_q4_k_int8_dual(
        left_bytes,
        right_bytes,
        q8k,
        left_output,
        right_output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
    );
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn prefill_raw_split_q4k_q8k(
    bytes: &[u8],
    q8k: &[Q8KBlock],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    rows_per_projection: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    let split = rows_per_projection * bytes_per_row;
    gemv_q4_k_int8_dual(
        &bytes[..split],
        &bytes[split..],
        q8k,
        gate_output,
        up_output,
        rows_per_projection,
        cols,
        seq_len,
        bytes_per_row,
    );
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::engine::quantized_weight_types::QuantizedWeight;
    use rnb_core::tensor::Tensor;

    fn f32_weight(rows: usize, cols: usize, value: f32) -> QuantizedWeight {
        QuantizedWeight::new(
            Tensor::from_vec(vec![value; rows * cols], &[rows, cols]),
            GGMLType::F32,
            rows,
            cols,
        )
    }

    fn q4k_candidate_weight() -> QuantizedWeight {
        let raw = vec![0u8; 144];
        QuantizedWeight::new(Tensor::from_vec(raw, &[144]), GGMLType::Q4_K, 1, 256)
    }

    #[test]
    fn prefill_quantized_input_defers_activation_quantization_until_used() {
        let weight = q4k_candidate_weight();
        let input = vec![0.25f32; 256];

        let quantized = prefill_quantized_input_for_weight(&weight, &input);

        assert!(quantized.q8.get().is_none());
        assert!(quantized.q8k.get().is_none());
        assert!(quantized.q8k().is_some());
        assert!(quantized.q8k.get().is_some());
        assert!(quantized.q8.get().is_none());
    }

    #[test]
    fn prefill_dual_gemv_keeps_f32_inputs_unquantized() {
        let left = f32_weight(1, 32, 1.0);
        let right = f32_weight(1, 32, 2.0);
        let mut input = vec![0.001f32; 32];
        input[0] = 1.0;
        let quantized = prefill_quantized_input_for_weight(&left, &input);

        let (left_out, right_out) =
            prefill_dual_gemv_q8_or_f32(&left, &right, &input, &quantized).expect("dual gemv");

        let expected_left = left.gemv_vec(&input).expect("direct left");
        let expected_right = right.gemv_vec(&input).expect("direct right");
        assert_eq!(left_out, expected_left);
        assert_eq!(right_out, expected_right);
        assert!(quantized.q8.get().is_none());
    }
}
