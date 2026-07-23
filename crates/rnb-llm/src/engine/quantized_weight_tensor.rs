#[cfg(not(feature = "cuda"))]
use super::dequant::dequantize_bytes_to_f32;
use super::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
#[cfg(not(feature = "cuda"))]
use rnb_loader::GGMLType;

impl QuantizedWeight {
    #[cfg(feature = "cuda")]
    pub(super) fn gemv_vec_exact_f32(&self, input: &[f32]) -> crate::error::Result<Vec<f32>> {
        self.gemv_vec(input)
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn gemv_vec_exact_f32(&self, input: &[f32]) -> crate::error::Result<Vec<f32>> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let seq_len = input.len() / self.cols;
        let bytes_per_row = bytes.len() / self.rows;
        let mut output = vec![0.0f32; seq_len * self.rows];

        for row in 0..self.rows {
            let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let row_f32 = dequantize_bytes_to_f32(row_bytes, self.ggml_type);
            for s in 0..seq_len {
                let x = &input[s * self.cols..(s + 1) * self.cols];
                let acc = row_f32
                    .iter()
                    .zip(x.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
                output[s * self.rows + row] = acc;
            }
        }
        Ok(output)
    }

    /// Embedding gather: token indices -> F32 [seq_len, cols] (on-the-fly dequant)
    #[cfg(feature = "cuda")]
    pub(super) fn gather(&self, indices: &[u32]) -> crate::error::Result<Tensor> {
        let out = self.embedding_gather_cuda(indices)?;
        Ok(Tensor::from_vec(out, &[indices.len(), self.cols]))
    }

    /// Embedding gather: token indices -> F32 [seq_len, cols] (on-the-fly dequant)
    #[cfg(not(feature = "cuda"))]
    pub(super) fn gather(&self, indices: &[u32]) -> crate::error::Result<Tensor> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("gather: no bytes".into()))?;
        let bytes_per_row = bytes.len() / self.rows;
        let mut out = Vec::with_capacity(indices.len() * self.cols);

        if self.ggml_type == GGMLType::F32 {
            // F32 direct: no dequant needed.
            let f32_data = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.rows * self.cols)
            };
            for &idx in indices {
                let start = idx as usize * self.cols;
                out.extend_from_slice(&f32_data[start..start + self.cols]);
            }
        } else {
            for &idx in indices {
                let row_start = idx as usize * bytes_per_row;
                let row_bytes = &bytes[row_start..row_start + bytes_per_row];
                let f32_row = dequantize_bytes_to_f32(row_bytes, self.ggml_type);
                out.extend_from_slice(&f32_row);
            }
        }
        Ok(Tensor::from_vec(out, &[indices.len(), self.cols]))
    }
}
