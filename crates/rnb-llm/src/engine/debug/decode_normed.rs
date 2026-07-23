//! Debug next-decode normed hidden helper.

use super::*;
use crate::engine::norm::apply_model_norm_unit_offset;

impl Engine {
    pub fn debug_decode_next_hidden_normed(
        &mut self,
        token: u32,
    ) -> crate::error::Result<Vec<f32>> {
        let (hidden, _logits) = self.debug_decode_next_hidden_and_logits(token)?;
        Ok(hidden)
    }

    /// 한 번의 decode forward 안에서 normalized hidden 과 logits 을 동시에
    /// 돌려준다. MTP drafter calibration trace dump 처럼 hidden + argmax 페어가
    /// 필요한 외부 도구용 helper.
    ///
    /// 반환:
    /// - `hidden`: `output_norm` 적용 직후의 길이 `hidden_dim` 벡터
    /// - `logits`: `forward(&[token])` 가 돌려준 그대로의 logits (또는 backend
    ///   argmax-only path 라면 empty — caller 가 `last_backend_argmax_token` 으로
    ///   fallback)
    pub fn debug_decode_next_hidden_and_logits(
        &mut self,
        token: u32,
    ) -> crate::error::Result<(Vec<f32>, Vec<f32>)> {
        if self.weights.is_none() {
            let hidden_dim = self.metadata.hidden_dim;
            return Ok((vec![0.0f32; hidden_dim], Vec::new()));
        }
        let norm_eps = self.metadata.norm_eps;

        let logits = self.forward(&[token])?;

        let weights = self.weights.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward("weights missing after decode".into())
        })?;
        let scratch = self.scratch.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward("scratch missing after decode".into())
        })?;
        let hidden = &scratch.hidden[..self.metadata.hidden_dim];
        let hidden_tensor = Tensor::from_slice(hidden, &[1, self.metadata.hidden_dim]);
        let gemma_runtime_flavor = detect_gemma_runtime_flavor(&self.metadata, weights);
        let normed = if gemma_skip_output_norm() {
            hidden_tensor
        } else if gemma_effective_unit_offset_output_norm_decode(
            self.architecture,
            gemma_runtime_flavor,
        ) {
            apply_model_norm_unit_offset(&hidden_tensor, &weights.output_norm, norm_eps)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else {
            apply_model_norm(
                &hidden_tensor,
                &weights.output_norm,
                norm_eps,
                self.architecture,
            )
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        };
        let hidden_out = kernels::tensor_as_f32_slice(&normed).to_vec();
        Ok((hidden_out, logits))
    }
}
