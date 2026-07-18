//! Debug prefill drift trace helpers.

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct PrefillDriftTrace {
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub tokens: Vec<u32>,
    pub records: Vec<PrefillDriftRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefillDriftRecord {
    pub layer_idx: Option<usize>,
    pub stage: &'static str,
    pub row: Vec<f32>,
}

impl PrefillDriftTrace {
    pub fn validate(&self) -> Result<(), String> {
        for record in &self.records {
            if record.row.len() != self.hidden_dim {
                return Err(format!(
                    "record stage={} layer={:?} row_len={} hidden_dim={}",
                    record.stage,
                    record.layer_idx,
                    record.row.len(),
                    self.hidden_dim
                ));
            }
        }
        Ok(())
    }
}

fn last_token_row(tensor: &Tensor, seq_len: usize, hidden_dim: usize) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let data = kernels::tensor_as_f32_slice(tensor);
    let start = (seq_len - 1) * hidden_dim;
    data[start..start + hidden_dim].to_vec()
}

fn last_token_tensor(tensor: &Tensor, seq_len: usize, hidden_dim: usize) -> Tensor {
    Tensor::from_slice(
        &last_token_row(tensor, seq_len, hidden_dim),
        &[1, hidden_dim],
    )
}

impl Engine {
    pub fn debug_prefill_drift_layer_outputs(
        &self,
        tokens: &[u32],
    ) -> crate::error::Result<PrefillDriftTrace> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward("drift trace requires weights".into())
        })?;

        let hidden_dim = self.metadata.hidden_dim;
        if tokens.is_empty() {
            return Ok(PrefillDriftTrace {
                hidden_dim,
                seq_len: 0,
                tokens: Vec::new(),
                records: Vec::new(),
            });
        }

        let seq_len = tokens.len();
        let pos_start = self.kv_cache.current_len();
        let num_heads = self.metadata.num_heads;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rope_theta = self.metadata.rope_theta;
        let norm_eps = self.metadata.norm_eps;

        let mut kv_cache = self.kv_cache.clone();
        let raw_hidden = weights.token_embd.gather(tokens)?;
        let mut hidden =
            apply_embedding_scale(raw_hidden.clone(), &self.metadata, self.architecture);
        let gemma_per_layer_base = prepare_gemma_per_layer_base(
            weights,
            if gemma_ple_pre_emb_scale_base() {
                &raw_hidden
            } else {
                &hidden
            },
            tokens,
            &self.metadata,
            self.architecture,
            norm_eps,
        )?;

        let mut trace = PrefillDriftTrace {
            hidden_dim,
            seq_len,
            tokens: tokens.to_vec(),
            records: vec![PrefillDriftRecord {
                layer_idx: None,
                stage: "embedding_scaled",
                row: last_token_row(&hidden, seq_len, hidden_dim),
            }],
        };

        for layer_idx in 0..self.metadata.num_layers {
            hidden = run_prefill_layers_cpu_range(
                &mut kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                gemma_per_layer_base.as_ref(),
                hidden,
                layer_idx..layer_idx + 1,
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
            )?;

            trace.records.push(PrefillDriftRecord {
                layer_idx: Some(layer_idx),
                stage: "layer_output",
                row: last_token_row(&hidden, seq_len, hidden_dim),
            });
        }

        let hidden = if gemma_ple_after_final_norm() {
            if let Some(base) = prepare_gemma_per_layer_base(
                weights,
                &hidden,
                &[0u32; 0],
                &self.metadata,
                self.architecture,
                norm_eps,
            )? {
                if let Some(gemma) = weights.gemma_per_layer.as_ref() {
                    let layer_idx = self.metadata.num_layers.saturating_sub(1);
                    let updated = apply_gemma_per_layer_branch(
                        hidden,
                        &base,
                        layer_idx,
                        gemma,
                        &self.metadata,
                        self.architecture,
                        norm_eps,
                    )?;
                    trace.records.push(PrefillDriftRecord {
                        layer_idx: Some(layer_idx),
                        stage: "final_gemma_per_layer_output",
                        row: last_token_row(&updated, seq_len, hidden_dim),
                    });
                    updated
                } else {
                    hidden
                }
            } else {
                hidden
            }
        } else {
            hidden
        };

        let last_hidden = last_token_tensor(&hidden, seq_len, hidden_dim);
        let gemma_runtime_flavor = detect_gemma_runtime_flavor(&self.metadata, weights);
        let normed = if gemma_skip_output_norm() {
            last_hidden
        } else if gemma_effective_unit_offset_output_norm_prefill(
            self.architecture,
            gemma_runtime_flavor,
        ) {
            kernels::norm::rms_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else {
            apply_model_norm(
                &last_hidden,
                &weights.output_norm,
                norm_eps,
                self.architecture,
            )
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        };

        trace.records.push(PrefillDriftRecord {
            layer_idx: None,
            stage: "final_normed",
            row: kernels::tensor_as_f32_slice(&normed).to_vec(),
        });
        kv_cache.set_len(pos_start + seq_len);

        trace.validate().map_err(crate::error::LlmError::Forward)?;
        Ok(trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_trace_validate_accepts_matching_rows() {
        let trace = PrefillDriftTrace {
            hidden_dim: 3,
            seq_len: 2,
            tokens: vec![1, 2],
            records: vec![PrefillDriftRecord {
                layer_idx: Some(0),
                stage: "layer_output",
                row: vec![1.0, 2.0, 3.0],
            }],
        };

        assert_eq!(trace.validate(), Ok(()));
    }

    #[test]
    fn drift_trace_validate_rejects_bad_row_len() {
        let trace = PrefillDriftTrace {
            hidden_dim: 3,
            seq_len: 2,
            tokens: vec![1, 2],
            records: vec![PrefillDriftRecord {
                layer_idx: Some(0),
                stage: "layer_output",
                row: vec![1.0, 2.0],
            }],
        };

        let err = trace.validate().unwrap_err();
        assert!(err.contains("row_len=2"));
        assert!(err.contains("hidden_dim=3"));
    }

    #[test]
    fn last_token_row_extracts_tail_row() {
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        assert_eq!(last_token_row(&tensor, 2, 3), vec![4.0, 5.0, 6.0]);
    }
}
