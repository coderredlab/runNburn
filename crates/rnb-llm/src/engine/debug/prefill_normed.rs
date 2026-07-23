//! Debug prefill normed hidden helpers.

use super::*;
use crate::engine::norm::apply_model_norm_unit_offset;

impl Engine {
    pub fn debug_prefill_last_hidden_normed(
        &self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<f32>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => return Ok(vec![0.0f32; self.metadata.hidden_dim]),
        };
        if tokens.is_empty() {
            return Ok(Vec::new());
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
        let hidden = apply_embedding_scale(raw_hidden.clone(), &self.metadata, self.architecture);
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

        let hidden = run_prefill_layers_cpu_range(
            &mut kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            gemma_per_layer_base.as_ref(),
            hidden,
            0..self.metadata.num_layers,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )?;

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
                    apply_gemma_per_layer_branch(
                        hidden,
                        &base,
                        layer_idx,
                        gemma,
                        &self.metadata,
                        self.architecture,
                        norm_eps,
                    )?
                } else {
                    hidden
                }
            } else {
                hidden
            }
        } else {
            hidden
        };

        let last_hidden = if seq_len > 1 {
            let hidden_data = kernels::tensor_as_f32_slice(&hidden);
            let hidden_dim = self.metadata.hidden_dim;
            let start = (seq_len - 1) * hidden_dim;
            Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim])
        } else {
            hidden
        };
        let gemma_runtime_flavor = detect_gemma_runtime_flavor(&self.metadata, weights);
        let normed = if gemma_skip_output_norm() {
            last_hidden
        } else if gemma_effective_unit_offset_output_norm_prefill(
            self.architecture,
            gemma_runtime_flavor,
        ) {
            apply_model_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
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
        kv_cache.set_len(pos_start + seq_len);
        Ok(kernels::tensor_as_f32_slice(&normed).to_vec())
    }

    pub fn debug_prefill_layer_hidden_normed(
        &self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<Vec<f32>>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => {
                return Ok(vec![
                    vec![0.0f32; self.metadata.hidden_dim];
                    self.metadata.num_layers
                ])
            }
        };
        if tokens.is_empty() {
            return Ok(Vec::new());
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
        let gemma_runtime_flavor = detect_gemma_runtime_flavor(&self.metadata, weights);

        let mut result = Vec::with_capacity(self.metadata.num_layers);
        for layer_idx in 0..self.metadata.num_layers {
            hidden = run_prefill_layers_cpu_range(
                &mut kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                gemma_per_layer_base.as_ref(),
                hidden.clone(),
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

            let hidden =
                if gemma_ple_after_final_norm() && layer_idx + 1 == self.metadata.num_layers {
                    if let Some(base) = prepare_gemma_per_layer_base(
                        weights,
                        &hidden,
                        &[0u32; 0],
                        &self.metadata,
                        self.architecture,
                        norm_eps,
                    )? {
                        if let Some(gemma) = weights.gemma_per_layer.as_ref() {
                            apply_gemma_per_layer_branch(
                                hidden.clone(),
                                &base,
                                layer_idx,
                                gemma,
                                &self.metadata,
                                self.architecture,
                                norm_eps,
                            )?
                        } else {
                            hidden.clone()
                        }
                    } else {
                        hidden.clone()
                    }
                } else {
                    hidden.clone()
                };

            let last_hidden = if seq_len > 1 {
                let hidden_data = kernels::tensor_as_f32_slice(&hidden);
                let hidden_dim = self.metadata.hidden_dim;
                let start = (seq_len - 1) * hidden_dim;
                Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim])
            } else {
                hidden
            };
            let normed = if gemma_skip_output_norm() {
                last_hidden
            } else if gemma_effective_unit_offset_output_norm_prefill(
                self.architecture,
                gemma_runtime_flavor,
            ) {
                apply_model_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
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
            result.push(kernels::tensor_as_f32_slice(&normed).to_vec());
        }
        kv_cache.set_len(pos_start + seq_len);
        Ok(result)
    }
}
