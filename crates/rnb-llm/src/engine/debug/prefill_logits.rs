//! Debug prefill layer logits helpers.

use super::*;

impl Engine {
    /// Debug-only seam: collect the first-decode-step logits after each prefill layer.
    ///
    /// This does not mutate the engine state; it replays the CPU prefill path on a cloned KV
    /// cache so diagnostics can compare how the distribution evolves layer by layer.
    pub fn debug_prefill_layer_logits(
        &self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<Vec<f32>>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => {
                return Ok(vec![
                    vec![0.0f32; self.metadata.vocab_size];
                    self.metadata.num_layers
                ]);
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

        let mut per_layer_logits = Vec::with_capacity(self.metadata.num_layers);
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
            let logits = finalize_prefill_logits(
                &mut kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                hidden.clone(),
                seq_len,
                pos_start,
                norm_eps,
                None,
            )?;
            per_layer_logits.push(logits);
        }

        Ok(per_layer_logits)
    }
}
