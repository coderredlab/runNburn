//! Debug prefill layer snapshot helpers.

use super::*;

impl Engine {
    pub fn debug_prefill_layer_snapshots(
        &self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<PrefillLayerSnapshot>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => return Ok(Vec::new()),
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

        let mut snapshots = Vec::with_capacity(self.metadata.num_layers);
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

            let hidden_data = kernels::tensor_as_f32_slice(&hidden);
            let hidden_last = hidden_data
                [(seq_len - 1) * self.metadata.hidden_dim..seq_len * self.metadata.hidden_dim]
                .to_vec();

            let cache_layer_idx =
                shared_kv_source_layer(&self.metadata, self.architecture, layer_idx)
                    .unwrap_or(layer_idx);
            let cached = kv_cache.read_up_to(cache_layer_idx, pos_start + seq_len);
            let (cached_k, cached_v) = cached.as_slices();
            let cache_kv_dim = cached_k.len() / (pos_start + seq_len);
            let k_f32: Vec<f32> = cached_k
                [(pos_start + seq_len - 1) * cache_kv_dim..(pos_start + seq_len) * cache_kv_dim]
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let v_f32: Vec<f32> = cached_v
                [(pos_start + seq_len - 1) * cache_kv_dim..(pos_start + seq_len) * cache_kv_dim]
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();

            snapshots.push(PrefillLayerSnapshot {
                layer_idx,
                hidden_last,
                cache_layer_idx,
                cached_k_last: k_f32,
                cached_v_last: v_f32,
            });
        }

        Ok(snapshots)
    }
}
