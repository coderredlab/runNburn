//! Draft decode path used by speculative decoding.

use super::*;

impl Engine {
    /// Early-exit draft forward: 앞쪽 max_layer개 레이어만 실행.
    /// Speculative decoding의 draft phase에서 사용.
    pub(crate) fn forward_decode_draft(
        &mut self,
        token: u32,
        max_layer: usize,
    ) -> crate::error::Result<Vec<f32>> {
        let mut scratch = self.scratch.take().expect("ScratchBuffers not initialized");
        let metadata = &self.metadata;
        let architecture = self.architecture;
        let weights = self.weights.as_ref().unwrap();
        let hidden_dim = metadata.hidden_dim;

        // 1. Embedding lookup → scratch.hidden
        {
            let embd = &weights.token_embd;
            let embd_bytes = embd.data.as_bytes().expect("token_embd bytes");
            let row_bytes = embd_bytes.len() / embd.rows;
            let row_start = token as usize * row_bytes;
            let row_data = &embd_bytes[row_start..row_start + row_bytes];
            let f32_row = dequantize_bytes_to_f32(row_data, embd.ggml_type);
            scratch.hidden[..hidden_dim].copy_from_slice(&f32_row[..hidden_dim]);
        }
        apply_embedding_scale_inplace(&mut scratch.hidden[..hidden_dim], metadata, architecture);
        let gemma_per_layer_base = prepare_gemma_per_layer_base(
            weights,
            &Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
            &[token],
            metadata,
            architecture,
            metadata.norm_eps,
        )?;

        let pos = self.kv_cache.current_len();
        let actual_max = max_layer.min(metadata.num_layers);

        // 2. Layer loop: 0..actual_max only (no profiling, no gpu_runtime for draft)
        let kv_cache = &mut self.kv_cache;
        for layer_idx in 0..actual_max {
            let dynamic_base_storage = if gemma_ple_dynamic_base() {
                prepare_gemma_per_layer_base(
                    weights,
                    &Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                    &[token],
                    metadata,
                    architecture,
                    metadata.norm_eps,
                )?
            } else {
                None
            };
            let ple_base = if gemma_ple_dynamic_base() {
                dynamic_base_storage.as_ref()
            } else {
                gemma_per_layer_base.as_ref()
            };
            let ple_base = if gemma_ple_layer_enabled(layer_idx) {
                ple_base
            } else {
                None
            };
            let ple_after_out_scale = gemma_ple_after_out_scale()
                || gemma_ple_layer34_hard_fix_applies(architecture, layer_idx, metadata.num_layers);
            if gemma_ple_before_layer() {
                if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                    let updated = apply_gemma_per_layer_branch(
                        Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                        base,
                        layer_idx,
                        gemma,
                        metadata,
                        architecture,
                        metadata.norm_eps,
                    )?;
                    let updated_data = kernels::tensor_as_f32_slice(&updated);
                    scratch.hidden[..hidden_dim].copy_from_slice(&updated_data[..hidden_dim]);
                }
            }
            match &weights.layers[layer_idx] {
                LayerType::Attention(w) => {
                    if !gemma_ple_disable_attention_layer(layer_idx) {
                        decode_attention_layer(
                            kv_cache,
                            metadata,
                            architecture,
                            &mut scratch,
                            w,
                            weights.rope_freqs.as_ref(),
                            layer_idx,
                            pos,
                            None,
                            None,
                            None,
                            None,
                            None,
                            #[cfg(feature = "vulkan")]
                            None,
                        )?;
                    }
                }
                LayerType::GatedDeltaNet(w) => {
                    decode_gdn_layer(
                        kv_cache,
                        metadata,
                        &mut scratch,
                        w,
                        layer_idx,
                        #[cfg(feature = "vulkan")]
                        None,
                    )?;
                }
                LayerType::NemotronMamba2(w) => {
                    models::nemotron::mamba::decode_mamba2_layer(
                        kv_cache,
                        metadata,
                        &mut scratch,
                        w,
                        layer_idx,
                        metadata.norm_eps,
                    )?;
                }
                LayerType::NemotronMoE(w) => {
                    models::nemotron::moe::decode_moe_layer(
                        metadata,
                        &mut scratch,
                        w,
                        metadata.norm_eps,
                    )?;
                }
            }
            if gemma_ple_before_layer() {
                if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                    apply_layer_output_scale_inplace(
                        &mut scratch.hidden[..hidden_dim],
                        w.out_scale.as_ref(),
                        layer_idx,
                    );
                }
                continue;
            }
            if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                if !ple_after_out_scale {
                    if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref())
                    {
                        let updated = apply_gemma_per_layer_branch(
                            Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                            base,
                            layer_idx,
                            gemma,
                            metadata,
                            architecture,
                            metadata.norm_eps,
                        )?;
                        let updated_data = kernels::tensor_as_f32_slice(&updated);
                        scratch.hidden[..hidden_dim].copy_from_slice(&updated_data[..hidden_dim]);
                    }
                }
                apply_layer_output_scale_inplace(
                    &mut scratch.hidden[..hidden_dim],
                    w.out_scale.as_ref(),
                    layer_idx,
                );
                if ple_after_out_scale {
                    if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref())
                    {
                        let updated = apply_gemma_per_layer_branch(
                            Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                            base,
                            layer_idx,
                            gemma,
                            metadata,
                            architecture,
                            metadata.norm_eps,
                        )?;
                        let updated_data = kernels::tensor_as_f32_slice(&updated);
                        scratch.hidden[..hidden_dim].copy_from_slice(&updated_data[..hidden_dim]);
                    }
                }
            } else if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                let updated = apply_gemma_per_layer_branch(
                    Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                    base,
                    layer_idx,
                    gemma,
                    metadata,
                    architecture,
                    metadata.norm_eps,
                )?;
                let updated_data = kernels::tensor_as_f32_slice(&updated);
                scratch.hidden[..hidden_dim].copy_from_slice(&updated_data[..hidden_dim]);
            }
        }
        // 3. Final norm + output logits
        let output_norm_data = kernels::tensor_as_f32_slice(&weights.output_norm);
        if gemma_skip_output_norm() {
            scratch.norm_buf[..hidden_dim].copy_from_slice(&scratch.hidden[..hidden_dim]);
        } else {
            apply_model_norm_into(
                &scratch.hidden[..hidden_dim],
                output_norm_data,
                metadata.norm_eps,
                &mut scratch.norm_buf[..hidden_dim],
                architecture,
            );
        }
        weights
            .output
            .gemv_into(&scratch.norm_buf[..hidden_dim], &mut scratch.logits)?;
        apply_logit_softcapping(&mut scratch.logits, metadata.final_logit_softcapping);

        // 4. KV cache update
        kv_cache.set_len(pos + 1);

        let result = scratch.logits.clone();
        self.scratch = Some(scratch);
        Ok(result)
    }
}
