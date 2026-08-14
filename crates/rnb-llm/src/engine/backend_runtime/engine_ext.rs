use crate::engine::state::Engine;

use super::{
    backend_output_argmax_supported_for_runtime, force_prewarm_backend_output_weight_for_runtime,
};

#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;

impl Engine {
    /// Returns true if model weights are loaded (not a mock engine).
    pub fn has_weights(&self) -> bool {
        self.weights.is_some() && self.scratch.is_some()
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp_runtime_ready()
    }

    pub fn prewarm_output_weight_for_runtime(&self) -> bool {
        let Some(weights) = self.weights.as_ref() else {
            return false;
        };
        force_prewarm_backend_output_weight_for_runtime(weights)
    }

    pub(crate) fn prewarm_mtp_device_verify_static_weights(&self) -> crate::error::Result<()> {
        let Some(weights) = self.weights.as_ref() else {
            return Ok(());
        };
        #[cfg(feature = "cuda")]
        {
            let mut slices = Vec::new();
            if let Some(raw) = weights.token_embd.data.as_bytes() {
                slices.push(raw);
            }
            if let Some(raw) = weights.output.data.as_bytes() {
                slices.push(raw);
            }
            fn push_weight<'a>(
                slices: &mut Vec<&'a [u8]>,
                weight: &'a crate::engine::quantized_weight_types::QuantizedWeight,
            ) {
                if let Some(raw) = weight.data.as_bytes() {
                    slices.push(raw);
                }
            }
            fn push_attention_weights<'a>(
                slices: &mut Vec<&'a [u8]>,
                attn: &'a crate::engine::layer_weights::AttentionLayerWeights,
            ) {
                for weight in [
                    &attn.q_weight,
                    &attn.k_weight,
                    &attn.v_weight,
                    &attn.o_weight,
                ] {
                    push_weight(slices, weight);
                }
                if let Some(moe) = attn.shared_expert_moe.as_ref() {
                    for weight in [&moe.shared_gate, &moe.shared_up, &moe.shared_down] {
                        push_weight(slices, weight);
                    }
                } else {
                    for weight in [
                        &attn.ffn_gate_weight,
                        &attn.ffn_up_weight,
                        &attn.ffn_down_weight,
                    ] {
                        push_weight(slices, weight);
                    }
                }
            }
            if let Some(mtp_weights) = self
                .mtp_runtime
                .as_ref()
                .and_then(crate::engine::mtp::EngineMtpRuntime::in_model_weights)
            {
                push_weight(&mut slices, &mtp_weights.eh_proj);
                if let Some(weight) = mtp_weights.embed_tokens.as_ref() {
                    push_weight(&mut slices, weight);
                }
                if let Some(weight) = mtp_weights.shared_head_head.as_ref() {
                    push_weight(&mut slices, weight);
                }
                push_attention_weights(&mut slices, &mtp_weights.block);
            }
            for layer in &weights.layers {
                match layer {
                    crate::engine::layer_weights::LayerType::Attention(attn) => {
                        push_attention_weights(&mut slices, attn);
                        if self.architecture == rnb_loader::Architecture::Gemma4 {
                            if let Some(moe) = attn.moe.as_ref() {
                                if let Some(raw) = moe.gate_up_bytes() {
                                    slices.push(raw);
                                }
                                if let Some(raw) = moe.down_bytes() {
                                    slices.push(raw);
                                }
                            }
                        }
                    }
                    crate::engine::layer_weights::LayerType::GatedDeltaNet(gdn) => {
                        for weight in [
                            &gdn.qkv_weight,
                            &gdn.gate_weight,
                            &gdn.ssm_alpha,
                            &gdn.ssm_beta,
                            &gdn.ssm_out,
                        ] {
                            push_weight(&mut slices, weight);
                        }
                        if let Some(moe) = gdn.shared_expert_moe.as_ref() {
                            for weight in [&moe.shared_gate, &moe.shared_up, &moe.shared_down] {
                                push_weight(&mut slices, weight);
                            }
                        } else {
                            for weight in [
                                &gdn.ffn_gate_weight,
                                &gdn.ffn_up_weight,
                                &gdn.ffn_down_weight,
                            ] {
                                push_weight(&mut slices, weight);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !slices.is_empty() {
                cuda_runtime::prewarm_q4k_weight_slices_pinned(&slices).map_err(|err| {
                    crate::error::LlmError::Forward(format!(
                        "CUDA MTP static weight prewarm failed: {err}"
                    ))
                })?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = weights;
        }
        Ok(())
    }

    pub(crate) fn ensure_mtp_device_verify_static_weights_prewarmed(
        &mut self,
    ) -> crate::error::Result<()> {
        #[cfg(feature = "cuda")]
        {
            let needs_prewarm = self
                .scratch
                .as_ref()
                .is_some_and(|scratch| !scratch.device_verify_static_weights_warmed);
            if !needs_prewarm {
                return Ok(());
            }
            self.prewarm_mtp_device_verify_static_weights()?;
            if let Some(scratch) = self.scratch.as_mut() {
                scratch.device_verify_static_weights_warmed = true;
            }
        }
        Ok(())
    }

    pub fn backend_output_argmax_supported_for_runtime(&self) -> bool {
        if !backend_output_argmax_architecture_supported(self.architecture) {
            return false;
        }
        let Some(weights) = self.weights.as_ref() else {
            return false;
        };
        backend_output_argmax_supported_for_runtime(weights)
    }

    pub fn clear_sequence_state(&mut self) -> crate::error::Result<()> {
        let has_loaded_model = self.has_weights();
        self.kv_cache.clear();
        #[cfg(feature = "cuda")]
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.cuda_decode_kv_authoritative = false;
        }
        self.sequence_cursor = None;
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.backend_argmax_token = None;
        }
        if let Some(model) = self
            .weights
            .as_mut()
            .and_then(|weights| weights.deepseek4.as_mut())
        {
            model.clear_state();
        }
        self.mtp_clear_sequence_state();
        if has_loaded_model {
            self.backend_runtime.clear_sequence_state()
        } else {
            Ok(())
        }
    }

    #[allow(dead_code)]
    pub(crate) fn materialize_sequence_state(&mut self) -> crate::error::Result<()> {
        #[cfg(feature = "cuda")]
        {
            let current_len = self.kv_cache.current_len();
            let max_kv_dim = (0..self.kv_cache.num_layers())
                .filter(|&layer_idx| self.kv_cache.layer_uses_kvarn(layer_idx))
                .map(|layer_idx| self.kv_cache.layer_kv_dim(layer_idx))
                .max()
                .unwrap_or(0);
            let mut materialized_key = vec![0u16; current_len.saturating_mul(max_kv_dim)];
            let mut materialized_value = vec![0u16; current_len.saturating_mul(max_kv_dim)];
            for layer_idx in 0..self.kv_cache.num_layers() {
                let kv_dim = self.kv_cache.layer_kv_dim(layer_idx);
                if self.kv_cache.layer_uses_kvarn(layer_idx) {
                    let count = current_len.saturating_mul(kv_dim);
                    let key = &mut materialized_key[..count];
                    let value = &mut materialized_value[..count];
                    let synced = cuda_runtime::sync_device_kv_cache_f16_to_host(
                        layer_idx,
                        key,
                        value,
                        kv_dim,
                        current_len,
                    )
                    .map_err(crate::error::LlmError::Forward)?;
                    if synced {
                        self.kv_cache
                            .replace_layer_f16_range_compacted(
                                layer_idx,
                                0,
                                current_len,
                                key,
                                value,
                            )
                            .map_err(crate::error::LlmError::Forward)?;
                    }
                } else {
                    let (key, value) = self.kv_cache.get_up_to_mut(layer_idx, current_len);
                    cuda_runtime::sync_device_kv_cache_f16_to_host(
                        layer_idx,
                        key,
                        value,
                        kv_dim,
                        current_len,
                    )
                    .map_err(crate::error::LlmError::Forward)?;
                }
            }
            for ssm in &mut self.kv_cache.ssm_states {
                let Some(state) = ssm.as_mut() else {
                    continue;
                };
                cuda_runtime::sync_delta_state_cache(&mut state.conv_state)
                    .map_err(crate::error::LlmError::Forward)?;
                cuda_runtime::sync_delta_state_cache(&mut state.delta_state)
                    .map_err(crate::error::LlmError::Forward)?;
            }
            if let Some(scratch) = self.scratch.as_mut() {
                scratch.cuda_decode_kv_authoritative = false;
            }
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            // pm31: delta residency 시 host delta 가 stale. speculative checkpoint 직전
            // device 잔류 delta 를 host 로 sync(carrier 없는 layer 는 no-op). cuda 분기와 동형.
            for (layer_idx, ssm) in self.kv_cache.ssm_states.iter_mut().enumerate() {
                let Some(state) = ssm.as_mut() else {
                    continue;
                };
                crate::engine::metal_runtime::metal_sync_delta_state(
                    layer_idx,
                    &mut state.delta_state,
                )
                .map_err(crate::error::LlmError::Forward)?;
            }
        }
        Ok(())
    }

    pub(crate) fn invalidate_resident_sequence_state_after_restore(
        &mut self,
    ) -> crate::error::Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.kv_cache.ssm_states.iter().any(Option::is_some) {
                cuda_runtime::reset_state_for_engine_init()
                    .map_err(crate::error::LlmError::Forward)?;
            }
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            crate::engine::metal_runtime::metal_clear_sequence_state()
                .map_err(crate::error::LlmError::Forward)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn finalize_resident_sequence_state_after_restore(
        &mut self,
        restored_resident: bool,
    ) -> crate::error::Result<()> {
        if should_invalidate_resident_sequence_state_after_restore(restored_resident) {
            self.invalidate_resident_sequence_state_after_restore()?;
        }
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn has_active_gpu_prefill_path(&self) -> bool {
        self.backend_runtime.has_active_gpu_prefill_path()
    }

    #[cfg(not(feature = "vulkan"))]
    pub(in crate::engine) fn has_active_gpu_prefill_path(&self) -> bool {
        false
    }

    #[cfg(feature = "vulkan")]
    pub fn prefill_runtime_counters(&self) -> Option<gpu::RuntimeCounters> {
        self.backend_runtime.prefill_runtime_counters()
    }

    pub(in crate::engine) fn prefill_chunk_size(&self) -> usize {
        #[cfg(feature = "vulkan")]
        {
            return gpu::active_prefill_chunk_size(
                self.has_active_gpu_prefill_path(),
                self.metadata.hidden_dim,
            );
        }
        #[cfg(not(feature = "vulkan"))]
        usize::MAX
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn decode_backend_layers_allowed(&self) -> bool {
        gpu::decode_layers_policy_allows()
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn decode_backend_max_layer(&self) -> usize {
        gpu::decode_layers_policy_max_layer()
    }
}

fn backend_output_argmax_architecture_supported(architecture: rnb_loader::Architecture) -> bool {
    architecture != rnb_loader::Architecture::DeepSeek4
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn should_invalidate_resident_sequence_state_after_restore(
    restored_resident: bool,
) -> bool {
    !restored_resident
}

#[cfg(test)]
mod tests {
    use rnb_loader::Architecture;

    #[test]
    fn deepseek4_disables_generic_backend_argmax_decode() {
        assert!(!super::backend_output_argmax_architecture_supported(
            Architecture::DeepSeek4
        ));
        assert!(super::backend_output_argmax_architecture_supported(
            Architecture::LLaMA
        ));
    }
    #[test]
    fn resident_delta_restore_keeps_weight_cache_policy() {
        assert!(!super::should_invalidate_resident_sequence_state_after_restore(true));
    }

    #[test]
    fn missing_resident_delta_restore_invalidates_sequence_state_policy() {
        assert!(super::should_invalidate_resident_sequence_state_after_restore(false));
    }
}
