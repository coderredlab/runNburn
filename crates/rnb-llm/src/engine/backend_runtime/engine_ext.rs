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
            if let Some(runtime) = self.mtp_runtime.as_ref() {
                if let Some(raw) = runtime.eh_proj_raw_bytes() {
                    slices.push(raw);
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

    pub fn backend_output_argmax_supported_for_runtime(&self) -> bool {
        let Some(weights) = self.weights.as_ref() else {
            return false;
        };
        backend_output_argmax_supported_for_runtime(weights)
    }

    pub fn clear_sequence_state(&mut self) -> crate::error::Result<()> {
        self.kv_cache.clear();
        self.mtp_clear_sequence_state();
        self.backend_runtime.clear_sequence_state()
    }

    #[allow(dead_code)]
    pub(crate) fn materialize_sequence_state(&mut self) -> crate::error::Result<()> {
        #[cfg(feature = "cuda")]
        {
            for ssm in &mut self.kv_cache.ssm_states {
                let Some(state) = ssm.as_mut() else {
                    continue;
                };
                cuda_runtime::sync_delta_state_cache(&mut state.delta_state)
                    .map_err(crate::error::LlmError::Forward)?;
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

#[cfg(any(feature = "cuda", test))]
pub(crate) fn should_invalidate_resident_sequence_state_after_restore(
    restored_resident: bool,
) -> bool {
    !restored_resident
}

#[cfg(test)]
mod tests {
    #[test]
    fn resident_delta_restore_keeps_weight_cache_policy() {
        assert!(!super::should_invalidate_resident_sequence_state_after_restore(true));
    }

    #[test]
    fn missing_resident_delta_restore_invalidates_sequence_state_policy() {
        assert!(super::should_invalidate_resident_sequence_state_after_restore(false));
    }
}
