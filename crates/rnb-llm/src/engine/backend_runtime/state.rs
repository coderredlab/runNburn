use crate::engine::layer_weights::ModelWeights;
use crate::engine::types::ModelMetadata;

#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime::LayerRuntime as RuntimeFacade;
#[cfg(all(feature = "vulkan", test))]
use crate::engine::gpu_runtime::Quant as RuntimeQuant;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(all(feature = "vulkan", test))]
use rnb_loader::GGMLType;

#[cfg(feature = "vulkan")]
pub(in crate::engine) type GpuRuntime = RuntimeFacade;
#[cfg(all(feature = "vulkan", test))]
pub(in crate::engine) type GpuQuant = RuntimeQuant;

#[derive(Default)]
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(in crate::engine) struct EngineBackendRuntime {
    #[cfg(feature = "vulkan")]
    gpu_layer_runtime: Option<GpuRuntime>,
    decode_gpu_layer_prefixes: Option<(usize, usize)>,
    base_decode_gpu_layer_prefixes: Option<(usize, usize)>,
}

impl EngineBackendRuntime {
    pub(in crate::engine) fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn from_gpu_runtime(gpu_layer_runtime: Option<GpuRuntime>) -> Self {
        Self {
            gpu_layer_runtime,
            decode_gpu_layer_prefixes: None,
            base_decode_gpu_layer_prefixes: None,
        }
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn take_gpu_runtime(&mut self) -> Option<GpuRuntime> {
        self.gpu_layer_runtime.take()
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn restore_gpu_runtime(&mut self, gpu_layer_runtime: Option<GpuRuntime>) {
        self.gpu_layer_runtime = gpu_layer_runtime;
    }
    pub(in crate::engine) fn set_initial_decode_gpu_layer_prefixes(
        &mut self,
        layers: Option<(usize, usize)>,
    ) {
        self.base_decode_gpu_layer_prefixes = layers;
        self.decode_gpu_layer_prefixes = layers;
    }

    pub(in crate::engine) fn restore_initial_decode_gpu_layer_prefixes(&mut self) {
        self.decode_gpu_layer_prefixes = self.base_decode_gpu_layer_prefixes;
    }

    pub(in crate::engine) fn set_decode_gpu_layer_prefixes(
        &mut self,
        layers: Option<(usize, usize)>,
    ) {
        self.decode_gpu_layer_prefixes = layers;
    }

    pub(in crate::engine) fn decode_attention_layer_uses_gpu(&self, layer_idx: usize) -> bool {
        self.decode_gpu_layer_prefixes
            .is_none_or(|(attention_layers, _)| layer_idx < attention_layers)
    }

    pub(in crate::engine) fn decode_ffn_layer_uses_gpu(&self, layer_idx: usize) -> bool {
        self.decode_gpu_layer_prefixes
            .is_none_or(|(_, ffn_layers)| layer_idx < ffn_layers)
    }
    pub(in crate::engine) fn decode_all_layers_use_gpu(&self, num_layers: usize) -> bool {
        self.decode_gpu_layer_prefixes
            .is_none_or(|(attention_layers, ffn_layers)| {
                attention_layers >= num_layers && ffn_layers >= num_layers
            })
    }

    pub(in crate::engine) fn decode_embedding_uses_gpu(&self) -> bool {
        self.decode_gpu_layer_prefixes.is_none()
    }
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(in crate::engine) fn has_active_gpu_prefill_path(&self) -> bool {
        #[cfg(feature = "vulkan")]
        {
            return self.gpu_layer_runtime.is_some();
        }
        #[cfg(not(feature = "vulkan"))]
        false
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn prefill_runtime_counters(&self) -> Option<gpu::RuntimeCounters> {
        gpu::prefill_runtime_counters(self.gpu_layer_runtime.as_ref())
    }

    pub(in crate::engine) fn clear_sequence_state(&mut self) -> crate::error::Result<()> {
        #[cfg(feature = "cuda")]
        {
            cuda_runtime::clear_sequence_state_cache().map_err(crate::error::LlmError::Forward)?;
        }
        #[cfg(feature = "metal")]
        {
            metal_runtime::metal_clear_sequence_state().map_err(crate::error::LlmError::Forward)?;
            metal_runtime::metal_clear_gemma_prefill_f16kv_residents();
        }
        #[cfg(feature = "vulkan")]
        {
            if let Some(runtime) = self.gpu_layer_runtime.as_mut() {
                runtime
                    .clear_sequence_state()
                    .map_err(crate::error::LlmError::Forward)?;
            }
        }
        Ok(())
    }
}

impl Drop for EngineBackendRuntime {
    fn drop(&mut self) {
        #[cfg(feature = "metal")]
        crate::engine::metal_runtime::metal_clear_gemma_prefill_f16kv_residents();
        #[cfg(feature = "mediatek")]
        {
            crate::engine::mediatek_runtime::clear_gated_gelu_ffn_f32_cache();
            crate::engine::mediatek_runtime::clear_gated_gelu_ffn_quantized_cache();
        }
    }
}

#[cfg(all(feature = "vulkan", test))]
pub(in crate::engine) fn init_layer_gemv_for_test(
    max_input: usize,
    max_output: usize,
    chunk_size: usize,
) -> Result<GpuRuntime, String> {
    gpu::init_layer_gemv_for_test(max_input, max_output, chunk_size)
}

#[cfg(all(feature = "vulkan", test))]
pub(in crate::engine) fn ggml_to_quant_for_test(ggml_type: GGMLType) -> Option<GpuQuant> {
    gpu::ggml_to_quant(ggml_type)
}

pub(in crate::engine) fn init_engine_backend_runtime(
    metadata: &ModelMetadata,
    weights: &ModelWeights,
    ffn_inner_dim: usize,
) -> EngineBackendRuntime {
    #[cfg(feature = "vulkan")]
    {
        return EngineBackendRuntime::from_gpu_runtime(super::init_prefill_layer_runtime(
            metadata,
            weights,
            ffn_inner_dim,
        ));
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (metadata, weights, ffn_inner_dim);
        EngineBackendRuntime::new()
    }
}

/// 레이어가 recurrent (GDN) 인지 attention 인지 판별.
/// Qwen3.5: full_attention_interval > 0이면,
/// layer_idx % interval == (interval - 1)인 레이어가 attention.
pub(in crate::engine) fn is_attention_layer(
    layer_idx: usize,
    full_attention_interval: usize,
) -> bool {
    if full_attention_interval == 0 {
        return true;
    }
    layer_idx % full_attention_interval == (full_attention_interval - 1)
}

#[cfg(test)]
mod tests {
    use super::EngineBackendRuntime;

    #[test]
    fn full_device_decode_requires_every_attention_and_ffn_layer_on_gpu() {
        let mut runtime = EngineBackendRuntime::new();
        assert!(runtime.decode_all_layers_use_gpu(52));

        runtime.set_initial_decode_gpu_layer_prefixes(Some((52, 30)));
        assert!(!runtime.decode_all_layers_use_gpu(52));

        runtime.set_decode_gpu_layer_prefixes(Some((52, 52)));
        assert!(runtime.decode_all_layers_use_gpu(52));
        runtime.restore_initial_decode_gpu_layer_prefixes();
        assert!(!runtime.decode_all_layers_use_gpu(52));
    }
}
