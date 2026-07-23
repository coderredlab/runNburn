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
}

impl EngineBackendRuntime {
    pub(in crate::engine) fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn from_gpu_runtime(gpu_layer_runtime: Option<GpuRuntime>) -> Self {
        Self { gpu_layer_runtime }
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn take_gpu_runtime(&mut self) -> Option<GpuRuntime> {
        self.gpu_layer_runtime.take()
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn restore_gpu_runtime(&mut self, gpu_layer_runtime: Option<GpuRuntime>) {
        self.gpu_layer_runtime = gpu_layer_runtime;
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
