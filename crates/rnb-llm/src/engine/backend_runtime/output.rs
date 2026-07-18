#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
use crate::engine::layer_weights::ModelWeights;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::engine::logits::use_token_embedding_as_output;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
use crate::engine::types::ScratchBuffers;
#[cfg(all(test, feature = "vulkan"))]
use rnb_loader::GGMLType;

#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;

pub(in crate::engine) fn output_logits_enabled_for_runtime() -> bool {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::output_logits_enabled();
    }
    #[cfg(feature = "vulkan")]
    {
        return gpu::gpu_output_logits_enabled();
    }
    #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
    false
}

#[allow(unused_variables)]
pub(in crate::engine) fn try_backend_output_logits_for_runtime(
    weights: &ModelWeights,
    scratch: &mut ScratchBuffers,
    hidden_dim: usize,
    profiling: bool,
    use_gpu_output_logits: bool,
    #[cfg(feature = "vulkan")] mut gpu_runtime: Option<&mut gpu::Runtime>,
) -> bool {
    scratch.backend_argmax_token = None;

    #[cfg(feature = "vulkan")]
    if use_gpu_output_logits {
        let raw = weights.output.data.as_bytes().unwrap_or(&[]);
        if gpu::try_output_logits(
            gpu_runtime.as_deref_mut(),
            weights.output.ggml_type,
            weights.output.rows,
            weights.output.cols,
            raw,
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.logits,
            profiling,
        ) {
            return true;
        }
    }

    #[cfg(feature = "metal")]
    if scratch.backend_argmax_only && !use_token_embedding_as_output() {
        if let Some(raw) = weights.output.data.as_bytes() {
            if let Some(token) = metal_runtime::metal_output_argmax_token_if_supported(
                weights.output.ggml_type,
                raw,
                weights.output.rows,
                weights.output.cols,
                &scratch.norm_buf[..hidden_dim],
            ) {
                scratch.backend_argmax_token = Some(token);
                return true;
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(raw) = weights.output.data.as_bytes() {
            if let Some(token) = cuda_runtime::try_output_logits_into_if_enabled(
                weights.output.ggml_type,
                weights.output.rows,
                weights.output.cols,
                raw,
                &scratch.norm_buf[..hidden_dim],
                use_token_embedding_as_output(),
                &mut scratch.logits,
                !scratch.backend_argmax_only,
            ) {
                scratch.backend_argmax_token = Some(token);
                return true;
            }
            if let Some(logits) = cuda_runtime::try_output_logits_if_enabled(
                weights.output.ggml_type,
                weights.output.rows,
                weights.output.cols,
                raw,
                &scratch.norm_buf[..hidden_dim],
                use_token_embedding_as_output(),
            ) {
                scratch.logits[..weights.output.rows].copy_from_slice(&logits);
                return true;
            }
        }
    }

    false
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_backend_output_weight_for_runtime(weights: &ModelWeights) {
    #[cfg(feature = "cuda")]
    {
        if std::env::var("RNB_CUDA_OUTPUT_PREWARM").ok().as_deref() != Some("1") {
            return;
        }
        let Some(raw) = weights.output.data.as_bytes() else {
            return;
        };
        let _ = cuda_runtime::prewarm_output_weight(
            weights.output.ggml_type,
            weights.output.rows,
            weights.output.cols,
            raw,
        );
    }
}

pub(in crate::engine) fn force_prewarm_backend_output_weight_for_runtime(
    weights: &ModelWeights,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        let Some(raw) = weights.output.data.as_bytes() else {
            return false;
        };
        return cuda_runtime::prewarm_output_weight(
            weights.output.ggml_type,
            weights.output.rows,
            weights.output.cols,
            raw,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = weights;
        false
    }
}

pub(in crate::engine) fn backend_output_argmax_supported_for_runtime(
    weights: &ModelWeights,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        if use_token_embedding_as_output() || weights.output.data.as_bytes().is_none() {
            return false;
        }
        return matches!(
            weights.output.ggml_type,
            rnb_loader::GGMLType::Q6_K | rnb_loader::GGMLType::Q8_0
        );
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        if use_token_embedding_as_output() || weights.output.data.as_bytes().is_none() {
            return false;
        }
        return matches!(
            weights.output.ggml_type,
            rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K
        );
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        let _ = weights;
        false
    }
}

#[cfg(all(test, feature = "vulkan"))]
pub(in crate::engine) fn ggml_to_gpu_output_quant(ggml_type: GGMLType) -> Option<gpu::Quant> {
    gpu::output_logits_quant_for_test(ggml_type)
}
