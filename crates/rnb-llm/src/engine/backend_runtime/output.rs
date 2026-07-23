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
) -> crate::error::Result<bool> {
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
            return Ok(true);
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
                return Ok(true);
            }
        }
    }

    #[cfg(feature = "cuda")]
    if use_gpu_output_logits && !use_token_embedding_as_output() {
        if let Some(raw) = weights.output.data.as_bytes() {
            if let Some(token) = cuda_runtime::try_output_logits_into_if_enabled(
                weights.output.ggml_type,
                weights.output.rows,
                weights.output.cols,
                raw,
                &scratch.norm_buf[..hidden_dim],
                false,
                &mut scratch.logits,
                !scratch.backend_argmax_only,
            ) {
                scratch.backend_argmax_token = Some(token);
                return Ok(true);
            }
            if let Some(result) = cuda_runtime::decode_gemv(
                weights.output.ggml_type,
                raw,
                weights.output.rows,
                weights.output.cols,
                &scratch.norm_buf[..hidden_dim],
            ) {
                let logits = result.map_err(crate::error::LlmError::Forward)?;
                if logits.len() != weights.output.rows {
                    return Err(crate::error::LlmError::Forward(format!(
                        "CUDA output GEMV returned {} logits, expected {}",
                        logits.len(),
                        weights.output.rows
                    )));
                }
                scratch.logits[..weights.output.rows].copy_from_slice(&logits);
                if scratch.backend_argmax_only {
                    let mut best_token = 0usize;
                    let mut best_value = f32::NEG_INFINITY;
                    for (token, &value) in logits.iter().enumerate() {
                        if value > best_value {
                            best_token = token;
                            best_value = value;
                        }
                    }
                    scratch.backend_argmax_token = Some(best_token as u32);
                }
                return Ok(true);
            }
        }
    }

    Ok(false)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn prewarm_backend_output_weight_for_runtime(weights: &ModelWeights) {
    #[cfg(feature = "cuda")]
    {
        if crate::engine::policy::env_string("RNB_CUDA_OUTPUT_PREWARM").as_deref() != Some("1") {
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
        return weights.output.ggml_type != rnb_loader::GGMLType::I32;
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
