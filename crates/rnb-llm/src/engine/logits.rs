#[cfg(feature = "cuda")]
use super::backend_runtime::{
    prefill_output_argmax_token_cuda, prefill_output_logits_cuda,
    prewarm_backend_output_weight_for_runtime,
};
use super::cpu_runtime::kernels;
use super::layer_weights::ModelWeights;
use super::models::gemma::GemmaRuntimeFlavor;
use super::models::gemma::{
    apply_gemma_per_layer_branch, apply_logit_softcapping, detect_gemma_runtime_flavor,
    prepare_gemma_per_layer_base, use_gemma_block_semantics,
};
use super::models::gemma::{gemma_disable_e2bit_local_defaults, gemma_ple_after_final_norm};
use super::norm::apply_model_norm;
use super::policy;
#[cfg(feature = "cuda")]
use super::prefill::hidden_carrier;
use super::trace::emit_final_dump;
use super::types::ModelMetadata;
use crate::kv_cache::KVCache;
use crate::sampler::greedy::greedy_sample;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;

pub(super) fn use_token_embedding_as_output() -> bool {
    policy::use_token_embedding_as_output()
}

pub(super) fn gemma_skip_output_norm() -> bool {
    policy::gemma_skip_output_norm_enabled()
}

pub(super) fn gemma_default_unit_offset_output_norm(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
) -> bool {
    if gemma_disable_e2bit_local_defaults() {
        return false;
    }
    matches!(architecture, ModelArchitecture::Gemma4)
        && gemma_runtime_flavor == GemmaRuntimeFlavor::Gemma4E2BIt
}

pub(super) fn gemma_effective_unit_offset_output_norm_prefill(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
) -> bool {
    if policy::gemma_output_norm_prefill_unit_offset_disabled() {
        return false;
    }
    use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_output_norm_enabled()
            || gemma_default_unit_offset_output_norm(architecture, gemma_runtime_flavor))
}

pub(super) fn gemma_effective_unit_offset_output_norm_decode(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
) -> bool {
    if policy::gemma_output_norm_decode_unit_offset_disabled() {
        return false;
    }
    use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_output_norm_enabled()
            || gemma_default_unit_offset_output_norm(architecture, gemma_runtime_flavor))
}

pub(super) fn finalize_prefill_logits(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: Tensor,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    last_layer_hidden_out: Option<&mut Vec<f32>>,
) -> crate::error::Result<Vec<f32>> {
    let hidden = if gemma_ple_after_final_norm() {
        if let Some(base) = prepare_gemma_per_layer_base(
            weights,
            &hidden,
            &[0u32; 0],
            metadata,
            architecture,
            norm_eps,
        )? {
            if let Some(gemma) = weights.gemma_per_layer.as_ref() {
                let layer_idx = metadata.num_layers.saturating_sub(1);
                apply_gemma_per_layer_branch(
                    hidden,
                    &base,
                    layer_idx,
                    gemma,
                    metadata,
                    architecture,
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
    let profiling = policy::profiling_enabled();
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let t_out = std::time::Instant::now();
    let last_hidden = if seq_len > 1 {
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        let hidden_dim = metadata.hidden_dim;
        let start = (seq_len - 1) * hidden_dim;
        Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim])
    } else {
        hidden
    };
    let normed = if gemma_skip_output_norm() {
        last_hidden
    } else if gemma_effective_unit_offset_output_norm_prefill(architecture, gemma_runtime_flavor) {
        kernels::norm::rms_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    } else {
        apply_model_norm(&last_hidden, &weights.output_norm, norm_eps, architecture)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    };
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    emit_final_dump("prefill_normed", normed_data);
    if let Some(out) = last_layer_hidden_out {
        out.clear();
        out.extend_from_slice(normed_data);
    }
    #[cfg(feature = "cuda")]
    let logits = if !use_token_embedding_as_output() {
        prefill_output_logits_cuda(weights, normed_data)
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let logits = None;
    let mut logits = match logits {
        Some(logits) => logits,
        None if use_token_embedding_as_output() => weights.token_embd.gemv_vec(normed_data)?,
        None => {
            let logits = weights.output.gemv_vec(normed_data)?;
            #[cfg(feature = "cuda")]
            prewarm_backend_output_weight_for_runtime(weights);
            logits
        }
    };
    apply_logit_softcapping(&mut logits, metadata.final_logit_softcapping);
    emit_final_dump("prefill_logits", &logits);
    if profiling {
        eprintln!(
            "  [FWD] output_logits    {:.1}ms (vocab={})",
            t_out.elapsed().as_micros() as f64 / 1000.0,
            metadata.vocab_size
        );
    }
    kv_cache.set_len(pos_start + seq_len);
    Ok(logits)
}

#[cfg(feature = "cuda")]
pub(super) fn finalize_prefill_argmax_token_cuda_only(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: Tensor,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<u32>> {
    if use_token_embedding_as_output() {
        return Ok(None);
    }
    let hidden = if gemma_ple_after_final_norm() {
        if let Some(base) = prepare_gemma_per_layer_base(
            weights,
            &hidden,
            &[0u32; 0],
            metadata,
            architecture,
            norm_eps,
        )? {
            if let Some(gemma) = weights.gemma_per_layer.as_ref() {
                let layer_idx = metadata.num_layers.saturating_sub(1);
                apply_gemma_per_layer_branch(
                    hidden,
                    &base,
                    layer_idx,
                    gemma,
                    metadata,
                    architecture,
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
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let last_hidden = if seq_len > 1 {
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        let hidden_dim = metadata.hidden_dim;
        let start = (seq_len - 1) * hidden_dim;
        Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim])
    } else {
        hidden
    };
    let normed = if gemma_skip_output_norm() {
        last_hidden
    } else if gemma_effective_unit_offset_output_norm_prefill(architecture, gemma_runtime_flavor) {
        kernels::norm::rms_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    } else {
        apply_model_norm(&last_hidden, &weights.output_norm, norm_eps, architecture)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    };
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    let Some(token) = prefill_output_argmax_token_cuda(weights, normed_data) else {
        return Ok(None);
    };
    kv_cache.set_len(pos_start + seq_len);
    Ok(Some(token))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_prefill_argmax_token_cuda_only_carrier(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: hidden_carrier::PrefillHidden,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<u32>> {
    if use_token_embedding_as_output() {
        return Ok(None);
    }
    match hidden {
        hidden_carrier::PrefillHidden::Host(hidden) => finalize_prefill_argmax_token_cuda_only(
            kv_cache,
            metadata,
            architecture,
            weights,
            hidden,
            seq_len,
            pos_start,
            norm_eps,
        ),
        hidden_carrier::PrefillHidden::Device(device) => {
            if gemma_ple_after_final_norm() {
                let hidden = hidden_carrier::materialize_device_hidden(
                    device,
                    None,
                    "final_ple_after_norm",
                )?;
                return finalize_prefill_argmax_token_cuda_only(
                    kv_cache,
                    metadata,
                    architecture,
                    weights,
                    hidden,
                    seq_len,
                    pos_start,
                    norm_eps,
                );
            }
            let last_row = seq_len.checked_sub(1).ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "CUDA output argmax carrier requires non-empty prefill".to_string(),
                )
            })?;
            let last_hidden = hidden_carrier::materialize_device_hidden_row(
                device,
                last_row,
                None,
                "output_last_row",
            )?;
            let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
            let normed = if gemma_skip_output_norm() {
                last_hidden
            } else if gemma_effective_unit_offset_output_norm_prefill(
                architecture,
                gemma_runtime_flavor,
            ) {
                kernels::norm::rms_norm_unit_offset(&last_hidden, &weights.output_norm, norm_eps)
                    .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
            } else {
                apply_model_norm(&last_hidden, &weights.output_norm, norm_eps, architecture)
                    .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
            };
            let normed_data = kernels::tensor_as_f32_slice(&normed);
            let Some(token) = prefill_output_argmax_token_cuda(weights, normed_data) else {
                return Ok(None);
            };
            kv_cache.set_len(pos_start + seq_len);
            Ok(Some(token))
        }
    }
}

/// Verify용: 모든 토큰 위치의 logit을 반환.
pub(super) fn finalize_prefill_all_logits(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: Tensor,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Vec<Vec<f32>>> {
    let hidden_data = kernels::tensor_as_f32_slice(&hidden);
    let hidden_dim = metadata.hidden_dim;
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let mut all_logits = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let start = t * hidden_dim;
        let token_hidden =
            Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim]);
        let normed = if gemma_effective_unit_offset_output_norm_prefill(
            architecture,
            gemma_runtime_flavor,
        ) {
            kernels::norm::rms_norm_unit_offset(&token_hidden, &weights.output_norm, norm_eps)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else {
            apply_model_norm(&token_hidden, &weights.output_norm, norm_eps, architecture)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        };
        let normed_data = kernels::tensor_as_f32_slice(&normed);
        let exact_output = policy::exact_output_gemv_enabled();
        let mut logits = if use_token_embedding_as_output() {
            if exact_output {
                weights.token_embd.gemv_vec_exact_f32(normed_data)?
            } else {
                weights.token_embd.gemv_vec(normed_data)?
            }
        } else if exact_output {
            weights.output.gemv_vec_exact_f32(normed_data)?
        } else {
            weights.output.gemv_vec(normed_data)?
        };
        apply_logit_softcapping(&mut logits, metadata.final_logit_softcapping);
        all_logits.push(logits);
    }

    kv_cache.set_len(pos_start + seq_len);
    Ok(all_logits)
}

/// Greedy verify용: 모든 토큰 위치의 target argmax만 반환한다.
pub(super) fn finalize_prefill_argmax_tokens(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: Tensor,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> crate::error::Result<Vec<u32>> {
    let hidden_data = kernels::tensor_as_f32_slice(&hidden);
    let hidden_dim = metadata.hidden_dim;
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let mut target_tokens = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let start = t * hidden_dim;
        let token_hidden =
            Tensor::from_slice(&hidden_data[start..start + hidden_dim], &[1, hidden_dim]);
        let normed = if gemma_effective_unit_offset_output_norm_prefill(
            architecture,
            gemma_runtime_flavor,
        ) {
            kernels::norm::rms_norm_unit_offset(&token_hidden, &weights.output_norm, norm_eps)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else {
            apply_model_norm(&token_hidden, &weights.output_norm, norm_eps, architecture)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        };
        let normed_data = kernels::tensor_as_f32_slice(&normed);
        let exact_output = policy::exact_output_gemv_enabled();
        #[cfg(feature = "cuda")]
        if !use_token_embedding_as_output() && !exact_output {
            if let Some(token) = prefill_output_argmax_token_cuda(weights, normed_data) {
                target_tokens.push(token);
                continue;
            }
        }
        let mut logits = if use_token_embedding_as_output() {
            if exact_output {
                weights.token_embd.gemv_vec_exact_f32(normed_data)?
            } else {
                weights.token_embd.gemv_vec(normed_data)?
            }
        } else if exact_output {
            weights.output.gemv_vec_exact_f32(normed_data)?
        } else {
            weights.output.gemv_vec(normed_data)?
        };
        apply_logit_softcapping(&mut logits, metadata.final_logit_softcapping);
        target_tokens.push(greedy_sample(&logits));
    }

    kv_cache.set_len(pos_start + seq_len);
    Ok(target_tokens)
}
