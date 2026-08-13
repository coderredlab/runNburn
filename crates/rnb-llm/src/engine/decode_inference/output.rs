//! Final decode normalization and output-logit projection.

use super::*;

fn argmax_token_excluding(logits: &[f32], excluded: Option<u32>) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .filter(|(token, _)| Some(*token as u32) != excluded)
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(token, _)| token as u32)
}

pub(super) fn finalize_decode_logits(
    weights: &ModelWeights,
    scratch: &mut ScratchBuffers,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden_dim: usize,
    profiling: bool,
    verbose: bool,
    use_backend_output_logits: bool,
    #[cfg(feature = "vulkan")] gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    let t_out = std::time::Instant::now();
    let output_norm_data = kernels::tensor_as_f32_slice(&weights.output_norm);
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let t_norm = std::time::Instant::now();
    if gemma_skip_output_norm() {
        scratch.norm_buf[..hidden_dim].copy_from_slice(&scratch.hidden[..hidden_dim]);
    } else if gemma_effective_unit_offset_output_norm_decode(architecture, gemma_runtime_flavor) {
        apply_model_norm_unit_offset_into(
            &scratch.hidden[..hidden_dim],
            output_norm_data,
            metadata.norm_eps,
            &mut scratch.norm_buf[..hidden_dim],
        );
    } else {
        apply_model_norm_into(
            &scratch.hidden[..hidden_dim],
            output_norm_data,
            metadata.norm_eps,
            &mut scratch.norm_buf[..hidden_dim],
            architecture,
        );
    }
    emit_final_dump("decode_normed", &scratch.norm_buf[..hidden_dim]);
    let norm_ms = t_norm.elapsed().as_micros() as f64 / 1000.0;
    let t_gemv = std::time::Instant::now();
    let backend_output_ok = backend_runtime::try_backend_output_logits_for_runtime(
        weights,
        scratch,
        hidden_dim,
        profiling,
        use_backend_output_logits,
        #[cfg(feature = "vulkan")]
        gpu_runtime,
    )?;
    if !backend_output_ok {
        let f64_logit = crate::engine::policy::env_string("RNB_OUTPUT_F64_LOGIT").is_some();
        if use_token_embedding_as_output() {
            if f64_logit {
                weights
                    .token_embd
                    .gemv_into_f64_logit(&scratch.norm_buf[..hidden_dim], &mut scratch.logits)?;
            } else {
                weights
                    .token_embd
                    .gemv_into(&scratch.norm_buf[..hidden_dim], &mut scratch.logits)?;
            }
        } else if f64_logit {
            weights
                .output
                .gemv_into_f64_logit(&scratch.norm_buf[..hidden_dim], &mut scratch.logits)?;
        } else {
            weights
                .output
                .gemv_into(&scratch.norm_buf[..hidden_dim], &mut scratch.logits)?;
        }
    }
    let needs_host_ranking = !scratch.backend_argmax_only || scratch.backend_argmax_token.is_none();
    if needs_host_ranking {
        super::super::models::muse_glimmer::scale_logits_inplace(
            &mut scratch.logits,
            metadata.logit_scale,
        );
        apply_logit_softcapping(&mut scratch.logits, metadata.final_logit_softcapping);
    }
    if scratch.backend_argmax_only {
        if scratch.backend_argmax_token.is_none() {
            scratch.backend_argmax_token =
                argmax_token_excluding(&scratch.logits, scratch.backend_argmax_excluded_token);
            if scratch.backend_argmax_token.is_none() {
                return Err(crate::error::LlmError::Forward(
                    "decode argmax has no eligible output token".to_string(),
                ));
            }
        }
    } else {
        emit_final_dump("decode_logits", &scratch.logits);
        if crate::engine::policy::env_string("RNB_CUDA_EAGER_LOGITS_RANGE").as_deref() == Some("1")
        {
            let min = scratch.logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = scratch
                .logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let (argmax_idx, argmax_val) = scratch
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            eprintln!("[cu87 eager logits] range=[{min:.4}, {max:.4}] argmax_idx={argmax_idx} argmax_val={argmax_val:.4}");
        }
    }
    let gemv_ms = t_gemv.elapsed().as_micros() as f64 / 1000.0;
    if verbose {
        eprintln!(
            "  [DEC] output: norm={:.2}ms, gemv={:.1}ms (vocab={})",
            norm_ms, gemv_ms, metadata.vocab_size
        );
    } else if profiling {
        eprintln!(
            "  [DEC] output_logits    {:.1}ms (vocab={})",
            t_out.elapsed().as_micros() as f64 / 1000.0,
            metadata.vocab_size
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::argmax_token_excluding;

    #[test]
    fn host_argmax_skips_the_excluded_token() {
        let logits = [1.0, 4.0, 3.0];
        assert_eq!(argmax_token_excluding(&logits, None), Some(1));
        assert_eq!(argmax_token_excluding(&logits, Some(1)), Some(2));
        assert_eq!(argmax_token_excluding(&logits[..1], Some(0)), None);
    }
}
