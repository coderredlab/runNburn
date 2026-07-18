//! Final decode normalization and output-logit projection.

use super::*;

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
        kernels::norm::rms_norm_unit_offset_into(
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
    );
    if !backend_output_ok {
        let f64_logit = std::env::var("RNB_OUTPUT_F64_LOGIT").is_ok();
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
    if !scratch.backend_argmax_only {
        apply_logit_softcapping(&mut scratch.logits, metadata.final_logit_softcapping);
        emit_final_dump("decode_logits", &scratch.logits);
        if std::env::var("RNB_CUDA_EAGER_LOGITS_RANGE").ok().as_deref() == Some("1") {
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
