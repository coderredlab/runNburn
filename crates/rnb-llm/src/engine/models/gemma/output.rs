use super::*;

pub(in crate::engine) fn apply_embedding_scale(
    hidden: Tensor,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
) -> Tensor {
    if env_flag("RNB_DISABLE_GEMMA_EMB_SCALE") {
        return hidden;
    }
    if !use_gemma_block_semantics(architecture) {
        return hidden;
    }

    let scale = (metadata.hidden_dim as f32).sqrt();
    let mut data = kernels::tensor_as_f32_slice(&hidden).to_vec();
    scale_f32_inplace(&mut data, scale);
    Tensor::from_vec(data, hidden.shape())
}

pub(in crate::engine) fn apply_embedding_scale_inplace(
    hidden: &mut [f32],
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
) {
    if env_flag("RNB_DISABLE_GEMMA_EMB_SCALE") {
        return;
    }
    if !use_gemma_block_semantics(architecture) {
        return;
    }

    let scale = if env_flag("RNB_GEMMA_BF16_EMBED_SCALE") {
        let raw = (metadata.hidden_dim as f32).sqrt();
        let bits = raw.to_bits();
        let bf16_bits = bits & 0xFFFF0000;
        f32::from_bits(bf16_bits)
    } else {
        (metadata.hidden_dim as f32).sqrt()
    };
    scale_f32_inplace(hidden, scale);
}

pub(in crate::engine) fn apply_layer_output_scale(
    hidden: Tensor,
    out_scale: Option<&Tensor>,
    layer_idx: usize,
) -> Tensor {
    let Some(scale_data) = active_layer_output_scale(out_scale, layer_idx) else {
        return hidden;
    };
    let mut data = kernels::tensor_as_f32_slice(&hidden).to_vec();
    scale_f32_inplace(&mut data, scale_data[0]);
    Tensor::from_vec(data, hidden.shape())
}

pub(in crate::engine) fn apply_layer_output_scale_inplace(
    hidden: &mut [f32],
    out_scale: Option<&Tensor>,
    layer_idx: usize,
) {
    if env_flag("RNB_DISABLE_GEMMA_OUT_SCALE") {
        return;
    }
    if !gemma_enable_out_scale_opt_in() {
        return;
    }
    if gemma_skip_out_scale_enabled(layer_idx) {
        return;
    }
    let Some(scale) = out_scale else {
        return;
    };
    let scale_data = kernels::tensor_as_f32_slice(scale);
    if scale_data.is_empty() {
        return;
    }
    scale_f32_inplace(hidden, scale_data[0]);
}

pub(in crate::engine) fn active_layer_output_scale(
    out_scale: Option<&Tensor>,
    layer_idx: usize,
) -> Option<&[f32]> {
    if env_flag("RNB_DISABLE_GEMMA_OUT_SCALE") {
        return None;
    }
    if !gemma_enable_out_scale_opt_in() {
        return None;
    }
    if gemma_skip_out_scale_enabled(layer_idx) {
        return None;
    }
    let scale = out_scale?;
    let scale_data = kernels::tensor_as_f32_slice(scale);
    if scale_data.is_empty() {
        return None;
    }
    Some(&scale_data[..1])
}

pub(in crate::engine) fn apply_logit_softcapping(logits: &mut [f32], softcap: f32) {
    if env_flag("RNB_DISABLE_LOGIT_SOFTCAP") {
        return;
    }
    if softcap <= 0.0 {
        return;
    }
    for logit in logits.iter_mut() {
        *logit = softcap * (*logit / softcap).tanh();
    }
}

pub(in crate::engine) fn emit_decode_layer_target_trace(
    tokenizer: &Tokenizer,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: &[f32],
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<()> {
    let Some(targets) = decode_trace_targets() else {
        return Ok(());
    };
    if !decode_trace_layer_enabled(layer_idx) {
        return Ok(());
    }

    let hidden_tensor = Tensor::from_slice(hidden, &[1, hidden.len()]);
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);
    let normed = if gemma_skip_output_norm() {
        hidden_tensor
    } else if gemma_effective_unit_offset_output_norm_decode(architecture, gemma_runtime_flavor) {
        apply_model_norm_unit_offset(&hidden_tensor, &weights.output_norm, norm_eps)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    } else {
        apply_model_norm(&hidden_tensor, &weights.output_norm, norm_eps, architecture)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    };
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    let exact_output = super::policy::exact_output_gemv_enabled();
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

    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((id, val)) = ranked.first() {
        let piece = tokenizer.decode_token(*id as u32).replace('\n', "\\n");
        eprintln!(
            "[decode-trace] layer={} top1 id={} logit={:.4} piece={:?}",
            layer_idx, id, val, piece
        );
    }
    for target in targets {
        let mut found = None;
        for (rank, (id, val)) in ranked.iter().enumerate() {
            let piece = tokenizer.decode_token(*id as u32);
            if piece == target || piece.starts_with(&target) {
                found = Some((rank + 1, *id, *val, piece));
                break;
            }
        }
        match found {
            Some((rank, id, val, piece)) => eprintln!(
                "[decode-trace] layer={} target={:?} rank=#{} id={} logit={:.4} piece={:?}",
                layer_idx, target, rank, id, val, piece
            ),
            None => eprintln!(
                "[decode-trace] layer={} target={:?} rank=none",
                layer_idx, target
            ),
        }
    }
    Ok(())
}
