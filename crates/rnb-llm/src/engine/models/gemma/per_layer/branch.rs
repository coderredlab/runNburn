use super::*;

pub(in crate::engine) fn apply_gemma_per_layer_branch(
    hidden: Tensor,
    base: &GemmaPerLayerBase,
    layer_idx: usize,
    weights: &GemmaPerLayerWeights,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    if matches!(architecture, ModelArchitecture::Gemma4) {
        return super::gemma4::apply_gemma4_per_layer_branch(
            hidden, base, layer_idx, weights, metadata, norm_eps,
        );
    }

    let hidden_dim = metadata.hidden_dim;
    let d = metadata.embedding_length_per_layer_input;
    let total_d = metadata.num_layers * d;
    let hidden_shape = hidden.shape().to_vec();
    let hidden_data = kernels::tensor_as_f32_slice(&hidden);
    let seq_len = hidden_data.len() / hidden_dim;
    let gemma_runtime_flavor = if metadata.num_layers == 35
        && metadata.hidden_dim == 1536
        && metadata.num_heads == 8
        && metadata.num_kv_heads == 1
        && metadata.head_dim == 512
        && metadata.embedding_length_per_layer_input == 256
    {
        GemmaRuntimeFlavor::Gemma4E2BIt
    } else {
        GemmaRuntimeFlavor::Generic
    };
    let ple_layer_idx = gemma_ple_effective_layer_idx(layer_idx, metadata.num_layers);
    let layer34_hard_fix = gemma_runtime_flavor == GemmaRuntimeFlavor::Gemma4E2BIt
        && gemma_ple_layer34_hard_fix_applies(architecture, layer_idx, metadata.num_layers);
    let layer = &weights.layers[ple_layer_idx];

    if layer.inp_gate.rows != d || layer.inp_gate.cols != hidden_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma inp_gate shape mismatch at layer {}: [{}x{}], expected [{}x{}]",
            layer_idx, layer.inp_gate.rows, layer.inp_gate.cols, d, hidden_dim
        )));
    }
    if layer.proj.rows != hidden_dim || layer.proj.cols != d {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma proj shape mismatch at layer {}: [{}x{}], expected [{}x{}]",
            layer_idx, layer.proj.rows, layer.proj.cols, hidden_dim, d
        )));
    }
    if layer.post_norm.numel() != hidden_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma post_norm mismatch at layer {}: got {}, expected {}",
            layer_idx,
            layer.post_norm.numel(),
            hidden_dim
        )));
    }

    if layer34_hard_fix {
        let source = if gemma_ple_token_only() {
            &base.token
        } else {
            &base.model
        };
        let mut out = hidden_data.to_vec();
        let rows: Box<dyn Iterator<Item = usize>> = if seq_len > 1 {
            Box::new(std::iter::once(seq_len - 1))
        } else {
            Box::new(std::iter::once(0))
        };
        for t in rows {
            let base_off = t * total_d + ple_layer_idx * d;
            let hidden_off = t * hidden_dim;
            let per_layer_row = &source[base_off..base_off + d];
            let projected = layer.proj.gemv_vec(per_layer_row)?;
            out[hidden_off..hidden_off + hidden_dim].copy_from_slice(&projected[..hidden_dim]);
        }
        return Ok(Tensor::from_vec(out, &hidden_shape));
    }

    let mut per_layer_input = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let source = if layer34_hard_fix || gemma_ple_model_only() {
            &base.model
        } else if gemma_ple_token_only() {
            &base.token
        } else {
            &base.mixed
        };
        let base_off = t * total_d + ple_layer_idx * d;
        let dst_off = t * d;
        per_layer_input[dst_off..dst_off + d].copy_from_slice(&source[base_off..base_off + d]);
    }
    let mut gate = layer.inp_gate.gemv_vec(hidden_data)?;
    if !env_flag("RNB_GEMMA_PLE_SKIP_GATE_MUL") {
        if env_flag("RNB_GEMMA_PLE_RAW_GATE_MUL") {
            kernels::elementwise::mul_inplace(&mut gate, &per_layer_input);
        } else {
            apply_model_gate_mul_inplace(&mut gate, &per_layer_input, architecture);
        }
    }
    emit_ple_trace(
        "branch",
        layer_idx,
        "per_layer_input",
        &per_layer_input[..per_layer_input.len().min(d)],
    );
    emit_ple_trace("branch", layer_idx, "gate", &gate[..gate.len().min(d)]);
    emit_ple_dump(layer_idx, "per_layer_input", &per_layer_input);
    emit_ple_dump(layer_idx, "gate", &gate);

    let projected = layer.proj.gemv_vec(&gate)?;
    let projected_tensor = Tensor::from_vec(projected, &[seq_len, hidden_dim]);
    emit_ple_trace(
        "branch",
        layer_idx,
        "projected",
        &kernels::tensor_as_f32_slice(&projected_tensor)[..hidden_dim.min(4)],
    );
    emit_ple_dump(
        layer_idx,
        "projected",
        kernels::tensor_as_f32_slice(&projected_tensor),
    );
    if env_flag("RNB_GEMMA_PLE_RESIDUAL_PRE_NORM") {
        let mut combined = hidden_data.to_vec();
        kernels::elementwise::add_inplace(
            &mut combined,
            kernels::tensor_as_f32_slice(&projected_tensor),
        );
        emit_ple_trace(
            "branch",
            layer_idx,
            "combined_pre_norm",
            &combined[..hidden_dim.min(4)],
        );
        let combined_tensor = Tensor::from_vec(combined, &[seq_len, hidden_dim]);
        let normed = if env_flag("RNB_GEMMA_PLE_POST_NORM_MUL_ONLY") {
            let mut combined = kernels::tensor_as_f32_slice(&combined_tensor).to_vec();
            kernels::elementwise::mul_inplace(
                &mut combined,
                kernels::tensor_as_f32_slice(&layer.post_norm),
            );
            Tensor::from_vec(combined, &[seq_len, hidden_dim])
        } else if env_flag("RNB_GEMMA_PLE_UNIT_OFFSET_POST_NORM") {
            kernels::norm::rms_norm_unit_offset(&combined_tensor, &layer.post_norm, norm_eps)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else if env_flag("RNB_GEMMA_PLE_POST_NORM_HIDDEN_ONLY") {
            apply_model_norm(&hidden, &layer.post_norm, norm_eps, architecture)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        } else if gemma_ple_skip_post_norm() || layer34_hard_fix {
            combined_tensor
        } else {
            apply_model_norm(&combined_tensor, &layer.post_norm, norm_eps, architecture)
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
        };
        return Ok(normed);
    }
    let normed = if env_flag("RNB_GEMMA_PLE_POST_NORM_MUL_ONLY") {
        let mut projected = kernels::tensor_as_f32_slice(&projected_tensor).to_vec();
        kernels::elementwise::mul_inplace(
            &mut projected,
            kernels::tensor_as_f32_slice(&layer.post_norm),
        );
        Tensor::from_vec(projected, &[seq_len, hidden_dim])
    } else if env_flag("RNB_GEMMA_PLE_POST_NORM_HIDDEN_ONLY") {
        apply_model_norm(&hidden, &layer.post_norm, norm_eps, architecture)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    } else if env_flag("RNB_GEMMA_PLE_UNIT_OFFSET_POST_NORM") {
        kernels::norm::rms_norm_unit_offset(&projected_tensor, &layer.post_norm, norm_eps)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    } else if gemma_ple_skip_post_norm() || layer34_hard_fix {
        projected_tensor
    } else {
        apply_model_norm(&projected_tensor, &layer.post_norm, norm_eps, architecture)
            .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?
    };
    let mut branch_data = kernels::tensor_as_f32_slice(&normed).to_vec();
    if let Some(scale) = gemma_ple_branch_scale() {
        for v in &mut branch_data {
            *v *= scale;
        }
    }
    emit_ple_trace(
        "branch",
        layer_idx,
        "post_norm",
        &branch_data[..hidden_dim.min(4)],
    );
    emit_ple_dump(layer_idx, "post_norm", &branch_data);
    let mut out = hidden_data.to_vec();
    if let Some(alpha) = gemma_ple_hidden_blend_alpha() {
        let keep = alpha.clamp(0.0, 1.0);
        let inject = 1.0 - keep;
        for (dst, src) in out.iter_mut().zip(branch_data.iter()) {
            *dst = *dst * keep + *src * inject;
        }
    } else if layer34_hard_fix {
        if seq_len == 0 {
            out.clear();
        } else {
            let row_off = (seq_len - 1) * hidden_dim;
            out[row_off..row_off + hidden_dim]
                .copy_from_slice(&branch_data[row_off..row_off + hidden_dim]);
        }
    } else if gemma_ple_replace_hidden() {
        out.copy_from_slice(&branch_data);
    } else {
        kernels::elementwise::add_inplace(&mut out, &branch_data);
    }
    emit_ple_trace("branch", layer_idx, "hidden_out", &out[..hidden_dim.min(4)]);
    emit_ple_dump(layer_idx, "hidden_out", &out);
    Ok(Tensor::from_vec(out, &hidden_shape))
}

pub(in crate::engine) struct GemmaPerLayerBranchOutput {
    pub(in crate::engine) hidden: Tensor,
    pub(in crate::engine) output_scale_applied: bool,
}

pub(in crate::engine) fn apply_gemma_per_layer_branch_with_output_scale(
    hidden: Tensor,
    base: &GemmaPerLayerBase,
    layer_idx: usize,
    weights: &GemmaPerLayerWeights,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    norm_eps: f32,
    out_scale: Option<&Tensor>,
) -> crate::error::Result<GemmaPerLayerBranchOutput> {
    if matches!(architecture, ModelArchitecture::Gemma4) {
        return super::gemma4::apply_gemma4_per_layer_branch_with_output_scale(
            hidden, base, layer_idx, weights, metadata, norm_eps, out_scale,
        )
        .map(|output| GemmaPerLayerBranchOutput {
            hidden: output.hidden,
            output_scale_applied: output.output_scale_applied,
        });
    }
    apply_gemma_per_layer_branch(
        hidden,
        base,
        layer_idx,
        weights,
        metadata,
        architecture,
        norm_eps,
    )
    .map(|hidden| GemmaPerLayerBranchOutput {
        hidden,
        output_scale_applied: false,
    })
}

pub(in crate::engine) fn gemma_ple_global_only(
    metadata: &ModelMetadata,
    layer: &AttentionLayerWeights,
) -> bool {
    if env_flag("RNB_GEMMA_PLE_GLOBAL_ONLY") {
        return layer.q_weight.rows / metadata.num_heads
            > metadata.embedding_length_per_layer_input;
    }
    false
}
