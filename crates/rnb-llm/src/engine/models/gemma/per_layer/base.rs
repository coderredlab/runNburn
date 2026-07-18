use super::*;

pub(in crate::engine) fn prepare_gemma_per_layer_base(
    weights: &ModelWeights,
    hidden: &Tensor,
    tokens: &[u32],
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    norm_eps: f32,
) -> crate::error::Result<Option<GemmaPerLayerBase>> {
    if !gemma_per_layer_enabled_for_model(weights, metadata, architecture) {
        return Ok(None);
    };
    let Some(gemma) = weights.gemma_per_layer.as_ref() else {
        return Ok(None);
    };
    let gemma_runtime_flavor = detect_gemma_runtime_flavor(metadata, weights);

    let d = metadata.embedding_length_per_layer_input;
    let total_d = metadata.num_layers * d;
    if gemma.layers.len() != metadata.num_layers {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma per-layer mismatch: layer weights={}, metadata layers={}",
            gemma.layers.len(),
            metadata.num_layers
        )));
    }
    if gemma.token_embd.cols != total_d || gemma.model_proj.rows != total_d {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma per-layer shape mismatch: token cols={}, model_proj rows={}, expected total_d={}",
            gemma.token_embd.cols, gemma.model_proj.rows, total_d
        )));
    }
    if gemma.proj_norm.numel() != d {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma per-layer proj_norm mismatch: got {}, expected {}",
            gemma.proj_norm.numel(),
            d
        )));
    }

    let token_branch = gemma.token_embd.gather(tokens)?;
    let token_scale = gemma_ple_token_scale().unwrap_or_else(|| (d as f32).sqrt());
    let mix_scale = 2.0f32.sqrt().recip();
    let token_data = kernels::tensor_as_f32_slice(&token_branch)
        .iter()
        .map(|v| v * token_scale)
        .collect::<Vec<_>>();
    let projected = gemma.model_proj.gemv(hidden)?;
    let projected_scale =
        gemma_ple_projected_scale().unwrap_or_else(|| (metadata.hidden_dim as f32).sqrt().recip());
    let projected_data = kernels::tensor_as_f32_slice(&projected)
        .iter()
        .map(|v| v * projected_scale)
        .collect::<Vec<_>>();
    let proj_norm = kernels::tensor_as_f32_slice(&gemma.proj_norm);
    let seq_len = tokens.len();
    let mut normalized = if env_flag("RNB_GEMMA_PLE_SKIP_PROJ_NORM") {
        projected_data.clone()
    } else {
        vec![0.0f32; projected_data.len()]
    };

    if !env_flag("RNB_GEMMA_PLE_SKIP_PROJ_NORM") {
        // Gemma4 contract §4a: plain RMSNorm(per_layer_proj_norm) applied to every (token, layer)
        // slice. No unit-offset, no layer-34 hard-fix (that was a Session 46 experiment that the
        // runtime contract explicitly rejects — see §8 of the contract doc).
        let is_gemma4 = matches!(architecture, ModelArchitecture::Gemma4);
        for t in 0..seq_len {
            let row_off = t * total_d;
            for layer_idx in 0..metadata.num_layers {
                let off = row_off + layer_idx * d;
                let use_unit_offset = !is_gemma4
                    && (env_flag("RNB_GEMMA_PLE_UNIT_OFFSET_PROJ_NORM")
                        || (gemma_runtime_flavor == GemmaRuntimeFlavor::Gemma4E2BIt
                            && gemma_ple_layer34_hard_fix_applies(
                                architecture,
                                layer_idx,
                                metadata.num_layers,
                            )));
                if use_unit_offset {
                    kernels::norm::rms_norm_unit_offset_into(
                        &projected_data[off..off + d],
                        proj_norm,
                        norm_eps,
                        &mut normalized[off..off + d],
                    );
                } else {
                    apply_model_norm_into(
                        &projected_data[off..off + d],
                        proj_norm,
                        norm_eps,
                        &mut normalized[off..off + d],
                        architecture,
                    );
                }
            }
        }
    }

    emit_ple_trace(
        "base",
        usize::MAX,
        "token_data",
        &token_data[..token_data.len().min(d)],
    );
    emit_ple_trace(
        "base",
        usize::MAX,
        "projected_normed",
        &normalized[..normalized.len().min(d)],
    );
    emit_ple_dump(usize::MAX, "token_data", &token_data);
    emit_ple_dump(usize::MAX, "projected_normed", &normalized);

    let mut mixed = token_data.clone();
    kernels::elementwise::add_inplace(&mut mixed, &normalized);
    if let Some(scale) = gemma_ple_base_mix_scale() {
        for v in mixed.iter_mut() {
            *v *= scale;
        }
    } else if !env_flag("RNB_GEMMA_PLE_SKIP_MIX_SCALE") {
        for v in mixed.iter_mut() {
            *v *= mix_scale;
        }
    }
    emit_ple_trace("base", usize::MAX, "base_out", &mixed[..mixed.len().min(d)]);
    emit_ple_dump(usize::MAX, "base_out", &mixed);
    Ok(Some(GemmaPerLayerBase {
        mixed,
        token: token_data,
        model: normalized,
    }))
}
