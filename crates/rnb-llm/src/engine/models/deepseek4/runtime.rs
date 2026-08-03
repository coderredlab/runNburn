use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::{LlmError, Result};
use rnb_core::tensor::Tensor;

use super::attention::{
    attention_prefill_batch_scratch_bytes_per_token, forward_attention,
    forward_attention_batch_if_supported,
};
use super::math::{hyper_head, hyper_post, hyper_pre, rms_norm};
use super::moe::{forward_moe, forward_moe_batch};
use super::weights::DeepSeek4Weights;

pub(in crate::engine) struct DeepSeek4ForwardOutput {
    pub(in crate::engine) logits: Vec<f32>,
    pub(in crate::engine) final_hidden_rows: Vec<f32>,
    pub(in crate::engine) extracted_features: Vec<f32>,
}

pub(in crate::engine) fn forward_tokens(
    model: &mut DeepSeek4Weights,
    token_embedding: &QuantizedWeight,
    output_norm: &Tensor,
    output: &QuantizedWeight,
    tokens: &[u32],
    expected_position: usize,
    target_layers: &[usize],
    all_logits: bool,
) -> Result<DeepSeek4ForwardOutput> {
    ensure_state_position(model.state.position, expected_position)?;
    if tokens.is_empty() {
        return Ok(DeepSeek4ForwardOutput {
            logits: Vec::new(),
            final_hidden_rows: Vec::new(),
            extracted_features: Vec::new(),
        });
    }
    if tokens.len() == 1 {
        return forward_single_token(
            model,
            token_embedding,
            output_norm,
            output,
            tokens[0],
            target_layers,
        );
    }
    let config = &model.config;
    let seq_len = tokens.len();
    let row_width = config.hc_count * config.hidden_dim;
    let start_position = model.state.position;
    let embeddings = token_embedding.gather(tokens)?;
    let embeddings = kernels::tensor_as_f32_slice(&embeddings);
    debug_assert_eq!(embeddings.len(), seq_len * config.hidden_dim);
    let mut hidden = vec![0.0f32; seq_len * row_width];
    for token in 0..seq_len {
        let embedding = &embeddings[token * config.hidden_dim..(token + 1) * config.hidden_dim];
        for copy in 0..config.hc_count {
            let start = token * row_width + copy * config.hidden_dim;
            hidden[start..start + config.hidden_dim].copy_from_slice(embedding);
        }
    }
    let mut layer_features = vec![None; target_layers.len()];

    for (layer_index, (layer, attention_state)) in
        model.layers.iter().zip(&mut model.state.layers).enumerate()
    {
        capture_dspark_inputs(
            layer_index,
            &hidden,
            seq_len,
            config,
            target_layers,
            &mut layer_features,
        );
        let attention_batch_tokens =
            if crate::engine::backend_runtime::metal_deepseek4_attention_prefill_batch_requested() {
                attention_prefill_batch_scratch_bytes_per_token(&layer.attention, config)
                    .map(|scratch_bytes_per_token| {
                        crate::engine::backend_runtime::metal_deepseek4_attention_prefill_batch_tokens(
                            seq_len,
                            scratch_bytes_per_token,
                        )
                    })
                    .unwrap_or(1)
            } else {
                1
            };
        let attention_hidden = if attention_batch_tokens >= 2 {
            let mut next_hidden = Vec::with_capacity(hidden.len());
            let mut chunk_start_token = 0;
            for residual_chunk in hidden.chunks(attention_batch_tokens * row_width) {
                let chunk_len = residual_chunk.len() / row_width;
                let mut mixes = Vec::with_capacity(chunk_len);
                let mut attention_inputs = Vec::with_capacity(chunk_len * config.hidden_dim);
                for residual in residual_chunk.chunks_exact(row_width) {
                    let mut mix = hyper_pre(residual, &layer.attn_hc, config);
                    attention_inputs.extend(rms_norm(
                        &mix.branch,
                        &layer.attn_norm,
                        config.norm_eps,
                    ));
                    mix.branch = Vec::new();
                    mixes.push(mix);
                }
                let batched = forward_attention_batch_if_supported(
                    &attention_inputs,
                    start_position + chunk_start_token,
                    &layer.attention,
                    attention_state,
                    config,
                )?;
                if let Some(attention_outputs) = batched {
                    debug_assert_eq!(attention_outputs.len(), chunk_len * config.hidden_dim);
                    for ((residual, mix), attn_output) in residual_chunk
                        .chunks_exact(row_width)
                        .zip(mixes)
                        .zip(attention_outputs.chunks_exact(config.hidden_dim))
                    {
                        next_hidden.extend(hyper_post(attn_output, residual, mix, config));
                    }
                } else {
                    for (token, ((residual, mix), attn_input)) in residual_chunk
                        .chunks_exact(row_width)
                        .zip(mixes)
                        .zip(attention_inputs.chunks_exact(config.hidden_dim))
                        .enumerate()
                    {
                        let attn_output = forward_attention(
                            attn_input,
                            start_position + chunk_start_token + token,
                            &layer.attention,
                            attention_state,
                            config,
                        )?;
                        next_hidden.extend(hyper_post(&attn_output, residual, mix, config));
                    }
                }
                chunk_start_token += chunk_len;
            }
            next_hidden
        } else {
            let mut next_hidden = Vec::with_capacity(hidden.len());
            for (token, residual) in hidden.chunks_exact(row_width).enumerate() {
                let mix = hyper_pre(residual, &layer.attn_hc, config);
                let attn_input = rms_norm(&mix.branch, &layer.attn_norm, config.norm_eps);
                let attn_output = forward_attention(
                    &attn_input,
                    start_position + token,
                    &layer.attention,
                    attention_state,
                    config,
                )?;
                next_hidden.extend(hyper_post(&attn_output, residual, mix, config));
            }
            next_hidden
        };
        hidden = attention_hidden;

        let mut mixes = Vec::with_capacity(seq_len);
        let mut ffn_inputs = Vec::with_capacity(seq_len * config.hidden_dim);
        for residual in hidden.chunks_exact(row_width) {
            let mix = hyper_pre(residual, &layer.ffn_hc, config);
            ffn_inputs.extend(rms_norm(&mix.branch, &layer.ffn_norm, config.norm_eps));
            mixes.push(mix);
        }
        let ffn_outputs = forward_moe_batch(&ffn_inputs, tokens, &layer.moe, config)?;
        let mut next_hidden = Vec::with_capacity(hidden.len());
        for ((residual, mix), ffn_output) in hidden
            .chunks_exact(row_width)
            .zip(mixes)
            .zip(ffn_outputs.chunks_exact(config.hidden_dim))
        {
            next_hidden.extend(hyper_post(ffn_output, residual, mix, config));
        }
        hidden = next_hidden;
    }
    capture_dspark_inputs(
        model.layers.len(),
        &hidden,
        seq_len,
        config,
        target_layers,
        &mut layer_features,
    );

    let hidden_rows = if all_logits {
        hidden.as_slice()
    } else {
        &hidden[(seq_len - 1) * row_width..seq_len * row_width]
    };
    let mut final_hidden_rows =
        Vec::with_capacity((if all_logits { seq_len } else { 1 }) * config.hidden_dim);
    for hidden_row in hidden_rows.chunks_exact(row_width) {
        let collapsed = hyper_head(
            hidden_row,
            &model.output_hc_function,
            &model.output_hc_scale,
            &model.output_hc_base,
            config,
        );
        final_hidden_rows.extend(rms_norm(&collapsed, output_norm, config.norm_eps));
    }
    let logits = output.gemv_vec(&final_hidden_rows)?;
    let extracted_features = transpose_dspark_inputs(layer_features, seq_len, config.hidden_dim)?;
    model.state.position += seq_len;
    Ok(DeepSeek4ForwardOutput {
        logits,
        final_hidden_rows,
        extracted_features,
    })
}

fn forward_single_token(
    model: &mut DeepSeek4Weights,
    token_embedding: &QuantizedWeight,
    output_norm: &Tensor,
    output: &QuantizedWeight,
    token: u32,
    target_layers: &[usize],
) -> Result<DeepSeek4ForwardOutput> {
    let position = model.state.position;
    let embedding = token_embedding.gather(&[token])?;
    let embedding = kernels::tensor_as_f32_slice(&embedding);
    let mut hidden = Vec::with_capacity(model.config.hc_count * model.config.hidden_dim);
    for _ in 0..model.config.hc_count {
        hidden.extend_from_slice(embedding);
    }
    let mut layer_features = vec![None; target_layers.len()];
    for (layer_index, (layer, attention_state)) in
        model.layers.iter().zip(&mut model.state.layers).enumerate()
    {
        capture_dspark_inputs(
            layer_index,
            &hidden,
            1,
            &model.config,
            target_layers,
            &mut layer_features,
        );
        let residual = hidden;
        let mix = hyper_pre(&residual, &layer.attn_hc, &model.config);
        let attn_input = rms_norm(&mix.branch, &layer.attn_norm, model.config.norm_eps);
        let attn_output = forward_attention(
            &attn_input,
            position,
            &layer.attention,
            attention_state,
            &model.config,
        )?;
        hidden = hyper_post(&attn_output, &residual, mix, &model.config);

        let residual = hidden;
        let mix = hyper_pre(&residual, &layer.ffn_hc, &model.config);
        let ffn_input = rms_norm(&mix.branch, &layer.ffn_norm, model.config.norm_eps);
        let ffn_output = forward_moe(&ffn_input, token, &layer.moe, &model.config)?;
        hidden = hyper_post(&ffn_output, &residual, mix, &model.config);
    }
    capture_dspark_inputs(
        model.layers.len(),
        &hidden,
        1,
        &model.config,
        target_layers,
        &mut layer_features,
    );
    let collapsed = hyper_head(
        &hidden,
        &model.output_hc_function,
        &model.output_hc_scale,
        &model.output_hc_base,
        &model.config,
    );
    let final_hidden_rows = rms_norm(&collapsed, output_norm, model.config.norm_eps);
    let logits = output.gemv_vec(&final_hidden_rows)?;
    let extracted_features = transpose_dspark_inputs(layer_features, 1, model.config.hidden_dim)?;
    model.state.position += 1;
    Ok(DeepSeek4ForwardOutput {
        logits,
        final_hidden_rows,
        extracted_features,
    })
}

fn capture_dspark_inputs(
    layer_index: usize,
    hidden: &[f32],
    seq_len: usize,
    config: &super::weights::DeepSeek4Config,
    target_layers: &[usize],
    layer_features: &mut [Option<Vec<f32>>],
) {
    for (slot, &target_layer) in target_layers.iter().enumerate() {
        if target_layer == layer_index {
            layer_features[slot] = Some(hc_mean_rows(hidden, seq_len, config));
        }
    }
}

fn hc_mean_rows(
    hidden: &[f32],
    seq_len: usize,
    config: &super::weights::DeepSeek4Config,
) -> Vec<f32> {
    let row_width = config.hc_count * config.hidden_dim;
    let mut means = vec![0.0f32; seq_len * config.hidden_dim];
    let scale = 1.0 / config.hc_count as f32;
    for (input_row, output_row) in hidden
        .chunks_exact(row_width)
        .zip(means.chunks_exact_mut(config.hidden_dim))
    {
        for copy in input_row.chunks_exact(config.hidden_dim) {
            for (output, value) in output_row.iter_mut().zip(copy) {
                *output += value * scale;
            }
        }
    }
    means
}

fn transpose_dspark_inputs(
    layer_features: Vec<Option<Vec<f32>>>,
    seq_len: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>> {
    if layer_features.is_empty() {
        return Ok(Vec::new());
    }
    if layer_features.iter().any(Option::is_none) {
        return Err(LlmError::Forward(
            "DSpark target layer feature was not captured".to_string(),
        ));
    }
    let layer_features = layer_features
        .into_iter()
        .map(Option::unwrap)
        .collect::<Vec<_>>();
    let mut features = Vec::with_capacity(seq_len * layer_features.len() * hidden_dim);
    for token in 0..seq_len {
        let start = token * hidden_dim;
        for layer in &layer_features {
            features.extend_from_slice(&layer[start..start + hidden_dim]);
        }
    }
    Ok(features)
}

fn ensure_state_position(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(LlmError::Forward(format!(
        "DeepSeek4 sequence state position {actual} does not match engine position {expected}"
    )))
}

#[cfg(test)]
mod tests {
    use super::ensure_state_position;

    #[test]
    fn rejects_desynchronized_sequence_position() {
        assert!(ensure_state_position(7, 7).is_ok());
        assert!(ensure_state_position(0, 7).is_err());
        assert!(ensure_state_position(7, 0).is_err());
    }
}
