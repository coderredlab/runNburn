use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::Result;
use rnb_core::tensor::Tensor;

use super::attention::forward_attention;
use super::math::{hyper_head, hyper_post, hyper_pre, rms_norm};
use super::moe::{forward_moe, forward_moe_batch};
use super::weights::DeepSeek4Weights;

pub(in crate::engine) fn forward_tokens(
    model: &mut DeepSeek4Weights,
    token_embedding: &QuantizedWeight,
    output_norm: &Tensor,
    output: &QuantizedWeight,
    tokens: &[u32],
    expected_position: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if model.state.position != expected_position {
        model.state.clear();
    }
    if tokens.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if tokens.len() == 1 {
        return forward_single_token(model, token_embedding, output_norm, output, tokens[0]);
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

    for (layer, attention_state) in model.layers.iter().zip(&mut model.state.layers) {
        let mut attention_hidden = Vec::with_capacity(hidden.len());
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
            attention_hidden.extend(hyper_post(&attn_output, residual, mix, config));
        }
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

    let last_hidden = &hidden[(seq_len - 1) * row_width..seq_len * row_width];
    let mut final_hidden = hyper_head(
        last_hidden,
        &model.output_hc_function,
        &model.output_hc_scale,
        &model.output_hc_base,
        config,
    );
    final_hidden = rms_norm(&final_hidden, output_norm, config.norm_eps);
    let logits = output.gemv_vec(&final_hidden)?;
    model.state.position += seq_len;
    Ok((logits, final_hidden))
}

fn forward_single_token(
    model: &mut DeepSeek4Weights,
    token_embedding: &QuantizedWeight,
    output_norm: &Tensor,
    output: &QuantizedWeight,
    token: u32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let position = model.state.position;
    let embedding = token_embedding.gather(&[token])?;
    let embedding = kernels::tensor_as_f32_slice(&embedding);
    let mut hidden = Vec::with_capacity(model.config.hc_count * model.config.hidden_dim);
    for _ in 0..model.config.hc_count {
        hidden.extend_from_slice(embedding);
    }
    for (layer, attention_state) in model.layers.iter().zip(&mut model.state.layers) {
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
    let mut final_hidden = hyper_head(
        &hidden,
        &model.output_hc_function,
        &model.output_hc_scale,
        &model.output_hc_base,
        &model.config,
    );
    final_hidden = rms_norm(&final_hidden, output_norm, model.config.norm_eps);
    let logits = output.gemv_vec(&final_hidden)?;
    model.state.position += 1;
    Ok((logits, final_hidden))
}
