use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::Result;
use rnb_core::tensor::Tensor;

use super::attention::forward_attention;
use super::math::{hyper_head, hyper_post, hyper_pre, rms_norm};
use super::moe::forward_moe;
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
    let mut logits = Vec::new();
    let mut final_hidden = Vec::new();
    for &token in tokens {
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

        final_hidden = hyper_head(
            &hidden,
            &model.output_hc_function,
            &model.output_hc_scale,
            &model.output_hc_base,
            &model.config,
        );
        final_hidden = rms_norm(&final_hidden, output_norm, model.config.norm_eps);
        logits = output.gemv_vec(&final_hidden)?;
        model.state.position += 1;
    }
    Ok((logits, final_hidden))
}
