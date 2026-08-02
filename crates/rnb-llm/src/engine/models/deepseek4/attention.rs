use crate::engine::dense_dispatch::gemv_f32;
use crate::error::Result;

use super::math::{
    apply_rope, fp4_quantize_inplace, fp8_quantize_inplace, hadamard_inplace, rms_norm,
    rms_unit_inplace, tensor_f32,
};
use super::state::{AttentionState, CompressorState};
use super::weights::{AttentionWeights, CompressorWeights, DeepSeek4Config, IndexerWeights};

pub(super) fn forward_attention(
    input: &[f32],
    position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let qr = rms_norm(
        &weights.q_a.gemv_vec(input)?,
        &weights.q_a_norm,
        config.norm_eps,
    );
    let mut query = weights.q_b.gemv_vec(&qr)?;
    let compressed_layer = weights.compressor.is_some();
    for head in query.chunks_exact_mut(config.head_dim) {
        rms_unit_inplace(head, config.norm_eps);
        let rope_start = config.head_dim - config.rope_dim;
        apply_rope(
            &mut head[rope_start..],
            position,
            config,
            compressed_layer,
            false,
        );
    }

    let mut kv = rms_norm(
        &weights.kv.gemv_vec(input)?,
        &weights.kv_norm,
        config.norm_eps,
    );
    let rope_start = config.head_dim - config.rope_dim;
    apply_rope(
        &mut kv[rope_start..],
        position,
        config,
        compressed_layer,
        false,
    );
    fp8_quantize_inplace(&mut kv[..rope_start], 64);

    if let (Some(compressor), Some(compressor_state)) = (&weights.compressor, &mut state.compressor)
    {
        update_compressor(input, position, compressor, compressor_state, config, false)?;
    }
    if let (Some(indexer), Some(indexer_state)) = (&weights.indexer, &mut state.indexer_compressor)
    {
        update_compressor(
            input,
            position,
            &indexer.compressor,
            indexer_state,
            config,
            true,
        )?;
    }

    state.window.push_back(kv);
    while state.window.len() > config.window_size {
        state.window.pop_front();
    }

    let compressed_indices =
        if let (Some(indexer), Some(index_state)) = (&weights.indexer, &state.indexer_compressor) {
            select_indexed_compressed(input, &qr, position, indexer, index_state, config)?
        } else {
            state
                .compressor
                .as_ref()
                .map(|compressor| (0..compressor.compressed.len()).collect())
                .unwrap_or_default()
        };

    let compressed = state.compressor.as_ref();
    let selected_count = state.window.len() + compressed_indices.len();
    let mut attention_output = vec![0.0f32; config.num_heads * config.head_dim];
    if selected_count > 0 {
        let sinks = tensor_f32(&weights.sinks);
        let scale = (config.head_dim as f32).sqrt().recip();
        let mut scores = Vec::with_capacity(selected_count);
        for head_index in 0..config.num_heads {
            let head = &query[head_index * config.head_dim..(head_index + 1) * config.head_dim];
            scores.clear();
            for key in &state.window {
                scores.push(dot(head, key) * scale);
            }
            if let Some(compressed) = compressed {
                for &index in &compressed_indices {
                    scores.push(dot(head, &compressed.compressed[index]) * scale);
                }
            }
            let max_score = scores.iter().copied().fold(sinks[head_index], f32::max);
            let mut denominator = (sinks[head_index] - max_score).exp();
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            let output = &mut attention_output
                [head_index * config.head_dim..(head_index + 1) * config.head_dim];
            let mut score_index = 0;
            for key in &state.window {
                let probability = scores[score_index] / denominator;
                score_index += 1;
                for (dst, &value) in output.iter_mut().zip(key) {
                    *dst += probability * value;
                }
            }
            if let Some(compressed) = compressed {
                for &index in &compressed_indices {
                    let probability = scores[score_index] / denominator;
                    score_index += 1;
                    for (dst, &value) in output.iter_mut().zip(&compressed.compressed[index]) {
                        *dst += probability * value;
                    }
                }
            }
            apply_rope(
                &mut output[rope_start..],
                position,
                config,
                compressed_layer,
                true,
            );
        }
    }

    let heads_per_group = config.num_heads / config.output_groups;
    let group_input_len = heads_per_group * config.head_dim;
    let mut low_rank = Vec::with_capacity(config.output_groups * config.output_lora_rank);
    for (group, projection) in weights.output_a_groups.iter().enumerate() {
        let start = group * group_input_len;
        low_rank.extend(projection.gemv_vec(&attention_output[start..start + group_input_len])?);
    }
    weights.output_b.gemv_vec(&low_rank)
}

fn update_compressor(
    input: &[f32],
    position: usize,
    weights: &CompressorWeights,
    state: &mut CompressorState,
    config: &DeepSeek4Config,
    rotate_fp4: bool,
) -> Result<()> {
    let value = weights.kv.gemv_vec(input)?;
    let mut score = weights.gate.gemv_vec(input)?;
    let ape = tensor_f32(&weights.ape);
    let output_dim = value.len();
    let ape_row =
        &ape[(position % weights.ratio) * output_dim..(position % weights.ratio + 1) * output_dim];
    for (dst, &bias) in score.iter_mut().zip(ape_row) {
        *dst += bias;
    }
    state.current_values.push(value);
    state.current_scores.push(score);
    if state.current_values.len() != weights.ratio {
        return Ok(());
    }

    let overlap = weights.ratio == 4;
    let head_dim = weights.head_dim;
    let mut compressed = vec![0.0f32; head_dim];
    for feature in 0..head_dim {
        let mut max_score = f32::NEG_INFINITY;
        if overlap {
            for score in &state.previous {
                max_score = max_score.max(score[output_dim + feature]);
            }
            for score in &state.current_scores {
                max_score = max_score.max(score[head_dim + feature]);
            }
        } else {
            for score in &state.current_scores {
                max_score = max_score.max(score[feature]);
            }
        }
        if !max_score.is_finite() {
            continue;
        }
        let mut denominator = 0.0f32;
        if overlap {
            for score in &state.previous {
                denominator += (score[output_dim + feature] - max_score).exp();
            }
            for score in &state.current_scores {
                denominator += (score[head_dim + feature] - max_score).exp();
            }
            for (value, score) in state.previous.iter().zip(&state.previous) {
                compressed[feature] +=
                    value[feature] * (score[output_dim + feature] - max_score).exp();
            }
            for (value, score) in state.current_values.iter().zip(&state.current_scores) {
                compressed[feature] +=
                    value[head_dim + feature] * (score[head_dim + feature] - max_score).exp();
            }
        } else {
            for score in &state.current_scores {
                denominator += (score[feature] - max_score).exp();
            }
            for (value, score) in state.current_values.iter().zip(&state.current_scores) {
                compressed[feature] += value[feature] * (score[feature] - max_score).exp();
            }
        }
        compressed[feature] /= denominator;
    }

    if overlap {
        state.previous = state.current_values.clone();
        for (value, score) in state.previous.iter_mut().zip(&state.current_scores) {
            value.extend_from_slice(score);
        }
    }
    state.current_values.clear();
    state.current_scores.clear();

    compressed = rms_norm(&compressed, &weights.norm, config.norm_eps);
    let rope_start = head_dim - config.rope_dim;
    apply_rope(
        &mut compressed[rope_start..],
        position + 1 - weights.ratio,
        config,
        true,
        false,
    );
    if rotate_fp4 {
        hadamard_inplace(&mut compressed);
        fp4_quantize_inplace(&mut compressed, 32);
    } else {
        fp8_quantize_inplace(&mut compressed[..rope_start], 64);
    }
    state.compressed.push(compressed);
    Ok(())
}

fn select_indexed_compressed(
    input: &[f32],
    qr: &[f32],
    position: usize,
    weights: &IndexerWeights,
    state: &CompressorState,
    config: &DeepSeek4Config,
) -> Result<Vec<usize>> {
    if state.compressed.is_empty() {
        return Ok(Vec::new());
    }
    let mut query = weights.q_b.gemv_vec(qr)?;
    for head in query.chunks_exact_mut(config.index_head_dim) {
        let rope_start = config.index_head_dim - config.rope_dim;
        apply_rope(&mut head[rope_start..], position, config, true, false);
        hadamard_inplace(head);
        fp4_quantize_inplace(head, 32);
    }
    let projection = tensor_f32(&weights.projection);
    let mut head_weights = vec![0.0f32; config.index_heads];
    gemv_f32(
        projection,
        input,
        &mut head_weights,
        config.index_heads,
        config.hidden_dim,
        1,
    );
    let weight_scale = (config.index_head_dim * config.index_heads) as f32;
    let weight_scale = weight_scale.sqrt().recip();
    for weight in &mut head_weights {
        *weight *= weight_scale;
    }

    let mut scores: Vec<(usize, f32)> = state
        .compressed
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let score = query
                .chunks_exact(config.index_head_dim)
                .zip(&head_weights)
                .map(|(head, &weight)| dot(head, key).max(0.0) * weight)
                .sum::<f32>();
            (index, score)
        })
        .collect();
    scores.sort_unstable_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    scores.truncate(config.index_topk.min(scores.len()));
    Ok(scores.into_iter().map(|(index, _)| index).collect())
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}
