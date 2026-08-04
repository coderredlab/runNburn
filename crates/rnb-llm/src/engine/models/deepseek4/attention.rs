use crate::engine::backend_runtime::{
    cuda_deepseek4_q8_output_projection_if_supported,
    metal_deepseek4_attention_prefill_compressor_fused_requested,
    metal_deepseek4_attention_prefill_index_batch_requested,
    metal_deepseek4_attention_prefill_output_batch_requested,
    metal_deepseek4_prefill_q8_multi_gemm_if_supported, metal_deepseek4_q8_multi_gemv_if_supported,
    metal_deepseek4_q8_output_chain_if_supported, metal_deepseek4_q_front_if_supported,
    metal_prefill_gdn_proj_into_if_supported,
};
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::Result;
use rayon::prelude::*;
use rnb_loader::GGMLType;

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
    if let Some(output) = forward_attention_metal_decode(input, position, weights, state, config)? {
        return Ok(output);
    }
    let profile_enabled = crate::engine::moe_profile::is_enabled();
    let mut mark = profile_enabled.then(std::time::Instant::now);
    let mut lap = |key: &'static str, mark: &mut Option<std::time::Instant>| {
        if let Some(start) = mark {
            crate::engine::moe_profile::record_moe_profile(key, start.elapsed());
            *mark = Some(std::time::Instant::now());
        }
    };
    let qr = rms_norm(
        &weights.q_a.gemv_vec(input)?,
        &weights.q_a_norm,
        config.norm_eps,
    );
    let mut query = weights.q_b.gemv_vec(&qr)?;
    lap("deepseek4:decode:attn:proj_q", &mut mark);
    let kv = rms_norm(
        &weights.kv.gemv_vec(input)?,
        &weights.kv_norm,
        config.norm_eps,
    );
    lap("deepseek4:decode:attn:proj_kv", &mut mark);
    let output = forward_attention_projected(
        input, &qr, None, &mut query, kv, position, weights, state, config, None, None,
    );
    lap("deepseek4:decode:attn:core", &mut mark);
    output
}

fn forward_attention_metal_decode(
    input: &[f32],
    position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
) -> Result<Option<Vec<f32>>> {
    let mut front_weights = vec![&weights.kv];
    if let Some(compressor) = &weights.compressor {
        front_weights.extend([&compressor.kv, &compressor.gate]);
    }
    if let Some(indexer) = &weights.indexer {
        front_weights.extend([&indexer.compressor.kv, &indexer.compressor.gate]);
    }
    let front_inputs = vec![input; front_weights.len()];
    let Some(front_outputs) = project_attention_decode_phase(&front_weights, &front_inputs)? else {
        return Ok(None);
    };
    let mut front_outputs = front_outputs.into_iter();
    let kv = front_outputs.next().expect("DeepSeek4 kv projection");
    let compressor_projected = weights.compressor.as_ref().map(|_| {
        (
            front_outputs
                .next()
                .expect("DeepSeek4 compressor value projection"),
            front_outputs
                .next()
                .expect("DeepSeek4 compressor score projection"),
        )
    });
    let indexer_compressor_projected = weights.indexer.as_ref().map(|_| {
        (
            front_outputs
                .next()
                .expect("DeepSeek4 index compressor value projection"),
            front_outputs
                .next()
                .expect("DeepSeek4 index compressor score projection"),
        )
    });
    debug_assert!(front_outputs.next().is_none());

    let mut query_weights = vec![&weights.q_b];
    if let Some(indexer) = &weights.indexer {
        query_weights.push(&indexer.q_b);
    }
    let (qr, query_outputs) = match metal_deepseek4_q_front_if_supported(
        &weights.q_a,
        &weights.q_a_norm,
        &query_weights,
        input,
        config.norm_eps,
    )? {
        Some(outputs) => outputs,
        None => {
            let qr = rms_norm(
                &weights.q_a.gemv_vec(input)?,
                &weights.q_a_norm,
                config.norm_eps,
            );
            let query_inputs = vec![qr.as_slice(); query_weights.len()];
            let Some(query_outputs) =
                project_attention_decode_phase(&query_weights, &query_inputs)?
            else {
                return Ok(None);
            };
            (qr, query_outputs)
        }
    };
    let mut query_outputs = query_outputs.into_iter();
    let mut query = query_outputs.next().expect("DeepSeek4 q_b projection");
    let index_query = weights.indexer.as_ref().map(|_| {
        query_outputs
            .next()
            .expect("DeepSeek4 index q_b projection")
    });
    debug_assert!(query_outputs.next().is_none());
    let kv = rms_norm(&kv, &weights.kv_norm, config.norm_eps);
    let compressor_projected = compressor_projected
        .as_ref()
        .map(|(value, score)| (value.as_slice(), score.as_slice()));
    let indexer_compressor_projected = indexer_compressor_projected
        .as_ref()
        .map(|(value, score)| (value.as_slice(), score.as_slice()));
    Ok(Some(forward_attention_projected(
        input,
        &qr,
        index_query.as_deref(),
        &mut query,
        kv,
        position,
        weights,
        state,
        config,
        compressor_projected,
        indexer_compressor_projected,
    )?))
}

fn project_attention_decode_phase(
    weights: &[&QuantizedWeight],
    inputs: &[&[f32]],
) -> Result<Option<Vec<Vec<f32>>>> {
    debug_assert_eq!(weights.len(), inputs.len());
    let q8_indices = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (weight.ggml_type == GGMLType::Q8_0).then_some(index))
        .collect::<Vec<_>>();
    if q8_indices.is_empty() {
        return Ok(None);
    }
    let q8_weights = q8_indices
        .iter()
        .map(|&index| weights[index])
        .collect::<Vec<_>>();
    let q8_inputs = q8_indices
        .iter()
        .map(|&index| inputs[index])
        .collect::<Vec<_>>();
    let Some(q8_outputs) = metal_deepseek4_q8_multi_gemv_if_supported(&q8_weights, &q8_inputs)?
    else {
        return Ok(None);
    };
    let mut q8_outputs = q8_outputs.into_iter();
    let mut outputs = Vec::with_capacity(weights.len());
    for (weight, input) in weights.iter().zip(inputs) {
        if weight.ggml_type == GGMLType::Q8_0 {
            outputs.push(q8_outputs.next().expect("DeepSeek4 Q8_0 projection"));
        } else {
            outputs.push(weight.gemv_vec(input)?);
        }
    }
    debug_assert!(q8_outputs.next().is_none());
    Ok(Some(outputs))
}

// The adapter and Metal backend each materialize projection output once. Count both copies plus
// normalized input and returned attention output so the runtime policy caps the actual peak.
pub(super) fn attention_prefill_batch_scratch_bytes_per_token(
    weights: &AttentionWeights,
    config: &DeepSeek4Config,
) -> Option<usize> {
    let query_dim = config.num_heads.checked_mul(config.head_dim)?;
    let compressor_rows = weights.compressor.as_ref().map_or(Some(0), |compressor| {
        compressor.kv.rows.checked_add(compressor.gate.rows)
    })?;
    let index_query_rows = weights
        .indexer
        .as_ref()
        .map_or(0, |indexer| indexer.q_b.rows);
    let indexer_compressor_rows = weights.indexer.as_ref().map_or(Some(0), |indexer| {
        indexer
            .compressor
            .kv
            .rows
            .checked_add(indexer.compressor.gate.rows)
    })?;
    config
        .hidden_dim
        .checked_mul(2)?
        .checked_add(weights.q_a.rows.checked_mul(2)?)?
        .checked_add(query_dim.checked_mul(2)?)?
        .checked_add(index_query_rows.checked_mul(2)?)?
        .checked_add(weights.kv.rows.checked_mul(2)?)?
        .checked_add(compressor_rows.checked_mul(2)?)?
        .checked_add(indexer_compressor_rows.checked_mul(2)?)?
        .checked_mul(std::mem::size_of::<f32>())
}

struct CompressorProjectionBatch {
    values: Vec<f32>,
    scores: Vec<f32>,
    value_dim: usize,
    score_dim: usize,
}

impl CompressorProjectionBatch {
    fn rows(&self, token: usize) -> (&[f32], &[f32]) {
        let value_start = token * self.value_dim;
        let score_start = token * self.score_dim;
        (
            &self.values[value_start..value_start + self.value_dim],
            &self.scores[score_start..score_start + self.score_dim],
        )
    }
}

pub(super) fn forward_attention_batch_if_supported(
    inputs: &[f32],
    start_position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
) -> Result<Option<Vec<f32>>> {
    if inputs.len() <= config.hidden_dim || !inputs.len().is_multiple_of(config.hidden_dim) {
        return Ok(None);
    }
    let seq_len = inputs.len() / config.hidden_dim;
    let Some(q_a) =
        metal_prefill_gdn_proj_into_if_supported(&weights.q_a, inputs, seq_len, config.hidden_dim)?
    else {
        return Ok(None);
    };
    let q_rank = weights.q_a.rows;
    if q_a.len() != seq_len * q_rank {
        return Ok(None);
    }
    let mut qr = Vec::with_capacity(q_a.len());
    for row in q_a.chunks_exact(q_rank) {
        qr.extend(rms_norm(row, &weights.q_a_norm, config.norm_eps));
    }
    drop(q_a);

    let Some(mut queries) =
        metal_prefill_gdn_proj_into_if_supported(&weights.q_b, &qr, seq_len, q_rank)?
    else {
        return Ok(None);
    };

    let Some(mut kv) =
        metal_prefill_gdn_proj_into_if_supported(&weights.kv, inputs, seq_len, config.hidden_dim)?
    else {
        return Ok(None);
    };
    let query_dim = config.num_heads * config.head_dim;
    let kv_dim = weights.kv.rows;
    if queries.len() != seq_len * query_dim || kv.len() != seq_len * kv_dim {
        return Ok(None);
    }
    let kv_norm = tensor_f32(&weights.kv_norm);
    if kv_norm.len() != kv_dim {
        return Ok(None);
    }
    for row in kv.chunks_exact_mut(kv_dim) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let scale = (mean_square + config.norm_eps).sqrt().recip();
        for (value, &gain) in row.iter_mut().zip(kv_norm) {
            *value *= scale * gain;
        }
    }
    let (compressor_batch, indexer_compressor_batch) =
        project_compressor_fused_batch(inputs, seq_len, weights)?;
    let index_queries = if metal_deepseek4_attention_prefill_index_batch_requested() {
        match &weights.indexer {
            Some(indexer) => {
                metal_prefill_gdn_proj_into_if_supported(&indexer.q_b, &qr, seq_len, q_rank)?
            }
            None => None,
        }
    } else {
        None
    };

    if let (Some(indexer), Some(index_queries)) = (&weights.indexer, &index_queries) {
        debug_assert_eq!(index_queries.len(), seq_len * indexer.q_b.rows);
    }

    let mut attention_outputs = Vec::with_capacity(seq_len * query_dim);
    for token in 0..seq_len {
        let input = &inputs[token * config.hidden_dim..(token + 1) * config.hidden_dim];
        let qr = &qr[token * q_rank..(token + 1) * q_rank];
        let query = &mut queries[token * query_dim..(token + 1) * query_dim];
        let token_kv = kv[token * kv_dim..(token + 1) * kv_dim].to_vec();
        let position = start_position + token;
        update_attention_compressors(
            input,
            position,
            weights,
            state,
            config,
            compressor_batch.as_ref().map(|batch| batch.rows(token)),
            indexer_compressor_batch
                .as_ref()
                .map(|batch| batch.rows(token)),
        )?;
        let index_query = index_queries.as_ref().map(|queries| {
            let index_dim = weights
                .indexer
                .as_ref()
                .expect("index batch weights")
                .q_b
                .rows;
            &queries[token * index_dim..(token + 1) * index_dim]
        });
        attention_outputs.extend(forward_attention_core_projected(
            input,
            qr,
            index_query,
            query,
            token_kv,
            position,
            weights,
            state,
            config,
        )?);
    }
    drop(qr);
    drop(queries);
    drop(kv);

    let output = if metal_deepseek4_attention_prefill_output_batch_requested() {
        project_attention_output_batch(&attention_outputs, seq_len, weights, config)?
    } else {
        let mut output = Vec::with_capacity(seq_len * config.hidden_dim);
        for attention_output in attention_outputs.chunks_exact(query_dim) {
            output.extend(project_attention_output(attention_output, weights, config)?);
        }
        output
    };
    Ok(Some(output))
}

#[allow(clippy::too_many_arguments)]
fn forward_attention_projected(
    input: &[f32],
    qr: &[f32],
    index_query: Option<&[f32]>,
    query: &mut [f32],
    kv: Vec<f32>,
    position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
    compressor_projected: Option<(&[f32], &[f32])>,
    indexer_compressor_projected: Option<(&[f32], &[f32])>,
) -> Result<Vec<f32>> {
    let profile_enabled = crate::engine::moe_profile::is_enabled();
    let mut mark = profile_enabled.then(std::time::Instant::now);
    let mut lap = |key: &'static str, mark: &mut Option<std::time::Instant>| {
        if let Some(start) = mark {
            crate::engine::moe_profile::record_moe_profile(key, start.elapsed());
            *mark = Some(std::time::Instant::now());
        }
    };
    update_attention_compressors(
        input,
        position,
        weights,
        state,
        config,
        compressor_projected,
        indexer_compressor_projected,
    )?;
    lap("deepseek4:decode:attn:c:compressors", &mut mark);
    let attention_output = forward_attention_core_projected(
        input,
        qr,
        index_query,
        query,
        kv,
        position,
        weights,
        state,
        config,
    )?;
    lap("deepseek4:decode:attn:c:coreproj", &mut mark);
    let output = project_attention_output(&attention_output, weights, config);
    lap("deepseek4:decode:attn:c:out", &mut mark);
    output
}

#[allow(clippy::too_many_arguments)]
fn forward_attention_core_projected(
    input: &[f32],
    qr: &[f32],
    index_query: Option<&[f32]>,
    query: &mut [f32],
    mut kv: Vec<f32>,
    position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let profile_enabled = crate::engine::moe_profile::is_enabled();
    let mut mark = profile_enabled.then(std::time::Instant::now);
    let mut lap = |key: &'static str, mark: &mut Option<std::time::Instant>| {
        if let Some(start) = mark {
            crate::engine::moe_profile::record_moe_profile(key, start.elapsed());
            *mark = Some(std::time::Instant::now());
        }
    };
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

    let rope_start = config.head_dim - config.rope_dim;
    apply_rope(
        &mut kv[rope_start..],
        position,
        config,
        compressed_layer,
        false,
    );
    fp8_quantize_inplace(&mut kv[..rope_start], 64);

    state.window.push_back(kv);
    while state.window.len() > config.window_size {
        state.window.pop_front();
    }
    lap("deepseek4:decode:attn:c:prep", &mut mark);

    let compressed_indices =
        if let (Some(indexer), Some(index_state)) = (&weights.indexer, &state.indexer_compressor) {
            select_indexed_compressed(
                input,
                qr,
                index_query,
                position,
                indexer,
                index_state,
                config,
            )?
        } else {
            state
                .compressor
                .as_ref()
                .map(|compressor| (0..compressor.compressed.len()).collect())
                .unwrap_or_default()
        };
    lap("deepseek4:decode:attn:c:index", &mut mark);

    let compressed = state.compressor.as_ref();
    let selected_count = state.window.len() + compressed_indices.len();
    let mut attention_output = vec![0.0f32; config.num_heads * config.head_dim];
    if selected_count > 0 {
        let sinks = tensor_f32(&weights.sinks);
        let scale = (config.head_dim as f32).sqrt().recip();
        let compute_head = |head_index: usize, output: &mut [f32]| {
            let head = &query[head_index * config.head_dim..(head_index + 1) * config.head_dim];
            let mut scores = Vec::with_capacity(selected_count);
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
        };
        if selected_count.saturating_mul(config.head_dim) >= config.hidden_dim {
            attention_output
                .par_chunks_mut(config.head_dim)
                .enumerate()
                .for_each(|(head_index, output)| compute_head(head_index, output));
        } else {
            attention_output
                .chunks_mut(config.head_dim)
                .enumerate()
                .for_each(|(head_index, output)| compute_head(head_index, output));
        }
    }
    lap("deepseek4:decode:attn:c:heads", &mut mark);

    Ok(attention_output)
}

pub(super) fn project_attention_output(
    attention_output: &[f32],
    weights: &AttentionWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let heads_per_group = config.num_heads / config.output_groups;
    let group_input_len = heads_per_group * config.head_dim;
    let group_weights = weights.output_a_groups.iter().collect::<Vec<_>>();
    let group_inputs = (0..config.output_groups)
        .map(|group| {
            let start = group * group_input_len;
            &attention_output[start..start + group_input_len]
        })
        .collect::<Vec<_>>();
    if let Some(output) = metal_deepseek4_q8_output_chain_if_supported(
        &group_weights,
        &group_inputs,
        &weights.output_b,
    )? {
        return Ok(output);
    }
    if let Some(group_outputs) =
        metal_deepseek4_q8_multi_gemv_if_supported(&group_weights, &group_inputs)?
    {
        let low_rank = group_outputs.into_iter().flatten().collect::<Vec<_>>();
        let output_weights = [&weights.output_b];
        let output_inputs = [low_rank.as_slice()];
        if let Some(mut output) =
            metal_deepseek4_q8_multi_gemv_if_supported(&output_weights, &output_inputs)?
        {
            return Ok(output.swap_remove(0));
        }
        return weights.output_b.gemv_vec(&low_rank);
    }

    if let Some(output) = cuda_deepseek4_q8_output_projection_if_supported(
        &weights.output_a_groups,
        &weights.output_b,
        attention_output,
    )? {
        return Ok(output);
    }

    let mut low_rank = Vec::with_capacity(config.output_groups * config.output_lora_rank);
    for (group, projection) in weights.output_a_groups.iter().enumerate() {
        let start = group * group_input_len;
        low_rank.extend(projection.gemv_vec(&attention_output[start..start + group_input_len])?);
    }
    weights.output_b.gemv_vec(&low_rank)
}

fn project_attention_output_batch(
    attention_outputs: &[f32],
    seq_len: usize,
    weights: &AttentionWeights,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let attention_dim = config.num_heads * config.head_dim;
    let group_input_len = attention_dim / config.output_groups;
    let low_rank_dim = config.output_groups * config.output_lora_rank;
    debug_assert_eq!(attention_outputs.len(), seq_len * attention_dim);

    let mut low_rank = vec![0.0f32; seq_len * low_rank_dim];
    for (group, projection) in weights.output_a_groups.iter().enumerate() {
        debug_assert_eq!(projection.cols, group_input_len);
        debug_assert_eq!(projection.rows, config.output_lora_rank);
        let group_start = group * group_input_len;
        let mut group_inputs = Vec::with_capacity(seq_len * group_input_len);
        for attention_output in attention_outputs.chunks_exact(attention_dim) {
            group_inputs
                .extend_from_slice(&attention_output[group_start..group_start + group_input_len]);
        }
        let projected = match metal_prefill_gdn_proj_into_if_supported(
            projection,
            &group_inputs,
            seq_len,
            group_input_len,
        )? {
            Some(projected) => projected,
            None => projection.gemv_vec(&group_inputs)?,
        };
        debug_assert_eq!(projected.len(), seq_len * config.output_lora_rank);
        for (token, row) in projected.chunks_exact(config.output_lora_rank).enumerate() {
            let start = token * low_rank_dim + group * config.output_lora_rank;
            low_rank[start..start + config.output_lora_rank].copy_from_slice(row);
        }
    }

    debug_assert_eq!(weights.output_b.cols, low_rank_dim);
    match metal_prefill_gdn_proj_into_if_supported(
        &weights.output_b,
        &low_rank,
        seq_len,
        low_rank_dim,
    )? {
        Some(output) => Ok(output),
        None => weights.output_b.gemv_vec(&low_rank),
    }
}

fn project_compressor_fused_batch(
    inputs: &[f32],
    seq_len: usize,
    weights: &AttentionWeights,
) -> Result<(
    Option<CompressorProjectionBatch>,
    Option<CompressorProjectionBatch>,
)> {
    if !metal_deepseek4_attention_prefill_compressor_fused_requested() {
        return Ok((None, None));
    }
    let mut projection_weights = Vec::with_capacity(4);
    if let Some(compressor) = &weights.compressor {
        projection_weights.extend([&compressor.kv, &compressor.gate]);
    }
    if let Some(indexer) = &weights.indexer {
        projection_weights.extend([&indexer.compressor.kv, &indexer.compressor.gate]);
    }
    let Some(outputs) =
        metal_deepseek4_prefill_q8_multi_gemm_if_supported(&projection_weights, inputs, seq_len)?
    else {
        return Ok((None, None));
    };
    let mut outputs = outputs.into_iter();
    let compressor_batch =
        weights
            .compressor
            .as_ref()
            .map(|compressor| CompressorProjectionBatch {
                values: outputs.next().expect("DeepSeek4 compressor values"),
                scores: outputs.next().expect("DeepSeek4 compressor scores"),
                value_dim: compressor.kv.rows,
                score_dim: compressor.gate.rows,
            });
    let indexer_compressor_batch =
        weights
            .indexer
            .as_ref()
            .map(|indexer| CompressorProjectionBatch {
                values: outputs.next().expect("DeepSeek4 index compressor values"),
                scores: outputs.next().expect("DeepSeek4 index compressor scores"),
                value_dim: indexer.compressor.kv.rows,
                score_dim: indexer.compressor.gate.rows,
            });
    debug_assert!(outputs.next().is_none());
    Ok((compressor_batch, indexer_compressor_batch))
}

#[allow(clippy::too_many_arguments)]
fn update_attention_compressors(
    input: &[f32],
    position: usize,
    weights: &AttentionWeights,
    state: &mut AttentionState,
    config: &DeepSeek4Config,
    compressor_projected: Option<(&[f32], &[f32])>,
    indexer_compressor_projected: Option<(&[f32], &[f32])>,
) -> Result<()> {
    if let (Some(compressor), Some(compressor_state)) = (&weights.compressor, &mut state.compressor)
    {
        match compressor_projected {
            Some((value, score)) => update_compressor_projected(
                value.to_vec(),
                score.to_vec(),
                position,
                compressor,
                compressor_state,
                config,
                false,
            )?,
            None => {
                update_compressor(input, position, compressor, compressor_state, config, false)?
            }
        }
    }
    if let (Some(indexer), Some(indexer_state)) = (&weights.indexer, &mut state.indexer_compressor)
    {
        match indexer_compressor_projected {
            Some((value, score)) => update_compressor_projected(
                value.to_vec(),
                score.to_vec(),
                position,
                &indexer.compressor,
                indexer_state,
                config,
                true,
            )?,
            None => update_compressor(
                input,
                position,
                &indexer.compressor,
                indexer_state,
                config,
                true,
            )?,
        }
    }
    Ok(())
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
    let score = weights.gate.gemv_vec(input)?;
    update_compressor_projected(value, score, position, weights, state, config, rotate_fp4)
}

#[allow(clippy::too_many_arguments)]
fn update_compressor_projected(
    value: Vec<f32>,
    mut score: Vec<f32>,
    position: usize,
    weights: &CompressorWeights,
    state: &mut CompressorState,
    config: &DeepSeek4Config,
    rotate_fp4: bool,
) -> Result<()> {
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
    projected_query: Option<&[f32]>,
    position: usize,
    weights: &IndexerWeights,
    state: &CompressorState,
    config: &DeepSeek4Config,
) -> Result<Vec<usize>> {
    if state.compressed.is_empty() {
        return Ok(Vec::new());
    }
    let mut query = match projected_query {
        Some(query) => query.to_vec(),
        None => weights.q_b.gemv_vec(qr)?,
    };
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
    crate::engine::dense_dispatch::dot_f32_row(left, right)
}

#[cfg(test)]
mod tests {

    #[test]
    fn accelerated_dot_stays_within_attention_tolerance() {
        let left = (0..512)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.0078125)
            .collect::<Vec<_>>();
        let right = (0..512)
            .map(|index| ((index * 53 % 113) as f32 - 56.0) * 0.00390625)
            .collect::<Vec<_>>();
        let expected = left.iter().zip(&right).map(|(&a, &b)| a * b).sum::<f32>();
        let actual = super::dot(&left, &right);
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, actual {actual}, tolerance {tolerance}"
        );
    }
}
