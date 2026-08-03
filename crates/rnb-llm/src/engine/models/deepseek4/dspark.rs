use std::collections::VecDeque;

use rayon::prelude::*;
use rnb_core::tensor::Tensor;
use rnb_loader::{Architecture, LoadedModel};

use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::{LlmError, Result};

use super::attention::project_attention_output;
use super::math::{
    apply_rope, hyper_head, hyper_post, hyper_pre, rms_norm, rms_unit_inplace, tensor_f32,
};
use super::moe::forward_moe_batch;
use super::weights::{
    load_deepseek4_weights, DeepSeek4Config, DeepSeek4Weights, F32WeightLoader,
    QuantizedWeightLoader,
};

#[derive(Clone)]
pub(crate) struct DsparkSequenceState {
    committed_keys: Vec<VecDeque<Vec<f32>>>,
    position: usize,
}

impl DsparkSequenceState {
    pub(in crate::engine) fn heap_byte_size(&self) -> u64 {
        self.committed_keys
            .iter()
            .flat_map(|keys| keys.iter())
            .map(|key| key.capacity() as u64 * std::mem::size_of::<f32>() as u64)
            .sum()
    }
}

pub(crate) struct DsparkDraft {
    pub(crate) tokens: Vec<u32>,
    pub(crate) confidences: Vec<f32>,
}

pub(crate) struct DsparkRuntime {
    model: DeepSeek4Weights,
    encoder_projection: QuantizedWeight,
    encoder_norm: Tensor,
    output_norm: Tensor,
    markov_w1: QuantizedWeight,
    markov_w2: QuantizedWeight,
    confidence_projection: QuantizedWeight,
    vocab_size: usize,
    target_layers: Vec<usize>,
    block_size: usize,
    mask_token_id: u32,
    sparse_moe_cuda_enabled: bool,
    committed_keys: Vec<VecDeque<Vec<f32>>>,
    position: usize,
}

impl DsparkRuntime {
    pub(in crate::engine) fn load(
        model: &LoadedModel,
        sparse_moe_cuda_enabled: bool,
        load_f32: F32WeightLoader,
        load_quantized: QuantizedWeightLoader,
    ) -> Result<Self> {
        if model.metadata.architecture != Architecture::DFlash {
            return Err(LlmError::ModelLoad(format!(
                "DSpark sidecar must use dflash architecture, got {:?}",
                model.metadata.architecture
            )));
        }
        let metadata = model.metadata.deepseek4.as_ref().ok_or_else(|| {
            LlmError::ModelLoad("DFlash sidecar has no DeepSeek4-compatible metadata".into())
        })?;
        let block_size = metadata
            .dspark_block_size
            .filter(|size| *size > 0)
            .ok_or_else(|| {
                LlmError::ModelLoad("DFlash sidecar has no positive dflash.block_size".into())
            })?;
        let mask_token_id = metadata.dspark_mask_token_id.ok_or_else(|| {
            LlmError::ModelLoad("DFlash sidecar has no tokenizer.ggml.mask_token_id".into())
        })?;
        if metadata.dspark_target_layers.is_empty() {
            return Err(LlmError::ModelLoad(
                "DFlash sidecar has no dflash.target_layers".into(),
            ));
        }
        super::dspark_contract::validate_dspark_weight_contract(model)?;

        let mut dspark =
            load_deepseek4_weights(model, load_f32, load_quantized).ok_or_else(|| {
                LlmError::ModelLoad("failed to load DSpark DeepSeek4 stage weights".into())
            })?;
        dspark.set_sparse_moe_cuda_enabled(sparse_moe_cuda_enabled);
        let committed_keys = (0..dspark.layers.len())
            .map(|_| VecDeque::with_capacity(model.metadata.sliding_window))
            .collect();

        Ok(Self {
            model: dspark,
            encoder_projection: load_quantized(model, "fc.weight"),
            encoder_norm: load_f32(model, "enc.output_norm.weight"),
            output_norm: load_f32(model, "output_norm.weight"),
            markov_w1: load_quantized(model, "markov_w1.weight"),
            markov_w2: load_quantized(model, "markov_w2.weight"),
            confidence_projection: load_quantized(model, "conf_proj.weight"),
            vocab_size: model.metadata.vocab_size,
            target_layers: metadata.dspark_target_layers.clone(),
            block_size,
            mask_token_id,
            committed_keys,
            sparse_moe_cuda_enabled,
            position: 0,
        })
    }

    pub(in crate::engine) fn validate_target(
        &self,
        target_architecture: Architecture,
        target: &crate::engine::types::ModelMetadata,
    ) -> Result<()> {
        if target_architecture != Architecture::DeepSeek4 {
            return Err(LlmError::ModelLoad(format!(
                "DSpark requires a DeepSeek4 target, got {target_architecture:?}"
            )));
        }
        if target.hidden_dim != self.model.config.hidden_dim {
            return Err(LlmError::ModelLoad(format!(
                "DSpark hidden {} != target hidden {}",
                self.model.config.hidden_dim, target.hidden_dim
            )));
        }
        if target.vocab_size != self.vocab_size {
            return Err(LlmError::ModelLoad(format!(
                "DSpark vocab {} != target vocab {}",
                self.vocab_size, target.vocab_size
            )));
        }
        if self
            .target_layers
            .iter()
            .any(|&layer| layer == 0 || layer > target.num_layers)
        {
            return Err(LlmError::ModelLoad(format!(
                "DSpark target layers {:?} exceed target trunk depth {}",
                self.target_layers, target.num_layers
            )));
        }
        Ok(())
    }

    pub(in crate::engine) fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    pub(in crate::engine) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(in crate::engine) fn sparse_moe_cuda_enabled(&self) -> bool {
        self.sparse_moe_cuda_enabled
    }

    pub(in crate::engine) fn clear(&mut self) {
        for keys in &mut self.committed_keys {
            keys.clear();
        }
        self.position = 0;
    }

    pub(in crate::engine) fn sequence_state_heap_byte_size_estimate(&self) -> u64 {
        self.committed_keys
            .iter()
            .flat_map(|keys| keys.iter())
            .map(|key| key.len() as u64 * std::mem::size_of::<f32>() as u64)
            .sum()
    }

    pub(in crate::engine) fn capture_sequence_state(&self) -> DsparkSequenceState {
        DsparkSequenceState {
            committed_keys: self.committed_keys.clone(),
            position: self.position,
        }
    }

    pub(in crate::engine) fn restore_sequence_state(
        &mut self,
        state: &DsparkSequenceState,
    ) -> Result<()> {
        if state.committed_keys.len() != self.model.layers.len() {
            return Err(LlmError::Forward(format!(
                "DSpark snapshot has {} stages, runtime has {}",
                state.committed_keys.len(),
                self.model.layers.len()
            )));
        }
        self.committed_keys.clone_from(&state.committed_keys);
        self.position = state.position;
        Ok(())
    }

    pub(in crate::engine) fn observe_target_batch(
        &mut self,
        features: &[f32],
        token_count: usize,
        start_position: usize,
    ) -> Result<()> {
        if start_position != self.position {
            return Err(LlmError::Forward(format!(
                "DSpark target feature position {start_position} != runtime position {}",
                self.position
            )));
        }
        if token_count == 0 {
            return Ok(());
        }
        let feature_dim = self.target_layers.len() * self.model.config.hidden_dim;
        if features.len() != token_count * feature_dim {
            return Err(LlmError::Forward(format!(
                "DSpark target feature length {} != {token_count} × {feature_dim}",
                features.len()
            )));
        }

        let projected = self.encoder_projection.gemv_vec(features)?;
        let fused = rms_norm_rows(
            &projected,
            token_count,
            self.model.config.hidden_dim,
            &self.encoder_norm,
            self.model.config.norm_eps,
        );
        for (stage_index, layer) in self.model.layers.iter().enumerate() {
            let projected_keys = layer.attention.kv.gemv_vec(&fused)?;
            let mut keys = rms_norm_rows(
                &projected_keys,
                token_count,
                self.model.config.head_dim,
                &layer.attention.kv_norm,
                self.model.config.norm_eps,
            );
            for (token, key) in keys
                .chunks_exact_mut(self.model.config.head_dim)
                .enumerate()
            {
                apply_key_rope(key, start_position + token, &self.model.config);
                self.committed_keys[stage_index].push_back(key.to_vec());
                while self.committed_keys[stage_index].len() > self.model.config.window_size {
                    self.committed_keys[stage_index].pop_front();
                }
            }
        }
        self.position += token_count;
        Ok(())
    }

    pub(in crate::engine) fn draft(
        &mut self,
        anchor_token: u32,
        target_embedding: &QuantizedWeight,
        target_output: &QuantizedWeight,
    ) -> Result<DsparkDraft> {
        let seq_len = self.block_size;
        let mut token_ids = vec![self.mask_token_id; seq_len];
        token_ids[0] = anchor_token;
        let embeddings = target_embedding.gather(&token_ids)?;
        let embeddings = kernels::tensor_as_f32_slice(&embeddings);
        let row_width = self.model.config.hc_count * self.model.config.hidden_dim;
        let mut hidden = vec![0.0f32; seq_len * row_width];
        for (token, embedding) in embeddings
            .chunks_exact(self.model.config.hidden_dim)
            .enumerate()
        {
            for copy in 0..self.model.config.hc_count {
                let start = token * row_width + copy * self.model.config.hidden_dim;
                hidden[start..start + self.model.config.hidden_dim].copy_from_slice(embedding);
            }
        }

        for (stage_index, layer) in self.model.layers.iter().enumerate() {
            let mut mixes = Vec::with_capacity(seq_len);
            let mut attention_inputs = Vec::with_capacity(seq_len * self.model.config.hidden_dim);
            for residual in hidden.chunks_exact(row_width) {
                let mix = hyper_pre(residual, &layer.attn_hc, &self.model.config);
                attention_inputs.extend(rms_norm(
                    &mix.branch,
                    &layer.attn_norm,
                    self.model.config.norm_eps,
                ));
                mixes.push(mix);
            }
            let attention_outputs = forward_noncausal_attention_block(
                &attention_inputs,
                self.position,
                &layer.attention,
                &self.committed_keys[stage_index],
                &self.model.config,
            )?;
            let mut after_attention = Vec::with_capacity(hidden.len());
            for ((residual, mix), output) in hidden
                .chunks_exact(row_width)
                .zip(mixes)
                .zip(attention_outputs.chunks_exact(self.model.config.hidden_dim))
            {
                after_attention.extend(hyper_post(output, residual, mix, &self.model.config));
            }
            hidden = after_attention;

            let mut mixes = Vec::with_capacity(seq_len);
            let mut ffn_inputs = Vec::with_capacity(seq_len * self.model.config.hidden_dim);
            for residual in hidden.chunks_exact(row_width) {
                let mix = hyper_pre(residual, &layer.ffn_hc, &self.model.config);
                ffn_inputs.extend(rms_norm(
                    &mix.branch,
                    &layer.ffn_norm,
                    self.model.config.norm_eps,
                ));
                mixes.push(mix);
            }
            let ffn_outputs =
                forward_moe_batch(&ffn_inputs, &token_ids, &layer.moe, &self.model.config)?;
            let mut after_ffn = Vec::with_capacity(hidden.len());
            for ((residual, mix), output) in hidden
                .chunks_exact(row_width)
                .zip(mixes)
                .zip(ffn_outputs.chunks_exact(self.model.config.hidden_dim))
            {
                after_ffn.extend(hyper_post(output, residual, mix, &self.model.config));
            }
            hidden = after_ffn;
        }

        let mut collapsed = Vec::with_capacity(seq_len * self.model.config.hidden_dim);
        for row in hidden.chunks_exact(row_width) {
            collapsed.extend(hyper_head(
                row,
                &self.model.output_hc_function,
                &self.model.output_hc_scale,
                &self.model.output_hc_base,
                &self.model.config,
            ));
        }
        let normalized = rms_norm_rows(
            &collapsed,
            seq_len,
            self.model.config.hidden_dim,
            &self.output_norm,
            self.model.config.norm_eps,
        );
        let base_logits = target_output.gemv_vec(&normalized)?;
        let vocab_size = target_output.rows;
        if base_logits.len() != seq_len * vocab_size {
            return Err(LlmError::Forward(format!(
                "DSpark base logits length {} != {seq_len} × {vocab_size}",
                base_logits.len()
            )));
        }

        let mut previous = anchor_token;
        let mut tokens = Vec::with_capacity(seq_len);
        let mut confidences = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let markov = self.markov_w1.gather(&[previous])?;
            let markov = kernels::tensor_as_f32_slice(&markov);
            let bias = self.markov_w2.gemv_vec(markov)?;
            let base = &base_logits[position * vocab_size..(position + 1) * vocab_size];
            let token = base
                .iter()
                .zip(&bias)
                .enumerate()
                .max_by(
                    |(_, (left_base, left_bias)), (_, (right_base, right_bias))| {
                        (*left_base + *left_bias)
                            .partial_cmp(&(*right_base + *right_bias))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    },
                )
                .map(|(index, _)| index as u32)
                .ok_or_else(|| LlmError::Forward("DSpark produced empty logits".into()))?;

            let mut confidence_input =
                Vec::with_capacity(self.model.config.hidden_dim + markov.len());
            confidence_input.extend_from_slice(
                &collapsed[position * self.model.config.hidden_dim
                    ..(position + 1) * self.model.config.hidden_dim],
            );
            confidence_input.extend_from_slice(markov);
            let confidence = self.confidence_projection.gemv_vec(&confidence_input)?;
            confidences.push(1.0 / (1.0 + (-confidence[0]).exp()));
            tokens.push(token);
            previous = token;
        }

        Ok(DsparkDraft {
            tokens,
            confidences,
        })
    }
}

fn rms_norm_rows(input: &[f32], rows: usize, cols: usize, weight: &Tensor, eps: f32) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * cols);
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(cols) {
        output.extend(rms_norm(row, weight, eps));
    }
    output
}

fn apply_key_rope(key: &mut [f32], position: usize, config: &DeepSeek4Config) {
    let rope_start = config.head_dim - config.rope_dim;
    apply_rope(&mut key[rope_start..], position, config, false, false);
}

fn forward_noncausal_attention_block(
    inputs: &[f32],
    start_position: usize,
    weights: &super::weights::AttentionWeights,
    committed: &VecDeque<Vec<f32>>,
    config: &DeepSeek4Config,
) -> Result<Vec<f32>> {
    let seq_len = inputs.len() / config.hidden_dim;
    let q_rank = weights.q_a.rows;
    let q_a = weights.q_a.gemv_vec(inputs)?;
    let qr = rms_norm_rows(&q_a, seq_len, q_rank, &weights.q_a_norm, config.norm_eps);
    let mut queries = weights.q_b.gemv_vec(&qr)?;
    let projected_keys = weights.kv.gemv_vec(inputs)?;
    let mut block_keys = rms_norm_rows(
        &projected_keys,
        seq_len,
        config.head_dim,
        &weights.kv_norm,
        config.norm_eps,
    );
    for (token, key) in block_keys.chunks_exact_mut(config.head_dim).enumerate() {
        apply_key_rope(key, start_position + token, config);
    }
    for (token, query) in queries
        .chunks_exact_mut(config.num_heads * config.head_dim)
        .enumerate()
    {
        for head in query.chunks_exact_mut(config.head_dim) {
            rms_unit_inplace(head, config.norm_eps);
            let rope_start = config.head_dim - config.rope_dim;
            apply_rope(
                &mut head[rope_start..],
                start_position + token,
                config,
                false,
                false,
            );
        }
    }

    let prior_count = config
        .window_size
        .saturating_sub(seq_len)
        .min(committed.len());
    let prior_start = committed.len() - prior_count;
    let prior = committed.iter().skip(prior_start).collect::<Vec<_>>();
    let current = block_keys.chunks_exact(config.head_dim).collect::<Vec<_>>();
    let sinks = tensor_f32(&weights.sinks);
    let scale = (config.head_dim as f32).sqrt().recip();
    let rope_start = config.head_dim - config.rope_dim;
    let mut attention = vec![0.0f32; seq_len * config.num_heads * config.head_dim];
    attention
        .par_chunks_mut(config.num_heads * config.head_dim)
        .enumerate()
        .for_each(|(token, token_output)| {
            let token_query = &queries[token * config.num_heads * config.head_dim
                ..(token + 1) * config.num_heads * config.head_dim];
            for (head_index, output) in token_output.chunks_exact_mut(config.head_dim).enumerate() {
                let query =
                    &token_query[head_index * config.head_dim..(head_index + 1) * config.head_dim];
                let mut scores = prior
                    .iter()
                    .map(|key| dot(query, key) * scale)
                    .chain(current.iter().map(|key| dot(query, key) * scale))
                    .collect::<Vec<_>>();
                let max_score = scores.iter().copied().fold(sinks[head_index], f32::max);
                let mut denominator = (sinks[head_index] - max_score).exp();
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }
                for (probability, key) in scores.iter().map(|score| *score / denominator).zip(
                    prior
                        .iter()
                        .map(|key| key.as_slice())
                        .chain(current.iter().copied()),
                ) {
                    for (dst, &value) in output.iter_mut().zip(key) {
                        *dst += probability * value;
                    }
                }
                apply_rope(
                    &mut output[rope_start..],
                    start_position + token,
                    config,
                    false,
                    true,
                );
            }
        });

    let mut output = Vec::with_capacity(seq_len * config.hidden_dim);
    for row in attention.chunks_exact(config.num_heads * config.head_dim) {
        output.extend(project_attention_output(row, weights, config)?);
    }
    Ok(output)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}
