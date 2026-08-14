use std::collections::VecDeque;

use rnb_core::tensor::Tensor;
use rnb_loader::{Architecture, GGMLType, LoadedModel};

use crate::engine::cpu_runtime::kernels;
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::error::{LlmError, Result};

use super::super::layer_weights::AttentionLayerWeights;

const SUPPORTED_TARGET_LAYERS: &[usize] = &[2, 14, 26, 38, 50];
const SUPPORTED_BLOCK_SIZE: usize = 16;

pub(crate) struct MuseDflashDraft {
    pub(crate) tokens: Vec<u32>,
    pub(crate) probabilities: Vec<f32>,
}

#[derive(Clone)]
struct DflashLayerCache {
    keys: VecDeque<Vec<u16>>,
    values: VecDeque<Vec<u16>>,
}

#[derive(Clone)]
pub(crate) struct MuseDflashSequenceState {
    layers: Vec<DflashLayerCache>,
    position: usize,
}

#[derive(Clone)]
pub(crate) struct MuseDflashCheckpoint {
    position: usize,
    layer_lengths: Vec<usize>,
    front_keys: Vec<Vec<Vec<u16>>>,
    front_values: Vec<Vec<Vec<u16>>>,
}

impl MuseDflashSequenceState {
    pub(in crate::engine) fn heap_byte_size(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|layer| layer.keys.iter().chain(layer.values.iter()))
            .map(|row| row.capacity().saturating_mul(std::mem::size_of::<u16>()) as u64)
            .sum()
    }
}

pub(crate) struct MuseDflashRuntime {
    encoder_projection: QuantizedWeight,
    encoder_norm: Tensor,
    output_norm: Tensor,
    layers: Vec<AttentionLayerWeights>,
    layer_cache: Vec<DflashLayerCache>,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    window_size: usize,
    rope_theta: f32,
    norm_eps: f32,
    target_layers: Vec<usize>,
    block_size: usize,
    mask_token_id: u32,
    position: usize,
}

impl MuseDflashRuntime {
    pub(in crate::engine) fn load(
        model: &LoadedModel,
        load_f32: fn(&LoadedModel, &str) -> Tensor,
        load_quantized: fn(&LoadedModel, &str) -> QuantizedWeight,
        load_layer: fn(&LoadedModel, usize) -> AttentionLayerWeights,
    ) -> Result<Self> {
        validate_weight_contract(model)?;
        let dflash = model
            .metadata
            .dflash
            .as_ref()
            .expect("validated DFlash metadata");
        let layers = (0..model.metadata.num_layers)
            .map(|layer| load_layer(model, layer))
            .collect::<Vec<_>>();
        let layer_cache = (0..model.metadata.num_layers)
            .map(|_| DflashLayerCache {
                keys: VecDeque::with_capacity(model.metadata.sliding_window),
                values: VecDeque::with_capacity(model.metadata.sliding_window),
            })
            .collect();
        Ok(Self {
            encoder_projection: load_quantized(model, "fc.weight"),
            encoder_norm: load_f32(model, "enc.output_norm.weight"),
            output_norm: load_f32(model, "output_norm.weight"),
            layers,
            layer_cache,
            hidden_dim: model.metadata.hidden_size,
            num_heads: model.metadata.num_heads,
            num_kv_heads: model.metadata.num_kv_heads,
            head_dim: model.metadata.head_dim,
            vocab_size: model.metadata.vocab_size,
            max_seq_len: model.metadata.max_seq_len,
            window_size: model.metadata.sliding_window,
            rope_theta: model.metadata.rope_theta,
            norm_eps: model.metadata.norm_eps,
            target_layers: dflash.target_layers.clone(),
            block_size: dflash.block_size,
            mask_token_id: dflash.mask_token_id,
            position: 0,
        })
    }

    pub(in crate::engine) fn validate_target(
        &self,
        architecture: Architecture,
        metadata: &crate::engine::types::ModelMetadata,
    ) -> Result<()> {
        if architecture != Architecture::MuseGlimmer {
            return Err(LlmError::ModelLoad(format!(
                "Muse DFlash requires a MuseGlimmer target, got {architecture:?}"
            )));
        }
        if metadata.num_layers != 52
            || metadata.hidden_dim != 6656
            || metadata.vocab_size != 202048
            || metadata.num_heads != 32
            || metadata.num_kv_heads != 2
            || metadata.head_dim != 128
            || self.target_layers != SUPPORTED_TARGET_LAYERS
        {
            return Err(LlmError::ModelLoad(format!(
                "Muse DFlash target mismatch: layers={} hidden={} vocab={} heads={}/{} head_dim={}, sidecar target_layers={:?}",
                metadata.num_layers,
                metadata.hidden_dim,
                metadata.vocab_size,
                metadata.num_heads,
                metadata.num_kv_heads,
                metadata.head_dim,
                self.target_layers,
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

    pub(in crate::engine) fn window_size(&self) -> usize {
        self.window_size
    }

    pub(in crate::engine) fn clear(&mut self) {
        for layer in &mut self.layer_cache {
            layer.keys.clear();
            layer.values.clear();
        }
        self.position = 0;
    }

    pub(in crate::engine) fn checkpoint(&self) -> MuseDflashCheckpoint {
        MuseDflashCheckpoint {
            position: self.position,
            layer_lengths: self
                .layer_cache
                .iter()
                .map(|layer| layer.keys.len())
                .collect(),
            front_keys: self
                .layer_cache
                .iter()
                .map(|layer| layer.keys.iter().take(self.block_size).cloned().collect())
                .collect(),
            front_values: self
                .layer_cache
                .iter()
                .map(|layer| layer.values.iter().take(self.block_size).cloned().collect())
                .collect(),
        }
    }

    pub(in crate::engine) fn restore_checkpoint(
        &mut self,
        checkpoint: &MuseDflashCheckpoint,
    ) -> Result<()> {
        let appended = self
            .position
            .checked_sub(checkpoint.position)
            .ok_or_else(|| {
                LlmError::Forward("Muse DFlash checkpoint is ahead of runtime".to_string())
            })?;
        if appended > self.block_size
            || checkpoint.layer_lengths.len() != self.layer_cache.len()
            || checkpoint.front_keys.len() != self.layer_cache.len()
            || checkpoint.front_values.len() != self.layer_cache.len()
        {
            return Err(LlmError::Forward(
                "Muse DFlash checkpoint does not match runtime".to_string(),
            ));
        }
        for (layer_index, layer) in self.layer_cache.iter_mut().enumerate() {
            for _ in 0..appended.min(layer.keys.len()) {
                layer.keys.pop_back();
                layer.values.pop_back();
            }
            let original_len = checkpoint.layer_lengths[layer_index];
            let evicted = appended.saturating_sub(self.window_size.saturating_sub(original_len));
            if evicted > checkpoint.front_keys[layer_index].len()
                || evicted > checkpoint.front_values[layer_index].len()
            {
                return Err(LlmError::Forward(
                    "Muse DFlash checkpoint did not retain enough evicted rows".to_string(),
                ));
            }
            for row in checkpoint.front_keys[layer_index][..evicted].iter().rev() {
                layer.keys.push_front(row.clone());
            }
            for row in checkpoint.front_values[layer_index][..evicted].iter().rev() {
                layer.values.push_front(row.clone());
            }
            if layer.keys.len() != original_len || layer.values.len() != original_len {
                return Err(LlmError::Forward(
                    "Muse DFlash checkpoint restored an invalid cache length".to_string(),
                ));
            }
        }
        self.position = checkpoint.position;
        Ok(())
    }

    pub(in crate::engine) fn capture_sequence_state(&self) -> MuseDflashSequenceState {
        MuseDflashSequenceState {
            layers: self.layer_cache.clone(),
            position: self.position,
        }
    }

    pub(in crate::engine) fn sequence_state_heap_byte_size_estimate(&self) -> u64 {
        self.capture_sequence_state().heap_byte_size()
    }

    pub(in crate::engine) fn restore_sequence_state(
        &mut self,
        state: &MuseDflashSequenceState,
    ) -> Result<()> {
        if state.layers.len() != self.layers.len()
            || state.layers.iter().any(|layer| {
                layer.keys.len() != layer.values.len() || layer.keys.len() > self.window_size
            })
        {
            return Err(LlmError::Forward(
                "Muse DFlash sequence state does not match runtime".to_string(),
            ));
        }
        self.layer_cache.clone_from(&state.layers);
        self.position = state.position;
        Ok(())
    }

    pub(in crate::engine) fn observe_target_batch(
        &mut self,
        features: &[f32],
        token_count: usize,
        start_position: usize,
    ) -> Result<()> {
        let feature_dim = self.target_layers.len() * self.hidden_dim;
        if features.len() != token_count * feature_dim {
            return Err(LlmError::Forward(format!(
                "Muse DFlash target feature length {} != {} tokens * {}",
                features.len(),
                token_count,
                feature_dim
            )));
        }
        if start_position != self.position {
            return Err(LlmError::Forward(format!(
                "Muse DFlash target position {start_position} != draft position {}",
                self.position
            )));
        }
        if start_position + token_count > self.max_seq_len {
            return Err(LlmError::Forward(
                "Muse DFlash KV cache overflow".to_string(),
            ));
        }

        let encoder_norm = kernels::tensor_as_f32_slice(&self.encoder_norm);
        if let Some(layer_kv) = crate::engine::backend_runtime::dflash_cache_seed_if_supported(
            &self.encoder_projection,
            encoder_norm,
            &self.layers,
            features,
            token_count,
            start_position,
            self.hidden_dim,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.norm_eps,
        )? {
            let kv_dim = self.num_kv_heads * self.head_dim;
            if layer_kv.len() != self.layer_cache.len()
                || layer_kv.iter().any(|(keys, values)| {
                    keys.len() != token_count * kv_dim || values.len() != token_count * kv_dim
                })
            {
                return Err(LlmError::Forward(
                    "Metal DFlash cache seed returned an invalid shape".to_string(),
                ));
            }
            for (cache, (keys, values)) in self.layer_cache.iter_mut().zip(layer_kv) {
                for token in 0..token_count {
                    let row = token * kv_dim;
                    cache.keys.push_back(keys[row..row + kv_dim].to_vec());
                    cache.values.push_back(values[row..row + kv_dim].to_vec());
                    if cache.keys.len() > self.window_size {
                        cache.keys.pop_front();
                        cache.values.pop_front();
                    }
                }
            }
            self.position = start_position + token_count;
            return Ok(());
        }

        let projected = self.encoder_projection.gemv_vec(features)?;
        let fused = rms_norm_rows(
            &projected,
            self.hidden_dim,
            &self.encoder_norm,
            self.norm_eps,
        );
        let kv_dim = self.num_kv_heads * self.head_dim;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut keys = layer.k_weight.gemv_vec(&fused)?;
            let values = layer.v_weight.gemv_vec(&fused)?;
            apply_qk_norm_rows(
                &mut keys,
                token_count,
                self.num_kv_heads,
                self.head_dim,
                layer.k_norm.as_ref().expect("validated DFlash k_norm"),
                self.norm_eps,
            );
            kernels::rope::rope_neox_inplace(
                &mut keys,
                start_position,
                self.head_dim,
                kv_dim,
                self.rope_theta,
            );
            let cache = &mut self.layer_cache[layer_index];
            for token in 0..token_count {
                let row = token * kv_dim;
                cache.keys.push_back(
                    keys[row..row + kv_dim]
                        .iter()
                        .map(|value| half::f16::from_f32(*value).to_bits())
                        .collect(),
                );
                cache.values.push_back(
                    values[row..row + kv_dim]
                        .iter()
                        .map(|value| half::f16::from_f32(*value).to_bits())
                        .collect(),
                );
                if cache.keys.len() > self.window_size {
                    cache.keys.pop_front();
                    cache.values.pop_front();
                }
            }
        }
        self.position = start_position + token_count;
        Ok(())
    }

    pub(in crate::engine) fn retain_verified_prefix(
        &mut self,
        checkpoint: &MuseDflashCheckpoint,
        features: &[f32],
        token_count: usize,
    ) -> Result<()> {
        self.restore_checkpoint(checkpoint)?;
        let position = self.position;
        self.observe_target_batch(features, token_count, position)
    }

    pub(in crate::engine) fn draft(
        &mut self,
        anchor_token: u32,
        max_draft_tokens: usize,
        confidence_cutoff: Option<f32>,
        target_embedding: &QuantizedWeight,
        target_output: &QuantizedWeight,
    ) -> Result<MuseDflashDraft> {
        if max_draft_tokens == 0 || max_draft_tokens >= self.block_size {
            return Err(LlmError::Forward(format!(
                "Muse DFlash draft size {max_draft_tokens} must be in 1..{}",
                self.block_size
            )));
        }
        let seq_len = self.block_size;
        let mut tokens = vec![self.mask_token_id; seq_len];
        tokens[0] = anchor_token;
        let hidden = target_embedding.gather(&tokens)?;
        let mut hidden = kernels::tensor_as_f32_slice(&hidden).to_vec();
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let cache = &self.layer_cache[layer_index];
            let prior_count = self.window_size.saturating_sub(1).min(cache.keys.len());
            let prior_k = cache
                .keys
                .iter()
                .skip(cache.keys.len() - prior_count)
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let prior_v = cache
                .values
                .iter()
                .skip(cache.values.len() - prior_count)
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            if crate::engine::backend_runtime::dflash_q4k_layer_chain_if_supported(
                &layer.q_weight,
                &layer.k_weight,
                &layer.v_weight,
                &layer.o_weight,
                &layer.ffn_gate_weight,
                &layer.ffn_up_weight,
                &layer.ffn_down_weight,
                layer_index,
                self.layers.len(),
                &prior_k,
                &prior_v,
                &mut hidden,
                kernels::tensor_as_f32_slice(&layer.attn_norm),
                kernels::tensor_as_f32_slice(
                    layer.q_norm.as_ref().expect("validated DFlash q_norm"),
                ),
                kernels::tensor_as_f32_slice(
                    layer.k_norm.as_ref().expect("validated DFlash k_norm"),
                ),
                kernels::tensor_as_f32_slice(&layer.ffn_norm),
                self.num_heads,
                self.num_kv_heads,
                (self.head_dim as f32).sqrt().recip(),
                self.rope_theta,
                self.position,
                self.window_size,
                layer.ffn_gate_weight.rows,
                self.hidden_dim,
                self.norm_eps,
            )? {
                continue;
            }

            let normalized =
                rms_norm_rows(&hidden, self.hidden_dim, &layer.attn_norm, self.norm_eps);
            let mut query = layer.q_weight.gemv_vec(&normalized)?;
            let mut key = layer.k_weight.gemv_vec(&normalized)?;
            let value = layer.v_weight.gemv_vec(&normalized)?;
            apply_qk_norm_rows(
                &mut query,
                seq_len,
                self.num_heads,
                self.head_dim,
                layer.q_norm.as_ref().expect("validated DFlash q_norm"),
                self.norm_eps,
            );
            apply_qk_norm_rows(
                &mut key,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
                layer.k_norm.as_ref().expect("validated DFlash k_norm"),
                self.norm_eps,
            );
            kernels::rope::rope_neox_inplace(
                &mut query,
                self.position,
                self.head_dim,
                q_dim,
                self.rope_theta,
            );
            kernels::rope::rope_neox_inplace(
                &mut key,
                self.position,
                self.head_dim,
                kv_dim,
                self.rope_theta,
            );

            let attention = if let Some(attention) =
                crate::engine::backend_runtime::metal_dflash_attention_if_supported(
                    &query,
                    &prior_k,
                    &prior_v,
                    &key,
                    &value,
                    seq_len,
                    self.position,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.window_size,
                )? {
                attention
            } else {
                let mut all_k = Vec::with_capacity((prior_count + seq_len) * kv_dim);
                let mut all_v = Vec::with_capacity((prior_count + seq_len) * kv_dim);
                all_k.extend(
                    prior_k
                        .iter()
                        .map(|bits| half::f16::from_bits(*bits).to_f32()),
                );
                all_v.extend(
                    prior_v
                        .iter()
                        .map(|bits| half::f16::from_bits(*bits).to_f32()),
                );
                all_k.extend_from_slice(&key);
                all_v.extend_from_slice(&value);
                let kv_len = prior_count + seq_len;
                crate::engine::backend_runtime::prefill_attention_non_causal_if_supported(
                    &query,
                    &all_k,
                    &all_v,
                    seq_len,
                    kv_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    (self.head_dim as f32).sqrt().recip(),
                    Some(self.window_size),
                )?
                .unwrap_or_else(|| {
                    noncausal_attention(
                        &query,
                        &all_k,
                        &all_v,
                        seq_len,
                        kv_len,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        Some(self.window_size),
                    )
                })
            };
            let projected = layer.o_weight.gemv_vec(&attention)?;
            add_inplace(&mut hidden, &projected)?;

            let normalized =
                rms_norm_rows(&hidden, self.hidden_dim, &layer.ffn_norm, self.norm_eps);
            let mut gate = layer.ffn_gate_weight.gemv_vec(&normalized)?;
            let up = layer.ffn_up_weight.gemv_vec(&normalized)?;
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = (*gate / (1.0 + (-*gate).exp())) * up;
            }
            let ffn = layer.ffn_down_weight.gemv_vec(&gate)?;
            add_inplace(&mut hidden, &ffn)?;
        }

        let normalized = rms_norm_rows(&hidden, self.hidden_dim, &self.output_norm, self.norm_eps);
        let logits = target_output
            .gemv_vec(&normalized[self.hidden_dim..(max_draft_tokens + 1) * self.hidden_dim])?;
        let expected_logits = max_draft_tokens * self.vocab_size;
        if logits.len() != expected_logits {
            return Err(LlmError::Forward(format!(
                "Muse DFlash logits length {} != {}",
                logits.len(),
                expected_logits
            )));
        }
        let collect_probabilities =
            confidence_cutoff.is_some() || crate::runtime::mtp_trace_enabled();
        let mut result_tokens = Vec::with_capacity(max_draft_tokens);
        let mut probabilities = Vec::with_capacity(
            collect_probabilities
                .then_some(max_draft_tokens)
                .unwrap_or_default(),
        );
        for row in logits.chunks_exact(self.vocab_size) {
            let (token, max_logit) = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap_or((0, f32::NEG_INFINITY));
            result_tokens.push(token as u32);
            if collect_probabilities {
                let sum = row
                    .iter()
                    .map(|value| (*value - max_logit).exp())
                    .sum::<f32>();
                let probability = sum.recip();
                probabilities.push(probability);
                if confidence_cutoff.is_some_and(|cutoff| probability < cutoff) {
                    break;
                }
            }
        }
        Ok(MuseDflashDraft {
            tokens: result_tokens,
            probabilities,
        })
    }
}

fn validate_weight_contract(model: &LoadedModel) -> Result<()> {
    if model.metadata.architecture != Architecture::DFlash {
        return Err(LlmError::ModelLoad(format!(
            "Muse DFlash sidecar must use dflash architecture, got {:?}",
            model.metadata.architecture
        )));
    }
    if model.metadata.deepseek4.is_some() {
        return Err(LlmError::ModelLoad(
            "DeepSeek-compatible DFlash sidecar is not Muse DFlash".to_string(),
        ));
    }
    let dflash =
        model.metadata.dflash.as_ref().ok_or_else(|| {
            LlmError::ModelLoad("DFlash sidecar has no generic metadata".to_string())
        })?;
    if dflash.block_size != SUPPORTED_BLOCK_SIZE || dflash.target_layers != SUPPORTED_TARGET_LAYERS
    {
        return Err(LlmError::ModelLoad(format!(
            "Muse DFlash contract mismatch: block_size={} target_layers={:?}",
            dflash.block_size, dflash.target_layers
        )));
    }
    if model.metadata.num_layers != 5
        || model.metadata.hidden_size != 6656
        || model.metadata.intermediate_size != 19968
        || model.metadata.num_heads != 32
        || model.metadata.num_kv_heads != 8
        || model.metadata.head_dim != 128
        || model.metadata.sliding_window != 2048
    {
        return Err(LlmError::ModelLoad(format!(
            "unsupported Muse DFlash shape: layers={} hidden={} ffn={} heads={}/{} head_dim={} window={}",
            model.metadata.num_layers,
            model.metadata.hidden_size,
            model.metadata.intermediate_size,
            model.metadata.num_heads,
            model.metadata.num_kv_heads,
            model.metadata.head_dim,
            model.metadata.sliding_window
        )));
    }
    require_matrix(model, "fc.weight", 6656, 5 * 6656)?;
    require_numel(model, "enc.output_norm.weight", 6656)?;
    require_numel(model, "output_norm.weight", 6656)?;
    for layer in 0..5 {
        require_matrix(model, &format!("blk.{layer}.attn_q.weight"), 4096, 6656)?;
        require_matrix(model, &format!("blk.{layer}.attn_k.weight"), 1024, 6656)?;
        require_matrix(model, &format!("blk.{layer}.attn_v.weight"), 1024, 6656)?;
        require_matrix(
            model,
            &format!("blk.{layer}.attn_output.weight"),
            6656,
            4096,
        )?;
        require_matrix(model, &format!("blk.{layer}.ffn_gate.weight"), 19968, 6656)?;
        require_matrix(model, &format!("blk.{layer}.ffn_up.weight"), 19968, 6656)?;
        require_matrix(model, &format!("blk.{layer}.ffn_down.weight"), 6656, 19968)?;
        for name in ["attn_norm", "ffn_norm"] {
            require_numel(model, &format!("blk.{layer}.{name}.weight"), 6656)?;
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            require_numel(model, &format!("blk.{layer}.{name}.weight"), 128)?;
        }
    }
    Ok(())
}

fn require_matrix(model: &LoadedModel, name: &str, rows: usize, cols: usize) -> Result<()> {
    let shape = model
        .float_shapes
        .get(name)
        .ok_or_else(|| LlmError::ModelLoad(format!("Muse DFlash missing matrix {name}")))?;
    if shape.as_slice() != [rows, cols] {
        return Err(LlmError::ModelLoad(format!(
            "Muse DFlash {name} shape {shape:?} != [{rows}, {cols}]"
        )));
    }
    if !model.tensor_ggml_types.contains_key(name) {
        return Err(LlmError::ModelLoad(format!(
            "Muse DFlash missing type for {name}"
        )));
    }
    Ok(())
}

fn require_numel(model: &LoadedModel, name: &str, expected: usize) -> Result<()> {
    let tensor = model
        .weights
        .get(name)
        .ok_or_else(|| LlmError::ModelLoad(format!("Muse DFlash missing tensor {name}")))?;
    if tensor.numel() != expected || model.tensor_ggml_types.get(name) != Some(&GGMLType::F32) {
        return Err(LlmError::ModelLoad(format!(
            "Muse DFlash {name} must be F32[{expected}]"
        )));
    }
    Ok(())
}

fn rms_norm_rows(input: &[f32], width: usize, weight: &Tensor, eps: f32) -> Vec<f32> {
    let weight = kernels::tensor_as_f32_slice(weight);
    let mut output = vec![0.0; input.len()];
    for (src, dst) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        kernels::norm::rms_norm_into(src, weight, eps, dst);
    }
    output
}

fn apply_qk_norm_rows(
    values: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    weight: &Tensor,
    eps: f32,
) {
    let weight = kernels::tensor_as_f32_slice(weight);
    debug_assert_eq!(values.len(), tokens * heads * head_dim);
    for head in values.chunks_exact_mut(head_dim) {
        let input = head.to_vec();
        kernels::norm::rms_norm_into(&input, weight, eps, head);
    }
}

#[allow(clippy::too_many_arguments)]
fn noncausal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; seq_len * num_heads * head_dim];
    for token in 0..seq_len {
        for head in 0..num_heads {
            let kv_head = head / heads_per_kv;
            let q_start = (token * num_heads + head) * head_dim;
            let query = &q[q_start..q_start + head_dim];
            let global_position = kv_len - seq_len + token;
            let first_key = sliding_window
                .map(|window| (global_position + 1).saturating_sub(window))
                .unwrap_or(0);
            let mut scores = Vec::with_capacity(kv_len - first_key);
            let mut max_score = f32::NEG_INFINITY;
            for position in first_key..kv_len {
                let k_start = (position * num_kv_heads + kv_head) * head_dim;
                let score = query
                    .iter()
                    .zip(&k[k_start..k_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale;
                max_score = max_score.max(score);
                scores.push(score);
            }
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - max_score).exp();
                    *score
                })
                .sum::<f32>();
            let out_start = (token * num_heads + head) * head_dim;
            for (score_index, probability) in scores.into_iter().enumerate() {
                let position = first_key + score_index;
                let v_start = (position * num_kv_heads + kv_head) * head_dim;
                for offset in 0..head_dim {
                    output[out_start + offset] += probability / denominator * v[v_start + offset];
                }
            }
        }
    }
    output
}

fn add_inplace(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(LlmError::Forward(format!(
            "Muse DFlash residual length {} != {}",
            left.len(),
            right.len()
        )));
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left += *right;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::noncausal_attention;

    #[test]
    fn noncausal_attention_applies_standard_swa_per_query() {
        let output = noncausal_attention(
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 2.0, 4.0, 8.0],
            2,
            4,
            1,
            1,
            1,
            Some(2),
        );

        assert!((output[0] - 14.0 / 3.0).abs() < 1e-6);
        assert!((output[1] - 6.0).abs() < 1e-6);
    }
}
