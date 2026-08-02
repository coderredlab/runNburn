use super::state::DeepSeek4State;
use crate::engine::models::shared_expert_moe::{
    load_shared_expert_moe_layer, SharedExpertMoELayerWeights,
};
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::convert::ggml_quant_params;
use rnb_loader::{DeepSeek4Metadata, GGMLType, LoadedModel};

pub(super) type F32WeightLoader = fn(&LoadedModel, &str) -> Tensor;
pub(super) type QuantizedWeightLoader = fn(&LoadedModel, &str) -> QuantizedWeight;

pub(in crate::engine) struct DeepSeek4Weights {
    pub(super) config: DeepSeek4Config,
    pub(super) layers: Vec<DeepSeek4LayerWeights>,
    pub(super) output_hc_function: Tensor,
    pub(super) output_hc_scale: Tensor,
    pub(super) output_hc_base: Tensor,
    pub(super) state: DeepSeek4State,
}

impl DeepSeek4Weights {
    pub(in crate::engine) fn clear_state(&mut self) {
        self.state.clear();
    }
    pub(in crate::engine) fn set_sparse_moe_cuda_enabled(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.moe.prefer_sparse_moe_cuda = enabled;
        }
    }
}

pub(super) struct DeepSeek4Config {
    pub(super) hidden_dim: usize,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dim: usize,
    pub(super) output_groups: usize,
    pub(super) output_lora_rank: usize,
    pub(super) window_size: usize,
    pub(super) index_heads: usize,
    pub(super) index_head_dim: usize,
    pub(super) index_topk: usize,
    pub(super) hc_count: usize,
    pub(super) sinkhorn_iterations: usize,
    pub(super) hc_eps: f32,
    pub(super) norm_eps: f32,
    pub(super) expert_count: usize,
    pub(super) expert_used_count: usize,
    pub(super) expert_ffn_dim: usize,
    pub(super) expert_scale: f32,
    pub(super) rope_theta: f32,
    pub(super) compress_rope_theta: f32,
    pub(super) rope_factor: f32,
    pub(super) rope_original_context_length: usize,
    pub(super) rope_yarn_beta_fast: f32,
    pub(super) rope_yarn_beta_slow: f32,
}

pub(super) struct HyperConnectionWeights {
    pub(super) function: Tensor,
    pub(super) scale: Tensor,
    pub(super) base: Tensor,
}

pub(super) struct DeepSeek4LayerWeights {
    pub(super) attn_norm: Tensor,
    pub(super) attention: AttentionWeights,
    pub(super) attn_hc: HyperConnectionWeights,
    pub(super) ffn_norm: Tensor,
    pub(super) moe: DeepSeek4MoeWeights,
    pub(super) ffn_hc: HyperConnectionWeights,
}

pub(super) struct AttentionWeights {
    pub(super) q_a: QuantizedWeight,
    pub(super) q_a_norm: Tensor,
    pub(super) q_b: QuantizedWeight,
    pub(super) kv: QuantizedWeight,
    pub(super) kv_norm: Tensor,
    pub(super) sinks: Tensor,
    pub(super) output_a_groups: Vec<QuantizedWeight>,
    pub(super) output_b: QuantizedWeight,
    pub(super) compressor: Option<CompressorWeights>,
    pub(super) indexer: Option<IndexerWeights>,
}

pub(super) struct CompressorWeights {
    pub(super) ratio: usize,
    pub(super) head_dim: usize,
    pub(super) ape: Tensor,
    pub(super) kv: QuantizedWeight,
    pub(super) gate: QuantizedWeight,
    pub(super) norm: Tensor,
}

pub(super) struct IndexerWeights {
    pub(super) q_b: QuantizedWeight,
    pub(super) projection: Tensor,
    pub(super) compressor: CompressorWeights,
}

pub(super) struct DeepSeek4MoeWeights {
    pub(super) weights: SharedExpertMoELayerWeights,
    pub(super) hash_routes: Option<Vec<i32>>,
    pub(super) routed_clamp: f32,
    pub(super) shared_clamp: f32,
    pub(super) prefer_sparse_moe_cuda: bool,
}

pub(in crate::engine) fn load_deepseek4_weights(
    model: &LoadedModel,
    load_f32: F32WeightLoader,
    load_quantized: QuantizedWeightLoader,
) -> Option<DeepSeek4Weights> {
    let metadata = model.metadata.deepseek4.as_ref()?;
    let config = build_config(model, metadata);
    let mut layers = Vec::with_capacity(model.metadata.num_layers);
    for layer_index in 0..model.metadata.num_layers {
        let prefix = format!("blk.{layer_index}");
        let ratio = metadata.compress_ratios[layer_index];
        let compressor = (ratio > 0).then(|| CompressorWeights {
            ratio,
            head_dim: config.head_dim,
            ape: load_f32(model, &format!("{prefix}.attn_compressor_ape.weight")),
            kv: load_quantized(model, &format!("{prefix}.attn_compressor_kv.weight")),
            gate: load_quantized(model, &format!("{prefix}.attn_compressor_gate.weight")),
            norm: load_f32(model, &format!("{prefix}.attn_compressor_norm.weight")),
        });
        let indexer = (ratio == 4).then(|| IndexerWeights {
            q_b: load_quantized(model, &format!("{prefix}.indexer.attn_q_b.weight")),
            projection: load_f32(model, &format!("{prefix}.indexer.proj.weight")),
            compressor: CompressorWeights {
                ratio,
                head_dim: config.index_head_dim,
                ape: load_f32(model, &format!("{prefix}.indexer_compressor_ape.weight")),
                kv: load_quantized(model, &format!("{prefix}.indexer_compressor_kv.weight")),
                gate: load_quantized(model, &format!("{prefix}.indexer_compressor_gate.weight")),
                norm: load_f32(model, &format!("{prefix}.indexer_compressor_norm.weight")),
            },
        });
        let shared_moe =
            load_shared_expert_moe_layer(model, layer_index, true, load_f32, load_quantized)
                .expect("DeepSeek4 requires a routed and shared expert in every trunk layer");
        let hash_routes = (layer_index < metadata.hash_layer_count)
            .then(|| load_i32_weight(model, &format!("{prefix}.ffn_gate_tid2eid.weight")));
        layers.push(DeepSeek4LayerWeights {
            attn_norm: load_f32(model, &format!("{prefix}.attn_norm.weight")),
            attention: AttentionWeights {
                q_a: load_quantized(model, &format!("{prefix}.attn_q_a.weight")),
                q_a_norm: load_f32(model, &format!("{prefix}.attn_q_a_norm.weight")),
                q_b: load_quantized(model, &format!("{prefix}.attn_q_b.weight")),
                kv: load_quantized(model, &format!("{prefix}.attn_kv.weight")),
                kv_norm: load_f32(model, &format!("{prefix}.attn_kv_a_norm.weight")),
                sinks: load_f32(model, &format!("{prefix}.attn_sinks.weight")),
                output_a_groups: load_output_a_groups(
                    model,
                    &format!("{prefix}.attn_output_a.weight"),
                    config.output_groups,
                    config.output_lora_rank,
                    config.num_heads * config.head_dim / config.output_groups,
                ),
                output_b: load_quantized(model, &format!("{prefix}.attn_output_b.weight")),
                compressor,
                indexer,
            },
            attn_hc: load_hc(model, &prefix, "attn", load_f32),
            ffn_norm: load_f32(model, &format!("{prefix}.ffn_norm.weight")),
            moe: DeepSeek4MoeWeights {
                weights: shared_moe,
                hash_routes,
                routed_clamp: metadata.swiglu_clamp_exp[layer_index],
                shared_clamp: metadata.swiglu_clamp_shared[layer_index],
                prefer_sparse_moe_cuda: false,
            },
            ffn_hc: load_hc(model, &prefix, "ffn", load_f32),
        });
    }
    let state = DeepSeek4State::new(&metadata.compress_ratios);
    Some(DeepSeek4Weights {
        config,
        layers,
        output_hc_function: load_f32(model, "output_hc_fn.weight"),
        output_hc_scale: load_f32(model, "output_hc_scale.weight"),
        output_hc_base: load_f32(model, "output_hc_base.weight"),
        state,
    })
}

fn build_config(model: &LoadedModel, metadata: &DeepSeek4Metadata) -> DeepSeek4Config {
    DeepSeek4Config {
        hidden_dim: model.metadata.hidden_size,
        num_heads: model.metadata.num_heads,
        head_dim: model.metadata.head_dim,
        rope_dim: model.metadata.rope_dim,
        output_groups: metadata.output_group_count,
        output_lora_rank: metadata.output_lora_rank,
        window_size: model.metadata.sliding_window,
        index_heads: metadata.indexer.head_count,
        index_head_dim: metadata.indexer.key_length,
        index_topk: metadata.indexer.top_k,
        hc_count: metadata.hyper_connection_count,
        sinkhorn_iterations: metadata.sinkhorn_iterations,
        hc_eps: metadata.hyper_connection_eps,
        norm_eps: model.metadata.norm_eps,
        expert_count: model.metadata.expert_count,
        expert_used_count: model.metadata.expert_used_count,
        expert_ffn_dim: model.metadata.expert_feed_forward_length,
        expert_scale: model.metadata.expert_weights_scale,
        rope_theta: model.metadata.rope_theta,
        compress_rope_theta: metadata.compress_rope_theta,
        rope_factor: metadata.rope_scaling_factor,
        rope_original_context_length: metadata.rope_original_context_length,
        rope_yarn_beta_fast: metadata.rope_yarn_beta_fast,
        rope_yarn_beta_slow: metadata.rope_yarn_beta_slow,
    }
}

fn load_hc(
    model: &LoadedModel,
    prefix: &str,
    kind: &str,
    load_f32: F32WeightLoader,
) -> HyperConnectionWeights {
    HyperConnectionWeights {
        function: load_f32(model, &format!("{prefix}.hc_{kind}_fn.weight")),
        scale: load_f32(model, &format!("{prefix}.hc_{kind}_scale.weight")),
        base: load_f32(model, &format!("{prefix}.hc_{kind}_base.weight")),
    }
}

fn load_i32_weight(model: &LoadedModel, name: &str) -> Vec<i32> {
    let tensor = model
        .weights
        .get(name)
        .unwrap_or_else(|| panic!("DeepSeek4 missing {name}"));
    tensor
        .as_bytes()
        .unwrap_or_else(|| panic!("DeepSeek4 {name} has no host bytes"))
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn load_output_a_groups(
    model: &LoadedModel,
    name: &str,
    groups: usize,
    rows_per_group: usize,
    cols_per_group: usize,
) -> Vec<QuantizedWeight> {
    let tensor = model
        .weights
        .get(name)
        .unwrap_or_else(|| panic!("DeepSeek4 missing {name}"));
    let quant = model
        .tensor_ggml_types
        .get(name)
        .copied()
        .unwrap_or(GGMLType::Q8_0);
    let (block_elements, block_bytes) = ggml_quant_params(quant);
    let bytes_per_row = cols_per_group.div_ceil(block_elements) * block_bytes;
    let bytes_per_group = rows_per_group * bytes_per_row;
    let raw_len = tensor.as_bytes().map_or(0, <[u8]>::len);
    assert_eq!(
        raw_len,
        groups * bytes_per_group,
        "DeepSeek4 {name} byte shape"
    );
    (0..groups)
        .map(|group| {
            let start = group * bytes_per_group;
            let slice = tensor
                .slice(&[start..start + bytes_per_group])
                .unwrap_or_else(|error| panic!("DeepSeek4 {name} group slice failed: {error}"));
            QuantizedWeight::new(slice, quant, rows_per_group, cols_per_group)
        })
        .collect()
}
