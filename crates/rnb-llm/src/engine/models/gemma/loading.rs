use crate::engine::layer_weights::{
    GemmaPerLayerLayerWeights, GemmaPerLayerWeights, MoeLayerWeights,
};
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::LoadedModel;

type F32WeightLoader = fn(&LoadedModel, &str) -> Tensor;
type QuantizedWeightLoader = fn(&LoadedModel, &str) -> QuantizedWeight;

pub(in crate::engine) fn load_gemma_moe_layer(
    model: &LoadedModel,
    i: usize,
    post_ffw_norm: &Option<Tensor>,
    load_f32_weight: F32WeightLoader,
) -> Option<MoeLayerWeights> {
    let router_key = format!("blk.{i}.ffn_gate_inp.weight");
    let exps_key = format!("blk.{i}.ffn_gate_up_exps.weight");
    let down_exps_key = format!("blk.{i}.ffn_down_exps.weight");
    if !(model.metadata.expert_count > 0
        && model.weights.contains_key(&router_key)
        && model.weights.contains_key(&exps_key)
        && model.weights.contains_key(&down_exps_key))
    {
        return None;
    }

    let n_embd = model.metadata.hidden_size;
    let router_w = model.weights.get(&router_key).unwrap().clone();
    let router_scale_key = format!("blk.{i}.ffn_gate_inp.scale");
    let router_scale = if model.weights.contains_key(&router_scale_key) {
        model.weights.get(&router_scale_key).unwrap().clone()
    } else {
        Tensor::from_slice(&vec![1.0f32; n_embd], &[n_embd])
    };
    let gate_up_exps = model.weights.get(&exps_key).unwrap().clone();
    let down_exps = model.weights.get(&down_exps_key).unwrap().clone();
    let down_scale_key = format!("blk.{i}.ffn_down_exps.scale");
    let down_scale = if model.weights.contains_key(&down_scale_key) {
        load_f32_weight(model, &down_scale_key)
    } else {
        Tensor::from_slice(
            &vec![1.0f32; model.metadata.expert_count],
            &[model.metadata.expert_count],
        )
    };
    let down_quant = *model
        .tensor_ggml_types
        .get(&down_exps_key)
        .unwrap_or(&rnb_loader::GGMLType::Q5_1);
    let pffn1_key = format!("blk.{i}.post_ffw_norm_1.weight");
    let pffn2_key = format!("blk.{i}.post_ffw_norm_2.weight");
    let preffn2_key = format!("blk.{i}.pre_ffw_norm_2.weight");
    let post_ffw_norm_1 = if model.weights.contains_key(&pffn1_key) {
        load_f32_weight(model, &pffn1_key)
    } else if let Some(p) = post_ffw_norm {
        p.clone()
    } else {
        Tensor::from_slice(&vec![0.0f32; n_embd], &[n_embd])
    };
    let post_ffw_norm_2 = if model.weights.contains_key(&pffn2_key) {
        load_f32_weight(model, &pffn2_key)
    } else if let Some(p) = post_ffw_norm {
        p.clone()
    } else {
        Tensor::from_slice(&vec![0.0f32; n_embd], &[n_embd])
    };
    let pre_ffw_norm_2 = if model.weights.contains_key(&preffn2_key) {
        load_f32_weight(model, &preffn2_key)
    } else {
        Tensor::from_slice(&vec![0.0f32; n_embd], &[n_embd])
    };
    Some(MoeLayerWeights {
        router_w,
        router_scale,
        gate_up_exps,
        down_exps,
        down_scale,
        down_quant,
        post_ffw_norm_1,
        post_ffw_norm_2,
        pre_ffw_norm_2,
        n_embd,
        n_ff: model.metadata.expert_feed_forward_length,
        n_expert: model.metadata.expert_count,
        n_expert_used: model.metadata.expert_used_count,
        // Filled in later by the `.rnb` MoE wiring loop
        // after `packed_model` is loaded.
        packed_model: None,
        gate_up_rnb_name: None,
        down_rnb_name: None,
        router_rnb_name: None,
        rank_to_original: None,
        shadow_model: None,
        shadow_gate_up_rnb_name: None,
        // Filled later by Task 7 (`packed_wiring`) when the `.rnb` is a v3
        // sidecar carrying MoE hot/cold tier metadata.
        gate_up_residency: None,
        down_residency: None,
    })
}

pub(in crate::engine) fn load_gemma_per_layer(
    model: &LoadedModel,
    num_layers: usize,
    load_f32_weight: F32WeightLoader,
    load_quantized_weight: QuantizedWeightLoader,
) -> Option<GemmaPerLayerWeights> {
    if !(model.metadata.embedding_length_per_layer_input > 0
        && model.weights.contains_key("per_layer_token_embd.weight")
        && model.weights.contains_key("per_layer_model_proj.weight")
        && model.weights.contains_key("per_layer_proj_norm.weight")
        && model.weights.contains_key("blk.0.inp_gate.weight")
        && model.weights.contains_key("blk.0.proj.weight")
        && model.weights.contains_key("blk.0.post_norm.weight"))
    {
        return None;
    }

    Some(GemmaPerLayerWeights {
        token_embd: load_quantized_weight(model, "per_layer_token_embd.weight"),
        model_proj: load_quantized_weight(model, "per_layer_model_proj.weight"),
        proj_norm: load_f32_weight(model, "per_layer_proj_norm.weight"),
        layers: (0..num_layers)
            .map(|i| GemmaPerLayerLayerWeights {
                inp_gate: load_quantized_weight(model, &format!("blk.{i}.inp_gate.weight")),
                proj: load_quantized_weight(model, &format!("blk.{i}.proj.weight")),
                post_norm: load_f32_weight(model, &format!("blk.{i}.post_norm.weight")),
            })
            .collect(),
    })
}
