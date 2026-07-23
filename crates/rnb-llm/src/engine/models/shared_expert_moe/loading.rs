use crate::engine::layer_weights::SharedExpertMoELayerWeights;
use crate::engine::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;
use rnb_loader::{GGMLType, LoadedModel};

type F32WeightLoader = fn(&LoadedModel, &str) -> Tensor;
type QuantizedWeightLoader = fn(&LoadedModel, &str) -> QuantizedWeight;

pub(in crate::engine) fn load_shared_expert_moe_layer(
    model: &LoadedModel,
    i: usize,
    has_shared_expert_moe: bool,
    load_f32_weight: F32WeightLoader,
    load_quantized_weight: QuantizedWeightLoader,
) -> Option<SharedExpertMoELayerWeights> {
    if !has_shared_expert_moe {
        return None;
    }

    let n_embd = model.metadata.hidden_size;
    let n_ff = model.metadata.expert_feed_forward_length;
    let n_expert = model.metadata.expert_count;
    let n_expert_used = model.metadata.expert_used_count;
    let router_name = format!("blk.{i}.ffn_gate_inp.weight");
    let router_selection_bias_names = [
        format!("blk.{i}.exp_probs_b"),
        format!("blk.{i}.exp_probs_b.bias"),
    ];
    let gate_exps_name = format!("blk.{i}.ffn_gate_exps.weight");
    let up_exps_name = format!("blk.{i}.ffn_up_exps.weight");
    let down_exps_name = format!("blk.{i}.ffn_down_exps.weight");
    let shared_scale_name = format!("blk.{i}.ffn_gate_inp_shexp.weight");
    let router_w = load_f32_weight(model, &router_name);
    let router_selection_bias = router_selection_bias_names
        .iter()
        .find(|name| model.weights.contains_key(name.as_str()))
        .map(|name| load_f32_weight(model, name));
    let gate_exps = model
        .weights
        .get(&gate_exps_name)
        .unwrap_or_else(|| panic!("shared-expert MoE: missing {gate_exps_name}"))
        .clone();
    let up_exps = model
        .weights
        .get(&up_exps_name)
        .unwrap_or_else(|| panic!("shared-expert MoE: missing {up_exps_name}"))
        .clone();
    let down_exps = model
        .weights
        .get(&down_exps_name)
        .unwrap_or_else(|| panic!("shared-expert MoE: missing {down_exps_name}"))
        .clone();
    let gate_quant = model
        .tensor_ggml_types
        .get(&gate_exps_name)
        .copied()
        .unwrap_or(GGMLType::Q4_K);
    let up_quant = model
        .tensor_ggml_types
        .get(&up_exps_name)
        .copied()
        .unwrap_or(GGMLType::Q4_K);
    let down_quant = model
        .tensor_ggml_types
        .get(&down_exps_name)
        .copied()
        .unwrap_or(GGMLType::Q5_K);
    let shared_expert_gated = model.weights.contains_key(&shared_scale_name);
    let shared_input_scale = if shared_expert_gated {
        load_f32_weight(model, &shared_scale_name)
    } else {
        Tensor::from_slice::<f32>(&[], &[0])
    };
    let shared_gate = load_quantized_weight(model, &format!("blk.{i}.ffn_gate_shexp.weight"));
    let shared_up = load_quantized_weight(model, &format!("blk.{i}.ffn_up_shexp.weight"));
    let shared_down = load_quantized_weight(model, &format!("blk.{i}.ffn_down_shexp.weight"));
    Some(SharedExpertMoELayerWeights {
        router_w,
        router_selection_bias,
        expert_gating_func: model.metadata.expert_gating_func,
        expert_weights_norm: model.metadata.expert_weights_norm,
        expert_weights_scale: model.metadata.expert_weights_scale,
        gate_exps,
        gate_quant,
        up_exps,
        up_quant,
        down_exps,
        down_quant,
        shared_input_scale,
        shared_expert_gated,
        shared_gate,
        shared_up,
        shared_down,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        prefer_sparse_moe_cuda: false,
        sparse_page_cache: None,
    })
}
