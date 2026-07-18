mod layers;
mod output;
mod quantized;
mod tensor_values;

use super::backend_runtime::is_attention_layer;
use super::layer_weights::{LayerType, ModelWeights, MtpLayerWeights};
use super::models::{gemma, nemotron, shared_expert_moe};
use super::packed_runtime::PackedModel;
use output::load_packed_output_weight;
use quantized::load_quantized_weight;
use rnb_loader::{Architecture as ModelArchitecture, LoadedModel, ModelLayerKind};
use tensor_values::load_f32_weight;

pub(super) fn load_model_weights(
    model: &LoadedModel,
    packed_model: Option<&PackedModel>,
    num_layers: usize,
    full_attention_interval: usize,
    _gguf_path: &std::path::Path,
) -> ModelWeights {
    eprintln!("[INFO] Loading weights (quantized, zero-copy)...");

    // embedding table → 양자화 유지 (gather 시 on-the-fly dequant)
    let token_embd = load_quantized_weight(model, "token_embd.weight");
    eprintln!(
        "[INFO] token_embd: {:?} [{}x{}]",
        token_embd.ggml_type, token_embd.rows, token_embd.cols
    );

    let output_norm_name =
        if super::policy::use_norm_weight_enabled() && model.weights.contains_key("norm.weight") {
            "norm.weight"
        } else {
            "output_norm.weight"
        };
    let output_norm = load_f32_weight(model, output_norm_name);

    // output weight → 양자화 유지 (matmul)
    let output = if model.weights.contains_key("output.weight") {
        let w = load_quantized_weight(model, "output.weight");
        eprintln!(
            "[INFO] output.weight: {:?} [{}x{}]",
            w.ggml_type, w.rows, w.cols
        );
        w
    } else {
        load_packed_output_weight(packed_model, &token_embd)
    };

    let mut layers = Vec::with_capacity(num_layers);
    let mut n_attn = 0usize;
    let mut n_gdn = 0usize;

    for i in 0..num_layers {
        let split_moe_layer = matches!(
            model.metadata.architecture,
            ModelArchitecture::Qwen35MoE | ModelArchitecture::Hy3 | ModelArchitecture::GlmDsa
        ) && model
            .weights
            .contains_key(&format!("blk.{i}.ffn_gate_exps.weight"));
        let shared_expert_moe_weights = shared_expert_moe::load_shared_expert_moe_layer(
            model,
            i,
            split_moe_layer,
            tensor_values::load_f32_weight,
            quantized::load_quantized_weight,
        );

        let layer_kind =
            layer_kind_for_load(&model.metadata.layer_kinds, i, full_attention_interval);
        if model.metadata.architecture == ModelArchitecture::NemotronHMoE {
            match layer_kind {
                ModelLayerKind::Recurrent => {
                    layers.push(LayerType::NemotronMamba2(
                        nemotron::mamba::load_mamba2_layer(
                            model,
                            i,
                            tensor_values::load_f32_weight,
                            quantized::load_quantized_weight,
                        ),
                    ));
                    n_gdn += 1;
                }
                ModelLayerKind::MoE => {
                    layers.push(LayerType::NemotronMoE(nemotron::moe::load_moe_layer(
                        model,
                        i,
                        tensor_values::load_f32_weight,
                        tensor_values::load_optional_f32_weight,
                        quantized::load_quantized_weight,
                    )));
                    n_gdn += 1;
                }
                ModelLayerKind::Attention => {
                    layers.push(LayerType::Attention(layers::load_attention_layer(
                        model, i, false, true, None,
                    )));
                    n_attn += 1;
                }
            }
        } else if layer_kind != ModelLayerKind::Recurrent {
            layers.push(LayerType::Attention(layers::load_attention_layer(
                model,
                i,
                split_moe_layer,
                false,
                shared_expert_moe_weights,
            )));
            n_attn += 1;
        } else {
            layers.push(LayerType::GatedDeltaNet(layers::load_gdn_layer(
                model,
                i,
                split_moe_layer,
                shared_expert_moe_weights,
            )));
            n_gdn += 1;
        }
    }

    eprintln!(
        "[INFO] Weights loaded: {} layers ({} attention + {} GDN)",
        num_layers, n_attn, n_gdn
    );

    ModelWeights {
        token_embd,
        output_norm,
        output,
        layers,
        gemma_per_layer: gemma::load_gemma_per_layer(
            model,
            num_layers,
            tensor_values::load_f32_weight,
            quantized::load_quantized_weight,
        ),
        glm_dsa_attention: (model.metadata.architecture == ModelArchitecture::GlmDsa).then(|| {
            super::models::glm_dsa::load_attention_layers(
                model,
                num_layers,
                tensor_values::load_f32_weight,
                quantized::load_quantized_weight,
            )
        }),
        rope_freqs: tensor_values::load_optional_f32_weight(model, "rope_freqs.weight"),
    }
}

pub(super) fn load_mtp_layer_weights(model: &LoadedModel) -> Option<MtpLayerWeights> {
    let mtp = model.metadata.mtp.as_ref()?;
    if mtp.nextn_predict_layers == 0 {
        return None;
    }
    if mtp.nextn_predict_layers != 1 {
        eprintln!(
            "[WARN] MTP runtime currently supports one nextn layer, model has {}",
            mtp.nextn_predict_layers
        );
        return None;
    }

    let layer_index = mtp.first_mtp_layer;
    let has_shared_expert_moe = matches!(
        model.metadata.architecture,
        ModelArchitecture::Qwen35MoE | ModelArchitecture::GlmDsa
    );
    let shared_expert_moe_weights = shared_expert_moe::load_shared_expert_moe_layer(
        model,
        layer_index,
        has_shared_expert_moe,
        tensor_values::load_f32_weight,
        quantized::load_quantized_weight,
    );
    let block = layers::load_attention_layer(
        model,
        layer_index,
        has_shared_expert_moe,
        false,
        shared_expert_moe_weights,
    );

    let optional_quantized = |name: String| {
        model
            .weights
            .contains_key(&name)
            .then(|| load_quantized_weight(model, &name))
    };

    Some(MtpLayerWeights {
        layer_index,
        eh_proj: load_quantized_weight(model, &format!("blk.{layer_index}.nextn.eh_proj.weight")),
        enorm: load_f32_weight(model, &format!("blk.{layer_index}.nextn.enorm.weight")),
        hnorm: load_f32_weight(model, &format!("blk.{layer_index}.nextn.hnorm.weight")),
        shared_head_norm: load_f32_weight(
            model,
            &format!("blk.{layer_index}.nextn.shared_head_norm.weight"),
        ),
        embed_tokens: optional_quantized(format!("blk.{layer_index}.nextn.embed_tokens.weight")),
        shared_head_head: optional_quantized(format!(
            "blk.{layer_index}.nextn.shared_head_head.weight"
        )),
        block,
        glm_dsa_attention: (model.metadata.architecture == ModelArchitecture::GlmDsa).then(|| {
            super::models::glm_dsa::load_attention_layer(
                model,
                layer_index,
                tensor_values::load_f32_weight,
                quantized::load_quantized_weight,
            )
        }),
    })
}

fn layer_kind_for_load(
    layer_kinds: &[ModelLayerKind],
    layer_idx: usize,
    full_attention_interval: usize,
) -> ModelLayerKind {
    layer_kinds.get(layer_idx).copied().unwrap_or_else(|| {
        if is_attention_layer(layer_idx, full_attention_interval) {
            ModelLayerKind::Attention
        } else {
            ModelLayerKind::Recurrent
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_layer_kind_overrides_interval_pattern() {
        let kinds = [
            ModelLayerKind::Recurrent,
            ModelLayerKind::MoE,
            ModelLayerKind::Attention,
        ];

        assert_eq!(layer_kind_for_load(&kinds, 0, 4), ModelLayerKind::Recurrent);
        assert_eq!(layer_kind_for_load(&kinds, 1, 4), ModelLayerKind::MoE);
        assert_eq!(layer_kind_for_load(&kinds, 2, 4), ModelLayerKind::Attention);
    }

    #[test]
    fn missing_loader_layer_kind_uses_legacy_attention_interval() {
        assert_eq!(layer_kind_for_load(&[], 0, 4), ModelLayerKind::Recurrent);
        assert_eq!(layer_kind_for_load(&[], 3, 4), ModelLayerKind::Attention);
        assert_eq!(layer_kind_for_load(&[], 0, 0), ModelLayerKind::Attention);
    }
}
