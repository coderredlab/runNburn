use rnb_core::tensor::Tensor;
use rnb_loader::{DFlashMetadata, DeepSeek4Metadata, GGMLType, LoadedModel};

use crate::error::{LlmError, Result};
const SUPPORTED_BLOCK_SIZE: usize = 5;
const SUPPORTED_TARGET_LAYERS: &[usize] = &[41, 42, 43];
const SUPPORTED_STAGE_COUNT: usize = SUPPORTED_TARGET_LAYERS.len();

pub(super) fn validate_dspark_weight_contract(model: &LoadedModel) -> Result<()> {
    let metadata = model.metadata.deepseek4.as_ref().ok_or_else(|| {
        LlmError::ModelLoad("DFlash sidecar has no DeepSeek4-compatible metadata".into())
    })?;
    let dflash = model.metadata.dflash.as_ref().ok_or_else(|| {
        LlmError::ModelLoad("DFlash sidecar has no generic DFlash metadata".into())
    })?;
    validate_runtime_contract(
        dflash,
        model.metadata.num_layers,
        model.metadata.sliding_window,
        &metadata.compress_ratios,
    )?;
    let hidden = model.metadata.hidden_size;
    let vocab = model.metadata.vocab_size;
    let extracted_width = dflash
        .target_layers
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| LlmError::ModelLoad("DSpark target feature width overflows usize".into()))?;
    require_matrix(model, "fc.weight", hidden, extracted_width)?;
    require_numel(model, "enc.output_norm.weight", hidden)?;
    require_numel(model, "output_norm.weight", hidden)?;

    let markov_w1 = require_matrix_rows(model, "markov_w1.weight", vocab)?;
    require_matrix(model, "markov_w2.weight", vocab, markov_w1)?;
    require_matrix(
        model,
        "conf_proj.weight",
        1,
        hidden.checked_add(markov_w1).ok_or_else(|| {
            LlmError::ModelLoad("DSpark confidence input width overflows usize".into())
        })?,
    )?;

    validate_trunk_weights(model, metadata)
}

fn validate_trunk_weights(model: &LoadedModel, metadata: &DeepSeek4Metadata) -> Result<()> {
    let hidden = model.metadata.hidden_size;
    let heads = model.metadata.num_heads;
    let head_dim = model.metadata.head_dim;
    let q_rank = metadata.q_lora_rank;
    let output_groups = metadata.output_group_count;
    let output_rank = metadata.output_lora_rank;
    let hc = metadata.hyper_connection_count;
    let experts = model.metadata.expert_count;
    let used_experts = model.metadata.expert_used_count;
    let expert_ffn = model.metadata.expert_feed_forward_length;
    if output_groups == 0 || heads % output_groups != 0 {
        return Err(LlmError::ModelLoad(format!(
            "DSpark attention heads {heads} are not divisible by output groups {output_groups}"
        )));
    }
    if hc == 0 || experts == 0 || used_experts == 0 || used_experts > experts {
        return Err(LlmError::ModelLoad(format!(
            "DSpark invalid HC/expert contract: hc={hc} experts={experts} used={used_experts}"
        )));
    }

    let hc_width = hc
        .checked_mul(hidden)
        .ok_or_else(|| LlmError::ModelLoad("DSpark HC width overflows usize".into()))?;
    let hc_mix = hc
        .checked_mul(hc + 2)
        .ok_or_else(|| LlmError::ModelLoad("DSpark HC mix width overflows usize".into()))?;
    require_matrix(model, "output_hc_fn.weight", hc, hc_width)?;
    require_numel(model, "output_hc_scale.weight", 1)?;
    require_numel(model, "output_hc_base.weight", hc)?;

    for layer in 0..model.metadata.num_layers {
        let prefix = format!("blk.{layer}");
        require_numel(model, &format!("{prefix}.attn_norm.weight"), hidden)?;
        require_matrix(model, &format!("{prefix}.attn_q_a.weight"), q_rank, hidden)?;
        require_numel(model, &format!("{prefix}.attn_q_a_norm.weight"), q_rank)?;
        require_matrix(
            model,
            &format!("{prefix}.attn_q_b.weight"),
            heads * head_dim,
            q_rank,
        )?;
        require_matrix(model, &format!("{prefix}.attn_kv.weight"), head_dim, hidden)?;
        require_numel(model, &format!("{prefix}.attn_kv_a_norm.weight"), head_dim)?;
        require_numel(model, &format!("{prefix}.attn_sinks.weight"), heads)?;
        require_numel(
            model,
            &format!("{prefix}.attn_output_a.weight"),
            output_rank
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(head_dim))
                .ok_or_else(|| {
                    LlmError::ModelLoad("DSpark attention output-A size overflows usize".into())
                })?,
        )?;
        require_matrix(
            model,
            &format!("{prefix}.attn_output_b.weight"),
            hidden,
            output_groups * output_rank,
        )?;
        validate_hc(model, &prefix, "attn", hc_mix, hc_width)?;

        require_numel(model, &format!("{prefix}.ffn_norm.weight"), hidden)?;
        require_matrix(
            model,
            &format!("{prefix}.ffn_gate_inp.weight"),
            experts,
            hidden,
        )?;
        require_numel(
            model,
            &format!("{prefix}.ffn_gate_exps.weight"),
            experts
                .checked_mul(expert_ffn)
                .and_then(|value| value.checked_mul(hidden))
                .ok_or_else(|| LlmError::ModelLoad("DSpark gate experts size overflows".into()))?,
        )?;
        require_numel(
            model,
            &format!("{prefix}.ffn_up_exps.weight"),
            experts
                .checked_mul(expert_ffn)
                .and_then(|value| value.checked_mul(hidden))
                .ok_or_else(|| LlmError::ModelLoad("DSpark up experts size overflows".into()))?,
        )?;
        require_numel(
            model,
            &format!("{prefix}.ffn_down_exps.weight"),
            experts
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(expert_ffn))
                .ok_or_else(|| LlmError::ModelLoad("DSpark down experts size overflows".into()))?,
        )?;
        require_matrix(
            model,
            &format!("{prefix}.ffn_gate_shexp.weight"),
            expert_ffn,
            hidden,
        )?;
        require_matrix(
            model,
            &format!("{prefix}.ffn_up_shexp.weight"),
            expert_ffn,
            hidden,
        )?;
        require_matrix(
            model,
            &format!("{prefix}.ffn_down_shexp.weight"),
            hidden,
            expert_ffn,
        )?;
        let shared_scale = format!("{prefix}.ffn_gate_inp_shexp.weight");
        if model.weights.contains_key(&shared_scale) {
            require_numel(model, &shared_scale, hidden)?;
        }
        if layer < metadata.hash_layer_count {
            validate_hash_routes(
                model,
                &format!("{prefix}.ffn_gate_tid2eid.weight"),
                experts,
                used_experts,
            )?;
        } else if ![
            format!("{prefix}.exp_probs_b"),
            format!("{prefix}.exp_probs_b.bias"),
        ]
        .iter()
        .any(|name| model.weights.contains_key(name))
        {
            return Err(LlmError::ModelLoad(format!(
                "DSpark missing router selection bias for layer {layer}"
            )));
        }
        validate_hc(model, &prefix, "ffn", hc_mix, hc_width)?;
    }
    Ok(())
}

fn validate_hc(
    model: &LoadedModel,
    prefix: &str,
    kind: &str,
    mix_count: usize,
    input_width: usize,
) -> Result<()> {
    require_matrix(
        model,
        &format!("{prefix}.hc_{kind}_fn.weight"),
        mix_count,
        input_width,
    )?;
    require_numel(model, &format!("{prefix}.hc_{kind}_scale.weight"), 3)?;
    require_numel(model, &format!("{prefix}.hc_{kind}_base.weight"), mix_count)?;
    Ok(())
}

fn validate_hash_routes(
    model: &LoadedModel,
    name: &str,
    experts: usize,
    used_experts: usize,
) -> Result<()> {
    if model.tensor_ggml_types.get(name) != Some(&GGMLType::I32) {
        return Err(LlmError::ModelLoad(format!("DSpark {name} must use I32")));
    }
    let expected = model
        .metadata
        .vocab_size
        .checked_mul(used_experts)
        .ok_or_else(|| LlmError::ModelLoad(format!("DSpark {name} size overflows usize")))?;
    let shape = raw_logical_shape(model, name)?;
    let actual = shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim));
    if actual != Some(expected) {
        return Err(LlmError::ModelLoad(format!(
            "DSpark {name} shape {shape:?} has {:?} elements, expected {expected}",
            actual
        )));
    }
    let bytes = model
        .weights
        .get(name)
        .and_then(|tensor| tensor.as_bytes())
        .ok_or_else(|| LlmError::ModelLoad(format!("DSpark {name} has no host bytes")))?;
    for chunk in bytes.chunks_exact(4) {
        let expert = i32::from_le_bytes(chunk.try_into().expect("four-byte I32 chunk"));
        if expert < 0 || expert as usize >= experts {
            return Err(LlmError::ModelLoad(format!(
                "DSpark {name} contains expert {expert} outside 0..{experts}"
            )));
        }
    }
    Ok(())
}

fn require_matrix_rows(model: &LoadedModel, name: &str, rows: usize) -> Result<usize> {
    let shape = logical_shape(model, name)?;
    if shape.len() != 2 || shape[0] != rows || shape[1] == 0 {
        return Err(LlmError::ModelLoad(format!(
            "DSpark {name} shape {shape:?} != [{rows}, nonzero]"
        )));
    }
    Ok(shape[1])
}

fn require_matrix(model: &LoadedModel, name: &str, rows: usize, cols: usize) -> Result<()> {
    let shape = logical_shape(model, name)?;
    if shape != [rows, cols] {
        return Err(LlmError::ModelLoad(format!(
            "DSpark {name} shape {shape:?} != [{rows}, {cols}]"
        )));
    }
    Ok(())
}

fn require_numel(model: &LoadedModel, name: &str, expected: usize) -> Result<()> {
    let shape = logical_shape(model, name)?;
    let actual = shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim));
    if actual != Some(expected) {
        return Err(LlmError::ModelLoad(format!(
            "DSpark {name} shape {shape:?} has {:?} elements, expected {expected}",
            actual
        )));
    }
    Ok(())
}

fn logical_shape<'a>(model: &'a LoadedModel, name: &str) -> Result<&'a [usize]> {
    let shape = raw_logical_shape(model, name)?;
    let ggml_type = model
        .tensor_ggml_types
        .get(name)
        .ok_or_else(|| LlmError::ModelLoad(format!("DSpark tensor {name} has no GGML type")))?;
    if matches!(
        ggml_type,
        GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 | GGMLType::F64
    ) {
        return Err(LlmError::ModelLoad(format!(
            "DSpark tensor {name} has unsupported numeric type {ggml_type:?}"
        )));
    }
    Ok(shape)
}

fn raw_logical_shape<'a>(model: &'a LoadedModel, name: &str) -> Result<&'a [usize]> {
    let tensor = model
        .weights
        .get(name)
        .ok_or_else(|| LlmError::ModelLoad(format!("DSpark missing required tensor {name}")))?;
    Ok(select_logical_shape(
        model.float_shapes.get(name).map(Vec::as_slice),
        tensor,
    ))
}

fn select_logical_shape<'a>(
    quantized_shape: Option<&'a [usize]>,
    tensor: &'a Tensor,
) -> &'a [usize] {
    quantized_shape.unwrap_or_else(|| tensor.shape())
}

fn validate_runtime_contract(
    dflash: &DFlashMetadata,
    stage_count: usize,
    sliding_window: usize,
    compress_ratios: &[usize],
) -> Result<()> {
    if dflash.block_size != SUPPORTED_BLOCK_SIZE {
        return Err(LlmError::ModelLoad(format!(
            "DSpark block size {} is unsupported; expected {SUPPORTED_BLOCK_SIZE}",
            dflash.block_size
        )));
    }
    if dflash.target_layers != SUPPORTED_TARGET_LAYERS {
        return Err(LlmError::ModelLoad(format!(
            "DSpark target layers {:?} are unsupported; expected {SUPPORTED_TARGET_LAYERS:?}",
            dflash.target_layers
        )));
    }
    if stage_count != SUPPORTED_STAGE_COUNT {
        return Err(LlmError::ModelLoad(format!(
            "DSpark stage count {stage_count} is unsupported; expected {SUPPORTED_STAGE_COUNT}"
        )));
    }
    if sliding_window == 0 {
        return Err(LlmError::ModelLoad(
            "DSpark requires a positive sliding attention window".into(),
        ));
    }
    if let Some((layer, ratio)) = compress_ratios
        .iter()
        .copied()
        .enumerate()
        .find(|(_, ratio)| *ratio != 0)
    {
        return Err(LlmError::ModelLoad(format!(
            "DSpark layer {layer} uses unsupported attention compression ratio {ratio}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_fc_uses_mapped_tensor_shape_when_quantized_shape_is_absent() {
        let tensor = Tensor::from_slice(&[0.0f32; 12], &[4, 3]);
        assert_eq!(select_logical_shape(None, &tensor), [4, 3]);
        assert_eq!(
            select_logical_shape(Some(&[4, 3]), &Tensor::from_slice(&[0u8; 12], &[12])),
            [4, 3]
        );
    }

    #[test]
    fn rejects_runtime_metadata_outside_supported_dspark_contract() {
        let supported = DFlashMetadata {
            block_size: SUPPORTED_BLOCK_SIZE,
            target_layers: SUPPORTED_TARGET_LAYERS.to_vec(),
            mask_token_id: 128799,
        };
        let compress_ratios = [0, 0, 0];
        assert!(
            validate_runtime_contract(&supported, SUPPORTED_STAGE_COUNT, 0, &compress_ratios,)
                .is_err()
        );
        assert!(
            validate_runtime_contract(&supported, SUPPORTED_STAGE_COUNT, 1, &[0, 4, 0],).is_err()
        );

        let wrong_block = DFlashMetadata {
            block_size: SUPPORTED_BLOCK_SIZE + 1,
            ..supported.clone()
        };
        assert!(validate_runtime_contract(
            &wrong_block,
            SUPPORTED_STAGE_COUNT,
            1,
            &compress_ratios,
        )
        .is_err());

        let wrong_layers = DFlashMetadata {
            target_layers: vec![40, 41, 42],
            ..supported.clone()
        };
        assert!(validate_runtime_contract(
            &wrong_layers,
            SUPPORTED_STAGE_COUNT,
            1,
            &compress_ratios,
        )
        .is_err());
        assert!(validate_runtime_contract(
            &supported,
            SUPPORTED_STAGE_COUNT + 1,
            1,
            &compress_ratios,
        )
        .is_err());
    }
}
