use super::super::cpu_runtime::kernels;
use super::super::layer_weights::{
    AttentionLayerWeights, GdnLayerWeights, SharedExpertMoELayerWeights,
};
use super::super::models::gemma::load_gemma_moe_layer;
use super::quantized::{dummy_quantized_weight, fuse_gate_up, load_quantized_weight};
use super::tensor_values::{load_f32_weight, load_optional_f32_weight};
use rnb_core::tensor::Tensor;
use rnb_loader::{Architecture, LoadedModel};

pub(super) fn load_attention_layer(
    model: &LoadedModel,
    i: usize,
    has_shared_expert_moe: bool,
    skip_dense_ffn: bool,
    shared_expert_moe_weights: Option<SharedExpertMoELayerWeights>,
) -> AttentionLayerWeights {
    let compressed_mla = model.metadata.architecture == Architecture::GlmDsa;
    // Optional bias (Qwen2 등)
    let q_bias = load_optional_f32_weight(model, &format!("blk.{i}.attn_q.bias"));
    let k_bias = load_optional_f32_weight(model, &format!("blk.{i}.attn_k.bias"));
    let v_bias = load_optional_f32_weight(model, &format!("blk.{i}.attn_v.bias"));

    // Q/K norm (Qwen3.5 attention layers)
    let q_norm = load_optional_f32_weight(model, &format!("blk.{i}.attn_q_norm.weight"));
    let k_norm = load_optional_f32_weight(model, &format!("blk.{i}.attn_k_norm.weight"));

    // post_attention_norm (Qwen3.5)
    let post_attn_norm =
        load_optional_f32_weight(model, &format!("blk.{i}.post_attention_norm.weight"));
    let post_ffw_norm = load_optional_f32_weight(model, &format!("blk.{i}.post_ffw_norm.weight"));
    let out_scale = load_optional_f32_weight(model, &format!("blk.{i}.layer_output_scale.weight"));

    // Gemma4 26B-A4B hybrid MoE loading (None if dense-only layer).
    let moe_weights = load_gemma_moe_layer(model, i, &post_ffw_norm, load_f32_weight);

    let v_name = format!("blk.{i}.attn_v.weight");
    let k_name = format!("blk.{i}.attn_k.weight");
    let v_proj_missing = !compressed_mla && !model.weights.contains_key(&v_name);
    // Alias K weight as V placeholder when V projection is absent (gemma4 full-attn layers).
    // Forward/decode paths re-derive V from K at runtime; this avoids a zero placeholder
    // that would otherwise break shape checks (k.rows == v.rows expected).
    let v_load_name = if v_proj_missing { &k_name } else { &v_name };

    let mut weights = AttentionLayerWeights {
        attn_norm: load_f32_weight(model, &format!("blk.{i}.attn_norm.weight")),
        q_weight: if compressed_mla {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.attn_q.weight"))
        },
        k_weight: if compressed_mla {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &k_name)
        },
        v_weight: if compressed_mla {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, v_load_name)
        },
        o_weight: load_quantized_weight(model, &format!("blk.{i}.attn_output.weight")),
        q_bias,
        k_bias,
        v_bias,
        q_norm,
        k_norm,
        post_attn_norm: post_attn_norm.clone(),
        out_scale,
        post_ffw_norm,
        ffn_norm: {
            let name = format!("blk.{i}.ffn_norm.weight");
            if skip_dense_ffn {
                load_f32_weight(model, &format!("blk.{i}.attn_norm.weight"))
            } else if model.weights.contains_key(&name) {
                load_f32_weight(model, &name)
            } else if let Some(ref pan) = post_attn_norm {
                pan.clone()
            } else {
                load_f32_weight(model, &name) // will warn
            }
        },
        ffn_gate_weight: if has_shared_expert_moe || skip_dense_ffn {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_gate.weight"))
        },
        ffn_up_weight: if has_shared_expert_moe || skip_dense_ffn {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_up.weight"))
        },
        ffn_down_weight: if has_shared_expert_moe || skip_dense_ffn {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_down.weight"))
        },
        ffn_gate_up_fused: None,
        moe: moe_weights,
        shared_expert_moe: shared_expert_moe_weights,
        v_proj_missing,
    };

    if weights.shared_expert_moe.is_none() && !skip_dense_ffn {
        weights.ffn_gate_up_fused = fuse_gate_up(&weights.ffn_gate_weight, &weights.ffn_up_weight);
    }
    weights
}

pub(super) fn load_gdn_layer(
    model: &LoadedModel,
    i: usize,
    has_shared_expert_moe: bool,
    shared_expert_moe_weights: Option<SharedExpertMoELayerWeights>,
) -> GdnLayerWeights {
    // GDN (Gated Delta Net) layer
    let mut weights = GdnLayerWeights {
        attn_norm: load_f32_weight(model, &format!("blk.{i}.attn_norm.weight")),
        post_attn_norm: load_gdn_post_attention_norm(model, i),
        qkv_weight: load_quantized_weight(model, &format!("blk.{i}.attn_qkv.weight")),
        gate_weight: load_quantized_weight(model, &format!("blk.{i}.attn_gate.weight")),
        ssm_a: load_f32_weight(model, &format!("blk.{i}.ssm_a")),
        ssm_alpha: load_quantized_weight(model, &format!("blk.{i}.ssm_alpha.weight")),
        ssm_beta: load_quantized_weight(model, &format!("blk.{i}.ssm_beta.weight")),
        ssm_conv1d: load_gdn_conv1d(model, i),
        ssm_dt_bias: load_f32_weight(model, &format!("blk.{i}.ssm_dt.bias")),
        ssm_norm: load_f32_weight(model, &format!("blk.{i}.ssm_norm.weight")),
        ssm_out: load_quantized_weight(model, &format!("blk.{i}.ssm_out.weight")),
        ffn_gate_weight: if has_shared_expert_moe {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_gate.weight"))
        },
        ffn_up_weight: if has_shared_expert_moe {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_up.weight"))
        },
        ffn_down_weight: if has_shared_expert_moe {
            dummy_quantized_weight()
        } else {
            load_quantized_weight(model, &format!("blk.{i}.ffn_down.weight"))
        },
        ffn_gate_up_fused: None,
        shared_expert_moe: shared_expert_moe_weights,
    };

    if weights.shared_expert_moe.is_none() {
        weights.ffn_gate_up_fused = fuse_gate_up(&weights.ffn_gate_weight, &weights.ffn_up_weight);
    }
    weights
}

fn load_gdn_conv1d(model: &LoadedModel, i: usize) -> Tensor {
    // ssm_conv1d: GGUF는 [channels, kernel_size]로 저장, 우리는 [kernel_size, channels] 필요
    let conv_raw = load_f32_weight(model, &format!("blk.{i}.ssm_conv1d.weight"));
    let conv_shape = conv_raw.shape();
    if conv_shape.len() == 2 && conv_shape[0] > conv_shape[1] {
        // [channels, kernel_size] → transpose to [kernel_size, channels]
        let data = kernels::tensor_as_f32_slice(&conv_raw);
        let channels = conv_shape[0];
        let ksize = conv_shape[1];
        let mut transposed = vec![0.0f32; channels * ksize];
        for c in 0..channels {
            for k in 0..ksize {
                transposed[k * channels + c] = data[c * ksize + k];
            }
        }
        Tensor::from_slice(&transposed, &[ksize, channels])
    } else {
        conv_raw
    }
}

fn load_gdn_post_attention_norm(model: &LoadedModel, i: usize) -> Tensor {
    // post_attention_norm — Qwen3.5 GDN layers에서는 ffn_norm 역할
    let post_norm_name = format!("blk.{i}.post_attention_norm.weight");
    if model.weights.contains_key(&post_norm_name) {
        load_f32_weight(model, &post_norm_name)
    } else {
        load_f32_weight(model, &format!("blk.{i}.ffn_norm.weight"))
    }
}
