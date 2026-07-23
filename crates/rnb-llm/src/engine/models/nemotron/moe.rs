//! Nemotron-H MoE-only blocks.
//!
//! Nemotron-H MoE uses router bias, routed up/down experts, optional latent
//! projection, and a shared up/down expert. It must not reuse the Qwen35 MoE
//! view, which assumes gate/up/down expert triplets.

use super::policy::{prefill_projection_path, PrefillProjectionPath};
use crate::engine::norm::apply_plain_rms_norm;
#[cfg(any(not(feature = "cuda"), test))]
use crate::engine::norm::{add_f32_inplace, axpby_f32_inplace, relu_sqr_f32_inplace};
use crate::engine::quantized_weight_types::{QuantizedWeight, QuantizedWeightDescriptor};
use crate::engine::types::ScratchBuffers;
use crate::engine::{backend_runtime, cpu_runtime::kernels, types::ModelMetadata};
use rnb_core::tensor::Tensor;
#[cfg(feature = "cuda")]
use rnb_loader::GGMLType;
use rnb_loader::LoadedModel;
#[cfg(any(not(feature = "cuda"), test))]
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[allow(dead_code)]
pub(in crate::engine) struct NemotronMoELayerWeights {
    pub(in crate::engine) norm: Tensor,
    pub(in crate::engine) router: QuantizedWeight,
    pub(in crate::engine) router_bias: Option<Tensor>,
    pub(in crate::engine) expert_down: QuantizedWeight,
    pub(in crate::engine) expert_up: QuantizedWeight,
    pub(in crate::engine) shared_expert_down: QuantizedWeight,
    pub(in crate::engine) shared_expert_up: QuantizedWeight,
    pub(in crate::engine) latent_down: Option<QuantizedWeight>,
    pub(in crate::engine) latent_up: Option<QuantizedWeight>,
}

struct NemotronMoePrefillPrepared {
    normed: Tensor,
    routes: Vec<rnb_model_nemotron::Route>,
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
    seq_len: usize,
}

#[cfg(feature = "cuda")]
type NemotronQ8SparseDeviceInputs<'a> = (
    &'a [u8],
    &'a [u8],
    Vec<&'a [u8]>,
    Vec<&'a [u8]>,
    Vec<f32>,
    Vec<u32>,
    usize,
);

#[cfg(feature = "cuda")]
type NemotronQ8SparseDeviceRoutePackInputs<'a> = (
    &'a [u8],
    &'a [u8],
    Vec<&'a [u8]>,
    Vec<&'a [u8]>,
    Vec<u32>,
    usize,
);

pub(in crate::engine) fn load_moe_layer(
    model: &LoadedModel,
    layer_idx: usize,
    load_f32_weight: fn(&LoadedModel, &str) -> Tensor,
    load_optional_f32_weight: fn(&LoadedModel, &str) -> Option<Tensor>,
    load_quantized_weight: fn(&LoadedModel, &str) -> QuantizedWeight,
) -> NemotronMoELayerWeights {
    let latent_down_name = format!("blk.{layer_idx}.ffn_down.weight");
    let latent_up_name = format!("blk.{layer_idx}.ffn_up.weight");
    NemotronMoELayerWeights {
        norm: load_f32_weight(model, &format!("blk.{layer_idx}.attn_norm.weight")),
        router: load_quantized_weight(model, &format!("blk.{layer_idx}.ffn_gate_inp.weight")),
        router_bias: load_optional_f32_weight(model, &format!("blk.{layer_idx}.exp_probs_b.bias")),
        expert_down: load_expert_tensor(
            model,
            &format!("blk.{layer_idx}.ffn_down_exps.weight"),
            load_quantized_weight,
        ),
        expert_up: load_expert_tensor(
            model,
            &format!("blk.{layer_idx}.ffn_up_exps.weight"),
            load_quantized_weight,
        ),
        shared_expert_down: load_quantized_weight(
            model,
            &format!("blk.{layer_idx}.ffn_down_shexp.weight"),
        ),
        shared_expert_up: load_quantized_weight(
            model,
            &format!("blk.{layer_idx}.ffn_up_shexp.weight"),
        ),
        latent_down: model
            .weights
            .contains_key(&latent_down_name)
            .then(|| load_quantized_weight(model, &latent_down_name)),
        latent_up: model
            .weights
            .contains_key(&latent_up_name)
            .then(|| load_quantized_weight(model, &latent_up_name)),
    }
}

fn load_expert_tensor(
    model: &LoadedModel,
    name: &str,
    load_quantized_weight: fn(&LoadedModel, &str) -> QuantizedWeight,
) -> QuantizedWeight {
    let mut weight = load_quantized_weight(model, name);
    let Some(shape) = model.float_shapes.get(name) else {
        return weight;
    };
    if shape.len() == 3 {
        weight.rows = shape[0] * shape[1];
        weight.cols = shape[2];
        weight.descriptor =
            QuantizedWeightDescriptor::new(weight.rows, weight.cols, weight.ggml_type);
    }
    weight
}

fn prepare_moe_prefill(
    metadata: &ModelMetadata,
    hidden: &Tensor,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
) -> crate::error::Result<NemotronMoePrefillPrepared> {
    let hidden_dim = metadata.hidden_dim;
    let seq_len = hidden.shape().first().copied().unwrap_or(1);
    let stage_start = (seq_len > 1).then(Instant::now);
    let normed = apply_plain_rms_norm(hidden, &w.norm, norm_eps)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?;
    record_prefill_moe_stage("norm", stage_start);
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    let n_expert = w.router.rows.max(metadata.expert_used_count).max(1);
    let n_ff = w.expert_up.rows / n_expert;
    let expert_used = metadata.expert_used_count.max(1).min(n_expert);
    let bias = w.router_bias.as_ref().map(kernels::tensor_as_f32_slice);
    let stage_start = (seq_len > 1).then(Instant::now);
    let router_logits = router_logits(&w.router, normed_data, seq_len, hidden_dim, n_expert)?;
    record_prefill_moe_stage("router", stage_start);
    let stage_start = (seq_len > 1).then(Instant::now);
    let routes = (0..seq_len)
        .map(|token_idx| {
            route_token(
                &router_logits[token_idx * n_expert..(token_idx + 1) * n_expert],
                bias,
                expert_used,
                metadata.expert_weights_scale,
            )
        })
        .collect::<Vec<_>>();
    record_prefill_moe_stage("routing", stage_start);

    Ok(NemotronMoePrefillPrepared {
        normed,
        routes,
        n_expert,
        n_ff,
        hidden_dim,
        seq_len,
    })
}

pub(in crate::engine) fn forward_moe_layer(
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, None)
}

pub(in crate::engine) fn forward_moe_layer_for_prefill(
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
    layer_idx: Option<usize>,
) -> crate::error::Result<Tensor> {
    #[cfg(all(feature = "cuda", not(test)))]
    {
        let seq_len = hidden.shape().first().copied().unwrap_or(1);
        let current_layer = layer_idx.unwrap_or(0);
        let input = backend_runtime::upload_hidden_device_output_f32(
            kernels::tensor_as_f32_slice(&hidden),
            seq_len,
            metadata.hidden_dim,
        )?;
        let device_hidden = crate::engine::prefill::hidden_carrier::DevicePrefillHidden {
            output: input,
            producer_layer_idx: current_layer.saturating_sub(1),
        };
        return match forward_moe_layer_from_device_for_chain(
            metadata,
            device_hidden,
            w,
            norm_eps,
            current_layer,
            current_layer.saturating_sub(1),
        )? {
            NemotronMamba2ToMoeDeviceAttempt::Done { output, .. } => {
                let values = backend_runtime::download_nemotron_device_layer_output(output)?;
                Ok(Tensor::from_vec(values, hidden.shape()))
            }
            NemotronMamba2ToMoeDeviceAttempt::Materialize {
                device_hidden,
                reason,
            } => {
                let _ = device_hidden.output.release()?;
                Err(crate::error::LlmError::Forward(format!(
                    "CUDA Nemotron MoE execution is unavailable for layer {current_layer}: {reason}; CPU fallback is disabled"
                )))
            }
        };
    }
    #[cfg(any(not(feature = "cuda"), test))]
    {
        let input = kernels::tensor_as_f32_slice(&hidden);
        emit_carrier_tensor_trace(
            layer_idx,
            "host",
            "input",
            hidden.shape().first().copied().unwrap_or(1),
            metadata.hidden_dim,
            input,
        );
        let prepared = prepare_moe_prefill(metadata, &hidden, w, norm_eps)?;
        emit_carrier_route_trace(layer_idx, "host", &prepared.routes);
        let normed_data = kernels::tensor_as_f32_slice(&prepared.normed);
        emit_carrier_tensor_trace(
            layer_idx,
            "host",
            "normed",
            prepared.seq_len,
            prepared.hidden_dim,
            normed_data,
        );
        let mut out = vec![0.0f32; input.len()];
        let routes = &prepared.routes;
        let n_expert = prepared.n_expert;
        let n_ff = prepared.n_ff;
        let hidden_dim = prepared.hidden_dim;
        let seq_len = prepared.seq_len;
        let stage_start = (seq_len > 1).then(Instant::now);
        if let Some(output) = prefill_shared_sparse_q8_fused(
            w,
            routes,
            n_expert,
            n_ff,
            hidden_dim,
            seq_len,
            normed_data,
            input,
        )? {
            record_prefill_moe_stage("shared_sparse_fused", stage_start);
            emit_carrier_tensor_trace(layer_idx, "host", "output", seq_len, hidden_dim, &output);
            return Ok(Tensor::from_vec(output, hidden.shape()));
        }
        prefetch_sparse_q5_batch(w, routes, n_expert, n_ff, hidden_dim)?;
        let stage_start = (seq_len > 1).then(Instant::now);
        let shared_all = if let Some(shared_all) =
            prefill_shared_q8_fused(w, hidden_dim, seq_len, normed_data)?
        {
            record_prefill_moe_stage("shared_fused", stage_start);
            shared_all
        } else {
            let mut shared_mid_all = quantized_project(
                &w.shared_expert_up,
                normed_data,
                seq_len,
                "nemotron_shared_up",
            )?;
            record_prefill_moe_stage("shared_up", stage_start);
            let stage_start = (seq_len > 1).then(Instant::now);
            for value in &mut shared_mid_all {
                *value = rnb_model_nemotron::relu_sqr(*value);
            }
            record_prefill_moe_stage("shared_act", stage_start);
            let stage_start = (seq_len > 1).then(Instant::now);
            let shared_all = quantized_project(
                &w.shared_expert_down,
                &shared_mid_all,
                seq_len,
                "nemotron_shared_down",
            )?;
            record_prefill_moe_stage("shared_down", stage_start);
            shared_all
        };

        let stage_start = (seq_len > 1).then(Instant::now);
        if let Some(sparse_all) =
            prefill_sparse_q5_batch(w, routes, n_expert, n_ff, hidden_dim, seq_len, normed_data)?
        {
            record_prefill_moe_stage("sparse", stage_start);
            let stage_start = (seq_len > 1).then(Instant::now);
            out.copy_from_slice(input);
            add_f32_inplace(&mut out, &shared_all);
            add_f32_inplace(&mut out, &sparse_all);
            record_prefill_moe_stage("residual", stage_start);
            emit_carrier_tensor_trace(layer_idx, "host", "output", seq_len, hidden_dim, &out);
            return Ok(Tensor::from_vec(out, hidden.shape()));
        }

        for token_idx in 0..seq_len {
            let start = token_idx * hidden_dim;
            let route = &routes[token_idx];

            let mut token_out = shared_all[start..start + hidden_dim].to_vec();
            let h = &normed_data[start..start + hidden_dim];
            for (&expert_idx, &weight) in route.experts.iter().zip(route.weights.iter()) {
                let mut mid =
                    expert_project(&w.expert_up, expert_idx, n_expert, n_ff, hidden_dim, h)?;
                relu_sqr_f32_inplace(&mut mid);
                let down =
                    expert_project(&w.expert_down, expert_idx, n_expert, hidden_dim, n_ff, &mid)?;
                axpby_f32_inplace(&mut token_out, &down, weight, 1.0);
            }

            out[start..start + hidden_dim].copy_from_slice(&input[start..start + hidden_dim]);
            add_f32_inplace(&mut out[start..start + hidden_dim], &token_out);
        }

        emit_carrier_tensor_trace(layer_idx, "host", "output", seq_len, hidden_dim, &out);
        Ok(Tensor::from_vec(out, hidden.shape()))
    }
}

pub(in crate::engine) fn forward_moe_layer_chain_smoke(
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
    layer_idx: usize,
    next_layer_idx: Option<usize>,
    next_layer_kind: Option<&'static str>,
) -> crate::error::Result<Tensor> {
    if !device_prefill_chain_smoke_enabled() {
        return forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, Some(layer_idx));
    }
    let Some(next_layer_idx) = next_layer_idx else {
        #[cfg(feature = "cuda")]
        emit_device_prefill_chain_fallback_trace(layer_idx, None, "no_next_layer");
        return forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, Some(layer_idx));
    };
    let Some(next_layer_kind) = next_layer_kind else {
        #[cfg(feature = "cuda")]
        emit_device_prefill_chain_fallback_trace(layer_idx, Some(next_layer_idx), "no_next_layer");
        return forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, Some(layer_idx));
    };

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (layer_idx, next_layer_idx, next_layer_kind);
        return forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, Some(layer_idx));
    }

    #[cfg(feature = "cuda")]
    {
        let input = kernels::tensor_as_f32_slice(&hidden);
        let prepared = prepare_moe_prefill(metadata, &hidden, w, norm_eps)?;
        let normed_data = kernels::tensor_as_f32_slice(&prepared.normed);
        let stage_start = (prepared.seq_len > 1).then(Instant::now);

        if let Some(output) = prefill_shared_sparse_q8_handoff_download(
            w,
            &prepared,
            normed_data,
            input,
            layer_idx,
            next_layer_idx,
            next_layer_kind,
        )? {
            record_prefill_moe_stage("shared_sparse_fused", stage_start);
            return Ok(Tensor::from_vec(output, hidden.shape()));
        }

        forward_moe_layer_for_prefill(metadata, hidden, w, norm_eps, Some(layer_idx))
    }
}

fn record_prefill_moe_stage(stage: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        let key = match stage {
            "norm" => "nemotron:prefill:moe:norm",
            "router" => "nemotron:prefill:moe:router",
            "shared_up" => "nemotron:prefill:moe:shared_up",
            "shared_act" => "nemotron:prefill:moe:shared_act",
            "shared_down" => "nemotron:prefill:moe:shared_down",
            "shared_fused" => "nemotron:prefill:moe:shared_fused",
            "shared_sparse_fused" => "nemotron:prefill:moe:shared_sparse_fused",
            "routing" => "nemotron:prefill:moe:routing",
            "sparse" => "nemotron:prefill:moe:sparse",
            "residual" => "nemotron:prefill:moe:residual",
            _ => return,
        };
        crate::engine::moe_profile::record_moe_profile(key, start.elapsed());
    }
}

#[cfg(any(not(feature = "cuda"), test))]
fn prefill_shared_q8_fused(
    w: &NemotronMoELayerWeights,
    hidden_dim: usize,
    seq_len: usize,
    normed_data: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    if seq_len <= 1
        || w.shared_expert_up.ggml_type != rnb_loader::GGMLType::Q8_0
        || w.shared_expert_down.ggml_type != rnb_loader::GGMLType::Q8_0
    {
        return Ok(None);
    }
    let Some(shared_up) = w.shared_expert_up.data.as_bytes() else {
        return Ok(None);
    };
    let Some(shared_down) = w.shared_expert_down.data.as_bytes() else {
        return Ok(None);
    };
    let shared_ff = w.shared_expert_up.rows;
    if w.shared_expert_up.cols != hidden_dim
        || w.shared_expert_down.rows != hidden_dim
        || w.shared_expert_down.cols != shared_ff
    {
        return Ok(None);
    }
    backend_runtime::nemotron_q8_shared_prefill(
        shared_up,
        shared_down,
        shared_ff,
        hidden_dim,
        seq_len,
        normed_data,
    )
}

#[cfg(any(not(feature = "cuda"), test))]
fn device_prefill_probe_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL").as_deref() == Some("1")
}

#[cfg(test)]
fn device_prefill_handoff_smoke_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL_HANDOFF_SMOKE").as_deref()
        == Some("1")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in crate::engine) fn device_prefill_chain_smoke_enabled() -> bool {
    false
}

#[allow(dead_code)]
fn device_prefill_chain_between_layers_d2h_bytes(elem_count: usize) -> crate::error::Result<usize> {
    elem_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            crate::error::LlmError::Forward(
                "CUDA device prefill chain trace D2H byte length overflow".to_string(),
            )
        })
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn device_route_upload_bytes(route_slots: usize) -> usize {
    route_slots
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(route_slots.saturating_mul(std::mem::size_of::<f32>()))
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn device_route_pack_expert_major_order(expert_ids: &[u32], token_ids: &[u32]) -> Vec<usize> {
    debug_assert_eq!(expert_ids.len(), token_ids.len());
    let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
    order.sort_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
    order
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) enum NemotronMamba2ToMoeDeviceAttempt {
    Done {
        output: backend_runtime::NemotronDeviceLayerOutput,
        router_logits_d2h_bytes: usize,
        route_h2d_bytes: usize,
    },
    Materialize {
        device_hidden: crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
        reason: &'static str,
    },
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn emit_device_prefill_chain_trace(
    layer_idx: usize,
    next_layer_idx: usize,
    next_layer_kind: &'static str,
    between_layers_d2h_bytes: usize,
) {
    if crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL_TRACE").as_deref() != Some("1") {
        return;
    }

    eprintln!(
        "[cuda:device-prefill-chain] op=nemotron_moe_boundary layers={},{} next_layer_kind={} handoff_output_device=1 between_layers_d2h_bytes={} between_layers_reason=next_layer_requires_host chain_supported=1",
        layer_idx,
        next_layer_idx,
        next_layer_kind,
        between_layers_d2h_bytes
    );
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn emit_device_prefill_chain_fallback_trace(
    layer_idx: usize,
    next_layer_idx: Option<usize>,
    reason: &'static str,
) {
    if crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL_TRACE").as_deref() != Some("1") {
        return;
    }

    match next_layer_idx {
        Some(next_layer_idx) => eprintln!(
            "[cuda:device-prefill-chain] op=nemotron_moe_boundary layers={},{} chain_supported=0 reason={}",
            layer_idx, next_layer_idx, reason
        ),
        None => eprintln!(
            "[cuda:device-prefill-chain] op=nemotron_moe_boundary layers={},none chain_supported=0 reason={}",
            layer_idx, reason
        ),
    }
}

#[cfg(feature = "cuda")]
struct NemotronMoeRoutePlan<'a> {
    routes: &'a [rnb_model_nemotron::Route],
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
    seq_len: usize,
}

#[cfg(feature = "cuda")]
impl<'a> NemotronMoeRoutePlan<'a> {
    fn from_prepared(prepared: &'a NemotronMoePrefillPrepared) -> Self {
        Self {
            routes: &prepared.routes,
            n_expert: prepared.n_expert,
            n_ff: prepared.n_ff,
            hidden_dim: prepared.hidden_dim,
            seq_len: prepared.seq_len,
        }
    }
}

#[cfg(feature = "cuda")]
fn prepare_q8_shared_sparse_device_route_inputs<'a>(
    w: &'a NemotronMoELayerWeights,
    plan: &NemotronMoeRoutePlan<'_>,
    layer_idx: usize,
    trace_next_layer_idx: Option<usize>,
) -> crate::error::Result<Option<NemotronQ8SparseDeviceInputs<'a>>> {
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED")
        .as_deref()
        != Some("1")
    {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "cuda_feature_disabled",
        );
        return Ok(None);
    }
    if plan.seq_len <= 1
        || w.shared_expert_up.ggml_type != GGMLType::Q8_0
        || w.shared_expert_down.ggml_type != GGMLType::Q8_0
        || w.expert_up.ggml_type != GGMLType::Q5_0
        || w.expert_down.ggml_type != GGMLType::Q5_1
    {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    }
    let Some(shared_up) = w.shared_expert_up.data.as_bytes() else {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    };
    let Some(shared_down) = w.shared_expert_down.data.as_bytes() else {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    };
    let shared_ff = w.shared_expert_up.rows;
    if w.shared_expert_up.cols != plan.hidden_dim
        || w.shared_expert_down.rows != plan.hidden_dim
        || w.shared_expert_down.cols != shared_ff
    {
        emit_device_prefill_chain_fallback_trace(layer_idx, trace_next_layer_idx, "shape_mismatch");
        return Ok(None);
    }

    let mut up_weights = Vec::new();
    let mut down_weights = Vec::new();
    let mut expert_ids = Vec::new();
    let mut route_weights = Vec::new();
    let mut token_ids = Vec::new();
    for (token_idx, route) in plan.routes.iter().enumerate() {
        for (&expert_idx, &weight) in route.experts.iter().zip(route.weights.iter()) {
            expert_ids.push(expert_idx as u32);
            up_weights.push(expert_raw_slice(
                &w.expert_up,
                expert_idx,
                plan.n_expert,
                plan.n_ff,
            )?);
            down_weights.push(expert_raw_slice(
                &w.expert_down,
                expert_idx,
                plan.n_expert,
                plan.hidden_dim,
            )?);
            route_weights.push(weight);
            token_ids.push(token_idx as u32);
        }
    }
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_EXPERT_MAJOR").as_deref()
        != Some("0")
    {
        let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
        order.sort_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
        up_weights = order.iter().map(|&idx| up_weights[idx]).collect();
        down_weights = order.iter().map(|&idx| down_weights[idx]).collect();
        route_weights = order.iter().map(|&idx| route_weights[idx]).collect();
        token_ids = order.iter().map(|&idx| token_ids[idx]).collect();
    }

    Ok(Some((
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
    )))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn prepare_q8_shared_sparse_device_route_pack_inputs<'a>(
    w: &'a NemotronMoELayerWeights,
    expert_ids: &[u32],
    seq_len: usize,
    expert_used: usize,
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
    layer_idx: usize,
    trace_next_layer_idx: Option<usize>,
) -> crate::error::Result<Option<NemotronQ8SparseDeviceRoutePackInputs<'a>>> {
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED")
        .as_deref()
        != Some("1")
    {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "cuda_feature_disabled",
        );
        return Ok(None);
    }
    if seq_len <= 1
        || expert_used == 0
        || expert_ids.len() != seq_len.saturating_mul(expert_used)
        || w.shared_expert_up.ggml_type != GGMLType::Q8_0
        || w.shared_expert_down.ggml_type != GGMLType::Q8_0
        || w.expert_up.ggml_type != GGMLType::Q5_0
        || w.expert_down.ggml_type != GGMLType::Q5_1
    {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    }
    let Some(shared_up) = w.shared_expert_up.data.as_bytes() else {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    };
    let Some(shared_down) = w.shared_expert_down.data.as_bytes() else {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            trace_next_layer_idx,
            "quant_unsupported",
        );
        return Ok(None);
    };
    let shared_ff = w.shared_expert_up.rows;
    if w.shared_expert_up.cols != hidden_dim
        || w.shared_expert_down.rows != hidden_dim
        || w.shared_expert_down.cols != shared_ff
    {
        emit_device_prefill_chain_fallback_trace(layer_idx, trace_next_layer_idx, "shape_mismatch");
        return Ok(None);
    }

    let token_ids = (0..expert_ids.len())
        .map(|slot| (slot / expert_used) as u32)
        .collect::<Vec<_>>();
    let order = device_route_pack_expert_major_order(expert_ids, &token_ids);
    let order_indices = order
        .iter()
        .map(|&idx| {
            u32::try_from(idx).map_err(|_| {
                crate::error::LlmError::Forward(format!(
                    "Nemotron device route pack order index exceeds u32: {idx}"
                ))
            })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;

    let mut up_weights = Vec::with_capacity(expert_ids.len());
    let mut down_weights = Vec::with_capacity(expert_ids.len());
    for &expert_idx in expert_ids {
        let expert_idx = expert_idx as usize;
        if expert_idx >= n_expert {
            return Err(crate::error::LlmError::Forward(format!(
                "Nemotron device route pack expert id out of range: got {expert_idx}, expected < {n_expert}"
            )));
        }
        up_weights.push(expert_raw_slice(&w.expert_up, expert_idx, n_expert, n_ff)?);
        down_weights.push(expert_raw_slice(
            &w.expert_down,
            expert_idx,
            n_expert,
            hidden_dim,
        )?);
    }
    up_weights = order.iter().map(|&idx| up_weights[idx]).collect();
    down_weights = order.iter().map(|&idx| down_weights[idx]).collect();

    Ok(Some((
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        order_indices,
        shared_ff,
    )))
}

#[cfg(feature = "cuda")]
fn prepare_q8_shared_sparse_device_inputs<'a>(
    w: &'a NemotronMoELayerWeights,
    prepared: &NemotronMoePrefillPrepared,
    layer_idx: usize,
    trace_next_layer_idx: Option<usize>,
) -> crate::error::Result<Option<NemotronQ8SparseDeviceInputs<'a>>> {
    let plan = NemotronMoeRoutePlan::from_prepared(prepared);
    prepare_q8_shared_sparse_device_route_inputs(w, &plan, layer_idx, trace_next_layer_idx)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn release_router_logits_for_chain(
    router: &backend_runtime::NemotronDeviceRouterLogitsOutput,
) -> crate::error::Result<()> {
    match router.release_router_logits() {
        Ok(true) => Ok(()),
        Ok(false) => Err(crate::error::LlmError::Forward(format!(
            "CUDA router logits device tensor was already missing"
        ))),
        Err(err) => Err(err),
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn release_normalized_for_chain(
    router: &backend_runtime::NemotronDeviceRouterLogitsOutput,
) -> crate::error::Result<()> {
    match router.release_normalized() {
        Ok(true) => Ok(()),
        Ok(false) => Err(crate::error::LlmError::Forward(format!(
            "CUDA normalized device tensor was already missing"
        ))),
        Err(err) => Err(err),
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cleanup_route_pack_error(
    route_pack: backend_runtime::NemotronDeviceRoutePack,
    err: crate::error::LlmError,
) -> crate::error::LlmError {
    match backend_runtime::release_nemotron_device_route_pack(route_pack) {
        Ok(()) => err,
        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
            "{err}; route pack cleanup failed: {cleanup_err}"
        )),
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cleanup_router_tensors_for_chain(
    router: backend_runtime::NemotronDeviceRouterLogitsOutput,
) -> crate::error::Result<()> {
    let logits_cleanup = release_router_logits_for_chain(&router);
    let normalized_cleanup = release_normalized_for_chain(&router);
    match (logits_cleanup, normalized_cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) | (Ok(()), Err(err)) => Err(err),
        (Err(first), Err(second)) => Err(crate::error::LlmError::Forward(format!(
            "{first}; cleanup failed: {second}"
        ))),
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn download_router_logits_for_chain(
    router: &backend_runtime::NemotronDeviceRouterLogitsOutput,
) -> crate::error::Result<Vec<f32>> {
    let download = backend_runtime::download_cuda_device_tensor_f32(router.router_logits_id);
    let cleanup = release_router_logits_for_chain(router);
    match (download, cleanup) {
        (Ok(logits), Ok(())) => Ok(logits),
        (Ok(_), Err(cleanup_err)) => Err(cleanup_err),
        (Err(err), Ok(())) => Err(err),
        (Err(err), Err(cleanup_err)) => Err(crate::error::LlmError::Forward(format!(
            "{err}; cleanup failed: {cleanup_err}"
        ))),
    }
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn checked_router_logits_len_or_cleanup<ReleaseNormalized>(
    seq_len: usize,
    n_expert: usize,
    release_normalized: ReleaseNormalized,
) -> crate::error::Result<usize>
where
    ReleaseNormalized: FnOnce() -> crate::error::Result<()>,
{
    match seq_len.checked_mul(n_expert) {
        Some(len) => Ok(len),
        None => {
            let msg = "Nemotron router logits length overflow";
            match release_normalized() {
                Ok(()) => Err(crate::error::LlmError::Forward(msg.to_string())),
                Err(cleanup_err) => Err(crate::error::LlmError::Forward(format!(
                    "{msg}; cleanup failed: {cleanup_err}"
                ))),
            }
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn device_hidden_cleanup_error<Cleanup>(
    err: crate::error::LlmError,
    cleanup: Cleanup,
) -> crate::error::LlmError
where
    Cleanup: FnOnce() -> crate::error::Result<()>,
{
    match cleanup() {
        Ok(()) => err,
        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
            "{err}; device hidden cleanup failed: {cleanup_err}"
        )),
    }
}

#[cfg(feature = "cuda")]
fn release_device_hidden_for_chain(
    device_hidden: crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
) -> crate::error::Result<()> {
    match device_hidden.output.release() {
        Ok(true) => Ok(()),
        Ok(false) => Err(crate::error::LlmError::Forward(
            "CUDA device hidden tensor was already missing".to_string(),
        )),
        Err(err) => Err(err),
    }
}

#[cfg(feature = "cuda")]
fn cleanup_device_hidden_error(
    device_hidden: crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    err: crate::error::LlmError,
) -> crate::error::LlmError {
    device_hidden_cleanup_error(err, || release_device_hidden_for_chain(device_hidden))
}

#[cfg(feature = "cuda")]
fn cleanup_device_layer_output_error(
    output: backend_runtime::NemotronDeviceLayerOutput,
    err: crate::error::LlmError,
) -> crate::error::LlmError {
    match output.release() {
        Ok(true) => err,
        Ok(false) => crate::error::LlmError::Forward(format!(
            "{err}; device output cleanup failed: CUDA device output tensor was already missing"
        )),
        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
            "{err}; device output cleanup failed: {cleanup_err}"
        )),
    }
}

#[cfg(feature = "cuda")]
fn device_router_weight_f32_for_chain(
    router: &QuantizedWeight,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(test)]
    {
        if router.ggml_type == GGMLType::I32 {
            return Ok(None);
        }
        let bytes = router.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("Nemotron router weight has no bytes".to_string())
        })?;
        if router.ggml_type == GGMLType::F32 && bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(crate::error::LlmError::Forward(
                "Nemotron router F32 byte length is not divisible by four".to_string(),
            ));
        }
        return Ok(Some(crate::engine::dequant::dequantize_bytes_to_f32(
            bytes,
            router.ggml_type,
        )));
    }
    #[cfg(not(test))]
    {
        if router.ggml_type == GGMLType::I32 {
            return Ok(None);
        }
        let expected_len = router.rows.checked_mul(router.cols).ok_or_else(|| {
            crate::error::LlmError::Forward("Nemotron router shape overflow".to_string())
        })?;
        let token_ids = (0..router.rows)
            .map(|row| {
                u32::try_from(row).map_err(|_| {
                    crate::error::LlmError::Forward(format!(
                        "Nemotron router row exceeds CUDA token index range: {row}"
                    ))
                })
            })
            .collect::<crate::error::Result<Vec<_>>>()?;
        let values = router.embedding_gather_cuda(&token_ids)?;
        if values.len() != expected_len {
            return Err(crate::error::LlmError::Forward(format!(
                "Nemotron router F32 length {} != rows*cols {}",
                values.len(),
                expected_len
            )));
        }
        Ok(Some(values))
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::engine) fn forward_moe_layer_from_device_for_chain(
    metadata: &ModelMetadata,
    device_hidden: crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
    layer_idx: usize,
    trace_prev_layer_idx: usize,
) -> crate::error::Result<NemotronMamba2ToMoeDeviceAttempt> {
    let router_weight_f32 = match device_router_weight_f32_for_chain(&w.router) {
        Ok(Some(weight)) => weight,
        Ok(None) => {
            return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
                device_hidden,
                reason: "unsupported_router_quant",
            });
        }
        Err(err) => return Err(cleanup_device_hidden_error(device_hidden, err)),
    };

    let seq_len = device_hidden.output.output_desc.rows();
    let hidden_dim = metadata.hidden_dim;
    let n_expert = w.router.rows.max(metadata.expert_used_count).max(1);
    let expert_used = metadata.expert_used_count.max(1).min(n_expert);
    if crate::engine::policy::cuda_nemotron_carrier_tensor_trace_enabled() {
        match backend_runtime::download_cuda_device_tensor_f32(device_hidden.output.output_id) {
            Ok(input_copy) => emit_carrier_tensor_trace(
                Some(layer_idx),
                "device",
                "input",
                seq_len,
                hidden_dim,
                &input_copy,
            ),
            Err(err) => return Err(cleanup_device_hidden_error(device_hidden, err)),
        }
    }
    if w.expert_up.rows % n_expert != 0 {
        return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
            device_hidden,
            reason: "unsupported_moe_quant",
        });
    }
    let bias = w.router_bias.as_ref().map(kernels::tensor_as_f32_slice);
    let norm = kernels::tensor_as_f32_slice(&w.norm);

    let router = match backend_runtime::nemotron_router_logits_from_device_f32(
        device_hidden.output.output_id,
        device_hidden.output.output_desc,
        norm,
        &router_weight_f32,
        seq_len,
        hidden_dim,
        n_expert,
        norm_eps,
    ) {
        Ok(router) => router,
        Err(err) => return Err(cleanup_device_hidden_error(device_hidden, err)),
    };
    if crate::engine::policy::cuda_nemotron_carrier_tensor_trace_enabled() {
        match backend_runtime::download_cuda_device_tensor_f32(router.normalized_id) {
            Ok(normed_copy) => emit_carrier_tensor_trace(
                Some(layer_idx),
                "device",
                "normed",
                seq_len,
                hidden_dim,
                &normed_copy,
            ),
            Err(err) => {
                let err = match cleanup_router_tensors_for_chain(router) {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; router cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        }
    }
    if crate::engine::policy::cuda_nemotron_device_route_pack_enabled()
        && crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_EXPERT_MAJOR").as_deref()
            != Some("0")
    {
        let route_pack = match backend_runtime::nemotron_device_route_pack_from_logits(
            router.router_logits_id,
            router.router_logits_desc,
            bias,
            seq_len,
            n_expert,
            expert_used,
            metadata.expert_weights_scale,
        ) {
            Ok(route_pack) => route_pack,
            Err(err) => {
                let err = match cleanup_router_tensors_for_chain(router) {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; router cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        };
        let expert_ids = match backend_runtime::nemotron_device_route_pack_expert_ids(&route_pack) {
            Ok(expert_ids) => expert_ids,
            Err(err) => {
                let err = cleanup_route_pack_error(route_pack, err);
                let err = match cleanup_router_tensors_for_chain(router) {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; router cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        };
        if let Err(err) = release_router_logits_for_chain(&router) {
            let err = cleanup_route_pack_error(route_pack, err);
            let err = match release_normalized_for_chain(&router) {
                Ok(()) => err,
                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                    "{err}; normalized cleanup failed: {cleanup_err}"
                )),
            };
            return Err(cleanup_device_hidden_error(device_hidden, err));
        }
        let n_ff = w.expert_up.rows / n_expert;
        let (shared_up, shared_down, up_weights, down_weights, order_indices, shared_ff) =
            match prepare_q8_shared_sparse_device_route_pack_inputs(
                w,
                &expert_ids,
                seq_len,
                expert_used,
                n_expert,
                n_ff,
                hidden_dim,
                layer_idx,
                Some(trace_prev_layer_idx),
            ) {
                Ok(Some(inputs)) => inputs,
                Ok(None) => {
                    let err = match backend_runtime::release_nemotron_device_route_pack(route_pack)
                    {
                        Ok(()) => match release_normalized_for_chain(&router) {
                            Ok(()) => None,
                            Err(err) => Some(err),
                        },
                        Err(err) => Some(err),
                    };
                    if let Some(err) = err {
                        return Err(cleanup_device_hidden_error(device_hidden, err));
                    }
                    return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
                        device_hidden,
                        reason: "unsupported_moe_quant",
                    });
                }
                Err(err) => {
                    let err = cleanup_route_pack_error(route_pack, err);
                    let err = match release_normalized_for_chain(&router) {
                        Ok(()) => err,
                        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                            "{err}; normalized cleanup failed: {cleanup_err}"
                        )),
                    };
                    return Err(cleanup_device_hidden_error(device_hidden, err));
                }
            };
        let sorted_route_pack = match backend_runtime::nemotron_reorder_device_route_pack(
            &route_pack,
            &order_indices,
        ) {
            Ok(sorted_route_pack) => sorted_route_pack,
            Err(err) => {
                let err = cleanup_route_pack_error(route_pack, err);
                let err = match release_normalized_for_chain(&router) {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; normalized cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        };
        if let Err(err) = backend_runtime::release_nemotron_device_route_pack(route_pack) {
            let err = cleanup_route_pack_error(sorted_route_pack, err);
            let err = match release_normalized_for_chain(&router) {
                Ok(()) => err,
                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                    "{err}; normalized cleanup failed: {cleanup_err}"
                )),
            };
            return Err(cleanup_device_hidden_error(device_hidden, err));
        }
        let output_result =
            backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack_ids(
                shared_up,
                shared_down,
                &up_weights,
                &down_weights,
                &sorted_route_pack,
                shared_ff,
                n_ff,
                hidden_dim,
                seq_len,
                router.normalized_id,
                device_hidden.output.output_id,
                device_hidden.output.output_desc,
            );
        let sorted_cleanup = backend_runtime::release_nemotron_device_route_pack(sorted_route_pack);
        let Some(output) = (match output_result {
            Ok(output) => output,
            Err(err) => {
                let err = match sorted_cleanup {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; sorted route pack cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        }) else {
            if let Err(err) = sorted_cleanup {
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
            return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
                device_hidden,
                reason: "backend_unavailable",
            });
        };
        if let Err(err) = sorted_cleanup {
            return Err(cleanup_device_layer_output_error(output, err));
        }
        if crate::engine::policy::cuda_nemotron_carrier_tensor_trace_enabled() {
            match backend_runtime::download_cuda_device_tensor_f32(output.output_id) {
                Ok(output_copy) => emit_carrier_tensor_trace(
                    Some(layer_idx),
                    "device_route_pack",
                    "output",
                    seq_len,
                    hidden_dim,
                    &output_copy,
                ),
                Err(err) => return Err(cleanup_device_layer_output_error(output, err)),
            }
        }
        if crate::engine::models::nemotron::mamba::device_prefill_trace_enabled() {
            eprintln!(
                "[cuda:device-prefill-chain] op=nemotron_mamba2_moe_boundary layers={},{} hidden_d2h_bytes=0 router_logits_d2h_bytes=0 route_h2d_bytes=0 reason=device_router_pack",
                trace_prev_layer_idx, layer_idx
            );
        }
        return Ok(NemotronMamba2ToMoeDeviceAttempt::Done {
            output,
            router_logits_d2h_bytes: 0,
            route_h2d_bytes: 0,
        });
    }
    let router_logits_d2h_bytes = match router.router_logits_desc.byte_len() {
        Some(bytes) => bytes,
        None => {
            let err = crate::error::LlmError::Forward(
                "Nemotron router logits byte length overflow".to_string(),
            );
            let err = match cleanup_router_tensors_for_chain(router) {
                Ok(()) => err,
                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                    "{err}; router cleanup failed: {cleanup_err}"
                )),
            };
            return Err(cleanup_device_hidden_error(device_hidden, err));
        }
    };
    let logits = match download_router_logits_for_chain(&router) {
        Ok(logits) => logits,
        Err(err) => {
            let err = match release_normalized_for_chain(&router) {
                Ok(()) => err,
                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                    "{err}; normalized cleanup failed: {cleanup_err}"
                )),
            };
            return Err(cleanup_device_hidden_error(device_hidden, err));
        }
    };
    let expected_logits = match checked_router_logits_len_or_cleanup(seq_len, n_expert, || {
        release_normalized_for_chain(&router)
    }) {
        Ok(expected_logits) => expected_logits,
        Err(err) => return Err(cleanup_device_hidden_error(device_hidden, err)),
    };
    if logits.len() != expected_logits {
        let err = crate::error::LlmError::Forward(format!(
            "Nemotron router logits length mismatch: got {}, expected {}",
            logits.len(),
            expected_logits
        ));
        let err = match release_normalized_for_chain(&router) {
            Ok(()) => err,
            Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                "{err}; normalized cleanup failed: {cleanup_err}"
            )),
        };
        return Err(cleanup_device_hidden_error(device_hidden, err));
    }

    let routes = (0..seq_len)
        .map(|token_idx| {
            route_token(
                &logits[token_idx * n_expert..(token_idx + 1) * n_expert],
                bias,
                expert_used,
                metadata.expert_weights_scale,
            )
        })
        .collect::<Vec<_>>();
    emit_carrier_route_trace(Some(layer_idx), "device", &routes);
    let plan = NemotronMoeRoutePlan {
        routes: &routes,
        n_expert,
        n_ff: w.expert_up.rows / n_expert,
        hidden_dim,
        seq_len,
    };
    let (shared_up, shared_down, up_weights, down_weights, route_weights, token_ids, shared_ff) =
        match prepare_q8_shared_sparse_device_route_inputs(
            w,
            &plan,
            layer_idx,
            Some(trace_prev_layer_idx),
        ) {
            Ok(Some(inputs)) => inputs,
            Ok(None) => {
                if let Err(err) = release_normalized_for_chain(&router) {
                    return Err(cleanup_device_hidden_error(device_hidden, err));
                }
                return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
                    device_hidden,
                    reason: "unsupported_moe_quant",
                });
            }
            Err(err) => {
                let err = match release_normalized_for_chain(&router) {
                    Ok(()) => err,
                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                        "{err}; normalized cleanup failed: {cleanup_err}"
                    )),
                };
                return Err(cleanup_device_hidden_error(device_hidden, err));
            }
        };
    let route_h2d_bytes = device_route_upload_bytes(route_weights.len());
    let Some(output) = (match backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_ids(
        shared_up,
        shared_down,
        &up_weights,
        &down_weights,
        &route_weights,
        &token_ids,
        shared_ff,
        plan.n_ff,
        hidden_dim,
        seq_len,
        router.normalized_id,
        device_hidden.output.output_id,
        device_hidden.output.output_desc,
    ) {
        Ok(output) => output,
        Err(err) => return Err(cleanup_device_hidden_error(device_hidden, err)),
    }) else {
        return Ok(NemotronMamba2ToMoeDeviceAttempt::Materialize {
            device_hidden,
            reason: "backend_unavailable",
        });
    };
    if crate::engine::policy::cuda_nemotron_carrier_tensor_trace_enabled() {
        match backend_runtime::download_cuda_device_tensor_f32(output.output_id) {
            Ok(output_copy) => emit_carrier_tensor_trace(
                Some(layer_idx),
                "device",
                "output",
                seq_len,
                hidden_dim,
                &output_copy,
            ),
            Err(err) => return Err(cleanup_device_layer_output_error(output, err)),
        }
    }
    if crate::engine::models::nemotron::mamba::device_prefill_trace_enabled() {
        eprintln!(
            "[cuda:device-prefill-chain] op=nemotron_mamba2_moe_boundary layers={},{} hidden_d2h_bytes=0 router_logits_d2h_bytes={} route_h2d_bytes={} reason=device_router_host_topk",
            trace_prev_layer_idx, layer_idx, router_logits_d2h_bytes, route_h2d_bytes
        );
    }
    Ok(NemotronMamba2ToMoeDeviceAttempt::Done {
        output,
        router_logits_d2h_bytes,
        route_h2d_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn prefill_shared_sparse_q8_handoff_download(
    w: &NemotronMoELayerWeights,
    prepared: &NemotronMoePrefillPrepared,
    normed_data: &[f32],
    residual: &[f32],
    layer_idx: usize,
    next_layer_idx: usize,
    next_layer_kind: &'static str,
) -> crate::error::Result<Option<Vec<f32>>> {
    if residual.len() != prepared.seq_len * prepared.hidden_dim {
        emit_device_prefill_chain_fallback_trace(layer_idx, Some(next_layer_idx), "shape_mismatch");
        return Ok(None);
    }
    let Some((
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
    )) = prepare_q8_shared_sparse_device_inputs(w, prepared, layer_idx, Some(next_layer_idx))?
    else {
        return Ok(None);
    };

    let Some(handoff) = backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_output(
        shared_up,
        shared_down,
        &up_weights,
        &down_weights,
        &route_weights,
        &token_ids,
        shared_ff,
        prepared.n_ff,
        prepared.hidden_dim,
        prepared.seq_len,
        normed_data,
        residual,
    )?
    else {
        emit_device_prefill_chain_fallback_trace(
            layer_idx,
            Some(next_layer_idx),
            "backend_returned_none",
        );
        return Ok(None);
    };

    let output_elems = handoff
        .output_desc
        .rows()
        .checked_mul(handoff.output_desc.cols())
        .ok_or_else(|| {
            crate::error::LlmError::Forward(
                "CUDA device prefill chain trace D2H byte length overflow".to_string(),
            )
        })?;
    let between_layers_d2h_bytes = device_prefill_chain_between_layers_d2h_bytes(output_elems)?;
    let output = backend_runtime::download_nemotron_device_prefill_handoff(handoff)?;
    emit_device_prefill_chain_trace(
        layer_idx,
        next_layer_idx,
        next_layer_kind,
        between_layers_d2h_bytes,
    );
    Ok(Some(output))
}

#[cfg(any(not(feature = "cuda"), test))]
#[allow(clippy::too_many_arguments)]
fn prefill_shared_sparse_q8_fused(
    w: &NemotronMoELayerWeights,
    routes: &[rnb_model_nemotron::Route],
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
    seq_len: usize,
    normed_data: &[f32],
    residual: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED")
        .as_deref()
        != Some("1")
        || seq_len <= 1
        || w.shared_expert_up.ggml_type != rnb_loader::GGMLType::Q8_0
        || w.shared_expert_down.ggml_type != rnb_loader::GGMLType::Q8_0
        || w.expert_up.ggml_type != rnb_loader::GGMLType::Q5_0
        || w.expert_down.ggml_type != rnb_loader::GGMLType::Q5_1
    {
        return Ok(None);
    }
    let Some(shared_up) = w.shared_expert_up.data.as_bytes() else {
        return Ok(None);
    };
    let Some(shared_down) = w.shared_expert_down.data.as_bytes() else {
        return Ok(None);
    };
    let shared_ff = w.shared_expert_up.rows;
    if w.shared_expert_up.cols != hidden_dim
        || w.shared_expert_down.rows != hidden_dim
        || w.shared_expert_down.cols != shared_ff
        || residual.len() != seq_len * hidden_dim
    {
        return Ok(None);
    }

    let mut up_weights = Vec::new();
    let mut down_weights = Vec::new();
    let mut expert_ids = Vec::new();
    let mut route_weights = Vec::new();
    let mut token_ids = Vec::new();
    for (token_idx, route) in routes.iter().enumerate() {
        for (&expert_idx, &weight) in route.experts.iter().zip(route.weights.iter()) {
            expert_ids.push(expert_idx as u32);
            up_weights.push(expert_raw_slice(&w.expert_up, expert_idx, n_expert, n_ff)?);
            down_weights.push(expert_raw_slice(
                &w.expert_down,
                expert_idx,
                n_expert,
                hidden_dim,
            )?);
            route_weights.push(weight);
            token_ids.push(token_idx as u32);
        }
    }
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_EXPERT_MAJOR").as_deref()
        != Some("0")
    {
        let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
        order.sort_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
        up_weights = order.iter().map(|&idx| up_weights[idx]).collect();
        down_weights = order.iter().map(|&idx| down_weights[idx]).collect();
        route_weights = order.iter().map(|&idx| route_weights[idx]).collect();
        token_ids = order.iter().map(|&idx| token_ids[idx]).collect();
    }

    if let (Some(up_all), Some(down_all)) =
        (w.expert_up.data.as_bytes(), w.expert_down.data.as_bytes())
    {
        if let Some(output) =
            backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
                shared_up,
                shared_down,
                up_all,
                down_all,
                &expert_ids,
                &route_weights,
                &token_ids,
                shared_ff,
                n_expert,
                n_ff,
                hidden_dim,
                seq_len,
                normed_data,
                residual,
            )?
        {
            return Ok(Some(output));
        }
    }

    #[cfg(feature = "cuda")]
    if device_prefill_handoff_smoke_enabled() && !device_prefill_chain_smoke_enabled() {
        if let Some(handoff) =
            backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_output(
                shared_up,
                shared_down,
                &up_weights,
                &down_weights,
                &route_weights,
                &token_ids,
                shared_ff,
                n_ff,
                hidden_dim,
                seq_len,
                normed_data,
                residual,
            )?
        {
            return backend_runtime::download_nemotron_device_prefill_handoff(handoff).map(Some);
        }
    }

    if device_prefill_probe_enabled() {
        if let Some(output) =
            backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe_device_probe(
                shared_up,
                shared_down,
                &up_weights,
                &down_weights,
                &route_weights,
                &token_ids,
                shared_ff,
                n_ff,
                hidden_dim,
                seq_len,
                normed_data,
                residual,
            )?
        {
            return Ok(Some(output));
        }
    }

    backend_runtime::nemotron_q8_shared_q5_sparse_prefill_moe(
        shared_up,
        shared_down,
        &up_weights,
        &down_weights,
        &route_weights,
        &token_ids,
        shared_ff,
        n_ff,
        hidden_dim,
        seq_len,
        normed_data,
        residual,
    )
}

#[cfg(any(not(feature = "cuda"), test))]
#[allow(clippy::too_many_arguments)]
fn prefill_sparse_q5_batch(
    w: &NemotronMoELayerWeights,
    routes: &[rnb_model_nemotron::Route],
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
    seq_len: usize,
    normed_data: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    if w.expert_up.ggml_type != rnb_loader::GGMLType::Q5_0
        || !matches!(
            w.expert_down.ggml_type,
            rnb_loader::GGMLType::Q5_1 | rnb_loader::GGMLType::Q8_0
        )
    {
        return Ok(None);
    }

    let mut up_weights = Vec::new();
    let mut down_weights = Vec::new();
    let mut expert_ids = Vec::new();
    let mut route_weights = Vec::new();
    let mut token_ids = Vec::new();
    for (token_idx, route) in routes.iter().enumerate() {
        for (&expert_idx, &weight) in route.experts.iter().zip(route.weights.iter()) {
            expert_ids.push(expert_idx as u32);
            up_weights.push(expert_raw_slice(&w.expert_up, expert_idx, n_expert, n_ff)?);
            down_weights.push(expert_raw_slice(
                &w.expert_down,
                expert_idx,
                n_expert,
                hidden_dim,
            )?);
            route_weights.push(weight);
            token_ids.push(token_idx as u32);
        }
    }
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_EXPERT_MAJOR").as_deref()
        != Some("0")
    {
        let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
        order.sort_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
        up_weights = order.iter().map(|&idx| up_weights[idx]).collect();
        down_weights = order.iter().map(|&idx| down_weights[idx]).collect();
        expert_ids = order.iter().map(|&idx| expert_ids[idx]).collect();
        route_weights = order.iter().map(|&idx| route_weights[idx]).collect();
        token_ids = order.iter().map(|&idx| token_ids[idx]).collect();
    }

    if w.expert_down.ggml_type == rnb_loader::GGMLType::Q5_1 {
        if let (Some(up_all), Some(down_all)) =
            (w.expert_up.data.as_bytes(), w.expert_down.data.as_bytes())
        {
            if let Some(output) = backend_runtime::nemotron_q5_sparse_relu_sqr_full_layer_by_token(
                up_all,
                down_all,
                &expert_ids,
                &route_weights,
                &token_ids,
                seq_len,
                n_expert,
                n_ff,
                hidden_dim,
                normed_data,
            )? {
                return Ok(Some(output));
            }
        }
    } else if w.expert_down.ggml_type == rnb_loader::GGMLType::Q8_0 {
        if let (Some(up_all), Some(down_all)) =
            (w.expert_up.data.as_bytes(), w.expert_down.data.as_bytes())
        {
            if let Some(output) =
                backend_runtime::nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
                    up_all,
                    down_all,
                    &expert_ids,
                    &route_weights,
                    &token_ids,
                    seq_len,
                    n_expert,
                    n_ff,
                    hidden_dim,
                    normed_data,
                )?
            {
                return Ok(Some(output));
            }
        }
    }

    if w.expert_down.ggml_type == rnb_loader::GGMLType::Q8_0 {
        let output = backend_runtime::nemotron_q5_q8_sparse_relu_sqr_by_token(
            &up_weights,
            &down_weights,
            &route_weights,
            &token_ids,
            seq_len,
            n_ff,
            hidden_dim,
            normed_data,
        )?;
        prewarm_prefill_routes(&up_weights, &down_weights, &expert_ids, &token_ids, seq_len)?;
        Ok(output)
    } else {
        let output = backend_runtime::nemotron_q5_sparse_relu_sqr_by_token(
            &up_weights,
            &down_weights,
            &route_weights,
            &token_ids,
            seq_len,
            n_ff,
            hidden_dim,
            normed_data,
        )?;
        prewarm_prefill_routes(&up_weights, &down_weights, &expert_ids, &token_ids, seq_len)?;
        Ok(output)
    }
}

#[cfg(any(not(feature = "cuda"), test))]
fn prefetch_sparse_q5_batch(
    w: &NemotronMoELayerWeights,
    routes: &[rnb_model_nemotron::Route],
    n_expert: usize,
    n_ff: usize,
    hidden_dim: usize,
) -> crate::error::Result<()> {
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_COPY_PREFETCH").as_deref()
        != Some("1")
    {
        return Ok(());
    }
    if w.expert_up.ggml_type != rnb_loader::GGMLType::Q5_0
        || !matches!(
            w.expert_down.ggml_type,
            rnb_loader::GGMLType::Q5_1 | rnb_loader::GGMLType::Q8_0
        )
    {
        return Ok(());
    }

    let mut up_weights = Vec::new();
    let mut down_weights = Vec::new();
    let mut expert_ids = Vec::new();
    let mut token_ids = Vec::new();
    for (token_idx, route) in routes.iter().enumerate() {
        for &expert_idx in &route.experts {
            expert_ids.push(expert_idx as u32);
            up_weights.push(expert_raw_slice(&w.expert_up, expert_idx, n_expert, n_ff)?);
            down_weights.push(expert_raw_slice(
                &w.expert_down,
                expert_idx,
                n_expert,
                hidden_dim,
            )?);
            token_ids.push(token_idx as u32);
        }
    }
    if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_EXPERT_MAJOR").as_deref()
        != Some("0")
    {
        let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
        order.sort_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
        up_weights = order.iter().map(|&idx| up_weights[idx]).collect();
        down_weights = order.iter().map(|&idx| down_weights[idx]).collect();
    }

    backend_runtime::nemotron_prefill_sparse_copy_prefetch(
        &up_weights,
        &down_weights,
        n_ff,
        hidden_dim,
        w.expert_down.ggml_type == rnb_loader::GGMLType::Q8_0,
    )?;
    Ok(())
}

#[cfg(any(not(feature = "cuda"), test))]
fn prewarm_prefill_routes<'a>(
    up_weights: &[&'a [u8]],
    down_weights: &[&'a [u8]],
    expert_ids: &[u32],
    token_ids: &[u32],
    seq_len: usize,
) -> crate::error::Result<()> {
    let last_route_enabled =
        crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_LAST_ROUTE_PREWARM")
            .as_deref()
            != Some("0");
    let topk = crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_ROUTE_PREWARM_TOPK")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if !last_route_enabled && topk == 0 {
        return Ok(());
    }

    let last_token = seq_len.saturating_sub(1) as u32;
    let mut weights = Vec::<&'a [u8]>::new();
    let mut seen_weights = HashSet::new();

    if last_route_enabled {
        for (slot, &token_id) in token_ids.iter().enumerate() {
            if token_id == last_token {
                push_prewarm_slot(
                    up_weights,
                    down_weights,
                    slot,
                    &mut seen_weights,
                    &mut weights,
                );
            }
        }
    }

    if topk > 0 {
        let mut counts = HashMap::<u32, (usize, usize)>::new();
        for (slot, &expert_id) in expert_ids.iter().enumerate() {
            counts
                .entry(expert_id)
                .and_modify(|(count, _)| *count += 1)
                .or_insert((1, slot));
        }
        let mut ranked = counts.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|(left_id, (left_count, _)), (right_id, (right_count, _))| {
            right_count
                .cmp(left_count)
                .then_with(|| left_id.cmp(right_id))
        });
        for (_, (_, slot)) in ranked.into_iter().take(topk) {
            push_prewarm_slot(
                up_weights,
                down_weights,
                slot,
                &mut seen_weights,
                &mut weights,
            );
        }
    }

    backend_runtime::prewarm_q4k_weight_slices(&weights)
}

#[cfg(any(not(feature = "cuda"), test))]
fn push_prewarm_slot<'a>(
    up_weights: &[&'a [u8]],
    down_weights: &[&'a [u8]],
    slot: usize,
    seen_weights: &mut HashSet<(usize, usize)>,
    weights: &mut Vec<&'a [u8]>,
) {
    for weight in [up_weights[slot], down_weights[slot]] {
        if seen_weights.insert((weight.as_ptr() as usize, weight.len())) {
            weights.push(weight);
        }
    }
}

fn router_logits(
    router: &QuantizedWeight,
    normed_data: &[f32],
    seq_len: usize,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] hidden_dim: usize,
    n_expert: usize,
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    if router.ggml_type == GGMLType::F32 {
        if let Some(bytes) = router.data.as_bytes() {
            let weights = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            if weights.len() == n_expert * hidden_dim {
                return backend_runtime::qwen_moe_prefill_router_logits(
                    weights,
                    n_expert,
                    hidden_dim,
                    normed_data,
                );
            }
        }
    }
    let logits = quantized_project(router, normed_data, seq_len, "nemotron_router")?;
    if logits.len() != seq_len * n_expert {
        return Err(crate::error::LlmError::Forward(format!(
            "Nemotron-H router logits len {} != expected {}",
            logits.len(),
            seq_len * n_expert
        )));
    }
    Ok(logits)
}

pub(in crate::engine) fn decode_moe_layer(
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
) -> crate::error::Result<()> {
    let hidden_dim = metadata.hidden_dim;
    if let Some(output) =
        decode_moe_layer_combined_q5(metadata, &scratch.hidden[..hidden_dim], w, norm_eps)?
    {
        scratch.hidden[..hidden_dim].copy_from_slice(&output[..hidden_dim]);
        return Ok(());
    }
    let input = Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]);
    let output = forward_moe_layer(metadata, input, w, norm_eps)?;
    let output_data = kernels::tensor_as_f32_slice(&output);
    scratch.hidden[..hidden_dim].copy_from_slice(&output_data[..hidden_dim]);
    Ok(())
}

fn decode_moe_layer_combined_q5(
    metadata: &ModelMetadata,
    input: &[f32],
    w: &NemotronMoELayerWeights,
    norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    if w.expert_up.ggml_type != rnb_loader::GGMLType::Q5_0
        || w.expert_down.ggml_type != rnb_loader::GGMLType::Q5_1
    {
        return Ok(None);
    }
    let hidden_dim = metadata.hidden_dim;
    let n_expert = w.router.rows.max(metadata.expert_used_count).max(1);
    let n_ff = w.expert_up.rows / n_expert;
    if input.len() != hidden_dim || n_ff == 0 {
        return Ok(None);
    }

    let hidden = Tensor::from_slice(input, &[1, hidden_dim]);
    let stage_start = std::time::Instant::now();
    let normed = apply_plain_rms_norm(&hidden, &w.norm, norm_eps)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?;
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:norm",
        stage_start.elapsed(),
    );
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    let stage_start = std::time::Instant::now();
    let router_logits = router_logits(&w.router, normed_data, 1, hidden_dim, n_expert)?;
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:router",
        stage_start.elapsed(),
    );
    let bias = w.router_bias.as_ref().map(kernels::tensor_as_f32_slice);
    let stage_start = std::time::Instant::now();
    let route = route_token(
        &router_logits[..n_expert],
        bias,
        metadata.expert_used_count.max(1).min(n_expert),
        metadata.expert_weights_scale,
    );
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:routing",
        stage_start.elapsed(),
    );
    let shared_up = w.shared_expert_up.data.as_bytes();
    let shared_down = w.shared_expert_down.data.as_bytes();
    let (Some(shared_up), Some(shared_down)) = (shared_up, shared_down) else {
        return Ok(None);
    };
    let stage_start = std::time::Instant::now();
    let mut up_weights = Vec::with_capacity(route.experts.len());
    let mut down_weights = Vec::with_capacity(route.experts.len());
    for &expert_idx in &route.experts {
        up_weights.push(expert_raw_slice(&w.expert_up, expert_idx, n_expert, n_ff)?);
        down_weights.push(expert_raw_slice(
            &w.expert_down,
            expert_idx,
            n_expert,
            hidden_dim,
        )?);
    }
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:slice",
        stage_start.elapsed(),
    );

    let stage_start = std::time::Instant::now();
    let moe_out = if backend_runtime::nemotron_q8_shared_q5_sparse_decode_enabled()
        && w.shared_expert_up.ggml_type == rnb_loader::GGMLType::Q8_0
        && w.shared_expert_down.ggml_type == rnb_loader::GGMLType::Q8_0
    {
        let shared_ff = w.shared_expert_up.rows;
        if w.expert_up.ggml_type == rnb_loader::GGMLType::Q5_0
            && w.expert_down.ggml_type == rnb_loader::GGMLType::Q5_1
        {
            if let (Some(up_all), Some(down_all)) =
                (w.expert_up.data.as_bytes(), w.expert_down.data.as_bytes())
            {
                if let Some(output) =
                    backend_runtime::nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
                        shared_up,
                        shared_down,
                        up_all,
                        down_all,
                        &route
                            .experts
                            .iter()
                            .map(|&expert| expert as u32)
                            .collect::<Vec<_>>(),
                        &route.weights,
                        shared_ff,
                        n_expert,
                        n_ff,
                        hidden_dim,
                        normed_data,
                    )?
                {
                    Some(output)
                } else {
                    backend_runtime::nemotron_q8_shared_q5_sparse_decode_moe(
                        shared_up,
                        shared_down,
                        &up_weights,
                        &down_weights,
                        &route.weights,
                        shared_ff,
                        n_ff,
                        hidden_dim,
                        normed_data,
                    )?
                }
            } else {
                backend_runtime::nemotron_q8_shared_q5_sparse_decode_moe(
                    shared_up,
                    shared_down,
                    &up_weights,
                    &down_weights,
                    &route.weights,
                    shared_ff,
                    n_ff,
                    hidden_dim,
                    normed_data,
                )?
            }
        } else {
            backend_runtime::nemotron_q8_shared_q5_sparse_decode_moe(
                shared_up,
                shared_down,
                &up_weights,
                &down_weights,
                &route.weights,
                shared_ff,
                n_ff,
                hidden_dim,
                normed_data,
            )?
        }
    } else if w.shared_expert_up.ggml_type == rnb_loader::GGMLType::Q5_0
        && w.shared_expert_down.ggml_type == rnb_loader::GGMLType::Q5_1
    {
        backend_runtime::nemotron_q5_decode_moe_shared_sparse(
            shared_up,
            shared_down,
            &up_weights,
            &down_weights,
            &route.weights,
            n_ff,
            hidden_dim,
            normed_data,
        )?
    } else {
        None
    };
    let Some(moe_out) = moe_out else {
        return Ok(None);
    };
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:cuda_shared_sparse",
        stage_start.elapsed(),
    );
    let stage_start = std::time::Instant::now();
    let mut output = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        output[i] = input[i] + moe_out[i];
    }
    crate::engine::moe_profile::record_moe_profile(
        "nemotron:decode:moe:residual",
        stage_start.elapsed(),
    );
    Ok(Some(output))
}

fn route_token(
    logits: &[f32],
    bias: Option<&[f32]>,
    expert_used: usize,
    expert_weights_scale: f32,
) -> rnb_model_nemotron::Route {
    rnb_model_nemotron::sigmoid_topk_route(logits, bias, expert_used, expert_weights_scale)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CarrierTensorSummary {
    elems: usize,
    finite: usize,
    min: f32,
    max: f32,
    first: f32,
    last: f32,
    checksum: f64,
    bit_hash: u64,
}

fn carrier_route_hash(routes: &[rnb_model_nemotron::Route]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for route in routes {
        for &expert in &route.experts {
            hash ^= expert as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for &weight in &route.weights {
            hash ^= weight.to_bits() as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn carrier_tensor_summary(values: &[f32]) -> CarrierTensorSummary {
    let mut finite = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut checksum = 0.0f64;
    let mut bit_hash = 0xcbf29ce484222325_u64;
    for &value in values {
        bit_hash ^= value.to_bits() as u64;
        bit_hash = bit_hash.wrapping_mul(0x100000001b3);
        if value.is_finite() {
            finite += 1;
            min = min.min(value);
            max = max.max(value);
            checksum += value as f64;
        }
    }
    if finite == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }
    CarrierTensorSummary {
        elems: values.len(),
        finite,
        min,
        max,
        first: values.first().copied().unwrap_or(f32::NAN),
        last: values.last().copied().unwrap_or(f32::NAN),
        checksum,
        bit_hash,
    }
}

fn format_route_edge(route: Option<&rnb_model_nemotron::Route>) -> String {
    let Some(route) = route else {
        return "none".to_string();
    };
    let experts = route
        .experts
        .iter()
        .map(|expert| expert.to_string())
        .collect::<Vec<_>>()
        .join("/");
    let weights = route
        .weights
        .iter()
        .map(|weight| format!("{weight:.6}"))
        .collect::<Vec<_>>()
        .join("/");
    format!("experts={experts} weights={weights}")
}

fn emit_carrier_route_trace(
    layer_idx: Option<usize>,
    path: &'static str,
    routes: &[rnb_model_nemotron::Route],
) {
    if !crate::engine::policy::cuda_nemotron_carrier_route_trace_enabled() {
        return;
    }
    let layer = layer_idx
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    eprintln!(
        "[cuda:device-prefill-chain] op=nemotron_carrier_route_trace layer={} path={} tokens={} route_hash=0x{:016x} first=\"{}\" last=\"{}\"",
        layer,
        path,
        routes.len(),
        carrier_route_hash(routes),
        format_route_edge(routes.first()),
        format_route_edge(routes.last())
    );
}

fn emit_carrier_tensor_trace(
    layer_idx: Option<usize>,
    path: &'static str,
    stage: &'static str,
    rows: usize,
    cols: usize,
    values: &[f32],
) {
    if !crate::engine::policy::cuda_nemotron_carrier_tensor_trace_enabled() {
        return;
    }
    let layer = layer_idx
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let summary = carrier_tensor_summary(values);
    eprintln!(
        "[cuda:device-prefill-chain] op=nemotron_carrier_tensor_trace layer={} path={} stage={} rows={} cols={} elems={} finite={} min={:.9} max={:.9} first={:.9} last={:.9} checksum={:.9} bit_hash=0x{:016x}",
        layer,
        path,
        stage,
        rows,
        cols,
        summary.elems,
        summary.finite,
        summary.min,
        summary.max,
        summary.first,
        summary.last,
        summary.checksum,
        summary.bit_hash
    );
}

fn quantized_project(
    weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
    label: &str,
) -> crate::error::Result<Vec<f32>> {
    if seq_len > 1 {
        match prefill_projection_path(seq_len, "RNB_CUDA_NEMOTRON_MOE_PREFILL_Q_GEMV") {
            PrefillProjectionPath::F32Gemm => {
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection(weight, input)?
                {
                    return Ok(output);
                }
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection_q(weight, input)?
                {
                    return Ok(output);
                }
                return weight.gemv_vec(input);
            }
            PrefillProjectionPath::QuantizedGemv => {
                if let Some(output) =
                    backend_runtime::gdn_prefill_quantized_projection_q(weight, input)?
                {
                    return Ok(output);
                }
                return weight.gemv_vec(input);
            }
        }
    } else {
        let mut output = vec![0.0f32; weight.rows];
        if backend_runtime::decode_gemv_into_if_supported(weight, input, &mut output, label, false)?
        {
            return Ok(output);
        }
    }
    weight.gemv_vec(input)
}

#[cfg(any(not(feature = "cuda"), test))]
fn expert_project(
    weight: &QuantizedWeight,
    expert_idx: usize,
    n_expert: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> crate::error::Result<Vec<f32>> {
    let bytes = expert_raw_slice(weight, expert_idx, n_expert, rows)?;
    if let Some(output) =
        backend_runtime::q5_basic_gemv_raw(weight.ggml_type, bytes, rows, cols, input)?
    {
        return Ok(output);
    }
    let expert_weight = QuantizedWeight::new(
        Tensor::from_vec(bytes.to_vec(), &[bytes.len()]),
        weight.ggml_type,
        rows,
        cols,
    );
    if let Some(output) = backend_runtime::q5_basic_gemv(&expert_weight, input)? {
        return Ok(output);
    }
    expert_weight.gemv_vec(input)
}

fn expert_raw_slice(
    weight: &QuantizedWeight,
    expert_idx: usize,
    n_expert: usize,
    rows: usize,
) -> crate::error::Result<&[u8]> {
    let bytes = weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("Nemotron expert weight has no bytes".into())
    })?;
    if expert_idx >= n_expert {
        return Err(crate::error::LlmError::Forward(format!(
            "Nemotron expert index {expert_idx} out of {n_expert}"
        )));
    }
    let bytes_per_expert = bytes.len() / n_expert;
    if bytes_per_expert % rows != 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "Nemotron expert bytes {} not divisible by rows {rows}",
            bytes_per_expert
        )));
    }
    let start = expert_idx * bytes_per_expert;
    let end = start + bytes_per_expert;
    Ok(&bytes[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_loader::GGMLType;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        value: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                value: crate::engine::policy::env_os_string(key),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn device_prefill_probe_is_opt_in() {
        let _lock = env_lock().lock().expect("env lock poisoned");
        let _env_guard = EnvVarGuard::new("RNB_CUDA_DEVICE_PREFILL");
        std::env::remove_var("RNB_CUDA_DEVICE_PREFILL");

        assert!(!super::device_prefill_probe_enabled());

        std::env::set_var("RNB_CUDA_DEVICE_PREFILL", "1");
        assert!(super::device_prefill_probe_enabled());
        std::env::remove_var("RNB_CUDA_DEVICE_PREFILL");
    }

    #[test]
    fn device_prefill_handoff_smoke_is_opt_in() {
        let _lock = env_lock().lock().expect("env lock poisoned");
        let _env_guard = EnvVarGuard::new("RNB_CUDA_DEVICE_PREFILL_HANDOFF_SMOKE");
        std::env::remove_var("RNB_CUDA_DEVICE_PREFILL_HANDOFF_SMOKE");

        assert!(!super::device_prefill_handoff_smoke_enabled());

        std::env::set_var("RNB_CUDA_DEVICE_PREFILL_HANDOFF_SMOKE", "1");
        assert!(super::device_prefill_handoff_smoke_enabled());
        std::env::remove_var("RNB_CUDA_DEVICE_PREFILL_HANDOFF_SMOKE");
    }

    #[test]
    fn device_prefill_chain_between_layers_bytes_are_f32_bytes() {
        let bytes = super::device_prefill_chain_between_layers_d2h_bytes(8 * 2688)
            .expect("chain byte count");

        assert_eq!(bytes, 8 * 2688 * std::mem::size_of::<f32>());
    }

    #[test]
    fn device_prefill_chain_between_layers_bytes_detect_overflow() {
        let err = super::device_prefill_chain_between_layers_d2h_bytes(usize::MAX)
            .expect_err("overflow should fail");

        assert!(err
            .to_string()
            .contains("chain trace D2H byte length overflow"));
    }

    #[test]
    fn device_route_upload_bytes_count_indices_and_weights() {
        assert_eq!(super::device_route_upload_bytes(0), 0);
        assert_eq!(super::device_route_upload_bytes(3), 3 * 4 + 3 * 4);
    }

    #[test]
    fn carrier_route_trace_hash_changes_when_route_changes() {
        let left = vec![rnb_model_nemotron::Route {
            experts: vec![1, 7],
            weights: vec![0.75, 0.25],
        }];
        let right = vec![rnb_model_nemotron::Route {
            experts: vec![1, 8],
            weights: vec![0.75, 0.25],
        }];

        assert_ne!(
            super::carrier_route_hash(&left),
            super::carrier_route_hash(&right)
        );
    }

    #[test]
    fn carrier_tensor_summary_records_finite_stats() {
        let summary = super::carrier_tensor_summary(&[1.0, -3.0, 2.0, f32::NAN]);

        assert_eq!(summary.elems, 4);
        assert_eq!(summary.finite, 3);
        assert_eq!(summary.min, -3.0);
        assert_eq!(summary.max, 2.0);
        assert_eq!(summary.first, 1.0);
        assert!(summary.last.is_nan());
        assert_ne!(
            summary.bit_hash,
            super::carrier_tensor_summary(&[1.0, -3.0, 2.0, 0.0]).bit_hash
        );
    }

    #[test]
    fn device_route_pack_order_sorts_by_expert_then_token() {
        let expert_ids = [4_u32, 1, 4, 0, 1, 0];
        let token_ids = [0_u32, 0, 1, 1, 2, 2];
        let order = super::device_route_pack_expert_major_order(&expert_ids, &token_ids);

        assert_eq!(order, vec![3, 5, 1, 4, 0, 2]);
    }

    #[test]
    fn router_logits_len_overflow_releases_normalized_before_error() {
        let released = std::cell::Cell::new(false);

        let err = super::checked_router_logits_len_or_cleanup(usize::MAX, 2, || {
            released.set(true);
            Ok(())
        })
        .expect_err("overflow should fail");

        assert!(released.get());
        assert!(err.to_string().contains("router logits length overflow"));
    }

    #[test]
    fn router_logits_len_overflow_reports_cleanup_failure() {
        let err = super::checked_router_logits_len_or_cleanup(usize::MAX, 2, || {
            Err(crate::error::LlmError::Forward(
                "normalized cleanup failed".to_string(),
            ))
        })
        .expect_err("overflow should fail");

        let msg = err.to_string();
        assert!(msg.contains("router logits length overflow"));
        assert!(msg.contains("cleanup failed"));
        assert!(msg.contains("normalized cleanup failed"));
    }

    #[test]
    fn device_hidden_cleanup_error_preserves_primary_error() {
        let err = super::device_hidden_cleanup_error(
            crate::error::LlmError::Forward("primary failure".to_string()),
            || {
                Err(crate::error::LlmError::Forward(
                    "input cleanup failed".to_string(),
                ))
            },
        );

        let msg = err.to_string();
        assert!(msg.contains("primary failure"));
        assert!(msg.contains("input cleanup failed"));
    }

    #[test]
    fn nemotron_moe_forward_adds_shared_and_routed_experts_to_residual() {
        let metadata = ModelMetadata {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size: 1,
            max_seq_len: 1,
            hidden_dim: 2,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 1.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            head_count_kv_per_layer: None,
            embedding_length_per_layer_input: 0,
            expert_used_count: 0,
            expert_weights_scale: 1.0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
        };
        let weights = NemotronMoELayerWeights {
            norm: Tensor::from_slice(&[1.0, 1.0], &[2]),
            router: qweight(&[1.0, 0.0], 1, 2),
            router_bias: None,
            expert_down: qweight(&[1.0, 0.0], 2, 1),
            expert_up: qweight(&[2.0, 0.0], 1, 2),
            shared_expert_down: qweight(&[0.5, 0.0], 2, 1),
            shared_expert_up: qweight(&[1.0, 0.0], 1, 2),
            latent_down: None,
            latent_up: None,
        };

        let out = forward_moe_layer(
            &metadata,
            Tensor::from_slice(&[1.0, 0.0], &[1, 2]),
            &weights,
            0.0,
        )
        .unwrap();

        let data = kernels::tensor_as_f32_slice(&out);
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert_eq!(data[1], 0.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_router_weight_f32_reads_f32_router() {
        let router = qweight(&[1.25, -2.5, 3.75, 4.5], 2, 2);

        let prepared = device_router_weight_f32_for_chain(&router)
            .unwrap()
            .expect("F32 router is supported");

        assert_eq!(prepared, vec![1.25, -2.5, 3.75, 4.5]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_router_weight_f32_rejects_malformed_f32_bytes() {
        let router =
            QuantizedWeight::new(Tensor::from_vec(vec![1u8, 2, 3], &[3]), GGMLType::F32, 1, 1);

        let err = device_router_weight_f32_for_chain(&router).unwrap_err();

        assert!(err
            .to_string()
            .contains("Nemotron router F32 byte length is not divisible by four"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_router_weight_f32_dequantizes_q80_router() {
        let mut bytes = Vec::with_capacity(34);
        bytes.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        let mut qs = [0i8; 32];
        qs[0] = 2;
        qs[1] = -4;
        qs[2] = 7;
        bytes.extend(qs.iter().map(|&q| q as u8));
        let router = QuantizedWeight::new(Tensor::from_vec(bytes, &[34]), GGMLType::Q8_0, 1, 32);

        let prepared = device_router_weight_f32_for_chain(&router)
            .unwrap()
            .expect("Q8_0 router is supported");

        assert_eq!(prepared.len(), 32);
        assert_eq!(prepared[0], 1.0);
        assert_eq!(prepared[1], -2.0);
        assert_eq!(prepared[2], 3.5);
    }

    fn qweight(data: &[f32], rows: usize, cols: usize) -> QuantizedWeight {
        QuantizedWeight::new(
            Tensor::from_slice(data, &[rows, cols]),
            GGMLType::F32,
            rows,
            cols,
        )
    }
}
