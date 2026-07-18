//! GLM-DSA layer-major sparse-MoE prefill.

use super::moe_types::{expert_bytes_per_row, SharedExpertMoEView};
use super::moe_view::shared_expert::compute_shared_expert;
use super::page_cache::SparseExpertPageCache;
use super::routing::hy3_sigmoid_topk_route;
use crate::engine::backend_runtime;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::norm::{apply_model_gate_mul_inplace, apply_model_norm_into};
use crate::engine::{cuda_runtime, policy, ModelArchitecture};
use crate::error::{LlmError, Result};

pub(super) fn enabled() -> bool {
    std::env::var("RNB_CUDA_GLM_BATCH_MOE")
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "off" | "no"))
}

fn compute_shared_batch(
    view: &SharedExpertMoEView<'_>,
    normalized: &[f32],
    seq_len: usize,
) -> Result<Option<Vec<f32>>> {
    let batch_enabled = std::env::var("RNB_CUDA_GLM_BATCH_SHARED")
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "off" | "no"));
    if !batch_enabled {
        return Ok(None);
    }
    let Some(gate_result) = cuda_runtime::prefill_gemv(
        view.shared_gate_quant,
        view.shared_gate_bytes,
        view.n_ff,
        view.n_embd,
        normalized,
        seq_len,
    ) else {
        return Ok(None);
    };
    let Some(up_result) = cuda_runtime::prefill_gemv(
        view.shared_up_quant,
        view.shared_up_bytes,
        view.n_ff,
        view.n_embd,
        normalized,
        seq_len,
    ) else {
        return Ok(None);
    };
    let mut gate = gate_result.map_err(LlmError::Forward)?;
    let up = up_result.map_err(LlmError::Forward)?;
    if gate.len() != seq_len.saturating_mul(view.n_ff) || up.len() != gate.len() {
        return Err(LlmError::Forward(format!(
            "GLM batched shared gate/up mismatch: gate={} up={} expected={}",
            gate.len(),
            up.len(),
            seq_len.saturating_mul(view.n_ff)
        )));
    }
    for token in 0..seq_len {
        let start = token * view.n_ff;
        apply_model_gate_mul_inplace(
            &mut gate[start..start + view.n_ff],
            &up[start..start + view.n_ff],
            ModelArchitecture::GlmDsa,
        );
    }
    let Some(down_result) = cuda_runtime::prefill_gemv(
        view.shared_down_quant,
        view.shared_down_bytes,
        view.n_embd,
        view.n_ff,
        &gate,
        seq_len,
    ) else {
        return Ok(None);
    };
    let mut down = down_result.map_err(LlmError::Forward)?;
    if down.len() != seq_len.saturating_mul(view.n_embd) {
        return Err(LlmError::Forward(format!(
            "GLM batched shared down mismatch: got={} expected={}",
            down.len(),
            seq_len.saturating_mul(view.n_embd)
        )));
    }
    if view.shared_expert_gated {
        for token in 0..seq_len {
            let hidden_start = token * view.n_embd;
            let gate_dot = normalized[hidden_start..hidden_start + view.n_embd]
                .iter()
                .zip(view.shared_input_scale.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let scalar = 1.0 / (1.0 + (-gate_dot).exp());
            for value in &mut down[hidden_start..hidden_start + view.n_embd] {
                *value *= scalar;
            }
        }
    }
    Ok(Some(down))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward(
    view: &SharedExpertMoEView<'_>,
    residual: &mut [f32],
    ffn_norm: &[f32],
    norm_eps: f32,
    seq_len: usize,
    hidden_dim: usize,
    page_cache: Option<&SparseExpertPageCache>,
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
    direct_file: bool,
) -> Result<()> {
    let selected_count = view.n_expert_used.min(view.n_expert);
    if selected_count == 0 || view.expert_gating_func != 2 {
        return Err(LlmError::Forward(
            "GLM batched sparse prefill requires sigmoid top-k experts".into(),
        ));
    }
    let selection_bias = view.router_selection_bias.ok_or_else(|| {
        LlmError::Forward("GLM batched sparse prefill requires router selection bias".into())
    })?;
    if residual.len() != seq_len.saturating_mul(hidden_dim) {
        return Err(LlmError::Forward(format!(
            "GLM batched sparse residual mismatch: got={} expected={}",
            residual.len(),
            seq_len.saturating_mul(hidden_dim)
        )));
    }

    let mut normalized = vec![0.0f32; residual.len()];
    for token in 0..seq_len {
        let start = token * hidden_dim;
        apply_model_norm_into(
            &residual[start..start + hidden_dim],
            ffn_norm,
            norm_eps,
            &mut normalized[start..start + hidden_dim],
            ModelArchitecture::GlmDsa,
        );
    }

    let mut router_logits = vec![0.0f32; seq_len * view.n_expert];
    gemv_f32(
        view.router_w,
        &normalized,
        &mut router_logits,
        view.n_expert,
        hidden_dim,
        seq_len,
    );

    let gate_bytes_per_expert = view.n_ff.saturating_mul(expert_bytes_per_row(
        hidden_dim,
        view.gate_quant,
        "gate_exps",
    ));
    let up_bytes_per_expert =
        view.n_ff
            .saturating_mul(expert_bytes_per_row(hidden_dim, view.up_quant, "up_exps"));
    let down_bytes_per_expert = hidden_dim.saturating_mul(expert_bytes_per_row(
        view.n_ff,
        view.down_quant,
        "down_exps",
    ));
    let expected_gate = view.n_expert.saturating_mul(gate_bytes_per_expert);
    let expected_up = view.n_expert.saturating_mul(up_bytes_per_expert);
    let expected_down = view.n_expert.saturating_mul(down_bytes_per_expert);
    if view.gate_exps_bytes.len() < expected_gate
        || view.up_exps_bytes.len() < expected_up
        || view.down_exps_bytes.len() < expected_down
    {
        return Err(LlmError::Forward(format!(
            "GLM batched sparse expert bytes mismatch: gate={}/{} up={}/{} down={}/{}",
            view.gate_exps_bytes.len(),
            expected_gate,
            view.up_exps_bytes.len(),
            expected_up,
            view.down_exps_bytes.len(),
            expected_down
        )));
    }

    let slot_count = seq_len.saturating_mul(selected_count);
    let mut gate_slots = Vec::with_capacity(slot_count);
    let mut up_slots = Vec::with_capacity(slot_count);
    let mut down_slots = Vec::with_capacity(slot_count);
    let mut route_weights = Vec::with_capacity(slot_count);
    let mut token_ids = Vec::with_capacity(slot_count);
    let mut selected_union = vec![false; view.n_expert];
    let mut idx_all = vec![0usize; view.n_expert];
    let mut probs = vec![0.0f32; view.n_expert];
    let mut selected_weights = vec![0.0f32; selected_count];
    for token in 0..seq_len {
        let logits_start = token * view.n_expert;
        let retained = hy3_sigmoid_topk_route(
            &router_logits[logits_start..logits_start + view.n_expert],
            selection_bias,
            selected_count,
            view.expert_weights_norm,
            view.expert_weights_scale,
            policy::moe_adaptive_top_p(),
            &mut idx_all,
            &mut probs,
            &mut selected_weights,
        );
        if retained != selected_count {
            return Err(LlmError::Forward(format!(
                "GLM batched sparse adaptive route retained {retained} of {selected_count} experts"
            )));
        }
        for slot in 0..selected_count {
            let expert = idx_all[slot];
            selected_union[expert] = true;
            let gate_start = expert * gate_bytes_per_expert;
            let up_start = expert * up_bytes_per_expert;
            let down_start = expert * down_bytes_per_expert;
            gate_slots.push(&view.gate_exps_bytes[gate_start..gate_start + gate_bytes_per_expert]);
            up_slots.push(&view.up_exps_bytes[up_start..up_start + up_bytes_per_expert]);
            down_slots.push(&view.down_exps_bytes[down_start..down_start + down_bytes_per_expert]);
            route_weights.push(selected_weights[slot]);
            token_ids.push(token as u32);
        }
    }

    if let (Some(cache), Some(layer_index)) = (page_cache, view.layer_idx) {
        let selected = selected_union
            .iter()
            .enumerate()
            .filter_map(|(expert, &used)| used.then_some(expert))
            .collect::<Vec<_>>();
        cache.touch(layer_index, &selected);
    }

    let sparse = backend_runtime::glm_moe_prefill_sparse_experts_iq2xxs_iq3xxs_by_token(
        &gate_slots,
        &up_slots,
        &down_slots,
        file_regions,
        direct_file,
        &route_weights,
        &token_ids,
        seq_len,
        view.n_ff,
        hidden_dim,
        &normalized,
    )
    .map_err(LlmError::Forward)?;
    if sparse.len() != residual.len() {
        return Err(LlmError::Forward(format!(
            "GLM batched sparse output mismatch: got={} expected={}",
            sparse.len(),
            residual.len()
        )));
    }

    let shared = if let Some(shared) = compute_shared_batch(view, &normalized, seq_len)? {
        shared
    } else {
        let mut shared = vec![0.0f32; residual.len()];
        for token in 0..seq_len {
            let start = token * hidden_dim;
            let h = &normalized[start..start + hidden_dim];
            let gate_scalar = if view.shared_expert_gated {
                let gate_dot = h
                    .iter()
                    .zip(view.shared_input_scale.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                1.0 / (1.0 + (-gate_dot).exp())
            } else {
                1.0
            };
            shared[start..start + hidden_dim].copy_from_slice(&compute_shared_expert(
                view,
                h,
                gate_scalar,
                false,
                false,
                true,
            ));
        }
        shared
    };
    for ((residual, &sparse), &shared) in residual.iter_mut().zip(sparse.iter()).zip(shared.iter())
    {
        *residual += sparse + shared;
    }

    Ok(())
}
