//! pm113: GLM-DSA Metal token-batch sparse-MoE prefill.
//!
//! 기존 per-token dispatch(토큰마다 command buffer commit/wait)를 레이어당 단일
//! command buffer 로 배치한다. 라우팅/trace 기록 순서는 기존 per-token 경로와
//! 동일(layer-major, token 오름차순)해서 route trace 가 byte-identical 하다.
//! 미지원 quant/빌드에서는 `Ok(None)` 으로 기존 경로에 fallback 한다.

use super::moe_types::{down_bytes_per_row, expert_bytes_per_row, SharedExpertMoEView};
use super::page_cache::SparseExpertPageCache;
use super::routing::hy3_sigmoid_topk_route;
use crate::engine::backend_runtime;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::norm::apply_model_norm_into;
use crate::engine::{policy, ModelArchitecture};
use crate::error::Result;
use rnb_loader::GGMLType;

/// decode 의 `glm_iq_metal_batch_eligible` 과 같은 quant 조합만 허용하고
/// dispatch pipeline select 플래그를 돌려준다.
fn quant_select(view: &SharedExpertMoEView<'_>) -> Option<(bool, bool, bool, bool)> {
    let gate_up_iq2s = match (view.gate_quant, view.up_quant) {
        (GGMLType::IQ2_XXS, GGMLType::IQ2_XXS) => false,
        (GGMLType::IQ2_S, GGMLType::IQ2_S) => true,
        _ => return None,
    };
    let down_iq4xs = match view.down_quant {
        GGMLType::IQ3_XXS => false,
        GGMLType::IQ4_XS => true,
        _ => return None,
    };
    let (shared_gate_up_q6k, shared_down_q8_0) = match (
        view.shared_gate_quant,
        view.shared_up_quant,
        view.shared_down_quant,
    ) {
        (GGMLType::Q5_K, GGMLType::Q5_K, GGMLType::Q6_K) => (false, false),
        (GGMLType::Q6_K, GGMLType::Q6_K, GGMLType::Q8_0) => (true, true),
        _ => return None,
    };
    Some((
        gate_up_iq2s,
        down_iq4xs,
        shared_gate_up_q6k,
        shared_down_q8_0,
    ))
}

/// 반환 `Ok(Some(output))` = attn_out + moe_out (residual 포함). `Ok(None)` = fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward(
    view: &SharedExpertMoEView<'_>,
    attn_out: &[f32],
    ffn_norm: &[f32],
    norm_eps: f32,
    seq_len: usize,
    hidden_dim: usize,
    page_cache: Option<&SparseExpertPageCache>,
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
) -> Result<Option<Vec<f32>>> {
    let Some((gate_up_iq2s, down_iq4xs, shared_gate_up_q6k, shared_down_q8_0)) = quant_select(view)
    else {
        return Ok(None);
    };
    let selected_count = view.n_expert_used.min(view.n_expert);
    if selected_count == 0 || selected_count > 8 || view.expert_gating_func != 2 {
        return Ok(None);
    }
    let Some(selection_bias) = view.router_selection_bias else {
        return Ok(None);
    };
    if view.shared_gate_bytes.is_empty()
        || view.shared_up_bytes.is_empty()
        || view.shared_down_bytes.is_empty()
    {
        return Ok(None);
    }

    let mut normalized = vec![0.0f32; seq_len * hidden_dim];
    for token in 0..seq_len {
        let start = token * hidden_dim;
        apply_model_norm_into(
            &attn_out[start..start + hidden_dim],
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

    let gate_bpe = view.n_ff * expert_bytes_per_row(hidden_dim, view.gate_quant, "gate_exps");
    let up_bpe = view.n_ff * expert_bytes_per_row(hidden_dim, view.up_quant, "up_exps");
    let down_bpe = hidden_dim * down_bytes_per_row(view.n_ff, view.down_quant);

    let slots = selected_count + 1;
    let mut gate_slots: Vec<&[u8]> = Vec::with_capacity(seq_len * slots);
    let mut up_slots: Vec<&[u8]> = Vec::with_capacity(seq_len * slots);
    let mut down_slots: Vec<&[u8]> = Vec::with_capacity(seq_len * slots);
    let mut route_weights_all = Vec::with_capacity(seq_len * slots);
    let mut selected_per_token = Vec::with_capacity(seq_len);
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
            // adaptive top-p 로 slot 수가 토큰마다 달라지면 batch 불가 — trace
            // 기록 전이므로 부작용 없이 fallback.
            return Ok(None);
        }
        for slot in 0..selected_count {
            let expert = idx_all[slot];
            selected_union[expert] = true;
            gate_slots.push(&view.gate_exps_bytes[expert * gate_bpe..(expert + 1) * gate_bpe]);
            up_slots.push(&view.up_exps_bytes[expert * up_bpe..(expert + 1) * up_bpe]);
            down_slots.push(&view.down_exps_bytes[expert * down_bpe..(expert + 1) * down_bpe]);
            route_weights_all.push(selected_weights[slot]);
        }
        let h = &normalized[token * hidden_dim..(token + 1) * hidden_dim];
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
        gate_slots.push(view.shared_gate_bytes);
        up_slots.push(view.shared_up_bytes);
        down_slots.push(view.shared_down_bytes);
        route_weights_all.push(gate_scalar);
        selected_per_token.push(idx_all[..selected_count].to_vec());
    }

    let mut sparse_out = vec![0.0f32; seq_len * hidden_dim];
    let used = backend_runtime::glm_moe_prefill_iq_batch_into(
        &gate_slots,
        &up_slots,
        &down_slots,
        &route_weights_all,
        seq_len,
        selected_count,
        view.n_ff,
        hidden_dim,
        &normalized,
        &mut sparse_out,
        gate_up_iq2s,
        down_iq4xs,
        shared_gate_up_q6k,
        shared_down_q8_0,
        false,
        false,
        file_regions,
    )
    .map_err(crate::error::LlmError::Forward)?;
    if !used {
        return Ok(None);
    }

    // trace / page-cache 부작용은 backend 성공 이후에만 — 기존 per-token 경로와
    // 같은 layer-major/token 오름차순 기록 순서.
    if let Some(layer) = view.layer_idx {
        for selected in &selected_per_token {
            crate::engine::moe_trace::record_selection(layer, selected);
        }
        if let Some(cache) = page_cache {
            let selected = selected_union
                .iter()
                .enumerate()
                .filter_map(|(expert, &used)| used.then_some(expert))
                .collect::<Vec<_>>();
            cache.touch(layer, &selected);
        }
    }

    let mut output = attn_out.to_vec();
    for (dst, &value) in output.iter_mut().zip(sparse_out.iter()) {
        *dst += value;
    }
    Ok(Some(output))
}
