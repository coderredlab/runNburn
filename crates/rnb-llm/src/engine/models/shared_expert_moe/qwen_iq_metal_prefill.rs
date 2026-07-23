//! pm123: Qwen3.6 MoE prefill 의 IQ expert(IQ3_XXS gate/up + IQ4_XS down) 를 Metal
//! token-batch 로 라우팅한다. GLM IQ batch 커널(`glm_moe_prefill_iq_batch`)을 재사용하되
//! routing 만 Qwen softmax top-k(`qwen35_softmax_topk_route`)로 바꾼다. unsloth
//! UD-Q3_K_M/Q2_K_XL 처럼 experts 가 IQ 인 aggressive-quant 모델의 prefill CPU fallback 제거.
//! 미지원 quant/조건에서는 `Ok(None)` 으로 기존 경로에 fallback.

use super::moe_types::{down_bytes_per_row, expert_bytes_per_row, SharedExpertMoEView};
use super::routing::qwen35_softmax_topk_route;
use crate::engine::backend_runtime;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::norm::apply_model_norm_into;
use crate::engine::ModelArchitecture;
use crate::error::Result;
use rnb_loader::GGMLType;

/// sparse gate/up + down + shared quant 이 GLM IQ batch 커널이 지원하는 조합인지 확인하고
/// dispatch 플래그(gate_up_iq3xxs, down_iq4xs, shared_gate_up_q8_0, shared_down_q8_0)를 돌려준다.
fn quant_select(view: &SharedExpertMoEView<'_>) -> Option<(bool, bool)> {
    // sparse gate/up = IQ3_XXS, down = IQ4_XS (unsloth UD-Q3_K_M). 그 외는 fallback.
    if view.gate_quant != GGMLType::IQ3_XXS || view.up_quant != GGMLType::IQ3_XXS {
        return None;
    }
    let down_iq4xs = match view.down_quant {
        GGMLType::IQ4_XS => true,
        GGMLType::IQ3_XXS => false,
        _ => return None,
    };
    // shared expert = 전부 Q8_0 (unsloth Qwen). 그 외는 fallback.
    if view.shared_gate_quant != GGMLType::Q8_0
        || view.shared_up_quant != GGMLType::Q8_0
        || view.shared_down_quant != GGMLType::Q8_0
    {
        return None;
    }
    Some((
        down_iq4xs, /*shared_gate_up_q8_0 & shared_down_q8_0 = true*/ true,
    ))
}

/// `Ok(Some(output))` = attn_out + moe_out (residual 포함). `Ok(None)` = fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward(
    view: &SharedExpertMoEView<'_>,
    attn_out: &[f32],
    ffn_norm: &[f32],
    norm_eps: f32,
    seq_len: usize,
    hidden_dim: usize,
    architecture: ModelArchitecture,
) -> Result<Option<Vec<f32>>> {
    let Some((down_iq4xs, shared_q8_0)) = quant_select(view) else {
        return Ok(None);
    };
    let selected_count = view.n_expert_used.min(view.n_expert);
    if selected_count == 0 || selected_count > 8 {
        return Ok(None);
    }
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
            architecture,
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

    let mut idx_all = vec![0usize; view.n_expert];
    let mut probs = vec![0.0f32; view.n_expert];
    let mut selected_weights = vec![0.0f32; selected_count];
    for token in 0..seq_len {
        let logits_start = token * view.n_expert;
        let retained = qwen35_softmax_topk_route(
            &router_logits[logits_start..logits_start + view.n_expert],
            selected_count,
            &mut idx_all,
            &mut probs,
            &mut selected_weights,
            false,
        );
        if retained != selected_count {
            return Ok(None);
        }
        for slot in 0..selected_count {
            let expert = idx_all[slot];
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
        false,       // gate_up_iq2s
        down_iq4xs,  // down = IQ4_XS
        false,       // shared_gate_up_q6k
        shared_q8_0, // shared_down_q8_0
        true,        // gate_up_iq3xxs
        shared_q8_0, // shared_gate_up_q8_0
        None,        // file_regions (mmap direct 미사용)
    )
    .map_err(crate::error::LlmError::Forward)?;
    if !used {
        return Ok(None);
    }

    if let Some(layer) = view.layer_idx {
        for selected in &selected_per_token {
            crate::engine::moe_trace::record_selection(layer, selected);
        }
    }

    let mut output = attn_out.to_vec();
    for (dst, &value) in output.iter_mut().zip(sparse_out.iter()) {
        *dst += value;
    }
    Ok(Some(output))
}
