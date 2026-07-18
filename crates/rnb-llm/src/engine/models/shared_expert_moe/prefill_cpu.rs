//! AArch64 CPU expert-major prefill for Qwen3.5/3.6 MoE.

#[cfg(target_arch = "aarch64")]
use super::jit_request::qwen35_moe_jit_load_requested;
#[cfg(target_arch = "aarch64")]
use super::moe_types::{down_bytes_per_row, expert_bytes_per_row, SharedExpertMoEView};
#[cfg(target_arch = "aarch64")]
use super::prefill_cpu_expert_group::compute_expert_groups;
#[cfg(any(target_arch = "aarch64", test))]
use super::routing::qwen35_softmax_topk_route;
#[cfg(target_arch = "aarch64")]
use crate::engine::cpu_runtime::kernels;
#[cfg(target_arch = "aarch64")]
use crate::engine::dense_dispatch;
#[cfg(target_arch = "aarch64")]
use crate::engine::moe_profile::{
    is_enabled as moe_profile_enabled, record_moe_profile, record_moe_profile_by_layer,
};
#[cfg(target_arch = "aarch64")]
use crate::engine::norm::{apply_model_gate_mul_inplace, apply_model_norm_into};
#[cfg(target_arch = "aarch64")]
use crate::engine::policy;
#[cfg(target_arch = "aarch64")]
use crate::engine::quantized_dispatch::prefill_raw_quantized_batch;
#[cfg(target_arch = "aarch64")]
use crate::engine::quantized_dispatch::{
    prefill_raw_dual_q4k_q8k, quantize_raw_q8k, QuantizedQ8KBlock,
};
#[cfg(target_arch = "aarch64")]
use rnb_core::tensor::Tensor;
#[cfg(target_arch = "aarch64")]
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};
#[cfg(any(target_arch = "aarch64", test))]
use std::time::Duration;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(any(target_arch = "aarch64", test))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct QwenRouteSlot {
    pub(super) expert: usize,
    pub(super) token: usize,
    pub(super) rank: usize,
    pub(super) weight: f32,
}

#[cfg(any(target_arch = "aarch64", test))]
#[cfg_attr(test, allow(dead_code))]
#[derive(Debug)]
pub(super) struct QwenExpertGroupOutput {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) output: Vec<f32>,
    pub(super) gather_elapsed: Duration,
    pub(super) gate_up_elapsed: Duration,
    pub(super) activation_elapsed: Duration,
    pub(super) down_elapsed: Duration,
}

#[cfg(any(target_arch = "aarch64", test))]
const MIN_AVERAGE_ROUTES_PER_EXPERT: usize = 8;

#[cfg(any(target_arch = "aarch64", test))]
fn has_expert_batch_density(seq_len: usize, n_expert: usize, n_expert_used: usize) -> bool {
    seq_len.saturating_mul(n_expert_used) >= n_expert.saturating_mul(MIN_AVERAGE_ROUTES_PER_EXPERT)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn qwen35_cpu_expert_major_enabled(
    architecture: ModelArchitecture,
    view: &SharedExpertMoEView<'_>,
    seq_len: usize,
) -> bool {
    !cfg!(feature = "cuda")
        && architecture == ModelArchitecture::Qwen35MoE
        && seq_len > 1
        && policy::qwen35_cpu_expert_major_enabled()
        && has_expert_batch_density(seq_len, view.n_expert, view.n_expert_used)
        && !policy::moe_mixed_precision_requested()
        && !crate::engine::moe_trace::route_trace_is_active()
        && !crate::engine::moe_trace::predictor_trace_is_active()
        && !crate::engine::moe_trace::is_active()
        && !qwen35_moe_jit_load_requested()
        && matches!(view.expert_gating_func, 0 | 1)
        && view.gate_quant == GGMLType::Q4_K
        && view.up_quant == GGMLType::Q4_K
        && matches!(
            view.down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        && view.shared_gate_quant == GGMLType::Q4_K
        && view.shared_up_quant == GGMLType::Q4_K
        && matches!(
            view.shared_down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        && view.moe_section_decode.is_none()
        && view.gate_residency.is_none()
        && view.up_residency.is_none()
        && view.down_residency.is_none()
}

#[cfg(any(target_arch = "aarch64", test))]
fn qwen_expert_major_slots(
    logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<QwenRouteSlot> {
    let selected_len = n_expert_used.min(n_expert);
    let mut slots = Vec::with_capacity(seq_len * selected_len);
    let mut idx_all = vec![0usize; n_expert];
    let mut probs = vec![0.0f32; n_expert];
    let mut selected_weights = vec![0.0f32; selected_len];

    for token in 0..seq_len {
        let token_logits = &logits[token * n_expert..(token + 1) * n_expert];
        let selected = qwen35_softmax_topk_route(
            token_logits,
            n_expert_used,
            &mut idx_all,
            &mut probs,
            &mut selected_weights,
            false,
        );
        for rank in 0..selected {
            slots.push(QwenRouteSlot {
                expert: idx_all[rank],
                token,
                rank,
                weight: selected_weights[rank],
            });
        }
    }
    slots.sort_unstable_by_key(|slot| (slot.expert, slot.token));
    slots
}

#[cfg(target_arch = "aarch64")]
fn expert_group_shape(slots: &[QwenRouteSlot]) -> (usize, usize) {
    let mut groups = 0usize;
    let mut max_group = 0usize;
    let mut start = 0usize;
    while start < slots.len() {
        let expert = slots[start].expert;
        let mut end = start + 1;
        while end < slots.len() && slots[end].expert == expert {
            end += 1;
        }
        groups += 1;
        max_group = max_group.max(end - start);
        start = end;
    }
    (groups, max_group)
}

#[cfg(any(target_arch = "aarch64", test))]
fn reduce_weighted_expert_groups(
    slots: &[QwenRouteSlot],
    groups: &[QwenExpertGroupOutput],
    seq_len: usize,
    n_expert_used: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut reduced = vec![0.0f32; seq_len * hidden_dim];
    for rank in 0..n_expert_used {
        for group in groups {
            let group_slots = &slots[group.start..group.end];
            debug_assert_eq!(group.output.len(), group_slots.len() * hidden_dim);
            for (group_row, slot) in group_slots.iter().enumerate() {
                if slot.rank != rank {
                    continue;
                }
                let source = &group.output[group_row * hidden_dim..(group_row + 1) * hidden_dim];
                let destination =
                    &mut reduced[slot.token * hidden_dim..(slot.token + 1) * hidden_dim];
                for (dst, &value) in destination.iter_mut().zip(source.iter()) {
                    let weighted = value * slot.weight;
                    *dst += weighted;
                }
            }
        }
    }
    reduced
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn record_stage(layer_idx: usize, stage: &'static str, elapsed: Duration) {
    let key = match stage {
        "norm" => "qwen35moe:prefill:expert_major:norm",
        "routing" => "qwen35moe:prefill:expert_major:routing",
        "shared" => "qwen35moe:prefill:expert_major:shared",
        "sparse" => "qwen35moe:prefill:expert_major:sparse",
        "gather" => "qwen35moe:prefill:expert_major:gather",
        "gate_up" => "qwen35moe:prefill:expert_major:gate_up",
        "activation" => "qwen35moe:prefill:expert_major:activation",
        "down" => "qwen35moe:prefill:expert_major:down",
        "scatter" => "qwen35moe:prefill:expert_major:scatter",
        "reduce" => "qwen35moe:prefill:expert_major:reduce",
        "finalize" => "qwen35moe:prefill:expert_major:finalize",
        "total" => "qwen35moe:prefill:expert_major:total",
        _ => unreachable!("unknown Qwen CPU prefill stage"),
    };
    record_moe_profile(key, elapsed);
    record_moe_profile_by_layer(
        "qwen35moe:prefill:expert_major",
        Some(layer_idx),
        stage,
        elapsed,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn finish_stage(layer_idx: usize, stage: &'static str, started: Option<Instant>) {
    if let Some(started) = started {
        record_stage(layer_idx, stage, started.elapsed());
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_qwen35_cpu_expert_major(
    architecture: ModelArchitecture,
    hidden: Tensor,
    ffn_norm_data: &[f32],
    view: &SharedExpertMoEView<'_>,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<Tensor> {
    debug_assert!(qwen35_cpu_expert_major_enabled(architecture, view, seq_len));
    let profile_enabled = moe_profile_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let norm_start = profile_enabled.then(Instant::now);
    let mut hidden_data = kernels::tensor_as_f32_slice(&hidden).to_vec();
    let mut normalized = vec![0.0f32; seq_len * hidden_dim];
    for token in 0..seq_len {
        let range = token * hidden_dim..(token + 1) * hidden_dim;
        apply_model_norm_into(
            &hidden_data[range.clone()],
            ffn_norm_data,
            norm_eps,
            &mut normalized[range],
            architecture,
        );
    }
    finish_stage(layer_idx, "norm", norm_start);

    let route_start = profile_enabled.then(Instant::now);
    let mut router_logits = vec![0.0f32; seq_len * view.n_expert];
    dense_dispatch::gemv_f32(
        view.router_w,
        &normalized,
        &mut router_logits,
        view.n_expert,
        hidden_dim,
        seq_len,
    );
    let slots = qwen_expert_major_slots(&router_logits, seq_len, view.n_expert, view.n_expert_used);
    let (group_count, max_group) = expert_group_shape(&slots);
    finish_stage(layer_idx, "routing", route_start);

    let shared_start = profile_enabled.then(Instant::now);
    let shared_gate_up_bpr = expert_bytes_per_row(hidden_dim, GGMLType::Q4_K, "shared_gate_up");
    let shared_down_bpr = down_bytes_per_row(view.n_ff, view.shared_down_quant);
    let mut shared_gate = vec![0.0f32; seq_len * view.n_ff];
    let mut shared_up = vec![0.0f32; seq_len * view.n_ff];
    let normalized_q8k = quantize_raw_q8k(&normalized);
    prefill_raw_dual_q4k_q8k(
        view.shared_gate_bytes,
        view.shared_up_bytes,
        &normalized_q8k,
        &mut shared_gate,
        &mut shared_up,
        view.n_ff,
        hidden_dim,
        seq_len,
        shared_gate_up_bpr,
    );
    apply_model_gate_mul_inplace(&mut shared_gate, &shared_up, architecture);
    let mut shared_output = vec![0.0f32; seq_len * hidden_dim];
    prefill_raw_quantized_batch(
        view.shared_down_bytes,
        &shared_gate,
        &mut shared_output,
        hidden_dim,
        view.n_ff,
        seq_len,
        shared_down_bpr,
        view.shared_down_quant,
    );
    if view.shared_expert_gated {
        for token in 0..seq_len {
            let h = &normalized[token * hidden_dim..(token + 1) * hidden_dim];
            let gate_dot: f32 = h
                .iter()
                .zip(view.shared_input_scale.iter())
                .map(|(a, b)| a * b)
                .sum();
            let gate = 1.0 / (1.0 + (-gate_dot).exp());
            for value in &mut shared_output[token * hidden_dim..(token + 1) * hidden_dim] {
                *value *= gate;
            }
        }
    }
    finish_stage(layer_idx, "shared", shared_start);

    let sparse_start = profile_enabled.then(Instant::now);
    let gate_up_bpr = expert_bytes_per_row(hidden_dim, GGMLType::Q4_K, "gate_up_exps");
    let down_bpr = down_bytes_per_row(view.n_ff, view.down_quant);
    let per_gate_up = view.n_ff * gate_up_bpr;
    let per_down = hidden_dim * down_bpr;
    let expert_groups = compute_expert_groups(
        &slots,
        &normalized_q8k,
        view.gate_exps_bytes,
        view.up_exps_bytes,
        view.down_exps_bytes,
        per_gate_up,
        gate_up_bpr,
        per_down,
        down_bpr,
        view.n_ff,
        hidden_dim,
        view.down_quant,
        architecture,
        profile_enabled,
    );

    let reduce_start = profile_enabled.then(Instant::now);
    let sparse_output = reduce_weighted_expert_groups(
        &slots,
        &expert_groups,
        seq_len,
        view.n_expert_used,
        hidden_dim,
    );
    if profile_enabled {
        let sum_elapsed = |select: fn(&QwenExpertGroupOutput) -> Duration| {
            expert_groups
                .iter()
                .fold(Duration::ZERO, |total, group| total + select(group))
        };
        record_stage(
            layer_idx,
            "gather",
            sum_elapsed(|group| group.gather_elapsed),
        );
        record_stage(
            layer_idx,
            "gate_up",
            sum_elapsed(|group| group.gate_up_elapsed),
        );
        record_stage(
            layer_idx,
            "activation",
            sum_elapsed(|group| group.activation_elapsed),
        );
        record_stage(layer_idx, "down", sum_elapsed(|group| group.down_elapsed));
        record_stage(layer_idx, "scatter", Duration::ZERO);
        record_stage(
            layer_idx,
            "reduce",
            reduce_start
                .expect("profile timer exists when profiling is enabled")
                .elapsed(),
        );
    }
    finish_stage(layer_idx, "sparse", sparse_start);

    let finalize_start = profile_enabled.then(Instant::now);
    for ((hidden, sparse), shared) in hidden_data
        .iter_mut()
        .zip(sparse_output.iter())
        .zip(shared_output.iter())
    {
        *hidden += sparse + shared;
    }
    if policy::profiling_enabled() {
        let retained_f32_elements = hidden_data.len()
            + normalized.len()
            + router_logits.len()
            + shared_gate.len()
            + shared_up.len()
            + shared_output.len()
            + sparse_output.len()
            + expert_groups
                .iter()
                .map(|group| group.output.len())
                .sum::<usize>();
        let retained_f32_bytes = retained_f32_elements * std::mem::size_of::<f32>();
        let retained_input_bytes = normalized_q8k.len() * std::mem::size_of::<QuantizedQ8KBlock>();
        let concurrent_groups = rayon::current_num_threads().max(1).min(group_count);
        let per_worker_bytes = max_group
            * ((hidden_dim / 256) * std::mem::size_of::<QuantizedQ8KBlock>()
                + (2 * view.n_ff + hidden_dim) * std::mem::size_of::<f32>());
        let retained_bytes = retained_f32_bytes + retained_input_bytes;
        let scratch_upper_bytes = retained_bytes + concurrent_groups * per_worker_bytes;
        eprintln!(
            "  [QWEN L{layer_idx}] moe_cpu_expert_major routes={} groups={} max_group={} retained_mib={:.1} scratch_upper_mib={:.1}",
            slots.len(),
            group_count,
            max_group,
            retained_bytes as f64 / (1024.0 * 1024.0),
            scratch_upper_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    finish_stage(layer_idx, "finalize", finalize_start);
    finish_stage(layer_idx, "total", total_start);
    Ok(Tensor::from_vec(hidden_data, &[seq_len, hidden_dim]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_major_slots_keep_original_token_rank() {
        let logits = [4.0f32, 1.0, 3.0, 2.0, 0.0, 5.0, 2.0, 1.0];
        let slots = qwen_expert_major_slots(&logits, 2, 4, 2);
        assert_eq!(slots.len(), 4);
        assert!(slots
            .windows(2)
            .all(|pair| { (pair[0].expert, pair[0].token) <= (pair[1].expert, pair[1].token) }));
        let token0 = slots
            .iter()
            .filter(|slot| slot.token == 0)
            .collect::<Vec<_>>();
        assert_eq!(token0.len(), 2);
        assert!(token0.iter().any(|slot| slot.expert == 0 && slot.rank == 0));
        assert!(token0.iter().any(|slot| slot.expert == 2 && slot.rank == 1));
        let token1 = slots
            .iter()
            .filter(|slot| slot.token == 1)
            .collect::<Vec<_>>();
        assert!(token1.iter().any(|slot| slot.expert == 1 && slot.rank == 0));
        assert!(token1.iter().any(|slot| slot.expert == 2 && slot.rank == 1));
    }

    fn cpu_prefill_route_arrays(
        router_w: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        normalized: &[f32],
        seq_len: usize,
        n_expert_used: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
        let mut logits = vec![0.0f32; seq_len * n_expert];
        crate::engine::dense_dispatch::gemv_f32(
            router_w,
            normalized,
            &mut logits,
            n_expert,
            hidden_dim,
            seq_len,
        );
        let selected_len = n_expert_used.min(n_expert);
        let route_capacity = seq_len * selected_len;
        let mut expert_ids = Vec::with_capacity(route_capacity);
        let mut route_weights = Vec::with_capacity(route_capacity);
        let mut token_ids = Vec::with_capacity(route_capacity);
        let mut idx_all = vec![0usize; n_expert];
        let mut probs = vec![0.0f32; n_expert];
        let mut selected_weights = vec![0.0f32; selected_len];
        for token in 0..seq_len {
            let token_logits = &logits[token * n_expert..(token + 1) * n_expert];
            let selected = qwen35_softmax_topk_route(
                token_logits,
                n_expert_used,
                &mut idx_all,
                &mut probs,
                &mut selected_weights,
                false,
            );
            for rank in 0..selected {
                expert_ids.push(idx_all[rank] as u32);
                route_weights.push(selected_weights[rank]);
                token_ids.push(token as u32);
            }
        }
        (expert_ids, route_weights, token_ids)
    }

    #[test]
    fn cpu_prefill_route_arrays_match_expert_major_slots_bitwise() {
        const SEQ_LEN: usize = 3;
        const HIDDEN_DIM: usize = 4;
        const N_EXPERT: usize = 5;
        const N_EXPERT_USED: usize = 2;
        let router_w = [
            0.5f32, -0.25, 0.75, 0.1, -0.3, 0.8, 0.2, -0.6, 1.1, 0.4, -0.2, 0.3, -0.7, 0.6, 0.9,
            -0.1, 0.2, -1.0, 0.35, 0.55,
        ];
        let normalized = [
            0.4f32, -0.8, 1.2, 0.3, -0.5, 0.9, 0.7, -1.1, 1.3, 0.2, -0.4, 0.8,
        ];

        let (expert_ids, route_weights, token_ids) = cpu_prefill_route_arrays(
            &router_w,
            N_EXPERT,
            HIDDEN_DIM,
            &normalized,
            SEQ_LEN,
            N_EXPERT_USED,
        );
        let mut logits = vec![0.0f32; SEQ_LEN * N_EXPERT];
        crate::engine::dense_dispatch::gemv_f32(
            &router_w,
            &normalized,
            &mut logits,
            N_EXPERT,
            HIDDEN_DIM,
            SEQ_LEN,
        );
        let mut expert_major = qwen_expert_major_slots(&logits, SEQ_LEN, N_EXPERT, N_EXPERT_USED);
        expert_major.sort_unstable_by_key(|slot| (slot.token, slot.rank));

        assert_eq!(expert_ids.len(), SEQ_LEN * N_EXPERT_USED);
        assert_eq!(route_weights.len(), expert_ids.len());
        assert_eq!(token_ids.len(), expert_ids.len());
        for (array_rank, slot) in expert_major.iter().enumerate() {
            assert_eq!(expert_ids[array_rank] as usize, slot.expert);
            assert_eq!(token_ids[array_rank] as usize, slot.token);
            assert_eq!(array_rank % N_EXPERT_USED, slot.rank);
            assert_eq!(route_weights[array_rank].to_bits(), slot.weight.to_bits());
        }
    }

    #[test]
    fn expert_major_reduction_preserves_route_rank_order() {
        let slots = [
            QwenRouteSlot {
                expert: 0,
                token: 0,
                rank: 1,
                weight: 0.25,
            },
            QwenRouteSlot {
                expert: 1,
                token: 0,
                rank: 0,
                weight: 0.75,
            },
        ];
        let groups = [
            QwenExpertGroupOutput {
                start: 0,
                end: 1,
                output: vec![4.0, 8.0],
                gather_elapsed: Duration::ZERO,
                gate_up_elapsed: Duration::ZERO,
                activation_elapsed: Duration::ZERO,
                down_elapsed: Duration::ZERO,
            },
            QwenExpertGroupOutput {
                start: 1,
                end: 2,
                output: vec![8.0, 4.0],
                gather_elapsed: Duration::ZERO,
                gate_up_elapsed: Duration::ZERO,
                activation_elapsed: Duration::ZERO,
                down_elapsed: Duration::ZERO,
            },
        ];
        let reduced = reduce_weighted_expert_groups(&slots, &groups, 1, 2, 2);
        assert_eq!(reduced, vec![7.0, 5.0]);
    }

    #[test]
    fn expert_major_density_requires_eight_average_routes() {
        assert!(!has_expert_batch_density(255, 256, 8));
        assert!(has_expert_batch_density(256, 256, 8));
    }
}
