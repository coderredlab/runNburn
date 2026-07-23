fn qwen35_moe_route_sort_enabled_from_env(value: Option<&str>) -> bool {
    value
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

fn qwen35_moe_route_sort_enabled() -> bool {
    qwen35_moe_route_sort_enabled_from_env(
        crate::engine::policy::env_string("RNB_QWEN35_MOE_ROUTE_SORT").as_deref(),
    )
}

fn qwen35_select_topk_logits_bounded(
    logits: &[f32],
    n_expert_used: usize,
    idx_out: &mut [usize],
    selected_weights: &mut [f32],
) -> usize {
    let selected_len = n_expert_used.min(logits.len());
    assert!(selected_len <= 32);
    assert!(idx_out.len() >= selected_len);
    assert!(selected_weights.len() >= selected_len);
    if selected_len == 0 {
        return 0;
    }

    let mut best_vals = [f32::NEG_INFINITY; 32];
    let mut best_ids = [usize::MAX; 32];
    for (expert, &value) in logits.iter().enumerate() {
        for rank in 0..selected_len {
            if value > best_vals[rank] || (value == best_vals[rank] && expert < best_ids[rank]) {
                for shift in (rank + 1..selected_len).rev() {
                    best_vals[shift] = best_vals[shift - 1];
                    best_ids[shift] = best_ids[shift - 1];
                }
                best_vals[rank] = value;
                best_ids[rank] = expert;
                break;
            }
        }
    }

    let selected_max = best_vals[0];
    let mut selected_sum = 0.0f32;
    for rank in 0..selected_len {
        let weight = (best_vals[rank] - selected_max).exp();
        idx_out[rank] = best_ids[rank];
        selected_weights[rank] = weight;
        selected_sum += weight;
    }
    if selected_sum != 0.0 {
        for weight in &mut selected_weights[..selected_len] {
            *weight /= selected_sum;
        }
    }
    selected_len
}

pub(in crate::engine) fn qwen35_softmax_topk_route(
    logits: &[f32],
    n_expert_used: usize,
    idx_all: &mut [usize],
    probs: &mut [f32],
    selected_weights: &mut [f32],
    fill_probs: bool,
) -> usize {
    let n_expert = logits.len();
    assert!(idx_all.len() >= n_expert);
    assert!(probs.len() >= n_expert);
    let selected_len = n_expert_used.min(n_expert);
    assert!(selected_weights.len() >= selected_len);

    let idx_all = &mut idx_all[..n_expert];
    if !fill_probs && selected_len <= 32 && qwen35_moe_route_sort_enabled() {
        return qwen35_select_topk_logits_bounded(logits, selected_len, idx_all, selected_weights);
    }
    for (i, dst) in idx_all.iter_mut().enumerate() {
        *dst = i;
    }
    if fill_probs {
        let probs = &mut probs[..n_expert];
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for (p, &logit) in probs.iter_mut().zip(logits.iter()) {
            *p = (logit - max_l).exp();
        }
        let sum_all: f32 = probs.iter().sum();
        if sum_all != 0.0 {
            for p in probs.iter_mut() {
                *p /= sum_all;
            }
        }
        if selected_len < n_expert {
            idx_all.select_nth_unstable_by(selected_len, |&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let selected = &idx_all[..selected_len];
        let selected_weights = &mut selected_weights[..selected_len];
        for (dst, &expert) in selected_weights.iter_mut().zip(selected.iter()) {
            *dst = probs[expert];
        }
        let selected_sum: f32 = selected_weights.iter().sum();
        if selected_sum != 0.0 {
            for weight in selected_weights.iter_mut() {
                *weight /= selected_sum;
            }
        }
        return selected_len;
    }
    if selected_len < n_expert {
        idx_all.select_nth_unstable_by(selected_len, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let selected = &idx_all[..selected_len];
    let selected_weights = &mut selected_weights[..selected_len];
    let selected_max = selected
        .iter()
        .map(|&expert| logits[expert])
        .fold(f32::NEG_INFINITY, f32::max);
    for (dst, &expert) in selected_weights.iter_mut().zip(selected.iter()) {
        *dst = (logits[expert] - selected_max).exp();
    }
    let selected_sum: f32 = selected_weights.iter().sum();
    if selected_sum != 0.0 {
        for weight in selected_weights.iter_mut() {
            *weight /= selected_sum;
        }
    }
    if qwen35_moe_route_sort_enabled() {
        let mut pairs_stack = [(0usize, 0.0f32, 0.0f32); 32];
        let mut pairs_heap;
        let pairs: &mut [(usize, f32, f32)] = if selected_len <= pairs_stack.len() {
            for (slot, (&expert, &weight)) in pairs_stack
                .iter_mut()
                .zip(selected.iter().zip(selected_weights.iter()))
                .take(selected_len)
            {
                *slot = (expert, weight, logits[expert]);
            }
            &mut pairs_stack[..selected_len]
        } else {
            pairs_heap = selected
                .iter()
                .copied()
                .zip(selected_weights.iter().copied())
                .map(|(expert, weight)| (expert, weight, logits[expert]))
                .collect::<Vec<_>>();
            pairs_heap.as_mut_slice()
        };
        pairs.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        for (i, &(expert, weight, _)) in pairs.iter().enumerate() {
            idx_all[i] = expert;
            selected_weights[i] = weight;
        }
    }
    let _ = probs;
    selected_len
}

pub(in crate::engine) fn hy3_sigmoid_topk_route(
    logits: &[f32],
    selection_bias: &[f32],
    n_expert_used: usize,
    normalize_selected: bool,
    scale: f32,
    adaptive_top_p: Option<f32>,
    idx_all: &mut [usize],
    probs: &mut [f32],
    selected_weights: &mut [f32],
) -> usize {
    let n_expert = logits.len();
    assert_eq!(selection_bias.len(), n_expert);
    assert!(idx_all.len() >= n_expert);
    assert!(probs.len() >= n_expert);
    let selected_len = n_expert_used.min(n_expert);
    assert!(selected_weights.len() >= selected_len);

    let probs = &mut probs[..n_expert];
    let idx_all = &mut idx_all[..n_expert];
    for (i, ((p, &logit), idx)) in probs
        .iter_mut()
        .zip(logits.iter())
        .zip(idx_all.iter_mut())
        .enumerate()
    {
        *p = 1.0 / (1.0 + (-logit).exp());
        *idx = i;
    }
    idx_all.sort_unstable_by(|&a, &b| {
        let a_score = probs[a] + selection_bias[a];
        let b_score = probs[b] + selection_bias[b];
        b_score.total_cmp(&a_score).then_with(|| a.cmp(&b))
    });
    for (dst, &expert) in selected_weights[..selected_len]
        .iter_mut()
        .zip(idx_all[..selected_len].iter())
    {
        *dst = probs[expert];
    }
    let mut retained_len = selected_len;
    if normalize_selected {
        let sum: f32 = selected_weights[..selected_len].iter().sum();
        if sum > 0.0 {
            for weight in &mut selected_weights[..selected_len] {
                *weight /= sum;
            }
        }
        if let Some(top_p) = adaptive_top_p {
            let mut retained_mass = 0.0f32;
            for (rank, &weight) in selected_weights[..selected_len].iter().enumerate() {
                retained_mass += weight;
                if retained_mass >= top_p {
                    retained_len = rank + 1;
                    break;
                }
            }
            if retained_mass > 0.0 {
                for weight in &mut selected_weights[..retained_len] {
                    *weight /= retained_mass;
                }
            }
        }
    }
    for weight in &mut selected_weights[..retained_len] {
        *weight *= scale;
    }
    retained_len
}

#[cfg(test)]
mod tests {
    use super::{
        hy3_sigmoid_topk_route, qwen35_moe_route_sort_enabled_from_env, qwen35_softmax_topk_route,
    };

    #[test]
    fn qwen35_softmax_topk_route_selects_by_global_prob_and_renormalizes() {
        let logits = [0.0, 4.0, 3.0, 2.0];
        let mut idx = [0usize; 4];
        let mut probs = [0.0f32; 4];
        let mut weights = [0.0f32; 2];

        let selected_len =
            qwen35_softmax_topk_route(&logits, 2, &mut idx, &mut probs, &mut weights, false);

        assert_eq!(selected_len, 2);
        assert_eq!(&idx[..selected_len], &[1, 2]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn qwen35_softmax_topk_route_can_fill_full_probs_for_trace_paths() {
        let logits = [0.0, 4.0, 3.0, 2.0];
        let mut idx = [0usize; 4];
        let mut probs = [0.0f32; 4];
        let mut weights = [0.0f32; 2];

        let selected_len =
            qwen35_softmax_topk_route(&logits, 2, &mut idx, &mut probs, &mut weights, true);

        assert_eq!(selected_len, 2);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs[1] > probs[2]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn qwen35_bounded_topk_selects_sorted_experts_without_full_index_partition() {
        let logits = [1.0, 4.0, 4.0, -2.0, 3.0];
        let mut idx = [usize::MAX; 3];
        let mut weights = [0.0f32; 3];

        let selected = super::qwen35_select_topk_logits_bounded(&logits, 3, &mut idx, &mut weights);

        assert_eq!(selected, 3);
        assert_eq!(&idx, &[1, 2, 4]);
        let e4 = 1.0f32;
        let e3 = (-1.0f32).exp();
        let sum = e4 + e4 + e3;
        assert!((weights[0] - e4 / sum).abs() < 1e-6);
        assert!((weights[1] - e4 / sum).abs() < 1e-6);
        assert!((weights[2] - e3 / sum).abs() < 1e-6);
    }

    #[test]
    fn qwen35_moe_route_sort_defaults_on_with_falsey_optout() {
        assert!(qwen35_moe_route_sort_enabled_from_env(None));
        assert!(qwen35_moe_route_sort_enabled_from_env(Some("1")));
        assert!(qwen35_moe_route_sort_enabled_from_env(Some("true")));
        assert!(!qwen35_moe_route_sort_enabled_from_env(Some("0")));
        assert!(!qwen35_moe_route_sort_enabled_from_env(Some("false")));
        assert!(!qwen35_moe_route_sort_enabled_from_env(Some("off")));
        assert!(!qwen35_moe_route_sort_enabled_from_env(Some("no")));
    }

    #[test]
    fn hy3_sigmoid_topk_route_uses_bias_only_for_selection_then_normalizes_and_scales() {
        let logits = [0.0, 2.0, -2.0, 1.0];
        let bias = [1.0, 0.0, 2.0, 0.0];
        let mut idx = [0usize; 4];
        let mut probs = [0.0f32; 4];
        let mut weights = [0.0f32; 2];

        let selected_len = hy3_sigmoid_topk_route(
            &logits,
            &bias,
            2,
            true,
            2.826,
            None,
            &mut idx,
            &mut probs,
            &mut weights,
        );

        assert_eq!(selected_len, 2);
        assert_eq!(&idx[..selected_len], &[2, 0]);
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert!((probs[2] - 0.119_202_92).abs() < 1e-6);
        assert!((weights.iter().sum::<f32>() - 2.826).abs() < 1e-6);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn hy3_adaptive_top_p_keeps_minimal_prefix_and_renormalizes() {
        let logits = [3.0, 2.0, 1.0, 0.0];
        let bias = [0.0; 4];
        let mut idx = [0usize; 4];
        let mut probs = [0.0f32; 4];
        let mut weights = [0.0f32; 4];

        let selected_len = hy3_sigmoid_topk_route(
            &logits,
            &bias,
            4,
            true,
            2.0,
            Some(0.8),
            &mut idx,
            &mut probs,
            &mut weights,
        );

        assert_eq!(selected_len, 3);
        assert_eq!(&idx[..selected_len], &[0, 1, 2]);
        assert!((weights[..selected_len].iter().sum::<f32>() - 2.0).abs() < 1e-6);
    }
}
