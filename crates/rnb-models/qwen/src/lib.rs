#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

pub fn softmax_topk_route(logits: &[f32], expert_used_count: usize) -> Route {
    let n_expert = logits.len();
    let selected_len = expert_used_count.min(n_expert);
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&logit| (logit - max_l).exp()).collect();
    let sum_all: f32 = probs.iter().sum();
    for prob in &mut probs {
        *prob /= sum_all;
    }

    let mut experts: Vec<usize> = (0..n_expert).collect();
    if selected_len < n_expert {
        experts.select_nth_unstable_by(selected_len, |&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    experts.truncate(selected_len);

    let mut weights: Vec<f32> = experts.iter().map(|&idx| probs[idx]).collect();
    let selected_sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= selected_sum;
    }

    Route { experts, weights }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_topk_route_selects_by_probability_and_renormalizes() {
        let logits = [0.0, 3.0, 2.0, 1.0];
        let route = softmax_topk_route(&logits, 2);

        assert_eq!(route.experts, vec![1, 2]);
        let sum: f32 = route.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(route.weights[0] > route.weights[1]);
    }
}
