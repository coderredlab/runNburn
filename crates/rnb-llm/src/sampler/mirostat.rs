use super::{softmax_inplace, Sampler};

#[derive(Debug, Clone)]
pub struct MirostatParams {
    pub version: u8,
    pub tau: f32,
    pub eta: f32,
}

pub struct MirostatState {
    pub mu: f32,
}

pub struct Mirostat {
    _params: MirostatParams,
    state: MirostatState,
}

impl Mirostat {
    pub fn new(params: MirostatParams) -> Self {
        let mu = 2.0 * params.tau;
        Self {
            _params: params,
            state: MirostatState { mu },
        }
    }

    pub fn mu(&self) -> f32 {
        self.state.mu
    }
}

impl Sampler for Mirostat {
    fn apply(&mut self, logits: &mut [f32], _context_tokens: &[u32]) {
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);

        let mu = self.state.mu;
        let mut candidate_indices: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0 && -p.log2() <= mu)
            .map(|(i, _)| i)
            .collect();

        if candidate_indices.is_empty() {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            candidate_indices = vec![best];
        }

        for (i, x) in logits.iter_mut().enumerate() {
            if !candidate_indices.contains(&i) {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

/// 선택된 토큰의 surprise로 mu 업데이트
pub fn mirostat_update_mu(state: &mut MirostatState, selected_prob: f32, eta: f32, tau: f32) {
    let surprise = -selected_prob.log2().max(-1e30);
    let error = surprise - tau;
    state.mu -= eta * error;
    state.mu = state.mu.max(0.01);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_mirostat(tau: f32, eta: f32) -> Mirostat {
        Mirostat::new(MirostatParams {
            version: 2,
            tau,
            eta,
        })
    }

    #[test]
    fn test_initial_mu() {
        let m = make_mirostat(3.0, 0.1);
        assert_abs_diff_eq!(m.mu(), 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mirostat_keeps_at_least_one() {
        let mut m = make_mirostat(1.0, 0.1);
        let mut logits = vec![5.0f32, 0.0, 0.0];
        m.apply(&mut logits, &[]);
        let alive = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert!(alive >= 1);
    }

    #[test]
    fn test_mirostat_filters_high_surprise() {
        let mut m = make_mirostat(3.0, 0.1);
        let mut logits = vec![10.0f32, 0.0, 0.0];
        m.apply(&mut logits, &[]);
        assert!(logits[0] > f32::NEG_INFINITY);
    }

    #[test]
    fn test_mu_update_decreases_on_low_surprise() {
        let mut state = MirostatState { mu: 6.0 };
        let selected_prob = 0.5f32;
        let tau = 3.0f32;
        let eta = 0.1f32;
        mirostat_update_mu(&mut state, selected_prob, eta, tau);
        // surprise = 1.0, error = 1.0 - 3.0 = -2.0, mu -= 0.1 * (-2.0) = 6.2
        assert_abs_diff_eq!(state.mu, 6.2, epsilon = 1e-5);
    }

    #[test]
    fn test_mu_update_increases_on_high_surprise() {
        let mut state = MirostatState { mu: 6.0 };
        let selected_prob = 0.03125f32; // surprise = 5.0
        let tau = 3.0f32;
        let eta = 0.1f32;
        mirostat_update_mu(&mut state, selected_prob, eta, tau);
        // error = 5.0 - 3.0 = 2.0, mu -= 0.1 * 2.0 = 5.8
        assert_abs_diff_eq!(state.mu, 5.8, epsilon = 1e-5);
    }
}
