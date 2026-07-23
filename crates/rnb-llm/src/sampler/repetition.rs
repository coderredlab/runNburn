use super::Sampler;
use std::collections::HashMap;

/// Repetition Penalty
pub struct RepetitionPenalty {
    pub penalty: f32,
}

impl RepetitionPenalty {
    pub fn new(penalty: f32) -> Self {
        assert!(penalty > 0.0, "repetition_penalty must be > 0");
        Self { penalty }
    }
}

impl Sampler for RepetitionPenalty {
    fn apply(&mut self, logits: &mut [f32], context_tokens: &[u32]) {
        for &token in context_tokens {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }
}

/// Presence Penalty
pub struct PresencePenalty {
    pub penalty: f32,
}

impl PresencePenalty {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl Sampler for PresencePenalty {
    fn apply(&mut self, logits: &mut [f32], context_tokens: &[u32]) {
        let appeared: std::collections::HashSet<u32> = context_tokens.iter().cloned().collect();
        for &token in &appeared {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= self.penalty;
            }
        }
    }
}

/// Frequency Penalty
pub struct FrequencyPenalty {
    pub penalty: f32,
}

impl FrequencyPenalty {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl Sampler for FrequencyPenalty {
    fn apply(&mut self, logits: &mut [f32], context_tokens: &[u32]) {
        let mut freq: HashMap<u32, usize> = HashMap::new();
        for &token in context_tokens {
            *freq.entry(token).or_insert(0) += 1;
        }
        for (token, count) in freq {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= self.penalty * count as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_repetition_penalty_positive_logit() {
        let mut logits = vec![0.0f32, 2.0, 0.0];
        let mut rp = RepetitionPenalty::new(2.0);
        rp.apply(&mut logits, &[1]);
        assert_abs_diff_eq!(logits[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![0.0f32, -2.0, 0.0];
        let mut rp = RepetitionPenalty::new(2.0);
        rp.apply(&mut logits, &[1]);
        assert_abs_diff_eq!(logits[1], -4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_repetition_penalty_not_in_context() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut rp = RepetitionPenalty::new(2.0);
        rp.apply(&mut logits, &[0]);
        assert_abs_diff_eq!(logits[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_presence_penalty() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut pp = PresencePenalty::new(1.0);
        pp.apply(&mut logits, &[1, 1, 1]);
        assert_abs_diff_eq!(logits[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_frequency_penalty() {
        let mut logits = vec![5.0f32, 3.0, 2.0];
        let mut fp = FrequencyPenalty::new(0.5);
        fp.apply(&mut logits, &[0, 0, 0, 1]);
        assert_abs_diff_eq!(logits[0], 3.5, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[1], 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[2], 2.0, epsilon = 1e-6);
    }
}
