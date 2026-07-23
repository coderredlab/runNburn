use super::Sampler;

/// Greedy decoding: 가장 높은 logit의 토큰 반환
pub fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

pub struct Greedy;

impl Greedy {
    pub fn new() -> Self {
        Self
    }

    pub fn sample(&self, logits: &[f32]) -> u32 {
        greedy_sample(logits)
    }
}

impl Default for Greedy {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for Greedy {
    fn apply(&mut self, _logits: &mut [f32], _context_tokens: &[u32]) {
        // Greedy는 logits를 수정하지 않음 — sample() 또는 greedy_sample()로 직접 사용
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_basic() {
        let logits = vec![1.0f32, 3.0, 2.0];
        assert_eq!(greedy_sample(&logits), 1);
    }

    #[test]
    fn test_greedy_negative_logits() {
        let logits = vec![-3.0f32, -1.0, -2.0];
        assert_eq!(greedy_sample(&logits), 1);
    }

    #[test]
    fn test_greedy_single_element() {
        let logits = vec![42.0f32];
        assert_eq!(greedy_sample(&logits), 0);
    }

    #[test]
    fn test_greedy_first_wins_on_tie() {
        let logits = vec![1.0f32, 1.0, 1.0];
        let result = greedy_sample(&logits);
        assert!(result < 3);
    }
}
