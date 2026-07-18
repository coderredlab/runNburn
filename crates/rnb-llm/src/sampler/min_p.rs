use super::{softmax_inplace, Sampler};

pub struct MinP {
    pub min_p: f32,
}

impl MinP {
    pub fn new(min_p: f32) -> Self {
        assert!(min_p >= 0.0, "min_p must be >= 0");
        Self { min_p }
    }
}

impl Sampler for MinP {
    fn apply(&mut self, logits: &mut [f32], _context_tokens: &[u32]) {
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);

        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_prob * self.min_p;

        for (logit, prob) in logits.iter_mut().zip(probs.iter()) {
            if *prob < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_p_basic() {
        let mut logits = vec![3.0f32, 0.0, 0.0];
        let mut mp = MinP::new(0.1);
        mp.apply(&mut logits, &[]);
        assert!(logits[0] > f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_zero_is_noop() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut mp = MinP::new(0.0);
        mp.apply(&mut logits, &[]);
        assert!(logits.iter().all(|&x| x > f32::NEG_INFINITY));
    }

    #[test]
    fn test_min_p_uniform() {
        let mut logits = vec![1.0f32; 3];
        let mut mp = MinP::new(0.5);
        mp.apply(&mut logits, &[]);
        let alive = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert_eq!(alive, 3);
    }

    #[test]
    fn test_min_p_keeps_at_least_one() {
        let mut logits = vec![5.0f32, 1.0, 1.0];
        let mut mp = MinP::new(0.99);
        mp.apply(&mut logits, &[]);
        let alive = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert!(alive >= 1);
    }
}
