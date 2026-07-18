use super::{softmax_inplace, Sampler};

pub struct TopP {
    pub p: f32,
}

impl TopP {
    pub fn new(p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "top_p must be in [0, 1]");
        Self { p }
    }
}

impl Sampler for TopP {
    fn apply(&mut self, logits: &mut [f32], _context_tokens: &[u32]) {
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);

        let mut indexed: Vec<(f32, usize)> =
            probs.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut cumsum = 0.0f32;
        let mut cutoff_reached = false;
        for (prob, idx) in &indexed {
            if cutoff_reached {
                logits[*idx] = f32::NEG_INFINITY;
                continue;
            }
            cumsum += prob;
            if cumsum > self.p {
                cutoff_reached = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_p_filters_low_prob() {
        let mut logits = vec![10.0f32, 0.0, 0.0];
        let mut tp = TopP::new(0.95);
        tp.apply(&mut logits, &[]);
        assert!(logits[0] > f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_1_is_noop() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut tp = TopP::new(1.0);
        tp.apply(&mut logits, &[]);
        assert!(logits.iter().all(|&x| x > f32::NEG_INFINITY));
    }

    #[test]
    fn test_top_p_keeps_minimum_one() {
        let mut logits = vec![5.0f32, 1.0, 1.0];
        let mut tp = TopP::new(0.001);
        tp.apply(&mut logits, &[]);
        let alive = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert!(alive >= 1);
    }

    #[test]
    fn test_top_p_uniform_keeps_enough() {
        let mut logits = vec![1.0f32; 5];
        let mut tp = TopP::new(0.5);
        tp.apply(&mut logits, &[]);
        let alive = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert!(alive >= 2);
    }
}
