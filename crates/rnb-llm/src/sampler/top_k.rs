use super::Sampler;

pub struct TopK {
    pub k: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "top_k must be > 0");
        Self { k }
    }
}

impl Sampler for TopK {
    fn apply(&mut self, logits: &mut [f32], _context_tokens: &[u32]) {
        if self.k >= logits.len() {
            return;
        }

        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[self.k - 1];

        let mut kept = 0;
        for x in logits.iter_mut() {
            if *x >= threshold && kept < self.k {
                kept += 1;
            } else {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_basic() {
        let mut logits = vec![1.0f32, 3.0, 2.0];
        let mut tk = TopK::new(2);
        tk.apply(&mut logits, &[]);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 3.0);
        assert_eq!(logits[2], 2.0);
    }

    #[test]
    fn test_top_k_1() {
        let mut logits = vec![0.5f32, -1.0, 2.0, 1.5];
        let mut tk = TopK::new(1);
        tk.apply(&mut logits, &[]);
        assert_eq!(logits[2], 2.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_larger_than_vocab_is_noop() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let original = logits.clone();
        let mut tk = TopK::new(10);
        tk.apply(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_all_equal() {
        let mut logits = vec![1.0f32; 5];
        let mut tk = TopK::new(3);
        tk.apply(&mut logits, &[]);
        let kept = logits.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        assert_eq!(kept, 3);
    }
}
