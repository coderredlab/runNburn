use super::Sampler;

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
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut indexed = Vec::new();
        let mut sum = 0.0f32;
        for (index, &logit) in logits.iter().enumerate() {
            if logit == f32::NEG_INFINITY {
                continue;
            }
            let probability = (logit - max).exp();
            sum += probability;
            indexed.push((probability, index));
        }
        if sum > 0.0 {
            for (probability, _) in &mut indexed {
                *probability /= sum;
            }
        }
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
    use crate::sampler::softmax_inplace;

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

    #[test]
    fn sparse_candidates_match_dense_reference() {
        let source = vec![
            f32::NEG_INFINITY,
            4.0,
            f32::NEG_INFINITY,
            2.0,
            3.0,
            f32::NEG_INFINITY,
            1.0,
        ];
        for p in [0.1, 0.5, 0.9, 0.999] {
            let mut expected = source.clone();
            let mut probs = expected.clone();
            softmax_inplace(&mut probs);
            let mut indexed = probs
                .iter()
                .enumerate()
                .map(|(index, &probability)| (probability, index))
                .collect::<Vec<_>>();
            indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut cumsum = 0.0f32;
            let mut cutoff_reached = false;
            for (probability, index) in indexed {
                if cutoff_reached {
                    expected[index] = f32::NEG_INFINITY;
                    continue;
                }
                cumsum += probability;
                if cumsum > p {
                    cutoff_reached = true;
                }
            }

            let mut actual = source.clone();
            TopP::new(p).apply(&mut actual, &[]);
            assert_eq!(actual, expected, "p={p}");
        }
    }
}
