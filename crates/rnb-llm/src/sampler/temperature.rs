use super::Sampler;

pub struct Temperature {
    pub temperature: f32,
}

impl Temperature {
    pub fn new(temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        Self { temperature }
    }
}

impl Sampler for Temperature {
    fn apply(&mut self, logits: &mut [f32], _context_tokens: &[u32]) {
        for x in logits.iter_mut() {
            *x /= self.temperature;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_temperature_scaling_half() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut t = Temperature::new(0.5);
        t.apply(&mut logits, &[]);
        assert_abs_diff_eq!(logits[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[1], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[2], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let original = logits.clone();
        let mut t = Temperature::new(1.0);
        t.apply(&mut logits, &[]);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_temperature_two() {
        let mut logits = vec![4.0f32, 2.0, 0.0];
        let mut t = Temperature::new(2.0);
        t.apply(&mut logits, &[]);
        assert_abs_diff_eq!(logits[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(logits[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_temperature_zero_panics() {
        Temperature::new(0.0);
    }
}
