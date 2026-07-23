pub fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6_f32;
    let coeff = 0.044_715_f32;
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geglu_gate_matches_gemma_activation_rule() {
        assert_eq!(gelu_tanh(0.0), 0.0);
        assert!(gelu_tanh(2.0) > 1.9);
        assert!(gelu_tanh(-2.0) < 0.0);
    }
}
