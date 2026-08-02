mod vision;
mod vision_encoder;
mod vision_math;
mod vision_preprocess;

pub use rnb_core::image::RgbImage;
pub use vision::{
    inspect_gemma4_vision_projector, Gemma4VisionCapability, Gemma4VisionError,
    GEMMA4_VISION_MERGE_SIZE, GEMMA4_VISION_ROPE_THETA,
};
pub use vision_encoder::{
    encode_gemma4_vision_intermediate, Gemma4VisionLayerSummary, Gemma4VisionOutput,
};
pub use vision_preprocess::{
    gemma4_smart_resize, prepare_gemma4_vision_intermediate, Gemma4TensorStats,
    Gemma4VisionIntermediate, GEMMA4_MAX_IMAGE_TOKENS, GEMMA4_MIN_IMAGE_TOKENS,
};

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
