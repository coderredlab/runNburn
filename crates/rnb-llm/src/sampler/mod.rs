pub mod greedy;
pub mod min_p;
pub mod mirostat;
pub mod repetition;
pub mod temperature;
pub mod top_k;
pub mod top_p;

pub use greedy::Greedy;
pub use min_p::MinP;
pub use mirostat::{Mirostat, MirostatState};
pub use repetition::{FrequencyPenalty, PresencePenalty, RepetitionPenalty};
pub use temperature::Temperature;
pub use top_k::TopK;
pub use top_p::TopP;

use rand::Rng;

/// logits를 in-place로 수정 (필터링/스케일링)
pub trait Sampler: Send {
    fn apply(&mut self, logits: &mut [f32], context_tokens: &[u32]);
}

/// softmax 유틸리티 (in-place)
pub fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in logits.iter_mut() {
            *x /= sum;
        }
    }
}

/// 확률 분포에서 카테고리 샘플링
pub fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.gen::<f32>();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
    greedy: bool,
}

impl SamplerChain {
    pub fn from_params(params: &crate::generate::GenerateParams) -> Self {
        let mut chain: Vec<Box<dyn Sampler>> = Vec::new();

        if params.repetition_penalty != 1.0 {
            chain.push(Box::new(RepetitionPenalty::new(params.repetition_penalty)));
        }
        if params.presence_penalty != 0.0 {
            chain.push(Box::new(PresencePenalty::new(params.presence_penalty)));
        }
        if params.frequency_penalty != 0.0 {
            chain.push(Box::new(FrequencyPenalty::new(params.frequency_penalty)));
        }

        let is_greedy = params.temperature == 0.0;

        if !is_greedy {
            if let Some(mirostat_params) = &params.mirostat {
                chain.push(Box::new(Mirostat::new(mirostat_params.clone())));
            } else {
                if params.temperature > 0.0 {
                    chain.push(Box::new(Temperature::new(params.temperature)));
                }
                if params.top_k > 0 {
                    chain.push(Box::new(TopK::new(params.top_k)));
                }
                if params.top_p < 1.0 {
                    chain.push(Box::new(TopP::new(params.top_p)));
                }
                if params.min_p > 0.0 {
                    chain.push(Box::new(MinP::new(params.min_p)));
                }
            }
        }

        Self {
            samplers: chain,
            greedy: is_greedy,
        }
    }

    pub fn sample(
        &mut self,
        logits: &mut [f32],
        context_tokens: &[u32],
        rng: &mut impl Rng,
    ) -> u32 {
        for sampler in &mut self.samplers {
            sampler.apply(logits, context_tokens);
        }

        if self.greedy {
            return greedy::greedy_sample(logits);
        }

        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        sample_from_probs(&probs, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::GenerateParams;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn default_params() -> GenerateParams {
        GenerateParams {
            max_tokens: 10,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat: None,
            stop_tokens: vec![],
            ignore_eos: false,
            seed: Some(42),
            constraint: None,
            spec_enabled: false,
            spec_k: 4,
            spec_depth: 0.5,
        }
    }

    #[test]
    fn test_sampler_chain_greedy() {
        let mut params = default_params();
        params.temperature = 0.0;
        let mut chain = SamplerChain::from_params(&params);
        let mut logits = vec![1.0f32, 5.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(42);
        let token = chain.sample(&mut logits, &[], &mut rng);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_sampler_chain_temperature_top_k() {
        let mut params = default_params();
        params.temperature = 1.0;
        params.top_k = 2;
        let mut chain = SamplerChain::from_params(&params);
        let mut logits = vec![1.0f32, 3.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(42);
        let token = chain.sample(&mut logits, &[], &mut rng);
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_sampler_chain_repetition_penalty_order() {
        let mut params = default_params();
        params.repetition_penalty = 2.0;
        params.temperature = 0.0;
        let mut chain = SamplerChain::from_params(&params);
        // logits=[1.0, 5.0, 2.0], context=[1] → logits[1]/=2 → [1.0, 2.5, 2.0]
        // greedy → token 1 (2.5가 최대)
        let mut logits = vec![1.0f32, 5.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(42);
        let token = chain.sample(&mut logits, &[1], &mut rng);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_sampler_chain_no_samplers_when_defaults() {
        let mut params = default_params();
        params.temperature = 0.0;
        let mut chain = SamplerChain::from_params(&params);
        let mut logits = vec![0.1f32, 0.9, 0.5];
        let mut rng = SmallRng::seed_from_u64(42);
        assert_eq!(chain.sample(&mut logits, &[], &mut rng), 1);
    }
}
