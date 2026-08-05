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

/// 필터링 뒤 살아 있는 logit만 확률 버퍼에 보존해 같은 분포와 RNG 소비로 샘플링한다.
///
/// Top-k/top-p가 대부분을 `-inf`로 만든 뒤에도 전체 vocab을 복사하고 `exp`하던 비용을
/// 피한다. 누적 순서는 원래 token index 순서라 dense softmax 경로와 결과가 같다.
fn sample_from_logits(logits: &[f32], rng: &mut impl Rng) -> u32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut weighted = Vec::new();
    let mut sum = 0.0f32;
    for (index, &logit) in logits.iter().enumerate() {
        if logit == f32::NEG_INFINITY {
            continue;
        }
        let probability = (logit - max).exp();
        sum += probability;
        weighted.push((index, probability));
    }
    if sum > 0.0 {
        for (_, probability) in &mut weighted {
            *probability /= sum;
        }
    }

    let r = rng.gen::<f32>();
    let mut cumsum = 0.0f32;
    for (index, probability) in weighted {
        cumsum += probability;
        if r <= cumsum {
            return index as u32;
        }
    }
    (logits.len() - 1) as u32
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

        sample_from_logits(logits, rng)
    }

    /// sampler processor를 적용한 뒤 정규화된 확률 분포를 `probs`에 채운다.
    ///
    /// `sample()`의 token-only 결과 대신 processed 분포 자체가 필요한 호출부가 쓴다.
    /// speculative accept 확률은 draft와 target 양쪽의 processed 분포를 같은 순서의
    /// 같은 processor로 만들어야 정의되므로, 그 계산은 이 API를 거쳐야 한다.
    /// greedy chain은 processor가 비어 있어 raw softmax와 같다.
    pub(crate) fn processed_probs_into(
        &mut self,
        logits: &mut [f32],
        context_tokens: &[u32],
        probs: &mut Vec<f32>,
    ) {
        for sampler in &mut self.samplers {
            sampler.apply(logits, context_tokens);
        }
        probs.clear();
        probs.extend_from_slice(logits);
        softmax_inplace(probs);
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

    #[test]
    fn sparse_sampling_matches_dense_softmax_for_fixed_rng() {
        let logits = vec![
            f32::NEG_INFINITY,
            1.25,
            f32::NEG_INFINITY,
            -0.75,
            3.5,
            f32::NEG_INFINITY,
            2.0,
            -4.0,
        ];
        for seed in 0..256 {
            let mut dense_probs = logits.clone();
            softmax_inplace(&mut dense_probs);
            let mut dense_rng = SmallRng::seed_from_u64(seed);
            let expected = sample_from_probs(&dense_probs, &mut dense_rng);

            let mut sparse_rng = SmallRng::seed_from_u64(seed);
            assert_eq!(sample_from_logits(&logits, &mut sparse_rng), expected);
        }
    }

    fn dense_reference_sample(
        chain: &mut SamplerChain,
        logits: &mut [f32],
        context_tokens: &[u32],
        rng: &mut impl Rng,
    ) -> u32 {
        for sampler in &mut chain.samplers {
            sampler.apply(logits, context_tokens);
        }
        if chain.greedy {
            return greedy::greedy_sample(logits);
        }
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        sample_from_probs(&probs, rng)
    }

    #[test]
    fn compact_chain_matches_dense_reference_with_penalties() {
        let mut params = default_params();
        params.temperature = 0.8;
        params.top_k = 10;
        params.top_p = 0.9;
        params.min_p = 0.05;
        params.repetition_penalty = 1.1;
        params.presence_penalty = 0.2;
        params.frequency_penalty = 0.05;
        let source = (0..4096)
            .map(|index| ((index * 53) % 257) as f32 * 0.03125 - 4.0)
            .collect::<Vec<_>>();
        let context = [3, 3, 17, 511, 1024, 2048, 4095];

        for seed in 0..64 {
            let mut expected_logits = source.clone();
            let mut expected_chain = SamplerChain::from_params(&params);
            let mut expected_rng = SmallRng::seed_from_u64(seed);
            let expected = dense_reference_sample(
                &mut expected_chain,
                &mut expected_logits,
                &context,
                &mut expected_rng,
            );

            let mut actual_logits = source.clone();
            let mut actual_chain = SamplerChain::from_params(&params);
            let mut actual_rng = SmallRng::seed_from_u64(seed);
            let actual = actual_chain.sample(&mut actual_logits, &context, &mut actual_rng);
            assert_eq!(actual_logits, expected_logits, "seed={seed}");
            assert_eq!(actual, expected, "seed={seed}");
        }
    }
}
