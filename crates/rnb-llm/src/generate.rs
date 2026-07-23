use crate::constrained::StructuredDecoder;
pub use crate::sampler::mirostat::MirostatParams;
use crate::sampler::SamplerChain;
use crate::tokenizer::{TokenStreamDecoder, Tokenizer};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

thread_local! {
    static ACTIVE_CANCELLATION: RefCell<Option<GenerationCancellation>> =
        const { RefCell::new(None) };
}

#[derive(Clone, Default)]
pub struct GenerationCancellation {
    cancelled: Arc<AtomicBool>,
}

impl GenerationCancellation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

struct CancellationScope {
    previous: Option<GenerationCancellation>,
}

impl CancellationScope {
    fn enter(cancellation: &GenerationCancellation) -> Self {
        let previous =
            ACTIVE_CANCELLATION.with(|active| active.borrow_mut().replace(cancellation.clone()));
        Self { previous }
    }
}

impl Drop for CancellationScope {
    fn drop(&mut self) {
        ACTIVE_CANCELLATION.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

pub(crate) fn check_generation_cancellation() -> crate::error::Result<()> {
    let cancelled = ACTIVE_CANCELLATION.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(GenerationCancellation::is_cancelled)
    });
    if cancelled {
        Err(crate::error::LlmError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct GeneratedTextStream {
    decoder: TokenStreamDecoder,
    text: String,
    callback_stopped: bool,
}

impl GeneratedTextStream {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(
        &mut self,
        tokenizer: &Tokenizer,
        token: u32,
        callback: &mut impl FnMut(&str) -> bool,
    ) -> bool {
        let piece = self.decoder.push(tokenizer, token);
        self.text.push_str(&piece);
        if piece.is_empty() {
            return true;
        }
        let should_continue = callback(&piece);
        self.callback_stopped = !should_continue;
        should_continue
    }

    pub(crate) fn finish(mut self, callback: &mut impl FnMut(&str) -> bool) -> String {
        let piece = self.decoder.finish();
        self.text.push_str(&piece);
        if !piece.is_empty() && !self.callback_stopped {
            callback(&piece);
        }
        self.text
    }
}

/// 텍스트 생성 파라미터
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub mirostat: Option<MirostatParams>,
    pub stop_tokens: Vec<u32>,
    /// Suppress the model's default EOS logit while still honoring explicit stop tokens.
    pub ignore_eos: bool,
    pub seed: Option<u64>,
    pub constraint: Option<crate::GenerationConstraint>,
    /// Speculative decoding 활성화 여부
    pub spec_enabled: bool,
    /// Speculative draft 토큰 수 (한 라운드당)
    pub spec_k: usize,
    /// Draft early-exit depth (0.0~1.0, 전체 레이어 대비 비율)
    pub spec_depth: f32,
}

impl GenerateParams {
    pub(crate) fn should_stop(&self, token: u32, eos: u32) -> bool {
        self.stop_tokens.contains(&token) || (!self.ignore_eos && token == eos)
    }

    pub(crate) fn suppress_eos_logit(
        &self,
        logits: &mut [f32],
        eos: u32,
    ) -> crate::error::Result<()> {
        if !self.ignore_eos {
            return Ok(());
        }
        let logits_len = logits.len();
        let eos_logit = logits.get_mut(eos as usize).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "EOS token {eos} exceeds logits length {}",
                logits_len
            ))
        })?;
        *eos_logit = f32::NEG_INFINITY;
        Ok(())
    }
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
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
            seed: None,
            constraint: None,
            spec_enabled: false,
            spec_k: 4,
            spec_depth: 0.5,
        }
    }
}

/// 텍스트 생성 결과
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub tokens_per_second: f32,
    pub generated_token_ids: Vec<u32>,
    pub prompt_token_ids: Vec<u32>,
    pub cached_prompt_tokens: usize,
}

impl GenerateResult {
    pub fn new(
        text: String,
        tokens_generated: usize,
        prompt_tokens: usize,
        elapsed_secs: f32,
        generated_token_ids: Vec<u32>,
    ) -> Self {
        let tokens_per_second = if elapsed_secs > 0.0 {
            tokens_generated as f32 / elapsed_secs
        } else {
            0.0
        };
        Self {
            text,
            tokens_generated,
            prompt_tokens,
            tokens_per_second,
            generated_token_ids,
            prompt_token_ids: Vec::new(),
            cached_prompt_tokens: 0,
        }
    }

    pub(crate) fn with_prompt_token_ids(mut self, prompt_token_ids: Vec<u32>) -> Self {
        self.prompt_token_ids = prompt_token_ids;
        self
    }
}

/// 스트리밍 생성 루프
pub fn generate_stream(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
    callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    if params.constraint.is_some() {
        return generate_stream_impl(engine, prompt, params, callback);
    }
    match select_generate_route(
        params.spec_enabled,
        engine.has_weights(),
        engine.mtp_spec_requested()
            && (crate::mtp_generate::mtp_greedy_verify_allowed(params)
                || engine.mtp_explicitly_forced()),
    ) {
        GenerateRoute::Mtp => {
            // pm118: 미설정/auto 로 진입한 MTP 는 spec_k 도 모델별 auto policy 값
            // (GLM nextn 1층 k=1 등). 명시 truthy 강제일 때만 caller 값 유지.
            let spec_k = engine.mtp_effective_spec_k(params.spec_k);
            if spec_k == params.spec_k {
                crate::mtp_generate::generate_stream_mtp(engine, prompt, params, callback)
            } else {
                let params = GenerateParams {
                    spec_k,
                    ..params.clone()
                };
                crate::mtp_generate::generate_stream_mtp(engine, prompt, &params, callback)
            }
        }
        GenerateRoute::Speculative => {
            crate::speculative::generate_stream_speculative(engine, prompt, params, callback)
        }
        GenerateRoute::Standard => generate_stream_impl(engine, prompt, params, callback),
    }
}

pub fn generate_stream_cancellable(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
    cancellation: &GenerationCancellation,
    callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let _scope = CancellationScope::enter(cancellation);
    check_generation_cancellation()?;
    generate_stream(engine, prompt, params, callback)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerateRoute {
    Standard,
    Speculative,
    Mtp,
}

fn select_generate_route(
    spec_enabled: bool,
    engine_has_weights: bool,
    mtp_requested: bool,
) -> GenerateRoute {
    if !engine_has_weights {
        return GenerateRoute::Standard;
    }
    if mtp_requested {
        return GenerateRoute::Mtp;
    }
    if spec_enabled {
        return GenerateRoute::Speculative;
    }
    GenerateRoute::Standard
}

/// 표준 (non-speculative) 스트리밍 생성 루프
pub(crate) fn generate_stream_impl(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
    callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let prompt_tokens = tokenize_prompt(engine, prompt);
    generate_stream_impl_with_tokens(engine, prompt_tokens, params, None, callback)
}

pub(crate) fn generate_stream_resuming(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
    state: &crate::engine::EngineSequenceState,
    callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let Some(prompt_tokens) = resumed_prompt_tokens(engine, prompt, state) else {
        return generate_stream(engine, prompt, params, callback);
    };

    let cached_tokens = state.token_len();
    engine.restore_sequence_state(state)?;
    let route = if params.constraint.is_some() {
        GenerateRoute::Standard
    } else {
        select_generate_route(
            params.spec_enabled,
            engine.has_weights(),
            engine.mtp_spec_requested()
                && (crate::mtp_generate::mtp_greedy_verify_allowed(params)
                    || engine.mtp_explicitly_forced()),
        )
    };
    let mut result = match route {
        GenerateRoute::Mtp => {
            let spec_k = engine.mtp_effective_spec_k(params.spec_k);
            if spec_k == params.spec_k {
                crate::mtp_generate::generate_stream_mtp_resuming(
                    engine,
                    prompt_tokens,
                    cached_tokens,
                    params,
                    callback,
                )
            } else {
                let params = GenerateParams {
                    spec_k,
                    ..params.clone()
                };
                crate::mtp_generate::generate_stream_mtp_resuming(
                    engine,
                    prompt_tokens,
                    cached_tokens,
                    &params,
                    callback,
                )
            }
        }
        GenerateRoute::Standard | GenerateRoute::Speculative => generate_stream_impl_with_tokens(
            engine,
            prompt_tokens,
            params,
            Some(cached_tokens),
            callback,
        ),
    }?;
    result.cached_prompt_tokens = cached_tokens;
    Ok(result)
}

pub(crate) fn generate_stream_resuming_cancellable(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
    state: &crate::engine::EngineSequenceState,
    cancellation: &GenerationCancellation,
    callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let _scope = CancellationScope::enter(cancellation);
    check_generation_cancellation()?;
    generate_stream_resuming(engine, prompt, params, state, callback)
}

fn tokenize_prompt(engine: &crate::engine::Engine, prompt: &str) -> Vec<u32> {
    let mut prompt_tokens = Vec::new();
    if engine.tokenizer.should_add_bos() {
        prompt_tokens.push(engine.tokenizer.vocab.special.bos);
    }
    prompt_tokens.extend(engine.tokenizer.encode(prompt));
    prompt_tokens
}

fn resumed_prompt_tokens(
    engine: &crate::engine::Engine,
    prompt: &str,
    state: &crate::engine::EngineSequenceState,
) -> Option<Vec<u32>> {
    let prompt_tokens = tokenize_prompt(engine, prompt);
    if state.matches_prompt_prefix(&prompt_tokens) {
        return (state.token_len() < prompt_tokens.len()).then_some(prompt_tokens);
    }

    let state_tokens = state.resume_token_ids();
    if let Some((prompt_prefix, append_text)) = state.prompt_resume_alignment() {
        if let Some(prompt_suffix) = prompt.strip_prefix(prompt_prefix) {
            let mut resumed_tokens = state_tokens.to_vec();
            resumed_tokens.extend(engine.tokenizer.encode(append_text));
            resumed_tokens.extend(engine.tokenizer.encode(prompt_suffix));
            return (state.token_len() < resumed_tokens.len()).then_some(resumed_tokens);
        }
    }

    let decoded_tokens = if engine.tokenizer.should_add_bos()
        && state_tokens.first() == Some(&engine.tokenizer.vocab.special.bos)
    {
        &state_tokens[1..]
    } else {
        state_tokens
    };
    let decoded_prefix = engine.tokenizer.decode(decoded_tokens);
    let suffix = prompt.strip_prefix(&decoded_prefix)?;
    let mut resumed_tokens = state_tokens.to_vec();
    resumed_tokens.extend(engine.tokenizer.encode(suffix));
    (state.token_len() < resumed_tokens.len()).then_some(resumed_tokens)
}

fn generate_stream_impl_with_tokens(
    engine: &mut crate::engine::Engine,
    prompt_tokens: Vec<u32>,
    params: &GenerateParams,
    resume_from: Option<usize>,
    mut callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let start = Instant::now();
    check_generation_cancellation()?;
    let prompt_len = prompt_tokens.len();

    let mut rng = match params.seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    };

    let mut sampler = SamplerChain::from_params(params);
    let mut constraint = params
        .constraint
        .as_ref()
        .map(|constraint| StructuredDecoder::new(&engine.tokenizer, constraint))
        .transpose()?;

    let forward_tokens = if let Some(resume_from) = resume_from {
        &prompt_tokens[resume_from..]
    } else {
        engine.clear_sequence_state()?;
        prompt_tokens.as_slice()
    };
    let mut logits = if constraint.is_some() {
        engine.forward_prompt_with_logits(forward_tokens)?
    } else {
        engine.forward_prompt(forward_tokens)?
    };
    let mut backend_argmax = engine.last_backend_argmax_token();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut generated_text = GeneratedTextStream::new();

    for _ in 0..params.max_tokens {
        check_generation_cancellation()?;
        let eos = engine.tokenizer.vocab.special.eos;
        let token = if let Some(constraint) = constraint.as_mut() {
            constraint.mask_logits(&mut logits)?;
            params.suppress_eos_logit(&mut logits, eos)?;
            sampler.sample(&mut logits, &generated_tokens, &mut rng)
        } else if let Some(token) =
            backend_argmax.filter(|token| !params.ignore_eos || *token != eos)
        {
            token
        } else {
            params.suppress_eos_logit(&mut logits, eos)?;
            sampler.sample(&mut logits, &generated_tokens, &mut rng)
        };

        if params.should_stop(token, eos) {
            break;
        }
        if let Some(constraint) = constraint.as_mut() {
            constraint.consume_token(token)?;
        }

        generated_tokens.push(token);
        let should_continue = generated_text.push(&engine.tokenizer, token, &mut callback);
        check_generation_cancellation()?;
        if !should_continue
            || constraint
                .as_ref()
                .is_some_and(StructuredDecoder::is_stopped)
        {
            break;
        }

        if constraint.is_some() {
            logits = engine.forward_with_logits(&[token])?;
            backend_argmax = None;
        } else {
            logits = engine.forward(&[token])?;
            backend_argmax = engine.last_backend_argmax_token();
        }
    }

    check_generation_cancellation()?;
    let elapsed = start.elapsed().as_secs_f32();
    Ok(GenerateResult::new(
        generated_text.finish(&mut callback),
        generated_tokens.len(),
        prompt_len,
        elapsed,
        generated_tokens,
    )
    .with_prompt_token_ids(prompt_tokens))
}

/// Blocking 생성 (전체 결과 반환)
pub fn generate(
    engine: &mut crate::engine::Engine,
    prompt: &str,
    params: &GenerateParams,
) -> crate::error::Result<GenerateResult> {
    generate_stream(engine, prompt, params, |_| true)
}

/// Filters decoded token text against OpenAI-style string stop sequences.
///
/// The filter retains a suffix that might become a stop sequence after the
/// next token piece, so callers never emit a partial stop marker.
pub struct TextStopFilter {
    stops: Vec<String>,
    pending: String,
    matched: bool,
}

impl TextStopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            matched: false,
        }
    }

    pub fn matched(&self) -> bool {
        self.matched
    }

    pub fn push(&mut self, piece: &str, mut emit: impl FnMut(&str) -> bool) -> bool {
        if self.stops.is_empty() {
            return emit(piece);
        }
        self.pending.push_str(piece);
        if let Some(index) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            if index != 0 {
                emit(&self.pending[..index]);
            }
            self.pending.clear();
            self.matched = true;
            return false;
        }

        let keep = self
            .pending
            .char_indices()
            .filter_map(|(index, _)| {
                let suffix = &self.pending[index..];
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(suffix))
                    .then_some(suffix.len())
            })
            .max()
            .unwrap_or(0);
        let emit_len = self.pending.len() - keep;
        if emit_len == 0 {
            return true;
        }
        if !emit(&self.pending[..emit_len]) {
            return false;
        }
        self.pending.drain(..emit_len);
        true
    }

    pub fn finish(&mut self, mut emit: impl FnMut(&str) -> bool) -> bool {
        if self.pending.is_empty() {
            return true;
        }
        let emitted = emit(&self.pending);
        self.pending.clear();
        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, ModelMetadata};
    use crate::tokenizer::{
        bpe::Tokenizer as BpeTokenizer,
        vocab::{SpecialTokens, Vocab},
    };

    /// vocab_size=16, greedy는 모든 logit이 같으면 마지막 인덱스(15)를 반환
    /// eos를 15보다 작은 값으로 설정하면 greedy가 eos를 선택하지 않음
    fn make_engine_non_eos_greedy(eos_id: u32) -> Engine {
        // vocab_size=16, greedy → 항상 index 15 반환 (eos_id != 15이면 계속 생성)
        let vocab_size = 16usize;
        let tokens: Vec<String> = (0..vocab_size).map(|i| format!("t{}", i)).collect();
        let special = SpecialTokens {
            bos: 1,
            eos: eos_id,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let tokenizer = BpeTokenizer::new(vocab, vec![]);
        let metadata = ModelMetadata {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size,
            max_seq_len: 64,
            hidden_dim: 8,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            head_count_kv_per_layer: None,
            embedding_length_per_layer_input: 0,
            expert_used_count: 0,
            expert_weights_scale: 1.0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
        };
        Engine::mock(tokenizer, metadata)
    }

    /// eos=greedy 선택 토큰(마지막 인덱스) 설정 → 즉시 종료되는 engine
    fn make_engine_eos_immediately() -> Engine {
        // vocab_size=16, greedy → index 15, eos=15 → 즉시 종료
        let vocab_size = 16usize;
        let eos_id = (vocab_size - 1) as u32;
        let tokens: Vec<String> = (0..vocab_size).map(|i| format!("t{}", i)).collect();
        let special = SpecialTokens {
            bos: 1,
            eos: eos_id,
            pad: None,
        };
        let vocab = Vocab::new(tokens, special);
        let tokenizer = BpeTokenizer::new(vocab, vec![]);
        let metadata = ModelMetadata {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size,
            max_seq_len: 64,
            hidden_dim: 8,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            head_count_kv_per_layer: None,
            embedding_length_per_layer_input: 0,
            expert_used_count: 0,
            expert_weights_scale: 1.0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
        };
        Engine::mock(tokenizer, metadata)
    }

    #[test]
    fn test_generate_max_tokens_limit() {
        // eos=9, greedy → 15 (≠ 9) → max_tokens=5까지 생성
        let mut engine = make_engine_non_eos_greedy(9);
        let params = GenerateParams {
            max_tokens: 5,
            temperature: 0.0,
            stop_tokens: vec![],
            seed: Some(0),
            ..GenerateParams::default()
        };
        let result = engine.generate("prompt", &params).expect("generate ok");
        assert_eq!(result.tokens_generated, 5);
    }

    #[test]
    fn test_generate_stream_callback_abort() {
        // eos=9, greedy → 15 → 계속 생성, 3번째 콜백에서 중단
        let mut engine = make_engine_non_eos_greedy(9);
        let params = GenerateParams {
            max_tokens: 10,
            temperature: 0.0,
            stop_tokens: vec![],
            seed: Some(0),
            ..GenerateParams::default()
        };

        let mut count = 0usize;
        let result = engine
            .generate_stream("hi", &params, |_piece| {
                count += 1;
                count < 3
            })
            .expect("generate_stream ok");

        assert_eq!(result.tokens_generated, 3);
    }

    #[test]
    fn test_generate_stop_token() {
        // greedy → 15 = eos → 즉시 종료
        let mut engine = make_engine_eos_immediately();
        let params = GenerateParams {
            max_tokens: 100,
            temperature: 0.0,
            stop_tokens: vec![],
            seed: Some(0),
            ..GenerateParams::default()
        };
        let result = engine.generate("x", &params).expect("ok");
        assert_eq!(result.tokens_generated, 0);
    }

    #[test]
    fn test_generate_ignore_eos_uses_next_best_token_until_limit() {
        let mut engine = make_engine_eos_immediately();
        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            ignore_eos: true,
            seed: Some(0),
            ..GenerateParams::default()
        };
        let result = engine.generate("x", &params).expect("ok");
        assert_eq!(result.tokens_generated, 3);
        assert_eq!(result.generated_token_ids, vec![14, 14, 14]);
    }

    #[test]
    fn test_generate_result_fields() {
        // eos=9, greedy → 15 → 3개 생성
        let mut engine = make_engine_non_eos_greedy(9);
        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            ..GenerateParams::default()
        };
        let result = engine.generate("ab", &params).expect("ok");
        assert_eq!(result.tokens_generated, 3);
        assert!(result.tokens_per_second >= 0.0);
    }

    /// mv27-task10b-4c-3: verify the fullpath sampler-bypass mechanism.
    ///
    /// Without bypass: vocab_size=16, all-zero logits, temperature=0 (greedy)
    /// → sampler returns the last index (15). With bypass:
    /// `scratch.backend_argmax_token = Some(7)` is honored before the sampler
    /// chain, so the loop emits token 7 every iteration regardless of logits.
    ///
    /// This is a pure-mechanism test — it does not exercise GPU dispatch.
    /// `force_backend_argmax_token_for_test` mimics what the real GPU
    /// fullpath does (write `output.last_token_id` into scratch). Mock engine
    /// has `weights=None`, so `forward_prefill_cpu` returns zero logits and
    /// never overwrites `scratch.backend_argmax_token`, keeping the seeded
    /// value across all forward calls in the loop.
    #[test]
    fn test_generate_fullpath_backend_argmax_bypass() {
        // eos=9, greedy → would-be 15 if sampler ran. We seed Some(7), so the
        // bypass should emit 7 instead. 7 != eos (9) so the loop runs to
        // max_tokens.
        let mut engine = make_engine_non_eos_greedy(9);
        engine.force_backend_argmax_token_for_test(Some(7));

        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            stop_tokens: vec![],
            seed: Some(0),
            ..GenerateParams::default()
        };

        let mut emitted: Vec<String> = Vec::new();
        let result = engine
            .generate_stream("hi", &params, |piece| {
                emitted.push(piece.to_string());
                true
            })
            .expect("generate_stream ok");

        // bypass should have emitted token 7 ("t7") on every iteration.
        assert_eq!(result.tokens_generated, 3);
        assert_eq!(emitted, vec!["t7", "t7", "t7"]);
    }

    /// Bypass off (None) — sampler path is preserved unchanged. Equivalent to
    /// the existing `test_generate_max_tokens_limit` (greedy → 15) but with
    /// the scratch buffer explicitly seeded to None to prove the
    /// `if let Some(t)` branch never triggers when partial-offload paths run.
    #[test]
    fn test_generate_fullpath_bypass_disabled_when_none() {
        let mut engine = make_engine_non_eos_greedy(9);
        engine.force_backend_argmax_token_for_test(None);

        let params = GenerateParams {
            max_tokens: 2,
            temperature: 0.0,
            stop_tokens: vec![],
            seed: Some(0),
            ..GenerateParams::default()
        };

        let mut emitted: Vec<String> = Vec::new();
        let result = engine
            .generate_stream("hi", &params, |piece| {
                emitted.push(piece.to_string());
                true
            })
            .expect("generate_stream ok");

        // No bypass → greedy returns last index (15) → "t15" emitted.
        assert_eq!(result.tokens_generated, 2);
        assert_eq!(emitted, vec!["t15", "t15"]);
    }

    #[test]
    fn mtp_request_routes_to_dedicated_mtp_generation_without_spec_flag() {
        assert_eq!(select_generate_route(false, true, true), GenerateRoute::Mtp);
    }

    #[test]
    fn mtp_request_takes_priority_over_generic_speculation() {
        assert_eq!(select_generate_route(true, true, true), GenerateRoute::Mtp);
    }

    #[test]
    fn mtp_request_does_not_route_mock_engines_without_weights() {
        assert_eq!(
            select_generate_route(false, false, true),
            GenerateRoute::Standard
        );
    }

    #[test]
    fn cancelled_generation_stops_before_sequence_mutation() {
        let mut engine = make_engine_non_eos_greedy(9);
        let cancellation = GenerationCancellation::new();
        cancellation.cancel();
        let error = generate_stream_cancellable(
            &mut engine,
            "hi",
            &GenerateParams::default(),
            &cancellation,
            |_| true,
        )
        .unwrap_err();

        assert!(matches!(error, crate::error::LlmError::Cancelled));
        assert_eq!(engine.kv_cache.current_len(), 0);
    }

    #[test]
    fn text_stop_filter_handles_sequence_split_across_tokens() {
        let mut filter = TextStopFilter::new(vec!["</stop>".to_string()]);
        let mut output = String::new();

        assert!(filter.push("answer</st", |text| {
            output.push_str(text);
            true
        }));
        assert!(!filter.push("op>ignored", |text| {
            output.push_str(text);
            true
        }));

        assert_eq!(output, "answer");
        assert!(filter.matched());
    }

    #[test]
    fn text_stop_filter_flushes_partial_prefix_at_end() {
        let mut filter = TextStopFilter::new(vec!["END".to_string()]);
        let mut output = String::new();
        assert!(filter.push("value E", |text| {
            output.push_str(text);
            true
        }));
        assert!(filter.finish(|text| {
            output.push_str(text);
            true
        }));
        assert_eq!(output, "value E");
    }

    #[test]
    fn resumed_prompt_preserves_generated_token_path_across_bpe_reencoding() {
        let tokens = ["<unk>", "<bos>", "<eos>", "a", "b", "ab", "c"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let mut tokenizer = BpeTokenizer::new_gpt2(Vocab::new(tokens, special), vec![(3, 4)]);
        tokenizer.set_add_bos_token(false);
        let mut engine = make_engine_non_eos_greedy(9);
        engine.tokenizer = tokenizer;
        engine.kv_cache.append(0, 0, &[0.0, 0.0], &[0.0, 0.0]);
        let state = engine.capture_sequence_state(vec![3, 4]).unwrap();

        assert_eq!(tokenize_prompt(&engine, "abc"), vec![5, 6]);
        assert_eq!(
            resumed_prompt_tokens(&engine, "abc", &state),
            Some(vec![3, 4, 6])
        );
        assert_eq!(state.token_len(), 1);
    }

    #[test]
    fn resumed_prompt_keeps_generation_only_prefix_before_visible_output() {
        let tokens = ["<unk>", "<bos>", "<eos>", "a", "b", "ab", "c", "h"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let special = SpecialTokens {
            bos: 1,
            eos: 2,
            pad: None,
        };
        let mut tokenizer = BpeTokenizer::new_gpt2(Vocab::new(tokens, special), vec![(3, 4)]);
        tokenizer.set_add_bos_token(false);
        let mut engine = make_engine_non_eos_greedy(9);
        engine.tokenizer = tokenizer;
        engine.kv_cache.append(0, 0, &[0.0, 0.0], &[0.0, 0.0]);
        let state = engine
            .capture_sequence_state_with_prompt_alignment(
                vec![7, 3, 4],
                "ab".to_string(),
                String::new(),
            )
            .unwrap();

        assert_eq!(
            resumed_prompt_tokens(&engine, "abc", &state),
            Some(vec![7, 3, 4, 6])
        );
    }
}
