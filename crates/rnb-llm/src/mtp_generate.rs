use crate::engine::verify_window::{MtpVerifyBonus, MtpVerifyWindowRequest};
use crate::engine::Engine;
use crate::generate::{GenerateParams, GenerateResult};
use crate::sampler::SamplerChain;
use crate::speculative::{
    replay_committed_verify_prefix_argmax, verify_state_plan_after_batch_round, SpecCheckpoint,
    VerifyStatePlan,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

struct MtpStats {
    rounds: usize,
    drafted: usize,
    accepted: usize,
    carried: usize,
    target_verify_steps: usize,
    target_verify_invocations: usize,
}

impl MtpStats {
    fn add_target_verify(&mut self, logical_steps: usize, invocations: usize) {
        self.target_verify_steps += logical_steps;
        self.target_verify_invocations += invocations;
    }

    fn add_target_replay_invocations(&mut self, invocations: usize) {
        self.target_verify_invocations += invocations;
    }

    fn accept_rate(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32 * 100.0
        }
    }

    fn report(&self) -> String {
        format!(
            "  [MTP] rounds={}, drafted={}, accepted={}, carried={}, rate={:.1}%, target_verify_steps={}, target_verify_invocations={}",
            self.rounds,
            self.drafted,
            self.accepted,
            self.carried,
            self.accept_rate(),
            self.target_verify_steps,
            self.target_verify_invocations,
        )
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Default)]
struct MtpPhaseTimings {
    checkpoint_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    retain_ms: f64,
}

impl MtpPhaseTimings {
    fn report(&self, rounds: usize) -> String {
        let rounds = rounds.max(1) as f64;
        format!(
            "  [MTP] phase_ms_per_round checkpoint={:.1}ms/round draft={:.1}ms/round verify={:.1}ms/round retain={:.1}ms/round",
            self.checkpoint_ms / rounds,
            self.draft_ms / rounds,
            self.verify_ms / rounds,
            self.retain_ms / rounds,
        )
    }
}

fn mtp_greedy_verify_allowed(params: &GenerateParams) -> bool {
    params.temperature == 0.0
        && params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
        && params.mirostat.is_none()
}

fn mtp_target_needs_sequential(architecture: rnb_loader::Architecture) -> bool {
    matches!(
        architecture,
        rnb_loader::Architecture::Qwen35
            | rnb_loader::Architecture::Qwen35MoE
            | rnb_loader::Architecture::Gemma4
            | rnb_loader::Architecture::GlmDsa
    )
}

fn mtp_effective_k(
    requested_k: usize,
    tokens_remaining: usize,
    target_needs_sequential: bool,
    sequential_multi: bool,
) -> usize {
    let capped = requested_k.max(1).min(tokens_remaining);
    if target_needs_sequential && !sequential_multi {
        capped.min(1)
    } else {
        capped
    }
}

fn mtp_batch_effective_k(requested_k: usize, tokens_remaining: usize) -> usize {
    requested_k.max(1).min(tokens_remaining)
}

fn mtp_batch_verify_default_enabled(_target_needs_sequential: bool) -> bool {
    false
}

fn mtp_batch_verify_requested(forced: bool, disabled: bool, default_enabled: bool) -> bool {
    !disabled && (forced || default_enabled)
}

fn mtp_external_batch_verify_allowed(
    architecture: rnb_loader::Architecture,
    forced: bool,
    disabled: bool,
    default_enabled: bool,
) -> bool {
    mtp_batch_verify_requested(forced, disabled, default_enabled)
        && !mtp_target_needs_sequential(architecture)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpVerifyExecution {
    Sequential,
    BatchPrefill,
    DeviceResident,
}

fn mtp_verify_execution(device_verify: bool, batch_verify: bool) -> MtpVerifyExecution {
    if device_verify {
        MtpVerifyExecution::DeviceResident
    } else if batch_verify {
        MtpVerifyExecution::BatchPrefill
    } else {
        MtpVerifyExecution::Sequential
    }
}

fn mtp_full_engine_checkpoint_required(
    verify_execution: MtpVerifyExecution,
    target_has_recurrent_state: bool,
) -> bool {
    target_has_recurrent_state && verify_execution == MtpVerifyExecution::BatchPrefill
}

fn mtp_verify_state_plan(
    verify_execution: MtpVerifyExecution,
    target_has_recurrent_state: bool,
    spec_k: usize,
    n_accepted: usize,
    no_bonus_verify: bool,
    verify_window_tokens: usize,
    prefix_state_available: bool,
) -> VerifyStatePlan {
    debug_assert_eq!(verify_execution, MtpVerifyExecution::BatchPrefill);
    let mut state_plan =
        verify_state_plan_after_batch_round(target_has_recurrent_state, spec_k, n_accepted);
    let current_state_matches_committed =
        no_bonus_verify && state_plan.committed_verify_tokens == verify_window_tokens;
    if current_state_matches_committed || prefix_state_available {
        state_plan.replay_committed_prefix = false;
    }
    state_plan
}

fn mtp_prefix_state_restore_required(
    state_plan: Option<&VerifyStatePlan>,
    committed_tokens: usize,
    verify_window_tokens: usize,
) -> bool {
    committed_tokens != verify_window_tokens
        && !state_plan.is_some_and(|plan| plan.replay_committed_prefix)
}

fn mtp_prefix_snapshot_tokens(
    verify_execution: MtpVerifyExecution,
    target_has_recurrent_state: bool,
    verify_request: &MtpVerifyWindowRequest,
) -> Vec<usize> {
    let mut prefix_tokens = verify_request.prefix_tokens();
    if cfg!(not(feature = "cuda"))
        && verify_execution == MtpVerifyExecution::BatchPrefill
        && target_has_recurrent_state
    {
        prefix_tokens.retain(|prefix_tokens| *prefix_tokens == 1);
    }
    prefix_tokens
}

fn mtp_prefix_restore_base_kv_len(
    verify_execution: MtpVerifyExecution,
    full_checkpoint_kv_len: Option<usize>,
    round_base_kv_len: usize,
) -> crate::error::Result<usize> {
    match verify_execution {
        MtpVerifyExecution::BatchPrefill => full_checkpoint_kv_len.ok_or_else(|| {
            crate::error::LlmError::Forward(
                "MTP batch prefix restore requires full engine checkpoint".to_string(),
            )
        }),
        MtpVerifyExecution::DeviceResident => Ok(round_base_kv_len),
        MtpVerifyExecution::Sequential => Err(crate::error::LlmError::Forward(
            "MTP sequential verify cannot restore a verify-window prefix".to_string(),
        )),
    }
}

fn mtp_no_bonus_verify_enabled(
    verify_execution: MtpVerifyExecution,
    env_override: Option<bool>,
) -> bool {
    env_override.unwrap_or(matches!(
        verify_execution,
        MtpVerifyExecution::DeviceResident
    ))
}

fn mtp_no_bonus_verify_for_round(
    verify_execution: MtpVerifyExecution,
    env_override: Option<bool>,
    k: usize,
) -> bool {
    if let Some(override_value) = env_override {
        return override_value;
    }
    mtp_no_bonus_verify_enabled(verify_execution, None) && k > 1
}

fn mtp_no_bonus_verify_env_override() -> Option<bool> {
    crate::runtime::spec_batch_no_bonus_override()
}

#[cfg(test)]
fn mtp_verified_carry_depth(
    enabled: bool,
    verify_execution: MtpVerifyExecution,
    emit_k: usize,
    tokens_remaining: usize,
) -> usize {
    mtp_verified_runway_extra_depth(
        enabled,
        verify_execution,
        emit_k,
        tokens_remaining,
        Some(1),
        1,
    )
}

fn mtp_verified_runway_extra_depth(
    enabled: bool,
    verify_execution: MtpVerifyExecution,
    emit_k: usize,
    tokens_remaining: usize,
    max_extra_override: Option<usize>,
    confidence: usize,
) -> usize {
    if enabled
        && verify_execution == MtpVerifyExecution::DeviceResident
        && tokens_remaining > emit_k
        && confidence > 0
    {
        let room = tokens_remaining - emit_k;
        let adaptive_extra = emit_k.saturating_mul(2).max(2);
        room.min(max_extra_override.unwrap_or(adaptive_extra))
            .min(confidence)
    } else {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpRunwayAfterEmit {
    Continue,
    Resume { next_token: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MtpVerifiedRunway {
    tokens: Vec<u32>,
    cursor: usize,
    next_token: u32,
}

impl MtpVerifiedRunway {
    fn new(tokens: Vec<u32>, next_token: u32) -> Option<Self> {
        if tokens.is_empty() {
            return None;
        }
        Some(Self {
            tokens,
            cursor: 0,
            next_token,
        })
    }

    fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.cursor)
    }

    fn pop_emit(&mut self) -> Option<(u32, MtpRunwayAfterEmit)> {
        let token = *self.tokens.get(self.cursor)?;
        self.cursor += 1;
        let after_emit = if self.cursor < self.tokens.len() {
            MtpRunwayAfterEmit::Continue
        } else {
            MtpRunwayAfterEmit::Resume {
                next_token: self.next_token,
            }
        };
        Some((token, after_emit))
    }
}

fn mtp_hidden_prefix_rows(
    hidden_rows: &[f32],
    hidden_dim: usize,
    rows: usize,
) -> crate::error::Result<&[f32]> {
    let len = rows.checked_mul(hidden_dim).ok_or_else(|| {
        crate::error::LlmError::Forward("MTP hidden prefix row length overflow".to_string())
    })?;
    hidden_rows.get(..len).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "MTP hidden prefix rows mismatch: got {}, need {}",
            hidden_rows.len(),
            len
        ))
    })
}

fn draft_only_should_emit(token: u32, eos: u32, stop_tokens: &[u32]) -> bool {
    token != eos && !stop_tokens.contains(&token)
}

fn next_token_from_current_logits(
    engine: &Engine,
    logits: &mut [f32],
    sampler: &mut SamplerChain,
    generated_tokens: &[u32],
    rng: &mut SmallRng,
) -> crate::error::Result<u32> {
    if let Some(token) = engine.last_backend_argmax_token() {
        return Ok(token);
    }
    if logits.is_empty() {
        return Err(crate::error::LlmError::Forward(
            "MTP generate missing target logits".to_string(),
        ));
    }
    Ok(sampler.sample(logits, generated_tokens, rng))
}

pub(crate) fn generate_stream_mtp(
    engine: &mut Engine,
    prompt: &str,
    params: &GenerateParams,
    mut callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    if !engine.mtp_runtime_ready() {
        return Err(crate::error::LlmError::Forward(
            "RNB_MTP=1 but loaded model does not expose an MTP runtime".to_string(),
        ));
    }
    if !mtp_greedy_verify_allowed(params) {
        return Err(crate::error::LlmError::Forward(
            "MTP generate currently requires greedy sampling".to_string(),
        ));
    }

    // mc78 Task 12: External drafter (Gemma 4 assistant) 분기.
    // InModel (nextn) 경로는 아래 기존 코드가 그대로 처리한다.
    if engine.mtp_is_external_runtime() {
        return generate_with_external_drafter(engine, prompt, params, callback);
    }

    let start = Instant::now();
    let bos = engine.tokenizer.vocab.special.bos;
    let eos = engine.tokenizer.vocab.special.eos;
    let mut prompt_tokens = Vec::new();
    if engine.tokenizer.should_add_bos() {
        prompt_tokens.push(bos);
    }
    prompt_tokens.extend(engine.tokenizer.encode(prompt));
    let prompt_len = prompt_tokens.len();

    let mut rng = match params.seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    };
    let mut sampler = SamplerChain::from_params(params);

    let prompt_prefill_start = Instant::now();
    engine.clear_sequence_state()?;
    let mut logits = engine.forward(&prompt_tokens)?;
    let prompt_prefill_ms = elapsed_ms(prompt_prefill_start);
    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let mut tokens_remaining = params.max_tokens;
    let mut current_token = next_token_from_current_logits(
        engine,
        &mut logits,
        &mut sampler,
        &generated_tokens,
        &mut rng,
    )?;
    if !crate::runtime::mtp_draft_only_enabled() && crate::runtime::mtp_device_verify_enabled() {
        engine.prewarm_mtp_device_verify_static_weights()?;
    }
    let decode_loop_start = Instant::now();

    if crate::runtime::mtp_draft_only_enabled() {
        let mut stats = MtpStats {
            rounds: 0,
            drafted: 0,
            accepted: 0,
            carried: 0,
            target_verify_steps: 0,
            target_verify_invocations: 0,
        };
        let mut phase = MtpPhaseTimings::default();

        while tokens_remaining > 0 {
            crate::generate::check_generation_cancellation()?;
            if !draft_only_should_emit(current_token, eos, &params.stop_tokens) {
                break;
            }

            generated_tokens.push(current_token);
            stats.accepted += 1;
            let piece = engine.tokenizer.decode_token(current_token).to_string();
            generated_text.push_str(&piece);
            if !callback(&piece) {
                break;
            }
            tokens_remaining -= 1;
            if tokens_remaining == 0 {
                break;
            }

            let k = params.spec_k.max(1).min(tokens_remaining);
            stats.rounds += 1;
            stats.drafted += k;
            let phase_start = Instant::now();
            let draft_tokens = engine.mtp_draft_tokens(current_token, k)?;
            phase.draft_ms += elapsed_ms(phase_start);
            if draft_tokens.is_empty() {
                break;
            }

            for token in draft_tokens {
                current_token = token;
                if !draft_only_should_emit(current_token, eos, &params.stop_tokens) {
                    tokens_remaining = 0;
                    break;
                }
                generated_tokens.push(current_token);
                stats.accepted += 1;
                let piece = engine.tokenizer.decode_token(current_token).to_string();
                generated_text.push_str(&piece);
                if !callback(&piece) {
                    tokens_remaining = 0;
                    break;
                }
                tokens_remaining -= 1;
                if tokens_remaining == 0 {
                    break;
                }
            }
        }

        if crate::runtime::profiling_enabled() || crate::runtime::spec_profile_enabled() {
            eprintln!("[MTP] draft-only mode: target verification disabled");
            eprintln!("{}", stats.report());
            eprintln!("{}", phase.report(stats.rounds));
            eprintln!(
                "  [MTP] wall_split prompt_prefill={:.1}ms decode_loop={:.1}ms",
                prompt_prefill_ms,
                elapsed_ms(decode_loop_start)
            );
        }

        let elapsed = start.elapsed().as_secs_f32();
        return Ok(GenerateResult::new(
            generated_text,
            generated_tokens.len(),
            prompt_len,
            elapsed,
            generated_tokens,
        ));
    }

    let mut stats = MtpStats {
        rounds: 0,
        drafted: 0,
        accepted: 0,
        carried: 0,
        target_verify_steps: 0,
        target_verify_invocations: 0,
    };
    let mut phase = MtpPhaseTimings::default();
    let trace_mtp = crate::runtime::mtp_trace_enabled();
    let target_needs_sequential = mtp_target_needs_sequential(engine.architecture());
    let sequential_multi = crate::runtime::spec_mtp_sequential_multi_enabled();
    let batch_verify = mtp_batch_verify_requested(
        crate::runtime::mtp_batch_verify_enabled(),
        crate::runtime::mtp_batch_verify_disabled(),
        mtp_batch_verify_default_enabled(target_needs_sequential),
    );
    let verify_execution =
        mtp_verify_execution(crate::runtime::mtp_device_verify_enabled(), batch_verify);
    let fast_retain = crate::runtime::mtp_fast_retain_enabled();
    let no_bonus_verify_override = mtp_no_bonus_verify_env_override();
    let verified_runway_enabled = crate::runtime::mtp_shadow_precompute_enabled()
        && verify_execution == MtpVerifyExecution::DeviceResident;
    let verified_runway_max_extra = crate::runtime::mtp_runway_max_extra();
    let mut verified_runway: Option<MtpVerifiedRunway> = None;
    let mut verified_runway_confidence = 0usize;

    while tokens_remaining > 0 {
        crate::generate::check_generation_cancellation()?;
        let mut clear_runway = false;
        let runway_after_emit = verified_runway.as_mut().and_then(|runway| {
            let (token, after_emit) = runway.pop_emit()?;
            current_token = token;
            if matches!(after_emit, MtpRunwayAfterEmit::Resume { .. }) {
                clear_runway = true;
            }
            Some(after_emit)
        });
        if clear_runway {
            verified_runway = None;
        }

        if params.stop_tokens.contains(&current_token) || current_token == eos {
            break;
        }

        generated_tokens.push(current_token);
        let piece = engine.tokenizer.decode_token(current_token).to_string();
        generated_text.push_str(&piece);
        if !callback(&piece) {
            break;
        }
        tokens_remaining -= 1;
        if tokens_remaining == 0 {
            break;
        }
        if let Some(after_emit) = runway_after_emit {
            match after_emit {
                MtpRunwayAfterEmit::Continue => continue,
                MtpRunwayAfterEmit::Resume { next_token } => {
                    current_token = next_token;
                    continue;
                }
            }
        }

        let k = if matches!(
            verify_execution,
            MtpVerifyExecution::BatchPrefill | MtpVerifyExecution::DeviceResident
        ) {
            mtp_batch_effective_k(params.spec_k, tokens_remaining)
        } else {
            mtp_effective_k(
                params.spec_k,
                tokens_remaining,
                target_needs_sequential,
                sequential_multi,
            )
        };
        let runway_extra_depth = mtp_verified_runway_extra_depth(
            verified_runway_enabled,
            verify_execution,
            k,
            tokens_remaining,
            verified_runway_max_extra,
            verified_runway_confidence,
        );
        let draft_k = k + runway_extra_depth;
        stats.rounds += 1;
        stats.drafted += draft_k;

        let round_base_kv_len = engine.kv_cache.current_len();
        let phase_start = Instant::now();
        let full_checkpoint =
            if mtp_full_engine_checkpoint_required(verify_execution, target_needs_sequential) {
                Some(SpecCheckpoint::save_engine(engine)?)
            } else {
                None
            };
        let mtp_checkpoint = engine.mtp_checkpoint();
        phase.checkpoint_ms += elapsed_ms(phase_start);

        let phase_start = Instant::now();
        let draft_tokens = engine.mtp_draft_tokens(current_token, draft_k)?;
        phase.draft_ms += elapsed_ms(phase_start);
        if draft_tokens.len() != draft_k {
            return Err(crate::error::LlmError::Forward(format!(
                "MTP draft produced {} tokens, expected {}",
                draft_tokens.len(),
                draft_k
            )));
        }

        if matches!(
            verify_execution,
            MtpVerifyExecution::BatchPrefill | MtpVerifyExecution::DeviceResident
        ) {
            let no_bonus_verify = runway_extra_depth == 0
                && mtp_no_bonus_verify_for_round(
                    verify_execution,
                    no_bonus_verify_override,
                    draft_k,
                );
            let verify_request = MtpVerifyWindowRequest::new(
                current_token,
                &draft_tokens,
                if no_bonus_verify {
                    MtpVerifyBonus::Omit
                } else {
                    MtpVerifyBonus::Include
                },
            );
            let verify_input = verify_request.verify_tokens();
            let prefix_tokens = mtp_prefix_snapshot_tokens(
                verify_execution,
                target_needs_sequential,
                &verify_request,
            );
            let phase_start = Instant::now();
            let window = match verify_execution {
                MtpVerifyExecution::DeviceResident => {
                    engine.forward_mtp_device_verify_window_argmax_collect_mtp(&verify_request)?
                }
                MtpVerifyExecution::BatchPrefill => engine
                    .forward_prefill_argmax_tokens_collect_mtp_prefix_states_deferred_observe(
                        &verify_input,
                        &prefix_tokens,
                    )?,
                MtpVerifyExecution::Sequential => unreachable!("sequential verify handled below"),
            };
            stats.add_target_verify(window.len(), 1);

            let mut n_accepted = 0usize;
            let mut stopped = false;
            let mut next_forced_token = None;
            let mut target_trace = trace_mtp.then(Vec::new);

            for i in 0..draft_k {
                let target_token = *window.target_tokens.get(i).ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "MTP batch verify missing target token at {i}"
                    ))
                })?;
                if let Some(trace) = target_trace.as_mut() {
                    let draft_token = draft_tokens[i];
                    trace.push(format!(
                        "{}:{}{}{}",
                        i,
                        engine.tokenizer.decode_token(target_token),
                        if target_token == draft_tokens[i] {
                            "=draft:"
                        } else {
                            "!=draft:"
                        },
                        engine.tokenizer.decode_token(draft_token),
                    ));
                }

                if target_token != draft_tokens[i] {
                    next_forced_token = Some(target_token);
                    break;
                }

                if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                    stopped = true;
                    break;
                }

                n_accepted += 1;
                stats.accepted += 1;
                if i < k {
                    generated_tokens.push(draft_tokens[i]);
                    let piece = engine.tokenizer.decode_token(draft_tokens[i]).to_string();
                    generated_text.push_str(&piece);
                    if !callback(&piece) {
                        stopped = true;
                        tokens_remaining -= 1;
                        break;
                    }

                    tokens_remaining -= 1;
                    if tokens_remaining == 0 {
                        break;
                    }
                }
            }
            phase.verify_ms += elapsed_ms(phase_start);

            if let Some(trace) = target_trace {
                eprintln!(
                    "[MTP_TRACE] round={} accepted={} targets=[{}]",
                    stats.rounds,
                    n_accepted,
                    trace.join(", ")
                );
            }

            if stopped || tokens_remaining == 0 {
                break;
            }

            if n_accepted == draft_k {
                if no_bonus_verify {
                    let (next_token, final_hidden_row) = engine
                        .forward_verify_argmax_sequential_collect_mtp(draft_tokens[draft_k - 1])?;
                    stats.add_target_verify(1, 1);
                    let mut committed_verify_input = verify_input;
                    committed_verify_input.push(draft_tokens[draft_k - 1]);
                    let mut committed_hidden_rows = window.mtp_hidden_rows;
                    committed_hidden_rows.extend_from_slice(&final_hidden_row);
                    let phase_start = Instant::now();
                    if fast_retain {
                        engine.mtp_retain_draft_after_spec(
                            mtp_checkpoint.as_ref(),
                            &committed_verify_input,
                            verify_request.full_accept_committed_tokens(),
                            draft_k,
                            &committed_hidden_rows,
                        )?;
                    } else {
                        engine.mtp_restore_checkpoint(mtp_checkpoint.as_ref());
                        engine.mtp_observe_target_batch(
                            &committed_verify_input,
                            &committed_hidden_rows,
                        )?;
                    }
                    phase.verify_ms += elapsed_ms(phase_start);
                    if verified_runway_enabled {
                        verified_runway_confidence = verified_runway_confidence.max(1);
                    }
                    current_token = next_token;
                } else {
                    let phase_start = Instant::now();
                    if fast_retain {
                        engine.mtp_retain_draft_after_spec(
                            mtp_checkpoint.as_ref(),
                            &verify_input,
                            verify_request.full_accept_committed_tokens(),
                            draft_k,
                            &window.mtp_hidden_rows,
                        )?;
                    } else {
                        engine.mtp_restore_checkpoint(mtp_checkpoint.as_ref());
                        engine.mtp_observe_target_batch(&verify_input, &window.mtp_hidden_rows)?;
                    }
                    phase.verify_ms += elapsed_ms(phase_start);
                    let next_token = *window.target_tokens.get(draft_k).ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "MTP batch full-window verify missing bonus token".to_string(),
                        )
                    })?;
                    if runway_extra_depth > 0 {
                        let runway_tokens = draft_tokens[k..draft_k].to_vec();
                        if let Some(runway) = MtpVerifiedRunway::new(runway_tokens, next_token) {
                            stats.carried += runway.len();
                            verified_runway_confidence = verified_runway_confidence
                                .saturating_add(runway.len())
                                .max(1);
                            verified_runway = Some(runway);
                        }
                    } else if verified_runway_enabled {
                        verified_runway_confidence = verified_runway_confidence.max(1);
                    }
                    current_token = next_token;
                }
                continue;
            }

            let committed_tokens = 1 + n_accepted;
            let state_plan = (verify_execution == MtpVerifyExecution::BatchPrefill).then(|| {
                mtp_verify_state_plan(
                    verify_execution,
                    target_needs_sequential,
                    draft_k,
                    n_accepted,
                    no_bonus_verify,
                    window.len(),
                    window.prefix_state_for(committed_tokens).is_some(),
                )
            });
            debug_assert!(state_plan
                .as_ref()
                .is_none_or(|plan| plan.committed_verify_tokens == committed_tokens));

            let next_token = if state_plan
                .as_ref()
                .is_some_and(|plan| plan.replay_committed_prefix)
            {
                let checkpoint = full_checkpoint.as_ref().ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "MTP batch replay requires full engine checkpoint".to_string(),
                    )
                })?;
                let phase_start = Instant::now();
                let next_token = replay_committed_verify_prefix_argmax(
                    engine,
                    checkpoint,
                    &verify_input,
                    committed_tokens,
                )?;
                stats.add_target_replay_invocations(committed_tokens);
                phase.retain_ms += elapsed_ms(phase_start);
                next_token
            } else {
                let phase_start = Instant::now();
                if mtp_prefix_state_restore_required(
                    state_plan.as_ref(),
                    committed_tokens,
                    window.len(),
                ) {
                    let restore_base_kv_len = mtp_prefix_restore_base_kv_len(
                        verify_execution,
                        full_checkpoint.as_ref().map(|checkpoint| checkpoint.kv_len),
                        round_base_kv_len,
                    )?;
                    let prefix_state =
                        window.prefix_state_for(committed_tokens).ok_or_else(|| {
                            crate::error::LlmError::Forward(
                                "MTP batch verify missing prefix state for rejected window"
                                    .to_string(),
                            )
                        })?;
                    if verify_execution == MtpVerifyExecution::BatchPrefill {
                        full_checkpoint
                            .as_ref()
                            .ok_or_else(|| {
                                crate::error::LlmError::Forward(
                                    "MTP batch prefix restore requires full engine checkpoint"
                                        .to_string(),
                                )
                            })?
                            .restore_engine(engine)?;
                    }
                    engine.restore_verify_window_prefix_state(restore_base_kv_len, prefix_state)?;
                }

                let committed_hidden = window.mtp_hidden_prefix_rows(committed_tokens)?;
                engine.mtp_retain_draft_after_spec(
                    mtp_checkpoint.as_ref(),
                    &verify_input,
                    committed_tokens,
                    draft_k,
                    committed_hidden,
                )?;
                phase.retain_ms += elapsed_ms(phase_start);

                next_forced_token.ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "MTP batch verify did not produce next token".to_string(),
                    )
                })?
            };
            if runway_extra_depth > 0 && n_accepted > k {
                let runway_tokens = draft_tokens[k..n_accepted].to_vec();
                if let Some(runway) = MtpVerifiedRunway::new(runway_tokens, next_token) {
                    stats.carried += runway.len();
                    verified_runway_confidence = verified_runway_confidence
                        .saturating_add(runway.len())
                        .max(1);
                    verified_runway = Some(runway);
                }
            } else if runway_extra_depth > 0 {
                verified_runway_confidence = 0;
            } else if n_accepted < k {
                verified_runway_confidence = 0;
            }
            current_token = next_token;
            continue;
        }

        let mut verify_input = Vec::with_capacity(k + 1);
        verify_input.push(current_token);
        verify_input.extend_from_slice(&draft_tokens);

        let mut n_accepted = 0usize;
        let mut stopped = false;
        let mut next_forced_token = None;
        let mut observed_hidden_rows = Vec::new();
        let mut target_trace = trace_mtp.then(Vec::new);

        let phase_start = Instant::now();
        for i in 0..k {
            let (target_token, hidden_rows) =
                engine.forward_verify_argmax_sequential_collect_mtp(verify_input[i])?;
            stats.add_target_verify(1, 1);
            observed_hidden_rows.extend_from_slice(&hidden_rows);

            if let Some(trace) = target_trace.as_mut() {
                let draft_token = draft_tokens[i];
                trace.push(format!(
                    "{}:{}{}{}",
                    i,
                    engine.tokenizer.decode_token(target_token),
                    if target_token == draft_tokens[i] {
                        "=draft:"
                    } else {
                        "!=draft:"
                    },
                    engine.tokenizer.decode_token(draft_token),
                ));
            }

            if target_token != draft_tokens[i] {
                next_forced_token = Some(target_token);
                break;
            }

            if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                stopped = true;
                break;
            }

            generated_tokens.push(draft_tokens[i]);
            let piece = engine.tokenizer.decode_token(draft_tokens[i]).to_string();
            generated_text.push_str(&piece);
            if !callback(&piece) {
                stopped = true;
                n_accepted += 1;
                stats.accepted += 1;
                tokens_remaining -= 1;
                break;
            }

            n_accepted += 1;
            stats.accepted += 1;
            tokens_remaining -= 1;
            if tokens_remaining == 0 {
                break;
            }
        }

        if !stopped && tokens_remaining > 0 && n_accepted == k {
            let (target_token, hidden_rows) =
                engine.forward_verify_argmax_sequential_collect_mtp(verify_input[k])?;
            stats.add_target_verify(1, 1);
            observed_hidden_rows.extend_from_slice(&hidden_rows);
            next_forced_token = Some(target_token);
        }
        phase.verify_ms += elapsed_ms(phase_start);

        if let Some(trace) = target_trace {
            eprintln!(
                "[MTP_TRACE] round={} accepted={} targets=[{}]",
                stats.rounds,
                n_accepted,
                trace.join(", ")
            );
        }

        if stopped || tokens_remaining == 0 {
            break;
        }

        let committed_tokens = 1 + n_accepted;
        let committed_hidden = mtp_hidden_prefix_rows(
            &observed_hidden_rows,
            engine.metadata.hidden_dim,
            committed_tokens,
        )?;
        let phase_start = Instant::now();
        if fast_retain {
            engine.mtp_retain_draft_after_spec(
                mtp_checkpoint.as_ref(),
                &verify_input,
                committed_tokens,
                k,
                committed_hidden,
            )?;
        } else {
            engine.mtp_restore_checkpoint(mtp_checkpoint.as_ref());
            engine.mtp_observe_target_batch(&verify_input[..committed_tokens], committed_hidden)?;
        }
        phase.retain_ms += elapsed_ms(phase_start);

        current_token = next_forced_token.ok_or_else(|| {
            crate::error::LlmError::Forward("MTP verify did not produce next token".to_string())
        })?;
    }

    if crate::runtime::profiling_enabled() || crate::runtime::spec_profile_enabled() {
        eprintln!("{}", stats.report());
        eprintln!("{}", phase.report(stats.rounds));
        eprintln!(
            "  [MTP] wall_split prompt_prefill={:.1}ms decode_loop={:.1}ms",
            prompt_prefill_ms,
            elapsed_ms(decode_loop_start)
        );
    }

    let elapsed = start.elapsed().as_secs_f32();
    Ok(GenerateResult::new(
        generated_text,
        generated_tokens.len(),
        prompt_len,
        elapsed,
        generated_tokens,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// mc78 Task 12 — External drafter draft → verify → commit loop
// ─────────────────────────────────────────────────────────────────────────────

/// External drafter (Gemma 4 assistant 모델) 를 이용한 speculative generate.
///
/// 알고리즘:
/// 1. Prefill prompt → 첫 token 은 target 단독 greedy 로 결정.
/// 2. 매 round: drafter 가 N 개 draft 생성 → target batch verify → commit_kv_through.
/// 3. accept 된 draft 토큰 + target correction token 1 개 emit.
/// 4. EOS/stop token/max_tokens 도달 시 종료.
///
/// Greedy only: `mtp_greedy_verify_allowed` 가 이미 외부에서 체크됐다.
fn generate_with_external_drafter(
    engine: &mut Engine,
    prompt: &str,
    params: &GenerateParams,
    mut callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let draft_n = params.spec_k.max(1);

    let start = Instant::now();
    let bos = engine.tokenizer.vocab.special.bos;
    let eos = engine.tokenizer.vocab.special.eos;

    // ── 1. 프롬프트 토크나이즈 ──────────────────────────────────────────────
    let mut prompt_tokens = Vec::new();
    if engine.tokenizer.should_add_bos() {
        prompt_tokens.push(bos);
    }
    prompt_tokens.extend(engine.tokenizer.encode(prompt));
    let prompt_len = prompt_tokens.len();

    // ── 2. Prefill ──────────────────────────────────────────────────────────
    let prompt_prefill_start = Instant::now();
    engine.clear_sequence_state()?;
    let mut logits = engine.forward(&prompt_tokens)?;
    let prompt_prefill_ms = elapsed_ms(prompt_prefill_start);

    // ── 3. 첫 token: target 단독 greedy (drafter 루프 진입 전) ─────────────
    // dummy SamplerChain (temperature=0 greedy) 으로 argmax 뽑는다.
    // `next_token_from_current_logits` 는 backend_argmax_token 이 있으면 sampler 무시.
    let mut dummy_rng = rand::rngs::SmallRng::from_entropy();
    let mut dummy_sampler = SamplerChain::from_params(params);
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut generated_text = String::new();
    let mut tokens_remaining = params.max_tokens;

    let first_token = next_token_from_current_logits(
        engine,
        &mut logits,
        &mut dummy_sampler,
        &generated_tokens,
        &mut dummy_rng,
    )?;

    // EOS 즉시 체크
    if params.stop_tokens.contains(&first_token) || first_token == eos {
        let elapsed = start.elapsed().as_secs_f32();
        return Ok(GenerateResult::new(
            generated_text,
            0,
            prompt_len,
            elapsed,
            generated_tokens,
        ));
    }

    generated_tokens.push(first_token);
    let first_piece = engine.tokenizer.decode_token(first_token).to_string();
    generated_text.push_str(&first_piece);
    if !callback(&first_piece) {
        let elapsed = start.elapsed().as_secs_f32();
        return Ok(GenerateResult::new(
            generated_text,
            generated_tokens.len(),
            prompt_len,
            elapsed,
            generated_tokens,
        ));
    }
    tokens_remaining -= 1;

    let mut current_token = first_token;

    // ── 4. Stats ────────────────────────────────────────────────────────────
    let mut stats = MtpStats {
        rounds: 0,
        drafted: 0,
        accepted: 0,
        carried: 0,
        target_verify_steps: 0,
        target_verify_invocations: 0,
    };
    let decode_loop_start = Instant::now();

    // mc78 timing diag
    let mut t_setup = 0.0f64;
    let mut t_drafter = 0.0f64;
    let mut t_verify = 0.0f64;
    let t_commit_emit = 0.0f64;
    let timing_enabled = std::env::var("RNB_MC78_TIMING").is_ok();
    let target_needs_sequential = mtp_target_needs_sequential(engine.architecture());
    let external_batch_verify = mtp_external_batch_verify_allowed(
        engine.architecture(),
        crate::runtime::mtp_batch_verify_enabled(),
        crate::runtime::mtp_batch_verify_disabled(),
        mtp_batch_verify_default_enabled(target_needs_sequential),
    );
    if crate::runtime::mtp_batch_verify_enabled() && target_needs_sequential {
        eprintln!(
            "[MTP/ext] batch verify disabled for {:?}; using sequential target verify",
            engine.architecture()
        );
    }

    // ── 5. Draft → Verify → Commit 루프 ────────────────────────────────────
    while tokens_remaining > 0 {
        crate::generate::check_generation_cancellation()?;
        // 5a. target 의 현재 hidden state + KV position 을 owned 값으로 복사.
        //     &self borrow 를 여기서 끝내야 이후 &mut borrow 가 가능하다.
        let t0 = std::time::Instant::now();
        let target_last_hidden: Vec<f32> = engine.last_hidden_for_decode().to_vec();
        let shared_kv = engine.shared_kv_view(); // owned SharedKvStates
        let position_before_verify = engine.kv_cache.current_len() as u32;
        if timing_enabled {
            t_setup += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 5b. mc78 wiring correctness fix — Stepper 를 bypass 하고 drafter_forward 직접 호출.
        //
        // 올바른 inputs_embeds 구성 (reference: drafter_backbone_calibrate_test.rs:214-295):
        //   첫 절반 = target.token_embd_row(prev_tok) * sqrt(backbone_hidden)
        //   둘째 절반 = target_last_hidden (매 draft step 고정)
        //   position_id = last_validated_position + step  (0-based step)
        //
        // 기존 ExternalDrafterStepper::draft_n 은 [last_target_hidden ; last_drafter_hidden]
        // 구성으로 spec drift 가 있었음. Stepper 구조체 자체는 trait completeness 로 keep.
        let t_d0 = std::time::Instant::now();
        let drafts: Vec<u32> = {
            let drafter_arc = engine
                .mtp_external_runtime_mut()
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "generate_with_external_drafter: External runtime vanished".to_string(),
                    )
                })?
                .drafter()
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "generate_with_external_drafter: Drafter not loaded".to_string(),
                    )
                })?
                .clone();

            let backbone = drafter_arc.backbone_hidden;
            let embed_scale = (backbone as f32).sqrt();
            // last_validated_position 은 multi-step draft 동안 constant
            // (backbone.rs:75 의 spec §7). drafter 가 sliding/full attention
            // 의 cached KV 를 reuse 하므로 position 갱신 불필요.
            let last_validated_position = position_before_verify.saturating_sub(1);

            let mut inputs = vec![0.0f32; 2 * backbone];
            // step 0 의 last_hidden = target 의 prefill 후 last_hidden.
            // step k>=1 의 last_hidden = drafter 의 projected_hidden_{k-1}.
            // backbone.rs:56-58 의 spec §7 Eagle pattern.
            let mut current_last_hidden: Vec<f32> = target_last_hidden.clone();

            let mut result = Vec::with_capacity(draft_n);
            let mut prev_tok = current_token;

            for _step in 0..draft_n {
                let mut prev_embd = engine.token_embd_row(prev_tok);
                debug_assert_eq!(prev_embd.len(), backbone);
                for v in prev_embd.iter_mut() {
                    *v *= embed_scale;
                }
                inputs[..backbone].copy_from_slice(&prev_embd);
                inputs[backbone..].copy_from_slice(&current_last_hidden);

                let out = rnb_mtp::drafter::drafter_forward(
                    &drafter_arc,
                    &inputs,
                    &shared_kv,
                    last_validated_position,
                );
                let tok = crate::external_drafter::argmax(&out.logits);
                result.push(tok);
                prev_tok = tok;
                current_last_hidden = out.projected_hidden; // Eagle: 다음 step input
            }
            result
        };
        if timing_enabled {
            t_drafter += t_d0.elapsed().as_secs_f64() * 1000.0;
        }

        let effective_n = drafts.len().min(tokens_remaining);
        if effective_n == 0 {
            break;
        }
        let drafts_used = &drafts[..effective_n];
        stats.rounds += 1;
        stats.drafted += effective_n;

        // 5c. verify_seq = [current_token, t1, ..., tN].
        let mut verify_seq = Vec::with_capacity(effective_n + 1);
        verify_seq.push(current_token);
        verify_seq.extend_from_slice(drafts_used);

        let t_v0 = std::time::Instant::now();
        let mut stopped = false;
        let mut accepted_draft_tokens = 0usize;
        let mut target_token = None;

        if external_batch_verify {
            let target_preds = engine.forward_batch_verify(&verify_seq, position_before_verify)?;
            if timing_enabled {
                t_verify += t_v0.elapsed().as_secs_f64() * 1000.0;
            }
            // target_preds.len() == verify_seq.len() == effective_n + 1
            // target_preds[i] = 모델이 verify_seq[i] 를 소비한 뒤 예측한 다음 토큰
            let outcome =
                rnb_mtp::verify::verify_greedy(drafts_used, &target_preds[..effective_n + 1])
                    .map_err(|e| {
                        crate::error::LlmError::Forward(format!(
                            "external drafter verify_greedy: {e:?}"
                        ))
                    })?;

            let new_kv_position = position_before_verify + outcome.accepted_draft_tokens as u32 + 1;
            engine.commit_kv_through(new_kv_position);

            accepted_draft_tokens = outcome.accepted_draft_tokens;
            stats.accepted += outcome.accepted_draft_tokens;
            stats.add_target_verify(1, 1);
            target_token = Some(outcome.target_token);
        } else {
            for i in 0..effective_n {
                let (target_tok, _) =
                    engine.forward_verify_argmax_sequential_collect_mtp(verify_seq[i])?;
                stats.add_target_verify(1, 1);
                if target_tok != drafts_used[i] {
                    target_token = Some(target_tok);
                    break;
                }
                accepted_draft_tokens += 1;
                stats.accepted += 1;
                let tok = drafts_used[i];
                if params.stop_tokens.contains(&tok) || tok == eos {
                    stopped = true;
                    break;
                }
                generated_tokens.push(tok);
                let piece = engine.tokenizer.decode_token(tok).to_string();
                generated_text.push_str(&piece);
                if !callback(&piece) {
                    stopped = true;
                    tokens_remaining -= 1;
                    break;
                }
                tokens_remaining -= 1;
                if tokens_remaining == 0 {
                    stopped = true;
                    break;
                }
            }

            if !stopped && tokens_remaining > 0 && accepted_draft_tokens == effective_n {
                let (target_tok, _) =
                    engine.forward_verify_argmax_sequential_collect_mtp(verify_seq[effective_n])?;
                stats.add_target_verify(1, 1);
                target_token = Some(target_tok);
            }
            if timing_enabled {
                t_verify += t_v0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        if external_batch_verify {
            // 5f. accept 된 draft 토큰 emit.
            for i in 0..accepted_draft_tokens {
                let tok = drafts_used[i];
                if params.stop_tokens.contains(&tok) || tok == eos {
                    stopped = true;
                    break;
                }
                generated_tokens.push(tok);
                let piece = engine.tokenizer.decode_token(tok).to_string();
                generated_text.push_str(&piece);
                if !callback(&piece) {
                    stopped = true;
                    tokens_remaining -= 1;
                    break;
                }
                tokens_remaining -= 1;
                if tokens_remaining == 0 {
                    stopped = true;
                    break;
                }
            }
        }

        // 5g. target correction token emit (accept 가 끝났고 아직 여유 있을 때).
        if !stopped && tokens_remaining > 0 {
            let target_tok = target_token.ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "external drafter verify did not produce target token".to_string(),
                )
            })?;
            if params.stop_tokens.contains(&target_tok) || target_tok == eos {
                stopped = true;
            } else {
                generated_tokens.push(target_tok);
                let piece = engine.tokenizer.decode_token(target_tok).to_string();
                generated_text.push_str(&piece);
                if !callback(&piece) {
                    stopped = true;
                }
                tokens_remaining -= 1;
                current_token = target_tok;
            }
        }

        if stopped || tokens_remaining == 0 {
            break;
        }

        // 5h. drafter state 전진 (accept 수만큼 position shift).
        if let Some(runtime) = engine.mtp_external_runtime_mut() {
            runtime.shift_for_accept(accepted_draft_tokens + 1);
        }
    }

    if crate::runtime::profiling_enabled() || crate::runtime::spec_profile_enabled() {
        eprintln!("[MTP/ext] {}", stats.report());
        eprintln!(
            "  [MTP/ext] wall_split prompt_prefill={:.1}ms decode_loop={:.1}ms",
            prompt_prefill_ms,
            elapsed_ms(decode_loop_start),
        );
    }
    if timing_enabled {
        eprintln!(
            "[MTP/ext-timing] rounds={} setup={:.1}ms drafter={:.1}ms verify={:.1}ms commit_emit={:.1}ms",
            stats.rounds, t_setup, t_drafter, t_verify, t_commit_emit
        );
    }

    let elapsed = start.elapsed().as_secs_f32();
    Ok(GenerateResult::new(
        generated_text,
        generated_tokens.len(),
        prompt_len,
        elapsed,
        generated_tokens,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_generation_requires_greedy_sampling() {
        let mut params = GenerateParams {
            temperature: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat: None,
            ..GenerateParams::default()
        };

        assert!(mtp_greedy_verify_allowed(&params));

        params.repetition_penalty = 1.1;
        assert!(!mtp_greedy_verify_allowed(&params));
    }

    #[test]
    fn mtp_stats_report_includes_verify_steps_and_acceptance() {
        let stats = MtpStats {
            rounds: 3,
            drafted: 12,
            accepted: 7,
            carried: 2,
            target_verify_steps: 15,
            target_verify_invocations: 15,
        };

        let report = stats.report();

        assert!(report.contains("rounds=3"));
        assert!(report.contains("drafted=12"));
        assert!(report.contains("accepted=7"));
        assert!(report.contains("carried=2"));
        assert!(report.contains("target_verify_steps=15"));
        assert!(report.contains("target_verify_invocations=15"));
    }

    #[test]
    fn cpu_batch_prefill_first_draft_reject_uses_prefix_snapshot_without_replay() {
        assert!(mtp_full_engine_checkpoint_required(
            MtpVerifyExecution::BatchPrefill,
            true,
        ));
        assert!(!mtp_full_engine_checkpoint_required(
            MtpVerifyExecution::Sequential,
            true,
        ));
        assert!(!mtp_full_engine_checkpoint_required(
            MtpVerifyExecution::DeviceResident,
            true,
        ));

        let draft_k = 2;
        let plan = mtp_verify_state_plan(
            MtpVerifyExecution::BatchPrefill,
            true,
            draft_k,
            0,
            true,
            draft_k,
            true,
        );
        let mut stats = MtpStats {
            rounds: 1,
            drafted: draft_k,
            accepted: 0,
            carried: 0,
            target_verify_steps: 0,
            target_verify_invocations: 0,
        };
        stats.add_target_verify(draft_k, 1);
        if plan.replay_committed_prefix {
            stats.add_target_replay_invocations(plan.committed_verify_tokens);
        }

        assert_eq!(plan.committed_verify_tokens, 1);
        assert!(!plan.replay_committed_prefix);
        assert!(mtp_prefix_state_restore_required(
            Some(&plan),
            plan.committed_verify_tokens,
            draft_k,
        ));
        assert_eq!(stats.target_verify_invocations, 1);
    }

    #[test]
    fn cpu_batch_prefill_prefix_gt_one_without_snapshot_keeps_replay() {
        let draft_k = 3;
        let plan = mtp_verify_state_plan(
            MtpVerifyExecution::BatchPrefill,
            true,
            draft_k,
            1,
            true,
            draft_k,
            false,
        );

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(plan.replay_committed_prefix);
        assert!(!mtp_prefix_state_restore_required(
            Some(&plan),
            plan.committed_verify_tokens,
            draft_k,
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cpu_batch_prefill_k_gt_one_collects_only_restorable_prefix_snapshot() {
        let request = MtpVerifyWindowRequest::new(7, &[11, 13, 17], MtpVerifyBonus::Omit);

        assert_eq!(
            mtp_prefix_snapshot_tokens(MtpVerifyExecution::BatchPrefill, true, &request),
            vec![1],
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_prefill_k_gt_one_preserves_resident_prefix_snapshots() {
        let request = MtpVerifyWindowRequest::new(7, &[11, 13, 17], MtpVerifyBonus::Omit);

        assert_eq!(
            mtp_prefix_snapshot_tokens(MtpVerifyExecution::BatchPrefill, true, &request),
            vec![1, 2],
        );
    }

    #[test]
    fn device_resident_k_gt_one_preserves_resident_prefix_snapshots() {
        let request = MtpVerifyWindowRequest::new(7, &[11, 13, 17], MtpVerifyBonus::Omit);

        assert_eq!(
            mtp_prefix_snapshot_tokens(MtpVerifyExecution::DeviceResident, true, &request),
            vec![1, 2],
        );
    }

    #[test]
    fn device_resident_partial_reject_uses_round_base_without_full_checkpoint() {
        let committed_tokens = 1;
        let verify_window_tokens = 3;

        assert!(mtp_prefix_state_restore_required(
            None,
            committed_tokens,
            verify_window_tokens,
        ));
        assert_eq!(
            mtp_prefix_restore_base_kv_len(MtpVerifyExecution::DeviceResident, None, 41,).unwrap(),
            41,
        );
    }

    #[test]
    fn cpu_batch_prefill_no_bonus_last_draft_reject_skips_replay_invocations() {
        let draft_k = 3;
        let verify_window_tokens = draft_k;
        let plan = mtp_verify_state_plan(
            MtpVerifyExecution::BatchPrefill,
            true,
            draft_k,
            draft_k - 1,
            true,
            verify_window_tokens,
            false,
        );
        let mut stats = MtpStats {
            rounds: 1,
            drafted: draft_k,
            accepted: draft_k - 1,
            carried: 0,
            target_verify_steps: 0,
            target_verify_invocations: 0,
        };
        stats.add_target_verify(verify_window_tokens, 1);
        if plan.replay_committed_prefix {
            stats.add_target_replay_invocations(plan.committed_verify_tokens);
        }

        assert_eq!(plan.committed_verify_tokens, verify_window_tokens);
        assert!(!plan.replay_committed_prefix);
        assert!(!mtp_prefix_state_restore_required(
            Some(&plan),
            plan.committed_verify_tokens,
            verify_window_tokens,
        ));
        assert_eq!(stats.target_verify_invocations, 1);
        assert_ne!(
            stats.target_verify_invocations,
            1 + plan.committed_verify_tokens,
        );
    }

    #[test]
    fn cpu_batch_prefill_bonus_reject_uses_prefix_snapshot_without_replay() {
        let draft_k = 1;
        let verify_window_tokens = draft_k + 1;
        let plan = mtp_verify_state_plan(
            MtpVerifyExecution::BatchPrefill,
            true,
            draft_k,
            0,
            false,
            verify_window_tokens,
            true,
        );

        assert_eq!(plan.committed_verify_tokens, 1);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn batch_prefill_full_accept_keeps_post_verify_state_without_replay() {
        let plan = mtp_verify_state_plan(
            MtpVerifyExecution::BatchPrefill,
            true,
            1,
            1,
            false,
            2,
            false,
        );

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(!plan.replay_committed_prefix);
        assert!(!mtp_prefix_state_restore_required(Some(&plan), 2, 2));
    }

    #[test]
    fn mtp_target_invocations_count_batch_once_plus_replay_calls() {
        let mut stats = MtpStats {
            rounds: 1,
            drafted: 1,
            accepted: 0,
            carried: 0,
            target_verify_steps: 0,
            target_verify_invocations: 0,
        };

        stats.add_target_verify(2, 1);
        stats.add_target_replay_invocations(1);

        assert_eq!(stats.target_verify_steps, 2);
        assert_eq!(stats.target_verify_invocations, 2);
        assert!(stats.report().contains("target_verify_invocations=2"));
    }

    #[test]
    fn mtp_sequential_verify_counts_each_decode_call() {
        let mut stats = MtpStats {
            rounds: 1,
            drafted: 1,
            accepted: 1,
            carried: 0,
            target_verify_steps: 0,
            target_verify_invocations: 0,
        };

        stats.add_target_verify(1, 1);
        stats.add_target_verify(1, 1);

        assert_eq!(stats.target_verify_steps, 2);
        assert_eq!(stats.target_verify_invocations, 2);
    }

    #[test]
    fn mtp_phase_report_includes_verify_ms_per_round() {
        let timings = MtpPhaseTimings {
            checkpoint_ms: 3.0,
            draft_ms: 6.0,
            verify_ms: 30.0,
            retain_ms: 0.0,
        };

        let report = timings.report(3);

        assert!(report.contains("draft=2.0ms/round"));
        assert!(report.contains("verify=10.0ms/round"));
    }

    #[test]
    fn sequential_mtp_verify_caps_default_k_to_one() {
        assert_eq!(mtp_effective_k(4, 4, true, false), 1);
        assert_eq!(mtp_effective_k(4, 4, true, true), 4);
        assert_eq!(mtp_effective_k(4, 2, false, false), 2);
    }

    #[test]
    fn gemma4_external_mtp_verify_stays_sequential_until_batch_parity_exists() {
        assert!(mtp_target_needs_sequential(
            rnb_loader::Architecture::Gemma4
        ));
        assert!(!mtp_external_batch_verify_allowed(
            rnb_loader::Architecture::Gemma4,
            true,
            false,
            false
        ));
    }

    #[test]
    fn batch_mtp_verify_uses_requested_window_up_to_remaining_tokens() {
        assert_eq!(mtp_batch_effective_k(8, 8), 8);
        assert_eq!(mtp_batch_effective_k(4, 4), 4);
        assert_eq!(mtp_batch_effective_k(4, 1), 1);
    }

    #[test]
    fn mtp_batch_verify_disable_overrides_default() {
        assert!(!mtp_batch_verify_requested(false, true, true));
        assert!(mtp_batch_verify_requested(true, false, false));
        assert!(mtp_batch_verify_requested(false, false, true));
    }

    #[test]
    fn mtp_device_verify_takes_precedence_over_prefill_batch() {
        assert_eq!(
            mtp_verify_execution(true, false),
            MtpVerifyExecution::DeviceResident
        );
        assert_eq!(
            mtp_verify_execution(true, true),
            MtpVerifyExecution::DeviceResident
        );
        assert_eq!(
            mtp_verify_execution(false, true),
            MtpVerifyExecution::BatchPrefill
        );
        assert_eq!(
            mtp_verify_execution(false, false),
            MtpVerifyExecution::Sequential
        );
    }

    #[test]
    fn mtp_batch_verify_default_stays_off_until_adopted() {
        assert!(!mtp_batch_verify_default_enabled(true));
    }

    #[test]
    fn mtp_device_verify_defaults_to_no_bonus_window() {
        assert!(mtp_no_bonus_verify_enabled(
            MtpVerifyExecution::DeviceResident,
            None
        ));
        assert!(!mtp_no_bonus_verify_enabled(
            MtpVerifyExecution::BatchPrefill,
            None
        ));
        assert!(!mtp_no_bonus_verify_enabled(
            MtpVerifyExecution::DeviceResident,
            Some(false)
        ));
        assert!(mtp_no_bonus_verify_enabled(
            MtpVerifyExecution::BatchPrefill,
            Some(true)
        ));
    }

    #[test]
    fn mtp_device_verify_includes_bonus_for_single_draft_by_default() {
        assert!(!mtp_no_bonus_verify_for_round(
            MtpVerifyExecution::DeviceResident,
            None,
            1
        ));
        assert!(mtp_no_bonus_verify_for_round(
            MtpVerifyExecution::DeviceResident,
            None,
            2
        ));
        assert!(mtp_no_bonus_verify_for_round(
            MtpVerifyExecution::DeviceResident,
            Some(true),
            1
        ));
        assert!(!mtp_no_bonus_verify_for_round(
            MtpVerifyExecution::DeviceResident,
            Some(false),
            2
        ));
    }

    #[test]
    fn mtp_verified_carry_uses_one_extra_device_draft_when_room_remains() {
        assert_eq!(
            mtp_verified_carry_depth(true, MtpVerifyExecution::DeviceResident, 4, 5),
            1
        );
        assert_eq!(
            mtp_verified_carry_depth(true, MtpVerifyExecution::DeviceResident, 4, 4),
            0
        );
        assert_eq!(
            mtp_verified_carry_depth(true, MtpVerifyExecution::BatchPrefill, 4, 5),
            0
        );
        assert_eq!(
            mtp_verified_carry_depth(false, MtpVerifyExecution::DeviceResident, 4, 5),
            0
        );
    }

    #[test]
    fn mtp_verified_runway_depth_can_stack_multiple_future_tokens() {
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::DeviceResident,
                1,
                8,
                None,
                0,
            ),
            0
        );
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::DeviceResident,
                1,
                8,
                None,
                1,
            ),
            1
        );
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::DeviceResident,
                1,
                8,
                None,
                3,
            ),
            2
        );
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::DeviceResident,
                1,
                8,
                Some(4),
                4,
            ),
            4
        );
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::DeviceResident,
                4,
                5,
                Some(4),
                4,
            ),
            1
        );
        assert_eq!(
            mtp_verified_runway_extra_depth(
                true,
                MtpVerifyExecution::BatchPrefill,
                1,
                8,
                Some(4),
                4,
            ),
            0
        );
    }

    #[test]
    fn mtp_verified_runway_consumes_tokens_before_resume_token() {
        let mut runway = MtpVerifiedRunway::new(vec![11, 12], 13).expect("non-empty runway");

        assert_eq!(runway.pop_emit(), Some((11, MtpRunwayAfterEmit::Continue)));
        assert_eq!(
            runway.pop_emit(),
            Some((12, MtpRunwayAfterEmit::Resume { next_token: 13 }))
        );
        assert_eq!(runway.pop_emit(), None);
        assert!(MtpVerifiedRunway::new(Vec::new(), 13).is_none());
    }

    #[test]
    fn mtp_hidden_prefix_rows_slices_committed_rows() {
        assert_eq!(
            mtp_hidden_prefix_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 2).unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn draft_only_emit_rejects_stop_and_eos_tokens() {
        assert!(draft_only_should_emit(7, 2, &[3, 4]));
        assert!(!draft_only_should_emit(2, 2, &[3, 4]));
        assert!(!draft_only_should_emit(4, 2, &[3, 4]));
    }
}
