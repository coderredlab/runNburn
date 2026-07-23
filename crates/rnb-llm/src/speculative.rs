use crate::kv_cache::{KVCache, SsmLayerState};

/// Speculative decoding을 위한 KV cache + SSM state checkpoint.
/// Draft 시작 전에 저장하고, reject 시 복원한다.
pub struct SpecCheckpoint {
    pub kv_len: usize,
    ssm_states: Vec<Option<SsmLayerCheckpoint>>,
    scratch: Option<crate::engine::ScratchBuffers>,
    mtp: Option<crate::engine::mtp::EngineMtpCheckpoint>,
    #[cfg(feature = "cuda")]
    resident_delta_snapshots: Vec<ResidentDeltaSnapshot>,
}

#[derive(Clone)]
struct SsmLayerCheckpoint {
    conv_state: Vec<f32>,
    delta_state: Option<Vec<f32>>,
    conv_kernel: usize,
    conv_channels: usize,
}

#[cfg(feature = "cuda")]
struct ResidentDeltaSnapshot {
    layer_idx: usize,
    snapshot: Option<crate::engine::cuda_runtime::DeltaStateSnapshot>,
}

#[cfg(feature = "cuda")]
impl Drop for ResidentDeltaSnapshot {
    fn drop(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            let _ = crate::engine::cuda_runtime::free_delta_state_snapshot(snapshot);
        }
    }
}

impl SpecCheckpoint {
    /// 현재 KV cache 상태를 checkpoint로 저장.
    /// SSM states는 clone됨 (GDN 레이어당 ~1MB).
    pub fn save(kv_cache: &KVCache) -> Self {
        Self {
            kv_len: kv_cache.current_len(),
            ssm_states: checkpoint_ssm_states(&kv_cache.ssm_states, &[]),
            scratch: None,
            mtp: None,
            #[cfg(feature = "cuda")]
            resident_delta_snapshots: Vec::new(),
        }
    }

    pub fn save_engine(engine: &mut Engine) -> crate::error::Result<Self> {
        #[cfg(feature = "cuda")]
        let resident_delta_snapshots = save_resident_delta_snapshots(&mut engine.kv_cache)?;
        #[cfg(feature = "cuda")]
        let resident_delta_layers = resident_delta_snapshots
            .iter()
            .map(|snapshot| snapshot.layer_idx)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "cuda"))]
        engine.materialize_sequence_state()?;
        Ok(Self {
            kv_len: engine.kv_cache.current_len(),
            ssm_states: checkpoint_ssm_states(
                &engine.kv_cache.ssm_states,
                #[cfg(feature = "cuda")]
                &resident_delta_layers,
                #[cfg(not(feature = "cuda"))]
                &[],
            ),
            scratch: engine.scratch_checkpoint(),
            mtp: engine.mtp_checkpoint(),
            #[cfg(feature = "cuda")]
            resident_delta_snapshots,
        })
    }

    /// Checkpoint 시점으로 KV cache + SSM state 복원.
    /// KV cache의 데이터는 남아있지만 current_len 기준으로 무시됨.
    /// &self로 받아서 restore 후에도 kv_len 등 필드 참조 가능.
    pub fn restore(&self, kv_cache: &mut KVCache) {
        kv_cache.set_len(self.kv_len);
        restore_ssm_states_preserving_allocations(&mut kv_cache.ssm_states, &self.ssm_states);
    }

    pub fn restore_engine(&self, engine: &mut Engine) -> crate::error::Result<()> {
        self.restore(&mut engine.kv_cache);
        engine.restore_scratch_checkpoint(&self.scratch);
        engine.mtp_restore_checkpoint(self.mtp.as_ref());
        #[cfg(feature = "cuda")]
        {
            let restored_resident = restore_resident_delta_snapshots(
                &mut engine.kv_cache,
                &self.resident_delta_snapshots,
            )?;
            engine.finalize_resident_sequence_state_after_restore(restored_resident)?;
        }
        #[cfg(not(feature = "cuda"))]
        engine.invalidate_resident_sequence_state_after_restore()?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn save_resident_delta_snapshots(
    kv_cache: &mut KVCache,
) -> crate::error::Result<Vec<ResidentDeltaSnapshot>> {
    let mut snapshots = Vec::new();
    for (layer_idx, ssm) in kv_cache.ssm_states.iter_mut().enumerate() {
        let Some(state) = ssm.as_mut() else {
            continue;
        };
        let snapshot =
            crate::engine::cuda_runtime::snapshot_delta_state_cache(&mut state.delta_state)
                .map_err(crate::error::LlmError::Forward)?;
        if snapshot.is_some() {
            snapshots.push(ResidentDeltaSnapshot {
                layer_idx,
                snapshot,
            });
        }
    }
    Ok(snapshots)
}

#[cfg(feature = "cuda")]
fn restore_resident_delta_snapshots(
    kv_cache: &mut KVCache,
    snapshots: &[ResidentDeltaSnapshot],
) -> crate::error::Result<bool> {
    let mut restored_any = false;
    for snapshot in snapshots {
        let Some(snapshot_state) = snapshot.snapshot.as_ref() else {
            continue;
        };
        let state = kv_cache
            .ssm_states
            .get_mut(snapshot.layer_idx)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "missing SSM state for resident delta snapshot layer {}",
                    snapshot.layer_idx
                ))
            })?;
        let restored = crate::engine::cuda_runtime::restore_delta_state_cache(
            &mut state.delta_state,
            snapshot_state,
        )
        .map_err(crate::error::LlmError::Forward)?;
        if !restored {
            return Err(crate::error::LlmError::Forward(format!(
                "missing resident delta state for snapshot layer {}",
                snapshot.layer_idx
            )));
        }
        restored_any = true;
    }
    Ok(restored_any)
}

fn restore_ssm_states_preserving_allocations(
    current: &mut Vec<Option<SsmLayerState>>,
    saved: &[Option<SsmLayerCheckpoint>],
) {
    if current.len() != saved.len() {
        *current = saved
            .iter()
            .map(|saved| {
                saved.as_ref().and_then(|saved| {
                    saved.delta_state.as_ref().map(|delta_state| SsmLayerState {
                        conv_state: saved.conv_state.clone(),
                        delta_state: delta_state.clone(),
                        conv_kernel: saved.conv_kernel,
                        conv_channels: saved.conv_channels,
                    })
                })
            })
            .collect();
        return;
    }

    for (current, saved) in current.iter_mut().zip(saved.iter()) {
        match (current.as_mut(), saved.as_ref()) {
            (Some(current), Some(saved))
                if current.conv_state.len() == saved.conv_state.len()
                    && saved
                        .delta_state
                        .as_ref()
                        .is_none_or(|delta| current.delta_state.len() == delta.len()) =>
            {
                current.conv_state.copy_from_slice(&saved.conv_state);
                if let Some(delta_state) = saved.delta_state.as_ref() {
                    current.delta_state.copy_from_slice(delta_state);
                }
                current.conv_kernel = saved.conv_kernel;
                current.conv_channels = saved.conv_channels;
            }
            (_, Some(saved)) => {
                *current = saved.delta_state.as_ref().map(|delta_state| SsmLayerState {
                    conv_state: saved.conv_state.clone(),
                    delta_state: delta_state.clone(),
                    conv_kernel: saved.conv_kernel,
                    conv_channels: saved.conv_channels,
                });
            }
            (_, None) => {
                *current = None;
            }
        }
    }
}

fn checkpoint_ssm_states(
    states: &[Option<SsmLayerState>],
    resident_delta_layers: &[usize],
) -> Vec<Option<SsmLayerCheckpoint>> {
    states
        .iter()
        .enumerate()
        .map(|(layer_idx, state)| {
            state.as_ref().map(|state| {
                let delta_state = if resident_delta_layers.contains(&layer_idx) {
                    None
                } else {
                    Some(state.delta_state.clone())
                };
                SsmLayerCheckpoint {
                    conv_state: state.conv_state.clone(),
                    delta_state,
                    conv_kernel: state.conv_kernel,
                    conv_channels: state.conv_channels,
                }
            })
        })
        .collect()
}

use crate::engine::Engine;
use crate::generate::{GenerateParams, GenerateResult, GeneratedTextStream};
use crate::sampler::SamplerChain;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

/// Speculative decoding 통계
struct SpecStats {
    total_rounds: usize,
    total_accepted: usize,
    total_drafted: usize,
    target_verify_steps: usize,
}

impl SpecStats {
    fn add_target_verify_steps(&mut self, steps: usize) {
        self.target_verify_steps += steps;
    }

    fn accept_rate(&self) -> f32 {
        if self.total_drafted > 0 {
            self.total_accepted as f32 / self.total_drafted as f32 * 100.0
        } else {
            0.0
        }
    }

    fn report(&self) -> String {
        format!(
            "  [SPEC] rounds={}, drafted={}, accepted={}, rate={:.1}%, target_verify_steps={}",
            self.total_rounds,
            self.total_drafted,
            self.total_accepted,
            self.accept_rate(),
            self.target_verify_steps,
        )
    }
}

#[derive(Default)]
struct SpecPhaseTimings {
    checkpoint_ms: f64,
    draft_ms: f64,
    restore_ms: f64,
    verify_ms: f64,
    observe_ms: f64,
    replay_ms: f64,
}

impl SpecPhaseTimings {
    fn add_checkpoint(&mut self, ms: f64) {
        self.checkpoint_ms += ms;
    }

    fn add_draft(&mut self, ms: f64) {
        self.draft_ms += ms;
    }

    fn add_restore(&mut self, ms: f64) {
        self.restore_ms += ms;
    }

    fn add_verify(&mut self, ms: f64) {
        self.verify_ms += ms;
    }

    fn add_observe(&mut self, ms: f64) {
        self.observe_ms += ms;
    }

    fn add_replay(&mut self, ms: f64) {
        self.replay_ms += ms;
    }

    fn report(&self, rounds: usize) -> String {
        let rounds = rounds.max(1) as f64;
        format!(
            "  [SPEC] phase_ms_per_round checkpoint={:.1}ms/round draft={:.1}ms/round restore={:.1}ms/round verify={:.1}ms/round observe={:.1}ms/round replay={:.1}ms/round",
            self.checkpoint_ms / rounds,
            self.draft_ms / rounds,
            self.restore_ms / rounds,
            self.verify_ms / rounds,
            self.observe_ms / rounds,
            self.replay_ms / rounds,
        )
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecCheckpointScope {
    FullEngine,
    MtpOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecVerifyMode {
    SequentialIncremental,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecBatchVerifyTargetSource {
    ArgmaxTokens,
    Logits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecPreVerifyRestore {
    None,
    FullEngine,
    MtpOnly,
}

fn spec_verify_mode(target_needs_sequential: bool, force_batch: bool) -> SpecVerifyMode {
    if target_needs_sequential && !force_batch {
        SpecVerifyMode::SequentialIncremental
    } else {
        SpecVerifyMode::Batch
    }
}

fn spec_batch_verify_target_source(argmax_verify: bool) -> SpecBatchVerifyTargetSource {
    if argmax_verify {
        SpecBatchVerifyTargetSource::ArgmaxTokens
    } else {
        SpecBatchVerifyTargetSource::Logits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecBatchMtpObserveMode {
    Immediate,
    DeferredUntilCommit,
}

fn spec_batch_mtp_observe_mode(
    use_mtp: bool,
    target_source: SpecBatchVerifyTargetSource,
) -> SpecBatchMtpObserveMode {
    if use_mtp && target_source == SpecBatchVerifyTargetSource::ArgmaxTokens {
        SpecBatchMtpObserveMode::DeferredUntilCommit
    } else {
        SpecBatchMtpObserveMode::Immediate
    }
}

fn spec_batch_prefix_snapshot_k_for_verify_input(k: usize, no_bonus: bool) -> usize {
    if no_bonus {
        k.saturating_sub(1)
    } else {
        k
    }
}

fn spec_batch_verify_input_tokens(
    first_token: u32,
    draft_tokens: &[u32],
    include_bonus: bool,
) -> Vec<u32> {
    let keep_drafts = if include_bonus {
        draft_tokens.len()
    } else {
        draft_tokens.len().saturating_sub(1)
    };
    let mut verify_input = Vec::with_capacity(1 + keep_drafts);
    verify_input.push(first_token);
    verify_input.extend_from_slice(&draft_tokens[..keep_drafts]);
    verify_input
}

fn effective_spec_k_for_round(
    requested_k: usize,
    use_mtp: bool,
    verify_mode: SpecVerifyMode,
    mtp_sequential_multi: bool,
) -> usize {
    if use_mtp && verify_mode == SpecVerifyMode::SequentialIncremental && !mtp_sequential_multi {
        requested_k.min(1)
    } else {
        requested_k
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn spec_batch_prefix_snapshot_tokens_with_capability(
    requested: bool,
    use_mtp: bool,
    target_needs_sequential: bool,
    verify_mode: SpecVerifyMode,
    argmax_verify: bool,
    k: usize,
    trace_mtp: bool,
    prefix_restore_supported: bool,
) -> Option<usize> {
    if requested
        && use_mtp
        && target_needs_sequential
        && verify_mode == SpecVerifyMode::Batch
        && argmax_verify
        && matches!(k, 1 | 2)
        && !trace_mtp
        && prefix_restore_supported
    {
        Some(k)
    } else {
        None
    }
}

fn spec_batch_prefix_restore_supported() -> bool {
    cfg!(feature = "cuda")
}

fn spec_checkpoint_scope(
    use_mtp: bool,
    verify_mode: SpecVerifyMode,
    trace_mtp: bool,
    batch_prefix_snapshot_tokens: Option<usize>,
) -> SpecCheckpointScope {
    if use_mtp
        && !trace_mtp
        && (verify_mode == SpecVerifyMode::SequentialIncremental
            || (verify_mode == SpecVerifyMode::Batch && batch_prefix_snapshot_tokens == Some(1)))
    {
        SpecCheckpointScope::MtpOnly
    } else {
        SpecCheckpointScope::FullEngine
    }
}

fn spec_pre_verify_restore(
    checkpoint_scope: SpecCheckpointScope,
    use_mtp: bool,
    verify_mode: SpecVerifyMode,
    trace_mtp: bool,
) -> SpecPreVerifyRestore {
    match checkpoint_scope {
        SpecCheckpointScope::MtpOnly => {
            if use_mtp && verify_mode == SpecVerifyMode::Batch && !trace_mtp {
                SpecPreVerifyRestore::MtpOnly
            } else {
                SpecPreVerifyRestore::None
            }
        }
        SpecCheckpointScope::FullEngine => {
            if use_mtp && verify_mode == SpecVerifyMode::Batch && !trace_mtp {
                SpecPreVerifyRestore::MtpOnly
            } else {
                SpecPreVerifyRestore::FullEngine
            }
        }
    }
}

fn spec_phase_report_enabled(global_profile: bool, spec_profile: bool) -> bool {
    global_profile || spec_profile
}

fn should_retain_mtp_draft_after_verify(
    use_mtp_draft_retention: bool,
    stopped: bool,
    tokens_remaining: usize,
) -> bool {
    use_mtp_draft_retention && !stopped && tokens_remaining > 0
}

fn spec_argmax_verify_allowed(params: &GenerateParams) -> bool {
    params.temperature == 0.0
        && params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
        && params.mirostat.is_none()
}

/// Speculative decoding으로 텍스트 생성.
///
/// 매 라운드:
/// 1. current logits에서 first_token 샘플 & emit
/// 2. early-exit draft로 K개 토큰 생성
/// 3. checkpoint restore 후 [first_token, d0..d_{k-1}] verify prefill
/// 4. accept/reject — reject 시 다음 라운드에서 target을 first_token으로 뽑음
pub fn generate_stream_speculative(
    engine: &mut Engine,
    prompt: &str,
    params: &GenerateParams,
    mut callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    // Mock engine fallback (weights 없으면 draft 불가)
    if !engine.has_weights() {
        return crate::generate::generate_stream_impl(engine, prompt, params, callback);
    }
    if engine.mtp_spec_requested() {
        return crate::mtp_generate::generate_stream_mtp(engine, prompt, params, callback);
    }

    let start = Instant::now();
    let spec_k = params.spec_k;
    let draft_depth = (engine.metadata.num_layers as f32 * params.spec_depth).max(1.0) as usize;

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

    engine.clear_sequence_state()?;

    // Prefill
    let mut logits = engine.forward_prompt(&prompt_tokens)?;
    let mut forced_next_token: Option<u32> = None;

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut generated_text = GeneratedTextStream::new();
    let mut tokens_remaining = params.max_tokens;

    let mut stats = SpecStats {
        total_rounds: 0,
        total_accepted: 0,
        total_drafted: 0,
        target_verify_steps: 0,
    };
    let mut phase_timings = SpecPhaseTimings::default();
    let use_mtp = engine.mtp_spec_requested();
    if use_mtp && !engine.mtp_runtime_ready() {
        return Err(crate::error::LlmError::Forward(
            "RNB_MTP=1 but loaded model does not expose an MTP runtime".to_string(),
        ));
    }
    let trace_mtp = use_mtp && crate::runtime::mtp_trace_enabled();
    let argmax_verify = spec_argmax_verify_allowed(params);
    let force_batch_verify = crate::runtime::spec_force_batch_verify_enabled();
    let decode_fast_window = crate::runtime::spec_decode_fast_window_enabled();
    let batch_no_bonus_requested = crate::runtime::spec_batch_no_bonus_enabled();
    let batch_prefix_snapshot_requested = crate::runtime::spec_batch_prefix_snapshot_enabled();

    while tokens_remaining > 0 {
        crate::generate::check_generation_cancellation()?;
        let target_needs_sequential = target_verify_needs_sequential(engine);
        let verify_mode = spec_verify_mode(target_needs_sequential, force_batch_verify);
        // Sample first token from current logits
        let first_token = if let Some(token) = forced_next_token.take() {
            token
        } else {
            sampler.sample(&mut logits, &generated_tokens, &mut rng)
        };
        if params.stop_tokens.contains(&first_token) || first_token == eos {
            break;
        }
        generated_tokens.push(first_token);
        if !generated_text.push(&engine.tokenizer, first_token, &mut callback) {
            break;
        }
        tokens_remaining -= 1;
        if tokens_remaining == 0 {
            break;
        }

        // --- Speculative round ---
        let mtp_sequential_multi = crate::runtime::spec_mtp_sequential_multi_enabled();
        let k = effective_spec_k_for_round(spec_k, use_mtp, verify_mode, mtp_sequential_multi)
            .min(tokens_remaining);
        let batch_argmax_no_bonus =
            verify_mode == SpecVerifyMode::Batch && argmax_verify && batch_no_bonus_requested;
        let prefix_snapshot_k =
            spec_batch_prefix_snapshot_k_for_verify_input(k, batch_argmax_no_bonus);
        let batch_prefix_snapshot_tokens = spec_batch_prefix_snapshot_tokens_with_capability(
            batch_prefix_snapshot_requested,
            use_mtp,
            target_needs_sequential,
            verify_mode,
            argmax_verify,
            prefix_snapshot_k,
            trace_mtp,
            spec_batch_prefix_restore_supported(),
        );
        let round_base_kv_len = engine.kv_cache.current_len();
        stats.total_rounds += 1;
        stats.total_drafted += k;

        // 1. Checkpoint
        let checkpoint_scope = spec_checkpoint_scope(
            use_mtp,
            verify_mode,
            trace_mtp,
            batch_prefix_snapshot_tokens,
        );
        let mut full_ckpt = None;
        let mut mtp_ckpt = None;
        let phase_start = Instant::now();
        match checkpoint_scope {
            SpecCheckpointScope::FullEngine => {
                full_ckpt = Some(SpecCheckpoint::save_engine(engine)?);
            }
            SpecCheckpointScope::MtpOnly => {
                mtp_ckpt = engine.mtp_checkpoint();
            }
        }
        phase_timings.add_checkpoint(elapsed_ms(phase_start));
        if trace_mtp {
            let diag_logits = engine.forward_verify_all_logits_sequential(&[first_token])?;
            eprintln!(
                "[MTP_TRACE] round={} direct_next_after_first={}",
                stats.total_rounds,
                trace_top_token(engine, &diag_logits[0])
            );
            full_ckpt
                .as_ref()
                .expect("MTP trace requires full speculative checkpoint")
                .restore_engine(engine)?;
        }

        // 2. Draft K tokens using early-exit
        let phase_start = Instant::now();
        let draft_tokens = if use_mtp {
            engine.mtp_draft_tokens(first_token, k)?
        } else {
            let mut draft_tokens = Vec::with_capacity(k);
            let mut last_token = first_token;
            for _ in 0..k {
                crate::generate::check_generation_cancellation()?;
                let draft_logits = engine.forward_decode_draft(last_token, draft_depth)?;
                // Draft uses greedy (argmax)
                let t = draft_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                draft_tokens.push(t);
                last_token = t;
            }
            draft_tokens
        };
        phase_timings.add_draft(elapsed_ms(phase_start));
        if trace_mtp {
            eprintln!(
                "[MTP_TRACE] round={} first={} drafts=[{}]",
                stats.total_rounds,
                trace_token(engine, first_token),
                draft_tokens
                    .iter()
                    .map(|&token| trace_token(engine, token))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // 3. Rollback to checkpoint
        let phase_start = Instant::now();
        match spec_pre_verify_restore(checkpoint_scope, use_mtp, verify_mode, trace_mtp) {
            SpecPreVerifyRestore::FullEngine => full_ckpt
                .as_ref()
                .expect("full speculative checkpoint missing")
                .restore_engine(engine)?,
            SpecPreVerifyRestore::MtpOnly => match checkpoint_scope {
                SpecCheckpointScope::FullEngine => {
                    let mtp = full_ckpt
                        .as_ref()
                        .and_then(|checkpoint| checkpoint.mtp.as_ref());
                    engine.mtp_restore_checkpoint(mtp);
                }
                SpecCheckpointScope::MtpOnly => {
                    engine.mtp_restore_checkpoint(mtp_ckpt.as_ref());
                }
            },
            SpecPreVerifyRestore::None => {}
        }
        phase_timings.add_restore(elapsed_ms(phase_start));

        // 4. Verify. Normal batch input includes one bonus row:
        // [first_token, d0, ..., d_{k-1}]. The opt-in no-bonus path omits
        // the last draft row and computes it later only if all drafts accept.
        let verify_input =
            spec_batch_verify_input_tokens(first_token, &draft_tokens, !batch_argmax_no_bonus);

        if verify_mode == SpecVerifyMode::SequentialIncremental {
            if trace_mtp {
                eprintln!(
                    "[MTP_TRACE] round={} verify_mode=sequential-incremental",
                    stats.total_rounds
                );
            }
            if decode_fast_window && argmax_verify {
                let phase_start = Instant::now();
                let diagnostic_checkpoint = SpecCheckpoint::save_engine(engine)?;
                let window = engine.forward_verify_window_argmax_collect_mtp(&verify_input)?;
                diagnostic_checkpoint.restore_engine(engine)?;
                phase_timings.add_verify(elapsed_ms(phase_start));
                eprintln!(
                    "[SPEC] decode-fast-window diagnostic tokens={} hidden_rows={}",
                    window.len(),
                    window.hidden_rows()
                );
            }
            let mut n_accepted = 0usize;
            let mut draft_matches = 0usize;
            let mut stopped = false;
            let mut next_logits: Option<Vec<f32>> = None;
            let mut next_forced_token: Option<u32> = None;
            let mut target_trace = trace_mtp.then(Vec::new);
            let mut observed_tokens = Vec::with_capacity(1 + k);
            let mut observed_hidden_rows = Vec::new();

            let phase_start = Instant::now();
            for i in 0..k {
                let (target_token, logits_i, hidden_rows) = if argmax_verify {
                    let (target_token, hidden_rows) =
                        engine.forward_verify_argmax_sequential_collect_mtp(verify_input[i])?;
                    stats.add_target_verify_steps(1);
                    (target_token, None, hidden_rows)
                } else {
                    let (mut step_logits, hidden_rows) = engine
                        .forward_verify_all_logits_sequential_collect_mtp(
                            &[verify_input[i]],
                            false,
                        )?;
                    stats.add_target_verify_steps(1);
                    let logits_i = step_logits.pop().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "incremental spec verify produced no logits".to_string(),
                        )
                    })?;
                    let mut verify_logits = logits_i.clone();
                    let target_token =
                        sampler.sample(&mut verify_logits, &generated_tokens, &mut rng);
                    (target_token, Some(logits_i), hidden_rows)
                };
                if use_mtp {
                    observed_tokens.push(verify_input[i]);
                    observed_hidden_rows.extend_from_slice(&hidden_rows);
                }
                if let Some(trace) = target_trace.as_mut() {
                    trace.push(format!(
                        "{}:{}{}",
                        i,
                        trace_token(engine, target_token),
                        if target_token == draft_tokens[i] {
                            "=draft"
                        } else {
                            "!=draft"
                        }
                    ));
                }

                if target_token == draft_tokens[i] {
                    if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                        stopped = true;
                        break;
                    }
                    generated_tokens.push(draft_tokens[i]);
                    if !generated_text.push(&engine.tokenizer, draft_tokens[i], &mut callback) {
                        stopped = true;
                        n_accepted += 1;
                        draft_matches += 1;
                        tokens_remaining -= 1;
                        break;
                    }
                    n_accepted += 1;
                    draft_matches += 1;
                    tokens_remaining -= 1;
                    if tokens_remaining == 0 {
                        break;
                    }
                } else {
                    if let Some(logits_i) = logits_i {
                        next_logits = Some(logits_i);
                    } else {
                        next_forced_token = Some(target_token);
                    }
                    break;
                }
            }

            if !stopped && tokens_remaining > 0 && n_accepted == k {
                let (target_token, logits_i, hidden_rows) = if argmax_verify {
                    let (target_token, hidden_rows) =
                        engine.forward_verify_argmax_sequential_collect_mtp(verify_input[k])?;
                    stats.add_target_verify_steps(1);
                    (target_token, None, hidden_rows)
                } else {
                    let (mut bonus_logits, hidden_rows) = engine
                        .forward_verify_all_logits_sequential_collect_mtp(
                            &[verify_input[k]],
                            false,
                        )?;
                    stats.add_target_verify_steps(1);
                    let logits_i = bonus_logits.pop().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "incremental spec bonus verify produced no logits".to_string(),
                        )
                    })?;
                    (0, Some(logits_i), hidden_rows)
                };
                if use_mtp {
                    observed_tokens.push(verify_input[k]);
                    observed_hidden_rows.extend_from_slice(&hidden_rows);
                }
                if let Some(logits_i) = logits_i {
                    next_logits = Some(logits_i);
                } else {
                    next_forced_token = Some(target_token);
                }
            }
            phase_timings.add_verify(elapsed_ms(phase_start));

            if let Some(trace) = target_trace {
                eprintln!(
                    "[MTP_TRACE] round={} accepted={} targets=[{}]",
                    stats.total_rounds,
                    n_accepted,
                    trace.join(", ")
                );
            }

            stats.total_accepted += draft_matches;

            if should_retain_mtp_draft_after_verify(
                use_mtp && checkpoint_scope == SpecCheckpointScope::MtpOnly,
                stopped,
                tokens_remaining,
            ) {
                let phase_start = Instant::now();
                engine.mtp_retain_draft_after_spec(
                    mtp_ckpt.as_ref(),
                    &verify_input,
                    1 + n_accepted,
                    k,
                    &observed_hidden_rows,
                )?;
                phase_timings.add_observe(elapsed_ms(phase_start));
            } else if use_mtp && !observed_tokens.is_empty() {
                let phase_start = Instant::now();
                engine.mtp_observe_target_batch(&observed_tokens, &observed_hidden_rows)?;
                phase_timings.add_observe(elapsed_ms(phase_start));
            }

            if stopped || tokens_remaining == 0 {
                break;
            }
            if let Some(token) = next_forced_token {
                forced_next_token = Some(token);
            } else {
                logits = next_logits.ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "incremental spec verify did not produce next logits".to_string(),
                    )
                })?;
            }
            continue;
        }

        let batch_target_source = spec_batch_verify_target_source(argmax_verify);
        if batch_target_source == SpecBatchVerifyTargetSource::ArgmaxTokens {
            let phase_start = Instant::now();
            let defer_mtp_observe = spec_batch_mtp_observe_mode(use_mtp, batch_target_source)
                == SpecBatchMtpObserveMode::DeferredUntilCommit;
            let window = if let Some(prefix_tokens) = batch_prefix_snapshot_tokens {
                if defer_mtp_observe {
                    engine.forward_prefill_argmax_tokens_collect_mtp_prefix_state_deferred_observe(
                        &verify_input,
                        prefix_tokens,
                    )?
                } else {
                    engine.forward_prefill_argmax_tokens_collect_mtp_prefix_state(
                        &verify_input,
                        prefix_tokens,
                    )?
                }
            } else if defer_mtp_observe {
                engine.forward_prefill_argmax_tokens_collect_mtp_deferred_observe(&verify_input)?
            } else {
                engine.forward_prefill_argmax_tokens_collect_mtp(&verify_input)?
            };
            stats.add_target_verify_steps(window.len());
            phase_timings.add_verify(elapsed_ms(phase_start));
            if trace_mtp {
                eprintln!(
                    "[MTP_TRACE] round={} verify_mode=batch-argmax tops=[{}]",
                    stats.total_rounds,
                    window
                        .target_tokens
                        .iter()
                        .take(k)
                        .enumerate()
                        .map(|(idx, &token)| format!("{}:{}", idx, trace_token(engine, token)))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let mut n_accepted = 0usize;
            let mut draft_matches = 0usize;
            let mut stopped = false;
            let mut target_trace = trace_mtp.then(Vec::new);

            for i in 0..k {
                let target_token = *window.target_tokens.get(i).ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "batch argmax verify missing target token at {i}"
                    ))
                })?;
                if let Some(trace) = target_trace.as_mut() {
                    trace.push(format!(
                        "{}:{}{}",
                        i,
                        trace_token(engine, target_token),
                        if target_token == draft_tokens[i] {
                            "=draft"
                        } else {
                            "!=draft"
                        }
                    ));
                }

                if target_token == draft_tokens[i] {
                    if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                        stopped = true;
                        break;
                    }
                    generated_tokens.push(draft_tokens[i]);
                    if !generated_text.push(&engine.tokenizer, draft_tokens[i], &mut callback) {
                        stopped = true;
                        n_accepted += 1;
                        draft_matches += 1;
                        tokens_remaining -= 1;
                        break;
                    }
                    n_accepted += 1;
                    draft_matches += 1;
                    tokens_remaining -= 1;
                    if tokens_remaining == 0 {
                        break;
                    }
                } else {
                    break;
                }
            }

            if let Some(trace) = target_trace {
                eprintln!(
                    "[MTP_TRACE] round={} accepted={} targets=[{}]",
                    stats.total_rounds,
                    n_accepted,
                    trace.join(", ")
                );
            }

            stats.total_accepted += draft_matches;

            if batch_argmax_no_bonus && n_accepted == k && tokens_remaining == 0 {
                break;
            }

            let committed_verify_tokens = 1 + n_accepted;
            let current_state_matches_committed =
                batch_argmax_no_bonus && committed_verify_tokens == window.len();
            let prefix_snapshot_available = current_state_matches_committed
                || window
                    .prefix_state
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.prefix_tokens == committed_verify_tokens);
            let state_plan = verify_state_plan_after_spec_round_with_prefix_snapshot(
                target_needs_sequential,
                verify_mode,
                k,
                n_accepted,
                prefix_snapshot_available,
            );
            if batch_argmax_no_bonus && n_accepted == k && !stopped && tokens_remaining > 0 {
                let phase_start = Instant::now();
                if defer_mtp_observe && use_mtp {
                    let mtp_checkpoint = match checkpoint_scope {
                        SpecCheckpointScope::FullEngine => full_ckpt
                            .as_ref()
                            .and_then(|checkpoint| checkpoint.mtp.as_ref()),
                        SpecCheckpointScope::MtpOnly => mtp_ckpt.as_ref(),
                    };
                    engine.mtp_restore_checkpoint(mtp_checkpoint);
                    engine.mtp_observe_target_batch(&verify_input, &window.mtp_hidden_rows)?;
                }
                let last_draft = *draft_tokens.last().ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "batch no-bonus verify missing last draft token".to_string(),
                    )
                })?;
                let (next_token, hidden_rows) =
                    engine.forward_verify_argmax_sequential_collect_mtp(last_draft)?;
                stats.add_target_verify_steps(1);
                if use_mtp && !hidden_rows.is_empty() {
                    engine.mtp_observe_target_batch(&[last_draft], &hidden_rows)?;
                }
                phase_timings.add_verify(elapsed_ms(phase_start));
                forced_next_token = Some(next_token);
                continue;
            }
            let next_token = if state_plan.replay_committed_prefix {
                let ckpt = full_ckpt.as_ref().ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "spec argmax replay requires full engine checkpoint".to_string(),
                    )
                })?;
                let phase_start = Instant::now();
                let token = replay_committed_verify_prefix_argmax(
                    engine,
                    ckpt,
                    &verify_input,
                    state_plan.committed_verify_tokens,
                )?;
                phase_timings.add_replay(elapsed_ms(phase_start));
                token
            } else if target_needs_sequential
                && verify_mode == SpecVerifyMode::Batch
                && n_accepted < k
                && !current_state_matches_committed
            {
                let prefix_state = window.prefix_state.as_ref().ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "batch argmax verify missing prefix state snapshot".to_string(),
                    )
                })?;
                if prefix_state.prefix_tokens != state_plan.committed_verify_tokens {
                    return Err(crate::error::LlmError::Forward(format!(
                        "batch argmax prefix snapshot length {} != committed {}",
                        prefix_state.prefix_tokens, state_plan.committed_verify_tokens
                    )));
                }
                let phase_start = Instant::now();
                if let Some(ckpt) = full_ckpt.as_ref() {
                    ckpt.restore_engine(engine)?;
                    engine.restore_verify_window_prefix_state(ckpt.kv_len, prefix_state)?;
                } else {
                    engine.restore_verify_window_prefix_state(round_base_kv_len, prefix_state)?;
                }
                if should_restore_mtp_before_prefix_observe(
                    checkpoint_scope,
                    use_mtp,
                    target_needs_sequential,
                    verify_mode,
                    n_accepted < k,
                ) {
                    engine.mtp_restore_checkpoint(mtp_ckpt.as_ref());
                }
                let committed = verify_input
                    .get(..state_plan.committed_verify_tokens)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "batch prefix committed token count {} exceeds verify input {}",
                            state_plan.committed_verify_tokens,
                            verify_input.len()
                        ))
                    })?;
                let committed_hidden =
                    window.mtp_hidden_prefix_rows(state_plan.committed_verify_tokens)?;
                engine.mtp_observe_target_batch(committed, committed_hidden)?;
                phase_timings.add_replay(elapsed_ms(phase_start));
                *window
                    .target_tokens
                    .get(state_plan.committed_verify_tokens - 1)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "batch argmax verify missing next token".to_string(),
                        )
                    })?
            } else {
                let final_len = round_base_kv_len + state_plan.committed_verify_tokens;
                engine
                    .kv_cache
                    .set_len(final_len.min(engine.kv_cache.max_seq_len));
                let mtp_checkpoint = match checkpoint_scope {
                    SpecCheckpointScope::FullEngine => full_ckpt
                        .as_ref()
                        .and_then(|checkpoint| checkpoint.mtp.as_ref()),
                    SpecCheckpointScope::MtpOnly => mtp_ckpt.as_ref(),
                };
                if defer_mtp_observe && use_mtp {
                    engine.mtp_restore_checkpoint(mtp_checkpoint);
                    let committed = verify_input
                        .get(..state_plan.committed_verify_tokens)
                        .ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "batch committed token count {} exceeds verify input {}",
                                state_plan.committed_verify_tokens,
                                verify_input.len()
                            ))
                        })?;
                    let committed_hidden =
                        window.mtp_hidden_prefix_rows(state_plan.committed_verify_tokens)?;
                    engine.mtp_observe_target_batch(committed, committed_hidden)?;
                } else {
                    engine
                        .mtp_retain_after_spec(mtp_checkpoint, state_plan.committed_verify_tokens);
                }
                *window
                    .target_tokens
                    .get(state_plan.committed_verify_tokens - 1)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "batch argmax verify missing next token".to_string(),
                        )
                    })?
            };

            if stopped || tokens_remaining == 0 {
                break;
            }
            forced_next_token = Some(next_token);
            continue;
        }

        let phase_start = Instant::now();
        let all_logits = engine.forward_prefill_all_logits(&verify_input)?;
        stats.add_target_verify_steps(all_logits.len());
        phase_timings.add_verify(elapsed_ms(phase_start));
        if trace_mtp {
            eprintln!(
                "[MTP_TRACE] round={} verify_mode={} tops=[{}]",
                stats.total_rounds,
                "batch",
                all_logits
                    .iter()
                    .take(k)
                    .enumerate()
                    .map(|(idx, logits)| format!("{}:{}", idx, trace_top_token(engine, logits)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        // all_logits[i] = logits from position ckpt.kv_len+i
        //   [0] verifies draft_tokens[0], [1] verifies draft_tokens[1], ...
        //   [k] = logits after last draft token (for bonus)

        // 5. Accept/Reject
        let mut n_accepted = 0usize; // consecutive accepted drafts (for KV len)
        let mut draft_matches = 0usize; // stats: matching drafts only
        let mut stopped = false;
        let mut target_trace = trace_mtp.then(Vec::new);

        for i in 0..k {
            let mut verify_logits = all_logits[i].clone();
            let target_token = sampler.sample(&mut verify_logits, &generated_tokens, &mut rng);
            if let Some(trace) = target_trace.as_mut() {
                trace.push(format!(
                    "{}:{}{}",
                    i,
                    trace_token(engine, target_token),
                    if target_token == draft_tokens[i] {
                        "=draft"
                    } else {
                        "!=draft"
                    }
                ));
            }

            if target_token == draft_tokens[i] {
                // Accept: emit draft token
                if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                    stopped = true;
                    break;
                }
                generated_tokens.push(draft_tokens[i]);
                if !generated_text.push(&engine.tokenizer, draft_tokens[i], &mut callback) {
                    stopped = true;
                    n_accepted += 1;
                    draft_matches += 1;
                    tokens_remaining -= 1;
                    break;
                }
                n_accepted += 1;
                draft_matches += 1;
                tokens_remaining -= 1;
                if tokens_remaining == 0 {
                    break;
                }
            } else {
                // Reject: DON'T emit target token here.
                // Set logits = all_logits[i] so next iteration samples target_token
                // as the first_token of next round.
                break;
            }
        }
        if let Some(trace) = target_trace {
            eprintln!(
                "[MTP_TRACE] round={} accepted={} targets=[{}]",
                stats.total_rounds,
                n_accepted,
                trace.join(", ")
            );
        }

        stats.total_accepted += draft_matches;

        let state_plan =
            verify_state_plan_after_spec_round(target_needs_sequential, verify_mode, k, n_accepted);
        if state_plan.replay_committed_prefix {
            let ckpt = full_ckpt.as_ref().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "spec replay requires full engine checkpoint".to_string(),
                )
            })?;
            let phase_start = Instant::now();
            logits = replay_committed_verify_prefix(
                engine,
                ckpt,
                &verify_input,
                state_plan.committed_verify_tokens,
            )?;
            phase_timings.add_replay(elapsed_ms(phase_start));
        } else {
            let ckpt = full_ckpt.as_ref().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "batched spec state retain requires full engine checkpoint".to_string(),
                )
            })?;
            let final_len = ckpt.kv_len + state_plan.committed_verify_tokens;
            engine
                .kv_cache
                .set_len(final_len.min(engine.kv_cache.max_seq_len));
            engine.mtp_retain_after_spec(ckpt.mtp.as_ref(), state_plan.committed_verify_tokens);
            logits = all_logits[state_plan.committed_verify_tokens - 1].clone();
        }

        if stopped {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let profiling = spec_phase_report_enabled(
        crate::runtime::profiling_enabled(),
        crate::runtime::spec_profile_enabled(),
    );
    if profiling && stats.total_rounds > 0 {
        eprintln!("{}", stats.report());
        eprintln!("{}", phase_timings.report(stats.total_rounds));
    }

    Ok(GenerateResult::new(
        generated_text.finish(&mut callback),
        generated_tokens.len(),
        prompt_len,
        elapsed,
        generated_tokens,
    ))
}

fn trace_token(engine: &Engine, token: u32) -> String {
    format!(
        "{}:{:?}",
        token,
        engine.tokenizer.decode_token(token).to_string()
    )
}

fn trace_top_token(engine: &Engine, logits: &[f32]) -> String {
    let token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0);
    trace_token(engine, token)
}

fn target_verify_needs_sequential(engine: &Engine) -> bool {
    architecture_verify_needs_sequential(engine.architecture())
}

fn architecture_verify_needs_sequential(architecture: rnb_loader::Architecture) -> bool {
    matches!(
        architecture,
        rnb_loader::Architecture::Qwen35
            | rnb_loader::Architecture::Qwen35MoE
            | rnb_loader::Architecture::GlmDsa
    )
}

/// pm116: recurrent(GDN/Mamba) state 를 가진 아키텍처. needs_sequential 과 별개 —
/// GLM(GlmDsa)은 sequential 분류지만 attention-only 라 prefix state 가 필요 없다.
/// batch verify 의 prefix snapshot/replay 판단은 이 함수를 쓴다.
pub(crate) fn architecture_has_recurrent_state(architecture: rnb_loader::Architecture) -> bool {
    matches!(
        architecture,
        rnb_loader::Architecture::Qwen35
            | rnb_loader::Architecture::Qwen35MoE
            | rnb_loader::Architecture::NemotronHMoE
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerifyStatePlan {
    pub(crate) committed_verify_tokens: usize,
    pub(crate) replay_committed_prefix: bool,
}

fn verify_state_plan_after_spec_round(
    target_has_recurrent_state: bool,
    verify_mode: SpecVerifyMode,
    spec_k: usize,
    n_accepted: usize,
) -> VerifyStatePlan {
    verify_state_plan_after_spec_round_with_prefix_snapshot(
        target_has_recurrent_state,
        verify_mode,
        spec_k,
        n_accepted,
        false,
    )
}

pub(crate) fn verify_state_plan_after_batch_round(
    target_has_recurrent_state: bool,
    spec_k: usize,
    n_accepted: usize,
) -> VerifyStatePlan {
    verify_state_plan_after_spec_round(
        target_has_recurrent_state,
        SpecVerifyMode::Batch,
        spec_k,
        n_accepted,
    )
}

fn verify_state_plan_after_spec_round_with_prefix_snapshot(
    target_has_recurrent_state: bool,
    verify_mode: SpecVerifyMode,
    spec_k: usize,
    n_accepted: usize,
    prefix_snapshot_available: bool,
) -> VerifyStatePlan {
    debug_assert!(n_accepted <= spec_k);
    VerifyStatePlan {
        committed_verify_tokens: 1 + n_accepted,
        replay_committed_prefix: target_has_recurrent_state
            && verify_mode == SpecVerifyMode::Batch
            && n_accepted < spec_k
            && !prefix_snapshot_available,
    }
}

fn should_restore_mtp_before_prefix_observe(
    checkpoint_scope: SpecCheckpointScope,
    use_mtp: bool,
    target_has_recurrent_state: bool,
    verify_mode: SpecVerifyMode,
    rejected_before_all_drafts: bool,
) -> bool {
    use_mtp
        && checkpoint_scope == SpecCheckpointScope::MtpOnly
        && target_has_recurrent_state
        && verify_mode == SpecVerifyMode::Batch
        && rejected_before_all_drafts
}

fn replay_committed_verify_prefix(
    engine: &mut Engine,
    checkpoint: &SpecCheckpoint,
    verify_input: &[u32],
    committed_tokens: usize,
) -> crate::error::Result<Vec<f32>> {
    let committed = verify_input.get(..committed_tokens).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "spec replay prefix {} exceeds verify input {}",
            committed_tokens,
            verify_input.len()
        ))
    })?;
    checkpoint.restore_engine(engine)?;
    let replay_logits = engine.forward_verify_all_logits_sequential(committed)?;
    replay_logits.last().cloned().ok_or_else(|| {
        crate::error::LlmError::Forward(
            "spec replay committed prefix produced no logits".to_string(),
        )
    })
}

pub(crate) fn replay_committed_verify_prefix_argmax(
    engine: &mut Engine,
    checkpoint: &SpecCheckpoint,
    verify_input: &[u32],
    committed_tokens: usize,
) -> crate::error::Result<u32> {
    let committed = verify_input.get(..committed_tokens).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "spec argmax replay prefix {} exceeds verify input {}",
            committed_tokens,
            verify_input.len()
        ))
    })?;
    checkpoint.restore_engine(engine)?;
    let mut target_token = None;
    let mut mtp_hidden_rows = Vec::new();
    for &token in committed {
        let (next_token, hidden_rows) =
            engine.forward_verify_argmax_sequential_collect_mtp(token)?;
        target_token = Some(next_token);
        mtp_hidden_rows.extend_from_slice(&hidden_rows);
    }
    if engine.mtp_spec_requested() && !committed.is_empty() {
        engine.mtp_observe_target_batch(committed, &mtp_hidden_rows)?;
    }
    target_token.ok_or_else(|| {
        crate::error::LlmError::Forward(
            "spec argmax replay committed prefix produced no token".to_string(),
        )
    })
}

/// Two-model speculative decoding: 작은 모델(draft)로 추측, 큰 모델(target)로 검증.
/// draft_engine과 target_engine은 같은 vocab을 공유해야 함.
pub fn generate_stream_two_model(
    draft_engine: &mut Engine,
    target_engine: &mut Engine,
    prompt: &str,
    params: &GenerateParams,
    mut callback: impl FnMut(&str) -> bool,
) -> crate::error::Result<GenerateResult> {
    let start = Instant::now();
    let spec_k = params.spec_k;

    let bos = target_engine.tokenizer.vocab.special.bos;
    let eos = target_engine.tokenizer.vocab.special.eos;
    let mut prompt_tokens = Vec::new();
    if target_engine.tokenizer.should_add_bos() {
        prompt_tokens.push(bos);
    }
    prompt_tokens.extend(target_engine.tokenizer.encode(prompt));
    let prompt_len = prompt_tokens.len();

    let mut rng = match params.seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    };
    let mut sampler = SamplerChain::from_params(params);

    // 두 엔진 모두 prefill
    draft_engine.clear_sequence_state()?;
    target_engine.clear_sequence_state()?;

    let profiling = crate::runtime::profiling_enabled();
    let t_prefill = Instant::now();
    let _ = draft_engine.forward_prompt(&prompt_tokens)?;
    if profiling {
        eprintln!(
            "  [SPEC2] draft prefill: {:.0}ms",
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );
    }
    let t_prefill = Instant::now();
    let mut logits = target_engine.forward_prompt(&prompt_tokens)?;
    if profiling {
        eprintln!(
            "  [SPEC2] target prefill: {:.0}ms",
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );
    }

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut generated_text = GeneratedTextStream::new();
    let mut tokens_remaining = params.max_tokens;

    let mut stats = SpecStats {
        total_rounds: 0,
        total_accepted: 0,
        total_drafted: 0,
        target_verify_steps: 0,
    };

    while tokens_remaining > 0 {
        crate::generate::check_generation_cancellation()?;
        // Sample first token from target logits
        let first_token = sampler.sample(&mut logits, &generated_tokens, &mut rng);
        if params.stop_tokens.contains(&first_token) || first_token == eos {
            break;
        }
        generated_tokens.push(first_token);
        if !generated_text.push(&target_engine.tokenizer, first_token, &mut callback) {
            break;
        }
        tokens_remaining -= 1;
        if tokens_remaining == 0 {
            break;
        }

        // --- Speculative round ---
        let k = spec_k.min(tokens_remaining);
        stats.total_rounds += 1;
        stats.total_drafted += k;

        // 1. Checkpoint target engine state
        let target_ckpt = SpecCheckpoint::save_engine(target_engine)?;

        // 2. Draft K tokens using DRAFT engine (full forward, not early-exit)
        let mut draft_tokens = Vec::with_capacity(k);
        let mut last_token = first_token;
        for _ in 0..k {
            crate::generate::check_generation_cancellation()?;
            let draft_logits = draft_engine.forward(&[last_token])?;
            let t = draft_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            draft_tokens.push(t);
            last_token = t;
        }

        // 3. Rollback target engine (draft engine keeps its KV — it advances normally)
        target_ckpt.restore_engine(target_engine)?;

        // 4. Verify: target prefill [first_token, d0, ..., d_{k-1}]
        let mut verify_input = Vec::with_capacity(1 + k);
        verify_input.push(first_token);
        verify_input.extend_from_slice(&draft_tokens);
        let all_logits = if target_verify_needs_sequential(target_engine) {
            target_engine.forward_verify_all_logits_sequential(&verify_input)?
        } else {
            target_engine.forward_prefill_all_logits(&verify_input)?
        };
        stats.add_target_verify_steps(all_logits.len());

        // 5. Accept/Reject
        let mut n_accepted = 0usize;
        let mut draft_matches = 0usize;
        let mut stopped = false;

        for i in 0..k {
            let mut verify_logits = all_logits[i].clone();
            let target_token = sampler.sample(&mut verify_logits, &generated_tokens, &mut rng);

            if target_token == draft_tokens[i] {
                if params.stop_tokens.contains(&draft_tokens[i]) || draft_tokens[i] == eos {
                    stopped = true;
                    break;
                }
                generated_tokens.push(draft_tokens[i]);
                if !generated_text.push(&target_engine.tokenizer, draft_tokens[i], &mut callback) {
                    stopped = true;
                    n_accepted += 1;
                    draft_matches += 1;
                    tokens_remaining -= 1;
                    break;
                }
                n_accepted += 1;
                draft_matches += 1;
                tokens_remaining -= 1;
                if tokens_remaining == 0 {
                    break;
                }
            } else {
                break;
            }
        }

        stats.total_accepted += draft_matches;

        let state_plan = verify_state_plan_after_spec_round(
            target_verify_needs_sequential(target_engine),
            spec_verify_mode(target_verify_needs_sequential(target_engine), false),
            k,
            n_accepted,
        );
        if state_plan.replay_committed_prefix {
            logits = replay_committed_verify_prefix(
                target_engine,
                &target_ckpt,
                &verify_input,
                state_plan.committed_verify_tokens,
            )?;
        } else {
            let final_len = target_ckpt.kv_len + state_plan.committed_verify_tokens;
            target_engine
                .kv_cache
                .set_len(final_len.min(target_engine.kv_cache.max_seq_len));
            logits = all_logits[state_plan.committed_verify_tokens - 1].clone();
        }

        // Draft engine KV를 target과 동기화 (accepted 위치까지만 유효)
        let draft_final_len = target_ckpt.kv_len + state_plan.committed_verify_tokens;
        draft_engine
            .kv_cache
            .set_len(draft_final_len.min(draft_engine.kv_cache.max_seq_len));
        if stopped {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    if profiling && stats.total_rounds > 0 {
        eprintln!("{}", stats.report().replacen("[SPEC]", "[SPEC2]", 1));
    }

    Ok(GenerateResult::new(
        generated_text.finish(&mut callback),
        generated_tokens.len(),
        prompt_len,
        elapsed,
        generated_tokens,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KVCache;

    #[test]
    fn test_checkpoint_save_restore_kv_len() {
        let mut cache = KVCache::new(2, 64, 2, 4);
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(0, 0, &k, &v);
        cache.append(1, 0, &k, &v);
        assert_eq!(cache.current_len(), 1);

        let ckpt = SpecCheckpoint::save(&cache);
        assert_eq!(ckpt.kv_len, 1);

        // Draft simulation: add more KV
        cache.append(0, 1, &k, &v);
        cache.append(1, 1, &k, &v);
        cache.set_len(2);
        assert_eq!(cache.current_len(), 2);

        // Restore
        ckpt.restore(&mut cache);
        assert_eq!(cache.current_len(), 1);
        assert_eq!(ckpt.kv_len, 1); // still accessible after restore
    }

    #[test]
    fn test_checkpoint_save_restore_ssm_state() {
        let mut cache = KVCache::new(2, 64, 2, 4);
        cache.init_ssm_state(0, 3, 16, 4, 8, 8);

        if let Some(ssm) = cache.get_ssm_state_mut(0) {
            ssm.conv_state[0] = 42.0;
            ssm.delta_state[0] = 99.0;
        }

        let ckpt = SpecCheckpoint::save(&cache);

        // Draft simulation: modify SSM state
        if let Some(ssm) = cache.get_ssm_state_mut(0) {
            ssm.conv_state[0] = 0.0;
            ssm.delta_state[0] = 0.0;
        }

        // Restore
        ckpt.restore(&mut cache);
        let ssm = cache.get_ssm_state(0).unwrap();
        assert_eq!(ssm.conv_state[0], 42.0);
        assert_eq!(ssm.delta_state[0], 99.0);
    }

    #[test]
    fn engine_checkpoint_restore_preserves_ssm_delta_allocation() {
        let mut engine = make_spec_test_engine(9);
        engine.kv_cache.init_ssm_state(0, 3, 4, 1, 2, 2);

        let original_ptr = {
            let ssm = engine.kv_cache.get_ssm_state_mut(0).unwrap();
            ssm.conv_state[0] = 42.0;
            ssm.delta_state[0] = 99.0;
            ssm.delta_state.as_ptr()
        };
        let checkpoint = SpecCheckpoint::save_engine(&mut engine).unwrap();

        {
            let ssm = engine.kv_cache.get_ssm_state_mut(0).unwrap();
            ssm.conv_state[0] = 0.0;
            ssm.delta_state[0] = 0.0;
        }

        checkpoint.restore_engine(&mut engine).unwrap();

        let ssm = engine.kv_cache.get_ssm_state(0).unwrap();
        assert_eq!(ssm.conv_state[0], 42.0);
        assert_eq!(ssm.delta_state[0], 99.0);
        assert_eq!(ssm.delta_state.as_ptr(), original_ptr);
    }

    #[test]
    fn ssm_checkpoint_omits_delta_host_copy_for_resident_layers() {
        let states = vec![Some(SsmLayerState {
            conv_state: vec![1.0, 2.0],
            delta_state: vec![3.0, 4.0, 5.0, 6.0],
            conv_kernel: 3,
            conv_channels: 1,
        })];

        let checkpoint = checkpoint_ssm_states(&states, &[0]);

        let layer = checkpoint[0].as_ref().unwrap();
        assert_eq!(layer.conv_state, vec![1.0, 2.0]);
        assert!(layer.delta_state.is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn engine_checkpoint_restore_keeps_cuda_resident_delta_snapshot() {
        let _ = crate::engine::cuda_runtime::reset_state_for_engine_init();
        let mut engine = make_spec_test_engine(9);
        let num_heads = 2usize;
        let head_k_dim = 128usize;
        let head_v_dim = 4usize;
        engine
            .kv_cache
            .init_ssm_state(0, 3, 4, num_heads, head_v_dim, head_k_dim);
        let state = engine.kv_cache.get_ssm_state_mut(0).unwrap();
        for (i, value) in state.delta_state.iter_mut().enumerate() {
            *value = ((i as f32 % 23.0) - 11.0) * 0.00390625;
        }

        let mut expected_state = state.delta_state.clone();
        let mut q = vec![0.0f32; num_heads * head_k_dim];
        let mut k = vec![0.0f32; num_heads * head_k_dim];
        let mut v = vec![0.0f32; num_heads * head_v_dim];
        for (i, value) in q.iter_mut().enumerate() {
            *value = ((i as f32 % 17.0) - 8.0) * 0.015625;
        }
        for (i, value) in k.iter_mut().enumerate() {
            *value = ((i as f32 % 19.0) - 9.0) * 0.01171875;
        }
        for (i, value) in v.iter_mut().enumerate() {
            *value = ((i as f32 % 13.0) - 6.0) * 0.03125;
        }
        let gate = vec![-0.03125f32, -0.0625f32];
        let beta = vec![0.25f32, 0.5f32];
        let mut expected_out = vec![0.0f32; num_heads * head_v_dim];
        crate::engine::cpu_runtime::kernels::delta_net::delta_net_scan_into(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut expected_state,
            &mut expected_out,
            1,
            num_heads,
            head_k_dim,
            head_v_dim,
        );

        let first = crate::engine::cuda_runtime::try_delta_step_resident_if_supported(
            &mut engine.kv_cache.get_ssm_state_mut(0).unwrap().delta_state,
            &q,
            &k,
            &v,
            &gate,
            &beta,
            num_heads,
            head_k_dim,
            head_v_dim,
        );
        let Some(first) = first else {
            return;
        };
        if let Err(err) = first {
            eprintln!("skipping CUDA resident checkpoint test: {err}");
            return;
        }

        let checkpoint = SpecCheckpoint::save_engine(&mut engine).unwrap();
        crate::engine::cuda_runtime::try_delta_step_resident_if_supported(
            &mut engine.kv_cache.get_ssm_state_mut(0).unwrap().delta_state,
            &q,
            &k,
            &v,
            &gate,
            &beta,
            num_heads,
            head_k_dim,
            head_v_dim,
        )
        .expect("CUDA resident delta should be enabled")
        .expect("second resident delta call");

        checkpoint.restore_engine(&mut engine).unwrap();
        let restored = crate::engine::cuda_runtime::sync_delta_state_cache(
            &mut engine.kv_cache.get_ssm_state_mut(0).unwrap().delta_state,
        )
        .unwrap();
        assert!(restored, "restore should keep CUDA resident delta state");

        let state = engine.kv_cache.get_ssm_state(0).unwrap();
        for (i, (actual, expected)) in state.delta_state.iter().zip(&expected_state).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.001,
                "restored engine delta state {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
        }
    }

    #[test]
    fn stateful_target_keeps_incremental_state_after_partial_reject() {
        let plan =
            verify_state_plan_after_spec_round(true, SpecVerifyMode::SequentialIncremental, 2, 1);

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn decode_fast_window_candidate_uses_incremental_state_plan() {
        let plan =
            verify_state_plan_after_spec_round(true, SpecVerifyMode::SequentialIncremental, 2, 1);

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn stateful_target_keeps_verify_state_when_all_drafts_accept() {
        let plan =
            verify_state_plan_after_spec_round(true, SpecVerifyMode::SequentialIncremental, 2, 2);

        assert_eq!(plan.committed_verify_tokens, 3);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn attention_only_target_can_truncate_after_partial_reject() {
        let plan = verify_state_plan_after_spec_round(false, SpecVerifyMode::Batch, 2, 1);

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn spec_phase_timings_report_keeps_all_mtp_phase_labels() {
        let mut timings = SpecPhaseTimings::default();
        timings.add_checkpoint(1.0);
        timings.add_draft(2.0);
        timings.add_restore(3.0);
        timings.add_verify(4.0);
        timings.add_observe(5.0);
        timings.add_replay(6.0);

        let report = timings.report(2);

        assert!(report.contains("checkpoint=0.5ms/round"));
        assert!(report.contains("draft=1.0ms/round"));
        assert!(report.contains("restore=1.5ms/round"));
        assert!(report.contains("verify=2.0ms/round"));
        assert!(report.contains("observe=2.5ms/round"));
        assert!(report.contains("replay=3.0ms/round"));
    }

    #[test]
    fn spec_stats_report_includes_target_verify_steps() {
        let stats = SpecStats {
            total_rounds: 10,
            total_accepted: 9,
            total_drafted: 19,
            target_verify_steps: 19,
        };

        let report = stats.report();

        assert!(report.contains("rounds=10"));
        assert!(report.contains("accepted=9"));
        assert!(report.contains("target_verify_steps=19"));
    }

    #[test]
    fn spec_phase_report_can_be_enabled_without_forward_profile() {
        assert!(!spec_phase_report_enabled(false, false));
        assert!(spec_phase_report_enabled(true, false));
        assert!(spec_phase_report_enabled(false, true));
    }

    #[test]
    fn spec_argmax_verify_requires_plain_greedy_sampling() {
        let mut params = GenerateParams {
            temperature: 0.0,
            ..GenerateParams::default()
        };
        assert!(spec_argmax_verify_allowed(&params));

        params.repetition_penalty = 1.1;
        assert!(!spec_argmax_verify_allowed(&params));

        params.repetition_penalty = 1.0;
        params.presence_penalty = 0.1;
        assert!(!spec_argmax_verify_allowed(&params));

        params.presence_penalty = 0.0;
        params.frequency_penalty = 0.1;
        assert!(!spec_argmax_verify_allowed(&params));

        params.frequency_penalty = 0.0;
        params.temperature = 0.7;
        assert!(!spec_argmax_verify_allowed(&params));
    }

    #[test]
    fn mtp_draft_state_is_retained_only_when_generation_continues() {
        assert!(should_retain_mtp_draft_after_verify(true, false, 1));
        assert!(!should_retain_mtp_draft_after_verify(true, false, 0));
        assert!(!should_retain_mtp_draft_after_verify(true, true, 1));
        assert!(!should_retain_mtp_draft_after_verify(false, false, 1));
    }

    #[test]
    fn mtp_incremental_verify_uses_mtp_only_checkpoint_without_trace() {
        let scope = spec_checkpoint_scope(true, SpecVerifyMode::SequentialIncremental, false, None);

        assert_eq!(scope, SpecCheckpointScope::MtpOnly);
    }

    #[test]
    fn mtp_forced_batch_verify_uses_full_checkpoint() {
        let scope = spec_checkpoint_scope(true, SpecVerifyMode::Batch, false, None);

        assert_eq!(scope, SpecCheckpointScope::FullEngine);
    }

    #[test]
    fn mtp_batch_prefix_snapshot_k1_uses_mtp_only_checkpoint() {
        let scope = spec_checkpoint_scope(true, SpecVerifyMode::Batch, false, Some(1));

        assert_eq!(scope, SpecCheckpointScope::MtpOnly);
    }

    #[test]
    fn mtp_batch_prefix_snapshot_k2_keeps_full_checkpoint() {
        let scope = spec_checkpoint_scope(true, SpecVerifyMode::Batch, false, Some(2));

        assert_eq!(scope, SpecCheckpointScope::FullEngine);
    }

    #[test]
    fn mtp_batch_verify_restores_only_mtp_before_target_verify() {
        let restore = spec_pre_verify_restore(
            SpecCheckpointScope::FullEngine,
            true,
            SpecVerifyMode::Batch,
            false,
        );

        assert_eq!(restore, SpecPreVerifyRestore::MtpOnly);
    }

    #[test]
    fn mtp_batch_prefix_snapshot_restores_mtp_before_target_verify() {
        let restore = spec_pre_verify_restore(
            SpecCheckpointScope::MtpOnly,
            true,
            SpecVerifyMode::Batch,
            false,
        );

        assert_eq!(restore, SpecPreVerifyRestore::MtpOnly);
    }

    #[test]
    fn non_mtp_batch_verify_restores_full_engine_before_target_verify() {
        let restore = spec_pre_verify_restore(
            SpecCheckpointScope::FullEngine,
            false,
            SpecVerifyMode::Batch,
            false,
        );

        assert_eq!(restore, SpecPreVerifyRestore::FullEngine);
    }

    #[test]
    fn mtp_trace_keeps_full_pre_verify_restore() {
        let restore = spec_pre_verify_restore(
            SpecCheckpointScope::FullEngine,
            true,
            SpecVerifyMode::Batch,
            true,
        );

        assert_eq!(restore, SpecPreVerifyRestore::FullEngine);
    }

    #[test]
    fn mtp_incremental_verify_does_not_restore_target_before_verify() {
        let restore = spec_pre_verify_restore(
            SpecCheckpointScope::MtpOnly,
            true,
            SpecVerifyMode::SequentialIncremental,
            false,
        );

        assert_eq!(restore, SpecPreVerifyRestore::None);
    }

    #[test]
    fn batch_verify_greedy_uses_argmax_token_source() {
        assert_eq!(
            spec_batch_verify_target_source(true),
            SpecBatchVerifyTargetSource::ArgmaxTokens
        );
        assert_eq!(
            spec_batch_verify_target_source(false),
            SpecBatchVerifyTargetSource::Logits
        );
    }

    #[test]
    fn mtp_batch_argmax_defers_mtp_observe_until_commit() {
        assert_eq!(
            spec_batch_mtp_observe_mode(true, SpecBatchVerifyTargetSource::ArgmaxTokens),
            SpecBatchMtpObserveMode::DeferredUntilCommit
        );
        assert_eq!(
            spec_batch_mtp_observe_mode(false, SpecBatchVerifyTargetSource::ArgmaxTokens),
            SpecBatchMtpObserveMode::Immediate
        );
    }

    #[test]
    fn batch_no_bonus_verify_input_omits_last_draft_row() {
        assert_eq!(
            spec_batch_verify_input_tokens(10, &[20, 30], false),
            vec![10, 20]
        );
        assert_eq!(
            spec_batch_verify_input_tokens(10, &[20, 30], true),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn batch_no_bonus_prefix_snapshot_uses_penultimate_row() {
        assert_eq!(spec_batch_prefix_snapshot_k_for_verify_input(2, true), 1);
        assert_eq!(spec_batch_prefix_snapshot_k_for_verify_input(4, true), 3);
        assert_eq!(spec_batch_prefix_snapshot_k_for_verify_input(4, false), 4);
        assert_eq!(spec_batch_prefix_snapshot_k_for_verify_input(1, true), 0);
    }

    #[test]
    fn mtp_sequential_verify_caps_effective_k_to_one() {
        assert_eq!(
            effective_spec_k_for_round(4, true, SpecVerifyMode::SequentialIncremental, false),
            1
        );
        assert_eq!(
            effective_spec_k_for_round(4, true, SpecVerifyMode::Batch, false),
            4
        );
        assert_eq!(
            effective_spec_k_for_round(4, false, SpecVerifyMode::SequentialIncremental, false),
            4
        );
    }

    #[test]
    fn mtp_sequential_multi_verify_keeps_requested_k_when_enabled() {
        assert_eq!(
            effective_spec_k_for_round(4, true, SpecVerifyMode::SequentialIncremental, true),
            4
        );
    }

    #[test]
    fn qwen35_default_verify_window_mode_stays_sequential() {
        assert_eq!(
            spec_verify_mode(true, false),
            SpecVerifyMode::SequentialIncremental
        );
    }

    #[test]
    fn qwen35_dense_verify_window_mode_stays_sequential() {
        assert!(architecture_verify_needs_sequential(
            rnb_loader::Architecture::Qwen35
        ));
    }

    #[test]
    fn stateful_batch_verify_replays_committed_prefix() {
        let plan = verify_state_plan_after_spec_round(true, SpecVerifyMode::Batch, 4, 1);

        assert_eq!(plan.committed_verify_tokens, 2);
        assert!(plan.replay_committed_prefix);
    }

    #[test]
    fn stateful_batch_verify_uses_prefix_snapshot_instead_of_replay() {
        let plan = verify_state_plan_after_spec_round_with_prefix_snapshot(
            true,
            SpecVerifyMode::Batch,
            1,
            0,
            true,
        );

        assert_eq!(plan.committed_verify_tokens, 1);
        assert!(!plan.replay_committed_prefix);
    }

    #[test]
    fn mtp_only_batch_prefix_restore_resets_mtp_before_observing_prefix() {
        assert!(should_restore_mtp_before_prefix_observe(
            SpecCheckpointScope::MtpOnly,
            true,
            true,
            SpecVerifyMode::Batch,
            true,
        ));
    }

    #[test]
    fn batch_prefix_snapshot_k2_captures_two_committed_tokens() {
        assert_eq!(
            spec_batch_prefix_snapshot_tokens_with_capability(
                true,
                true,
                true,
                SpecVerifyMode::Batch,
                true,
                2,
                false,
                true,
            ),
            Some(2)
        );
    }

    #[test]
    fn batch_prefix_snapshot_requires_prefix_restore_support() {
        assert_eq!(
            spec_batch_prefix_snapshot_tokens_with_capability(
                true,
                true,
                true,
                SpecVerifyMode::Batch,
                true,
                1,
                false,
                false,
            ),
            None
        );
    }

    #[test]
    fn stateful_batch_verify_keeps_prefill_state_when_all_drafts_accept() {
        let plan = verify_state_plan_after_spec_round(true, SpecVerifyMode::Batch, 4, 4);

        assert_eq!(plan.committed_verify_tokens, 5);
        assert!(!plan.replay_committed_prefix);
    }

    // --- Speculative generate loop tests ---
    use crate::engine::{Engine, ModelMetadata};
    use crate::generate::GenerateParams;
    use crate::tokenizer::{
        bpe::Tokenizer as BpeTokenizer,
        vocab::{SpecialTokens, Vocab},
    };

    fn make_spec_test_engine(eos_id: u32) -> Engine {
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
            num_layers: 4,
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
    fn test_speculative_generate_respects_max_tokens() {
        // Mock engine: has_weights() == false → fallback to standard generate
        // spec_enabled=true but mock → standard path, verifies wiring works
        let mut engine = make_spec_test_engine(9);
        let params = GenerateParams {
            max_tokens: 5,
            temperature: 0.0,
            spec_enabled: true,
            spec_k: 4,
            spec_depth: 0.5,
            seed: Some(0),
            ..GenerateParams::default()
        };
        let result = engine.generate("prompt", &params).unwrap();
        assert_eq!(result.tokens_generated, 5);
    }

    #[test]
    fn test_speculative_generate_stops_on_eos() {
        // eos=15 = greedy argmax on zero logits → 즉시 EOS
        let mut engine = make_spec_test_engine(15);
        let params = GenerateParams {
            max_tokens: 100,
            temperature: 0.0,
            spec_enabled: true,
            spec_k: 4,
            spec_depth: 0.5,
            seed: Some(0),
            ..GenerateParams::default()
        };
        let result = engine.generate("x", &params).unwrap();
        assert_eq!(result.tokens_generated, 0);
    }
}
