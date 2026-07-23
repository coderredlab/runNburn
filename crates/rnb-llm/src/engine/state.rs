use super::backend_runtime;
use super::layer_weights::ModelWeights;
use super::mtp::{EngineMtpRuntime, EngineMtpSequenceState, EngineMtpState};
use super::types::{ModelMetadata, ScratchBuffers};
use crate::kv_cache::{KVCache, KVCacheSnapshot, KvCacheMetrics};
use crate::tokenizer::Tokenizer;
use rnb_loader::Architecture as ModelArchitecture;

#[derive(Clone)]
struct PromptResumeAlignment {
    prompt_prefix: String,
    append_text: String,
}

#[derive(Clone)]
pub struct EngineSequenceState {
    token_ids: Vec<u32>,
    cached_token_len: usize,
    prompt_alignment: Option<PromptResumeAlignment>,
    kv_cache: KVCacheSnapshot,
    mtp: Option<EngineMtpSequenceState>,
}

impl EngineSequenceState {
    pub fn token_len(&self) -> usize {
        self.cached_token_len
    }

    pub fn byte_size(&self) -> u64 {
        (std::mem::size_of::<Self>() as u64)
            .saturating_add(self.kv_cache.byte_size())
            .saturating_add(
                self.mtp
                    .as_ref()
                    .map_or(0, EngineMtpSequenceState::heap_byte_size),
            )
            .saturating_add(
                self.token_ids
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()) as u64,
            )
            .saturating_add(self.prompt_alignment.as_ref().map_or(0, |alignment| {
                alignment.prompt_prefix.capacity() as u64 + alignment.append_text.capacity() as u64
            }))
    }

    pub fn matches_prompt_prefix(&self, prompt_tokens: &[u32]) -> bool {
        prompt_tokens.starts_with(&self.token_ids)
    }

    pub(crate) fn resume_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub(crate) fn prompt_resume_alignment(&self) -> Option<(&str, &str)> {
        self.prompt_alignment.as_ref().map(|alignment| {
            (
                alignment.prompt_prefix.as_str(),
                alignment.append_text.as_str(),
            )
        })
    }
}

pub struct Engine {
    pub tokenizer: Tokenizer,
    pub kv_cache: KVCache,
    pub metadata: ModelMetadata,
    pub(super) architecture: ModelArchitecture,
    pub(super) host_memory_plan: crate::engine::policy::HostMemoryPlan,
    pub(super) weights: Option<ModelWeights>,
    pub(super) scratch: Option<ScratchBuffers>,
    pub(super) mtp: Option<EngineMtpState>,
    pub(super) mtp_runtime: Option<EngineMtpRuntime>,
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(super) backend_runtime: backend_runtime::EngineBackendRuntime,
    /// Per-token monotonic counter used to tag `memtrace` step rows. Only
    /// meaningful when `RNB_MEMTRACE_FILE` is set; otherwise increments cheaply
    /// without side effects.
    pub(super) memtrace_step: std::sync::atomic::AtomicUsize,

    /// Legacy lazy binding flag for older full-table GPU `token_embd` paths.
    /// Production Vulkan fullpath stages compact Q6_K token rows instead of
    /// binding the whole embedding table as one storage buffer.
    ///
    /// Reset to `false` whenever weights are replaced (in practice: never after
    /// engine init, but safe to reset if `weights` is ever re-assigned).
    #[cfg(feature = "vulkan")]
    #[allow(dead_code)]
    pub(super) fullpath_token_embd_bound: bool,

    /// mt83 Stage B: last forward step 의 last decoder layer 의 normed hidden
    /// (output_norm 적용 직후, lm_head 입력과 같은 형태). prefill/decode
    /// 모두 forward 끝에서 갱신된다. Spec drafter 의 cross-attention 입력
    /// 으로 사용. 길이 = `metadata.hidden_dim`. forward 호출 전이면 빈 Vec.
    pub(super) last_layer_hidden_cached: Vec<f32>,

    /// pm118: `RNB_MTP` 미설정/auto 일 때의 auto policy `enabled` 판정 캐시.
    /// CUDA 에선 policy 가 free VRAM 을 참조해 런 중 판정이 흔들릴 수 있어
    /// 엔진당 1회로 고정한다 (policy 입력은 로드 후 불변).
    pub(super) mtp_auto_requested_cache: std::sync::OnceLock<bool>,
}

impl Engine {
    pub const fn host_memory_plan(&self) -> crate::engine::policy::HostMemoryPlan {
        self.host_memory_plan
    }

    pub fn kv_cache_metrics(&self) -> KvCacheMetrics {
        self.kv_cache.metrics()
    }

    pub fn sequence_state_byte_size_estimate(&self) -> u64 {
        let mtp_bytes = if self.mtp_spec_requested() {
            self.mtp_runtime.as_ref().map_or(0, |runtime| {
                runtime.sequence_state_heap_byte_size_estimate()
            })
        } else {
            0
        };
        (std::mem::size_of::<EngineSequenceState>() as u64)
            .saturating_add(self.kv_cache.snapshot_byte_size())
            .saturating_add(mtp_bytes)
            .saturating_add(
                self.kv_cache
                    .current_len()
                    .saturating_mul(std::mem::size_of::<u32>()) as u64,
            )
    }

    pub fn durable_sequence_state_supported(&self) -> bool {
        if cfg!(any(
            feature = "vulkan",
            feature = "opencl",
            feature = "mediatek"
        )) {
            return false;
        }
        if !self.mtp_spec_requested() {
            return true;
        }
        if crate::runtime::mtp_draft_only_enabled()
            || crate::runtime::mtp_shadow_precompute_enabled()
        {
            return false;
        }
        self.mtp_runtime
            .as_ref()
            .is_some_and(EngineMtpRuntime::durable_sequence_state_supported)
    }

    pub fn capture_sequence_state(
        &mut self,
        token_ids: Vec<u32>,
    ) -> crate::error::Result<EngineSequenceState> {
        self.capture_sequence_state_inner(token_ids, None)
    }

    pub fn capture_sequence_state_with_prompt_alignment(
        &mut self,
        token_ids: Vec<u32>,
        prompt_prefix: String,
        append_text: String,
    ) -> crate::error::Result<EngineSequenceState> {
        self.capture_sequence_state_inner(
            token_ids,
            Some(PromptResumeAlignment {
                prompt_prefix,
                append_text,
            }),
        )
    }

    fn capture_sequence_state_inner(
        &mut self,
        token_ids: Vec<u32>,
        prompt_alignment: Option<PromptResumeAlignment>,
    ) -> crate::error::Result<EngineSequenceState> {
        if !self.durable_sequence_state_supported() {
            return Err(crate::error::LlmError::Unsupported(
                "durable sequence snapshots are unsupported by the active runtime".to_string(),
            ));
        }
        self.materialize_sequence_state()?;
        let cached_len = self.kv_cache.current_len();
        if cached_len == 0 || cached_len > token_ids.len() {
            return Err(crate::error::LlmError::Forward(format!(
                "cannot capture sequence state: {} tokens for KV length {cached_len}",
                token_ids.len()
            )));
        }
        let mtp = if self.mtp_spec_requested() {
            Some(
                self.mtp_runtime
                    .as_ref()
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "cannot capture MTP sequence state without an active runtime"
                                .to_string(),
                        )
                    })?
                    .capture_sequence_state(),
            )
        } else {
            None
        };
        Ok(EngineSequenceState {
            token_ids,
            cached_token_len: cached_len,
            prompt_alignment,
            kv_cache: self.kv_cache.snapshot(),
            mtp,
        })
    }

    pub fn restore_sequence_state(
        &mut self,
        state: &EngineSequenceState,
    ) -> crate::error::Result<()> {
        if !self.durable_sequence_state_supported() {
            return Err(crate::error::LlmError::Unsupported(
                "durable sequence snapshots are unsupported by the active runtime".to_string(),
            ));
        }
        let mtp_requested = self.mtp_spec_requested();
        if mtp_requested != state.mtp.is_some() {
            return Err(crate::error::LlmError::Forward(
                "sequence snapshot MTP mode does not match the active runtime".to_string(),
            ));
        }
        self.clear_sequence_state()?;
        self.kv_cache
            .restore_snapshot(&state.kv_cache)
            .map_err(crate::error::LlmError::Forward)?;
        if let Some(mtp_state) = state.mtp.as_ref() {
            self.mtp_runtime
                .as_mut()
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "cannot restore MTP sequence state without an active runtime".to_string(),
                    )
                })?
                .restore_sequence_state(mtp_state)?;
        }
        self.last_layer_hidden_cached.clear();
        Ok(())
    }
}
