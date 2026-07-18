use super::backend_runtime;
use super::layer_weights::ModelWeights;
use super::mtp::{EngineMtpRuntime, EngineMtpState};
use super::packed_runtime;
use super::types::{ModelMetadata, ScratchBuffers};
use crate::kv_cache::{CompactKVCacheSnapshot, KVCache};
use crate::tokenizer::Tokenizer;
use rnb_loader::Architecture as ModelArchitecture;
use std::sync::Arc;

#[derive(Clone)]
pub struct EngineSequenceState {
    token_ids: Vec<u32>,
    kv_cache: CompactKVCacheSnapshot,
}

impl EngineSequenceState {
    pub fn token_len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn byte_size(&self) -> u64 {
        (std::mem::size_of::<Self>() as u64)
            .saturating_add(self.kv_cache.byte_size())
            .saturating_add(
                self.token_ids
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()) as u64,
            )
    }

    pub fn matches_prompt_prefix(&self, prompt_tokens: &[u32]) -> bool {
        prompt_tokens.starts_with(&self.token_ids)
    }
}

pub struct Engine {
    pub tokenizer: Tokenizer,
    pub kv_cache: KVCache,
    pub metadata: ModelMetadata,
    pub(super) architecture: ModelArchitecture,
    pub(super) host_memory_plan: rnb_runtime::policy::HostMemoryPlan,
    pub(super) weights: Option<ModelWeights>,
    pub(super) scratch: Option<ScratchBuffers>,
    pub(super) mtp: Option<EngineMtpState>,
    pub(super) mtp_runtime: Option<EngineMtpRuntime>,
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(super) backend_runtime: backend_runtime::EngineBackendRuntime,
    /// .rnb packed weights (mmap). QuantizedWeight의 packed_gemm_data가 이 mmap을 참조하므로
    /// Engine보다 먼저 drop되면 안 됨 — Engine struct 안에 함께 소유.
    ///
    /// `Arc` 로 공유하는 이유: Session 64 axis B 이후 `MoeLayerWeights` 가
    /// 같은 PackedModel 을 참조해 MoE tensor bytes 를 .rnb 에서 가져가므로,
    /// dense 경로와 MoE 경로가 같은 mmap 수명을 공유한다.
    #[allow(dead_code)]
    pub(super) packed_model: Option<Arc<packed_runtime::PackedModel>>,
    /// Diagnostic Q2_K shadow expert bytes (`<stem>.shadow.rnb`,
    /// `QuantType::RawQ2K`). Populated only when `RNB_SHADOW_WEIGHTS=1`.
    #[allow(dead_code)]
    pub(super) shadow_model: Option<Arc<packed_runtime::PackedModel>>,
    /// Per-token monotonic counter used to tag `memtrace` step rows. Only
    /// meaningful when `RNB_MEMTRACE_FILE` is set; otherwise increments cheaply
    /// without side effects.
    pub(super) memtrace_step: std::sync::atomic::AtomicUsize,

    /// Retained `.rnb` MoE section bytes. Held at engine scope so the
    /// `(offset, len)` pairs inside each layer's
    /// `SharedExpertMoELayerWeights::moe_section_decode` remain valid for the
    /// engine lifetime. `None` when no MoE section was loaded.
    #[allow(dead_code)]
    pub(super) moe_section_decode_bytes: Option<Arc<memmap2::Mmap>>,

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
}

impl Engine {
    pub const fn host_memory_plan(&self) -> rnb_runtime::policy::HostMemoryPlan {
        self.host_memory_plan
    }

    pub fn sequence_state_byte_size_estimate(&self) -> u64 {
        (std::mem::size_of::<EngineSequenceState>() as u64)
            .saturating_add(self.kv_cache.compact_snapshot_byte_size())
            .saturating_add(
                self.kv_cache
                    .current_len()
                    .saturating_mul(std::mem::size_of::<u32>()) as u64,
            )
    }

    pub fn durable_sequence_state_supported(&self) -> bool {
        !self.mtp_spec_requested()
            && !cfg!(any(
                feature = "cuda",
                feature = "vulkan",
                feature = "metal",
                feature = "opencl",
                feature = "mediatek"
            ))
    }

    pub fn capture_sequence_state(
        &mut self,
        mut token_ids: Vec<u32>,
    ) -> crate::error::Result<EngineSequenceState> {
        if !self.durable_sequence_state_supported() {
            return Err(crate::error::LlmError::Unsupported(
                "durable sequence snapshots require the CPU runtime".to_string(),
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
        token_ids.truncate(cached_len);
        token_ids.shrink_to_fit();
        Ok(EngineSequenceState {
            token_ids,
            kv_cache: self.kv_cache.compact_snapshot(),
        })
    }

    pub fn restore_sequence_state(
        &mut self,
        state: &EngineSequenceState,
    ) -> crate::error::Result<()> {
        if !self.durable_sequence_state_supported() {
            return Err(crate::error::LlmError::Unsupported(
                "durable sequence snapshots require the CPU runtime".to_string(),
            ));
        }
        self.clear_sequence_state()?;
        self.kv_cache
            .restore_compact(&state.kv_cache)
            .map_err(crate::error::LlmError::Forward)?;
        self.last_layer_hidden_cached.clear();
        Ok(())
    }
}
