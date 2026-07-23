#[cfg(feature = "cuda")]
use super::backend_runtime;
use super::*;
use crate::error::{LlmError, Result};
use crate::kv_cache::{KVCache, KVCacheSnapshot};
use rnb_core::tensor::Tensor;
use rnb_loader::{MtpLayerTensors, MtpMetadata};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpAutoPolicy {
    pub enabled: bool,
    pub spec_k: usize,
    pub device_verify: bool,
    pub min_free_vram_mib: usize,
    pub resource: Option<MtpAutoResourceHint>,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpAutoResourceHint {
    pub total_vram_mib: usize,
    pub free_vram_mib: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngineMtpState {
    pub(crate) total_block_count: usize,
    pub(crate) trunk_layers: usize,
    pub(crate) first_mtp_layer: usize,
    pub(crate) nextn_predict_layers: usize,
    pub(crate) layer_tensors: Vec<MtpLayerTensors>,
}

impl EngineMtpState {
    pub(crate) fn from_loader_parts(
        mtp: Option<&MtpMetadata>,
        layer_tensors: &[MtpLayerTensors],
    ) -> Result<Option<Self>> {
        let Some(mtp) = mtp else {
            if layer_tensors.is_empty() {
                return Ok(None);
            }
            return Err(LlmError::ModelLoad(
                "MTP tensors were present without MTP metadata".to_string(),
            ));
        };

        if layer_tensors.len() != mtp.nextn_predict_layers {
            return Err(LlmError::ModelLoad(format!(
                "MTP tensor layer count {} != nextn_predict_layers {}",
                layer_tensors.len(),
                mtp.nextn_predict_layers
            )));
        }

        Ok(Some(Self {
            total_block_count: mtp.total_block_count,
            trunk_layers: mtp.trunk_layers,
            first_mtp_layer: mtp.first_mtp_layer,
            nextn_predict_layers: mtp.nextn_predict_layers,
            layer_tensors: layer_tensors.to_vec(),
        }))
    }
}

pub(crate) struct InModelMtpRuntime {
    metadata: ModelMetadata,
    weights: MtpLayerWeights,
    kv_cache: KVCache,
    scratch: ScratchBuffers,
    next_pos: usize,
    last_hidden: Option<Vec<f32>>,
    recent_hidden_rows: Vec<f32>,
    recent_hidden_count: usize,
}

#[derive(Clone)]
pub(crate) struct InModelMtpCheckpoint {
    kv_len: usize,
    next_pos: usize,
    last_hidden: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// EngineMtpRuntime enum — InModel (nextn) 또는 External drafter 중 하나.
// ---------------------------------------------------------------------------

pub(crate) enum EngineMtpRuntime {
    InModel(InModelMtpRuntime),
    External(crate::external_drafter::ExternalDrafterRuntime),
}

#[derive(Clone)]
pub(crate) enum EngineMtpCheckpoint {
    InModel(InModelMtpCheckpoint),
    /// External drafter 는 decode 경계 간 stateless — checkpoint 불필요.
    External,
}

#[derive(Clone)]
pub(crate) enum EngineMtpSequenceState {
    InModel(InModelMtpSequenceState),
    External(crate::external_drafter::ExternalDrafterSequenceState),
}

#[derive(Clone)]
pub(crate) struct InModelMtpSequenceState {
    kv_cache: KVCacheSnapshot,
    next_pos: usize,
    last_hidden: Option<Vec<f32>>,
    recent_hidden_rows: Vec<f32>,
    recent_hidden_count: usize,
}

impl EngineMtpSequenceState {
    pub(crate) fn heap_byte_size(&self) -> u64 {
        match self {
            Self::InModel(state) => state.heap_byte_size(),
            Self::External(state) => state.heap_byte_size(),
        }
    }
}

impl InModelMtpSequenceState {
    fn heap_byte_size(&self) -> u64 {
        self.kv_cache
            .byte_size()
            .saturating_sub(std::mem::size_of::<KVCacheSnapshot>() as u64)
            .saturating_add(self.last_hidden.as_ref().map_or(0, |hidden| {
                hidden.capacity().saturating_mul(std::mem::size_of::<f32>())
            }) as u64)
            .saturating_add(
                self.recent_hidden_rows
                    .capacity()
                    .saturating_mul(std::mem::size_of::<f32>()) as u64,
            )
    }
}

fn in_model_sequence_state_heap_byte_size_estimate(
    kv_cache: &KVCache,
    last_hidden: &Option<Vec<f32>>,
    recent_hidden_rows: &[f32],
) -> u64 {
    kv_cache
        .snapshot_byte_size()
        .saturating_sub(std::mem::size_of::<KVCacheSnapshot>() as u64)
        .saturating_add(last_hidden.as_ref().map_or(0, |hidden| {
            hidden.len().saturating_mul(std::mem::size_of::<f32>())
        }) as u64)
        .saturating_add(
            recent_hidden_rows
                .len()
                .saturating_mul(std::mem::size_of::<f32>()) as u64,
        )
}

impl EngineMtpRuntime {
    pub(crate) fn checkpoint(&self) -> EngineMtpCheckpoint {
        match self {
            EngineMtpRuntime::InModel(r) => EngineMtpCheckpoint::InModel(r.checkpoint()),
            EngineMtpRuntime::External(_) => EngineMtpCheckpoint::External,
        }
    }

    pub(crate) fn restore(&mut self, ckpt: &EngineMtpCheckpoint) {
        match (self, ckpt) {
            (EngineMtpRuntime::InModel(r), EngineMtpCheckpoint::InModel(c)) => r.restore(c),
            (EngineMtpRuntime::External(_), EngineMtpCheckpoint::External) => {}
            _ => panic!("EngineMtpRuntime / EngineMtpCheckpoint variant mismatch"),
        }
    }

    pub(crate) fn durable_sequence_state_supported(&self) -> bool {
        match self {
            EngineMtpRuntime::InModel(runtime) => {
                runtime.kv_cache.ssm_states.iter().all(Option::is_none)
            }
            EngineMtpRuntime::External(_) => true,
        }
    }

    pub(crate) fn sequence_state_heap_byte_size_estimate(&self) -> u64 {
        match self {
            EngineMtpRuntime::InModel(runtime) => in_model_sequence_state_heap_byte_size_estimate(
                &runtime.kv_cache,
                &runtime.last_hidden,
                &runtime.recent_hidden_rows,
            ),
            EngineMtpRuntime::External(runtime) => runtime.sequence_state_heap_byte_size_estimate(),
        }
    }

    pub(crate) fn capture_sequence_state(&self) -> EngineMtpSequenceState {
        match self {
            EngineMtpRuntime::InModel(runtime) => {
                EngineMtpSequenceState::InModel(runtime.capture_sequence_state())
            }
            EngineMtpRuntime::External(runtime) => {
                EngineMtpSequenceState::External(runtime.capture_sequence_state())
            }
        }
    }

    pub(crate) fn restore_sequence_state(&mut self, state: &EngineMtpSequenceState) -> Result<()> {
        match (self, state) {
            (EngineMtpRuntime::InModel(runtime), EngineMtpSequenceState::InModel(state)) => {
                runtime.restore_sequence_state(state)
            }
            (EngineMtpRuntime::External(runtime), EngineMtpSequenceState::External(state)) => {
                runtime
                    .restore_sequence_state(state)
                    .map_err(LlmError::Forward)
            }
            _ => Err(LlmError::Forward(
                "MTP sequence snapshot runtime variant mismatch".to_string(),
            )),
        }
    }

    pub(crate) fn clear_sequence_state(&mut self) {
        match self {
            EngineMtpRuntime::InModel(r) => r.clear_sequence_state(),
            EngineMtpRuntime::External(r) => r.shift_for_accept(0),
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn in_model_weights(&self) -> Option<&MtpLayerWeights> {
        match self {
            EngineMtpRuntime::InModel(runtime) => Some(&runtime.weights),
            EngineMtpRuntime::External(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn shared_expert_moe(
        &self,
    ) -> Option<&super::layer_weights::SharedExpertMoELayerWeights> {
        match self {
            EngineMtpRuntime::InModel(runtime) => runtime.weights.block.shared_expert_moe.as_ref(),
            EngineMtpRuntime::External(_) => None,
        }
    }

    pub(crate) fn as_in_model_mut(&mut self) -> Option<&mut InModelMtpRuntime> {
        if let EngineMtpRuntime::InModel(r) = self {
            Some(r)
        } else {
            None
        }
    }

    pub(crate) fn as_external_mut(
        &mut self,
    ) -> Option<&mut crate::external_drafter::ExternalDrafterRuntime> {
        if let EngineMtpRuntime::External(r) = self {
            Some(r)
        } else {
            None
        }
    }

    /// External variant 여부. 두 variant 모두 "ready" 판정에 활용.
    pub(crate) fn is_external(&self) -> bool {
        matches!(self, EngineMtpRuntime::External(_))
    }
}

impl InModelMtpRuntime {
    pub(super) fn new(base_metadata: &ModelMetadata, weights: MtpLayerWeights) -> Self {
        let mut metadata = base_metadata.clone();
        metadata.num_layers = 1;
        metadata.full_attention_interval = 1;
        metadata.head_count_kv_per_layer = None;
        metadata.sliding_window_pattern.clear();

        let layout = resolve_attention_layout(&metadata, &weights.block, None).ok();
        let num_kv_heads = layout
            .map(|l| l.num_kv_heads)
            .unwrap_or(metadata.num_kv_heads);
        let head_dim = layout.map(|l| l.head_dim).unwrap_or(metadata.head_dim);
        let kv_cache = KVCache::new_per_layer(metadata.max_seq_len, &[num_kv_heads], &[head_dim]);
        let ffn_inner_dim = mtp_block_ffn_inner_dim(&weights);
        let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);

        Self {
            metadata,
            weights,
            kv_cache,
            scratch,
            next_pos: 0,
            last_hidden: None,
            recent_hidden_rows: Vec::new(),
            recent_hidden_count: 0,
        }
    }

    fn clear_sequence_state(&mut self) {
        self.kv_cache.clear();
        self.next_pos = 0;
        self.last_hidden = None;
        self.recent_hidden_rows.clear();
        self.recent_hidden_count = 0;
    }

    fn checkpoint(&self) -> InModelMtpCheckpoint {
        InModelMtpCheckpoint {
            kv_len: self.kv_cache.current_len(),
            next_pos: self.next_pos,
            last_hidden: self.last_hidden.clone(),
        }
    }

    fn restore(&mut self, checkpoint: &InModelMtpCheckpoint) {
        self.kv_cache
            .set_len(checkpoint.kv_len.min(self.kv_cache.max_seq_len));
        self.next_pos = checkpoint.next_pos;
        self.last_hidden = checkpoint.last_hidden.clone();
        self.recent_hidden_rows.clear();
        self.recent_hidden_count = 0;
    }

    fn capture_sequence_state(&self) -> InModelMtpSequenceState {
        capture_in_model_sequence_state(
            &self.kv_cache,
            self.next_pos,
            &self.last_hidden,
            &self.recent_hidden_rows,
            self.recent_hidden_count,
        )
    }

    fn restore_sequence_state(&mut self, state: &InModelMtpSequenceState) -> Result<()> {
        restore_in_model_sequence_state(
            &mut self.kv_cache,
            &mut self.next_pos,
            &mut self.last_hidden,
            &mut self.recent_hidden_rows,
            &mut self.recent_hidden_count,
            self.metadata.hidden_dim,
            state,
        )
    }
}

fn capture_in_model_sequence_state(
    kv_cache: &KVCache,
    next_pos: usize,
    last_hidden: &Option<Vec<f32>>,
    recent_hidden_rows: &[f32],
    recent_hidden_count: usize,
) -> InModelMtpSequenceState {
    InModelMtpSequenceState {
        kv_cache: kv_cache.snapshot(),
        next_pos,
        last_hidden: last_hidden.clone(),
        recent_hidden_rows: recent_hidden_rows.to_vec(),
        recent_hidden_count,
    }
}

#[allow(clippy::too_many_arguments)]
fn restore_in_model_sequence_state(
    kv_cache: &mut KVCache,
    next_pos: &mut usize,
    last_hidden: &mut Option<Vec<f32>>,
    recent_hidden_rows: &mut Vec<f32>,
    recent_hidden_count: &mut usize,
    hidden_dim: usize,
    state: &InModelMtpSequenceState,
) -> Result<()> {
    if state
        .last_hidden
        .as_ref()
        .is_some_and(|hidden| hidden.len() != hidden_dim)
    {
        return Err(LlmError::Forward(
            "MTP sequence snapshot last hidden length mismatch".to_string(),
        ));
    }
    let expected_recent_len = state
        .recent_hidden_count
        .checked_mul(hidden_dim)
        .ok_or_else(|| {
            LlmError::Forward("MTP sequence snapshot hidden length overflow".to_string())
        })?;
    if state.recent_hidden_rows.len() != expected_recent_len {
        return Err(LlmError::Forward(format!(
            "MTP sequence snapshot recent hidden length {} != {expected_recent_len}",
            state.recent_hidden_rows.len()
        )));
    }
    if state.next_pos > kv_cache.max_seq_len {
        return Err(LlmError::Forward(format!(
            "MTP sequence snapshot position {} exceeds max sequence length {}",
            state.next_pos, kv_cache.max_seq_len
        )));
    }
    kv_cache
        .restore_snapshot(&state.kv_cache)
        .map_err(LlmError::Forward)?;
    *next_pos = state.next_pos;
    last_hidden.clone_from(&state.last_hidden);
    recent_hidden_rows.clone_from(&state.recent_hidden_rows);
    *recent_hidden_count = state.recent_hidden_count;
    Ok(())
}

fn mtp_block_ffn_inner_dim(weights: &MtpLayerWeights) -> usize {
    weights
        .block
        .shared_expert_moe
        .as_ref()
        .map(|moe| moe.n_ff)
        .unwrap_or(weights.block.ffn_gate_weight.rows)
        .max(1)
}

fn mtp_block_cache_len_after(cache_pos_start: usize, seq_len: usize, max_seq_len: usize) -> usize {
    cache_pos_start.saturating_add(seq_len).min(max_seq_len)
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    if divisor == 0 {
        return value;
    }
    value.saturating_add(divisor - 1) / divisor
}

fn bytes_to_mib_ceil(bytes: usize) -> usize {
    ceil_div(bytes, 1024 * 1024)
}

fn align_mib(value: usize, align: usize) -> usize {
    ceil_div(value, align).saturating_mul(align)
}

fn q4k_like_matrix_mib(rows: usize, cols: usize) -> usize {
    // K-quants are slightly above 4-bit due scales/mins. 5/8 keeps the policy
    // conservative without baking in a specific GGUF tensor size.
    bytes_to_mib_ceil(rows.saturating_mul(cols).saturating_mul(5) / 8)
}
#[cfg(feature = "cuda")]
pub(in crate::engine) fn mtp_device_rope_neox(
    architecture: ModelArchitecture,
    has_text_mrope: bool,
    gemma_neox_enabled: bool,
) -> bool {
    has_text_mrope
        || matches!(architecture, ModelArchitecture::Hy3)
        || (matches!(architecture, ModelArchitecture::Gemma4) && gemma_neox_enabled)
}

fn mtp_dense_decode_work_units(metadata: &ModelMetadata) -> usize {
    metadata
        .hidden_dim
        .saturating_mul(metadata.num_layers.max(1))
}

fn mtp_dense_decode_work_threshold(metadata: &ModelMetadata) -> usize {
    metadata
        .hidden_dim
        .max(4096)
        .saturating_mul(32)
        .min(4096 * 40)
}

fn mtp_device_verify_min_free_vram_mib(metadata: &ModelMetadata, spec_k: usize) -> usize {
    let window_tokens = spec_k.saturating_add(1).max(2);
    let token_embd_mib = q4k_like_matrix_mib(metadata.vocab_size, metadata.hidden_dim);
    let output_mib = q4k_like_matrix_mib(metadata.vocab_size, metadata.hidden_dim);
    let eh_proj_mib = bytes_to_mib_ceil(
        metadata
            .hidden_dim
            .saturating_mul(metadata.hidden_dim)
            .saturating_mul(4),
    );
    let row_workspace_mib = bytes_to_mib_ceil(
        metadata
            .hidden_dim
            .saturating_mul(window_tokens)
            .saturating_mul(4)
            .saturating_mul(16),
    )
    .max(256);
    let layer_workspace_mib = align_mib(
        bytes_to_mib_ceil(
            metadata
                .hidden_dim
                .saturating_mul(metadata.num_layers.max(1))
                .saturating_mul(4),
        ),
        128,
    )
    .clamp(512, 2048);
    align_mib(
        token_embd_mib
            .saturating_add(output_mib)
            .saturating_add(eh_proj_mib)
            .saturating_add(row_workspace_mib)
            .saturating_add(layer_workspace_mib)
            .saturating_add(512),
        256,
    )
}

fn current_mtp_auto_resource_hint() -> Option<MtpAutoResourceHint> {
    #[cfg(feature = "cuda")]
    {
        return backend_runtime::cuda_memory_info()
            .ok()
            .map(|info| MtpAutoResourceHint {
                total_vram_mib: info.total_bytes / (1024 * 1024),
                free_vram_mib: info.free_bytes / (1024 * 1024),
            });
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn mtp_auto_policy_for_model(
    architecture: ModelArchitecture,
    metadata: &ModelMetadata,
    has_mtp_runtime: bool,
    device_verify_supported: bool,
    resource: Option<MtpAutoResourceHint>,
) -> MtpAutoPolicy {
    let dense_spec_k = 1;
    let dense_min_free_vram_mib = mtp_device_verify_min_free_vram_mib(metadata, dense_spec_k);
    if !has_mtp_runtime {
        return MtpAutoPolicy {
            enabled: false,
            spec_k: 4,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "no-mtp-runtime",
        };
    }
    // pm116: GLM-DSA 는 Metal 에서 batch verify 채택 (64-decode 3페어 전승
    // 1197→929 ms/tok -22%, accept 77.8%, nextn 1층이라 k=1). device verify 가
    // 필요 없어 아래 가드보다 먼저 판정. CUDA 는 1-token verify 이득
    // 미확인(pc-cuda-glm52 보류) — Metal 빌드만 auto ON.
    if architecture == ModelArchitecture::GlmDsa {
        return MtpAutoPolicy {
            enabled: cfg!(all(feature = "metal", not(feature = "cuda"))),
            spec_k: 1,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "glm-dsa-metal-batch-verify-auto",
        };
    }
    if !device_verify_supported {
        return MtpAutoPolicy {
            enabled: false,
            spec_k: 1,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "mtp-device-verify-unsupported",
        };
    }

    let device_verify_possible =
        resource.is_some_and(|resource| resource.free_vram_mib >= dense_min_free_vram_mib);

    match architecture {
        ModelArchitecture::Qwen35MoE if resource.is_none() => MtpAutoPolicy {
            enabled: false,
            spec_k: dense_spec_k,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "cuda-resource-info-unavailable",
        },
        ModelArchitecture::Qwen35MoE if !device_verify_possible => MtpAutoPolicy {
            enabled: false,
            spec_k: dense_spec_k,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "insufficient-free-vram-for-mtp-device-verify",
        },
        ModelArchitecture::Qwen35MoE => MtpAutoPolicy {
            enabled: true,
            spec_k: dense_spec_k,
            device_verify: true,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "qwen35moe-k1-device-verify-auto",
        },
        ModelArchitecture::Qwen35
            if mtp_dense_decode_work_units(metadata)
                < mtp_dense_decode_work_threshold(metadata) =>
        {
            MtpAutoPolicy {
                enabled: false,
                spec_k: dense_spec_k,
                device_verify: device_verify_possible,
                min_free_vram_mib: dense_min_free_vram_mib,
                resource,
                reason: "dense-decode-work-below-mtp-threshold",
            }
        }
        ModelArchitecture::Qwen35 if resource.is_none() => MtpAutoPolicy {
            enabled: false,
            spec_k: dense_spec_k,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "cuda-resource-info-unavailable",
        },
        ModelArchitecture::Qwen35 if !device_verify_possible => MtpAutoPolicy {
            enabled: false,
            spec_k: dense_spec_k,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "insufficient-free-vram-for-mtp-device-verify",
        },
        ModelArchitecture::Qwen35 => MtpAutoPolicy {
            enabled: true,
            spec_k: dense_spec_k,
            device_verify: true,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "dense-qwen35-device-verify-auto",
        },
        _ => MtpAutoPolicy {
            enabled: false,
            spec_k: 1,
            device_verify: false,
            min_free_vram_mib: dense_min_free_vram_mib,
            resource,
            reason: "unmeasured-mtp-policy",
        },
    }
}

impl Engine {
    pub fn mtp_runtime_ready(&self) -> bool {
        match self.mtp_runtime.as_ref() {
            Some(EngineMtpRuntime::InModel(_)) => self.mtp.is_some(),
            Some(EngineMtpRuntime::External(_)) => true,
            None => false,
        }
    }

    /// External drafter (Gemma 4 assistant 모델) 를 engine 에 연결한다.
    ///
    /// 체크:
    /// 1. target 이 Gemma 4 아키텍처인지 (`ModelArchitecture::Gemma4`).
    /// 2. drafter backbone_hidden == target hidden_dim.
    /// 3. drafter 가 SWA 레이어를 가지면 target 도 sliding_window_pattern 이 있어야 한다.
    pub fn attach_external_drafter(&mut self, drafter: rnb_mtp::drafter::Drafter) -> Result<()> {
        use rnb_loader::Architecture as ModelArchitecture;

        if self.architecture != ModelArchitecture::Gemma4 {
            return Err(LlmError::Forward(format!(
                "external drafter requires target arch 'gemma4', target is '{:?}'",
                self.architecture,
            )));
        }

        let target_hidden = self.metadata.hidden_dim;
        if drafter.backbone_hidden != target_hidden {
            return Err(LlmError::Forward(format!(
                "drafter backbone_hidden={} != target hidden_dim={}",
                drafter.backbone_hidden, target_hidden,
            )));
        }

        let has_swa_layer = drafter.layers.iter().any(|l| l.is_sliding_window);
        let target_has_swa = !self.metadata.sliding_window_pattern.is_empty();
        if has_swa_layer && !target_has_swa {
            return Err(LlmError::Forward(
                "drafter has SWA layers but target has no sliding_window_pattern".into(),
            ));
        }

        // cu47 Phase 1: drafter Q4_K weight 의 device cache prewarm.
        // 매 forward call 의 weight reupload 제거 (cu46 의 ε 원인 일부).
        // env 비활성 또는 cuda 비활성 시 noop. ABAB 4-run mc78_chat_essay_ko
        // (193 decode tokens) generate median Δ -0.06% ε — drafter weight 가
        // 작아서 backend runtime의 자체 cache가 이미 충분, 외부 prewarm 효과 ε.
        // Phase 2 (hidden state device buffer + forward entry/exit roundtrip)
        // 가 진짜 lever.
        #[cfg(feature = "cuda")]
        {
            if crate::engine::policy::drafter_cuda_enabled() {
                match rnb_mtp::drafter::drafter_prewarm_weights_cuda_full(
                    &drafter.layers,
                    Some(&drafter.pre_projection),
                    Some(&drafter.post_projection),
                ) {
                    Ok(_n) => {}
                    Err(e) => {
                        eprintln!("[cu47] drafter cuda prewarm failed: {e:?} — fallback host")
                    }
                }
            }
        }

        let runtime =
            crate::external_drafter::ExternalDrafterRuntime::new(std::sync::Arc::new(drafter));
        self.mtp_runtime = Some(EngineMtpRuntime::External(runtime));
        Ok(())
    }

    pub fn mtp_auto_policy(&self) -> MtpAutoPolicy {
        mtp_auto_policy_for_model(
            self.architecture,
            &self.metadata,
            self.mtp_runtime_ready(),
            self.mtp_device_verify_supported_by_weights(),
            current_mtp_auto_resource_hint(),
        )
    }

    pub(crate) fn mtp_device_verify_requested(&self) -> bool {
        crate::engine::runtime::policy::mtp_device_verify_override()
            .unwrap_or_else(|| self.mtp_auto_policy().device_verify)
    }

    fn mtp_device_verify_supported_by_weights(&self) -> bool {
        let Some(weights) = self.weights.as_ref() else {
            return false;
        };
        let token_embd_supported = matches!(
            weights.token_embd.ggml_type,
            rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K | rnb_loader::GGMLType::Q8_0
        ) && weights.token_embd.cols == self.metadata.hidden_dim
            && weights.token_embd.data.as_bytes().is_some();
        let output_supported = matches!(
            weights.output.ggml_type,
            rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K | rnb_loader::GGMLType::Q8_0
        ) && weights.output.cols == self.metadata.hidden_dim
            && weights.output.data.as_bytes().is_some()
            && weights.output_norm.numel() == self.metadata.hidden_dim;
        token_embd_supported && output_supported
    }

    /// pm118: `RNB_MTP` 미설정과 `auto` 는 모델별 auto policy `enabled` 로
    /// 위임한다 (채택된 최적화는 무-env 기본 ON — GLM+Metal batch verify 등,
    /// policy 가 no-mtp-runtime/아키텍처/자원 조건으로 안전 gate). 명시
    /// falsey 는 off, 그 외 명시값은 기존처럼 강제 on. auto 판정은 엔진당
    /// 1회 캐시 (`mtp_auto_requested_cache`).
    pub(crate) fn mtp_spec_requested(&self) -> bool {
        match super::policy::env_string("RNB_MTP") {
            None => *self
                .mtp_auto_requested_cache
                .get_or_init(|| self.mtp_auto_policy().enabled),
            Some(value) => {
                let value = value.trim().to_ascii_lowercase();
                match value.as_str() {
                    "" => *self
                        .mtp_auto_requested_cache
                        .get_or_init(|| self.mtp_auto_policy().enabled),
                    "0" | "false" | "off" | "no" => false,
                    "auto" => *self
                        .mtp_auto_requested_cache
                        .get_or_init(|| self.mtp_auto_policy().enabled),
                    _ => true,
                }
            }
        }
    }

    pub(crate) fn mtp_explicitly_forced(&self) -> bool {
        super::policy::env_string("RNB_MTP").is_some_and(|value| {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no" | "auto")
        })
    }

    /// pm118: MTP 라운드당 draft 수 — `RNB_MTP` 가 명시 truthy 로 강제된 경우만
    /// caller 요청값을 쓰고, 미설정/auto 진입은 모델별 auto policy `spec_k`
    /// (GLM nextn 1층 k=1 등) 를 쓴다.
    pub(crate) fn mtp_effective_spec_k(&self, requested: usize) -> usize {
        let explicit_force = self.mtp_explicitly_forced();
        if explicit_force {
            requested.max(1)
        } else {
            self.mtp_auto_policy().spec_k.max(1)
        }
    }

    // ── mc78 Task 12 — External drafter 접근자 (mtp_generate.rs 용) ─────────

    /// External runtime 이 존재하는지 여부. `mtp_runtime` 필드가 `pub(super)` 라
    /// crate root 레벨 `mtp_generate.rs` 에서 직접 접근 불가 → 여기서 위임.
    pub(crate) fn mtp_is_external_runtime(&self) -> bool {
        self.mtp_runtime
            .as_ref()
            .map(|r| r.is_external())
            .unwrap_or(false)
    }

    /// External drafter runtime 의 `&mut ExternalDrafterRuntime` 반환.
    /// InModel runtime 이거나 없으면 `None`.
    pub(crate) fn mtp_external_runtime_mut(
        &mut self,
    ) -> Option<&mut crate::external_drafter::ExternalDrafterRuntime> {
        self.mtp_runtime.as_mut()?.as_external_mut()
    }

    pub(crate) fn mtp_checkpoint(&self) -> Option<EngineMtpCheckpoint> {
        self.mtp_runtime.as_ref().map(|r| r.checkpoint())
    }

    pub(crate) fn mtp_restore_checkpoint(&mut self, checkpoint: Option<&EngineMtpCheckpoint>) {
        let (Some(runtime), Some(checkpoint)) = (self.mtp_runtime.as_mut(), checkpoint) else {
            return;
        };
        runtime.restore(checkpoint);
    }

    pub(crate) fn mtp_clear_sequence_state(&mut self) {
        if let Some(runtime) = self.mtp_runtime.as_mut() {
            runtime.clear_sequence_state();
        }
    }

    pub(crate) fn mtp_observe_prompt_batch(
        &mut self,
        tokens: &[u32],
        hidden_rows: &[f32],
    ) -> Result<()> {
        if !self.mtp_spec_requested() || tokens.is_empty() {
            return Ok(());
        }
        let Some(mut runtime) = self.mtp_runtime.take() else {
            return Err(LlmError::Forward(
                "RNB_MTP=1 but model has no loaded MTP runtime".to_string(),
            ));
        };
        let result = (|| {
            let Some(inner) = runtime.as_in_model_mut() else {
                // External drafter (mc78): in-model observation is a no-op.
                // The external drafter has its own KV-share path via shared_kv_view.
                return Ok(());
            };
            let weights = self.weights.as_ref().ok_or_else(|| {
                LlmError::Forward("RNB_MTP=1 requires loaded model weights".to_string())
            })?;
            let target_pos_start = self.kv_cache.current_len().saturating_sub(tokens.len());
            observe_target_batch(
                inner,
                weights,
                self.architecture,
                tokens,
                hidden_rows,
                target_pos_start,
            )
        })();
        self.mtp_runtime = Some(runtime);
        result
    }

    pub(crate) fn mtp_retain_after_spec(
        &mut self,
        checkpoint: Option<&EngineMtpCheckpoint>,
        kept_tokens: usize,
    ) {
        let Some(runtime) = self.mtp_runtime.as_mut().and_then(|r| r.as_in_model_mut()) else {
            return;
        };
        let checkpoint = match checkpoint {
            Some(EngineMtpCheckpoint::InModel(c)) => c,
            _ => return,
        };
        let base_len = checkpoint.kv_len;
        runtime
            .kv_cache
            .set_len((base_len + kept_tokens).min(runtime.kv_cache.max_seq_len));
        runtime.next_pos = (checkpoint.next_pos + kept_tokens).min(runtime.kv_cache.max_seq_len);
        if kept_tokens == 0 {
            runtime.last_hidden = checkpoint.last_hidden.clone();
            return;
        }
        let hidden_dim = runtime.metadata.hidden_dim;
        let row = kept_tokens - 1;
        if row < runtime.recent_hidden_count {
            let start = row * hidden_dim;
            runtime.last_hidden =
                Some(runtime.recent_hidden_rows[start..start + hidden_dim].to_vec());
        }
    }

    pub(crate) fn mtp_retain_draft_after_spec(
        &mut self,
        checkpoint: Option<&EngineMtpCheckpoint>,
        _verify_tokens: &[u32],
        committed_tokens: usize,
        drafted_tokens: usize,
        hidden_rows: &[f32],
    ) -> Result<()> {
        if !self.mtp_spec_requested() || committed_tokens == 0 {
            return Ok(());
        }
        let checkpoint = match checkpoint {
            Some(EngineMtpCheckpoint::InModel(c)) => c,
            _ => {
                return Err(LlmError::Forward(
                    "MTP draft retain requires an InModel MTP checkpoint".to_string(),
                ))
            }
        };
        let Some(mut runtime) = self.mtp_runtime.take() else {
            return Err(LlmError::Forward(
                "RNB_MTP=1 but model has no loaded MTP runtime".to_string(),
            ));
        };
        let result = (|| {
            let Some(inner) = runtime.as_in_model_mut() else {
                // External drafter (mc78): retain is a no-op. External path
                // manages its own draft state in generate_with_external_drafter.
                return Ok(());
            };
            let hidden_dim = inner.metadata.hidden_dim;
            let last_hidden = committed_last_hidden_row(hidden_rows, hidden_dim, committed_tokens)?;
            let keep_blocks = retained_draft_kv_tokens(drafted_tokens, committed_tokens);
            inner
                .kv_cache
                .set_len((checkpoint.kv_len + keep_blocks).min(inner.kv_cache.max_seq_len));
            let next_pos_tokens = retained_draft_next_pos_tokens(drafted_tokens, committed_tokens);
            inner.next_pos =
                (checkpoint.next_pos + next_pos_tokens).min(inner.kv_cache.max_seq_len);

            let committed_len = committed_tokens * hidden_dim;
            inner.recent_hidden_rows.clear();
            inner
                .recent_hidden_rows
                .extend_from_slice(&hidden_rows[..committed_len]);
            inner.recent_hidden_count = committed_tokens;
            inner.last_hidden = Some(last_hidden);
            Ok(())
        })();
        self.mtp_runtime = Some(runtime);
        result
    }

    pub(crate) fn mtp_observe_target_batch(
        &mut self,
        tokens: &[u32],
        hidden_rows: &[f32],
    ) -> Result<()> {
        if !self.mtp_spec_requested() || tokens.is_empty() {
            return Ok(());
        }
        let Some(mut runtime) = self.mtp_runtime.take() else {
            return Err(LlmError::Forward(
                "RNB_MTP=1 but model has no loaded MTP runtime".to_string(),
            ));
        };
        let result = (|| {
            let Some(inner) = runtime.as_in_model_mut() else {
                // External drafter (mc78): in-model observation is a no-op.
                return Ok(());
            };
            let weights = self.weights.as_ref().ok_or_else(|| {
                LlmError::Forward("RNB_MTP=1 requires loaded model weights".to_string())
            })?;
            let target_pos_start = self.kv_cache.current_len().saturating_sub(tokens.len());
            observe_target_batch(
                inner,
                weights,
                self.architecture,
                tokens,
                hidden_rows,
                target_pos_start,
            )
        })();
        self.mtp_runtime = Some(runtime);
        result
    }

    pub(crate) fn mtp_draft_tokens(&mut self, first_token: u32, n_max: usize) -> Result<Vec<u32>> {
        if n_max == 0 {
            return Ok(Vec::new());
        }
        if !self.mtp_spec_requested() {
            return Ok(Vec::new());
        }
        let Some(mut runtime) = self.mtp_runtime.take() else {
            return Err(LlmError::Forward(
                "RNB_MTP=1 but model has no loaded MTP runtime".to_string(),
            ));
        };
        let result = (|| {
            let Some(inner) = runtime.as_in_model_mut() else {
                // External drafter (mc78): in-model draft_tokens never called.
                // External path uses ExternalDrafterStepper via
                // generate_with_external_drafter; this branch only fires for
                // legacy bench code paths that haven't migrated.
                return Ok(Vec::new());
            };
            let weights = self.weights.as_ref().ok_or_else(|| {
                LlmError::Forward("RNB_MTP=1 requires loaded model weights".to_string())
            })?;
            draft_tokens(inner, weights, self.architecture, first_token, n_max)
        })();
        self.mtp_runtime = Some(runtime);
        result
    }
}

fn retained_draft_kv_tokens(drafted_tokens: usize, committed_tokens: usize) -> usize {
    drafted_tokens.min(committed_tokens)
}

fn retained_draft_next_pos_tokens(drafted_tokens: usize, committed_tokens: usize) -> usize {
    retained_draft_kv_tokens(drafted_tokens, committed_tokens)
}

fn committed_last_hidden_row(
    hidden_rows: &[f32],
    hidden_dim: usize,
    committed_tokens: usize,
) -> Result<Vec<f32>> {
    if committed_tokens == 0 {
        return Err(LlmError::Forward(
            "MTP retain requires at least one committed token".to_string(),
        ));
    }
    let expected = committed_tokens * hidden_dim;
    if hidden_rows.len() < expected {
        return Err(LlmError::Forward(format!(
            "MTP retain hidden rows mismatch: got {}, need at least {}",
            hidden_rows.len(),
            expected
        )));
    }
    let start = (committed_tokens - 1) * hidden_dim;
    Ok(hidden_rows[start..start + hidden_dim].to_vec())
}

fn observe_target_batch(
    runtime: &mut InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    tokens: &[u32],
    hidden_rows: &[f32],
    target_pos_start: usize,
) -> Result<()> {
    let hidden_dim = runtime.metadata.hidden_dim;
    if hidden_rows.len() != tokens.len() * hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP observe hidden rows mismatch: got {}, expected {}",
            hidden_rows.len(),
            tokens.len() * hidden_dim
        )));
    }

    runtime.recent_hidden_rows.clear();
    runtime.recent_hidden_rows.extend_from_slice(hidden_rows);
    runtime.recent_hidden_count = tokens.len();

    let pending_hidden = runtime.last_hidden.clone();
    let (pair_tokens, pair_hiddens, pair_positions) = build_shifted_mtp_pairs(
        pending_hidden.as_deref(),
        tokens,
        hidden_rows,
        hidden_dim,
        target_pos_start,
    )?;
    if pair_tokens.is_empty() {
        let last_start = (tokens.len() - 1) * hidden_dim;
        runtime.last_hidden = Some(hidden_rows[last_start..last_start + hidden_dim].to_vec());
        runtime.next_pos = target_pos_start + tokens.len();
        return Ok(());
    }

    let hidden = run_mtp_block(
        runtime,
        weights,
        architecture,
        &pair_tokens,
        &pair_hiddens,
        pair_positions[0],
    )?;
    let _ = hidden;

    let last_start = (tokens.len() - 1) * hidden_dim;
    runtime.last_hidden = Some(hidden_rows[last_start..last_start + hidden_dim].to_vec());
    runtime.next_pos = target_pos_start + tokens.len();
    Ok(())
}

fn build_shifted_mtp_pairs<'a>(
    pending_hidden: Option<&'a [f32]>,
    tokens: &'a [u32],
    hidden_rows: &'a [f32],
    hidden_dim: usize,
    target_pos_start: usize,
) -> Result<(Vec<u32>, Vec<&'a [f32]>, Vec<usize>)> {
    if hidden_rows.len() != tokens.len() * hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP observe hidden rows mismatch: got {}, expected {}",
            hidden_rows.len(),
            tokens.len() * hidden_dim
        )));
    }
    if let Some(hidden) = pending_hidden {
        if hidden.len() != hidden_dim {
            return Err(LlmError::Forward(format!(
                "MTP pending hidden mismatch: got {}, expected {}",
                hidden.len(),
                hidden_dim
            )));
        }
    }

    let pair_count = pending_hidden.map(|_| 1).unwrap_or(0) + tokens.len().saturating_sub(1);
    let mut pair_tokens = Vec::with_capacity(pair_count);
    let mut pair_hiddens = Vec::with_capacity(pair_count);
    let mut pair_positions = Vec::with_capacity(pair_count);

    if let Some(hidden) = pending_hidden {
        if let Some(&token) = tokens.first() {
            pair_tokens.push(token);
            pair_hiddens.push(hidden);
            pair_positions.push(target_pos_start);
        }
    }
    for i in 0..tokens.len().saturating_sub(1) {
        let start = i * hidden_dim;
        pair_tokens.push(tokens[i + 1]);
        pair_hiddens.push(&hidden_rows[start..start + hidden_dim]);
        pair_positions.push(target_pos_start + i + 1);
    }

    Ok((pair_tokens, pair_hiddens, pair_positions))
}

fn draft_tokens(
    runtime: &mut InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    first_token: u32,
    n_max: usize,
) -> Result<Vec<u32>> {
    let mut h = runtime.last_hidden.clone().ok_or_else(|| {
        LlmError::Forward("MTP draft requested before target hidden was observed".to_string())
    })?;
    let mut cond_token = first_token;
    let mut out = Vec::with_capacity(n_max);

    for _ in 0..n_max {
        #[cfg(feature = "cuda")]
        let device_step =
            run_mtp_device_draft_step(runtime, weights, architecture, cond_token, &h)?;
        #[cfg(not(feature = "cuda"))]
        let device_step: Option<(Vec<f32>, u32)> = None;
        let (hidden, token) = if let Some(step) = device_step {
            step
        } else {
            let hidden = run_mtp_block(
                runtime,
                weights,
                architecture,
                &[cond_token],
                &[&h],
                runtime.next_pos,
            )?;
            let token = mtp_argmax(runtime, weights, architecture, &hidden)?;
            (hidden, token)
        };
        h = hidden;
        out.push(token);
        cond_token = token;
        runtime.next_pos += 1;
    }
    runtime.last_hidden = Some(h);
    Ok(out)
}

#[cfg(feature = "cuda")]
fn run_mtp_device_draft_step(
    runtime: &mut InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    cond_token: u32,
    target_hidden: &[f32],
) -> Result<Option<(Vec<f32>, u32)>> {
    if architecture != ModelArchitecture::Qwen35MoE
        || !super::policy::mtp_device_draft_enabled()
        || runtime.weights.block.shared_expert_moe.is_none()
    {
        return Ok(None);
    }
    let hidden_dim = runtime.metadata.hidden_dim;
    if target_hidden.len() != hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP device draft target hidden length {} != hidden_dim {hidden_dim}",
            target_hidden.len()
        )));
    }
    let head = runtime
        .weights
        .shared_head_head
        .as_ref()
        .unwrap_or(&weights.output);
    if !matches!(
        head.ggml_type,
        rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K | rnb_loader::GGMLType::Q8_0
    ) {
        return Ok(None);
    }
    if head.cols != hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP device draft output cols {} != hidden_dim {hidden_dim}",
            head.cols
        )));
    }
    let output_weight = head.data.as_bytes().ok_or_else(|| {
        LlmError::Forward("MTP device draft output raw bytes missing".to_string())
    })?;
    let output_norm = kernels::tensor_as_f32_slice(&runtime.weights.shared_head_norm);
    if output_norm.len() != hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP device draft output norm length {} != hidden_dim {hidden_dim}",
            output_norm.len()
        )));
    }

    let embed_weight = runtime
        .weights
        .embed_tokens
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let tok_embd = embed_weight.gather(&[cond_token])?;
    let tok_data = kernels::tensor_as_f32_slice(&tok_embd);
    let enorm = kernels::tensor_as_f32_slice(&runtime.weights.enorm);
    let hnorm = kernels::tensor_as_f32_slice(&runtime.weights.hnorm);
    let mut combined = vec![0.0_f32; hidden_dim * 2];
    apply_model_norm_into(
        tok_data,
        enorm,
        runtime.metadata.norm_eps,
        &mut combined[..hidden_dim],
        architecture,
    );
    apply_model_norm_into(
        target_hidden,
        hnorm,
        runtime.metadata.norm_eps,
        &mut combined[hidden_dim..],
        architecture,
    );
    let projected = runtime.weights.eh_proj.gemv_vec(&combined)?;
    emit_mtp_finite_trace(
        "mtp-device-draft",
        runtime.weights.layer_index,
        "eh_projected",
        &projected,
    );

    let (rope_dim, rope_theta, proportional_rope) = resolve_rope_params(
        &runtime.metadata,
        architecture,
        0,
        runtime.metadata.head_dim,
    );
    if proportional_rope {
        return Ok(None);
    }
    let qwen_mrope_dim = qwen_text_mrope_dim(
        &runtime.metadata,
        architecture,
        rope_dim,
        runtime.metadata.head_dim,
    );
    let device_rope_dim = qwen_mrope_dim.unwrap_or(rope_dim);
    let device_layer_idx = runtime.weights.layer_index;
    let layer = super::inference::build_mtp_device_draft_attention_moe_layer(
        &runtime.weights.block,
        &runtime.metadata,
        &runtime.kv_cache,
        device_layer_idx,
    )?;
    let result = super::cuda_runtime::qwen35_mtp_device_draft_step(
        super::cuda_runtime::MtpDeviceDraftRequest {
            projected_hidden: &projected,
            pos_start: runtime.next_pos,
            hidden_dim,
            rope_dim: device_rope_dim,
            rope_neox: qwen_mrope_dim.is_some(),
            rope_theta,
            layer: &layer,
            output_weight,
            output_quant: head.ggml_type as u32,
            output_rows: head.rows,
            output_cols: head.cols,
            output_norm,
            norm_eps: runtime.metadata.norm_eps,
        },
    )
    .map_err(|err| {
        LlmError::Forward(format!(
            "Qwen MTP device draft layer execution failed: {err}"
        ))
    })?;
    let kv = result.attention_kv;
    let expected_kv_values = runtime.weights.block.k_weight.rows;
    if kv.layer_idx != device_layer_idx
        || kv.window_tokens != 1
        || kv.kv_rows != expected_kv_values
        || kv.k_bits.len() != expected_kv_values
        || kv.v_bits.len() != expected_kv_values
    {
        return Err(LlmError::Forward(format!(
            "Qwen MTP device draft K/V result mismatch: layer={} window={} rows={} k={} v={}",
            kv.layer_idx,
            kv.window_tokens,
            kv.kv_rows,
            kv.k_bits.len(),
            kv.v_bits.len()
        )));
    }
    let cache_pos_start = runtime.kv_cache.current_len();
    runtime
        .kv_cache
        .replace_layer_f16_range(0, cache_pos_start, 1, &kv.k_bits, &kv.v_bits);
    runtime.kv_cache.set_len(mtp_block_cache_len_after(
        cache_pos_start,
        1,
        runtime.kv_cache.max_seq_len,
    ));
    emit_mtp_finite_trace(
        "mtp-device-draft",
        runtime.weights.layer_index,
        "output_hidden",
        &result.hidden,
    );
    Ok(Some((result.hidden, result.token)))
}

fn run_mtp_block(
    runtime: &mut InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    tokens: &[u32],
    hiddens: &[&[f32]],
    rope_pos_start: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(tokens.len(), hiddens.len());
    let prep_start = std::time::Instant::now();
    let hidden_dim = runtime.metadata.hidden_dim;
    let seq_len = tokens.len();
    let mut combined = vec![0.0f32; seq_len * hidden_dim * 2];

    let embed_weight = runtime
        .weights
        .embed_tokens
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let tok_embd = embed_weight.gather(tokens)?;
    let tok_data = kernels::tensor_as_f32_slice(&tok_embd);
    let enorm = kernels::tensor_as_f32_slice(&runtime.weights.enorm);
    let hnorm = kernels::tensor_as_f32_slice(&runtime.weights.hnorm);

    for row in 0..seq_len {
        let dst = row * hidden_dim * 2;
        let tok = &tok_data[row * hidden_dim..(row + 1) * hidden_dim];
        apply_model_norm_into(
            tok,
            enorm,
            runtime.metadata.norm_eps,
            &mut combined[dst..dst + hidden_dim],
            architecture,
        );
        apply_model_norm_into(
            hiddens[row],
            hnorm,
            runtime.metadata.norm_eps,
            &mut combined[dst + hidden_dim..dst + hidden_dim * 2],
            architecture,
        );
    }

    let projection_start = std::time::Instant::now();
    let prep_ms = prep_start.elapsed().as_secs_f64() * 1000.0;
    let projected = runtime.weights.eh_proj.gemv_vec(&combined)?;
    let projection_ms = projection_start.elapsed().as_secs_f64() * 1000.0;
    emit_mtp_finite_trace(
        "mtp",
        runtime.weights.layer_index,
        "eh_projected",
        &projected,
    );
    if architecture == ModelArchitecture::GlmDsa {
        return run_glm_mtp_block(runtime, projected, seq_len, rope_pos_start);
    }
    if seq_len == 1 && super::policy::mtp_decode_block_enabled() {
        let layer_start = std::time::Instant::now();
        let hidden =
            run_mtp_block_decode(runtime, weights, architecture, projected, rope_pos_start)?;
        if crate::engine::policy::env_os_string("RNB_MTP_DRAFT_TRACE").is_some() {
            eprintln!(
                "[mtp-draft-block] prep={prep_ms:.3}ms projection={projection_ms:.3}ms layer={:.3}ms",
                layer_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(hidden);
    }
    let hidden = Tensor::from_vec(projected, &[seq_len, hidden_dim]);
    let cache_pos_start = runtime.kv_cache.current_len();
    let hidden = forward_attention_layer_with_rope_pos(
        &mut runtime.kv_cache,
        &runtime.metadata,
        architecture,
        hidden,
        &runtime.weights.block,
        weights.rope_freqs.as_ref(),
        0,
        seq_len,
        cache_pos_start,
        rope_pos_start,
        runtime.metadata.num_heads,
        runtime.metadata.num_kv_heads,
        runtime.metadata.head_dim,
        runtime.metadata.num_kv_heads * runtime.metadata.head_dim,
        runtime.metadata.rope_theta,
        runtime.metadata.norm_eps,
    )?;
    runtime.kv_cache.set_len(mtp_block_cache_len_after(
        cache_pos_start,
        seq_len,
        runtime.kv_cache.max_seq_len,
    ));
    let data = kernels::tensor_as_f32_slice(&hidden);
    let last = data[(seq_len - 1) * hidden_dim..seq_len * hidden_dim].to_vec();
    Ok(last)
}

fn run_glm_mtp_block(
    runtime: &mut InModelMtpRuntime,
    projected: Vec<f32>,
    seq_len: usize,
    rope_pos_start: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = runtime.metadata.hidden_dim;
    if projected.len() != seq_len.saturating_mul(hidden_dim) {
        return Err(LlmError::Forward(format!(
            "GLM MTP block projection mismatch: got {}, expected {}",
            projected.len(),
            seq_len.saturating_mul(hidden_dim)
        )));
    }
    let mla = runtime
        .weights
        .glm_dsa_attention
        .as_ref()
        .ok_or_else(|| LlmError::Forward("GLM MTP MLA weights are not loaded".to_string()))?;
    let cache_pos_start = runtime.kv_cache.current_len();
    for row in 0..seq_len {
        runtime.scratch.hidden[..hidden_dim]
            .copy_from_slice(&projected[row * hidden_dim..(row + 1) * hidden_dim]);
        super::models::glm_dsa::decode_layer_with_positions(
            &mut runtime.kv_cache,
            &runtime.metadata,
            &mut runtime.scratch,
            &runtime.weights.block,
            mla,
            0,
            runtime.weights.layer_index,
            cache_pos_start + row,
            rope_pos_start + row,
        )?;
    }
    runtime.kv_cache.set_len(mtp_block_cache_len_after(
        cache_pos_start,
        seq_len,
        runtime.kv_cache.max_seq_len,
    ));
    emit_mtp_finite_trace(
        "glm-mtp",
        runtime.weights.layer_index,
        "output_hidden",
        &runtime.scratch.hidden[..hidden_dim],
    );
    Ok(runtime.scratch.hidden[..hidden_dim].to_vec())
}

fn run_mtp_block_decode(
    runtime: &mut InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    projected: Vec<f32>,
    rope_pos_start: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = runtime.metadata.hidden_dim;
    if projected.len() != hidden_dim {
        return Err(LlmError::Forward(format!(
            "MTP decode block projection mismatch: got {}, expected {}",
            projected.len(),
            hidden_dim
        )));
    }
    emit_mtp_finite_trace(
        "mtp-decode",
        runtime.weights.layer_index,
        "input_projected",
        &projected,
    );
    runtime.scratch.hidden[..hidden_dim].copy_from_slice(&projected);
    let cache_pos_start = runtime.kv_cache.current_len();
    decode_attention_layer_with_rope_pos(
        &mut runtime.kv_cache,
        &runtime.metadata,
        architecture,
        &mut runtime.scratch,
        &runtime.weights.block,
        weights.rope_freqs.as_ref(),
        0,
        cache_pos_start,
        rope_pos_start,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "vulkan")]
        None,
    )?;
    runtime.kv_cache.set_len(mtp_block_cache_len_after(
        cache_pos_start,
        1,
        runtime.kv_cache.max_seq_len,
    ));
    emit_mtp_finite_trace(
        "mtp-decode",
        runtime.weights.layer_index,
        "output_hidden",
        &runtime.scratch.hidden[..hidden_dim],
    );
    Ok(runtime.scratch.hidden[..hidden_dim].to_vec())
}

fn mtp_argmax(
    runtime: &InModelMtpRuntime,
    weights: &ModelWeights,
    architecture: ModelArchitecture,
    hidden: &[f32],
) -> Result<u32> {
    let hidden_dim = runtime.metadata.hidden_dim;
    let mut normed = vec![0.0f32; hidden_dim];
    let norm = kernels::tensor_as_f32_slice(&runtime.weights.shared_head_norm);
    apply_model_norm_into(
        hidden,
        norm,
        runtime.metadata.norm_eps,
        &mut normed,
        architecture,
    );

    let head = runtime
        .weights
        .shared_head_head
        .as_ref()
        .unwrap_or(&weights.output);
    #[cfg(feature = "cuda")]
    if let Some(token) = mtp_cuda_argmax_token(head, &normed) {
        return Ok(token);
    }
    let mut logits = head.gemv_vec(&normed)?;
    apply_logit_softcapping(&mut logits, runtime.metadata.final_logit_softcapping);
    // pm117 계측: `RNB_MTP_TRACE=1` 이면 draft top-1 softmax prob / logit margin
    // 을 출력 — verify 의 `[MTP_TRACE] round= accepted=` 와 조인해 draft 확신도와
    // accept 의 상관을 실측하기 위한 진단 (adaptive-k 게이트 설계용).
    if crate::runtime::mtp_trace_enabled() {
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        let denom: f32 = logits.iter().map(|&v| (v - top1).exp()).sum();
        let prob = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        eprintln!("[MTP_DRAFT_PROB] prob={prob:.4} margin={:.3}", top1 - top2);
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| LlmError::Forward("MTP head produced empty logits".to_string()))
}

#[cfg(feature = "cuda")]
fn mtp_cuda_argmax_token(head: &QuantizedWeight, normed: &[f32]) -> Option<u32> {
    if !super::policy::mtp_output_argmax_enabled() {
        return None;
    }
    backend_runtime::mtp_output_argmax_token_cuda(head, normed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_mtp_metadata() -> MtpMetadata {
        MtpMetadata {
            total_block_count: 41,
            trunk_layers: 40,
            first_mtp_layer: 40,
            nextn_predict_layers: 1,
        }
    }

    fn qwen_mtp_layer() -> MtpLayerTensors {
        MtpLayerTensors {
            layer_index: 40,
            eh_proj_weight: "blk.40.nextn.eh_proj.weight".to_string(),
            enorm_weight: "blk.40.nextn.enorm.weight".to_string(),
            hnorm_weight: "blk.40.nextn.hnorm.weight".to_string(),
            shared_head_norm_weight: "blk.40.nextn.shared_head_norm.weight".to_string(),
            embed_tokens_weight: None,
            shared_head_head_weight: None,
        }
    }

    fn policy_metadata(hidden_dim: usize, num_layers: usize) -> ModelMetadata {
        ModelMetadata {
            num_layers,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            vocab_size: 32000,
            max_seq_len: 4096,
            hidden_dim,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 1_000_000.0,
            rope_dim: 64,
            rope_dim_swa: 64,
            rope_sections: [11, 11, 10, 0],
            norm_eps: 1.0e-6,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 0.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: Vec::new(),
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
        }
    }

    fn policy_resource(total_vram_mib: usize, free_vram_mib: usize) -> MtpAutoResourceHint {
        MtpAutoResourceHint {
            total_vram_mib,
            free_vram_mib,
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn mtp_device_rope_mode_matches_model_attention_semantics() {
        assert!(mtp_device_rope_neox(
            ModelArchitecture::Qwen35MoE,
            true,
            false,
        ));
        assert!(mtp_device_rope_neox(ModelArchitecture::Gemma4, false, true,));
        assert!(!mtp_device_rope_neox(
            ModelArchitecture::Gemma4,
            false,
            false,
        ));
        assert!(mtp_device_rope_neox(ModelArchitecture::Hy3, false, false,));
        assert!(!mtp_device_rope_neox(
            ModelArchitecture::Qwen35,
            false,
            true,
        ));
    }

    #[test]
    fn mtp_auto_policy_enables_dense_qwen35_when_work_and_vram_clear_thresholds() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(4096, 40),
            true,
            true,
            Some(policy_resource(12 * 1024, 10 * 1024)),
        );

        assert!(policy.enabled);
        assert_eq!(policy.spec_k, 1);
        assert!(policy.device_verify);
        assert_eq!(policy.reason, "dense-qwen35-device-verify-auto");
    }

    #[test]
    fn mtp_auto_policy_enables_qwen35_moe_k1_when_vram_covers_device_verify() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35MoE,
            &policy_metadata(2048, 40),
            true,
            true,
            Some(policy_resource(12 * 1024, 10 * 1024)),
        );

        assert!(policy.enabled);
        assert_eq!(policy.spec_k, 1);
        assert!(policy.device_verify);
        assert_eq!(policy.reason, "qwen35moe-k1-device-verify-auto");
    }

    #[test]
    fn mtp_auto_policy_keeps_gemma_device_verify_off_until_validated() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Gemma4,
            &policy_metadata(2560, 42),
            true,
            true,
            Some(policy_resource(24 * 1024, 20 * 1024)),
        );

        assert!(!policy.enabled);
        assert!(!policy.device_verify);
        assert_eq!(policy.reason, "unmeasured-mtp-policy");
    }

    #[test]
    fn mtp_auto_policy_disables_small_qwen35_but_recommends_k1_for_forced_runs() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(1536, 28),
            true,
            true,
            Some(policy_resource(12 * 1024, 10 * 1024)),
        );

        assert!(!policy.enabled);
        assert_eq!(policy.spec_k, 1);
        assert_eq!(policy.reason, "dense-decode-work-below-mtp-threshold");
    }

    #[test]
    fn mtp_auto_policy_disables_models_without_mtp_runtime() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(4096, 40),
            false,
            true,
            Some(policy_resource(12 * 1024, 10 * 1024)),
        );

        assert!(!policy.enabled);
        assert_eq!(policy.spec_k, 4);
        assert_eq!(policy.reason, "no-mtp-runtime");
    }

    #[test]
    fn mtp_auto_policy_requires_cuda_resource_hint_for_auto_enable() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(4096, 40),
            true,
            true,
            None,
        );

        assert!(!policy.enabled);
        assert!(!policy.device_verify);
        assert_eq!(policy.reason, "cuda-resource-info-unavailable");
    }

    #[test]
    fn mtp_auto_policy_disables_when_free_vram_cannot_cover_workspace() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(4096, 40),
            true,
            true,
            Some(policy_resource(12 * 1024, 512)),
        );

        assert!(!policy.enabled);
        assert!(!policy.device_verify);
        assert_eq!(
            policy.reason,
            "insufficient-free-vram-for-mtp-device-verify"
        );
    }

    #[test]
    fn mtp_auto_policy_disables_when_device_verifier_is_not_supported() {
        let policy = mtp_auto_policy_for_model(
            ModelArchitecture::Qwen35,
            &policy_metadata(4096, 40),
            true,
            false,
            Some(policy_resource(12 * 1024, 10 * 1024)),
        );

        assert!(!policy.enabled);
        assert!(!policy.device_verify);
        assert_eq!(policy.reason, "mtp-device-verify-unsupported");
    }

    #[test]
    fn runtime_state_keeps_loader_mtp_layer_tensors() {
        let metadata = qwen_mtp_metadata();
        let tensors = [qwen_mtp_layer()];

        let state = EngineMtpState::from_loader_parts(Some(&metadata), &tensors)
            .unwrap()
            .expect("MTP state");

        assert_eq!(state.total_block_count, 41);
        assert_eq!(state.trunk_layers, 40);
        assert_eq!(state.first_mtp_layer, 40);
        assert_eq!(state.nextn_predict_layers, 1);
        assert_eq!(
            state.layer_tensors[0].eh_proj_weight,
            tensors[0].eh_proj_weight
        );
    }

    #[test]
    fn runtime_state_is_absent_without_loader_mtp_metadata() {
        let state = EngineMtpState::from_loader_parts(None, &[]).unwrap();

        assert!(state.is_none());
    }

    #[test]
    fn mtp_shift_pairs_use_next_token_with_previous_hidden_for_prompt() {
        let tokens = [10, 11, 12];
        let hidden_rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (pair_tokens, pair_hiddens, pair_positions) =
            build_shifted_mtp_pairs(None, &tokens, &hidden_rows, 2, 0).unwrap();

        assert_eq!(pair_tokens, vec![11, 12]);
        assert_eq!(pair_hiddens[0], &[1.0, 2.0]);
        assert_eq!(pair_hiddens[1], &[3.0, 4.0]);
        assert_eq!(pair_positions, vec![1, 2]);
    }

    #[test]
    fn mtp_shift_pairs_continue_from_pending_hidden_across_batches() {
        let pending = [9.0, 10.0];
        let tokens = [20, 21];
        let hidden_rows = [1.0, 2.0, 3.0, 4.0];

        let (pair_tokens, pair_hiddens, pair_positions) =
            build_shifted_mtp_pairs(Some(&pending), &tokens, &hidden_rows, 2, 5).unwrap();

        assert_eq!(pair_tokens, vec![20, 21]);
        assert_eq!(pair_hiddens[0], &[9.0, 10.0]);
        assert_eq!(pair_hiddens[1], &[1.0, 2.0]);
        assert_eq!(pair_positions, vec![5, 6]);
    }

    #[test]
    fn mtp_checkpoint_keeps_kv_len_instead_of_cache_clone() {
        let checkpoint = InModelMtpCheckpoint {
            kv_len: 7,
            next_pos: 9,
            last_hidden: None,
        };

        assert_eq!(checkpoint.kv_len, 7);
        assert_eq!(checkpoint.next_pos, 9);
    }

    #[test]
    fn mtp_block_cache_len_advances_by_sequence_length() {
        assert_eq!(mtp_block_cache_len_after(6, 1, 128), 7);
        assert_eq!(mtp_block_cache_len_after(6, 4, 128), 10);
        assert_eq!(mtp_block_cache_len_after(126, 4, 128), 128);
    }

    #[test]
    fn retained_draft_kv_tokens_keep_generated_blocks_not_unrun_final_token() {
        assert_eq!(retained_draft_kv_tokens(4, 1), 1);
        assert_eq!(retained_draft_kv_tokens(4, 4), 4);
        assert_eq!(retained_draft_kv_tokens(4, 5), 4);
    }

    #[test]
    fn retained_draft_next_pos_tracks_retained_mtp_blocks() {
        assert_eq!(retained_draft_next_pos_tokens(1, 2), 1);
        assert_eq!(retained_draft_next_pos_tokens(4, 1), 1);
        assert_eq!(retained_draft_next_pos_tokens(4, 4), 4);
        assert_eq!(retained_draft_next_pos_tokens(4, 5), 4);
    }

    #[test]
    fn committed_last_hidden_uses_last_committed_target_row() {
        let hidden_rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let hidden = committed_last_hidden_row(&hidden_rows, 2, 2).unwrap();

        assert_eq!(hidden, vec![3.0, 4.0]);
    }

    #[test]
    fn committed_last_hidden_rejects_missing_target_row() {
        let err = committed_last_hidden_row(&[1.0, 2.0], 2, 2).unwrap_err();

        assert!(err.to_string().contains("MTP retain hidden rows mismatch"));
    }
    #[test]
    fn durable_in_model_state_restores_kv_and_hidden_exactly() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        cache.append(0, 0, &[1.0, -2.0], &[3.0, -4.0]);
        cache.append(0, 1, &[5.0, -6.0], &[7.0, -8.0]);
        let mut next_pos = 2;
        let mut last_hidden = Some(vec![9.0, 10.0]);
        let mut recent_hidden_rows = vec![1.0, 2.0, 3.0, 4.0];
        let mut recent_hidden_count = 2;

        let state = capture_in_model_sequence_state(
            &cache,
            next_pos,
            &last_hidden,
            &recent_hidden_rows,
            recent_hidden_count,
        );
        let estimated_bytes = in_model_sequence_state_heap_byte_size_estimate(
            &cache,
            &last_hidden,
            &recent_hidden_rows,
        );
        assert_eq!(
            EngineMtpSequenceState::InModel(state.clone()).heap_byte_size(),
            estimated_bytes
        );

        cache.clear();
        next_pos = 0;
        last_hidden = None;
        recent_hidden_rows.clear();
        recent_hidden_count = 0;
        restore_in_model_sequence_state(
            &mut cache,
            &mut next_pos,
            &mut last_hidden,
            &mut recent_hidden_rows,
            &mut recent_hidden_count,
            2,
            &state,
        )
        .unwrap();

        let expected_key = [1.0, -2.0, 5.0, -6.0].map(|value| half::f16::from_f32(value).to_bits());
        let expected_value =
            [3.0, -4.0, 7.0, -8.0].map(|value| half::f16::from_f32(value).to_bits());
        let (key, value) = cache.get(0);
        assert_eq!(key, expected_key);
        assert_eq!(value, expected_value);
        assert_eq!(next_pos, 2);
        assert_eq!(last_hidden, Some(vec![9.0, 10.0]));
        assert_eq!(recent_hidden_rows, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(recent_hidden_count, 2);
    }
}
