#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("feature \"metal\" is only supported on macOS (Apple Silicon)");

use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOp, BackendOutput,
    BackendRequest, BackendResult,
};

#[cfg(test)]
pub(crate) static METAL_TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSize,
};
#[cfg(target_os = "macos")]
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::collections::HashMap;

#[cfg(target_os = "macos")]
mod device;

#[cfg(target_os = "macos")]
mod compute;
#[cfg(all(test, target_os = "macos"))]
mod qwen_moe_llama_id_microbench;

#[cfg(target_os = "macos")]
mod ffn_chain;
#[cfg(target_os = "macos")]
pub use ffn_chain::{
    QwenMoeLlamaIdInput, QwenMoeLlamaIdQuant, QwenMoeLlamaIdQuantSet, QwenMoeLlamaIdTrace,
};

#[cfg(target_os = "macos")]
pub struct QwenMoeLlamaIdPrefillRequest<'a> {
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_expert_bytes: usize,
    pub down_expert_bytes: usize,
    pub selected_experts: &'a [u32],
    pub route_weights: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub shared_route_weights: &'a [f32],
    pub sparse_quant: QwenMoeLlamaIdQuantSet,
    pub shared_quant: QwenMoeLlamaIdQuantSet,
    pub norm_all: &'a [f32],
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub layer_idx: usize,
}

#[cfg(target_os = "macos")]
pub struct QwenMoeLlamaIdPrefillOutput {
    pub values: Vec<f32>,
    pub trace: QwenMoeLlamaIdTrace,
}

#[cfg(target_os = "macos")]
mod attn_chain;

#[cfg(target_os = "macos")]
mod gdn_chain;

#[cfg(target_os = "macos")]
mod gated_proj_chain;
#[cfg(target_os = "macos")]
mod gdn_conv_delta_chain;
#[cfg(target_os = "macos")]
mod gdn_proj_chain;
#[cfg(target_os = "macos")]
mod prefill_atn_core_chain;
#[cfg(target_os = "macos")]
mod prefill_attn_chain;
#[cfg(target_os = "macos")]
pub use compute::GlmMlaLayerFusedOut;
pub use gdn_proj_chain::{PrefillProjTrace, TensoropsQuant};

#[cfg(target_os = "macos")]
type ResidentKey = (usize, usize);

#[cfg(target_os = "macos")]
/// pm112: GLM MoE decode 의 UD-quant 레이어 조합 선택. 기본값은
/// IQ2_XXS gate/up + IQ3_XXS down + Q5K/Q5K/Q6K shared.
#[derive(Clone, Copy, Default)]
pub struct GlmMoeQuantSelect {
    /// sparse gate/up 이 IQ2_S (기본 IQ2_XXS)
    pub gate_up_iq2s: bool,
    /// sparse down 이 IQ4_XS (기본 IQ3_XXS)
    pub down_iq4xs: bool,
    /// shared gate/up 이 Q6_K (기본 Q5_K)
    pub shared_gate_up_q6k: bool,
    /// shared down 이 Q8_0 (기본 Q6_K)
    pub shared_down_q8_0: bool,
    /// sparse gate/up 이 IQ3_XXS (기본 IQ2_XXS/IQ2_S). gate_up_iq2s 보다 우선.
    pub gate_up_iq3xxs: bool,
    /// shared gate/up 이 Q8_0 (unsloth Qwen; 기본 Q5_K/Q6_K)
    pub shared_gate_up_q8_0: bool,
}

/// pm116 M2: pread 준비물 — plan/staging/handles. run 은 `glm_direct_run_preads`.
struct GlmDirectPreadPrep {
    plan: Vec<((usize, usize), usize, u64, usize)>,
    handles: [Option<std::sync::Arc<std::fs::File>>; 3],
    staging: Retained<ProtocolObject<dyn MTLBuffer>>,
    contents: usize,
}

impl GlmDirectPreadPrep {
    #[allow(clippy::type_complexity)]
    fn to_map(
        &self,
    ) -> std::collections::HashMap<(usize, usize), (Retained<ProtocolObject<dyn MTLBuffer>>, u32)>
    {
        self.plan
            .iter()
            .map(|&(dedup_key, _, _, stage_off)| {
                (dedup_key, (self.staging.clone(), stage_off as u32))
            })
            .collect()
    }
}

/// pm116 M2: prep 의 plan 을 병렬 pread 로 실행. Retained buffer 를 스레드로
/// 넘기지 않도록 contents 주소만 받는다 (구간은 plan 에서 서로 배타 배정).
fn glm_direct_run_preads(
    plan: &[((usize, usize), usize, u64, usize)],
    handles: &[Option<std::sync::Arc<std::fs::File>>; 3],
    contents: usize,
    threads: usize,
) {
    use std::os::unix::fs::FileExt;
    if plan.is_empty() {
        return;
    }
    // pm117: `RNB_METAL_GLM_MOE_PREFILL_PROFILE=1` 이면 stage wall 과 스레드별
    // 순수 pread syscall 시간을 분해 출력 (spawn/join/분배 오버헤드 규명용).
    static PREAD_DETAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let detail = *PREAD_DETAIL
        .get_or_init(|| std::env::var("RNB_METAL_GLM_MOE_PREFILL_PROFILE").as_deref() == Ok("1"));
    let chunk = plan.len().div_ceil(threads.max(1));
    let wall_start = detail.then(std::time::Instant::now);
    let mut syscall_ms: Vec<f64> = Vec::new();
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for part in plan.chunks(chunk) {
            joins.push(scope.spawn(move || {
                let mut busy = 0.0f64;
                for &((_, len), pool_id, file_off, stage_off) in part {
                    let file = handles[pool_id].as_ref().expect("handle");
                    // SAFETY: stage_off..+len 구간은 plan 에서 서로 배타적으로
                    // 배정되며 staging 은 아직 이 스테이지의 GPU 가 읽지 않는다.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut((contents + stage_off) as *mut u8, len)
                    };
                    let t0 = detail.then(std::time::Instant::now);
                    file.read_exact_at(dst, file_off)
                        .unwrap_or_else(|e| panic!("GLM direct-file pread: {e}"));
                    if let Some(t0) = t0 {
                        busy += t0.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                busy
            }));
        }
        if detail {
            syscall_ms = joins.into_iter().map(|j| j.join().unwrap()).collect();
        }
    });
    if let Some(wall_start) = wall_start {
        let bytes: usize = plan.iter().map(|&((_, len), ..)| len).sum();
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let max = syscall_ms.iter().cloned().fold(0.0, f64::max);
        let sum: f64 = syscall_ms.iter().sum();
        eprintln!(
            "[glm-pread-detail] slices={} bytes={} threads={} wall={wall_ms:.2} syscall_max={max:.2} syscall_sum={sum:.2}",
            plan.len(),
            bytes,
            syscall_ms.len(),
        );
    }
}

#[cfg(target_os = "macos")]
fn resident_key(raw: &[u8]) -> ResidentKey {
    (raw.as_ptr() as usize, raw.len())
}

/// pm115 M1: GLM prefill direct-file 소스 — expert weight 를 mmap 대신 파일에서
/// pread 로 스테이징 buffer 에 읽는다 (wiring page fault + page cache 경합 우회).
/// `base_addrs`/`lens` 는 mmap 상의 tensor 범위 — slice 포인터에서 파일 오프셋을
/// 역산하고, 범위 밖 slice(shared expert 등)는 mmap wrap 으로 남긴다.
pub struct GlmPrefillDirectFile<'a> {
    /// gate/up/down exps tensor 순.
    pub paths: [&'a std::path::Path; 3],
    pub base_offsets: [u64; 3],
    pub base_addrs: [usize; 3],
    pub lens: [usize; 3],
}

/// pm115 M1: 기본 ON (ABAB -31% 채택), `RNB_METAL_GLM_PREFILL_DIRECT_FILE=0` opt-out.
#[cfg(target_os = "macos")]
fn glm_prefill_direct_file_enabled() -> bool {
    std::env::var("RNB_METAL_GLM_PREFILL_DIRECT_FILE").as_deref() != Ok("0")
}

/// pm117: direct-file pread 가 OS page cache 를 쓰도록 기본 ON (F_NOCACHE 제거).
/// `RNB_METAL_GLM_PREAD_PAGECACHE=0` opt-out 시 pm115 의 F_NOCACHE 로 복귀.
fn glm_pread_pagecache_enabled() -> bool {
    static PREAD_PAGECACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PREAD_PAGECACHE
        .get_or_init(|| std::env::var("RNB_METAL_GLM_PREAD_PAGECACHE").as_deref() != Ok("0"))
}

/// pm117 후속 3: stage-major overlap 은 seq 가 작으면 순손해 — 겹칠 GPU 시간이
/// 없는데 라운드당 3 command buffer commit/wait 만 순증 (MTP verify seq 2 에서
/// OFF 가 -2.4~-4.3%, seq 16 -11.7% / seq 54 부터 ON 승 스윕). 경계 기본 32,
/// `RNB_METAL_GLM_STAGE_OVERLAP_MIN_SEQ` 로 override (0 = 항상 overlap).
fn glm_stage_overlap_min_seq() -> usize {
    static MIN_SEQ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_SEQ.get_or_init(|| {
        std::env::var("RNB_METAL_GLM_STAGE_OVERLAP_MIN_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    })
}

/// pm112: weight buffer 상주 LRU. queue-attached MTLResidencySet 에 buffer 를 등록해
/// per-submit wiring(command buffer drv 구간)을 없애고, wired 총량은 예산 안으로
/// 유지한다. 무제한 wiring 은 금지 — GLM-5.2 20토큰 런에서 wired 38GB 스래싱 관측.
/// 예산 산출식: `recommendedMaxWorkingSetSize()/2` (carrier/KV/page cache 몫을 남김).
/// `RNB_METAL_WEIGHT_RESIDENCY_MB` env 는 진단 override.
#[cfg(target_os = "macos")]
struct WeightResidencyLru {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// key → (buffer, wired bytes, last-use tick). buffer 는 removeAllocation 용 보관.
    entries: HashMap<ResidentKey, (Retained<ProtocolObject<dyn MTLBuffer>>, u64, u64)>,
    /// lazy eviction 큐: (Reverse(tick), key). touch 마다 push 되고, evict 시
    /// pop 한 tick 이 entries 의 최신 tick 과 다르면 stale 로 버린다.
    /// O(n) 최소 스캔 evict 는 예산 초과 상태의 decode 에서 콜당 ms 급이라 금지.
    queue: std::collections::BinaryHeap<std::cmp::Reverse<(u64, ResidentKey)>>,
    total_bytes: u64,
    budget_bytes: u64,
    tick: u64,
    dirty: bool,
}

#[cfg(target_os = "macos")]
impl WeightResidencyLru {
    fn new(ctx: &compute::MetalContext) -> Self {
        let desc = MTLResidencySetDescriptor::new();
        let set = ctx
            .device
            .newResidencySetWithDescriptor_error(&desc)
            .expect("Metal: residency set 생성 실패");
        ctx.queue.addResidencySet(&set);
        let budget_bytes = std::env::var("RNB_METAL_WEIGHT_RESIDENCY_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or_else(|| ctx.device.recommendedMaxWorkingSetSize() / 2);
        Self {
            set,
            entries: HashMap::new(),
            queue: std::collections::BinaryHeap::new(),
            total_bytes: 0,
            budget_bytes,
            tick: 0,
            dirty: false,
        }
    }

    /// 재사용이면 last-use 만 갱신, 신규면 set 에 등록. 예산 초과분은 lazy-heap
    /// LRU 로 evict (stale tick 은 pop 시 버림 — amortized O(log n)).
    fn touch(&mut self, key: ResidentKey, buf: &Retained<ProtocolObject<dyn MTLBuffer>>) {
        self.tick += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.2 = self.tick;
            self.queue.push(std::cmp::Reverse((self.tick, key)));
            self.maybe_compact_queue();
            return;
        }
        let bytes = buf.length() as u64;
        self.set.addAllocation(ProtocolObject::from_ref(&**buf));
        self.entries.insert(key, (buf.clone(), bytes, self.tick));
        self.queue.push(std::cmp::Reverse((self.tick, key)));
        self.total_bytes += bytes;
        self.dirty = true;
        while self.total_bytes > self.budget_bytes && self.entries.len() > 1 {
            let Some(std::cmp::Reverse((tick, victim))) = self.queue.pop() else {
                break;
            };
            match self.entries.get(&victim) {
                // 최신 tick 과 일치할 때만 진짜 LRU — 아니면 stale 항목.
                Some(entry) if entry.2 == tick => {
                    let (buf, bytes, _) = self.entries.remove(&victim).expect("victim exists");
                    self.set.removeAllocation(ProtocolObject::from_ref(&*buf));
                    self.total_bytes -= bytes;
                }
                _ => {}
            }
        }
        self.maybe_compact_queue();
    }

    /// stale 항목 누적으로 큐가 entries 의 4배를 넘으면 최신 tick 만으로 재구성.
    fn maybe_compact_queue(&mut self) {
        if self.queue.len() > self.entries.len().saturating_mul(4) + 1024 {
            self.queue = self
                .entries
                .iter()
                .map(|(&key, entry)| std::cmp::Reverse((entry.2, key)))
                .collect();
        }
    }

    /// wrap batch 끝에서 add/remove 반영. dirty 아닐 때 no-op.
    fn commit_if_dirty(&mut self) {
        if self.dirty {
            self.set.commit();
            self.dirty = false;
        }
    }
}

#[cfg(target_os = "macos")]
fn qwen_moe_combined_shared_sparse_routes(
    shared_token_ids: &[u32],
    shared_route_weights: &[f32],
    sparse_token_ids: &[u32],
    sparse_route_weights: &[f32],
) -> (Vec<u32>, Vec<f32>, u32) {
    assert_eq!(
        shared_token_ids.len(),
        shared_route_weights.len(),
        "Qwen MoE shared route/token length"
    );
    assert_eq!(
        sparse_token_ids.len(),
        sparse_route_weights.len(),
        "Qwen MoE sparse route/token length"
    );
    let sparse_base = u32::try_from(shared_token_ids.len())
        .expect("Qwen MoE combined shared route count overflow");
    let mut token_ids = Vec::with_capacity(shared_token_ids.len() + sparse_token_ids.len());
    token_ids.extend_from_slice(shared_token_ids);
    token_ids.extend_from_slice(sparse_token_ids);
    let mut route_weights =
        Vec::with_capacity(shared_route_weights.len() + sparse_route_weights.len());
    route_weights.extend_from_slice(shared_route_weights);
    route_weights.extend_from_slice(sparse_route_weights);
    (token_ids, route_weights, sparse_base)
}

#[cfg(target_os = "macos")]
fn qwen_moe_prefill_shared_combined_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_SHARED_COMBINED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(target_os = "macos")]
fn qwen_moe_prefill_should_upload_norm(
    gather_enabled: bool,
    id_matmul_requested: bool,
    id_gate_up_requested: bool,
    id_primitive_requested: bool,
    mulmmid_v3_requested: bool,
    mulmmid_v4_requested: bool,
) -> bool {
    gather_enabled
        || id_matmul_requested
        || id_gate_up_requested
        || id_primitive_requested
        || mulmmid_v3_requested
        || mulmmid_v4_requested
}

#[cfg(target_os = "macos")]
fn qwen_moe_prefill_mulmmid_v3_scratch_budget_bytes() -> usize {
    const DEFAULT_MB: usize = 512;
    let mb = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_SCRATCH_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MB);
    mb.saturating_mul(1024 * 1024)
}

#[cfg(target_os = "macos")]
const QWEN_MOE_LLAMA_ID_SCRATCH_BUDGET_BYTES: usize = 768 * 1024 * 1024;

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_quant_supported(
    sparse: QwenMoeLlamaIdQuantSet,
    shared: QwenMoeLlamaIdQuantSet,
) -> bool {
    let sparse_supported = sparse.gate == QwenMoeLlamaIdQuant::Q4K
        && sparse.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            sparse.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q5K | QwenMoeLlamaIdQuant::Q6K
        );
    let shared_q8 = shared
        == (QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        });
    let shared_q4 = shared.gate == QwenMoeLlamaIdQuant::Q4K
        && shared.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            shared.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q6K
        );
    sparse_supported && (shared_q8 || shared_q4)
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_weight_bytes(
    quant: QwenMoeLlamaIdQuant,
    input_dim: usize,
    output_dim: usize,
) -> Result<usize, ffn_chain::QwenMoeLlamaIdError> {
    let (block_elements, block_bytes) = match quant {
        QwenMoeLlamaIdQuant::Q4K => (256, 144),
        QwenMoeLlamaIdQuant::Q5K => (256, 176),
        QwenMoeLlamaIdQuant::Q6K => (256, 210),
        QwenMoeLlamaIdQuant::Q8Zero => (32, 34),
    };
    if input_dim == 0 || output_dim == 0 || input_dim % block_elements != 0 {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidShape);
    }
    output_dim
        .checked_mul(input_dim / block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(ffn_chain::QwenMoeLlamaIdError::InvalidShape)
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_prepare(
    request: &QwenMoeLlamaIdPrefillRequest<'_>,
    pipeline_available: bool,
    scratch_budget_bytes: usize,
) -> Result<ffn_chain::QwenMoeLlamaIdPreflight, ffn_chain::QwenMoeLlamaIdError> {
    if !qwen_moe_llama_id_quant_supported(request.sparse_quant, request.shared_quant) {
        return Ok(ffn_chain::QwenMoeLlamaIdPreflight::Fallback(
            ffn_chain::QwenMoeLlamaIdFallbackReason::UnsupportedQuant,
        ));
    }
    if request.seq_len == 0 || request.hidden_dim == 0 || request.ffn_dim == 0 {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidShape);
    }

    let expected_gate = qwen_moe_llama_id_weight_bytes(
        request.sparse_quant.gate,
        request.hidden_dim,
        request.ffn_dim,
    )?;
    let expected_up = qwen_moe_llama_id_weight_bytes(
        request.sparse_quant.up,
        request.hidden_dim,
        request.ffn_dim,
    )?;
    let expected_down = qwen_moe_llama_id_weight_bytes(
        request.sparse_quant.down,
        request.ffn_dim,
        request.hidden_dim,
    )?;
    if request.gate_expert_bytes != expected_gate
        || request.up_expert_bytes != expected_up
        || request.down_expert_bytes != expected_down
        || request.gate_all.is_empty()
        || request.gate_all.len() % request.gate_expert_bytes != 0
        || request.up_all.len() % request.up_expert_bytes != 0
        || request.down_all.len() % request.down_expert_bytes != 0
    {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidExpertArena);
    }
    let n_expert = request.gate_all.len() / request.gate_expert_bytes;
    if n_expert == 0
        || request.up_all.len() / request.up_expert_bytes != n_expert
        || request.down_all.len() / request.down_expert_bytes != n_expert
    {
        return Err(ffn_chain::QwenMoeLlamaIdError::ExpertArenaMismatch);
    }

    let shared_gate_bytes = qwen_moe_llama_id_weight_bytes(
        request.shared_quant.gate,
        request.hidden_dim,
        request.ffn_dim,
    )?;
    let shared_up_bytes = qwen_moe_llama_id_weight_bytes(
        request.shared_quant.up,
        request.hidden_dim,
        request.ffn_dim,
    )?;
    let shared_down_bytes = qwen_moe_llama_id_weight_bytes(
        request.shared_quant.down,
        request.ffn_dim,
        request.hidden_dim,
    )?;
    if request.shared_gate.len() != shared_gate_bytes
        || request.shared_up.len() != shared_up_bytes
        || request.shared_down.len() != shared_down_bytes
    {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidExpertArena);
    }

    if request.selected_experts.is_empty() {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidShape);
    }
    if request.selected_experts.len() % request.seq_len != 0 {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    let n_expert_used = request.selected_experts.len() / request.seq_len;
    if n_expert_used == 0 || request.route_weights.len() != request.selected_experts.len() {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    if request.shared_route_weights.len() != request.seq_len {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    let norm_elements = request
        .seq_len
        .checked_mul(request.hidden_dim)
        .ok_or(ffn_chain::QwenMoeLlamaIdError::InvalidShape)?;
    if request.norm_all.len() != norm_elements {
        return Err(ffn_chain::QwenMoeLlamaIdError::InvalidShape);
    }
    for token_routes in request.selected_experts.chunks_exact(n_expert_used) {
        for (rank, &expert) in token_routes.iter().enumerate() {
            if expert as usize >= n_expert {
                return Err(ffn_chain::QwenMoeLlamaIdError::ExpertIdOutOfRange);
            }
            if token_routes[..rank].contains(&expert) {
                return Err(ffn_chain::QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
            }
        }
    }

    ffn_chain::qwen_moe_llama_id_preflight(
        pipeline_available,
        true,
        request.seq_len,
        n_expert,
        n_expert_used,
        request.hidden_dim,
        request.ffn_dim,
        scratch_budget_bytes,
    )
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_pipelines_available(
    ctx: &compute::MetalContext,
    sparse_quant: QwenMoeLlamaIdQuantSet,
    shared_quant: QwenMoeLlamaIdQuantSet,
) -> bool {
    let sparse_down_available = match sparse_quant.down {
        QwenMoeLlamaIdQuant::Q4K => true,
        QwenMoeLlamaIdQuant::Q5K => ctx.qwen_moe_llama_mul_mm_id_q5k_f32_pipeline().is_some(),
        QwenMoeLlamaIdQuant::Q6K => ctx.qwen_moe_llama_mul_mm_id_q6k_f32_pipeline().is_some(),
        QwenMoeLlamaIdQuant::Q8Zero => false,
    };
    let shared_available = match shared_quant {
        QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        } => ctx
            .qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline()
            .is_some(),
        QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down,
        } if matches!(down, QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q6K) => {
            ctx.qwen_moe_shared_mul_mm_q4k_f32_pipeline().is_some()
                && match down {
                    QwenMoeLlamaIdQuant::Q4K => {
                        ctx.qwen_moe_shared_mul_mm_q4k_f32_pipeline().is_some()
                    }
                    QwenMoeLlamaIdQuant::Q6K => {
                        ctx.qwen_moe_shared_mul_mm_q6k_f32_pipeline().is_some()
                    }
                    _ => false,
                }
        }
        _ => false,
    };
    ctx.qwen_moe_llama_id_map0_pipeline().is_some()
        && ctx.qwen_moe_llama_mul_mm_id_q4k_f32_pipeline().is_some()
        && sparse_down_available
        && shared_available
        && ctx.qwen_moe_llama_quantize_q8k_f32_pipeline().is_some()
        && ctx.qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline().is_some()
        && (sparse_quant.down != QwenMoeLlamaIdQuant::Q6K
            || ctx.qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline().is_some())
        && ctx.qwen_moe_chain_cast_large_slots_pipeline().is_some()
        && ctx.qwen_moe_llama_swiglu_f32_pipeline().is_some()
        && ctx
            .qwen_moe_llama_weighted_rank_reduce_f32_pipeline()
            .is_some()
        && ctx
            .qwen_moe_llama_expert_order_reduce_f32_pipeline()
            .is_some()
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_upload_f32(
    ctx: &compute::MetalContext,
    values: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, ffn_chain::QwenMoeLlamaIdError> {
    let ptr = std::ptr::NonNull::new(values.as_ptr() as *mut std::ffi::c_void)
        .ok_or(ffn_chain::QwenMoeLlamaIdError::DeviceMapFailed)?;
    let bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ffn_chain::QwenMoeLlamaIdError::InvalidShape)?;
    unsafe {
        ctx.device
            .newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(ffn_chain::QwenMoeLlamaIdError::AllocationFailed)
    }
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_command_result(
    status: MTLCommandBufferStatus,
    error: Option<String>,
) -> Result<(), String> {
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(format!(
        "Metal Qwen MoE llama ID prefill command failed status={status:?} error={}",
        error.unwrap_or_else(|| "no NSError attached".to_string())
    ))
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_error(error: ffn_chain::QwenMoeLlamaIdError) -> String {
    format!("Metal Qwen MoE llama ID prefill failed: {error:?}")
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_fallback_reason(
    reason: ffn_chain::QwenMoeLlamaIdFallbackReason,
) -> &'static str {
    match reason {
        ffn_chain::QwenMoeLlamaIdFallbackReason::PipelineUnavailable => "pipeline_unavailable",
        ffn_chain::QwenMoeLlamaIdFallbackReason::UnsupportedQuant => "unsupported_quant",
        ffn_chain::QwenMoeLlamaIdFallbackReason::ScratchOverBudget => "scratch_over_budget",
    }
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_trace_value_truthy(value: Option<&str>) -> bool {
    value
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_trace_enabled() -> bool {
    let value = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TRACE").ok();
    qwen_moe_llama_id_trace_value_truthy(value.as_deref())
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_enabled() -> bool {
    let value = std::env::var("RNB_METAL_QWEN_PREFILL_CHAIN_TRACE").ok();
    qwen_moe_llama_id_trace_value_truthy(value.as_deref())
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_ms(value: Option<f64>) -> String {
    match value.filter(|value| value.is_finite() && *value > 0.0) {
        Some(value) if value < 0.001 => "<0.001".to_string(),
        Some(value) => format!("{value:.3}"),
        None => "unmeasured".to_string(),
    }
}
#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_stage_ms(value: Option<f64>, unsupported_reason: Option<&str>) -> String {
    match value {
        Some(value) => qwen_moe_llama_id_ms(Some(value)),
        None => unsupported_reason
            .map(|reason| format!("unsupported({reason})"))
            .unwrap_or_else(|| "unmeasured".to_string()),
    }
}

#[cfg(target_os = "macos")]
fn qwen_moe_llama_id_quant_name(quant: QwenMoeLlamaIdQuant) -> &'static str {
    match quant {
        QwenMoeLlamaIdQuant::Q4K => "q4_k",
        QwenMoeLlamaIdQuant::Q5K => "q5_k",
        QwenMoeLlamaIdQuant::Q6K => "q6_k",
        QwenMoeLlamaIdQuant::Q8Zero => "q8_0",
    }
}

#[cfg(target_os = "macos")]
fn format_qwen_moe_llama_id_fallback_trace(reason: &str, wrapper_wall_ms: f64) -> String {
    format!(
        "[metal-qwen-moe-prefill-accum] mode=llama_id fallback_reason={reason} gpu_ms=unmeasured wrapper_wall_ms={} readback_ms=unmeasured",
        qwen_moe_llama_id_ms(Some(wrapper_wall_ms)),
    )
}

#[cfg(target_os = "macos")]
fn format_qwen_moe_llama_id_success_trace(
    trace: ffn_chain::QwenMoeLlamaIdTrace,
    layer_idx: usize,
    sparse_quant: QwenMoeLlamaIdQuantSet,
    shared_quant: QwenMoeLlamaIdQuantSet,
    gpu_ms: Option<f64>,
    wrapper_wall_ms: f64,
    readback_ms: f64,
) -> String {
    let dispatch_count = trace.map_dispatches
        + trace.gate_dispatches
        + trace.up_dispatches
        + trace.activation_dispatches
        + trace.down_dispatches
        + trace.reduce_dispatches;
    format!(
        "[metal-qwen-moe-prefill-accum] mode=llama_id fallback_reason=none layer_idx={layer_idx} sparse_gate_quant={} sparse_up_quant={} sparse_down_quant={} shared_gate_quant={} shared_up_quant={} shared_down_quant={} kernel_dtype=f32 weights=raw trace_scope=diagnostic exclude_from_d20=true stage_timing_source={} stage_timing_fallback_reason={} stage_timing_unsupported_reason={} dispatch_count={dispatch_count} map={} gate={} up={} activation={} down={} reduce={} scratch_bytes={} gpu_ms={} wrapper_wall_ms={} readback_ms={} map_ms={} gate_ms={} up_ms={} activation_ms={} down_ms={} reduce_ms={}",
        qwen_moe_llama_id_quant_name(sparse_quant.gate),
        qwen_moe_llama_id_quant_name(sparse_quant.up),
        qwen_moe_llama_id_quant_name(sparse_quant.down),
        qwen_moe_llama_id_quant_name(shared_quant.gate),
        qwen_moe_llama_id_quant_name(shared_quant.up),
        qwen_moe_llama_id_quant_name(shared_quant.down),
        trace.stage_timing_source.unwrap_or("unsupported"),
        trace.stage_timing_fallback_reason.unwrap_or("none"),
        trace.stage_timing_unsupported_reason.unwrap_or("none"),
        trace.map_dispatches,
        trace.gate_dispatches,
        trace.up_dispatches,
        trace.activation_dispatches,
        trace.down_dispatches,
        trace.reduce_dispatches,
        trace.scratch_bytes,
        qwen_moe_llama_id_ms(gpu_ms),
        qwen_moe_llama_id_ms(Some(wrapper_wall_ms)),
        qwen_moe_llama_id_ms(Some(readback_ms)),
        qwen_moe_llama_id_stage_ms(trace.map_ms, trace.stage_timing_unsupported_reason),
        qwen_moe_llama_id_stage_ms(trace.gate_ms, trace.stage_timing_unsupported_reason),
        qwen_moe_llama_id_stage_ms(trace.up_ms, trace.stage_timing_unsupported_reason),
        qwen_moe_llama_id_stage_ms(trace.activation_ms, trace.stage_timing_unsupported_reason),
        qwen_moe_llama_id_stage_ms(trace.down_ms, trace.stage_timing_unsupported_reason),
        qwen_moe_llama_id_stage_ms(trace.reduce_ms, trace.stage_timing_unsupported_reason),
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn format_qwen_moe_mulmmid_v4_trace(
    shared_preserved: bool,
    slots: usize,
    seq_len: usize,
    hidden_dim: usize,
    n_expert: usize,
    n_expert_used: usize,
    id_block_count: usize,
    rank_block_count: usize,
    gpu_ms: f64,
    wall_ms: f64,
    encode_wait_ms: f64,
    scratch: ffn_chain::QwenMoeMulMmIdV4Scratch,
) -> String {
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let dispatch_count = 3 + n_expert_used;
    let removed_down_scratch_bytes = slots
        .saturating_mul(hidden_dim)
        .saturating_mul(std::mem::size_of::<f32>());
    format!(
        "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason=none dispatch_count={dispatch_count} shared_preserved={shared_preserved} slots={slots} seq_len={seq_len} n_expert_used={n_expert_used} n_expert={n_expert} rank_dispatch_count={n_expert_used} rank_count={n_expert_used} id_block_count={id_block_count} rank_block_count={rank_block_count} gate_ms=unmeasured up_ms=unmeasured silu_ms=unmeasured rank_down_scatter_ms=unmeasured gpu_ms={gpu_ms:.3} wall_ms={wall_ms:.3} encode_wait_ms={encode_wait_ms:.3} scratch_mib={:.1} gate_up_scratch_mib={:.1} gate_mib={:.1} up_mib={:.1} down_scratch_mib={:.1} down_mib={:.1} metadata_scratch_mib={:.1} metadata_mib={:.1} removed_down_scratch_mib={:.1}",
        mib(scratch.total_bytes),
        mib(scratch.gate_bytes + scratch.up_bytes),
        mib(scratch.gate_bytes),
        mib(scratch.up_bytes),
        mib(scratch.down_bytes),
        mib(scratch.down_bytes),
        mib(scratch.metadata_bytes),
        mib(scratch.metadata_bytes),
        mib(removed_down_scratch_bytes),
    )
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
struct QwenMoeV4AttemptState {
    requested: bool,
    preflight_failed: bool,
}

#[cfg(target_os = "macos")]
impl QwenMoeV4AttemptState {
    fn new(requested: bool) -> Self {
        Self {
            requested,
            preflight_failed: false,
        }
    }

    fn record_preflight_failure(&mut self) {
        self.preflight_failed = true;
    }

    fn skip_id_paths(self) -> bool {
        self.requested && self.preflight_failed
    }
}

// 연속 GDN run chain 의 per-layer spec. seam(rnb-runtime metal_inference)이
// `gdn_decode_chain_run` 호출에 쓰도록 crate 밖으로 노출(1.3/1.4 연결).
#[cfg(target_os = "macos")]
pub use gdn_chain::GdnChainSpecRef;

// 2단계: attn carrier chain spec. seam 이 `decode_chain_run`(attn+gdn 합류) 호출에 쓴다.
#[cfg(target_os = "macos")]
pub use attn_chain::AttnChainSpecRef;

/// chain run 의 per-layer spec — attn carrier / GDN carrier 둘 다 단일 command buffer 에
/// 합류시키기 위한 enum. `decode_chain_run` 이 spec 종류에 따라 attn/gdn encode 를 같은
/// encoder 에 인코드한다. hidden 은 공유 device buffer 라 attn↔gdn 경계도 host 안 거친다.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub enum ChainLayerSpecRef<'a> {
    Attn(AttnChainSpecRef<'a>),
    AttnMoeQwen(AttnMoeQwenChainSpecRef<'a>),
    Gdn(GdnChainSpecRef<'a>),
    GdnMoeQwen(GdnMoeQwenChainSpecRef<'a>),
}

/// Attention core와 Qwen MoE FFN을 같은 decode encoder에 이어 붙이는 per-layer spec.
/// q/k/o와 sparse/shared gate/up은 Q4_K, v는 Q4_K|Q6_K,
/// sparse/shared down은 동일한 Q4_K|Q5_K|Q6_K quant code를 사용한다.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct AttnMoeQwenChainSpecRef<'a> {
    pub layer: usize,
    pub norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub q_raw: &'a [u8],
    pub k_raw: &'a [u8],
    pub v_raw: &'a [u8],
    pub o_raw: &'a [u8],
    pub router_w: &'a [f32],
    pub gate_exps_raw: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_exps_raw: &'a [u8],
    pub up_expert_bytes: usize,
    pub down_exps_raw: &'a [u8],
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate_raw: &'a [u8],
    pub shared_up_raw: &'a [u8],
    pub shared_down_raw: &'a [u8],
    pub q_q: u8,
    pub k_q: u8,
    pub v_q: u8,
    pub o_q: u8,
    pub down_quant: u8,
    pub shared_q8_0: bool,
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub kvarn: Option<rnb_backend_api::KvarnChainView<'a>>,
    pub pos: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub capacity: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct GdnMoeQwenChainSpecRef<'a> {
    pub layer: usize,
    pub conv_state: &'a [f32],
    pub delta_state: &'a [f32],
    pub attn_norm_weight: &'a [f32],
    pub dt_bias_weight: &'a [f32],
    pub ssm_a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub ssm_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub qkv_raw: &'a [u8],
    pub gate_raw: &'a [u8],
    pub alpha_raw: &'a [u8],
    pub beta_raw: &'a [u8],
    pub ssm_out_raw: &'a [u8],
    pub router_w: &'a [f32],
    pub gate_exps_raw: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_exps_raw: &'a [u8],
    pub up_expert_bytes: usize,
    pub down_exps_raw: &'a [u8],
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate_raw: &'a [u8],
    pub shared_up_raw: &'a [u8],
    pub shared_down_raw: &'a [u8],
    pub qkv_q: u8,
    pub gate_q: u8,
    pub alpha_q: u8,
    pub beta_q: u8,
    pub ssm_out_q: u8,
    pub down_quant: u8,
    pub shared_q8_0: bool,
    pub hidden_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub z_dim: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub eps: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct DecodeOutputArgmaxSpecRef<'a> {
    pub norm_weight: &'a [f32],
    pub output_raw: &'a [u8],
    pub output_quant: u8,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeChainOptions {
    pub collect_timing: bool,
    pub delta_resident: bool,
}

impl Default for DecodeChainOptions {
    fn default() -> Self {
        Self {
            collect_timing: false,
            delta_resident: true,
        }
    }
}

#[cfg(target_os = "macos")]
type QwenMoeDecodeChainCarrierKey = (usize, usize, usize, usize, usize, usize);

#[cfg(target_os = "macos")]
fn qwen_moe_decode_chain_carrier_key(
    layer: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    n_ff: usize,
    n_expert: usize,
    down_quant: u8,
) -> QwenMoeDecodeChainCarrierKey {
    (
        layer,
        n_expert_used,
        hidden_dim,
        n_ff,
        n_expert,
        down_quant as usize,
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutputArgmaxReport {
    pub attempted: bool,
    pub did_run: bool,
    pub token_id: Option<u32>,
    pub cmd_buffers: usize,
    pub wait_us: u64,
    pub readback_bytes: usize,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeChainReport {
    pub did_run: bool,
    pub pm51_extension_hit: bool,
    pub legacy_chain_hit: bool,
    pub qwen_moe_layers: usize,
    pub fallback_reason: Option<&'static str>,
    pub cmd_buffers: usize,
    pub wait_us: u64,
    pub upload_bytes: usize,
    pub readback_bytes: usize,
    pub state_syncs: usize,
    pub hidden_readback_bytes: usize,
    pub argmax_only: bool,
    pub output_argmax: OutputArgmaxReport,
}

impl DecodeChainReport {
    pub fn fallback(reason: &'static str) -> Self {
        Self {
            fallback_reason: Some(reason),
            ..Self::default()
        }
    }

    pub fn total_cmd_buffers(&self) -> usize {
        self.cmd_buffers + self.output_argmax.cmd_buffers
    }

    pub fn total_wait_us(&self) -> u64 {
        self.wait_us + self.output_argmax.wait_us
    }

    pub fn total_readback_bytes(&self) -> usize {
        self.readback_bytes + self.output_argmax.readback_bytes
    }
}

#[cfg(target_os = "macos")]
type PrefillGdnFullFfnCarrierKey = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
    u32,
);

#[cfg(target_os = "macos")]
type QwenGdnPrefillCarrierKey = (usize, usize, usize, usize, usize, usize, usize, u32);

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QwenAtnOTailCarrierKey {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    n_rot: usize,
    rope_theta_bits: u32,
    scale_bits: u32,
    norm_eps_bits: u32,
    pos_start: usize,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AtnCoreKey {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    n_rot: usize,
    rope_theta_bits: u32,
    scale_bits: u32,
    norm_eps_bits: u32,
    pos_start: usize,
    q_quant: TensoropsQuant,
    k_quant: TensoropsQuant,
    v_quant: TensoropsQuant,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AtnFullLayerKey {
    core: AtnCoreKey,
    ffn_dim: usize,
    o_quant: TensoropsQuant,
    ffn_gate_quant: TensoropsQuant,
    ffn_up_quant: TensoropsQuant,
    ffn_down_quant: TensoropsQuant,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AtnOTailKey {
    core: AtnCoreKey,
    o_quant: TensoropsQuant,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnCoreWeightView<'a> {
    pub raw: &'a [u8],
    pub quant: TensoropsQuant,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnCoreBackendRequest<'a> {
    pub hidden: &'a [f32],
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_weight: PrefillAtnCoreWeightView<'a>,
    pub k_weight: PrefillAtnCoreWeightView<'a>,
    pub v_weight: PrefillAtnCoreWeightView<'a>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub pos_start: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnFullLayerBackendRequest<'a> {
    pub core: PrefillAtnCoreBackendRequest<'a>,
    pub o_weight: PrefillAtnCoreWeightView<'a>,
    pub ffn_norm_w: &'a [f32],
    pub ffn_gate_weight: PrefillAtnCoreWeightView<'a>,
    pub ffn_up_weight: PrefillAtnCoreWeightView<'a>,
    pub ffn_down_weight: PrefillAtnCoreWeightView<'a>,
    pub ffn_dim: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnOTailBackendRequest<'a> {
    pub core: PrefillAtnCoreBackendRequest<'a>,
    pub o_weight: PrefillAtnCoreWeightView<'a>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnCoreBackendSpecRef<'a> {
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_weight: PrefillAtnCoreWeightView<'a>,
    pub k_weight: PrefillAtnCoreWeightView<'a>,
    pub v_weight: PrefillAtnCoreWeightView<'a>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub pos_start: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub enum GdnBackendWeightRef<'a> {
    F32 {
        values: &'a [f32],
        rows: usize,
        cols: usize,
    },
    Quant(PrefillAtnCoreWeightView<'a>),
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct PrefillAtnOTailBackendSpecRef<'a> {
    pub core: PrefillAtnCoreBackendSpecRef<'a>,
    pub o_weight: PrefillAtnCoreWeightView<'a>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct QwenPrefillGdnBackendSpecRef<'a> {
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub dt_rank: usize,
    pub conv_kernel_size: usize,
    pub attn_norm_w: &'a [f32],
    pub qkv_weight: GdnBackendWeightRef<'a>,
    pub gate_weight: GdnBackendWeightRef<'a>,
    pub alpha_weight: GdnBackendWeightRef<'a>,
    pub beta_weight: GdnBackendWeightRef<'a>,
    pub conv_state: &'a [f32],
    pub conv_kernel: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub delta_state: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_weight: GdnBackendWeightRef<'a>,
    pub post_attn_norm_w: &'a [f32],
    pub norm_eps: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenRouteAlgorithm {
    SelectedSoftmaxTopKLowerExpertTieV1,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub struct QwenMoePrefillBackendSpecRef<'a> {
    pub ffn_norm_w: &'a [f32],
    pub norm_eps: f32,
    pub router_w: &'a [f32],
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_expert_bytes: usize,
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub sparse_quant: QwenMoeLlamaIdQuantSet,
    pub shared_quant: QwenMoeLlamaIdQuantSet,
    pub route_algorithm: QwenRouteAlgorithm,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub enum QwenPrefillChainSpecRef<'a> {
    Attention {
        layer_idx: usize,
        core: PrefillAtnOTailBackendSpecRef<'a>,
        moe: QwenMoePrefillBackendSpecRef<'a>,
    },
    Gdn {
        layer_idx: usize,
        layer: QwenPrefillGdnBackendSpecRef<'a>,
        moe: QwenMoePrefillBackendSpecRef<'a>,
    },
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct QwenPrefillChainBackendOut {
    pub hidden: Vec<f32>,
    pub attention_kv: Vec<(usize, Vec<u16>, Vec<u16>)>,
    pub gdn_states: Vec<(usize, Vec<f32>, Vec<f32>)>,
    pub hidden_uploads: usize,
    pub hidden_readbacks: usize,
    pub intermediate_hidden_transfers: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct QwenPrefillChainTransferLedger {
    hidden_uploads: usize,
    hidden_readbacks: usize,
    intermediate_hidden_transfers: usize,
}

#[cfg(target_os = "macos")]
impl QwenPrefillChainTransferLedger {
    fn record_initial_hidden_upload(&mut self) {
        self.hidden_uploads += 1;
    }

    fn record_final_hidden_readback(&mut self) {
        self.hidden_readbacks += 1;
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_layer_index(spec: &QwenPrefillChainSpecRef<'_>) -> usize {
    match spec {
        QwenPrefillChainSpecRef::Attention { layer_idx, .. }
        | QwenPrefillChainSpecRef::Gdn { layer_idx, .. } => *layer_idx,
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_layer_indices_supported(
    specs: &[QwenPrefillChainSpecRef<'_>],
) -> Result<bool, String> {
    for pair in specs.windows(2) {
        let previous = qwen_prefill_chain_layer_index(&pair[0]);
        let current = qwen_prefill_chain_layer_index(&pair[1]);
        if current == previous {
            return Err(format!(
                "Metal Qwen prefill chain duplicate layer index {current}"
            ));
        }
        if previous.checked_add(1) != Some(current) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(all(test, target_os = "macos"))]
mod tests_fixture;

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

/// Metal 추론 백엔드.
///
/// `new()` 에서 device / command queue / Q4_K pipeline 을 한 번만 빌드한다.
/// 이후 `gemv_q4k` 호출마다 per-call device open / MSL compile 비용이 없다.
///
/// `Retained<ProtocolObject<dyn MTL*>>` 은 `!Send + !Sync` 이므로 이 구조체도
/// `Send/Sync` 가 아니다. Backend 는 단일 스레드에서 사용하는 한 문제없다.
#[cfg(target_os = "macos")]
pub struct MetalBackend {
    device_name: Option<String>,
    ctx: Option<compute::MetalContext>,
    /// weight slice identity(`raw.as_ptr() as usize`, `raw.len()`) → (NoCopy buffer, page_offset).
    /// 같은 weight 재호출 시 wrap 생략(zero-copy residency). `MTLBuffer` 가
    /// `!Send+!Sync` 라 `RefCell` + 단일 스레드(thread_local) 사용 전제.
    resident: RefCell<HashMap<ResidentKey, (Retained<ProtocolObject<dyn MTLBuffer>>, u32)>>,
    /// Immutable small constants copied once and reused by the whole-model decode chain.
    constant_f32: RefCell<HashMap<ResidentKey, Retained<ProtocolObject<dyn MTLBuffer>>>>,
    constant_u32: RefCell<HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// One shared hidden buffer per model width; decode calls are serialized on this backend.
    decode_chain_hidden: RefCell<HashMap<usize, Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// pm112: GLM MLA decode chain scratch. shape 키 별 1회 alloc.
    glm_mla_carriers: RefCell<HashMap<(usize, usize, usize, usize), ffn_chain::GlmMlaCarrier>>,
    /// pm112: wrap 된 weight buffer 를 queue-attached MTLResidencySet 에 등록해
    /// 매 command buffer submit 의 wiring(drv) 비용을 줄인다. 예산 초과 시 LRU evict
    /// (무제한 wiring 은 RAM 스래싱 — 20토큰 GLM 런에서 wired 38GB 관측).
    /// 기본 ON (MLA dense GEMV 경로 의존), `RNB_METAL_WEIGHT_RESIDENCY=0` opt-out.
    weight_residency: RefCell<Option<WeightResidencyLru>>,
    /// (hidden_dim, ffn_dim) → FfnCarrier. shape 별 1회 alloc 후 재사용.
    ffn_carriers: RefCell<HashMap<(usize, usize), ffn_chain::FfnCarrier>>,
    /// (hidden_dim, ffn_dim, seq_len) → PrefillFfnCarrier. pm33 prefill FFN chain(M>1).
    prefill_ffn_carriers: RefCell<HashMap<(usize, usize, usize), ffn_chain::PrefillFfnCarrier>>,
    /// (hidden_dim, ffn_dim, seq_len) → Qwen MoE gather-only private intermediate FFN carrier.
    qwen_moe_prefill_private_ffn_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::PrefillFfnCarrier>>,
    /// (hidden_dim, ffn_dim, seq_len) → Qwen MoE sparse prefill accumulator. Opt-in only.
    qwen_moe_prefill_accum_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::QwenMoePrefillAccumCarrier>>,
    /// (slots, hidden_dim, ffn_dim) → Qwen MoE id-matmul prefill scratch. Opt-in only.
    qwen_moe_prefill_id_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::QwenMoePrefillIdCarrier>>,
    /// Full allocation plan → idle llama-ID carriers. A carrier is removed while live.
    qwen_moe_llama_id_carrier_pool:
        RefCell<HashMap<ffn_chain::QwenMoeLlamaIdPlan, Vec<ffn_chain::QwenMoeLlamaIdCarrier>>>,
    qwen_prefill_chain_moe_carrier_pool:
        RefCell<HashMap<ffn_chain::QwenMoeLlamaIdPlan, Vec<ffn_chain::QwenMoeLlamaIdCarrier>>>,
    /// (slots, hidden_dim, ffn_dim) → Qwen MoE id gate/up prefill scratch. Opt-in only.
    qwen_moe_prefill_id_gate_up_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::QwenMoePrefillIdGateUpCarrier>>,
    /// (slots, hidden_dim, ffn_dim) → Qwen MoE id gate/up f16 prefill scratch. Opt-in only.
    qwen_moe_prefill_id_gate_up_f16_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::QwenMoePrefillIdGateUpF16Carrier>>,
    /// (seq_len, n_used, hidden_dim, ffn_dim, n_expert, block_count, budget) → Qwen MoE mul-mm-id v3 scratch.
    qwen_moe_prefill_mulmmid_v3_carriers: RefCell<
        HashMap<
            (usize, usize, usize, usize, usize, usize, usize),
            ffn_chain::QwenMoeMulMmIdV3Carrier,
        >,
    >,
    /// (seq_len, n_used, hidden_dim, ffn_dim, n_expert, id_blocks, rank_blocks, budget) → Qwen MoE mul-mm-id v4 scratch.
    qwen_moe_prefill_mulmmid_v4_carriers: RefCell<
        HashMap<
            (usize, usize, usize, usize, usize, usize, usize, usize),
            ffn_chain::QwenMoeMulMmIdV4Carrier,
        >,
    >,
    /// (slots, hidden_dim, ffn_dim) → Qwen MoE decode batch carrier. Opt-in only.
    qwen_moe_decode_carriers:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::QwenMoeDecodeCarrier>>,
    /// pm115 M1: GLM prefill direct-file 경로 — mmap wiring page fault 를 pread 로
    /// 우회. 파일 핸들 캐시(F_NOCACHE) + 재사용 스테이징 buffer (grow-only).
    glm_direct_file_handles: RefCell<HashMap<std::path::PathBuf, std::sync::Arc<std::fs::File>>>,
    /// slot → (staging buffer, cap). slot 0 = 통합 경로, 1..=3 = staged 텐서별
    /// (gate/up/down — 이전 스테이지 GPU 가 읽는 동안 덮어쓰지 않도록 분리).
    glm_direct_staging: RefCell<HashMap<usize, (Retained<ProtocolObject<dyn MTLBuffer>>, usize)>>,
    /// pm116 M2: (slots, n_embd, n_ff) → 스테이지-major full scratch (seq grow-only).
    glm_prefill_stage_scratch:
        RefCell<HashMap<(usize, usize, usize), ffn_chain::GlmPrefillStageScratch>>,
    /// (slots, hidden_dim, ffn_dim, down_quant) → Qwen MoE id-style decode carrier. Opt-in only.
    qwen_moe_decode_id_carriers:
        RefCell<HashMap<(usize, usize, usize, usize), ffn_chain::QwenMoeDecodeIdCarrier>>,
    /// (layer, n_used, hidden_dim, ffn_dim, n_expert, down_quant) → Qwen MoE decode chain tail.
    qwen_moe_decode_chain_carriers:
        RefCell<HashMap<QwenMoeDecodeChainCarrierKey, ffn_chain::QwenMoeDecodeChainCarrier>>,
    /// pm35 M2: prefill GDN in_proj/gate single-GEMM carrier. (hidden, n_out, seq) 키.
    prefill_gdn_proj_carriers:
        RefCell<HashMap<(usize, usize, usize), gdn_proj_chain::PrefillGdnProjCarrier>>,
    /// pm58: prefill GDN F32 alpha/beta dual projection carrier. (hidden, n_out, seq) 키.
    prefill_gdn_f32_dual_carriers:
        RefCell<HashMap<(usize, usize, usize), gdn_proj_chain::PrefillGdnF32DualCarrier>>,
    /// pm58: Qwen MoE router F32 prefill projection carrier. (hidden, n_expert, seq) 키.
    qwen_moe_router_f32_carriers:
        RefCell<HashMap<(usize, usize, usize), gdn_proj_chain::PrefillF32ProjCarrier>>,
    /// pm65: Qwen MoE Metal router top-k output carrier. (seq, n_expert, n_used) 키.
    qwen_moe_route_array_carriers:
        RefCell<HashMap<(usize, usize, usize), QwenMoePrefillRouteArrayCarrier>>,
    /// (vocab_rows, hidden_dim) -> output argmax scratch buffers reused across decode steps.
    output_argmax_carriers: RefCell<HashMap<(usize, usize), compute::OutputArgmaxScratch>>,
    /// pm44 M1-3: prefill GDN gated→ssm_out fused carrier. (d_inner, head_v_dim, n_out, seq) 키.
    prefill_gated_proj_carriers: RefCell<
        HashMap<(usize, usize, usize, usize), gated_proj_chain::PrefillGdnGatedProjCarrier>,
    >,
    /// pm45 M2: prefill GDN conv→delta fused carrier. shape 별 1회 alloc. 키:
    /// (seq, conv_channels, conv_kernel, num_k_heads, num_v_heads, head_k_dim, head_v_dim, chunk_size).
    prefill_gdn_conv_delta_carriers: RefCell<
        HashMap<
            (usize, usize, usize, usize, usize, usize, usize, usize),
            gdn_conv_delta_chain::PrefillGdnConvDeltaCarrier,
        >,
    >,
    /// pm45 M3-1: prefill GDN full chain(conv→delta→gated→ssm_out) fused carrier. 키:
    /// (seq, conv_channels, conv_kernel, num_k_heads, num_v_heads, head_k_dim, head_v_dim, chunk_size, n_out).
    prefill_gdn_full_carriers: RefCell<
        HashMap<
            (
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
            ),
            gdn_conv_delta_chain::PrefillGdnFullCarrier,
        >,
    >,
    /// pm49: prefill GDN full chain + post-attn FFN tail carrier. 키:
    /// (seq, conv shape 7축, hidden_dim, ffn_dim, down_q6, norm_eps_bits).
    prefill_gdn_full_ffn_carriers: RefCell<
        HashMap<PrefillGdnFullFfnCarrierKey, gdn_conv_delta_chain::PrefillGdnFullFfnCarrier>,
    >,
    /// pm48 ②: prefill attention 2차 device chain(rope/qk_norm→cast→flash) carrier. 키:
    /// (seq, num_heads, num_kv_heads, head_dim).
    prefill_attn_chain_carriers:
        RefCell<HashMap<(usize, usize, usize, usize), prefill_attn_chain::PrefillAttnChainCarrier>>,
    /// pm50 M1: prefill dense gated ATN core(q/k/v→flash→gate) carrier.
    prefill_atn_core_carriers:
        RefCell<HashMap<AtnCoreKey, prefill_atn_core_chain::PrefillAtnCoreCarrier>>,
    /// pm70: prefill dense gated ATN full layer(q/k/v→flash→o_proj→FFN) carrier.
    prefill_atn_full_layer_carriers:
        RefCell<HashMap<AtnFullLayerKey, prefill_atn_core_chain::PrefillAtnFullLayerCarrier>>,
    /// pm108: prefill dense gated ATN o-tail(q/k/v→flash→o_proj+residual) carrier.
    prefill_atn_o_tail_carriers:
        RefCell<HashMap<AtnOTailKey, prefill_atn_core_chain::PrefillAtnOTailCarrier>>,
    qwen_prefill_gdn_carriers:
        RefCell<HashMap<QwenGdnPrefillCarrierKey, gdn_chain::QwenGdnPrefillCarrier>>,
    qwen_prefill_atn_o_tail_carriers:
        RefCell<HashMap<QwenAtnOTailCarrierKey, prefill_atn_core_chain::PrefillAtnOTailCarrier>>,
    /// (hidden_dim, q_out_dim, kv_dim) → QkvCarrier. shape 별 1회 alloc 후 재사용.
    qkv_carriers: RefCell<HashMap<(usize, usize, usize), ffn_chain::QkvCarrier>>,
    /// (hidden_dim, qkv_dim, gate_dim) → GdnInprojCarrier. shape 별 1회 alloc 후 재사용.
    gdn_inproj_carriers: RefCell<HashMap<(usize, usize, usize), ffn_chain::GdnInprojCarrier>>,
    /// (hidden_dim, q_dim) → OChainCarrier. shape 별 1회 alloc 후 재사용.
    o_chain_carriers: RefCell<HashMap<(usize, usize), ffn_chain::OChainCarrier>>,
    /// layer → KvResident. KV cache device residency — decode 매 토큰 전체 KV
    /// 업로드 대신 1 token append. attn_decode 가 device buffer 를 직접 읽음.
    kv_residents: RefCell<HashMap<usize, compute::KvResident>>,
    /// layer → compressed KVarN KV device residency.
    kvarn_residents: RefCell<HashMap<usize, compute::KvarnResident>>,
    /// layer → AttnCarrier. attention layer 전체 단일 command buffer carrier.
    /// KV state(KvResident)를 내장해 layer 별로 분리(같은 shape 라도 layer 마다 1개).
    attn_carriers: RefCell<HashMap<usize, attn_chain::AttnCarrier>>,
    /// Qwen MoE attention layer core-only carrier. Dense FFN scratch ownership과 분리한다.
    attn_moe_carriers: RefCell<HashMap<usize, attn_chain::AttnCarrier>>,
    gdn_carriers: RefCell<HashMap<usize, gdn_chain::GdnCarrier>>,
    /// MoE GDN layer core-only carrier. Dense FFN carrier와 shape/state를 분리한다.
    gdn_core_carriers: RefCell<HashMap<usize, gdn_chain::GdnCarrier>>,
    /// milestone 3: batched(B-lane) GDN core carrier. (layer, B) 별 1회 alloc, single-token
    /// `gdn_carriers` 와 분리(프로덕션 single-token 경로 불변). MTP verify body fusion 전용.
    gdn_batch_carriers: RefCell<HashMap<(usize, usize), gdn_chain::GdnBatchCarrier>>,
    /// milestone 4: batched(B-lane) attention core carrier. `attn_moe_carriers`(single-token)
    /// 와 분리(프로덕션 single-token 경로 불변). MTP verify mixed-chain body fusion 전용.
    attn_batch_carriers: RefCell<HashMap<(usize, usize), attn_chain::AttnBatchCarrier>>,
}

#[cfg(target_os = "macos")]
struct QwenMoeLlamaIdCarrierLease<'a> {
    pool:
        &'a RefCell<HashMap<ffn_chain::QwenMoeLlamaIdPlan, Vec<ffn_chain::QwenMoeLlamaIdCarrier>>>,
    key: ffn_chain::QwenMoeLlamaIdPlan,
    carrier: Option<ffn_chain::QwenMoeLlamaIdCarrier>,
}

#[cfg(target_os = "macos")]
impl std::ops::Deref for QwenMoeLlamaIdCarrierLease<'_> {
    type Target = ffn_chain::QwenMoeLlamaIdCarrier;

    fn deref(&self) -> &Self::Target {
        self.carrier.as_ref().expect("live llama-ID carrier lease")
    }
}

#[cfg(target_os = "macos")]
impl std::ops::DerefMut for QwenMoeLlamaIdCarrierLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.carrier.as_mut().expect("live llama-ID carrier lease")
    }
}

#[cfg(target_os = "macos")]
impl Drop for QwenMoeLlamaIdCarrierLease<'_> {
    fn drop(&mut self) {
        if let Some(carrier) = self.carrier.take() {
            self.pool
                .borrow_mut()
                .entry(self.key)
                .or_default()
                .push(carrier);
        }
    }
}

#[cfg(target_os = "macos")]
struct QwenMoePrefillPendingGroup {
    group_len: usize,
    input: Option<Vec<f32>>,
    gate_wb: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_off: u32,
    up_wb: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_off: u32,
    down_wb: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_off: u32,
    group_start: u32,
    total_elems: u32,
}

#[cfg(target_os = "macos")]
struct QwenMoePrefillRouteArrayCarrier {
    slots: usize,
    expert_ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    token_ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[cfg(target_os = "macos")]
impl QwenMoePrefillRouteArrayCarrier {
    fn new(ctx: &compute::MetalContext, slots: usize) -> Self {
        let shared = MTLResourceOptions::StorageModeShared;
        Self {
            slots,
            expert_ids_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<u32>(), shared)
                .expect("Qwen MoE route expert_ids buffer"),
            route_weights_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<f32>(), shared)
                .expect("Qwen MoE route weights buffer"),
            token_ids_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<u32>(), shared)
                .expect("Qwen MoE route token_ids buffer"),
        }
    }

    fn read_u32(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<u32> {
        let contents = buf.contents();
        let slice = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const u32, len) };
        slice.to_vec()
    }

    fn read_f32(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
        let contents = buf.contents();
        let slice = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, len) };
        slice.to_vec()
    }

    fn readback(&self) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
        (
            Self::read_u32(&self.expert_ids_dev, self.slots),
            Self::read_f32(&self.route_weights_dev, self.slots),
            Self::read_u32(&self.token_ids_dev, self.slots),
        )
    }
}

#[cfg(target_os = "macos")]
fn attn_resident_capacity_for(kv_len: usize, requested_capacity: usize) -> usize {
    let min_cap = kv_len.max(1);
    let requested = requested_capacity.max(min_cap);
    let target = min_cap.saturating_add(1024);
    let rounded = target.checked_next_power_of_two().unwrap_or(target);
    rounded.min(requested).max(min_cap)
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Default)]
pub struct MetalBackend {}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_name", &self.device_name)
            .field("ctx", &self.ctx.as_ref().map(|_| "<MetalContext>"))
            .field("resident_entries", &self.resident.borrow().len())
            .finish()
    }
}

impl MetalBackend {
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            let ctx = compute::build_metal_context();

            // device 이름은 ctx 안의 device 에서 직접 가져온다 (device 를 두 번 열지 않는다).
            let device_name = ctx.as_ref().map(|c| {
                use objc2_metal::MTLDevice as _;
                c.device.name().to_string()
            });

            if let Some(ref name) = device_name {
                eprintln!("[INFO] Metal device: {name}");
            } else {
                eprintln!("[WARN] Metal: no system default device");
            }

            Self {
                device_name,
                ctx,
                resident: RefCell::new(HashMap::new()),
                constant_f32: RefCell::new(HashMap::new()),
                constant_u32: RefCell::new(HashMap::new()),
                decode_chain_hidden: RefCell::new(HashMap::new()),
                glm_mla_carriers: RefCell::new(HashMap::new()),
                weight_residency: RefCell::new(None),
                ffn_carriers: RefCell::new(HashMap::new()),
                prefill_ffn_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_private_ffn_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_accum_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_id_carriers: RefCell::new(HashMap::new()),
                qwen_moe_llama_id_carrier_pool: RefCell::new(HashMap::new()),
                qwen_prefill_chain_moe_carrier_pool: RefCell::new(HashMap::new()),
                qwen_moe_prefill_id_gate_up_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_id_gate_up_f16_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_mulmmid_v3_carriers: RefCell::new(HashMap::new()),
                qwen_moe_prefill_mulmmid_v4_carriers: RefCell::new(HashMap::new()),
                qwen_moe_decode_carriers: RefCell::new(HashMap::new()),
                glm_direct_file_handles: RefCell::new(HashMap::new()),
                glm_direct_staging: RefCell::new(HashMap::new()),
                glm_prefill_stage_scratch: RefCell::new(HashMap::new()),
                qwen_moe_decode_id_carriers: RefCell::new(HashMap::new()),
                qwen_moe_decode_chain_carriers: RefCell::new(HashMap::new()),
                prefill_gdn_proj_carriers: RefCell::new(HashMap::new()),
                prefill_gdn_f32_dual_carriers: RefCell::new(HashMap::new()),
                qwen_moe_router_f32_carriers: RefCell::new(HashMap::new()),
                qwen_moe_route_array_carriers: RefCell::new(HashMap::new()),
                output_argmax_carriers: RefCell::new(HashMap::new()),
                prefill_gated_proj_carriers: RefCell::new(HashMap::new()),
                prefill_gdn_conv_delta_carriers: RefCell::new(HashMap::new()),
                prefill_gdn_full_carriers: RefCell::new(HashMap::new()),
                prefill_gdn_full_ffn_carriers: RefCell::new(HashMap::new()),
                prefill_attn_chain_carriers: RefCell::new(HashMap::new()),
                prefill_atn_core_carriers: RefCell::new(HashMap::new()),
                prefill_atn_full_layer_carriers: RefCell::new(HashMap::new()),
                prefill_atn_o_tail_carriers: RefCell::new(HashMap::new()),
                qwen_prefill_gdn_carriers: RefCell::new(HashMap::new()),
                qwen_prefill_atn_o_tail_carriers: RefCell::new(HashMap::new()),
                qkv_carriers: RefCell::new(HashMap::new()),
                gdn_inproj_carriers: RefCell::new(HashMap::new()),
                o_chain_carriers: RefCell::new(HashMap::new()),
                kv_residents: RefCell::new(HashMap::new()),
                kvarn_residents: RefCell::new(HashMap::new()),
                attn_carriers: RefCell::new(HashMap::new()),
                attn_moe_carriers: RefCell::new(HashMap::new()),
                gdn_carriers: RefCell::new(HashMap::new()),
                gdn_core_carriers: RefCell::new(HashMap::new()),
                gdn_batch_carriers: RefCell::new(HashMap::new()),
                attn_batch_carriers: RefCell::new(HashMap::new()),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::default()
        }
    }

    /// 열린 Metal device 의 이름 (다음 milestone 의 facade/로깅에서 사용).
    #[cfg(target_os = "macos")]
    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    /// sequence 경계(새 prompt/turn, in-process engine 재사용)에서 호출.
    /// carrier 의 device-resident sequence state(attention/GDN carrier + 그 안의
    /// KvResident.filled, conv/delta state)를 evict 한다. 다음 layer 호출에서 fresh
    /// 재생성되어 `if filled == 0` prefill 재동기화 가드가 다시 작동한다.
    /// weight resident 캐시(`resident`)와 shape-only carrier(ffn/qkv/gdn_inproj/o_chain)는
    /// sequence-independent 라 유지 — drop 하면 다음 sequence 가 재캐싱 비용만 무는다.
    #[cfg(target_os = "macos")]
    pub fn clear_sequence_state(&self) {
        self.attn_carriers.borrow_mut().clear();
        self.attn_moe_carriers.borrow_mut().clear();
        self.gdn_carriers.borrow_mut().clear();
        self.gdn_core_carriers.borrow_mut().clear();
        self.qwen_moe_decode_chain_carriers.borrow_mut().clear();
        self.kv_residents.borrow_mut().clear();
        self.kvarn_residents.borrow_mut().clear();
        self.attn_batch_carriers.borrow_mut().clear();
    }

    /// pm112: weight residency 활성 여부. 기본 ON — MLA dense GEMV Metal 경로가
    /// per-submit wiring 제거에 의존한다. `RNB_METAL_WEIGHT_RESIDENCY=0` 진단 opt-out.
    #[cfg(target_os = "macos")]
    fn weight_residency_enabled(&self) -> bool {
        std::env::var("RNB_METAL_WEIGHT_RESIDENCY").as_deref() != Ok("0")
    }

    /// pm112: weight residency LRU lazy init. opt-out 시 no-op.
    #[cfg(target_os = "macos")]
    fn ensure_weight_residency(&self, ctx: &compute::MetalContext) {
        if !self.weight_residency_enabled() {
            return;
        }
        let mut slot = self.weight_residency.borrow_mut();
        if slot.is_none() {
            *slot = Some(WeightResidencyLru::new(ctx));
        }
    }

    /// pm112: resident wrap 된 weight buffer 를 residency LRU 에 등록/갱신.
    /// 신규 등록이 있을 때만 commit (그 외 no-op 수준).
    #[cfg(target_os = "macos")]
    fn touch_weight_residency(
        &self,
        key: ResidentKey,
        buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    ) {
        if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
            lru.touch(key, buf);
            lru.commit_if_dirty();
        }
    }

    /// pm31: device 잔류 delta_state(residency 경로)를 host `out` 으로 읽는다.
    /// speculative checkpoint(`materialize_sequence_state`) 직전 host delta 를 최신화하는
    /// 데 쓴다(CUDA `sync_delta_state_cache` 와 동형). carrier 없는 layer(attn 또는
    /// 미초기화 GDN)는 `false` 반환(host `out` 미변경). `out.len()` == delta_state_len 가정.
    #[cfg(target_os = "macos")]
    pub fn sync_delta_state(&self, layer: usize, out: &mut [f32]) -> bool {
        let carriers = self.gdn_carriers.borrow();
        let Some(carrier) = carriers.get(&layer) else {
            return false;
        };
        debug_assert_eq!(out.len(), carrier.delta_state_len);
        out.copy_from_slice(&carrier.readback_delta_state());
        true
    }

    /// Attention chain carrier 의 device KV 동기화 진행도를 반환한다.
    /// carrier 가 없으면 caller 가 host prior KV 를 보내야 하므로 None.
    #[cfg(target_os = "macos")]
    pub fn attn_carrier_kv_filled(&self, layer: usize) -> Option<usize> {
        self.attn_carriers
            .borrow()
            .get(&layer)
            .map(attn_chain::AttnCarrier::kv_filled)
            .or_else(|| {
                self.attn_moe_carriers
                    .borrow()
                    .get(&layer)
                    .map(attn_chain::AttnCarrier::kv_filled)
            })
    }

    /// test 전용 생성자: env(`RNB_METAL_KV_INT8`) 대신 호출자가 직접 kv_int8 을
    /// 지정한 ctx 를 주입한다. int8 test 가 `new()`(env 경로)를 쓰면 env race 가
    /// 나므로 분리한다. 나머지 필드 init 은 `new()` 와 동일하게 따른다.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn new_with_kv_int8(kv_int8: bool) -> Self {
        let ctx = compute::build_metal_context_with_kv_int8(kv_int8);

        let device_name = ctx.as_ref().map(|c| {
            use objc2_metal::MTLDevice as _;
            c.device.name().to_string()
        });

        Self {
            device_name,
            ctx,
            resident: RefCell::new(HashMap::new()),
            constant_f32: RefCell::new(HashMap::new()),
            constant_u32: RefCell::new(HashMap::new()),
            decode_chain_hidden: RefCell::new(HashMap::new()),
            glm_mla_carriers: RefCell::new(HashMap::new()),
            weight_residency: RefCell::new(None),
            ffn_carriers: RefCell::new(HashMap::new()),
            prefill_ffn_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_private_ffn_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_accum_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_id_carriers: RefCell::new(HashMap::new()),
            qwen_moe_llama_id_carrier_pool: RefCell::new(HashMap::new()),
            qwen_prefill_chain_moe_carrier_pool: RefCell::new(HashMap::new()),
            qwen_moe_prefill_id_gate_up_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_id_gate_up_f16_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_mulmmid_v3_carriers: RefCell::new(HashMap::new()),
            qwen_moe_prefill_mulmmid_v4_carriers: RefCell::new(HashMap::new()),
            qwen_moe_decode_carriers: RefCell::new(HashMap::new()),
            glm_direct_file_handles: RefCell::new(HashMap::new()),
            glm_direct_staging: RefCell::new(HashMap::new()),
            glm_prefill_stage_scratch: RefCell::new(HashMap::new()),
            qwen_moe_decode_id_carriers: RefCell::new(HashMap::new()),
            qwen_moe_decode_chain_carriers: RefCell::new(HashMap::new()),
            prefill_gdn_proj_carriers: RefCell::new(HashMap::new()),
            prefill_gdn_f32_dual_carriers: RefCell::new(HashMap::new()),
            qwen_moe_router_f32_carriers: RefCell::new(HashMap::new()),
            qwen_moe_route_array_carriers: RefCell::new(HashMap::new()),
            output_argmax_carriers: RefCell::new(HashMap::new()),
            prefill_gated_proj_carriers: RefCell::new(HashMap::new()),
            prefill_gdn_conv_delta_carriers: RefCell::new(HashMap::new()),
            prefill_gdn_full_carriers: RefCell::new(HashMap::new()),
            prefill_gdn_full_ffn_carriers: RefCell::new(HashMap::new()),
            prefill_attn_chain_carriers: RefCell::new(HashMap::new()),
            prefill_atn_core_carriers: RefCell::new(HashMap::new()),
            prefill_atn_full_layer_carriers: RefCell::new(HashMap::new()),
            prefill_atn_o_tail_carriers: RefCell::new(HashMap::new()),
            qwen_prefill_gdn_carriers: RefCell::new(HashMap::new()),
            qwen_prefill_atn_o_tail_carriers: RefCell::new(HashMap::new()),
            qkv_carriers: RefCell::new(HashMap::new()),
            gdn_inproj_carriers: RefCell::new(HashMap::new()),
            o_chain_carriers: RefCell::new(HashMap::new()),
            kv_residents: RefCell::new(HashMap::new()),
            kvarn_residents: RefCell::new(HashMap::new()),
            attn_carriers: RefCell::new(HashMap::new()),
            attn_moe_carriers: RefCell::new(HashMap::new()),
            gdn_carriers: RefCell::new(HashMap::new()),
            gdn_core_carriers: RefCell::new(HashMap::new()),
            gdn_batch_carriers: RefCell::new(HashMap::new()),
            attn_batch_carriers: RefCell::new(HashMap::new()),
        }
    }
}

#[cfg(target_os = "macos")]
impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    pub fn gemv_f32(&self, weight: &[f32], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        compute::gemv_f32(weight, input, n, k)
    }

    /// Q4_K GEMV: weight 는 N × (K/256) 개의 Q4_K super-block (raw bytes),
    /// input 은 K 개의 f32. 결과 out[N] 을 반환.
    ///
    /// 캐싱된 `MetalContext` 를 재사용하므로 per-call MSL compile 비용 없음.
    pub fn gemv_q4k(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q4k_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q4_K GEMV (SIMD-group 협력 커널). `gemv_q4k` 와 동일 결과, memory-bandwidth
    /// 활용을 위해 SIMD-group(32 lane)이 1 row 를 협력 처리.
    pub fn gemv_q4k_simd(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q4k_simd_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q4_K GEMV (llama.cpp coalesced 커널 이식). `gemv_q4k`/`gemv_q4k_simd` 와 동일
    /// 수치 결과. q4k_simd 의 fallback(stride-32, uncoalesced) 경로를 대체할 coalesced
    /// 메모리 접근 후보 — 측정/테스트용 복사 경로. chain 미연결.
    pub fn gemv_q4k_coalesced(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q4k_coalesced_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q4_K GEMV nsg2 variant(threadgroup 당 SIMD-group 2개). 측정/테스트용 복사 경로.
    pub fn gemv_q4k_coalesced_nsg2(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q4k_coalesced_nsg2_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q5_K GEMV (llama.cpp nr0=2 multi-row coalesced 커널 이식). `gemv_q5k` 와 동일
    /// 수치 결과. q5k_simd 의 fallback(stride-32, uncoalesced) 경로를 대체할 coalesced
    /// 후보 — 측정/테스트용 복사 경로. chain 은 encode_gemv_q5k_simd 의 gemv_coalesced 분기.
    pub fn gemv_q5k_coalesced(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q5k_coalesced_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q6_K GEMV (llama.cpp nr0=2 multi-row coalesced 커널 이식). `gemv_q6k` 와 동일
    /// 수치 결과. q6k_simd 의 fallback(stride-32, uncoalesced) 경로를 대체할 coalesced
    /// 후보 — 측정/테스트용 복사 경로. chain 은 encode_gemv_q6k_simd 의 gemv_coalesced 분기.
    pub fn gemv_q6k_coalesced(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q6k_coalesced_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q6_K GEMV nsg2 variant(threadgroup 당 SIMD-group 2개). 측정/테스트용 복사 경로.
    pub fn gemv_q6k_coalesced_nsg2(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q6k_coalesced_nsg2_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q8_0 GEMV (llama.cpp NR0=2 multi-row coalesced 커널 이식). `gemv_q8_0` 와 동일
    /// 수치 결과. block size=32(K/32 blocks). 측정/테스트용 복사 경로. chain 은
    /// encode_gemv_q8_0 의 gemv_coalesced 분기.
    pub fn gemv_q8_0_coalesced(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gemv_q8_0_coalesced_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Single-token decode attention (QK^T → online softmax → AV) device 커널.
    /// CPU `attention_decode_flash`(production default) 와 token-identical 목표.
    /// KV cache 는 f16 bits(u16). 측정/테스트용 복사 경로.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        q: &[f32],
        k_cache: &[u16],
        v_cache: &[u16],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::attn_decode_with_ctx(
            ctx,
            q,
            k_cache,
            v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
        )
    }

    /// Text M-RoPE in-place(decode 1 token) device 커널. CPU
    /// `rope_mrope_text_inplace` 와 token-identical. 측정/테스트용 복사 경로.
    pub fn rope_mrope(
        &self,
        data: &[f32],
        head_dim: usize,
        dim: usize,
        n_rot: usize,
        theta: f32,
        pos: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::rope_mrope_with_ctx(ctx, data, head_dim, dim, n_rot, theta, pos)
    }

    /// Per-head q/k RMSNorm: `data`=[num_heads*head_dim], `weight`=[head_dim]
    /// (head 공유). head 별 head_dim 슬라이스를 각각 RMSNorm. CPU `rms_norm_into`
    /// 와 token-identical (표준 Qwen3 attn q_norm/k_norm).
    pub fn qk_norm(
        &self,
        data: &[f32],
        weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::qk_norm_with_ctx(ctx, data, weight, num_heads, head_dim, eps)
    }

    /// GDN decode(seq_len=1) depthwise causal conv1d + SiLU: `input`/`weight`=
    /// [kernel_size*channels], 반환=[channels]. CPU `ssm_conv1d_silu_into`(seq_len=1)
    /// 와 token-identical. GDN layer carrier 의 conv 부품.
    pub fn ssm_conv1d_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        channels: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::ssm_conv1d_silu_with_ctx(ctx, input, weight, channels, kernel_size)
    }

    /// pm43: GDN prefill(seq_len>1) batch depthwise causal conv1d + SiLU. `input`=
    /// [(seq_len+kernel_size-1)*channels], `weight`=[kernel_size*channels], 반환=[seq_len*channels].
    /// CPU `ssm_conv1d_silu_into`(seq_len) 와 1:1. decode `ssm_conv1d_silu` 의 batch 확장.
    pub fn prefill_conv1d_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::ssm_conv1d_silu_batch_with_ctx(ctx, input, weight, seq_len, channels, kernel_size)
    }

    /// pm43: GDN prefill gated RMSNorm + SiLU(batch). `out_in`/`z`=[rows*cols], `weight`=[cols](ssm_norm),
    /// 반환=[rows*cols] = rmsnorm(out_in per row)·silu(z). CPU gdn_forward gated_norm+silu 1:1.
    pub fn prefill_gated_norm_silu(
        &self,
        out_in: &[f32],
        z: &[f32],
        weight: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gated_rmsnorm_silu_batch_with_ctx(ctx, out_in, z, weight, rows, cols, eps)
    }

    /// GDN delta_net recurrent scan 1-step(decode, seq_len=1). `state` 입력은
    /// in-place 갱신되어 `state_after` 로 반환. 반환=`(output, state_after)`.
    /// CPU `delta_net_scan_into`(seq_len=1) 와 token-identical. GDN layer carrier 부품.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_net_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        num_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::delta_net_step_with_ctx(
            ctx,
            q,
            k,
            v,
            gate,
            beta,
            state,
            num_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
        )
    }

    /// pm39 M2: GDN delta_net chunkwise parallel scan(prefill, seq_len>1). `state` 입력은
    /// chunk-loop hand-off 로 in-place 갱신되어 `state_after` 로 반환. 반환=`(output, state_after)`.
    /// CPU `delta_net_scan_chunkwise`(M1 oracle, 같은 chunk_size) 와 token-identical.
    /// GQA 는 caller 가 q/k 를 num_heads 로 repeat 푼 뒤 넘긴다.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_net_scan_chunk(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::delta_net_scan_chunk_with_ctx(
            ctx, q, k, v, gate, beta, state, seq_len, num_heads, head_k_dim, head_v_dim, chunk_size,
        )
    }

    /// pm58: GDN delta_net autoregressive prefill scan. Same sequential recurrence as
    /// CPU `delta_net_scan`; used to evaluate the llama.cpp-style Qwen3Next Metal path.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_net_scan_ar(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::delta_net_scan_ar_with_ctx(
            ctx, q, k, v, gate, beta, state, seq_len, num_heads, head_k_dim, head_v_dim,
        )
    }

    /// pm48 ①: dense causal GQA prefill attention compute(simdgroup matmul2d flash). q[seq*nh*hd]
    /// f32(rope/qk_norm 적용 완료), k/v[kv*nkv*hd] f16 bits → out[seq*nh*hd] f32. host 입출력 seam.
    /// `flash_attn_prefill_tg_pipeline`이 None이면 None을 반환해 caller가 CPU로 후퇴한다.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_attention(
        &self,
        q: &[f32],
        k_f16: &[u16],
        v_f16: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Option<Vec<f32>> {
        let ctx = self.ctx.as_ref()?;
        if ctx.flash_attn_prefill_tg_pipeline.is_none() {
            return None;
        }
        let (out, _gpu_ms) = compute::prefill_flash_attention_with_ctx(
            ctx,
            q,
            k_f16,
            v_f16,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        );
        Some(out)
    }

    /// pm48 ②: prefill attention 2차 device-resident chain. rope/qk_norm(q,k) → cast(k,v→f16) →
    /// flash 를 단일 command buffer 로. 입력(host): q_proj(gate split 후, norm 전), k_proj(norm 전),
    /// v(f32), q_norm/k_norm weight. 반환 `(attn_out[seq*q_dim], k_f16[seq*kv_dim], v_f16)`.
    /// k_f16/v_f16 은 device 에서 만든 f16 KV — 호출자가 host KV cache 에 1회 write.
    /// `flash_attn_prefill_tg_pipeline`/`cast_f32_f16_pipeline` 없으면 None.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attn_chain(
        &self,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        theta: f32,
        eps: f32,
        pos_start: usize,
        scale: f32,
        layer_idx: usize,
        timing_enabled: bool,
    ) -> Option<(Vec<f32>, Vec<u16>, Vec<u16>)> {
        let ctx = self.ctx.as_ref()?;
        if ctx.flash_attn_prefill_tg_pipeline.is_none() || ctx.cast_f32_f16_pipeline.is_none() {
            return None;
        }
        let key = (seq_len, num_heads, num_kv_heads, head_dim);
        let mut carriers = self.prefill_attn_chain_carriers.borrow_mut();
        let carrier = carriers.entry(key).or_insert_with(|| {
            prefill_attn_chain::PrefillAttnChainCarrier::new(
                ctx,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                n_rot,
                theta,
                eps,
                pos_start,
                scale,
            )
        });
        let trace = prefill_attn_chain::PrefillAttnChainTrace {
            layer_idx,
            timing_enabled,
        };
        Some(prefill_attn_chain::prefill_attn_chain_dispatch(
            ctx,
            carrier,
            q_in,
            k_in,
            v_in,
            q_norm_w,
            k_norm_w,
            Some(trace),
        ))
    }

    #[cfg(target_os = "macos")]
    fn atn_core_tensorops_v2_ready(ctx: &compute::MetalContext, quant: TensoropsQuant) -> bool {
        match quant {
            TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q8_0 => ctx.gemm_q8_0_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q2K => ctx.gemm_q2k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q3K => ctx.gemm_q3k_tensorops_v2_pipeline.is_some(),
        }
    }

    #[cfg(target_os = "macos")]
    fn atn_core_expected_raw_len(view: PrefillAtnCoreWeightView<'_>) -> Option<usize> {
        let (block_elements, block_bytes) = match view.quant {
            TensoropsQuant::Q4K => (256usize, 144usize),
            TensoropsQuant::Q5K => (256, 176),
            TensoropsQuant::Q6K => (256, 210),
            TensoropsQuant::Q8_0 => (32, 34),
            TensoropsQuant::Q2K => (256, 84),
            TensoropsQuant::Q3K => (256, 110),
        };
        if view.cols == 0 || view.cols % block_elements != 0 {
            return None;
        }
        view.rows
            .checked_mul(view.cols / block_elements)?
            .checked_mul(block_bytes)
    }

    #[cfg(target_os = "macos")]
    fn atn_core_validate_weight(
        role: &str,
        view: PrefillAtnCoreWeightView<'_>,
    ) -> std::result::Result<(), String> {
        let Some(expected) = Self::atn_core_expected_raw_len(view) else {
            return Err(format!(
                "Metal prefill ATN core: invalid {role} weight shape rows={} cols={} quant={:?}",
                view.rows, view.cols, view.quant
            ));
        };
        if view.raw.len() != expected {
            return Err(format!(
                "Metal prefill ATN core: invalid {role} raw bytes len={} expected={} rows={} cols={} quant={:?}",
                view.raw.len(),
                expected,
                view.rows,
                view.cols,
                view.quant
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn atn_core_checked_mul(a: usize, b: usize, what: &str) -> std::result::Result<usize, String> {
        a.checked_mul(b)
            .ok_or_else(|| format!("Metal prefill ATN core: {what} overflow: {a} * {b}"))
    }

    #[cfg(target_os = "macos")]
    fn atn_core_require_eq(
        what: &str,
        actual: usize,
        expected: usize,
    ) -> std::result::Result<(), String> {
        if actual != expected {
            return Err(format!(
                "Metal prefill ATN core: invalid {what}: got {actual}, expected {expected}"
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn qwen_prefill_chain_attention_preflight(
        ctx: &compute::MetalContext,
        spec: PrefillAtnOTailBackendSpecRef<'_>,
    ) -> Result<bool, String> {
        let core = spec.core;
        if core.seq_len == 0
            || core.hidden_dim == 0
            || core.num_heads == 0
            || core.num_kv_heads == 0
            || core.num_heads % core.num_kv_heads != 0
            || core.head_dim != 256
            || core.pos_start != 0
            || core.n_rot > core.head_dim
            || core.n_rot % 2 != 0
            || !core.norm_eps.is_finite()
            || core.norm_eps <= 0.0
            || (core.n_rot > 0 && (!core.rope_theta.is_finite() || core.rope_theta <= 0.0))
            || !core.scale.is_finite()
            || ctx.flash_attn_prefill_tg_pipeline.is_none()
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.prefill_split_gate_pipeline.is_none()
            || ctx.prefill_gate_apply_pipeline.is_none()
            || !Self::atn_core_tensorops_v2_ready(ctx, core.q_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.k_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.v_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, spec.o_weight.quant)
        {
            return Ok(false);
        }
        let q_dim = Self::atn_core_checked_mul(core.num_heads, core.head_dim, "q_dim")?;
        let kv_dim = Self::atn_core_checked_mul(core.num_kv_heads, core.head_dim, "kv_dim")?;
        let q_rows = Self::atn_core_checked_mul(core.q_dim, 2, "q weight rows")?;
        Self::atn_core_require_eq("q_dim", core.q_dim, q_dim)?;
        Self::atn_core_require_eq("kv_dim", core.kv_dim, kv_dim)?;
        Self::atn_core_require_eq("attn_norm_w len", core.attn_norm_w.len(), core.hidden_dim)?;
        Self::atn_core_require_eq("q_norm_w len", core.q_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("k_norm_w len", core.k_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("q weight rows", core.q_weight.rows, q_rows)?;
        Self::atn_core_require_eq("q weight cols", core.q_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("k weight rows", core.k_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("k weight cols", core.k_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("v weight rows", core.v_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("v weight cols", core.v_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight rows", spec.o_weight.rows, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight cols", spec.o_weight.cols, core.q_dim)?;
        Self::atn_core_validate_weight("q", core.q_weight)?;
        Self::atn_core_validate_weight("k", core.k_weight)?;
        Self::atn_core_validate_weight("v", core.v_weight)?;
        Self::atn_core_validate_weight("o", spec.o_weight)?;
        Ok(true)
    }

    #[cfg(target_os = "macos")]
    fn qwen_prefill_chain_moe_preflight(
        ctx: &compute::MetalContext,
        seq_len: usize,
        spec: QwenMoePrefillBackendSpecRef<'_>,
    ) -> Result<Option<ffn_chain::QwenMoeLlamaIdPlan>, String> {
        if !qwen_moe_llama_id_quant_supported(spec.sparse_quant, spec.shared_quant) {
            return Ok(None);
        }
        if spec.n_expert_used > 8 {
            return Ok(None);
        }
        if spec.n_expert == 0
            || spec.n_expert_used == 0
            || spec.n_expert_used > spec.n_expert
            || spec.hidden_dim == 0
            || spec.ffn_dim == 0
            || spec.ffn_norm_w.len() != spec.hidden_dim
            || spec.shared_input_scale.len() != spec.hidden_dim
            || !spec.norm_eps.is_finite()
            || spec.norm_eps <= 0.0
        {
            return Err("invalid Metal Qwen prefill MoE shape".to_string());
        }
        let router_len = spec
            .n_expert
            .checked_mul(spec.hidden_dim)
            .ok_or_else(|| "Metal Qwen prefill router shape overflow".to_string())?;
        if spec.router_w.len() != router_len {
            return Err("invalid Metal Qwen prefill router weight length".to_string());
        }
        let gate_bytes =
            qwen_moe_llama_id_weight_bytes(spec.sparse_quant.gate, spec.hidden_dim, spec.ffn_dim)
                .map_err(qwen_moe_llama_id_error)?;
        let up_bytes =
            qwen_moe_llama_id_weight_bytes(spec.sparse_quant.up, spec.hidden_dim, spec.ffn_dim)
                .map_err(qwen_moe_llama_id_error)?;
        let down_bytes =
            qwen_moe_llama_id_weight_bytes(spec.sparse_quant.down, spec.ffn_dim, spec.hidden_dim)
                .map_err(qwen_moe_llama_id_error)?;
        let gate_arena = gate_bytes
            .checked_mul(spec.n_expert)
            .ok_or_else(|| "Metal Qwen prefill gate arena overflow".to_string())?;
        let up_arena = up_bytes
            .checked_mul(spec.n_expert)
            .ok_or_else(|| "Metal Qwen prefill up arena overflow".to_string())?;
        let down_arena = down_bytes
            .checked_mul(spec.n_expert)
            .ok_or_else(|| "Metal Qwen prefill down arena overflow".to_string())?;
        let shared_gate_bytes =
            qwen_moe_llama_id_weight_bytes(spec.shared_quant.gate, spec.hidden_dim, spec.ffn_dim)
                .map_err(qwen_moe_llama_id_error)?;
        let shared_up_bytes =
            qwen_moe_llama_id_weight_bytes(spec.shared_quant.up, spec.hidden_dim, spec.ffn_dim)
                .map_err(qwen_moe_llama_id_error)?;
        let shared_down_bytes =
            qwen_moe_llama_id_weight_bytes(spec.shared_quant.down, spec.ffn_dim, spec.hidden_dim)
                .map_err(qwen_moe_llama_id_error)?;
        if spec.gate_expert_bytes != gate_bytes
            || spec.up_expert_bytes != up_bytes
            || spec.down_expert_bytes != down_bytes
            || spec.gate_all.len() != gate_arena
            || spec.up_all.len() != up_arena
            || spec.down_all.len() != down_arena
            || spec.shared_gate.len() != shared_gate_bytes
            || spec.shared_up.len() != shared_up_bytes
            || spec.shared_down.len() != shared_down_bytes
        {
            return Err("invalid Metal Qwen prefill MoE expert arena".to_string());
        }
        let pipeline_available =
            qwen_moe_llama_id_pipelines_available(ctx, spec.sparse_quant, spec.shared_quant);
        let required_bytes = ffn_chain::qwen_moe_prefill_device_route_required_bytes(
            seq_len,
            spec.n_expert,
            spec.n_expert_used,
            spec.hidden_dim,
            spec.ffn_dim,
        )
        .map_err(qwen_moe_llama_id_error)?;
        match ffn_chain::qwen_moe_prefill_device_route_preflight(
            pipeline_available,
            true,
            seq_len,
            spec.n_expert,
            spec.n_expert_used,
            spec.hidden_dim,
            spec.ffn_dim,
            QWEN_MOE_LLAMA_ID_SCRATCH_BUDGET_BYTES,
        )
        .map_err(qwen_moe_llama_id_error)?
        {
            ffn_chain::QwenMoeLlamaIdPreflight::Run(plan) => Ok(Some(plan)),
            ffn_chain::QwenMoeLlamaIdPreflight::Fallback(reason) => {
                if qwen_prefill_chain_trace_enabled() {
                    eprintln!(
                        "[metal-qwen-prefill-chain] fallback_reason={} seq_len={} n_expert={} n_expert_used={} hidden_dim={} ffn_dim={} required_bytes={} budget_bytes={}",
                        qwen_moe_llama_id_fallback_reason(reason),
                        seq_len,
                        spec.n_expert,
                        spec.n_expert_used,
                        spec.hidden_dim,
                        spec.ffn_dim,
                        required_bytes,
                        QWEN_MOE_LLAMA_ID_SCRATCH_BUDGET_BYTES,
                    );
                }
                Ok(None)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn qwen_prefill_chain_prepare_moe_weights(
        &self,
        ctx: &compute::MetalContext,
        specs: &[QwenPrefillChainSpecRef<'_>],
    ) -> Vec<ffn_chain::QwenMoePrefillWeightBuffers> {
        let mut resident = self.resident.borrow_mut();
        let mut wrap = |raw: &[u8]| {
            let key = resident_key(raw);
            let entry = resident
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1 as usize)
        };
        let weights = specs
            .iter()
            .map(|spec| {
                let moe = match *spec {
                    QwenPrefillChainSpecRef::Attention { moe, .. }
                    | QwenPrefillChainSpecRef::Gdn { moe, .. } => moe,
                };
                let (gate_all, gate_offset) = wrap(moe.gate_all);
                let (up_all, up_offset) = wrap(moe.up_all);
                let (down_all, down_offset) = wrap(moe.down_all);
                let (shared_gate, shared_gate_offset) = wrap(moe.shared_gate);
                let (shared_up, shared_up_offset) = wrap(moe.shared_up);
                let (shared_down, shared_down_offset) = wrap(moe.shared_down);
                ffn_chain::QwenMoePrefillWeightBuffers {
                    gate_all,
                    gate_offset,
                    up_all,
                    up_offset,
                    down_all,
                    down_offset,
                    shared_gate,
                    shared_gate_offset,
                    shared_up,
                    shared_up_offset,
                    shared_down,
                    shared_down_offset,
                }
            })
            .collect();
        weights
    }

    #[cfg(target_os = "macos")]
    pub fn qwen_prefill_chain_run(
        &self,
        hidden: &[f32],
        specs: &[QwenPrefillChainSpecRef<'_>],
    ) -> Result<Option<QwenPrefillChainBackendOut>, String> {
        if specs.is_empty() {
            let transfer_ledger = QwenPrefillChainTransferLedger::default();
            return Ok(Some(QwenPrefillChainBackendOut {
                hidden: hidden.to_vec(),
                attention_kv: Vec::new(),
                gdn_states: Vec::new(),
                hidden_uploads: transfer_ledger.hidden_uploads,
                hidden_readbacks: transfer_ledger.hidden_readbacks,
                intermediate_hidden_transfers: transfer_ledger.intermediate_hidden_transfers,
            }));
        }
        if !qwen_prefill_chain_layer_indices_supported(specs)? {
            return Ok(None);
        }
        let Some(ctx) = self.ctx.as_ref() else {
            return Ok(None);
        };

        let attention_key = |spec: PrefillAtnOTailBackendSpecRef<'_>| QwenAtnOTailCarrierKey {
            seq_len: spec.core.seq_len,
            num_heads: spec.core.num_heads,
            num_kv_heads: spec.core.num_kv_heads,
            head_dim: spec.core.head_dim,
            hidden_dim: spec.core.hidden_dim,
            q_dim: spec.core.q_dim,
            kv_dim: spec.core.kv_dim,
            n_rot: spec.core.n_rot,
            rope_theta_bits: spec.core.rope_theta.to_bits(),
            scale_bits: spec.core.scale.to_bits(),
            norm_eps_bits: spec.core.norm_eps.to_bits(),
            pos_start: spec.core.pos_start,
        };
        let gdn_key = |spec: QwenPrefillGdnBackendSpecRef<'_>| {
            (
                spec.seq_len,
                spec.hidden_dim,
                spec.d_inner,
                spec.d_state,
                spec.n_group,
                spec.dt_rank,
                spec.conv_kernel_size,
                spec.norm_eps.to_bits(),
            )
        };

        let (chain_seq_len, chain_hidden_dim) = match specs[0] {
            QwenPrefillChainSpecRef::Attention { core, .. } => {
                (core.core.seq_len, core.core.hidden_dim)
            }
            QwenPrefillChainSpecRef::Gdn { layer, .. } => (layer.seq_len, layer.hidden_dim),
        };
        let hidden_elements = chain_seq_len
            .checked_mul(chain_hidden_dim)
            .ok_or_else(|| "Metal Qwen prefill chain hidden shape overflow".to_string())?;
        if hidden.len() != hidden_elements {
            return Err(format!(
                "Metal Qwen prefill chain hidden length mismatch: got {}, expected {hidden_elements}",
                hidden.len()
            ));
        }

        let mut plans = Vec::with_capacity(specs.len());
        for spec in specs {
            let (seq_len, hidden_dim, moe) = match *spec {
                QwenPrefillChainSpecRef::Attention { core, moe, .. } => {
                    if !Self::qwen_prefill_chain_attention_preflight(ctx, core)? {
                        return Ok(None);
                    }
                    (core.core.seq_len, core.core.hidden_dim, moe)
                }
                QwenPrefillChainSpecRef::Gdn { layer, moe, .. } => {
                    if !gdn_chain::qwen_gdn_prefill_preflight(ctx, layer)? {
                        return Ok(None);
                    }
                    (
                        layer.seq_len,
                        layer.hidden_dim,
                        QwenMoePrefillBackendSpecRef {
                            ffn_norm_w: layer.post_attn_norm_w,
                            ..moe
                        },
                    )
                }
            };
            if seq_len != chain_seq_len
                || hidden_dim != chain_hidden_dim
                || moe.hidden_dim != chain_hidden_dim
            {
                return Err("Metal Qwen prefill chain layer shape mismatch".to_string());
            }
            let Some(plan) = Self::qwen_prefill_chain_moe_preflight(ctx, seq_len, moe)? else {
                return Ok(None);
            };
            plans.push(plan);
        }
        let moe_weights = self.qwen_prefill_chain_prepare_moe_weights(ctx, specs);

        let mut transfer_ledger = QwenPrefillChainTransferLedger::default();
        let hidden_a = ffn_chain::empty_f32_buf(ctx, hidden_elements);
        let hidden_b = ffn_chain::empty_f32_buf(ctx, hidden_elements);
        gdn_chain::upload_shared(&hidden_a, hidden);
        transfer_ledger.record_initial_hidden_upload();
        let stage_dump_dir =
            std::env::var_os("RNB_TASK11_MOE_STAGE_DUMP_DIR").filter(|value| !value.is_empty());
        let mut stage_snapshots = Vec::new();
        if stage_dump_dir.is_some() {
            for (spec, plan) in specs.iter().zip(plans.iter()) {
                let layer_idx = qwen_prefill_chain_layer_index(spec);
                if layer_idx > 3 {
                    continue;
                }
                let ids_elements = plan.n_expert.checked_mul(plan.n_tokens).ok_or_else(|| {
                    format!("Metal Qwen prefill L{layer_idx} stage dump ID shape overflow")
                })?;
                stage_snapshots.push((
                    layer_idx,
                    (
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, plan.n_expert),
                        ffn_chain::empty_f32_buf(ctx, ids_elements),
                    ),
                ));
            }
        }
        let mut gdn_stage_snapshots = Vec::new();
        if stage_dump_dir.is_some() {
            for spec in specs {
                let QwenPrefillChainSpecRef::Gdn {
                    layer_idx, layer, ..
                } = *spec
                else {
                    continue;
                };
                if layer_idx > 3 {
                    continue;
                }
                let elements = layer.seq_len.checked_mul(layer.dt_rank).ok_or_else(|| {
                    format!("Metal Qwen prefill L{layer_idx} GDN stage dump shape overflow")
                })?;
                gdn_stage_snapshots.push((
                    layer_idx,
                    (
                        ffn_chain::empty_f32_buf(ctx, elements),
                        ffn_chain::empty_f32_buf(ctx, elements),
                        ffn_chain::empty_f32_buf(ctx, elements),
                        ffn_chain::empty_f32_buf(ctx, elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                        ffn_chain::empty_f32_buf(ctx, hidden_elements),
                    ),
                ));
            }
        }

        {
            let mut carriers = self.qwen_prefill_atn_o_tail_carriers.borrow_mut();
            for spec in specs {
                if let QwenPrefillChainSpecRef::Attention { core, .. } = *spec {
                    let key = attention_key(core);
                    carriers.entry(key).or_insert_with(|| {
                        prefill_atn_core_chain::PrefillAtnOTailCarrier::new(
                            ctx,
                            core.core.seq_len,
                            core.core.num_heads,
                            core.core.num_kv_heads,
                            core.core.head_dim,
                            core.core.hidden_dim,
                            core.core.q_dim,
                            core.core.kv_dim,
                            core.core.n_rot,
                            core.core.rope_theta,
                            core.core.scale,
                            core.core.norm_eps,
                            core.core.pos_start,
                        )
                    });
                }
            }
        }
        {
            let mut carriers = self.qwen_prefill_gdn_carriers.borrow_mut();
            for spec in specs {
                if let QwenPrefillChainSpecRef::Gdn { layer, .. } = *spec {
                    let key = gdn_key(layer);
                    carriers.entry(key).or_insert_with(|| {
                        gdn_chain::QwenGdnPrefillCarrier::new(
                            ctx,
                            layer.seq_len,
                            layer.hidden_dim,
                            layer.d_inner,
                            layer.d_state,
                            layer.n_group,
                            layer.dt_rank,
                            layer.conv_kernel_size,
                            layer.norm_eps,
                        )
                    });
                }
            }
        }

        let mut moe_carriers = Vec::new();
        for &plan in &plans {
            if moe_carriers
                .iter()
                .any(|(existing, _): &(ffn_chain::QwenMoeLlamaIdPlan, _)| *existing == plan)
            {
                continue;
            }
            let cached = self
                .qwen_prefill_chain_moe_carrier_pool
                .borrow_mut()
                .get_mut(&plan)
                .and_then(Vec::pop);
            let carrier = match cached {
                Some(carrier) => carrier,
                None => ffn_chain::QwenMoeLlamaIdCarrier::new_device_routes(ctx, plan)
                    .map_err(qwen_moe_llama_id_error)?,
            };
            moe_carriers.push((
                plan,
                QwenMoeLlamaIdCarrierLease {
                    pool: &self.qwen_prefill_chain_moe_carrier_pool,
                    key: plan,
                    carrier: Some(carrier),
                },
            ));
        }

        enum LayerDeviceOut {
            Attention {
                layer_idx: usize,
                k: Retained<ProtocolObject<dyn MTLBuffer>>,
                v: Retained<ProtocolObject<dyn MTLBuffer>>,
                len: usize,
            },
            Gdn {
                layer_idx: usize,
                conv: Retained<ProtocolObject<dyn MTLBuffer>>,
                delta: Retained<ProtocolObject<dyn MTLBuffer>>,
                conv_len: usize,
                delta_len: usize,
            },
        }
        let mut layer_outputs = Vec::with_capacity(specs.len());
        for spec in specs {
            match *spec {
                QwenPrefillChainSpecRef::Attention {
                    layer_idx, core, ..
                } => {
                    let len = core
                        .core
                        .seq_len
                        .checked_mul(core.core.kv_dim)
                        .ok_or_else(|| "Metal Qwen prefill KV shape overflow".to_string())?;
                    let padded_len = core.core.seq_len.next_multiple_of(64) * core.core.kv_dim;
                    layer_outputs.push(LayerDeviceOut::Attention {
                        layer_idx,
                        k: ffn_chain::empty_f16_buf_with_zeroed_tail(ctx, len, padded_len),
                        v: ffn_chain::empty_f16_buf_with_zeroed_tail(ctx, len, padded_len),
                        len,
                    });
                }
                QwenPrefillChainSpecRef::Gdn {
                    layer_idx, layer, ..
                } => {
                    let conv_channels = layer
                        .n_group
                        .checked_mul(layer.d_state)
                        .and_then(|qk| qk.checked_mul(2))
                        .and_then(|qk| layer.d_inner.checked_add(qk))
                        .ok_or_else(|| "Metal Qwen GDN conv shape overflow".to_string())?;
                    let conv_len = (layer.conv_kernel_size - 1)
                        .checked_mul(conv_channels)
                        .ok_or_else(|| "Metal Qwen GDN conv state overflow".to_string())?;
                    let delta_len = layer
                        .d_inner
                        .checked_mul(layer.d_state)
                        .ok_or_else(|| "Metal Qwen GDN delta state overflow".to_string())?;
                    layer_outputs.push(LayerDeviceOut::Gdn {
                        layer_idx,
                        conv: ffn_chain::empty_f32_buf(ctx, conv_len),
                        delta: ffn_chain::empty_f32_buf(ctx, delta_len),
                        conv_len,
                        delta_len,
                    });
                }
            }
        }

        let profile_first_attention_command =
            std::env::var("RNB_METAL_PROFILE_FIRST_ATTENTION_COMMAND").as_deref() == Ok("1");
        let attention_profile_input =
            profile_first_attention_command.then(|| ffn_chain::empty_f32_buf(ctx, hidden_elements));
        let mut attention_profile_position = None;
        let profile_first_attention =
            std::env::var("RNB_METAL_PROFILE_FIRST_ATTENTION").as_deref() == Ok("1");
        let mut attention_sampler = profile_first_attention
            .then(|| ffn_chain::QwenMoeLlamaIdStageSampler::new(ctx))
            .transpose()
            .map_err(|reason| format!("Metal first-attention profiler unavailable: {reason}"))?;
        let mut attention_sampled = false;
        let cmd = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| "Metal: failed to create Qwen prefill command buffer".to_string())?;
        let enc = compute::try_chain_compute_encoder(ctx, &cmd)?;
        let encode_result = (|| -> Result<(), String> {
            for (position, spec) in specs.iter().enumerate() {
                let layer_idx = qwen_prefill_chain_layer_index(spec);
                let moe = match (*spec, &layer_outputs[position]) {
                    (
                        QwenPrefillChainSpecRef::Attention { core, moe, .. },
                        LayerDeviceOut::Attention { k, v, .. },
                    ) => {
                        let key = attention_key(core);
                        let carriers = self.qwen_prefill_atn_o_tail_carriers.borrow();
                        let carrier = carriers.get(&key).ok_or_else(|| {
                            "Metal Qwen prefill attention carrier missing".to_string()
                        })?;
                        if attention_profile_position.is_none() {
                            if let Some(profile_input) = attention_profile_input.as_ref() {
                                let moe_carrier = moe_carriers
                                    .iter()
                                    .find(|(plan, _)| *plan == plans[position])
                                    .map(|(_, carrier)| &**carrier)
                                    .ok_or_else(|| {
                                        "Metal Qwen prefill profile MoE carrier missing".to_string()
                                    })?;
                                moe_carrier
                                    .encode_profile_hidden_copy(ctx, &enc, &hidden_a, profile_input)
                                    .map_err(qwen_moe_llama_id_error)?;
                                compute::chain_barrier(ctx, &enc);
                                attention_profile_position = Some(position);
                            }
                        }
                        let sampler = if !attention_sampled {
                            attention_sampled = true;
                            attention_sampler.as_mut()
                        } else {
                            None
                        };
                        prefill_atn_core_chain::encode_prefill_atn_o_tail_ops_profiled(
                            ctx,
                            &enc,
                            carrier,
                            core,
                            &hidden_a,
                            &hidden_b,
                            (k, v),
                            sampler,
                        )?;
                        moe
                    }
                    (
                        QwenPrefillChainSpecRef::Gdn { layer, moe, .. },
                        LayerDeviceOut::Gdn { conv, delta, .. },
                    ) => {
                        let key = gdn_key(layer);
                        let carriers = self.qwen_prefill_gdn_carriers.borrow();
                        let carrier = carriers
                            .get(&key)
                            .ok_or_else(|| "Metal Qwen prefill GDN carrier missing".to_string())?;
                        if let Some((_, snapshot)) = gdn_stage_snapshots
                            .iter()
                            .find(|(snapshot_layer, _)| *snapshot_layer == layer_idx)
                        {
                            gdn_chain::encode_qwen_gdn_prefill_ops_with_stage_snapshots(
                                ctx,
                                &enc,
                                carrier,
                                layer,
                                &hidden_a,
                                &hidden_b,
                                (conv, delta),
                                Some(gdn_chain::QwenGdnPrefillStageSnapshots {
                                    alpha_raw: &snapshot.0,
                                    beta_raw: &snapshot.1,
                                    alpha_processed: &snapshot.2,
                                    beta_processed: &snapshot.3,
                                    norm: &snapshot.4,
                                    hidden_input: &snapshot.5,
                                }),
                            )?;
                        } else {
                            gdn_chain::encode_qwen_gdn_prefill_ops(
                                ctx,
                                &enc,
                                carrier,
                                layer,
                                &hidden_a,
                                &hidden_b,
                                (conv, delta),
                            )?;
                        }
                        QwenMoePrefillBackendSpecRef {
                            ffn_norm_w: layer.post_attn_norm_w,
                            ..moe
                        }
                    }
                    _ => {
                        return Err("Metal Qwen prefill chain internal output ordering mismatch"
                            .to_string());
                    }
                };
                compute::chain_barrier(ctx, &enc);
                let carrier = moe_carriers
                    .iter()
                    .find(|(plan, _)| *plan == plans[position])
                    .map(|(_, carrier)| &**carrier)
                    .ok_or_else(|| "Metal Qwen prefill MoE carrier missing".to_string())?;
                let stage_snapshot = stage_snapshots
                    .iter()
                    .find(|(snapshot_layer, _)| *snapshot_layer == layer_idx)
                    .map(|(_, snapshot)| snapshot);
                if let Some(snapshot) = stage_snapshot {
                    ffn_chain::encode_qwen_moe_prefill_layer_ops_with_stage_snapshots(
                        ctx,
                        &enc,
                        carrier,
                        moe,
                        &moe_weights[position],
                        &hidden_b,
                        &hidden_a,
                        &snapshot.0,
                        &snapshot.4,
                        &snapshot.5,
                        &snapshot.6,
                        &snapshot.7,
                    )
                    .map_err(qwen_moe_llama_id_error)?;
                    ffn_chain::encode_qwen_moe_prefill_stage_snapshots(
                        ctx,
                        &enc,
                        carrier,
                        &hidden_a,
                        &snapshot.1,
                        &snapshot.2,
                        &snapshot.3,
                    )
                    .map_err(qwen_moe_llama_id_error)?;
                } else {
                    ffn_chain::encode_qwen_moe_prefill_layer_ops(
                        ctx,
                        &enc,
                        carrier,
                        moe,
                        &moe_weights[position],
                        &hidden_b,
                        &hidden_a,
                    )
                    .map_err(qwen_moe_llama_id_error)?;
                }
                compute::chain_barrier(ctx, &enc);
            }
            Ok(())
        })();
        enc.endEncoding();
        encode_result?;
        let attention_profile_start = attention_sampler
            .as_ref()
            .map(|_| ffn_chain::QwenMoeLlamaIdStageSampler::calibration(ctx));

        cmd.commit();
        cmd.waitUntilCompleted();
        let status = cmd.status();
        if status != MTLCommandBufferStatus::Completed {
            let error = cmd.error().map(|error| format!("{error:?}"));
            return Err(format!(
                "Metal Qwen prefill chain command failed status={status:?} error={}",
                error.unwrap_or_else(|| "no NSError attached".to_string())
            ));
        }
        if let (Some(sampler), Some(start)) = (attention_sampler.as_ref(), attention_profile_start)
        {
            let end = ffn_chain::QwenMoeLlamaIdStageSampler::calibration(ctx);
            let mut trace = QwenMoeLlamaIdTrace::default();
            sampler
                .resolve(start, end, &mut trace)
                .map_err(|reason| format!("Metal first-attention profiler failed: {reason}"))?;
            eprintln!(
                "[RNB_METAL_FIRST_ATTENTION_PROFILE] norm_cast_ms={:.3} qkv_ms={:.3} split_norm_rope_ms={:.3} kv_cast_ms={:.3} flash_ms={:.3} gate_ms={:.3}",
                trace.map_ms.unwrap_or_default(),
                trace.gate_ms.unwrap_or_default(),
                trace.up_ms.unwrap_or_default(),
                trace.activation_ms.unwrap_or_default(),
                trace.down_ms.unwrap_or_default(),
                trace.reduce_ms.unwrap_or_default(),
            );
        }
        if let (Some(profile_input), Some(position)) =
            (attention_profile_input.as_ref(), attention_profile_position)
        {
            let (layer_idx, core) = match specs[position] {
                QwenPrefillChainSpecRef::Attention {
                    layer_idx, core, ..
                } => (layer_idx, core),
                _ => unreachable!("profile position must be attention"),
            };
            let carrier_key = attention_key(core);
            let carriers = self.qwen_prefill_atn_o_tail_carriers.borrow();
            let carrier = carriers.get(&carrier_key).ok_or_else(|| {
                "Metal Qwen prefill profile attention carrier missing".to_string()
            })?;
            let profile_output = ffn_chain::empty_f32_buf(ctx, hidden_elements);
            let kv_len = core.core.seq_len * core.core.kv_dim;
            let padded_kv_len = core.core.seq_len.next_multiple_of(64) * core.core.kv_dim;
            let profile_k = ffn_chain::empty_f16_buf_with_zeroed_tail(ctx, kv_len, padded_kv_len);
            let profile_v = ffn_chain::empty_f16_buf_with_zeroed_tail(ctx, kv_len, padded_kv_len);
            let profile_cmd = ctx.queue.commandBuffer().ok_or_else(|| {
                "Metal Qwen prefill profile command buffer creation failed".to_string()
            })?;
            let profile_enc = compute::try_chain_compute_encoder(ctx, &profile_cmd)?;
            prefill_atn_core_chain::encode_prefill_atn_o_tail_ops(
                ctx,
                &profile_enc,
                carrier,
                core,
                profile_input,
                &profile_output,
                (&profile_k, &profile_v),
            )?;
            profile_enc.endEncoding();
            profile_cmd.commit();
            profile_cmd.waitUntilCompleted();
            if profile_cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal Qwen prefill profile attention command failed".to_string());
            }
            eprintln!(
                "[RNB_METAL_FIRST_ATTENTION_COMMAND] layer={layer_idx} gpu_ms={:.3}",
                (profile_cmd.GPUEndTime() - profile_cmd.GPUStartTime()) * 1000.0,
            );
        }
        if std::env::var("RNB_METAL_PROFILE_FIRST_GDN_MOE_COMMAND").as_deref() == Ok("1") {
            let (position, layer_idx, layer, moe) = specs
                .iter()
                .enumerate()
                .find_map(|(position, spec)| match *spec {
                    QwenPrefillChainSpecRef::Gdn {
                        layer_idx,
                        layer,
                        moe,
                    } => Some((position, layer_idx, layer, moe)),
                    _ => None,
                })
                .ok_or_else(|| "Metal Qwen prefill profile GDN layer missing".to_string())?;
            let profile_input = ffn_chain::empty_f32_buf(ctx, hidden_elements);
            gdn_chain::upload_shared(&profile_input, hidden);
            let profile_gdn_output = ffn_chain::empty_f32_buf(ctx, hidden_elements);
            let conv_channels = layer
                .n_group
                .checked_mul(layer.d_state)
                .and_then(|qk| qk.checked_mul(2))
                .and_then(|qk| layer.d_inner.checked_add(qk))
                .ok_or_else(|| "Metal Qwen GDN profile conv shape overflow".to_string())?;
            let conv_len = (layer.conv_kernel_size - 1)
                .checked_mul(conv_channels)
                .ok_or_else(|| "Metal Qwen GDN profile conv state overflow".to_string())?;
            let delta_len = layer
                .d_inner
                .checked_mul(layer.d_state)
                .ok_or_else(|| "Metal Qwen GDN profile delta state overflow".to_string())?;
            let profile_conv = ffn_chain::empty_f32_buf(ctx, conv_len);
            let profile_delta = ffn_chain::empty_f32_buf(ctx, delta_len);
            let carrier_key = gdn_key(layer);
            let gdn_carriers = self.qwen_prefill_gdn_carriers.borrow();
            let gdn_carrier = gdn_carriers
                .get(&carrier_key)
                .ok_or_else(|| "Metal Qwen prefill profile GDN carrier missing".to_string())?;
            let gdn_cmd = ctx.queue.commandBuffer().ok_or_else(|| {
                "Metal Qwen prefill profile GDN command buffer creation failed".to_string()
            })?;
            let gdn_enc = compute::try_chain_compute_encoder(ctx, &gdn_cmd)?;
            gdn_chain::encode_qwen_gdn_prefill_ops(
                ctx,
                &gdn_enc,
                gdn_carrier,
                layer,
                &profile_input,
                &profile_gdn_output,
                (&profile_conv, &profile_delta),
            )?;
            gdn_enc.endEncoding();
            gdn_cmd.commit();
            gdn_cmd.waitUntilCompleted();
            if gdn_cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal Qwen prefill profile GDN command failed".to_string());
            }
            let moe_carrier = moe_carriers
                .iter()
                .find(|(plan, _)| *plan == plans[position])
                .map(|(_, carrier)| &**carrier)
                .ok_or_else(|| "Metal Qwen prefill profile MoE carrier missing".to_string())?;
            let profile_moe = QwenMoePrefillBackendSpecRef {
                ffn_norm_w: layer.post_attn_norm_w,
                ..moe
            };
            for stop in [
                ffn_chain::QwenMoeProfileStop::Route,
                ffn_chain::QwenMoeProfileStop::SparseGate,
                ffn_chain::QwenMoeProfileStop::SparseUp,
                ffn_chain::QwenMoeProfileStop::SparseActivation,
                ffn_chain::QwenMoeProfileStop::SparseDown,
                ffn_chain::QwenMoeProfileStop::SharedGateUp,
                ffn_chain::QwenMoeProfileStop::SharedActivation,
                ffn_chain::QwenMoeProfileStop::SharedDown,
            ] {
                let prefix_cmd = ctx.queue.commandBuffer().ok_or_else(|| {
                    "Metal Qwen prefill profile MoE prefix command creation failed".to_string()
                })?;
                let prefix_enc = compute::try_chain_compute_encoder(ctx, &prefix_cmd)?;
                ffn_chain::encode_qwen_moe_prefill_profile_prefix(
                    ctx,
                    &prefix_enc,
                    moe_carrier,
                    profile_moe,
                    &moe_weights[position],
                    &profile_gdn_output,
                    stop,
                )
                .map_err(qwen_moe_llama_id_error)?;
                prefix_enc.endEncoding();
                prefix_cmd.commit();
                prefix_cmd.waitUntilCompleted();
                if prefix_cmd.status() != MTLCommandBufferStatus::Completed {
                    return Err("Metal Qwen prefill profile MoE prefix failed".to_string());
                }
                eprintln!(
                    "[RNB_METAL_MOE_PREFIX] stop={stop:?} gpu_ms={:.3}",
                    (prefix_cmd.GPUEndTime() - prefix_cmd.GPUStartTime()) * 1000.0,
                );
            }
            let profile_final = ffn_chain::empty_f32_buf(ctx, hidden_elements);
            let moe_cmd = ctx.queue.commandBuffer().ok_or_else(|| {
                "Metal Qwen prefill profile MoE command buffer creation failed".to_string()
            })?;
            let moe_enc = compute::try_chain_compute_encoder(ctx, &moe_cmd)?;
            ffn_chain::encode_qwen_moe_prefill_layer_ops(
                ctx,
                &moe_enc,
                moe_carrier,
                QwenMoePrefillBackendSpecRef {
                    ffn_norm_w: layer.post_attn_norm_w,
                    ..moe
                },
                &moe_weights[position],
                &profile_gdn_output,
                &profile_final,
            )
            .map_err(qwen_moe_llama_id_error)?;
            moe_enc.endEncoding();
            moe_cmd.commit();
            moe_cmd.waitUntilCompleted();
            if moe_cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal Qwen prefill profile MoE command failed".to_string());
            }
            eprintln!(
                "[RNB_METAL_FIRST_GDN_MOE_COMMAND] layer={layer_idx} gdn_gpu_ms={:.3} moe_gpu_ms={:.3}",
                (gdn_cmd.GPUEndTime() - gdn_cmd.GPUStartTime()) * 1000.0,
                (moe_cmd.GPUEndTime() - moe_cmd.GPUStartTime()) * 1000.0,
            );
        }
        if let Some(dir) = &stage_dump_dir {
            std::fs::create_dir_all(dir).map_err(|error| {
                format!("Metal Qwen prefill stage dump directory failed: {error}")
            })?;
            let write_f32 = |name: &str, values: &[f32]| -> Result<(), String> {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                std::fs::write(std::path::Path::new(dir).join(name), bytes)
                    .map_err(|error| format!("Metal Qwen prefill stage dump failed: {error}"))
            };
            for (
                layer_idx,
                (raw_sparse, _combined, shared, final_hidden, hidden_in, norm, tpe, ids),
            ) in &stage_snapshots
            {
                let shared = ffn_chain::readback(shared, hidden_elements);
                let sparse = ffn_chain::readback(raw_sparse, hidden_elements);
                let final_hidden = ffn_chain::readback(final_hidden, hidden_elements);
                let hidden_in = ffn_chain::readback(hidden_in, hidden_elements);
                let norm = ffn_chain::readback(norm, hidden_elements);
                let tpe = ffn_chain::readback(tpe, tpe.length() / std::mem::size_of::<f32>());
                let ids = ffn_chain::readback(ids, ids.length() / std::mem::size_of::<f32>());
                write_f32(&format!("L{layer_idx}_shared.bin"), &shared)?;
                write_f32(&format!("L{layer_idx}_sparse.bin"), &sparse)?;
                write_f32(&format!("L{layer_idx}_final.bin"), &final_hidden)?;
                write_f32(&format!("L{layer_idx}_hidden_in.bin"), &hidden_in)?;
                write_f32(&format!("L{layer_idx}_norm.bin"), &norm)?;
                write_f32(&format!("L{layer_idx}_tpe.bin"), &tpe)?;
                write_f32(&format!("L{layer_idx}_ids.bin"), &ids)?;
            }
            for (
                layer_idx,
                (alpha_raw, beta_raw, alpha_processed, beta_processed, norm, hidden_input),
            ) in &gdn_stage_snapshots
            {
                let elements = alpha_raw.length() / std::mem::size_of::<f32>();
                let alpha_raw = ffn_chain::readback(alpha_raw, elements);
                let beta_raw = ffn_chain::readback(beta_raw, elements);
                let alpha_processed = ffn_chain::readback(alpha_processed, elements);
                let beta_processed = ffn_chain::readback(beta_processed, elements);
                let norm = ffn_chain::readback(norm, hidden_elements);
                let hidden_input = ffn_chain::readback(hidden_input, hidden_elements);
                write_f32(&format!("L{layer_idx}_gdn_alpha_raw.bin"), &alpha_raw)?;
                write_f32(&format!("L{layer_idx}_gdn_beta_raw.bin"), &beta_raw)?;
                write_f32(
                    &format!("L{layer_idx}_gdn_alpha_processed.bin"),
                    &alpha_processed,
                )?;
                write_f32(
                    &format!("L{layer_idx}_gdn_beta_processed.bin"),
                    &beta_processed,
                )?;
                write_f32(&format!("L{layer_idx}_gdn_norm.bin"), &norm)?;
                write_f32(&format!("L{layer_idx}_gdn_hidden_input.bin"), &hidden_input)?;
                if let Some(QwenPrefillChainSpecRef::Gdn { layer, .. }) = specs
                    .iter()
                    .find(|spec| qwen_prefill_chain_layer_index(spec) == *layer_idx)
                {
                    write_f32(
                        &format!("L{layer_idx}_gdn_attn_norm_weight.bin"),
                        layer.attn_norm_w,
                    )?;
                    write_f32(&format!("L{layer_idx}_gdn_dt_bias.bin"), layer.dt_bias)?;
                    write_f32(&format!("L{layer_idx}_gdn_ssm_a.bin"), layer.ssm_a)?;
                }
            }
        }

        let final_hidden = ffn_chain::readback(&hidden_a, hidden_elements);
        transfer_ledger.record_final_hidden_readback();
        let mut attention_kv = Vec::new();
        let mut gdn_states = Vec::new();
        for output in layer_outputs {
            match output {
                LayerDeviceOut::Attention {
                    layer_idx,
                    k,
                    v,
                    len,
                } => {
                    let k = unsafe {
                        std::slice::from_raw_parts(k.contents().as_ptr().cast::<u16>(), len)
                    }
                    .to_vec();
                    let v = unsafe {
                        std::slice::from_raw_parts(v.contents().as_ptr().cast::<u16>(), len)
                    }
                    .to_vec();
                    attention_kv.push((layer_idx, k, v));
                }
                LayerDeviceOut::Gdn {
                    layer_idx,
                    conv,
                    delta,
                    conv_len,
                    delta_len,
                } => {
                    gdn_states.push((
                        layer_idx,
                        ffn_chain::readback(&conv, conv_len),
                        ffn_chain::readback(&delta, delta_len),
                    ));
                }
            }
        }
        Ok(Some(QwenPrefillChainBackendOut {
            hidden: final_hidden,
            attention_kv,
            gdn_states,
            hidden_uploads: transfer_ledger.hidden_uploads,
            hidden_readbacks: transfer_ledger.hidden_readbacks,
            intermediate_hidden_transfers: transfer_ledger.intermediate_hidden_transfers,
        }))
    }

    /// pm50 M1: dense gated ATN core carrier. Env policy is owned by runtime; backend only
    /// checks shape/capability and performs the Metal dispatch.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_atn_core_if_supported(
        &self,
        req: PrefillAtnCoreBackendRequest<'_>,
    ) -> std::result::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
        let Some(ctx) = self.ctx.as_ref() else {
            return Ok(None);
        };
        if ctx.flash_attn_prefill_tg_pipeline.is_none()
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.prefill_split_gate_pipeline.is_none()
            || ctx.prefill_gate_apply_pipeline.is_none()
            || !Self::atn_core_tensorops_v2_ready(ctx, req.q_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.k_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.v_weight.quant)
        {
            return Ok(None);
        }

        if req.seq_len == 0 {
            return Err("Metal prefill ATN core: seq_len must be > 0".to_string());
        }
        if req.num_heads == 0 {
            return Err("Metal prefill ATN core: num_heads must be > 0".to_string());
        }
        if req.num_kv_heads == 0 {
            return Err("Metal prefill ATN core: num_kv_heads must be > 0".to_string());
        }
        if req.num_heads % req.num_kv_heads != 0 {
            return Err(format!(
                "Metal prefill ATN core: num_heads must be divisible by num_kv_heads: {} % {}",
                req.num_heads, req.num_kv_heads
            ));
        }
        Self::atn_core_require_eq("head_dim", req.head_dim, 256)?;
        Self::atn_core_require_eq("pos_start", req.pos_start, 0)?;
        if req.n_rot > req.head_dim {
            return Err(format!(
                "Metal prefill ATN core: n_rot {} exceeds head_dim {}",
                req.n_rot, req.head_dim
            ));
        }
        let expected_q_dim = Self::atn_core_checked_mul(req.num_heads, req.head_dim, "q_dim")?;
        let expected_kv_dim = Self::atn_core_checked_mul(req.num_kv_heads, req.head_dim, "kv_dim")?;
        Self::atn_core_require_eq("q_dim", req.q_dim, expected_q_dim)?;
        Self::atn_core_require_eq("kv_dim", req.kv_dim, expected_kv_dim)?;
        let expected_hidden_len =
            Self::atn_core_checked_mul(req.seq_len, req.hidden_dim, "hidden len")?;
        let expected_q_weight_rows = Self::atn_core_checked_mul(req.q_dim, 2, "q weight rows")?;
        Self::atn_core_require_eq("hidden len", req.hidden.len(), expected_hidden_len)?;
        Self::atn_core_require_eq("attn_norm_w len", req.attn_norm_w.len(), req.hidden_dim)?;
        Self::atn_core_require_eq("q_norm_w len", req.q_norm_w.len(), req.head_dim)?;
        Self::atn_core_require_eq("k_norm_w len", req.k_norm_w.len(), req.head_dim)?;
        Self::atn_core_require_eq("q weight rows", req.q_weight.rows, expected_q_weight_rows)?;
        Self::atn_core_require_eq("q weight cols", req.q_weight.cols, req.hidden_dim)?;
        Self::atn_core_require_eq("k weight rows", req.k_weight.rows, req.kv_dim)?;
        Self::atn_core_require_eq("k weight cols", req.k_weight.cols, req.hidden_dim)?;
        Self::atn_core_require_eq("v weight rows", req.v_weight.rows, req.kv_dim)?;
        Self::atn_core_require_eq("v weight cols", req.v_weight.cols, req.hidden_dim)?;
        Self::atn_core_validate_weight("q", req.q_weight)?;
        Self::atn_core_validate_weight("k", req.k_weight)?;
        Self::atn_core_validate_weight("v", req.v_weight)?;

        let (q_w_buf, q_w_off, k_w_buf, k_w_off, v_w_buf, v_w_off) = {
            let mut resident = self.resident.borrow_mut();
            let mut wrap = |raw: &[u8]| {
                let entry = resident
                    .entry(resident_key(raw))
                    .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
                (entry.0.clone(), entry.1)
            };
            let (q_w_buf, q_w_off) = wrap(req.q_weight.raw);
            let (k_w_buf, k_w_off) = wrap(req.k_weight.raw);
            let (v_w_buf, v_w_off) = wrap(req.v_weight.raw);
            (q_w_buf, q_w_off, k_w_buf, k_w_off, v_w_buf, v_w_off)
        };

        let key = AtnCoreKey {
            seq_len: req.seq_len,
            num_heads: req.num_heads,
            num_kv_heads: req.num_kv_heads,
            head_dim: req.head_dim,
            hidden_dim: req.hidden_dim,
            q_dim: req.q_dim,
            kv_dim: req.kv_dim,
            n_rot: req.n_rot,
            rope_theta_bits: req.rope_theta.to_bits(),
            scale_bits: req.scale.to_bits(),
            norm_eps_bits: req.norm_eps.to_bits(),
            pos_start: req.pos_start,
            q_quant: req.q_weight.quant,
            k_quant: req.k_weight.quant,
            v_quant: req.v_weight.quant,
        };
        let mut carriers = self.prefill_atn_core_carriers.borrow_mut();
        let carrier = carriers.entry(key).or_insert_with(|| {
            prefill_atn_core_chain::PrefillAtnCoreCarrier::new(
                ctx,
                req.seq_len,
                req.num_heads,
                req.num_kv_heads,
                req.head_dim,
                req.hidden_dim,
                req.q_dim,
                req.kv_dim,
                req.n_rot,
                req.rope_theta,
                req.scale,
                req.norm_eps,
                req.pos_start,
            )
        });
        let out = prefill_atn_core_chain::prefill_atn_core_dispatch(
            ctx,
            carrier,
            prefill_atn_core_chain::PrefillAtnCoreDispatchRequest {
                hidden: req.hidden,
                attn_norm_w: req.attn_norm_w,
                q_norm_w: req.q_norm_w,
                k_norm_w: req.k_norm_w,
                q_w_buf: &q_w_buf,
                q_w_off,
                q_quant: req.q_weight.quant,
                k_w_buf: &k_w_buf,
                k_w_off,
                k_quant: req.k_weight.quant,
                v_w_buf: &v_w_buf,
                v_w_off,
                v_quant: req.v_weight.quant,
            },
        )?;
        Ok(Some(out))
    }

    /// pm108: dense gated ATN o-tail carrier. ATN core 결과를 host로 읽지 않고
    /// o_proj + residual까지 같은 device buffer 흐름으로 잇는다.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_atn_o_tail_if_supported(
        &self,
        req: PrefillAtnOTailBackendRequest<'_>,
    ) -> std::result::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
        let Some(ctx) = self.ctx.as_ref() else {
            return Ok(None);
        };
        let core = req.core;
        if ctx.flash_attn_prefill_tg_pipeline.is_none()
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.prefill_split_gate_pipeline.is_none()
            || ctx.prefill_gate_apply_pipeline.is_none()
            || !Self::atn_core_tensorops_v2_ready(ctx, core.q_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.k_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.v_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.o_weight.quant)
        {
            return Ok(None);
        }

        if core.seq_len == 0 {
            return Err("Metal prefill ATN o-tail: seq_len must be > 0".to_string());
        }
        if core.num_heads == 0 {
            return Err("Metal prefill ATN o-tail: num_heads must be > 0".to_string());
        }
        if core.num_kv_heads == 0 {
            return Err("Metal prefill ATN o-tail: num_kv_heads must be > 0".to_string());
        }
        if core.num_heads % core.num_kv_heads != 0 {
            return Err(format!(
                "Metal prefill ATN o-tail: num_heads must be divisible by num_kv_heads: {} % {}",
                core.num_heads, core.num_kv_heads
            ));
        }
        Self::atn_core_require_eq("head_dim", core.head_dim, 256)?;
        Self::atn_core_require_eq("pos_start", core.pos_start, 0)?;
        if core.n_rot > core.head_dim {
            return Err(format!(
                "Metal prefill ATN o-tail: n_rot {} exceeds head_dim {}",
                core.n_rot, core.head_dim
            ));
        }
        let expected_q_dim = Self::atn_core_checked_mul(core.num_heads, core.head_dim, "q_dim")?;
        let expected_kv_dim =
            Self::atn_core_checked_mul(core.num_kv_heads, core.head_dim, "kv_dim")?;
        Self::atn_core_require_eq("q_dim", core.q_dim, expected_q_dim)?;
        Self::atn_core_require_eq("kv_dim", core.kv_dim, expected_kv_dim)?;
        let expected_hidden_len =
            Self::atn_core_checked_mul(core.seq_len, core.hidden_dim, "hidden len")?;
        let expected_q_weight_rows = Self::atn_core_checked_mul(core.q_dim, 2, "q weight rows")?;
        Self::atn_core_require_eq("hidden len", core.hidden.len(), expected_hidden_len)?;
        Self::atn_core_require_eq("attn_norm_w len", core.attn_norm_w.len(), core.hidden_dim)?;
        Self::atn_core_require_eq("q_norm_w len", core.q_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("k_norm_w len", core.k_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("q weight rows", core.q_weight.rows, expected_q_weight_rows)?;
        Self::atn_core_require_eq("q weight cols", core.q_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("k weight rows", core.k_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("k weight cols", core.k_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("v weight rows", core.v_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("v weight cols", core.v_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight rows", req.o_weight.rows, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight cols", req.o_weight.cols, core.q_dim)?;
        Self::atn_core_validate_weight("q", core.q_weight)?;
        Self::atn_core_validate_weight("k", core.k_weight)?;
        Self::atn_core_validate_weight("v", core.v_weight)?;
        Self::atn_core_validate_weight("o", req.o_weight)?;

        let core_key = AtnCoreKey {
            seq_len: core.seq_len,
            num_heads: core.num_heads,
            num_kv_heads: core.num_kv_heads,
            head_dim: core.head_dim,
            hidden_dim: core.hidden_dim,
            q_dim: core.q_dim,
            kv_dim: core.kv_dim,
            n_rot: core.n_rot,
            rope_theta_bits: core.rope_theta.to_bits(),
            scale_bits: core.scale.to_bits(),
            norm_eps_bits: core.norm_eps.to_bits(),
            pos_start: core.pos_start,
            q_quant: core.q_weight.quant,
            k_quant: core.k_weight.quant,
            v_quant: core.v_weight.quant,
        };
        let key = AtnOTailKey {
            core: core_key,
            o_quant: req.o_weight.quant,
        };
        let mut carriers = self.prefill_atn_o_tail_carriers.borrow_mut();
        let carrier = carriers.entry(key).or_insert_with(|| {
            prefill_atn_core_chain::PrefillAtnOTailCarrier::new(
                ctx,
                core.seq_len,
                core.num_heads,
                core.num_kv_heads,
                core.head_dim,
                core.hidden_dim,
                core.q_dim,
                core.kv_dim,
                core.n_rot,
                core.rope_theta,
                core.scale,
                core.norm_eps,
                core.pos_start,
            )
        });
        let out = prefill_atn_core_chain::prefill_atn_o_tail_dispatch(
            ctx,
            carrier,
            prefill_atn_core_chain::PrefillAtnOTailDispatchRequest {
                hidden: core.hidden,
                spec: PrefillAtnOTailBackendSpecRef {
                    core: PrefillAtnCoreBackendSpecRef {
                        attn_norm_w: core.attn_norm_w,
                        q_norm_w: core.q_norm_w,
                        k_norm_w: core.k_norm_w,
                        q_weight: core.q_weight,
                        k_weight: core.k_weight,
                        v_weight: core.v_weight,
                        seq_len: core.seq_len,
                        num_heads: core.num_heads,
                        num_kv_heads: core.num_kv_heads,
                        head_dim: core.head_dim,
                        hidden_dim: core.hidden_dim,
                        q_dim: core.q_dim,
                        kv_dim: core.kv_dim,
                        n_rot: core.n_rot,
                        rope_theta: core.rope_theta,
                        scale: core.scale,
                        norm_eps: core.norm_eps,
                        pos_start: core.pos_start,
                    },
                    o_weight: req.o_weight,
                },
            },
        )?;
        Ok(Some(out))
    }

    /// pm70: dense gated ATN full layer carrier. ATN core 결과를 host로 읽지 않고
    /// o_proj + FFN tail까지 같은 device buffer 흐름으로 잇는다.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_atn_full_layer_if_supported(
        &self,
        req: PrefillAtnFullLayerBackendRequest<'_>,
    ) -> std::result::Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
        let Some(ctx) = self.ctx.as_ref() else {
            return Ok(None);
        };
        let core = req.core;
        if ctx.flash_attn_prefill_tg_pipeline.is_none()
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.prefill_split_gate_pipeline.is_none()
            || ctx.prefill_gate_apply_pipeline.is_none()
            || !Self::atn_core_tensorops_v2_ready(ctx, core.q_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.k_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, core.v_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.o_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.ffn_gate_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.ffn_up_weight.quant)
            || !Self::atn_core_tensorops_v2_ready(ctx, req.ffn_down_weight.quant)
        {
            return Ok(None);
        }

        if core.seq_len == 0 {
            return Err("Metal prefill ATN full layer: seq_len must be > 0".to_string());
        }
        if core.num_heads == 0 {
            return Err("Metal prefill ATN full layer: num_heads must be > 0".to_string());
        }
        if core.num_kv_heads == 0 {
            return Err("Metal prefill ATN full layer: num_kv_heads must be > 0".to_string());
        }
        if core.num_heads % core.num_kv_heads != 0 {
            return Err(format!(
                "Metal prefill ATN full layer: num_heads must be divisible by num_kv_heads: {} % {}",
                core.num_heads, core.num_kv_heads
            ));
        }
        Self::atn_core_require_eq("head_dim", core.head_dim, 256)?;
        Self::atn_core_require_eq("pos_start", core.pos_start, 0)?;
        if core.n_rot > core.head_dim {
            return Err(format!(
                "Metal prefill ATN full layer: n_rot {} exceeds head_dim {}",
                core.n_rot, core.head_dim
            ));
        }
        let expected_q_dim = Self::atn_core_checked_mul(core.num_heads, core.head_dim, "q_dim")?;
        let expected_kv_dim =
            Self::atn_core_checked_mul(core.num_kv_heads, core.head_dim, "kv_dim")?;
        Self::atn_core_require_eq("q_dim", core.q_dim, expected_q_dim)?;
        Self::atn_core_require_eq("kv_dim", core.kv_dim, expected_kv_dim)?;
        let expected_hidden_len =
            Self::atn_core_checked_mul(core.seq_len, core.hidden_dim, "hidden len")?;
        let expected_q_weight_rows = Self::atn_core_checked_mul(core.q_dim, 2, "q weight rows")?;
        Self::atn_core_require_eq("hidden len", core.hidden.len(), expected_hidden_len)?;
        Self::atn_core_require_eq("attn_norm_w len", core.attn_norm_w.len(), core.hidden_dim)?;
        Self::atn_core_require_eq("q_norm_w len", core.q_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("k_norm_w len", core.k_norm_w.len(), core.head_dim)?;
        Self::atn_core_require_eq("ffn_norm_w len", req.ffn_norm_w.len(), core.hidden_dim)?;
        Self::atn_core_require_eq("q weight rows", core.q_weight.rows, expected_q_weight_rows)?;
        Self::atn_core_require_eq("q weight cols", core.q_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("k weight rows", core.k_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("k weight cols", core.k_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("v weight rows", core.v_weight.rows, core.kv_dim)?;
        Self::atn_core_require_eq("v weight cols", core.v_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight rows", req.o_weight.rows, core.hidden_dim)?;
        Self::atn_core_require_eq("o weight cols", req.o_weight.cols, core.q_dim)?;
        Self::atn_core_require_eq("ffn gate rows", req.ffn_gate_weight.rows, req.ffn_dim)?;
        Self::atn_core_require_eq("ffn gate cols", req.ffn_gate_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("ffn up rows", req.ffn_up_weight.rows, req.ffn_dim)?;
        Self::atn_core_require_eq("ffn up cols", req.ffn_up_weight.cols, core.hidden_dim)?;
        Self::atn_core_require_eq("ffn down rows", req.ffn_down_weight.rows, core.hidden_dim)?;
        Self::atn_core_require_eq("ffn down cols", req.ffn_down_weight.cols, req.ffn_dim)?;
        Self::atn_core_validate_weight("q", core.q_weight)?;
        Self::atn_core_validate_weight("k", core.k_weight)?;
        Self::atn_core_validate_weight("v", core.v_weight)?;
        Self::atn_core_validate_weight("o", req.o_weight)?;
        Self::atn_core_validate_weight("ffn gate", req.ffn_gate_weight)?;
        Self::atn_core_validate_weight("ffn up", req.ffn_up_weight)?;
        Self::atn_core_validate_weight("ffn down", req.ffn_down_weight)?;

        let (
            q_w_buf,
            q_w_off,
            k_w_buf,
            k_w_off,
            v_w_buf,
            v_w_off,
            o_w_buf,
            o_w_off,
            ffn_gate_w_buf,
            ffn_gate_w_off,
            ffn_up_w_buf,
            ffn_up_w_off,
            ffn_down_w_buf,
            ffn_down_w_off,
        ) = {
            let mut resident = self.resident.borrow_mut();
            let mut wrap = |raw: &[u8]| {
                let entry = resident
                    .entry(resident_key(raw))
                    .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
                (entry.0.clone(), entry.1)
            };
            let (q_w_buf, q_w_off) = wrap(core.q_weight.raw);
            let (k_w_buf, k_w_off) = wrap(core.k_weight.raw);
            let (v_w_buf, v_w_off) = wrap(core.v_weight.raw);
            let (o_w_buf, o_w_off) = wrap(req.o_weight.raw);
            let (ffn_gate_w_buf, ffn_gate_w_off) = wrap(req.ffn_gate_weight.raw);
            let (ffn_up_w_buf, ffn_up_w_off) = wrap(req.ffn_up_weight.raw);
            let (ffn_down_w_buf, ffn_down_w_off) = wrap(req.ffn_down_weight.raw);
            (
                q_w_buf,
                q_w_off,
                k_w_buf,
                k_w_off,
                v_w_buf,
                v_w_off,
                o_w_buf,
                o_w_off,
                ffn_gate_w_buf,
                ffn_gate_w_off,
                ffn_up_w_buf,
                ffn_up_w_off,
                ffn_down_w_buf,
                ffn_down_w_off,
            )
        };

        let core_key = AtnCoreKey {
            seq_len: core.seq_len,
            num_heads: core.num_heads,
            num_kv_heads: core.num_kv_heads,
            head_dim: core.head_dim,
            hidden_dim: core.hidden_dim,
            q_dim: core.q_dim,
            kv_dim: core.kv_dim,
            n_rot: core.n_rot,
            rope_theta_bits: core.rope_theta.to_bits(),
            scale_bits: core.scale.to_bits(),
            norm_eps_bits: core.norm_eps.to_bits(),
            pos_start: core.pos_start,
            q_quant: core.q_weight.quant,
            k_quant: core.k_weight.quant,
            v_quant: core.v_weight.quant,
        };
        let key = AtnFullLayerKey {
            core: core_key,
            ffn_dim: req.ffn_dim,
            o_quant: req.o_weight.quant,
            ffn_gate_quant: req.ffn_gate_weight.quant,
            ffn_up_quant: req.ffn_up_weight.quant,
            ffn_down_quant: req.ffn_down_weight.quant,
        };
        let mut carriers = self.prefill_atn_full_layer_carriers.borrow_mut();
        let carrier = carriers.entry(key).or_insert_with(|| {
            prefill_atn_core_chain::PrefillAtnFullLayerCarrier::new(
                ctx,
                core.seq_len,
                core.num_heads,
                core.num_kv_heads,
                core.head_dim,
                core.hidden_dim,
                core.q_dim,
                core.kv_dim,
                req.ffn_dim,
                core.n_rot,
                core.rope_theta,
                core.scale,
                core.norm_eps,
                core.pos_start,
            )
        });
        let out = prefill_atn_core_chain::prefill_atn_full_layer_dispatch(
            ctx,
            carrier,
            prefill_atn_core_chain::PrefillAtnFullLayerDispatchRequest {
                core: prefill_atn_core_chain::PrefillAtnCoreDispatchRequest {
                    hidden: core.hidden,
                    attn_norm_w: core.attn_norm_w,
                    q_norm_w: core.q_norm_w,
                    k_norm_w: core.k_norm_w,
                    q_w_buf: &q_w_buf,
                    q_w_off,
                    q_quant: core.q_weight.quant,
                    k_w_buf: &k_w_buf,
                    k_w_off,
                    k_quant: core.k_weight.quant,
                    v_w_buf: &v_w_buf,
                    v_w_off,
                    v_quant: core.v_weight.quant,
                },
                o_w_buf: &o_w_buf,
                o_w_off,
                o_quant: req.o_weight.quant,
                ffn_norm_w: req.ffn_norm_w,
                ffn_gate_w_buf: &ffn_gate_w_buf,
                ffn_gate_w_off,
                ffn_gate_quant: req.ffn_gate_weight.quant,
                ffn_up_w_buf: &ffn_up_w_buf,
                ffn_up_w_off,
                ffn_up_quant: req.ffn_up_weight.quant,
                ffn_down_w_buf: &ffn_down_w_buf,
                ffn_down_w_off,
                ffn_down_quant: req.ffn_down_weight.quant,
            },
        )?;
        Ok(Some(out))
    }

    /// Per-row L2 normalize (+ scale): `input`=[n_rows*dim], 반환=[n_rows*dim].
    /// `scale`=1.0 이면 CPU `l2_norm_into` 와 token-identical(GDN k). q 는 1/√head_k_dim.
    pub fn l2_norm(
        &self,
        input: &[f32],
        n_rows: usize,
        dim: usize,
        eps: f32,
        scale: f32,
    ) -> Vec<f32> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::l2_norm_with_ctx(ctx, input, n_rows, dim, eps, scale)
    }

    /// GDN alpha/beta 전처리: `beta=sigmoid(beta)`, `alpha=softplus(alpha+dt_bias)*ssm_a`.
    /// 반환 `(alpha_after, beta_after)`. CPU `decode_gdn_layer_qwen` 전처리와 token-identical.
    pub fn gdn_alpha_beta(
        &self,
        alpha: &[f32],
        beta: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        num_heads: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::gdn_alpha_beta_with_ctx(ctx, alpha, beta, dt_bias, ssm_a, num_heads)
    }

    /// Q8_0 GEMV (검증용 host 복사 경로). block_bytes = N*(K/32)*34.
    pub fn gemv_q8_0(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gemv_q8_0_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q5_K GEMV (검증용 host 복사 경로). block_bytes = N*(K/256)*176.
    pub fn gemv_q5k(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gemv_q5k_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q5_K GEMV (SIMD-group + pm21 P1 lane-saturation, 검증용 host 복사 경로).
    /// block_bytes = N*(K/256)*176.
    pub fn gemv_q5k_simd(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gemv_q5k_simd_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q6_K GEMV (검증용 host 복사 경로, baseline 1-thread/row 커널).
    /// block_bytes = N*(K/256)*210.
    pub fn gemv_q6k(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gemv_q6k_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q6_K GEMV (SIMD-group 협력 커널, 검증용 host 복사 경로).
    /// block_bytes = N*(K/256)*210.
    pub fn gemv_q6k_simd(&self, block_bytes: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gemv_q6k_simd_with_ctx(ctx, block_bytes, input, n, k)
    }

    /// Q6_K GEMV (SIMD-group 협력 + resident NoCopy weight cache).
    /// output projection/FFN down 같이 같은 mmap weight 를 반복 호출하는 decode 경로용.
    pub fn gemv_q6k_simd_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let key = resident_key(raw);
        let mut cache = self.resident.borrow_mut();
        let entry = cache
            .entry(key)
            .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
        compute::gemv_q6k_simd_dispatch(ctx, &entry.0, entry.1, input, n, k)
    }

    fn output_argmax_quant_simd_resident(
        &self,
        quant: u8,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> OutputArgmaxReport {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let key = resident_key(raw);
        let (weight_buf, weight_off) = {
            let mut cache = self.resident.borrow_mut();
            let entry = cache
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        let use_scratch = std::env::var("RNB_METAL_OUTPUT_ARGMAX_SCRATCH")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true);
        let dispatch = if use_scratch {
            let mut carriers = self.output_argmax_carriers.borrow_mut();
            let scratch = carriers
                .entry((n, k))
                .or_insert_with(|| compute::OutputArgmaxScratch::new(ctx, n, k));
            compute::gemv_quant_simd_argmax_dispatch_reuse(
                ctx,
                quant,
                &weight_buf,
                weight_off,
                input,
                n,
                k,
                scratch,
            )
        } else {
            compute::gemv_quant_simd_argmax_dispatch_once(
                ctx,
                quant,
                &weight_buf,
                weight_off,
                input,
                n,
                k,
            )
        };
        OutputArgmaxReport {
            attempted: true,
            did_run: true,
            token_id: Some(dispatch.token_id),
            cmd_buffers: 1,
            wait_us: dispatch.wait_us,
            readback_bytes: std::mem::size_of::<u32>(),
            fallback_reason: None,
        }
    }

    /// Q4_K output projection + GPU-side argmax. Reads back only one u32 token id.
    pub fn output_argmax_q4k_simd_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> OutputArgmaxReport {
        self.output_argmax_quant_simd_resident(0, raw, input, n, k)
    }

    /// Q6_K output projection + GPU-side argmax. Reads back only one u32 token id.
    pub fn output_argmax_q6k_simd_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> OutputArgmaxReport {
        self.output_argmax_quant_simd_resident(2, raw, input, n, k)
    }

    /// gated attention q/gate split(검증용 host 복사 경로). `q_full`=[num_heads*hd*2]
    /// (head 별 [query hd | gate hd] 인터리브). 반환 `(query, gate)` 각 [num_heads*hd].
    /// CPU split(`decode_attention_post_qkv.rs:34-41`)과 bit-identical.
    pub fn metal_split_qgate_into(
        &self,
        q_full: &[f32],
        num_heads: usize,
        hd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::split_qgate_with_ctx(ctx, q_full, num_heads, hd)
    }

    /// Partial RoPE(인접페어, decode 1 token) device 커널. CPU
    /// `rope_partial_inplace`(rope.rs:387)와 일치(theta_scale host f32 precompute).
    /// production Qwen3.5 9B 의 attention RoPE 경로. 측정/테스트용 복사 경로.
    pub fn metal_rope_partial_into(
        &self,
        input: &[f32],
        head_dim: usize,
        dim: usize,
        n_rot: usize,
        theta: f32,
        pos: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::rope_partial_with_ctx(ctx, input, head_dim, dim, n_rot, theta, pos)
    }

    /// gated attention `attn_out *= sigmoid(gate)`(elementwise) device 커널. CPU
    /// `sigmoid_inplace`+`mul_inplace`(decode.rs:751-757)와 일치. 측정/테스트용 복사 경로.
    pub fn metal_gate_apply_into(&self, attn: &[f32], gate: &[f32]) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::gate_apply_with_ctx(ctx, attn, gate)
    }

    /// k/v(f32) step 들을 device KV buffer 에 순차 f16 append 하고 채운 구간 readback.
    /// kv_append 커널 검증/측정용. carrier 의 device KV append 부품.
    pub fn kv_append_steps(
        &self,
        k_steps: &[Vec<f32>],
        v_steps: &[Vec<f32>],
        kv_dim: usize,
        capacity: usize,
    ) -> (Vec<u16>, Vec<u16>) {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        compute::kv_append_steps_with_ctx(ctx, k_steps, v_steps, kv_dim, capacity)
    }

    /// KV residency attention: host KV(전체 kv_len, f16 bits)에서 device 에 아직
    /// 없는 token 만 incremental 복사(첫 토큰=prefill 전체, 이후=1개씩) 후 q 로
    /// attention compute. 매 토큰 전체 KV 업로드를 제거한다. capacity=max_seq_len.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_kv_resident(
        &self,
        layer: usize,
        k_all: &[u16],
        v_all: &[u16],
        q: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
        capacity: usize,
    ) -> Option<Vec<f32>> {
        let ctx = self
            .ctx
            .as_ref()
            .expect("MetalBackend: no Metal context — device was not available at new()");
        {
            let mut cache = self.kv_residents.borrow_mut();
            let resident_capacity = attn_resident_capacity_for(kv_len, capacity);
            let recreate = cache
                .get(&layer)
                .map(|entry| {
                    entry.kv_int8
                        || entry.num_kv_heads != num_kv_heads
                        || entry.head_dim != head_dim
                        || entry.capacity < kv_len
                })
                .unwrap_or(true);
            if recreate {
                cache.insert(
                    layer,
                    compute::KvResident::new_f16(ctx, num_kv_heads, head_dim, resident_capacity),
                );
            }
            let entry = cache
                .get_mut(&layer)
                .expect("MetalBackend: KV resident not initialized for layer");
            entry.ensure_filled(k_all, v_all, kv_len);
        }
        let cache = self.kv_residents.borrow();
        let kv = cache
            .get(&layer)
            .expect("MetalBackend: KV resident not initialized for layer");
        if kv.kv_int8 {
            return None;
        }
        Some(compute::attn_decode_resident_dispatch(
            ctx,
            q,
            kv,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
        ))
    }

    pub fn kvarn_attention_decode(
        &self,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
    ) -> Result<Vec<f32>, String> {
        request.validate()?;
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| "MetalBackend: no Metal context".to_string())?;
        let mut cache = self.kvarn_residents.borrow_mut();
        let recreate = cache
            .get(&request.layer_idx())
            .map(|entry| !entry.matches(request))
            .unwrap_or(true);
        if recreate {
            cache.insert(
                request.layer_idx(),
                compute::KvarnResident::new(ctx, request)?,
            );
        }
        let resident = cache
            .get_mut(&request.layer_idx())
            .ok_or_else(|| "MetalBackend: KVarN resident initialization failed".to_string())?;
        compute::kvarn_attention_decode_with_ctx(ctx, resident, request)
    }

    /// zero-copy NoCopy Q4_K GEMV. `raw` 는 mmap 내부 포인터(loader 소유)로,
    /// 복사 없이 `newBufferWithBytesNoCopy` 로 wrap 해 GPU 가 직접 읽는다.
    /// weight `(ptr,len)` 로 resident 캐싱 — 첫 호출만 wrap(복사 0), 이후 재사용.
    pub fn gemv_q4k_resident(&self, raw: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let key = resident_key(raw);
        let mut cache = self.resident.borrow_mut();
        let entry = cache
            .entry(key)
            .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
        compute::gemv_q4k_dispatch(ctx, &entry.0, entry.1, input, n, k)
    }

    /// SIMD-group 협력 커널 버전의 zero-copy NoCopy Q4_K GEMV.
    /// `gemv_q4k_resident` 와 동일 resident 캐시 사용, dispatch 만 SIMD 협력 커널.
    pub fn gemv_q4k_simd_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let key = resident_key(raw);
        let mut cache = self.resident.borrow_mut();
        let entry = cache
            .entry(key)
            .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
        compute::gemv_q4k_simd_dispatch(ctx, &entry.0, entry.1, input, n, k)
    }

    /// pm112: GLM MLA decode 용 Q5_K GEMV. weight 는 `(ptr,len)` resident 캐시로
    /// 1회 zero-copy wrap, 이후 input 만 업로드.
    pub fn gemv_q5k_mla_resident(&self, raw: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let key = resident_key(raw);
        let (buf, off) = {
            let mut cache = self.resident.borrow_mut();
            let entry = cache
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        self.touch_weight_residency(key, &buf);
        compute::gemv_mla_dispatch(ctx, compute::MlaGemvQuant::Q5K, &buf, off, input, n, k)
    }

    /// pm112: GLM MLA decode 용 Q8_0 GEMV. resident 캐시는 `gemv_q5k_mla_resident` 와 동일.
    pub fn gemv_q8_0_mla_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let key = resident_key(raw);
        let (buf, off) = {
            let mut cache = self.resident.borrow_mut();
            let entry = cache
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        self.touch_weight_residency(key, &buf);
        compute::gemv_mla_dispatch(ctx, compute::MlaGemvQuant::Q8_0, &buf, off, input, n, k)
    }

    /// pm112: GLM MLA per-head Q8_0 GEMV 배치 (k_b/v_b). weight resident wrap,
    /// head 오프셋 encode `heads` 개 + commit/wait 1회.
    pub fn glm_mla_head_gemv_q8_0_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        heads: usize,
        n_per_head: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let key = resident_key(raw);
        let (buf, off) = {
            let mut cache = self.resident.borrow_mut();
            let entry = cache
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        self.touch_weight_residency(key, &buf);
        compute::glm_mla_head_gemv_q8_0_dispatch(ctx, &buf, off, input, heads, n_per_head, k)
    }

    /// pm113: GLM MLA prefill slot-batch GEMV (slot = token*heads+head, 단일 dispatch).
    pub fn glm_mla_head_gemv_q8_0_slots_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        slots: usize,
        heads: usize,
        n_per_head: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let (buf, off) = self.glm_mla_wrap(ctx, raw);
        compute::glm_mla_head_gemv_q8_0_slots_dispatch(
            ctx, &buf, off, input, slots, heads, n_per_head, k,
        )
    }

    /// pm117: GLM MLA prefill slot-batch Q5_K GEMV (o/q_a — Q8_0 slots 와 동일 구조).
    pub fn glm_mla_head_gemv_q5k_slots_resident(
        &self,
        raw: &[u8],
        input: &[f32],
        slots: usize,
        heads: usize,
        n_per_head: usize,
        k: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let (buf, off) = self.glm_mla_wrap(ctx, raw);
        compute::glm_mla_head_gemv_q5k_slots_dispatch(
            ctx, &buf, off, input, slots, heads, n_per_head, k,
        )
    }

    /// pm119: MLA 층 전체 (front→rope→attn→v_b→o) 단일 command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_layer_fused_resident(
        &self,
        qa_raw: &[u8],
        qa_norm_weight: &[f32],
        qb_raw: &[u8],
        kva_raw: &[u8],
        kb_raw: &[u8],
        kv_norm_weight: &[f32],
        cache_base: &[u16],
        vb_raw: &[u8],
        o_raw: &[u8],
        o_is_q5k: bool,
        normed: &[f32],
        seq_len: usize,
        hidden: usize,
        q_rank: usize,
        q_dim: usize,
        kv_width: usize,
        heads: usize,
        qk_dim: usize,
        q_nope_dim: usize,
        kv_rank: usize,
        rope_dim: usize,
        pos_start: usize,
        value_dim: usize,
        o_rows: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> compute::GlmMlaLayerFusedOut {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let (qa_buf, qa_off) = self.glm_mla_wrap(ctx, qa_raw);
        let (qb_buf, qb_off) = self.glm_mla_wrap(ctx, qb_raw);
        let (kva_buf, kva_off) = self.glm_mla_wrap(ctx, kva_raw);
        let (kb_buf, kb_off) = self.glm_mla_wrap(ctx, kb_raw);
        let (vb_buf, vb_off) = self.glm_mla_wrap(ctx, vb_raw);
        let (o_buf, o_off) = self.glm_mla_wrap(ctx, o_raw);
        compute::glm_mla_layer_fused_dispatch(
            ctx,
            &qa_buf,
            qa_off,
            qa_norm_weight,
            &qb_buf,
            qb_off,
            &kva_buf,
            kva_off,
            &kb_buf,
            kb_off,
            kv_norm_weight,
            cache_base,
            &vb_buf,
            vb_off,
            &o_buf,
            o_off,
            o_is_q5k,
            normed,
            seq_len,
            hidden,
            q_rank,
            q_dim,
            kv_width,
            heads,
            qk_dim,
            q_nope_dim,
            kv_rank,
            rope_dim,
            pos_start,
            value_dim,
            o_rows,
            eps,
            theta,
            scale,
        )
    }

    /// pm118 연장: front 4 dispatch (q_a→rms→q_b/kv_a→pack→k_b) 단일 command
    /// buffer. 반환 (q, kv_raw, q_absorbed).
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_front_slots_fused_resident(
        &self,
        qa_raw: &[u8],
        qa_norm_weight: &[f32],
        qb_raw: &[u8],
        kva_raw: &[u8],
        kb_raw: &[u8],
        normed: &[f32],
        seq_len: usize,
        hidden: usize,
        q_rank: usize,
        q_dim: usize,
        kv_width: usize,
        heads: usize,
        qk_dim: usize,
        q_nope_dim: usize,
        kv_rank: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let (qa_buf, qa_off) = self.glm_mla_wrap(ctx, qa_raw);
        let (qb_buf, qb_off) = self.glm_mla_wrap(ctx, qb_raw);
        let (kva_buf, kva_off) = self.glm_mla_wrap(ctx, kva_raw);
        let (kb_buf, kb_off) = self.glm_mla_wrap(ctx, kb_raw);
        compute::glm_mla_front_slots_fused_dispatch(
            ctx,
            &qa_buf,
            qa_off,
            qa_norm_weight,
            &qb_buf,
            qb_off,
            &kva_buf,
            kva_off,
            &kb_buf,
            kb_off,
            normed,
            seq_len,
            hidden,
            q_rank,
            q_dim,
            kv_width,
            heads,
            qk_dim,
            q_nope_dim,
            kv_rank,
            eps,
        )
    }

    /// pm118: v_b slots → o slots 단일 command buffer (중간 concat device-resident).
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_vb_o_fused_resident(
        &self,
        vb_raw: &[u8],
        o_raw: &[u8],
        o_is_q5k: bool,
        latent: &[f32],
        slots: usize,
        heads: usize,
        value_dim: usize,
        kv_rank: usize,
        o_rows: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let (vb_buf, vb_off) = self.glm_mla_wrap(ctx, vb_raw);
        let (o_buf, o_off) = self.glm_mla_wrap(ctx, o_raw);
        compute::glm_mla_vb_o_fused_dispatch(
            ctx, &vb_buf, vb_off, &o_buf, o_off, o_is_q5k, latent, slots, heads, value_dim,
            kv_rank, o_rows,
        )
    }

    /// pm116: GLM MLA prefill attention (absorbed form) 단일 dispatch.
    /// weight 가 없어 residency wrap 불필요 — 입력 3개는 호출마다 업로드.
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_prefill_attn(
        &self,
        q_absorbed: &[f32],
        q_pe: &[f32],
        cache: &[u16],
        slots: usize,
        head_count: usize,
        kv_rank: usize,
        rope_dim: usize,
        pos_start: usize,
        scale: f32,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        compute::glm_mla_prefill_attn_dispatch(
            ctx, q_absorbed, q_pe, cache, slots, head_count, kv_rank, rope_dim, pos_start, scale,
        )
    }

    /// pm112: MLA weight 를 resident wrap + residency 등록하고 (buffer, offset) 반환.
    #[allow(clippy::type_complexity)]
    fn glm_mla_wrap(
        &self,
        ctx: &compute::MetalContext,
        raw: &[u8],
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u32) {
        let key = resident_key(raw);
        let (buf, off) = {
            let mut cache = self.resident.borrow_mut();
            let entry = cache
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        self.touch_weight_residency(key, &buf);
        (buf, off)
    }

    /// pm112: GLM MLA front chain (q_a→rms→q_b, kv_a, k_b 64-head) 단일 command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_front(
        &self,
        qa: &[u8],
        qa_norm: &[u8],
        qb: &[u8],
        kva: &[u8],
        kb: &[u8],
        hidden: usize,
        q_rank: usize,
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
        kv_rank: usize,
        q_nope: usize,
        value_dim: usize,
        eps: f32,
        input: &[f32],
        q_out: &mut [f32],
        kv_out: &mut [f32],
        qabs_out: &mut [f32],
    ) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let qa_w = self.glm_mla_wrap(ctx, qa);
        let qa_norm_w = self.glm_mla_wrap(ctx, qa_norm);
        let qb_w = self.glm_mla_wrap(ctx, qb);
        let kva_w = self.glm_mla_wrap(ctx, kva);
        let kb_w = self.glm_mla_wrap(ctx, kb);
        let mut carriers = self.glm_mla_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden, q_dim, kv_rank, heads))
            .or_insert_with(|| {
                ffn_chain::GlmMlaCarrier::new(
                    ctx, hidden, q_rank, q_dim, kv_dim, heads, kv_rank, q_nope, value_dim, eps,
                )
            });
        ffn_chain::glm_mla_front_dispatch(
            ctx,
            carrier,
            (&qa_w.0, qa_w.1),
            (&qa_norm_w.0, qa_norm_w.1),
            (&qb_w.0, qb_w.1),
            (&kva_w.0, kva_w.1),
            (&kb_w.0, kb_w.1),
            input,
            q_out,
            kv_out,
            qabs_out,
        );
    }

    /// pm112: GLM MLA back chain (v_b 64-head → o) 단일 command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn glm_mla_back(
        &self,
        vb: &[u8],
        o: &[u8],
        hidden: usize,
        q_rank: usize,
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
        kv_rank: usize,
        q_nope: usize,
        value_dim: usize,
        eps: f32,
        latent: &[f32],
        out: &mut [f32],
    ) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        self.ensure_weight_residency(ctx);
        let vb_w = self.glm_mla_wrap(ctx, vb);
        let o_w = self.glm_mla_wrap(ctx, o);
        let mut carriers = self.glm_mla_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden, q_dim, kv_rank, heads))
            .or_insert_with(|| {
                ffn_chain::GlmMlaCarrier::new(
                    ctx, hidden, q_rank, q_dim, kv_dim, heads, kv_rank, q_nope, value_dim, eps,
                )
            });
        ffn_chain::glm_mla_back_dispatch(
            ctx,
            carrier,
            (&vb_w.0, vb_w.1),
            (&o_w.0, o_w.1),
            latent,
            out,
        );
    }

    /// pm33: prefill FFN chain(M>1). normed[seq_len*hidden] + gate/up(Q4_K)/down(Q4_K|Q6_K)
    /// raw weight → down 결과[seq_len*hidden](residual 전). weight 는 `(ptr,len)` 키로 resident wrap,
    /// carrier 는 (hidden,ffn,seq_len) 키로 1회 alloc. 단일 command buffer batch GEMM chain.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_ffn_chain(
        &self,
        normed: &[f32],
        gate_w: &[u8],
        up_w: &[u8],
        down_w: &[u8],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (gate_wb, gate_off) = wrap(gate_w);
        let (up_wb, up_off) = wrap(up_w);
        let (down_wb, down_off) = wrap(down_w);
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);
        let up_off_buf = ffn_chain::u32_buf(ctx, up_off);
        let down_off_buf = ffn_chain::u32_buf(ctx, down_off);

        let mut carriers = self.prefill_ffn_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, ffn_dim, seq_len))
            .or_insert_with(|| {
                ffn_chain::PrefillFfnCarrier::new(ctx, hidden_dim, ffn_dim, seq_len)
            });
        ffn_chain::prefill_ffn_chain_dispatch(
            ctx,
            carrier,
            normed,
            &gate_wb,
            &gate_off_buf,
            &up_wb,
            &up_off_buf,
            &down_wb,
            &down_off_buf,
            down_is_q6k,
            seq_len,
        )
    }

    pub fn qwen_moe_prefill_sparse_accum_supported(&self, down_is_q6k: bool) -> bool {
        let Some(ctx) = self.ctx.as_ref() else {
            return false;
        };
        ffn_chain::prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k)
    }

    pub fn qwen_moe_prefill_mulmmid_v3_supported(&self, down_is_q6k: bool) -> bool {
        let Some(ctx) = self.ctx.as_ref() else {
            return false;
        };
        ffn_chain::prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k)
            && ffn_chain::qwen_moe_prefill_mulmmid_v3_supported(ctx, down_is_q6k)
    }

    pub fn qwen_moe_prefill_mulmmid_v4_supported(&self, down_is_q6k: bool) -> bool {
        let Some(ctx) = self.ctx.as_ref() else {
            return false;
        };
        ffn_chain::qwen_moe_prefill_mulmmid_v4_supported(ctx, down_is_q6k)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen_moe_prefill_sparse_accum_flush_batch(
        &self,
        ctx: &compute::MetalContext,
        accum: &ffn_chain::QwenMoePrefillAccumCarrier,
        pending: &mut Vec<QwenMoePrefillPendingGroup>,
        down_is_q6k: bool,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
        token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
        route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    ) -> usize {
        if pending.is_empty() {
            return 0;
        }
        let timing = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TIME")
            .ok()
            .is_some_and(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            });
        let wall0 = timing.then(std::time::Instant::now);
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        for group in pending.iter() {
            if let Some(input) = group.input.as_ref() {
                let mut ffn_carriers = self.prefill_ffn_carriers.borrow_mut();
                let carrier = ffn_carriers
                    .entry((hidden_dim, ffn_dim, group.group_len))
                    .or_insert_with(|| {
                        ffn_chain::PrefillFfnCarrier::new(ctx, hidden_dim, ffn_dim, group.group_len)
                    });
                ffn_chain::prefill_ffn_chain_v2_scatter_accum_encode(
                    ctx,
                    &enc,
                    carrier,
                    accum,
                    input,
                    &group.gate_wb,
                    group.gate_off,
                    &group.up_wb,
                    group.up_off,
                    &group.down_wb,
                    group.down_off,
                    down_is_q6k,
                    group.group_len,
                    token_ids_buf,
                    route_weights_buf,
                    group.group_start,
                    group.total_elems,
                );
            } else {
                let norm_all_buf =
                    norm_all_buf.expect("Qwen MoE prefill gather requires norm_all buffer");
                let mut ffn_carriers = self.qwen_moe_prefill_private_ffn_carriers.borrow_mut();
                let carrier = ffn_carriers
                    .entry((hidden_dim, ffn_dim, group.group_len))
                    .or_insert_with(|| {
                        ffn_chain::PrefillFfnCarrier::new_private_intermediates(
                            ctx,
                            hidden_dim,
                            ffn_dim,
                            group.group_len,
                        )
                    });
                ffn_chain::prefill_ffn_chain_v2_scatter_accum_encode_gather(
                    ctx,
                    &enc,
                    carrier,
                    accum,
                    norm_all_buf,
                    &group.gate_wb,
                    group.gate_off,
                    &group.up_wb,
                    group.up_off,
                    &group.down_wb,
                    group.down_off,
                    down_is_q6k,
                    group.group_len,
                    token_ids_buf,
                    route_weights_buf,
                    group.group_start,
                    group.total_elems,
                );
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let groups = pending.len();
        if timing {
            let wall_ms = wall0
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
            eprintln!(
                "[metal-qwen-moe-prefill-accum-time] groups={groups} gpu_ms={gpu_ms:.3} wall_ms={wall_ms:.3}"
            );
        }
        pending.clear();
        groups
    }

    /// Qwen3.6 MoE sparse prefill experiment. Expert groups are still submitted
    /// independently, but each group accumulates on Metal into token-major output
    /// so the host reads the sparse layer output once instead of once per expert.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen_moe_prefill_sparse_accum(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all: &[f32],
    ) -> Vec<f32> {
        self.qwen_moe_prefill_sparse_accum_with_shared(
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
            None,
        )
    }

    /// Shared expert + sparse expert prefill accumulator. This keeps quantized
    /// weights raw and only adds the shared expert's activation output into the
    /// same token-major accumulator used by sparse experts.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen_moe_prefill_shared_sparse_accum(
        &self,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route_weights: &[f32],
        shared_token_ids: &[u32],
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all: &[f32],
    ) -> Vec<f32> {
        self.qwen_moe_prefill_sparse_accum_with_shared(
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
            Some((
                shared_gate,
                shared_up,
                shared_down,
                shared_route_weights,
                shared_token_ids,
            )),
        )
    }

    pub fn qwen_moe_llama_id_prefill(
        &self,
        request: QwenMoeLlamaIdPrefillRequest<'_>,
    ) -> Result<Option<QwenMoeLlamaIdPrefillOutput>, String> {
        self.qwen_moe_llama_id_prefill_requested(
            request,
            crate::compute::qwen_moe_llama_id_requested(),
        )
    }

    fn qwen_moe_llama_id_prefill_requested(
        &self,
        request: QwenMoeLlamaIdPrefillRequest<'_>,
        requested: bool,
    ) -> Result<Option<QwenMoeLlamaIdPrefillOutput>, String> {
        let trace_enabled = qwen_moe_llama_id_trace_enabled();
        let wrapper_start = std::time::Instant::now();
        let trace_fallback = |reason: &str| {
            if trace_enabled {
                eprintln!(
                    "{}",
                    format_qwen_moe_llama_id_fallback_trace(
                        reason,
                        wrapper_start.elapsed().as_secs_f64() * 1000.0,
                    )
                );
            }
        };
        let trace_error = |stage: &str, error: String| {
            if trace_enabled {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=llama_id fallback_reason=error error_stage={stage} error={error} gpu_ms=unmeasured wrapper_wall_ms={} readback_ms=unmeasured",
                    qwen_moe_llama_id_ms(Some(
                        wrapper_start.elapsed().as_secs_f64() * 1000.0
                    )),
                );
            }
            error
        };

        if !requested {
            trace_fallback("request_off");
            return Ok(None);
        }
        let plan =
            match qwen_moe_llama_id_prepare(&request, true, QWEN_MOE_LLAMA_ID_SCRATCH_BUDGET_BYTES)
            {
                Ok(ffn_chain::QwenMoeLlamaIdPreflight::Run(plan)) => plan,
                Ok(ffn_chain::QwenMoeLlamaIdPreflight::Fallback(reason)) => {
                    trace_fallback(qwen_moe_llama_id_fallback_reason(reason));
                    return Ok(None);
                }
                Err(error) => {
                    return Err(trace_error("prepare", qwen_moe_llama_id_error(error)));
                }
            };
        let Some(ctx) = self.ctx.as_ref() else {
            trace_fallback("context_unavailable");
            return Ok(None);
        };
        if !qwen_moe_llama_id_pipelines_available(ctx, request.sparse_quant, request.shared_quant) {
            trace_fallback("pipeline_unavailable");
            return Ok(None);
        }

        let cached_carrier = self
            .qwen_moe_llama_id_carrier_pool
            .borrow_mut()
            .get_mut(&plan)
            .and_then(Vec::pop);
        let (carrier, refresh_routes) = match cached_carrier {
            Some(carrier) => (carrier, true),
            None => (
                ffn_chain::QwenMoeLlamaIdCarrier::new(
                    ctx,
                    plan,
                    request.selected_experts,
                    request.route_weights,
                    request.shared_route_weights,
                )
                .map_err(|error| trace_error("carrier", qwen_moe_llama_id_error(error)))?,
                false,
            ),
        };
        let mut carrier = QwenMoeLlamaIdCarrierLease {
            pool: &self.qwen_moe_llama_id_carrier_pool,
            key: plan,
            carrier: Some(carrier),
        };
        if refresh_routes {
            carrier
                .refresh_routes(
                    request.selected_experts,
                    request.route_weights,
                    request.shared_route_weights,
                )
                .map_err(|error| trace_error("carrier_routes", qwen_moe_llama_id_error(error)))?;
        }
        let norm_all = qwen_moe_llama_id_upload_f32(ctx, request.norm_all)
            .map_err(|error| trace_error("upload_norm", qwen_moe_llama_id_error(error)))?;
        let resident_weight = |raw: &[u8]| {
            let mut resident = self.resident.borrow_mut();
            let entry = resident
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1 as usize)
        };
        let (gate_all, gate_all_offset) = resident_weight(request.gate_all);
        let (up_all, up_all_offset) = resident_weight(request.up_all);
        let (down_all, down_all_offset) = resident_weight(request.down_all);
        let (shared_gate, shared_gate_offset) = resident_weight(request.shared_gate);
        let (shared_up, shared_up_offset) = resident_weight(request.shared_up);
        let (shared_down, shared_down_offset) = resident_weight(request.shared_down);

        let (mut stage_sampler, stage_timing_unsupported_reason) = if trace_enabled {
            match ffn_chain::QwenMoeLlamaIdStageSampler::new(ctx) {
                Ok(sampler) => (Some(sampler), None),
                Err(reason) => (None, Some(reason)),
            }
        } else {
            (None, None)
        };
        if let Some(counter_fallback_reason) = stage_timing_unsupported_reason {
            let (trace, gpu_ms) = ffn_chain::qwen_moe_llama_prefill_trace_split(
                ctx,
                &carrier,
                &norm_all,
                &gate_all,
                gate_all_offset,
                &up_all,
                up_all_offset,
                &down_all,
                down_all_offset,
                request.sparse_quant,
                &shared_gate,
                shared_gate_offset,
                &shared_up,
                shared_up_offset,
                &shared_down,
                shared_down_offset,
                request.shared_quant,
                true,
                counter_fallback_reason,
            )
            .map_err(|error| trace_error("trace_split", qwen_moe_llama_id_error(error)))?;
            let readback_start = std::time::Instant::now();
            let values =
                ffn_chain::readback(carrier.output(), request.seq_len * request.hidden_dim);
            let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "{}",
                format_qwen_moe_llama_id_success_trace(
                    trace,
                    request.layer_idx,
                    request.sparse_quant,
                    request.shared_quant,
                    Some(gpu_ms),
                    wrapper_start.elapsed().as_secs_f64() * 1000.0,
                    readback_ms,
                )
            );
            return Ok(Some(QwenMoeLlamaIdPrefillOutput { values, trace }));
        }

        let counter_start = stage_sampler
            .as_ref()
            .map(|_| ffn_chain::QwenMoeLlamaIdStageSampler::calibration(ctx));

        let command = ctx.queue.commandBuffer().ok_or_else(|| {
            trace_error(
                "command_buffer",
                qwen_moe_llama_id_error(ffn_chain::QwenMoeLlamaIdError::CommandBufferFailed),
            )
        })?;
        let encoder = crate::compute::chain_compute_encoder(ctx, &command);
        let trace = match ffn_chain::qwen_moe_llama_prefill_encode(
            ctx,
            &encoder,
            &carrier,
            &norm_all,
            &gate_all,
            gate_all_offset,
            &up_all,
            up_all_offset,
            &down_all,
            down_all_offset,
            request.sparse_quant,
            &shared_gate,
            shared_gate_offset,
            &shared_up,
            shared_up_offset,
            &shared_down,
            shared_down_offset,
            request.shared_quant,
            true,
            stage_sampler.as_mut(),
        ) {
            Ok(trace) => trace,
            Err(error) => {
                encoder.endEncoding();
                return Err(trace_error("encode", qwen_moe_llama_id_error(error)));
            }
        };
        let mut trace = trace;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        let counter_end = stage_sampler
            .as_ref()
            .map(|_| ffn_chain::QwenMoeLlamaIdStageSampler::calibration(ctx));
        if let Err(error) = qwen_moe_llama_id_command_result(
            command.status(),
            command.error().map(|error| format!("{error:?}")),
        ) {
            let gpu_start = command.GPUStartTime();
            let gpu_end = command.GPUEndTime();
            let gpu_ms = if gpu_start.is_finite() && gpu_end.is_finite() && gpu_end > gpu_start {
                Some((gpu_end - gpu_start) * 1000.0)
            } else {
                None
            };
            if trace_enabled {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=llama_id fallback_reason=error error_stage=post_submit error={error} gpu_ms={} wrapper_wall_ms={} readback_ms=unmeasured",
                    qwen_moe_llama_id_ms(gpu_ms),
                    qwen_moe_llama_id_ms(Some(
                        wrapper_start.elapsed().as_secs_f64() * 1000.0
                    )),
                );
            }
            return Err(error);
        }

        if let (Some(sampler), Some(start), Some(end)) =
            (stage_sampler.as_ref(), counter_start, counter_end)
        {
            if let Err(reason) = sampler.resolve(start, end, &mut trace) {
                trace.stage_timing_unsupported_reason = Some(reason);
            }
        } else if trace_enabled {
            trace.stage_timing_unsupported_reason =
                stage_timing_unsupported_reason.or(Some("counter_sampler_unavailable"));
        }

        let gpu_start = command.GPUStartTime();
        let gpu_end = command.GPUEndTime();
        let gpu_ms = if gpu_start.is_finite() && gpu_end.is_finite() && gpu_end > gpu_start {
            Some((gpu_end - gpu_start) * 1000.0)
        } else {
            None
        };
        let readback_start = std::time::Instant::now();
        let values = ffn_chain::readback(carrier.output(), request.seq_len * request.hidden_dim);
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        if trace_enabled {
            eprintln!(
                "{}",
                format_qwen_moe_llama_id_success_trace(
                    trace,
                    request.layer_idx,
                    request.sparse_quant,
                    request.shared_quant,
                    gpu_ms,
                    wrapper_start.elapsed().as_secs_f64() * 1000.0,
                    readback_ms,
                )
            );
        }
        Ok(Some(QwenMoeLlamaIdPrefillOutput { values, trace }))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_qwen_moe_prefill_id_primitive(
        &self,
        ctx: &compute::MetalContext,
        accum: &ffn_chain::QwenMoePrefillAccumCarrier,
        gate_all_wb: &ProtocolObject<dyn MTLBuffer>,
        gate_all_off: u32,
        up_all_wb: &ProtocolObject<dyn MTLBuffer>,
        up_all_off: u32,
        down_all_wb: &ProtocolObject<dyn MTLBuffer>,
        down_all_off: u32,
        gate_all_len: usize,
        up_all_len: usize,
        down_all_len: usize,
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        timing: bool,
        trace: bool,
    ) -> Result<Vec<f32>, ffn_chain::QwenMoeIdPreflightReason> {
        const TILE_SLOTS: usize = ffn_chain::QWEN_MOE_ID_PRIMITIVE_TILE_SLOTS;
        const FFN_TILE: usize = ffn_chain::QWEN_MOE_ID_PRIMITIVE_FFN_TILE;
        const HIDDEN_TILE: usize = ffn_chain::QWEN_MOE_ID_PRIMITIVE_HIDDEN_TILE;
        const PARTIAL_BUDGET_BYTES: usize = 256 * 1024 * 1024;

        if hidden_dim == 0 || ffn_dim == 0 || hidden_dim % 256 != 0 || ffn_dim % FFN_TILE != 0 {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidPrimitiveShape);
        }

        let ffn_tiles = ffn_dim / FFN_TILE;
        let hidden_tile = HIDDEN_TILE.min(hidden_dim);
        let pipelines_available = ctx.qwen_moe_id_q4_gate_up_tile_pipeline.is_some()
            && ctx.qwen_moe_id_silu_mul_tile_pipeline.is_some()
            && ctx.qwen_moe_id_q6_down_partial_tile_pipeline.is_some()
            && ctx.qwen_moe_id_partial_reduce_scatter_pipeline.is_some();
        let preflight_shape = ffn_chain::QwenMoeIdPrimitiveShape {
            tile_slots: TILE_SLOTS,
            ffn_tile: FFN_TILE,
            hidden_tile,
            ffn_tiles_per_pass: ffn_tiles,
        };
        let plan = ffn_chain::preflight_qwen_moe_id_primitive_inputs(
            pipelines_available,
            down_is_q6k,
            gate_all_len,
            up_all_len,
            down_all_len,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            token_ids,
            route_weights,
            seq_len,
            hidden_dim,
            ffn_dim,
            TILE_SLOTS,
            preflight_shape,
            PARTIAL_BUDGET_BYTES,
        )?;
        let n_expert = plan.expert_counts.len();
        let max_count = plan
            .expert_counts
            .iter()
            .map(|&count| count as usize)
            .max()
            .unwrap_or(0);
        if max_count == 0 {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena);
        }
        let scratch_shape = ffn_chain::QwenMoeIdPrimitiveShape {
            tile_slots: TILE_SLOTS,
            ffn_tile: FFN_TILE,
            hidden_tile,
            ffn_tiles_per_pass: ffn_tiles,
        };
        if scratch_shape.partial_bytes() > PARTIAL_BUDGET_BYTES {
            return Err(ffn_chain::QwenMoeIdPreflightReason::PartialBufferOverBudget);
        }
        let predicted_dispatch_count = ffn_chain::qwen_moe_id_primitive_predicted_dispatch_count(
            plan.block_experts.len(),
            hidden_dim,
            scratch_shape,
        )?;
        let dispatch_budget = ffn_chain::qwen_moe_id_primitive_dispatch_budget(
            plan.block_experts.len(),
            hidden_dim,
            scratch_shape,
        )?;
        if predicted_dispatch_count > dispatch_budget {
            if trace || timing {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=id_primitive fallback_reason=dispatch_count_over_budget predicted_dispatch_count={predicted_dispatch_count} dispatch_budget={dispatch_budget}"
                );
            }
            return Err(ffn_chain::QwenMoeIdPreflightReason::DispatchCountOverBudget);
        }

        let checked_weight_offset = |base: u32,
                                     expert: usize,
                                     stride: usize|
         -> Result<u32, ffn_chain::QwenMoeIdPreflightReason> {
            let expert_offset = expert
                .checked_mul(stride)
                .ok_or(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena)?;
            let byte_offset = (base as usize)
                .checked_add(expert_offset)
                .ok_or(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena)?;
            u32::try_from(byte_offset)
                .map_err(|_| ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena)
        };

        let gate_tile = ffn_chain::private_f32_buf(ctx, TILE_SLOTS * FFN_TILE);
        let up_tile = ffn_chain::private_f32_buf(ctx, TILE_SLOTS * FFN_TILE);
        let act_tile = ffn_chain::private_f32_buf(ctx, TILE_SLOTS * FFN_TILE);
        let partial = ffn_chain::private_f32_buf(ctx, ffn_tiles * TILE_SLOTS * hidden_tile);
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        let mut group_keepalive: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        let mut group_count = 0usize;
        let mut block_count_total = 0usize;
        let mut dispatch_count = 0usize;
        let mut q4_gate_up_wall_ms = 0.0f64;
        let mut silu_wall_ms = 0.0f64;
        let mut q6_down_partial_wall_ms = 0.0f64;
        let mut reduce_scatter_wall_ms = 0.0f64;
        let collect_trace_timing = timing || trace;
        let encode0 = collect_trace_timing.then(std::time::Instant::now);

        for expert in 0..n_expert {
            let count = plan.expert_counts[expert] as usize;
            if count == 0 {
                continue;
            }
            group_count += 1;
            let group_start = plan.expert_offsets[expert] as usize;
            let down_weight_off = checked_weight_offset(down_all_off, expert, down_expert_bytes)?;

            for local0 in (0..count).step_by(TILE_SLOTS) {
                let block_slots = (count - local0).min(TILE_SLOTS);
                block_count_total += 1;
                let slot0 = group_start
                    .checked_add(local0)
                    .ok_or(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena)?;
                let slot1 = slot0
                    .checked_add(block_slots)
                    .ok_or(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena)?;
                let block_token_ids_buf =
                    ffn_chain::shared_u32_buf(ctx, &plan.token_ids[slot0..slot1]);
                let block_route_weights_buf =
                    ffn_chain::shared_f32_buf(ctx, &plan.route_weights[slot0..slot1]);
                let mut local_expert_offsets = vec![0u32; n_expert];
                let mut local_expert_counts = vec![0u32; n_expert];
                local_expert_offsets[expert] = 0;
                local_expert_counts[expert] = u32::try_from(block_slots)
                    .map_err(|_| ffn_chain::QwenMoeIdPreflightReason::BlockCountOverflow)?;
                let block_experts = [expert as u32];
                let block_local0 = [0u32];
                let local_expert_offsets_buf =
                    ffn_chain::shared_u32_buf(ctx, &local_expert_offsets);
                let local_expert_counts_buf = ffn_chain::shared_u32_buf(ctx, &local_expert_counts);
                let block_experts_buf = ffn_chain::shared_u32_buf(ctx, &block_experts);
                let block_local0_buf = ffn_chain::shared_u32_buf(ctx, &block_local0);
                group_keepalive.push(block_token_ids_buf.clone());
                group_keepalive.push(block_route_weights_buf.clone());
                group_keepalive.push(local_expert_offsets_buf.clone());
                group_keepalive.push(local_expert_counts_buf.clone());
                group_keepalive.push(block_experts_buf.clone());
                group_keepalive.push(block_local0_buf.clone());

                for hidden0 in (0..hidden_dim).step_by(hidden_tile) {
                    let active_hidden_tile = (hidden_dim - hidden0).min(hidden_tile);
                    for ffn_tile_idx in 0..ffn_tiles {
                        let ffn_tile0 = ffn_tile_idx * FFN_TILE;
                        let q4_stage0 = collect_trace_timing.then(std::time::Instant::now);
                        compute::encode_qwen_moe_id_q4_gate_up_tile(
                            ctx,
                            &enc,
                            gate_all_wb,
                            gate_all_off,
                            up_all_wb,
                            up_all_off,
                            &accum.norm_all_dev,
                            &gate_tile,
                            &up_tile,
                            &block_token_ids_buf,
                            &local_expert_offsets_buf,
                            &local_expert_counts_buf,
                            &block_experts_buf,
                            &block_local0_buf,
                            ffn_dim,
                            hidden_dim,
                            1,
                            gate_expert_bytes,
                            up_expert_bytes,
                            ffn_tile0,
                            FFN_TILE,
                        )?;
                        dispatch_count += 1;
                        if let Some(start) = q4_stage0 {
                            q4_gate_up_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
                        }

                        let silu_stage0 = collect_trace_timing.then(std::time::Instant::now);
                        compute::encode_qwen_moe_id_silu_mul_tile(
                            ctx,
                            &enc,
                            &gate_tile,
                            &up_tile,
                            &act_tile,
                            block_slots * FFN_TILE,
                        )?;
                        dispatch_count += 1;
                        if let Some(start) = silu_stage0 {
                            silu_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
                        }

                        let q6_stage0 = collect_trace_timing.then(std::time::Instant::now);
                        compute::encode_qwen_moe_id_q6_down_partial_tile(
                            ctx,
                            &enc,
                            down_all_wb,
                            down_weight_off,
                            &act_tile,
                            &partial,
                            hidden_dim,
                            hidden0,
                            active_hidden_tile,
                            ffn_dim,
                            ffn_tile0,
                            FFN_TILE,
                            block_slots,
                            ffn_tile_idx,
                        )?;
                        dispatch_count += 1;
                        if let Some(start) = q6_stage0 {
                            q6_down_partial_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
                        }
                    }

                    let reduce_stage0 = collect_trace_timing.then(std::time::Instant::now);
                    compute::encode_qwen_moe_id_partial_reduce_scatter(
                        ctx,
                        &enc,
                        &partial,
                        &accum.out_dev,
                        &block_token_ids_buf,
                        &block_route_weights_buf,
                        ffn_tiles,
                        block_slots,
                        active_hidden_tile,
                        hidden_dim,
                        hidden0,
                        0,
                    )?;
                    dispatch_count += 1;
                    if let Some(start) = reduce_stage0 {
                        reduce_scatter_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
        let encode_wait_ms = encode0
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if trace {
            let scratch_mib = scratch_shape.scratch_bytes() as f64 / (1024.0 * 1024.0);
            let partial_mib = scratch_shape.partial_bytes() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=id_primitive fallback_reason=none group_count={group_count} block_count={block_count_total} dispatch_count={dispatch_count} slots={} experts={n_expert} max_count={max_count} q4_gate_up_wall_ms={q4_gate_up_wall_ms:.3} silu_wall_ms={silu_wall_ms:.3} q6_down_partial_wall_ms={q6_down_partial_wall_ms:.3} reduce_scatter_wall_ms={reduce_scatter_wall_ms:.3} gpu_ms={gpu_ms:.3} encode_wait_ms={encode_wait_ms:.3} scratch_mib={scratch_mib:.1} partial_mib={partial_mib:.1}",
                expert_ids.len()
            );
        }
        if timing {
            eprintln!(
                "[metal-qwen-moe-prefill-accum-time] mode=id_primitive fallback_reason=none dispatch_count={dispatch_count} q4_gate_up_wall_ms={q4_gate_up_wall_ms:.3} silu_wall_ms={silu_wall_ms:.3} q6_down_partial_wall_ms={q6_down_partial_wall_ms:.3} reduce_scatter_wall_ms={reduce_scatter_wall_ms:.3} gpu_ms={gpu_ms:.3} encode_wait_ms={encode_wait_ms:.3}"
            );
        }
        let read0 = timing.then(std::time::Instant::now);
        let out = accum.readback();
        if timing {
            let read_ms = read0
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!("[metal-qwen-moe-prefill-accum-time] readback_ms={read_ms:.3}");
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_qwen_moe_prefill_mulmmid_v3(
        &self,
        ctx: &compute::MetalContext,
        accum: &ffn_chain::QwenMoePrefillAccumCarrier,
        gate_all_wb: &ProtocolObject<dyn MTLBuffer>,
        gate_all_off: u32,
        up_all_wb: &ProtocolObject<dyn MTLBuffer>,
        up_all_off: u32,
        down_all_wb: &ProtocolObject<dyn MTLBuffer>,
        down_all_off: u32,
        gate_all_len: usize,
        up_all_len: usize,
        down_all_len: usize,
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all: &[f32],
        shared_preserved: bool,
        trace: bool,
    ) -> Result<Vec<f32>, ffn_chain::QwenMoeIdPreflightReason> {
        if !ffn_chain::qwen_moe_prefill_mulmmid_v3_supported(ctx, down_is_q6k) {
            return Err(ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable);
        }
        if !ffn_chain::qwen_moe_mulmmid_v3_dims_supported(hidden_dim, ffn_dim) {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidPrimitiveShape);
        }
        if seq_len == 0 || expert_ids.len() % seq_len != 0 {
            return Err(ffn_chain::QwenMoeIdPreflightReason::RankCountMismatch);
        }
        if gate_expert_bytes == 0
            || up_expert_bytes == 0
            || down_expert_bytes == 0
            || gate_all_len % gate_expert_bytes != 0
            || up_all_len % up_expert_bytes != 0
            || down_all_len % down_expert_bytes != 0
        {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena);
        }
        let n_gate = gate_all_len / gate_expert_bytes;
        let n_up = up_all_len / up_expert_bytes;
        let n_down = down_all_len / down_expert_bytes;
        if n_gate != n_up || n_gate != n_down {
            return Err(ffn_chain::QwenMoeIdPreflightReason::ExpertArenaMismatch);
        }
        let n_expert_used = expert_ids.len() / seq_len;
        let plan = ffn_chain::build_qwen_moe_mul_mm_id_v3_plan_token_major(
            expert_ids,
            token_ids,
            route_weights,
            n_gate,
            seq_len,
            n_expert_used,
            64,
        )?;
        let block_count = plan.block_experts.len();
        let budget_bytes = qwen_moe_prefill_mulmmid_v3_scratch_budget_bytes();
        let mut carriers = self.qwen_moe_prefill_mulmmid_v3_carriers.borrow_mut();
        let carrier = match carriers.entry((
            seq_len,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            n_gate,
            block_count,
            budget_bytes,
        )) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(ffn_chain::QwenMoeMulMmIdV3Carrier::new(
                    ctx,
                    seq_len,
                    n_expert_used,
                    hidden_dim,
                    ffn_dim,
                    n_gate,
                    block_count,
                    budget_bytes,
                )?)
            }
        };

        accum.upload_norm_all(norm_all);
        let token_ids_sorted_buf = ffn_chain::shared_u32_buf(ctx, &plan.token_ids_sorted);
        let dst_slots_sorted_buf = ffn_chain::shared_u32_buf(ctx, &plan.dst_slots_sorted);
        let route_weights_token_rank_buf =
            ffn_chain::shared_f32_buf(ctx, &plan.route_weights_token_rank);
        let expert_offsets_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_offsets);
        let expert_counts_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_counts);
        let block_experts_buf = ffn_chain::shared_u32_buf(ctx, &plan.block_experts);
        let block_local0_buf = ffn_chain::shared_u32_buf(ctx, &plan.block_local0);

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        ffn_chain::qwen_moe_prefill_mulmmid_v3_encode(
            ctx,
            &enc,
            carrier,
            accum,
            &accum.norm_all_dev,
            gate_all_wb,
            gate_all_off,
            up_all_wb,
            up_all_off,
            down_all_wb,
            down_all_off,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            &token_ids_sorted_buf,
            &dst_slots_sorted_buf,
            &route_weights_token_rank_buf,
            &expert_offsets_buf,
            &expert_counts_buf,
            &block_experts_buf,
            &block_local0_buf,
            block_count,
            down_is_q6k,
        )?;
        enc.endEncoding();
        let wall_start = std::time::Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall_ms = wall_start.elapsed().as_micros() as f64 / 1000.0;
        let gpu_ms = ((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0).max(0.0);

        if trace {
            let scratch = carrier.scratch();
            let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason=none dispatch_count=5 shared_preserved={shared_preserved} slots={} experts={} blocks={} gpu_ms={:.3} wall_ms={:.3} scratch_mib={:.1} gate_mib={:.1} up_mib={:.1} down_mib={:.1} metadata_mib={:.1}",
                expert_ids.len(),
                n_gate,
                block_count,
                gpu_ms,
                wall_ms,
                mib(scratch.total_bytes),
                mib(scratch.gate_bytes),
                mib(scratch.up_bytes),
                mib(scratch.down_bytes),
                mib(scratch.metadata_bytes),
            );
        }
        Ok(accum.readback())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_qwen_moe_prefill_mulmmid_v4(
        &self,
        ctx: &compute::MetalContext,
        accum: &ffn_chain::QwenMoePrefillAccumCarrier,
        gate_all_wb: &ProtocolObject<dyn MTLBuffer>,
        gate_all_off: u32,
        up_all_wb: &ProtocolObject<dyn MTLBuffer>,
        up_all_off: u32,
        down_all_wb: &ProtocolObject<dyn MTLBuffer>,
        down_all_off: u32,
        gate_all_len: usize,
        up_all_len: usize,
        down_all_len: usize,
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        shared_preserved: bool,
        trace: bool,
    ) -> Result<Vec<f32>, ffn_chain::QwenMoeIdPreflightReason> {
        if !ffn_chain::qwen_moe_prefill_mulmmid_v4_supported(ctx, down_is_q6k) {
            return Err(ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable);
        }
        if !ffn_chain::qwen_moe_mulmmid_v4_dims_supported(hidden_dim, ffn_dim) {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidPrimitiveShape);
        }
        if seq_len == 0 || expert_ids.len() % seq_len != 0 {
            return Err(ffn_chain::QwenMoeIdPreflightReason::RankCountMismatch);
        }
        if gate_expert_bytes == 0
            || up_expert_bytes == 0
            || down_expert_bytes == 0
            || gate_all_len % gate_expert_bytes != 0
            || up_all_len % up_expert_bytes != 0
            || down_all_len % down_expert_bytes != 0
        {
            return Err(ffn_chain::QwenMoeIdPreflightReason::InvalidExpertArena);
        }
        let n_gate = gate_all_len / gate_expert_bytes;
        let n_up = up_all_len / up_expert_bytes;
        let n_down = down_all_len / down_expert_bytes;
        if n_gate != n_up || n_gate != n_down {
            return Err(ffn_chain::QwenMoeIdPreflightReason::ExpertArenaMismatch);
        }
        let n_expert_used = expert_ids.len() / seq_len;
        let plan = ffn_chain::build_qwen_moe_mul_mm_id_v4_plan_token_major(
            expert_ids,
            token_ids,
            route_weights,
            n_gate,
            seq_len,
            n_expert_used,
            64,
        )?;
        let id_block_count = plan.id_block_experts.len();
        let rank_block_count = plan.rank_block_experts.len();
        let carrier_block_budget = id_block_count.max(rank_block_count);
        let budget_bytes = qwen_moe_prefill_mulmmid_v3_scratch_budget_bytes();
        let mut carriers = self.qwen_moe_prefill_mulmmid_v4_carriers.borrow_mut();
        let carrier = match carriers.entry((
            seq_len,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            n_gate,
            id_block_count,
            rank_block_count,
            budget_bytes,
        )) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(ffn_chain::QwenMoeMulMmIdV4Carrier::new(
                    ctx,
                    seq_len,
                    n_expert_used,
                    hidden_dim,
                    ffn_dim,
                    n_gate,
                    carrier_block_budget,
                    budget_bytes,
                )?)
            }
        };

        let token_ids_sorted_buf = ffn_chain::shared_u32_buf(ctx, &plan.token_ids_sorted);
        let route_weights_sorted_buf = ffn_chain::shared_f32_buf(ctx, &plan.route_weights_sorted);
        let expert_offsets_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_offsets);
        let expert_counts_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_counts);
        let id_block_experts_buf = ffn_chain::shared_u32_buf(ctx, &plan.id_block_experts);
        let id_block_local0_buf = ffn_chain::shared_u32_buf(ctx, &plan.id_block_local0);
        let expert_rank_offsets_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_rank_offsets);
        let expert_rank_counts_buf = ffn_chain::shared_u32_buf(ctx, &plan.expert_rank_counts);
        let rank_block_experts_buf = ffn_chain::shared_u32_buf(ctx, &plan.rank_block_experts);
        let rank_block_local0_buf = ffn_chain::shared_u32_buf(ctx, &plan.rank_block_local0);

        let encode0 = std::time::Instant::now();
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        ffn_chain::qwen_moe_prefill_mulmmid_v4_encode(
            ctx,
            &enc,
            carrier,
            accum,
            &accum.norm_all_dev,
            gate_all_wb,
            gate_all_off,
            up_all_wb,
            up_all_off,
            down_all_wb,
            down_all_off,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            &token_ids_sorted_buf,
            &route_weights_sorted_buf,
            &expert_offsets_buf,
            &expert_counts_buf,
            &id_block_experts_buf,
            &id_block_local0_buf,
            id_block_count,
            &expert_rank_offsets_buf,
            &expert_rank_counts_buf,
            &rank_block_experts_buf,
            &rank_block_local0_buf,
            &plan.rank_block_offsets,
            &plan.rank_block_counts,
            down_is_q6k,
        )?;
        enc.endEncoding();
        let wall_start = std::time::Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall_ms = wall_start.elapsed().as_micros() as f64 / 1000.0;
        let encode_wait_ms = encode0.elapsed().as_micros() as f64 / 1000.0;
        let gpu_ms = ((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0).max(0.0);

        if trace {
            eprintln!(
                "{}",
                format_qwen_moe_mulmmid_v4_trace(
                    shared_preserved,
                    expert_ids.len(),
                    seq_len,
                    hidden_dim,
                    n_gate,
                    n_expert_used,
                    id_block_count,
                    rank_block_count,
                    gpu_ms,
                    wall_ms,
                    encode_wait_ms,
                    carrier.scratch(),
                )
            );
        }
        Ok(accum.readback())
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn qwen_moe_prefill_sparse_accum_with_shared(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        down_is_q6k: bool,
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all: &[f32],
        shared: Option<(&[u8], &[u8], &[u8], &[f32], &[u32])>,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(expert_ids.len(), route_weights.len(), "expert/route slots");
        assert_eq!(expert_ids.len(), token_ids.len(), "expert/token slots");
        assert_eq!(norm_all.len(), seq_len * hidden_dim, "norm_all shape");
        if let Some((_, _, _, shared_route_weights, shared_token_ids)) = shared {
            assert_eq!(
                shared_route_weights.len(),
                seq_len,
                "Qwen MoE shared route weights"
            );
            assert_eq!(shared_token_ids.len(), seq_len, "Qwen MoE shared token ids");
        }
        assert!(
            ffn_chain::prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k),
            "Qwen MoE prefill accum requires tensorops v2 FFN path"
        );

        let token_ids_buf = ffn_chain::shared_u32_buf(ctx, token_ids);
        let route_weights_buf = ffn_chain::shared_f32_buf(ctx, route_weights);
        let mut accum_carriers = self.qwen_moe_prefill_accum_carriers.borrow_mut();
        let accum = accum_carriers
            .entry((hidden_dim, ffn_dim, seq_len))
            .or_insert_with(|| {
                ffn_chain::QwenMoePrefillAccumCarrier::new(ctx, hidden_dim, seq_len)
            });
        accum.clear();

        let gather_enabled = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_GATHER")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true);
        let id_matmul_requested = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_MATMUL")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
        let id_gate_up_requested = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_GATE_UP")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
        let id_gate_up_f16_requested = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_F16")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
        let id_primitive_requested = crate::compute::qwen_moe_prefill_id_primitive_requested();
        let mulmmid_v3_requested = crate::compute::qwen_moe_prefill_mulmmid_v3_requested();
        let mulmmid_v4_requested = crate::compute::qwen_moe_prefill_mulmmid_v4_requested();
        let timing = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TIME")
            .ok()
            .is_some_and(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            });
        let combine_shared_with_sparse = qwen_moe_prefill_shared_combined_requested()
            && shared.is_some()
            && !id_matmul_requested
            && !id_gate_up_requested
            && !id_primitive_requested
            && !mulmmid_v3_requested
            && !mulmmid_v4_requested;
        let upload0 = timing.then(std::time::Instant::now);
        if qwen_moe_prefill_should_upload_norm(
            gather_enabled,
            id_matmul_requested,
            id_gate_up_requested,
            id_primitive_requested,
            mulmmid_v3_requested,
            mulmmid_v4_requested,
        ) {
            accum.upload_norm_all(norm_all);
        }
        if timing {
            let upload_ms = upload0
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!("[metal-qwen-moe-prefill-accum-time] upload_norm_ms={upload_ms:.3}");
        }
        let norm_all_buf: Option<&ProtocolObject<dyn MTLBuffer>> = if gather_enabled {
            Some(&*accum.norm_all_dev)
        } else {
            None
        };

        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (gate_all_wb, gate_all_off) = wrap(gate_all);
        let (up_all_wb, up_all_off) = wrap(up_all);
        let (down_all_wb, down_all_off) = wrap(down_all);

        let trace = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TRACE")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);

        let mut shared_pending: Option<QwenMoePrefillPendingGroup> = None;
        let mut shared_preserved_for_mulmmid = shared.is_none();
        let mut v4_attempt = QwenMoeV4AttemptState::new(mulmmid_v4_requested);
        if let Some((shared_gate, shared_up, shared_down, shared_route_weights, shared_token_ids)) =
            shared
        {
            let (gate_wb, gate_off) = wrap(shared_gate);
            let (up_wb, up_off) = wrap(shared_up);
            let (down_wb, down_off) = wrap(shared_down);
            let input = if gather_enabled {
                None
            } else {
                Some(norm_all.to_vec())
            };
            let total_elems = seq_len
                .checked_mul(hidden_dim)
                .and_then(|v| u32::try_from(v).ok())
                .expect("Qwen MoE shared prefill element count overflow");
            let mut pending = vec![QwenMoePrefillPendingGroup {
                group_len: seq_len,
                input,
                gate_wb,
                gate_off,
                up_wb,
                up_off,
                down_wb,
                down_off,
                group_start: 0,
                total_elems,
            }];
            if combine_shared_with_sparse {
                shared_pending = pending.pop();
                if trace {
                    eprintln!("[metal-qwen-moe-prefill-accum] shared queued seq_len={seq_len}");
                }
            } else {
                let shared_token_ids_buf = ffn_chain::shared_u32_buf(ctx, shared_token_ids);
                let shared_route_weights_buf = ffn_chain::shared_f32_buf(ctx, shared_route_weights);
                let flushed = self.qwen_moe_prefill_sparse_accum_flush_batch(
                    ctx,
                    accum,
                    &mut pending,
                    down_is_q6k,
                    hidden_dim,
                    ffn_dim,
                    norm_all_buf,
                    &shared_token_ids_buf,
                    &shared_route_weights_buf,
                );
                if (mulmmid_v3_requested || mulmmid_v4_requested) && flushed > 0 {
                    shared_preserved_for_mulmmid = true;
                }
                if trace {
                    eprintln!(
                        "[metal-qwen-moe-prefill-accum] shared groups={flushed} seq_len={seq_len}"
                    );
                }
            }
        }

        if mulmmid_v4_requested {
            match self.try_qwen_moe_prefill_mulmmid_v4(
                ctx,
                accum,
                &gate_all_wb,
                gate_all_off,
                &up_all_wb,
                up_all_off,
                &down_all_wb,
                down_all_off,
                gate_all.len(),
                up_all.len(),
                down_all.len(),
                gate_expert_bytes,
                up_expert_bytes,
                down_expert_bytes,
                expert_ids,
                route_weights,
                token_ids,
                down_is_q6k,
                seq_len,
                hidden_dim,
                ffn_dim,
                shared_preserved_for_mulmmid,
                trace,
            ) {
                Ok(out) => return out,
                Err(reason) => {
                    v4_attempt.record_preflight_failure();
                    if trace || timing {
                        eprintln!(
                            "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason={}",
                            reason.as_mulmmid_v3_str()
                        );
                    }
                }
            }
        }

        if mulmmid_v3_requested && !v4_attempt.skip_id_paths() {
            match self.try_qwen_moe_prefill_mulmmid_v3(
                ctx,
                accum,
                &gate_all_wb,
                gate_all_off,
                &up_all_wb,
                up_all_off,
                &down_all_wb,
                down_all_off,
                gate_all.len(),
                up_all.len(),
                down_all.len(),
                gate_expert_bytes,
                up_expert_bytes,
                down_expert_bytes,
                expert_ids,
                route_weights,
                token_ids,
                down_is_q6k,
                seq_len,
                hidden_dim,
                ffn_dim,
                norm_all,
                shared_preserved_for_mulmmid,
                trace,
            ) {
                Ok(out) => return out,
                Err(reason) if trace || timing => {
                    eprintln!(
                        "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason={}",
                        reason.as_mulmmid_v3_str()
                    );
                }
                Err(_) => {}
            }
        }

        if id_primitive_requested && !v4_attempt.skip_id_paths() {
            match self.try_qwen_moe_prefill_id_primitive(
                ctx,
                accum,
                &gate_all_wb,
                gate_all_off,
                &up_all_wb,
                up_all_off,
                &down_all_wb,
                down_all_off,
                gate_all.len(),
                up_all.len(),
                down_all.len(),
                gate_expert_bytes,
                up_expert_bytes,
                down_expert_bytes,
                expert_ids,
                route_weights,
                token_ids,
                down_is_q6k,
                seq_len,
                hidden_dim,
                ffn_dim,
                timing,
                trace,
            ) {
                Ok(out) => return out,
                Err(reason) if trace || timing => {
                    eprintln!(
                        "[metal-qwen-moe-prefill-accum] mode=id_primitive fallback_reason={}",
                        reason.as_str()
                    );
                }
                Err(_) => {}
            }
        }

        if id_gate_up_requested && !v4_attempt.skip_id_paths() {
            let supported_f32 = ffn_chain::qwen_moe_prefill_id_gate_up_supported(ctx, down_is_q6k);
            let supported_f16 =
                ffn_chain::qwen_moe_prefill_id_gate_up_f16_supported(ctx, down_is_q6k);
            let supported = if id_gate_up_f16_requested {
                supported_f16
            } else {
                supported_f32
            };
            let valid_expert_bytes = gate_expert_bytes > 0
                && up_expert_bytes > 0
                && down_expert_bytes > 0
                && gate_all.len() % gate_expert_bytes == 0
                && up_all.len() % up_expert_bytes == 0
                && down_all.len() % down_expert_bytes == 0;
            let n_expert = if valid_expert_bytes {
                gate_all.len() / gate_expert_bytes
            } else {
                0
            };
            let expert_arenas_match = valid_expert_bytes
                && up_all.len() / up_expert_bytes == n_expert
                && down_all.len() / down_expert_bytes == n_expert;
            let sorted_by_expert = expert_ids.windows(2).all(|pair| pair[0] <= pair[1]);
            if supported && expert_arenas_match && sorted_by_expert && !expert_ids.is_empty() {
                const ID_GATE_UP_BLOCK_ROWS: usize = 64;
                let slots = expert_ids.len();
                let mut expert_offsets = vec![0u32; n_expert];
                let mut expert_counts = vec![0u32; n_expert];
                let mut block_experts = Vec::new();
                let mut block_local0 = Vec::new();
                let mut max_count = 0usize;
                let mut group_count = 0usize;
                let mut group_start = 0usize;
                while group_start < slots {
                    let expert = expert_ids[group_start] as usize;
                    assert!(expert < n_expert, "Qwen MoE expert id out of range");
                    let mut group_end = group_start + 1;
                    while group_end < slots && expert_ids[group_end] as usize == expert {
                        group_end += 1;
                    }
                    for &token in &token_ids[group_start..group_end] {
                        assert!((token as usize) < seq_len, "Qwen MoE token id out of range");
                    }
                    let count = group_end - group_start;
                    expert_offsets[expert] = group_start as u32;
                    expert_counts[expert] = count as u32;
                    for local0 in (0..count).step_by(ID_GATE_UP_BLOCK_ROWS) {
                        block_experts.push(expert as u32);
                        block_local0.push(local0 as u32);
                    }
                    max_count = max_count.max(count);
                    group_count += 1;
                    group_start = group_end;
                }
                debug_assert!(!block_experts.is_empty());
                debug_assert_eq!(block_experts.len(), block_local0.len());
                let block_count = block_experts.len();

                let expert_offsets_buf = ffn_chain::shared_u32_buf(ctx, &expert_offsets);
                let expert_counts_buf = ffn_chain::shared_u32_buf(ctx, &expert_counts);
                let block_experts_buf = ffn_chain::shared_u32_buf(ctx, &block_experts);
                let block_local0_buf = ffn_chain::shared_u32_buf(ctx, &block_local0);
                let (gate_all_wb, gate_all_off) = wrap(gate_all);
                let (up_all_wb, up_all_off) = wrap(up_all);
                let (down_all_wb, down_all_off) = wrap(down_all);
                if id_gate_up_f16_requested {
                    let mut id_carriers =
                        self.qwen_moe_prefill_id_gate_up_f16_carriers.borrow_mut();
                    let id_carrier = id_carriers
                        .entry((slots, hidden_dim, ffn_dim))
                        .or_insert_with(|| {
                            ffn_chain::QwenMoePrefillIdGateUpF16Carrier::new(ctx, slots, ffn_dim)
                        });
                    let cmd = ctx.queue.commandBuffer().expect("command buffer");
                    let enc = cmd.computeCommandEncoder().expect("compute encoder");
                    ffn_chain::qwen_moe_prefill_id_gate_up_f16_encode(
                        ctx,
                        &enc,
                        id_carrier,
                        &accum.norm_all_dev,
                        &gate_all_wb,
                        gate_all_off,
                        &up_all_wb,
                        up_all_off,
                        gate_expert_bytes,
                        up_expert_bytes,
                        &token_ids_buf,
                        &expert_offsets_buf,
                        &expert_counts_buf,
                        &block_experts_buf,
                        &block_local0_buf,
                        block_count,
                        hidden_dim,
                    );
                    let mut ffn_carriers = self.qwen_moe_prefill_private_ffn_carriers.borrow_mut();
                    for expert in 0..n_expert {
                        let count = expert_counts[expert] as usize;
                        if count == 0 {
                            continue;
                        }
                        let group_start = expert_offsets[expert];
                        let total_elems = count
                            .checked_mul(hidden_dim)
                            .and_then(|v| u32::try_from(v).ok())
                            .expect("Qwen MoE id gate/up f16 scatter element count overflow");
                        let down_expert_byte_offset = expert
                            .checked_mul(down_expert_bytes)
                            .and_then(|v| u32::try_from(v).ok())
                            .expect("Qwen MoE id gate/up f16 down offset overflow");
                        let down_off = down_all_off
                            .checked_add(down_expert_byte_offset)
                            .expect("Qwen MoE id gate/up f16 down buffer offset overflow");
                        let carrier = ffn_carriers
                            .entry((hidden_dim, ffn_dim, count))
                            .or_insert_with(|| {
                                ffn_chain::PrefillFfnCarrier::new_private_intermediates(
                                    ctx, hidden_dim, ffn_dim, count,
                                )
                            });
                        ffn_chain::qwen_moe_prefill_id_gate_up_f16_down_scatter_encode(
                            ctx,
                            &enc,
                            id_carrier,
                            carrier,
                            accum,
                            &down_all_wb,
                            down_off,
                            down_is_q6k,
                            group_start,
                            count,
                            total_elems,
                            &token_ids_buf,
                            &route_weights_buf,
                        );
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    if trace {
                        let scratch_mib = id_carrier.scratch_bytes() as f64 / (1024.0 * 1024.0);
                        eprintln!(
                            "[metal-qwen-moe-prefill-accum] mode=id_gate_up_f16 groups={group_count} slots={slots} experts={n_expert} max_count={max_count} blocks={block_count} scratch_mib={scratch_mib:.1}"
                        );
                    }
                    let read0 = timing.then(std::time::Instant::now);
                    let out = accum.readback();
                    if timing {
                        let read_ms = read0
                            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                            .unwrap_or(0.0);
                        eprintln!("[metal-qwen-moe-prefill-accum-time] readback_ms={read_ms:.3}");
                    }
                    return out;
                }
                let mut id_carriers = self.qwen_moe_prefill_id_gate_up_carriers.borrow_mut();
                let id_carrier = id_carriers
                    .entry((slots, hidden_dim, ffn_dim))
                    .or_insert_with(|| {
                        ffn_chain::QwenMoePrefillIdGateUpCarrier::new(ctx, slots, ffn_dim)
                    });
                let cmd = ctx.queue.commandBuffer().expect("command buffer");
                let enc = cmd.computeCommandEncoder().expect("compute encoder");
                ffn_chain::qwen_moe_prefill_id_gate_up_encode(
                    ctx,
                    &enc,
                    id_carrier,
                    &accum.norm_all_dev,
                    &gate_all_wb,
                    gate_all_off,
                    &up_all_wb,
                    up_all_off,
                    gate_expert_bytes,
                    up_expert_bytes,
                    &token_ids_buf,
                    &expert_offsets_buf,
                    &expert_counts_buf,
                    &block_experts_buf,
                    &block_local0_buf,
                    block_count,
                    hidden_dim,
                );
                let mut ffn_carriers = self.qwen_moe_prefill_private_ffn_carriers.borrow_mut();
                for expert in 0..n_expert {
                    let count = expert_counts[expert] as usize;
                    if count == 0 {
                        continue;
                    }
                    let group_start = expert_offsets[expert];
                    let total_elems = count
                        .checked_mul(hidden_dim)
                        .and_then(|v| u32::try_from(v).ok())
                        .expect("Qwen MoE id gate/up scatter element count overflow");
                    let down_expert_byte_offset = expert
                        .checked_mul(down_expert_bytes)
                        .and_then(|v| u32::try_from(v).ok())
                        .expect("Qwen MoE id gate/up down offset overflow");
                    let down_off = down_all_off
                        .checked_add(down_expert_byte_offset)
                        .expect("Qwen MoE id gate/up down buffer offset overflow");
                    let carrier = ffn_carriers
                        .entry((hidden_dim, ffn_dim, count))
                        .or_insert_with(|| {
                            ffn_chain::PrefillFfnCarrier::new_private_intermediates(
                                ctx, hidden_dim, ffn_dim, count,
                            )
                        });
                    ffn_chain::qwen_moe_prefill_id_gate_up_down_scatter_encode(
                        ctx,
                        &enc,
                        id_carrier,
                        carrier,
                        accum,
                        &down_all_wb,
                        down_off,
                        down_is_q6k,
                        group_start,
                        count,
                        total_elems,
                        &token_ids_buf,
                        &route_weights_buf,
                    );
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if trace {
                    let scratch_mib = id_carrier.scratch_bytes() as f64 / (1024.0 * 1024.0);
                    eprintln!(
                        "[metal-qwen-moe-prefill-accum] mode=id_gate_up groups={group_count} slots={slots} experts={n_expert} max_count={max_count} blocks={block_count} scratch_mib={scratch_mib:.1}"
                    );
                }
                let read0 = timing.then(std::time::Instant::now);
                let out = accum.readback();
                if timing {
                    let read_ms = read0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    eprintln!("[metal-qwen-moe-prefill-accum-time] readback_ms={read_ms:.3}");
                }
                return out;
            } else if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=id_gate_up unsupported requested_f16={id_gate_up_f16_requested} supported={supported} supported_f32={supported_f32} supported_f16={supported_f16} expert_arenas_match={expert_arenas_match} sorted_by_expert={sorted_by_expert} slots={}",
                    expert_ids.len()
                );
            }
        }

        if id_matmul_requested && !v4_attempt.skip_id_paths() {
            let supported = ffn_chain::qwen_moe_prefill_id_matmul_supported(ctx, down_is_q6k);
            let valid_expert_bytes = gate_expert_bytes > 0
                && up_expert_bytes > 0
                && down_expert_bytes > 0
                && gate_all.len() % gate_expert_bytes == 0
                && up_all.len() % up_expert_bytes == 0
                && down_all.len() % down_expert_bytes == 0;
            let n_expert = if valid_expert_bytes {
                gate_all.len() / gate_expert_bytes
            } else {
                0
            };
            let expert_arenas_match = valid_expert_bytes
                && up_all.len() / up_expert_bytes == n_expert
                && down_all.len() / down_expert_bytes == n_expert;
            let sorted_by_expert = expert_ids.windows(2).all(|pair| pair[0] <= pair[1]);
            if supported && expert_arenas_match && sorted_by_expert && !expert_ids.is_empty() {
                const ID_MATMUL_BLOCK_ROWS: usize = 64;
                let slots = expert_ids.len();
                let mut expert_offsets = vec![0u32; n_expert];
                let mut expert_counts = vec![0u32; n_expert];
                let mut block_experts = Vec::new();
                let mut block_local0 = Vec::new();
                let mut max_count = 0usize;
                let mut group_count = 0usize;
                let mut group_start = 0usize;
                while group_start < slots {
                    let expert = expert_ids[group_start] as usize;
                    assert!(expert < n_expert, "Qwen MoE expert id out of range");
                    let mut group_end = group_start + 1;
                    while group_end < slots && expert_ids[group_end] as usize == expert {
                        group_end += 1;
                    }
                    for &token in &token_ids[group_start..group_end] {
                        assert!((token as usize) < seq_len, "Qwen MoE token id out of range");
                    }
                    let count = group_end - group_start;
                    expert_offsets[expert] = group_start as u32;
                    expert_counts[expert] = count as u32;
                    for local0 in (0..count).step_by(ID_MATMUL_BLOCK_ROWS) {
                        block_experts.push(expert as u32);
                        block_local0.push(local0 as u32);
                    }
                    max_count = max_count.max(count);
                    group_count += 1;
                    group_start = group_end;
                }
                debug_assert!(!block_experts.is_empty());
                debug_assert_eq!(block_experts.len(), block_local0.len());
                let block_count = block_experts.len();

                let expert_offsets_buf = ffn_chain::shared_u32_buf(ctx, &expert_offsets);
                let expert_counts_buf = ffn_chain::shared_u32_buf(ctx, &expert_counts);
                let block_experts_buf = ffn_chain::shared_u32_buf(ctx, &block_experts);
                let block_local0_buf = ffn_chain::shared_u32_buf(ctx, &block_local0);
                let (gate_all_wb, gate_all_off) = wrap(gate_all);
                let (up_all_wb, up_all_off) = wrap(up_all);
                let (down_all_wb, down_all_off) = wrap(down_all);
                let mut id_carriers = self.qwen_moe_prefill_id_carriers.borrow_mut();
                let carrier = id_carriers
                    .entry((slots, hidden_dim, ffn_dim))
                    .or_insert_with(|| {
                        ffn_chain::QwenMoePrefillIdCarrier::new(ctx, slots, hidden_dim, ffn_dim)
                    });
                let cmd = ctx.queue.commandBuffer().expect("command buffer");
                let enc = cmd.computeCommandEncoder().expect("compute encoder");
                ffn_chain::qwen_moe_prefill_id_matmul_encode(
                    ctx,
                    &enc,
                    carrier,
                    accum,
                    &accum.norm_all_dev,
                    &gate_all_wb,
                    gate_all_off,
                    &up_all_wb,
                    up_all_off,
                    &down_all_wb,
                    down_all_off,
                    gate_expert_bytes,
                    up_expert_bytes,
                    down_expert_bytes,
                    &token_ids_buf,
                    &route_weights_buf,
                    &expert_offsets_buf,
                    &expert_counts_buf,
                    &block_experts_buf,
                    &block_local0_buf,
                    &expert_offsets,
                    &expert_counts,
                    block_count,
                    n_expert,
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if trace {
                    let scratch_mib = carrier.scratch_bytes() as f64 / (1024.0 * 1024.0);
                    eprintln!(
                        "[metal-qwen-moe-prefill-accum] mode=id_matmul groups={group_count} slots={slots} experts={n_expert} max_count={max_count} blocks={block_count} scratch_mib={scratch_mib:.1}"
                    );
                }
                let read0 = timing.then(std::time::Instant::now);
                let out = accum.readback();
                if timing {
                    let read_ms = read0
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    eprintln!("[metal-qwen-moe-prefill-accum-time] readback_ms={read_ms:.3}");
                }
                return out;
            } else if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=id_matmul unsupported supported={supported} expert_arenas_match={expert_arenas_match} sorted_by_expert={sorted_by_expert} slots={}",
                    expert_ids.len()
                );
            }
        }

        const QWEN_MOE_PREFILL_GROUP_BATCH_DEFAULT: usize = 256;
        const QWEN_MOE_PREFILL_GROUP_BATCH_MAX: usize = 256;
        let batch_width = match std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_GROUP_BATCH") {
            Ok(value) => {
                let lower = value.to_ascii_lowercase();
                if matches!(lower.as_str(), "0" | "false" | "off" | "no") {
                    1
                } else {
                    value
                        .parse::<usize>()
                        .ok()
                        .filter(|v| *v > 0)
                        .unwrap_or(QWEN_MOE_PREFILL_GROUP_BATCH_DEFAULT)
                        .min(QWEN_MOE_PREFILL_GROUP_BATCH_MAX)
                }
            }
            Err(_) => QWEN_MOE_PREFILL_GROUP_BATCH_DEFAULT,
        };
        let reuse_group_len_requested =
            std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_REUSE_GROUPLEN")
                .map(|v| {
                    !matches!(
                        v.to_ascii_lowercase().as_str(),
                        "0" | "false" | "off" | "no"
                    )
                })
                .unwrap_or(true);
        let reuse_group_len_in_batch = gather_enabled && reuse_group_len_requested;
        let mut pending: Vec<QwenMoePrefillPendingGroup> = Vec::with_capacity(batch_width);
        let mut submitted_batches = 0usize;
        let mut submitted_groups = 0usize;
        let (default_token_ids_buf, default_route_weights_buf, sparse_group_start_base) =
            if combine_shared_with_sparse {
                let (_, _, _, shared_route_weights, shared_token_ids) =
                    shared.expect("Qwen MoE shared route missing");
                let (combined_token_ids, combined_route_weights, sparse_base) =
                    qwen_moe_combined_shared_sparse_routes(
                        shared_token_ids,
                        shared_route_weights,
                        token_ids,
                        route_weights,
                    );
                pending.push(
                    shared_pending
                        .take()
                        .expect("Qwen MoE shared pending missing"),
                );
                (
                    ffn_chain::shared_u32_buf(ctx, &combined_token_ids),
                    ffn_chain::shared_f32_buf(ctx, &combined_route_weights),
                    sparse_base,
                )
            } else {
                (token_ids_buf.clone(), route_weights_buf.clone(), 0)
            };

        let mut group_start = 0usize;
        while group_start < expert_ids.len() {
            let expert = expert_ids[group_start] as usize;
            let mut group_end = group_start + 1;
            while group_end < expert_ids.len() && expert_ids[group_end] as usize == expert {
                group_end += 1;
            }
            let group_len = group_end - group_start;
            let total_elems = group_len
                .checked_mul(hidden_dim)
                .and_then(|v| u32::try_from(v).ok())
                .expect("Qwen MoE prefill group element count overflow");
            let gate_start = expert
                .checked_mul(gate_expert_bytes)
                .expect("Qwen MoE gate offset overflow");
            let up_start = expert
                .checked_mul(up_expert_bytes)
                .expect("Qwen MoE up offset overflow");
            let down_start = expert
                .checked_mul(down_expert_bytes)
                .expect("Qwen MoE down offset overflow");
            let gate_end = gate_start
                .checked_add(gate_expert_bytes)
                .expect("Qwen MoE gate end overflow");
            let up_end = up_start
                .checked_add(up_expert_bytes)
                .expect("Qwen MoE up end overflow");
            let down_end = down_start
                .checked_add(down_expert_bytes)
                .expect("Qwen MoE down end overflow");
            assert!(
                gate_end <= gate_all.len(),
                "Qwen MoE gate expert slice out of range"
            );
            assert!(
                up_end <= up_all.len(),
                "Qwen MoE up expert slice out of range"
            );
            assert!(
                down_end <= down_all.len(),
                "Qwen MoE down expert slice out of range"
            );

            let mut input = if gather_enabled {
                None
            } else {
                Some(vec![0.0f32; group_len * hidden_dim])
            };
            for local in 0..group_len {
                let token = token_ids[group_start + local] as usize;
                assert!(token < seq_len, "Qwen MoE token id out of range");
                if let Some(input) = input.as_mut() {
                    input[local * hidden_dim..(local + 1) * hidden_dim]
                        .copy_from_slice(&norm_all[token * hidden_dim..(token + 1) * hidden_dim]);
                }
            }

            let gate_off = u32::try_from(
                (gate_all_off as usize)
                    .checked_add(gate_start)
                    .expect("Qwen MoE gate buffer offset overflow"),
            )
            .expect("Qwen MoE gate buffer offset exceeds u32");
            let up_off = u32::try_from(
                (up_all_off as usize)
                    .checked_add(up_start)
                    .expect("Qwen MoE up buffer offset overflow"),
            )
            .expect("Qwen MoE up buffer offset exceeds u32");
            let down_off = u32::try_from(
                (down_all_off as usize)
                    .checked_add(down_start)
                    .expect("Qwen MoE down buffer offset overflow"),
            )
            .expect("Qwen MoE down buffer offset exceeds u32");

            if !pending.is_empty()
                && (pending.len() >= batch_width
                    || (!reuse_group_len_in_batch
                        && pending.iter().any(|group| group.group_len == group_len)))
            {
                let flushed = self.qwen_moe_prefill_sparse_accum_flush_batch(
                    ctx,
                    accum,
                    &mut pending,
                    down_is_q6k,
                    hidden_dim,
                    ffn_dim,
                    norm_all_buf,
                    &default_token_ids_buf,
                    &default_route_weights_buf,
                );
                if flushed > 0 {
                    submitted_batches += 1;
                    submitted_groups += flushed;
                }
            }

            pending.push(QwenMoePrefillPendingGroup {
                group_len,
                input,
                gate_wb: gate_all_wb.clone(),
                gate_off,
                up_wb: up_all_wb.clone(),
                up_off,
                down_wb: down_all_wb.clone(),
                down_off,
                group_start: sparse_group_start_base
                    .checked_add(group_start as u32)
                    .expect("Qwen MoE combined sparse group_start overflow"),
                total_elems,
            });
            group_start = group_end;
        }
        let flushed = self.qwen_moe_prefill_sparse_accum_flush_batch(
            ctx,
            accum,
            &mut pending,
            down_is_q6k,
            hidden_dim,
            ffn_dim,
            norm_all_buf,
            &default_token_ids_buf,
            &default_route_weights_buf,
        );
        if flushed > 0 {
            submitted_batches += 1;
            submitted_groups += flushed;
        }
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] submit groups={submitted_groups} batches={submitted_batches} batch_width={batch_width} gather={gather_enabled} reuse_group_len={reuse_group_len_in_batch} shared_combined={combine_shared_with_sparse}"
            );
        }
        let read0 = timing.then(std::time::Instant::now);
        let out = accum.readback();
        if timing {
            let read_ms = read0
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!("[metal-qwen-moe-prefill-accum-time] readback_ms={read_ms:.3}");
        }
        out
    }

    /// Qwen3.6 MoE decode sparse/shared expert id-style batch. Sparse expert weights
    /// are passed as one contiguous quantized arena and selected by expert id on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen_moe_decode_sparse_experts_id(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        shared_expert_id: u32,
        down_quant: u8,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(expert_ids.len(), route_weights.len(), "expert/route slots");
        assert!(!expert_ids.is_empty(), "Qwen MoE decode id empty route");
        assert_eq!(input.len(), n_embd, "Qwen MoE decode id input length");
        assert!(
            matches!(down_quant, 1 | 2),
            "Qwen MoE decode id supports Q5_K/Q6_K down"
        );

        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (gate_all_wb, gate_all_off) = wrap(gate_all);
        let (up_all_wb, up_all_off) = wrap(up_all);
        let (down_all_wb, down_all_off) = wrap(down_all);
        let (shared_gate_wb, shared_gate_off) = wrap(shared_gate);
        let (shared_up_wb, shared_up_off) = wrap(shared_up);
        let (shared_down_wb, shared_down_off) = wrap(shared_down);

        let mut carriers = self.qwen_moe_decode_id_carriers.borrow_mut();
        let carrier = carriers
            .entry((expert_ids.len(), n_embd, n_ff, down_quant as usize))
            .or_insert_with(|| {
                ffn_chain::QwenMoeDecodeIdCarrier::new(
                    ctx,
                    expert_ids.len(),
                    n_embd,
                    n_ff,
                    gate_expert_bytes,
                    up_expert_bytes,
                    down_expert_bytes,
                    shared_expert_id,
                )
            });

        ffn_chain::qwen_moe_decode_id_dispatch(
            ctx,
            carrier,
            &gate_all_wb,
            gate_all_off,
            &up_all_wb,
            up_all_off,
            &down_all_wb,
            down_all_off,
            &shared_gate_wb,
            shared_gate_off,
            &shared_up_wb,
            shared_up_off,
            &shared_down_wb,
            shared_down_off,
            input,
            expert_ids,
            route_weights,
            down_quant,
        )
    }

    /// Qwen3.6 MoE decode sparse/shared expert batch. Each selected expert keeps
    /// quantized raw weights as-is; this only batches gate/up/down GEMV dispatches
    /// into one command buffer and reuses small activation/output buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen_moe_decode_sparse_experts(
        &self,
        gate: &[&[u8]],
        up: &[&[u8]],
        down: &[&[u8]],
        route_weights: &[f32],
        down_quant: u8,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(gate.len(), route_weights.len(), "gate/route slots");
        assert_eq!(up.len(), route_weights.len(), "up/route slots");
        assert_eq!(down.len(), route_weights.len(), "down/route slots");
        assert_eq!(input.len(), n_embd, "Qwen MoE decode input length");
        assert!(!route_weights.is_empty(), "Qwen MoE decode empty route");
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let mut gate_w = Vec::with_capacity(route_weights.len());
        let mut gate_off = Vec::with_capacity(route_weights.len());
        let mut up_w = Vec::with_capacity(route_weights.len());
        let mut up_off = Vec::with_capacity(route_weights.len());
        let mut down_w = Vec::with_capacity(route_weights.len());
        let mut down_off = Vec::with_capacity(route_weights.len());
        for slot in 0..route_weights.len() {
            let (w, off) = wrap(gate[slot]);
            gate_w.push(w);
            gate_off.push(off);
            let (w, off) = wrap(up[slot]);
            up_w.push(w);
            up_off.push(off);
            let (w, off) = wrap(down[slot]);
            down_w.push(w);
            down_off.push(off);
        }

        let mut carriers = self.qwen_moe_decode_carriers.borrow_mut();
        let carrier = carriers
            .entry((route_weights.len(), n_embd, n_ff))
            .or_insert_with(|| {
                ffn_chain::QwenMoeDecodeCarrier::new(ctx, route_weights.len(), n_embd, n_ff)
            });
        ffn_chain::qwen_moe_decode_dispatch(
            ctx,
            carrier,
            input,
            &gate_w,
            &gate_off,
            &up_w,
            &up_off,
            &down_w,
            &down_off,
            route_weights,
            down_quant,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn glm_moe_decode_iq2xxs_iq3xxs(
        &self,
        gate: &[&[u8]],
        up: &[&[u8]],
        down: &[&[u8]],
        route_weights: &[f32],
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route_weight: f32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        select: GlmMoeQuantSelect,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let sparse_slots = route_weights.len();
        assert_eq!(gate.len(), sparse_slots, "GLM MoE gate/route slots");
        assert_eq!(up.len(), sparse_slots, "GLM MoE up/route slots");
        assert_eq!(down.len(), sparse_slots, "GLM MoE down/route slots");
        assert!(
            sparse_slots > 0 && sparse_slots <= 8,
            "GLM MoE decode supports one to eight sparse experts"
        );
        assert_eq!(input.len(), n_embd, "GLM MoE decode input length");
        assert!(!shared_gate.is_empty(), "GLM MoE shared gate is empty");
        assert!(!shared_up.is_empty(), "GLM MoE shared up is empty");
        assert!(!shared_down.is_empty(), "GLM MoE shared down is empty");

        self.ensure_weight_residency(ctx);
        let residency_enabled = self.weight_residency_enabled();
        let wrap = |raw: &[u8]| {
            let key = resident_key(raw);
            let mut resident = self.resident.borrow_mut();
            let entry = resident
                .entry(key)
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            if residency_enabled {
                if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                    lru.touch(key, &entry.0);
                }
            }
            (entry.0.clone(), entry.1)
        };
        let slots = sparse_slots + 1;
        let mut gate_w = Vec::with_capacity(slots);
        let mut gate_off = Vec::with_capacity(slots);
        let mut up_w = Vec::with_capacity(slots);
        let mut up_off = Vec::with_capacity(slots);
        let mut down_w = Vec::with_capacity(slots);
        let mut down_off = Vec::with_capacity(slots);
        for slot in 0..sparse_slots {
            let (weight, offset) = wrap(gate[slot]);
            gate_w.push(weight);
            gate_off.push(offset);
            let (weight, offset) = wrap(up[slot]);
            up_w.push(weight);
            up_off.push(offset);
            let (weight, offset) = wrap(down[slot]);
            down_w.push(weight);
            down_off.push(offset);
        }
        let (weight, offset) = wrap(shared_gate);
        gate_w.push(weight);
        gate_off.push(offset);
        let (weight, offset) = wrap(shared_up);
        up_w.push(weight);
        up_off.push(offset);
        let (weight, offset) = wrap(shared_down);
        down_w.push(weight);
        down_off.push(offset);
        if residency_enabled {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.commit_if_dirty();
            }
        }

        let mut all_route_weights = Vec::with_capacity(slots);
        all_route_weights.extend_from_slice(route_weights);
        all_route_weights.push(shared_route_weight);
        let mut carriers = self.qwen_moe_decode_carriers.borrow_mut();
        let carrier = carriers
            .entry((slots, n_embd, n_ff))
            .or_insert_with(|| ffn_chain::QwenMoeDecodeCarrier::new(ctx, slots, n_embd, n_ff));
        ffn_chain::glm_moe_decode_iq2xxs_iq3xxs_dispatch(
            ctx,
            carrier,
            input,
            sparse_slots,
            &gate_w,
            &gate_off,
            &up_w,
            &up_off,
            &down_w,
            &down_off,
            &all_route_weights,
            select,
        )
    }

    /// pm115 M1: unique expert slice 를 파일에서 pread(F_NOCACHE) 로 재사용
    /// 스테이징 buffer 에 병렬로 읽고 `(dedup_key → (buffer, offset))` 맵을
    /// 돌려준다. tensor 범위 밖 slice 는 맵에 넣지 않는다 (mmap wrap fallback).
    #[allow(clippy::type_complexity)]
    fn glm_prefill_stage_direct_file(
        &self,
        ctx: &compute::MetalContext,
        source: &GlmPrefillDirectFile<'_>,
        pools: &[&[&[u8]]; 3],
    ) -> std::collections::HashMap<(usize, usize), (Retained<ProtocolObject<dyn MTLBuffer>>, u32)>
    {
        self.glm_prefill_stage_direct_file_pools(
            ctx,
            source,
            &[(0, pools[0]), (1, pools[1]), (2, pools[2])],
            0,
        )
    }

    /// pm116 M2: pool 부분집합만 스테이징하는 일반형. `staging_slot` 으로 스테이징
    /// buffer 를 분리해 이전 스테이지 GPU 가 읽는 동안 덮어쓰기를 방지한다.
    #[allow(clippy::type_complexity)]
    fn glm_prefill_stage_direct_file_pools(
        &self,
        ctx: &compute::MetalContext,
        source: &GlmPrefillDirectFile<'_>,
        pools: &[(usize, &[&[u8]])],
        staging_slot: usize,
    ) -> std::collections::HashMap<(usize, usize), (Retained<ProtocolObject<dyn MTLBuffer>>, u32)>
    {
        let Some(prep) = self.glm_prefill_prepare_direct_pread(ctx, source, pools, staging_slot)
        else {
            return std::collections::HashMap::new();
        };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(2, 8);
        glm_direct_run_preads(&prep.plan, &prep.handles, prep.contents, threads);
        prep.to_map()
    }

    /// pm116 M2: pread 계획/스테이징 준비만 수행 — 실제 읽기는
    /// `glm_direct_run_preads` 로 분리해 caller 가 여러 pool 을 동시에 시작하고
    /// 스테이지별로 join 할 수 있게 한다.
    #[allow(clippy::type_complexity)]
    fn glm_prefill_prepare_direct_pread(
        &self,
        ctx: &compute::MetalContext,
        source: &GlmPrefillDirectFile<'_>,
        pools: &[(usize, &[&[u8]])],
        staging_slot: usize,
    ) -> Option<GlmDirectPreadPrep> {
        const STAGE_ALIGN: usize = 256;

        // 1) unique in-range slice 수집 + 스테이징 오프셋 배정 (bump).
        let mut plan: Vec<((usize, usize), usize, u64, usize)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut stage_total = 0usize;
        for &(pool_id, pool) in pools.iter() {
            let base_addr = source.base_addrs[pool_id];
            let base_len = source.lens[pool_id];
            for raw in pool.iter() {
                let addr = raw.as_ptr() as usize;
                let dedup_key = (addr, raw.len());
                if !seen.insert(dedup_key) {
                    continue;
                }
                let Some(rel) = addr.checked_sub(base_addr) else {
                    continue;
                };
                if rel + raw.len() > base_len {
                    continue;
                }
                let file_off = source.base_offsets[pool_id] + rel as u64;
                plan.push((dedup_key, pool_id, file_off, stage_total));
                stage_total += raw.len().next_multiple_of(STAGE_ALIGN);
            }
        }
        if plan.is_empty() {
            return None;
        }

        // 2) 파일 핸들. pm117 채택: pread 는 page cache 를 쓴다 (F_NOCACHE 제거)
        // — expert 접근 빈도 skew(LFU hit 28~55% 실측) 덕에 hot expert 가
        // cache 에서 ~50GB/s 로 서빙되고, evictable 이라 명시 상주 캐시의
        // 메모리 압박 리스크가 없다. decode 3페어 -8.7/-9.8/-15.3%,
        // prefill 2페어 -13.1/-4.7% same-index 전승, token-identical.
        // `RNB_METAL_GLM_PREAD_PAGECACHE=0` opt-out 시 F_NOCACHE 복귀.
        let use_pagecache = glm_pread_pagecache_enabled();
        let mut handles: [Option<std::sync::Arc<std::fs::File>>; 3] = [None, None, None];
        for &(pool_id, _) in pools.iter() {
            let path = source.paths[pool_id];
            let mut cache = self.glm_direct_file_handles.borrow_mut();
            let entry = cache.entry(path.to_path_buf()).or_insert_with(|| {
                let file = std::fs::File::open(path)
                    .unwrap_or_else(|e| panic!("GLM direct-file open {path:?}: {e}"));
                #[cfg(target_os = "macos")]
                if !use_pagecache {
                    unsafe {
                        use std::os::fd::AsRawFd;
                        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
                    }
                }
                std::sync::Arc::new(file)
            });
            handles[pool_id] = Some(std::sync::Arc::clone(entry));
        }

        // 3) 스테이징 buffer (grow-only 재사용, slot 별 분리).
        let staging = {
            let mut stagings = self.glm_direct_staging.borrow_mut();
            let need_new = match stagings.get(&staging_slot) {
                Some((_, cap)) => *cap < stage_total,
                None => true,
            };
            if need_new {
                let cap = stage_total.next_power_of_two();
                let buf = ctx
                    .device
                    .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                    .expect("GLM direct-file staging buffer");
                stagings.insert(staging_slot, (buf, cap));
            }
            stagings.get(&staging_slot).expect("staged").0.clone()
        };
        let contents = staging.contents().as_ptr() as usize;

        // 4) residency 등록 (스테이징 1개 buffer — per-submit wiring 반복 방지).
        if self.weight_residency_enabled() {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.touch((contents, 0), &staging);
                lru.commit_if_dirty();
            }
        }

        Some(GlmDirectPreadPrep {
            plan,
            handles,
            staging,
            contents,
        })
    }

    /// pm113: GLM MoE prefill token-batch — 토큰별 slot weight 슬라이스를 resident
    /// wrap(중복 expert 는 캐시 dedup)하고 전체 stage 를 단일 command buffer 로
    /// dispatch. slot 배열은 `[seq][sparse+1]` flatten (마지막이 shared).
    #[allow(clippy::too_many_arguments)]
    pub fn glm_moe_prefill_iq_batch(
        &self,
        gate: &[&[u8]],
        up: &[&[u8]],
        down: &[&[u8]],
        route_weights: &[f32],
        seq_len: usize,
        sparse_slots: usize,
        n_ff: usize,
        n_embd: usize,
        input_all: &[f32],
        select: GlmMoeQuantSelect,
        direct_file: Option<&GlmPrefillDirectFile<'_>>,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let slots = sparse_slots + 1;
        assert!(sparse_slots > 0 && sparse_slots <= 8);
        assert_eq!(gate.len(), seq_len * slots, "GLM MoE prefill gate slots");
        assert_eq!(up.len(), seq_len * slots, "GLM MoE prefill up slots");
        assert_eq!(down.len(), seq_len * slots, "GLM MoE prefill down slots");
        assert_eq!(route_weights.len(), seq_len * slots);
        assert_eq!(input_all.len(), seq_len * n_embd);

        self.ensure_weight_residency(ctx);
        let residency_enabled = self.weight_residency_enabled();
        // pm116: `RNB_METAL_GLM_MOE_PREFILL_PROFILE=1` — 레이어(호출)당 pread/wrap
        // wall 을 1줄 출력. encode/gpu 분해는 dispatch 쪽 동일 게이트 줄과 짝.
        let profiling = std::env::var("RNB_METAL_GLM_MOE_PREFILL_PROFILE").as_deref() == Ok("1");
        // pm116 M2: staged overlap — 텐서별 pread 를 이전 스테이지 GPU 와 겹침.
        // direct-file 경로에서만 의미 (pread 가 없으면 겹칠 것도 없음).
        // 537tok ABAB 7페어 same-index 전승/-3.2% 로 채택, 기본 ON
        // (`RNB_METAL_GLM_PREFILL_STAGE_OVERLAP=0` opt-out).
        if direct_file.is_some()
            && glm_prefill_direct_file_enabled()
            && std::env::var("RNB_METAL_GLM_PREFILL_STAGE_OVERLAP").as_deref() != Ok("0")
            && seq_len >= glm_stage_overlap_min_seq()
        {
            return self.glm_moe_prefill_iq_batch_staged(
                ctx,
                gate,
                up,
                down,
                route_weights,
                seq_len,
                sparse_slots,
                n_ff,
                n_embd,
                input_all,
                select,
                direct_file.expect("checked"),
                profiling,
            );
        }
        let mut mark = std::time::Instant::now();
        // pm115 M1: direct-file 경로 — unique expert 를 pread 로 스테이징 buffer 에
        // 읽어 mmap wiring page fault / page cache 경합을 우회한다. 범위 밖
        // slice(shared expert)는 아래 mmap wrap 이 그대로 처리.
        let direct_staged = if direct_file.is_some() && glm_prefill_direct_file_enabled() {
            self.glm_prefill_stage_direct_file(
                ctx,
                direct_file.expect("checked"),
                &[gate, up, down],
            )
        } else {
            std::collections::HashMap::new()
        };
        let pread_ms = mark.elapsed().as_secs_f64() * 1000.0;
        let staged_unique = direct_staged.len();
        mark = std::time::Instant::now();
        // 토큰들이 같은 expert 를 반복 선택하므로 (unique ~수십 << seq*slots)
        // per-call dedup 맵으로 resident 캐시 조회/LRU touch 를 unique 당 1회로.
        let mut wrapped: std::collections::HashMap<(usize, usize), _> =
            std::collections::HashMap::new();
        let mut wrap = |raw: &[u8]| {
            let dedup_key = (raw.as_ptr() as usize, raw.len());
            if let Some(entry) = direct_staged.get(&dedup_key) {
                return entry.clone();
            }
            wrapped
                .entry(dedup_key)
                .or_insert_with(|| {
                    let key = resident_key(raw);
                    let mut resident = self.resident.borrow_mut();
                    let entry = resident
                        .entry(key)
                        .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
                    if residency_enabled {
                        if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                            lru.touch(key, &entry.0);
                        }
                    }
                    (entry.0.clone(), entry.1)
                })
                .clone()
        };
        let total = seq_len * slots;
        let mut gate_w = Vec::with_capacity(total);
        let mut gate_off = Vec::with_capacity(total);
        let mut up_w = Vec::with_capacity(total);
        let mut up_off = Vec::with_capacity(total);
        let mut down_w = Vec::with_capacity(total);
        let mut down_off = Vec::with_capacity(total);
        for slot in 0..total {
            let (weight, offset) = wrap(gate[slot]);
            gate_w.push(weight);
            gate_off.push(offset);
            let (weight, offset) = wrap(up[slot]);
            up_w.push(weight);
            up_off.push(offset);
            let (weight, offset) = wrap(down[slot]);
            down_w.push(weight);
            down_off.push(offset);
        }
        drop(wrap);
        if residency_enabled {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.commit_if_dirty();
            }
        }
        if profiling {
            let wrap_ms = mark.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[glm-moe-prefill-profile] pread={pread_ms:.1} staged={staged_unique} wrap={wrap_ms:.1} mmap_unique={}",
                wrapped.len()
            );
        }

        let mut carriers = self.qwen_moe_decode_carriers.borrow_mut();
        let carrier = carriers
            .entry((slots, n_embd, n_ff))
            .or_insert_with(|| ffn_chain::QwenMoeDecodeCarrier::new(ctx, slots, n_embd, n_ff));
        ffn_chain::glm_moe_prefill_iq_batch_dispatch(
            ctx,
            carrier,
            input_all,
            seq_len,
            sparse_slots,
            &gate_w,
            &gate_off,
            &up_w,
            &up_off,
            &down_w,
            &down_off,
            route_weights,
            select,
        )
    }

    /// pm116 M2: 스테이지-major staged overlap 경로. gate pread → gate commit(no wait)
    /// → up pread(cb0 GPU 와 겹침) → up commit → down pread(겹침) → tail wait.
    /// 산술/커널은 기존 경로와 동일 — dispatch 순서와 scratch 레이아웃만 재배열.
    #[allow(clippy::too_many_arguments)]
    fn glm_moe_prefill_iq_batch_staged(
        &self,
        ctx: &compute::MetalContext,
        gate: &[&[u8]],
        up: &[&[u8]],
        down: &[&[u8]],
        route_weights: &[f32],
        seq_len: usize,
        sparse_slots: usize,
        n_ff: usize,
        n_embd: usize,
        input_all: &[f32],
        select: GlmMoeQuantSelect,
        source: &GlmPrefillDirectFile<'_>,
        profiling: bool,
    ) -> Vec<f32> {
        let slots = sparse_slots + 1;
        let residency_enabled = self.weight_residency_enabled();
        let wrapped: RefCell<
            std::collections::HashMap<
                (usize, usize),
                (Retained<ProtocolObject<dyn MTLBuffer>>, u32),
            >,
        > = RefCell::new(std::collections::HashMap::new());

        let input_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
            let ptr = std::ptr::NonNull::new(input_all.as_ptr() as *mut std::ffi::c_void)
                .expect("prefill input ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(
                    ptr,
                    std::mem::size_of_val(input_all),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: GLM prefill input buffer")
        };

        let mut carriers = self.qwen_moe_decode_carriers.borrow_mut();
        let carrier = carriers
            .entry((slots, n_embd, n_ff))
            .or_insert_with(|| ffn_chain::QwenMoeDecodeCarrier::new(ctx, slots, n_embd, n_ff));
        let mut scratches = self.glm_prefill_stage_scratch.borrow_mut();
        let scratch = scratches
            .entry((slots, n_embd, n_ff))
            .and_modify(|entry| {
                if !entry.fits(seq_len) {
                    *entry =
                        ffn_chain::GlmPrefillStageScratch::new(ctx, seq_len, slots, n_embd, n_ff);
                }
            })
            .or_insert_with(|| {
                ffn_chain::GlmPrefillStageScratch::new(ctx, seq_len, slots, n_embd, n_ff)
            });

        // 텐서별 pread + wrap. staged 맵 hit 는 스테이징 버퍼, miss 는 mmap wrap.
        let wrap_list = |staged: &std::collections::HashMap<
            (usize, usize),
            (Retained<ProtocolObject<dyn MTLBuffer>>, u32),
        >,
                         slices: &[&[u8]]| {
            let mut w = Vec::with_capacity(slices.len());
            let mut off = Vec::with_capacity(slices.len());
            for raw in slices {
                let dedup_key = (raw.as_ptr() as usize, raw.len());
                let entry = if let Some(entry) = staged.get(&dedup_key) {
                    entry.clone()
                } else {
                    wrapped
                        .borrow_mut()
                        .entry(dedup_key)
                        .or_insert_with(|| {
                            let key = resident_key(raw);
                            let mut resident = self.resident.borrow_mut();
                            let entry = resident
                                .entry(key)
                                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
                            if residency_enabled {
                                if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                                    lru.touch(key, &entry.0);
                                }
                            }
                            (entry.0.clone(), entry.1)
                        })
                        .clone()
                };
                w.push(entry.0);
                off.push(entry.1);
            }
            (w, off)
        };

        // pread 는 SSD-bound 라 동시 실행이 gate 완료를 지연시킨다 (M2v2 -19% 실측).
        // 순차 3분할: gate pread → cb0 commit (GPU 시작) → up pread(겹침) →
        // cb1 → down pread(겹침) → tail wait.
        let mut mark = std::time::Instant::now();
        let gate_staged = self.glm_prefill_stage_direct_file_pools(ctx, source, &[(0, gate)], 1);
        let pread_gate_ms = mark.elapsed().as_secs_f64() * 1000.0;
        let (gate_w, gate_off) = wrap_list(&gate_staged, gate);
        carrier.upload_gate_offsets(&gate_off[..slots]);
        ffn_chain::glm_moe_prefill_iq_stage_gate_up_dispatch(
            ctx,
            carrier,
            &scratch.gate_full,
            &input_buf,
            seq_len,
            sparse_slots,
            &gate_w,
            &gate_off,
            true,
            select.gate_up_iq2s,
            select.shared_gate_up_q6k,
        );

        // 스테이지 2: up — pread 가 gate GPU 실행과 겹친다.
        mark = std::time::Instant::now();
        let up_staged = self.glm_prefill_stage_direct_file_pools(ctx, source, &[(1, up)], 2);
        let pread_up_ms = mark.elapsed().as_secs_f64() * 1000.0;
        let (up_w, up_off) = wrap_list(&up_staged, up);
        carrier.upload_up_offsets(&up_off[..slots]);
        ffn_chain::glm_moe_prefill_iq_stage_gate_up_dispatch(
            ctx,
            carrier,
            &scratch.up_full,
            &input_buf,
            seq_len,
            sparse_slots,
            &up_w,
            &up_off,
            false,
            select.gate_up_iq2s,
            select.shared_gate_up_q6k,
        );

        // 스테이지 3: down — pread 가 gate/up GPU 실행과 겹친 뒤 tail 에서 wait.
        mark = std::time::Instant::now();
        let down_staged = self.glm_prefill_stage_direct_file_pools(ctx, source, &[(2, down)], 3);
        let pread_down_ms = mark.elapsed().as_secs_f64() * 1000.0;
        let (down_w, down_off) = wrap_list(&down_staged, down);
        carrier.upload_down_offsets(&down_off[..slots]);
        if residency_enabled {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.commit_if_dirty();
            }
        }
        mark = std::time::Instant::now();
        let out = ffn_chain::glm_moe_prefill_iq_stage_tail_dispatch(
            ctx,
            carrier,
            scratch,
            seq_len,
            sparse_slots,
            &down_w,
            &down_off,
            route_weights,
            select.down_iq4xs,
            select.shared_down_q8_0,
        );
        if profiling {
            eprintln!(
                "[glm-moe-prefill-profile] staged_overlap pread_gate={pread_gate_ms:.1} pread_up={pread_up_ms:.1} pread_down={pread_down_ms:.1} staged={} mmap_unique={} tail_wait={:.1}",
                gate_staged.len() + up_staged.len() + down_staged.len(),
                wrapped.borrow().len(),
                mark.elapsed().as_secs_f64() * 1000.0
            );
        }
        out
    }

    /// pm35 M2: prefill GDN proj(in_proj/gate) single batch GEMM. normed[seq*hidden] + weight raw
    /// (Q4_K|Q6_K) → out[seq*n_out]. weight `(ptr,len)` 키 resident wrap, carrier (hidden,n_out,seq) 키.
    pub fn prefill_gdn_proj(
        &self,
        normed: &[f32],
        weight: &[u8],
        quant: TensoropsQuant,
        seq_len: usize,
        hidden_dim: usize,
        n_out: usize,
    ) -> Vec<f32> {
        self.prefill_gdn_proj_with_trace(normed, weight, quant, seq_len, hidden_dim, n_out, None)
    }

    /// ATN M0 timing path: 기존 GDN projection seam 과 같은 커널을 쓰되 role/layer tag만 붙인다.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_gdn_proj_with_trace(
        &self,
        normed: &[f32],
        weight: &[u8],
        quant: TensoropsQuant,
        seq_len: usize,
        hidden_dim: usize,
        n_out: usize,
        trace: Option<PrefillProjTrace>,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let (w_buf, off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(weight))
                .or_insert_with(|| compute::wrap_nocopy(ctx, weight));
            (e.0.clone(), e.1)
        };
        let off_buf = gdn_proj_chain::u32_buf(ctx, off);
        let mut carriers = self.prefill_gdn_proj_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, n_out, seq_len))
            .or_insert_with(|| {
                gdn_proj_chain::PrefillGdnProjCarrier::new(ctx, hidden_dim, n_out, seq_len)
            });
        gdn_proj_chain::prefill_gdn_proj_dispatch(
            ctx, carrier, normed, &w_buf, &off_buf, quant, seq_len, trace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gdn_quant_dual_proj(
        &self,
        normed: &[f32],
        left_weight: &[u8],
        left_quant: TensoropsQuant,
        right_weight: &[u8],
        right_quant: TensoropsQuant,
        seq_len: usize,
        hidden_dim: usize,
        n_out: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(normed.len(), seq_len * hidden_dim, "normed shape");
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (left_wb, left_off) = wrap(left_weight);
        let (right_wb, right_off) = wrap(right_weight);
        let left_off_buf = gdn_proj_chain::u32_buf(ctx, left_off);
        let right_off_buf = gdn_proj_chain::u32_buf(ctx, right_off);
        let mut carriers = self.prefill_gdn_f32_dual_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, n_out, seq_len))
            .or_insert_with(|| {
                gdn_proj_chain::PrefillGdnF32DualCarrier::new(ctx, hidden_dim, n_out, seq_len)
            });
        gdn_proj_chain::prefill_gdn_quant_dual_dispatch(
            ctx,
            carrier,
            normed,
            &left_wb,
            &left_off_buf,
            left_quant,
            &right_wb,
            &right_off_buf,
            right_quant,
            seq_len,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gdn_f32_dual_proj(
        &self,
        normed: &[f32],
        left_weight: &[u8],
        right_weight: &[u8],
        seq_len: usize,
        hidden_dim: usize,
        n_out: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(normed.len(), seq_len * hidden_dim, "normed shape");
        assert!(
            left_weight.len() >= n_out * hidden_dim * std::mem::size_of::<f32>(),
            "left F32 projection weight shape"
        );
        assert!(
            right_weight.len() >= n_out * hidden_dim * std::mem::size_of::<f32>(),
            "right F32 projection weight shape"
        );
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (left_wb, left_off) = wrap(left_weight);
        let (right_wb, right_off) = wrap(right_weight);
        let left_off_buf = gdn_proj_chain::u32_buf(ctx, left_off);
        let right_off_buf = gdn_proj_chain::u32_buf(ctx, right_off);
        let mut carriers = self.prefill_gdn_f32_dual_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, n_out, seq_len))
            .or_insert_with(|| {
                gdn_proj_chain::PrefillGdnF32DualCarrier::new(ctx, hidden_dim, n_out, seq_len)
            });
        gdn_proj_chain::prefill_gdn_f32_dual_dispatch(
            ctx,
            carrier,
            normed,
            &left_wb,
            &left_off_buf,
            &right_wb,
            &right_off_buf,
        )
    }

    /// Qwen MoE router F32 projection: router_w[n_expert, hidden] x normed[seq, hidden]
    /// -> logits[seq, n_expert]. Weight is viewed as bytes without changing precision.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn qwen_moe_prefill_router_logits(
        &self,
        router_w: &[f32],
        normed: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(router_w.len(), n_expert * hidden_dim, "router_w shape");
        assert_eq!(normed.len(), seq_len * hidden_dim, "router norm shape");
        let router_bytes = unsafe {
            std::slice::from_raw_parts(
                router_w.as_ptr() as *const u8,
                std::mem::size_of_val(router_w),
            )
        };
        let (w_buf, off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(router_bytes))
                .or_insert_with(|| compute::wrap_nocopy(ctx, router_bytes));
            (e.0.clone(), e.1)
        };
        let off_buf = gdn_proj_chain::u32_buf(ctx, off);
        let mut carriers = self.qwen_moe_router_f32_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, n_expert, seq_len))
            .or_insert_with(|| {
                gdn_proj_chain::PrefillF32ProjCarrier::new(ctx, hidden_dim, n_expert, seq_len)
            });
        let logits =
            gdn_proj_chain::prefill_f32_proj_dispatch(ctx, carrier, normed, &w_buf, &off_buf);
        logits
    }

    /// Qwen MoE router projection + top-k route arrays on Metal. Router logits
    /// stay on device; only selected expert ids, route weights, and token ids
    /// are read back.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn qwen_moe_prefill_route_arrays(
        &self,
        router_w: &[f32],
        normed: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        seq_len: usize,
        n_expert_used: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(router_w.len(), n_expert * hidden_dim, "router_w shape");
        assert_eq!(normed.len(), seq_len * hidden_dim, "router norm shape");
        let n_used = n_expert_used.min(n_expert);
        assert!(n_used > 0, "Qwen MoE route top-k requires n_used > 0");
        assert!(
            n_used <= 32,
            "Qwen MoE route top-k Metal kernel supports up to 32 selected experts"
        );
        let router_bytes = unsafe {
            std::slice::from_raw_parts(
                router_w.as_ptr() as *const u8,
                std::mem::size_of_val(router_w),
            )
        };
        let (w_buf, off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(router_bytes))
                .or_insert_with(|| compute::wrap_nocopy(ctx, router_bytes));
            (e.0.clone(), e.1)
        };
        let off_buf = gdn_proj_chain::u32_buf(ctx, off);
        let mut proj_carriers = self.qwen_moe_router_f32_carriers.borrow_mut();
        let proj = proj_carriers
            .entry((hidden_dim, n_expert, seq_len))
            .or_insert_with(|| {
                gdn_proj_chain::PrefillF32ProjCarrier::new(ctx, hidden_dim, n_expert, seq_len)
            });
        let slots = seq_len
            .checked_mul(n_used)
            .expect("Qwen MoE route slot count overflow");
        let mut route_carriers = self.qwen_moe_route_array_carriers.borrow_mut();
        let route = route_carriers
            .entry((seq_len, n_expert, n_used))
            .or_insert_with(|| QwenMoePrefillRouteArrayCarrier::new(ctx, slots));

        let n_expert_buf = gdn_proj_chain::u32_buf(ctx, n_expert as u32);
        let n_used_buf = gdn_proj_chain::u32_buf(ctx, n_used as u32);
        let seq_len_buf = gdn_proj_chain::u32_buf(ctx, seq_len as u32);
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        gdn_proj_chain::prefill_f32_proj_encode_no_readback(
            ctx, &enc, proj, normed, &w_buf, &off_buf,
        );
        enc.setComputePipelineState(&ctx.qwen_moe_prefill_topk_pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&proj.out_dev), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&route.expert_ids_dev), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&route.route_weights_dev), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&route.token_ids_dev), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&n_expert_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&n_used_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&seq_len_buf), 0, 6);
        }
        let tgw = ctx
            .qwen_moe_prefill_topk_pipeline
            .threadExecutionWidth()
            .max(1);
        let grid = MTLSize {
            width: seq_len.div_ceil(tgw),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tgw,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        route.readback()
    }

    /// pm44 M1-3: gated_rmsnorm_silu + ssm_out GEMM fused (단일 command buffer).
    /// out_in[seq*d_inner] + z[seq*d_inner] + norm[head_v_dim] + weight(Q4K/Q5K/Q6K) →
    /// out[seq*n_out]. d_inner = out_in.len()/seq_len.
    /// carrier 캐시 키: (d_inner, head_v_dim, n_out, seq_len).
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gated_norm_silu_project(
        &self,
        out_in: &[f32],
        z: &[f32],
        norm: &[f32],
        weight: &[u8],
        quant: TensoropsQuant,
        seq_len: usize,
        head_v_dim: usize,
        n_out: usize,
        norm_eps: f32,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let d_inner = out_in.len() / seq_len;
        debug_assert_eq!(
            out_in.len(),
            seq_len * d_inner,
            "out_in must be seq*d_inner"
        );
        let (w_buf, off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(weight))
                .or_insert_with(|| compute::wrap_nocopy(ctx, weight));
            (e.0.clone(), e.1)
        };
        let off_buf = gdn_proj_chain::u32_buf(ctx, off);
        let mut carriers = self.prefill_gated_proj_carriers.borrow_mut();
        let carrier = carriers
            .entry((d_inner, head_v_dim, n_out, seq_len))
            .or_insert_with(|| {
                gated_proj_chain::PrefillGdnGatedProjCarrier::new(
                    ctx, seq_len, d_inner, head_v_dim, n_out, norm_eps,
                )
            });
        gated_proj_chain::prefill_gated_norm_silu_project_dispatch(
            ctx, carrier, out_in, z, norm, &w_buf, &off_buf, quant,
        )
    }

    /// pm45 M2: GDN prefill conv→delta device-resident chain (단일 command buffer).
    /// conv1d_silu → split_conv_qkv → l2_norm(q,k) → repeat_qk → delta_net_scan_chunk 를
    /// 단일 encoder 에 묶어 op 별 host 왕복(conv_data readback + q/k/v upload) 제거.
    /// conv_input[(seq+ks-1)*conv_channels] + conv_weight[ks*conv_channels] +
    /// gate/beta[seq*num_v_heads] + state[num_v_heads*head_v_dim*head_k_dim] →
    /// `(out[seq*num_v_heads*head_v_dim], state_after)`. CPU gdn_forward conv→delta seam 과 1:1.
    /// scale(q 만 1/√head_k_dim)은 l2_norm(q)에 흡수. carrier 키:
    /// (seq, conv_channels, conv_kernel, num_k_heads, num_v_heads, head_k_dim, head_v_dim, chunk_size).
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gdn_conv_delta_chain(
        &self,
        conv_input: &[f32],
        conv_weight: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        norm_eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let mut carriers = self.prefill_gdn_conv_delta_carriers.borrow_mut();
        let carrier = carriers
            .entry((
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                chunk_size,
            ))
            .or_insert_with(|| {
                gdn_conv_delta_chain::PrefillGdnConvDeltaCarrier::new(
                    ctx,
                    seq_len,
                    conv_channels,
                    conv_kernel,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    chunk_size,
                    norm_eps,
                )
            });
        gdn_conv_delta_chain::prefill_gdn_conv_delta_chain_dispatch(
            ctx,
            carrier,
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
        )
    }

    /// pm45 M3-1: GDN prefill full chain(conv→delta→gated→ssm_out) device-resident (단일 command buffer).
    /// M2(conv→delta) chain 끝 delta output 을 readback 하지 않고 같은 encoder 에 이어서
    /// gated_rmsnorm_silu → cast → ssm_out GEMM_v2(Q4K/Q5K/Q6K) 를 encode. host 왕복 1회
    /// (upload → commit/wait → proj+state readback) — M2↔M1 사이 delta output readback(~18MB/layer)
    /// + gated input upload 제거. M2 method 와 동일 입력 + M1 입력(z/ssm_norm/ssm_out weight/n_out) 추가.
    /// 반환 `(proj[seq*n_out], state_after)`. carrier 키: M2 키 + n_out.
    /// resident weight 캐시는 M1 ssm_out 패턴(weight.as_ptr() → NoCopy) 재사용.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gdn_full_chain(
        &self,
        conv_input: &[f32],
        conv_weight: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        z: &[f32],
        ssm_norm: &[f32],
        ssm_out_weight: &[u8],
        ssm_out_quant: TensoropsQuant,
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        n_out: usize,
        norm_eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let (w_buf, off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(ssm_out_weight))
                .or_insert_with(|| compute::wrap_nocopy(ctx, ssm_out_weight));
            (e.0.clone(), e.1)
        };
        let off_buf = gdn_proj_chain::u32_buf(ctx, off);
        let mut carriers = self.prefill_gdn_full_carriers.borrow_mut();
        let carrier = carriers
            .entry((
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                chunk_size,
                n_out,
            ))
            .or_insert_with(|| {
                gdn_conv_delta_chain::PrefillGdnFullCarrier::new(
                    ctx,
                    seq_len,
                    conv_channels,
                    conv_kernel,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    chunk_size,
                    n_out,
                    norm_eps,
                )
            });
        gdn_conv_delta_chain::prefill_gdn_full_chain_dispatch(
            ctx,
            carrier,
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
            z,
            ssm_norm,
            &w_buf,
            &off_buf,
            ssm_out_quant,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "macos")]
    pub fn prefill_gdn_full_ffn_chain_if_supported(
        &self,
        hidden: &[f32],
        conv_input: &[f32],
        conv_weight: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        z: &[f32],
        ssm_norm: &[f32],
        ssm_out_weight: &[u8],
        ssm_out_quant: TensoropsQuant,
        post_norm_w: &[f32],
        ffn_gate_weight: &[u8],
        ffn_up_weight: &[u8],
        ffn_down_weight: &[u8],
        ffn_down_is_q6k: bool,
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_eps: f32,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let ctx = self.ctx.as_ref()?;
        let ssm_out_v2 = match ssm_out_quant {
            TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q8_0 => ctx.gemm_q8_0_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q2K => ctx.gemm_q2k_tensorops_v2_pipeline.is_some(),
            TensoropsQuant::Q3K => ctx.gemm_q3k_tensorops_v2_pipeline.is_some(),
        };
        if !ctx.tensorops_capable
            || ctx.cast_f32_f16_pipeline.is_none()
            || !ssm_out_v2
            || ctx.gemm_q4k_tensorops_v2_pipeline.is_none()
            || (ffn_down_is_q6k && ctx.gemm_q6k_tensorops_v2_pipeline.is_none())
        {
            return None;
        }

        let wrap_weight = |raw: &[u8]| {
            let mut resident = self.resident.borrow_mut();
            let entry = resident
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (entry.0.clone(), entry.1)
        };
        let (ssm_out_buf, ssm_out_off) = wrap_weight(ssm_out_weight);
        let (ffn_gate_buf, ffn_gate_off) = wrap_weight(ffn_gate_weight);
        let (ffn_up_buf, ffn_up_off) = wrap_weight(ffn_up_weight);
        let (ffn_down_buf, ffn_down_off) = wrap_weight(ffn_down_weight);
        let ssm_out_off_buf = ffn_chain::u32_buf(ctx, ssm_out_off);
        let ffn_gate_off_buf = ffn_chain::u32_buf(ctx, ffn_gate_off);
        let ffn_up_off_buf = ffn_chain::u32_buf(ctx, ffn_up_off);
        let ffn_down_off_buf = ffn_chain::u32_buf(ctx, ffn_down_off);

        let mut carriers = self.prefill_gdn_full_ffn_carriers.borrow_mut();
        let carrier = carriers
            .entry((
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                chunk_size,
                hidden_dim,
                ffn_dim,
                ffn_down_is_q6k,
                norm_eps.to_bits(),
            ))
            .or_insert_with(|| {
                gdn_conv_delta_chain::PrefillGdnFullFfnCarrier::new(
                    ctx,
                    seq_len,
                    conv_channels,
                    conv_kernel,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    chunk_size,
                    hidden_dim,
                    ffn_dim,
                    norm_eps,
                )
            });
        Some(gdn_conv_delta_chain::prefill_gdn_full_ffn_chain_dispatch(
            ctx,
            carrier,
            hidden,
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
            z,
            ssm_norm,
            &ssm_out_buf,
            &ssm_out_off_buf,
            ssm_out_quant,
            post_norm_w,
            &ffn_gate_buf,
            &ffn_gate_off_buf,
            &ffn_up_buf,
            &ffn_up_off_buf,
            &ffn_down_buf,
            &ffn_down_off_buf,
            ffn_down_is_q6k,
        ))
    }

    /// attention O chain (Q4_K o_proj + residual, 단일 command buffer).
    /// attn_out/hidden(host) + o weight raw(mmap, resident). 반환: residual 적용 hidden.
    pub fn attention_o_chain_resident(
        &self,
        attn_out: &[f32],
        hidden: &[f32],
        o_raw: &[u8],
        hidden_dim: usize,
        q_dim: usize,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let (o_w, o_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(o_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, o_raw));
            (e.0.clone(), e.1)
        };
        let o_off_buf = ffn_chain::u32_buf(ctx, o_off);

        let mut carriers = self.o_chain_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, q_dim))
            .or_insert_with(|| ffn_chain::OChainCarrier::new(ctx, hidden_dim, q_dim));

        ffn_chain::o_chain_dispatch(ctx, carrier, attn_out, hidden, &o_w, &o_off_buf)
    }

    /// GDN qkv+gate device-resident chain (Q4_K, 단일 command buffer 2 GEMV).
    /// norm_input(host)은 norm 완료된 것. qkv/gate weight raw 는 mmap(loader 소유)
    /// resident NoCopy. carrier shape 별 재사용. 반환: (qkv, gate).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_inproj_chain_resident(
        &self,
        norm_input: &[f32],
        qkv_raw: &[u8],
        gate_raw: &[u8],
        hidden_dim: usize,
        qkv_dim: usize,
        gate_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let (qkv_w, qkv_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(qkv_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, qkv_raw));
            (e.0.clone(), e.1)
        };
        let (gate_w, gate_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(gate_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, gate_raw));
            (e.0.clone(), e.1)
        };
        let qkv_off_buf = ffn_chain::u32_buf(ctx, qkv_off);
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);

        let mut carriers = self.gdn_inproj_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, qkv_dim, gate_dim))
            .or_insert_with(|| {
                ffn_chain::GdnInprojCarrier::new(ctx, hidden_dim, qkv_dim, gate_dim)
            });

        ffn_chain::gdn_inproj_chain_dispatch(
            ctx,
            carrier,
            norm_input,
            &qkv_w,
            &qkv_off_buf,
            &gate_w,
            &gate_off_buf,
        )
    }

    /// QKV projection device-resident chain (Q4_K q/k/v, 단일 command buffer 3 GEMV).
    /// norm_input(host)은 chain 진입 전 norm 완료된 것. q/k/v weight raw 는 mmap 포인터
    /// (loader 소유) — resident NoCopy 캐시 재사용. carrier 는 shape 별 재사용.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_qkv_chain_resident(
        &self,
        norm_input: &[f32],
        q_raw: &[u8],
        k_raw: &[u8],
        v_raw: &[u8],
        hidden_dim: usize,
        q_out_dim: usize,
        kv_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        // q/k/v weight NoCopy resident 캐시 (gemv_q4k_resident 와 동일 키).
        let (q_w, q_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(q_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, q_raw));
            (e.0.clone(), e.1)
        };
        let (k_w, k_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(k_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, k_raw));
            (e.0.clone(), e.1)
        };
        let (v_w, v_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(v_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, v_raw));
            (e.0.clone(), e.1)
        };
        let q_off_buf = ffn_chain::u32_buf(ctx, q_off);
        let k_off_buf = ffn_chain::u32_buf(ctx, k_off);
        let v_off_buf = ffn_chain::u32_buf(ctx, v_off);

        let mut carriers = self.qkv_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, q_out_dim, kv_dim))
            .or_insert_with(|| ffn_chain::QkvCarrier::new(ctx, hidden_dim, q_out_dim, kv_dim));

        ffn_chain::qkv_chain_dispatch(
            ctx, carrier, norm_input, &q_w, &q_off_buf, &k_w, &k_off_buf, &v_w, &v_off_buf,
        )
    }

    /// FFN device-resident chain (Q4_K gate/up + Q4_K|Q6_K down + RMSNorm/SiLU/residual).
    /// weight raw 는 mmap 포인터(loader 소유) — resident NoCopy 캐시 재사용.
    /// norm weight 는 작아서 복사 업로드. carrier 는 (hidden_dim, ffn_dim) 별 재사용.
    /// `down_is_q6k`: down weight 가 Q6_K(block 210B) 면 true, Q4_K(144B) 면 false.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_q4k_resident(
        &self,
        hidden: &[f32],
        norm_weight: &[f32],
        gate_raw: &[u8],
        up_raw: &[u8],
        down_raw: &[u8],
        hidden_dim: usize,
        ffn_dim: usize,
        norm_eps: f32,
        down_is_q6k: bool,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        // weight NoCopy resident 캐시(gate/up/down) — gemv_q4k_resident 와 동일 키.
        let (gate_w, gate_off) = {
            let mut resident = self.resident.borrow_mut();
            let e = resident
                .entry(resident_key(gate_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, gate_raw));
            (e.0.clone(), e.1)
        };
        let (up_w, up_off) = {
            let mut resident = self.resident.borrow_mut();
            let e = resident
                .entry(resident_key(up_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, up_raw));
            (e.0.clone(), e.1)
        };
        let (down_w, down_off) = {
            let mut resident = self.resident.borrow_mut();
            let e = resident
                .entry(resident_key(down_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, down_raw));
            (e.0.clone(), e.1)
        };

        // norm weight: 작아서(hidden_dim f32) 매 호출 복사 업로드.
        let norm_w_buf = ffn_chain::shared_f32_buf(ctx, norm_weight);

        // scalar offset buffer(작음 — 매 호출 생성).
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);
        let up_off_buf = ffn_chain::u32_buf(ctx, up_off);
        let down_off_buf = ffn_chain::u32_buf(ctx, down_off);

        // carrier 재사용.
        let mut carriers = self.ffn_carriers.borrow_mut();
        let carrier = carriers
            .entry((hidden_dim, ffn_dim))
            .or_insert_with(|| ffn_chain::FfnCarrier::new(ctx, hidden_dim, ffn_dim, norm_eps));

        ffn_chain::ffn_chain_dispatch(
            ctx,
            carrier,
            hidden,
            &norm_w_buf,
            &gate_w,
            &gate_off_buf,
            &up_w,
            &up_off_buf,
            &down_w,
            &down_off_buf,
            down_is_q6k,
        )
    }

    /// Attention layer 전체 device-resident chain (표준 Qwen3: norm→q/k/v GEMV→
    /// q/k norm→rope→kv_append→attn→o→residual, 단일 command buffer). q/k/v/o weight
    /// raw 는 mmap 포인터(loader 소유) resident NoCopy. norm/q_norm/k_norm 은 작아서
    /// 복사 업로드. carrier 는 layer 별 재사용(KV state 분리). prior_k/prior_v 는 이전
    /// 토큰들(host f16 bits, [pos*kv_dim]) — KV_dev[0..pos] 동기화용. 반환: residual hidden.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_layer_resident(
        &self,
        layer: usize,
        hidden: &[f32],
        norm_weight: &[f32],
        q_raw: &[u8],
        k_raw: &[u8],
        v_raw: &[u8],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        o_raw: &[u8],
        v_is_q6k: bool,
        ffn_norm_weight: &[f32],
        ffn_gate_raw: &[u8],
        ffn_up_raw: &[u8],
        ffn_down_raw: &[u8],
        ffn_down_is_q6k: bool,
        prior_k: &[u16],
        prior_v: &[u16],
        pos: usize,
        hidden_dim: usize,
        q_dim: usize,
        q_out_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        n_rot: usize,
        capacity: usize,
        ffn_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Vec<f32> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        // q/k/v/o weight NoCopy resident 캐시 (gemv_q4k_resident 와 동일 키=raw ptr).
        let (q_w, q_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(q_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, q_raw));
            (e.0.clone(), e.1)
        };
        let (k_w, k_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(k_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, k_raw));
            (e.0.clone(), e.1)
        };
        let (v_w, v_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(v_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, v_raw));
            (e.0.clone(), e.1)
        };
        let (o_w, o_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(o_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, o_raw));
            (e.0.clone(), e.1)
        };
        // FFN gate/up/down weight NoCopy resident 캐시.
        let (ffn_gate_w, ffn_gate_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(ffn_gate_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, ffn_gate_raw));
            (e.0.clone(), e.1)
        };
        let (ffn_up_w, ffn_up_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(ffn_up_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, ffn_up_raw));
            (e.0.clone(), e.1)
        };
        let (ffn_down_w, ffn_down_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(ffn_down_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, ffn_down_raw));
            (e.0.clone(), e.1)
        };
        let q_off_buf = ffn_chain::u32_buf(ctx, q_off);
        let k_off_buf = ffn_chain::u32_buf(ctx, k_off);
        let v_off_buf = ffn_chain::u32_buf(ctx, v_off);
        let o_off_buf = ffn_chain::u32_buf(ctx, o_off);
        let ffn_gate_off_buf = ffn_chain::u32_buf(ctx, ffn_gate_off);
        let ffn_up_off_buf = ffn_chain::u32_buf(ctx, ffn_up_off);
        let ffn_down_off_buf = ffn_chain::u32_buf(ctx, ffn_down_off);

        // norm/q_norm/k_norm/ffn_norm weight: 작아서 복사 업로드.
        let norm_w_buf = ffn_chain::shared_f32_buf(ctx, norm_weight);
        let q_norm_w_buf = ffn_chain::shared_f32_buf(ctx, q_norm_weight);
        let k_norm_w_buf = ffn_chain::shared_f32_buf(ctx, k_norm_weight);
        let ffn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ffn_norm_weight);

        // carrier 는 layer 별 재사용 (KV state 분리).
        let mut carriers = self.attn_carriers.borrow_mut();
        let carrier = carriers.entry(layer).or_insert_with(|| {
            attn_chain::AttnCarrier::new(
                ctx,
                hidden_dim,
                q_dim,
                q_out_dim,
                kv_dim,
                head_dim,
                num_heads,
                num_kv_heads,
                n_rot,
                capacity,
                ffn_dim,
                eps,
                theta,
                scale,
            )
        });

        attn_chain::attn_chain_dispatch(
            ctx,
            carrier,
            hidden,
            &norm_w_buf,
            &q_w,
            &q_off_buf,
            &k_w,
            &k_off_buf,
            &v_w,
            &v_off_buf,
            &q_norm_w_buf,
            &k_norm_w_buf,
            &o_w,
            &o_off_buf,
            v_is_q6k,
            &ffn_norm_w_buf,
            &ffn_gate_w,
            &ffn_gate_off_buf,
            &ffn_up_w,
            &ffn_up_off_buf,
            &ffn_down_w,
            &ffn_down_off_buf,
            ffn_down_is_q6k,
            prior_k,
            prior_v,
            pos,
        )
    }

    /// GDN layer 전체를 단일 command buffer device chain 으로 실행. qkv/gate/alpha/beta/
    /// ssm_out/ffn weight 는 Q4_K(ffn_down 만 Q6_K 가능), norm 류는 f32. conv_state·
    /// delta_state 는 dispatch 경계 host↔device sync. 반환 `(hidden, conv_state, delta_state)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_resident(
        &self,
        layer: usize,
        hidden: &[f32],
        conv_state: &[f32],
        delta_state: &[f32],
        attn_norm_weight: &[f32],
        qkv_raw: &[u8],
        gate_raw: &[u8],
        alpha_raw: &[u8],
        beta_raw: &[u8],
        dt_bias_weight: &[f32],
        ssm_a_weight: &[f32],
        conv1d_weight: &[f32],
        ssm_norm_weight: &[f32],
        ssm_out_raw: &[u8],
        ffn_norm_weight: &[f32],
        ffn_gate_raw: &[u8],
        ffn_up_raw: &[u8],
        ffn_down_raw: &[u8],
        qkv_q: u8,
        gate_q: u8,
        alpha_q: u8,
        beta_q: u8,
        ssm_out_q: u8,
        ffn_gate_q: u8,
        ffn_up_q: u8,
        ffn_down_q: u8,
        hidden_dim: usize,
        conv_channels: usize,
        conv_kernel: usize,
        z_dim: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        ffn_dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        // Q4_K GEMV weight NoCopy resident 캐시(raw `(ptr,len)` 키).
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (qkv_w, qkv_off) = wrap(qkv_raw);
        let (gate_w, gate_off) = wrap(gate_raw);
        let (alpha_w, alpha_off) = wrap(alpha_raw);
        let (beta_w, beta_off) = wrap(beta_raw);
        let (ssm_out_w, ssm_out_off) = wrap(ssm_out_raw);
        let (ffn_gate_w, ffn_gate_off) = wrap(ffn_gate_raw);
        let (ffn_up_w, ffn_up_off) = wrap(ffn_up_raw);
        let (ffn_down_w, ffn_down_off) = wrap(ffn_down_raw);

        let qkv_off_buf = ffn_chain::u32_buf(ctx, qkv_off);
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);
        let alpha_off_buf = ffn_chain::u32_buf(ctx, alpha_off);
        let beta_off_buf = ffn_chain::u32_buf(ctx, beta_off);
        let ssm_out_off_buf = ffn_chain::u32_buf(ctx, ssm_out_off);
        let ffn_gate_off_buf = ffn_chain::u32_buf(ctx, ffn_gate_off);
        let ffn_up_off_buf = ffn_chain::u32_buf(ctx, ffn_up_off);
        let ffn_down_off_buf = ffn_chain::u32_buf(ctx, ffn_down_off);

        // f32 weight: 작아서 복사 업로드.
        let attn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, attn_norm_weight);
        let dt_bias_w_buf = ffn_chain::shared_f32_buf(ctx, dt_bias_weight);
        let ssm_a_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_a_weight);
        let conv1d_w_buf = ffn_chain::shared_f32_buf(ctx, conv1d_weight);
        let ssm_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_norm_weight);
        let ffn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ffn_norm_weight);

        let mut carriers = self.gdn_carriers.borrow_mut();
        let carrier = carriers.entry(layer).or_insert_with(|| {
            gdn_chain::GdnCarrier::new(
                ctx,
                hidden_dim,
                conv_channels,
                conv_kernel,
                z_dim,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                ffn_dim,
                eps,
            )
        });

        gdn_chain::gdn_chain_dispatch(
            ctx,
            carrier,
            hidden,
            conv_state,
            delta_state,
            &attn_norm_w_buf,
            &qkv_w,
            &qkv_off_buf,
            &gate_w,
            &gate_off_buf,
            &alpha_w,
            &alpha_off_buf,
            &beta_w,
            &beta_off_buf,
            &dt_bias_w_buf,
            &ssm_a_w_buf,
            &conv1d_w_buf,
            &ssm_norm_w_buf,
            &ssm_out_w,
            &ssm_out_off_buf,
            &ffn_norm_w_buf,
            &ffn_gate_w,
            &ffn_gate_off_buf,
            &ffn_up_w,
            &ffn_up_off_buf,
            &ffn_down_w,
            &ffn_down_off_buf,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
            ffn_gate_q,
            ffn_up_q,
            ffn_down_q,
            false,
        )
    }

    /// MoE GDN layer의 FFN 앞 core만 단일 command buffer device chain으로 실행한다.
    /// FFN routing/experts는 caller의 기존 경로가 그대로 이어서 처리한다.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_core_resident(
        &self,
        layer: usize,
        hidden: &[f32],
        conv_state: &[f32],
        delta_state: &[f32],
        attn_norm_weight: &[f32],
        qkv_raw: &[u8],
        gate_raw: &[u8],
        alpha_raw: &[u8],
        beta_raw: &[u8],
        dt_bias_weight: &[f32],
        ssm_a_weight: &[f32],
        conv1d_weight: &[f32],
        ssm_norm_weight: &[f32],
        ssm_out_raw: &[u8],
        qkv_q: u8,
        gate_q: u8,
        alpha_q: u8,
        beta_q: u8,
        ssm_out_q: u8,
        hidden_dim: usize,
        conv_channels: usize,
        conv_kernel: usize,
        z_dim: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let (qkv_w, qkv_off) = wrap(qkv_raw);
        let (gate_w, gate_off) = wrap(gate_raw);
        let (alpha_w, alpha_off) = wrap(alpha_raw);
        let (beta_w, beta_off) = wrap(beta_raw);
        let (ssm_out_w, ssm_out_off) = wrap(ssm_out_raw);

        let qkv_off_buf = ffn_chain::u32_buf(ctx, qkv_off);
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);
        let alpha_off_buf = ffn_chain::u32_buf(ctx, alpha_off);
        let beta_off_buf = ffn_chain::u32_buf(ctx, beta_off);
        let ssm_out_off_buf = ffn_chain::u32_buf(ctx, ssm_out_off);

        let attn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, attn_norm_weight);
        let dt_bias_w_buf = ffn_chain::shared_f32_buf(ctx, dt_bias_weight);
        let ssm_a_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_a_weight);
        let conv1d_w_buf = ffn_chain::shared_f32_buf(ctx, conv1d_weight);
        let ssm_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_norm_weight);

        let mut carriers = self.gdn_core_carriers.borrow_mut();
        let carrier = carriers.entry(layer).or_insert_with(|| {
            gdn_chain::GdnCarrier::new(
                ctx,
                hidden_dim,
                conv_channels,
                conv_kernel,
                z_dim,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                1,
                eps,
            )
        });

        gdn_chain::gdn_core_chain_dispatch(
            ctx,
            carrier,
            hidden,
            conv_state,
            delta_state,
            &attn_norm_w_buf,
            &qkv_w,
            &qkv_off_buf,
            &gate_w,
            &gate_off_buf,
            &alpha_w,
            &alpha_off_buf,
            &beta_w,
            &beta_off_buf,
            &dt_bias_w_buf,
            &ssm_a_w_buf,
            &conv1d_w_buf,
            &ssm_norm_w_buf,
            &ssm_out_w,
            &ssm_out_off_buf,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
            false,
        )
    }

    /// MoE GDN layer의 core와 Qwen35MoE FFN tail을 같은 command buffer에서 실행한다.
    /// Expert weights는 raw quant NoCopy buffer를 그대로 쓰고, route/top-k/shared gate도
    /// device 안에서 계산한다.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_moe_layer_resident(
        &self,
        layer: usize,
        hidden: &[f32],
        conv_state: &[f32],
        delta_state: &[f32],
        attn_norm_weight: &[f32],
        qkv_raw: &[u8],
        gate_raw: &[u8],
        alpha_raw: &[u8],
        beta_raw: &[u8],
        dt_bias_weight: &[f32],
        ssm_a_weight: &[f32],
        conv1d_weight: &[f32],
        ssm_norm_weight: &[f32],
        ssm_out_raw: &[u8],
        ffn_norm_weight: &[f32],
        router_w: &[f32],
        gate_exps: &[u8],
        gate_expert_bytes: usize,
        up_exps: &[u8],
        up_expert_bytes: usize,
        down_exps: &[u8],
        down_expert_bytes: usize,
        shared_input_scale: &[f32],
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        qkv_q: u8,
        gate_q: u8,
        alpha_q: u8,
        beta_q: u8,
        ssm_out_q: u8,
        down_quant: u8,
        hidden_dim: usize,
        conv_channels: usize,
        conv_kernel: usize,
        z_dim: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        n_ff: usize,
        n_expert: usize,
        n_expert_used: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let router_bytes = unsafe {
            std::slice::from_raw_parts(
                router_w.as_ptr() as *const u8,
                std::mem::size_of_val(router_w),
            )
        };

        let (qkv_w, qkv_off) = wrap(qkv_raw);
        let (gate_w, gate_off) = wrap(gate_raw);
        let (alpha_w, alpha_off) = wrap(alpha_raw);
        let (beta_w, beta_off) = wrap(beta_raw);
        let (ssm_out_w, ssm_out_off) = wrap(ssm_out_raw);
        let (router_wb, router_off) = wrap(router_bytes);
        let (gate_exps_w, gate_exps_off) = wrap(gate_exps);
        let (up_exps_w, up_exps_off) = wrap(up_exps);
        let (down_exps_w, down_exps_off) = wrap(down_exps);
        let (shared_gate_w, shared_gate_off) = wrap(shared_gate);
        let (shared_up_w, shared_up_off) = wrap(shared_up);
        let (shared_down_w, shared_down_off) = wrap(shared_down);

        let qkv_off_buf = ffn_chain::u32_buf(ctx, qkv_off);
        let gate_off_buf = ffn_chain::u32_buf(ctx, gate_off);
        let alpha_off_buf = ffn_chain::u32_buf(ctx, alpha_off);
        let beta_off_buf = ffn_chain::u32_buf(ctx, beta_off);
        let ssm_out_off_buf = ffn_chain::u32_buf(ctx, ssm_out_off);
        let router_off_buf = ffn_chain::u32_buf(ctx, router_off);

        let attn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, attn_norm_weight);
        let dt_bias_w_buf = ffn_chain::shared_f32_buf(ctx, dt_bias_weight);
        let ssm_a_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_a_weight);
        let conv1d_w_buf = ffn_chain::shared_f32_buf(ctx, conv1d_weight);
        let ssm_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ssm_norm_weight);
        let ffn_norm_w_buf = ffn_chain::shared_f32_buf(ctx, ffn_norm_weight);
        let shared_input_scale_buf = ffn_chain::shared_f32_buf(ctx, shared_input_scale);

        let mut gdn_carriers = self.gdn_core_carriers.borrow_mut();
        let carrier = gdn_carriers.entry(layer).or_insert_with(|| {
            gdn_chain::GdnCarrier::new(
                ctx,
                hidden_dim,
                conv_channels,
                conv_kernel,
                z_dim,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                1,
                eps,
            )
        });
        let mut moe_carriers = self.qwen_moe_decode_chain_carriers.borrow_mut();
        let moe_carrier = moe_carriers
            .entry(qwen_moe_decode_chain_carrier_key(
                layer,
                n_expert_used,
                hidden_dim,
                n_ff,
                n_expert,
                down_quant,
            ))
            .or_insert_with(|| {
                ffn_chain::QwenMoeDecodeChainCarrier::new(
                    ctx,
                    n_expert_used,
                    hidden_dim,
                    n_ff,
                    n_expert,
                    gate_expert_bytes,
                    up_expert_bytes,
                    down_expert_bytes,
                    n_expert as u32,
                    eps,
                )
            });

        let result = gdn_chain::gdn_moe_chain_dispatch(
            ctx,
            carrier,
            moe_carrier,
            hidden,
            conv_state,
            delta_state,
            &attn_norm_w_buf,
            &qkv_w,
            &qkv_off_buf,
            &gate_w,
            &gate_off_buf,
            &alpha_w,
            &alpha_off_buf,
            &beta_w,
            &beta_off_buf,
            &dt_bias_w_buf,
            &ssm_a_w_buf,
            &conv1d_w_buf,
            &ssm_norm_w_buf,
            &ssm_out_w,
            &ssm_out_off_buf,
            &ffn_norm_w_buf,
            &router_wb,
            &router_off_buf,
            &shared_input_scale_buf,
            &gate_exps_w,
            gate_exps_off,
            &up_exps_w,
            up_exps_off,
            &down_exps_w,
            down_exps_off,
            &shared_gate_w,
            shared_gate_off,
            &shared_up_w,
            shared_up_off,
            &shared_down_w,
            shared_down_off,
            gate_exps,
            up_exps,
            down_exps,
            shared_gate,
            shared_up,
            shared_down,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
            down_quant,
        );
        if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_ROUTE_TRACE").is_some() {
            let (expert_ids, route_weights) = moe_carrier.debug_routes();
            eprintln!(
                "[metal-gdn-moe-route] layer={} experts={:?} weights={:?}",
                layer, expert_ids, route_weights
            );
            if layer == 34 {
                if let Some(&expert_id) = expert_ids.first() {
                    if expert_id < n_expert as u32 {
                        let normed = moe_carrier.debug_normed();
                        let expert = expert_id as usize;
                        if let Some(gate_off) = (gate_exps_off as usize)
                            .checked_add(expert.saturating_mul(gate_expert_bytes))
                            .and_then(|off| u32::try_from(off).ok())
                        {
                            let gate_ref = compute::gemv_q4k_dispatch(
                                ctx,
                                &gate_exps_w,
                                gate_off,
                                &normed,
                                n_ff,
                                hidden_dim,
                            );
                            let gate_expert_slice = gate_exps
                                .get(expert.saturating_mul(gate_expert_bytes)..)
                                .and_then(|tail| tail.get(..gate_expert_bytes));
                            let gate_copy_ref = gate_expert_slice.map(|expert_bytes| {
                                compute::gemv_q4k_with_ctx(
                                    ctx,
                                    expert_bytes,
                                    &normed,
                                    n_ff,
                                    hidden_dim,
                                )
                            });
                            let gate_slice_nocopy_ref = gate_expert_slice.map(|expert_bytes| {
                                let (slice_w, slice_off) = compute::wrap_nocopy(ctx, expert_bytes);
                                let values = compute::gemv_q4k_dispatch(
                                    ctx, &slice_w, slice_off, &normed, n_ff, hidden_dim,
                                );
                                (slice_off, values)
                            });
                            let gate_bpr = gate_expert_bytes / n_ff.max(1);
                            for row in 0..4.min(n_ff) {
                                let row_off = expert
                                    .saturating_mul(gate_expert_bytes)
                                    .saturating_add(row.saturating_mul(gate_bpr));
                                if row_off + 4 <= gate_exps.len() {
                                    let d = half::f16::from_bits(u16::from_le_bytes([
                                        gate_exps[row_off],
                                        gate_exps[row_off + 1],
                                    ]))
                                    .to_f32();
                                    let dmin = half::f16::from_bits(u16::from_le_bytes([
                                        gate_exps[row_off + 2],
                                        gate_exps[row_off + 3],
                                    ]))
                                    .to_f32();
                                    eprintln!(
                                        "[metal-gdn-moe-q4k-host-row] layer={} expert={} row={} row_off={} gate_bpr={} d={:.6e} dmin={:.6e} out={:?}",
                                        layer,
                                        expert_id,
                                        row,
                                        row_off,
                                        gate_bpr,
                                        d,
                                        dmin,
                                        gate_ref.get(row).copied()
                                    );
                                    let mut bad = 0usize;
                                    let mut first_bad = None;
                                    let mut max_d = 0.0f32;
                                    let mut max_dmin = 0.0f32;
                                    for block in 0..(gate_bpr / 144) {
                                        let block_off = row_off + block * 144;
                                        if block_off + 4 > gate_exps.len() {
                                            break;
                                        }
                                        let d = half::f16::from_bits(u16::from_le_bytes([
                                            gate_exps[block_off],
                                            gate_exps[block_off + 1],
                                        ]))
                                        .to_f32();
                                        let dmin = half::f16::from_bits(u16::from_le_bytes([
                                            gate_exps[block_off + 2],
                                            gate_exps[block_off + 3],
                                        ]))
                                        .to_f32();
                                        if d.is_finite() && dmin.is_finite() {
                                            max_d = max_d.max(d.abs());
                                            max_dmin = max_dmin.max(dmin.abs());
                                        } else {
                                            bad += 1;
                                            first_bad.get_or_insert(block);
                                        }
                                    }
                                    eprintln!(
                                        "[metal-gdn-moe-q4k-host-row-scan] layer={} expert={} row={} blocks={} bad={} first_bad={:?} max_d={:.6e} max_dmin={:.6e}",
                                        layer,
                                        expert_id,
                                        row,
                                        gate_bpr / 144,
                                        bad,
                                        first_bad,
                                        max_d,
                                        max_dmin
                                    );
                                }
                            }
                            let mut nonfinite = 0usize;
                            let mut max_abs = 0.0f32;
                            for &value in &gate_ref {
                                if value.is_finite() {
                                    max_abs = max_abs.max(value.abs());
                                } else {
                                    nonfinite += 1;
                                }
                            }
                            let mut head = [0.0f32; 4];
                            for (dst, &value) in head.iter_mut().zip(gate_ref.iter()) {
                                *dst = value;
                            }
                            let copy_head = gate_copy_ref.as_ref().map(|values| {
                                let mut head = [0.0f32; 4];
                                for (dst, &value) in head.iter_mut().zip(values.iter()) {
                                    *dst = value;
                                }
                                head
                            });
                            let copy_nonfinite = gate_copy_ref.as_ref().map(|values| {
                                values.iter().filter(|value| !value.is_finite()).count()
                            });
                            let slice_nocopy_head =
                                gate_slice_nocopy_ref.as_ref().map(|(_, values)| {
                                    let mut head = [0.0f32; 4];
                                    for (dst, &value) in head.iter_mut().zip(values.iter()) {
                                        *dst = value;
                                    }
                                    head
                                });
                            let slice_nocopy_nonfinite =
                                gate_slice_nocopy_ref.as_ref().map(|(_, values)| {
                                    values.iter().filter(|value| !value.is_finite()).count()
                                });
                            let slice_nocopy_off =
                                gate_slice_nocopy_ref.as_ref().map(|(off, _)| *off);
                            eprintln!(
                                "[metal-gdn-moe-q4k-ref] layer={} expert={} gate_off={} nonfinite={} max_abs={:.6e} head={:?} copy_nonfinite={:?} copy_head={:?} slice_nocopy_off={:?} slice_nocopy_nonfinite={:?} slice_nocopy_head={:?}",
                                layer,
                                expert_id,
                                gate_off,
                                nonfinite,
                                max_abs,
                                head,
                                copy_nonfinite,
                                copy_head,
                                slice_nocopy_off,
                                slice_nocopy_nonfinite,
                                slice_nocopy_head
                            );
                        }
                    }
                }
            }
            for (name, nonfinite, max_abs, head) in moe_carrier.debug_buffer_stats() {
                eprintln!(
                    "[metal-gdn-moe-stats] layer={} buf={} nonfinite={} max_abs={:.6e} head={:?}",
                    layer, name, nonfinite, max_abs, head
                );
            }
            let mut hidden_nonfinite = 0usize;
            let mut hidden_max_abs = 0.0f32;
            for &value in &result.0 {
                if value.is_finite() {
                    hidden_max_abs = hidden_max_abs.max(value.abs());
                } else {
                    hidden_nonfinite += 1;
                }
            }
            let mut hidden_head = [0.0f32; 4];
            for (dst, &value) in hidden_head.iter_mut().zip(result.0.iter()) {
                *dst = value;
            }
            eprintln!(
                "[metal-gdn-moe-stats] layer={} buf=hidden nonfinite={} max_abs={:.6e} head={:?}",
                layer, hidden_nonfinite, hidden_max_abs, hidden_head
            );
        }
        result
    }

    /// 연속된 attn carrier + GDN carrier layer N개를 **단일 command buffer** 로 실행한다
    /// (M5-B 2단계, 1단계 GDN-only chain 을 일반화). specs 가 attn/gdn 혼합 enum 이라
    /// 9B 의 32 layer(attn 8 + gdn 24, 교대)를 전부 하나의 command buffer 로 묶는다.
    /// hidden 은 공유 device buffer 라 attn↔gdn 경계도 host 안 거치고 device 직결(같은
    /// command buffer 안 자동 hazard 로 순서 보장).
    ///
    /// - `hidden`: specs[0] 입력 hidden(in/out). chain 종료 후 마지막 layer 출력으로 덮어쓴다.
    /// - `specs`: 실행할 attn/gdn layer 들(연속). 각 spec 이 자기 weight/quant/shape/입력 state 보유.
    /// - `out_states`: layer 별 state. GDN 은 `Some((conv_new, delta_new))`, attn 은 `None`
    ///   (KV 가 device 소유라 readback 불필요 — carrier 가 incremental append 로 누적).
    ///
    /// attn KV 동기화(2.1 dispatch 패턴): 첫 token(`filled == 0`)만 encode 전 `ensure_filled`
    /// 로 prior KV(host f16) device init, commit 후 `filled = pos+1` 갱신. GDN state 는
    /// chain 양끝에서만 host↔device sync(진입 upload, commit 후 readback).
    #[cfg(target_os = "macos")]
    pub fn decode_chain_run(
        &self,
        hidden: &mut [f32],
        specs: &[ChainLayerSpecRef<'_>],
        out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
        options: DecodeChainOptions,
        output_argmax: Option<DecodeOutputArgmaxSpecRef<'_>>,
    ) -> DecodeChainReport {
        if specs.is_empty() {
            return DecodeChainReport {
                did_run: true,
                ..DecodeChainReport::default()
            };
        }
        assert_eq!(
            specs.len(),
            out_states.len(),
            "decode_chain_run: specs/out_states length mismatch"
        );
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");

        // ── RNB_METAL_CHAIN_GPU_TIME=1 계측 ──────────────────────────────────
        // 함수 전체 wall(fn_ms) + 구간별(encode/gpu/commit_wait/hidden·state readback)
        // 분해. pm30: pm29 가 chain-wall − commit_wait 로 역산한 encode ~31ms 가 실은
        // dispatch 인코딩 5.9ms 뿐임을 직접 측정해 정정. 미지 ~28ms 가 함수 안(GDN
        // state readback)인지 밖(decode loop)인지 가른다.
        static CHAIN_TOKEN_IDX: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let chain_gpu_time = options.collect_timing;
        // pm31: GDN delta state device residency(default ON). delta 는 첫 토큰만 device 로
        // seed 하고 이후 upload/readback skip(delta_net_step in-place 누적) → 매 토큰 9ms
        // readback 의 96.7%(delta) 제거. conv(3.3%)는 CUDA 와 동일하게 비-resident 유지.
        // opt-out: RNB_METAL_DELTA_RESIDENT=0(매 토큰 host 왕복, ABAB 측정 기준선용).
        let delta_resident = options.delta_resident;
        let token_idx = if chain_gpu_time {
            CHAIN_TOKEN_IDX.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        } else {
            0
        };
        let t_fn_start = if chain_gpu_time {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 공유 hidden buffer — specs[0] 입력 1회 upload, layer 들이 같은 buffer 를 in/out 으로
        // 누적(residual_add in-place), 마지막에 readback. attn↔gdn 모두 이 buffer 공유.
        let hidden_dim = hidden.len();
        let mut report = DecodeChainReport {
            did_run: true,
            cmd_buffers: 1,
            upload_bytes: hidden_dim * std::mem::size_of::<f32>(),
            ..DecodeChainReport::default()
        };
        let shared_hidden = {
            let mut cache = self.decode_chain_hidden.borrow_mut();
            cache
                .entry(hidden_dim)
                .or_insert_with(|| ffn_chain::empty_f32_buf(ctx, hidden_dim))
                .clone()
        };
        gdn_chain::upload_shared(&shared_hidden, hidden);
        let constant_u32 = |value| {
            let mut cache = self.constant_u32.borrow_mut();
            cache
                .entry(value)
                .or_insert_with(|| ffn_chain::u32_buf(ctx, value))
                .clone()
        };
        let constant_f32 = |data: &[f32]| {
            let key = (data.as_ptr() as usize, data.len());
            let mut cache = self.constant_f32.borrow_mut();
            cache
                .entry(key)
                .or_insert_with(|| ffn_chain::shared_f32_buf(ctx, data))
                .clone()
        };

        // 양자화 GEMV weight NoCopy resident wrap. self.resident 를 borrow_mut→drop 하므로
        // closure 호출마다 borrow scope 안전(attn/gdn 공통).
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };

        // encode(dispatch 인코딩) 시작 — command buffer 생성 ~ endEncoding.
        let t_encode_start = if chain_gpu_time {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = compute::chain_compute_encoder(ctx, &cmd);

        // attn layer 의 (layer_idx, kv_len) — commit 후 filled 갱신용. encode 시점에 기록.
        let mut attn_filled_updates: Vec<(usize, usize)> = Vec::new();
        let mut attn_moe_filled_updates: Vec<(usize, usize)> = Vec::new();

        for spec in specs {
            match spec {
                ChainLayerSpecRef::Attn(s) => {
                    let (q_w, q_off) = wrap(s.q_raw);
                    let (k_w, k_off) = wrap(s.k_raw);
                    let (v_w, v_off) = wrap(s.v_raw);
                    let (o_w, o_off) = wrap(s.o_raw);
                    let (ffn_gate_w, ffn_gate_off) = wrap(s.ffn_gate_raw);
                    let (ffn_up_w, ffn_up_off) = wrap(s.ffn_up_raw);
                    let (ffn_down_w, ffn_down_off) = wrap(s.ffn_down_raw);

                    let q_off_buf = constant_u32(q_off);
                    let k_off_buf = constant_u32(k_off);
                    let v_off_buf = constant_u32(v_off);
                    let o_off_buf = constant_u32(o_off);
                    let ffn_gate_off_buf = constant_u32(ffn_gate_off);
                    let ffn_up_off_buf = constant_u32(ffn_up_off);
                    let ffn_down_off_buf = constant_u32(ffn_down_off);

                    let norm_w_buf = constant_f32(s.norm_weight);
                    let q_norm_w_buf = constant_f32(s.q_norm_weight);
                    let k_norm_w_buf = constant_f32(s.k_norm_weight);
                    let ffn_norm_w_buf = constant_f32(s.ffn_norm_weight);

                    let mut carriers = self.attn_carriers.borrow_mut();
                    let carrier = carriers.entry(s.layer).or_insert_with(|| {
                        attn_chain::AttnCarrier::new(
                            ctx,
                            s.hidden_dim,
                            s.q_dim,
                            s.q_out_dim,
                            s.kv_dim,
                            s.head_dim,
                            s.num_heads,
                            s.num_kv_heads,
                            s.n_rot,
                            s.capacity,
                            s.ffn_dim,
                            s.eps,
                            s.theta,
                            s.scale,
                        )
                    });
                    // 첫 token(filled==0)만 prefill KV(host f16, 0..pos)를 device 로 1회 init.
                    // 이후 토큰은 chain 안 kv_append 로 device 누적(host roundtrip 없음).
                    carrier.kv_ensure_filled(s.prior_k, s.prior_v, s.pos);
                    attn_filled_updates.push((s.layer, s.pos + 1));

                    attn_chain::attn_chain_encode(
                        ctx,
                        &enc,
                        carrier,
                        &shared_hidden,
                        &norm_w_buf,
                        &q_w,
                        &q_off_buf,
                        &k_w,
                        &k_off_buf,
                        &v_w,
                        &v_off_buf,
                        &q_norm_w_buf,
                        &k_norm_w_buf,
                        &o_w,
                        &o_off_buf,
                        s.v_is_q6k,
                        &ffn_norm_w_buf,
                        &ffn_gate_w,
                        &ffn_gate_off_buf,
                        &ffn_up_w,
                        &ffn_up_off_buf,
                        &ffn_down_w,
                        &ffn_down_off_buf,
                        s.ffn_down_is_q6k,
                        s.pos,
                    );
                }
                ChainLayerSpecRef::AttnMoeQwen(s) => {
                    let router_bytes = unsafe {
                        std::slice::from_raw_parts(
                            s.router_w.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(s.router_w),
                        )
                    };
                    let (q_w, q_off) = wrap(s.q_raw);
                    let (k_w, k_off) = wrap(s.k_raw);
                    let (v_w, v_off) = wrap(s.v_raw);
                    let (o_w, o_off) = wrap(s.o_raw);
                    let (router_w, router_off) = wrap(router_bytes);
                    let (gate_exps_w, gate_exps_off) = wrap(s.gate_exps_raw);
                    let (up_exps_w, up_exps_off) = wrap(s.up_exps_raw);
                    let (down_exps_w, down_exps_off) = wrap(s.down_exps_raw);
                    let (shared_gate_w, shared_gate_off) = wrap(s.shared_gate_raw);
                    let (shared_up_w, shared_up_off) = wrap(s.shared_up_raw);
                    let (shared_down_w, shared_down_off) = wrap(s.shared_down_raw);

                    let q_off_buf = constant_u32(q_off);
                    let k_off_buf = constant_u32(k_off);
                    let v_off_buf = constant_u32(v_off);
                    let o_off_buf = constant_u32(o_off);
                    let router_off_buf = constant_u32(router_off);

                    let norm_w_buf = constant_f32(s.norm_weight);
                    let q_norm_w_buf = constant_f32(s.q_norm_weight);
                    let k_norm_w_buf = constant_f32(s.k_norm_weight);
                    let ffn_norm_w_buf = constant_f32(s.ffn_norm_weight);
                    let shared_input_scale_buf = constant_f32(s.shared_input_scale);

                    // KVarn chain attention: host KVarn view 로 device resident 를
                    // 업데이트하고 params(kv_len/tail_len +1)/append slot 을 만든다.
                    // 새 토큰은 chain 안에서 device tail-append, attention 은 in-chain
                    // kvarn 커널로 실행(per-op commit/wait 제거). host = source of truth.
                    let kvarn_dummy_query;
                    let kvarn_state: Option<(
                        Retained<ProtocolObject<dyn MTLBuffer>>,
                        u32,
                        usize,
                        Retained<ProtocolObject<dyn MTLBuffer>>,
                        Retained<ProtocolObject<dyn MTLBuffer>>,
                        Retained<ProtocolObject<dyn MTLBuffer>>,
                        Retained<ProtocolObject<dyn MTLBuffer>>,
                    )> = if let Some(view) = s.kvarn {
                        kvarn_dummy_query = vec![0.0f32; view.num_heads * view.head_dim];
                        let request = rnb_backend_api::KvarnDecodeRequest::new(
                            view.layer_idx,
                            &kvarn_dummy_query,
                            view.packed_blocks,
                            view.sink_key,
                            view.sink_value,
                            view.tail_key,
                            view.tail_value,
                            view.kv_len,
                            view.tail_start,
                            view.num_heads,
                            view.num_kv_heads,
                            view.head_dim,
                            view.key_bits,
                            view.value_bits,
                            view.group,
                            view.sink_tokens,
                            view.block_bytes,
                            view.scale,
                            view.sliding_window,
                            view.softcap,
                        );
                        let mut residents = self.kvarn_residents.borrow_mut();
                        let recreate = residents
                            .get(&s.layer)
                            .map(|r| !r.matches(request))
                            .unwrap_or(true);
                        if recreate {
                            residents.insert(
                                s.layer,
                                compute::KvarnResident::new(ctx, request)
                                    .expect("kvarn chain resident new"),
                            );
                        }
                        residents
                            .get_mut(&s.layer)
                            .expect("kvarn chain resident")
                            .update(ctx, request)
                            .expect("kvarn chain resident update");
                        let (params_buf, append_slot, num_blocks) =
                            compute::kvarn_chain_params_buf(ctx, request)
                                .expect("kvarn chain params");
                        let num_splits = num_blocks.clamp(1, 8);
                        let num_splits_buf = constant_u32(num_splits as u32);
                        let partial_acc = crate::ffn_chain::empty_f32_buf(
                            ctx,
                            num_splits * view.num_heads * view.head_dim,
                        );
                        let partial_m =
                            crate::ffn_chain::empty_f32_buf(ctx, num_splits * view.num_heads);
                        let partial_s =
                            crate::ffn_chain::empty_f32_buf(ctx, num_splits * view.num_heads);
                        Some((
                            params_buf,
                            append_slot,
                            num_splits,
                            num_splits_buf,
                            partial_acc,
                            partial_m,
                            partial_s,
                        ))
                    } else {
                        None
                    };

                    let mut carriers = self.attn_moe_carriers.borrow_mut();
                    let carrier = carriers.entry(s.layer).or_insert_with(|| {
                        attn_chain::AttnCarrier::new(
                            ctx,
                            s.hidden_dim,
                            s.q_dim,
                            s.q_out_dim,
                            s.kv_dim,
                            s.head_dim,
                            s.num_heads,
                            s.num_kv_heads,
                            s.n_rot,
                            s.capacity,
                            1,
                            s.eps,
                            s.theta,
                            s.scale,
                        )
                    });
                    if s.kvarn.is_none() {
                        carrier.kv_ensure_filled(s.prior_k, s.prior_v, s.pos);
                        attn_moe_filled_updates.push((s.layer, s.pos + 1));
                    }

                    let kvarn_residents_ref = self.kvarn_residents.borrow();
                    let kvarn_ctx = kvarn_state.as_ref().map(
                        |(
                            params_buf,
                            append_slot,
                            num_splits,
                            num_splits_buf,
                            partial_acc,
                            partial_m,
                            partial_s,
                        )| {
                            compute::KvarnChainEncode {
                                resident: kvarn_residents_ref
                                    .get(&s.layer)
                                    .expect("kvarn chain resident for encode"),
                                params_buf: &**params_buf,
                                num_splits_buf: &**num_splits_buf,
                                partial_acc: &**partial_acc,
                                partial_m: &**partial_m,
                                partial_s: &**partial_s,
                                append_slot: *append_slot,
                                num_splits: *num_splits,
                                num_heads: s.num_heads,
                            }
                        },
                    );
                    attn_chain::attn_chain_encode_core(
                        ctx,
                        &enc,
                        carrier,
                        &shared_hidden,
                        &norm_w_buf,
                        &q_w,
                        &q_off_buf,
                        &k_w,
                        &k_off_buf,
                        &v_w,
                        &v_off_buf,
                        &q_norm_w_buf,
                        &k_norm_w_buf,
                        &o_w,
                        &o_off_buf,
                        s.q_q,
                        s.k_q,
                        s.v_q,
                        s.o_q,
                        s.pos,
                        kvarn_ctx,
                    );

                    let mut moe_carriers = self.qwen_moe_decode_chain_carriers.borrow_mut();
                    let moe_carrier = moe_carriers
                        .entry(qwen_moe_decode_chain_carrier_key(
                            s.layer,
                            s.n_expert_used,
                            s.hidden_dim,
                            s.n_ff,
                            s.n_expert,
                            s.down_quant,
                        ))
                        .or_insert_with(|| {
                            ffn_chain::QwenMoeDecodeChainCarrier::new(
                                ctx,
                                s.n_expert_used,
                                s.hidden_dim,
                                s.n_ff,
                                s.n_expert,
                                s.gate_expert_bytes,
                                s.up_expert_bytes,
                                s.down_expert_bytes,
                                s.n_expert as u32,
                                s.eps,
                            )
                        });
                    ffn_chain::qwen_moe_decode_chain_encode(
                        ctx,
                        &enc,
                        moe_carrier,
                        &shared_hidden,
                        &ffn_norm_w_buf,
                        &router_w,
                        &router_off_buf,
                        &shared_input_scale_buf,
                        &gate_exps_w,
                        gate_exps_off,
                        &up_exps_w,
                        up_exps_off,
                        &down_exps_w,
                        down_exps_off,
                        &shared_gate_w,
                        shared_gate_off,
                        &shared_up_w,
                        shared_up_off,
                        &shared_down_w,
                        shared_down_off,
                        s.down_quant,
                        s.shared_q8_0,
                    );
                    report.qwen_moe_layers += 1;
                }
                ChainLayerSpecRef::Gdn(s) => {
                    let (qkv_w, qkv_off) = wrap(s.qkv_raw);
                    let (gate_w, gate_off) = wrap(s.gate_raw);
                    let (alpha_w, alpha_off) = wrap(s.alpha_raw);
                    let (beta_w, beta_off) = wrap(s.beta_raw);
                    let (ssm_out_w, ssm_out_off) = wrap(s.ssm_out_raw);

                    let qkv_off_buf = constant_u32(qkv_off);
                    let gate_off_buf = constant_u32(gate_off);
                    let alpha_off_buf = constant_u32(alpha_off);
                    let beta_off_buf = constant_u32(beta_off);
                    let ssm_out_off_buf = constant_u32(ssm_out_off);

                    let attn_norm_w_buf = constant_f32(s.attn_norm_weight);
                    let dt_bias_w_buf = constant_f32(s.dt_bias_weight);
                    let ssm_a_w_buf = constant_f32(s.ssm_a_weight);
                    let conv1d_w_buf = constant_f32(s.conv1d_weight);
                    let ssm_norm_w_buf = constant_f32(s.ssm_norm_weight);
                    let ffn_norm_w_buf = constant_f32(s.ffn_norm_weight);

                    let mut carriers = self.gdn_carriers.borrow_mut();
                    let carrier = carriers.entry(s.layer).or_insert_with(|| {
                        gdn_chain::GdnCarrier::new(
                            ctx,
                            s.hidden_dim,
                            s.conv_channels,
                            s.conv_kernel,
                            s.z_dim,
                            s.num_v_heads,
                            s.num_k_heads,
                            s.head_k_dim,
                            s.head_v_dim,
                            s.ffn_dim,
                            s.eps,
                        )
                    });
                    if delta_resident {
                        carrier.upload_states_resident(s.conv_state, s.delta_state);
                    } else {
                        carrier.upload_states(s.conv_state, s.delta_state);
                    }
                    report.upload_bytes +=
                        (s.conv_state.len() + s.delta_state.len()) * std::mem::size_of::<f32>();
                    report.state_syncs += 1;

                    let (ffn_gate_w, ffn_gate_off) = wrap(s.ffn_gate_raw);
                    let (ffn_up_w, ffn_up_off) = wrap(s.ffn_up_raw);
                    let (ffn_down_w, ffn_down_off) = wrap(s.ffn_down_raw);
                    let ffn_gate_off_buf = constant_u32(ffn_gate_off);
                    let ffn_up_off_buf = constant_u32(ffn_up_off);
                    let ffn_down_off_buf = constant_u32(ffn_down_off);
                    gdn_chain::gdn_chain_encode(
                        ctx,
                        &enc,
                        carrier,
                        &shared_hidden,
                        &attn_norm_w_buf,
                        &qkv_w,
                        &qkv_off_buf,
                        &gate_w,
                        &gate_off_buf,
                        &alpha_w,
                        &alpha_off_buf,
                        &beta_w,
                        &beta_off_buf,
                        &dt_bias_w_buf,
                        &ssm_a_w_buf,
                        &conv1d_w_buf,
                        &ssm_norm_w_buf,
                        &ssm_out_w,
                        &ssm_out_off_buf,
                        &ffn_norm_w_buf,
                        &ffn_gate_w,
                        &ffn_gate_off_buf,
                        &ffn_up_w,
                        &ffn_up_off_buf,
                        &ffn_down_w,
                        &ffn_down_off_buf,
                        s.qkv_q,
                        s.gate_q,
                        s.alpha_q,
                        s.beta_q,
                        s.ssm_out_q,
                        s.ffn_gate_q,
                        s.ffn_up_q,
                        s.ffn_down_q,
                    );
                }
                ChainLayerSpecRef::GdnMoeQwen(s) => {
                    let router_bytes = unsafe {
                        std::slice::from_raw_parts(
                            s.router_w.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(s.router_w),
                        )
                    };
                    let (qkv_w, qkv_off) = wrap(s.qkv_raw);
                    let (gate_w, gate_off) = wrap(s.gate_raw);
                    let (alpha_w, alpha_off) = wrap(s.alpha_raw);
                    let (beta_w, beta_off) = wrap(s.beta_raw);
                    let (ssm_out_w, ssm_out_off) = wrap(s.ssm_out_raw);
                    let (router_w, router_off) = wrap(router_bytes);
                    let (gate_exps_w, gate_exps_off) = wrap(s.gate_exps_raw);
                    let (up_exps_w, up_exps_off) = wrap(s.up_exps_raw);
                    let (down_exps_w, down_exps_off) = wrap(s.down_exps_raw);
                    let (shared_gate_w, shared_gate_off) = wrap(s.shared_gate_raw);
                    let (shared_up_w, shared_up_off) = wrap(s.shared_up_raw);
                    let (shared_down_w, shared_down_off) = wrap(s.shared_down_raw);

                    let qkv_off_buf = constant_u32(qkv_off);
                    let gate_off_buf = constant_u32(gate_off);
                    let alpha_off_buf = constant_u32(alpha_off);
                    let beta_off_buf = constant_u32(beta_off);
                    let ssm_out_off_buf = constant_u32(ssm_out_off);
                    let router_off_buf = constant_u32(router_off);

                    let attn_norm_w_buf = constant_f32(s.attn_norm_weight);
                    let dt_bias_w_buf = constant_f32(s.dt_bias_weight);
                    let ssm_a_w_buf = constant_f32(s.ssm_a_weight);
                    let conv1d_w_buf = constant_f32(s.conv1d_weight);
                    let ssm_norm_w_buf = constant_f32(s.ssm_norm_weight);
                    let ffn_norm_w_buf = constant_f32(s.ffn_norm_weight);
                    let shared_input_scale_buf = constant_f32(s.shared_input_scale);

                    let mut carriers = self.gdn_carriers.borrow_mut();
                    let carrier = carriers.entry(s.layer).or_insert_with(|| {
                        gdn_chain::GdnCarrier::new(
                            ctx,
                            s.hidden_dim,
                            s.conv_channels,
                            s.conv_kernel,
                            s.z_dim,
                            s.num_v_heads,
                            s.num_k_heads,
                            s.head_k_dim,
                            s.head_v_dim,
                            1,
                            s.eps,
                        )
                    });
                    if delta_resident {
                        carrier.upload_states_resident(s.conv_state, s.delta_state);
                    } else {
                        carrier.upload_states(s.conv_state, s.delta_state);
                    }
                    report.upload_bytes +=
                        (s.conv_state.len() + s.delta_state.len()) * std::mem::size_of::<f32>();
                    report.state_syncs += 1;

                    gdn_chain::gdn_core_chain_encode(
                        ctx,
                        &enc,
                        carrier,
                        &shared_hidden,
                        &attn_norm_w_buf,
                        &qkv_w,
                        &qkv_off_buf,
                        &gate_w,
                        &gate_off_buf,
                        &alpha_w,
                        &alpha_off_buf,
                        &beta_w,
                        &beta_off_buf,
                        &dt_bias_w_buf,
                        &ssm_a_w_buf,
                        &conv1d_w_buf,
                        &ssm_norm_w_buf,
                        &ssm_out_w,
                        &ssm_out_off_buf,
                        s.qkv_q,
                        s.gate_q,
                        s.alpha_q,
                        s.beta_q,
                        s.ssm_out_q,
                        true, // pm132: MoE fusion은 후속, 현재 기존 동작 유지
                    );

                    let mut moe_carriers = self.qwen_moe_decode_chain_carriers.borrow_mut();
                    let moe_carrier = moe_carriers
                        .entry(qwen_moe_decode_chain_carrier_key(
                            s.layer,
                            s.n_expert_used,
                            s.hidden_dim,
                            s.n_ff,
                            s.n_expert,
                            s.down_quant,
                        ))
                        .or_insert_with(|| {
                            ffn_chain::QwenMoeDecodeChainCarrier::new(
                                ctx,
                                s.n_expert_used,
                                s.hidden_dim,
                                s.n_ff,
                                s.n_expert,
                                s.gate_expert_bytes,
                                s.up_expert_bytes,
                                s.down_expert_bytes,
                                s.n_expert as u32,
                                s.eps,
                            )
                        });
                    ffn_chain::qwen_moe_decode_chain_encode(
                        ctx,
                        &enc,
                        moe_carrier,
                        &shared_hidden,
                        &ffn_norm_w_buf,
                        &router_w,
                        &router_off_buf,
                        &shared_input_scale_buf,
                        &gate_exps_w,
                        gate_exps_off,
                        &up_exps_w,
                        up_exps_off,
                        &down_exps_w,
                        down_exps_off,
                        &shared_gate_w,
                        shared_gate_off,
                        &shared_up_w,
                        shared_up_off,
                        &shared_down_w,
                        shared_down_off,
                        s.down_quant,
                        s.shared_q8_0,
                    );
                    report.qwen_moe_layers += 1;
                }
            }
            // 진단(RNB_METAL_CHAIN_SKIP_BARRIER=1): layer 내부 barrier 는 생략되지만
            // layer 경계는 강제 barrier 로 순차성만 유지 → "layer 순차 + layer 내 병렬"
            // achievable 하한을 측정한다. 비-skip 모드에선 각 encode 끝 barrier 가 이미
            // 경계라 이 블록은 실행되지 않는다(무해).
            if compute::chain_skip_barrier() {
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
        }

        let mut output_token_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>> = None;
        if let Some(tail) = output_argmax {
            assert_eq!(
                tail.cols, hidden_dim,
                "decode_chain output tail cols must match hidden dim"
            );
            assert_eq!(
                tail.norm_weight.len(),
                hidden_dim,
                "decode_chain output norm weight length mismatch"
            );
            assert!(
                matches!(tail.output_quant, 0 | 2),
                "decode_chain output tail supports Q4_K/Q6_K only"
            );
            assert!(tail.rows > 0, "decode_chain output tail rows must be > 0");

            let (out_w, out_off) = wrap(tail.output_raw);
            let norm_w_buf = ffn_chain::shared_f32_buf(ctx, tail.norm_weight);
            let normed_buf = ffn_chain::empty_f32_buf(ctx, hidden_dim);
            let logits_buf = ffn_chain::empty_f32_buf(ctx, tail.rows);
            let token_buf = ctx
                .device
                .newBufferWithLength_options(
                    std::mem::size_of::<u32>(),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: failed to create decode output token buffer");
            let dim_buf = ffn_chain::u32_buf(ctx, hidden_dim as u32);
            let eps_buf = ffn_chain::f32_buf(ctx, tail.eps);
            let rows_buf = ffn_chain::u32_buf(ctx, tail.rows as u32);
            let cols_buf = ffn_chain::u32_buf(ctx, tail.cols as u32);
            let out_off_buf = ffn_chain::u32_buf(ctx, out_off);

            ffn_chain::encode_rms_norm(
                ctx,
                &enc,
                &shared_hidden,
                &norm_w_buf,
                &normed_buf,
                &dim_buf,
                &eps_buf,
            );
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            match tail.output_quant {
                0 => compute::encode_gemv_q4k_simd(
                    ctx,
                    &enc,
                    &out_w,
                    &normed_buf,
                    &logits_buf,
                    &rows_buf,
                    &cols_buf,
                    &out_off_buf,
                    tail.rows,
                ),
                2 => compute::encode_gemv_q6k_simd(
                    ctx,
                    &enc,
                    &out_w,
                    &normed_buf,
                    &logits_buf,
                    &rows_buf,
                    &cols_buf,
                    &out_off_buf,
                    tail.rows,
                ),
                _ => unreachable!("validated above"),
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            compute::encode_argmax_f32(ctx, &enc, &logits_buf, &token_buf, &rows_buf);
            report.argmax_only = true;
            report.output_argmax = OutputArgmaxReport {
                attempted: true,
                did_run: true,
                token_id: None,
                cmd_buffers: 0,
                wait_us: 0,
                readback_bytes: std::mem::size_of::<u32>(),
                fallback_reason: None,
            };
            output_token_buf = Some(token_buf);
        }

        enc.endEncoding();

        // encode(dispatch 인코딩) 구간 종료 — command buffer 생성~endEncoding.
        let encode_ms = t_encode_start
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // ── RNB_METAL_CHAIN_GPU_TIME=1 계측 ──────────────────────────────────
        // commit+waitUntilCompleted 구간을 CPU wall 로 감싸고, 완료 후
        // MTLCommandBuffer 의 GPUStartTime/GPUEndTime 으로 순수 GPU 실행 시간을
        // 분리한다. readback 시간은 별도 측정. token 번호는 static atomic 으로 추적.
        let t_commit_start = if chain_gpu_time {
            Some(std::time::Instant::now())
        } else {
            None
        };

        cmd.commit();
        cmd.waitUntilCompleted();

        if let Some(token_buf) = output_token_buf.as_ref() {
            let token = unsafe { *(token_buf.contents().as_ptr() as *const u32) };
            report.output_argmax.token_id = Some(token);
        }

        let (commit_wait_ms, gpu_ms) = if let Some(t0) = t_commit_start {
            let commit_wait_ms = t0.elapsed().as_secs_f64() * 1000.0;
            // GPUStartTime/GPUEndTime: Metal 이 GPU 실제 실행 구간을 CFTimeInterval(초)로 반환.
            // waitUntilCompleted 후에만 유효한 값이 채워짐.
            let gpu_start = cmd.GPUStartTime();
            let gpu_end = cmd.GPUEndTime();
            let gpu_ms = (gpu_end - gpu_start) * 1000.0;
            (commit_wait_ms, gpu_ms)
        } else {
            (0.0, 0.0)
        };
        report.wait_us = (commit_wait_ms * 1000.0).round() as u64;

        let t_readback_start = if chain_gpu_time {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 마지막 hidden 을 host 로 반영(공유 buffer 가 모든 layer 누적 결과).
        hidden.copy_from_slice(&ffn_chain::readback(&shared_hidden, hidden_dim));
        report.hidden_readback_bytes = hidden_dim * std::mem::size_of::<f32>();
        report.readback_bytes += report.hidden_readback_bytes;

        let hidden_readback_ms = t_readback_start
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // attn carrier 의 KV filled 갱신(commit 후 — kv_append 가 device 에 반영됐으므로).
        {
            let mut carriers = self.attn_carriers.borrow_mut();
            for (layer, kv_len) in attn_filled_updates {
                if let Some(carrier) = carriers.get_mut(&layer) {
                    carrier.kv_set_filled(kv_len);
                }
            }
        }
        {
            let mut carriers = self.attn_moe_carriers.borrow_mut();
            for (layer, kv_len) in attn_moe_filled_updates {
                if let Some(carrier) = carriers.get_mut(&layer) {
                    carrier.kv_set_filled(kv_len);
                }
            }
        }

        // GDN layer state readback(attn 은 None — KV device 소유).
        let t_state_start = if chain_gpu_time {
            Some(std::time::Instant::now())
        } else {
            None
        };
        {
            let gdn_carriers = self.gdn_carriers.borrow();
            for (spec, out) in specs.iter().zip(out_states.iter_mut()) {
                match spec {
                    ChainLayerSpecRef::Attn(_) => *out = None,
                    ChainLayerSpecRef::AttnMoeQwen(s) => {
                        *out = if s.kvarn.is_some() {
                            let carriers = self.attn_moe_carriers.borrow();
                            let carrier = carriers
                                .get(&s.layer)
                                .expect("attn moe carrier for kv readback");
                            let (k, v) = carrier.readback_new_kv();
                            report.readback_bytes +=
                                (k.len() + v.len()) * std::mem::size_of::<f32>();
                            report.state_syncs += 1;
                            Some((k, v))
                        } else {
                            None
                        };
                    }
                    ChainLayerSpecRef::Gdn(s) => {
                        let carrier = gdn_carriers
                            .get(&s.layer)
                            .expect("decode_chain_run: gdn carrier missing after encode");
                        // pm31: delta residency 시 conv 만 readback, delta 는 device 잔류라
                        // 빈 Vec(caller 가 skip) → 9ms readback 의 96.7% 제거.
                        *out = if delta_resident {
                            let conv = carrier.readback_conv_state();
                            report.readback_bytes += conv.len() * std::mem::size_of::<f32>();
                            report.state_syncs += 1;
                            Some((conv, Vec::new()))
                        } else {
                            let states = carrier.readback_states();
                            report.readback_bytes +=
                                (states.0.len() + states.1.len()) * std::mem::size_of::<f32>();
                            report.state_syncs += 1;
                            Some(states)
                        };
                    }
                    ChainLayerSpecRef::GdnMoeQwen(s) => {
                        let carrier = gdn_carriers
                            .get(&s.layer)
                            .expect("decode_chain_run: gdn moe carrier missing after encode");
                        *out = if delta_resident {
                            let conv = carrier.readback_conv_state();
                            report.readback_bytes += conv.len() * std::mem::size_of::<f32>();
                            report.state_syncs += 1;
                            Some((conv, Vec::new()))
                        } else {
                            let states = carrier.readback_states();
                            report.readback_bytes +=
                                (states.0.len() + states.1.len()) * std::mem::size_of::<f32>();
                            report.state_syncs += 1;
                            Some(states)
                        };
                    }
                }
            }
        }
        let state_readback_ms = t_state_start
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        if chain_gpu_time {
            let fn_ms = t_fn_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let cpu_overhead_ms = commit_wait_ms - gpu_ms;
            eprintln!(
                "[chain-gputime] token={} fn_ms={:.2} encode_ms={:.2} gpu_ms={:.2} commit_wait_ms={:.2} cpu_overhead_ms={:.2} hidden_readback_ms={:.2} state_readback_ms={:.2}",
                token_idx, fn_ms, encode_ms, gpu_ms, commit_wait_ms, cpu_overhead_ms,
                hidden_readback_ms, state_readback_ms
            );
        }

        report
    }

    /// 소량 배치(seq_len=B) 디코드 체인 — MTP speculative verify 의 groundwork.
    ///
    /// `hidden` 은 `[batch * hidden_dim]` (행 i = lane i 의 입력 임베딩). B lane 을 **연속
    /// 시퀀스**로 처리한다: attn lane i 는 pos=base_pos+i 에서 prefix `[0..base_pos+i]` 를
    /// attend 하고, GDN lane i 는 lane i-1 의 conv/delta state 를 이어받아 전진한다. 이는
    /// MTP verify 시맨틱(후보 토큰 d_0..d_{B-1} 을 pos..pos+B-1 에 순차 forward)과 동일하다.
    ///
    /// **milestone 1 (이 함수) = 정확성 우선.** lane 을 검증된 단일-토큰 `decode_chain_run`
    /// 위에서 순차로 돌리며(lane 당 1 command buffer) state 를 threading 한다. resident weight
    /// wrap(`self.resident`)/carrier 는 lane 간 재사용되어 weight **업로드**는 amortize 되지만,
    /// GPU 의 weight **읽기**는 lane 마다 반복된다(진짜 amortization = B-column GEMV + device
    /// resident cross-token state advance 는 milestone 2). `batch == 1` 은 `decode_chain_run`
    /// 과 바이트 동일한 hot 경로를 그대로 탄다.
    ///
    /// 반환: lane 당 `DecodeChainReport`(각 `output_argmax.token_id` 에 그 위치의 argmax
    /// 토큰). `out_states` 에는 마지막 lane 처리 후의 per-layer state 를 채운다.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn decode_chain_run_batched_impl(
        &self,
        hidden: &mut [f32],
        batch: usize,
        specs: &[ChainLayerSpecRef<'_>],
        out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
        options: DecodeChainOptions,
        output_argmax: Option<DecodeOutputArgmaxSpecRef<'_>>,
        mut out_attn_kv: Option<&mut Vec<Option<(Vec<u16>, Vec<u16>)>>>,
    ) -> Vec<DecodeChainReport> {
        assert!(batch >= 1, "decode_chain_run_batched: batch must be >= 1");
        assert_eq!(
            specs.len(),
            out_states.len(),
            "decode_chain_run_batched: specs/out_states length mismatch"
        );
        if specs.is_empty() {
            return (0..batch)
                .map(|_| DecodeChainReport {
                    did_run: true,
                    ..DecodeChainReport::default()
                })
                .collect();
        }
        let hidden_dim = hidden.len() / batch;
        assert_eq!(
            hidden_dim * batch,
            hidden.len(),
            "decode_chain_run_batched: hidden length must equal batch * hidden_dim"
        );

        // GDN layer 의 threaded state(conv, delta). base spec 에서 seed(비-GDN layer 는 empty).
        // lane i 처리 후 out_states 의 새 state 로 갱신 → lane i+1 이 이어받는다.
        let mut conv_thread: Vec<Vec<f32>> = vec![Vec::new(); specs.len()];
        let mut delta_thread: Vec<Vec<f32>> = vec![Vec::new(); specs.len()];
        for (li, spec) in specs.iter().enumerate() {
            match spec {
                ChainLayerSpecRef::Gdn(s) => {
                    conv_thread[li] = s.conv_state.to_vec();
                    delta_thread[li] = s.delta_state.to_vec();
                }
                ChainLayerSpecRef::GdnMoeQwen(s) => {
                    conv_thread[li] = s.conv_state.to_vec();
                    delta_thread[li] = s.delta_state.to_vec();
                }
                _ => {}
            }
        }

        // batch>1 이고 output tail 이 있으면: chain body 는 lane 별로 돌리되(output_argmax=None),
        // 최종 출력 프로젝션(vocab weight, 프로덕션에서 가장 큰 단일 행렬 중 하나)은 B lane 을
        // 모아 **B-column GEMV 로 weight 를 1회만 읽어** 계산한다(milestone 2 amortization 배선).
        // batch==1 은 기존 per-lane decode_chain_run(output_argmax) 경로 그대로 → 바이트 동일.
        // B-column GEMV 커널의 sumf[BCOL_MAX=8] 한계 — batch>8 이면 per-lane 출력으로 폴백(정확).
        // collect 모드(MTP verify)는 batch==1 도 배치 carrier 경로로 처리한다 — full-reject
        // (n_accepted=0 → committed=1 재실행) 시 프로덕션 stateful carrier(attn_moe/gdn)를
        // 건드리지 않고 out_attn_kv/out_states 를 host 로 반환하기 위함(host = source of truth).
        let collect = out_attn_kv.is_some();
        let fused_lo = if collect { 1usize } else { 2 };
        let want_batched_tail = (fused_lo..=8).contains(&batch) && output_argmax.is_some();
        let body_argmax = if want_batched_tail {
            None
        } else {
            output_argmax
        };

        let mut reports = Vec::with_capacity(batch);
        // milestone 3/4: layer 가 모두 GdnMoeQwen | AttnMoeQwen 이고 batch 2..=8 이면 **body
        // fusion** 경로 — layer 당 dense weight(q/k/v/o·qkv/gate/ssm_out·shared)를 B-column GEMV 로
        // 1회만 읽어 B lane 을 한 command buffer 로 처리한다(MoE sparse expert 는 lane 별). batch==1 /
        // 비-MoE-Qwen spec / opt-out 은 아래 검증된 per-lane 경로로 폴백(정확·불변). batched attn
        // fusion 은 자체 f16 KV 라 ctx.kv_int8 무관하게 정확 — KVarn attn chain 만 per-lane 폴백.
        let fused_ran = {
            let all_moe_qwen = !specs.is_empty()
                && specs.iter().all(|s| {
                    matches!(
                        s,
                        ChainLayerSpecRef::GdnMoeQwen(_) | ChainLayerSpecRef::AttnMoeQwen(_)
                    )
                });
            let has_attn = specs
                .iter()
                .any(|s| matches!(s, ChainLayerSpecRef::AttnMoeQwen(_)));
            // batched attn 은 자체 f16 KvResident(new_f16) + f16 커널(kv_append_at/attn_decode_at)
            // + host F16 kv_cache prior 로 처리 → ctx.kv_int8 과 무관하게 항상 f16-정확. 따라서
            // 게이트는 KVarn 만 배제(KVarn attn 은 배치 미지원 → per-lane 폴백).
            let attn_fusable = !has_attn
                || specs.iter().all(|s| match s {
                    ChainLayerSpecRef::AttnMoeQwen(a) => a.kvarn.is_none(),
                    _ => true,
                });
            let enabled = std::env::var_os("RNB_METAL_BATCH_FUSED").map_or(true, |v| v != "0");
            if all_moe_qwen && attn_fusable && (fused_lo..=8).contains(&batch) && enabled {
                self.decode_chain_run_batched_moe_fused(
                    hidden,
                    batch,
                    hidden_dim,
                    specs,
                    out_states,
                    &mut reports,
                    out_attn_kv.take(),
                );
                true
            } else {
                false
            }
        };
        if !fused_ran {
            for lane in 0..batch {
                // 현재 threaded state 를 lane-local buffer 로 snapshot — per-lane spec 이 이걸
                // borrow 하는 동안 conv_thread/delta_thread 는 자유롭게 갱신할 수 있다(별도 Vec).
                let lane_conv = conv_thread.clone();
                let lane_delta = delta_thread.clone();
                let lane_specs: Vec<ChainLayerSpecRef<'_>> = specs
                    .iter()
                    .enumerate()
                    .map(|(li, spec)| {
                        let mut s = *spec;
                        match &mut s {
                            ChainLayerSpecRef::Attn(a) => a.pos += lane,
                            ChainLayerSpecRef::AttnMoeQwen(a) => a.pos += lane,
                            ChainLayerSpecRef::Gdn(g) => {
                                g.conv_state = &lane_conv[li];
                                g.delta_state = &lane_delta[li];
                            }
                            ChainLayerSpecRef::GdnMoeQwen(g) => {
                                g.conv_state = &lane_conv[li];
                                g.delta_state = &lane_delta[li];
                            }
                        }
                        s
                    })
                    .collect();

                let mut row = hidden[lane * hidden_dim..(lane + 1) * hidden_dim].to_vec();
                let mut lane_states: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; specs.len()];
                let report = self.decode_chain_run(
                    &mut row,
                    &lane_specs,
                    &mut lane_states,
                    options,
                    body_argmax,
                );
                hidden[lane * hidden_dim..(lane + 1) * hidden_dim].copy_from_slice(&row);

                // 다음 lane 으로 GDN state threading(conv_thread/delta_thread 는 lane_conv/lane_delta
                // 와 별개 Vec 이라 lane_specs 의 borrow 와 충돌하지 않는다). delta 가 empty 면
                // (delta_resident) device 가 잔류 소유 → thread 값은 미사용.
                for (li, st) in lane_states.iter().enumerate() {
                    if let Some((conv_new, delta_new)) = st {
                        if !conv_new.is_empty() {
                            conv_thread[li] = conv_new.clone();
                        }
                        if !delta_new.is_empty() {
                            delta_thread[li] = delta_new.clone();
                        }
                    }
                }

                if lane + 1 == batch {
                    for (dst, src) in out_states.iter_mut().zip(lane_states.into_iter()) {
                        *dst = src;
                    }
                }
                reports.push(report);
            }
        }

        if want_batched_tail {
            let tail = output_argmax.expect("want_batched_tail implies Some");
            self.decode_chain_batched_output_argmax(hidden, batch, hidden_dim, &tail, &mut reports);
        }

        reports
    }

    /// B lane(2..=8) 배치 디코드 체인. `batch == 1` 은 `decode_chain_run` 과 바이트 동일 hot
    /// 경로. 모든 layer 가 GdnMoeQwen|AttnMoeQwen 이고(+attn 은 f16 KV·non-KVarn) fused
    /// single-command-buffer body(dense weight 1회 읽기)로 처리, 그 외엔 검증된 per-lane 폴백.
    /// `out_states` 에 마지막 lane 처리 후 per-layer state 를 채운다.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_chain_run_batched(
        &self,
        hidden: &mut [f32],
        batch: usize,
        specs: &[ChainLayerSpecRef<'_>],
        out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
        options: DecodeChainOptions,
        output_argmax: Option<DecodeOutputArgmaxSpecRef<'_>>,
    ) -> Vec<DecodeChainReport> {
        self.decode_chain_run_batched_impl(
            hidden,
            batch,
            specs,
            out_states,
            options,
            output_argmax,
            None,
        )
    }

    /// `decode_chain_run_batched` 의 MTP verify 변형: 위와 동일하게 실행하되, 각 AttnMoeQwen
    /// layer 가 이번 pass 에서 device append 한 window(모든 lane, slot `base_pos..base_pos+batch`)의
    /// **post-rope f16 bits** K/V 를 `out_attn_kv`(layer 순서; attn=Some, 그 외 None)에 채운다.
    /// 반환 K/V 는 slot-major contiguous `[batch*kv_dim]`(slot i 는 `[i*kv_dim..]`). 엔진은
    /// accept-n 커밋에서 앞 n slot 을 host kv_cache 에 append 한다(host = source of truth,
    /// 다음 라운드 prior 를 정확히 공급). **f16 KV + non-KVarn + batch 2..=8(fused 경로) 전제** —
    /// fallback 경로면 attn 항목은 None 으로 남는다(MTP 는 fused 경로만 사용). partial-accept 는
    /// 엔진이 `batch = committed` 로 재실행해 그 pass 의 K/V(= committed window)를 받는다.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_chain_run_batched_collect_attn_kv(
        &self,
        hidden: &mut [f32],
        batch: usize,
        specs: &[ChainLayerSpecRef<'_>],
        out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
        options: DecodeChainOptions,
        output_argmax: Option<DecodeOutputArgmaxSpecRef<'_>>,
        out_attn_kv: &mut Vec<Option<(Vec<u16>, Vec<u16>)>>,
    ) -> Vec<DecodeChainReport> {
        out_attn_kv.clear();
        out_attn_kv.resize(specs.len(), None);
        self.decode_chain_run_batched_impl(
            hidden,
            batch,
            specs,
            out_states,
            options,
            output_argmax,
            Some(out_attn_kv),
        )
    }

    /// MTP verify accept-n 커밋(partial): 직전 `decode_chain_run_batched_collect_attn_kv`
    /// (또는 batched) forward 가 device 에 보존한 prefix state 에서, **n lane 처리 후**(=committed
    /// n 토큰)의 GDN conv/delta state 를 재실행 없이 readback 한다(layer 순서; GdnMoeQwen=Some,
    /// 그 외 None). 엔진은 이를 host ssm_state 에 써서 partial-accept 재실행을 제거한다.
    /// `n == batch` 는 forward out_states(prefix-B)와 동일. carrier 는 (layer,batch) 로 보존되어
    /// 다음 forward 전까지 유효.
    #[cfg(target_os = "macos")]
    pub fn decode_chain_run_batched_gdn_prefix(
        &self,
        specs: &[ChainLayerSpecRef<'_>],
        batch: usize,
        n: usize,
    ) -> Vec<Option<(Vec<f32>, Vec<f32>)>> {
        let carriers = self.gdn_batch_carriers.borrow();
        specs
            .iter()
            .map(|s| match s {
                ChainLayerSpecRef::GdnMoeQwen(g) => carriers
                    .get(&(g.layer, batch))
                    .map(|c| c.readback_prefix_states(n)),
                _ => None,
            })
            .collect()
    }

    /// milestone 3/4 body fusion: 모든 layer 가 `GdnMoeQwen` 또는 `AttnMoeQwen` 인 chain 을
    /// B lane 에 대해 **layer 당 dense weight 1회 읽기**로 한 command buffer 에 처리한다.
    ///   - GDN core(qkv/gate/alpha/beta/ssm_out): B-column GEMV(`gdn_core_chain_encode_bcol`,
    ///     conv rolling buffer + device delta 순차).
    ///   - ATTN core(q/k/v/o): B-column GEMV(`attn_core_chain_encode_bcol`) + per-lane
    ///     split/qk-norm/rope/kv-append/attn/gate. lane i 는 pos=base_pos+i, KV slot base_pos+i 를
    ///     device append 후 `[0..base_pos+i]` attend(같은 cb barrier 로 lane i+1 이 lane i 를 봄).
    ///   - MoE(`encode_batched_moe_fused`): router+sparse expert 는 lane 별 순차(routing 발산이라
    ///     per-lane), shared expert(dense)는 B-column GEMV 로 weight 1회 읽기 amortize.
    /// GDN conv/delta state 는 chain 양끝 host sync(진입 upload, commit 후 readback → out_states);
    /// ATTN KV 는 device f16 resident(prior upload → device append)이며 out_states 는 None(host
    /// KVarn/f16 cache 가 source of truth). 결과는 lane 별 검증된 single-token decode 와 동일
    /// (reduction 순서 차 rel<3e-3). f16 KV + non-KVarn 전제(gate 에서 보장).
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn decode_chain_run_batched_moe_fused(
        &self,
        hidden: &mut [f32],
        batch: usize,
        hidden_dim: usize,
        specs: &[ChainLayerSpecRef<'_>],
        out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
        reports: &mut Vec<DecodeChainReport>,
        out_attn_kv: Option<&mut Vec<Option<(Vec<u16>, Vec<u16>)>>>,
    ) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        let constant_u32 = |value| {
            let mut cache = self.constant_u32.borrow_mut();
            cache
                .entry(value)
                .or_insert_with(|| ffn_chain::u32_buf(ctx, value))
                .clone()
        };
        let constant_f32 = |data: &[f32]| {
            let key = (data.as_ptr() as usize, data.len());
            let mut cache = self.constant_f32.borrow_mut();
            cache
                .entry(key)
                .or_insert_with(|| ffn_chain::shared_f32_buf(ctx, data))
                .clone()
        };
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };

        // 공유 hidden buffer [B*hidden_dim] (row i = lane i). 진입 1회 upload, layer 들이 in/out
        // 누적, 마지막에 readback. (batched 경로는 hot single-token 경로와 분리 — 매 호출 alloc.)
        let shared_hidden = ffn_chain::shared_f32_buf(ctx, hidden);

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = compute::chain_compute_encoder(ctx, &cmd);

        let mut attn_layers = 0usize;
        for spec in specs {
            match spec {
                ChainLayerSpecRef::GdnMoeQwen(s) => {
                    let (qkv_w, qkv_off) = wrap(s.qkv_raw);
                    let (gate_w, gate_off) = wrap(s.gate_raw);
                    let (alpha_w, alpha_off) = wrap(s.alpha_raw);
                    let (beta_w, beta_off) = wrap(s.beta_raw);
                    let (ssm_out_w, ssm_out_off) = wrap(s.ssm_out_raw);
                    let qkv_off_buf = constant_u32(qkv_off);
                    let gate_off_buf = constant_u32(gate_off);
                    let alpha_off_buf = constant_u32(alpha_off);
                    let beta_off_buf = constant_u32(beta_off);
                    let ssm_out_off_buf = constant_u32(ssm_out_off);
                    let attn_norm_w_buf = constant_f32(s.attn_norm_weight);
                    let dt_bias_w_buf = constant_f32(s.dt_bias_weight);
                    let ssm_a_w_buf = constant_f32(s.ssm_a_weight);
                    let conv1d_w_buf = constant_f32(s.conv1d_weight);
                    let ssm_norm_w_buf = constant_f32(s.ssm_norm_weight);

                    {
                        let mut carriers = self.gdn_batch_carriers.borrow_mut();
                        let carrier = carriers.entry((s.layer, batch)).or_insert_with(|| {
                            gdn_chain::GdnBatchCarrier::new(
                                ctx,
                                batch,
                                s.hidden_dim,
                                s.conv_channels,
                                s.conv_kernel,
                                s.z_dim,
                                s.num_v_heads,
                                s.num_k_heads,
                                s.head_k_dim,
                                s.head_v_dim,
                                s.eps,
                            )
                        });
                        carrier.upload_states(s.conv_state, s.delta_state);
                        gdn_chain::gdn_core_chain_encode_bcol(
                            ctx,
                            &enc,
                            carrier,
                            &shared_hidden,
                            &attn_norm_w_buf,
                            &qkv_w,
                            &qkv_off_buf,
                            &gate_w,
                            &gate_off_buf,
                            &alpha_w,
                            &alpha_off_buf,
                            &beta_w,
                            &beta_off_buf,
                            &dt_bias_w_buf,
                            &ssm_a_w_buf,
                            &conv1d_w_buf,
                            &ssm_norm_w_buf,
                            &ssm_out_w,
                            &ssm_out_off_buf,
                            s.qkv_q,
                            s.gate_q,
                            s.alpha_q,
                            s.beta_q,
                            s.ssm_out_q,
                        );
                    }

                    self.encode_batched_moe_fused(
                        ctx,
                        &enc,
                        &shared_hidden,
                        batch,
                        s.layer,
                        s.ffn_norm_weight,
                        s.router_w,
                        s.shared_input_scale,
                        s.gate_exps_raw,
                        s.up_exps_raw,
                        s.down_exps_raw,
                        s.shared_gate_raw,
                        s.shared_up_raw,
                        s.shared_down_raw,
                        s.n_expert_used,
                        s.hidden_dim,
                        s.n_ff,
                        s.n_expert,
                        s.gate_expert_bytes,
                        s.up_expert_bytes,
                        s.down_expert_bytes,
                        s.down_quant,
                        s.shared_q8_0,
                        s.eps,
                    );
                }
                ChainLayerSpecRef::AttnMoeQwen(s) => {
                    attn_layers += 1;
                    let (q_w, q_off) = wrap(s.q_raw);
                    let (k_w, k_off) = wrap(s.k_raw);
                    let (v_w, v_off) = wrap(s.v_raw);
                    let (o_w, o_off) = wrap(s.o_raw);
                    let q_off_buf = constant_u32(q_off);
                    let k_off_buf = constant_u32(k_off);
                    let v_off_buf = constant_u32(v_off);
                    let o_off_buf = constant_u32(o_off);
                    let norm_w_buf = constant_f32(s.norm_weight);
                    let q_norm_w_buf = constant_f32(s.q_norm_weight);
                    let k_norm_w_buf = constant_f32(s.k_norm_weight);

                    {
                        let mut carriers = self.attn_batch_carriers.borrow_mut();
                        let carrier = carriers.entry((s.layer, batch)).or_insert_with(|| {
                            attn_chain::AttnBatchCarrier::new(
                                ctx,
                                batch,
                                s.hidden_dim,
                                s.q_dim,
                                s.q_out_dim,
                                s.kv_dim,
                                s.head_dim,
                                s.num_heads,
                                s.num_kv_heads,
                                s.n_rot,
                                s.capacity,
                                s.eps,
                                s.theta,
                                s.scale,
                            )
                        });
                        carrier.upload_prior(s.prior_k, s.prior_v);
                        attn_chain::attn_core_chain_encode_bcol(
                            ctx,
                            &enc,
                            carrier,
                            &shared_hidden,
                            &norm_w_buf,
                            &q_w,
                            &q_off_buf,
                            &k_w,
                            &k_off_buf,
                            &v_w,
                            &v_off_buf,
                            &q_norm_w_buf,
                            &k_norm_w_buf,
                            &o_w,
                            &o_off_buf,
                            s.q_q,
                            s.k_q,
                            s.v_q,
                            s.o_q,
                            s.pos,
                        );
                    }

                    self.encode_batched_moe_fused(
                        ctx,
                        &enc,
                        &shared_hidden,
                        batch,
                        s.layer,
                        s.ffn_norm_weight,
                        s.router_w,
                        s.shared_input_scale,
                        s.gate_exps_raw,
                        s.up_exps_raw,
                        s.down_exps_raw,
                        s.shared_gate_raw,
                        s.shared_up_raw,
                        s.shared_down_raw,
                        s.n_expert_used,
                        s.hidden_dim,
                        s.n_ff,
                        s.n_expert,
                        s.gate_expert_bytes,
                        s.up_expert_bytes,
                        s.down_expert_bytes,
                        s.down_quant,
                        s.shared_q8_0,
                        s.eps,
                    );
                }
                _ => unreachable!("fused path gated on GdnMoeQwen|AttnMoeQwen"),
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        hidden.copy_from_slice(&ffn_chain::readback(&shared_hidden, batch * hidden_dim));

        // GDN layer: conv/delta state readback → out_states. ATTN layer: None(KV 는 host truth).
        {
            let carriers = self.gdn_batch_carriers.borrow();
            for (spec, out) in specs.iter().zip(out_states.iter_mut()) {
                match spec {
                    ChainLayerSpecRef::GdnMoeQwen(s) => {
                        let carrier = carriers
                            .get(&(s.layer, batch))
                            .expect("fused gdn batch carrier after encode");
                        *out = Some(carrier.readback_states());
                    }
                    ChainLayerSpecRef::AttnMoeQwen(_) => *out = None,
                    _ => unreachable!("fused path gated on GdnMoeQwen|AttnMoeQwen"),
                }
            }
        }

        // ATTN layer: verify pass 가 device append 한 window(slot base_pos..base_pos+batch)의
        // post-rope f16 K/V 를 layer 순서로 readback → out_attn_kv(attn=Some, 그 외 None).
        // 엔진이 MTP accept-n 커밋에서 host kv_cache 에 append 한다(slot-major [batch*kv_dim]).
        if let Some(out_kv) = out_attn_kv {
            debug_assert_eq!(out_kv.len(), specs.len());
            let carriers = self.attn_batch_carriers.borrow();
            for (spec, slot) in specs.iter().zip(out_kv.iter_mut()) {
                match spec {
                    ChainLayerSpecRef::AttnMoeQwen(s) => {
                        let carrier = carriers
                            .get(&(s.layer, batch))
                            .expect("fused attn batch carrier after encode");
                        *slot = Some(carrier.readback_kv_slots(s.pos, batch));
                    }
                    _ => *slot = None,
                }
            }
        }

        let moe_layers = specs.len();
        for _ in 0..batch {
            reports.push(DecodeChainReport {
                did_run: true,
                cmd_buffers: 1,
                qwen_moe_layers: moe_layers,
                upload_bytes: hidden_dim * std::mem::size_of::<f32>(),
                ..DecodeChainReport::default()
            });
        }
        if std::env::var_os("RNB_METAL_BATCH_FUSED_TRACE").is_some() {
            eprintln!(
                "[batch-fused] did_run=true layers={} (attn={} gdn={}) batch={} \
                 dense_weight_reads_per_layer=1 moe_shared=bcol-fused moe_sparse=per-lane",
                moe_layers,
                attn_layers,
                moe_layers - attn_layers,
                batch
            );
        }
    }

    /// 배치 fused chain 의 MoE 를 encode 한다. router+sparse expert 는 lane 별(routing 발산으로
    /// amortize 불가), shared expert(dense, always-active)는 **B-column GEMV 로 weight 1회 읽기**로
    /// amortize(가중치를 B lane 이 공유). GDN·ATTN core 양쪽 arm 이 공유(spec struct 는 다르지만
    /// MoE 필드 이름/의미 동일).
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_moe_fused(
        &self,
        ctx: &compute::MetalContext,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        shared_hidden: &ProtocolObject<dyn MTLBuffer>,
        batch: usize,
        layer: usize,
        ffn_norm_weight: &[f32],
        router_w: &[f32],
        shared_input_scale: &[f32],
        gate_exps_raw: &[u8],
        up_exps_raw: &[u8],
        down_exps_raw: &[u8],
        shared_gate_raw: &[u8],
        shared_up_raw: &[u8],
        shared_down_raw: &[u8],
        n_expert_used: usize,
        hidden_dim_s: usize,
        n_ff: usize,
        n_expert: usize,
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        down_quant: u8,
        shared_q8_0: bool,
        eps: f32,
    ) {
        let constant_u32 = |value| {
            let mut cache = self.constant_u32.borrow_mut();
            cache
                .entry(value)
                .or_insert_with(|| ffn_chain::u32_buf(ctx, value))
                .clone()
        };
        let constant_f32 = |data: &[f32]| {
            let key = (data.as_ptr() as usize, data.len());
            let mut cache = self.constant_f32.borrow_mut();
            cache
                .entry(key)
                .or_insert_with(|| ffn_chain::shared_f32_buf(ctx, data))
                .clone()
        };
        let wrap = |raw: &[u8]| {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, raw));
            (e.0.clone(), e.1)
        };
        let router_bytes = unsafe {
            std::slice::from_raw_parts(
                router_w.as_ptr().cast::<u8>(),
                std::mem::size_of_val(router_w),
            )
        };
        let (router_w_buf, router_off) = wrap(router_bytes);
        let (gate_exps_w, gate_exps_off) = wrap(gate_exps_raw);
        let (up_exps_w, up_exps_off) = wrap(up_exps_raw);
        let (down_exps_w, down_exps_off) = wrap(down_exps_raw);
        let (shared_gate_w, shared_gate_off) = wrap(shared_gate_raw);
        let (shared_up_w, shared_up_off) = wrap(shared_up_raw);
        let (shared_down_w, shared_down_off) = wrap(shared_down_raw);
        let router_off_buf = constant_u32(router_off);
        let ffn_norm_w_buf = constant_f32(ffn_norm_weight);
        let shared_input_scale_buf = constant_f32(shared_input_scale);

        let mut moe_carriers = self.qwen_moe_decode_chain_carriers.borrow_mut();
        let moe_carrier = moe_carriers
            .entry(qwen_moe_decode_chain_carrier_key(
                layer,
                n_expert_used,
                hidden_dim_s,
                n_ff,
                n_expert,
                down_quant,
            ))
            .or_insert_with(|| {
                ffn_chain::QwenMoeDecodeChainCarrier::new(
                    ctx,
                    n_expert_used,
                    hidden_dim_s,
                    n_ff,
                    n_expert,
                    gate_expert_bytes,
                    up_expert_bytes,
                    down_expert_bytes,
                    n_expert as u32,
                    eps,
                )
            });
        let b_buf = constant_u32(batch as u32);
        let gate_shared_off_buf = constant_u32(shared_gate_off);
        let up_shared_off_buf = constant_u32(shared_up_off);
        let down_shared_off_buf = constant_u32(shared_down_off);
        let silu_total_buf = constant_u32((batch * n_ff) as u32);
        ffn_chain::qwen_moe_decode_chain_encode_batched_shared_fused(
            ctx,
            enc,
            moe_carrier,
            shared_hidden,
            batch,
            &ffn_norm_w_buf,
            &router_w_buf,
            &router_off_buf,
            &shared_input_scale_buf,
            &gate_exps_w,
            gate_exps_off,
            &up_exps_w,
            up_exps_off,
            &down_exps_w,
            down_exps_off,
            &shared_gate_w,
            shared_gate_off,
            &shared_up_w,
            shared_up_off,
            &shared_down_w,
            shared_down_off,
            down_quant,
            shared_q8_0,
            &b_buf,
            &gate_shared_off_buf,
            &up_shared_off_buf,
            &down_shared_off_buf,
            &silu_total_buf,
        );
    }

    /// milestone 2: B lane 의 최종 hidden 을 모아 출력 프로젝션(rms_norm → vocab GEMV → argmax)을
    /// **B-column GEMV 로 weight 1회 읽기**로 계산한다. 기존 per-lane 경로는 vocab weight 를 B 번
    /// 읽었는데(가장 큰 행렬 중 하나), 여기서 1번으로 amortize. 단일 command buffer.
    #[cfg(target_os = "macos")]
    fn decode_chain_batched_output_argmax(
        &self,
        hidden: &[f32],
        batch: usize,
        hidden_dim: usize,
        tail: &DecodeOutputArgmaxSpecRef<'_>,
        reports: &mut [DecodeChainReport],
    ) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(
            tail.cols, hidden_dim,
            "batched output tail cols must match hidden dim"
        );
        assert_eq!(
            tail.norm_weight.len(),
            hidden_dim,
            "batched output norm weight length mismatch"
        );
        assert!(
            matches!(tail.output_quant, 0 | 2),
            "batched output tail supports Q4_K/Q6_K only"
        );
        assert!(tail.rows > 0, "batched output tail rows must be > 0");

        // vocab weight NoCopy resident wrap (self.resident 재사용 — lane 간/토큰 간 1회 업로드).
        let (out_w, out_off) = {
            let mut r = self.resident.borrow_mut();
            let e = r
                .entry(resident_key(tail.output_raw))
                .or_insert_with(|| compute::wrap_nocopy(ctx, tail.output_raw));
            (e.0.clone(), e.1)
        };
        let hidden_dev = ffn_chain::shared_f32_buf(ctx, hidden); // [batch*hidden_dim]
        let normed = ffn_chain::empty_f32_buf(ctx, batch * hidden_dim);
        let logits = ffn_chain::empty_f32_buf(ctx, batch * tail.rows);
        let norm_w_buf = ffn_chain::shared_f32_buf(ctx, tail.norm_weight);
        let dim_buf = ffn_chain::u32_buf(ctx, hidden_dim as u32);
        let eps_buf = ffn_chain::f32_buf(ctx, tail.eps);
        let rows_buf = ffn_chain::u32_buf(ctx, tail.rows as u32);
        let cols_buf = ffn_chain::u32_buf(ctx, tail.cols as u32);
        let out_off_buf = ffn_chain::u32_buf(ctx, out_off);
        let b_buf = ffn_chain::u32_buf(ctx, batch as u32);
        let token_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = (0..batch)
            .map(|_| {
                ctx.device
                    .newBufferWithLength_options(
                        std::mem::size_of::<u32>(),
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .expect("Metal: batched output token buffer")
            })
            .collect();

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = compute::chain_compute_encoder(ctx, &cmd);
        // 컬럼(lane)별 rms_norm → normed[lane*hidden_dim..] (column-major-by-lane 배치).
        for lane in 0..batch {
            ffn_chain::encode_rms_norm_io_offset(
                ctx,
                &enc,
                &hidden_dev,
                lane * hidden_dim * std::mem::size_of::<f32>(),
                &norm_w_buf,
                &normed,
                lane * hidden_dim * std::mem::size_of::<f32>(),
                &dim_buf,
                &eps_buf,
            );
        }
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        // vocab weight 1회 읽기: out_w × normed[B*hidden_dim] → logits[B*rows].
        compute::encode_gemv_quant_bcol(
            ctx,
            &enc,
            tail.output_quant,
            &out_w,
            &normed,
            &logits,
            &rows_buf,
            &cols_buf,
            &out_off_buf,
            &b_buf,
            tail.rows,
        );
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        for lane in 0..batch {
            compute::encode_argmax_f32_at(
                ctx,
                &enc,
                &logits,
                lane * tail.rows * std::mem::size_of::<f32>(),
                &token_bufs[lane],
                &rows_buf,
            );
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        for (lane, report) in reports.iter_mut().enumerate().take(batch) {
            let token = unsafe { *(token_bufs[lane].contents().as_ptr() as *const u32) };
            report.argmax_only = true;
            report.output_argmax = OutputArgmaxReport {
                attempted: true,
                did_run: true,
                token_id: Some(token),
                cmd_buffers: 0,
                wait_us: 0,
                readback_bytes: std::mem::size_of::<u32>(),
                fallback_reason: None,
            };
        }
    }
}

impl Backend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::Metal).with_op(BackendOp::MatMul)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        if self.capabilities().supports(request.op()) {
            Ok(BackendOutput::new(request.op()))
        } else {
            Err(BackendError::unsupported(self.kind(), request.op()))
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use std::ffi::OsString;

    #[cfg(target_os = "macos")]
    use crate::METAL_TEST_ENV_LOCK as ENV_LOCK;

    #[cfg(target_os = "macos")]
    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    #[cfg(target_os = "macos")]
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_resident_key_includes_slice_length() {
        let data = [7u8; 32];
        let short = &data[..16];
        let long = &data[..32];

        assert_eq!(short.as_ptr(), long.as_ptr());
        assert_ne!(resident_key(short), resident_key(long));
    }

    #[test]
    fn qwen_moe_combined_shared_sparse_routes_prefixes_shared_slots() {
        let shared_tokens = [0u32, 1, 2];
        let shared_weights = [0.7f32, 0.8, 0.9];
        let sparse_tokens = [2u32, 0, 1, 2];
        let sparse_weights = [0.1f32, 0.2, 0.3, 0.4];

        let (tokens, weights, sparse_base) = super::qwen_moe_combined_shared_sparse_routes(
            &shared_tokens,
            &shared_weights,
            &sparse_tokens,
            &sparse_weights,
        );

        assert_eq!(sparse_base, 3);
        assert_eq!(tokens, [0, 1, 2, 2, 0, 1, 2]);
        assert_eq!(weights, [0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_prefill_shared_combined_defaults_on_with_falsey_opt_out() {
        let _lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = "RNB_METAL_QWEN35_MOE_PREFILL_SHARED_COMBINED";

        let _unset = EnvGuard::remove(key);
        assert!(super::qwen_moe_prefill_shared_combined_requested());
        drop(_unset);

        for value in ["0", "false", "off", "no"] {
            let _guard = EnvGuard::set(key, value);
            assert!(!super::qwen_moe_prefill_shared_combined_requested());
        }

        let _guard = EnvGuard::set(key, "1");
        assert!(super::qwen_moe_prefill_shared_combined_requested());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_prefill_uploads_norm_for_id_primitive() {
        assert!(!super::qwen_moe_prefill_should_upload_norm(
            false, false, false, false, false, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            false, false, false, true, false, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            true, false, false, false, false, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            false, true, false, false, false, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            false, false, true, false, false, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            false, false, false, false, true, false
        ));
        assert!(super::qwen_moe_prefill_should_upload_norm(
            false, false, false, false, false, true
        ));
    }

    #[cfg(target_os = "macos")]
    struct QwenMoeLlamaIdFixture {
        gate_all: Vec<u8>,
        up_all: Vec<u8>,
        down_all: Vec<u8>,
        shared_gate: Vec<u8>,
        shared_up: Vec<u8>,
        shared_down: Vec<u8>,
        norm_all: Vec<f32>,
        down_quant: QwenMoeLlamaIdQuant,
        shared_quant: QwenMoeLlamaIdQuantSet,
        down_expert_bytes: usize,
    }

    #[cfg(target_os = "macos")]
    impl QwenMoeLlamaIdFixture {
        fn new(down_quant: QwenMoeLlamaIdQuant) -> Self {
            Self::with_shared(
                down_quant,
                QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q8Zero,
                    up: QwenMoeLlamaIdQuant::Q8Zero,
                    down: QwenMoeLlamaIdQuant::Q8Zero,
                },
            )
        }

        fn with_shared(
            down_quant: QwenMoeLlamaIdQuant,
            shared_quant: QwenMoeLlamaIdQuantSet,
        ) -> Self {
            const DIM: usize = 256;
            const N_EXPERT: usize = 2;
            let weight_bytes = |quant| {
                super::qwen_moe_llama_id_weight_bytes(quant, DIM, DIM)
                    .expect("fixture quant layout")
            };
            let gate_expert_bytes = weight_bytes(QwenMoeLlamaIdQuant::Q4K);
            let down_expert_bytes = weight_bytes(down_quant);
            Self {
                gate_all: vec![0; N_EXPERT * gate_expert_bytes],
                up_all: vec![0; N_EXPERT * gate_expert_bytes],
                down_all: vec![0; N_EXPERT * down_expert_bytes],
                shared_gate: vec![0; weight_bytes(shared_quant.gate)],
                shared_up: vec![0; weight_bytes(shared_quant.up)],
                shared_down: vec![0; weight_bytes(shared_quant.down)],
                norm_all: vec![0.0; 2 * DIM],
                down_quant,
                shared_quant,
                down_expert_bytes,
            }
        }

        fn request(&self) -> QwenMoeLlamaIdPrefillRequest<'_> {
            QwenMoeLlamaIdPrefillRequest {
                gate_all: &self.gate_all,
                up_all: &self.up_all,
                down_all: &self.down_all,
                gate_expert_bytes: 256 * 144,
                up_expert_bytes: 256 * 144,
                down_expert_bytes: self.down_expert_bytes,
                selected_experts: &[0, 1, 1, 0],
                route_weights: &[0.7, 0.3, 0.6, 0.4],
                shared_gate: &self.shared_gate,
                shared_up: &self.shared_up,
                shared_down: &self.shared_down,
                shared_route_weights: &[0.5, 0.25],
                sparse_quant: QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: self.down_quant,
                },
                shared_quant: self.shared_quant,
                norm_all: &self.norm_all,
                seq_len: 2,
                hidden_dim: 256,
                ffn_dim: 256,
                layer_idx: 7,
            }
        }

        fn chain_spec(&self) -> QwenMoePrefillBackendSpecRef<'_> {
            QwenMoePrefillBackendSpecRef {
                ffn_norm_w: &self.norm_all[..256],
                norm_eps: 1.0e-6,
                router_w: &self.norm_all,
                gate_all: &self.gate_all,
                up_all: &self.up_all,
                down_all: &self.down_all,
                gate_expert_bytes: 256 * 144,
                up_expert_bytes: 256 * 144,
                down_expert_bytes: self.down_expert_bytes,
                shared_input_scale: &self.norm_all[..256],
                shared_gate: &self.shared_gate,
                shared_up: &self.shared_up,
                shared_down: &self.shared_down,
                sparse_quant: QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: self.down_quant,
                },
                shared_quant: self.shared_quant,
                route_algorithm: QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
                n_expert: 2,
                n_expert_used: 2,
                hidden_dim: 256,
                ffn_dim: 256,
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_preflight_accepts_actual_and_existing_product_tuples() {
        let q4_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q4K,
        };
        let q6_shared = QwenMoeLlamaIdQuantSet {
            down: QwenMoeLlamaIdQuant::Q6K,
            ..q4_shared
        };
        let q8_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        };
        for (down_quant, shared_quant) in [
            (QwenMoeLlamaIdQuant::Q4K, q4_shared),
            (QwenMoeLlamaIdQuant::Q6K, q6_shared),
            (QwenMoeLlamaIdQuant::Q5K, q8_shared),
            (QwenMoeLlamaIdQuant::Q6K, q8_shared),
        ] {
            let fixture = QwenMoeLlamaIdFixture::with_shared(down_quant, shared_quant);
            let ffn_chain::QwenMoeLlamaIdPreflight::Run(plan) =
                super::qwen_moe_llama_id_prepare(&fixture.request(), true, usize::MAX)
                    .expect("valid product")
            else {
                panic!("supported product must run");
            };
            assert_eq!(plan.n_tokens, 2);
            assert_eq!(plan.n_expert, 2);
            assert_eq!(plan.n_expert_used, 2);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_prefill_chain_moe_preflight_accepts_exact_q8_and_q4_shared_tuples() {
        let backend = MetalBackend::new();
        let ctx = backend.ctx.as_ref().expect("Metal context");
        let q8_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        };
        let q4_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q6K,
        };
        for fixture in [
            QwenMoeLlamaIdFixture::with_shared(QwenMoeLlamaIdQuant::Q5K, q8_shared),
            QwenMoeLlamaIdFixture::with_shared(QwenMoeLlamaIdQuant::Q6K, q4_shared),
        ] {
            assert!(
                MetalBackend::qwen_prefill_chain_moe_preflight(ctx, 2, fixture.chain_spec())
                    .expect("valid chain MoE preflight")
                    .is_some()
            );
        }

        let mixed_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        };
        let mixed = QwenMoeLlamaIdFixture::with_shared(QwenMoeLlamaIdQuant::Q5K, mixed_shared);
        assert!(
            MetalBackend::qwen_prefill_chain_moe_preflight(ctx, 2, mixed.chain_spec())
                .expect("mixed chain MoE preflight")
                .is_none()
        );
    }

    #[cfg(target_os = "macos")]
    fn qwen_prefill_chain_dummy_spec(layer_idx: usize) -> QwenPrefillChainSpecRef<'static> {
        let weight = PrefillAtnCoreWeightView {
            raw: &[],
            quant: TensoropsQuant::Q4K,
            rows: 0,
            cols: 0,
        };
        QwenPrefillChainSpecRef::Attention {
            layer_idx,
            core: PrefillAtnOTailBackendSpecRef {
                core: PrefillAtnCoreBackendSpecRef {
                    attn_norm_w: &[],
                    q_norm_w: &[],
                    k_norm_w: &[],
                    q_weight: weight,
                    k_weight: weight,
                    v_weight: weight,
                    seq_len: 0,
                    num_heads: 0,
                    num_kv_heads: 0,
                    head_dim: 0,
                    hidden_dim: 0,
                    q_dim: 0,
                    kv_dim: 0,
                    n_rot: 0,
                    rope_theta: 0.0,
                    scale: 0.0,
                    norm_eps: 0.0,
                    pos_start: 0,
                },
                o_weight: weight,
            },
            moe: QwenMoePrefillBackendSpecRef {
                ffn_norm_w: &[],
                norm_eps: 1.0e-6,
                router_w: &[],
                gate_all: &[],
                up_all: &[],
                down_all: &[],
                gate_expert_bytes: 0,
                up_expert_bytes: 0,
                down_expert_bytes: 0,
                shared_input_scale: &[],
                shared_gate: &[],
                shared_up: &[],
                shared_down: &[],
                sparse_quant: QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: QwenMoeLlamaIdQuant::Q5K,
                },
                shared_quant: QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: QwenMoeLlamaIdQuant::Q6K,
                },
                route_algorithm: QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
                n_expert: 0,
                n_expert_used: 0,
                hidden_dim: 0,
                ffn_dim: 0,
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn qwen_prefill_chain_attention_boundary_spec(
        norm_eps: f32,
        rope_theta: f32,
        scale: f32,
        n_rot: usize,
        head_dim: usize,
    ) -> QwenPrefillChainSpecRef<'static> {
        let mut spec = qwen_prefill_chain_dummy_spec(0);
        let QwenPrefillChainSpecRef::Attention { core, .. } = &mut spec else {
            unreachable!("dummy Qwen chain spec is attention")
        };
        core.core.seq_len = 1;
        core.core.num_heads = 1;
        core.core.num_kv_heads = 1;
        core.core.head_dim = head_dim;
        core.core.hidden_dim = 256;
        core.core.q_dim = head_dim;
        core.core.kv_dim = head_dim;
        core.core.n_rot = n_rot;
        core.core.rope_theta = rope_theta;
        core.core.scale = scale;
        core.core.norm_eps = norm_eps;
        spec
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_prefill_chain_attention_boundary_rejects_before_allocation_or_submit() {
        let backend = MetalBackend::new();
        assert!(backend.ctx.is_some(), "Metal context");
        let cache_counts = || {
            (
                backend.resident.borrow().len(),
                backend.qwen_prefill_atn_o_tail_carriers.borrow().len(),
                backend.qwen_prefill_chain_moe_carrier_pool.borrow().len(),
            )
        };
        let before = cache_counts();
        let cases = [
            (
                "non-finite norm epsilon",
                qwen_prefill_chain_attention_boundary_spec(f32::NAN, 1_000_000.0, 0.0625, 256, 256),
            ),
            (
                "non-positive norm epsilon",
                qwen_prefill_chain_attention_boundary_spec(0.0, 1_000_000.0, 0.0625, 256, 256),
            ),
            (
                "non-finite rope theta",
                qwen_prefill_chain_attention_boundary_spec(1.0e-6, f32::NAN, 0.0625, 256, 256),
            ),
            (
                "non-positive rope theta",
                qwen_prefill_chain_attention_boundary_spec(1.0e-6, 0.0, 0.0625, 256, 256),
            ),
            (
                "non-finite attention scale",
                qwen_prefill_chain_attention_boundary_spec(
                    1.0e-6,
                    1_000_000.0,
                    f32::INFINITY,
                    256,
                    256,
                ),
            ),
            (
                "odd rotary dimension",
                qwen_prefill_chain_attention_boundary_spec(1.0e-6, 1_000_000.0, 0.0625, 255, 256),
            ),
            (
                "unsupported head dimension",
                qwen_prefill_chain_attention_boundary_spec(1.0e-6, 1_000_000.0, 0.0625, 128, 128),
            ),
        ];
        let hidden = vec![0.0; 256];
        for (label, spec) in cases {
            assert!(
                backend
                    .qwen_prefill_chain_run(&hidden, &[spec])
                    .unwrap_or_else(|error| panic!(
                        "{label} must be unsupported, not malformed: {error}"
                    ))
                    .is_none(),
                "{label} must fall back before command submission"
            );
            assert_eq!(
                cache_counts(),
                before,
                "{label} mutated backend state before fallback"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_prefill_chain_facade_returns_none_for_rank_above_product_limit() {
        let fixture = tests_fixture::qwen_moe_llama_prefill_layer_chain_fixture();
        let backend = MetalBackend::new();
        assert!(backend.ctx.is_some(), "Metal context");
        let mut spec = fixture.attention_spec(0);
        let QwenPrefillChainSpecRef::Attention { moe, .. } = &mut spec else {
            unreachable!("fixture spec is attention")
        };
        moe.n_expert = 9;
        moe.n_expert_used = 9;
        let before = (
            backend.resident.borrow().len(),
            backend.qwen_prefill_atn_o_tail_carriers.borrow().len(),
            backend.qwen_prefill_chain_moe_carrier_pool.borrow().len(),
        );
        assert!(
            backend
                .qwen_prefill_chain_run(&fixture.hidden, &[spec])
                .expect("rank 9 is unsupported, not malformed")
                .is_none(),
            "rank 9 must fall back at the public backend facade"
        );
        assert_eq!(
            (
                backend.resident.borrow().len(),
                backend.qwen_prefill_atn_o_tail_carriers.borrow().len(),
                backend.qwen_prefill_chain_moe_carrier_pool.borrow().len(),
            ),
            before,
            "rank 9 mutated backend state before fallback"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_prefill_chain_rejects_duplicate_layer_indices_before_preflight() {
        let backend = MetalBackend::new();
        let specs = [
            qwen_prefill_chain_dummy_spec(3),
            qwen_prefill_chain_dummy_spec(3),
        ];
        let error = backend
            .qwen_prefill_chain_run(&[], &specs)
            .expect_err("duplicate layer indices must be malformed");
        assert!(error.contains("duplicate layer index 3"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_prefill_chain_returns_none_for_noncontiguous_layer_indices_before_preflight() {
        let backend = MetalBackend::new();
        let specs = [
            qwen_prefill_chain_dummy_spec(3),
            qwen_prefill_chain_dummy_spec(5),
        ];
        assert!(backend
            .qwen_prefill_chain_run(&[], &specs)
            .expect("noncontiguous range is unsupported")
            .is_none());
    }

    #[cfg(target_os = "macos")]
    fn assert_qwen_prefill_chain_f32_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label} bit mismatch at index {index}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn assert_qwen_prefill_chain_owned_once(output: &QwenPrefillChainBackendOut) {
        assert_eq!(output.hidden_uploads, 1, "hidden must upload exactly once");
        assert_eq!(
            output.hidden_readbacks, 1,
            "hidden must read back exactly once"
        );
        assert_eq!(
            output.intermediate_hidden_transfers, 0,
            "hidden must remain device-resident between layers"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_layer_chain_two_layer_one_shot_matches_sequential_bits() {
        const ATTN_LAYER: usize = 12;
        const GDN_LAYER: usize = 13;

        let fixture = tests_fixture::qwen_moe_llama_prefill_layer_chain_fixture();
        let combined_backend = MetalBackend::new();
        let combined_specs = [
            fixture.attention_spec(ATTN_LAYER),
            fixture.gdn_spec(GDN_LAYER),
        ];
        let combined = match combined_backend
            .qwen_prefill_chain_run(&fixture.hidden, &combined_specs)
            .expect("valid two-layer Qwen prefill chain")
        {
            Some(output) => output,
            None => {
                eprintln!("skipped: required Metal Qwen prefill chain capability is unavailable");
                return;
            }
        };

        let sequential_backend = MetalBackend::new();
        let attention_spec = [fixture.attention_spec(ATTN_LAYER)];
        let attention = sequential_backend
            .qwen_prefill_chain_run(&fixture.hidden, &attention_spec)
            .expect("valid sequential attention layer")
            .expect("capability accepted by the combined run");
        let gdn_spec = [fixture.gdn_spec(GDN_LAYER)];
        let gdn = sequential_backend
            .qwen_prefill_chain_run(&attention.hidden, &gdn_spec)
            .expect("valid sequential GDN layer seeded from attention hidden")
            .expect("capability accepted by the combined run");

        assert_qwen_prefill_chain_owned_once(&combined);
        assert_qwen_prefill_chain_owned_once(&attention);
        assert_qwen_prefill_chain_owned_once(&gdn);
        assert_eq!(combined.attention_kv.len(), 1);
        assert_eq!(combined.gdn_states.len(), 1);
        assert_eq!(attention.attention_kv.len(), 1);
        assert!(attention.gdn_states.is_empty());
        assert!(gdn.attention_kv.is_empty());
        assert_eq!(gdn.gdn_states.len(), 1);

        let (combined_attn_layer, combined_k, combined_v) = &combined.attention_kv[0];
        let (sequential_attn_layer, sequential_k, sequential_v) = &attention.attention_kv[0];
        assert_eq!(*combined_attn_layer, ATTN_LAYER);
        assert_eq!(*sequential_attn_layer, ATTN_LAYER);
        assert_eq!(combined_k, sequential_k, "attention K bits drifted");
        assert_eq!(combined_v, sequential_v, "attention V bits drifted");

        let (combined_gdn_layer, combined_conv, combined_delta) = &combined.gdn_states[0];
        let (sequential_gdn_layer, sequential_conv, sequential_delta) = &gdn.gdn_states[0];
        assert_eq!(*combined_gdn_layer, GDN_LAYER);
        assert_eq!(*sequential_gdn_layer, GDN_LAYER);
        assert_qwen_prefill_chain_f32_bits_eq(
            combined_conv,
            sequential_conv,
            "GDN final conv state",
        );
        assert_qwen_prefill_chain_f32_bits_eq(
            combined_delta,
            sequential_delta,
            "GDN final delta state",
        );
        assert_qwen_prefill_chain_f32_bits_eq(
            &combined.hidden,
            &gdn.hidden,
            "two-layer final hidden",
        );
        assert!(
            combined
                .hidden
                .iter()
                .zip(&fixture.hidden)
                .any(|(&actual, &seed)| actual.to_bits() != seed.to_bits()),
            "real quantized layers must change hidden values"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_layer_chain_40_contiguous_layers_keep_single_ownership() {
        const LAYER_COUNT: usize = 40;
        const FIRST_LAYER: usize = 20;

        let fixture = tests_fixture::qwen_moe_llama_prefill_layer_chain_fixture();
        let specs = fixture.contiguous_specs(FIRST_LAYER, LAYER_COUNT);
        let backend = MetalBackend::new();
        let output = match backend
            .qwen_prefill_chain_run(&fixture.hidden, &specs)
            .expect("valid 40-layer Qwen prefill chain")
        {
            Some(output) => output,
            None => {
                eprintln!("skipped: required Metal Qwen prefill chain capability is unavailable");
                return;
            }
        };

        assert_qwen_prefill_chain_owned_once(&output);
        assert_eq!(
            output.hidden.len(),
            tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::SEQ_LEN
                * tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM
        );
        assert_eq!(output.attention_kv.len(), LAYER_COUNT / 2);
        assert_eq!(output.gdn_states.len(), LAYER_COUNT / 2);
        assert_eq!(
            output
                .attention_kv
                .iter()
                .map(|(layer_idx, _, _)| *layer_idx)
                .collect::<Vec<_>>(),
            (FIRST_LAYER..FIRST_LAYER + LAYER_COUNT)
                .step_by(2)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output
                .gdn_states
                .iter()
                .map(|(layer_idx, _, _)| *layer_idx)
                .collect::<Vec<_>>(),
            (FIRST_LAYER + 1..FIRST_LAYER + LAYER_COUNT)
                .step_by(2)
                .collect::<Vec<_>>()
        );
        for (_, k, v) in &output.attention_kv {
            assert_eq!(
                k.len(),
                tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::expected_kv_len()
            );
            assert_eq!(
                v.len(),
                tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::expected_kv_len()
            );
        }
        for (_, conv, delta) in &output.gdn_states {
            assert_eq!(
                conv.len(),
                tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::expected_conv_state_len()
            );
            assert_eq!(
                delta.len(),
                tests_fixture::QwenMoeLlamaPrefillLayerChainFixture::expected_delta_state_len()
            );
        }
        assert!(output.hidden.iter().all(|value| value.is_finite()));
        assert!(
            output
                .hidden
                .iter()
                .zip(&fixture.hidden)
                .any(|(&actual, &seed)| actual.to_bits() != seed.to_bits()),
            "40 real quantized layers must change hidden values"
        );

        assert_eq!(
            backend.qwen_prefill_atn_o_tail_carriers.borrow().len(),
            1,
            "same-shape attention layers must reuse one carrier"
        );
        assert_eq!(
            backend.qwen_prefill_gdn_carriers.borrow().len(),
            1,
            "same-shape GDN layers must reuse one carrier"
        );
        let moe_pool = backend.qwen_prefill_chain_moe_carrier_pool.borrow();
        assert_eq!(
            moe_pool.len(),
            1,
            "shape-compatible quant tuples must reuse one carrier plan"
        );
        assert_eq!(
            moe_pool
                .values()
                .map(|carriers| carriers.len())
                .sum::<usize>(),
            1,
            "carrier pool must not allocate one full carrier per layer"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_prefill_reads_page_offset_weights_and_reuses_residents() {
        const OFFSET: usize = std::mem::align_of::<u16>();
        const METAL_PAGE: usize = 16384;
        let _env_lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _llama_id = EnvGuard::set("RNB_METAL_QWEN35_MOE_LLAMA_ID", "1");
        let backend = MetalBackend::new();
        let fixture = QwenMoeLlamaIdFixture::new(QwenMoeLlamaIdQuant::Q5K);
        let mapped = |raw: &[u8]| {
            let mut map = memmap2::MmapMut::map_anon(OFFSET + raw.len()).expect("anonymous mmap");
            map[..OFFSET].fill(0xff);
            map[OFFSET..].copy_from_slice(raw);
            map
        };
        let gate_all = mapped(&fixture.gate_all);
        let up_all = mapped(&fixture.up_all);
        let down_all = mapped(&fixture.down_all);
        let shared_gate = mapped(&fixture.shared_gate);
        let shared_up = mapped(&fixture.shared_up);
        let shared_down = mapped(&fixture.shared_down);
        let raw_weights = [
            &gate_all[OFFSET..],
            &up_all[OFFSET..],
            &down_all[OFFSET..],
            &shared_gate[OFFSET..],
            &shared_up[OFFSET..],
            &shared_down[OFFSET..],
        ];
        for raw in raw_weights {
            assert_eq!(
                raw.as_ptr() as usize & (METAL_PAGE - 1),
                OFFSET,
                "mmap weight must exercise a nonzero page offset"
            );
        }
        let request = || {
            let mut request = fixture.request();
            request.gate_all = &gate_all[OFFSET..];
            request.up_all = &up_all[OFFSET..];
            request.down_all = &down_all[OFFSET..];
            request.shared_gate = &shared_gate[OFFSET..];
            request.shared_up = &shared_up[OFFSET..];
            request.shared_down = &shared_down[OFFSET..];
            request
        };

        assert!(backend.resident.borrow().is_empty());
        for iteration in 0..2 {
            let output = backend
                .qwen_moe_llama_id_prefill(request())
                .expect("llama-ID prefill")
                .expect("llama-ID path must run");
            assert!(
                output.values.iter().all(|&value| value == 0.0),
                "prefixed bytes must be skipped by the exact NoCopy page offset"
            );
            assert_eq!(
                backend.resident.borrow().len(),
                raw_weights.len(),
                "iteration {iteration} must reuse the six raw residents"
            );
        }
        let resident = backend.resident.borrow();
        for raw in raw_weights {
            let (_, offset) = resident
                .get(&resident_key(raw))
                .expect("raw weight resident");
            assert_eq!(*offset as usize, OFFSET);
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_carrier_pool_allocates_once_and_refreshes_routes_exactly() {
        const DIM: usize = 256;
        let q4 = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q4K,
        };
        let mut fixture = QwenMoeLlamaIdFixture::with_shared(QwenMoeLlamaIdQuant::Q4K, q4);
        let expert_rows = |first: f32, second: f32| {
            (0..2 * DIM)
                .flat_map(|row| {
                    let value = if row < DIM { first } else { second };
                    std::iter::repeat_n(value, DIM)
                })
                .collect::<Vec<_>>()
        };
        let shared_rows = |value: f32| vec![value; DIM * DIM];
        fixture.gate_all = quantize_rows_q4k(&expert_rows(0.01, 0.025), 2 * DIM, DIM);
        fixture.up_all = quantize_rows_q4k(&expert_rows(0.015, 0.03), 2 * DIM, DIM);
        fixture.down_all = quantize_rows_q4k(&expert_rows(0.02, 0.04), 2 * DIM, DIM);
        fixture.shared_gate = quantize_rows_q4k(&shared_rows(0.0125), DIM, DIM);
        fixture.shared_up = quantize_rows_q4k(&shared_rows(0.0175), DIM, DIM);
        fixture.shared_down = quantize_rows_q4k(&shared_rows(0.0225), DIM, DIM);
        for (index, value) in fixture.norm_all.iter_mut().enumerate() {
            let token_scale = if index < DIM { 0.5 } else { 0.75 };
            *value = token_scale * ((index % DIM + 1) as f32 / DIM as f32);
        }

        let selected_a = [0, 1, 0, 1];
        let weights_a = [0.8, 0.2, 0.65, 0.35];
        let shared_a = [0.4, 0.55];
        let selected_b = [1, 0, 1, 0];
        let weights_b = [0.9, 0.1, 0.25, 0.75];
        let shared_b = [0.2, 0.7];
        let request = |selected_experts, route_weights, shared_route_weights| {
            let mut request = fixture.request();
            request.selected_experts = selected_experts;
            request.route_weights = route_weights;
            request.shared_route_weights = shared_route_weights;
            request
        };
        let ffn_chain::QwenMoeLlamaIdPreflight::Run(plan) = super::qwen_moe_llama_id_prepare(
            &request(&selected_a, &weights_a, &shared_a),
            true,
            usize::MAX,
        )
        .expect("valid pool test plan") else {
            panic!("pool test shape must run");
        };
        let run = |backend: &MetalBackend, selected, weights, shared| {
            backend
                .qwen_moe_llama_id_prefill_requested(request(selected, weights, shared), true)
                .expect("llama-ID pooled prefill")
                .expect("llama-ID path must run")
                .values
        };

        let backend = MetalBackend::new();
        let output_a = run(&backend, &selected_a, &weights_a, &shared_a);
        {
            let pool = backend.qwen_moe_llama_id_carrier_pool.borrow();
            assert_eq!(pool.values().map(Vec::len).sum::<usize>(), 1);
            let carriers = pool.get(&plan).expect("full-shape pool entry");
            assert_eq!(carriers.len(), 1);
            assert_eq!(
                carriers[0].debug_routes(),
                (selected_a.to_vec(), weights_a.to_vec(), shared_a.to_vec())
            );
        }

        let output_b = run(&backend, &selected_b, &weights_b, &shared_b);
        {
            let pool = backend.qwen_moe_llama_id_carrier_pool.borrow();
            assert_eq!(
                pool.values().map(Vec::len).sum::<usize>(),
                1,
                "steady repeated shape must not allocate another carrier"
            );
            let carriers = pool.get(&plan).expect("reused full-shape pool entry");
            assert_eq!(carriers.len(), 1);
            assert_eq!(
                carriers[0].debug_routes(),
                (selected_b.to_vec(), weights_b.to_vec(), shared_b.to_vec())
            );
        }

        let fresh = MetalBackend::new();
        let fresh_b = run(&fresh, &selected_b, &weights_b, &shared_b);
        assert_eq!(
            output_b
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fresh_b
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reused scratch and refreshed routes must match a fresh carrier exactly"
        );
        assert!(
            output_a
                .iter()
                .zip(&output_b)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "different expert and shared routes must produce different output"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_fallbacks_stay_presubmit() {
        let fixture = QwenMoeLlamaIdFixture::new(QwenMoeLlamaIdQuant::Q5K);
        assert_eq!(
            super::qwen_moe_llama_id_prepare(&fixture.request(), false, usize::MAX)
                .expect("pipeline fallback"),
            ffn_chain::QwenMoeLlamaIdPreflight::Fallback(
                ffn_chain::QwenMoeLlamaIdFallbackReason::PipelineUnavailable,
            )
        );
        assert_eq!(
            super::qwen_moe_llama_id_prepare(&fixture.request(), true, 0).expect("budget fallback"),
            ffn_chain::QwenMoeLlamaIdPreflight::Fallback(
                ffn_chain::QwenMoeLlamaIdFallbackReason::ScratchOverBudget,
            )
        );

        let mut unsupported = fixture.request();
        unsupported.sparse_quant.gate = QwenMoeLlamaIdQuant::Q5K;
        assert_eq!(
            super::qwen_moe_llama_id_prepare(&unsupported, true, usize::MAX)
                .expect("unsupported fallback"),
            ffn_chain::QwenMoeLlamaIdPreflight::Fallback(
                ffn_chain::QwenMoeLlamaIdFallbackReason::UnsupportedQuant,
            )
        );

        let mut position_invalid = fixture.request();
        position_invalid.shared_quant.gate = QwenMoeLlamaIdQuant::Q4K;
        assert_eq!(
            super::qwen_moe_llama_id_prepare(&position_invalid, true, usize::MAX)
                .expect("position fallback"),
            ffn_chain::QwenMoeLlamaIdPreflight::Fallback(
                ffn_chain::QwenMoeLlamaIdFallbackReason::UnsupportedQuant,
            )
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_trace_truthy_values_are_pure() {
        for value in ["1", "true", "on", "yes", "TRACE"] {
            assert!(super::qwen_moe_llama_id_trace_value_truthy(Some(value)));
        }
        for value in [None, Some("0"), Some("false"), Some("off"), Some("no")] {
            assert!(!super::qwen_moe_llama_id_trace_value_truthy(value));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_trace_reports_measured_stages_and_metadata() {
        let trace = ffn_chain::QwenMoeLlamaIdTrace {
            map_dispatches: 1,
            gate_dispatches: 2,
            up_dispatches: 2,
            activation_dispatches: 2,
            down_dispatches: 2,
            reduce_dispatches: 2,
            map_ms: Some(0.1),
            gate_ms: Some(0.2),
            up_ms: Some(0.3),
            activation_ms: Some(0.4),
            down_ms: Some(0.5),
            reduce_ms: Some(0.6),
            stage_timing_source: Some("metal_counter_timestamp"),
            scratch_bytes: 4096,
            ..ffn_chain::QwenMoeLlamaIdTrace::default()
        };
        let sparse_quant = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q5K,
        };
        let shared_quant = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        };
        let line = super::format_qwen_moe_llama_id_success_trace(
            trace,
            7,
            sparse_quant,
            shared_quant,
            Some(1.25),
            2.5,
            0.25,
        );
        assert!(line.contains("mode=llama_id fallback_reason=none"));
        assert!(line.contains("layer_idx=7"));
        assert!(line.contains("sparse_gate_quant=q4_k sparse_up_quant=q4_k sparse_down_quant=q5_k"));
        assert!(line.contains("shared_gate_quant=q8_0 shared_up_quant=q8_0 shared_down_quant=q8_0"));
        assert!(line.contains("kernel_dtype=f32 weights=raw"));
        assert!(line.contains("exclude_from_d20=true"));
        assert!(line.contains("stage_timing_source=metal_counter_timestamp"));
        assert!(line.contains("dispatch_count=11 map=1 gate=2 up=2"));
        assert!(line.contains("activation=2 down=2 reduce=2 scratch_bytes=4096"));
        assert!(line.contains("gpu_ms=1.250 wrapper_wall_ms=2.500 readback_ms=0.250"));
        assert!(line.contains("map_ms=0.100"));
        assert!(line.contains("reduce_ms=0.600"));

        let split = ffn_chain::QwenMoeLlamaIdTrace {
            stage_timing_source: Some("trace_split_command_buffers"),
            stage_timing_fallback_reason: Some("dispatch_boundary_unsupported"),
            ..trace
        };
        let split_line = super::format_qwen_moe_llama_id_success_trace(
            split,
            7,
            sparse_quant,
            shared_quant,
            Some(1.25),
            2.5,
            0.25,
        );
        assert!(split_line.contains("stage_timing_source=trace_split_command_buffers"));
        assert!(split_line.contains("stage_timing_fallback_reason=dispatch_boundary_unsupported"));
        assert!(split_line.contains("map_ms=0.100"));

        let unsupported = ffn_chain::QwenMoeLlamaIdTrace {
            stage_timing_unsupported_reason: Some("counter_resolve_failed"),
            ..ffn_chain::QwenMoeLlamaIdTrace::default()
        };
        let unsupported_line = super::format_qwen_moe_llama_id_success_trace(
            unsupported,
            7,
            sparse_quant,
            shared_quant,
            Some(1.25),
            2.5,
            0.25,
        );
        assert!(unsupported_line.contains("map_ms=unsupported(counter_resolve_failed)"));
        assert!(!unsupported_line.contains("map_ms=0"));

        let fallback =
            super::format_qwen_moe_llama_id_fallback_trace("scratch_over_budget", 0.0005);
        assert!(fallback.contains("fallback_reason=scratch_over_budget"));
        assert!(fallback.contains("gpu_ms=unmeasured"));
        assert!(fallback.contains("wrapper_wall_ms=<0.001"));
        assert!(!fallback.contains("wrapper_wall_ms=0"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_llama_id_invalid_inputs_and_execution_failures_are_errors() {
        let fixture = QwenMoeLlamaIdFixture::new(QwenMoeLlamaIdQuant::Q5K);

        let mut invalid_layout = fixture.request();
        invalid_layout.route_weights = &[];
        assert!(super::qwen_moe_llama_id_prepare(&invalid_layout, false, usize::MAX).is_err());

        let mut zero_rank = fixture.request();
        zero_rank.selected_experts = &[];
        zero_rank.route_weights = &[];
        assert!(super::qwen_moe_llama_id_prepare(&zero_rank, false, usize::MAX).is_err());

        let mut arena_mismatch = fixture.request();
        arena_mismatch.up_all = &fixture.up_all[..arena_mismatch.up_expert_bytes];
        assert!(super::qwen_moe_llama_id_prepare(&arena_mismatch, false, usize::MAX).is_err());

        let allocation_error =
            super::qwen_moe_llama_id_error(ffn_chain::QwenMoeLlamaIdError::AllocationFailed);
        assert!(allocation_error.contains("AllocationFailed"));
        let post_submit = super::qwen_moe_llama_id_command_result(
            MTLCommandBufferStatus::Error,
            Some("simulated".to_string()),
        );
        assert!(post_submit.is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_mulmmid_v4_trace_reports_rank_scatter_abi() {
        let scratch = ffn_chain::QwenMoeMulMmIdV4Scratch {
            gate_bytes: 2 * 1024 * 1024,
            up_bytes: 2 * 1024 * 1024,
            down_bytes: 0,
            metadata_bytes: 512 * 1024,
            total_bytes: 4 * 1024 * 1024 + 512 * 1024,
        };
        let trace = super::format_qwen_moe_mulmmid_v4_trace(
            true, 8, 4, 8192, 128, 2, 7, 5, 1.25, 2.5, 2.75, scratch,
        );
        assert!(trace.contains("mode=mulmmid_v4"));
        assert!(trace.contains("fallback_reason=none"));
        assert!(trace.contains("dispatch_count=5"));
        assert!(trace.contains("shared_preserved=true"));
        assert!(trace.contains("slots=8"));
        assert!(trace.contains("seq_len=4"));
        assert!(trace.contains("n_expert_used=2"));
        assert!(trace.contains("n_expert=128"));
        assert!(trace.contains("id_block_count=7"));
        assert!(trace.contains("rank_dispatch_count=2"));
        assert!(trace.contains("rank_block_count=5"));
        assert!(trace.contains("encode_wait_ms=2.750"));
        assert!(trace.contains("gate_up_scratch_mib=4.0"));
        assert!(trace.contains("metadata_scratch_mib=0.5"));
        assert!(trace.contains("down_scratch_mib=0.0"));
        assert!(trace.contains("removed_down_scratch_mib=0.2"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_moe_v4_attempt_state_skips_id_paths_only_after_requested_failure() {
        let mut requested = super::QwenMoeV4AttemptState::new(true);
        assert!(!requested.skip_id_paths());
        requested.record_preflight_failure();
        assert!(requested.skip_id_paths());

        let mut not_requested = super::QwenMoeV4AttemptState::new(false);
        not_requested.record_preflight_failure();
        assert!(!not_requested.skip_id_paths());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_mulmmid_v4_backend_try_smoke_q4_down() {
        let _env_lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let _llama_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_LLAMA_ID", "0");
        let backend = MetalBackend::new();
        let ctx = backend.ctx.as_ref().expect("no metal device");
        let hidden_dim = 256usize;
        let ffn_dim = 256usize;
        let seq_len = 1usize;
        let n_expert = 1usize;
        let gate_stride = ffn_dim * (hidden_dim / 256) * 144;
        let up_stride = gate_stride;
        let down_stride = hidden_dim * (ffn_dim / 256) * 144;
        let gate_raw = vec![0u8; gate_stride * n_expert];
        let up_raw = vec![0u8; up_stride * n_expert];
        let down_raw = vec![0u8; down_stride * n_expert];
        let (gate_wb, gate_off) = compute::wrap_nocopy(ctx, &gate_raw);
        let (up_wb, up_off) = compute::wrap_nocopy(ctx, &up_raw);
        let (down_wb, down_off) = compute::wrap_nocopy(ctx, &down_raw);
        let accum = ffn_chain::QwenMoePrefillAccumCarrier::new(ctx, hidden_dim, seq_len);
        accum.upload_norm_all(&vec![0.0f32; hidden_dim * seq_len]);
        let out = backend
            .try_qwen_moe_prefill_mulmmid_v4(
                ctx,
                &accum,
                &gate_wb,
                gate_off,
                &up_wb,
                up_off,
                &down_wb,
                down_off,
                gate_raw.len(),
                up_raw.len(),
                down_raw.len(),
                gate_stride,
                up_stride,
                down_stride,
                &[0],
                &[1.0],
                &[0],
                false,
                seq_len,
                hidden_dim,
                ffn_dim,
                false,
                false,
            )
            .expect("v4 q4 backend try");
        assert_eq!(out.len(), hidden_dim * seq_len);
    }

    #[test]
    fn metal_backend_advertises_matmul() {
        let backend = MetalBackend::new();
        assert!(backend.capabilities().supports(BackendOp::MatMul));
    }

    // pm33 Task 1: Q4_K batch GEMM(M>1) 커널 correctness 헬퍼/테스트.
    // CPU dequant reference 와 rel<1e-3 일치 — naive 커널이 dequant 1:1 인지 검증(튜닝은 별도).
    #[cfg(target_os = "macos")]
    fn det_vals(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * scale).collect()
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_rms_norm_batch_matches_cpu() {
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            eprintln!("[pm49] no Metal ctx; skip");
            return;
        };
        let rows = 3usize;
        let cols = 17usize;
        let eps = 1e-5f32;
        let input = det_vals(rows * cols, 0.031);
        let weight = det_vals(cols, 0.17);

        let in_buf = crate::ffn_chain::empty_f32_buf(ctx, rows * cols);
        let w_buf = crate::ffn_chain::empty_f32_buf(ctx, cols);
        let out_buf = crate::ffn_chain::empty_f32_buf(ctx, rows * cols);
        let cols_buf = crate::ffn_chain::u32_buf(ctx, cols as u32);
        let eps_buf = crate::ffn_chain::shared_f32_buf(ctx, &[eps]);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                in_buf.contents().as_ptr() as *mut f32,
                input.len(),
            );
            std::ptr::copy_nonoverlapping(
                weight.as_ptr(),
                w_buf.contents().as_ptr() as *mut f32,
                weight.len(),
            );
        }

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        crate::compute::encode_rms_norm_batch(
            ctx, &enc, &in_buf, &w_buf, &out_buf, &cols_buf, &eps_buf, rows,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = crate::ffn_chain::readback(&out_buf, rows * cols);
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let sum = input[base..base + cols].iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0f32 / (sum / cols as f32 + eps).sqrt();
            for c in 0..cols {
                expected[base + c] = input[base + c] * inv * weight[c];
            }
        }
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[pm49] rms_norm_batch max_abs={max_abs:.3e}");
        assert!(max_abs < 2e-5, "rms_norm_batch mismatch: {max_abs}");
    }

    /// n개 row 각각 `quantize_q4_k_vec` → packed bytes 연결 (weight[N,K] Q4_K).
    #[cfg(target_os = "macos")]
    fn quantize_rows_q4k(vals: &[f32], n: usize, k: usize) -> Vec<u8> {
        use rnb_cpu::quantize::quant::quantize_q4_k_vec;
        let mut out = Vec::new();
        for row in 0..n {
            out.extend(quantize_q4_k_vec(&vals[row * k..(row + 1) * k]));
        }
        out
    }

    /// weight[N,K] Q4_K bytes, input[M,K] f32 → out[M,N], out[tok*N+row] = dequant(w[row])·input[tok].
    #[cfg(target_os = "macos")]
    fn cpu_q4k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        tests_fixture::q4k_gemm_reference(wb, n, k, input, m)
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn gemm_q4k_batch_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        let (n, k, m) = (8usize, 512usize, 4usize); // K=512 = 2 superblocks
        let wb = quantize_rows_q4k(&det_vals(n * k, 0.05), n, k);
        let input = det_vals(m * k, 0.1);
        let cpu = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
        let gpu = crate::compute::gemm_q4k_with_ctx(&ctx, &wb, &input, n, k, m);
        assert_eq!(gpu.len(), m * n);
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn prefill_atn_o_tail_matches_core_plus_host_o_proj_oracle() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            eprintln!("[pm108] no Metal ctx; skipping ATN o-tail oracle");
            return;
        }

        let seq_len = 4usize;
        let hidden_dim = 256usize;
        let head_dim = 256usize;
        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let q_dim = 256usize;
        let kv_dim = 256usize;
        let n_rot = 64usize;
        let norm_eps = 1e-5f32;
        let rope_theta = 1_000_000.0f32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let pos_start = 0usize;

        let hidden = det_vals(seq_len * hidden_dim, 0.031);
        let attn_norm_w = det_vals(hidden_dim, 0.017)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>();
        let q_norm_w = det_vals(head_dim, 0.013)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>();
        let k_norm_w = det_vals(head_dim, 0.011)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>();
        let q_w = quantize_rows_q4k(
            &det_vals(q_dim * 2 * hidden_dim, 0.007),
            q_dim * 2,
            hidden_dim,
        );
        let k_w = quantize_rows_q4k(&det_vals(kv_dim * hidden_dim, 0.009), kv_dim, hidden_dim);
        let v_w = quantize_rows_q4k(&det_vals(kv_dim * hidden_dim, 0.005), kv_dim, hidden_dim);
        let o_w = quantize_rows_q4k(&det_vals(hidden_dim * q_dim, 0.006), hidden_dim, q_dim);

        let q_view = PrefillAtnCoreWeightView {
            raw: &q_w,
            quant: TensoropsQuant::Q4K,
            rows: q_dim * 2,
            cols: hidden_dim,
        };
        let k_view = PrefillAtnCoreWeightView {
            raw: &k_w,
            quant: TensoropsQuant::Q4K,
            rows: kv_dim,
            cols: hidden_dim,
        };
        let v_view = PrefillAtnCoreWeightView {
            raw: &v_w,
            quant: TensoropsQuant::Q4K,
            rows: kv_dim,
            cols: hidden_dim,
        };
        let o_view = PrefillAtnCoreWeightView {
            raw: &o_w,
            quant: TensoropsQuant::Q4K,
            rows: hidden_dim,
            cols: q_dim,
        };
        let core_req = PrefillAtnCoreBackendRequest {
            hidden: &hidden,
            attn_norm_w: &attn_norm_w,
            q_norm_w: &q_norm_w,
            k_norm_w: &k_norm_w,
            q_weight: q_view,
            k_weight: k_view,
            v_weight: v_view,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        };

        let Some((attn_out, core_k, core_v)) = backend
            .prefill_atn_core_if_supported(core_req)
            .expect("ATN core dispatch")
        else {
            eprintln!("[pm108] ATN core unsupported; skipping o-tail oracle");
            return;
        };
        let Some((o_tail_hidden, o_tail_k, o_tail_v)) = backend
            .prefill_atn_o_tail_if_supported(PrefillAtnOTailBackendRequest {
                core: core_req,
                o_weight: o_view,
            })
            .expect("ATN o-tail dispatch")
        else {
            eprintln!("[pm108] ATN o-tail unsupported; skipping o-tail oracle");
            return;
        };

        assert_eq!(o_tail_k, core_k, "o-tail K bits must match core");
        assert_eq!(o_tail_v, core_v, "o-tail V bits must match core");

        let o_input = attn_out
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        let o_proj = cpu_q4k_gemm_reference(&o_w, hidden_dim, q_dim, &o_input, seq_len);
        let expected_hidden = hidden
            .iter()
            .zip(o_proj.iter())
            .map(|(&h, &o)| h + o)
            .collect::<Vec<_>>();
        let max_abs = o_tail_hidden
            .iter()
            .zip(expected_hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = o_tail_hidden
            .iter()
            .zip(expected_hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / o_tail_hidden.len() as f32;
        eprintln!("[pm108] ATN o-tail oracle max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
        assert!(
            max_abs < 2.5e-1 && mean_abs < 2.0e-2,
            "ATN o-tail mismatch: max_abs={max_abs} mean_abs={mean_abs}"
        );
    }

    // pm33 튜닝(T10): Q4_K dequant-공유 커널이 naive 와 동일 출력(CPU ref rel<1e-3)인지.
    // superblock dequant 1번 공유라도 결과는 naive 와 같아야 함(reduction 순서만 차이).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn gemm_q4k_shared_batch_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        // M > tg_width(256)도 커버: M=300으로 grid-stride token 분할 확인.
        let (n, k, m) = (8usize, 512usize, 300usize);
        let wb = quantize_rows_q4k(&det_vals(n * k, 0.05), n, k);
        let input = det_vals(m * k, 0.1);
        let cpu = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
        let gpu = crate::compute::gemm_q4k_shared_with_ctx(&ctx, &wb, &input, n, k, m);
        assert_eq!(gpu.len(), m * n);
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    // pm33 Task 2: Q6_K batch GEMM(M>1) 커널 correctness.
    // f32→Q6_K 양자화 함수가 없어 raw byte 를 결정적으로 hand-build(make_q6k_block 패턴),
    // dequantize_q6_k 로 ground truth 생성 → GPU 와 rel<1e-3.

    /// 결정적 BlockQ6_K bytes 1개 조립 (tile_q6k.rs:make_q6k_block 패턴 복제).
    #[cfg(target_os = "macos")]
    fn make_q6k_block(d_val: f32, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        block[0..128].copy_from_slice(&ql);
        block[128..192].copy_from_slice(&qh);
        for (i, &s) in scales.iter().enumerate() {
            block[192 + i] = s as u8;
        }
        block[208..210].copy_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
        block
    }

    /// n개 row × nb개 block 의 Q6_K weight bytes 를 결정적으로 생성.
    #[cfg(target_os = "macos")]
    fn build_q6k_rows(n: usize, k: usize) -> Vec<u8> {
        let nb = k / 256;
        let mut out = Vec::with_capacity(n * nb * 210);
        for row in 0..n {
            for blk in 0..nb {
                let seed = row * nb + blk;
                let d_val = 0.02 + (seed % 7) as f32 * 0.004;
                let mut scales = [0i8; 16];
                for (i, s) in scales.iter_mut().enumerate() {
                    *s = (((seed + i) % 13) as i32 - 6) as i8;
                }
                let mut ql = [0u8; 128];
                for (i, q) in ql.iter_mut().enumerate() {
                    *q = ((seed * 3 + i * 5) % 256) as u8;
                }
                let mut qh = [0u8; 64];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = ((seed * 7 + i * 11) % 256) as u8;
                }
                out.extend(make_q6k_block(d_val, scales, ql, qh));
            }
        }
        out
    }

    /// weight[N,K] Q6_K bytes, input[M,K] f32 → out[M,N], dequantize_q6_k ground truth.
    #[cfg(target_os = "macos")]
    fn cpu_q6k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        use rnb_cpu::quantize::blocks::BlockQ6_K;
        use rnb_cpu::quantize::dequant::dequantize_q6_k;
        let nb = k / 256;
        let bpr = nb * 210;
        let mut out = vec![0f32; m * n];
        let mut deq = [0f32; 256];
        let mut deq_row = vec![0f32; k];
        for row in 0..n {
            let rb = &wb[row * bpr..(row + 1) * bpr];
            for sb in 0..nb {
                let blk = unsafe { &*(rb[sb * 210..].as_ptr() as *const BlockQ6_K) };
                dequantize_q6_k(blk, &mut deq);
                deq_row[sb * 256..(sb + 1) * 256].copy_from_slice(&deq);
            }
            for tok in 0..m {
                let inp = &input[tok * k..(tok + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += deq_row[kk] * inp[kk];
                }
                out[tok * n + row] = acc;
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn cpu_qwen_moe_sparse_accum_reference(
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        expert_ids: &[u32],
        token_ids: &[u32],
        route_weights: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_all: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * hidden_dim];
        for slot in 0..expert_ids.len() {
            let expert = expert_ids[slot] as usize;
            let token = token_ids[slot] as usize;
            let gate_start = expert * gate_expert_bytes;
            let up_start = expert * up_expert_bytes;
            let down_start = expert * down_expert_bytes;
            let input = &norm_all[token * hidden_dim..(token + 1) * hidden_dim];
            let gate = cpu_q4k_gemm_reference(
                &gate_all[gate_start..gate_start + gate_expert_bytes],
                ffn_dim,
                hidden_dim,
                input,
                1,
            );
            let up = cpu_q4k_gemm_reference(
                &up_all[up_start..up_start + up_expert_bytes],
                ffn_dim,
                hidden_dim,
                input,
                1,
            );
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = cpu_q6k_gemm_reference(
                &down_all[down_start..down_start + down_expert_bytes],
                hidden_dim,
                ffn_dim,
                &act,
                1,
            );
            let weight = route_weights[slot];
            for h in 0..hidden_dim {
                out[token * hidden_dim + h] += weight * down[h];
            }
        }
        out
    }

    /// pm36: 결정적 BlockQ5_K bytes 1개 조립 (176B layout).
    /// d(0-1) dmin(2-3) scales[12](4-15) qh[32](16-47) qs[128](48-175).
    /// quantize_q5_k_vec 부재 → raw byte hand-build, dequantize_q5_k 로 ground truth.
    /// scales 는 임의 byte 라도 get_scale_min_k4 가 GPU/CPU 동일 추출 → 일치.
    #[cfg(target_os = "macos")]
    fn make_q5k_block(
        d_val: f32,
        dmin_val: f32,
        scales: [u8; 12],
        qh: [u8; 32],
        qs: [u8; 128],
    ) -> Vec<u8> {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(dmin_val).to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..48].copy_from_slice(&qh);
        block[48..176].copy_from_slice(&qs);
        block
    }

    /// n개 row × nb개 block 의 Q5_K weight bytes 를 결정적으로 생성.
    #[cfg(target_os = "macos")]
    fn build_q5k_rows(n: usize, k: usize) -> Vec<u8> {
        let nb = k / 256;
        let mut out = Vec::with_capacity(n * nb * 176);
        for row in 0..n {
            for blk in 0..nb {
                let seed = row * nb + blk;
                let d_val = 0.02 + (seed % 7) as f32 * 0.004;
                let dmin_val = 0.01 + (seed % 5) as f32 * 0.003;
                let mut scales = [0u8; 12];
                for (i, s) in scales.iter_mut().enumerate() {
                    *s = ((seed * 2 + i * 3) % 256) as u8;
                }
                let mut qh = [0u8; 32];
                for (i, q) in qh.iter_mut().enumerate() {
                    *q = ((seed * 7 + i * 11) % 256) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((seed * 3 + i * 5) % 256) as u8;
                }
                out.extend(make_q5k_block(d_val, dmin_val, scales, qh, qs));
            }
        }
        out
    }

    /// weight[N,K] Q5_K bytes, input[M,K] f32 → out[M,N], dequantize_q5_k ground truth.
    #[cfg(target_os = "macos")]
    fn cpu_q5k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        use rnb_cpu::quantize::blocks::BlockQ5_K;
        use rnb_cpu::quantize::dequant::dequantize_q5_k;
        let nb = k / 256;
        let bpr = nb * 176;
        let mut out = vec![0f32; m * n];
        let mut deq = [0f32; 256];
        let mut deq_row = vec![0f32; k];
        for row in 0..n {
            let rb = &wb[row * bpr..(row + 1) * bpr];
            for sb in 0..nb {
                let blk = unsafe { &*(rb[sb * 176..].as_ptr() as *const BlockQ5_K) };
                dequantize_q5_k(blk, &mut deq);
                deq_row[sb * 256..(sb + 1) * 256].copy_from_slice(&deq);
            }
            for tok in 0..m {
                let inp = &input[tok * k..(tok + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += deq_row[kk] * inp[kk];
                }
                out[tok * n + row] = acc;
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn gemm_q6k_batch_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        let (n, k, m) = (8usize, 512usize, 4usize); // K=512 = 2 superblocks
        let wb = build_q6k_rows(n, k);
        let input = det_vals(m * k, 0.1);
        let cpu = cpu_q6k_gemm_reference(&wb, n, k, &input, m);
        let gpu = crate::compute::gemm_q6k_with_ctx(&ctx, &wb, &input, n, k, m);
        assert_eq!(gpu.len(), m * n);
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    // pm33 튜닝(T11): Q6_K dequant-공유 커널 correctness. M=300(grid-stride token 분할).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn gemm_q6k_shared_batch_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        let (n, k, m) = (8usize, 512usize, 300usize);
        let wb = build_q6k_rows(n, k);
        let input = det_vals(m * k, 0.1);
        let cpu = cpu_q6k_gemm_reference(&wb, n, k, &input, m);
        let gpu = crate::compute::gemm_q6k_shared_with_ctx(&ctx, &wb, &input, n, k, m);
        assert_eq!(gpu.len(), m * n);
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    // pm33 Task 3: silu_mul 이 batch(dim=m*ffn)에서 깨지지 않는지 격리 확인.
    // 커널은 순수 1D element-wise(gid>=dim return)라 dim 만 m*ffn 로 키우면 batch 동작.
    // backend thin 메서드 없이 encode_silu_mul 직접 호출(Phase2 chain 이 쓰는 그 함수).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn silu_mul_batch_matches_cpu() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        let (m, ffn) = (4usize, 256usize);
        let dim = m * ffn;
        let gate = det_vals(dim, 0.3);
        let up = det_vals(dim, 0.2);
        let cpu: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let gate_buf = crate::ffn_chain::shared_f32_buf(&ctx, &gate);
        let up_buf = crate::ffn_chain::shared_f32_buf(&ctx, &up);
        let dim_buf = crate::ffn_chain::u32_buf(&ctx, dim as u32);
        let cmd = ctx.queue.commandBuffer().expect("cmd buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        crate::ffn_chain::encode_silu_mul(&ctx, &enc, &gate_buf, &up_buf, &dim_buf, dim);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu = crate::ffn_chain::readback(&gate_buf, dim);
        for i in 0..dim {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    // pm33 Task 5: prefill FFN chain(M>1) 전체(gate→up→silu→down) vs CPU reference.
    // chain 입력 = normed(caller 가 norm 적용), 반환 = down(residual 전). down 은 Q6_K.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn prefill_ffn_chain_matches_cpu() {
        let backend = MetalBackend::new();
        let (hid, ffn, m) = (256usize, 512usize, 4usize);
        // gate/up weight: [ffn, hid] Q4_K. down weight: [hid, ffn] Q6_K(hand-build).
        let gate_w = quantize_rows_q4k(&det_vals(ffn * hid, 0.05), ffn, hid);
        let up_w = quantize_rows_q4k(&det_vals(ffn * hid, 0.04), ffn, hid);
        let down_w = build_q6k_rows(hid, ffn);
        let normed = det_vals(m * hid, 0.1);

        // CPU reference: gate/up GEMM → silu → down GEMM.
        let g = cpu_q4k_gemm_reference(&gate_w, ffn, hid, &normed, m); // [m, ffn]
        let u = cpu_q4k_gemm_reference(&up_w, ffn, hid, &normed, m);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
            .collect();
        let cpu = cpu_q6k_gemm_reference(&down_w, hid, ffn, &act, m); // [m, hid]

        let gpu = backend.prefill_ffn_chain(&normed, &gate_w, &up_w, &down_w, true, m, hid, ffn);
        assert_eq!(gpu.len(), m * hid);
        // pm34 M7: M5 default = tensorops(half staging) → global rel(half GEMM 표준). 비-M5 는
        // naive(f32)라 더 작음. element-wise rel 은 want≈0 에서 ill-defined.
        let mut max_abs = 0f32;
        let mut max_w = 0f32;
        for i in 0..m * hid {
            max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
            max_w = max_w.max(cpu[i].abs());
        }
        let global_rel = max_abs / max_w.max(1e-3);
        eprintln!("[pm34] prefill_ffn_chain (default) global_rel={global_rel:.2e}");
        assert!(
            global_rel < 1e-2,
            "prefill_ffn_chain mismatch: global_rel={global_rel}"
        );
    }

    /// pm34 M7: naive opt-out(RNB_METAL_PREFILL_FFN_KERNEL=naive) fallback 경로가 CPU ref 와
    /// f32 정밀 일치(rel<1e-3). default 는 tensorops(prefill_ffn_chain_matches_cpu 가 검증).
    /// **env 설정 때문에 단독 실행**: `cargo test ... prefill_ffn_chain_naive -- --ignored`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device; sets env, run single-threaded"]
    fn prefill_ffn_chain_naive_opt_out_matches_cpu() {
        std::env::set_var("RNB_METAL_PREFILL_FFN_KERNEL", "naive");
        let backend = MetalBackend::new();
        let (hid, ffn, m) = (256usize, 512usize, 4usize);
        let gate_w = quantize_rows_q4k(&det_vals(ffn * hid, 0.05), ffn, hid);
        let up_w = quantize_rows_q4k(&det_vals(ffn * hid, 0.04), ffn, hid);
        let down_w = build_q6k_rows(hid, ffn);
        let normed = det_vals(m * hid, 0.1);
        let g = cpu_q4k_gemm_reference(&gate_w, ffn, hid, &normed, m);
        let u = cpu_q4k_gemm_reference(&up_w, ffn, hid, &normed, m);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
            .collect();
        let cpu = cpu_q6k_gemm_reference(&down_w, hid, ffn, &act, m);
        let gpu = backend.prefill_ffn_chain(&normed, &gate_w, &up_w, &down_w, true, m, hid, ffn);
        std::env::remove_var("RNB_METAL_PREFILL_FFN_KERNEL");
        assert_eq!(gpu.len(), m * hid);
        // naive = f32 staging → element-wise rel<1e-3 정밀 검증.
        for i in 0..m * hid {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "idx {i}: gpu {} cpu {} rel {}",
                gpu[i],
                cpu[i],
                rel
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires TensorOps-capable Metal device; sets env, run single-threaded"]
    fn qwen_moe_id_primitive_runtime_matches_cpu_sparse_accum() {
        let _env_lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _primitive_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        let backend = MetalBackend::new();
        let ctx = backend.ctx.as_ref().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[qwen-id-primitive] not tensorops-capable; skipping runtime oracle");
            return;
        }

        let (hidden_dim, ffn_dim, seq_len) = (256usize, 512usize, 3usize);
        let gate0 = quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.031), ffn_dim, hidden_dim);
        let gate1 = quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.043), ffn_dim, hidden_dim);
        let up0 = quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.037), ffn_dim, hidden_dim);
        let up1 = quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.029), ffn_dim, hidden_dim);
        let down0 = build_q6k_rows(hidden_dim, ffn_dim);
        let down1 = build_q6k_rows(hidden_dim, ffn_dim);
        let gate_expert_bytes = gate0.len();
        let up_expert_bytes = up0.len();
        let down_expert_bytes = down0.len();
        let mut gate_all = Vec::new();
        gate_all.extend_from_slice(&gate0);
        gate_all.extend_from_slice(&gate1);
        let mut up_all = Vec::new();
        up_all.extend_from_slice(&up0);
        up_all.extend_from_slice(&up1);
        let mut down_all = Vec::new();
        down_all.extend_from_slice(&down0);
        down_all.extend_from_slice(&down1);
        let expert_ids = [0u32, 0, 1, 1];
        let token_ids = [0u32, 2, 0, 1];
        let route_weights = [0.6f32, 0.5, 0.4, 0.7];
        let norm_all = det_vals(seq_len * hidden_dim, 0.025);
        let want = cpu_qwen_moe_sparse_accum_reference(
            &gate_all,
            &up_all,
            &down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            &expert_ids,
            &token_ids,
            &route_weights,
            seq_len,
            hidden_dim,
            ffn_dim,
            &norm_all,
        );

        let accum = ffn_chain::QwenMoePrefillAccumCarrier::new(ctx, hidden_dim, seq_len);
        accum.clear();
        accum.upload_norm_all(&norm_all);
        let (gate_wb, gate_off) = compute::wrap_nocopy(ctx, &gate_all);
        let (up_wb, up_off) = compute::wrap_nocopy(ctx, &up_all);
        let (down_wb, down_off) = compute::wrap_nocopy(ctx, &down_all);
        let got = backend
            .try_qwen_moe_prefill_id_primitive(
                ctx,
                &accum,
                &gate_wb,
                gate_off,
                &up_wb,
                up_off,
                &down_wb,
                down_off,
                gate_all.len(),
                up_all.len(),
                down_all.len(),
                gate_expert_bytes,
                up_expert_bytes,
                down_expert_bytes,
                &expert_ids,
                &route_weights,
                &token_ids,
                true,
                seq_len,
                hidden_dim,
                ffn_dim,
                false,
                true,
            )
            .expect("id primitive should run on valid TensorOps fixture");

        let mut max_abs = 0.0f32;
        let mut max_want = 0.0f32;
        for i in 0..want.len() {
            max_abs = max_abs.max((got[i] - want[i]).abs());
            max_want = max_want.max(want[i].abs());
        }
        let global_rel = max_abs / max_want.max(1e-3);
        eprintln!("[qwen-id-primitive] runtime sparse accum global_rel={global_rel:.3e}");
        assert!(
            global_rel < 2e-2,
            "id primitive runtime sparse accum mismatch: global_rel={global_rel}"
        );
    }

    /// pm22 Task 1: per-slot int8 quantize/dequant CPU reference. metal 커널이 1:1
    /// emulate 할 ground truth — roundtrip 오차 한계 + RNE tie 정확값 + zero slot 검증.
    #[test]
    fn kv_int8_ref_roundtrip_and_tie() {
        use crate::compute::{dequant_slot_i8_ref, quantize_slot_i8_ref};
        let v: Vec<f32> = (0..256)
            .map(|i| ((i as f32 * 0.137).sin()) * (1.0 + (i == 13) as i32 as f32 * 8.0))
            .collect();
        let (q, scale) = quantize_slot_i8_ref(&v);
        let dq = dequant_slot_i8_ref(&q, scale);
        let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let tol = max_abs / 254.0 + 1e-6;
        for (a, b) in v.iter().zip(dq.iter()) {
            assert!((a - b).abs() <= tol);
        }
        // RNE tie: max=254 → scale=2.0, x/scale = [127, 0.5, 1.5, -0.5, -1.5]
        // → round_ties_even → [127, 0, 2, 0, -2]
        let tie = vec![254.0f32, 1.0, 3.0, -1.0, -3.0];
        let (tq, ts) = quantize_slot_i8_ref(&tie);
        assert_eq!(ts, 2.0);
        assert_eq!(tq, vec![127i8, 0, 2, 0, -2]);
        // zero slot
        let (zq, zs) = quantize_slot_i8_ref(&vec![0.0; 8]);
        assert_eq!(zs, 0.0);
        assert!(zq.iter().all(|&x| x == 0));
    }

    /// pm26: F32 GEMV chain 커널이 CPU f32 GEMV reference 와 일치하는지 device 검증.
    /// 27B 의 ssm_alpha/beta(F32 raw)를 무손실 device GEMV 로 돌리기 위한 경로
    /// (gdn_quant_code 가 F32 를 None 처리해 48 GDN 이 host per-op 추락했던 것의 fix).
    /// weight_bytes = N*K*4 (f32 LE). alpha/beta-like shape(N=num_v_heads, K=hidden).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn gemv_f32_chain_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host")
        };
        let (n, k) = (32usize, 256usize);
        let weight: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let input: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let mut expect = vec![0f32; n];
        for r in 0..n {
            for c in 0..k {
                expect[r] += weight[r * k + c] * input[c];
            }
        }
        let wbytes: Vec<u8> = weight.iter().flat_map(|f| f.to_le_bytes()).collect();
        let got = crate::compute::gemv_f32_with_ctx(&ctx, &wbytes, &input, n, k);
        assert_eq!(got.len(), n);
        for r in 0..n {
            assert!(
                (got[r] - expect[r]).abs() < 1e-3,
                "row {r}: got {} expect {}",
                got[r],
                expect[r]
            );
        }
    }

    /// pm22 Task 2: kv_append_i8 metal 커널이 CPU quantize_slot_i8_ref 와 per-slot
    /// 1:1 일치하는지 device 검증. 9B shape 일부(head_dim=256, nkv=4)로 2 step append.
    /// GPU rint(RNE) == CPU round_ties_even → q exact match, scale 은 1e-4 상대오차 허용.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn kv_append_i8_matches_cpu_reference() {
        let Some(ctx) = crate::compute::build_metal_context_with_kv_int8(true) else {
            panic!("no Metal device — run on macOS host")
        };
        use crate::compute::{kv_append_i8_steps_with_ctx, quantize_slot_i8_ref};
        let (head_dim, nkv, cap) = (256usize, 4usize, 4usize);
        let kv_dim = nkv * head_dim;
        let mk = |s: f32| {
            (0..kv_dim)
                .map(|i| ((i as f32 + s) * 0.07).sin())
                .collect::<Vec<_>>()
        };
        let k_steps = vec![mk(1.0), mk(2.0)];
        let v_steps = vec![mk(3.0), mk(4.0)];
        let (ki, _vi, ks, _vs) =
            kv_append_i8_steps_with_ctx(&ctx, &k_steps, &v_steps, head_dim, nkv, cap);
        for (pos, k) in k_steps.iter().enumerate() {
            for h in 0..nkv {
                let (rq, rs) = quantize_slot_i8_ref(&k[h * head_dim..(h + 1) * head_dim]);
                assert!(
                    (ks[pos * nkv + h] - rs).abs() <= rs * 1e-4 + 1e-9,
                    "scale mismatch pos{pos} h{h}"
                );
                for d in 0..head_dim {
                    assert_eq!(
                        ki[pos * kv_dim + h * head_dim + d],
                        rq[d],
                        "q mismatch pos{pos} h{h} d{d}"
                    );
                }
            }
        }
    }

    /// pm22 Task 3: int8 KvResident 의 prefill 변환(ensure_filled_i8)이 host f16 KV 를
    /// CPU reference quantize 와 exact 일치하게 device int8/scale 에 채우는지 검증.
    /// ensure_filled_i8 와 reference 가 같은 quantize_slot_i8_ref 를 타니 정확히 일치해야 함.
    #[test]
    #[ignore = "requires Metal device"]
    fn kv_resident_ensure_filled_i8_matches_reference() {
        let Some(ctx) = crate::compute::build_metal_context_with_kv_int8(true) else {
            panic!("no Metal device — run on macOS host")
        };
        use crate::compute::{f16_bits_to_f32, quantize_slot_i8_ref, KvResident};
        let (nkv, hd, cap) = (4usize, 256usize, 4usize);
        let kv_dim = nkv * hd;
        let n = 2usize;
        let mut k16 = vec![0u16; n * kv_dim];
        let mut v16 = vec![0u16; n * kv_dim];
        for i in 0..n * kv_dim {
            k16[i] = half::f16::from_f32(((i as f32) * 0.013).sin()).to_bits();
            v16[i] = half::f16::from_f32(((i as f32) * 0.017).cos()).to_bits();
        }
        let mut kv = KvResident::new(&ctx, nkv, hd, cap);
        kv.ensure_filled_i8(&k16, &v16, n);
        let ki = unsafe {
            std::slice::from_raw_parts(
                kv.k_i8.as_ref().unwrap().contents().as_ptr() as *const i8,
                n * kv_dim,
            )
        };
        let ks = unsafe {
            std::slice::from_raw_parts(
                kv.k_scale.as_ref().unwrap().contents().as_ptr() as *const f32,
                n * nkv,
            )
        };
        for pos in 0..n {
            for h in 0..nkv {
                let off = pos * kv_dim + h * hd;
                let slot: Vec<f32> = (0..hd).map(|d| f16_bits_to_f32(k16[off + d])).collect();
                let (rq, rs) = quantize_slot_i8_ref(&slot);
                assert_eq!(ks[pos * nkv + h], rs, "scale pos{pos} h{h}");
                for d in 0..hd {
                    assert_eq!(ki[off + d], rq[d], "q pos{pos} h{h} d{d}");
                }
            }
        }
        assert_eq!(kv.filled, n);
    }

    /// pm22 Task 4: int8 KV decode attention 커널이 f16 attn_decode reference 와
    /// 의미상 동등한지 검증. 같은 f32 KV 를 (a) per-slot int8 양자화 → attn_decode_i8,
    /// (b) f16 bits → attn_decode_with_ctx 로 각각 돌려 max relative error < 2e-2 확인.
    /// per-slot int8 + f16 accumulator 중첩 오차가 천장. 핵심 검증점:
    /// K dequant ksc(simd_sum 밖), V dequant vv = v_i8 * vsc(rescale/else 양쪽).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn attn_decode_i8_matches_f16_within_tol() {
        let Some(ctx) = crate::compute::build_metal_context_with_kv_int8(true) else {
            panic!("no Metal device — run on macOS host")
        };
        use crate::compute::{attn_decode_i8_with_ctx, quantize_slot_i8_ref};
        let (nh, nkv, hd, kl) = (4usize, 2usize, 64usize, 5usize);
        let kv_dim = nkv * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..nh * hd).map(|i| ((i as f32) * 0.03).sin()).collect();
        let kf: Vec<f32> = (0..kl * kv_dim)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let vf: Vec<f32> = (0..kl * kv_dim)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();
        let mut ki = vec![0i8; kl * kv_dim];
        let mut vi = vec![0i8; kl * kv_dim];
        let mut ks = vec![0f32; kl * nkv];
        let mut vs = vec![0f32; kl * nkv];
        for j in 0..kl {
            for h in 0..nkv {
                let off = j * kv_dim + h * hd;
                let (kq, ksc) = quantize_slot_i8_ref(&kf[off..off + hd]);
                let (vq, vsc) = quantize_slot_i8_ref(&vf[off..off + hd]);
                ks[j * nkv + h] = ksc;
                vs[j * nkv + h] = vsc;
                for d in 0..hd {
                    ki[off + d] = kq[d];
                    vi[off + d] = vq[d];
                }
            }
        }
        let out_i8 = attn_decode_i8_with_ctx(&ctx, &q, &ki, &vi, &ks, &vs, nh, nkv, hd, kl, scale);
        // f16 reference: 같은 f32 KV → f16 bits → attn_decode_with_ctx.
        // attn_decode_with_ctx 는 f16 전용(int8 ctx 가드 assert!(!ctx.kv_int8)) 이므로
        // f16 ctx 를 별도로 만들어 호출한다(int8 ctx 로 호출하면 panic).
        let Some(ctx_f16) = crate::compute::build_metal_context_with_kv_int8(false) else {
            panic!("no Metal device — run on macOS host")
        };
        let k16: Vec<u16> = kf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v16: Vec<u16> = vf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let out_f16 =
            crate::compute::attn_decode_with_ctx(&ctx_f16, &q, &k16, &v16, nh, nkv, hd, kl, scale);
        // 출력 element 값이 0 근처면 per-element |a-b|/|b| 가 폭발하므로(near-zero
        // 분모) 양자화 정확도 메트릭으로 부적절하다. 출력 전체 스케일(max|out_f16|)로
        // 정규화한 relative error 로 판정한다 — int8 KV vs f16 KV 의미 동등성 측정.
        let scale_ref = out_f16
            .iter()
            .fold(0.0f32, |m, &x| m.max(x.abs()))
            .max(1e-4);
        let mut max_rel = 0.0f32;
        for (a, b) in out_i8.iter().zip(out_f16.iter()) {
            max_rel = max_rel.max((a - b).abs() / scale_ref);
        }
        eprintln!("attn_decode_i8 vs f16 max_rel(scale-normalized) = {max_rel}");
        eprintln!("out_i8[0..6]  = {:?}", &out_i8[0..6]);
        eprintln!("out_f16[0..6] = {:?}", &out_f16[0..6]);
        assert!(
            max_rel < 2e-2,
            "max_rel {max_rel} >= 2e-2 (per-slot int8 + f16 acc 중첩 오차)"
        );
    }

    /// pm54: split-K int8 KV decode attention 은 기존 int8 커널과 같은 의미여야 한다.
    /// 긴 kv_len 에서 KV 축 병렬도를 늘리기 위한 opt-in fast path 이므로, 먼저 기존
    /// int8 attention 을 reference 로 삼아 scale-normalized 오차를 고정한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn attn_decode_i8_splitk_matches_i8_within_tol() {
        let Some(ctx) = crate::compute::build_metal_context_with_kv_int8(true) else {
            panic!("no Metal device — run on macOS host")
        };
        use crate::compute::{
            attn_decode_i8_splitk_with_ctx, attn_decode_i8_with_ctx, quantize_slot_i8_ref,
        };
        let (nh, nkv, hd, kl) = (8usize, 2usize, 256usize, 2049usize);
        let kv_dim = nkv * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..nh * hd)
            .map(|i| (((i * 17 + 3) as f32) * 0.007).sin())
            .collect();
        let kf: Vec<f32> = (0..kl * kv_dim)
            .map(|i| (((i * 13 + 11) as f32) * 0.005).sin())
            .collect();
        let vf: Vec<f32> = (0..kl * kv_dim)
            .map(|i| (((i * 19 + 5) as f32) * 0.006).cos())
            .collect();
        let mut ki = vec![0i8; kl * kv_dim];
        let mut vi = vec![0i8; kl * kv_dim];
        let mut ks = vec![0f32; kl * nkv];
        let mut vs = vec![0f32; kl * nkv];
        for j in 0..kl {
            for h in 0..nkv {
                let off = j * kv_dim + h * hd;
                let (kq, ksc) = quantize_slot_i8_ref(&kf[off..off + hd]);
                let (vq, vsc) = quantize_slot_i8_ref(&vf[off..off + hd]);
                ks[j * nkv + h] = ksc;
                vs[j * nkv + h] = vsc;
                for d in 0..hd {
                    ki[off + d] = kq[d];
                    vi[off + d] = vq[d];
                }
            }
        }
        let expected =
            attn_decode_i8_with_ctx(&ctx, &q, &ki, &vi, &ks, &vs, nh, nkv, hd, kl, scale);
        let actual = attn_decode_i8_splitk_with_ctx(
            &ctx, &q, &ki, &vi, &ks, &vs, nh, nkv, hd, kl, scale, 16,
        );

        let scale_ref = expected
            .iter()
            .fold(0.0f32, |m, &x| m.max(x.abs()))
            .max(1e-4);
        let mut max_rel = 0.0f32;
        for (a, b) in actual.iter().zip(expected.iter()) {
            max_rel = max_rel.max((a - b).abs() / scale_ref);
        }
        eprintln!("attn_decode_i8_splitk vs i8 max_rel(scale-normalized) = {max_rel}");
        assert!(
            max_rel < 2e-2,
            "max_rel {max_rel} >= 2e-2 (split-K reduction drift)"
        );
    }

    /// footgun(pm17): in-process 로 engine 을 재사용(반복 측정/multi-turn)하면 직전
    /// sequence 가 carrier 에 남긴 device KV state(`KvResident::filled != 0`)가 그대로
    /// 살아, 새 sequence 의 prefill 재동기화를 `if filled == 0` 가드에서 skip → stale KV
    /// 로 틀린 출력. `clear_sequence_state` 가 carrier(attn/gdn) + KvResident 를 모두
    /// evict 해 다음 호출에서 fresh(filled=0) 재생성되게 하는지 검증.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn clear_sequence_state_evicts_carrier_state() {
        let backend = MetalBackend::new();
        let ctx = backend.ctx.as_ref().expect("Metal device");
        // 직전 sequence 가 채워둔 device KV state 흉내.
        let mut kv = compute::KvResident::new(ctx, 8, 8, 8);
        kv.filled = 5;
        backend.kv_residents.borrow_mut().insert(3, kv);
        assert!(!backend.kv_residents.borrow().is_empty());

        backend.clear_sequence_state();

        assert!(
            backend.kv_residents.borrow().is_empty(),
            "clear_sequence_state must evict carrier KV state"
        );
        assert!(backend.attn_carriers.borrow().is_empty());
        assert!(backend.gdn_carriers.borrow().is_empty());
    }

    /// pm31: materialize_sequence_state(speculative checkpoint) 경로. delta residency 시
    /// host delta 가 stale 이라, speculative save 직전 device 잔류 delta 를 host 로 sync 해야
    /// checkpoint 가 정확하다(CUDA `sync_delta_state_cache` 와 동형). seed 한 carrier 의
    /// device delta 를 `sync_delta_state` 로 읽는지 + carrier 없는 layer 는 false 인지 검증.
    #[cfg(target_os = "macos")]
    #[test]
    fn sync_delta_state_reads_seeded_carrier_delta() {
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            return; // GPU 없는 CI 환경 skip
        };
        let mut carrier = gdn_chain::GdnCarrier::new(ctx, 64, 16, 4, 32, 2, 2, 8, 8, 64, 1e-6);
        let conv = vec![0.0_f32; carrier.conv_state_len];
        let delta = vec![3.0_f32; carrier.delta_state_len];
        carrier.upload_states_resident(&conv, &delta);
        backend.gdn_carriers.borrow_mut().insert(7, carrier);

        // carrier 존재 → device delta 가 host out 으로 sync.
        let mut out = vec![0.0_f32; 2 * 8 * 8];
        assert!(
            backend.sync_delta_state(7, &mut out),
            "carrier 7 exists → sync ok"
        );
        assert_eq!(out, delta, "device delta synced to host out");

        // carrier 없는 layer → false(안전, host out 안 건드림).
        let mut out2 = vec![9.0_f32; 2 * 8 * 8];
        assert!(
            !backend.sync_delta_state(999, &mut out2),
            "no carrier → false"
        );
        assert!(
            out2.iter().all(|&v| v == 9.0),
            "unknown layer leaves out untouched"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn clear_sequence_state_evicts_int8_kv_resident() {
        // int8 ctx 로 carrier 를 만든 뒤 clear_sequence_state 가 int8 KvResident 도
        // 동일하게 evict 하는지(carrier 경로 == f16 경로) 확인.
        let backend = MetalBackend::new_with_kv_int8(true);
        let ctx = backend.ctx.as_ref().expect("Metal device");
        // 직전 sequence 가 채워둔 device KV state 흉내(int8 경로).
        let mut kv = compute::KvResident::new(ctx, 8, 8, 8);
        assert!(kv.kv_int8, "int8 ctx 에서 KvResident 는 int8 경로여야 함");
        kv.filled = 5;
        backend.kv_residents.borrow_mut().insert(3, kv);
        assert!(!backend.kv_residents.borrow().is_empty());

        backend.clear_sequence_state();

        assert!(
            backend.kv_residents.borrow().is_empty(),
            "clear_sequence_state must evict int8 carrier KV state"
        );
        assert!(backend.attn_carriers.borrow().is_empty());
        assert!(backend.gdn_carriers.borrow().is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_opens_system_default_device() {
        let name = device::system_default_device_name();
        assert!(name.is_some(), "expected a Metal device on this Mac");
        eprintln!("device = {name:?}");
    }

    /// pm21: REST 분해 측정법 결정용 — 이 기기(M5)가 어떤 counter sampling
    /// boundary 를 지원하는지 출력. AtDispatchBoundary 지원 시 production 단일
    /// encoder 안에서 dispatch 별 GPU timestamp 직접 측정 가능(왜곡 0). 미지원 시
    /// stage boundary(=encoder 경계) 기반 측정으로 폴백.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_counter_sampling_capability() {
        use objc2_metal::{MTLCounterSamplingPoint, MTLCreateSystemDefaultDevice, MTLDevice};
        let device = MTLCreateSystemDefaultDevice().expect("no metal device");
        let stage = device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary);
        let dispatch = device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary);
        eprintln!(
            "[counter-cap] device={:?} stage_boundary={stage} dispatch_boundary={dispatch}",
            device.name().to_string()
        );
    }

    /// pm34 feasibility 게이트: Metal 4 `mpp::tensor_ops::matmul2d`(M5 GPU neural
    /// accelerator) 셰이더가 우리 런타임 컴파일 경로(`newLibraryWithSource` +
    /// `MTLLanguageVersion::Version4_0`)로 컴파일되는지 확인. liuliu/cider 는 둘 다
    /// `xcrun metal` AOT 라 런타임 컴파일 증거가 없었음 — 이게 simdgroup→TensorOps
    /// 전환의 GO/NO-GO 결정타. 컴파일만 검증(run 은 다음 milestone).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal 4 TensorOps-capable SDK/runtime (macOS 26.2+, Apple10/M5); run on real hardware"]
    fn metal4_tensorops_compiles_at_runtime() {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
            MTLLibrary,
        };

        let device = MTLCreateSystemDefaultDevice().expect("no metal device");

        const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void poc_matmul_fp16(
    device half *A_buf [[buffer(0)]],
    device half *B_buf [[buffer(1)]],
    device half *C_buf [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(256, 64));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(32, 256));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(32, 64));
    constexpr auto desc = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(A, B, C);
}
"#;

        let source = NSString::from_str(SRC);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);

        match device.newLibraryWithSource_options_error(&source, Some(&options)) {
            Ok(lib) => {
                let name = NSString::from_str("poc_matmul_fp16");
                assert!(
                    lib.newFunctionWithName(&name).is_some(),
                    "poc_matmul_fp16 function not found in compiled library"
                );
                eprintln!(
                    "✅ [pm34] Metal 4 mpp::tensor_ops::matmul2d compiled at RUNTIME \
                     (newLibraryWithSource + Version4_0)"
                );
            }
            Err(e) => panic!("❌ [pm34] Metal 4 tensorops runtime compile FAILED: {e:?}"),
        }
    }

    /// pm99 probe: MPP TensorOps 의 right input cooperative tensor 를 런타임 컴파일러가
    /// 받는지 확인한다. 이게 되면 Q4_K/Q6_K raw block 을 persistent F16 cache 로 풀지 않고,
    /// 커널 안 dequant 값을 TensorOps 입력 가까이로 밀어 넣는 다음 후보를 검증할 수 있다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal 4 TensorOps-capable SDK/runtime (macOS 26.2+, Apple10/M5); run on real hardware"]
    fn metal4_tensorops_right_coop_input_compiles_at_runtime() {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
            MTLLibrary,
        };

        let device = MTLCreateSystemDefaultDevice().expect("no metal device");

        const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void coop_right_probe(
    device half  *A_buf [[buffer(0)]],
    device float *C_buf [[buffer(1)]],
    threadgroup char *shmem [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    threadgroup half *B_stage = (threadgroup half *)shmem;
    for (uint i = tid; i < BK * BN; i += 32u) {
        B_stage[i] = half(0.0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(BK, BM));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        B_stage, dextents<int32_t, 2>(BN, BK));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(BN, BM));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
    right.load(B);
    op.run(A, right, C);
}
"#;

        let source = NSString::from_str(SRC);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);

        match device.newLibraryWithSource_options_error(&source, Some(&options)) {
            Ok(lib) => {
                let name = NSString::from_str("coop_right_probe");
                assert!(
                    lib.newFunctionWithName(&name).is_some(),
                    "coop_right_probe function not found in compiled library"
                );
                eprintln!("[pm99] right cooperative TensorOps input compile probe passed");
            }
            Err(e) => {
                panic!("[pm99] right cooperative TensorOps input compile probe failed: {e:?}")
            }
        }
    }

    /// pm99 probe: right input cooperative tensor 를 `load()` 없이 직접 초기화할 수 있는지
    /// 확인한다. 통과하면 raw quant dequant 값을 thread-local cooperative storage 로 바로
    /// 넣는 커널을 만들 수 있고, threadgroup half weight tile 왕복을 줄일 수 있다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal 4 TensorOps-capable SDK/runtime (macOS 26.2+, Apple10/M5); run on real hardware"]
    fn metal4_tensorops_right_coop_direct_fill_compiles_at_runtime() {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
            MTLLibrary,
        };

        let device = MTLCreateSystemDefaultDevice().expect("no metal device");

        const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void coop_right_direct_fill_probe(
    device half  *A_buf [[buffer(0)]],
    device float *C_buf [[buffer(1)]])
{
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(BK, BM));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(BN, BM));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto right = op.template get_right_input_cooperative_tensor<half, half, float>();

    for (ushort i = 0; i < right.get_capacity(); ++i) {
        auto idx = right.get_multidimensional_index(i);
        right[i] = half(idx[0] + idx[1]);
    }

    op.run(A, right, C);
}
"#;

        let source = NSString::from_str(SRC);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);

        match device.newLibraryWithSource_options_error(&source, Some(&options)) {
            Ok(lib) => {
                let name = NSString::from_str("coop_right_direct_fill_probe");
                assert!(
                    lib.newFunctionWithName(&name).is_some(),
                    "coop_right_direct_fill_probe function not found in compiled library"
                );
                eprintln!("[pm99] right cooperative TensorOps direct-fill compile probe passed");
            }
            Err(e) => {
                panic!("[pm99] right cooperative TensorOps direct-fill compile probe failed: {e:?}")
            }
        }
    }

    /// pm34 M1 Task 2: capability 판정 + 조건부 파이프라인. capability=true(M5)든 false든
    /// context build 가 panic 하지 않아야 한다(코덱스 B2 — eager build_pipeline panic 회피).
    /// Task 2 시점엔 파이프라인 미연결이라 capable 여도 None. Task 3 에서 is_some() 으로 강화.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_context_builds_without_tensorops_panic() {
        let ctx = crate::compute::build_metal_context();
        assert!(ctx.is_some(), "metal context build must not panic");
        let ctx = ctx.unwrap();
        eprintln!("[pm34] tensorops_capable={}", ctx.tensorops_capable);
        // Task 3: pipeline 연결됨 → capable 이면 Some, 아니면 None(둘 다 panic 없음).
        if ctx.tensorops_capable {
            assert!(ctx.prefill_ffn_tensorops_pipeline.is_some());
        } else {
            assert!(ctx.prefill_ffn_tensorops_pipeline.is_none());
        }
    }

    /// pm34 M2 go/no-go: tensorops staging matmul2d PoC 가 CPU ref 와 일치하는지(rel<1e-3).
    /// 비대칭 A[16x32]·B[32x16]=C[16x16] — transpose/layout 오류 노출. 실패 시 pm34 재평가.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_poc_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm34] not tensorops-capable; skipping PoC exec");
            return;
        }
        // 비대칭 deterministic 입력: A[16x32], B[32x16] -> C[16x16]
        let a: Vec<f32> = (0..16 * 32).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..32 * 16).map(|i| (i % 5) as f32 * 0.2).collect();
        let got = crate::compute::run_tensorops_poc(&ctx, &a, &b);
        // CPU ref: C[m,n] = sum_k A[m,k]*B[k,n], K=32
        let mut want = vec![0f32; 256];
        for m in 0..16 {
            for n in 0..16 {
                let mut acc = 0f32;
                for k in 0..32 {
                    acc += a[m * 32 + k] * b[k * 16 + n];
                }
                want[m * 16 + n] = acc;
            }
        }
        let mut max_rel = 0f32;
        for i in 0..256 {
            let denom = want[i].abs().max(1e-4);
            max_rel = max_rel.max((got[i] - want[i]).abs() / denom);
        }
        eprintln!("[pm34] tensorops PoC max_rel={max_rel:.2e}");
        eprintln!("[pm34] C[0..4]={:?} want={:?}", &got[0..4], &want[0..4]);
        assert!(max_rel < 1e-3, "tensorops PoC mismatch: max_rel={max_rel}");
    }

    /// pm46 Phase 1: GDN delta scan STEP5 outer product GEMM(matmul2d f16) + scalar 가 f32 oracle
    /// 과 일치하는지(R2 게이트 correctness). 실측 shape nh=48, C=38, HV=HK=128. gemm rel<1e-2
    /// (f16 input round 허용), scalar rel<1e-4. GPU-time 도 출력(1회 — median 은 bench 진입점).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn step5_outer_gemm_matches_f32_oracle() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm46] not tensorops-capable; skipping STEP5 microbench");
            return;
        }
        let (nh, c, hv, hk) = (48usize, 38usize, 128usize, 128usize);
        // 결정적 입력(인덱스 기반, ±0.x 범위 → f16 round 영향 관찰 가능)
        let us: Vec<f32> = (0..nh * c * hv)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let kk: Vec<f32> = (0..nh * c * hk)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        // f32 oracle: C[h,vi,ki] = Σ_j Us[h,j,vi] * Kk[h,j,ki]
        let mut oracle = vec![0f32; nh * hv * hk];
        for h in 0..nh {
            for vi in 0..hv {
                for ki in 0..hk {
                    let mut acc = 0f32;
                    for j in 0..c {
                        acc += us[h * c * hv + j * hv + vi] * kk[h * c * hk + j * hk + ki];
                    }
                    oracle[h * hv * hk + vi * hk + ki] = acc;
                }
            }
        }
        // correctness (1회, 결정적)
        let (gemm, _) = crate::compute::run_step5_outer_gemm(&ctx, &us, &kk, c, nh);
        let (scalar, _) = crate::compute::run_step5_outer_scalar(&ctx, &us, &kk, c, nh);
        let max_rel = |got: &[f32]| {
            got.iter()
                .zip(&oracle)
                .map(|(g, o)| (g - o).abs() / o.abs().max(1e-4))
                .fold(0f32, f32::max)
        };
        let gemm_rel = max_rel(&gemm);
        let scalar_rel = max_rel(&scalar);
        assert!(
            scalar_rel < 1e-4,
            "scalar STEP5 mismatch: max_rel={scalar_rel}"
        );
        assert!(
            gemm_rel < 1e-2,
            "f16 GEMM STEP5 max_rel={gemm_rel} exceeds 1e-2"
        );

        // 성능: warmup 3 + 측정 10회 median (R2 게이트). 0.0xms 단위라 1회 노이즈 회피.
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let mut g_v = Vec::new();
        let mut s_v = Vec::new();
        for i in 0..13 {
            let (_, gms) = crate::compute::run_step5_outer_gemm(&ctx, &us, &kk, c, nh);
            let (_, sms) = crate::compute::run_step5_outer_scalar(&ctx, &us, &kk, c, nh);
            if i >= 3 {
                g_v.push(gms);
                s_v.push(sms);
            }
        }
        let g_med = median(g_v);
        let s_med = median(s_v);
        eprintln!("[pm46] STEP5 correctness: gemm rel={gemm_rel:.2e} scalar rel={scalar_rel:.2e}");
        eprintln!(
            "[pm46] STEP5 median(10) gemm={g_med:.4}ms scalar={s_med:.4}ms speedup={:.1}x (nh={nh} C={c})",
            s_med / g_med
        );
    }

    /// pm48 prefill flash attention microbench: GPU naive lane-parallel flash vs CPU
    /// `attention_batch_f16`(f16 NEON batch, 새 macOS default). ROI 가늠 — GPU 격리 GPU-time 이
    /// CPU wall 을 이기나(리뷰 F6: 격리조차 느리면 천장→중단, 빠르면 device-resident 본격).
    /// 1 layer 분량(27B Qwen3.6: seq=kv=1139, nh=24, nkv=4, hd=256, causal, GQA 6:1).
    /// correctness: GPU vs CPU f16 max_rel < 5e-2(둘 다 f16 acc, branched online softmax 등가).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn pm48_prefill_flash_microbench() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            eprintln!("[pm48] no metal ctx; skip");
            return;
        };
        let (seq, kv, nh, nkv, hd) = (1139usize, 1139usize, 24usize, 4usize, 256usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        // 결정적 random data (인덱스 해시 → 약 [-0.5, 0.5]).
        let r = |i: usize| -> f32 {
            (((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 500.0 - 1.0) * 0.5
        };
        let q: Vec<f32> = (0..seq * nh * hd).map(r).collect();
        let k_f16: Vec<u16> = (0..kv * nkv * hd)
            .map(|i| half::f16::from_f32(r(i + 7)).to_bits())
            .collect();
        let v_f16: Vec<u16> = (0..kv * nkv * hd)
            .map(|i| half::f16::from_f32(r(i + 13)).to_bits())
            .collect();

        // correctness: GPU flash vs CPU attention_batch_f16.
        let (gpu_out, _) = crate::compute::run_prefill_flash_microbench(
            &ctx, &q, &k_f16, &v_f16, seq, kv, nh, nkv, hd, scale,
        );
        let cpu_out = rnb_cpu::kernels::attention::attention_batch_f16(
            &q, &k_f16, &v_f16, seq, kv, nh, nkv, hd, scale, None, None,
        );
        let max_rel = gpu_out
            .iter()
            .zip(&cpu_out)
            .map(|(g, c)| (g - c).abs() / c.abs().max(1e-3))
            .fold(0f32, f32::max);
        // microbench 는 attn_decode 의 branched per-element online softmax + f16 acc 라, CPU
        // attention_batch_f16(batch tile FA2 rescale + f16 acc)와 rescale 시점이 달라 drift(리뷰 F5).
        // 실제 device-resident 본격은 f32 acc(spec §4)로 정밀화 → 이 단계는 ROI 속도 가늠이 목적,
        // correctness 는 sanity(터무니없는 값/레이아웃 버그)만 확인.
        eprintln!("[pm48] flash GPU vs CPU(f16 batch) max_rel={max_rel:.3e} (branched vs tile FA2 f16 acc)");
        assert!(
            max_rel < 0.5,
            "flash sanity max_rel={max_rel} (레이아웃/버그 의심)"
        );

        // 성능: warmup 3 + median 10. GPU GPU-time vs CPU wall(rayon, full-chain 685ms 와 동일 조건).
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (mut g_v, mut c_v) = (Vec::new(), Vec::new());
        for i in 0..13 {
            let (_, gms) = crate::compute::run_prefill_flash_microbench(
                &ctx, &q, &k_f16, &v_f16, seq, kv, nh, nkv, hd, scale,
            );
            let t0 = std::time::Instant::now();
            let _ = rnb_cpu::kernels::attention::attention_batch_f16(
                &q, &k_f16, &v_f16, seq, kv, nh, nkv, hd, scale, None, None,
            );
            let cms = t0.elapsed().as_secs_f64() * 1000.0;
            if i >= 3 {
                g_v.push(gms);
                c_v.push(cms);
            }
        }
        let g_med = median(g_v);
        let c_med = median(c_v);
        eprintln!(
            "[pm48] 1-layer median(10): GPU flash={g_med:.3}ms CPU f16 batch={c_med:.3}ms speedup={:.2}x",
            c_med / g_med
        );
        eprintln!(
            "[pm48] 16-layer 추정: GPU {:.1}ms vs CPU {:.1}ms (full-chain attn-compute f16 685ms 참조)",
            g_med * 16.0,
            c_med * 16.0
        );

        // ===== pm48 simdgroup matmul2d flash 커널 (go/no-go 게이트) =====
        if !ctx.tensorops_capable {
            eprintln!("[pm48-simd] not tensorops-capable; skipping simdgroup flash 커널");
            return;
        }

        // (a) correctness: 작은 케이스(seq=64, hd=256, nh=4, nkv=1) → f32 acc eager oracle 대비.
        //     simd 커널이 f32 acc/m/l 이라 eager(f32 누적, q f16 round + k/v f16 dequant)와 가까워야 함.
        {
            let (s2, kv2, nh2, nkv2, hd2) = (64usize, 64usize, 4usize, 1usize, 256usize);
            let sc2 = 1.0f32 / (hd2 as f32).sqrt();
            let q2: Vec<f32> = (0..s2 * nh2 * hd2).map(r).collect();
            let k2: Vec<u16> = (0..kv2 * nkv2 * hd2)
                .map(|i| half::f16::from_f32(r(i + 7)).to_bits())
                .collect();
            let v2: Vec<u16> = (0..kv2 * nkv2 * hd2)
                .map(|i| half::f16::from_f32(r(i + 13)).to_bits())
                .collect();
            // f32 eager oracle: QK^T scale → causal softmax → PV. q f16 round, k/v f16 dequant, f32 누적.
            let to_f16 = |x: f32| half::f16::from_f32(x).to_f32();
            let mut oracle = vec![0f32; s2 * nh2 * hd2];
            let hpg = nh2 / nkv2;
            for h in 0..nh2 {
                let kvh = h / hpg;
                for qi in 0..s2 {
                    let gpos = (kv2 - s2) + qi; // causal: 허용 kv pos ≤ gpos
                    let mut scores = vec![f32::NEG_INFINITY; gpos + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, sj) in scores.iter_mut().enumerate() {
                        let mut dot = 0f32;
                        for d in 0..hd2 {
                            let qf = to_f16(q2[(qi * nh2 + h) * hd2 + d]);
                            let kf = to_f16(
                                half::f16::from_bits(k2[(j * nkv2 + kvh) * hd2 + d]).to_f32(),
                            );
                            dot += qf * kf;
                        }
                        let x = dot * sc2;
                        *sj = x;
                        if x > mx {
                            mx = x;
                        }
                    }
                    let mut den = 0f32;
                    let mut acc = vec![0f32; hd2];
                    for (j, &x) in scores.iter().enumerate() {
                        let p = (x - mx).exp();
                        den += p;
                        for (d, a) in acc.iter_mut().enumerate() {
                            let vf = to_f16(
                                half::f16::from_bits(v2[(j * nkv2 + kvh) * hd2 + d]).to_f32(),
                            );
                            *a += p * vf;
                        }
                    }
                    let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
                    for (d, a) in acc.iter().enumerate() {
                        oracle[(qi * nh2 + h) * hd2 + d] = a * inv;
                    }
                }
            }
            let (flash_small, _) = crate::compute::prefill_flash_attention_with_ctx(
                &ctx, &q2, &k2, &v2, s2, kv2, nh2, nkv2, hd2, sc2,
            );
            let rel = flash_small
                .iter()
                .zip(&oracle)
                .map(|(s, o)| (s - o).abs() / o.abs().max(1e-3))
                .fold(0f32, f32::max);
            let max_abs = flash_small
                .iter()
                .zip(&oracle)
                .map(|(s, o)| (s - o).abs())
                .fold(0f32, f32::max);
            let mean_abs = flash_small
                .iter()
                .zip(&oracle)
                .map(|(s, o)| (s - o).abs())
                .sum::<f32>()
                / (flash_small.len() as f32);
            eprintln!(
                "[attention-tg] correctness vs f32 eager oracle (seq=64): max_rel={rel:.3e} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
            );
            assert!(
                rel < 0.1,
                "threadgroup flash correctness max_rel={rel} (≥0.1 = f16 precision 범위 초과)"
            );
        }

        // (b) 성능: 큰 케이스(1139) threadgroup flash. warmup 3 + median 10.
        let mut simd_v = Vec::new();
        for i in 0..13 {
            let (_, sms) = crate::compute::prefill_flash_attention_with_ctx(
                &ctx, &q, &k_f16, &v_f16, seq, kv, nh, nkv, hd, scale,
            );
            if i >= 3 {
                simd_v.push(sms);
            }
        }
        let s_med = median(simd_v);
        eprintln!(
            "[pm48-simd] 1-layer median(10): GPU simd={s_med:.3}ms GPU naive={g_med:.3}ms CPU f16={c_med:.3}ms"
        );
        eprintln!(
            "[pm48-simd] speedup: simd vs CPU={:.2}x | simd vs naive={:.2}x",
            c_med / s_med,
            g_med / s_med
        );
        let beats_cpu = s_med < c_med;
        let beats_naive_1_3x = (g_med / s_med) >= 1.3;
        let go = beats_cpu || beats_naive_1_3x;
        eprintln!(
            "[pm48-simd] GO/NO-GO = {} (simd<CPU={} | simd≥1.3x naive={})",
            if go { "GO" } else { "NO-GO" },
            beats_cpu,
            beats_naive_1_3x
        );
        eprintln!(
            "[pm48-simd] 16-layer 추정: GPU simd {:.1}ms vs CPU {:.1}ms",
            s_med * 16.0,
            c_med * 16.0
        );
    }

    /// pm47 ② STEP4+STEP5 GEMM 통합 커널이 CPU f32 oracle 과 의미동등 drift 내인지.
    /// reviewer 케이스: cs∈{8,16,37,48}(M 패딩 활성/꽉) + multi-chunk mixed(seq=86 cs=48 → 48+38)
    /// + gate=0 경계(decay=1, q_scaled=q 최댓값). 실제 gate=softplus(≥0)×ssm_a(A=-exp(A_log)<0)≤0
    /// 보장(gdn_prefill.rs apply_dt_gate_inplace)이라 q_scaled=exp(g_cum)·q ≤ q → f16 overflow
    /// 불가(reviewer M5 우려 해소). 게이트: out/state rel < 5e-2(hard, 의미동등).
    /// 누적 drift ≤ 8.4e-3(= 2× STEP5 단독 4.21e-3) 은 모니터(초과 시 fp32 부분적용 검토).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn delta_scan_step45gemm_matches_cpu_oracle() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm47] not tensorops-capable; skipping step45gemm oracle");
            return;
        }
        let (nh, hk, hv) = (4usize, 128usize, 128usize);
        // (cs, seq, gate_base, label). gate = gate_base + (i%5)*0.01 (음수 유지 / 양수 유지).
        let cases: [(usize, usize, f32, &str); 6] = [
            (8, 16, -0.1, "cs8 2chunk"),
            (16, 32, -0.1, "cs16 2chunk"),
            (37, 74, -0.1, "cs37 2chunk"),
            (48, 96, -0.1, "cs48 2chunk full-M"),
            (48, 86, -0.1, "cs48 mixed(48+38)"),
            (38, 76, 0.0, "cs38 gate=0 경계 decay1"),
        ];
        let mut worst_out = 0f32;
        let mut worst_state = 0f32;
        for (cs, seq, gate_base, label) in cases {
            let q: Vec<f32> = (0..seq * nh * hk)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
                .collect();
            let k: Vec<f32> = (0..seq * nh * hk)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect();
            let v: Vec<f32> = (0..seq * nh * hv)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
                .collect();
            // gate≤0 보장(실제 GDN: softplus×A<0). .min(0.0)으로 gate=0 경계(decay=1) 반영.
            let gate: Vec<f32> = (0..seq * nh)
                .map(|i| (gate_base + (i % 5) as f32 * 0.01).min(0.0))
                .collect();
            let beta: Vec<f32> = (0..seq * nh).map(|i| 0.5 + (i % 3) as f32 * 0.1).collect();
            let state0 = vec![0.0f32; nh * hv * hk];

            let mut state_cpu = state0.clone();
            let out_cpu = rnb_cpu::kernels::delta_net::delta_net_scan_chunkwise(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                &mut state_cpu,
                seq,
                nh,
                hk,
                hv,
                cs,
            );
            let (out_gpu, state_gpu) = crate::compute::delta_net_scan_chunk_step45gemm_full(
                &ctx, &q, &k, &v, &gate, &beta, &state0, seq, nh, hk, hv, cs,
            );
            let rel = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs() / y.abs().max(1e-3))
                    .fold(0f32, f32::max)
            };
            let out_rel = rel(&out_gpu, &out_cpu);
            let state_rel = rel(&state_gpu, &state_cpu);
            worst_out = worst_out.max(out_rel);
            worst_state = worst_state.max(state_rel);
            eprintln!(
                "[pm47] step45gemm vs oracle [{label}]: out_rel={out_rel:.2e} state_rel={state_rel:.2e} (seq={seq} cs={cs})"
            );
            assert!(out_rel < 5e-2, "step45gemm [{label}] out drift {out_rel}");
            assert!(
                state_rel < 5e-2,
                "step45gemm [{label}] state drift {state_rel}"
            );
        }
        eprintln!(
            "[pm47] step45gemm worst: out_rel={worst_out:.2e} state_rel={worst_state:.2e} (누적 모니터 ≤8.4e-3 = 2× STEP5 4.21e-3)"
        );
    }

    /// pm47 STEP4 inter microbench correctness: GEMM(device-staging transpose) == scalar mat-vec.
    /// staging transpose 방향(state[vi,ki]→b_dev[ki,vi]) 검증 + f16 cast drift 확인. cs 4종(M 패딩
    /// 활성/꽉). r<cs·vi<hv 만 비교(r>=cs 패딩 무시).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn step4_inter_gemm_matches_scalar() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm47] not tensorops-capable; skipping STEP4 inter microbench");
            return;
        }
        let (nh, hk, hv) = (4usize, 128usize, 128usize);
        const CPAD: usize = 48;
        for cs in [8usize, 16, 38, 48] {
            let q: Vec<f32> = (0..nh * cs * hk)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
                .collect();
            let state: Vec<f32> = (0..nh * hv * hk)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect();
            let (g, _) = crate::compute::run_step4_inter_gemm(&ctx, &q, &state, cs, nh);
            let (s, _) = crate::compute::run_step4_inter_scalar(&ctx, &q, &state, cs, nh);
            let mut max_rel = 0f32;
            for h in 0..nh {
                for r in 0..cs {
                    for vi in 0..hv {
                        let idx = h * CPAD * hv + r * hv + vi;
                        let denom = s[idx].abs().max(1e-4);
                        max_rel = max_rel.max((g[idx] - s[idx]).abs() / denom);
                    }
                }
            }
            eprintln!("[pm47] step4 inter cs={cs} gemm vs scalar max_rel={max_rel:.2e}");
            assert!(
                max_rel < 5e-2,
                "step4 inter cs={cs} mismatch max_rel={max_rel}"
            );
        }
    }

    /// pm47 STEP4 inter 성능 게이트 (reviewer 권고): inter 단독 GEMM vs scalar mat-vec GPU-time
    /// median. STEP5(outer, scalar 최악)는 21x였으나 inter는 mat-vec(연속 reduction)라 scalar가
    /// 이미 빠를 수 있음 → ratio 가 1.x대면 ≥3% prefill 게이트 통과 난망(delta scan 축 종료 신호),
    /// 5x+ 면 STEP4 GEMM 진행 정당. staging(device transpose) 포함 end-to-end.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device (perf)"]
    fn step4_inter_perf_gemm_vs_scalar() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm47] not tensorops-capable; skipping STEP4 inter perf");
            return;
        }
        let (nh, hk, hv, cs) = (48usize, 128usize, 128usize, 38usize); // 27B 실측
        let q: Vec<f32> = (0..nh * cs * hk)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let state: Vec<f32> = (0..nh * hv * hk)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let median = |mut x: Vec<f64>| {
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            x[x.len() / 2]
        };
        let mut sc = Vec::new();
        let mut gm = Vec::new();
        for i in 0..13 {
            let (_, s_ms) = crate::compute::run_step4_inter_scalar(&ctx, &q, &state, cs, nh);
            let (_, g_ms) = crate::compute::run_step4_inter_gemm(&ctx, &q, &state, cs, nh);
            if i >= 3 {
                sc.push(s_ms);
                gm.push(g_ms);
            }
        }
        let s_med = median(sc);
        let g_med = median(gm);
        eprintln!(
            "[pm47] STEP4 inter GPU-time median: scalar={s_med:.4}ms gemm={g_med:.4}ms ratio={:.2}x (nh={nh} cs={cs})",
            s_med / g_med
        );
    }

    /// pm34 M3: synthetic Q4_K dequant → staging(transposed) → matmul2d 가 CPU ref 와 일치하는지.
    /// dense PoC(M2) 통과 ≠ packed Q4_K 인덱싱 통과 — block(256)/sub-block(32) 좌표를 검증.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_poc_q4k_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm34] not tensorops-capable; skipping Q4_K PoC");
            return;
        }
        let (n, k, m) = (16usize, 256usize, 16usize); // n=weight rows, m=tokens, K=1 superblock
        let wb = quantize_rows_q4k(&det_vals(n * k, 0.05), n, k);
        let input = det_vals(m * k, 0.1);
        let want = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
        let got = crate::compute::run_tensorops_poc_q4k(&ctx, &wb, &input);
        assert_eq!(got.len(), m * n);
        let mut max_rel = 0f32;
        for i in 0..m * n {
            let denom = want[i].abs().max(1e-4);
            max_rel = max_rel.max((got[i] - want[i]).abs() / denom);
        }
        eprintln!("[pm34] q4k tensorops max_rel={max_rel:.2e}");
        eprintln!("[pm34] q4k C[0..4]={:?} want={:?}", &got[0..4], &want[0..4]);
        assert!(max_rel < 1e-3, "q4k tensorops mismatch: max_rel={max_rel}");
    }

    /// pm34 M4: 임의 K K-loop multiply_accumulate(S2). 실모델 K(4096+) 처리 기반.
    /// M=N=16 tile 고정, K∈{512, 1024, 4096} (multi-superblock). C host zero-init.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q4k_kloop_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm34] not tensorops-capable; skipping K-loop");
            return;
        }
        for k in [512usize, 1024, 4096] {
            let (n, m) = (16usize, 16usize);
            let wb = quantize_rows_q4k(&det_vals(n * k, 0.03), n, k);
            let input = det_vals(m * k, 0.07);
            let want = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_tensorops_poc_q4k_ktile(&ctx, &wb, &input, k);
            assert_eq!(got.len(), m * n);
            let mut max_rel = 0f32;
            for i in 0..m * n {
                let denom = want[i].abs().max(1e-3);
                max_rel = max_rel.max((got[i] - want[i]).abs() / denom);
            }
            eprintln!("[pm34] q4k K={k} K-loop max_rel={max_rel:.2e}");
            assert!(max_rel < 2e-3, "q4k K={k} mismatch: max_rel={max_rel}");
        }
    }

    /// pm34 M4: M/N grid tiling 임의 shape. 16 미배수(bound check) + 실모델급 shape.
    /// (N,K,M): (48,512,40) 16미배수, (5120,256,128) hidden급 N, (256,4096,16) K큰.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q4k_grid_tiling_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm34] not tensorops-capable; skipping grid tiling");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize),
            (5120, 256, 128),
            (256, 4096, 16),
        ] {
            let wb = quantize_rows_q4k(&det_vals(n * k, 0.02), n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q4k_tensorops(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_rel = 0f32;
            for i in 0..m * n {
                let denom = want[i].abs().max(1e-3);
                max_rel = max_rel.max((got[i] - want[i]).abs() / denom);
            }
            eprintln!("[pm34] q4k grid (N={n},K={k},M={m}) max_rel={max_rel:.2e}");
            assert!(
                max_rel < 2e-3,
                "q4k grid (N={n},K={k},M={m}) mismatch: max_rel={max_rel}"
            );
        }
    }

    /// pm40 M1: llama 패턴 Q4_K GEMM(v2 = device-direct B activation + cooperative tensor C)이
    /// CPU ref 와 동일한지. 비대칭 shape(N_out/K/M_tok 다름) + bound check(32/64 미배수). go/no-go.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q4k_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm40] not tensorops-capable; skipping v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize), // 32/64 미배수 bound check
            (5120, 256, 128),             // hidden급 N
            (256, 4096, 16),              // K 큰
            (6144, 2048, 70),             // FFN급 + M 미배수
        ] {
            let wb = quantize_rows_q4k(&det_vals(n * k, 0.02), n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q4k_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_rel = 0f32;
            for i in 0..m * n {
                let denom = want[i].abs().max(1e-3);
                max_rel = max_rel.max((got[i] - want[i]).abs() / denom);
            }
            eprintln!("[pm40] q4k v2 (N={n},K={k},M={m}) max_rel={max_rel:.2e}");
            assert!(
                max_rel < 5e-3,
                "q4k v2 (N={n},K={k},M={m}) mismatch: max_rel={max_rel}"
            );
        }
    }

    /// pm101: Q4_K gate/up pair dispatch는 기존 v2 GEMM 두 번과 같은 값을 내야 한다.
    /// weight는 raw Q4_K 그대로 두고, 커널 내부 threadgroup tile에서만 transient dequant한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q4k_v2_pair_matches_two_v2_runs() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm101] not tensorops-capable; skipping q4k v2 pair");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize),
            (512, 2048, 141),
            (6144, 2048, 70),
        ] {
            let gate_wb = quantize_rows_q4k(&det_vals(n * k, 0.017), n, k);
            let up_wb = quantize_rows_q4k(&det_vals(n * k, 0.023), n, k);
            let input = det_vals(m * k, 0.05);
            let gate_ref = crate::compute::run_q4k_tensorops_v2(&ctx, &gate_wb, &input, n, k, m);
            let up_ref = crate::compute::run_q4k_tensorops_v2(&ctx, &up_wb, &input, n, k, m);
            let (gate_pair, up_pair) =
                crate::compute::run_q4k_tensorops_v2_pair(&ctx, &gate_wb, &up_wb, &input, n, k, m);
            assert_eq!(gate_pair.len(), m * n);
            assert_eq!(up_pair.len(), m * n);

            let mut gate_max_rel = 0f32;
            let mut up_max_rel = 0f32;
            for i in 0..m * n {
                gate_max_rel = gate_max_rel
                    .max((gate_pair[i] - gate_ref[i]).abs() / gate_ref[i].abs().max(1e-3));
                up_max_rel =
                    up_max_rel.max((up_pair[i] - up_ref[i]).abs() / up_ref[i].abs().max(1e-3));
            }
            eprintln!(
                "[pm101] q4k v2 pair (N={n},K={k},M={m}) gate_rel={gate_max_rel:.2e} up_rel={up_max_rel:.2e}"
            );
            assert!(
                gate_max_rel < 1e-6,
                "q4k v2 pair gate mismatch: max_rel={gate_max_rel}"
            );
            assert!(
                up_max_rel < 1e-6,
                "q4k v2 pair up mismatch: max_rel={up_max_rel}"
            );
        }
    }

    /// pm42 M3 step1: Q5_K v2(device-direct B activation + cooperative tensor C, 64×128 winner 타일)이
    /// CPU ref 와 의미 동등. half staging GEMM → global_rel(max|err|/max|want|)<1e-2 (q5k v1 grid 동일 기준).
    /// shape: n<64/m<128 single-tile bound + m>128 multi-tile bound + 실모델급 K=5120(half 누적).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q5k_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm42] not tensorops-capable; skipping q5k v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize), // n<64, m<128 single-tile bound
            (256, 1024, 200),             // m>128 + n>64 multi-tile bound
            (2560, 5120, 37),             // 실모델급 K(half 누적 tol)
        ] {
            let wb = build_q5k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q5k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q5k_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm42] q5k v2 (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q5k v2 (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    /// pm42 M3 step1: Q6_K v2(NK=128=superblock 절반, cooperative tensor C, 64×128 타일)이 CPU ref 와
    /// 의미 동등. FFN down(Q6_K)/GDN in_proj(Q6_K). half staging GEMM → global_rel<1e-2 (q6k v1 grid 기준).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q6k_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm42] not tensorops-capable; skipping q6k v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize), // n<64, m<128 single-tile bound
            (256, 1024, 200),             // m>128 + n>64 multi-tile bound
            (2560, 5120, 37),             // 실모델급 K(half 누적 tol)
        ] {
            let wb = build_q6k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q6k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q6k_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm42] q6k v2 (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q6k v2 (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn build_q8_0_rows(n: usize, k: usize) -> Vec<u8> {
        let nb = k / 32;
        let mut out = Vec::with_capacity(n * nb * 34);
        for row in 0..n {
            for blk in 0..nb {
                let seed = row * 13 + blk * 7 + 1;
                let d_val = 0.01 + (seed % 7) as f32 * 0.002;
                out.extend_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
                for i in 0..32i32 {
                    let q = (((seed as i32 * 3 + i * 5) % 251) - 125) as i8;
                    out.push(q as u8);
                }
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    fn cpu_q8_0_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        use rnb_cpu::quantize::{dequantize_q8_0, BlockQ8_0};
        let nb = k / 32;
        let bpr = nb * 34;
        let mut out = vec![0f32; m * n];
        let mut deq = [0f32; 32];
        let mut deq_row = vec![0f32; k];
        for row in 0..n {
            let rb = &wb[row * bpr..(row + 1) * bpr];
            for b in 0..nb {
                let blk = unsafe { &*(rb[b * 34..].as_ptr() as *const BlockQ8_0) };
                dequantize_q8_0(blk, &mut deq);
                deq_row[b * 32..(b + 1) * 32].copy_from_slice(&deq);
            }
            for tok in 0..m {
                let inp = &input[tok * k..(tok + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += deq_row[kk] * inp[kk];
                }
                out[tok * n + row] = acc;
            }
        }
        out
    }

    /// Q8_0 v2(NK=32=block, cooperative C, 64×128 타일)이 CPU dequant dot 와 의미 동등.
    /// unsloth UD attn/GDN/shared projection(전부 Q8_0). global_rel<1e-2 (half staging).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q8_0_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[q8_0] not tensorops-capable; skipping q8_0 v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize), // n<64, m<128 single-tile bound
            (256, 2048, 200),             // m>128 + n>64 multi-tile, K=hidden
            (2560, 4096, 37),             // 실모델급 K(attn_output q_dim)
        ] {
            let wb = build_q8_0_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q8_0_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q8_0_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[q8_0] v2 (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q8_0 v2 (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn build_q3k_rows(n: usize, k: usize) -> Vec<u8> {
        let nb = k / 256;
        let mut out = Vec::with_capacity(n * nb * 110);
        for row in 0..n {
            for blk in 0..nb {
                let seed = row * 13 + blk * 7 + 1;
                for i in 0..32usize {
                    out.push(((seed * 3 + i * 5) % 256) as u8); // hmask
                }
                for i in 0..64usize {
                    out.push(((seed * 7 + i * 11) % 256) as u8); // qs (low 2 bits)
                }
                for i in 0..12usize {
                    out.push(((seed * 5 + i * 3) % 256) as u8); // scales (6-bit packed)
                }
                let d_val = 0.01 + (seed % 7) as f32 * 0.002;
                out.extend_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    fn cpu_q3k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        use rnb_cpu::quantize::blocks::BlockQ3_K;
        use rnb_cpu::quantize::dequant::dequantize_q3_k;
        let nb = k / 256;
        let bpr = nb * 110;
        let mut out = vec![0f32; m * n];
        let mut deq = [0f32; 256];
        let mut deq_row = vec![0f32; k];
        for row in 0..n {
            let rb = &wb[row * bpr..(row + 1) * bpr];
            for b in 0..nb {
                let blk = unsafe { &*(rb[b * 110..].as_ptr() as *const BlockQ3_K) };
                dequantize_q3_k(blk, &mut deq);
                deq_row[b * 256..(b + 1) * 256].copy_from_slice(&deq);
            }
            for tok in 0..m {
                let inp = &input[tok * k..(tok + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += deq_row[kk] * inp[kk];
                }
                out[tok * n + row] = acc;
            }
        }
        out
    }

    /// pm123: Q3_K v2(NK=128, 256-superblock 절반, cooperative C 64×128)이 CPU dequant dot 와
    /// 의미 동등. Q3_K prefill projection(common quant coverage). global_rel<1e-2 (half staging).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q3k_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm123] not tensorops-capable; skipping q3k v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize),
            (256, 1024, 200),
            (2560, 5120, 37),
        ] {
            let wb = build_q3k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q3k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q3k_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm123] q3k v2 (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q3k v2 (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn build_q2k_rows(n: usize, k: usize) -> Vec<u8> {
        let nb = k / 256;
        let mut out = Vec::with_capacity(n * nb * 84);
        for row in 0..n {
            for blk in 0..nb {
                let seed = row * 13 + blk * 7 + 1;
                for i in 0..16usize {
                    out.push(((seed * 5 + i * 3) % 256) as u8); // scales (low4=scale, high4=min)
                }
                for i in 0..64usize {
                    out.push(((seed * 7 + i * 11) % 256) as u8); // qs (2-bit)
                }
                let d_val = 0.01 + (seed % 7) as f32 * 0.002;
                let dmin_val = 0.005 + (seed % 5) as f32 * 0.001;
                out.extend_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
                out.extend_from_slice(&half::f16::from_f32(dmin_val).to_le_bytes());
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    fn cpu_q2k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
        use rnb_cpu::quantize::blocks::BlockQ2_K;
        use rnb_cpu::quantize::dequant::dequantize_q2_k;
        let nb = k / 256;
        let bpr = nb * 84;
        let mut out = vec![0f32; m * n];
        let mut deq = [0f32; 256];
        let mut deq_row = vec![0f32; k];
        for row in 0..n {
            let rb = &wb[row * bpr..(row + 1) * bpr];
            for b in 0..nb {
                let blk = unsafe { &*(rb[b * 84..].as_ptr() as *const BlockQ2_K) };
                dequantize_q2_k(blk, &mut deq);
                deq_row[b * 256..(b + 1) * 256].copy_from_slice(&deq);
            }
            for tok in 0..m {
                let inp = &input[tok * k..(tok + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += deq_row[kk] * inp[kk];
                }
                out[tok * n + row] = acc;
            }
        }
        out
    }

    /// pm123: Q2_K v2(NK=128, 256-superblock 절반) CPU dequant dot 대비 의미 동등. common quant coverage.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q2k_v2_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm123] not tensorops-capable; skipping q2k v2");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize),
            (256, 1024, 200),
            (2560, 5120, 37),
        ] {
            let wb = build_q2k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q2k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q2k_tensorops_v2(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm123] q2k v2 (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q2k v2 (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    /// pm123: Q3_K decode GEMV(1 simdgroup/row)이 CPU dequant dot 와 의미 동등.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn gemv_q3k_simd_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        for (n, k) in [(48usize, 512usize), (256, 2048), (2560, 5120)] {
            let wb = build_q3k_rows(n, k);
            let input = det_vals(k, 0.05);
            let want = cpu_q3k_gemm_reference(&wb, n, k, &input, 1);
            let got =
                crate::compute::run_gemv_simd(&ctx, &ctx.q3k_simd_pipeline, &wb, &input, n, k);
            assert_eq!(got.len(), n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm123] q3k gemv (N={n},K={k}) global_rel={global_rel:.2e}");
            assert!(global_rel < 1e-2, "q3k gemv (N={n},K={k}) rel={global_rel}");
        }
    }

    /// pm123: Q2_K decode GEMV(1 simdgroup/row)이 CPU dequant dot 와 의미 동등.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn gemv_q2k_simd_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        for (n, k) in [(48usize, 512usize), (256, 2048), (2560, 5120)] {
            let wb = build_q2k_rows(n, k);
            let input = det_vals(k, 0.05);
            let want = cpu_q2k_gemm_reference(&wb, n, k, &input, 1);
            let got =
                crate::compute::run_gemv_simd(&ctx, &ctx.q2k_simd_pipeline, &wb, &input, n, k);
            assert_eq!(got.len(), n);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm123] q2k gemv (N={n},K={k}) global_rel={global_rel:.2e}");
            assert!(global_rel < 1e-2, "q2k gemv (N={n},K={k}) rel={global_rel}");
        }
    }

    /// pm123: iq_common.metal 추출 후 glm+common concat 이 런타임 컴파일되는지 검증(리팩터 회귀).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn glm_iq_common_concat_compiles() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        crate::compute::iq_glm_concat_compiles(&ctx);
    }

    /// pm40 M1: v1(현 64×32 A+B+C threadgroup) vs v2(llama device-direct B + cooperative C) per-call.
    /// 구조 개선(threadgroup 20KB→8KB, barrier chunk당 2→1, B staging 제거)의 가치 측정.
    /// RNB_PM40_BENCH_SHAPE="N,K,M" override(default 27B FFN gate급).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "perf micro-bench, requires M5 Metal device"]
    fn bench_q4k_tensorops_v1_vs_v2() {
        use std::time::Instant;
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm40] not tensorops-capable; skipping bench");
            return;
        }
        let shape = std::env::var("RNB_PM40_BENCH_SHAPE").ok();
        let (n, k, m) = shape
            .as_deref()
            .and_then(|s| {
                let p: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (p.len() == 3).then(|| (p[0], p[1], p[2]))
            })
            .unwrap_or((13824, 5120, 1139)); // 27B FFN gate(ffn,hidden) × 1139 tok
        let wb = quantize_rows_q4k(&det_vals(n * k, 0.02), n, k);
        let input = det_vals(m * k, 0.05);
        let iters = 10;
        let bench = |f: &dyn Fn() -> Vec<f32>| -> f64 {
            let _ = f();
            let t = Instant::now();
            for _ in 0..iters {
                let _ = f();
            }
            t.elapsed().as_secs_f64() * 1000.0 / iters as f64
        };
        let v1_out = crate::compute::run_q4k_tensorops(&ctx, &wb, &input, n, k, m);
        let v1 = bench(&|| crate::compute::run_q4k_tensorops(&ctx, &wb, &input, n, k, m));
        eprintln!("[pm40 M2 bench] N={n} K={k} M={m} | v1(현 64x32 A+B+C tg 20KB) {v1:.2}ms");
        // 타일 스윕 결과(perf-journal pm41): 64×128 4sg=correct winner. NSG≠4(8/16sg)는
        // 더 빠르나(최대 5.18x) rel=1.0 틀림 — matmul2d NSG는 타일과 구조적 연동(별도 R&D).
        for (name, fn_name, nra, nrb) in [
            (
                "v2_64x32(llama구조)",
                "gemm_q4k_tensorops_v2",
                64usize,
                32usize,
            ),
            ("v2_64x128(winner)", "gemm_q4k_tensorops_v2_64x128", 64, 128),
        ] {
            let out = crate::compute::run_q4k_tensorops_v2_variant(
                &ctx, &wb, &input, n, k, m, fn_name, nra, nrb,
            );
            let mut max_rel = 0f32;
            for i in 0..m * n {
                let denom = v1_out[i].abs().max(1e-3);
                max_rel = max_rel.max((out[i] - v1_out[i]).abs() / denom);
            }
            let ms = bench(&|| {
                crate::compute::run_q4k_tensorops_v2_variant(
                    &ctx, &wb, &input, n, k, m, fn_name, nra, nrb,
                )
            });
            eprintln!(
                "  {name}: {ms:.2}ms | rel_vs_v1={max_rel:.2e} | v1/this={:.2}x",
                v1 / ms
            );
        }
    }

    /// pm34 M5: Q6_K M/N grid tiling. FFN mixed precision(Q4_K gate/up + Q6_K down 등).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q6k_grid_tiling_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm34] not tensorops-capable; skipping q6k grid");
            return;
        }
        for (n, k, m) in [(48usize, 512usize, 40usize), (256, 1024, 64)] {
            let wb = build_q6k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q6k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q6k_tensorops(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            // half staging GEMM: element-wise rel 은 want≈0 에서 ill-defined → global rel
            // (max|err| / max|want|) 로 판정(half GEMM 표준). chain 정확도는 의미 동등성이 hard.
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm34] q6k grid (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q6k grid (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    /// pm36: Q5_K M/N grid tiling tensorops. GDN ssm_out(Q5_K) + 9B 전체 GDN projection.
    /// shape: 16 미배수 bound check + 실모델급 K=5120(half 누적 tol 검증).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_q5k_grid_tiling_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm36] not tensorops-capable; skipping q5k grid");
            return;
        }
        for (n, k, m) in [
            (48usize, 512usize, 40usize),
            (256, 1024, 64),
            (2560, 5120, 37),
        ] {
            let wb = build_q5k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q5k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q5k_tensorops(&ctx, &wb, &input, n, k, m);
            assert_eq!(got.len(), m * n);
            // half staging GEMM: global rel(max|err| / max|want|) 판정(q6k 와 동일 기준).
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm36] q5k grid (N={n},K={k},M={m}) global_rel={global_rel:.2e} max_abs={max_abs:.4} max_w={max_w:.2}");
            assert!(
                global_rel < 1e-2,
                "q5k grid (N={n},K={k},M={m}) mismatch: global_rel={global_rel}"
            );
        }
    }

    /// pm35 M2 Task1: GDN in_proj(N=conv_ch)/gate(N=d_inner) shape 의 단발 tensorops GEMM sanity.
    /// in_proj/gate 는 hidden→N batch GEMM. 27B 대표: hidden=5120, conv_ch=10240(Q6_K), d_inner=6144(Q4_K).
    /// K=5120(실제 K — half 누적이 tol 1e-2 넘는지 실제 K로 검증). N 만 축소(타일 bound 유지).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn tensorops_gdn_proj_shape_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm35] not tensorops-capable; skipping gdn proj shape");
            return;
        }
        // gate: Q4_K, N=d_inner 급, K=hidden=5120
        for (n, k, m) in [(2560usize, 5120usize, 37usize)] {
            let wb = quantize_rows_q4k(&det_vals(n * k, 0.02), n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q4k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q4k_tensorops(&ctx, &wb, &input, n, k, m);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm35] gdn gate Q4_K (N={n},K={k},M={m}) global_rel={global_rel:.2e}");
            assert!(global_rel < 1e-2, "gdn gate Q4_K mismatch: {global_rel}");
        }
        // in_proj: Q6_K, K=hidden=5120(실제 K), N=conv_ch 축소만
        for (n, k, m) in [(2560usize, 5120usize, 37usize)] {
            let wb = build_q6k_rows(n, k);
            let input = det_vals(m * k, 0.05);
            let want = cpu_q6k_gemm_reference(&wb, n, k, &input, m);
            let got = crate::compute::run_q6k_tensorops(&ctx, &wb, &input, n, k, m);
            let mut max_abs = 0f32;
            let mut max_w = 0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((got[i] - want[i]).abs());
                max_w = max_w.max(want[i].abs());
            }
            let global_rel = max_abs / max_w.max(1e-3);
            eprintln!("[pm35] gdn in_proj Q6_K (N={n},K={k},M={m}) global_rel={global_rel:.2e}");
            assert!(global_rel < 1e-2, "gdn in_proj Q6_K mismatch: {global_rel}");
        }
    }

    /// pm35 M2 Task2: PrefillGdnProjCarrier + prefill_gdn_proj_dispatch single-GEMM correctness.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn gdn_proj_dispatch_matches_cpu_reference() {
        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        if !ctx.tensorops_capable {
            eprintln!("[pm35] not tensorops-capable; skip");
            return;
        }
        let (hidden, n_out, seq) = (2560usize, 2560usize, 37usize);
        let wb = quantize_rows_q4k(&det_vals(n_out * hidden, 0.02), n_out, hidden);
        let normed = det_vals(seq * hidden, 0.05);
        let want = cpu_q4k_gemm_reference(&wb, n_out, hidden, &normed, seq);
        let (w_buf, off) = crate::compute::wrap_nocopy(&ctx, &wb);
        let off_buf = crate::gdn_proj_chain::u32_buf(&ctx, off);
        let carrier = crate::gdn_proj_chain::PrefillGdnProjCarrier::new(&ctx, hidden, n_out, seq);
        let got = crate::gdn_proj_chain::prefill_gdn_proj_dispatch(
            &ctx,
            &carrier,
            &normed,
            &w_buf,
            &off_buf,
            crate::gdn_proj_chain::TensoropsQuant::Q4K,
            seq,
            None,
        );
        let mut max_abs = 0f32;
        let mut max_w = 0f32;
        for i in 0..seq * n_out {
            max_abs = max_abs.max((got[i] - want[i]).abs());
            max_w = max_w.max(want[i].abs());
        }
        let gr = max_abs / max_w.max(1e-3);
        eprintln!("[pm35] gdn_proj dispatch Q4_K global_rel={gr:.2e}");
        assert!(gr < 1e-2, "gdn_proj Q4_K mismatch: {gr}");
    }

    /// pm35 M2 Task3: MetalBackend::prefill_gdn_proj (carrier pool + resident wrap) end-to-end.
    /// METAL thread_local 은 runtime 전용 → metal 크레이트 테스트는 MetalBackend::new() 직접 생성.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn metal_backend_prefill_gdn_proj_matches_cpu() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            eprintln!("[pm35] no metal ctx; skip");
            return;
        }
        let (hidden, n_out, seq) = (2560usize, 2560usize, 37usize);
        let wb = quantize_rows_q4k(&det_vals(n_out * hidden, 0.02), n_out, hidden);
        let normed = det_vals(seq * hidden, 0.05);
        let want = cpu_q4k_gemm_reference(&wb, n_out, hidden, &normed, seq);
        let got = backend.prefill_gdn_proj(&normed, &wb, TensoropsQuant::Q4K, seq, hidden, n_out);
        let mut max_abs = 0f32;
        let mut max_w = 0f32;
        for i in 0..seq * n_out {
            max_abs = max_abs.max((got[i] - want[i]).abs());
            max_w = max_w.max(want[i].abs());
        }
        assert!(max_abs / max_w.max(1e-3) < 1e-2);
    }

    /// pm21: M5 순수 read 메모리 BW 천장 측정(GEMV BW 67% 가 진짜 낮은지/거의 천장인지 판가름).
    /// 1GB device buffer 를 grid-stride float4 read + reduce. GPU time 으로 effective read BW.
    /// GEMV(P1 182GB/s)와 비교: 천장 가까우면 GEMV 더 못 짬, 멀면 kernel 여지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_memory_bandwidth_ceiling() {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
            MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
        };
        use std::ptr::NonNull;

        let device = MTLCreateSystemDefaultDevice().expect("no metal device");
        let queue = device.newCommandQueue().expect("queue");
        let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mem_bw_read(
    device const float4* data [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant uint& n4         [[buffer(2)]],
    uint gid   [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    float4 acc = float4(0.0);
    for (uint i = gid; i < n4; i += gsize) {
        acc += data[i];
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
"#;
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(src), None)
            .expect("compile");
        let func = lib
            .newFunctionWithName(&NSString::from_str("mem_bw_read"))
            .expect("fn");
        let pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>> = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("pipeline");

        let shared = MTLResourceOptions::StorageModeShared;
        let gb: usize = 1024 * 1024 * 1024; // 1 GiB
        let n4: usize = gb / 16; // float4 count
        let data: Retained<ProtocolObject<dyn MTLBuffer>> = device
            .newBufferWithLength_options(gb, shared)
            .expect("data buf");

        // grid 크기 스윕(occupancy 채우기 위해 충분히 큰 thread 수).
        for &total_threads in &[131072usize, 262144, 524288, 1048576] {
            let out: Retained<ProtocolObject<dyn MTLBuffer>> = device
                .newBufferWithLength_options(total_threads * 4, shared)
                .expect("out buf");
            let n4_u = n4 as u32;
            let n4_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                let ptr = NonNull::new(&n4_u as *const u32 as *mut std::ffi::c_void).unwrap();
                device
                    .newBufferWithBytes_length_options(ptr, 4, shared)
                    .expect("n4 buf")
            };
            let tg = 256usize;
            let mut best_bw = 0.0f64;
            // warmup 1 + 측정 5, best(min time = max BW).
            for run in 0..6 {
                let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                    queue.commandBuffer().expect("cmd");
                let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
                    cmd.computeCommandEncoder().expect("enc");
                enc.setComputePipelineState(&pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&data), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&n4_buf), 0, 2);
                }
                let grid = MTLSize {
                    width: total_threads / tg,
                    height: 1,
                    depth: 1,
                };
                let tgs = MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if run == 0 {
                    continue; // warmup
                }
                let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
                let bw = gb as f64 / (gpu_ms / 1000.0) / 1e9;
                if bw > best_bw {
                    best_bw = bw;
                }
            }
            eprintln!("[mem-bw] threads={total_threads:>8} best_read_BW={best_bw:.0} GB/s");
        }
        eprintln!("[mem-bw] 참고: P1 GEMV = 182 GB/s (weight 4.26GB / 23.43ms)");
    }

    /// pm21: GEMV BW 63% 병목 진단. 같은 weight access(144B block, lane stride)를
    /// **byte load(144 uchar) vs float4 load(9 float4)** 로 측정. float4 가 byte 보다
    /// 훨씬 빠르면 → weight 를 vectorized load(uint4/float4) + 레지스터 unpack 하는 게
    /// GEMV lever. 순수 read 천장 290, P1 GEMV(chain) 182 참고.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_q4k_read_only_bandwidth() {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
            MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
        };
        use std::ptr::NonNull;

        let device = MTLCreateSystemDefaultDevice().expect("no metal device");
        let queue = device.newCommandQueue().expect("queue");
        // 두 커널: 동일 stride 분할, weight load 방식만 byte(144 uchar) vs float4(9 float4).
        let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void q4k_ro_byte(
    device const uchar* w [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& N [[buffer(2)]], constant uint& K [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;
    uint nb = K / 256u;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        device const uchar* blk = w + (row * nb + b) * 144u;
        for (uint i = 0; i < 144u; i++) acc += (float)blk[i];
    }
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}
kernel void q4k_ro_vec4(
    device const uchar* w [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& N [[buffer(2)]], constant uint& K [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;
    uint nb = K / 256u;
    float4 acc = float4(0.0f);
    for (uint b = lane; b < nb; b += 32u) {
        device const float4* b4 = (device const float4*)(w + (row * nb + b) * 144u);
        for (uint i = 0; i < 9u; i++) acc += b4[i];   // 144B = 9 float4
    }
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) out[row] = total;
}
"#;
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(src), None)
            .expect("compile");
        let shared = MTLResourceOptions::StorageModeShared;
        let k: usize = 4096;
        let n: usize = 131072; // weight = N*16*144 = 288 MiB
        let nb = k / 256;
        let wbytes = n * nb * 144;
        let w: Retained<ProtocolObject<dyn MTLBuffer>> = device
            .newBufferWithLength_options(wbytes, shared)
            .expect("w buf");
        let out: Retained<ProtocolObject<dyn MTLBuffer>> = device
            .newBufferWithLength_options(n * 4, shared)
            .expect("out buf");
        let mk_u32 = |v: u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(&v as *const u32 as *mut std::ffi::c_void).unwrap();
                device
                    .newBufferWithBytes_length_options(ptr, 4, shared)
                    .expect("u32")
            }
        };
        let n_buf = mk_u32(n as u32);
        let k_buf = mk_u32(k as u32);

        for fname in ["q4k_ro_byte", "q4k_ro_vec4"] {
            let func = lib
                .newFunctionWithName(&NSString::from_str(fname))
                .expect("fn");
            let pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>> = device
                .newComputePipelineStateWithFunction_error(&func)
                .expect("pipeline");
            let mut best_bw = 0.0f64;
            for run in 0..6 {
                let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                    queue.commandBuffer().expect("cmd");
                let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
                    cmd.computeCommandEncoder().expect("enc");
                enc.setComputePipelineState(&pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&w), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 3);
                }
                let grid = MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if run == 0 {
                    continue;
                }
                let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
                let bw = wbytes as f64 / (gpu_ms / 1000.0) / 1e9;
                if bw > best_bw {
                    best_bw = bw;
                }
            }
            eprintln!("[q4k-ro] {fname:14} best_read_BW={best_bw:.0} GB/s (순수 290, P1 GEMV 182)");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q4k_matches_cpu_reference() {
        let block = tests_fixture::q4k_block_fixed();
        let input = vec![1.0f32; 256];
        let reference = tests_fixture::q4k_dequant_sum(&block);
        let gpu = MetalBackend::new().gemv_q4k(&block, &input, 1, 256);
        assert_eq!(gpu.len(), 1);
        let rel = (gpu[0] - reference).abs() / reference.abs().max(1e-6);
        assert!(rel < 1e-3, "gpu={} ref={} rel={rel}", gpu[0], reference);
    }

    /// pm27: chain carrier 시뮬 — 작은 GEMV(chunk_n row)를 한 command buffer 에 여러 개
    /// 연속 dispatch(각 다른 weight offset=cold read). chain decode 처럼 boost clock(긴
    /// 연속 부하) + carrier 묶음(1 commit) 조건을 재현해, p1_bandwidth 의 "작은 dispatch
    /// 18%"가 GPU clock(짧은 단일이라 idle)때문인지 dispatch overhead 본질인지 격리한다.
    ///   chunk_n=total → 단일 큰 dispatch(=p1_bandwidth, ~80%) baseline.
    ///   chunk_n 작음 → boost+묶음에서도 80%면 clock artifact(chain 48%는 작은 N/attn 등),
    ///                  낮으면 작은 dispatch 본질(lever=fuse/speculative).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_gemv_q4k_chain_sim_bandwidth() {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLResourceOptions, MTLSize,
        };
        use std::ptr::NonNull;

        let Some(ctx) = compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host");
        };
        let shared = MTLResourceOptions::StorageModeShared;
        let mk_u32 = |v: u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let stored = v;
            unsafe {
                let ptr = NonNull::new(&stored as *const u32 as *mut std::ffi::c_void).unwrap();
                ctx.device
                    .newBufferWithBytes_length_options(ptr, 4, shared)
                    .expect("u32")
            }
        };
        let off_buf = mk_u32(0);

        let k = 5120usize;
        let nb = k / 256;
        let row_bytes = nb * 144;
        // 576MB weight 풀: chunk 별 다른 offset 으로 cold read.
        // serial vs concurrent(MTLDispatchType.Concurrent): concurrent 면 dispatch 들이
        // 독립(다른 weight offset·out offset, read-only weight)이라 GPU 가 경계 drain/launch
        // idle 을 overlap → 작은 dispatch 의 overhead 가 숨겨지는지(lever 상한) 측정.
        let total_n = (576 * 1024 * 1024) / row_bytes;
        let wbytes = total_n * row_bytes;
        let w: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
            .device
            .newBufferWithLength_options(wbytes, shared)
            .expect("w buf");
        unsafe {
            std::ptr::write_bytes(w.contents().as_ptr() as *mut u8, 0xAB, wbytes);
        }
        let input = vec![1.0f32; k];
        let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
            let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).unwrap();
            ctx.device
                .newBufferWithBytes_length_options(ptr, k * 4, shared)
                .expect("i buf")
        };
        let out: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
            .device
            .newBufferWithLength_options(total_n * 4, shared)
            .expect("out buf");
        let k_buf = mk_u32(k as u32);

        // chunk_n: 2048(12MB)·8192(47MB)·17408(=27B ffn N, 100MB)·total(단일 576MB baseline)
        for &chunk_n in &[2048usize, 8192, 17408, total_n] {
            let n_chunks = (total_n / chunk_n).max(1);
            let used_bytes = n_chunks * chunk_n * row_bytes;
            let nchunk_buf = mk_u32(chunk_n as u32);
            for &(dt_label, concurrent) in &[("serial    ", false), ("concurrent", true)] {
                eprint!(
                    "[chain-sim] {dt_label} chunk_n={chunk_n:7} n_chunks={n_chunks:3} ({:4}MB tot) raw:",
                    used_bytes / (1024 * 1024)
                );
                let mut samples: Vec<f64> = Vec::new();
                for run in 0..12usize {
                    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                        ctx.queue.commandBuffer().expect("cmd");
                    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = if concurrent
                    {
                        cmd.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                            .expect("enc")
                    } else {
                        cmd.computeCommandEncoder().expect("enc")
                    };
                    for c in 0..n_chunks {
                        let woff = c * chunk_n * row_bytes;
                        let ooff = c * chunk_n * 4;
                        enc.setComputePipelineState(&ctx.q4k_simd_pipeline);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&w), woff, 0);
                            enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&out), ooff, 2);
                            enc.setBuffer_offset_atIndex(Some(&nchunk_buf), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
                            enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
                        }
                        let grid = MTLSize {
                            width: chunk_n,
                            height: 1,
                            depth: 1,
                        };
                        let tg = MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        };
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
                    let bw = used_bytes as f64 / (gpu_ms / 1000.0) / 1e9;
                    eprint!(" {bw:.0}");
                    if run >= 5 {
                        samples.push(bw);
                    }
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = samples[samples.len() / 2];
                eprintln!("  | median(run5+)={med:.0} ({:.0}%)", (med / 290.0) * 100.0);
            }
        }
    }

    /// pm27: 27B 주력 GEMV K shape(5120 nb20 / 6144 nb24 / 17408 nb68)에서 P1
    /// (gemv_q4k_simd: stride fallback) 의 weight read BW(GB/s). ~288MiB weight 로
    /// L2/SLC cache 회피 → DRAM BW. M5 unified ~290 GB/s 천장 참고. 8 run median.
    /// (pm27 M1a 측정: P1 micro 66-86% → "coalescing 48%" 설계 전제 반증. chain 48%는
    /// 커널 아닌 chain 레벨(N/occupancy/dispatch) 문제 — 다음 axis 진단 도구로 보존.)
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device; run on real hardware"]
    fn metal_gemv_q4k_p1_bandwidth() {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ptr::NonNull;

        let Some(ctx) = compute::build_metal_context() else {
            panic!("no Metal device — run on macOS host");
        };
        let shared = MTLResourceOptions::StorageModeShared;
        let mk_u32 = |v: u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let stored = v;
            unsafe {
                let ptr = NonNull::new(&stored as *const u32 as *mut std::ffi::c_void).unwrap();
                ctx.device
                    .newBufferWithBytes_length_options(ptr, 4, shared)
                    .expect("u32")
            }
        };
        let off_buf = mk_u32(0);
        // cache 효과 진단: 같은 K=5120(nb=20)에서 weight 크기(MB)를 스윕. 작은 weight(SLC fit)가
        // 큰 weight(cold DRAM)보다 빠르면 micro 가 cache 도움을 받는 것 → chain(매 layer cold
        // weight)과의 gap 원인. 14MB(<SLC) ~ 576MB(≫SLC). + 27B 실제 shape(6144/17408).
        for &(k, mb) in &[
            (5120usize, 14usize),
            (5120, 64),
            (5120, 288),
            (5120, 576),
            (6144, 288),
            (17408, 288),
        ] {
            let nb = k / 256;
            let row_bytes = nb * 144;
            let n = ((mb * 1024 * 1024) / row_bytes).max(1);
            let wbytes = n * row_bytes;
            // weight read pattern 만 중요(BW 측정). 단 lazy page mapping 으로 측정이
            // 출렁이지 않게 전체를 touch(물리 페이지 강제 매핑) — 값 자체는 무관.
            let w: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
                .device
                .newBufferWithLength_options(wbytes, shared)
                .expect("w buf");
            unsafe {
                std::ptr::write_bytes(w.contents().as_ptr() as *mut u8, 0xAB, wbytes);
            }
            let input = vec![1.0f32; k];
            let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
                let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).unwrap();
                ctx.device
                    .newBufferWithBytes_length_options(ptr, k * 4, shared)
                    .expect("i buf")
            };
            let out: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
                .device
                .newBufferWithLength_options(n * 4, shared)
                .expect("out buf");
            let n_buf = mk_u32(n as u32);
            let k_buf = mk_u32(k as u32);

            // P1-stride(gemv_q4k_simd) weight read BW. 12 run, 앞 5 = GPU clock ramp warmup
            // 제외(idle clock→boost 까지 3-4 dispatch 걸림, 진단 확인), 7 run median. buffer
            // touch 로 lazy page 제거. raw 도 출력해 ramp/노이즈 패턴 확인.
            eprint!("[gemv-bw] K={k:5} nb={nb:2} W={mb:4}MB N={n:7} P1 raw:");
            let mut samples: Vec<f64> = Vec::new();
            for run in 0..12usize {
                let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                    ctx.queue.commandBuffer().expect("cmd");
                let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
                    cmd.computeCommandEncoder().expect("enc");
                enc.setComputePipelineState(&ctx.q4k_simd_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&w), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
                }
                let grid = MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
                let bw = wbytes as f64 / (gpu_ms / 1000.0) / 1e9;
                eprint!(" {bw:.0}");
                if run >= 5 {
                    samples.push(bw); // 앞 5 = clock ramp warmup
                }
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = samples[samples.len() / 2];
            eprintln!("  | median(run5+)={med:.0} ({:.0}%)", (med / 290.0) * 100.0);
        }
    }

    /// M1: single-token decode attention 커널이 CPU `attention_decode_flash`
    /// (production default f16-acc online softmax) 와 token-identical 한지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_attn_decode_matches_cpu_reference() {
        let fx = tests_fixture::attn_decode_fixture();
        // pm25: f16 attn_decode 경로 검증 — carrier default ON 후 new()는 int8 ctx 라
        // f16 헬퍼 assert(!ctx.kv_int8)에 걸린다. f16 ctx 를 명시한다.
        let gpu = MetalBackend::new_with_kv_int8(false).attn_decode(
            &fx.q,
            &fx.k_cache,
            &fx.v_cache,
            fx.num_heads,
            fx.num_kv_heads,
            fx.head_dim,
            fx.kv_len,
            fx.scale,
        );
        assert_eq!(gpu.len(), fx.reference.len());
        let mut max_rel = 0.0f32;
        for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1e-6);
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-3 || abs < 1e-3,
                "idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
            );
        }
        eprintln!("attn_decode max_rel = {max_rel}");
    }

    /// M1: attn_decode 가 다양한 shape(GQA factor·head_dim·kv_len)에서 일치하는지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_attn_decode_matches_cpu_reference_various_shapes() {
        // pm25: f16 attn_decode 경로 검증 — f16 ctx 명시(carrier default ON 회피).
        let backend = MetalBackend::new_with_kv_int8(false);
        // (num_heads, num_kv_heads, head_dim, kv_len)
        for &(nh, nkv, hd, kl) in &[
            (8usize, 2usize, 64usize, 5usize), // GQA factor 4, head_dim 64
            (4, 4, 128, 7),                    // MHA, 더 긴 kv
            (2, 1, 256, 2),                    // head_dim 256(lane-local 상한)
            (6, 3, 96, 1),                     // kv_len 1, head_dim 96
        ] {
            let fx = tests_fixture::attn_decode_fixture_shaped(nh, nkv, hd, kl);
            let gpu =
                backend.attn_decode(&fx.q, &fx.k_cache, &fx.v_cache, nh, nkv, hd, kl, fx.scale);
            assert_eq!(gpu.len(), fx.reference.len());
            let mut max_rel = 0.0f32;
            for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(
                    rel < 1e-3 || abs < 1e-3,
                    "shape=({nh},{nkv},{hd},{kl}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
            eprintln!("shape=({nh},{nkv},{hd},{kl}) max_rel={max_rel}");
        }
    }

    /// M2 KV residency: incremental fill(kv_len 1→2→3 순차 호출, 매 호출 1개 append)
    /// 후 마지막 attention 이 host(전체 KV 업로드, kv_len=3) 와 동일. capacity > kv_len.
    #[test]
    fn attn_resident_capacity_tracks_live_kv_window() {
        assert_eq!(attn_resident_capacity_for(0, 1_048_576), 2048);
        assert_eq!(attn_resident_capacity_for(4557, 1_048_576), 8192);
        assert_eq!(attn_resident_capacity_for(4557, 8192), 8192);
        assert_eq!(attn_resident_capacity_for(4557, 5000), 5000);
        assert_eq!(attn_resident_capacity_for(8193, 1_048_576), 16384);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_attn_decode_kv_resident_matches_host() {
        // pm25: f16 KvResident 경로 검증 — f16 ctx 명시(carrier default ON 회피).
        let backend = MetalBackend::new_with_kv_int8(false);
        let fx = tests_fixture::attn_decode_fixture();
        let capacity = 16usize; // > kv_len(3): 커널은 kv_len 까지만 읽어야 함
        let mut gpu = Vec::new();
        for kvl in 1..=fx.kv_len {
            gpu = backend
                .attn_decode_kv_resident(
                    0,
                    &fx.k_cache,
                    &fx.v_cache,
                    &fx.q,
                    fx.num_heads,
                    fx.num_kv_heads,
                    fx.head_dim,
                    kvl,
                    fx.scale,
                    capacity,
                )
                .expect("f16 KV resident path should run");
        }
        assert_eq!(gpu.len(), fx.reference.len());
        let mut max_rel = 0.0f32;
        for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1e-6);
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-3 || abs < 1e-3,
                "idx={i} gpu={g} ref={r} rel={rel}"
            );
        }
        eprintln!("attn_decode_resident max_rel = {max_rel}");
    }

    /// Default Metal context may enable int8 KV for carrier paths. Standalone
    /// f16 resident attention must still allocate f16 buffers instead of the
    /// int8 dummy buffers.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_attn_decode_kv_resident_runs_with_default_ctx() {
        let backend = MetalBackend::new();
        let fx = tests_fixture::attn_decode_fixture();
        let gpu = backend
            .attn_decode_kv_resident(
                0,
                &fx.k_cache,
                &fx.v_cache,
                &fx.q,
                fx.num_heads,
                fx.num_kv_heads,
                fx.head_dim,
                fx.kv_len,
                fx.scale,
                16,
            )
            .expect("default ctx resident attention should run on f16 buffers");
        assert_eq!(gpu.len(), fx.reference.len());
        for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1e-6);
            assert!(
                rel < 1e-3 || abs < 1e-3,
                "idx={i} gpu={g} ref={r} rel={rel}"
            );
        }
    }

    /// pm17 gated attn: split_qgate 가 host 인터리브 분리(post_qkv.rs:34-41)와
    /// bit-identical 한지 (단순 copy).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_split_qgate_matches_cpu() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            return;
        }
        for (num_heads, hd) in [(16usize, 256usize), (4, 128)] {
            let q_full: Vec<f32> = (0..num_heads * hd * 2)
                .map(|i| (i as f32) * 0.01 - 5.0)
                .collect();
            let mut q_ref = vec![0f32; num_heads * hd];
            let mut g_ref = vec![0f32; num_heads * hd];
            for h in 0..num_heads {
                for d in 0..hd {
                    q_ref[h * hd + d] = q_full[h * hd * 2 + d];
                    g_ref[h * hd + d] = q_full[h * hd * 2 + hd + d];
                }
            }
            let (q_got, g_got) = backend.metal_split_qgate_into(&q_full, num_heads, hd);
            assert_eq!(q_got, q_ref, "query nh={num_heads} hd={hd}");
            assert_eq!(g_got, g_ref, "gate nh={num_heads} hd={hd}");
        }
    }

    // host `rope_partial_inplace`(rope.rs:387-419) 1:1 재현(decode 단일 토큰).
    // theta_scale = theta.powf(-2/n_rot) (clamp 후, f32).
    #[cfg(test)]
    fn cpu_rope_partial(
        data: &mut [f32],
        pos: usize,
        head_dim: usize,
        dim: usize,
        n_rot: usize,
        theta: f32,
    ) {
        let n_rot = n_rot.min(head_dim);
        let theta_scale: f32 = theta.powf(-2.0_f32 / n_rot as f32);
        let seq_len = data.len() / dim;
        for t in 0..seq_len {
            let p = pos + t;
            let token = &mut data[t * dim..(t + 1) * dim];
            for chunk in token.chunks_mut(head_dim) {
                let mut angle = p as f32;
                let mut i = 0usize;
                while i < n_rot {
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let x0 = chunk[i];
                    let x1 = chunk[i + 1];
                    chunk[i] = x0 * cos_a - x1 * sin_a;
                    chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                    angle *= theta_scale;
                    i += 2;
                }
            }
        }
    }

    /// pm17 gated attn: rope_partial(인접페어) 가 host `rope_partial_inplace`(9B
    /// production RoPE)와 rel<1e-3 인지. head_dim=256, n_rot=64, theta=1e7.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_rope_partial_matches_cpu() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            return;
        }
        let (hd, n_rot) = (256usize, 64usize);
        for num_heads in [16usize, 4] {
            let dim = hd * num_heads;
            for pos in [0usize, 5, 17] {
                let input: Vec<f32> = (0..dim)
                    .map(|i| ((i * 7 % 29) as f32) * 0.1 - 1.4)
                    .collect();
                let mut cpu = input.clone();
                cpu_rope_partial(&mut cpu, pos, hd, dim, n_rot, 10_000_000.0);
                let got =
                    backend.metal_rope_partial_into(&input, hd, dim, n_rot, 10_000_000.0, pos);
                for (a, b) in got.iter().zip(cpu.iter()) {
                    let rel = (a - b).abs() / (b.abs().max(1e-3));
                    assert!(
                        rel < 1e-3,
                        "rel={rel} got={a} cpu={b} pos={pos} nh={num_heads}"
                    );
                }
            }
        }
    }

    /// pm17 gated attn: gate_apply 가 host sigmoid·mul(decode.rs:751-757)와 rel<1e-5 인지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gate_apply_matches_cpu() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            return;
        }
        for n in [4096usize, 1024, 13] {
            let attn: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
            let gate: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.2 - 2.0).collect();
            let mut cpu = attn.clone();
            for i in 0..n {
                let s = 1.0f32 / (1.0 + (-gate[i]).exp());
                cpu[i] *= s;
            }
            let got = backend.metal_gate_apply_into(&attn, &gate);
            for (a, b) in got.iter().zip(cpu.iter()) {
                let rel = (a - b).abs() / (b.abs().max(1e-3));
                assert!(rel < 1e-5, "rel={rel} got={a} cpu={b} n={n}");
            }
        }
    }

    /// M2: text M-RoPE 커널이 CPU `rope_mrope_text_inplace` 와 token-identical 한지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_rope_mrope_matches_cpu_reference() {
        let fx = tests_fixture::rope_mrope_fixture();
        let gpu = MetalBackend::new().rope_mrope(
            &fx.data,
            fx.head_dim,
            fx.dim,
            fx.n_rot,
            fx.theta,
            fx.pos,
        );
        assert_eq!(gpu.len(), fx.reference.len());
        let mut max_rel = 0.0f32;
        for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1e-6);
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-3 || abs < 1e-4,
                "idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
            );
        }
        eprintln!("rope_mrope max_rel = {max_rel}");
    }

    /// pm12 carrier: per-head q/k norm 커널이 CPU `rms_norm_into` 와 token-identical 한지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_qk_norm_matches_cpu_reference() {
        let fx = tests_fixture::qk_norm_fixture();
        let gpu =
            MetalBackend::new().qk_norm(&fx.data, &fx.weight, fx.num_heads, fx.head_dim, fx.eps);
        assert_eq!(gpu.len(), fx.reference.len());
        let mut max_rel = 0.0f32;
        for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1e-6);
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-3 || abs < 1e-4,
                "idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
            );
        }
        eprintln!("qk_norm max_rel = {max_rel}");
    }

    /// pm12 carrier: kv_append 커널이 device KV 에 f16 으로 정확히 append 하는지
    /// (CPU `half::f16::from_f32` round-to-nearest-even 와 bit-identical).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_kv_append_matches_cpu_reference() {
        let kv_dim = 16usize;
        let capacity = 4usize;
        let n_steps = 3usize;
        let k_steps: Vec<Vec<f32>> = (0..n_steps)
            .map(|p| {
                (0..kv_dim)
                    .map(|d| (((p * kv_dim + d) % 7) as f32) * 0.25)
                    .collect()
            })
            .collect();
        let v_steps: Vec<Vec<f32>> = (0..n_steps)
            .map(|p| {
                (0..kv_dim)
                    .map(|d| (((p * 3 + d) % 5) as f32) * 0.5)
                    .collect()
            })
            .collect();
        // pm25: f16 kv_append 경로 검증 — f16 ctx 명시(carrier default ON 회피).
        let (kout, vout) = MetalBackend::new_with_kv_int8(false)
            .kv_append_steps(&k_steps, &v_steps, kv_dim, capacity);
        assert_eq!(kout.len(), n_steps * kv_dim);
        for p in 0..n_steps {
            for d in 0..kv_dim {
                let kref = half::f16::from_f32(k_steps[p][d]).to_bits();
                let vref = half::f16::from_f32(v_steps[p][d]).to_bits();
                assert_eq!(kout[p * kv_dim + d], kref, "k pos={p} d={d}");
                assert_eq!(vout[p * kv_dim + d], vref, "v pos={p} d={d}");
            }
        }
    }

    /// 여러 shape(num_heads·head_dim)에서 q/k norm 커널 정확도.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_qk_norm_matches_cpu_reference_various_shapes() {
        // (num_heads, head_dim): 9B q(40,128)/kv(8,128) 근사 + 작은/홀수 head_dim.
        for &(nh, hd) in &[(40usize, 128usize), (8, 128), (2, 64), (1, 96), (3, 192)] {
            let fx = tests_fixture::qk_norm_fixture_shaped(nh, hd);
            let gpu = MetalBackend::new().qk_norm(&fx.data, &fx.weight, nh, hd, fx.eps);
            assert_eq!(gpu.len(), fx.reference.len(), "shape ({nh},{hd})");
            for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "shape ({nh},{hd}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm15 GDN carrier: ssm_conv1d_silu 커널이 CPU `ssm_conv1d_silu_into`(seq_len=1)
    /// 와 token-identical 한지. depthwise causal conv + SiLU, channel 독립.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_ssm_conv1d_silu_matches_cpu_reference() {
        // (channels, kernel_size): 9B conv_channels 근사 + 작은/홀수 shape.
        for &(ch, ks) in &[(384usize, 4usize), (256, 4), (130, 3), (64, 2), (97, 4)] {
            let fx = tests_fixture::ssm_conv1d_silu_fixture_shaped(ch, ks);
            let gpu = MetalBackend::new().ssm_conv1d_silu(&fx.input, &fx.weight, ch, ks);
            assert_eq!(gpu.len(), fx.reference.len(), "shape ({ch},{ks})");
            for (i, (&g, &r)) in gpu.iter().zip(fx.reference.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "shape ({ch},{ks}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm43: prefill batch(seq_len>1) conv1d+silu 가 CPU `ssm_conv1d_silu_into`(seq_len) 와 일치.
    /// f32 GPU vs f32 CPU(둘 다 exact, exp 미세 차만) → max_rel<1e-4. 실모델급 채널(5120) 포함.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_prefill_conv1d_silu_batch_matches_cpu() {
        use rnb_cpu::kernels::conv::ssm_conv1d_silu_into;
        let backend = MetalBackend::new();
        for (channels, kernel_size, seq_len) in
            [(64usize, 4usize, 37usize), (256, 3, 100), (5120, 4, 16)]
        {
            let total = seq_len + kernel_size - 1;
            let input = det_vals(total * channels, 0.05);
            let weight = det_vals(kernel_size * channels, 0.1);
            let mut cpu = vec![0f32; seq_len * channels];
            ssm_conv1d_silu_into(&input, &weight, &mut cpu, seq_len, channels, kernel_size);
            let gpu = backend.prefill_conv1d_silu(&input, &weight, seq_len, channels, kernel_size);
            assert_eq!(gpu.len(), seq_len * channels);
            let mut max_rel = 0f32;
            for i in 0..seq_len * channels {
                let denom = cpu[i].abs().max(1e-4);
                max_rel = max_rel.max((gpu[i] - cpu[i]).abs() / denom);
            }
            eprintln!("[pm43] conv batch (ch={channels},ks={kernel_size},seq={seq_len}) max_rel={max_rel:.2e}");
            assert!(
                max_rel < 1e-4,
                "conv batch (ch={channels},ks={kernel_size},seq={seq_len}) mismatch: {max_rel}"
            );
        }
    }

    /// pm43: prefill gated RMSNorm+SiLU(batch) 가 CPU(per-row rms_norm·weight · silu(z)) 와 일치.
    /// f32 rsqrt/exp + GPU tree reduction 미세 차 → max_rel<1e-3. 실모델급 cols(5120) 포함.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_prefill_gated_norm_silu_matches_cpu() {
        let backend = MetalBackend::new();
        let eps = 1e-5f32;
        for (rows, cols) in [(37usize * 16, 128usize), (100, 256), (16, 5120)] {
            let out_in = det_vals(rows * cols, 0.05);
            let z = det_vals(rows * cols, 0.1);
            let weight = det_vals(cols, 0.5);
            // CPU ref: per-row rms_norm(out_in)*weight, then *silu(z).
            let mut cpu = vec![0f32; rows * cols];
            for r in 0..rows {
                let mut ss = 0.0f32;
                for i in 0..cols {
                    let v = out_in[r * cols + i];
                    ss += v * v;
                }
                let inv = 1.0f32 / (ss / cols as f32 + eps).sqrt();
                for i in 0..cols {
                    let normed = out_in[r * cols + i] * inv * weight[i];
                    let zz = z[r * cols + i];
                    cpu[r * cols + i] = normed * (zz / (1.0 + (-zz).exp()));
                }
            }
            let gpu = backend.prefill_gated_norm_silu(&out_in, &z, &weight, rows, cols, eps);
            assert_eq!(gpu.len(), rows * cols);
            let mut max_rel = 0f32;
            for i in 0..rows * cols {
                let denom = cpu[i].abs().max(1e-4);
                max_rel = max_rel.max((gpu[i] - cpu[i]).abs() / denom);
            }
            eprintln!("[pm43] gated_norm (rows={rows},cols={cols}) max_rel={max_rel:.2e}");
            assert!(
                max_rel < 1e-3,
                "gated_norm (rows={rows},cols={cols}) mismatch: {max_rel}"
            );
        }
    }

    /// pm44 M1-3: prefill_gated_norm_silu_project(fused) vs 분리 경로
    /// (gated_rmsnorm_silu_batch_with_ctx + prefill_gdn_proj) 와 수치 동일 검증.
    /// 같은 gated 커널 + 같은 Q5_K v2 GEMM 이므로 max_rel < 1e-5 (산술 동일).
    /// shape: seq=3, num_v_heads=2, head_v_dim=128 → d_inner=256, n_out=6.
    /// Q5_K weight: n_out rows × (d_inner/256) superblocks.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires M5 Metal device"]
    fn metal_prefill_gated_norm_silu_project_matches_separate() {
        let backend = MetalBackend::new();
        let ctx = match backend.ctx.as_ref() {
            Some(c) => c,
            None => {
                eprintln!("[pm44] no Metal ctx; skip");
                return;
            }
        };
        if !ctx.tensorops_capable {
            eprintln!("[pm44] not tensorops-capable; skip");
            return;
        }

        let seq = 3usize;
        let num_v_heads = 2usize;
        let head_v_dim = 128usize;
        let d_inner = num_v_heads * head_v_dim; // 256
        let n_out = 6usize;
        let eps = 1e-5f32;

        // 입력: det_vals 로 결정적 생성 (기존 테스트와 동일 패턴)
        let out_in = det_vals(seq * d_inner, 0.05);
        let z = det_vals(seq * d_inner, 0.1);
        let norm = det_vals(head_v_dim, 0.5);

        // Q5_K weight: n_out rows × 1 superblock(K=256, 176 bytes each)
        let w_q5k = build_q5k_rows(n_out, d_inner);

        // ── separate 경로 (oracle) ──────────────────────────────────────
        // 1) gated_rmsnorm_silu: rows = seq * num_v_heads, cols = head_v_dim
        let gated = compute::gated_rmsnorm_silu_batch_with_ctx(
            ctx,
            &out_in,
            &z,
            &norm,
            seq * num_v_heads,
            head_v_dim,
            eps,
        );
        // 2) ssm_out GEMM: input=[seq*d_inner], K=d_inner, N=n_out, M=seq
        let separate =
            backend.prefill_gdn_proj(&gated, &w_q5k, TensoropsQuant::Q5K, seq, d_inner, n_out);

        // ── fused 경로 ──────────────────────────────────────────────────
        let fused = backend.prefill_gated_norm_silu_project(
            &out_in,
            &z,
            &norm,
            &w_q5k,
            TensoropsQuant::Q5K,
            seq,
            head_v_dim,
            n_out,
            eps,
        );

        assert_eq!(fused.len(), separate.len(), "length mismatch");

        let mut max_rel = 0f32;
        for i in 0..fused.len() {
            let denom = separate[i].abs().max(1e-6);
            max_rel = max_rel.max((fused[i] - separate[i]).abs() / denom);
        }
        eprintln!("[pm44] fused vs separate max_rel={max_rel:.2e}");
        assert!(max_rel < 1e-5, "fused vs separate mismatch: {max_rel}");
    }

    /// pm45 M2: prefill_gdn_conv_delta_chain(fused) vs 기존 backend method 순차 호출 oracle
    /// (prefill_conv1d_silu → CPU split → l2_norm(q,k) → CPU repeat → delta_net_scan_chunk).
    /// 같은 GPU 커널들을 같은 순서로 타므로 max_rel < 1e-4 (산술 동일, GPU 재배치만 device 잔류).
    /// 작은 shape: seq=4, num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, chunk_size=4.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_prefill_gdn_conv_delta_chain_matches_separate() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            eprintln!("[pm45] no Metal ctx; skip");
            return;
        }
        let seq = 4usize;
        let num_k_heads = 2usize;
        let num_v_heads = 4usize; // GQA factor 2
        let head_k_dim = 8usize;
        let head_v_dim = 8usize;
        let chunk_size = 4usize;
        let conv_kernel = 4usize;
        let eps = 1e-5f32;

        let q_dim = head_k_dim * num_k_heads;
        let k_dim = q_dim;
        let v_dim = head_v_dim * num_v_heads;
        // conv_channels = q+k+v + 잔여(gate 등 skip 경로 검증).
        let conv_channels = q_dim + k_dim + v_dim + 4;
        let total_conv_len = (conv_kernel - 1) + seq;

        let conv_input = det_vals(total_conv_len * conv_channels, 0.013);
        let conv_weight = det_vals(conv_kernel * conv_channels, 0.07);
        let gate = det_vals(seq * num_v_heads, 0.05);
        let beta = det_vals(seq * num_v_heads, 0.1);
        let state = det_vals(num_v_heads * head_v_dim * head_k_dim, 0.02);

        // ── separate oracle: 기존 backend method 순차 호출 ──────────────────
        // 1) conv1d_silu(batch) GPU.
        let conv_data =
            backend.prefill_conv1d_silu(&conv_input, &conv_weight, seq, conv_channels, conv_kernel);
        // 2) split_conv_qkv (CPU oracle, host `gdn_prefill.rs split_conv_qkv` 1:1).
        let mut q_data = vec![0f32; seq * q_dim];
        let mut k_data = vec![0f32; seq * k_dim];
        let mut v_data = vec![0f32; seq * v_dim];
        for t in 0..seq {
            let off = t * conv_channels;
            q_data[t * q_dim..(t + 1) * q_dim].copy_from_slice(&conv_data[off..off + q_dim]);
            k_data[t * k_dim..(t + 1) * k_dim]
                .copy_from_slice(&conv_data[off + q_dim..off + q_dim + k_dim]);
            v_data[t * v_dim..(t + 1) * v_dim]
                .copy_from_slice(&conv_data[off + q_dim + k_dim..off + q_dim + k_dim + v_dim]);
        }
        // 3) l2_norm: rows = seq*num_k_heads, dim = head_k_dim. q scale=1/√hk, k scale=1.0.
        let norm_rows = seq * num_k_heads;
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let q_normed = backend.l2_norm(&q_data, norm_rows, head_k_dim, eps, q_scale);
        let k_normed = backend.l2_norm(&k_data, norm_rows, head_k_dim, eps, 1.0);
        // 4) repeat_qk (CPU oracle, host `gdn_prefill.rs repeat_qk_for_value_heads` 1:1).
        let mut q_final = Vec::with_capacity(seq * num_v_heads * head_k_dim);
        let mut k_final = Vec::with_capacity(seq * num_v_heads * head_k_dim);
        for t in 0..seq {
            for vh in 0..num_v_heads {
                let kh = vh % num_k_heads;
                let src = (t * num_k_heads + kh) * head_k_dim;
                q_final.extend_from_slice(&q_normed[src..src + head_k_dim]);
                k_final.extend_from_slice(&k_normed[src..src + head_k_dim]);
            }
        }
        // 5) delta_net_scan_chunk: num_heads = num_v_heads(repeat 후).
        let (sep_out, sep_state) = backend.delta_net_scan_chunk(
            &q_final,
            &k_final,
            &v_data,
            &gate,
            &beta,
            &state,
            seq,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
        );

        // ── fused 경로 ──────────────────────────────────────────────────
        let (fused_out, fused_state) = backend.prefill_gdn_conv_delta_chain(
            &conv_input,
            &conv_weight,
            &gate,
            &beta,
            &state,
            seq,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            eps,
        );

        assert_eq!(fused_out.len(), sep_out.len(), "out length mismatch");
        assert_eq!(fused_state.len(), sep_state.len(), "state length mismatch");

        let mut max_rel = 0f32;
        for i in 0..fused_out.len() {
            let denom = sep_out[i].abs().max(1e-6);
            max_rel = max_rel.max((fused_out[i] - sep_out[i]).abs() / denom);
        }
        for i in 0..fused_state.len() {
            let denom = sep_state[i].abs().max(1e-6);
            max_rel = max_rel.max((fused_state[i] - sep_state[i]).abs() / denom);
        }
        eprintln!("[pm45] conv_delta fused vs separate max_rel={max_rel:.2e}");
        assert!(
            max_rel < 1e-4,
            "conv_delta fused vs separate mismatch: {max_rel}"
        );
    }

    /// pm45 M3-1: prefill_gdn_full_chain(conv→delta→gated→ssm_out 통합) vs 분리 경로
    /// (prefill_gdn_conv_delta_chain → 그 output 을 prefill_gated_norm_silu_project) 순차 비교.
    /// 같은 GPU 커널을 같은 순서로 타고 delta output 만 device 잔류라 max_rel < 1e-4.
    /// 작은 shape: seq=4, num_k_heads=2, num_v_heads=4, head_k=8, head_v=8, chunk=4, n_out=6.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_prefill_gdn_full_chain_matches_separate() {
        let backend = MetalBackend::new();
        let ctx = match backend.ctx.as_ref() {
            Some(c) => c,
            None => {
                eprintln!("[pm45 M3-1] no Metal ctx; skip");
                return;
            }
        };
        if !ctx.tensorops_capable {
            eprintln!("[pm45 M3-1] not tensorops-capable; skip");
            return;
        }
        let seq = 4usize;
        let num_k_heads = 2usize;
        let num_v_heads = 4usize; // GQA factor 2
        let head_k_dim = 8usize;
        // d_inner = num_v_heads*head_v_dim 가 Q5_K superblock(256) 배수여야 ssm_out weight 생성 가능.
        let head_v_dim = 64usize;
        let chunk_size = 4usize;
        let conv_kernel = 4usize;
        let n_out = 6usize;
        let eps = 1e-5f32;

        let q_dim = head_k_dim * num_k_heads;
        let k_dim = q_dim;
        let v_dim = head_v_dim * num_v_heads;
        let conv_channels = q_dim + k_dim + v_dim + 4;
        let total_conv_len = (conv_kernel - 1) + seq;
        let d_inner = num_v_heads * head_v_dim; // = v_dim

        let conv_input = det_vals(total_conv_len * conv_channels, 0.013);
        let conv_weight = det_vals(conv_kernel * conv_channels, 0.07);
        let gate = det_vals(seq * num_v_heads, 0.05);
        let beta = det_vals(seq * num_v_heads, 0.1);
        let state = det_vals(num_v_heads * head_v_dim * head_k_dim, 0.02);
        let z = det_vals(seq * d_inner, 0.1);
        let ssm_norm = det_vals(head_v_dim, 0.5);
        // ssm_out Q5_K weight: n_out rows × (d_inner/256 = 1) superblock.
        let w_q5k = build_q5k_rows(n_out, d_inner);

        // ── separate 경로 (oracle): conv_delta chain → gated_norm_silu_project ──
        let (delta_out, _state_sep) = backend.prefill_gdn_conv_delta_chain(
            &conv_input,
            &conv_weight,
            &gate,
            &beta,
            &state,
            seq,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            eps,
        );
        let sep_proj = backend.prefill_gated_norm_silu_project(
            &delta_out,
            &z,
            &ssm_norm,
            &w_q5k,
            TensoropsQuant::Q5K,
            seq,
            head_v_dim,
            n_out,
            eps,
        );

        // ── fused 경로: full chain ──────────────────────────────────────
        let (full_proj, _full_state) = backend.prefill_gdn_full_chain(
            &conv_input,
            &conv_weight,
            &gate,
            &beta,
            &state,
            &z,
            &ssm_norm,
            &w_q5k,
            TensoropsQuant::Q5K,
            seq,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            n_out,
            eps,
        );

        assert_eq!(full_proj.len(), sep_proj.len(), "proj length mismatch");
        let mut max_rel = 0f32;
        for i in 0..full_proj.len() {
            let denom = sep_proj[i].abs().max(1e-6);
            max_rel = max_rel.max((full_proj[i] - sep_proj[i]).abs() / denom);
        }
        eprintln!("[pm45 M3-1] full chain vs separate max_rel={max_rel:.2e}");
        assert!(max_rel < 1e-4, "full chain vs separate mismatch: {max_rel}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_prefill_gdn_full_ffn_chain_matches_existing_path() {
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            eprintln!("[pm49] no Metal ctx; skip");
            return;
        };
        if !ctx.tensorops_capable
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.gemm_q4k_tensorops_v2_pipeline.is_none()
            || ctx.gemm_q6k_tensorops_v2_pipeline.is_none()
        {
            eprintln!("[pm49] tensorops v2 unavailable; skip");
            return;
        }

        let seq = 5usize;
        let hidden_dim = 256usize;
        let ffn_dim = 512usize;
        let num_k_heads = 1usize;
        let num_v_heads = 1usize;
        let head_k_dim = 8usize;
        let head_v_dim = 256usize;
        let chunk_size = 4usize;
        let conv_kernel = 4usize;
        let conv_channels = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim + 4;
        let total_conv_len = seq + conv_kernel - 1;
        let eps = 1e-5f32;

        let hidden = det_vals(seq * hidden_dim, 0.011);
        let conv_input = det_vals(total_conv_len * conv_channels, 0.013);
        let conv_weight = det_vals(conv_kernel * conv_channels, 0.07);
        let gate = det_vals(seq * num_v_heads, 0.05);
        let beta = det_vals(seq * num_v_heads, 0.1);
        let state = det_vals(num_v_heads * head_v_dim * head_k_dim, 0.02);
        let z = det_vals(seq * num_v_heads * head_v_dim, 0.1);
        let ssm_norm = det_vals(head_v_dim, 0.5);
        let post_norm = det_vals(hidden_dim, 0.23);

        let ssm_out_q4 = quantize_rows_q4k(
            &det_vals(hidden_dim * num_v_heads * head_v_dim, 0.019),
            hidden_dim,
            num_v_heads * head_v_dim,
        );
        let gate_q4 =
            quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.021), ffn_dim, hidden_dim);
        let up_q4 = quantize_rows_q4k(&det_vals(ffn_dim * hidden_dim, 0.017), ffn_dim, hidden_dim);
        let down_q4 =
            quantize_rows_q4k(&det_vals(hidden_dim * ffn_dim, 0.013), hidden_dim, ffn_dim);

        let (proj, state_expected) = backend.prefill_gdn_full_chain(
            &conv_input,
            &conv_weight,
            &gate,
            &beta,
            &state,
            &z,
            &ssm_norm,
            &ssm_out_q4,
            TensoropsQuant::Q4K,
            seq,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            hidden_dim,
            eps,
        );
        let mut hidden_plus = hidden.clone();
        for i in 0..hidden_plus.len() {
            hidden_plus[i] += proj[i];
        }
        let mut normed = vec![0.0f32; hidden_plus.len()];
        for r in 0..seq {
            let base = r * hidden_dim;
            let sum = hidden_plus[base..base + hidden_dim]
                .iter()
                .map(|v| v * v)
                .sum::<f32>();
            let inv = 1.0f32 / (sum / hidden_dim as f32 + eps).sqrt();
            for c in 0..hidden_dim {
                normed[base + c] = hidden_plus[base + c] * inv * post_norm[c];
            }
        }
        let ffn_down = backend.prefill_ffn_chain(
            &normed, &gate_q4, &up_q4, &down_q4, false, seq, hidden_dim, ffn_dim,
        );
        let mut expected = hidden_plus;
        for i in 0..expected.len() {
            expected[i] += ffn_down[i];
        }

        let (got, state_got) = backend
            .prefill_gdn_full_ffn_chain_if_supported(
                &hidden,
                &conv_input,
                &conv_weight,
                &gate,
                &beta,
                &state,
                &z,
                &ssm_norm,
                &ssm_out_q4,
                TensoropsQuant::Q4K,
                &post_norm,
                &gate_q4,
                &up_q4,
                &down_q4,
                false,
                seq,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                chunk_size,
                hidden_dim,
                ffn_dim,
                eps,
            )
            .expect("full+ffn chain should be supported in this fixture");

        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let state_abs = state_got
            .iter()
            .zip(state_expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[pm49] full_ffn max_abs={max_abs:.3e} state_abs={state_abs:.3e}");
        assert!(max_abs < 3e-2, "hidden mismatch: {max_abs}");
        assert!(state_abs < 2e-5, "state mismatch: {state_abs}");
    }

    /// pm15 GDN carrier: delta_net_step 커널이 CPU `delta_net_scan_into`(seq_len=1)
    /// 와 token-identical 한지. output + 갱신된 state 둘 다 검증.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_delta_net_step_matches_cpu_reference() {
        // (num_v_heads, num_k_heads, head_k_dim, head_v_dim): non-GQA(nk=nv) + GQA(nk<nv).
        // head_k_dim 은 4 배수(NEON ref 분기 조건).
        for &(nv, nk, hk, hv) in &[
            (4usize, 4usize, 128usize, 128usize),
            (8, 8, 64, 64),
            (2, 2, 128, 64),
            (6, 6, 32, 96),
            (4, 2, 64, 64), // GQA factor 2
            (8, 2, 64, 64), // GQA factor 4
        ] {
            let fx = tests_fixture::delta_net_step_fixture_shaped(nv, nk, hk, hv);
            let (gpu_out, gpu_state) = MetalBackend::new().delta_net_step(
                &fx.q,
                &fx.k,
                &fx.v,
                &fx.gate,
                &fx.beta,
                &fx.state_in,
                nv,
                nk,
                hk,
                hv,
            );
            assert_eq!(
                gpu_out.len(),
                fx.ref_out.len(),
                "out len ({nv},{nk},{hk},{hv})"
            );
            assert_eq!(
                gpu_state.len(),
                fx.ref_state.len(),
                "state len ({nv},{nk},{hk},{hv})"
            );
            for (i, (&g, &r)) in gpu_out.iter().zip(fx.ref_out.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "out ({nv},{nk},{hk},{hv}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
            for (i, (&g, &r)) in gpu_state.iter().zip(fx.ref_state.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "state ({nv},{nk},{hk},{hv}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm39 M2: delta_net_scan_chunk(chunkwise prefill, seq_len>1) 커널이 CPU
    /// `delta_net_scan_chunkwise`(M1 oracle, 같은 chunk_size) 와 token-identical 한지.
    /// output + hand-off 된 state 둘 다 검증. remainder chunk / chunk_size=1 포함.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_delta_net_scan_chunk_matches_cpu_reference() {
        // (seq_len, num_heads, head_k_dim, head_v_dim, chunk_size)
        for &(sl, nh, hk, hv, cs) in &[
            (4usize, 2usize, 8usize, 8usize, 2usize),
            (12, 2, 16, 16, 4),
            (7, 3, 8, 12, 3), // remainder (7 = 3+3+1)
            (16, 2, 32, 32, 8),
            (37, 3, 16, 8, 16), // remainder (37 = 16+16+5)
            (5, 4, 4, 4, 1),    // chunk_size=1 == per-token recurrence
        ] {
            let fx = tests_fixture::delta_net_scan_chunk_fixture_shaped(sl, nh, hk, hv, cs);
            let (gpu_out, gpu_state) = MetalBackend::new().delta_net_scan_chunk(
                &fx.q,
                &fx.k,
                &fx.v,
                &fx.gate,
                &fx.beta,
                &fx.state_in,
                sl,
                nh,
                hk,
                hv,
                cs,
            );
            assert_eq!(
                gpu_out.len(),
                fx.ref_out.len(),
                "out len ({sl},{nh},{hk},{hv},{cs})"
            );
            assert_eq!(
                gpu_state.len(),
                fx.ref_state.len(),
                "state len ({sl},{nh},{hk},{hv},{cs})"
            );
            for (i, (&g, &r)) in gpu_out.iter().zip(fx.ref_out.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "out ({sl},{nh},{hk},{hv},{cs}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
            for (i, (&g, &r)) in gpu_state.iter().zip(fx.ref_state.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "state ({sl},{nh},{hk},{hv},{cs}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm58: llama.cpp Qwen3Next GDN path 와 같은 row-parallel autoregressive prefill scan.
    /// chunkwise oracle 이 아니라 CPU sequential recurrence(`delta_net_scan`)와 직접 비교한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_delta_net_scan_ar_matches_sequential_reference() {
        use rnb_cpu::kernels::delta_net::delta_net_scan;

        for &(sl, nh, hk, hv, cs) in &[
            (7usize, 2usize, 128usize, 128usize, 4usize),
            (9usize, 3usize, 64usize, 96usize, 3usize),
            (5usize, 4usize, 32usize, 64usize, 2usize),
        ] {
            let fx = tests_fixture::delta_net_scan_chunk_fixture_shaped(sl, nh, hk, hv, cs);
            let mut ref_state = fx.state_in.clone();
            let ref_out = delta_net_scan(
                &fx.q,
                &fx.k,
                &fx.v,
                &fx.gate,
                &fx.beta,
                &mut ref_state,
                sl,
                nh,
                hk,
                hv,
            );
            let (gpu_out, gpu_state) = MetalBackend::new().delta_net_scan_ar(
                &fx.q,
                &fx.k,
                &fx.v,
                &fx.gate,
                &fx.beta,
                &fx.state_in,
                sl,
                nh,
                hk,
                hv,
            );
            assert_eq!(
                gpu_out.len(),
                ref_out.len(),
                "out len ({sl},{nh},{hk},{hv})"
            );
            assert_eq!(
                gpu_state.len(),
                ref_state.len(),
                "state len ({sl},{nh},{hk},{hv})"
            );
            for (i, (&g, &r)) in gpu_out.iter().zip(ref_out.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "out ({sl},{nh},{hk},{hv}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
            for (i, (&g, &r)) in gpu_state.iter().zip(ref_state.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "state ({sl},{nh},{hk},{hv}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm39 M2: delta_net_scan_chunk GPU 커널 per-call 측정(standalone). production CPU
    /// `delta_net_scan`(rayon+NEON) 과 chunkwise GPU per-call wall 비교 — REFACTOR 필요성 1차 판정.
    /// standalone 은 buffer 생성/readback 왕복 포함(seam device-resident 보다 보수적).
    /// RNB_PM39_BENCH_SHAPE="sl,nh,hk,hv,cs" 로 shape override(default 대표 27B GDN).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "perf micro-bench, requires a Metal device"]
    fn bench_delta_net_scan_chunk_vs_cpu() {
        use rnb_cpu::kernels::delta_net::{delta_net_scan, delta_net_scan_chunkwise};
        use std::time::Instant;

        let shape = std::env::var("RNB_PM39_BENCH_SHAPE").ok();
        let (sl, nh, hk, hv, cs) = shape
            .as_deref()
            .and_then(|s| {
                let p: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (p.len() == 5).then(|| (p[0], p[1], p[2], p[3], p[4]))
            })
            .unwrap_or((1139, 32, 128, 128, 32)); // 대표 27B GDN prefill

        let fx = tests_fixture::delta_net_scan_chunk_fixture_shaped(sl, nh, hk, hv, cs);
        let be = MetalBackend::new();
        let iters = 10;

        // GPU warmup + 측정.
        let _ = be.delta_net_scan_chunk(
            &fx.q,
            &fx.k,
            &fx.v,
            &fx.gate,
            &fx.beta,
            &fx.state_in,
            sl,
            nh,
            hk,
            hv,
            cs,
        );
        let t = Instant::now();
        for _ in 0..iters {
            let _ = be.delta_net_scan_chunk(
                &fx.q,
                &fx.k,
                &fx.v,
                &fx.gate,
                &fx.beta,
                &fx.state_in,
                sl,
                nh,
                hk,
                hv,
                cs,
            );
        }
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // production CPU(delta_net_scan: rayon head 병렬 + NEON) per-call.
        let t = Instant::now();
        for _ in 0..iters {
            let mut s = fx.state_in.clone();
            let _ = delta_net_scan(
                &fx.q, &fx.k, &fx.v, &fx.gate, &fx.beta, &mut s, sl, nh, hk, hv,
            );
        }
        let cpu_prod_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // CPU chunkwise oracle(single-thread reference) per-call — 같은 알고리즘 비교용.
        let t = Instant::now();
        for _ in 0..iters {
            let mut s = fx.state_in.clone();
            let _ = delta_net_scan_chunkwise(
                &fx.q, &fx.k, &fx.v, &fx.gate, &fx.beta, &mut s, sl, nh, hk, hv, cs,
            );
        }
        let cpu_chunk_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!(
            "[pm39 bench] sl={sl} nh={nh} hk={hk} hv={hv} cs={cs} | GPU {gpu_ms:.2}ms/call | \
             CPU-prod(rayon+neon) {cpu_prod_ms:.2}ms/call | CPU-chunkwise-ref {cpu_chunk_ms:.2}ms/call | \
             GPU vs prod = {:.2}x",
            cpu_prod_ms / gpu_ms
        );
    }

    /// pm16 GDN carrier: l2_norm 커널이 CPU `l2_norm_into` 와 token-identical 한지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_l2_norm_matches_cpu_reference() {
        use rnb_cpu::kernels::norm::l2_norm_into;
        let eps = 1.0e-6f32;
        // (n_rows, dim): 9B GDN q/k head 근사 + 작은/홀수.
        for &(nr, d) in &[(4usize, 128usize), (8, 64), (2, 96), (1, 256), (3, 50)] {
            let input: Vec<f32> = (0..nr * d)
                .map(|i| (((i % 13) as f32) - 6.0) * 0.05)
                .collect();
            let mut reference = vec![0.0f32; nr * d];
            l2_norm_into(&input, eps, &mut reference, d);
            let gpu = MetalBackend::new().l2_norm(&input, nr, d, eps, 1.0);
            assert_eq!(gpu.len(), reference.len(), "shape ({nr},{d})");
            for (i, (&g, &r)) in gpu.iter().zip(reference.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "shape ({nr},{d}) idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
        }
    }

    /// pm16 GDN carrier: gdn_alpha_beta 커널이 CPU 전처리(beta sigmoid + alpha-gate)와
    /// token-identical 한지. `beta=sigmoid(beta)`, `alpha=softplus(alpha+dt_bias)*ssm_a`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gdn_alpha_beta_matches_cpu_reference() {
        let residual_inputs = [
            (
                0xbf99_3991u32,
                0xbf0b_d805u32,
                0xc08e_0000u32,
                0xbd10_9b43u32,
            ),
            (
                0xc015_a0ceu32,
                0xbef9_5bebu32,
                0xc086_0000u32,
                0xbd39_adacu32,
            ),
            (
                0xc028_c000u32,
                0xbfda_f119u32,
                0xc081_0000u32,
                0xbd15_3261u32,
            ),
            (
                0x3f23_a88du32,
                0xbe0e_2e59u32,
                0xc091_0000u32,
                0xbda8_1c2eu32,
            ),
            (
                0xc07f_faedu32,
                0xbfcf_2d6eu32,
                0xc081_0000u32,
                0xbd15_3261u32,
            ),
            (
                0xc03d_f401u32,
                0xbe2f_e4edu32,
                0xc08a_0000u32,
                0xbcb7_d5bfu32,
            ),
            (
                0xc02d_8b97u32,
                0xbf79_1ae4u32,
                0xc086_0000u32,
                0xbd39_adacu32,
            ),
        ];
        let mut alpha: Vec<f32> = residual_inputs
            .iter()
            .map(|input| f32::from_bits(input.0))
            .collect();
        let mut beta: Vec<f32> = residual_inputs
            .iter()
            .map(|input| f32::from_bits(input.1))
            .collect();
        let mut dt_bias: Vec<f32> = residual_inputs
            .iter()
            .map(|input| f32::from_bits(input.2))
            .collect();
        let mut ssm_a: Vec<f32> = residual_inputs
            .iter()
            .map(|input| f32::from_bits(input.3))
            .collect();
        for index in 0..4096usize {
            let alpha_integer = ((index * 40_503) % 32_768) as i32 - 16_384;
            let beta_integer = ((index * 17_389) % 16_384) as i32 - 8_192;
            let bias_integer = ((index * 13) % 65) as i32 - 32;
            alpha.push(alpha_integer as f32 / 1024.0);
            beta.push(beta_integer as f32 / 1024.0);
            dt_bias.push(bias_integer as f32 / 64.0);
            ssm_a.push(-((index % 31 + 1) as f32) / 256.0);
        }

        let mut expected_beta = beta.clone();
        for value in &mut expected_beta {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        let expected_alpha: Vec<f32> = alpha
            .iter()
            .zip(&dt_bias)
            .zip(&ssm_a)
            .map(|((&raw, &bias), &scale)| (1.0 + (raw + bias).exp()).ln() * scale)
            .collect();
        let (gpu_alpha, gpu_beta) =
            MetalBackend::new().gdn_alpha_beta(&alpha, &beta, &dt_bias, &ssm_a, alpha.len());
        for index in 0..alpha.len() {
            assert_eq!(
                gpu_alpha[index].to_bits(),
                expected_alpha[index].to_bits(),
                "alpha idx={index} raw={:08x} dt={:08x} ssm_a={:08x}",
                alpha[index].to_bits(),
                dt_bias[index].to_bits(),
                ssm_a[index].to_bits()
            );
            assert_eq!(
                gpu_beta[index].to_bits(),
                expected_beta[index].to_bits(),
                "beta idx={index} raw={:08x}",
                beta[index].to_bits()
            );
        }
    }

    /// pm16: Q8_0 GEMV 커널이 CPU `dequantize_q8_0` 와 token-identical 한지(9B GDN alpha/beta).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q8_0_matches_cpu_reference() {
        use rnb_cpu::quantize::{dequantize_q8_0, BlockQ8_0};
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        for i in 0..32 {
            bytes[2 + i] = (((i as i32) - 16) as i8) as u8; // -16..15
        }
        #[repr(C, align(2))]
        struct A([u8; 34]);
        let mut a = A([0; 34]);
        a.0.copy_from_slice(&bytes);
        let blk: &BlockQ8_0 = unsafe { &*(a.0.as_ptr() as *const BlockQ8_0) };
        let mut out = [0.0f32; 32];
        dequantize_q8_0(blk, &mut out);
        let input = vec![1.0f32; 32];
        let ref_sum: f32 = out.iter().zip(&input).map(|(o, i)| o * i).sum();
        let gpu = MetalBackend::new().gemv_q8_0(&bytes, &input, 1, 32);
        let rel = (gpu[0] - ref_sum).abs() / ref_sum.abs().max(1e-6);
        assert!(rel < 1e-3, "q8_0 gpu={} ref={ref_sum} rel={rel}", gpu[0]);
    }

    /// pm16: Q5_K GEMV 커널이 CPU `dequantize_q5_k` 와 token-identical 한지(9B GDN qkv/ssm_out).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q5k_matches_cpu_reference() {
        use rnb_cpu::quantize::{dequantize_q5_k, BlockQ5_K};
        let mut bytes = vec![0u8; 176];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        bytes[2] = 0x00;
        bytes[3] = 0x38; // dmin = 0.5
        for i in 0..4 {
            bytes[4 + i] = 10;
            bytes[8 + i] = 5;
        }
        for i in 0..4 {
            bytes[12 + i] = 0x48;
        }
        for i in 0..32 {
            bytes[16 + i] = 0x55; // qh
        }
        for i in 0..128 {
            bytes[48 + i] = 0x35; // qs
        }
        #[repr(C, align(2))]
        struct A([u8; 176]);
        let mut a = A([0; 176]);
        a.0.copy_from_slice(&bytes);
        let blk: &BlockQ5_K = unsafe { &*(a.0.as_ptr() as *const BlockQ5_K) };
        let mut out = [0.0f32; 256];
        dequantize_q5_k(blk, &mut out);
        let input = vec![1.0f32; 256];
        let ref_sum: f32 = out.iter().sum();
        let gpu = MetalBackend::new().gemv_q5k(&bytes, &input, 1, 256);
        let rel = (gpu[0] - ref_sum).abs() / ref_sum.abs().max(1e-6);
        assert!(rel < 1e-3, "q5k gpu={} ref={ref_sum} rel={rel}", gpu[0]);
    }

    /// SIMD-group 협력 커널이 CPU reference 와 일치하는지(단일 블록).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q4k_simd_matches_cpu_reference() {
        let block = tests_fixture::q4k_block_fixed();
        let input = vec![1.0f32; 256];
        let reference = tests_fixture::q4k_dequant_sum(&block);
        let gpu = MetalBackend::new().gemv_q4k_simd(&block, &input, 1, 256);
        assert_eq!(gpu.len(), 1);
        let rel = (gpu[0] - reference).abs() / reference.abs().max(1e-6);
        assert!(rel < 1e-3, "gpu={} ref={} rel={rel}", gpu[0], reference);
    }

    /// llama.cpp coalesced 이식 커널이 CPU reference(`dequantize_q4_k`)와 일치하는지.
    /// **27B 실제 GEMV shape 3개 전부**: K=5120(nb20)·K=6144(nb24)·K=17408(nb68) —
    /// 우리 q4k_simd fallback(stride-32) 이 타는 non-pow2/큰 num_blocks 경로. 블록마다
    /// 다른 byte 패턴 + 변동 input → row/block offset 인덱싱·dequant·input 가중 전부 검증.
    /// reference 는 블록별 dequant(256) ⋅ input(256) 내적 합산(fp32). rel<1e-3 게이트.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q4k_coalesced_matches_cpu_reference() {
        let backend = MetalBackend::new();
        // 27B 실제 shape: hidden=5120, GDN/attn in-dim=6144, ffn in-dim=17408.
        // nr0=2 multi-row: 1 threadgroup 이 row 2개 처리. 홀수 N(7/13) 은 마지막 group 의
        // row1 이 weight OOB 라 has_row1 가드로 skip — 마지막 row 정확성 검증 필수.
        for &(n, k) in &[
            (8usize, 5120usize),
            (7usize, 5120usize),
            (16usize, 6144usize),
            (13usize, 6144usize),
            (8usize, 17408usize),
            (7usize, 17408usize),
        ] {
            let num_blocks = k / 256;
            // 블록마다(그리고 row 마다) 다른 byte 패턴 — 같은 값 반복이면 row/block 인덱싱
            // 버그를 못 잡으므로 결정적 비반복 패턴으로 채운다. 단 d/dmin(byte 0-3)은
            // 임의 byte 면 f16 exponent 0x1f(NaN/Inf)에 빠질 수 있어 유효 f16 으로 명시.
            let total_bytes = n * num_blocks * 144;
            let mut weight: Vec<u8> = (0..total_bytes)
                .map(|i| ((i as u64 * 1103515245 + 12345) >> 7 & 0xFF) as u8)
                .collect();
            let nblocks_total = n * num_blocks;
            for bi in 0..nblocks_total {
                let base = bi * 144;
                // d, dmin 을 블록마다 살짝 다른 유효 f16 (~0.5..1.0, 0.1..0.35)로.
                let d = half::f16::from_f32(0.5 + ((bi % 6) as f32) * 0.1);
                let dmin = half::f16::from_f32(0.1 + ((bi % 5) as f32) * 0.05);
                let db = d.to_bits().to_le_bytes();
                let mb = dmin.to_bits().to_le_bytes();
                weight[base] = db[0];
                weight[base + 1] = db[1];
                weight[base + 2] = mb[0];
                weight[base + 3] = mb[1];
            }
            let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

            // CPU reference: 각 row 의 num_blocks 개 블록을 dequant 한 뒤 input 슬라이스와 내적.
            let mut reference = vec![0.0f32; n];
            for r in 0..n {
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let off = (r * num_blocks + b) * 144;
                    let blk = &weight[off..off + 144];
                    let deq = tests_fixture::q4k_dequant(blk);
                    let x_base = b * 256;
                    for l in 0..256 {
                        acc += deq[l] * input[x_base + l];
                    }
                }
                reference[r] = acc;
            }

            let gpu = backend.gemv_q4k_coalesced(&weight, &input, n, k);
            assert_eq!(gpu.len(), n);
            let gpu_nsg2 = backend.gemv_q4k_coalesced_nsg2(&weight, &input, n, k);
            assert_eq!(gpu_nsg2.len(), n);
            for r in 0..n {
                let rel = (gpu[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                let rel_nsg2 = (gpu_nsg2[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel}",
                    gpu[r],
                    reference[r]
                );
                assert!(
                    rel_nsg2 < 1e-3,
                    "nsg2 n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel_nsg2}",
                    gpu_nsg2[r],
                    reference[r]
                );
            }
        }
    }

    /// llama.cpp q5_K coalesced 이식 커널이 CPU reference(`dequantize_q5_k`)와 일치하는지.
    /// 27B 실제 GEMV shape(K=5120/6144/17408) × 짝/홀 N(7/13 → 마지막 group row1 OOB
    /// has_row1 가드). 블록·row 마다 다른 byte 패턴 + 변동 input. d/dmin(byte 0-3)만
    /// 유효 f16 으로 명시(나머지 random byte 면 NaN f16 위험). rel<1e-3 게이트.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q5k_coalesced_matches_cpu_reference() {
        let backend = MetalBackend::new();
        for &(n, k) in &[
            (8usize, 5120usize),
            (7usize, 5120usize),
            (16usize, 6144usize),
            (13usize, 6144usize),
            (8usize, 17408usize),
            (7usize, 17408usize),
        ] {
            let num_blocks = k / 256;
            const BB: usize = 176;
            let total_bytes = n * num_blocks * BB;
            let mut weight: Vec<u8> = (0..total_bytes)
                .map(|i| ((i as u64 * 1103515245 + 12345) >> 7 & 0xFF) as u8)
                .collect();
            let nblocks_total = n * num_blocks;
            for bi in 0..nblocks_total {
                let base = bi * BB;
                // d(0-1), dmin(2-3) 만 유효 f16 (~0.5..1.0, 0.1..0.35).
                let d = half::f16::from_f32(0.5 + ((bi % 6) as f32) * 0.1);
                let dmin = half::f16::from_f32(0.1 + ((bi % 5) as f32) * 0.05);
                let db = d.to_bits().to_le_bytes();
                let mb = dmin.to_bits().to_le_bytes();
                weight[base] = db[0];
                weight[base + 1] = db[1];
                weight[base + 2] = mb[0];
                weight[base + 3] = mb[1];
            }
            let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

            let mut reference = vec![0.0f32; n];
            for r in 0..n {
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let off = (r * num_blocks + b) * BB;
                    let blk = &weight[off..off + BB];
                    let deq = tests_fixture::q5k_dequant(blk);
                    let x_base = b * 256;
                    for l in 0..256 {
                        acc += deq[l] * input[x_base + l];
                    }
                }
                reference[r] = acc;
            }

            let gpu = backend.gemv_q5k_coalesced(&weight, &input, n, k);
            assert_eq!(gpu.len(), n);
            for r in 0..n {
                let rel = (gpu[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "q5k n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel}",
                    gpu[r],
                    reference[r]
                );
            }
        }
    }

    /// llama.cpp q6_K coalesced 이식 커널이 CPU reference(`dequantize_q6_k`)와 일치하는지.
    /// 27B 실제 GEMV shape × 짝/홀 N. d(byte 208-209)만 유효 f16, scales(192-207)은 i8 라
    /// random byte 도 유효. rel<1e-3 게이트.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q6k_coalesced_matches_cpu_reference() {
        let backend = MetalBackend::new();
        for &(n, k) in &[
            (8usize, 5120usize),
            (7usize, 5120usize),
            (16usize, 6144usize),
            (13usize, 6144usize),
            (8usize, 17408usize),
            (7usize, 17408usize),
        ] {
            let num_blocks = k / 256;
            const BB: usize = 210;
            let total_bytes = n * num_blocks * BB;
            let mut weight: Vec<u8> = (0..total_bytes)
                .map(|i| ((i as u64 * 1103515245 + 12345) >> 7 & 0xFF) as u8)
                .collect();
            let nblocks_total = n * num_blocks;
            for bi in 0..nblocks_total {
                let base = bi * BB;
                // d(208-209) 만 유효 f16 (~0.02..0.05, q6_K scale 은 작음).
                let d = half::f16::from_f32(0.02 + ((bi % 6) as f32) * 0.005);
                let db = d.to_bits().to_le_bytes();
                weight[base + 208] = db[0];
                weight[base + 209] = db[1];
            }
            let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

            let mut reference = vec![0.0f32; n];
            for r in 0..n {
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let off = (r * num_blocks + b) * BB;
                    let blk = &weight[off..off + BB];
                    let deq = tests_fixture::q6k_dequant(blk);
                    let x_base = b * 256;
                    for l in 0..256 {
                        acc += deq[l] * input[x_base + l];
                    }
                }
                reference[r] = acc;
            }

            let gpu = backend.gemv_q6k_coalesced(&weight, &input, n, k);
            assert_eq!(gpu.len(), n);
            let gpu_nsg2 = backend.gemv_q6k_coalesced_nsg2(&weight, &input, n, k);
            assert_eq!(gpu_nsg2.len(), n);
            for r in 0..n {
                let rel = (gpu[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                let rel_nsg2 = (gpu_nsg2[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "q6k n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel}",
                    gpu[r],
                    reference[r]
                );
                assert!(
                    rel_nsg2 < 1e-3,
                    "q6k nsg2 n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel_nsg2}",
                    gpu_nsg2[r],
                    reference[r]
                );
            }
        }
    }

    /// llama.cpp q8_0 coalesced 이식 커널이 CPU reference(`dequantize_q8_0`)와 일치하는지.
    /// block size=32(K/32 blocks). 27B shape × 짝/홀 N. d(byte 0-1)만 유효 f16,
    /// qs(2-33)은 i8 라 random byte 도 유효. rel<1e-3 게이트.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q8_0_coalesced_matches_cpu_reference() {
        let backend = MetalBackend::new();
        for &(n, k) in &[
            (8usize, 5120usize),
            (7usize, 5120usize),
            (16usize, 6144usize),
            (13usize, 6144usize),
            (8usize, 17408usize),
            (7usize, 17408usize),
        ] {
            let num_blocks = k / 32;
            const BB: usize = 34;
            let total_bytes = n * num_blocks * BB;
            let mut weight: Vec<u8> = (0..total_bytes)
                .map(|i| ((i as u64 * 1103515245 + 12345) >> 7 & 0xFF) as u8)
                .collect();
            let nblocks_total = n * num_blocks;
            for bi in 0..nblocks_total {
                let base = bi * BB;
                // d(0-1) 만 유효 f16 (~0.01..0.04).
                let d = half::f16::from_f32(0.01 + ((bi % 6) as f32) * 0.005);
                let db = d.to_bits().to_le_bytes();
                weight[base] = db[0];
                weight[base + 1] = db[1];
            }
            let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

            let mut reference = vec![0.0f32; n];
            for r in 0..n {
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let off = (r * num_blocks + b) * BB;
                    let blk = &weight[off..off + BB];
                    let deq = tests_fixture::q8_0_dequant(blk);
                    let x_base = b * 32;
                    for l in 0..32 {
                        acc += deq[l] * input[x_base + l];
                    }
                }
                reference[r] = acc;
            }

            let gpu = backend.gemv_q8_0_coalesced(&weight, &input, n, k);
            assert_eq!(gpu.len(), n);
            for r in 0..n {
                let rel = (gpu[r] - reference[r]).abs() / reference[r].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "q8_0 n={n} k={k} nb={num_blocks} r={r} gpu={} ref={} rel={rel}",
                    gpu[r],
                    reference[r]
                );
            }
        }
    }

    /// SIMD 협력 커널이 기존 gemv_q4k 와 동일 결과(multi-row, multi-block, 다양 input).
    /// K=12288(48 블록 > 32 lane, stride 순회) + K=4096(16 블록 < 32 lane, 일부 idle) 둘 다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q4k_simd_matches_baseline_kernel() {
        let backend = MetalBackend::new();
        for &(n, k) in &[
            (4usize, 4096usize),
            (3usize, 12288usize),
            (5usize, 512usize),
        ] {
            let num_blocks = k / 256;
            let single = tests_fixture::q4k_block_fixed();
            let weight: Vec<u8> = single
                .iter()
                .cycle()
                .take(n * num_blocks * 144)
                .copied()
                .collect();
            let input: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

            let baseline = backend.gemv_q4k(&weight, &input, n, k);
            let simd = backend.gemv_q4k_simd(&weight, &input, n, k);
            assert_eq!(simd.len(), n);
            for i in 0..n {
                let rel = (simd[i] - baseline[i]).abs() / baseline[i].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "n={n} k={k} i={i} simd={} baseline={} rel={rel}",
                    simd[i],
                    baseline[i]
                );
            }
        }
    }

    /// Q6_K SIMD 커널이 CPU reference(`dequantize_q6_k`)와 일치하는지(단일 블록).
    /// baseline gemv_q6k 와 동일 수치 결과 — dequant 이식 정확도 검증.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q6k_simd_matches_cpu_reference() {
        let block = tests_fixture::q6k_block_fixed();
        let weights = tests_fixture::q6k_dequant(&block);
        let input = vec![1.0f32; 256];
        // CPU reference: dequant 결과와 input(all-1) 의 내적 = dequant 합산.
        let reference: f32 = weights.iter().sum();
        let backend = MetalBackend::new();
        let gpu = backend.gemv_q6k_simd(&block, &input, 1, 256);
        assert_eq!(gpu.len(), 1);
        let rel = (gpu[0] - reference).abs() / reference.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "simd gpu={} cpu_ref={reference} rel={rel}",
            gpu[0]
        );
    }

    /// Q6_K SIMD 커널이 baseline gemv_q6k 커널과 동일 결과(multi-row, multi-block).
    /// K=512(2 블록, num_blocks>lane → stride 분할), K=256(1 블록, 일부 lane idle) 둘 다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q6k_simd_matches_baseline_kernel() {
        let backend = MetalBackend::new();
        let single = tests_fixture::q6k_block_fixed();
        // K=4096(nb16→sub-block m2,T32, 9B attn_v) · K=12288(nb48→stride, 9B ffn_down)
        // 포함 — pm21 P1 lane-saturation 분기 전부 커버.
        for &(n, k) in &[
            (4usize, 512usize),
            (3usize, 256usize),
            (5usize, 1024usize),
            (4usize, 4096usize),
            (3usize, 12288usize),
        ] {
            let num_blocks = k / 256;
            let weight: Vec<u8> = single
                .iter()
                .cycle()
                .take(n * num_blocks * 210)
                .copied()
                .collect();
            let input: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

            let baseline = backend.gemv_q6k(&weight, &input, n, k);
            let simd = backend.gemv_q6k_simd(&weight, &input, n, k);
            assert_eq!(simd.len(), n);
            for i in 0..n {
                let rel = (simd[i] - baseline[i]).abs() / baseline[i].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "n={n} k={k} i={i} simd={} baseline={} rel={rel}",
                    simd[i],
                    baseline[i]
                );
            }
        }
    }

    /// Q6_K resident NoCopy 경로가 SIMD 복사 경로와 동일해야 output projection 에서
    /// 매 decode마다 output weight 를 다시 업로드하지 않는다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q6k_simd_resident_matches_copy_kernel() {
        let backend = MetalBackend::new();
        let single = tests_fixture::q6k_block_fixed();
        let n = 4usize;
        let k = 1024usize;
        let num_blocks = k / 256;
        let weight: Vec<u8> = single
            .iter()
            .cycle()
            .take(n * num_blocks * 210)
            .copied()
            .collect();
        let input: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) * 0.07).collect();

        let copy = backend.gemv_q6k_simd(&weight, &input, n, k);
        let resident = backend.gemv_q6k_simd_resident(&weight, &input, n, k);

        assert_eq!(resident.len(), n);
        for i in 0..n {
            let rel = (resident[i] - copy[i]).abs() / copy[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} resident={} copy={} rel={rel}",
                resident[i],
                copy[i]
            );
        }
    }

    /// Output projection argmax probe must read back only the selected token id,
    /// not the full logits vector. The token must match the existing resident GEMV
    /// path plus host argmax.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_output_argmax_q6k_resident_matches_full_logits_argmax() {
        let backend = MetalBackend::new();
        let single = tests_fixture::q6k_block_fixed();
        let n = 8usize;
        let k = 1024usize;
        let num_blocks = k / 256;
        let weight: Vec<u8> = single
            .iter()
            .cycle()
            .take(n * num_blocks * 210)
            .copied()
            .collect();
        let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();

        let logits = backend.gemv_q6k_simd_resident(&weight, &input, n, k);
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        let report = backend.output_argmax_q6k_simd_resident(&weight, &input, n, k);

        assert!(report.did_run);
        assert_eq!(report.token_id, expected);
        assert_eq!(report.cmd_buffers, 1);
        assert_eq!(
            report.readback_bytes, 4,
            "argmax path must read back only the u32 token id"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_output_argmax_q4k_resident_matches_full_logits_argmax() {
        let backend = MetalBackend::new();
        let single = tests_fixture::q4k_block_fixed();
        let n = 8usize;
        let k = 1024usize;
        let num_blocks = k / 256;
        let weight: Vec<u8> = single
            .iter()
            .cycle()
            .take(n * num_blocks * 144)
            .copied()
            .collect();
        let input: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.025).collect();

        let logits = backend.gemv_q4k_simd_resident(&weight, &input, n, k);
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        let report = backend.output_argmax_q4k_simd_resident(&weight, &input, n, k);

        assert!(report.did_run);
        assert_eq!(report.token_id, expected);
        assert_eq!(report.cmd_buffers, 1);
        assert_eq!(
            report.readback_bytes, 4,
            "argmax path must read back only the u32 token id"
        );
    }

    /// Q5_K SIMD 커널(pm21 P1 lane-saturation)이 baseline gemv_q5k 와 일치하는지.
    /// K=4096(nb16→m2,T32, 9B qkv/ssm_out) · 2048(nb8→m4,T32) · 12288(stride) · 512
    /// 으로 m 분기 전부 커버. baseline/simd 는 동일 dequant 이므로 임의 유효 byte 패턴
    /// 으로 충분(fixture 불필요) — reduction 순서만 다름.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q5k_simd_matches_baseline_kernel() {
        let backend = MetalBackend::new();
        let single: Vec<u8> = (0..176u32).map(|i| ((i * 7 + 13) % 251) as u8).collect();
        for &(n, k) in &[
            (4usize, 4096usize),
            (3usize, 12288usize),
            (5usize, 2048usize),
            (4usize, 512usize),
        ] {
            let num_blocks = k / 256;
            let weight: Vec<u8> = single
                .iter()
                .cycle()
                .take(n * num_blocks * 176)
                .copied()
                .collect();
            let input: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

            let baseline = backend.gemv_q5k(&weight, &input, n, k);
            let simd = backend.gemv_q5k_simd(&weight, &input, n, k);
            assert_eq!(simd.len(), n);
            for i in 0..n {
                let rel = (simd[i] - baseline[i]).abs() / baseline[i].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "n={n} k={k} i={i} simd={} baseline={} rel={rel}",
                    simd[i],
                    baseline[i]
                );
            }
        }
    }

    /// NoCopy zero-copy 경로가 복사 경로와 동일 결과를 내는지(실기기).
    /// page_off ≠ 0 을 강제: 3-page 버퍼 내부의 page 경계 + 500 위치에 block 을 둬,
    /// page_align 이 내린 aligned base 가 항상 우리 할당 안(유효 매핑)이도록 한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_q4k_resident_matches_reference() {
        let block = tests_fixture::q4k_block_fixed();
        let input = vec![1.0f32; 256];
        let reference = tests_fixture::q4k_dequant_sum(&block);

        const PAGE: usize = 16384;
        let mut buf = vec![0u8; 3 * PAGE];
        let base = buf.as_ptr() as usize;
        let first_page = (base + PAGE - 1) & !(PAGE - 1); // buf 내부 첫 page 경계
        let target = first_page + 500; // page 중간(off=500), buf 내부 보장
        let off = target - base;
        buf[off..off + block.len()].copy_from_slice(&block);
        let raw = &buf[off..off + block.len()];
        assert_ne!((raw.as_ptr() as usize) % PAGE, 0, "page_off ≠ 0 강제");

        let backend = MetalBackend::new();
        let gpu = backend.gemv_q4k_resident(raw, &input, 1, 256);
        assert_eq!(gpu.len(), 1);
        let rel = (gpu[0] - reference).abs() / reference.abs().max(1e-6);
        assert!(rel < 1e-3, "gpu={} ref={} rel={rel}", gpu[0], reference);
    }

    /// 같은 weight 로 2회 호출 시 resident 캐시가 buffer 를 재사용(entry 1개)하는지.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_resident_cache_reuses_buffer() {
        let block = tests_fixture::q4k_block_fixed();
        let input = vec![1.0f32; 256];
        let reference = tests_fixture::q4k_dequant_sum(&block);

        const PAGE: usize = 16384;
        let mut buf = vec![0u8; 3 * PAGE];
        let base = buf.as_ptr() as usize;
        let first_page = (base + PAGE - 1) & !(PAGE - 1);
        let off = first_page + 500 - base;
        buf[off..off + block.len()].copy_from_slice(&block);
        let raw = &buf[off..off + block.len()];

        let backend = MetalBackend::new();
        // 같은 raw 슬라이스로 2회 — 2번째는 캐시 hit.
        let r1 = backend.gemv_q4k_resident(raw, &input, 1, 256);
        let r2 = backend.gemv_q4k_resident(raw, &input, 1, 256);
        assert!((r1[0] - reference).abs() / reference.abs().max(1e-6) < 1e-3);
        assert_eq!(r1, r2);
        assert_eq!(backend.resident.borrow().len(), 1, "캐시 entry 1개");
    }

    /// pm113: MLA prefill slot-batch GEMV — slot = token*heads+head 가 weight
    /// head(slot%heads) 를 커널 내부에서 고르는지 CPU dequant-dot oracle 대조.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_slots_gemv_matches_cpu_oracle() {
        let heads = 3usize;
        let n_per_head = 4usize;
        let k = 64usize; // 2 Q8_0 blocks/row
        let tokens = 2usize;
        let slots = tokens * heads;
        let raw = tests_fixture::scaled_q8_0_matrix(heads * n_per_head, k, 7);

        let mut input = vec![0.0f32; slots * k];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.25;
        }

        let nb = k / 32;
        let row_bytes = nb * 34;
        let mut reference = vec![0.0f32; slots * n_per_head];
        for slot in 0..slots {
            let head = slot % heads;
            for row in 0..n_per_head {
                let row_start = (head * n_per_head + row) * row_bytes;
                let mut acc = 0.0f32;
                for block in 0..nb {
                    let deq = tests_fixture::q8_0_dequant(
                        &raw[row_start + block * 34..row_start + (block + 1) * 34],
                    );
                    let x = &input[slot * k + block * 32..slot * k + (block + 1) * 32];
                    acc += deq.iter().zip(x).map(|(&w, &v)| w * v).sum::<f32>();
                }
                reference[slot * n_per_head + row] = acc;
            }
        }

        let backend = MetalBackend::new();
        let out = backend
            .glm_mla_head_gemv_q8_0_slots_resident(&raw, &input, slots, heads, n_per_head, k);
        assert_eq!(out.len(), reference.len());
        for (i, (&metal_v, &cpu_v)) in out.iter().zip(&reference).enumerate() {
            let rel = (metal_v - cpu_v).abs() / cpu_v.abs().max(1e-5);
            assert!(
                rel < 1e-3,
                "slot-batch mismatch at {i}: metal={metal_v} cpu={cpu_v} rel={rel}"
            );
        }
    }

    /// pm117: MLA prefill slot-batch Q5_K GEMV — Q8_0 slots 와 같은 slot/head 매핑을
    /// Q5_K super-block dequant-dot oracle 로 대조 (o/q_a dense 는 heads=1).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_slots_q5k_gemv_matches_cpu_oracle() {
        let heads = 2usize;
        let n_per_head = 3usize;
        let k = 512usize; // 2 Q5_K super-blocks/row
        let tokens = 2usize;
        let slots = tokens * heads;
        let raw = tests_fixture::scaled_q5k_matrix(heads * n_per_head, k, 5);

        let mut input = vec![0.0f32; slots * k];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 0.125;
        }

        let nb = k / 256;
        let row_bytes = nb * 176;
        let mut reference = vec![0.0f32; slots * n_per_head];
        for slot in 0..slots {
            let head = slot % heads;
            for row in 0..n_per_head {
                let row_start = (head * n_per_head + row) * row_bytes;
                let mut acc = 0.0f32;
                for block in 0..nb {
                    let deq = tests_fixture::q5k_dequant(
                        &raw[row_start + block * 176..row_start + (block + 1) * 176],
                    );
                    let x = &input[slot * k + block * 256..slot * k + (block + 1) * 256];
                    acc += deq.iter().zip(x).map(|(&w, &v)| w * v).sum::<f32>();
                }
                reference[slot * n_per_head + row] = acc;
            }
        }

        let backend = MetalBackend::new();
        let out =
            backend.glm_mla_head_gemv_q5k_slots_resident(&raw, &input, slots, heads, n_per_head, k);
        assert_eq!(out.len(), reference.len());
        for (i, (&metal_v, &cpu_v)) in out.iter().zip(&reference).enumerate() {
            let rel = (metal_v - cpu_v).abs() / cpu_v.abs().max(1e-5);
            assert!(
                rel < 1e-3,
                "q5k slot-batch mismatch at {i}: metal={metal_v} cpu={cpu_v} rel={rel}"
            );
        }
    }

    /// pm119: layer fused (front→kv rms/rope→q_pe rope→attn→v_b→o 단일 cb) 를
    /// CPU 참조 체인 (glm_dsa prefill_layer 산술 재현) 과 대조. GPU rms/rope/
    /// f16 캐스팅의 수치 차이만 허용 — projected rel<1e-2 (양자화 GEMV 체인
    /// 누적), cache_tail 은 f16 나름 rel<1e-2.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_layer_fused_matches_cpu_reference() {
        let seq_len = 3usize;
        let hidden = 256usize;
        let q_rank = 64usize;
        let heads = 2usize;
        let q_nope_dim = 32usize;
        let rope_dim = 16usize;
        let qk_dim = q_nope_dim + rope_dim;
        let q_dim = heads * qk_dim;
        let kv_rank = 32usize;
        let kv_width = kv_rank + rope_dim;
        let value_dim = 128usize;
        let o_rows = 5usize;
        let o_cols = heads * value_dim;
        let pos_start = 2usize;
        let eps = 1e-5f32;
        let theta = 10000.0f32;
        let scale = 1.0f32 / (qk_dim as f32).sqrt();

        let qa_raw = tests_fixture::scaled_q5k_matrix(q_rank, hidden, 5);
        let qb_raw = tests_fixture::scaled_q8_0_matrix(q_dim, q_rank, 7);
        let kva_raw = tests_fixture::scaled_q8_0_matrix(kv_width, hidden, 9);
        let kb_raw = tests_fixture::scaled_q8_0_matrix(heads * kv_rank, q_nope_dim, 11);
        let vb_raw = tests_fixture::scaled_q8_0_matrix(heads * value_dim, kv_rank, 13);
        let o_raw = tests_fixture::scaled_q5k_matrix(o_rows, o_cols, 15);
        let mut qa_norm_w = vec![0.0f32; q_rank];
        for (i, v) in qa_norm_w.iter_mut().enumerate() {
            *v = 1.0 + ((i % 7) as f32 - 3.0) * 0.05;
        }
        let mut kv_norm_w = vec![0.0f32; kv_rank];
        for (i, v) in kv_norm_w.iter_mut().enumerate() {
            *v = 1.0 + ((i % 5) as f32 - 2.0) * 0.04;
        }
        let mut normed = vec![0.0f32; seq_len * hidden];
        for (i, v) in normed.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.11;
        }
        let mut cache_base = vec![0u16; pos_start * kv_width];
        for (i, v) in cache_base.iter_mut().enumerate() {
            *v = half::f16::from_f32(((i % 19) as f32 - 9.0) * 0.07).to_bits();
        }

        let backend = MetalBackend::new();
        // ---- CPU 참조 체인 (분리 slots dispatch + CPU rms/rope/f16/attn) ----
        let qa = backend
            .glm_mla_head_gemv_q5k_slots_resident(&qa_raw, &normed, seq_len, 1, q_rank, hidden);
        let mut qa_norm = vec![0.0f32; qa.len()];
        for token in 0..seq_len {
            let row = &qa[token * q_rank..(token + 1) * q_rank];
            let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / q_rank as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for (i, out) in qa_norm[token * q_rank..(token + 1) * q_rank]
                .iter_mut()
                .enumerate()
            {
                *out = row[i] * inv_rms * qa_norm_w[i];
            }
        }
        let q_ref = backend
            .glm_mla_head_gemv_q8_0_slots_resident(&qb_raw, &qa_norm, seq_len, 1, q_dim, q_rank);
        let kv_ref = backend
            .glm_mla_head_gemv_q8_0_slots_resident(&kva_raw, &normed, seq_len, 1, kv_width, hidden);
        let slots = seq_len * heads;
        let mut qnope = vec![0.0f32; slots * q_nope_dim];
        for slot in 0..slots {
            let token = slot / heads;
            let head = slot % heads;
            let base = token * q_dim + head * qk_dim;
            qnope[slot * q_nope_dim..(slot + 1) * q_nope_dim]
                .copy_from_slice(&q_ref[base..base + q_nope_dim]);
        }
        let qabs_ref = backend.glm_mla_head_gemv_q8_0_slots_resident(
            &kb_raw, &qnope, slots, heads, kv_rank, q_nope_dim,
        );
        // CPU rope (rope_inplace 수식) — theta_scale 누적곱.
        let theta_scale = theta.powf(-2.0f32 / rope_dim as f32);
        let rope = |vals: &mut [f32], pos: usize| {
            let mut angle = pos as f32;
            let mut i = 0usize;
            while i < vals.len() {
                let (c, s) = (angle.cos(), angle.sin());
                let (x0, x1) = (vals[i], vals[i + 1]);
                vals[i] = x0 * c - x1 * s;
                vals[i + 1] = x0 * s + x1 * c;
                angle *= theta_scale;
                i += 2;
            }
        };
        // cache (f16 bits) = base + CPU 생성 tail.
        let cache_len = pos_start + seq_len;
        let mut cache_ref = vec![0u16; cache_len * kv_width];
        cache_ref[..cache_base.len()].copy_from_slice(&cache_base);
        for token in 0..seq_len {
            let raw = &kv_ref[token * kv_width..(token + 1) * kv_width];
            let mean_sq = raw[..kv_rank].iter().map(|&v| v * v).sum::<f32>() / kv_rank as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let mut row = vec![0.0f32; kv_width];
            for i in 0..kv_rank {
                row[i] = raw[i] * inv_rms * kv_norm_w[i];
            }
            row[kv_rank..].copy_from_slice(&raw[kv_rank..]);
            rope(&mut row[kv_rank..], pos_start + token);
            for (i, &v) in row.iter().enumerate() {
                cache_ref[(pos_start + token) * kv_width + i] = half::f16::from_f32(v).to_bits();
            }
        }
        // q_pe (CPU pack + rope).
        let mut qpe_ref = vec![0.0f32; slots * rope_dim];
        for slot in 0..slots {
            let token = slot / heads;
            let head = slot % heads;
            let base = token * q_dim + head * qk_dim + q_nope_dim;
            let out = &mut qpe_ref[slot * rope_dim..(slot + 1) * rope_dim];
            out.copy_from_slice(&q_ref[base..base + rope_dim]);
            rope(out, pos_start + token);
        }
        // attn (기존 GPU attn 커널 — pm116 채택 계보).
        let latent_ref = backend.glm_mla_prefill_attn(
            &qabs_ref, &qpe_ref, &cache_ref, slots, heads, kv_rank, rope_dim, pos_start, scale,
        );
        let concat_ref = backend.glm_mla_head_gemv_q8_0_slots_resident(
            &vb_raw,
            &latent_ref,
            slots,
            heads,
            value_dim,
            kv_rank,
        );
        let projected_ref = backend.glm_mla_head_gemv_q5k_slots_resident(
            &o_raw,
            &concat_ref,
            seq_len,
            1,
            o_rows,
            o_cols,
        );

        // ---- fused ----
        let out = backend.glm_mla_layer_fused_resident(
            &qa_raw,
            &qa_norm_w,
            &qb_raw,
            &kva_raw,
            &kb_raw,
            &kv_norm_w,
            &cache_base,
            &vb_raw,
            &o_raw,
            true,
            &normed,
            seq_len,
            hidden,
            q_rank,
            q_dim,
            kv_width,
            heads,
            qk_dim,
            q_nope_dim,
            kv_rank,
            rope_dim,
            pos_start,
            value_dim,
            o_rows,
            eps,
            theta,
            scale,
        );
        assert_eq!(out.cache_tail.len(), seq_len * kv_width);
        for (i, (&g, &w)) in out
            .cache_tail
            .iter()
            .zip(&cache_ref[pos_start * kv_width..])
            .enumerate()
        {
            let gv = half::f16::from_bits(g).to_f32();
            let wv = half::f16::from_bits(w).to_f32();
            let rel = (gv - wv).abs() / wv.abs().max(1e-3);
            assert!(rel < 1e-2, "cache_tail mismatch at {i}: got={gv} want={wv}");
        }
        assert_eq!(out.projected.len(), projected_ref.len());
        for (i, (&g, &w)) in out.projected.iter().zip(&projected_ref).enumerate() {
            let rel = (g - w).abs() / w.abs().max(1e-3);
            assert!(
                rel < 1e-2,
                "projected mismatch at {i}: got={g} want={w} rel={rel}"
            );
        }
    }

    /// pm118 연장: front fused (q_a→rms→q_b/kv_a→pack→k_b 단일 cb) 를 분리
    /// dispatch + CPU rms 참조 체인과 대조. GPU rms 는 리덕션 순서가 달라
    /// bit-identical 은 아님 — rel<1e-3.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_front_slots_fused_matches_split_chain() {
        let seq_len = 3usize;
        let hidden = 256usize; // Q5_K super-block
        let q_rank = 64usize;
        let heads = 2usize;
        let q_nope_dim = 32usize;
        let rope_dim = 16usize;
        let qk_dim = q_nope_dim + rope_dim;
        let q_dim = heads * qk_dim;
        let kv_rank = 32usize;
        let kv_width = kv_rank + rope_dim;
        let eps = 1e-5f32;

        let qa_raw = tests_fixture::scaled_q5k_matrix(q_rank, hidden, 5);
        let qb_raw = tests_fixture::scaled_q8_0_matrix(q_dim, q_rank, 7);
        let kva_raw = tests_fixture::scaled_q8_0_matrix(kv_width, hidden, 9);
        let kb_raw = tests_fixture::scaled_q8_0_matrix(heads * kv_rank, q_nope_dim, 11);
        let mut qa_norm_w = vec![0.0f32; q_rank];
        for (i, v) in qa_norm_w.iter_mut().enumerate() {
            *v = 1.0 + ((i % 7) as f32 - 3.0) * 0.05;
        }
        let mut normed = vec![0.0f32; seq_len * hidden];
        for (i, v) in normed.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.11;
        }

        let backend = MetalBackend::new();
        // 참조 체인: 분리 slots dispatch + CPU rms.
        let qa = backend
            .glm_mla_head_gemv_q5k_slots_resident(&qa_raw, &normed, seq_len, 1, q_rank, hidden);
        let mut qa_norm = vec![0.0f32; qa.len()];
        for token in 0..seq_len {
            let row = &qa[token * q_rank..(token + 1) * q_rank];
            let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / q_rank as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for (i, out) in qa_norm[token * q_rank..(token + 1) * q_rank]
                .iter_mut()
                .enumerate()
            {
                *out = row[i] * inv_rms * qa_norm_w[i];
            }
        }
        let q_ref = backend
            .glm_mla_head_gemv_q8_0_slots_resident(&qb_raw, &qa_norm, seq_len, 1, q_dim, q_rank);
        let kv_ref = backend
            .glm_mla_head_gemv_q8_0_slots_resident(&kva_raw, &normed, seq_len, 1, kv_width, hidden);
        let slots = seq_len * heads;
        let mut qnope = vec![0.0f32; slots * q_nope_dim];
        for slot in 0..slots {
            let token = slot / heads;
            let head = slot % heads;
            let base = token * q_dim + head * qk_dim;
            qnope[slot * q_nope_dim..(slot + 1) * q_nope_dim]
                .copy_from_slice(&q_ref[base..base + q_nope_dim]);
        }
        let qabs_ref = backend.glm_mla_head_gemv_q8_0_slots_resident(
            &kb_raw, &qnope, slots, heads, kv_rank, q_nope_dim,
        );

        let (q, kv_raw, qabs) = backend.glm_mla_front_slots_fused_resident(
            &qa_raw, &qa_norm_w, &qb_raw, &kva_raw, &kb_raw, &normed, seq_len, hidden, q_rank,
            q_dim, kv_width, heads, qk_dim, q_nope_dim, kv_rank, eps,
        );
        let assert_close = |name: &str, got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len(), "{name} len");
            for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                let rel = (g - w).abs() / w.abs().max(1e-5);
                assert!(
                    rel < 1e-3,
                    "{name} mismatch at {i}: got={g} want={w} rel={rel}"
                );
            }
        };
        assert_close("q", &q, &q_ref);
        assert_close("kv_raw", &kv_raw, &kv_ref);
        assert_close("q_absorbed", &qabs, &qabs_ref);
    }

    /// pm118: v_b slots → o slots 단일 command buffer fused 가 분리 dispatch 2회와
    /// bit-identical 인지 대조 (같은 slots 커널·같은 입력 — 산술 불변).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_vb_o_fused_matches_split_dispatch() {
        let heads = 2usize;
        let value_dim = 128usize; // o_cols = heads*value_dim = 256 (Q5_K super-block)
        let kv_rank = 64usize;
        let o_rows = 3usize;
        let tokens = 2usize;
        let slots = tokens * heads;
        let o_cols = heads * value_dim;
        let vb_raw = tests_fixture::scaled_q8_0_matrix(heads * value_dim, kv_rank, 7);
        let o_raw = tests_fixture::scaled_q5k_matrix(o_rows, o_cols, 5);

        let mut latent = vec![0.0f32; slots * kv_rank];
        for (i, v) in latent.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.25;
        }

        let backend = MetalBackend::new();
        let mid = backend.glm_mla_head_gemv_q8_0_slots_resident(
            &vb_raw, &latent, slots, heads, value_dim, kv_rank,
        );
        let reference =
            backend.glm_mla_head_gemv_q5k_slots_resident(&o_raw, &mid, tokens, 1, o_rows, o_cols);
        let fused = backend.glm_mla_vb_o_fused_resident(
            &vb_raw, &o_raw, true, &latent, slots, heads, value_dim, kv_rank, o_rows,
        );
        assert_eq!(fused, reference, "fused vs split must be bit-identical");
    }

    /// pm116: GLM MLA prefill attention — slot 별 causal score/softmax/latent 누적을
    /// CPU 2-pass softmax 참조 구현과 대조 (online softmax 수치 순서 차이만 허용).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_glm_mla_prefill_attn_matches_cpu_oracle() {
        let heads = 2usize;
        let tokens = 3usize;
        let pos_start = 1usize;
        let kv_rank = 64usize;
        let rope_dim = 16usize;
        let kv_width = kv_rank + rope_dim;
        let slots = tokens * heads;
        let cache_len = pos_start + tokens;
        let scale = 1.0f32 / ((kv_rank + rope_dim) as f32).sqrt();

        let mut q_absorbed = vec![0.0f32; slots * kv_rank];
        for (i, v) in q_absorbed.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 0.05;
        }
        let mut q_pe = vec![0.0f32; slots * rope_dim];
        for (i, v) in q_pe.iter_mut().enumerate() {
            *v = ((i % 11) as f32 - 5.0) * 0.07;
        }
        let mut cache = vec![0u16; cache_len * kv_width];
        for (i, v) in cache.iter_mut().enumerate() {
            *v = half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.03).to_bits();
        }

        // CPU 참조: glm_dsa::prefill_layer 스칼라 루프와 동일한 2-pass softmax.
        let mut reference = vec![0.0f32; slots * kv_rank];
        for slot in 0..slots {
            let token = slot / heads;
            let attend_len = pos_start + token + 1;
            let mut scores = vec![0.0f32; attend_len];
            for (j, score) in scores.iter_mut().enumerate() {
                let row = &cache[j * kv_width..(j + 1) * kv_width];
                let latent_dot = q_absorbed[slot * kv_rank..(slot + 1) * kv_rank]
                    .iter()
                    .zip(&row[..kv_rank])
                    .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                    .sum::<f32>();
                let rope_dot = q_pe[slot * rope_dim..(slot + 1) * rope_dim]
                    .iter()
                    .zip(&row[kv_rank..])
                    .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                    .sum::<f32>();
                *score = (latent_dot + rope_dot) * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            for score in scores.iter_mut() {
                *score /= sum;
            }
            let out = &mut reference[slot * kv_rank..(slot + 1) * kv_rank];
            for (j, &p) in scores.iter().enumerate() {
                let row = &cache[j * kv_width..j * kv_width + kv_rank];
                for (o, &bits) in out.iter_mut().zip(row) {
                    *o += p * half::f16::from_bits(bits).to_f32();
                }
            }
        }

        let backend = MetalBackend::new();
        let out = backend.glm_mla_prefill_attn(
            &q_absorbed,
            &q_pe,
            &cache,
            slots,
            heads,
            kv_rank,
            rope_dim,
            pos_start,
            scale,
        );
        assert_eq!(out.len(), reference.len());
        for (i, (&metal_v, &cpu_v)) in out.iter().zip(&reference).enumerate() {
            let rel = (metal_v - cpu_v).abs() / cpu_v.abs().max(1e-5);
            assert!(
                rel < 1e-3,
                "prefill attn mismatch at {i}: metal={metal_v} cpu={cpu_v} rel={rel}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemv_f32_matches_cpu() {
        // weight [N=2 x K=3] row-major, input [K=3]
        let weight = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = vec![1.0f32, 1.0, 1.0];
        let n = 2usize;
        let k = 3usize;
        let cpu: Vec<f32> = (0..n)
            .map(|r| (0..k).map(|c| weight[r * k + c] * input[c]).sum())
            .collect();
        let gpu = MetalBackend::new().gemv_f32(&weight, &input, n, k);
        assert_eq!(gpu.len(), n);
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert!((g - c).abs() < 1e-4, "gpu={g} cpu={c}");
        }
    }

    /// SIMD 협력 커널 vs 기존 커널 GEMV 시간 비교 (ABAB median, decode 차원).
    /// 실행: `cargo test -p rnb-backend-metal --features metal --release \
    ///         -- --ignored --nocapture perf_q4k_simd_vs_baseline`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "perf — requires Metal device; run manually"]
    fn perf_q4k_simd_vs_baseline() {
        use std::time::Instant;
        let backend = MetalBackend::new();
        const RUNS: usize = 9;

        for &(n, k) in &[(4096usize, 4096usize), (4096usize, 12288usize)] {
            let num_blocks = n * (k / 256);
            let single = tests_fixture::q4k_block_fixed();
            let weight: Vec<u8> = single
                .iter()
                .cycle()
                .take(num_blocks * 144)
                .copied()
                .collect();
            let input = vec![1.0f32; k];

            // warmup 양쪽
            let _ = backend.gemv_q4k(&weight, &input, n, k);
            let _ = backend.gemv_q4k_simd(&weight, &input, n, k);

            let mut a = Vec::new(); // baseline
            let mut b = Vec::new(); // simd
            for _ in 0..RUNS {
                let t = Instant::now();
                let _ = backend.gemv_q4k(&weight, &input, n, k);
                a.push(t.elapsed().as_secs_f64() * 1000.0);
                let t = Instant::now();
                let _ = backend.gemv_q4k_simd(&weight, &input, n, k);
                b.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let med = |v: &[f64]| {
                let mut s = v.to_vec();
                s.sort_by(|x, y| x.partial_cmp(y).unwrap());
                s[s.len() / 2]
            };
            // CPU NEON(멀티스레드) 같은 N,K — 순수 metal vs 순수 CPU 격리.
            let bytes_per_row = (k / 256) * 144;
            let q8k = rnb_cpu::gemm::activation_q8::quantize_input_q8k(&input);
            let mut cpu_out = vec![0.0f32; n];
            rnb_cpu::gemm::neon_dot::gemv_q4_k_int8(
                &weight,
                &q8k,
                &mut cpu_out,
                n,
                k,
                1,
                bytes_per_row,
            );
            let mut c = Vec::new();
            for _ in 0..RUNS {
                let t = Instant::now();
                rnb_cpu::gemm::neon_dot::gemv_q4_k_int8(
                    &weight,
                    &q8k,
                    &mut cpu_out,
                    n,
                    k,
                    1,
                    bytes_per_row,
                );
                c.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let (am, bm, cm) = (med(&a), med(&b), med(&c));
            eprintln!(
                "[simd-perf] N={n} K={k}: metal_baseline={am:.3}ms metal_simd={bm:.3}ms \
                 cpu_neon={cm:.3}ms | simd vs cpu={:.2}x",
                cm / bm
            );
        }
    }

    /// # Q4_K GEMV perf (캐싱 후 — N=4096, K=4096)
    ///
    /// 커널 상태: MetalContext 캐싱 (device/queue/pipeline 한 번만 빌드),
    /// per-call = buffer alloc + encode + dispatch + readback 만.
    ///
    /// 단계 분해: `RNB_METAL_TIMING=1` 시 alloc/dispatch/readback 3단계 시간 출력.
    /// (device/compile 단계는 new() 에서 처리되므로 per-call 타이밍에서 제거됨)
    ///
    /// 실행: `RNB_METAL_TIMING=1 cargo test -p rnb-backend-metal --features metal --release \
    ///         -- --ignored --nocapture perf_baseline_q4k_n4096_k4096`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "perf baseline — requires Metal device; run manually for measurement"]
    fn perf_baseline_q4k_n4096_k4096() {
        use std::time::Instant;

        const N: usize = 4096;
        const K: usize = 4096;
        const RUNS: usize = 5;

        // weight 버퍼 구성: N × (K/256) × 1 블록 = N×16 블록 반복
        let single_block = tests_fixture::q4k_block_fixed();
        let num_blocks = N * (K / 256); // 4096 × 16 = 65536 blocks
        let block_bytes: Vec<u8> = single_block
            .iter()
            .cycle()
            .take(num_blocks * 144)
            .copied()
            .collect();
        let input = vec![1.0f32; K];

        // MetalContext 는 new() 에서 한 번만 빌드됨 (device open + MSL compile 여기서만)
        let backend = MetalBackend::new();

        eprintln!(
            "\n[metal-perf] Q4_K GEMV N={N} K={K}  weight={:.1}MB  input={:.1}KB",
            (block_bytes.len() as f64) / 1_048_576.0,
            (input.len() * 4) as f64 / 1024.0,
        );
        eprintln!("[metal-perf] kernel state: MetalContext cached (device/queue/pipeline reused)");
        eprintln!("[metal-perf] --- warmup (run 0) ---");

        // warmup: run 0 (pipeline 캐싱됐으나 GPU driver/buffer 첫 실행 warmup)
        let _ = backend.gemv_q4k(&block_bytes, &input, N, K);

        // 5회 측정
        eprintln!(
            "[metal-perf] --- measured runs 1-{RUNS} (RNB_METAL_TIMING=1 for stage detail) ---"
        );
        let mut gpu_times_ms = Vec::with_capacity(RUNS);
        for i in 1..=RUNS {
            let t = Instant::now();
            let _ = backend.gemv_q4k(&block_bytes, &input, N, K);
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[metal-perf]   run {i}: {elapsed:.2}ms");
            gpu_times_ms.push(elapsed);
        }

        // CPU reference: dequant + dot
        eprintln!("[metal-perf] --- CPU reference (dequant+dot, 5 runs) ---");
        let mut cpu_times_ms = Vec::with_capacity(RUNS);
        for i in 1..=RUNS {
            let t = Instant::now();
            let mut out = vec![0.0f32; N];
            for row in 0..N {
                let mut acc = 0.0f32;
                for blk in 0..(K / 256) {
                    let off = (row * (K / 256) + blk) % num_blocks;
                    let bslice = &block_bytes[off * 144..(off + 1) * 144];
                    acc += tests_fixture::q4k_dequant_sum(bslice);
                }
                out[row] = acc;
            }
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[metal-perf]   cpu run {i}: {elapsed:.2}ms");
            cpu_times_ms.push(elapsed);
            let _ = out[0];
        }

        // median 계산
        let gpu_median = {
            let mut v = gpu_times_ms.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let cpu_median = {
            let mut v = cpu_times_ms.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        eprintln!(
            "\n[metal-perf] Q4_K GEMV N={N} K={K}: GPU median={gpu_median:.1}ms/call, CPU median={cpu_median:.1}ms/call"
        );
        eprintln!(
            "[metal-perf] (GPU 단계 분해: RNB_METAL_TIMING=1 으로 재실행하면 alloc/dispatch/readback 별 시간 출력)"
        );
        eprintln!("[metal-perf] NOTE: device/compile 비용 제거됨 (MetalContext 캐싱)");
    }

    /// pm48: prefill(seq_len>1) fused qk_norm→rope device 커널이 CPU oracle
    /// (per-head `rms_norm_into` → `rope_mrope_text_inplace`)과 일치하는지.
    /// CPU ground-truth 순서: projection.rs 가 qk_norm 먼저, forward/rope.rs 가 그다음
    /// rope_mrope_text. partial rope(n_rot < head_dim) 의 passthrough tail 도 같이 검증.
    /// Qwen3.6 27B 근사 shape(hd=256, n_rot=64, theta=1e7, GQA q24/k4) 포함.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_prefill_rope_qk_norm_matches_cpu_oracle() {
        use rnb_cpu::kernels::norm::rms_norm_into;
        use rnb_cpu::kernels::rope::rope_partial_inplace;

        let ctx = crate::compute::build_metal_context().expect("metal ctx");
        let eps = 1.0e-6f32;
        // (seq_len, num_heads, head_dim, n_rot, theta, pos_start, label)
        let cases: [(usize, usize, usize, usize, f32, usize, &str); 4] = [
            (8, 4, 256, 64, 1.0e7, 0, "small hd256 n_rot64 pos0"),
            (8, 4, 256, 64, 1.0e7, 13, "small hd256 n_rot64 pos13"),
            (12, 24, 256, 64, 1.0e7, 5, "27B q-side: nh24 hd256 n_rot64"),
            (12, 4, 256, 64, 1.0e7, 5, "27B k-side: nkv4 hd256 n_rot64"),
        ];

        let mut worst = 0f32;
        for (seq_len, num_heads, head_dim, n_rot, theta, pos_start, label) in cases {
            // 결정적 입력 (qk_norm_fixture 패턴) — token·head·dim 다 변주되도록.
            let total = seq_len * num_heads * head_dim;
            let data: Vec<f32> = (0..total)
                .map(|i| (((i % 13) as f32) - 6.0) * 0.05) // -0.3..0.3
                .collect();
            let weight: Vec<f32> = (0..head_dim)
                .map(|i| 0.5 + ((i % 5) as f32) * 0.1) // 0.5..0.9
                .collect();

            // CPU oracle: per-head RMSNorm (data layout = [seq*nh, head_dim] head-contiguous),
            // 그다음 token 별 partial RoPE (adjacent-pair, rope_partial_inplace 는 dim=nh*head_dim
            // 단위). 27B Qwen3.6 default(non-iMRoPE) 경로 — kernel 도 adjacent-pair 로 일치.
            let mut reference = vec![0.0f32; total];
            for h in 0..(seq_len * num_heads) {
                let off = h * head_dim;
                rms_norm_into(
                    &data[off..off + head_dim],
                    &weight,
                    eps,
                    &mut reference[off..off + head_dim],
                );
            }
            // rope: per-token (pos = pos_start + t). dim = num_heads*head_dim per token.
            let dim = num_heads * head_dim;
            for t in 0..seq_len {
                let toff = t * dim;
                rope_partial_inplace(
                    &mut reference[toff..toff + dim],
                    pos_start + t,
                    head_dim,
                    dim,
                    n_rot,
                    theta,
                );
            }

            let (gpu, gpu_ms) = crate::compute::prefill_rope_qk_norm_oracle_with_ctx(
                &ctx, &data, &weight, seq_len, num_heads, head_dim, n_rot, theta, eps, pos_start,
            );
            assert_eq!(gpu.len(), reference.len(), "[{label}] len");

            let mut max_rel = 0f32;
            for (i, (&g, &r)) in gpu.iter().zip(reference.iter()).enumerate() {
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-6);
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(
                    rel < 1e-3 || abs < 1e-4,
                    "[{label}] idx={i} gpu={g} ref={r} abs={abs} rel={rel}"
                );
            }
            worst = worst.max(max_rel);
            eprintln!(
                "[pm48] prefill rope+qk_norm [{label}] max_rel={max_rel:.2e} gpu_ms={gpu_ms:.3}"
            );
        }
        eprintln!("[pm48] prefill rope+qk_norm worst max_rel={worst:.2e} (< 1e-3 통과)");
    }

    #[test]
    fn decode_chain_report_defaults_to_no_run_and_counts_argmax_report() {
        let mut report = DecodeChainReport::default();
        assert!(!report.did_run);
        assert!(report.fallback_reason.is_none());
        assert_eq!(report.qwen_moe_layers, 0);

        report.cmd_buffers = 1;
        report.wait_us = 17;
        report.readback_bytes = 32;
        report.qwen_moe_layers = 2;
        report.output_argmax = OutputArgmaxReport {
            attempted: true,
            did_run: true,
            token_id: Some(3),
            cmd_buffers: 1,
            wait_us: 5,
            readback_bytes: 4,
            fallback_reason: None,
        };

        assert_eq!(report.total_cmd_buffers(), 2);
        assert_eq!(report.total_wait_us(), 22);
        assert_eq!(report.total_readback_bytes(), 36);
        assert_eq!(report.qwen_moe_layers, 2);
    }

    #[test]
    fn qwen_moe_decode_chain_carrier_key_includes_layer() {
        let layer0 = qwen_moe_decode_chain_carrier_key(0, 2, 256, 256, 4, 2);
        let layer1 = qwen_moe_decode_chain_carrier_key(1, 2, 256, 256, 4, 2);
        assert_ne!(
            layer0, layer1,
            "same-shape Qwen MoE decode chain carriers must be isolated by layer"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_two_same_shape_layers_matches_split_runs() {
        let fixture = tests_fixture::qwen_moe_gdn_decode_chain_fixture();
        let backend_chained = MetalBackend::new();
        let backend_split = MetalBackend::new();
        if backend_chained.ctx.is_none() || backend_split.ctx.is_none() {
            return;
        }
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };

        let mut chained_hidden = fixture.hidden.clone();
        let mut chained_states = vec![None, None];
        let chained_layer0 = fixture.layer0_spec();
        let chained_layer1 = fixture.layer1_spec();
        let chained_report = backend_chained.decode_chain_run(
            &mut chained_hidden,
            &[
                ChainLayerSpecRef::GdnMoeQwen(chained_layer0),
                ChainLayerSpecRef::GdnMoeQwen(chained_layer1),
            ],
            &mut chained_states,
            options,
            None,
        );

        let mut split_hidden = fixture.hidden.clone();
        let mut split_states0 = vec![None];
        let split_layer0 = fixture.layer0_spec();
        let split_report0 = backend_split.decode_chain_run(
            &mut split_hidden,
            &[ChainLayerSpecRef::GdnMoeQwen(split_layer0)],
            &mut split_states0,
            options,
            None,
        );
        let mut split_states1 = vec![None];
        let split_layer1 = fixture.layer1_spec();
        let split_report1 = backend_split.decode_chain_run(
            &mut split_hidden,
            &[ChainLayerSpecRef::GdnMoeQwen(split_layer1)],
            &mut split_states1,
            options,
            None,
        );

        assert!(chained_report.did_run);
        assert!(split_report0.did_run);
        assert!(split_report1.did_run);
        assert_eq!(chained_report.qwen_moe_layers, 2);
        assert_eq!(
            split_report0.qwen_moe_layers + split_report1.qwen_moe_layers,
            2
        );
        for (i, (a, b)) in chained_hidden.iter().zip(split_hidden.iter()).enumerate() {
            let abs = (*a - *b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                rel < 1e-3 || abs < 1e-4,
                "hidden mismatch i={i} chain={a} split={b} abs={abs} rel={rel}"
            );
        }
    }

    /// milestone 1 배치 디코드 체인 oracle: `decode_chain_run_batched` 가
    /// (1) batch=1 에서 `decode_chain_run` 과 동일, (2) batch=2 동일 토큰이 lane 별로
    /// 오염 없이 순차 처리, (3) batch=2 구별 토큰이 진짜 시퀀스(token0 → token1)로
    /// 처리되어 각 lane 이 순차 단일-토큰 decode 와 일치함을 검증한다.
    /// 참조는 검증된 `decode_chain_run` 을 conv/delta state threading 하며 lane 수만큼
    /// 순차 호출한 결과다(기존 chain-vs-split oracle 과 동일한 방법론). B token argmax
    /// 추출도 함께 확인한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_matches_sequential_single_token_runs() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 1e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        let probe = MetalBackend::new();
        if probe.ctx.is_none() {
            return;
        }

        let fixture = tests_fixture::qwen_moe_gdn_decode_chain_fixture();
        // delta_resident=false → out_states 가 conv·delta 를 모두 반환해 threading 이
        // 명시적(배치 경로와 참조가 동일한 방식으로 state 전달).
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();
        assert_eq!(
            hidden_dim, 256,
            "fixture hidden_dim must be 256 for the Q4_K output projection"
        );

        // 두 입력 임베딩: emb_a = fixture 기본, emb_b = 구별되는 변형.
        let emb_a = fixture.hidden.clone();
        let emb_b: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();

        // 출력 프로젝션(argmax) fixture: rows=8, cols=hidden_dim, Q4_K(1 superblock/row).
        let out_rows = 8usize;
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * out_rows);
            for _ in 0..out_rows {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: out_rows,
            cols: hidden_dim,
            eps: 1e-6,
        };

        // 단일 GdnMoeQwen layer 순차 decode 참조: 한 backend 에서 conv/delta 를 threading 하며
        // 토큰 수만큼 `decode_chain_run` 을 순차 호출. 반환: (per-token hidden, per-token token id).
        let sequential_reference = |embs: &[Vec<f32>]| -> (Vec<Vec<f32>>, Vec<Option<u32>>) {
            let backend = MetalBackend::new();
            let mut conv = fixture.layer0_spec().conv_state.to_vec();
            let mut delta = fixture.layer0_spec().delta_state.to_vec();
            let mut hiddens = Vec::new();
            let mut tokens = Vec::new();
            for emb in embs {
                let mut spec = fixture.layer0_spec();
                spec.conv_state = conv.as_slice();
                spec.delta_state = delta.as_slice();
                let mut row = emb.clone();
                let mut states = vec![None];
                let rep = backend.decode_chain_run(
                    &mut row,
                    &[ChainLayerSpecRef::GdnMoeQwen(spec)],
                    &mut states,
                    options,
                    Some(argmax_spec()),
                );
                if let Some((c, d)) = states[0].as_ref() {
                    if !c.is_empty() {
                        conv = c.clone();
                    }
                    if !d.is_empty() {
                        delta = d.clone();
                    }
                }
                hiddens.push(row);
                tokens.push(rep.output_argmax.token_id);
            }
            (hiddens, tokens)
        };

        // ── Property 1: batch=1 배치 경로 == decode_chain_run(바이트 동일 기대) ──────────
        {
            let backend_single = MetalBackend::new();
            let mut h_single = emb_a.clone();
            let mut st_single = vec![None];
            let rep_single = backend_single.decode_chain_run(
                &mut h_single,
                &[ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())],
                &mut st_single,
                options,
                Some(argmax_spec()),
            );

            let backend_batched = MetalBackend::new();
            let mut h_batched = emb_a.clone();
            let mut st_batched = vec![None];
            let reps = backend_batched.decode_chain_run_batched(
                &mut h_batched,
                1,
                &[ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())],
                &mut st_batched,
                options,
                Some(argmax_spec()),
            );
            assert_eq!(reps.len(), 1, "batch=1 must return one report");
            assert!(reps[0].did_run);
            assert_close(&h_batched, &h_single, "batch=1 hidden", 1e-5);
            assert_eq!(
                reps[0].output_argmax.token_id, rep_single.output_argmax.token_id,
                "batch=1 token id must match single decode"
            );
        }

        // ── Property 2: batch=2 동일 토큰, 순차. lane0=첫 decode, lane1=두 번째 순차 decode ──
        {
            let embs_same = [emb_a.clone(), emb_a.clone()];
            let (ref_h, ref_tok) = sequential_reference(&embs_same);

            let backend = MetalBackend::new();
            let mut h = Vec::with_capacity(2 * hidden_dim);
            h.extend_from_slice(&emb_a);
            h.extend_from_slice(&emb_a);
            let mut st = vec![None];
            let reps = backend.decode_chain_run_batched(
                &mut h,
                2,
                &[ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())],
                &mut st,
                options,
                Some(argmax_spec()),
            );
            assert_eq!(reps.len(), 2, "batch=2 must return two reports");
            assert_close(
                &h[0..hidden_dim],
                &ref_h[0],
                "batch2-same lane0 hidden",
                1e-4,
            );
            assert_close(
                &h[hidden_dim..2 * hidden_dim],
                &ref_h[1],
                "batch2-same lane1 hidden",
                1e-4,
            );
            assert_eq!(
                reps[0].output_argmax.token_id, ref_tok[0],
                "batch2-same lane0 token"
            );
            assert_eq!(
                reps[1].output_argmax.token_id, ref_tok[1],
                "batch2-same lane1 token"
            );
        }

        // ── Property 3: batch=2 구별 토큰(진짜 시퀀스) — MTP verify 정확성 게이트 ──────────
        {
            let embs_distinct = [emb_a.clone(), emb_b.clone()];
            let (ref_h, ref_tok) = sequential_reference(&embs_distinct);

            let backend = MetalBackend::new();
            let mut h = Vec::with_capacity(2 * hidden_dim);
            h.extend_from_slice(&emb_a);
            h.extend_from_slice(&emb_b);
            let mut st = vec![None];
            let reps = backend.decode_chain_run_batched(
                &mut h,
                2,
                &[ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())],
                &mut st,
                options,
                Some(argmax_spec()),
            );
            assert_eq!(reps.len(), 2);
            assert_close(
                &h[0..hidden_dim],
                &ref_h[0],
                "batch2-distinct lane0 hidden",
                1e-4,
            );
            assert_close(
                &h[hidden_dim..2 * hidden_dim],
                &ref_h[1],
                "batch2-distinct lane1 hidden",
                1e-4,
            );
            assert_eq!(
                reps[0].output_argmax.token_id, ref_tok[0],
                "batch2-distinct lane0 token"
            );
            assert_eq!(
                reps[1].output_argmax.token_id, ref_tok[1],
                "batch2-distinct lane1 token"
            );
        }
    }

    /// milestone 4 배치 디코드 체인 oracle(attn): `decode_chain_run_batched` 의 fused 경로가
    /// AttnMoeQwen layer 를 batch=2 구별 토큰(진짜 시퀀스 pos, pos+1)으로 처리한 결과가
    /// 검증된 순차 단일-토큰 `decode_chain_run`(같은 backend, KV device append 로 threading)과
    /// 일치함을 검증한다. f16 KV backend(kv_int8=false)에서 batched attn fusion 이 활성화된다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_attn_matches_sequential_single_token_runs() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 3e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        let probe = MetalBackend::new_with_kv_int8(false);
        if probe.ctx.is_none() {
            return;
        }
        let fixture = tests_fixture::qwen_moe_attn_decode_chain_fixture();
        let base_pos = tests_fixture::QwenMoeAttnDecodeChainFixture::BASE_POS;
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();
        assert_eq!(hidden_dim, 256, "fixture hidden_dim must be 256");

        let emb_a = fixture.hidden.clone();
        let emb_b: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();

        let out_rows = 8usize;
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * out_rows);
            for _ in 0..out_rows {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: out_rows,
            cols: hidden_dim,
            eps: 1e-6,
        };

        // 순차 참조: ONE backend 를 재사용해 토큰마다 pos=base_pos+i 로 decode_chain_run.
        // KV 는 carrier(layer 0)에 device append 되어 lane→lane 이어진다(MTP verify 시맨틱).
        let sequential_reference = |embs: &[Vec<f32>]| -> (Vec<Vec<f32>>, Vec<Option<u32>>) {
            let backend = MetalBackend::new_with_kv_int8(false);
            let mut hiddens = Vec::new();
            let mut tokens = Vec::new();
            for (i, emb) in embs.iter().enumerate() {
                let spec = fixture.spec(0, base_pos + i);
                let mut row = emb.clone();
                let mut states = vec![None];
                let rep = backend.decode_chain_run(
                    &mut row,
                    &[ChainLayerSpecRef::AttnMoeQwen(spec)],
                    &mut states,
                    options,
                    Some(argmax_spec()),
                );
                hiddens.push(row);
                tokens.push(rep.output_argmax.token_id);
            }
            (hiddens, tokens)
        };

        let embs_distinct = [emb_a.clone(), emb_b.clone()];
        let (ref_h, ref_tok) = sequential_reference(&embs_distinct);

        let backend = MetalBackend::new_with_kv_int8(false);
        let mut h = Vec::with_capacity(2 * hidden_dim);
        h.extend_from_slice(&emb_a);
        h.extend_from_slice(&emb_b);
        let mut st = vec![None];
        let reps = backend.decode_chain_run_batched(
            &mut h,
            2,
            &[ChainLayerSpecRef::AttnMoeQwen(fixture.spec(0, base_pos))],
            &mut st,
            options,
            Some(argmax_spec()),
        );
        assert_eq!(reps.len(), 2);
        assert!(reps[0].did_run && reps[1].did_run);
        assert_close(
            &h[0..hidden_dim],
            &ref_h[0],
            "attn batch2 lane0 hidden",
            2e-3,
        );
        assert_close(
            &h[hidden_dim..2 * hidden_dim],
            &ref_h[1],
            "attn batch2 lane1 hidden",
            2e-3,
        );
        assert_eq!(
            reps[0].output_argmax.token_id, ref_tok[0],
            "attn batch2 lane0 token"
        );
        assert_eq!(
            reps[1].output_argmax.token_id, ref_tok[1],
            "attn batch2 lane1 token"
        );
    }

    /// milestone 4 배치 디코드 체인 oracle(mixed): AttnMoeQwen + GdnMoeQwen 를 이어붙인
    /// 혼합 chain(실제 unsloth 구조)이 batch=2 구별 토큰에서 fused single-command-buffer 경로로
    /// 처리한 결과가 순차 단일-토큰 참조(attn KV device threading + GDN conv/delta host threading)와
    /// 일치함을 검증한다. attn·GDN core 가 모두 한 command buffer 안에서 B-position fuse 된다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_mixed_attn_gdn_matches_sequential_single_token_runs() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 3e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        let probe = MetalBackend::new_with_kv_int8(false);
        if probe.ctx.is_none() {
            return;
        }
        let attn_fx = tests_fixture::qwen_moe_attn_decode_chain_fixture();
        let gdn_fx = tests_fixture::qwen_moe_gdn_decode_chain_fixture();
        let base_pos = tests_fixture::QwenMoeAttnDecodeChainFixture::BASE_POS;
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = attn_fx.hidden.len();
        assert_eq!(
            hidden_dim,
            gdn_fx.hidden.len(),
            "fixtures must share hidden_dim"
        );

        let emb_a = attn_fx.hidden.clone();
        let emb_b: Vec<f32> = attn_fx
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();

        let out_rows = 8usize;
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * out_rows);
            for _ in 0..out_rows {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: out_rows,
            cols: hidden_dim,
            eps: 1e-6,
        };
        // 순차 참조: ONE backend. attn KV 는 carrier device append, GDN conv/delta 는 out_states.
        let sequential_reference = |embs: &[Vec<f32>]| -> (Vec<Vec<f32>>, Vec<Option<u32>>) {
            let backend = MetalBackend::new_with_kv_int8(false);
            let mut conv = gdn_fx.layer1_spec().conv_state.to_vec();
            let mut delta = gdn_fx.layer1_spec().delta_state.to_vec();
            let mut hiddens = Vec::new();
            let mut tokens = Vec::new();
            for (i, emb) in embs.iter().enumerate() {
                let mut g = gdn_fx.layer1_spec();
                g.conv_state = &conv;
                g.delta_state = &delta;
                let a_spec = ChainLayerSpecRef::AttnMoeQwen(attn_fx.spec(0, base_pos + i));
                let g_spec = ChainLayerSpecRef::GdnMoeQwen(g);
                let mut row = emb.clone();
                let mut states = vec![None, None];
                let rep = backend.decode_chain_run(
                    &mut row,
                    &[a_spec, g_spec],
                    &mut states,
                    options,
                    Some(argmax_spec()),
                );
                if let Some((c, d)) = states[1].as_ref() {
                    if !c.is_empty() {
                        conv = c.clone();
                    }
                    if !d.is_empty() {
                        delta = d.clone();
                    }
                }
                hiddens.push(row);
                tokens.push(rep.output_argmax.token_id);
            }
            (hiddens, tokens)
        };

        let embs_distinct = [emb_a.clone(), emb_b.clone()];
        let (ref_h, ref_tok) = sequential_reference(&embs_distinct);

        let backend = MetalBackend::new_with_kv_int8(false);
        let conv0 = gdn_fx.layer1_spec().conv_state.to_vec();
        let delta0 = gdn_fx.layer1_spec().delta_state.to_vec();
        let mut g0 = gdn_fx.layer1_spec();
        g0.conv_state = &conv0;
        g0.delta_state = &delta0;
        let a_spec = ChainLayerSpecRef::AttnMoeQwen(attn_fx.spec(0, base_pos));
        let g_spec = ChainLayerSpecRef::GdnMoeQwen(g0);
        let mut h = Vec::with_capacity(2 * hidden_dim);
        h.extend_from_slice(&emb_a);
        h.extend_from_slice(&emb_b);
        let mut st = vec![None, None];
        let reps = backend.decode_chain_run_batched(
            &mut h,
            2,
            &[a_spec, g_spec],
            &mut st,
            options,
            Some(argmax_spec()),
        );
        assert_eq!(reps.len(), 2);
        assert!(reps[0].did_run && reps[1].did_run);
        assert_eq!(
            reps[0].qwen_moe_layers, 2,
            "mixed chain must fuse 2 MoE layers"
        );
        assert_close(
            &h[0..hidden_dim],
            &ref_h[0],
            "mixed batch2 lane0 hidden",
            2e-3,
        );
        assert_close(
            &h[hidden_dim..2 * hidden_dim],
            &ref_h[1],
            "mixed batch2 lane1 hidden",
            2e-3,
        );
        assert_eq!(
            reps[0].output_argmax.token_id, ref_tok[0],
            "mixed batch2 lane0 token"
        );
        assert_eq!(
            reps[1].output_argmax.token_id, ref_tok[1],
            "mixed batch2 lane1 token"
        );
    }

    /// milestone 4 MTP commit-contract oracle: `decode_chain_run_batched_collect_attn_kv` 가
    /// 반환한 window post-rope f16 K/V 를 prior 에 이어붙여 다음 토큰(pos=base+2)을 계산하면
    /// 순차 단일-토큰 참조의 3번째 토큰과 일치함을 검증한다 — 즉 배치가 반환한 커밋 K/V 가
    /// device append 된 실제 window 와 정확히 같음(엔진의 host kv_cache append 경로가 올바르게
    /// prior 를 재구성함)을 보장한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_attn_committed_kv_matches_sequential() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 3e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        let probe = MetalBackend::new_with_kv_int8(false);
        if probe.ctx.is_none() {
            return;
        }
        let fixture = tests_fixture::qwen_moe_attn_decode_chain_fixture();
        let base_pos = tests_fixture::QwenMoeAttnDecodeChainFixture::BASE_POS;
        let kv_dim = tests_fixture::QwenMoeAttnDecodeChainFixture::KV_DIM;
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();

        let emb_a = fixture.hidden.clone();
        let emb_b: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();
        let emb_c: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.25 + ((i % 7) as f32 - 3.0) * 0.0011)
            .collect();

        let out_rows = 8usize;
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * out_rows);
            for _ in 0..out_rows {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: out_rows,
            cols: hidden_dim,
            eps: 1e-6,
        };

        // 순차 참조(ONE backend, KV device-threaded): 토큰 0,1,2 → 3번째 hidden/token 이 ground truth.
        let backend_ref = MetalBackend::new_with_kv_int8(false);
        let mut ref_h2 = Vec::new();
        let mut ref_tok2 = None;
        for (i, emb) in [&emb_a, &emb_b, &emb_c].iter().enumerate() {
            let spec = fixture.spec(0, base_pos + i);
            let mut row = (*emb).clone();
            let mut states = vec![None];
            let rep = backend_ref.decode_chain_run(
                &mut row,
                &[ChainLayerSpecRef::AttnMoeQwen(spec)],
                &mut states,
                options,
                Some(argmax_spec()),
            );
            if i == 2 {
                ref_h2 = row;
                ref_tok2 = rep.output_argmax.token_id;
            }
        }

        // 배치 verify(batch=2, tokens 0,1) + collect: layer0 attn window K/V(2 slot) 반환.
        let backend = MetalBackend::new_with_kv_int8(false);
        let mut h = Vec::with_capacity(2 * hidden_dim);
        h.extend_from_slice(&emb_a);
        h.extend_from_slice(&emb_b);
        let mut st = vec![None];
        let mut attn_kv: Vec<Option<(Vec<u16>, Vec<u16>)>> = Vec::new();
        let _ = backend.decode_chain_run_batched_collect_attn_kv(
            &mut h,
            2,
            &[ChainLayerSpecRef::AttnMoeQwen(fixture.spec(0, base_pos))],
            &mut st,
            options,
            Some(argmax_spec()),
            &mut attn_kv,
        );
        assert_eq!(attn_kv.len(), 1, "one layer");
        let (win_k, win_v) = attn_kv[0]
            .as_ref()
            .expect("attn layer must return committed KV");
        assert_eq!(win_k.len(), 2 * kv_dim, "committed K = batch*kv_dim");
        assert_eq!(win_v.len(), 2 * kv_dim, "committed V = batch*kv_dim");

        // orig prior[0..base] ++ committed window[base..base+2] = prior'[0..base+2].
        let base_k = &fixture.spec(0, base_pos).prior_k[..base_pos * kv_dim];
        let base_v = &fixture.spec(0, base_pos).prior_v[..base_pos * kv_dim];
        let mut prior_k = base_k.to_vec();
        prior_k.extend_from_slice(win_k);
        let mut prior_v = base_v.to_vec();
        prior_v.extend_from_slice(win_v);

        // prior' 로 3번째 토큰(emb_c, pos=base+2)을 fresh backend 에서 계산 → 순차 참조와 일치?
        let backend_cont = MetalBackend::new_with_kv_int8(false);
        let spec_cont = fixture.spec_with_prior(0, base_pos + 2, &prior_k, &prior_v);
        let mut row = emb_c.clone();
        let mut states = vec![None];
        let rep = backend_cont.decode_chain_run(
            &mut row,
            &[ChainLayerSpecRef::AttnMoeQwen(spec_cont)],
            &mut states,
            options,
            Some(argmax_spec()),
        );
        assert_close(&row, &ref_h2, "committed-kv continuation hidden", 2e-3);
        assert_eq!(
            rep.output_argmax.token_id, ref_tok2,
            "committed-kv continuation token"
        );
    }

    /// milestone 4 MTP full-reject 엣지 oracle: collect 모드 batch==1(n_accepted=0 → committed=1
    /// 재실행)이 (a) fused 배치 carrier 경로로 처리되고 (b) out_attn_kv 에 1 slot K/V 를 채우고
    /// (c) 프로덕션 stateful carrier(attn_moe_carriers/gdn_carriers)를 절대 건드리지 않으며
    /// (d) 결과가 검증된 single-token decode_chain_run 과 일치함을 확인한다.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_collect_batch1_untouches_production_carriers() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 3e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        let probe = MetalBackend::new_with_kv_int8(false);
        if probe.ctx.is_none() {
            return;
        }
        let fixture = tests_fixture::qwen_moe_attn_decode_chain_fixture();
        let base_pos = tests_fixture::QwenMoeAttnDecodeChainFixture::BASE_POS;
        let kv_dim = tests_fixture::QwenMoeAttnDecodeChainFixture::KV_DIM;
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * 8);
            for _ in 0..8 {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: 8,
            cols: hidden_dim,
            eps: 1e-6,
        };

        // 참조: single-token decode_chain_run(pos=base_pos).
        let backend_ref = MetalBackend::new_with_kv_int8(false);
        let mut ref_row = fixture.hidden.clone();
        let mut ref_st = vec![None];
        let ref_rep = backend_ref.decode_chain_run(
            &mut ref_row,
            &[ChainLayerSpecRef::AttnMoeQwen(fixture.spec(0, base_pos))],
            &mut ref_st,
            options,
            Some(argmax_spec()),
        );

        // collect batch==1: fused 배치 carrier 경로.
        let backend = MetalBackend::new_with_kv_int8(false);
        let mut h = fixture.hidden.clone();
        let mut st = vec![None];
        let mut attn_kv: Vec<Option<(Vec<u16>, Vec<u16>)>> = Vec::new();
        let reps = backend.decode_chain_run_batched_collect_attn_kv(
            &mut h,
            1,
            &[ChainLayerSpecRef::AttnMoeQwen(fixture.spec(0, base_pos))],
            &mut st,
            options,
            Some(argmax_spec()),
            &mut attn_kv,
        );
        assert_eq!(reps.len(), 1);
        assert!(reps[0].did_run);
        // (b) out_attn_kv 1 slot.
        let (k, v) = attn_kv[0]
            .as_ref()
            .expect("batch1 collect must return committed KV");
        assert_eq!(k.len(), kv_dim, "batch1 committed K = 1*kv_dim");
        assert_eq!(v.len(), kv_dim, "batch1 committed V = 1*kv_dim");
        // (c) 프로덕션 stateful carrier 미접촉.
        assert!(
            backend.attn_moe_carriers.borrow().is_empty(),
            "batch1 collect must NOT touch production attn_moe_carriers"
        );
        assert!(
            backend.gdn_carriers.borrow().is_empty(),
            "batch1 collect must NOT touch production gdn_carriers"
        );
        assert!(
            !backend.attn_batch_carriers.borrow().is_empty(),
            "batch1 collect must use batch attn carrier"
        );
        // (d) 결과가 single-token decode 와 일치.
        assert_close(&h, &ref_row, "batch1 collect hidden", 2e-3);
        assert_eq!(
            reps[0].output_argmax.token_id, ref_rep.output_argmax.token_id,
            "batch1 collect token"
        );
    }

    /// milestone 4 게이트 완화 oracle: batched attn fusion 이 **int8 ctx**(kv_int8=true)에서도
    /// f16 순차 참조와 일치함을 검증한다 — batched attn 은 자체 f16 KvResident + f16 커널이라
    /// ctx.kv_int8 과 무관하게 정확(게이트에서 ctx_f16 조건 제거의 근거). fused 가 실제로 돌았음은
    /// 프로덕션 attn_moe_carriers 미접촉으로 확인(per-lane 폴백이면 채워짐).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_attn_int8_ctx_matches_f16_sequential() {
        fn assert_close(a: &[f32], b: &[f32], label: &str, tol_abs: f32) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(
                    rel < 3e-3 || abs < tol_abs,
                    "{label} mismatch i={i} a={x} b={y} abs={abs} rel={rel}"
                );
            }
        }

        // int8 ctx 를 만들 수 있어야 하고(디바이스 존재), f16 ctx 참조도 필요.
        let probe = MetalBackend::new_with_kv_int8(true);
        if probe.ctx.is_none() {
            return;
        }
        let fixture = tests_fixture::qwen_moe_attn_decode_chain_fixture();
        let base_pos = tests_fixture::QwenMoeAttnDecodeChainFixture::BASE_POS;
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();
        let emb_a = fixture.hidden.clone();
        let emb_b: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();
        let out_norm = vec![1.0f32; hidden_dim];
        let out_raw = {
            let block = tests_fixture::q4k_block_fixed();
            let mut v = Vec::with_capacity(block.len() * 8);
            for _ in 0..8 {
                v.extend_from_slice(&block);
            }
            v
        };
        let argmax_spec = || DecodeOutputArgmaxSpecRef {
            norm_weight: &out_norm,
            output_raw: &out_raw,
            output_quant: 0,
            rows: 8,
            cols: hidden_dim,
            eps: 1e-6,
        };

        // f16 ctx 순차 참조(tokens 0,1).
        let backend_ref = MetalBackend::new_with_kv_int8(false);
        let mut ref_h = Vec::new();
        let mut ref_tok = Vec::new();
        for (i, emb) in [&emb_a, &emb_b].iter().enumerate() {
            let mut row = (*emb).clone();
            let mut states = vec![None];
            let rep = backend_ref.decode_chain_run(
                &mut row,
                &[ChainLayerSpecRef::AttnMoeQwen(
                    fixture.spec(0, base_pos + i),
                )],
                &mut states,
                options,
                Some(argmax_spec()),
            );
            ref_h.push(row);
            ref_tok.push(rep.output_argmax.token_id);
        }

        // int8 ctx 배치 fused(batch=2).
        let backend = MetalBackend::new_with_kv_int8(true);
        let mut h = Vec::with_capacity(2 * hidden_dim);
        h.extend_from_slice(&emb_a);
        h.extend_from_slice(&emb_b);
        let mut st = vec![None];
        let reps = backend.decode_chain_run_batched(
            &mut h,
            2,
            &[ChainLayerSpecRef::AttnMoeQwen(fixture.spec(0, base_pos))],
            &mut st,
            options,
            Some(argmax_spec()),
        );
        assert_eq!(reps.len(), 2);
        // fused 가 돌았음(per-lane 폴백 아님) → 프로덕션 attn_moe_carriers 미접촉.
        assert!(
            backend.attn_moe_carriers.borrow().is_empty(),
            "int8 ctx batched attn must fuse (not per-lane), leaving production carriers untouched"
        );
        assert_close(
            &h[0..hidden_dim],
            &ref_h[0],
            "int8-ctx fused attn lane0",
            2e-3,
        );
        assert_close(
            &h[hidden_dim..2 * hidden_dim],
            &ref_h[1],
            "int8-ctx fused attn lane1",
            2e-3,
        );
        assert_eq!(
            reps[0].output_argmax.token_id, ref_tok[0],
            "int8-ctx lane0 token"
        );
        assert_eq!(
            reps[1].output_argmax.token_id, ref_tok[1],
            "int8-ctx lane1 token"
        );
    }

    /// milestone 4 bottleneck2 oracle: batched GDN ping-pong 이 보존한 prefix-n state 를
    /// `decode_chain_run_batched_gdn_prefix` 로 재실행 없이 뽑은 값이, 순차 단일-토큰 참조의
    /// n-토큰 후 conv/delta state 와 일치함을 검증한다(partial-accept 커밋 재실행 제거의 근거).
    /// prefix-2 == forward out_states(final), prefix-1 == token0 후 state.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_chain_batched_gdn_prefix_matches_sequential() {
        fn assert_close(a: &[f32], b: &[f32], label: &str) {
            assert_eq!(a.len(), b.len(), "{label}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let abs = (*x - *y).abs();
                let rel = abs / y.abs().max(1e-6);
                assert!(rel < 3e-3 || abs < 1e-4, "{label} i={i} a={x} b={y}");
            }
        }

        let probe = MetalBackend::new();
        if probe.ctx.is_none() {
            return;
        }
        let fixture = tests_fixture::qwen_moe_gdn_decode_chain_fixture();
        let options = DecodeChainOptions {
            collect_timing: false,
            delta_resident: false,
        };
        let hidden_dim = fixture.hidden.len();
        let emb_a = fixture.hidden.clone();
        let emb_b: Vec<f32> = fixture
            .hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.5 + ((i % 5) as f32 - 2.0) * 0.0007)
            .collect();

        // 순차 참조: token0 → prefix-1 state, token1(threaded) → prefix-2 state.
        let backend_ref = MetalBackend::new();
        let mut row = emb_a.clone();
        let mut st0 = vec![None];
        backend_ref.decode_chain_run(
            &mut row,
            &[ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())],
            &mut st0,
            options,
            None,
        );
        let (conv1, delta1) = st0[0].clone().expect("prefix-1 state");
        let mut spec1 = fixture.layer0_spec();
        spec1.conv_state = &conv1;
        spec1.delta_state = &delta1;
        let mut row1 = emb_b.clone();
        let mut st1 = vec![None];
        backend_ref.decode_chain_run(
            &mut row1,
            &[ChainLayerSpecRef::GdnMoeQwen(spec1)],
            &mut st1,
            options,
            None,
        );
        let (conv2, delta2) = st1[0].clone().expect("prefix-2 state");

        // batched batch=2 → fused ping-pong. out_states = prefix-2.
        let backend = MetalBackend::new();
        let mut h = Vec::with_capacity(2 * hidden_dim);
        h.extend_from_slice(&emb_a);
        h.extend_from_slice(&emb_b);
        let mut st = vec![None];
        let specs = [ChainLayerSpecRef::GdnMoeQwen(fixture.layer0_spec())];
        backend.decode_chain_run_batched(&mut h, 2, &specs, &mut st, options, None);
        let (bconv2, bdelta2) = st[0].clone().expect("batched final state");
        assert_close(&bconv2, &conv2, "batched final conv == seq prefix-2");
        assert_close(&bdelta2, &delta2, "batched final delta == seq prefix-2");

        // prefix readback: n=1 == seq prefix-1, n=2 == seq prefix-2.
        let p1 = backend.decode_chain_run_batched_gdn_prefix(&specs, 2, 1);
        let (pc1, pd1) = p1[0].clone().expect("gdn prefix-1");
        assert_close(&pc1, &conv1, "gdn_prefix(1) conv == seq prefix-1");
        assert_close(&pd1, &delta1, "gdn_prefix(1) delta == seq prefix-1");
        let p2 = backend.decode_chain_run_batched_gdn_prefix(&specs, 2, 2);
        let (pc2, pd2) = p2[0].clone().expect("gdn prefix-2");
        assert_close(&pc2, &conv2, "gdn_prefix(2) conv == seq prefix-2");
        assert_close(&pd2, &delta2, "gdn_prefix(2) delta == seq prefix-2");
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn kvarn_attention_decode_matches_cpu_reference() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            return;
        }
        let head_dim = 128usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let config = rnb_cpu::quantize::kvarn::KvarnConfig::K4_V4_G64;
        let sink_len = config.sink_tokens;
        let tail_len = 5usize;
        let kv_len = sink_len + config.group + tail_len;
        let row_width = num_kv_heads * head_dim;
        let mut key = vec![0u16; kv_len * row_width];
        let mut value = vec![0u16; kv_len * row_width];
        for (index, bits) in key.iter_mut().enumerate() {
            let sample = ((index * 17 % 101) as f32 - 50.0) * 0.00625;
            *bits = half::f16::from_f32(sample).to_bits();
        }
        for (index, bits) in value.iter_mut().enumerate() {
            let sample = ((index * 29 % 113) as f32 - 56.0) * 0.0078125;
            *bits = half::f16::from_f32(sample).to_bits();
        }
        let block_start = sink_len * row_width;
        let block_end = block_start + config.group * row_width;
        let block = rnb_cpu::quantize::kvarn::KvarnBlock::quantize(
            config,
            num_kv_heads,
            head_dim,
            &key[block_start..block_end],
            &value[block_start..block_end],
        )
        .expect("quantize KVarN block");
        let layout =
            rnb_cpu::quantize::kvarn::KvarnDeviceRecordLayout::new(config, num_kv_heads, head_dim)
                .expect("KVarN device layout");
        let mut packed = Vec::new();
        block.append_device_record(&mut packed);
        let blocks = [block];
        let tail_start = sink_len + config.group;
        let view = rnb_cpu::quantize::kvarn::KvarnKvView {
            config,
            num_kv_heads,
            head_dim,
            sink_key: &key[..block_start],
            sink_value: &value[..block_start],
            blocks: &blocks,
            device_layout: layout,
            device_blocks: &packed,
            tail_start,
            tail_key: &key[block_end..],
            tail_value: &value[block_end..],
            len: kv_len,
        };
        let query = (0..num_heads * head_dim)
            .map(|index| ((index * 13 % 79) as f32 - 39.0) * 0.009)
            .collect::<Vec<_>>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut expected = vec![0.0f32; query.len()];
        rnb_cpu::quantize::kvarn::attention_decode(
            &query,
            view,
            &mut expected,
            num_heads,
            scale,
            None,
            None,
        );
        let request = rnb_backend_api::KvarnDecodeRequest::new(
            10_902,
            &query,
            &packed,
            view.sink_key,
            view.sink_value,
            view.tail_key,
            view.tail_value,
            kv_len,
            tail_start,
            num_heads,
            num_kv_heads,
            head_dim,
            config.key_bits,
            config.value_bits,
            config.group,
            config.sink_tokens,
            layout.block_bytes,
            scale,
            None,
            None,
        );
        let actual = backend
            .kvarn_attention_decode(request)
            .expect("Metal KVarN attention");
        let max_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2.0e-3,
            "Metal KVarN attention diverged from CPU reference: max_diff={max_diff}"
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn kvarn_attention_decode_appended_matches_cpu_reference() {
        // decode chain in-chain 경로: prior(kv_len-1) view 를 resident 로 올린 뒤 새 토큰을
        // device tail-append + in-chain kvarn attention 으로 계산한 결과가, full(kv_len) view
        // CPU reference 와 일치하는지 검증한다(append slot/params +1/tail 오프셋 격리).
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            return;
        };
        let head_dim = 128usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let config = rnb_cpu::quantize::kvarn::KvarnConfig::K4_V4_G64;
        let sink_len = config.sink_tokens;
        let tail_len = 5usize;
        let kv_len = sink_len + config.group + tail_len;
        let row_width = num_kv_heads * head_dim;
        let mut key = vec![0u16; kv_len * row_width];
        let mut value = vec![0u16; kv_len * row_width];
        for (index, bits) in key.iter_mut().enumerate() {
            let sample = ((index * 17 % 101) as f32 - 50.0) * 0.00625;
            *bits = half::f16::from_f32(sample).to_bits();
        }
        for (index, bits) in value.iter_mut().enumerate() {
            let sample = ((index * 29 % 113) as f32 - 56.0) * 0.0078125;
            *bits = half::f16::from_f32(sample).to_bits();
        }
        let block_start = sink_len * row_width;
        let block_end = block_start + config.group * row_width;
        let block = rnb_cpu::quantize::kvarn::KvarnBlock::quantize(
            config,
            num_kv_heads,
            head_dim,
            &key[block_start..block_end],
            &value[block_start..block_end],
        )
        .expect("quantize KVarN block");
        let layout =
            rnb_cpu::quantize::kvarn::KvarnDeviceRecordLayout::new(config, num_kv_heads, head_dim)
                .expect("KVarN device layout");
        let mut packed = Vec::new();
        block.append_device_record(&mut packed);
        let blocks = [block];
        let tail_start = sink_len + config.group;
        let query = (0..num_heads * head_dim)
            .map(|index| ((index * 13 % 79) as f32 - 39.0) * 0.009)
            .collect::<Vec<_>>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        // reference: full(kv_len) view via CPU.
        let full_view = rnb_cpu::quantize::kvarn::KvarnKvView {
            config,
            num_kv_heads,
            head_dim,
            sink_key: &key[..block_start],
            sink_value: &value[..block_start],
            blocks: &blocks,
            device_layout: layout,
            device_blocks: &packed,
            tail_start,
            tail_key: &key[block_end..],
            tail_value: &value[block_end..],
            len: kv_len,
        };
        let mut expected = vec![0.0f32; query.len()];
        rnb_cpu::quantize::kvarn::attention_decode(
            &query,
            full_view,
            &mut expected,
            num_heads,
            scale,
            None,
            None,
        );
        // prior(kv_len-1) view: 마지막 tail 토큰 1개 제외.
        let prior_tail_tokens = tail_len - 1;
        let prior_tail_end = block_end + prior_tail_tokens * row_width;
        let prior_request = rnb_backend_api::KvarnDecodeRequest::new(
            10_903,
            &query,
            &packed,
            &key[..block_start],
            &value[..block_start],
            &key[block_end..prior_tail_end],
            &value[block_end..prior_tail_end],
            kv_len - 1,
            tail_start,
            num_heads,
            num_kv_heads,
            head_dim,
            config.key_bits,
            config.value_bits,
            config.group,
            config.sink_tokens,
            layout.block_bytes,
            scale,
            None,
            None,
        );
        let new_k: Vec<f32> = key[prior_tail_end..]
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        let new_v: Vec<f32> = value[prior_tail_end..]
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        let mut resident =
            crate::compute::KvarnResident::new(ctx, prior_request).expect("KVarN resident");
        let actual = crate::compute::kvarn_attention_decode_appended_with_ctx(
            ctx,
            &mut resident,
            prior_request,
            &new_k,
            &new_v,
        )
        .expect("appended KVarN attention");
        let max_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 3.0e-3,
            "appended KVarN attention diverged from CPU reference: max_diff={max_diff}"
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn kvarn_attention_decode_splitk_matches_cpu_reference() {
        // split-K: block 을 num_splits 로 쪼갠 뒤 combine + un-Hadamard + sink/tail merge 가
        // full(kv_len) CPU reference 와 일치하는지(multi-block = multi-split combine 격리).
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            return;
        };
        let head_dim = 128usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let config = rnb_cpu::quantize::kvarn::KvarnConfig::K4_V4_G64;
        let sink_len = config.sink_tokens;
        let num_blocks = 3usize;
        let tail_len = 5usize;
        let kv_len = sink_len + num_blocks * config.group + tail_len;
        let row_width = num_kv_heads * head_dim;
        let mut key = vec![0u16; kv_len * row_width];
        let mut value = vec![0u16; kv_len * row_width];
        for (index, bits) in key.iter_mut().enumerate() {
            *bits = half::f16::from_f32(((index * 17 % 101) as f32 - 50.0) * 0.00625).to_bits();
        }
        for (index, bits) in value.iter_mut().enumerate() {
            *bits = half::f16::from_f32(((index * 29 % 113) as f32 - 56.0) * 0.0078125).to_bits();
        }
        let block_start = sink_len * row_width;
        let layout =
            rnb_cpu::quantize::kvarn::KvarnDeviceRecordLayout::new(config, num_kv_heads, head_dim)
                .expect("KVarN device layout");
        let mut packed = Vec::new();
        let mut blocks = Vec::new();
        for b in 0..num_blocks {
            let s = block_start + b * config.group * row_width;
            let e = s + config.group * row_width;
            let block = rnb_cpu::quantize::kvarn::KvarnBlock::quantize(
                config,
                num_kv_heads,
                head_dim,
                &key[s..e],
                &value[s..e],
            )
            .expect("quantize KVarN block");
            block.append_device_record(&mut packed);
            blocks.push(block);
        }
        let tail_off = block_start + num_blocks * config.group * row_width;
        let tail_start = sink_len + num_blocks * config.group;
        let query = (0..num_heads * head_dim)
            .map(|index| ((index * 13 % 79) as f32 - 39.0) * 0.009)
            .collect::<Vec<_>>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let full_view = rnb_cpu::quantize::kvarn::KvarnKvView {
            config,
            num_kv_heads,
            head_dim,
            sink_key: &key[..block_start],
            sink_value: &value[..block_start],
            blocks: &blocks,
            device_layout: layout,
            device_blocks: &packed,
            tail_start,
            tail_key: &key[tail_off..],
            tail_value: &value[tail_off..],
            len: kv_len,
        };
        let mut expected = vec![0.0f32; query.len()];
        rnb_cpu::quantize::kvarn::attention_decode(
            &query,
            full_view,
            &mut expected,
            num_heads,
            scale,
            None,
            None,
        );
        let request = rnb_backend_api::KvarnDecodeRequest::new(
            10_904,
            &query,
            &packed,
            &key[..block_start],
            &value[..block_start],
            &key[tail_off..],
            &value[tail_off..],
            kv_len,
            tail_start,
            num_heads,
            num_kv_heads,
            head_dim,
            config.key_bits,
            config.value_bits,
            config.group,
            config.sink_tokens,
            layout.block_bytes,
            scale,
            None,
            None,
        );
        let mut resident =
            crate::compute::KvarnResident::new(ctx, request).expect("KVarN resident");
        let actual =
            crate::compute::kvarn_attention_decode_splitk_with_ctx(ctx, &mut resident, request)
                .expect("splitk KVarN attention");
        let max_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 3.0e-3,
            "splitk KVarN attention diverged from CPU reference: max_diff={max_diff}"
        );
    }
}
