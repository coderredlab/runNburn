pub mod request_queue;

use rnb_backend_api::{BackendCapabilities, BackendKind, BackendOp};
use rnb_memory::{MemoryBudget, MemoryTier};
use rnb_model_ir::{LayerId, LayerKind, ModelGraph};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    UnsupportedLayerOp { layer: LayerId, op: BackendOp },
    MissingMemoryBudget { layer: LayerId, tier: MemoryTier },
}

pub type ScheduleResult<T> = Result<T, ScheduleError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    layer: LayerId,
    op: BackendOp,
    backend: BackendKind,
    memory_tier: MemoryTier,
}

impl Placement {
    pub const fn new(
        layer: LayerId,
        op: BackendOp,
        backend: BackendKind,
        memory_tier: MemoryTier,
    ) -> Self {
        Self {
            layer,
            op,
            backend,
            memory_tier,
        }
    }

    pub const fn layer(self) -> LayerId {
        self.layer
    }

    pub const fn op(self) -> BackendOp {
        self.op
    }

    pub const fn backend(self) -> BackendKind {
        self.backend
    }

    pub const fn memory_tier(self) -> MemoryTier {
        self.memory_tier
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SchedulePlan {
    placements: Vec<Placement>,
}

impl SchedulePlan {
    pub fn new() -> Self {
        Self {
            placements: Vec::new(),
        }
    }

    pub fn push(&mut self, placement: Placement) {
        self.placements.push(placement);
    }

    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }
}

pub fn schedule_model_graph(
    graph: &ModelGraph,
    capabilities: &[BackendCapabilities],
    memory_budgets: &[MemoryBudget],
) -> ScheduleResult<SchedulePlan> {
    let mut plan = SchedulePlan::new();
    for layer in graph.layers() {
        let op = op_for_layer_kind(layer.kind);
        let Some(backend) = capabilities
            .iter()
            .find(|capabilities| capabilities.supports(op))
            .map(BackendCapabilities::backend)
        else {
            return Err(ScheduleError::UnsupportedLayerOp {
                layer: layer.id,
                op,
            });
        };
        let tier = memory_tier_for_backend(backend);
        if !memory_budgets
            .iter()
            .any(|budget| budget.tier() == tier && budget.available_bytes() > 0)
        {
            return Err(ScheduleError::MissingMemoryBudget {
                layer: layer.id,
                tier,
            });
        }
        plan.push(Placement::new(layer.id, op, backend, tier));
    }
    Ok(plan)
}

pub const fn op_for_layer_kind(kind: LayerKind) -> BackendOp {
    match kind {
        LayerKind::Attention => BackendOp::Attention,
        LayerKind::FeedForward => BackendOp::MatMul,
        LayerKind::Gdn => BackendOp::Gdn,
        LayerKind::Moe => BackendOp::MoE,
    }
}

pub const fn memory_tier_for_backend(backend: BackendKind) -> MemoryTier {
    match backend {
        BackendKind::Cpu => MemoryTier::Ram,
        BackendKind::Cuda | BackendKind::Vulkan | BackendKind::OpenCl | BackendKind::Metal => {
            MemoryTier::Vram
        }
        BackendKind::MediaTekNpu => MemoryTier::Npu,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice1BoundaryPlan {
    pub cpu_prefix_layer_range: Range<usize>,
    pub window_layer_range: Range<usize>,
    pub attention_layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillExecutionPath {
    Cpu,
    Slice1GpuCandidate,
    /// Full GPU offload prefill path (mv27 task 10b-4).
    /// Selected when `RNB_GPU_FULLPATH=1` and the same eligibility
    /// preconditions as `Slice1GpuCandidate` hold.
    Fullpath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeExecutionProfile {
    CpuOnly,
    MobileVulkanPrefillCpuDecode,
    DesktopVulkanGpuResident,
    CudaGpuResident,
    ExperimentalFullpath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionProfileRequest {
    pub is_android_target: bool,
    pub is_mobile_target: bool,
    pub is_desktop_target: bool,
    pub cpu_available: bool,
    pub vulkan_available: bool,
    pub cuda_available: bool,
    pub fullpath_requested: bool,
    /// mv40 (2026-05-07): mobile 에선 vulkan partial path 도 default OFF.
    /// `RNB_FORCE_MOBILE_VULKAN=1` 명시 시에만 활성. caller 가 env parse.
    pub force_mobile_vulkan_requested: bool,
    pub requested_profile: Option<RuntimeExecutionProfile>,
}

impl ExecutionProfileRequest {
    pub const fn cpu_only() -> Self {
        Self {
            is_android_target: false,
            is_mobile_target: false,
            is_desktop_target: false,
            cpu_available: true,
            vulkan_available: false,
            cuda_available: false,
            fullpath_requested: false,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        }
    }
}

pub fn select_runtime_execution_profile(
    request: ExecutionProfileRequest,
) -> RuntimeExecutionProfile {
    if let Some(profile) = request.requested_profile {
        return profile;
    }

    // mv31 (2026-05-06) + mv40 (2026-05-07) — Vulkan execution policy.
    //
    // Mobile-class GPU (Adreno / Mali) 는 ARM CPU와 fp ops 비트 일치를
    // 보장 못 한다 (mv31). 그리고 mv39/mv40 측정 + 학계 (arxiv 2505.06461,
    // arxiv 2410.03613) 에서 mobile GPU 의 LLM inference 가 ARM CPU NEON 보다
    // 느림이 확정 — Mali ALU utilization 평균 <3%, Qwen3.5 0.8B 411 prompt
    // attention 1.92x 느림. partial path 도 default 로 비활성한다. 코드는
    // NPU/FlashAttention 등 미래 axis 위해 유지하되 opt-in env
    // `RNB_FORCE_MOBILE_VULKAN=1` 일 때만 진입 가능.
    //
    // Desktop GPU (NVIDIA/AMD/Intel) 는 Vulkan fp ops 정밀도가 mobile
    // 보다 robust 하다. CUDA가 있으면 CUDA 우선. CUDA 없는 desktop +
    // Vulkan 조합에서는 fullpath default ON.
    let is_mobile_class = request.is_android_target || request.is_mobile_target;

    if is_mobile_class {
        if request.fullpath_requested {
            eprintln!(
                "[scheduler] WARN: RNB_GPU_FULLPATH=1 ignored on mobile target — \
                 mv31 policy: Adreno/Mali fp ops are not bit-equal to ARM CPU. \
                 mv40 policy: mobile vulkan partial path 도 default OFF — \
                 measurements + literature show mobile GPU is slower than \
                 ARM CPU NEON for LLM inference. Falling back to CpuOnly."
            );
        }
        if request.force_mobile_vulkan_requested
            && request.cpu_available
            && request.vulkan_available
        {
            return RuntimeExecutionProfile::MobileVulkanPrefillCpuDecode;
        }
        return RuntimeExecutionProfile::CpuOnly;
    }

    if request.is_desktop_target && request.cuda_available {
        return RuntimeExecutionProfile::CudaGpuResident;
    }

    if request.is_desktop_target && request.vulkan_available {
        // mv31 정책: desktop Vulkan default fullpath. env로 비활성하려면
        // `RNB_GPU_FULLPATH=0` 명시.
        if request.fullpath_requested {
            return RuntimeExecutionProfile::ExperimentalFullpath;
        }
        // env 미설정 시에도 desktop Vulkan default는 fullpath.
        return RuntimeExecutionProfile::ExperimentalFullpath;
    }

    // Fullpath env가 들어왔지만 desktop Vulkan path 부재 (CUDA 없고 Vulkan 없음)
    // 같은 경계 케이스는 CPU-only.
    RuntimeExecutionProfile::CpuOnly
}

pub fn select_prefill_path_for_profile(
    profile: RuntimeExecutionProfile,
    token_count: usize,
    has_slice1_plan: bool,
    has_active_gpu_prefill_path: bool,
) -> PrefillExecutionPath {
    if token_count <= 1 || !has_active_gpu_prefill_path {
        return PrefillExecutionPath::Cpu;
    }

    match profile {
        RuntimeExecutionProfile::CpuOnly => PrefillExecutionPath::Cpu,
        RuntimeExecutionProfile::ExperimentalFullpath => PrefillExecutionPath::Fullpath,
        RuntimeExecutionProfile::MobileVulkanPrefillCpuDecode
        | RuntimeExecutionProfile::DesktopVulkanGpuResident
        | RuntimeExecutionProfile::CudaGpuResident => {
            if has_slice1_plan {
                PrefillExecutionPath::Slice1GpuCandidate
            } else {
                PrefillExecutionPath::Cpu
            }
        }
    }
}

pub fn plan_slice1_boundary(
    num_layers: usize,
    full_attention_interval: usize,
) -> Option<Slice1BoundaryPlan> {
    if full_attention_interval == 0 || num_layers < full_attention_interval {
        return None;
    }

    let attention_layer_idx = full_attention_interval - 1;
    Some(Slice1BoundaryPlan {
        cpu_prefix_layer_range: 0..attention_layer_idx,
        window_layer_range: attention_layer_idx..attention_layer_idx + 1,
        attention_layer_idx,
    })
}

pub fn should_attempt_slice1_gpu_prefill(token_count: usize, has_slice1_plan: bool) -> bool {
    token_count > 1 && has_slice1_plan
}

/// `RNB_GPU_FULLPATH=1` 환경변수가 설정됐는지 검사.
///
/// mv27 task 10b-4 fullpath GPU prefill/decode 진입 gate.
/// runtime policy env parsing 은 scheduler 가 소유 (rnb-llm 의 boundary 규칙
/// `llm_does_not_parse_runtime_policy_env_directly` 위반 회피).
pub fn fullpath_gpu_prefill_requested() -> bool {
    std::env::var("RNB_GPU_FULLPATH")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// mv40 (2026-05-07) — mobile vulkan partial path opt-in.
/// Default 폐기 (CpuOnly). 명시적으로 `RNB_FORCE_MOBILE_VULKAN=1` 일 때만 진입.
pub fn force_mobile_vulkan_requested() -> bool {
    std::env::var("RNB_FORCE_MOBILE_VULKAN")
        .map(|value| value != "0")
        .unwrap_or(false)
}

pub fn select_prefill_path(
    token_count: usize,
    has_slice1_plan: bool,
    has_active_gpu_prefill_path: bool,
    fullpath_requested: bool,
) -> PrefillExecutionPath {
    let profile = if fullpath_requested {
        RuntimeExecutionProfile::ExperimentalFullpath
    } else {
        RuntimeExecutionProfile::MobileVulkanPrefillCpuDecode
    };
    select_prefill_path_for_profile(
        profile,
        token_count,
        has_slice1_plan,
        has_active_gpu_prefill_path,
    )
}

pub fn plan_moe_jit_load_order(selected: &[usize], probabilities: &[f32]) -> Vec<usize> {
    let mut experts = selected.to_vec();
    experts.sort_by(|&left, &right| {
        probabilities[right]
            .partial_cmp(&probabilities[left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    experts
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_model_ir::{LayerSpec, ModelKind};

    #[test]
    fn schedule_plan_records_backend_and_memory_placement() {
        let mut plan = SchedulePlan::new();
        plan.push(Placement::new(
            LayerId(3),
            BackendOp::MoE,
            BackendKind::Cuda,
            MemoryTier::Vram,
        ));

        assert_eq!(plan.placements().len(), 1);
        assert_eq!(plan.placements()[0].layer(), LayerId(3));
        assert_eq!(plan.placements()[0].backend(), BackendKind::Cuda);
        assert_eq!(plan.placements()[0].memory_tier(), MemoryTier::Vram);
    }

    #[test]
    fn schedules_model_layers_with_backend_capabilities_and_memory_budget() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Attention));
        graph.push_layer(LayerSpec::new(LayerId(1), LayerKind::Moe));
        let capabilities = [
            BackendCapabilities::new(BackendKind::Cuda)
                .with_op(BackendOp::Attention)
                .with_op(BackendOp::MoE),
            BackendCapabilities::new(BackendKind::Cpu).with_op(BackendOp::MatMul),
        ];
        let memory_budgets = [MemoryBudget::new(MemoryTier::Vram, 1024, 128)];

        let plan = schedule_model_graph(&graph, &capabilities, &memory_budgets).unwrap();

        assert_eq!(plan.placements().len(), 2);
        assert_eq!(plan.placements()[0].backend(), BackendKind::Cuda);
        assert_eq!(plan.placements()[1].op(), BackendOp::MoE);
        assert_eq!(plan.placements()[1].memory_tier(), MemoryTier::Vram);
    }

    #[test]
    fn mediatek_npu_matmul_uses_npu_memory_tier() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::FeedForward));
        let capabilities =
            [BackendCapabilities::new(BackendKind::MediaTekNpu).with_op(BackendOp::MatMul)];
        let memory_budgets = [MemoryBudget::new(MemoryTier::Npu, 64 * 1024 * 1024, 0)];

        let plan = schedule_model_graph(&graph, &capabilities, &memory_budgets).unwrap();

        assert_eq!(plan.placements().len(), 1);
        assert_eq!(plan.placements()[0].backend(), BackendKind::MediaTekNpu);
        assert_eq!(plan.placements()[0].memory_tier(), MemoryTier::Npu);
    }

    #[test]
    fn mediatek_npu_requires_npu_memory_budget() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::FeedForward));
        let capabilities =
            [BackendCapabilities::new(BackendKind::MediaTekNpu).with_op(BackendOp::MatMul)];
        let memory_budgets = [MemoryBudget::new(MemoryTier::Ram, 1024, 0)];

        let error = schedule_model_graph(&graph, &capabilities, &memory_budgets).unwrap_err();

        assert_eq!(
            error,
            ScheduleError::MissingMemoryBudget {
                layer: LayerId(0),
                tier: MemoryTier::Npu
            }
        );
    }

    #[test]
    fn scheduler_rejects_missing_memory_budget_for_selected_backend() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Moe));
        let capabilities = [BackendCapabilities::new(BackendKind::Cuda).with_op(BackendOp::MoE)];
        let memory_budgets = [MemoryBudget::new(MemoryTier::Ram, 1024, 0)];

        let error = schedule_model_graph(&graph, &capabilities, &memory_budgets).unwrap_err();

        assert_eq!(
            error,
            ScheduleError::MissingMemoryBudget {
                layer: LayerId(0),
                tier: MemoryTier::Vram
            }
        );
    }

    #[test]
    fn scheduler_rejects_unsupported_layer_op() {
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Gdn));
        let capabilities = [BackendCapabilities::new(BackendKind::Cpu).with_op(BackendOp::MatMul)];
        let memory_budgets = [MemoryBudget::new(MemoryTier::Ram, 1024, 0)];

        let error = schedule_model_graph(&graph, &capabilities, &memory_budgets).unwrap_err();

        assert_eq!(
            error,
            ScheduleError::UnsupportedLayerOp {
                layer: LayerId(0),
                op: BackendOp::Gdn
            }
        );
    }

    #[test]
    fn plans_slice1_prefill_boundary_from_layer_counts() {
        let plan = plan_slice1_boundary(8, 4).expect("slice1 boundary should exist");

        assert_eq!(plan.cpu_prefix_layer_range, 0..3);
        assert_eq!(plan.window_layer_range, 3..4);
        assert_eq!(plan.attention_layer_idx, 3);
        assert_eq!(
            select_prefill_path(8, true, true, false),
            PrefillExecutionPath::Slice1GpuCandidate
        );
        assert_eq!(
            select_prefill_path(1, true, true, false),
            PrefillExecutionPath::Cpu
        );
        assert_eq!(
            select_prefill_path(8, true, false, false),
            PrefillExecutionPath::Cpu
        );
        // Fullpath requested + slice1 eligibility holds -> Fullpath.
        assert_eq!(
            select_prefill_path(8, true, true, true),
            PrefillExecutionPath::Fullpath
        );
        // Fullpath is a model-wide GPU path and does not require a hybrid
        // slice1 boundary. Pure-attention models therefore remain eligible.
        assert_eq!(
            select_prefill_path(8, false, true, true),
            PrefillExecutionPath::Fullpath
        );
        assert_eq!(
            select_prefill_path(8, true, false, true),
            PrefillExecutionPath::Cpu
        );
    }

    #[test]
    fn rejects_slice1_prefill_boundary_without_attention_interval() {
        assert_eq!(plan_slice1_boundary(8, 0), None);
        assert_eq!(plan_slice1_boundary(2, 4), None);
        assert!(!should_attempt_slice1_gpu_prefill(8, false));
    }

    #[test]
    fn mv40_mobile_vulkan_default_off_falls_back_to_cpu() {
        // mv40: mobile + vulkan_available + force_mobile_vulkan=false → CpuOnly.
        // 학계 (arxiv 2505.06461 등) + 자체 측정으로 mobile GPU 가 LLM inference 에서
        // ARM CPU NEON 보다 느림이 확정됨. 기본 폐기.
        let request = ExecutionProfileRequest {
            is_android_target: true,
            is_mobile_target: true,
            is_desktop_target: false,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: false,
            fullpath_requested: false,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::CpuOnly
        );
    }

    #[test]
    fn mv40_mobile_vulkan_force_opt_in_activates_partial_path() {
        // mv40: mobile + vulkan_available + force_mobile_vulkan=true (opt-in) →
        // MobileVulkanPrefillCpuDecode. 코드는 NPU/FlashAttention 같은 미래 axis
        // 위해 살아있고, 명시적 force 일 때만 진입.
        let request = ExecutionProfileRequest {
            is_android_target: true,
            is_mobile_target: true,
            is_desktop_target: false,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: false,
            fullpath_requested: false,
            force_mobile_vulkan_requested: true,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::MobileVulkanPrefillCpuDecode
        );
        assert_eq!(
            select_prefill_path_for_profile(
                RuntimeExecutionProfile::MobileVulkanPrefillCpuDecode,
                8,
                true,
                true,
            ),
            PrefillExecutionPath::Slice1GpuCandidate
        );
    }

    #[test]
    fn runtime_execution_profile_falls_back_to_cpu_without_mobile_vulkan() {
        let request = ExecutionProfileRequest {
            is_android_target: true,
            is_mobile_target: true,
            is_desktop_target: false,
            cpu_available: true,
            vulkan_available: false,
            cuda_available: false,
            fullpath_requested: false,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::CpuOnly
        );
    }

    #[test]
    fn runtime_execution_profile_prefers_desktop_cuda_over_vulkan() {
        let request = ExecutionProfileRequest {
            is_android_target: false,
            is_mobile_target: false,
            is_desktop_target: true,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: true,
            fullpath_requested: false,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::CudaGpuResident
        );
    }

    #[test]
    fn runtime_execution_profile_selects_desktop_vulkan_without_cuda() {
        // mv31 정책: desktop + vulkan + no cuda → fullpath default ON
        // (env 미설정 시에도 ExperimentalFullpath).
        let request = ExecutionProfileRequest {
            is_android_target: false,
            is_mobile_target: false,
            is_desktop_target: true,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: false,
            fullpath_requested: false,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::ExperimentalFullpath
        );
    }

    #[test]
    fn mv31_mv40_mobile_fullpath_request_is_ignored() {
        // mv31: mobile + fullpath_requested 는 warn 후 무시.
        // mv40: mobile vulkan partial 도 default OFF → CpuOnly 로 떨어짐.
        let request = ExecutionProfileRequest {
            is_android_target: true,
            is_mobile_target: true,
            is_desktop_target: false,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: false,
            fullpath_requested: true,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::CpuOnly
        );
    }

    #[test]
    fn desktop_explicit_fullpath_experiment_returns_fullpath() {
        // Desktop 에서 fullpath 시도는 mv31 정책 그대로 (cuda 없을 때).
        let request = ExecutionProfileRequest {
            is_android_target: false,
            is_mobile_target: false,
            is_desktop_target: true,
            cpu_available: true,
            vulkan_available: true,
            cuda_available: false,
            fullpath_requested: true,
            force_mobile_vulkan_requested: false,
            requested_profile: None,
        };

        assert_eq!(
            select_runtime_execution_profile(request),
            RuntimeExecutionProfile::ExperimentalFullpath
        );
        assert_eq!(
            select_prefill_path_for_profile(
                RuntimeExecutionProfile::ExperimentalFullpath,
                8,
                true,
                true,
            ),
            PrefillExecutionPath::Fullpath
        );
        assert_eq!(
            select_prefill_path_for_profile(
                RuntimeExecutionProfile::ExperimentalFullpath,
                8,
                false,
                true,
            ),
            PrefillExecutionPath::Fullpath
        );
        assert_eq!(
            select_prefill_path_for_profile(
                RuntimeExecutionProfile::ExperimentalFullpath,
                1,
                true,
                true,
            ),
            PrefillExecutionPath::Cpu
        );
    }

    #[test]
    fn plans_moe_jit_load_order_by_probability_then_expert_id() {
        let selected = [7, 3, 5, 1];
        let probabilities = [0.0, 0.4, 0.0, 0.9, 0.0, 0.9, 0.0, 0.4];

        let order = plan_moe_jit_load_order(&selected, &probabilities);

        assert_eq!(order, vec![3, 5, 1, 7]);
    }
}
