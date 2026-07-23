use crate::backend::{compiled_capabilities_for, BackendKind};
use rnb_backend_api::BackendCapabilities;
use rnb_memory::MemoryBudget;
use rnb_model_ir::ModelGraph;
use rnb_platform::{OperatingSystem, RuntimeTarget};
use rnb_scheduler::{
    schedule_model_graph, select_runtime_execution_profile, ExecutionProfileRequest,
    RuntimeExecutionProfile, ScheduleError, SchedulePlan,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    UnsupportedBackend {
        target: RuntimeTarget,
        backend: BackendKind,
    },
    Schedule(ScheduleError),
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    target: RuntimeTarget,
    backends: Vec<BackendKind>,
    memory_budgets: Vec<MemoryBudget>,
    requested_execution_profile: Option<RuntimeExecutionProfile>,
}

impl RuntimeConfig {
    pub fn new(target: RuntimeTarget) -> Self {
        Self {
            target,
            backends: vec![BackendKind::Cpu],
            memory_budgets: Vec::new(),
            requested_execution_profile: None,
        }
    }

    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        if !self.backends.contains(&backend) {
            self.backends.push(backend);
        }
        self
    }

    pub fn with_memory_budget(mut self, budget: MemoryBudget) -> Self {
        self.memory_budgets.push(budget);
        self
    }

    pub fn with_execution_profile(mut self, profile: RuntimeExecutionProfile) -> Self {
        self.requested_execution_profile = Some(profile);
        self
    }

    pub fn target(&self) -> RuntimeTarget {
        self.target
    }

    pub fn backends(&self) -> &[BackendKind] {
        &self.backends
    }

    pub fn memory_budgets(&self) -> &[MemoryBudget] {
        &self.memory_budgets
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSession {
    config: RuntimeConfig,
    execution_profile: RuntimeExecutionProfile,
}

impl RuntimeSession {
    pub fn new(config: RuntimeConfig) -> RuntimeResult<Self> {
        for &backend in config.backends() {
            if !target_supports_backend(config.target(), backend) {
                return Err(RuntimeError::UnsupportedBackend {
                    target: config.target(),
                    backend,
                });
            }
        }
        let execution_profile = resolve_execution_profile(&config);
        Ok(Self {
            config,
            execution_profile,
        })
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn schedule_model(&self, graph: &ModelGraph) -> RuntimeResult<SchedulePlan> {
        let capabilities = self.compiled_capabilities();
        schedule_model_graph(graph, &capabilities, self.config.memory_budgets())
            .map_err(RuntimeError::Schedule)
    }

    pub fn execution_profile(&self) -> RuntimeExecutionProfile {
        self.execution_profile
    }

    fn compiled_capabilities(&self) -> Vec<BackendCapabilities> {
        self.config
            .backends()
            .iter()
            .filter_map(|backend| compiled_capabilities_for(*backend))
            .collect()
    }
}

fn resolve_execution_profile(config: &RuntimeConfig) -> RuntimeExecutionProfile {
    select_runtime_execution_profile(ExecutionProfileRequest {
        is_android_target: config.target().is_android(),
        is_mobile_target: config.target().is_mobile(),
        is_desktop_target: config.target().is_desktop(),
        cpu_available: config.backends().contains(&BackendKind::Cpu),
        vulkan_available: config.backends().contains(&BackendKind::Vulkan),
        cuda_available: config.backends().contains(&BackendKind::Cuda),
        fullpath_requested: false,
        force_mobile_vulkan_requested: false,
        requested_profile: config.requested_execution_profile,
    })
}

pub fn target_supports_backend(target: RuntimeTarget, backend: BackendKind) -> bool {
    match backend {
        BackendKind::Cpu => true,
        BackendKind::Cuda => matches!(target.os, OperatingSystem::Linux | OperatingSystem::Windows),
        BackendKind::Vulkan => {
            !matches!(target.os, OperatingSystem::Ios | OperatingSystem::Unknown)
        }
        BackendKind::OpenCl => {
            !matches!(target.os, OperatingSystem::Ios | OperatingSystem::Unknown)
        }
        BackendKind::Metal => cfg!(target_os = "macos"),
        BackendKind::MediaTekNpu => {
            matches!(target.os, OperatingSystem::Android)
                && matches!(target.arch, rnb_platform::CpuArch::Aarch64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_platform::{CpuArch, FormFactor};

    #[cfg(feature = "cpu")]
    use rnb_memory::{MemoryBudget, MemoryTier};
    #[cfg(feature = "cpu")]
    use rnb_model_ir::{LayerId, LayerKind, LayerSpec, ModelGraph, ModelKind};

    #[test]
    fn runtime_rejects_cuda_on_android() {
        let target = RuntimeTarget::new(
            OperatingSystem::Android,
            CpuArch::Aarch64,
            FormFactor::Mobile,
        );
        let config = RuntimeConfig::new(target).with_backend(BackendKind::Cuda);

        assert!(matches!(
            RuntimeSession::new(config),
            Err(RuntimeError::UnsupportedBackend {
                backend: BackendKind::Cuda,
                ..
            })
        ));
    }

    #[test]
    fn runtime_accepts_cpu_on_android() {
        let target = RuntimeTarget::new(
            OperatingSystem::Android,
            CpuArch::Aarch64,
            FormFactor::Mobile,
        );
        let config = RuntimeConfig::new(target);

        assert!(RuntimeSession::new(config).is_ok());
    }

    #[test]
    fn runtime_accepts_mediatek_npu_on_android_aarch64() {
        let target = RuntimeTarget::new(
            OperatingSystem::Android,
            CpuArch::Aarch64,
            FormFactor::Mobile,
        );
        let config = RuntimeConfig::new(target).with_backend(BackendKind::MediaTekNpu);

        assert!(RuntimeSession::new(config).is_ok());
    }

    #[test]
    fn runtime_rejects_mediatek_npu_on_non_android_target() {
        let target =
            RuntimeTarget::new(OperatingSystem::Linux, CpuArch::X86_64, FormFactor::Desktop);
        let config = RuntimeConfig::new(target).with_backend(BackendKind::MediaTekNpu);

        assert!(matches!(
            RuntimeSession::new(config),
            Err(RuntimeError::UnsupportedBackend {
                backend: BackendKind::MediaTekNpu,
                ..
            })
        ));
    }

    #[test]
    fn runtime_session_execution_profile_mobile_default_is_cpu_only() {
        // mv40 (2026-05-07): mobile + vulkan backend 등록되어 있어도 default OFF.
        // resolve_execution_profile() 가 force_mobile_vulkan_requested=false 로
        // 호출하므로 항상 CpuOnly. opt-in 진입은 caller (engine) 가 env 읽어서.
        let target = RuntimeTarget::new(
            OperatingSystem::Android,
            CpuArch::Aarch64,
            FormFactor::Mobile,
        );
        let config = RuntimeConfig::new(target).with_backend(BackendKind::Vulkan);

        let session = RuntimeSession::new(config).unwrap();

        assert_eq!(
            session.execution_profile(),
            RuntimeExecutionProfile::CpuOnly
        );
    }

    #[test]
    fn runtime_session_execution_profile_preserves_explicit_request() {
        let target =
            RuntimeTarget::new(OperatingSystem::Linux, CpuArch::X86_64, FormFactor::Desktop);
        let config = RuntimeConfig::new(target)
            .with_backend(BackendKind::Vulkan)
            .with_execution_profile(RuntimeExecutionProfile::ExperimentalFullpath);

        let session = RuntimeSession::new(config).unwrap();

        assert_eq!(
            session.execution_profile(),
            RuntimeExecutionProfile::ExperimentalFullpath
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_schedules_model_through_scheduler_boundary() {
        let target =
            RuntimeTarget::new(OperatingSystem::Linux, CpuArch::X86_64, FormFactor::Desktop);
        let config = RuntimeConfig::new(target).with_memory_budget(MemoryBudget::new(
            MemoryTier::Ram,
            1024,
            0,
        ));
        let session = RuntimeSession::new(config).unwrap();
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Attention));

        let plan = session.schedule_model(&graph).unwrap();

        assert_eq!(plan.placements().len(), 1);
        assert_eq!(plan.placements()[0].backend(), BackendKind::Cpu);
        assert_eq!(plan.placements()[0].memory_tier(), MemoryTier::Ram);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_surfaces_scheduler_errors() {
        let target =
            RuntimeTarget::new(OperatingSystem::Linux, CpuArch::X86_64, FormFactor::Desktop);
        let config = RuntimeConfig::new(target);
        let session = RuntimeSession::new(config).unwrap();
        let mut graph = ModelGraph::new(ModelKind::DecoderOnly);
        graph.push_layer(LayerSpec::new(LayerId(0), LayerKind::Attention));

        let error = session.schedule_model(&graph).unwrap_err();

        assert!(matches!(
            error,
            RuntimeError::Schedule(ScheduleError::MissingMemoryBudget {
                tier: MemoryTier::Ram,
                ..
            })
        ));
    }
}
