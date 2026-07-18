use rnb_backend_api::BackendKind;

pub const ACCELERATOR_BACKENDS: [GpuBackend; 4] = [
    GpuBackend::Cuda,
    GpuBackend::Vulkan,
    GpuBackend::OpenCl,
    GpuBackend::Metal,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Cuda,
    Vulkan,
    OpenCl,
    Metal,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBackendSupport {
    pub backend: GpuBackend,
    pub name: &'static str,
    pub feature_name: Option<&'static str>,
    pub compiled: bool,
    pub runtime_entrypoints: bool,
}

impl GpuBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::OpenCl => "opencl",
            Self::Metal => "metal",
            Self::None => "none",
        }
    }

    pub const fn feature_name(self) -> Option<&'static str> {
        match self {
            Self::Cuda => Some("cuda"),
            Self::Vulkan => Some("vulkan"),
            Self::OpenCl => Some("opencl"),
            Self::Metal => Some("metal"),
            Self::None => None,
        }
    }

    pub const fn is_accelerator(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn compiled(self) -> bool {
        match self {
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Vulkan => cfg!(feature = "vulkan"),
            Self::OpenCl => cfg!(feature = "opencl"),
            Self::Metal => cfg!(feature = "metal"),
            Self::None => true,
        }
    }

    pub fn runtime_entrypoints(self) -> bool {
        match self {
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Vulkan => cfg!(feature = "vulkan"),
            Self::OpenCl => false,
            Self::Metal => false,
            Self::None => false,
        }
    }

    pub fn support(self) -> GpuBackendSupport {
        GpuBackendSupport {
            backend: self,
            name: self.name(),
            feature_name: self.feature_name(),
            compiled: self.compiled(),
            runtime_entrypoints: self.runtime_entrypoints(),
        }
    }

    pub fn from_env_var(name: &str) -> Option<Self> {
        match crate::policy::env_string(name)
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("cuda") => Some(Self::Cuda),
            Some("vulkan") => Some(Self::Vulkan),
            Some("opencl") => Some(Self::OpenCl),
            Some("metal") => Some(Self::Metal),
            Some("off") | Some("0") | Some("none") => Some(Self::None),
            Some(_) | None => None,
        }
    }

    pub fn from_moe_jit_env() -> Option<Self> {
        match Self::from_env_var("RNB_MOE_JIT_BACKEND") {
            Some(Self::None) | None => None,
            Some(backend) => Some(backend),
        }
    }

    pub fn from_env() -> Option<Self> {
        Self::from_moe_jit_env()
    }

    pub const fn backend_kind(self) -> Option<BackendKind> {
        match self {
            Self::Cuda => Some(BackendKind::Cuda),
            Self::Vulkan => Some(BackendKind::Vulkan),
            Self::OpenCl => Some(BackendKind::OpenCl),
            Self::Metal => Some(BackendKind::Metal),
            Self::None => None,
        }
    }
}
