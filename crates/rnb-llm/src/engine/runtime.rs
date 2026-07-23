//! Runtime dependencies used by the LLM engine.
//!
//! The crate-level `crate::runtime` module is the boundary to `rnb-runtime`.
//! This engine-local facade keeps existing engine call sites stable while making
//! the runtime dependency groups visible in one place.

pub(crate) mod cpu_runtime {
    pub(crate) use crate::runtime::cpu::*;
}
pub(crate) mod cpu_phase_runtime {
    pub(crate) use crate::runtime::cpu_phase::*;
}
#[cfg(feature = "cuda")]
pub(crate) mod compute_runtime {
    pub(crate) use crate::runtime::compute::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda_runtime {
    pub(crate) use crate::runtime::cuda::*;
}

#[cfg(feature = "metal")]
pub(crate) mod metal_runtime {
    pub(crate) use crate::runtime::metal::*;
}

pub(crate) mod gemm_runtime {
    pub(crate) use crate::runtime::gemm::*;
}
#[cfg(feature = "mediatek")]
pub(crate) mod mediatek_runtime {
    pub(crate) use crate::runtime::mediatek::*;
}

#[cfg(feature = "vulkan")]
pub(crate) mod gpu_runtime {
    pub(crate) use crate::runtime::gpu::*;
}

pub(crate) mod memory_runtime {
    pub(crate) use crate::runtime::memory::*;
}

pub(crate) mod platform_runtime {
    pub(crate) use crate::runtime::platform::*;
}
#[cfg(feature = "cuda")]
pub(crate) mod tuning_runtime {
    pub(crate) use crate::runtime::tuning::*;
}

#[cfg(any(feature = "cuda", test))]
pub(crate) mod workspace_runtime {
    pub(crate) use crate::runtime::workspace::*;
}

pub(crate) mod policy {
    pub(crate) use crate::runtime::policy::*;
}
