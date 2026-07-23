use rnb_backend_api::{BackendCapabilities, BackendKind};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackendRegistry {
    backends: Vec<BackendKind>,
}

impl BackendRegistry {
    pub fn compiled() -> Self {
        #[allow(unused_mut)]
        let mut registry = Self::default();
        #[cfg(feature = "cpu")]
        registry.push(BackendKind::Cpu);
        #[cfg(feature = "cuda")]
        registry.push(BackendKind::Cuda);
        #[cfg(feature = "vulkan")]
        registry.push(BackendKind::Vulkan);
        #[cfg(feature = "opencl")]
        registry.push(BackendKind::OpenCl);
        #[cfg(feature = "metal")]
        registry.push(BackendKind::Metal);
        #[cfg(feature = "mediatek")]
        registry.push(BackendKind::MediaTekNpu);
        registry
    }

    pub fn push(&mut self, backend: BackendKind) {
        if !self.backends.contains(&backend) {
            self.backends.push(backend);
        }
    }

    pub fn contains(&self, backend: BackendKind) -> bool {
        self.backends.contains(&backend)
    }

    pub fn backends(&self) -> &[BackendKind] {
        &self.backends
    }
}

pub(crate) fn compiled_capabilities_for(backend: BackendKind) -> Option<BackendCapabilities> {
    match backend {
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_cpu::CpuBackend::new(),
                ))
            }
            #[cfg(not(feature = "cpu"))]
            {
                None
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_cuda::CudaBackend::new(),
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        }
        BackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_vulkan::VulkanBackend::new(),
                ))
            }
            #[cfg(not(feature = "vulkan"))]
            {
                None
            }
        }
        BackendKind::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_opencl::OpenClBackend::new(),
                ))
            }
            #[cfg(not(feature = "opencl"))]
            {
                None
            }
        }
        BackendKind::Metal => {
            #[cfg(feature = "metal")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_metal::MetalBackend::new(),
                ))
            }
            #[cfg(not(feature = "metal"))]
            {
                None
            }
        }
        BackendKind::MediaTekNpu => {
            #[cfg(feature = "mediatek")]
            {
                Some(rnb_backend_api::Backend::capabilities(
                    &rnb_backend_mediatek::MediaTekBackend::new(),
                ))
            }
            #[cfg(not(feature = "mediatek"))]
            {
                None
            }
        }
    }
}
