use rnb_backend_api::{
    BackendError, BackendErrorKind, BackendKind, BackendOutput, BackendRequest, BackendResult,
};

pub fn execute_backend_request(
    backend: BackendKind,
    request: BackendRequest,
) -> BackendResult<BackendOutput> {
    match backend {
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                let mut backend = rnb_backend_cpu::CpuBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let mut backend = rnb_backend_cuda::CudaBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
        BackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                let mut backend = rnb_backend_vulkan::VulkanBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
        BackendKind::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                let mut backend = rnb_backend_opencl::OpenClBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "opencl"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
        BackendKind::Metal => {
            #[cfg(feature = "metal")]
            {
                let mut backend = rnb_backend_metal::MetalBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
        BackendKind::MediaTekNpu => {
            #[cfg(feature = "mediatek")]
            {
                let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
                rnb_backend_api::Backend::execute(&mut backend, request)
            }
            #[cfg(not(feature = "mediatek"))]
            {
                Err(compiled_backend_missing(backend, request))
            }
        }
    }
}

fn compiled_backend_missing(backend: BackendKind, request: BackendRequest) -> BackendError {
    BackendError::new(
        BackendErrorKind::UnsupportedOp,
        backend,
        Some(request.op()),
        format!("{backend:?} backend is not compiled into this runtime"),
    )
}
