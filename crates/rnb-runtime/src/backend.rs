pub use rnb_backend_api::{
    AttentionRequest, BackendError, BackendErrorKind, BackendKind, BackendOp, BackendOutput,
    BackendRequest, BackendResult, BackendWorkload, DecodeWeightKind, GdnRequest, KvBucketView,
    MatMulRequest, MoeJitByteRange, MoeJitExpertLoad, MoeJitLoadRequest, MoeJitLoadSink,
    MoeRequest, MoeRouteSlot, QuantFormat, QuantizedWeightView, ScalarType, TensorShape,
    TransformedSourceQuant, TransformedWeightLayout, TransformedWeightView,
};

mod execution;
mod gpu;
mod moe_jit;
mod registry;
pub use execution::execute_backend_request;
pub use gpu::{GpuBackend, GpuBackendSupport, ACCELERATOR_BACKENDS};
pub use moe_jit::{default_moe_jit_loader, moe_jit_report};
pub(crate) use registry::compiled_capabilities_for;
pub use registry::BackendRegistry;

pub fn compiled_accelerators() -> Vec<GpuBackend> {
    ACCELERATOR_BACKENDS
        .into_iter()
        .filter(|backend| backend.compiled())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn backend_registry_contains_compiled_backend_kinds() {
        let registry = BackendRegistry::compiled();

        #[cfg(feature = "cpu")]
        assert!(registry.contains(BackendKind::Cpu));
        #[cfg(feature = "cuda")]
        assert!(registry.contains(BackendKind::Cuda));
        #[cfg(feature = "vulkan")]
        assert!(registry.contains(BackendKind::Vulkan));
        #[cfg(feature = "opencl")]
        assert!(registry.contains(BackendKind::OpenCl));
        #[cfg(feature = "metal")]
        assert!(registry.contains(BackendKind::Metal));
        #[cfg(feature = "mediatek")]
        assert!(registry.contains(BackendKind::MediaTekNpu));
        #[cfg(not(feature = "mediatek"))]
        assert!(!registry.contains(BackendKind::MediaTekNpu));
        #[cfg(not(any(
            feature = "cpu",
            feature = "cuda",
            feature = "vulkan",
            feature = "opencl",
            feature = "metal",
            feature = "mediatek"
        )))]
        assert!(registry.backends().is_empty());
    }

    #[test]
    fn gpu_backend_support_reports_compile_time_feature() {
        let support = GpuBackend::Cuda.support();

        assert_eq!(support.backend, GpuBackend::Cuda);
        assert_eq!(support.name, "cuda");
        assert_eq!(support.feature_name, Some("cuda"));
        assert_eq!(support.compiled, cfg!(feature = "cuda"));
        assert_eq!(support.runtime_entrypoints, cfg!(feature = "cuda"));
    }

    #[test]
    fn opencl_is_capability_only_until_runtime_exists() {
        let support = GpuBackend::OpenCl.support();

        assert_eq!(support.backend, GpuBackend::OpenCl);
        assert_eq!(support.name, "opencl");
        assert_eq!(support.feature_name, Some("opencl"));
        assert_eq!(support.compiled, cfg!(feature = "opencl"));
        assert!(!support.runtime_entrypoints);
    }

    #[test]
    fn metal_reports_compile_time_feature_and_no_runtime_entrypoint_yet() {
        let support = GpuBackend::Metal.support();
        assert_eq!(support.name, "metal");
        assert_eq!(support.feature_name, Some("metal"));
        assert_eq!(support.compiled, cfg!(feature = "metal"));
        assert!(!support.runtime_entrypoints);
        assert_eq!(GpuBackend::Metal.backend_kind(), Some(BackendKind::Metal));
    }

    #[test]
    fn gpu_backend_env_parses_backend_axis_only() {
        let _guard = ENV_LOCK.get_or_init(Default::default).lock().unwrap();
        unsafe {
            std::env::set_var("RNB_TEST_GPU_BACKEND", "vulkan");
        }
        assert_eq!(
            GpuBackend::from_env_var("RNB_TEST_GPU_BACKEND"),
            Some(GpuBackend::Vulkan)
        );
        unsafe {
            std::env::set_var("RNB_TEST_GPU_BACKEND", "mobile");
        }
        assert_eq!(GpuBackend::from_env_var("RNB_TEST_GPU_BACKEND"), None);
        unsafe {
            std::env::remove_var("RNB_TEST_GPU_BACKEND");
        }
    }

    #[test]
    fn runtime_reexports_typed_backend_requests() {
        let matmul = MatMulRequest::new(
            TensorShape::new(16, 32),
            TensorShape::new(1, 32),
            QuantFormat::Q4K,
            ScalarType::F32,
        );
        let request = BackendRequest::matmul(matmul);

        assert_eq!(request.op(), BackendOp::MatMul);
        assert_eq!(request.workload(), BackendWorkload::MatMul(matmul));

        let gdn = GdnRequest::new(32, 2048, 128, 16, 4, QuantFormat::Q6K);
        let request = BackendRequest::gdn(gdn);

        assert_eq!(request.op(), BackendOp::Gdn);
        assert_eq!(request.workload(), BackendWorkload::Gdn(gdn));

        let transformed_source = vec![0xA5u8; 144];
        let transformed = TransformedWeightView::new(
            TransformedWeightLayout::Q4kCompactMetadata,
            TransformedSourceQuant::DenseQ4kRowPair,
            1,
            256,
            0x1111,
            1,
            256,
            0x2222,
            &transformed_source,
        )
        .expect("runtime reexports transformed weight view");

        assert_eq!(
            transformed.layout(),
            TransformedWeightLayout::Q4kCompactMetadata
        );
        assert_eq!(
            transformed.source_quant(),
            TransformedSourceQuant::DenseQ4kRowPair
        );
        assert_eq!(transformed.source_bytes(), transformed_source.as_slice());
    }

    #[test]
    fn runtime_facade_does_not_wildcard_reexport_backend_crates() {
        let lib_rs = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"))
            .expect("read lib.rs");
        let cuda_wildcard = concat!("pub use ", "rnb_backend_cuda::*");
        let vulkan_wildcard = concat!("pub use ", "rnb_backend_vulkan::*");

        assert!(!lib_rs.contains(cuda_wildcard));
        assert!(!lib_rs.contains(vulkan_wildcard));
    }

    #[test]
    fn runtime_executes_typed_request_through_compiled_backend_adapter() {
        let request = BackendRequest::matmul(MatMulRequest::new(
            TensorShape::new(16, 32),
            TensorShape::new(1, 32),
            QuantFormat::Q4K,
            ScalarType::F32,
        ));
        let result = execute_backend_request(BackendKind::Cpu, request);

        #[cfg(feature = "cpu")]
        assert_eq!(
            result.expect("cpu backend should execute matmul").op(),
            BackendOp::MatMul
        );
        #[cfg(not(feature = "cpu"))]
        assert!(matches!(
            result,
            Err(err)
                if err.kind() == BackendErrorKind::UnsupportedOp
                    && err.backend() == BackendKind::Cpu
                    && err.op() == Some(BackendOp::MatMul)
        ));
    }

    #[test]
    fn mediatek_npu_request_reports_backend_not_compiled() {
        let request = BackendRequest::matmul(MatMulRequest::new(
            TensorShape::new(16, 32),
            TensorShape::new(1, 32),
            QuantFormat::Q4K,
            ScalarType::F32,
        ));
        let result = execute_backend_request(BackendKind::MediaTekNpu, request);

        assert!(matches!(
            result,
            Err(err)
                if err.kind() == BackendErrorKind::UnsupportedOp
                    && err.backend() == BackendKind::MediaTekNpu
                    && err.op() == Some(BackendOp::MatMul)
        ));
    }

    #[cfg(feature = "mediatek")]
    #[test]
    fn mediatek_feature_registers_backend_without_fake_capabilities() {
        let registry = BackendRegistry::compiled();
        assert!(registry.contains(BackendKind::MediaTekNpu));

        let capabilities = compiled_capabilities_for(BackendKind::MediaTekNpu)
            .expect("mediatek feature should register a backend capability contract");
        assert_eq!(capabilities.backend(), BackendKind::MediaTekNpu);
        assert!(!capabilities.supports(BackendOp::MatMul));
    }

    #[cfg(feature = "mediatek")]
    #[test]
    fn mediatek_feature_dispatches_to_backend_but_stays_unsupported() {
        let request = BackendRequest::matmul(MatMulRequest::new(
            TensorShape::new(16, 32),
            TensorShape::new(1, 32),
            QuantFormat::Q4K,
            ScalarType::F32,
        ));
        let result = execute_backend_request(BackendKind::MediaTekNpu, request);

        assert!(matches!(
            result,
            Err(err)
                if err.kind() == BackendErrorKind::UnsupportedOp
                    && err.backend() == BackendKind::MediaTekNpu
                    && err.op() == Some(BackendOp::MatMul)
        ));
    }

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
}
