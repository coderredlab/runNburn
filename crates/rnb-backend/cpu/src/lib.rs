use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOp, BackendOutput,
    BackendRequest, BackendResult,
};

#[derive(Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::Cpu)
            .with_op(BackendOp::MatMul)
            .with_op(BackendOp::Attention)
            .with_op(BackendOp::Gdn)
            .with_op(BackendOp::MoE)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        if self.capabilities().supports(request.op()) {
            Ok(BackendOutput::new(request.op()))
        } else {
            Err(BackendError::unsupported(self.kind(), request.op()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_supports_core_llm_ops() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();

        assert!(caps.supports(BackendOp::MatMul));
        assert!(caps.supports(BackendOp::Attention));
        assert!(caps.supports(BackendOp::MoE));
        assert!(!caps.supports(BackendOp::Sampler));
    }
}
