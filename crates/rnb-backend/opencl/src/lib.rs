use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOutput, BackendRequest,
    BackendResult,
};

#[derive(Debug, Default)]
pub struct OpenClBackend;

impl OpenClBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for OpenClBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::OpenCl
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::OpenCl)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        Err(BackendError::unsupported(self.kind(), request.op()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_backend_api::BackendOp;

    #[test]
    fn opencl_backend_has_no_fake_fallback_execution() {
        let mut backend = OpenClBackend::new();

        assert!(!backend.capabilities().supports(BackendOp::MatMul));
        assert!(matches!(
            backend.execute(BackendRequest::new(BackendOp::MatMul)),
            Err(err) if err.backend() == BackendKind::OpenCl && err.op() == Some(BackendOp::MatMul)
        ));
    }
}
