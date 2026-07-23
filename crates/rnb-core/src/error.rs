use thiserror::Error;

#[derive(Debug, Error)]
pub enum RnbError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("unsupported op `{op}` on backend `{backend}`")]
    UnsupportedOp { op: String, backend: String },

    #[error("unsupported dtype `{dtype}` on backend `{backend}`")]
    UnsupportedDType { dtype: String, backend: String },

    #[error("out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },

    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("backend error: {0}")]
    BackendError(String),
}

pub type Result<T> = std::result::Result<T, RnbError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RnbError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: RnbError = io_err.into();
        assert!(matches!(err, RnbError::IoError(_)));
    }
}
