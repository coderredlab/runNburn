use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("invalid GGUF magic bytes")]
    InvalidMagic,

    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("parse error at offset {offset}: {msg}")]
    ParseError { offset: usize, msg: String },

    #[error("unsupported GGML type: {0}")]
    UnsupportedGGMLType(u32),

    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("missing metadata key: {0}")]
    MissingKey(String),

    #[error("type mismatch for key '{key}': expected {expected}")]
    TypeMismatch { key: String, expected: String },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("rnb-core error: {0}")]
    CoreError(#[from] rnb_core::error::RnbError),
}
