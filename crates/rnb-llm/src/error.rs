use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model load error: {0}")]
    ModelLoad(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid chat request: {0}")]
    InvalidChatRequest(String),
    #[error("generation cancelled")]
    Cancelled,
    #[error("forward pass error: {0}")]
    Forward(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, LlmError>;
