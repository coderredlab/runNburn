#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvarnAttentionBackend {
    Cuda,
    Metal,
}

impl KvarnAttentionBackend {
    fn env_name(self) -> &'static str {
        match self {
            Self::Cuda => "RNB_CUDA_KVARN_ATTN",
            Self::Metal => "RNB_METAL_KVARN_ATTN",
        }
    }

    fn host_min_attention_elements(self) -> usize {
        match self {
            Self::Cuda => 8 * 1024 * 1024,
            Self::Metal => 48 * 1024 * 1024,
        }
    }
}

pub fn kvarn_attention_enabled(
    backend: KvarnAttentionBackend,
    device_output: bool,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> bool {
    match std::env::var(backend.env_name()).as_deref() {
        Ok("0") => false,
        Ok(_) => true,
        Err(_) => {
            device_output
                || kv_len.saturating_mul(num_heads).saturating_mul(head_dim)
                    >= backend.host_min_attention_elements()
        }
    }
}

pub fn metal_kvarn_chain_attention_enabled() -> bool {
    std::env::var("RNB_METAL_KVARN_CHAIN_ATTN").as_deref() == Ok("1")
}
