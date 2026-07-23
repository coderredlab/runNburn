/// OS page (4 KiB); larger reserves break linear scaling on small prompts.
const STAGING_RESERVE_DEFAULT_BYTES: usize = 4096;

pub struct StagingPolicy {
    pub reserve_bytes: usize,
}

impl Default for StagingPolicy {
    fn default() -> Self {
        let reserve_bytes = std::env::var("RNB_GPU_STAGING_RESERVE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(STAGING_RESERVE_DEFAULT_BYTES);
        Self { reserve_bytes }
    }
}

impl StagingPolicy {
    pub fn bytes_for(&self, seq_len: usize, hidden: usize) -> usize {
        let raw = seq_len.saturating_mul(hidden).saturating_mul(2); // f16 = 2B
        raw.saturating_add(self.reserve_bytes)
    }
}
