const KV_STRIPES: usize = 2; // K stripe + V stripe
const F16_BYTES: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct KvResidentLayout {
    pub num_layers: usize,
    pub max_ctx: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvResidentLayout {
    pub fn compute(
        num_layers: usize,
        max_ctx: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            num_layers,
            max_ctx,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn resolve_max_ctx(gguf_context_length: usize) -> usize {
        std::env::var("RNB_MAX_CTX")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(|env: usize| env.min(gguf_context_length))
            .unwrap_or(gguf_context_length)
    }

    /// Layout: KV[layer][cursor][k_or_v][kv_head][head_dim] : f16
    pub fn total_bytes(&self) -> usize {
        self.num_layers * self.max_ctx * KV_STRIPES * self.num_kv_heads * self.head_dim * F16_BYTES
    }

    pub fn k_offset_bytes(&self, layer: usize, cursor: usize) -> usize {
        let per_layer = self.max_ctx * KV_STRIPES * self.num_kv_heads * self.head_dim * F16_BYTES;
        let per_token_pair = KV_STRIPES * self.num_kv_heads * self.head_dim * F16_BYTES;
        layer * per_layer + cursor * per_token_pair
    }

    pub fn v_offset_bytes(&self, layer: usize, cursor: usize) -> usize {
        let kv_pair_half = self.num_kv_heads * self.head_dim * F16_BYTES;
        self.k_offset_bytes(layer, cursor) + kv_pair_half
    }
}
