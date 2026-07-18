//! Prefill KV handoff and slice1 boundary planning helpers.

use super::*;

pub(in crate::engine) fn new_empty_kv_cache(metadata: &ModelMetadata) -> KVCache {
    KVCache::new(
        metadata.num_layers,
        metadata.max_seq_len,
        metadata.num_kv_heads,
        metadata.head_dim,
    )
}

pub(super) fn take_kv_cache_for_handoff(
    metadata: &ModelMetadata,
    kv_cache: &mut KVCache,
) -> KVCache {
    std::mem::replace(kv_cache, new_empty_kv_cache(metadata))
}

pub(in crate::engine) fn plan_slice1_boundary(
    metadata: &ModelMetadata,
) -> Option<Slice1BoundaryPlan> {
    crate::runtime::scheduler::plan_slice1_boundary(
        metadata.num_layers,
        metadata.full_attention_interval,
    )
}
