//! GDN prefill layer forward path.

use super::*;

/// GDN (Gated Delta Net) 레이어 forward
pub(super) fn forward_gdn_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    models::qwen::forward_gdn_layer_impl(
        kv_cache,
        metadata,
        hidden,
        w,
        layer_idx,
        seq_len,
        norm_eps,
        None,
        #[cfg(feature = "vulkan")]
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
}

pub(super) fn forward_gdn_layer_collect_prefix_state(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
    prefix_collector: Option<&mut crate::engine::verify_window::GdnPrefixStateCollector>,
) -> crate::error::Result<Tensor> {
    models::qwen::forward_gdn_layer_impl(
        kv_cache,
        metadata,
        hidden,
        w,
        layer_idx,
        seq_len,
        norm_eps,
        prefix_collector,
        #[cfg(feature = "vulkan")]
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
}

#[cfg(feature = "vulkan")]
pub(super) fn forward_gdn_layer_with_gpu(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: Tensor,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
    gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
    deferred_gdn_flush: Option<&mut DeferredGdnConvStateFlush>,
) -> crate::error::Result<Tensor> {
    models::qwen::forward_gdn_layer_impl(
        kv_cache,
        metadata,
        hidden,
        w,
        layer_idx,
        seq_len,
        norm_eps,
        None,
        gpu_runtime,
        deferred_gdn_flush,
    )
}
