//! GDN decode engine bridge.

use super::*;

/// GDN (Gated Delta Net) layer decode (seq_len=1). Operates in-place on scratch buffers.
pub(super) fn decode_gdn_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    w: &GdnLayerWeights,
    layer_idx: usize,
    #[cfg(feature = "vulkan")] gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    models::qwen::decode_gdn_layer_qwen(
        kv_cache,
        metadata,
        scratch,
        w,
        layer_idx,
        #[cfg(feature = "vulkan")]
        gpu_runtime,
    )
}
