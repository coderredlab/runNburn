#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime::{
    AttentionKvMaterializeRangeRequest, AttentionKvMaterializeRequest,
};

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn materialize_gdn_conv_state_untracked(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    conv_state: &mut [f32],
) -> crate::error::Result<usize> {
    runtime
        .materialize_gdn_conv_state_f32_for_layer_untracked(layer_idx, conv_state)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn materialize_attention_kv_range_untracked(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    num_kv_heads: usize,
    pos_start: usize,
    kv_len: usize,
    head_dim: usize,
) -> crate::error::Result<((Vec<u16>, Vec<u16>), usize)> {
    let request = AttentionKvMaterializeRangeRequest::new(
        layer_idx,
        num_kv_heads,
        pos_start,
        kv_len,
        head_dim,
    );
    gpu::materialize_attention_kv_range_untracked(runtime, request)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn record_batched_materialization_download(
    runtime: &mut gpu::Runtime,
    total_bytes: usize,
) {
    runtime.record_batched_materialization_download(total_bytes);
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn append_attention_kv_f32_for_layer(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    pos: usize,
    k: &[f32],
    v: &[f32],
) -> crate::error::Result<()> {
    runtime
        .append_attention_kv_f32_for_layer(layer_idx, pos, k, v)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn attention_decode_window_grouped_from_mirror_for_layer(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    q: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    pos_start: usize,
    output: &mut [f32],
) -> crate::error::Result<()> {
    runtime
        .attention_decode_window_grouped_from_mirror_for_layer(
            layer_idx,
            q,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            pos_start,
            output,
        )
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn materialize_attention_kv_for_layer(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    num_kv_heads: usize,
    total_tokens: usize,
    head_dim: usize,
    kv_dim: usize,
) -> crate::error::Result<(Vec<u16>, Vec<u16>)> {
    let request =
        AttentionKvMaterializeRequest::new(layer_idx, num_kv_heads, total_tokens, head_dim, kv_dim);
    gpu::materialize_attention_kv(runtime, request)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn materialize_attention_kv_for_layer_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    num_kv_heads: usize,
    total_tokens: usize,
    head_dim: usize,
    kv_dim: usize,
) -> Option<crate::error::Result<(Vec<u16>, Vec<u16>)>> {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        return Some(materialize_attention_kv_for_layer(
            runtime,
            layer_idx,
            num_kv_heads,
            total_tokens,
            head_dim,
            kv_dim,
        ));
    }
    None
}
