//! Vulkan decode attention helpers.

#[cfg(feature = "vulkan")]
use super::*;

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_attention_single_head(
    runtime: &mut gpu::Runtime,
    layer_idx: usize,
    kv_cache_layer: usize,
    pos: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    head_dim: usize,
    kv_len: usize,
    output: &mut [f32],
) -> Result<bool, String> {
    let gpu_kv_ready = runtime
        .append_attention_kv_f32_for_layer(kv_cache_layer, pos, k, v)
        .is_ok();
    if gpu_kv_ready {
        runtime
            .attention_decode_gpu_kv_mirror_for_layer(layer_idx, q, head_dim, kv_len, output)
            .map(|()| true)
    } else {
        runtime
            .attention_decode_f16_cache(q, cached_k_f16, cached_v_f16, head_dim, kv_len, output)
            .map(|()| true)
    }
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_decode_attention_single_head_if_supported(
    #[cfg(feature = "vulkan")] runtime: Option<&mut gpu::Runtime>,
    layer_idx: usize,
    kv_cache_layer: usize,
    pos: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    head_dim: usize,
    kv_len: usize,
    output: &mut [f32],
) -> Result<bool, String> {
    #[cfg(feature = "vulkan")]
    if let Some(runtime) = runtime {
        return try_decode_attention_single_head(
            runtime,
            layer_idx,
            kv_cache_layer,
            pos,
            q,
            k,
            v,
            cached_k_f16,
            cached_v_f16,
            head_dim,
            kv_len,
            output,
        );
    }
    Ok(false)
}
