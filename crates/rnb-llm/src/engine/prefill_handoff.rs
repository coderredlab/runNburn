#[cfg(feature = "vulkan")]
use super::backend_runtime;
use super::KVCache;
pub(super) use crate::runtime::scheduler::{PrefillExecutionPath, Slice1BoundaryPlan};

pub(super) struct SliceWindowHandoff {
    pub(super) hidden_after_window: Vec<f32>,
    #[cfg(test)]
    pub(super) next_layer_idx: usize,
    #[cfg(test)]
    pub(super) next_pos: usize,
    pub(super) cpu_kv_cache: KVCache,
}

pub(super) struct PrefillHandoff {
    pub(super) logits: Vec<f32>,
    #[cfg(test)]
    pub(super) next_pos: usize,
    pub(super) cpu_kv_cache: KVCache,
}

pub(super) struct GpuPrefillExecutor {
    pub(super) boundary_plan: Slice1BoundaryPlan,
}

pub struct PrefillLayerSnapshot {
    pub layer_idx: usize,
    pub hidden_last: Vec<f32>,
    pub cache_layer_idx: usize,
    pub cached_k_last: Vec<f32>,
    pub cached_v_last: Vec<f32>,
}

#[derive(Default)]
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(super) struct DeferredGdnConvStateFlush {
    pub(super) touched_layers: Vec<usize>,
}

#[derive(Default)]
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(super) struct DeferredAttentionKvMaterialization {
    pub(super) layer_idx: usize,
    pub(super) pos_start: usize,
    pub(super) kv_len: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) pending: bool,
}

impl DeferredGdnConvStateFlush {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(super) fn mark_touched(&mut self, layer_idx: usize) {
        if !self.touched_layers.contains(&layer_idx) {
            self.touched_layers.push(layer_idx);
        }
    }
}

#[cfg(feature = "vulkan")]
impl DeferredGdnConvStateFlush {
    pub(super) fn flush_into_kv_cache_untracked(
        &mut self,
        kv_cache: &mut KVCache,
        vk: &mut backend_runtime::GpuRuntime,
    ) -> crate::error::Result<usize> {
        let mut total_bytes = 0usize;
        let mut flushed = Vec::with_capacity(self.touched_layers.len());
        for &layer_idx in &self.touched_layers {
            let conv_state_len = kv_cache
                .get_ssm_state(layer_idx)
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "SSM state not initialized for layer {layer_idx}"
                    ))
                })?
                .conv_state
                .len();
            let mut conv_state = vec![0.0f32; conv_state_len];
            total_bytes += backend_runtime::materialize_gdn_conv_state_untracked(
                vk,
                layer_idx,
                &mut conv_state,
            )?;
            flushed.push((layer_idx, conv_state));
        }
        for (layer_idx, conv_state) in flushed {
            kv_cache
                .get_ssm_state_mut(layer_idx)
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "SSM state not initialized for layer {layer_idx}"
                    ))
                })?
                .conv_state
                .copy_from_slice(&conv_state);
        }
        self.touched_layers.clear();
        Ok(total_bytes)
    }
}

impl DeferredAttentionKvMaterialization {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(super) fn mark_touched(
        &mut self,
        layer_idx: usize,
        pos_start: usize,
        kv_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.layer_idx = layer_idx;
        self.pos_start = pos_start;
        self.kv_len = kv_len;
        self.num_kv_heads = num_kv_heads;
        self.head_dim = head_dim;
        self.pending = true;
    }
}

#[cfg(feature = "vulkan")]
impl DeferredAttentionKvMaterialization {
    pub(super) fn flush_into_kv_cache_untracked(
        &mut self,
        kv_cache: &mut KVCache,
        vk: &mut backend_runtime::GpuRuntime,
    ) -> crate::error::Result<usize> {
        if !self.pending || self.kv_len == 0 {
            return Ok(0);
        }

        let ((k_bits, v_bits), total_bytes) =
            backend_runtime::materialize_attention_kv_range_untracked(
                vk,
                self.layer_idx,
                self.num_kv_heads,
                self.pos_start,
                self.kv_len,
                self.head_dim,
            )?;

        kv_cache.replace_layer_f16_range(
            self.layer_idx,
            self.pos_start,
            self.kv_len,
            &k_bits,
            &v_bits,
        );
        self.pending = false;
        Ok(total_bytes)
    }
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_attention_kv_into_cache_untracked(
    kv_cache: &mut KVCache,
    vk: &mut backend_runtime::GpuRuntime,
    layer_idx: usize,
    num_kv_heads: usize,
    total_tokens: usize,
    head_dim: usize,
    kv_dim: usize,
) -> crate::error::Result<()> {
    let (k_bits, v_bits) = backend_runtime::materialize_attention_kv_for_layer(
        vk,
        layer_idx,
        num_kv_heads,
        total_tokens,
        head_dim,
        kv_dim,
    )?;
    kv_cache.replace_layer_f16(layer_idx, total_tokens, &k_bits, &v_bits);
    Ok(())
}
