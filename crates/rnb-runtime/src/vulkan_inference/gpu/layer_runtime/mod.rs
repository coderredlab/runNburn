use crate::vulkan_backend::WeightId;

use super::{Quant, RuntimeCounters};

// Re-exports of backend-vulkan types that callers (rnb-llm engine wiring,
// mv27-task10b-4c-2b) need to construct fullpath inputs. We keep the
// re-export at the rnb-runtime level so rnb-llm doesn't reach into
// `rnb_backend_vulkan::*` paths directly.
//
// `LayerWeightHandles` and `kv_head_shard_byte_range` are intentionally NOT
// re-exported: they are wrapper internals. The wrapper's contract is "rnb-llm
// hands raw weight bytes; the wrapper does the per-kv-head sharding and
// `LayerWeightHandles` assembly internally." Re-exporting them would invite
// external callers to construct handles or compute shard byte ranges by
// hand, breaking that abstraction. If 4c-2b actually needs them later, the
// re-exports can be added back deliberately.
pub use rnb_backend_vulkan::full_path::{
    FullPathDecodeStepInput, FullPathDecodeStepOutput, FullPathPrefillInput, FullPathPrefillOutput,
};
pub use rnb_backend_vulkan::kv_resident::KvResidentLayout;
pub use rnb_backend_vulkan::staging::StagingPolicy;
pub use rnb_backend_vulkan::{GpuBuffer, GpuWeightMode, QuantType};
pub use rnb_loader::ModelLayerKind;

mod fullpath;

pub use fullpath::{AttentionRawWeights, GdnRawWeights, LayerRawWeights};

pub struct LayerRuntime {
    inner: crate::vulkan_backend::PrefillLayerRuntime,
    /// Tracks whether `bind_token_embd` has already been called on this
    /// runtime. The Q6_K embed table is meant to be uploaded once at engine
    /// init; a second call with different bytes is a silent re-upload that
    /// destroys + recreates the host-visible buffer in the backend, which
    /// usually indicates a bug at the call site (e.g. a re-init loop). We
    /// emit a debug log if it ever happens — no behavior change.
    ///
    /// `Engine::fullpath_token_embd_bound` (in rnb-llm) already gates the
    /// normal init path so the second call is unreachable from `forward_*`.
    /// This flag is a second line of defense for any caller that bypasses
    /// that gate (test harness, future re-init that resets only the engine
    /// flag, weights-replaced path); don't delete one without the other.
    token_embd_bound: bool,
}

impl LayerRuntime {
    pub(super) fn from_backend(inner: crate::vulkan_backend::PrefillLayerRuntime) -> Self {
        Self {
            inner,
            token_embd_bound: false,
        }
    }

    pub fn runtime_counters(&self) -> RuntimeCounters {
        self.inner.runtime_counters()
    }

    pub fn reset_runtime_counters(&mut self) {
        self.inner.reset_runtime_counters();
    }

    pub fn clear_sequence_state(&mut self) -> Result<(), String> {
        self.inner.clear_sequence_state()
    }

    pub fn gemv(
        &mut self,
        id: WeightId,
        raw_bytes: &[u8],
        rows: usize,
        cols: usize,
        quant: Quant,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .gemv(id, raw_bytes, rows, cols, quant, input, output)
    }

    pub fn gemv_multi(
        &mut self,
        input: &[f32],
        weights: &[(WeightId, &[u8], usize, usize, Quant)],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), String> {
        self.inner.gemv_multi(input, weights, outputs)
    }

    pub fn gemv_multi_async(
        &mut self,
        input: &[f32],
        weights: &[(WeightId, &[u8], usize, usize, Quant)],
    ) -> Result<(), String> {
        self.inner.gemv_multi_async(input, weights)
    }

    pub fn wait_async(&mut self, outputs: &mut [&mut [f32]]) -> Result<(), String> {
        self.inner.wait_async(outputs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain(
        &mut self,
        hidden: &mut [f32],
        norm_weight: &[f32],
        norm_eps: f32,
        hidden_dim: usize,
        gate_id: WeightId,
        gate_raw: &[u8],
        gate_rows: usize,
        gate_cols: usize,
        gate_quant: Quant,
        up_id: WeightId,
        up_raw: &[u8],
        up_rows: usize,
        up_cols: usize,
        up_quant: Quant,
        down_id: WeightId,
        down_raw: &[u8],
        down_rows: usize,
        down_cols: usize,
        down_quant: Quant,
    ) -> Result<(), String> {
        self.inner.ffn_chain(
            hidden,
            norm_weight,
            norm_eps,
            hidden_dim,
            gate_id,
            gate_raw,
            gate_rows,
            gate_cols,
            gate_quant,
            up_id,
            up_raw,
            up_rows,
            up_cols,
            up_quant,
            down_id,
            down_raw,
            down_rows,
            down_cols,
            down_quant,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_block_window_for_layer(
        &mut self,
        layer: u16,
        input_all: &[f32],
        cols: usize,
        q_raw: &[u8],
        q_rows: usize,
        q_quant: Quant,
        k_raw: &[u8],
        k_rows: usize,
        k_quant: Quant,
        v_raw: &[u8],
        v_rows: usize,
        v_quant: Quant,
        pos_start: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        o_raw: &[u8],
        o_rows: usize,
        o_cols: usize,
        o_quant: Quant,
        residual_all: &[f32],
        ffn_norm_weight: &[f32],
        norm_eps: f32,
        gate_id: WeightId,
        gate_raw: &[u8],
        gate_rows: usize,
        gate_cols: usize,
        gate_quant: Quant,
        up_id: WeightId,
        up_raw: &[u8],
        up_rows: usize,
        up_cols: usize,
        up_quant: Quant,
        down_id: WeightId,
        down_raw: &[u8],
        down_rows: usize,
        down_cols: usize,
        down_quant: Quant,
        out_all: &mut [f32],
    ) -> Result<(), String> {
        self.inner.attention_block_window_for_layer(
            layer,
            input_all,
            cols,
            q_raw,
            q_rows,
            q_quant,
            k_raw,
            k_rows,
            k_quant,
            v_raw,
            v_rows,
            v_quant,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            o_raw,
            o_rows,
            o_cols,
            o_quant,
            residual_all,
            ffn_norm_weight,
            norm_eps,
            gate_id,
            gate_raw,
            gate_rows,
            gate_cols,
            gate_quant,
            up_id,
            up_raw,
            up_rows,
            up_cols,
            up_quant,
            down_id,
            down_raw,
            down_rows,
            down_cols,
            down_quant,
            out_all,
        )
    }

    pub fn gemv_window(
        &mut self,
        id: WeightId,
        raw_bytes: &[u8],
        rows: usize,
        cols: usize,
        quant: Quant,
        input_all: &[f32],
        output_all: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .gemv_window(id, raw_bytes, rows, cols, quant, input_all, output_all)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_window_with_residual_from_resident_input(
        &mut self,
        out_hidden_all: &mut [f32],
        hidden_dim: usize,
        residual_all: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        gate_id: WeightId,
        gate_raw: &[u8],
        gate_rows: usize,
        gate_cols: usize,
        gate_quant: Quant,
        up_id: WeightId,
        up_raw: &[u8],
        up_rows: usize,
        up_cols: usize,
        up_quant: Quant,
        down_id: WeightId,
        down_raw: &[u8],
        down_rows: usize,
        down_cols: usize,
        down_quant: Quant,
    ) -> Result<(), String> {
        self.inner
            .ffn_chain_window_with_residual_from_resident_input(
                out_hidden_all,
                hidden_dim,
                residual_all,
                norm_weight,
                norm_eps,
                gate_id,
                gate_raw,
                gate_rows,
                gate_cols,
                gate_quant,
                up_id,
                up_raw,
                up_rows,
                up_cols,
                up_quant,
                down_id,
                down_raw,
                down_rows,
                down_cols,
                down_quant,
            )
    }

    pub fn write_gdn_conv_state_f32_for_layer(
        &mut self,
        layer_idx: usize,
        conv_state: &[f32],
    ) -> Result<(), String> {
        self.inner
            .write_gdn_conv_state_f32_for_layer(layer_idx, conv_state)
    }

    pub fn materialize_gdn_conv_state_f32_for_layer(
        &mut self,
        layer_idx: usize,
        conv_state: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .materialize_gdn_conv_state_f32_for_layer(layer_idx, conv_state)
    }

    pub fn materialize_gdn_conv_state_f32_for_layer_untracked(
        &mut self,
        layer_idx: usize,
        conv_state: &mut [f32],
    ) -> Result<usize, String> {
        self.inner
            .materialize_gdn_conv_state_f32_for_layer_untracked(layer_idx, conv_state)
    }

    pub fn record_batched_materialization_download(&mut self, total_bytes: usize) {
        self.inner
            .record_batched_materialization_download(total_bytes);
    }

    pub fn materialize_attention_kv_f16_for_layer(
        &mut self,
        layer_idx: usize,
        count: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        self.inner
            .materialize_attention_kv_f16_for_layer(layer_idx, count)
    }

    pub fn materialize_attention_kv_f16_grouped_for_layer(
        &mut self,
        layer_idx: usize,
        num_kv_heads: usize,
        values_per_head: usize,
        head_dim: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        self.inner.materialize_attention_kv_f16_grouped_for_layer(
            layer_idx,
            num_kv_heads,
            values_per_head,
            head_dim,
        )
    }

    pub fn materialize_attention_kv_f16_range_for_layer_untracked(
        &mut self,
        layer_idx: usize,
        start: usize,
        count: usize,
    ) -> Result<((Vec<u16>, Vec<u16>), usize), String> {
        self.inner
            .materialize_attention_kv_f16_range_for_layer_untracked(layer_idx, start, count)
    }

    pub fn materialize_attention_kv_f16_grouped_range_for_layer_untracked(
        &mut self,
        layer_idx: usize,
        num_kv_heads: usize,
        pos_start: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Result<((Vec<u16>, Vec<u16>), usize), String> {
        self.inner
            .materialize_attention_kv_f16_grouped_range_for_layer_untracked(
                layer_idx,
                num_kv_heads,
                pos_start,
                kv_len,
                head_dim,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_qkv_conv_window_from_resident_state(
        &mut self,
        layer_idx: usize,
        qkv_id: WeightId,
        qkv_raw: &[u8],
        qkv_rows: usize,
        qkv_cols: usize,
        qkv_quant: Quant,
        kernel: &[f32],
        input_all: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        conv_channels: usize,
        kernel_size: usize,
        conv_out: &mut [f32],
    ) -> Result<(), String> {
        self.inner.gdn_qkv_conv_window_from_resident_state(
            layer_idx,
            qkv_id,
            qkv_raw,
            qkv_rows,
            qkv_cols,
            qkv_quant,
            kernel,
            input_all,
            seq_len,
            hidden_dim,
            conv_channels,
            kernel_size,
            conv_out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_window(
        &mut self,
        hidden_all: &mut [f32],
        hidden_dim: usize,
        norm_weight: &[f32],
        norm_eps: f32,
        gate_id: WeightId,
        gate_raw: &[u8],
        gate_rows: usize,
        gate_cols: usize,
        gate_quant: Quant,
        up_id: WeightId,
        up_raw: &[u8],
        up_rows: usize,
        up_cols: usize,
        up_quant: Quant,
        down_id: WeightId,
        down_raw: &[u8],
        down_rows: usize,
        down_cols: usize,
        down_quant: Quant,
    ) -> Result<(), String> {
        self.inner.ffn_chain_window(
            hidden_all,
            hidden_dim,
            norm_weight,
            norm_eps,
            gate_id,
            gate_raw,
            gate_rows,
            gate_cols,
            gate_quant,
            up_id,
            up_raw,
            up_rows,
            up_cols,
            up_quant,
            down_id,
            down_raw,
            down_rows,
            down_cols,
            down_quant,
        )
    }

    pub fn append_attention_kv_f32_for_layer(
        &mut self,
        layer_idx: usize,
        pos: usize,
        k_slice: &[f32],
        v_slice: &[f32],
    ) -> Result<(), String> {
        self.inner
            .append_attention_kv_f32_for_layer(layer_idx, pos, k_slice, v_slice)
    }

    pub fn attention_decode_gpu_kv_mirror_for_layer(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        head_dim: usize,
        kv_len: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .attention_decode_gpu_kv_mirror_for_layer(layer_idx, q, head_dim, kv_len, out)
    }

    pub fn attention_decode_f16_cache(
        &mut self,
        q: &[f32],
        k_cache_f16: &[u16],
        v_cache_f16: &[u16],
        head_dim: usize,
        kv_len: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .attention_decode_f16_cache(q, k_cache_f16, v_cache_f16, head_dim, kv_len, out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_window_grouped_from_mirror_for_layer(
        &mut self,
        layer_idx: usize,
        q_all: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        pos_start: usize,
        out_all: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .attention_decode_window_grouped_from_mirror_for_layer(
                layer_idx,
                q_all,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                pos_start,
                out_all,
            )
    }

    pub fn rms_norm_window(
        &mut self,
        input_all: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        hidden_dim: usize,
        out_all: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .rms_norm_window(input_all, norm_weight, norm_eps, hidden_dim, out_all)
    }
}
