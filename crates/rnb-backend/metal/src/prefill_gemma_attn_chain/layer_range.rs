use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use super::{
    compute, copy_f32, empty_f32_buf, encode_gemma_full_layer_tail, encode_gemma_qkv_o,
    ensure_command_completed, readback_f32, GemmaPrefillFfnDispatchSpec,
    GemmaPrefillFullLayerCarrier, GemmaPrefillQkvODispatchSpec, MetalContext,
};

pub(crate) struct GemmaPrefillLayerRangeLayerDispatchRequest<'a> {
    pub layer_idx: usize,
    pub attn_norm_w: &'a [f32],
    pub attention: GemmaPrefillQkvODispatchSpec<'a>,
    pub post_attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub post_ffn_norm_w: &'a [f32],
    pub out_scale: Option<f32>,
    pub ffn: GemmaPrefillFfnDispatchSpec<'a>,
}

pub(crate) struct GemmaPrefillLayerRangeDispatchRequest<'a> {
    pub hidden: &'a [f32],
}

pub(crate) struct GemmaPrefillLayerRangeState {
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elements: usize,
}

impl GemmaPrefillLayerRangeState {
    pub(crate) fn new(ctx: &MetalContext, req: GemmaPrefillLayerRangeDispatchRequest<'_>) -> Self {
        let hidden_dev = empty_f32_buf(ctx, req.hidden.len());
        copy_f32(req.hidden, &hidden_dev);
        Self {
            hidden_dev,
            hidden_elements: req.hidden.len(),
        }
    }

    pub(crate) fn finish(&self) -> Vec<f32> {
        readback_f32(&self.hidden_dev, self.hidden_elements)
    }
}

pub(crate) struct GemmaPrefillLayerRangePending {
    layer_idx: usize,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_len: usize,
    completed: bool,
}

impl Drop for GemmaPrefillLayerRangePending {
    fn drop(&mut self) {
        if !self.completed {
            self.command.waitUntilCompleted();
        }
    }
}

pub(crate) fn prefill_gemma_layer_range_submit(
    ctx: &MetalContext,
    state: &GemmaPrefillLayerRangeState,
    carrier: &GemmaPrefillFullLayerCarrier,
    slot: usize,
    req: GemmaPrefillLayerRangeLayerDispatchRequest<'_>,
) -> Result<GemmaPrefillLayerRangePending, String> {
    let attention = &carrier.attention;
    if state.hidden_elements != attention.seq_len * attention.hidden_dim {
        return Err("Metal Gemma prefill range: hidden shape changed between layers".to_string());
    }
    let buffers = carrier.range_buffers(ctx, slot);
    buffers.upload(
        req.attn_norm_w,
        req.attention.q_norm_w,
        req.attention.k_norm_w,
        req.attention.rope_freq_factors,
        req.post_attn_norm_w,
        req.ffn_norm_w,
        req.post_ffn_norm_w,
        req.out_scale,
    );
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal Gemma prefill range: command buffer creation failed".to_string())?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| "Metal Gemma prefill range: encoder creation failed".to_string())?;
    compute::encode_rms_norm_batch(
        ctx,
        &encoder,
        &state.hidden_dev,
        buffers.attn_norm_w_dev,
        &attention.normed_dev,
        &attention.hidden_dim_buf,
        &attention.eps_buf,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_gemma_qkv_o(
        ctx,
        &encoder,
        attention,
        buffers.attention,
        &req.attention,
        None,
        true,
        0,
        &attention.seq_buf,
    );
    encode_gemma_full_layer_tail(
        ctx,
        &encoder,
        carrier,
        buffers,
        &state.hidden_dev,
        &req.ffn,
        req.out_scale.is_some(),
    );
    encoder.endEncoding();
    command.commit();

    Ok(GemmaPrefillLayerRangePending {
        layer_idx: req.layer_idx,
        command,
        k_f16_dev: buffers.attention.k_f16_dev.clone(),
        v_f16_dev: buffers.attention.v_f16_dev.clone(),
        kv_len: attention.seq_len * attention.kv_dim,
        completed: false,
    })
}

pub(crate) fn prefill_gemma_layer_range_complete(
    mut pending: GemmaPrefillLayerRangePending,
    on_kv: &mut impl FnMut(usize, &[u16], &[u16]) -> Result<(), String>,
) -> Result<(), String> {
    pending.command.waitUntilCompleted();
    pending.completed = true;
    ensure_command_completed(&pending.command)?;
    let k_bits = unsafe {
        std::slice::from_raw_parts(
            pending.k_f16_dev.contents().as_ptr().cast::<u16>(),
            pending.kv_len,
        )
    };
    let v_bits = unsafe {
        std::slice::from_raw_parts(
            pending.v_f16_dev.contents().as_ptr().cast::<u16>(),
            pending.kv_len,
        )
    };
    on_kv(pending.layer_idx, k_bits, v_bits)
}
