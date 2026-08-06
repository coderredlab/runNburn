use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use super::{
    compute, copy_f32, empty_f32_buf, encode_gemma_full_layer_tail, encode_gemma_qkv_o,
    ensure_command_completed, readback_f32, readback_u16, GemmaPrefillFfnDispatchSpec,
    GemmaPrefillFullLayerCarrier, GemmaPrefillQkvODispatchSpec, MetalContext,
};

pub(crate) struct GemmaPrefillLayerRangeLayerDispatchRequest<'a> {
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

pub(crate) fn prefill_gemma_layer_range_dispatch(
    ctx: &MetalContext,
    state: &GemmaPrefillLayerRangeState,
    carrier: &GemmaPrefillFullLayerCarrier,
    req: GemmaPrefillLayerRangeLayerDispatchRequest<'_>,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let attention = &carrier.attention;
    if state.hidden_elements != attention.seq_len * attention.hidden_dim {
        return Err("Metal Gemma prefill range: hidden shape changed between layers".to_string());
    }
    attention.upload_weights(
        req.attention.q_norm_w,
        req.attention.k_norm_w,
        req.attention.rope_freq_factors,
    );
    carrier.upload_weights(
        Some(req.attn_norm_w),
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
        &carrier.attn_norm_w_dev,
        &attention.normed_dev,
        &attention.hidden_dim_buf,
        &attention.eps_buf,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_gemma_qkv_o(ctx, &encoder, attention, &req.attention);
    encode_gemma_full_layer_tail(
        ctx,
        &encoder,
        carrier,
        &state.hidden_dev,
        &req.ffn,
        req.out_scale.is_some(),
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    ensure_command_completed(&command)?;

    Ok((
        readback_u16(&attention.k_f16_dev, attention.seq_len * attention.kv_dim),
        readback_u16(&attention.v_f16_dev, attention.seq_len * attention.kv_dim),
    ))
}
