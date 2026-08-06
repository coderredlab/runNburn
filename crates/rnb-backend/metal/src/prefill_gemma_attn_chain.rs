//! Gemma 4 prefill attention carrier.
//!
//! Every layer owns Q/K/V. This carrier keeps normalized-input cast, the three
//! projections, Q/K norm + NeoX RoPE, flash attention, and O projection in one
//! command buffer. Only O output and the K/V rows required by the host cache
//! cross back after the command completes.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::compute::{
    self, encode_cast_f32_to_f16, encode_flash_attn_prefill_gemma, encode_prefill_neox_qk_norm,
    MetalContext,
};
use crate::ffn_chain::{
    empty_f16_buf, empty_f16_buf_with_zeroed_tail, empty_f32_buf, f32_buf, shared_f32_buf, u32_buf,
};
use crate::prefill_atn_core_chain::{encode_quant_gemm_v2, ensure_command_completed};
use crate::TensoropsQuant;

pub(crate) struct GemmaPrefillQkvOTailCarrier {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    rope_freq_factors_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_rope_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_rope_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    num_heads_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    num_kv_heads_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    head_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_rot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    zero_rot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pos_start_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl GemmaPrefillQkvOTailCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let kv_elems = seq_len * kv_dim;
        let padded_kv_elems = seq_len.next_multiple_of(64) * kv_dim;
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            q_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_dev: empty_f32_buf(ctx, kv_elems),
            v_dev: empty_f32_buf(ctx, kv_elems),
            q_norm_w_dev: empty_f32_buf(ctx, head_dim),
            k_norm_w_dev: empty_f32_buf(ctx, head_dim),
            v_norm_w_dev: shared_f32_buf(ctx, &vec![1.0; head_dim]),
            rope_freq_factors_dev: shared_f32_buf(ctx, &vec![1.0; head_dim / 2]),
            q_rope_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_rope_dev: empty_f32_buf(ctx, kv_elems),
            v_norm_dev: empty_f32_buf(ctx, kv_elems),
            k_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            v_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            attn_out_dev: empty_f32_buf(ctx, seq_len * q_dim),
            attn_out_f16_dev: empty_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            seq_buf: u32_buf(ctx, seq_len as u32),
            num_heads_buf: u32_buf(ctx, num_heads as u32),
            num_kv_heads_buf: u32_buf(ctx, num_kv_heads as u32),
            head_dim_buf: u32_buf(ctx, head_dim as u32),
            n_rot_buf: u32_buf(ctx, head_dim as u32),
            zero_rot_buf: u32_buf(ctx, 0),
            theta_buf: f32_buf(ctx, rope_theta),
            eps_buf: f32_buf(ctx, norm_eps),
            pos_start_buf: u32_buf(ctx, 0),
            scale_buf: f32_buf(ctx, scale),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            kv_dim_buf: u32_buf(ctx, kv_dim as u32),
            normed_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            q_elems_buf: u32_buf(ctx, (seq_len * q_dim) as u32),
            kv_elems_buf: u32_buf(ctx, kv_elems as u32),
        }
    }

    fn upload(
        &self,
        normed: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        rope_freq_factors: Option<&[f32]>,
    ) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        debug_assert_eq!(q_norm_w.len(), self.head_dim);
        debug_assert_eq!(k_norm_w.len(), self.head_dim);
        copy_f32(normed, &self.normed_dev);
        copy_f32(q_norm_w, &self.q_norm_w_dev);
        copy_f32(k_norm_w, &self.k_norm_w_dev);
        if let Some(factors) = rope_freq_factors {
            debug_assert_eq!(factors.len(), self.head_dim / 2);
            copy_f32(factors, &self.rope_freq_factors_dev);
        } else {
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.rope_freq_factors_dev.contents().as_ptr().cast::<f32>(),
                    self.head_dim / 2,
                )
                .fill(1.0);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct GemmaPrefillQkvOTailDispatchRequest<'a> {
    pub normed: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub q_w_off: u32,
    pub rope_freq_factors: Option<&'a [f32]>,
    pub v_from_k: bool,
    pub q_quant: TensoropsQuant,
    pub k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub k_w_off: u32,
    pub k_quant: TensoropsQuant,
    pub v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub v_w_off: u32,
    pub v_quant: TensoropsQuant,
    pub o_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub o_w_off: u32,
    pub o_quant: TensoropsQuant,
    pub sliding_window: Option<usize>,
    pub softcap: Option<f32>,
}

pub(crate) fn prefill_gemma_qkv_o_tail_dispatch(
    ctx: &MetalContext,
    carrier: &GemmaPrefillQkvOTailCarrier,
    req: GemmaPrefillQkvOTailDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    carrier.upload(
        req.normed,
        req.q_norm_w,
        req.k_norm_w,
        req.rope_freq_factors,
    );

    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal Gemma prefill carrier: command buffer creation failed".to_string())?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| "Metal Gemma prefill carrier: encoder creation failed".to_string())?;

    encode_cast_f32_to_f16(
        ctx,
        &encoder,
        &carrier.normed_dev,
        &carrier.normed_f16_dev,
        &carrier.normed_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier(ctx, &encoder);
    for (enabled, quant, weight, offset, output, output_dim, output_dim_buf) in [
        (
            true,
            req.q_quant,
            req.q_w_buf,
            req.q_w_off,
            &carrier.q_dev,
            carrier.q_dim,
            &carrier.q_dim_buf,
        ),
        (
            true,
            req.k_quant,
            req.k_w_buf,
            req.k_w_off,
            &carrier.k_dev,
            carrier.kv_dim,
            &carrier.kv_dim_buf,
        ),
        (
            !req.v_from_k,
            req.v_quant,
            req.v_w_buf,
            req.v_w_off,
            &carrier.v_dev,
            carrier.kv_dim,
            &carrier.kv_dim_buf,
        ),
    ] {
        if !enabled {
            continue;
        }
        encode_quant_gemm_v2(
            ctx,
            &encoder,
            quant,
            weight,
            offset,
            &carrier.normed_f16_dev,
            output,
            output_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.seq_buf,
            output_dim,
            carrier.seq_len,
        );
    }
    compute::chain_barrier(ctx, &encoder);
    encode_prefill_neox_qk_norm(
        ctx,
        &encoder,
        &carrier.q_dev,
        &carrier.q_norm_w_dev,
        &carrier.q_rope_dev,
        &carrier.num_heads_buf,
        &carrier.head_dim_buf,
        &carrier.n_rot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_start_buf,
        &carrier.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_heads,
    );
    encode_prefill_neox_qk_norm(
        ctx,
        &encoder,
        &carrier.k_dev,
        &carrier.k_norm_w_dev,
        &carrier.k_rope_dev,
        &carrier.num_kv_heads_buf,
        &carrier.head_dim_buf,
        &carrier.n_rot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_start_buf,
        &carrier.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_kv_heads,
    );
    encode_prefill_neox_qk_norm(
        ctx,
        &encoder,
        if req.v_from_k {
            &carrier.k_dev
        } else {
            &carrier.v_dev
        },
        &carrier.v_norm_w_dev,
        &carrier.v_norm_dev,
        &carrier.num_kv_heads_buf,
        &carrier.head_dim_buf,
        &carrier.zero_rot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_start_buf,
        &carrier.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_kv_heads,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_cast_f32_to_f16(
        ctx,
        &encoder,
        &carrier.k_rope_dev,
        &carrier.k_f16_dev,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    encode_cast_f32_to_f16(
        ctx,
        &encoder,
        &carrier.v_norm_dev,
        &carrier.v_f16_dev,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_flash_attn_prefill_gemma(
        ctx,
        &encoder,
        &carrier.q_rope_dev,
        &carrier.k_f16_dev,
        &carrier.v_f16_dev,
        &carrier.attn_out_dev,
        &carrier.num_heads_buf,
        &carrier.num_kv_heads_buf,
        &carrier.seq_buf,
        &carrier.seq_buf,
        &carrier.scale_buf,
        req.sliding_window,
        req.softcap,
        carrier.head_dim,
        carrier.num_heads,
        carrier.seq_len,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_cast_f32_to_f16(
        ctx,
        &encoder,
        &carrier.attn_out_dev,
        &carrier.attn_out_f16_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, &encoder);
    encode_quant_gemm_v2(
        ctx,
        &encoder,
        req.o_quant,
        req.o_w_buf,
        req.o_w_off,
        &carrier.attn_out_f16_dev,
        &carrier.o_proj_dev,
        &carrier.hidden_dim_buf,
        &carrier.q_dim_buf,
        &carrier.seq_buf,
        carrier.hidden_dim,
        carrier.seq_len,
    );

    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    ensure_command_completed(&command)?;
    Ok((
        readback_f32(&carrier.o_proj_dev, carrier.seq_len * carrier.hidden_dim),
        readback_u16(&carrier.k_f16_dev, carrier.seq_len * carrier.kv_dim),
        readback_u16(&carrier.v_f16_dev, carrier.seq_len * carrier.kv_dim),
    ))
}

fn copy_f32(src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            dst.contents().as_ptr().cast::<f32>(),
            src.len(),
        );
    }
}

fn readback_f32(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<f32>(), len).to_vec() }
}

fn readback_u16(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<u16>(), len).to_vec() }
}
