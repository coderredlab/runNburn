//! Gemma 4 prefill attention carrier.
//!
//! Every layer owns Q/K/V. This carrier keeps normalized-input cast, the three
//! projections, Q/K norm + NeoX RoPE, flash attention, and O projection in one
//! command buffer. Only O output and the K/V rows required by the host cache
//! cross back after the command completes.

mod layer_range;

pub(crate) use layer_range::{
    prefill_gemma_layer_range_complete, prefill_gemma_layer_range_submit,
    GemmaPrefillLayerRangeDispatchRequest, GemmaPrefillLayerRangeLayerDispatchRequest,
    GemmaPrefillLayerRangePending, GemmaPrefillLayerRangeState,
};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};
use std::cell::OnceCell;

use crate::compute::{
    self, encode_cast_f32_to_f16, encode_flash_attn_prefill_gemma, encode_prefill_neox_qk_norm,
    MetalContext,
};
use crate::ffn_chain::{
    empty_f16_buf, empty_f16_buf_with_zeroed_tail, empty_f32_buf, f32_buf, private_f16_buf,
    private_f32_buf, shared_f32_buf, u32_buf,
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

pub(crate) struct GemmaPrefillFullLayerCarrier {
    pub attention: GemmaPrefillQkvOTailCarrier,
    ffn_dim: usize,
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_attn_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_ffn_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    range_alt: OnceCell<GemmaPrefillLayerRangeBuffers>,
}

#[derive(Clone, Copy)]
pub(crate) struct GemmaPrefillAttentionBufferRefs<'a> {
    q_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    rope_freq_factors_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[derive(Clone, Copy)]
pub(crate) struct GemmaPrefillLayerRangeBufferRefs<'a> {
    attention: GemmaPrefillAttentionBufferRefs<'a>,
    attn_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    post_attn_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    post_ffn_norm_w_dev: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    out_scale_buf: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
}

struct GemmaPrefillLayerRangeBuffers {
    q_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    rope_freq_factors_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl GemmaPrefillLayerRangeBuffers {
    fn new(ctx: &MetalContext, attention: &GemmaPrefillQkvOTailCarrier) -> Self {
        let kv_elems = attention.seq_len * attention.kv_dim;
        let padded_kv_elems = attention.seq_len.next_multiple_of(64) * attention.kv_dim;
        Self {
            q_norm_w_dev: empty_f32_buf(ctx, attention.head_dim),
            k_norm_w_dev: empty_f32_buf(ctx, attention.head_dim),
            rope_freq_factors_dev: shared_f32_buf(ctx, &vec![1.0; attention.head_dim / 2]),
            k_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            v_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            attn_norm_w_dev: empty_f32_buf(ctx, attention.hidden_dim),
            post_attn_norm_w_dev: empty_f32_buf(ctx, attention.hidden_dim),
            ffn_norm_w_dev: empty_f32_buf(ctx, attention.hidden_dim),
            post_ffn_norm_w_dev: empty_f32_buf(ctx, attention.hidden_dim),
            out_scale_buf: f32_buf(ctx, 1.0),
        }
    }

    fn refs(&self) -> GemmaPrefillLayerRangeBufferRefs<'_> {
        GemmaPrefillLayerRangeBufferRefs {
            attention: GemmaPrefillAttentionBufferRefs {
                q_norm_w_dev: &self.q_norm_w_dev,
                k_norm_w_dev: &self.k_norm_w_dev,
                rope_freq_factors_dev: &self.rope_freq_factors_dev,
                k_f16_dev: &self.k_f16_dev,
                v_f16_dev: &self.v_f16_dev,
            },
            attn_norm_w_dev: &self.attn_norm_w_dev,
            post_attn_norm_w_dev: &self.post_attn_norm_w_dev,
            ffn_norm_w_dev: &self.ffn_norm_w_dev,
            post_ffn_norm_w_dev: &self.post_ffn_norm_w_dev,
            out_scale_buf: &self.out_scale_buf,
        }
    }
}

impl GemmaPrefillLayerRangeBufferRefs<'_> {
    fn upload(
        &self,
        attn_norm_w: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        rope_freq_factors: Option<&[f32]>,
        post_attn_norm_w: &[f32],
        ffn_norm_w: &[f32],
        post_ffn_norm_w: &[f32],
        out_scale: Option<f32>,
    ) {
        copy_f32(attn_norm_w, self.attn_norm_w_dev);
        copy_f32(q_norm_w, self.attention.q_norm_w_dev);
        copy_f32(k_norm_w, self.attention.k_norm_w_dev);
        if let Some(factors) = rope_freq_factors {
            copy_f32(factors, self.attention.rope_freq_factors_dev);
        } else {
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.attention
                        .rope_freq_factors_dev
                        .contents()
                        .as_ptr()
                        .cast::<f32>(),
                    q_norm_w.len() / 2,
                )
                .fill(1.0);
            }
        }
        copy_f32(post_attn_norm_w, self.post_attn_norm_w_dev);
        copy_f32(ffn_norm_w, self.ffn_norm_w_dev);
        copy_f32(post_ffn_norm_w, self.post_ffn_norm_w_dev);
        if let Some(out_scale) = out_scale {
            copy_f32(std::slice::from_ref(&out_scale), self.out_scale_buf);
        }
    }
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

    fn upload_weights(
        &self,
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        rope_freq_factors: Option<&[f32]>,
    ) {
        debug_assert_eq!(q_norm_w.len(), self.head_dim);
        debug_assert_eq!(k_norm_w.len(), self.head_dim);
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

    fn upload(
        &self,
        normed: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        rope_freq_factors: Option<&[f32]>,
    ) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        copy_f32(normed, &self.normed_dev);
        self.upload_weights(q_norm_w, k_norm_w, rope_freq_factors);
    }

    fn primary_buffers(&self) -> GemmaPrefillAttentionBufferRefs<'_> {
        GemmaPrefillAttentionBufferRefs {
            q_norm_w_dev: &self.q_norm_w_dev,
            k_norm_w_dev: &self.k_norm_w_dev,
            rope_freq_factors_dev: &self.rope_freq_factors_dev,
            k_f16_dev: &self.k_f16_dev,
            v_f16_dev: &self.v_f16_dev,
        }
    }
}

impl GemmaPrefillFullLayerCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        ffn_dim: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
    ) -> Self {
        Self {
            attention: GemmaPrefillQkvOTailCarrier::new(
                ctx,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_dim,
                q_dim,
                rope_theta,
                scale,
                norm_eps,
            ),
            ffn_dim,
            hidden_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            attn_norm_w_dev: shared_f32_buf(ctx, &vec![0.0; hidden_dim]),
            post_attn_norm_w_dev: shared_f32_buf(ctx, &vec![0.0; hidden_dim]),
            post_attn_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            ffn_norm_w_dev: shared_f32_buf(ctx, &vec![0.0; hidden_dim]),
            ffn_normed_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            ffn_normed_f16_dev: private_f16_buf(ctx, seq_len * hidden_dim),
            ffn_gate_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            ffn_up_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            ffn_act_f16_dev: private_f16_buf(ctx, seq_len * ffn_dim),
            ffn_down_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            post_ffn_norm_w_dev: shared_f32_buf(ctx, &vec![0.0; hidden_dim]),
            post_ffn_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            out_scale_buf: f32_buf(ctx, 1.0),
            ffn_dim_buf: u32_buf(ctx, ffn_dim as u32),
            ffn_elems_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
            range_alt: OnceCell::new(),
        }
    }

    fn upload_weights(
        &self,
        attn_norm_w: Option<&[f32]>,
        post_attn_norm_w: &[f32],
        ffn_norm_w: &[f32],
        post_ffn_norm_w: &[f32],
        out_scale: Option<f32>,
    ) {
        if let Some(attn_norm_w) = attn_norm_w {
            debug_assert_eq!(attn_norm_w.len(), self.attention.hidden_dim);
            copy_f32(attn_norm_w, &self.attn_norm_w_dev);
        }
        debug_assert_eq!(post_attn_norm_w.len(), self.attention.hidden_dim);
        debug_assert_eq!(ffn_norm_w.len(), self.attention.hidden_dim);
        debug_assert_eq!(post_ffn_norm_w.len(), self.attention.hidden_dim);
        copy_f32(post_attn_norm_w, &self.post_attn_norm_w_dev);
        copy_f32(ffn_norm_w, &self.ffn_norm_w_dev);
        copy_f32(post_ffn_norm_w, &self.post_ffn_norm_w_dev);
        if let Some(out_scale) = out_scale {
            copy_f32(std::slice::from_ref(&out_scale), &self.out_scale_buf);
        }
    }

    fn upload(
        &self,
        hidden: &[f32],
        post_attn_norm_w: &[f32],
        ffn_norm_w: &[f32],
        post_ffn_norm_w: &[f32],
    ) {
        debug_assert_eq!(
            hidden.len(),
            self.attention.seq_len * self.attention.hidden_dim
        );
        copy_f32(hidden, &self.hidden_dev);
        self.upload_weights(None, post_attn_norm_w, ffn_norm_w, post_ffn_norm_w, None);
    }

    pub(crate) fn range_buffers(
        &self,
        ctx: &MetalContext,
        slot: usize,
    ) -> GemmaPrefillLayerRangeBufferRefs<'_> {
        if slot == 0 {
            GemmaPrefillLayerRangeBufferRefs {
                attention: self.attention.primary_buffers(),
                attn_norm_w_dev: &self.attn_norm_w_dev,
                post_attn_norm_w_dev: &self.post_attn_norm_w_dev,
                ffn_norm_w_dev: &self.ffn_norm_w_dev,
                post_ffn_norm_w_dev: &self.post_ffn_norm_w_dev,
                out_scale_buf: &self.out_scale_buf,
            }
        } else {
            self.range_alt
                .get_or_init(|| GemmaPrefillLayerRangeBuffers::new(ctx, &self.attention))
                .refs()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct GemmaPrefillQkvODispatchSpec<'a> {
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

pub(crate) struct GemmaPrefillQkvOTailDispatchRequest<'a> {
    pub normed: &'a [f32],
    pub spec: GemmaPrefillQkvODispatchSpec<'a>,
}

pub(crate) struct GemmaPrefillFfnDispatchSpec<'a> {
    pub ffn_gate_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_gate_w_off: u32,
    pub ffn_gate_quant: TensoropsQuant,
    pub ffn_up_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_up_w_off: u32,
    pub ffn_up_quant: TensoropsQuant,
    pub ffn_down_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_down_w_off: u32,
    pub ffn_down_quant: TensoropsQuant,
}

pub(crate) struct GemmaPrefillFullLayerDispatchRequest<'a> {
    pub attention: GemmaPrefillQkvOTailDispatchRequest<'a>,
    pub hidden: &'a [f32],
    pub post_attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub post_ffn_norm_w: &'a [f32],
    pub ffn: GemmaPrefillFfnDispatchSpec<'a>,
}

fn encode_gemma_qkv_o(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &GemmaPrefillQkvOTailCarrier,
    buffers: GemmaPrefillAttentionBufferRefs<'_>,
    req: &GemmaPrefillQkvODispatchSpec<'_>,
) {
    encode_cast_f32_to_f16(
        ctx,
        encoder,
        &carrier.normed_dev,
        &carrier.normed_f16_dev,
        &carrier.normed_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier(ctx, encoder);
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
            encoder,
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
    compute::chain_barrier(ctx, encoder);
    encode_prefill_neox_qk_norm(
        ctx,
        encoder,
        &carrier.q_dev,
        buffers.q_norm_w_dev,
        &carrier.q_rope_dev,
        &carrier.num_heads_buf,
        &carrier.head_dim_buf,
        &carrier.n_rot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_start_buf,
        buffers.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_heads,
    );
    encode_prefill_neox_qk_norm(
        ctx,
        encoder,
        &carrier.k_dev,
        buffers.k_norm_w_dev,
        &carrier.k_rope_dev,
        &carrier.num_kv_heads_buf,
        &carrier.head_dim_buf,
        &carrier.n_rot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_start_buf,
        buffers.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_kv_heads,
    );
    encode_prefill_neox_qk_norm(
        ctx,
        encoder,
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
        buffers.rope_freq_factors_dev,
        carrier.seq_len,
        carrier.num_kv_heads,
    );
    compute::chain_barrier(ctx, encoder);
    encode_cast_f32_to_f16(
        ctx,
        encoder,
        &carrier.k_rope_dev,
        buffers.k_f16_dev,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    encode_cast_f32_to_f16(
        ctx,
        encoder,
        &carrier.v_norm_dev,
        buffers.v_f16_dev,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    compute::chain_barrier(ctx, encoder);
    encode_flash_attn_prefill_gemma(
        ctx,
        encoder,
        &carrier.q_rope_dev,
        buffers.k_f16_dev,
        buffers.v_f16_dev,
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
    compute::chain_barrier(ctx, encoder);
    encode_cast_f32_to_f16(
        ctx,
        encoder,
        &carrier.attn_out_dev,
        &carrier.attn_out_f16_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, encoder);
    encode_quant_gemm_v2(
        ctx,
        encoder,
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
}

pub(crate) fn prefill_gemma_qkv_o_tail_dispatch(
    ctx: &MetalContext,
    carrier: &GemmaPrefillQkvOTailCarrier,
    req: GemmaPrefillQkvOTailDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    carrier.upload(
        req.normed,
        req.spec.q_norm_w,
        req.spec.k_norm_w,
        req.spec.rope_freq_factors,
    );

    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal Gemma prefill carrier: command buffer creation failed".to_string())?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| "Metal Gemma prefill carrier: encoder creation failed".to_string())?;
    encode_gemma_qkv_o(ctx, &encoder, carrier, carrier.primary_buffers(), &req.spec);
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

fn encode_gemma_full_layer_tail(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &GemmaPrefillFullLayerCarrier,
    buffers: GemmaPrefillLayerRangeBufferRefs<'_>,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn: &GemmaPrefillFfnDispatchSpec<'_>,
    apply_out_scale: bool,
) {
    let attention = &carrier.attention;
    compute::chain_barrier(ctx, encoder);
    compute::encode_rms_norm_batch(
        ctx,
        encoder,
        &attention.o_proj_dev,
        buffers.post_attn_norm_w_dev,
        &carrier.post_attn_dev,
        &attention.hidden_dim_buf,
        &attention.eps_buf,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, encoder);
    crate::ffn_chain::encode_residual_add(
        ctx,
        encoder,
        hidden_dev,
        &carrier.post_attn_dev,
        &attention.normed_elems_buf,
        attention.seq_len * attention.hidden_dim,
    );
    compute::chain_barrier(ctx, encoder);
    compute::encode_rms_norm_batch(
        ctx,
        encoder,
        hidden_dev,
        buffers.ffn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &attention.hidden_dim_buf,
        &attention.eps_buf,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, encoder);
    encode_cast_f32_to_f16(
        ctx,
        encoder,
        &carrier.ffn_normed_dev,
        &carrier.ffn_normed_f16_dev,
        &attention.normed_elems_buf,
        attention.seq_len * attention.hidden_dim,
    );
    compute::chain_barrier(ctx, encoder);
    encode_quant_gemm_v2(
        ctx,
        encoder,
        ffn.ffn_gate_quant,
        ffn.ffn_gate_w_buf,
        ffn.ffn_gate_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_gate_dev,
        &carrier.ffn_dim_buf,
        &attention.hidden_dim_buf,
        &attention.seq_buf,
        carrier.ffn_dim,
        attention.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        encoder,
        ffn.ffn_up_quant,
        ffn.ffn_up_w_buf,
        ffn.ffn_up_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_dim_buf,
        &attention.hidden_dim_buf,
        &attention.seq_buf,
        carrier.ffn_dim,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, encoder);
    compute::encode_gelu_mul_to_f16(
        ctx,
        encoder,
        &carrier.ffn_gate_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_elems_buf,
        attention.seq_len * carrier.ffn_dim,
    );
    compute::chain_barrier(ctx, encoder);
    encode_quant_gemm_v2(
        ctx,
        encoder,
        ffn.ffn_down_quant,
        ffn.ffn_down_w_buf,
        ffn.ffn_down_w_off,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_down_dev,
        &attention.hidden_dim_buf,
        &carrier.ffn_dim_buf,
        &attention.seq_buf,
        attention.hidden_dim,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, encoder);
    compute::encode_rms_norm_batch(
        ctx,
        encoder,
        &carrier.ffn_down_dev,
        buffers.post_ffn_norm_w_dev,
        &carrier.post_ffn_dev,
        &attention.hidden_dim_buf,
        &attention.eps_buf,
        attention.seq_len,
    );
    compute::chain_barrier(ctx, encoder);
    if apply_out_scale {
        crate::ffn_chain::encode_residual_add_scaled(
            ctx,
            encoder,
            hidden_dev,
            &carrier.post_ffn_dev,
            &attention.normed_elems_buf,
            buffers.out_scale_buf,
            attention.seq_len * attention.hidden_dim,
        );
    } else {
        crate::ffn_chain::encode_residual_add(
            ctx,
            encoder,
            hidden_dev,
            &carrier.post_ffn_dev,
            &attention.normed_elems_buf,
            attention.seq_len * attention.hidden_dim,
        );
    }
}

pub(crate) fn prefill_gemma_full_layer_dispatch(
    ctx: &MetalContext,
    carrier: &GemmaPrefillFullLayerCarrier,
    req: GemmaPrefillFullLayerDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let attention = &carrier.attention;
    attention.upload(
        req.attention.normed,
        req.attention.spec.q_norm_w,
        req.attention.spec.k_norm_w,
        req.attention.spec.rope_freq_factors,
    );
    carrier.upload(
        req.hidden,
        req.post_attn_norm_w,
        req.ffn_norm_w,
        req.post_ffn_norm_w,
    );

    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        "Metal Gemma prefill full layer: command buffer creation failed".to_string()
    })?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| "Metal Gemma prefill full layer: encoder creation failed".to_string())?;
    let buffers = carrier.range_buffers(ctx, 0);
    encode_gemma_qkv_o(
        ctx,
        &encoder,
        attention,
        buffers.attention,
        &req.attention.spec,
    );
    encode_gemma_full_layer_tail(
        ctx,
        &encoder,
        carrier,
        buffers,
        &carrier.hidden_dev,
        &req.ffn,
        false,
    );

    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    ensure_command_completed(&command)?;
    Ok((
        readback_f32(
            &carrier.hidden_dev,
            attention.seq_len * attention.hidden_dim,
        ),
        readback_u16(&attention.k_f16_dev, attention.seq_len * attention.kv_dim),
        readback_u16(&attention.v_f16_dev, attention.seq_len * attention.kv_dim),
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
