//! pm50 M1: Qwen dense gated attention prefill core carrier.
//!
//! q/k/v projection, q/gate split, qk_norm+RoPE, flash attention, gate apply를
//! 단일 command buffer로 묶고 기존 `PrefillFusedAttention` seam과 같은
//! `(attn_out, k_f16, v_f16)`을 host로 돌려준다.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
};

use crate::compute::{
    self, encode_cast_f32_to_f16, encode_flash_attn_prefill, encode_prefill_gate_apply,
    encode_prefill_rope_qk_norm, encode_prefill_split_q_gate, encode_rms_norm_batch,
    encode_silu_mul_to_f16, MetalContext,
};
use crate::ffn_chain::{
    empty_f16_buf, empty_f16_buf_with_zeroed_tail, empty_f32_buf, f32_buf, readback,
    shared_f32_buf, u32_buf, QwenMoeLlamaIdStage, QwenMoeLlamaIdStageSampler,
};
use crate::{PrefillAtnOTailBackendSpecRef, TensoropsQuant};

pub(crate) struct PrefillAtnCoreCarrier {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_full_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_len_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pos_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    rope_cos_sin_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_cols_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct PrefillAtnFullLayerCarrier {
    pub core: PrefillAtnCoreCarrier,
    pub ffn_dim: usize,
    o_in_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct PrefillAtnOTailCarrier {
    pub core: PrefillAtnCoreCarrier,
    o_in_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

fn prefill_rope_cos_sin(
    seq_len: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    pos_start: usize,
) -> Vec<f32> {
    let n_rot = n_rot.min(head_dim);
    if n_rot == 0 {
        return vec![1.0, 0.0];
    }
    let theta_scale = theta.powf(-2.0f32 / n_rot as f32);
    let mut table = Vec::with_capacity(seq_len * n_rot);
    for token in 0..seq_len {
        let mut angle = (pos_start + token) as f32;
        for _ in 0..(n_rot / 2) {
            table.push(angle.cos());
            table.push(angle.sin());
            angle *= theta_scale;
        }
    }
    table
}

impl PrefillAtnCoreCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
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
            hidden_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            attn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            q_norm_w_dev: empty_f32_buf(ctx, head_dim),
            k_norm_w_dev: empty_f32_buf(ctx, head_dim),
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            q_full_dev: empty_f32_buf(ctx, seq_len * q_dim * 2),
            q_dev: empty_f32_buf(ctx, seq_len * q_dim),
            gate_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            v_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            q_normed_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_normed_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            k_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            v_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            attn_out_dev: empty_f32_buf(ctx, seq_len * q_dim),
            attn_gated_dev: empty_f32_buf(ctx, seq_len * q_dim),
            seq_buf: u32_buf(ctx, seq_len as u32),
            kv_len_buf: u32_buf(ctx, seq_len as u32),
            nh_buf: u32_buf(ctx, num_heads as u32),
            nkv_buf: u32_buf(ctx, num_kv_heads as u32),
            hd_buf: u32_buf(ctx, head_dim as u32),
            nrot_buf: u32_buf(ctx, n_rot as u32),
            theta_buf: f32_buf(ctx, rope_theta),
            eps_buf: f32_buf(ctx, norm_eps),
            pos_buf: u32_buf(ctx, pos_start as u32),
            rope_cos_sin_dev: shared_f32_buf(
                ctx,
                &prefill_rope_cos_sin(seq_len, head_dim, n_rot, rope_theta, pos_start),
            ),
            scale_buf: f32_buf(ctx, scale),
            hidden_cols_buf: u32_buf(ctx, hidden_dim as u32),
            q_n_buf: u32_buf(ctx, (q_dim * 2) as u32),
            kv_n_buf: u32_buf(ctx, kv_dim as u32),
            hidden_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            q_elems_buf: u32_buf(ctx, (seq_len * q_dim) as u32),
            kv_elems_buf: u32_buf(ctx, (seq_len * kv_dim) as u32),
        }
    }

    fn upload(&self, hidden: &[f32], attn_norm_w: &[f32], q_norm_w: &[f32], k_norm_w: &[f32]) {
        debug_assert_eq!(hidden.len(), self.seq_len * self.hidden_dim);
        debug_assert_eq!(attn_norm_w.len(), self.hidden_dim);
        debug_assert_eq!(q_norm_w.len(), self.head_dim);
        debug_assert_eq!(k_norm_w.len(), self.head_dim);
        copy_f32(hidden, &self.hidden_dev);
        copy_f32(attn_norm_w, &self.attn_norm_w_dev);
        copy_f32(q_norm_w, &self.q_norm_w_dev);
        copy_f32(k_norm_w, &self.k_norm_w_dev);
    }
}

impl PrefillAtnFullLayerCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
        let core = PrefillAtnCoreCarrier::new(
            ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        );
        Self {
            core,
            ffn_dim,
            o_in_f16_dev: empty_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            ffn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            ffn_normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            ffn_gate_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_up_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_act_f16_dev: empty_f16_buf(ctx, seq_len * ffn_dim),
            ffn_down_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            ffn_dim_buf: u32_buf(ctx, ffn_dim as u32),
            ffn_elems_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
        }
    }

    fn upload(
        &self,
        hidden: &[f32],
        attn_norm_w: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        ffn_norm_w: &[f32],
    ) {
        self.core.upload(hidden, attn_norm_w, q_norm_w, k_norm_w);
        debug_assert_eq!(ffn_norm_w.len(), self.core.hidden_dim);
        copy_f32(ffn_norm_w, &self.ffn_norm_w_dev);
    }
}

impl PrefillAtnOTailCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
        let core = PrefillAtnCoreCarrier::new(
            ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        );
        Self {
            core,
            o_in_f16_dev: empty_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
        }
    }

    fn upload_hidden(&self, hidden: &[f32]) {
        debug_assert_eq!(hidden.len(), self.core.seq_len * self.core.hidden_dim);
        copy_f32(hidden, &self.core.hidden_dev);
    }
}

fn copy_f32(src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.contents().as_ptr() as *mut f32, src.len());
    }
}

fn ensure_command_completed(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), String> {
    let status = cmd.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let error = cmd
        .error()
        .map(|err| format!("{err:?}"))
        .unwrap_or_else(|| "no NSError attached".to_string());
    Err(format!(
        "Metal prefill ATN core: command buffer failed status={status:?} error={error}"
    ))
}

#[allow(clippy::too_many_arguments)]
fn encode_quant_gemm_v2(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    quant: TensoropsQuant,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    w_off: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    match quant {
        TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q8_0 => compute::encode_gemm_q8_0_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q2K => compute::encode_gemm_q2k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q3K => compute::encode_gemm_q3k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct PrefillAtnCoreDispatchRequest<'a> {
    pub hidden: &'a [f32],
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub q_w_off: u32,
    pub q_quant: TensoropsQuant,
    pub k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub k_w_off: u32,
    pub k_quant: TensoropsQuant,
    pub v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub v_w_off: u32,
    pub v_quant: TensoropsQuant,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct PrefillAtnFullLayerDispatchRequest<'a> {
    pub core: PrefillAtnCoreDispatchRequest<'a>,
    pub o_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub o_w_off: u32,
    pub o_quant: TensoropsQuant,
    pub ffn_norm_w: &'a [f32],
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

pub(crate) struct PrefillAtnOTailDispatchRequest<'a> {
    pub hidden: &'a [f32],
    pub spec: PrefillAtnOTailBackendSpecRef<'a>,
}

#[derive(Clone, Copy)]
struct PrefillAtnCoreOpsWeights<'a> {
    q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    q_w_off: u32,
    q_quant: TensoropsQuant,
    k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    k_w_off: u32,
    k_quant: TensoropsQuant,
    v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    v_w_off: u32,
    v_quant: TensoropsQuant,
}

impl<'a> PrefillAtnCoreOpsWeights<'a> {
    fn from_dispatch(req: &PrefillAtnCoreDispatchRequest<'a>) -> Self {
        Self {
            q_w_buf: req.q_w_buf,
            q_w_off: req.q_w_off,
            q_quant: req.q_quant,
            k_w_buf: req.k_w_buf,
            k_w_off: req.k_w_off,
            k_quant: req.k_quant,
            v_w_buf: req.v_w_buf,
            v_w_off: req.v_w_off,
            v_quant: req.v_quant,
        }
    }
}

#[derive(Clone, Copy)]
enum PrefillAtnNormMode {
    LegacyTree,
    Exact { eps: f32, n_rot: usize },
}

fn encode_atn_core_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnCoreCarrier,
    weights: PrefillAtnCoreOpsWeights<'_>,
    attn_norm_w: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w: &ProtocolObject<dyn MTLBuffer>,
    norm_mode: PrefillAtnNormMode,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    mut stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.begin(enc, QwenMoeLlamaIdStage::Map);
    }
    match norm_mode {
        PrefillAtnNormMode::LegacyTree => {
            encode_rms_norm_batch(
                ctx,
                enc,
                hidden_in,
                attn_norm_w,
                &carrier.normed_dev,
                &carrier.hidden_cols_buf,
                &carrier.eps_buf,
                carrier.seq_len,
            );
            compute::chain_barrier(ctx, enc);
        }
        PrefillAtnNormMode::Exact { eps, .. } => {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                hidden_in,
                attn_norm_w,
                &carrier.normed_dev,
                carrier.seq_len,
                carrier.hidden_dim,
                eps,
            )
            .map_err(|error| {
                format!("Metal prefill ATN exact residual RMS norm failed: {error:?}")
            })?;
        }
    }
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.normed_dev,
        &carrier.normed_f16_dev,
        &carrier.hidden_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Gate);
    }
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.q_quant,
        weights.q_w_buf,
        weights.q_w_off,
        &carrier.normed_f16_dev,
        &carrier.q_full_dev,
        &carrier.q_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.q_dim * 2,
        carrier.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.k_quant,
        weights.k_w_buf,
        weights.k_w_off,
        &carrier.normed_f16_dev,
        &carrier.k_dev,
        &carrier.kv_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.kv_dim,
        carrier.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.v_quant,
        weights.v_w_buf,
        weights.v_w_off,
        &carrier.normed_f16_dev,
        &carrier.v_dev,
        &carrier.kv_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.kv_dim,
        carrier.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Up);
    }
    encode_prefill_split_q_gate(
        ctx,
        enc,
        &carrier.q_full_dev,
        &carrier.q_dev,
        &carrier.gate_dev,
        &carrier.seq_buf,
        &carrier.nh_buf,
        &carrier.hd_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    match norm_mode {
        PrefillAtnNormMode::LegacyTree => {
            encode_prefill_rope_qk_norm(
                ctx,
                enc,
                &carrier.q_dev,
                q_norm_w,
                &carrier.q_normed_dev,
                &carrier.nh_buf,
                &carrier.hd_buf,
                &carrier.nrot_buf,
                &carrier.theta_buf,
                &carrier.eps_buf,
                &carrier.pos_buf,
                carrier.seq_len,
                carrier.num_heads,
            );
            encode_prefill_rope_qk_norm(
                ctx,
                enc,
                &carrier.k_dev,
                k_norm_w,
                &carrier.k_normed_dev,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &carrier.nrot_buf,
                &carrier.theta_buf,
                &carrier.eps_buf,
                &carrier.pos_buf,
                carrier.seq_len,
                carrier.num_kv_heads,
            );
            compute::chain_barrier(ctx, enc);
        }
        PrefillAtnNormMode::Exact { eps, n_rot } => {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &carrier.q_dev,
                q_norm_w,
                &carrier.q_normed_dev,
                carrier.seq_len * carrier.num_heads,
                carrier.head_dim,
                eps,
            )
            .map_err(|error| format!("Metal prefill ATN exact q RMS norm failed: {error:?}"))?;
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &carrier.k_dev,
                k_norm_w,
                &carrier.k_normed_dev,
                carrier.seq_len * carrier.num_kv_heads,
                carrier.head_dim,
                eps,
            )
            .map_err(|error| format!("Metal prefill ATN exact k RMS norm failed: {error:?}"))?;
            compute::encode_prefill_rope_only(
                ctx,
                enc,
                &carrier.q_normed_dev,
                &carrier.q_normed_dev,
                &carrier.rope_cos_sin_dev,
                carrier.num_heads,
                carrier.head_dim,
                n_rot,
                carrier.seq_len,
            )
            .map_err(|error| format!("Metal prefill ATN q RoPE failed: {error:?}"))?;
            compute::encode_prefill_rope_only(
                ctx,
                enc,
                &carrier.k_normed_dev,
                &carrier.k_normed_dev,
                &carrier.rope_cos_sin_dev,
                carrier.num_kv_heads,
                carrier.head_dim,
                n_rot,
                carrier.seq_len,
            )
            .map_err(|error| format!("Metal prefill ATN k RoPE failed: {error:?}"))?;
        }
    }
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Activation);
    }
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.k_normed_dev,
        kv_out.0,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.v_dev,
        kv_out.1,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Down);
    }
    encode_flash_attn_prefill(
        ctx,
        enc,
        &carrier.q_normed_dev,
        kv_out.0,
        kv_out.1,
        &carrier.attn_out_dev,
        &carrier.nh_buf,
        &carrier.nkv_buf,
        &carrier.kv_len_buf,
        &carrier.seq_buf,
        &carrier.scale_buf,
        carrier.num_heads,
        carrier.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Reduce);
    }
    encode_prefill_gate_apply(
        ctx,
        enc,
        &carrier.attn_out_dev,
        &carrier.gate_dev,
        &carrier.attn_gated_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_atn_o_tail_bound_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    weights: PrefillAtnCoreOpsWeights<'_>,
    attn_norm_w: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w: &ProtocolObject<dyn MTLBuffer>,
    norm_mode: PrefillAtnNormMode,
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_w_off: u32,
    o_quant: TensoropsQuant,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    let core = &carrier.core;
    encode_atn_core_ops(
        ctx,
        enc,
        core,
        weights,
        attn_norm_w,
        q_norm_w,
        k_norm_w,
        norm_mode,
        hidden_in,
        kv_out,
        stage_sampler,
    )?;
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &core.attn_gated_dev,
        &carrier.o_in_f16_dev,
        &core.q_elems_buf,
        core.seq_len * core.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    encode_quant_gemm_v2(
        ctx,
        enc,
        o_quant,
        o_w_buf,
        o_w_off,
        &carrier.o_in_f16_dev,
        hidden_out,
        &core.hidden_cols_buf,
        &carrier.q_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    crate::ffn_chain::encode_residual_add(
        ctx,
        enc,
        hidden_out,
        hidden_in,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    compute::chain_barrier(ctx, enc);
    Ok(())
}

pub(crate) fn encode_prefill_atn_o_tail_ops_profiled(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    spec: PrefillAtnOTailBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    let core = &carrier.core;
    let spec_core = spec.core;
    let carrier_shape = (
        core.seq_len,
        core.num_heads,
        core.num_kv_heads,
        core.head_dim,
        core.hidden_dim,
        core.q_dim,
        core.kv_dim,
    );
    let spec_shape = (
        spec_core.seq_len,
        spec_core.num_heads,
        spec_core.num_kv_heads,
        spec_core.head_dim,
        spec_core.hidden_dim,
        spec_core.q_dim,
        spec_core.kv_dim,
    );
    if carrier_shape != spec_shape {
        return Err(format!(
            "Metal prefill ATN o-tail: carrier/spec shape mismatch: carrier={carrier_shape:?} spec={spec_shape:?}"
        ));
    }
    if spec_core.attn_norm_w.len() != core.hidden_dim
        || spec_core.q_norm_w.len() != core.head_dim
        || spec_core.k_norm_w.len() != core.head_dim
    {
        return Err(format!(
            "Metal prefill ATN o-tail: norm length mismatch: attn={} q={} k={} expected={}/{}/{}",
            spec_core.attn_norm_w.len(),
            spec_core.q_norm_w.len(),
            spec_core.k_norm_w.len(),
            core.hidden_dim,
            core.head_dim,
            core.head_dim,
        ));
    }
    let hidden_bytes = core
        .seq_len
        .checked_mul(core.hidden_dim)
        .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Metal prefill ATN o-tail: hidden buffer size overflow".to_string())?;
    let kv_bytes = core
        .seq_len
        .checked_mul(core.kv_dim)
        .and_then(|len| len.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| "Metal prefill ATN o-tail: KV buffer size overflow".to_string())?;
    if hidden_in.length() < hidden_bytes || hidden_out.length() < hidden_bytes {
        return Err(format!(
            "Metal prefill ATN o-tail: hidden buffer too small: in={} out={} required={hidden_bytes}",
            hidden_in.length(),
            hidden_out.length(),
        ));
    }
    if kv_out.0.length() < kv_bytes || kv_out.1.length() < kv_bytes {
        return Err(format!(
            "Metal prefill ATN o-tail: KV buffer too small: k={} v={} required={kv_bytes}",
            kv_out.0.length(),
            kv_out.1.length(),
        ));
    }

    // 한 command buffer에서 큰 shape carrier를 여러 layer가 공유하므로 norm source는
    // 호출별 immutable buffer로 유지해 뒤 layer의 CPU 준비가 앞 dispatch를 덮지 않게 한다.
    let attn_norm_w = shared_f32_buf(ctx, spec_core.attn_norm_w);
    let q_norm_w = shared_f32_buf(ctx, spec_core.q_norm_w);
    let k_norm_w = shared_f32_buf(ctx, spec_core.k_norm_w);
    let (q_w_buf, q_w_off) = compute::wrap_nocopy(ctx, spec_core.q_weight.raw);
    let (k_w_buf, k_w_off) = compute::wrap_nocopy(ctx, spec_core.k_weight.raw);
    let (v_w_buf, v_w_off) = compute::wrap_nocopy(ctx, spec_core.v_weight.raw);
    let (o_w_buf, o_w_off) = compute::wrap_nocopy(ctx, spec.o_weight.raw);

    encode_prefill_atn_o_tail_bound_ops(
        ctx,
        enc,
        carrier,
        PrefillAtnCoreOpsWeights {
            q_w_buf: &q_w_buf,
            q_w_off,
            q_quant: spec_core.q_weight.quant,
            k_w_buf: &k_w_buf,
            k_w_off,
            k_quant: spec_core.k_weight.quant,
            v_w_buf: &v_w_buf,
            v_w_off,
            v_quant: spec_core.v_weight.quant,
        },
        &attn_norm_w,
        &q_norm_w,
        &k_norm_w,
        PrefillAtnNormMode::Exact {
            eps: spec_core.norm_eps,
            n_rot: spec_core.n_rot,
        },
        &o_w_buf,
        o_w_off,
        spec.o_weight.quant,
        hidden_in,
        hidden_out,
        kv_out,
        stage_sampler,
    )?;
    Ok(())
}

pub(crate) fn encode_prefill_atn_o_tail_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    spec: PrefillAtnOTailBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
) -> Result<(), String> {
    encode_prefill_atn_o_tail_ops_profiled(
        ctx, enc, carrier, spec, hidden_in, hidden_out, kv_out, None,
    )
}

pub(crate) fn prefill_atn_core_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnCoreCarrier,
    req: PrefillAtnCoreDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    carrier.upload(req.hidden, req.attn_norm_w, req.q_norm_w, req.k_norm_w);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal prefill ATN core: command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal prefill ATN core: compute encoder creation failed".to_string())?;

    encode_atn_core_ops(
        ctx,
        &enc,
        carrier,
        PrefillAtnCoreOpsWeights::from_dispatch(&req),
        &carrier.attn_norm_w_dev,
        &carrier.q_norm_w_dev,
        &carrier.k_norm_w_dev,
        PrefillAtnNormMode::LegacyTree,
        &carrier.hidden_dev,
        (&carrier.k_f16_dev, &carrier.v_f16_dev),
        None,
    )?;

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let attn_out = {
        let c = carrier.attn_gated_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const f32, carrier.seq_len * carrier.q_dim)
        }
        .to_vec()
    };
    let k_bits = {
        let c = carrier.k_f16_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u16, carrier.seq_len * carrier.kv_dim)
        }
        .to_vec()
    };
    let v_bits = {
        let c = carrier.v_f16_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u16, carrier.seq_len * carrier.kv_dim)
        }
        .to_vec()
    };
    Ok((attn_out, k_bits, v_bits))
}

pub(crate) fn prefill_atn_o_tail_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnOTailCarrier,
    req: PrefillAtnOTailDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let core = &carrier.core;
    carrier.upload_hidden(req.hidden);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal prefill ATN o-tail: command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal prefill ATN o-tail: compute encoder creation failed".to_string())?;

    encode_prefill_atn_o_tail_ops(
        ctx,
        &enc,
        carrier,
        req.spec,
        &core.hidden_dev,
        &carrier.o_proj_dev,
        (&core.k_f16_dev, &core.v_f16_dev),
    )?;

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let hidden = readback(&carrier.o_proj_dev, core.seq_len * core.hidden_dim);
    let k_bits = {
        let c = core.k_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    let v_bits = {
        let c = core.v_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    Ok((hidden, k_bits, v_bits))
}

pub(crate) fn prefill_atn_full_layer_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnFullLayerCarrier,
    req: PrefillAtnFullLayerDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let core = &carrier.core;
    carrier.upload(
        req.core.hidden,
        req.core.attn_norm_w,
        req.core.q_norm_w,
        req.core.k_norm_w,
        req.ffn_norm_w,
    );

    let cmd = ctx.queue.commandBuffer().ok_or_else(|| {
        "Metal prefill ATN full layer: command buffer creation failed".to_string()
    })?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
        "Metal prefill ATN full layer: compute encoder creation failed".to_string()
    })?;

    encode_atn_core_ops(
        ctx,
        &enc,
        core,
        PrefillAtnCoreOpsWeights::from_dispatch(&req.core),
        &core.attn_norm_w_dev,
        &core.q_norm_w_dev,
        &core.k_norm_w_dev,
        PrefillAtnNormMode::LegacyTree,
        &core.hidden_dev,
        (&core.k_f16_dev, &core.v_f16_dev),
        None,
    )?;

    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &core.attn_gated_dev,
        &carrier.o_in_f16_dev,
        &core.q_elems_buf,
        core.seq_len * core.q_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.o_quant,
        req.o_w_buf,
        req.o_w_off,
        &carrier.o_in_f16_dev,
        &carrier.o_proj_dev,
        &core.hidden_cols_buf,
        &carrier.q_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.o_proj_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );

    encode_rms_norm_batch(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.ffn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &core.hidden_cols_buf,
        &core.eps_buf,
        core.seq_len,
    );
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.ffn_normed_dev,
        &carrier.ffn_normed_f16_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_gate_quant,
        req.ffn_gate_w_buf,
        req.ffn_gate_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_gate_dev,
        &carrier.ffn_dim_buf,
        &core.hidden_cols_buf,
        &core.seq_buf,
        carrier.ffn_dim,
        core.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_up_quant,
        req.ffn_up_w_buf,
        req.ffn_up_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_dim_buf,
        &core.hidden_cols_buf,
        &core.seq_buf,
        carrier.ffn_dim,
        core.seq_len,
    );
    encode_silu_mul_to_f16(
        ctx,
        &enc,
        &carrier.ffn_gate_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_elems_buf,
        core.seq_len * carrier.ffn_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_down_quant,
        req.ffn_down_w_buf,
        req.ffn_down_w_off,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_down_dev,
        &core.hidden_cols_buf,
        &carrier.ffn_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.ffn_down_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let hidden = readback(&core.hidden_dev, core.seq_len * core.hidden_dim);
    let k_bits = {
        let c = core.k_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    let v_bits = {
        let c = core.v_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    Ok((hidden, k_bits, v_bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_cpu::kernels::{norm::rms_norm_into, rope::rope_partial_inplace};

    fn assert_f32_bits(label: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{label} length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{label} bit mismatch at {index}: got={got:?} expected={expected:?}"
            );
        }
    }

    // RoPE 회전 적용은 Metal 커널이 FMA(fma)로 융합하는 반면 CPU 참조는 분리된
    // mul/sub 라 몇 ULP 어긋난다(cos/sin 테이블 자체는 CPU 계산으로 bit-identical).
    // 따라서 회전 결과는 bit-exact 가 아니라 tight tolerance 로 검증한다.
    fn assert_f32_close(label: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{label} length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            let tol = 1.0e-5 * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= tol,
                "{label} mismatch at {index}: got={got:?} expected={expected:?} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_prefill_exact_qk_norm_and_rope_match_cpu_bits() {
        const SEQ_LEN: usize = 4;
        const HEAD_DIM: usize = 256;
        const N_ROT: usize = 64;
        const THETA: f32 = 10_000_000.0;
        const EPS: f32 = 1.0e-6;

        let ctx = compute::build_metal_context().expect("Metal context");
        for (num_heads, pos_start, label) in [(16usize, 0usize, "q"), (2usize, 4552usize, "k")] {
            let cols = HEAD_DIM;
            let rows = SEQ_LEN * num_heads;
            let len = rows * cols;
            let input: Vec<f32> = (0..len)
                .map(|index| {
                    let centered = (index.wrapping_mul(37) % 257) as f32 - 128.0;
                    centered * 0.000_976_562_5
                })
                .collect();
            let weight: Vec<f32> = (0..cols)
                .map(|index| 0.75 + (index % 17) as f32 * 0.015_625)
                .collect();
            let mut norm_expected = vec![0.0f32; len];
            rms_norm_into(&input, &weight, EPS, &mut norm_expected);
            let mut rope_expected = norm_expected.clone();
            rope_partial_inplace(
                &mut rope_expected,
                pos_start,
                HEAD_DIM,
                num_heads * HEAD_DIM,
                N_ROT,
                THETA,
            );

            let input_dev = shared_f32_buf(&ctx, &input);
            let weight_dev = shared_f32_buf(&ctx, &weight);
            let norm_dev = empty_f32_buf(&ctx, len);
            let rope_dev = empty_f32_buf(&ctx, len);
            let cmd = ctx.queue.commandBuffer().expect("command buffer");
            let rope_cos_sin_dev = shared_f32_buf(
                &ctx,
                &prefill_rope_cos_sin(SEQ_LEN, HEAD_DIM, N_ROT, THETA, pos_start),
            );
            let enc = compute::try_chain_compute_encoder(&ctx, &cmd).expect("compute encoder");
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                &ctx,
                &enc,
                &input_dev,
                &weight_dev,
                &norm_dev,
                rows,
                cols,
                EPS,
            )
            .expect("exact RMS encode");
            compute::encode_prefill_rope_only(
                &ctx,
                &enc,
                &norm_dev,
                &rope_dev,
                &rope_cos_sin_dev,
                num_heads,
                HEAD_DIM,
                N_ROT,
                SEQ_LEN,
            )
            .expect("RoPE-only encode");
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            ensure_command_completed(&cmd).expect("command completed");

            assert_f32_bits(
                &format!("{label} qk norm"),
                &readback(&norm_dev, len),
                &norm_expected,
            );
            assert_f32_close(
                &format!("{label} RoPE"),
                &readback(&rope_dev, len),
                &rope_expected,
            );
        }
    }
}
