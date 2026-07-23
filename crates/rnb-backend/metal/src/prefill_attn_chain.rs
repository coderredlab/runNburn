//! pm48 ②: prefill attention 2차 device-resident chain.
//!
//! 1차(host 입출력 flash seam)는 layer 당 q upload(seq*q_dim f32) + KV upload + out download +
//! CPU rope/qk_norm 가 GPU 커널 3.5x 이득을 잠식해 27B에서 −0.03%(tie)였다. 2차는 rope/qk_norm
//! 과 flash 를 **단일 command buffer + device buffer 공유**로 묶어 (1) CPU rope/qk_norm 제거,
//! (2) q 의 rope 후 재upload 제거, (3) 중간 q_normed/k_normed device 잔류를 달성한다.
//!
//! 입력(host): q_proj 출력(gate split 후, norm 전) f32, k_proj 출력(norm 전) f32, v f32,
//!   q_norm/k_norm weight f32. 출력(host readback): attn_out f32 + k_normed f16 bits + v f16 bits
//!   (host KV cache 가 decode 에서 source-of-truth 라 device 에서 만든 f16 KV 를 1회 readback).
//!
//! gdn_conv_delta_chain.rs 의 carrier(shape별 1회 alloc) + 단일 encoder(RAW hazard 자동 직렬화)
//! 패턴을 그대로 따른다.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::compute::{
    encode_cast_f32_to_f16, encode_flash_attn_prefill, encode_prefill_rope_qk_norm, MetalContext,
};
use crate::ffn_chain::{empty_f16_buf_with_zeroed_tail, empty_f32_buf, shared_f32_buf, u32_buf};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillAttnChainTrace {
    pub layer_idx: usize,
    pub timing_enabled: bool,
}

/// prefill attention chain 의 device-resident 중간 buffer + scalar. shape 별 1회 alloc 후 재사용.
pub(crate) struct PrefillAttnChainCarrier {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    // device 중간 buffer.
    q_in_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*q_dim] q_proj(split 후, norm 전)
    k_in_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] k_proj(norm 전)
    v_in_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] v_proj(f32)
    q_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*q_dim] rope+qk_norm 후
    k_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] rope+qk_norm 후
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] half (flash KV + cache)
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] half
    q_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [head_dim]
    k_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [head_dim]
    out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,  // [seq*q_dim] attn_out f32
    // scalar buffer.
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_len_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pos_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*kv_dim] cast n
    scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillAttnChainCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        theta: f32,
        eps: f32,
        pos_start: usize,
        scale: f32,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let kv_len = seq_len; // prefill: pos_start==0, kv_len == seq_len
        let kv_elems = seq_len * kv_dim;
        let padded_kv_elems = seq_len.next_multiple_of(64) * kv_dim;
        let f32_scalar = |v: f32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            shared_f32_buf(ctx, std::slice::from_ref(&v))
        };
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            q_in_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_in_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            v_in_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            q_normed_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_normed_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            k_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            v_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            q_norm_w_dev: empty_f32_buf(ctx, head_dim),
            k_norm_w_dev: empty_f32_buf(ctx, head_dim),
            out_dev: empty_f32_buf(ctx, seq_len * q_dim),
            nh_buf: u32_buf(ctx, num_heads as u32),
            nkv_buf: u32_buf(ctx, num_kv_heads as u32),
            hd_buf: u32_buf(ctx, head_dim as u32),
            kv_len_buf: u32_buf(ctx, kv_len as u32),
            seq_buf: u32_buf(ctx, seq_len as u32),
            nrot_buf: u32_buf(ctx, n_rot as u32),
            theta_buf: f32_scalar(theta),
            eps_buf: f32_scalar(eps),
            pos_buf: u32_buf(ctx, pos_start as u32),
            kv_elems_buf: u32_buf(ctx, (seq_len * kv_dim) as u32),
            scale_buf: f32_scalar(scale),
        }
    }

    fn upload(&self, q_in: &[f32], k_in: &[f32], v_in: &[f32], q_norm_w: &[f32], k_norm_w: &[f32]) {
        let cp = |src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>| unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.contents().as_ptr() as *mut f32,
                src.len(),
            );
        };
        debug_assert_eq!(q_in.len(), self.seq_len * self.q_dim, "q_in len");
        debug_assert_eq!(k_in.len(), self.seq_len * self.kv_dim, "k_in len");
        debug_assert_eq!(v_in.len(), self.seq_len * self.kv_dim, "v_in len");
        debug_assert_eq!(q_norm_w.len(), self.head_dim, "q_norm_w len");
        debug_assert_eq!(k_norm_w.len(), self.head_dim, "k_norm_w len");
        cp(q_in, &self.q_in_dev);
        cp(k_in, &self.k_in_dev);
        cp(v_in, &self.v_in_dev);
        cp(q_norm_w, &self.q_norm_w_dev);
        cp(k_norm_w, &self.k_norm_w_dev);
    }
}

/// pm48 ②: rope/qk_norm(q) → rope/qk_norm(k) → cast(k_normed→f16) → cast(v→f16) → flash 를
/// 단일 command buffer 에 연속 encode. RAW hazard 자동 직렬화. host 왕복 1회(upload → commit/wait
/// → attn_out + k_f16 + v_f16 readback). 반환 `(attn_out[seq*q_dim], k_f16[seq*kv_dim], v_f16)`.
///
/// CPU ground-truth 동등성: q/k 의 rope/qk_norm 은 prefill_rope_qk_norm 커널(oracle 4.95e-5,
/// 순서 qk_norm→rope, n_rot partial)이 projection.rs qk_norm + rope.rs rope_mrope_text 와 1:1.
/// flash 는 1차 seam 과 동일 커널(27B argmax token-identical 검증됨). v 는 norm 없음(GDN 아닌
/// 표준 attention v) — 호출자가 v_norm/v_rotation 미적용 layer 만 진입시킨다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAttnChainCarrier,
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    trace: Option<PrefillAttnChainTrace>,
) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
    let total0 = std::time::Instant::now();
    let upload0 = std::time::Instant::now();
    carrier.upload(q_in, k_in, v_in, q_norm_w, k_norm_w);
    let upload_ms = upload0.elapsed().as_secs_f64() * 1000.0;
    let seq = carrier.seq_len;
    let kv_elems = seq * carrier.kv_dim;

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder"); // 단일 encoder = RAW 자동 직렬화

    // 1) rope+qk_norm(q): q_in → q_normed (per-head, num_heads).
    encode_prefill_rope_qk_norm(
        ctx,
        &enc,
        &carrier.q_in_dev,
        &carrier.q_norm_w_dev,
        &carrier.q_normed_dev,
        &carrier.nh_buf,
        &carrier.hd_buf,
        &carrier.nrot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_buf,
        seq,
        carrier.num_heads,
    );
    // 2) rope+qk_norm(k): k_in → k_normed (per-head, num_kv_heads).
    encode_prefill_rope_qk_norm(
        ctx,
        &enc,
        &carrier.k_in_dev,
        &carrier.k_norm_w_dev,
        &carrier.k_normed_dev,
        &carrier.nkv_buf,
        &carrier.hd_buf,
        &carrier.nrot_buf,
        &carrier.theta_buf,
        &carrier.eps_buf,
        &carrier.pos_buf,
        seq,
        carrier.num_kv_heads,
    );
    // 3) cast k_normed f32 → f16 (flash KV + host cache).
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.k_normed_dev,
        &carrier.k_f16_dev,
        &carrier.kv_elems_buf,
        kv_elems,
    );
    // 4) cast v f32 → f16.
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.v_in_dev,
        &carrier.v_f16_dev,
        &carrier.kv_elems_buf,
        kv_elems,
    );
    // 5) flash: q_normed(device f32) + k_f16/v_f16(device half) → out_dev.
    encode_flash_attn_prefill(
        ctx,
        &enc,
        &carrier.q_normed_dev,
        &carrier.k_f16_dev,
        &carrier.v_f16_dev,
        &carrier.out_dev,
        &carrier.nh_buf,
        &carrier.nkv_buf,
        &carrier.kv_len_buf,
        &carrier.seq_buf,
        &carrier.scale_buf,
        carrier.num_heads,
        seq,
    );

    enc.endEncoding();
    let wait0 = std::time::Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let commit_wait_ms = wait0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;

    let rb_attn0 = std::time::Instant::now();
    let attn_out = {
        let c = carrier.out_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, seq * carrier.q_dim) }
            .to_vec()
    };
    let readback_attn_ms = rb_attn0.elapsed().as_secs_f64() * 1000.0;
    let rb_k0 = std::time::Instant::now();
    let k_f16 = {
        let c = carrier.k_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, kv_elems) }.to_vec()
    };
    let readback_k_ms = rb_k0.elapsed().as_secs_f64() * 1000.0;
    let rb_v0 = std::time::Instant::now();
    let v_f16 = {
        let c = carrier.v_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, kv_elems) }.to_vec()
    };
    let readback_v_ms = rb_v0.elapsed().as_secs_f64() * 1000.0;
    if let Some(trace) = trace.filter(|trace| trace.timing_enabled) {
        eprintln!(
            "[prefill-atn-chain-time] layer={} seq={} heads={} kv_heads={} head_dim={} upload_ms={upload_ms:.3} gpu_ms={gpu_ms:.3} commit_wait_ms={commit_wait_ms:.3} readback_attn_ms={readback_attn_ms:.3} readback_k_ms={readback_k_ms:.3} readback_v_ms={readback_v_ms:.3} total_ms={:.3}",
            trace.layer_idx,
            carrier.seq_len,
            carrier.num_heads,
            carrier.num_kv_heads,
            carrier.head_dim,
            total0.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (attn_out, k_f16, v_f16)
}
