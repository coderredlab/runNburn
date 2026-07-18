//! Attention layer 전체를 단일 command buffer device-resident chain 으로.
//!
//! pm12 carrier: 표준 Qwen3 attention layer(9B 의 8개 attention layer)의
//! `attn_norm → q/k/v GEMV → q/k norm(per-head) → rope → kv_append → attn → o → residual`
//! 을 hidden·KV 전부 device buffer 에 두고 layer 당 command buffer 1개로 돌린다.
//! per-op dispatch overhead·CPU roundtrip 제거 → batch_probe 4.7x 천장 향해.
//!
//! KV: KvResident(device, incremental). prior KV(host f16)는 ensure_filled 로 1회
//! 동기화, 새 토큰은 device append(kv_append). host KV cache 가 source of truth 라
//! fallback 경로와 일관 유지.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

use crate::compute::{
    self, chain_barrier, chain_compute_encoder, encode_attn_decode, encode_attn_decode_i8,
    encode_attn_decode_i8_splitk, encode_gate_apply, encode_kv_append, encode_kv_append_i8,
    encode_qk_norm, encode_rope_partial, encode_split_qgate, KvResident, MetalContext,
};
use crate::ffn_chain::{
    empty_f32_buf, encode_residual_add, encode_rms_norm, encode_silu_mul, f32_buf, readback,
    u32_buf,
};

/// Attention layer 의 device-resident 중간 버퍼 + 불변 scalar 버퍼 + KV.
/// shape 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local.
pub(crate) struct AttnCarrier {
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize, // gated: q_dim*2 ([query|gate] 인터리브). 비-gated면 q_dim.
    pub kv_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub scale: f32,

    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_full_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // q GEMV 출력 [q_out_dim] (인터리브)
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,      // split 후 query [q_dim]
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // split 후 gate [q_dim]
    k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_splitk_acc_dev: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_m_dev: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_s_dev: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_splits_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    o_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv: KvResident,

    hdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    qdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    qoutdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // q GEMV 의 N = q_out_dim
    kvdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // host f32 precompute(theta^(-2/n_rot))
    k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,    // q/k/v GEMV 의 K = hidden_dim
    k_qdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,      // o GEMV 의 K = q_dim

    // FFN (carrier 통합): ffn_norm→gate/up GEMV→silu_mul→down GEMV→residual
    pub ffn_dim: usize,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    fdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // gate/up GEMV 의 N = ffn_dim
    k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // down GEMV 의 K = ffn_dim
}

impl AttnCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden_dim: usize,
        q_dim: usize,
        q_out_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        n_rot: usize,
        capacity: usize,
        ffn_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Self {
        // host `rope_partial_inplace`(rope.rs:400-401)와 동일 식·타입(f32, clamp 후).
        let nr = n_rot.min(head_dim);
        let theta_scale: f32 = theta.powf(-2.0_f32 / nr as f32);
        let splitk_splits = ctx.attn_splitk_splits;
        let (attn_splitk_acc_dev, attn_splitk_m_dev, attn_splitk_s_dev, attn_splitk_splits_buf) =
            if splitk_splits > 1 {
                (
                    Some(empty_f32_buf(ctx, splitk_splits * num_heads * head_dim)),
                    Some(empty_f32_buf(ctx, splitk_splits * num_heads)),
                    Some(empty_f32_buf(ctx, splitk_splits * num_heads)),
                    Some(u32_buf(ctx, splitk_splits as u32)),
                )
            } else {
                (None, None, None, None)
            };
        Self {
            hidden_dim,
            q_dim,
            q_out_dim,
            kv_dim,
            num_heads,
            num_kv_heads,
            scale,
            ffn_dim,
            hidden_dev: empty_f32_buf(ctx, hidden_dim),
            normed_dev: empty_f32_buf(ctx, hidden_dim),
            q_full_dev: empty_f32_buf(ctx, q_out_dim),
            q_dev: empty_f32_buf(ctx, q_dim),
            gate_dev: empty_f32_buf(ctx, q_dim),
            k_dev: empty_f32_buf(ctx, kv_dim),
            v_dev: empty_f32_buf(ctx, kv_dim),
            attn_out_dev: empty_f32_buf(ctx, q_dim),
            attn_splitk_acc_dev,
            attn_splitk_m_dev,
            attn_splitk_s_dev,
            attn_splitk_splits_buf,
            o_out_dev: empty_f32_buf(ctx, hidden_dim),
            kv: KvResident::new(ctx, num_kv_heads, head_dim, capacity),
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            qdim_buf: u32_buf(ctx, q_dim as u32),
            qoutdim_buf: u32_buf(ctx, q_out_dim as u32),
            kvdim_buf: u32_buf(ctx, kv_dim as u32),
            hd_buf: u32_buf(ctx, head_dim as u32),
            eps_buf: f32_buf(ctx, eps),
            nh_buf: u32_buf(ctx, num_heads as u32),
            nkv_buf: u32_buf(ctx, num_kv_heads as u32),
            nrot_buf: u32_buf(ctx, n_rot as u32),
            theta_scale_buf: f32_buf(ctx, theta_scale),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            k_qdim_buf: u32_buf(ctx, q_dim as u32),
            ffn_normed_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_gate_dev: empty_f32_buf(ctx, ffn_dim),
            ffn_up_dev: empty_f32_buf(ctx, ffn_dim),
            ffn_down_dev: empty_f32_buf(ctx, hidden_dim),
            fdim_buf: u32_buf(ctx, ffn_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
        }
    }

    /// chain run(2단계)이 attn layer encode 전 호출 — 첫 token(filled==0)만 prior KV
    /// (host f16, 0..pos)를 device KV 로 1회 init. `attn_chain_dispatch` 의 동일 가드.
    pub(crate) fn kv_ensure_filled(&mut self, prior_k: &[u16], prior_v: &[u16], pos: usize) {
        if self.kv.filled == 0 {
            if self.kv.kv_int8 {
                self.kv.ensure_filled_i8(prior_k, prior_v, pos);
            } else {
                self.kv.ensure_filled(prior_k, prior_v, pos);
            }
        }
    }

    /// chain run commit 후 호출 — kv_append 가 device 에 새 token 을 누적했으므로
    /// filled 를 kv_len(=pos+1)로 갱신(`attn_chain_dispatch` 의 commit 후 처리와 동일).
    pub(crate) fn kv_set_filled(&mut self, kv_len: usize) {
        self.kv.filled = kv_len;
    }

    pub(crate) fn kv_filled(&self) -> usize {
        self.kv.filled
    }

    /// hidden host slice → hidden_dev 업로드(StorageModeShared contents 직접 쓰기).
    fn upload_hidden(&self, hidden: &[f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_dim);
        let contents = self.hidden_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.hidden_dim,
            );
        }
    }
}

/// Attention layer 한 token 실행. weight buffer/offset(NoCopy resident)은 caller 준비.
/// prior_k/prior_v: 이전 토큰들(host f16 bits, [pos*kv_dim]) — KV_dev[0..pos] 동기화용.
/// pos: 새 토큰의 KV 슬롯(= 이전 토큰 수). kv_len = pos+1.
/// 단일 command buffer 단일 encoder 로 ①~⑪. 반환: residual 적용된 hidden(hidden_dim).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &mut AttnCarrier,
    hidden: &[f32],
    norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_off_buf: &ProtocolObject<dyn MTLBuffer>,
    k_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_w_buf: &ProtocolObject<dyn MTLBuffer>,
    v_off_buf: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_is_q6k: bool,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
    prior_k: &[u16],
    prior_v: &[u16],
    pos: usize,
) -> Vec<f32> {
    let hidden_dim = carrier.hidden_dim;
    let kv_len = pos + 1;

    carrier.upload_hidden(hidden);
    // 첫 호출(filled==0)에만 prefill KV(host f16, 0..pos)를 device 로 1회 init.
    // 이후 토큰은 kv_append 로 device 누적(host roundtrip 없음) — carrier 가 device
    // KV 를 완전 소유. host KV cache 는 carrier 활성 layer 에서 안 읽힘.
    if carrier.kv.filled == 0 {
        if carrier.kv.kv_int8 {
            carrier.kv.ensure_filled_i8(prior_k, prior_v, pos);
        } else {
            carrier.kv.ensure_filled(prior_k, prior_v, pos);
        }
    }

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = chain_compute_encoder(ctx, &cmd);

    attn_chain_encode(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        norm_w_buf,
        q_w_buf,
        q_off_buf,
        k_w_buf,
        k_off_buf,
        v_w_buf,
        v_off_buf,
        q_norm_w_buf,
        k_norm_w_buf,
        o_w_buf,
        o_off_buf,
        v_is_q6k,
        ffn_norm_w_buf,
        ffn_gate_w_buf,
        ffn_gate_off_buf,
        ffn_up_w_buf,
        ffn_up_off_buf,
        ffn_down_w_buf,
        ffn_down_off_buf,
        ffn_down_is_q6k,
        pos,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    carrier.kv.filled = kv_len;
    readback(&carrier.hidden_dev, hidden_dim)
}

/// Attention layer 의 encode ①~⑪(RMS norm부터 O projection residual까지)를
/// 주어진 encoder 에 encode 만 한다. upload/KV init/commit/readback/filled 갱신은
/// caller 가 관리하고, `hidden_dev` 는 다음 FFN/GDN layer 와 그대로 공유한다.
///
/// `ctx.chain_profile` 가 All 이 아니면 dispatch 를 class(gemv/small/attn) 별로 격리
/// emit 한다(pm21 REST 분해 측정용 차감법). 측정 모드에서 중간 buffer 는 stale 이지만
/// 각 dispatch 의 GPU work 양은 동일하므로 GPU time 비중 측정에 유효.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_chain_encode_core(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_off_buf: &ProtocolObject<dyn MTLBuffer>,
    k_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_w_buf: &ProtocolObject<dyn MTLBuffer>,
    v_off_buf: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_is_q6k: bool,
    pos: usize,
) {
    let hidden_dim = carrier.hidden_dim;
    let q_dim = carrier.q_dim;
    let q_out_dim = carrier.q_out_dim;
    let kv_dim = carrier.kv_dim;
    let num_heads = carrier.num_heads;
    let num_kv_heads = carrier.num_kv_heads;
    // carrier 에 head_dim raw 없음 — kv_dim/num_kv_heads 로 유도(정수 정확).
    let head_dim = kv_dim / num_kv_heads;
    let kv_len = pos + 1;

    // 가변 scalar(매 토큰): pos, kv_len, scale.
    let pos_buf = u32_buf(ctx, pos as u32);
    let kl_buf = u32_buf(ctx, kv_len as u32);
    let scale_buf = f32_buf(ctx, carrier.scale);

    let p = ctx.chain_profile;

    // ① attn_norm: hidden_dev → normed_dev [small]
    if p.emit_small() {
        encode_rms_norm(
            ctx,
            enc,
            hidden_dev,
            norm_w_buf,
            &carrier.normed_dev,
            &carrier.hdim_buf,
            &carrier.eps_buf,
        );
    }
    chain_barrier(ctx, enc); // ① norm 완료 → 그룹 {q·k·v} 독립 GEMV 진입 (normed_dev read-only 공유)
                             // ② q: normed_dev → q_full_dev (N=q_out_dim, K=hidden_dim) — gated 면 [query|gate] 인터리브 [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_q4k_auto(
            ctx,
            enc,
            q_w_buf,
            &carrier.normed_dev,
            &carrier.q_full_dev,
            &carrier.qoutdim_buf,
            &carrier.k_hidden_buf,
            q_off_buf,
            q_out_dim,
        );
    }
    // ③ k: normed_dev → k_dev (N=kv_dim, K=hidden_dim) — 그룹 내부(barrier 없음, q와 독립 write) [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_q4k_auto(
            ctx,
            enc,
            k_w_buf,
            &carrier.normed_dev,
            &carrier.k_dev,
            &carrier.kvdim_buf,
            &carrier.k_hidden_buf,
            k_off_buf,
            kv_dim,
        );
    }
    // ④ v: normed_dev → v_dev (Q4_K_M 에서 attn_v 는 layer 별 Q4_K 또는 Q6_K) — 그룹 내부 [gemv]
    if p.emit_gemv() {
        if v_is_q6k {
            compute::encode_gemv_q6k_auto(
                ctx,
                enc,
                v_w_buf,
                &carrier.normed_dev,
                &carrier.v_dev,
                &carrier.kvdim_buf,
                &carrier.k_hidden_buf,
                v_off_buf,
                kv_dim,
            );
        } else {
            compute::encode_gemv_q4k_auto(
                ctx,
                enc,
                v_w_buf,
                &carrier.normed_dev,
                &carrier.v_dev,
                &carrier.kvdim_buf,
                &carrier.k_hidden_buf,
                v_off_buf,
                kv_dim,
            );
        }
    }
    chain_barrier(ctx, enc); // 그룹 {q·k·v} 완료
                             // ②.5 split: q_full_dev → q_dev(query), gate_dev (q 결과 의존, 그룹 후로 재배치 — pm28 2차) [small]
    if p.emit_small() {
        encode_split_qgate(
            ctx,
            enc,
            &carrier.q_full_dev,
            &carrier.q_dev,
            &carrier.gate_dev,
            &carrier.hd_buf,
            num_heads,
            q_dim / num_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑤ q_norm: per-head RMSNorm q_dev (in-place) [small]
    if p.emit_small() {
        encode_qk_norm(
            ctx,
            enc,
            &carrier.q_dev,
            q_norm_w_buf,
            &carrier.q_dev,
            &carrier.hd_buf,
            &carrier.eps_buf,
            num_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑥ k_norm: per-head RMSNorm k_dev (in-place) [small]
    if p.emit_small() {
        encode_qk_norm(
            ctx,
            enc,
            &carrier.k_dev,
            k_norm_w_buf,
            &carrier.k_dev,
            &carrier.hd_buf,
            &carrier.eps_buf,
            num_kv_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑦ rope: q_dev / k_dev (in-place, partial 인접페어 — 9B production RoPE) [small]
    if p.emit_small() {
        encode_rope_partial(
            ctx,
            enc,
            &carrier.q_dev,
            &carrier.hd_buf,
            &carrier.qdim_buf,
            &carrier.nrot_buf,
            &carrier.theta_scale_buf,
            &pos_buf,
            num_heads,
        );
        encode_rope_partial(
            ctx,
            enc,
            &carrier.k_dev,
            &carrier.hd_buf,
            &carrier.kvdim_buf,
            &carrier.nrot_buf,
            &carrier.theta_scale_buf,
            &pos_buf,
            num_kv_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑧ kv_append: k_dev/v_dev(f32) → KV_dev[pos] (f16) [small]
    if p.emit_small() {
        if ctx.kv_int8 {
            encode_kv_append_i8(
                ctx,
                enc,
                &carrier.k_dev,
                &carrier.v_dev,
                carrier.kv.k_i8.as_ref().unwrap(),
                carrier.kv.v_i8.as_ref().unwrap(),
                carrier.kv.k_scale.as_ref().unwrap(),
                carrier.kv.v_scale.as_ref().unwrap(),
                &carrier.hd_buf,
                &carrier.nkv_buf,
                &pos_buf,
                num_kv_heads,
            );
        } else {
            encode_kv_append(
                ctx,
                enc,
                &carrier.k_dev,
                &carrier.v_dev,
                &carrier.kv.k_buf,
                &carrier.kv.v_buf,
                &carrier.kvdim_buf,
                &pos_buf,
                kv_dim,
            );
        }
    }
    chain_barrier(ctx, enc);
    // ⑨ attn: q_dev + KV_dev[0..kv_len] → attn_out_dev [attn]
    if p.emit_attn() {
        if ctx.kv_int8 {
            if ctx.attn_splitk_splits > 1 && kv_len >= ctx.attn_splitk_min_kv {
                encode_attn_decode_i8_splitk(
                    ctx,
                    enc,
                    &carrier.q_dev,
                    carrier.kv.k_i8.as_ref().unwrap(),
                    carrier.kv.v_i8.as_ref().unwrap(),
                    carrier.kv.k_scale.as_ref().unwrap(),
                    carrier.kv.v_scale.as_ref().unwrap(),
                    carrier
                        .attn_splitk_acc_dev
                        .as_ref()
                        .expect("splitk acc buffer missing"),
                    carrier
                        .attn_splitk_m_dev
                        .as_ref()
                        .expect("splitk m buffer missing"),
                    carrier
                        .attn_splitk_s_dev
                        .as_ref()
                        .expect("splitk s buffer missing"),
                    &carrier.attn_out_dev,
                    &carrier.nh_buf,
                    &carrier.nkv_buf,
                    &carrier.hd_buf,
                    &kl_buf,
                    &scale_buf,
                    carrier
                        .attn_splitk_splits_buf
                        .as_ref()
                        .expect("splitk splits buffer missing"),
                    num_heads,
                    head_dim,
                    ctx.attn_splitk_splits,
                );
            } else {
                encode_attn_decode_i8(
                    ctx,
                    enc,
                    &carrier.q_dev,
                    carrier.kv.k_i8.as_ref().unwrap(),
                    carrier.kv.v_i8.as_ref().unwrap(),
                    carrier.kv.k_scale.as_ref().unwrap(),
                    carrier.kv.v_scale.as_ref().unwrap(),
                    &carrier.attn_out_dev,
                    &carrier.nh_buf,
                    &carrier.nkv_buf,
                    &carrier.hd_buf,
                    &kl_buf,
                    &scale_buf,
                    num_heads,
                    head_dim,
                );
            }
        } else {
            encode_attn_decode(
                ctx,
                enc,
                &carrier.q_dev,
                &carrier.kv.k_buf,
                &carrier.kv.v_buf,
                &carrier.attn_out_dev,
                &carrier.nh_buf,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &kl_buf,
                &scale_buf,
                num_heads,
            );
        }
    }
    chain_barrier(ctx, enc);
    // ⑨.5 gated: attn_out_dev *= sigmoid(gate_dev) (q_dim elementwise) [small]
    if p.emit_small() {
        encode_gate_apply(
            ctx,
            enc,
            &carrier.attn_out_dev,
            &carrier.gate_dev,
            &carrier.qdim_buf,
            q_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑩ o: attn_out_dev → o_out_dev (N=hidden_dim, K=q_dim) [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_q4k_auto(
            ctx,
            enc,
            o_w_buf,
            &carrier.attn_out_dev,
            &carrier.o_out_dev,
            &carrier.hdim_buf,
            &carrier.k_qdim_buf,
            o_off_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑪ residual: hidden_dev += o_out_dev (attention 출력) [small]
    if p.emit_small() {
        encode_residual_add(
            ctx,
            enc,
            hidden_dev,
            &carrier.o_out_dev,
            &carrier.hdim_buf,
            hidden_dim,
        );
    }

    chain_barrier(ctx, enc); // attention residual 완료 → 다음 FFN/layer
}

#[allow(clippy::too_many_arguments)]
fn attn_dense_ffn_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
) {
    // FFN (carrier 통합, 같은 command buffer):
    // ⑫ ffn_norm → ⑬ gate GEMV → ⑭ up GEMV → ⑮ silu_mul → ⑯ down GEMV → ⑰ residual
    let hidden_dim = carrier.hidden_dim;
    let ffn_dim = carrier.ffn_dim;
    let p = ctx.chain_profile;

    // ⑫ ffn_norm [small]
    if p.emit_small() {
        encode_rms_norm(
            ctx,
            enc,
            hidden_dev,
            ffn_norm_w_buf,
            &carrier.ffn_normed_dev,
            &carrier.hdim_buf,
            &carrier.eps_buf,
        );
    }
    chain_barrier(ctx, enc); // ffn_norm 완료 → 그룹 B(ffn_gate·up) 독립 GEMV 진입
                             // ⑬ gate GEMV [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_q4k_auto(
            ctx,
            enc,
            ffn_gate_w_buf,
            &carrier.ffn_normed_dev,
            &carrier.ffn_gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            ffn_gate_off_buf,
            ffn_dim,
        );
    }
    // ⑭ up GEMV [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_q4k_auto(
            ctx,
            enc,
            ffn_up_w_buf,
            &carrier.ffn_normed_dev,
            &carrier.ffn_up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            ffn_up_off_buf,
            ffn_dim,
        );
    }
    chain_barrier(ctx, enc); // 그룹 B(ffn_gate·up) 완료
                             // ⑮ silu_mul [small]
    if p.emit_small() {
        encode_silu_mul(
            ctx,
            enc,
            &carrier.ffn_gate_dev,
            &carrier.ffn_up_dev,
            &carrier.fdim_buf,
            ffn_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑯ down GEMV [gemv]
    if p.emit_gemv() {
        if ffn_down_is_q6k {
            compute::encode_gemv_q6k_auto(
                ctx,
                enc,
                ffn_down_w_buf,
                &carrier.ffn_gate_dev,
                &carrier.ffn_down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                ffn_down_off_buf,
                hidden_dim,
            );
        } else {
            compute::encode_gemv_q4k_auto(
                ctx,
                enc,
                ffn_down_w_buf,
                &carrier.ffn_gate_dev,
                &carrier.ffn_down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                ffn_down_off_buf,
                hidden_dim,
            );
        }
    }
    chain_barrier(ctx, enc);
    // ⑰ residual [small]
    if p.emit_small() {
        encode_residual_add(
            ctx,
            enc,
            hidden_dev,
            &carrier.ffn_down_dev,
            &carrier.hdim_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc); // layer 경계 — 다음 layer 의 norm 이 이 layer hidden 완료를 보게
}

/// Attention core ①~⑪와 기존 dense FFN ⑫~⑰를 같은 encoder에 이어 encode한다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_off_buf: &ProtocolObject<dyn MTLBuffer>,
    k_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_w_buf: &ProtocolObject<dyn MTLBuffer>,
    v_off_buf: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_is_q6k: bool,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
    pos: usize,
) {
    attn_chain_encode_core(
        ctx,
        enc,
        carrier,
        hidden_dev,
        norm_w_buf,
        q_w_buf,
        q_off_buf,
        k_w_buf,
        k_off_buf,
        v_w_buf,
        v_off_buf,
        q_norm_w_buf,
        k_norm_w_buf,
        o_w_buf,
        o_off_buf,
        v_is_q6k,
        pos,
    );
    attn_dense_ffn_chain_encode(
        ctx,
        enc,
        carrier,
        hidden_dev,
        ffn_norm_w_buf,
        ffn_gate_w_buf,
        ffn_gate_off_buf,
        ffn_up_w_buf,
        ffn_up_off_buf,
        ffn_down_w_buf,
        ffn_down_off_buf,
        ffn_down_is_q6k,
    );
}

/// 연속 chain run 의 attention carrier layer 하나의 per-layer 인자를 borrow 로 묶은 spec.
/// `attn_layer_resident` 가 호출당 받던 인자(weight raw `&[u8]` 7개 + f32 weight `&[f32]`
/// 4개 + v/ffn_down quant 구분 bool + prior KV(host f16) + pos + shape)를 그대로 담는다.
/// weight 는 `ModelWeights` 의 mmap slice 를 borrow 하므로 lifetime `<'a>`.
///
/// q/k/o/ffn_gate/ffn_up 은 항상 Q4_K(seam 가드), v 와 ffn_down 만 Q4_K|Q6_K 라
/// `v_is_q6k`/`ffn_down_is_q6k` bool 로 구분(GDN 처럼 quant code 8개 안 들고 다님).
///
/// prior_k/prior_v 는 첫 token(`KvResident::filled == 0`)에 device KV 를 1회 init 하는
/// host f16 bits([pos*kv_dim]). chain run 이 attn layer encode 전 `ensure_filled`,
/// commit 후 `filled = pos+1` 갱신을 처리한다(2.1 dispatch wrapper 와 동일).
///
/// 2단계 enum `ChainLayerSpecRef::Attn` 으로 묶여 `decode_chain_run` 이 GDN 과 같은
/// 단일 command buffer 에 합류시킨다.
#[allow(dead_code)] // 2단계에서 호출자 연결.
pub struct AttnChainSpecRef<'a> {
    pub layer: usize,
    // f32 weight (작아서 복사 업로드).
    pub norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    // 양자화 GEMV weight raw bytes (NoCopy resident wrap 대상). q/k/o/ffn_gate/ffn_up = Q4_K.
    pub q_raw: &'a [u8],
    pub k_raw: &'a [u8],
    pub v_raw: &'a [u8],
    pub o_raw: &'a [u8],
    pub ffn_gate_raw: &'a [u8],
    pub ffn_up_raw: &'a [u8],
    pub ffn_down_raw: &'a [u8],
    // v / ffn_down 만 Q4_K|Q6_K 구분.
    pub v_is_q6k: bool,
    pub ffn_down_is_q6k: bool,
    // prior KV(host f16 bits, [pos*kv_dim]) — 첫 token KV init 용.
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub pos: usize,
    // shape (carrier entry 생성 + dispatch 에 사용).
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub capacity: usize,
    pub ffn_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
}
