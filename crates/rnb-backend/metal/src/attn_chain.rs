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
    self, chain_barrier, chain_barrier_resources, chain_compute_encoder, encode_attn_decode,
    encode_attn_decode_at, encode_attn_decode_f16_gqa16, encode_attn_decode_gqa_splitk,
    encode_attn_decode_i8, encode_attn_decode_i8_gqa_splitk, encode_attn_decode_i8_splitk,
    encode_attn_decode_qk_norm_rope_batch, encode_attn_decode_qk_norm_rope_pair,
    encode_attn_decode_splitk, encode_attn_decode_splitk_at, encode_attn_decode_splitk_batch,
    encode_attn_decode_window, encode_gate_apply, encode_gate_apply_at, encode_gemv_quant_bcol,
    encode_kv_append, encode_kv_append_batch, encode_kv_append_i8, encode_prefill_gate_apply,
    encode_prefill_split_q_gate, encode_qk_norm, encode_rms_norm_batch, encode_rope_partial,
    encode_split_qgate, encode_split_qgate_at, KvResident, MetalContext,
};
use crate::ffn_chain::{
    empty_f32_buf, encode_fused_post_attn_residual_ffn_rms_norm,
    encode_fused_post_ffn_residual_add, encode_fused_post_ffn_residual_next_rms_norm,
    encode_fused_residual_rms_norm, encode_residual_add, encode_rms_norm, encode_silu_mul, f32_buf,
    readback, shared_u32_buf, u32_buf,
};

// f16 grouped kernel은 register pressure 회귀 때문에 명시적 opt-in만 허용한다.
fn gqa_group_requested(value: Option<&str>) -> bool {
    value == Some("1")
}

// int8 grouped kernel은 KV tile 공유 이득이 검증돼 default-on이고 falsey 값만 opt-out이다.
fn int8_gqa_group_enabled(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(value) => {
            !value.eq_ignore_ascii_case("0")
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("off")
                && !value.eq_ignore_ascii_case("no")
        }
    }
}

fn int8_gqa_matrix_requested(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Attention layer 의 device-resident 중간 버퍼 + 불변 scalar 버퍼 + KV.
/// shape 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local.
pub(crate) struct AttnCarrier {
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize, // packed gated: q_dim*2. Separate gate models keep q_dim.
    pub kv_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub scale: f32,

    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_full_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    qoutdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kvdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_norm_eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_qdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,

    pub ffn_dim: usize,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    fdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
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
        post_norm_eps: f32,
        theta: f32,
        scale: f32,
        force_f16_kv: bool,
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
            kv: if force_f16_kv {
                KvResident::new_f16(ctx, num_kv_heads, head_dim, capacity)
            } else {
                KvResident::new(ctx, num_kv_heads, head_dim, capacity)
            },
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            qdim_buf: u32_buf(ctx, q_dim as u32),
            qoutdim_buf: u32_buf(ctx, q_out_dim as u32),
            kvdim_buf: u32_buf(ctx, kv_dim as u32),
            hd_buf: u32_buf(ctx, head_dim as u32),
            eps_buf: f32_buf(ctx, eps),
            post_norm_eps_buf: f32_buf(ctx, post_norm_eps),
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

    pub(crate) fn normed_buffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.normed_dev.clone()
    }

    /// decode chain KVarn attn: 방금 계산한 새 토큰 k/v(post-rope, device f32)를
    /// readback (host KVarn cache append 용, host=source of truth).
    pub(crate) fn readback_new_kv(&self) -> (Vec<f32>, Vec<f32>) {
        (
            readback(&self.k_dev, self.kv_dim),
            readback(&self.v_dev, self.kv_dim),
        )
    }

    /// hidden host slice → hidden_dev 업로드(StorageModeShared contents 직접 쓰기).
    fn upload_hidden(&self, hidden: &[f32]) {
        crate::carrier_validation::assert_exact_len(
            "attention hidden",
            hidden.len(),
            self.hidden_dim,
        );
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
        None,
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
        None,
        None,
        v_is_q6k,
        0,
        ffn_norm_w_buf,
        None,
        ffn_gate_w_buf,
        ffn_gate_off_buf,
        ffn_up_w_buf,
        ffn_up_off_buf,
        ffn_down_w_buf,
        ffn_down_off_buf,
        ffn_down_is_q6k,
        true,
        None,
        false,
        pos,
        None,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    carrier.kv.filled = kv_len;
    readback(&carrier.hidden_dev, hidden_dim)
}

/// Attention layer 의 encode ①~⑪(RMS norm부터 O projection residual까지)를
pub(crate) fn attn_chain_encode_core(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    pre_normed_dev: Option<&ProtocolObject<dyn MTLBuffer>>,
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
    q_q: u8,
    k_q: u8,
    v_q: u8,
    o_q: u8,
    post_attn_norm_w_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    attn_gate: Option<(
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    )>,
    apply_rope: bool,
    sliding_window: Option<usize>,
    muse_semantics: bool,
    pos: usize,
    fused_ffn_norm_w_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    kvarn: Option<compute::KvarnChainEncode>,
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
    let window_buf = u32_buf(ctx, sliding_window.unwrap_or(0) as u32);

    let p = ctx.chain_profile;
    let int8_gqa_group = kvarn.is_none()
        && carrier.kv.kv_int8
        && ctx.attn_splitk_splits > 1
        && kv_len >= ctx.attn_splitk_min_kv
        && int8_gqa_group_enabled(std::env::var("RNB_METAL_ATTN_GQA_GROUP").ok().as_deref())
        && head_dim == 256
        && num_heads == num_kv_heads * 8;
    let int8_gqa_matrix = int8_gqa_group
        && ctx.tensorops_capable
        && int8_gqa_matrix_requested(std::env::var("RNB_METAL_ATTN_GQA_MATRIX").ok().as_deref());
    let int8_gqa_fused_append = int8_gqa_matrix;
    let muse_gqa16 = muse_semantics && head_dim == 128 && num_heads == num_kv_heads * 16;
    let muse_norm_fusion = muse_semantics;

    // ① attn_norm: hidden_dev → normed_dev [small]. An adjacent Muse layer can
    // supply this value from its fused post-FFN tail.
    let normed_dev = pre_normed_dev.unwrap_or(&carrier.normed_dev);
    if pre_normed_dev.is_none() {
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
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [&*carrier.normed_dev]);
        } else {
            chain_barrier(ctx, enc);
        }
    }
    // ② q: normed_dev → q_full_dev (N=q_out_dim, K=hidden_dim) — gated 면 [query|gate] 인터리브 [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_quant(
            ctx,
            enc,
            q_q,
            q_w_buf,
            normed_dev,
            &carrier.q_full_dev,
            &carrier.qoutdim_buf,
            &carrier.k_hidden_buf,
            q_off_buf,
            q_out_dim,
        );
    }
    // ③ k: normed_dev → k_dev (N=kv_dim, K=hidden_dim) — 그룹 내부(barrier 없음, q와 독립 write) [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_quant(
            ctx,
            enc,
            k_q,
            k_w_buf,
            normed_dev,
            &carrier.k_dev,
            &carrier.kvdim_buf,
            &carrier.k_hidden_buf,
            k_off_buf,
            kv_dim,
        );
    }
    // ④ v: normed_dev → v_dev (Q4_K_M 에서 attn_v 는 layer 별 Q4_K 또는 Q6_K) — 그룹 내부 [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_quant(
            ctx,
            enc,
            v_q,
            v_w_buf,
            normed_dev,
            &carrier.v_dev,
            &carrier.kvdim_buf,
            &carrier.k_hidden_buf,
            v_off_buf,
            kv_dim,
        );
    }
    if let Some((gate_w_buf, gate_off_buf)) = attn_gate {
        if p.emit_gemv() {
            compute::encode_gemv_quant(
                ctx,
                enc,
                0,
                gate_w_buf,
                normed_dev,
                &carrier.gate_dev,
                &carrier.qdim_buf,
                &carrier.k_hidden_buf,
                gate_off_buf,
                q_dim,
            );
        }
    }
    if muse_semantics {
        chain_barrier_resources(
            ctx,
            enc,
            [
                &*carrier.q_full_dev,
                &*carrier.k_dev,
                &*carrier.v_dev,
                &*carrier.gate_dev,
            ],
        );
    } else {
        chain_barrier(ctx, enc);
    }
    if attn_gate.is_none() {
        // Packed Q+gate tensors store [query|gate] within each head.
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
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [&*carrier.q_dev, &*carrier.gate_dev]);
        } else {
            chain_barrier(ctx, enc);
        }
    }
    let query_dev = if attn_gate.is_some() {
        &carrier.q_full_dev
    } else {
        &carrier.q_dev
    };
    let fused_qk_norm_rope = muse_semantics && apply_rope;
    if p.emit_small() {
        if fused_qk_norm_rope {
            encode_attn_decode_qk_norm_rope_pair(
                ctx,
                enc,
                query_dev,
                q_norm_w_buf,
                &carrier.k_dev,
                k_norm_w_buf,
                &carrier.nh_buf,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &carrier.nrot_buf,
                &carrier.theta_scale_buf,
                &carrier.eps_buf,
                &pos_buf,
                num_heads + num_kv_heads,
            );
        } else {
            encode_qk_norm(
                ctx,
                enc,
                query_dev,
                q_norm_w_buf,
                query_dev,
                &carrier.hd_buf,
                &carrier.eps_buf,
                num_heads,
            );
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
    }
    if muse_semantics {
        chain_barrier_resources(ctx, enc, [query_dev, &*carrier.k_dev]);
    } else {
        chain_barrier(ctx, enc);
    }
    if p.emit_small() && apply_rope && !fused_qk_norm_rope {
        encode_rope_partial(
            ctx,
            enc,
            query_dev,
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
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [query_dev, &*carrier.k_dev]);
        } else {
            chain_barrier(ctx, enc);
        }
    }
    // ⑧ kv_append: k_dev/v_dev(f32) → KV_dev[pos] (fused TensorOps는 ⑨에서 처리) [small]
    if p.emit_small() {
        if let Some(kv) = kvarn {
            kv.resident.encode_tail_append(
                ctx,
                enc,
                &carrier.k_dev,
                &carrier.v_dev,
                kv_dim,
                kv.append_slot,
            );
        } else if carrier.kv.kv_int8 {
            if !int8_gqa_fused_append {
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
            }
        } else if !muse_gqa16 {
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
    let muse_fused_append = muse_gqa16 && p.emit_attn();
    if !int8_gqa_fused_append && !muse_fused_append {
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [&*carrier.kv.k_buf, &*carrier.kv.v_buf]);
        } else {
            chain_barrier(ctx, enc);
        }
    }
    if p.emit_attn() {
        if let Some(kv) = kvarn {
            compute::encode_kvarn_attention_splitk(
                ctx,
                enc,
                kv.resident,
                query_dev,
                &carrier.attn_out_dev,
                kv.params_buf,
                kv.num_splits_buf,
                kv.partial_acc,
                kv.partial_m,
                kv.partial_s,
                kv.num_heads,
                kv.num_splits,
            );
        } else if carrier.kv.kv_int8 {
            if ctx.attn_splitk_splits > 1 && kv_len >= ctx.attn_splitk_min_kv {
                if int8_gqa_group {
                    encode_attn_decode_i8_gqa_splitk(
                        ctx,
                        enc,
                        query_dev,
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
                        num_kv_heads,
                        head_dim,
                        ctx.attn_splitk_splits,
                        int8_gqa_matrix,
                        &carrier.k_dev,
                        &carrier.v_dev,
                    );
                } else {
                    encode_attn_decode_i8_splitk(
                        ctx,
                        enc,
                        query_dev,
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
                }
            } else {
                encode_attn_decode_i8(
                    ctx,
                    enc,
                    query_dev,
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
        } else if muse_gqa16 {
            encode_attn_decode_f16_gqa16(
                ctx,
                enc,
                query_dev,
                &carrier.kv.k_buf,
                &carrier.kv.v_buf,
                &carrier.attn_out_dev,
                &carrier.nh_buf,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &kl_buf,
                &scale_buf,
                &window_buf,
                &carrier.gate_dev,
                &pos_buf,
                &carrier.k_dev,
                &carrier.v_dev,
                num_heads,
            );
        } else if muse_semantics && sliding_window.is_some() {
            encode_attn_decode_window(
                ctx,
                enc,
                query_dev,
                &carrier.kv.k_buf,
                &carrier.kv.v_buf,
                &carrier.attn_out_dev,
                &carrier.nh_buf,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &kl_buf,
                &scale_buf,
                &window_buf,
                num_heads,
            );
        } else {
            let gqa_group =
                gqa_group_requested(std::env::var("RNB_METAL_ATTN_GQA_GROUP").ok().as_deref())
                    && num_heads > num_kv_heads
                    && ctx.attn_splitk_splits > 1
                    && kv_len >= ctx.attn_splitk_min_kv;
            if gqa_group {
                encode_attn_decode_gqa_splitk(
                    ctx,
                    enc,
                    query_dev,
                    &carrier.kv.k_buf,
                    &carrier.kv.v_buf,
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
                    num_kv_heads,
                    head_dim,
                    ctx.attn_splitk_splits,
                );
            } else if ctx.attn_splitk_splits > 1 && kv_len >= ctx.attn_splitk_min_kv {
                encode_attn_decode_splitk(
                    ctx,
                    enc,
                    query_dev,
                    &carrier.kv.k_buf,
                    &carrier.kv.v_buf,
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
                encode_attn_decode(
                    ctx,
                    enc,
                    query_dev,
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
    }
    let muse_fused_gate = muse_gqa16 && p.emit_attn();
    if !muse_fused_gate {
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [&*carrier.attn_out_dev]);
        } else {
            chain_barrier(ctx, enc);
        }
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
    }
    if muse_semantics {
        chain_barrier_resources(ctx, enc, [&*carrier.attn_out_dev]);
    } else {
        chain_barrier(ctx, enc);
    }
    // ⑩ o: attn_out_dev → o_out_dev (N=hidden_dim, K=q_dim) [gemv]
    if p.emit_gemv() {
        compute::encode_gemv_quant(
            ctx,
            enc,
            o_q,
            o_w_buf,
            &carrier.attn_out_dev,
            &carrier.o_out_dev,
            &carrier.hdim_buf,
            &carrier.k_qdim_buf,
            o_off_buf,
            hidden_dim,
        );
    }
    if muse_semantics {
        chain_barrier_resources(ctx, enc, [&*carrier.o_out_dev]);
    } else {
        chain_barrier(ctx, enc);
    }
    let fused_muse_norms = if muse_norm_fusion && p.emit_small() {
        post_attn_norm_w_buf.zip(fused_ffn_norm_w_buf)
    } else {
        None
    };
    if let Some((post_attn_norm_w_buf, ffn_norm_w_buf)) = fused_muse_norms {
        encode_fused_post_attn_residual_ffn_rms_norm(
            ctx,
            enc,
            &carrier.o_out_dev,
            post_attn_norm_w_buf,
            hidden_dev,
            ffn_norm_w_buf,
            &carrier.ffn_normed_dev,
            &carrier.hdim_buf,
            &carrier.post_norm_eps_buf,
            &carrier.eps_buf,
        );
        chain_barrier_resources(ctx, enc, [hidden_dev, &*carrier.ffn_normed_dev]);
    } else {
        let attn_residual = if let Some(post_attn_norm_w_buf) = post_attn_norm_w_buf {
            if p.emit_small() {
                encode_rms_norm(
                    ctx,
                    enc,
                    &carrier.o_out_dev,
                    post_attn_norm_w_buf,
                    &carrier.ffn_normed_dev,
                    &carrier.hdim_buf,
                    &carrier.post_norm_eps_buf,
                );
            }
            if muse_semantics {
                chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
            } else {
                chain_barrier(ctx, enc);
            }
            &carrier.ffn_normed_dev
        } else {
            &carrier.o_out_dev
        };
        if let Some(ffn_norm_w_buf) = fused_ffn_norm_w_buf.filter(|_| p.emit_small()) {
            encode_fused_residual_rms_norm(
                ctx,
                enc,
                hidden_dev,
                attn_residual,
                ffn_norm_w_buf,
                &carrier.ffn_normed_dev,
                &carrier.hdim_buf,
                &carrier.eps_buf,
            );
            if muse_semantics {
                chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
            } else {
                chain_barrier(ctx, enc);
            }
        } else {
            if p.emit_small() {
                encode_residual_add(
                    ctx,
                    enc,
                    hidden_dev,
                    attn_residual,
                    &carrier.hdim_buf,
                    hidden_dim,
                );
            }
            if muse_semantics {
                chain_barrier_resources(ctx, enc, [hidden_dev]);
            } else {
                chain_barrier(ctx, enc);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_dense_ffn_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    post_ffn_norm_w_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    next_attn_norm: Option<(&ProtocolObject<dyn MTLBuffer>, f32)>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
    muse_semantics: bool,
    input_already_normed: bool,
) {
    let hidden_dim = carrier.hidden_dim;
    let muse_norm_fusion = muse_semantics;
    let ffn_dim = carrier.ffn_dim;
    let p = ctx.chain_profile;
    let fuse_muse_swiglu = muse_semantics;

    if !input_already_normed {
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
        if muse_semantics {
            chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
        } else {
            chain_barrier(ctx, enc);
        }
    }
    if p.emit_gemv() {
        if fuse_muse_swiglu {
            compute::encode_gemv_q4k_swiglu_pair(
                ctx,
                enc,
                ffn_gate_w_buf,
                ffn_up_w_buf,
                &carrier.ffn_normed_dev,
                &carrier.ffn_gate_dev,
                &carrier.fdim_buf,
                &carrier.k_hidden_buf,
                ffn_gate_off_buf,
                ffn_up_off_buf,
                ffn_dim,
            );
            chain_barrier_resources(ctx, enc, [&*carrier.ffn_gate_dev]);
        } else {
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
            chain_barrier(ctx, enc);
        }
    }
    if p.emit_small() && !fuse_muse_swiglu {
        encode_silu_mul(
            ctx,
            enc,
            &carrier.ffn_gate_dev,
            &carrier.ffn_up_dev,
            &carrier.fdim_buf,
            ffn_dim,
        );
        chain_barrier(ctx, enc);
    }
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
    if muse_semantics {
        chain_barrier_resources(ctx, enc, [&*carrier.ffn_down_dev]);
    } else {
        chain_barrier(ctx, enc);
    }
    if let (true, Some(post_ffn_norm_w_buf)) = (muse_norm_fusion, post_ffn_norm_w_buf) {
        if p.emit_small() {
            if let Some((next_norm_w_buf, next_eps)) = next_attn_norm {
                encode_fused_post_ffn_residual_next_rms_norm(
                    ctx,
                    enc,
                    &carrier.ffn_down_dev,
                    post_ffn_norm_w_buf,
                    hidden_dev,
                    next_norm_w_buf,
                    &carrier.normed_dev,
                    &carrier.hdim_buf,
                    &carrier.post_norm_eps_buf,
                    next_eps,
                );
                chain_barrier_resources(ctx, enc, [hidden_dev, &*carrier.normed_dev]);
            } else {
                encode_fused_post_ffn_residual_add(
                    ctx,
                    enc,
                    &carrier.ffn_down_dev,
                    post_ffn_norm_w_buf,
                    hidden_dev,
                    &carrier.hdim_buf,
                    &carrier.post_norm_eps_buf,
                );
                chain_barrier_resources(ctx, enc, [hidden_dev]);
            }
        }
    } else {
        let ffn_residual = if let Some(post_ffn_norm_w_buf) = post_ffn_norm_w_buf {
            if p.emit_small() {
                encode_rms_norm(
                    ctx,
                    enc,
                    &carrier.ffn_down_dev,
                    post_ffn_norm_w_buf,
                    &carrier.o_out_dev,
                    &carrier.hdim_buf,
                    &carrier.post_norm_eps_buf,
                );
            }
            chain_barrier(ctx, enc);
            &carrier.o_out_dev
        } else {
            &carrier.ffn_down_dev
        };
        if p.emit_small() {
            encode_residual_add(
                ctx,
                enc,
                hidden_dev,
                ffn_residual,
                &carrier.hdim_buf,
                hidden_dim,
            );
        }
        chain_barrier(ctx, enc);
    }
}

/// Attention core ①~⑪와 기존 dense FFN ⑫~⑰를 같은 encoder에 이어 encode한다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    pre_normed_dev: Option<&ProtocolObject<dyn MTLBuffer>>,
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
    post_attn_norm_w_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    attn_gate: Option<(
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    )>,
    v_is_q6k: bool,
    o_q: u8,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    post_ffn_norm_w_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
    apply_rope: bool,
    sliding_window: Option<usize>,
    muse_semantics: bool,
    pos: usize,
    next_attn_norm: Option<(&ProtocolObject<dyn MTLBuffer>, f32)>,
) -> bool {
    attn_chain_encode_core(
        ctx,
        enc,
        carrier,
        hidden_dev,
        pre_normed_dev,
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
        0,
        0,
        if v_is_q6k { 2 } else { 0 },
        o_q,
        post_attn_norm_w_buf,
        attn_gate,
        apply_rope,
        sliding_window,
        muse_semantics,
        pos,
        Some(ffn_norm_w_buf).filter(|_| muse_semantics),
        None,
    );
    attn_dense_ffn_chain_encode(
        ctx,
        enc,
        carrier,
        hidden_dev,
        ffn_norm_w_buf,
        post_ffn_norm_w_buf,
        next_attn_norm,
        ffn_gate_w_buf,
        ffn_gate_off_buf,
        ffn_up_w_buf,
        ffn_up_off_buf,
        ffn_down_w_buf,
        ffn_down_off_buf,
        ffn_down_is_q6k,
        muse_semantics,
        muse_semantics,
    );
    muse_semantics
        && ctx.chain_profile.emit_small()
        && post_ffn_norm_w_buf.is_some()
        && next_attn_norm.is_some()
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
#[derive(Clone, Copy)]
pub struct AttnChainSpecRef<'a> {
    pub layer: usize,
    pub norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub post_attn_norm_weight: Option<&'a [f32]>,
    pub post_ffn_norm_weight: Option<&'a [f32]>,
    pub q_raw: &'a [u8],
    pub k_raw: &'a [u8],
    pub o_q: u8,
    pub v_raw: &'a [u8],
    pub o_raw: &'a [u8],
    pub attn_gate_raw: Option<&'a [u8]>,
    pub ffn_gate_raw: &'a [u8],
    pub ffn_up_raw: &'a [u8],
    pub ffn_down_raw: &'a [u8],
    pub v_is_q6k: bool,
    pub ffn_down_is_q6k: bool,
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub pos: usize,
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
    pub post_norm_eps: f32,
    pub theta: f32,
    pub scale: f32,
    pub apply_rope: bool,
    pub sliding_window: Option<usize>,
    pub muse_semantics: bool,
}

/// Batched(B-lane) attention core carrier — milestone 4 (MTP verify body fusion, mixed
/// attn+GDN chain). 단일-토큰 `AttnCarrier` 와 완전히 분리(프로덕션 single-token 경로
/// 불변). 한 command buffer 안에서 B lane 의 attention core 를 처리하며 **q/k/v/o weight 를
/// layer 당 1회만** 읽는다:
///   - q/k/v/o 는 B-column GEMV(`encode_gemv_quant_bcol`)로 amortize.
///   - split/qk-norm/rope/kv-append/attn/gate 는 lane 슬롯 offset 으로 per-lane 순차.
///   - KV 는 device-resident f16. 첫 verify 는 host prior 를 upload하고, 이후 verify 는
///     이미 device 에 남은 durable prefix를 재사용한다. lane i 는 slot base_pos+i 에 append 후
///     kv_len=base_pos+i+1 로 `[0..base_pos+i]` 를 attend — 같은 command buffer의 barrier가
///     lane i+1이 lane i의 slot을 host post-commit 없이 보도록 보장한다.
/// (layer, B) 별 1회 alloc 후 토큰 간 재사용. `!Send+!Sync` 라 thread_local.
/// f16 KV 전용(int8/KVarn chain 은 caller 가 per-lane 경로로 폴백).
#[allow(dead_code)]
pub(crate) struct AttnBatchCarrier {
    pub b: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub capacity: usize,

    normed_all: Retained<ProtocolObject<dyn MTLBuffer>>, // [b*hidden_dim]
    q_full_all: Retained<ProtocolObject<dyn MTLBuffer>>, // [b*q_out_dim] (gated 인터리브)
    q_all: Retained<ProtocolObject<dyn MTLBuffer>>,      // [b*q_dim]
    gate_all: Retained<ProtocolObject<dyn MTLBuffer>>,   // [b*q_dim]
    k_all: Retained<ProtocolObject<dyn MTLBuffer>>,      // [b*kv_dim]
    v_all: Retained<ProtocolObject<dyn MTLBuffer>>,      // [b*kv_dim]
    attn_out_all: Retained<ProtocolObject<dyn MTLBuffer>>, // [b*q_dim]
    attn_splitk_acc_all: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_m_all: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_s_all: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_splitk_splits_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    o_out_all: Retained<ProtocolObject<dyn MTLBuffer>>, // [b*hidden_dim]
    kv: KvResident,

    hdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    qdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    qoutdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kvdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // q/k/v GEMV 의 K = hidden_dim
    k_qdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,   // o GEMV 의 K = q_dim
    scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    b_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pos_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    kl_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    kv_lens_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_all_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_all_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl AttnBatchCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        b: usize,
        hidden_dim: usize,
        q_dim: usize,
        q_out_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        n_rot: usize,
        capacity: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Self {
        let resident_kv_dim = crate::carrier_validation::checked_product(
            "attention KV dimension",
            num_kv_heads,
            head_dim,
        );
        crate::carrier_validation::assert_exact_len(
            "attention KV dimension",
            kv_dim,
            resident_kv_dim,
        );
        // host `rope_partial_inplace` 와 동일 식(AttnCarrier::new 와 일치).
        let nr = n_rot.min(head_dim);
        let theta_scale: f32 = theta.powf(-2.0_f32 / nr as f32);
        let splitk_splits = ctx.attn_splitk_splits;
        let (attn_splitk_acc_all, attn_splitk_m_all, attn_splitk_s_all, attn_splitk_splits_buf) =
            if splitk_splits > 1 {
                (
                    Some(empty_f32_buf(ctx, b * splitk_splits * num_heads * head_dim)),
                    Some(empty_f32_buf(ctx, b * splitk_splits * num_heads)),
                    Some(empty_f32_buf(ctx, b * splitk_splits * num_heads)),
                    Some(u32_buf(ctx, splitk_splits as u32)),
                )
            } else {
                (None, None, None, None)
            };
        Self {
            b,
            hidden_dim,
            q_dim,
            q_out_dim,
            kv_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            capacity,
            normed_all: empty_f32_buf(ctx, b * hidden_dim),
            q_full_all: empty_f32_buf(ctx, b * q_out_dim),
            q_all: empty_f32_buf(ctx, b * q_dim),
            gate_all: empty_f32_buf(ctx, b * q_dim),
            k_all: empty_f32_buf(ctx, b * kv_dim),
            v_all: empty_f32_buf(ctx, b * kv_dim),
            attn_out_all: empty_f32_buf(ctx, b * q_dim),
            attn_splitk_acc_all,
            attn_splitk_m_all,
            attn_splitk_s_all,
            attn_splitk_splits_buf,
            o_out_all: empty_f32_buf(ctx, b * hidden_dim),
            kv: KvResident::new_f16(ctx, num_kv_heads, head_dim, capacity),
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
            scale_buf: f32_buf(ctx, scale),
            b_buf: u32_buf(ctx, b as u32),
            pos_bufs: (0..b).map(|_| u32_buf(ctx, 0)).collect(),
            kl_bufs: (0..b).map(|_| u32_buf(ctx, 0)).collect(),
            kv_lens_buf: shared_u32_buf(ctx, &vec![0; b]),
            hidden_all_buf: u32_buf(ctx, (b * hidden_dim) as u32),
            q_all_elems_buf: u32_buf(ctx, (b * q_dim) as u32),
        }
    }

    /// chain 진입 전: host KV를 source of truth로 검증하되, device에 이미 존재하는 durable
    /// prefix는 유지하고 아직 채워지지 않은 suffix만 upload한다. 거절 뒤 base_pos가 직전
    /// speculative window 안으로 되돌아오면 그 prefix는 유효하고 window는 이어지는 append가
    /// 덮어쓴다. 새 sequence에서는 backend가 carrier를 evict한다.
    pub(crate) fn upload_prior(&mut self, prior_k: &[u16], prior_v: &[u16], base_pos: usize) {
        crate::carrier_validation::assert_exact_len(
            "attention KV dimension",
            self.kv_dim,
            self.kv.kv_dim,
        );
        crate::carrier_validation::checked_slot_range_end(
            "attention prior KV",
            0,
            base_pos,
            self.kv.capacity,
        );
        let expected = crate::carrier_validation::checked_product(
            "attention prior KV",
            base_pos,
            self.kv.kv_dim,
        );
        crate::carrier_validation::assert_exact_len("attention prior K", prior_k.len(), expected);
        crate::carrier_validation::assert_exact_len("attention prior V", prior_v.len(), expected);

        let start_slot = self.kv.filled.min(base_pos);
        let start = crate::carrier_validation::checked_product(
            "attention prior KV upload offset",
            start_slot,
            self.kv.kv_dim,
        );
        let missing = expected - start;
        if missing != 0 {
            unsafe {
                let kp = self.kv.k_buf.contents().as_ptr() as *mut u16;
                let vp = self.kv.v_buf.contents().as_ptr() as *mut u16;
                std::ptr::copy_nonoverlapping(prior_k.as_ptr().add(start), kp.add(start), missing);
                std::ptr::copy_nonoverlapping(prior_v.as_ptr().add(start), vp.add(start), missing);
            }
        }
        self.kv.filled = base_pos;
    }

    pub(crate) fn mark_encoded(&mut self, end_pos: usize) {
        crate::carrier_validation::checked_slot_range_end(
            "attention encoded KV",
            0,
            end_pos,
            self.kv.capacity,
        );
        self.kv.filled = end_pos;
    }

    /// chain commit 후: 방금 verify pass 가 device append 한 window slot
    /// `[start_slot .. start_slot+count)` 의 **post-rope f16 bits** K/V 를 host 로 readback.
    /// MTP verify accept-n 커밋에서 엔진이 host kv_cache 에 append 하는 데 쓴다(host =
    /// source of truth). 반환은 slot-major contiguous `[count*kv_dim]`(slot i 는
    /// `[i*kv_dim..]`), K/V 각각. `kv_append` 이 쓴 그대로의 f16 bits(추가 변환 없음).
    pub(crate) fn readback_kv_slots(
        &self,
        start_slot: usize,
        count: usize,
    ) -> (Vec<u16>, Vec<u16>) {
        crate::carrier_validation::assert_exact_len(
            "attention KV dimension",
            self.kv_dim,
            self.kv.kv_dim,
        );
        crate::carrier_validation::checked_slot_range_end(
            "attention KV readback",
            start_slot,
            count,
            self.kv.capacity,
        );
        let start = crate::carrier_validation::checked_product(
            "attention KV readback offset",
            start_slot,
            self.kv.kv_dim,
        );
        let len = crate::carrier_validation::checked_product(
            "attention KV readback length",
            count,
            self.kv.kv_dim,
        );
        unsafe {
            let kp = self.kv.k_buf.contents().as_ptr() as *const u16;
            let vp = self.kv.v_buf.contents().as_ptr() as *const u16;
            (
                std::slice::from_raw_parts(kp.add(start), len).to_vec(),
                std::slice::from_raw_parts(vp.add(start), len).to_vec(),
            )
        }
    }
}

/// Batched attention core encode (milestone 4). `shared_hidden` = `[b*hidden_dim]`(lane i =
/// row i). B lane 의 attention core(norm → q/k/v → split → qk-norm → rope → kv-append →
/// attn → gate → o → residual)를 한 encoder 에 encode 하되 q/k/v/o weight 는 B-column GEMV 로
/// **1회만** 읽는다. lane i 는 pos=base_pos+i 에서 `[0..base_pos+i]` 를 attend. 결과는 lane 별
/// 검증된 single-token `attn_chain_encode_core`(f16 KV 경로)와 동일(reduction 순서 차 rel<3e-3).
/// MoE 는 caller 가 lane 별로 이어 붙인다. f16 KV 전용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_core_chain_encode_bcol(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &AttnBatchCarrier,
    shared_hidden: &ProtocolObject<dyn MTLBuffer>,
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
    q_q: u8,
    k_q: u8,
    v_q: u8,
    o_q: u8,
    base_pos: usize,
) {
    let b = carrier.b;
    let hidden_dim = carrier.hidden_dim;
    let q_dim = carrier.q_dim;
    let q_out_dim = carrier.q_out_dim;
    let kv_dim = carrier.kv_dim;
    let num_heads = carrier.num_heads;
    let num_kv_heads = carrier.num_kv_heads;
    let head_dim = carrier.head_dim;
    let f32b = std::mem::size_of::<f32>();
    crate::carrier_validation::assert_exact_len(
        "attention KV dimension",
        kv_dim,
        carrier.kv.kv_dim,
    );
    crate::carrier_validation::checked_slot_range_end(
        "attention KV encode",
        base_pos,
        b,
        carrier.kv.capacity,
    );

    // per-lane pos/kv_len scalar buffer는 carrier와 함께 재사용한다. verify round마다
    // tiny MTLBuffer를 다시 만들면 batch 경로의 command-encode 절감분을 상쇄한다.
    for i in 0..b {
        let pos = u32::try_from(base_pos + i).expect("validated attention KV position");
        let kv_len = u32::try_from(base_pos + i + 1).expect("validated attention KV length");
        unsafe {
            (carrier.pos_bufs[i].contents().as_ptr() as *mut u32).write(pos);
            (carrier.kl_bufs[i].contents().as_ptr() as *mut u32).write(kv_len);
            (carrier.kv_lens_buf.contents().as_ptr() as *mut u32)
                .add(i)
                .write(kv_len);
        }
    }

    // 1. all-lane rms_norm: shared_hidden -> normed_all.
    encode_rms_norm_batch(
        ctx,
        enc,
        shared_hidden,
        norm_w_buf,
        &carrier.normed_all,
        &carrier.hdim_buf,
        &carrier.eps_buf,
        b,
    );
    chain_barrier(ctx, enc);

    // 2. B-column q/k/v(weight 1회): normed_all -> q_full_all / k_all / v_all.
    encode_gemv_quant_bcol(
        ctx,
        enc,
        q_q,
        q_w_buf,
        &carrier.normed_all,
        &carrier.q_full_all,
        &carrier.qoutdim_buf,
        &carrier.k_hidden_buf,
        q_off_buf,
        &carrier.b_buf,
        q_out_dim,
    );
    encode_gemv_quant_bcol(
        ctx,
        enc,
        k_q,
        k_w_buf,
        &carrier.normed_all,
        &carrier.k_all,
        &carrier.kvdim_buf,
        &carrier.k_hidden_buf,
        k_off_buf,
        &carrier.b_buf,
        kv_dim,
    );
    encode_gemv_quant_bcol(
        ctx,
        enc,
        v_q,
        v_w_buf,
        &carrier.normed_all,
        &carrier.v_all,
        &carrier.kvdim_buf,
        &carrier.k_hidden_buf,
        v_off_buf,
        &carrier.b_buf,
        kv_dim,
    );
    chain_barrier(ctx, enc);

    // 3. all-lane split_qgate. prefill helper가 비활성화된 진단 구성은 기존 lane 경로.
    if ctx.prefill_split_gate_pipeline.is_some() {
        encode_prefill_split_q_gate(
            ctx,
            enc,
            &carrier.q_full_all,
            &carrier.q_all,
            &carrier.gate_all,
            &carrier.b_buf,
            &carrier.nh_buf,
            &carrier.hd_buf,
            b * q_dim,
        );
    } else {
        for i in 0..b {
            encode_split_qgate_at(
                ctx,
                enc,
                &carrier.q_full_all,
                i * q_out_dim * f32b,
                &carrier.q_all,
                i * q_dim * f32b,
                &carrier.gate_all,
                i * q_dim * f32b,
                &carrier.hd_buf,
                num_heads,
                q_dim / num_heads,
            );
        }
    }
    chain_barrier(ctx, enc);

    // 4-5. q/k per-head RMSNorm + RoPE를 q와 k 각 한 dispatch로 처리한다.
    encode_attn_decode_qk_norm_rope_batch(
        ctx,
        enc,
        &carrier.q_all,
        q_norm_w_buf,
        &carrier.nh_buf,
        &carrier.hd_buf,
        &carrier.nrot_buf,
        &carrier.theta_scale_buf,
        &carrier.eps_buf,
        &carrier.pos_bufs[0],
        &carrier.b_buf,
        b,
        num_heads,
    );
    encode_attn_decode_qk_norm_rope_batch(
        ctx,
        enc,
        &carrier.k_all,
        k_norm_w_buf,
        &carrier.nkv_buf,
        &carrier.hd_buf,
        &carrier.nrot_buf,
        &carrier.theta_scale_buf,
        &carrier.eps_buf,
        &carrier.pos_bufs[0],
        &carrier.b_buf,
        b,
        num_kv_heads,
    );
    chain_barrier(ctx, enc);

    // 6. all-lane kv_append: k_all/v_all -> KV[base_pos..base_pos+b].
    encode_kv_append_batch(
        ctx,
        enc,
        &carrier.k_all,
        &carrier.v_all,
        &carrier.kv.k_buf,
        &carrier.kv.v_buf,
        &carrier.kvdim_buf,
        &carrier.pos_bufs[0],
        &carrier.b_buf,
        b * kv_dim,
    );
    chain_barrier(ctx, enc);

    // 7. attention: 모든 lane이 긴-context면 part/reduce를 각각 한 dispatch로 합친다.
    // threshold를 가로지르는 한 window만 lane별 single-token 정책으로 처리한다.
    let first_kv_len = base_pos + 1;
    if ctx.attn_splitk_splits > 1 && first_kv_len >= ctx.attn_splitk_min_kv {
        encode_attn_decode_splitk_batch(
            ctx,
            enc,
            &carrier.q_all,
            &carrier.kv.k_buf,
            &carrier.kv.v_buf,
            carrier
                .attn_splitk_acc_all
                .as_ref()
                .expect("batch splitk acc buffer missing"),
            carrier
                .attn_splitk_m_all
                .as_ref()
                .expect("batch splitk m buffer missing"),
            carrier
                .attn_splitk_s_all
                .as_ref()
                .expect("batch splitk s buffer missing"),
            &carrier.attn_out_all,
            &carrier.nh_buf,
            &carrier.nkv_buf,
            &carrier.hd_buf,
            &carrier.kv_lens_buf,
            &carrier.scale_buf,
            carrier
                .attn_splitk_splits_buf
                .as_ref()
                .expect("batch splitk splits buffer missing"),
            &carrier.b_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            ctx.attn_splitk_splits,
            b,
        );
    } else {
        for i in 0..b {
            let kv_len = base_pos + i + 1;
            if ctx.attn_splitk_splits > 1 && kv_len >= ctx.attn_splitk_min_kv {
                let partial_acc_off = i * ctx.attn_splitk_splits * num_heads * head_dim * f32b;
                let partial_stats_off = i * ctx.attn_splitk_splits * num_heads * f32b;
                encode_attn_decode_splitk_at(
                    ctx,
                    enc,
                    &carrier.q_all,
                    i * q_dim * f32b,
                    &carrier.kv.k_buf,
                    &carrier.kv.v_buf,
                    carrier
                        .attn_splitk_acc_all
                        .as_ref()
                        .expect("batch splitk acc buffer missing"),
                    partial_acc_off,
                    carrier
                        .attn_splitk_m_all
                        .as_ref()
                        .expect("batch splitk m buffer missing"),
                    carrier
                        .attn_splitk_s_all
                        .as_ref()
                        .expect("batch splitk s buffer missing"),
                    partial_stats_off,
                    &carrier.attn_out_all,
                    i * q_dim * f32b,
                    &carrier.nh_buf,
                    &carrier.nkv_buf,
                    &carrier.hd_buf,
                    &carrier.kl_bufs[i],
                    &carrier.scale_buf,
                    carrier
                        .attn_splitk_splits_buf
                        .as_ref()
                        .expect("batch splitk splits buffer missing"),
                    num_heads,
                    head_dim,
                    ctx.attn_splitk_splits,
                );
            } else {
                encode_attn_decode_at(
                    ctx,
                    enc,
                    &carrier.q_all,
                    i * q_dim * f32b,
                    &carrier.kv.k_buf,
                    &carrier.kv.v_buf,
                    &carrier.attn_out_all,
                    i * q_dim * f32b,
                    &carrier.nh_buf,
                    &carrier.nkv_buf,
                    &carrier.hd_buf,
                    &carrier.kl_bufs[i],
                    &carrier.scale_buf,
                    num_heads,
                );
            }
        }
    }
    chain_barrier(ctx, enc);

    // 8. all-lane gate_apply. 진단 구성에서 prefill helper가 없으면 기존 lane 경로.
    if ctx.prefill_gate_apply_pipeline.is_some() {
        encode_prefill_gate_apply(
            ctx,
            enc,
            &carrier.attn_out_all,
            &carrier.gate_all,
            &carrier.attn_out_all,
            &carrier.q_all_elems_buf,
            b * q_dim,
        );
    } else {
        for i in 0..b {
            encode_gate_apply_at(
                ctx,
                enc,
                &carrier.attn_out_all,
                i * q_dim * f32b,
                &carrier.gate_all,
                i * q_dim * f32b,
                &carrier.qdim_buf,
                q_dim,
            );
        }
    }
    chain_barrier(ctx, enc);

    // 9. B-column o(weight 1회): attn_out_all -> o_out_all (N=hidden_dim, K=q_dim).
    encode_gemv_quant_bcol(
        ctx,
        enc,
        o_q,
        o_w_buf,
        &carrier.attn_out_all,
        &carrier.o_out_all,
        &carrier.hdim_buf,
        &carrier.k_qdim_buf,
        o_off_buf,
        &carrier.b_buf,
        hidden_dim,
    );
    chain_barrier(ctx, enc);

    // 10. all-lane residual: shared_hidden += o_out_all.
    encode_residual_add(
        ctx,
        enc,
        shared_hidden,
        &carrier.o_out_all,
        &carrier.hidden_all_buf,
        b * hidden_dim,
    );
    chain_barrier(ctx, enc);

    let _ = head_dim;
}

#[cfg(test)]
mod tests {
    use super::{gqa_group_requested, int8_gqa_group_enabled, int8_gqa_matrix_requested};

    #[test]
    fn gqa_group_requires_explicit_opt_in() {
        assert!(!gqa_group_requested(None));
        assert!(!gqa_group_requested(Some("0")));
        assert!(!gqa_group_requested(Some("false")));
        assert!(gqa_group_requested(Some("1")));
    }

    #[test]
    fn int8_gqa_group_is_default_on_with_explicit_opt_out() {
        assert!(int8_gqa_group_enabled(None));
        assert!(!int8_gqa_group_enabled(Some("0")));
        assert!(!int8_gqa_group_enabled(Some("false")));
        assert!(!int8_gqa_group_enabled(Some("OFF")));
        assert!(!int8_gqa_group_enabled(Some("no")));
        assert!(int8_gqa_group_enabled(Some("1")));
    }

    #[test]
    fn int8_gqa_matrix_requires_explicit_opt_in() {
        assert!(!int8_gqa_matrix_requested(None));
        assert!(!int8_gqa_matrix_requested(Some("0")));
        assert!(!int8_gqa_matrix_requested(Some("false")));
        assert!(int8_gqa_matrix_requested(Some("1")));
    }
}
