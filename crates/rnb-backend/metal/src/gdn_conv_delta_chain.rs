//! pm45 M2: GDN prefill conv→delta device-resident chain. conv1d_silu(GPU) 와
//! delta_net_scan(GPU) 사이의 CPU 재배치 4개(split_conv_qkv, l2_norm, repeat_qk, scale)를
//! 단일 MTLComputeCommandEncoder 에 연속 encode 해 op 별 host 왕복(conv_data readback +
//! q/k/v upload)을 제거한다. M1(gated→ssm_out fused)과 같은 패턴 — 1 commit/wait/readback.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions,
};
use std::cell::Cell;

use crate::compute::{self, MetalContext};
pub(crate) use crate::ffn_chain::{empty_f16_buf, empty_f32_buf, u32_buf};
pub(crate) use crate::gdn_proj_chain::TensoropsQuant;

thread_local! {
    static PREFILL_GDN_FULL_PROFILE: Cell<(usize, f64, f64, f64, f64)> =
        const { Cell::new((0, 0.0, 0.0, 0.0, 0.0)) };
    static PREFILL_GDN_FULL_FFN_PROFILE: Cell<(usize, f64, f64, f64, f64)> =
        const { Cell::new((0, 0.0, 0.0, 0.0, 0.0)) };
}

fn prefill_gdn_ar_enabled() -> bool {
    std::env::var("RNB_METAL_PREFILL_GDN_AR")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

/// conv→delta chain 의 device-resident 중간 buffer + scalar 묶음. shape 별 1회 alloc 후 재사용.
/// gate/beta/state/conv_weight 는 conv 와 무관(in_proj 후 별도)하거나 매 호출 갱신이라 host upload.
pub(crate) struct PrefillGdnConvDeltaCarrier {
    pub seq_len: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub chunk_size: usize,
    pub norm_eps: f32,
    // 중간 device buffer (전부 carrier 잔류 — host 안 거침).
    pub conv_input_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [(seq+ks-1)*conv_channels]
    pub conv_data_dev: Retained<ProtocolObject<dyn MTLBuffer>>,  // [seq*conv_channels]
    pub q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,          // [seq*q_dim]
    pub k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,          // [seq*k_dim]
    pub v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,          // [seq*v_dim]
    pub q_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // [seq*q_dim] (scale=1/√hk 흡수)
    pub k_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // [seq*k_dim]
    pub q_final_dev: Retained<ProtocolObject<dyn MTLBuffer>>,    // [seq*num_v_heads*head_k_dim]
    pub k_final_dev: Retained<ProtocolObject<dyn MTLBuffer>>,    // [seq*num_v_heads*head_k_dim]
    pub gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,       // [seq*num_v_heads]
    pub beta_dev: Retained<ProtocolObject<dyn MTLBuffer>>,       // [seq*num_v_heads]
    pub state_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [num_v_heads*head_v_dim*head_k_dim]
    pub conv_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [ks*conv_channels]
    pub out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // [seq*num_v_heads*head_v_dim]
    // scalar device buffer.
    pub seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq_len)
    pub cc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(conv_channels)
    pub ks_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(conv_kernel)
    pub q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(q_dim)
    pub k_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(k_dim)
    pub v_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(v_dim)
    pub nk_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(num_k_heads)
    pub nv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(num_v_heads)
    pub hk_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(head_k_dim)
    pub hv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(head_v_dim)
    pub cs_buf: Retained<ProtocolObject<dyn MTLBuffer>>,  // u32(chunk_size)
    pub eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // f32(norm_eps)
    pub q_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // f32(1/√head_k_dim) — l2_norm(q) 흡수
    pub k_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // f32(1.0)
}

impl PrefillGdnConvDeltaCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        norm_eps: f32,
    ) -> Self {
        let q_dim = head_k_dim * num_k_heads;
        let k_dim = q_dim;
        let v_dim = head_v_dim * num_v_heads;
        let total_conv_len = (conv_kernel - 1) + seq_len;
        let final_len = seq_len * num_v_heads * head_k_dim;
        let f32_scalar = |v: f32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let p = std::ptr::NonNull::new(&v as *const f32 as *mut std::ffi::c_void).unwrap();
                ctx.device
                    .newBufferWithBytes_length_options(p, 4, MTLResourceOptions::StorageModeShared)
                    .expect("f32 scalar buf")
            }
        };
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        Self {
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            norm_eps,
            conv_input_dev: empty_f32_buf(ctx, total_conv_len * conv_channels),
            conv_data_dev: empty_f32_buf(ctx, seq_len * conv_channels),
            q_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_dev: empty_f32_buf(ctx, seq_len * k_dim),
            v_dev: empty_f32_buf(ctx, seq_len * v_dim),
            q_normed_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_normed_dev: empty_f32_buf(ctx, seq_len * k_dim),
            q_final_dev: empty_f32_buf(ctx, final_len),
            k_final_dev: empty_f32_buf(ctx, final_len),
            gate_dev: empty_f32_buf(ctx, seq_len * num_v_heads),
            beta_dev: empty_f32_buf(ctx, seq_len * num_v_heads),
            state_dev: empty_f32_buf(ctx, num_v_heads * head_v_dim * head_k_dim),
            conv_w_dev: empty_f32_buf(ctx, conv_kernel * conv_channels),
            out_dev: empty_f32_buf(ctx, seq_len * num_v_heads * head_v_dim),
            seq_buf: u32_buf(ctx, seq_len as u32),
            cc_buf: u32_buf(ctx, conv_channels as u32),
            ks_buf: u32_buf(ctx, conv_kernel as u32),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            k_dim_buf: u32_buf(ctx, k_dim as u32),
            v_dim_buf: u32_buf(ctx, v_dim as u32),
            nk_buf: u32_buf(ctx, num_k_heads as u32),
            nv_buf: u32_buf(ctx, num_v_heads as u32),
            hk_buf: u32_buf(ctx, head_k_dim as u32),
            hv_buf: u32_buf(ctx, head_v_dim as u32),
            cs_buf: u32_buf(ctx, chunk_size as u32),
            eps_buf: f32_scalar(norm_eps),
            q_scale_buf: f32_scalar(q_scale),
            k_scale_buf: f32_scalar(1.0),
        }
    }

    /// host 입력을 device buffer 로 1회 업로드. conv_input/conv_weight/gate/beta/state.
    fn upload(
        &self,
        conv_input: &[f32],
        conv_weight: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
    ) {
        let total_conv_len = (self.conv_kernel - 1) + self.seq_len;
        debug_assert_eq!(
            conv_input.len(),
            total_conv_len * self.conv_channels,
            "conv_input len"
        );
        debug_assert_eq!(
            conv_weight.len(),
            self.conv_kernel * self.conv_channels,
            "conv_weight len"
        );
        debug_assert_eq!(gate.len(), self.seq_len * self.num_v_heads, "gate len");
        debug_assert_eq!(beta.len(), self.seq_len * self.num_v_heads, "beta len");
        debug_assert_eq!(
            state.len(),
            self.num_v_heads * self.head_v_dim * self.head_k_dim,
            "state len"
        );
        let cp = |src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>| unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.contents().as_ptr() as *mut f32,
                src.len(),
            );
        };
        cp(conv_input, &self.conv_input_dev);
        cp(conv_weight, &self.conv_w_dev);
        cp(gate, &self.gate_dev);
        cp(beta, &self.beta_dev);
        cp(state, &self.state_dev);
    }
}

/// pm45 M2: conv1d_silu → split_conv_qkv → l2_norm(q,k) → repeat_qk → delta_net_scan_chunk 를
/// 단일 MTLComputeCommandEncoder 에 연속 encode. RAW hazard 자동 직렬화(barrier 불요),
/// host 왕복 1회(upload → commit/wait → out+state readback).
/// scale(q 만 1/√head_k_dim)은 l2_norm(q)의 scale 인자로 흡수(CPU 는 l2_norm→repeat→scale
/// 순이지만 scale 은 elementwise 라 l2_norm(q)에 흡수해도 동일 — repeat 는 copy).
/// 반환 `(out[seq*num_v_heads*head_v_dim], state_after)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_conv_delta_chain_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnConvDeltaCarrier,
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    carrier.upload(conv_input, conv_weight, gate, beta, state);
    let seq = carrier.seq_len;
    let conv_out_len = seq * carrier.conv_channels;
    // l2_norm row 정의: q_tensor=[seq*num_k_heads, head_k_dim].
    let norm_rows = seq * carrier.num_k_heads;

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder"); // 단일 encoder = RAW hazard 자동 직렬화

    // 1) conv1d_silu(batch): conv_input → conv_data.
    compute::encode_ssm_conv1d_silu_batch(
        ctx,
        &enc,
        &carrier.conv_input_dev,
        &carrier.conv_w_dev,
        &carrier.conv_data_dev,
        &carrier.cc_buf,
        &carrier.ks_buf,
        &carrier.seq_buf,
        conv_out_len,
    );

    // 2) split_conv_qkv: conv_data → q/k/v (device 3-way 분리, conv_offset=0).
    compute::encode_split_conv_qkv(
        ctx,
        &enc,
        &carrier.conv_data_dev,
        0,
        &carrier.q_dev,
        &carrier.k_dev,
        &carrier.v_dev,
        &carrier.seq_buf,
        &carrier.cc_buf,
        &carrier.q_dim_buf,
        &carrier.k_dim_buf,
        &carrier.v_dim_buf,
        seq,
        carrier.conv_channels,
    );

    // 3) l2_norm(q, scale=1/√head_k_dim) + l2_norm(k, scale=1.0). row=norm_rows, dim=head_k_dim.
    compute::encode_l2_norm(
        ctx,
        &enc,
        &carrier.q_dev,
        0,
        &carrier.q_normed_dev,
        0,
        &carrier.hk_buf,
        &carrier.eps_buf,
        &carrier.q_scale_buf,
        norm_rows,
    );
    compute::encode_l2_norm(
        ctx,
        &enc,
        &carrier.k_dev,
        0,
        &carrier.k_normed_dev,
        0,
        &carrier.hk_buf,
        &carrier.eps_buf,
        &carrier.k_scale_buf,
        norm_rows,
    );

    // 4) repeat_qk: q_normed/k_normed → q_final/k_final (num_v_heads 로 GQA 순환 repeat).
    //    num_v==num_k 면 identity copy.
    compute::encode_repeat_qk(
        ctx,
        &enc,
        &carrier.q_normed_dev,
        &carrier.k_normed_dev,
        &carrier.q_final_dev,
        &carrier.k_final_dev,
        &carrier.seq_buf,
        &carrier.nk_buf,
        &carrier.nv_buf,
        &carrier.hk_buf,
        seq,
        carrier.num_v_heads,
        carrier.head_k_dim,
    );

    // 5) delta_net_scan_chunk: q_final/k_final/v/gate/beta/state → out + state(in-place).
    //    num_heads = num_v_heads(repeat 후), state 는 in-place hand-off.
    compute::encode_delta_net_scan_chunk(
        ctx,
        &enc,
        &carrier.q_final_dev,
        &carrier.k_final_dev,
        &carrier.v_dev,
        &carrier.gate_dev,
        &carrier.beta_dev,
        &carrier.state_dev,
        &carrier.out_dev,
        &carrier.seq_buf,
        &carrier.hk_buf,
        &carrier.hv_buf,
        &carrier.cs_buf,
        &carrier.nv_buf,
        carrier.num_v_heads,
        carrier.head_v_dim,
        carrier.chunk_size,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let out_len = seq * carrier.num_v_heads * carrier.head_v_dim;
    let state_len = carrier.num_v_heads * carrier.head_v_dim * carrier.head_k_dim;
    let out = {
        let c = carrier.out_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, out_len) }.to_vec()
    };
    let state_after = {
        let c = carrier.state_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, state_len) }.to_vec()
    };
    (out, state_after)
}

/// pm45 M3-1: M2(conv→delta) + M1(gated→ssm_out) 통합 device-resident chain. M2 chain 끝
/// delta output(device, [seq*num_v_heads*head_v_dim])을 readback 하지 않고 같은 encoder 에
/// 이어서 gated_rmsnorm_silu → cast f32→f16 → ssm_out Q5_K(또는 Q4_K/Q6_K) GEMM_v2 를 encode.
/// M2 의 모든 중간 buffer(`PrefillGdnConvDeltaCarrier` 재사용) + M1 단계 buffer(gated/gated_f16/proj)
/// + M1 scalar(ssm_norm/eps/n/k/m/cols/gated_elems)를 통합 carrier 에 잔류. host 왕복 1회
/// (upload → commit/wait → proj+state readback). delta output 은 device 잔류(readback 제거).
pub(crate) struct PrefillGdnFullCarrier {
    /// M2 stage(conv→delta) 의 모든 device buffer + scalar. out_dev 가 M1 입력으로 직결.
    pub inner: PrefillGdnConvDeltaCarrier,
    // M1 stage(gated→ssm_out) device buffer.
    pub n_out: usize,                                   // ssm_out proj 출력 rows
    pub d_inner: usize,                                 // = num_v_heads * head_v_dim
    pub z_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] gate(z)
    pub ssm_norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [head_v_dim] ssm_norm weight
    pub gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] gated 출력
    pub gated_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] half
    pub proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*n_out] ssm_out proj
    // M1 scalar device buffer.
    pub gated_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq*d_inner)
    pub n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,           // u32(n_out)
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,           // u32(d_inner)
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>,           // u32(seq_len)
    // pm46 ③ 2단계: STEP5 GEMM device workspace (full chain only, KPAD=48). hv=hk=128 시만 사용.
    pub su_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub kh_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub step5_temp_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    // pm47 ② STEP4 GEMM device workspace (a_cat[nv*48*176], b_cat[nv*176*hv] half, step4_temp[nv*48*hv] f32).
    pub a_cat_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub b_cat_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub step4_temp_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillGdnFullCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        n_out: usize,
        norm_eps: f32,
    ) -> Self {
        let inner = PrefillGdnConvDeltaCarrier::new(
            ctx,
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            norm_eps,
        );
        let d_inner = num_v_heads * head_v_dim;
        Self {
            inner,
            n_out,
            d_inner,
            z_dev: empty_f32_buf(ctx, seq_len * d_inner),
            ssm_norm_dev: empty_f32_buf(ctx, head_v_dim),
            gated_dev: empty_f32_buf(ctx, seq_len * d_inner),
            gated_f16_dev: empty_f16_buf(ctx, seq_len * d_inner),
            proj_dev: empty_f32_buf(ctx, seq_len * n_out),
            gated_elems_buf: u32_buf(ctx, (seq_len * d_inner) as u32),
            n_buf: u32_buf(ctx, n_out as u32),
            k_buf: u32_buf(ctx, d_inner as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
            // STEP5 GEMM workspace: su_half[nv*hv*48], kh_half[nv*48*hk] (half), temp[nv*hv*hk] (f32).
            su_half_dev: empty_f16_buf(ctx, num_v_heads * head_v_dim * 48),
            kh_half_dev: empty_f16_buf(ctx, num_v_heads * 48 * head_k_dim),
            step5_temp_dev: empty_f32_buf(ctx, num_v_heads * head_v_dim * head_k_dim),
            // STEP4 GEMM workspace: a_cat[nv*CPAD48*KPAD176], b_cat[nv*KPAD176*hv] (half),
            // step4_temp[nv*CPAD48*hv] (f32). hv=hk=128 시만 사용.
            a_cat_half_dev: empty_f16_buf(ctx, num_v_heads * 48 * 176),
            b_cat_half_dev: empty_f16_buf(ctx, num_v_heads * 176 * head_v_dim),
            step4_temp_dev: empty_f32_buf(ctx, num_v_heads * 48 * head_v_dim),
        }
    }

    /// M1 입력(z, ssm_norm) device upload. M2 입력(conv_input/weight/gate/beta/state)은
    /// dispatch 안에서 inner.upload 로 처리.
    fn upload_m1(&self, z: &[f32], ssm_norm: &[f32]) {
        debug_assert_eq!(z.len(), self.inner.seq_len * self.d_inner, "z len");
        debug_assert_eq!(ssm_norm.len(), self.inner.head_v_dim, "ssm_norm len");
        unsafe {
            std::ptr::copy_nonoverlapping(
                z.as_ptr(),
                self.z_dev.contents().as_ptr() as *mut f32,
                z.len(),
            );
            std::ptr::copy_nonoverlapping(
                ssm_norm.as_ptr(),
                self.ssm_norm_dev.contents().as_ptr() as *mut f32,
                ssm_norm.len(),
            );
        }
    }
}

pub(crate) struct PrefillGdnFullFfnCarrier {
    pub full: PrefillGdnFullCarrier,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub post_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub post_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub post_normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_act_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub norm_eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillGdnFullFfnCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        conv_channels: usize,
        conv_kernel: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        chunk_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_eps: f32,
    ) -> Self {
        let full = PrefillGdnFullCarrier::new(
            ctx,
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            hidden_dim,
            norm_eps,
        );
        Self {
            full,
            hidden_dim,
            ffn_dim,
            hidden_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            post_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            post_normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            post_normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            ffn_gate_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_up_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_act_f16_dev: empty_f16_buf(ctx, seq_len * ffn_dim),
            ffn_down_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            hidden_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            ffn_act_elems_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            ffn_dim_buf: u32_buf(ctx, ffn_dim as u32),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
            norm_eps_buf: crate::ffn_chain::f32_buf(ctx, norm_eps),
        }
    }

    fn upload_tail_inputs(&self, hidden: &[f32], post_norm_w: &[f32]) {
        debug_assert_eq!(
            hidden.len(),
            self.full.inner.seq_len * self.hidden_dim,
            "hidden len"
        );
        debug_assert_eq!(post_norm_w.len(), self.hidden_dim, "post_norm_w len");
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.hidden_dev.contents().as_ptr() as *mut f32,
                hidden.len(),
            );
            std::ptr::copy_nonoverlapping(
                post_norm_w.as_ptr(),
                self.post_norm_w_dev.contents().as_ptr() as *mut f32,
                post_norm_w.len(),
            );
        }
    }
}

#[derive(Clone, Copy)]
enum PrefillGdnSsmProjection<'a> {
    ResidentQuant {
        weight: &'a ProtocolObject<dyn MTLBuffer>,
        offset: u32,
        quant: TensoropsQuant,
    },
    Backend(crate::GdnBackendWeightRef<'a>),
}

struct PrefillGdnStageResult {
    delta_path: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_gdn_full_stage(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillGdnFullCarrier,
    conv_weight_dev: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_dev: &ProtocolObject<dyn MTLBuffer>,
    state_dev: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_weight: crate::GdnBackendWeightRef<'_>,
) -> Result<(), String> {
    encode_gdn_full_m2_m1_stage(
        ctx,
        enc,
        carrier,
        conv_weight_dev,
        ssm_norm_dev,
        state_dev,
        false,
        PrefillGdnSsmProjection::Backend(ssm_out_weight),
    )
    .map(|_| ())
}

pub(crate) fn encode_qwen_gdn_conv_state_in(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillGdnFullCarrier,
    state_in: &ProtocolObject<dyn MTLBuffer>,
) {
    let inner = &carrier.inner;
    let state_rows = inner.conv_kernel - 1;
    let state_rows_buf = u32_buf(ctx, state_rows as u32);
    let zero_buf = u32_buf(ctx, 0);
    compute::encode_split_conv_qkv(
        ctx,
        enc,
        state_in,
        0,
        &inner.conv_input_dev,
        &inner.k_dev,
        &inner.v_dev,
        &state_rows_buf,
        &inner.cc_buf,
        &inner.cc_buf,
        &zero_buf,
        &zero_buf,
        state_rows,
        inner.conv_channels,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_gdn_conv_state_out(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillGdnFullCarrier,
    state_out: &ProtocolObject<dyn MTLBuffer>,
) {
    let inner = &carrier.inner;
    let state_rows_buf = u32_buf(ctx, (inner.conv_kernel - 1) as u32);
    let zero_buf = u32_buf(ctx, 0);
    compute::encode_split_conv_qkv(
        ctx,
        enc,
        &inner.conv_input_dev,
        inner.seq_len * inner.conv_channels * std::mem::size_of::<f32>(),
        state_out,
        &inner.k_dev,
        &inner.v_dev,
        &state_rows_buf,
        &inner.cc_buf,
        &inner.cc_buf,
        &zero_buf,
        &zero_buf,
        inner.conv_kernel - 1,
        inner.conv_channels,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_gdn_full_m2_m1_stage(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillGdnFullCarrier,
    conv_weight_dev: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_dev: &ProtocolObject<dyn MTLBuffer>,
    state_dev: &ProtocolObject<dyn MTLBuffer>,
    exact_reductions: bool,
    projection: PrefillGdnSsmProjection<'_>,
) -> Result<PrefillGdnStageResult, String> {
    let inner = &carrier.inner;
    let seq = inner.seq_len;
    let conv_out_len = seq * inner.conv_channels;
    let norm_rows = seq * inner.num_k_heads;

    compute::encode_ssm_conv1d_silu_batch(
        ctx,
        enc,
        &inner.conv_input_dev,
        conv_weight_dev,
        &inner.conv_data_dev,
        &inner.cc_buf,
        &inner.ks_buf,
        &inner.seq_buf,
        conv_out_len,
    );
    compute::chain_barrier(ctx, enc);
    compute::encode_split_conv_qkv(
        ctx,
        enc,
        &inner.conv_data_dev,
        0,
        &inner.q_dev,
        &inner.k_dev,
        &inner.v_dev,
        &inner.seq_buf,
        &inner.cc_buf,
        &inner.q_dim_buf,
        &inner.k_dim_buf,
        &inner.v_dim_buf,
        seq,
        inner.conv_channels,
    );
    compute::chain_barrier(ctx, enc);
    if exact_reductions {
        compute::encode_qwen_prefill_l2_norm_exact(
            ctx,
            enc,
            &inner.q_dev,
            &inner.q_normed_dev,
            norm_rows,
            inner.head_k_dim,
            inner.norm_eps,
            1.0 / (inner.head_k_dim as f32).sqrt(),
        )
        .map_err(|error| format!("Qwen GDN q L2 encode failed: {error:?}"))?;
        compute::encode_qwen_prefill_l2_norm_exact(
            ctx,
            enc,
            &inner.k_dev,
            &inner.k_normed_dev,
            norm_rows,
            inner.head_k_dim,
            inner.norm_eps,
            1.0,
        )
        .map_err(|error| format!("Qwen GDN k L2 encode failed: {error:?}"))?;
    } else {
        compute::encode_l2_norm(
            ctx,
            enc,
            &inner.q_dev,
            0,
            &inner.q_normed_dev,
            0,
            &inner.hk_buf,
            &inner.eps_buf,
            &inner.q_scale_buf,
            norm_rows,
        );
        compute::encode_l2_norm(
            ctx,
            enc,
            &inner.k_dev,
            0,
            &inner.k_normed_dev,
            0,
            &inner.hk_buf,
            &inner.eps_buf,
            &inner.k_scale_buf,
            norm_rows,
        );
        compute::chain_barrier(ctx, enc);
    }
    compute::encode_repeat_qk(
        ctx,
        enc,
        &inner.q_normed_dev,
        &inner.k_normed_dev,
        &inner.q_final_dev,
        &inner.k_final_dev,
        &inner.seq_buf,
        &inner.nk_buf,
        &inner.nv_buf,
        &inner.hk_buf,
        seq,
        inner.num_v_heads,
        inner.head_k_dim,
    );
    compute::chain_barrier(ctx, enc);

    let use_ar = prefill_gdn_ar_enabled()
        && inner.head_k_dim % compute::SIMD_WIDTH == 0
        && matches!(inner.head_k_dim / compute::SIMD_WIDTH, 1 | 2 | 4 | 8);
    let use_step45_gemm = ctx.delta_net_scan_chunk_step45gemm_pipeline.is_some()
        && inner.head_v_dim == 128
        && inner.head_k_dim == 128
        && inner.chunk_size <= 48
        && std::env::var("RNB_METAL_PREFILL_DELTA_STEP45_GEMM").as_deref() != Ok("0");
    let delta_path = if use_ar {
        compute::encode_delta_net_scan_ar(
            ctx,
            enc,
            &inner.q_final_dev,
            &inner.k_final_dev,
            &inner.v_dev,
            &inner.gate_dev,
            &inner.beta_dev,
            state_dev,
            &inner.out_dev,
            &inner.seq_buf,
            &inner.hk_buf,
            &inner.hv_buf,
            &inner.nv_buf,
            inner.num_v_heads,
            inner.head_k_dim,
            inner.head_v_dim,
        );
        "ar"
    } else if use_step45_gemm {
        compute::encode_delta_net_scan_chunk_step45gemm(
            ctx,
            enc,
            &inner.q_final_dev,
            &inner.k_final_dev,
            &inner.v_dev,
            &inner.gate_dev,
            &inner.beta_dev,
            state_dev,
            &inner.out_dev,
            &inner.seq_buf,
            &inner.hk_buf,
            &inner.hv_buf,
            &inner.cs_buf,
            &inner.nv_buf,
            &carrier.su_half_dev,
            &carrier.kh_half_dev,
            &carrier.step5_temp_dev,
            &carrier.a_cat_half_dev,
            &carrier.b_cat_half_dev,
            &carrier.step4_temp_dev,
            inner.num_v_heads,
            inner.head_v_dim,
            inner.chunk_size,
        );
        "step45"
    } else {
        compute::encode_delta_net_scan_chunk(
            ctx,
            enc,
            &inner.q_final_dev,
            &inner.k_final_dev,
            &inner.v_dev,
            &inner.gate_dev,
            &inner.beta_dev,
            state_dev,
            &inner.out_dev,
            &inner.seq_buf,
            &inner.hk_buf,
            &inner.hv_buf,
            &inner.cs_buf,
            &inner.nv_buf,
            inner.num_v_heads,
            inner.head_v_dim,
            inner.chunk_size,
        );
        "chunk"
    };
    compute::chain_barrier(ctx, enc);

    if exact_reductions {
        compute::encode_qwen_prefill_gated_rmsnorm_silu_exact(
            ctx,
            enc,
            &inner.out_dev,
            &carrier.z_dev,
            ssm_norm_dev,
            &carrier.gated_dev,
            seq * inner.num_v_heads,
            inner.head_v_dim,
            inner.norm_eps,
        )
        .map_err(|error| format!("Qwen GDN gated RMSNorm+SiLU encode failed: {error:?}"))?;
    } else {
        compute::encode_gated_rmsnorm_silu(
            ctx,
            enc,
            &inner.out_dev,
            &carrier.z_dev,
            ssm_norm_dev,
            &carrier.gated_dev,
            &inner.hv_buf,
            &inner.eps_buf,
            seq * inner.num_v_heads,
        );
        compute::chain_barrier(ctx, enc);
    }
    let quant_projection = matches!(
        projection,
        PrefillGdnSsmProjection::ResidentQuant { .. }
            | PrefillGdnSsmProjection::Backend(crate::GdnBackendWeightRef::Quant(_))
    );
    if quant_projection {
        compute::encode_cast_f32_to_f16(
            ctx,
            enc,
            &carrier.gated_dev,
            &carrier.gated_f16_dev,
            &carrier.gated_elems_buf,
            seq * carrier.d_inner,
        );
    }
    if quant_projection {
        compute::chain_barrier(ctx, enc);
    }
    match projection {
        PrefillGdnSsmProjection::ResidentQuant {
            weight,
            offset,
            quant,
        } => match quant {
            TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
                ctx,
                enc,
                weight,
                offset,
                &carrier.gated_f16_dev,
                &carrier.proj_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                seq,
            ),
            TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops_v2(
                ctx,
                enc,
                weight,
                offset,
                &carrier.gated_f16_dev,
                &carrier.proj_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                seq,
            ),
            TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
                ctx,
                enc,
                weight,
                offset,
                &carrier.gated_f16_dev,
                &carrier.proj_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                seq,
            ),
        },
        PrefillGdnSsmProjection::Backend(weight) => {
            crate::gdn_proj_chain::encode_qwen_gdn_projection(
                ctx,
                enc,
                weight,
                &carrier.gated_dev,
                &carrier.gated_f16_dev,
                &carrier.proj_dev,
                0,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                carrier.d_inner,
                seq,
            )?;
        }
    }
    compute::chain_barrier(ctx, enc);
    Ok(PrefillGdnStageResult { delta_path })
}

/// pm45 M3-1: conv1d_silu → split → l2_norm → repeat → delta_net_scan → gated_rmsnorm_silu →
/// cast f32→f16 → ssm_out GEMM_v2 를 단일 MTLComputeCommandEncoder 에 연속 encode. M2 dispatch 와
/// 동일한 conv→delta encode 순서(같은 커널/같은 chunk_size)를 그대로 타고, delta out_dev 를 readback
/// 하지 않고 곧장 M1 encode 입력으로 직결. RAW hazard 자동 직렬화, host 왕복 1회.
/// 반환 `(proj[seq*n_out], state_after)`. delta output 은 device 잔류(readback 제거 = M3-1 이득).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_full_chain_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnFullCarrier,
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    quant: TensoropsQuant,
) -> (Vec<f32>, Vec<f32>) {
    let inner = &carrier.inner;
    let t_up0 = std::time::Instant::now();
    inner.upload(conv_input, conv_weight, gate, beta, state);
    carrier.upload_m1(z, ssm_norm);
    let t_up = t_up0.elapsed().as_secs_f64() * 1000.0;
    let off = unsafe { *(off_buf.contents().as_ptr() as *const u32) };

    let seq = inner.seq_len;
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder"); // 단일 encoder = RAW hazard 자동 직렬화
    let delta_path = encode_gdn_full_m2_m1_stage(
        ctx,
        &enc,
        carrier,
        &inner.conv_w_dev,
        &carrier.ssm_norm_dev,
        &inner.state_dev,
        false,
        PrefillGdnSsmProjection::ResidentQuant {
            weight: w_buf,
            offset: off,
            quant,
        },
    )
    .expect("standalone GDN full-chain projection")
    .delta_path;

    enc.endEncoding();
    let t_cw0 = std::time::Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let t_cw = t_cw0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = if std::env::var("RNB_METAL_PREFILL_GDN_FULL_TIME").as_deref() == Ok("1") {
        Some((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0)
    } else {
        None
    };

    // proj + state 만 readback (delta out_dev 은 device 잔류).
    let proj_len = seq * carrier.n_out;
    let state_len = inner.num_v_heads * inner.head_v_dim * inner.head_k_dim;
    let t_rb0 = std::time::Instant::now();
    let proj = {
        let c = carrier.proj_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, proj_len) }.to_vec()
    };
    let state_after = {
        let c = inner.state_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, state_len) }.to_vec()
    };
    let t_rb = t_rb0.elapsed().as_secs_f64() * 1000.0;
    if let Some(gpu_ms) = gpu_ms {
        PREFILL_GDN_FULL_PROFILE.with(|c| {
            let (cnt, gpu_sum, cw_sum, up_sum, rb_sum) = c.get();
            let next = (
                cnt + 1,
                gpu_sum + gpu_ms,
                cw_sum + t_cw,
                up_sum + t_up,
                rb_sum + t_rb,
            );
            c.set(next);
            eprintln!(
                "[prefill-gdn-full] call#{} delta_path={delta_path} gpu={gpu_ms:.3}ms cw_wall={t_cw:.3} up={t_up:.3} rb={t_rb:.3} || cumul gpu={:.1} cw_wall={:.1} up={:.1} rb={:.1} launch_overhead(cw-gpu)={:.1}ms",
                next.0,
                next.1,
                next.2,
                next.3,
                next.4,
                next.2 - next.1,
            );
        });
    }
    (proj, state_after)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_full_ffn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnFullFfnCarrier,
    hidden: &[f32],
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_quant: TensoropsQuant,
    post_norm_w: &[f32],
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_is_q6k: bool,
) -> (Vec<f32>, Vec<f32>) {
    let full = &carrier.full;
    let inner = &full.inner;
    let seq = inner.seq_len;
    let t_up0 = std::time::Instant::now();
    inner.upload(conv_input, conv_weight, gate, beta, state);
    full.upload_m1(z, ssm_norm);
    carrier.upload_tail_inputs(hidden, post_norm_w);
    let t_up = t_up0.elapsed().as_secs_f64() * 1000.0;

    let ssm_out_off = unsafe { *(ssm_out_off_buf.contents().as_ptr() as *const u32) };
    let gate_off = unsafe { *(ffn_gate_off_buf.contents().as_ptr() as *const u32) };
    let up_off = unsafe { *(ffn_up_off_buf.contents().as_ptr() as *const u32) };
    let down_off = unsafe { *(ffn_down_off_buf.contents().as_ptr() as *const u32) };

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    let _delta_path = encode_gdn_full_m2_m1_stage(
        ctx,
        &enc,
        full,
        &inner.conv_w_dev,
        &full.ssm_norm_dev,
        &inner.state_dev,
        false,
        PrefillGdnSsmProjection::ResidentQuant {
            weight: ssm_out_w_buf,
            offset: ssm_out_off,
            quant: ssm_out_quant,
        },
    )
    .expect("standalone GDN full-FFN projection")
    .delta_path;

    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &carrier.hidden_dev,
        &full.proj_dev,
        &carrier.hidden_elems_buf,
        seq * carrier.hidden_dim,
    );
    compute::encode_rms_norm_batch(
        ctx,
        &enc,
        &carrier.hidden_dev,
        &carrier.post_norm_w_dev,
        &carrier.post_normed_dev,
        &carrier.hidden_dim_buf,
        &carrier.norm_eps_buf,
        seq,
    );
    compute::encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.post_normed_dev,
        &carrier.post_normed_f16_dev,
        &carrier.hidden_elems_buf,
        seq * carrier.hidden_dim,
    );
    compute::encode_gemm_q4k_tensorops_v2(
        ctx,
        &enc,
        ffn_gate_w_buf,
        gate_off,
        &carrier.post_normed_f16_dev,
        &carrier.ffn_gate_dev,
        &carrier.ffn_dim_buf,
        &carrier.k_hidden_buf,
        &full.m_buf,
        carrier.ffn_dim,
        seq,
    );
    compute::encode_gemm_q4k_tensorops_v2(
        ctx,
        &enc,
        ffn_up_w_buf,
        up_off,
        &carrier.post_normed_f16_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_dim_buf,
        &carrier.k_hidden_buf,
        &full.m_buf,
        carrier.ffn_dim,
        seq,
    );
    compute::encode_silu_mul_to_f16(
        ctx,
        &enc,
        &carrier.ffn_gate_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_act_elems_buf,
        seq * carrier.ffn_dim,
    );
    if ffn_down_is_q6k {
        compute::encode_gemm_q6k_tensorops_v2(
            ctx,
            &enc,
            ffn_down_w_buf,
            down_off,
            &carrier.ffn_act_f16_dev,
            &carrier.ffn_down_dev,
            &carrier.hidden_dim_buf,
            &carrier.k_ffn_buf,
            &full.m_buf,
            carrier.hidden_dim,
            seq,
        );
    } else {
        compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            &enc,
            ffn_down_w_buf,
            down_off,
            &carrier.ffn_act_f16_dev,
            &carrier.ffn_down_dev,
            &carrier.hidden_dim_buf,
            &carrier.k_ffn_buf,
            &full.m_buf,
            carrier.hidden_dim,
            seq,
        );
    }
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &carrier.hidden_dev,
        &carrier.ffn_down_dev,
        &carrier.hidden_elems_buf,
        seq * carrier.hidden_dim,
    );

    enc.endEncoding();
    let t_cw0 = std::time::Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let t_cw = t_cw0.elapsed().as_secs_f64() * 1000.0;

    let t_rb0 = std::time::Instant::now();
    let hidden_after = crate::ffn_chain::readback(&carrier.hidden_dev, seq * carrier.hidden_dim);
    let state_after = crate::ffn_chain::readback(
        &inner.state_dev,
        inner.num_v_heads * inner.head_v_dim * inner.head_k_dim,
    );
    let t_rb = t_rb0.elapsed().as_secs_f64() * 1000.0;

    if std::env::var("RNB_METAL_PREFILL_GDN_FULL_FFN_TIME").as_deref() == Ok("1") {
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
        PREFILL_GDN_FULL_FFN_PROFILE.with(|c| {
            let (cnt, gpu_sum, cw_sum, up_sum, rb_sum) = c.get();
            let next = (
                cnt + 1,
                gpu_sum + gpu_ms,
                cw_sum + t_cw,
                up_sum + t_up,
                rb_sum + t_rb,
            );
            c.set(next);
            eprintln!(
                "[prefill-gdn-full-ffn] call#{} gpu={gpu_ms:.3}ms cw_wall={t_cw:.3} up={t_up:.3} rb={t_rb:.3} || cumul gpu={:.1} cw_wall={:.1} up={:.1} rb={:.1} launch_overhead(cw-gpu)={:.1}ms",
                next.0,
                next.1,
                next.2,
                next.3,
                next.4,
                next.2 - next.1,
            );
        });
    }

    (hidden_after, state_after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn prefill_gdn_ar_defaults_on_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_PREFILL_GDN_AR";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert!(prefill_gdn_ar_enabled());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(!prefill_gdn_ar_enabled(), "{value} should opt out");
        }

        std::env::set_var(key, "1");
        assert!(prefill_gdn_ar_enabled());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_real_shape_matches_legacy_stage() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        const SEQ: usize = 128;
        const CONV_CHANNELS: usize = 8192;
        const CONV_KERNEL: usize = 4;
        const NUM_K_HEADS: usize = 16;
        const NUM_V_HEADS: usize = 32;
        const HEAD_DIM: usize = 128;
        const D_INNER: usize = NUM_V_HEADS * HEAD_DIM;
        const CHUNK_SIZE: usize = 38;
        const EPS: f32 = 1e-6;

        let sample = |index: usize, modulus: usize, scale: f32| {
            ((index.wrapping_mul(37).wrapping_add(11) % modulus) as f32 - (modulus / 2) as f32)
                * scale
        };
        let conv_input = (0..(SEQ + CONV_KERNEL - 1) * CONV_CHANNELS)
            .map(|i| sample(i, 257, 1e-4))
            .collect::<Vec<_>>();
        let conv_weight = (0..CONV_KERNEL * CONV_CHANNELS)
            .map(|i| sample(i, 31, 2e-3))
            .collect::<Vec<_>>();
        let gate = (0..SEQ * NUM_V_HEADS)
            .map(|i| -0.03 - (i % 7) as f32 * 0.001)
            .collect::<Vec<_>>();
        let beta = (0..SEQ * NUM_V_HEADS)
            .map(|i| 0.01 + (i % 5) as f32 * 0.0005)
            .collect::<Vec<_>>();
        let state_initial = (0..D_INNER * HEAD_DIM)
            .map(|i| sample(i, 67, 2e-5))
            .collect::<Vec<_>>();
        let z = (0..SEQ * D_INNER)
            .map(|i| sample(i, 43, 0.01))
            .collect::<Vec<_>>();
        let ssm_norm = (0..HEAD_DIM)
            .map(|i| 0.75 + (i % 13) as f32 * 0.01)
            .collect::<Vec<_>>();
        let ssm_out_values = (0..D_INNER)
            .map(|i| sample(i, 59, 0.002))
            .collect::<Vec<_>>();
        let ssm_out_raw = rnb_cpu::quantize::quant::quantize_q4_k_vec(&ssm_out_values);
        let ssm_out_buf = unsafe {
            ctx.device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(ssm_out_raw.as_ptr() as *mut std::ffi::c_void)
                        .expect("Q4_K weight pointer"),
                    ssm_out_raw.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Q4_K weight buffer")
        };
        let ssm_out_off_buf = u32_buf(&ctx, 0);
        let carrier = PrefillGdnFullCarrier::new(
            &ctx,
            SEQ,
            CONV_CHANNELS,
            CONV_KERNEL,
            NUM_K_HEADS,
            NUM_V_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            CHUNK_SIZE,
            1,
            EPS,
        );

        carrier
            .inner
            .upload(&conv_input, &conv_weight, &gate, &beta, &state_initial);
        carrier.upload_m1(&z, &ssm_norm);
        let state_dev = empty_f32_buf(&ctx, state_initial.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                state_initial.as_ptr(),
                state_dev.contents().as_ptr() as *mut f32,
                state_initial.len(),
            );
        }
        let conv_state_len = (CONV_KERNEL - 1) * CONV_CHANNELS;
        let conv_state_dev = empty_f32_buf(&ctx, conv_state_len);
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = crate::compute::chain_compute_encoder(&ctx, &cmd);
        encode_qwen_gdn_full_stage(
            &ctx,
            &enc,
            &carrier,
            &carrier.inner.conv_w_dev,
            &carrier.ssm_norm_dev,
            &state_dev,
            crate::GdnBackendWeightRef::Quant(crate::PrefillAtnCoreWeightView {
                raw: &ssm_out_raw,
                quant: TensoropsQuant::Q4K,
                rows: 1,
                cols: D_INNER,
            }),
        )
        .expect("Task11 GDN stage encode");
        encode_qwen_gdn_conv_state_out(&ctx, &enc, &carrier, &conv_state_dev);
        compute::chain_barrier(&ctx, &enc);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let task11_hidden = crate::ffn_chain::readback(&carrier.proj_dev, SEQ);
        let task11_delta_state = crate::ffn_chain::readback(&state_dev, state_initial.len());
        let task11_conv_state = crate::ffn_chain::readback(&conv_state_dev, conv_state_len);

        let (legacy_hidden, legacy_delta_state) = prefill_gdn_full_chain_dispatch(
            &ctx,
            &carrier,
            &conv_input,
            &conv_weight,
            &gate,
            &beta,
            &state_initial,
            &z,
            &ssm_norm,
            &ssm_out_buf,
            &ssm_out_off_buf,
            TensoropsQuant::Q4K,
        );
        let legacy_conv_state = unsafe {
            std::slice::from_raw_parts(
                (carrier.inner.conv_input_dev.contents().as_ptr() as *const f32)
                    .add(SEQ * CONV_CHANNELS),
                conv_state_len,
            )
            .to_vec()
        };

        for (index, (&task11, &legacy)) in task11_hidden.iter().zip(&legacy_hidden).enumerate() {
            assert_eq!(
                task11.to_bits(),
                legacy.to_bits(),
                "hidden projection index={index}"
            );
        }
        for (index, (&task11, &legacy)) in task11_delta_state
            .iter()
            .zip(&legacy_delta_state)
            .enumerate()
        {
            assert_eq!(
                task11.to_bits(),
                legacy.to_bits(),
                "delta state index={index}"
            );
        }
        for (index, (&task11, &legacy)) in
            task11_conv_state.iter().zip(&legacy_conv_state).enumerate()
        {
            assert_eq!(
                task11.to_bits(),
                legacy.to_bits(),
                "conv state index={index}"
            );
        }
    }
}
