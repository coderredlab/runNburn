//! GDN(Gated Delta Net) layer 전체를 단일 command buffer device-resident chain 으로.
//!
//! pm16 carrier: 9B 의 24 GDN layer 의
//! `attn_norm → qkv/gate/alpha/beta GEMV → alpha/beta 전처리 → conv1d_silu →
//!  q/k l2_norm(+q scale) → delta_net(GQA) → ssm_norm + z silu·mul → ssm_out GEMV
//!  + residual → FFN` 을 hidden·conv_state·delta_state 를 device buffer 에 두고
//! layer 당 command buffer 1개로 돌린다. per-op dispatch overhead 제거.
//!
//! state(conv_state·delta_state)는 dispatch 경계에서 host(KVCache ssm_state)와
//! 양방향 sync — host 가 source of truth(fallback 일관). 단일 command buffer 안에서는
//! 모든 중간값이 device 에 남아 chain 이 안 끊긴다.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::compute::{
    self, chain_barrier, chain_compute_encoder, encode_delta_net_step, encode_gdn_alpha_beta,
    encode_gemv_quant, encode_l2_norm, encode_qk_norm, encode_ssm_conv1d_silu,
    gemv_coalesced_dispatch_for, gemv_pipeline_for, gemv_simd_pipeline_for, MetalContext,
    SIMD_WIDTH,
};
use crate::ffn_chain::{
    empty_f16_buf, empty_f32_buf, encode_residual_add, encode_rms_norm, encode_silu_mul, f32_buf,
    readback, u32_buf,
};
use crate::gdn_conv_delta_chain::PrefillGdnFullCarrier;

/// GDN layer 의 device-resident 중간 버퍼 + 불변 scalar 버퍼.
/// shape 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local.
/// (일부 shape 필드는 M5 layer-hidden 상주 확장 때 참조용으로 보관.)
#[allow(dead_code)]
pub(crate) struct GdnCarrier {
    pub hidden_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub conv_state_len: usize, // (conv_kernel-1)*conv_channels
    pub z_dim: usize,          // d_inner
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub q_dim: usize, // num_k_heads * head_k_dim
    pub k_dim: usize, // num_k_heads * head_k_dim
    pub v_dim: usize, // num_v_heads * head_v_dim
    pub ffn_dim: usize,
    pub delta_state_len: usize, // num_v_heads * head_v_dim * head_k_dim

    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    conv_input_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [conv_state | qkv] = conv_kernel*conv_channels
    z_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    alpha_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    beta_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    conv_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    delta_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    delta_state_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ssm_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,

    hdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    conv_ch_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    conv_k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    zdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nk_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hk_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // 1/sqrt(head_k_dim)
    one_buf: Retained<ProtocolObject<dyn MTLBuffer>>,     // 1.0 (k l2_norm scale)
    k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // GEMV K = hidden_dim
    fdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,    // FFN gate/up N = ffn_dim
    k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,   // FFN down K = ffn_dim

    // pm31: delta state device residency. 첫 토큰에 seed 후 true → 이후 upload skip.
    // delta_state_dev 가 토큰 간 device 잔류하며 delta_net_step 이 in-place 누적.
    delta_seeded: bool,
}

impl GdnCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden_dim: usize,
        conv_channels: usize,
        conv_kernel: usize,
        z_dim: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        ffn_dim: usize,
        eps: f32,
    ) -> Self {
        let q_dim = num_k_heads * head_k_dim;
        let k_dim = num_k_heads * head_k_dim;
        let v_dim = num_v_heads * head_v_dim;
        let conv_state_len = (conv_kernel - 1) * conv_channels;
        let delta_state_len = num_v_heads * head_v_dim * head_k_dim;
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        Self {
            hidden_dim,
            conv_channels,
            conv_kernel,
            conv_state_len,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            q_dim,
            k_dim,
            v_dim,
            ffn_dim,
            delta_state_len,
            hidden_dev: empty_f32_buf(ctx, hidden_dim),
            norm_dev: empty_f32_buf(ctx, hidden_dim),
            conv_input_dev: empty_f32_buf(ctx, conv_kernel * conv_channels),
            z_dev: empty_f32_buf(ctx, z_dim),
            alpha_dev: empty_f32_buf(ctx, num_v_heads),
            beta_dev: empty_f32_buf(ctx, num_v_heads),
            conv_out_dev: empty_f32_buf(ctx, conv_channels),
            q_norm_dev: empty_f32_buf(ctx, q_dim),
            k_norm_dev: empty_f32_buf(ctx, k_dim),
            delta_out_dev: empty_f32_buf(ctx, z_dim),
            delta_state_dev: empty_f32_buf(ctx, delta_state_len),
            gated_dev: empty_f32_buf(ctx, z_dim),
            ssm_proj_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_normed_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_gate_dev: empty_f32_buf(ctx, ffn_dim),
            ffn_up_dev: empty_f32_buf(ctx, ffn_dim),
            ffn_down_dev: empty_f32_buf(ctx, hidden_dim),
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            conv_ch_buf: u32_buf(ctx, conv_channels as u32),
            conv_k_buf: u32_buf(ctx, conv_kernel as u32),
            zdim_buf: u32_buf(ctx, z_dim as u32),
            nv_buf: u32_buf(ctx, num_v_heads as u32),
            nk_buf: u32_buf(ctx, num_k_heads as u32),
            hk_buf: u32_buf(ctx, head_k_dim as u32),
            hv_buf: u32_buf(ctx, head_v_dim as u32),
            eps_buf: f32_buf(ctx, eps),
            q_scale_buf: f32_buf(ctx, q_scale),
            one_buf: f32_buf(ctx, 1.0),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            fdim_buf: u32_buf(ctx, ffn_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
            delta_seeded: false,
        }
    }

    /// host slice → device buffer(StorageModeShared contents 직접 쓰기).
    fn upload(buf: &ProtocolObject<dyn MTLBuffer>, src: &[f32]) {
        let contents = buf.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), contents.as_ptr() as *mut f32, src.len());
        }
    }

    /// device buffer 의 `[offset..offset+len]`(f32 index) 를 host Vec 으로 읽는다.
    fn readback_at(buf: &ProtocolObject<dyn MTLBuffer>, offset: usize, len: usize) -> Vec<f32> {
        let contents = buf.contents();
        unsafe {
            let p = (contents.as_ptr() as *const f32).add(offset);
            std::slice::from_raw_parts(p, len).to_vec()
        }
    }

    /// chain 진입 전 이 layer 의 state(conv·delta)를 carrier device buffer 로 올린다.
    /// hidden 은 chain 공유 buffer 라 여기서 다루지 않는다(caller 가 1회 upload).
    /// conv_state 는 `conv_input_dev[0..conv_state_len]` 에, delta_state 는 `delta_state_dev` 에.
    /// 1.3/1.4 에서 `gdn_decode_chain_run` 이 호출.
    pub(crate) fn upload_states(&self, conv_state: &[f32], delta_state: &[f32]) {
        debug_assert_eq!(conv_state.len(), self.conv_state_len);
        debug_assert_eq!(delta_state.len(), self.delta_state_len);
        Self::upload(&self.conv_input_dev, conv_state);
        Self::upload(&self.delta_state_dev, delta_state);
    }

    /// chain commit 후 이 layer 의 새 state 를 host 로 읽는다. conv 는
    /// `conv_input_dev[conv_channels..]`(offset 비대칭), delta 는 `delta_state_dev[0..]`.
    /// `gdn_chain_dispatch` 의 readback 과 동일.
    pub(crate) fn readback_states(&self) -> (Vec<f32>, Vec<f32>) {
        let conv_new = Self::readback_at(
            &self.conv_input_dev,
            self.conv_channels,
            self.conv_state_len,
        );
        let delta_new = Self::readback_at(&self.delta_state_dev, 0, self.delta_state_len);
        (conv_new, delta_new)
    }

    /// pm31 delta residency upload: conv 는 매 토큰 device 로 올리고(readback bytes 의
    /// 3.3% 라 CUDA 와 동일하게 비-resident 유지), delta 는 **첫 토큰만** seed 한다.
    /// 이후 토큰은 delta upload 를 skip → `delta_state_dev` 가 토큰 간 device 잔류하며
    /// `delta_net_step` 이 in-place 누적(host↔device 왕복 제거, 96.7% bytes).
    pub(crate) fn upload_states_resident(&mut self, conv_state: &[f32], delta_state: &[f32]) {
        debug_assert_eq!(conv_state.len(), self.conv_state_len);
        debug_assert_eq!(delta_state.len(), self.delta_state_len);
        Self::upload(&self.conv_input_dev, conv_state);
        if !self.delta_seeded {
            Self::upload(&self.delta_state_dev, delta_state);
            self.delta_seeded = true;
        }
    }

    /// pm31: 현재 device delta_state 를 host 로 읽는다(materialize / 검증용).
    pub(crate) fn readback_delta_state(&self) -> Vec<f32> {
        Self::readback_at(&self.delta_state_dev, 0, self.delta_state_len)
    }

    /// pm31: conv_state 만 host 로 읽는다(delta residency 경로 — delta 는 device 잔류라
    /// readback skip). `readback_states` 의 conv 부분과 동일(offset 비대칭 = conv_channels).
    pub(crate) fn readback_conv_state(&self) -> Vec<f32> {
        Self::readback_at(
            &self.conv_input_dev,
            self.conv_channels,
            self.conv_state_len,
        )
    }
}

/// Qwen prefill GDN scratch. The caller owns the command buffer and supplies the
/// ping-pong hidden/state outputs; this carrier only owns reusable intermediates.
pub(crate) struct QwenGdnPrefillCarrier {
    pub(crate) full: PrefillGdnFullCarrier,
    norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dt_rank_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    zero_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenGdnPrefillStageSnapshots<'a> {
    pub(crate) alpha_raw: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) beta_raw: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) alpha_processed: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) beta_processed: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) norm: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) hidden_input: &'a ProtocolObject<dyn MTLBuffer>,
}

impl QwenGdnPrefillCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        hidden_dim: usize,
        d_inner: usize,
        d_state: usize,
        n_group: usize,
        dt_rank: usize,
        conv_kernel_size: usize,
        norm_eps: f32,
    ) -> Self {
        let conv_channels = d_inner + 2 * n_group * d_state;
        let head_v_dim = d_inner / dt_rank;
        const BUDGET_F32: usize = (31 * 1024) / 4;
        let hv = head_v_dim as f64;
        let chunk_size =
            ((-hv + (hv * hv + 8.0 * BUDGET_F32 as f64).sqrt()) / 4.0).floor() as usize;
        let chunk_size = std::env::var("RNB_METAL_DELTA_SCAN_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value >= 1)
            .unwrap_or_else(|| chunk_size.clamp(1, 64));
        Self {
            full: PrefillGdnFullCarrier::new(
                ctx,
                seq_len,
                conv_channels,
                conv_kernel_size,
                n_group,
                dt_rank,
                d_state,
                head_v_dim,
                chunk_size,
                hidden_dim,
                norm_eps,
            ),
            norm_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            norm_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            hidden_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            norm_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            dt_rank_buf: u32_buf(ctx, dt_rank as u32),
            zero_buf: u32_buf(ctx, 0),
        }
    }
}

fn copy_host_f32(src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            dst.contents().as_ptr().cast::<f32>(),
            src.len(),
        );
    }
}

fn gdn_weight_shape(weight: crate::GdnBackendWeightRef<'_>) -> (usize, usize) {
    match weight {
        crate::GdnBackendWeightRef::F32 { rows, cols, .. } => (rows, cols),
        crate::GdnBackendWeightRef::Quant(view) => (view.rows, view.cols),
    }
}

fn gdn_quant_weight_supported(
    ctx: &MetalContext,
    weight: crate::GdnBackendWeightRef<'_>,
) -> Result<bool, String> {
    match weight {
        crate::GdnBackendWeightRef::F32 { values, rows, cols } => {
            if values.len()
                != rows
                    .checked_mul(cols)
                    .ok_or("GDN F32 weight size overflow")?
            {
                return Err("GDN F32 weight length mismatch".to_string());
            }
            Ok(true)
        }
        crate::GdnBackendWeightRef::Quant(view) => {
            let blocks_per_row = view
                .cols
                .checked_div(256)
                .filter(|_| view.cols % 256 == 0)
                .ok_or_else(|| "GDN quant weight cols must be divisible by 256".to_string())?;
            let block_bytes = match view.quant {
                crate::TensoropsQuant::Q4K => 144,
                crate::TensoropsQuant::Q5K => 176,
                crate::TensoropsQuant::Q6K => 210,
            };
            let expected = view
                .rows
                .checked_mul(blocks_per_row)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or_else(|| "GDN quant weight size overflow".to_string())?;
            if view.raw.len() != expected {
                return Err("GDN quant weight length mismatch".to_string());
            }
            Ok(ctx.tensorops_capable
                && ctx.cast_f32_f16_pipeline.is_some()
                && match view.quant {
                    crate::TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
                    crate::TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
                    crate::TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
                })
        }
    }
}

/// Validates one GDN layer before the owner allocates or submits a command
/// buffer. `Ok(false)` is a capability fallback; malformed inputs are errors.
pub(crate) fn qwen_gdn_prefill_preflight(
    ctx: &MetalContext,
    spec: crate::QwenPrefillGdnBackendSpecRef<'_>,
) -> Result<bool, String> {
    if spec.seq_len == 0
        || spec.hidden_dim == 0
        || spec.d_inner == 0
        || spec.d_state == 0
        || spec.n_group == 0
        || spec.dt_rank == 0
        || spec.conv_kernel_size < 2
        || spec.d_state > 256
        || !spec.norm_eps.is_finite()
        || spec.norm_eps <= 0.0
    {
        return Err("invalid Qwen GDN prefill shape".to_string());
    }
    let checked_mul = |left: usize, right: usize, name: &str| {
        left.checked_mul(right)
            .ok_or_else(|| format!("Qwen GDN {name} overflow"))
    };
    let qk_channels = checked_mul(spec.n_group, spec.d_state, "qk channels")?;
    let conv_channels = checked_mul(qk_channels, 2, "conv qk channels")?
        .checked_add(spec.d_inner)
        .ok_or_else(|| "Qwen GDN conv channels overflow".to_string())?;
    let conv_state_len = checked_mul(
        spec.conv_kernel_size
            .checked_sub(1)
            .ok_or_else(|| "Qwen GDN conv state rows underflow".to_string())?,
        conv_channels,
        "conv state length",
    )?;
    let conv_kernel_len = checked_mul(spec.conv_kernel_size, conv_channels, "conv kernel length")?;
    let delta_state_len = checked_mul(spec.d_inner, spec.d_state, "delta state length")?;
    let hidden_elements = checked_mul(spec.seq_len, spec.hidden_dim, "hidden elements")?;
    let conv_rows = spec
        .seq_len
        .checked_add(spec.conv_kernel_size - 1)
        .ok_or_else(|| "Qwen GDN conv rows overflow".to_string())?;
    let conv_input_elements = checked_mul(conv_rows, conv_channels, "conv input elements")?;
    let conv_data_elements = checked_mul(spec.seq_len, conv_channels, "conv data elements")?;
    let qk_elements = checked_mul(spec.seq_len, qk_channels, "qk elements")?;
    let inner_elements = checked_mul(spec.seq_len, spec.d_inner, "inner elements")?;
    let final_qk_width = checked_mul(spec.dt_rank, spec.d_state, "final qk width")?;
    let final_qk_elements = checked_mul(spec.seq_len, final_qk_width, "final qk elements")?;
    let head_v_dim = spec.d_inner / spec.dt_rank;
    let su_workspace = checked_mul(spec.d_inner, 48, "step5 su workspace")?;
    let kh_workspace = checked_mul(
        checked_mul(spec.dt_rank, 48, "step5 kh heads")?,
        spec.d_state,
        "step5 kh workspace",
    )?;
    let step5_workspace = checked_mul(
        checked_mul(spec.dt_rank, head_v_dim, "step5 heads")?,
        spec.d_state,
        "step5 workspace",
    )?;
    let step4_a_workspace = checked_mul(
        checked_mul(spec.dt_rank, 48, "step4 a heads")?,
        176,
        "step4 a workspace",
    )?;
    let step4_b_workspace = checked_mul(
        checked_mul(spec.dt_rank, 176, "step4 b heads")?,
        head_v_dim,
        "step4 b workspace",
    )?;
    let step4_workspace = checked_mul(spec.d_inner, 48, "step4 workspace")?;
    for (elements, name) in [
        (hidden_elements, "hidden elements"),
        (delta_state_len, "delta state elements"),
        (conv_data_elements, "conv data elements"),
        (conv_input_elements, "conv input elements"),
        (final_qk_elements, "final qk elements"),
        (qk_elements, "qk elements"),
        (inner_elements, "inner elements"),
        (conv_state_len, "conv state elements"),
        (conv_kernel_len, "conv kernel elements"),
        (su_workspace, "step5 su workspace"),
        (kh_workspace, "step5 kh workspace"),
        (step5_workspace, "step5 workspace"),
        (step4_a_workspace, "step4 a workspace"),
        (step4_b_workspace, "step4 b workspace"),
        (step4_workspace, "step4 workspace"),
    ] {
        u32::try_from(elements).map_err(|_| format!("Qwen GDN {name} exceed u32"))?;
    }
    for elements in [
        hidden_elements,
        conv_state_len,
        conv_kernel_len,
        conv_input_elements,
        conv_data_elements,
        qk_elements,
        inner_elements,
        final_qk_elements,
        delta_state_len,
        step5_workspace,
        step4_workspace,
    ] {
        checked_mul(elements, std::mem::size_of::<f32>(), "buffer bytes")?;
    }
    for elements in [
        hidden_elements,
        su_workspace,
        kh_workspace,
        step4_a_workspace,
        step4_b_workspace,
    ] {
        checked_mul(elements, std::mem::size_of::<u16>(), "buffer bytes")?;
    }
    if spec.seq_len > u32::MAX as usize
        || spec.hidden_dim > u32::MAX as usize
        || conv_channels > u32::MAX as usize
        || spec.conv_kernel_size > u32::MAX as usize
        || spec.d_inner > u32::MAX as usize
        || spec.d_state > u32::MAX as usize
        || spec.n_group > u32::MAX as usize
        || spec.dt_rank > u32::MAX as usize
        || spec.d_inner % spec.dt_rank != 0
        || spec.dt_rank % spec.n_group != 0
        || spec.attn_norm_w.len() != spec.hidden_dim
        || spec.post_attn_norm_w.len() != spec.hidden_dim
        || spec.conv_state.len() != conv_state_len
        || spec.conv_kernel.len() != conv_kernel_len
        || spec.dt_bias.len() != spec.dt_rank
        || spec.ssm_a.len() != spec.dt_rank
        || spec.delta_state.len() != delta_state_len
        || spec.ssm_norm.len() != spec.d_inner / spec.dt_rank
        || gdn_weight_shape(spec.qkv_weight) != (conv_channels, spec.hidden_dim)
        || gdn_weight_shape(spec.gate_weight) != (spec.d_inner, spec.hidden_dim)
        || gdn_weight_shape(spec.alpha_weight) != (spec.dt_rank, spec.hidden_dim)
        || gdn_weight_shape(spec.beta_weight) != (spec.dt_rank, spec.hidden_dim)
        || gdn_weight_shape(spec.ssm_out_weight) != (spec.hidden_dim, spec.d_inner)
    {
        return Err("invalid Qwen GDN prefill shape".to_string());
    }
    const BUDGET_F32: usize = (31 * 1024) / 4;
    let hv = head_v_dim as f64;
    let default_chunk = ((-hv + (hv * hv + 8.0 * BUDGET_F32 as f64).sqrt()) / 4.0).floor() as usize;
    let chunk_size = std::env::var("RNB_METAL_DELTA_SCAN_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 1)
        .unwrap_or_else(|| default_chunk.clamp(1, 64));
    let chunk_square = checked_mul(chunk_size, chunk_size, "chunk square")?;
    let chunk_f32 = checked_mul(chunk_size, head_v_dim, "chunk rows")?
        .checked_add(checked_mul(2, chunk_square, "chunk matrices")?)
        .and_then(|elements| elements.checked_add(256))
        .ok_or_else(|| "Qwen GDN chunk workspace overflow".to_string())?;
    let chunk_bytes = checked_mul(
        chunk_f32,
        std::mem::size_of::<f32>(),
        "chunk workspace bytes",
    )?;
    if chunk_bytes > 32 * 1024 {
        return Ok(false);
    }
    for weight in [
        spec.qkv_weight,
        spec.gate_weight,
        spec.alpha_weight,
        spec.beta_weight,
        spec.ssm_out_weight,
    ] {
        if !gdn_quant_weight_supported(ctx, weight)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn encode_qwen_gdn_prefill_rms_norm_f32_exact(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenGdnPrefillCarrier,
    input: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
) {
    enc.setComputePipelineState(&ctx.qwen_gdn_prefill_rms_norm_f32_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&carrier.hidden_dim_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&carrier.full.inner.eps_buf), 0, 4);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: carrier.full.inner.seq_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    compute::chain_barrier(ctx, enc);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_gdn_prefill_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenGdnPrefillCarrier,
    spec: crate::QwenPrefillGdnBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    state_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
) -> Result<(), String> {
    encode_qwen_gdn_prefill_ops_with_stage_snapshots(
        ctx, enc, carrier, spec, hidden_in, hidden_out, state_out, None,
    )
}

fn encode_gdn_stage_snapshot(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenGdnPrefillCarrier,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    dt_rank: usize,
) {
    crate::compute::encode_split_conv_qkv(
        ctx,
        enc,
        input,
        0,
        output,
        output,
        output,
        &carrier.full.inner.seq_buf,
        &carrier.dt_rank_buf,
        &carrier.dt_rank_buf,
        &carrier.zero_buf,
        &carrier.zero_buf,
        seq_len,
        dt_rank,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_gdn_prefill_ops_with_stage_snapshots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenGdnPrefillCarrier,
    spec: crate::QwenPrefillGdnBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    state_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    stage_snapshots: Option<QwenGdnPrefillStageSnapshots<'_>>,
) -> Result<(), String> {
    // Appends one complete Qwen prefill GDN core to `enc`. No command-buffer
    // lifecycle operation and no readback is performed here.
    if !qwen_gdn_prefill_preflight(ctx, spec)? {
        return Err("Qwen GDN prefill capability unavailable".to_string());
    }
    let conv_channels = spec.d_inner + 2 * spec.n_group * spec.d_state;
    let inner = &carrier.full.inner;
    if inner.seq_len != spec.seq_len
        || carrier.full.n_out != spec.hidden_dim
        || carrier.full.d_inner != spec.d_inner
        || inner.conv_channels != conv_channels
        || inner.conv_kernel != spec.conv_kernel_size
        || inner.num_k_heads != spec.n_group
        || inner.num_v_heads != spec.dt_rank
        || inner.head_k_dim != spec.d_state
    {
        return Err("Qwen GDN prefill carrier shape mismatch".to_string());
    }

    let attn_norm_w_dev = empty_f32_buf(ctx, spec.hidden_dim);
    let dt_bias_dev = empty_f32_buf(ctx, spec.dt_rank);
    let ssm_a_dev = empty_f32_buf(ctx, spec.dt_rank);
    let conv_weight_dev = empty_f32_buf(ctx, spec.conv_kernel.len());
    let ssm_norm_dev = empty_f32_buf(ctx, spec.ssm_norm.len());
    copy_host_f32(spec.attn_norm_w, &attn_norm_w_dev);
    copy_host_f32(spec.dt_bias, &dt_bias_dev);
    copy_host_f32(spec.ssm_a, &ssm_a_dev);
    copy_host_f32(spec.conv_kernel, &conv_weight_dev);
    copy_host_f32(spec.ssm_norm, &ssm_norm_dev);
    copy_host_f32(spec.conv_state, state_out.0);
    copy_host_f32(spec.delta_state, state_out.1);
    crate::gdn_conv_delta_chain::encode_qwen_gdn_conv_state_in(
        ctx,
        enc,
        &carrier.full,
        state_out.0,
    );

    if let Some(snapshots) = stage_snapshots.as_ref() {
        crate::compute::encode_split_conv_qkv(
            ctx,
            enc,
            hidden_in,
            0,
            snapshots.hidden_input,
            snapshots.hidden_input,
            snapshots.hidden_input,
            &carrier.full.inner.seq_buf,
            &carrier.hidden_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.zero_buf,
            &carrier.zero_buf,
            spec.seq_len,
            spec.hidden_dim,
        );
        compute::chain_barrier(ctx, enc);
    }
    encode_qwen_gdn_prefill_rms_norm_f32_exact(
        ctx,
        enc,
        carrier,
        hidden_in,
        &attn_norm_w_dev,
        &carrier.norm_dev,
    );
    if let Some(snapshots) = stage_snapshots.as_ref() {
        crate::compute::encode_split_conv_qkv(
            ctx,
            enc,
            &carrier.norm_dev,
            0,
            snapshots.norm,
            snapshots.norm,
            snapshots.norm,
            &carrier.full.inner.seq_buf,
            &carrier.hidden_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.zero_buf,
            &carrier.zero_buf,
            spec.seq_len,
            spec.hidden_dim,
        );
        compute::chain_barrier(ctx, enc);
    }
    let quant_input = matches!(spec.qkv_weight, crate::GdnBackendWeightRef::Quant(_))
        || matches!(spec.gate_weight, crate::GdnBackendWeightRef::Quant(_))
        || matches!(spec.alpha_weight, crate::GdnBackendWeightRef::Quant(_))
        || matches!(spec.beta_weight, crate::GdnBackendWeightRef::Quant(_));
    if quant_input {
        compute::encode_cast_f32_to_f16(
            ctx,
            enc,
            &carrier.norm_dev,
            &carrier.norm_f16_dev,
            &carrier.norm_elems_buf,
            spec.seq_len * spec.hidden_dim,
        );
    }
    if quant_input {
        compute::chain_barrier(ctx, enc);
    }

    crate::gdn_proj_chain::encode_qwen_gdn_projection(
        ctx,
        enc,
        spec.qkv_weight,
        &carrier.norm_dev,
        &carrier.norm_f16_dev,
        &inner.conv_input_dev,
        spec.conv_state.len() * std::mem::size_of::<f32>(),
        &inner.cc_buf,
        &carrier.hidden_dim_buf,
        &inner.seq_buf,
        conv_channels,
        spec.hidden_dim,
        spec.seq_len,
    )?;
    crate::gdn_proj_chain::encode_qwen_gdn_projection(
        ctx,
        enc,
        spec.gate_weight,
        &carrier.norm_dev,
        &carrier.norm_f16_dev,
        &carrier.full.z_dev,
        0,
        &carrier.full.k_buf,
        &carrier.hidden_dim_buf,
        &inner.seq_buf,
        spec.d_inner,
        spec.hidden_dim,
        spec.seq_len,
    )?;
    crate::gdn_proj_chain::encode_qwen_gdn_projection(
        ctx,
        enc,
        spec.alpha_weight,
        &carrier.norm_dev,
        &carrier.norm_f16_dev,
        &inner.gate_dev,
        0,
        &carrier.dt_rank_buf,
        &carrier.hidden_dim_buf,
        &inner.seq_buf,
        spec.dt_rank,
        spec.hidden_dim,
        spec.seq_len,
    )?;
    crate::gdn_proj_chain::encode_qwen_gdn_projection(
        ctx,
        enc,
        spec.beta_weight,
        &carrier.norm_dev,
        &carrier.norm_f16_dev,
        &inner.beta_dev,
        0,
        &carrier.dt_rank_buf,
        &carrier.hidden_dim_buf,
        &inner.seq_buf,
        spec.dt_rank,
        spec.hidden_dim,
        spec.seq_len,
    )?;
    compute::chain_barrier(ctx, enc);
    if let Some(snapshots) = stage_snapshots.as_ref() {
        encode_gdn_stage_snapshot(
            ctx,
            enc,
            carrier,
            &inner.gate_dev,
            snapshots.alpha_raw,
            spec.seq_len,
            spec.dt_rank,
        );
        encode_gdn_stage_snapshot(
            ctx,
            enc,
            carrier,
            &inner.beta_dev,
            snapshots.beta_raw,
            spec.seq_len,
            spec.dt_rank,
        );
        compute::chain_barrier(ctx, enc);
    }

    for token in 0..spec.seq_len {
        enc.setComputePipelineState(&ctx.gdn_alpha_beta_pipeline);
        unsafe {
            let offset = token * spec.dt_rank * std::mem::size_of::<f32>();
            enc.setBuffer_offset_atIndex(Some(&inner.gate_dev), offset, 0);
            enc.setBuffer_offset_atIndex(Some(&inner.beta_dev), offset, 1);
            enc.setBuffer_offset_atIndex(Some(&dt_bias_dev), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&ssm_a_dev), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&carrier.dt_rank_buf), 0, 4);
        }
        let width = ctx.gdn_alpha_beta_pipeline.threadExecutionWidth().max(1);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: spec.dt_rank.div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    compute::chain_barrier(ctx, enc);
    if let Some(snapshots) = stage_snapshots.as_ref() {
        encode_gdn_stage_snapshot(
            ctx,
            enc,
            carrier,
            &inner.gate_dev,
            snapshots.alpha_processed,
            spec.seq_len,
            spec.dt_rank,
        );
        encode_gdn_stage_snapshot(
            ctx,
            enc,
            carrier,
            &inner.beta_dev,
            snapshots.beta_processed,
            spec.seq_len,
            spec.dt_rank,
        );
        compute::chain_barrier(ctx, enc);
    }

    crate::gdn_conv_delta_chain::encode_qwen_gdn_full_stage(
        ctx,
        enc,
        &carrier.full,
        &conv_weight_dev,
        &ssm_norm_dev,
        state_out.1,
        spec.ssm_out_weight,
    )?;
    crate::gdn_conv_delta_chain::encode_qwen_gdn_conv_state_out(
        ctx,
        enc,
        &carrier.full,
        state_out.0,
    );

    compute::encode_split_conv_qkv(
        ctx,
        enc,
        hidden_in,
        0,
        hidden_out,
        &inner.k_dev,
        &inner.v_dev,
        &inner.seq_buf,
        &carrier.hidden_dim_buf,
        &carrier.hidden_dim_buf,
        &carrier.zero_buf,
        &carrier.zero_buf,
        spec.seq_len,
        spec.hidden_dim,
    );
    compute::chain_barrier(ctx, enc);
    encode_residual_add(
        ctx,
        enc,
        hidden_out,
        &carrier.full.proj_dev,
        &carrier.hidden_elems_buf,
        spec.seq_len * spec.hidden_dim,
    );
    Ok(())
}

/// GDN layer 한 token 실행. weight buffer/offset(NoCopy resident)은 caller 준비.
/// 모든 GEMV weight 는 Q4_K, ffn_down 만 Q6_K 가능(ffn_down_is_q6k). conv1d/dt_bias/
/// ssm_a/ssm_norm/attn_norm/ffn_norm 은 f32 shared buffer.
/// 반환: `(hidden, conv_state_new, delta_state_new)` — caller 가 host ssm_state 에 반영.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &mut GdnCarrier,
    hidden: &[f32],
    conv_state: &[f32],
    delta_state: &[f32],
    attn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_w_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_off_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_w_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_off_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_bias_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_w_buf: &ProtocolObject<dyn MTLBuffer>,
    conv1d_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_q: u8,
    gate_q: u8,
    alpha_q: u8,
    beta_q: u8,
    ssm_out_q: u8,
    ffn_gate_q: u8,
    ffn_up_q: u8,
    ffn_down_q: u8,
    delta_resident: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden_dim = carrier.hidden_dim;
    let conv_channels = carrier.conv_channels;
    let conv_state_len = carrier.conv_state_len;

    debug_assert_eq!(hidden.len(), hidden_dim);
    debug_assert_eq!(conv_state.len(), conv_state_len);
    debug_assert_eq!(delta_state.len(), carrier.delta_state_len);

    // host → device sync (dispatch 경계). delta resident 모드는 첫 토큰만 seed 하고
    // 이후 토큰은 device 누적 state를 유지한다.
    GdnCarrier::upload(&carrier.hidden_dev, hidden);
    if delta_resident {
        carrier.upload_states_resident(conv_state, delta_state);
    } else {
        GdnCarrier::upload(&carrier.conv_input_dev, conv_state); // conv_input[0..conv_state_len]
        GdnCarrier::upload(&carrier.delta_state_dev, delta_state);
    }

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = chain_compute_encoder(ctx, &cmd);

    gdn_chain_encode(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        attn_norm_w_buf,
        qkv_w_buf,
        qkv_off_buf,
        gate_w_buf,
        gate_off_buf,
        alpha_w_buf,
        alpha_off_buf,
        beta_w_buf,
        beta_off_buf,
        dt_bias_w_buf,
        ssm_a_w_buf,
        conv1d_w_buf,
        ssm_norm_w_buf,
        ssm_out_w_buf,
        ssm_out_off_buf,
        ffn_norm_w_buf,
        ffn_gate_w_buf,
        ffn_gate_off_buf,
        ffn_up_w_buf,
        ffn_up_off_buf,
        ffn_down_w_buf,
        ffn_down_off_buf,
        qkv_q,
        gate_q,
        alpha_q,
        beta_q,
        ssm_out_q,
        ffn_gate_q,
        ffn_up_q,
        ffn_down_q,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // device → host sync. conv_state_new = conv_input[conv_channels..conv_kernel*conv_channels].
    let hidden_out = readback(&carrier.hidden_dev, hidden_dim);
    let conv_state_new =
        GdnCarrier::readback_at(&carrier.conv_input_dev, conv_channels, conv_state_len);
    let delta_state_new = if delta_resident {
        Vec::new()
    } else {
        GdnCarrier::readback_at(&carrier.delta_state_dev, 0, carrier.delta_state_len)
    };
    (hidden_out, conv_state_new, delta_state_new)
}

/// GDN core만 한 token 실행한다. MoE FFN layer에서 FFN routing/experts는 기존 경로를
/// 유지하고, `attn_norm..ssm_out+residual` 제출만 하나의 command buffer로 묶기 위한 경로.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_core_chain_dispatch(
    ctx: &MetalContext,
    carrier: &mut GdnCarrier,
    hidden: &[f32],
    conv_state: &[f32],
    delta_state: &[f32],
    attn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_w_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_off_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_w_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_off_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_bias_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_w_buf: &ProtocolObject<dyn MTLBuffer>,
    conv1d_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_q: u8,
    gate_q: u8,
    alpha_q: u8,
    beta_q: u8,
    ssm_out_q: u8,
    delta_resident: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden_dim = carrier.hidden_dim;
    let conv_channels = carrier.conv_channels;
    let conv_state_len = carrier.conv_state_len;

    debug_assert_eq!(hidden.len(), hidden_dim);
    debug_assert_eq!(conv_state.len(), conv_state_len);
    debug_assert_eq!(delta_state.len(), carrier.delta_state_len);

    GdnCarrier::upload(&carrier.hidden_dev, hidden);
    if delta_resident {
        carrier.upload_states_resident(conv_state, delta_state);
    } else {
        GdnCarrier::upload(&carrier.conv_input_dev, conv_state);
        GdnCarrier::upload(&carrier.delta_state_dev, delta_state);
    }

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = chain_compute_encoder(ctx, &cmd);

    gdn_core_chain_encode(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        attn_norm_w_buf,
        qkv_w_buf,
        qkv_off_buf,
        gate_w_buf,
        gate_off_buf,
        alpha_w_buf,
        alpha_off_buf,
        beta_w_buf,
        beta_off_buf,
        dt_bias_w_buf,
        ssm_a_w_buf,
        conv1d_w_buf,
        ssm_norm_w_buf,
        ssm_out_w_buf,
        ssm_out_off_buf,
        qkv_q,
        gate_q,
        alpha_q,
        beta_q,
        ssm_out_q,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let hidden_out = readback(&carrier.hidden_dev, hidden_dim);
    let conv_state_new =
        GdnCarrier::readback_at(&carrier.conv_input_dev, conv_channels, conv_state_len);
    let delta_state_new = if delta_resident {
        Vec::new()
    } else {
        GdnCarrier::readback_at(&carrier.delta_state_dev, 0, carrier.delta_state_len)
    };
    (hidden_out, conv_state_new, delta_state_new)
}

/// GDN core 뒤 Qwen35MoE FFN tail까지 같은 command buffer에 encode한다.
/// 기존 MoE 경로의 core->host->CPU fanout 경계를 제거하기 위한 decode path다.
fn qwen_moe_route_from_logits_cpu(
    logits: &[f32],
    n_used: usize,
    shared_gate: f32,
    shared_expert_id: u32,
) -> (Vec<u32>, Vec<f32>) {
    let selected_len = n_used.min(logits.len());
    let mut idx = (0..logits.len()).collect::<Vec<_>>();
    if selected_len < logits.len() {
        idx.select_nth_unstable_by(selected_len, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let selected = &idx[..selected_len];
    let selected_max = selected
        .iter()
        .map(|&expert| logits[expert])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut route_weights = Vec::with_capacity(selected_len + 1);
    for &expert in selected {
        route_weights.push((logits[expert] - selected_max).exp());
    }
    let selected_sum = route_weights.iter().sum::<f32>();
    if selected_sum != 0.0 {
        for weight in route_weights.iter_mut() {
            *weight /= selected_sum;
        }
    }

    let mut expert_ids = selected
        .iter()
        .map(|&expert| expert as u32)
        .collect::<Vec<_>>();
    expert_ids.push(shared_expert_id);
    route_weights.push(shared_gate);
    (expert_ids, route_weights)
}

fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_moe_chain_dispatch(
    ctx: &MetalContext,
    carrier: &mut GdnCarrier,
    moe_carrier: &crate::ffn_chain::QwenMoeDecodeChainCarrier,
    hidden: &[f32],
    conv_state: &[f32],
    delta_state: &[f32],
    attn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_w_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_off_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_w_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_off_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_bias_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_w_buf: &ProtocolObject<dyn MTLBuffer>,
    conv1d_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_off_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_input_scale_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off: u32,
    down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_off: u32,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off: u32,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off: u32,
    down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_off: u32,
    gate_sparse_raw: &[u8],
    up_sparse_raw: &[u8],
    down_sparse_raw: &[u8],
    gate_shared_raw: &[u8],
    up_shared_raw: &[u8],
    down_shared_raw: &[u8],
    qkv_q: u8,
    gate_q: u8,
    alpha_q: u8,
    beta_q: u8,
    ssm_out_q: u8,
    down_quant: u8,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden_dim = carrier.hidden_dim;
    let conv_channels = carrier.conv_channels;
    let conv_state_len = carrier.conv_state_len;

    debug_assert_eq!(hidden.len(), hidden_dim);
    debug_assert_eq!(conv_state.len(), conv_state_len);
    debug_assert_eq!(delta_state.len(), carrier.delta_state_len);

    GdnCarrier::upload(&carrier.hidden_dev, hidden);
    GdnCarrier::upload(&carrier.conv_input_dev, conv_state);
    GdnCarrier::upload(&carrier.delta_state_dev, delta_state);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = chain_compute_encoder(ctx, &cmd);

    gdn_core_chain_encode(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        attn_norm_w_buf,
        qkv_w_buf,
        qkv_off_buf,
        gate_w_buf,
        gate_off_buf,
        alpha_w_buf,
        alpha_off_buf,
        beta_w_buf,
        beta_off_buf,
        dt_bias_w_buf,
        ssm_a_w_buf,
        conv1d_w_buf,
        ssm_norm_w_buf,
        ssm_out_w_buf,
        ssm_out_off_buf,
        qkv_q,
        gate_q,
        alpha_q,
        beta_q,
        ssm_out_q,
    );

    if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_CPU_ROUTE_CHAIN").is_some() {
        crate::ffn_chain::qwen_moe_decode_chain_encode_router_logits(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            ffn_norm_w_buf,
            router_w_buf,
            router_off_buf,
            gate_sparse_off,
            up_sparse_off,
            down_sparse_off,
            gate_shared_off,
            up_shared_off,
            down_shared_off,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let router_logits = moe_carrier.read_router_logits();
        let normed = moe_carrier.debug_normed();
        let shared_input_scale = readback(shared_input_scale_buf, hidden_dim);
        let shared_dot = normed
            .iter()
            .zip(shared_input_scale.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let shared_gate = sigmoid_f32(shared_dot);
        let (expert_ids, route_weights) = qwen_moe_route_from_logits_cpu(
            &router_logits,
            moe_carrier.n_used(),
            shared_gate,
            moe_carrier.n_expert() as u32,
        );
        let wrap_selected = |sparse_raw: &[u8],
                             shared_raw: &[u8],
                             per_expert: usize|
         -> (Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, Vec<u32>) {
            let mut weights = Vec::with_capacity(expert_ids.len());
            let mut offsets = Vec::with_capacity(expert_ids.len());
            for &expert in &expert_ids {
                let raw = if expert == moe_carrier.n_expert() as u32 {
                    shared_raw
                } else {
                    let start = expert as usize * per_expert;
                    &sparse_raw[start..start + per_expert]
                };
                let (weight, offset) = moe_carrier.wrap_selected_slice(ctx, raw);
                weights.push(weight);
                offsets.push(offset);
            }
            (weights, offsets)
        };
        let (gate_selected_w, gate_selected_off) =
            wrap_selected(gate_sparse_raw, gate_shared_raw, moe_carrier.per_gate());
        let (up_selected_w, up_selected_off) =
            wrap_selected(up_sparse_raw, up_shared_raw, moe_carrier.per_up());
        let (down_selected_w, down_selected_off) =
            wrap_selected(down_sparse_raw, down_shared_raw, moe_carrier.per_down());

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = chain_compute_encoder(ctx, &cmd);
        let force_full_arena =
            std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_CPU_ROUTE_FULL_ARENA").is_some();
        let force_bound_offsets =
            std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_CPU_ROUTE_BOUND_OFFSETS").is_some();
        let selected_buffers = !force_full_arena
            && !force_bound_offsets
            && crate::ffn_chain::qwen_moe_decode_chain_encode_after_route_selected_buffers(
                ctx,
                &enc,
                moe_carrier,
                &carrier.hidden_dev,
                &gate_selected_w,
                &gate_selected_off,
                &up_selected_w,
                &up_selected_off,
                &down_selected_w,
                &down_selected_off,
                &expert_ids,
                &route_weights,
                down_quant,
            );
        if !selected_buffers && force_bound_offsets {
            crate::ffn_chain::qwen_moe_decode_chain_encode_after_route_bound_offsets(
                ctx,
                &enc,
                moe_carrier,
                &carrier.hidden_dev,
                gate_sparse_w,
                gate_sparse_off,
                up_sparse_w,
                up_sparse_off,
                down_sparse_w,
                down_sparse_off,
                gate_shared_w,
                gate_shared_off,
                up_shared_w,
                up_shared_off,
                down_shared_w,
                down_shared_off,
                &expert_ids,
                &route_weights,
                down_quant,
            );
        } else if !selected_buffers {
            moe_carrier.upload_routes(&expert_ids, &route_weights);
            crate::ffn_chain::qwen_moe_decode_chain_encode_after_route(
                ctx,
                &enc,
                moe_carrier,
                &carrier.hidden_dev,
                gate_sparse_w,
                up_sparse_w,
                down_sparse_w,
                gate_shared_w,
                up_shared_w,
                down_shared_w,
                down_quant,
            );
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    } else if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_ARG_TABLE").is_some() {
        crate::ffn_chain::qwen_moe_decode_chain_encode_route(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            ffn_norm_w_buf,
            router_w_buf,
            router_off_buf,
            shared_input_scale_buf,
            gate_sparse_w,
            gate_sparse_off,
            up_sparse_w,
            up_sparse_off,
            down_sparse_w,
            down_sparse_off,
            gate_shared_w,
            gate_shared_off,
            up_shared_w,
            up_shared_off,
            down_shared_w,
            down_shared_off,
            down_quant,
        );
        let argument_table =
            crate::ffn_chain::qwen_moe_decode_chain_encode_after_route_argument_table(
                ctx,
                &enc,
                moe_carrier,
                &carrier.hidden_dev,
                gate_sparse_raw,
                up_sparse_raw,
                down_sparse_raw,
                gate_shared_raw,
                up_shared_raw,
                down_shared_raw,
                down_quant,
            );
        assert!(
            argument_table,
            "Qwen MoE argument-table path requires <=256 sparse experts and Q5_K/Q6_K down"
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    } else if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_SPLIT_ROUTE_SELECTED").is_some() {
        crate::ffn_chain::qwen_moe_decode_chain_encode_route(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            ffn_norm_w_buf,
            router_w_buf,
            router_off_buf,
            shared_input_scale_buf,
            gate_sparse_w,
            gate_sparse_off,
            up_sparse_w,
            up_sparse_off,
            down_sparse_w,
            down_sparse_off,
            gate_shared_w,
            gate_shared_off,
            up_shared_w,
            up_shared_off,
            down_shared_w,
            down_shared_off,
            down_quant,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let (expert_ids, route_weights) = moe_carrier.debug_routes();
        let wrap_selected = |sparse_raw: &[u8],
                             shared_raw: &[u8],
                             per_expert: usize|
         -> (Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, Vec<u32>) {
            let mut weights = Vec::with_capacity(expert_ids.len());
            let mut offsets = Vec::with_capacity(expert_ids.len());
            for &expert in &expert_ids {
                let raw = if expert == moe_carrier.n_expert() as u32 {
                    shared_raw
                } else {
                    let start = expert as usize * per_expert;
                    &sparse_raw[start..start + per_expert]
                };
                let (weight, offset) = moe_carrier.wrap_selected_slice(ctx, raw);
                weights.push(weight);
                offsets.push(offset);
            }
            (weights, offsets)
        };
        let (gate_selected_w, gate_selected_off) =
            wrap_selected(gate_sparse_raw, gate_shared_raw, moe_carrier.per_gate());
        let (up_selected_w, up_selected_off) =
            wrap_selected(up_sparse_raw, up_shared_raw, moe_carrier.per_up());
        let (down_selected_w, down_selected_off) =
            wrap_selected(down_sparse_raw, down_shared_raw, moe_carrier.per_down());

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = chain_compute_encoder(ctx, &cmd);
        let selected_buffers =
            crate::ffn_chain::qwen_moe_decode_chain_encode_after_route_selected_buffers(
                ctx,
                &enc,
                moe_carrier,
                &carrier.hidden_dev,
                &gate_selected_w,
                &gate_selected_off,
                &up_selected_w,
                &up_selected_off,
                &down_selected_w,
                &down_selected_off,
                &expert_ids,
                &route_weights,
                down_quant,
            );
        assert!(
            selected_buffers,
            "Qwen MoE split-route selected buffers require <=9 slots and Q5_K/Q6_K down"
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    } else if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_SPLIT_ROUTE").is_some() {
        crate::ffn_chain::qwen_moe_decode_chain_encode_route(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            ffn_norm_w_buf,
            router_w_buf,
            router_off_buf,
            shared_input_scale_buf,
            gate_sparse_w,
            gate_sparse_off,
            up_sparse_w,
            up_sparse_off,
            down_sparse_w,
            down_sparse_off,
            gate_shared_w,
            gate_shared_off,
            up_shared_w,
            up_shared_off,
            down_shared_w,
            down_shared_off,
            down_quant,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = chain_compute_encoder(ctx, &cmd);
        crate::ffn_chain::qwen_moe_decode_chain_encode_after_route(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            gate_sparse_w,
            up_sparse_w,
            down_sparse_w,
            gate_shared_w,
            up_shared_w,
            down_shared_w,
            down_quant,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    } else {
        crate::ffn_chain::qwen_moe_decode_chain_encode(
            ctx,
            &enc,
            moe_carrier,
            &carrier.hidden_dev,
            ffn_norm_w_buf,
            router_w_buf,
            router_off_buf,
            shared_input_scale_buf,
            gate_sparse_w,
            gate_sparse_off,
            up_sparse_w,
            up_sparse_off,
            down_sparse_w,
            down_sparse_off,
            gate_shared_w,
            gate_shared_off,
            up_shared_w,
            up_shared_off,
            down_shared_w,
            down_shared_off,
            down_quant,
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let hidden_out = readback(&carrier.hidden_dev, hidden_dim);
    let conv_state_new =
        GdnCarrier::readback_at(&carrier.conv_input_dev, conv_channels, conv_state_len);
    let delta_state_new =
        GdnCarrier::readback_at(&carrier.delta_state_dev, 0, carrier.delta_state_len);
    (hidden_out, conv_state_new, delta_state_new)
}

/// GDN core(`attn_norm → qkv/gate/alpha/beta → conv/delta → ssm_out → residual`)만
/// encode한다. Dense FFN tail은 `gdn_chain_encode`가 이어서 담당한다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_core_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &GdnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    attn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_w_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_off_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_w_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_off_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_bias_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_w_buf: &ProtocolObject<dyn MTLBuffer>,
    conv1d_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_q: u8,
    gate_q: u8,
    alpha_q: u8,
    beta_q: u8,
    ssm_out_q: u8,
) {
    let hidden_dim = carrier.hidden_dim;
    let conv_channels = carrier.conv_channels;
    let conv_state_len = carrier.conv_state_len;
    let z_dim = carrier.z_dim;
    let num_v_heads = carrier.num_v_heads;
    let num_k_heads = carrier.num_k_heads;
    let head_v_dim = carrier.head_v_dim;
    let q_dim = carrier.q_dim;
    let k_dim = carrier.k_dim;

    let p = ctx.chain_profile;

    if p.emit_small() {
        encode_rms_norm(
            ctx,
            enc,
            hidden_dev,
            attn_norm_w_buf,
            &carrier.norm_dev,
            &carrier.hdim_buf,
            &carrier.eps_buf,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_gemv() {
        let coalesced_dispatch = if ctx.gemv_coalesced {
            gemv_coalesced_dispatch_for(ctx, qkv_q)
        } else {
            None
        };
        let simd_pipe = if coalesced_dispatch.is_none() && ctx.gemv_simd {
            gemv_simd_pipeline_for(ctx, qkv_q)
        } else {
            None
        };
        let pipeline = coalesced_dispatch
            .as_ref()
            .map(|dispatch| dispatch.pipeline)
            .or(simd_pipe)
            .unwrap_or_else(|| gemv_pipeline_for(ctx, qkv_q));
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(qkv_w_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&carrier.norm_dev), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&carrier.conv_input_dev), conv_state_len * 4, 2);
            enc.setBuffer_offset_atIndex(Some(&carrier.conv_ch_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&carrier.k_hidden_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(qkv_off_buf), 0, 5);
        }
        let (grid, tg) = if let Some(dispatch) = coalesced_dispatch {
            (
                MTLSize {
                    width: conv_channels.div_ceil(dispatch.rows_per_threadgroup),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: dispatch.threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                },
            )
        } else if simd_pipe.is_some() {
            (
                MTLSize {
                    width: conv_channels,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            let tg_width = pipeline.threadExecutionWidth().max(1);
            (
                MTLSize {
                    width: conv_channels.div_ceil(tg_width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            )
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            gate_q,
            gate_w_buf,
            &carrier.norm_dev,
            &carrier.z_dev,
            &carrier.zdim_buf,
            &carrier.k_hidden_buf,
            gate_off_buf,
            z_dim,
        );
        encode_gemv_quant(
            ctx,
            enc,
            alpha_q,
            alpha_w_buf,
            &carrier.norm_dev,
            &carrier.alpha_dev,
            &carrier.nv_buf,
            &carrier.k_hidden_buf,
            alpha_off_buf,
            num_v_heads,
        );
        encode_gemv_quant(
            ctx,
            enc,
            beta_q,
            beta_w_buf,
            &carrier.norm_dev,
            &carrier.beta_dev,
            &carrier.nv_buf,
            &carrier.k_hidden_buf,
            beta_off_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_gdn_alpha_beta(
            ctx,
            enc,
            &carrier.alpha_dev,
            &carrier.beta_dev,
            dt_bias_w_buf,
            ssm_a_w_buf,
            &carrier.nv_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_ssm_conv1d_silu(
            ctx,
            enc,
            &carrier.conv_input_dev,
            conv1d_w_buf,
            &carrier.conv_out_dev,
            &carrier.conv_ch_buf,
            &carrier.conv_k_buf,
            conv_channels,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_l2_norm(
            ctx,
            enc,
            &carrier.conv_out_dev,
            0,
            &carrier.q_norm_dev,
            0,
            &carrier.hk_buf,
            &carrier.eps_buf,
            &carrier.q_scale_buf,
            num_k_heads,
        );
        encode_l2_norm(
            ctx,
            enc,
            &carrier.conv_out_dev,
            q_dim * 4,
            &carrier.k_norm_dev,
            0,
            &carrier.hk_buf,
            &carrier.eps_buf,
            &carrier.one_buf,
            num_k_heads,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_delta_net_step(
            ctx,
            enc,
            &carrier.q_norm_dev,
            &carrier.k_norm_dev,
            &carrier.conv_out_dev,
            (q_dim + k_dim) * 4,
            &carrier.alpha_dev,
            &carrier.beta_dev,
            &carrier.delta_state_dev,
            &carrier.delta_out_dev,
            &carrier.hk_buf,
            &carrier.hv_buf,
            &carrier.nk_buf,
            num_v_heads,
            head_v_dim,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_qk_norm(
            ctx,
            enc,
            &carrier.delta_out_dev,
            ssm_norm_w_buf,
            &carrier.gated_dev,
            &carrier.hv_buf,
            &carrier.eps_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_silu_mul(
            ctx,
            enc,
            &carrier.z_dev,
            &carrier.gated_dev,
            &carrier.zdim_buf,
            z_dim,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            ssm_out_q,
            ssm_out_w_buf,
            &carrier.z_dev,
            &carrier.ssm_proj_dev,
            &carrier.hdim_buf,
            &carrier.zdim_buf,
            ssm_out_off_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc);

    if p.emit_small() {
        encode_residual_add(
            ctx,
            enc,
            hidden_dev,
            &carrier.ssm_proj_dev,
            &carrier.hdim_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc);
}

/// GDN layer 의 encode 14단계 + FFN 를 주어진 encoder 에 encode 만 한다(upload/commit/readback 없음).
/// `hidden_dev` 는 공유 가능한 hidden buffer(여러 layer 가 하나의 command buffer 로 묶일 때
/// 같은 buffer 를 넘긴다). 중간 buffer/state 는 carrier 소유 그대로 사용.
///
/// `ctx.chain_profile` 가 All 이 아니면 dispatch 를 class(gemv/small) 별로 격리 emit
/// 한다(pm21 REST 분해 측정용 차감법). GDN 은 attention compute 가 없어 attn class 에선
/// 아무 dispatch 도 emit 하지 않는다(attn 시간은 attention layer 만 기여).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &GdnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    attn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_w_buf: &ProtocolObject<dyn MTLBuffer>,
    alpha_off_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_w_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_off_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_bias_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_w_buf: &ProtocolObject<dyn MTLBuffer>,
    conv1d_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_out_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_q: u8,
    gate_q: u8,
    alpha_q: u8,
    beta_q: u8,
    ssm_out_q: u8,
    ffn_gate_q: u8,
    ffn_up_q: u8,
    ffn_down_q: u8,
) {
    let hidden_dim = carrier.hidden_dim;
    let conv_channels = carrier.conv_channels;
    let conv_state_len = carrier.conv_state_len;
    let z_dim = carrier.z_dim;
    let num_v_heads = carrier.num_v_heads;
    let num_k_heads = carrier.num_k_heads;
    let head_v_dim = carrier.head_v_dim;
    let q_dim = carrier.q_dim;
    let k_dim = carrier.k_dim;
    let ffn_dim = carrier.ffn_dim;

    let p = ctx.chain_profile;

    // ① attn_norm: hidden_dev → norm_dev [small]
    if p.emit_small() {
        encode_rms_norm(
            ctx,
            enc,
            hidden_dev,
            attn_norm_w_buf,
            &carrier.norm_dev,
            &carrier.hdim_buf,
            &carrier.eps_buf,
        );
    }
    chain_barrier(ctx, enc); // ① norm 완료 → 그룹 A(qkv·gate·alpha·beta) 독립 GEMV 진입
                             // ② qkv GEMV: norm_dev → conv_input_dev[conv_state_len..] (inline, out offset) [gemv]
                             // simd(lane-saturation) 경로는 1-SIMD-group/row(grid=N, tg=32), baseline 은 1-thread/row
                             // (grid=N.div_ceil(tg_width)) — buffer 레이아웃(0 w,1 in,2 out,3 N,4 K,5 off)은 공통.
    if p.emit_gemv() {
        // coalesced(nr0=2) > simd(1 SIMD-group/row) > baseline 우선순위. inline qkv 도
        // encode_gemv_q4k_auto 경로와 동일하게 RNB_METAL_GEMV_COALESCED 를 따른다(이전엔
        // 이 inline 경로만 토글에서 누락돼 uncoalesced 로 남아있었음).
        let coalesced_dispatch = if ctx.gemv_coalesced {
            gemv_coalesced_dispatch_for(ctx, qkv_q)
        } else {
            None
        };
        let simd_pipe = if coalesced_dispatch.is_none() && ctx.gemv_simd {
            gemv_simd_pipeline_for(ctx, qkv_q)
        } else {
            None
        };
        let pipeline = coalesced_dispatch
            .as_ref()
            .map(|dispatch| dispatch.pipeline)
            .or(simd_pipe)
            .unwrap_or_else(|| gemv_pipeline_for(ctx, qkv_q));
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(qkv_w_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&carrier.norm_dev), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&carrier.conv_input_dev), conv_state_len * 4, 2);
            enc.setBuffer_offset_atIndex(Some(&carrier.conv_ch_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&carrier.k_hidden_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(qkv_off_buf), 0, 5);
        }
        // coalesced 면 dispatch layout 이 row/tg 를 결정한다. simd 는 1 SIMD-group/row(grid=N).
        let (grid, tg) = if let Some(dispatch) = coalesced_dispatch {
            (
                MTLSize {
                    width: conv_channels.div_ceil(dispatch.rows_per_threadgroup),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: dispatch.threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                },
            )
        } else if simd_pipe.is_some() {
            (
                MTLSize {
                    width: conv_channels,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            let tg_width = pipeline.threadExecutionWidth().max(1);
            (
                MTLSize {
                    width: conv_channels.div_ceil(tg_width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            )
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    // ③ gate GEMV: norm_dev → z_dev (N=z_dim, K=hidden_dim) [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            gate_q,
            gate_w_buf,
            &carrier.norm_dev,
            &carrier.z_dev,
            &carrier.zdim_buf,
            &carrier.k_hidden_buf,
            gate_off_buf,
            z_dim,
        );
    }
    // ④ alpha GEMV: norm_dev → alpha_dev (N=num_v_heads) [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            alpha_q,
            alpha_w_buf,
            &carrier.norm_dev,
            &carrier.alpha_dev,
            &carrier.nv_buf,
            &carrier.k_hidden_buf,
            alpha_off_buf,
            num_v_heads,
        );
    }
    // ⑤ beta GEMV: norm_dev → beta_dev (N=num_v_heads) [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            beta_q,
            beta_w_buf,
            &carrier.norm_dev,
            &carrier.beta_dev,
            &carrier.nv_buf,
            &carrier.k_hidden_buf,
            beta_off_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc); // 그룹 A(qkv·gate·alpha·beta) 완료
                             // ⑥ alpha/beta 전처리: beta=sigmoid, alpha=softplus(alpha+dt_bias)*ssm_a (in-place) [small]
    if p.emit_small() {
        encode_gdn_alpha_beta(
            ctx,
            enc,
            &carrier.alpha_dev,
            &carrier.beta_dev,
            dt_bias_w_buf,
            ssm_a_w_buf,
            &carrier.nv_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑦ conv1d + silu: conv_input_dev → conv_out_dev [small]
    if p.emit_small() {
        encode_ssm_conv1d_silu(
            ctx,
            enc,
            &carrier.conv_input_dev,
            conv1d_w_buf,
            &carrier.conv_out_dev,
            &carrier.conv_ch_buf,
            &carrier.conv_k_buf,
            conv_channels,
        );
    }
    chain_barrier(ctx, enc); // ⑦ conv1d 완료 → 그룹 {⑧ q·⑨ k l2_norm} 독립 진입 (conv_out read-only)
                             // ⑧ q l2_norm + scale: conv_out_dev[0..q_dim] → q_norm_dev (num_k_heads rows) [small]
    if p.emit_small() {
        encode_l2_norm(
            ctx,
            enc,
            &carrier.conv_out_dev,
            0,
            &carrier.q_norm_dev,
            0,
            &carrier.hk_buf,
            &carrier.eps_buf,
            &carrier.q_scale_buf,
            num_k_heads,
        );
    }
    // ⑨ k l2_norm: conv_out_dev[q_dim..q_dim+k_dim] → k_norm_dev — ⑧ 과 독립 그룹(barrier 없음 — pm28 2차) [small]
    if p.emit_small() {
        encode_l2_norm(
            ctx,
            enc,
            &carrier.conv_out_dev,
            q_dim * 4,
            &carrier.k_norm_dev,
            0,
            &carrier.hk_buf,
            &carrier.eps_buf,
            &carrier.one_buf,
            num_k_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑩ delta_net(GQA): q_norm/k_norm/v(conv_out offset q+k) + alpha/beta + state → delta_out [small]
    if p.emit_small() {
        encode_delta_net_step(
            ctx,
            enc,
            &carrier.q_norm_dev,
            &carrier.k_norm_dev,
            &carrier.conv_out_dev,
            (q_dim + k_dim) * 4,
            &carrier.alpha_dev,
            &carrier.beta_dev,
            &carrier.delta_state_dev,
            &carrier.delta_out_dev,
            &carrier.hk_buf,
            &carrier.hv_buf,
            &carrier.nk_buf,
            num_v_heads,
            head_v_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑪ ssm_norm: per-head RMSNorm delta_out_dev → gated_dev (weight[head_v_dim] head 공유) [small]
    if p.emit_small() {
        encode_qk_norm(
            ctx,
            enc,
            &carrier.delta_out_dev,
            ssm_norm_w_buf,
            &carrier.gated_dev,
            &carrier.hv_buf,
            &carrier.eps_buf,
            num_v_heads,
        );
    }
    chain_barrier(ctx, enc);
    // ⑫ z silu·mul: z_dev = silu(z_dev) * gated_dev [small]
    if p.emit_small() {
        encode_silu_mul(
            ctx,
            enc,
            &carrier.z_dev,
            &carrier.gated_dev,
            &carrier.zdim_buf,
            z_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑬ ssm_out GEMV: z_dev → ssm_proj_dev (N=hidden_dim, K=z_dim) [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            ssm_out_q,
            ssm_out_w_buf,
            &carrier.z_dev,
            &carrier.ssm_proj_dev,
            &carrier.hdim_buf,
            &carrier.zdim_buf,
            ssm_out_off_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ⑭ residual: hidden_dev += ssm_proj_dev [small]
    if p.emit_small() {
        encode_residual_add(
            ctx,
            enc,
            hidden_dev,
            &carrier.ssm_proj_dev,
            &carrier.hdim_buf,
            hidden_dim,
        );
    }

    // FFN (carrier 통합, 같은 command buffer): ffn_norm→gate/up→silu→down→residual
    chain_barrier(ctx, enc); // ⑭ residual 완료 → FFN
                             // ffn_norm [small]
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
                             // ffn gate GEMV [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            ffn_gate_q,
            ffn_gate_w_buf,
            &carrier.ffn_normed_dev,
            &carrier.ffn_gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            ffn_gate_off_buf,
            ffn_dim,
        );
    }
    // ffn up GEMV [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            ffn_up_q,
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
                             // ffn silu_mul [small]
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
    // ffn down GEMV [gemv]
    if p.emit_gemv() {
        encode_gemv_quant(
            ctx,
            enc,
            ffn_down_q,
            ffn_down_w_buf,
            &carrier.ffn_gate_dev,
            &carrier.ffn_down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            ffn_down_off_buf,
            hidden_dim,
        );
    }
    chain_barrier(ctx, enc);
    // ffn residual [small]
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

/// 공유 hidden buffer 에 host slice 를 1회 올린다(chain 진입 시 specs[0] 입력).
/// `GdnCarrier::upload` 와 동일한 StorageModeShared contents 직접 쓰기.
/// 1.3/1.4 의 `gdn_decode_chain_run`(lib.rs) 이 호출.
#[allow(dead_code)] // 1.3/1.4 에서 연결.
pub(crate) fn upload_shared(buf: &ProtocolObject<dyn MTLBuffer>, src: &[f32]) {
    let contents = buf.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), contents.as_ptr() as *mut f32, src.len());
    }
}

/// 연속된 GDN carrier layer 하나의 per-layer 인자를 borrow 로 묶은 spec.
/// `gdn_layer_resident` 가 호출당 받던 인자(weight raw `&[u8]` 8개 + f32 weight `&[f32]`
/// 6개 + quant code 8개 + shape 10개 + 입력 state)를 그대로 담는다. weight 는
/// `ModelWeights` 의 mmap slice 를 borrow 하므로 lifetime `<'a>`.
///
/// shape(hidden_dim..eps)는 9B 의 24 GDN layer 가 동일하지만, caller 가 layer 별로
/// self-contained spec 을 만들 수 있게 각 spec 이 자기 shape 를 들고 있는다(carrier
/// entry 생성에 그대로 사용). `rnb-backend-metal` 에 정의해 상위(runtime/rnb-llm)가
/// 쓴다 — crate 의존 방향이 rnb-llm→rnb-runtime→rnb-backend-metal 이라 backend 가
/// runtime 타입을 import 할 수 없어, spec 타입은 backend 가 소유한다.
///
/// 1.3 seam / 1.4 decode loop 에서 `gdn_decode_chain_run` 으로 연결된다.
pub struct GdnChainSpecRef<'a> {
    pub layer: usize,
    // 입력 state(이 layer 의 직전 conv·delta state). chain 진입 시 carrier 로 upload.
    pub conv_state: &'a [f32],
    pub delta_state: &'a [f32],
    // f32 weight (작아서 복사 업로드).
    pub attn_norm_weight: &'a [f32],
    pub dt_bias_weight: &'a [f32],
    pub ssm_a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub ssm_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    // 양자화 GEMV weight raw bytes (NoCopy resident wrap 대상).
    pub qkv_raw: &'a [u8],
    pub gate_raw: &'a [u8],
    pub alpha_raw: &'a [u8],
    pub beta_raw: &'a [u8],
    pub ssm_out_raw: &'a [u8],
    pub ffn_gate_raw: &'a [u8],
    pub ffn_up_raw: &'a [u8],
    pub ffn_down_raw: &'a [u8],
    // quant code (Q4_K / Q6_K 구분).
    pub qkv_q: u8,
    pub gate_q: u8,
    pub alpha_q: u8,
    pub beta_q: u8,
    pub ssm_out_q: u8,
    pub ffn_gate_q: u8,
    pub ffn_up_q: u8,
    pub ffn_down_q: u8,
    // shape (layer 불변이지만 self-contained: carrier entry 생성에 사용).
    pub hidden_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub z_dim: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn empty_qwen_gdn_spec(
        seq_len: usize,
        hidden_dim: usize,
        d_inner: usize,
        d_state: usize,
        n_group: usize,
        dt_rank: usize,
        conv_kernel_size: usize,
    ) -> crate::QwenPrefillGdnBackendSpecRef<'static> {
        let conv_channels = d_inner + 2 * n_group * d_state;
        let qkv_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: conv_channels,
            cols: hidden_dim,
        };
        let gate_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: d_inner,
            cols: hidden_dim,
        };
        let alpha_beta_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: dt_rank,
            cols: hidden_dim,
        };
        let ssm_out_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: hidden_dim,
            cols: d_inner,
        };
        crate::QwenPrefillGdnBackendSpecRef {
            seq_len,
            hidden_dim,
            d_inner,
            d_state,
            n_group,
            dt_rank,
            conv_kernel_size,
            attn_norm_w: &[],
            qkv_weight,
            gate_weight,
            alpha_weight: alpha_beta_weight,
            beta_weight: alpha_beta_weight,
            conv_state: &[],
            conv_kernel: &[],
            dt_bias: &[],
            ssm_a: &[],
            delta_state: &[],
            ssm_norm: &[],
            ssm_out_weight,
            post_attn_norm_w: &[],
            norm_eps: 1e-6,
        }
    }

    /// pm31 GDN delta state device residency: carrier 가 delta 를 **첫 토큰만 seed**
    /// 하고 이후 upload 는 skip 한다(delta_state_dev 가 토큰 간 device 잔류 →
    /// delta_net_step in-place 누적, CUDA `resident_delta_state` 패턴과 동형).
    /// conv 는 매 토큰 upload 유지(delta 가 readback bytes 96.7%, conv 3.3% 는 그대로).
    #[test]
    fn gdn_carrier_seeds_delta_once_then_skips_upload() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return; // GPU 없는 CI 환경 skip
        };
        // 작은 self-contained shape: conv_state_len=3*16=48, delta_state_len=2*8*8=128.
        let mut carrier = GdnCarrier::new(&ctx, 64, 16, 4, 32, 2, 2, 8, 8, 64, 1e-6);
        assert!(!carrier.delta_seeded, "new carrier: delta not seeded");

        let conv = vec![0.5_f32; carrier.conv_state_len];
        let delta_a = vec![1.0_f32; carrier.delta_state_len];
        carrier.upload_states_resident(&conv, &delta_a);
        assert!(carrier.delta_seeded, "after first upload: seeded");
        assert_eq!(
            carrier.readback_delta_state(),
            delta_a,
            "first upload seeds delta to a"
        );

        // 두 번째 upload: delta 는 이미 seeded → skip(device 가 a 잔류).
        let delta_b = vec![2.0_f32; carrier.delta_state_len];
        carrier.upload_states_resident(&conv, &delta_b);
        assert_eq!(
            carrier.readback_delta_state(),
            delta_a,
            "second upload: delta_b skipped, device still holds a"
        );
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_preflight_rejects_overflow() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        let empty_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: 0,
            cols: 0,
        };
        let spec = crate::QwenPrefillGdnBackendSpecRef {
            seq_len: 1,
            hidden_dim: 2,
            d_inner: 2,
            d_state: 1,
            n_group: 1,
            dt_rank: 1,
            conv_kernel_size: (1usize << 62) + 1,
            attn_norm_w: &[],
            qkv_weight: empty_weight,
            gate_weight: empty_weight,
            alpha_weight: empty_weight,
            beta_weight: empty_weight,
            conv_state: &[],
            conv_kernel: &[],
            dt_bias: &[],
            ssm_a: &[],
            delta_state: &[],
            ssm_norm: &[],
            ssm_out_weight: empty_weight,
            post_attn_norm_w: &[],
            norm_eps: 1e-6,
        };
        let error = qwen_gdn_prefill_preflight(&ctx, spec)
            .expect_err("overflowing GDN shape must fail before carrier allocation");
        assert!(
            error.contains("overflow"),
            "overflow preflight returned unexpected error: {error}"
        );
    }
    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_preflight_rejects_u32_conv_data_overflow() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        const SEQ_LEN: usize = 65_536;
        const HIDDEN_DIM: usize = 256;
        const D_INNER: usize = 32_768;
        const D_STATE: usize = 256;
        const N_GROUP: usize = 64;
        const DT_RANK: usize = 256;
        const CONV_KERNEL: usize = 2;
        const CONV_CHANNELS: usize = D_INNER + 2 * N_GROUP * D_STATE;
        assert_eq!(
            SEQ_LEN.checked_mul(CONV_CHANNELS),
            Some(u32::MAX as usize + 1)
        );
        let qkv_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: CONV_CHANNELS,
            cols: HIDDEN_DIM,
        };
        let gate_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: D_INNER,
            cols: HIDDEN_DIM,
        };
        let alpha_beta_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: DT_RANK,
            cols: HIDDEN_DIM,
        };
        let ssm_out_weight = crate::GdnBackendWeightRef::F32 {
            values: &[],
            rows: HIDDEN_DIM,
            cols: D_INNER,
        };
        let spec = crate::QwenPrefillGdnBackendSpecRef {
            seq_len: SEQ_LEN,
            hidden_dim: HIDDEN_DIM,
            d_inner: D_INNER,
            d_state: D_STATE,
            n_group: N_GROUP,
            dt_rank: DT_RANK,
            conv_kernel_size: CONV_KERNEL,
            attn_norm_w: &[],
            qkv_weight,
            gate_weight,
            alpha_weight: alpha_beta_weight,
            beta_weight: alpha_beta_weight,
            conv_state: &[],
            conv_kernel: &[],
            dt_bias: &[],
            ssm_a: &[],
            delta_state: &[],
            ssm_norm: &[],
            ssm_out_weight,
            post_attn_norm_w: &[],
            norm_eps: 1e-6,
        };
        let error = qwen_gdn_prefill_preflight(&ctx, spec)
            .expect_err("u32 conv-data overflow must reject before carrier allocation");
        assert!(
            error.contains("conv data elements exceed u32"),
            "unexpected preflight error: {error}"
        );
    }
    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_preflight_rejects_u32_delta_state_overflow() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        const D_INNER: usize = 16_777_216;
        const D_STATE: usize = 256;
        assert_eq!(D_INNER.checked_mul(D_STATE), Some(u32::MAX as usize + 1));
        let spec = empty_qwen_gdn_spec(1, 256, D_INNER, D_STATE, 1, 1, 2);
        let error = qwen_gdn_prefill_preflight(&ctx, spec)
            .expect_err("u32 delta-state overflow must reject before carrier allocation");
        assert!(
            error.contains("delta state elements exceed u32"),
            "unexpected preflight error: {error}"
        );
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_preflight_rejects_u32_conv_input_overflow() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        const SEQ_LEN: usize = 1;
        const CONV_KERNEL: usize = 65_536;
        const D_INNER: usize = 32_768;
        const D_STATE: usize = 256;
        const N_GROUP: usize = 64;
        const CONV_CHANNELS: usize = D_INNER + 2 * N_GROUP * D_STATE;
        let conv_rows = SEQ_LEN + CONV_KERNEL - 1;
        assert_eq!(
            conv_rows.checked_mul(CONV_CHANNELS),
            Some(u32::MAX as usize + 1)
        );
        let spec = empty_qwen_gdn_spec(SEQ_LEN, 256, D_INNER, D_STATE, N_GROUP, 256, CONV_KERNEL);
        let error = qwen_gdn_prefill_preflight(&ctx, spec)
            .expect_err("u32 conv-input overflow must reject before carrier allocation");
        assert!(
            error.contains("conv input elements exceed u32"),
            "unexpected preflight error: {error}"
        );
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_preflight_rejects_u32_final_qk_overflow() {
        let Some(ctx) = crate::compute::build_metal_context() else {
            return;
        };
        const SEQ_LEN: usize = 65_536;
        const DT_RANK: usize = 256;
        const D_STATE: usize = 256;
        assert_eq!(
            SEQ_LEN.checked_mul(DT_RANK * D_STATE),
            Some(u32::MAX as usize + 1)
        );
        let spec = empty_qwen_gdn_spec(SEQ_LEN, 256, 256, D_STATE, 1, DT_RANK, 2);
        let error = qwen_gdn_prefill_preflight(&ctx, spec)
            .expect_err("u32 final-qk overflow must reject before carrier allocation");
        assert!(
            error.contains("final qk elements exceed u32"),
            "unexpected preflight error: {error}"
        );
    }
}
