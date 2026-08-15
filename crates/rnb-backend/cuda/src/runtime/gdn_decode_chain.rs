//! Qwen3.6 GDN decode layer device-resident chain (cu203).
//!
//! target-only decode 의 GDN core (attn_norm → qkv/gate → alpha/beta → conv →
//! delta scan → gated norm → ssm_out → residual) 를 단일 stream 의 device 실행으로
//! 묶어 층당 수십 회의 host↔device 왕복을 hidden 1왕복으로 줄인다.
//!
//! conv/delta state 는 `resident_delta_state_ptr` registry (cu202 계약)를 그대로
//! 사용한다 — host 사본은 decode 동안 stale 이 되고, snapshot/checkpoint 는
//! `materialize_sequence_state` 가 conv/delta 둘 다 sync 하며, 새 시퀀스는
//! `clear_resident_delta_states` 가 registry 를 비운다.
//!
//! 커널 선택 계약: qkv/gate/ssm_out 은 기존 decode q8dot 커널
//! (`rnb_q{4,5,6}k_gemv_q8dot_warp8`), head_k_dim=128 delta 는 같은 수식의
//! 4-warp reduction을 쓰고 진단 대조에서만 기존 shared-memory reduction으로 돌아간다.

use super::*;

/// GDN decode chain weight/dims 요청. F32 weight 슬라이스는 모델 소유 tensor 라
/// 엔진 수명 동안 stable 하므로 `resident_f32_ptr_stable_source` 로 상주시킨다.
#[allow(clippy::too_many_arguments)]
pub(super) struct QwenGdnDecodeChainRequest<'a> {
    pub hidden: &'a mut [f32],
    pub conv_state: &'a mut [f32],
    pub delta_state: &'a mut [f32],
    pub attn_norm: &'a [f32],
    pub qkv_weights: &'a [u8],
    pub qkv_quant: u32,
    pub gate_weights: &'a [u8],
    pub alpha_weights: &'a [f32],
    pub beta_weights: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub conv_kernel_weights: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_weights: &'a [u8],
    pub ssm_out_quant: u32,
    pub n_embd: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub d_inner: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub norm_eps: f32,
}

const WS_ALIGN: usize = 256;

fn align_up(offset: usize, align: usize) -> usize {
    offset.div_ceil(align) * align
}

impl CudaState {
    /// 층당 왕복이 hidden H2D 1회 + hidden D2H 1회가 되도록 GDN core 전체를
    /// device 에서 실행한다. 반환 시 `hidden` 에 ssm residual add 까지 반영된다.
    /// conv/delta host 사본은 갱신하지 않는다 (resident 계약).
    pub(super) fn qwen35_gdn_decode_core_chain(
        &mut self,
        req: QwenGdnDecodeChainRequest<'_>,
    ) -> Result<(), String> {
        let n_embd = req.n_embd;
        let ch = req.conv_channels;
        let d_inner = req.d_inner;
        let heads = req.num_v_heads;
        let f4 = std::mem::size_of::<f32>();

        // workspace 오프셋 배치 (전부 256B 정렬, 단일 할당 재사용).
        let mut cursor = 0usize;
        let mut slot = |bytes: usize| {
            let offset = cursor;
            cursor = align_up(cursor + bytes, WS_ALIGN);
            offset
        };
        let hidden_off = slot(n_embd * f4);
        let normed_off = slot(n_embd * f4);
        let qs_off = slot(n_embd);
        let ds_off = slot((n_embd / 32) * f4);
        let qkv_off = slot(ch * f4);
        let z_off = slot(d_inner * f4);
        let alpha_off = slot(heads * f4);
        let beta_off = slot(heads * f4);
        let gate_prep_off = slot(heads * f4);
        let beta_prep_off = slot(heads * f4);
        let conv_input_off = slot(req.conv_kernel * ch * f4);
        let conv_out_off = slot(ch * f4);
        let q_rep_off = slot(heads * req.head_k_dim * f4);
        let k_rep_off = slot(heads * req.head_k_dim * f4);
        let v_off = slot(heads * req.head_v_dim * f4);
        let delta_out_off = slot(d_inner * f4);
        let gated_off = slot(d_inner * f4);
        let qs2_off = slot(d_inner);
        let ds2_off = slot((d_inner / 32) * f4);
        let ssm_proj_off = slot(n_embd * f4);
        let total_bytes = cursor;

        let ws = ensure_device_buffer(
            &self.api,
            &mut self.qwen_gdn_decode_chain_workspace,
            &mut self.qwen_gdn_decode_chain_workspace_capacity,
            total_bytes,
        )?;
        let at = |off: usize| ws + off as u64;

        // resident weights (allocation-identity key — 모델 tensor 는 stable).
        let attn_norm_dev = self.resident_f32_ptr_stable_source(req.attn_norm)?;
        let alpha_w_dev = self.resident_f32_ptr_stable_source(req.alpha_weights)?;
        let beta_w_dev = self.resident_f32_ptr_stable_source(req.beta_weights)?;
        let dt_bias_dev = self.resident_f32_ptr_stable_source(req.dt_bias)?;
        let ssm_a_dev = self.resident_f32_ptr_stable_source(req.ssm_a)?;
        let conv_kernel_dev = self.resident_f32_ptr_stable_source(req.conv_kernel_weights)?;
        let ssm_norm_dev = self.resident_f32_ptr_stable_source(req.ssm_norm)?;

        // resident states (device 가 진실; host 는 materialize 때만 sync).
        let conv_state_dev = self.resident_delta_state_ptr(req.conv_state)?;
        let delta_state_dev = self.resident_delta_state_ptr(req.delta_state)?;

        // 1. hidden H2D
        unsafe {
            self.api.memcpy_htod_async(
                at(hidden_off),
                req.hidden.as_ptr().cast::<libc::c_void>(),
                n_embd * f4,
                self.stream,
            )?;
        }

        // 2. attn RMSNorm (plain, unit_offset=false)
        self.launch_rms_norm_f32(
            at(hidden_off),
            attn_norm_dev,
            at(normed_off),
            req.norm_eps,
            n_embd,
            false,
            false,
        )?;

        // 3. normed Q8_1 양자화 → qkv/gate 공유
        self.launch_quantize_q8_1_by_32(at(normed_off), at(qs_off), at(ds_off), n_embd)?;

        // 4. qkv projection (rows=conv_channels)
        match req.qkv_quant {
            12 => self.launch_q4k_gemv_q8dot_to_dev(
                req.qkv_weights,
                ch,
                n_embd / 256,
                at(qs_off),
                at(ds_off),
                at(qkv_off),
            )?,
            14 => self.launch_q6k_gemv_q8dot_to_dev(
                req.qkv_weights,
                ch,
                n_embd / 256,
                at(qs_off),
                at(ds_off),
                at(qkv_off),
            )?,
            other => return Err(format!("GDN chain unsupported qkv quant {other}")),
        }

        // 5. gate projection (rows=d_inner, Q4_K)
        self.launch_q4k_gemv_q8dot_to_dev(
            req.gate_weights,
            d_inner,
            n_embd / 256,
            at(qs_off),
            at(ds_off),
            at(z_off),
        )?;

        // 6. alpha/beta F32 GEMV (normed 입력)
        self.sgemm_device(alpha_w_dev, heads, n_embd, at(normed_off), 1, at(alpha_off))?;
        self.sgemm_device(beta_w_dev, heads, n_embd, at(normed_off), 1, at(beta_off))?;

        // 7. delta gate/beta 준비 (softplus·sigmoid)
        self.launch_gdn_prepare_delta_gate_beta_f32(
            at(gate_prep_off),
            at(beta_prep_off),
            at(alpha_off),
            at(beta_off),
            dt_bias_dev,
            ssm_a_dev,
            heads,
            heads,
        )?;

        // 8. conv input 조립 → conv1d+silu → conv state 갱신
        self.launch_gdn_build_conv_input_f32(
            at(conv_input_off),
            conv_state_dev,
            at(qkv_off),
            1,
            ch,
            req.conv_kernel,
        )?;
        self.launch_ssm_conv1d_silu_dev(
            at(conv_input_off),
            conv_kernel_dev,
            at(conv_out_off),
            1,
            ch,
            req.conv_kernel,
        )?;
        // new_state = conv_input[ch .. conv_kernel*ch] (shift+append 와 동일한 한 번의 DtoD).
        unsafe {
            self.api.memcpy_dtod_async(
                conv_state_dev,
                at(conv_input_off) + (ch * f4) as u64,
                (req.conv_kernel - 1) * ch * f4,
                self.stream,
            )?;
        }

        // 9. q/k l2norm+scale+GQA rep, v split (fused)
        self.launch_gdn_prepare_delta_qkv_f32(
            at(q_rep_off),
            at(k_rep_off),
            at(v_off),
            at(conv_out_off),
            1,
            ch,
            req.num_k_heads,
            heads,
            req.head_k_dim,
            req.head_v_dim,
            req.norm_eps,
            1.0 / (req.head_k_dim as f32).sqrt(),
        )?;

        // 10. delta-net decode (resident state 갱신 포함)
        let warp128 = tuning::gdn_delta_warp128_enabled();
        self.launch_delta_net_decode_dev(
            at(delta_out_off),
            delta_state_dev,
            at(q_rep_off),
            at(k_rep_off),
            at(v_off),
            at(gate_prep_off),
            at(beta_prep_off),
            warp128,
            heads,
            req.head_k_dim,
            req.head_v_dim,
        )?;

        // 11. head별 RMSNorm + silu(z) gate (fused)
        self.launch_gdn_gated_norm_silu_dev(
            at(gated_off),
            at(delta_out_off),
            at(z_off),
            ssm_norm_dev,
            heads,
            req.head_v_dim,
            req.norm_eps,
        )?;

        // 12. ssm_out projection (rows=n_embd)
        self.launch_quantize_q8_1_by_32(at(gated_off), at(qs2_off), at(ds2_off), d_inner)?;
        match req.ssm_out_quant {
            12 => self.launch_q4k_gemv_q8dot_to_dev(
                req.ssm_out_weights,
                n_embd,
                d_inner / 256,
                at(qs2_off),
                at(ds2_off),
                at(ssm_proj_off),
            )?,
            13 => self.launch_q5k_gemv_q8dot_to_dev(
                req.ssm_out_weights,
                n_embd,
                d_inner / 256,
                at(qs2_off),
                at(ds2_off),
                at(ssm_proj_off),
            )?,
            14 => self.launch_q6k_gemv_q8dot_to_dev(
                req.ssm_out_weights,
                n_embd,
                d_inner / 256,
                at(qs2_off),
                at(ds2_off),
                at(ssm_proj_off),
            )?,
            other => return Err(format!("GDN chain unsupported ssm_out quant {other}")),
        }

        // 13. residual add + hidden D2H
        self.launch_add_f32_inplace(at(hidden_off), at(ssm_proj_off), n_embd)?;
        unsafe {
            self.api.memcpy_dtoh_async(
                req.hidden.as_mut_ptr().cast::<libc::c_void>(),
                at(hidden_off),
                n_embd * f4,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }
}
