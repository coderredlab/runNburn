#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;
#[cfg(feature = "cuda")]
use rnb_loader::GGMLType;

#[cfg(feature = "cuda")]
fn cuda_error(err: String) -> crate::error::LlmError {
    crate::error::LlmError::Forward(err)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) struct GdnPrefillChainShape {
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub dt_rank: usize,
    pub conv_kernel: usize,
    pub conv_state_len: usize,
    pub delta_state_len: usize,
}

#[derive(Debug, PartialEq)]
pub(in crate::engine) struct GdnPrefillChainOutput {
    pub(in crate::engine) ssm_projection: Vec<f32>,
    pub(in crate::engine) ssm_projection_d2h_bytes: usize,
    #[cfg(feature = "cuda")]
    pub(in crate::engine) device_output:
        Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    #[cfg(feature = "cuda")]
    pub(in crate::engine) device_residual:
        Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    #[cfg(feature = "cuda")]
    pub(in crate::engine) device_moe_input:
        Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    pub(in crate::engine) conv_state_d2h_bytes: usize,
    pub(in crate::engine) delta_state_d2h_bytes: usize,
}

#[cfg(feature = "cuda")]
impl GdnPrefillChainOutput {
    pub(in crate::engine) fn release_device_output_if_present(
        &mut self,
    ) -> crate::error::Result<()> {
        let Some((output_id, _output_desc)) = self.device_output.take() else {
            return Ok(());
        };
        match cuda_runtime::release_device_tensor(output_id).map_err(cuda_error)? {
            true => Ok(()),
            false => Err(cuda_error(
                "CUDA GDN prefill chain device output was already missing".to_string(),
            )),
        }
    }

    pub(in crate::engine) fn release_device_carriers_if_present(
        &mut self,
    ) -> crate::error::Result<()> {
        for (label, slot) in [
            ("residual", &mut self.device_residual),
            ("moe_input", &mut self.device_moe_input),
        ] {
            let Some((output_id, _output_desc)) = slot.take() else {
                continue;
            };
            match cuda_runtime::release_device_tensor(output_id).map_err(cuda_error)? {
                true => {}
                false => {
                    return Err(cuda_error(format!(
                        "CUDA GDN prefill chain {label} carrier was already missing"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl GdnPrefillChainOutput {
    pub(in crate::engine) fn release_device_output_if_present(
        &mut self,
    ) -> crate::error::Result<()> {
        Ok(())
    }

    pub(in crate::engine) fn release_device_carriers_if_present(
        &mut self,
    ) -> crate::error::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl From<GdnPrefillChainShape> for crate::engine::cuda_runtime::GdnPrefillChainShape {
    fn from(shape: GdnPrefillChainShape) -> Self {
        Self {
            seq_len: shape.seq_len,
            hidden_dim: shape.hidden_dim,
            d_inner: shape.d_inner,
            d_state: shape.d_state,
            n_group: shape.n_group,
            dt_rank: shape.dt_rank,
            conv_kernel: shape.conv_kernel,
            conv_state_len: shape.conv_state_len,
            delta_state_len: shape.delta_state_len,
        }
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn ensure_gdn_prefill_chunk_supported(
    seq_len: usize,
    hidden_dim: usize,
) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::ensure_gdn_prefill_chunk_supported(seq_len, hidden_dim)
            .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(())
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn try_gdn_prefill_chain_if_supported(
    shape: &GdnPrefillChainShape,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        let cuda_shape = (*shape).into();
        return cuda_runtime::gdn_prefill_chain(&cuda_shape)
            .map(|output| output.is_some())
            .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[cfg(feature = "cuda")]
fn chain_k_quant_weight_raw<'a>(
    weight: &'a QuantizedWeight,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], GGMLType)> {
    if !matches!(
        weight.ggml_type,
        GGMLType::Q4_K | GGMLType::Q6_K | GGMLType::Q8_0
    ) {
        return Err(crate::error::LlmError::Forward(format!(
            "CUDA GDN prefill chain {label} must be Q4_K, Q6_K or Q8_0, got {:?}",
            weight.ggml_type
        )));
    }
    let bytes = weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!("CUDA GDN prefill chain {label} raw bytes missing"))
    })?;
    Ok((bytes, weight.ggml_type))
}

#[cfg(feature = "cuda")]
fn chain_q4k_weight_raw<'a>(
    weight: &'a QuantizedWeight,
    label: &'static str,
) -> crate::error::Result<&'a [u8]> {
    if weight.ggml_type != GGMLType::Q4_K {
        return Err(crate::error::LlmError::Forward(format!(
            "CUDA GDN prefill chain {label} must be Q4_K, got {:?}",
            weight.ggml_type
        )));
    }
    weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!("CUDA GDN prefill chain {label} raw bytes missing"))
    })
}

#[cfg(feature = "cuda")]
fn chain_q4k_or_f32_weight<'a>(
    weight: &'a QuantizedWeight,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], &'a [f32], GGMLType)> {
    match weight.ggml_type {
        GGMLType::Q4_K => Ok((chain_q4k_weight_raw(weight, label)?, &[], GGMLType::Q4_K)),
        GGMLType::F32 => {
            if weight.data.dtype() != rnb_core::tensor::DType::F32 {
                return Err(crate::error::LlmError::Forward(format!(
                    "CUDA GDN prefill chain {label} dtype {:?} does not match F32",
                    weight.data.dtype()
                )));
            }
            let values = crate::engine::kernels::tensor_as_f32_slice(&weight.data);
            let expected = weight.rows.checked_mul(weight.cols).ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "CUDA GDN prefill chain {label} shape overflow: rows={} cols={}",
                    weight.rows, weight.cols
                ))
            })?;
            if values.len() != expected {
                return Err(crate::error::LlmError::Forward(format!(
                    "CUDA GDN prefill chain {label} F32 len {} != rows*cols {expected}",
                    values.len()
                )));
            }
            Ok((&[], values, GGMLType::F32))
        }
        other => Err(crate::error::LlmError::Forward(format!(
            "CUDA GDN prefill chain {label} must be F32 or Q4_K, got {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn gdn_prefill_chain_q4k(
    shape: &GdnPrefillChainShape,
    hidden: &[f32],
    #[cfg(feature = "cuda")] hidden_device: Option<(
        cuda_runtime::DeviceTensorId,
        cuda_runtime::DeviceTensorDesc,
    )>,
    attn_norm: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    conv_state: &mut [f32],
    conv_kernel: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    delta_state: &mut [f32],
    ssm_norm: &[f32],
    ssm_out: &QuantizedWeight,
    post_attn_norm: &[f32],
    keep_host_output: bool,
    norm_eps: f32,
) -> crate::error::Result<Option<GdnPrefillChainOutput>> {
    #[cfg(feature = "cuda")]
    {
        if !try_gdn_prefill_chain_if_supported(shape)? {
            return Ok(None);
        }
        let (qkv_q4k, qkv_quant) = chain_k_quant_weight_raw(qkv_weight, "qkv")?;
        let (gate_q4k, gate_quant) = chain_k_quant_weight_raw(gate_weight, "gate")?;
        let (alpha_q4k, alpha_f32, alpha_quant) = chain_q4k_or_f32_weight(alpha_weight, "alpha")?;
        let (beta_q4k, beta_f32, beta_quant) = chain_q4k_or_f32_weight(beta_weight, "beta")?;
        let (ssm_out_q4k, ssm_out_quant) = chain_k_quant_weight_raw(ssm_out, "ssm_out")?;
        let output = cuda_runtime::gdn_prefill_chain_q4k(cuda_runtime::GdnPrefillChainQ4KRequest {
            shape: (*shape).into(),
            hidden,
            hidden_device,
            attn_norm,
            qkv_q4k,
            qkv_quant,
            gate_q4k,
            gate_quant,
            alpha_q4k,
            alpha_f32,
            alpha_quant,
            beta_q4k,
            beta_f32,
            beta_quant,
            conv_state,
            conv_kernel,
            dt_bias,
            ssm_a,
            delta_state,
            ssm_norm,
            ssm_out_q4k,
            ssm_out_quant,
            ssm_out_rows: ssm_out.rows,
            ssm_out_cols: ssm_out.cols,
            norm_eps,
            keep_host_output,
            post_attn_norm: Some(post_attn_norm),
        })
        .map_err(cuda_error)?;
        return Ok(output.map(|output| GdnPrefillChainOutput {
            ssm_projection: output.ssm_projection,
            ssm_projection_d2h_bytes: output.ssm_projection_d2h_bytes,
            device_output: output.device_output,
            device_residual: output.device_residual,
            device_moe_input: output.device_moe_input,
            conv_state_d2h_bytes: output.conv_state_d2h_bytes,
            delta_state_d2h_bytes: output.delta_state_d2h_bytes,
        }));
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(unused_variables))]
pub(in crate::engine) fn ssm_prefill_conv1d_silu(
    input: &[f32],
    kernel: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::ssm_prefill_conv1d_silu(
            input,
            kernel,
            seq_len,
            channels,
            kernel_size,
        )
        .map_err(cuda_error);
    }
    // pm43: Metal seam(gpu_gdn 공통 facade — cuda 전용이던 conv1d+silu 를 Metal 도). f32 exact.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(metal_runtime::metal_prefill_conv1d_silu_into_if_supported(
            input,
            kernel,
            seq_len,
            channels,
            kernel_size,
        ));
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn ssm_prefill_delta_net(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::ssm_prefill_delta_net_resident(
            state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn ssm_prefill_delta_net_snapshot(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    snapshot_after_tokens: usize,
) -> crate::error::Result<Option<(Vec<f32>, crate::engine::cuda_runtime::DeltaStateSnapshot)>> {
    let (output, snapshot) = cuda_runtime::ssm_prefill_delta_net_resident_snapshot(
        state,
        q,
        k,
        v,
        gate,
        beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        snapshot_after_tokens,
    )
    .map_err(cuda_error)?;
    Ok(snapshot.map(|snapshot| (output, snapshot)))
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn ssm_prefill_delta_net_snapshots(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    snapshot_after_tokens: &[usize],
) -> crate::error::Result<
    Option<(
        Vec<f32>,
        Vec<crate::engine::cuda_runtime::DeltaStateSnapshot>,
    )>,
> {
    let (output, snapshots) = cuda_runtime::ssm_prefill_delta_net_resident_snapshots(
        state,
        q,
        k,
        v,
        gate,
        beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        snapshot_after_tokens,
    )
    .map_err(cuda_error)?;
    if snapshots.is_empty() {
        Ok(None)
    } else {
        Ok(Some((output, snapshots)))
    }
}

#[cfg(any(not(feature = "cuda"), test))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn nemotron_mamba2_decode_scan(
    state: &mut [f32],
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    a: &[f32],
    d: &[f32],
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    n_group: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_mamba2_decode_scan(
            state, x, b, c, dt, a, d, num_heads, head_dim, state_dim, n_group,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[cfg(any(not(feature = "cuda"), test))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn nemotron_mamba2_prefill_scan(
    state: &mut [f32],
    conv_activated: &[f32],
    dt_data: &[f32],
    a: &[f32],
    d: &[f32],
    seq_len: usize,
    d_inner: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    n_group: usize,
    state_dim: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::nemotron_mamba2_prefill_scan(
            state,
            conv_activated,
            dt_data,
            a,
            d,
            seq_len,
            d_inner,
            conv_channels,
            bc_dim,
            num_heads,
            head_dim,
            n_group,
            state_dim,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(unused_variables))]
pub(in crate::engine) fn gdn_prefill_gated_norm_silu_project(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    weight: &QuantizedWeight,
    seq_len: usize,
    head_dim: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    let Some(view) = weight.backend_view() else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::gdn_prefill_gated_norm_silu_project(
            output,
            z,
            norm,
            backend_ggml_type(view.quant()),
            Some(view.raw()),
            seq_len,
            head_dim,
            view.rows(),
            view.cols(),
            norm_eps,
        )
        .map_err(cuda_error);
    }
    // pm44: Metal seam — fused gated_norm_silu_project 경로 연결. cols는 backend method 내부에서 유도.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(
            crate::engine::metal_runtime::metal_prefill_gated_norm_silu_project_into_if_supported(
                output,
                z,
                norm,
                backend_ggml_type(view.quant()),
                view.raw(),
                seq_len,
                head_dim,
                view.rows(),
                norm_eps,
            ),
        );
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Ok(None)
}

/// pm45 M2: GDN prefill conv→delta device-resident chain facade. conv1d_silu →
/// split_conv_qkv → l2_norm → repeat_qk → scale(q) → delta_net_scan 을 단일 GPU chain 으로.
/// Some((output, state_after)) 면 caller 가 그 자리에 합류, None 이면 기존 conv/delta 경로.
/// conv_input/conv_weight 는 raw(scale 미적용), gate/beta/state 는 delta scan 에 넘기던 그대로.
/// cuda 는 별도 full-layer chain(`gdn_prefill_chain_q4k`)이 conv→delta 를 이미 흡수하므로
/// 여기선 metal 전용(cuda 없을 때) — cuda 빌드는 Ok(None) 으로 기존 경로 유지.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn gdn_prefill_conv_delta_chain(
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<(Vec<f32>, Vec<f32>)>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(
            metal_runtime::metal_prefill_gdn_conv_delta_chain_into_if_supported(
                conv_input,
                conv_weight,
                gate,
                beta,
                state,
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                norm_eps,
            ),
        );
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

/// pm45 M3-1: GDN prefill full chain(conv→delta→gated→ssm_out) facade. M2(conv→delta) chain 끝
/// delta output 을 host 로 readback 하지 않고 같은 GPU command buffer 에 이어서 gated_rmsnorm_silu →
/// ssm_out proj 까지 device-resident 로 묶는다. Some((proj, state_after)) 면 caller 가 그 자리에서
/// proj_vec 까지 바로 얻고 별도 gated_norm_silu_project 호출 skip, None 이면 기존 conv→delta(또는
/// CPU) 경로 + 별도 gated proj 경로. M2 facade 입력 + M1 입력(z/ssm_norm/ssm_out weight) 추가.
/// ssm_out weight 가 backend_view 미제공 또는 tensorops 미지원 quant 면 None(분리 경로 fallback).
/// cuda 는 full-layer chain(`gdn_prefill_chain_q4k`)이 이미 흡수 → metal 전용(cuda 없을 때).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn gdn_prefill_full_chain(
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    ssm_out_weight: &QuantizedWeight,
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<(Vec<f32>, Vec<f32>)>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(view) = ssm_out_weight.backend_view() else {
            return Ok(None);
        };
        return Ok(
            crate::engine::metal_runtime::metal_prefill_gdn_full_chain_into_if_supported(
                conv_input,
                conv_weight,
                gate,
                beta,
                state,
                z,
                ssm_norm,
                backend_ggml_type(view.quant()),
                view.raw(),
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                view.rows(),
                norm_eps,
            ),
        );
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn gdn_prefill_full_ffn_chain(
    hidden: &[f32],
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    ssm_out_weight: &QuantizedWeight,
    post_norm_w: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    hidden_dim: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<(Vec<f32>, Vec<f32>)>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        if ssm_out_weight.rows != hidden_dim
            || ssm_out_weight.cols != num_v_heads * head_v_dim
            || post_norm_w.len() != hidden_dim
            || ffn_gate_weight.cols != hidden_dim
            || ffn_up_weight.cols != hidden_dim
            || ffn_gate_weight.rows != ffn_up_weight.rows
            || ffn_down_weight.rows != hidden_dim
            || ffn_down_weight.cols != ffn_gate_weight.rows
        {
            return Ok(None);
        }
        let Some(ssm_out_view) = ssm_out_weight.backend_view() else {
            return Ok(None);
        };
        let Some(gate_view) = ffn_gate_weight.backend_view() else {
            return Ok(None);
        };
        let Some(up_view) = ffn_up_weight.backend_view() else {
            return Ok(None);
        };
        let Some(down_view) = ffn_down_weight.backend_view() else {
            return Ok(None);
        };
        return Ok(
            crate::engine::metal_runtime::metal_prefill_gdn_full_ffn_chain_into_if_supported(
                hidden,
                conv_input,
                conv_weight,
                gate,
                beta,
                state,
                z,
                ssm_norm,
                backend_ggml_type(ssm_out_view.quant()),
                ssm_out_view.raw(),
                post_norm_w,
                backend_ggml_type(gate_view.quant()),
                gate_view.raw(),
                backend_ggml_type(up_view.quant()),
                up_view.raw(),
                backend_ggml_type(down_view.quant()),
                down_view.raw(),
                seq_len,
                conv_channels,
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                hidden_dim,
                ffn_gate_weight.rows,
                norm_eps,
            ),
        );
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(unused_variables))]
pub(in crate::engine) fn gdn_prefill_gated_norm_silu(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    seq_len: usize,
    rows: usize,
    cols: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::gdn_prefill_gated_norm_silu(
            output, z, norm, seq_len, rows, cols, norm_eps,
        )
        .map_err(cuda_error);
    }
    // pm43: Metal seam(gpu_gdn 공통 facade). rmsnorm(output)·silu(z) batch. f32 — 27B 의미 동등.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(
            metal_runtime::metal_prefill_gated_norm_silu_into_if_supported(
                output, z, norm, seq_len, rows, cols, norm_eps,
            ),
        );
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    static GDN_CHAIN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn qwen35_chain_shape(seq_len: usize) -> GdnPrefillChainShape {
        GdnPrefillChainShape {
            seq_len,
            hidden_dim: 2048,
            d_inner: 4096,
            d_state: 128,
            n_group: 16,
            dt_rank: 32,
            conv_kernel: 4,
            conv_state_len: 3 * (4096 + 2 * 16 * 128),
            delta_state_len: 4096 * 128,
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_prefill_chain_bridge_defaults_on_and_allows_opt_out() {
        let _guard = GDN_CHAIN_ENV_LOCK.lock().expect("GDN chain env lock");
        let shape = qwen35_chain_shape(32);
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
        }
        assert!(try_gdn_prefill_chain_if_supported(&shape).expect("default chain bridge"));

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "0");
        }
        assert!(!try_gdn_prefill_chain_if_supported(&shape).expect("explicit off chain bridge"));

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "1");
        }
        assert!(try_gdn_prefill_chain_if_supported(&shape).expect("enabled chain bridge"));

        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_prefill_chain_q4k_skips_weight_validation_when_disabled() {
        let _guard = GDN_CHAIN_ENV_LOCK.lock().expect("GDN chain env lock");
        let shape = qwen35_chain_shape(32);
        let invalid_weight = QuantizedWeight::new(
            rnb_core::tensor::Tensor::zeros(&[1], rnb_core::tensor::DType::U8),
            GGMLType::F16,
            1,
            1,
        );
        let mut conv_state = Vec::new();
        let mut delta_state = Vec::new();
        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "0");
        }

        let output = gdn_prefill_chain_q4k(
            &shape,
            &[],
            #[cfg(feature = "cuda")]
            None,
            &[],
            &invalid_weight,
            &invalid_weight,
            &invalid_weight,
            &invalid_weight,
            &mut conv_state,
            &[],
            &[],
            &[],
            &mut delta_state,
            &[],
            &invalid_weight,
            &[],
            true,
            1e-6,
        )
        .expect("disabled chain should not validate weight formats");

        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
        }

        assert!(output.is_none());
    }
}
