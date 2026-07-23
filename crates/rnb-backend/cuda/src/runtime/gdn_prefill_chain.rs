#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillChainShape {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPrefillChainPlan {
    Disabled,
    Q4KDeviceChain,
}

pub struct GdnPrefillChainQ4KRequest<'a> {
    pub shape: GdnPrefillChainShape,
    pub hidden: &'a [f32],
    pub hidden_device: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    pub attn_norm: &'a [f32],
    pub qkv_q4k: &'a [u8],
    pub qkv_quant: u32,
    pub gate_q4k: &'a [u8],
    pub gate_quant: u32,
    pub alpha_q4k: &'a [u8],
    pub alpha_f32: &'a [f32],
    pub alpha_quant: u32,
    pub beta_q4k: &'a [u8],
    pub beta_f32: &'a [f32],
    pub beta_quant: u32,
    pub conv_state: &'a mut [f32],
    pub conv_kernel: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub delta_state: &'a mut [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_q4k: &'a [u8],
    pub ssm_out_quant: u32,
    pub ssm_out_rows: usize,
    pub ssm_out_cols: usize,
    pub norm_eps: f32,
    pub keep_host_output: bool,
    pub keep_device_output: bool,
    pub post_attn_norm: Option<&'a [f32]>,
    pub keep_device_moe_input: bool,
}

#[derive(Debug, PartialEq)]
pub struct GdnPrefillChainQ4KOutput {
    pub ssm_projection: Vec<f32>,
    pub ssm_projection_d2h_bytes: usize,
    pub device_output: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    pub device_residual: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    pub device_moe_input: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    pub conv_state_d2h_bytes: usize,
    pub delta_state_d2h_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillChainDims {
    pub conv_channels: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub q_dim: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

pub fn validate_gdn_prefill_chain_shape(shape: &GdnPrefillChainShape) -> Result<(), String> {
    if shape.seq_len == 0 {
        return Err("GDN prefill chain requires seq_len > 0".to_string());
    }
    if shape.hidden_dim == 0 {
        return Err("GDN prefill chain requires hidden_dim > 0".to_string());
    }
    if shape.d_inner == 0 {
        return Err("GDN prefill chain requires d_inner > 0".to_string());
    }
    if shape.d_state == 0 {
        return Err("GDN prefill chain requires d_state > 0".to_string());
    }
    if shape.n_group == 0 {
        return Err("GDN prefill chain requires n_group > 0".to_string());
    }
    if shape.dt_rank == 0 {
        return Err("GDN prefill chain requires dt_rank > 0".to_string());
    }
    if shape.conv_kernel == 0 {
        return Err("GDN prefill chain requires conv_kernel > 0".to_string());
    }
    if shape.d_inner % shape.dt_rank != 0 {
        return Err(format!(
            "GDN prefill chain requires d_inner divisible by dt_rank: d_inner={} dt_rank={}",
            shape.d_inner, shape.dt_rank
        ));
    }

    let conv_channels = shape
        .n_group
        .checked_mul(shape.d_state)
        .and_then(|bc| bc.checked_mul(2))
        .and_then(|bc2| shape.d_inner.checked_add(bc2))
        .ok_or_else(|| "GDN prefill chain conv channel count overflow".to_string())?;
    let expected_conv_state_len = shape
        .conv_kernel
        .saturating_sub(1)
        .checked_mul(conv_channels)
        .ok_or_else(|| "GDN prefill chain conv_state_len overflow".to_string())?;
    if shape.conv_state_len != expected_conv_state_len {
        return Err(format!(
            "GDN prefill chain conv_state_len mismatch: got {} expected {}",
            shape.conv_state_len, expected_conv_state_len
        ));
    }

    let expected_delta_state_len = shape
        .d_inner
        .checked_mul(shape.d_state)
        .ok_or_else(|| "GDN prefill chain delta_state_len overflow".to_string())?;
    if shape.delta_state_len != expected_delta_state_len {
        return Err(format!(
            "GDN prefill chain delta_state_len mismatch: got {} expected {}",
            shape.delta_state_len, expected_delta_state_len
        ));
    }

    Ok(())
}

pub fn derive_gdn_prefill_chain_dims(
    shape: &GdnPrefillChainShape,
) -> Result<GdnPrefillChainDims, String> {
    validate_gdn_prefill_chain_shape(shape)?;
    let head_v_dim = shape.d_inner / shape.dt_rank;
    let q_dim = shape
        .n_group
        .checked_mul(shape.d_state)
        .ok_or_else(|| "GDN prefill chain q_dim overflow".to_string())?;
    let k_dim = q_dim;
    let v_dim = shape.d_inner;
    let conv_channels = shape
        .d_inner
        .checked_add(
            q_dim
                .checked_add(k_dim)
                .ok_or_else(|| "GDN prefill chain q/k dim overflow".to_string())?,
        )
        .ok_or_else(|| "GDN prefill chain conv channel count overflow".to_string())?;
    Ok(GdnPrefillChainDims {
        conv_channels,
        num_v_heads: shape.dt_rank,
        num_k_heads: shape.n_group,
        head_k_dim: shape.d_state,
        head_v_dim,
        q_dim,
        k_dim,
        v_dim,
    })
}

pub fn plan_gdn_prefill_chain_for_test(
    shape: &GdnPrefillChainShape,
    forced: bool,
) -> Result<GdnPrefillChainPlan, String> {
    validate_gdn_prefill_chain_shape(shape)?;
    if !forced {
        return Ok(GdnPrefillChainPlan::Disabled);
    }
    Ok(GdnPrefillChainPlan::Q4KDeviceChain)
}

pub fn plan_gdn_prefill_chain(shape: &GdnPrefillChainShape) -> Result<GdnPrefillChainPlan, String> {
    plan_gdn_prefill_chain_for_test(shape, crate::tuning::gdn_prefill_chain_enabled())
}

pub fn build_gdn_prefill_chain_conv_input_for_test(
    shape: &GdnPrefillChainShape,
    conv_state: &[f32],
    qkv_rows: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let dims = derive_gdn_prefill_chain_dims(shape)?;
    if conv_state.len() != shape.conv_state_len {
        return Err(format!(
            "GDN prefill chain conv_state length mismatch: got {} expected {}",
            conv_state.len(),
            shape.conv_state_len
        ));
    }
    let expected_qkv_len = shape
        .seq_len
        .checked_mul(dims.conv_channels)
        .ok_or_else(|| "GDN prefill chain qkv row length overflow".to_string())?;
    if qkv_rows.len() != expected_qkv_len {
        return Err(format!(
            "GDN prefill chain qkv row length mismatch: got {} expected {}",
            qkv_rows.len(),
            expected_qkv_len
        ));
    }

    let total_conv_len = shape
        .seq_len
        .checked_add(shape.conv_kernel.saturating_sub(1))
        .ok_or_else(|| "GDN prefill chain conv input token count overflow".to_string())?;
    let total_values = total_conv_len
        .checked_mul(dims.conv_channels)
        .ok_or_else(|| "GDN prefill chain conv input length overflow".to_string())?;
    let mut conv_input = vec![0.0f32; total_values];
    conv_input[..shape.conv_state_len].copy_from_slice(conv_state);
    conv_input[shape.conv_state_len..].copy_from_slice(qkv_rows);

    let final_state_start = shape
        .seq_len
        .checked_mul(dims.conv_channels)
        .ok_or_else(|| "GDN prefill chain final conv state offset overflow".to_string())?;
    let final_state =
        conv_input[final_state_start..final_state_start + shape.conv_state_len].to_vec();
    Ok((conv_input, final_state))
}

pub fn gdn_prefill_chain_conv_state_after_prefix_for_test(
    shape: &GdnPrefillChainShape,
    conv_input: &[f32],
    prefix_tokens: usize,
) -> Result<Vec<f32>, String> {
    let dims = derive_gdn_prefill_chain_dims(shape)?;
    if prefix_tokens > shape.seq_len {
        return Err(format!(
            "GDN prefill chain prefix out of range: prefix_tokens={} seq_len={}",
            prefix_tokens, shape.seq_len
        ));
    }
    let expected_len = shape
        .seq_len
        .checked_add(shape.conv_kernel.saturating_sub(1))
        .and_then(|tokens| tokens.checked_mul(dims.conv_channels))
        .ok_or_else(|| "GDN prefill chain conv input length overflow".to_string())?;
    if conv_input.len() != expected_len {
        return Err(format!(
            "GDN prefill chain conv input length mismatch: got {} expected {}",
            conv_input.len(),
            expected_len
        ));
    }
    let start = prefix_tokens
        .checked_mul(dims.conv_channels)
        .ok_or_else(|| "GDN prefill chain prefix offset overflow".to_string())?;
    Ok(conv_input[start..start + shape.conv_state_len].to_vec())
}
