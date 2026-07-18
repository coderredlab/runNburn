use super::super::mtp_verify::{validate_mtp_verify_k_quant_matrix, Qwen35MtpGdnProjectionRequest};
use super::super::*;

pub fn upload_device_tensor_f32(
    desc: rnb_backend_api::DeviceTensorDesc,
    input: &[f32],
) -> Result<rnb_backend_api::DeviceTensorId, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .upload_device_tensor_f32(desc, input)
}

pub fn download_device_tensor_f32(id: rnb_backend_api::DeviceTensorId) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.download_device_tensor_f32(id)
}

pub fn download_device_tensor_f32_row(
    id: rnb_backend_api::DeviceTensorId,
    row: usize,
) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.download_device_tensor_f32_row(id, row)
}

pub fn release_device_tensor(id: rnb_backend_api::DeviceTensorId) -> Result<bool, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.release_device_tensor(id)
}

fn copy_device_tensor_to_owned_slot(
    state: &mut CudaState,
    src_dev: u64,
    bytes: usize,
    desc: rnb_backend_api::DeviceTensorDesc,
    label: &str,
) -> Result<rnb_backend_api::DeviceTensorId, String> {
    let output_dev = unsafe { state.api.mem_alloc(bytes)? };
    let copy_result = unsafe {
        state
            .api
            .memcpy_dtod_async(output_dev, src_dev, bytes, state.stream)
    }
    .and_then(|_| state.stream_synchronize());
    if let Err(err) = copy_result {
        let _ = unsafe { state.api.mem_free(output_dev) };
        return Err(err);
    }
    match state.insert_device_tensor_slot(output_dev, bytes, desc) {
        Ok(output_id) => Ok(output_id),
        Err(err) => {
            let _ = unsafe { state.api.mem_free(output_dev) };
            Err(format!("{label} device tensor slot insert failed: {err}"))
        }
    }
}

pub fn ssm_conv1d_silu(
    input: &[f32],
    kernel: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, String> {
    let total_len = seq_len
        .checked_add(kernel_size.saturating_sub(1))
        .ok_or_else(|| "SSM conv input length overflow".to_string())?;
    if input.len() != total_len * channels {
        return Err(format!(
            "SSM conv input len mismatch: got {}, expected {}",
            input.len(),
            total_len * channels
        ));
    }
    if kernel.len() != kernel_size * channels {
        return Err(format!(
            "SSM conv kernel len mismatch: got {}, expected {}",
            kernel.len(),
            kernel_size * channels
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .ssm_conv1d_silu(input, kernel, seq_len, channels, kernel_size)
}

pub fn gdn_gated_norm_silu(
    delta_out: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    rows: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if delta_out.len() != rows * head_dim || z.len() != rows * head_dim {
        return Err(format!(
            "GDN gated norm input length mismatch: delta={} z={} expected={}",
            delta_out.len(),
            z.len(),
            rows * head_dim
        ));
    }
    if norm_weight.len() != head_dim {
        return Err(format!(
            "GDN gated norm weight length mismatch: got {}, expected {head_dim}",
            norm_weight.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .gdn_gated_norm_silu(delta_out, z, norm_weight, rows, head_dim, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_gated_norm_silu_f32_gemm(
    delta_out: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    proj_weights: &[f32],
    seq_len: usize,
    head_dim: usize,
    proj_rows: usize,
    proj_cols: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let rows = seq_len
        .checked_mul(proj_cols / head_dim)
        .ok_or_else(|| "GDN gated GEMM row count overflow".to_string())?;
    if delta_out.len() != rows * head_dim || z.len() != seq_len * proj_cols {
        return Err(format!(
            "GDN gated GEMM input length mismatch: delta={} z={} expected_delta={} expected_z={}",
            delta_out.len(),
            z.len(),
            rows * head_dim,
            seq_len * proj_cols
        ));
    }
    if norm_weight.len() != head_dim {
        return Err(format!(
            "GDN gated GEMM norm weight length mismatch: got {}, expected {head_dim}",
            norm_weight.len()
        ));
    }
    if proj_weights.len() != proj_rows * proj_cols {
        return Err(format!(
            "GDN gated GEMM projection weight length mismatch: got {}, expected {}",
            proj_weights.len(),
            proj_rows * proj_cols
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .gdn_gated_norm_silu_f32_gemm(
            delta_out,
            z,
            norm_weight,
            proj_weights,
            seq_len,
            head_dim,
            proj_rows,
            proj_cols,
            eps,
        )
}

pub fn gdn_prefill_chain_q4k(
    request: GdnPrefillChainQ4KRequest<'_>,
) -> Result<GdnPrefillChainQ4KOutput, String> {
    let dims = derive_gdn_prefill_chain_dims(&request.shape)?;
    let hidden_values = request
        .shape
        .seq_len
        .checked_mul(request.shape.hidden_dim)
        .ok_or_else(|| "GDN Q4K prefill chain hidden length overflow".to_string())?;
    let hidden_bytes = hidden_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "GDN Q4K prefill chain hidden byte length overflow".to_string())?;
    if let Some((_, hidden_desc)) = request.hidden_device {
        if hidden_desc.rows() != request.shape.seq_len
            || hidden_desc.cols() != request.shape.hidden_dim
        {
            return Err(format!(
                "GDN Q4K prefill chain device hidden shape mismatch: got {}x{}, expected {}x{}",
                hidden_desc.rows(),
                hidden_desc.cols(),
                request.shape.seq_len,
                request.shape.hidden_dim
            ));
        }
        if hidden_desc.dtype() != rnb_backend_api::ScalarType::F32 {
            return Err(format!(
                "GDN Q4K prefill chain device hidden expects F32, got {:?}",
                hidden_desc.dtype()
            ));
        }
        if !matches!(
            hidden_desc.role(),
            rnb_backend_api::DeviceTensorRole::Hidden
                | rnb_backend_api::DeviceTensorRole::MoeOutput
        ) {
            return Err(format!(
                "GDN Q4K prefill chain device hidden role mismatch: got {:?}",
                hidden_desc.role()
            ));
        }
    } else if request.hidden.len() != hidden_values {
        return Err(format!(
            "GDN Q4K prefill chain hidden length mismatch: got {}, expected {hidden_values}",
            request.hidden.len()
        ));
    }
    if request.attn_norm.len() != request.shape.hidden_dim {
        return Err(format!(
            "GDN Q4K prefill chain attn_norm length mismatch: got {}, expected {}",
            request.attn_norm.len(),
            request.shape.hidden_dim
        ));
    }
    if request.conv_state.len() != request.shape.conv_state_len {
        return Err(format!(
            "GDN Q4K prefill chain conv_state length mismatch: got {}, expected {}",
            request.conv_state.len(),
            request.shape.conv_state_len
        ));
    }
    let conv_kernel_values = request
        .shape
        .conv_kernel
        .checked_mul(dims.conv_channels)
        .ok_or_else(|| "GDN Q4K prefill chain conv_kernel length overflow".to_string())?;
    if request.conv_kernel.len() != conv_kernel_values {
        return Err(format!(
            "GDN Q4K prefill chain conv_kernel length mismatch: got {}, expected {conv_kernel_values}",
            request.conv_kernel.len()
        ));
    }
    if request.dt_bias.len() != dims.num_v_heads {
        return Err(format!(
            "GDN Q4K prefill chain dt_bias length mismatch: got {}, expected {}",
            request.dt_bias.len(),
            dims.num_v_heads
        ));
    }
    if request.ssm_a.len() != dims.num_v_heads {
        return Err(format!(
            "GDN Q4K prefill chain ssm_a length mismatch: got {}, expected {}",
            request.ssm_a.len(),
            dims.num_v_heads
        ));
    }
    if request.delta_state.len() != request.shape.delta_state_len {
        return Err(format!(
            "GDN Q4K prefill chain delta_state length mismatch: got {}, expected {}",
            request.delta_state.len(),
            request.shape.delta_state_len
        ));
    }
    if request.ssm_norm.len() != dims.head_v_dim {
        return Err(format!(
            "GDN Q4K prefill chain ssm_norm length mismatch: got {}, expected {}",
            request.ssm_norm.len(),
            dims.head_v_dim
        ));
    }
    if request.ssm_out_rows == 0 {
        return Err("GDN Q4K prefill chain requires ssm_out_rows > 0".to_string());
    }
    if request.ssm_out_cols != dims.v_dim {
        return Err(format!(
            "GDN Q4K prefill chain ssm_out_cols mismatch: got {}, expected {}",
            request.ssm_out_cols, dims.v_dim
        ));
    }
    let post_attn_norm = if request.keep_device_moe_input {
        if request.ssm_out_rows != request.shape.hidden_dim {
            return Err(format!(
                "GDN Q4K prefill chain MoE input requires ssm_out_rows == hidden_dim: got {}, expected {}",
                request.ssm_out_rows, request.shape.hidden_dim
            ));
        }
        let post_attn_norm = request
            .post_attn_norm
            .ok_or_else(|| "GDN Q4K prefill chain MoE input requires post_attn_norm".to_string())?;
        if post_attn_norm.len() != request.shape.hidden_dim {
            return Err(format!(
                "GDN Q4K prefill chain post_attn_norm length mismatch: got {}, expected {}",
                post_attn_norm.len(),
                request.shape.hidden_dim
            ));
        }
        Some(post_attn_norm)
    } else {
        None
    };
    let _ssm_out_blocks_per_row = validate_mtp_verify_k_quant_matrix(
        "GDN ssm_out",
        request.ssm_out_quant,
        request.ssm_out_q4k,
        request.ssm_out_rows,
        request.ssm_out_cols,
        dims.v_dim,
    )?;

    let output_values = request
        .shape
        .seq_len
        .checked_mul(request.ssm_out_rows)
        .ok_or_else(|| "GDN Q4K prefill chain output length overflow".to_string())?;
    let output_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "GDN Q4K prefill chain output byte overflow".to_string())?;
    if !request.keep_host_output && !request.keep_device_output && !request.keep_device_moe_input {
        return Err(
            "GDN Q4K prefill chain requires host output, device output, or MoE carriers"
                .to_string(),
        );
    }
    let conv_state_d2h_bytes = std::mem::size_of_val(request.conv_state);
    let delta_state_d2h_bytes = std::mem::size_of_val(request.delta_state);

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let plan = qwen35_mtp_verify_buffer_plan(request.shape.seq_len, request.shape.hidden_dim, 0)?;
    let verify_tokens = vec![0_u32; request.shape.seq_len];
    let buffers = state.stage_mtp_verify_window(&plan, &verify_tokens, &[])?;
    if let Some((hidden_id, hidden_desc)) = request.hidden_device {
        let hidden_dev = state.device_tensor_ptr(hidden_id, hidden_desc)?;
        unsafe {
            state.api.memcpy_dtod_async(
                buffers.hidden_rows_dev,
                hidden_dev,
                hidden_bytes,
                state.stream,
            )?;
        }
    } else {
        unsafe {
            state.api.memcpy_htod_async(
                buffers.hidden_rows_dev,
                request.hidden.as_ptr().cast::<libc::c_void>(),
                hidden_bytes,
                state.stream,
            )?;
        }
    }
    let projection_buffers = state.stage_mtp_verify_gdn_input_projections_q4k(
        &buffers,
        Qwen35MtpGdnProjectionRequest {
            attn_norm: request.attn_norm,
            qkv_q4k: request.qkv_q4k,
            qkv_quant: request.qkv_quant,
            qkv_rows: dims.conv_channels,
            qkv_cols: request.shape.hidden_dim,
            gate_q4k: request.gate_q4k,
            gate_rows: dims.v_dim,
            gate_cols: request.shape.hidden_dim,
            alpha_q4k: request.alpha_q4k,
            alpha_f32: request.alpha_f32,
            alpha_quant: request.alpha_quant,
            alpha_rows: dims.num_v_heads,
            alpha_cols: request.shape.hidden_dim,
            beta_q4k: request.beta_q4k,
            beta_f32: request.beta_f32,
            beta_quant: request.beta_quant,
            beta_rows: dims.num_v_heads,
            beta_cols: request.shape.hidden_dim,
            norm_eps: request.norm_eps,
        },
    )?;
    let conv_buffers = state.stage_mtp_verify_gdn_conv1d_silu(
        &projection_buffers,
        request.conv_state,
        request.conv_kernel,
        request.shape.conv_kernel,
    )?;
    let delta_buffers = state.stage_mtp_verify_gdn_delta_inputs(
        &conv_buffers,
        &projection_buffers,
        request.dt_bias,
        request.ssm_a,
        dims.num_k_heads,
        dims.num_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
        request.norm_eps,
    )?;
    let scan_buffers =
        state.stage_mtp_verify_gdn_delta_scan(&delta_buffers, request.delta_state, true)?;
    let ssm_buffers = state.stage_mtp_verify_gdn_ssm_out_q4k(
        &scan_buffers,
        &projection_buffers,
        request.ssm_norm,
        request.ssm_out_q4k,
        request.ssm_out_quant,
        request.ssm_out_rows,
        request.ssm_out_cols,
        request.norm_eps,
    )?;
    let final_conv_state = state.stage_mtp_verify_gdn_conv_final_state_deferred(&conv_buffers)?;
    let (ssm_projection, ssm_projection_d2h_bytes) = if request.keep_host_output {
        let mut ssm_projection = vec![0.0f32; output_values];
        unsafe {
            state.api.memcpy_dtoh_async(
                ssm_projection.as_mut_ptr().cast::<libc::c_void>(),
                ssm_buffers.ssm_out_dev,
                output_bytes,
                state.stream,
            )?;
        }
        (ssm_projection, output_bytes)
    } else {
        (Vec::new(), 0)
    };
    state.stream_synchronize()?;
    request.conv_state.copy_from_slice(&final_conv_state);
    let mut device_output = if request.keep_device_output {
        let output_desc = rnb_backend_api::DeviceTensorDesc::new(
            request.shape.seq_len,
            request.ssm_out_rows,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MambaOutput,
        );
        let output_id = copy_device_tensor_to_owned_slot(
            state,
            ssm_buffers.ssm_out_dev,
            output_bytes,
            output_desc,
            "GDN Q4K prefill chain ssm output",
        )?;
        Some((output_id, output_desc))
    } else {
        None
    };
    let (device_residual, device_moe_input) = if let Some(post_attn_norm) = post_attn_norm {
        if let Err(err) = state.launch_add_f32_inplace(
            buffers.hidden_rows_dev,
            ssm_buffers.ssm_out_dev,
            output_values,
        ) {
            if let Some((output_id, _)) = device_output.take() {
                let _ = state.release_device_tensor(output_id);
            }
            return Err(err);
        }
        if let Err(err) = state.stage_mtp_verify_hidden_rows_rms_norm(
            &buffers,
            post_attn_norm,
            request.norm_eps,
            false,
        ) {
            if let Some((output_id, _)) = device_output.take() {
                let _ = state.release_device_tensor(output_id);
            }
            return Err(err);
        }
        let residual_desc = rnb_backend_api::DeviceTensorDesc::new(
            request.shape.seq_len,
            request.shape.hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Hidden,
        );
        let residual_id = match copy_device_tensor_to_owned_slot(
            state,
            buffers.hidden_rows_dev,
            output_bytes,
            residual_desc,
            "GDN Q4K prefill chain residual carrier",
        ) {
            Ok(id) => id,
            Err(err) => {
                if let Some((output_id, _)) = device_output.take() {
                    let _ = state.release_device_tensor(output_id);
                }
                return Err(err);
            }
        };
        let moe_input_desc = rnb_backend_api::DeviceTensorDesc::new(
            request.shape.seq_len,
            request.shape.hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Normalized,
        );
        let moe_input_id = match copy_device_tensor_to_owned_slot(
            state,
            buffers.scratch_hidden_dev,
            output_bytes,
            moe_input_desc,
            "GDN Q4K prefill chain MoE input carrier",
        ) {
            Ok(id) => id,
            Err(err) => {
                if let Some((output_id, _)) = device_output.take() {
                    let _ = state.release_device_tensor(output_id);
                }
                let _ = state.release_device_tensor(residual_id);
                return Err(err);
            }
        };
        (
            Some((residual_id, residual_desc)),
            Some((moe_input_id, moe_input_desc)),
        )
    } else {
        (None, None)
    };
    Ok(GdnPrefillChainQ4KOutput {
        ssm_projection,
        ssm_projection_d2h_bytes,
        device_output,
        device_residual,
        device_moe_input,
        conv_state_d2h_bytes,
        delta_state_d2h_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_decode(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<Vec<f32>, String> {
    delta_net_decode_impl(
        state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_decode_resident(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<Vec<f32>, String> {
    delta_net_decode_impl(
        state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim, false,
    )
}

#[allow(clippy::too_many_arguments)]
fn delta_net_decode_impl(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    sync_state_to_host: bool,
) -> Result<Vec<f32>, String> {
    let expected_qk = num_heads
        .checked_mul(head_k_dim)
        .ok_or_else(|| "delta q/k size overflow".to_string())?;
    let expected_v = num_heads
        .checked_mul(head_v_dim)
        .ok_or_else(|| "delta v size overflow".to_string())?;
    let expected_state = expected_v
        .checked_mul(head_k_dim)
        .ok_or_else(|| "delta state size overflow".to_string())?;
    if q.len() != expected_qk || k.len() != expected_qk {
        return Err(format!(
            "delta q/k len mismatch: q={}, k={}, expected={expected_qk}",
            q.len(),
            k.len()
        ));
    }
    if v.len() != expected_v || gate.len() != num_heads || beta.len() != num_heads {
        return Err(format!(
            "delta input len mismatch: v={}, gate={}, beta={}, expected_v={}, heads={num_heads}",
            v.len(),
            gate.len(),
            beta.len(),
            expected_v
        ));
    }
    if state.len() != expected_state {
        return Err(format!(
            "delta state len mismatch: got {}, expected {expected_state}",
            state.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .delta_net_decode(
            state,
            q,
            k,
            v,
            gate,
            beta,
            num_heads,
            head_k_dim,
            head_v_dim,
            sync_state_to_host,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_mamba2_decode_scan(
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
) -> Result<Vec<f32>, String> {
    if num_heads == 0 || head_dim == 0 || state_dim == 0 || n_group == 0 {
        return Err("Nemotron Mamba2 scan dimensions must be non-zero".to_string());
    }
    if state_dim > 256 {
        return Err(format!(
            "Nemotron Mamba2 scan state_dim={state_dim} exceeds CUDA block width 256"
        ));
    }
    if num_heads % n_group != 0 {
        return Err(format!(
            "Nemotron Mamba2 heads must be divisible by groups: heads={num_heads}, groups={n_group}"
        ));
    }
    let d_inner = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Nemotron Mamba2 d_inner overflow".to_string())?;
    let bc_dim = n_group
        .checked_mul(state_dim)
        .ok_or_else(|| "Nemotron Mamba2 bc_dim overflow".to_string())?;
    let expected_state = d_inner
        .checked_mul(state_dim)
        .ok_or_else(|| "Nemotron Mamba2 state size overflow".to_string())?;
    if x.len() != d_inner || b.len() != bc_dim || c.len() != bc_dim {
        return Err(format!(
            "Nemotron Mamba2 scan input length mismatch: x={} b={} c={} expected x={} bc={}",
            x.len(),
            b.len(),
            c.len(),
            d_inner,
            bc_dim
        ));
    }
    if dt.len() != num_heads || a.len() < num_heads || d.len() < num_heads {
        return Err(format!(
            "Nemotron Mamba2 scan head vector length mismatch: dt={} a={} d={} heads={num_heads}",
            dt.len(),
            a.len(),
            d.len()
        ));
    }
    if state.len() != expected_state {
        return Err(format!(
            "Nemotron Mamba2 state len mismatch: got {}, expected {expected_state}",
            state.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .nemotron_mamba2_decode_scan(
            state, x, b, c, dt, a, d, num_heads, head_dim, state_dim, n_group,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_mamba2_prefill_scan(
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
) -> Result<Vec<f32>, String> {
    if seq_len == 0
        || d_inner == 0
        || conv_channels == 0
        || bc_dim == 0
        || num_heads == 0
        || head_dim == 0
        || n_group == 0
        || state_dim == 0
    {
        return Err("Nemotron Mamba2 prefill scan dimensions must be non-zero".to_string());
    }
    if state_dim > 256 {
        return Err(format!(
            "Nemotron Mamba2 prefill scan state_dim={state_dim} exceeds CUDA block width 256"
        ));
    }
    let expected_d_inner = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Nemotron Mamba2 prefill d_inner overflow".to_string())?;
    if d_inner != expected_d_inner {
        return Err(format!(
            "Nemotron Mamba2 prefill d_inner mismatch: d_inner={d_inner}, heads={num_heads}, head_dim={head_dim}"
        ));
    }
    if num_heads % n_group != 0 {
        return Err(format!(
            "Nemotron Mamba2 prefill heads must be divisible by groups: heads={num_heads}, groups={n_group}"
        ));
    }
    let expected_bc_dim = n_group
        .checked_mul(state_dim)
        .ok_or_else(|| "Nemotron Mamba2 prefill bc_dim overflow".to_string())?;
    if bc_dim != expected_bc_dim {
        return Err(format!(
            "Nemotron Mamba2 prefill bc_dim mismatch: bc_dim={bc_dim}, groups={n_group}, state_dim={state_dim}"
        ));
    }
    let expected_conv = seq_len
        .checked_mul(conv_channels)
        .ok_or_else(|| "Nemotron Mamba2 prefill conv size overflow".to_string())?;
    let expected_dt = seq_len
        .checked_mul(num_heads)
        .ok_or_else(|| "Nemotron Mamba2 prefill dt size overflow".to_string())?;
    let expected_state = d_inner
        .checked_mul(state_dim)
        .ok_or_else(|| "Nemotron Mamba2 prefill state size overflow".to_string())?;
    if conv_activated.len() != expected_conv || dt_data.len() != expected_dt {
        return Err(format!(
            "Nemotron Mamba2 prefill input length mismatch: conv={} dt={} expected conv={} dt={}",
            conv_activated.len(),
            dt_data.len(),
            expected_conv,
            expected_dt
        ));
    }
    if a.len() < num_heads || d.len() < num_heads || state.len() != expected_state {
        return Err(format!(
            "Nemotron Mamba2 prefill head/state length mismatch: a={} d={} state={} heads={} expected_state={expected_state}",
            a.len(),
            d.len(),
            state.len(),
            num_heads
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .nemotron_mamba2_prefill_scan(
            state,
            conv_activated,
            dt_data,
            a,
            d,
            seq_len,
            conv_channels,
            bc_dim,
            num_heads,
            head_dim,
            state_dim,
            n_group,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_mamba2_prefill_device(
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    ssm_in_quant: u32,
    ssm_in: &[u8],
    ssm_in_rows: usize,
    ssm_in_cols: usize,
    ssm_out_quant: u32,
    ssm_out: &[u8],
    ssm_out_rows: usize,
    ssm_out_cols: usize,
    input_norm: &[f32],
    conv_kernel: &[f32],
    conv_bias: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_d: &[f32],
    ssm_norm: &[f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    d_inner: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    n_group: usize,
    d_state: usize,
    conv_kernel_size: usize,
    norm_eps: f32,
) -> Result<NemotronMamba2DeviceOutput, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        ssm_in_quant,
        ssm_in,
        ssm_in_rows,
        ssm_in_cols,
        ssm_out_quant,
        ssm_out,
        ssm_out_rows,
        ssm_out_cols,
        input_norm,
        conv_kernel,
        conv_bias,
        dt_bias,
        ssm_a,
        ssm_d,
        ssm_norm,
        conv_state,
        delta_state,
        seq_len,
        hidden_dim,
        d_inner,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        n_group,
        d_state,
        conv_kernel_size,
        norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_prefill(
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
) -> Result<Vec<f32>, String> {
    delta_net_prefill_impl(
        state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_prefill_resident(
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
) -> Result<Vec<f32>, String> {
    delta_net_prefill_impl(
        state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim, false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_prefill_resident_snapshot(
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
) -> Result<(Vec<f32>, Option<DeltaStateSnapshot>), String> {
    delta_net_prefill_snapshot_impl(
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
        false,
        Some(snapshot_after_tokens),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_net_prefill_resident_snapshots(
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
) -> Result<(Vec<f32>, Vec<DeltaStateSnapshot>), String> {
    delta_net_prefill_snapshots_impl(
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
        false,
        snapshot_after_tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn delta_net_prefill_impl(
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
    sync_state_to_host: bool,
) -> Result<Vec<f32>, String> {
    delta_net_prefill_snapshot_impl(
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
        sync_state_to_host,
        None,
    )
    .map(|(output, _)| output)
}

#[allow(clippy::too_many_arguments)]
fn delta_net_prefill_snapshot_impl(
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
    sync_state_to_host: bool,
    snapshot_after_tokens: Option<usize>,
) -> Result<(Vec<f32>, Option<DeltaStateSnapshot>), String> {
    let expected_qk = seq_len
        .checked_mul(num_heads)
        .and_then(|x| x.checked_mul(head_k_dim))
        .ok_or_else(|| "delta prefill q/k size overflow".to_string())?;
    let expected_v = seq_len
        .checked_mul(num_heads)
        .and_then(|x| x.checked_mul(head_v_dim))
        .ok_or_else(|| "delta prefill v size overflow".to_string())?;
    let expected_gate = seq_len
        .checked_mul(num_heads)
        .ok_or_else(|| "delta prefill gate size overflow".to_string())?;
    let expected_state = num_heads
        .checked_mul(head_v_dim)
        .and_then(|x| x.checked_mul(head_k_dim))
        .ok_or_else(|| "delta prefill state size overflow".to_string())?;
    if q.len() != expected_qk || k.len() != expected_qk {
        return Err(format!(
            "delta prefill q/k len mismatch: q={}, k={}, expected={expected_qk}",
            q.len(),
            k.len()
        ));
    }
    if v.len() != expected_v || gate.len() != expected_gate || beta.len() != expected_gate {
        return Err(format!(
            "delta prefill input len mismatch: v={}, gate={}, beta={}, expected_v={}, expected_gate={expected_gate}",
            v.len(),
            gate.len(),
            beta.len(),
            expected_v
        ));
    }
    if state.len() != expected_state {
        return Err(format!(
            "delta prefill state len mismatch: got {}, expected {expected_state}",
            state.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .delta_net_prefill_with_snapshot(
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
            sync_state_to_host,
            snapshot_after_tokens,
        )
}

#[allow(clippy::too_many_arguments)]
fn delta_net_prefill_snapshots_impl(
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
    sync_state_to_host: bool,
    snapshot_after_tokens: &[usize],
) -> Result<(Vec<f32>, Vec<DeltaStateSnapshot>), String> {
    let expected_qk = seq_len
        .checked_mul(num_heads)
        .and_then(|x| x.checked_mul(head_k_dim))
        .ok_or_else(|| "delta prefill q/k size overflow".to_string())?;
    let expected_v = seq_len
        .checked_mul(num_heads)
        .and_then(|x| x.checked_mul(head_v_dim))
        .ok_or_else(|| "delta prefill v size overflow".to_string())?;
    let expected_gate = seq_len
        .checked_mul(num_heads)
        .ok_or_else(|| "delta prefill gate size overflow".to_string())?;
    let expected_state = num_heads
        .checked_mul(head_v_dim)
        .and_then(|x| x.checked_mul(head_k_dim))
        .ok_or_else(|| "delta prefill state size overflow".to_string())?;
    if q.len() != expected_qk || k.len() != expected_qk {
        return Err(format!(
            "delta prefill q/k len mismatch: q={}, k={}, expected={expected_qk}",
            q.len(),
            k.len()
        ));
    }
    if v.len() != expected_v || gate.len() != expected_gate || beta.len() != expected_gate {
        return Err(format!(
            "delta prefill input len mismatch: v={}, gate={}, beta={}, expected_v={}, expected_gate={expected_gate}",
            v.len(),
            gate.len(),
            beta.len(),
            expected_v
        ));
    }
    if state.len() != expected_state {
        return Err(format!(
            "delta prefill state len mismatch: got {}, expected {expected_state}",
            state.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .delta_net_prefill_with_snapshots(
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
            sync_state_to_host,
            snapshot_after_tokens,
        )
}

pub fn reset_delta_state_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if let Some(state) = guard.as_mut() {
        state.clear_resident_q4k_cache()?;
        state.clear_resident_delta_states()?;
    }
    Ok(())
}

pub fn sync_delta_state_cache(state: &mut [f32]) -> Result<bool, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(cuda_state) = guard.as_mut() else {
        return Ok(false);
    };
    cuda_state.sync_resident_delta_state(state)
}

pub fn snapshot_delta_state_cache(state: &mut [f32]) -> Result<Option<DeltaStateSnapshot>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(cuda_state) = guard.as_mut() else {
        return Ok(None);
    };
    cuda_state.snapshot_resident_delta_state(state)
}

pub fn restore_delta_state_cache(
    state: &mut [f32],
    snapshot: &DeltaStateSnapshot,
) -> Result<bool, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(cuda_state) = guard.as_mut() else {
        return Ok(false);
    };
    cuda_state.restore_resident_delta_state(state, snapshot)
}

pub fn free_delta_state_snapshot(snapshot: DeltaStateSnapshot) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(cuda_state) = guard.as_mut() else {
        return Ok(());
    };
    cuda_state.free_delta_state_snapshot(snapshot)
}
