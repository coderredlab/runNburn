use rnb_loader::GGMLType;

use super::{backend, dequant, dequant_type, Result};
use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId};

pub type DeltaStateSnapshot = backend::DeltaStateSnapshot;
pub type GdnPrefillChainShape = backend::GdnPrefillChainShape;
pub type GdnPrefillChainQ4KOutput = backend::GdnPrefillChainQ4KOutput;

pub type NemotronPrefillWorkspaceSummary = rnb_backend_cuda::NemotronPrefillWorkspaceSummary;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillChainOutput;

pub struct GdnPrefillChainQ4KRequest<'a> {
    pub shape: GdnPrefillChainShape,
    pub hidden: &'a [f32],
    pub hidden_device: Option<(DeviceTensorId, DeviceTensorDesc)>,
    pub attn_norm: &'a [f32],
    pub qkv_q4k: &'a [u8],
    pub qkv_quant: GGMLType,
    pub gate_q4k: &'a [u8],
    pub gate_quant: GGMLType,
    pub alpha_q4k: &'a [u8],
    pub alpha_f32: &'a [f32],
    pub alpha_quant: GGMLType,
    pub beta_q4k: &'a [u8],
    pub beta_f32: &'a [f32],
    pub beta_quant: GGMLType,
    pub conv_state: &'a mut [f32],
    pub conv_kernel: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub delta_state: &'a mut [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_q4k: &'a [u8],
    pub ssm_out_quant: GGMLType,
    pub ssm_out_rows: usize,
    pub ssm_out_cols: usize,
    pub norm_eps: f32,
    pub keep_host_output: bool,
    pub post_attn_norm: Option<&'a [f32]>,
}

pub fn begin_nemotron_prefill_workspace(
    plan: &crate::NemotronPrefillWorkspacePlan,
) -> Result<NemotronPrefillWorkspaceSummary> {
    let sparse_slots = plan.route_slots.max(1);
    let moe_sparse_mid_bytes = sparse_slots
        .checked_mul(plan.n_ff)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Nemotron workspace sparse mid byte overflow".to_string())?;
    let moe_shared_mid_bytes = plan
        .chunk_len
        .checked_mul(plan.shared_ff)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Nemotron workspace shared mid byte overflow".to_string())?;
    let enabled = !matches!(
        plan.decision,
        crate::NemotronPrefillWorkspaceDecision::Fallback
    );

    rnb_backend_cuda::begin_nemotron_prefill_workspace(
        rnb_backend_cuda::NemotronPrefillWorkspaceConfig {
            hidden_bytes: plan.hidden_bytes,
            normalized_bytes: plan.normalized_bytes,
            router_logits_bytes: plan.router_logits_bytes,
            route_bytes: plan.route_bytes,
            moe_shared_mid_bytes,
            moe_sparse_mid_bytes,
            required_workspace_bytes: plan.required_workspace_bytes,
            enabled,
        },
    )
    .map_err(|err| format!("CUDA Nemotron prefill workspace begin failed: {err}"))
}

pub fn end_nemotron_prefill_workspace() -> Result<NemotronPrefillWorkspaceSummary> {
    rnb_backend_cuda::end_nemotron_prefill_workspace()
        .map_err(|err| format!("CUDA Nemotron prefill workspace end failed: {err}"))
}

fn gdn_prefill_batch_quant_supported(ggml_type: GGMLType) -> bool {
    matches!(
        ggml_type,
        GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q4_K
            | GGMLType::Q5_K
            | GGMLType::Q6_K
            | GGMLType::IQ4_XS
    )
}

fn gdn_prefill_effective_gemv_mode(
    ggml_type: GGMLType,
    seq_len: usize,
    requested_mode: &str,
) -> String {
    if gdn_prefill_batch_quant_supported(ggml_type) && seq_len <= 2 && requested_mode == "f32" {
        "q".to_string()
    } else {
        requested_mode.to_string()
    }
}

pub fn gdn_prefill_quantized_projection(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    if cols == 0 {
        return Ok(None);
    }
    let seq_len = input.len() / cols;
    let Some(mode) = backend::tuning::gdn_prefill_gemv_mode_for_seq(seq_len) else {
        return Ok(None);
    };
    let mode = gdn_prefill_effective_gemv_mode(ggml_type, seq_len, &mode);
    let out = if mode == "f32" {
        if ggml_type == GGMLType::Q4_K {
            if let Some(out) = backend::q4k_f32_gemm_batch_cached(bytes, rows, cols, input)? {
                return Ok(Some(out));
            }
        }
        if ggml_type == GGMLType::Q8_0 {
            if let Some(out) = backend::q8_0_f32_gemm_batch_cached(bytes, rows, cols, input)? {
                return Ok(Some(out));
            }
        }
        let weights_f32 = dequant::dequantize_bytes_to_f32(bytes, dequant_type(ggml_type));
        backend::f32_gemm_batch(&weights_f32, rows, cols, input)
    } else {
        if !gdn_prefill_batch_quant_supported(ggml_type)
            && !(ggml_type == GGMLType::Q8_0 && backend::tuning::prefill_q8_0_batch_enabled())
        {
            return Ok(None);
        }
        match ggml_type {
            GGMLType::Q5_0 => backend::q5_0_gemv_batch(bytes, rows, cols, input),
            GGMLType::Q5_1 => backend::q5_1_gemv_batch(bytes, rows, cols, input),
            GGMLType::Q4_K => backend::q4k_gemv_batch(bytes, rows, cols, input),
            GGMLType::Q5_K => backend::q5k_gemv_batch(bytes, rows, cols, input),
            GGMLType::Q6_K => backend::q6k_gemv_batch(bytes, rows, cols, input),
            GGMLType::IQ4_XS => backend::iq4_xs_gemv_batch(bytes, rows, cols, input),
            GGMLType::Q8_0 if backend::tuning::prefill_q8_0_batch_enabled() => {
                backend::q8_0_gemv_batch(bytes, rows, cols, input)
            }
            _ => return Ok(None),
        }
    }
    .map_err(|err| {
        format!("CUDA GDN prefill GEMV failed for {ggml_type:?} [{rows}x{cols}]: {err}")
    })?;
    Ok(Some(out))
}

pub fn gdn_prefill_quantized_projection_q(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    if cols == 0 {
        return Ok(None);
    }
    if !gdn_prefill_batch_quant_supported(ggml_type) {
        return Ok(None);
    }
    let out = match ggml_type {
        GGMLType::Q5_0 => backend::q5_0_gemv_batch(bytes, rows, cols, input),
        GGMLType::Q5_1 => backend::q5_1_gemv_batch(bytes, rows, cols, input),
        GGMLType::Q4_K => backend::q4k_gemv_batch(bytes, rows, cols, input),
        GGMLType::Q5_K => backend::q5k_gemv_batch(bytes, rows, cols, input),
        GGMLType::Q6_K => backend::q6k_gemv_batch(bytes, rows, cols, input),
        GGMLType::IQ4_XS => backend::iq4_xs_gemv_batch(bytes, rows, cols, input),
        _ => return Ok(None),
    }
    .map_err(|err| {
        format!("CUDA GDN prefill quant GEMV failed for {ggml_type:?} [{rows}x{cols}]: {err}")
    })?;
    Ok(Some(out))
}

pub fn ensure_gdn_prefill_chunk_supported(seq_len: usize, hidden_dim: usize) -> Result<()> {
    if !backend::tuning::gdn_prefill_enabled() {
        return Ok(());
    }
    Err(backend::gdn_prefill_chunk_unimplemented_for_test(seq_len, hidden_dim).unwrap_err())
}

pub fn gdn_prefill_chain(shape: &GdnPrefillChainShape) -> Result<Option<GdnPrefillChainOutput>> {
    match backend::plan_gdn_prefill_chain(shape)
        .map_err(|err| format!("CUDA GDN prefill chain failed: {err}"))?
    {
        backend::GdnPrefillChainPlan::Disabled => Ok(None),
        backend::GdnPrefillChainPlan::Q4KDeviceChain => Ok(Some(GdnPrefillChainOutput)),
    }
}

pub fn gdn_prefill_chain_q4k(
    request: GdnPrefillChainQ4KRequest<'_>,
) -> Result<Option<GdnPrefillChainQ4KOutput>> {
    match backend::plan_gdn_prefill_chain(&request.shape)
        .map_err(|err| format!("CUDA GDN prefill chain failed: {err}"))?
    {
        backend::GdnPrefillChainPlan::Disabled => Ok(None),
        backend::GdnPrefillChainPlan::Q4KDeviceChain => {
            let f32_projection_mode =
                backend::tuning::gdn_prefill_gemv_mode_for_seq(request.shape.seq_len).as_deref()
                    == Some("f32");
            let alpha_dequant =
                (f32_projection_mode && request.alpha_quant != GGMLType::F32).then(|| {
                    dequant::dequantize_bytes_to_f32(
                        request.alpha_q4k,
                        dequant_type(request.alpha_quant),
                    )
                });
            let beta_dequant =
                (f32_projection_mode && request.beta_quant != GGMLType::F32).then(|| {
                    dequant::dequantize_bytes_to_f32(
                        request.beta_q4k,
                        dequant_type(request.beta_quant),
                    )
                });
            backend::gdn_prefill_chain_q4k(backend::GdnPrefillChainQ4KRequest {
                shape: request.shape,
                hidden: request.hidden,
                hidden_device: request.hidden_device,
                attn_norm: request.attn_norm,
                qkv_q4k: request.qkv_q4k,
                qkv_quant: request.qkv_quant as u32,
                gate_q4k: request.gate_q4k,
                gate_quant: request.gate_quant as u32,
                alpha_q4k: request.alpha_q4k,
                alpha_f32: alpha_dequant.as_deref().unwrap_or(request.alpha_f32),
                alpha_quant: request.alpha_quant as u32,
                beta_q4k: request.beta_q4k,
                beta_f32: beta_dequant.as_deref().unwrap_or(request.beta_f32),
                beta_quant: request.beta_quant as u32,
                conv_state: request.conv_state,
                conv_kernel: request.conv_kernel,
                dt_bias: request.dt_bias,
                ssm_a: request.ssm_a,
                delta_state: request.delta_state,
                ssm_norm: request.ssm_norm,
                ssm_out_q4k: request.ssm_out_q4k,
                ssm_out_quant: request.ssm_out_quant as u32,
                ssm_out_rows: request.ssm_out_rows,
                ssm_out_cols: request.ssm_out_cols,
                norm_eps: request.norm_eps,
                keep_host_output: request.keep_host_output,
                keep_device_output: backend::tuning::gdn_prefill_chain_device_output_enabled(),
                post_attn_norm: request.post_attn_norm,
                keep_device_moe_input: backend::tuning::gdn_prefill_chain_moe_input_device_enabled(
                ),
            })
            .map(Some)
            .map_err(|err| format!("CUDA GDN Q4K prefill chain failed: {err}"))
        }
    }
}

pub fn ssm_prefill_conv1d_silu(
    input: &[f32],
    kernel: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::prefill_conv_enabled() {
        return Ok(None);
    }
    backend::ssm_conv1d_silu(input, kernel, seq_len, channels, kernel_size)
        .map(Some)
        .map_err(|err| format!("CUDA prefill conv1d+silu failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_delta_net(
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
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::prefill_delta_enabled() {
        return Ok(None);
    }
    backend::delta_net_prefill(
        state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim,
    )
    .map(Some)
    .map_err(|err| format!("CUDA prefill delta-net failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_delta_net_resident(
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
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::prefill_delta_enabled() {
        return Ok(None);
    }
    let result = if backend::tuning::delta_state_sync_each_step_enabled() {
        backend::delta_net_prefill(
            state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim,
        )
    } else {
        backend::delta_net_prefill_resident(
            state, q, k, v, gate, beta, seq_len, num_heads, head_k_dim, head_v_dim,
        )
    };
    result
        .map(Some)
        .map_err(|err| format!("CUDA resident prefill delta-net failed: {err}"))
}

pub fn sync_delta_state_cache(state: &mut [f32]) -> Result<bool> {
    backend::sync_delta_state_cache(state)
        .map_err(|err| format!("CUDA delta state sync failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_delta_net_resident_snapshot(
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
) -> Result<(Vec<f32>, Option<DeltaStateSnapshot>)> {
    backend::delta_net_prefill_resident_snapshot(
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
    .map_err(|err| format!("CUDA resident prefill delta-net snapshot failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_delta_net_resident_snapshots(
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
) -> Result<(Vec<f32>, Vec<DeltaStateSnapshot>)> {
    backend::delta_net_prefill_resident_snapshots(
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
    .map_err(|err| format!("CUDA resident prefill delta-net snapshots failed: {err}"))
}

pub fn snapshot_delta_state_cache(state: &mut [f32]) -> Result<Option<DeltaStateSnapshot>> {
    backend::snapshot_delta_state_cache(state)
        .map_err(|err| format!("CUDA delta state snapshot failed: {err}"))
}

pub fn restore_delta_state_cache(state: &mut [f32], snapshot: &DeltaStateSnapshot) -> Result<bool> {
    backend::restore_delta_state_cache(state, snapshot)
        .map_err(|err| format!("CUDA delta state restore failed: {err}"))
}

pub fn free_delta_state_snapshot(snapshot: DeltaStateSnapshot) -> Result<()> {
    backend::free_delta_state_snapshot(snapshot)
        .map_err(|err| format!("CUDA delta state snapshot free failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn try_delta_step_resident_if_supported(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Option<std::result::Result<Vec<f32>, String>> {
    backend::tuning::delta_net_enabled().then(|| {
        if backend::tuning::delta_state_sync_each_step_enabled() {
            backend::delta_net_decode(
                state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim,
            )
        } else {
            backend::delta_net_decode_resident(
                state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim,
            )
        }
    })
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
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_mamba2_decode_scan(
        state, x, b, c, dt, a, d, num_heads, head_dim, state_dim, n_group,
    )
    .map(Some)
    .map_err(|err| format!("CUDA Nemotron Mamba2 decode scan failed: {err}"))
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
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_mamba2_prefill_scan(
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
    .map(Some)
    .map_err(|err| format!("CUDA Nemotron Mamba2 prefill scan failed: {err}"))
}

pub struct NemotronMamba2DeviceOutput {
    pub output_id: rnb_backend_api::DeviceTensorId,
    pub output_desc: rnb_backend_api::DeviceTensorDesc,
    pub conv_state_d2h_bytes: usize,
    pub delta_state_d2h_bytes: usize,
}

pub struct NemotronDeviceRouterLogitsOutput {
    pub normalized_id: rnb_backend_api::DeviceTensorId,
    pub normalized_desc: rnb_backend_api::DeviceTensorDesc,
    pub router_logits_id: rnb_backend_api::DeviceTensorId,
    pub router_logits_desc: rnb_backend_api::DeviceTensorDesc,
}

#[derive(Debug)]
pub struct NemotronDeviceRoutePack {
    inner: backend::BackendNemotronDeviceRoutePack,
}

impl NemotronDeviceRoutePack {
    pub fn slots(&self) -> usize {
        self.inner.slots()
    }

    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    pub fn expert_used(&self) -> usize {
        self.inner.expert_used()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_router_logits_from_device_f32(
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    norm_weight: &[f32],
    router_weight_f32: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    n_expert: usize,
    norm_eps: f32,
) -> Result<NemotronDeviceRouterLogitsOutput> {
    let output = backend::nemotron_router_logits_from_device_f32(
        input_id,
        input_desc,
        norm_weight,
        router_weight_f32,
        seq_len,
        hidden_dim,
        n_expert,
        norm_eps,
    )
    .map_err(|err| format!("CUDA Nemotron router logits from device failed: {err}"))?;
    Ok(NemotronDeviceRouterLogitsOutput {
        normalized_id: output.normalized_id,
        normalized_desc: output.normalized_desc,
        router_logits_id: output.router_logits_id,
        router_logits_desc: output.router_logits_desc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_device_route_pack_from_logits(
    router_logits_id: rnb_backend_api::DeviceTensorId,
    router_logits_desc: rnb_backend_api::DeviceTensorDesc,
    bias: Option<&[f32]>,
    seq_len: usize,
    n_expert: usize,
    expert_used: usize,
    expert_weight_scale: f32,
) -> Result<NemotronDeviceRoutePack> {
    let inner = backend::nemotron_device_route_pack_from_logits(
        router_logits_id,
        router_logits_desc,
        bias,
        seq_len,
        n_expert,
        expert_used,
        expert_weight_scale,
    )
    .map_err(|err| format!("CUDA Nemotron device route pack failed: {err}"))?;
    Ok(NemotronDeviceRoutePack { inner })
}

pub fn nemotron_device_route_pack_expert_ids(route: &NemotronDeviceRoutePack) -> Result<Vec<u32>> {
    backend::nemotron_device_route_pack_expert_ids(&route.inner)
        .map_err(|err| format!("CUDA Nemotron device route pack expert ids download failed: {err}"))
}

pub fn nemotron_reorder_device_route_pack(
    route: &NemotronDeviceRoutePack,
    order_indices: &[u32],
) -> Result<NemotronDeviceRoutePack> {
    let inner = backend::nemotron_reorder_device_route_pack(&route.inner, order_indices)
        .map_err(|err| format!("CUDA Nemotron device route pack reorder failed: {err}"))?;
    Ok(NemotronDeviceRoutePack { inner })
}

pub fn release_nemotron_device_route_pack(route: NemotronDeviceRoutePack) -> Result<()> {
    backend::release_nemotron_device_route_pack(route.inner)
        .map_err(|err| format!("CUDA Nemotron device route pack release failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_pack: &NemotronDeviceRoutePack,
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
) -> Result<Option<rnb_backend_api::DeviceTensorId>> {
    backend::nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        &route_pack.inner,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
        residual_desc,
    )
    .map_err(|err| {
        format!("CUDA Nemotron device route-pack Q8 shared + Q5 sparse prefill MoE failed: {err}")
    })
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
) -> Result<NemotronMamba2DeviceOutput> {
    let output = backend::nemotron_mamba2_prefill_device(
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
    .map_err(|err| format!("CUDA Nemotron Mamba2 device prefill failed: {err}"))?;
    Ok(NemotronMamba2DeviceOutput {
        output_id: output.output_id,
        output_desc: output.output_desc,
        conv_state_d2h_bytes: output.conv_state_d2h_bytes,
        delta_state_d2h_bytes: output.delta_state_d2h_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_gated_norm_silu_project(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    weight_ggml_type: GGMLType,
    weight_bytes: Option<&[u8]>,
    seq_len: usize,
    head_dim: usize,
    rows: usize,
    cols: usize,
    norm_eps: f32,
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::gdn_gated_norm_gemm_enabled_for_seq(seq_len)
        || seq_len <= 1
        || !matches!(
            weight_ggml_type,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
    {
        return Ok(None);
    }
    let Some(weight_bytes) = weight_bytes else {
        return Err("GDN ssm_out has no quantized bytes".to_string());
    };
    let weights_f32 =
        dequant::dequantize_bytes_to_f32(weight_bytes, dequant_type(weight_ggml_type));
    backend::gdn_gated_norm_silu_f32_gemm(
        output,
        z,
        norm,
        &weights_f32,
        seq_len,
        head_dim,
        rows,
        cols,
        norm_eps,
    )
    .map(Some)
    .map_err(|err| format!("CUDA GDN gated norm+silu+GEMM failed: {err}"))
}

pub fn gdn_prefill_gated_norm_silu(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    seq_len: usize,
    rows: usize,
    cols: usize,
    norm_eps: f32,
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::gdn_gated_norm_enabled() || seq_len <= 1 {
        return Ok(None);
    }
    backend::gdn_gated_norm_silu(output, z, norm, rows, cols, norm_eps)
        .map(Some)
        .map_err(|err| format!("CUDA GDN gated norm+silu failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_prefill_batch_quant_supports_iq4_xs() {
        assert!(gdn_prefill_batch_quant_supported(GGMLType::IQ4_XS));
    }

    #[test]
    fn short_verify_window_prefers_quant_batch_mode() {
        assert_eq!(
            gdn_prefill_effective_gemv_mode(GGMLType::Q4_K, 2, "f32"),
            "q"
        );
        assert_eq!(
            gdn_prefill_effective_gemv_mode(GGMLType::IQ4_XS, 2, "f32"),
            "q"
        );
    }

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

    #[test]
    fn gdn_prefill_chain_facade_defaults_on_and_allows_opt_out() {
        let shape = qwen35_chain_shape(32);
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
        }
        assert_eq!(
            gdn_prefill_chain(&shape).expect("default chain facade"),
            Some(GdnPrefillChainOutput)
        );

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "0");
        }
        assert_eq!(
            gdn_prefill_chain(&shape).expect("explicit off chain facade"),
            None
        );

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "1");
        }
        assert_eq!(
            gdn_prefill_chain(&shape).expect("enabled chain facade"),
            Some(GdnPrefillChainOutput)
        );

        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
        }
    }

    #[test]
    fn nemotron_device_route_pack_facade_downloads_experts() {
        let seq_len = 2usize;
        let n_expert = 4usize;
        let expert_used = 2usize;
        let logits = [0.5_f32, -0.25, 1.0, 0.125, -0.5, 0.75, 0.25, 1.25];
        let bias = [0.0_f32, 0.1, -0.2, 0.0];
        let logits_desc = rnb_backend_api::DeviceTensorDesc::new(
            seq_len,
            n_expert,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::RouterLogits,
        );
        let logits_id = crate::cuda_inference::cuda::upload_device_tensor_f32(logits_desc, &logits)
            .expect("upload logits");

        let route = nemotron_device_route_pack_from_logits(
            logits_id,
            logits_desc,
            Some(&bias),
            seq_len,
            n_expert,
            expert_used,
            1.0,
        )
        .expect("route pack");

        assert_eq!(route.slots(), seq_len * expert_used);
        assert_eq!(route.seq_len(), seq_len);
        assert_eq!(route.expert_used(), expert_used);
        let experts = nemotron_device_route_pack_expert_ids(&route).expect("download experts");
        assert_eq!(experts.len(), seq_len * expert_used);
        let order = [1_u32, 0, 3, 2];
        let sorted =
            nemotron_reorder_device_route_pack(&route, &order).expect("reorder route pack");
        let sorted_experts =
            nemotron_device_route_pack_expert_ids(&sorted).expect("download sorted experts");
        assert_eq!(
            sorted_experts,
            order
                .iter()
                .map(|&idx| experts[idx as usize])
                .collect::<Vec<_>>()
        );

        release_nemotron_device_route_pack(sorted).expect("release sorted route");
        release_nemotron_device_route_pack(route).expect("release route");
        assert!(
            crate::cuda_inference::cuda::release_device_tensor(logits_id).expect("release logits")
        );
    }
}
