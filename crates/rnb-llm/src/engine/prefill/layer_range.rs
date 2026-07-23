//! CPU prefill layer-range execution paths.

use super::*;
#[cfg(feature = "cuda")]
use crate::engine::backend_runtime;
use std::ops::Range;
use std::time::Instant;

pub(in crate::engine) fn run_prefill_layers_cpu_range(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    run_prefill_layers_cpu_range_impl(
        kv_cache,
        metadata,
        architecture,
        weights,
        gemma_per_layer_base,
        hidden,
        layer_range,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        true,
        None,
    )
    .and_then(|hidden| hidden.into_host_for_layer(None, "range_end"))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn run_prefill_layers_cpu_range_mtp_resident_kv(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    run_prefill_layers_cpu_range_impl(
        kv_cache,
        metadata,
        architecture,
        weights,
        gemma_per_layer_base,
        hidden,
        layer_range,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        false,
        None,
    )
    .and_then(|hidden| hidden.into_host_for_layer(None, "range_end"))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn run_prefill_layers_cpu_range_carrier(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
) -> crate::error::Result<hidden_carrier::PrefillHidden> {
    run_prefill_layers_cpu_range_impl(
        kv_cache,
        metadata,
        architecture,
        weights,
        gemma_per_layer_base,
        hidden,
        layer_range,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        true,
        None,
    )
}

pub(in crate::engine) fn run_prefill_layers_cpu_range_collect_prefix_state(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
    prefix_collector: Option<&mut verify_window::GdnPrefixStateCollector>,
) -> crate::error::Result<Tensor> {
    run_prefill_layers_cpu_range_impl(
        kv_cache,
        metadata,
        architecture,
        weights,
        gemma_per_layer_base,
        hidden,
        layer_range,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        true,
        prefix_collector,
    )
    .and_then(|hidden| hidden.into_host_for_layer(None, "range_end"))
}

fn prefill_layer_kind_name(layer: &LayerType) -> &'static str {
    match layer {
        LayerType::Attention(_) => "attention",
        LayerType::GatedDeltaNet(_) => "gated_delta_net",
        LayerType::NemotronMamba2(_) => "nemotron_mamba2",
        LayerType::NemotronMoE(_) => "nemotron_moe",
    }
}

fn next_prefill_pair_layer_idx(
    layer_idx: usize,
    layer_range_end: usize,
    layer_count: usize,
) -> Option<usize> {
    layer_idx
        .checked_add(1)
        .filter(|&idx| idx < layer_range_end && idx < layer_count)
}

#[cfg(any(feature = "cuda", test))]
fn device_hidden_materialize_reason(next_layer_kind: Option<&'static str>) -> &'static str {
    match next_layer_kind {
        None => "range_end",
        Some("attention") => "attention_requires_host",
        Some("nemotron_moe") => "unsupported_router_quant",
        Some("nemotron_mamba2") => "feature_disabled",
        Some("gated_delta_net") => "feature_disabled",
        Some(_) => "feature_disabled",
    }
}

#[cfg(any(feature = "cuda", test))]
fn qwen_gdn_moe_output_device_enabled() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::engine::tuning_runtime::gdn_prefill_chain_moe_output_device_enabled()
    }
    #[cfg(not(feature = "cuda"))]
    {
        crate::engine::policy::env_string("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE").as_deref()
            == Some("1")
    }
}

#[cfg(any(feature = "cuda", test))]
fn device_carrier_accepts_layer_kind(
    architecture: ModelArchitecture,
    layer_kind: Option<&'static str>,
) -> bool {
    (architecture == ModelArchitecture::NemotronHMoE
        && matches!(layer_kind, Some("nemotron_moe" | "nemotron_mamba2")))
        || (architecture == ModelArchitecture::Qwen35MoE
            && qwen_gdn_moe_output_device_enabled()
            && matches!(layer_kind, Some("gated_delta_net" | "attention")))
}

#[cfg(feature = "cuda")]
fn replace_qwen_attention_device_kv(
    kv_cache: &mut KVCache,
    layer: &crate::engine::cuda_runtime::MtpDeviceVerifyAttentionMoeLayer<'_>,
    attention_kv: &crate::engine::cuda_runtime::MtpDeviceVerifyAttentionKvState,
    pos_start: usize,
    seq_len: usize,
    kv_dim: usize,
) -> crate::error::Result<()> {
    let expected_values = seq_len.checked_mul(kv_dim).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "Qwen attention device KV size overflow: seq_len={seq_len} kv_dim={kv_dim}"
        ))
    })?;
    let common_contract_valid = layer.prior_tokens == pos_start
        && attention_kv.layer_idx == layer.layer_index
        && attention_kv.window_tokens == seq_len
        && attention_kv.kv_rows == kv_dim;
    if attention_kv.device_resident {
        if !common_contract_valid
            || !attention_kv.k_bits.is_empty()
            || !attention_kv.v_bits.is_empty()
        {
            return Err(crate::error::LlmError::Forward(format!(
                "Qwen resident attention device KV contract mismatch: layer={} result_layer={} prior={} pos_start={} window={} result_window={} kv_dim={} result_kv_rows={} k_bits={} v_bits={}",
                layer.layer_index,
                attention_kv.layer_idx,
                layer.prior_tokens,
                pos_start,
                seq_len,
                attention_kv.window_tokens,
                kv_dim,
                attention_kv.kv_rows,
                attention_kv.k_bits.len(),
                attention_kv.v_bits.len(),
            )));
        }
        return Ok(());
    }
    if !common_contract_valid
        || attention_kv.k_bits.len() != expected_values
        || attention_kv.v_bits.len() != expected_values
    {
        return Err(crate::error::LlmError::Forward(format!(
            "Qwen attention device KV contract mismatch: layer={} result_layer={} prior={} pos_start={} window={} result_window={} kv_dim={} result_kv_rows={} k_bits={} v_bits={} expected_values={}",
            layer.layer_index,
            attention_kv.layer_idx,
            layer.prior_tokens,
            pos_start,
            seq_len,
            attention_kv.window_tokens,
            kv_dim,
            attention_kv.kv_rows,
            attention_kv.k_bits.len(),
            attention_kv.v_bits.len(),
            expected_values
        )));
    }
    kv_cache
        .replace_layer_f16_range_compacted(
            layer.layer_index,
            pos_start,
            seq_len,
            &attention_kv.k_bits,
            &attention_kv.v_bits,
        )
        .map_err(crate::error::LlmError::Forward)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn release_qwen_attention_device_intermediates(
    normalized: backend_runtime::NemotronDeviceLayerOutput,
    residual: backend_runtime::NemotronDeviceLayerOutput,
    keep_id: Option<crate::engine::cuda_runtime::DeviceTensorId>,
) -> crate::error::Result<()> {
    let mut errors = Vec::new();
    for (label, output) in [("normalized", normalized), ("residual", residual)] {
        if keep_id == Some(output.output_id) {
            continue;
        }
        match output.release() {
            Ok(true) => {}
            Ok(false) => errors.push(format!("{label} tensor was already missing")),
            Err(err) => errors.push(format!("{label} release failed: {err}")),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(crate::error::LlmError::Forward(errors.join("; ")))
    }
}

#[cfg(feature = "cuda")]
fn trace_cuda_device_hidden_hash(
    stage: &str,
    producer_layer_idx: usize,
    consumer_layer_idx: Option<usize>,
    output: &backend_runtime::NemotronDeviceLayerOutput,
) -> crate::error::Result<()> {
    if crate::engine::policy::env_os_string("RNB_DEBUG_HIDDEN_HASH_TRACE").is_none() {
        return Ok(());
    }
    if let Some(filter) = crate::engine::policy::env_string("RNB_DEBUG_HIDDEN_HASH_LAYER") {
        let matches_consumer = consumer_layer_idx
            .map(|idx| debug_layer_matches_spec(&filter, idx))
            .unwrap_or(false);
        if !matches_consumer && !debug_layer_matches_spec(&filter, producer_layer_idx) {
            return Ok(());
        }
    }
    let values = backend_runtime::download_cuda_device_tensor_f32(output.output_id)?;
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &value in &values {
        if value.is_nan() {
            nan_count += 1;
        }
        if value.is_finite() {
            finite_count += 1;
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
    }
    if finite_count == 0 {
        min_value = f32::NAN;
        max_value = f32::NAN;
    }
    let consumer = consumer_layer_idx
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "[hidden-hash] stage={} producer={} consumer={} rows={} cols={} len={} finite={} nan={} min={:.6e} max={:.6e} hash=0x{:016x} first_bits=0x{:08x} last_bits=0x{:08x}",
        stage,
        producer_layer_idx,
        consumer,
        output.output_desc.rows(),
        output.output_desc.cols(),
        values.len(),
        finite_count,
        nan_count,
        min_value,
        max_value,
        hash_f32_bits(&values),
        values.first().copied().unwrap_or(0.0).to_bits(),
        values.last().copied().unwrap_or(0.0).to_bits()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn trace_host_hidden_hash(stage: &str, layer_idx: usize, values: &[f32]) {
    if crate::engine::policy::env_os_string("RNB_DEBUG_HIDDEN_HASH_TRACE").is_none() {
        return;
    }
    if let Some(filter) = crate::engine::policy::env_string("RNB_DEBUG_HIDDEN_HASH_LAYER") {
        if !debug_layer_matches_spec(&filter, layer_idx) {
            return;
        }
    }
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        }
        if value.is_finite() {
            finite_count += 1;
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
    }
    if finite_count == 0 {
        min_value = f32::NAN;
        max_value = f32::NAN;
    }
    eprintln!(
        "[hidden-hash] stage={} layer={} len={} finite={} nan={} min={:.6e} max={:.6e} hash=0x{:016x} first_bits=0x{:08x} last_bits=0x{:08x}",
        stage,
        layer_idx,
        values.len(),
        finite_count,
        nan_count,
        min_value,
        max_value,
        hash_f32_bits(values),
        values.first().copied().unwrap_or(0.0).to_bits(),
        values.last().copied().unwrap_or(0.0).to_bits()
    );
}

#[cfg(feature = "cuda")]
fn hash_f32_bits(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        hash ^= value.to_bits() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(feature = "cuda")]
fn debug_layer_matches_spec(raw: &str, layer_idx: usize) -> bool {
    for term in raw
        .split(',')
        .map(str::trim)
        .filter(|term| !term.is_empty())
    {
        if let Some((start, end)) = term.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                if start <= layer_idx && layer_idx <= end {
                    return true;
                }
            }
        } else if term.parse::<usize>().is_ok_and(|want| want == layer_idx) {
            return true;
        }
    }
    false
}

#[cfg(feature = "cuda")]
fn gemma4_partial_ple_host_carrier_requested() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_GEMMA_PARTIAL_PLE_CARRIER")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemma4_partial_ple_host_carrier_pending(
    architecture: ModelArchitecture,
    metadata: &ModelMetadata,
    w: &AttentionLayerWeights,
    layer_idx: usize,
    ple_base: Option<&GemmaPerLayerBase>,
    gemma: Option<&GemmaPerLayerWeights>,
    gemma4_ple_fused: bool,
    gemma4_output_scale_fused: bool,
) -> bool {
    gemma4_partial_ple_host_carrier_requested()
        && matches!(architecture, ModelArchitecture::Gemma4)
        && !gemma4_ple_fused
        && !gemma4_output_scale_fused
        && !gemma_ple_before_layer()
        && !gemma_ple_after_out_scale()
        && !gemma_ple_use_layer_input()
        && !gemma_ple_pre_norm_input()
        && gemma_ple_layer_enabled(layer_idx)
        && ple_base.is_some()
        && gemma.is_some()
        && !gemma_ple_global_only(metadata, w)
        && !super::policy::gemma_ple_global_only_enabled()
}

#[cfg(feature = "cuda")]
fn gemma4_partial_ple_release_error(
    output: backend_runtime::NemotronDeviceLayerOutput,
    message: String,
) -> crate::error::LlmError {
    match output.release() {
        Ok(true) => crate::error::LlmError::Forward(message),
        Ok(false) => crate::error::LlmError::Forward(format!(
            "{message}; CUDA Gemma partial PLE device output cleanup failed: tensor was already missing"
        )),
        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
            "{message}; CUDA Gemma partial PLE device output cleanup failed: {cleanup_err}"
        )),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn apply_gemma4_partial_ple_host_carrier(
    device_output: backend_runtime::NemotronDeviceLayerOutput,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    w: &AttentionLayerWeights,
    base: &GemmaPerLayerBase,
    gemma: &GemmaPerLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<backend_runtime::NemotronDeviceLayerOutput> {
    let desc = device_output.output_desc;
    if desc.rows() != seq_len
        || desc.cols() != metadata.hidden_dim
        || desc.dtype() != crate::engine::cuda_runtime::ScalarType::F32
    {
        return Err(gemma4_partial_ple_release_error(
            device_output,
            format!(
                "CUDA Gemma partial PLE carrier shape mismatch: got {}x{} {:?}, expected {}x{} F32",
                desc.rows(),
                desc.cols(),
                desc.dtype(),
                seq_len,
                metadata.hidden_dim
            ),
        ));
    }
    let bytes = match desc.byte_len() {
        Some(bytes) => bytes,
        None => {
            return Err(gemma4_partial_ple_release_error(
                device_output,
                "CUDA Gemma partial PLE carrier byte length overflow".to_string(),
            ));
        }
    };
    let materialized = hidden_carrier::PrefillHidden::Device(hidden_carrier::DevicePrefillHidden {
        output: device_output,
        producer_layer_idx: layer_idx,
    })
    .into_host_for_layer(layer_idx.checked_add(1), "gemma4_partial_ple_host")?;
    let ple_output = apply_gemma_per_layer_branch_with_output_scale(
        materialized,
        base,
        layer_idx,
        gemma,
        metadata,
        architecture,
        norm_eps,
        w.out_scale.as_ref(),
    )?;
    let mut hidden = ple_output.hidden;
    if !ple_output.output_scale_applied {
        hidden = apply_layer_output_scale(hidden, w.out_scale.as_ref(), layer_idx);
    }
    let values = kernels::tensor_as_f32_slice(&hidden);
    let h2d_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            crate::error::LlmError::Forward(
                "CUDA Gemma partial PLE reupload byte length overflow".to_string(),
            )
        })?;
    let uploaded =
        backend_runtime::upload_hidden_device_output_f32(values, seq_len, metadata.hidden_dim)?;
    if super::policy::cuda_device_prefill_trace_enabled() {
        let consumer = layer_idx
            .checked_add(1)
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "none".to_string());
        eprintln!(
            "[cuda:device-prefill-chain] op=gemma4_partial_ple_reupload layers={},{} d2h_bytes={} h2d_bytes={} reason=host_ple",
            layer_idx, consumer, bytes, h2d_bytes
        );
    }
    trace_cuda_device_hidden_hash(
        "gemma-partial-ple-reupload",
        layer_idx,
        layer_idx.checked_add(1),
        &uploaded,
    )?;
    Ok(uploaded)
}

#[cfg(feature = "cuda")]
fn nemotron_prefill_workspace_reserve_bytes(total_vram_bytes: usize) -> usize {
    total_vram_bytes / 10
}

#[cfg(any(feature = "cuda", test))]
fn nemotron_prefill_v2_workspace_decision_label(
    decision: crate::engine::workspace_runtime::NemotronPrefillWorkspaceDecision,
) -> &'static str {
    match decision {
        crate::engine::workspace_runtime::NemotronPrefillWorkspaceDecision::Fits => "fits",
        crate::engine::workspace_runtime::NemotronPrefillWorkspaceDecision::Chunk { .. } => "chunk",
        crate::engine::workspace_runtime::NemotronPrefillWorkspaceDecision::Fallback => "fallback",
    }
}

#[cfg(any(feature = "cuda", test))]
fn nemotron_prefill_v2_workspace_trace_line(
    plan: &crate::engine::workspace_runtime::NemotronPrefillWorkspacePlan,
) -> String {
    format!(
        "[cuda:device-prefill-v2] op=workspace_plan seq_len={} chunk_len={} hidden_dim={} n_expert={} expert_used={} n_ff={} shared_ff={} free_bytes={} total_bytes={} reserve_bytes={} usable_bytes={} required_bytes={} route_slots={} hidden_bytes={} persistent_hidden_bytes={} normalized_bytes={} router_logits_bytes={} route_bytes={} moe_intermediate_bytes={} mamba_state_sync_bytes={} attention_handoff_bytes={} decision={}",
        plan.seq_len,
        plan.chunk_len,
        plan.hidden_dim,
        plan.n_expert,
        plan.expert_used,
        plan.n_ff,
        plan.shared_ff,
        plan.free_vram_bytes,
        plan.total_vram_bytes,
        plan.reserve_bytes,
        plan.usable_vram_bytes,
        plan.required_workspace_bytes,
        plan.route_slots,
        plan.hidden_bytes,
        plan.persistent_hidden_bytes,
        plan.normalized_bytes,
        plan.router_logits_bytes,
        plan.route_bytes,
        plan.moe_intermediate_bytes,
        plan.mamba_state_sync_bytes,
        plan.attention_handoff_bytes,
        nemotron_prefill_v2_workspace_decision_label(plan.decision)
    )
}

#[cfg(any(feature = "cuda", test))]
fn nemotron_prefill_v2_workspace_lifecycle_trace_line(
    op: &str,
    active: bool,
    arena_bytes: usize,
    live_leases: usize,
    hit_bytes: usize,
    miss_bytes: usize,
    owned_alloc_count: usize,
) -> String {
    format!(
        "[cuda:device-prefill-v2] op={} active={} arena_bytes={} live_leases={} hit_bytes={} miss_bytes={} owned_alloc_count={}",
        op, active, arena_bytes, live_leases, hit_bytes, miss_bytes, owned_alloc_count
    )
}

#[cfg(feature = "cuda")]
fn nemotron_prefill_v2_trace_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn nemotron_prefill_workspace_shape(
    metadata: &ModelMetadata,
    weights: &ModelWeights,
) -> Option<(usize, usize, usize, usize)> {
    weights.layers.iter().find_map(|layer| {
        let LayerType::NemotronMoE(w) = layer else {
            return None;
        };
        let n_expert = w.router.rows.max(metadata.expert_used_count).max(1);
        let expert_used = metadata.expert_used_count.max(1).min(n_expert);
        let n_ff = (w.expert_up.rows / n_expert).max(1);
        let shared_ff = w.shared_expert_up.rows.max(1);
        Some((n_expert, expert_used, n_ff, shared_ff))
    })
}

#[cfg(feature = "cuda")]
fn maybe_plan_nemotron_prefill_v2_workspace(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    seq_len: usize,
) -> Option<crate::engine::workspace_runtime::NemotronPrefillWorkspacePlan> {
    if architecture != ModelArchitecture::NemotronHMoE
        || !super::policy::cuda_nemotron_device_prefill_v2_enabled()
    {
        return None;
    }

    if !super::policy::cuda_nemotron_prefill_workspace_enabled() {
        if super::policy::cuda_device_prefill_trace_enabled() {
            eprintln!("[cuda:device-prefill-v2] op=workspace_plan decision=disabled reason=policy");
        }
        return None;
    }

    let Some((n_expert, expert_used, n_ff, shared_ff)) =
        nemotron_prefill_workspace_shape(metadata, weights)
    else {
        if super::policy::cuda_device_prefill_trace_enabled() {
            eprintln!(
                "[cuda:device-prefill-v2] op=workspace_plan decision=fallback reason=no_nemotron_moe_shape"
            );
        }
        return None;
    };

    let memory = match crate::engine::cuda_runtime::cuda_memory_info() {
        Ok(memory) => memory,
        Err(err) => {
            if super::policy::cuda_device_prefill_trace_enabled() {
                eprintln!(
                    "[cuda:device-prefill-v2] op=workspace_plan decision=fallback reason=backend_unavailable error={}",
                    nemotron_prefill_v2_trace_value(&err)
                );
            }
            return None;
        }
    };
    let reserve_bytes = nemotron_prefill_workspace_reserve_bytes(memory.total_bytes);
    let request = crate::engine::workspace_runtime::NemotronPrefillWorkspaceRequest {
        seq_len,
        hidden_dim: metadata.hidden_dim,
        n_expert,
        expert_used,
        n_ff,
        shared_ff,
        free_vram_bytes: memory.free_bytes,
        total_vram_bytes: memory.total_bytes,
        reserve_bytes,
    };

    match crate::engine::workspace_runtime::plan_nemotron_prefill_workspace(request) {
        Ok(plan) => {
            if super::policy::cuda_device_prefill_trace_enabled() {
                eprintln!("{}", nemotron_prefill_v2_workspace_trace_line(&plan));
            }
            Some(plan)
        }
        Err(err) => {
            if super::policy::cuda_device_prefill_trace_enabled() {
                eprintln!(
                    "[cuda:device-prefill-v2] op=workspace_plan decision=fallback reason={}",
                    nemotron_prefill_v2_trace_value(&err)
                );
            }
            None
        }
    }
}

#[cfg(feature = "cuda")]
enum NemotronMoeMamba2DevicePairAttempt {
    Done(backend_runtime::NemotronDeviceLayerOutput),
    MambaUnavailable,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_run_nemotron_moe_mamba2_device_pair(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: &Tensor,
    weights: &ModelWeights,
    moe_w: &models::nemotron::moe::NemotronMoELayerWeights,
    layer_idx: usize,
    next_layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<NemotronMoeMamba2DevicePairAttempt> {
    let Some(LayerType::NemotronMamba2(mamba_w)) = weights.layers.get(next_layer_idx) else {
        return Ok(NemotronMoeMamba2DevicePairAttempt::MambaUnavailable);
    };
    let uploaded = backend_runtime::upload_hidden_device_output_f32(
        kernels::tensor_as_f32_slice(hidden),
        seq_len,
        metadata.hidden_dim,
    )?;
    let device_hidden = hidden_carrier::DevicePrefillHidden {
        output: uploaded,
        producer_layer_idx: layer_idx.saturating_sub(1),
    };
    let moe_output = match models::nemotron::moe::forward_moe_layer_from_device_for_chain(
        metadata,
        device_hidden,
        moe_w,
        norm_eps,
        layer_idx,
        layer_idx.saturating_sub(1),
    )? {
        models::nemotron::moe::NemotronMamba2ToMoeDeviceAttempt::Done { output, .. } => output,
        models::nemotron::moe::NemotronMamba2ToMoeDeviceAttempt::Materialize {
            device_hidden,
            reason,
        } => {
            let _ = device_hidden.output.release()?;
            return Err(crate::error::LlmError::Forward(format!(
                "CUDA Nemotron MoE execution is unavailable for layer {layer_idx}: {reason}; CPU fallback is disabled"
            )));
        }
    };
    match models::nemotron::mamba::forward_mamba2_layer_from_device(
        kv_cache,
        metadata,
        moe_output,
        mamba_w,
        next_layer_idx,
        seq_len,
        norm_eps,
    )? {
        models::nemotron::mamba::NemotronMamba2DeviceAttempt::Done(mamba_output, trace) => {
            if models::nemotron::mamba::device_prefill_trace_enabled() {
                eprintln!(
                    "[cuda:device-prefill-chain] op=nemotron_moe_mamba2_boundary layers={},{} moe_output_device=1 mamba2_input_device=1 boundary_d2h_bytes={} mamba2_output_device=1 mamba2_hidden_d2h_bytes={} mamba2_state_d2h_bytes={} reason=device_mamba2",
                    layer_idx,
                    next_layer_idx,
                    trace.boundary_d2h_bytes,
                    trace.hidden_d2h_bytes,
                    trace.mamba2_state_d2h_bytes()
                );
            }
            Ok(NemotronMoeMamba2DevicePairAttempt::Done(mamba_output))
        }
        models::nemotron::mamba::NemotronMamba2DeviceAttempt::Fallback(moe_output) => {
            let _ = moe_output.release()?;
            Err(crate::error::LlmError::Forward(format!(
                "CUDA Mamba2 execution is unavailable for layer {next_layer_idx}; CPU fallback is disabled"
            )))
        }
    }
}

fn observe_prefill_layer_output(
    hidden: &Tensor,
    metadata: &ModelMetadata,
    seq_len: usize,
    layer_idx: usize,
    layer_start: Option<Instant>,
    profiler: &mut super::profile::PrefillLayerProfiler,
) {
    let hidden_data = kernels::tensor_as_f32_slice(hidden);
    #[cfg(feature = "cuda")]
    trace_host_hidden_hash("host-layer-output", layer_idx, hidden_data);
    if dump_bin_dir().is_some() {
        dump_bin("prefill", layer_idx, "layer_out", hidden_data);
    }
    let last_row = &hidden_data[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
    emit_layer_trace("prefill", layer_idx, last_row);
    if let Some(layer_start) = layer_start {
        profiler.record(layer_idx, layer_start.elapsed().as_secs_f64() * 1000.0);
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_chain_enabled() -> bool {
    crate::engine::policy::env_string("RNB_METAL_QWEN_PREFILL_CHAIN")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_chain_diag_layers() -> Option<usize> {
    crate::engine::policy::env_string("RNB_METAL_QWEN_PREFILL_CHAIN_DIAG_LAYERS")
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_chain_trace_enabled() -> bool {
    crate::engine::policy::env_string("RNB_METAL_QWEN_PREFILL_CHAIN_TRACE").is_some_and(|value| {
        !matches!(
            value.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_chain_fallback(
    reason: &'static str,
    seq_len: usize,
    layer_range: &Range<usize>,
    layer_idx: Option<usize>,
) -> crate::error::Result<Option<Tensor>> {
    if metal_qwen_prefill_chain_trace_enabled() {
        let layer = layer_idx
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "none".to_string());
        eprintln!(
            "[metal:qwen-prefill-chain] eligible_hit=0 stage=llm reason={reason} seq_len={seq_len} layers={}..{} layer={layer}",
            layer_range.start, layer_range.end
        );
    }
    Ok(None)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_route_sort_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_ROUTE_SORT")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_moe_trace_requirement(active: bool) -> Option<&'static str> {
    active.then_some("moe_trace_active")
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_host_requirement(
    weights: &ModelWeights,
    layer_range: &Range<usize>,
    profiler_enabled: bool,
) -> Option<&'static str> {
    if profiler_enabled {
        return Some("profiler");
    }
    if crate::engine::policy::profiling_enabled() {
        return Some("profile");
    }
    if crate::engine::moe_profile::is_enabled() {
        return Some("moe_profile");
    }
    if dump_bin_dir().is_some() {
        return Some("binary_dump");
    }
    if layer_trace_enabled() {
        return Some("layer_trace");
    }
    if attn_trace_enabled() {
        return Some("attention_trace");
    }
    if let Some(reason) =
        metal_qwen_prefill_moe_trace_requirement(crate::engine::moe_trace::is_active())
    {
        return Some(reason);
    }
    if crate::engine::moe_trace::route_trace_is_active() {
        return Some("moe_route_trace");
    }
    if crate::engine::moe_trace::predictor_trace_is_active() {
        return Some("moe_predictor_trace");
    }
    if crate::engine::models::shared_expert_moe::jit_request::qwen35_moe_jit_load_requested() {
        return Some("moe_jit_load");
    }
    if crate::engine::policy::env_os_string("RNB_DEBUG_HIDDEN_HASH_TRACE").is_some() {
        return Some("hidden_hash_trace");
    }
    if crate::engine::policy::env_os_string("RNB_PREFILL_NAN_TRACE").is_some() {
        return Some("nan_trace");
    }
    if crate::engine::policy::env_os_string("RNB_DEBUG_GDN_STAGE_DUMP_DIR").is_some() {
        return Some("gdn_stage_dump");
    }
    if crate::engine::policy::env_string("RNB_VULKAN_LAYER_TRACE").is_some_and(|value| value != "0")
    {
        return Some("vulkan_layer_trace");
    }
    for layer_idx in layer_range.clone() {
        if targeted_attn_trace_enabled(layer_idx) {
            return Some("targeted_attention_trace");
        }
        if weights.layers.get(layer_idx).is_some_and(|layer| {
            matches!(layer, LayerType::GatedDeltaNet(_))
                && models::qwen::debug_gdn_stage_trace_enabled(layer_idx)
        }) {
            return Some("gdn_stage_trace");
        }
    }
    None
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_quant_weight_ref(
    weight: &QuantizedWeight,
) -> Option<crate::engine::metal_runtime::MetalQuantWeightRef<'_>> {
    Some(crate::engine::metal_runtime::MetalQuantWeightRef {
        ggml_type: weight.ggml_type,
        raw: weight.data.as_bytes()?,
        rows: weight.rows,
        cols: weight.cols,
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_attention_spec<'a>(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    w: &'a AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
) -> Option<crate::engine::metal_runtime::MetalPrefillAtnOTailSpec<'a>> {
    if use_gemma_block_semantics(architecture)
        || w.v_proj_missing
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || w.moe.is_some()
        || w.shared_expert_moe.is_none()
        || w.ffn_gate_up_fused.is_some()
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || shared_kv_source_layer(metadata, architecture, layer_idx).is_some()
    {
        return None;
    }
    let layer_kv_override = metadata
        .head_count_kv_per_layer
        .as_ref()
        .and_then(|values| values.get(layer_idx).copied());
    let layout = resolve_attention_layout(metadata, w, layer_kv_override).ok()?;
    if !layout.has_gated_attn || layout.head_dim != 256 {
        return None;
    }
    let q_norm = w.q_norm.as_ref()?;
    let k_norm = w.k_norm.as_ref()?;
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim == 0 || rope_dim >= layout.head_dim {
        return None;
    }
    if qwen_text_mrope_dim(metadata, architecture, rope_dim, layout.head_dim).is_some()
        || gemma_rope_freq_factors(
            rope_freqs,
            metadata,
            architecture,
            layer_idx,
            layout.head_dim,
        )
        .is_some()
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
    {
        return None;
    }
    Some(crate::engine::metal_runtime::MetalPrefillAtnOTailSpec {
        core: crate::engine::metal_runtime::MetalPrefillAtnCoreSpec {
            attn_norm_w: kernels::tensor_as_f32_slice(&w.attn_norm),
            q_norm_w: kernels::tensor_as_f32_slice(q_norm),
            k_norm_w: kernels::tensor_as_f32_slice(k_norm),
            q_weight: metal_qwen_quant_weight_ref(&w.q_weight)?,
            k_weight: metal_qwen_quant_weight_ref(&w.k_weight)?,
            v_weight: metal_qwen_quant_weight_ref(&w.v_weight)?,
            seq_len,
            num_heads: layout.num_heads,
            num_kv_heads: layout.num_kv_heads,
            head_dim: layout.head_dim,
            hidden_dim: metadata.hidden_dim,
            q_dim: layout.q_dim,
            kv_dim: layout.kv_dim,
            n_rot: rope_dim,
            rope_theta,
            scale: resolve_attention_scale(metadata, architecture),
            norm_eps,
            pos_start,
        },
        o_weight: metal_qwen_quant_weight_ref(&w.o_weight)?,
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_moe_product_tuple(
    n_expert_used: usize,
    sparse_gate: rnb_loader::GGMLType,
    sparse_up: rnb_loader::GGMLType,
    sparse_down: rnb_loader::GGMLType,
    shared_gate: rnb_loader::GGMLType,
    shared_up: rnb_loader::GGMLType,
    shared_down: rnb_loader::GGMLType,
) -> bool {
    let sparse = sparse_gate == rnb_loader::GGMLType::Q4_K
        && sparse_up == rnb_loader::GGMLType::Q4_K
        && matches!(
            sparse_down,
            rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q5_K | rnb_loader::GGMLType::Q6_K
        );
    let shared_q4 = shared_gate == rnb_loader::GGMLType::Q4_K
        && shared_up == rnb_loader::GGMLType::Q4_K
        && matches!(
            shared_down,
            rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K
        );
    n_expert_used <= 8 && sparse && shared_q4
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_moe_weights<'a>(
    ffn_norm_w: &'a Tensor,
    moe_w: &'a SharedExpertMoELayerWeights,
    norm_eps: f32,
) -> Option<crate::engine::metal_runtime::MetalQwenMoePrefillWeights<'a>> {
    let product_tuple = metal_qwen_prefill_moe_product_tuple(
        moe_w.n_expert_used,
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
        moe_w.shared_gate.ggml_type,
        moe_w.shared_up.ggml_type,
        moe_w.shared_down.ggml_type,
    );
    if !norm_eps.is_finite()
        || norm_eps <= 0.0
        || !product_tuple
        || !matches!(moe_w.expert_gating_func, 0 | 1)
        || !moe_w.shared_expert_gated
    {
        return None;
    }
    let expert_bytes = crate::engine::models::shared_expert_moe::moe_types::sparse_expert_bytes(
        moe_w.n_embd,
        moe_w.n_ff,
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
    )?;
    Some(crate::engine::metal_runtime::MetalQwenMoePrefillWeights {
        ffn_norm_w: kernels::tensor_as_f32_slice(ffn_norm_w),
        norm_eps,
        router_w: moe_w.router_f32()?,
        gate_all: moe_w.gate_exps_bytes()?,
        up_all: moe_w.up_exps_bytes()?,
        down_all: moe_w.down_exps_bytes()?,
        gate_expert_bytes: expert_bytes.gate,
        up_expert_bytes: expert_bytes.up,
        down_expert_bytes: expert_bytes.down,
        shared_input_scale: kernels::tensor_as_f32_slice(&moe_w.shared_input_scale),
        shared_gate: moe_w.shared_gate.data.as_bytes()?,
        shared_up: moe_w.shared_up.data.as_bytes()?,
        shared_down: moe_w.shared_down.data.as_bytes()?,
        sparse_quant: crate::engine::metal_runtime::MetalGgmlQuantSet {
            gate: moe_w.gate_quant,
            up: moe_w.up_quant,
            down: moe_w.down_quant,
        },
        shared_quant: crate::engine::metal_runtime::MetalGgmlQuantSet {
            gate: moe_w.shared_gate.ggml_type,
            up: moe_w.shared_up.ggml_type,
            down: moe_w.shared_down.ggml_type,
        },
        route: crate::engine::metal_runtime::Qwen35RoutePolicy {
            algorithm:
                crate::engine::metal_runtime::Qwen35RouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
            n_expert: moe_w.n_expert,
            n_expert_used: moe_w.n_expert_used,
        },
        hidden_dim: moe_w.n_embd,
        ffn_dim: moe_w.n_ff,
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn apply_metal_qwen_prefill_chain_output(
    kv_cache: &mut KVCache,
    weights: &ModelWeights,
    layer_range: &Range<usize>,
    seq_len: usize,
    pos_start: usize,
    hidden_dim: usize,
    output: crate::engine::metal_runtime::MetalQwenPrefillChainOut,
) -> crate::error::Result<Tensor> {
    let requested_layers = layer_range.end.saturating_sub(layer_range.start);
    let mut expected_attention_layers = 0usize;
    let mut expected_gdn_layers = 0usize;
    for layer_idx in layer_range.clone() {
        match weights.layers.get(layer_idx) {
            Some(LayerType::Attention(_)) => expected_attention_layers += 1,
            Some(LayerType::GatedDeltaNet(_)) => expected_gdn_layers += 1,
            _ => {
                return Err(crate::error::LlmError::Forward(format!(
                    "Metal Qwen prefill output references unsupported requested layer {layer_idx}"
                )));
            }
        }
    }
    let ownership = hidden_carrier::validate_metal_qwen_prefill_ownership(
        requested_layers,
        expected_attention_layers,
        expected_gdn_layers,
        output.attention_kv.len(),
        output.gdn_states.len(),
        output.hidden_uploads,
        output.hidden_readbacks,
        output.intermediate_hidden_transfers,
    )?;
    debug_assert_eq!(ownership.requested_layers, requested_layers);
    debug_assert_eq!(ownership.hidden_uploads, 1);
    debug_assert_eq!(ownership.hidden_readbacks, 1);
    let expected_hidden_len = seq_len.checked_mul(hidden_dim).ok_or_else(|| {
        crate::error::LlmError::Forward(
            "Metal Qwen prefill output hidden length overflow".to_string(),
        )
    })?;
    if output.hidden.len() != expected_hidden_len {
        return Err(crate::error::LlmError::Forward(format!(
            "Metal Qwen prefill output hidden length {} != {expected_hidden_len}",
            output.hidden.len()
        )));
    }

    let mut seen_attention = vec![false; requested_layers];
    for (layer_idx, k_bits, v_bits) in &output.attention_kv {
        let Some(offset) = layer_idx
            .checked_sub(layer_range.start)
            .filter(|&offset| offset < requested_layers)
        else {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill attention KV key {layer_idx} is outside requested range {}..{}",
                layer_range.start, layer_range.end
            )));
        };
        if seen_attention[offset]
            || !matches!(
                weights.layers.get(*layer_idx),
                Some(LayerType::Attention(_))
            )
        {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill attention KV key {layer_idx} is duplicate or not attention"
            )));
        }
        seen_attention[offset] = true;
        let expected_kv_len = seq_len
            .checked_mul(kv_cache.layer_kv_dim(*layer_idx))
            .ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "Metal Qwen prefill attention KV length overflow at layer {layer_idx}"
                ))
            })?;
        if k_bits.len() != expected_kv_len || v_bits.len() != expected_kv_len {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill attention KV length mismatch at layer {layer_idx}: k={} v={} expected={expected_kv_len}",
                k_bits.len(),
                v_bits.len()
            )));
        }
    }
    let mut seen_gdn = vec![false; requested_layers];
    for (layer_idx, conv_state, delta_state) in &output.gdn_states {
        let Some(offset) = layer_idx
            .checked_sub(layer_range.start)
            .filter(|&offset| offset < requested_layers)
        else {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill GDN state key {layer_idx} is outside requested range {}..{}",
                layer_range.start, layer_range.end
            )));
        };
        let Some(state) = kv_cache.get_ssm_state(*layer_idx) else {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill GDN state key {layer_idx} has no initialized destination"
            )));
        };
        if seen_gdn[offset]
            || !matches!(
                weights.layers.get(*layer_idx),
                Some(LayerType::GatedDeltaNet(_))
            )
            || conv_state.len() != state.conv_state.len()
            || delta_state.len() != state.delta_state.len()
        {
            return Err(crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill GDN state mismatch at layer {layer_idx}: conv={} expected_conv={} delta={} expected_delta={}",
                conv_state.len(),
                state.conv_state.len(),
                delta_state.len(),
                state.delta_state.len()
            )));
        }
        seen_gdn[offset] = true;
    }

    for (layer_idx, k_bits, v_bits) in &output.attention_kv {
        kv_cache.append_bits_range(*layer_idx, pos_start, seq_len, k_bits, v_bits);
    }
    for (layer_idx, conv_state, delta_state) in &output.gdn_states {
        let state = kv_cache.get_ssm_state_mut(*layer_idx).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "Metal Qwen prefill GDN state destination disappeared at layer {layer_idx}"
            ))
        })?;
        state.conv_state.copy_from_slice(conv_state);
        state.delta_state.copy_from_slice(delta_state);
    }
    hidden_carrier::record_metal_qwen_prefill_ownership(ownership);
    Ok(Tensor::from_vec(output.hidden, &[seq_len, hidden_dim]))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
fn try_run_metal_qwen_prefill_chain(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    hidden: &Tensor,
    layer_range: &Range<usize>,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    profiler_enabled: bool,
) -> crate::error::Result<Option<Tensor>> {
    if !metal_qwen_prefill_chain_enabled() {
        return metal_qwen_prefill_chain_fallback("disabled", seq_len, layer_range, None);
    }
    if architecture != ModelArchitecture::Qwen35MoE {
        return metal_qwen_prefill_chain_fallback(
            "unsupported_architecture",
            seq_len,
            layer_range,
            None,
        );
    }
    if layer_range.start >= layer_range.end || layer_range.end > weights.layers.len() {
        return metal_qwen_prefill_chain_fallback(
            "invalid_layer_range",
            seq_len,
            layer_range,
            None,
        );
    }
    if pos_start != 0 {
        return metal_qwen_prefill_chain_fallback("nonzero_pos_start", seq_len, layer_range, None);
    }
    if !metal_qwen_route_sort_enabled() {
        return metal_qwen_prefill_chain_fallback(
            "route_sort_disabled",
            seq_len,
            layer_range,
            None,
        );
    }
    if let Some(reason) =
        metal_qwen_prefill_host_requirement(weights, layer_range, profiler_enabled)
    {
        return metal_qwen_prefill_chain_fallback(reason, seq_len, layer_range, None);
    }
    let Some(expected_hidden_len) = seq_len.checked_mul(metadata.hidden_dim) else {
        return metal_qwen_prefill_chain_fallback(
            "hidden_length_overflow",
            seq_len,
            layer_range,
            None,
        );
    };
    if kernels::tensor_as_f32_slice(hidden).len() != expected_hidden_len {
        return metal_qwen_prefill_chain_fallback(
            "hidden_length_mismatch",
            seq_len,
            layer_range,
            None,
        );
    }

    let output = {
        let mut layers = Vec::with_capacity(layer_range.end - layer_range.start);
        for layer_idx in layer_range.clone() {
            match &weights.layers[layer_idx] {
                LayerType::Attention(w) => {
                    let Some(moe_w) = w.shared_expert_moe.as_ref() else {
                        return metal_qwen_prefill_chain_fallback(
                            "attention_missing_moe",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    let Some(core) = metal_qwen_prefill_attention_spec(
                        metadata,
                        architecture,
                        w,
                        weights.rope_freqs.as_ref(),
                        layer_idx,
                        seq_len,
                        pos_start,
                        norm_eps,
                    ) else {
                        return metal_qwen_prefill_chain_fallback(
                            "unsupported_attention_spec",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    let Some(moe) = metal_qwen_prefill_moe_weights(
                        select_ffn_pre_norm_weight(w, architecture),
                        moe_w,
                        norm_eps,
                    ) else {
                        return metal_qwen_prefill_chain_fallback(
                            "unsupported_attention_moe",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    layers.push(
                        crate::engine::metal_runtime::MetalQwenPrefillChainLayer::Attention {
                            layer_idx,
                            core,
                            moe,
                        },
                    );
                }
                LayerType::GatedDeltaNet(w) => {
                    if w.ffn_gate_up_fused.is_some() {
                        return metal_qwen_prefill_chain_fallback(
                            "gdn_fused_gate_up",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    }
                    let Some(moe_w) = w.shared_expert_moe.as_ref() else {
                        return metal_qwen_prefill_chain_fallback(
                            "gdn_missing_moe",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    let Some(state) = kv_cache.get_ssm_state(layer_idx) else {
                        return metal_qwen_prefill_chain_fallback(
                            "gdn_missing_state",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    let Some(layer) = models::qwen::metal_qwen_prefill_gdn_spec(
                        w,
                        state,
                        seq_len,
                        metadata.hidden_dim,
                        metadata.ssm_d_inner,
                        metadata.ssm_d_state,
                        metadata.ssm_n_group,
                        metadata.ssm_dt_rank,
                        metadata.ssm_conv_kernel,
                        norm_eps,
                    ) else {
                        return metal_qwen_prefill_chain_fallback(
                            "unsupported_gdn_spec",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    let Some(moe) =
                        metal_qwen_prefill_moe_weights(&w.post_attn_norm, moe_w, norm_eps)
                    else {
                        return metal_qwen_prefill_chain_fallback(
                            "unsupported_gdn_moe",
                            seq_len,
                            layer_range,
                            Some(layer_idx),
                        );
                    };
                    layers.push(
                        crate::engine::metal_runtime::MetalQwenPrefillChainLayer::Gdn {
                            layer_idx,
                            layer,
                            moe,
                        },
                    );
                }
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => {
                    return metal_qwen_prefill_chain_fallback(
                        "unsupported_layer_kind",
                        seq_len,
                        layer_range,
                        Some(layer_idx),
                    );
                }
            }
        }
        crate::engine::metal_runtime::metal_qwen_prefill_chain_run(
            kernels::tensor_as_f32_slice(hidden),
            &layers,
        )
        .map_err(crate::error::LlmError::Forward)?
    };
    let Some(output) = output else {
        return metal_qwen_prefill_chain_fallback(
            "runtime_or_backend_preflight_none",
            seq_len,
            layer_range,
            None,
        );
    };
    apply_metal_qwen_prefill_chain_output(
        kv_cache,
        weights,
        layer_range,
        seq_len,
        pos_start,
        metadata.hidden_dim,
        output,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_layers_cpu_range_impl(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
    mirror_attention_kv_to_host: bool,
    mut prefix_collector: Option<&mut verify_window::GdnPrefixStateCollector>,
) -> crate::error::Result<hidden_carrier::PrefillHidden> {
    #[cfg(feature = "cuda")]
    let nemotron_workspace_plan =
        maybe_plan_nemotron_prefill_v2_workspace(metadata, architecture, weights, seq_len);

    #[cfg(feature = "cuda")]
    let nemotron_workspace_active = if let Some(plan) = nemotron_workspace_plan.as_ref() {
        match crate::engine::cuda_runtime::begin_nemotron_prefill_workspace(plan) {
            Ok(summary) => {
                if super::policy::cuda_device_prefill_trace_enabled() {
                    eprintln!(
                        "{}",
                        nemotron_prefill_v2_workspace_lifecycle_trace_line(
                            "workspace_begin",
                            summary.active,
                            summary.arena_bytes,
                            summary.live_leases,
                            summary.hit_bytes,
                            summary.miss_bytes,
                            summary.owned_alloc_count,
                        )
                    );
                }
                summary.active
            }
            Err(err) => {
                if super::policy::cuda_device_prefill_trace_enabled() {
                    eprintln!(
                        "[cuda:device-prefill-v2] op=workspace_begin decision=fallback reason={}",
                        nemotron_prefill_v2_trace_value(&err)
                    );
                }
                false
            }
        }
    } else {
        false
    };

    #[cfg(feature = "cuda")]
    let qwen_attention_device_chain = if prefix_collector.is_none()
        && architecture == ModelArchitecture::Qwen35MoE
        && qwen_gdn_moe_output_device_enabled()
        && crate::engine::inference::qwen35_prefill_attention_moe_device_layers_supported(
            weights, seq_len,
        ) {
        let (base_rope_dim, device_rope_theta, proportional_rope) =
            resolve_rope_params(metadata, architecture, 0, head_dim);
        if proportional_rope {
            None
        } else {
            let qwen_mrope_dim =
                qwen_text_mrope_dim(metadata, architecture, base_rope_dim, head_dim);
            Some((
                crate::engine::inference::build_mtp_device_verify_attention_moe_layers(
                    weights,
                    metadata,
                    architecture,
                    kv_cache,
                )?,
                qwen_mrope_dim.unwrap_or(base_rope_dim),
                qwen_mrope_dim.is_some(),
                device_rope_theta,
            ))
        }
    } else {
        None
    };

    let mut profiler = super::profile::PrefillLayerProfiler::new(weights.layers.len());
    let mut hidden = hidden_carrier::PrefillHidden::Host(hidden);
    let mut layer_idx = layer_range.start;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if prefix_collector.is_none() && gemma_per_layer_base.is_none() {
        let chain_end = match metal_qwen_prefill_chain_diag_layers() {
            Some(0) => layer_range.start,
            Some(prefix_layers) => layer_range
                .start
                .saturating_add(prefix_layers)
                .min(layer_range.end),
            None => layer_range.end,
        };
        if chain_end > layer_range.start {
            let chain_range = layer_range.start..chain_end;
            let chain_input = hidden
                .as_host()
                .expect("Metal Qwen prefill chain input must be host-resident");
            if let Some(chain_hidden) = try_run_metal_qwen_prefill_chain(
                kv_cache,
                metadata,
                architecture,
                weights,
                chain_input,
                &chain_range,
                seq_len,
                pos_start,
                norm_eps,
                profiler.enabled(),
            )? {
                hidden = hidden_carrier::PrefillHidden::Host(chain_hidden);
                layer_idx = chain_end;
                if layer_idx == layer_range.end {
                    return Ok(hidden);
                }
            }
        }
    }
    while layer_idx < layer_range.end {
        if let Err(error) = crate::generate::check_generation_cancellation() {
            #[cfg(feature = "cuda")]
            if nemotron_workspace_active {
                let _ = crate::engine::cuda_runtime::end_nemotron_prefill_workspace();
            }
            return Err(error);
        }
        crate::engine::decode_attn_prewarm::maybe_spawn_prefill_attention_prewarm(
            weights, layer_idx,
        );
        #[cfg(feature = "cuda")]
        if matches!(hidden, hidden_carrier::PrefillHidden::Device(_)) {
            let current_layer_kind = weights.layers.get(layer_idx).map(prefill_layer_kind_name);
            if architecture == ModelArchitecture::Qwen35MoE && qwen_gdn_moe_output_device_enabled()
            {
                if !device_carrier_accepts_layer_kind(architecture, current_layer_kind) {
                    let reason = device_hidden_materialize_reason(current_layer_kind);
                    hidden = hidden
                        .into_host_for_layer(Some(layer_idx), reason)
                        .map(hidden_carrier::PrefillHidden::Host)?;
                } else if let Some(LayerType::GatedDeltaNet(w)) = weights.layers.get(layer_idx) {
                    let device_hidden = hidden.take_device().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "expected CUDA device hidden carrier".to_string(),
                        )
                    })?;
                    match models::qwen::try_forward_gdn_layer_from_device_input(
                        kv_cache,
                        metadata,
                        &device_hidden.output,
                        w,
                        layer_idx,
                        seq_len,
                        norm_eps,
                    ) {
                        Ok(Some(output)) => {
                            match device_hidden.output.release() {
                                Ok(true) => {}
                                Ok(false) => {
                                    return Err(crate::error::LlmError::Forward(
                                        "CUDA Qwen device hidden tensor was already missing"
                                            .to_string(),
                                    ));
                                }
                                Err(err) => return Err(err),
                            }
                            hidden = hidden_carrier::PrefillHidden::Device(
                                hidden_carrier::DevicePrefillHidden {
                                    output,
                                    producer_layer_idx: layer_idx,
                                },
                            );
                            layer_idx += 1;
                            continue;
                        }
                        Ok(None) => {
                            hidden = hidden_carrier::PrefillHidden::Device(device_hidden)
                                .into_host_for_layer(Some(layer_idx), "feature_disabled")
                                .map(hidden_carrier::PrefillHidden::Host)?;
                        }
                        Err(err) => {
                            let cleanup = device_hidden.output.release();
                            return Err(match cleanup {
                                Ok(true) => err,
                                Ok(false) => crate::error::LlmError::Forward(format!(
                                    "{err}; CUDA Qwen device hidden cleanup failed: tensor was already missing"
                                )),
                                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                                    "{err}; CUDA Qwen device hidden cleanup failed: {cleanup_err}"
                                )),
                            });
                        }
                    }
                } else if let Some(LayerType::Attention(attention_w)) =
                    weights.layers.get(layer_idx)
                {
                    let Some(moe_w) = attention_w.shared_expert_moe.as_ref() else {
                        hidden = hidden
                            .into_host_for_layer(Some(layer_idx), "feature_disabled")
                            .map(hidden_carrier::PrefillHidden::Host)?;
                        continue;
                    };
                    let device_layer =
                        qwen_attention_device_chain
                            .as_ref()
                            .and_then(|(layers, _, _, _)| {
                                layers.iter().find(|layer| layer.layer_index == layer_idx)
                            });
                    let Some(device_layer) = device_layer else {
                        hidden = hidden
                            .into_host_for_layer(Some(layer_idx), "feature_disabled")
                            .map(hidden_carrier::PrefillHidden::Host)?;
                        continue;
                    };
                    let (_, device_rope_dim, rope_neox, device_rope_theta) =
                        qwen_attention_device_chain
                            .as_ref()
                            .expect("Qwen attention device layer requires chain configuration");
                    let device_hidden = hidden.take_device().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "expected CUDA device hidden carrier".to_string(),
                        )
                    })?;
                    let attention_output =
                        match backend_runtime::qwen35_prefill_attention_device_input(
                            &device_hidden.output,
                            device_layer,
                            seq_len,
                            metadata.hidden_dim,
                            *device_rope_dim,
                            *rope_neox,
                            *device_rope_theta,
                            pos_start,
                            norm_eps,
                            mirror_attention_kv_to_host,
                        ) {
                            Ok(output) => output,
                            Err(err) => {
                                let cleanup = device_hidden.output.release();
                                return Err(match cleanup {
                                    Ok(true) => err,
                                    Ok(false) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Qwen attention input cleanup failed: tensor was already missing"
                                    )),
                                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Qwen attention input cleanup failed: {cleanup_err}"
                                    )),
                                });
                            }
                        };
                    let attention_kv = attention_output.attention_kv;
                    let normalized = attention_output.normalized;
                    let residual = attention_output.residual;
                    let moe_output =
                        match models::shared_expert_moe::try_forward_ffn_qwen35moe_device_input_carrier(
                            moe_w,
                            seq_len,
                            metadata.hidden_dim,
                            normalized.output_id,
                            normalized.output_desc,
                            residual.output_id,
                            residual.output_desc,
                        ) {
                            Ok(Some(output)) => output,
                            Ok(None) => {
                                let intermediates_cleanup =
                                    release_qwen_attention_device_intermediates(
                                        normalized, residual, None,
                                    );
                                let input_cleanup = device_hidden.output.release();
                                return Err(crate::error::LlmError::Forward(format!(
                                    "CUDA Qwen attention device-input MoE returned no output after support check; intermediate cleanup={intermediates_cleanup:?}; input cleanup={input_cleanup:?}"
                                )));
                            }
                            Err(err) => {
                                let intermediates_cleanup =
                                    release_qwen_attention_device_intermediates(
                                        normalized, residual, None,
                                    );
                                let input_cleanup = device_hidden.output.release();
                                return Err(crate::error::LlmError::Forward(format!(
                                    "{err}; CUDA Qwen attention intermediate cleanup={intermediates_cleanup:?}; input cleanup={input_cleanup:?}"
                                )));
                            }
                        };
                    if let Err(err) = release_qwen_attention_device_intermediates(
                        normalized,
                        residual,
                        Some(moe_output.output_id),
                    ) {
                        let output_cleanup = moe_output.release();
                        let input_cleanup = device_hidden.output.release();
                        return Err(crate::error::LlmError::Forward(format!(
                            "{err}; CUDA Qwen attention MoE output cleanup={output_cleanup:?}; input cleanup={input_cleanup:?}"
                        )));
                    }
                    if let Err(err) = replace_qwen_attention_device_kv(
                        kv_cache,
                        device_layer,
                        &attention_kv,
                        pos_start,
                        seq_len,
                        kv_dim,
                    ) {
                        let output_cleanup = moe_output.release();
                        let input_cleanup = device_hidden.output.release();
                        return Err(crate::error::LlmError::Forward(format!(
                            "{err}; CUDA Qwen attention output cleanup={output_cleanup:?}; input cleanup={input_cleanup:?}"
                        )));
                    }
                    if let Err(err) = device_hidden.output.release() {
                        let output_cleanup = moe_output.release();
                        return Err(crate::error::LlmError::Forward(format!(
                            "{err}; CUDA Qwen attention output cleanup={output_cleanup:?}"
                        )));
                    }
                    if super::policy::cuda_device_prefill_trace_enabled() {
                        eprintln!(
                            "[cuda:qwen-attention-prefill-chain] layer={} tokens={} hidden={} hidden_d2h_bytes=0 kv_d2h_bytes={}",
                            layer_idx,
                            seq_len,
                            metadata.hidden_dim,
                            attention_kv
                                .k_bits
                                .len()
                                .saturating_add(attention_kv.v_bits.len())
                                .saturating_mul(std::mem::size_of::<u16>())
                        );
                    }
                    hidden = hidden_carrier::PrefillHidden::Device(
                        hidden_carrier::DevicePrefillHidden {
                            output: moe_output,
                            producer_layer_idx: layer_idx,
                        },
                    );
                    layer_idx += 1;
                    continue;
                } else {
                    let reason = device_hidden_materialize_reason(current_layer_kind);
                    hidden = hidden
                        .into_host_for_layer(Some(layer_idx), reason)
                        .map(hidden_carrier::PrefillHidden::Host)?;
                }
            } else if architecture == ModelArchitecture::NemotronHMoE
                && super::policy::cuda_nemotron_device_hidden_carrier_enabled()
            {
                if !device_carrier_accepts_layer_kind(architecture, current_layer_kind) {
                    let reason = device_hidden_materialize_reason(current_layer_kind);
                    hidden = hidden
                        .into_host_for_layer(Some(layer_idx), reason)
                        .map(hidden_carrier::PrefillHidden::Host)?;
                } else if let Some(LayerType::NemotronMoE(w)) = weights.layers.get(layer_idx) {
                    let device_hidden = hidden.take_device().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "expected CUDA device hidden carrier".to_string(),
                        )
                    })?;
                    let producer_layer_idx = device_hidden.producer_layer_idx;
                    match models::nemotron::moe::forward_moe_layer_from_device_for_chain(
                        metadata,
                        device_hidden,
                        w,
                        norm_eps,
                        layer_idx,
                        producer_layer_idx,
                    )? {
                        models::nemotron::moe::NemotronMamba2ToMoeDeviceAttempt::Done {
                            output,
                            ..
                        } => {
                            hidden = hidden_carrier::PrefillHidden::Device(
                                hidden_carrier::DevicePrefillHidden {
                                    output,
                                    producer_layer_idx: layer_idx,
                                },
                            );
                            layer_idx += 1;
                            continue;
                        }
                        models::nemotron::moe::NemotronMamba2ToMoeDeviceAttempt::Materialize {
                            device_hidden,
                            reason,
                        } => {
                            let _ = device_hidden.output.release()?;
                            return Err(crate::error::LlmError::Forward(format!(
                                "CUDA Nemotron MoE execution is unavailable for layer {layer_idx}: {reason}; CPU fallback is disabled"
                            )));
                        }
                    }
                } else if let Some(LayerType::NemotronMamba2(w)) = weights.layers.get(layer_idx) {
                    let device_hidden = hidden.take_device().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "expected CUDA device hidden carrier".to_string(),
                        )
                    })?;
                    match models::nemotron::mamba::forward_mamba2_layer_from_device(
                        kv_cache,
                        metadata,
                        device_hidden.output,
                        w,
                        layer_idx,
                        seq_len,
                        norm_eps,
                    )? {
                        models::nemotron::mamba::NemotronMamba2DeviceAttempt::Done(
                            mamba_output,
                            _trace,
                        ) => {
                            hidden = hidden_carrier::PrefillHidden::Device(
                                hidden_carrier::DevicePrefillHidden {
                                    output: mamba_output,
                                    producer_layer_idx: layer_idx,
                                },
                            );
                            if models::nemotron::mamba::device_prefill_trace_enabled() {
                                let next_layer_idx = next_prefill_pair_layer_idx(
                                    layer_idx,
                                    layer_range.end,
                                    weights.layers.len(),
                                );
                                eprintln!(
                                    "[cuda:device-prefill-chain] op=device_hidden_carry layers={},{} hidden_device=1 d2h_bytes=0 reason=next_layer_accepts_device",
                                    layer_idx,
                                    next_layer_idx
                                        .map(|idx| idx.to_string())
                                        .unwrap_or_else(|| "none".to_string())
                                );
                            }
                            layer_idx += 1;
                            continue;
                        }
                        models::nemotron::mamba::NemotronMamba2DeviceAttempt::Fallback(
                            input_output,
                        ) => {
                            let _ = input_output.release()?;
                            return Err(crate::error::LlmError::Forward(format!(
                                "CUDA Mamba2 execution is unavailable for layer {layer_idx}; CPU fallback is disabled"
                            )));
                        }
                    }
                } else {
                    let reason = device_hidden_materialize_reason(current_layer_kind);
                    hidden = hidden
                        .into_host_for_layer(Some(layer_idx), reason)
                        .map(hidden_carrier::PrefillHidden::Host)?;
                }
            } else if architecture == ModelArchitecture::Gemma4 {
                let device_hidden = hidden.take_device().ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "expected CUDA device hidden carrier".to_string(),
                    )
                })?;
                trace_cuda_device_hidden_hash(
                    "gemma-device-input",
                    device_hidden.producer_layer_idx,
                    Some(layer_idx),
                    &device_hidden.output,
                )?;
                if let Some(LayerType::Attention(w)) = weights.layers.get(layer_idx) {
                    let gemma_runtime_flavor = if metadata.num_layers == 35
                        && metadata.hidden_dim == 1536
                        && metadata.num_heads == 8
                        && metadata.num_kv_heads == 1
                        && metadata.head_dim == 512
                        && metadata.embedding_length_per_layer_input == 256
                    {
                        GemmaRuntimeFlavor::Gemma4E2BIt
                    } else {
                        GemmaRuntimeFlavor::Generic
                    };
                    let layer_kv_override = metadata
                        .head_count_kv_per_layer
                        .as_ref()
                        .and_then(|v| v.get(layer_idx).copied());
                    let kv_source_layer = shared_kv_source_layer(metadata, architecture, layer_idx);
                    let layout = if kv_source_layer.is_some() {
                        resolve_attention_layout_gemma4_reuse(metadata, w, layer_kv_override)?
                    } else {
                        resolve_attention_layout(metadata, w, layer_kv_override)?
                    };
                    let ple_base_for_layer = if gemma_ple_layer_enabled(layer_idx) {
                        gemma_per_layer_base
                    } else {
                        None
                    };
                    #[cfg(feature = "cuda")]
                    let ple_fusion = if !gemma_ple_before_layer() && !gemma_ple_after_out_scale() {
                        match (ple_base_for_layer, weights.gemma_per_layer.as_ref()) {
                            (Some(base), Some(gemma)) => Some(Gemma4PrefillPleFusion {
                                base,
                                weights: gemma,
                            }),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    #[cfg(not(feature = "cuda"))]
                    let ple_fusion: Option<Gemma4PrefillPleFusion<'_>> = None;
                    if let Some(kv_source_layer) = kv_source_layer {
                        let kv_len = pos_start + seq_len;
                        let (cached_k_f16, cached_v_f16) =
                            kv_cache.get_up_to(kv_source_layer, kv_len);
                        let device_attempt = match layout.head_dim {
                            256 => {
                                try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain_from_device(
                                    metadata,
                                    architecture,
                                    gemma_runtime_flavor,
                                    &device_hidden,
                                    w,
                                    weights.rope_freqs.as_ref(),
                                    layout,
                                    true,
                                    cached_k_f16,
                                    cached_v_f16,
                                    layer_idx,
                                    seq_len,
                                    kv_len,
                                    pos_start,
                                    norm_eps,
                                    ple_fusion.as_ref(),
                                )
                            }
                            512 => try_prefill_q4k_f16_reuse_q_hd512_dense_chain_from_device(
                                metadata,
                                architecture,
                                gemma_runtime_flavor,
                                &device_hidden,
                                w,
                                weights.rope_freqs.as_ref(),
                                layout,
                                true,
                                cached_k_f16,
                                cached_v_f16,
                                layer_idx,
                                seq_len,
                                kv_len,
                                pos_start,
                                norm_eps,
                                ple_fusion.as_ref(),
                            ),
                            _ => Ok(None),
                        };
                        match device_attempt {
                            Ok(Some(output)) => {
                                match device_hidden.output.release() {
                                    Ok(true) => {}
                                    Ok(false) => {
                                        return Err(crate::error::LlmError::Forward(
                                            "CUDA Gemma device hidden tensor was already missing"
                                                .to_string(),
                                        ));
                                    }
                                    Err(err) => return Err(err),
                                }
                                hidden = hidden_carrier::PrefillHidden::Device(
                                    hidden_carrier::DevicePrefillHidden {
                                        output,
                                        producer_layer_idx: layer_idx,
                                    },
                                );
                                layer_idx += 1;
                                continue;
                            }
                            Ok(None) => {
                                hidden = hidden_carrier::PrefillHidden::Device(device_hidden)
                                    .into_host_for_layer(Some(layer_idx), "feature_disabled")
                                    .map(hidden_carrier::PrefillHidden::Host)?;
                            }
                            Err(err) => {
                                let cleanup = device_hidden.output.release();
                                return Err(match cleanup {
                                    Ok(true) => err,
                                    Ok(false) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Gemma device hidden cleanup failed: tensor was already missing"
                                    )),
                                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Gemma device hidden cleanup failed: {cleanup_err}"
                                    )),
                                });
                            }
                        }
                    } else {
                        let device_attempt = match layout.head_dim {
                            256 => try_prefill_q4k_f16_qkv_hd256_window_dense_chain_from_device(
                                metadata,
                                architecture,
                                gemma_runtime_flavor,
                                &device_hidden,
                                w,
                                weights.rope_freqs.as_ref(),
                                layout,
                                false,
                                layer_idx,
                                seq_len,
                                pos_start,
                                norm_eps,
                                ple_fusion.as_ref(),
                            ),
                            512 => try_prefill_q4k_f16_qkv_hd512_dense_chain_from_device(
                                metadata,
                                architecture,
                                gemma_runtime_flavor,
                                &device_hidden,
                                w,
                                weights.rope_freqs.as_ref(),
                                layout,
                                false,
                                layer_idx,
                                seq_len,
                                pos_start,
                                norm_eps,
                                ple_fusion.as_ref(),
                            ),
                            _ => Ok(None),
                        };
                        match device_attempt {
                            Ok(Some(output)) => {
                                kv_cache
                                    .replace_layer_f16_range_compacted(
                                        layer_idx,
                                        pos_start,
                                        seq_len,
                                        &output.k_bits,
                                        &output.v_bits,
                                    )
                                    .map_err(crate::error::LlmError::Forward)?;
                                let Some(device_output) = output.device_output else {
                                    let cleanup = device_hidden.output.release();
                                    return Err(match cleanup {
                                        Ok(true) => crate::error::LlmError::Forward(
                                            "CUDA Gemma QKV device chain did not return device output"
                                                .to_string(),
                                        ),
                                        Ok(false) => crate::error::LlmError::Forward(
                                            "CUDA Gemma QKV device chain did not return device output; cleanup failed: tensor was already missing"
                                                .to_string(),
                                        ),
                                        Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                                            "CUDA Gemma QKV device chain did not return device output; cleanup failed: {cleanup_err}"
                                        )),
                                    });
                                };
                                let mut device_output = device_output;
                                if gemma4_partial_ple_host_carrier_pending(
                                    architecture,
                                    metadata,
                                    w,
                                    layer_idx,
                                    ple_base_for_layer,
                                    weights.gemma_per_layer.as_ref(),
                                    output.gemma4_ple_fused,
                                    output.gemma4_output_scale_fused,
                                ) {
                                    let (Some(base), Some(gemma)) =
                                        (ple_base_for_layer, weights.gemma_per_layer.as_ref())
                                    else {
                                        let err = gemma4_partial_ple_release_error(
                                            device_output,
                                            "CUDA Gemma partial PLE carrier missing PLE weights"
                                                .to_string(),
                                        );
                                        let cleanup = device_hidden.output.release();
                                        return Err(match cleanup {
                                            Ok(true) => err,
                                            Ok(false) => crate::error::LlmError::Forward(format!(
                                                "{err}; input cleanup failed: tensor was already missing"
                                            )),
                                            Err(cleanup_err) => crate::error::LlmError::Forward(
                                                format!(
                                                    "{err}; input cleanup failed: {cleanup_err}"
                                                ),
                                            ),
                                        });
                                    };
                                    device_output = match apply_gemma4_partial_ple_host_carrier(
                                        device_output,
                                        metadata,
                                        architecture,
                                        w,
                                        base,
                                        gemma,
                                        layer_idx,
                                        seq_len,
                                        norm_eps,
                                    ) {
                                        Ok(device_output) => device_output,
                                        Err(err) => {
                                            let cleanup = device_hidden.output.release();
                                            return Err(match cleanup {
                                                Ok(true) => err,
                                                Ok(false) => crate::error::LlmError::Forward(
                                                    format!(
                                                        "{err}; CUDA Gemma device hidden cleanup failed: tensor was already missing"
                                                    ),
                                                ),
                                                Err(cleanup_err) => {
                                                    crate::error::LlmError::Forward(format!(
                                                        "{err}; CUDA Gemma device hidden cleanup failed: {cleanup_err}"
                                                    ))
                                                }
                                            });
                                        }
                                    };
                                }
                                trace_cuda_device_hidden_hash(
                                    "gemma-device-output",
                                    layer_idx,
                                    layer_idx.checked_add(1),
                                    &device_output,
                                )?;
                                match device_hidden.output.release() {
                                    Ok(true) => {}
                                    Ok(false) => {
                                        return Err(crate::error::LlmError::Forward(
                                            "CUDA Gemma device hidden tensor was already missing"
                                                .to_string(),
                                        ));
                                    }
                                    Err(err) => return Err(err),
                                }
                                hidden = hidden_carrier::PrefillHidden::Device(
                                    hidden_carrier::DevicePrefillHidden {
                                        output: device_output,
                                        producer_layer_idx: layer_idx,
                                    },
                                );
                                layer_idx += 1;
                                continue;
                            }
                            Ok(None) => {
                                hidden = hidden_carrier::PrefillHidden::Device(device_hidden)
                                    .into_host_for_layer(Some(layer_idx), "feature_disabled")
                                    .map(hidden_carrier::PrefillHidden::Host)?;
                            }
                            Err(err) => {
                                let cleanup = device_hidden.output.release();
                                return Err(match cleanup {
                                    Ok(true) => err,
                                    Ok(false) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Gemma device hidden cleanup failed: tensor was already missing"
                                    )),
                                    Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                                        "{err}; CUDA Gemma device hidden cleanup failed: {cleanup_err}"
                                    )),
                                });
                            }
                        }
                    }
                } else {
                    hidden = hidden_carrier::PrefillHidden::Device(device_hidden)
                        .into_host_for_layer(Some(layer_idx), "unsupported_layer")
                        .map(hidden_carrier::PrefillHidden::Host)?;
                }
            } else {
                hidden = hidden
                    .into_host_for_layer(Some(layer_idx), "feature_disabled")
                    .map(hidden_carrier::PrefillHidden::Host)?;
            }
        }
        let mut hidden_tensor = hidden.into_host_for_layer(Some(layer_idx), "feature_disabled")?;
        // cu94: prefill layer-entry hidden NaN/max scan (RNB_PREFILL_NAN_TRACE=1).
        if crate::engine::policy::env_string("RNB_PREFILL_NAN_TRACE").is_some() {
            let data = kernels::tensor_as_f32_slice(&hidden_tensor);
            let mut nan_count = 0usize;
            let mut inf_count = 0usize;
            let mut max_abs: f32 = 0.0;
            for &v in data.iter() {
                if v.is_nan() {
                    nan_count += 1;
                } else if v.is_infinite() {
                    inf_count += 1;
                } else if v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }
            eprintln!(
                "[cu94-prefill-entry] layer={layer_idx} elems={} nan={nan_count} inf={inf_count} max_abs={max_abs:.4}",
                data.len()
            );
        }
        let layer_start = profiler.enabled().then(Instant::now);
        let pre_layer_hidden = hidden_tensor.clone();
        let ple_base = gemma_per_layer_base;
        let ple_base = if gemma_ple_layer_enabled(layer_idx) {
            ple_base
        } else {
            None
        };
        if gemma_ple_before_layer() {
            if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                hidden_tensor = apply_gemma_per_layer_branch(
                    if gemma_ple_use_layer_input() || gemma_ple_pre_norm_input() {
                        pre_layer_hidden.clone()
                    } else {
                        hidden_tensor
                    },
                    base,
                    layer_idx,
                    gemma,
                    metadata,
                    architecture,
                    norm_eps,
                )?;
            }
        }
        let mut gemma4_ple_fused = false;
        let mut gemma4_output_scale_fused = false;
        match &weights.layers[layer_idx] {
            LayerType::Attention(w) => {
                if architecture == ModelArchitecture::GlmDsa {
                    let glm_layers = weights.glm_dsa_attention.as_ref().ok_or_else(|| {
                        crate::error::LlmError::Forward(
                            "GLM DSA attention weights are not loaded".into(),
                        )
                    })?;
                    let glm = glm_layers.get(layer_idx).ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "GLM DSA attention layer {layer_idx} is missing"
                        ))
                    })?;
                    hidden_tensor = models::glm_dsa::prefill_layer(
                        kv_cache,
                        metadata,
                        hidden_tensor,
                        w,
                        glm,
                        layer_idx,
                        seq_len,
                        pos_start,
                    )?;
                } else if !gemma_ple_disable_attention_layer(layer_idx) {
                    #[cfg(feature = "cuda")]
                    let ple_fusion = if matches!(architecture, ModelArchitecture::Gemma4)
                        && !gemma_ple_before_layer()
                        && !gemma_ple_after_out_scale()
                    {
                        match (ple_base, weights.gemma_per_layer.as_ref()) {
                            (Some(base), Some(gemma)) => Some(Gemma4PrefillPleFusion {
                                base,
                                weights: gemma,
                            }),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    #[cfg(not(feature = "cuda"))]
                    let ple_fusion: Option<Gemma4PrefillPleFusion<'_>> = None;
                    let attention_output = forward_attention_layer_with_gemma4_ple_fusion(
                        kv_cache,
                        metadata,
                        architecture,
                        hidden_tensor,
                        w,
                        weights.rope_freqs.as_ref(),
                        layer_idx,
                        seq_len,
                        pos_start,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        kv_dim,
                        rope_theta,
                        norm_eps,
                        ple_fusion.as_ref(),
                    )?;
                    let attention_gemma4_ple_fused = attention_output.gemma4_ple_fused;
                    let attention_gemma4_output_scale_fused =
                        attention_output.gemma4_output_scale_fused;
                    #[cfg(feature = "cuda")]
                    if let Some(device_output) = attention_output.device_output {
                        let mut device_output = device_output;
                        if gemma4_partial_ple_host_carrier_pending(
                            architecture,
                            metadata,
                            w,
                            layer_idx,
                            ple_base,
                            weights.gemma_per_layer.as_ref(),
                            attention_gemma4_ple_fused,
                            attention_gemma4_output_scale_fused,
                        ) {
                            let (Some(base), Some(gemma)) =
                                (ple_base, weights.gemma_per_layer.as_ref())
                            else {
                                return Err(gemma4_partial_ple_release_error(
                                    device_output,
                                    "CUDA Gemma partial PLE carrier missing PLE weights"
                                        .to_string(),
                                ));
                            };
                            device_output = apply_gemma4_partial_ple_host_carrier(
                                device_output,
                                metadata,
                                architecture,
                                w,
                                base,
                                gemma,
                                layer_idx,
                                seq_len,
                                norm_eps,
                            )?;
                        }
                        trace_cuda_device_hidden_hash(
                            "gemma-host-device-output",
                            layer_idx,
                            layer_idx.checked_add(1),
                            &device_output,
                        )?;
                        hidden = hidden_carrier::PrefillHidden::Device(
                            hidden_carrier::DevicePrefillHidden {
                                output: device_output,
                                producer_layer_idx: layer_idx,
                            },
                        );
                        if let Some(layer_start) = layer_start {
                            profiler
                                .record(layer_idx, layer_start.elapsed().as_secs_f64() * 1000.0);
                        }
                        layer_idx += 1;
                        continue;
                    }
                    hidden_tensor = attention_output.hidden;
                    gemma4_ple_fused = attention_gemma4_ple_fused;
                    gemma4_output_scale_fused = attention_gemma4_output_scale_fused;
                }
            }
            LayerType::GatedDeltaNet(w) => {
                #[cfg(feature = "cuda")]
                if prefix_collector.is_none()
                    && architecture == ModelArchitecture::Qwen35MoE
                    && qwen_gdn_moe_output_device_enabled()
                {
                    if let Some(output) =
                        models::qwen::try_forward_gdn_layer_from_host_to_device_output(
                            kv_cache,
                            metadata,
                            &hidden_tensor,
                            w,
                            layer_idx,
                            seq_len,
                            norm_eps,
                        )?
                    {
                        hidden = hidden_carrier::PrefillHidden::Device(
                            hidden_carrier::DevicePrefillHidden {
                                output,
                                producer_layer_idx: layer_idx,
                            },
                        );
                        if let Some(layer_start) = layer_start {
                            profiler
                                .record(layer_idx, layer_start.elapsed().as_secs_f64() * 1000.0);
                        }
                        layer_idx += 1;
                        continue;
                    }
                }
                hidden_tensor = if prefix_collector.is_some() {
                    forward_gdn_layer_collect_prefix_state(
                        kv_cache,
                        metadata,
                        hidden_tensor,
                        w,
                        layer_idx,
                        seq_len,
                        norm_eps,
                        prefix_collector.as_deref_mut(),
                    )?
                } else {
                    forward_gdn_layer(
                        kv_cache,
                        metadata,
                        hidden_tensor,
                        w,
                        layer_idx,
                        seq_len,
                        norm_eps,
                    )?
                };
            }
            LayerType::NemotronMamba2(w) => {
                hidden_tensor = models::nemotron::mamba::forward_mamba2_layer(
                    kv_cache,
                    metadata,
                    hidden_tensor,
                    w,
                    layer_idx,
                    seq_len,
                    norm_eps,
                )?;
            }
            LayerType::NemotronMoE(w) => {
                if architecture == ModelArchitecture::NemotronHMoE
                    && (models::nemotron::moe::device_prefill_chain_smoke_enabled()
                        || models::nemotron::mamba::device_prefill_mamba2_enabled())
                {
                    let next_layer_idx = next_prefill_pair_layer_idx(
                        layer_idx,
                        layer_range.end,
                        weights.layers.len(),
                    );
                    let next_layer_kind = next_layer_idx
                        .and_then(|idx| weights.layers.get(idx))
                        .map(prefill_layer_kind_name);
                    #[cfg(feature = "cuda")]
                    if let Some(next_idx) = next_layer_idx {
                        let after_pair_layer_idx = next_idx
                            .checked_add(1)
                            .filter(|&idx| idx < layer_range.end && idx < weights.layers.len());
                        match try_run_nemotron_moe_mamba2_device_pair(
                            kv_cache,
                            metadata,
                            &hidden_tensor,
                            weights,
                            w,
                            layer_idx,
                            next_idx,
                            seq_len,
                            norm_eps,
                        )? {
                            NemotronMoeMamba2DevicePairAttempt::Done(pair_output) => {
                                if super::policy::cuda_nemotron_device_hidden_carrier_enabled() {
                                    hidden = hidden_carrier::PrefillHidden::Device(
                                        hidden_carrier::DevicePrefillHidden {
                                            output: pair_output,
                                            producer_layer_idx: next_idx,
                                        },
                                    );
                                    if models::nemotron::mamba::device_prefill_trace_enabled() {
                                        eprintln!(
                                            "[cuda:device-prefill-chain] op=device_hidden_carry layers={},{} hidden_device=1 d2h_bytes=0 reason=next_layer_accepts_device",
                                            next_idx,
                                            after_pair_layer_idx
                                                .map(|idx| idx.to_string())
                                                .unwrap_or_else(|| "none".to_string())
                                        );
                                    }
                                } else {
                                    let next_after_pair_kind = after_pair_layer_idx
                                        .and_then(|idx| weights.layers.get(idx))
                                        .map(prefill_layer_kind_name);
                                    let reason =
                                        device_hidden_materialize_reason(next_after_pair_kind);
                                    hidden_tensor = hidden_carrier::PrefillHidden::Device(
                                        hidden_carrier::DevicePrefillHidden {
                                            output: pair_output,
                                            producer_layer_idx: next_idx,
                                        },
                                    )
                                    .into_host_for_layer(after_pair_layer_idx, reason)?;
                                    observe_prefill_layer_output(
                                        &hidden_tensor,
                                        metadata,
                                        seq_len,
                                        next_idx,
                                        layer_start,
                                        &mut profiler,
                                    );
                                    hidden = hidden_carrier::PrefillHidden::Host(hidden_tensor);
                                }
                                layer_idx = next_idx.checked_add(1).ok_or_else(|| {
                                    crate::error::LlmError::Forward(
                                        "CUDA MoE/Mamba2 pair layer cursor overflow".to_string(),
                                    )
                                })?;
                                continue;
                            }
                            NemotronMoeMamba2DevicePairAttempt::MambaUnavailable => {}
                        }
                    }
                    #[cfg(feature = "cuda")]
                    {
                        hidden_tensor = models::nemotron::moe::forward_moe_layer_chain_smoke(
                            metadata,
                            hidden_tensor,
                            w,
                            norm_eps,
                            layer_idx,
                            next_layer_idx,
                            next_layer_kind,
                        )?;
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        hidden_tensor = models::nemotron::moe::forward_moe_layer_chain_smoke(
                            metadata,
                            hidden_tensor,
                            w,
                            norm_eps,
                            layer_idx,
                            next_layer_idx,
                            next_layer_kind,
                        )?;
                    }
                } else {
                    hidden_tensor = models::nemotron::moe::forward_moe_layer_for_prefill(
                        metadata,
                        hidden_tensor,
                        w,
                        norm_eps,
                        Some(layer_idx),
                    )?;
                }
            }
        }
        if gemma_ple_before_layer() {
            if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                hidden_tensor =
                    apply_layer_output_scale(hidden_tensor, w.out_scale.as_ref(), layer_idx);
            }
            let hidden_data = kernels::tensor_as_f32_slice(&hidden_tensor);
            let last_row =
                &hidden_data[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
            emit_layer_trace("prefill", layer_idx, last_row);
            if let Some(layer_start) = layer_start {
                profiler.record(layer_idx, layer_start.elapsed().as_secs_f64() * 1000.0);
            }
            hidden = hidden_carrier::PrefillHidden::Host(hidden_tensor);
            layer_idx += 1;
            continue;
        }
        if let LayerType::Attention(w) = &weights.layers[layer_idx] {
            let post_layer_hidden = hidden_tensor.clone();
            if !gemma_ple_after_out_scale() {
                if !gemma4_ple_fused {
                    if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref())
                    {
                        if gemma_ple_global_only(metadata, w) {
                        } else if super::policy::gemma_ple_global_only_enabled() {
                            hidden_tensor = apply_layer_output_scale(
                                hidden_tensor,
                                w.out_scale.as_ref(),
                                layer_idx,
                            );
                            let hidden_data = kernels::tensor_as_f32_slice(&hidden_tensor);
                            let last_row = &hidden_data[(seq_len - 1) * metadata.hidden_dim
                                ..seq_len * metadata.hidden_dim];
                            emit_layer_trace("prefill", layer_idx, last_row);
                            if let Some(layer_start) = layer_start {
                                profiler.record(
                                    layer_idx,
                                    layer_start.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            hidden = hidden_carrier::PrefillHidden::Host(hidden_tensor);
                            layer_idx += 1;
                            continue;
                        }
                        let ple_output = apply_gemma_per_layer_branch_with_output_scale(
                            if gemma_ple_use_layer_input() {
                                pre_layer_hidden.clone()
                            } else if gemma_ple_pre_norm_input() {
                                post_layer_hidden.clone()
                            } else {
                                hidden_tensor
                            },
                            base,
                            layer_idx,
                            gemma,
                            metadata,
                            architecture,
                            norm_eps,
                            w.out_scale.as_ref(),
                        )?;
                        hidden_tensor = ple_output.hidden;
                        gemma4_output_scale_fused = ple_output.output_scale_applied;
                    }
                }
            }
            if !gemma4_output_scale_fused {
                hidden_tensor =
                    apply_layer_output_scale(hidden_tensor, w.out_scale.as_ref(), layer_idx);
            }
            if !gemma4_ple_fused && gemma_ple_after_out_scale() {
                if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                    if gemma_ple_global_only(metadata, w) {
                    } else if super::policy::gemma_ple_global_only_enabled() {
                        hidden_tensor = apply_layer_output_scale(
                            hidden_tensor,
                            w.out_scale.as_ref(),
                            layer_idx,
                        );
                        if let Some(layer_start) = layer_start {
                            profiler
                                .record(layer_idx, layer_start.elapsed().as_secs_f64() * 1000.0);
                        }
                        hidden = hidden_carrier::PrefillHidden::Host(hidden_tensor);
                        layer_idx += 1;
                        continue;
                    }
                    hidden_tensor = apply_gemma_per_layer_branch(
                        if gemma_ple_use_layer_input() {
                            pre_layer_hidden.clone()
                        } else if gemma_ple_pre_norm_input() {
                            post_layer_hidden.clone()
                        } else {
                            hidden_tensor
                        },
                        base,
                        layer_idx,
                        gemma,
                        metadata,
                        architecture,
                        norm_eps,
                    )?;
                }
            }
        } else if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
            hidden_tensor = apply_gemma_per_layer_branch(
                if gemma_ple_use_layer_input() {
                    pre_layer_hidden
                } else if gemma_ple_pre_norm_input() {
                    hidden_tensor.clone()
                } else {
                    hidden_tensor
                },
                base,
                layer_idx,
                gemma,
                metadata,
                architecture,
                norm_eps,
            )?;
        }
        observe_prefill_layer_output(
            &hidden_tensor,
            metadata,
            seq_len,
            layer_idx,
            layer_start,
            &mut profiler,
        );
        hidden = hidden_carrier::PrefillHidden::Host(hidden_tensor);
        layer_idx += 1;
    }
    #[cfg(feature = "cuda")]
    if nemotron_workspace_active {
        // cu19: any leftover `PrefillHidden::Device` from the last layer holds a
        // live lease on the workspace arena (workspace-backed device tensors).
        // Closing the workspace while leases are outstanding fails with
        // `live_leases != 0`. Materialize to host first so the workspace can
        // close cleanly. `range_end` mirrors the materialize reason used by
        // the in-loop transition into the next attention/dense layer.
        if matches!(hidden, hidden_carrier::PrefillHidden::Device(_)) {
            hidden = hidden
                .into_host_for_layer(None, "range_end_workspace_close")
                .map(hidden_carrier::PrefillHidden::Host)?;
        }
        let summary = crate::engine::cuda_runtime::end_nemotron_prefill_workspace()
            .map_err(crate::error::LlmError::Forward)?;
        if super::policy::cuda_device_prefill_trace_enabled() {
            eprintln!(
                "{}",
                nemotron_prefill_v2_workspace_lifecycle_trace_line(
                    "workspace_end",
                    summary.active,
                    summary.arena_bytes,
                    summary.live_leases,
                    summary.hit_bytes,
                    summary.miss_bytes,
                    summary.owned_alloc_count,
                )
            );
        }
    }

    profiler.report(weights);
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn metal_qwen_prefill_chain_defaults_on_with_falsey_opt_out() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = "RNB_METAL_QWEN_PREFILL_CHAIN";
        let previous = crate::engine::policy::env_string(key);

        unsafe {
            std::env::remove_var(key);
        }
        assert!(super::metal_qwen_prefill_chain_enabled());

        for value in ["0", "false", "off", "no"] {
            unsafe {
                std::env::set_var(key, value);
            }
            assert!(!super::metal_qwen_prefill_chain_enabled(), "{value}");
        }

        for value in ["1", "true", "on", "yes", "typo"] {
            unsafe {
                std::env::set_var(key, value);
            }
            assert!(super::metal_qwen_prefill_chain_enabled(), "{value}");
        }

        unsafe {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
    #[test]
    fn device_hidden_materialize_reason_tracks_next_layer_kind() {
        assert_eq!(super::device_hidden_materialize_reason(None), "range_end");
        assert_eq!(
            super::device_hidden_materialize_reason(Some("attention")),
            "attention_requires_host"
        );
        assert_eq!(
            super::device_hidden_materialize_reason(Some("nemotron_moe")),
            "unsupported_router_quant"
        );
        assert_eq!(
            super::device_hidden_materialize_reason(Some("gated_delta_net")),
            "feature_disabled"
        );
    }

    #[test]
    fn device_carrier_accepts_mamba2_consumer_for_nemotron() {
        assert!(super::device_carrier_accepts_layer_kind(
            super::ModelArchitecture::NemotronHMoE,
            Some("nemotron_mamba2")
        ));
    }

    #[test]
    fn device_carrier_accepts_qwen_gdn_and_attention_when_output_carry_enabled() {
        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE", "0");
        }
        assert!(!super::device_carrier_accepts_layer_kind(
            super::ModelArchitecture::Qwen35MoE,
            Some("gated_delta_net")
        ));
        assert!(!super::device_carrier_accepts_layer_kind(
            super::ModelArchitecture::Qwen35MoE,
            Some("attention")
        ));

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE", "1");
        }
        assert!(super::device_carrier_accepts_layer_kind(
            super::ModelArchitecture::Qwen35MoE,
            Some("gated_delta_net")
        ));
        assert!(super::device_carrier_accepts_layer_kind(
            super::ModelArchitecture::Qwen35MoE,
            Some("attention")
        ));
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE");
        }
    }

    #[test]
    fn pair_next_layer_stays_inside_requested_range() {
        assert_eq!(super::next_prefill_pair_layer_idx(2, 3, 8), None);
        assert_eq!(super::next_prefill_pair_layer_idx(2, 4, 8), Some(3));
    }

    #[test]
    fn pair_next_layer_stays_inside_loaded_layers() {
        assert_eq!(super::next_prefill_pair_layer_idx(2, 8, 3), None);
        assert_eq!(super::next_prefill_pair_layer_idx(2, 8, 4), Some(3));
    }

    #[test]
    fn nemotron_prefill_v2_workspace_trace_line_reports_decision_and_bytes() {
        let plan = crate::engine::workspace_runtime::NemotronPrefillWorkspacePlan {
            seq_len: 128,
            chunk_len: 56,
            hidden_dim: 16,
            n_expert: 8,
            expert_used: 2,
            n_ff: 64,
            shared_ff: 32,
            free_vram_bytes: 50_000,
            total_vram_bytes: 10_000_000,
            reserve_bytes: 0,
            usable_vram_bytes: 50_000,
            route_slots: 112,
            normalized_bytes: 3584,
            router_logits_bytes: 1792,
            route_bytes: 1344,
            hidden_bytes: 3584,
            persistent_hidden_bytes: 7168,
            moe_intermediate_bytes: 35840,
            mamba_state_sync_bytes: 0,
            attention_handoff_bytes: 3584,
            required_workspace_bytes: 49728,
            decision: crate::engine::workspace_runtime::NemotronPrefillWorkspaceDecision::Chunk {
                chunk_len: 56,
            },
        };

        let line = super::nemotron_prefill_v2_workspace_trace_line(&plan);

        assert!(line.contains("[cuda:device-prefill-v2] op=workspace_plan"));
        assert!(line.contains("seq_len=128"));
        assert!(line.contains("chunk_len=56"));
        assert!(line.contains("decision=chunk"));
        assert!(line.contains("required_bytes=49728"));
        assert!(line.contains("usable_bytes=50000"));
        assert!(line.contains("route_slots=112"));
        assert!(line.contains("attention_handoff_bytes=3584"));
    }

    #[test]
    fn nemotron_prefill_v2_workspace_lifecycle_trace_line_reports_summary_fields() {
        let line = super::nemotron_prefill_v2_workspace_lifecycle_trace_line(
            "workspace_begin",
            true,
            4096,
            0,
            2048,
            128,
            3,
        );

        assert_eq!(
            line,
            "[cuda:device-prefill-v2] op=workspace_begin active=true arena_bytes=4096 live_leases=0 hit_bytes=2048 miss_bytes=128 owned_alloc_count=3"
        );
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_prefill_layer_chain_requires_host_for_active_moe_histogram_trace() {
        assert_eq!(
            super::metal_qwen_prefill_moe_trace_requirement(true),
            Some("moe_trace_active")
        );
        assert_eq!(super::metal_qwen_prefill_moe_trace_requirement(false), None);
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_moe_llama_prefill_layer_chain_accepts_supported_moe_tuples() {
        use rnb_loader::GGMLType;

        for sparse_down in [GGMLType::Q4_K, GGMLType::Q5_K, GGMLType::Q6_K] {
            assert!(super::metal_qwen_prefill_moe_product_tuple(
                8,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                sparse_down,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
            ));
        }
        for shared_down in [GGMLType::Q4_K, GGMLType::Q6_K] {
            assert!(super::metal_qwen_prefill_moe_product_tuple(
                8,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                shared_down,
            ));
        }
        assert!(!super::metal_qwen_prefill_moe_product_tuple(
            9,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q6_K,
        ));
        for malformed in [
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
            ],
            [
                GGMLType::Q8_0,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q5_K,
            ],
        ] {
            assert!(!super::metal_qwen_prefill_moe_product_tuple(
                8,
                malformed[0],
                malformed[1],
                malformed[2],
                malformed[3],
                malformed[4],
                malformed[5],
            ));
        }
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_validates_ownership_counts() {
        let report = super::hidden_carrier::validate_metal_qwen_prefill_ownership(
            40, 10, 30, 10, 30, 1, 1, 0,
        )
        .expect("valid one-shot ownership");
        assert_eq!(report.requested_layers, 40);
        assert_eq!(report.attention_kv_writes, 10);
        assert_eq!(report.gdn_state_writes, 30);
        assert_eq!(report.hidden_uploads, 1);
        assert_eq!(report.hidden_readbacks, 1);
        assert_eq!(report.intermediate_hidden_transfers, 0);

        for invalid in [
            [0, 0, 0, 0, 0, 1, 1, 0],
            [40, 10, 30, 9, 30, 1, 1, 0],
            [40, 10, 30, 10, 29, 1, 1, 0],
            [40, 10, 30, 10, 30, 2, 1, 0],
            [40, 10, 30, 10, 30, 1, 2, 0],
            [40, 10, 30, 10, 30, 1, 1, 1],
        ] {
            assert!(
                super::hidden_carrier::validate_metal_qwen_prefill_ownership(
                    invalid[0], invalid[1], invalid[2], invalid[3], invalid[4], invalid[5],
                    invalid[6], invalid[7],
                )
                .is_err(),
                "invalid ownership counts unexpectedly accepted: {invalid:?}"
            );
        }
    }
}

#[cfg(feature = "vulkan")]
pub(super) fn run_prefill_layers_cpu_range_with_gpu(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    weights: &ModelWeights,
    gemma_per_layer_base: Option<&GemmaPerLayerBase>,
    mut hidden: Tensor,
    layer_range: Range<usize>,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
    mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
    mut deferred_gdn_flush: Option<&mut DeferredGdnConvStateFlush>,
) -> crate::error::Result<Tensor> {
    for layer_idx in layer_range {
        let pre_layer_hidden = hidden.clone();
        let ple_base = gemma_per_layer_base;
        let ple_base = if gemma_ple_layer_enabled(layer_idx) {
            ple_base
        } else {
            None
        };
        if gemma_ple_before_layer() {
            if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                hidden = apply_gemma_per_layer_branch(
                    if gemma_ple_use_layer_input() || gemma_ple_pre_norm_input() {
                        pre_layer_hidden.clone()
                    } else {
                        hidden
                    },
                    base,
                    layer_idx,
                    gemma,
                    metadata,
                    ModelArchitecture::Gemma,
                    norm_eps,
                )?;
            }
        }
        match &weights.layers[layer_idx] {
            LayerType::Attention(w) => {
                if !gemma_ple_disable_attention_layer(layer_idx) {
                    hidden = forward_attention_layer(
                        kv_cache,
                        metadata,
                        ModelArchitecture::Qwen35,
                        hidden,
                        w,
                        weights.rope_freqs.as_ref(),
                        layer_idx,
                        seq_len,
                        pos_start,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        kv_dim,
                        rope_theta,
                        norm_eps,
                    )?;
                }
            }
            LayerType::GatedDeltaNet(w) => {
                hidden = forward_gdn_layer_with_gpu(
                    kv_cache,
                    metadata,
                    hidden,
                    w,
                    layer_idx,
                    seq_len,
                    norm_eps,
                    gpu_runtime.as_deref_mut(),
                    deferred_gdn_flush.as_deref_mut(),
                )?;
            }
            LayerType::NemotronMamba2(w) => {
                hidden = models::nemotron::mamba::forward_mamba2_layer(
                    kv_cache, metadata, hidden, w, layer_idx, seq_len, norm_eps,
                )?;
            }
            LayerType::NemotronMoE(w) => {
                hidden = models::nemotron::moe::forward_moe_layer_for_prefill(
                    metadata,
                    hidden,
                    w,
                    norm_eps,
                    Some(layer_idx),
                )?;
            }
        }
        if gemma_ple_before_layer() {
            if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                hidden = apply_layer_output_scale(hidden, w.out_scale.as_ref(), layer_idx);
            }
            continue;
        }
        if let LayerType::Attention(w) = &weights.layers[layer_idx] {
            let post_layer_hidden = hidden.clone();
            if !gemma_ple_after_out_scale() {
                if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                    if gemma_ple_global_only(metadata, w) {
                    } else if super::policy::gemma_ple_global_only_enabled() {
                        hidden = apply_layer_output_scale(hidden, w.out_scale.as_ref(), layer_idx);
                        continue;
                    }
                    hidden = apply_gemma_per_layer_branch(
                        if gemma_ple_use_layer_input() {
                            pre_layer_hidden.clone()
                        } else if gemma_ple_pre_norm_input() {
                            post_layer_hidden.clone()
                        } else {
                            hidden
                        },
                        base,
                        layer_idx,
                        gemma,
                        metadata,
                        ModelArchitecture::Gemma,
                        norm_eps,
                    )?;
                }
            }
            hidden = apply_layer_output_scale(hidden, w.out_scale.as_ref(), layer_idx);
            if gemma_ple_after_out_scale() {
                if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
                    hidden = apply_gemma_per_layer_branch(
                        if gemma_ple_use_layer_input() {
                            pre_layer_hidden.clone()
                        } else if gemma_ple_pre_norm_input() {
                            post_layer_hidden.clone()
                        } else {
                            hidden
                        },
                        base,
                        layer_idx,
                        gemma,
                        metadata,
                        ModelArchitecture::Gemma,
                        norm_eps,
                    )?;
                }
            }
        } else if let (Some(base), Some(gemma)) = (ple_base, weights.gemma_per_layer.as_ref()) {
            hidden = apply_gemma_per_layer_branch(
                if gemma_ple_use_layer_input() {
                    pre_layer_hidden
                } else if gemma_ple_pre_norm_input() {
                    hidden.clone()
                } else {
                    hidden
                },
                base,
                layer_idx,
                gemma,
                metadata,
                ModelArchitecture::Gemma,
                norm_eps,
            )?;
        }
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        if dump_bin_dir().is_some() {
            dump_bin("prefill", layer_idx, "layer_out", hidden_data);
        }
        let last_row =
            &hidden_data[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
        emit_layer_trace("prefill", layer_idx, last_row);
    }
    Ok(hidden)
}
