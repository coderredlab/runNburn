use super::super::*;
use rnb_memory::ExpertBundleObservationReceipt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35DeviceMoeOutput {
    pub output_id: rnb_backend_api::DeviceTensorId,
    pub output_desc: rnb_backend_api::DeviceTensorDesc,
}

fn log_qwen35_device_moe_api_phase(
    label: &str,
    token_count: usize,
    n_expert: usize,
    n_expert_used: usize,
    elapsed: std::time::Duration,
) {
    if tuning::qwen35_device_moe_phase_profile_enabled() {
        eprintln!(
            "  [CUDA-QWEN35-MOE api] {:24} {:.3}ms tokens={} experts={} used={}",
            label,
            elapsed.as_secs_f64() * 1000.0,
            token_count,
            n_expert,
            n_expert_used
        );
    }
}

fn device_input_selected_base_temp_slab_enabled() -> bool {
    qwen35_selected_base_temp_slab_ptrs_enabled()
}

fn device_input_selected_base_device_slot_ptrs_enabled() -> bool {
    qwen35_selected_base_device_slot_ptrs_enabled()
}

fn device_input_selected_base_direct_sparse_enabled() -> bool {
    qwen35_selected_base_direct_sparse_enabled()
}

fn device_input_selected_base_existing_resident_mixed_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_EXISTING_RESIDENT_MIXED")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn device_input_use_mixed_resident_sparse(
    explicit_mixed_resident_sparse: bool,
    existing_resident_roles: usize,
) -> bool {
    explicit_mixed_resident_sparse
        || (existing_resident_roles > 0
            && device_input_selected_base_existing_resident_mixed_enabled())
}

fn device_input_use_mixed_device_slot_ptrs(auto_mixed_resident_sparse: bool) -> bool {
    auto_mixed_resident_sparse || qwen35_selected_base_mixed_device_slot_ptrs_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_input_selected_base_temp_slab_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS");
        }
        assert!(device_input_selected_base_temp_slab_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS", "1");
        }
        assert!(device_input_selected_base_temp_slab_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS", "0");
        }
        assert!(!device_input_selected_base_temp_slab_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS");
        }
    }

    #[test]
    fn device_input_selected_base_device_slot_ptrs_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS");
        }
        assert!(device_input_selected_base_device_slot_ptrs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
        }
        assert!(device_input_selected_base_device_slot_ptrs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "0");
        }
        assert!(!device_input_selected_base_device_slot_ptrs_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn device_input_selected_base_direct_sparse_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE");
        }
        assert!(device_input_selected_base_direct_sparse_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
        }
        assert!(device_input_selected_base_direct_sparse_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "0");
        }
        assert!(!device_input_selected_base_direct_sparse_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE");
        }
    }

    #[test]
    fn device_input_existing_resident_hit_auto_enables_mixed_sparse() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_EXISTING_RESIDENT_MIXED");
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT");
        }
        assert!(!device_input_use_mixed_resident_sparse(false, 0));
        assert!(device_input_use_mixed_resident_sparse(false, 1));
        assert!(device_input_use_mixed_resident_sparse(true, 0));

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_EXISTING_RESIDENT_MIXED", "0");
        }
        assert!(!device_input_use_mixed_resident_sparse(false, 1));
        assert!(device_input_use_mixed_resident_sparse(true, 0));

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_EXISTING_RESIDENT_MIXED");
        }
    }

    #[test]
    fn device_input_auto_mixed_uses_device_slot_ptrs() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS");
        }
        assert!(!device_input_use_mixed_device_slot_ptrs(false));
        assert!(device_input_use_mixed_device_slot_ptrs(true));

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS", "1");
        }
        assert!(device_input_use_mixed_device_slot_ptrs(false));

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn device_input_device_sparse_route_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE");
        }
        assert!(!qwen35_device_sparse_route_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE", "1");
        }
        assert!(qwen35_device_sparse_route_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE", "0");
        }
        assert!(!qwen35_device_sparse_route_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE");
        }
    }

    #[test]
    fn device_input_selected_base_copy_stream_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM");
        }
        assert!(qwen35_selected_base_copy_stream_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
        }
        assert!(qwen35_selected_base_copy_stream_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "0");
        }
        assert!(!qwen35_selected_base_copy_stream_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM");
        }
    }

    #[test]
    fn device_input_selected_base_pinned_staging_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING");
        }
        assert!(!qwen35_selected_base_pinned_staging_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING", "1");
        }
        assert!(qwen35_selected_base_pinned_staging_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING", "0");
        }
        assert!(!qwen35_selected_base_pinned_staging_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING");
        }
    }

    #[test]
    fn device_input_selected_base_overlap_staging_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING");
        }
        assert!(!qwen35_selected_base_overlap_staging_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING", "1");
        }
        assert!(qwen35_selected_base_overlap_staging_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING", "0");
        }
        assert!(!qwen35_selected_base_overlap_staging_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING");
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_by_token(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_route: &[f32],
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slots = gate_weights.len();
    if slots == 0 {
        return f32_shared_expert(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            n_ff,
            n_embd,
            input,
        );
    }
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 combined shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    if shared_gate.len() != n_ff * n_embd
        || shared_up.len() != n_ff * n_embd
        || shared_down.len() != n_embd * n_ff
    {
        return Err("Qwen35 combined shared f32 weight shape mismatch".to_string());
    }
    validate_qwen35_sparse_token_batch(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
    if expert_ids.len() != slots {
        return Err(format!(
            "Qwen35 combined expert id length mismatch: got {}, expected {slots}",
            expert_ids.len()
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
        .qwen35_prefill_moe_f32_shared_sparse_by_token(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
    token_count: usize,
    n_expert_used: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Option<Qwen35DeviceMoeOutput>, String> {
    qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_impl(
        shared_gate,
        shared_up,
        shared_down,
        shared_input_scale,
        gate_all,
        up_all,
        down_all,
        router_w,
        n_expert,
        hidden_dim,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        token_count,
        n_expert_used,
        down_quant,
        n_ff,
        n_embd,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
    token_count: usize,
    n_expert_used: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Option<Qwen35DeviceMoeOutput>, String> {
    qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_impl(
        shared_gate,
        shared_up,
        shared_down,
        shared_input_scale,
        gate_all,
        up_all,
        down_all,
        router_w,
        n_expert,
        hidden_dim,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        token_count,
        n_expert_used,
        down_quant,
        n_ff,
        n_embd,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_impl(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
    token_count: usize,
    n_expert_used: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    reuse_residual_output: bool,
) -> Result<Option<Qwen35DeviceMoeOutput>, String> {
    if shared_gate.len() != n_ff * n_embd
        || shared_up.len() != n_ff * n_embd
        || shared_down.len() != n_embd * n_ff
    {
        return Err(
            "Qwen35 device-input selected-base shared f32 weight shape mismatch".to_string(),
        );
    }
    if shared_input_scale.len() != n_embd {
        return Err(format!(
            "Qwen35 device-input selected-base shared scale len mismatch: got {}, expected {n_embd}",
            shared_input_scale.len()
        ));
    }
    let selected_base_n_expert = qwen35_selected_base_full_layer_expert_count(
        gate_all, up_all, down_all, down_quant, n_ff, n_embd,
    )?;
    if selected_base_n_expert != n_expert {
        return Err(format!(
            "Qwen35 device-input selected-base expert count mismatch: router={n_expert} selected_base={selected_base_n_expert}"
        ));
    }

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    let phase_profile = tuning::qwen35_device_moe_phase_profile_enabled();
    let total_start = phase_profile.then(std::time::Instant::now);
    let phase_start = phase_profile.then(std::time::Instant::now);
    let mut route_pack = state.qwen35_prefill_device_topk_route_pack_device_input(
        router_w,
        n_expert,
        hidden_dim,
        input_id,
        input_desc,
        token_count,
        n_expert_used,
    )?;
    if let Some(start) = phase_start {
        log_qwen35_device_moe_api_phase(
            "route_topk",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    let phase_start = phase_profile.then(std::time::Instant::now);
    route_pack.sort_by_expert_token()?;
    if let Some(start) = phase_start {
        log_qwen35_device_moe_api_phase(
            "route_sort",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    let selected_base_device_slot_ptrs = device_input_selected_base_device_slot_ptrs_enabled();
    let explicit_mixed_resident_sparse =
        selected_base_device_slot_ptrs && qwen35_selected_base_mixed_resident_enabled();
    let exact_resident_moe_layer = selected_base_device_slot_ptrs
        && !explicit_mixed_resident_sparse
        && device_input_selected_base_existing_resident_mixed_enabled()
        && state.qwen35_selected_base_has_exact_resident_moe_layer(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        );
    let existing_resident_roles = if exact_resident_moe_layer {
        1
    } else if selected_base_device_slot_ptrs
        && !explicit_mixed_resident_sparse
        && device_input_selected_base_existing_resident_mixed_enabled()
    {
        state.qwen35_selected_base_existing_resident_role_count_by_token(
            gate_all,
            up_all,
            down_all,
            &route_pack.expert_ids,
            down_quant,
            n_ff,
            n_embd,
        )?
    } else {
        0
    };
    let auto_mixed_resident_sparse = selected_base_device_slot_ptrs
        && !explicit_mixed_resident_sparse
        && device_input_use_mixed_resident_sparse(false, existing_resident_roles);
    let mixed_resident_sparse = selected_base_device_slot_ptrs
        && device_input_use_mixed_resident_sparse(
            explicit_mixed_resident_sparse,
            existing_resident_roles,
        );
    let direct_sparse = !mixed_resident_sparse
        && selected_base_device_slot_ptrs
        && device_input_selected_base_direct_sparse_enabled();
    let phase_start = phase_profile.then(std::time::Instant::now);
    let selected = if direct_sparse || mixed_resident_sparse {
        None
    } else {
        Some(qwen35_selected_base_sparse_inputs_from_full_layer(
            gate_all,
            up_all,
            down_all,
            &route_pack.expert_ids,
            &route_pack.route_weights,
            &route_pack.token_ids,
            down_quant,
            n_ff,
            n_embd,
        )?)
    };
    if let Some(start) = phase_start {
        log_qwen35_device_moe_api_phase(
            "selected_gather",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    let phase_start = phase_profile.then(std::time::Instant::now);
    let overlap_selected_base_staging = direct_sparse
        && qwen35_selected_base_copy_stream_enabled()
        && qwen35_selected_base_overlap_staging_enabled();
    let fused_selected_sparse_boundary =
        direct_sparse && qwen35_selected_sparse_fused_boundary_enabled();
    let (prepared_sparse, deferred_selected_base) = if selected_base_device_slot_ptrs {
        if mixed_resident_sparse {
            if qwen35_selected_base_resident_admission_enabled() {
                state.qwen35_admit_selected_base_resident_pages_by_token(
                    gate_all,
                    up_all,
                    down_all,
                    &route_pack.expert_ids,
                    &route_pack.route_weights,
                    down_quant,
                    n_ff,
                    n_embd,
                    token_count,
                )?;
            }
            let prepared = if device_input_use_mixed_device_slot_ptrs(auto_mixed_resident_sparse) {
                state.qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
                    gate_all,
                    up_all,
                    down_all,
                    &route_pack.expert_ids,
                    down_quant,
                    n_ff,
                    n_embd,
                )?
            } else {
                state.qwen35_prepare_selected_base_mixed_resident_temp_slots_by_token(
                    gate_all,
                    up_all,
                    down_all,
                    &route_pack.expert_ids,
                    down_quant,
                    n_ff,
                    n_embd,
                )?
            };
            (Some(prepared), None)
        } else if fused_selected_sparse_boundary || overlap_selected_base_staging {
            (
                None,
                Some(DeferredQwen35SelectedBaseSparse {
                    gate_all,
                    up_all,
                    down_all,
                    expert_ids: &route_pack.expert_ids,
                    down_quant,
                    n_ff,
                    n_embd,
                }),
            )
        } else {
            (
                Some(
                    state.qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
                        gate_all,
                        up_all,
                        down_all,
                        &route_pack.expert_ids,
                        down_quant,
                        n_ff,
                        n_embd,
                    )?,
                ),
                None,
            )
        }
    } else if device_input_selected_base_temp_slab_enabled() {
        (
            Some(state.qwen35_prepare_selected_base_temp_slab_slots_by_token(
                gate_all,
                up_all,
                down_all,
                &route_pack.expert_ids,
                down_quant,
                n_ff,
                n_embd,
            )?),
            None,
        )
    } else {
        (None, None)
    };
    if let Some(start) = phase_start {
        log_qwen35_device_moe_api_phase(
            "selected_temp_slab",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    let empty_sparse_slots: [&[u8]; 0] = [];
    let (gate_weights, up_weights, down_weights) = match selected.as_ref() {
        Some(selected) => (
            selected.gate_weights.as_slice(),
            selected.up_weights.as_slice(),
            selected.down_weights.as_slice(),
        ),
        None => (
            &empty_sparse_slots[..],
            &empty_sparse_slots[..],
            &empty_sparse_slots[..],
        ),
    };
    let phase_start = phase_profile.then(std::time::Instant::now);
    let output_id = if reuse_residual_output {
        state.qwen35_prefill_moe_f32_shared_sparse_by_token_device_input_reuse_residual(
            shared_gate,
            shared_up,
            shared_down,
            shared_input_scale,
            gate_weights,
            up_weights,
            down_weights,
            &route_pack.expert_ids,
            &route_pack.route_weights,
            &route_pack.token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            prepared_sparse,
            deferred_selected_base,
        )?
    } else {
        state.qwen35_prefill_moe_f32_shared_sparse_by_token_device_input(
            shared_gate,
            shared_up,
            shared_down,
            shared_input_scale,
            gate_weights,
            up_weights,
            down_weights,
            &route_pack.expert_ids,
            &route_pack.route_weights,
            &route_pack.token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            prepared_sparse,
            deferred_selected_base,
        )?
    };
    if let Some(start) = phase_start {
        log_qwen35_device_moe_api_phase(
            "shared_sparse_device",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    if let Some(start) = total_start {
        log_qwen35_device_moe_api_phase(
            "total",
            token_count,
            n_expert,
            n_expert_used,
            start.elapsed(),
        );
    }
    Ok(Some(Qwen35DeviceMoeOutput {
        output_id,
        output_desc: rnb_backend_api::DeviceTensorDesc::new(
            token_count,
            n_embd,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MoeOutput,
        ),
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: &[f32],
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    shared_down_quant: u32,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::qwen35_shared_q4_f32_cache_enabled_for_seq(token_count) {
        return Ok(None);
    }
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 cached shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    validate_qwen35_sparse_token_batch(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
    if expert_ids.len() != gate_weights.len() {
        return Err(format!(
            "Qwen35 cached shared expert id length mismatch: got {}, expected {}",
            expert_ids.len(),
            gate_weights.len()
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
        .qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            shared_down_quant,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    shared_down_quant: u32,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::qwen35_shared_q4_f32_cache_enabled_for_seq(token_count) {
        return Ok(None);
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
        .qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            shared_down_quant,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_q4_shared_sparse_device_topk_selected_base_by_token_cached(
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    token_count: usize,
    n_expert_used: usize,
    shared_down_quant: u32,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::qwen35_shared_q4_f32_cache_enabled_for_seq(token_count) {
        return Ok(None);
    }
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 cached device-topk selected-base shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    let selected_base_n_expert = qwen35_selected_base_full_layer_expert_count(
        gate_all, up_all, down_all, down_quant, n_ff, n_embd,
    )?;
    if selected_base_n_expert != n_expert {
        return Err(format!(
            "Qwen35 cached device-topk selected-base expert count mismatch: router={n_expert} selected_base={selected_base_n_expert}"
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut route_pack = state.qwen35_prefill_device_topk_route_pack(
        router_w,
        n_expert,
        hidden_dim,
        norm_all,
        token_count,
        n_expert_used,
    )?;
    route_pack.sort_by_expert_token()?;
    state.qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
        shared_gate,
        shared_up,
        shared_down,
        shared_route,
        gate_all,
        up_all,
        down_all,
        &route_pack.expert_ids,
        &route_pack.route_weights,
        &route_pack.token_ids,
        token_count,
        shared_down_quant,
        down_quant,
        n_ff,
        n_embd,
        norm_all,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    shared_down_quant: u32,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::qwen35_full_layer_shared_q4_f32_cache_enabled() {
        return Ok(None);
    }
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 cached full-layer shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    validate_qwen35_sparse_full_layer_batch(
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            shared_down_quant,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_selected_base_by_token(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let selected_base_device_slot_ptrs = qwen35_selected_base_device_slot_ptrs_enabled();
    let mixed_resident_sparse =
        selected_base_device_slot_ptrs && qwen35_selected_base_mixed_resident_enabled();
    let direct_sparse = !mixed_resident_sparse
        && selected_base_device_slot_ptrs
        && qwen35_selected_base_direct_sparse_enabled();
    if direct_sparse || mixed_resident_sparse {
        if shared_route.len() != token_count {
            return Err(format!(
                "Qwen35 combined selected-base shared route length mismatch: got {}, expected {token_count}",
                shared_route.len()
            ));
        }
        if shared_gate.len() != n_ff * n_embd
            || shared_up.len() != n_ff * n_embd
            || shared_down.len() != n_embd * n_ff
        {
            return Err(
                "Qwen35 combined selected-base shared f32 weight shape mismatch".to_string(),
            );
        }
        validate_qwen35_sparse_full_layer_batch(
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
        )?;
        let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
        let mut guard = compute
            .lock()
            .map_err(|_| "cuda compute state lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(CudaState::open()?);
        }
        let state = guard.as_mut().expect("cuda compute state initialized");
        let prepared_sparse = if mixed_resident_sparse {
            if qwen35_selected_base_resident_admission_enabled() {
                state.qwen35_admit_selected_base_resident_pages_by_token(
                    gate_all,
                    up_all,
                    down_all,
                    expert_ids,
                    route_weights,
                    down_quant,
                    n_ff,
                    n_embd,
                    token_count,
                )?;
            }
            if qwen35_selected_base_mixed_device_slot_ptrs_enabled() {
                state.qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
                    gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
                )?
            } else {
                state.qwen35_prepare_selected_base_mixed_resident_temp_slots_by_token(
                    gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
                )?
            }
        } else {
            state.qwen35_prepare_selected_base_residency_aware_device_slot_ptrs_by_token(
                gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
            )?
        };
        let empty_sparse_slots: [&[u8]; 0] = [];
        return state.qwen35_prefill_moe_f32_shared_sparse_by_token_prepared(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            &empty_sparse_slots,
            &empty_sparse_slots,
            &empty_sparse_slots,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
            Some(prepared_sparse),
        );
    }
    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        down_quant,
        n_ff,
        n_embd,
    )?;
    qwen35_prefill_moe_f32_shared_sparse_by_token(
        shared_gate,
        shared_up,
        shared_down,
        shared_route,
        &selected.gate_weights,
        &selected.up_weights,
        &selected.down_weights,
        expert_ids,
        selected.route_weights,
        selected.token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_by_token(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    token_count: usize,
    n_expert_used: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<f32>, String> {
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 device-topk selected-base shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    let selected_base_n_expert = qwen35_selected_base_full_layer_expert_count(
        gate_all, up_all, down_all, down_quant, n_ff, n_embd,
    )?;
    if selected_base_n_expert != n_expert {
        return Err(format!(
            "Qwen35 device-topk selected-base expert count mismatch: router={n_expert} selected_base={selected_base_n_expert}"
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut route_pack = state.qwen35_prefill_device_topk_route_pack(
        router_w,
        n_expert,
        hidden_dim,
        norm_all,
        token_count,
        n_expert_used,
    )?;
    route_pack.sort_by_expert_token()?;
    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        gate_all,
        up_all,
        down_all,
        &route_pack.expert_ids,
        &route_pack.route_weights,
        &route_pack.token_ids,
        down_quant,
        n_ff,
        n_embd,
    )?;
    state.qwen35_prefill_moe_f32_shared_sparse_by_token(
        shared_gate,
        shared_up,
        shared_down,
        shared_route,
        &selected.gate_weights,
        &selected.up_weights,
        &selected.down_weights,
        &route_pack.expert_ids,
        selected.route_weights,
        selected.token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        norm_all,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token(
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    shared_route: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if expert_ids.is_empty() {
        return f32_shared_expert(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            n_ff,
            n_embd,
            input,
        );
    }
    if shared_route.len() != token_count {
        return Err(format!(
            "Qwen35 full-layer combined shared route length mismatch: got {}, expected {token_count}",
            shared_route.len()
        ));
    }
    if shared_gate.len() != n_ff * n_embd
        || shared_up.len() != n_ff * n_embd
        || shared_down.len() != n_embd * n_ff
    {
        return Err("Qwen35 full-layer combined shared f32 weight shape mismatch".to_string());
    }
    validate_qwen35_sparse_full_layer_batch(
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_register_moe_layer(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool, String> {
    if !tuning::moe_layer_cache_enabled() {
        return Ok(false);
    }
    validate_qwen35_moe_layer_weights(gate_all, up_all, down_all, down_quant, n_ff, n_embd)?;
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
        .register_qwen35_moe_layer_without_eviction(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )
}

pub fn qwen35_prefill_device_topk_route_slots(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
    n_expert_used: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let pack = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .qwen35_prefill_device_topk_route_pack(
            router_w,
            n_expert,
            hidden_dim,
            norm_all,
            seq_len,
            n_expert_used,
        )?;
    Ok((pack.expert_ids, pack.route_weights, pack.token_ids))
}

pub fn clear_moe_layer_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_resident_moe_layer_cache()
}

pub fn qwen35_expert(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != n_embd {
        return Err(format!(
            "Qwen35 expert input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 expert dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    if gate_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "Qwen35 expert gate byte mismatch: got {}, expected {}",
            gate_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if up_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "Qwen35 expert up byte mismatch: got {}, expected {}",
            up_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if down_weights.len() != n_embd * down_row_bytes {
        return Err(format!(
            "Qwen35 expert down byte mismatch: got {}, expected {}",
            down_weights.len(),
            n_embd * down_row_bytes
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
        .qwen35_expert(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            false,
        )
}

fn qwen35_iq4xs_down_row_bytes(down_quant: u32, n_ff: usize) -> Result<usize, String> {
    match down_quant {
        12 => Ok((n_ff / 256) * 144),
        13 => Ok((n_ff / 256) * 176),
        14 => Ok((n_ff / 256) * 210),
        23 => Ok((n_ff / 256) * 136),
        other => Err(format!(
            "unsupported Qwen35 CUDA IQ4_XS down quant code {other}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_qwen35_sparse_iq4xs(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<(), String> {
    let selected = gate_weights.len();
    if selected == 0 {
        return Ok(());
    }
    if up_weights.len() != selected
        || down_weights.len() != selected
        || route_weights.len() != selected
    {
        return Err("Qwen35 sparse IQ4_XS expert batch length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse IQ4_XS expert input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 sparse IQ4_XS expert dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 136;
    let down_row_bytes = qwen35_iq4xs_down_row_bytes(down_quant, n_ff)?;
    for (i, weights) in gate_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse IQ4_XS gate[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in up_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse IQ4_XS up[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in down_weights.iter().enumerate() {
        if weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "Qwen35 sparse IQ4_XS down[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_embd * down_row_bytes
            ));
        }
    }
    Ok(())
}

pub fn qwen35_sparse_experts(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let selected = gate_weights.len();
    if selected == 0 {
        return Ok(vec![0.0; n_embd]);
    }
    if up_weights.len() != selected
        || down_weights.len() != selected
        || route_weights.len() != selected
    {
        return Err("Qwen35 sparse expert batch length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse expert input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 sparse expert dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let (gate_row_bytes, down_row_bytes) = match down_quant {
        11 => ((n_embd / 256) * 84, (n_ff / 256) * 110),
        12 => ((n_embd / 256) * 144, (n_ff / 256) * 144),
        13 => ((n_embd / 256) * 144, (n_ff / 256) * 176),
        14 => ((n_embd / 256) * 144, (n_ff / 256) * 210),
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    for (i, weights) in gate_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse gate[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in up_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse up[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in down_weights.iter().enumerate() {
        if weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "Qwen35 sparse down[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_embd * down_row_bytes
            ));
        }
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
        .qwen35_sparse_experts(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_prepare_selected_bundle_residency(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<bool>, String> {
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
        .qwen35_prepare_selected_bundle_residency(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            n_ff,
            n_embd,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_per_slot_resident(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
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
        .qwen35_sparse_experts_per_slot_resident(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_iq4xs(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if gate_weights.is_empty() {
        return Ok(vec![0.0; n_embd]);
    }
    validate_qwen35_sparse_iq4xs(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_sparse_experts_iq4xs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_into(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    let selected = gate_weights.len();
    if selected == 0 {
        output.fill(0.0);
        return Ok(());
    }
    if up_weights.len() != selected
        || down_weights.len() != selected
        || route_weights.len() != selected
    {
        return Err("Qwen35 sparse expert output batch length mismatch".to_string());
    }
    if input.len() != n_embd || output.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse expert output shape mismatch: input={} output={} expected={n_embd}",
            input.len(),
            output.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 sparse expert output dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let (gate_row_bytes, down_row_bytes) = match down_quant {
        11 => ((n_embd / 256) * 84, (n_ff / 256) * 110),
        12 => ((n_embd / 256) * 144, (n_ff / 256) * 144),
        13 => ((n_embd / 256) * 144, (n_ff / 256) * 176),
        14 => ((n_embd / 256) * 144, (n_ff / 256) * 210),
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    for (i, weights) in gate_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse output gate[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in up_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse output up[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in down_weights.iter().enumerate() {
        if weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "Qwen35 sparse output down[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_embd * down_row_bytes
            ));
        }
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
        .qwen35_sparse_experts_into(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
            output,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_iq4xs_into(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if gate_weights.is_empty() {
        output.fill(0.0);
        return Ok(());
    }
    if output.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse IQ4_XS output length mismatch: got {}, expected {n_embd}",
            output.len()
        ));
    }
    validate_qwen35_sparse_iq4xs(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_sparse_experts_iq4xs_into(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            output,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_add_residual_into(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> Result<(), String> {
    let selected = gate_weights.len();
    if selected == 0 {
        return Ok(());
    }
    if up_weights.len() != selected
        || down_weights.len() != selected
        || route_weights.len() != selected
    {
        return Err("Qwen35 sparse residual batch length mismatch".to_string());
    }
    if input.len() != n_embd || residual.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse residual shape mismatch: input={} residual={} expected={n_embd}",
            input.len(),
            residual.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 sparse residual dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
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
        .qwen35_sparse_experts_add_residual_into(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
            residual,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_iq4xs_add_residual_into(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> Result<(), String> {
    if gate_weights.is_empty() {
        return Ok(());
    }
    if residual.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse IQ4_XS residual length mismatch: got {}, expected {n_embd}",
            residual.len()
        ));
    }
    validate_qwen35_sparse_iq4xs(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_sparse_experts_iq4xs_add_residual_into(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            residual,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_decode_moe_shared_sparse_into(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: u32,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: f32,
    shared_down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    validate_qwen35_sparse_token_batch(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        &vec![0; gate_weights.len()],
        1,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
    validate_qwen35_sparse_token_batch(
        &[shared_gate],
        &[shared_up],
        &[shared_down],
        &[shared_route],
        &[0],
        1,
        shared_down_quant,
        n_ff,
        n_embd,
        input,
    )?;
    if output.len() != n_embd {
        return Err(format!(
            "Qwen35 shared+sparse output length mismatch: got {}, expected {n_embd}",
            output.len()
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
        .qwen35_decode_moe_shared_sparse_into(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            shared_down_quant,
            n_ff,
            n_embd,
            input,
            output,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_device_roundtrip(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let selected = gate_weights.len();
    if selected == 0 {
        return Ok(vec![0.0; n_embd]);
    }
    if up_weights.len() != selected
        || down_weights.len() != selected
        || route_weights.len() != selected
    {
        return Err("Qwen35 sparse expert device batch length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Qwen35 sparse expert device input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 sparse expert device dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    for (i, weights) in gate_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse device gate[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in up_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 sparse device up[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in down_weights.iter().enumerate() {
        if weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "Qwen35 sparse device down[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_embd * down_row_bytes
            ));
        }
    }

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let input_dev = state.compute_input_ptr(std::mem::size_of_val(input))?;
    let output_bytes = n_embd * std::mem::size_of::<f32>();
    let output_dev = state.compute_output_ptr(output_bytes)?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
    }
    state.qwen35_sparse_experts_to_dev(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        layer_idx,
        selected_expert_ids,
        bundle_observation_receipt,
        down_quant,
        n_ff,
        n_embd,
        input_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; n_embd];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            output_bytes,
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_by_token(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if gate_weights.is_empty() {
        return Ok(vec![0.0; token_count * n_embd]);
    }
    validate_qwen35_sparse_token_batch(
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )?;
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
        .qwen35_sparse_experts_by_token(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_experts_selected_base_by_token(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        down_quant,
        n_ff,
        n_embd,
    )?;
    qwen35_sparse_experts_by_token(
        &selected.gate_weights,
        &selected.up_weights,
        &selected.down_weights,
        selected.route_weights,
        selected.token_ids,
        token_count,
        down_quant,
        n_ff,
        n_embd,
        input,
    )
}
