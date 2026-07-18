use super::super::*;
use rnb_core::tensor::FileBackedRegion;

#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_experts_iq2xxs_iq3xxs(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slot_count = gate_weights.len();
    if up_weights.len() != slot_count
        || down_weights.len() != slot_count
        || route_weights.len() != slot_count
    {
        return Err(format!(
            "GLM sparse expert slot mismatch: gate={} up={} down={} route={}",
            slot_count,
            up_weights.len(),
            down_weights.len(),
            route_weights.len()
        ));
    }

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut primary_guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if primary_guard.is_none() {
        *primary_guard = Some(CudaState::open()?);
    }
    let primary_state = primary_guard
        .as_mut()
        .expect("cuda compute state initialized");

    if !tuning::glm_expert_parallel_enabled() || slot_count < 2 {
        return primary_state.glm_sparse_experts_iq2xxs_iq3xxs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            n_ff,
            n_embd,
            input,
        );
    }

    let primary_ordinal = CudaState::configured_device_ordinal();
    let secondary_ordinal = tuning::glm_expert_parallel_secondary_device(primary_ordinal);
    if secondary_ordinal == primary_ordinal {
        return Err(format!(
            "GLM expert parallel requires distinct CUDA ordinals, both resolved to {primary_ordinal}"
        ));
    }

    let mut secondary_guard = GLM_EXPERT_PARALLEL_CUDA_COMPUTE
        .lock()
        .map_err(|_| "GLM secondary cuda compute state lock poisoned".to_string())?;
    if secondary_guard.is_none() {
        let secondary_state = CudaState::open_ordinal(secondary_ordinal);
        let primary_restore = primary_state.set_current();
        let secondary_state = match (secondary_state, primary_restore) {
            (Ok(state), Ok(())) => state,
            (Err(open_error), Ok(())) => {
                return Err(format!(
                    "opening GLM secondary CUDA device failed: {open_error}"
                ));
            }
            (Ok(_), Err(restore_error)) => {
                return Err(format!(
                    "restoring GLM primary CUDA context failed: {restore_error}"
                ));
            }
            (Err(open_error), Err(restore_error)) => {
                return Err(format!(
                    "opening GLM secondary CUDA device failed: {open_error}; restoring primary context also failed: {restore_error}"
                ));
            }
        };
        *secondary_guard = Some(secondary_state);
    }
    let secondary_state = secondary_guard
        .as_mut()
        .expect("GLM secondary cuda compute state initialized");
    let primary_slots = tuning::glm_expert_parallel_primary_slots(slot_count);

    GLM_EXPERT_PARALLEL_LOGGED.get_or_init(|| {
        eprintln!(
            "[cuda] GLM expert parallel enabled: primary_ordinal={primary_ordinal} primary_slots={primary_slots} secondary_ordinal={secondary_ordinal} secondary_slots={}",
            slot_count - primary_slots
        );
    });

    let (mut primary_output, secondary_output) =
        std::thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>), String> {
            let secondary_handle = scope.spawn(|| {
                secondary_state.glm_sparse_experts_iq2xxs_iq3xxs(
                    &gate_weights[primary_slots..],
                    &up_weights[primary_slots..],
                    &down_weights[primary_slots..],
                    &route_weights[primary_slots..],
                    n_ff,
                    n_embd,
                    input,
                )
            });
            let primary_result = primary_state.glm_sparse_experts_iq2xxs_iq3xxs(
                &gate_weights[..primary_slots],
                &up_weights[..primary_slots],
                &down_weights[..primary_slots],
                &route_weights[..primary_slots],
                n_ff,
                n_embd,
                input,
            );
            let secondary_result = secondary_handle
                .join()
                .map_err(|_| "GLM secondary CUDA worker panicked".to_string())?;
            let primary_output =
                primary_result.map_err(|err| format!("GLM primary CUDA device failed: {err}"))?;
            let secondary_output = secondary_result
                .map_err(|err| format!("GLM secondary CUDA device failed: {err}"))?;
            Ok((primary_output, secondary_output))
        })?;

    if primary_output.len() != secondary_output.len() {
        return Err(format!(
            "GLM expert parallel output mismatch: primary={} secondary={}",
            primary_output.len(),
            secondary_output.len()
        ));
    }
    for (primary, secondary) in primary_output.iter_mut().zip(secondary_output) {
        *primary += secondary;
    }
    Ok(primary_output)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_experts_iq2xxs_iq3xxs_by_token(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    file_regions: Option<&[FileBackedRegion; 3]>,
    direct_file: bool,
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slot_count = gate_weights.len();
    if token_count == 0 || slot_count == 0 || slot_count % token_count != 0 {
        return Err(format!(
            "GLM batched sparse slots must be non-zero and divisible by token_count: slots={slot_count} token_count={token_count}"
        ));
    }
    if up_weights.len() != slot_count
        || down_weights.len() != slot_count
        || route_weights.len() != slot_count
        || token_ids.len() != slot_count
    {
        return Err(format!(
            "GLM batched sparse slot mismatch: gate={} up={} down={} route={} token_ids={}",
            slot_count,
            up_weights.len(),
            down_weights.len(),
            route_weights.len(),
            token_ids.len()
        ));
    }

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut primary_guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if primary_guard.is_none() {
        *primary_guard = Some(CudaState::open()?);
    }
    let primary_state = primary_guard
        .as_mut()
        .expect("cuda compute state initialized");

    let slots_per_token = slot_count / token_count;
    if !tuning::glm_expert_parallel_enabled() || slots_per_token < 2 {
        return primary_state.glm_sparse_experts_iq2xxs_iq3xxs_by_token(
            gate_weights,
            up_weights,
            down_weights,
            file_regions,
            direct_file,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
        );
    }

    let primary_ordinal = CudaState::configured_device_ordinal();
    let secondary_ordinal = tuning::glm_expert_parallel_secondary_device(primary_ordinal);
    if secondary_ordinal == primary_ordinal {
        return Err(format!(
            "GLM expert parallel requires distinct CUDA ordinals, both resolved to {primary_ordinal}"
        ));
    }
    let mut secondary_guard = GLM_EXPERT_PARALLEL_CUDA_COMPUTE
        .lock()
        .map_err(|_| "GLM secondary cuda compute state lock poisoned".to_string())?;
    if secondary_guard.is_none() {
        let secondary_state = CudaState::open_ordinal(secondary_ordinal);
        let primary_restore = primary_state.set_current();
        let secondary_state = match (secondary_state, primary_restore) {
            (Ok(state), Ok(())) => state,
            (Err(open_error), Ok(())) => {
                return Err(format!(
                    "opening GLM secondary CUDA device failed: {open_error}"
                ));
            }
            (Ok(_), Err(restore_error)) => {
                return Err(format!(
                    "restoring GLM primary CUDA context failed: {restore_error}"
                ));
            }
            (Err(open_error), Err(restore_error)) => {
                return Err(format!(
                    "opening GLM secondary CUDA device failed: {open_error}; restoring primary context also failed: {restore_error}"
                ));
            }
        };
        *secondary_guard = Some(secondary_state);
    }
    let secondary_state = secondary_guard
        .as_mut()
        .expect("GLM secondary cuda compute state initialized");

    let primary_slots_per_token = tuning::glm_expert_parallel_primary_slots(slots_per_token);
    let secondary_slots_per_token = slots_per_token - primary_slots_per_token;
    let mut primary_gate = Vec::with_capacity(token_count * primary_slots_per_token);
    let mut primary_up = Vec::with_capacity(token_count * primary_slots_per_token);
    let mut primary_down = Vec::with_capacity(token_count * primary_slots_per_token);
    let mut primary_route = Vec::with_capacity(token_count * primary_slots_per_token);
    let mut primary_token_ids = Vec::with_capacity(token_count * primary_slots_per_token);
    let mut secondary_gate = Vec::with_capacity(token_count * secondary_slots_per_token);
    let mut secondary_up = Vec::with_capacity(token_count * secondary_slots_per_token);
    let mut secondary_down = Vec::with_capacity(token_count * secondary_slots_per_token);
    let mut secondary_route = Vec::with_capacity(token_count * secondary_slots_per_token);
    let mut secondary_token_ids = Vec::with_capacity(token_count * secondary_slots_per_token);
    for token in 0..token_count {
        let start = token * slots_per_token;
        let split = start + primary_slots_per_token;
        let end = start + slots_per_token;
        primary_gate.extend_from_slice(&gate_weights[start..split]);
        primary_up.extend_from_slice(&up_weights[start..split]);
        primary_down.extend_from_slice(&down_weights[start..split]);
        primary_route.extend_from_slice(&route_weights[start..split]);
        primary_token_ids.extend_from_slice(&token_ids[start..split]);
        secondary_gate.extend_from_slice(&gate_weights[split..end]);
        secondary_up.extend_from_slice(&up_weights[split..end]);
        secondary_down.extend_from_slice(&down_weights[split..end]);
        secondary_route.extend_from_slice(&route_weights[split..end]);
        secondary_token_ids.extend_from_slice(&token_ids[split..end]);
    }

    GLM_EXPERT_PARALLEL_LOGGED.get_or_init(|| {
        eprintln!(
            "[cuda] GLM expert parallel enabled: primary_ordinal={primary_ordinal} primary_slots={primary_slots_per_token} secondary_ordinal={secondary_ordinal} secondary_slots={secondary_slots_per_token}"
        );
    });
    let (mut primary_output, secondary_output) =
        std::thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>), String> {
            let secondary_handle = scope.spawn(|| {
                secondary_state.glm_sparse_experts_iq2xxs_iq3xxs_by_token(
                    &secondary_gate,
                    &secondary_up,
                    &secondary_down,
                    file_regions,
                    direct_file,
                    &secondary_route,
                    &secondary_token_ids,
                    token_count,
                    n_ff,
                    n_embd,
                    input,
                )
            });
            let primary_result = primary_state.glm_sparse_experts_iq2xxs_iq3xxs_by_token(
                &primary_gate,
                &primary_up,
                &primary_down,
                file_regions,
                direct_file,
                &primary_route,
                &primary_token_ids,
                token_count,
                n_ff,
                n_embd,
                input,
            );
            let secondary_result = secondary_handle
                .join()
                .map_err(|_| "GLM secondary CUDA worker panicked".to_string())?;
            let primary_output =
                primary_result.map_err(|err| format!("GLM primary CUDA device failed: {err}"))?;
            let secondary_output = secondary_result
                .map_err(|err| format!("GLM secondary CUDA device failed: {err}"))?;
            Ok((primary_output, secondary_output))
        })?;
    if primary_output.len() != secondary_output.len() {
        return Err(format!(
            "GLM expert parallel output mismatch: primary={} secondary={}",
            primary_output.len(),
            secondary_output.len()
        ));
    }
    for (primary, secondary) in primary_output.iter_mut().zip(secondary_output) {
        *primary += secondary;
    }
    Ok(primary_output)
}

pub fn glm_shared_expert_q5k_q6k(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != n_embd {
        return Err(format!(
            "GLM shared expert input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "GLM shared expert dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_bytes = n_ff * (n_embd / 256) * 176;
    let down_bytes = n_embd * (n_ff / 256) * 210;
    if gate_weights.len() != gate_bytes || up_weights.len() != gate_bytes {
        return Err(format!(
            "GLM shared Q5_K gate/up byte mismatch: gate={} up={} expected={gate_bytes}",
            gate_weights.len(),
            up_weights.len()
        ));
    }
    if down_weights.len() != down_bytes {
        return Err(format!(
            "GLM shared Q6_K down byte mismatch: got {}, expected {down_bytes}",
            down_weights.len()
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
        .glm_shared_expert_q5k_q6k(gate_weights, up_weights, down_weights, n_ff, n_embd, input)
}
