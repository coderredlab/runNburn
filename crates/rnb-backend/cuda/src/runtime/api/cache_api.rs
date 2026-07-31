use super::super::*;

fn cache_env_enabled_or(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

pub fn cuda_memory_info() -> Result<CudaMemoryInfo, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if let Some(state) = guard.as_ref() {
        let (free_bytes, total_bytes) = unsafe { state.api.mem_get_info() }?;
        return Ok(CudaMemoryInfo {
            free_bytes,
            total_bytes,
        });
    }
    drop(guard);

    let state = CudaState::open()?;
    let (free_bytes, total_bytes) = unsafe { state.api.mem_get_info() }?;
    Ok(CudaMemoryInfo {
        free_bytes,
        total_bytes,
    })
}

pub fn gemma4_selected_moe_admitted(
    gate_up_weight_bytes: usize,
    down_weight_bytes: usize,
    n_embd: usize,
    n_ff: usize,
) -> Result<bool, String> {
    let weight_bytes = gate_up_weight_bytes.max(down_weight_bytes);
    let input_bytes = n_embd.max(n_ff).saturating_mul(std::mem::size_of::<f32>());
    let output_bytes = n_ff
        .saturating_mul(2)
        .max(n_embd)
        .saturating_mul(std::mem::size_of::<f32>());
    let required_bytes = weight_bytes
        .saturating_add(input_bytes)
        .saturating_add(output_bytes);

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_ref()
        .expect("cuda compute state initialized")
        .selected_moe_transient_admission_allowed(required_bytes)
}

pub fn cuda_weight_residency_counters() -> Result<CudaWeightResidencyCounters, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    Ok(guard
        .as_ref()
        .map(CudaState::weight_residency_counters)
        .unwrap_or_default())
}

pub fn clear_q4k_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_resident_q4k_cache()
}

pub fn clear_q4_f32_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_resident_q4_f32_cache()
}

pub fn clear_decode_attention_kv_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_decode_attention_kv_cache()
}

pub fn clear_host_registered_ranges() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut primary_guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let primary_result = primary_guard
        .as_mut()
        .map(CudaState::clear_host_registered_ranges)
        .unwrap_or(Ok(()));

    let mut secondary_guard = GLM_EXPERT_PARALLEL_CUDA_COMPUTE
        .lock()
        .map_err(|_| "GLM secondary cuda compute state lock poisoned".to_string())?;
    let secondary_result = secondary_guard
        .as_mut()
        .map(CudaState::clear_host_registered_ranges)
        .unwrap_or(Ok(()));

    primary_result?;
    secondary_result
}

pub fn clear_sequence_state_cache() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_decode_attention_kv_cache()?;
    state.clear_resident_delta_states()
}

pub fn release_q4_f32_after_prefill() -> Result<(), String> {
    if !tuning::q4k_prefill_f32_gemm_enabled() || !tuning::q4_f32_release_after_prefill_enabled() {
        return Ok(());
    }
    clear_q4_f32_cache()
}

pub fn release_q8_0_prefill_f32_after_prefill() -> Result<(), String> {
    if !cache_env_enabled_or("RNB_CUDA_Q8_0_PREFILL_F32_GEMM", false)
        || !cache_env_enabled_or("RNB_CUDA_Q8_0_RELEASE_AFTER_PREFILL", true)
    {
        return Ok(());
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_resident_q8_prefill_projection_cache()
}
