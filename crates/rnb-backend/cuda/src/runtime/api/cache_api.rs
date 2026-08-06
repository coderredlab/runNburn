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

fn selected_moe_transient_required_bytes(
    gate_up_weight_bytes: usize,
    down_weight_bytes: usize,
    n_embd: usize,
    n_ff: usize,
) -> usize {
    let weight_bytes = gate_up_weight_bytes.max(down_weight_bytes);
    let input_bytes = n_embd.max(n_ff).saturating_mul(std::mem::size_of::<f32>());
    let output_bytes = n_ff
        .saturating_mul(2)
        .max(n_embd)
        .saturating_mul(std::mem::size_of::<f32>());
    weight_bytes
        .saturating_add(input_bytes)
        .saturating_add(output_bytes)
}

pub fn gemma4_selected_moe_admitted(
    gate_up_weight_bytes: usize,
    down_weight_bytes: usize,
    n_embd: usize,
    n_ff: usize,
) -> Result<bool, String> {
    let required_bytes = selected_moe_transient_required_bytes(
        gate_up_weight_bytes,
        down_weight_bytes,
        n_embd,
        n_ff,
    );

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

/// Ends model-owned CUDA cache lifetimes while preserving the loaded CUDA
/// context and modules. Allocation-identity F32 entries are removed explicitly
/// so a later engine cannot reuse weights from the previous model generation.
pub fn reset_state_for_engine_init() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    state.clear_resident_moe_layer_cache()?;
    state.clear_resident_q4k_cache()?;
    state.clear_moe_slice_cache()?;
    state.clear_stable_resident_f32_sources()?;
    state.clear_resident_delta_states()
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

#[cfg(test)]
mod tests {
    use super::{reset_state_for_engine_init, selected_moe_transient_required_bytes};
    use crate::runtime::{cuda_test_env_lock, CudaState, DEFAULT_CUDA_COMPUTE};
    use std::sync::Mutex;

    #[test]
    fn selected_moe_transient_budget_uses_largest_weight_and_shape_scaled_scratch() {
        assert_eq!(
            selected_moe_transient_required_bytes(3_000, 5_000, 100, 200),
            5_000 + 200 * 4 + 400 * 4
        );
        assert_eq!(
            selected_moe_transient_required_bytes(7_000, 5_000, 300, 100),
            7_000 + 300 * 4 + 300 * 4
        );
    }

    #[test]
    fn engine_init_reset_ends_stable_f32_source_generation() {
        let _guard = cuda_test_env_lock();
        reset_state_for_engine_init().expect("reset CUDA state before test");

        fn read_stable_source(data: &[f32]) -> Result<Vec<f32>, String> {
            let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
            let mut guard = compute
                .lock()
                .map_err(|_| "cuda compute state lock poisoned".to_string())?;
            if guard.is_none() {
                *guard = Some(CudaState::open()?);
            }
            let state = guard.as_mut().expect("cuda compute state initialized");
            let ptr = state.resident_f32_ptr_stable_source(data)?;
            let mut output = vec![0.0f32; data.len()];
            unsafe {
                state.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    ptr,
                    std::mem::size_of_val(data),
                    state.stream,
                )?;
            }
            state.stream_synchronize()?;
            Ok(output)
        }

        let mut source = vec![1.0f32, 2.0, 3.0, 4.0];
        let first = match read_stable_source(&source) {
            Ok(values) => values,
            Err(err) => {
                eprintln!("skipping CUDA stable F32 reset test: {err}");
                return;
            }
        };
        assert_eq!(first, source);

        source.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        reset_state_for_engine_init().expect("reset CUDA state between model generations");
        let second = read_stable_source(&source).expect("reload stable F32 source");
        assert_eq!(second, source);
    }
}
