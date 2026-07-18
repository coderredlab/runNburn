use super::super::*;

fn inactive_nemotron_prefill_workspace_summary() -> NemotronPrefillWorkspaceSummary {
    NemotronPrefillWorkspaceSummary {
        active: false,
        arena_bytes: 0,
        live_leases: 0,
        hit_bytes: 0,
        miss_bytes: 0,
        owned_alloc_count: 0,
    }
}

pub fn begin_nemotron_prefill_workspace(
    config: NemotronPrefillWorkspaceConfig,
) -> Result<NemotronPrefillWorkspaceSummary, String> {
    if !config.enabled {
        return Ok(inactive_nemotron_prefill_workspace_summary());
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
        .begin_nemotron_prefill_workspace(config)
}

pub fn end_nemotron_prefill_workspace() -> Result<NemotronPrefillWorkspaceSummary, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.end_nemotron_prefill_workspace()
}
