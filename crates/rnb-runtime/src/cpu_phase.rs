use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, OnceLock};

static FULL_PREFILL_WORKER_CORES: OnceLock<Vec<usize>> = OnceLock::new();

fn build_full_prefill_pool(
    worker_count: usize,
    affinity_cores: Option<&[usize]>,
) -> Result<ThreadPool, String> {
    if worker_count == 0 {
        return Err("full-core prefill pool requires at least one worker".to_string());
    }

    let mut builder = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .thread_name(|index| format!("rnb-prefill-{index}"));
    if let Some(affinity_cores) = affinity_cores {
        let affinity_cores = Arc::new(affinity_cores.to_vec());
        builder = builder.start_handler(move |worker_index| {
            if let Err(error) = rnb_platform::android::set_cpu_affinity(
                "full-prefill-worker",
                affinity_cores.as_slice(),
            ) {
                eprintln!("[WARN] {error} (worker {worker_index})");
            }
        });
    }
    builder
        .build()
        .map_err(|error| format!("failed to build full-core prefill pool: {error}"))
}

/// Saves the broad affinity mask before Android narrows the caller and global
/// Rayon workers to the decode-oriented big-core mask.
pub fn configure_full_prefill_pool(worker_cores: Vec<usize>) -> Result<(), String> {
    if worker_cores.is_empty() {
        return Err("full-core prefill affinity mask is empty".to_string());
    }
    match FULL_PREFILL_WORKER_CORES.set(worker_cores) {
        Ok(()) => Ok(()),
        Err(worker_cores) if FULL_PREFILL_WORKER_CORES.get() == Some(&worker_cores) => Ok(()),
        Err(_) => Err("full-core prefill affinity mask changed after configuration".to_string()),
    }
}

pub fn install_full_prefill<R: Send>(operation: impl FnOnce() -> R + Send) -> Result<R, String> {
    let worker_cores = FULL_PREFILL_WORKER_CORES
        .get()
        .ok_or_else(|| "full-core prefill pool was not configured".to_string())?;
    let pool = build_full_prefill_pool(worker_cores.len(), Some(worker_cores))?;
    Ok(pool.install(operation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedicated_pool_runs_nested_rayon_work_on_requested_workers() {
        let pool = build_full_prefill_pool(2, None).unwrap();
        assert_eq!(pool.install(rayon::current_num_threads), 2);
    }

    #[test]
    fn dedicated_pool_rejects_zero_workers() {
        assert!(build_full_prefill_pool(0, None).is_err());
    }
}
