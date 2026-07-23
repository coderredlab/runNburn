use super::platform_runtime::{
    apply_worker_affinity, current_cpu_affinity_cores, plan_cpu_runtime_threads,
    read_allowed_cpu_list,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuAssistWorkload {
    Default,
    HybridMoe,
    WideMoe,
}

pub(super) fn configure_cpu_runtime(
    architecture: Option<rnb_loader::Architecture>,
) -> crate::error::Result<()> {
    let cpu_affinity = super::policy::cpu_affinity();
    let user_set = super::policy::rayon_num_threads();
    let available_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let cpu_assist_workload = cpu_assist_workload_for_architecture(architecture);
    if matches!(cpu_assist_workload, CpuAssistWorkload::WideMoe)
        && super::policy::moe_prefill_full_cores_enabled()
    {
        let worker_cores = read_allowed_cpu_list()
            .or_else(current_cpu_affinity_cores)
            .ok_or_else(|| {
                crate::error::LlmError::ModelLoad(
                    "failed to read the full-core prefill affinity mask".to_string(),
                )
            })?;
        let worker_count = worker_cores.len();
        super::cpu_phase_runtime::configure_full_prefill_pool(worker_cores)
            .map_err(crate::error::LlmError::ModelLoad)?;
        eprintln!("[INFO] MoE prefill full-core pool: {worker_count} workers");
    }
    let thread_plan = plan_cpu_runtime_threads(
        cpu_affinity.as_deref(),
        super::policy::big_cores_requested(),
        false,
        false,
        user_set.as_deref(),
        available_parallelism,
    );
    let requested_threads = thread_plan.requested_threads;
    let actual_threads = {
        let mut builder = rayon::ThreadPoolBuilder::new();
        builder = builder.num_threads(requested_threads);
        if let Some(cores) = thread_plan.worker_affinity_cores.clone() {
            let worker_cores = std::sync::Arc::new(cores);
            builder = builder.start_handler(move |worker_index| {
                if let Err(err) =
                    apply_worker_affinity("rayon-worker", worker_index, worker_cores.as_slice())
                {
                    eprintln!("[WARN] {} (worker {})", err, worker_index);
                }
            });
        }
        match builder.build_global() {
            Ok(()) => requested_threads,
            Err(e) => {
                let current = rayon::current_num_threads();
                eprintln!(
                    "[WARN] Rayon global pool unchanged: {} (requested {}, actual {})",
                    e, requested_threads, current
                );
                current
            }
        }
    };
    if actual_threads == requested_threads {
        eprintln!("[INFO] Rayon threads: {}", actual_threads);
    } else {
        eprintln!(
            "[INFO] Rayon threads: requested {}, actual {}",
            requested_threads, actual_threads
        );
    }
    Ok(())
}

fn cpu_assist_workload_for_architecture(
    architecture: Option<rnb_loader::Architecture>,
) -> CpuAssistWorkload {
    match architecture {
        Some(rnb_loader::Architecture::NemotronHMoE) => CpuAssistWorkload::HybridMoe,
        Some(rnb_loader::Architecture::Qwen35MoE) => CpuAssistWorkload::WideMoe,
        _ => CpuAssistWorkload::Default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_uses_hybrid_moe_cpu_assist_workload() {
        assert_eq!(
            cpu_assist_workload_for_architecture(Some(rnb_loader::Architecture::NemotronHMoE)),
            CpuAssistWorkload::HybridMoe
        );
        assert_eq!(
            cpu_assist_workload_for_architecture(Some(rnb_loader::Architecture::Qwen35MoE)),
            CpuAssistWorkload::WideMoe
        );
        assert_eq!(
            cpu_assist_workload_for_architecture(None),
            CpuAssistWorkload::Default
        );
    }
}
