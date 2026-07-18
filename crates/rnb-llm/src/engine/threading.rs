use super::platform_runtime::{
    apply_worker_affinity, plan_model_cpu_runtime_threads, CpuAssistWorkload,
};

pub(super) fn configure_cpu_runtime(
    path: &std::path::Path,
    moe_section_decode_sidecar: bool,
    architecture: Option<rnb_loader::Architecture>,
) {
    let cpu_affinity = super::policy::cpu_affinity();
    let user_set = super::policy::rayon_num_threads();
    let force_gguf = super::policy::force_gguf_enabled();
    let available_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let cpu_assist_workload = cpu_assist_workload_for_architecture(architecture);
    let desktop_cpu_assist_threads = force_gguf
        && (cfg!(feature = "cuda")
            || (cfg!(feature = "metal")
                && matches!(cpu_assist_workload, CpuAssistWorkload::WideMoe)));
    let thread_plan = plan_model_cpu_runtime_threads(
        path,
        moe_section_decode_sidecar,
        cpu_affinity.as_deref(),
        super::policy::big_cores_requested(),
        force_gguf,
        user_set.as_deref(),
        available_parallelism,
        desktop_cpu_assist_threads,
        cpu_assist_workload,
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
