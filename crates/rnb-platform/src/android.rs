use std::path::Path;

pub fn is_current() -> bool {
    cfg!(target_os = "android")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoAffinityDecision {
    pub pin_cores: Option<Vec<usize>>,
    pub thread_count_hint: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerAffinityPlan {
    pub worker_cores: Vec<usize>,
    pub thread_count_hint: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuAffinityMode {
    Auto,
    All,
    Big,
    Little,
    List,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuAssistWorkload {
    Default,
    HybridMoe,
    WideMoe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuRuntimeThreadPlan {
    pub requested_threads: usize,
    pub worker_affinity_cores: Option<Vec<usize>>,
}

pub fn parse_cpu_list(explicit: &str) -> Option<Vec<usize>> {
    let cores: Vec<usize> = explicit
        .split(',')
        .map(str::trim)
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    if cores.is_empty() {
        None
    } else {
        Some(cores)
    }
}

pub fn cpu_affinity_mode_from_env(
    explicit: Option<&str>,
    legacy_big_cores: bool,
) -> (CpuAffinityMode, bool) {
    match explicit.map(str::trim).map(str::to_ascii_lowercase) {
        Some(value) if value == "auto" => (CpuAffinityMode::Auto, true),
        Some(value) if value == "big" => (CpuAffinityMode::Big, true),
        Some(value) if value == "little" => (CpuAffinityMode::Little, true),
        Some(value) if value == "all" => (CpuAffinityMode::All, true),
        Some(value) if parse_cpu_list(&value).is_some() => (CpuAffinityMode::List, true),
        Some(_) => (CpuAffinityMode::All, true),
        None if legacy_big_cores => (CpuAffinityMode::Big, false),
        None => (default_affinity_mode_for_target(), false),
    }
}

fn default_affinity_mode_for_target() -> CpuAffinityMode {
    if cfg!(target_arch = "aarch64") {
        CpuAffinityMode::Auto
    } else {
        CpuAffinityMode::All
    }
}

pub fn parse_allowed_cpu_list(raw: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>().ok()?;
            let end = end.parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.parse::<usize>().ok()?);
        }
    }
    if cpus.is_empty() {
        None
    } else {
        Some(cpus)
    }
}

pub fn filter_cores_by_allowed_list(selected: &[usize], allowed: &[usize]) -> Vec<usize> {
    selected
        .iter()
        .copied()
        .filter(|core| allowed.contains(core))
        .collect()
}

pub fn effective_worker_affinity_cores(requested: &[usize], actual: &[usize]) -> Vec<usize> {
    let effective = filter_cores_by_allowed_list(requested, actual);
    if effective.is_empty() {
        requested.to_vec()
    } else {
        effective
    }
}

pub fn worker_affinity_plan_from_actual(
    requested: &[usize],
    actual: &[usize],
) -> WorkerAffinityPlan {
    let worker_cores = effective_worker_affinity_cores(requested, actual);
    WorkerAffinityPlan {
        thread_count_hint: worker_cores.len().max(1),
        worker_cores,
    }
}

pub fn worker_plan_after_affinity(requested: &[usize]) -> WorkerAffinityPlan {
    let Some(actual) = current_cpu_affinity_cores() else {
        return WorkerAffinityPlan {
            worker_cores: requested.to_vec(),
            thread_count_hint: requested.len().max(1),
        };
    };
    let plan = worker_affinity_plan_from_actual(requested, &actual);
    if plan.worker_cores != requested {
        eprintln!(
            "[INFO] CPU affinity: effective worker cores {:?}, threads {} (requested {:?})",
            plan.worker_cores, plan.thread_count_hint, requested
        );
    }
    plan
}

pub fn decide_auto_affinity(
    big_cores: &[usize],
    allowed_cpus: Option<&[usize]>,
    all_detected_cores: &[usize],
) -> AutoAffinityDecision {
    decide_auto_affinity_with_cap(big_cores, allowed_cpus, all_detected_cores, Some(4))
}

pub fn decide_auto_affinity_with_cap(
    big_cores: &[usize],
    allowed_cpus: Option<&[usize]>,
    all_detected_cores: &[usize],
    fallback_cap: Option<usize>,
) -> AutoAffinityDecision {
    if big_cores.len() >= 4 {
        return AutoAffinityDecision {
            pin_cores: Some(big_cores.to_vec()),
            thread_count_hint: big_cores.len(),
        };
    }

    let thread_count_hint = allowed_cpus
        .filter(|cpus| !cpus.is_empty())
        .map(|cpus| cpus.len())
        .or_else(|| (!all_detected_cores.is_empty()).then_some(all_detected_cores.len()))
        .unwrap_or(1);
    let thread_count_hint = fallback_cap
        .filter(|cap| *cap > 0)
        .map(|cap| thread_count_hint.min(cap))
        .unwrap_or(thread_count_hint);

    AutoAffinityDecision {
        pin_cores: None,
        thread_count_hint,
    }
}

pub fn read_auto_fallback_cap() -> Option<usize> {
    std::env::var("RNB_AUTO_FALLBACK_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|cap| *cap > 0)
        .or(Some(4))
}

pub fn read_allowed_cpu_list() -> Option<Vec<usize>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find(|line| line.starts_with("Cpus_allowed_list:"))?;
    let raw = line.split_once(':')?.1.trim();
    parse_allowed_cpu_list(raw)
}

pub fn read_cpu_max_freqs() -> Vec<(usize, u64)> {
    (0..16)
        .filter_map(|cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|freq| (cpu, freq))
        })
        .collect()
}

pub fn select_cores_by_frequency(
    freqs: &[(usize, u64)],
    descending: bool,
    limit: usize,
) -> Vec<usize> {
    let mut freqs = freqs.to_vec();
    freqs.sort_by(|a, b| {
        if descending {
            b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        } else {
            a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        }
    });
    freqs.into_iter().take(limit).map(|(id, _)| id).collect()
}

pub fn detect_all_cores_from_freqs(freqs: &[(usize, u64)]) -> Vec<usize> {
    let mut ids: Vec<usize> = freqs.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids
}

pub fn detect_big_cores_from_freqs(freqs: &[(usize, u64)]) -> Vec<usize> {
    if freqs.is_empty() {
        return vec![];
    }

    select_cores_by_frequency(freqs, true, 4)
}

pub fn detect_little_cores_from_freqs(freqs: &[(usize, u64)]) -> Vec<usize> {
    if freqs.is_empty() {
        return vec![];
    }

    select_cores_by_frequency(freqs, false, 4)
}

pub fn detect_big_cores() -> Vec<usize> {
    let freqs = read_cpu_max_freqs();
    detect_big_cores_from_freqs(&freqs)
}

pub fn detect_little_cores() -> Vec<usize> {
    let freqs = read_cpu_max_freqs();
    detect_little_cores_from_freqs(&freqs)
}

pub fn format_affinity_error(label: &str, cores: &[usize], errno: i32) -> String {
    format!("CPU affinity apply failed for {label} cores {cores:?}: errno={errno}")
}

pub fn worker_affinity_core(worker_index: usize, cores: &[usize]) -> Option<usize> {
    if cores.is_empty() {
        None
    } else {
        Some(cores[worker_index % cores.len()])
    }
}

pub fn apply_worker_affinity(
    label: &str,
    worker_index: usize,
    cores: &[usize],
) -> Result<Option<usize>, String> {
    let Some(core) = worker_affinity_core(worker_index, cores) else {
        return Ok(None);
    };
    set_cpu_affinity(label, &[core]).map(|()| Some(core))
}

pub fn requested_rayon_threads(
    user_set: Option<&str>,
    selected_core_count: Option<usize>,
    available_parallelism: usize,
    is_aarch64: bool,
) -> usize {
    if let Some(val) = user_set {
        let parsed = val.parse::<usize>().unwrap_or(available_parallelism.max(1));
        if let Some(count) = selected_core_count.filter(|count| *count > 0) {
            return parsed.min(count);
        }
        return parsed;
    }

    if is_aarch64 {
        if let Some(count) = selected_core_count.filter(|count| *count > 0) {
            return count;
        }
        return 4;
    }

    available_parallelism.max(1)
}

fn desktop_forced_gguf_default_threads(
    available_parallelism: usize,
    workload: CpuAssistWorkload,
) -> usize {
    let available = available_parallelism.max(1);
    let percent = match workload {
        CpuAssistWorkload::Default => 25,
        CpuAssistWorkload::HybridMoe => 50,
        CpuAssistWorkload::WideMoe => 66,
    };
    let proportional = available.saturating_mul(percent).div_ceil(100);
    proportional.clamp(4, 16).min(available)
}

fn should_use_desktop_forced_gguf_threads(
    path: &Path,
    affinity_explicit: bool,
    force_gguf: bool,
    moe_section_decode_sidecar: bool,
    rayon_threads_value: Option<&str>,
    desktop_cpu_assist_threads: bool,
) -> bool {
    let mobile_arm = cfg!(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    ));
    if mobile_arm
        || affinity_explicit
        || !desktop_cpu_assist_threads
        || !force_gguf
        || moe_section_decode_sidecar
        || rayon_threads_value.is_some()
    {
        return false;
    }

    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

pub fn plan_cpu_runtime_threads(
    affinity_value: Option<&str>,
    legacy_big_cores: bool,
    default_big_affinity: bool,
    moe_section_default_big_affinity: bool,
    rayon_threads_value: Option<&str>,
    available_parallelism: usize,
) -> CpuRuntimeThreadPlan {
    let (mut affinity_mode, affinity_explicit) =
        cpu_affinity_mode_from_env(affinity_value, legacy_big_cores);
    if default_big_affinity || moe_section_default_big_affinity {
        affinity_mode = CpuAffinityMode::Big;
    }

    let worker_affinity_cores = selected_worker_affinity_cores(
        affinity_mode,
        affinity_value,
        affinity_explicit,
        default_big_affinity,
        moe_section_default_big_affinity,
    );
    let thread_target_count = worker_affinity_cores.thread_count_hint;
    let requested_threads = requested_rayon_threads(
        rayon_threads_value,
        thread_target_count,
        available_parallelism,
        // Mobile ARM (Linux/Android) uses affinity-derived core counts; Apple
        // Silicon is a desktop aarch64 target and must use available_parallelism.
        cfg!(all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "android")
        )),
    );

    CpuRuntimeThreadPlan {
        requested_threads,
        worker_affinity_cores: worker_affinity_cores.worker_cores,
    }
}

pub fn plan_model_cpu_runtime_threads(
    path: &Path,
    moe_section_decode_sidecar: bool,
    affinity_value: Option<&str>,
    legacy_big_cores: bool,
    force_gguf: bool,
    rayon_threads_value: Option<&str>,
    available_parallelism: usize,
    desktop_cpu_assist_threads: bool,
    cpu_assist_workload: CpuAssistWorkload,
) -> CpuRuntimeThreadPlan {
    let (_, affinity_explicit) = cpu_affinity_mode_from_env(affinity_value, legacy_big_cores);
    let dense_packed_default_big_affinity = crate::packed_rnb_default_big_affinity(
        path,
        affinity_explicit,
        force_gguf,
        moe_section_decode_sidecar,
    );
    let moe_section_default_big_affinity = !affinity_explicit && moe_section_decode_sidecar;

    let mut plan = plan_cpu_runtime_threads(
        affinity_value,
        legacy_big_cores,
        dense_packed_default_big_affinity,
        moe_section_default_big_affinity,
        rayon_threads_value,
        available_parallelism,
    );
    if should_use_desktop_forced_gguf_threads(
        path,
        affinity_explicit,
        force_gguf,
        moe_section_decode_sidecar,
        rayon_threads_value,
        desktop_cpu_assist_threads,
    ) {
        plan.requested_threads =
            desktop_forced_gguf_default_threads(available_parallelism, cpu_assist_workload);
    }
    plan
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedWorkerAffinity {
    worker_cores: Option<Vec<usize>>,
    thread_count_hint: Option<usize>,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn selected_worker_affinity_cores(
    affinity_mode: CpuAffinityMode,
    affinity_value: Option<&str>,
    affinity_explicit: bool,
    default_big_affinity: bool,
    moe_section_default_big_affinity: bool,
) -> SelectedWorkerAffinity {
    let allowed_cpus = read_allowed_cpu_list();
    let cpu_freqs = read_cpu_max_freqs();
    let all_detected_cores = detect_all_cores_from_freqs(&cpu_freqs);
    match affinity_mode {
        CpuAffinityMode::Auto => {
            let mut big_cores = detect_big_cores_from_freqs(&cpu_freqs);
            if let Some(ref allowed) = allowed_cpus {
                big_cores = filter_cores_by_allowed_list(&big_cores, allowed);
            }
            let fallback_cap = read_auto_fallback_cap();
            let decision = decide_auto_affinity_with_cap(
                &big_cores,
                allowed_cpus.as_deref(),
                &all_detected_cores,
                fallback_cap,
            );
            if let Some(cores) = decision.pin_cores {
                match set_cpu_affinity("auto-big", &cores) {
                    Ok(()) => {
                        eprintln!("[INFO] CPU affinity: auto -> big cores {:?}", cores);
                        let plan = worker_plan_after_affinity(&cores);
                        SelectedWorkerAffinity {
                            worker_cores: Some(plan.worker_cores),
                            thread_count_hint: Some(plan.thread_count_hint),
                        }
                    }
                    Err(err) => {
                        eprintln!("[WARN] {}", err);
                        SelectedWorkerAffinity {
                            worker_cores: None,
                            thread_count_hint: Some(decision.thread_count_hint),
                        }
                    }
                }
            } else {
                eprintln!(
                    "[INFO] CPU affinity: auto -> all allowed (usable big {}, threads {}, cap {:?})",
                    big_cores.len(),
                    decision.thread_count_hint,
                    fallback_cap
                );
                SelectedWorkerAffinity {
                    worker_cores: None,
                    thread_count_hint: Some(decision.thread_count_hint),
                }
            }
        }
        CpuAffinityMode::Big => {
            let mut big_cores = detect_big_cores();
            if let Some(ref allowed) = allowed_cpus {
                big_cores = filter_cores_by_allowed_list(&big_cores, allowed);
            }
            if !big_cores.is_empty() {
                match set_cpu_affinity("big", &big_cores) {
                    Ok(()) => {
                        if moe_section_default_big_affinity {
                            eprintln!(
                                "[INFO] CPU affinity: MoE section MOE_DECODE default -> big cores {:?}",
                                big_cores
                            );
                        } else if default_big_affinity {
                            eprintln!(
                                "[INFO] CPU affinity: packed .rnb default -> big cores {:?}",
                                big_cores
                            );
                        } else {
                            eprintln!("[INFO] CPU affinity: big cores {:?}", big_cores);
                        }
                        let plan = worker_plan_after_affinity(&big_cores);
                        SelectedWorkerAffinity {
                            worker_cores: Some(plan.worker_cores),
                            thread_count_hint: Some(plan.thread_count_hint),
                        }
                    }
                    Err(err) => {
                        eprintln!("[WARN] {}", err);
                        SelectedWorkerAffinity {
                            worker_cores: None,
                            thread_count_hint: Some(big_cores.len()),
                        }
                    }
                }
            } else {
                eprintln!(
                    "[WARN] CPU affinity: requested big cores, detection failed; using all cores"
                );
                SelectedWorkerAffinity {
                    worker_cores: None,
                    thread_count_hint: Some(big_cores.len()),
                }
            }
        }
        CpuAffinityMode::Little => {
            let mut little_cores = detect_little_cores();
            if let Some(ref allowed) = allowed_cpus {
                little_cores = filter_cores_by_allowed_list(&little_cores, allowed);
            }
            if !little_cores.is_empty() {
                match set_cpu_affinity("little", &little_cores) {
                    Ok(()) => {
                        eprintln!("[INFO] CPU affinity: little cores {:?}", little_cores);
                        let plan = worker_plan_after_affinity(&little_cores);
                        SelectedWorkerAffinity {
                            worker_cores: Some(plan.worker_cores),
                            thread_count_hint: Some(plan.thread_count_hint),
                        }
                    }
                    Err(err) => {
                        eprintln!("[WARN] {}", err);
                        SelectedWorkerAffinity {
                            worker_cores: None,
                            thread_count_hint: Some(little_cores.len()),
                        }
                    }
                }
            } else {
                eprintln!(
                    "[WARN] CPU affinity: requested little cores, detection failed; using all cores"
                );
                SelectedWorkerAffinity {
                    worker_cores: None,
                    thread_count_hint: Some(little_cores.len()),
                }
            }
        }
        CpuAffinityMode::List => {
            let mut cores = affinity_explicit
                .then(|| affinity_value.and_then(parse_cpu_list))
                .flatten()
                .unwrap_or_default();
            if let Some(ref allowed) = allowed_cpus {
                cores = filter_cores_by_allowed_list(&cores, allowed);
            }
            if !cores.is_empty() {
                match set_cpu_affinity("explicit", &cores) {
                    Ok(()) => {
                        eprintln!("[INFO] CPU affinity: explicit cores {:?}", cores);
                        let plan = worker_plan_after_affinity(&cores);
                        SelectedWorkerAffinity {
                            worker_cores: Some(plan.worker_cores),
                            thread_count_hint: Some(plan.thread_count_hint),
                        }
                    }
                    Err(err) => {
                        eprintln!("[WARN] {}", err);
                        SelectedWorkerAffinity {
                            worker_cores: None,
                            thread_count_hint: Some(cores.len()),
                        }
                    }
                }
            } else {
                eprintln!("[WARN] CPU affinity: explicit core list parse failed; using all cores");
                SelectedWorkerAffinity {
                    worker_cores: None,
                    thread_count_hint: Some(cores.len()),
                }
            }
        }
        CpuAffinityMode::All => {
            let thread_count_hint = if affinity_explicit {
                allowed_cpus
                    .as_ref()
                    .map(|cpus| cpus.len())
                    .or_else(|| Some(all_detected_cores.len()))
            } else {
                Some(4)
            };
            if affinity_explicit {
                eprintln!("[INFO] CPU affinity: all cores");
            } else {
                eprintln!("[INFO] CPU affinity: all cores (mobile default 4T baseline)");
            }
            SelectedWorkerAffinity {
                worker_cores: None,
                thread_count_hint,
            }
        }
    }
}

#[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
)))]
fn selected_worker_affinity_cores(
    affinity_mode: CpuAffinityMode,
    _affinity_value: Option<&str>,
    affinity_explicit: bool,
    _default_big_affinity: bool,
    _moe_section_default_big_affinity: bool,
) -> SelectedWorkerAffinity {
    if matches!(
        affinity_mode,
        CpuAffinityMode::Auto
            | CpuAffinityMode::Big
            | CpuAffinityMode::Little
            | CpuAffinityMode::List
    ) {
        eprintln!(
            "[WARN] CPU affinity: non-default affinity requested on unsupported architecture; using all cores"
        );
    } else if !affinity_explicit {
        eprintln!("[INFO] CPU affinity: all cores (default auto threads)");
    } else {
        eprintln!("[INFO] CPU affinity: all cores");
    }
    SelectedWorkerAffinity {
        worker_cores: None,
        thread_count_hint: None,
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
pub fn current_cpu_affinity_cores() -> Option<Vec<usize>> {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut mask);
        if rc != 0 {
            return None;
        }

        let mut cores = Vec::new();
        for core in 0..(std::mem::size_of::<libc::cpu_set_t>() * 8) {
            if libc::CPU_ISSET(core, &mask) {
                cores.push(core);
            }
        }
        Some(cores)
    }
}

#[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
)))]
pub fn current_cpu_affinity_cores() -> Option<Vec<usize>> {
    None
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
pub fn set_cpu_affinity(label: &str, cores: &[usize]) -> Result<(), String> {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        for &core in cores {
            libc::CPU_SET(core, &mut mask);
        }
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
        if rc == 0 {
            Ok(())
        } else {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
            Err(format_affinity_error(label, cores, errno))
        }
    }
}

#[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
)))]
pub fn set_cpu_affinity(label: &str, cores: &[usize]) -> Result<(), String> {
    Err(format_affinity_error(label, cores, -1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_allowed_cpu_ranges() {
        assert_eq!(
            parse_allowed_cpu_list("0-2,4,6-7"),
            Some(vec![0, 1, 2, 4, 6, 7])
        );
        assert_eq!(parse_allowed_cpu_list("4-2"), None);
    }

    #[test]
    fn parses_explicit_cpu_list() {
        assert_eq!(parse_cpu_list("0,2,7"), Some(vec![0, 2, 7]));
        assert_eq!(parse_cpu_list("0,bad,7"), None);
    }

    #[test]
    fn cpu_affinity_mode_parses_env_and_legacy_default() {
        assert_eq!(
            cpu_affinity_mode_from_env(Some("all"), true),
            (CpuAffinityMode::All, true)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(Some("auto"), false),
            (CpuAffinityMode::Auto, true)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(Some("big"), false),
            (CpuAffinityMode::Big, true)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(Some("little"), false),
            (CpuAffinityMode::Little, true)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(Some("0,2,7"), false),
            (CpuAffinityMode::List, true)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(None, true),
            (CpuAffinityMode::Big, false)
        );
        let target_default = default_affinity_mode_for_target();
        assert_eq!(
            cpu_affinity_mode_from_env(None, false),
            (target_default, false)
        );
        assert_eq!(
            cpu_affinity_mode_from_env(Some("weird"), false),
            (CpuAffinityMode::All, true)
        );
    }

    #[test]
    fn auto_affinity_prefers_four_big_cores() {
        let decision = decide_auto_affinity(
            &[4, 5, 6, 7],
            Some(&[0, 1, 2, 3]),
            &[0, 1, 2, 3, 4, 5, 6, 7],
        );
        assert_eq!(decision.pin_cores, Some(vec![4, 5, 6, 7]));
        assert_eq!(decision.thread_count_hint, 4);
    }

    #[test]
    fn auto_affinity_caps_fallback_threads() {
        let decision = decide_auto_affinity_with_cap(
            &[6, 7],
            Some(&[0, 1, 2, 3, 4, 5, 6, 7]),
            &[0, 1, 2, 3, 4, 5, 6, 7],
            Some(4),
        );
        assert_eq!(decision.pin_cores, None);
        assert_eq!(decision.thread_count_hint, 4);
    }

    #[test]
    fn auto_affinity_keeps_lower_allowed_count_under_cap() {
        let decision = decide_auto_affinity_with_cap(
            &[4, 5],
            Some(&[0, 1, 2, 3, 4, 5]),
            &[0, 1, 2, 3, 4, 5, 6, 7],
            Some(6),
        );
        assert_eq!(
            decision,
            AutoAffinityDecision {
                pin_cores: None,
                thread_count_hint: 6,
            }
        );
    }

    #[test]
    fn core_frequency_helpers_pick_big_and_little_tiers() {
        let freqs = vec![
            (0, 3_000_000),
            (1, 2_600_000),
            (2, 2_500_000),
            (3, 2_400_000),
            (4, 2_300_000),
            (5, 2_200_000),
            (6, 2_100_000),
            (7, 2_000_000),
        ];

        assert_eq!(detect_big_cores_from_freqs(&freqs), vec![0, 1, 2, 3]);
        assert_eq!(detect_little_cores_from_freqs(&freqs), vec![7, 6, 5, 4]);
    }

    #[test]
    fn affinity_filter_and_effective_plan_keep_applied_cores() {
        assert_eq!(
            filter_cores_by_allowed_list(&[7, 4, 5, 6], &[0, 1, 2, 3, 4, 5]),
            vec![4, 5]
        );
        assert_eq!(
            effective_worker_affinity_cores(&[7, 4, 5, 6], &[7, 4]),
            vec![7, 4]
        );

        let plan = worker_affinity_plan_from_actual(&[7, 4, 5, 6], &[4, 6]);
        assert_eq!(plan.worker_cores, vec![4, 6]);
        assert_eq!(plan.thread_count_hint, 2);
    }

    #[test]
    fn affinity_error_mentions_label_cores_and_errno() {
        let msg = format_affinity_error("big", &[4, 5, 6, 7], -1);
        assert!(msg.contains("big"));
        assert!(msg.contains("[4, 5, 6, 7]"));
        assert!(msg.contains("errno="));
    }

    #[test]
    fn worker_affinity_round_robins_cores() {
        assert_eq!(worker_affinity_core(0, &[7, 4]), Some(7));
        assert_eq!(worker_affinity_core(1, &[7, 4]), Some(4));
        assert_eq!(worker_affinity_core(2, &[7, 4]), Some(7));
        assert_eq!(worker_affinity_core(0, &[]), None);
    }

    #[test]
    fn worker_affinity_apply_ignores_empty_core_list() {
        assert_eq!(
            apply_worker_affinity("test-worker", 0, &[]).expect("empty affinity should be ok"),
            None
        );
    }

    #[test]
    fn requested_threads_follow_selected_core_count() {
        assert_eq!(requested_rayon_threads(None, Some(4), 8, true), 4);
        assert_eq!(requested_rayon_threads(None, Some(3), 8, true), 3);
        assert_eq!(requested_rayon_threads(None, None, 8, false), 8);
        assert_eq!(requested_rayon_threads(Some("6"), Some(8), 8, true), 6);
        assert_eq!(requested_rayon_threads(Some("bad"), Some(3), 8, true), 3);
        assert_eq!(requested_rayon_threads(Some("4"), Some(3), 8, true), 3);
    }

    #[test]
    fn cpu_runtime_thread_plan_keeps_rayon_override() {
        let plan = plan_cpu_runtime_threads(None, false, false, false, Some("2"), 8);

        assert_eq!(plan.requested_threads, 2);
    }

    #[test]
    fn model_cpu_runtime_thread_plan_matches_expanded_inputs() {
        let path = Path::new("model.rnb");

        let compact = plan_model_cpu_runtime_threads(
            path,
            true,
            Some("all"),
            false,
            false,
            None,
            8,
            false,
            CpuAssistWorkload::Default,
        );
        let expanded = plan_cpu_runtime_threads(Some("all"), false, false, false, None, 8);

        assert_eq!(compact, expanded);
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn model_cpu_runtime_thread_plan_scales_forced_gguf_desktop_threads_by_percent() {
        let path = Path::new("model.gguf");

        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                32,
                true,
                CpuAssistWorkload::Default,
            )
            .requested_threads,
            8
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                16,
                true,
                CpuAssistWorkload::Default,
            )
            .requested_threads,
            4
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                8,
                true,
                CpuAssistWorkload::Default,
            )
            .requested_threads,
            4
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                Some("12"),
                32,
                true,
                CpuAssistWorkload::Default,
            )
            .requested_threads,
            12
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                32,
                false,
                CpuAssistWorkload::Default,
            )
            .requested_threads,
            32
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn hybrid_moe_cpu_assist_workload_uses_half_desktop_threads() {
        let path = Path::new("model.gguf");

        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                32,
                true,
                CpuAssistWorkload::HybridMoe,
            )
            .requested_threads,
            16
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                Some("12"),
                32,
                true,
                CpuAssistWorkload::HybridMoe,
            )
            .requested_threads,
            12
        );
    }

    #[test]
    fn wide_moe_cpu_assist_workload_uses_two_thirds_desktop_threads() {
        assert_eq!(
            desktop_forced_gguf_default_threads(18, CpuAssistWorkload::WideMoe),
            12
        );
        assert_eq!(
            desktop_forced_gguf_default_threads(32, CpuAssistWorkload::WideMoe),
            16
        );
        assert_eq!(
            desktop_forced_gguf_default_threads(8, CpuAssistWorkload::WideMoe),
            6
        );
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    )))]
    #[test]
    fn wide_moe_model_cpu_runtime_thread_plan_applies_to_desktop_forced_gguf() {
        let path = Path::new("model.gguf");

        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                None,
                18,
                true,
                CpuAssistWorkload::WideMoe,
            )
            .requested_threads,
            12
        );
        assert_eq!(
            plan_model_cpu_runtime_threads(
                path,
                false,
                None,
                false,
                true,
                Some("10"),
                18,
                true,
                CpuAssistWorkload::WideMoe,
            )
            .requested_threads,
            10
        );
    }
}
