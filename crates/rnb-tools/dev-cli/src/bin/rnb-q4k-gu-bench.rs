#[cfg(target_arch = "aarch64")]
#[path = "../q4k_gu_microbench.rs"]
mod q4k_gu_microbench;

#[cfg(target_arch = "aarch64")]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn run() -> Result<(), String> {
    use q4k_gu_microbench::{
        run_q4k_gu_bench, run_q4k_gu_row_bench, run_q4k_gu_sidecar_bench, Q4KGuBenchConfig,
        Q4KGuRouteSelection, Q4KGuRowBenchConfig, Q4KGuSidecarBenchConfig,
    };

    let args = Args::parse()?;
    if args.help {
        println!("{}", usage());
        return Ok(());
    }

    let cores_to_apply = if let Some(cores) = args.cores.as_deref() {
        Some(cores)
    } else {
        env_cores()
    };
    if let Some(cores) = cores_to_apply {
        let parsed = parse_cores(cores)?;
        apply_affinity(&parsed)?;
        eprintln!("[INFO] pinned process to cores {:?}", parsed);
    }
    if let Some(allowed) = current_affinity_cores() {
        eprintln!("[INFO] effective allowed cores {:?}", allowed);
    }

    let results = match args.mode {
        BenchMode::Block => {
            let default = Q4KGuBenchConfig::default();
            let config = Q4KGuBenchConfig {
                blocks: args.blocks,
                iters: args.iters.unwrap_or(default.iters),
                warmup_iters: args.warmup_iters.unwrap_or(default.warmup_iters),
                repeats: args.repeats,
            };
            eprintln!(
                "[INFO] q4k gu block bench: blocks={} iters={} warmup={} repeats={}",
                config.blocks, config.iters, config.warmup_iters, config.repeats
            );
            run_q4k_gu_bench(config)
        }
        BenchMode::Row => {
            let default = Q4KGuRowBenchConfig::default();
            let config = Q4KGuRowBenchConfig {
                rows: args.rows,
                blocks_per_row: args.blocks,
                selected_rows: args.selected_rows,
                iters: args.iters.unwrap_or(default.iters),
                warmup_iters: args.warmup_iters.unwrap_or(default.warmup_iters),
                repeats: args.repeats,
            };
            eprintln!(
                "[INFO] q4k gu row bench: rows={} blocks_per_row={} selected_rows={} iters={} warmup={} repeats={}",
                config.rows,
                config.blocks_per_row,
                config.selected_rows,
                config.iters,
                config.warmup_iters,
                config.repeats
            );
            run_q4k_gu_row_bench(config)
        }
        BenchMode::Sidecar => {
            let default = Q4KGuSidecarBenchConfig::default();
            let config = Q4KGuSidecarBenchConfig {
                layer: args.layer,
                layer_count: args.layer_count,
                first_expert: args.first_expert,
                selected_experts: args.selected_rows,
                iters: args.iters.unwrap_or(default.iters),
                warmup_iters: args.warmup_iters.unwrap_or(default.warmup_iters),
                repeats: args.repeats,
            };
            let path = args
                .rnb
                .as_deref()
                .ok_or_else(|| "--mode sidecar requires --rnb <path.rnb>".to_string())?;
            let route_trace = if let Some(path) = args.route_trace.as_deref() {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("read route trace {} failed: {e}", path.display()))?;
                let parsed = parse_route_trace_text(&text)?;
                let trace: Vec<Q4KGuRouteSelection> = parsed
                    .into_iter()
                    .map(|line| Q4KGuRouteSelection {
                        layer: line.layer,
                        experts: line.experts,
                    })
                    .collect();
                eprintln!(
                    "[INFO] route trace: path={} selections={}",
                    path.display(),
                    trace.len()
                );
                Some(trace)
            } else {
                None
            };
            let file = std::fs::File::open(path)
                .map_err(|e| format!("open sidecar {} failed: {e}", path.display()))?;
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("mmap sidecar {} failed: {e}", path.display()))?
            };
            eprintln!(
                "[INFO] q4k gu sidecar bench: path={} layer={} layer_count={} first_expert={} selected_experts={} iters={} warmup={} repeats={}",
                path.display(),
                config.layer,
                config.layer_count,
                config.first_expert,
                config.selected_experts,
                config.iters,
                config.warmup_iters,
                config.repeats
            );
            run_q4k_gu_sidecar_bench(&mmap, config, route_trace.as_deref())?
        }
    };
    println!("variant,repeat,elapsed_ms,ns_per_iter,ns_per_row,ns_per_block,checksum");
    for r in results {
        println!(
            "{},{},{:.3},{:.3},{:.3},{:.3},{}",
            r.variant,
            r.repeat,
            r.elapsed_ns as f64 / 1_000_000.0,
            r.ns_per_iter,
            r.ns_per_row,
            r.ns_per_block,
            r.checksum
        );
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[derive(Debug)]
struct Args {
    mode: BenchMode,
    rnb: Option<std::path::PathBuf>,
    blocks: usize,
    rows: usize,
    selected_rows: usize,
    layer: usize,
    layer_count: usize,
    first_expert: usize,
    route_trace: Option<std::path::PathBuf>,
    iters: Option<u64>,
    warmup_iters: Option<u64>,
    repeats: usize,
    cores: Option<String>,
    help: bool,
}

#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Copy)]
enum BenchMode {
    Block,
    Row,
    Sidecar,
}

#[cfg(target_arch = "aarch64")]
impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            mode: BenchMode::Block,
            rnb: None,
            blocks: 8,
            rows: 512,
            selected_rows: 8,
            layer: 0,
            layer_count: 1,
            first_expert: 0,
            route_trace: None,
            iters: None,
            warmup_iters: None,
            repeats: 3,
            cores: None,
            help: false,
        };
        let mut raw = std::env::args().skip(1);
        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "-h" | "--help" => args.help = true,
                "--mode" => args.mode = parse_mode(&next_string(&mut raw, "--mode")?)?,
                "--rnb" => {
                    args.rnb = Some(std::path::PathBuf::from(next_string(&mut raw, "--rnb")?))
                }
                "--blocks" => args.blocks = parse_next(&mut raw, "--blocks")?,
                "--rows" => args.rows = parse_next(&mut raw, "--rows")?,
                "--selected-rows" => args.selected_rows = parse_next(&mut raw, "--selected-rows")?,
                "--layer" => args.layer = parse_next(&mut raw, "--layer")?,
                "--layer-count" => args.layer_count = parse_next(&mut raw, "--layer-count")?,
                "--first-expert" => args.first_expert = parse_next(&mut raw, "--first-expert")?,
                "--route-trace" => {
                    args.route_trace = Some(std::path::PathBuf::from(next_string(
                        &mut raw,
                        "--route-trace",
                    )?))
                }
                "--iters" => args.iters = Some(parse_next(&mut raw, "--iters")?),
                "--warmup" => args.warmup_iters = Some(parse_next(&mut raw, "--warmup")?),
                "--repeats" => args.repeats = parse_next(&mut raw, "--repeats")?,
                "--cores" => args.cores = Some(next_string(&mut raw, "--cores")?),
                other => return Err(format!("unknown argument: {other}\n{}", usage())),
            }
        }
        if args.blocks == 0 || args.rows == 0 || args.selected_rows == 0 || args.repeats == 0 {
            return Err(format!(
                "--blocks, --rows, --selected-rows, and --repeats must be > 0\n{}",
                usage()
            ));
        }
        if matches!(args.iters, Some(0)) || matches!(args.warmup_iters, Some(0)) {
            return Err(format!("--iters and --warmup must be > 0\n{}", usage()));
        }
        if args.layer_count == 0 {
            return Err(format!("--layer-count must be > 0\n{}", usage()));
        }
        Ok(args)
    }
}

#[cfg(target_arch = "aarch64")]
fn parse_mode(raw: &str) -> Result<BenchMode, String> {
    match raw {
        "block" => Ok(BenchMode::Block),
        "row" => Ok(BenchMode::Row),
        "sidecar" => Ok(BenchMode::Sidecar),
        _ => Err(format!(
            "bad --mode {raw:?}; expected block, row, or sidecar\n{}",
            usage()
        )),
    }
}

#[cfg(target_arch = "aarch64")]
fn parse_next<T: std::str::FromStr>(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let value = next_string(raw, flag)?;
    value
        .parse::<T>()
        .map_err(|e| format!("bad {flag} value {value:?}: {e}"))
}

#[cfg(target_arch = "aarch64")]
fn next_string(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next()
        .ok_or_else(|| format!("{flag} requires a value\n{}", usage()))
}

#[cfg(target_arch = "aarch64")]
fn env_cores() -> Option<&'static str> {
    match std::env::var("RNB_CPU_AFFINITY").ok().as_deref() {
        Some("big") => Some("4,5,6,7"),
        _ => None,
    }
}

#[cfg(target_arch = "aarch64")]
fn usage() -> &'static str {
    "usage: rnb-q4k-gu-bench [--mode block|row|sidecar] [--rnb PATH] [--blocks N] [--rows N] [--selected-rows N] [--layer N] [--layer-count N] [--first-expert N] [--route-trace PATH] [--iters N] [--warmup N] [--repeats N] [--cores LIST]\n\
     \n\
     Measures Q4_K gate/up MoE section decode kernel components on aarch64.\n\
     block mode reports per-block synthetic component costs.\n\
     row mode reports rows x selected-rows x blocks_per_row traversal costs; --blocks is blocks_per_row.\n\
     sidecar mode mmaps --rnb and traverses actual MOE_DECODE gate/up rows; --selected-rows is selected experts.\n\
     --route-trace replays RNB_MOE_ROUTE_TRACE_FILE lines instead of synthetic expert selection.\n\
     --cores 4,5,6,7 can be used for Lenovo/Flip4 big-core runs.\n\
     RNB_CPU_AFFINITY=big is treated as --cores 4,5,6,7."
}

#[cfg(target_arch = "aarch64")]
fn parse_cores(raw: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(
            part.parse::<usize>()
                .map_err(|e| format!("bad core id {part:?}: {e}"))?,
        );
    }
    if out.is_empty() {
        return Err("empty core list".to_string());
    }
    Ok(out)
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn apply_affinity(cores: &[usize]) -> Result<(), String> {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut mask);
        for &core in cores {
            libc::CPU_SET(core, &mut mask);
        }
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
        if rc != 0 {
            let errno = std::io::Error::last_os_error();
            return Err(format!("sched_setaffinity({cores:?}) failed: {errno}"));
        }
    }
    Ok(())
}

#[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "linux", target_os = "android"))
))]
fn apply_affinity(_cores: &[usize]) -> Result<(), String> {
    Err("--cores is only supported on linux/android".to_string())
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn current_affinity_cores() -> Option<Vec<usize>> {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut mask);
        if rc != 0 {
            return None;
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &mask) {
                out.push(cpu);
            }
        }
        Some(out)
    }
}

#[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "linux", target_os = "android"))
))]
fn current_affinity_cores() -> Option<Vec<usize>> {
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(any(target_arch = "aarch64", test)), allow(dead_code))]
struct RouteTraceLine {
    layer: usize,
    experts: Vec<usize>,
}

#[cfg_attr(not(any(target_arch = "aarch64", test)), allow(dead_code))]
fn parse_route_trace_text(text: &str) -> Result<Vec<RouteTraceLine>, String> {
    let mut out = Vec::new();
    for (line_idx, raw) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split(',').map(str::trim);
        let layer_raw = parts
            .next()
            .ok_or_else(|| format!("route trace line {line_no}: missing layer"))?;
        let layer = layer_raw
            .parse::<usize>()
            .map_err(|e| format!("route trace line {line_no}: bad layer {layer_raw:?}: {e}"))?;
        let mut experts = Vec::new();
        for expert_raw in parts {
            if expert_raw.is_empty() {
                return Err(format!("route trace line {line_no}: empty expert field"));
            }
            experts.push(expert_raw.parse::<usize>().map_err(|e| {
                format!("route trace line {line_no}: bad expert {expert_raw:?}: {e}")
            })?);
        }
        if experts.is_empty() {
            return Err(format!("route trace line {line_no}: no experts"));
        }
        out.push(RouteTraceLine { layer, experts });
    }
    if out.is_empty() {
        return Err("route trace is empty".to_string());
    }
    Ok(out)
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("rnb-q4k-gu-bench is aarch64-only");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_route_trace_text_accepts_layer_and_experts() {
        let parsed = parse_route_trace_text("3,9,2,5\n4,1,0\n").unwrap();
        assert_eq!(
            parsed,
            vec![
                RouteTraceLine {
                    layer: 3,
                    experts: vec![9, 2, 5],
                },
                RouteTraceLine {
                    layer: 4,
                    experts: vec![1, 0],
                },
            ]
        );
    }
}
