use rnb_moe_cache_sim::common::cache::CachePolicy;
use rnb_moe_cache_sim::common::model::ModelMeta;
use rnb_moe_cache_sim::common::predictor_trace::{
    analyze_combined_scored_trace, analyze_current_router_trace, analyze_online_layer_hot_trace,
    analyze_predictor_trace, analyze_prev_same_layer_trace,
    analyze_union_prev_same_layer_prev_group_trace, parse_predictor_trace_jsonl,
    PredictorAnalysisRow,
};
use rnb_moe_cache_sim::common::route_shape::{analyze_route_shape, RouteShapeAnalysis};
use rnb_moe_cache_sim::common::trace::{parse_jsonl_trace, parse_rnb_route_csv, TraceEvent};
use rnb_moe_cache_sim::pc::hardware::HardwareMeta;
use rnb_moe_cache_sim::pc::simulate::{simulate_pc_cache_with_predictor_options, PcSimulationRow};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

struct Args {
    trace: PathBuf,
    trace_format: TraceFormat,
    model_meta: PathBuf,
    hardware_meta: PathBuf,
    policies: Vec<CachePolicy>,
    cache_gb: Vec<u64>,
    warmup_steps: usize,
    eval_start_step: usize,
    static_hot_pct: Vec<u8>,
    window_steps: usize,
    predictor_recall_pct: Vec<u8>,
    predictor_extra_ratio: Vec<u16>,
    json_out: Option<PathBuf>,
}

struct PredictorArgs {
    trace: PathBuf,
    lookahead_groups: Vec<usize>,
    top_n: Vec<usize>,
    sources: Vec<PredictorSource>,
    skip_groups: usize,
    json_out: Option<PathBuf>,
}

struct RouteShapeArgs {
    trace: PathBuf,
    trace_format: TraceFormat,
    max_group: usize,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Copy)]
enum PredictorSource {
    RouterTop,
    RouterCurrent,
    PrevSameLayer,
    PrevLayerUnion,
    OnlineLayerHot,
    CombinedScored,
}

#[derive(Clone, Copy)]
enum TraceFormat {
    Jsonl,
    RnbRouteCsv,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.first().map(String::as_str) == Some("predictor") {
        return run_predictor(parse_predictor_args(&raw)?);
    }
    if raw.first().map(String::as_str) == Some("route-shape") {
        return run_route_shape(parse_route_shape_args(&raw)?);
    }
    let args = parse_args(&raw)?;
    run_pc(args)
}

fn run_pc(args: Args) -> Result<(), String> {
    let trace_text = fs::read_to_string(&args.trace)
        .map_err(|e| format!("read trace {}: {}", args.trace.display(), e))?;
    let events = parse_trace(&trace_text, args.trace_format)?;
    let model = read_json::<ModelMeta>(&args.model_meta)?;
    let hardware = read_json::<HardwareMeta>(&args.hardware_meta)?;

    let mut rows = Vec::<OutputRow>::new();
    for policy in &args.policies {
        for &cache_gb in &args.cache_gb {
            for &predictor_recall_pct in &args.predictor_recall_pct {
                for &predictor_extra_ratio in &args.predictor_extra_ratio {
                    let cache_bytes = cache_gb.saturating_mul(1024 * 1024 * 1024);
                    for &static_hot_pct in &args.static_hot_pct {
                        let row = simulate_pc_cache_with_predictor_options(
                            &events,
                            &model,
                            &hardware,
                            *policy,
                            cache_bytes,
                            args.warmup_steps,
                            args.window_steps,
                            args.eval_start_step,
                            static_hot_pct,
                            predictor_recall_pct,
                            predictor_extra_ratio,
                        )?;
                        rows.push(OutputRow {
                            policy: policy_name(*policy).to_string(),
                            row,
                        });
                    }
                }
            }
        }
    }

    print_table(&model.name, &hardware.name, events.len(), &rows);
    if let Some(path) = args.json_out {
        let json = serde_json::to_string_pretty(&rows)
            .map_err(|e| format!("serialize json output: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write {}: {}", path.display(), e))?;
    }
    Ok(())
}

fn run_predictor(args: PredictorArgs) -> Result<(), String> {
    let trace_text = fs::read_to_string(&args.trace)
        .map_err(|e| format!("read trace {}: {}", args.trace.display(), e))?;
    let lines = parse_predictor_trace_jsonl(&trace_text)?;
    let eval_lines = lines.get(args.skip_groups..).ok_or_else(|| {
        format!(
            "bad --skip-groups {}: trace has {} groups",
            args.skip_groups,
            lines.len()
        )
    })?;

    let mut rows = Vec::<PredictorAnalysisRow>::new();
    for &source in &args.sources {
        match source {
            PredictorSource::RouterTop => {
                for &lookahead_groups in &args.lookahead_groups {
                    for &top_n in &args.top_n {
                        rows.push(analyze_predictor_trace(
                            eval_lines,
                            lookahead_groups,
                            top_n,
                        )?);
                    }
                }
            }
            PredictorSource::RouterCurrent => {
                for &top_n in &args.top_n {
                    rows.push(analyze_current_router_trace(eval_lines, top_n)?);
                }
            }
            PredictorSource::PrevSameLayer => {
                for &top_n in &args.top_n {
                    rows.push(analyze_prev_same_layer_trace(eval_lines, top_n)?);
                }
            }
            PredictorSource::PrevLayerUnion => {
                for &top_n in &args.top_n {
                    rows.push(analyze_union_prev_same_layer_prev_group_trace(
                        eval_lines, top_n,
                    )?);
                }
            }
            PredictorSource::OnlineLayerHot => {
                for &top_n in &args.top_n {
                    rows.push(analyze_online_layer_hot_trace(eval_lines, top_n)?);
                }
            }
            PredictorSource::CombinedScored => {
                for &top_n in &args.top_n {
                    rows.push(analyze_combined_scored_trace(eval_lines, top_n)?);
                }
            }
        }
    }

    print_predictor_table(lines.len(), args.skip_groups, &rows);
    if let Some(path) = args.json_out {
        let json = serde_json::to_string_pretty(&rows)
            .map_err(|e| format!("serialize json output: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write {}: {}", path.display(), e))?;
    }
    Ok(())
}

fn run_route_shape(args: RouteShapeArgs) -> Result<(), String> {
    let trace_text = fs::read_to_string(&args.trace)
        .map_err(|e| format!("read trace {}: {}", args.trace.display(), e))?;
    let events = parse_trace(&trace_text, args.trace_format)?;
    let analysis = analyze_route_shape(&events, args.max_group)?;

    print_route_shape_table(&analysis);
    if let Some(path) = args.json_out {
        let json = serde_json::to_string_pretty(&analysis)
            .map_err(|e| format!("serialize json output: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write {}: {}", path.display(), e))?;
    }
    Ok(())
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    if raw.first().map(String::as_str) != Some("pc") {
        return Err(usage());
    }
    let mut trace = None;
    let mut trace_format = TraceFormat::Jsonl;
    let mut model_meta = None;
    let mut hardware_meta = None;
    let mut policies = None;
    let mut cache_gb = None;
    let mut warmup_steps = 0usize;
    let mut eval_start_step = None;
    let mut static_hot_pct = vec![50u8];
    let mut window_steps = 8usize;
    let mut predictor_recall_pct = vec![100u8];
    let mut predictor_extra_ratio = vec![0u16];
    let mut json_out = None;
    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--trace" => {
                i += 1;
                trace = raw.get(i).map(PathBuf::from);
            }
            "--trace-format" => {
                i += 1;
                trace_format = parse_trace_format(raw.get(i).ok_or_else(usage)?)?;
            }
            "--model-meta" => {
                i += 1;
                model_meta = raw.get(i).map(PathBuf::from);
            }
            "--hardware-meta" => {
                i += 1;
                hardware_meta = raw.get(i).map(PathBuf::from);
            }
            "--policy" => {
                i += 1;
                policies = Some(parse_policies(raw.get(i).ok_or_else(usage)?)?);
            }
            "--cache-gb" => {
                i += 1;
                cache_gb = Some(parse_cache_gb(raw.get(i).ok_or_else(usage)?)?);
            }
            "--warmup-steps" => {
                i += 1;
                warmup_steps = raw
                    .get(i)
                    .ok_or_else(usage)?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --warmup-steps value: {e}"))?;
            }
            "--window-steps" => {
                i += 1;
                window_steps = raw
                    .get(i)
                    .ok_or_else(usage)?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --window-steps value: {e}"))?;
            }
            "--eval-start-step" => {
                i += 1;
                eval_start_step = Some(
                    raw.get(i)
                        .ok_or_else(usage)?
                        .parse::<usize>()
                        .map_err(|e| format!("bad --eval-start-step value: {e}"))?,
                );
            }
            "--static-hot-pct" => {
                i += 1;
                static_hot_pct = parse_u8_list(raw.get(i).ok_or_else(usage)?, "--static-hot-pct")?;
                for value in &static_hot_pct {
                    if *value > 100 {
                        return Err(format!("bad --static-hot-pct value {value}: max 100"));
                    }
                }
            }
            "--predictor-recall-pct" => {
                i += 1;
                predictor_recall_pct =
                    parse_u8_list(raw.get(i).ok_or_else(usage)?, "--predictor-recall-pct")?;
                for value in &predictor_recall_pct {
                    if *value > 100 {
                        return Err(format!("bad --predictor-recall-pct value {value}: max 100"));
                    }
                }
            }
            "--predictor-extra-ratio" => {
                i += 1;
                predictor_extra_ratio =
                    parse_u16_list(raw.get(i).ok_or_else(usage)?, "--predictor-extra-ratio")?;
            }
            "--json-out" => {
                i += 1;
                json_out = raw.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        i += 1;
    }

    Ok(Args {
        trace: trace.ok_or_else(usage)?,
        trace_format,
        model_meta: model_meta.ok_or_else(usage)?,
        hardware_meta: hardware_meta.ok_or_else(usage)?,
        policies: policies.unwrap_or_else(|| vec![CachePolicy::Lru]),
        cache_gb: cache_gb.ok_or_else(usage)?,
        warmup_steps,
        eval_start_step: eval_start_step.unwrap_or(warmup_steps),
        static_hot_pct,
        window_steps,
        predictor_recall_pct,
        predictor_extra_ratio,
        json_out,
    })
}

fn parse_predictor_args(raw: &[String]) -> Result<PredictorArgs, String> {
    let mut trace = None;
    let mut lookahead_groups = vec![1usize];
    let mut top_n = vec![8usize, 16usize];
    let mut sources = vec![PredictorSource::RouterTop];
    let mut skip_groups = 0usize;
    let mut json_out = None;
    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--trace" => {
                i += 1;
                trace = raw.get(i).map(PathBuf::from);
            }
            "--lookahead-groups" => {
                i += 1;
                lookahead_groups =
                    parse_usize_list(raw.get(i).ok_or_else(usage)?, "--lookahead-groups")?;
            }
            "--top-n" => {
                i += 1;
                top_n = parse_usize_list(raw.get(i).ok_or_else(usage)?, "--top-n")?;
            }
            "--source" => {
                i += 1;
                sources = parse_predictor_sources(raw.get(i).ok_or_else(usage)?)?;
            }
            "--skip-groups" => {
                i += 1;
                skip_groups = raw
                    .get(i)
                    .ok_or_else(usage)?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --skip-groups value: {e}"))?;
            }
            "--json-out" => {
                i += 1;
                json_out = raw.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        i += 1;
    }

    Ok(PredictorArgs {
        trace: trace.ok_or_else(usage)?,
        lookahead_groups,
        top_n,
        sources,
        skip_groups,
        json_out,
    })
}

fn parse_route_shape_args(raw: &[String]) -> Result<RouteShapeArgs, String> {
    let mut trace = None;
    let mut trace_format = TraceFormat::RnbRouteCsv;
    let mut max_group = 4usize;
    let mut json_out = None;
    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--trace" => {
                i += 1;
                trace = raw.get(i).map(PathBuf::from);
            }
            "--trace-format" => {
                i += 1;
                trace_format = parse_trace_format(raw.get(i).ok_or_else(usage)?)?;
            }
            "--max-group" => {
                i += 1;
                max_group = raw
                    .get(i)
                    .ok_or_else(usage)?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --max-group value: {e}"))?;
            }
            "--json-out" => {
                i += 1;
                json_out = raw.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        i += 1;
    }

    Ok(RouteShapeArgs {
        trace: trace.ok_or_else(usage)?,
        trace_format,
        max_group,
        json_out,
    })
}

fn usage() -> String {
    "usage: rnb-moe-tier pc --trace PATH --model-meta PATH --hardware-meta PATH --cache-gb LIST \
     [--trace-format jsonl|rnb-route-csv] \
     [--policy lru,static-hot,lru-static-hot,static-hot-adaptive,layer-quota-static,rank-weighted-static,gate-up-static,least-stale,router-current-jit,prev-step-prefetch,online-layer-hot-prefetch,online-layer-hot-staged,adaptive-lfu-lru,window-lfu-lru] \
     [--warmup-steps N] [--eval-start-step N] [--static-hot-pct LIST] [--window-steps N] \
     [--predictor-recall-pct LIST] [--predictor-extra-ratio LIST] [--json-out PATH]\n\
     note: --window-steps is LFU window for window-lfu-lru and lookahead group count for least-stale\n\n\
     usage: rnb-moe-tier predictor --trace PATH [--source router-top,router-current,prev-same-layer,prev-layer-union,online-layer-hot,combined-scored] [--lookahead-groups LIST] [--top-n LIST] [--skip-groups N] [--json-out PATH]\n\
     usage: rnb-moe-tier route-shape --trace PATH [--trace-format jsonl|rnb-route-csv] [--max-group N] [--json-out PATH]"
        .to_string()
}

fn parse_trace_format(raw: &str) -> Result<TraceFormat, String> {
    match raw {
        "jsonl" => Ok(TraceFormat::Jsonl),
        "rnb-route-csv" => Ok(TraceFormat::RnbRouteCsv),
        other => Err(format!("unknown trace format: {other}")),
    }
}

fn parse_trace(text: &str, format: TraceFormat) -> Result<Vec<TraceEvent>, String> {
    match format {
        TraceFormat::Jsonl => parse_jsonl_trace(text),
        TraceFormat::RnbRouteCsv => parse_rnb_route_csv(text),
    }
}

fn parse_policies(raw: &str) -> Result<Vec<CachePolicy>, String> {
    raw.split(',')
        .map(CachePolicy::from_str)
        .collect::<Result<Vec<_>, _>>()
}

fn parse_cache_gb(raw: &str) -> Result<Vec<u64>, String> {
    raw.split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<u64>()
                .map_err(|e| format!("bad --cache-gb value {part}: {e}"))
        })
        .collect()
}

fn parse_u8_list(raw: &str, name: &str) -> Result<Vec<u8>, String> {
    raw.split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<u8>()
                .map_err(|e| format!("bad {name} value {part}: {e}"))
        })
        .collect()
}

fn parse_u16_list(raw: &str, name: &str) -> Result<Vec<u16>, String> {
    raw.split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<u16>()
                .map_err(|e| format!("bad {name} value {part}: {e}"))
        })
        .collect()
}

fn parse_usize_list(raw: &str, name: &str) -> Result<Vec<usize>, String> {
    raw.split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<usize>()
                .map_err(|e| format!("bad {name} value {part}: {e}"))
        })
        .collect()
}

fn parse_predictor_sources(raw: &str) -> Result<Vec<PredictorSource>, String> {
    raw.split(',')
        .map(str::trim)
        .map(|part| match part {
            "router-top" => Ok(PredictorSource::RouterTop),
            "router-current" => Ok(PredictorSource::RouterCurrent),
            "prev-same-layer" => Ok(PredictorSource::PrevSameLayer),
            "prev-layer-union" => Ok(PredictorSource::PrevLayerUnion),
            "online-layer-hot" => Ok(PredictorSource::OnlineLayerHot),
            "combined-scored" => Ok(PredictorSource::CombinedScored),
            other => Err(format!("bad --source value {other}")),
        })
        .collect()
}

fn read_json<T>(path: &PathBuf) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e))
}

#[derive(serde::Serialize)]
struct OutputRow {
    policy: String,
    #[serde(flatten)]
    row: PcSimulationRow,
}

fn print_table(model: &str, hardware: &str, events: usize, rows: &[OutputRow]) {
    println!("model={model} hardware={hardware} events={events}");
    println!(
        "policy             vram_cache  warmup  eval_from  static  window  recall  extra  eval_steps  hit_rate  miss_rate  miss_bytes/token  copy_ms/token  resident_entries  break_even_hit_rate"
    );
    for output in rows {
        let row = &output.row;
        println!(
            "{:<18} {:>10} {:>7} {:>9} {:>6}% {:>7} {:>6}% {:>5}% {:>11} {:>8.1}% {:>9.1}% {:>17} {:>14.2} {:>17} {:>19}",
            output.policy,
            format_bytes(row.cache_bytes),
            row.warmup_steps,
            row.eval_start_step,
            row.static_hot_pct,
            row.window_steps,
            row.predictor_recall_pct,
            row.predictor_extra_ratio,
            row.evaluated_steps,
            row.hit_rate * 100.0,
            row.miss_rate * 100.0,
            format_bytes(row.miss_bytes_per_token as u64),
            row.copy_ms_per_token,
            row.resident_entries,
            row.break_even_hit_rate
                .map(|v| format!("{:.1}%", v * 100.0))
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }
}

fn print_predictor_table(events: usize, skip_groups: usize, rows: &[PredictorAnalysisRow]) {
    println!("predictor_events={events} skip_groups={skip_groups}");
    println!(
        "source           lookahead  top_n  samples  avg_recall  avg_precision  false_pos_ratio"
    );
    for row in rows {
        println!(
            "{:<17} {:>9} {:>6} {:>8} {:>10.1}% {:>13.1}% {:>16.2}",
            row.source,
            row.lookahead_groups,
            row.top_n,
            row.samples,
            row.avg_recall * 100.0,
            row.avg_precision * 100.0,
            row.avg_false_positive_ratio,
        );
    }
}

fn print_route_shape_table(analysis: &RouteShapeAnalysis) {
    let full_pct = if analysis.total_groups == 0 {
        0.0
    } else {
        (analysis.full_groups as f64 * 100.0) / analysis.total_groups as f64
    };
    println!(
        "route_shape max_group={} events={} layers={} groups={} full_groups={} full_group_pct={:.1}%",
        analysis.max_group,
        analysis.total_events,
        analysis.layers.len(),
        analysis.total_groups,
        analysis.full_groups,
        full_pct
    );
    print!("layer  events  experts  groups  full  max_run");
    for len in 1..=analysis.max_group {
        print!("  len{len}");
    }
    println!();
    for layer in &analysis.layers {
        print!(
            "{:>5} {:>7} {:>8} {:>7} {:>5} {:>8}",
            layer.layer,
            layer.events,
            layer.unique_experts,
            layer.groups,
            layer.full_groups,
            layer.max_run_len,
        );
        for len in 1..=analysis.max_group {
            print!(" {:>5}", layer.len_hist.get(len).copied().unwrap_or(0));
        }
        println!();
    }
}

fn policy_name(policy: CachePolicy) -> &'static str {
    match policy {
        CachePolicy::Lru => "lru",
        CachePolicy::StaticHot => "static-hot",
        CachePolicy::StaticHotLru => "lru-static-hot",
        CachePolicy::StaticHotAdaptive => "static-hot-adaptive",
        CachePolicy::LayerQuotaStatic => "layer-quota-static",
        CachePolicy::RankWeightedStatic => "rank-weighted-static",
        CachePolicy::GateUpStatic => "gate-up-static",
        CachePolicy::LeastStale => "least-stale",
        CachePolicy::RouterCurrentJit => "router-current-jit",
        CachePolicy::PrevStepPrefetch => "prev-step-prefetch",
        CachePolicy::OnlineLayerHotPrefetch => "online-layer-hot",
        CachePolicy::OnlineLayerHotStaged => "online-layer-hot-staged",
        CachePolicy::AdaptiveLfuLru => "adaptive-lfu-lru",
        CachePolicy::WindowLfuLru => "window-lfu-lru",
    }
}

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    if bytes >= GB && bytes % GB == 0 {
        format!("{}GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else {
        format!("{bytes}B")
    }
}
