//! Expert-popularity profiler for MoE GGUF models.
//!
//! Loads the model, runs prefill on each prompt in the given corpus files,
//! and records how often each `(layer, expert)` was selected by the top-k
//! router. Outputs a JSON sidecar consumed by `rnb-convert` (Session 64
//! axis B) to produce a hot-sorted `.rnb`.
//!
//! The recording is driven by the global `moe_trace` state inside the
//! engine: when `moe_trace::is_active()` returns true, each call to
//! `MoeLayerView::forward_with_logits` appends the selected top-k expert
//! ids to an atomic histogram. The profiler post-processes that snapshot
//! into per-layer counts and prints a short heavy-tail summary.
//!
//! Usage:
//!   rnb-moe-profile <gguf-path> <out.json> <corpus1.txt> [<corpus2.txt> ...] \
//!       [--max-tokens-per-prompt N] [--max-prompts-per-corpus N]
//!
//! Corpus file format: one prompt per line, blank lines skipped.
//!
//! Output JSON schema (in4):
//! ```json
//! {
//!   "gguf_path": "...",
//!   "n_layer": <usize>,
//!   "n_expert": <usize>,
//!   "n_expert_used": <usize>,
//!   "total_prompts": <usize>,
//!   "total_tokens": <usize>,
//!   "max_tokens_per_prompt": <usize>,
//!   "max_prompts_per_corpus": <usize>,
//!   "corpora": [<paths>],
//!   "hit_counts": [
//!     [<u64>; n_expert],   // layer 0
//!     [<u64>; n_expert],   // layer 1
//!     ...
//!   ]
//! }
//! ```
//!
//! `hit_counts[layer][expert]` is the raw selection count for that
//! `(layer, expert)` pair across all profiled tokens. Diagnostic conversion
//! callers that need `popularity_order` (rank → original_id) sort each row
//! descending and use the resulting indices.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rnb_llm::{moe_trace, Engine};
use rnb_loader::load_model;

struct Args {
    gguf: PathBuf,
    out_json: PathBuf,
    corpora: Vec<PathBuf>,
    max_tokens: usize,
    max_prompts: usize,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.len() < 3 {
        return Err(usage());
    }

    let mut max_tokens = 128usize;
    let mut max_prompts = 50usize;
    let mut positional = Vec::<String>::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--max-tokens-per-prompt" => {
                i += 1;
                let v = raw.get(i).ok_or_else(|| usage())?;
                max_tokens = v
                    .parse::<usize>()
                    .map_err(|e| format!("bad --max-tokens-per-prompt: {}", e))?;
            }
            "--max-prompts-per-corpus" => {
                i += 1;
                let v = raw.get(i).ok_or_else(|| usage())?;
                max_prompts = v
                    .parse::<usize>()
                    .map_err(|e| format!("bad --max-prompts-per-corpus: {}", e))?;
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag: {}", flag));
            }
            _ => positional.push(raw[i].clone()),
        }
        i += 1;
    }

    if positional.len() < 3 {
        return Err(usage());
    }
    Ok(Args {
        gguf: PathBuf::from(&positional[0]),
        out_json: PathBuf::from(&positional[1]),
        corpora: positional[2..].iter().map(PathBuf::from).collect(),
        max_tokens,
        max_prompts,
    })
}

fn usage() -> String {
    "usage: rnb-moe-profile <gguf-path> <out.json> <corpus.txt> [<corpus.txt> ...]\n\
         \x20          [--max-tokens-per-prompt N] [--max-prompts-per-corpus N]"
        .to_string()
}

fn read_corpus(path: &Path, max_prompts: usize) -> std::io::Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .take(max_prompts)
        .map(|s| s.to_string())
        .collect())
}

fn write_json(
    out: &Path,
    args: &Args,
    shape: (usize, usize, usize),
    hits: &[Vec<u64>],
    stats: &RunStats,
) -> std::io::Result<()> {
    let (n_layer, n_expert, n_used) = shape;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!(
        "  \"gguf_path\": \"{}\",\n",
        args.gguf
            .display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    ));
    s.push_str(&format!("  \"n_layer\": {},\n", n_layer));
    s.push_str(&format!("  \"n_expert\": {},\n", n_expert));
    s.push_str(&format!("  \"n_expert_used\": {},\n", n_used));
    s.push_str(&format!("  \"total_prompts\": {},\n", stats.total_prompts));
    s.push_str(&format!("  \"total_tokens\": {},\n", stats.total_tokens));
    s.push_str(&format!(
        "  \"max_tokens_per_prompt\": {},\n",
        args.max_tokens
    ));
    s.push_str(&format!(
        "  \"max_prompts_per_corpus\": {},\n",
        args.max_prompts
    ));
    s.push_str("  \"corpora\": [\n");
    for (i, c) in args.corpora.iter().enumerate() {
        let sep = if i + 1 == args.corpora.len() { "" } else { "," };
        s.push_str(&format!(
            "    \"{}\"{}\n",
            c.display()
                .to_string()
                .replace('\\', "\\\\")
                .replace('"', "\\\""),
            sep
        ));
    }
    s.push_str("  ],\n");
    s.push_str("  \"hit_counts\": [\n");
    for (l, row) in hits.iter().enumerate() {
        s.push_str("    [");
        for (e, v) in row.iter().enumerate() {
            if e > 0 {
                s.push_str(", ");
            }
            s.push_str(&v.to_string());
        }
        let sep = if l + 1 == hits.len() { "" } else { "," };
        s.push_str(&format!("]{}\n", sep));
    }
    s.push_str("  ]\n");
    s.push_str("}\n");

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = fs::File::create(out)?;
    f.write_all(s.as_bytes())?;
    Ok(())
}

#[derive(Default)]
struct RunStats {
    total_prompts: usize,
    total_tokens: usize,
}

fn heavy_tail_summary(hits: &[Vec<u64>], n_expert_used: usize) {
    // Summary across all MoE layers combined. Layers whose counts are all
    // zero (non-MoE layers or never-hit layers) are skipped.
    let mut layer_summaries = Vec::<(usize, f64, f64, f64)>::new(); // (layer, top10%, top30%, top50%)
    let mut global_hits = vec![0u64; hits.first().map(|r| r.len()).unwrap_or(0)];
    let mut moe_layer_count = 0usize;

    for (l, row) in hits.iter().enumerate() {
        let total: u64 = row.iter().sum();
        if total == 0 {
            continue;
        }
        moe_layer_count += 1;
        let mut sorted: Vec<u64> = row.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let n = sorted.len();
        let top10 = (n as f64 * 0.10).ceil() as usize;
        let top30 = (n as f64 * 0.30).ceil() as usize;
        let top50 = (n as f64 * 0.50).ceil() as usize;
        let sum10: u64 = sorted[..top10].iter().sum();
        let sum30: u64 = sorted[..top30].iter().sum();
        let sum50: u64 = sorted[..top50].iter().sum();
        let total_f = total as f64;
        layer_summaries.push((
            l,
            sum10 as f64 / total_f,
            sum30 as f64 / total_f,
            sum50 as f64 / total_f,
        ));
        for (e, v) in row.iter().enumerate() {
            global_hits[e] = global_hits[e].saturating_add(*v);
        }
    }

    if moe_layer_count == 0 {
        eprintln!("no MoE activity recorded — is this a MoE model?");
        return;
    }

    eprintln!(
        "\n[heavy-tail] per-layer cumulative hit fraction (expected k/n = {:.2}% just from random):",
        100.0 * n_expert_used as f64 / hits.first().map(|r| r.len()).unwrap_or(1) as f64
    );
    eprintln!("  layer | top10% | top30% | top50%");
    for (l, t10, t30, t50) in &layer_summaries {
        eprintln!(
            "  {:>5} | {:>5.1}% | {:>5.1}% | {:>5.1}%",
            l,
            100.0 * t10,
            100.0 * t30,
            100.0 * t50
        );
    }

    // Global aggregate (sum over all MoE layers, treated as one distribution).
    let total: u64 = global_hits.iter().sum();
    let mut sorted: Vec<u64> = global_hits.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let n = sorted.len();
    let top10 = (n as f64 * 0.10).ceil() as usize;
    let top30 = (n as f64 * 0.30).ceil() as usize;
    let top50 = (n as f64 * 0.50).ceil() as usize;
    let s10: u64 = sorted[..top10].iter().sum();
    let s30: u64 = sorted[..top30].iter().sum();
    let s50: u64 = sorted[..top50].iter().sum();
    let tf = total as f64;
    eprintln!(
        "\n[heavy-tail] global (all MoE layers combined): top10%={:.1}%  top30%={:.1}%  top50%={:.1}%",
        100.0 * s10 as f64 / tf,
        100.0 * s30 as f64 / tf,
        100.0 * s50 as f64 / tf,
    );

    if layer_summaries.iter().any(|(_, _, t30, _)| *t30 >= 0.75) {
        eprintln!(
            "\n[heavy-tail] ✅ at least one layer clears the S64 axis-B threshold (top30% ≥ 75%)."
        );
    } else {
        eprintln!(
            "\n[heavy-tail] ⚠ no layer clears top30% ≥ 75%. Consider larger hot_frac in axis B or dynamic prefill-aware cache."
        );
    }
}

fn main() -> ExitCode {
    // Forward panics from rayon workers / main to stderr so we don't lose
    // the traceback when stdout is redirected to a file.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("\n=== PANIC ===\n{info}\n=============");
    }));

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::from(2);
        }
    };

    eprintln!("probing metadata: {}", args.gguf.display());
    let (n_layer, n_expert, n_used) = match load_model(&args.gguf) {
        Ok(loaded) => (
            loaded.metadata.num_layers,
            loaded.metadata.expert_count,
            loaded.metadata.expert_used_count,
        ),
        Err(e) => {
            eprintln!("load_model failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "  n_layer={} n_expert={} n_expert_used={}",
        n_layer, n_expert, n_used
    );
    if n_expert == 0 {
        eprintln!("error: this model has no MoE experts (expert_count = 0)");
        return ExitCode::from(1);
    }

    eprintln!("loading engine: {}", args.gguf.display());
    let mut engine = match Engine::from_gguf(&args.gguf) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Engine::from_gguf failed: {:?}", e);
            return ExitCode::from(1);
        }
    };

    moe_trace::init(n_layer, n_expert);
    moe_trace::reset();
    moe_trace::enable();

    let mut stats = RunStats::default();
    let t_all = Instant::now();
    for corpus_path in &args.corpora {
        let prompts = match read_corpus(corpus_path, args.max_prompts) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("corpus read failed: {}: {}", corpus_path.display(), e);
                return ExitCode::from(1);
            }
        };
        eprintln!(
            "\n[corpus] {} ({} prompts, cap {}/corpus, max {} tok/prompt)",
            corpus_path.display(),
            prompts.len(),
            args.max_prompts,
            args.max_tokens
        );
        for (i, p) in prompts.iter().enumerate() {
            eprintln!(
                "  prompt {:>3}: encoding '{}'...",
                i,
                &p.chars().take(40).collect::<String>()
            );
            let mut tokens = engine.tokenizer.encode(p);
            if tokens.len() > args.max_tokens {
                tokens.truncate(args.max_tokens);
            }
            if tokens.is_empty() {
                eprintln!("  prompt {:>3}: empty after encode, skipping", i);
                continue;
            }
            eprintln!(
                "  prompt {:>3}: {} tok, kv_cache.current_len={}, forwarding...",
                i,
                tokens.len(),
                engine.kv_cache.current_len()
            );
            let t0 = Instant::now();
            match engine.forward(&tokens) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "  prompt {}: forward failed ({} tok): {:?}",
                        i,
                        tokens.len(),
                        e
                    );
                    // Still try to clear KV state before next prompt to
                    // avoid cascading failures.
                    engine.kv_cache.clear();
                    continue;
                }
            }
            let elapsed = t0.elapsed();
            stats.total_prompts += 1;
            stats.total_tokens += tokens.len();
            eprintln!("  prompt {:>3}: done in {:.2}s", i, elapsed.as_secs_f64());

            // Clear KV cache between prompts so each prompt is a fresh
            // prefill (MoE routing depends on attention output; we want
            // per-prompt independent statistics).
            engine.kv_cache.clear();
        }
    }
    let wall = t_all.elapsed();
    eprintln!(
        "\n[done] {} prompts, {} tokens, wall={:.2}s",
        stats.total_prompts,
        stats.total_tokens,
        wall.as_secs_f64()
    );

    moe_trace::disable();
    let hits = match moe_trace::snapshot() {
        Some(h) => h,
        None => {
            eprintln!("moe_trace snapshot missing — did init fail?");
            return ExitCode::from(1);
        }
    };

    heavy_tail_summary(&hits, n_used);

    if let Err(e) = write_json(
        &args.out_json,
        &args,
        (n_layer, n_expert, n_used),
        &hits,
        &stats,
    ) {
        eprintln!("failed to write {}: {}", args.out_json.display(), e);
        return ExitCode::from(1);
    }
    eprintln!("\n[wrote] {}", args.out_json.display());

    ExitCode::SUCCESS
}
