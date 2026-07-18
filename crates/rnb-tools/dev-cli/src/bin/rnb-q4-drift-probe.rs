use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rnb_llm::{Engine, PrefillDriftRecord, PrefillDriftTrace};

fn parse_token_list(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.parse::<u32>().expect("invalid token id"))
            }
        })
        .collect()
}

fn load_prompt() -> String {
    if let Ok(prompt) = std::env::var("RNB_PROMPT") {
        return prompt;
    }
    if let Ok(path) = std::env::var("RNB_PROMPT_FILE") {
        return fs::read_to_string(path).expect("failed to read RNB_PROMPT_FILE");
    }
    "Hello".to_string()
}

fn build_tokens(engine: &Engine, prompt: &str) -> Vec<u32> {
    if let Ok(raw) = std::env::var("RNB_DRIFT_FORCE_TOKENS") {
        return parse_token_list(&raw);
    }

    let mut tokens = Vec::new();
    if std::env::var("RNB_NO_BOS").is_err() && engine.tokenizer.should_add_bos() {
        tokens.push(engine.tokenizer.vocab.special.bos);
    }
    tokens.extend(engine.tokenizer.encode(prompt));
    tokens
}

fn json_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn record_basename(record: &PrefillDriftRecord) -> String {
    match (record.layer_idx, record.stage) {
        (None, "embedding_scaled") => "embedding_scaled".to_string(),
        (None, "final_normed") => "final_normed".to_string(),
        (Some(layer), "layer_output") => format!("layer_{layer:03}_output"),
        (Some(layer), "final_gemma_per_layer_output") => {
            format!("layer_{layer:03}_final_gemma_per_layer_output")
        }
        (Some(layer), stage) => format!("layer_{layer:03}_{stage}"),
        (None, stage) => stage.to_string(),
    }
}

fn write_f32_bin(path: &Path, row: &[f32]) -> io::Result<()> {
    let mut file = File::create(path)?;
    for value in row {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_manifest(
    dump_dir: &Path,
    model_path: &str,
    prompt: &str,
    trace: &PrefillDriftTrace,
) -> io::Result<()> {
    let mut file = File::create(dump_dir.join("manifest.json"))?;

    writeln!(file, "{{")?;
    writeln!(file, "  \"source\": \"rust-q4-native\",")?;
    writeln!(file, "  \"model\": \"{}\",", json_escape(model_path))?;
    writeln!(file, "  \"prompt\": \"{}\",", json_escape(prompt))?;
    writeln!(file, "  \"hidden_dim\": {},", trace.hidden_dim)?;
    writeln!(file, "  \"seq_len\": {},", trace.seq_len)?;
    write!(file, "  \"tokens\": [")?;
    for (idx, token) in trace.tokens.iter().enumerate() {
        if idx > 0 {
            write!(file, ", ")?;
        }
        write!(file, "{token}")?;
    }
    writeln!(file, "],")?;
    writeln!(file, "  \"records\": [")?;
    for (idx, record) in trace.records.iter().enumerate() {
        let basename = record_basename(record);
        let layer = record
            .layer_idx
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let comma = if idx + 1 == trace.records.len() {
            ""
        } else {
            ","
        };
        writeln!(
            file,
            "    {{\"name\": \"{}\", \"layer\": {}, \"stage\": \"{}\", \"path\": \"{}.bin\"}}{}",
            json_escape(&basename),
            layer,
            json_escape(record.stage),
            json_escape(&basename),
            comma
        )?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    Ok(())
}

fn write_trace(
    dump_dir: &Path,
    model_path: &str,
    prompt: &str,
    trace: &PrefillDriftTrace,
) -> io::Result<()> {
    fs::create_dir_all(dump_dir)?;
    for record in &trace.records {
        let basename = record_basename(record);
        write_f32_bin(&dump_dir.join(format!("{basename}.bin")), &record.row)?;
    }
    write_manifest(dump_dir, model_path, prompt, trace)
}

fn main() {
    let model_path = std::env::var("RNB_MODEL").expect("RNB_MODEL is required");
    let dump_dir = PathBuf::from(
        std::env::var("RNB_DRIFT_DUMP_DIR")
            .unwrap_or_else(|_| "/tmp/rnb-q4-drift-rust".to_string()),
    );
    let prompt = load_prompt();

    let engine = Engine::from_gguf(Path::new(&model_path)).expect("Engine::from_gguf failed");
    let tokens = build_tokens(&engine, &prompt);
    let trace = engine
        .debug_prefill_drift_layer_outputs(&tokens)
        .expect("debug_prefill_drift_layer_outputs failed");

    write_trace(&dump_dir, &model_path, &prompt, &trace).expect("failed to write drift trace");

    println!("[rnb-q4-drift-probe] model={model_path}");
    println!("[rnb-q4-drift-probe] dump_dir={}", dump_dir.display());
    println!(
        "[rnb-q4-drift-probe] seq_len={} hidden_dim={} records={}",
        trace.seq_len,
        trace.hidden_dim,
        trace.records.len()
    );
    println!("[rnb-q4-drift-probe] tokens={:?}", trace.tokens);
    for record in &trace.records {
        println!(
            "[rnb-q4-drift-probe] wrote {}.bin len={}",
            record_basename(record),
            record.row.len()
        );
    }
}
