fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_configs(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_token_ids(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<u32>().ok())
        .collect()
}

fn apply_config(cfg: &str, tracked_keys: &[&str]) {
    for key in tracked_keys {
        unsafe {
            std::env::remove_var(key);
        }
    }
    if cfg == "baseline" {
        return;
    }

    for pair in cfg.split_whitespace() {
        if let Some((key, value)) = pair.split_once('=') {
            unsafe {
                std::env::set_var(key, value);
            }
        } else {
            unsafe {
                std::env::set_var(pair, "1");
            }
        }
    }
}

fn suppress_selected_pieces(engine: &rnb_llm::Engine, logits: &mut [f32]) {
    let Ok(raw) = std::env::var("RNB_SUPPRESS_PIECES") else {
        return;
    };
    let targets = raw.split(';').filter(|s| !s.is_empty()).collect::<Vec<_>>();
    if targets.is_empty() {
        return;
    }
    for (id, logit) in logits.iter_mut().enumerate() {
        let piece = engine.tokenizer.decode_token(id as u32);
        if targets.iter().any(|target| piece == *target) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn top_token(engine: &rnb_llm::Engine, logits: &[f32]) -> (usize, f32, String) {
    let (id, val) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let piece = engine
        .tokenizer
        .decode_token(id as u32)
        .replace('\n', "\\n");
    (id, val, piece)
}

fn best_rank(
    engine: &rnb_llm::Engine,
    logits: &[f32],
    targets: &[&str],
) -> Option<(usize, String, f32)> {
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut best = None;
    for (rank, (id, val)) in ranked.iter().enumerate() {
        let piece = engine.tokenizer.decode_token(*id as u32);
        if targets
            .iter()
            .any(|target| piece == *target || piece.starts_with(target))
        {
            best = Some((rank + 1, piece, *val));
            break;
        }
    }
    best
}

fn decode_preview(engine: &mut rnb_llm::Engine, mut logits: Vec<f32>, max_steps: usize) -> String {
    let mut out = String::new();
    for _ in 0..max_steps {
        suppress_selected_pieces(engine, &mut logits);
        let (token_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let piece = engine.tokenizer.decode_token(token_id as u32);
        out.push_str(&piece);
        if piece == "</s>" || piece == "<eos>" {
            break;
        }
        logits = engine
            .forward(&[token_id as u32])
            .unwrap_or_else(|_| vec![0.0; engine.metadata.vocab_size]);
    }
    out.replace('\n', "\\n")
}

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let prompt = std::env::var("RNB_PROMPT").unwrap_or_else(|_| "대한민국의 수도는".to_string());
    let no_bos = std::env::var("RNB_NO_BOS").is_ok();
    let decode_steps = env_or("RNB_GEMMA4_LAYER_ABLATION_DECODE_STEPS", 0usize);
    let decode_token_id = std::env::var("RNB_GEMMA4_LAYER_ABLATION_DECODE_TOKEN_ID")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    let force_tokens = std::env::var("RNB_GEMMA4_LAYER_ABLATION_FORCE_TOKENS")
        .ok()
        .map(|raw| parse_token_ids(&raw))
        .unwrap_or_default();
    let configs = parse_configs(
        &std::env::var("RNB_GEMMA4_LAYER_ABLATION_CONFIGS").unwrap_or_else(|_| {
            "baseline;\
RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER=13;\
RNB_GEMMA_DISABLE_ATTN_LAYER=13;\
RNB_GEMMA_SKIP_POST_ATTN_LAYER=13"
                .to_string()
        }),
    );
    let tracked_keys = [
        "RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER",
        "RNB_GEMMA_SKIP_POST_ATTN_LAYER",
        "RNB_GEMMA_DISABLE_ATTN_LAYER",
        "RNB_GEMMA_SKIP_FFN_LAYER",
        "RNB_GEMMA_SKIP_OUT_SCALE_LAYER",
        "RNB_GEMMA_SHARED_KV_SOURCE_SWA",
        "RNB_GEMMA_SHARED_KV_SOURCE_FULL",
    ];
    let targets = ["서울", "서울특별시", " 서울"];

    println!("[gemma4-layer-ablation-probe] model_path = {model_path}");
    println!("[gemma4-layer-ablation-probe] prompt     = {prompt:?}");
    println!("[gemma4-layer-ablation-probe] no_bos     = {no_bos}");
    println!("[gemma4-layer-ablation-probe] decode     = {decode_steps}");
    println!(
        "[gemma4-layer-ablation-probe] decode_tok = {:?}",
        decode_token_id
    );
    println!(
        "[gemma4-layer-ablation-probe] force_toks = {:?}",
        force_tokens
    );
    println!("[gemma4-layer-ablation-probe] configs    = {:?}", configs);

    let mut engine = rnb_llm::Engine::from_gguf(std::path::Path::new(&model_path))
        .expect("Engine::from_gguf failed");
    let bos_id = engine.tokenizer.vocab.special.bos;
    let mut tokens = Vec::new();
    if !no_bos {
        tokens.push(bos_id);
    }
    tokens.extend(engine.tokenizer.encode(&prompt));

    for cfg in configs {
        apply_config(&cfg, &tracked_keys);
        engine.kv_cache.clear();

        let mut logits = engine.forward(&tokens).expect("forward failed");
        if let Some(token_id) = decode_token_id {
            logits = engine
                .forward(&[token_id])
                .expect("decode-token forward failed");
        } else if !force_tokens.is_empty() {
            for &token_id in &force_tokens {
                logits = engine
                    .forward(&[token_id])
                    .expect("force-token forward failed");
            }
        }
        suppress_selected_pieces(&engine, &mut logits);
        let (top_id, top_logit, top_piece) = top_token(&engine, &logits);
        let best = best_rank(&engine, &logits, &targets);

        println!("\n=== {cfg} ===");
        println!("top1: id={top_id} logit={top_logit:.4} piece={top_piece:?}");
        match best {
            Some((rank, piece, logit)) => {
                println!("best_target: rank=#{rank} logit={logit:.4} piece={piece:?}");
            }
            None => {
                println!("best_target: not found");
            }
        }

        if decode_steps > 0 {
            let preview = decode_preview(&mut engine, logits.clone(), decode_steps);
            println!("decode_preview: {preview:?}");
        }
    }
}
