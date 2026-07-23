fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_targets(raw: &str) -> Vec<String> {
    raw.split(';')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_layers(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn print_target_ranks(
    engine: &rnb_llm::Engine,
    ranked: &[(usize, f32)],
    targets: &[String],
    max_matches: usize,
) {
    for target in targets {
        let mut matches = Vec::new();
        for (rank, (id, val)) in ranked.iter().enumerate() {
            let piece = engine.tokenizer.decode_token(*id as u32);
            if piece == *target || piece.starts_with(target) {
                matches.push((rank + 1, *id as u32, *val, piece));
                if matches.len() >= max_matches {
                    break;
                }
            }
        }
        if matches.is_empty() {
            println!("{target:?}: no match");
        } else {
            println!("{target:?}:");
            for (rank, id, val, piece) in matches {
                println!("  rank=#{rank} id={id} logit={val:.4} piece={piece:?}");
            }
        }
    }
}

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let prompt = std::env::var("RNB_PROMPT").unwrap_or_else(|_| "대한민국의 수도는".to_string());
    let no_bos = std::env::var("RNB_NO_BOS").is_ok();
    let top_k = env_or("RNB_GEMMA4_TARGET_RANK_TOPK", 20usize);
    let layerwise = std::env::var("RNB_GEMMA4_TARGET_RANK_LAYERWISE")
        .ok()
        .map(|raw| parse_layers(&raw))
        .unwrap_or_default();
    let targets = parse_targets(
        &std::env::var("RNB_GEMMA4_TARGET_RANK_TARGETS").unwrap_or_else(|_| {
            "서울; 서울;서울특별시; 서울특별시;입니다; 입니다;수도; 수도;대한민국;대한;민국"
                .to_string()
        }),
    );

    println!("[gemma4-target-rank-probe] model_path = {model_path}");
    println!("[gemma4-target-rank-probe] prompt     = {prompt:?}");
    println!("[gemma4-target-rank-probe] no_bos     = {no_bos}");
    println!("[gemma4-target-rank-probe] top_k      = {top_k}");
    println!("[gemma4-target-rank-probe] layerwise  = {:?}", layerwise);
    println!("[gemma4-target-rank-probe] targets    = {:?}", targets);

    let mut engine = rnb_llm::Engine::from_gguf(std::path::Path::new(&model_path))
        .expect("Engine::from_gguf failed");
    let bos_id = engine.tokenizer.vocab.special.bos;
    let mut tokens = Vec::new();
    if !no_bos {
        tokens.push(bos_id);
    }
    tokens.extend(engine.tokenizer.encode(&prompt));

    let logits = engine.forward(&tokens).expect("forward failed");
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== top-{top_k} logits ===");
    for (rank, (id, val)) in ranked.iter().take(top_k).enumerate() {
        let piece = engine
            .tokenizer
            .decode_token(*id as u32)
            .replace('\n', "\\n");
        println!(
            "#{:03} id={} logit={:.4} piece={:?}",
            rank + 1,
            id,
            val,
            piece
        );
    }

    println!("\n=== target ranks ===");
    print_target_ranks(&engine, &ranked, &targets, 5);

    if !layerwise.is_empty() {
        println!("\n=== layerwise target ranks ===");
        let layer_logits = engine
            .debug_prefill_layer_logits(&tokens)
            .expect("debug_prefill_layer_logits failed");
        for layer_idx in layerwise {
            let Some(logits) = layer_logits.get(layer_idx) else {
                continue;
            };
            let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("\n--- after layer {layer_idx} ---");
            if let Some((id, val)) = ranked.first() {
                let piece = engine
                    .tokenizer
                    .decode_token(*id as u32)
                    .replace('\n', "\\n");
                println!("top1: id={id} logit={val:.4} piece={piece:?}");
            }
            print_target_ranks(&engine, &ranked, &targets, 3);
        }
    }
}
