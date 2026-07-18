// Gemma4 diagnostic seam: dump first-decode-step top-K logits and rank of target Korean tokens.
//
// Usage:
//   RNB_MODEL=/data/local/tmp/rnb/gemma-4-E2B-it-Q4_K_M.gguf \
//   RNB_PROMPT="대한민국의 수도는" \
//   RNB_NO_BOS=1 RNB_GEMMA4_FIRST_DECODE_TOPK=50 \
//   ./rnb-gemma4-first-decode-logits
//
// Printed:
//   prompt token ids
//   top-K logits after prefill (= first decode step distribution)
//   rank + logit value of every target piece found in vocab (e.g., "서울", "수도", "대한민국")

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_ranked_logits(
    engine: &rnb_llm::Engine,
    logits: &[f32],
    top_k: usize,
    label: &str,
    targets: &[&str],
) {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== top-{top_k} logits ({label}) ===");
    for (rank, (id, val)) in ranked.iter().take(top_k).enumerate() {
        let piece = engine
            .tokenizer
            .decode_token(*id as u32)
            .replace('\n', "\\n");
        println!(
            "#{:03} id={:<7} logit={:10.4} piece={:?}",
            rank + 1,
            id,
            val,
            piece
        );
    }

    println!("\n=== target token ranks ({label}) ===");
    for target in targets {
        let mut best: Option<(usize, u32, f32, String)> = None;
        for (rank, (id, val)) in ranked.iter().enumerate() {
            let piece = engine.tokenizer.decode_token(*id as u32);
            if piece == *target || piece.starts_with(target) {
                if best.as_ref().map(|(r, _, _, _)| rank < *r).unwrap_or(true) {
                    best = Some((rank, *id as u32, *val, piece));
                    break;
                }
            }
        }
        match best {
            Some((rank, id, val, piece)) => println!(
                "  {target:?}: rank=#{}, id={}, logit={:.4}, piece={:?}",
                rank + 1,
                id,
                val,
                piece
            ),
            None => println!("  {target:?}: NOT FOUND in vocab"),
        }
    }
}

fn parse_layer_list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn main() {
    let model_path =
        std::env::var("RNB_MODEL").unwrap_or_else(|_| "gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let prompt = std::env::var("RNB_PROMPT").unwrap_or_else(|_| "대한민국의 수도는".to_string());
    let top_k: usize = env_or("RNB_GEMMA4_FIRST_DECODE_TOPK", 50);
    let no_bos = std::env::var("RNB_NO_BOS").is_ok();
    let layerwise = std::env::var("RNB_GEMMA4_FIRST_DECODE_LAYERWISE").is_ok();
    let tokenwise_prompt = std::env::var("RNB_GEMMA4_FIRST_DECODE_TOKENWISE_PROMPT").is_ok();
    let snapshot_layers = std::env::var("RNB_GEMMA4_FIRST_DECODE_SNAPSHOT_LAYERS")
        .ok()
        .map(|s| parse_layer_list(&s))
        .unwrap_or_default();
    let targets = [
        "서울",
        "수도",
        "대한민국",
        "대한",
        "민국",
        "수도는",
        " 서울",
    ];

    println!("[gemma4-first-decode-logits] model_path = {model_path}");
    println!("[gemma4-first-decode-logits] prompt     = {prompt:?}");
    println!("[gemma4-first-decode-logits] top_k      = {top_k}");
    println!("[gemma4-first-decode-logits] no_bos     = {no_bos}");
    println!("[gemma4-first-decode-logits] layerwise  = {layerwise}");
    println!("[gemma4-first-decode-logits] tokenwise  = {tokenwise_prompt}");
    println!(
        "[gemma4-first-decode-logits] snapshots = {:?}",
        snapshot_layers
    );

    let mut engine = rnb_llm::Engine::from_gguf(&std::path::PathBuf::from(&model_path))
        .expect("Engine::from_gguf failed");

    let bos_id = engine.tokenizer.vocab.special.bos;
    let mut tokens: Vec<u32> = Vec::new();
    if !no_bos {
        tokens.push(bos_id);
    }
    tokens.extend(engine.tokenizer.encode(&prompt));

    println!("\n=== prompt tokens (n={}) ===", tokens.len());
    for (i, &id) in tokens.iter().enumerate() {
        let piece = engine.tokenizer.decode_token(id).replace('\n', "\\n");
        println!("#{i:02} id={id} piece={piece:?}");
    }

    println!("\n=== running prefill + first decode step ===");
    let t0 = std::time::Instant::now();
    let logits = engine.forward(&tokens).expect("forward failed");
    println!(
        "[gemma4-first-decode-logits] forward took {:.2}s, logits.len = {}",
        t0.elapsed().as_secs_f32(),
        logits.len()
    );

    print_ranked_logits(&engine, &logits, top_k, "first decode step", &targets);

    if tokenwise_prompt {
        println!("\n=== tokenwise prompt replay ===");
        let mut tokenwise_engine =
            rnb_llm::Engine::from_gguf(&std::path::PathBuf::from(&model_path))
                .expect("Engine::from_gguf failed for tokenwise replay");
        let mut tokenwise_logits = vec![];
        let t0 = std::time::Instant::now();
        for &token in &tokens {
            tokenwise_logits = tokenwise_engine
                .forward(&[token])
                .expect("tokenwise forward failed");
        }
        println!(
            "[gemma4-first-decode-logits] tokenwise replay took {:.2}s, logits.len = {}",
            t0.elapsed().as_secs_f32(),
            tokenwise_logits.len()
        );
        print_ranked_logits(
            &tokenwise_engine,
            &tokenwise_logits,
            top_k,
            "first decode step (tokenwise prompt)",
            &targets,
        );
    }

    if layerwise {
        println!("\n=== layerwise last-token logits ===");
        let layerwise_logits = engine
            .debug_prefill_layer_logits(&tokens)
            .expect("debug_prefill_layer_logits failed");
        for (layer_idx, layer_logits) in layerwise_logits.iter().enumerate() {
            print_ranked_logits(
                &engine,
                layer_logits,
                top_k,
                &format!("after layer {layer_idx}"),
                &targets,
            );
        }
    }

    if !snapshot_layers.is_empty() {
        println!("\n=== layer snapshots ===");
        let snapshots = engine
            .debug_prefill_layer_snapshots(&tokens)
            .expect("debug_prefill_layer_snapshots failed");
        for snap in snapshots
            .iter()
            .filter(|s| snapshot_layers.contains(&s.layer_idx))
        {
            println!(
                "[snapshot] layer={} cache_layer={} hidden_head={:?} k_head={:?} v_head={:?}",
                snap.layer_idx,
                snap.cache_layer_idx,
                &snap.hidden_last[..snap.hidden_last.len().min(8)],
                &snap.cached_k_last[..snap.cached_k_last.len().min(8)],
                &snap.cached_v_last[..snap.cached_v_last.len().min(8)],
            );
        }
    }

    println!("\n=== done ===");
}
