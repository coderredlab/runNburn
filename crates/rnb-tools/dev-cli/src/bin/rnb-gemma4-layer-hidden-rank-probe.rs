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

fn dk<T>(
    bytes: &[u8],
    block_bytes: usize,
    elems_per_block: usize,
    f: fn(&T, &mut [f32; 256]),
) -> Vec<f32> {
    let nb = bytes.len() / block_bytes;
    let mut out = vec![0.0f32; nb * elems_per_block];
    for (bi, chunk) in bytes.chunks_exact(block_bytes).enumerate() {
        let block = unsafe { &*(chunk.as_ptr() as *const T) };
        let mut tmp = [0.0f32; 256];
        f(block, &mut tmp);
        out[bi * elems_per_block..(bi + 1) * elems_per_block].copy_from_slice(&tmp);
    }
    out
}

fn dequantize_for_debug(bytes: &[u8], ggml_type: rnb_loader::GGMLType) -> Vec<f32> {
    use rnb_loader::GGMLType;
    match ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        GGMLType::Q8_0 => {
            let nb = bytes.len() / 34;
            let mut out = vec![0.0f32; nb * 32];
            for (bi, chunk) in bytes.chunks_exact(34).enumerate() {
                let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                for i in 0..32 {
                    out[bi * 32 + i] = chunk[2 + i] as i8 as f32 * d;
                }
            }
            out
        }
        GGMLType::Q4_K => {
            dk::<rnb_cpu::quantize::BlockQ4_K>(bytes, 144, 256, rnb_cpu::quantize::dequantize_q4_k)
        }
        GGMLType::Q5_K => {
            dk::<rnb_cpu::quantize::BlockQ5_K>(bytes, 176, 256, rnb_cpu::quantize::dequantize_q5_k)
        }
        GGMLType::Q6_K => {
            dk::<rnb_cpu::quantize::BlockQ6_K>(bytes, 210, 256, rnb_cpu::quantize::dequantize_q6_k)
        }
        _ => vec![],
    }
}

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let prompt = std::env::var("RNB_PROMPT").unwrap_or_else(|_| "대한민국의 수도는".to_string());
    let no_bos = std::env::var("RNB_NO_BOS").is_ok();
    let layerwise = std::env::var("RNB_GEMMA4_HIDDEN_RANK_LAYERWISE")
        .ok()
        .map(|raw| parse_layers(&raw))
        .unwrap_or_default();
    let decode_token = std::env::var("RNB_GEMMA4_HIDDEN_RANK_DECODE_TOKEN")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    let targets = parse_targets(
        &std::env::var("RNB_GEMMA4_HIDDEN_RANK_TARGETS")
            .unwrap_or_else(|_| "서울; 서울;fem;Fem;Tat;garçon;입니다; 입니다".to_string()),
    );

    let mut engine = rnb_llm::Engine::from_gguf(std::path::Path::new(&model_path))
        .expect("Engine::from_gguf failed");
    let loaded =
        rnb_loader::load_model(std::path::Path::new(&model_path)).expect("load_model failed");
    let bos_id = engine.tokenizer.vocab.special.bos;
    let mut tokens = Vec::new();
    if !no_bos {
        tokens.push(bos_id);
    }
    tokens.extend(engine.tokenizer.encode(&prompt));
    let hidden = if let Some(token) = decode_token {
        let _ = engine.forward(&tokens).expect("prefill failed");
        engine
            .debug_decode_next_hidden_normed(token)
            .expect("debug_decode_next_hidden_normed failed")
    } else {
        engine
            .debug_prefill_last_hidden_normed(&tokens)
            .expect("debug_prefill_last_hidden_normed failed")
    };
    let hidden_l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    let layer_hiddens = if layerwise.is_empty() {
        Vec::new()
    } else {
        engine
            .debug_prefill_layer_hidden_normed(&tokens)
            .expect("debug_prefill_layer_hidden_normed failed")
    };

    let tensor = loaded
        .weights
        .get("token_embd.weight")
        .expect("token_embd.weight missing");
    let ggml_type = loaded
        .tensor_ggml_types
        .get("token_embd.weight")
        .copied()
        .expect("token_embd ggml type missing");
    let float_shape = loaded
        .float_shapes
        .get("token_embd.weight")
        .expect("token_embd float shape missing");
    let rows = float_shape[0];
    let cols = float_shape[1];
    let bytes = tensor.as_bytes().expect("token_embd bytes missing");
    let bytes_per_row = bytes.len() / rows;

    println!("[gemma4-layer-hidden-rank-probe] model_path = {model_path}");
    println!("[gemma4-layer-hidden-rank-probe] prompt     = {prompt:?}");
    println!("[gemma4-layer-hidden-rank-probe] no_bos     = {no_bos}");
    println!(
        "[gemma4-layer-hidden-rank-probe] layerwise  = {:?}",
        layerwise
    );
    println!(
        "[gemma4-layer-hidden-rank-probe] decode_tok = {:?}",
        decode_token
    );
    println!("[gemma4-layer-hidden-rank-probe] token_embd type={ggml_type:?} rows={rows} cols={cols} bytes_per_row={bytes_per_row}");
    println!("[gemma4-layer-hidden-rank-probe] hidden_l2  = {hidden_l2:.6}");
    println!("[gemma4-layer-hidden-rank-probe] targets = {:?}", targets);

    for target in &targets {
        let mut found = false;
        for token_id in 0..engine.metadata.vocab_size {
            let piece = engine.tokenizer.decode_token(token_id as u32);
            if piece == *target {
                found = true;
                let row = &bytes[token_id * bytes_per_row..(token_id + 1) * bytes_per_row];
                let vals = dequantize_for_debug(row, ggml_type);
                let l2 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let mean_abs = vals.iter().map(|v| v.abs()).sum::<f32>() / vals.len() as f32;
                let dot = vals
                    .iter()
                    .zip(hidden.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                let cosine = if l2 > 0.0 && hidden_l2 > 0.0 {
                    dot / (l2 * hidden_l2)
                } else {
                    0.0
                };
                let head = vals.iter().take(8).copied().collect::<Vec<_>>();
                println!(
                    "{:?}: id={} dot={:.4} cosine={:.6} l2={:.4} max_abs={:.4} mean_abs={:.4} head={:?}",
                    target, token_id, dot, cosine, l2, max_abs, mean_abs, head
                );
                break;
            }
        }
        if !found {
            println!("{:?}: not found", target);
        }
    }

    if !layerwise.is_empty() {
        println!("\n=== layerwise ===");
        for layer_idx in layerwise {
            let Some(hidden) = layer_hiddens.get(layer_idx) else {
                continue;
            };
            let hidden_l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!("\n--- after layer {layer_idx} hidden_l2={hidden_l2:.6} ---");
            for target in &targets {
                let mut found = false;
                for token_id in 0..engine.metadata.vocab_size {
                    let piece = engine.tokenizer.decode_token(token_id as u32);
                    if piece == *target {
                        found = true;
                        let row = &bytes[token_id * bytes_per_row..(token_id + 1) * bytes_per_row];
                        let vals = dequantize_for_debug(row, ggml_type);
                        let l2 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let dot = vals
                            .iter()
                            .zip(hidden.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>();
                        let cosine = if l2 > 0.0 && hidden_l2 > 0.0 {
                            dot / (l2 * hidden_l2)
                        } else {
                            0.0
                        };
                        println!(
                            "{:?}: id={} dot={:.4} cosine={:.6}",
                            target, token_id, dot, cosine
                        );
                        break;
                    }
                }
                if !found {
                    println!("{:?}: not found", target);
                }
            }
        }
    }
}
