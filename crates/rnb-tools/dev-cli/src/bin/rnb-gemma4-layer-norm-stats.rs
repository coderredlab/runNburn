fn parse_layers(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn stats_f32(values: &[f32]) -> String {
    if values.is_empty() {
        return "empty".to_string();
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let abs_mean = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let neg = values.iter().filter(|&&v| v < 0.0).count();
    let near_zero = values.iter().filter(|&&v| v.abs() < 0.05).count();
    let near_one_offset = values.iter().map(|v| 1.0 + *v).collect::<Vec<_>>();
    let off_mean = near_one_offset.iter().sum::<f32>() / near_one_offset.len() as f32;
    let off_min = near_one_offset.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let off_max = near_one_offset
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let head = values.iter().take(8).copied().collect::<Vec<_>>();

    format!(
        "mean={mean:.6} abs_mean={abs_mean:.6} min={min:.6} max={max:.6} neg={neg}/{} near_zero={near_zero}/{} one_plus(mean={off_mean:.6} min={off_min:.6} max={off_max:.6}) head={head:?}",
        values.len(),
        values.len()
    )
}

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let layers = std::env::var("RNB_GEMMA4_LAYER_NORM_LAYERS")
        .map(|raw| parse_layers(&raw))
        .unwrap_or_else(|_| vec![11, 12, 13, 14]);

    let loaded = rnb_loader::load_model(std::path::Path::new(&model_path)).unwrap();
    let tensor_suffixes = [
        ("attn_norm", "attn_norm.weight"),
        ("attn_q_norm", "attn_q_norm.weight"),
        ("attn_k_norm", "attn_k_norm.weight"),
        ("post_attention_norm", "post_attention_norm.weight"),
        ("ffn_norm", "ffn_norm.weight"),
        ("ffn_post_norm", "ffn_post_norm.weight"),
        ("ple_post_norm", "post_norm.weight"),
        ("out_scale", "layer_output_scale.weight"),
    ];

    println!("[gemma4-layer-norm-stats] model_path = {model_path}");
    println!("[gemma4-layer-norm-stats] layers     = {:?}", layers);

    for layer_idx in layers {
        println!("\n=== layer {layer_idx} ===");
        for (label, suffix) in tensor_suffixes {
            let name = format!("blk.{layer_idx}.{suffix}");
            let Some(tensor) = loaded.weights.get(&name) else {
                println!("{label:>15}: missing");
                continue;
            };

            let ggml = loaded.tensor_ggml_types.get(&name).copied();
            let stored_shape = tensor.shape().to_vec();
            let float_shape = loaded.float_shapes.get(&name).cloned();

            print!(
                "{label:>15}: ggml={ggml:?} stored={stored_shape:?} float_shape={float_shape:?}"
            );

            match ggml {
                Some(rnb_loader::GGMLType::F32) => {
                    if let Some(bytes) = tensor.as_bytes() {
                        let vals = bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect::<Vec<_>>();
                        print!(" {}", stats_f32(&vals));
                    }
                }
                _ => {
                    print!(" non-f32");
                }
            }

            println!();
        }
    }
}
