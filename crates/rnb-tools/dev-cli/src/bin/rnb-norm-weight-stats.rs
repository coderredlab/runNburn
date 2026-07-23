use rnb_core::tensor::DType;

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-e4b-it-Q4_K_M.gguf".to_string());
    let weights = std::env::var("RNB_WEIGHTS").unwrap_or_else(|_| {
        "output_norm.weight,blk.0.attn_norm.weight,blk.0.attn_q_norm.weight,blk.0.attn_k_norm.weight,blk.0.post_attention_norm.weight,blk.0.ffn_norm.weight,blk.0.post_ffw_norm.weight,per_layer_proj_norm.weight,blk.0.post_norm.weight".to_string()
    });

    let model = rnb_loader::load_model(&std::path::PathBuf::from(&model_path)).unwrap();

    for name in weights.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        println!("=== {name} ===");
        let Some(tensor) = model.weights.get(name) else {
            println!("missing");
            continue;
        };

        match tensor.dtype() {
            DType::F32 => {
                if let Some(bytes) = tensor.as_bytes() {
                    let data: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mean = data.iter().copied().sum::<f32>() / data.len() as f32;
                    let first: Vec<String> =
                        data.iter().take(8).map(|v| format!("{v:.6}")).collect();
                    println!(
                        "len={} min={:.6} max={:.6} mean={:.6} first8=[{}]",
                        data.len(),
                        min,
                        max,
                        mean,
                        first.join(", ")
                    );
                } else {
                    println!("f32 bytes unavailable");
                }
            }
            other => {
                println!("dtype={other:?} shape={:?}", tensor.shape());
            }
        }
    }
}
