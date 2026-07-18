fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let loaded = rnb_loader::load_model(std::path::Path::new(&model_path)).unwrap();

    let mut names = loaded.weights.keys().cloned().collect::<Vec<_>>();
    names.sort();

    for name in names {
        let interesting = name.starts_with("per_layer_")
            || name.contains(".inp_gate.weight")
            || name.contains(".proj.weight")
            || name.contains(".post_norm.weight");
        if !interesting {
            continue;
        }
        let tensor = loaded.weights.get(&name).unwrap();
        let ggml = loaded.tensor_ggml_types.get(&name).copied();
        let float_shape = loaded.float_shapes.get(&name).cloned();
        print!(
            "{} ggml={:?} stored={:?} float_shape={:?}",
            name,
            ggml,
            tensor.shape(),
            float_shape
        );

        if matches!(ggml, Some(rnb_loader::GGMLType::F32)) {
            if let Some(bytes) = tensor.as_bytes() {
                let vals = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>();
                if !vals.is_empty() {
                    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                    let min = vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let head = vals.iter().take(8).copied().collect::<Vec<_>>();
                    print!(
                        " mean={:.6} min={:.6} max={:.6} head={:?}",
                        mean, min, max, head
                    );
                }
            }
        }

        println!();
    }
}
