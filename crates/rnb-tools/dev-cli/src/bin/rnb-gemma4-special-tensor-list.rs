fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let loaded = rnb_loader::load_model(std::path::Path::new(&model_path)).unwrap();

    for (name, tensor) in &loaded.weights {
        let interesting = name == "per_layer_token_embd.weight"
            || name == "per_layer_model_proj.weight"
            || name == "per_layer_proj_norm.weight"
            || name.ends_with(".inp_gate.weight")
            || name.ends_with(".proj.weight")
            || name.ends_with(".post_norm.weight");
        if interesting {
            println!(
                "{} ggml={:?} dtype={:?} stored={:?} float_shape={:?}",
                name,
                loaded.tensor_ggml_types.get(name),
                tensor.dtype(),
                tensor.shape(),
                loaded.float_shapes.get(name)
            );
        }
    }
}
