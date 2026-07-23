fn main() {
    let model_path =
        std::env::var("RNB_MODEL").unwrap_or_else(|_| "models/gemma-4-E2B.Q4_K_M.gguf".to_string());

    let mmap = rnb_core::memory::mmap::MmapLoader::load(std::path::Path::new(&model_path)).unwrap();
    let gguf = rnb_loader::gguf::parser::GGUFFile::parse(&mmap[..]).unwrap();

    let keys = [
        "general.architecture",
        "gemma4.feed_forward_length",
        "gemma4.embedding_length",
        "gemma4.block_count",
        "gemma4.attention.head_count",
        "gemma4.attention.head_count_kv",
        "gemma4.embedding_length_per_layer_input",
        "gemma4.attention.shared_kv_layers",
        "tokenizer.ggml.model",
        "tokenizer.chat_template",
    ];

    for key in keys {
        print!("{key}: ");
        match gguf.metadata.iter().find(|(k, _)| k == key) {
            Some((_, v)) => println!("{v:?}"),
            None => println!("missing"),
        }
    }
}
