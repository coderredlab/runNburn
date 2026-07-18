// Dump all Gemma4 metadata keys — used to verify what's actually in the GGUF.
fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());

    let mmap = rnb_core::memory::mmap::MmapLoader::load(std::path::Path::new(&model_path)).unwrap();
    let gguf = rnb_loader::gguf::parser::GGUFFile::parse(&mmap[..]).unwrap();

    for (k, v) in &gguf.metadata {
        if k.starts_with("gemma") || k.starts_with("tokenizer.ggml.") {
            let summary = format!("{v:?}");
            let short = if summary.len() > 300 {
                format!(
                    "{}... [truncated, {} chars total]",
                    &summary[..300],
                    summary.len()
                )
            } else {
                summary
            };
            println!("{k}: {short}");
        }
    }
}
