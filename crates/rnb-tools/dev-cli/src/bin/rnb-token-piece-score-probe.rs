fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string());
    let query = std::env::var("RNB_QUERY")
        .unwrap_or_else(|_| "대한,민국,의,국의,수도, 수도,는,도는,민, 수".to_string());

    let mmap = rnb_core::memory::mmap::MmapLoader::load(std::path::Path::new(&model_path)).unwrap();
    let gguf = rnb_loader::gguf::parser::GGUFFile::parse(&mmap[..]).unwrap();

    let tokens =
        rnb_loader::gguf::metadata::get_string_array(&gguf.metadata, "tokenizer.ggml.tokens")
            .expect("missing tokenizer.ggml.tokens");
    let scores = rnb_loader::gguf::metadata::get_f32_array(&gguf.metadata, "tokenizer.ggml.scores")
        .expect("missing tokenizer.ggml.scores");

    let vocab = rnb_llm::tokenizer::vocab::Vocab::new(
        tokens.clone(),
        rnb_llm::tokenizer::vocab::SpecialTokens {
            bos: rnb_loader::gguf::metadata::get_u32(&gguf.metadata, "tokenizer.ggml.bos_token_id")
                .unwrap_or(1),
            eos: rnb_loader::gguf::metadata::get_u32(&gguf.metadata, "tokenizer.ggml.eos_token_id")
                .unwrap_or(2),
            pad: rnb_loader::gguf::metadata::get_u32_opt(
                &gguf.metadata,
                "tokenizer.ggml.padding_token_id",
            ),
        },
    );

    for piece in query.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match vocab.token_id(piece) {
            Some(id) => {
                let score = scores.get(id as usize).copied().unwrap_or(f32::NAN);
                println!("piece={piece:?} id={id} score={score}");
            }
            None => println!("piece={piece:?} missing"),
        }
    }
}
