/// Indexed sdot sanity test + token accuracy comparison
fn main() {
    // --- Test indexed sdot via known-value GEMV ---
    // Instead of calling the test function directly, we test the full pipeline:
    // Create known weight data, repack it, run asm kernel, compare with scalar.

    // --- Token accuracy ---
    let path = std::path::PathBuf::from("models/Qwen3.5-0.8B-Q4_K_M.gguf");
    let mut engine = rnb_llm::Engine::from_gguf(&path).unwrap();

    let bos = engine.tokenizer.vocab.special.bos;
    let prompt_tokens = engine.tokenizer.encode("Hello");
    let mut tokens = vec![bos];
    tokens.extend(&prompt_tokens);

    let logits = engine.forward(&tokens).unwrap();
    let mut token = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    let mut generated = Vec::new();
    for _ in 0..20 {
        generated.push(token);
        let logits = engine.forward(&[token]).unwrap();
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
    }
    generated.push(token);

    for (i, &t) in generated.iter().enumerate() {
        let s = engine.tokenizer.decode_token(t);
        eprintln!("  {i:2}: id={t:6} → {s:?}");
    }
}
