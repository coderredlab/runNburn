//! Full-device decode must preserve KV state when a caller switches back to multi-token prefill.
//!
//! This needs the product-size Muse GGUF and a CUDA device large enough to admit all layers, so it
//! is fixture-gated and ignored by default:
//! `RNB_TARGET_MODEL=/path/to/muse.gguf cargo test -p rnb-llm --features cuda --test muse_full_device_continuation_test -- --ignored --nocapture`.

#[cfg(feature = "cuda")]
use rnb_llm::Engine;

#[cfg(feature = "cuda")]
fn next_token(engine: &Engine, logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return engine
            .last_backend_argmax_token()
            .expect("backend argmax token");
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .expect("non-empty logits")
}

#[cfg(feature = "cuda")]
fn run_continuation(model: &std::path::Path, carrier: bool) -> Vec<u32> {
    unsafe {
        std::env::set_var("RNB_MTP", "0");
        std::env::set_var("RNB_KV_CACHE_FORMAT", "f16");
        if carrier {
            std::env::remove_var("RNB_CUDA_DECODE_DEVICE_CHAIN");
        } else {
            std::env::set_var("RNB_CUDA_DECODE_DEVICE_CHAIN", "0");
        }
    }
    let mut engine = Engine::from_gguf(model).expect("load Muse engine");
    let mut prompt = Vec::new();
    if engine.tokenizer.should_add_bos() {
        prompt.push(engine.tokenizer.vocab.special.bos);
    }
    prompt.extend(
        engine.tokenizer.encode(
            "Jane Austen's novels are often described as comedies of manners. Explain why.",
        ),
    );

    let logits = engine.forward(&prompt).expect("prompt prefill");
    let first = next_token(&engine, &logits);
    let logits = engine.forward(&[first]).expect("single-token decode");
    let second = next_token(&engine, &logits);

    let mut logits = engine
        .forward(&[second, second])
        .expect("multi-token continuation prefill");
    let mut generated = Vec::new();
    for _ in 0..8 {
        let token = next_token(&engine, &logits);
        generated.push(token);
        logits = engine.forward(&[token]).expect("continued decode");
    }
    let verify_token = next_token(&engine, &logits);
    let mut verify_rows = engine
        .forward_prefill_all_logits(&[verify_token, verify_token])
        .expect("all-logits continuation prefill");
    let mut logits = verify_rows.pop().expect("all-logits final row");
    for _ in 0..4 {
        let token = next_token(&engine, &logits);
        generated.push(token);
        logits = engine.forward(&[token]).expect("post-verify decode");
    }
    if carrier {
        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_DEVICE_CHAIN", "0");
        }
    }
    let fallback_token = next_token(&engine, &logits);
    logits = engine
        .forward(&[fallback_token])
        .expect("device-to-eager decode fallback");
    for _ in 0..4 {
        let token = next_token(&engine, &logits);
        generated.push(token);
        logits = engine.forward(&[token]).expect("eager fallback decode");
    }
    generated
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn muse_full_device_decode_matches_eager_after_multitoken_continuation() {
    let model = std::env::var("RNB_TARGET_MODEL").expect("set RNB_TARGET_MODEL to the Muse GGUF");
    let model = std::path::Path::new(&model);
    let eager = run_continuation(model, false);
    let carrier = run_continuation(model, true);
    assert_eq!(carrier, eager);
}
