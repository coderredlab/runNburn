//! mt83 Stage B acceptance test.
//!
//! `Engine::kv_view()`, `Engine::last_layer_hidden()`, `Engine::token_embd_row()`
//! 가 spec 시그니처대로 동작하는지 실제 GGUF fixture 로 검증한다.
//!
//! Fixture 가 없으면 (CI / local 환경) skip. controller 는 명시적으로 환경 변수
//! 를 세팅한 뒤 실행한다:
//!
//! ```bash
//! RNB_TARGET_FIXTURE=/path/to/gemma-4-e4b-it-q4_k_m.gguf \
//!   cargo test -p rnb-llm --test kv_borrow_test
//! ```

use std::path::PathBuf;

use rnb_llm::Engine;

const FIXTURE_ENV: &str = "RNB_TARGET_FIXTURE";

fn load_fixture() -> Option<PathBuf> {
    let raw = std::env::var(FIXTURE_ENV).ok()?;
    let path = PathBuf::from(&raw);
    if !path.exists() {
        panic!("${FIXTURE_ENV} points to missing file: {path:?}");
    }
    Some(path)
}

#[test]
fn kv_view_after_prefill_matches_token_count() {
    let Some(path) = load_fixture() else {
        eprintln!("skip kv_view_after_prefill_matches_token_count: ${FIXTURE_ENV} not set");
        return;
    };

    let mut engine = Engine::from_gguf(&path).expect("load engine from fixture");

    // 짧은 prompt — token 수만 확보되면 OK.
    let prompt = "hello world test prompt input here";
    let tokens = engine.tokenizer.encode(prompt);
    assert!(
        !tokens.is_empty(),
        "tokenizer must produce at least one token"
    );
    let prefill_len = tokens.len();
    engine.forward(&tokens).expect("forward prefill");

    let kv = engine.kv_view();
    assert_eq!(
        kv.pos(),
        prefill_len,
        "pos should equal prefilled token count"
    );
    assert!(kv.n_layers() >= 1, "n_layers must be >= 1");

    // K, V slice length sanity for layer 0
    let k0 = kv.k_layer(0);
    let v0 = kv.v_layer(0);
    let kv_dim0 = kv.kv_dim_for_layer(0);
    assert!(kv_dim0 > 0, "kv_dim_for_layer must be positive");
    assert_eq!(
        k0.len(),
        kv.pos() * kv_dim0,
        "k_layer length mismatch (expected pos * kv_dim)"
    );
    assert_eq!(
        v0.len(),
        kv.pos() * kv_dim0,
        "v_layer length mismatch (expected pos * kv_dim)"
    );
    // Sanity: dequant 값이 NaN/Inf 가 아니어야 한다.
    assert!(
        k0.iter().all(|v| v.is_finite()),
        "K cache must be finite after prefill"
    );
    assert!(
        v0.iter().all(|v| v.is_finite()),
        "V cache must be finite after prefill"
    );
}

#[test]
fn last_layer_hidden_dim_matches_metadata() {
    let Some(path) = load_fixture() else {
        eprintln!("skip last_layer_hidden_dim_matches_metadata: ${FIXTURE_ENV} not set");
        return;
    };
    let mut engine = Engine::from_gguf(&path).expect("load engine from fixture");
    let tokens = engine.tokenizer.encode("test");
    assert!(!tokens.is_empty());
    engine.forward(&tokens).expect("forward prefill");

    let hidden = engine.last_layer_hidden();
    assert!(
        !hidden.is_empty(),
        "last_layer_hidden must be populated after forward"
    );
    let hidden_size = engine.metadata.hidden_dim;
    assert_eq!(
        hidden.len(),
        hidden_size,
        "last_layer_hidden length must equal metadata.hidden_dim"
    );
    assert!(
        hidden.iter().all(|v| v.is_finite()),
        "last_layer_hidden must be finite"
    );
}

#[test]
fn token_embd_row_dim_matches_metadata() {
    let Some(path) = load_fixture() else {
        eprintln!("skip token_embd_row_dim_matches_metadata: ${FIXTURE_ENV} not set");
        return;
    };
    let engine = Engine::from_gguf(&path).expect("load engine from fixture");
    let row = engine.token_embd_row(0);
    let hidden_size = engine.metadata.hidden_dim;
    assert_eq!(
        row.len(),
        hidden_size,
        "token_embd_row length must equal hidden_dim"
    );
    assert!(
        row.iter().all(|v| v.is_finite()),
        "token_embd_row must be finite"
    );
}
