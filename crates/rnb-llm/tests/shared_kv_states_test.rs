//! mt84 Stage β.4 — `Engine::shared_kv_states_for_drafter()` 의 layout 검증.
//!
//! Gemma 4 E4B target GGUF 가 있을 때만 동작 (fixture-gated). default `cargo test`
//! 실행에서는 `#[ignore]` 로 skip 되며, 실기기 / fixture 가 있는 환경에서만 다음
//! 명령으로 명시 실행:
//!
//! ```bash
//! RNB_TARGET_MODEL=/path/to/gemma-4-e4b-it-q4_k_m.gguf \
//!   cargo test -p rnb-llm --test shared_kv_states_test -- --ignored --nocapture
//! ```
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md` §6.

use std::path::PathBuf;

use rnb_llm::Engine;

#[test]
#[ignore]
fn shared_kv_states_layout() {
    let path = std::env::var("RNB_TARGET_MODEL")
        .expect("set RNB_TARGET_MODEL=path/to/gemma-4-e4b-it-q4_k_m.gguf");
    let path = PathBuf::from(path);
    let mut engine = Engine::from_gguf(&path).expect("load target engine from RNB_TARGET_MODEL");

    // Synthetic 5-token prefill — token id 자체는 vocab 안에 들기만 하면 됨.
    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    engine
        .forward(&prompt_tokens)
        .expect("forward prefill on 5-token prompt");

    let states = engine.shared_kv_states_for_drafter();

    // Gemma4 E4B target 기준 shape:
    // - num_kv_heads = 2
    // - sliding layer head_dim = 256 (key_length_swa)
    // - full layer head_dim = 512 (key_length_full)
    assert_eq!(
        states.sliding_attention.head_dim, 256,
        "sliding head_dim should be 256 for Gemma4 E4B"
    );
    assert_eq!(
        states.full_attention.head_dim, 512,
        "full head_dim should be 512 for Gemma4 E4B"
    );
    assert_eq!(states.sliding_attention.n_kv_heads, 2);
    assert_eq!(states.full_attention.n_kv_heads, 2);

    assert_eq!(states.sliding_attention.seq_len, 5);
    assert_eq!(states.full_attention.seq_len, 5);

    // Flat length sanity: n_kv_heads * seq_len * head_dim.
    assert_eq!(
        states.sliding_attention.k.len(),
        2 * 5 * 256,
        "sliding K length mismatch"
    );
    assert_eq!(
        states.sliding_attention.v.len(),
        2 * 5 * 256,
        "sliding V length mismatch"
    );
    assert_eq!(
        states.full_attention.k.len(),
        2 * 5 * 512,
        "full K length mismatch"
    );
    assert_eq!(
        states.full_attention.v.len(),
        2 * 5 * 512,
        "full V length mismatch"
    );

    // F16 → F32 dequant 결과가 NaN/Inf 가 아니고, all-zero 도 아니어야 한다.
    assert!(
        states.sliding_attention.k.iter().all(|x| x.is_finite()),
        "sliding K must be finite"
    );
    assert!(
        states.sliding_attention.v.iter().all(|x| x.is_finite()),
        "sliding V must be finite"
    );
    assert!(
        states.full_attention.k.iter().all(|x| x.is_finite()),
        "full K must be finite"
    );
    assert!(
        states.full_attention.v.iter().all(|x| x.is_finite()),
        "full V must be finite"
    );
    assert!(
        states.sliding_attention.k.iter().any(|x| x.abs() > 1e-6),
        "sliding K must contain non-trivial values after prefill"
    );
}
