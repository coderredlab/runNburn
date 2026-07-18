//! End-to-end smoke for external drafter MTP path (mc78 Task 12).
//!
//! 실제 GGUF 파일이 필요하므로 `#[ignore]` 로 CI 에서 제외한다.
//! 실행 방법:
//!   RNB_TARGET_MODEL=<gemma4.gguf> [RNB_DRAFTER_MODEL=<assistant.gguf>] \
//!     cargo test -p rnb-llm --test external_drafter_smoke_test -- --ignored
//!
//! RNB_DRAFTER_MODEL 생략 시 auto_drafter::find_sibling_drafter 가 sibling 탐색.
//! RNB_MTP=1 이 set 돼야 generate_stream_mtp 경로로 진입한다.

#[test]
#[ignore]
fn mtp_external_emits_nontrivial_tokens() {
    let target_path = std::env::var("RNB_TARGET_MODEL")
        .expect("RNB_TARGET_MODEL env var must be set for this test (Gemma 4 target GGUF)");

    // RNB_MTP=1 을 여기서 강제해 generate_stream_mtp 경로를 탄다.
    unsafe {
        std::env::set_var("RNB_MTP", "1");
    }

    let path = std::path::Path::new(&target_path);
    let mut engine =
        rnb_llm::Engine::from_gguf(path).expect("Failed to load target engine from GGUF");

    assert!(
        engine.mtp_runtime_ready(),
        "External drafter must auto-attach at init (check RNB_DRAFTER_MODEL or sibling GGUF)"
    );

    let params = rnb_llm::generate::GenerateParams {
        max_tokens: 20,
        temperature: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        spec_enabled: false,
        spec_k: 4,
        ..rnb_llm::generate::GenerateParams::default()
    };

    let mut out_tokens: Vec<String> = Vec::new();
    let result = rnb_llm::generate::generate_stream(
        &mut engine,
        "The capital of France is",
        &params,
        |piece| {
            out_tokens.push(piece.to_string());
            true
        },
    )
    .expect("generate_stream failed");

    assert!(
        result.tokens_generated > 0,
        "Expected at least one token, got 0"
    );
    assert!(
        !out_tokens.is_empty(),
        "Callback should have received at least one piece"
    );
    eprintln!(
        "[smoke] tokens_generated={} text={:?}",
        result.tokens_generated,
        out_tokens.join("")
    );
}

/// greedy (temperature=0) 조건에서 MTP ON 경로와 OFF (standard) 경로가
/// 완전히 동일한 토큰 시퀀스를 생성하는지 검증한다.
///
/// verify_greedy 보장: greedy sampling 은 argmax 이므로 RNG 미사용.
/// MTP 는 draft 토큰을 제안하지만 target verify 단계에서 candidate 가 argmax 와
/// 일치할 때만 accept 되므로, 최종 시퀀스는 순수 greedy 와 반드시 동일해야 한다.
///
/// 실행 방법:
///   RNB_TARGET_MODEL=<gemma4.gguf> RNB_DRAFTER_MODEL=<assistant.gguf> \
///     cargo test -p rnb-llm --test external_drafter_smoke_test \
///     -- --ignored mtp_on_off_produce_same_greedy_sequence --nocapture
#[test]
#[ignore]
fn mtp_on_off_produce_same_greedy_sequence() {
    let target = std::env::var("RNB_TARGET_MODEL")
        .expect("RNB_TARGET_MODEL env var must be set (Gemma 4 target GGUF)");
    let drafter = std::env::var("RNB_DRAFTER_MODEL")
        .expect("RNB_DRAFTER_MODEL env var must be set (Gemma 4 assistant GGUF)");

    let params = rnb_llm::generate::GenerateParams {
        max_tokens: 16,
        temperature: 0.0, // greedy — argmax, RNG 미사용
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: Some(0),
        spec_enabled: false,
        spec_k: 4,
        ..rnb_llm::generate::GenerateParams::default()
    };
    let prompt = "The capital of France is";

    // ── OFF run: auto-drafter 비활성화, standard generate_stream 경로 ─────────
    unsafe {
        std::env::set_var("RNB_MTP_DISABLE_AUTO_DRAFTER", "1");
        std::env::remove_var("RNB_MTP");
    }
    let path_off = std::path::Path::new(&target);
    let mut engine_off = rnb_llm::Engine::from_gguf(path_off).expect("load target (OFF)");
    assert!(
        !engine_off.mtp_runtime_ready(),
        "MTP must be disabled for OFF run (check RNB_MTP_DISABLE_AUTO_DRAFTER)"
    );

    let mut pieces_off: Vec<String> = Vec::new();
    let result_off =
        rnb_llm::generate::generate_stream(&mut engine_off, prompt, &params, |piece| {
            pieces_off.push(piece.to_string());
            true
        })
        .expect("generate_stream failed (OFF)");

    eprintln!(
        "[off] tokens_generated={} text={:?}",
        result_off.tokens_generated, result_off.text
    );

    // ── ON run: drafter 연결, MTP generate_stream_mtp 경로 ────────────────────
    unsafe {
        std::env::remove_var("RNB_MTP_DISABLE_AUTO_DRAFTER");
        std::env::set_var("RNB_DRAFTER_MODEL", &drafter);
        std::env::set_var("RNB_MTP", "1");
    }
    let path_on = std::path::Path::new(&target);
    let mut engine_on = rnb_llm::Engine::from_gguf(path_on).expect("load target (ON)");
    assert!(
        engine_on.mtp_runtime_ready(),
        "MTP must be enabled for ON run (check RNB_DRAFTER_MODEL / auto-attach)"
    );

    let mut pieces_on: Vec<String> = Vec::new();
    let result_on = rnb_llm::generate::generate_stream(&mut engine_on, prompt, &params, |piece| {
        pieces_on.push(piece.to_string());
        true
    })
    .expect("generate_stream failed (ON)");

    eprintln!(
        "[on]  tokens_generated={} text={:?}",
        result_on.tokens_generated, result_on.text
    );

    assert_eq!(
        result_off.text, result_on.text,
        "greedy MTP (ON) must produce identical text to plain greedy (OFF).\n\
         OFF: {:?}\n ON: {:?}",
        result_off.text, result_on.text
    );
}
