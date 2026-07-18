//! ExternalDrafterRuntime state machine — reset / shift_for_accept 동작 검증.
//!
//! 실제 Drafter weight 는 필요 없음 (forward 호출 안 함). state field 만 확인.

use rnb_llm::external_drafter::ExternalDrafterRuntime;

#[test]
fn reset_initializes_position_and_clears_drafts() {
    let mut rt = ExternalDrafterRuntime::new_stub_for_test(2560);
    let hidden = vec![0.1f32; 2560];
    rt.reset(&hidden, 100);
    assert_eq!(rt.position(), 100);
    assert!(rt.accumulated_drafts().is_empty());
    assert_eq!(rt.last_target_hidden(), &hidden[..]);
}

#[test]
fn shift_for_accept_drops_state_past_accept_point() {
    let mut rt = ExternalDrafterRuntime::new_stub_for_test(2560);
    rt.reset(&vec![0.0f32; 2560], 50);
    rt.test_push_draft(11);
    rt.test_push_draft(12);
    rt.test_push_draft(13);
    rt.shift_for_accept(1);
    assert_eq!(rt.accumulated_drafts().len(), 0);
    assert_eq!(rt.position(), 51);
}

#[test]
fn shift_for_accept_zero_advances_position_by_zero() {
    let mut rt = ExternalDrafterRuntime::new_stub_for_test(2560);
    rt.reset(&vec![0.0f32; 2560], 50);
    rt.shift_for_accept(0);
    assert_eq!(rt.position(), 50);
}

#[test]
fn argmax_picks_largest_logit() {
    use rnb_llm::external_drafter::test_argmax;
    assert_eq!(test_argmax(&[0.1, 0.9, 0.3, 0.05]), 1);
    assert_eq!(test_argmax(&[-1.0, -2.0, -0.5]), 2);
}

/// RNB_TARGET_MODEL + RNB_DRAFTER_MODEL 환경변수가 필요한 실기기 통합 테스트.
/// `cargo test -- --ignored attach_succeeds_for_gemma4_with_real_drafter` 로 실행.
#[test]
#[ignore]
fn attach_succeeds_for_gemma4_with_real_drafter() {
    let target = std::env::var("RNB_TARGET_MODEL").expect("RNB_TARGET_MODEL");
    let drafter_path = std::env::var("RNB_DRAFTER_MODEL").expect("RNB_DRAFTER_MODEL");

    let mut engine =
        rnb_llm::Engine::from_gguf(std::path::Path::new(&target)).expect("load target engine");

    let drafter =
        rnb_mtp::drafter::load_drafter(std::path::Path::new(&drafter_path)).expect("load drafter");
    engine
        .attach_external_drafter(drafter)
        .expect("attach drafter");
    assert!(
        engine.mtp_runtime_ready(),
        "mtp runtime should report ready after attach"
    );
}
