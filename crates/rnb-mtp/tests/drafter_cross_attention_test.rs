//! mt83 Stage C acceptance test — drafter cross-attention forward.
//!
//! **mt84 Stage γ 에서 폐기**: mt83 의 `drafter_cross_attention_forward`
//! API 가 제거되고 transformers source verbatim 의 `backbone::drafter_forward`
//! 로 교체됨 (spec
//! `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §1-§5). 본 test 의 가설 (`KvBorrow` + `kv_share_map` 기반 cross-attention)
//! 자체가 `SharedKvStates` dict 기반 backbone reuse 와 호환 안 됨.
//!
//! Stage δ 의 새 `drafter_backbone_calibrate_test` 가 본 test 의 acceptance
//! 역할을 대신한다. 본 파일은 기록 목적의 빈 placeholder 로 보존.

#[test]
#[ignore = "mt84 Stage γ: replaced by drafter_backbone_test / drafter_backbone_calibrate_test"]
fn cross_attention_produces_finite_cluster_logits() {
    // mt83 자산 — Stage δ 의 새 calibrate test 가 본 acceptance 를 대체.
}
