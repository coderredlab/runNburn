//! mt83 Stage D acceptance test — superseded by mt84 Stage δ.
//!
//! mt83 의 9-variant grid (`KvShareMap` × `ClusterTokenStrategy`) 은
//! `2026-05-14-gemma4-assistant-backbone-reuse-design.md` (mt84) 에서 single
//! variant 로 축소됐다. drafter forward 도 transformers source verbatim 의
//! `backbone::drafter_forward` (SharedKvStates dict 기반) 로 재작성됨.
//!
//! 본 placeholder 는 mt83 historical reference. 실제 calibration acceptance
//! 는 `tests/drafter_backbone_calibrate_test.rs` (mt84 Stage δ) 가 담당.

#[test]
#[ignore = "mt84 Stage δ: superseded by drafter_backbone_calibrate_test"]
fn calibrate_drafter_top1_threshold() {
    // mt83 자산 — 본 placeholder 는 historical only.
}
