//! mt84 Stage δ — drafter backbone forward 의 single-variant calibration.
//!
//! mt83 의 9-variant grid (`KvShareMap` × `ClusterTokenStrategy`) 폐기.
//! mt84 의 architecture 는 spec §Calibration 재설계 에서 결정:
//!
//! - **KvShareMap 폐기** — drafter 의 각 layer 가 `layer_type` (sliding / full)
//!   에 따라 자동으로 target 의 `store_full_length_kv` layer 의 K/V 를 빌려옴.
//!   매핑이 자동이라 enum dispatch 불필요. [`super::backbone::drafter_forward`]
//!   가 `SharedKvStates` 의 `sliding_attention` / `full_attention` 중 하나를
//!   layer 마다 선택하는 방식.
//!
//! - **ClusterTokenStrategy 폐기** — token_ordering permutation 이 GGUF 의
//!   `mtp.token_ordering.weight` 에서 단일 source 로 들어옴.
//!   [`super::vq_head::ClusterTokenTable::Permutation`] 만 사용.
//!
//! Calibration test (`drafter_backbone_calibrate_test.rs`) 가 외부 helper 없이
//! `backbone::drafter_forward` 와 `Engine` 의 public API 만으로 single variant
//! top-1 ≥ 0.40 을 검증한다. 본 모듈은 Stage δ 진행 중 단계별로 helper 가
//! 필요해지면 (e.g. diagnostic δ.a-d 의 공유 fixture) 다시 채워질 placeholder.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §Calibration 재설계 + §Acceptance Criteria Stage 2.
