//! Drafter ↔ target KV cache 공유 infrastructure.
//!
//! mt1 skeleton — 본격 구현은 mt2+.
//!
//! # 설계 메모
//!
//! Google 의 MTP 발표에서 핵심:
//! > "draft models seamlessly utilize the target model's activations and share
//! >  its KV cache, meaning they don't have to waste time recalculating context
//! >  the larger model has already figured out"
//!
//! 즉 drafter 가 target 의 KV cache 일부를 read-only 로 access. drafter forward
//! 에서 K/V projection 재계산 X. 단:
//! - drafter 의 hidden_dim / head 수가 target 과 같아야 (Google assistant 모델
//!   spec 확인 필요)
//! - drafter 의 layer 수는 적을 수 있음 (target 의 일부 layer 만 KV share)
//!
//! 향후 작업 (mt2+):
//! - target KVCache 를 read-only borrow 하는 API (rnb-llm 의 KVCache 와 통합)
//! - drafter 의 partial layer mapping (target layer 0-3 share, drafter 은 5
//!   layer 만)
//! - rollback 처리 (verify 에서 mismatch 시 drafter 가 cache 했던 state 폐기)
