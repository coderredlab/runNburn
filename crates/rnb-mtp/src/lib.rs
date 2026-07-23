//! Multi-Token Prediction (MTP) helper for `rnb-llm` generation loop.
//!
//! mt1 (2026-05-07) — skeleton. Google 의 Gemma 4 MTP drafters
//! (<https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>,
//! 2026-05-06) 을 runNburn 에 적용하기 위한 helper crate.
//!
//! # 역할 분담
//!
//! - `rnb-llm` 이 generation control flow / sampler / KV cache 소유 (spec 정책)
//! - `rnb-mtp` 는 MTP 의 specific helper:
//!   - drafter weight loader
//!   - parallel verify (target.forward 의 N+1 token vs drafter 의 N token 비교)
//!   - drafter ↔ target KV cache 공유 infrastructure
//!
//! # Dependency
//!
//! `rnb-llm → rnb-mtp` (helper). rnb-mtp 는 backend / runtime / scheduler 의존 X.
//! 향후 다른 speculative decoding 기법 (Medusa, EAGLE, Lookahead 등) 도 같은
//! crate 에 추가 가능.
//!
//! # MTP 작동 원리 (Google 발표 요약)
//!
//! 1. drafter (small assistant model) 가 N future token 빠르게 제안
//! 2. target (main model) 이 drafter 의 N token + 자기 1 token 을 single forward
//!    pass 에 parallel verify
//! 3. 일치 prefix 까지 accept + target 의 1 token 추가 → N+1 token emit
//! 4. drafter 가 target 의 activation / KV cache share → 재계산 0
//!
//! 결과: decode 최대 3x speedup, output identical (target verification 보장).

pub mod drafter;
pub mod kv_share;
pub mod verify;

pub use drafter::{SharedKvLayer, SharedKvStates};

/// MTP helper 의 공통 error.
#[derive(Debug, thiserror::Error)]
pub enum MtpError {
    #[error("drafter weight load failed: {0}")]
    DrafterLoad(String),
    #[error("verify shape mismatch: {0}")]
    VerifyShape(String),
    #[error("kv cache share misconfigured: {0}")]
    KvShare(String),
}

pub type MtpResult<T> = Result<T, MtpError>;
