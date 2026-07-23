//! Parallel verify — target.forward 의 N+1 token vs drafter 의 N token 비교.
//!
//! mt1 skeleton — 본격 구현은 mt2+.
//!
//! # 설계 메모
//!
//! - target 이 drafter 의 N token sequence + 자기 prediction 1 token 을 single
//!   forward pass 에 처리 (input = drafter 가 제안한 token sequence)
//! - 각 position 의 target prediction 과 drafter 의 다음 token 비교
//! - 첫 mismatch 까지 accept + target 의 1 token 추가
//! - sampler 의 stochastic 경로 (top-p, top-k 등) 지원: stochastic 일치 보장
//!   알고리즘 (rejection sampling) 적용
//!
//! 향후 작업 (mt2+):
//! - greedy mode (temperature=0) 의 deterministic verify
//! - stochastic mode 의 rejection sampling
//! - rnb-llm 의 sampler API 와 통합

use crate::{MtpError, MtpResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreedyVerifyOutcome {
    pub accepted_draft_tokens: usize,
    pub target_token: u32,
}

pub fn verify_greedy(
    draft_tokens: &[u32],
    target_predictions: &[u32],
) -> MtpResult<GreedyVerifyOutcome> {
    if target_predictions.len() != draft_tokens.len() + 1 {
        return Err(MtpError::VerifyShape(format!(
            "target predictions length {} must equal draft length {} + 1",
            target_predictions.len(),
            draft_tokens.len()
        )));
    }

    let accepted_draft_tokens = draft_tokens
        .iter()
        .zip(target_predictions.iter())
        .take_while(|(draft, target)| draft == target)
        .count();

    Ok(GreedyVerifyOutcome {
        accepted_draft_tokens,
        target_token: target_predictions[accepted_draft_tokens],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_verify_accepts_matching_draft_prefix_and_returns_next_target_token() {
        let outcome = verify_greedy(&[10, 11, 12], &[10, 11, 12, 13]).unwrap();

        assert_eq!(outcome.accepted_draft_tokens, 3);
        assert_eq!(outcome.target_token, 13);
    }

    #[test]
    fn greedy_verify_stops_at_first_mismatch() {
        let outcome = verify_greedy(&[10, 11, 12], &[10, 99, 12, 13]).unwrap();

        assert_eq!(outcome.accepted_draft_tokens, 1);
        assert_eq!(outcome.target_token, 99);
    }
}
