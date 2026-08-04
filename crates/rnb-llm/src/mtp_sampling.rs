//! MTP/speculative round의 accept 판정과 emit 회계를 한곳에서 소유한다.
//!
//! Backend별 verify execution(Metal batched decode-chain, Vulkan fullpath, CUDA device
//! resident, batch prefill, sequential, external drafter, DSpark)은 target prediction을
//! 만드는 방법만 다르고, "draft prefix 중 어디까지 target과 일치하는가"와 "일치한 토큰을
//! stop/callback/token budget에 어떻게 대입하는가"는 같은 계약이다. 그 계약을 여기서만
//! 정의한다.
//!
//! API가 클로저가 아니라 상태 머신인 이유는 sequential execution 때문이다. sequential은
//! target 예측을 얻을 때 `engine`을 mutable로 빌리고 emit할 때 `engine.tokenizer`를
//! 불변으로 빌리므로, 두 동작을 각각 클로저로 캡처하면 동시 차용이 되어 컴파일되지 않는다.
//! 호출부가 루프를 소유하고 판정만 위임하면 차용이 순차적으로 유지된다.
//!
//! 확률적 sampling(modified rejection sampling)이 추가되면 [`GreedyRound::observe`]의
//! 판정만 교체되고 emit 회계와 호출부 골격은 그대로 재사용된다.

use crate::generate::GenerateParams;

/// 한 위치를 판정한 뒤 호출부가 취해야 할 동작.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoundAction {
    /// target이 draft와 달라 round가 끝났다. 호출부는 루프를 벗어난다.
    Reject,
    /// stop token이라 round가 끝났다. 해당 토큰은 accept하지도 출력하지도 않는다.
    Stop,
    /// accept했지만 emit 범위 밖이다(verified runway 영역). 다음 위치로 진행한다.
    AcceptWithoutEmit,
    /// accept했고 출력해야 한다. 호출부는 토큰을 밀어 넣은 뒤
    /// [`GreedyRound::after_emit`]을 호출한다.
    Emit,
}

/// 한 round의 accept/emit 회계.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct GreedyRound {
    accepted: usize,
    emitted: usize,
    examined: usize,
    stopped: bool,
    mismatch_target: Option<u32>,
}

impl GreedyRound {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 위치 `index`의 draft와 target을 대조한다.
    ///
    /// `emit_limit` 이상의 위치는 accept만 하고 출력하지 않는다. verified runway가
    /// 미리 확정해 둘 토큰이 여기에 해당한다.
    pub(crate) fn observe(
        &mut self,
        index: usize,
        draft_token: u32,
        target_token: u32,
        emit_limit: usize,
        params: &GenerateParams,
        eos: u32,
    ) -> RoundAction {
        self.examined += 1;

        if target_token != draft_token {
            self.mismatch_target = Some(target_token);
            return RoundAction::Reject;
        }

        if params.should_stop(draft_token, eos) {
            self.stopped = true;
            return RoundAction::Stop;
        }

        self.accepted += 1;

        if index >= emit_limit {
            return RoundAction::AcceptWithoutEmit;
        }

        self.emitted += 1;
        RoundAction::Emit
    }

    /// 출력 직후의 budget/callback 회계. 계속 진행할 수 있으면 `true`.
    ///
    /// callback이 거부한 토큰도 이미 accept와 emit에 포함되어 있다. 이는 리팩터 이전
    /// 모든 execution의 공통 동작이며 committed prefix 길이에도 그대로 반영된다.
    pub(crate) fn after_emit(&mut self, keep_going: bool, tokens_remaining: &mut usize) -> bool {
        *tokens_remaining -= 1;
        if !keep_going {
            self.stopped = true;
            return false;
        }
        *tokens_remaining != 0
    }

    /// target과 연속으로 일치해 accept된 draft 토큰 수. committed prefix는 `1 + accepted`다.
    pub(crate) fn accepted(&self) -> usize {
        self.accepted
    }

    /// 실제로 출력 스트림에 밀어 넣은 draft 토큰 수.
    pub(crate) fn emitted(&self) -> usize {
        self.emitted
    }

    /// stop token, callback 거부로 round가 끝났다.
    ///
    /// budget이 정확히 0이 되어 끝난 경우는 호출부가 `tokens_remaining == 0`으로도
    /// 판정할 수 있도록 `true`로 만들지 않는다.
    pub(crate) fn stopped(&self) -> bool {
        self.stopped
    }

    /// 첫 불일치 위치의 target 토큰. 다음 round의 current token이 된다.
    pub(crate) fn mismatch_target(&self) -> Option<u32> {
        self.mismatch_target
    }

    /// target 예측을 실제로 조회한 횟수.
    pub(crate) fn examined(&self) -> usize {
        self.examined
    }
}

/// draft prefix와 target 예측의 최장 일치 길이.
///
/// stop/budget과 무관한 순수 수학이며, target 예측을 한 번에 전부 갖고 있는 execution이
/// accept 수학과 emit 회계를 분리해 수행할 때 사용한다. 확률적 경로에서는 이 함수가
/// modified rejection sampling으로 교체된다.
pub(crate) fn greedy_matched_prefix(draft_tokens: &[u32], target_tokens: &[u32]) -> usize {
    draft_tokens
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count()
}

/// 한 위치에서 두 draft 커플링의 accept 확률.
///
/// 확률적 MTP의 이득을 판정하는 값이다. §Draft proposal이 draft를 `q`에서 샘플링하도록
/// 규정한 이유가 여기 있다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AcceptanceProbabilities {
    /// draft를 `q`의 argmax로 뽑는 현재 방식의 accept 확률 `p(argmax q)`.
    ///
    /// 이 커플링은 draft가 point mass이므로 modified rejection sampling을 적용해도
    /// `sum_v min(p, q) = min(p(v̂), 1) = p(v̂)`로 같은 값이다. 즉 draft를 argmax로
    /// 두는 한 rejection sampling의 이득은 정확히 0이다.
    pub greedy_draft: f32,
    /// draft를 `q`에서 샘플링하고 modified rejection sampling을 쓸 때의 accept 확률
    /// `sum_v min(p(v), q(v))`.
    pub sampled_draft_rejection: f32,
}

/// 정규화된 target 분포 `p`와 draft 분포 `q`에서 두 커플링의 accept 확률을 계산한다.
///
/// 길이가 다르면 `None`이다. 확률 합이 1에서 벗어나는지는 검사하지 않는다 — 호출부가
/// 같은 softmax로 만든 row를 넘긴다는 계약이다.
pub(crate) fn acceptance_probabilities(p: &[f32], q: &[f32]) -> Option<AcceptanceProbabilities> {
    if p.len() != q.len() || p.is_empty() {
        return None;
    }

    let mut best_index = 0usize;
    let mut best_mass = q[0];
    let mut overlap = 0.0f32;
    for (index, (&p_v, &q_v)) in p.iter().zip(q).enumerate() {
        if q_v > best_mass {
            best_mass = q_v;
            best_index = index;
        }
        overlap += p_v.min(q_v);
    }

    Some(AcceptanceProbabilities {
        greedy_draft: p[best_index],
        sampled_draft_rejection: overlap,
    })
}

/// 여러 위치의 accept 확률을 누적해 평균을 낸다.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AcceptanceStats {
    positions: usize,
    greedy_sum: f64,
    rejection_sum: f64,
}

impl AcceptanceStats {
    pub(crate) fn record(&mut self, probabilities: AcceptanceProbabilities) {
        self.positions += 1;
        self.greedy_sum += f64::from(probabilities.greedy_draft);
        self.rejection_sum += f64::from(probabilities.sampled_draft_rejection);
    }

    pub(crate) fn positions(&self) -> usize {
        self.positions
    }

    /// `(greedy 평균, rejection 평균)`. 표본이 없으면 `None`.
    pub(crate) fn means(&self) -> Option<(f64, f64)> {
        if self.positions == 0 {
            return None;
        }
        let n = self.positions as f64;
        Some((self.greedy_sum / n, self.rejection_sum / n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harness {
        round: GreedyRound,
        emitted: Vec<u32>,
        remaining: usize,
    }

    /// 호출부 골격을 그대로 재현한다. 실제 execution들이 이 형태를 공유한다.
    fn drive(
        draft: &[u32],
        target: &[u32],
        emit_limit: usize,
        params: &GenerateParams,
        eos: u32,
        budget: usize,
        callback_rejects_at: Option<usize>,
    ) -> Harness {
        let mut harness = Harness {
            round: GreedyRound::new(),
            emitted: Vec::new(),
            remaining: budget,
        };

        for i in 0..draft.len() {
            match harness
                .round
                .observe(i, draft[i], target[i], emit_limit, params, eos)
            {
                RoundAction::Reject | RoundAction::Stop => break,
                RoundAction::AcceptWithoutEmit => continue,
                RoundAction::Emit => {}
            }
            harness.emitted.push(draft[i]);
            let keep_going = callback_rejects_at != Some(harness.emitted.len() - 1);
            if !harness.round.after_emit(keep_going, &mut harness.remaining) {
                break;
            }
        }

        harness
    }

    fn params_with_stop(stop: Vec<u32>) -> GenerateParams {
        GenerateParams {
            stop_tokens: stop,
            ..GenerateParams::default()
        }
    }

    #[test]
    fn full_accept_consumes_budget_and_emits_every_token() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 11, 12], &[10, 11, 12], 3, &params, 99, 8, None);

        assert_eq!(h.round.accepted(), 3);
        assert_eq!(h.round.emitted(), 3);
        assert!(!h.round.stopped());
        assert_eq!(h.round.mismatch_target(), None);
        assert_eq!(h.emitted, vec![10, 11, 12]);
        assert_eq!(h.remaining, 5);
    }

    #[test]
    fn mismatch_stops_accepting_and_reports_target_token() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 11, 12], &[10, 77, 12], 3, &params, 99, 8, None);

        assert_eq!(h.round.accepted(), 1);
        assert_eq!(h.round.emitted(), 1);
        assert_eq!(h.round.mismatch_target(), Some(77));
        assert_eq!(h.round.examined(), 2);
        assert!(!h.round.stopped());
        assert_eq!(h.emitted, vec![10]);
        assert_eq!(h.remaining, 7);
    }

    #[test]
    fn stop_token_is_neither_accepted_nor_emitted() {
        let params = params_with_stop(vec![11]);
        let h = drive(&[10, 11, 12], &[10, 11, 12], 3, &params, 99, 8, None);

        assert_eq!(h.round.accepted(), 1);
        assert_eq!(h.round.emitted(), 1);
        assert!(h.round.stopped());
        assert_eq!(h.round.mismatch_target(), None);
        assert_eq!(h.emitted, vec![10]);
        assert_eq!(h.remaining, 7);
    }

    #[test]
    fn eos_stops_the_round_before_commit() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 42, 12], &[10, 42, 12], 3, &params, 42, 8, None);

        assert_eq!(h.round.accepted(), 1);
        assert!(h.round.stopped());
        assert_eq!(h.emitted, vec![10]);
    }

    #[test]
    fn rejected_callback_token_still_counts_as_accepted_and_emitted() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 11, 12], &[10, 11, 12], 3, &params, 99, 8, Some(1));

        assert_eq!(h.round.accepted(), 2);
        assert_eq!(h.round.emitted(), 2);
        assert!(h.round.stopped());
        assert_eq!(h.emitted, vec![10, 11]);
        assert_eq!(h.remaining, 6);
    }

    #[test]
    fn exhausted_budget_breaks_without_setting_stopped() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 11, 12], &[10, 11, 12], 3, &params, 99, 2, None);

        assert_eq!(h.round.accepted(), 2);
        assert_eq!(h.round.emitted(), 2);
        assert!(!h.round.stopped());
        assert_eq!(h.emitted, vec![10, 11]);
        assert_eq!(h.remaining, 0);
    }

    #[test]
    fn tokens_beyond_emit_limit_are_accepted_without_emitting() {
        let params = params_with_stop(vec![]);
        let h = drive(&[10, 11, 12], &[10, 11, 12], 1, &params, 99, 8, None);

        assert_eq!(h.round.accepted(), 3);
        assert_eq!(h.round.emitted(), 1);
        assert_eq!(h.emitted, vec![10]);
        assert_eq!(h.remaining, 7);
    }

    #[test]
    fn greedy_matched_prefix_counts_leading_equal_tokens_only() {
        assert_eq!(greedy_matched_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(greedy_matched_prefix(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(greedy_matched_prefix(&[1, 2, 3], &[9, 2, 3]), 0);
        assert_eq!(greedy_matched_prefix(&[], &[1, 2]), 0);
    }

    #[test]
    fn point_mass_draft_gains_nothing_from_rejection_sampling() {
        // q가 argmax point mass면 sum_v min(p, q) = p(v̂)다. draft를 argmax로 두는 한
        // modified rejection sampling의 이득이 0이라는 사실을 고정한다.
        let p = [0.5, 0.3, 0.2];
        let q = [1.0, 0.0, 0.0];
        let probs = acceptance_probabilities(&p, &q).expect("same length");

        assert!((probs.greedy_draft - 0.5).abs() < 1e-6);
        assert!((probs.sampled_draft_rejection - 0.5).abs() < 1e-6);
    }

    #[test]
    fn matching_distributions_accept_always_only_with_a_sampled_draft() {
        // p == q인 완벽한 drafter. argmax 커플링은 p(v̂)에서 멈추고 rejection sampling만
        // 1.0에 도달한다. 이 격차가 확률적 MTP의 유일한 이득 원천이다.
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];
        let probs = acceptance_probabilities(&p, &q).expect("same length");

        assert!((probs.greedy_draft - 0.5).abs() < 1e-6);
        assert!((probs.sampled_draft_rejection - 1.0).abs() < 1e-6);
    }

    #[test]
    fn disjoint_support_never_accepts_a_sampled_draft() {
        let p = [0.0, 1.0];
        let q = [1.0, 0.0];
        let probs = acceptance_probabilities(&p, &q).expect("same length");

        assert!(probs.greedy_draft.abs() < 1e-6);
        assert!(probs.sampled_draft_rejection.abs() < 1e-6);
    }

    #[test]
    fn acceptance_probabilities_reject_mismatched_or_empty_rows() {
        assert!(acceptance_probabilities(&[0.5, 0.5], &[1.0]).is_none());
        assert!(acceptance_probabilities(&[], &[]).is_none());
    }

    #[test]
    fn acceptance_stats_average_each_coupling_separately() {
        let mut stats = AcceptanceStats::default();
        assert!(stats.means().is_none());

        stats.record(AcceptanceProbabilities {
            greedy_draft: 0.2,
            sampled_draft_rejection: 0.8,
        });
        stats.record(AcceptanceProbabilities {
            greedy_draft: 0.4,
            sampled_draft_rejection: 0.6,
        });

        let (greedy, rejection) = stats.means().expect("two samples");
        assert_eq!(stats.positions(), 2);
        assert!((greedy - 0.3).abs() < 1e-6);
        assert!((rejection - 0.7).abs() < 1e-6);
    }
}
