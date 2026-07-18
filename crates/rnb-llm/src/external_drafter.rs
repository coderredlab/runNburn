//! External drafter runtime — wraps rnb-mtp's Drafter for engine integration.
//!
//! 상태 머신 책임:
//! - `reset`            — 새 생성 시작 시 position + hidden state 초기화
//! - `shift_for_accept` — target 이 N 개 토큰을 accept 하면 position 전진 + 임시 draft 삭제

use rnb_mtp::drafter::Drafter;
use std::sync::Arc;

/// Engine 에서 external drafter (rnb-mtp `Drafter`) 와 상호작용하는 상태 머신.
pub struct ExternalDrafterRuntime {
    /// 실제 drafter. `None` 은 test stub 전용.
    drafter: Option<Arc<Drafter>>,

    /// `Drafter::backbone_hidden` — 미리 캐시해두는 scalar.
    backbone_hidden: usize,

    /// Target 모델의 마지막 레이어 hidden state (len = backbone_hidden).
    last_target_hidden: Vec<f32>,

    /// Drafter 자체의 마지막 hidden state (len = backbone_hidden).
    last_drafter_hidden: Vec<f32>,

    /// KV cache 상 현재 위치 (accept 후 전진).
    position: u32,

    /// 현재 스텝에서 drafter 가 생성한 draft token id 목록.
    accumulated_drafts: Vec<u32>,

    /// 직전 토큰 id (drafter embedding lookup 용).
    prev_token_id: u32,
}

impl ExternalDrafterRuntime {
    /// 실제 drafter 를 받아 초기화.
    pub fn new(drafter: Arc<Drafter>) -> Self {
        let backbone_hidden = drafter.backbone_hidden;
        Self {
            drafter: Some(drafter),
            backbone_hidden,
            last_target_hidden: Vec::new(),
            last_drafter_hidden: vec![0.0f32; backbone_hidden],
            position: 0,
            accumulated_drafts: Vec::new(),
            prev_token_id: 0,
        }
    }

    /// Test-only — 실제 Drafter 없이 state machine 만 검증할 때 사용.
    #[doc(hidden)]
    pub fn new_stub_for_test(backbone_hidden: usize) -> Self {
        Self {
            drafter: None,
            backbone_hidden,
            last_target_hidden: Vec::new(),
            last_drafter_hidden: vec![0.0f32; backbone_hidden],
            position: 0,
            accumulated_drafts: Vec::new(),
            prev_token_id: 0,
        }
    }

    // ── 상태 전이 ──────────────────────────────────────────────────────────

    /// 새 생성 시작 시 호출. position + hidden 초기화, draft 버퍼 비움.
    ///
    /// `target_last_hidden` 길이는 반드시 `backbone_hidden` 과 같아야 한다.
    pub fn reset(&mut self, target_last_hidden: &[f32], position: u32) {
        assert_eq!(
            target_last_hidden.len(),
            self.backbone_hidden,
            "target_last_hidden len {} != backbone_hidden {}",
            target_last_hidden.len(),
            self.backbone_hidden,
        );
        self.last_target_hidden.clear();
        self.last_target_hidden
            .extend_from_slice(target_last_hidden);
        self.position = position;
        self.accumulated_drafts.clear();
        self.last_drafter_hidden.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Target 이 `accepted` 개 토큰을 검증 통과시켰을 때 호출.
    ///
    /// - `position` 을 `accepted` 만큼 전진시킨다.
    /// - `accumulated_drafts` 와 drafter hidden 을 리셋한다.
    pub fn shift_for_accept(&mut self, accepted: usize) {
        self.position = self.position.saturating_add(accepted as u32);
        self.accumulated_drafts.clear();
        self.last_drafter_hidden.iter_mut().for_each(|x| *x = 0.0);
    }

    // ── Accessor ───────────────────────────────────────────────────────────

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn accumulated_drafts(&self) -> &[u32] {
        &self.accumulated_drafts
    }

    pub fn last_target_hidden(&self) -> &[f32] {
        &self.last_target_hidden
    }

    pub fn last_drafter_hidden(&self) -> &[f32] {
        &self.last_drafter_hidden
    }

    pub fn last_drafter_hidden_mut(&mut self) -> &mut [f32] {
        &mut self.last_drafter_hidden
    }

    pub fn drafter(&self) -> Option<&Arc<Drafter>> {
        self.drafter.as_ref()
    }

    pub fn set_prev_token(&mut self, token_id: u32) {
        self.prev_token_id = token_id;
    }

    pub fn prev_token(&self) -> u32 {
        self.prev_token_id
    }

    pub fn backbone_hidden(&self) -> usize {
        self.backbone_hidden
    }

    // ── Test helpers ────────────────────────────────────────────────────────

    /// Test-only — `accumulated_drafts` 에 토큰 하나 push.
    #[doc(hidden)]
    pub fn test_push_draft(&mut self, token_id: u32) {
        self.accumulated_drafts.push(token_id);
    }
}

use crate::draft_stepper::DraftStepper;
use rnb_mtp::drafter::{drafter_forward, SharedKvStates};

// NOTE (mc78): kept for trait completeness; production loop bypasses this and calls drafter_forward directly. Will be revisited.
pub struct ExternalDrafterStepper<'rt, 'kv> {
    runtime: &'rt mut ExternalDrafterRuntime,
    shared_kv: &'kv SharedKvStates,
}

impl<'rt, 'kv> ExternalDrafterStepper<'rt, 'kv> {
    pub fn new(runtime: &'rt mut ExternalDrafterRuntime, shared_kv: &'kv SharedKvStates) -> Self {
        Self { runtime, shared_kv }
    }
}

impl<'rt, 'kv> DraftStepper for ExternalDrafterStepper<'rt, 'kv> {
    fn reset(&mut self, target_last_hidden: &[f32], position: u32) {
        self.runtime.reset(target_last_hidden, position);
    }

    fn draft_n(&mut self, n: usize) -> Vec<u32> {
        let drafter = self
            .runtime
            .drafter()
            .expect("ExternalDrafterStepper called without a loaded Drafter")
            .clone();
        let mut drafts = Vec::with_capacity(n);
        let backbone = self.runtime.backbone_hidden();
        let mut inputs = vec![0.0f32; 2 * backbone];

        for i in 0..n {
            inputs[..backbone].copy_from_slice(self.runtime.last_target_hidden());
            inputs[backbone..].copy_from_slice(self.runtime.last_drafter_hidden());

            let out = drafter_forward(
                &drafter,
                &inputs,
                self.shared_kv,
                self.runtime.position() + i as u32,
            );
            let tok = argmax(&out.logits);
            drafts.push(tok);
            self.runtime
                .last_drafter_hidden_mut()
                .copy_from_slice(&out.projected_hidden);
            self.runtime.set_prev_token(tok);
            self.runtime.test_push_draft(tok);
        }
        drafts
    }

    fn shift_for_accept(&mut self, accepted: usize) {
        self.runtime.shift_for_accept(accepted);
    }
}

pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let (idx, _) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
    idx as u32
}

#[doc(hidden)]
pub fn test_argmax(logits: &[f32]) -> u32 {
    argmax(logits)
}
