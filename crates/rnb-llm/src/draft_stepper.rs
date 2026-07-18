//! Draft step abstraction shared by in-model nextn and external drafter paths.

/// Drives the speculative draft phase of MTP. Each impl produces N draft
/// tokens given a (target_last_hidden, position) anchor and supports rolling
/// internal state forward after partial accept.
pub trait DraftStepper {
    /// Reset state to start a new draft cycle anchored at `position` with
    /// `target_last_hidden` = target Engine's last layer hidden after the
    /// most recently committed token.
    fn reset(&mut self, target_last_hidden: &[f32], position: u32);

    /// Produce `n` autoregressive draft tokens.
    fn draft_n(&mut self, n: usize) -> Vec<u32>;

    /// Shift internal state forward by `accepted` accepted tokens so the
    /// next `reset` anchors correctly. `accepted ∈ [0, n]`.
    fn shift_for_accept(&mut self, accepted: usize);
}
