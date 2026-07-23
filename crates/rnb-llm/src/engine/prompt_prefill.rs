use super::Engine;

fn should_use_full_prefill_pool(
    architecture: rnb_loader::Architecture,
    token_count: usize,
    enabled: bool,
) -> bool {
    enabled && token_count > 1 && matches!(architecture, rnb_loader::Architecture::Qwen35MoE)
}

impl Engine {
    fn prompt_uses_full_prefill_pool(&self, tokens: &[u32]) -> bool {
        should_use_full_prefill_pool(
            self.architecture(),
            tokens.len(),
            super::policy::moe_prefill_full_cores_enabled(),
        )
    }

    pub(crate) fn forward_prompt(&mut self, tokens: &[u32]) -> crate::error::Result<Vec<f32>> {
        if self.prompt_uses_full_prefill_pool(tokens) {
            super::cpu_phase_runtime::install_full_prefill(|| self.forward(tokens))
                .map_err(crate::error::LlmError::Forward)?
        } else {
            self.forward(tokens)
        }
    }

    pub(crate) fn forward_prompt_with_logits(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<f32>> {
        if self.prompt_uses_full_prefill_pool(tokens) {
            super::cpu_phase_runtime::install_full_prefill(|| self.forward_with_logits(tokens))
                .map_err(crate::error::LlmError::Forward)?
        } else {
            self.forward_with_logits(tokens)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_prefill_pool_is_limited_to_multi_token_qwen35_moe() {
        assert!(should_use_full_prefill_pool(
            rnb_loader::Architecture::Qwen35MoE,
            2,
            true
        ));
        assert!(!should_use_full_prefill_pool(
            rnb_loader::Architecture::Qwen35MoE,
            1,
            true
        ));
        assert!(!should_use_full_prefill_pool(
            rnb_loader::Architecture::Gemma,
            2,
            true
        ));
        assert!(!should_use_full_prefill_pool(
            rnb_loader::Architecture::Qwen35MoE,
            2,
            false
        ));
    }
}
