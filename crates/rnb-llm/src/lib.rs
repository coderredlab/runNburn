pub mod auto_drafter;
pub mod chat;
pub mod constrained;
pub mod draft_stepper;
pub mod engine;
pub mod error;
pub mod external_drafter;
pub mod generate;
pub mod kv_cache;
mod mtp_generate;
mod runtime;
pub mod sampler;
pub mod speculative;
pub mod tokenizer;
pub mod tool_call;

pub use chat::{ChatMessage, ChatTemplateOptions};
pub use constrained::GenerationConstraint;
pub use engine::moe::MoeLayerView;
pub use engine::moe_trace;
pub use engine::{
    dump_chain_diag_aggregate, Engine, EngineLoadConfig, EngineSequenceState, PrefillDriftRecord,
    PrefillDriftTrace,
};
pub use generate::{
    GenerateParams, GenerateResult, GenerationCancellation, MirostatParams, TextStopFilter,
};
pub use kv_cache::KVCache;
pub use sampler::SamplerChain;
pub use tokenizer::Tokenizer;
pub use tool_call::{
    parse_assistant_output, ParsedAssistantOutput, ParsedToolCall, ToolCallFormat,
};

// SharedKvStates / SharedKvLayer 는 rnb-mtp 에 정의된다.
// rnb-llm 은 rnb-mtp 에 의존 (helper direction). 편의를 위해 re-export.
pub use rnb_mtp::{SharedKvLayer, SharedKvStates};

pub fn reset_metal_prefill_atn_full_counters() {
    engine::reset_metal_prefill_atn_full_counters();
}

pub fn report_metal_prefill_atn_full_counters(label: &str) {
    engine::report_metal_prefill_atn_full_counters(label);
}

pub fn reset_metal_prefill_atn_o_tail_counters() {
    engine::reset_metal_prefill_atn_o_tail_counters();
}

pub fn report_metal_prefill_atn_o_tail_counters(label: &str) {
    engine::report_metal_prefill_atn_o_tail_counters(label);
}

pub fn reset_metal_decode_parity_counters() {
    engine::reset_metal_decode_parity_counters();
}

pub fn report_metal_decode_parity_counters(label: &str) {
    engine::report_metal_decode_parity_counters(label);
}

/// Read-only borrow of a target's KV cache. Used by speculative drafter
/// (e.g. `rnb_mtp::drafter`) for cross-attention without owning a private
/// KV cache.
///
/// 본 trait 의 `k_layer`/`v_layer` 는 매 호출마다 F16 (u16 bits) 캐시를 f32 로 dequant
/// 한 `Vec<f32>` 를 반환한다. KVCache 의 native dtype 이 f16 이라 zero-copy 슬라이스
/// 가 불가능하다. drafter 의 cross-attention 은 layer 당 1-2 회 호출이므로 이
/// dequant 비용은 sub-ms 수준이며 별도 f32 view 캐시는 두지 않는다.
pub trait KvBorrow {
    /// Dequant 된 K cache for `target_layer_idx`. 길이 = `pos() * kv_dim_for_layer(target_layer_idx)`.
    /// layout: row-major `[pos, kv_dim]`.
    fn k_layer(&self, target_layer_idx: usize) -> Vec<f32>;

    /// Dequant 된 V cache for `target_layer_idx`. 같은 layout.
    fn v_layer(&self, target_layer_idx: usize) -> Vec<f32>;

    /// Current cache position (cached token 수).
    fn pos(&self) -> usize;

    /// KV-head dim 합 (`head_dim * num_kv_heads`) for given target layer.
    fn kv_dim_for_layer(&self, target_layer_idx: usize) -> usize;

    /// Target decoder layer 수.
    fn n_layers(&self) -> usize;
}

pub fn compiled_runtime_backends() -> Vec<&'static str> {
    crate::runtime::BackendRegistry::compiled()
        .backends()
        .iter()
        .map(|backend| match backend {
            crate::runtime::BackendKind::Cpu => "cpu",
            crate::runtime::BackendKind::Cuda => "cuda",
            crate::runtime::BackendKind::Vulkan => "vulkan",
            crate::runtime::BackendKind::OpenCl => "opencl",
            crate::runtime::BackendKind::Metal => "metal",
            crate::runtime::BackendKind::MediaTekNpu => "mediatek-npu",
        })
        .collect()
}

#[cfg(test)]
mod runtime_boundary_tests;
