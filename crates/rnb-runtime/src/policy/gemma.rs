use super::base::{env_flag, env_layer_matches};

pub fn gemma_tokenizer_bpe_enabled() -> bool {
    // mt89 (2026-05-14): default ON. opt-out via RNB_GEMMA_TOKENIZER_BPE_OFF=1.
    // Gemma 4 GGUF (model=="gemma4") 만 영향. 다른 model ("gpt2" / "llama" / etc)
    // 는 model_init.rs 의 다른 분기 path 라 무관.
    !env_flag("RNB_GEMMA_TOKENIZER_BPE_OFF")
}

pub fn debug_gemma_layout_enabled() -> bool {
    env_flag("RNB_DEBUG_GEMMA_LAYOUT")
}

pub fn gemma_neox_rope_enabled() -> bool {
    !env_flag("RNB_GEMMA_DISABLE_NEOX_ROPE")
}

pub fn gemma_skip_output_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_SKIP_OUTPUT_NORM")
}

pub fn gemma_output_norm_prefill_unit_offset_disabled() -> bool {
    env_flag("RNB_GEMMA_DISABLE_OUTPUT_NORM_PREFILL_UNIT_OFFSET")
}

pub fn gemma_output_norm_decode_unit_offset_disabled() -> bool {
    env_flag("RNB_GEMMA_DISABLE_OUTPUT_NORM_DECODE_UNIT_OFFSET")
}

pub fn gemma_unit_offset_output_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_OUTPUT_NORM")
}

pub fn gemma_unit_offset_attn_ffn_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_ATTN_FFN_NORM")
}

pub fn gemma_unit_offset_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_NORM")
}

pub fn gemma_unit_offset_main_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_MAIN_NORM")
}

pub fn gemma_unit_offset_attn_norm_enabled(layer_idx: usize) -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_ATTN_ONLY")
        || env_layer_matches("RNB_GEMMA_UNIT_OFFSET_ATTN_NORM_LAYER", layer_idx)
}

pub fn gemma_v_norm_enabled() -> bool {
    !env_flag("RNB_DISABLE_GEMMA_V_NORM")
}

pub fn gemma_reused_reapply_k_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_REUSED_REAPPLY_K_NORM")
}

pub fn gemma_unit_offset_ffn_pre_norm_enabled(layer_idx: usize) -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_FFN_ONLY")
        || env_flag("RNB_GEMMA_UNIT_OFFSET_FFN_PRE_ONLY")
        || env_layer_matches("RNB_GEMMA_UNIT_OFFSET_FFN_NORM_LAYER", layer_idx)
}

pub fn gemma_unit_offset_ffn_post_norm_enabled() -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_POST_FFW_ONLY")
        || env_flag("RNB_GEMMA_UNIT_OFFSET_FFN_POST_ONLY")
}

pub fn gemma_ple_global_only_enabled() -> bool {
    env_flag("RNB_GEMMA_PLE_GLOBAL_ONLY")
}

pub fn gemma_qk_norm_disabled() -> bool {
    env_flag("RNB_DISABLE_GEMMA_QK_NORM")
}

pub fn gemma4_moe_expert_major_enabled() -> bool {
    // mc81 (2026-07-14): default ON after Flip4 ABAB; diagnostic opt-out only.
    !env_flag("RNB_GEMMA4_MOE_EXPERT_MAJOR_OFF")
}
