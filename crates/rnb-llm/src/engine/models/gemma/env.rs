//! Gemma4 runtime env-knob helpers and the `GemmaRuntimeFlavor` tag.
//!
//! Each `RNB_GEMMA_*` knob is a small env-var probe. They are grouped here so
//! the hot forward paths in `engine.rs` don't drown in direct env lookups.

use rnb_loader::Architecture as ModelArchitecture;

pub(in crate::engine) fn env_flag(key: &str) -> bool {
    crate::engine::policy::env_flag(key)
}

pub(in crate::engine) fn env_string(key: &str) -> Option<String> {
    crate::engine::policy::env_string(key)
}

fn env_f32(key: &str) -> Option<f32> {
    crate::engine::policy::env_f32(key)
}

fn env_isize(key: &str) -> Option<isize> {
    crate::engine::policy::env_isize(key)
}

pub(in crate::engine) fn env_usize(key: &str) -> Option<usize> {
    crate::engine::policy::env_usize(key)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum GemmaRuntimeFlavor {
    Generic,
    Gemma4E2BIt,
}

pub(in crate::engine) fn gemma_ple_after_out_scale() -> bool {
    if env_flag("RNB_GEMMA_PLE_AFTER_OUT_SCALE") {
        true
    } else if env_flag("RNB_GEMMA_PLE_BEFORE_LAYER") {
        false
    } else {
        false
    }
}

pub(in crate::engine) fn gemma_ple_branch_scale() -> Option<f32> {
    if let Some(v) = env_f32("RNB_GEMMA_PLE_BRANCH_SCALE") {
        return Some(v);
    }
    if env_flag("RNB_GEMMA_PLE_AFTER_OUT_SCALE")
        && env_flag("RNB_GEMMA_PLE_REPLACE_HIDDEN")
        && env_flag("RNB_GEMMA_PLE_SKIP_POST_NORM")
        && env_flag("RNB_GEMMA_PLE_MODEL_ONLY")
    {
        return Some(1.0);
    }
    None
}

pub(in crate::engine) fn gemma_ple_base_mix_scale() -> Option<f32> {
    env_f32("RNB_GEMMA_PLE_BASE_MIX_SCALE")
}

pub(in crate::engine) fn gemma_ple_dynamic_base() -> bool {
    env_flag("RNB_GEMMA_PLE_DYNAMIC_BASE")
}

pub(in crate::engine) fn gemma_ple_layer_enabled(layer_idx: usize) -> bool {
    let mut enabled = true;
    if let Some(min_idx) = env_usize("RNB_GEMMA_PLE_LAYER_MIN") {
        enabled &= layer_idx >= min_idx;
    }
    if let Some(max_idx) = env_usize("RNB_GEMMA_PLE_LAYER_MAX") {
        enabled &= layer_idx <= max_idx;
    }
    if let Some(want) = env_usize("RNB_GEMMA_PLE_LAYER_ONLY") {
        return enabled && layer_idx == want;
    }
    if gemma_ple_layer34_fix() {
        enabled &= layer_idx == 34;
    }
    enabled
}

pub(in crate::engine) fn gemma_ple_use_layer_input() -> bool {
    env_flag("RNB_GEMMA_PLE_USE_LAYER_INPUT")
}

pub(in crate::engine) fn gemma_ple_pre_emb_scale_base() -> bool {
    env_flag("RNB_GEMMA_PLE_PRE_EMB_SCALE_BASE")
}

pub(in crate::engine) fn gemma_ple_before_layer() -> bool {
    if env_flag("RNB_GEMMA_PLE_BEFORE_LAYER") {
        true
    } else if env_flag("RNB_GEMMA_PLE_AFTER_OUT_SCALE") {
        false
    } else {
        false
    }
}

pub(in crate::engine) fn gemma_ple_pre_norm_input() -> bool {
    env_flag("RNB_GEMMA_PLE_PRE_NORM_INPUT")
}

pub(in crate::engine) fn gemma_ple_layer34_fix() -> bool {
    // HF Gemma4TextDecoderLayer applies the per-layer-embd branch on ALL layers,
    // so the "layer 34 only" restriction is wrong by the reference model. Leave it
    // behind an explicit env opt-in for legacy comparison only.
    if crate::engine::policy::gemma_ple_global_only_enabled() {
        return false;
    }
    env_flag("RNB_GEMMA_PLE_LAYER34_FIX") || env_flag("RNB_GEMMA_PLE_LAYER34_MODEL_FIX")
}

pub(in crate::engine) fn gemma_ple_layer34_hard_fix_applies(
    architecture: ModelArchitecture,
    layer_idx: usize,
    num_layers: usize,
) -> bool {
    !matches!(architecture, ModelArchitecture::Gemma4)
        && gemma_ple_layer34_fix()
        && gemma_ple_effective_layer_idx(layer_idx, num_layers) == 34
}

pub(in crate::engine) fn gemma_ple_disable_attention_layer(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_DISABLE_ATTN_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_unit_offset_post_attn_enabled(layer_idx: usize) -> bool {
    env_flag("RNB_GEMMA_UNIT_OFFSET_POST_ATTN_ONLY")
        || crate::engine::policy::env_layer_matches(
            "RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER",
            layer_idx,
        )
}

pub(in crate::engine) fn gemma_disable_e2bit_local_defaults() -> bool {
    env_flag("RNB_DISABLE_GEMMA_E2BIT_LOCAL_DEFAULTS")
}

pub(in crate::engine) fn gemma_default_unit_offset_post_attn_enabled(
    _architecture: ModelArchitecture,
    _gemma_runtime_flavor: GemmaRuntimeFlavor,
    _layer_idx: usize,
) -> bool {
    // Disabled by default for Gemma4 E2B-it: empirical probes with SWA source=12
    // show post_attn unit-offset net-harmful. Earlier 13/14/24/34 default was tuned
    // for the older SWA=11 path; swap to keep env seam only.
    false
}

pub(in crate::engine) fn gemma_effective_unit_offset_post_attn_enabled(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    layer_idx: usize,
) -> bool {
    gemma_unit_offset_post_attn_enabled(layer_idx)
        || gemma_default_unit_offset_post_attn_enabled(
            architecture,
            gemma_runtime_flavor,
            layer_idx,
        )
}

pub(in crate::engine) fn gemma_unit_offset_post_attn_prefill_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches(
        "RNB_GEMMA_UNIT_OFFSET_POST_ATTN_PREFILL_LAYER",
        layer_idx,
    )
}

pub(in crate::engine) fn gemma_skip_post_attn_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_POST_ATTN_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_skip_post_attn_prefill_enabled_env(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_POST_ATTN_PREFILL_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_default_skip_post_attn_prefill_enabled(
    _architecture: ModelArchitecture,
    _gemma_runtime_flavor: GemmaRuntimeFlavor,
    _layer_idx: usize,
) -> bool {
    false
}

pub(in crate::engine) fn gemma_effective_skip_post_attn_prefill_enabled(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    layer_idx: usize,
) -> bool {
    gemma_skip_post_attn_prefill_enabled_env(layer_idx)
        || gemma_default_skip_post_attn_prefill_enabled(
            architecture,
            gemma_runtime_flavor,
            layer_idx,
        )
}

pub(in crate::engine) fn gemma_skip_post_attn_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_POST_ATTN_DECODE_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_post_attn_decode_plain_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_POST_ATTN_DECODE_PLAIN_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_post_attn_blend_source_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches(
        "RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_LAYER",
        layer_idx,
    )
}

pub(in crate::engine) fn gemma_post_attn_blend_source_decode_alpha() -> f32 {
    env_f32("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_ALPHA")
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.25)
}

pub(in crate::engine) fn gemma_pre_residual_blend_source_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches(
        "RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_LAYER",
        layer_idx,
    )
}

pub(in crate::engine) fn gemma_pre_residual_blend_source_decode_alpha() -> f32 {
    env_f32("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_ALPHA")
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.25)
}

pub(in crate::engine) fn gemma_decode_hidden_snapshots_needed() -> bool {
    env_string("RNB_GEMMA_REUSE_SOURCE_HIDDEN_DECODE_LAYER").is_some()
        || env_string("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_LAYER").is_some()
        || env_string("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_LAYER").is_some()
}

pub(in crate::engine) fn gemma_disable_attn_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_DISABLE_ATTN_DECODE_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_disable_layer_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_DISABLE_LAYER_DECODE", layer_idx)
}

pub(in crate::engine) fn gemma_reuse_source_hidden_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches(
        "RNB_GEMMA_REUSE_SOURCE_HIDDEN_DECODE_LAYER",
        layer_idx,
    )
}

pub(in crate::engine) fn gemma_reuse_source_hidden_decode_blend_alpha() -> f32 {
    env_f32("RNB_GEMMA_REUSE_SOURCE_HIDDEN_DECODE_ALPHA")
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.25)
}

pub(in crate::engine) fn gemma_blend_source_decode_src_layer() -> usize {
    env_usize("RNB_GEMMA_BLEND_SOURCE_DECODE_SRC_LAYER").unwrap_or(11)
}

pub(in crate::engine) fn gemma_skip_ffn_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_FFN_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_skip_ffn_decode_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_FFN_DECODE_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_default_skip_ffn_enabled(
    _architecture: ModelArchitecture,
    _gemma_runtime_flavor: GemmaRuntimeFlavor,
    _layer_idx: usize,
) -> bool {
    false
}

pub(in crate::engine) fn gemma_effective_skip_ffn_enabled(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    layer_idx: usize,
) -> bool {
    gemma_skip_ffn_enabled(layer_idx)
        || gemma_default_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
}

pub(in crate::engine) fn gemma_effective_skip_ffn_decode_enabled(
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    layer_idx: usize,
) -> bool {
    gemma_skip_ffn_decode_enabled(layer_idx)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
}

pub(in crate::engine) fn gemma_skip_out_scale_enabled(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_GEMMA_SKIP_OUT_SCALE_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_enable_out_scale_opt_in() -> bool {
    if env_flag("RNB_GEMMA_DISABLE_OUT_SCALE") {
        return false;
    }
    true
}

pub(in crate::engine) fn force_tokenwise_prefill_attn_layer(layer_idx: usize) -> bool {
    crate::engine::policy::env_layer_matches("RNB_FORCE_TOKENWISE_PREFILL_ATTN_LAYER", layer_idx)
}

pub(in crate::engine) fn gemma_ple_model_only() -> bool {
    if env_flag("RNB_GEMMA_PLE_TOKEN_ONLY") {
        false
    } else if env_flag("RNB_GEMMA_PLE_MODEL_ONLY") {
        true
    } else {
        false
    }
}

pub(in crate::engine) fn gemma_ple_token_only() -> bool {
    env_flag("RNB_GEMMA_PLE_TOKEN_ONLY")
}

pub(in crate::engine) fn gemma_ple_skip_post_norm() -> bool {
    if env_flag("RNB_GEMMA_PLE_UNIT_OFFSET_POST_NORM") {
        false
    } else if env_flag("RNB_GEMMA_PLE_SKIP_POST_NORM") {
        true
    } else {
        false
    }
}

pub(in crate::engine) fn gemma_ple_replace_hidden() -> bool {
    env_flag("RNB_GEMMA_PLE_REPLACE_HIDDEN")
}

pub(in crate::engine) fn gemma_ple_hidden_blend_alpha() -> Option<f32> {
    env_f32("RNB_GEMMA_PLE_BLEND_HIDDEN")
}

pub(in crate::engine) fn gemma_ple_after_final_norm() -> bool {
    env_flag("RNB_GEMMA_PLE_AFTER_FINAL_NORM")
}

pub(in crate::engine) fn gemma_ple_effective_layer_idx(
    layer_idx: usize,
    num_layers: usize,
) -> usize {
    let mut idx = layer_idx.min(num_layers.saturating_sub(1));
    if env_flag("RNB_GEMMA_PLE_LAYER_REVERSE") {
        idx = num_layers.saturating_sub(1).saturating_sub(idx);
    }
    if let Some(delta) = env_isize("RNB_GEMMA_PLE_LAYER_OFFSET") {
        let n = num_layers.max(1) as isize;
        let shifted = ((idx as isize + delta) % n + n) % n;
        idx = shifted as usize;
    }
    idx
}

pub(in crate::engine) fn gemma_ple_token_scale() -> Option<f32> {
    env_f32("RNB_GEMMA_PLE_TOKEN_SCALE")
}

pub(in crate::engine) fn gemma_ple_projected_scale() -> Option<f32> {
    env_f32("RNB_GEMMA_PLE_PROJECTED_SCALE")
}
