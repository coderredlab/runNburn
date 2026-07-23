use std::path::PathBuf;

use super::base::{env_flag, env_string, env_usize, layer_matches_spec};

pub fn debug_kv_trace_enabled() -> bool {
    env_flag("RNB_DEBUG_KV_TRACE")
}

pub fn debug_layer_trace_enabled() -> bool {
    env_flag("RNB_DEBUG_LAYER_TRACE")
}

pub fn debug_attn_trace_enabled() -> bool {
    env_flag("RNB_DEBUG_ATTN_TRACE")
}

pub fn debug_targeted_attn_trace_enabled(layer_idx: usize) -> bool {
    env_usize("RNB_DEBUG_ATTN_LAYER").is_some_and(|target| layer_idx == target)
        || (layer_idx == 34 && env_flag("RNB_DEBUG_LAYER34_ATTN"))
}

pub fn debug_ple_trace_enabled() -> bool {
    env_flag("RNB_DEBUG_PLE_TRACE")
}

pub fn dump_bin_dir() -> Option<PathBuf> {
    std::env::var_os("RNB_DUMP_BIN_DIR").map(PathBuf::from)
}

pub fn dump_bin_layer_enabled(layer_idx: usize) -> bool {
    if layer_idx == usize::MAX {
        return true;
    }
    env_string("RNB_DUMP_BIN_LAYER_FILTER")
        .as_deref()
        .map(|raw| layer_matches_spec(raw, layer_idx))
        .unwrap_or(true)
}

pub fn dump_bin_log_enabled() -> bool {
    env_flag("RNB_DUMP_BIN_LOG")
}

pub fn debug_ple_layer_matches(layer_idx: usize) -> bool {
    env_usize("RNB_DEBUG_PLE_LAYER").is_none_or(|want| layer_idx == want)
}

pub fn debug_ple_dump_layer() -> Option<usize> {
    env_usize("RNB_DEBUG_PLE_DUMP_LAYER")
}

pub fn debug_ple_dump_take() -> usize {
    env_usize("RNB_DEBUG_PLE_DUMP_TAKE").unwrap_or(16)
}

pub fn debug_gemma4_ple_replay_dir() -> Option<PathBuf> {
    std::env::var_os("RNB_DEBUG_GEMMA4_PLE_REPLAY_DIR").map(PathBuf::from)
}

pub fn debug_gemma4_ple_replay_layer() -> Option<usize> {
    env_usize("RNB_DEBUG_GEMMA4_PLE_REPLAY_LAYER")
}

pub fn debug_final_dump_enabled() -> bool {
    env_flag("RNB_DEBUG_FINAL_DUMP")
}

pub fn debug_final_dump_take() -> usize {
    env_usize("RNB_DEBUG_FINAL_DUMP_TAKE").unwrap_or(16)
}

pub fn debug_decode_targets() -> Option<Vec<String>> {
    let targets = env_string("RNB_DEBUG_DECODE_TARGETS")?
        .split(';')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    (!targets.is_empty()).then_some(targets)
}

pub fn debug_decode_layer_enabled(layer_idx: usize) -> bool {
    env_string("RNB_DEBUG_DECODE_LAYER_FILTER")
        .as_deref()
        .map(|raw| layer_matches_spec(raw, layer_idx))
        .unwrap_or(true)
}
