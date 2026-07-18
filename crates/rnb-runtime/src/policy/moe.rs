use std::path::PathBuf;

use super::base::{env_f32, env_flag, env_string, env_terms, env_truthy_override, env_usize};

pub fn moe_section_decode_enabled() -> bool {
    env_truthy_override("RNB_MOE_DECODE").unwrap_or(false)
}

pub fn moe_section_decode_disabled() -> bool {
    !moe_section_decode_enabled()
}

pub fn moe_profile_enabled() -> bool {
    env_flag("RNB_MOE_PROFILE")
}

pub fn qwen35_cpu_expert_major_enabled() -> bool {
    !env_flag("RNB_QWEN35_CPU_EXPERT_MAJOR_OFF")
}

pub fn moe_profile_by_layer_enabled() -> bool {
    env_flag("RNB_MOE_PROFILE_BY_LAYER")
}

pub fn moe_predictor_trace_top_n() -> Option<usize> {
    env_usize("RNB_MOE_PREDICTOR_TRACE_TOP_N")
}

pub fn moe_predictor_trace_top_n_limit() -> usize {
    moe_predictor_trace_top_n()
        .filter(|&value| value > 0)
        .unwrap_or(16)
}

pub fn moe_route_trace_file() -> Option<PathBuf> {
    std::env::var_os("RNB_MOE_ROUTE_TRACE_FILE").map(PathBuf::from)
}

pub fn moe_route_trace_enabled() -> bool {
    moe_route_trace_file().is_some()
}

pub fn moe_predictor_trace_file() -> Option<PathBuf> {
    std::env::var_os("RNB_MOE_PREDICTOR_TRACE_FILE").map(PathBuf::from)
}

pub fn moe_predictor_trace_enabled() -> bool {
    moe_predictor_trace_file().is_some()
}

/// Diagnostic cumulative mass threshold over Hy3's pretrained top-k experts.
pub fn moe_adaptive_top_p() -> Option<f32> {
    env_f32("RNB_MOE_ADAPTIVE_TOP_P")
        .filter(|value| value.is_finite() && *value > 0.0 && *value < 1.0)
}

pub fn cuda_q2k_q3k_sparse_moe_enabled(auto_enabled: bool) -> bool {
    env_truthy_override("RNB_CUDA_Q2K_Q3K_SPARSE_MOE").unwrap_or(auto_enabled)
}

pub fn cuda_q2k_q3k_mixed_resident_cpu_enabled(auto_enabled: bool) -> bool {
    env_truthy_override("RNB_CUDA_Q2Q3_MIXED_RESIDENT_CPU").unwrap_or(auto_enabled)
}

pub fn cuda_cache_trace_enabled() -> bool {
    env_string("RNB_CUDA_CACHE_TRACE").as_deref() == Some("1")
}

pub fn cuda_decode_moe_combined_enabled() -> bool {
    env_string("RNB_CUDA_DECODE_MOE_COMBINED").as_deref() == Some("1")
}

pub fn glm_dsa_batch_prefill_enabled(
    seq_len: usize,
    expert_count: usize,
    expert_used_count: usize,
) -> bool {
    let auto_enabled = seq_len > 1
        && expert_count > 0
        && expert_used_count > 0
        && seq_len.saturating_mul(expert_used_count) >= expert_count;
    env_truthy_override("RNB_GLM_DSA_BATCH_PREFILL").unwrap_or(auto_enabled)
}

const MOE_MIXED_PRECISION_ENV: &str = "RNB_HOBBIT";
const MOE_MIXED_PRECISION_T1_ENV: &str = "RNB_HOBBIT_T1";
const MOE_MIXED_PRECISION_T2_ENV: &str = "RNB_HOBBIT_T2";
const MOE_MIXED_PRECISION_LOW_PATH_ENV: &str = "RNB_HOBBIT_LOW_PATH";

pub fn moe_mixed_precision_requested() -> bool {
    env_flag(MOE_MIXED_PRECISION_ENV)
}

pub fn moe_mixed_precision_enabled(has_shadow: bool) -> bool {
    moe_mixed_precision_requested() && has_shadow
}

pub fn moe_mixed_precision_thresholds() -> (f32, f32) {
    let t1 = env_f32(MOE_MIXED_PRECISION_T1_ENV).unwrap_or(0.6);
    let t2 = env_f32(MOE_MIXED_PRECISION_T2_ENV).unwrap_or(0.9);
    (t1, t2)
}

pub fn moe_mixed_precision_low_path() -> Option<String> {
    env_string(MOE_MIXED_PRECISION_LOW_PATH_ENV)
}

pub fn packed_only_match_terms() -> Vec<String> {
    env_terms("RNB_PACKED_ONLY_MATCH")
}

pub fn packed_skip_match_terms() -> Vec<String> {
    env_terms("RNB_PACKED_SKIP_MATCH")
}

pub fn packed_skip_q4k_enabled() -> bool {
    env_flag("RNB_PACKED_SKIP_Q4K")
}

pub fn packed_skip_q5k_enabled() -> bool {
    env_flag("RNB_PACKED_SKIP_Q5K")
}

pub fn packed_skip_q6k_enabled() -> bool {
    env_flag("RNB_PACKED_SKIP_Q6K")
}
