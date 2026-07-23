use std::sync::OnceLock;

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

pub fn expanded_weight_cache_allowed() -> bool {
    env_bool("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", false)
}

fn expanded_env_bool(name: &str, default: bool) -> bool {
    expanded_weight_cache_allowed() && env_bool(name, default)
}

fn expanded_env_force(name: &str) -> bool {
    expanded_weight_cache_allowed()
        && std::env::var(name)
            .ok()
            .map(|value| value.eq_ignore_ascii_case("force"))
            .unwrap_or(false)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

pub fn output_logits_enabled() -> bool {
    env_bool("RNB_CUDA_OUTPUT_LOGITS", true)
}

pub fn output_argmax_enabled() -> bool {
    env_bool("RNB_CUDA_OUTPUT_ARGMAX", false)
}

pub fn q6k_output_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_OUTPUT_WARP8", true)
}

pub fn q6k_fused_argmax_gpu_reduce_enabled(rows: usize) -> bool {
    env_bool("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", rows >= 8192)
}
pub fn q8_0_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_GEMV_WARP4", true)
}

pub fn q8_0_gemv_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_GEMV_WARP8", true)
}

pub fn q4k_gemv_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GEMV_WARP8", true)
}

pub fn q4k_packed_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_PACKED_GEMV_WARP4", false)
}

pub fn q6k_packed_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_PACKED_GEMV_WARP4", false)
}

pub fn q6k_packed_batch_warp4_enabled(blocks_per_row: usize) -> bool {
    blocks_per_row >= 8 && env_bool("RNB_CUDA_Q6_PACKED_BATCH_WARP4", true)
}

pub fn q6k_packed_batch_seq8_enabled(seq_len: usize, blocks_per_row: usize) -> bool {
    let _ = (seq_len, blocks_per_row);
    env_bool("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", false)
}

pub fn q8_0_output_q8dot_argmax_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", false)
}

pub fn dense_expert_graph_enabled() -> bool {
    env_bool("RNB_CUDA_DENSE_EXPERT_GRAPH", false)
}

pub fn cu69_dense_chain_graph_enabled() -> bool {
    env_is_one("RNB_CU69_DENSE_CHAIN_GRAPH")
}

pub fn cu69_dense_chain_graph_trace_enabled() -> bool {
    env_is_one("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE")
}

pub fn cu71_layer_segment_graph_enabled() -> bool {
    env_is_one("RNB_CU71_LAYER_SEGMENT_GRAPH")
}

/// cu74: persistent cooperative decode kernel for Gemma4 E2B.
/// Opt-in only; eager dispatch remains default until token-by-token
/// correctness and ABAB gates pass.
pub fn persistent_decode_enabled() -> bool {
    env_is_one("RNB_CUDA_PERSISTENT_DECODE")
}

pub fn cu71_layer_segment_graph_trace_enabled() -> bool {
    env_is_one("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE")
}

pub fn qwen35_decode_moe_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MOE_GRAPH", true)
}

pub fn qwen35_selected_sparse_compound_graph_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", true)
}

pub fn qwen35_selected_sparse_compound_graph_zero_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", false)
}

pub fn q4k_gemv_batch_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GEMV_BATCH_WARP8", true)
}

pub fn q4k_batch_raw_seq4_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let default = seq_len >= 8 && rows >= 1024 && blocks_per_row >= 4;
    env_bool("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", default)
}

pub fn q4k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q4K_MMQ_TILE32", true)
}

pub fn q6k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q6K_MMQ_TILE32", true)
}
pub fn q8_0_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 128 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q8_0_MMQ_TILE32", true)
}

pub fn q4k_gate_up_batch_seq2_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GATE_UP_BATCH_SEQ2_Q8DOT", true)
}

pub fn q4k_prefill_f32_gemm_enabled() -> bool {
    expanded_env_bool("RNB_CUDA_Q4K_PREFILL_F32_GEMM", false)
}

pub fn qwen35_shared_q4_f32_cache_enabled_for_seq(seq_len: usize) -> bool {
    expanded_weight_cache_allowed()
        && (q4k_prefill_f32_gemm_enabled()
            || (env_bool("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", false)
                && qwen35_short_window_prefill(seq_len)))
}

pub fn qwen35_full_layer_shared_q4_f32_cache_enabled() -> bool {
    expanded_weight_cache_allowed()
        && (q4k_prefill_f32_gemm_enabled()
            || env_bool("RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE", false))
}

pub fn q4_f32_release_after_prefill_enabled() -> bool {
    env_bool("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL", false)
}

pub fn q6k_gemv_batch_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GEMV_BATCH_WARP8", true)
}

pub fn q6k_gemv_batch_seq2_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GEMV_BATCH_SEQ2_WARP8", true)
}

pub fn resident_q4k_touch_hits_enabled() -> bool {
    env_bool("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS", false)
}

pub fn resident_q4k_arena_enabled() -> bool {
    env_bool("RNB_CUDA_RESIDENT_Q4K_ARENA", false)
}

pub fn glm_direct_file_prefill_enabled(auto_enabled: bool) -> bool {
    let Ok(value) = std::env::var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL") else {
        return auto_enabled;
    };
    let value = value.to_ascii_lowercase();
    !matches!(value.as_str(), "0" | "false" | "off" | "no")
}

pub fn glm_direct_file_pipeline_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE", true)
}

pub fn glm_direct_file_expert_stream_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_DIRECT_FILE_EXPERT_STREAM", true)
}

pub fn glm_direct_file_io_uring_enabled() -> bool {
    let Ok(value) = std::env::var("RNB_CUDA_GLM_DIRECT_FILE_IO_URING") else {
        return cfg!(target_os = "linux");
    };
    let value = value.to_ascii_lowercase();
    !matches!(value.as_str(), "0" | "false" | "off" | "no")
}

pub fn glm_direct_file_io_uring_forced() -> bool {
    std::env::var_os("RNB_CUDA_GLM_DIRECT_FILE_IO_URING").is_some()
}

pub fn glm_direct_file_io_uring_queue_depth(request_count: usize) -> usize {
    let default = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    std::env::var("RNB_CUDA_GLM_DIRECT_FILE_IO_URING_DEPTH")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
        .min(request_count.max(1))
}

pub fn glm_expert_grouped_enabled(token_count: usize, slot_count: usize) -> bool {
    token_count > 1 && slot_count > token_count && env_bool("RNB_CUDA_GLM_EXPERT_GROUPED", true)
}

pub fn glm_expert_parallel_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_EXPERT_PARALLEL", false)
}

pub fn glm_expert_parallel_secondary_device(primary_ordinal: i32) -> i32 {
    std::env::var("RNB_CUDA_GLM_EXPERT_PARALLEL_SECONDARY_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .unwrap_or(if primary_ordinal == 0 { 1 } else { 0 })
}

pub fn glm_expert_parallel_primary_slots(slot_count: usize) -> usize {
    if slot_count < 2 {
        return slot_count;
    }
    std::env::var("RNB_CUDA_GLM_EXPERT_PARALLEL_PRIMARY_SLOTS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(slot_count.div_ceil(2))
        .clamp(1, slot_count - 1)
}

pub fn resident_q4k_batch_pinned_staging_enabled(slab_bytes: usize, missing_len: usize) -> bool {
    if let Ok(value) = std::env::var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED") {
        let value = value.to_ascii_lowercase();
        return !matches!(value.as_str(), "0" | "false" | "off" | "no");
    }
    if missing_len < 2 || slab_bytes == 0 {
        return false;
    }
    let min_bytes = std::env::var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2 * 1024 * 1024);
    slab_bytes >= min_bytes
}

pub fn qwen35_decode_resident_batch_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH", false)
}

pub fn qwen35_prefill_hot_resident_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT", false)
}

pub fn qwen35_prefill_hot_resident_min_tokens() -> usize {
    std::env::var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MIN_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(128)
}

pub fn qwen35_prefill_hot_resident_budget_bytes(resident_q4k_limit: usize) -> usize {
    std::env::var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or_else(|| (resident_q4k_limit / 512).clamp(8 * 1024 * 1024, 16 * 1024 * 1024))
}

pub fn mtp_expert_trace_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_EXPERT_TRACE", false)
}

pub fn mtp_expert_hot_resident_enabled() -> bool {
    match std::env::var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => env_bool("RNB_MTP_DEVICE_VERIFY", false),
    }
}

pub fn cubin_modules_enabled() -> bool {
    env_bool("RNB_CUDA_CUBIN_MODULES", true)
}

pub fn mtp_verify_output_q6k_token2_enabled(token_count: usize) -> bool {
    token_count == 2 && env_bool("RNB_CUDA_MTP_VERIFY_OUTPUT_Q6K_TOKEN2", true)
}

pub fn q8_0_gemv_batch_token2_enabled(seq_len: usize) -> bool {
    seq_len == 2 && env_bool("RNB_CUDA_Q8_0_GEMV_BATCH_TOKEN2", true)
}

pub fn mtp_verify_gdn_qkv_warp_enabled(window_tokens: usize) -> bool {
    window_tokens == 1
        || (window_tokens == 2 && env_bool("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2", true))
}

pub fn mtp_verify_router_stable_key_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY", true)
}

pub fn mtp_verify_snapshot_pool_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SNAPSHOT_POOL", true)
}

pub fn mtp_expert_extra_resident_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT", true)
}

pub fn mtp_expert_extra_resident_budget_bytes(resident_q4k_limit: usize) -> usize {
    mtp_expert_extra_resident_budget_bytes_for_layer(resident_q4k_limit, 0)
}

pub fn mtp_expert_extra_resident_budget_bytes_for_layer(
    resident_q4k_limit: usize,
    layer_observations: usize,
) -> usize {
    if !mtp_expert_extra_resident_enabled() {
        return 0;
    }
    let cold_bytes = resident_q4k_limit / 256;
    let warm_bytes = (resident_q4k_limit / 128)
        .min(8 * 1024 * 1024)
        .max(cold_bytes);
    let default_bytes = if layer_observations >= 8 {
        warm_bytes
    } else {
        cold_bytes
    };
    let Some(raw) = std::env::var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB").ok() else {
        return default_bytes;
    };
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return default_bytes;
    }
    raw.parse::<usize>()
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or(default_bytes)
}

pub fn qwen35_decode_q4k_arena_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA", true)
}

pub fn prefill_output_logits_requested() -> bool {
    env_bool("RNB_CUDA_PREFILL_OUTPUT_LOGITS", true)
}

pub fn prefill_gemv_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_GEMV", true)
}

// cu29 Phase 2: hd=128 fused QKV+RoPE+f16-pack path. Llama / Mistral 등
// qk-norm 없는 dense hd128 모델에서 host RoPE round-trip 제거. nsys 진단으로
// D2H sync wait 81% 가 진짜 lever 확정 (cu28). 측정 후 default ON 가능.
pub fn hd128_fused_qkv_rope_enabled() -> bool {
    env_bool("RNB_CUDA_HD128_FUSED_QKV_ROPE", false)
}

pub fn prefill_q4k_f16_gemm_enabled() -> bool {
    expanded_env_bool("RNB_CUDA_Q4K_PREFILL_F16_GEMM", false)
}

pub fn prefill_q4k_f16_qkv_gemm_enabled() -> bool {
    expanded_env_bool(
        "RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM",
        prefill_q4k_f16_gemm_enabled(),
    )
}

pub fn prefill_q4k_f16_o_proj_enabled() -> bool {
    expanded_env_bool(
        "RNB_CUDA_Q4K_PREFILL_F16_O_PROJ",
        prefill_q4k_f16_gemm_enabled(),
    )
}

pub fn prefill_q4k_f16_o_proj_force_enabled() -> bool {
    expanded_env_force("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ")
}

pub fn prefill_delta_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DELTA", true)
}

pub fn prefill_delta_k128_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DELTA_K128_WARP4", true)
}

pub fn prefill_moe_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE", true)
}

pub fn prefill_moe_enabled_for_seq(seq_len: usize) -> bool {
    prefill_moe_enabled() && !qwen35_short_window_prefill(seq_len)
}

pub fn prefill_moe_full_layer_enabled() -> bool {
    // 2026-05-26: Qwen3.6 35B full-layer prefill probe hit CUDA 719 and
    // kernel Xid79 on RTX 3080. Keep the original env quarantined unless the
    // caller also opts into the explicit retry gate and the device-side slot
    // pointer path that survived the controlled RTX 3080 retry.
    env_bool("RNB_CUDA_PREFILL_MOE_FULL_LAYER", false)
        && env_bool("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", false)
        && qwen35_full_layer_device_slot_ptrs_enabled()
}

pub fn prefill_moe_full_layer_min_expert_permille() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(750)
        .clamp(1, 1000)
}

pub fn prefill_moe_weight_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH", false)
}

pub fn prefill_moe_weight_prefetch_pinned_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED", false)
}

pub fn prefill_moe_range_slab_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_RANGE_SLAB", false)
}

pub fn prefill_moe_range_slab_max_gap_experts() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_RANGE_SLAB_MAX_GAP_EXPERTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(3)
}

pub fn prefill_moe_range_slab_max_overhead_permille() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_RANGE_SLAB_MAX_OVERHEAD_PERMILLE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(1250)
        .max(1000)
}

pub fn qwen35_full_layer_device_slot_ptrs_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", false)
}

pub fn moe_layer_cache_enabled() -> bool {
    env_bool("RNB_CUDA_MOE_LAYER_CACHE", false)
}
pub fn mtp_verify_resident_moe_layer_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_MOE_LAYER", true)
}

pub fn mtp_verify_missing_moe_layer_promotion_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_MISSING_MOE_LAYER_PROMOTION", false)
}

pub fn mtp_verify_resident_conv_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_CONV", true)
}

pub fn mtp_verify_resident_attn_kv_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_ATTN_KV", true)
}

pub fn mtp_verify_gdn_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_GDN_GRAPH", true)
}

pub fn mtp_verify_window2_graphs_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS", true)
}

pub fn mtp_verify_attention_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_ATTENTION_GRAPH", true)
}

pub fn mtp_verify_q8_multi_projection_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_Q8_MULTI_PROJECTION", true)
}

pub fn mtp_verify_f32_multi_projection_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_F32_MULTI_PROJECTION", true)
}

pub fn mtp_verify_shared_scale_add_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SHARED_SCALE_ADD", true)
}

pub fn mtp_verify_segment_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SEGMENT_GRAPH", true)
}

pub fn mtp_verify_selected_q8_gate_up_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_Q8_GATE_UP", true)
}

pub fn mtp_verify_selected_gate_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2", true)
}

pub fn mtp_verify_selected_gate_pair2_silu_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU", true)
}

pub fn mtp_verify_selected_down_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2", true)
}

pub fn mtp_verify_selected_pair_map_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP", true)
}

pub fn mtp_verify_selected_gate_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8", true)
}

pub fn mtp_verify_selected_gate_warp_reduce_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE", true)
}

pub fn mtp_verify_selected_down_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_WARP8", true)
}

pub fn q6k_argmax_batched_single_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE", true)
}

pub fn q6k_warp8_unrolled_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_WARP8_UNROLLED", true)
}

pub fn prefill_f32_gemm_allowed(
    quant_supported: bool,
    seq_len: usize,
    rows: usize,
    cols: usize,
) -> bool {
    if !env_bool("RNB_CUDA_PREFILL_F32_GEMM", true) || seq_len <= 1 || !quant_supported {
        return false;
    }
    let min_seq = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(128);
    if seq_len < min_seq {
        return false;
    }
    let max_rows = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(8192);
    let max_cols = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(4096);
    rows <= max_rows && cols <= max_cols
}

pub fn prefill_f32_gemm_trace_enabled() -> bool {
    env_is_one("RNB_CUDA_F32_GEMM_TRACE")
}

pub fn layer_gemv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_LAYER_GEMV", true))
}

pub fn delta_net_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("RNB_CUDA_DELTA_NET") {
        Ok(_) => env_bool("RNB_CUDA_DELTA_NET", true),
        Err(_) => layer_gemv_enabled(),
    })
}

pub fn delta_state_sync_each_step_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_DELTA_STATE_SYNC_EACH_STEP", false))
}

pub fn decode_attention_enabled() -> bool {
    env_is_one("RNB_CUDA_DECODE_ATTN")
}

pub fn decode_attention_kv_cache_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_KV_CACHE", true)
}

pub fn decode_attention_sliding_window_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW", false)
}

pub fn decode_attention_hd512_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD512", true)
}

pub fn decode_attention_hd256_split_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD256_SPLIT", true)
}

pub fn decode_attention_hd256_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or(256)
}

pub fn mtp_verify_attention_hd256_split_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_ATTN_HD256_SPLIT", true)
}

pub fn mtp_verify_attention_hd256_query_tile_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE", true)
}

fn cuda_arch_supports_ampere_mma(arch: &str) -> bool {
    arch.strip_prefix("sm_")
        .and_then(|cc| cc.parse::<u32>().ok())
        .is_some_and(|cc| cc >= 80)
}

pub fn compiled_ampere_mma_supported() -> bool {
    option_env!("RNB_CUDA_COMPILED_ARCH").is_some_and(cuda_arch_supports_ampere_mma)
}

pub fn mtp_verify_attention_hd256_mma_stream_k_enabled() -> bool {
    compiled_ampere_mma_supported() && env_bool("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K", true)
}

pub fn mtp_verify_attention_hd256_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or_else(decode_attention_hd256_split_chunk_size)
}

pub fn decode_attention_hd512_split_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", true)
}

pub fn decode_attention_hd512_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or(512)
}

pub fn gdn_prefill_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL", false)
}

pub fn gdn_prefill_chain_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN", true)
}

pub fn gdn_prefill_chain_device_output_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_DEVICE_OUTPUT", false)
}

pub fn gdn_prefill_chain_moe_input_device_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_INPUT_DEVICE", true)
}

pub fn gdn_prefill_chain_moe_output_device_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE", true)
}

pub fn gdn_prefill_chain_skip_host_projection_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_SKIP_HOST_PROJECTION", true)
}

pub fn qwen35_device_moe_phase_profile_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", false)
}

pub fn qwen35_device_moe_inplace_residual_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", true)
}

pub fn qwen35_q4_gate_up_silu_fused_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", true)
}

pub fn qwen35_q4_gate_up_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", true)
}

pub fn qwen35_q4_gate_up_q8dot_q4_down_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN", true)
}

pub fn qwen35_q4_down_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT", true)
}
pub fn qwen35_q4_gate_up_q8dot_mmq_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", true)
}

pub fn qwen35_q5_down_q8dot_mmq_enabled(token_count: usize) -> bool {
    token_count >= 32 && env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", true)
}

pub fn qwen35_q5_down_q8dot_mmq_group32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32", true)
}

pub fn qwen35_q5_down_q8dot_mmq_group64_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64", true)
}

pub fn qwen35_q4_gate_up_q8dot_mmq_group16_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", true)
}

pub fn qwen35_q4_gate_up_q8dot_mmq_group32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", true)
}

pub fn qwen35_q4_gate_up_q8_handoff_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF", true)
}

pub fn qwen35_selected_base_stream_enabled(token_count: usize) -> bool {
    match std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => token_count >= 32,
    }
}

pub fn qwen35_q4_gate_up_silu_pack4_f32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", true)
}

pub fn prefill_conv_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_CONV", true)
}

pub fn prefill_temp_coalesce_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_COALESCE", false)
}

pub fn prefill_temp_run_coalesce_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE", true)
}

pub fn prefill_temp_pinned_staging_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_PINNED_STAGING", false)
}

pub fn prefill_temp_host_register_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER", false)
}

pub fn prefill_temp_host_register_min_slots() -> usize {
    std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(128)
}

pub fn prefill_temp_host_register_granularity_bytes() -> usize {
    const DEFAULT_GRANULARITY_BYTES: usize = 4096;
    const MIN_GRANULARITY_BYTES: usize = 4096;
    const MAX_GRANULARITY_BYTES: usize = 64 * 1024 * 1024;
    let bytes = std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .and_then(|kb| kb.checked_mul(1024))
        .unwrap_or(DEFAULT_GRANULARITY_BYTES)
        .clamp(MIN_GRANULARITY_BYTES, MAX_GRANULARITY_BYTES);
    bytes.next_power_of_two()
}

pub fn prefill_temp_host_register_min_bytes() -> usize {
    std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(1024 * 1024)
}

pub fn prefill_down_copy_overlap_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP", false)
}

pub fn prefill_moe_sync_before_sparse_enabled() -> bool {
    // The sparse phase can grow shared scratch buffers. Allowing this sync to
    // be disabled risks freeing queued shared-phase buffers before the stream
    // has consumed them.
    let _ = std::env::var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
    true
}

pub fn group4_down_row8_enabled() -> bool {
    env_bool("RNB_CUDA_GROUP4_DOWN_ROW8", false)
}

pub fn qwen35_q4_down_group4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", true)
}

pub fn qwen35_down_token_major_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", false)
}

pub fn qwen35_q6_down_full4_split_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", false)
}

pub fn qwen35_q6_down_full4_fastpath_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", false)
}

pub fn qwen35_q6_down_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", false)
}

pub fn qwen35_q6_down_run_batched_ref_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", false)
}

pub fn qwen35_q6_down_run_batched8_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", false)
}

pub fn qwen35_q6_down_run_tiled4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", false)
}

pub fn qwen35_q6_down_pack4_f32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", true)
}

pub fn qwen35_q6_down_pack4_f32_vec4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32_VEC4", true)
}

pub fn group2_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_GROUP2_DOWN_WARP4", false)
}

pub fn mtp_verify_group2_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_GROUP2_DOWN_WARP4", false)
}

pub fn q6k_group4_down_lowreg_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GROUP4_DOWN_LOWREG", false)
}

pub fn gdn_gated_norm_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_GATED_NORM", true)
}

pub fn gdn_gated_norm_gemm_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_GATED_NORM_GEMM", true)
}

pub fn gdn_gated_norm_gemm_enabled_for_seq(seq_len: usize) -> bool {
    gdn_gated_norm_gemm_enabled() && !qwen35_short_window_prefill(seq_len)
}

pub fn gdn_prefill_gemv_mode() -> Option<String> {
    let raw = std::env::var("RNB_CUDA_GDN_PREFILL_GEMV").unwrap_or_else(|_| "f32".to_string());
    let mode = raw.to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "false" | "off" | "no") {
        None
    } else if matches!(mode.as_str(), "" | "1" | "true" | "on" | "yes") {
        Some("f32".to_string())
    } else {
        Some(mode)
    }
}

pub fn gdn_prefill_gemv_mode_for_seq(seq_len: usize) -> Option<String> {
    if qwen35_short_window_prefill(seq_len) {
        None
    } else {
        gdn_prefill_gemv_mode()
    }
}

pub fn qwen35_short_window_prefill(seq_len: usize) -> bool {
    seq_len > 1 && seq_len <= qwen35_short_window_prefill_max_seq()
}

fn qwen35_short_window_prefill_max_seq() -> usize {
    std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn prefill_flash_attention_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_FLASH_ATTN", true)
}

pub fn prefill_flash_attention_hd512_w256_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", true)
}

pub fn prefill_flash_attention_min_seq(head_dim: usize) -> usize {
    let env_name = match head_dim {
        128 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD128_MIN_SEQ",
        256 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD256_MIN_SEQ",
        512 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD512_MIN_SEQ",
        _ => "RNB_CUDA_PREFILL_FLASH_ATTN_MIN_SEQ",
    };
    std::env::var(env_name)
        .or_else(|_| std::env::var("RNB_CUDA_PREFILL_FLASH_ATTN_MIN_SEQ"))
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(match head_dim {
            128 => 128,
            _ => 1,
        })
}

pub fn moe_route_hist_enabled() -> bool {
    env_is_one("RNB_CUDA_MOE_ROUTE_HIST")
}

pub fn shared_f32_enabled() -> bool {
    std::env::var("RNB_CUDA_SHARED_F32").ok().as_deref() != Some("0")
}

pub fn nemotron_q5_full_layer_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_FULL_LAYER", false)
}

pub fn nemotron_q5_layer_cache_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_LAYER_CACHE", true)
}

pub fn nemotron_q8_shared_q5_sparse_decode_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_Q5_SPARSE_DECODE", true)
}

pub fn nemotron_q8_shared_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_WARP4", true)
}

pub fn nemotron_q8_shared_cublas_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_CUBLAS", true)
}

pub fn nemotron_decode_sparse_copy_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_DECODE_COPY_PREFETCH", false)
}

pub fn nemotron_prefill_sparse_copy_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_COPY_PREFETCH", false)
}

pub fn nemotron_prefill_sparse_input_pinned_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", false)
}

pub fn nemotron_prefill_q8_shared_fused_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_FUSED", false)
}

pub fn nemotron_prefill_q8_shared_sparse_fused_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", false)
}

pub fn nemotron_prefill_group4_enabled(token_count: usize, slots: usize) -> bool {
    if !env_bool("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", true) {
        return false;
    }
    let min_tokens = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(16);
    let min_slots = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(64);
    token_count >= min_tokens && slots >= min_slots
}

pub fn prefill_q8_0_batch_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_Q8_0_BATCH", false)
}

pub fn nemotron_q5_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_DOWN_WARP4", false)
}

pub fn nemotron_q8_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_DOWN_WARP4", false)
}

pub fn qwen_moe_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_QWEN_MOE_CUDA_GATE_UP", true))
}

pub fn qwen_moe_batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_BATCH_MOE", qwen_moe_gate_up_enabled()))
}

pub fn qwen_moe_device_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_DEVICE_DECODE", true))
}

pub fn cu65_device_qkv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU65_DEVICE_QKV"))
}

pub fn cu68_layer_graph_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU68_LAYER_GRAPH"))
}

pub fn cu63_device_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU63_DEVICE_DECODE"))
}

pub fn cu63_sync_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU63_SYNC_DIAG"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cuda_runtime_policy() {
        unsafe {
            std::env::remove_var("RNB_CUDA_OUTPUT_LOGITS");
            std::env::remove_var("RNB_CUDA_OUTPUT_ARGMAX");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_INPUT_DEVICE");
            std::env::remove_var("RNB_CUDA_PREFILL_CONV");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_PINNED_STAGING");
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
            std::env::remove_var("RNB_CUDA_GROUP2_DOWN_WARP4");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GROUP2_DOWN_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_GROUP4_DOWN_LOWREG");
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_WARP8");
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
            std::env::remove_var("RNB_CUDA_MOE_LAYER_CACHE");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_RESIDENT_MOE_LAYER");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_MISSING_MOE_LAYER_PROMOTION");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
            std::env::remove_var("RNB_CUDA_PREFILL_OUTPUT_LOGITS");
            std::env::remove_var("RNB_CUDA_DELTA_STATE_SYNC_EACH_STEP");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        }
        assert!(output_logits_enabled());
        assert!(prefill_output_logits_requested());
        assert!(!output_argmax_enabled());
        assert!(q6k_output_warp8_enabled());
        assert!(q8_0_gemv_warp4_enabled());
        assert!(q8_0_gemv_warp8_enabled());
        assert!(q4k_gemv_warp8_enabled());
        assert!(!q4k_packed_gemv_warp4_enabled());
        assert!(!q6k_packed_gemv_warp4_enabled());
        assert!(qwen35_decode_moe_graph_enabled());
        assert!(qwen35_selected_sparse_compound_graph_enabled());
        assert!(!q8_0_output_q8dot_argmax_enabled());
        assert!(q4k_gemv_batch_warp8_enabled());
        assert!(q4k_batch_raw_seq4_enabled(1115, 2560, 10));
        assert!(!q4k_batch_raw_seq4_enabled(4, 2560, 10));
        assert!(q6k_packed_batch_warp4_enabled(14));
        assert!(!q6k_packed_batch_warp4_enabled(7));
        assert!(q6k_gemv_batch_warp8_enabled());
        assert!(!resident_q4k_touch_hits_enabled());
        assert!(!resident_q4k_arena_enabled());
        assert!(!resident_q4k_batch_pinned_staging_enabled(1024 * 1024, 2));
        assert!(resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            2
        ));
        assert!(qwen35_decode_q4k_arena_enabled());
        assert!(!gdn_prefill_enabled());
        assert!(gdn_prefill_chain_enabled());
        assert!(gdn_prefill_chain_moe_input_device_enabled());
        assert!(gdn_prefill_chain_moe_output_device_enabled());
        assert!(gdn_prefill_chain_skip_host_projection_enabled());
        assert!(prefill_conv_enabled());
        assert!(!prefill_moe_full_layer_enabled());
        assert!(!moe_layer_cache_enabled());
        assert!(mtp_verify_resident_moe_layer_enabled());
        assert!(!mtp_verify_missing_moe_layer_promotion_enabled());
        assert!(!mtp_expert_trace_enabled());
        assert!(!mtp_expert_hot_resident_enabled());
        assert!(mtp_expert_extra_resident_enabled());
        assert!(!prefill_temp_coalesce_enabled());
        assert!(prefill_temp_run_coalesce_enabled());
        assert!(!prefill_temp_pinned_staging_enabled());
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_down_copy_overlap_enabled());
        assert!(prefill_delta_k128_warp4_enabled());
        assert!(!delta_state_sync_each_step_enabled());
        assert!(!group4_down_row8_enabled());
        assert!(!group2_down_warp4_enabled());
        assert!(!mtp_verify_group2_down_warp4_enabled());
        assert!(!q6k_group4_down_lowreg_enabled());
        assert!(!nemotron_prefill_group4_enabled(15, 64));
        assert!(!nemotron_prefill_group4_enabled(16, 63));
        assert!(nemotron_prefill_group4_enabled(16, 64));
    }

    #[test]
    fn nemotron_prefill_group4_policy_is_model_scoped() {
        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
        }
        assert!(!nemotron_prefill_group4_enabled(1, 6));
        assert!(!nemotron_prefill_group4_enabled(16, 32));
        assert!(nemotron_prefill_group4_enabled(16, 64));

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", "0");
        }
        assert!(!nemotron_prefill_group4_enabled(128, 768));

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", "1");
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS", "32");
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS", "128");
        }
        assert!(!nemotron_prefill_group4_enabled(16, 768));
        assert!(!nemotron_prefill_group4_enabled(128, 64));
        assert!(nemotron_prefill_group4_enabled(128, 768));

        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
        }
    }

    #[test]
    fn nemotron_sparse_input_pinned_staging_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED");
        }
        assert!(!nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", "1");
        }
        assert!(nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", "0");
        }
        assert!(!nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED");
        }
    }

    #[test]
    fn qwen35_device_moe_phase_profile_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE");
        }
        assert!(!qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", "1");
        }
        assert!(qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", "0");
        }
        assert!(!qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE");
        }
    }

    #[test]
    fn qwen35_device_moe_inplace_residual_is_default_on_with_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL");
        }
        assert!(qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", "0");
        }
        assert!(!qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", "1");
        }
        assert!(qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL");
        }
    }

    #[test]
    fn prefill_f32_gemm_applies_shape_thresholds() {
        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM", "1");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ", "8");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS", "128");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS", "256");
        }
        assert!(prefill_f32_gemm_allowed(true, 8, 128, 256));
        assert!(!prefill_f32_gemm_allowed(true, 7, 128, 256));
        assert!(!prefill_f32_gemm_allowed(false, 8, 128, 256));
        assert!(!prefill_f32_gemm_allowed(true, 8, 129, 256));
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS");
        }
    }

    #[test]
    fn gdn_prefill_gemv_defaults_to_f32_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
        }
        assert_eq!(gdn_prefill_gemv_mode().as_deref(), Some("f32"));

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_GEMV", "0");
        }
        assert_eq!(gdn_prefill_gemv_mode(), None);

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_GEMV", "q");
        }
        assert_eq!(gdn_prefill_gemv_mode().as_deref(), Some("q"));

        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
        }
    }

    #[test]
    fn qwen35_short_window_prefill_policy_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
            std::env::remove_var("RNB_CUDA_GDN_GATED_NORM_GEMM");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
            std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL");
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
        }

        assert_eq!(gdn_prefill_gemv_mode_for_seq(1).as_deref(), Some("f32"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(1));
        assert!(prefill_moe_enabled_for_seq(1));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(1));

        assert_eq!(gdn_prefill_gemv_mode_for_seq(2).as_deref(), Some("f32"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(2));
        assert!(prefill_moe_enabled_for_seq(2));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(2));

        unsafe {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
            std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
            std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "16");
        }

        assert_eq!(gdn_prefill_gemv_mode_for_seq(16), None);
        assert!(!gdn_gated_norm_gemm_enabled_for_seq(16));
        assert!(!prefill_moe_enabled_for_seq(16));
        assert!(qwen35_shared_q4_f32_cache_enabled_for_seq(16));

        assert_eq!(gdn_prefill_gemv_mode_for_seq(17).as_deref(), Some("f32"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(17));
        assert!(prefill_moe_enabled_for_seq(17));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(17));
        assert!(!q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "1");
        }
        assert!(qwen35_shared_q4_f32_cache_enabled_for_seq(17));
        assert!(!q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL", "1");
        }
        assert!(q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL");
        }
    }

    #[test]
    fn prefill_moe_full_layer_threshold_is_configurable_permille() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE");
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 750);

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE", "950");
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 950);

        unsafe {
            std::env::set_var(
                "RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE",
                "5000",
            );
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 1000);

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE");
        }
    }

    #[test]
    fn prefill_moe_weight_prefetch_pinned_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED");
        }
        assert!(!prefill_moe_weight_prefetch_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED", "1");
        }
        assert!(prefill_moe_weight_prefetch_pinned_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED");
        }
    }

    #[test]
    fn prefill_temp_coalesce_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
        }
        assert!(!prefill_temp_coalesce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_COALESCE", "1");
        }
        assert!(prefill_temp_coalesce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
        }
    }

    #[test]
    fn prefill_temp_run_coalesce_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
        }
        assert!(prefill_temp_run_coalesce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE", "0");
        }
        assert!(!prefill_temp_run_coalesce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
        }
    }

    #[test]
    fn glm_direct_file_prefill_uses_constrained_default_and_allows_override() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL");
        }
        assert!(!glm_direct_file_prefill_enabled(false));
        assert!(glm_direct_file_prefill_enabled(true));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL", "0");
        }
        assert!(!glm_direct_file_prefill_enabled(true));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL", "1");
        }
        assert!(glm_direct_file_prefill_enabled(false));

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL");
        }
    }

    #[test]
    fn glm_direct_file_pipeline_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE");
        }
        assert!(glm_direct_file_pipeline_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE", "0");
        }
        assert!(!glm_direct_file_pipeline_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE");
        }
    }

    #[test]
    fn glm_expert_grouped_defaults_on_for_batch_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_EXPERT_GROUPED");
        }
        assert!(glm_expert_grouped_enabled(61, 488));
        assert!(!glm_expert_grouped_enabled(1, 8));
        assert!(!glm_expert_grouped_enabled(61, 61));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_EXPERT_GROUPED", "0");
        }
        assert!(!glm_expert_grouped_enabled(61, 488));

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_EXPERT_GROUPED");
        }
    }

    #[test]
    fn prefill_temp_host_register_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES");
        }
        assert!(!prefill_temp_host_register_enabled());
        assert_eq!(prefill_temp_host_register_min_slots(), 128);
        assert_eq!(prefill_temp_host_register_granularity_bytes(), 4096);
        assert_eq!(prefill_temp_host_register_min_bytes(), 1024 * 1024);

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER", "1");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS", "64");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB", "64");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES", "1048576");
        }
        assert!(prefill_temp_host_register_enabled());
        assert_eq!(prefill_temp_host_register_min_slots(), 64);
        assert_eq!(prefill_temp_host_register_granularity_bytes(), 64 * 1024);
        assert_eq!(prefill_temp_host_register_min_bytes(), 1024 * 1024);

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES");
        }
    }

    #[test]
    fn mtp_verify_window2_graphs_default_on_and_allow_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS");
        }
        assert!(mtp_verify_window2_graphs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS", "0");
        }
        assert!(!mtp_verify_window2_graphs_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS");
        }
    }

    #[test]
    fn mtp_verify_gdn_qkv_warp_defaults_on_for_window2_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2");
        }
        assert!(mtp_verify_gdn_qkv_warp_enabled(1));
        assert!(mtp_verify_gdn_qkv_warp_enabled(2));
        assert!(!mtp_verify_gdn_qkv_warp_enabled(3));

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2", "0");
        }
        assert!(mtp_verify_gdn_qkv_warp_enabled(1));
        assert!(!mtp_verify_gdn_qkv_warp_enabled(2));

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2");
        }
    }

    #[test]
    fn mtp_verify_router_stable_key_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY");
        }
        assert!(mtp_verify_router_stable_key_enabled());
        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY", "0");
        }
        assert!(!mtp_verify_router_stable_key_enabled());
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_pair2_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2");
        }
        assert!(mtp_verify_selected_gate_pair2_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2", "0");
        }
        assert!(!mtp_verify_selected_gate_pair2_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_pair2_silu_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU");
        }
        assert!(mtp_verify_selected_gate_pair2_silu_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU", "0");
        }
        assert!(!mtp_verify_selected_gate_pair2_silu_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU");
        }
    }

    #[test]
    fn mtp_verify_selected_down_pair2_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2");
        }
        assert!(mtp_verify_selected_down_pair2_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2", "0");
        }
        assert!(!mtp_verify_selected_down_pair2_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2");
        }
    }

    #[test]
    fn mtp_verify_selected_pair_map_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP");
        }
        assert!(mtp_verify_selected_pair_map_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP", "0");
        }
        assert!(!mtp_verify_selected_pair_map_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8");
        }
        assert!(mtp_verify_selected_gate_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8", "0");
        }
        assert!(!mtp_verify_selected_gate_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_warp_reduce_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE");
        }
        assert!(mtp_verify_selected_gate_warp_reduce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE", "0");
        }
        assert!(!mtp_verify_selected_gate_warp_reduce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE");
        }
    }

    #[test]
    fn q6k_argmax_batched_single_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE");
        }
        assert!(q6k_argmax_batched_single_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE", "0");
        }
        assert!(!q6k_argmax_batched_single_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE");
        }
    }

    #[test]
    fn cubin_modules_default_on_and_allow_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_CUBIN_MODULES");
        }
        assert!(cubin_modules_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_CUBIN_MODULES", "0");
        }
        assert!(!cubin_modules_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_CUBIN_MODULES");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_silu_fused_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED");
        }
        assert!(qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "1");
        }
        assert!(qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "0");
        }
        assert!(!qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_q8dot_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32");
        }
        assert!(qwen35_q4_gate_up_q8dot_enabled());
        assert!(qwen35_q4_gate_up_q8dot_q4_down_enabled());
        assert!(qwen35_q4_down_q8dot_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_group16_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_group32_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "0");
        }
        assert!(!qwen35_q4_gate_up_q8dot_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_q4_down_enabled());
        assert!(!qwen35_q4_down_q8dot_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_group16_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_group32_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32");
        }
    }

    #[test]
    fn qwen35_selected_base_stream_defaults_on_for_long_prefill_and_allows_opt_out() {
        let key = "RNB_CUDA_QWEN35_SELECTED_BASE_STREAM";
        unsafe {
            std::env::remove_var(key);
        }
        assert!(!qwen35_selected_base_stream_enabled(31));
        assert!(qwen35_selected_base_stream_enabled(32));
        unsafe {
            std::env::set_var(key, "0");
        }
        assert!(!qwen35_selected_base_stream_enabled(32));
        unsafe {
            std::env::set_var(key, "1");
        }
        assert!(qwen35_selected_base_stream_enabled(31));
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn decode_attention_kv_cache_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_KV_CACHE");
        }
        assert!(decode_attention_kv_cache_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_KV_CACHE", "0");
        }
        assert!(!decode_attention_kv_cache_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_KV_CACHE");
        }
    }

    #[test]
    fn decode_attention_sliding_window_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW");
        }
        assert!(!decode_attention_sliding_window_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW", "1");
        }
        assert!(decode_attention_sliding_window_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW");
        }
    }

    #[test]
    fn decode_attention_hd512_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512");
        }
        assert!(decode_attention_hd512_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512", "0");
        }
        assert!(!decode_attention_hd512_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512");
        }
    }

    #[test]
    fn decode_attention_hd256_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
        }
        assert!(decode_attention_hd256_split_enabled());
        assert_eq!(decode_attention_hd256_split_chunk_size(), 256);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT", "0");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "128");
        }
        assert!(!decode_attention_hd256_split_enabled());
        assert_eq!(decode_attention_hd256_split_chunk_size(), 128);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "7");
        }
        assert_eq!(decode_attention_hd256_split_chunk_size(), 256);

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
        }
    }

    #[test]
    fn mtp_verify_attention_hd256_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K");
        }
        assert!(mtp_verify_attention_hd256_split_enabled());
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 256);
        assert!(mtp_verify_attention_hd256_query_tile_enabled());
        assert_eq!(
            mtp_verify_attention_hd256_mma_stream_k_enabled(),
            compiled_ampere_mma_supported()
        );
        assert!(!cuda_arch_supports_ampere_mma("sm_75"));
        assert!(cuda_arch_supports_ampere_mma("sm_80"));
        assert!(cuda_arch_supports_ampere_mma("sm_86"));

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT", "0");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK", "128");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE", "0");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K", "0");
        }
        assert!(!mtp_verify_attention_hd256_split_enabled());
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 128);
        assert!(!mtp_verify_attention_hd256_query_tile_enabled());
        assert!(!mtp_verify_attention_hd256_mma_stream_k_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK", "7");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "512");
        }
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 512);

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K");
        }
    }

    #[test]
    fn decode_attention_hd512_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK");
        }
        assert!(decode_attention_hd512_split_enabled());
        assert_eq!(decode_attention_hd512_split_chunk_size(), 512);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", "0");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK", "128");
        }
        assert!(!decode_attention_hd512_split_enabled());
        assert_eq!(decode_attention_hd512_split_chunk_size(), 128);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK", "7");
        }
        assert_eq!(decode_attention_hd512_split_chunk_size(), 512);

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK");
        }
    }

    #[test]
    fn prefill_flash_hd512_w256_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
        }
        assert!(prefill_flash_attention_hd512_w256_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", "0");
        }
        assert!(!prefill_flash_attention_hd512_w256_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
        }
    }

    #[test]
    fn prefill_down_copy_overlap_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
        }
        assert!(!prefill_down_copy_overlap_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP", "1");
        }
        assert!(prefill_down_copy_overlap_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
        }
    }

    #[test]
    fn prefill_moe_full_layer_stays_quarantined_after_xid79() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY");
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER", "1");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", "1");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");
        }
        assert!(prefill_moe_full_layer_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY");
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn qwen35_full_layer_device_slot_ptrs_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
        assert!(!qwen35_full_layer_device_slot_ptrs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");
        }
        assert!(qwen35_full_layer_device_slot_ptrs_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn prefill_delta_k128_warp4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
        }
        assert!(prefill_delta_k128_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4", "0");
        }
        assert!(!prefill_delta_k128_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
        }
    }

    #[test]
    fn group4_down_row8_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
        }
        assert!(!group4_down_row8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_GROUP4_DOWN_ROW8", "1");
        }
        assert!(group4_down_row8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
        }
    }

    #[test]
    fn qwen35_q4_down_group4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4");
        }
        assert!(qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "1");
        }
        assert!(qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "0");
        }
        assert!(!qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4");
        }
    }

    #[test]
    fn q6k_output_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
        }
        assert!(q6k_output_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_OUTPUT_WARP8", "0");
        }
        assert!(!q6k_output_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
        }
    }

    #[test]
    fn q6k_argmax_gpu_reduce_defaults_on_for_large_outputs() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE");
        }
        assert!(!q6k_fused_argmax_gpu_reduce_enabled(4096));
        assert!(q6k_fused_argmax_gpu_reduce_enabled(8192));
        assert!(q6k_fused_argmax_gpu_reduce_enabled(248_320));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", "0");
        }
        assert!(!q6k_fused_argmax_gpu_reduce_enabled(248_320));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", "1");
        }
        assert!(q6k_fused_argmax_gpu_reduce_enabled(4096));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE");
        }
    }

    #[test]
    fn packed_long_gemv_warp4_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
        }
        assert!(!q4k_packed_gemv_warp4_enabled());
        assert!(!q6k_packed_gemv_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4", "1");
            std::env::set_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4", "1");
        }
        assert!(q4k_packed_gemv_warp4_enabled());
        assert!(q6k_packed_gemv_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
        }
    }

    #[test]
    fn q4k_batch_raw_seq4_defaults_on_for_long_prefill_and_allows_overrides() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
        }
        assert!(q4k_batch_raw_seq4_enabled(1115, 2560, 10));
        assert!(!q4k_batch_raw_seq4_enabled(4, 2560, 10));
        assert!(!q4k_batch_raw_seq4_enabled(1115, 512, 10));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "0");
        }
        assert!(!q4k_batch_raw_seq4_enabled(1115, 2560, 10));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "1");
        }
        assert!(q4k_batch_raw_seq4_enabled(4, 2560, 10));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
        }
    }

    #[test]
    fn q4k_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE32");
        }
        assert!(q4k_mmq_tile32_enabled(1115, 2560, 10));
        assert!(!q4k_mmq_tile32_enabled(31, 2560, 10));
        assert!(!q4k_mmq_tile32_enabled(1115, 512, 10));
        assert!(!q4k_mmq_tile32_enabled(1115, 2560, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_MMQ_TILE32", "0");
        }
        assert!(!q4k_mmq_tile32_enabled(1115, 2560, 10));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE32");
        }
    }

    #[test]
    fn q6k_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE32");
        }
        assert!(q6k_mmq_tile32_enabled(1115, 8192, 8));
        assert!(!q6k_mmq_tile32_enabled(31, 8192, 8));
        assert!(!q6k_mmq_tile32_enabled(1115, 512, 8));
        assert!(!q6k_mmq_tile32_enabled(1115, 8192, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_MMQ_TILE32", "0");
        }
        assert!(!q6k_mmq_tile32_enabled(1115, 8192, 8));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE32");
        }
    }

    #[test]
    fn q8_0_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_MMQ_TILE32");
        }
        assert!(q8_0_mmq_tile32_enabled(1139, 2048, 64));
        assert!(!q8_0_mmq_tile32_enabled(31, 2048, 64));
        assert!(!q8_0_mmq_tile32_enabled(1139, 127, 64));
        assert!(!q8_0_mmq_tile32_enabled(1139, 2048, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_MMQ_TILE32", "0");
        }
        assert!(!q8_0_mmq_tile32_enabled(1139, 2048, 64));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_MMQ_TILE32");
        }
    }

    #[test]
    fn qwen35_q5_down_q8dot_mmq_is_prefill_only_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64");
        }
        assert!(qwen35_q5_down_q8dot_mmq_enabled(1139));
        assert!(qwen35_q5_down_q8dot_mmq_enabled(32));
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(31));
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(2));
        assert!(qwen35_q5_down_q8dot_mmq_group32_enabled());
        assert!(qwen35_q5_down_q8dot_mmq_group64_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64", "0");
        }
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(1139));
        assert!(!qwen35_q5_down_q8dot_mmq_group32_enabled());
        assert!(!qwen35_q5_down_q8dot_mmq_group64_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_q8_handoff_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF");
        }
        assert!(qwen35_q4_gate_up_q8_handoff_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF", "0");
        }
        assert!(!qwen35_q4_gate_up_q8_handoff_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF");
        }
    }

    #[test]
    fn q6k_packed_batch_warp4_defaults_on_for_mid_blocks_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        }
        assert!(q6k_packed_batch_warp4_enabled(14));
        assert!(q6k_packed_batch_warp4_enabled(8));
        assert!(!q6k_packed_batch_warp4_enabled(7));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4", "0");
        }
        assert!(!q6k_packed_batch_warp4_enabled(14));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        }
    }

    #[test]
    fn q6k_packed_batch_seq8_stays_opt_in_after_abab_regression() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8");
        }
        assert!(!q6k_packed_batch_seq8_enabled(8, 8));
        assert!(!q6k_packed_batch_seq8_enabled(7, 8));
        assert!(!q6k_packed_batch_seq8_enabled(8, 7));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", "1");
        }
        assert!(q6k_packed_batch_seq8_enabled(1115, 14));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8");
        }
    }

    #[test]
    fn q4k_prefill_f16_split_flags_follow_global_and_allow_overrides() {
        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
        }
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
        }
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
        }
        assert!(prefill_q4k_f16_gemm_enabled());
        assert!(prefill_q4k_f16_qkv_gemm_enabled());
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "0");
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "1");
        }
        assert!(prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "force");
        }
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
        }
    }

    #[test]
    fn q8_output_q8dot_argmax_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX");
        }
        assert!(!q8_0_output_q8dot_argmax_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", "1");
        }
        assert!(q8_0_output_q8dot_argmax_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX");
        }
    }

    #[test]
    fn dense_expert_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DENSE_EXPERT_GRAPH");
        }
        assert!(!dense_expert_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DENSE_EXPERT_GRAPH", "1");
        }
        assert!(dense_expert_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DENSE_EXPERT_GRAPH");
        }
    }

    #[test]
    fn cu69_dense_chain_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH");
        }
        assert!(!cu69_dense_chain_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
        }
        assert!(cu69_dense_chain_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH");
        }
    }

    #[test]
    fn cu69_dense_chain_graph_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE");
        }
        assert!(!cu69_dense_chain_graph_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE", "1");
        }
        assert!(cu69_dense_chain_graph_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE");
        }
    }

    #[test]
    fn cu71_layer_segment_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH");
        }
        assert!(!cu71_layer_segment_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CU71_LAYER_SEGMENT_GRAPH", "1");
        }
        assert!(cu71_layer_segment_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH");
        }
    }

    #[test]
    fn cu71_layer_segment_graph_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE");
        }
        assert!(!cu71_layer_segment_graph_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE", "1");
        }
        assert!(cu71_layer_segment_graph_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE");
        }
    }

    #[test]
    fn qwen35_decode_moe_graph_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
        }
        assert!(qwen35_decode_moe_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MOE_GRAPH", "0");
        }
        assert!(!qwen35_decode_moe_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
        }
    }

    #[test]
    fn qwen35_selected_sparse_compound_graph_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
        }
        assert!(qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", "0");
        }
        assert!(!qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", "1");
        }
        assert!(qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
        }
    }

    #[test]
    fn qwen35_selected_sparse_compound_graph_zero_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO");
        }
        assert!(!qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", "0");
        }
        assert!(!qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", "1");
        }
        assert!(qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO");
        }
    }

    #[test]
    fn q8_0_gemv_warp4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
        }
        assert!(q8_0_gemv_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_GEMV_WARP4", "0");
        }
        assert!(!q8_0_gemv_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
        }
    }

    #[test]
    fn q8_0_gemv_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
        }
        assert!(q8_0_gemv_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_GEMV_WARP8", "0");
        }
        assert!(!q8_0_gemv_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
        }
    }

    #[test]
    fn q4k_gemv_batch_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
        }
        assert!(q4k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8", "0");
        }
        assert!(!q4k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
        }
    }

    #[test]
    fn q6k_gemv_batch_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
        }
        assert!(q6k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8", "0");
        }
        assert!(!q6k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
        }
    }

    #[test]
    fn resident_q4k_touch_hits_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
        }
        assert!(!resident_q4k_touch_hits_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS", "1");
        }
        assert!(resident_q4k_touch_hits_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
        }
    }

    #[test]
    fn resident_q4k_arena_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
        }
        assert!(!resident_q4k_arena_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_ARENA", "1");
        }
        assert!(resident_q4k_arena_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
        }
    }

    #[test]
    fn resident_q4k_batch_pinned_staging_scales_with_batch_size() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
        }
        assert!(!resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            1
        ));
        assert!(!resident_q4k_batch_pinned_staging_enabled(1024 * 1024, 2));
        assert!(resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            2
        ));

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED", "0");
        }
        assert!(!resident_q4k_batch_pinned_staging_enabled(
            16 * 1024 * 1024,
            8
        ));

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED", "1");
        }
        assert!(resident_q4k_batch_pinned_staging_enabled(1, 1));

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES", "4096");
        }
        assert!(resident_q4k_batch_pinned_staging_enabled(4096, 2));

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
        }
    }

    #[test]
    fn qwen35_decode_q4k_arena_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
        }
        assert!(qwen35_decode_q4k_arena_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA", "0");
        }
        assert!(!qwen35_decode_q4k_arena_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
        }
    }

    #[test]
    fn qwen35_decode_resident_batch_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH");
        }
        assert!(!qwen35_decode_resident_batch_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH", "1");
        }
        assert!(qwen35_decode_resident_batch_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH");
        }
    }

    #[test]
    fn qwen35_prefill_hot_resident_defaults_off_and_caps_auto_budget() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB");
        }
        assert!(!qwen35_prefill_hot_resident_enabled());
        assert_eq!(
            qwen35_prefill_hot_resident_budget_bytes(8 * 1024 * 1024 * 1024),
            16 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT", "1");
            std::env::set_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB", "6");
        }
        assert!(qwen35_prefill_hot_resident_enabled());
        assert_eq!(
            qwen35_prefill_hot_resident_budget_bytes(8 * 1024 * 1024 * 1024),
            6 * 1024 * 1024
        );

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB");
        }
    }

    #[test]
    fn prefill_moe_sync_before_sparse_stays_on_after_xid79() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
        }
        assert!(prefill_moe_sync_before_sparse_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE", "0");
        }
        assert!(prefill_moe_sync_before_sparse_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
        }
    }

    #[test]
    fn mtp_expert_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
        }
        assert!(!mtp_expert_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_TRACE", "1");
        }
        assert!(mtp_expert_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
        }
    }

    #[test]
    fn mtp_expert_hot_resident_follows_device_verify_and_allows_override() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
        assert!(!mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
        }
        assert!(mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT", "0");
        }
        assert!(!mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT", "1");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
        assert!(mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
    }

    #[test]
    fn mtp_expert_extra_resident_budget_scales_with_cache_limit() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
        }
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024),
            1024 * 1024
        );
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes_for_layer(1024 * 1024 * 1024, 0),
            4 * 1024 * 1024
        );
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes_for_layer(1024 * 1024 * 1024, 8),
            8 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB", "32");
        }
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024),
            32 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT", "0");
        }
        assert_eq!(mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024), 0);

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
        }
    }
}
