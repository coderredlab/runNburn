use super::base::{
    env_flag, env_flag_default_on, env_present_os, env_string, env_truthy_override, env_usize,
};

fn env_truthy_nonempty_override(name: &str) -> Option<bool> {
    env_string(name).map(|value| {
        let value = value.to_ascii_lowercase();
        !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
    })
}

pub fn profiling_enabled() -> bool {
    env_flag("RNB_PROFILE")
}

pub fn spec_profile_enabled() -> bool {
    env_flag("RNB_SPEC_PROFILE")
}

pub fn profiling_level() -> u32 {
    std::env::var("RNB_PROFILE")
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok().or(Some(1)))
        .unwrap_or(0)
}

pub fn gemv_profile_enabled() -> bool {
    env_flag("RNB_GEMV_PROFILE")
}

pub fn prefill_layer_profile_enabled() -> bool {
    env_flag("RNB_PREFILL_LAYER_PROFILE")
}

pub fn packed_decode_enabled() -> bool {
    env_flag("RNB_PACKED_DECODE")
}

// cu41 Phase 1 + cu57 (2026-05-23): decode device-resident hidden chain.
// cu57 step 64-67d 의 wire-level carrier-readiness gating + 4 모델 long-decode
// 200 token bit-exact 검증 완료 후 default ON 승격. env="0" 시 disable.
pub fn cuda_decode_device_chain_enabled() -> bool {
    env_flag_default_on("RNB_CUDA_DECODE_DEVICE_CHAIN")
}

// cu44 step 20 + cu57: chain function 끝의 Gemma4 layer_output_scale device
// apply. cu57 default ON 승격. ON 시 chain function 안에서 hidden_dev *= scale
// + host 의 apply_layer_output_scale_inplace skip (double apply 방지).
pub fn cuda_decode_device_out_scale_enabled() -> bool {
    env_flag_default_on("RNB_CUDA_DECODE_DEVICE_OUT_SCALE")
}

// cu203: Qwen3.6 dense decode FFN device chain (norm→gate/up silu→down→residual
// 1왕복). 기존 GEMV별 host↔device 왕복이 토큰당 수천 memcpy + GPU idle 44%의
// 주범이라 device chain 으로 대체. env="0" 시 기존 개별 GEMV 경로로 복귀.
pub fn qwen35_dense_ffn_chain_enabled() -> bool {
    env_flag_default_on("RNB_CUDA_QWEN35_DENSE_FFN_CHAIN")
}

// cu203: Qwen3.6 GDN decode 층 core (norm→qkv/gate→alpha/beta→conv→delta→
// gated norm→ssm_out→residual) device chain. 층당 수십 회의 host↔device 왕복을
// hidden 1왕복으로 줄인다. env="0" 시 기존 단계별 경로로 복귀.
pub fn qwen35_gdn_decode_chain_enabled() -> bool {
    env_flag_default_on("RNB_CUDA_QWEN35_GDN_DECODE_CHAIN")
}

// cu52 step 51 + cu57: K/V projection 의 device output + f16 pack + attention
// 의 last_token_{k,v}_dev 전달. cu57 default ON 승격. KV cache 의 host f16
// 변환 + H2D round-trip 제거. CHAIN + OUT_SCALE 함께 활성 필요 (carrier-readiness).
pub fn cuda_decode_device_kv_cache_enabled() -> bool {
    env_flag_default_on("RNB_CUDA_DECODE_DEVICE_KV_CACHE")
}

// Diagnostic: warm dense attention projection (Q/K/V/O) mmap pages during the
// final prefill layers so the first decode step (D1) does not re-fault them
// from disk on large offloaded MoE models. Default off; enable only after an
// ABAB run proves a win on the target device.
pub fn hy3_decode_attn_prewarm_enabled() -> bool {
    env_flag("RNB_HY3_DECODE_ATTN_PREWARM")
}

pub fn q4k_kernel_backend() -> Option<String> {
    env_string("RNB_Q4K_KERNEL_BACKEND")
}

pub fn use_token_embedding_as_output() -> bool {
    env_flag("RNB_USE_TOKEN_EMBD_OUTPUT")
}

pub fn exact_output_gemv_enabled() -> bool {
    env_flag("RNB_EXACT_OUTPUT_GEMV")
}

pub fn tied_output_q8_disabled() -> bool {
    env_flag("RNB_DISABLE_TIED_OUTPUT_Q8")
}

/// Diagnostic opt-out for the macOS Metal Gemma path that preserves a tied
/// token embedding in its original GGUF quantization.
pub fn metal_gemma_tied_output_q8_forced() -> bool {
    env_truthy_override("RNB_METAL_GEMMA_TIED_OUTPUT_Q8").unwrap_or(false)
}

pub fn q80_prepack_load_enabled() -> bool {
    env_flag("RNB_Q80_PREPACK_LOAD")
}

pub fn use_norm_weight_enabled() -> bool {
    env_flag("RNB_USE_NORM_WEIGHT")
}

pub fn cpu_affinity() -> Option<String> {
    env_string("RNB_CPU_AFFINITY")
}

pub fn big_cores_requested() -> bool {
    env_flag("RNB_BIG_CORES")
}

pub fn rayon_num_threads() -> Option<String> {
    env_string("RAYON_NUM_THREADS")
}

pub fn moe_prefill_full_cores_enabled() -> bool {
    env_truthy_nonempty_override("RNB_MOE_PREFILL_FULL_CORES")
        .unwrap_or(cfg!(all(target_arch = "aarch64", target_os = "android")))
}

pub fn max_ctx_override() -> Option<usize> {
    env_usize("RNB_MAX_CTX")
}

pub fn cache_read_disabled() -> bool {
    env_flag("RNB_DISABLE_CACHE_READ")
}

pub fn flash_decode_enabled() -> bool {
    env_flag_default_on("RNB_FLASH_DECODE")
}

pub fn prefill_f16_cache_requested() -> bool {
    env_flag("RNB_PREFILL_F16_CACHE")
}

/// mt93 step5 opt-in. When set, force all attention compute math to fp32 by
/// disabling the architecture-default f16 KV cache round-trip (e.g. Gemma4
/// prefill, which would otherwise stage K/V through `half::f16` before the
/// fp32 FlashAttention kernel). Production default OFF — used only by the
/// drift probe to measure whether softmax burst observed in mt93-2/3/4 is
/// caused by the f16 round-trip on K/V activations.
pub fn attn_fp32_enabled() -> bool {
    env_flag("RNB_ATTN_FP32")
}

pub fn q4k_fused_prefill_attention_requested() -> bool {
    env_truthy_override("RNB_CUDA_Q4K_FUSED_PREFILL_ATTN").unwrap_or(true)
}

pub fn decode_device_hidden_enabled() -> bool {
    env_flag("RNB_CUDA_DECODE_DEVICE_HIDDEN")
}

pub fn drafter_cuda_enabled() -> bool {
    // cu65 (2026-05-22): default ON 으로 승격 (cu59-62 의 -7.5% generate lever).
    // env="0" 시 disable, 기존 env="1" 유지 (backward compat). drafter 가
    // attached + cuda feature 활성 시 자동 활성.
    env_flag_default_on("RNB_MTP_DRAFTER_CUDA")
}

pub fn drafter_metal_enabled() -> bool {
    env_flag_default_on("RNB_MTP_DRAFTER_METAL")
}

pub fn mtp_spec_requested() -> bool {
    env_string("RNB_MTP").is_some_and(|value| value != "0")
}

pub fn mtp_trace_enabled() -> bool {
    env_flag("RNB_MTP_TRACE")
}

pub fn mtp_dflash_adaptive_target_enabled() -> bool {
    env_truthy_override("RNB_MTP_DFLASH_ADAPTIVE_TARGET").unwrap_or(true)
}

// Metal의 16-slot DFlash verify는 한 invocation 비용이 커서 paired Muse drafter가
// 유효한 후보를 일찍 끊지 않도록 낮은 confidence gate를 쓴다. 다른 backend의
// 기존 8-slot 비용/수락률 계약은 유지한다.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
const MTP_DFLASH_CONFIDENCE_CUTOFF_DEFAULT: f32 = 0.08;
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
const MTP_DFLASH_CONFIDENCE_CUTOFF_DEFAULT: f32 = 0.15;

pub fn mtp_dflash_confidence_cutoff() -> Option<f32> {
    let Some(raw) = env_string("RNB_MTP_DFLASH_CONFIDENCE_CUTOFF") else {
        return Some(MTP_DFLASH_CONFIDENCE_CUTOFF_DEFAULT);
    };
    let normalized = raw.to_ascii_lowercase();
    if matches!(normalized.as_str(), "" | "0" | "false" | "off" | "no") {
        return None;
    }
    raw.parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
        .or(Some(MTP_DFLASH_CONFIDENCE_CUTOFF_DEFAULT))
}

/// `RNB_MTP_ACCEPT_PROBE=1`이면 MTP sequential verify가 target 분포 p와 draft 분포 q를
/// 모두 만들어 accept 확률을 계측한다. 진단 전용이며 draft device fast path를 끄므로
/// 느려진다.
pub fn mtp_accept_probe_enabled() -> bool {
    env_flag("RNB_MTP_ACCEPT_PROBE")
}

/// `RNB_MTP_DEVICE_VERIFY_TRACE=1`이면 device verify round마다 layer graph 빌드,
/// 커널 실행, state commit 시간을 분해해 출력한다.
pub fn mtp_device_verify_trace_enabled() -> bool {
    env_flag("RNB_MTP_DEVICE_VERIFY_TRACE")
}

/// accept probe가 p와 q에 적용할 가상 sampling plan.
///
/// 제품 MTP 진입은 greedy로 제한되어 있어 `temperature>0` 경로를 실제로 생성하며
/// 계측할 수 없다. 대신 greedy로 생성하되 계측만 이 plan으로 수행해 accept 확률을
/// 추정한다. 생성 경로가 달라 컨텍스트가 완전히 같지는 않다는 한계가 있다.
pub fn mtp_accept_probe_temperature() -> Option<f32> {
    env_string("RNB_MTP_ACCEPT_PROBE_TEMP").and_then(|value| value.parse().ok())
}

pub fn mtp_accept_probe_top_p() -> Option<f32> {
    env_string("RNB_MTP_ACCEPT_PROBE_TOP_P").and_then(|value| value.parse().ok())
}

pub fn mtp_accept_probe_top_k() -> Option<usize> {
    env_string("RNB_MTP_ACCEPT_PROBE_TOP_K").and_then(|value| value.parse().ok())
}

pub fn spec_force_batch_verify_enabled() -> bool {
    env_string("RNB_SPEC_FORCE_BATCH_VERIFY").is_some_and(|value| value == "1")
}

pub fn spec_decode_fast_window_enabled() -> bool {
    env_string("RNB_SPEC_DECODE_FAST_WINDOW").is_some_and(|value| value == "1")
}

pub fn spec_batch_no_bonus_enabled() -> bool {
    env_string("RNB_SPEC_BATCH_NO_BONUS").is_some_and(|value| value == "1")
}

pub fn spec_batch_no_bonus_override() -> Option<bool> {
    env_truthy_override("RNB_SPEC_BATCH_NO_BONUS")
}

pub fn spec_batch_prefix_snapshot_enabled() -> bool {
    env_string("RNB_SPEC_BATCH_PREFIX_SNAPSHOT").is_some_and(|value| value == "1")
}

pub fn spec_mtp_sequential_multi_enabled() -> bool {
    env_string("RNB_SPEC_MTP_SEQUENTIAL_MULTI").is_some_and(|value| value == "1")
}

pub fn mtp_batch_verify_enabled() -> bool {
    env_string("RNB_MTP_BATCH_VERIFY").is_some_and(|value| value == "1")
}

pub fn mtp_batch_verify_disabled() -> bool {
    env_string("RNB_MTP_BATCH_VERIFY")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub fn mtp_batch_decode_chain_enabled() -> bool {
    env_flag_default_on("RNB_MTP_BATCH_DECODE_CHAIN")
}

pub fn mtp_fast_retain_enabled() -> bool {
    env_string("RNB_MTP_FAST_RETAIN")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(cfg!(any(
            feature = "cuda",
            feature = "vulkan",
            all(target_arch = "aarch64", target_os = "android")
        )))
}

pub fn mtp_device_verify_override() -> Option<bool> {
    env_truthy_override("RNB_MTP_DEVICE_VERIFY")
}

pub fn mtp_device_verify_enabled() -> bool {
    mtp_device_verify_override().unwrap_or(false)
}

pub fn mtp_draft_only_enabled() -> bool {
    env_truthy_override("RNB_MTP_DRAFT_ONLY").unwrap_or(false)
}

pub fn mtp_shadow_precompute_enabled() -> bool {
    env_truthy_override("RNB_MTP_SHADOW_PRECOMPUTE").unwrap_or(false)
}

pub fn mtp_runway_max_extra() -> Option<usize> {
    env_usize("RNB_MTP_RUNWAY_MAX_EXTRA")
}

pub fn mtp_output_argmax_override() -> Option<bool> {
    env_truthy_override("RNB_MTP_OUTPUT_ARGMAX")
}

pub fn mtp_output_argmax_enabled() -> bool {
    mtp_output_argmax_override().unwrap_or_else(mtp_device_verify_enabled)
}

pub fn mtp_decode_block_enabled() -> bool {
    env_truthy_override("RNB_MTP_DECODE_BLOCK").unwrap_or(true)
}

pub fn mtp_device_draft_enabled() -> bool {
    env_truthy_override("RNB_CUDA_MTP_DEVICE_DRAFT").unwrap_or_else(mtp_device_verify_enabled)
}

pub fn cuda_prefill_argmax_only_enabled() -> bool {
    env_truthy_override("RNB_CUDA_PREFILL_ARGMAX_ONLY").unwrap_or(false)
}

pub fn cuda_device_verify_sync_delta_enabled() -> bool {
    env_truthy_nonempty_override("RNB_CUDA_DEVICE_VERIFY_SYNC_DELTA").unwrap_or(false)
}

pub fn cuda_device_verify_decode_enabled() -> bool {
    env_truthy_nonempty_override("RNB_CUDA_DEVICE_VERIFY_DECODE").unwrap_or(true)
}

pub fn cuda_device_prefill_trace_enabled() -> bool {
    env_string("RNB_CUDA_DEVICE_PREFILL_TRACE").as_deref() == Some("1")
}

pub fn cuda_nemotron_device_hidden_carrier_enabled() -> bool {
    true
}

pub fn cuda_nemotron_device_route_pack_enabled() -> bool {
    true
}

pub fn cuda_nemotron_device_prefill_v2_enabled() -> bool {
    true
}

pub fn cuda_nemotron_prefill_workspace_enabled() -> bool {
    true
}

pub fn cuda_nemotron_q8_sparse_expert_enabled() -> bool {
    true
}

pub fn cuda_nemotron_attention_device_input_enabled() -> bool {
    true
}

pub fn cuda_nemotron_carrier_route_trace_enabled() -> bool {
    env_truthy_override("RNB_CUDA_NEMOTRON_CARRIER_ROUTE_TRACE").unwrap_or(false)
}

pub fn cuda_nemotron_carrier_tensor_trace_enabled() -> bool {
    env_truthy_override("RNB_CUDA_NEMOTRON_CARRIER_TENSOR_TRACE").unwrap_or(false)
}

pub fn qwen_profile_all_layers_enabled() -> bool {
    env_present_os("RNB_QWEN_PROFILE_ALL_LAYERS")
}
