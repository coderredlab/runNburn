use std::{cell::OnceCell, ptr::NonNull, sync::OnceLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLArgumentEncoder, MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLDispatchType, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::ffn_chain::QwenMoeLlamaIdError;

const GEMV_F32_SRC: &str = include_str!("gemv_f32.metal");
const GEMV_Q4K_SRC: &str = include_str!("gemv_q4k.metal");
const GEMM_Q4K_SRC: &str = include_str!("gemm_q4k.metal");
const GEMM_Q4K_SHARED_SRC: &str = include_str!("gemm_q4k_shared.metal");
const GEMV_Q4K_SIMD_SRC: &str = include_str!("gemv_q4k_simd.metal");
const GEMV_Q4K_COALESCED_SRC: &str = include_str!("gemv_q4k_coalesced.metal");
const GEMV_Q4K_COALESCED_NSG2_SRC: &str = include_str!("gemv_q4k_coalesced_nsg2.metal");
const GEMV_Q6K_SRC: &str = include_str!("gemv_q6k.metal");
const GEMM_Q6K_SRC: &str = include_str!("gemm_q6k.metal");
const GEMM_Q6K_SHARED_SRC: &str = include_str!("gemm_q6k_shared.metal");
// pm34 M2: Metal 4 mpp::tensor_ops go/no-go PoC 셰이더.
const GEMM_TENSOROPS_POC_SRC: &str = include_str!("gemm_tensorops_poc.metal");
#[cfg(test)]
const GEMM_DENSE_F16_MICROBENCH_SRC: &str = include_str!("gemm_dense_f16_microbench.metal");
// pm48 ②: prefill(seq_len>1) fused qk_norm→rope device 커널. production 승격
// (device-resident attention chain 의 rope/qk_norm 단계, ctx 캐시 pipeline).
const PREFILL_ROPE_QK_NORM_SRC: &str = include_str!("prefill_rope_qk_norm.metal");
const GEMV_Q6K_SIMD_SRC: &str = include_str!("gemv_q6k_simd.metal");
const GEMV_Q6K_COALESCED_SRC: &str = include_str!("gemv_q6k_coalesced.metal");
const GEMV_Q6K_COALESCED_NSG2_SRC: &str = include_str!("gemv_q6k_coalesced_nsg2.metal");
const GEMV_Q5K_SRC: &str = include_str!("gemv_q5k.metal");
const GEMV_Q5K_SIMD_SRC: &str = include_str!("gemv_q5k_simd.metal");
const GEMV_Q5K_COALESCED_SRC: &str = include_str!("gemv_q5k_coalesced.metal");
const GEMV_Q8_0_SRC: &str = include_str!("gemv_q8_0.metal");
const GEMV_Q8_0_COALESCED_SRC: &str = include_str!("gemv_q8_0_coalesced.metal");
const GEMV_Q8_0_MLA_SLOTS_SRC: &str = include_str!("gemv_q8_0_mla_slots.metal");
const OUTPUT_ARGMAX_SRC: &str = include_str!("output_argmax.metal");

/// Apple Silicon SIMD-group width. 32 고정(구조적 상수 — threadExecutionWidth 와 일치).
pub(crate) const SIMD_WIDTH: usize = 32;
const RMS_NORM_SRC: &str = include_str!("rms_norm.metal");
const SILU_MUL_SRC: &str = include_str!("silu_mul.metal");
const RESIDUAL_ADD_SRC: &str = include_str!("residual_add.metal");
const QWEN_MOE_PREFILL_SCATTER_SRC: &str = include_str!("qwen_moe_prefill_scatter.metal");
const QWEN_MOE_PREFILL_ROUTE_EXACT_SRC: &str = include_str!("qwen_moe_prefill_route_exact.metal");
const QWEN_MOE_PREFILL_ID_MATMUL_SRC: &str = include_str!("qwen_moe_prefill_id_matmul.metal");
const QWEN_MOE_DECODE_SLOTS_SRC: &str = include_str!("qwen_moe_decode_slots.metal");
const GLM_MOE_DECODE_IQ_SRC: &str = include_str!("glm_moe_decode_iq.metal");
const ATTN_DECODE_SRC: &str = include_str!("attn_decode.metal");
const ATTN_DECODE_I8_SRC: &str = include_str!("attn_decode_i8.metal");
const ATTN_DECODE_I8_SPLITK_SRC: &str = include_str!("attn_decode_i8_splitk.metal");
const ROPE_MROPE_SRC: &str = include_str!("rope_mrope.metal");
const QK_NORM_SRC: &str = include_str!("qk_norm.metal");
const KV_APPEND_SRC: &str = include_str!("kv_append.metal");
const KV_APPEND_I8_SRC: &str = include_str!("kv_append_i8.metal");
const SSM_CONV_SILU_SRC: &str = include_str!("ssm_conv1d_silu.metal");
const DELTA_NET_STEP_SRC: &str = include_str!("delta_net_step.metal");
// pm39 M2: GDN delta_net chunkwise parallel scan (prefill, seq_len>1).
const DELTA_NET_SCAN_CHUNK_SRC: &str = include_str!("delta_net_scan_chunk.metal");
const DELTA_NET_SCAN_AR_SRC: &str = include_str!("delta_net_scan_ar.metal");
const DELTA_NET_SCAN_CHUNK_STEP45GEMM_SRC: &str =
    include_str!("delta_net_scan_chunk_step45gemm.metal");
const L2_NORM_SRC: &str = include_str!("l2_norm.metal");
const GDN_ALPHA_BETA_SRC: &str = concat!(
    include_str!("qwen_moe_prefill_route_exact.metal"),
    "\n",
    include_str!("gdn_alpha_beta.metal")
);
const SPLIT_QGATE_SRC: &str = include_str!("split_qgate.metal");
// pm45 M2-1: GDN prefill conv→delta device chain — conv_data 3-way q/k/v 분리.
const SPLIT_CONV_QKV_SRC: &str = include_str!("split_conv_qkv.metal");
// pm45 M2-2: GDN GQA q/k num_v_heads 순환 브로드캐스트(repeat).
const REPEAT_QK_SRC: &str = include_str!("repeat_qk.metal");
const ROPE_PARTIAL_SRC: &str = include_str!("rope_partial.metal");
const GATE_APPLY_SRC: &str = include_str!("gate_apply.metal");
const PREFILL_SPLIT_GATE_SRC: &str = include_str!("prefill_split_gate.metal");
const PREFILL_GATE_APPLY_SRC: &str = include_str!("prefill_gate_apply.metal");

/// `RNB_METAL_TIMING=1` 일 때 단계별 경과를 eprintln 으로 출력한다.
fn timing_enabled() -> bool {
    std::env::var("RNB_METAL_TIMING").as_deref() == Ok("1")
}

fn attn_resident_trace_enabled() -> bool {
    std::env::var("RNB_METAL_ATTN_RESIDENT_TRACE").as_deref() == Ok("1")
}

fn env_falsey(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn attn_splitk_splits_from_env() -> usize {
    match std::env::var("RNB_METAL_ATTN_SPLITK") {
        Ok(v)
            if matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            ) =>
        {
            0
        }
        Ok(v) => match v.parse::<usize>() {
            Ok(1) => 32,
            Ok(n) => n.clamp(2, 64),
            Err(_) => 32,
        },
        Err(_) => 32,
    }
}

fn gemv_coalesced_nsg2_enabled(gemv_coalesced: bool) -> bool {
    gemv_coalesced && !env_falsey("RNB_METAL_GEMV_COALESCED_NSG2")
}

fn qwen_moe_prefill_down_scatter_enabled() -> bool {
    !env_falsey("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER")
}

pub(crate) fn qwen_moe_prefill_id_primitive_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
}

pub(crate) fn qwen_moe_llama_id_requested() -> bool {
    !env_falsey("RNB_METAL_QWEN35_MOE_LLAMA_ID")
}

pub(crate) fn qwen_moe_prefill_mulmmid_v3_requested() -> bool {
    !qwen_moe_llama_id_requested()
        && std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
            .unwrap_or(false)
}

pub(crate) fn qwen_moe_prefill_mulmmid_v4_requested() -> bool {
    !qwen_moe_llama_id_requested()
        && std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
            .unwrap_or(false)
}

#[cfg(test)]
mod env_tests {
    use super::{
        attn_splitk_splits_from_env, gemv_coalesced_nsg2_enabled, qwen_moe_llama_id_requested,
        qwen_moe_prefill_down_scatter_enabled, qwen_moe_prefill_id_primitive_requested,
        qwen_moe_prefill_mulmmid_v3_requested, qwen_moe_prefill_mulmmid_v4_requested,
    };
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn attn_splitk_defaults_on_and_preserves_opt_out() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prev = std::env::var("RNB_METAL_ATTN_SPLITK").ok();

        std::env::remove_var("RNB_METAL_ATTN_SPLITK");
        assert_eq!(attn_splitk_splits_from_env(), 32);

        std::env::set_var("RNB_METAL_ATTN_SPLITK", "1");
        assert_eq!(attn_splitk_splits_from_env(), 32);

        std::env::set_var("RNB_METAL_ATTN_SPLITK", "64");
        assert_eq!(attn_splitk_splits_from_env(), 64);

        std::env::set_var("RNB_METAL_ATTN_SPLITK", "999");
        assert_eq!(attn_splitk_splits_from_env(), 64);

        for off in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_ATTN_SPLITK", off);
            assert_eq!(attn_splitk_splits_from_env(), 0);
        }

        match prev {
            Some(value) => std::env::set_var("RNB_METAL_ATTN_SPLITK", value),
            None => std::env::remove_var("RNB_METAL_ATTN_SPLITK"),
        }
    }

    #[test]
    fn gemv_coalesced_nsg2_defaults_on_with_falsey_opt_out() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prev = std::env::var("RNB_METAL_GEMV_COALESCED_NSG2").ok();

        std::env::remove_var("RNB_METAL_GEMV_COALESCED_NSG2");
        assert!(gemv_coalesced_nsg2_enabled(true));
        assert!(!gemv_coalesced_nsg2_enabled(false));

        for off in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_GEMV_COALESCED_NSG2", off);
            assert!(!gemv_coalesced_nsg2_enabled(true));
        }

        std::env::set_var("RNB_METAL_GEMV_COALESCED_NSG2", "1");
        assert!(gemv_coalesced_nsg2_enabled(true));

        match prev {
            Some(value) => std::env::set_var("RNB_METAL_GEMV_COALESCED_NSG2", value),
            None => std::env::remove_var("RNB_METAL_GEMV_COALESCED_NSG2"),
        }
    }

    #[test]
    fn qwen_moe_prefill_down_scatter_defaults_on_with_falsey_opt_out() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prev = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER").ok();

        std::env::remove_var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER");
        assert!(qwen_moe_prefill_down_scatter_enabled());

        for off in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER", off);
            assert!(!qwen_moe_prefill_down_scatter_enabled());
        }

        std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER", "1");
        assert!(qwen_moe_prefill_down_scatter_enabled());

        match prev {
            Some(value) => std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER", value),
            None => std::env::remove_var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER"),
        }
    }

    #[test]
    fn qwen_moe_prefill_id_primitive_is_opt_in_with_falsey_opt_out() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prev = std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE").ok();

        std::env::remove_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE");
        assert!(!qwen_moe_prefill_id_primitive_requested());

        for off in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", off);
            assert!(!qwen_moe_prefill_id_primitive_requested());
        }

        std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        assert!(qwen_moe_prefill_id_primitive_requested());

        for on in ["true", "on", "yes"] {
            std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", on);
            assert!(qwen_moe_prefill_id_primitive_requested(), "{on}");
        }

        std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "typo");
        assert!(!qwen_moe_prefill_id_primitive_requested());

        match prev {
            Some(value) => std::env::set_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", value),
            None => std::env::remove_var("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE"),
        }
    }

    #[test]
    fn qwen_moe_llama_id_defaults_on_with_falsey_opt_out() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let prev = std::env::var(key).ok();

        std::env::remove_var(key);
        assert!(qwen_moe_llama_id_requested());

        for off in ["0", "false", "off", "no"] {
            std::env::set_var(key, off);
            assert!(!qwen_moe_llama_id_requested(), "{off}");
        }

        for on in ["1", "true", "on", "yes", "typo"] {
            std::env::set_var(key, on);
            assert!(qwen_moe_llama_id_requested(), "{on}");
        }

        match prev {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen_moe_llama_id_suppresses_legacy_mulmmid_requests() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let llama_key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let v3_key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3";
        let v4_key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4";
        let prev_llama = std::env::var(llama_key).ok();
        let prev_v3 = std::env::var(v3_key).ok();
        let prev_v4 = std::env::var(v4_key).ok();

        std::env::set_var(v3_key, "1");
        std::env::set_var(v4_key, "true");

        for llama_on in ["1", "true", "on", "yes"] {
            std::env::set_var(llama_key, llama_on);
            assert!(!qwen_moe_prefill_mulmmid_v3_requested(), "{llama_on}");
            assert!(!qwen_moe_prefill_mulmmid_v4_requested(), "{llama_on}");
        }

        for llama_off in ["0", "false", "off", "no", "typo"] {
            std::env::set_var(llama_key, llama_off);

            std::env::set_var(v3_key, "1");
            std::env::set_var(v4_key, "0");
            assert!(qwen_moe_prefill_mulmmid_v3_requested(), "{llama_off}");
            assert!(!qwen_moe_prefill_mulmmid_v4_requested(), "{llama_off}");

            std::env::set_var(v3_key, "0");
            std::env::set_var(v4_key, "true");
            assert!(!qwen_moe_prefill_mulmmid_v3_requested(), "{llama_off}");
            assert!(qwen_moe_prefill_mulmmid_v4_requested(), "{llama_off}");
        }

        std::env::set_var(v3_key, "1");
        std::env::set_var(v4_key, "true");
        std::env::remove_var(llama_key);
        assert!(qwen_moe_prefill_mulmmid_v3_requested());
        assert!(qwen_moe_prefill_mulmmid_v4_requested());

        match prev_llama {
            Some(value) => std::env::set_var(llama_key, value),
            None => std::env::remove_var(llama_key),
        }
        match prev_v3 {
            Some(value) => std::env::set_var(v3_key, value),
            None => std::env::remove_var(v3_key),
        }
        match prev_v4 {
            Some(value) => std::env::set_var(v4_key, value),
            None => std::env::remove_var(v4_key),
        }
    }
}

/// Apple Silicon page size. 16KB. (sysconf(_SC_PAGESIZE) 로도 얻을 수 있으나
/// Apple Silicon 은 고정 16384 — 구조적 상수라 하드코딩 허용.)
const METAL_PAGE: usize = 16384;

/// mmap 내부 포인터(`raw.as_ptr() as usize`)와 weight 길이로
/// NoCopy buffer 인자를 계산한다.
///
/// 반환: `(aligned_ptr, page_offset, buf_len)`
/// - `aligned_ptr`: page 경계로 내린 base (NoCopy buffer 시작).
/// - `page_offset`: aligned_ptr 부터 실제 weight 까지 byte offset (커널에 전달).
/// - `buf_len`: `page_offset + raw_len` 을 page 배수로 올림 (NoCopy length).
fn page_align(ptr: usize, raw_len: usize) -> (usize, usize, usize) {
    let aligned = ptr & !(METAL_PAGE - 1);
    let page_offset = ptr - aligned;
    let span = page_offset + raw_len;
    let buf_len = span.div_ceil(METAL_PAGE) * METAL_PAGE;
    (aligned, page_offset, buf_len)
}

// ---------------------------------------------------------------------------
// MetalContext — device/queue/pipeline 을 한 번만 생성하고 재사용한다.
// ---------------------------------------------------------------------------

/// Metal device, command queue, Q4_K compute pipeline 을 캐싱하는 구조체.
///
/// `build_metal_context()` 로 한 번 생성한 뒤 `gemv_q4k_with_ctx()` 에 넘기면
/// per-call device open / MSL compile 비용이 사라진다.
///
/// `Retained<ProtocolObject<dyn MTL*>>` 은 `!Send + !Sync` 이므로 이 구조체도
/// `Send/Sync` 가 아니다. `MetalBackend` 를 단일 스레드에서 사용하는 한 문제없다.
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub q4k_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q4k_simd_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp kernel_mul_mv_q4_K_f32_impl single-row 이식(coalesced 접근 측정용).
    /// q4k_simd 의 fallback(stride-32) 경로 대체 후보. standalone — chain 미연결.
    pub q4k_coalesced_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp N_SG=2/N_R0=2 shape: threadgroup 당 SIMD-group 2개, output row 4개.
    pub q4k_coalesced_nsg2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm33: Q4_K batch GEMM(M>1) naive 커널. prefill FFN gate/up + (down Q4_K 시) 경로.
    /// grid 2D (row, tok). 자작 quantized batch matmul — decode GEMV(M=1) 의 M축 확장.
    pub gemm_q4k_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm33 튜닝: Q4_K batch GEMM dequant-공유 커널. threadgroup=row 1개, superblock dequant
    /// 1번→threadgroup memory 전 token 공유. naive 의 dequant M배 중복을 1배로.
    pub gemm_q4k_shared_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q6k_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q6k_simd_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp kernel_mul_mv_q6_K_f32_impl nr0=2 multi-row 이식(coalesced + activation reuse).
    /// `gemv_coalesced` flag 시 q6k_simd 대체. q4k_coalesced 와 같은 flag 에 묶임.
    pub q6k_coalesced_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp N_SG=2/N_R0=2 shape: threadgroup 당 SIMD-group 2개, output row 4개.
    pub q6k_coalesced_nsg2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm33: Q6_K batch GEMM(M>1) naive 커널. prefill FFN down(Q6_K) 경로. grid 2D (row, tok).
    pub gemm_q6k_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm33 튜닝: Q6_K batch GEMM dequant-공유 커널 (gemm_q4k_shared 패턴).
    pub gemm_q6k_shared_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q5k_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q5k_simd_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp kernel_mul_mv_q5_K_f32_impl nr0=2 multi-row 이식(coalesced + activation reuse).
    /// `gemv_coalesced` flag 시 q5k_simd 대체. q4k_coalesced 와 같은 flag 에 묶임.
    pub q5k_coalesced_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q8_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// llama.cpp kernel_mul_mv_q8_0_f32_impl NR0=2 multi-row 이식(coalesced + activation reuse).
    /// `gemv_coalesced` flag 시 q8_0 대체. q4k_coalesced 와 같은 flag 에 묶임.
    pub q8_0_coalesced_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub q8_0_mla_slots_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm51: output projection logits -> token id reduction. Used after output GEMV so
    /// host reads back one u32 instead of the full logits vector.
    pub output_argmax_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm26: chain 용 F32 GEMV(gemv_f32_chain). 27B GDN 의 F32 ssm_alpha/beta 무손실 device 화.
    pub f32_chain_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Router 전용 F32 GEMV. output row당 SIMD-group 하나가 K를 coalesced load한다.
    pub f32_chain_simd_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub prefill_f32_proj_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rms_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rms_norm_batch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm43: GDN prefill gated RMSNorm+SiLU(batch, rows>1). rmsnorm(out)*silu(z) fused per-row.
    pub gated_rmsnorm_silu_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub silu_mul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub silu_mul_f16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub silu_mul_half_f16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub residual_add_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_gather_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_gather_f16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_scatter_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_topk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_topk_chain_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_router_f32_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_rms_norm_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_gdn_prefill_rms_norm_f32_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_prefill_shared_gate_chain_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_prefill_l2_norm_exact_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_prefill_gated_rmsnorm_silu_chain_exact_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qwen_moe_decode_q4k_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_down_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_pair_slots_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q5k_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q6k_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_selected_slots_coalesced_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_selected_slots_nsg2_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_selected_pair_slots_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q5k_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q6k_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub glm_moe_decode_iq2xxs_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub glm_moe_decode_iq3xxs_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub glm_moe_decode_iq4xs_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub glm_moe_decode_iq2s_selected_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_table_slots_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q5k_table_slots_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q6k_table_slots_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_q4k_table_arg_encoder:
        Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub qwen_moe_decode_q5k_table_arg_encoder:
        Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub qwen_moe_decode_q6k_table_arg_encoder:
        Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub qwen_moe_decode_silu_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_reduce_slots_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub qwen_moe_decode_route_shared_pipeline:
        OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub attn_decode_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm22: per-slot int8 KV decode attention 커널. f16 attn_decode 와 택일(ctx.kv_int8 분기).
    pub attn_decode_i8_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm54: split-K int8 KV decode attention. 긴 KV 에서 query-head 단위 저점유를
    /// KV 축 병렬(partial + reduce)로 보완하는 opt-in path.
    pub attn_decode_i8_splitk_part_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub attn_decode_i8_splitk_reduce_pipeline:
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rope_mrope_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub qk_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub kv_append_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm22: per-slot int8 KV append 커널. f16 kv_append 와 택일(ctx.kv_int8 분기).
    pub kv_append_i8_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub ssm_conv_silu_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm43: prefill batch(seq_len>1) conv1d+silu. decode(ssm_conv_silu) 의 seq_len rows 확장(grid=seq_len*channels).
    pub ssm_conv_silu_batch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub delta_net_step_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm39 M2: GDN delta_net chunkwise parallel scan(prefill, seq_len>1). f32, CPU
    /// `delta_net_scan_chunkwise` oracle 과 token-identical. decode delta_net_step 의 chunk 확장.
    pub delta_net_scan_chunk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm58: Qwen3Next long prefill candidate. Sequential recurrence inside one Metal kernel,
    /// row-parallel over state rows. Pipelines are specialized by head_k_dim / SIMD_WIDTH.
    pub delta_net_scan_ar1_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub delta_net_scan_ar2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub delta_net_scan_ar4_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub delta_net_scan_ar8_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub l2_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub gdn_alpha_beta_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub split_qgate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm45 M2-1: GDN prefill conv_data → 연속 q/k/v 분리(bit-identical copy).
    pub split_conv_qkv_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// pm45 M2-2: GDN GQA q/k 를 num_v_heads 로 순환 repeat(bit-identical gather).
    pub repeat_qk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rope_partial_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub gate_apply_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) prefill_split_gate_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) prefill_gate_apply_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// chain GEMV(Q4_K/Q6_K)에 SIMD-group 협력 커널 사용 여부. pm21 승격으로 **default on**
    /// (in-process 측정: chain GPU/commit_wait −11.7%, wall −8% 모든 A<B, token-identical).
    /// `RNB_METAL_GEMV_SIMD=0` 으로 baseline 강제(측정 비교용 보존).
    pub(crate) gemv_simd: bool,
    /// chain decode 의 dispatch class 선택 (env RNB_METAL_CHAIN_PROFILE, REST 분해 측정용).
    /// default All(production). gemv/small/attn 으로 class 별 격리 측정(차감법).
    pub(crate) chain_profile: ChainProfile,
    /// attn_decode K·V read 주소 wrap mask (측정 게이트 pm22, env RNB_METAL_ATTN_READ_WINDOW).
    /// default 0xFFFFFFFF(normal). window(2^n)면 mask=window-1 → read traffic cap, compute 불변.
    /// normal vs capped GPU time 차로 K·V read traffic 순비용 측정(int8 KV 효과 upper bound).
    pub(crate) attn_read_mask_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// RNB_METAL_KV_INT8=1 → KV cache per-slot int8 양자화(pm22). f16 버퍼와 택일.
    pub(crate) kv_int8: bool,
    /// RNB_METAL_ATTN_SPLITK={1|N}. unset/1이면 32 split, 0이면 기존 attn_decode_i8.
    pub(crate) attn_splitk_splits: usize,
    /// 짧은 KV 에서 split/reduce 오버헤드를 피하는 opt-in threshold.
    pub(crate) attn_splitk_min_kv: usize,
    /// chain decode compute encoder 의 dispatch type. true 면 concurrent(독립 dispatch
    /// 가 GPU 에서 overlap — 의존성은 `chain_barrier` 로 명시), false 면 serial(dispatch
    /// 간 자동 hazard tracking). pm28 production default ON(opt-out): adversarial audit
    /// race 0 + 27B −12.3%/9B −5.8% 무손실. RNB_METAL_CHAIN_CONCURRENT=0 으로 serial 강제.
    pub(crate) chain_concurrent: bool,
    /// RNB_METAL_GEMV_COALESCED=1 → chain q4k GEMV 를 llama식 nr0=2 multi-row coalesced 커널로
    /// dispatch(grid=ceil(N/2), activation reuse). 측정용. default(미설정) 시 기존 q4k_simd.
    pub(crate) gemv_coalesced: bool,
    pub(crate) gemv_coalesced_nsg2: bool,
    /// pm33 튜닝: prefill FFN chain 의 batch GEMM 커널 선택. default false=dequant-공유(shared),
    /// `RNB_METAL_PREFILL_FFN_NAIVE=1` 이면 naive(grid 2D, dequant M배). naive vs shared 비교용.
    pub(crate) prefill_ffn_naive: bool,
    /// pm34: M5 neural accelerator(`mpp::tensor_ops`) 가용 여부. build 시 1회 판정(캐시).
    pub(crate) tensorops_capable: bool,
    /// pm34: capability=true 일 때만 Some. false 면 None(파이프라인 미생성 — non-M5 panic 회피).
    /// go/no-go PoC(16×16×32 dense) 검증용 — run_tensorops_poc 테스트 자산.
    pub(crate) prefill_ffn_tensorops_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm34 M6: production Q4_K tensorops GEMM(M/N grid tiling). chain gate/up/down(Q4_K).
    /// capability=true 일 때만 Some.
    pub(crate) gemm_q4k_tensorops_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm34 M6: production Q6_K tensorops GEMM. chain down(Q6_K mixed precision).
    pub(crate) gemm_q6k_tensorops_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm36: production Q5_K tensorops GEMM. GDN ssm_out(Q5_K) + 9B 전체 GDN projection.
    pub(crate) gemm_q5k_tensorops_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm42 M3: production v2 tensorops GEMM(llama 패턴 — weight threadgroup dequant +
    /// activation f16 device-direct + C cooperative tensor, 64×128 winner 타일). v1 대비 ~4x.
    /// capability=true 시 Some. FFN chain gate/up/down(Q4_K)+down(Q6_K) + GDN/ATN proj(Q4_K/Q5_K/Q6_K).
    pub(crate) gemm_q4k_tensorops_v2_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_v2_pair_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q5k_tensorops_v2_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q6k_tensorops_v2_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_v2_scatter_accum_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q6k_tensorops_v2_scatter_accum_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_id_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_id_v2_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_id_v2_f16_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q4k_tensorops_id_v2_64x128_f16_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gemm_q6k_tensorops_id_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_library: OnceLock<Retained<ProtocolObject<dyn MTLLibrary>>>,
    #[cfg(test)]
    qwen_moe_llama_library_init_count: std::sync::atomic::AtomicUsize,
    pub(crate) qwen_moe_llama_id_map0_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_id_build_blocks_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q4k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q4k_f16_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q5k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q5k_f16_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q6k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q6k_f16_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_chain_small_pipelines:
        [OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>; 3],
    pub(crate) qwen_moe_chain_large_pipelines:
        [OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>; 6],
    pub(crate) qwen_moe_chain_cast_large_slots_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_quantize_q8k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_shared_mul_mm_q4k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_shared_mul_mm_q6k_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_swiglu_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_weighted_rank_reduce_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_llama_expert_order_reduce_f32_pipeline:
        OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_v3_q6_down_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_v3_q4_down_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_v4_q6_down_scatter_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_v4_q4_down_scatter_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_v3_token_rank_combine_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_id_partial_reduce_scatter_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_id_q4_gate_up_tile_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_id_silu_mul_tile_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) qwen_moe_id_q6_down_partial_tile_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm42 M3 step2: f32→f16 cast(v2 GEMM activation 변환용). capability=true 시 Some
    /// (cast 커널이 matmul2d include 한 GEMM_TENSOROPS_POC_SRC 에 있어 build_pipeline_v4 필요).
    pub(crate) cast_f32_f16_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm47 ② GDN delta scan STEP4+STEP5 GEMM 커널(matmul2d). capability=true 시 Some.
    pub(crate) delta_net_scan_chunk_step45gemm_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm48 ① flash attention prefill 커널(simdgroup matmul2d). capability=true 시 Some.
    /// dense causal GQA prefill attention compute(host 입출력 seam). HD=256 고정.
    pub(crate) flash_attn_prefill_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// pm48 ② prefill qk_norm→rope fused 커널(device-resident attention chain 부품).
    /// per-head RMSNorm → text M-RoPE(partial n_rot) 를 device q/k 에 in-chain 적용. 항상 build.
    pub(crate) prefill_rope_qk_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) prefill_rope_only_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// chain decode 측정 모드: dispatch 를 class 별로 emit 해 GPU time 을 격리(pm21 REST 분해).
/// M5 는 dispatch-boundary counter 미지원이라 production 단일 encoder 안 dispatch 별
/// timestamp 가 불가 → class on/off 차감법으로 GEMV vs 소형커널 vs attention 비중을 잰다.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ChainProfile {
    /// production: 모든 dispatch emit.
    All,
    /// weight GEMV(q/k/v/o, qkv/gate/alpha/beta/ssm_out, ffn gate/up/down) 만.
    Gemv,
    /// kv 독립 소형 커널(norm/rope/residual/silu/conv/delta_net/l2_norm/split/gate_apply/kv_append) 만.
    Small,
    /// attention compute(attn_decode, kv 의존) 만.
    Attn,
}

impl ChainProfile {
    pub(crate) fn emit_gemv(self) -> bool {
        matches!(self, Self::All | Self::Gemv)
    }
    pub(crate) fn emit_small(self) -> bool {
        matches!(self, Self::All | Self::Small)
    }
    pub(crate) fn emit_attn(self) -> bool {
        matches!(self, Self::All | Self::Attn)
    }
}

pub(crate) struct GemvCoalescedDispatch<'a> {
    pub pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
    pub rows_per_threadgroup: usize,
    pub threads_per_threadgroup: usize,
}

/// MSL source 를 컴파일해 named kernel 의 compute pipeline 을 빌드한다.
/// 컴파일/함수 조회/pipeline 빌드 실패 시 panic (fake fallback 없음).
fn build_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &str,
    fn_name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let source = NSString::from_str(src);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|e| panic!("Metal: failed to compile {fn_name}.metal: {e:?}"));
    let name = NSString::from_str(fn_name);
    let function = library
        .newFunctionWithName(&name)
        .unwrap_or_else(|| panic!("Metal: function '{fn_name}' not found"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Metal: failed to create pipeline '{fn_name}': {e:?}"))
}

/// Router top-k 전용 strict FP pipeline. 다른 Metal 커널의 전역 compile policy는
/// 건드리지 않고 이 library만 fast-math를 끄고 safe math mode로 컴파일한다.
#[allow(deprecated)]
fn build_pipeline_safe_math(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &str,
    fn_name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    use objc2_metal::{MTLCompileOptions, MTLMathMode};

    let source = NSString::from_str(src);
    let options = MTLCompileOptions::new();
    options.setFastMathEnabled(false);
    options.setMathMode(MTLMathMode::Safe);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .unwrap_or_else(|e| panic!("Metal: failed to compile safe-math {fn_name}.metal: {e:?}"));
    let name = NSString::from_str(fn_name);
    let function = library
        .newFunctionWithName(&name)
        .unwrap_or_else(|| panic!("Metal: function '{fn_name}' not found"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Metal: failed to create pipeline '{fn_name}': {e:?}"))
}

fn build_pipeline_with_argument_encoder(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &str,
    fn_name: &str,
    arg_buffer_index: usize,
) -> (
    Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
) {
    let source = NSString::from_str(src);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|e| panic!("Metal: failed to compile {fn_name}.metal: {e:?}"));
    let name = NSString::from_str(fn_name);
    let function = library
        .newFunctionWithName(&name)
        .unwrap_or_else(|| panic!("Metal: function '{fn_name}' not found"));
    let argument_encoder = unsafe { function.newArgumentEncoderWithBufferIndex(arg_buffer_index) };
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Metal: failed to create pipeline '{fn_name}': {e:?}"));
    (pipeline, argument_encoder)
}

fn lazy_qwen_moe_decode_pipeline<'a>(
    device: &ProtocolObject<dyn MTLDevice>,
    cell: &'a OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    fn_name: &str,
) -> &'a Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    cell.get_or_init(|| build_pipeline(device, QWEN_MOE_DECODE_SLOTS_SRC, fn_name))
}

fn lazy_glm_moe_decode_pipeline<'a>(
    device: &ProtocolObject<dyn MTLDevice>,
    cell: &'a OnceCell<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    fn_name: &str,
) -> &'a Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    cell.get_or_init(|| build_pipeline(device, GLM_MOE_DECODE_IQ_SRC, fn_name))
}

fn lazy_qwen_moe_llama_pipeline<'a>(
    ctx: &'a MetalContext,
    cell: &'a OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    fn_name: &str,
) -> &'a Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let library = ctx.qwen_moe_llama_library.get_or_init(|| {
        #[cfg(test)]
        ctx.qwen_moe_llama_library_init_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        build_library_v4(
            &ctx.device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_prefill_id_matmul",
        )
    });
    cell.get_or_init(|| build_pipeline_from_library(&ctx.device, library, fn_name))
}

impl MetalContext {
    pub(crate) fn qwen_moe_decode_q4k_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q4k_slots_pipeline,
            "qwen_moe_decode_q4k_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_q4k_down_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q4k_down_slots_pipeline,
            "qwen_moe_decode_q4k_down_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_q5k_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q5k_slots_pipeline,
            "qwen_moe_decode_q5k_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_q6k_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q6k_slots_pipeline,
            "qwen_moe_decode_q6k_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_q4k_selected_slots_coalesced_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q4k_selected_slots_coalesced_pipeline,
            "qwen_moe_decode_q4k_selected_slots_coalesced",
        )
    }

    pub(crate) fn qwen_moe_decode_q5k_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q5k_selected_slots_pipeline,
            "qwen_moe_decode_q5k_selected_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_q6k_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_q6k_selected_slots_pipeline,
            "qwen_moe_decode_q6k_selected_slots",
        )
    }

    pub(crate) fn glm_moe_decode_iq2xxs_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_glm_moe_decode_pipeline(
            &self.device,
            &self.glm_moe_decode_iq2xxs_selected_slots_pipeline,
            "glm_moe_decode_iq2xxs_selected_slots",
        )
    }

    pub(crate) fn glm_moe_decode_iq3xxs_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_glm_moe_decode_pipeline(
            &self.device,
            &self.glm_moe_decode_iq3xxs_selected_slots_pipeline,
            "glm_moe_decode_iq3xxs_selected_slots",
        )
    }

    pub(crate) fn glm_moe_decode_iq4xs_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_glm_moe_decode_pipeline(
            &self.device,
            &self.glm_moe_decode_iq4xs_selected_slots_pipeline,
            "glm_moe_decode_iq4xs_selected_slots",
        )
    }

    pub(crate) fn glm_moe_decode_iq2s_selected_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_glm_moe_decode_pipeline(
            &self.device,
            &self.glm_moe_decode_iq2s_selected_slots_pipeline,
            "glm_moe_decode_iq2s_selected_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_silu_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_silu_slots_pipeline,
            "qwen_moe_decode_silu_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_reduce_slots_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_reduce_slots_pipeline,
            "qwen_moe_decode_reduce_slots",
        )
    }

    pub(crate) fn qwen_moe_decode_route_shared_pipeline(
        &self,
    ) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        lazy_qwen_moe_decode_pipeline(
            &self.device,
            &self.qwen_moe_decode_route_shared_pipeline,
            "qwen_moe_decode_route_shared",
        )
    }

    pub(crate) fn qwen_moe_llama_id_map0_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_id_map0_pipeline,
                "qwen_moe_llama_id_map0",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_id_build_blocks_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_id_build_blocks_pipeline,
                "qwen_moe_llama_id_build_blocks",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q4k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q4k_f32_pipeline,
                "qwen_moe_llama_mul_mm_id_q4k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q4k_f16_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q4k_f16_pipeline,
                "qwen_moe_llama_mul_mm_id_q4k_f16",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q5k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q5k_f32_pipeline,
                "qwen_moe_llama_mul_mm_id_q5k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q5k_f16_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q5k_f16_pipeline,
                "qwen_moe_llama_mul_mm_id_q5k_f16",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q6k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q6k_f32_pipeline,
                "qwen_moe_llama_mul_mm_id_q6k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q6k_f16_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q6k_f16_pipeline,
                "qwen_moe_llama_mul_mm_id_q6k_f16",
            )
        })
    }

    pub(crate) fn qwen_moe_chain_small_pipeline(
        &self,
        variant: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        const NAMES: [&str; 3] = [
            "qwen_moe_chain_small_q4k_f32",
            "qwen_moe_chain_small_q5k_f32",
            "qwen_moe_chain_small_q6k_f32",
        ];
        let cell = self.qwen_moe_chain_small_pipelines.get(variant)?;
        self.tensorops_capable
            .then(|| lazy_qwen_moe_llama_pipeline(self, cell, NAMES[variant]))
    }

    pub(crate) fn qwen_moe_chain_large_pipeline(
        &self,
        variant: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        const NAMES: [&str; 6] = [
            "qwen_moe_chain_large_q4k_f32",
            "qwen_moe_chain_large_q4k_f16",
            "qwen_moe_chain_large_q5k_f16",
            "qwen_moe_chain_large_q6k_f16",
            "qwen_moe_chain_large_q4k_f16_dense",
            "qwen_moe_chain_large_q6k_f16_dense",
        ];
        let cell = self.qwen_moe_chain_large_pipelines.get(variant)?;
        self.tensorops_capable
            .then(|| lazy_qwen_moe_llama_pipeline(self, cell, NAMES[variant]))
    }

    pub(crate) fn qwen_moe_chain_cast_large_slots_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_chain_cast_large_slots_pipeline,
                "qwen_moe_chain_cast_large_slots_f32_f16",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_quantize_q8k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_quantize_q8k_f32_pipeline,
                "qwen_moe_llama_quantize_q8k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline,
                "qwen_moe_llama_mul_mm_id_q4k_q8k",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline,
                "qwen_moe_llama_mul_mm_id_q6k_q8k",
            )
        })
    }

    pub(crate) fn qwen_moe_shared_mul_mm_q4k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_shared_mul_mm_q4k_f32_pipeline,
                "qwen_moe_shared_mul_mm_q4k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_shared_mul_mm_q6k_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_shared_mul_mm_q6k_f32_pipeline,
                "qwen_moe_shared_mul_mm_q6k_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline,
                "qwen_moe_llama_shared_mul_mm_q8_0_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_swiglu_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_swiglu_f32_pipeline,
                "qwen_moe_llama_swiglu_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_weighted_rank_reduce_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_weighted_rank_reduce_f32_pipeline,
                "qwen_moe_llama_weighted_rank_reduce_f32",
            )
        })
    }

    pub(crate) fn qwen_moe_llama_expert_order_reduce_f32_pipeline(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.tensorops_capable.then(|| {
            lazy_qwen_moe_llama_pipeline(
                self,
                &self.qwen_moe_llama_expert_order_reduce_f32_pipeline,
                "qwen_moe_llama_expert_order_reduce_f32",
            )
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod qwen_moe_llama_lazy_pipeline_tests {
    use std::sync::atomic::Ordering;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLComputePipelineState;

    use super::{build_metal_context_with_opts, MetalContext};

    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
    type Getter = for<'a> fn(&'a MetalContext) -> Option<&'a Pipeline>;

    const GETTERS: [Getter; 14] = [
        MetalContext::qwen_moe_llama_id_map0_pipeline,
        MetalContext::qwen_moe_llama_id_build_blocks_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q4k_f32_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q4k_f16_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q5k_f32_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q5k_f16_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q6k_f32_pipeline,
        MetalContext::qwen_moe_llama_mul_mm_id_q6k_f16_pipeline,
        MetalContext::qwen_moe_shared_mul_mm_q4k_f32_pipeline,
        MetalContext::qwen_moe_shared_mul_mm_q6k_f32_pipeline,
        MetalContext::qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline,
        MetalContext::qwen_moe_llama_swiglu_f32_pipeline,
        MetalContext::qwen_moe_llama_weighted_rank_reduce_f32_pipeline,
        MetalContext::qwen_moe_llama_expert_order_reduce_f32_pipeline,
    ];

    fn address(pipeline: &Pipeline) -> usize {
        std::ptr::from_ref(pipeline).cast::<()>() as usize
    }

    #[test]
    fn llama_library_is_lazy_and_tensorops_gated() {
        let mut ctx = build_metal_context_with_opts(false, false).expect("no metal device");
        assert!(ctx.qwen_moe_llama_library.get().is_none());
        assert!(ctx.qwen_moe_llama_id_map0_pipeline.get().is_none());

        ctx.tensorops_capable = false;
        for getter in GETTERS {
            assert!(getter(&ctx).is_none());
        }
        assert!(ctx.qwen_moe_llama_library.get().is_none());
        assert_eq!(
            ctx.qwen_moe_llama_library_init_count
                .load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn llama_library_initializes_once_and_getters_are_stable() {
        let ctx = build_metal_context_with_opts(false, false).expect("no metal device");
        if !ctx.tensorops_capable {
            assert!(ctx.qwen_moe_llama_library.get().is_none());
            return;
        }

        let addresses = GETTERS
            .map(|getter| address(getter(&ctx).expect("tensorops-capable pipeline missing")));

        assert!(ctx.qwen_moe_llama_library.get().is_some());
        assert_eq!(
            ctx.qwen_moe_llama_library_init_count
                .load(Ordering::Relaxed),
            1
        );
        for (getter, expected_address) in GETTERS.into_iter().zip(addresses) {
            let first = getter(&ctx).expect("tensorops-capable pipeline missing");
            let second = getter(&ctx).expect("tensorops-capable pipeline missing");
            assert_eq!(address(first), expected_address);
            assert!(std::ptr::eq(first, second));
        }
        assert_eq!(
            ctx.qwen_moe_llama_library_init_count
                .load(Ordering::Relaxed),
            1
        );
    }
}

/// pm34: Version4_0(Metal 4 mpp::tensor_ops)로 MSL 을 컴파일해 pipeline 을 빌드한다.
/// capability=true 일 때만 호출할 것 — 실패 시 panic(기존 build_pipeline 정책 동일).
fn build_pipeline_v4(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &str,
    fn_name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let library = build_library_v4(device, src, fn_name);
    build_pipeline_from_library(device, &library, fn_name)
}

fn build_library_v4(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &str,
    source_name: &str,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    use objc2_metal::{MTLCompileOptions, MTLMathMode};

    let source = NSString::from_str(src);
    let options = MTLCompileOptions::new();
    options.setFastMathEnabled(false);
    options.setMathMode(MTLMathMode::Safe);
    device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .unwrap_or_else(|e| panic!("Metal: failed to compile {source_name}.metal (v4): {e:?}"))
}

fn build_pipeline_from_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    fn_name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let name = NSString::from_str(fn_name);
    let function = library
        .newFunctionWithName(&name)
        .unwrap_or_else(|| panic!("Metal: function '{fn_name}' not found"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Metal: failed to create pipeline '{fn_name}': {e:?}"))
}

/// `MetalContext` 를 빌드한다.
///
/// - device 없음 → `None` 반환 (fake fallback 없음).
/// - MSL 컴파일 / pipeline 빌드 실패 → panic.
pub fn build_metal_context() -> Option<MetalContext> {
    // pm25: KV int8 default ON(opt-out). 단 carrier(chain+attn) 전체가 활성일 때만 —
    // ctx.kv_int8=true 인데 carrier off 면 per-op f16 헬퍼(attn_decode_with_ctx 등)의
    // assert!(!ctx.kv_int8) 와 충돌한다. carrier env 중 하나라도 "0" 이면 f16 으로 후퇴.
    // (int8 KvResident 자체는 carrier 진입 시 lazy 생성이라 carrier-ineligible 모델엔 무영향.)
    let opt_out = |name: &str| std::env::var(name).as_deref() != Ok("0");
    let kv_int8 = opt_out("RNB_METAL_KV_INT8")
        && opt_out("RNB_METAL_DECODE_CHAIN")
        && opt_out("RNB_METAL_GDN_LAYER")
        && opt_out("RNB_METAL_ATTN_LAYER");
    // pm28 승격: concurrent chain default ON(opt-out). adversarial audit race 0 +
    // 27B −12.3%/9B −5.8% 무손실 + serial 은 barrier no-op(dual-path 아님).
    // RNB_METAL_CHAIN_CONCURRENT=0 으로 serial 강제(측정 비교/fallback).
    let chain_concurrent = opt_out("RNB_METAL_CHAIN_CONCURRENT");
    build_metal_context_with_opts(kv_int8, chain_concurrent)
}

/// kv_int8 만 지정(chain_concurrent off 고정). test 전용 — production 은 env 경로
/// `build_metal_context`(pm28 concurrent default ON)를 쓴다. int8 test 가 ctx 를 직접
/// 주입(env race 회피)할 때 serial+kv_int8 ctx 를 만든다.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn build_metal_context_with_kv_int8(kv_int8: bool) -> Option<MetalContext> {
    build_metal_context_with_opts(kv_int8, false)
}

/// kv_int8 + chain_concurrent 를 직접 받는 실체 생성자(env read 없음). `build_metal_context`
/// 가 env 파싱 후 호출하고, test 는 직접 호출해 serial/concurrent 양쪽 ctx 를 만든다.
pub fn build_metal_context_with_opts(
    kv_int8: bool,
    chain_concurrent: bool,
) -> Option<MetalContext> {
    let device: Retained<ProtocolObject<dyn MTLDevice>> = MTLCreateSystemDefaultDevice()?;

    let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = device
        .newCommandQueue()
        .expect("Metal: failed to create command queue");

    // pm30: coalesced(nr0=2 llama식) GEMV default ON 승격 — 27B −14.0% / 9B −11.5% ABAB winner
    // (모든 A>모든 B 겹침0, token-identical). q4k/q5k/q6k/q8_0 + GDN inline qkv 전 q4k 경로.
    // RNB_METAL_GEMV_COALESCED=0 으로 baseline(simd) opt-out(측정 비교 보존).
    let gemv_coalesced = std::env::var("RNB_METAL_GEMV_COALESCED").as_deref() != Ok("0");
    // pm80: llama.cpp Metal Q4_K/Q6_K N_SG=2/N_R0=2 variant default ON.
    // Qwen3.6 x4/d100에서 token-identical, wall 소폭 개선. opt-out:
    // RNB_METAL_GEMV_COALESCED_NSG2=0|false|off|no.
    let gemv_coalesced_nsg2 = gemv_coalesced_nsg2_enabled(gemv_coalesced);
    // pm33: prefill FFN batch GEMM 커널. default naive(−5% vs CPU). dequant-공유(shared)는
    // 측정상 naive보다 +43% 느려(병렬도 손실, dequant-bound 아님) 폐기 → opt-in 보존만.
    // 진짜 천장은 simdgroup_matrix(llama kernel_mul_mm) — 다음 axis. RNB_METAL_PREFILL_FFN_SHARED=1.
    let prefill_ffn_naive = std::env::var("RNB_METAL_PREFILL_FFN_SHARED").as_deref() != Ok("1");
    let attn_splitk_splits = attn_splitk_splits_from_env();
    let attn_splitk_min_kv = env_usize("RNB_METAL_ATTN_SPLITK_MIN_KV", 1024);

    // pm34: M5 neural accelerator(mpp::tensor_ops) capability 1회 판정(캐시).
    // Task 3 에서 GEMM_TENSOROPS_POC_SRC 로 파이프라인 build 연결. 현재(Task 2)는 None
    // (capability=true 여도) — 빌드 안전, non-M5 panic 회피.
    let tensorops_capable = crate::device::tensorops_capability_for_device(&device);
    let qwen_moe_prefill_id_matmul_requested =
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_MATMUL")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
    let qwen_moe_prefill_id_gate_up_requested =
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_GATE_UP")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
    let qwen_moe_prefill_mulmmid_v3_requested = qwen_moe_prefill_mulmmid_v3_requested();
    let qwen_moe_prefill_mulmmid_v4_requested = qwen_moe_prefill_mulmmid_v4_requested();
    let qwen_moe_prefill_q4_id_requested = qwen_moe_prefill_id_matmul_requested
        || qwen_moe_prefill_id_gate_up_requested
        || qwen_moe_prefill_mulmmid_v3_requested
        || qwen_moe_prefill_mulmmid_v4_requested;
    let qwen_moe_llama_library = OnceLock::new();
    #[cfg(test)]
    let qwen_moe_llama_library_init_count = std::sync::atomic::AtomicUsize::new(0);
    let qwen_moe_llama_id_map0_pipeline = OnceLock::new();
    let qwen_moe_llama_id_build_blocks_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q4k_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q4k_f16_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q5k_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q5k_f16_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q6k_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q6k_f16_pipeline = OnceLock::new();
    let qwen_moe_shared_mul_mm_q4k_f32_pipeline = OnceLock::new();
    let qwen_moe_shared_mul_mm_q6k_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_quantize_q8k_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline = OnceLock::new();
    let qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline = OnceLock::new();
    let qwen_moe_llama_swiglu_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_weighted_rank_reduce_f32_pipeline = OnceLock::new();
    let qwen_moe_llama_expert_order_reduce_f32_pipeline = OnceLock::new();
    let qwen_moe_chain_small_pipelines = std::array::from_fn(|_| OnceLock::new());
    let qwen_moe_chain_large_pipelines = std::array::from_fn(|_| OnceLock::new());
    let qwen_moe_chain_cast_large_slots_pipeline = OnceLock::new();
    // capability=true 일 때만 build_pipeline_v4 호출(non-M5/구형 OS panic 회피, 코덱스 B2).
    let prefill_ffn_tensorops_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_tensorops_poc",
        ))
    } else {
        None
    };
    // pm34 M6: production tensorops 커널 2개(같은 src, fn명만 다름). capability=true 일 때만.
    let gemm_q4k_tensorops_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> =
        if tensorops_capable {
            Some(build_pipeline_v4(
                &device,
                GEMM_TENSOROPS_POC_SRC,
                "gemm_q4k_tensorops",
            ))
        } else {
            None
        };
    let gemm_q6k_tensorops_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> =
        if tensorops_capable {
            Some(build_pipeline_v4(
                &device,
                GEMM_TENSOROPS_POC_SRC,
                "gemm_q6k_tensorops",
            ))
        } else {
            None
        };
    let gemm_q5k_tensorops_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> =
        if tensorops_capable {
            Some(build_pipeline_v4(
                &device,
                GEMM_TENSOROPS_POC_SRC,
                "gemm_q5k_tensorops",
            ))
        } else {
            None
        };
    // pm42 M3: production v2 GEMM(64×128 winner 타일) + cast. 같은 src, fn명만 다름. capability 시만.
    let gemm_q4k_tensorops_v2_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q4k_tensorops_v2_64x128",
        ))
    } else {
        None
    };
    let qwen_moe_prefill_gate_up_pair_requested =
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_GATE_UP_PAIR")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
    let gemm_q4k_tensorops_v2_pair_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_gate_up_pair_requested {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q4k_tensorops_v2_pair_64x128",
        ))
    } else {
        None
    };
    let gemm_q5k_tensorops_v2_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q5k_tensorops_v2_64x128",
        ))
    } else {
        None
    };
    let gemm_q6k_tensorops_v2_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q6k_tensorops_v2_64x128",
        ))
    } else {
        None
    };
    let qwen_moe_prefill_down_scatter_requested = qwen_moe_prefill_down_scatter_enabled();
    let gemm_q4k_tensorops_v2_scatter_accum_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_down_scatter_requested {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q4k_tensorops_v2_scatter_accum_64x64",
        ))
    } else {
        None
    };
    let gemm_q6k_tensorops_v2_scatter_accum_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_down_scatter_requested {
        Some(build_pipeline_v4(
            &device,
            GEMM_TENSOROPS_POC_SRC,
            "gemm_q6k_tensorops_v2_scatter_accum_64x64",
        ))
    } else {
        None
    };
    let cast_f32_f16_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> =
        if tensorops_capable {
            Some(build_pipeline_v4(
                &device,
                GEMM_TENSOROPS_POC_SRC,
                "cast_f32_to_f16",
            ))
        } else {
            None
        };
    let gemm_q4k_tensorops_id_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_q4_id_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "gemm_q4k_tensorops_id",
        ))
    } else {
        None
    };
    let gemm_q4k_tensorops_id_v2_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_q4_id_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "gemm_q4k_tensorops_id_v2_64x64",
        ))
    } else {
        None
    };
    let gemm_q4k_tensorops_id_v2_f16_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_q4_id_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "gemm_q4k_tensorops_id_v2_64x64_f16",
        ))
    } else {
        None
    };
    let qwen_moe_prefill_id_f16_64x128_requested =
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_F16_64X128")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false);
    let gemm_q4k_tensorops_id_v2_64x128_f16_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable
        && qwen_moe_prefill_q4_id_requested
        && qwen_moe_prefill_id_f16_64x128_requested
    {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "gemm_q4k_tensorops_id_v2_64x128_f16",
        ))
    } else {
        None
    };
    let gemm_q6k_tensorops_id_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_id_matmul_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "gemm_q6k_tensorops_id",
        ))
    } else {
        None
    };
    let qwen_moe_v3_q6_down_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_mulmmid_v3_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_v3_q6_down",
        ))
    } else {
        None
    };
    let qwen_moe_v3_q4_down_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_mulmmid_v3_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_v3_q4_down",
        ))
    } else {
        None
    };
    let qwen_moe_v4_q6_down_scatter_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_mulmmid_v4_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_v4_q6_down_scatter",
        ))
    } else {
        None
    };
    let qwen_moe_v4_q4_down_scatter_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_mulmmid_v4_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_v4_q4_down_scatter",
        ))
    } else {
        None
    };
    let qwen_moe_v3_token_rank_combine_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_mulmmid_v3_requested {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_v3_token_rank_combine",
        ))
    } else {
        None
    };
    let qwen_moe_id_partial_reduce_scatter_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_id_primitive_requested() {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_id_partial_reduce_scatter",
        ))
    } else {
        None
    };
    let qwen_moe_id_q4_gate_up_tile_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_id_primitive_requested() {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_id_q4_gate_up_tile",
        ))
    } else {
        None
    };
    let qwen_moe_id_silu_mul_tile_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_id_primitive_requested() {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_id_silu_mul_tile",
        ))
    } else {
        None
    };
    let qwen_moe_id_q6_down_partial_tile_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable && qwen_moe_prefill_id_primitive_requested() {
        Some(build_pipeline_v4(
            &device,
            QWEN_MOE_PREFILL_ID_MATMUL_SRC,
            "qwen_moe_id_q6_down_partial_tile",
        ))
    } else {
        None
    };
    // pm47 ② STEP4+STEP5 GEMM 커널(별 src). capability 시만.
    let delta_net_scan_chunk_step45gemm_pipeline: Option<
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = if tensorops_capable {
        Some(build_pipeline_v4(
            &device,
            DELTA_NET_SCAN_CHUNK_STEP45GEMM_SRC,
            "delta_net_scan_chunk_step45gemm",
        ))
    } else {
        None
    };
    // pm48 ① flash attention prefill 커널(simdgroup matmul2d). capability 시만.
    let flash_attn_prefill_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> =
        if tensorops_capable {
            Some(build_pipeline_v4(
                &device,
                FLASH_ATTN_PREFILL_SIMD_SRC,
                "attn_prefill_flash_simd",
            ))
        } else {
            None
        };
    let q4k_pipeline = build_pipeline(&device, GEMV_Q4K_SRC, "gemv_q4k");
    let q4k_simd_pipeline = build_pipeline(&device, GEMV_Q4K_SIMD_SRC, "gemv_q4k_simd");
    let q4k_coalesced_pipeline =
        build_pipeline(&device, GEMV_Q4K_COALESCED_SRC, "gemv_q4k_coalesced");
    let q4k_coalesced_nsg2_pipeline = build_pipeline(
        &device,
        GEMV_Q4K_COALESCED_NSG2_SRC,
        "gemv_q4k_coalesced_nsg2",
    );
    let gemm_q4k_pipeline = build_pipeline(&device, GEMM_Q4K_SRC, "gemm_q4k");
    let gemm_q4k_shared_pipeline = build_pipeline(&device, GEMM_Q4K_SHARED_SRC, "gemm_q4k_shared");
    let q6k_pipeline = build_pipeline(&device, GEMV_Q6K_SRC, "gemv_q6k");
    let q6k_simd_pipeline = build_pipeline(&device, GEMV_Q6K_SIMD_SRC, "gemv_q6k_simd");
    let q6k_coalesced_pipeline =
        build_pipeline(&device, GEMV_Q6K_COALESCED_SRC, "gemv_q6k_coalesced");
    let q6k_coalesced_nsg2_pipeline = build_pipeline(
        &device,
        GEMV_Q6K_COALESCED_NSG2_SRC,
        "gemv_q6k_coalesced_nsg2",
    );
    let gemm_q6k_pipeline = build_pipeline(&device, GEMM_Q6K_SRC, "gemm_q6k");
    let gemm_q6k_shared_pipeline = build_pipeline(&device, GEMM_Q6K_SHARED_SRC, "gemm_q6k_shared");
    let q5k_pipeline = build_pipeline(&device, GEMV_Q5K_SRC, "gemv_q5k");
    let q5k_simd_pipeline = build_pipeline(&device, GEMV_Q5K_SIMD_SRC, "gemv_q5k_simd");
    let q5k_coalesced_pipeline =
        build_pipeline(&device, GEMV_Q5K_COALESCED_SRC, "gemv_q5k_coalesced");
    let q8_0_pipeline = build_pipeline(&device, GEMV_Q8_0_SRC, "gemv_q8_0");
    let q8_0_coalesced_pipeline =
        build_pipeline(&device, GEMV_Q8_0_COALESCED_SRC, "gemv_q8_0_coalesced");
    let q8_0_mla_slots_pipeline =
        build_pipeline(&device, GEMV_Q8_0_MLA_SLOTS_SRC, "gemv_q8_0_mla_slots");
    let output_argmax_pipeline = build_pipeline(&device, OUTPUT_ARGMAX_SRC, "argmax_f32");
    let f32_chain_pipeline = build_pipeline(&device, GEMV_F32_SRC, "gemv_f32_chain");
    let f32_chain_simd_pipeline = build_pipeline(&device, GEMV_F32_SRC, "gemv_f32_chain_simd");
    let prefill_f32_proj_pipeline = build_pipeline(&device, GEMV_F32_SRC, "prefill_f32_proj");
    let rms_norm_pipeline = build_pipeline(&device, RMS_NORM_SRC, "rms_norm");
    let rms_norm_batch_pipeline = build_pipeline(&device, RMS_NORM_SRC, "rms_norm_batch");
    let gated_rmsnorm_silu_pipeline =
        build_pipeline(&device, RMS_NORM_SRC, "gated_rmsnorm_silu_batch");
    let silu_mul_pipeline = build_pipeline(&device, SILU_MUL_SRC, "silu_mul");
    let silu_mul_f16_pipeline = build_pipeline(&device, SILU_MUL_SRC, "silu_mul_to_f16");
    let silu_mul_half_f16_pipeline = build_pipeline(&device, SILU_MUL_SRC, "silu_mul_half_to_f16");
    let residual_add_pipeline = build_pipeline(&device, RESIDUAL_ADD_SRC, "residual_add");
    let qwen_moe_prefill_gather_pipeline = build_pipeline(
        &device,
        QWEN_MOE_PREFILL_SCATTER_SRC,
        "qwen_moe_prefill_gather_normed",
    );
    let qwen_moe_prefill_gather_f16_pipeline = build_pipeline(
        &device,
        QWEN_MOE_PREFILL_SCATTER_SRC,
        "qwen_moe_prefill_gather_normed_f16",
    );
    let qwen_moe_prefill_scatter_pipeline = build_pipeline(
        &device,
        QWEN_MOE_PREFILL_SCATTER_SRC,
        "qwen_moe_prefill_scatter_accum",
    );
    let qwen_moe_prefill_router_f32_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_moe_prefill_router_f32_exact",
    );
    let qwen_moe_prefill_rms_norm_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_moe_prefill_rms_norm_exact",
    );
    let qwen_gdn_prefill_rms_norm_f32_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_gdn_prefill_rms_norm_f32_exact",
    );
    let qwen_moe_prefill_shared_gate_chain_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_moe_prefill_shared_gate_chain_exact",
    );
    let qwen_prefill_l2_norm_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_prefill_l2_norm_exact",
    );
    let qwen_prefill_gated_rmsnorm_silu_chain_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_prefill_gated_rmsnorm_silu_chain_exact",
    );
    let qwen_moe_prefill_topk_pipeline = build_pipeline(
        &device,
        QWEN_MOE_PREFILL_SCATTER_SRC,
        "qwen_moe_prefill_topk_from_logits",
    );
    let qwen_moe_prefill_topk_chain_exact_pipeline = build_pipeline_safe_math(
        &device,
        QWEN_MOE_PREFILL_ROUTE_EXACT_SRC,
        "qwen_moe_prefill_topk_from_logits_chain_exact",
    );
    let qwen_moe_decode_q4k_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_q4k_down_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_q4k_pair_slots_pipeline =
        if std::env::var_os("RNB_METAL_QWEN35_MOE_GATE_UP_PAIR").is_some() {
            Some(build_pipeline(
                &device,
                QWEN_MOE_DECODE_SLOTS_SRC,
                "qwen_moe_decode_q4k_pair_slots",
            ))
        } else {
            None
        };
    let qwen_moe_decode_q5k_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_q6k_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_q4k_selected_slots_coalesced_pipeline = OnceCell::new();
    let qwen_moe_decode_q4k_selected_slots_nsg2_pipeline =
        if std::env::var_os("RNB_METAL_QWEN35_MOE_SELECTED_Q4K_NSG2").is_some() {
            Some(build_pipeline(
                &device,
                QWEN_MOE_DECODE_SLOTS_SRC,
                "qwen_moe_decode_q4k_selected_slots_nsg2",
            ))
        } else {
            None
        };
    let qwen_moe_decode_q4k_selected_pair_slots_pipeline =
        if std::env::var_os("RNB_METAL_QWEN35_MOE_SELECTED_GATE_UP_PAIR").is_some() {
            Some(build_pipeline(
                &device,
                QWEN_MOE_DECODE_SLOTS_SRC,
                "qwen_moe_decode_q4k_selected_pair_slots",
            ))
        } else {
            None
        };
    let qwen_moe_decode_q5k_selected_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_q6k_selected_slots_pipeline = OnceCell::new();
    let glm_moe_decode_iq2xxs_selected_slots_pipeline = OnceCell::new();
    let glm_moe_decode_iq3xxs_selected_slots_pipeline = OnceCell::new();
    let glm_moe_decode_iq4xs_selected_slots_pipeline = OnceCell::new();
    let glm_moe_decode_iq2s_selected_slots_pipeline = OnceCell::new();
    let (
        qwen_moe_decode_q4k_table_slots_pipeline,
        qwen_moe_decode_q5k_table_slots_pipeline,
        qwen_moe_decode_q6k_table_slots_pipeline,
        qwen_moe_decode_q4k_table_arg_encoder,
        qwen_moe_decode_q5k_table_arg_encoder,
        qwen_moe_decode_q6k_table_arg_encoder,
    ) = if std::env::var_os("RNB_METAL_QWEN35_MOE_GDN_ARG_TABLE").is_some() {
        let (q4_pipeline, q4_encoder) = build_pipeline_with_argument_encoder(
            &device,
            QWEN_MOE_DECODE_SLOTS_SRC,
            "qwen_moe_decode_q4k_table_slots",
            0,
        );
        let (q5_pipeline, q5_encoder) = build_pipeline_with_argument_encoder(
            &device,
            QWEN_MOE_DECODE_SLOTS_SRC,
            "qwen_moe_decode_q5k_table_slots",
            0,
        );
        let (q6_pipeline, q6_encoder) = build_pipeline_with_argument_encoder(
            &device,
            QWEN_MOE_DECODE_SLOTS_SRC,
            "qwen_moe_decode_q6k_table_slots",
            0,
        );
        (
            Some(q4_pipeline),
            Some(q5_pipeline),
            Some(q6_pipeline),
            Some(q4_encoder),
            Some(q5_encoder),
            Some(q6_encoder),
        )
    } else {
        (None, None, None, None, None, None)
    };
    let qwen_moe_decode_silu_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_reduce_slots_pipeline = OnceCell::new();
    let qwen_moe_decode_route_shared_pipeline = OnceCell::new();
    let attn_decode_pipeline = build_pipeline(&device, ATTN_DECODE_SRC, "attn_decode");
    let attn_decode_i8_pipeline = build_pipeline(&device, ATTN_DECODE_I8_SRC, "attn_decode_i8");
    let attn_decode_i8_splitk_part_pipeline = build_pipeline(
        &device,
        ATTN_DECODE_I8_SPLITK_SRC,
        "attn_decode_i8_splitk_part",
    );
    let attn_decode_i8_splitk_reduce_pipeline = build_pipeline(
        &device,
        ATTN_DECODE_I8_SPLITK_SRC,
        "attn_decode_i8_splitk_reduce",
    );
    let rope_mrope_pipeline = build_pipeline(&device, ROPE_MROPE_SRC, "rope_mrope");
    let qk_norm_pipeline = build_pipeline(&device, QK_NORM_SRC, "qk_norm");
    let kv_append_pipeline = build_pipeline(&device, KV_APPEND_SRC, "kv_append");
    let kv_append_i8_pipeline = build_pipeline(&device, KV_APPEND_I8_SRC, "kv_append_i8");
    let ssm_conv_silu_pipeline = build_pipeline(&device, SSM_CONV_SILU_SRC, "ssm_conv1d_silu");
    let ssm_conv_silu_batch_pipeline =
        build_pipeline(&device, SSM_CONV_SILU_SRC, "ssm_conv1d_silu_batch");
    let delta_net_step_pipeline = build_pipeline(&device, DELTA_NET_STEP_SRC, "delta_net_step");
    let delta_net_scan_chunk_pipeline =
        build_pipeline(&device, DELTA_NET_SCAN_CHUNK_SRC, "delta_net_scan_chunk");
    let delta_net_scan_ar1_pipeline =
        build_pipeline(&device, DELTA_NET_SCAN_AR_SRC, "delta_net_scan_ar1");
    let delta_net_scan_ar2_pipeline =
        build_pipeline(&device, DELTA_NET_SCAN_AR_SRC, "delta_net_scan_ar2");
    let delta_net_scan_ar4_pipeline =
        build_pipeline(&device, DELTA_NET_SCAN_AR_SRC, "delta_net_scan_ar4");
    let delta_net_scan_ar8_pipeline =
        build_pipeline(&device, DELTA_NET_SCAN_AR_SRC, "delta_net_scan_ar8");
    let l2_norm_pipeline = build_pipeline(&device, L2_NORM_SRC, "l2_norm");
    let gdn_alpha_beta_pipeline =
        build_pipeline_safe_math(&device, GDN_ALPHA_BETA_SRC, "gdn_alpha_beta");
    let split_qgate_pipeline = build_pipeline(&device, SPLIT_QGATE_SRC, "split_qgate");
    let split_conv_qkv_pipeline = build_pipeline(&device, SPLIT_CONV_QKV_SRC, "split_conv_qkv");
    let repeat_qk_pipeline = build_pipeline(&device, REPEAT_QK_SRC, "repeat_qk");
    let rope_partial_pipeline = build_pipeline(&device, ROPE_PARTIAL_SRC, "rope_partial");
    let gate_apply_pipeline = build_pipeline(&device, GATE_APPLY_SRC, "gate_apply");
    let prefill_atn_core_requested = !env_falsey("RNB_METAL_PREFILL_ATN_FULL_LAYER");
    let prefill_split_gate_pipeline = if prefill_atn_core_requested {
        Some(build_pipeline(
            &device,
            PREFILL_SPLIT_GATE_SRC,
            "prefill_split_q_gate",
        ))
    } else {
        None
    };
    let prefill_gate_apply_pipeline = if prefill_atn_core_requested {
        Some(build_pipeline(
            &device,
            PREFILL_GATE_APPLY_SRC,
            "prefill_gate_apply",
        ))
    } else {
        None
    };
    // pm48 ②: prefill qk_norm→rope device chain 부품(always build — non-tensorops 장비도 chain 외
    // 단독 사용 가능, 작은 커널). oracle 4.95e-5 검증 완료.
    let prefill_rope_qk_norm_pipeline =
        build_pipeline(&device, PREFILL_ROPE_QK_NORM_SRC, "prefill_rope_qk_norm");
    let prefill_rope_only_pipeline =
        build_pipeline_safe_math(&device, PREFILL_ROPE_QK_NORM_SRC, "prefill_rope_only");

    // pm21 승격: default on. baseline 은 RNB_METAL_GEMV_SIMD=0 으로 opt-out(측정 비교 보존).
    let gemv_simd = std::env::var("RNB_METAL_GEMV_SIMD")
        .map(|v| v != "0")
        .unwrap_or(true);
    let chain_profile = match std::env::var("RNB_METAL_CHAIN_PROFILE").as_deref() {
        Ok("gemv") => ChainProfile::Gemv,
        Ok("small") => ChainProfile::Small,
        Ok("attn") => ChainProfile::Attn,
        _ => ChainProfile::All,
    };

    // 측정 게이트(pm22): attn_decode K·V read 주소 wrap window. window(2^n)면 mask=window-1,
    // 미설정/비-2^n 이면 0xFFFFFFFF(normal). build 시 1회 buffer 생성(매 dispatch bind index 9).
    let attn_read_window = std::env::var("RNB_METAL_ATTN_READ_WINDOW")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let attn_read_mask: u32 = if attn_read_window >= 2 && attn_read_window.is_power_of_two() {
        attn_read_window - 1
    } else {
        0xFFFF_FFFF
    };
    let attn_read_mask_buf = unsafe {
        let nn = NonNull::new(&attn_read_mask as *const u32 as *mut std::ffi::c_void)
            .expect("read_mask ptr null");
        device
            .newBufferWithBytes_length_options(nn, 4, MTLResourceOptions::StorageModeShared)
            .expect("Metal: failed to create attn_read_mask buffer")
    };

    Some(MetalContext {
        device,
        queue,
        q4k_pipeline,
        q4k_simd_pipeline,
        q4k_coalesced_pipeline,
        q4k_coalesced_nsg2_pipeline,
        gemm_q4k_pipeline,
        gemm_q4k_shared_pipeline,
        q6k_pipeline,
        q6k_simd_pipeline,
        q6k_coalesced_pipeline,
        q6k_coalesced_nsg2_pipeline,
        gemm_q6k_pipeline,
        gemm_q6k_shared_pipeline,
        q5k_pipeline,
        q5k_simd_pipeline,
        q5k_coalesced_pipeline,
        q8_0_pipeline,
        q8_0_coalesced_pipeline,
        q8_0_mla_slots_pipeline,
        output_argmax_pipeline,
        f32_chain_pipeline,
        f32_chain_simd_pipeline,
        prefill_f32_proj_pipeline,
        rms_norm_pipeline,
        rms_norm_batch_pipeline,
        gated_rmsnorm_silu_pipeline,
        silu_mul_pipeline,
        silu_mul_f16_pipeline,
        silu_mul_half_f16_pipeline,
        residual_add_pipeline,
        qwen_moe_prefill_gather_pipeline,
        qwen_moe_prefill_gather_f16_pipeline,
        qwen_moe_prefill_scatter_pipeline,
        qwen_moe_prefill_topk_pipeline,
        qwen_moe_prefill_topk_chain_exact_pipeline,
        qwen_moe_prefill_router_f32_exact_pipeline,
        qwen_moe_prefill_rms_norm_exact_pipeline,
        qwen_gdn_prefill_rms_norm_f32_exact_pipeline,
        qwen_moe_prefill_shared_gate_chain_exact_pipeline,
        qwen_prefill_l2_norm_exact_pipeline,
        qwen_prefill_gated_rmsnorm_silu_chain_exact_pipeline,
        qwen_moe_decode_q4k_slots_pipeline,
        qwen_moe_decode_q4k_down_slots_pipeline,
        qwen_moe_decode_q4k_pair_slots_pipeline,
        qwen_moe_decode_q5k_slots_pipeline,
        qwen_moe_decode_q6k_slots_pipeline,
        qwen_moe_decode_q4k_selected_slots_coalesced_pipeline,
        qwen_moe_decode_q4k_selected_slots_nsg2_pipeline,
        qwen_moe_decode_q4k_selected_pair_slots_pipeline,
        qwen_moe_decode_q5k_selected_slots_pipeline,
        qwen_moe_decode_q6k_selected_slots_pipeline,
        glm_moe_decode_iq2xxs_selected_slots_pipeline,
        glm_moe_decode_iq3xxs_selected_slots_pipeline,
        glm_moe_decode_iq4xs_selected_slots_pipeline,
        glm_moe_decode_iq2s_selected_slots_pipeline,
        qwen_moe_decode_q4k_table_slots_pipeline,
        qwen_moe_decode_q5k_table_slots_pipeline,
        qwen_moe_decode_q6k_table_slots_pipeline,
        qwen_moe_decode_q4k_table_arg_encoder,
        qwen_moe_decode_q5k_table_arg_encoder,
        qwen_moe_decode_q6k_table_arg_encoder,
        qwen_moe_decode_silu_slots_pipeline,
        qwen_moe_decode_reduce_slots_pipeline,
        qwen_moe_decode_route_shared_pipeline,
        attn_decode_pipeline,
        attn_decode_i8_pipeline,
        attn_decode_i8_splitk_part_pipeline,
        attn_decode_i8_splitk_reduce_pipeline,
        rope_mrope_pipeline,
        qk_norm_pipeline,
        kv_append_pipeline,
        kv_append_i8_pipeline,
        ssm_conv_silu_pipeline,
        ssm_conv_silu_batch_pipeline,
        delta_net_step_pipeline,
        delta_net_scan_chunk_pipeline,
        delta_net_scan_ar1_pipeline,
        delta_net_scan_ar2_pipeline,
        delta_net_scan_ar4_pipeline,
        delta_net_scan_ar8_pipeline,
        l2_norm_pipeline,
        gdn_alpha_beta_pipeline,
        split_qgate_pipeline,
        split_conv_qkv_pipeline,
        repeat_qk_pipeline,
        rope_partial_pipeline,
        gate_apply_pipeline,
        prefill_split_gate_pipeline,
        prefill_gate_apply_pipeline,
        gemv_simd,
        chain_profile,
        attn_read_mask_buf,
        kv_int8,
        attn_splitk_splits,
        attn_splitk_min_kv,
        chain_concurrent,
        gemv_coalesced,
        gemv_coalesced_nsg2,
        prefill_ffn_naive,
        tensorops_capable,
        prefill_ffn_tensorops_pipeline,
        gemm_q4k_tensorops_pipeline,
        gemm_q6k_tensorops_pipeline,
        gemm_q5k_tensorops_pipeline,
        gemm_q4k_tensorops_v2_pipeline,
        gemm_q4k_tensorops_v2_pair_pipeline,
        gemm_q5k_tensorops_v2_pipeline,
        gemm_q6k_tensorops_v2_pipeline,
        gemm_q4k_tensorops_v2_scatter_accum_pipeline,
        gemm_q6k_tensorops_v2_scatter_accum_pipeline,
        gemm_q4k_tensorops_id_pipeline,
        gemm_q4k_tensorops_id_v2_pipeline,
        gemm_q4k_tensorops_id_v2_f16_pipeline,
        gemm_q4k_tensorops_id_v2_64x128_f16_pipeline,
        gemm_q6k_tensorops_id_pipeline,
        qwen_moe_llama_library,
        #[cfg(test)]
        qwen_moe_llama_library_init_count,
        qwen_moe_llama_id_map0_pipeline,
        qwen_moe_llama_id_build_blocks_pipeline,
        qwen_moe_llama_mul_mm_id_q4k_f32_pipeline,
        qwen_moe_llama_mul_mm_id_q4k_f16_pipeline,
        qwen_moe_llama_mul_mm_id_q5k_f32_pipeline,
        qwen_moe_llama_mul_mm_id_q5k_f16_pipeline,
        qwen_moe_llama_mul_mm_id_q6k_f32_pipeline,
        qwen_moe_llama_mul_mm_id_q6k_f16_pipeline,
        qwen_moe_chain_small_pipelines,
        qwen_moe_chain_large_pipelines,
        qwen_moe_chain_cast_large_slots_pipeline,
        qwen_moe_llama_quantize_q8k_f32_pipeline,
        qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline,
        qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline,
        qwen_moe_shared_mul_mm_q4k_f32_pipeline,
        qwen_moe_shared_mul_mm_q6k_f32_pipeline,
        qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline,
        qwen_moe_llama_swiglu_f32_pipeline,
        qwen_moe_llama_weighted_rank_reduce_f32_pipeline,
        qwen_moe_llama_expert_order_reduce_f32_pipeline,
        qwen_moe_v3_q6_down_pipeline,
        qwen_moe_v3_q4_down_pipeline,
        qwen_moe_v4_q6_down_scatter_pipeline,
        qwen_moe_v4_q4_down_scatter_pipeline,
        qwen_moe_v3_token_rank_combine_pipeline,
        qwen_moe_id_partial_reduce_scatter_pipeline,
        qwen_moe_id_q4_gate_up_tile_pipeline,
        qwen_moe_id_silu_mul_tile_pipeline,
        qwen_moe_id_q6_down_partial_tile_pipeline,
        cast_f32_f16_pipeline,
        delta_net_scan_chunk_step45gemm_pipeline,
        flash_attn_prefill_pipeline,
        prefill_rope_qk_norm_pipeline,
        prefill_rope_only_pipeline,
    })
}

/// chain decode 용 compute encoder. `ctx.chain_concurrent` 면 Concurrent dispatch type
/// (독립 dispatch 가 GPU 에서 overlap — 의존성은 `chain_barrier` 로 명시), 아니면 serial
/// (dispatch 간 자동 hazard tracking). pm28 concurrent dispatch lever.
pub(crate) fn chain_compute_encoder(
    ctx: &MetalContext,
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
    if ctx.chain_concurrent {
        cmd.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("Metal: concurrent compute encoder")
    } else {
        cmd.computeCommandEncoder().expect("Metal: compute encoder")
    }
}

pub(crate) fn try_chain_compute_encoder(
    ctx: &MetalContext,
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, String> {
    let encoder = if ctx.chain_concurrent {
        cmd.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
    } else {
        cmd.computeCommandEncoder()
    };
    encoder.ok_or_else(|| "Metal: failed to create Qwen prefill compute encoder".to_string())
}

/// concurrent encoder 의 dispatch 의존성 경계. `ctx.chain_concurrent` 면 buffer-scope
/// memory barrier(이 encoder 의 이전 모든 dispatch 완료 + side effect 가시화 → 이후
/// dispatch), serial 이면 no-op(자동 hazard tracking 이 처리). 독립 dispatch 그룹
/// 내부엔 호출하지 않는다(overlap 시키려는 지점).
pub(crate) fn chain_barrier(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
) {
    if ctx.chain_concurrent {
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }
}

fn qwen_moe_llama_id_counter_support(
    dispatch_boundary: bool,
    timestamp_set: bool,
) -> Result<(), &'static str> {
    if !dispatch_boundary {
        return Err("dispatch_boundary_unsupported");
    }
    if !timestamp_set {
        return Err("timestamp_counter_set_unavailable");
    }
    Ok(())
}

pub(crate) fn qwen_moe_llama_id_timestamp_counter_set(
    ctx: &MetalContext,
) -> Result<Retained<ProtocolObject<dyn MTLCounterSet>>, &'static str> {
    let dispatch_boundary = ctx
        .device
        .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary);
    let counter_sets = ctx.device.counterSets().ok_or("counter_sets_unavailable")?;
    let timestamp_name = unsafe { MTLCommonCounterSetTimestamp };
    let timestamp_set = counter_sets
        .iter()
        .find(|set| set.name().to_string() == timestamp_name.to_string());
    qwen_moe_llama_id_counter_support(dispatch_boundary, timestamp_set.is_some())?;
    timestamp_set.ok_or("timestamp_counter_set_unavailable")
}

#[cfg(test)]
mod qwen_moe_llama_id_counter_tests {
    use super::qwen_moe_llama_id_counter_support;

    #[test]
    fn requires_dispatch_boundary_and_timestamp_set() {
        assert_eq!(qwen_moe_llama_id_counter_support(true, true), Ok(()));
        assert_eq!(
            qwen_moe_llama_id_counter_support(false, true),
            Err("dispatch_boundary_unsupported")
        );
        assert_eq!(
            qwen_moe_llama_id_counter_support(true, false),
            Err("timestamp_counter_set_unavailable")
        );
    }
}

pub(crate) fn encode_rms_norm_batch(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    cols_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) {
    enc.setComputePipelineState(&ctx.rms_norm_batch_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(cols_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 4);
    }
    let grid = MTLSize {
        width: rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// gemv_q4k_with_ctx — 캐싱된 컨텍스트로 per-call 버퍼 alloc + encode + dispatch
// ---------------------------------------------------------------------------

/// Q4_K GEMV (복사 경로): weight 를 GPU 버퍼로 복사 업로드 후 dispatch (offset = 0).
/// zero-copy 경로는 `wrap_nocopy` + `gemv_q4k_dispatch` 참조.
pub fn gemv_q4k_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );

    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemv_q4k_dispatch(ctx, &w_buf, 0, input, n, k)
}

/// mmap weight 포인터를 page-align 해 NoCopy MTLBuffer 로 wrap (복사 0).
/// `raw` 는 mmap 내부 포인터(loader 소유) — deallocator=None 이라 메모리 해제 안 함.
/// 반환: `(NoCopy buffer, page_offset)`. page_offset 은 dispatch 의 `weight_byte_offset`.
pub(crate) fn wrap_nocopy(
    ctx: &MetalContext,
    raw: &[u8],
) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u32) {
    let (aligned, page_off, buf_len) = page_align(raw.as_ptr() as usize, raw.len());
    let shared = MTLResourceOptions::StorageModeShared;
    let buf = unsafe {
        let ptr = NonNull::new(aligned as *mut std::ffi::c_void).expect("aligned ptr is null");
        ctx.device
            .newBufferWithBytesNoCopy_length_options_deallocator(ptr, buf_len, shared, None)
            .expect("Metal: failed to create NoCopy weight buffer")
    };
    (buf, page_off as u32)
}

/// per-slot symmetric int8 양자화 reference (CPU ground truth — metal 커널이 1:1 emulate).
/// scale = max|v|/127, q = clamp(round_ties_even(v/scale), -127, 127).
/// max_abs==0 이면 scale=0, q=0 (zero slot).
pub(crate) fn quantize_slot_i8_ref(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if max_abs == 0.0 {
        return (vec![0i8; v.len()], 0.0);
    }
    let scale = max_abs / 127.0;
    let q = v
        .iter()
        .map(|&x| (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// int8 slot dequant reference. q * scale 로 f32 복원. (test 전용 — production 은 GPU dequant)
#[cfg(test)]
pub(crate) fn dequant_slot_i8_ref(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&x| x as f32 * scale).collect()
}

/// u16 f16 bits → f32 (ensure_filled_i8 host 변환용).
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// weight MTLBuffer + page offset 으로 input/output/scalar alloc + encode + dispatch + readback.
/// weight buffer 처리(복사 or NoCopy wrap)는 caller 담당 — 캐싱/zero-copy 무관 공유 경로.
/// `RNB_METAL_TIMING=1` 시 input_alloc/dispatch/readback 단계 시간을 stderr 에 출력.
pub(crate) fn gemv_q4k_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q4_K block size)");
    assert_eq!(input.len(), k);
    let timing = timing_enabled();
    let shared = MTLResourceOptions::StorageModeShared;

    // (a) input/output/scalar buffer alloc (weight 는 caller 가 이미 준비 — zero-copy 면 복사 0)
    let t0 = std::time::Instant::now();
    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let t_alloc = t0.elapsed();

    // (b) encode, dispatch & wait
    let t1 = std::time::Instant::now();
    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.q4k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0); // weight_bytes (page-aligned base)
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1); // input
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2); // out
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3); // N
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4); // K
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5); // weight_byte_offset
    }
    let tg_width = ctx.q4k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    let t_encode = t1.elapsed(); // CPU: encoder 생성~endEncoding (launch overhead)
    let tc = std::time::Instant::now();
    cmd.commit();
    let t_commit = tc.elapsed(); // CPU: GPU 큐 submit
    let tw = std::time::Instant::now();
    cmd.waitUntilCompleted();
    let t_wait = tw.elapsed(); // GPU 실행 + 동기 대기
    let t_dispatch = t1.elapsed();

    // (c) readback
    let t2 = std::time::Instant::now();
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    let result = out_slice.to_vec();
    let t_readback = t2.elapsed();

    if timing {
        eprintln!(
            "[metal-timing] gemv_q4k_dispatch N={n} K={k}: input_alloc={:.2} \
             encode={:.2} commit={:.2} wait={:.2} dispatch={:.2} readback={:.2}ms",
            t_alloc.as_secs_f64() * 1000.0,
            t_encode.as_secs_f64() * 1000.0,
            t_commit.as_secs_f64() * 1000.0,
            t_wait.as_secs_f64() * 1000.0,
            t_dispatch.as_secs_f64() * 1000.0,
            t_readback.as_secs_f64() * 1000.0,
        );
    }
    result
}

/// pm112: GLM MLA decode dense GEMV 의 quant 종별.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MlaGemvQuant {
    Q5K,
    Q8_0,
}

/// pm112: GLM MLA decode 용 단발 GEMV dispatch (Q5_K | Q8_0).
/// resident wrap 된 weight buffer + page offset, input 업로드 → 단일 command
/// buffer → 동기 readback. 커널 선택은 quant 별 auto encode 를 따른다.
pub(crate) fn gemv_mla_dispatch(
    ctx: &MetalContext,
    quant: MlaGemvQuant,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    match quant {
        MlaGemvQuant::Q5K => assert_eq!(k % 256, 0, "Q5_K K must be a multiple of 256"),
        MlaGemvQuant::Q8_0 => assert_eq!(k % 32, 0, "Q8_0 K must be a multiple of 32"),
    }
    assert_eq!(input.len(), k);
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = std::mem::size_of_val(input);
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(n * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&weight_byte_offset);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    match quant {
        MlaGemvQuant::Q5K => encode_gemv_q5k_auto(
            ctx, &enc, w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, n,
        ),
        MlaGemvQuant::Q8_0 => encode_gemv_q8_0(
            ctx, &enc, w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, n,
        ),
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out.to_vec()
}

/// pm112: q8_0 GEMV 를 buffer 바인딩 오프셋으로 encode. GLM MLA per-head 배치
/// (k_b/v_b 64 heads) 에서 weight/input/out 을 head 별 오프셋으로 묶는다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q8_0_at(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    w_byte_offset: usize,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    in_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_byte_offset: usize,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    zero_off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    let pipeline = if ctx.gemv_coalesced {
        &ctx.q8_0_coalesced_pipeline
    } else {
        &ctx.q8_0_pipeline
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), w_byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), in_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), out_byte_offset, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(zero_off_buf), 0, 5);
    }
    let (grid_width, tg_width) = if ctx.gemv_coalesced {
        (n.div_ceil(2), SIMD_WIDTH)
    } else {
        let tg_width = ctx.q8_0_pipeline.threadExecutionWidth().max(1);
        (n.div_ceil(tg_width), tg_width)
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_width,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
}

/// pm112: GLM MLA per-head Q8_0 GEMV 배치. head 마다 weight/input/out 오프셋을
/// 바꿔 단일 command buffer 에 `heads` 개 GEMV 를 encode 한다 (commit/wait 1회).
/// input 은 `[heads * k]` 로 packing 된 f32, 반환은 `[heads * n_per_head]`.
pub(crate) fn glm_mla_head_gemv_q8_0_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    page_off: u32,
    input: &[f32],
    heads: usize,
    n_per_head: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 32, 0, "Q8_0 K must be a multiple of 32");
    assert_eq!(input.len(), heads * k);
    // Q8_0 블록 = 2B(f16 d) + 32B(q8) = 34B/32값.
    let w_head_bytes = n_per_head * (k / 32) * 34;
    let in_head_bytes = k * std::mem::size_of::<f32>();
    let out_head_bytes = n_per_head * std::mem::size_of::<f32>();
    assert_eq!(w_head_bytes % 4, 0);
    assert_eq!(page_off as usize % 4, 0);
    let shared = MTLResourceOptions::StorageModeShared;

    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, std::mem::size_of_val(input), shared)
            .expect("Metal: failed to create input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(heads * out_head_bytes, shared)
        .expect("Metal: failed to create output buffer");
    let n_u32 = n_per_head as u32;
    let k_u32 = k as u32;
    let zero_u32 = 0u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let zero_off_buf = mk_u32(&zero_u32);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    for head in 0..heads {
        encode_gemv_q8_0_at(
            ctx,
            &enc,
            w_buf,
            page_off as usize + head * w_head_bytes,
            &i_buf,
            head * in_head_bytes,
            &o_buf,
            head * out_head_bytes,
            &n_buf,
            &k_buf,
            &zero_off_buf,
            n_per_head,
        );
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, heads * n_per_head) };
    out.to_vec()
}

/// pm113: GLM MLA prefill slot-batch GEMV. input `[slots][k]` (slot = token*heads+head),
/// out `[slots][n_per_head]`. weight head 선택은 커널 내부 `slot % heads` — 단일 dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mla_head_gemv_q8_0_slots_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    page_off: u32,
    input: &[f32],
    slots: usize,
    heads: usize,
    n_per_head: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 32, 0, "Q8_0 K must be a multiple of 32");
    assert_eq!(input.len(), slots * k);
    let shared = MTLResourceOptions::StorageModeShared;

    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, std::mem::size_of_val(input), shared)
            .expect("Metal: failed to create input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(slots * n_per_head * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");
    let n_u32 = n_per_head as u32;
    let k_u32 = k as u32;
    let off_u32 = page_off;
    let heads_u32 = heads as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let heads_buf = mk_u32(&heads_u32);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&ctx.q8_0_mla_slots_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&heads_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_per_head.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, slots * n_per_head) };
    out.to_vec()
}

/// Q4_K GEMV 를 device buffer in/out 으로 compute encoder 에 encode (commit/readback 안 함).
/// in_buf(K f32) → out_buf(N f32). weight/n/k/off scalar buffer 는 caller 준비.
/// chain 안에서 normed_dev → gate_dev 처럼 device-resident 연결에 사용.
pub(crate) fn encode_gemv_q4k(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.q4k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.q4k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// pm34 M2 go/no-go: tensorops staging matmul2d PoC.
// ---------------------------------------------------------------------------

/// pm34 M2: tensorops staging matmul2d PoC dispatch. A[16x32] f32 · B[32x16] f32 -> C[16x16] f32.
/// capability=true 전제(pipeline Some). 셰이더가 A/B 를 threadgroup half 로 staging 후 matmul2d.
/// go/no-go 검증 전용 — milestone 4 에서 실 Q4_K 커널로 대체 예정.
#[allow(dead_code)]
pub fn run_tensorops_poc(ctx: &MetalContext, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), 16 * 32, "A must be 16x32");
    assert_eq!(b.len(), 32 * 16, "B must be 32x16");
    let pipeline = ctx
        .prefill_ffn_tensorops_pipeline
        .as_ref()
        .expect("tensorops pipeline not built (capability=false?)");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |data: &[f32]| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("ptr null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of_val(data), shared)
                .expect("Metal: failed to create input buffer")
        }
    };
    let a_buf = mk(a);
    let b_buf = mk(b);
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(16 * 16 * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*a_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*b_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
    }
    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, 16 * 16) };
    out.to_vec()
}

/// pm34 M3: synthetic Q4_K dequant 좌표 검증 PoC. weight(16 rows×144B) + input(16×256 f32)
/// → C[16×16] (= out[tok*16+row], cpu_q4k_gemm_reference 매핑). 커널 안 Q4_K dequant→staging
/// transposed → matmul2d. 파이프라인 즉석 빌드(PoC 1회). go/no-go 후 milestone 4 실커널로 대체.
#[allow(dead_code)]
pub fn run_tensorops_poc_q4k(ctx: &MetalContext, wb: &[u8], input: &[f32]) -> Vec<f32> {
    assert_eq!(wb.len(), 16 * 144, "weight must be 16 rows × 144 bytes");
    assert_eq!(input.len(), 16 * 256, "input must be 16×256");
    let pipeline = build_pipeline_v4(
        &ctx.device,
        GEMM_TENSOROPS_POC_SRC,
        "gemm_tensorops_poc_q4k",
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(wb.as_ptr() as *mut std::ffi::c_void).expect("wb ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, wb.len(), shared)
            .expect("Metal: weight buffer")
    };
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, std::mem::size_of_val(input), shared)
            .expect("Metal: input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(16 * 16 * std::mem::size_of::<f32>(), shared)
        .expect("Metal: output buffer");

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
    }
    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, 16 * 16) };
    out.to_vec()
}

/// pm34 M4: 임의 K(256 배수) K-loop multiply_accumulate 검증. weight(16 rows × nb*144) +
/// input(16 × K) → C[16×16] (= cpu_q4k_gemm_reference, n=m=16). C 는 host zero-init.
#[allow(dead_code)]
pub fn run_tensorops_poc_q4k_ktile(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(wb.len(), 16 * num_blocks * 144, "weight bytes mismatch");
    assert_eq!(input.len(), 16 * k, "input must be 16×K");
    let pipeline = build_pipeline_v4(
        &ctx.device,
        GEMM_TENSOROPS_POC_SRC,
        "gemm_q4k_tensorops_mn16",
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let w_buf = mk_bytes(wb.as_ptr() as *const _, wb.len());
    let i_buf = mk_bytes(input.as_ptr() as *const _, std::mem::size_of_val(input));
    // C zero-init (multiply_accumulate 누적 대상)
    let zeros = vec![0f32; 16 * 16];
    let o_buf = mk_bytes(
        zeros.as_ptr() as *const _,
        std::mem::size_of_val(&zeros[..]),
    );
    let k_u32 = k as u32;
    let k_buf = mk_bytes(&k_u32 as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 3);
    }
    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, 16 * 16) };
    out.to_vec()
}

/// pm46 Phase 1: GDN delta scan STEP5 outer product microbench (R2 게이트).
/// `Us`[nh][C×HV], `Kk`[nh][C×HK] → `C`[nh][HV×HK] (= Σ_j Us[j,vi]·Kk[j,ki]). HV=HK=128 고정.
/// pm48 prefill flash attention microbench 커널(ROI 가늠용).
/// naive lane-parallel: grid=(num_heads, seq_len), 1 threadgroup=(head h, query qi), 32 lane.
/// `attn_decode` 를 query 축으로 확장. causal global_pos=(kv_len-seq_len)+qi, GQA(kv_h=h/hpg),
/// head_dim<=256. branched online softmax(decode 패턴, f16 V acc). acc 는 lane register(head256
/// threadgroup 무관 — 리뷰 HIGH-1 device workspace 강제 회피).
#[cfg(test)]
const FLASH_ATTN_PREFILL_MICROBENCH_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void attn_prefill_flash_naive(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        out          [[buffer(3)]],
    constant uint&       num_heads    [[buffer(4)]],
    constant uint&       num_kv_heads [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       kv_len       [[buffer(7)]],
    constant uint&       seq_len      [[buffer(8)]],
    constant float&      scale        [[buffer(9)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint h = gid.x;
    uint qi = gid.y;
    if (h >= num_heads || qi >= seq_len) return;
    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_off = (qi * num_heads + h) * head_dim;
    uint global_pos = (kv_len - seq_len) + qi;
    float qf[8];
    half acc[8];
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        qf[nloc] = (float)(half)q[q_off + d];
        acc[nloc] = (half)0.0f;
        nloc++;
    }
    float m = -INFINITY;
    float s = 0.0f;
    for (uint j = 0u; j <= global_pos; j++) {
        uint kv_off = j * kv_dim + kv_h * head_dim;
        float partial = 0.0f;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            float kf = (float)as_type<half>(k_cache[kv_off + d]);
            partial += qf[idx] * kf;
            idx++;
        }
        float x = simd_sum(partial) * scale;
        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) s *= alpha;
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                if (rescale) a *= alpha;
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = (half)(a + vv);
                idx++;
            }
            s += 1.0f;
            m = x;
        } else {
            float p = exp(x - m);
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = (half)(a + vv * p);
                idx++;
            }
            s += p;
        }
    }
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    uint out_off = (qi * num_heads + h) * head_dim;
    uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        out[out_off + d] = (float)acc[idx] * inv_s;
        idx++;
    }
}
"#;

/// pm48 simdgroup matmul2d flash attention prefill 커널 (go/no-go 게이트).
/// 1 threadgroup = (head h, query block of BR rows). execution_simdgroups<4>=128 lane.
/// QK^T(M=BR,N=BC,K=HD) + online softmax(f32 m/l/acc rescale) + P·V(M=BR,N=HD,K=BC) 둘 다 matmul2d.
/// Q/K/V/P/S staging 은 device half/f32 workspace(threadgroup 32KiB 초과 회피, step45gemm 패턴).
/// matmul2d descriptor 는 컴파일타임 상수라 BR/BC/HD 고정(BR=32,BC=64,HD=256, 전부 16배수).
/// pm48: production const(seam `metal_prefill_attention_flash_if_supported` 가 ctx 캐시 pipeline 사용).
const FLASH_ATTN_PREFILL_SIMD_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint BR = 32u;   // query block rows
constant constexpr uint BC = 64u;   // kv block cols
constant constexpr uint HD = 256u;  // head_dim (27B Qwen3.6 고정)
constant constexpr uint TG = 128u;  // threads per threadgroup (execution_simdgroups<4>)

// 1 threadgroup = (h, query-block). grid=(num_heads, ceil(seq/BR)). tg=128(simdgroups<4>).
//  device workspace 는 threadgroup linear index(tg_lin=h*num_qblk+qb) 로 분리.
//  - q_half  [n_tg * BR * HD]   : 이 block 의 Q rows (f16, [BR][HD] row-major)
//  - k_half  [n_tg * BC * HD]   : 현재 KV block 의 K (f16 [BC][HD])
//  - v_half  [n_tg * BC * HD]   : 현재 KV block 의 V (f16 [BC][HD])
//  - s_dev   [n_tg * BR * BC]   : QK^T 결과 (f32, [BR][BC] row-major, C[M=r][N=c])
//  - p_half  [n_tg * BR * BC]   : P=exp(S-m) (f16, [BR][BC] 동일 layout)
//  - pv_dev  [n_tg * BR * HD]   : P·V block 결과 (f32, [BR][HD] row-major, C[M=r][N=d])
//  - acc_dev [n_tg * BR * HD]   : running output accumulator (f32 [BR][HD], block 간 rescale)
//  m[BR], l[BR] 은 threadgroup(작음, f32).
kernel void attn_prefill_flash_simd(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        out          [[buffer(3)]],
    constant uint&       num_heads    [[buffer(4)]],
    constant uint&       num_kv_heads [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       kv_len       [[buffer(7)]],
    constant uint&       seq_len      [[buffer(8)]],
    constant float&      scale        [[buffer(9)]],
    constant uint&       num_qblk     [[buffer(10)]],
    device half*         q_half       [[buffer(11)]],
    device half*         k_half       [[buffer(12)]],
    device half*         v_half       [[buffer(13)]],
    device float*        s_dev        [[buffer(14)]],
    device half*         p_half       [[buffer(15)]],
    device float*        pv_dev       [[buffer(16)]],
    device float*        acc_dev      [[buffer(17)]],
    uint2 gid     [[threadgroup_position_in_grid]],
    uint2 lane2   [[thread_position_in_threadgroup]])
{
    uint lane = lane2.x;
    const uint tg_size = TG;
    uint h   = gid.x;
    uint qb  = gid.y;            // query block index
    if (h >= num_heads || qb >= num_qblk) return;
    uint tg_lin  = h * num_qblk + qb;
    uint q_base  = tg_lin * BR * HD;
    uint kv_base = tg_lin * BC * HD;
    uint s_base  = tg_lin * BR * BC;
    uint pv_base = tg_lin * BR * HD;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h   = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_row0 = qb * BR;       // global first query row

    threadgroup float m[BR];
    threadgroup float l[BR];

    // init m/l/acc.
    for (uint r = lane; r < BR; r += tg_size) { m[r] = -INFINITY; l[r] = 0.0f; }
    for (uint p = lane; p < BR * HD; p += tg_size) acc_dev[pv_base + p] = 0.0f;
    // load Q block (rows beyond seq_len → 0; matmul reads them but masked out at store).
    for (uint p = lane; p < BR * HD; p += tg_size) {
        uint r = p / HD, d = p % HD;
        uint gq = q_row0 + r;
        half val = (half)0;
        if (gq < seq_len) val = (half)q[(gq * num_heads + h) * HD + d];
        q_half[q_base + p] = val;
    }
    threadgroup_barrier(mem_flags::mem_device);

    uint nblk = (kv_len + BC - 1u) / BC;
    for (uint kb = 0u; kb < nblk; kb++) {
        uint kc0 = kb * BC;          // global first kv col of this block
        // causal block skip: 이 query block 의 최대 global pos < 이 kv block 의 최소 pos → future, 종료.
        uint q_max_global = (kv_len - seq_len) + min(q_row0 + BR - 1u, seq_len - 1u);
        if (kc0 > q_max_global) break;

        // load K/V block.
        for (uint p = lane; p < BC * HD; p += tg_size) {
            uint c = p / HD, d = p % HD;
            uint gk = kc0 + c;
            half kval = (half)0, vval = (half)0;
            if (gk < kv_len) {
                uint off = gk * kv_dim + kv_h * head_dim + d;
                kval = as_type<half>(k_cache[off]);
                vval = as_type<half>(v_cache[off]);
            }
            k_half[kv_base + p] = kval;
            v_half[kv_base + p] = vval;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // S[BR×BC] = Q·K^T → S[r][c] = sum_d Q[r][d]*K[c][d].  (matmul2d 규약: C[M][N]=sum_K A[M][K]*B[K][N];
        // step45gemm 검증 규약 = standard, transA/B=false 이면 A=[M][K]/B=[K][N] row-major).  M=BR(r),N=BC(c),K=HD(d).
        //   A=Q [BR][HD] row-major → extents(inner=HD,outer=BR), transA=false → A[r][d]=Q[r][d]. ✓
        //   B 는 B[K=d][N=c]=K[c][d] 필요 → K [BC][HD] (inner=HD,outer=BC) + transB=true 로 [HD][BC] 전치. ✓
        //   C=S [BR][BC] row-major → extents(inner=BC,outer=BR), transC=false → s_dev[r*BC+c]. ✓
        {
            auto A = tensor<device half, dextents<int32_t,2>, tensor_inline>(
                q_half + q_base, dextents<int32_t,2>(HD, BR));
            auto B = tensor<device half, dextents<int32_t,2>, tensor_inline>(
                k_half + kv_base, dextents<int32_t,2>(HD, BC));
            auto C = tensor<device float, dextents<int32_t,2>, tensor_inline>(
                s_dev + s_base, dextents<int32_t,2>(BC, BR));
            constexpr auto desc = matmul2d_descriptor(
                BR, BC, HD, false, true, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;
            op.run(A, B, C);
        }
        threadgroup_barrier(mem_flags::mem_device);
        // s_dev[r*BC + c] = S(row r, col c).

        // online softmax: per row r, scan this block's BC cols (f32 m/l/acc).
        for (uint r = lane; r < BR; r += tg_size) {
            uint gq = q_row0 + r;
            float global_pos = (gq < seq_len) ? (float)((kv_len - seq_len) + gq) : -1.0f;
            float m_old = m[r];
            float m_new = m_old;
            // pass 1: block-local new max (scale + causal mask).
            for (uint c = 0u; c < BC; c++) {
                uint gk = kc0 + c;
                if (gk >= kv_len) break;
                bool masked = (gq >= seq_len) || ((float)gk > global_pos);
                if (masked) continue;
                float x = s_dev[s_base + r * BC + c] * scale;
                if (x > m_new) m_new = x;
            }
            float alpha = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;
            // rescale running sum + acc.
            l[r] *= alpha;
            for (uint d = 0u; d < HD; d++) acc_dev[pv_base + r * HD + d] *= alpha;
            // pass 2: p = exp(x - m_new), accumulate l, write p_half (masked → 0).
            float lsum = 0.0f;
            for (uint c = 0u; c < BC; c++) {
                uint gk = kc0 + c;
                float pval = 0.0f;
                if (gk < kv_len) {
                    bool masked = (gq >= seq_len) || ((float)gk > global_pos);
                    if (!masked) {
                        float x = s_dev[s_base + r * BC + c] * scale;
                        pval = exp(x - m_new);
                        lsum += pval;
                    }
                }
                p_half[s_base + r * BC + c] = (half)pval;  // [BR][BC] same layout as s_dev
            }
            l[r] += lsum;
            m[r] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // PV[BR×HD] = P·V → O[r][d] = sum_c P[r][c]*V[c][d].  M=BR(r),N=HD(d),K=BC(c).
        //   A=P [BR][BC] row-major → extents(inner=BC,outer=BR), transA=false → A[r][c]=P(r,c). ✓
        //   B=V [BC][HD] row-major → extents(inner=HD,outer=BC), transB=false → B[c][d]=V[c][d]. ✓
        //   C [BR][HD] row-major → extents(inner=HD,outer=BR), transC=false → pv_dev[r*HD+d]. ✓
        {
            auto A = tensor<device half, dextents<int32_t,2>, tensor_inline>(
                p_half + s_base, dextents<int32_t,2>(BC, BR));
            auto B = tensor<device half, dextents<int32_t,2>, tensor_inline>(
                v_half + kv_base, dextents<int32_t,2>(HD, BC));
            auto C = tensor<device float, dextents<int32_t,2>, tensor_inline>(
                pv_dev + pv_base, dextents<int32_t,2>(HD, BR));
            constexpr auto desc = matmul2d_descriptor(
                BR, HD, BC, false, false, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;
            op.run(A, B, C);
        }
        threadgroup_barrier(mem_flags::mem_device);
        // pv_dev[r*HD + d] = PV(row r, dim d).

        // acc += pv (acc already rescaled by alpha above).
        for (uint p = lane; p < BR * HD; p += tg_size) {
            acc_dev[pv_base + p] += pv_dev[pv_base + p];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // normalize + store.
    for (uint p = lane; p < BR * HD; p += tg_size) {
        uint r = p / HD, d = p % HD;
        uint gq = q_row0 + r;
        if (gq >= seq_len) continue;
        float inv = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
        out[(gq * num_heads + h) * HD + d] = acc_dev[pv_base + r * HD + d] * inv;
    }
}
"#;

/// pm48 ②: flash attention prefill 을 device buffer in/out 으로 encoder 에 encode(chain 부품).
/// q[seq*nh*hd] f32 device, k/v[kv*nkv*hd] f16 device, out[seq*nh*hd] f32 device. workspace 7개는
/// carrier 가 n_tg(=nh*ceil(seq/BR)) 기준으로 1회 alloc. scalar buffer 도 carrier 잔류.
/// `prefill_flash_attention_with_ctx` 와 동일 dispatch geometry/커널(BR=32,BC=64,HD=256,tg=128).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_flash_attn_prefill(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    nkv_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_buf: &ProtocolObject<dyn MTLBuffer>,
    scale_buf: &ProtocolObject<dyn MTLBuffer>,
    nqb_buf: &ProtocolObject<dyn MTLBuffer>,
    q_ws: &ProtocolObject<dyn MTLBuffer>,
    k_ws: &ProtocolObject<dyn MTLBuffer>,
    v_ws: &ProtocolObject<dyn MTLBuffer>,
    s_ws: &ProtocolObject<dyn MTLBuffer>,
    p_ws: &ProtocolObject<dyn MTLBuffer>,
    pv_ws: &ProtocolObject<dyn MTLBuffer>,
    acc_ws: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    num_qblk: usize,
) {
    let pipeline = ctx
        .flash_attn_prefill_pipeline
        .as_ref()
        .expect("flash_attn_prefill_pipeline must be built (tensorops_capable)");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(nkv_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(kv_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(seq_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(scale_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(nqb_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(q_ws), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_ws), 0, 12);
        enc.setBuffer_offset_atIndex(Some(v_ws), 0, 13);
        enc.setBuffer_offset_atIndex(Some(s_ws), 0, 14);
        enc.setBuffer_offset_atIndex(Some(p_ws), 0, 15);
        enc.setBuffer_offset_atIndex(Some(pv_ws), 0, 16);
        enc.setBuffer_offset_atIndex(Some(acc_ws), 0, 17);
    }
    let grid = MTLSize {
        width: num_heads,
        height: num_qblk,
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm48 production simdgroup matmul2d flash attention prefill dispatch (host 입출력 seam, 1차).
/// q[seq*nh*hd] f32, k/v[kv*nkv*hd] f16 bits → out[seq*nh*hd] f32. dense causal GQA 전용.
/// pipeline 은 ctx 캐시(`flash_attn_prefill_pipeline`)를 재사용(즉석 build 안 함). 호출 전
/// 호출자(seam)가 `head_dim==256 + nh%nkv==0 + tensorops_capable + pipeline.is_some()` 보장.
/// grid=(num_heads, ceil(seq/BR)), tg=128. device workspace(half/f32) 는 threadgroup 당 분리.
/// 반환 (out, GPU-time ms).
#[allow(clippy::too_many_arguments)]
pub fn prefill_flash_attention_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, f64) {
    assert_eq!(q.len(), seq_len * num_heads * head_dim, "q len");
    assert_eq!(k_f16.len(), kv_len * num_kv_heads * head_dim, "k len");
    assert_eq!(v_f16.len(), kv_len * num_kv_heads * head_dim, "v len");
    assert_eq!(
        head_dim, 256,
        "simd flash 커널은 HD=256 고정(matmul2d 컴파일타임 상수)"
    );
    const BR: usize = 32;
    const BC: usize = 64;
    const HD: usize = 256;
    let num_qblk = seq_len.div_ceil(BR);
    let n_tg = num_heads * num_qblk;
    let pipeline = ctx
        .flash_attn_prefill_pipeline
        .as_ref()
        .expect("flash_attn_prefill_pipeline must be built (tensorops_capable)")
        .clone();
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let mk_zeroed = |len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        ctx.device
            .newBufferWithLength_options(len, shared)
            .expect("Metal: workspace buffer")
    };
    let q_buf = mk(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk(k_f16.as_ptr() as *const _, std::mem::size_of_val(k_f16));
    let v_buf = mk(v_f16.as_ptr() as *const _, std::mem::size_of_val(v_f16));
    let o_buf = mk_zeroed(seq_len * num_heads * head_dim * std::mem::size_of::<f32>());
    // device workspace per threadgroup (half = 2 bytes, f32 = 4 bytes).
    let q_ws = mk_zeroed(n_tg * BR * HD * 2);
    let k_ws = mk_zeroed(n_tg * BC * HD * 2);
    let v_ws = mk_zeroed(n_tg * BC * HD * 2);
    let s_ws = mk_zeroed(n_tg * BR * BC * 4);
    let p_ws = mk_zeroed(n_tg * BR * BC * 2);
    let pv_ws = mk_zeroed(n_tg * BR * HD * 4);
    let acc_ws = mk_zeroed(n_tg * BR * HD * 4);
    let nh_u = num_heads as u32;
    let nkv_u = num_kv_heads as u32;
    let hd_u = head_dim as u32;
    let kv_u = kv_len as u32;
    let seq_u = seq_len as u32;
    let nqb_u = num_qblk as u32;
    let nh_buf = mk(&nh_u as *const u32 as *const _, 4);
    let nkv_buf = mk(&nkv_u as *const u32 as *const _, 4);
    let hd_buf = mk(&hd_u as *const u32 as *const _, 4);
    let kv_buf = mk(&kv_u as *const u32 as *const _, 4);
    let seq_buf = mk(&seq_u as *const u32 as *const _, 4);
    let scale_buf = mk(&scale as *const f32 as *const _, 4);
    let nqb_buf = mk(&nqb_u as *const u32 as *const _, 4);
    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*nkv_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&*hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&*kv_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&*seq_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&*scale_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(&*nqb_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(&*q_ws), 0, 11);
        enc.setBuffer_offset_atIndex(Some(&*k_ws), 0, 12);
        enc.setBuffer_offset_atIndex(Some(&*v_ws), 0, 13);
        enc.setBuffer_offset_atIndex(Some(&*s_ws), 0, 14);
        enc.setBuffer_offset_atIndex(Some(&*p_ws), 0, 15);
        enc.setBuffer_offset_atIndex(Some(&*pv_ws), 0, 16);
        enc.setBuffer_offset_atIndex(Some(&*acc_ws), 0, 17);
    }
    let grid = MTLSize {
        width: num_heads,
        height: num_qblk,
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe {
        std::slice::from_raw_parts(
            contents.as_ptr() as *const f32,
            seq_len * num_heads * head_dim,
        )
    };
    (out.to_vec(), gpu_ms)
}

/// pm48 simd flash microbench(테스트 자산). production `prefill_flash_attention_with_ctx` 위임.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn run_prefill_flash_microbench_simd(
    ctx: &MetalContext,
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, f64) {
    prefill_flash_attention_with_ctx(
        ctx,
        q,
        k_f16,
        v_f16,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
}

/// pm48 prefill flash microbench dispatch. q[seq*nh*hd] f32, k/v[kv_len*nkv*hd] f16 bits.
/// 반환 (out[seq*nh*hd] f32, 순수 GPU-time ms). branched online softmax(attn_decode 등가).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn run_prefill_flash_microbench(
    ctx: &MetalContext,
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, f64) {
    assert_eq!(q.len(), seq_len * num_heads * head_dim, "q len");
    assert_eq!(k_f16.len(), kv_len * num_kv_heads * head_dim, "k len");
    assert_eq!(v_f16.len(), kv_len * num_kv_heads * head_dim, "v len");
    assert!(head_dim <= 256, "head_dim<=256 (lane*8)");
    let pipeline = build_pipeline_v4(
        &ctx.device,
        FLASH_ATTN_PREFILL_MICROBENCH_SRC,
        "attn_prefill_flash_naive",
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let q_buf = mk(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk(k_f16.as_ptr() as *const _, std::mem::size_of_val(k_f16));
    let v_buf = mk(v_f16.as_ptr() as *const _, std::mem::size_of_val(v_f16));
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(
            seq_len * num_heads * head_dim * std::mem::size_of::<f32>(),
            shared,
        )
        .expect("Metal: output buffer");
    let nh_u = num_heads as u32;
    let nkv_u = num_kv_heads as u32;
    let hd_u = head_dim as u32;
    let kv_u = kv_len as u32;
    let seq_u = seq_len as u32;
    let nh_buf = mk(&nh_u as *const u32 as *const _, 4);
    let nkv_buf = mk(&nkv_u as *const u32 as *const _, 4);
    let hd_buf = mk(&hd_u as *const u32 as *const _, 4);
    let kv_buf = mk(&kv_u as *const u32 as *const _, 4);
    let seq_buf = mk(&seq_u as *const u32 as *const _, 4);
    let scale_buf = mk(&scale as *const f32 as *const _, 4);
    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*nkv_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&*hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&*kv_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&*seq_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&*scale_buf), 0, 9);
    }
    let grid = MTLSize {
        width: num_heads,
        height: seq_len,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe {
        std::slice::from_raw_parts(
            contents.as_ptr() as *const f32,
            seq_len * num_heads * head_dim,
        )
    };
    (out.to_vec(), gpu_ms)
}

/// 반환: (결과 f32, 순수 GPU-time ms). `fn_name`= "gemm_step5_outer_f16"(matmul2d) 또는
/// "step5_outer_scalar_ref"(현 STEP5 thread-vi 등가). 파이프라인 즉석 build(microbench 1회용).
#[cfg(test)]
fn run_step5_microbench_inner(
    ctx: &MetalContext,
    fn_name: &str,
    us: &[f32],
    kk: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    const HV: usize = 128;
    const HK: usize = 128;
    assert_eq!(us.len(), nh * c_real * HV, "Us must be nh*C*HV");
    assert_eq!(kk.len(), nh * c_real * HK, "Kk must be nh*C*HK");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_DENSE_F16_MICROBENCH_SRC, fn_name);
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let us_buf = mk(us.as_ptr() as *const _, std::mem::size_of_val(us));
    let kk_buf = mk(kk.as_ptr() as *const _, std::mem::size_of_val(kk));
    let c_u32 = c_real as u32;
    let c_buf = mk(&c_u32 as *const u32 as *const _, std::mem::size_of::<u32>());
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(nh * HV * HK * std::mem::size_of::<f32>(), shared)
        .expect("Metal: output buffer");

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*us_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*kk_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*c_buf), 0, 3);
    }
    // grid = nh threadgroups (head batch), tg = 128 threads
    // (gemm: execution_simdgroups<4>=128 lanes / scalar: vi 0..HV=128).
    let grid = MTLSize {
        width: nh,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, nh * HV * HK) };
    (out.to_vec(), gpu_ms)
}

/// STEP5 outer product를 matmul2d f16 GEMM(fp32 accumulate)으로. 반환 (결과, GPU-time ms).
#[cfg(test)]
pub fn run_step5_outer_gemm(
    ctx: &MetalContext,
    us: &[f32],
    kk: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    run_step5_microbench_inner(ctx, "gemm_step5_outer_f16", us, kk, c_real, nh)
}

/// STEP5 outer product를 현 scalar(thread-vi-per-row) 등가로. 반환 (결과, GPU-time ms).
#[cfg(test)]
pub fn run_step5_outer_scalar(
    ctx: &MetalContext,
    us: &[f32],
    kk: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    run_step5_microbench_inner(ctx, "step5_outer_scalar_ref", us, kk, c_real, nh)
}

/// pm47 STEP4 inter(S·q dense) microbench (성능 게이트). `q`[nh][C×HK], `state`[nh][HV×HK]
/// → `inter`[nh][CPAD×HV] (= Σ_ki state[vi,ki]·q[r,ki], r<c_real만 유효, γ_r·intra 제외).
/// 반환 (결과 f32 nh*CPAD*HV, 순수 GPU-time ms). `fn_name`="gemm_step4_inter_f16"(matmul2d
/// device-staging) 또는 "step4_inter_scalar_ref"(현 STEP4 inter thread-vi mat-vec 등가).
/// gemm 은 device half workspace(a_dev/b_dev) 사용(state 32KB threadgroup 초과). scalar 는 무시.
#[cfg(test)]
fn run_step4_microbench_inner(
    ctx: &MetalContext,
    fn_name: &str,
    q: &[f32],
    state: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    const HV: usize = 128;
    const HK: usize = 128;
    const CPAD: usize = 48;
    assert_eq!(q.len(), nh * c_real * HK, "q must be nh*C*HK");
    assert_eq!(state.len(), nh * HV * HK, "state must be nh*HV*HK");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_DENSE_F16_MICROBENCH_SRC, fn_name);
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let q_buf = mk(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let state_buf = mk(state.as_ptr() as *const _, std::mem::size_of_val(state));
    let c_u32 = c_real as u32;
    let c_buf = mk(&c_u32 as *const u32 as *const _, std::mem::size_of::<u32>());
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(nh * CPAD * HV * std::mem::size_of::<f32>(), shared)
        .expect("Metal: output buffer");
    // device half workspace (gemm 만 사용, scalar 는 buffer 4/5 무시). half = 2 bytes.
    let a_dev: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(nh * CPAD * HK * 2, shared)
        .expect("Metal: a_dev");
    let b_dev: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(nh * HK * HV * 2, shared)
        .expect("Metal: b_dev");

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*state_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*c_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*a_dev), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*b_dev), 0, 5);
    }
    let grid = MTLSize {
        width: nh,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, nh * CPAD * HV) };
    (out.to_vec(), gpu_ms)
}

/// STEP4 inter 를 matmul2d f16 GEMM(device-staging, fp32 accumulate)으로. 반환 (결과, GPU-time ms).
#[cfg(test)]
pub fn run_step4_inter_gemm(
    ctx: &MetalContext,
    q: &[f32],
    state: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    run_step4_microbench_inner(ctx, "gemm_step4_inter_f16", q, state, c_real, nh)
}

/// STEP4 inter 를 현 scalar(thread-vi mat-vec) 등가로. 반환 (결과, GPU-time ms).
#[cfg(test)]
pub fn run_step4_inter_scalar(
    ctx: &MetalContext,
    q: &[f32],
    state: &[f32],
    c_real: usize,
    nh: usize,
) -> (Vec<f32>, f64) {
    run_step4_microbench_inner(ctx, "step4_inter_scalar_ref", q, state, c_real, nh)
}

/// pm34 M4: M/N grid tiling tensorops GEMM. weight[N,K] Q4_K + input[M,K] f32 → out[M,N] f32.
/// 각 threadgroup = C 의 16×16 tile(threadgroup-C accumulate → device strided write). 임의 M/N/K.
#[allow(dead_code)]
pub fn run_q4k_tensorops(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let nb = k / 256;
    assert_eq!(wb.len(), n * nb * 144, "weight bytes mismatch");
    assert_eq!(input.len(), m * k, "input must be M*K");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_TENSOROPS_POC_SRC, "gemm_q4k_tensorops");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let w_buf = mk(wb.as_ptr() as *const _, wb.len());
    let i_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let zeros = vec![0f32; m * n];
    let o_buf = mk(
        zeros.as_ptr() as *const _,
        std::mem::size_of_val(&zeros[..]),
    );
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let m_u32 = m as u32;
    let n_buf = mk(&n_u32 as *const u32 as *const _, 4);
    let k_buf = mk(&k_u32 as *const u32 as *const _, 4);
    let m_buf = mk(&m_u32 as *const u32 as *const _, 4);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*m_buf), 0, 5);
    }
    // Q4_K: 64×32 tile, 4 simdgroup(128 thread).
    let grid = MTLSize {
        width: n.div_ceil(32),
        height: m.div_ceil(64),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out.to_vec()
}

/// pm40 M1: llama 패턴 Q4_K GEMM(gemm_q4k_tensorops_v2) standalone runner. weight[N_out,K] Q4_K +
/// input[M_tok,K] f32 → out[M_tok,N_out] f32. activation 은 host 에서 f32→f16 변환 후 device tensor 직접.
/// device-direct B(activation) + cooperative tensor C. 현 run_q4k_tensorops(A+B+C threadgroup) 와 동일
/// 결과(의미 동등, f16 staging)여야. correctness/micro-bench 용.
pub fn run_q4k_tensorops_v2(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
) -> Vec<f32> {
    run_q4k_tensorops_v2_variant(
        ctx,
        wb,
        input,
        n_out,
        k,
        m_tok,
        "gemm_q4k_tensorops_v2",
        64,
        32,
    )
}

/// pm101: Q4_K v2 gate/up pair standalone runner. 두 raw Q4_K weight를 한 dispatch에서
/// 순차 처리하며, 결과는 기존 run_q4k_tensorops_v2 두 번과 bit-identical이어야 한다.
pub fn run_q4k_tensorops_v2_pair(
    ctx: &MetalContext,
    gate_wb: &[u8],
    up_wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nb = k / 256;
    let expected = n_out * nb * 144;
    assert_eq!(gate_wb.len(), expected, "gate weight bytes mismatch");
    assert_eq!(up_wb.len(), expected, "up weight bytes mismatch");
    assert_eq!(input.len(), m_tok * k, "input must be M_tok*K");

    let pipeline = build_pipeline_v4(
        &ctx.device,
        GEMM_TENSOROPS_POC_SRC,
        "gemm_q4k_tensorops_v2_pair_64x128",
    );
    let input_f16: Vec<u16> = input
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let gate_w_buf = mk(gate_wb.as_ptr() as *const _, gate_wb.len());
    let up_w_buf = mk(up_wb.as_ptr() as *const _, up_wb.len());
    let i_buf = mk(
        input_f16.as_ptr() as *const _,
        std::mem::size_of_val(&input_f16[..]),
    );
    let gate_zeros = vec![0f32; m_tok * n_out];
    let up_zeros = vec![0f32; m_tok * n_out];
    let gate_o_buf = mk(
        gate_zeros.as_ptr() as *const _,
        std::mem::size_of_val(&gate_zeros[..]),
    );
    let up_o_buf = mk(
        up_zeros.as_ptr() as *const _,
        std::mem::size_of_val(&up_zeros[..]),
    );
    let n_u32 = n_out as u32;
    let k_u32 = k as u32;
    let m_u32 = m_tok as u32;
    let n_buf = mk(&n_u32 as *const u32 as *const _, 4);
    let k_buf = mk(&k_u32 as *const u32 as *const _, 4);
    let m_buf = mk(&m_u32 as *const u32 as *const _, 4);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    encode_tensorops_v2_pair_dispatch(
        &pipeline,
        &enc,
        &gate_w_buf,
        0,
        &up_w_buf,
        0,
        &i_buf,
        &gate_o_buf,
        &up_o_buf,
        &n_buf,
        &k_buf,
        &m_buf,
        n_out,
        m_tok,
        64,
        128,
        64,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let gate_contents: NonNull<std::ffi::c_void> = gate_o_buf.contents();
    let up_contents: NonNull<std::ffi::c_void> = up_o_buf.contents();
    let gate_out: &[f32] =
        unsafe { std::slice::from_raw_parts(gate_contents.as_ptr() as *const f32, m_tok * n_out) };
    let up_out: &[f32] =
        unsafe { std::slice::from_raw_parts(up_contents.as_ptr() as *const f32, m_tok * n_out) };
    (gate_out.to_vec(), up_out.to_vec())
}

/// pm40 M2: v2 타일 variant runner. fn_name=커널 entry, nra=weight row tile, nrb=token tile.
/// threadgroup=nra*NK(64)*2(weight only). grid=(M_tok/nrb, N_out/nra). 타일/sg 스윕 측정용.
/// Q4_K(144B/super, NK=64) 고정 — q5k/q6k 는 run_q{5,6}k_tensorops_v2 가 byte/nk 만 바꿔 inner 호출.
#[allow(clippy::too_many_arguments)]
pub fn run_q4k_tensorops_v2_variant(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
    fn_name: &str,
    nra: usize,
    nrb: usize,
) -> Vec<f32> {
    run_tensorops_v2_inner(ctx, wb, input, n_out, k, m_tok, fn_name, nra, nrb, 144, 64)
}

/// pm42 M3 step1: v2 cooperative-tensor GEMM 의 quant-generic runner. bytes_per_super/nk 만 quant
/// 별로 다름(Q4_K 144/64, Q5_K 176/64, Q6_K 210/128). weight bytes → threadgroup dequant,
/// activation f32→f16 device-direct tensor, C cooperative. grid=(M_tok/nrb, N_out/nra), tg=128.
#[allow(clippy::too_many_arguments)]
fn run_tensorops_v2_inner(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
    fn_name: &str,
    nra: usize,
    nrb: usize,
    bytes_per_super: usize,
    nk: usize,
) -> Vec<f32> {
    let nb = k / 256;
    assert_eq!(
        wb.len(),
        n_out * nb * bytes_per_super,
        "weight bytes mismatch"
    );
    assert_eq!(input.len(), m_tok * k, "input must be M_tok*K");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_TENSOROPS_POC_SRC, fn_name);
    // activation f32 → f16 (matmul2d device tensor 는 half 강제).
    let input_f16: Vec<u16> = input
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let w_buf = mk(wb.as_ptr() as *const _, wb.len());
    let i_buf = mk(
        input_f16.as_ptr() as *const _,
        std::mem::size_of_val(&input_f16[..]),
    );
    let zeros = vec![0f32; m_tok * n_out];
    let o_buf = mk(
        zeros.as_ptr() as *const _,
        std::mem::size_of_val(&zeros[..]),
    );
    let n_u32 = n_out as u32;
    let k_u32 = k as u32;
    let m_u32 = m_tok as u32;
    let n_buf = mk(&n_u32 as *const u32 as *const _, 4);
    let k_buf = mk(&k_u32 as *const u32 as *const _, 4);
    let m_buf = mk(&m_u32 as *const u32 as *const _, 4);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*m_buf), 0, 5);
        // threadgroup shmem: NRA*NK half (weight only). NK=64(Q4_K/Q5_K) | 128(Q6_K).
        enc.setThreadgroupMemoryLength_atIndex(nra * nk * 2, 0);
    }
    // v2 grid: x=token(NRB), y=weight row(NRA). run_q4k_tensorops 와 축 반대(llama 매핑).
    let grid = MTLSize {
        width: m_tok.div_ceil(nrb),
        height: n_out.div_ceil(nra),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m_tok * n_out) };
    out.to_vec()
}

/// pm42 M3 step1: Q5_K v2 GEMM(64×128 winner 타일). GDN ssm_out(Q5_K) + 9B GDN projection.
/// dequant 만 Q4_K 와 다름(qh high-bit, 176B/super, NK=64). standalone correctness/runner.
pub fn run_q5k_tensorops_v2(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
) -> Vec<f32> {
    run_tensorops_v2_inner(
        ctx,
        wb,
        input,
        n_out,
        k,
        m_tok,
        "gemm_q5k_tensorops_v2_64x128",
        64,
        128,
        176,
        64,
    )
}

/// pm42 M3 step1: Q6_K v2 GEMM(64×128 winner 타일). FFN down(Q6_K) + GDN in_proj(Q6_K).
/// dequant Q6_K(210B/super, NK=128=superblock 절반). standalone correctness/runner.
pub fn run_q6k_tensorops_v2(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n_out: usize,
    k: usize,
    m_tok: usize,
) -> Vec<f32> {
    run_tensorops_v2_inner(
        ctx,
        wb,
        input,
        n_out,
        k,
        m_tok,
        "gemm_q6k_tensorops_v2_64x128",
        64,
        128,
        210,
        128,
    )
}

/// pm34 M5: Q6_K M/N grid tiling tensorops GEMM. weight[N,K] Q6_K(210B/sb) + input[M,K] → out[M,N].
#[allow(dead_code)]
pub fn run_q6k_tensorops(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let nb = k / 256;
    assert_eq!(wb.len(), n * nb * 210, "q6k weight bytes mismatch");
    assert_eq!(input.len(), m * k, "input must be M*K");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_TENSOROPS_POC_SRC, "gemm_q6k_tensorops");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let w_buf = mk(wb.as_ptr() as *const _, wb.len());
    let i_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let zeros = vec![0f32; m * n];
    let o_buf = mk(
        zeros.as_ptr() as *const _,
        std::mem::size_of_val(&zeros[..]),
    );
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let m_u32 = m as u32;
    let n_buf = mk(&n_u32 as *const u32 as *const _, 4);
    let k_buf = mk(&k_u32 as *const u32 as *const _, 4);
    let m_buf = mk(&m_u32 as *const u32 as *const _, 4);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*m_buf), 0, 5);
    }
    // pm35: Q6_K 64×32 tile, 4 simdgroup(128 thread). Q4_K 와 동일 NA 타일.
    let grid = MTLSize {
        width: n.div_ceil(32),
        height: m.div_ceil(64),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out.to_vec()
}

/// pm36: Q5_K tensorops 64×32 GEMM 테스트 헬퍼(run_q6k_tensorops 1:1, 176B/superblock).
/// GDN ssm_out(Q5_K) + 9B 전체 GDN projection correctness 검증용.
#[cfg(target_os = "macos")]
pub fn run_q5k_tensorops(
    ctx: &MetalContext,
    wb: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let nb = k / 256;
    assert_eq!(wb.len(), n * nb * 176, "q5k weight bytes mismatch");
    assert_eq!(input.len(), m * k, "input must be M*K");
    let pipeline = build_pipeline_v4(&ctx.device, GEMM_TENSOROPS_POC_SRC, "gemm_q5k_tensorops");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let w_buf = mk(wb.as_ptr() as *const _, wb.len());
    let i_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let zeros = vec![0f32; m * n];
    let o_buf = mk(
        zeros.as_ptr() as *const _,
        std::mem::size_of_val(&zeros[..]),
    );
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let m_u32 = m as u32;
    let n_buf = mk(&n_u32 as *const u32 as *const _, 4);
    let k_buf = mk(&k_u32 as *const u32 as *const _, 4);
    let m_buf = mk(&m_u32 as *const u32 as *const _, 4);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*m_buf), 0, 5);
    }
    // pm36: Q5_K 64×32 tile, 4 simdgroup(128 thread). Q4_K 와 동일 NA 타일(KC=64).
    let grid = MTLSize {
        width: n.div_ceil(32),
        height: m.div_ceil(64),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out.to_vec()
}

/// pm34 M6: tensorops GEMM encode-only(공유 command encoder, commit/readback 없음).
/// weight_byte_offset 은 Metal-native 인코더 offset 으로 적용(weight=uchar* 1-byte align →
/// 임의 offset 유효, 커널은 buffer 시작=weight base). out_buf 는 커널이 C_stage 0-init 후
/// strided write → host pre-zero 불요. 5-buffer(off 버퍼 없음).
/// tile: (bm, bn) 출력 타일, nsg = simdgroup 수(tg = nsg*32 threads). grid=(ceil(N/bn),ceil(M/bm)).
#[allow(clippy::too_many_arguments)]
fn encode_tensorops_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
    bm: usize,
    bn: usize,
    nsg: usize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 5);
    }
    let grid = MTLSize {
        width: n.div_ceil(bn),
        height: m.div_ceil(bm),
        depth: 1,
    };
    let tg = MTLSize {
        width: nsg * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm34 M6: Q4_K tensorops GEMM encode(chain gate/up/down Q4_K). capability=true 전제.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q4k_tensorops_pipeline
        .as_ref()
        .expect("gemm_q4k_tensorops_pipeline missing (capability=false?)");
    // Q4_K: 64×32 tile, K_chunk=64, 4 simdgroup(NA 최적).
    encode_tensorops_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        32,
        4,
    );
}

/// pm34 M6: Q6_K tensorops GEMM encode(chain down Q6_K). capability=true 전제.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_tensorops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q6k_tensorops_pipeline
        .as_ref()
        .expect("gemm_q6k_tensorops_pipeline missing (capability=false?)");
    // pm35: Q6_K 64×32 tile, K_chunk=128, 4 simdgroup(NA 최적, Q4_K 와 동일 패턴).
    encode_tensorops_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        32,
        4,
    );
}

fn set_u32_bytes(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: u32, index: usize) {
    unsafe {
        let ptr = NonNull::new(&value as *const u32 as *mut std::ffi::c_void)
            .expect("u32 bytes ptr null");
        enc.setBytes_length_atIndex(ptr, std::mem::size_of::<u32>(), index);
    }
}

fn set_f32_bytes(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: f32, index: usize) {
    unsafe {
        let ptr = NonNull::new(&value as *const f32 as *mut std::ffi::c_void)
            .expect("f32 bytes ptr null");
        enc.setBytes_length_atIndex(ptr, std::mem::size_of::<f32>(), index);
    }
}

fn qwen_moe_shared_mul_mm_checked_tiles(
    elements: usize,
    tile: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    elements
        .checked_add(tile - 1)
        .map(|rounded| rounded / tile)
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_shared_mul_mm_qk_f32(
    pipeline: Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
    block_bytes: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n == 0 || k == 0 || m == 0 || weight_offset % std::mem::align_of::<u16>() != 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let blocks_per_row = qwen_moe_shared_mul_mm_checked_tiles(k, 256)?;
    let weight_bytes = n
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let weight_end = weight_offset
        .checked_add(weight_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidExpertArena)?;
    if weight_end > weights.length() {
        return Err(QwenMoeLlamaIdError::InvalidExpertArena);
    }

    let input_bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let output_bytes = m
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if input.length() < input_bytes || output.length() < output_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let pipeline = pipeline.ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }
    let n_u32 = u32::try_from(n).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let k_u32 = u32::try_from(k).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let m_u32 = u32::try_from(m).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let m_tiles = qwen_moe_shared_mul_mm_checked_tiles(m, 32)?;
    let n_tiles = qwen_moe_shared_mul_mm_checked_tiles(n, 64)?;

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weights), weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
        enc.setThreadgroupMemoryLength_atIndex(8192, 0);
    }
    set_u32_bytes(enc, n_u32, 3);
    set_u32_bytes(enc, k_u32, 4);
    set_u32_bytes(enc, m_u32, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m_tiles,
            height: n_tiles,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_shared_mul_mm_q4k_f32(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    encode_qwen_moe_shared_mul_mm_qk_f32(
        ctx.qwen_moe_shared_mul_mm_q4k_f32_pipeline(),
        enc,
        weights,
        weight_offset,
        input,
        output,
        n,
        k,
        m,
        144,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_shared_mul_mm_q6k_f32(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    encode_qwen_moe_shared_mul_mm_qk_f32(
        ctx.qwen_moe_shared_mul_mm_q6k_f32_pipeline(),
        enc,
        weights,
        weight_offset,
        input,
        output,
        n,
        k,
        m,
        210,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_id_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 10);
    }
    set_u32_bytes(enc, n as u32, 6);
    set_u32_bytes(enc, k as u32, 7);
    set_u32_bytes(enc, expert_stride_bytes as u32, 8);
    let grid = MTLSize {
        width: n.div_ceil(32),
        height: block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

fn qwen_moe_prefill_id_v2_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ID_V2")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_id_v2_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    const BM: usize = 64;
    const BN: usize = 64;
    const NK: usize = 64;
    let tg_mem =
        (BM * NK + BN * NK) * std::mem::size_of::<u16>() + (BM * BN) * std::mem::size_of::<f32>();
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 10);
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    set_u32_bytes(enc, n as u32, 6);
    set_u32_bytes(enc, k as u32, 7);
    set_u32_bytes(enc, expert_stride_bytes as u32, 8);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_id_v2_64x128_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    const BM: usize = 64;
    const BN: usize = 128;
    const NK: usize = 64;
    let tg_mem =
        (BM * NK + BN * NK) * std::mem::size_of::<u16>() + (BM * BN) * std::mem::size_of::<f32>();
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 10);
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    set_u32_bytes(enc, n as u32, 6);
    set_u32_bytes(enc, k as u32, 7);
    set_u32_bytes(enc, expert_stride_bytes as u32, 8);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_id(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    if qwen_moe_prefill_id_v2_requested() {
        if let Some(pipeline) = ctx.gemm_q4k_tensorops_id_v2_pipeline.as_ref() {
            encode_tensorops_id_v2_dispatch(
                pipeline,
                enc,
                w_buf,
                weight_byte_offset,
                in_buf,
                out_buf,
                token_ids_buf,
                expert_offsets_buf,
                expert_counts_buf,
                block_experts_buf,
                block_local0_buf,
                n,
                k,
                block_count,
                expert_stride_bytes,
            );
            return;
        }
    }
    let pipeline = ctx
        .gemm_q4k_tensorops_id_pipeline
        .as_ref()
        .expect("gemm_q4k_tensorops_id_pipeline missing (capability=false?)");
    encode_tensorops_id_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        n,
        k,
        block_count,
        expert_stride_bytes,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_id_f16(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    if let Some(pipeline) = ctx.gemm_q4k_tensorops_id_v2_64x128_f16_pipeline.as_ref() {
        encode_tensorops_id_v2_64x128_dispatch(
            pipeline,
            enc,
            w_buf,
            weight_byte_offset,
            in_buf,
            out_buf,
            token_ids_buf,
            expert_offsets_buf,
            expert_counts_buf,
            block_experts_buf,
            block_local0_buf,
            n,
            k,
            block_count,
            expert_stride_bytes,
        );
        return;
    }
    let pipeline = ctx
        .gemm_q4k_tensorops_id_v2_f16_pipeline
        .as_ref()
        .expect("gemm_q4k_tensorops_id_v2_f16_pipeline missing (capability=false?)");
    encode_tensorops_id_v2_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        n,
        k,
        block_count,
        expert_stride_bytes,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_tensorops_id(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) {
    let pipeline = ctx
        .gemm_q6k_tensorops_id_pipeline
        .as_ref()
        .expect("gemm_q6k_tensorops_id_pipeline missing (capability=false?)");
    encode_tensorops_id_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        n,
        k,
        block_count,
        expert_stride_bytes,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_v3_q6_down(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    act_half_buf: &ProtocolObject<dyn MTLBuffer>,
    down_token_rank_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_slots_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    const BN: usize = 32;
    let pipeline = ctx
        .qwen_moe_v3_q6_down_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(act_half_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(down_token_rank_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dst_slots_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 10);
    }
    set_u32_bytes(enc, n as u32, 6);
    set_u32_bytes(enc, k as u32, 7);
    set_u32_bytes(enc, expert_stride_bytes as u32, 8);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_v3_q4_down(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    act_half_buf: &ProtocolObject<dyn MTLBuffer>,
    down_token_rank_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_slots_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    expert_stride_bytes: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    const BM: usize = 64;
    const BN: usize = 64;
    const NK: usize = 64;
    let pipeline = ctx
        .qwen_moe_v3_q4_down_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    let tg_mem =
        (BM * NK + BN * NK) * std::mem::size_of::<u16>() + (BM * BN) * std::mem::size_of::<f32>();
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(act_half_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(down_token_rank_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dst_slots_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 10);
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    set_u32_bytes(enc, n as u32, 6);
    set_u32_bytes(enc, k as u32, 7);
    set_u32_bytes(enc, expert_stride_bytes as u32, 8);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_v4_q6_down_scatter(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    act_half_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_rank_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_rank_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_offset: usize,
    rank_block_count: usize,
    rank: usize,
    n_expert_used: usize,
    n: usize,
    k: usize,
    expert_stride_bytes: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    const BN: usize = 32;
    let pipeline = ctx
        .qwen_moe_v4_q6_down_scatter_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(act_half_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_sorted_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(route_weights_sorted_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_rank_offsets_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(expert_rank_counts_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(rank_block_experts_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(rank_block_local0_buf), 0, 11);
    }
    set_u32_bytes(enc, n as u32, 7);
    set_u32_bytes(enc, k as u32, 8);
    set_u32_bytes(enc, expert_stride_bytes as u32, 9);
    set_u32_bytes(enc, rank_block_offset as u32, 12);
    set_u32_bytes(enc, rank_block_count as u32, 13);
    set_u32_bytes(enc, rank as u32, 14);
    set_u32_bytes(enc, n_expert_used as u32, 15);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: rank_block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_v4_q4_down_scatter(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    act_half_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_rank_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_rank_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_offset: usize,
    rank_block_count: usize,
    rank: usize,
    n_expert_used: usize,
    n: usize,
    k: usize,
    expert_stride_bytes: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    const BM: usize = 64;
    const BN: usize = 64;
    const NK: usize = 64;
    let pipeline = ctx
        .qwen_moe_v4_q4_down_scatter_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    let tg_mem =
        (BM * NK + BN * NK) * std::mem::size_of::<u16>() + (BM * BN) * std::mem::size_of::<f32>();
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(act_half_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_sorted_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(route_weights_sorted_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(expert_rank_offsets_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(expert_rank_counts_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(rank_block_experts_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(rank_block_local0_buf), 0, 11);
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    set_u32_bytes(enc, n as u32, 7);
    set_u32_bytes(enc, k as u32, 8);
    set_u32_bytes(enc, expert_stride_bytes as u32, 9);
    set_u32_bytes(enc, rank_block_offset as u32, 12);
    set_u32_bytes(enc, rank_block_count as u32, 13);
    set_u32_bytes(enc, rank as u32, 14);
    set_u32_bytes(enc, n_expert_used as u32, 15);
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: rank_block_count.max(1),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

pub(crate) fn encode_qwen_moe_v3_token_rank_combine(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    down_token_rank_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    n_expert_used: usize,
    hidden_dim: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    let pipeline = ctx
        .qwen_moe_v3_token_rank_combine_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(down_token_rank_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
    }
    set_u32_bytes(enc, seq_len as u32, 3);
    set_u32_bytes(enc, n_expert_used as u32, 4);
    set_u32_bytes(enc, hidden_dim as u32, 5);
    let tgw = pipeline.threadExecutionWidth().max(1);
    let total = seq_len
        .checked_mul(hidden_dim)
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_id_partial_reduce_scatter(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    partial_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    ffn_tiles: usize,
    tile_slots: usize,
    hidden_tile: usize,
    hidden_dim: usize,
    hidden0: usize,
    group_start: u32,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    let pipeline = ctx
        .qwen_moe_id_partial_reduce_scatter_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(partial_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 3);
    }
    set_u32_bytes(enc, ffn_tiles as u32, 4);
    set_u32_bytes(enc, tile_slots as u32, 5);
    set_u32_bytes(enc, hidden_tile as u32, 6);
    set_u32_bytes(enc, hidden_dim as u32, 7);
    set_u32_bytes(enc, hidden0 as u32, 8);
    set_u32_bytes(enc, group_start, 9);
    let tgw = pipeline.threadExecutionWidth().max(1);
    let total = tile_slots * hidden_tile;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_id_q4_gate_up_tile(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_weight_byte_offset: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_out_buf: &ProtocolObject<dyn MTLBuffer>,
    up_out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    block_count: usize,
    gate_expert_stride_bytes: usize,
    up_expert_stride_bytes: usize,
    ffn_tile0: usize,
    ffn_tile: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    let pipeline = ctx
        .qwen_moe_id_q4_gate_up_tile_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up_w_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate_out_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(up_out_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(expert_offsets_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(expert_counts_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(block_experts_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(block_local0_buf), 0, 9);
    }
    set_u32_bytes(enc, n as u32, 10);
    set_u32_bytes(enc, k as u32, 11);
    set_u32_bytes(enc, gate_weight_byte_offset, 12);
    set_u32_bytes(enc, up_weight_byte_offset, 13);
    set_u32_bytes(enc, gate_expert_stride_bytes as u32, 14);
    set_u32_bytes(enc, up_expert_stride_bytes as u32, 15);
    set_u32_bytes(enc, ffn_tile0 as u32, 16);
    set_u32_bytes(enc, ffn_tile as u32, 17);
    let tile_slots = 64usize;
    set_u32_bytes(enc, tile_slots as u32, 18);
    let tgw = pipeline.threadExecutionWidth().max(1);
    let total = block_count * tile_slots * ffn_tile;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

pub(crate) fn encode_qwen_moe_id_silu_mul_tile(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    act_buf: &ProtocolObject<dyn MTLBuffer>,
    elems: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    let pipeline = ctx
        .qwen_moe_id_silu_mul_tile_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(act_buf), 0, 2);
    }
    set_u32_bytes(enc, elems as u32, 3);
    let tgw = pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: elems.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_id_q6_down_partial_tile(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    act_buf: &ProtocolObject<dyn MTLBuffer>,
    partial_buf: &ProtocolObject<dyn MTLBuffer>,
    hidden_dim: usize,
    hidden0: usize,
    hidden_tile: usize,
    ffn_dim: usize,
    ffn_tile0: usize,
    ffn_tile: usize,
    slots: usize,
    ffn_tile_idx: usize,
) -> Result<(), crate::ffn_chain::QwenMoeIdPreflightReason> {
    let pipeline = ctx
        .qwen_moe_id_q6_down_partial_tile_pipeline
        .as_ref()
        .ok_or(crate::ffn_chain::QwenMoeIdPreflightReason::PipelineUnavailable)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(act_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(partial_buf), 0, 2);
    }
    set_u32_bytes(enc, weight_byte_offset, 3);
    set_u32_bytes(enc, hidden_dim as u32, 4);
    set_u32_bytes(enc, hidden0 as u32, 5);
    set_u32_bytes(enc, hidden_tile as u32, 6);
    set_u32_bytes(enc, ffn_dim as u32, 7);
    set_u32_bytes(enc, ffn_tile0 as u32, 8);
    set_u32_bytes(enc, ffn_tile as u32, 9);
    set_u32_bytes(enc, slots as u32, 10);
    set_u32_bytes(enc, ffn_tile_idx as u32, 11);
    let tgw = pipeline.threadExecutionWidth().max(1);
    let total = slots * hidden_tile;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

/// pm42 M3 step2: v2 tensorops GEMM encode-only(공유 encoder). v1 encode_tensorops_dispatch 와 차이:
/// in_buf 가 f16(matmul2d device tensor 강제), grid 축 반대(x=token/NRB, y=row/NRA), threadgroup
/// = NRA*NK*2(weight only, cooperative C 는 register), tg=128(4sg). out_buf 는 커널 cT.store 로 write.
#[allow(clippy::too_many_arguments)]
fn encode_tensorops_v2_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
    nra: usize,
    nrb: usize,
    nk: usize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_f16_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 5);
        enc.setThreadgroupMemoryLength_atIndex(nra * nk * 2, 0);
    }
    // v2 grid: x=token(NRB), y=weight row(NRA). run_q4k_tensorops_v2_variant 와 동일.
    let grid = MTLSize {
        width: m.div_ceil(nrb),
        height: n.div_ceil(nra),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_v2_dispatch_offset(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: usize,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    output_byte_offset: usize,
    n: usize,
    k: usize,
    m: usize,
    nra: usize,
    nrb: usize,
    nk: usize,
) -> Result<(), &'static str> {
    if n == 0 || k == 0 || m == 0 {
        return Err("tensorops v2 dimensions must be non-zero");
    }
    let n_u32 = u32::try_from(n).map_err(|_| "tensorops v2 N overflow")?;
    let k_u32 = u32::try_from(k).map_err(|_| "tensorops v2 K overflow")?;
    let m_u32 = u32::try_from(m).map_err(|_| "tensorops v2 M overflow")?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(in_f16_buf), input_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), output_byte_offset, 2);
        enc.setThreadgroupMemoryLength_atIndex(nra * nk * 2, 0);
    }
    set_u32_bytes(enc, n_u32, 3);
    set_u32_bytes(enc, k_u32, 4);
    set_u32_bytes(enc, m_u32, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(nrb),
            height: n.div_ceil(nra),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_v2_pair_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_weight_byte_offset: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_out_buf: &ProtocolObject<dyn MTLBuffer>,
    up_out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
    nra: usize,
    nrb: usize,
    nk: usize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_w_buf), gate_weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(up_w_buf), up_weight_byte_offset as usize, 1);
        enc.setBuffer_offset_atIndex(Some(in_f16_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate_out_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(up_out_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 7);
        enc.setThreadgroupMemoryLength_atIndex(nra * nk * 2, 0);
    }
    let grid = MTLSize {
        width: m.div_ceil(nrb),
        height: n.div_ceil(nra),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm42 M3: Q4_K v2 GEMM encode(chain gate/up + down Q4_K). NRA=64,NRB=128,NK=64,4sg.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_v2(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q4k_tensorops_v2_pipeline
        .as_ref()
        .expect("gemm_q4k_tensorops_v2_pipeline missing (capability=false?)");
    encode_tensorops_v2_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        128,
        64,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_v2_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: usize,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    output_byte_offset: usize,
    n: usize,
    k: usize,
    m: usize,
) -> Result<(), &'static str> {
    let pipeline = ctx
        .gemm_q4k_tensorops_v2_pipeline
        .as_ref()
        .ok_or("gemm_q4k_tensorops_v2_pipeline missing")?;
    encode_tensorops_v2_dispatch_offset(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        input_byte_offset,
        out_buf,
        output_byte_offset,
        n,
        k,
        m,
        64,
        128,
        64,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_v2_pair(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_weight_byte_offset: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_out_buf: &ProtocolObject<dyn MTLBuffer>,
    up_out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) -> bool {
    let Some(pipeline) = ctx.gemm_q4k_tensorops_v2_pair_pipeline.as_ref() else {
        return false;
    };
    encode_tensorops_v2_pair_dispatch(
        pipeline,
        enc,
        gate_w_buf,
        gate_weight_byte_offset,
        up_w_buf,
        up_weight_byte_offset,
        in_f16_buf,
        gate_out_buf,
        up_out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        128,
        64,
    );
    true
}

/// pm42 M3: Q5_K v2 GEMM encode(GDN ssm_out Q5_K). NRA=64,NRB=128,NK=64,4sg.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q5k_tensorops_v2(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q5k_tensorops_v2_pipeline
        .as_ref()
        .expect("gemm_q5k_tensorops_v2_pipeline missing (capability=false?)");
    encode_tensorops_v2_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        128,
        64,
    );
}

/// pm42 M3: Q6_K v2 GEMM encode(chain down Q6_K + GDN in_proj Q6_K). NRA=64,NRB=128,NK=128,4sg.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_tensorops_v2(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q6k_tensorops_v2_pipeline
        .as_ref()
        .expect("gemm_q6k_tensorops_v2_pipeline missing (capability=false?)");
    encode_tensorops_v2_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        128,
        128,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_tensorops_v2_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: usize,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    output_byte_offset: usize,
    n: usize,
    k: usize,
    m: usize,
) -> Result<(), &'static str> {
    let pipeline = ctx
        .gemm_q6k_tensorops_v2_pipeline
        .as_ref()
        .ok_or("gemm_q6k_tensorops_v2_pipeline missing")?;
    encode_tensorops_v2_dispatch_offset(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        input_byte_offset,
        out_buf,
        output_byte_offset,
        n,
        k,
        m,
        64,
        128,
        128,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_v2_scatter_accum_dispatch(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    accum_out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
    group_start: u32,
    nra: usize,
    nrb: usize,
    nk: usize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), weight_byte_offset as usize, 0);
        enc.setBuffer_offset_atIndex(Some(in_f16_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(accum_out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 4);
        enc.setThreadgroupMemoryLength_atIndex(nra * nk * 2 + nra * nrb * 4, 0);
    }
    set_u32_bytes(enc, n as u32, 5);
    set_u32_bytes(enc, k as u32, 6);
    set_u32_bytes(enc, m as u32, 7);
    set_u32_bytes(enc, group_start, 8);
    let grid = MTLSize {
        width: m.div_ceil(nrb),
        height: n.div_ceil(nra),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_tensorops_v2_scatter_accum(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    accum_out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
    group_start: u32,
) {
    let pipeline = ctx
        .gemm_q4k_tensorops_v2_scatter_accum_pipeline
        .as_ref()
        .expect("gemm_q4k_tensorops_v2_scatter_accum_pipeline missing (capability=false?)");
    encode_tensorops_v2_scatter_accum_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        accum_out_buf,
        token_ids_buf,
        route_weights_buf,
        n,
        k,
        m,
        group_start,
        64,
        64,
        64,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_tensorops_v2_scatter_accum(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    accum_out_buf: &ProtocolObject<dyn MTLBuffer>,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    k: usize,
    m: usize,
    group_start: u32,
) {
    let pipeline = ctx
        .gemm_q6k_tensorops_v2_scatter_accum_pipeline
        .as_ref()
        .expect("gemm_q6k_tensorops_v2_scatter_accum_pipeline missing (capability=false?)");
    encode_tensorops_v2_scatter_accum_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_f16_buf,
        accum_out_buf,
        token_ids_buf,
        route_weights_buf,
        n,
        k,
        m,
        group_start,
        64,
        64,
        64,
    );
}

/// pm42 M3 step2: f32 device buffer → f16 device buffer cast(elementwise, encode-only).
/// v2 GEMM activation 변환용 — chain 에서 normed/silu결과 1회씩. n_buf=u32(elem 수).
pub(crate) fn encode_cast_f32_to_f16(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    let pipeline = ctx
        .cast_f32_f16_pipeline
        .as_ref()
        .expect("cast_f32_f16_pipeline missing (capability=false?)");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst_f16_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 2);
    }
    let tgw = 256usize;
    let grid = MTLSize {
        width: n.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

pub(crate) fn encode_cast_f32_to_f16_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_byte_offset: usize,
    n: usize,
) -> Result<(), &'static str> {
    if n == 0 {
        return Err("f32-to-f16 element count must be non-zero");
    }
    let n_u32 = u32::try_from(n).map_err(|_| "f32-to-f16 element count overflow")?;
    let pipeline = ctx
        .cast_f32_f16_pipeline
        .as_ref()
        .ok_or("cast_f32_f16_pipeline missing")?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst_f16_buf), dst_byte_offset, 1);
    }
    set_u32_bytes(enc, n_u32, 2);
    let tgw = 256usize;
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(tgw),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tgw,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// silu(gate) * up → f16 device buffer. Equivalent to encode_silu_mul followed by
/// encode_cast_f32_to_f16, but without materializing the f32 activation.
pub(crate) fn encode_silu_mul_to_f16(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.silu_mul_f16_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dst_f16_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
    }
    let tgw = ctx.silu_mul_f16_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_silu_mul_to_f16_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_byte_offset: usize,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    up_byte_offset: usize,
    dst_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_byte_offset: usize,
    n: usize,
) -> Result<(), &'static str> {
    if n == 0 {
        return Err("silu element count must be non-zero");
    }
    let n_u32 = u32::try_from(n).map_err(|_| "silu element count overflow")?;
    enc.setComputePipelineState(&ctx.silu_mul_f16_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_buf), gate_byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(up_buf), up_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(dst_f16_buf), dst_byte_offset, 2);
    }
    set_u32_bytes(enc, n_u32, 3);
    let tgw = ctx.silu_mul_f16_pipeline.threadExecutionWidth().max(1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(tgw),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tgw,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// pm36: Q5_K tensorops GEMM encode(GDN ssm_out + 9B GDN projection). capability=true 전제.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q5k_tensorops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    let pipeline = ctx
        .gemm_q5k_tensorops_pipeline
        .as_ref()
        .expect("gemm_q5k_tensorops_pipeline missing (capability=false?)");
    // pm36: Q5_K 64×32 tile, K_chunk=64(= Q4_K), 4 simdgroup(NA 최적).
    encode_tensorops_dispatch(
        pipeline,
        enc,
        w_buf,
        weight_byte_offset,
        in_buf,
        out_buf,
        n_buf,
        k_buf,
        m_buf,
        n,
        m,
        64,
        32,
        4,
    );
}

// ---------------------------------------------------------------------------
// gemm_q4k — Q4_K batch GEMM(M>1). decode GEMV(M=1) 의 M축 확장. grid 2D (row, tok).
// pm33 prefill FFN PoC. 커널은 gemm_q4k.metal(gemv_q4k.metal dequant 1:1 + M축).
// ---------------------------------------------------------------------------

/// Q4_K batch GEMM (복사 경로): weight[N,K] Q4_K bytes + input[M,K] f32 → out[M,N] f32.
/// weight 를 GPU 버퍼로 복사 업로드 후 dispatch (offset=0). zero-copy 는 `wrap_nocopy`+`gemm_q4k_dispatch`.
pub fn gemm_q4k_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemm_q4k_dispatch(ctx, &w_buf, 0, input, n, k, m)
}

/// weight MTLBuffer + page offset 으로 input[M,K]/output[M,N]/scalar alloc + encode + dispatch + readback.
/// weight buffer 처리(복사 or NoCopy wrap)는 caller 담당 — `gemv_q4k_dispatch` 의 M>1 아날로그.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_q4k_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q4_K block size)");
    assert_eq!(input.len(), m * k, "input length must be M*K");
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = m * n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let m_u32 = m as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let m_buf = mk_u32(&m_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_gemm_q4k(
        ctx, &enc, w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, &m_buf, n, m,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out_slice.to_vec()
}

/// Q4_K batch GEMM 를 device buffer in/out 으로 compute encoder 에 encode (commit/readback 안 함).
/// in_buf(M*K f32) → out_buf(M*N f32). chain 안에서 device-resident 연결에 사용. grid 2D (row, tok).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    enc.setComputePipelineState(&ctx.gemm_q4k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
    }
    let tg_width = ctx.gemm_q4k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: m,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q4_K batch GEMM dequant-공유 encode-only. grid=N threadgroups(각 row), tg=256,
/// acc[M]은 동적 threadgroup memory(`setThreadgroupMemoryLength`). superblock dequant 1번 → 전 token 공유.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q4k_shared(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    enc.setComputePipelineState(&ctx.gemm_q4k_shared_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
        // acc[M] f32 동적 threadgroup memory (Metal: 16 byte 배수 요구)
        let tg_mem = (m * std::mem::size_of::<f32>()).div_ceil(16) * 16;
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    let tg_width = 256usize; // 8 SIMD-group. M5 maxTotalThreadsPerThreadgroup(1024) 내.
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }; // N threadgroups(각 1 weight row)
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q4_K dequant-공유 batch GEMM (복사 경로, 테스트용). `gemm_q4k_with_ctx` 의 shared 커널 버전.
pub fn gemm_q4k_shared_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );
    assert_eq!(input.len(), m * k, "input length must be M*K");
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(m * n * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = 0u32;
    let m_u32 = m as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let m_buf = mk_u32(&m_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_gemm_q4k_shared(
        ctx, &enc, &w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, &m_buf, n, m,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out_slice.to_vec()
}

// ---------------------------------------------------------------------------
// gemm_q6k — Q6_K batch GEMM(M>1). prefill FFN down(Q6_K). gemm_q4k 미러(stride 210).
// ---------------------------------------------------------------------------

/// Q6_K batch GEMM (복사 경로): weight[N,K] Q6_K bytes + input[M,K] f32 → out[M,N] f32.
pub fn gemm_q6k_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemm_q6k_dispatch(ctx, &w_buf, 0, input, n, k, m)
}

/// `gemm_q4k_dispatch` 의 Q6_K 아날로그(encode 만 gemm_q6k pipeline).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_q6k_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q6_K block size)");
    assert_eq!(input.len(), m * k, "input length must be M*K");
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = m * n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let m_u32 = m as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let m_buf = mk_u32(&m_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_gemm_q6k(
        ctx, &enc, w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, &m_buf, n, m,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out_slice.to_vec()
}

/// Q6_K batch GEMM encode-only (device buffer in/out). grid 2D (row, tok). chain down(Q6_K) 용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    enc.setComputePipelineState(&ctx.gemm_q6k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
    }
    let tg_width = ctx.gemm_q6k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: m,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q6_K batch GEMM dequant-공유 encode-only (gemm_q4k_shared 패턴, stride 210).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_q6k_shared(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    enc.setComputePipelineState(&ctx.gemm_q6k_shared_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
        let tg_mem = (m * std::mem::size_of::<f32>()).div_ceil(16) * 16;
        enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    let tg_width = 256usize;
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q6_K dequant-공유 batch GEMM (복사 경로, 테스트용).
pub fn gemm_q6k_shared_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "block_bytes length mismatch"
    );
    assert_eq!(input.len(), m * k, "input length must be M*K");
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(m * n * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = 0u32;
    let m_u32 = m as u32;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);
    let m_buf = mk_u32(&m_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_gemm_q6k_shared(
        ctx, &enc, &w_buf, &i_buf, &o_buf, &n_buf, &k_buf, &off_buf, &m_buf, n, m,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * n) };
    out_slice.to_vec()
}

/// Q4_K GEMV (SIMD-group 협력, device-in/out) encode-only. `encode_gemv_q4k`와 동일
/// 시그니처·동일 결과(rel<1e-3, simd_sum reduction 순서만 다름). chain에서 baseline 대신
/// 교체 가능. row 당 1 SIMD-group(32 lane), grid = N row. (gemv_q4k_simd_dispatch와 동일
/// pipeline/grid/tg, commit·버퍼alloc 없는 encode-only.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q4k_simd(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    // RNB_METAL_GEMV_COALESCED=1: q4k chain GEMV 를 llama식 nr0=2 multi-row coalesced 커널로.
    // attn/GDN 모두 이 함수(encode_gemv_q4k_auto→여기)를 거치므로 한 곳에서 전 q4k 경로 전환.
    let coalesced = ctx.gemv_coalesced;
    let nsg2 = coalesced && ctx.gemv_coalesced_nsg2;
    let pipeline = if nsg2 {
        &ctx.q4k_coalesced_nsg2_pipeline
    } else if coalesced {
        &ctx.q4k_coalesced_pipeline
    } else {
        &ctx.q4k_simd_pipeline
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    // coalesced는 row 2개/tg, nsg2는 SIMD-group 2개로 row 4개/tg. 기존 simd는 1 row/tg.
    let grid_w = if nsg2 {
        n.div_ceil(4)
    } else if coalesced {
        n.div_ceil(2)
    } else {
        n
    };
    let grid = MTLSize {
        width: grid_w,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: if nsg2 { SIMD_WIDTH * 2 } else { SIMD_WIDTH },
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// chain용 Q4_K GEMV 진입점. ctx.gemv_simd flag로 baseline/simd 선택 (ABAB toggle).
/// baseline `encode_gemv_q4k`/standalone 경로는 보존 — 측정 비교용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q4k_auto(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    if ctx.gemv_simd {
        encode_gemv_q4k_simd(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    } else {
        encode_gemv_q4k(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q4k_auto_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    output_byte_offset: usize,
    n: usize,
) {
    let coalesced = ctx.gemv_coalesced;
    let nsg2 = coalesced && ctx.gemv_coalesced_nsg2;
    let (pipeline, grid_w, tg_w) = if ctx.gemv_simd {
        let pipeline = if nsg2 {
            &ctx.q4k_coalesced_nsg2_pipeline
        } else if coalesced {
            &ctx.q4k_coalesced_pipeline
        } else {
            &ctx.q4k_simd_pipeline
        };
        let rows_per_tg = if nsg2 {
            4
        } else if coalesced {
            2
        } else {
            1
        };
        let tg_w = if nsg2 { SIMD_WIDTH * 2 } else { SIMD_WIDTH };
        (pipeline, n.div_ceil(rows_per_tg), tg_w)
    } else {
        (
            &ctx.q4k_pipeline,
            n.div_ceil(ctx.q4k_pipeline.threadExecutionWidth().max(1)),
            ctx.q4k_pipeline.threadExecutionWidth().max(1),
        )
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), input_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), output_byte_offset, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_w,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// Q8_0 GEMV (device-in/out) encode. carrier 용 — gate/beta 등 Q8_0 weight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q8_0(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    // RNB_METAL_GEMV_COALESCED=1: q8_0 chain GEMV 를 llama식 NR0=2 multi-row coalesced 커널로.
    // q4k_simd 와 동일 flag(ctx.gemv_coalesced). q8_0 은 simd 변형이 없어 이 함수가 단일 진입점.
    if ctx.gemv_coalesced {
        enc.setComputePipelineState(&ctx.q8_0_coalesced_pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        }
        // coalesced(NR0=2): threadgroup 1개(32 lane)가 output row 2개 → grid=ceil(N/2).
        let grid = MTLSize {
            width: n.div_ceil(2),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: SIMD_WIDTH,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        return;
    }
    enc.setComputePipelineState(&ctx.q8_0_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.q8_0_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm26: chain 용 F32 GEMV(device-in/out) encode. 27B GDN 의 F32 ssm_alpha/beta 무손실
/// device 화. gemv_q8_0 와 동일 인터페이스(byte weight + byte offset), 1-thread/row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_f32(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.f32_chain_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.f32_chain_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Qwen MoE router 전용 F32 GEMV. threadgroup 하나가 output row 하나를 맡는다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_f32_router_simd(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.f32_chain_simd_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.f32_chain_simd_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q5_K GEMV (device-in/out) encode. carrier 용 — qkv/ssm_out 등 Q5_K weight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q5k(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.q5k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.q5k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q5_K GEMV (SIMD-group 협력 + pm21 P1 lane-saturation, device-in/out) encode-only.
/// `encode_gemv_q5k`와 동일 결과(rel<1e-3, simd_sum reduction 순서만 다름). row 당
/// 1 SIMD-group(32 lane), grid = N row. baseline(1-thread/row, div_ceil grid)과 grid/tg
/// 레이아웃이 다름에 주의.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q5k_simd(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    // RNB_METAL_GEMV_COALESCED=1: q5k chain GEMV 를 llama식 nr0=2 multi-row coalesced 커널로.
    // q4k_simd 와 동일 flag(ctx.gemv_coalesced). carrier(qkv/ssm_out) 전 경로 이 함수 거침.
    let coalesced = ctx.gemv_coalesced;
    let pipeline = if coalesced {
        &ctx.q5k_coalesced_pipeline
    } else {
        &ctx.q5k_simd_pipeline
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    // coalesced(nr0=2)는 threadgroup 1개가 output row 2개 → grid=ceil(N/2). 기존 simd 는 1 row/tg.
    let grid_w = if coalesced { n.div_ceil(2) } else { n };
    let grid = MTLSize {
        width: grid_w,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// chain용 Q5_K GEMV 진입점. ctx.gemv_simd flag로 baseline/simd 선택 (RNB_METAL_GEMV_SIMD).
/// baseline encode_gemv_q5k 보존 — 측정 비교용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q5k_auto(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    if ctx.gemv_simd {
        encode_gemv_q5k_simd(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    } else {
        encode_gemv_q5k(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q5k_auto_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    output_byte_offset: usize,
    n: usize,
) {
    let coalesced = ctx.gemv_coalesced;
    let (pipeline, grid_w, tg_w) = if ctx.gemv_simd {
        let pipeline = if coalesced {
            &ctx.q5k_coalesced_pipeline
        } else {
            &ctx.q5k_simd_pipeline
        };
        let rows_per_tg = if coalesced { 2 } else { 1 };
        (pipeline, n.div_ceil(rows_per_tg), SIMD_WIDTH)
    } else {
        (
            &ctx.q5k_pipeline,
            n.div_ceil(ctx.q5k_pipeline.threadExecutionWidth().max(1)),
            ctx.q5k_pipeline.threadExecutionWidth().max(1),
        )
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), input_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), output_byte_offset, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_w,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// quant code(0=Q4_K, 1=Q5_K, 2=Q6_K, 3=Q8_0)별 GEMV encode dispatch (device-in/out).
/// carrier 가 weight 별 quant 에 맞춰 분기 — 모델 특정 quant 하드코딩 금지.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_quant(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: u8,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    match quant {
        0 => encode_gemv_q4k_auto(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n),
        1 => encode_gemv_q5k_auto(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n),
        2 => encode_gemv_q6k_auto(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n),
        3 => encode_gemv_q8_0(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n),
        4 => encode_gemv_f32(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n),
        _ => panic!("unsupported gemv quant code {quant}"),
    }
}

/// F32 GEMV (검증용 host 복사 경로, chain F32 weight 지원). pm26: 27B 의 GDN
/// ssm_alpha/beta 가 F32 raw 라 Q4_K/Q5_K/Q6_K/Q8_0 carrier 가 받지 못해 host
/// per-op 로 추락했던 것을 무손실로 device 화하기 위한 경로. weight_bytes = N*K*4
/// (f32 little-endian). out[row] = sum_c weight[row*K + c] * input[c].
pub fn gemv_f32_with_ctx(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(
        weight_bytes.len(),
        n * k * 4,
        "f32 weight_bytes length (N*K*4)"
    );
    assert_eq!(input.len(), k, "f32 input length");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
        let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
        ctx.device
            .newBufferWithBytes_length_options(nn, len, shared)
            .expect("buf")
    };
    let w_buf = mk(weight_bytes.as_ptr() as *const _, weight_bytes.len());
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("out");
    let nn = n as u32;
    let kk = k as u32;
    let zero = 0u32;
    let n_buf = mk(&nn as *const u32 as *const _, 4);
    let k_buf = mk(&kk as *const u32 as *const _, 4);
    let off_buf = mk(&zero as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gemv_f32(
        ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n).to_vec() }
}

#[cfg(test)]
mod gemv_f32_router_simd_tests {
    use super::*;

    fn assert_case(ctx: &MetalContext, n: usize, k: usize, weight_byte_offset: usize) {
        assert_eq!(weight_byte_offset % std::mem::align_of::<f32>(), 0);
        let input: Vec<f32> = (0..k)
            .map(|c| ((c * 37 % 211) as f32 - 105.0) / 97.0)
            .collect();
        let weight: Vec<f32> = (0..n * k)
            .map(|i| ((i * 17 + i / k * 131) % 257) as f32 / 113.0 - 1.0)
            .collect();
        let expected: Vec<f32> = weight
            .chunks_exact(k)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .fold(0.0f32, |acc, (&w, &x)| acc + w * x)
            })
            .collect();

        let mut weight_bytes = vec![0xA5; weight_byte_offset + weight.len() * 4 + 20];
        for (i, value) in weight.iter().enumerate() {
            let start = weight_byte_offset + i * 4;
            weight_bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
            let ptr = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
            ctx.device
                .newBufferWithBytes_length_options(ptr, len, shared)
                .expect("buf")
        };
        let w_buf = mk(weight_bytes.as_ptr().cast(), weight_bytes.len());
        let in_buf = mk(
            input.as_ptr().cast(),
            std::mem::size_of_val(input.as_slice()),
        );
        let out_buf = ctx
            .device
            .newBufferWithLength_options(n * std::mem::size_of::<f32>(), shared)
            .expect("out");
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let offset_u32 = weight_byte_offset as u32;
        let n_buf = mk((&n_u32 as *const u32).cast(), 4);
        let k_buf = mk((&k_u32 as *const u32).cast(), 4);
        let off_buf = mk((&offset_u32 as *const u32).cast(), 4);

        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = cmd.computeCommandEncoder().expect("enc");
        encode_gemv_f32_router_simd(
            ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let contents: NonNull<std::ffi::c_void> = out_buf.contents();
        let got = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), n) };
        for (row, (&actual, &oracle)) in got.iter().zip(&expected).enumerate() {
            assert!(actual.is_finite(), "row {row}: non-finite output {actual}");
            let tolerance = 2.0e-3 + 2.0e-4 * oracle.abs();
            assert!(
                (actual - oracle).abs() <= tolerance,
                "row {row}: GPU={actual}, CPU={oracle}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn gemv_f32_chain_simd_matches_cpu_for_tail_offset_and_router_shape() {
        let ctx = build_metal_context().expect("no Metal device");
        assert_case(&ctx, 7, 2051, 12);
        assert_case(&ctx, 256, 2048, 20);
    }
}

/// quant code → q4k/q5k/q6k/q8_0 baseline pipeline (inline GEMV 의 out offset 용).
/// baseline 은 1-thread/row 레이아웃(grid=N.div_ceil(tg_width), tg=tg_width).
pub(crate) fn gemv_pipeline_for(
    ctx: &MetalContext,
    quant: u8,
) -> &ProtocolObject<dyn MTLComputePipelineState> {
    match quant {
        0 => &ctx.q4k_pipeline,
        1 => &ctx.q5k_pipeline,
        2 => &ctx.q6k_pipeline,
        3 => &ctx.q8_0_pipeline,
        _ => panic!("unsupported gemv quant code {quant}"),
    }
}

/// quant code → simd pipeline (inline GEMV 의 lane-saturation 경로). simd 커널이 있는
/// quant(Q4_K/Q5_K/Q6_K)만 Some, Q8_0 등은 None(baseline fallback). simd 는 1-SIMD-group
/// /row 레이아웃(grid=N threadgroups, tg=SIMD_WIDTH) — baseline 과 grid/tg 가 다르다.
pub(crate) fn gemv_simd_pipeline_for(
    ctx: &MetalContext,
    quant: u8,
) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
    match quant {
        0 => Some(&ctx.q4k_simd_pipeline),
        1 => Some(&ctx.q5k_simd_pipeline),
        2 => Some(&ctx.q6k_simd_pipeline),
        _ => None,
    }
}

/// quant code → coalesced(nr0=2 multi-row) dispatch layout. q4k/q6k 는 opt-in nsg2 시
/// threadgroup 당 SIMD-group 2개(row 4개), 그 외 coalesced 는 row 2개.
pub(crate) fn gemv_coalesced_dispatch_for(
    ctx: &MetalContext,
    quant: u8,
) -> Option<GemvCoalescedDispatch<'_>> {
    match quant {
        0 if ctx.gemv_coalesced_nsg2 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q4k_coalesced_nsg2_pipeline,
            rows_per_threadgroup: 4,
            threads_per_threadgroup: SIMD_WIDTH * 2,
        }),
        0 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q4k_coalesced_pipeline,
            rows_per_threadgroup: 2,
            threads_per_threadgroup: SIMD_WIDTH,
        }),
        1 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q5k_coalesced_pipeline,
            rows_per_threadgroup: 2,
            threads_per_threadgroup: SIMD_WIDTH,
        }),
        2 if ctx.gemv_coalesced_nsg2 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q6k_coalesced_nsg2_pipeline,
            rows_per_threadgroup: 4,
            threads_per_threadgroup: SIMD_WIDTH * 2,
        }),
        2 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q6k_coalesced_pipeline,
            rows_per_threadgroup: 2,
            threads_per_threadgroup: SIMD_WIDTH,
        }),
        3 => Some(GemvCoalescedDispatch {
            pipeline: &ctx.q8_0_coalesced_pipeline,
            rows_per_threadgroup: 2,
            threads_per_threadgroup: SIMD_WIDTH,
        }),
        _ => None,
    }
}

/// Q6_K GEMV (SIMD-group 협력, device-in/out) encode-only. `encode_gemv_q6k`와 동일
/// 시그니처·동일 결과(rel<1e-3, simd_sum reduction 순서만 다름). row 당 1 SIMD-group(32 lane),
/// grid = N row. Phase 1b.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q6k_simd(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    // RNB_METAL_GEMV_COALESCED=1: q6k chain GEMV 를 llama식 nr0=2 multi-row coalesced 커널로.
    // q4k_simd 와 동일 flag(ctx.gemv_coalesced). carrier(v/ffn_down) 전 경로 이 함수 거침.
    let coalesced = ctx.gemv_coalesced;
    let nsg2 = coalesced && ctx.gemv_coalesced_nsg2;
    let pipeline = if nsg2 {
        &ctx.q6k_coalesced_nsg2_pipeline
    } else if coalesced {
        &ctx.q6k_coalesced_pipeline
    } else {
        &ctx.q6k_simd_pipeline
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    // coalesced는 row 2개/tg, nsg2는 SIMD-group 2개로 row 4개/tg. 기존 simd는 1 row/tg.
    let grid_w = if nsg2 {
        n.div_ceil(4)
    } else if coalesced {
        n.div_ceil(2)
    } else {
        n
    };
    let grid = MTLSize {
        width: grid_w,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: if nsg2 { SIMD_WIDTH * 2 } else { SIMD_WIDTH },
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// chain용 Q6_K GEMV 진입점. ctx.gemv_simd flag로 baseline/simd 선택 (Phase 1의 RNB_METAL_GEMV_SIMD 재사용).
/// baseline encode_gemv_q6k/standalone 보존.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q6k_auto(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    if ctx.gemv_simd {
        encode_gemv_q6k_simd(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    } else {
        encode_gemv_q6k(ctx, enc, w_buf, in_buf, out_buf, n_buf, k_buf, off_buf, n);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_q6k_auto_offset(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    input_byte_offset: usize,
    output_byte_offset: usize,
    n: usize,
) {
    let coalesced = ctx.gemv_coalesced;
    let nsg2 = coalesced && ctx.gemv_coalesced_nsg2;
    let (pipeline, grid_w, tg_w) = if ctx.gemv_simd {
        let pipeline = if nsg2 {
            &ctx.q6k_coalesced_nsg2_pipeline
        } else if coalesced {
            &ctx.q6k_coalesced_pipeline
        } else {
            &ctx.q6k_simd_pipeline
        };
        let rows_per_tg = if nsg2 {
            4
        } else if coalesced {
            2
        } else {
            1
        };
        let tg_w = if nsg2 { SIMD_WIDTH * 2 } else { SIMD_WIDTH };
        (pipeline, n.div_ceil(rows_per_tg), tg_w)
    } else {
        (
            &ctx.q6k_pipeline,
            n.div_ceil(ctx.q6k_pipeline.threadExecutionWidth().max(1)),
            ctx.q6k_pipeline.threadExecutionWidth().max(1),
        )
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), input_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), output_byte_offset, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_w,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// Q6_K GEMV (SIMD 협력, 복사 경로). 측정/테스트용.
/// `gemv_q4k_simd_dispatch`와 동일 구조 — pipeline/grid/tg만 Q6_K용.
pub(crate) fn gemv_q6k_simd_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q6_K block size)");
    assert_eq!(input.len(), k);
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.q6k_simd_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
    }
    // 1 threadgroup = 1 SIMD-group(32 lane), 1 row 담당. grid = N row.
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out_slice.to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputArgmaxDispatch {
    pub token_id: u32,
    pub wait_us: u64,
}

pub(crate) struct OutputArgmaxScratch {
    n: usize,
    k: usize,
    input_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    token_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl OutputArgmaxScratch {
    pub(crate) fn new(ctx: &MetalContext, n: usize, k: usize) -> Self {
        assert!(n > 0, "N must be > 0");
        assert!(k > 0, "K must be > 0");
        let shared = MTLResourceOptions::StorageModeShared;
        let input_buf = ctx
            .device
            .newBufferWithLength_options(k * std::mem::size_of::<f32>(), shared)
            .expect("Metal: failed to create output argmax input buffer");
        let logits_buf = ctx
            .device
            .newBufferWithLength_options(n * std::mem::size_of::<f32>(), shared)
            .expect("Metal: failed to create output argmax logits buffer");
        let token_buf = ctx
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .expect("Metal: failed to create output argmax token buffer");
        let n_buf = ctx
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .expect("Metal: failed to create output argmax n buffer");
        let k_buf = ctx
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .expect("Metal: failed to create output argmax k buffer");
        let off_buf = ctx
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .expect("Metal: failed to create output argmax offset buffer");
        store_u32(&n_buf, n as u32);
        store_u32(&k_buf, k as u32);
        Self {
            n,
            k,
            input_buf,
            logits_buf,
            token_buf,
            n_buf,
            k_buf,
            off_buf,
        }
    }

    fn upload_input(&self, input: &[f32]) {
        assert_eq!(input.len(), self.k);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buf.contents().as_ptr() as *mut f32,
                input.len(),
            );
        }
    }

    fn token_id(&self) -> u32 {
        unsafe { *(self.token_buf.contents().as_ptr() as *const u32) }
    }
}

fn store_u32(buf: &ProtocolObject<dyn MTLBuffer>, value: u32) {
    unsafe {
        *(buf.contents().as_ptr() as *mut u32) = value;
    }
}

pub(crate) fn encode_argmax_f32(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    logits_buf: &ProtocolObject<dyn MTLBuffer>,
    token_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    enc.setComputePipelineState(&ctx.output_argmax_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(token_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 2);
    }
    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

pub(crate) fn gemv_quant_simd_argmax_dispatch_reuse(
    ctx: &MetalContext,
    quant: u8,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
    scratch: &mut OutputArgmaxScratch,
) -> OutputArgmaxDispatch {
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    assert_eq!(input.len(), k);
    assert!(matches!(quant, 0 | 2), "argmax supports Q4_K/Q6_K");
    assert!(n > 0, "N must be > 0");
    assert_eq!(scratch.n, n, "output argmax scratch N mismatch");
    assert_eq!(scratch.k, k, "output argmax scratch K mismatch");
    let started = std::time::Instant::now();
    scratch.upload_input(input);
    store_u32(&scratch.off_buf, weight_byte_offset);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    match quant {
        0 => encode_gemv_q4k_simd(
            ctx,
            &enc,
            w_buf,
            &scratch.input_buf,
            &scratch.logits_buf,
            &scratch.n_buf,
            &scratch.k_buf,
            &scratch.off_buf,
            n,
        ),
        2 => encode_gemv_q6k_simd(
            ctx,
            &enc,
            w_buf,
            &scratch.input_buf,
            &scratch.logits_buf,
            &scratch.n_buf,
            &scratch.k_buf,
            &scratch.off_buf,
            n,
        ),
        _ => unreachable!("argmax supports only Q4_K/Q6_K"),
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    encode_argmax_f32(
        ctx,
        &enc,
        &scratch.logits_buf,
        &scratch.token_buf,
        &scratch.n_buf,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    OutputArgmaxDispatch {
        token_id: scratch.token_id(),
        wait_us: started.elapsed().as_micros() as u64,
    }
}

pub(crate) fn gemv_quant_simd_argmax_dispatch_once(
    ctx: &MetalContext,
    quant: u8,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> OutputArgmaxDispatch {
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    assert_eq!(input.len(), k);
    assert!(matches!(quant, 0 | 2), "argmax supports Q4_K/Q6_K");
    assert!(n > 0, "N must be > 0");
    let started = std::time::Instant::now();
    let shared = MTLResourceOptions::StorageModeShared;

    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>())
    };

    let input_buf = mk_bytes(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let logits_buf = ctx
        .device
        .newBufferWithLength_options(n * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create logits buffer");
    let token_buf = ctx
        .device
        .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
        .expect("Metal: failed to create token buffer");
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    match quant {
        0 => encode_gemv_q4k_simd(
            ctx,
            &enc,
            w_buf,
            &input_buf,
            &logits_buf,
            &n_buf,
            &k_buf,
            &off_buf,
            n,
        ),
        2 => encode_gemv_q6k_simd(
            ctx,
            &enc,
            w_buf,
            &input_buf,
            &logits_buf,
            &n_buf,
            &k_buf,
            &off_buf,
            n,
        ),
        _ => unreachable!("argmax supports only Q4_K/Q6_K"),
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    encode_argmax_f32(ctx, &enc, &logits_buf, &token_buf, &n_buf);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    OutputArgmaxDispatch {
        token_id: unsafe { *(token_buf.contents().as_ptr() as *const u32) },
        wait_us: started.elapsed().as_micros() as u64,
    }
}

/// Q8_0 GEMV (host 복사 경로, 검증용). block_bytes = N*(K/32)*34.
pub fn gemv_q8_0_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 32;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 34,
        "q8_0 block_bytes length"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
        let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
        ctx.device
            .newBufferWithBytes_length_options(nn, len, shared)
            .expect("buf")
    };
    let w_buf = mk(block_bytes.as_ptr() as *const _, block_bytes.len());
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("out");
    let nn = n as u32;
    let kk = k as u32;
    let zero = 0u32;
    let n_buf = mk(&nn as *const u32 as *const _, 4);
    let k_buf = mk(&kk as *const u32 as *const _, 4);
    let off_buf = mk(&zero as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gemv_q8_0(
        ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n).to_vec() }
}

/// Q5_K GEMV (host 복사 경로, 검증용). block_bytes = N*(K/256)*176.
pub fn gemv_q5k_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 176,
        "q5k block_bytes length"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
        let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
        ctx.device
            .newBufferWithBytes_length_options(nn, len, shared)
            .expect("buf")
    };
    let w_buf = mk(block_bytes.as_ptr() as *const _, block_bytes.len());
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("out");
    let nn = n as u32;
    let kk = k as u32;
    let zero = 0u32;
    let n_buf = mk(&nn as *const u32 as *const _, 4);
    let k_buf = mk(&kk as *const u32 as *const _, 4);
    let off_buf = mk(&zero as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gemv_q5k(
        ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n).to_vec() }
}

/// Q5_K GEMV (SIMD-group + pm21 P1 lane-saturation, 검증용). block_bytes = N*(K/256)*176.
pub fn gemv_q5k_simd_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 176,
        "q5k_simd block_bytes length"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
        let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
        ctx.device
            .newBufferWithBytes_length_options(nn, len, shared)
            .expect("buf")
    };
    let w_buf = mk(block_bytes.as_ptr() as *const _, block_bytes.len());
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("out");
    let nn = n as u32;
    let kk = k as u32;
    let zero = 0u32;
    let n_buf = mk(&nn as *const u32 as *const _, 4);
    let k_buf = mk(&kk as *const u32 as *const _, 4);
    let off_buf = mk(&zero as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gemv_q5k_simd(
        ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n).to_vec() }
}

/// Q6_K GEMV (host 복사 경로, 검증용). block_bytes = N*(K/256)*210.
pub fn gemv_q6k_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "q6k block_bytes length"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk = |ptr: *const std::ffi::c_void, len: usize| unsafe {
        let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr");
        ctx.device
            .newBufferWithBytes_length_options(nn, len, shared)
            .expect("buf")
    };
    let w_buf = mk(block_bytes.as_ptr() as *const _, block_bytes.len());
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("out");
    let nn = n as u32;
    let kk = k as u32;
    let zero = 0u32;
    let n_buf = mk(&nn as *const u32 as *const _, 4);
    let k_buf = mk(&kk as *const u32 as *const _, 4);
    let off_buf = mk(&zero as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gemv_q6k(
        ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n).to_vec() }
}

/// Q6_K GEMV (SIMD 협력, host 복사 경로, 검증용). block_bytes = N*(K/256)*210.
pub fn gemv_q6k_simd_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "q6k_simd block_bytes length"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemv_q6k_simd_dispatch(ctx, &w_buf, 0, input, n, k)
}

/// Per-head q/k RMSNorm 을 encoder 에 encode (commit 안 함). in/weight/out device
/// buffer. grid = num_heads(threadgroup 1개=head 1개), tg=256(tree reduction).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qk_norm(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.qk_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 4);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// text M-RoPE in-place(data device buffer) 를 encoder 에 encode. grid=num_heads,
/// tg=1 (head 별 angle 누적 순차). num_heads = dim/head_dim.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rope_mrope(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    nrot_buf: &ProtocolObject<dyn MTLBuffer>,
    theta_buf: &ProtocolObject<dyn MTLBuffer>,
    pos_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.rope_mrope_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(data_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(nrot_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(theta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 5);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// gated attention q projection 출력(head 별 [query|gate] 인터리브, q_full device)을
/// 연속 query/gate device buffer 로 분리하는 split 을 encoder 에 encode. grid=num_heads.
pub(crate) fn encode_split_qgate(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_full_buf: &ProtocolObject<dyn MTLBuffer>,
    query_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    hd: usize,
) {
    enc.setComputePipelineState(&ctx.split_qgate_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_full_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(query_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 3);
    }
    // threadgroup=head, thread=head_dim 차원(d). 단순 copy 라 lane 간 독립 → bit-identical.
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: hd,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm45 M2-1: conv_data(device) → 연속 q/k/v device buffer 로 3-way 분리(bit-identical copy).
/// host `gdn_prefill.rs split_conv_qkv` 1:1. flat 1D grid (1 thread = 1 element, total 초과 return).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_split_conv_qkv(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    conv_buf: &ProtocolObject<dyn MTLBuffer>,
    conv_offset: usize,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len_buf: &ProtocolObject<dyn MTLBuffer>,
    conv_channels_buf: &ProtocolObject<dyn MTLBuffer>,
    q_dim_buf: &ProtocolObject<dyn MTLBuffer>,
    k_dim_buf: &ProtocolObject<dyn MTLBuffer>,
    v_dim_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    conv_channels: usize,
) {
    enc.setComputePipelineState(&ctx.split_conv_qkv_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(conv_buf), conv_offset, 0);
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(seq_len_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(conv_channels_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(q_dim_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(k_dim_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(v_dim_buf), 0, 8);
    }
    let total = seq_len * conv_channels;
    let tg_width = 256usize;
    let grid = MTLSize {
        width: total.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm45 M2-2: GQA q/k(num_k_heads)를 num_v_heads 로 순환 repeat(bit-identical gather).
/// host `gdn_prefill.rs repeat_qk_for_value_heads` 1:1. q/k 한 커널에서 동시 처리.
/// num_v_heads == num_k_heads 면 identity copy. flat 1D grid (1 thread = 1 element).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_repeat_qk(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_in_buf: &ProtocolObject<dyn MTLBuffer>,
    k_in_buf: &ProtocolObject<dyn MTLBuffer>,
    q_out_buf: &ProtocolObject<dyn MTLBuffer>,
    k_out_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len_buf: &ProtocolObject<dyn MTLBuffer>,
    num_k_heads_buf: &ProtocolObject<dyn MTLBuffer>,
    num_v_heads_buf: &ProtocolObject<dyn MTLBuffer>,
    head_k_dim_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
) {
    enc.setComputePipelineState(&ctx.repeat_qk_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(q_out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(k_out_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(seq_len_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(num_k_heads_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(num_v_heads_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(head_k_dim_buf), 0, 7);
    }
    let total = seq_len * num_v_heads * head_k_dim;
    let tg_width = 256usize;
    let grid = MTLSize {
        width: total.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Partial RoPE in-place(인접페어, decode 1 token) 을 encoder 에 encode. grid=num_heads.
/// `theta_scale_buf` 는 host f32 precompute(theta.powf(-2/n_rot), clamp 후) 값.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rope_partial(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    nrot_buf: &ProtocolObject<dyn MTLBuffer>,
    theta_scale_buf: &ProtocolObject<dyn MTLBuffer>,
    pos_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.rope_partial_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(data_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(nrot_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(theta_scale_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 5);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// gated attention `attn_out *= sigmoid(gate)` 를 encoder 에 encode. grid=n elementwise.
pub(crate) fn encode_gate_apply(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    attn_out_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.gate_apply_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(attn_out_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 2);
    }
    let tg_width = ctx.gate_apply_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// prefill gated q projection 출력([seq, heads, query|gate])을 연속 query/gate buffer로 분리한다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_prefill_split_q_gate(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_full_buf: &ProtocolObject<dyn MTLBuffer>,
    q_out_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    elems: usize,
) {
    let pipeline = ctx
        .prefill_split_gate_pipeline
        .as_ref()
        .expect("prefill split/gate pipeline must be built before ATN core dispatch");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_full_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(q_out_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(seq_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 5);
    }
    let tg_width = 256usize;
    let grid = MTLSize {
        width: elems.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// prefill gated attention output을 별도 output buffer에 쓴다.
pub(crate) fn encode_prefill_gate_apply(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    attn_out_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    elems_buf: &ProtocolObject<dyn MTLBuffer>,
    elems: usize,
) {
    let pipeline = ctx
        .prefill_gate_apply_pipeline
        .as_ref()
        .expect("prefill gate apply pipeline must be built before ATN core dispatch");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(attn_out_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(elems_buf), 0, 3);
    }
    let tg_width = pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: elems.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// kv_append(k/v device f32 → KV_dev[pos] f16) 을 encoder 에 encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_kv_append(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    k_in_buf: &ProtocolObject<dyn MTLBuffer>,
    v_in_buf: &ProtocolObject<dyn MTLBuffer>,
    k_cache_buf: &ProtocolObject<dyn MTLBuffer>,
    v_cache_buf: &ProtocolObject<dyn MTLBuffer>,
    kvd_buf: &ProtocolObject<dyn MTLBuffer>,
    pos_buf: &ProtocolObject<dyn MTLBuffer>,
    kv_dim: usize,
) {
    enc.setComputePipelineState(&ctx.kv_append_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(k_in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(v_in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(k_cache_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(v_cache_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(kvd_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 5);
    }
    let tg_width = ctx.kv_append_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: kv_dim.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// attn_decode(q device + KV device → o_buf) 를 encoder 에 encode. grid=num_heads,
/// tg=SIMD_WIDTH(1 head=1 SIMD-group).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_decode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    nkv_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    kl_buf: &ProtocolObject<dyn MTLBuffer>,
    scale_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.attn_decode_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(nkv_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(kl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(scale_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&ctx.attn_read_mask_buf), 0, 9);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// int8 KV decode attention(q device + int8 K·V + per-slot scale → o_buf)을 encoder 에
/// encode. buffer 0-10(read_mask 없음). grid=num_heads, tg=SIMD_WIDTH(1 head=1 SIMD-group).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_decode_i8(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_i8: &ProtocolObject<dyn MTLBuffer>,
    v_i8: &ProtocolObject<dyn MTLBuffer>,
    k_scale: &ProtocolObject<dyn MTLBuffer>,
    v_scale: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    nkv_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    kl_buf: &ProtocolObject<dyn MTLBuffer>,
    scale_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_dim: usize,
) {
    assert!(
        head_dim <= 8 * SIMD_WIDTH,
        "attn_decode_i8 head_dim {head_dim} > 256 (lane-local bound)"
    );
    enc.setComputePipelineState(&ctx.attn_decode_i8_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_i8), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_i8), 0, 2);
        enc.setBuffer_offset_atIndex(Some(k_scale), 0, 3);
        enc.setBuffer_offset_atIndex(Some(v_scale), 0, 4);
        enc.setBuffer_offset_atIndex(Some(o_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(nkv_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(kl_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(scale_buf), 0, 10);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// split-K int8 KV decode attention(q + int8 K/V + per-slot scale → o_buf).
/// partial buffers are [num_splits*num_heads*head_dim] and [num_splits*num_heads].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_decode_i8_splitk(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_i8: &ProtocolObject<dyn MTLBuffer>,
    v_i8: &ProtocolObject<dyn MTLBuffer>,
    k_scale: &ProtocolObject<dyn MTLBuffer>,
    v_scale: &ProtocolObject<dyn MTLBuffer>,
    partial_acc: &ProtocolObject<dyn MTLBuffer>,
    partial_m: &ProtocolObject<dyn MTLBuffer>,
    partial_s: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    nkv_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    kl_buf: &ProtocolObject<dyn MTLBuffer>,
    scale_buf: &ProtocolObject<dyn MTLBuffer>,
    splits_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_dim: usize,
    num_splits: usize,
) {
    assert!(
        head_dim <= 8 * SIMD_WIDTH,
        "attn_decode_i8_splitk head_dim {head_dim} > 256 (lane-local bound)"
    );
    assert!(num_splits >= 2, "attn_decode_i8_splitk requires >=2 splits");

    enc.setComputePipelineState(&ctx.attn_decode_i8_splitk_part_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_i8), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_i8), 0, 2);
        enc.setBuffer_offset_atIndex(Some(k_scale), 0, 3);
        enc.setBuffer_offset_atIndex(Some(v_scale), 0, 4);
        enc.setBuffer_offset_atIndex(Some(partial_acc), 0, 5);
        enc.setBuffer_offset_atIndex(Some(partial_m), 0, 6);
        enc.setBuffer_offset_atIndex(Some(partial_s), 0, 7);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(nkv_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(kl_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(scale_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(splits_buf), 0, 13);
    }
    let part_grid = MTLSize {
        width: num_heads,
        height: num_splits,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(part_grid, tg);
    chain_barrier(ctx, enc);

    enc.setComputePipelineState(&ctx.attn_decode_i8_splitk_reduce_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(partial_acc), 0, 0);
        enc.setBuffer_offset_atIndex(Some(partial_m), 0, 1);
        enc.setBuffer_offset_atIndex(Some(partial_s), 0, 2);
        enc.setBuffer_offset_atIndex(Some(o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(splits_buf), 0, 6);
    }
    let reduce_grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(reduce_grid, tg);
}

/// Q6_K GEMV 를 device buffer in/out 으로 compute encoder 에 encode (commit/readback 안 함).
/// `encode_gemv_q4k` 와 동일한 buffer 바인딩 규약 — pipeline 만 Q6_K(block 210B/256).
/// chain 의 down(N=hidden_dim, K=ffn_dim) GEMV 가 Q6_K 일 때 사용.
pub(crate) fn encode_gemv_q6k(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(&ctx.q6k_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
    }
    let tg_width = ctx.q6k_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: n.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// attn_decode — single-token decode attention (QK^T → online softmax → AV)
// ---------------------------------------------------------------------------
//
// M1(전체 device decode chain): attention compute 를 device 로. CPU
// `attention_decode_flash`(f16 accumulator + branched online softmax,
// production default) 와 token-identical 목표. grid = num_heads threadgroups,
// 각 1 SIMD-group(SIMD_WIDTH lane). KV cache 는 f16 bits(u16).

/// int8 KV decode attention oneshot(테스트용 복사 경로). q/k_i8/v_i8/k_scale/v_scale 를
/// buffer 로 올리고 o_buf + scalar buffer 를 만들어 단일 command buffer 로
/// encode_attn_decode_i8 → commit/wait → o readback. `attn_decode_with_ctx` 패턴과
/// 동일하되 K·V 는 char(int8), per-slot scale 은 f32 buffer.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_i8_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k_i8: &[i8],
    v_i8: &[i8],
    k_scale: &[f32],
    v_scale: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q.len(), num_heads * head_dim, "q length mismatch");
    assert_eq!(k_i8.len(), kv_len * kv_dim, "k_i8 length mismatch");
    assert_eq!(v_i8.len(), kv_len * kv_dim, "v_i8 length mismatch");
    assert_eq!(
        k_scale.len(),
        kv_len * num_kv_heads,
        "k_scale length mismatch"
    );
    assert_eq!(
        v_scale.len(),
        kv_len * num_kv_heads,
        "v_scale length mismatch"
    );
    assert!(
        num_heads % num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k_i8.as_ptr() as *const _, std::mem::size_of_val(k_i8));
    let v_buf = mk_bytes(v_i8.as_ptr() as *const _, std::mem::size_of_val(v_i8));
    let ks_buf = mk_bytes(k_scale.as_ptr() as *const _, std::mem::size_of_val(k_scale));
    let vs_buf = mk_bytes(v_scale.as_ptr() as *const _, std::mem::size_of_val(v_scale));

    let out_len = num_heads * head_dim;
    let out_bytes = out_len * std::mem::size_of::<f32>();
    let o_buf = ctx
        .device
        .newBufferWithLength_options(out_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let nh = num_heads as u32;
    let nkv = num_kv_heads as u32;
    let hd = head_dim as u32;
    let kl = kv_len as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_u32(&nh);
    let nkv_buf = mk_u32(&nkv);
    let hd_buf = mk_u32(&hd);
    let kl_buf = mk_u32(&kl);
    let scale_buf = mk_bytes(&scale as *const f32 as *const _, std::mem::size_of::<f32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_attn_decode_i8(
        ctx, &enc, &q_buf, &k_buf, &v_buf, &ks_buf, &vs_buf, &o_buf, &nh_buf, &nkv_buf, &hd_buf,
        &kl_buf, &scale_buf, num_heads, head_dim,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, out_len) };
    out_slice.to_vec()
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_i8_splitk_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k_i8: &[i8],
    v_i8: &[i8],
    k_scale: &[f32],
    v_scale: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    num_splits: usize,
) -> Vec<f32> {
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q.len(), num_heads * head_dim, "q length mismatch");
    assert_eq!(k_i8.len(), kv_len * kv_dim, "k_i8 length mismatch");
    assert_eq!(v_i8.len(), kv_len * kv_dim, "v_i8 length mismatch");
    assert_eq!(
        k_scale.len(),
        kv_len * num_kv_heads,
        "k_scale length mismatch"
    );
    assert_eq!(
        v_scale.len(),
        kv_len * num_kv_heads,
        "v_scale length mismatch"
    );
    assert!(
        num_heads % num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );
    assert!(
        head_dim <= 8 * SIMD_WIDTH,
        "head_dim must be <= {} (attn_decode_i8_splitk lane-local bound)",
        8 * SIMD_WIDTH
    );
    assert!(num_splits >= 2, "num_splits must be >= 2");

    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k_i8.as_ptr() as *const _, std::mem::size_of_val(k_i8));
    let v_buf = mk_bytes(v_i8.as_ptr() as *const _, std::mem::size_of_val(v_i8));
    let ks_buf = mk_bytes(k_scale.as_ptr() as *const _, std::mem::size_of_val(k_scale));
    let vs_buf = mk_bytes(v_scale.as_ptr() as *const _, std::mem::size_of_val(v_scale));

    let out_len = num_heads * head_dim;
    let o_buf = ctx
        .device
        .newBufferWithLength_options(out_len * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create output buffer");
    let partial_acc = ctx
        .device
        .newBufferWithLength_options(num_splits * out_len * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create splitk partial_acc buffer");
    let partial_m = ctx
        .device
        .newBufferWithLength_options(num_splits * num_heads * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create splitk partial_m buffer");
    let partial_s = ctx
        .device
        .newBufferWithLength_options(num_splits * num_heads * std::mem::size_of::<f32>(), shared)
        .expect("Metal: failed to create splitk partial_s buffer");

    let nh = num_heads as u32;
    let nkv = num_kv_heads as u32;
    let hd = head_dim as u32;
    let kl = kv_len as u32;
    let splits = num_splits as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_u32(&nh);
    let nkv_buf = mk_u32(&nkv);
    let hd_buf = mk_u32(&hd);
    let kl_buf = mk_u32(&kl);
    let splits_buf = mk_u32(&splits);
    let scale_buf = mk_bytes(&scale as *const f32 as *const _, std::mem::size_of::<f32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_attn_decode_i8_splitk(
        ctx,
        &enc,
        &q_buf,
        &k_buf,
        &v_buf,
        &ks_buf,
        &vs_buf,
        &partial_acc,
        &partial_m,
        &partial_s,
        &o_buf,
        &nh_buf,
        &nkv_buf,
        &hd_buf,
        &kl_buf,
        &scale_buf,
        &splits_buf,
        num_heads,
        head_dim,
        num_splits,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, out_len) };
    out_slice.to_vec()
}

/// Single-token decode attention. 측정/테스트용 복사 경로(매 호출 buffer 생성).
/// seam 접속 시엔 KV cache device buffer + resident 경로로 교체한다(M1b).
pub fn attn_decode_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(
        !ctx.kv_int8,
        "attn_decode_with_ctx: int8 ctx에서 f16 헬퍼 호출 금지"
    );
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q.len(), num_heads * head_dim, "q length mismatch");
    assert_eq!(k_cache.len(), kv_len * kv_dim, "k_cache length mismatch");
    assert_eq!(v_cache.len(), kv_len * kv_dim, "v_cache length mismatch");
    assert!(
        num_heads % num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );
    // 커널의 lane-local 배열(qf[8]/acc[8]) 상한: head_dim <= 8*SIMD_WIDTH.
    assert!(
        head_dim <= 8 * SIMD_WIDTH,
        "head_dim must be <= {} (attn_decode lane-local bound)",
        8 * SIMD_WIDTH
    );
    let shared = MTLResourceOptions::StorageModeShared;

    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k_cache.as_ptr() as *const _, std::mem::size_of_val(k_cache));
    let v_buf = mk_bytes(v_cache.as_ptr() as *const _, std::mem::size_of_val(v_cache));

    attn_decode_dispatch_buffers(
        ctx,
        &q_buf,
        &k_buf,
        &v_buf,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
    )
}

/// attn_decode 커널 dispatch (buffer 입력). q/k/v buffer 는 호출자 소유.
/// 복사 경로(`attn_decode_with_ctx`, 매번 buffer 생성)와 resident 경로
/// (`KvResident`, KV device buffer 재사용)가 공유. k_buf/v_buf 는
/// [≥kv_len*kv_dim] u16(f16 bits) — 커널은 j<kv_len 까지만 읽으니 capacity 가 더
/// 커도 무방. 단일 command buffer commit/wait + readback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_decode_dispatch_buffers(
    ctx: &MetalContext,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };

    let out_len = num_heads * head_dim;
    let out_bytes = out_len * std::mem::size_of::<f32>();
    let o_buf = ctx
        .device
        .newBufferWithLength_options(out_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let nh = num_heads as u32;
    let nkv = num_kv_heads as u32;
    let hd = head_dim as u32;
    let kl = kv_len as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_u32(&nh);
    let nkv_buf = mk_u32(&nkv);
    let hd_buf = mk_u32(&hd);
    let kl_buf = mk_u32(&kl);
    let scale_buf = mk_bytes(&scale as *const f32 as *const _, std::mem::size_of::<f32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.attn_decode_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&nh_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&nkv_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&kl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&scale_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&ctx.attn_read_mask_buf), 0, 9);
    }
    // grid = num_heads threadgroups, 각 1 SIMD-group(32 lane).
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    if attn_resident_trace_enabled() {
        eprintln!(
            "[metal-attn-decode] dispatch nh={num_heads} nkv={num_kv_heads} hd={head_dim} kv_len={kv_len} out_len={out_len}"
        );
    }
    cmd.commit();
    cmd.waitUntilCompleted();
    let status = cmd.status();
    if status != MTLCommandBufferStatus::Completed {
        let error = cmd
            .error()
            .map(|err| format!("{err:?}"))
            .unwrap_or_else(|| "no NSError attached".to_string());
        panic!("Metal attn_decode command failed status={status:?} error={error}");
    }

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, out_len) };
    out_slice.to_vec()
}

/// 단일 layer 의 KV cache 를 metal buffer 에 상주시켜, decode 매 토큰마다 전체 KV
/// (kv_len*kv_dim)를 업로드하던 것을 1 token(kv_dim) append 로 줄인다.
/// k_buf/v_buf 는 [capacity*kv_dim] u16(f16 bits), StorageModeShared 라 contents()
/// 에 직접 write. attn_decode 커널이 device buffer 를 직접 읽는다.
pub struct KvResident {
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// int8 경로(pm22) 전용 버퍼. per-slot int8 값([capacity*kv_dim] i8)과
    /// per-slot scale([capacity*num_kv_heads] f32). f16 k_buf/v_buf 와 택일.
    pub(crate) k_i8: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) v_i8: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) k_scale: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) v_scale: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub kv_dim: usize,
    pub capacity: usize,
    /// device buffer 에 이미 채워진 token 수(incremental append 추적).
    pub(crate) filled: usize,
    /// 이 cache 가 int8 양자화 경로인지(pm22). f16 버퍼와 택일. ctx.kv_int8 에서 결정.
    pub(crate) kv_int8: bool,
}

impl KvResident {
    pub fn new(ctx: &MetalContext, num_kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        Self::new_with_format(ctx, num_kv_heads, head_dim, capacity, ctx.kv_int8)
    }

    pub fn new_f16(
        ctx: &MetalContext,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Self {
        Self::new_with_format(ctx, num_kv_heads, head_dim, capacity, false)
    }

    fn new_with_format(
        ctx: &MetalContext,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        kv_int8: bool,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let shared = MTLResourceOptions::StorageModeShared;
        let mkb = |n: usize| {
            ctx.device
                .newBufferWithLength_options(n.max(1), shared)
                .expect("Metal: failed to create KV resident buffer")
        };
        if kv_int8 {
            // int8 경로: per-slot int8 값 + per-slot scale 만 alloc.
            // f16 k_buf/v_buf 는 1-byte dummy — Task 6 가드가 f16 커널 진입을
            // panic 으로 막아 dummy 는 절대 read 되지 않는다.
            let i8b = capacity * kv_dim; // char = 1byte
            let scb = capacity * num_kv_heads * std::mem::size_of::<f32>();
            Self {
                k_buf: mkb(1),
                v_buf: mkb(1),
                k_i8: Some(mkb(i8b)),
                v_i8: Some(mkb(i8b)),
                k_scale: Some(mkb(scb)),
                v_scale: Some(mkb(scb)),
                num_kv_heads,
                head_dim,
                kv_dim,
                capacity,
                filled: 0,
                kv_int8: true,
            }
        } else {
            let bytes = capacity * kv_dim * std::mem::size_of::<u16>();
            Self {
                k_buf: mkb(bytes),
                v_buf: mkb(bytes),
                k_i8: None,
                v_i8: None,
                k_scale: None,
                v_scale: None,
                num_kv_heads,
                head_dim,
                kv_dim,
                capacity,
                filled: 0,
                kv_int8: false,
            }
        }
    }

    /// host KV(전체 kv_len, f16 bits)에서 device 에 아직 없는 [filled, target_len)
    /// 구간만 복사한다. decode 첫 토큰은 prefill 전체(0..target_len), 이후 토큰은
    /// 1개씩만 복사 → 매 토큰 전체 KV 업로드 제거(residency 의 핵심 이득).
    pub fn ensure_filled(&mut self, k_all: &[u16], v_all: &[u16], target_len: usize) {
        assert!(
            target_len <= self.capacity,
            "kv target_len={target_len} exceeds capacity={}",
            self.capacity
        );
        assert!(
            k_all.len() >= target_len * self.kv_dim && v_all.len() >= target_len * self.kv_dim,
            "k_all/v_all shorter than target_len*kv_dim"
        );
        if target_len <= self.filled {
            return;
        }
        if attn_resident_trace_enabled() {
            eprintln!(
                "[metal-attn-resident] fill f16 {old}->{target_len} kv_dim={} capacity={} k_len={} v_len={}",
                self.kv_dim,
                self.capacity,
                k_all.len(),
                v_all.len(),
                old = self.filled
            );
        }
        unsafe {
            let kp = self.k_buf.contents().as_ptr() as *mut u16;
            let vp = self.v_buf.contents().as_ptr() as *mut u16;
            let start = self.filled * self.kv_dim;
            let count = (target_len - self.filled) * self.kv_dim;
            std::ptr::copy_nonoverlapping(k_all[start..].as_ptr(), kp.add(start), count);
            std::ptr::copy_nonoverlapping(v_all[start..].as_ptr(), vp.add(start), count);
        }
        self.filled = target_len;
    }

    /// int8 경로(pm22): host f16 KV([filled, target_len) 구간)를 per-slot int8 로
    /// 변환해 device 를 채운다. host f16 bits → f32 → quantize_slot_i8_ref →
    /// device int8/scale write. f16 ensure_filled 의 int8 대응(prefill 변환).
    pub fn ensure_filled_i8(&mut self, k_all: &[u16], v_all: &[u16], target_len: usize) {
        assert!(self.kv_int8, "ensure_filled_i8 on f16 KvResident");
        assert!(
            target_len <= self.capacity,
            "kv target_len={target_len} exceeds capacity={}",
            self.capacity
        );
        if target_len <= self.filled {
            return;
        }
        // pm23: first-token prior-KV 변환 비용 측정용 타이밍(env-gated). 1회성 비용이라
        // break-even 토큰 수 산정에 필요. 변환 slot 수도 같이 찍어 elem 단위 환산 가능.
        let timing = std::env::var("RNB_METAL_KV_INT8_TIMING").as_deref() == Ok("1");
        let t_convert = if timing {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let convert_from = self.filled;
        let kp = self.k_i8.as_ref().unwrap().contents().as_ptr() as *mut i8;
        let vp = self.v_i8.as_ref().unwrap().contents().as_ptr() as *mut i8;
        let ksp = self.k_scale.as_ref().unwrap().contents().as_ptr() as *mut f32;
        let vsp = self.v_scale.as_ref().unwrap().contents().as_ptr() as *mut f32;
        for pos in self.filled..target_len {
            for h in 0..self.num_kv_heads {
                let off = pos * self.kv_dim + h * self.head_dim;
                let ks: Vec<f32> = (0..self.head_dim)
                    .map(|d| f16_bits_to_f32(k_all[off + d]))
                    .collect();
                let vs: Vec<f32> = (0..self.head_dim)
                    .map(|d| f16_bits_to_f32(v_all[off + d]))
                    .collect();
                let (kq, ksc) = quantize_slot_i8_ref(&ks);
                let (vq, vsc) = quantize_slot_i8_ref(&vs);
                let sidx = pos * self.num_kv_heads + h;
                unsafe {
                    *ksp.add(sidx) = ksc;
                    *vsp.add(sidx) = vsc;
                    for d in 0..self.head_dim {
                        *kp.add(off + d) = kq[d];
                        *vp.add(off + d) = vq[d];
                    }
                }
            }
        }
        self.filled = target_len;
        if let Some(t0) = t_convert {
            let slots = (target_len - convert_from) * self.num_kv_heads;
            let elems = slots * self.head_dim * 2; // K + V
            let us = t0.elapsed().as_micros();
            eprintln!(
                "[kv_int8_convert] ensure_filled_i8: {convert_from}->{target_len} pos, \
                 {slots} slots, {elems} elems (K+V), {us}us ({:.3}ms)",
                us as f64 / 1000.0
            );
        }
    }

    /// rope/norm 끝난 k/v(f32, [kv_dim])를 device buffer 의 pos 슬롯에 f16 으로
    /// device append. host roundtrip 없이 kv_append 커널로 변환+write 한다.
    /// carrier 는 device buffer in 으로 encode-only 버전을 쓰지만, 이 함수는
    /// host f32 slice in 의 per-op/테스트 경로다.
    pub fn append_f32_step(
        &mut self,
        ctx: &MetalContext,
        k_f32: &[f32],
        v_f32: &[f32],
        pos: usize,
    ) {
        assert_eq!(k_f32.len(), self.kv_dim, "k_f32 len != kv_dim");
        assert_eq!(v_f32.len(), self.kv_dim, "v_f32 len != kv_dim");
        assert!(
            pos < self.capacity,
            "pos {pos} >= capacity {}",
            self.capacity
        );
        let shared = MTLResourceOptions::StorageModeShared;
        let mk = |s: &[f32]| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(s.as_ptr() as *mut std::ffi::c_void).expect("ptr null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, std::mem::size_of_val(s), shared)
                    .expect("Metal: failed to create kv_append in buffer")
            }
        };
        let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("ptr null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, std::mem::size_of::<u32>(), shared)
                    .expect("Metal: failed to create kv_append scalar buffer")
            }
        };
        let k_in = mk(k_f32);
        let v_in = mk(v_f32);
        let kvd = self.kv_dim as u32;
        let ps = pos as u32;
        let kvd_buf = mk_u32(&kvd);
        let ps_buf = mk_u32(&ps);

        let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
        let enc = cmd.computeCommandEncoder().expect("Metal: compute encoder");
        enc.setComputePipelineState(&ctx.kv_append_pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&k_in), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&v_in), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&kvd_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&ps_buf), 0, 5);
        }
        let tg_width = ctx.kv_append_pipeline.threadExecutionWidth().max(1);
        let grid = MTLSize {
            width: self.kv_dim.div_ceil(tg_width),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if pos + 1 > self.filled {
            self.filled = pos + 1;
        }
    }
}

/// 테스트/측정용: 빈 KvResident 에 k/v(f32) step 들을 순차 device append 하고
/// 채워진 구간을 readback 한다. kv_append 커널 정확도 검증용.
pub fn kv_append_steps_with_ctx(
    ctx: &MetalContext,
    k_steps: &[Vec<f32>],
    v_steps: &[Vec<f32>],
    kv_dim: usize,
    capacity: usize,
) -> (Vec<u16>, Vec<u16>) {
    assert!(!ctx.kv_int8, "kv_append_steps_with_ctx: f16 전용");
    assert_eq!(k_steps.len(), v_steps.len(), "k/v step count mismatch");
    let mut kv = KvResident::new(ctx, kv_dim, 1, capacity);
    for (pos, (k, v)) in k_steps.iter().zip(v_steps.iter()).enumerate() {
        kv.append_f32_step(ctx, k, v, pos);
    }
    let filled = k_steps.len();
    let kp = kv.k_buf.contents().as_ptr() as *const u16;
    let vp = kv.v_buf.contents().as_ptr() as *const u16;
    let kout = unsafe { std::slice::from_raw_parts(kp, filled * kv_dim) }.to_vec();
    let vout = unsafe { std::slice::from_raw_parts(vp, filled * kv_dim) }.to_vec();
    (kout, vout)
}

/// pm22: per-slot int8 KV append 커널 encode. buffer 0-8 bind + grid=num_kv_heads,
/// tg=SIMD_WIDTH(1 kv_head = 1 SIMD-group). caller 가 command buffer/encoder 소유.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_kv_append_i8(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    k_f32: &ProtocolObject<dyn MTLBuffer>,
    v_f32: &ProtocolObject<dyn MTLBuffer>,
    k_cache: &ProtocolObject<dyn MTLBuffer>,
    v_cache: &ProtocolObject<dyn MTLBuffer>,
    k_scale: &ProtocolObject<dyn MTLBuffer>,
    v_scale: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    nkv_buf: &ProtocolObject<dyn MTLBuffer>,
    pos_buf: &ProtocolObject<dyn MTLBuffer>,
    num_kv_heads: usize,
) {
    enc.setComputePipelineState(&ctx.kv_append_i8_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(k_f32), 0, 0);
        enc.setBuffer_offset_atIndex(Some(v_f32), 0, 1);
        enc.setBuffer_offset_atIndex(Some(k_cache), 0, 2);
        enc.setBuffer_offset_atIndex(Some(v_cache), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_scale), 0, 4);
        enc.setBuffer_offset_atIndex(Some(v_scale), 0, 5);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(nkv_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 8);
    }
    let grid = MTLSize {
        width: num_kv_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// 테스트/측정용: 빈 int8 KV cache 에 k/v(f32) step 들을 순차 device append 하고
/// 채워진 구간(int8 + per-slot scale)을 readback 한다. kv_append_i8 커널 정확도 검증용.
/// int8 ctx(`build_metal_context_with_kv_int8(true)`) 전제.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn kv_append_i8_steps_with_ctx(
    ctx: &MetalContext,
    k_steps: &[Vec<f32>],
    v_steps: &[Vec<f32>],
    head_dim: usize,
    num_kv_heads: usize,
    capacity: usize,
) -> (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>) {
    assert_eq!(k_steps.len(), v_steps.len(), "k/v step count mismatch");
    let kv_dim = num_kv_heads * head_dim;
    let shared = MTLResourceOptions::StorageModeShared;

    // int8 cache(char) + per-slot scale(f32) device buffer.
    let cache_bytes = capacity * kv_dim * std::mem::size_of::<i8>();
    let scale_bytes = capacity * num_kv_heads * std::mem::size_of::<f32>();
    let k_cache = ctx
        .device
        .newBufferWithLength_options(cache_bytes, shared)
        .expect("Metal: failed to create int8 k_cache");
    let v_cache = ctx
        .device
        .newBufferWithLength_options(cache_bytes, shared)
        .expect("Metal: failed to create int8 v_cache");
    let k_scale = ctx
        .device
        .newBufferWithLength_options(scale_bytes, shared)
        .expect("Metal: failed to create k_scale");
    let v_scale = ctx
        .device
        .newBufferWithLength_options(scale_bytes, shared)
        .expect("Metal: failed to create v_scale");

    let mk = |s: &[f32]| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let nn = NonNull::new(s.as_ptr() as *mut std::ffi::c_void).expect("ptr null");
        unsafe {
            ctx.device
                .newBufferWithBytes_length_options(nn, std::mem::size_of_val(s), shared)
                .expect("Metal: failed to create kv_append_i8 in buffer")
        }
    };
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let nn = NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("ptr null");
        unsafe {
            ctx.device
                .newBufferWithBytes_length_options(nn, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create kv_append_i8 scalar buffer")
        }
    };
    let hd = head_dim as u32;
    let nkv = num_kv_heads as u32;
    let hd_buf = mk_u32(&hd);
    let nkv_buf = mk_u32(&nkv);

    for (pos, (k, v)) in k_steps.iter().zip(v_steps.iter()).enumerate() {
        assert!(pos < capacity, "pos {pos} >= capacity {capacity}");
        assert_eq!(k.len(), kv_dim, "k step len != kv_dim");
        assert_eq!(v.len(), kv_dim, "v step len != kv_dim");
        let k_in = mk(k);
        let v_in = mk(v);
        let ps = pos as u32;
        let ps_buf = mk_u32(&ps);

        let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
        let enc = cmd.computeCommandEncoder().expect("Metal: compute encoder");
        encode_kv_append_i8(
            ctx,
            &enc,
            &k_in,
            &v_in,
            &k_cache,
            &v_cache,
            &k_scale,
            &v_scale,
            &hd_buf,
            &nkv_buf,
            &ps_buf,
            num_kv_heads,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let filled = k_steps.len();
    let kp = k_cache.contents().as_ptr() as *const i8;
    let vp = v_cache.contents().as_ptr() as *const i8;
    let ksp = k_scale.contents().as_ptr() as *const f32;
    let vsp = v_scale.contents().as_ptr() as *const f32;
    let ki = unsafe { std::slice::from_raw_parts(kp, filled * kv_dim) }.to_vec();
    let vi = unsafe { std::slice::from_raw_parts(vp, filled * kv_dim) }.to_vec();
    let ks = unsafe { std::slice::from_raw_parts(ksp, filled * num_kv_heads) }.to_vec();
    let vs = unsafe { std::slice::from_raw_parts(vsp, filled * num_kv_heads) }.to_vec();
    (ki, vi, ks, vs)
}

/// resident KV(device buffer) + q 로 attention compute. 전체 KV 업로드 없이 q 만
/// 업로드(작음). k/v 는 `KvResident` 의 device buffer 를 그대로 읽는다.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_decode_resident_dispatch(
    ctx: &MetalContext,
    q: &[f32],
    kv: &KvResident,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(
        !kv.kv_int8,
        "attn_decode_resident_dispatch: int8 KV는 carrier 경로 전용"
    );
    assert_eq!(q.len(), num_heads * head_dim, "q length mismatch");
    assert_eq!(
        num_kv_heads * head_dim,
        kv.kv_dim,
        "kv_dim mismatch with resident"
    );
    assert!(kv_len <= kv.capacity, "kv_len exceeds resident capacity");
    if attn_resident_trace_enabled() {
        eprintln!(
            "[metal-attn-resident] dispatch layer-kv kv_dim={} capacity={} filled={} kv_len={} q_len={} nh={num_heads} nkv={num_kv_heads} hd={head_dim}",
            kv.kv_dim,
            kv.capacity,
            kv.filled,
            kv_len,
            q.len()
        );
    }
    let shared = MTLResourceOptions::StorageModeShared;
    let q_buf = unsafe {
        let nn = NonNull::new(q.as_ptr() as *mut std::ffi::c_void).expect("q ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(nn, std::mem::size_of_val(q), shared)
            .expect("Metal: failed to create q buffer")
    };
    attn_decode_dispatch_buffers(
        ctx,
        &q_buf,
        &kv.k_buf,
        &kv.v_buf,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
    )
}

// ---------------------------------------------------------------------------
// rope_mrope — text M-RoPE in-place (decode 1 token)
// ---------------------------------------------------------------------------
//
// M2 부품: attention layer carrier 안에서 q/k 를 device buffer in-place 회전.
// CPU `rope_mrope_text_inplace` 와 token-identical. 1 thread=1 head.

/// Text M-RoPE in-place(decode 1 token). 측정/테스트용 복사 경로.
pub fn rope_mrope_with_ctx(
    ctx: &MetalContext,
    data: &[f32],
    head_dim: usize,
    dim: usize,
    n_rot: usize,
    theta: f32,
    pos: usize,
) -> Vec<f32> {
    assert!(head_dim > 0 && dim > 0, "head_dim/dim must be > 0");
    assert_eq!(data.len(), dim, "decode rope expects data.len()==dim");
    assert_eq!(dim % head_dim, 0, "dim must be a multiple of head_dim");
    let shared = MTLResourceOptions::StorageModeShared;

    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let d_buf = mk_bytes(data.as_ptr() as *const _, std::mem::size_of_val(data));
    let hd = head_dim as u32;
    let dm = dim as u32;
    let nr = n_rot as u32;
    let ps = pos as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let hd_buf = mk_u32(&hd);
    let dm_buf = mk_u32(&dm);
    let nr_buf = mk_u32(&nr);
    let theta_buf = mk_bytes(&theta as *const f32 as *const _, std::mem::size_of::<f32>());
    let ps_buf = mk_u32(&ps);

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.rope_mrope_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&d_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&hd_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&dm_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&nr_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&theta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&ps_buf), 0, 5);
    }
    let num_heads = dim / head_dim;
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = d_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, dim) };
    out_slice.to_vec()
}

/// gated attention q/gate split(decode). 측정/테스트용 복사 경로.
/// `q_full`=[num_heads*hd*2] (head 별 [query hd | gate hd] 인터리브).
/// 반환 `(query, gate)` 각 [num_heads*hd]. CPU split 과 bit-identical(단순 copy).
pub fn split_qgate_with_ctx(
    ctx: &MetalContext,
    q_full: &[f32],
    num_heads: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q_full.len(), num_heads * hd * 2, "q_full len mismatch");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let out_len = num_heads * hd;
    let qf_buf = mk_bytes(q_full.as_ptr() as *const _, std::mem::size_of_val(q_full));
    let zeros = vec![0u8; out_len * std::mem::size_of::<f32>()];
    let query_buf = mk_bytes(zeros.as_ptr() as *const _, zeros.len());
    let gate_buf = mk_bytes(zeros.as_ptr() as *const _, zeros.len());
    let hd_u = hd as u32;
    let hd_buf = mk_bytes(&hd_u as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_split_qgate(
        ctx, &enc, &qf_buf, &query_buf, &gate_buf, &hd_buf, num_heads, hd,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let q_contents: NonNull<std::ffi::c_void> = query_buf.contents();
    let g_contents: NonNull<std::ffi::c_void> = gate_buf.contents();
    let q_vec =
        unsafe { std::slice::from_raw_parts(q_contents.as_ptr() as *const f32, out_len).to_vec() };
    let g_vec =
        unsafe { std::slice::from_raw_parts(g_contents.as_ptr() as *const f32, out_len).to_vec() };
    (q_vec, g_vec)
}

/// Partial RoPE in-place(인접페어, decode 1 token). 측정/테스트용 복사 경로.
/// theta_scale 은 host f32 로 계산(theta.powf(-2/n_rot), clamp 후) — host
/// `rope_partial_inplace`(rope.rs:400-401)와 동일 식·타입. 반환=[dim].
pub fn rope_partial_with_ctx(
    ctx: &MetalContext,
    data: &[f32],
    head_dim: usize,
    dim: usize,
    n_rot: usize,
    theta: f32,
    pos: usize,
) -> Vec<f32> {
    assert!(head_dim > 0 && dim > 0, "head_dim/dim must be > 0");
    assert_eq!(data.len(), dim, "rope_partial expects data.len()==dim");
    assert_eq!(dim % head_dim, 0, "dim must be a multiple of head_dim");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let nr = n_rot.min(head_dim);
    let theta_scale: f32 = theta.powf(-2.0_f32 / nr as f32);
    let d_buf = mk_bytes(data.as_ptr() as *const _, std::mem::size_of_val(data));
    let hd = head_dim as u32;
    let dm = dim as u32;
    let nrv = n_rot as u32;
    let ps = pos as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let hd_buf = mk_u32(&hd);
    let dm_buf = mk_u32(&dm);
    let nr_buf = mk_u32(&nrv);
    let ts_buf = mk_bytes(
        &theta_scale as *const f32 as *const _,
        std::mem::size_of::<f32>(),
    );
    let ps_buf = mk_u32(&ps);

    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    let num_heads = dim / head_dim;
    encode_rope_partial(
        ctx, &enc, &d_buf, &hd_buf, &dm_buf, &nr_buf, &ts_buf, &ps_buf, num_heads,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = d_buf.contents();
    unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, dim).to_vec() }
}

/// gated attention `attn_out *= sigmoid(gate)`(elementwise). 측정/테스트용 복사 경로.
/// host `sigmoid_inplace`+`mul_inplace`(decode.rs:751-757)와 일치. 반환=[n].
pub fn gate_apply_with_ctx(ctx: &MetalContext, attn: &[f32], gate: &[f32]) -> Vec<f32> {
    assert_eq!(attn.len(), gate.len(), "attn/gate len mismatch");
    let n = attn.len();
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let a_buf = mk_bytes(attn.as_ptr() as *const _, std::mem::size_of_val(attn));
    let g_buf = mk_bytes(gate.as_ptr() as *const _, std::mem::size_of_val(gate));
    let nv = n as u32;
    let n_buf = mk_bytes(&nv as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    encode_gate_apply(ctx, &enc, &a_buf, &g_buf, &n_buf, n);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = a_buf.contents();
    unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n).to_vec() }
}

/// Per-head q/k RMSNorm(decode 1 token). 측정/테스트용 복사 경로.
/// `data`=[num_heads*head_dim], `weight`=[head_dim] (head 공유). head 별로
/// head_dim 슬라이스를 각각 RMSNorm 한다. CPU `rms_norm_into` 와 token-identical.
pub fn qk_norm_with_ctx(
    ctx: &MetalContext,
    data: &[f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert!(
        num_heads > 0 && head_dim > 0,
        "num_heads/head_dim must be > 0"
    );
    assert_eq!(
        data.len(),
        num_heads * head_dim,
        "qk_norm expects data.len()==num_heads*head_dim"
    );
    assert_eq!(
        weight.len(),
        head_dim,
        "qk_norm expects weight.len()==head_dim"
    );
    assert!(
        head_dim <= 256,
        "qk_norm tree reduction caps head_dim at 256"
    );
    let shared = MTLResourceOptions::StorageModeShared;

    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let in_buf = mk_bytes(data.as_ptr() as *const _, std::mem::size_of_val(data));
    let w_buf = mk_bytes(weight.as_ptr() as *const _, std::mem::size_of_val(weight));
    let out_len = num_heads * head_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: failed to create out buffer");
    let hd = head_dim as u32;
    let hd_buf = mk_bytes(&hd as *const u32 as *const _, std::mem::size_of::<u32>());
    let eps_buf = mk_bytes(&eps as *const f32 as *const _, std::mem::size_of::<f32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.qk_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&hd_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&eps_buf), 0, 4);
    }
    // grid = head 1개/threadgroup, tg = 256(tree reduction, head_dim<=256 grid-stride).
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, out_len) };
    out_slice.to_vec()
}

/// pm48 oracle: prefill(seq_len>1) fused qk_norm→rope device 커널 검증/측정용.
/// `data`=[seq_len*num_heads*head_dim] (token-major, per-token head 연속), `weight`=[head_dim]
/// (head 공유 norm weight). 반환 `(out, gpu_ms)` — out 은 norm 후 text M-RoPE 적용된
/// 같은 layout. CPU ground-truth 순서(per-head `rms_norm_into` → `rope_mrope_text_inplace`)와
/// 1:1. q/k 둘 다 이 함수를 각각 호출(GQA: q=num_heads, k=num_kv_heads).
/// pipeline 은 on-demand build (test-only, production ctx 캐시 무영향).
#[cfg(test)]
pub fn prefill_rope_qk_norm_oracle_with_ctx(
    ctx: &MetalContext,
    data: &[f32],
    weight: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    eps: f32,
    pos_start: usize,
) -> (Vec<f32>, f64) {
    assert!(
        seq_len > 0 && num_heads > 0 && head_dim > 0,
        "seq_len/num_heads/head_dim must be > 0"
    );
    assert_eq!(
        data.len(),
        seq_len * num_heads * head_dim,
        "expects data.len()==seq_len*num_heads*head_dim"
    );
    assert_eq!(weight.len(), head_dim, "expects weight.len()==head_dim");
    assert!(head_dim <= 256, "tree reduction caps head_dim at 256");
    assert!(n_rot <= head_dim, "n_rot must be <= head_dim");

    let pipeline = ctx.prefill_rope_qk_norm_pipeline.clone();
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let in_buf = mk_bytes(data.as_ptr() as *const _, std::mem::size_of_val(data));
    let w_buf = mk_bytes(weight.as_ptr() as *const _, std::mem::size_of_val(weight));
    let total = seq_len * num_heads * head_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(total * 4, shared)
        .expect("Metal: failed to create out buffer");
    let nh = num_heads as u32;
    let hd = head_dim as u32;
    let nr = n_rot as u32;
    let ps = pos_start as u32;
    let mk_u32 = |v: &u32| mk_bytes(v as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_u32(&nh);
    let hd_buf = mk_u32(&hd);
    let nr_buf = mk_u32(&nr);
    let theta_buf = mk_bytes(&theta as *const f32 as *const _, std::mem::size_of::<f32>());
    let eps_buf = mk_bytes(&eps as *const f32 as *const _, std::mem::size_of::<f32>());
    let ps_buf = mk_u32(&ps);

    let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc = cmd.computeCommandEncoder().expect("Metal: compute encoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&nh_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&hd_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&nr_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&theta_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&eps_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&ps_buf), 0, 8);
    }
    // grid = seq_len*num_heads threadgroups (1 (token,head) = 1 threadgroup).
    // tg = 256 (qk_norm tree reduction 정확 + head_dim<=256 grid-stride).
    let grid = MTLSize {
        width: seq_len * num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;

    let contents: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, total) };
    (out_slice.to_vec(), gpu_ms)
}

/// pm48 ②: prefill qk_norm→rope fused 를 device buffer in/out 으로 encoder 에 encode(chain 부품).
/// `in_buf`=[seq_len*num_heads*head_dim] device, `weight_buf`=[head_dim] device,
/// `out_buf`=[seq_len*num_heads*head_dim] device. scalar buffer 는 carrier 가 1회 alloc.
/// grid = seq_len*num_heads threadgroups, tg=256(tree reduction). oracle 4.95e-5 검증 커널.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_prefill_rope_qk_norm(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    hd_buf: &ProtocolObject<dyn MTLBuffer>,
    nrot_buf: &ProtocolObject<dyn MTLBuffer>,
    theta_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
    pos_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.prefill_rope_qk_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(hd_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(nrot_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(theta_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 8);
    }
    let grid = MTLSize {
        width: seq_len * num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_prefill_rope_only(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rope_cos_sin: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_dim: usize,
    n_rot: usize,
    seq_len: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if seq_len == 0
        || num_heads == 0
        || head_dim == 0
        || head_dim > 256
        || n_rot > head_dim
        || n_rot % 2 != 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let groups = seq_len
        .checked_mul(num_heads)
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let rope_table_bytes = seq_len
        .checked_mul(n_rot)
        .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)?;
    if rope_cos_sin.length() < rope_table_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let num_heads =
        u32::try_from(num_heads).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let head_dim =
        u32::try_from(head_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_rot = u32::try_from(n_rot).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;

    enc.setComputePipelineState(&ctx.prefill_rope_only_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
        enc.setBuffer_offset_atIndex(Some(rope_cos_sin), 0, 2);
    }
    set_u32_bytes(enc, num_heads, 3);
    set_u32_bytes(enc, head_dim, 4);
    set_u32_bytes(enc, n_rot, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    chain_barrier(ctx, enc);
    Ok(())
}

/// Decode(seq_len=1) depthwise causal conv1d + SiLU. `input`/`weight`=[kernel_size*channels],
/// 반환=[channels]. CPU `ssm_conv1d_silu_into`(seq_len=1) 와 token-identical.
/// GDN layer carrier 의 conv 부품(host readback 제거 → device chain 연속).
pub fn ssm_conv1d_silu_with_ctx(
    ctx: &MetalContext,
    input: &[f32],
    weight: &[f32],
    channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    assert!(
        channels > 0 && kernel_size > 0,
        "channels/kernel_size must be > 0"
    );
    assert_eq!(
        input.len(),
        kernel_size * channels,
        "ssm_conv1d_silu expects input.len()==kernel_size*channels"
    );
    assert_eq!(
        weight.len(),
        kernel_size * channels,
        "ssm_conv1d_silu expects weight.len()==kernel_size*channels"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let in_buf = mk_bytes(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let w_buf = mk_bytes(weight.as_ptr() as *const _, std::mem::size_of_val(weight));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(channels * 4, shared)
        .expect("Metal: failed to create out buffer");
    let ch = channels as u32;
    let ks = kernel_size as u32;
    let ch_buf = mk_bytes(&ch as *const u32 as *const _, std::mem::size_of::<u32>());
    let ks_buf = mk_bytes(&ks as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.ssm_conv_silu_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&ch_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&ks_buf), 0, 4);
    }
    // thread 1개 = channel 1개 (thread_position_in_grid). grid = ceil(channels/tg_width).
    let tg_width = ctx.ssm_conv_silu_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: channels.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, channels) };
    out_slice.to_vec()
}

/// pm43: prefill batch(seq_len>1) conv1d+silu. input[(seq_len+kernel_size-1)*channels] f32 +
/// weight[kernel_size*channels] f32 → out[seq_len*channels] f32. CPU `ssm_conv1d_silu_into`(seq_len)
/// 와 1:1(depthwise causal). grid=seq_len*channels. GDN conv1d 부품의 prefill batch 경로.
pub fn ssm_conv1d_silu_batch_with_ctx(
    ctx: &MetalContext,
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let total_len = seq_len + kernel_size - 1;
    assert_eq!(
        input.len(),
        total_len * channels,
        "conv input must be (seq+ks-1)*channels"
    );
    assert_eq!(
        weight.len(),
        kernel_size * channels,
        "conv weight must be ks*channels"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let in_buf = mk(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let w_buf = mk(weight.as_ptr() as *const _, std::mem::size_of_val(weight));
    let out_len = seq_len * channels;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: out buffer");
    let ch = channels as u32;
    let ks = kernel_size as u32;
    let sl = seq_len as u32;
    let ch_buf = mk(&ch as *const u32 as *const _, 4);
    let ks_buf = mk(&ks as *const u32 as *const _, 4);
    let sl_buf = mk(&sl as *const u32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc = cmd.computeCommandEncoder().expect("Metal: compute encoder");
    encode_ssm_conv1d_silu_batch(
        ctx, &enc, &in_buf, &w_buf, &out_buf, &ch_buf, &ks_buf, &sl_buf, out_len,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = out_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, out_len) }.to_vec()
}

/// pm45 M2: prefill batch conv1d+silu 를 encoder 에 encode(commit/readback 안 함).
/// in/weight/out + ch/ks/sl scalar device buffer. `out_len` = seq_len*channels (grid 산정용).
/// `ssm_conv1d_silu_batch_with_ctx` 의 encode 부분 추출 — conv→delta chain device-resident 연결용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ssm_conv1d_silu_batch(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    ch_buf: &ProtocolObject<dyn MTLBuffer>,
    ks_buf: &ProtocolObject<dyn MTLBuffer>,
    sl_buf: &ProtocolObject<dyn MTLBuffer>,
    out_len: usize,
) {
    enc.setComputePipelineState(&ctx.ssm_conv_silu_batch_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ch_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(ks_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(sl_buf), 0, 5);
    }
    let tg_width = ctx
        .ssm_conv_silu_batch_pipeline
        .threadExecutionWidth()
        .max(1);
    let grid = MTLSize {
        width: out_len.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm44 M1: gated RMSNorm+SiLU 를 device buffer in/out 으로 encode(commit/readback 안 함).
/// out_in_buf[rows*cols] + z_buf + w_buf[cols] → gated_buf[rows*cols]. chain device-resident 연결용.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gated_rmsnorm_silu(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    out_in_buf: &ProtocolObject<dyn MTLBuffer>,
    z_buf: &ProtocolObject<dyn MTLBuffer>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    gated_buf: &ProtocolObject<dyn MTLBuffer>,
    cols_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) {
    enc.setComputePipelineState(&ctx.gated_rmsnorm_silu_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out_in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(z_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gated_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(cols_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 5);
    }
    let grid = MTLSize {
        width: rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// pm43: GDN prefill gated RMSNorm + SiLU(batch). out_in[rows*cols](scan 출력) + z[rows*cols](gate) +
/// weight[cols](ssm_norm) → gated[rows*cols] = rmsnorm(out_in per row)·silu(z). CPU gdn_forward
/// gated_norm+silu 경로와 1:1. grid=rows(threadgroup 1개=row), tg=256.
pub fn gated_rmsnorm_silu_batch_with_ctx(
    ctx: &MetalContext,
    out_in: &[f32],
    z: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(out_in.len(), rows * cols, "gated out_in must be rows*cols");
    assert_eq!(z.len(), rows * cols, "gated z must be rows*cols");
    assert_eq!(weight.len(), cols, "gated weight(ssm_norm) must be cols");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk =
        |data: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            unsafe {
                let ptr = NonNull::new(data as *mut std::ffi::c_void).expect("ptr null");
                ctx.device
                    .newBufferWithBytes_length_options(ptr, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let out_buf = mk(out_in.as_ptr() as *const _, std::mem::size_of_val(out_in));
    let z_buf = mk(z.as_ptr() as *const _, std::mem::size_of_val(z));
    let w_buf = mk(weight.as_ptr() as *const _, std::mem::size_of_val(weight));
    let n = rows * cols;
    let gated_buf = ctx
        .device
        .newBufferWithLength_options(n * 4, shared)
        .expect("Metal: gated buffer");
    let cols_u = cols as u32;
    let cols_buf = mk(&cols_u as *const u32 as *const _, 4);
    let eps_buf = mk(&eps as *const f32 as *const _, 4);
    let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc = cmd.computeCommandEncoder().expect("Metal: compute encoder");
    encode_gated_rmsnorm_silu(
        ctx, &enc, &out_buf, &z_buf, &w_buf, &gated_buf, &cols_buf, &eps_buf, rows,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c: NonNull<std::ffi::c_void> = gated_buf.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, n) }.to_vec()
}

/// GDN delta_net recurrent scan 1-step(decode, seq_len=1). state in-place 갱신.
/// 반환: `(output[num_heads*head_v_dim], state_after[num_heads*head_v_dim*head_k_dim])`.
/// CPU `delta_net_scan_into`(seq_len=1) 와 token-identical. GDN layer carrier 부품.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_step_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    num_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        num_heads > 0 && num_k_heads > 0 && head_k_dim > 0 && head_v_dim > 0,
        "dims must be > 0"
    );
    assert!(
        num_heads % num_k_heads == 0,
        "num_heads must be multiple of num_k_heads (GQA)"
    );
    assert!(
        head_k_dim <= 256,
        "delta_net_step caps head_k_dim at 256 (shared)"
    );
    assert_eq!(q.len(), num_k_heads * head_k_dim, "q len");
    assert_eq!(k.len(), num_k_heads * head_k_dim, "k len");
    assert_eq!(v.len(), num_heads * head_v_dim, "v len");
    assert_eq!(gate.len(), num_heads, "gate len");
    assert_eq!(beta.len(), num_heads, "beta len");
    assert_eq!(
        state.len(),
        num_heads * head_v_dim * head_k_dim,
        "state len"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k.as_ptr() as *const _, std::mem::size_of_val(k));
    let v_buf = mk_bytes(v.as_ptr() as *const _, std::mem::size_of_val(v));
    let gate_buf = mk_bytes(gate.as_ptr() as *const _, std::mem::size_of_val(gate));
    let beta_buf = mk_bytes(beta.as_ptr() as *const _, std::mem::size_of_val(beta));
    // state 는 in-place 갱신용 mutable copy.
    let state_buf = mk_bytes(state.as_ptr() as *const _, std::mem::size_of_val(state));
    let out_len = num_heads * head_v_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: failed to create out buffer");
    let hk = head_k_dim as u32;
    let hv = head_v_dim as u32;
    let nk = num_k_heads as u32;
    let hk_buf = mk_bytes(&hk as *const u32 as *const _, std::mem::size_of::<u32>());
    let hv_buf = mk_bytes(&hv as *const u32 as *const _, std::mem::size_of::<u32>());
    let nk_buf = mk_bytes(&nk as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.delta_net_step_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&hk_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&hv_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&nk_buf), 0, 9);
    }
    // threadgroup 1개 = head 1개(grid=num_heads), thread vi = state row(tg=head_v_dim).
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: head_v_dim,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let out_c: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(out_c.as_ptr() as *const f32, out_len) };
    let state_c: NonNull<std::ffi::c_void> = state_buf.contents();
    let state_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(state_c.as_ptr() as *const f32, state.len()) };
    (out_slice.to_vec(), state_slice.to_vec())
}

/// pm39 M2: GDN delta_net chunkwise parallel scan(prefill, seq_len>1).
/// CPU `delta_net_scan_chunkwise`(M1 oracle) 와 token-identical(f32, 같은 reduction 순서).
/// `state` 입력은 chunk-loop hand-off 로 in-place 갱신되어 `state_after` 로 반환.
/// 반환=`(output[seq*num_heads*head_v_dim], state_after)`.
/// GQA 는 caller(prefill repeat_qk_for_value_heads)가 이미 q/k 를 num_heads 로 repeat 푼 뒤 넘긴다.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_scan_chunk_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        seq_len > 0 && num_heads > 0 && head_k_dim > 0 && head_v_dim > 0 && chunk_size > 0,
        "dims must be > 0"
    );
    // g_cum 정적 [256] — chunk_size 상한. (32KB threadgroup 한계로 실제 chunk_size 는 head dim 에
    //  비례해 더 작아짐 — 그 비례 정책은 seam 연결 단계에서. PoC 검증은 이 cap 안.)
    assert!(
        chunk_size <= 256,
        "delta_net_scan_chunk caps chunk_size at 256 (g_cum static)"
    );
    // threadgroup dynamic memory 총량: u_corr(cs*hv) + kk_sh(cs*cs) + qk_sh(cs*cs), f32.
    // M5 threadgroup 한계 32KiB(정적 g_cum[256]=1KiB 별도). 초과하면 chunk_size 를 줄여야 한다.
    let tg_dynamic = (chunk_size * head_v_dim + 2 * chunk_size * chunk_size) * 4;
    assert!(
        tg_dynamic + 256 * 4 <= 32 * 1024,
        "delta_net_scan_chunk threadgroup mem {} + g_cum 1KiB > 32KiB (chunk_size={chunk_size} \
         head_v_dim={head_v_dim} 줄일 것)",
        tg_dynamic
    );
    assert_eq!(q.len(), seq_len * num_heads * head_k_dim, "q len");
    assert_eq!(k.len(), seq_len * num_heads * head_k_dim, "k len");
    assert_eq!(v.len(), seq_len * num_heads * head_v_dim, "v len");
    assert_eq!(gate.len(), seq_len * num_heads, "gate len");
    assert_eq!(beta.len(), seq_len * num_heads, "beta len");
    assert_eq!(
        state.len(),
        num_heads * head_v_dim * head_k_dim,
        "state len"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k.as_ptr() as *const _, std::mem::size_of_val(k));
    let v_buf = mk_bytes(v.as_ptr() as *const _, std::mem::size_of_val(v));
    let gate_buf = mk_bytes(gate.as_ptr() as *const _, std::mem::size_of_val(gate));
    let beta_buf = mk_bytes(beta.as_ptr() as *const _, std::mem::size_of_val(beta));
    // state 는 in-place hand-off 용 mutable copy.
    let state_buf = mk_bytes(state.as_ptr() as *const _, std::mem::size_of_val(state));
    let out_len = seq_len * num_heads * head_v_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: failed to create out buffer");
    let sl = seq_len as u32;
    let hk = head_k_dim as u32;
    let hv = head_v_dim as u32;
    let cs = chunk_size as u32;
    let nh = num_heads as u32;
    let sl_buf = mk_bytes(&sl as *const u32 as *const _, std::mem::size_of::<u32>());
    let hk_buf = mk_bytes(&hk as *const u32 as *const _, std::mem::size_of::<u32>());
    let hv_buf = mk_bytes(&hv as *const u32 as *const _, std::mem::size_of::<u32>());
    let cs_buf = mk_bytes(&cs as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_bytes(&nh as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_delta_net_scan_chunk(
        ctx, &enc, &q_buf, &k_buf, &v_buf, &gate_buf, &beta_buf, &state_buf, &out_buf, &sl_buf,
        &hk_buf, &hv_buf, &cs_buf, &nh_buf, num_heads, head_v_dim, chunk_size,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // pm42 M0(measure): delta_scan 순수 GPU-time 계측 — wall(RNB_PROFILE) 대비 host 왕복(delta
    // state readback/commit-wait) 분리용. RNB_METAL_PREFILL_DELTA_SCAN_TIME=1. layer당 1줄 + 누적.
    if std::env::var("RNB_METAL_PREFILL_DELTA_SCAN_TIME").as_deref() == Ok("1") {
        use std::cell::Cell;
        thread_local! {
            static DSCAN_GPU: Cell<(usize, f64)> = const { Cell::new((0, 0.0)) };
        }
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
        DSCAN_GPU.with(|c| {
            let (cnt, sum) = c.get();
            c.set((cnt + 1, sum + gpu_ms));
            eprintln!(
                "[prefill-delta-scan] call#{} gpu={gpu_ms:.3}ms cumulative={:.1}ms (cs={chunk_size} hv={head_v_dim})",
                cnt + 1,
                sum + gpu_ms
            );
        });
    }

    let out_c: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(out_c.as_ptr() as *const f32, out_len) };
    let state_c: NonNull<std::ffi::c_void> = state_buf.contents();
    let state_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(state_c.as_ptr() as *const f32, state.len()) };
    (out_slice.to_vec(), state_slice.to_vec())
}

/// pm58: GDN delta_net autoregressive prefill scan. Same sequential recurrence as
/// `delta_net_scan`, but one Metal kernel keeps each state row in registers across `seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_scan_ar_with_ctx(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        seq_len > 0 && num_heads > 0 && head_k_dim > 0 && head_v_dim > 0,
        "dims must be > 0"
    );
    assert_eq!(
        head_k_dim % SIMD_WIDTH,
        0,
        "delta_net_scan_ar expects head_k_dim multiple of SIMD_WIDTH"
    );
    assert!(
        matches!(head_k_dim / SIMD_WIDTH, 1 | 2 | 4 | 8),
        "delta_net_scan_ar supports head_k_dim 32/64/128/256"
    );
    assert_eq!(q.len(), seq_len * num_heads * head_k_dim, "q len");
    assert_eq!(k.len(), seq_len * num_heads * head_k_dim, "k len");
    assert_eq!(v.len(), seq_len * num_heads * head_v_dim, "v len");
    assert_eq!(gate.len(), seq_len * num_heads, "gate len");
    assert_eq!(beta.len(), seq_len * num_heads, "beta len");
    assert_eq!(
        state.len(),
        num_heads * head_v_dim * head_k_dim,
        "state len"
    );

    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k.as_ptr() as *const _, std::mem::size_of_val(k));
    let v_buf = mk_bytes(v.as_ptr() as *const _, std::mem::size_of_val(v));
    let gate_buf = mk_bytes(gate.as_ptr() as *const _, std::mem::size_of_val(gate));
    let beta_buf = mk_bytes(beta.as_ptr() as *const _, std::mem::size_of_val(beta));
    let state_buf = mk_bytes(state.as_ptr() as *const _, std::mem::size_of_val(state));
    let out_len = seq_len * num_heads * head_v_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: failed to create out buffer");

    let sl = seq_len as u32;
    let hk = head_k_dim as u32;
    let hv = head_v_dim as u32;
    let nh = num_heads as u32;
    let sl_buf = mk_bytes(&sl as *const u32 as *const _, std::mem::size_of::<u32>());
    let hk_buf = mk_bytes(&hk as *const u32 as *const _, std::mem::size_of::<u32>());
    let hv_buf = mk_bytes(&hv as *const u32 as *const _, std::mem::size_of::<u32>());
    let nh_buf = mk_bytes(&nh as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    encode_delta_net_scan_ar(
        ctx, &enc, &q_buf, &k_buf, &v_buf, &gate_buf, &beta_buf, &state_buf, &out_buf, &sl_buf,
        &hk_buf, &hv_buf, &nh_buf, num_heads, head_k_dim, head_v_dim,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let out_c: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(out_c.as_ptr() as *const f32, out_len) };
    let state_c: NonNull<std::ffi::c_void> = state_buf.contents();
    let state_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(state_c.as_ptr() as *const f32, state.len()) };
    (out_slice.to_vec(), state_slice.to_vec())
}

/// pm47 ② STEP4+STEP5 GEMM full delta scan (oracle 검증용 host wrapper). su/kh/step5_temp(STEP5)
/// + a_cat/b_cat/step4_temp(STEP4) device workspace(buffer 12~17). 반환 (out, state).
pub fn delta_net_scan_chunk_step45gemm_full(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(head_v_dim, 128, "step45gemm: hv=128 고정");
    assert_eq!(head_k_dim, 128, "step45gemm: hk=128 고정");
    assert!(chunk_size <= 48, "step45gemm: cs<=48");
    assert_eq!(q.len(), seq_len * num_heads * head_k_dim, "q len");
    assert_eq!(k.len(), seq_len * num_heads * head_k_dim, "k len");
    assert_eq!(v.len(), seq_len * num_heads * head_v_dim, "v len");
    assert_eq!(gate.len(), seq_len * num_heads, "gate len");
    assert_eq!(beta.len(), seq_len * num_heads, "beta len");
    assert_eq!(
        state.len(),
        num_heads * head_v_dim * head_k_dim,
        "state len"
    );
    let pipeline = build_pipeline_v4(
        &ctx.device,
        DELTA_NET_SCAN_CHUNK_STEP45GEMM_SRC,
        "delta_net_scan_chunk_step45gemm",
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("ptr null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: buffer")
            }
        };
    let q_buf = mk_bytes(q.as_ptr() as *const _, std::mem::size_of_val(q));
    let k_buf = mk_bytes(k.as_ptr() as *const _, std::mem::size_of_val(k));
    let v_buf = mk_bytes(v.as_ptr() as *const _, std::mem::size_of_val(v));
    let gate_buf = mk_bytes(gate.as_ptr() as *const _, std::mem::size_of_val(gate));
    let beta_buf = mk_bytes(beta.as_ptr() as *const _, std::mem::size_of_val(beta));
    let state_buf = mk_bytes(state.as_ptr() as *const _, std::mem::size_of_val(state));
    let out_len = seq_len * num_heads * head_v_dim;
    let out_buf = ctx
        .device
        .newBufferWithLength_options(out_len * 4, shared)
        .expect("Metal: out buf");
    // STEP5 device workspace
    let kpad = 48usize;
    let su_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * head_v_dim * kpad * 2, shared)
        .expect("Metal: su_half");
    let kh_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * kpad * head_k_dim * 2, shared)
        .expect("Metal: kh_half");
    let temp_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * head_v_dim * head_k_dim * 4, shared)
        .expect("Metal: temp");
    // STEP4 device workspace (a_cat[CPAD×KPAD] half, b_cat[KPAD×hv] half, step4_temp[CPAD×hv] f32)
    let s4_cpad = 48usize;
    let s4_kpad = 176usize;
    let a_cat_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * s4_cpad * s4_kpad * 2, shared)
        .expect("Metal: a_cat");
    let b_cat_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * s4_kpad * head_v_dim * 2, shared)
        .expect("Metal: b_cat");
    let step4_temp_buf = ctx
        .device
        .newBufferWithLength_options(num_heads * s4_cpad * head_v_dim * 4, shared)
        .expect("Metal: step4_temp");
    let sl = seq_len as u32;
    let hk = head_k_dim as u32;
    let hv = head_v_dim as u32;
    let cs = chunk_size as u32;
    let nh = num_heads as u32;
    let sl_buf = mk_bytes(&sl as *const u32 as *const _, 4);
    let hk_buf = mk_bytes(&hk as *const u32 as *const _, 4);
    let hv_buf = mk_bytes(&hv as *const u32 as *const _, 4);
    let cs_buf = mk_bytes(&cs as *const u32 as *const _, 4);
    let nh_buf = mk_bytes(&nh as *const u32 as *const _, 4);

    let cmd = ctx.queue.commandBuffer().expect("Metal: cmd");
    let enc = cmd.computeCommandEncoder().expect("Metal: enc");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&*beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&*state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&*out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&*sl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&*hk_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(&*hv_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(&*cs_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(&*nh_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(&*su_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(&*kh_buf), 0, 13);
        enc.setBuffer_offset_atIndex(Some(&*temp_buf), 0, 14);
        enc.setBuffer_offset_atIndex(Some(&*a_cat_buf), 0, 15);
        enc.setBuffer_offset_atIndex(Some(&*b_cat_buf), 0, 16);
        enc.setBuffer_offset_atIndex(Some(&*step4_temp_buf), 0, 17);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * head_v_dim * 4, 0);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 1);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 2);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: head_v_dim,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let out_c: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(out_c.as_ptr() as *const f32, out_len) };
    let state_c: NonNull<std::ffi::c_void> = state_buf.contents();
    let state_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(state_c.as_ptr() as *const f32, state.len()) };
    (out_slice.to_vec(), state_slice.to_vec())
}

/// pm47 ② STEP4+STEP5 GEMM encode-only(full chain device-resident). encode_step5gemm + STEP4
/// device workspace(a_cat/b_cat/step4_temp, buffer 15/16/17). pipeline = step45gemm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_delta_net_scan_chunk_step45gemm(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_buf: &ProtocolObject<dyn MTLBuffer>,
    state_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    sl_buf: &ProtocolObject<dyn MTLBuffer>,
    hk_buf: &ProtocolObject<dyn MTLBuffer>,
    hv_buf: &ProtocolObject<dyn MTLBuffer>,
    cs_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    su_buf: &ProtocolObject<dyn MTLBuffer>,
    kh_buf: &ProtocolObject<dyn MTLBuffer>,
    temp_buf: &ProtocolObject<dyn MTLBuffer>,
    a_cat_buf: &ProtocolObject<dyn MTLBuffer>,
    b_cat_buf: &ProtocolObject<dyn MTLBuffer>,
    step4_temp_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_v_dim: usize,
    chunk_size: usize,
) {
    let pipeline = ctx
        .delta_net_scan_chunk_step45gemm_pipeline
        .as_ref()
        .expect("delta_net_scan_chunk_step45gemm pipeline missing (capability=false?)");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(sl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(hk_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(hv_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(cs_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(su_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(kh_buf), 0, 13);
        enc.setBuffer_offset_atIndex(Some(temp_buf), 0, 14);
        enc.setBuffer_offset_atIndex(Some(a_cat_buf), 0, 15);
        enc.setBuffer_offset_atIndex(Some(b_cat_buf), 0, 16);
        enc.setBuffer_offset_atIndex(Some(step4_temp_buf), 0, 17);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * head_v_dim * 4, 0);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 1);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 2);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: head_v_dim,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_delta_net_scan_ar(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_buf: &ProtocolObject<dyn MTLBuffer>,
    state_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    sl_buf: &ProtocolObject<dyn MTLBuffer>,
    hk_buf: &ProtocolObject<dyn MTLBuffer>,
    hv_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    assert_eq!(
        head_k_dim % SIMD_WIDTH,
        0,
        "delta_net_scan_ar expects head_k_dim multiple of SIMD_WIDTH"
    );
    let nsg = head_k_dim / SIMD_WIDTH;
    let pipeline = match nsg {
        1 => &ctx.delta_net_scan_ar1_pipeline,
        2 => &ctx.delta_net_scan_ar2_pipeline,
        4 => &ctx.delta_net_scan_ar4_pipeline,
        8 => &ctx.delta_net_scan_ar8_pipeline,
        _ => panic!("delta_net_scan_ar supports head_k_dim 32/64/128/256"),
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(sl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(hk_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(hv_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 10);
    }
    let grid = MTLSize {
        width: (head_v_dim + nsg - 1) / nsg,
        height: num_heads,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: nsg,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_delta_net_scan_chunk(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_buf: &ProtocolObject<dyn MTLBuffer>,
    state_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    sl_buf: &ProtocolObject<dyn MTLBuffer>,
    hk_buf: &ProtocolObject<dyn MTLBuffer>,
    hv_buf: &ProtocolObject<dyn MTLBuffer>,
    cs_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_v_dim: usize,
    chunk_size: usize,
) {
    enc.setComputePipelineState(&ctx.delta_net_scan_chunk_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(sl_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(hk_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(hv_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(cs_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 11);
        // dynamic threadgroup memory: u_corr[cs*hv] / kk_sh[cs*cs] / qk_sh[cs*cs] f32.
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * head_v_dim * 4, 0);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 1);
        enc.setThreadgroupMemoryLength_atIndex(chunk_size * chunk_size * 4, 2);
    }
    // threadgroup 1개 = head 1개(grid=num_heads), thread vi = state row(tg=head_v_dim).
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: head_v_dim,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_prefill_l2_norm_exact(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    eps: f32,
    scale: f32,
) -> Result<(), QwenMoeLlamaIdError> {
    if rows == 0 || cols == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let cols = u32::try_from(cols).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    enc.setComputePipelineState(&ctx.qwen_prefill_l2_norm_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    set_u32_bytes(enc, cols, 2);
    set_f32_bytes(enc, eps, 3);
    set_f32_bytes(enc, scale, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    chain_barrier(ctx, enc);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_prefill_gated_rmsnorm_silu_exact(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    z: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), QwenMoeLlamaIdError> {
    if rows == 0 || cols == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let cols = u32::try_from(cols).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    enc.setComputePipelineState(&ctx.qwen_prefill_gated_rmsnorm_silu_chain_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(z), 0, 1);
        enc.setBuffer_offset_atIndex(Some(weight), 0, 2);
        enc.setBuffer_offset_atIndex(Some(output), 0, 3);
    }
    set_u32_bytes(enc, cols, 4);
    set_f32_bytes(enc, eps, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    chain_barrier(ctx, enc);
    Ok(())
}

/// Per-row L2 normalize: `input`=[n_rows*dim], 반환=[n_rows*dim].
/// CPU `l2_norm_into` 와 token-identical. GDN q/k head 별 normalize 부품.
pub fn l2_norm_with_ctx(
    ctx: &MetalContext,
    input: &[f32],
    n_rows: usize,
    dim: usize,
    eps: f32,
    scale: f32,
) -> Vec<f32> {
    assert!(n_rows > 0 && dim > 0, "n_rows/dim must be > 0");
    assert_eq!(
        input.len(),
        n_rows * dim,
        "l2_norm expects input.len()==n_rows*dim"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let in_buf = mk_bytes(input.as_ptr() as *const _, std::mem::size_of_val(input));
    let out_buf = ctx
        .device
        .newBufferWithLength_options(input.len() * 4, shared)
        .expect("Metal: failed to create out buffer");
    let d = dim as u32;
    let d_buf = mk_bytes(&d as *const u32 as *const _, std::mem::size_of::<u32>());
    let eps_buf = mk_bytes(&eps as *const f32 as *const _, std::mem::size_of::<f32>());
    let scale_buf = mk_bytes(&scale as *const f32 as *const _, std::mem::size_of::<f32>());

    let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc = cmd.computeCommandEncoder().expect("Metal: encoder");
    enc.setComputePipelineState(&ctx.l2_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&d_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&eps_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&scale_buf), 0, 4);
    }
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = out_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, input.len()) };
    out_slice.to_vec()
}

/// GDN alpha/beta 전처리(head 별, in-place): `beta=sigmoid(beta)`,
/// `alpha=softplus(alpha+dt_bias)*ssm_a`. 반환 `(alpha_after, beta_after)`.
/// CPU `decode_gdn_layer_qwen` 의 beta sigmoid + alpha-gate 와 token-identical.
pub fn gdn_alpha_beta_with_ctx(
    ctx: &MetalContext,
    alpha: &[f32],
    beta: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    num_heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(num_heads > 0, "num_heads must be > 0");
    assert_eq!(alpha.len(), num_heads, "alpha len");
    assert_eq!(beta.len(), num_heads, "beta len");
    assert_eq!(dt_bias.len(), num_heads, "dt_bias len");
    assert_eq!(ssm_a.len(), num_heads, "ssm_a len");
    let shared = MTLResourceOptions::StorageModeShared;
    let mk_bytes =
        |ptr: *const std::ffi::c_void, len: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            let nn = NonNull::new(ptr as *mut std::ffi::c_void).expect("buffer ptr is null");
            unsafe {
                ctx.device
                    .newBufferWithBytes_length_options(nn, len, shared)
                    .expect("Metal: failed to create buffer")
            }
        };
    let alpha_buf = mk_bytes(alpha.as_ptr() as *const _, std::mem::size_of_val(alpha));
    let beta_buf = mk_bytes(beta.as_ptr() as *const _, std::mem::size_of_val(beta));
    let dt_buf = mk_bytes(dt_bias.as_ptr() as *const _, std::mem::size_of_val(dt_bias));
    let a_buf = mk_bytes(ssm_a.as_ptr() as *const _, std::mem::size_of_val(ssm_a));
    let nh = num_heads as u32;
    let nh_buf = mk_bytes(&nh as *const u32 as *const _, std::mem::size_of::<u32>());

    let cmd = ctx.queue.commandBuffer().expect("Metal: command buffer");
    let enc = cmd.computeCommandEncoder().expect("Metal: encoder");
    enc.setComputePipelineState(&ctx.gdn_alpha_beta_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&alpha_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&beta_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&dt_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&nh_buf), 0, 4);
    }
    let tg_width = ctx.gdn_alpha_beta_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: num_heads.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let a_c: NonNull<std::ffi::c_void> = alpha_buf.contents();
    let alpha_out: &[f32] =
        unsafe { std::slice::from_raw_parts(a_c.as_ptr() as *const f32, num_heads) };
    let b_c: NonNull<std::ffi::c_void> = beta_buf.contents();
    let beta_out: &[f32] =
        unsafe { std::slice::from_raw_parts(b_c.as_ptr() as *const f32, num_heads) };
    (alpha_out.to_vec(), beta_out.to_vec())
}

// ── GDN carrier encode-only helpers (commit 없이 encoder 에 dispatch) ──

/// ssm_conv1d_silu 를 encoder 에 encode. in/weight/out device buffer.
/// grid = ceil(channels/tg_width), thread 1=channel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ssm_conv1d_silu(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    ch_buf: &ProtocolObject<dyn MTLBuffer>,
    ks_buf: &ProtocolObject<dyn MTLBuffer>,
    channels: usize,
) {
    enc.setComputePipelineState(&ctx.ssm_conv_silu_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ch_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(ks_buf), 0, 4);
    }
    let tg_width = ctx.ssm_conv_silu_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: channels.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// delta_net_step 을 encoder 에 encode. state device buffer in-place 갱신.
/// grid=num_heads(threadgroup 1=head), tg=head_v_dim(thread vi=state row).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_delta_net_step(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    v_offset: usize,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_buf: &ProtocolObject<dyn MTLBuffer>,
    state_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    hk_buf: &ProtocolObject<dyn MTLBuffer>,
    hv_buf: &ProtocolObject<dyn MTLBuffer>,
    nk_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    head_v_dim: usize,
) {
    enc.setComputePipelineState(&ctx.delta_net_step_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(v_buf), v_offset, 2);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(beta_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(hk_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(hv_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(nk_buf), 0, 9);
    }
    let grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: head_v_dim,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// l2_norm 을 encoder 에 encode. grid=n_rows(threadgroup 1=row), tg=256.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_l2_norm(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    in_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_offset: usize,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
    scale_buf: &ProtocolObject<dyn MTLBuffer>,
    n_rows: usize,
) {
    enc.setComputePipelineState(&ctx.l2_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), in_offset, 0);
        enc.setBuffer_offset_atIndex(Some(out_buf), out_offset, 1);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(scale_buf), 0, 4);
    }
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// gdn_alpha_beta 를 encoder 에 encode (alpha/beta in-place 전처리).
/// grid=ceil(num_heads/tg_width), thread 1=head.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_alpha_beta(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    alpha_buf: &ProtocolObject<dyn MTLBuffer>,
    beta_buf: &ProtocolObject<dyn MTLBuffer>,
    dt_buf: &ProtocolObject<dyn MTLBuffer>,
    ssm_a_buf: &ProtocolObject<dyn MTLBuffer>,
    nh_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
) {
    enc.setComputePipelineState(&ctx.gdn_alpha_beta_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(alpha_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(beta_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dt_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ssm_a_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(nh_buf), 0, 4);
    }
    let tg_width = ctx.gdn_alpha_beta_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: num_heads.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// gemv_q4k_simd — SIMD-group 협력 Q4_K GEMV (memory-bound 최적화 후보)
// ---------------------------------------------------------------------------
//
// 현재 gemv_q4k 는 thread 당 1 output row 를 K 전체 순차 처리(GPU 병렬성 0,
// uncoalesced). decode GEMV 는 memory-bandwidth-bound(roofline intensity ~4
// ≪ Apple GPU 임계 ~50)라 bandwidth 활용이 핵심. SIMD-group(32 lane)이 1 row
// 를 블록 stride 로 분할 처리 + simd_sum reduction → 병렬성 + 부분 coalescing.
// (첫 milestone — multiple rows/simdgroup 은 후속.)

/// Q4_K GEMV (SIMD 협력, 복사 경로). 측정/테스트용.
pub fn gemv_q4k_simd_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemv_q4k_simd_dispatch(ctx, &w_buf, 0, input, n, k)
}

/// SIMD 협력 Q4_K GEMV dispatch. `gemv_q4k_dispatch` 와 동일 시그니처 —
/// resident 경로에서 교체 가능. threadgroup = 1 SIMD-group(32 lane), grid = N row.
pub(crate) fn gemv_q4k_simd_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q4_K block size)");
    assert_eq!(input.len(), k);
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.q4k_simd_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
    }
    // 1 threadgroup = 1 SIMD-group(32 lane), 1 row 담당. grid = N row.
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out_slice.to_vec()
}

// ---------------------------------------------------------------------------
// gemv_q4k_coalesced — llama.cpp kernel_mul_mv_q4_K_f32_impl single-row 이식
// ---------------------------------------------------------------------------
//
// gemv_q4k_simd 의 fallback(b += 32 stride) 경로는 인접 lane 이 144byte 떨어진
// super-block 을 읽어 coalescing 이 붕괴한다. llama 커널은 32 lane 을
// ix=lane/8(0..3) × it=lane%8(0..7) 로 나눠 super-block 을 stride-4 순회 →
// 인접 thread 가 super-block 내 인접 데이터를 읽어 coalesced. dequant 결과는
// gemv_q4k 와 수치 동일(fp32 누적). standalone 측정/테스트용 — chain 미연결.

/// Q4_K GEMV (coalesced, 복사 경로). 측정/테스트용.
pub fn gemv_q4k_coalesced_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    gemv_q4k_coalesced_dispatch(ctx, &w_buf, 0, input, n, k)
}

/// Coalesced Q4_K GEMV dispatch. `gemv_q4k_simd_dispatch` 와 동일 시그니처 —
/// resident 경로에서 교체 가능. threadgroup = 1 SIMD-group(32 lane), grid = N row.
pub(crate) fn gemv_q4k_coalesced_dispatch(
    ctx: &MetalContext,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(k % 256, 0, "K must be a multiple of 256 (Q4_K block size)");
    assert_eq!(input.len(), k);
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(&ctx.q4k_coalesced_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
    }
    // 1 threadgroup = 1 SIMD-group(32 lane), nr0=2 row 담당. grid = ceil(N/2).
    let grid = MTLSize {
        width: n.div_ceil(2),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: SIMD_WIDTH,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out_slice.to_vec()
}

// ---------------------------------------------------------------------------
// gemv_q5k/q6k/q8_0 coalesced — llama.cpp nr0=2 multi-row 이식 (측정/테스트용 복사 경로)
// ---------------------------------------------------------------------------
//
// q5k_simd/q6k_simd 의 fallback(stride-32) 은 인접 lane 이 super-block(176/210 byte)
// 간격으로 떨어진 데이터를 읽어 coalescing 이 붕괴한다. llama 커널은 32 lane 을
// quant 별 lane 매핑으로 super-block 을 stride-4(q5k)/stride-2(q6k)/stride-8(q8_0)
// 순회 → 인접 thread 가 super-block 내 인접 데이터를 읽어 coalesced. dequant 결과는
// 각 quant 의 baseline 과 수치 동일(fp32 누적). chain 은 encode_gemv_{q5k,q6k}_simd /
// encode_gemv_q8_0 의 ctx.gemv_coalesced 분기로 연결됨 — 여기는 standalone 테스트 경로.

/// Q5_K GEMV (coalesced, 복사 경로). 측정/테스트용.
pub fn gemv_q5k_coalesced_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 176,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    coalesced_dispatch(
        ctx,
        &ctx.q5k_coalesced_pipeline,
        &w_buf,
        0,
        input,
        n,
        k,
        256,
    )
}

/// Q6_K GEMV (coalesced, 복사 경로). 측정/테스트용.
pub fn gemv_q6k_coalesced_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    coalesced_dispatch(
        ctx,
        &ctx.q6k_coalesced_pipeline,
        &w_buf,
        0,
        input,
        n,
        k,
        256,
    )
}

/// Q4_K GEMV (coalesced nsg2, 복사 경로). 측정/테스트용.
pub fn gemv_q4k_coalesced_nsg2_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 144,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    coalesced_dispatch_with_layout(
        ctx,
        &ctx.q4k_coalesced_nsg2_pipeline,
        &w_buf,
        0,
        input,
        n,
        k,
        256,
        4,
        SIMD_WIDTH * 2,
    )
}

/// Q6_K GEMV (coalesced nsg2, 복사 경로). 측정/테스트용.
pub fn gemv_q6k_coalesced_nsg2_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 256;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 210,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    coalesced_dispatch_with_layout(
        ctx,
        &ctx.q6k_coalesced_nsg2_pipeline,
        &w_buf,
        0,
        input,
        n,
        k,
        256,
        4,
        SIMD_WIDTH * 2,
    )
}

/// Q8_0 GEMV (coalesced, 복사 경로). 측정/테스트용. block size=32(K/32 blocks).
pub fn gemv_q8_0_coalesced_with_ctx(
    ctx: &MetalContext,
    block_bytes: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_blocks = k / 32;
    assert_eq!(
        block_bytes.len(),
        n * num_blocks * 34,
        "block_bytes length mismatch"
    );
    let shared = MTLResourceOptions::StorageModeShared;
    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(block_bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("block_bytes ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, block_bytes.len(), shared)
            .expect("Metal: failed to create weight buffer")
    };
    coalesced_dispatch(
        ctx,
        &ctx.q8_0_coalesced_pipeline,
        &w_buf,
        0,
        input,
        n,
        k,
        32,
    )
}

/// 공통 coalesced GEMV dispatch (nr0=2 multi-row). threadgroup = 1 SIMD-group(32 lane),
/// grid = ceil(N/2). `block_elems` 는 quant 의 block element 수(K-quant 256, Q8_0 32) —
/// `K % block_elems == 0` 검사용. weight buffer/offset 는 호출측이 준비한다(NoCopy 호환).
#[allow(clippy::too_many_arguments)]
pub(crate) fn coalesced_dispatch(
    ctx: &MetalContext,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
    block_elems: usize,
) -> Vec<f32> {
    coalesced_dispatch_with_layout(
        ctx,
        pipeline,
        w_buf,
        weight_byte_offset,
        input,
        n,
        k,
        block_elems,
        2,
        SIMD_WIDTH,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn coalesced_dispatch_with_layout(
    ctx: &MetalContext,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: u32,
    input: &[f32],
    n: usize,
    k: usize,
    block_elems: usize,
    rows_per_threadgroup: usize,
    threads_per_threadgroup: usize,
) -> Vec<f32> {
    assert_eq!(
        k % block_elems,
        0,
        "K must be a multiple of block element count"
    );
    assert_eq!(input.len(), k);
    let shared = MTLResourceOptions::StorageModeShared;

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };
    let output_bytes = n * std::mem::size_of::<f32>();
    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let off_u32 = weight_byte_offset;
    let mk_u32 = |v: &u32| -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr =
                NonNull::new(v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr is null");
            ctx.device
                .newBufferWithBytes_length_options(ptr, std::mem::size_of::<u32>(), shared)
                .expect("Metal: failed to create u32 buffer")
        }
    };
    let n_buf = mk_u32(&n_u32);
    let k_buf = mk_u32(&k_u32);
    let off_buf = mk_u32(&off_u32);

    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = ctx
        .queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");
    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
    }
    // rows_per_threadgroup=2: 기존 1 SIMD-group, rows_per_threadgroup=4: nsg2.
    let grid = MTLSize {
        width: n.div_ceil(rows_per_threadgroup),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: threads_per_threadgroup,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out_slice.to_vec()
}

// ---------------------------------------------------------------------------
// gemv_f32 — PoC, stateless (변경 없음)
// ---------------------------------------------------------------------------

/// weight[N x K] * input[K] = out[N] 을 GPU 에서 계산해 반환.
/// device/compile 실패 시 panic (fake fallback 없음).
pub fn gemv_f32(weight: &[f32], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(weight.len(), n * k);
    assert_eq!(input.len(), k);

    // 1. device
    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().expect("Metal: no system default device");

    // 2. command queue
    let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = device
        .newCommandQueue()
        .expect("Metal: failed to create command queue");

    // 3. compile MSL source
    let source = NSString::from_str(GEMV_F32_SRC);
    let library: Retained<ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("Metal: failed to compile gemv_f32.metal");

    // 4. get function
    let fn_name = NSString::from_str("gemv_f32");
    let function = library
        .newFunctionWithName(&fn_name)
        .expect("Metal: function 'gemv_f32' not found in library");

    // 5. compute pipeline state
    let pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>> = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("Metal: failed to create compute pipeline state");

    // 6. buffers
    let weight_bytes = weight.len() * std::mem::size_of::<f32>();
    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let output_bytes = n * std::mem::size_of::<f32>();
    let k_u32 = k as u32;
    let k_bytes = std::mem::size_of::<u32>();

    let shared = MTLResourceOptions::StorageModeShared;

    let w_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr =
            NonNull::new(weight.as_ptr() as *mut std::ffi::c_void).expect("weight ptr is null");
        device
            .newBufferWithBytes_length_options(ptr, weight_bytes, shared)
            .expect("Metal: failed to create weight buffer")
    };

    let i_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut std::ffi::c_void).expect("input ptr is null");
        device
            .newBufferWithBytes_length_options(ptr, input_bytes, shared)
            .expect("Metal: failed to create input buffer")
    };

    let o_buf: Retained<ProtocolObject<dyn MTLBuffer>> = device
        .newBufferWithLength_options(output_bytes, shared)
        .expect("Metal: failed to create output buffer");

    let k_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr =
            NonNull::new(&k_u32 as *const u32 as *mut std::ffi::c_void).expect("k ptr is null");
        device
            .newBufferWithBytes_length_options(ptr, k_bytes, shared)
            .expect("Metal: failed to create K buffer")
    };

    // 7. encode & dispatch
    let cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>> = queue
        .commandBuffer()
        .expect("Metal: failed to create command buffer");

    let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd
        .computeCommandEncoder()
        .expect("Metal: failed to create compute command encoder");

    enc.setComputePipelineState(&pipeline);

    unsafe {
        enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&i_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 3);
    }

    // 1 thread per output row; threadgroup size = 1
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();

    // 8. commit & wait
    cmd.commit();
    cmd.waitUntilCompleted();

    // 9. read back output
    let contents: NonNull<std::ffi::c_void> = o_buf.contents();
    let out_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, n) };
    out_slice.to_vec()
}

#[cfg(test)]
mod align_tests {
    use super::page_align;

    const PAGE: usize = 16384;

    #[test]
    fn aligned_ptr_zero_offset() {
        // ptr 이 page 경계면 offset 0, len 은 raw_len round-up.
        let (aligned, off, buf_len) = page_align(PAGE, 100);
        assert_eq!(aligned, PAGE);
        assert_eq!(off, 0);
        assert_eq!(buf_len, PAGE); // 100 round-up to 16384
    }

    #[test]
    fn mid_page_ptr_has_offset() {
        // page 중간에서 시작: aligned 는 경계로 내림, off 는 그 차이.
        let ptr = PAGE + 500;
        let (aligned, off, buf_len) = page_align(ptr, 200);
        assert_eq!(aligned, PAGE);
        assert_eq!(off, 500);
        assert_eq!(buf_len, PAGE); // 500+200=700 round-up to 16384
    }

    #[test]
    fn weight_spanning_two_pages() {
        // off+len 이 한 page 를 넘으면 buf_len 은 2 page.
        let ptr = PAGE + 16000;
        let (aligned, off, buf_len) = page_align(ptr, 1000);
        assert_eq!(aligned, PAGE);
        assert_eq!(off, 16000);
        assert_eq!(buf_len, 2 * PAGE); // 16000+1000=17000 round-up to 32768
    }

    #[test]
    fn exact_page_multiple_len() {
        let (_, off, buf_len) = page_align(PAGE, PAGE);
        assert_eq!(off, 0);
        assert_eq!(buf_len, PAGE);
    }
}

/// batching 효과 상한 측정: M개 GEMV 를 (A) M번 개별 commit+wait vs
/// (B) 한 command buffer 에 M dispatch + commit/wait 1번. speedup 상한 정량화.
/// production 은 GEMV 사이 CPU 의존성으로 이 상한에 못 미치나, 상한이 크면
/// device-resident chain(multi-week) 투자 가치를 정량 근거로 판단.
#[cfg(all(test, target_os = "macos"))]
mod batch_probe {
    use super::*;
    use std::time::Instant;

    fn zero_buf(ctx: &MetalContext, bytes: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        ctx.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .expect("buf")
    }

    fn u32_buf(ctx: &MetalContext, v: u32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr = NonNull::new(&v as *const u32 as *mut std::ffi::c_void).expect("ptr");
            ctx.device
                .newBufferWithBytes_length_options(ptr, 4, MTLResourceOptions::StorageModeShared)
                .expect("u32 buf")
        }
    }

    #[test]
    #[ignore = "requires Metal device; batching 효과 상한 측정"]
    fn batching_upper_bound() {
        let ctx = build_metal_context().expect("no metal device");
        const N: usize = 3072;
        const K: usize = 3072;
        const RUNS: usize = 30;
        let num_blocks = K / 256;

        let n_buf = u32_buf(&ctx, N as u32);
        let k_buf = u32_buf(&ctx, K as u32);
        let off_buf = u32_buf(&ctx, 0);
        let tg_width = ctx.q4k_pipeline.threadExecutionWidth().max(1);
        let grid = MTLSize {
            width: N.div_ceil(tg_width),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        let input = zero_buf(&ctx, K * 4);
        let median = |t: &[f64]| {
            let mut v = t.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        // M 스윕: 작은 layer(7) ~ 전체 decode(196) 근사
        for &m in &[7usize, 14, 28, 56, 112, 196] {
            // weight 내용은 timing 무관 — zero 버퍼 (compute 시간 동일)
            let weights: Vec<_> = (0..m)
                .map(|_| zero_buf(&ctx, N * num_blocks * 144))
                .collect();
            let outputs: Vec<_> = (0..m).map(|_| zero_buf(&ctx, N * 4)).collect();
            let bind = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, j: usize| unsafe {
                enc.setBuffer_offset_atIndex(Some(&weights[j]), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&input), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&outputs[j]), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
            };

            // warmup (한 command buffer 경로)
            for _ in 0..3 {
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&ctx.q4k_pipeline);
                for j in 0..m {
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            // A: m번 개별 commit+wait
            let mut a_times = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let t = Instant::now();
                for j in 0..m {
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&ctx.q4k_pipeline);
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                }
                a_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            // B: 한 command buffer m dispatch + commit/wait 1
            let mut b_times = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let t = Instant::now();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&ctx.q4k_pipeline);
                for j in 0..m {
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                b_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let a = median(&a_times);
            let b = median(&b_times);
            eprintln!(
                "[batch-probe] M={m:>3} N={N} K={K}: A(개별)={a:>7.3}ms \
                 B(1 cmdbuf)={b:>7.3}ms speedup={:.2}x",
                a / b,
            );
        }
    }

    /// pm15 재검증: 실 decode shape(다양) + 순차 dependency(output→input chain) 에서도
    /// 단일 command buffer batch 가 4.7x 를 유지하는지. batch_probe(동일 shape, dep 없음)
    /// 와 달리 carrier(pm14) tie 의 모순을 격리한다.
    /// - speedup 유지 → shape/dep 는 범인 아님. carrier tie = 부분 batch(25%)+사이 host GDN → GDN device화 정당.
    /// - speedup 죽음 → shape/dep 가 batch 죽임 → metal device chain 천장.
    #[test]
    #[ignore = "requires Metal device; pm15 실 shape+순차 dep batch 재검증"]
    fn batching_real_decode_shapes() {
        let ctx = build_metal_context().expect("no metal device");
        const RUNS: usize = 30;
        const MAX_DIM: usize = 12288;
        // 9B 실 decode 1 layer GEMV 시퀀스 (N, K). hidden=4096, ffn=12288, GQA kv=1024.
        let layer_unit: [(usize, usize); 7] = [
            (4096, 4096),  // q_proj
            (1024, 4096),  // k_proj (GQA)
            (1024, 4096),  // v_proj (GQA)
            (4096, 4096),  // o_proj
            (12288, 4096), // ffn gate
            (12288, 4096), // ffn up
            (4096, 12288), // ffn down
        ];
        let tg_width = ctx.q4k_pipeline.threadExecutionWidth().max(1);
        let off_buf = u32_buf(&ctx, 0);
        let median = |t: &[f64]| {
            let mut v = t.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let grid_for = |n: usize| MTLSize {
            width: n.div_ceil(tg_width),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };

        for &num_layers in &[1usize, 2, 4, 8, 16, 32] {
            let seq: Vec<(usize, usize)> = (0..num_layers).flat_map(|_| layer_unit).collect();
            let m = seq.len();
            // weight 내용은 timing 무관(zero). shape 별 크기만 정확히.
            let weights: Vec<_> = seq
                .iter()
                .map(|&(n, k)| zero_buf(&ctx, n * (k / 256) * 144))
                .collect();
            let n_bufs: Vec<_> = seq.iter().map(|&(n, _)| u32_buf(&ctx, n as u32)).collect();
            let k_bufs: Vec<_> = seq.iter().map(|&(_, k)| u32_buf(&ctx, k as u32)).collect();
            // 순차 dependency: acts[j] -> acts[j+1] (read-after-write hazard 직렬화).
            let acts: Vec<_> = (0..=m).map(|_| zero_buf(&ctx, MAX_DIM * 4)).collect();
            let bind = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, j: usize| unsafe {
                enc.setBuffer_offset_atIndex(Some(&weights[j]), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&acts[j]), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&acts[j + 1]), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&n_bufs[j]), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&k_bufs[j]), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
            };

            // warmup (단일 command buffer 경로)
            for _ in 0..3 {
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&ctx.q4k_pipeline);
                for j in 0..m {
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_for(seq[j].0), tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            // A: m번 개별 commit+wait (per-op decode)
            let mut a_times = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let t = Instant::now();
                for j in 0..m {
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&ctx.q4k_pipeline);
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_for(seq[j].0), tg);
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                }
                a_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            // B: 한 command buffer m dispatch + commit/wait 1 (device chain)
            let mut b_times = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let t = Instant::now();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&ctx.q4k_pipeline);
                for j in 0..m {
                    bind(&enc, j);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_for(seq[j].0), tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                b_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let a = median(&a_times);
            let b = median(&b_times);
            eprintln!(
                "[batch-real] L={num_layers:>2} M={m:>3} (실shape+순차dep): \
                 A(개별)={a:>7.3}ms B(1 cmdbuf)={b:>7.3}ms speedup={:.2}x",
                a / b,
            );
        }
    }

    /// Roofline: 현재 q4k baseline(thread=1/row) vs q4k_simd(SIMD-group 협력) 커널이
    /// 실제 메모리 대역폭을 몇 % 활용하는지 측정. 단일 command buffer 에 M회 반복 dispatch
    /// 로 commit/wait overhead 를 amortize → 순수 kernel throughput 격리. weight=zero
    /// (메모리 read 량 동일, timing 무관). M5 Pro peak = 307 GB/s.
    /// 실행: cargo test -p rnb-backend-metal --features metal --release \
    ///         -- --ignored --nocapture roofline_q4k_bw
    #[test]
    #[ignore = "requires Metal device; GEMV 대역폭 roofline 측정"]
    fn roofline_q4k_bw() {
        let ctx = build_metal_context().expect("no metal device");
        const RUNS: usize = 30;
        const M: usize = 16; // 단일 cmdbuf 반복 → commit/wait amortize
        const PEAK_GBS: f64 = 307.0; // M5 Pro unified memory peak
                                     // 9B 실 decode GEMV shape (label, N, K). + 동일shape probe 비교용.
        let shapes: [(&str, usize, usize); 5] = [
            ("q/o     4096x4096 ", 4096, 4096),
            ("kv-gqa  1024x4096 ", 1024, 4096),
            ("gate/up 12288x4096", 12288, 4096),
            ("down    4096x12288", 4096, 12288),
            ("probe   3072x3072 ", 3072, 3072),
        ];
        let median = |t: &[f64]| {
            let mut v = t.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        eprintln!(
            "[roofline] M5 Pro peak={PEAK_GBS}GB/s, M={M} dispatch/cmdbuf, RUNS={RUNS} median"
        );
        for &(label, n, k) in &shapes {
            let num_blocks = k / 256;
            let wbytes = n * num_blocks * 144;
            let input = zero_buf(&ctx, k * 4);
            let n_buf = u32_buf(&ctx, n as u32);
            let k_buf = u32_buf(&ctx, k as u32);
            let off_buf = u32_buf(&ctx, 0);
            // buffer마다 다른 값으로 채워 cache/zero-page aliasing 제거 (Phase 0).
            let shared = MTLResourceOptions::StorageModeShared;
            let weights: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = (0..M)
                .map(|j| {
                    let buf = ctx
                        .device
                        .newBufferWithLength_options(wbytes, shared)
                        .expect("weight buf");
                    let ptr = buf.contents().as_ptr() as *mut u8;
                    let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, wbytes) };
                    let seed = (j as u8).wrapping_mul(31).wrapping_add(7);
                    for (i, b) in bytes.iter_mut().enumerate() {
                        *b = seed.wrapping_add((i as u8).wrapping_mul(13));
                    }
                    buf
                })
                .collect();
            let outputs: Vec<_> = (0..M).map(|_| zero_buf(&ctx, n * 4)).collect();
            let bind = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, j: usize| unsafe {
                enc.setBuffer_offset_atIndex(Some(&weights[j]), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&input), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&outputs[j]), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&off_buf), 0, 5);
            };
            // 단일 cmdbuf 에 M dispatch, RUNS median → per-GEMV ms 반환.
            let measure = |pipe: &ProtocolObject<dyn MTLComputePipelineState>,
                           grid: MTLSize,
                           tg: MTLSize|
             -> f64 {
                for _ in 0..3 {
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(pipe);
                    for j in 0..M {
                        bind(&enc, j);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                }
                let mut times = Vec::with_capacity(RUNS);
                for _ in 0..RUNS {
                    let t = Instant::now();
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(pipe);
                    for j in 0..M {
                        bind(&enc, j);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    times.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                median(&times) / M as f64
            };
            let gbs = |per_ms: f64| (wbytes as f64 / 1e9) / (per_ms / 1000.0);
            // baseline: 1-thread/row, grid=N/execWidth
            let exec_w = ctx.q4k_pipeline.threadExecutionWidth().max(1);
            let base_per = measure(
                &ctx.q4k_pipeline,
                MTLSize {
                    width: n.div_ceil(exec_w),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: exec_w,
                    height: 1,
                    depth: 1,
                },
            );
            // simd: row당 1 SIMD-group(32 lane), grid=N
            let simd_per = measure(
                &ctx.q4k_simd_pipeline,
                MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH,
                    height: 1,
                    depth: 1,
                },
            );
            let (bg, sg) = (gbs(base_per), gbs(simd_per));
            eprintln!(
                "[roofline] {label} ({:>5.1}MB): baseline={bg:>6.1}GB/s({:>4.1}%)  \
                 simd={sg:>6.1}GB/s({:>4.1}%)  simd/base={:.2}x",
                wbytes as f64 / 1_048_576.0,
                bg / PEAK_GBS * 100.0,
                sg / PEAK_GBS * 100.0,
                sg / bg,
            );
        }
    }
}

/// pm45 M2-1: split_conv_qkv Metal 커널이 host CPU split 과 bit-exact 인지 검증.
#[cfg(all(test, target_os = "macos"))]
mod split_conv_qkv_tests {
    use super::*;

    fn f32_buf(ctx: &MetalContext, data: &[f32]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let buf = ctx
            .device
            .newBufferWithLength_options(data.len() * 4, MTLResourceOptions::StorageModeShared)
            .expect("f32 buf");
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buf
    }

    fn out_buf(ctx: &MetalContext, len: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        ctx.device
            .newBufferWithLength_options(len * 4, MTLResourceOptions::StorageModeShared)
            .expect("out buf")
    }

    fn u32_buf(ctx: &MetalContext, v: u32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            let ptr = NonNull::new(&v as *const u32 as *mut std::ffi::c_void).expect("ptr");
            ctx.device
                .newBufferWithBytes_length_options(ptr, 4, MTLResourceOptions::StorageModeShared)
                .expect("u32 buf")
        }
    }

    fn read_f32(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len).to_vec() }
    }

    // host `gdn_prefill.rs split_conv_qkv` 1:1 oracle.
    fn cpu_split(
        conv: &[f32],
        seq: usize,
        cc: usize,
        q: usize,
        k: usize,
        v: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut qo = vec![0.0f32; seq * q];
        let mut ko = vec![0.0f32; seq * k];
        let mut vo = vec![0.0f32; seq * v];
        for t in 0..seq {
            let off = t * cc;
            qo[t * q..(t + 1) * q].copy_from_slice(&conv[off..off + q]);
            ko[t * k..(t + 1) * k].copy_from_slice(&conv[off + q..off + q + k]);
            vo[t * v..(t + 1) * v].copy_from_slice(&conv[off + q + k..off + q + k + v]);
        }
        (qo, ko, vo)
    }

    #[test]
    #[ignore = "requires Metal device; split_conv_qkv bit-exact"]
    fn split_conv_qkv_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let seq_len = 5usize;
        // 27B GDN: head_k_dim=128*num_k_heads=16 → q=k=2048, head_v_dim=128*num_v_heads=32 → v=4096.
        let (q_dim, k_dim, v_dim) = (2048usize, 2048usize, 4096usize);
        // 잔여 채널 8 추가 → q+k+v < conv_channels (gate 등 skip 경로 검증).
        let conv_channels = q_dim + k_dim + v_dim + 8;
        let conv: Vec<f32> = (0..seq_len * conv_channels)
            .map(|i| (i as f32) * 0.013 - 7.0)
            .collect();
        let (q_cpu, k_cpu, v_cpu) = cpu_split(&conv, seq_len, conv_channels, q_dim, k_dim, v_dim);

        let conv_buf = f32_buf(&ctx, &conv);
        let q_gpu_buf = out_buf(&ctx, seq_len * q_dim);
        let k_gpu_buf = out_buf(&ctx, seq_len * k_dim);
        let v_gpu_buf = out_buf(&ctx, seq_len * v_dim);
        let seq_b = u32_buf(&ctx, seq_len as u32);
        let cc_b = u32_buf(&ctx, conv_channels as u32);
        let q_b = u32_buf(&ctx, q_dim as u32);
        let k_b = u32_buf(&ctx, k_dim as u32);
        let v_b = u32_buf(&ctx, v_dim as u32);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_split_conv_qkv(
            &ctx,
            &enc,
            &conv_buf,
            0,
            &q_gpu_buf,
            &k_gpu_buf,
            &v_gpu_buf,
            &seq_b,
            &cc_b,
            &q_b,
            &k_b,
            &v_b,
            seq_len,
            conv_channels,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        assert_eq!(read_f32(&q_gpu_buf, seq_len * q_dim), q_cpu, "q mismatch");
        assert_eq!(read_f32(&k_gpu_buf, seq_len * k_dim), k_cpu, "k mismatch");
        assert_eq!(read_f32(&v_gpu_buf, seq_len * v_dim), v_cpu, "v mismatch");
    }

    // host `gdn_prefill.rs repeat_qk_for_value_heads` 1:1 oracle.
    fn cpu_repeat(
        q: &[f32],
        k: &[f32],
        seq: usize,
        nk: usize,
        nv: usize,
        hk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        if nv == nk {
            return (q.to_vec(), k.to_vec());
        }
        let mut qr = Vec::with_capacity(seq * nv * hk);
        let mut kr = Vec::with_capacity(seq * nv * hk);
        for t in 0..seq {
            for vh in 0..nv {
                let kh = vh % nk;
                let src = (t * nk + kh) * hk;
                qr.extend_from_slice(&q[src..src + hk]);
                kr.extend_from_slice(&k[src..src + hk]);
            }
        }
        (qr, kr)
    }

    #[test]
    #[ignore = "requires Metal device; repeat_qk GQA bit-exact"]
    fn repeat_qk_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let seq_len = 5usize;
        // 27B GDN: num_k_heads=16, num_v_heads=32, head_k_dim=128 (GQA 2x).
        let (nk, nv, hk) = (16usize, 32usize, 128usize);
        let q: Vec<f32> = (0..seq_len * nk * hk)
            .map(|i| (i as f32) * 0.011 - 3.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * nk * hk)
            .map(|i| (i as f32) * -0.007 + 1.5)
            .collect();
        let (q_cpu, k_cpu) = cpu_repeat(&q, &k, seq_len, nk, nv, hk);

        let q_in = f32_buf(&ctx, &q);
        let k_in = f32_buf(&ctx, &k);
        let q_gpu = out_buf(&ctx, seq_len * nv * hk);
        let k_gpu = out_buf(&ctx, seq_len * nv * hk);
        let seq_b = u32_buf(&ctx, seq_len as u32);
        let nk_b = u32_buf(&ctx, nk as u32);
        let nv_b = u32_buf(&ctx, nv as u32);
        let hk_b = u32_buf(&ctx, hk as u32);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_repeat_qk(
            &ctx, &enc, &q_in, &k_in, &q_gpu, &k_gpu, &seq_b, &nk_b, &nv_b, &hk_b, seq_len, nv, hk,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        assert_eq!(
            read_f32(&q_gpu, seq_len * nv * hk),
            q_cpu,
            "q repeat mismatch"
        );
        assert_eq!(
            read_f32(&k_gpu, seq_len * nv * hk),
            k_cpu,
            "k repeat mismatch"
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod argument_buffer_tests {
    use super::*;
    use objc2_foundation::NSRange;
    use objc2_metal::{MTLArgumentEncoder, MTLFunction, MTLResource, MTLResourceUsage};
    use std::alloc::{alloc_zeroed, dealloc, Layout};

    const ARG_BUFFER_PROBE_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct WeightTable {
    array<device const uchar*, 4> weight [[id(0)]];
};

kernel void argument_buffer_select_probe(
    constant WeightTable& table [[buffer(0)]],
    constant uint& index        [[buffer(1)]],
    device uint* out            [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    device const uchar* p = table.weight[index & 3u];
    out[0] = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}
"#;

    unsafe fn nocopy_buf(
        device: &ProtocolObject<dyn MTLDevice>,
        raw: &[u8],
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u32) {
        let (aligned, page_off, buf_len) = page_align(raw.as_ptr() as usize, raw.len());
        let ptr = NonNull::new(aligned as *mut std::ffi::c_void).expect("aligned ptr");
        let buf = device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                buf_len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .expect("NoCopy probe buffer");
        (buf, page_off as u32)
    }

    #[test]
    #[ignore = "requires Metal device; verifies GPU-indexed NoCopy buffer table"]
    fn argument_buffer_indexes_nocopy_weight_slices() {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().expect("no metal device");
        let queue = device.newCommandQueue().expect("argument-buffer queue");

        let source = NSString::from_str(ARG_BUFFER_PROBE_SRC);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .unwrap_or_else(|e| panic!("argument-buffer probe MSL compile failed: {e:?}"));
        let fn_name = NSString::from_str("argument_buffer_select_probe");
        let function = library
            .newFunctionWithName(&fn_name)
            .expect("argument-buffer probe function");
        let arg_encoder = unsafe { function.newArgumentEncoderWithBufferIndex(0) };
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| panic!("argument-buffer probe pipeline failed: {e:?}"));

        let layout = Layout::from_size_align(METAL_PAGE, METAL_PAGE).expect("page layout");
        let backing_ptr = unsafe { alloc_zeroed(layout) };
        assert!(!backing_ptr.is_null(), "page allocation failed");
        let backing = unsafe { std::slice::from_raw_parts_mut(backing_ptr, METAL_PAGE) };
        backing[64..68].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        backing[128..132].copy_from_slice(&0x5566_7788u32.to_le_bytes());
        backing[192..196].copy_from_slice(&0x99aa_bbccu32.to_le_bytes());
        backing[256..260].copy_from_slice(&0xddee_ff00u32.to_le_bytes());

        let slices = [
            &backing[64..80],
            &backing[128..144],
            &backing[192..208],
            &backing[256..272],
        ];
        let wrapped: Vec<_> = slices
            .iter()
            .map(|slice| unsafe { nocopy_buf(&device, slice) })
            .collect();

        let arg_len = arg_encoder.encodedLength() as usize;
        let arg_buf = device
            .newBufferWithLength_options(arg_len, MTLResourceOptions::StorageModeShared)
            .expect("argument buffer allocation");
        unsafe {
            arg_encoder.setArgumentBuffer_offset(Some(&arg_buf), 0);
            let mut buffers: Vec<*const ProtocolObject<dyn MTLBuffer>> = wrapped
                .iter()
                .map(|(buffer, _)| &**buffer as *const _)
                .collect();
            let mut offsets: Vec<usize> =
                wrapped.iter().map(|(_, offset)| *offset as usize).collect();
            arg_encoder.setBuffers_offsets_withRange(
                NonNull::new(buffers.as_mut_ptr()).expect("argument buffer table ptr"),
                NonNull::new(offsets.as_mut_ptr()).expect("argument buffer offset ptr"),
                NSRange {
                    location: 0,
                    length: buffers.len(),
                },
            );
        }

        let index_value = 2u32;
        let index_buf = unsafe {
            let ptr = NonNull::new(&index_value as *const u32 as *mut std::ffi::c_void)
                .expect("index ptr");
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    std::mem::size_of::<u32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("index buffer")
        };
        let out_buf = device
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .expect("out buffer");

        let cmd = queue.commandBuffer().expect("argument-buffer command");
        let enc = cmd
            .computeCommandEncoder()
            .expect("argument-buffer encoder");
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&arg_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&index_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        }
        enc.useResource_usage(
            ProtocolObject::<dyn MTLResource>::from_ref(&*arg_buf),
            MTLResourceUsage::Read,
        );
        enc.useResource_usage(
            ProtocolObject::<dyn MTLResource>::from_ref(&*index_buf),
            MTLResourceUsage::Read,
        );
        enc.useResource_usage(
            ProtocolObject::<dyn MTLResource>::from_ref(&*out_buf),
            MTLResourceUsage::Write,
        );
        for (buffer, _) in &wrapped {
            enc.useResource_usage(
                ProtocolObject::<dyn MTLResource>::from_ref(&**buffer),
                MTLResourceUsage::Read,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "argument-buffer command status"
        );

        let got = unsafe { *(out_buf.contents().as_ptr() as *const u32) };
        unsafe { dealloc(backing_ptr, layout) };
        assert_eq!(got, 0x99aa_bbcc);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod qwen_moe_shared_mul_mm_tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum SharedQuant {
        Q4K,
        Q6K,
    }

    fn shared_bytes(ctx: &MetalContext, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let ptr = NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void).expect("bytes ptr");
        unsafe {
            ctx.device
                .newBufferWithBytes_length_options(
                    ptr,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("shared bytes buffer")
        }
    }

    fn shared_f32(ctx: &MetalContext, values: &[f32]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let ptr = NonNull::new(values.as_ptr() as *mut std::ffi::c_void).expect("f32 ptr");
        unsafe {
            ctx.device
                .newBufferWithBytes_length_options(
                    ptr,
                    std::mem::size_of_val(values),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("shared f32 buffer")
        }
    }

    fn empty_f32(ctx: &MetalContext, elements: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        ctx.device
            .newBufferWithLength_options(
                elements * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("empty f32 buffer")
    }

    fn test_block(quant: SharedQuant, seed: usize) -> (Vec<u8>, [f32; 256]) {
        match quant {
            SharedQuant::Q4K => {
                let mut block = crate::tests_fixture::q4k_block_fixed();
                block[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                block[2..4].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
                for (index, value) in block[16..].iter_mut().enumerate() {
                    *value = ((seed * 29 + index * 17 + 11) & 0xff) as u8;
                }
                let values = crate::tests_fixture::q4k_dequant(&block);
                (block, values)
            }
            SharedQuant::Q6K => {
                let mut block = crate::tests_fixture::q6k_block_fixed();
                block[208..210].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                for (index, value) in block[..192].iter_mut().enumerate() {
                    *value = ((seed * 31 + index * 13 + 7) & 0xff) as u8;
                }
                let values = crate::tests_fixture::q6k_dequant(&block);
                (block, values)
            }
        }
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        for (index, (&candidate, &baseline)) in got.iter().zip(want).enumerate() {
            assert!(
                candidate.is_finite(),
                "{label}: candidate became non-finite at {index}: {candidate}"
            );
            assert!(
                baseline.is_finite(),
                "{label}: baseline became non-finite at {index}: {baseline}"
            );
            let abs = (candidate - baseline).abs();
            let rel = abs / baseline.abs().max(1.0);
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_abs <= 0.03 && max_rel <= 0.005,
            "{label}: max_abs={max_abs:e} max_rel={max_rel:e} worst={worst} got={} want={}",
            got[worst],
            want[worst],
        );
    }

    fn run_oracle(quant: SharedQuant) {
        let ctx = build_metal_context().expect("no Metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const WEIGHT_OFFSET: usize = 32;
        for (n, k, m) in [(65usize, 256usize, 128usize), (67, 259, 33)] {
            let blocks_per_row = k.div_ceil(256);
            let block_bytes = match quant {
                SharedQuant::Q4K => 144,
                SharedQuant::Q6K => 210,
            };
            let mut weight_arena =
                vec![0xa5u8; WEIGHT_OFFSET + n * blocks_per_row * block_bytes + 16];
            let mut dequantized = Vec::with_capacity(n * blocks_per_row * 256);
            let mut cursor = WEIGHT_OFFSET;
            for row in 0..n {
                for block_index in 0..blocks_per_row {
                    let (block, values) = test_block(quant, row * 131 + block_index * 17);
                    weight_arena[cursor..cursor + block_bytes].copy_from_slice(&block);
                    cursor += block_bytes;
                    dequantized.extend(
                        values
                            .into_iter()
                            .map(|value| half::f16::from_f32(value).to_f32()),
                    );
                }
            }

            let input = (0..m * k)
                .map(|index| ((index * 19 + 5) % 113) as f32 * 0.0002 - 0.011)
                .collect::<Vec<_>>();
            let oracle_input = input
                .iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<_>>();
            let weights_buf = shared_bytes(&ctx, &weight_arena);
            let input_buf = shared_f32(&ctx, &input);
            let output_buf = empty_f32(&ctx, m * n);

            let command = ctx.queue.commandBuffer().expect("command buffer");
            let encoder = command.computeCommandEncoder().expect("compute encoder");
            match quant {
                SharedQuant::Q4K => encode_qwen_moe_shared_mul_mm_q4k_f32(
                    &ctx,
                    &encoder,
                    &weights_buf,
                    WEIGHT_OFFSET,
                    &input_buf,
                    &output_buf,
                    n,
                    k,
                    m,
                ),
                SharedQuant::Q6K => encode_qwen_moe_shared_mul_mm_q6k_f32(
                    &ctx,
                    &encoder,
                    &weights_buf,
                    WEIGHT_OFFSET,
                    &input_buf,
                    &output_buf,
                    n,
                    k,
                    m,
                ),
            }
            .expect("shared quant encode");
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(
                command.status(),
                MTLCommandBufferStatus::Completed,
                "shared {quant:?} command failed for M={m} N={n} K={k}"
            );

            let got = unsafe {
                std::slice::from_raw_parts(output_buf.contents().as_ptr() as *const f32, m * n)
                    .to_vec()
            };
            let mut oracle = vec![0.0f32; m * n];
            let dequantized_row_stride = blocks_per_row * 256;
            for token in 0..m {
                let activation = &oracle_input[token * k..(token + 1) * k];
                for row in 0..n {
                    let weight_start = row * dequantized_row_stride;
                    oracle[token * n + row] = dequantized[weight_start..weight_start + k]
                        .iter()
                        .zip(activation)
                        .map(|(&weight, &value)| weight * value)
                        .sum();
                }
            }
            assert_close(&format!("{quant:?}/F32/M={m}/N={n}/K={k}"), &got, &oracle);
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_shared_mul_mm_q4k_f32_matches_cpu_oracle() {
        run_oracle(SharedQuant::Q4K);
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_shared_mul_mm_q6k_f32_matches_cpu_oracle() {
        run_oracle(SharedQuant::Q6K);
    }
}
