use super::*;
use crate::runtime::dense::dense_q6_batch_f16_down_enabled_for;
use crate::runtime::gemv::Q4kF16DenseChainOutput;
use crate::runtime::mtp_verify::{GGML_F32, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};
use crate::runtime::q6_f16_prewarm_enabled;
use rnb_backend_api::{
    BackendKind as MoeJitBackendKind, DeviceTensorDesc, DeviceTensorRole, MoeJitByteRange,
    MoeJitExpertLoad, MoeJitLoadRequest, MoeJitLoadSink, ScalarType,
};
use rnb_memory::{ExpertBundleCacheStats, ExpertBundleObservationReceipt};

#[path = "tests/mtp_verify.rs"]
mod mtp_verify_tests;

static CUDA_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

struct RuntimeTestGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Drop for RuntimeTestGuard {
    fn drop(&mut self) {
        reset_default_cuda_compute_for_test();
    }
}

fn lock_default_cuda_compute_for_test() -> Option<std::sync::MutexGuard<'static, Option<CudaState>>>
{
    DEFAULT_CUDA_COMPUTE.get().map(|compute| {
        compute
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    })
}

fn reset_default_cuda_compute_for_test() {
    let previous = lock_default_cuda_compute_for_test().and_then(|mut guard| guard.take());
    drop(previous);
}

#[test]
fn cuda_weight_residency_size_guards_allow_only_quant_sized_payloads() {
    assert_eq!(crate::runtime::q4k_raw_bytes_per_block_for_test(), 144);
    assert_eq!(
        crate::runtime::q4k_packed_q8dot_bytes_per_block_for_test(),
        148
    );
    assert_eq!(crate::runtime::q6k_raw_bytes_per_block_for_test(), 210);
    assert_eq!(
        crate::runtime::q6k_packed_q8dot_bytes_per_block_for_test(),
        274
    );

    assert!(crate::runtime::validate_q4k_packed_payload_bytes_for_test(148).is_ok());
    assert!(crate::runtime::validate_q6k_packed_payload_bytes_for_test(274).is_ok());

    let q4_f16_full = 256 * std::mem::size_of::<u16>();
    let q4_f32_full = 256 * std::mem::size_of::<f32>();
    let q6_f16_full = 256 * std::mem::size_of::<u16>();
    let q6_f32_full = 256 * std::mem::size_of::<f32>();

    assert!(crate::runtime::validate_q4k_packed_payload_bytes_for_test(q4_f16_full).is_err());
    assert!(crate::runtime::validate_q4k_packed_payload_bytes_for_test(q4_f32_full).is_err());
    assert!(crate::runtime::validate_q6k_packed_payload_bytes_for_test(q6_f16_full).is_err());
    assert!(crate::runtime::validate_q6k_packed_payload_bytes_for_test(q6_f32_full).is_err());
}

#[test]
fn cuda_weight_residency_counter_tracks_expanded_and_native_separately() {
    let mut counters = crate::runtime::CudaWeightResidencyCounters::default();
    counters.record_q4_expanded_f16_for_test(512);
    counters.record_q4_expanded_f32_for_test(1024);
    counters.record_q6_expanded_f16_for_test(512);
    counters.record_q6_expanded_f32_for_test(1024);
    counters.record_native_f32_for_test(4096);
    counters.record_packed_q8dot_for_test(148);

    assert_eq!(counters.expanded_diag_bytes, 3072);
    assert_eq!(counters.native_f32_bytes, 4096);
    assert_eq!(counters.packed_q8dot_bytes, 148);
    assert_eq!(counters.q4_expanded_f16_bytes, 512);
    assert_eq!(counters.q4_expanded_f32_bytes, 1024);
    assert_eq!(counters.q6_expanded_f16_bytes, 512);
    assert_eq!(counters.q6_expanded_f32_bytes, 1024);
}

#[test]
fn cuda_weight_residency_counters_track_raw_quant_and_transient_uploads() {
    let mut counters = crate::runtime::CudaWeightResidencyCounters::default();

    counters.record_q4_raw_quant(144);
    counters.record_q6_raw_quant(210);
    counters.record_q4_transient_quant_upload(288);
    counters.record_q6_transient_quant_upload(420);

    assert_eq!(counters.raw_quant_bytes, 354);
    assert_eq!(counters.q4_raw_quant_bytes, 144);
    assert_eq!(counters.q6_raw_quant_bytes, 210);
    assert_eq!(counters.transient_quant_upload_bytes, 708);
    assert_eq!(counters.q4_transient_quant_upload_bytes, 288);
    assert_eq!(counters.q6_transient_quant_upload_bytes, 420);
    assert_eq!(counters.expanded_diag_bytes, 0);
}

#[test]
fn cuda_weight_residency_counter_delta_includes_raw_quant_and_transient_uploads() {
    let mut before = crate::runtime::CudaWeightResidencyCounters::default();
    before.record_q4_raw_quant(144);
    before.record_q4_transient_quant_upload(144);

    let mut after = before;
    after.record_q4_raw_quant(288);
    after.record_q6_raw_quant(210);
    after.record_q4_transient_quant_upload(432);
    after.record_q6_transient_quant_upload(210);

    let delta = after.delta(before);
    assert_eq!(delta.raw_quant_bytes, 498);
    assert_eq!(delta.q4_raw_quant_bytes, 288);
    assert_eq!(delta.q6_raw_quant_bytes, 210);
    assert_eq!(delta.transient_quant_upload_bytes, 642);
    assert_eq!(delta.q4_transient_quant_upload_bytes, 432);
    assert_eq!(delta.q6_transient_quant_upload_bytes, 210);
}

#[test]
fn quant_resident_env_defaults_to_auto_budget() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::remove("RNB_CUDA_QUANT_RESIDENT_MB");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 9 * 1024, 4_700, 1_024)
            .expect("budget plan");

    assert!(plan.enabled);
    assert_eq!(plan.raw_quant_target_mib, 4_700);
    assert_eq!(plan.packed_promotion_target_mib, 932);
}

#[test]
fn quant_resident_env_off_disables_default_policy() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "off");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 9 * 1024, 4_700, 1_024)
            .expect("budget plan");

    assert_eq!(plan.raw_quant_target_mib, 0);
    assert_eq!(plan.packed_promotion_target_mib, 0);
    assert!(!plan.enabled);
}

#[test]
fn quant_resident_env_auto_uses_free_memory_after_reserve() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "auto");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 9 * 1024, 4_700, 1_024)
            .expect("budget plan");

    assert!(plan.enabled);
    assert!(plan.raw_quant_target_mib > 0);
    assert!(plan.raw_quant_target_mib <= 4_700);
    assert!(
        plan.raw_quant_target_mib + plan.packed_promotion_target_mib <= 6 * 1024,
        "10GB-class opt-in must stay near the M4 initial VRAM envelope"
    );
}

#[test]
fn quant_resident_env_numeric_caps_resident_budget() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "2048");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 9 * 1024, 4_700, 1_024)
            .expect("budget plan");

    assert_eq!(plan.raw_quant_target_mib, 2048);
    assert_eq!(plan.packed_promotion_target_mib, 0);
    assert!(plan.enabled);
}

#[test]
fn quant_resident_env_numeric_clamps_to_available_after_reserve() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "8192");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 5 * 1024, 4_700, 1_024)
            .expect("budget plan");

    assert_eq!(plan.raw_quant_target_mib, 1_536);
    assert_eq!(plan.packed_promotion_target_mib, 0);
    assert!(plan.enabled);
}

#[test]
fn quant_resident_env_numeric_clamps_to_model_and_packed_candidates() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "8192");

    let plan =
        crate::runtime::quant_resident_budget_plan_for_test(24 * 1024, 24 * 1024, 1_024, 256)
            .expect("budget plan");

    assert_eq!(plan.raw_quant_target_mib, 1_024);
    assert_eq!(plan.packed_promotion_target_mib, 256);
    assert!(plan.enabled);
}

#[test]
fn quant_resident_env_auto_disables_when_free_does_not_clear_reserve() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "auto");

    let plan = crate::runtime::quant_resident_budget_plan_for_test(10 * 1024, 1_024, 4_700, 1_024)
        .expect("budget plan");

    assert_eq!(plan.raw_quant_target_mib, 0);
    assert_eq!(plan.packed_promotion_target_mib, 0);
    assert!(!plan.enabled);
}

#[test]
fn cuda_quant_resident_q4_prewarm_off_does_not_open_cuda() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "off");
    reset_default_cuda_compute_for_test();
    let weights = make_test_q4k_weights(2, 4, 1, 31);
    let refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let warmed = crate::runtime::prewarm_quant_resident_q4k_weights(&refs)
        .expect("prewarm should be a no-op when quant resident is explicitly off");

    assert_eq!(warmed, 0);
    let state_opened = lock_default_cuda_compute_for_test()
        .map(|guard| guard.is_some())
        .unwrap_or(false);
    assert!(
        !state_opened,
        "explicit-off quant resident prewarm must not open CUDA state"
    );
}

#[test]
fn cuda_quant_resident_q4_prewarm_invalid_env_errors_before_cuda_open() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "invalid");
    reset_default_cuda_compute_for_test();
    let weights = make_test_q4k_weights(2, 4, 1, 31);
    let refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let err = crate::runtime::prewarm_quant_resident_q4k_weights(&refs)
        .expect_err("invalid quant resident env should fail before CUDA open");

    assert!(err.contains("RNB_CUDA_QUANT_RESIDENT_MB must be auto, off, or integer MiB"));
    let state_opened = lock_default_cuda_compute_for_test()
        .map(|guard| guard.is_some())
        .unwrap_or(false);
    assert!(
        !state_opened,
        "invalid quant resident env must not open CUDA state"
    );
}

#[test]
fn cuda_quant_resident_q4_prewarm_empty_candidates_still_validate_env() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "invalid");
    reset_default_cuda_compute_for_test();

    let err = crate::runtime::prewarm_quant_resident_q4k_weights(&[])
        .expect_err("invalid quant resident env should fail even with no candidates");

    assert!(err.contains("RNB_CUDA_QUANT_RESIDENT_MB must be auto, off, or integer MiB"));
    let state_opened = lock_default_cuda_compute_for_test()
        .map(|guard| guard.is_some())
        .unwrap_or(false);
    assert!(
        !state_opened,
        "invalid quant resident env must not open CUDA state"
    );
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_quant_resident_q4_prewarm_default_records_raw_quant() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::remove("RNB_CUDA_QUANT_RESIDENT_MB");
    reset_default_cuda_compute_for_test();
    let weights = make_test_q4k_weights(2, 4, 1, 37);
    let expected_bytes = weights.iter().map(Vec::len).sum::<usize>() as u64;
    let refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let warmed =
        crate::runtime::prewarm_quant_resident_q4k_weights(&refs).expect("prewarm default");

    assert_eq!(warmed, refs.len());
    let counters = lock_default_cuda_compute_for_test()
        .and_then(|guard| guard.as_ref().map(CudaState::weight_residency_counters))
        .expect("default CUDA state counters");
    assert_eq!(counters.q4_raw_quant_bytes, expected_bytes);
    assert_eq!(counters.expanded_diag_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_quant_resident_q4_prewarm_respects_fixed_budget_without_temp_upload() {
    let _guard = runtime_test_lock();
    let _env = EnvVarGuard::set("RNB_CUDA_QUANT_RESIDENT_MB", "1");
    reset_default_cuda_compute_for_test();
    let weights = make_test_q4k_weights(2, 256, 16, 41);
    let expected_first_bytes = weights[0].len() as u64;
    let refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let warmed =
        crate::runtime::prewarm_quant_resident_q4k_weights(&refs).expect("fixed budget prewarm");

    assert_eq!(warmed, 1);
    let counters = lock_default_cuda_compute_for_test()
        .and_then(|guard| guard.as_ref().map(CudaState::weight_residency_counters))
        .expect("default CUDA state counters");
    assert_eq!(counters.q4_raw_quant_bytes, expected_first_bytes);
    assert_eq!(counters.transient_quant_upload_bytes, 0);
    assert_eq!(counters.expanded_diag_bytes, 0);
}

#[test]
fn cuda_weight_residency_expanded_envs_require_single_allow_gate() {
    let _guard = runtime_test_lock();
    let vars = [
        "RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE",
        "RNB_CUDA_Q4K_PREFILL_F16_GEMM",
        "RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM",
        "RNB_CUDA_Q4K_PREFILL_F16_O_PROJ",
        "RNB_CUDA_Q4K_PREFILL_F32_GEMM",
        "RNB_CUDA_Q4K_BATCH_F16_GATE_UP",
        "RNB_CUDA_Q4K_BATCH_F16_DOWN",
        "RNB_CUDA_Q4K_BATCH_F32_GATE_UP",
        "RNB_CUDA_Q6K_BATCH_F16_DOWN",
        "RNB_CUDA_Q6K_BATCH_F32_DOWN",
        "RNB_CUDA_Q6_F16_CACHE_MB",
        "RNB_CUDA_Q6_F32_CACHE_MB",
        "RNB_CUDA_Q4_F32_CACHE_MB",
    ];
    let _cleared = vars
        .iter()
        .map(|&name| EnvVarGuard::remove(name))
        .collect::<Vec<_>>();

    let _q4_f16_prefill = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    let _q4_f16_gate = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");
    let _q4_f16_down = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_DOWN", "1");
    let _q6_f16_down = EnvVarGuard::set("RNB_CUDA_Q6K_BATCH_F16_DOWN", "1");

    assert!(!crate::tuning::prefill_q4k_f16_gemm_enabled());
    assert!(!crate::tuning::prefill_q4k_f16_qkv_gemm_enabled());
    assert!(!crate::tuning::prefill_q4k_f16_o_proj_enabled());
    assert!(!crate::runtime::dense_q4_batch_f16_gate_up_enabled_for_test(1024, 6144, false));
    assert!(!crate::runtime::dense_q4_batch_f16_down_enabled_for_test(
        true
    ));
    assert!(!dense_q6_batch_f16_down_enabled_for(false, 1024, 10752));

    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");

    assert!(crate::tuning::prefill_q4k_f16_gemm_enabled());
    assert!(crate::runtime::dense_q4_batch_f16_gate_up_enabled_for_test(
        1024, 6144, false
    ));
    assert!(crate::runtime::dense_q4_batch_f16_down_enabled_for_test(
        true
    ));
    assert!(dense_q6_batch_f16_down_enabled_for(false, 1024, 10752));
}

#[test]
fn cuda_dense_q6_down_dispatch_prefers_packed_over_expanded_when_diag_gate_is_set() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q6K_BATCH_F16_DOWN", "force");
    let _packed = EnvVarGuard::remove("RNB_CUDA_DENSE_Q6_PACKED_Q8DOT");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "1");

    assert_eq!(
        crate::runtime::dense_q6_down_dispatch_plan_for_test(true, 14, 1115, 10752, 2560, true,),
        crate::runtime::DenseDownDispatchPlanForTest::PackedQ8Dot
    );
}

#[test]
fn cuda_dense_q4_gate_up_dispatch_prefers_packed_over_expanded_when_diag_gate_is_set() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");
    let _packed = EnvVarGuard::remove("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT");

    assert_eq!(
        crate::runtime::dense_q4_gate_up_dispatch_plan_for_test(1115, 10752, 2560, true, true),
        crate::runtime::DenseGateUpDispatchPlanForTest::PackedQ8Dot
    );
}

#[test]
fn cuda_dense_q4_down_dispatch_prefers_raw_quant_over_expanded_when_diag_gate_is_set() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_DOWN", "1");

    assert_eq!(
        crate::runtime::dense_q4_down_dispatch_plan_for_test(1115, 10752, 2560, true),
        crate::runtime::DenseDownDispatchPlanForTest::RawQuant
    );
}

#[test]
fn cuda_q4_projection_dispatch_prefers_raw_quant_over_expanded_f16_when_diag_gate_is_set() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "1");

    for kind in [
        crate::runtime::DenseQ4ProjectionKindForTest::Qkv,
        crate::runtime::DenseQ4ProjectionKindForTest::Ple,
        crate::runtime::DenseQ4ProjectionKindForTest::Output,
    ] {
        assert_eq!(
            crate::runtime::dense_q4_projection_dispatch_plan_for_test(
                kind, 1115, 2560, 2560, true
            ),
            crate::runtime::DenseQ4ProjectionDispatchPlanForTest::RawQuant
        );
    }
}

#[test]
fn cuda_q4_projection_dispatch_uses_expanded_f16_when_output_raw_is_unsupported() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "1");

    assert_eq!(
        crate::runtime::dense_q4_projection_dispatch_plan_for_test(
            crate::runtime::DenseQ4ProjectionKindForTest::Output,
            1115,
            2560,
            2560,
            false,
        ),
        crate::runtime::DenseQ4ProjectionDispatchPlanForTest::ExpandedF16
    );
}

#[test]
fn cuda_q4_projection_dispatch_force_enables_output_expanded_f16_diagnostic() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "force");

    assert_eq!(
        crate::runtime::dense_q4_projection_dispatch_plan_for_test(
            crate::runtime::DenseQ4ProjectionKindForTest::Output,
            1115,
            2560,
            2560,
            true,
        ),
        crate::runtime::DenseQ4ProjectionDispatchPlanForTest::ExpandedF16
    );
}

fn q4k_f16_qkv_raw_projection_env_guards_for_test() -> Vec<EnvVarGuard> {
    [
        "RNB_CUDA_Q4K_FUSED_NAIVE",
        "RNB_CUDA_Q4K_FUSED_WMMA",
        "RNB_CUDA_Q4K_FUSED_WMMA_4WARP",
        "RNB_CUDA_Q4K_DP4A",
        "RNB_CUDA_Q4K_DP4A_TILE",
        "RNB_CUDA_Q4K_MMA",
        "RNB_CUDA_Q4K_MMA_4WARP",
    ]
    .iter()
    .map(|&name| EnvVarGuard::remove(name))
    .collect()
}

fn run_q4k_f16_qkv_gemm_batch_fixture_for_test(
    state: &mut CudaState,
) -> Result<Option<()>, String> {
    let q_rows = 512usize;
    let kv_rows = 256usize;
    let cols = 512usize;
    let seq_len = 2usize;
    let blocks_per_row = cols / 256;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 607)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 613)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 617)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.00390625)
        .collect::<Vec<_>>();

    state
        .q4k_f16_qkv_gemm_batch(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
        )
        .map(|output| output.map(|_| ()))
}

fn run_q4k_f16_qkv_postprocess_hd256_fixture_for_test(
    state: &mut CudaState,
) -> Result<Option<()>, String> {
    let q_rows = 512usize;
    let kv_rows = 256usize;
    let cols = 512usize;
    let seq_len = 2usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 619)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 631)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 641)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.00325)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.70 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + (i % 17) as f32 * 0.005859375)
        .collect::<Vec<_>>();

    state
        .q4k_f16_qkv_postprocess_hd256(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            10000.0,
            3,
            1.0e-5,
            true,
            true,
            true,
        )
        .map(|output| output.map(|_| ()))
}

fn run_q4k_f16_qkv_attention_hd512_fixture_for_test(
    state: &mut CudaState,
) -> Result<Option<()>, String> {
    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let cols = 1024usize;
    let seq_len = 2usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 643)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 647)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 653)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.003)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.75 + (i % 17) as f32 * 0.00625)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.80 + (i % 19) as f32 * 0.0046875)
        .collect::<Vec<_>>();

    state
        .q4k_f16_qkv_prefill_attention_hd512(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            1.0 / (head_dim as f32).sqrt(),
            10000.0,
            0,
            1.0e-5,
            true,
            true,
            true,
        )
        .map(|output| output.map(|_| ()))
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_qkv_raw_projection_runs_with_zero_resident_cache_and_no_expanded_admission() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::remove("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
    let _f16 = EnvVarGuard::remove("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
    let _raw_projection = q4k_f16_qkv_raw_projection_env_guards_for_test();
    let _dp4a = EnvVarGuard::set("RNB_CUDA_Q4K_DP4A", "1");

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = 0;
    let before = state.weight_residency_counters();
    let gemm = run_q4k_f16_qkv_gemm_batch_fixture_for_test(&mut state)
        .expect("Q4 raw QKV GEMM with zero resident cache");
    let delta = state.weight_residency_counters().delta(before);

    assert!(gemm.is_some());
    assert_eq!(state.resident_q4k_bytes, 0);
    assert_eq!(delta.q4_expanded_f16_bytes, 0);
    assert_eq!(delta.q4_expanded_f32_bytes, 0);
    assert_eq!(delta.q6_expanded_f16_bytes, 0);
    assert_eq!(delta.q6_expanded_f32_bytes, 0);
    assert_eq!(delta.packed_q8dot_bytes, 0);
    assert_eq!(delta.expanded_diag_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_output_projection_force_diag_admits_expanded_f16() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _global = EnvVarGuard::remove("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    let _qkv = EnvVarGuard::remove("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
    let _o_proj = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "force");
    let _q4_gate = EnvVarGuard::remove("RNB_CUDA_Q4K_BATCH_F16_GATE_UP");
    let _q4_down = EnvVarGuard::remove("RNB_CUDA_Q4K_BATCH_F16_DOWN");
    let _q6_down = EnvVarGuard::remove("RNB_CUDA_Q6K_BATCH_F16_DOWN");

    let n_embd = 512usize;
    let q_dim = 512usize;
    let n_ff = 1024usize;
    let seq_len = 2usize;
    let q_blocks = q_dim / 256;
    let hidden_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let o = make_test_q4k_weights(1, n_embd, q_blocks, 701)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, n_ff, hidden_blocks, 709)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, hidden_blocks, 719)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 727)
        .pop()
        .unwrap();
    let mut hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
        .collect::<Vec<_>>();
    let attn_out = (0..seq_len * q_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.75 + (i % 17) as f32 * 0.003)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.82 + (i % 19) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.91 + (i % 23) as f32 * 0.0015)
        .collect::<Vec<_>>();

    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        q_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut hidden,
        &attn_out,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense Q4_K attention+FFN batch chain with forced O-proj F16 diagnostic");

    let counters = lock_default_cuda_compute_for_test()
        .and_then(|guard| guard.as_ref().map(CudaState::weight_residency_counters))
        .expect("default CUDA state counters");
    assert!(hidden.iter().all(|value| value.is_finite()));
    assert!(counters.q4_expanded_f16_bytes > 0);
    assert_eq!(counters.q4_expanded_f32_bytes, 0);
    assert_eq!(counters.q6_expanded_f16_bytes, 0);
    assert_eq!(counters.q6_expanded_f32_bytes, 0);
    assert_eq!(counters.expanded_diag_bytes, counters.q4_expanded_f16_bytes);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_attention_projection_chain_does_not_admit_expanded_by_default() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::remove("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
    let _global = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let _o_proj = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "1");
    let _raw_projection = q4k_f16_qkv_raw_projection_env_guards_for_test();

    let mut state = CudaState::open().expect("open CUDA state");
    let before = state.weight_residency_counters();
    let chain = run_q4k_f16_qkv_attention_hd512_fixture_for_test(&mut state)
        .expect("Q4 F16 QKV attention chain default gate");
    let delta = state.weight_residency_counters().delta(before);

    assert!(chain.is_none());
    assert_eq!(delta.q4_expanded_f16_bytes, 0);
    assert_eq!(delta.q4_expanded_f32_bytes, 0);
    assert_eq!(delta.q6_expanded_f16_bytes, 0);
    assert_eq!(delta.q6_expanded_f32_bytes, 0);
    assert_eq!(delta.expanded_diag_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_qkv_f16_facades_do_not_admit_expanded_without_diag_gate() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::remove("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
    let _global = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let _raw_projection = q4k_f16_qkv_raw_projection_env_guards_for_test();

    let mut state = CudaState::open().expect("open CUDA state");
    let before = state.weight_residency_counters();
    let gemm = run_q4k_f16_qkv_gemm_batch_fixture_for_test(&mut state)
        .expect("Q4 F16 QKV GEMM default gate");
    let postprocess = run_q4k_f16_qkv_postprocess_hd256_fixture_for_test(&mut state)
        .expect("Q4 F16 QKV postprocess default gate");
    let delta = state.weight_residency_counters().delta(before);

    assert!(gemm.is_none());
    assert!(postprocess.is_none());
    assert_eq!(delta.q4_expanded_f16_bytes, 0);
    assert_eq!(delta.q4_expanded_f32_bytes, 0);
    assert_eq!(delta.q6_expanded_f16_bytes, 0);
    assert_eq!(delta.q6_expanded_f32_bytes, 0);
    assert_eq!(delta.expanded_diag_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_qkv_f16_facades_admit_expanded_only_with_diag_gate() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _global = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let _raw_projection = q4k_f16_qkv_raw_projection_env_guards_for_test();

    let mut state = CudaState::open().expect("open CUDA state");
    let before = state.weight_residency_counters();
    let gemm = run_q4k_f16_qkv_gemm_batch_fixture_for_test(&mut state)
        .expect("Q4 F16 QKV GEMM diagnostic gate");
    let postprocess = run_q4k_f16_qkv_postprocess_hd256_fixture_for_test(&mut state)
        .expect("Q4 F16 QKV postprocess diagnostic gate");
    let delta = state.weight_residency_counters().delta(before);

    assert!(gemm.is_some());
    assert!(postprocess.is_some());
    assert!(delta.q4_expanded_f16_bytes > 0);
    assert_eq!(delta.q4_expanded_f32_bytes, 0);
    assert_eq!(delta.q6_expanded_f16_bytes, 0);
    assert_eq!(delta.q6_expanded_f32_bytes, 0);
    assert_eq!(delta.expanded_diag_bytes, delta.q4_expanded_f16_bytes);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_dense_ffn_host_path_prefers_packed_or_raw_over_expanded_diag_candidates() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _q4_gate = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");
    let _q4_down = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_DOWN", "1");
    let _q6_down = EnvVarGuard::set("RNB_CUDA_Q6K_BATCH_F16_DOWN", "force");
    let _packed_q4 = EnvVarGuard::remove("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT");
    let _packed_q6 = EnvVarGuard::remove("RNB_CUDA_DENSE_Q6_PACKED_Q8DOT");
    reset_default_cuda_compute_for_test();

    let n_embd = 1024usize;
    let n_ff = 1024usize;
    let seq_len = 2usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 151)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 157)
        .pop()
        .unwrap();
    let down_q6 = make_test_q6k_weights(1, n_embd, down_blocks, 163)
        .pop()
        .unwrap();
    let down_q4 = make_test_q4k_weights(1, n_embd, down_blocks, 179)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 59.0) - 29.0) * 0.00390625)
        .collect::<Vec<_>>();

    dense_q4k_gelu_ffn_batch(&gate, &up, &down_q6, 14, n_ff, n_embd, seq_len, &input)
        .expect("Q6 down host FFN");
    dense_q4k_gelu_ffn_batch(&gate, &up, &down_q4, 12, n_ff, n_embd, seq_len, &input)
        .expect("Q4 down host FFN");

    let counters = lock_default_cuda_compute_for_test()
        .and_then(|guard| guard.as_ref().map(CudaState::weight_residency_counters))
        .expect("default CUDA state counters");
    assert_eq!(counters.q4_expanded_f16_bytes, 0);
    assert_eq!(counters.q4_expanded_f32_bytes, 0);
    assert_eq!(counters.q6_expanded_f16_bytes, 0);
    assert_eq!(counters.q6_expanded_f32_bytes, 0);
    assert!(counters.packed_q8dot_bytes > 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_dense_ffn_dev_input_path_prefers_packed_or_raw_over_expanded_diag_candidates() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _q4_gate = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");
    let _q6_down = EnvVarGuard::set("RNB_CUDA_Q6K_BATCH_F16_DOWN", "force");
    let _packed_q4 = EnvVarGuard::remove("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT");
    let _packed_q6 = EnvVarGuard::remove("RNB_CUDA_DENSE_Q6_PACKED_Q8DOT");

    let n_embd = 1024usize;
    let n_ff = 1024usize;
    let seq_len = 2usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 251)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 257)
        .pop()
        .unwrap();
    let down_q6 = make_test_q6k_weights(1, n_embd, down_blocks, 263)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 67.0) - 33.0) * 0.00390625)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    let input_bytes = std::mem::size_of_val(input.as_slice());
    let input_dev = state.compute_input_ptr(input_bytes).expect("input dev");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                state.stream,
            )
            .expect("input h2d");
    }
    let output_bytes = seq_len * n_embd * std::mem::size_of::<f32>();
    let output_dev = state.compute_output_ptr(output_bytes).expect("output dev");
    let mut trace_stage = std::time::Instant::now();

    state
        .dense_q4k_gelu_ffn_batch_dev_input_to_dev(
            &gate,
            &up,
            &down_q6,
            14,
            n_ff,
            n_embd,
            seq_len,
            input_dev,
            output_dev,
            None,
            None,
            &mut trace_stage,
        )
        .expect("Q6 down dev-input FFN");
    state.stream_synchronize().expect("dev-input FFN sync");

    let counters = state.weight_residency_counters();
    assert_eq!(counters.q4_expanded_f16_bytes, 0);
    assert_eq!(counters.q4_expanded_f32_bytes, 0);
    assert_eq!(counters.q6_expanded_f16_bytes, 0);
    assert_eq!(counters.q6_expanded_f32_bytes, 0);
    assert!(counters.packed_q8dot_bytes > 0);
}

#[test]
#[ignore]
fn cuda_weight_residency_q4_f16_transient_admission_records_expanded_counter() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let mut state = crate::runtime::CudaState::open().expect("cuda open");
    let rows = 1usize;
    let blocks = 1usize;
    let weights = vec![0u8; 144];
    let before = state.weight_residency_counters();
    let _ptr = state
        .transient_q4k_f16_ptr(&weights, rows, blocks)
        .expect("Q4 F16 transient");
    let delta = state.weight_residency_counters().delta(before);
    assert_eq!(delta.q4_expanded_f16_bytes, 512);
    assert_eq!(delta.expanded_diag_bytes, 512);
}

#[test]
#[ignore]
fn cuda_weight_residency_q4_packed_admission_records_packed_counter() {
    let _guard = runtime_test_lock();
    let mut state = crate::runtime::CudaState::open().expect("cuda open");
    state.resident_q4_packed_limit = usize::MAX;
    let rows = 1usize;
    let blocks = 1usize;
    let weights = vec![0u8; 144];
    let before = state.weight_residency_counters();
    let _ptr = state
        .resident_q4k_packed_ptrs(&weights, rows, blocks)
        .expect("Q4 packed admission");
    let delta = state.weight_residency_counters().delta(before);
    assert_eq!(delta.packed_q8dot_bytes, 148);
    assert_eq!(delta.expanded_diag_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_raw_resident_admission_records_raw_quant_without_expanded() {
    let _guard = runtime_test_lock();
    let _arena = EnvVarGuard::set("RNB_CUDA_RESIDENT_Q4K_ARENA", "0");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = usize::MAX;
    let weights = make_test_q4k_weights(1, 4, 1, 17).pop().unwrap();

    let before = state.weight_residency_counters();
    let _ptr = state
        .resident_q4k_weights_ptr(&weights)
        .expect("admit raw Q4_K resident");
    let delta = state.weight_residency_counters().delta(before);

    assert_eq!(delta.q4_raw_quant_bytes, weights.len() as u64);
    assert_eq!(delta.raw_quant_bytes, weights.len() as u64);
    assert_eq!(delta.expanded_diag_bytes, 0);
    assert_eq!(delta.packed_q8dot_bytes, 0);
    assert_eq!(delta.transient_quant_upload_bytes, 0);

    let hit_before = state.weight_residency_counters();
    let _hit_ptr = state
        .resident_q4k_weights_ptr(&weights)
        .expect("hit raw Q4_K resident");
    let hit_delta = state.weight_residency_counters().delta(hit_before);

    assert_eq!(hit_delta.q4_raw_quant_bytes, 0);
    assert_eq!(hit_delta.raw_quant_bytes, 0);
    assert_eq!(hit_delta.q4_transient_quant_upload_bytes, 0);
    assert_eq!(hit_delta.transient_quant_upload_bytes, 0);
    assert_eq!(hit_delta.expanded_diag_bytes, 0);
    assert_eq!(hit_delta.packed_q8dot_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_raw_temp_upload_records_transient_h2d_without_raw_resident() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = 0;
    let weights = make_test_q4k_weights(1, 4, 1, 23).pop().unwrap();

    let before = state.weight_residency_counters();
    let _ptr = state
        .resident_q4k_weights_ptr(&weights)
        .expect("upload temp Q4_K weight");
    let delta = state.weight_residency_counters().delta(before);

    assert_eq!(state.resident_q4k_bytes, 0);
    assert_eq!(delta.q4_raw_quant_bytes, 0);
    assert_eq!(delta.q4_transient_quant_upload_bytes, weights.len() as u64);
    assert_eq!(delta.transient_quant_upload_bytes, weights.len() as u64);
    assert_eq!(delta.expanded_diag_bytes, 0);
    assert_eq!(delta.packed_q8dot_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q4_raw_batch_slab_admission_records_payload_bytes_without_padding() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = usize::MAX;
    let weights = make_test_q4k_weights(2, 1, 1, 29);
    let slot_weights = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let local_ptrs = std::collections::HashMap::new();
    let tracked_keys = slot_weights
        .iter()
        .map(|weights| q4k_resident_key(weights))
        .collect::<HashSet<_>>();

    let cache_before = cache_snapshot();
    let before = state.weight_residency_counters();
    let admitted = state
        .batch_resident_q4k_slot_misses_many_on_stream_recording_expert_bundle_h2d(
            &[slot_weights.as_slice()],
            &local_ptrs,
            state.stream,
            &tracked_keys,
        )
        .expect("batch-admit raw Q4_K residents");
    let delta = state.weight_residency_counters().delta(before);
    let bundle_delta = cache_snapshot().delta(cache_before).expert_bundles;
    let payload_bytes = slot_weights
        .iter()
        .map(|weights| weights.len())
        .sum::<usize>() as u64;

    assert!(admitted);
    assert_eq!(payload_bytes, 288);
    assert_eq!(delta.q4_raw_quant_bytes, payload_bytes);
    assert_eq!(delta.raw_quant_bytes, payload_bytes);
    assert_eq!(delta.expanded_diag_bytes, 0);
    assert_eq!(delta.packed_q8dot_bytes, 0);
    assert_eq!(delta.transient_quant_upload_bytes, 0);
    assert_eq!(bundle_delta.h2d_bytes, payload_bytes);
    assert_eq!(bundle_delta.temp_h2d_bytes, 0);
    assert!(state.resident_q4k_bytes as u64 > payload_bytes);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_decode_selected_temp_slab_records_direct_h2d_payload() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let weights = make_test_q4k_weights(3, 1, 1, 31);
    let gate = [&weights[0][..]];
    let up = [&weights[1][..]];
    let down = [&weights[2][..]];
    let tracked_keys = weights
        .iter()
        .map(|weights| q4k_resident_key(weights))
        .collect::<HashSet<_>>();
    let before = cache_snapshot();

    state
        .temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(&gate, &up, &down, &tracked_keys)
        .expect("upload selected temp slab");

    let payload_bytes = weights.iter().map(Vec::len).sum::<usize>() as u64;
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.h2d_bytes, payload_bytes);
    assert_eq!(delta.temp_h2d_bytes, payload_bytes);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_decode_selected_mixed_records_only_direct_temp_h2d_payload() {
    let _guard = runtime_test_lock();
    let _arena = EnvVarGuard::set("RNB_CUDA_RESIDENT_Q4K_ARENA", "0");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = usize::MAX;
    let weights = make_test_q4k_weights(3, 1, 1, 37);
    let gate = [&weights[0][..]];
    let up = [&weights[1][..]];
    let down = [&weights[2][..]];
    let tracked_keys = weights
        .iter()
        .map(|weights| q4k_resident_key(weights))
        .collect::<HashSet<_>>();
    state
        .resident_q4k_weights_ptr(&weights[0])
        .expect("admit resident gate");
    let before = cache_snapshot();

    state
        .mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
            &gate,
            &up,
            &down,
            &tracked_keys,
        )
        .expect("upload mixed selected temp roles");

    let temp_payload_bytes = (weights[1].len() + weights[2].len()) as u64;
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.h2d_bytes, temp_payload_bytes);
    assert_eq!(delta.temp_h2d_bytes, temp_payload_bytes);
}
#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_decode_selected_mixed_excludes_temp_shared_tail_h2d() {
    let _guard = runtime_test_lock();
    let _arena = EnvVarGuard::set("RNB_CUDA_RESIDENT_Q4K_ARENA", "0");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4k_limit = usize::MAX;
    let selected = make_test_q4k_weights(3, 1, 1, 39);
    let shared = make_test_q4k_weights(3, 1, 1, 43);
    for weights in &selected {
        state
            .resident_q4k_weights_ptr(weights)
            .expect("admit selected resident role");
    }
    let gate = [&selected[0][..], &shared[0][..]];
    let up = [&selected[1][..], &shared[1][..]];
    let down = [&selected[2][..], &shared[2][..]];
    let tracked_keys = selected
        .iter()
        .map(|weights| q4k_resident_key(weights))
        .collect::<HashSet<_>>();
    let before = cache_snapshot();

    let (_, _, _, temp_slab_ptrs) = state
        .mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
            &gate,
            &up,
            &down,
            &tracked_keys,
        )
        .expect("upload only shared tail through temp slab");
    assert!(!temp_slab_ptrs.is_empty());

    let delta = cache_snapshot().delta(before);
    assert_eq!(delta.expert_bundles.h2d_bytes, 0);
    assert_eq!(delta.expert_bundles.temp_h2d_bytes, 0);
}

#[test]
fn device_tensor_storage_owned_and_workspace_are_distinct() {
    use crate::runtime::types::DeviceTensorStorage;

    let owned = DeviceTensorStorage::Owned;
    let workspace = DeviceTensorStorage::NemotronWorkspace {
        arena_id: 7,
        offset: 256,
        bytes: 1024,
    };

    assert!(owned.is_owned());
    assert!(!workspace.is_owned());
    assert_eq!(workspace.workspace_arena_id(), Some(7));
}

#[test]
fn nemotron_route_pack_storage_owned_and_workspace_are_distinct() {
    use crate::runtime::types::NemotronRoutePackStorage;

    let owned = NemotronRoutePackStorage::Owned;
    let workspace = NemotronRoutePackStorage::Workspace { arena_id: 9 };

    assert_ne!(owned, workspace);
    assert_eq!(workspace.workspace_arena_id(), Some(9));
    assert_eq!(owned.workspace_arena_id(), None);
}

#[test]
fn qwen35_group_meta_from_ids_matches_slice_group_meta_for_selected_slots() {
    let gate = [vec![0u8; 11], vec![1u8; 13], vec![2u8; 17], vec![3u8; 19]];
    let up = [vec![4u8; 23], vec![5u8; 29], vec![6u8; 31], vec![7u8; 37]];
    let expert_ids = [1u32, 1, 2, 2, 2, 3, 0, 0];
    let gate_refs = expert_ids
        .iter()
        .map(|&expert| gate[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let up_refs = expert_ids
        .iter()
        .map(|&expert| up[expert as usize].as_slice())
        .collect::<Vec<_>>();

    for max_group in [1usize, 2, 4, 8] {
        assert_eq!(
            build_group_meta_from_ids(&expert_ids, max_group),
            build_group_meta(&gate_refs, &up_refs, max_group)
        );
    }
}

#[test]
fn qwen35_group_shape_summary_counts_lengths_and_slots() {
    let summary = qwen35_group_shape_summary(&[0, 4, 4, 1, 5, 2, 7, 8, 15, 9]).unwrap();

    assert_eq!(summary.groups, 5);
    assert_eq!(summary.slots, 24);
    assert_eq!(summary.max_len, 9);
    assert_eq!(summary.len_hist[1], 1);
    assert_eq!(summary.len_hist[2], 1);
    assert_eq!(summary.len_hist[4], 1);
    assert_eq!(summary.len_hist[8], 1);
    assert_eq!(summary.overflow_groups, 1);
}

#[test]
fn qwen35_group_shape_summary_rejects_invalid_meta() {
    assert!(qwen35_group_shape_summary(&[0, 4, 4]).is_err());
    assert!(qwen35_group_shape_summary(&[0, 0]).is_err());
}

#[test]
fn qwen35_expert_run_batched_down_plan_preserves_same_expert_runs() {
    let plan =
        qwen35_expert_run_batched_down_plan(&[2, 2, 2, 2, 2, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9], 4)
            .expect("expert run-batched down plan");

    assert_eq!(
        plan,
        vec![
            Qwen35ExpertRun {
                expert_id: 2,
                slot_start: 0,
                len: 5,
                full_tiles: 1,
                tail: 1,
            },
            Qwen35ExpertRun {
                expert_id: 7,
                slot_start: 5,
                len: 2,
                full_tiles: 0,
                tail: 2,
            },
            Qwen35ExpertRun {
                expert_id: 9,
                slot_start: 7,
                len: 8,
                full_tiles: 2,
                tail: 0,
            },
        ]
    );
}

#[test]
fn qwen35_expert_run_batched_down_plan_rejects_zero_tile_slots() {
    let err = qwen35_expert_run_batched_down_plan(&[1, 1], 0)
        .expect_err("zero tile size should be rejected");

    assert!(err.contains("max tile slots must be non-zero"));
}

#[test]
fn qwen35_expert_run_tile_meta_splits_runs_into_kernel_tiles() {
    let meta = qwen35_expert_run_tile_meta(&[2, 2, 2, 2, 2, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9], 4)
        .expect("expert run tile meta");

    assert_eq!(meta, vec![0, 4, 4, 1, 5, 2, 7, 4, 11, 4]);
}

#[test]
fn qwen35_expert_run_tile_meta_rejects_zero_tile_slots() {
    let err =
        qwen35_expert_run_tile_meta(&[1, 1], 0).expect_err("zero tile size should be rejected");

    assert!(err.contains("max tile slots must be non-zero"));
}

#[test]
fn qwen35_expert_run_specialized_tile_meta_keeps_run_span() {
    let meta =
        qwen35_expert_run_specialized_tile_meta(&[2, 2, 2, 2, 2, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9], 4)
            .expect("expert run specialized tile meta");

    assert_eq!(
        meta,
        vec![
            Qwen35ExpertRunSpecializedTile {
                expert_id: 2,
                run_start: 0,
                run_len: 5,
                tile_start: 0,
                tile_len: 4,
            },
            Qwen35ExpertRunSpecializedTile {
                expert_id: 2,
                run_start: 0,
                run_len: 5,
                tile_start: 4,
                tile_len: 1,
            },
            Qwen35ExpertRunSpecializedTile {
                expert_id: 7,
                run_start: 5,
                run_len: 2,
                tile_start: 5,
                tile_len: 2,
            },
            Qwen35ExpertRunSpecializedTile {
                expert_id: 9,
                run_start: 7,
                run_len: 8,
                tile_start: 7,
                tile_len: 4,
            },
            Qwen35ExpertRunSpecializedTile {
                expert_id: 9,
                run_start: 7,
                run_len: 8,
                tile_start: 11,
                tile_len: 4,
            },
        ]
    );
}

#[test]
fn qwen35_expert_run_specialized_tile_words_encode_kernel_meta() {
    let words = qwen35_expert_run_specialized_tile_words(&[5, 5, 5, 8, 8], 4)
        .expect("expert run specialized tile words");

    assert_eq!(words, vec![5, 0, 3, 0, 3, 8, 3, 2, 3, 2]);
}

#[test]
fn qwen35_expert_run_specialized_tile_meta_rejects_zero_tile_slots() {
    let err = qwen35_expert_run_specialized_tile_meta(&[1, 1], 0)
        .expect_err("zero tile size should be rejected");

    assert!(err.contains("max tile slots must be non-zero"));
}

#[test]
fn qwen35_expert_run_down_weight_identity_rejects_mismatched_slice_inside_run() {
    let down_a = [1u8, 2, 3, 4];
    let down_b = [1u8, 2, 3, 5];
    let err = qwen35_expert_run_down_weight_identity(&[7, 7], &[&down_a, &down_b])
        .expect_err("same expert run must not reuse mismatched down slices");

    assert!(err.contains("down weight mismatch inside expert run"));
}

#[test]
fn qwen35_expert_run_down_weight_identity_allows_same_slice_inside_run() {
    let down_a = [1u8, 2, 3, 4];
    let down_b = [9u8, 8, 7, 6];

    qwen35_expert_run_down_weight_identity(&[7, 7, 8], &[&down_a, &down_a, &down_b])
        .expect("same expert run may reuse the same down slice");
}

#[test]
fn qwen35_selected_base_slices_match_direct_expert_slices() {
    let gate = [vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]].concat();
    let up = [vec![4u8; 10], vec![5u8; 10], vec![6u8; 10]].concat();
    let down = [vec![7u8; 12], vec![8u8; 12], vec![9u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 3,
    };
    let expert_ids = [2, 0, 2, 1];

    let selected = qwen35_selected_base_slices(&bases, &expert_ids).unwrap();

    assert_eq!(
        selected.gate_weights,
        vec![&gate[16..24], &gate[0..8], &gate[16..24], &gate[8..16]]
    );
    assert_eq!(
        selected.up_weights,
        vec![&up[20..30], &up[0..10], &up[20..30], &up[10..20]]
    );
    assert_eq!(
        selected.down_weights,
        vec![&down[24..36], &down[0..12], &down[24..36], &down[12..24]]
    );
}

#[test]
fn qwen35_selected_base_slices_reject_bad_expert_id() {
    let bytes = vec![0u8; 16];
    let bases = Qwen35SelectedExpertBases {
        gate_all: &bytes,
        up_all: &bytes,
        down_all: &bytes,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 8,
        down_bytes_per_expert: 8,
        n_expert: 2,
    };

    let err = qwen35_selected_base_slices(&bases, &[0, 2]).expect_err("expert 2 is out of range");

    assert!(err.contains("expert id out of range"));
}

#[test]
fn qwen35_selected_base_request_builds_existing_sparse_inputs() {
    let gate = [vec![1u8; 8], vec![2u8; 8]].concat();
    let up = [vec![3u8; 10], vec![4u8; 10]].concat();
    let down = [vec![5u8; 12], vec![6u8; 12]].concat();
    let request = Qwen35SelectedBaseSparseRequest {
        bases: Qwen35SelectedExpertBases {
            gate_all: &gate,
            up_all: &up,
            down_all: &down,
            gate_bytes_per_expert: 8,
            up_bytes_per_expert: 10,
            down_bytes_per_expert: 12,
            n_expert: 2,
        },
        expert_ids: &[1, 0],
        route_weights: &[0.25, 0.75],
        token_ids: &[0, 1],
    };

    let selected = qwen35_selected_base_sparse_inputs_for_test(&request).unwrap();

    assert_eq!(selected.gate_weights.len(), 2);
    assert_eq!(selected.up_weights.len(), 2);
    assert_eq!(selected.down_weights.len(), 2);
    assert_eq!(selected.gate_weights, vec![&gate[8..16], &gate[0..8]]);
    assert_eq!(selected.up_weights, vec![&up[10..20], &up[0..10]]);
    assert_eq!(selected.down_weights, vec![&down[12..24], &down[0..12]]);
    assert_eq!(selected.route_weights, &[0.25, 0.75]);
    assert_eq!(selected.token_ids, &[0, 1]);
}

#[test]
fn qwen35_resident_expert_page_candidates_deduplicate_by_expert_role() {
    let gate = [vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]].concat();
    let up = [vec![4u8; 10], vec![5u8; 10], vec![6u8; 10]].concat();
    let down = [vec![7u8; 12], vec![8u8; 12], vec![9u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 3,
    };

    let candidates = qwen35_resident_expert_page_candidates(
        7,
        &bases,
        &[2, 0, 2, 1],
        &[0.2, 0.7, 0.3, 0.6],
        128,
        12,
        12,
        14,
    )
    .unwrap();

    assert_eq!(candidates.len(), 9);
    let expert2_gate = candidates
        .iter()
        .find(|candidate| {
            candidate.expert_id == 2 && candidate.role == Qwen35ResidentExpertPageRole::Gate
        })
        .expect("expert 2 gate page");
    assert_eq!(expert2_gate.layer_idx, 7);
    assert_eq!(expert2_gate.quant, 12);
    assert_eq!(expert2_gate.byte_offset, 16);
    assert_eq!(expert2_gate.bytes, 8);
    assert_eq!(expert2_gate.reuse_count, 2);
    assert!((expert2_gate.route_weight_sum - 0.5).abs() < f32::EPSILON);
    assert_eq!(expert2_gate.window_tokens, 128);
}

#[test]
fn qwen35_resident_expert_page_plan_prioritizes_reuse_under_budget() {
    let gate = [vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]].concat();
    let up = [vec![4u8; 10], vec![5u8; 10], vec![6u8; 10]].concat();
    let down = [vec![7u8; 12], vec![8u8; 12], vec![9u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 3,
    };
    let candidates = qwen35_resident_expert_page_candidates(
        3,
        &bases,
        &[2, 0, 2, 1],
        &[0.2, 0.7, 0.3, 0.6],
        128,
        12,
        12,
        14,
    )
    .unwrap();

    let plan = qwen35_resident_expert_page_plan(&candidates, 30);

    assert_eq!(
        plan.selected
            .iter()
            .map(|page| (page.expert_id, page.role))
            .collect::<Vec<_>>(),
        vec![
            (2, Qwen35ResidentExpertPageRole::Gate),
            (2, Qwen35ResidentExpertPageRole::Up),
            (2, Qwen35ResidentExpertPageRole::Down),
        ]
    );
    assert_eq!(plan.selected_bytes, 30);
    assert_eq!(plan.spilled.len(), 6);
}

#[test]
fn qwen35_resident_expert_page_candidates_reject_bad_routes() {
    let bytes = vec![0u8; 16];
    let bases = Qwen35SelectedExpertBases {
        gate_all: &bytes,
        up_all: &bytes,
        down_all: &bytes,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 8,
        down_bytes_per_expert: 8,
        n_expert: 2,
    };

    let len_err = qwen35_resident_expert_page_candidates(0, &bases, &[0, 1], &[0.4], 8, 12, 12, 14)
        .expect_err("route weights must match expert ids");
    assert!(len_err.contains("resident expert page route length mismatch"));

    let expert_err =
        qwen35_resident_expert_page_candidates(0, &bases, &[0, 2], &[0.4, 0.6], 8, 12, 12, 14)
            .expect_err("expert 2 is out of range");
    assert!(expert_err.contains("expert id out of range"));
}

#[test]
fn qwen35_resident_expert_page_budget_respects_free_reserve_and_cache_headroom() {
    let gib = 1024usize * 1024 * 1024;

    let cache_limited = qwen35_resident_expert_page_budget(10 * 1024, 6 * 1024, 1024, 4 * gib, gib);
    assert_eq!(cache_limited.budget_bytes, 3 * gib);
    assert_eq!(cache_limited.evicting_budget_bytes, 4 * gib);

    let free_limited = qwen35_resident_expert_page_budget(10 * 1024, 1536, 1024, 4 * gib, gib);
    assert_eq!(free_limited.budget_bytes, 512 * 1024 * 1024);
    assert_eq!(free_limited.evicting_budget_bytes, 512 * 1024 * 1024);

    let exhausted = qwen35_resident_expert_page_budget(10 * 1024, 8 * 1024, 1024, gib, 2 * gib);
    assert_eq!(exhausted.budget_bytes, 0);
    assert_eq!(exhausted.evicting_budget_bytes, gib);
}

#[test]
fn qwen35_selected_sparse_boundary_stats_count_unique_uploads() {
    let _guard = runtime_test_lock();
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "1");
    let offsets = Qwen35SelectedBaseSlotOffsets {
        slots: vec![
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: 16,
                up_byte_offset: 20,
                down_byte_offset: 24,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 0,
                gate_byte_offset: 0,
                up_byte_offset: 0,
                down_byte_offset: 0,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: 16,
                up_byte_offset: 20,
                down_byte_offset: 24,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: 8,
                up_byte_offset: 10,
                down_byte_offset: 12,
            },
        ],
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
    };
    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 3, 4096)
        .expect("selected-base device pointer plan");

    let stats = qwen35_selected_sparse_boundary_stats_from_device_ptr_plan(
        offsets.slots.len(),
        &plan,
        false,
    );

    assert_eq!(stats.slots, 4);
    assert_eq!(stats.unique_experts, 3);
    assert_eq!(stats.selected_upload_calls, 3);
    assert_eq!(stats.selected_upload_bytes, 3 * (8 + 10 + 12));
    assert_eq!(stats.route_h2d_bytes, 4 * std::mem::size_of::<f32>());
    assert_eq!(stats.token_h2d_bytes, 4 * std::mem::size_of::<u32>());
    assert_eq!(
        stats.device_slot_h2d_bytes,
        (4 + 3) * std::mem::size_of::<u32>()
    );
}

#[test]
fn qwen35_selected_sparse_boundary_stats_count_descriptor_calls_and_launches() {
    let offsets = Qwen35SelectedBaseSlotOffsets {
        slots: vec![
            Qwen35SelectedBaseSlotOffset {
                expert_id: 0,
                gate_byte_offset: 0,
                up_byte_offset: 0,
                down_byte_offset: 0,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: 8,
                up_byte_offset: 8,
                down_byte_offset: 8,
            },
        ],
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 8,
        down_bytes_per_expert: 8,
    };
    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 2, 4096)
        .expect("selected-base device pointer plan");
    let mut stats = qwen35_selected_sparse_boundary_stats_from_device_ptr_plan(
        offsets.slots.len(),
        &plan,
        false,
    );

    assert_eq!(stats.total_descriptor_h2d_calls(), 4);
    assert_eq!(stats.total_kernel_launches(), 0);

    stats.slot_ptr_build_launches = 1;
    stats.zero_launches = 1;
    stats.gate_up_launches = 1;
    stats.silu_launches = 1;
    stats.down_launches = 1;
    stats.group_meta_h2d_calls = 1;

    assert_eq!(stats.total_descriptor_h2d_calls(), 5);
    assert_eq!(stats.total_kernel_launches(), 5);
}

#[test]
fn qwen35_selected_sparse_boundary_stats_use_device_slot_upload_plan_counts() {
    let device = PreparedQwen35DeviceSlotPtrs {
        expert_ids: vec![2, 0, 1, 2],
        expert_slab_indices: vec![0, 1, 2],
        gate_base: 0x1000,
        up_base: 0x2000,
        down_base: 0x3000,
        gate_expert_bytes: 8,
        up_expert_bytes: 10,
        down_expert_bytes: 12,
        selected_upload_calls: 3,
        selected_upload_bytes: 90,
        mixed_expert_ptrs: None,
        group_meta2: Vec::new(),
        group_meta4: Vec::new(),
        group_meta8: Vec::new(),
        group_meta16: Vec::new(),
        group_meta32: Vec::new(),
        group_meta64: Vec::new(),
    };

    let stats = qwen35_selected_sparse_boundary_stats_from_device_slots(4, &device, false);

    assert_eq!(stats.unique_experts, 3);
    assert_eq!(stats.selected_upload_calls, 3);
    assert_eq!(stats.selected_upload_bytes, 90);
}

#[test]
fn qwen35_selected_sparse_execution_descriptor_preserves_promoted_q6_pack4_stack() {
    let descriptor =
        qwen35_selected_sparse_execution_descriptor(Qwen35SelectedSparseDescriptorInput {
            slots: 8,
            token_count: 4,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::DeviceCompact {
                selected_experts: 3,
            },
            gate_up_group: Some(8),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: true,
            q4_gate_up_silu_pack4_group8: true,
            q6_down_q8dot: false,
            q6_down_pack4_f32: true,
            q6_down_pack4_f32_vec4: true,
        })
        .expect("descriptor");

    assert_eq!(
        descriptor.route_source,
        Qwen35SelectedSparseRouteSource::Host
    );
    assert_eq!(
        descriptor.slot_pointer_source,
        Qwen35SelectedSparseSlotPointerSource::DeviceCompact {
            selected_experts: 3
        }
    );
    assert_eq!(
        descriptor.activation_layout,
        Qwen35SelectedSparseActivationLayout::Pack4F32Group8
    );
    assert_eq!(
        descriptor.down_runner,
        Qwen35SelectedSparseDownRunner::Q6Pack4F32 { vec4_load: true }
    );
    assert_eq!(
        descriptor.accumulation_order,
        Qwen35SelectedSparseAccumulationOrder::ExistingGroupedByExpertToken
    );
    assert_eq!(
        descriptor.launches,
        Qwen35SelectedSparseLaunchPlan {
            slot_ptr_build: 1,
            zero: 1,
            gate_up: 1,
            silu: 0,
            down: 1,
        }
    );
    assert_eq!(
        descriptor.h2d,
        Qwen35SelectedSparseDescriptorH2dPlan {
            route_bytes: 8 * std::mem::size_of::<f32>(),
            token_bytes: 8 * std::mem::size_of::<u32>(),
            slot_descriptor_bytes: (8 + 3) * std::mem::size_of::<u32>(),
            group_meta_calls: 3,
        }
    );
    assert!(descriptor.exact_reference);
}

#[test]
fn qwen35_selected_sparse_execution_descriptor_matches_boundary_stats() {
    let offsets = Qwen35SelectedBaseSlotOffsets {
        slots: vec![
            Qwen35SelectedBaseSlotOffset {
                expert_id: 0,
                gate_byte_offset: 0,
                up_byte_offset: 0,
                down_byte_offset: 0,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: 8,
                up_byte_offset: 8,
                down_byte_offset: 8,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: 8,
                up_byte_offset: 8,
                down_byte_offset: 8,
            },
        ],
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 8,
        down_bytes_per_expert: 8,
    };
    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 2, 4096)
        .expect("selected-base device pointer plan");
    let mut stats = qwen35_selected_sparse_boundary_stats_from_device_ptr_plan(
        offsets.slots.len(),
        &plan,
        false,
    );
    let descriptor =
        qwen35_selected_sparse_execution_descriptor(Qwen35SelectedSparseDescriptorInput {
            slots: offsets.slots.len(),
            token_count: 2,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::DeviceCompact {
                selected_experts: 2,
            },
            gate_up_group: Some(8),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: true,
            q4_gate_up_silu_pack4_group8: true,
            q6_down_q8dot: false,
            q6_down_pack4_f32: true,
            q6_down_pack4_f32_vec4: true,
        })
        .expect("descriptor");

    stats.apply_execution_descriptor(&descriptor);

    assert_eq!(stats.descriptor_h2d_calls, 4);
    assert_eq!(stats.group_meta_h2d_calls, 3);
    assert_eq!(stats.total_descriptor_h2d_calls(), 7);
    assert_eq!(stats.slot_ptr_build_launches, 1);
    assert_eq!(stats.zero_launches, 1);
    assert_eq!(stats.gate_up_launches, 1);
    assert_eq!(stats.silu_launches, 0);
    assert_eq!(stats.down_launches, 1);
    assert_eq!(stats.total_kernel_launches(), 4);
    assert_eq!(
        stats.route_h2d_bytes,
        offsets.slots.len() * std::mem::size_of::<f32>()
    );
    assert_eq!(
        stats.token_h2d_bytes,
        offsets.slots.len() * std::mem::size_of::<u32>()
    );
    assert_eq!(
        stats.device_slot_h2d_bytes,
        (offsets.slots.len() + 2) * std::mem::size_of::<u32>()
    );
}

#[test]
fn qwen35_selected_sparse_runtime_descriptor_masks_vec4_when_pack4_down_is_inactive() {
    let descriptor =
        qwen35_selected_sparse_runtime_descriptor(Qwen35SelectedSparseRuntimeDescriptorInput {
            slots: 4,
            token_count: 2,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::Host,
            gate_up_group: Some(4),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: false,
            q4_gate_up_silu_pack4_group8: false,
            q6_down_q8dot: false,
            q6_down_pack4_f32: false,
            q6_down_pack4_f32_vec4_enabled: true,
        })
        .expect("runtime descriptor should mask inactive vec4 flag");

    assert_eq!(
        descriptor.down_runner,
        Qwen35SelectedSparseDownRunner::Q6Existing
    );
    assert_eq!(
        descriptor.activation_layout,
        Qwen35SelectedSparseActivationLayout::SeparateSilu
    );
    assert_eq!(descriptor.launches.silu, 1);
}

#[test]
fn qwen35_selected_sparse_runner_mode_stays_opt_in_and_requires_exact_reference() {
    let descriptor =
        qwen35_selected_sparse_runtime_descriptor(Qwen35SelectedSparseRuntimeDescriptorInput {
            slots: 4,
            token_count: 2,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::Host,
            gate_up_group: Some(4),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: false,
            q4_gate_up_silu_pack4_group8: false,
            q6_down_q8dot: false,
            q6_down_pack4_f32: false,
            q6_down_pack4_f32_vec4_enabled: true,
        })
        .expect("runtime descriptor");

    assert_eq!(
        qwen35_selected_sparse_runner_mode(false, false, None).unwrap(),
        Qwen35SelectedSparseRunnerMode::LegacyInline
    );
    assert_eq!(
        qwen35_selected_sparse_runner_mode(false, false, Some(&descriptor)).unwrap(),
        Qwen35SelectedSparseRunnerMode::LegacyInline
    );
    assert_eq!(
        qwen35_selected_sparse_runner_mode(true, false, Some(&descriptor)).unwrap(),
        Qwen35SelectedSparseRunnerMode::ExactReference
    );

    let missing = qwen35_selected_sparse_runner_mode(true, false, None)
        .expect_err("ABI mode needs a descriptor");
    assert!(missing.contains("requires descriptor"));

    let mut non_exact = descriptor;
    non_exact.exact_reference = false;
    let err = qwen35_selected_sparse_runner_mode(true, false, Some(&non_exact))
        .expect_err("ABI mode rejects non-exact descriptor");
    assert!(err.contains("requires exact-reference"));
}

#[test]
fn qwen35_selected_sparse_compound_runner_mode_stays_opt_in() {
    let descriptor =
        qwen35_selected_sparse_runtime_descriptor(Qwen35SelectedSparseRuntimeDescriptorInput {
            slots: 8,
            token_count: 4,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::DeviceCompact {
                selected_experts: 3,
            },
            gate_up_group: Some(8),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: true,
            q4_gate_up_silu_pack4_group8: true,
            q6_down_q8dot: false,
            q6_down_pack4_f32: true,
            q6_down_pack4_f32_vec4_enabled: true,
        })
        .expect("promoted selected sparse descriptor");
    let fallback_descriptor =
        qwen35_selected_sparse_runtime_descriptor(Qwen35SelectedSparseRuntimeDescriptorInput {
            slots: 4,
            token_count: 2,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::Host,
            gate_up_group: Some(4),
            down_group: Some(4),
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused: false,
            q4_gate_up_q8dot: false,
            q4_gate_up_silu_pack4_f32: false,
            q4_gate_up_silu_pack4_group8: false,
            q6_down_q8dot: false,
            q6_down_pack4_f32: false,
            q6_down_pack4_f32_vec4_enabled: true,
        })
        .expect("fallback selected sparse descriptor");

    assert_eq!(
        qwen35_selected_sparse_runner_mode(false, false, None).unwrap(),
        Qwen35SelectedSparseRunnerMode::LegacyInline
    );
    assert_eq!(
        qwen35_selected_sparse_runner_mode(false, true, Some(&descriptor)).unwrap(),
        Qwen35SelectedSparseRunnerMode::CompoundExactReference
    );
    assert_eq!(
        qwen35_selected_sparse_runner_mode(true, true, Some(&descriptor)).unwrap(),
        Qwen35SelectedSparseRunnerMode::CompoundExactReference
    );
    assert_eq!(
        qwen35_selected_sparse_runner_mode(false, true, Some(&fallback_descriptor)).unwrap(),
        Qwen35SelectedSparseRunnerMode::ExactReference
    );

    let missing = qwen35_selected_sparse_runner_mode(false, true, None)
        .expect_err("compound runner mode needs a descriptor");
    assert!(missing.contains("requires descriptor"));
}

#[test]
fn qwen35_selected_base_mixed_resident_plan_skips_resident_role_uploads() {
    let offsets = Qwen35SelectedBaseSlotOffsets {
        slots: vec![
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: 16,
                up_byte_offset: 20,
                down_byte_offset: 24,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 0,
                gate_byte_offset: 0,
                up_byte_offset: 0,
                down_byte_offset: 0,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: 16,
                up_byte_offset: 20,
                down_byte_offset: 24,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: 8,
                up_byte_offset: 10,
                down_byte_offset: 12,
            },
        ],
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
    };
    let resident_roles = std::collections::HashSet::from([
        Qwen35SelectedBaseResidentRole {
            role: Qwen35SelectedBaseWeightRole::Gate,
            expert_id: 2,
        },
        Qwen35SelectedBaseResidentRole {
            role: Qwen35SelectedBaseWeightRole::Down,
            expert_id: 1,
        },
    ]);

    let plan = qwen35_selected_base_mixed_resident_temp_plan(&offsets, 3, &resident_roles).unwrap();

    assert_eq!(plan.selected_experts, 3);
    assert_eq!(plan.resident_upload_bytes_saved, 8 + 12);
    assert_eq!(plan.slab_bytes, 3 * (8 + 10 + 12) - (8 + 12));
    assert_eq!(plan.uploads.len(), 7);
    assert_eq!(
        plan.expert_sources[2].unwrap().gate,
        Qwen35SelectedBaseMixedWeightSource::Resident
    );
    assert_eq!(
        plan.expert_sources[1].unwrap().down,
        Qwen35SelectedBaseMixedWeightSource::Resident
    );
    assert_eq!(
        plan.expert_sources[0].unwrap().gate,
        Qwen35SelectedBaseMixedWeightSource::Temp {
            slab_byte_offset: 0
        }
    );
    assert!(plan.expert_sources.iter().all(|source| source.is_some()));
    assert!(plan
        .uploads
        .iter()
        .all(|upload| !(upload.role == Qwen35SelectedBaseWeightRole::Gate
            && upload.src_byte_offset == 16)));
    assert!(plan
        .uploads
        .iter()
        .all(|upload| !(upload.role == Qwen35SelectedBaseWeightRole::Down
            && upload.src_byte_offset == 12)));
}

#[test]
fn qwen35_down_token_major_plan_groups_slots_by_token() {
    let plan =
        qwen35_down_token_major_plan(&[2, 0, 2, 1, 0], 3).expect("token-major down slot plan");

    assert_eq!(plan.token_offsets, vec![0, 2, 3, 5]);
    assert_eq!(plan.slot_indices, vec![1, 4, 3, 0, 2]);
}

#[test]
fn qwen35_down_token_major_plan_rejects_bad_token_id() {
    let err = qwen35_down_token_major_plan(&[0, 2], 2).expect_err("token 2 out of range");

    assert!(err.contains("token id out of range"));
}

#[test]
fn qwen35_group_meta_split_by_len_preserves_group_order() {
    let split = qwen35_group_meta_split_by_len(&[0, 4, 4, 1, 5, 4, 9, 2, 11, 3, 14, 4], 4)
        .expect("split group meta by length");

    assert_eq!(split.matching, vec![0, 4, 5, 4, 14, 4]);
    assert_eq!(split.other, vec![4, 1, 9, 2, 11, 3]);
}

#[test]
fn qwen35_group_meta_split_by_len_rejects_odd_meta() {
    let err = qwen35_group_meta_split_by_len(&[0, 4, 8], 4).expect_err("odd group meta");

    assert!(err.contains("group meta must contain start/len pairs"));
}

#[test]
fn qwen35_route_arrays_sort_keeps_expert_weight_token_lockstep() {
    let mut expert_ids = vec![2, 1, 2, 1];
    let mut route_weights = vec![0.2, 0.7, 0.3, 0.6];
    let mut token_ids = vec![0, 1, 1, 0];

    qwen35_sort_route_arrays_by_expert_token(
        &mut expert_ids,
        &mut route_weights,
        &mut token_ids,
        2,
        3,
        2,
    )
    .unwrap();

    assert_eq!(expert_ids, vec![1, 1, 2, 2]);
    assert_eq!(token_ids, vec![0, 1, 0, 1]);
    assert_eq!(route_weights, vec![0.6, 0.7, 0.2, 0.3]);
}

#[test]
fn qwen35_route_arrays_reject_mismatched_lengths_before_cuda() {
    let err = qwen35_validate_route_arrays_for_test(&Qwen35RouteArrays {
        expert_ids: &[0, 1],
        route_weights: &[0.5],
        token_ids: &[0, 0],
        seq_len: 1,
        n_expert: 2,
        n_expert_used: 2,
    })
    .expect_err("route weight length differs from selected slot count");

    assert!(err.contains("route array length mismatch"));
}

#[test]
fn qwen35_route_arrays_reject_out_of_range_expert_or_token() {
    let expert_err = qwen35_validate_route_arrays_for_test(&Qwen35RouteArrays {
        expert_ids: &[0, 3],
        route_weights: &[0.5, 0.5],
        token_ids: &[0, 0],
        seq_len: 1,
        n_expert: 3,
        n_expert_used: 2,
    })
    .expect_err("expert id 3 is out of range for three experts");
    assert!(expert_err.contains("expert id out of range"));

    let token_err = qwen35_validate_route_arrays_for_test(&Qwen35RouteArrays {
        expert_ids: &[0, 1],
        route_weights: &[0.5, 0.5],
        token_ids: &[0, 1],
        seq_len: 1,
        n_expert: 3,
        n_expert_used: 2,
    })
    .expect_err("token id 1 is out of range for one token");
    assert!(token_err.contains("token id out of range"));
}

#[test]
fn qwen35_full_layer_slot_ptr_plan_uses_device_build_only_for_contiguous_bases() {
    assert_eq!(
        qwen35_full_layer_slot_ptr_plan(false, false),
        Qwen35FullLayerSlotPtrPlan::HostPointerUpload
    );
    assert_eq!(
        qwen35_full_layer_slot_ptr_plan(true, false),
        Qwen35FullLayerSlotPtrPlan::DevicePointerBuild
    );
    assert_eq!(
        qwen35_full_layer_slot_ptr_plan(true, true),
        Qwen35FullLayerSlotPtrPlan::HostPointerUpload
    );
}

#[test]
fn qwen35_selected_base_full_layer_inputs_use_qwen_byte_sizes() {
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = [vec![1u8; gate_per_expert], vec![2u8; gate_per_expert]].concat();
    let up = [vec![3u8; gate_per_expert], vec![4u8; gate_per_expert]].concat();
    let down = [vec![5u8; down_per_expert], vec![6u8; down_per_expert]].concat();

    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        &gate,
        &up,
        &down,
        &[1, 0],
        &[0.25, 0.75],
        &[0, 1],
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    assert_eq!(
        selected.gate_weights,
        vec![
            &gate[gate_per_expert..gate_per_expert * 2],
            &gate[0..gate_per_expert]
        ]
    );
    assert_eq!(
        selected.down_weights,
        vec![
            &down[down_per_expert..down_per_expert * 2],
            &down[0..down_per_expert]
        ]
    );
}

#[test]
fn qwen35_selected_base_slot_offsets_from_full_layer_match_selected_slices() {
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = [
        vec![1u8; gate_per_expert],
        vec![2u8; gate_per_expert],
        vec![3u8; gate_per_expert],
    ]
    .concat();
    let up = [
        vec![4u8; gate_per_expert],
        vec![5u8; gate_per_expert],
        vec![6u8; gate_per_expert],
    ]
    .concat();
    let down = [
        vec![7u8; down_per_expert],
        vec![8u8; down_per_expert],
        vec![9u8; down_per_expert],
    ]
    .concat();
    let expert_ids = [2u32, 0, 1, 2];

    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();
    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        &[0.25, 0.25, 0.25, 0.25],
        &[0, 0, 1, 1],
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    assert_eq!(
        offsets.slots,
        vec![
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: gate_per_expert * 2,
                up_byte_offset: gate_per_expert * 2,
                down_byte_offset: down_per_expert * 2,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 0,
                gate_byte_offset: 0,
                up_byte_offset: 0,
                down_byte_offset: 0,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 1,
                gate_byte_offset: gate_per_expert,
                up_byte_offset: gate_per_expert,
                down_byte_offset: down_per_expert,
            },
            Qwen35SelectedBaseSlotOffset {
                expert_id: 2,
                gate_byte_offset: gate_per_expert * 2,
                up_byte_offset: gate_per_expert * 2,
                down_byte_offset: down_per_expert * 2,
            },
        ]
    );
    assert_eq!(offsets.gate_bytes_per_expert, gate_per_expert);
    assert_eq!(offsets.up_bytes_per_expert, gate_per_expert);
    assert_eq!(offsets.down_bytes_per_expert, down_per_expert);

    for (slot_idx, slot) in offsets.slots.iter().enumerate() {
        assert_eq!(
            selected.gate_weights[slot_idx],
            &gate[slot.gate_byte_offset..slot.gate_byte_offset + offsets.gate_bytes_per_expert]
        );
        assert_eq!(
            selected.up_weights[slot_idx],
            &up[slot.up_byte_offset..slot.up_byte_offset + offsets.up_bytes_per_expert]
        );
        assert_eq!(
            selected.down_weights[slot_idx],
            &down[slot.down_byte_offset..slot.down_byte_offset + offsets.down_bytes_per_expert]
        );
    }
}

#[test]
fn qwen35_selected_base_temp_slab_plan_materializes_slot_ptrs_from_offsets() {
    let _guard = runtime_test_lock();
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "1");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = vec![1u8; gate_per_expert * 3];
    let up = vec![2u8; gate_per_expert * 3];
    let down = vec![3u8; down_per_expert * 3];
    let expert_ids = [2u32, 0, 1, 2];
    let slab_base = 0x1000_0000u64;
    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    let plan = qwen35_selected_base_temp_slab_slot_ptr_plan(&offsets, slab_base).unwrap();

    assert_eq!(
        plan.gate_ptrs,
        vec![
            slab_base,
            slab_base + gate_per_expert as u64,
            slab_base + (gate_per_expert * 2) as u64,
            slab_base,
        ]
    );
    assert_eq!(
        plan.up_ptrs,
        vec![
            slab_base + (gate_per_expert * 3) as u64,
            slab_base + (gate_per_expert * 4) as u64,
            slab_base + (gate_per_expert * 5) as u64,
            slab_base + (gate_per_expert * 3) as u64,
        ]
    );
    assert_eq!(
        plan.down_ptrs,
        vec![
            slab_base + (gate_per_expert * 6) as u64,
            slab_base + (gate_per_expert * 6 + down_per_expert) as u64,
            slab_base + (gate_per_expert * 6 + down_per_expert * 2) as u64,
            slab_base + (gate_per_expert * 6) as u64,
        ]
    );
    assert_eq!(plan.slab_bytes, gate_per_expert * 6 + down_per_expert * 3);
    assert_eq!(
        plan.uploads
            .iter()
            .map(|entry| (
                entry.role,
                entry.src_byte_offset,
                entry.slab_byte_offset,
                entry.bytes
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Qwen35SelectedBaseWeightRole::Gate,
                gate_per_expert * 2,
                0,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Gate,
                0,
                gate_per_expert,
                gate_per_expert * 2,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                gate_per_expert * 2,
                gate_per_expert * 3,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                0,
                gate_per_expert * 4,
                gate_per_expert * 2,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                down_per_expert * 2,
                gate_per_expert * 6,
                down_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                0,
                gate_per_expert * 6 + down_per_expert,
                down_per_expert * 2,
            ),
        ]
    );
}

#[test]
fn qwen35_selected_base_temp_slab_device_ptr_plan_maps_experts_to_compact_slots() {
    let _guard = runtime_test_lock();
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "1");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = vec![1u8; gate_per_expert * 4];
    let up = vec![2u8; gate_per_expert * 4];
    let down = vec![3u8; down_per_expert * 4];
    let expert_ids = [2u32, 0, 1, 2, 0];
    let slab_base = 0x2000_0000u64;
    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 4, slab_base).unwrap();

    assert_eq!(plan.expert_slab_indices, vec![0, 1, 2, u32::MAX]);
    assert_eq!(plan.gate_base, slab_base);
    assert_eq!(plan.up_base, slab_base + (gate_per_expert * 3) as u64);
    assert_eq!(plan.down_base, slab_base + (gate_per_expert * 6) as u64);
    assert_eq!(plan.slab_bytes, gate_per_expert * 6 + down_per_expert * 3);
    assert_eq!(
        plan.uploads
            .iter()
            .map(|entry| (
                entry.role,
                entry.src_byte_offset,
                entry.slab_byte_offset,
                entry.bytes
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Qwen35SelectedBaseWeightRole::Gate,
                0,
                0,
                gate_per_expert * 3,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                0,
                gate_per_expert * 3,
                gate_per_expert * 3,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                0,
                gate_per_expert * 6,
                down_per_expert * 3,
            ),
        ]
    );
}

#[test]
fn qwen35_selected_base_temp_slab_device_ptr_plan_coalesces_adjacent_expert_ranges_without_extra_bytes(
) {
    let _guard = runtime_test_lock();
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "1");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = vec![1u8; gate_per_expert * 6];
    let up = vec![2u8; gate_per_expert * 6];
    let down = vec![3u8; down_per_expert * 6];
    let expert_ids = [1u32, 2, 3, 5, 1, 2];
    let slab_base = 0x2400_0000u64;
    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 6, slab_base).unwrap();

    assert_eq!(
        plan.expert_slab_indices,
        vec![u32::MAX, 0, 1, 2, u32::MAX, 3]
    );
    assert_eq!(plan.slab_bytes, gate_per_expert * 8 + down_per_expert * 4);
    assert_eq!(
        plan.uploads
            .iter()
            .map(|entry| (
                entry.role,
                entry.src_byte_offset,
                entry.slab_byte_offset,
                entry.bytes
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Qwen35SelectedBaseWeightRole::Gate,
                gate_per_expert,
                0,
                gate_per_expert * 3,
            ),
            (
                Qwen35SelectedBaseWeightRole::Gate,
                gate_per_expert * 5,
                gate_per_expert * 3,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                gate_per_expert,
                gate_per_expert * 4,
                gate_per_expert * 3,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                gate_per_expert * 5,
                gate_per_expert * 7,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                down_per_expert,
                gate_per_expert * 8,
                down_per_expert * 3,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                down_per_expert * 5,
                gate_per_expert * 8 + down_per_expert * 3,
                down_per_expert,
            ),
        ]
    );
}

#[test]
fn qwen35_selected_base_temp_slab_device_ptr_plan_range_upload_can_opt_out() {
    let _guard = runtime_test_lock();
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "0");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 210;
    let gate = vec![1u8; gate_per_expert * 4];
    let up = vec![2u8; gate_per_expert * 4];
    let down = vec![3u8; down_per_expert * 4];
    let expert_ids = [2u32, 0, 1, 2, 0];
    let slab_base = 0x2000_0000u64;
    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();

    let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, 4, slab_base).unwrap();

    assert_eq!(plan.expert_slab_indices, vec![1, 2, 0, u32::MAX]);
    assert_eq!(
        plan.uploads
            .iter()
            .map(|entry| (
                entry.role,
                entry.src_byte_offset,
                entry.slab_byte_offset,
                entry.bytes
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Qwen35SelectedBaseWeightRole::Gate,
                gate_per_expert * 2,
                0,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Gate,
                0,
                gate_per_expert,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Gate,
                gate_per_expert,
                gate_per_expert * 2,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                gate_per_expert * 2,
                gate_per_expert * 3,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                0,
                gate_per_expert * 4,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Up,
                gate_per_expert,
                gate_per_expert * 5,
                gate_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                down_per_expert * 2,
                gate_per_expert * 6,
                down_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                0,
                gate_per_expert * 6 + down_per_expert,
                down_per_expert,
            ),
            (
                Qwen35SelectedBaseWeightRole::Down,
                down_per_expert,
                gate_per_expert * 6 + down_per_expert * 2,
                down_per_expert,
            ),
        ]
    );
}

#[test]
fn qwen35_selected_base_offset_group_meta_matches_slice_group_meta() {
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q4k_weights(4, n_ff, n_embd / 256, 251).concat();
    let up = make_test_q4k_weights(4, n_ff, n_embd / 256, 257).concat();
    let down = make_test_q6k_weights(4, n_embd, n_ff / 256, 263).concat();
    let expert_ids = [2u32, 2, 2, 1, 1, 3, 3, 3, 3, 0];
    let route_weights = vec![0.125f32; expert_ids.len()];
    let token_ids = (0..expert_ids.len() as u32).collect::<Vec<_>>();
    let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();
    let selected = qwen35_selected_base_sparse_inputs_from_full_layer(
        &gate,
        &up,
        &down,
        &expert_ids,
        &route_weights,
        &token_ids,
        14,
        n_ff,
        n_embd,
    )
    .unwrap();
    let prepared_meta = PreparedQwen35SparseGroupMeta {
        group_meta2: qwen35_selected_base_group_meta_from_offsets(&offsets, 2),
        group_meta4: qwen35_selected_base_group_meta_from_offsets(&offsets, 4),
        group_meta8: qwen35_selected_base_group_meta_from_offsets(&offsets, 8),
        group_meta16: qwen35_selected_base_group_meta_from_offsets(&offsets, 16),
        group_meta32: qwen35_selected_base_group_meta_from_offsets(&offsets, 32),
        group_meta64: qwen35_selected_base_group_meta_from_offsets(&offsets, 64),
    };

    for max_group in [2usize, 4, 8, 16, 32, 64] {
        let expected = build_group_meta(&selected.gate_weights, &selected.up_weights, max_group);
        assert_eq!(
            qwen35_selected_base_group_meta_from_offsets(&offsets, max_group),
            expected,
            "max_group={max_group}"
        );
        assert_eq!(
            prepared_meta.group_meta_for_max_group(max_group).unwrap(),
            expected.as_slice(),
            "prepared max_group={max_group}"
        );
    }
}

#[test]
fn qwen35_selected_base_temp_slab_ptrs_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_selected_base_temp_slab_ptrs_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_temp_slab_ptrs_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_temp_slab_ptrs_enabled());
}

#[test]
fn qwen35_selected_base_range_upload_stays_opt_in() {
    let _guard = runtime_test_lock();
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD";
    let _range = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_base_range_upload_enabled());

    let _range_off = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_range_upload_enabled());

    let _range_on = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_range_upload_enabled());
}

#[test]
fn qwen35_selected_base_device_slot_ptrs_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_selected_base_device_slot_ptrs_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_device_slot_ptrs_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_device_slot_ptrs_enabled());
}

#[test]
fn qwen35_selected_base_mixed_resident_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_base_mixed_resident_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_mixed_resident_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_mixed_resident_enabled());
}

#[test]
fn qwen35_selected_sparse_execution_abi_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_SPARSE_EXECUTION_ABI";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_sparse_execution_abi_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_sparse_execution_abi_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_sparse_execution_abi_enabled());
}

#[test]
fn qwen35_selected_sparse_compound_runner_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_RUNNER";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_selected_sparse_compound_runner_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_sparse_compound_runner_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_sparse_compound_runner_enabled());
}

#[test]
fn qwen35_selected_base_mixed_device_slot_ptrs_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_base_mixed_device_slot_ptrs_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_mixed_device_slot_ptrs_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_mixed_device_slot_ptrs_enabled());
}

#[test]
fn qwen35_selected_base_resident_admission_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_base_resident_admission_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_resident_admission_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_resident_admission_enabled());
}

#[test]
fn qwen35_selected_base_resident_admission_token_window_filters_prefill() {
    let _lock = runtime_test_lock();
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_selected_base_resident_admission_token_window_allows(1139).unwrap());

    let _decode_only = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_resident_admission_token_window_allows(1).unwrap());
    assert!(!qwen35_selected_base_resident_admission_token_window_allows(1139).unwrap());
    drop(_decode_only);

    let _bad = EnvVarGuard::set(key, "abc");
    assert!(
        qwen35_selected_base_resident_admission_token_window_allows(1)
            .unwrap_err()
            .contains("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS")
    );
}

#[test]
fn qwen35_selected_base_resident_admission_cost_gate_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_selected_base_resident_admission_cost_gate_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_resident_admission_cost_gate_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_resident_admission_cost_gate_enabled());
}

#[test]
fn qwen35_selected_base_resident_admission_history_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_HISTORY";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_base_resident_admission_history_enabled());

    let _on = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_base_resident_admission_history_enabled());
    drop(_on);

    let _off = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_base_resident_admission_history_enabled());
}

#[test]
fn qwen35_selected_base_resident_admission_future_hits_defaults_zero_and_parses() {
    let key = "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS";
    let _guard = EnvVarGuard::remove(key);
    assert_eq!(
        qwen35_selected_base_resident_admission_future_hits().unwrap(),
        0
    );

    let _two = EnvVarGuard::set(key, "2");
    assert_eq!(
        qwen35_selected_base_resident_admission_future_hits().unwrap(),
        2
    );
    drop(_two);

    let _bad = EnvVarGuard::set(key, "abc");
    assert!(qwen35_selected_base_resident_admission_future_hits()
        .unwrap_err()
        .contains("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS"));
}

#[test]
fn qwen35_resident_expert_page_admission_cost_gate_requires_explicit_future_hits() {
    let gate = [vec![1u8; 8], vec![2u8; 8]].concat();
    let up = [vec![3u8; 10], vec![4u8; 10]].concat();
    let down = [vec![5u8; 12], vec![6u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 2,
    };
    let repeated_once = qwen35_resident_expert_page_candidates(
        0,
        &bases,
        &[0, 1, 0, 1],
        &[0.9, 0.8, 0.7, 0.6],
        2,
        12,
        12,
        14,
    )
    .unwrap();
    let non_profitable = qwen35_resident_expert_page_plan(&repeated_once, u64::MAX);

    let non_profitable_cost =
        qwen35_resident_expert_page_admission_cost(&non_profitable.selected, &[false; 6], 0)
            .unwrap();

    assert_eq!(non_profitable_cost.new_admission_bytes, 60);
    assert_eq!(non_profitable_cost.predicted_saved_bytes, 0);
    assert!(!non_profitable_cost.profitable);

    let repeated_twice = qwen35_resident_expert_page_candidates(
        0,
        &bases,
        &[0, 1, 0, 1, 0, 1],
        &[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        3,
        12,
        12,
        14,
    )
    .unwrap();
    let profitable = qwen35_resident_expert_page_plan(&repeated_twice, u64::MAX);

    let no_future_cost =
        qwen35_resident_expert_page_admission_cost(&profitable.selected, &[false; 6], 0).unwrap();

    assert_eq!(no_future_cost.new_admission_bytes, 60);
    assert_eq!(no_future_cost.predicted_saved_bytes, 0);
    assert!(!no_future_cost.profitable);

    let profitable_cost =
        qwen35_resident_expert_page_admission_cost(&profitable.selected, &[false; 6], 2).unwrap();

    assert_eq!(profitable_cost.new_admission_bytes, 60);
    assert_eq!(profitable_cost.predicted_saved_bytes, 120);
    assert!(profitable_cost.profitable);
}

#[test]
fn qwen35_resident_expert_page_admission_cost_accepts_page_specific_future_hits() {
    let gate = [vec![1u8; 8], vec![2u8; 8]].concat();
    let up = [vec![3u8; 10], vec![4u8; 10]].concat();
    let down = [vec![5u8; 12], vec![6u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 2,
    };
    let candidates = qwen35_resident_expert_page_candidates(
        0,
        &bases,
        &[0, 1, 0, 1],
        &[0.9, 0.8, 0.7, 0.6],
        2,
        12,
        12,
        14,
    )
    .unwrap();
    let plan = qwen35_resident_expert_page_plan(&candidates, u64::MAX);
    let page_future_hits = plan
        .selected
        .iter()
        .map(|candidate| if candidate.expert_id == 0 { 3 } else { 0 })
        .collect::<Vec<_>>();

    let cost = qwen35_resident_expert_page_admission_cost_with_future_hits(
        &plan.selected,
        &[false; 6],
        &page_future_hits,
    )
    .unwrap();

    assert_eq!(cost.new_admission_bytes, 60);
    assert_eq!(cost.predicted_saved_bytes, 90);
    assert!(cost.profitable);
}

#[test]
fn qwen35_resident_expert_page_admission_cost_rejects_negative_eviction_net() {
    let gate = [vec![1u8; 8], vec![2u8; 8]].concat();
    let up = [vec![3u8; 10], vec![4u8; 10]].concat();
    let down = [vec![5u8; 12], vec![6u8; 12]].concat();
    let bases = Qwen35SelectedExpertBases {
        gate_all: &gate,
        up_all: &up,
        down_all: &down,
        gate_bytes_per_expert: 8,
        up_bytes_per_expert: 10,
        down_bytes_per_expert: 12,
        n_expert: 2,
    };
    let candidates = qwen35_resident_expert_page_candidates(
        0,
        &bases,
        &[0, 1, 0, 1],
        &[0.9, 0.8, 0.7, 0.6],
        2,
        12,
        12,
        14,
    )
    .unwrap();
    let plan = qwen35_resident_expert_page_plan(&candidates, u64::MAX);
    let future_hits = vec![2; plan.selected.len()];

    let rejected = qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
        &plan.selected,
        &[false; 6],
        &future_hits,
        80,
    )
    .unwrap();

    assert_eq!(rejected.new_admission_bytes, 60);
    assert_eq!(rejected.eviction_cost_bytes, 80);
    assert_eq!(rejected.predicted_saved_bytes, 120);
    assert_eq!(rejected.net_saved_bytes, -20);
    assert!(!rejected.profitable);

    let accepted = qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
        &plan.selected,
        &[false; 6],
        &future_hits,
        30,
    )
    .unwrap();

    assert_eq!(accepted.new_admission_bytes, 60);
    assert_eq!(accepted.eviction_cost_bytes, 30);
    assert_eq!(accepted.predicted_saved_bytes, 120);
    assert_eq!(accepted.net_saved_bytes, 30);
    assert!(accepted.profitable);
}

#[test]
fn qwen35_resident_expert_page_future_hits_are_layer_role_specific() {
    let selected = vec![
        Qwen35ResidentExpertPageCandidate {
            layer_idx: 3,
            expert_id: 0,
            role: Qwen35ResidentExpertPageRole::Gate,
            quant: 12,
            byte_offset: 0,
            bytes: 8,
            reuse_count: 1,
            route_weight_sum: 0.5,
            window_tokens: 1,
        },
        Qwen35ResidentExpertPageCandidate {
            layer_idx: 3,
            expert_id: 0,
            role: Qwen35ResidentExpertPageRole::Down,
            quant: 14,
            byte_offset: 0,
            bytes: 12,
            reuse_count: 1,
            route_weight_sum: 0.5,
            window_tokens: 1,
        },
        Qwen35ResidentExpertPageCandidate {
            layer_idx: 3,
            expert_id: 1,
            role: Qwen35ResidentExpertPageRole::Gate,
            quant: 12,
            byte_offset: 8,
            bytes: 8,
            reuse_count: 1,
            route_weight_sum: 0.4,
            window_tokens: 1,
        },
    ];
    let future_same_layer = vec![
        Qwen35ResidentExpertPageCandidate {
            layer_idx: 3,
            expert_id: 0,
            role: Qwen35ResidentExpertPageRole::Gate,
            quant: 12,
            byte_offset: 0,
            bytes: 8,
            reuse_count: 1,
            route_weight_sum: 0.6,
            window_tokens: 1,
        },
        Qwen35ResidentExpertPageCandidate {
            layer_idx: 3,
            expert_id: 0,
            role: Qwen35ResidentExpertPageRole::Down,
            quant: 14,
            byte_offset: 0,
            bytes: 12,
            reuse_count: 1,
            route_weight_sum: 0.6,
            window_tokens: 1,
        },
    ];
    let future_other_layer = vec![Qwen35ResidentExpertPageCandidate {
        layer_idx: 4,
        expert_id: 1,
        role: Qwen35ResidentExpertPageRole::Gate,
        quant: 12,
        byte_offset: 8,
        bytes: 8,
        reuse_count: 1,
        route_weight_sum: 0.9,
        window_tokens: 1,
    }];
    let future_repeated_same_window = vec![
        selected[0].clone(),
        selected[0].clone(),
        selected[2].clone(),
    ];

    let hits = qwen35_resident_expert_page_future_hit_counts(
        &selected,
        &[
            &future_same_layer,
            &future_other_layer,
            &future_repeated_same_window,
        ],
    );

    assert_eq!(hits, vec![2, 1, 1]);
}

#[test]
fn qwen35_resident_expert_page_source_history_delays_current_window_hits() {
    let mut history = std::collections::HashMap::new();
    let keys = vec![(0x1000usize, 8usize), (0x2000, 10), (0x1000, 8)];

    let first_hits = qwen35_resident_expert_page_source_future_hits_and_observe(
        &mut history,
        keys.iter().copied(),
    );

    assert_eq!(first_hits, vec![0, 0, 0]);
    assert_eq!(history.get(&(0x1000, 8)), Some(&1));
    assert_eq!(history.get(&(0x2000, 10)), Some(&1));

    let second_hits = qwen35_resident_expert_page_source_future_hits_and_observe(
        &mut history,
        keys.iter().copied(),
    );

    assert_eq!(second_hits, vec![1, 1, 1]);
    assert_eq!(history.get(&(0x1000, 8)), Some(&2));
    assert_eq!(history.get(&(0x2000, 10)), Some(&2));
}

#[test]
fn qwen35_selected_sparse_fused_boundary_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_SELECTED_SPARSE_FUSED_BOUNDARY";
    let _guard = EnvVarGuard::remove(key);
    assert!(!qwen35_selected_sparse_fused_boundary_enabled());

    let _zero = EnvVarGuard::set(key, "0");
    assert!(!qwen35_selected_sparse_fused_boundary_enabled());
    drop(_zero);

    let _one = EnvVarGuard::set(key, "1");
    assert!(qwen35_selected_sparse_fused_boundary_enabled());
}

#[test]
fn qwen35_sparse_slot_count_uses_prepared_count_after_empty_weight_slices() {
    assert_eq!(qwen35_sparse_slot_count(7, None, None), 7);
    assert_eq!(qwen35_sparse_slot_count(0, None, Some(4)), 4);
    assert_eq!(qwen35_sparse_slot_count(0, Some(9), Some(4)), 9);
}

#[test]
fn cuda_qwen35_selected_base_resident_admission_feeds_mixed_temp_plan() {
    let _guard = runtime_test_lock();
    let _admit = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _cost_gate = EnvVarGuard::set(
        "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE",
        "0",
    );
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];
    let route_weights = [0.9f32, 0.8, 0.7, 0.6];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base resident admission CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_q4k_limit = usize::MAX;

    let stats = state
        .qwen35_admit_selected_base_resident_pages_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route_weights,
            12,
            n_ff,
            n_embd,
            2,
        )
        .expect("selected-base resident admission");
    assert_eq!(stats.selected_pages, 6);
    assert_eq!(stats.admitted_pages, 6);
    assert_eq!(state.resident_q4k.len(), 6);

    let prepared = state
        .qwen35_prepare_selected_base_mixed_resident_temp_slots_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("mixed resident prepared slots");
    assert!(prepared.temp_slab_ptrs.is_empty());
    assert_eq!(prepared.gate_ptrs.len(), expert_ids.len());
    assert_eq!(prepared.up_ptrs.len(), expert_ids.len());
    assert_eq!(prepared.down_ptrs.len(), expert_ids.len());
}

#[test]
fn cuda_qwen35_selected_base_resident_admission_cost_gate_skips_nonprofitable_uploads() {
    let _guard = runtime_test_lock();
    let _admit = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _cost_gate =
        EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];
    let route_weights = [0.9f32, 0.8, 0.7, 0.6];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base resident admission cost-gate CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_q4k_limit = usize::MAX;

    let stats = state
        .qwen35_admit_selected_base_resident_pages_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route_weights,
            12,
            n_ff,
            n_embd,
            2,
        )
        .expect("selected-base resident admission");

    assert!(stats.skipped_by_cost_gate);
    assert_eq!(stats.selected_pages, 6);
    assert_eq!(stats.admitted_pages, 0);
    assert_eq!(stats.already_resident_pages, 0);
    assert_eq!(stats.admission_cost_bytes, stats.selected_bytes);
    assert_eq!(stats.predicted_saved_bytes, 0);
    assert_eq!(state.resident_q4k.len(), 0);
}

#[test]
fn cuda_qwen35_selected_base_full_layer_resident_hit_skips_selected_uploads() {
    let _guard = runtime_test_lock();
    let _admission = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _admission_window = EnvVarGuard::set(
        "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS",
        "1",
    );
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let expert_ids = [1u32, 0, 1, 0];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base full-layer resident CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_moe_layer_limit = usize::MAX;

    for (down_quant, down_all) in [
        (
            12u32,
            make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 349).concat(),
        ),
        (
            14u32,
            make_test_q6k_weights(n_expert, n_embd, n_ff / 256, 353).concat(),
        ),
    ] {
        let gate_all = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
        let up_all = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 347).concat();
        assert!(
            state
                .register_qwen35_moe_layer_without_eviction(
                    &gate_all, &up_all, &down_all, down_quant, n_ff, n_embd,
                )
                .expect("register full-layer resident weights"),
            "full-layer resident weights should be newly registered"
        );
        let admission = state
            .qwen35_admit_selected_base_resident_pages_by_token(
                &gate_all,
                &up_all,
                &down_all,
                &expert_ids,
                &[0.9, 0.8, 0.7, 0.6],
                down_quant,
                n_ff,
                n_embd,
                1,
            )
            .expect("full-layer hit resident-page admission");
        assert_eq!(admission.candidate_pages, 0);
        assert_eq!(admission.admitted_pages, 0);
        assert!(state.resident_q4k.is_empty());

        let key = qwen35_moe_layer_key(&gate_all, &up_all, &down_all, down_quant, n_ff, n_embd);
        let (gate_base, up_base, down_base) = state
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.gate_base, entry.up_base, entry.down_base))
            .expect("registered full-layer resident entry");
        let prepared = state
            .qwen35_prepare_selected_base_residency_aware_device_slot_ptrs_by_token(
                &gate_all,
                &up_all,
                &down_all,
                &expert_ids,
                down_quant,
                n_ff,
                n_embd,
            )
            .expect("residency-aware selected-base device slot pointers");

        assert!(prepared.temp_slab_ptrs.is_empty());
        let device = prepared
            .device_slot_ptrs
            .as_ref()
            .expect("device slot metadata");
        assert_eq!(device.selected_upload_calls, 0);
        assert_eq!(device.selected_upload_bytes, 0);
        let mixed = device
            .mixed_expert_ptrs
            .as_ref()
            .expect("resident mixed expert pointers");
        for expert in 0..n_expert {
            assert_eq!(
                mixed.gate_ptrs[expert],
                gate_base + expert as u64 * device.gate_expert_bytes as u64
            );
            assert_eq!(
                mixed.up_ptrs[expert],
                up_base + expert as u64 * device.up_expert_bytes as u64
            );
            assert_eq!(
                mixed.down_ptrs[expert],
                down_base + expert as u64 * device.down_expert_bytes as u64
            );
        }
        state
            .clear_resident_moe_layer_cache()
            .expect("clear full-layer resident cache between quant cases");
    }
}

#[test]
fn cuda_qwen35_selected_base_resident_admission_history_reduces_repeated_upload_bytes() {
    let _guard = runtime_test_lock();
    let _admit = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _history = EnvVarGuard::set(
        "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_HISTORY",
        "1",
    );
    let _cost_gate =
        EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE");
    let _future_hits =
        EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS");
    let _decode_only = EnvVarGuard::set(
        "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS",
        "1",
    );
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];
    let route_weights = [0.9f32, 0.8, 0.7, 0.6];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base resident admission history CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_q4k_limit = usize::MAX;

    let expert_bytes = n_ff * (n_embd / 256) * 144;
    let selected_bytes = (n_expert * 3 * expert_bytes) as u64;
    let before = state
        .qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("mixed resident device slot ptrs before admission");
    let before_device = before
        .device_slot_ptrs
        .as_ref()
        .expect("device slot metadata before admission");
    assert_eq!(before_device.selected_upload_calls, n_expert * 3);
    assert_eq!(before_device.selected_upload_bytes as u64, selected_bytes);

    let first = state
        .qwen35_admit_selected_base_resident_pages_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route_weights,
            12,
            n_ff,
            n_embd,
            1,
        )
        .expect("first selected-base resident admission");
    assert!(first.skipped_by_cost_gate);
    assert_eq!(first.predicted_saved_bytes, 0);
    assert_eq!(first.admitted_pages, 0);

    let second = state
        .qwen35_admit_selected_base_resident_pages_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route_weights,
            12,
            n_ff,
            n_embd,
            1,
        )
        .expect("second selected-base resident admission");
    assert!(second.skipped_by_cost_gate);
    assert_eq!(second.predicted_saved_bytes, selected_bytes);
    assert_eq!(second.admitted_pages, 0);

    let third = state
        .qwen35_admit_selected_base_resident_pages_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route_weights,
            12,
            n_ff,
            n_embd,
            1,
        )
        .expect("third selected-base resident admission");
    assert!(!third.skipped_by_cost_gate);
    assert_eq!(third.predicted_saved_bytes, selected_bytes * 2);
    assert_eq!(third.admission_cost_bytes, selected_bytes);
    assert_eq!(third.admitted_pages, n_expert * 3);
    assert_eq!(state.resident_q4k.len(), n_expert * 3);

    let after = state
        .qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("mixed resident device slot ptrs after admission");
    assert!(after.temp_slab_ptrs.is_empty());
    let after_device = after
        .device_slot_ptrs
        .as_ref()
        .expect("device slot metadata after admission");
    assert_eq!(after_device.selected_upload_calls, 0);
    assert_eq!(after_device.selected_upload_bytes, 0);
}

#[test]
fn cuda_qwen35_selected_base_temp_slab_cache_reuses_repeated_expert_set_upload_bytes() {
    let _guard = runtime_test_lock();
    let _cache = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_CACHE", "1");
    let _range = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD", "1");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 3usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 431).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 433).concat();
    let sparse_down = make_test_q6k_weights(n_expert, n_embd, n_ff / 256, 439).concat();
    let expert_ids = [2u32, 0, 1, 2, 0];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base temp-slab cache CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };

    let first = state
        .qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            14,
            n_ff,
            n_embd,
        )
        .expect("first selected-base temp-slab prepare");
    let first_device = first
        .device_slot_ptrs
        .as_ref()
        .expect("first device slot metadata");
    assert!(first_device.selected_upload_calls > 0);
    assert!(first_device.selected_upload_bytes > 0);

    let second = state
        .qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            14,
            n_ff,
            n_embd,
        )
        .expect("second selected-base temp-slab prepare");
    let second_device = second
        .device_slot_ptrs
        .as_ref()
        .expect("second device slot metadata");
    assert_eq!(second_device.selected_upload_calls, 0);
    assert_eq!(second_device.selected_upload_bytes, 0);
    assert_eq!(
        second_device.expert_slab_indices,
        first_device.expert_slab_indices
    );
    assert_eq!(second_device.gate_base, first_device.gate_base);
    assert_eq!(second_device.up_base, first_device.up_base);
    assert_eq!(second_device.down_base, first_device.down_base);
}

#[test]
fn cuda_qwen35_selected_base_resident_admission_accounts_eviction_cost() {
    let _guard = runtime_test_lock();
    let _admit = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _history = EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_HISTORY");
    let _cost_gate =
        EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE");
    let _max_tokens =
        EnvVarGuard::remove("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS");
    let _cache_mb = EnvVarGuard::set("RNB_CUDA_Q4K_CACHE_MB", "1");
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];
    let route_weights = [0.9f32, 0.8, 0.7, 0.6];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping selected-base resident eviction-cost CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };

    let expert_bytes = n_ff * (n_embd / 256) * 144;
    let selected_bytes = (n_expert * 3 * expert_bytes) as u64;
    state.resident_q4k_limit = selected_bytes as usize;
    let unrelated = (0..(n_expert * 3))
        .map(|idx| vec![0x80u8 + idx as u8; expert_bytes])
        .collect::<Vec<_>>();
    for weights in &unrelated {
        assert!(state
            .preload_resident_q4k_weight_slice(weights)
            .expect("preload unrelated resident page"));
    }
    assert_eq!(state.resident_q4k_bytes, selected_bytes as usize);
    assert_eq!(state.resident_q4k.len(), n_expert * 3);

    {
        let _future_hits = EnvVarGuard::set(
            "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS",
            "2",
        );
        let rejected = state
            .qwen35_admit_selected_base_resident_pages_by_token(
                &sparse_gate,
                &sparse_up,
                &sparse_down,
                &expert_ids,
                &route_weights,
                12,
                n_ff,
                n_embd,
                2,
            )
            .expect("selected-base resident admission with zero net");

        assert!(rejected.skipped_by_cost_gate);
        assert_eq!(rejected.admission_cost_bytes, selected_bytes);
        assert_eq!(rejected.eviction_cost_bytes, selected_bytes);
        assert_eq!(rejected.predicted_saved_bytes, selected_bytes * 2);
        assert_eq!(rejected.net_saved_bytes, 0);
        assert_eq!(rejected.admitted_pages, 0);
        assert_eq!(state.resident_q4k.len(), n_expert * 3);
    }

    {
        let _future_hits = EnvVarGuard::set(
            "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS",
            "3",
        );
        let accepted = state
            .qwen35_admit_selected_base_resident_pages_by_token(
                &sparse_gate,
                &sparse_up,
                &sparse_down,
                &expert_ids,
                &route_weights,
                12,
                n_ff,
                n_embd,
                2,
            )
            .expect("selected-base resident admission with positive net");

        assert!(!accepted.skipped_by_cost_gate);
        assert_eq!(accepted.admission_cost_bytes, selected_bytes);
        assert_eq!(accepted.eviction_cost_bytes, selected_bytes);
        assert_eq!(accepted.predicted_saved_bytes, selected_bytes * 3);
        assert_eq!(accepted.net_saved_bytes, selected_bytes as i128);
        assert_eq!(accepted.admitted_pages, n_expert * 3);
        assert_eq!(state.resident_q4k.len(), n_expert * 3);
    }

    let prepared = state
        .qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("mixed resident device slot ptrs after eviction-cost admission");
    assert!(prepared.temp_slab_ptrs.is_empty());
    let device = prepared
        .device_slot_ptrs
        .as_ref()
        .expect("device slot metadata after eviction-cost admission");
    assert_eq!(device.selected_upload_calls, 0);
    assert_eq!(device.selected_upload_bytes, 0);
}

#[test]
fn cuda_qwen35_mixed_resident_device_slot_ptrs_skip_resident_uploads() {
    let _guard = runtime_test_lock();
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mixed resident device-slot CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_q4k_limit = usize::MAX;
    let expert_bytes = n_ff * (n_embd / 256) * 144;
    let gate0 = &sparse_gate[..expert_bytes];
    assert!(state
        .preload_resident_q4k_weight_slice(gate0)
        .expect("preload resident gate expert 0"));

    let prepared = state
        .qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("mixed resident device slot ptrs");

    assert!(prepared.gate_ptrs.is_empty());
    assert!(prepared.up_ptrs.is_empty());
    assert!(prepared.down_ptrs.is_empty());
    let device = prepared
        .device_slot_ptrs
        .as_ref()
        .expect("mixed resident device slot ptr metadata");
    assert!(device.mixed_expert_ptrs.is_some());
    assert_eq!(device.selected_upload_calls, 5);
    assert_eq!(
        device.selected_upload_bytes,
        (n_expert * 3 - 1) * expert_bytes
    );
}

#[test]
fn cuda_qwen35_residency_aware_prefill_ignores_partial_q4_residency() {
    let _guard = runtime_test_lock();
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping partial-resident selected-base CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_q4k_limit = usize::MAX;
    let expert_bytes = n_ff * (n_embd / 256) * 144;
    let gate0 = &sparse_gate[..expert_bytes];
    assert!(state
        .preload_resident_q4k_weight_slice(gate0)
        .expect("preload resident gate expert 0"));

    let prepared = state
        .qwen35_prepare_selected_base_residency_aware_device_slot_ptrs_by_token(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            12,
            n_ff,
            n_embd,
        )
        .expect("residency-aware selected-base device slot ptrs");

    assert_eq!(prepared.temp_slab_ptrs.len(), 1);
    let device = prepared
        .device_slot_ptrs
        .as_ref()
        .expect("selected-base device slot metadata");
    assert!(
        device.mixed_expert_ptrs.is_none(),
        "partial resident pages must not switch a later request to the mixed execution path"
    );
    assert_eq!(device.selected_upload_calls, n_expert * 3);
}

#[test]
fn cuda_qwen35_compact_slot_ptr_builder_matches_host_pointers() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping Qwen35 compact slot ptr CUDA test: {err}");
            return;
        }
    };
    let expert_ids = [2u32, 0, 1, 2, 0];
    let expert_slab_indices = [1u32, 2, 0, u32::MAX];
    let slots = expert_ids.len();
    let gate_base = 0x1000_0000u64;
    let up_base = 0x2000_0000u64;
    let down_base = 0x3000_0000u64;
    let gate_expert_bytes = 144usize * 256;
    let up_expert_bytes = 160usize * 256;
    let down_expert_bytes = 210usize * 256;
    let ptr_bytes = slots * std::mem::size_of::<u64>();
    let ids_bytes = std::mem::size_of_val(expert_ids.as_slice());
    let map_bytes = std::mem::size_of_val(expert_slab_indices.as_slice());

    let gate_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("gate ptrs dev") };
    let up_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("up ptrs dev") };
    let down_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("down ptrs dev") };
    let expert_ids_dev = unsafe { state.api.mem_alloc(ids_bytes).expect("expert ids dev") };
    let expert_slab_indices_dev =
        unsafe { state.api.mem_alloc(map_bytes).expect("expert slab map dev") };

    let run = (|| -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
        unsafe {
            state.api.memcpy_htod_async(
                expert_ids_dev,
                expert_ids.as_ptr().cast::<libc::c_void>(),
                ids_bytes,
                state.stream,
            )?;
            state.api.memcpy_htod_async(
                expert_slab_indices_dev,
                expert_slab_indices.as_ptr().cast::<libc::c_void>(),
                map_bytes,
                state.stream,
            )?;
        }
        state.launch_qwen35_build_q4k_compact_slot_ptrs(
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            expert_ids_dev,
            expert_slab_indices_dev,
            gate_base,
            up_base,
            down_base,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            slots,
        )?;
        let mut gate_ptrs = vec![0u64; slots];
        let mut up_ptrs = vec![0u64; slots];
        let mut down_ptrs = vec![0u64; slots];
        unsafe {
            state.api.memcpy_dtoh_async(
                gate_ptrs.as_mut_ptr().cast::<libc::c_void>(),
                gate_ptrs_dev,
                ptr_bytes,
                state.stream,
            )?;
            state.api.memcpy_dtoh_async(
                up_ptrs.as_mut_ptr().cast::<libc::c_void>(),
                up_ptrs_dev,
                ptr_bytes,
                state.stream,
            )?;
            state.api.memcpy_dtoh_async(
                down_ptrs.as_mut_ptr().cast::<libc::c_void>(),
                down_ptrs_dev,
                ptr_bytes,
                state.stream,
            )?;
        }
        state.stream_synchronize()?;
        Ok((gate_ptrs, up_ptrs, down_ptrs))
    })();

    unsafe {
        state.api.mem_free(gate_ptrs_dev).expect("free gate ptrs");
        state.api.mem_free(up_ptrs_dev).expect("free up ptrs");
        state.api.mem_free(down_ptrs_dev).expect("free down ptrs");
        state.api.mem_free(expert_ids_dev).expect("free expert ids");
        state
            .api
            .mem_free(expert_slab_indices_dev)
            .expect("free expert slab map");
    }

    let (gate_ptrs, up_ptrs, down_ptrs) = run.expect("compact slot ptr build");
    let expected_compact = [0u32, 1, 2, 0, 1];
    assert_eq!(
        gate_ptrs,
        expected_compact
            .iter()
            .map(|&idx| gate_base + idx as u64 * gate_expert_bytes as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        up_ptrs,
        expected_compact
            .iter()
            .map(|&idx| up_base + idx as u64 * up_expert_bytes as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        down_ptrs,
        expected_compact
            .iter()
            .map(|&idx| down_base + idx as u64 * down_expert_bytes as u64)
            .collect::<Vec<_>>()
    );
}

#[test]
fn cuda_qwen35_full_layer_slot_ptr_builder_emits_pair_map() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping Qwen35 full-layer pair map CUDA test: {err}");
            return;
        }
    };
    const INVALID_SLOT: u32 = u32::MAX;
    const SKIP_SLOT: u32 = u32::MAX - 1;
    let expert_ids = [2u32, 4, 6, 8, 10, 12, 14, 16, 6, 9, 10, 11, 2, 13, 15, 17];
    let expected = [
        12u32,
        INVALID_SLOT,
        8,
        INVALID_SLOT,
        10,
        INVALID_SLOT,
        INVALID_SLOT,
        INVALID_SLOT,
        SKIP_SLOT,
        INVALID_SLOT,
        SKIP_SLOT,
        INVALID_SLOT,
        SKIP_SLOT,
        INVALID_SLOT,
        INVALID_SLOT,
        INVALID_SLOT,
    ];
    let slots = expert_ids.len();
    let ptr_bytes = slots * std::mem::size_of::<u64>();
    let ids_bytes = std::mem::size_of_val(expert_ids.as_slice());
    let pair_bytes = std::mem::size_of_val(expected.as_slice());
    let gate_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("gate ptrs dev") };
    let up_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("up ptrs dev") };
    let down_ptrs_dev = unsafe { state.api.mem_alloc(ptr_bytes).expect("down ptrs dev") };
    let expert_ids_dev = unsafe { state.api.mem_alloc(ids_bytes).expect("expert ids dev") };
    let pair_slots_dev = unsafe { state.api.mem_alloc(pair_bytes).expect("pair slots dev") };

    let run = (|| -> Result<Vec<u32>, String> {
        unsafe {
            state.api.memcpy_htod_async(
                expert_ids_dev,
                expert_ids.as_ptr().cast::<libc::c_void>(),
                ids_bytes,
                state.stream,
            )?;
        }
        state.launch_qwen35_build_q4k_full_layer_slot_ptrs(
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            expert_ids_dev,
            pair_slots_dev,
            slots / 2,
            0x1000_0000,
            0x2000_0000,
            0x3000_0000,
            144 * 256,
            176 * 256,
            slots,
        )?;
        let mut actual = vec![0u32; slots];
        unsafe {
            state.api.memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                pair_slots_dev,
                pair_bytes,
                state.stream,
            )?;
        }
        state.stream_synchronize()?;
        Ok(actual)
    })();

    unsafe {
        state.api.mem_free(gate_ptrs_dev).expect("free gate ptrs");
        state.api.mem_free(up_ptrs_dev).expect("free up ptrs");
        state.api.mem_free(down_ptrs_dev).expect("free down ptrs");
        state.api.mem_free(expert_ids_dev).expect("free expert ids");
        state.api.mem_free(pair_slots_dev).expect("free pair slots");
    }

    assert_eq!(run.expect("full-layer pair map build"), expected);
}

#[test]
fn qwen35_down_token_major_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_down_token_major_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_down_token_major_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_down_token_major_enabled());
}

#[test]
fn qwen35_q6_down_full4_split_stays_opt_in() {
    let _env_lock = cuda_test_env_lock();
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_full4_split_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_full4_split_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_full4_split_enabled());
}

#[test]
fn qwen35_q6_down_full4_fastpath_stays_opt_in() {
    let _env_lock = cuda_test_env_lock();
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_full4_fastpath_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_full4_fastpath_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_full4_fastpath_enabled());
}

#[test]
fn qwen35_q6_down_q8dot_stays_opt_in() {
    let _env_lock = cuda_test_env_lock();
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_q8dot_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_q8dot_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_q8dot_enabled());
}

#[test]
fn qwen35_q6_down_run_batched_ref_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_run_batched_ref_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_run_batched_ref_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_run_batched_ref_enabled());
}

#[test]
fn qwen35_q6_down_run_batched8_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_run_batched8_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_run_batched8_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_run_batched8_enabled());
}

#[test]
fn qwen35_q6_down_run_tiled4_stays_opt_in() {
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4";
    let _guard = EnvVarGuard::remove(key);
    assert!(!crate::tuning::qwen35_q6_down_run_tiled4_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_run_tiled4_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_run_tiled4_enabled());
}

#[test]
fn qwen35_q6_down_pack4_f32_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32";
    let _guard = EnvVarGuard::remove(key);
    assert!(crate::tuning::qwen35_q6_down_pack4_f32_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_pack4_f32_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_pack4_f32_enabled());
}

#[test]
fn qwen35_q6_down_pack4_f32_vec4_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32_VEC4";
    let _guard = EnvVarGuard::remove(key);
    assert!(crate::tuning::qwen35_q6_down_pack4_f32_vec4_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q6_down_pack4_f32_vec4_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q6_down_pack4_f32_vec4_enabled());
}

#[test]
fn qwen35_q4_gate_up_silu_pack4_f32_defaults_on_and_allows_opt_out() {
    let key = "RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32";
    let _guard = EnvVarGuard::remove(key);
    assert!(crate::tuning::qwen35_q4_gate_up_silu_pack4_f32_enabled());

    let _off = EnvVarGuard::set(key, "0");
    assert!(!crate::tuning::qwen35_q4_gate_up_silu_pack4_f32_enabled());
    drop(_off);

    let _on = EnvVarGuard::set(key, "1");
    assert!(crate::tuning::qwen35_q4_gate_up_silu_pack4_f32_enabled());
}

#[test]
fn qwen35_device_topk_selected_base_rejects_router_expert_count_mismatch_before_cuda() {
    let n_ff = 256usize;
    let n_embd = 256usize;
    let n_expert = 2usize;
    let gate_per_expert = n_ff * (n_embd / 256) * 144;
    let down_per_expert = n_embd * (n_ff / 256) * 144;
    let shared_gate = vec![0.0f32; n_ff * n_embd];
    let shared_up = vec![0.0f32; n_ff * n_embd];
    let shared_down = vec![0.0f32; n_embd * n_ff];
    let shared_route = vec![1.0f32];
    let gate = vec![1u8; gate_per_expert * n_expert];
    let up = vec![2u8; gate_per_expert * n_expert];
    let down = vec![3u8; down_per_expert * n_expert];
    let router = vec![0.0f32; n_embd];
    let input = vec![0.0f32; n_embd];

    let err = qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &gate,
        &up,
        &down,
        &router,
        1,
        n_embd,
        &input,
        1,
        1,
        12,
        n_ff,
        n_embd,
    )
    .expect_err("router expert count must match selected-base full-layer weights");

    assert!(err.contains("expert count mismatch"));
}

#[test]
fn nemotron_workspace_layout_aligns_roles() {
    use crate::runtime::types::{NemotronPrefillWorkspaceLayout, NemotronWorkspaceSlice};

    fn assert_non_overlapping(prev: NemotronWorkspaceSlice, next: NemotronWorkspaceSlice) {
        assert!(
            prev.offset + prev.bytes <= next.offset,
            "workspace slices overlap: prev={prev:?}, next={next:?}"
        );
    }

    let layout =
        NemotronPrefillWorkspaceLayout::new(1024, 256, 128, 64, 4096, 8192).expect("layout");

    assert_eq!(layout.hidden_a.offset % 256, 0);
    assert_eq!(layout.hidden_b.offset % 256, 0);
    assert_eq!(layout.normalized.offset % 256, 0);
    assert_eq!(layout.route_pack.bytes, 64 * 2);
    assert_non_overlapping(layout.hidden_a, layout.hidden_b);
    assert_non_overlapping(layout.hidden_b, layout.normalized);
    assert_non_overlapping(layout.normalized, layout.router_logits);
    assert_non_overlapping(layout.router_logits, layout.route_pack);
    assert_non_overlapping(layout.route_pack, layout.moe_shared_mid);
    assert_non_overlapping(layout.moe_shared_mid, layout.moe_sparse_mid);
    assert!(layout.total_bytes >= 1024 * 2 + 256 + 128 + 64 * 2 + 4096 + 8192);
}

#[test]
fn nemotron_workspace_layout_reports_align_overflow() {
    use crate::runtime::types::NemotronPrefillWorkspaceLayout;

    let err = NemotronPrefillWorkspaceLayout::new(usize::MAX - 127, 1, 1, 1, 1, 1)
        .expect_err("layout overflow must return an error");

    assert!(err.contains("hidden_b"));
    assert!(err.contains("align"));
}

#[test]
fn nemotron_workspace_layout_reports_total_bytes_align_overflow() {
    use crate::runtime::types::NemotronPrefillWorkspaceLayout;

    let err = NemotronPrefillWorkspaceLayout::new(0, 0, 0, 0, 0, usize::MAX - 127)
        .expect_err("total byte align overflow must return an error");

    assert!(err.contains("total_bytes"));
    assert!(err.contains("align"));
}

#[test]
fn nemotron_workspace_disabled_begin_does_not_open_default_cuda_state() {
    let _guard = runtime_test_lock();

    let summary = begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
        hidden_bytes: 1024,
        normalized_bytes: 1024,
        router_logits_bytes: 256,
        route_bytes: 128,
        moe_shared_mid_bytes: 4096,
        moe_sparse_mid_bytes: 8192,
        required_workspace_bytes: 16 * 1024,
        enabled: false,
    })
    .expect("disabled begin returns inactive summary");

    assert_eq!(
        summary,
        NemotronPrefillWorkspaceSummary {
            active: false,
            arena_bytes: 0,
            live_leases: 0,
            hit_bytes: 0,
            miss_bytes: 0,
            owned_alloc_count: 0,
        }
    );
    assert!(
        lock_default_cuda_compute_for_test()
            .as_deref()
            .is_none_or(|state| state.is_none()),
        "disabled begin must not initialize CUDA state"
    );

    let err = end_nemotron_prefill_workspace()
        .expect_err("ending after disabled begin must still see uninitialized state");
    assert!(err.contains("cuda compute state is not initialized"));
}

#[test]
fn nemotron_workspace_router_logits_uses_workspace_and_releases_leases() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let input_desc = DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::MambaOutput);
    let input = [1.0f32, 2.0, 3.0, 4.0, -2.0, 1.0];
    let norm_weight = [1.0f32, 0.5, -1.0];
    let router_weight = [0.25f32, 0.5, -0.75, -1.0, 0.25, 0.5];
    let input_id = state
        .upload_device_tensor_f32(input_desc, &input)
        .expect("upload router input");
    let normalized_bytes = 2 * 3 * std::mem::size_of::<f32>();
    let logits_bytes = 2 * 2 * std::mem::size_of::<f32>();

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes: 0,
            normalized_bytes,
            router_logits_bytes: logits_bytes,
            route_bytes: 0,
            moe_shared_mid_bytes: 0,
            moe_sparse_mid_bytes: 0,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");

    let output = state
        .nemotron_router_logits_from_device_f32(
            input_id,
            input_desc,
            &norm_weight,
            &router_weight,
            2,
            3,
            2,
            1.0e-6,
        )
        .expect("router logits from workspace");
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary");
    assert_eq!(summary.live_leases, 2);
    assert_eq!(summary.owned_alloc_count, 0);
    assert!(summary.hit_bytes >= normalized_bytes + logits_bytes);

    assert!(state
        .release_device_tensor(output.normalized_id)
        .expect("release normalized"));
    assert!(state
        .release_device_tensor(output.router_logits_id)
        .expect("release logits"));
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary after release");
    assert_eq!(summary.live_leases, 0);
    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");
    assert!(state
        .release_device_tensor(input_id)
        .expect("release router input"));
}

#[test]
fn nemotron_workspace_hidden_output_skips_live_residual_hidden_slice() {
    use crate::runtime::types::DeviceTensorStorage;

    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let hidden_bytes = 2 * 3 * std::mem::size_of::<f32>();
    let normalized_bytes = hidden_bytes;

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes,
            normalized_bytes,
            router_logits_bytes: 0,
            route_bytes: 0,
            moe_shared_mid_bytes: 0,
            moe_sparse_mid_bytes: 0,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");

    let layout = state
        .nemotron_prefill_workspace
        .as_ref()
        .expect("workspace")
        .layout;
    let hidden_desc = DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::Hidden);
    let normalized_desc =
        DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::Normalized);
    let (residual_ptr, residual_storage) = state
        .nemotron_workspace_slice_ptr(layout.hidden_a)
        .expect("residual hidden lease");
    let residual_id = state
        .insert_device_tensor_slot_with_storage(
            residual_ptr,
            hidden_bytes,
            hidden_desc,
            residual_storage,
        )
        .expect("insert residual hidden");
    let (normalized_ptr, normalized_storage) = state
        .nemotron_workspace_slice_ptr(layout.normalized)
        .expect("normalized lease");
    let normalized_id = state
        .insert_device_tensor_slot_with_storage(
            normalized_ptr,
            normalized_bytes,
            normalized_desc,
            normalized_storage,
        )
        .expect("insert normalized");

    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary before output");
    assert_eq!(summary.live_leases, 2);

    let (output_ptr, output_storage) = state
        .nemotron_workspace_hidden_output_ptr(hidden_bytes)
        .expect("output hidden lease");

    assert_eq!(
        output_ptr,
        state.nemotron_prefill_workspace.as_ref().unwrap().ptr + layout.hidden_b.offset as u64
    );
    assert_eq!(
        output_storage,
        DeviceTensorStorage::NemotronWorkspace {
            arena_id: state.nemotron_prefill_workspace.as_ref().unwrap().id,
            offset: layout.hidden_b.offset,
            bytes: hidden_bytes,
        }
    );

    let output_id = state
        .insert_device_tensor_slot_with_storage(
            output_ptr,
            hidden_bytes,
            hidden_desc,
            output_storage,
        )
        .expect("insert output hidden");

    assert!(state
        .release_device_tensor(residual_id)
        .expect("release residual"));
    assert!(state
        .release_device_tensor(normalized_id)
        .expect("release normalized"));
    assert!(state
        .release_device_tensor(output_id)
        .expect("release output"));
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary after release");
    assert_eq!(summary.live_leases, 0);
    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");
}

#[test]
fn nemotron_workspace_route_pack_uses_original_and_reordered_halves() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let seq_len = 3usize;
    let n_expert = 5usize;
    let expert_used = 2usize;
    let slots = seq_len * expert_used;
    let ids_bytes = slots * std::mem::size_of::<u32>();
    let weights_bytes = slots * std::mem::size_of::<f32>();
    let route_bytes = ids_bytes + weights_bytes + ids_bytes;
    let logits = [
        0.5_f32, -0.25, 1.0, 0.125, -1.0, -0.5, 0.75, 0.25, 1.25, 0.0, 0.0, -0.1, 0.2, 0.3, 0.4,
    ];
    let logits_desc = DeviceTensorDesc::new(
        seq_len,
        n_expert,
        ScalarType::F32,
        DeviceTensorRole::RouterLogits,
    );
    let logits_id = state
        .upload_device_tensor_f32(logits_desc, &logits)
        .expect("upload logits");

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes: 0,
            normalized_bytes: 0,
            router_logits_bytes: 0,
            route_bytes,
            moe_shared_mid_bytes: 0,
            moe_sparse_mid_bytes: 0,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");

    let route = state
        .nemotron_device_route_pack_from_logits(
            logits_id,
            logits_desc,
            None,
            seq_len,
            n_expert,
            expert_used,
            1.0,
        )
        .expect("route pack");
    let order = vec![5_u32, 2, 0, 4, 1, 3];
    let sorted = state
        .nemotron_reorder_device_route_pack(route, &order)
        .expect("reorder route pack");

    assert!(matches!(
        route.storage,
        crate::runtime::types::NemotronRoutePackStorage::Workspace { .. }
    ));
    assert!(matches!(
        sorted.storage,
        crate::runtime::types::NemotronRoutePackStorage::Workspace { .. }
    ));
    assert_ne!(route.expert_ids_dev, sorted.expert_ids_dev);
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary");
    assert_eq!(summary.live_leases, 0);
    assert_eq!(summary.owned_alloc_count, 0);
    assert!(summary.hit_bytes >= route_bytes * 2);

    state
        .release_nemotron_device_route_pack(route)
        .expect("release route pack");
    state
        .release_nemotron_device_route_pack(sorted)
        .expect("release sorted route pack");
    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");
    assert!(state
        .release_device_tensor(logits_id)
        .expect("release logits"));
}

#[test]
fn nemotron_workspace_moe_mid_uses_workspace_without_live_lease() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let shared_mid_bytes = 3 * 96 * std::mem::size_of::<f32>();
    let sparse_mid_bytes = 3 * 64 * std::mem::size_of::<f32>();

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes: 1024,
            normalized_bytes: 0,
            router_logits_bytes: 0,
            route_bytes: 0,
            moe_shared_mid_bytes: shared_mid_bytes,
            moe_sparse_mid_bytes: sparse_mid_bytes,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");

    let workspace = state
        .nemotron_prefill_workspace
        .as_ref()
        .expect("workspace");
    let expected_shared = workspace
        .ptr
        .checked_add(workspace.layout.moe_shared_mid.offset as u64)
        .expect("shared ptr");
    let expected_sparse = workspace
        .ptr
        .checked_add(workspace.layout.moe_sparse_mid.offset as u64)
        .expect("sparse ptr");

    let (shared, sparse) = state
        .nemotron_workspace_moe_mid_ptrs(shared_mid_bytes, sparse_mid_bytes)
        .expect("workspace MoE mid ptrs");

    assert_eq!(shared, expected_shared);
    assert_eq!(sparse, expected_sparse);
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary");
    assert_eq!(summary.live_leases, 0);
    assert_eq!(summary.owned_alloc_count, 0);
    assert_eq!(summary.hit_bytes, shared_mid_bytes + sparse_mid_bytes);

    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");
}

#[test]
fn nemotron_workspace_moe_mid_short_slice_records_miss_without_live_lease() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let shared_mid_bytes = 3 * 96 * std::mem::size_of::<f32>();
    let sparse_mid_bytes = 3 * 64 * std::mem::size_of::<f32>();
    let requested = shared_mid_bytes + sparse_mid_bytes;

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes: 1024,
            normalized_bytes: 0,
            router_logits_bytes: 0,
            route_bytes: 0,
            moe_shared_mid_bytes: shared_mid_bytes - std::mem::size_of::<f32>(),
            moe_sparse_mid_bytes: sparse_mid_bytes,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");

    assert!(state
        .nemotron_workspace_moe_mid_ptrs(shared_mid_bytes, sparse_mid_bytes)
        .is_none());
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary");
    assert_eq!(summary.live_leases, 0);
    assert_eq!(summary.hit_bytes, 0);
    assert_eq!(summary.miss_bytes, requested);
    assert_eq!(summary.owned_alloc_count, 2);

    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");
}

#[test]
fn nemotron_workspace_moe_mid_inactive_records_miss_without_live_lease() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let shared_mid_bytes = 3 * 96 * std::mem::size_of::<f32>();
    let sparse_mid_bytes = 3 * 64 * std::mem::size_of::<f32>();
    let requested = shared_mid_bytes + sparse_mid_bytes;

    state
        .begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
            hidden_bytes: 1024,
            normalized_bytes: 0,
            router_logits_bytes: 0,
            route_bytes: 0,
            moe_shared_mid_bytes: shared_mid_bytes,
            moe_sparse_mid_bytes: sparse_mid_bytes,
            required_workspace_bytes: 4096,
            enabled: true,
        })
        .expect("begin workspace");
    state
        .end_nemotron_prefill_workspace()
        .expect("end workspace");

    assert!(state
        .nemotron_workspace_moe_mid_ptrs(shared_mid_bytes, sparse_mid_bytes)
        .is_none());
    let summary = state
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary");
    assert!(!summary.active);
    assert_eq!(summary.live_leases, 0);
    assert_eq!(summary.hit_bytes, 0);
    assert_eq!(summary.miss_bytes, requested);
    assert_eq!(summary.owned_alloc_count, 2);
}

fn cuda_test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn runtime_test_lock() -> RuntimeTestGuard {
    let guard = CUDA_TEST_LOCK
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reset_default_cuda_compute_for_test();
    RuntimeTestGuard { _guard: guard }
}

#[test]
fn cuda_device_tensor_upload_download_roundtrip() {
    let _guard = runtime_test_lock();
    let desc = rnb_backend_api::DeviceTensorDesc::new(
        2,
        3,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Hidden,
    );
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let id = upload_device_tensor_f32(desc, &input).expect("upload tensor");
    let output = download_device_tensor_f32(id).expect("download tensor");
    let row1 = download_device_tensor_f32_row(id, 1).expect("download tensor row");

    assert_eq!(output, input);
    assert_eq!(row1, vec![4.0, 5.0, 6.0]);

    assert!(release_device_tensor(id).expect("release tensor"));
    let err = download_device_tensor_f32(id).expect_err("released tensor must be missing");
    assert!(err.contains("missing CUDA device tensor id"));
}

#[test]
fn nemotron_device_route_pack_public_api_downloads_and_reorders_experts() {
    let _guard = runtime_test_lock();
    let seq_len = 3usize;
    let n_expert = 5usize;
    let expert_used = 2usize;
    let logits = [
        0.5_f32, -0.25, 1.0, 0.125, -1.0, -0.5, 0.75, 0.25, 1.25, 0.0, 0.0, -0.1, 0.2, 0.3, 0.4,
    ];
    let bias = [0.0_f32, 0.1, -0.2, 0.0, 0.05];
    let logits_desc = DeviceTensorDesc::new(
        seq_len,
        n_expert,
        ScalarType::F32,
        DeviceTensorRole::RouterLogits,
    );
    let logits_id = upload_device_tensor_f32(logits_desc, &logits).expect("upload logits");

    let route = nemotron_device_route_pack_from_logits(
        logits_id,
        logits_desc,
        Some(&bias),
        seq_len,
        n_expert,
        expert_used,
        1.25,
    )
    .expect("route pack");

    assert_eq!(route.slots(), seq_len * expert_used);
    assert_eq!(route.seq_len(), seq_len);
    assert_eq!(route.expert_used(), expert_used);

    let experts = nemotron_device_route_pack_expert_ids(&route).expect("download expert ids");
    let order = [5_u32, 2, 0, 4, 1, 3];
    let sorted =
        nemotron_reorder_device_route_pack(&route, &order).expect("reorder route pack by order");
    let sorted_experts =
        nemotron_device_route_pack_expert_ids(&sorted).expect("download sorted expert ids");
    let expected_sorted = order
        .iter()
        .map(|&idx| experts[idx as usize])
        .collect::<Vec<_>>();

    assert_eq!(sorted_experts, expected_sorted);
    release_nemotron_device_route_pack(sorted).expect("release sorted route");
    release_nemotron_device_route_pack(route).expect("release route");
    assert!(release_device_tensor(logits_id).expect("release logits"));
}

#[test]
fn nemotron_router_logits_from_device_rejects_shape_mismatch() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let input_desc = DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::MambaOutput);
    let input_id = state
        .upload_device_tensor_f32(input_desc, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let err = state
        .nemotron_router_logits_from_device_f32(
            input_id,
            input_desc,
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            2,
            4,
            2,
            1.0e-6,
        )
        .unwrap_err();

    assert!(err.contains("shape mismatch"));
    let _ = state.release_device_tensor(input_id);
}

#[test]
fn nemotron_router_logits_from_device_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let input_desc = DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::MambaOutput);
    let input = [1.0f32, 2.0, 3.0, 4.0, -2.0, 1.0];
    let norm_weight = [1.0f32, 0.5, -1.0];
    let router_weight = [0.25f32, 0.5, -0.75, -1.0, 0.25, 0.5];
    let input_id = state
        .upload_device_tensor_f32(input_desc, &input)
        .expect("upload router input");

    let output = state
        .nemotron_router_logits_from_device_f32(
            input_id,
            input_desc,
            &norm_weight,
            &router_weight,
            2,
            3,
            2,
            1.0e-6,
        )
        .expect("router logits from device");

    assert_eq!(
        output.normalized_desc,
        DeviceTensorDesc::new(2, 3, ScalarType::F32, DeviceTensorRole::Normalized)
    );
    assert_eq!(
        output.router_logits_desc,
        DeviceTensorDesc::new(2, 2, ScalarType::F32, DeviceTensorRole::RouterLogits)
    );

    let actual_norm = state
        .download_device_tensor_f32(output.normalized_id)
        .expect("download normalized");
    let actual_logits = state
        .download_device_tensor_f32(output.router_logits_id)
        .expect("download router logits");
    let expected_norm: Vec<f32> = input
        .chunks_exact(3)
        .flat_map(|row| cpu_rms_norm(row, &norm_weight, 1.0e-6, false))
        .collect();
    let mut expected_logits = Vec::with_capacity(4);
    for row in expected_norm.chunks_exact(3) {
        for expert in router_weight.chunks_exact(3) {
            expected_logits.push(row.iter().zip(expert).map(|(x, w)| x * w).sum::<f32>());
        }
    }

    assert_close_rows("router normalized", &actual_norm, &expected_norm, 1.0e-5);
    assert_close_rows("router logits", &actual_logits, &expected_logits, 1.0e-5);

    assert!(state
        .release_device_tensor(input_id)
        .expect("release input"));
    assert!(state
        .release_device_tensor(output.normalized_id)
        .expect("release normalized"));
    assert!(state
        .release_device_tensor(output.router_logits_id)
        .expect("release logits"));
}

#[test]
fn nemotron_router_logits_from_device_matches_f32_rms_reference_for_wide_rows() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let rows = 2usize;
    let hidden_dim = 2688usize;
    let n_expert = 4usize;
    let input_desc = DeviceTensorDesc::new(
        rows,
        hidden_dim,
        ScalarType::F32,
        DeviceTensorRole::MambaOutput,
    );
    let input = (0..rows * hidden_dim)
        .map(|idx| {
            let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let bucket = (idx % 127) as f32 - 63.0;
            sign * (bucket * 0.03125 + 0.001 * (idx % 17) as f32)
        })
        .collect::<Vec<_>>();
    let norm_weight = (0..hidden_dim)
        .map(|idx| 0.5 + (idx % 13) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let router_weight = (0..n_expert * hidden_dim)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.0078125)
        .collect::<Vec<_>>();
    let input_id = state
        .upload_device_tensor_f32(input_desc, &input)
        .expect("upload router input");

    let output = state
        .nemotron_router_logits_from_device_f32(
            input_id,
            input_desc,
            &norm_weight,
            &router_weight,
            rows,
            hidden_dim,
            n_expert,
            1.0e-5,
        )
        .expect("router logits from device");

    let actual_norm = state
        .download_device_tensor_f32(output.normalized_id)
        .expect("download normalized");
    let expected_norm: Vec<f32> = input
        .chunks_exact(hidden_dim)
        .flat_map(|row| cpu_rms_norm_div(row, &norm_weight, 1.0e-5, false))
        .collect();

    assert_close_rows("router normalized f32", &actual_norm, &expected_norm, 0.0);

    assert!(state
        .release_device_tensor(input_id)
        .expect("release input"));
    assert!(state
        .release_device_tensor(output.normalized_id)
        .expect("release normalized"));
    assert!(state
        .release_device_tensor(output.router_logits_id)
        .expect("release logits"));
}

#[test]
fn resident_f32_key_changes_when_contents_change() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![1.0f32, 2.0, 4.0];

    assert_ne!(f32_key(&left), f32_key(&right));
}

#[test]
fn nemotron_device_route_pack_matches_sigmoid_topk_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let seq_len = 3usize;
    let n_expert = 5usize;
    let expert_used = 2usize;
    let expert_weight_scale = 1.25f32;
    let logits = [
        0.5_f32, -0.25, 1.0, 0.125, -1.0, -0.5, 0.75, 0.25, 1.25, 0.0, 0.0, -0.1, 0.2, 0.3, 0.4,
    ];
    let bias = [0.0_f32, 0.1, -0.2, 0.0, 0.05];
    let logits_desc = DeviceTensorDesc::new(
        seq_len,
        n_expert,
        ScalarType::F32,
        DeviceTensorRole::RouterLogits,
    );
    let logits_id = state
        .upload_device_tensor_f32(logits_desc, &logits)
        .expect("upload logits");

    let route = state
        .nemotron_device_route_pack_from_logits(
            logits_id,
            logits_desc,
            Some(&bias),
            seq_len,
            n_expert,
            expert_used,
            expert_weight_scale,
        )
        .expect("route pack");

    let slots = seq_len * expert_used;
    assert_eq!(route.slots, slots);
    assert_eq!(route.seq_len, seq_len);
    assert_eq!(route.expert_used, expert_used);
    let mut actual_experts = vec![0_u32; slots];
    let mut actual_weights = vec![0.0_f32; slots];
    let mut actual_tokens = vec![0_u32; slots];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_experts.as_mut_ptr().cast::<libc::c_void>(),
                route.expert_ids_dev,
                std::mem::size_of_val(actual_experts.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_weights.as_mut_ptr().cast::<libc::c_void>(),
                route.route_weights_dev,
                std::mem::size_of_val(actual_weights.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_tokens.as_mut_ptr().cast::<libc::c_void>(),
                route.token_ids_dev,
                std::mem::size_of_val(actual_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected_experts = Vec::new();
    let mut expected_weights = Vec::new();
    let mut expected_tokens = Vec::new();
    for token in 0..seq_len {
        let token_logits = &logits[token * n_expert..(token + 1) * n_expert];
        let mut scored = token_logits
            .iter()
            .enumerate()
            .map(|(expert, &logit)| {
                let weight = 1.0 / (1.0 + (-logit).exp());
                let score = weight + bias[expert];
                (expert, score, weight)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let selected = &scored[..expert_used];
        let selected_sum = selected.iter().map(|(_, _, weight)| *weight).sum::<f32>();
        for &(expert, _, weight) in selected {
            expected_experts.push(expert as u32);
            expected_weights.push(weight / selected_sum * expert_weight_scale);
            expected_tokens.push(token as u32);
        }
    }

    assert_eq!(actual_experts, expected_experts);
    assert_eq!(actual_tokens, expected_tokens);
    assert_close_rows(
        "Nemotron device route pack weights",
        &actual_weights,
        &expected_weights,
        1e-6,
    );
    assert!(state
        .release_device_tensor(logits_id)
        .expect("release logits"));
    state
        .release_nemotron_device_route_pack(route)
        .expect("release route pack");
}

#[test]
fn cuda_qwen35_prefill_device_topk_route_matches_host_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 3usize;
    let hidden_dim = 4usize;
    let n_expert = 6usize;
    let n_expert_used = 2usize;
    let norm_all = [
        1.0_f32, -0.5, 0.25, 0.75, -0.25, 0.5, 1.5, -1.0, 0.125, -0.75, 0.5, 1.25,
    ];
    let router_w = [
        0.5_f32, -0.25, 0.0, 0.125, -0.75, 0.5, 0.25, -0.125, 0.25, 0.375, -0.5, 0.0, -0.125,
        -0.25, 0.75, 0.5, 0.0, 0.125, 0.5, -0.375, 0.5, 0.5, -0.25, 0.25,
    ];

    let actual = state
        .qwen35_prefill_device_topk_route_pack(
            &router_w,
            n_expert,
            hidden_dim,
            &norm_all,
            seq_len,
            n_expert_used,
        )
        .expect("device top-k route pack");

    let router_logits = f32_gemm_batch(&router_w, n_expert, hidden_dim, &norm_all).unwrap();
    let mut expected =
        qwen35_host_prefill_sparse_slots_for_test(&router_logits, seq_len, n_expert, n_expert_used);
    let mut actual_slots = actual.to_route_slots();
    expected.sort_unstable_by_key(|slot| (slot.expert, slot.token));
    actual_slots.sort_unstable_by_key(|slot| (slot.expert, slot.token));

    assert_eq!(
        actual_slots
            .iter()
            .map(|slot| (slot.expert, slot.token))
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|slot| (slot.expert, slot.token))
            .collect::<Vec<_>>()
    );
    assert_close_rows(
        "Qwen35 prefill device top-k route weights",
        &actual_slots
            .iter()
            .map(|slot| slot.weight)
            .collect::<Vec<_>>(),
        &expected.iter().map(|slot| slot.weight).collect::<Vec<_>>(),
        1.0e-6,
    );
}

#[test]
fn cuda_qwen35_prefill_device_topk_route_weights_match_host_softmax() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 1usize;
    let hidden_dim = 1usize;
    let n_expert = 8usize;
    let n_expert_used = 4usize;
    let norm_all = [1.0_f32];
    let router_w = [-2.75_f32, 0.125, 5.5, -0.875, 1.75, 3.25, -4.5, 0.625];

    let actual = state
        .qwen35_prefill_device_topk_route_pack(
            &router_w,
            n_expert,
            hidden_dim,
            &norm_all,
            seq_len,
            n_expert_used,
        )
        .expect("device top-k route pack");

    let router_logits = f32_gemm_batch(&router_w, n_expert, hidden_dim, &norm_all).unwrap();
    let expected =
        qwen35_host_prefill_sparse_slots_for_test(&router_logits, seq_len, n_expert, n_expert_used);
    let actual_slots = actual.to_route_slots();

    assert_eq!(
        actual_slots
            .iter()
            .map(|slot| (slot.expert, slot.token))
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|slot| (slot.expert, slot.token))
            .collect::<Vec<_>>()
    );
    assert_close_rows(
        "Qwen35 prefill device top-k route weights",
        &actual_slots
            .iter()
            .map(|slot| slot.weight)
            .collect::<Vec<_>>(),
        &expected.iter().map(|slot| slot.weight).collect::<Vec<_>>(),
        1.0e-6,
    );
}

#[test]
fn cuda_generic_moe_route_topk_matches_softmax_and_sigmoid_contracts() {
    let _guard = runtime_test_lock();
    let logits = [0.0_f32, 4.0, 3.0, 2.0, -1.0, 1.0, 2.0, 3.0];
    let (ids, weights, retained) =
        moe_route_topk_f32(&logits, None, 2, 4, 2, false, true, 1.0, None)
            .expect("CUDA softmax route");
    assert_eq!(ids, vec![1, 2, 3, 2]);
    assert_eq!(retained, vec![2, 2]);
    let expected_softmax = [
        1.0 / (1.0 + (-1.0_f32).exp()),
        1.0 / (1.0 + 1.0_f32.exp()),
        1.0 / (1.0 + (-1.0_f32).exp()),
        1.0 / (1.0 + 1.0_f32.exp()),
    ];
    assert_close_rows(
        "CUDA generic softmax route",
        &weights,
        &expected_softmax,
        1.0e-6,
    );

    let sigmoid_logits = [0.0_f32, 1.0, 2.0, 3.0];
    let bias = [2.0_f32, 0.0, 0.0, -2.0];
    let (ids, weights, retained) = moe_route_topk_f32(
        &sigmoid_logits,
        Some(&bias),
        1,
        4,
        3,
        true,
        true,
        2.0,
        Some(0.6),
    )
    .expect("CUDA sigmoid route");
    assert_eq!(&ids[..3], &[0, 2, 1]);
    assert_eq!(retained, vec![2]);
    let p0 = 0.5_f32;
    let p2 = 1.0 / (1.0 + (-2.0_f32).exp());
    let selected_sum = p0 + p2 + 1.0 / (1.0 + (-1.0_f32).exp());
    let retained_mass = (p0 + p2) / selected_sum;
    assert_close_rows(
        "CUDA generic sigmoid route",
        &weights[..2],
        &[
            2.0 * p0 / selected_sum / retained_mass,
            2.0 * p2 / selected_sum / retained_mass,
        ],
        1.0e-6,
    );
}

fn qwen35_host_prefill_sparse_slots_for_test(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<rnb_backend_api::MoeRouteSlot> {
    let selected_count = n_expert_used.min(n_expert);
    let mut sparse_slots = Vec::with_capacity(seq_len * selected_count);
    for token in 0..seq_len {
        let logits = &router_logits[token * n_expert..(token + 1) * n_expert];
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..selected_count];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        for &expert in selected {
            sparse_slots.push(rnb_backend_api::MoeRouteSlot::new(
                expert,
                token as u32,
                (logits[expert] - max_selected).exp() / selected_sum,
            ));
        }
    }
    sparse_slots
}

#[test]
fn nemotron_device_route_pack_reorder_matches_host_order() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().unwrap();
    let seq_len = 3usize;
    let n_expert = 5usize;
    let expert_used = 2usize;
    let logits = [
        0.5_f32, -0.25, 1.0, 0.125, -1.0, -0.5, 0.75, 0.25, 1.25, 0.0, 0.0, -0.1, 0.2, 0.3, 0.4,
    ];
    let logits_desc = DeviceTensorDesc::new(
        seq_len,
        n_expert,
        ScalarType::F32,
        DeviceTensorRole::RouterLogits,
    );
    let logits_id = state
        .upload_device_tensor_f32(logits_desc, &logits)
        .expect("upload logits");
    let route = state
        .nemotron_device_route_pack_from_logits(
            logits_id,
            logits_desc,
            None,
            seq_len,
            n_expert,
            expert_used,
            1.0,
        )
        .expect("route pack");
    let order = vec![5_u32, 2, 0, 4, 1, 3];
    let sorted = state
        .nemotron_reorder_device_route_pack(route, &order)
        .expect("reorder route pack");

    let slots = route.slots;
    let mut base_experts = vec![0_u32; slots];
    let mut base_weights = vec![0.0_f32; slots];
    let mut base_tokens = vec![0_u32; slots];
    let mut sorted_experts = vec![0_u32; slots];
    let mut sorted_weights = vec![0.0_f32; slots];
    let mut sorted_tokens = vec![0_u32; slots];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                base_experts.as_mut_ptr().cast::<libc::c_void>(),
                route.expert_ids_dev,
                std::mem::size_of_val(base_experts.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                base_weights.as_mut_ptr().cast::<libc::c_void>(),
                route.route_weights_dev,
                std::mem::size_of_val(base_weights.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                base_tokens.as_mut_ptr().cast::<libc::c_void>(),
                route.token_ids_dev,
                std::mem::size_of_val(base_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                sorted_experts.as_mut_ptr().cast::<libc::c_void>(),
                sorted.expert_ids_dev,
                std::mem::size_of_val(sorted_experts.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                sorted_weights.as_mut_ptr().cast::<libc::c_void>(),
                sorted.route_weights_dev,
                std::mem::size_of_val(sorted_weights.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                sorted_tokens.as_mut_ptr().cast::<libc::c_void>(),
                sorted.token_ids_dev,
                std::mem::size_of_val(sorted_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected_experts = order
        .iter()
        .map(|&idx| base_experts[idx as usize])
        .collect::<Vec<_>>();
    let expected_weights = order
        .iter()
        .map(|&idx| base_weights[idx as usize])
        .collect::<Vec<_>>();
    let expected_tokens = order
        .iter()
        .map(|&idx| base_tokens[idx as usize])
        .collect::<Vec<_>>();

    assert_eq!(sorted_experts, expected_experts);
    assert_eq!(sorted_tokens, expected_tokens);
    assert_close_rows(
        "Nemotron reordered route weights",
        &sorted_weights,
        &expected_weights,
        0.0,
    );
    assert!(state
        .release_device_tensor(logits_id)
        .expect("release logits"));
    state
        .release_nemotron_device_route_pack(route)
        .expect("release route pack");
    state
        .release_nemotron_device_route_pack(sorted)
        .expect("release sorted route pack");
}

fn allow_default_cuda_q4_f32_cache_for_test() {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = compute
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.is_none() {
        *guard = Some(CudaState::open().expect("open default CUDA state"));
    }
    guard
        .as_mut()
        .expect("default CUDA state initialized")
        .resident_q4_f32_limit = usize::MAX;
}

fn restore_env_var(name: &str, previous: Option<String>) {
    if let Some(value) = previous {
        std::env::set_var(name, value);
    } else {
        std::env::remove_var(name);
    }
}

fn cuda_driver_unavailable_for_test(err: &str) -> bool {
    err.contains("could not load CUDA driver library")
        || err.contains("missing CUDA driver symbol")
        || err.contains("cuInit failed with CUDA error 100")
        || err.contains("cuDeviceGet failed with CUDA error 100")
        || err.contains("cuDeviceGet failed with CUDA error 101")
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var(name).ok();
        std::env::set_var(name, value);
        Self { name, previous }
    }

    fn remove(name: &'static str) -> Self {
        let previous = std::env::var(name).ok();
        std::env::remove_var(name);
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        restore_env_var(self.name, self.previous.take());
    }
}

#[test]
fn cuda_loader_records_requested_expert_bytes() {
    let _guard = runtime_test_lock();
    let loader = CudaMoeJitLoader::default();
    let gate = vec![1u8; 80];
    let up = vec![2u8; 160];
    let down = vec![3u8; 240];
    loader.request_load(&MoeJitLoadRequest {
        backend_hint: Some(MoeJitBackendKind::Cuda),
        layer_idx: 3,
        experts: vec![7, 2],
        gate_bytes_per_expert: 10,
        up_bytes_per_expert: 20,
        down_bytes_per_expert: 30,
        expert_loads: vec![
            MoeJitExpertLoad {
                expert: 7,
                gate: MoeJitByteRange::from_tensor_slice(&gate, 70, 10),
                up: MoeJitByteRange::from_tensor_slice(&up, 140, 20),
                down: MoeJitByteRange::from_tensor_slice(&down, 210, 30),
            },
            MoeJitExpertLoad {
                expert: 2,
                gate: MoeJitByteRange::from_tensor_slice(&gate, 20, 10),
                up: MoeJitByteRange::from_tensor_slice(&up, 40, 20),
                down: MoeJitByteRange::from_tensor_slice(&down, 60, 30),
            },
        ],
    });

    let stats = loader.stats();
    assert_eq!(stats.requests, 1);
    assert_eq!(stats.requested_experts, 2);
    assert_eq!(stats.requested_bytes, 120);
    // Both experts use byte-identical role slices, so the content-addressed
    // resident cache copies one gate/up/down bundle while retaining demand stats.
    assert_eq!(stats.copied_bytes, 60);
    assert!(stats.resident_bytes >= 60);
    assert_eq!(stats.cuda_failures, 0);
}

#[test]
fn unique_q4k_slot_bytes_counts_repeated_slices_once() {
    let a = vec![1u8; 10];
    let b = vec![2u8; 20];
    let slots = [&a[..], &b[..], &a[..]];
    let bytes = unique_q4k_slot_bytes(slots.iter());
    assert_eq!(bytes, 30);
}

#[test]
fn qwen35_decode_resident_batch_bytes_counts_unique_gate_up_down_slots() {
    let gate0 = vec![1u8; 10];
    let gate1 = vec![2u8; 20];
    let up0 = vec![3u8; 30];
    let down0 = vec![4u8; 40];
    let gate = [&gate0[..], &gate1[..], &up0[..]];
    let up = [&up0[..], &gate1[..]];
    let down = [&down0[..], &gate0[..]];

    let bytes = qwen35_decode_resident_batch_bytes(&gate, &up, &down);

    assert_eq!(bytes, 100);
}

#[test]
fn qwen35_decode_resident_batch_requires_all_unique_slots_to_fit() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];

    assert!(qwen35_decode_resident_batch_fits(&gate, &up, &down, 60));
    assert!(!qwen35_decode_resident_batch_fits(&gate, &up, &down, 59));
}

#[test]
fn qwen35_decode_selected_slot_plan_uses_temp_slab_when_resident_batch_is_disabled() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];

    let plan = qwen35_decode_selected_slot_ptr_plan(&gate, &up, &down, 60, false, false);

    assert_eq!(plan, Qwen35DecodeSelectedSlotPtrPlan::TempSlab);
}

#[test]
fn qwen35_decode_selected_slot_plan_uses_temp_slab_when_batch_does_not_fit() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];

    let plan = qwen35_decode_selected_slot_ptr_plan(&gate, &up, &down, 59, true, false);

    assert_eq!(plan, Qwen35DecodeSelectedSlotPtrPlan::TempSlab);
}

#[test]
fn qwen35_decode_selected_slot_plan_uses_mixed_when_resident_slot_exists() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];

    let plan = qwen35_decode_selected_slot_ptr_plan(&gate, &up, &down, 59, false, true);

    assert_eq!(plan, Qwen35DecodeSelectedSlotPtrPlan::MixedResidentTemp);
}

#[test]
fn qwen35_decode_selected_slot_plan_uses_resident_batch_when_enabled_and_fits() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];

    let plan = qwen35_decode_selected_slot_ptr_plan(&gate, &up, &down, 60, true, true);

    assert_eq!(plan, Qwen35DecodeSelectedSlotPtrPlan::ResidentBatch);
}

#[test]
fn qwen35_decode_selected_expert_bundle_all_resident_is_one_hit() {
    let gate = vec![1u8; 10];
    let up = vec![2u8; 20];
    let down = vec![3u8; 30];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let resident_keys = [
        q4k_resident_key(&gate),
        q4k_resident_key(&up),
        q4k_resident_key(&down),
    ]
    .into_iter()
    .collect::<HashSet<_>>();

    let stats = CudaState::qwen35_decode_selected_expert_bundle_stats_for_test(
        &gate_slots,
        &up_slots,
        &down_slots,
        &resident_keys,
    );

    assert_eq!(stats.bundle_lookups, 1);
    assert_eq!(stats.bundle_hits, 1);
    assert_eq!(stats.bundle_partial_hits, 0);
    assert_eq!(stats.bundle_misses, 0);
}

#[test]
fn qwen35_decode_selected_expert_bundle_partial_and_miss_are_exclusive() {
    let gate0 = vec![1u8; 10];
    let gate1 = vec![2u8; 11];
    let up0 = vec![3u8; 20];
    let up1 = vec![4u8; 21];
    let down0 = vec![5u8; 30];
    let down1 = vec![6u8; 31];
    let gate_slots = [&gate0[..], &gate1[..]];
    let up_slots = [&up0[..], &up1[..]];
    let down_slots = [&down0[..], &down1[..]];
    let resident_keys = [q4k_resident_key(&gate0), q4k_resident_key(&up0)]
        .into_iter()
        .collect::<HashSet<_>>();

    let stats = CudaState::qwen35_decode_selected_expert_bundle_stats_for_test(
        &gate_slots,
        &up_slots,
        &down_slots,
        &resident_keys,
    );

    assert_eq!(stats.bundle_lookups, 2);
    assert_eq!(stats.bundle_hits, 0);
    assert_eq!(stats.bundle_partial_hits, 1);
    assert_eq!(stats.bundle_misses, 1);
    assert_eq!(
        stats.bundle_hits + stats.bundle_partial_hits + stats.bundle_misses,
        stats.bundle_lookups
    );
}

#[test]
fn qwen35_decode_selected_expert_bundle_alias_is_one_allocation_but_two_route_lookups() {
    let aliased = vec![7u8; 32];
    let slots = [&aliased[..], &aliased[..]];
    let resident_keys = [q4k_resident_key(&aliased)]
        .into_iter()
        .collect::<HashSet<_>>();

    let stats = CudaState::qwen35_decode_selected_expert_bundle_stats_for_test(
        &slots,
        &slots,
        &slots,
        &resident_keys,
    );

    assert_eq!(resident_keys.len(), 1);
    assert_eq!(stats.bundle_lookups, 2);
    assert_eq!(stats.bundle_hits, 2);
    assert_eq!(stats.bundle_partial_hits, 0);
    assert_eq!(stats.bundle_misses, 0);
}

fn assert_qwen35_alias_candidate(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    expected_role_bytes: [u64; 3],
) {
    let candidate =
        CudaState::qwen35_decode_expert_bundle_candidate_for_test(gate, up, down, &HashSet::new());
    let footprint = candidate.footprint();
    assert_eq!(
        [
            footprint.gate_bytes(),
            footprint.up_bytes(),
            footprint.down_bytes(),
        ],
        expected_role_bytes
    );
    let unique_payload_bytes = expected_role_bytes.into_iter().sum::<u64>();
    assert_eq!(footprint.total_bytes(), unique_payload_bytes);
    assert_eq!(candidate.missing_admission_bytes(), unique_payload_bytes);
    let decision = rnb_memory::evaluate_expert_bundle_admission(
        candidate,
        2,
        0,
        rnb_memory::CurrentLookupTransfer::ReplacesTempUpload,
    );
    assert_eq!(
        decision.cost.predicted_saved_bytes,
        u128::from(unique_payload_bytes) * 3
    );
    assert!(decision.admit);
}

#[test]
fn qwen35_decode_gate_up_alias_owns_payload_in_gate_role() {
    let gate_up = vec![1u8; 32];
    let down = vec![2u8; 48];
    assert_qwen35_alias_candidate(&gate_up, &gate_up, &down, [32, 0, 48]);
}

#[test]
fn qwen35_decode_up_down_alias_owns_payload_in_up_role() {
    let gate = vec![1u8; 32];
    let up_down = vec![2u8; 48];
    assert_qwen35_alias_candidate(&gate, &up_down, &up_down, [32, 48, 0]);
}

#[test]
fn qwen35_decode_all_role_alias_owns_payload_once() {
    let all_roles = vec![1u8; 32];
    assert_qwen35_alias_candidate(&all_roles, &all_roles, &all_roles, [32, 0, 0]);
}

#[test]
fn qwen35_decode_duplicate_expert_ids_observe_once_per_token() {
    let gate = vec![1u8; 64];
    let up = vec![2u8; 64];
    let down = vec![3u8; 64];
    let gate_slots = [&gate[..], &gate[..]];
    let up_slots = [&up[..], &up[..]];
    let down_slots = [&down[..], &down[..]];
    let resident_keys = HashSet::new();

    let stats = CudaState::qwen35_decode_selected_expert_bundle_stats_with_ids_for_test(
        &[7, 7],
        &gate_slots,
        &up_slots,
        &down_slots,
        &resident_keys,
    );

    assert_eq!(stats.bundle_lookups, 1);
    assert_eq!(stats.bundle_misses, 1);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_cold_one_shot_does_not_evict_resident_weights() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 768;
    let old = [vec![11u8; 256], vec![12u8; 256], vec![13u8; 256]];
    for weights in &old {
        assert!(state
            .preload_resident_q4k_weight_slice(weights)
            .expect("preload old resident"));
    }
    let gate = vec![21u8; 256];
    let up = vec![22u8; 256];
    let down = vec![23u8; 256];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(4),
            &[9],
        )
        .expect("cold bundle uses temporary slots");

    for weights in &old {
        assert!(state.q4k_weight_slice_is_resident(weights));
    }
    assert!(!state.q4k_weight_slice_is_resident(&gate));
    assert!(!state.q4k_weight_slice_is_resident(&up));
    assert!(!state.q4k_weight_slice_is_resident(&down));
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_repeated_route_is_admitted() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![31u8; 64];
    let up = vec![32u8; 64];
    let down = vec![33u8; 64];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let before = cache_snapshot();

    for _ in 0..3 {
        state
            .qwen35_decode_selected_q4k_ptrs_for_test(
                &gate_slots,
                &up_slots,
                &down_slots,
                &[1.0],
                Some(5),
                &[11],
            )
            .expect("observe repeated bundle");
    }

    assert!(state.q4k_weight_slice_is_resident(&gate));
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (4096 / 192, 3, 1)
    );
    let after = cache_snapshot();
    let delta = after.delta(before);
    assert_eq!(delta.expert_bundles.bundle_admissions, 1);
    assert_eq!(delta.expert_bundles.admitted_bytes, 192);
    assert_eq!(
        after.resident_payload_bytes,
        before.resident_payload_bytes + 192
    );
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_fused_error_retry_reuses_current_token_observation() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![34u8; 64];
    let up = vec![35u8; 64];
    let down = vec![36u8; 64];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let before = cache_snapshot();
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();

    let fused_error: Result<(), String> = state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(5),
            &[11],
            &mut bundle_observation_receipt,
        )
        .and_then(|_| Err("injected fused kernel launch failure".to_string()));
    assert_eq!(
        fused_error.expect_err("fused launch must fail"),
        "injected fused kernel launch failure"
    );
    assert!(bundle_observation_receipt.consumed());

    state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(5),
            &[11],
            &mut bundle_observation_receipt,
        )
        .expect("fallback reuses current-token policy result");

    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (4096 / 192, 1, 1)
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 1);
    assert_eq!(delta.bundle_misses, 1);
    assert_eq!(delta.bundle_admissions, 0);
    assert_eq!(delta.h2d_bytes, 384);
    assert_eq!(delta.temp_h2d_bytes, 384);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_combined_post_selected_failures_preserve_one_observation() {
    let _guard = runtime_test_lock();
    let n_ff = 256usize;
    let n_embd = 256usize;
    let resident_limit = 8 * 1024 * 1024;
    let gate = make_test_q2k_weights(n_ff, n_embd / 256, 301);
    let up = make_test_q2k_weights(n_ff, n_embd / 256, 311);
    let down = make_test_q3k_weights(n_embd, n_ff / 256, 321);
    let shared_gate = make_test_q2k_weights(n_ff, n_embd / 256, 331);
    let shared_up = make_test_q2k_weights(n_ff, n_embd / 256, 341);
    let shared_down = make_test_q3k_weights(n_embd, n_ff / 256, 351);
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let route_weights = [1.0f32];
    let input = vec![0.25f32; n_embd];
    let bundle_payload_bytes = gate.len() + up.len() + down.len();

    for failure_point in [
        Qwen35DecodeFailurePoint::CombinedShared,
        Qwen35DecodeFailurePoint::CombinedAdd,
        Qwen35DecodeFailurePoint::CombinedDtoh,
        Qwen35DecodeFailurePoint::CombinedSync,
    ] {
        let mut state = CudaState::open().expect("open CUDA state");
        state.qwen35_target_decode_q4k_limit_checked = true;
        state.resident_q4k_limit = resident_limit;
        state.inject_qwen35_decode_failure_for_test(failure_point);
        let before = cache_snapshot();
        let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
        let mut output = vec![0.0f32; n_embd];

        let err = state
            .qwen35_decode_moe_shared_sparse_into(
                &gate_slots,
                &up_slots,
                &down_slots,
                &route_weights,
                Some(12),
                &[17],
                &mut bundle_observation_receipt,
                11,
                &shared_gate,
                &shared_up,
                &shared_down,
                1.0,
                11,
                n_ff,
                n_embd,
                &input,
                &mut output,
            )
            .expect_err("injected combined failure");
        assert!(err.contains(&format!("{failure_point:?}")));
        assert!(bundle_observation_receipt.consumed());

        state
            .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
                &gate_slots,
                &up_slots,
                &down_slots,
                &route_weights,
                Some(12),
                &[17],
                &mut bundle_observation_receipt,
            )
            .expect("fallback reuses combined selected observation");

        assert_eq!(
            state.qwen35_expert_bundle_history_state_for_test(),
            (resident_limit / bundle_payload_bytes, 1, 1)
        );
        let delta = cache_snapshot().delta(before).expert_bundles;
        assert_eq!(delta.bundle_lookups, 1, "failure point {failure_point:?}");
        assert_eq!(delta.bundle_misses, 1, "failure point {failure_point:?}");
    }
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_residual_pre_selected_failure_fallback_observes_once() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q2k_weights(n_ff, n_embd / 256, 401);
    let up = make_test_q2k_weights(n_ff, n_embd / 256, 411);
    let down = make_test_q3k_weights(n_embd, n_ff / 256, 421);
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let route_weights = [1.0f32];
    let input = vec![0.25f32; n_embd];
    let mut residual = vec![0.5f32; n_embd];
    let before = cache_snapshot();
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    state.inject_qwen35_decode_failure_for_test(
        Qwen35DecodeFailurePoint::ResidualBeforeSelectedObservation,
    );

    let err = state
        .qwen35_sparse_experts_add_residual_into(
            &gate_slots,
            &up_slots,
            &down_slots,
            &route_weights,
            Some(13),
            &[19],
            &mut bundle_observation_receipt,
            11,
            n_ff,
            n_embd,
            &input,
            &mut residual,
        )
        .expect_err("injected pre-observation residual failure");
    assert!(err.contains("ResidualBeforeSelectedObservation"));
    assert!(!bundle_observation_receipt.consumed());

    state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &route_weights,
            Some(13),
            &[19],
            &mut bundle_observation_receipt,
        )
        .expect("fallback observes after pre-selected residual failure");

    assert!(bundle_observation_receipt.consumed());
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (
            state.resident_q4k_limit / (gate.len() + up.len() + down.len()),
            1,
            1
        )
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 1);
    assert_eq!(delta.bundle_misses, 1);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_admission_failure_after_observation_fallback_reuses_receipt() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![91u8; 64];
    let up = vec![92u8; 64];
    let down = vec![93u8; 64];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let before = cache_snapshot();
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    state
        .inject_qwen35_decode_failure_for_test(Qwen35DecodeFailurePoint::AdmissionAfterObservation);

    let err = state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(14),
            &[23],
            &mut bundle_observation_receipt,
        )
        .expect_err("injected admission failure");
    assert!(err.contains("AdmissionAfterObservation"));
    assert!(bundle_observation_receipt.consumed());

    state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(14),
            &[23],
            &mut bundle_observation_receipt,
        )
        .expect("fallback reuses admission observation");

    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (4096 / 192, 1, 1)
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 1);
    assert_eq!(delta.bundle_misses, 1);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_validation_failure_leaves_receipt_for_fallback_observation() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![94u8; 64];
    let up = vec![95u8; 64];
    let down = vec![96u8; 64];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    let before = cache_snapshot();
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();

    let err = state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(15),
            &[29, 31],
            &mut bundle_observation_receipt,
        )
        .expect_err("selected IDs beyond slots must fail");
    assert!(err.contains("selected expert IDs exceed CUDA slots"));
    assert!(!bundle_observation_receipt.consumed());

    state
        .qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(15),
            &[29],
            &mut bundle_observation_receipt,
        )
        .expect("fallback observes after validation failure");

    assert!(bundle_observation_receipt.consumed());
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (4096 / 192, 1, 1)
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 1);
    assert_eq!(delta.bundle_misses, 1);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_current_transfer_savings_cover_eviction_reload() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 1024;
    let gate = vec![41u8; 256];
    let up = vec![42u8; 256];
    let down = vec![43u8; 256];
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    for _ in 0..2 {
        state
            .qwen35_decode_selected_q4k_ptrs_for_test(
                &gate_slots,
                &up_slots,
                &down_slots,
                &[1.0],
                Some(6),
                &[13],
            )
            .expect("seed bundle history");
    }
    let old = vec![51u8; 1024];
    assert!(state
        .preload_resident_q4k_weight_slice(&old)
        .expect("preload expensive victim"));

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(6),
            &[13],
        )
        .expect("admit bundle when current and future transfer savings cover eviction");

    assert!(!state.q4k_weight_slice_is_resident(&old));
    assert!(state.q4k_weight_slice_is_resident(&gate));
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_partial_hit_uploads_only_missing_roles() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 1024;
    let gate = vec![61u8; 64];
    let up = vec![62u8; 64];
    let down = vec![63u8; 64];
    assert!(state
        .preload_resident_q4k_weight_slice(&gate)
        .expect("preload resident gate"));
    let gate_ptr = state.resident_q4k[&q4k_resident_key(&gate)].ptr;
    let gate_slots = [&gate[..]];
    let up_slots = [&up[..]];
    let down_slots = [&down[..]];
    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(7),
            &[17],
        )
        .expect("seed partial bundle history");
    let before = cache_snapshot();

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &gate_slots,
            &up_slots,
            &down_slots,
            &[1.0],
            Some(7),
            &[17],
        )
        .expect("admit missing bundle roles");

    assert_eq!(state.resident_q4k[&q4k_resident_key(&gate)].ptr, gate_ptr);
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_admissions, 1);
    assert_eq!(delta.admitted_bytes, 128);
    assert_eq!(delta.h2d_bytes, 128);
    assert_eq!(delta.temp_h2d_bytes, 0);
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 192);
    let bundle_key = rnb_memory::SparseExpertCacheKey::new(7, 17);
    assert_eq!(
        state.qwen35_q2q3_bundle_ownership.bundle_roles[&bundle_key].len(),
        3
    );
    for role in [
        q4k_resident_key(&gate),
        q4k_resident_key(&up),
        q4k_resident_key(&down),
    ] {
        assert_eq!(
            state.qwen35_q2q3_bundle_ownership.role_owners[&role].len(),
            1
        );
    }

    state.resident_q4k_limit = state.resident_q4k_bytes;
    let plan = state
        .resident_q4k_eviction_plan_for_incoming(1, &HashSet::new())
        .expect("plan atomic partial-to-full bundle eviction");
    assert_eq!(plan.reload_payload_bytes, 192);
    let eviction = state
        .execute_resident_q4k_eviction_plan(plan)
        .expect("evict atomic partial-to-full bundle");
    assert_eq!(eviction.bundle_evictions, 1);
    assert_eq!(eviction.evicted_bytes, 192);
    assert!(!state.q4k_weight_slice_is_resident(&gate));
    assert!(!state.q4k_weight_slice_is_resident(&up));
    assert!(!state.q4k_weight_slice_is_resident(&down));
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_alias_bundle_closure_is_protected_and_evicted_once_per_owner() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let shared = vec![91u8; 64];
    let a_up = vec![92u8; 64];
    let a_down = vec![93u8; 64];
    let b_up = vec![94u8; 64];
    let b_down = vec![95u8; 64];

    state
        .qwen35_with_decode_reuse_score_override_for_test(1, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&shared],
                &[&a_up],
                &[&a_down],
                &[1.0],
                Some(30),
                &[1],
            )
        })
        .expect("admit first aliased bundle");
    state
        .qwen35_with_decode_reuse_score_override_for_test(1, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&shared],
                &[&b_up],
                &[&b_down],
                &[1.0],
                Some(30),
                &[2],
            )
        })
        .expect("admit second aliased bundle");

    assert_eq!(state.qwen35_q2q3_bundle_ownership.bundle_roles.len(), 2);
    assert_eq!(state.qwen35_q2q3_bundle_ownership.role_owners.len(), 5);
    assert_eq!(
        state.qwen35_q2q3_bundle_ownership.role_owners[&q4k_resident_key(&shared)].len(),
        2
    );
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 320);
    state.resident_q4k_limit = state.resident_q4k_bytes;

    let protected = [q4k_resident_key(&b_down)]
        .into_iter()
        .collect::<HashSet<_>>();
    assert!(
        state
            .resident_q4k_eviction_plan_for_incoming(1, &protected)
            .is_none(),
        "one protected role must protect the complete alias closure"
    );
    for weights in [&shared, &a_up, &a_down, &b_up, &b_down] {
        assert!(state.q4k_weight_slice_is_resident(weights));
    }

    let plan = state
        .resident_q4k_eviction_plan_for_incoming(1, &HashSet::new())
        .expect("plan aliased bundle closure");
    assert_eq!(plan.units.len(), 1);
    assert_eq!(plan.units[0].bundles().len(), 2);
    assert_eq!(plan.units[0].roles().len(), 5);
    assert_eq!(plan.reload_payload_bytes, 320);
    let eviction = state
        .execute_resident_q4k_eviction_plan(plan)
        .expect("evict aliased bundle closure");
    assert_eq!(eviction.bundle_evictions, 2);
    assert_eq!(eviction.evicted_bytes, 320);
    for weights in [&shared, &a_up, &a_down, &b_up, &b_down] {
        assert!(!state.q4k_weight_slice_is_resident(weights));
    }
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_profitable_single_role_oom_evicts_within_remaining_net_and_retries() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let victim_gate = vec![101u8; 64];
    let victim_up = vec![102u8; 64];
    let victim_down = vec![103u8; 64];
    state
        .qwen35_with_decode_reuse_score_override_for_test(1, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&victim_gate],
                &[&victim_up],
                &[&victim_down],
                &[1.0],
                Some(40),
                &[1],
            )
        })
        .expect("admit OOM victim bundle");

    let gate = vec![111u8; 64];
    let up = vec![112u8; 64];
    let down = vec![113u8; 64];
    assert!(state
        .preload_resident_q4k_weight_slice(&gate)
        .expect("preload incoming gate"));
    assert!(state
        .preload_resident_q4k_weight_slice(&up)
        .expect("preload incoming up"));
    let before = cache_snapshot();
    state.inject_qwen35_resident_alloc_ooms_for_test(1);

    state
        .qwen35_with_decode_reuse_score_override_for_test(4, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&gate],
                &[&up],
                &[&down],
                &[1.0],
                Some(41),
                &[2],
            )
        })
        .expect("retry profitable single-role admission after injected OOM");

    assert!(!state.q4k_weight_slice_is_resident(&victim_gate));
    assert!(!state.q4k_weight_slice_is_resident(&victim_up));
    assert!(!state.q4k_weight_slice_is_resident(&victim_down));
    assert!(state.q4k_weight_slice_is_resident(&gate));
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_evictions, 1);
    assert_eq!(delta.evicted_bytes, 192);
    assert_eq!(delta.bundle_admissions, 1);
    assert_eq!(delta.admitted_bytes, 64);
    assert_eq!(delta.h2d_bytes, 64);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_slab_oom_without_profitable_victim_budget_skips_admission() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let victim_gate = vec![121u8; 64];
    let victim_up = vec![122u8; 64];
    let victim_down = vec![123u8; 64];
    state
        .qwen35_with_decode_reuse_score_override_for_test(1, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&victim_gate],
                &[&victim_up],
                &[&victim_down],
                &[1.0],
                Some(50),
                &[1],
            )
        })
        .expect("admit over-budget OOM victim bundle");

    let gate = vec![131u8; 32];
    let up = vec![132u8; 32];
    let down = vec![133u8; 32];
    state.inject_qwen35_resident_alloc_ooms_for_test(1);
    state
        .qwen35_with_decode_reuse_score_override_for_test(1, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&gate],
                &[&up],
                &[&down],
                &[1.0],
                Some(51),
                &[2],
            )
        })
        .expect("insufficient remaining net skips optional admission");
    assert!(state.q4k_weight_slice_is_resident(&victim_gate));
    assert!(state.q4k_weight_slice_is_resident(&victim_up));
    assert!(state.q4k_weight_slice_is_resident(&victim_down));
    assert!(!state.q4k_weight_slice_is_resident(&gate));
    assert!(!state.q4k_weight_slice_is_resident(&up));
    assert!(!state.q4k_weight_slice_is_resident(&down));
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 192);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_current_token_protected_key_is_not_evicted() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 64;
    let current = vec![71u8; 64];
    assert!(state
        .preload_resident_q4k_weight_slice(&current)
        .expect("preload current-token role"));
    let protected = [q4k_resident_key(&current)]
        .into_iter()
        .collect::<HashSet<_>>();

    state
        .evict_resident_q4k_until_protecting(64, &protected)
        .expect("protected eviction attempt");

    assert!(state.q4k_weight_slice_is_resident(&current));
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_regular_q4k_oom_evicts_bundle_before_global_offload() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let victim_gate = vec![141u8; 64];
    let victim_up = vec![142u8; 64];
    let victim_down = vec![143u8; 64];
    state
        .qwen35_with_decode_reuse_score_override_for_test(2, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&victim_gate],
                &[&victim_up],
                &[&victim_down],
                &[1.0],
                Some(60),
                &[1],
            )
        })
        .expect("admit OOM victim bundle");
    let keeper = vec![144u8; 64];
    assert!(state
        .preload_resident_q4k_weight_slice(&keeper)
        .expect("preload unowned keeper"));
    let incoming = vec![145u8; 128];
    let before = cache_snapshot();
    state.inject_qwen35_resident_alloc_ooms_for_test(1);

    state
        .resident_q4k_weights_ptr_pinned(&incoming)
        .expect("retry regular Q4K allocation after evicting one bundle");

    assert!(!state.q4k_weight_slice_is_resident(&victim_gate));
    assert!(!state.q4k_weight_slice_is_resident(&victim_up));
    assert!(!state.q4k_weight_slice_is_resident(&victim_down));
    assert!(state.q4k_weight_slice_is_resident(&keeper));
    assert!(state.q4k_weight_slice_is_resident(&incoming));
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_evictions, 1);
    assert_eq!(delta.evicted_bytes, 192);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_batch_q4k_oom_evicts_bundle_before_global_offload() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let victim_gate = vec![151u8; 64];
    let victim_up = vec![152u8; 64];
    let victim_down = vec![153u8; 64];
    state
        .qwen35_with_decode_reuse_score_override_for_test(2, |state| {
            state.qwen35_decode_selected_q4k_ptrs_for_test(
                &[&victim_gate],
                &[&victim_up],
                &[&victim_down],
                &[1.0],
                Some(61),
                &[1],
            )
        })
        .expect("admit batch OOM victim bundle");
    let keeper = vec![154u8; 64];
    assert!(state
        .preload_resident_q4k_weight_slice(&keeper)
        .expect("preload batch unowned keeper"));
    let incoming_gate = vec![155u8; 64];
    let incoming_up = vec![156u8; 64];
    let incoming = [&incoming_gate[..], &incoming_up[..]];
    let incoming_groups = [&incoming[..]];
    let before = cache_snapshot();
    state.inject_qwen35_resident_alloc_ooms_for_test(1);

    state
        .batch_resident_q4k_slot_misses_many(&incoming_groups, &HashMap::new())
        .expect("retry batch Q4K allocation after evicting one bundle");

    assert!(!state.q4k_weight_slice_is_resident(&victim_gate));
    assert!(!state.q4k_weight_slice_is_resident(&victim_up));
    assert!(!state.q4k_weight_slice_is_resident(&victim_down));
    assert!(state.q4k_weight_slice_is_resident(&keeper));
    assert!(state.q4k_weight_slice_is_resident(&incoming_gate));
    assert!(state.q4k_weight_slice_is_resident(&incoming_up));
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_evictions, 1);
    assert_eq!(delta.evicted_bytes, 192);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_no_layer_id_keeps_legacy_resident_promotion() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 1024;
    let gate = vec![81u8; 64];
    let up = vec![82u8; 64];
    let down = vec![83u8; 64];

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(&[&gate], &[&up], &[&down], &[1.0], None, &[])
        .expect("legacy anonymous resident promotion");

    assert!(state.q4k_weight_slice_is_resident(&gate));
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (0, 0, 0)
    );
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_no_layer_observes_selected_ids_once_and_excludes_shared_tail() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 0;
    let gate = vec![84u8; 64];
    let up = vec![85u8; 64];
    let down = vec![86u8; 64];
    let shared_gate = vec![87u8; 64];
    let shared_up = vec![88u8; 64];
    let shared_down = vec![89u8; 64];
    let before = cache_snapshot();

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &[&gate, &gate, &shared_gate],
            &[&up, &up, &shared_up],
            &[&down, &down, &shared_down],
            &[0.4, 0.6, 1.0],
            None,
            &[7, 7],
        )
        .expect("no-layer selected lookup with shared tail");

    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (0, 0, 0)
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 1);
    assert_eq!(delta.bundle_hits, 0);
    assert_eq!(delta.bundle_partial_hits, 0);
    assert_eq!(delta.bundle_misses, 1);
    assert_eq!(delta.bundle_admissions, 0);
    assert_eq!(delta.h2d_bytes, 192);
    assert_eq!(delta.temp_h2d_bytes, 192);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_layered_selected_policy_preserves_shared_hot_promotion() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 1024;
    let selected_gate = vec![101u8; 256];
    let selected_up = vec![102u8; 256];
    let selected_down = vec![103u8; 256];
    let shared_gate = vec![111u8; 256];
    let shared_up = vec![112u8; 256];
    let shared_down = vec![113u8; 256];

    state
        .qwen35_decode_selected_q4k_ptrs_for_test(
            &[&selected_gate, &shared_gate],
            &[&selected_up, &shared_up],
            &[&selected_down, &shared_down],
            &[0.75, 1.0],
            Some(9),
            &[23],
        )
        .expect("selected policy with always-on shared slot");

    assert!(!state.q4k_weight_slice_is_resident(&selected_gate));
    assert!(!state.q4k_weight_slice_is_resident(&selected_up));
    assert!(!state.q4k_weight_slice_is_resident(&selected_down));
    assert!(state.q4k_weight_slice_is_resident(&shared_gate));
    assert!(state.q4k_weight_slice_is_resident(&shared_up));
    assert!(state.q4k_weight_slice_is_resident(&shared_down));
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 0);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_cache_clear_records_payload_eviction() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![91u8; 64];
    let up = vec![92u8; 64];
    let down = vec![93u8; 64];
    for _ in 0..3 {
        state
            .qwen35_decode_selected_q4k_ptrs_for_test(
                &[&gate],
                &[&up],
                &[&down],
                &[1.0],
                Some(8),
                &[19],
            )
            .expect("admit bundle before clear");
    }
    let before_clear = cache_snapshot();
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 192);

    state
        .clear_resident_q4k_cache()
        .expect("clear resident Q4K cache");

    let after_clear = cache_snapshot();
    let eviction_delta = after_clear
        .expert_bundles
        .delta(before_clear.expert_bundles);
    assert_eq!(eviction_delta.bundle_evictions, 1);
    assert_eq!(eviction_delta.evicted_bytes, 192);
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 0);
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (0, 0, 0)
    );
    assert_eq!(
        after_clear.resident_payload_bytes + 192,
        before_clear.resident_payload_bytes
    );
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_qwen35_bundle_lru_eviction_records_logical_payload() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 4096;
    let gate = vec![121u8; 64];
    let up = vec![122u8; 64];
    let down = vec![123u8; 64];
    for _ in 0..3 {
        state
            .qwen35_decode_selected_q4k_ptrs_for_test(
                &[&gate],
                &[&up],
                &[&down],
                &[1.0],
                Some(10),
                &[29],
            )
            .expect("admit bundle before LRU eviction");
    }
    assert_eq!(state.resident_q4k_bytes, 576);
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 192);
    state.resident_q4k_limit = 576;
    let before_eviction = cache_snapshot();

    state
        .evict_resident_q4k_until(576)
        .expect("evict admitted bundle slab");

    let after_eviction = cache_snapshot();
    let eviction_delta = after_eviction
        .expert_bundles
        .delta(before_eviction.expert_bundles);
    assert_eq!(eviction_delta.bundle_evictions, 1);
    assert_eq!(eviction_delta.evicted_bytes, 192);
    assert_eq!(state.qwen35_q2q3_resident_payload_bytes, 0);
    assert_eq!(
        after_eviction.resident_payload_bytes + 192,
        before_eviction.resident_payload_bytes
    );
}

#[test]
fn qwen35_decode_selected_direct_h2d_counts_payload_and_temp_separately() {
    let _guard = runtime_test_lock();
    let before = cache_snapshot();

    cache_stats().record_expert_bundle_h2d(288, false);
    cache_stats().record_expert_bundle_h2d(123, true);

    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.h2d_bytes, 411);
    assert_eq!(delta.temp_h2d_bytes, 123);
}

#[test]
fn qwen35_decode_selected_bundle_snapshot_stays_coherent_during_concurrent_records() {
    const LOOKUP_WRITERS: usize = 3;
    const H2D_WRITERS: usize = 2;
    const RECORDS_PER_WRITER: u64 = 20_000;

    let stats = std::sync::Arc::new(jit::CudaCacheStats::default());
    let writer_count = LOOKUP_WRITERS + H2D_WRITERS;
    let start = std::sync::Arc::new(std::sync::Barrier::new(writer_count + 1));
    let active_writers = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(writer_count));

    let reader_stats = std::sync::Arc::clone(&stats);
    let reader_start = std::sync::Arc::clone(&start);
    let reader_active_writers = std::sync::Arc::clone(&active_writers);
    let reader = std::thread::spawn(move || {
        reader_start.wait();
        while reader_active_writers.load(std::sync::atomic::Ordering::Acquire) != 0 {
            let snapshot = reader_stats.expert_bundles();
            assert_eq!(
                snapshot.bundle_hits + snapshot.bundle_partial_hits + snapshot.bundle_misses,
                snapshot.bundle_lookups
            );
            assert_eq!(snapshot.h2d_bytes, snapshot.temp_h2d_bytes);
            std::thread::yield_now();
        }
        let snapshot = reader_stats.expert_bundles();
        assert_eq!(
            snapshot.bundle_hits + snapshot.bundle_partial_hits + snapshot.bundle_misses,
            snapshot.bundle_lookups
        );
        assert_eq!(snapshot.h2d_bytes, snapshot.temp_h2d_bytes);
    });

    let mut writers = Vec::with_capacity(writer_count);
    for writer_index in 0..LOOKUP_WRITERS {
        let writer_stats = std::sync::Arc::clone(&stats);
        let writer_start = std::sync::Arc::clone(&start);
        let writer_active_writers = std::sync::Arc::clone(&active_writers);
        writers.push(std::thread::spawn(move || {
            let residency = match writer_index {
                0 => rnb_memory::ExpertBundleResidency::Full,
                1 => rnb_memory::ExpertBundleResidency::Partial,
                _ => rnb_memory::ExpertBundleResidency::Miss,
            };
            let mut delta = rnb_memory::ExpertBundleCacheStats::default();
            delta.record_lookup(residency);
            writer_start.wait();
            for _ in 0..RECORDS_PER_WRITER {
                writer_stats.record_expert_bundles(delta);
            }
            writer_active_writers.fetch_sub(1, std::sync::atomic::Ordering::Release);
        }));
    }
    for _ in 0..H2D_WRITERS {
        let writer_stats = std::sync::Arc::clone(&stats);
        let writer_start = std::sync::Arc::clone(&start);
        let writer_active_writers = std::sync::Arc::clone(&active_writers);
        writers.push(std::thread::spawn(move || {
            writer_start.wait();
            for _ in 0..RECORDS_PER_WRITER {
                writer_stats.record_expert_bundle_h2d(1, true);
            }
            writer_active_writers.fetch_sub(1, std::sync::atomic::Ordering::Release);
        }));
    }

    for writer in writers {
        writer.join().expect("bundle stats writer");
    }
    reader.join().expect("bundle stats snapshot reader");

    let snapshot = stats.expert_bundles();
    assert_eq!(
        snapshot.bundle_lookups,
        LOOKUP_WRITERS as u64 * RECORDS_PER_WRITER
    );
    assert_eq!(snapshot.bundle_hits, RECORDS_PER_WRITER);
    assert_eq!(snapshot.bundle_partial_hits, RECORDS_PER_WRITER);
    assert_eq!(snapshot.bundle_misses, RECORDS_PER_WRITER);
    assert_eq!(snapshot.h2d_bytes, H2D_WRITERS as u64 * RECORDS_PER_WRITER);
    assert_eq!(snapshot.temp_h2d_bytes, snapshot.h2d_bytes);
}

#[test]
fn qwen35_mixed_temp_slot_upload_plan_overlaps_down_only_misses() {
    let resident_gate = vec![1u8; 10];
    let gate_up_shared = vec![2u8; 20];
    let up_only = vec![3u8; 30];
    let down_only = vec![4u8; 40];
    let gate = [&resident_gate[..], &gate_up_shared[..]];
    let up = [&up_only[..]];
    let down = [&gate_up_shared[..], &down_only[..]];
    let mut resident_keys = std::collections::HashSet::new();
    resident_keys.insert(q4k_resident_key(&resident_gate));

    let (plan, bytes) = qwen35_mixed_temp_slot_upload_plan(&gate, &up, &down, &resident_keys, true);

    assert_eq!(
        bytes,
        gate_up_shared.len() + up_only.len() + down_only.len()
    );
    assert_eq!(plan.len(), 3);
    assert_eq!(plan[0].key, q4k_resident_key(&gate_up_shared));
    assert_eq!(plan[0].stream, Qwen35TempUploadStream::Main);
    assert_eq!(plan[1].key, q4k_resident_key(&up_only));
    assert_eq!(plan[1].stream, Qwen35TempUploadStream::Main);
    assert_eq!(plan[2].key, q4k_resident_key(&down_only));
    assert_eq!(plan[2].stream, Qwen35TempUploadStream::Copy);
}

#[test]
fn qwen35_mixed_temp_slot_upload_plan_keeps_all_misses_on_main_without_overlap() {
    let gate0 = vec![1u8; 10];
    let up0 = vec![2u8; 20];
    let down0 = vec![3u8; 30];
    let gate = [&gate0[..]];
    let up = [&up0[..]];
    let down = [&down0[..]];
    let resident_keys = std::collections::HashSet::new();

    let (plan, bytes) =
        qwen35_mixed_temp_slot_upload_plan(&gate, &up, &down, &resident_keys, false);

    assert_eq!(bytes, gate0.len() + up0.len() + down0.len());
    assert_eq!(plan.len(), 3);
    assert!(plan
        .iter()
        .all(|entry| entry.stream == Qwen35TempUploadStream::Main));
}

#[test]
fn qwen35_moe_layer_limit_caps_implicit_cache_to_one_layer() {
    let configured_limit = 8192;
    let layer_bytes = 512;

    let implicit_limit = qwen35_moe_layer_effective_limit(configured_limit, layer_bytes, false);
    let explicit_limit = qwen35_moe_layer_effective_limit(configured_limit, layer_bytes, true);

    assert_eq!(implicit_limit, layer_bytes);
    assert_eq!(explicit_limit, configured_limit);
}

#[test]
fn q4k_resident_default_reserve_adds_mtp_workspace_budget() {
    assert_eq!(device_residency_default_reserve_mib(4096, false), 512);
    assert_eq!(device_residency_default_reserve_mib(4096, true), 1536);
    assert_eq!(device_residency_default_reserve_mib(8192, false), 512);
    assert_eq!(device_residency_default_reserve_mib(8192, true), 2560);
    assert_eq!(device_residency_default_reserve_mib(11917, false), 768);
    assert_eq!(device_residency_default_reserve_mib(11917, true), 3840);
    assert_eq!(device_residency_default_reserve_mib(16384, true), 5120);
}

#[test]
fn q4k_resident_configured_reserve_honors_env_override() {
    let _guard = EnvVarGuard::set("RNB_CUDA_Q4K_CACHE_RESERVE_MB", "1536");
    let _prefill_moe_guard = EnvVarGuard::remove("RNB_CUDA_PREFILL_MOE");

    assert_eq!(
        device_residency_configured_reserve_mib(10 * 1024, false).unwrap(),
        1536
    );
}

#[test]
fn q4k_resident_mtp_slot_cache_cap_scales_with_vram() {
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(4096), 1280);
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(8192), 2560);
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(10239), 3072);
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(11917), 3584);
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(16384), 4096);
    assert_eq!(q4k_resident_mtp_slot_cache_cap_mib(24576), 4096);
}

#[test]
fn q4k_resident_target_decode_cache_cap_scales_with_vram() {
    assert_eq!(q4k_resident_target_decode_cache_cap_mib(8192), 6144);
    assert_eq!(q4k_resident_target_decode_cache_cap_mib(11917), 8960);
    assert_eq!(q4k_resident_target_decode_cache_cap_mib(16384), 12 * 1024);
    assert_eq!(q4k_resident_target_decode_cache_cap_mib(24576), 12 * 1024);
}

#[test]
fn q4k_resident_nemotron_decode_cache_cap_scales_with_vram() {
    assert_eq!(q4k_resident_nemotron_decode_cache_cap_mib(4096), 2048);
    assert_eq!(q4k_resident_nemotron_decode_cache_cap_mib(8192), 4096);
    assert_eq!(q4k_resident_nemotron_decode_cache_cap_mib(10239), 6144);
    assert_eq!(q4k_resident_nemotron_decode_cache_cap_mib(11917), 6144);
}

#[test]
fn q4k_resident_auto_cache_cap_offloads_on_low_vram() {
    assert_eq!(q4k_resident_auto_cache_cap_mib(11917, false), Some(3584));
    assert_eq!(q4k_resident_auto_cache_cap_mib(11917, true), Some(3584));
    assert_eq!(q4k_resident_auto_cache_cap_mib(16384, false), None);
    assert_eq!(q4k_resident_auto_cache_cap_mib(16384, true), Some(4096));
    assert_eq!(q4k_resident_auto_cache_cap_mib(24576, false), None);
    assert_eq!(q4k_resident_auto_cache_cap_mib(24576, true), Some(4096));
}

#[test]
fn qwen35_decode_hot_resident_budget_scales_with_resident_limit() {
    assert_eq!(
        qwen35_decode_hot_resident_default_budget_bytes(1792 * 1024 * 1024),
        1792 * 1024 * 1024 / 320
    );
    assert_eq!(
        qwen35_decode_hot_resident_default_budget_bytes(8192 * 1024 * 1024),
        8192 * 1024 * 1024 / 320
    );
    assert_eq!(
        qwen35_decode_hot_resident_default_budget_bytes(16 * 1024 * 1024 * 1024),
        32 * 1024 * 1024
    );
}

#[test]
fn qwen35_decode_all_resident_touch_hits_stays_opt_out() {
    let _lock = runtime_test_lock();
    let key = "RNB_CUDA_QWEN35_DECODE_ALL_RESIDENT_SKIP_TOUCH";
    let _guard = EnvVarGuard::remove(key);
    assert!(qwen35_decode_all_resident_touch_hits_enabled());

    let _skip = EnvVarGuard::set(key, "1");
    assert!(!qwen35_decode_all_resident_touch_hits_enabled());
}

#[test]
fn q4k_residency_candidates_record_reuse_counts() {
    let a = vec![1u8; 32];
    let b = vec![2u8; 48];
    let mut unique = HashMap::new();
    let mut counts = HashMap::new();
    unique.insert(q4k_resident_key(&a), &a[..]);
    unique.insert(q4k_resident_key(&b), &b[..]);
    counts.insert(q4k_resident_key(&a), 3);
    counts.insert(q4k_resident_key(&b), 1);

    let candidates = q4k_residency_candidates(&unique, &counts);

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates.iter().map(|c| c.bytes()).sum::<u64>(), 80);
    assert!(candidates
        .iter()
        .any(|c| c.bytes() == 32 && c.reuse_weight() == 3));
    assert!(candidates
        .iter()
        .any(|c| c.bytes() == 48 && c.reuse_weight() == 1));
}

fn qwen35_gdn_prefill_chain_shape(seq_len: usize) -> crate::GdnPrefillChainShape {
    crate::GdnPrefillChainShape {
        seq_len,
        hidden_dim: 2048,
        d_inner: 4096,
        d_state: 128,
        n_group: 16,
        dt_rank: 32,
        conv_kernel: 4,
        conv_state_len: 3 * (4096 + 2 * 16 * 128),
        delta_state_len: 4096 * 128,
    }
}

fn small_gdn_prefill_chain_shape(seq_len: usize) -> crate::GdnPrefillChainShape {
    crate::GdnPrefillChainShape {
        seq_len,
        hidden_dim: 64,
        d_inner: 8,
        d_state: 128,
        n_group: 2,
        dt_rank: 2,
        conv_kernel: 4,
        conv_state_len: 3 * (8 + 2 * 2 * 128),
        delta_state_len: 8 * 128,
    }
}

fn q4k_gdn_prefill_chain_shape(seq_len: usize) -> crate::GdnPrefillChainShape {
    crate::GdnPrefillChainShape {
        seq_len,
        hidden_dim: 512,
        d_inner: 256,
        d_state: 128,
        n_group: 2,
        dt_rank: 2,
        conv_kernel: 4,
        conv_state_len: 3 * (256 + 2 * 2 * 128),
        delta_state_len: 256 * 128,
    }
}

#[test]
fn gdn_prefill_chain_shape_accepts_qwen35_dimensions() {
    let shape = qwen35_gdn_prefill_chain_shape(32);

    crate::validate_gdn_prefill_chain_shape(&shape).expect("valid Qwen35 GDN chain shape");
}

#[test]
fn gdn_prefill_chain_dims_match_qwen35_layout() {
    let shape = qwen35_gdn_prefill_chain_shape(32);

    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive Qwen35 chain dims");

    assert_eq!(dims.num_v_heads, 32);
    assert_eq!(dims.num_k_heads, 16);
    assert_eq!(dims.head_k_dim, 128);
    assert_eq!(dims.head_v_dim, 128);
    assert_eq!(dims.q_dim, 2048);
    assert_eq!(dims.k_dim, 2048);
    assert_eq!(dims.v_dim, 4096);
    assert_eq!(dims.conv_channels, 8192);
}

#[test]
fn gdn_prefill_chain_shape_rejects_zero_seq_len() {
    let shape = qwen35_gdn_prefill_chain_shape(0);

    let err = crate::validate_gdn_prefill_chain_shape(&shape).expect_err("zero seq_len rejects");
    assert!(err.contains("seq_len"));
}

#[test]
fn gdn_prefill_chain_shape_rejects_state_length_mismatch() {
    let mut shape = qwen35_gdn_prefill_chain_shape(32);
    shape.conv_state_len -= 1;

    let err =
        crate::validate_gdn_prefill_chain_shape(&shape).expect_err("bad conv state len rejects");
    assert!(err.contains("conv_state_len"));
}

#[test]
fn gdn_prefill_chain_forced_selects_q4k_device_chain() {
    let shape = qwen35_gdn_prefill_chain_shape(32);

    assert_eq!(
        crate::plan_gdn_prefill_chain_for_test(&shape, true).expect("forced chain policy"),
        crate::GdnPrefillChainPlan::Q4KDeviceChain
    );
}

#[test]
fn gdn_prefill_chain_env_policy_defaults_on_and_allows_opt_out() {
    let _guard = runtime_test_lock();
    let shape = qwen35_gdn_prefill_chain_shape(32);
    unsafe {
        std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
    }
    assert_eq!(
        crate::plan_gdn_prefill_chain(&shape).expect("default chain policy"),
        crate::GdnPrefillChainPlan::Q4KDeviceChain
    );

    unsafe {
        std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "0");
    }
    assert_eq!(
        crate::plan_gdn_prefill_chain(&shape).expect("explicit off chain policy"),
        crate::GdnPrefillChainPlan::Disabled
    );

    unsafe {
        std::env::set_var("RNB_CUDA_GDN_PREFILL_CHAIN", "1");
    }
    assert_eq!(
        crate::plan_gdn_prefill_chain(&shape).expect("enabled chain policy"),
        crate::GdnPrefillChainPlan::Q4KDeviceChain
    );

    unsafe {
        std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
    }
}

#[test]
fn gdn_prefill_chain_conv_input_tracks_prefix_and_final_state() {
    let shape = small_gdn_prefill_chain_shape(3);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let conv_state = (0..shape.conv_state_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let qkv_rows = (0..shape.seq_len * dims.conv_channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.046875)
        .collect::<Vec<_>>();

    let (conv_input, final_state) =
        crate::build_gdn_prefill_chain_conv_input_for_test(&shape, &conv_state, &qkv_rows)
            .expect("build chain conv input");
    let prefix_state =
        crate::gdn_prefill_chain_conv_state_after_prefix_for_test(&shape, &conv_input, 2)
            .expect("prefix chain conv state");

    let conv_state_len = shape.conv_state_len;
    assert_eq!(&conv_input[..conv_state_len], conv_state.as_slice());
    assert_eq!(&conv_input[conv_state_len..], qkv_rows.as_slice());
    assert_eq!(
        final_state,
        conv_input[shape.seq_len * dims.conv_channels
            ..shape.seq_len * dims.conv_channels + conv_state_len]
            .to_vec()
    );
    assert_eq!(
        prefix_state,
        conv_input[2 * dims.conv_channels..2 * dims.conv_channels + conv_state_len].to_vec()
    );
}

#[test]
fn gdn_prefill_chain_conv1d_stage_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let shape = small_gdn_prefill_chain_shape(3);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let conv_state = (0..shape.conv_state_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let qkv_rows = (0..shape.seq_len * dims.conv_channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.046875)
        .collect::<Vec<_>>();
    let kernel = (0..shape.conv_kernel * dims.conv_channels)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.00244140625)
        .collect::<Vec<_>>();
    let (conv_input, _final_state) =
        crate::build_gdn_prefill_chain_conv_input_for_test(&shape, &conv_state, &qkv_rows)
            .expect("build chain conv input");

    let actual = match ssm_conv1d_silu(
        &conv_input,
        &kernel,
        shape.seq_len,
        dims.conv_channels,
        shape.conv_kernel,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN chain conv1d parity test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN chain conv1d failed: {err}"),
    };

    let mut expected = vec![0.0f32; shape.seq_len * dims.conv_channels];
    for token_idx in 0..shape.seq_len {
        for channel_idx in 0..dims.conv_channels {
            let mut sum = 0.0f32;
            for k in 0..shape.conv_kernel {
                sum += conv_input[(token_idx + k) * dims.conv_channels + channel_idx]
                    * kernel[k * dims.conv_channels + channel_idx];
            }
            expected[token_idx * dims.conv_channels + channel_idx] = sum / (1.0 + (-sum).exp());
        }
    }

    assert_close_rows("GDN chain conv1d", &actual, &expected, 2.0e-5);
}

#[test]
fn gdn_prefill_chain_delta_stage_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");
    let shape = small_gdn_prefill_chain_shape(3);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let mut state = (0..shape.delta_state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.0029296875)
        .collect::<Vec<_>>();
    let mut expected_state = state.clone();
    let q = (0..shape.seq_len * dims.num_v_heads * dims.head_k_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.01171875)
        .collect::<Vec<_>>();
    let k = (0..shape.seq_len * dims.num_v_heads * dims.head_k_dim)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.009765625)
        .collect::<Vec<_>>();
    let v = (0..shape.seq_len * dims.num_v_heads * dims.head_v_dim)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.02734375)
        .collect::<Vec<_>>();
    let gate = (0..shape.seq_len * dims.num_v_heads)
        .map(|i| -0.015625 * (i as f32 + 1.0))
        .collect::<Vec<_>>();
    let beta = (0..shape.seq_len * dims.num_v_heads)
        .map(|i| 0.125 + 0.03125 * i as f32)
        .collect::<Vec<_>>();

    let expected = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        shape.seq_len,
        dims.num_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
    );
    let actual = match delta_net_prefill(
        &mut state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        shape.seq_len,
        dims.num_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN chain delta parity test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN chain delta failed: {err}"),
    };

    assert_close_rows("GDN chain delta output", &actual, &expected, 1.0e-3);
    assert_close_rows("GDN chain delta state", &state, &expected_state, 1.0e-3);
}

#[test]
fn gdn_prefill_chain_delta_snapshots_restore_prefix_states() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");
    let shape = small_gdn_prefill_chain_shape(3);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let snapshot_after_tokens = [1usize, 2usize];
    let mut state = (0..shape.delta_state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.0029296875)
        .collect::<Vec<_>>();
    let initial_state = state.clone();
    let q = (0..shape.seq_len * dims.num_v_heads * dims.head_k_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.01171875)
        .collect::<Vec<_>>();
    let k = (0..shape.seq_len * dims.num_v_heads * dims.head_k_dim)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.009765625)
        .collect::<Vec<_>>();
    let v = (0..shape.seq_len * dims.num_v_heads * dims.head_v_dim)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.02734375)
        .collect::<Vec<_>>();
    let gate = (0..shape.seq_len * dims.num_v_heads)
        .map(|i| -0.015625 * (i as f32 + 1.0))
        .collect::<Vec<_>>();
    let beta = (0..shape.seq_len * dims.num_v_heads)
        .map(|i| 0.125 + 0.03125 * i as f32)
        .collect::<Vec<_>>();
    let expected_states = snapshot_after_tokens
        .iter()
        .map(|&tokens| {
            let mut expected_state = initial_state.clone();
            let _ = cpu_delta_net_prefill_reference(
                &mut expected_state,
                &q[..tokens * dims.num_v_heads * dims.head_k_dim],
                &k[..tokens * dims.num_v_heads * dims.head_k_dim],
                &v[..tokens * dims.num_v_heads * dims.head_v_dim],
                &gate[..tokens * dims.num_v_heads],
                &beta[..tokens * dims.num_v_heads],
                tokens,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            );
            expected_state
        })
        .collect::<Vec<_>>();

    let (_actual, snapshots) = match delta_net_prefill_resident_snapshots(
        &mut state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        shape.seq_len,
        dims.num_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
        &snapshot_after_tokens,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN chain delta snapshot parity test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN chain delta snapshots failed: {err}"),
    };

    assert_eq!(snapshots.len(), snapshot_after_tokens.len());
    for (snapshot, expected_state) in snapshots.iter().zip(expected_states.iter()) {
        assert!(
            restore_delta_state_cache(&mut state, snapshot).expect("restore prefix snapshot"),
            "restore should find resident delta state"
        );
        assert!(
            sync_delta_state_cache(&mut state).expect("sync prefix snapshot"),
            "sync should find resident delta state"
        );
        assert_close_rows(
            "GDN chain delta prefix state",
            &state,
            expected_state,
            1.0e-3,
        );
    }

    for snapshot in snapshots {
        free_delta_state_snapshot(snapshot).expect("free prefix snapshot");
    }
}

#[test]
fn gdn_prefill_chain_gated_norm_ssm_out_stage_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let shape = small_gdn_prefill_chain_shape(3);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let proj_rows = 5usize;
    let delta_out = (0..shape.seq_len * dims.v_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.015625)
        .collect::<Vec<_>>();
    let z = (0..shape.seq_len * dims.v_dim)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.01953125)
        .collect::<Vec<_>>();
    let norm = (0..dims.head_v_dim)
        .map(|i| 0.75 + (i as f32 % 9.0) * 0.03125)
        .collect::<Vec<_>>();
    let proj = (0..proj_rows * dims.v_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.0078125)
        .collect::<Vec<_>>();

    let actual = match gdn_gated_norm_silu_f32_gemm(
        &delta_out,
        &z,
        &norm,
        &proj,
        shape.seq_len,
        dims.head_v_dim,
        proj_rows,
        dims.v_dim,
        1.0e-5,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN chain gated norm+ssm_out parity test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN chain gated norm+ssm_out failed: {err}"),
    };

    let mut gated = vec![0.0f32; shape.seq_len * dims.v_dim];
    for row_idx in 0..shape.seq_len * dims.num_v_heads {
        let start = row_idx * dims.head_v_dim;
        let row = &delta_out[start..start + dims.head_v_dim];
        let inv =
            1.0 / (row.iter().map(|v| v * v).sum::<f32>() / dims.head_v_dim as f32 + 1.0e-5).sqrt();
        for dim in 0..dims.head_v_dim {
            let idx = start + dim;
            let z_value = z[idx];
            gated[idx] = row[dim] * inv * norm[dim] * (z_value / (1.0 + (-z_value).exp()));
        }
    }
    let mut expected = Vec::with_capacity(shape.seq_len * proj_rows);
    for token_idx in 0..shape.seq_len {
        let input = &gated[token_idx * dims.v_dim..(token_idx + 1) * dims.v_dim];
        expected.extend(cpu_f32_gemv_rows(&proj, proj_rows, dims.v_dim, input));
    }

    assert_close_rows("GDN chain gated norm+ssm_out", &actual, &expected, 2.0e-4);
}

#[test]
fn gdn_prefill_chain_q4k_execution_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");
    let shape = q4k_gdn_prefill_chain_shape(2);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let hidden_blocks = shape.hidden_dim / 256;
    let ssm_blocks = dims.v_dim / 256;
    let hidden = (0..shape.seq_len * shape.hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0078125)
        .collect::<Vec<_>>();
    let attn_norm = (0..shape.hidden_dim)
        .map(|i| 0.75 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, dims.conv_channels, hidden_blocks, 701)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, dims.v_dim, hidden_blocks, 709)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 719)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 727)
        .pop()
        .unwrap();
    let initial_conv_state = (0..shape.conv_state_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01171875)
        .collect::<Vec<_>>();
    let mut conv_state = initial_conv_state.clone();
    let conv_kernel = (0..shape.conv_kernel * dims.conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009765625)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25_f32, 0.125];
    let ssm_a = [-0.75_f32, -0.5];
    let initial_delta_state = (0..shape.delta_state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();
    let mut delta_state = initial_delta_state.clone();
    let mut expected_delta_state = initial_delta_state;
    let ssm_norm = (0..dims.head_v_dim)
        .map(|i| 0.5 + (i % 17) as f32 * 0.00390625)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, shape.hidden_dim, ssm_blocks, 733)
        .pop()
        .unwrap();
    let post_attn_norm = (0..shape.hidden_dim)
        .map(|i| 0.625 + (i % 11) as f32 * 0.005859375)
        .collect::<Vec<_>>();

    let actual = match crate::gdn_prefill_chain_q4k(crate::GdnPrefillChainQ4KRequest {
        shape,
        hidden: &hidden,
        hidden_device: None,
        attn_norm: &attn_norm,
        qkv_q4k: &qkv,
        qkv_quant: GGML_Q4_K,
        gate_q4k: &gate,
        gate_quant: GGML_Q4_K,
        alpha_q4k: &alpha,
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        beta_q4k: &beta,
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        conv_state: &mut conv_state,
        conv_kernel: &conv_kernel,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        delta_state: &mut delta_state,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &ssm_out,
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: shape.hidden_dim,
        ssm_out_cols: dims.v_dim,
        norm_eps: 1.0e-5,
        keep_host_output: true,
        keep_device_output: true,
        post_attn_norm: Some(&post_attn_norm),
        keep_device_moe_input: true,
    }) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN Q4K chain execution test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN Q4K chain execution failed: {err}"),
    };

    let mut qkv_rows = Vec::with_capacity(shape.seq_len * dims.conv_channels);
    let mut gate_rows = Vec::with_capacity(shape.seq_len * dims.v_dim);
    let mut alpha_rows = Vec::with_capacity(shape.seq_len * dims.num_v_heads);
    let mut beta_rows = Vec::with_capacity(shape.seq_len * dims.num_v_heads);
    for hidden in hidden.chunks_exact(shape.hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        qkv_rows.extend(cpu_q4k_gemv_rows(
            &qkv,
            dims.conv_channels,
            hidden_blocks,
            &normed,
        ));
        gate_rows.extend(cpu_q4k_gemv_rows(&gate, dims.v_dim, hidden_blocks, &normed));
        alpha_rows.extend(cpu_q4k_gemv_rows(
            &alpha,
            dims.num_v_heads,
            hidden_blocks,
            &normed,
        ));
        beta_rows.extend(cpu_q4k_gemv_rows(
            &beta,
            dims.num_v_heads,
            hidden_blocks,
            &normed,
        ));
    }
    let (conv_input, expected_conv_state) =
        crate::build_gdn_prefill_chain_conv_input_for_test(&shape, &initial_conv_state, &qkv_rows)
            .expect("build chain conv input");
    let mut conv_out = vec![0.0f32; shape.seq_len * dims.conv_channels];
    for token_idx in 0..shape.seq_len {
        for channel_idx in 0..dims.conv_channels {
            let mut sum = 0.0f32;
            for k in 0..shape.conv_kernel {
                sum += conv_input[(token_idx + k) * dims.conv_channels + channel_idx]
                    * conv_kernel[k * dims.conv_channels + channel_idx];
            }
            conv_out[token_idx * dims.conv_channels + channel_idx] = sum / (1.0 + (-sum).exp());
        }
    }

    let q_scale = 1.0 / (dims.head_k_dim as f32).sqrt();
    let mut q = vec![0.0f32; shape.seq_len * dims.num_v_heads * dims.head_k_dim];
    let mut k = vec![0.0f32; q.len()];
    let mut v = vec![0.0f32; shape.seq_len * dims.num_v_heads * dims.head_v_dim];
    let mut delta_gate = vec![0.0f32; shape.seq_len * dims.num_v_heads];
    let mut delta_beta = vec![0.0f32; shape.seq_len * dims.num_v_heads];
    for token_idx in 0..shape.seq_len {
        let row = &conv_out[token_idx * dims.conv_channels..(token_idx + 1) * dims.conv_channels];
        for k_head in 0..dims.num_k_heads {
            let q_src = &row[k_head * dims.head_k_dim..(k_head + 1) * dims.head_k_dim];
            let k_src = &row[dims.q_dim + k_head * dims.head_k_dim
                ..dims.q_dim + (k_head + 1) * dims.head_k_dim];
            let q_inv =
                1.0 / (q_src.iter().map(|value| value * value).sum::<f32>() + 1.0e-5).sqrt();
            let k_inv =
                1.0 / (k_src.iter().map(|value| value * value).sum::<f32>() + 1.0e-5).sqrt();
            for v_head in (k_head..dims.num_v_heads).step_by(dims.num_k_heads) {
                let out = (token_idx * dims.num_v_heads + v_head) * dims.head_k_dim;
                for dim in 0..dims.head_k_dim {
                    q[out + dim] = q_src[dim] * q_inv * q_scale;
                    k[out + dim] = k_src[dim] * k_inv;
                }
            }
        }
        for v_head in 0..dims.num_v_heads {
            let v_src = dims.q_dim + dims.k_dim + v_head * dims.head_v_dim;
            let v_out = (token_idx * dims.num_v_heads + v_head) * dims.head_v_dim;
            v[v_out..v_out + dims.head_v_dim].copy_from_slice(&row[v_src..v_src + dims.head_v_dim]);
            let gate_idx = token_idx * dims.num_v_heads + v_head;
            let biased = alpha_rows[gate_idx] + dt_bias[v_head];
            delta_gate[gate_idx] = (1.0 + biased.exp()).ln() * ssm_a[v_head];
            delta_beta[gate_idx] = 1.0 / (1.0 + (-beta_rows[gate_idx]).exp());
        }
    }
    let delta_out = cpu_delta_net_prefill_reference(
        &mut expected_delta_state,
        &q,
        &k,
        &v,
        &delta_gate,
        &delta_beta,
        shape.seq_len,
        dims.num_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
    );
    let mut gated = vec![0.0f32; shape.seq_len * dims.v_dim];
    for row_idx in 0..shape.seq_len * dims.num_v_heads {
        let start = row_idx * dims.head_v_dim;
        let row = &delta_out[start..start + dims.head_v_dim];
        let inv = 1.0
            / (row.iter().map(|value| value * value).sum::<f32>() / dims.head_v_dim as f32
                + 1.0e-5)
                .sqrt();
        for dim in 0..dims.head_v_dim {
            let idx = start + dim;
            let z = gate_rows[idx];
            gated[idx] = row[dim] * inv * ssm_norm[dim] * (z / (1.0 + (-z).exp()));
        }
    }
    let mut expected = Vec::with_capacity(shape.seq_len * shape.hidden_dim);
    for token_idx in 0..shape.seq_len {
        let input = &gated[token_idx * dims.v_dim..(token_idx + 1) * dims.v_dim];
        expected.extend(cpu_q4k_gemv_rows(
            &ssm_out,
            shape.hidden_dim,
            ssm_blocks,
            input,
        ));
    }

    assert_close_rows(
        "GDN Q4K chain ssm projection",
        &actual.ssm_projection,
        &expected,
        0.02,
    );
    assert_eq!(
        actual.ssm_projection_d2h_bytes,
        std::mem::size_of_val(expected.as_slice())
    );
    let (device_output_id, device_output_desc) = actual
        .device_output
        .expect("GDN Q4K chain should keep opt-in device output");
    assert_eq!(
        device_output_desc,
        rnb_backend_api::DeviceTensorDesc::new(
            shape.seq_len,
            shape.hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MambaOutput,
        )
    );
    let device_projection = crate::runtime::download_device_tensor_f32(device_output_id)
        .expect("download GDN device output");
    assert_close_rows(
        "GDN Q4K chain device ssm projection",
        &device_projection,
        &expected,
        0.02,
    );
    assert!(
        crate::runtime::release_device_tensor(device_output_id).expect("release GDN device output")
    );
    let expected_residual = hidden
        .iter()
        .zip(expected.iter())
        .map(|(hidden, ssm)| hidden + ssm)
        .collect::<Vec<_>>();
    let mut expected_moe_input = Vec::with_capacity(expected_residual.len());
    for row in expected_residual.chunks_exact(shape.hidden_dim) {
        expected_moe_input.extend(cpu_rms_norm(row, &post_attn_norm, 1.0e-5, false));
    }
    let (device_residual_id, device_residual_desc) = actual
        .device_residual
        .expect("GDN Q4K chain should keep opt-in residual carrier");
    assert_eq!(
        device_residual_desc,
        rnb_backend_api::DeviceTensorDesc::new(
            shape.seq_len,
            shape.hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Hidden,
        )
    );
    let device_residual = crate::runtime::download_device_tensor_f32(device_residual_id)
        .expect("download GDN residual carrier");
    assert_close_rows(
        "GDN Q4K chain device residual carrier",
        &device_residual,
        &expected_residual,
        0.02,
    );
    assert!(crate::runtime::release_device_tensor(device_residual_id)
        .expect("release GDN residual carrier"));
    let (device_moe_input_id, device_moe_input_desc) = actual
        .device_moe_input
        .expect("GDN Q4K chain should keep opt-in MoE input carrier");
    assert_eq!(
        device_moe_input_desc,
        rnb_backend_api::DeviceTensorDesc::new(
            shape.seq_len,
            shape.hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Normalized,
        )
    );
    let device_moe_input = crate::runtime::download_device_tensor_f32(device_moe_input_id)
        .expect("download GDN MoE input carrier");
    assert_close_rows(
        "GDN Q4K chain device MoE input carrier",
        &device_moe_input,
        &expected_moe_input,
        0.02,
    );
    assert!(crate::runtime::release_device_tensor(device_moe_input_id)
        .expect("release GDN MoE input carrier"));
    assert_close_rows(
        "GDN Q4K chain conv state",
        &conv_state,
        &expected_conv_state,
        1.0e-3,
    );
    assert_close_rows(
        "GDN Q4K chain delta state",
        &delta_state,
        &expected_delta_state,
        1.0e-3,
    );
    assert_eq!(
        actual.conv_state_d2h_bytes,
        std::mem::size_of_val(expected_conv_state.as_slice())
    );
    assert_eq!(
        actual.delta_state_d2h_bytes,
        std::mem::size_of_val(expected_delta_state.as_slice())
    );
}

#[test]
fn gdn_prefill_chain_q4k_accepts_device_moe_output_hidden_input() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");
    let shape = q4k_gdn_prefill_chain_shape(2);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let hidden_blocks = shape.hidden_dim / 256;
    let ssm_blocks = dims.v_dim / 256;
    let hidden = (0..shape.seq_len * shape.hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0078125)
        .collect::<Vec<_>>();
    let hidden_desc = rnb_backend_api::DeviceTensorDesc::new(
        shape.seq_len,
        shape.hidden_dim,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let hidden_id =
        crate::runtime::upload_device_tensor_f32(hidden_desc, &hidden).expect("upload hidden");
    let attn_norm = (0..shape.hidden_dim)
        .map(|i| 0.75 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, dims.conv_channels, hidden_blocks, 701)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, dims.v_dim, hidden_blocks, 709)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 719)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 727)
        .pop()
        .unwrap();
    let conv_kernel = (0..shape.conv_kernel * dims.conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009765625)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25_f32, 0.125];
    let ssm_a = [-0.75_f32, -0.5];
    let ssm_norm = (0..dims.head_v_dim)
        .map(|i| 0.5 + (i % 17) as f32 * 0.00390625)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, shape.hidden_dim, ssm_blocks, 733)
        .pop()
        .unwrap();

    let initial_conv_state = (0..shape.conv_state_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01171875)
        .collect::<Vec<_>>();
    let initial_delta_state = (0..shape.delta_state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();
    let mut host_conv_state = initial_conv_state.clone();
    let mut host_delta_state = initial_delta_state.clone();
    let expected = match crate::gdn_prefill_chain_q4k(crate::GdnPrefillChainQ4KRequest {
        shape,
        hidden: &hidden,
        hidden_device: None,
        attn_norm: &attn_norm,
        qkv_q4k: &qkv,
        qkv_quant: GGML_Q4_K,
        gate_q4k: &gate,
        gate_quant: GGML_Q4_K,
        alpha_q4k: &alpha,
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        beta_q4k: &beta,
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        conv_state: &mut host_conv_state,
        conv_kernel: &conv_kernel,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        delta_state: &mut host_delta_state,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &ssm_out,
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: shape.hidden_dim,
        ssm_out_cols: dims.v_dim,
        norm_eps: 1.0e-5,
        keep_host_output: true,
        keep_device_output: false,
        post_attn_norm: None,
        keep_device_moe_input: false,
    }) {
        Ok(output) => output,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN Q4K chain device hidden input test: {err}");
            let _ = crate::runtime::release_device_tensor(hidden_id);
            return;
        }
        Err(err) => panic!("CUDA GDN Q4K chain host reference failed: {err}"),
    };

    let mut device_conv_state = initial_conv_state;
    let mut device_delta_state = initial_delta_state;
    let actual = match crate::gdn_prefill_chain_q4k(crate::GdnPrefillChainQ4KRequest {
        shape,
        hidden: &[],
        hidden_device: Some((hidden_id, hidden_desc)),
        attn_norm: &attn_norm,
        qkv_q4k: &qkv,
        qkv_quant: GGML_Q4_K,
        gate_q4k: &gate,
        gate_quant: GGML_Q4_K,
        alpha_q4k: &alpha,
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        beta_q4k: &beta,
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        conv_state: &mut device_conv_state,
        conv_kernel: &conv_kernel,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        delta_state: &mut device_delta_state,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &ssm_out,
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: shape.hidden_dim,
        ssm_out_cols: dims.v_dim,
        norm_eps: 1.0e-5,
        keep_host_output: true,
        keep_device_output: false,
        post_attn_norm: None,
        keep_device_moe_input: false,
    }) {
        Ok(output) => output,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN Q4K chain device hidden input test: {err}");
            let _ = crate::runtime::release_device_tensor(hidden_id);
            return;
        }
        Err(err) => panic!("CUDA GDN Q4K chain device input failed: {err}"),
    };
    assert!(crate::runtime::release_device_tensor(hidden_id).expect("release uploaded hidden"));

    assert_close_rows(
        "GDN Q4K chain device hidden input projection",
        &actual.ssm_projection,
        &expected.ssm_projection,
        1.0e-5,
    );
    assert_close_rows(
        "GDN Q4K chain device hidden input conv state",
        &device_conv_state,
        &host_conv_state,
        1.0e-5,
    );
    assert_close_rows(
        "GDN Q4K chain device hidden input delta state",
        &device_delta_state,
        &host_delta_state,
        1.0e-5,
    );
}

#[test]
fn gdn_prefill_chain_q4k_can_skip_host_projection_for_device_moe_carriers() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");
    let shape = q4k_gdn_prefill_chain_shape(2);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let hidden_blocks = shape.hidden_dim / 256;
    let ssm_blocks = dims.v_dim / 256;
    let hidden = (0..shape.seq_len * shape.hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0078125)
        .collect::<Vec<_>>();
    let attn_norm = (0..shape.hidden_dim)
        .map(|i| 0.75 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, dims.conv_channels, hidden_blocks, 701)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, dims.v_dim, hidden_blocks, 709)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 719)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, dims.num_v_heads, hidden_blocks, 727)
        .pop()
        .unwrap();
    let conv_kernel = (0..shape.conv_kernel * dims.conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009765625)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25_f32, 0.125];
    let ssm_a = [-0.75_f32, -0.5];
    let ssm_norm = (0..dims.head_v_dim)
        .map(|i| 0.5 + (i % 17) as f32 * 0.00390625)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, shape.hidden_dim, ssm_blocks, 733)
        .pop()
        .unwrap();
    let post_attn_norm = (0..shape.hidden_dim)
        .map(|i| 0.625 + (i % 11) as f32 * 0.005859375)
        .collect::<Vec<_>>();
    let mut conv_state = (0..shape.conv_state_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01171875)
        .collect::<Vec<_>>();
    let mut delta_state = (0..shape.delta_state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();

    let actual = match crate::gdn_prefill_chain_q4k(crate::GdnPrefillChainQ4KRequest {
        shape,
        hidden: &hidden,
        hidden_device: None,
        attn_norm: &attn_norm,
        qkv_q4k: &qkv,
        qkv_quant: GGML_Q4_K,
        gate_q4k: &gate,
        gate_quant: GGML_Q4_K,
        alpha_q4k: &alpha,
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        beta_q4k: &beta,
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        conv_state: &mut conv_state,
        conv_kernel: &conv_kernel,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        delta_state: &mut delta_state,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &ssm_out,
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: shape.hidden_dim,
        ssm_out_cols: dims.v_dim,
        norm_eps: 1.0e-5,
        keep_host_output: false,
        keep_device_output: false,
        post_attn_norm: Some(&post_attn_norm),
        keep_device_moe_input: true,
    }) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping GDN Q4K chain host projection skip test: {err}");
            return;
        }
        Err(err) => panic!("CUDA GDN Q4K chain host projection skip failed: {err}"),
    };

    assert!(
        actual.ssm_projection.is_empty(),
        "host projection must stay empty when disabled"
    );
    assert_eq!(actual.ssm_projection_d2h_bytes, 0);
    assert!(actual.device_output.is_none());
    let (residual_id, residual_desc) = actual
        .device_residual
        .expect("GDN Q4K chain should still return residual carrier");
    assert_eq!(
        residual_desc.role(),
        rnb_backend_api::DeviceTensorRole::Hidden
    );
    assert!(crate::runtime::release_device_tensor(residual_id).expect("release residual carrier"));
    let (moe_input_id, moe_input_desc) = actual
        .device_moe_input
        .expect("GDN Q4K chain should still return MoE input carrier");
    assert_eq!(
        moe_input_desc.role(),
        rnb_backend_api::DeviceTensorRole::Normalized
    );
    assert!(crate::runtime::release_device_tensor(moe_input_id).expect("release MoE input carrier"));
}

#[test]
fn gdn_prefill_chain_q4k_rejects_bad_ssm_out_before_state_mutation() {
    let shape = q4k_gdn_prefill_chain_shape(1);
    let dims = crate::derive_gdn_prefill_chain_dims(&shape).expect("derive test dims");
    let hidden = vec![0.0f32; shape.seq_len * shape.hidden_dim];
    let attn_norm = vec![1.0f32; shape.hidden_dim];
    let mut conv_state = vec![0.25f32; shape.conv_state_len];
    let original_conv_state = conv_state.clone();
    let conv_kernel = vec![0.0f32; shape.conv_kernel * dims.conv_channels];
    let dt_bias = vec![0.0f32; dims.num_v_heads];
    let ssm_a = vec![-1.0f32; dims.num_v_heads];
    let mut delta_state = vec![0.5f32; shape.delta_state_len];
    let original_delta_state = delta_state.clone();
    let ssm_norm = vec![1.0f32; dims.head_v_dim];

    let err = crate::gdn_prefill_chain_q4k(crate::GdnPrefillChainQ4KRequest {
        shape,
        hidden: &hidden,
        hidden_device: None,
        attn_norm: &attn_norm,
        qkv_q4k: &[],
        qkv_quant: GGML_Q4_K,
        gate_q4k: &[],
        gate_quant: GGML_Q4_K,
        alpha_q4k: &[],
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        beta_q4k: &[],
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        conv_state: &mut conv_state,
        conv_kernel: &conv_kernel,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        delta_state: &mut delta_state,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &[],
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: shape.hidden_dim,
        ssm_out_cols: dims.v_dim,
        norm_eps: 1.0e-5,
        keep_host_output: true,
        keep_device_output: false,
        post_attn_norm: None,
        keep_device_moe_input: false,
    })
    .expect_err("invalid ssm_out must fail before launching chain");

    assert!(
        err.contains("GDN ssm_out"),
        "unexpected validation error: {err}"
    );
    assert_eq!(conv_state, original_conv_state);
    assert_eq!(delta_state, original_delta_state);
}

#[test]
fn cuda_driver_can_launch_minimal_kernel() {
    let _guard = runtime_test_lock();
    let Ok(out) = launch_smoke_add_one_for_test(41.0) else {
        eprintln!("skipping CUDA smoke kernel test: CUDA driver unavailable");
        return;
    };
    assert!(
        (out - 42.0).abs() < 0.001,
        "CUDA smoke kernel output mismatch: got {out}"
    );
}
#[test]
fn cuda_driver_can_replay_minimal_graph() {
    let _guard = runtime_test_lock();
    let Ok(out) = launch_smoke_graph_add_one_for_test(41.0) else {
        eprintln!("skipping CUDA graph smoke test: CUDA driver graph unavailable");
        return;
    };
    assert!(
        (out - 43.0).abs() < 1e-6,
        "CUDA graph smoke output mismatch: got {out}"
    );
}

fn iq4_xs_reference_row_dot(row: &[u8], input: &[f32]) -> f32 {
    const BLOCK_BYTES: usize = 136;
    const KVALUES_IQ4NL: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];

    row.chunks_exact(BLOCK_BYTES)
        .enumerate()
        .map(|(block_idx, block)| {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = &block[8..136];
            let input = &input[block_idx * 256..(block_idx + 1) * 256];
            let mut acc = 0.0f32;
            for ib in 0..8 {
                let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
                let high = (((scales_h >> (2 * ib)) & 0x03) as u8) << 4;
                let dl = d * ((low | high) as f32 - 32.0);
                let q = &qs[ib * 16..(ib + 1) * 16];
                let base = ib * 32;
                for j in 0..16 {
                    acc += dl * KVALUES_IQ4NL[(q[j] & 0x0f) as usize] * input[base + j];
                    acc += dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] * input[base + j + 16];
                }
            }
            acc
        })
        .sum()
}

#[test]
fn cuda_iq4_xs_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 3usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 136;
    let mut weights = vec![0u8; rows * row_bytes];
    for (i, byte) in weights.iter_mut().enumerate() {
        *byte = ((i * 37 + 19) & 0xff) as u8;
    }
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let off = row * row_bytes + block * 136;
            weights[off..off + 2].copy_from_slice(
                &half::f16::from_f32(0.0004 + row as f32 * 0.0001)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
    let input = (0..cols)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.017)
        .collect::<Vec<_>>();
    let expected = weights
        .chunks_exact(row_bytes)
        .map(|row| iq4_xs_reference_row_dot(row, &input))
        .collect::<Vec<_>>();

    let actual = match iq4_xs_gemv_for_test(&weights, rows, cols, &input) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA IQ4_XS GEMV test: {err}");
            return;
        }
    };

    for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-3,
            "IQ4_XS CUDA row {row} mismatch: actual={actual} expected={expected} diff={diff}"
        );
    }
}

#[test]
fn cuda_iq4_xs_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 4usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 136;
    let mut weights = vec![0u8; rows * row_bytes];
    for (i, byte) in weights.iter_mut().enumerate() {
        *byte = ((i * 29 + 7) & 0xff) as u8;
    }
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let off = row * row_bytes + block * 136;
            weights[off..off + 2].copy_from_slice(
                &half::f16::from_f32(0.0003 + row as f32 * 0.00005)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
    let input = (0..seq_len * cols)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.011)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; seq_len * rows];
    for seq in 0..seq_len {
        let input_row = &input[seq * cols..(seq + 1) * cols];
        for row in 0..rows {
            let weight_row = &weights[row * row_bytes..(row + 1) * row_bytes];
            expected[seq * rows + row] = iq4_xs_reference_row_dot(weight_row, input_row);
        }
    }

    let actual = match iq4_xs_gemv_batch_for_test(&weights, rows, cols, &input) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA IQ4_XS batch GEMV test: {err}");
            return;
        }
    };

    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-3,
            "IQ4_XS CUDA batch idx {idx} mismatch: actual={actual} expected={expected} diff={diff}"
        );
    }
}

fn synthetic_iq_weights(rows: usize, cols: usize, block_bytes: usize, seed: usize) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let mut weights = vec![0u8; rows * blocks_per_row * block_bytes];
    for (index, byte) in weights.iter_mut().enumerate() {
        *byte = ((index * 37 + seed * 19 + 11) & 0xff) as u8;
    }
    for (block_idx, block) in weights.chunks_exact_mut(block_bytes).enumerate() {
        let scale = half::f16::from_f32(0.00008 + (block_idx % 7) as f32 * 0.000002);
        block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
    }
    weights
}

#[test]
fn cuda_glm_sparse_iq2xxs_iq3xxs_matches_cpu_reference() {
    use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};

    let _guard = runtime_test_lock();
    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let route = [0.625f32, 0.375];
    let input = (0..n_embd)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.003)
        .collect::<Vec<_>>();
    let gate = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 66, expert * 3))
        .collect::<Vec<_>>();
    let up = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 66, expert * 3 + 1))
        .collect::<Vec<_>>();
    let down = (0..selected)
        .map(|expert| synthetic_iq_weights(n_embd, n_ff, 98, expert * 3 + 2))
        .collect::<Vec<_>>();
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let mut expected = vec![0.0f32; n_embd];
    for expert in 0..selected {
        let gate_f32 = dequantize_bytes_to_f32(&gate[expert], DequantType::IQ2XXS);
        let up_f32 = dequantize_bytes_to_f32(&up[expert], DequantType::IQ2XXS);
        let down_f32 = dequantize_bytes_to_f32(&down[expert], DequantType::IQ3XXS);
        let mut activation = vec![0.0f32; n_ff];
        for row in 0..n_ff {
            let gate_value = gate_f32[row * n_embd..(row + 1) * n_embd]
                .iter()
                .zip(input.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let up_value = up_f32[row * n_embd..(row + 1) * n_embd]
                .iter()
                .zip(input.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            activation[row] = (gate_value / (1.0 + (-gate_value).exp())) * up_value;
        }
        for row in 0..n_embd {
            let value = down_f32[row * n_ff..(row + 1) * n_ff]
                .iter()
                .zip(activation.iter())
                .map(|(weight, activation)| weight * activation)
                .sum::<f32>();
            expected[row] += route[expert] * value;
        }
    }

    let actual = match glm_sparse_experts_iq2xxs_iq3xxs(
        &gate_refs, &up_refs, &down_refs, &route, n_ff, n_embd, &input,
    ) {
        Ok(output) => output,
        Err(err) => {
            eprintln!("skipping GLM sparse IQ2_XXS/IQ3_XXS CUDA test: {err}");
            return;
        }
    };
    for (row, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = 2.0e-4 + expected.abs() * 2.0e-4;
        assert!(
            diff <= tolerance,
            "GLM sparse IQ2_XXS/IQ3_XXS row {row} mismatch: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}

#[test]
fn cuda_glm_mla_prefill_attention_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let pos_start = 2usize;
    let seq_len = 3usize;
    let num_heads = 2usize;
    let kv_len = pos_start + seq_len;
    let kv_rank = 64usize;
    let rope_dim = 16usize;
    let kv_width = kv_rank + rope_dim;
    let query_count = seq_len * num_heads;
    let scale = 1.0 / ((kv_rank + rope_dim) as f32).sqrt();
    let q_absorbed = (0..query_count * kv_rank)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.0175)
        .collect::<Vec<_>>();
    let q_pe = (0..query_count * rope_dim)
        .map(|index| ((index % 13) as f32 - 6.0) * 0.025)
        .collect::<Vec<_>>();
    let cache = (0..kv_len * kv_width)
        .map(|index| half::f16::from_f32(((index % 29) as f32 - 14.0) * 0.0125).to_bits())
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; query_count * kv_rank];
    for token in 0..seq_len {
        let attend_len = pos_start + token + 1;
        for head in 0..num_heads {
            let query = token * num_heads + head;
            let q_latent = &q_absorbed[query * kv_rank..(query + 1) * kv_rank];
            let q_rope = &q_pe[query * rope_dim..(query + 1) * rope_dim];
            let mut scores = (0..attend_len)
                .map(|key| {
                    let cached = &cache[key * kv_width..(key + 1) * kv_width];
                    let latent_dot = q_latent
                        .iter()
                        .zip(&cached[..kv_rank])
                        .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                        .sum::<f32>();
                    let rope_dot = q_rope
                        .iter()
                        .zip(&cached[kv_rank..])
                        .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                        .sum::<f32>();
                    (latent_dot + rope_dot) * scale
                })
                .collect::<Vec<_>>();
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores {
                *score /= sum;
            }
            for dim in 0..kv_rank {
                expected[query * kv_rank + dim] = scores
                    .iter()
                    .enumerate()
                    .map(|(key, &probability)| {
                        probability * half::f16::from_bits(cache[key * kv_width + dim]).to_f32()
                    })
                    .sum();
            }
        }
    }

    let actual = match glm_mla_prefill_attention_f16(
        &q_absorbed,
        &q_pe,
        &cache,
        pos_start,
        seq_len,
        num_heads,
        kv_len,
        kv_rank,
        rope_dim,
        scale,
    ) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping CUDA GLM MLA attention test: {error}");
            return;
        }
    };
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
        assert!(
            diff <= tolerance,
            "GLM MLA attention index {index} mismatch: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}

#[test]
fn cuda_glm_sparse_iq2xxs_iq3xxs_by_token_matches_individual_dispatches() {
    let _guard = runtime_test_lock();
    let _parallel = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_PARALLEL", "0");
    let grouped_off = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_GROUPED", "0");
    let token_count = 2usize;
    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let route = [0.625f32, 0.375, 0.25, 0.75];
    let input = (0..token_count * n_embd)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.0025)
        .collect::<Vec<_>>();
    let gate = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 66, expert * 3))
        .collect::<Vec<_>>();
    let up = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 66, expert * 3 + 1))
        .collect::<Vec<_>>();
    let down = (0..selected)
        .map(|expert| synthetic_iq_weights(n_embd, n_ff, 98, expert * 3 + 2))
        .collect::<Vec<_>>();
    let mut gate_slots = Vec::with_capacity(token_count * selected);
    let mut up_slots = Vec::with_capacity(token_count * selected);
    let mut down_slots = Vec::with_capacity(token_count * selected);
    let mut token_ids = Vec::with_capacity(token_count * selected);
    for token in 0..token_count {
        for expert in 0..selected {
            gate_slots.push(gate[expert].as_slice());
            up_slots.push(up[expert].as_slice());
            down_slots.push(down[expert].as_slice());
            token_ids.push(token as u32);
        }
    }

    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        expected.extend(
            glm_sparse_experts_iq2xxs_iq3xxs(
                &gate_slots[token * selected..(token + 1) * selected],
                &up_slots[token * selected..(token + 1) * selected],
                &down_slots[token * selected..(token + 1) * selected],
                &route[token * selected..(token + 1) * selected],
                n_ff,
                n_embd,
                &input[token * n_embd..(token + 1) * n_embd],
            )
            .expect("individual GLM sparse dispatch"),
        );
    }
    let scalar = glm_sparse_experts_iq_by_token(
        &gate_slots,
        &up_slots,
        &down_slots,
        16,
        18,
        None,
        false,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("scalar batched GLM sparse dispatch");
    drop(grouped_off);
    let _grouped_on = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_GROUPED", "1");
    let grouped = glm_sparse_experts_iq_by_token(
        &gate_slots,
        &up_slots,
        &down_slots,
        16,
        18,
        None,
        false,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("expert-grouped GLM sparse dispatch");
    for (index, ((&scalar, &grouped), &expected)) in scalar
        .iter()
        .zip(grouped.iter())
        .zip(expected.iter())
        .enumerate()
    {
        let scalar_diff = (scalar - expected).abs();
        let grouped_diff = (grouped - expected).abs();
        let tolerance = 2.0e-4 + expected.abs() * 2.0e-4;
        assert!(
            scalar_diff <= tolerance,
            "GLM scalar batched sparse index {index} mismatch: actual={scalar} expected={expected} diff={scalar_diff} tolerance={tolerance}"
        );
        assert!(
            grouped_diff <= tolerance,
            "GLM expert-grouped sparse index {index} mismatch: actual={grouped} expected={expected} diff={grouped_diff} tolerance={tolerance}"
        );
    }
}

#[test]
fn cuda_glm_sparse_iq2s_iq4xs_by_token_matches_cpu_reference() {
    use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};

    let _guard = runtime_test_lock();
    let _parallel = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_PARALLEL", "0");
    let grouped_off = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_GROUPED", "0");
    let token_count = 2usize;
    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let route = [0.625f32, 0.375, 0.25, 0.75];
    let input = (0..token_count * n_embd)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.0025)
        .collect::<Vec<_>>();
    let gate = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 82, expert * 3))
        .collect::<Vec<_>>();
    let up = (0..selected)
        .map(|expert| synthetic_iq_weights(n_ff, n_embd, 82, expert * 3 + 1))
        .collect::<Vec<_>>();
    let down = (0..selected)
        .map(|expert| synthetic_iq_weights(n_embd, n_ff, 136, expert * 3 + 2))
        .collect::<Vec<_>>();
    let gate_f32 = gate
        .iter()
        .map(|weights| dequantize_bytes_to_f32(weights, DequantType::IQ2S))
        .collect::<Vec<_>>();
    let up_f32 = up
        .iter()
        .map(|weights| dequantize_bytes_to_f32(weights, DequantType::IQ2S))
        .collect::<Vec<_>>();
    let down_f32 = down
        .iter()
        .map(|weights| dequantize_bytes_to_f32(weights, DequantType::IQ4XS))
        .collect::<Vec<_>>();
    let mut gate_slots = Vec::with_capacity(token_count * selected);
    let mut up_slots = Vec::with_capacity(token_count * selected);
    let mut down_slots = Vec::with_capacity(token_count * selected);
    let mut token_ids = Vec::with_capacity(token_count * selected);
    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        for expert in 0..selected {
            gate_slots.push(gate[expert].as_slice());
            up_slots.push(up[expert].as_slice());
            down_slots.push(down[expert].as_slice());
            token_ids.push(token as u32);
            let mut activation = vec![0.0f32; n_ff];
            for row in 0..n_ff {
                let gate_value = gate_f32[expert][row * n_embd..(row + 1) * n_embd]
                    .iter()
                    .zip(token_input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>();
                let up_value = up_f32[expert][row * n_embd..(row + 1) * n_embd]
                    .iter()
                    .zip(token_input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>();
                activation[row] = (gate_value / (1.0 + (-gate_value).exp())) * up_value;
            }
            for row in 0..n_embd {
                let value = down_f32[expert][row * n_ff..(row + 1) * n_ff]
                    .iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>();
                expected[token * n_embd + row] += route[token * selected + expert] * value;
            }
        }
    }

    let scalar = glm_sparse_experts_iq_by_token(
        &gate_slots,
        &up_slots,
        &down_slots,
        22,
        23,
        None,
        false,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("scalar batched GLM IQ2_S/IQ4_XS dispatch");
    drop(grouped_off);
    let _grouped_on = EnvVarGuard::set("RNB_CUDA_GLM_EXPERT_GROUPED", "1");
    let grouped = glm_sparse_experts_iq_by_token(
        &gate_slots,
        &up_slots,
        &down_slots,
        22,
        23,
        None,
        false,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("expert-grouped GLM IQ2_S/IQ4_XS dispatch");

    for (index, ((&scalar, &grouped), &expected)) in
        scalar.iter().zip(&grouped).zip(&expected).enumerate()
    {
        let scalar_diff = (scalar - expected).abs();
        let grouped_diff = (grouped - expected).abs();
        let tolerance = 3.0e-4 + expected.abs() * 3.0e-4;
        assert!(
            scalar_diff <= tolerance,
            "GLM IQ2_S/IQ4_XS scalar index {index} mismatch: actual={scalar} expected={expected} diff={scalar_diff} tolerance={tolerance}"
        );
        assert!(
            grouped_diff <= tolerance,
            "GLM IQ2_S/IQ4_XS grouped index {index} mismatch: actual={grouped} expected={expected} diff={grouped_diff} tolerance={tolerance}"
        );
    }
}

#[test]
fn attention_prefill_rejects_oversized_sliding_window_before_cuda_init() {
    let err = attention_prefill_flash_f32(
        &[0.0],
        &[0.0],
        &[0.0],
        1,
        1,
        1,
        1,
        1,
        1.0,
        Some(u32::MAX as usize + 1),
        None,
    )
    .expect_err("oversized sliding window must be rejected");
    assert_eq!(
        err,
        "CUDA attention sliding window exceeds u32 kernel limits"
    );
}

#[test]
fn attention_prefill_flash_hd256_matches_cpu_reference() {
    let seq_len = 4usize;
    let kv_len = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }

    let actual = match attention_prefill_flash_hd256(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * v[v_off + d];
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd256_f16kv_window_matches_cpu_reference() {
    let _guard = runtime_test_lock();

    let seq_len = 6usize;
    let kv_len = 6usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }
    let k_f16 = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let v_f16 = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();

    let actual = match attention_prefill_flash_hd256_f16kv_window(
        &q,
        &k_f16,
        &v_f16,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd256 f16kv window attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        let start = (global_pos + 1).saturating_sub(window);
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1 - start);
            for j in start..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k_f16[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * half::f16::from_bits(*b).to_f32())
                    .sum::<f32>()
                    * scale;
                scores.push((j, dot));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            for (j, score) in scores {
                let p = (score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * half::f16::from_bits(v_f16[v_off + d]).to_f32();
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd256 f16kv window attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd512_matches_cpu_reference() {
    let seq_len = 4usize;
    let kv_len = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }

    let actual = match attention_prefill_flash_hd512(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd512 attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * v[v_off + d];
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd512 attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd512_w256_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev = std::env::var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256").ok();
    std::env::set_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", "1");

    let seq_len = 4usize;
    let kv_len = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }

    let actual = match attention_prefill_flash_hd512(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd512 w256 attention prefill test: {err}");
            if let Some(prev) = prev {
                std::env::set_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", prev);
            } else {
                std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
            }
            return;
        }
    };

    if let Some(prev) = prev {
        std::env::set_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", prev);
    } else {
        std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
    }

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * v[v_off + d];
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd512 w256 attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd512_f16kv_matches_cpu_reference() {
    let _guard = runtime_test_lock();

    let seq_len = 4usize;
    let kv_len = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }
    let k_f16 = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let v_f16 = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();

    let actual = match attention_prefill_flash_hd512_f16kv(
        &q,
        &k_f16,
        &v_f16,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd512 f16kv attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k_f16[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * half::f16::from_bits(v_f16[v_off + d]).to_f32();
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd512 f16kv attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd512_f16kv_window_matches_cpu_reference() {
    let _guard = runtime_test_lock();

    let seq_len = 6usize;
    let kv_len = 6usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }
    let k_f16 = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let v_f16 = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();

    let actual = match attention_prefill_flash_hd512_f16kv_window(
        &q,
        &k_f16,
        &v_f16,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd512 f16kv window attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        let start = (global_pos + 1).saturating_sub(window);
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1 - start);
            for j in start..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k_f16[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * half::f16::from_bits(*b).to_f32())
                    .sum::<f32>()
                    * scale;
                scores.push((j, dot));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            for (j, score) in scores {
                let p = (score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * half::f16::from_bits(v_f16[v_off + d]).to_f32();
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd512 f16kv window attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_prefill_flash_hd128_matches_cpu_reference() {
    let seq_len = 4usize;
    let kv_len = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 128usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.011;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 37) as f32 - 18.0) * 0.017;
    }

    let actual = match attention_prefill_flash_hd128(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd128 attention prefill test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * v[v_off + d];
                }
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd128 attention prefill mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_hd128_matches_cpu_reference() {
    let kv_len = 5usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 128usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 29) as f32 - 14.0) * 0.013;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 31) as f32 - 15.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 37) as f32 - 18.0) * 0.017;
        *x = half::f16::from_f32(value).to_bits();
    }

    let actual = match attention_decode_hd128(&q, &k, &v, kv_len, num_heads, num_kv_heads, scale) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd128 attention decode test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_row = &q[q_off..q_off + head_dim];
        let mut scores = Vec::with_capacity(kv_len);
        for j in 0..kv_len {
            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let dot = q_row
                .iter()
                .zip(k[k_off..k_off + head_dim].iter())
                .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push(dot);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
        for (j, score) in scores.iter().enumerate() {
            let p = (*score - max_score).exp() / denom;
            let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let out_off = h * head_dim;
            for d in 0..head_dim {
                expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA hd128 attention decode mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_cached_hd128_matches_cpu_reference_after_append() {
    let _guard = runtime_test_lock();
    let kv_len = 6usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 128usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 23) as f32 - 11.0) * 0.019;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 29) as f32 - 14.0) * 0.007;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 41) as f32 - 20.0) * 0.013;
        *x = half::f16::from_f32(value).to_bits();
    }

    let layer_index = 998usize;
    let warm_len = kv_len - 1;
    if let Err(err) = attention_decode_cached(
        layer_index,
        &q,
        &k[..warm_len * num_kv_heads * head_dim],
        &v[..warm_len * num_kv_heads * head_dim],
        warm_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        eprintln!("skipping CUDA cached hd128 attention decode test: {err}");
        return;
    }
    let actual = match attention_decode_cached(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA cached hd128 attention decode test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_row = &q[q_off..q_off + head_dim];
        let mut scores = Vec::with_capacity(kv_len);
        for j in 0..kv_len {
            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let dot = q_row
                .iter()
                .zip(k[k_off..k_off + head_dim].iter())
                .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push(dot);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
        for (j, score) in scores.iter().enumerate() {
            let p = (*score - max_score).exp() / denom;
            let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let out_off = h * head_dim;
            for d in 0..head_dim {
                expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA cached hd128 attention decode mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_cached_window_hd256_matches_cpu_reference_after_append() {
    let _guard = runtime_test_lock();
    let kv_len = 516usize;
    let window_start = 4usize;
    let window_len = kv_len - window_start;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 256usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 31) as f32 - 15.0) * 0.017;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 37) as f32 - 18.0) * 0.006;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 43) as f32 - 21.0) * 0.012;
        *x = half::f16::from_f32(value).to_bits();
    }

    let layer_index = 997usize;
    let warm_len = kv_len - 2;
    if let Err(err) = attention_decode_cached(
        layer_index,
        &q,
        &k[..warm_len * num_kv_heads * head_dim],
        &v[..warm_len * num_kv_heads * head_dim],
        warm_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        eprintln!("skipping CUDA cached window hd256 attention decode test: {err}");
        return;
    }
    let actual = match attention_decode_cached_window(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        window_start,
        window_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA cached window hd256 attention decode test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_row = &q[q_off..q_off + head_dim];
        let mut scores = Vec::with_capacity(window_len);
        for j in window_start..kv_len {
            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let dot = q_row
                .iter()
                .zip(k[k_off..k_off + head_dim].iter())
                .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push(dot);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
        for (local_j, score) in scores.iter().enumerate() {
            let p = (*score - max_score).exp() / denom;
            let j = window_start + local_j;
            let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let out_off = h * head_dim;
            for d in 0..head_dim {
                expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA cached window hd256 attention decode mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_cached_hd512_matches_cpu_reference_after_append() {
    let _guard = runtime_test_lock();
    let kv_len = 5usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let layer_index = 996usize;
    let warm_len = kv_len - 1;
    if let Err(err) = attention_decode_cached(
        layer_index,
        &q,
        &k[..warm_len * num_kv_heads * head_dim],
        &v[..warm_len * num_kv_heads * head_dim],
        warm_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        if err.contains("unsupported head_dim=512") {
            panic!("CUDA cached hd512 attention decode should be supported");
        }
        eprintln!("skipping CUDA cached hd512 attention decode test: {err}");
        return;
    }
    let actual = match attention_decode_cached(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            if err.contains("unsupported head_dim=512") {
                panic!("CUDA cached hd512 attention decode should be supported");
            }
            eprintln!("skipping CUDA cached hd512 attention decode test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_row = &q[q_off..q_off + head_dim];
        let mut scores = Vec::with_capacity(kv_len);
        for j in 0..kv_len {
            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let dot = q_row
                .iter()
                .zip(k[k_off..k_off + head_dim].iter())
                .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push(dot);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
        for (j, score) in scores.iter().enumerate() {
            let p = (*score - max_score).exp() / denom;
            let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let out_off = h * head_dim;
            for d in 0..head_dim {
                expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA cached hd512 attention decode mismatch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_hd512_len_device_matches_scalar_len() {
    let _guard = runtime_test_lock();
    let kv_len = 7usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let scalar = match attention_decode_hd512(&q, &k, &v, kv_len, num_heads, num_kv_heads, scale) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA hd512 scalar length attention decode test: {err}");
            return;
        }
    };
    let device_len =
        match attention_decode_hd512_len_device(&q, &k, &v, kv_len, num_heads, num_kv_heads, scale)
        {
            Ok(out) => out,
            Err(err) => {
                eprintln!("skipping CUDA hd512 device length attention decode test: {err}");
                return;
            }
        };

    let max_diff = scalar
        .iter()
        .zip(device_len.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "CUDA hd512 device length attention decode diverged from scalar path: max_diff={max_diff}"
    );
}

#[test]
fn device_kv_cache_materializes_exact_f16_bits() {
    let _guard = runtime_test_lock();
    let _ = clear_sequence_state_cache();
    let layer_index = 993usize;
    let kv_dim = 16usize;
    let token_count = 3usize;
    let key = (0..kv_dim * token_count)
        .map(|index| 0x3400u16.wrapping_add(index as u16))
        .collect::<Vec<_>>();
    let value = (0..kv_dim * token_count)
        .map(|index| 0x3c00u16.wrapping_sub(index as u16))
        .collect::<Vec<_>>();
    if let Err(err) = populate_device_kv_cache_f16(layer_index, &key, &value, kv_dim, token_count) {
        eprintln!("skipping CUDA device KV materialization test: {err}");
        return;
    }
    assert!(
        device_kv_cache_f16_matches(layer_index, kv_dim, token_count)
            .expect("device KV cache query")
    );

    let mut restored_key = vec![0u16; key.len()];
    let mut restored_value = vec![0u16; value.len()];
    assert!(sync_device_kv_cache_f16_to_host(
        layer_index,
        &mut restored_key,
        &mut restored_value,
        kv_dim,
        token_count,
    )
    .expect("device KV cache materialization"));
    assert_eq!(restored_key, key);
    assert_eq!(restored_value, value);

    clear_sequence_state_cache().expect("clear device sequence state");
    assert!(
        !device_kv_cache_f16_matches(layer_index, kv_dim, token_count)
            .expect("cleared device KV cache query")
    );
}

#[test]
fn launch_attention_decode_device_len_device_matches_scalar_launch() {
    let _guard = runtime_test_lock();
    let layer_index = 994usize;
    let kv_len = 7usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let q_bytes = num_heads * head_dim * std::mem::size_of::<f32>();
    let output_len = num_heads * head_dim;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let mut q = vec![0.0f32; output_len];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    if let Err(err) =
        populate_device_kv_cache_f16(layer_index, &k, &v, num_kv_heads * head_dim, kv_len)
    {
        eprintln!("skipping CUDA device attention launch test: {err}");
        return;
    }
    let q_dev = match acquire_decode_hidden_carrier(q_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA device attention launch test: {err}");
            return;
        }
    };
    let output_dev = match acquire_decode_attn_out_carrier(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA device attention launch test: {err}");
            return;
        }
    };
    if let Err(err) = upload_to_decode_hidden_carrier(&q, q_dev) {
        eprintln!("skipping CUDA device attention launch test: {err}");
        return;
    }
    if let Err(err) = launch_attention_decode_device(
        layer_index,
        q_dev,
        output_dev,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len - 1,
    ) {
        eprintln!("skipping CUDA scalar length device attention launch test: {err}");
        return;
    }
    let mut scalar = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut scalar) {
        eprintln!("skipping CUDA scalar length device attention download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA scalar length device attention sync: {err}");
        return;
    }

    if let Err(err) = upload_to_decode_hidden_carrier(&q, q_dev) {
        eprintln!("skipping CUDA device attention relaunch test: {err}");
        return;
    }
    if let Err(err) = launch_attention_decode_device_len_device(
        layer_index,
        q_dev,
        output_dev,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len - 1,
    ) {
        eprintln!("skipping CUDA device length attention launch test: {err}");
        return;
    }
    let mut device_len = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut device_len) {
        eprintln!("skipping CUDA device length attention download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA device length attention sync: {err}");
        return;
    }

    let max_diff = scalar
        .iter()
        .zip(device_len.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "CUDA device length attention launch diverged from scalar launch: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_cached_to_device_len_device_matches_scalar_to_device() {
    let _guard = runtime_test_lock();
    let layer_index = 993usize;
    let kv_len = 257usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let output_len = num_heads * head_dim;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; output_len];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let output_dev = match acquire_decode_attn_out_carrier(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA cached to-device attention test: {err}");
            return;
        }
    };
    if let Err(err) = attention_decode_cached_to_device(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached scalar to-device attention test: {err}");
        return;
    }
    let mut scalar = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut scalar) {
        eprintln!("skipping CUDA cached scalar to-device download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA cached scalar to-device sync: {err}");
        return;
    }

    if let Err(err) = attention_decode_cached_to_device_len_device(
        layer_index + 1,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length to-device attention test: {err}");
        return;
    }
    let mut device_len = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut device_len) {
        eprintln!("skipping CUDA cached device length to-device download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA cached device length to-device sync: {err}");
        return;
    }

    let max_diff = scalar
        .iter()
        .zip(device_len.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA cached device length to-device attention diverged from scalar path: max_diff={max_diff}"
    );
}

#[test]
fn attention_decode_cached_to_device_len_device_graph_captures_after_warmup() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA cached device length graph attention test: {err}");
            return;
        }
    };
    let layer_index = 992usize;
    let kv_len = 9usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let output_len = num_heads * head_dim;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; output_len];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let output_dev = match state.decode_attn_out_carrier_ptr(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA cached device length graph attention test: {err}");
            return;
        }
    };
    if let Err(err) = state.attention_decode_cached_to_device_len_device_graph(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length graph warmup test: {err}");
        return;
    }
    assert_eq!(state.cu68_attention_graph_warmed.len(), 1);
    assert_eq!(state.cu68_attention_graphs.len(), 0);

    if let Err(err) = state.attention_decode_cached_to_device_len_device_graph(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length graph capture test: {err}");
        return;
    }
    assert_eq!(state.cu68_attention_graphs.len(), 1);

    if let Err(err) = state.attention_decode_cached_to_device_len_device_graph(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length graph replay test: {err}");
        return;
    }
    assert_eq!(state.cu68_attention_graphs.len(), 1);
}

#[test]
fn attention_decode_cached_to_device_len_device_graph_preserves_hd512_split_for_long_kv() {
    let _guard = runtime_test_lock();
    let _split = EnvVarGuard::set("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", "1");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA cached device length split-preserve test: {err}");
            return;
        }
    };
    let layer_index = 990usize;
    let kv_len = 257usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let output_len = num_heads * head_dim;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; output_len];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let output_dev = match state.decode_attn_out_carrier_ptr(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA cached device length split-preserve test: {err}");
            return;
        }
    };
    if let Err(err) = state.attention_decode_cached_to_device_len_device_graph(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length split-preserve test: {err}");
        return;
    }

    assert_eq!(
        state.cu68_attention_graph_warmed.len(),
        0,
        "long hd512 decode must keep the split-K path instead of warming cu68 attention graph"
    );
    assert_eq!(
        state.cu68_attention_graphs.len(),
        0,
        "long hd512 decode must keep the split-K path instead of capturing cu68 attention graph"
    );
}

#[test]
fn attention_decode_cached_to_device_len_device_graph_api_matches_direct_api() {
    let _guard = runtime_test_lock();
    let layer_index = 991usize;
    let kv_len = 11usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let output_len = num_heads * head_dim;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; output_len];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let output_dev = match acquire_decode_attn_out_carrier(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA cached device length graph API test: {err}");
            return;
        }
    };
    if let Err(err) = attention_decode_cached_to_device_len_device(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev,
        None,
        None,
        None,
    ) {
        eprintln!("skipping CUDA cached device length direct API test: {err}");
        return;
    }
    let mut direct = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut direct) {
        eprintln!("skipping CUDA cached device length direct API download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA cached device length direct API sync: {err}");
        return;
    }

    for _ in 0..3 {
        if let Err(err) = attention_decode_cached_to_device_len_device_graph(
            layer_index + 1,
            &q,
            &k,
            &v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            output_dev,
            None,
            None,
            None,
        ) {
            eprintln!("skipping CUDA cached device length graph API test: {err}");
            return;
        }
    }
    let mut graph = vec![0.0f32; output_len];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut graph) {
        eprintln!("skipping CUDA cached device length graph API download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA cached device length graph API sync: {err}");
        return;
    }

    let max_diff = direct
        .iter()
        .zip(graph.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "CUDA cached device length graph API diverged from direct API: max_diff={max_diff}"
    );
}

#[test]
fn cu69_dense_chain_graph_state_starts_empty() {
    let _guard = runtime_test_lock();
    let state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph state test: {err}");
            return;
        }
    };
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 0);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 0);
}

#[test]
fn qwen35_compound_graph_state_starts_empty() {
    let _guard = runtime_test_lock();
    let state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA Qwen35 compound graph state test: {err}");
            return;
        }
    };
    assert_eq!(state.qwen35_compound_graphs.len(), 0);
}

#[test]
fn cu71_layer_segment_graph_state_starts_empty() {
    let _guard = runtime_test_lock();
    let state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA layer segment graph state test: {err}");
            return;
        }
    };
    assert_eq!(state.cu71_layer_segment_graph_warmed.len(), 0);
    assert_eq!(state.cu71_layer_segment_graphs.len(), 0);
}

fn cu71_valid_capture_inputs() -> Cu71LayerSegmentCaptureInputs {
    Cu71LayerSegmentCaptureInputs {
        qkv_ready: true,
        attention_ready: true,
        dense_ready: true,
        long_kv_split_preferred: false,
        would_allocate_during_capture: false,
        q_carrier_dev: 11,
        k_carrier_dev: 12,
        v_carrier_dev: 13,
        kv_cache_identity: 14,
        attn_out_dev: 15,
        hidden_dev: 16,
        dense_graph_identity: 17,
    }
}

fn cu71_valid_layer_segment_key() -> LayerSegmentGraphKey {
    LayerSegmentGraphKey {
        layer_idx: 3,
        n_embd: 1536,
        q_rows: 2048,
        kv_dim: 512,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        rope_theta_bits: 10_000.0f32.to_bits(),
        norm_eps_bits: 1e-6f32.to_bits(),
        attention_scale_bits: (1.0f32 / (256.0f32).sqrt()).to_bits(),
        q_quant: GGML_Q4_K as u32,
        k_quant: GGML_Q4_K as u32,
        v_quant: GGML_Q4_K as u32,
        o_quant: GGML_Q4_K as u32,
        gate_quant: GGML_Q4_K as u32,
        up_quant: GGML_Q4_K as u32,
        down_quant: GGML_Q6_K as u32,
        q_carrier_dev: 101,
        k_carrier_dev: 102,
        v_carrier_dev: 103,
        k_f16_dev: 104,
        v_f16_dev: 105,
        kv_cache_identity: 106,
        kv_bucket: LayerSegmentKvBucketKey::from_bucket_view(
            rnb_backend_api::KvBucketView::new(3, 2048, 17, 2048, 512, 0x3001, 0x3002)
                .expect("bucket view"),
        ),
        kv_len_dev: 107,
        attn_out_dev: 108,
        hidden_dev: 109,
        normed_dev: 110,
        proj_dev: 111,
        gate_dev: 112,
        up_dev: 113,
        q_norm_hash: 0x51,
        k_norm_hash: 0x52,
        post_attn_norm_hash: 0x53,
        ffn_norm_hash: 0x54,
        post_ffn_norm_hash: 0x55,
        q_weight_identity: 0x1001,
        k_weight_identity: 0x1002,
        v_weight_identity: 0x1003,
        o_weight_identity: 0x1004,
        gate_weight_identity: 0x1005,
        up_weight_identity: 0x1006,
        down_weight_identity: 0x1007,
        packed_gate_identity: 0x2001,
        packed_up_identity: 0x2002,
        packed_down_identity: 0x2003,
        global_attention: true,
        has_ple: true,
        has_layer_output_scale: false,
        has_post_attn_norm: true,
        has_post_ffn_norm: true,
        ffn_uses_gelu: true,
        q8dot_qkv: true,
        q8dot_o: true,
        q8dot_gate_up: true,
        q8dot_down: false,
    }
}

fn cu71_layer_segment_graph_runtime_context(
    state: &mut CudaState,
    fixture: &Cu69DenseChainGraphFixture,
    layer_idx: usize,
) -> Cu71LayerSegmentGraphRuntimeContext {
    let q_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let kv_bytes = fixture.q_dim * std::mem::size_of::<u16>();
    let q_carrier_dev = state.decode_q_carrier_ptr(q_bytes).expect("q carrier");
    let k_f16_dev = state
        .decode_k_f16_carrier_ptr(kv_bytes)
        .expect("k f16 carrier");
    let v_f16_dev = state
        .decode_v_f16_carrier_ptr(kv_bytes)
        .expect("v f16 carrier");
    Cu71LayerSegmentGraphRuntimeContext {
        layer_idx,
        q_rows: fixture.q_dim,
        kv_dim: fixture.q_dim,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: fixture.q_dim,
        rope_theta: 1_000_000.0,
        attention_scale: 1.0,
        q_quant: GGML_Q4_K,
        k_quant: GGML_Q4_K,
        v_quant: GGML_Q4_K,
        q_weight_identity: 0x5101,
        k_weight_identity: 0x5102,
        v_weight_identity: 0x5103,
        q_norm_hash: 0x6101,
        k_norm_hash: 0x6102,
        q_carrier_dev,
        k_f16_dev,
        v_f16_dev,
        kv_bucket: rnb_backend_api::KvBucketView::new(
            layer_idx,
            2048,
            17,
            2048,
            fixture.q_dim,
            0x3001,
            0x3002,
        )
        .expect("kv bucket"),
        long_kv_split_preferred: false,
    }
}

#[test]
fn cu71_layer_segment_key_captures_stable_identities() {
    let key = cu71_valid_layer_segment_key();
    assert_eq!(key, cu71_valid_layer_segment_key());

    let mut changed = key;
    changed.kv_cache_identity += 1;
    assert_ne!(key, changed);

    let mut changed = key;
    changed.q_norm_hash ^= 0x01;
    assert_ne!(key, changed);

    let mut changed = key;
    changed.packed_gate_identity += 1;
    assert_ne!(key, changed);
}

#[test]
fn cu71_layer_segment_kv_bucket_key_uses_stable_layout_identity() {
    let a = rnb_backend_api::KvBucketView::new(3, 2048, 17, 2048, 512, 0x1000, 0x2000)
        .expect("bucket view");
    let b = rnb_backend_api::KvBucketView::new(3, 2048, 18, 2048, 512, 0x1000, 0x2000)
        .expect("bucket view");

    assert_eq!(
        LayerSegmentKvBucketKey::from_bucket_view(a),
        LayerSegmentKvBucketKey::from_bucket_view(b)
    );

    let moved = rnb_backend_api::KvBucketView::new(3, 2048, 18, 2048, 512, 0x1008, 0x2000)
        .expect("bucket view");
    assert_ne!(
        LayerSegmentKvBucketKey::from_bucket_view(a),
        LayerSegmentKvBucketKey::from_bucket_view(moved)
    );

    for changed in [
        rnb_backend_api::KvBucketView::new(4, 2048, 17, 2048, 512, 0x1000, 0x2000)
            .expect("changed layer"),
        rnb_backend_api::KvBucketView::new(3, 1024, 17, 2048, 512, 0x1000, 0x2000)
            .expect("changed page size"),
        rnb_backend_api::KvBucketView::new(3, 2048, 17, 4096, 512, 0x1000, 0x2000)
            .expect("changed max length"),
        rnb_backend_api::KvBucketView::new(3, 2048, 17, 2048, 1024, 0x1000, 0x2000)
            .expect("changed row width"),
        rnb_backend_api::KvBucketView::new(3, 2048, 17, 2048, 512, 0x1000, 0x2008)
            .expect("changed V identity"),
    ] {
        assert_ne!(
            LayerSegmentKvBucketKey::from_bucket_view(a),
            LayerSegmentKvBucketKey::from_bucket_view(changed)
        );
    }
}

#[test]
fn cu71_layer_segment_key_captures_kv_bucket_identity() {
    let key = cu71_valid_layer_segment_key();
    let mut changed = key;
    changed.kv_bucket.k_identity += 1;

    assert_ne!(key, changed);
}

#[test]
fn cu71_layer_segment_graph_lifecycle_is_warm_capture_replay() {
    let key = cu71_valid_layer_segment_key();
    let mut warmed = HashSet::new();
    let mut captured = HashMap::new();

    assert_eq!(
        cu71_layer_segment_graph_step(
            true,
            cu71_valid_capture_inputs(),
            key,
            &mut warmed,
            &captured
        ),
        Cu71LayerSegmentGraphStep::Warm
    );
    assert!(warmed.contains(&key));

    assert_eq!(
        cu71_layer_segment_graph_step(
            true,
            cu71_valid_capture_inputs(),
            key,
            &mut warmed,
            &captured
        ),
        Cu71LayerSegmentGraphStep::Capture
    );

    captured.insert(key, SparseMoeGraph { graph: 1, exec: 2 });
    assert_eq!(
        cu71_layer_segment_graph_step(
            true,
            cu71_valid_capture_inputs(),
            key,
            &mut warmed,
            &captured
        ),
        Cu71LayerSegmentGraphStep::Replay
    );
}

#[test]
fn cu71_layer_segment_graph_lifecycle_preserves_preflight_rejections() {
    let key = cu71_valid_layer_segment_key();
    let mut warmed = HashSet::new();
    let captured = HashMap::new();
    let mut inputs = cu71_valid_capture_inputs();
    inputs.long_kv_split_preferred = true;

    assert_eq!(
        cu71_layer_segment_graph_step(true, inputs, key, &mut warmed, &captured),
        Cu71LayerSegmentGraphStep::Rejected(
            Cu71LayerSegmentCaptureRejectReason::SplitKAttentionPreferred
        )
    );
    assert!(warmed.is_empty());

    assert_eq!(
        cu71_layer_segment_graph_step(
            false,
            cu71_valid_capture_inputs(),
            key,
            &mut warmed,
            &captured
        ),
        Cu71LayerSegmentGraphStep::Disabled
    );
    assert!(warmed.is_empty());
}

#[test]
fn cu71_layer_segment_capture_requires_stable_device_buffers() {
    let mut inputs = cu71_valid_capture_inputs();
    inputs.kv_cache_identity = 0;
    assert_eq!(
        cu71_layer_segment_capture_decision(inputs),
        Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::MissingStableBuffer
        )
    );

    let mut inputs = cu71_valid_capture_inputs();
    inputs.dense_graph_identity = 0;
    assert_eq!(
        cu71_layer_segment_capture_decision(inputs),
        Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::MissingStableBuffer
        )
    );
}

#[test]
fn cu71_layer_segment_capture_rejects_allocation_and_split_k_eager_islands() {
    let mut inputs = cu71_valid_capture_inputs();
    inputs.would_allocate_during_capture = true;
    assert_eq!(
        cu71_layer_segment_capture_decision(inputs),
        Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::CaptureWouldAllocate
        )
    );

    let mut inputs = cu71_valid_capture_inputs();
    inputs.long_kv_split_preferred = true;
    assert_eq!(
        cu71_layer_segment_capture_decision(inputs),
        Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::SplitKAttentionPreferred
        )
    );
}

#[test]
fn attention_decode_cached_hd512_split_matches_cpu_reference_after_append() {
    let _guard = runtime_test_lock();
    let _split = EnvVarGuard::set("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", "1");
    let kv_len = 257usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0u16; kv_len * num_kv_heads * head_dim];
    let mut v = vec![0u16; kv_len * num_kv_heads * head_dim];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i % 47) as f32 - 23.0) * 0.009;
    }
    for (i, x) in k.iter_mut().enumerate() {
        let value = ((i % 53) as f32 - 26.0) * 0.005;
        *x = half::f16::from_f32(value).to_bits();
    }
    for (i, x) in v.iter_mut().enumerate() {
        let value = ((i % 59) as f32 - 29.0) * 0.011;
        *x = half::f16::from_f32(value).to_bits();
    }

    let layer_index = 995usize;
    let warm_len = kv_len - 1;
    if let Err(err) = attention_decode_cached(
        layer_index,
        &q,
        &k[..warm_len * num_kv_heads * head_dim],
        &v[..warm_len * num_kv_heads * head_dim],
        warm_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        eprintln!("skipping CUDA cached split hd512 attention decode test: {err}");
        return;
    }
    let actual = match attention_decode_cached(
        layer_index,
        &q,
        &k,
        &v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skipping CUDA cached split hd512 attention decode test: {err}");
            return;
        }
    };

    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let q_off = h * head_dim;
        let q_row = &q[q_off..q_off + head_dim];
        let mut scores = Vec::with_capacity(kv_len);
        for j in 0..kv_len {
            let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let dot = q_row
                .iter()
                .zip(k[k_off..k_off + head_dim].iter())
                .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push(dot);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
        for (j, score) in scores.iter().enumerate() {
            let p = (*score - max_score).exp() / denom;
            let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
            let out_off = h * head_dim;
            for d in 0..head_dim {
                expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
            }
        }
    }

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "CUDA cached split hd512 attention decode mismatch: max_diff={max_diff}"
    );
}

#[test]
fn cuda_q4k_block_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.03125).to_le_bytes());
    block[4..16].copy_from_slice(&[3, 5, 7, 11, 13, 17, 19, 23, 0x21, 0x43, 0x65, 0x07]);
    for (i, q) in block[16..].iter_mut().enumerate() {
        *q = ((i * 37 + 11) & 0xff) as u8;
    }
    let mut input = [0.0f32; 256];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 17.0) - 8.0) * 0.125;
    }
    let expected = cpu_q4k_block_dot(&block, &input);
    let actual = q4k_block_dot_for_test(&block, &input).expect("CUDA Q4_K dot");
    let diff = (actual - expected).abs();
    assert!(
        diff < 0.01,
        "CUDA Q4_K dot mismatch: expected {expected}, actual {actual}, diff {diff}"
    );
}
#[test]
fn cuda_q4k_row_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut row = vec![0u8; 144 * 8];
    for block_idx in 0..8usize {
        let block = &mut row[block_idx * 144..(block_idx + 1) * 144];
        block[0..2].copy_from_slice(
            &half::f16::from_f32(0.03125 + block_idx as f32 * 0.0078125).to_le_bytes(),
        );
        block[2..4].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
        for i in 0..12usize {
            block[4 + i] = ((block_idx * 19 + i * 7 + 3) & 0x3f) as u8;
        }
        for i in 0..128usize {
            block[16 + i] = ((block_idx * 53 + i * 29 + 5) & 0xff) as u8;
        }
    }
    let mut input = vec![0.0f32; 2048];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 23.0) - 11.0) * 0.0625;
    }
    let expected = row
        .chunks_exact(144)
        .enumerate()
        .map(|(block_idx, block)| {
            let block: &[u8; 144] = block.try_into().unwrap();
            let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                .try_into()
                .unwrap();
            cpu_q4k_block_dot(block, input)
        })
        .sum::<f32>();
    let actual = q4k_row_dot_for_test(&row, &input).expect("CUDA Q4_K row dot");
    let diff = (actual - expected).abs();
    assert!(
        diff < 0.05,
        "CUDA Q4_K row dot mismatch: expected {expected}, actual {actual}, diff {diff}"
    );
}

#[test]
fn cuda_q4k_embedding_gather_dequants_token_rows() {
    let _guard = runtime_test_lock();
    let rows = 4usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 127)
        .pop()
        .unwrap();
    let token_ids = [2u32, 0u32, 3u32];

    let mut expected = Vec::with_capacity(token_ids.len() * cols);
    for &token_id in &token_ids {
        let row_idx = token_id as usize;
        let row = &weights[row_idx * blocks_per_row * 144..(row_idx + 1) * blocks_per_row * 144];
        expected.extend(cpu_q4k_dequant_row(row, blocks_per_row));
    }

    let actual = q4k_embedding_gather_for_test(&weights, rows, cols, &token_ids)
        .expect("CUDA Q4_K embedding gather");

    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-6,
            "CUDA Q4_K embedding gather mismatch at {idx}: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

#[test]
fn cuda_q2k_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 11usize;
    let cols = 768usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q2k_weights(rows, blocks_per_row, 37);
    let input = (0..cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.01953125)
        .collect::<Vec<_>>();
    let expected = cpu_q2k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual = q2k_gemv(&weights, rows, cols, &input).expect("CUDA Q2_K GEMV");

    assert_close_rows_abs_rel("Q2_K GEMV", &actual, &expected, 0.01, 1.0e-4);
}

#[test]
fn cuda_q3k_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 11usize;
    let cols = 768usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q3k_weights(rows, blocks_per_row, 73);
    let input = (0..cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.017578125)
        .collect::<Vec<_>>();
    let expected = cpu_q3k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual = q3k_gemv(&weights, rows, cols, &input).expect("CUDA Q3_K GEMV");

    assert_close_rows_abs_rel("Q3_K GEMV", &actual, &expected, 0.02, 1.0e-4);
}

#[test]
fn cuda_q4k_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 6usize;
    let cols = 2048usize;
    let blocks_per_row = cols / 256;
    let mut weights = vec![0u8; rows * blocks_per_row * 144];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 144;
            let block = &mut weights[base..base + 144];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(
                    0.015625 + row_idx as f32 * 0.00390625 + block_idx as f32 * 0.001953125,
                )
                .to_le_bytes(),
            );
            block[2..4].copy_from_slice(
                &half::f16::from_f32(0.0078125 + row_idx as f32 * 0.0009765625).to_le_bytes(),
            );
            for i in 0..12usize {
                block[4 + i] = ((row_idx * 31 + block_idx * 19 + i * 5 + 7) & 0x3f) as u8;
            }
            for i in 0..128usize {
                block[16 + i] = ((row_idx * 47 + block_idx * 53 + i * 17 + 13) & 0xff) as u8;
            }
        }
    }
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 29.0) - 14.0) * 0.03125;
    }
    let mut expected = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * blocks_per_row * 144..(row_idx + 1) * blocks_per_row * 144];
        expected[row_idx] = row
            .chunks_exact(144)
            .enumerate()
            .map(|(block_idx, block)| {
                let block: &[u8; 144] = block.try_into().unwrap();
                let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                    .try_into()
                    .unwrap();
                cpu_q4k_block_dot(block, input)
            })
            .sum::<f32>();
    }
    let actual = q4k_gemv_for_test(&weights, rows, cols, &input).expect("CUDA Q4_K GEMV");
    for row_idx in 0..rows {
        let diff = (actual[row_idx] - expected[row_idx]).abs();
        assert!(
            diff < 0.05,
            "CUDA Q4_K GEMV row {row_idx} mismatch: expected {}, actual {}, diff {}",
            expected[row_idx],
            actual[row_idx],
            diff
        );
    }
}

#[test]
fn cuda_q5k_gemv_high_bits_match_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 2048usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q5k_weights(rows, blocks_per_row, 101);
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 31.0) - 15.0) * 0.02734375;
    }
    let expected = cpu_q5k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual = q5k_gemv_for_test(&weights, rows, cols, &input).expect("CUDA Q5_K GEMV");

    assert_close_rows("Q5_K high-bit GEMV", &actual, &expected, 0.08);
}

#[test]
fn cuda_q4k_gate_up_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 7usize;
    let cols = 1536usize;
    let blocks_per_row = cols / 256;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 31)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 73)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 37.0) - 18.0) * 0.015625;
    }
    let expected_gate = cpu_q4k_gemv_rows(&gate, rows, blocks_per_row, &input);
    let expected_up = cpu_q4k_gemv_rows(&up, rows, blocks_per_row, &input);

    let (actual_gate, actual_up) =
        q4k_gate_up_q8_for_test(&gate, &up, rows, cols, &input).expect("CUDA Q4_K gate/up Q8");

    assert_close_rows("Q4_K gate Q8", &actual_gate, &expected_gate, 0.08);
    assert_close_rows("Q4_K up Q8", &actual_up, &expected_up, 0.08);
}

#[test]
fn cuda_q4k_gate_up_batch_seq2_q8_dot_matches_separate_batch_path() {
    let _guard = runtime_test_lock();
    let rows = 11usize;
    let cols = 1024usize;
    let seq_len = 2usize;
    let blocks_per_row = cols / 256;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 41)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 97)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.013671875)
        .collect::<Vec<_>>();
    let expected_gate = q4k_gemv_batch(&gate, rows, cols, &input).expect("CUDA Q4_K gate batch");
    let expected_up = q4k_gemv_batch(&up, rows, cols, &input).expect("CUDA Q4_K up batch");

    let (actual_gate, actual_up) =
        q4k_gate_up_batch_seq2_q8_for_test(&gate, &up, rows, cols, &input)
            .expect("CUDA Q4_K gate/up seq2 Q8");

    assert_close_rows("Q4_K seq2 gate Q8", &actual_gate, &expected_gate, 0.05);
    assert_close_rows("Q4_K seq2 up Q8", &actual_up, &expected_up, 0.05);
}

#[test]
fn cuda_q4k_packed_gate_up_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 13usize;
    let cols = 1536usize;
    let blocks_per_row = cols / 256;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 37)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 89)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 47.0) - 23.0) * 0.01171875;
    }
    let expected_gate = cpu_q4k_gemv_rows(&gate, rows, blocks_per_row, &input);
    let expected_up = cpu_q4k_gemv_rows(&up, rows, blocks_per_row, &input);

    let (actual_gate, actual_up) = q4k_packed_gate_up_q8_for_test(&gate, &up, rows, cols, &input)
        .expect("CUDA packed Q4_K gate/up Q8");

    assert_close_rows("packed Q4_K gate Q8", &actual_gate, &expected_gate, 0.08);
    assert_close_rows("packed Q4_K up Q8", &actual_up, &expected_up, 0.08);
}

#[test]
fn cuda_q4k_packed_gate_up_batch_seq2_q8_dot_matches_unpacked_seq2_path() {
    let _guard = runtime_test_lock();
    let rows = 11usize;
    let cols = 1024usize;
    let seq_len = 2usize;
    let blocks_per_row = cols / 256;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 59)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 113)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.013671875)
        .collect::<Vec<_>>();

    let (expected_gate, expected_up) =
        q4k_gate_up_batch_seq2_q8_for_test(&gate, &up, rows, cols, &input)
            .expect("CUDA Q4_K gate/up seq2 Q8");
    let (actual_gate, actual_up) =
        q4k_packed_gate_up_batch_seq2_q8_for_test(&gate, &up, rows, cols, &input)
            .expect("CUDA packed Q4_K gate/up seq2 Q8");

    assert_close_rows(
        "packed seq2 Q4_K gate Q8",
        &actual_gate,
        &expected_gate,
        0.0001,
    );
    assert_close_rows("packed seq2 Q4_K up Q8", &actual_up, &expected_up, 0.0001);
}

#[test]
fn cuda_q4k_gemv_gelu_mul_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 17usize;
    let cols = 1536usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 131)
        .pop()
        .unwrap();
    let input = (0..cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.013671875)
        .collect::<Vec<_>>();
    let mul = (0..rows)
        .map(|i| ((i as f32 * 0.19).sin() * 0.5) + 0.75)
        .collect::<Vec<_>>();
    let mut expected = cpu_q4k_gemv_rows(&weights, rows, blocks_per_row, &input);
    for (value, &scale) in expected.iter_mut().zip(mul.iter()) {
        let x = *value;
        let x3 = x * x * x;
        let c = 0.7978845608028654f32;
        let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
        *value = gelu * scale;
    }

    let actual = q4k_gemv_gelu_mul_for_test(&weights, rows, cols, &input, &mul)
        .expect("CUDA Q4_K GELU mul GEMV");

    assert_close_rows("Q4_K GELU mul", &actual, &expected, 1e-3);
}

#[test]
fn cuda_rms_norm_add_then_rms_norm_matches_two_step_reference() {
    let _guard = runtime_test_lock();
    let len = 1536usize;
    let input = (0..len)
        .map(|i| ((i as f32 * 0.017).sin() * 0.7) + 0.03)
        .collect::<Vec<_>>();
    let residual = (0..len)
        .map(|i| ((i as f32 * 0.011).cos() * 0.5) - 0.02)
        .collect::<Vec<_>>();
    let post_weight = (0..len)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.003)
        .collect::<Vec<_>>();
    let pre_weight = (0..len)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.002)
        .collect::<Vec<_>>();
    let eps = 1e-6f32;

    let post_inv = (input.iter().map(|v| v * v).sum::<f32>() / len as f32 + eps)
        .sqrt()
        .recip();
    let mut expected_updated = residual.clone();
    for i in 0..len {
        expected_updated[i] += input[i] * post_inv * (1.0 + post_weight[i]);
    }
    let pre_inv = (expected_updated.iter().map(|v| v * v).sum::<f32>() / len as f32 + eps)
        .sqrt()
        .recip();
    let expected_output = expected_updated
        .iter()
        .zip(pre_weight.iter())
        .map(|(&value, &weight)| value * pre_inv * (1.0 + weight))
        .collect::<Vec<_>>();

    let (actual_updated, actual_output) = rms_norm_add_then_rms_norm_for_test(
        &input,
        &post_weight,
        &residual,
        &pre_weight,
        eps,
        true,
        true,
    )
    .expect("CUDA combined norm");

    for (idx, (&actual, &expected)) in actual_updated
        .iter()
        .zip(expected_updated.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 2e-5,
            "updated mismatch at {idx}: actual={actual} expected={expected} diff={diff}"
        );
    }
    for (idx, (&actual, &expected)) in actual_output.iter().zip(expected_output.iter()).enumerate()
    {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 2e-5,
            "output mismatch at {idx}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

#[test]
fn cuda_mtp_build_eh_input_matches_cpu_norm_concat() {
    let _guard = runtime_test_lock();
    let rows = 3usize;
    let hidden_dim = 512usize;
    let eps = 1.0e-6f32;
    let token_rows = (0..rows * hidden_dim)
        .map(|i| ((i as f32 * 0.013).sin() * 0.6) + 0.01)
        .collect::<Vec<_>>();
    let target_hidden_rows = (0..rows * hidden_dim)
        .map(|i| ((i as f32 * 0.017).cos() * 0.5) - 0.02)
        .collect::<Vec<_>>();
    let enorm = (0..hidden_dim)
        .map(|i| 0.75 + (i % 11) as f32 * 0.01)
        .collect::<Vec<_>>();
    let hnorm = (0..hidden_dim)
        .map(|i| 0.65 + (i % 13) as f32 * 0.012)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(rows * hidden_dim * 2);
    for row in 0..rows {
        let start = row * hidden_dim;
        expected.extend(cpu_rms_norm(
            &token_rows[start..start + hidden_dim],
            &enorm,
            eps,
            false,
        ));
        expected.extend(cpu_rms_norm(
            &target_hidden_rows[start..start + hidden_dim],
            &hnorm,
            eps,
            false,
        ));
    }

    let actual = mtp_build_eh_input_for_test(
        &token_rows,
        &target_hidden_rows,
        &enorm,
        &hnorm,
        rows,
        hidden_dim,
        eps,
    )
    .expect("CUDA MTP EH input");

    assert_close_rows("MTP EH input", &actual, &expected, 2e-5);
}

#[test]
fn cuda_rms_norm_add_then_rms_norm_q8_matches_two_step_reference() {
    let _guard = runtime_test_lock();
    let len = 1536usize;
    let input = (0..len)
        .map(|i| ((i as f32 * 0.013).sin() * 0.6) - 0.05)
        .collect::<Vec<_>>();
    let residual = (0..len)
        .map(|i| ((i as f32 * 0.019).cos() * 0.4) + 0.01)
        .collect::<Vec<_>>();
    let post_weight = (0..len)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0025)
        .collect::<Vec<_>>();
    let pre_weight = (0..len)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.00175)
        .collect::<Vec<_>>();
    let eps = 1e-6f32;

    let post_inv = (input.iter().map(|v| v * v).sum::<f32>() / len as f32 + eps)
        .sqrt()
        .recip();
    let mut expected_updated = residual.clone();
    for i in 0..len {
        expected_updated[i] += input[i] * post_inv * (1.0 + post_weight[i]);
    }
    let pre_inv = (expected_updated.iter().map(|v| v * v).sum::<f32>() / len as f32 + eps)
        .sqrt()
        .recip();
    let expected_output = expected_updated
        .iter()
        .zip(pre_weight.iter())
        .map(|(&value, &weight)| value * pre_inv * (1.0 + weight))
        .collect::<Vec<_>>();

    let (actual_updated, actual_output, q8_qs, q8_ds) = rms_norm_add_then_rms_norm_q8_for_test(
        &input,
        &post_weight,
        &residual,
        &pre_weight,
        eps,
        true,
        true,
    )
    .expect("CUDA combined norm Q8");

    assert_close_rows(
        "combined norm Q8 updated",
        &actual_updated,
        &expected_updated,
        2e-5,
    );
    assert_close_rows(
        "combined norm Q8 output",
        &actual_output,
        &expected_output,
        2e-5,
    );
    let mut dequant = Vec::with_capacity(q8_qs.len());
    for (idx, &q) in q8_qs.iter().enumerate() {
        dequant.push(q as f32 * q8_ds[idx / 32]);
    }
    assert_close_rows("combined norm Q8 dequant", &dequant, &expected_output, 0.01);
}

#[test]
fn cuda_gemma_ple_base_preload_returns_layer_slice() {
    let _guard = runtime_test_lock();
    let base = (0..96)
        .map(|idx| (idx as f32 - 31.0) * 0.03125)
        .collect::<Vec<_>>();
    let actual = gemma_ple_base_slice_for_test(&base, 32, 24).expect("CUDA Gemma PLE base slice");
    assert_close_rows("Gemma PLE base slice", &actual, &base[32..56], 0.0);
}

#[test]
fn cuda_q4k_gemv_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 6144usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 83)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 41.0) - 20.0) * 0.009765625;
    }
    let expected = cpu_q4k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual = q4k_gemv_q8_for_test(&weights, rows, cols, &input).expect("CUDA Q4_K Q8 GEMV");
    assert_close_rows("Q4_K Q8 GEMV", &actual, &expected, 0.12);
}

#[test]
fn cuda_q4k_packed_gemv_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 6144usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 97)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 43.0) - 21.0) * 0.0107421875;
    }
    let expected = cpu_q4k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual =
        q4k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q4_K Q8 GEMV");
    assert_close_rows("packed Q4_K Q8 GEMV", &actual, &expected, 0.12);
}

#[test]
fn cuda_q4k_transformed_view_matches_packed_q8_path() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 6144usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 113)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 47.0) - 23.0) * 0.009765625;
    }
    let expected = cpu_q4k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let packed =
        q4k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q4_K Q8 GEMV");
    let transformed = q4k_packed_q8_view_for_test(&weights, rows, cols, &input)
        .expect("CUDA transformed-view packed Q4_K Q8 GEMV");

    assert_close_rows(
        "transformed-view Q4_K Q8 GEMV",
        &transformed,
        &expected,
        0.12,
    );
    assert_close_rows(
        "transformed-view Q4_K vs packed",
        &transformed,
        &packed,
        1.0e-5,
    );
}

#[test]
fn cuda_q4k_payload_backed_cache_matches_packed_q8_path() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 6144usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 127)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 53.0) - 26.0) * 0.0087890625;
    }
    let expected = cpu_q4k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let packed =
        q4k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q4_K Q8 GEMV");
    let payload = q4k_packed_q8_payload_for_test(&weights, rows, cols, &input)
        .expect("CUDA payload-backed Q4_K Q8 GEMV");

    assert_close_rows("payload-backed Q4_K Q8 GEMV", &payload, &expected, 0.12);
    assert_close_rows("payload-backed Q4_K vs packed", &payload, &packed, 1.0e-5);
}

#[test]
fn cuda_q4k_qkv_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let q_rows = 11usize;
    let kv_rows = 5usize;
    let cols = 1536usize;
    let blocks_per_row = cols / 256;
    let q = make_test_q4k_weights(1, q_rows, blocks_per_row, 19)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, kv_rows, blocks_per_row, 43)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, kv_rows, blocks_per_row, 61)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 53.0) - 26.0) * 0.0078125;
    }
    let expected_q = cpu_q4k_gemv_rows(&q, q_rows, blocks_per_row, &input);
    let expected_k = cpu_q4k_gemv_rows(&k, kv_rows, blocks_per_row, &input);
    let expected_v = cpu_q4k_gemv_rows(&v, kv_rows, blocks_per_row, &input);

    let (actual_q, actual_k, actual_v) =
        q4k_qkv_q8_for_test(&q, &k, &v, q_rows, kv_rows, cols, &input).expect("CUDA Q4_K QKV Q8");

    assert_close_rows("Q4_K QKV Q8 q", &actual_q, &expected_q, 0.08);
    assert_close_rows("Q4_K QKV Q8 k", &actual_k, &expected_k, 0.08);
    assert_close_rows("Q4_K QKV Q8 v", &actual_v, &expected_v, 0.08);
}

#[test]
fn cuda_q6k_gemv_q8_dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 12288usize;
    let blocks_per_row = cols / 256;
    let mut weights = vec![0u8; rows * blocks_per_row * 210];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 210;
            let block = &mut weights[base..base + 210];
            for i in 0..128usize {
                block[i] = ((row_idx * 29 + block_idx * 31 + i * 17 + 11) & 0xff) as u8;
            }
            for i in 0..64usize {
                block[128 + i] = ((row_idx * 13 + block_idx * 19 + i * 23 + 5) & 0xff) as u8;
            }
            for i in 0..16usize {
                block[192 + i] =
                    (((row_idx * 7 + block_idx * 5 + i * 3 + 1) % 48) as i32 - 24) as i8 as u8;
            }
            block[208..210].copy_from_slice(
                &half::f16::from_f32(0.01171875 + row_idx as f32 * 0.00048828125).to_le_bytes(),
            );
        }
    }
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 43.0) - 21.0) * 0.0078125;
    }
    let expected = cpu_q6k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let actual = q6k_gemv_q8_for_test(&weights, rows, cols, &input).expect("CUDA Q6_K Q8 GEMV");
    assert_close_rows("Q6_K Q8 GEMV", &actual, &expected, 0.3);
    let packed_actual =
        q6k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q6_K Q8 GEMV");
    assert_close_rows("packed Q6_K Q8 GEMV", &packed_actual, &expected, 0.3);
}

#[test]
fn cuda_q6k_transformed_view_matches_packed_q8_path() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 12288usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 119)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 47.0) - 23.0) * 0.0068359375;
    }
    let expected = cpu_q6k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let packed =
        q6k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q6_K Q8 GEMV");
    let transformed = q6k_packed_q8_view_for_test(&weights, rows, cols, &input)
        .expect("CUDA transformed-view packed Q6_K Q8 GEMV");

    assert_close_rows(
        "transformed-view Q6_K Q8 GEMV",
        &transformed,
        &expected,
        0.3,
    );
    assert_close_rows(
        "transformed-view Q6_K vs packed",
        &transformed,
        &packed,
        1.0e-5,
    );
}

#[test]
fn cuda_q6k_payload_backed_cache_matches_packed_q8_path() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 12288usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 131)
        .pop()
        .unwrap();
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 59.0) - 29.0) * 0.005859375;
    }
    let expected = cpu_q6k_gemv_rows(&weights, rows, blocks_per_row, &input);
    let packed =
        q6k_packed_q8_for_test(&weights, rows, cols, &input).expect("CUDA packed Q6_K Q8 GEMV");
    let payload = q6k_packed_q8_payload_for_test(&weights, rows, cols, &input)
        .expect("CUDA payload-backed Q6_K Q8 GEMV");

    assert_close_rows("payload-backed Q6_K Q8 GEMV", &payload, &expected, 0.3);
    assert_close_rows("payload-backed Q6_K vs packed", &payload, &packed, 1.0e-5);
}

#[test]
fn cuda_gelu_mul_q8_1_matches_f32_activation_reference() {
    let _guard = runtime_test_lock();
    let len = 96usize;
    let mut gate = vec![0.0f32; len];
    let mut up = vec![0.0f32; len];
    for i in 0..len {
        gate[i] = ((i as f32 % 23.0) - 11.0) * 0.125;
        up[i] = ((i as f32 % 17.0) - 8.0) * 0.0625;
    }
    let mut expected = vec![0.0f32; len];
    for i in 0..len {
        let x = gate[i];
        let x3 = x * x * x;
        let c = 0.7978845608028654f32;
        let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
        expected[i] = gelu * up[i];
    }
    let (actual_f32, actual_q8_dequant) =
        gelu_mul_q8_1_for_test(&gate, &up).expect("CUDA GELU Q8_1 activation");
    assert_close_rows("GELU f32", &actual_f32, &expected, 1e-5);
    assert_close_rows("GELU Q8_1 dequant", &actual_q8_dequant, &expected, 0.01);
}

#[test]
fn cuda_silu_mul_q8_1_matches_f32_activation_reference() {
    let _guard = runtime_test_lock();
    let len = 96usize;
    let mut gate = vec![0.0f32; len];
    let mut up = vec![0.0f32; len];
    for i in 0..len {
        gate[i] = ((i as f32 % 29.0) - 14.0) * 0.09375;
        up[i] = ((i as f32 % 19.0) - 9.0) * 0.0546875;
    }
    let mut expected = vec![0.0f32; len];
    for i in 0..len {
        let x = gate[i];
        expected[i] = (x / (1.0 + (-x).exp())) * up[i];
    }
    let (actual_f32, actual_q8_dequant) =
        silu_mul_q8_1_for_test(&gate, &up).expect("CUDA SiLU Q8_1 activation");
    assert_close_rows("SiLU f32", &actual_f32, &expected, 1e-5);
    assert_close_rows("SiLU Q8_1 dequant", &actual_q8_dequant, &expected, 0.01);
}

#[test]
fn cuda_q5_0_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 96usize;
    let blocks_per_row = cols / 32;
    let mut weights = vec![0u8; rows * blocks_per_row * 22];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 22;
            let block = &mut weights[base..base + 22];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(0.03125 + row_idx as f32 * 0.001953125).to_le_bytes(),
            );
            for i in 0..4 {
                block[2 + i] = ((row_idx * 13 + block_idx * 17 + i * 29) & 0xff) as u8;
            }
            for i in 0..16 {
                block[6 + i] = ((row_idx * 23 + block_idx * 31 + i * 7) & 0xff) as u8;
            }
        }
    }
    let input = (0..cols)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.015625)
        .collect::<Vec<_>>();
    let expected = cpu_q5_basic_rows(&weights, rows, cols, 22, false, &input);
    let actual = q5_0_gemv(&weights, rows, cols, &input).expect("CUDA Q5_0 GEMV");
    assert_close_rows("Q5_0", &actual, &expected, 0.01);
}

#[test]
fn cuda_q5_1_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 96usize;
    let blocks_per_row = cols / 32;
    let mut weights = vec![0u8; rows * blocks_per_row * 24];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 24;
            let block = &mut weights[base..base + 24];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(0.03125 + row_idx as f32 * 0.001953125).to_le_bytes(),
            );
            block[2..4].copy_from_slice(
                &half::f16::from_f32(-0.25 + block_idx as f32 * 0.03125).to_le_bytes(),
            );
            for i in 0..4 {
                block[4 + i] = ((row_idx * 13 + block_idx * 17 + i * 29) & 0xff) as u8;
            }
            for i in 0..16 {
                block[8 + i] = ((row_idx * 23 + block_idx * 31 + i * 7) & 0xff) as u8;
            }
        }
    }
    let input = (0..cols)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.015625)
        .collect::<Vec<_>>();
    let expected = cpu_q5_basic_rows(&weights, rows, cols, 24, true, &input);
    let actual = q5_1_gemv(&weights, rows, cols, &input).expect("CUDA Q5_1 GEMV");
    assert_close_rows("Q5_1", &actual, &expected, 0.01);
}

#[test]
fn cuda_q5_basic_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 96usize;
    let seq_len = 3usize;
    for (label, block_bytes, has_min) in [("Q5_0", 22usize, false), ("Q5_1", 24usize, true)] {
        let weights = make_test_q5_basic_weights(rows, cols, block_bytes, has_min, 73);
        let input = (0..seq_len * cols)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.015625)
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(seq_len * rows);
        for token in 0..seq_len {
            expected.extend(cpu_q5_basic_rows(
                &weights,
                rows,
                cols,
                block_bytes,
                has_min,
                &input[token * cols..(token + 1) * cols],
            ));
        }
        let actual = if has_min {
            q5_1_gemv_batch(&weights, rows, cols, &input)
        } else {
            q5_0_gemv_batch(&weights, rows, cols, &input)
        }
        .unwrap_or_else(|err| panic!("CUDA {label} batch GEMV failed: {err}"));
        assert_close_rows(label, &actual, &expected, 0.01);
    }
}

#[test]
fn cuda_q4k_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let seq_len = 3usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 89)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual = q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K batch GEMV");
    assert_close_rows("Q4_K batch", &actual, &expected, 0.05);
}

#[test]
fn cuda_q4k_gemv_batch_seq4_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _seq4 = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "1");
    let rows = 1024usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 5usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 93)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.0078125)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual =
        q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K batch raw seq4 GEMV");
    assert_close_rows("Q4_K batch raw seq4", &actual, &expected, 0.05);
}

#[test]
fn cuda_q4k_gemv_batch_seq4_matches_warp8_bits() {
    let _guard = runtime_test_lock();
    let _q8dot = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_Q8DOT", "0");
    let rows = 1024usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 5usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 94)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.00625)
        .collect::<Vec<_>>();
    let warp8 = {
        let _seq4 = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "0");
        q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K warp8 batch GEMV")
    };
    let seq4 = {
        let _seq4 = EnvVarGuard::set("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "1");
        q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K raw seq4 batch GEMV")
    };
    assert_eq!(warp8.len(), seq4.len());
    for (idx, (a, b)) in warp8.iter().zip(seq4.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Q4_K raw seq4 bit mismatch at index {idx}: warp8={a} seq4={b}"
        );
    }
}

#[test]
fn cuda_q4k_gemv_batch_q8dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", "1");
    let rows = 1024usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 2usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 97)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.00390625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual = q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K batch Q8 dot GEMV");
    if let Some(prev) = prev {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT");
    }
    assert_close_rows("Q4_K batch Q8 dot", &actual, &expected, 0.08);
}

#[test]
fn cuda_q4k_mmq_tile32_matches_cpu_reference_with_tails() {
    let _guard = runtime_test_lock();
    let _mmq = EnvVarGuard::set("RNB_CUDA_Q4K_MMQ_TILE32", "1");
    let rows = 1057usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 37usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 98)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.00390625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let mut first_actual: Option<Vec<f32>> = None;
    for run in 0..16 {
        let actual = q4k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q4_K tiled MMQ");
        assert_close_rows(
            &format!("Q4_K tiled MMQ run {run}"),
            &actual,
            &expected,
            0.08,
        );
        if let Some(first) = first_actual.as_ref() {
            let mismatch = actual
                .iter()
                .zip(first)
                .position(|(actual, first)| actual.to_bits() != first.to_bits());
            assert!(
                mismatch.is_none(),
                "Q4_K tiled MMQ run {run} is not bitwise deterministic at {mismatch:?}"
            );
        } else {
            first_actual = Some(actual.clone());
        }
    }
}

#[test]
fn cuda_q4k_batch_dev_input_q8dot_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV").ok();
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV", "1");
    let rows = 1024usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 2usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 101)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.0048828125)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("allocate Q4_K dev-input test input");
    let output_bytes = seq_len * rows * std::mem::size_of::<f32>();
    let output_dev = state
        .compute_output_ptr(output_bytes)
        .expect("allocate Q4_K dev-input test output");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("copy Q4_K dev-input test input");
    }
    state
        .q4k_batch_dev_input_to_dev(
            &weights,
            rows,
            blocks_per_row,
            seq_len,
            input_dev,
            output_dev,
        )
        .expect("CUDA Q4_K dev-input batch Q8 dot GEMV");
    let mut actual = vec![0.0f32; seq_len * rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )
            .expect("copy Q4_K dev-input test output");
    }
    state
        .stream_synchronize()
        .expect("synchronize Q4_K dev-input test");
    restore_env_var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV", prev);
    assert_close_rows("Q4_K dev-input batch Q8 dot", &actual, &expected, 0.08);
}

#[test]
fn cuda_q6k_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let seq_len = 3usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 97)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual = q6k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q6_K batch GEMV");
    assert_close_rows("Q6_K batch", &actual, &expected, 0.05);
}

#[test]
fn cuda_q6k_mmq_tile32_matches_cpu_reference_with_tails() {
    let _guard = runtime_test_lock();
    let _mmq = EnvVarGuard::set("RNB_CUDA_Q6K_MMQ_TILE32", "1");
    let rows = 1057usize;
    let cols = 1024usize;
    let blocks_per_row = cols / 256;
    let seq_len = 37usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 99)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.00390625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let mut first_actual: Option<Vec<f32>> = None;
    for run in 0..16 {
        let actual = q6k_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q6_K tiled MMQ");
        assert_close_rows(
            &format!("Q6_K tiled MMQ run {run}"),
            &actual,
            &expected,
            0.3,
        );
        if let Some(first) = first_actual.as_ref() {
            let mismatch = actual
                .iter()
                .zip(first)
                .position(|(actual, first)| actual.to_bits() != first.to_bits());
            assert!(
                mismatch.is_none(),
                "Q6_K tiled MMQ run {run} is not bitwise deterministic at {mismatch:?}"
            );
        } else {
            first_actual = Some(actual.clone());
        }
    }
}

#[test]
fn cuda_q6k_packed_batch_warp4_min8_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _seq4 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ4", "1");
    let _warp4 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
    let rows = 64usize;
    let cols = 3584usize;
    let blocks_per_row = cols / 256;
    let seq_len = 5usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 109)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.0078125)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual = q6k_packed_batch_q8_for_test(&weights, rows, cols, &input, seq_len)
        .expect("CUDA packed Q6_K batch warp4 min8 Q8 GEMV");
    assert_close_rows("packed Q6_K batch warp4 min8 Q8", &actual, &expected, 0.3);
}

#[test]
fn cuda_q6k_packed_batch_warp4_min8_matches_warp8_tolerance() {
    let _guard = runtime_test_lock();
    let _seq4 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ4", "1");
    let rows = 64usize;
    let cols = 3584usize;
    let blocks_per_row = cols / 256;
    let seq_len = 5usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 113)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 53.0) - 26.0) * 0.00625)
        .collect::<Vec<_>>();
    let warp8 = {
        let _warp4 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_WARP4", "0");
        q6k_packed_batch_q8_for_test(&weights, rows, cols, &input, seq_len)
            .expect("CUDA packed Q6_K batch warp8 Q8 GEMV")
    };
    let warp4 = {
        let _warp4 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        q6k_packed_batch_q8_for_test(&weights, rows, cols, &input, seq_len)
            .expect("CUDA packed Q6_K batch warp4 min8 Q8 GEMV")
    };
    assert_close_rows(
        "packed Q6_K batch warp4 min8 vs warp8",
        &warp4,
        &warp8,
        1e-4,
    );
}

#[test]
fn cuda_q6k_packed_batch_selector_uses_seq8_warp4_when_env_forces_it() {
    let _guard = runtime_test_lock();
    let _seq8 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", "1");

    assert_eq!(
        crate::runtime::q6k_packed_batch_kernel_plan_for_test(
            9,  // seq_len
            14, // blocks_per_row
        ),
        crate::runtime::Q6PackedBatchKernelPlanForTest::Seq8Warp4
    );
}

#[test]
fn cuda_q6k_packed_batch_selector_keeps_seq8_default_off() {
    let _guard = runtime_test_lock();
    let _seq8 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8");
    let _seq4 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ4");
    let _warp4 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_WARP4");

    assert_eq!(
        crate::runtime::q6k_packed_batch_kernel_plan_for_test(
            9,  // seq_len
            14, // blocks_per_row
        ),
        crate::runtime::Q6PackedBatchKernelPlanForTest::Seq4Warp4
    );
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_q6k_packed_batch_seq8_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _seq8 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", "1");
    let _seq4 = EnvVarGuard::set("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ4", "1");
    let _warp4 = EnvVarGuard::remove("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
    let rows = 128usize;
    let blocks_per_row = 14usize;
    let cols = blocks_per_row * 256;
    let seq_len = 9usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 109)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 61.0) - 30.0) * 0.00390625)
        .collect::<Vec<_>>();

    let seq8 = crate::runtime::test_support::q6k_packed_batch_q8_for_test(
        &weights, rows, cols, &input, seq_len,
    )
    .expect("seq8 Q6 packed batch");

    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        let input_row = &input[token * cols..(token + 1) * cols];
        expected.extend(cpu_q6k_gemv_rows(&weights, rows, blocks_per_row, input_row));
    }

    assert_close_rows("q6 packed seq8 vs cpu", &seq8, &expected, 0.3);
}

#[test]
fn cuda_q6k_gemv_batch_seq2_warp8_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let seq_len = 2usize;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 103)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.01171875)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let input_dev = state
        .mem_alloc(std::mem::size_of_val(input.as_slice()))
        .unwrap();
    let output_dev = state
        .mem_alloc(seq_len * rows * std::mem::size_of::<f32>())
        .unwrap();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state
        .launch_q6k_gemv_batch_seq2_warp8_to_dev(
            &weights,
            rows,
            blocks_per_row,
            input_dev,
            output_dev,
        )
        .expect("CUDA Q6_K seq2 batch GEMV");
    let mut actual = vec![0.0f32; seq_len * rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();
    unsafe {
        state.api.mem_free(input_dev).unwrap();
        state.api.mem_free(output_dev).unwrap();
    }
    assert_close_rows("Q6_K seq2 batch warp8", &actual, &expected, 0.05);
}

#[test]
fn cuda_dense_q4k_gelu_ffn_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");

    let n_embd = 512usize;
    let n_ff = 512usize;
    let seq_len = 3usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 101)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 113)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 127)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.0078125)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(seq_len * n_embd);
    for token in 0..seq_len {
        let input_row = &input[token * n_embd..(token + 1) * n_embd];
        let mut gate_out = cpu_q4k_gemv_rows(&gate, n_ff, gate_blocks, input_row);
        let up_out = cpu_q4k_gemv_rows(&up, n_ff, gate_blocks, input_row);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            let x = *gate_value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *gate_value = gelu * *up_value;
        }
        expected.extend(cpu_q6k_gemv_rows(&down, n_embd, down_blocks, &gate_out));
    }

    let actual = dense_q4k_gelu_ffn_batch(&gate, &up, &down, 14, n_ff, n_embd, seq_len, &input)
        .expect("CUDA dense Q4_K GELU FFN batch");

    if let Some(prev) = prev {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    assert_close_rows("dense Q4_K GELU FFN batch", &actual, &expected, 0.2);
}

#[test]
fn cuda_glm_shared_expert_q5k_q6k_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let n_embd = 512usize;
    let n_ff = 512usize;
    let token_count = 3usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q5k_weights(n_ff, gate_blocks, 167);
    let up = make_test_q5k_weights(n_ff, gate_blocks, 173);
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 179)
        .pop()
        .unwrap();
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.00390625)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        let input_row = &input[token * n_embd..(token + 1) * n_embd];
        let mut gate_out = cpu_q5k_gemv_rows(&gate, n_ff, gate_blocks, input_row);
        let up_out = cpu_q5k_gemv_rows(&up, n_ff, gate_blocks, input_row);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            *gate_value = *gate_value / (1.0 + (-*gate_value).exp()) * *up_value;
        }
        expected.extend(cpu_q6k_gemv_rows(&down, n_embd, down_blocks, &gate_out));
    }

    let actual = glm_shared_expert_iq(&gate, &up, &down, 13, 14, n_ff, n_embd, &input)
        .expect("CUDA GLM shared expert batch");
    assert_close_rows("GLM shared expert batch", &actual, &expected, 0.3);
}

#[test]
fn cuda_glm_shared_expert_q6k_q8_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let n_embd = 512usize;
    let n_ff = 512usize;
    let token_count = 3usize;
    let gate_blocks = n_embd / 256;
    let gate = make_test_q6k_weights(1, n_ff, gate_blocks, 181)
        .pop()
        .unwrap();
    let up = make_test_q6k_weights(1, n_ff, gate_blocks, 191)
        .pop()
        .unwrap();
    let down = make_test_q8_0_weights(n_embd, n_ff, 193);
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.00390625)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        let input_row = &input[token * n_embd..(token + 1) * n_embd];
        let mut gate_out = cpu_q6k_gemv_rows(&gate, n_ff, gate_blocks, input_row);
        let up_out = cpu_q6k_gemv_rows(&up, n_ff, gate_blocks, input_row);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            *gate_value = *gate_value / (1.0 + (-*gate_value).exp()) * *up_value;
        }
        expected.extend(cpu_q8_0_rows(&down, n_embd, n_ff, &gate_out));
    }

    let actual = glm_shared_expert_iq(&gate, &up, &down, 14, 8, n_ff, n_embd, &input)
        .expect("CUDA GLM Q6_K/Q8_0 shared expert batch");
    assert_close_rows("GLM Q6_K/Q8_0 shared expert batch", &actual, &expected, 0.3);
}

#[test]
fn cuda_dense_q4k_gelu_ffn_batch_q8dot_matches_separate_batch_path() {
    let _guard = runtime_test_lock();
    let prev_dense = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "1");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", "1");

    let n_embd = 1024usize;
    let n_ff = 1024usize;
    let seq_len = 2usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 131)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 137)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 149)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 53.0) - 26.0) * 0.00390625)
        .collect::<Vec<_>>();

    let mut gate_out = q4k_gemv_batch(&gate, n_ff, n_embd, &input).expect("CUDA Q4_K gate batch");
    let up_out = q4k_gemv_batch(&up, n_ff, n_embd, &input).expect("CUDA Q4_K up batch");
    for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
        let x = *gate_value;
        let x3 = x * x * x;
        let c = 0.7978845608028654f32;
        let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
        *gate_value = gelu * *up_value;
    }
    let expected = q6k_gemv_batch(&down, n_embd, n_ff, &gate_out).expect("CUDA Q6_K down batch");
    let actual = dense_q4k_gelu_ffn_batch(&gate, &up, &down, 14, n_ff, n_embd, seq_len, &input)
        .expect("CUDA dense Q4_K GELU FFN batch q8dot");

    if let Some(prev) = prev_dense {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT");
    }
    assert_close_rows("dense Q4_K GELU FFN batch q8dot", &actual, &expected, 0.05);
}

#[test]
fn cuda_dense_q4k_silu_ffn_batch_q8dot_matches_separate_batch_path() {
    let _guard = runtime_test_lock();
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "1");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "1");
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", "1");

    let n_embd = 1024usize;
    let n_ff = 1024usize;
    let seq_len = 2usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 151)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 157)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 163)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 59.0) - 29.0) * 0.00390625)
        .collect::<Vec<_>>();

    let mut gate_out = q4k_gemv_batch(&gate, n_ff, n_embd, &input).expect("CUDA Q4_K gate batch");
    let up_out = q4k_gemv_batch(&up, n_ff, n_embd, &input).expect("CUDA Q4_K up batch");
    for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
        let x = *gate_value;
        *gate_value = (x / (1.0 + (-x).exp())) * *up_value;
    }
    for chunk in gate_out.chunks_mut(32) {
        let max_abs = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let d = max_abs / 127.0;
        for value in chunk {
            let q = (*value / d).round().clamp(-127.0, 127.0);
            *value = q * d;
        }
    }
    let expected = q6k_gemv_batch(&down, n_embd, n_ff, &gate_out).expect("CUDA Q6_K down batch");
    let actual = dense_q4k_silu_ffn_batch(&gate, &up, &down, 14, n_ff, n_embd, seq_len, &input)
        .expect("CUDA dense Q4_K SiLU FFN batch q8dot");

    restore_env_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev_gate);
    restore_env_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev_down);
    restore_env_var("RNB_CUDA_Q4K_BATCH_Q8DOT", prev_prefill);
    assert_close_rows("dense Q4_K SiLU FFN batch q8dot", &actual, &expected, 0.2);
}

#[test]
fn cuda_dense_q4k_gelu_ffn_batch_raw_q4_down_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _gate_q8dot = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down_q8dot = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let _expanded_down = EnvVarGuard::remove("RNB_CUDA_Q4K_BATCH_F16_DOWN");

    let n_embd = 512usize;
    let n_ff = 512usize;
    let seq_len = 3usize;
    let gate_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let gate = make_test_q4k_weights(1, n_ff, gate_blocks, 151)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, gate_blocks, 163)
        .pop()
        .unwrap();
    let down = make_test_q4k_weights(1, n_embd, down_blocks, 179)
        .pop()
        .unwrap();
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.005)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(seq_len * n_embd);
    for token in 0..seq_len {
        let input_row = &input[token * n_embd..(token + 1) * n_embd];
        let mut gate_out = cpu_q4k_gemv_rows(&gate, n_ff, gate_blocks, input_row);
        let up_out = cpu_q4k_gemv_rows(&up, n_ff, gate_blocks, input_row);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            let x = *gate_value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *gate_value = gelu * *up_value;
        }
        expected.extend(cpu_q4k_gemv_rows(&down, n_embd, down_blocks, &gate_out));
    }

    let actual = dense_q4k_gelu_ffn_batch(&gate, &up, &down, 12, n_ff, n_embd, seq_len, &input)
        .expect("CUDA dense Q4_K GELU FFN batch raw Q4_K down");

    assert_close_rows_abs_rel(
        "dense Q4_K GELU FFN batch raw Q4_K down",
        &actual,
        &expected,
        0.35,
        0.01,
    );
}

#[test]
fn cuda_dense_q4k_attention_output_gelu_ffn_batch_norm_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_batch = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", "0");

    let n_embd = 512usize;
    let q_dim = 512usize;
    let n_ff = 512usize;
    let seq_len = 3usize;
    let q_blocks = q_dim / 256;
    let hidden_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let o = make_test_q4k_weights(1, n_embd, q_blocks, 181)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, n_ff, hidden_blocks, 191)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, hidden_blocks, 193)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 197)
        .pop()
        .unwrap();
    let mut hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
        .collect::<Vec<_>>();
    let initial_hidden = hidden.clone();
    let attn_out = (0..seq_len * q_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.75 + (i % 17) as f32 * 0.003)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.82 + (i % 19) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.91 + (i % 23) as f32 * 0.0015)
        .collect::<Vec<_>>();
    let eps = 1.0e-5;

    let mut expected = hidden.clone();
    for token in 0..seq_len {
        let row = token * n_embd;
        let attn_row = &attn_out[token * q_dim..(token + 1) * q_dim];
        let o_proj = cpu_q4k_gemv_rows(&o, n_embd, q_blocks, attn_row);
        let post_attn = cpu_rms_norm(&o_proj, &post_attn_norm, eps, true);
        for i in 0..n_embd {
            expected[row + i] += post_attn[i];
        }
        let ffn_input = cpu_rms_norm(&expected[row..row + n_embd], &ffn_norm, eps, true);
        let mut gate_out = cpu_q4k_gemv_rows(&gate, n_ff, hidden_blocks, &ffn_input);
        let up_out = cpu_q4k_gemv_rows(&up, n_ff, hidden_blocks, &ffn_input);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            let x = *gate_value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *gate_value = gelu * *up_value;
        }
        let down_out = cpu_q6k_gemv_rows(&down, n_embd, down_blocks, &gate_out);
        let post_ffn = cpu_rms_norm(&down_out, &post_ffn_norm, eps, true);
        for i in 0..n_embd {
            expected[row + i] += post_ffn[i];
        }
    }

    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        q_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut hidden,
        &attn_out,
        eps,
        true,
        true,
        true,
    )
    .expect("CUDA dense Q4_K attention+FFN batch chain");

    let mut state = CudaState::open().expect("open CUDA state");
    let attn_out_bytes = std::mem::size_of_val(attn_out.as_slice());
    let attn_out_dev = state
        .compute_full_down_ptr(attn_out_bytes)
        .expect("attention output device scratch");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                attn_out_dev,
                attn_out.as_ptr().cast::<libc::c_void>(),
                attn_out_bytes,
                state.stream,
            )
            .expect("attention output h2d");
    }
    let mut device_hidden_input = initial_hidden;
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let output_id = state
        .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
            &o,
            &gate,
            &up,
            &down,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_dim,
            n_ff,
            n_embd,
            seq_len,
            &mut device_hidden_input,
            None,
            attn_out_dev,
            Some(output_desc),
            None,
            eps,
            true,
            true,
            true,
        )
        .expect("CUDA dense Q4_K attention+FFN device output")
        .expect("device output id");
    let device_hidden = state
        .download_device_tensor_f32(output_id)
        .expect("download device dense output");
    assert!(state
        .release_device_tensor(output_id)
        .expect("release dense device output"));

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_batch {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT");
    }
    assert_close_rows_abs_rel(
        "dense Q4_K attention+FFN batch chain",
        &hidden,
        &expected,
        0.35,
        0.015,
    );
    assert_close_rows_abs_rel(
        "dense Q4_K attention+FFN batch chain device output",
        &device_hidden,
        &expected,
        0.35,
        0.015,
    );
}

struct Cu69DenseChainGraphFixture {
    o: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    hidden: Vec<f32>,
    attn_out: Vec<f32>,
    n_embd: usize,
    q_dim: usize,
    n_ff: usize,
}

struct Cu69DenseChainGraphPleTail {
    gate: Vec<u8>,
    proj: Vec<u8>,
    post_norm: Vec<f32>,
    input: Vec<f32>,
    dim: usize,
}

fn cu69_dense_chain_graph_fixture(seed: usize) -> Cu69DenseChainGraphFixture {
    cu69_dense_chain_graph_fixture_with_dims(seed, 512, 512, 512)
}

fn cu69_dense_chain_graph_fixture_with_dims(
    seed: usize,
    n_embd: usize,
    q_dim: usize,
    n_ff: usize,
) -> Cu69DenseChainGraphFixture {
    let q_blocks = q_dim / 256;
    let hidden_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    Cu69DenseChainGraphFixture {
        o: make_test_q4k_weights(1, n_embd, q_blocks, 1181 + seed)
            .pop()
            .unwrap(),
        gate: make_test_q4k_weights(1, n_ff, hidden_blocks, 1191 + seed)
            .pop()
            .unwrap(),
        up: make_test_q4k_weights(1, n_ff, hidden_blocks, 1193 + seed)
            .pop()
            .unwrap(),
        down: make_test_q6k_weights(1, n_embd, down_blocks, 1197 + seed)
            .pop()
            .unwrap(),
        post_attn_norm: (0..n_embd)
            .map(|i| 0.75 + (i % 17) as f32 * 0.003)
            .collect(),
        ffn_norm: (0..n_embd)
            .map(|i| 0.82 + (i % 19) as f32 * 0.002)
            .collect(),
        post_ffn_norm: (0..n_embd)
            .map(|i| 0.91 + (i % 23) as f32 * 0.0015)
            .collect(),
        hidden: (0..n_embd)
            .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
            .collect(),
        attn_out: (0..q_dim)
            .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
            .collect(),
        n_embd,
        q_dim,
        n_ff,
    }
}

fn cu69_dense_chain_graph_expected(fixture: &Cu69DenseChainGraphFixture, eps: f32) -> Vec<f32> {
    let q_blocks = fixture.q_dim / 256;
    let hidden_blocks = fixture.n_embd / 256;
    let down_blocks = fixture.n_ff / 256;
    let mut expected = fixture.hidden.clone();

    let o_proj = cpu_q4k_gemv_rows(&fixture.o, fixture.n_embd, q_blocks, &fixture.attn_out);
    let post_attn = cpu_rms_norm(&o_proj, &fixture.post_attn_norm, eps, true);
    for i in 0..fixture.n_embd {
        expected[i] += post_attn[i];
    }
    let ffn_input = cpu_rms_norm(&expected, &fixture.ffn_norm, eps, true);
    let mut gate_out = cpu_q4k_gemv_rows(&fixture.gate, fixture.n_ff, hidden_blocks, &ffn_input);
    let up_out = cpu_q4k_gemv_rows(&fixture.up, fixture.n_ff, hidden_blocks, &ffn_input);
    for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
        let x = *gate_value;
        let x3 = x * x * x;
        let c = 0.7978845608028654f32;
        let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
        *gate_value = gelu * *up_value;
    }
    let down_out = cpu_q6k_gemv_rows(&fixture.down, fixture.n_embd, down_blocks, &gate_out);
    let post_ffn = cpu_rms_norm(&down_out, &fixture.post_ffn_norm, eps, true);
    for i in 0..fixture.n_embd {
        expected[i] += post_ffn[i];
    }
    expected
}

fn run_cu69_dense_chain_graph_fixture(
    state: &mut CudaState,
    fixture: &Cu69DenseChainGraphFixture,
    hidden_dev: u64,
    attn_out_dev: u64,
) -> Vec<f32> {
    run_cu69_dense_chain_graph_fixture_with_eps(state, fixture, hidden_dev, attn_out_dev, 1.0e-5)
}

fn run_cu69_dense_chain_graph_fixture_with_eps(
    state: &mut CudaState,
    fixture: &Cu69DenseChainGraphFixture,
    hidden_dev: u64,
    attn_out_dev: u64,
    norm_eps: f32,
) -> Vec<f32> {
    run_cu69_dense_chain_graph_fixture_with_eps_and_allowed(
        state,
        fixture,
        hidden_dev,
        attn_out_dev,
        norm_eps,
        true,
        None,
    )
}

fn run_cu69_dense_chain_graph_fixture_with_eps_and_allowed(
    state: &mut CudaState,
    fixture: &Cu69DenseChainGraphFixture,
    hidden_dev: u64,
    attn_out_dev: u64,
    norm_eps: f32,
    dense_chain_graph_allowed: bool,
    layer_segment_graph_context: Option<Cu71LayerSegmentGraphRuntimeContext>,
) -> Vec<f32> {
    let mut hidden = fixture.hidden.clone();
    let hidden_bytes = std::mem::size_of_val(hidden.as_slice());
    let attn_out_bytes = std::mem::size_of_val(fixture.attn_out.as_slice());
    for _ in 0..3 {
        unsafe {
            state
                .api
                .memcpy_htod_async(
                    hidden_dev,
                    hidden.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    state.stream,
                )
                .expect("hidden h2d");
            state
                .api
                .memcpy_htod_async(
                    attn_out_dev,
                    fixture.attn_out.as_ptr().cast::<libc::c_void>(),
                    attn_out_bytes,
                    state.stream,
                )
                .expect("attn h2d");
        }
        state
            .dense_q4k_attention_output_gelu_ffn_norm_residual(
                &fixture.o,
                &fixture.gate,
                &fixture.up,
                &fixture.down,
                14,
                Some(&fixture.post_attn_norm),
                &fixture.ffn_norm,
                Some(&fixture.post_ffn_norm),
                None,
                None,
                None,
                None,
                None,
                0,
                fixture.q_dim,
                fixture.n_ff,
                fixture.n_embd,
                &mut hidden,
                &fixture.attn_out,
                norm_eps,
                true,
                true,
                true,
                Some(hidden_dev),
                true,
                true,
                None,
                Some(attn_out_dev),
                true,
                dense_chain_graph_allowed,
                layer_segment_graph_context,
            )
            .expect("dense chain graph path");
    }
    let mut output = vec![0.0f32; fixture.n_embd];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                state.stream,
            )
            .expect("hidden dtoh");
    }
    state.stream_synchronize().expect("sync graph fixture");
    output
}

fn cu69_dense_chain_graph_q4k_ple_tail(seed: usize, n_embd: usize) -> Cu69DenseChainGraphPleTail {
    let ple_dim = 256usize;
    let hidden_blocks = n_embd / 256;
    let ple_blocks = ple_dim / 256;
    Cu69DenseChainGraphPleTail {
        gate: make_test_q4k_weights(1, ple_dim, hidden_blocks, 1319 + seed)
            .pop()
            .unwrap(),
        proj: make_test_q4k_weights(1, n_embd, ple_blocks, 1321 + seed)
            .pop()
            .unwrap(),
        post_norm: (0..n_embd)
            .map(|i| 0.84 + (i % 31) as f32 * 0.0013)
            .collect(),
        input: (0..ple_dim)
            .map(|i| 0.58 + ((i as f32 * 0.019).sin() * 0.18))
            .collect(),
        dim: ple_dim,
    }
}

fn cu69_dense_chain_graph_f32_ple_tail(seed: usize, n_embd: usize) -> Cu69DenseChainGraphPleTail {
    let ple_dim = 256usize;
    let gate = (0..ple_dim * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * (0.0005 + seed as f32 * 0.00001))
        .collect::<Vec<_>>();
    let proj = (0..n_embd * ple_dim)
        .map(|i| ((i as f32 % 47.0) - 23.0) * (0.0004 + seed as f32 * 0.00001))
        .collect::<Vec<_>>();
    Cu69DenseChainGraphPleTail {
        gate: f32_to_le_bytes(&gate),
        proj: f32_to_le_bytes(&proj),
        post_norm: (0..n_embd)
            .map(|i| 0.83 + (i % 29) as f32 * 0.0011)
            .collect(),
        input: (0..ple_dim)
            .map(|i| 0.61 + ((i as f32 * 0.017).cos() * 0.16))
            .collect(),
        dim: ple_dim,
    }
}

fn run_cu69_dense_chain_graph_fixture_with_tail(
    state: &mut CudaState,
    fixture: &Cu69DenseChainGraphFixture,
    hidden_dev: u64,
    attn_out_dev: u64,
    ple_tail: Option<&Cu69DenseChainGraphPleTail>,
    ple_input_device_offset: Option<usize>,
    layer_output_scale: Option<f32>,
    dense_chain_graph_allowed: bool,
) -> Vec<f32> {
    if let Some(tail) = ple_tail.filter(|_| ple_input_device_offset.is_some()) {
        state
            .upload_gemma_ple_base(&tail.input)
            .expect("upload graph PLE base");
    }
    let mut hidden = fixture.hidden.clone();
    let hidden_bytes = std::mem::size_of_val(hidden.as_slice());
    let attn_out_bytes = std::mem::size_of_val(fixture.attn_out.as_slice());
    for _ in 0..3 {
        unsafe {
            state
                .api
                .memcpy_htod_async(
                    hidden_dev,
                    hidden.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    state.stream,
                )
                .expect("hidden h2d");
            state
                .api
                .memcpy_htod_async(
                    attn_out_dev,
                    fixture.attn_out.as_ptr().cast::<libc::c_void>(),
                    attn_out_bytes,
                    state.stream,
                )
                .expect("attn h2d");
        }
        state
            .dense_q4k_attention_output_gelu_ffn_norm_residual(
                &fixture.o,
                &fixture.gate,
                &fixture.up,
                &fixture.down,
                14,
                Some(&fixture.post_attn_norm),
                &fixture.ffn_norm,
                Some(&fixture.post_ffn_norm),
                ple_tail.map(|tail| tail.gate.as_slice()),
                ple_tail.map(|tail| tail.proj.as_slice()),
                ple_tail.map(|tail| tail.post_norm.as_slice()),
                ple_tail.map(|tail| tail.input.as_slice()),
                ple_input_device_offset,
                ple_tail.map(|tail| tail.dim).unwrap_or(0),
                fixture.q_dim,
                fixture.n_ff,
                fixture.n_embd,
                &mut hidden,
                &fixture.attn_out,
                1.0e-5,
                true,
                true,
                true,
                Some(hidden_dev),
                true,
                true,
                layer_output_scale,
                Some(attn_out_dev),
                true,
                dense_chain_graph_allowed,
                None,
            )
            .expect("dense chain graph tail path");
    }
    let mut output = vec![0.0f32; fixture.n_embd];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                state.stream,
            )
            .expect("hidden dtoh");
    }
    state.stream_synchronize().expect("sync graph tail fixture");
    output
}

#[test]
fn cuda_dense_q4k_attention_output_gelu_ffn_norm_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let fixture = cu69_dense_chain_graph_fixture(0);
    let mut hidden = fixture.hidden.clone();
    let eps = 1.0e-5;
    let expected = cu69_dense_chain_graph_expected(&fixture, eps);

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain decode reference test: {err}");
            return;
        }
    };
    state
        .dense_q4k_attention_output_gelu_ffn_norm_residual(
            &fixture.o,
            &fixture.gate,
            &fixture.up,
            &fixture.down,
            14,
            Some(&fixture.post_attn_norm),
            &fixture.ffn_norm,
            Some(&fixture.post_ffn_norm),
            None,
            None,
            None,
            None,
            None,
            0,
            fixture.q_dim,
            fixture.n_ff,
            fixture.n_embd,
            &mut hidden,
            &fixture.attn_out,
            eps,
            true,
            true,
            true,
            None,
            false,
            false,
            None,
            None,
            true,
            false,
            None,
        )
        .expect("CUDA dense Q4_K attention+FFN decode chain");

    assert_close_rows_abs_rel(
        "dense Q4_K attention+FFN decode chain",
        &hidden,
        &expected,
        0.35,
        0.015,
    );
}

#[test]
fn cu69_dense_chain_graph_captures_q4k_ple_and_out_scale_after_warmup() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph Q4K PLE capture test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(9);
    let ple_tail = cu69_dense_chain_graph_q4k_ple_tail(9, fixture.n_embd);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");

    let eager_output = run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.75),
        false,
    );
    let graph_output = run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.75),
        true,
    );

    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph Q4K PLE out_scale replay",
        &graph_output,
        &eager_output,
        1.0e-4,
        1.0e-5,
    );
}

#[test]
fn cu69_dense_chain_graph_keys_include_layer_output_scale() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph out_scale key test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(10);
    let ple_tail = cu69_dense_chain_graph_q4k_ple_tail(10, fixture.n_embd);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");

    run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.75),
        true,
    );
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.625),
        true,
    );
    assert_eq!(state.cu69_dense_chain_graphs.len(), 2);
}

#[test]
fn cu69_dense_chain_graph_rejects_host_ple_input() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph host PLE input reject test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(11);
    let ple_tail = cu69_dense_chain_graph_q4k_ple_tail(11, fixture.n_embd);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");

    run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        None,
        Some(0.75),
        true,
    );

    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 0);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 0);
}

#[test]
fn cu69_dense_chain_graph_captures_f32_ple_and_out_scale_after_warmup() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph F32 PLE capture test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(12);
    let ple_tail = cu69_dense_chain_graph_f32_ple_tail(12, fixture.n_embd);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");

    let eager_output = run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.75),
        false,
    );
    let graph_output = run_cu69_dense_chain_graph_fixture_with_tail(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        Some(&ple_tail),
        Some(0),
        Some(0.75),
        true,
    );

    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph F32 PLE out_scale replay",
        &graph_output,
        &eager_output,
        1.0e-4,
        1.0e-5,
    );
}

#[test]
fn cu69_dense_chain_graph_captures_after_warmup() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _trace = EnvVarGuard::remove("RNB_CUDA_DENSE_CHAIN_TRACE");
    let _expert_trace = EnvVarGuard::remove("RNB_CUDA_DENSE_EXPERT_TRACE");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph capture test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(1);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");

    let output = run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);

    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph replay",
        &output,
        &cu69_dense_chain_graph_expected(&fixture, 1.0e-5),
        0.35,
        0.015,
    );
}

#[test]
fn cu71_layer_segment_graph_captures_dense_segment_after_warmup() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU71_LAYER_SEGMENT_GRAPH", "1");
    let _cu69_graph = EnvVarGuard::remove("RNB_CU69_DENSE_CHAIN_GRAPH");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA cu71 layer segment graph capture test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(13);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let layer_segment_context = cu71_layer_segment_graph_runtime_context(&mut state, &fixture, 0);

    let eager_output = run_cu69_dense_chain_graph_fixture_with_eps_and_allowed(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        1.0e-5,
        false,
        None,
    );
    let graph_output = run_cu69_dense_chain_graph_fixture_with_eps_and_allowed(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        1.0e-5,
        true,
        Some(layer_segment_context),
    );

    assert_eq!(state.cu71_layer_segment_graph_warmed.len(), 1);
    assert_eq!(state.cu71_layer_segment_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu71 layer segment graph dense segment replay",
        &graph_output,
        &eager_output,
        1.0e-4,
        1.0e-5,
    );
}

#[test]
fn cu69_dense_chain_graph_reuses_only_matching_key() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph key test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(2);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev_a = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier a");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let output_a =
        run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev_a, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph key a",
        &output_a,
        &cu69_dense_chain_graph_expected(&fixture, 1.0e-5),
        0.35,
        0.015,
    );

    let hidden_dev_b = unsafe { state.api.mem_alloc(hidden_bytes).expect("hidden carrier b") };
    assert_ne!(hidden_dev_a, hidden_dev_b);
    let output_b =
        run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev_b, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 2);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 2);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph key b",
        &output_b,
        &cu69_dense_chain_graph_expected(&fixture, 1.0e-5),
        0.35,
        0.015,
    );
}

#[test]
fn cu69_dense_chain_graph_keys_include_norm_eps() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph norm eps key test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(4);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let output_eps_a = run_cu69_dense_chain_graph_fixture_with_eps(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        1.0e-5,
    );
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph eps a",
        &output_eps_a,
        &cu69_dense_chain_graph_expected(&fixture, 1.0e-5),
        0.35,
        0.015,
    );

    let output_eps_b = run_cu69_dense_chain_graph_fixture_with_eps(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        2.0e-5,
    );
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 2);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 2);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph eps b",
        &output_eps_b,
        &cu69_dense_chain_graph_expected(&fixture, 2.0e-5),
        0.35,
        0.015,
    );
}

#[test]
fn cu69_dense_chain_graph_rekeys_after_mid_buffer_realloc() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph mid-buffer key test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(6);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let output_a =
        run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);

    let old_mid_a = state
        .compute_mid_a_ptr(fixture.n_ff * std::mem::size_of::<f32>())
        .expect("old mid a");
    let old_mid_b = state
        .compute_mid_b_ptr(fixture.n_ff * std::mem::size_of::<f32>())
        .expect("old mid b");
    let grow_bytes = hidden_bytes * 64;
    let new_mid_a = state.compute_mid_a_ptr(grow_bytes).expect("grow mid a");
    let new_mid_b = state.compute_mid_b_ptr(grow_bytes).expect("grow mid b");
    let output_b =
        run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);

    if old_mid_a != new_mid_a || old_mid_b != new_mid_b {
        assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 2);
        assert_eq!(state.cu69_dense_chain_graphs.len(), 2);
    }
    let expected = cu69_dense_chain_graph_expected(&fixture, 1.0e-5);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph mid realloc a",
        &output_a,
        &expected,
        0.35,
        0.015,
    );
    assert_close_rows_abs_rel(
        "cu69 dense chain graph mid realloc b",
        &output_b,
        &expected,
        0.35,
        0.015,
    );
}

#[test]
fn cu69_dense_chain_graph_replay_matches_eager_q8dot_path() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _o_q8 = EnvVarGuard::set("RNB_CUDA_Q4K_GEMV_Q8DOT", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "1");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "1");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph q8dot replay test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture_with_dims(7, 1024, 1024, 1024);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let graph_output =
        run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 1);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 1);

    let eager_output = run_cu69_dense_chain_graph_fixture_with_eps_and_allowed(
        &mut state,
        &fixture,
        hidden_dev,
        attn_out_dev,
        1.0e-5,
        false,
        None,
    );
    assert_close_rows_abs_rel(
        "cu69 dense chain graph q8dot replay",
        &graph_output,
        &eager_output,
        1.0e-4,
        1.0e-5,
    );
}

#[test]
fn cu69_dense_chain_graph_rejects_when_dense_trace_is_active() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _trace = EnvVarGuard::set("RNB_CUDA_DENSE_CHAIN_TRACE", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph trace reject test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(3);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 0);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 0);
}

#[test]
fn cu69_dense_chain_graph_rejects_when_dense_expert_graph_is_active() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _expert_graph = EnvVarGuard::set("RNB_CUDA_DENSE_EXPERT_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph expert graph reject test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(5);
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);
    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 0);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 0);
}

#[test]
fn cu69_dense_chain_graph_rejects_when_raw_weight_pin_falls_back_to_temp() {
    let _guard = runtime_test_lock();
    let _graph = EnvVarGuard::set("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
    let _gate = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA dense chain graph raw weight residency reject test: {err}");
            return;
        }
    };
    let fixture = cu69_dense_chain_graph_fixture(8);
    state.resident_q4k_limit = fixture
        .o
        .len()
        .saturating_add(fixture.gate.len())
        .saturating_add(fixture.up.len());
    let hidden_bytes = fixture.n_embd * std::mem::size_of::<f32>();
    let attn_out_bytes = fixture.q_dim * std::mem::size_of::<f32>();
    let hidden_dev = state
        .decode_hidden_carrier_ptr(hidden_bytes)
        .expect("hidden carrier");
    let attn_out_dev = state
        .decode_attn_out_carrier_ptr(attn_out_bytes)
        .expect("attn carrier");
    let output = run_cu69_dense_chain_graph_fixture(&mut state, &fixture, hidden_dev, attn_out_dev);

    assert_eq!(state.cu69_dense_chain_graph_warmed.len(), 0);
    assert_eq!(state.cu69_dense_chain_graphs.len(), 0);
    assert_close_rows_abs_rel(
        "cu69 dense chain graph raw weight temp fallback",
        &output,
        &cu69_dense_chain_graph_expected(&fixture, 1.0e-5),
        0.35,
        0.015,
    );
}

#[test]
fn cuda_dense_q4k_attention_output_gelu_ffn_batch_f32_ple_device_output_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _gate_q8dot = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    let _down_q8dot = EnvVarGuard::set("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");

    let n_embd = 512usize;
    let q_dim = 512usize;
    let n_ff = 512usize;
    let ple_dim = 256usize;
    let seq_len = 3usize;
    let q_blocks = q_dim / 256;
    let hidden_blocks = n_embd / 256;
    let down_blocks = n_ff / 256;
    let o = make_test_q4k_weights(1, n_embd, q_blocks, 811)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, n_ff, hidden_blocks, 821)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, hidden_blocks, 823)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, down_blocks, 827)
        .pop()
        .unwrap();
    let ple_gate = (0..ple_dim * n_embd)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.0007)
        .collect::<Vec<_>>();
    let ple_proj = (0..n_embd * ple_dim)
        .map(|i| ((i as f32 % 53.0) - 26.0) * 0.0006)
        .collect::<Vec<_>>();
    let ple_gate_bytes = f32_to_le_bytes(&ple_gate);
    let ple_proj_bytes = f32_to_le_bytes(&ple_proj);
    let mut hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
        .collect::<Vec<_>>();
    let initial_hidden = hidden.clone();
    let attn_out = (0..seq_len * q_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.75 + (i % 17) as f32 * 0.003)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.82 + (i % 19) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.91 + (i % 23) as f32 * 0.0015)
        .collect::<Vec<_>>();
    let ple_input = (0..seq_len * ple_dim)
        .map(|i| 0.6 + ((i as f32 * 0.017).sin() * 0.2))
        .collect::<Vec<_>>();
    let ple_post_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 29) as f32 * 0.0015)
        .collect::<Vec<_>>();
    let eps = 1.0e-5;

    let mut expected = hidden.clone();
    for token in 0..seq_len {
        let row = token * n_embd;
        let ple_row = token * ple_dim;
        let attn_row = &attn_out[token * q_dim..(token + 1) * q_dim];
        let o_proj = cpu_q4k_gemv_rows(&o, n_embd, q_blocks, attn_row);
        let post_attn = cpu_rms_norm(&o_proj, &post_attn_norm, eps, true);
        for i in 0..n_embd {
            expected[row + i] += post_attn[i];
        }
        let ffn_input = cpu_rms_norm(&expected[row..row + n_embd], &ffn_norm, eps, true);
        let mut gate_out = cpu_q4k_gemv_rows(&gate, n_ff, hidden_blocks, &ffn_input);
        let up_out = cpu_q4k_gemv_rows(&up, n_ff, hidden_blocks, &ffn_input);
        for (gate_value, up_value) in gate_out.iter_mut().zip(up_out.iter()) {
            let x = *gate_value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *gate_value = gelu * *up_value;
        }
        let down_out = cpu_q6k_gemv_rows(&down, n_embd, down_blocks, &gate_out);
        let post_ffn = cpu_rms_norm(&down_out, &post_ffn_norm, eps, true);
        for i in 0..n_embd {
            expected[row + i] += post_ffn[i];
        }
        let mut ple_gate_out =
            cpu_f32_gemv_rows(&ple_gate, ple_dim, n_embd, &expected[row..row + n_embd]);
        for (value, &scale) in ple_gate_out
            .iter_mut()
            .zip(ple_input[ple_row..ple_row + ple_dim].iter())
        {
            let x = *value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *value = gelu * scale;
        }
        let projected = cpu_f32_gemv_rows(&ple_proj, n_embd, ple_dim, &ple_gate_out);
        let normed = cpu_rms_norm(&projected, &ple_post_norm, eps, false);
        for i in 0..n_embd {
            expected[row + i] += normed[i];
        }
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let attn_out_bytes = std::mem::size_of_val(attn_out.as_slice());
    let attn_out_dev = state
        .compute_full_down_ptr(attn_out_bytes)
        .expect("attention output device scratch");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                attn_out_dev,
                attn_out.as_ptr().cast::<libc::c_void>(),
                attn_out_bytes,
                state.stream,
            )
            .expect("attention output h2d");
    }
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let output_id = state
        .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
            &o,
            &gate,
            &up,
            &down,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            Some(&ple_gate_bytes),
            Some(&ple_proj_bytes),
            Some(&ple_post_norm),
            Some(&ple_input),
            ple_dim,
            q_dim,
            n_ff,
            n_embd,
            seq_len,
            &mut hidden,
            None,
            attn_out_dev,
            Some(output_desc),
            None,
            eps,
            true,
            true,
            true,
        )
        .expect("CUDA dense Q4_K attention+FFN F32 PLE device output")
        .expect("device output id");
    let device_hidden = state
        .download_device_tensor_f32(output_id)
        .expect("download F32 PLE dense output");
    assert!(state
        .release_device_tensor(output_id)
        .expect("release F32 PLE dense output"));
    let ple_gate_key = f32_key(&ple_gate);
    let ple_proj_key = f32_key(&ple_proj);
    assert!(
        state
            .resident_f32
            .keys()
            .any(|key| key.len == ple_gate_key.len && key.bit_hash == ple_gate_key.bit_hash),
        "F32 PLE gate weight must remain resident"
    );
    assert!(
        state
            .resident_f32
            .keys()
            .any(|key| key.len == ple_proj_key.len && key.bit_hash == ple_proj_key.bit_hash),
        "F32 PLE projection weight must remain resident"
    );

    assert_eq!(initial_hidden.len(), seq_len * n_embd);
    assert_close_rows_abs_rel(
        "dense Q4_K attention+FFN batch chain F32 PLE device output",
        &device_hidden,
        &expected,
        0.35,
        0.015,
    );
}

#[test]
fn cuda_gemma4_ple_q4k_batch_norm_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let prev_batch = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
    std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", "0");

    let n_embd = 512usize;
    let ple_dim = 256usize;
    let seq_len = 3usize;
    let hidden_blocks = n_embd / 256;
    let ple_blocks = ple_dim / 256;
    let gate = make_test_q4k_weights(1, ple_dim, hidden_blocks, 307)
        .pop()
        .unwrap();
    let proj = make_test_q4k_weights(1, n_embd, ple_blocks, 311)
        .pop()
        .unwrap();
    let mut hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.006)
        .collect::<Vec<_>>();
    let ple_input = (0..seq_len * ple_dim)
        .map(|i| 0.65 + ((i as f32 * 0.013).sin() * 0.25))
        .collect::<Vec<_>>();
    let post_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 29) as f32 * 0.0015)
        .collect::<Vec<_>>();
    let eps = 1.0e-5;

    let mut expected = hidden.clone();
    for token in 0..seq_len {
        let hidden_off = token * n_embd;
        let ple_off = token * ple_dim;
        let mut gate_out = cpu_q4k_gemv_rows(
            &gate,
            ple_dim,
            hidden_blocks,
            &expected[hidden_off..hidden_off + n_embd],
        );
        for (value, &scale) in gate_out
            .iter_mut()
            .zip(ple_input[ple_off..ple_off + ple_dim].iter())
        {
            let x = *value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *value = gelu * scale;
        }
        let projected = cpu_q4k_gemv_rows(&proj, n_embd, ple_blocks, &gate_out);
        let normed = cpu_rms_norm(&projected, &post_norm, eps, false);
        for i in 0..n_embd {
            expected[hidden_off + i] += normed[i];
        }
    }
    let out_scale = [0.75f32];
    for value in &mut expected {
        *value *= out_scale[0];
    }

    gemma4_ple_q4k_batch_norm_residual(
        &gate,
        &proj,
        &post_norm,
        Some(&out_scale),
        &ple_input,
        ple_dim,
        n_embd,
        seq_len,
        &mut hidden,
        eps,
    )
    .expect("CUDA Gemma4 PLE Q4_K batch chain");

    if let Some(prev) = prev_batch {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT");
    }

    assert_close_rows_abs_rel("Gemma4 PLE Q4_K batch", &hidden, &expected, 0.35, 0.01);
}

#[test]
fn cuda_gemma4_ple_f32_batch_norm_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();

    let n_embd = 96usize;
    let ple_dim = 64usize;
    let seq_len = 3usize;
    let gate = (0..ple_dim * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.0008)
        .collect::<Vec<_>>();
    let proj = (0..n_embd * ple_dim)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.0006)
        .collect::<Vec<_>>();
    let gate_bytes = f32_to_le_bytes(&gate);
    let proj_bytes = f32_to_le_bytes(&proj);
    let mut hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.004)
        .collect::<Vec<_>>();
    let ple_input = (0..seq_len * ple_dim)
        .map(|i| 0.7 + ((i as f32 * 0.019).sin() * 0.15))
        .collect::<Vec<_>>();
    let post_norm = (0..n_embd)
        .map(|i| 0.85 + (i % 23) as f32 * 0.001)
        .collect::<Vec<_>>();
    let out_scale = [0.8f32];
    let eps = 1.0e-5;

    let mut expected = hidden.clone();
    for token in 0..seq_len {
        let hidden_off = token * n_embd;
        let ple_off = token * ple_dim;
        let mut gate_out = cpu_f32_gemv_rows(
            &gate,
            ple_dim,
            n_embd,
            &expected[hidden_off..hidden_off + n_embd],
        );
        for (value, &scale) in gate_out
            .iter_mut()
            .zip(ple_input[ple_off..ple_off + ple_dim].iter())
        {
            let x = *value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *value = gelu * scale;
        }
        let projected = cpu_f32_gemv_rows(&proj, n_embd, ple_dim, &gate_out);
        let normed = cpu_rms_norm(&projected, &post_norm, eps, false);
        for i in 0..n_embd {
            expected[hidden_off + i] += normed[i];
        }
    }
    for value in &mut expected {
        *value *= out_scale[0];
    }

    let mut state = CudaState::open().expect("open CUDA state");
    state
        .gemma4_ple_q4k_batch_norm_residual(
            &gate_bytes,
            &proj_bytes,
            &post_norm,
            Some(&out_scale),
            &ple_input,
            ple_dim,
            n_embd,
            seq_len,
            &mut hidden,
            eps,
        )
        .expect("CUDA Gemma4 PLE F32 batch chain");
    let gate_key = f32_key(&gate);
    let proj_key = f32_key(&proj);
    assert!(
        state
            .resident_f32
            .keys()
            .any(|key| key.len == gate_key.len && key.bit_hash == gate_key.bit_hash),
        "F32 PLE gate weight must remain resident"
    );
    assert!(
        state
            .resident_f32
            .keys()
            .any(|key| key.len == proj_key.len && key.bit_hash == proj_key.bit_hash),
        "F32 PLE projection weight must remain resident"
    );

    assert_close_rows_abs_rel("Gemma4 PLE F32 batch", &hidden, &expected, 0.03, 0.001);
}

#[test]
fn cuda_gemma4_ple_f32_debug_stages_match_cpu_reference() {
    let _guard = runtime_test_lock();

    let n_embd = 2560usize;
    let ple_dim = 256usize;
    let seq_len = 65usize;
    let gate = (0..ple_dim * n_embd)
        .map(|i| {
            let periodic = ((i as f32 % 89.0) - 44.0) * 0.00022;
            periodic + (i as f32 * 0.00037).sin() * 0.00011
        })
        .collect::<Vec<_>>();
    let proj = (0..n_embd * ple_dim)
        .map(|i| {
            let periodic = ((i as f32 % 83.0) - 41.0) * 0.00018;
            periodic + (i as f32 * 0.00041).cos() * 0.00009
        })
        .collect::<Vec<_>>();
    let gate_bytes = f32_to_le_bytes(&gate);
    let proj_bytes = f32_to_le_bytes(&proj);
    let hidden = (0..seq_len * n_embd)
        .map(|i| {
            let periodic = ((i as f32 % 97.0) - 48.0) * 0.006;
            periodic + (i as f32 * 0.013).sin() * 0.025
        })
        .collect::<Vec<_>>();
    let ple_input = (0..seq_len * ple_dim)
        .map(|i| 0.65 + ((i as f32 * 0.017).sin() * 0.18))
        .collect::<Vec<_>>();
    let post_norm = (0..n_embd)
        .map(|i| 0.78 + (i % 31) as f32 * 0.00125)
        .collect::<Vec<_>>();
    let out_scale = [0.0625f32];
    let eps = 1.0e-5;

    let mut expected_gate = vec![0.0f32; seq_len * ple_dim];
    let mut expected_gated = vec![0.0f32; seq_len * ple_dim];
    let mut expected_projected = vec![0.0f32; seq_len * n_embd];
    let mut expected_final = hidden.clone();
    for token in 0..seq_len {
        let hidden_off = token * n_embd;
        let ple_off = token * ple_dim;
        let mut gate_out = cpu_f32_gemv_rows(
            &gate,
            ple_dim,
            n_embd,
            &hidden[hidden_off..hidden_off + n_embd],
        );
        expected_gate[ple_off..ple_off + ple_dim].copy_from_slice(&gate_out);
        for (value, &scale) in gate_out
            .iter_mut()
            .zip(ple_input[ple_off..ple_off + ple_dim].iter())
        {
            let x = *value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *value = gelu * scale;
        }
        expected_gated[ple_off..ple_off + ple_dim].copy_from_slice(&gate_out);
        let projected = cpu_f32_gemv_rows(&proj, n_embd, ple_dim, &gate_out);
        expected_projected[hidden_off..hidden_off + n_embd].copy_from_slice(&projected);
        let normed = cpu_rms_norm(&projected, &post_norm, eps, false);
        for i in 0..n_embd {
            expected_final[hidden_off + i] += normed[i];
        }
    }
    for value in &mut expected_final {
        *value *= out_scale[0];
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let stages = state
        .gemma4_ple_f32_debug_stages(
            &gate_bytes,
            &proj_bytes,
            &post_norm,
            Some(&out_scale),
            &ple_input,
            ple_dim,
            n_embd,
            seq_len,
            &hidden,
            eps,
        )
        .expect("CUDA Gemma4 PLE F32 debug stages");

    for (label, actual, expected) in [
        ("gate", stages.gate.as_slice(), expected_gate.as_slice()),
        ("gated", stages.gated.as_slice(), expected_gated.as_slice()),
        (
            "projected",
            stages.projected.as_slice(),
            expected_projected.as_slice(),
        ),
        (
            "final",
            stages.final_hidden.as_slice(),
            expected_final.as_slice(),
        ),
    ] {
        let (idx, max_abs, max_rel) = max_abs_rel(actual, expected);
        eprintln!(
            "Gemma4 PLE F32 debug {label}: max_abs={max_abs:e} max_rel={max_rel:e} idx={idx}"
        );
    }

    assert_close_rows_abs_rel(
        "Gemma4 PLE F32 debug gate",
        &stages.gate,
        &expected_gate,
        5.0e-4,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 PLE F32 debug gated",
        &stages.gated,
        &expected_gated,
        5.0e-4,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 PLE F32 debug projected",
        &stages.projected,
        &expected_projected,
        1.0e-3,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 PLE F32 debug final",
        &stages.final_hidden,
        &expected_final,
        1.0e-3,
        5.0e-3,
    );
}

#[test]
fn cuda_gemma4_ple_f32_real_replay_dump_matches_host_reference() {
    let Some(dir) = std::env::var_os("RNB_DEBUG_GEMMA4_PLE_REPLAY_DIR") else {
        eprintln!(
            "skipping real Gemma4 PLE replay dump test: RNB_DEBUG_GEMMA4_PLE_REPLAY_DIR unset"
        );
        return;
    };
    let _guard = runtime_test_lock();
    let dir = std::path::PathBuf::from(dir);
    let layer_idx = std::env::var("RNB_DEBUG_GEMMA4_PLE_REPLAY_LAYER")
        .ok()
        .map(|raw| raw.parse::<usize>().expect("parse replay layer"))
        .unwrap_or(0);
    let host_path = |name: &str| dir.join(format!("host_L{layer_idx}_{name}.bin"));

    let hidden = read_f32_bin(&host_path("ple_hidden"));
    let ple_input = read_f32_bin(&host_path("ple_input"));
    let gate = read_f32_bin(&host_path("ple_gate_weight"));
    let proj = read_f32_bin(&host_path("ple_proj_weight"));
    let post_norm = read_f32_bin(&host_path("ple_post_norm"));
    let expected_gate = read_f32_bin(&host_path("ple_gate"));
    let expected_gated = read_f32_bin(&host_path("ple_gated"));
    let expected_projected = read_f32_bin(&host_path("ple_projected"));
    let expected_final = read_optional_f32_bin(&host_path("ple_final_scaled"))
        .unwrap_or_else(|| read_f32_bin(&host_path("ple_final")));
    let out_scale = read_optional_f32_bin(&host_path("ple_out_scale"));

    let n_embd = post_norm.len();
    assert_ne!(n_embd, 0, "replay post_norm must be non-empty");
    assert_eq!(
        hidden.len() % n_embd,
        0,
        "replay hidden len must divide by n_embd"
    );
    let seq_len = hidden.len() / n_embd;
    assert_ne!(seq_len, 0, "replay seq_len must be non-zero");
    assert_eq!(
        ple_input.len() % seq_len,
        0,
        "replay PLE input len must divide by seq_len"
    );
    let ple_dim = ple_input.len() / seq_len;
    assert_eq!(gate.len(), ple_dim * n_embd);
    assert_eq!(proj.len(), n_embd * ple_dim);
    assert_eq!(expected_gate.len(), seq_len * ple_dim);
    assert_eq!(expected_gated.len(), seq_len * ple_dim);
    assert_eq!(expected_projected.len(), seq_len * n_embd);
    assert_eq!(expected_final.len(), seq_len * n_embd);

    let mut state = CudaState::open().expect("open CUDA state");
    let stages = state
        .gemma4_ple_f32_debug_stages(
            &f32_to_le_bytes(&gate),
            &f32_to_le_bytes(&proj),
            &post_norm,
            out_scale.as_deref(),
            &ple_input,
            ple_dim,
            n_embd,
            seq_len,
            &hidden,
            1.0e-5,
        )
        .expect("CUDA Gemma4 real PLE replay stages");

    for (label, actual, expected) in [
        ("gate", stages.gate.as_slice(), expected_gate.as_slice()),
        ("gated", stages.gated.as_slice(), expected_gated.as_slice()),
        (
            "projected",
            stages.projected.as_slice(),
            expected_projected.as_slice(),
        ),
        (
            "final",
            stages.final_hidden.as_slice(),
            expected_final.as_slice(),
        ),
    ] {
        let (idx, max_abs, max_rel) = max_abs_rel(actual, expected);
        eprintln!(
            "Gemma4 real PLE replay {label}: max_abs={max_abs:e} max_rel={max_rel:e} idx={idx}"
        );
    }

    assert_close_rows_abs_rel(
        "Gemma4 real PLE replay gate",
        &stages.gate,
        &expected_gate,
        5.0e-4,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 real PLE replay gated",
        &stages.gated,
        &expected_gated,
        5.0e-4,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 real PLE replay projected",
        &stages.projected,
        &expected_projected,
        1.0e-3,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Gemma4 real PLE replay final",
        &stages.final_hidden,
        &expected_final,
        1.0e-3,
        5.0e-3,
    );

    let cuda_call = std::env::var("RNB_DEBUG_GEMMA4_PLE_REPLAY_BACKEND_CALL")
        .ok()
        .map(|raw| raw.parse::<usize>().expect("parse replay backend call"))
        .unwrap_or(0);
    for (name, expected) in [
        ("ple_hidden", hidden.as_slice()),
        ("ple_gate", expected_gate.as_slice()),
        ("ple_gated", expected_gated.as_slice()),
        ("ple_projected", expected_projected.as_slice()),
        ("ple_final_scaled", expected_final.as_slice()),
    ] {
        let path = dir.join(format!("cuda_call{cuda_call}_{name}.bin"));
        if let Some(actual) = read_optional_f32_bin(&path) {
            let (idx, max_abs, max_rel) = max_abs_rel(&actual, expected);
            eprintln!(
                "Gemma4 fused CUDA dump {name}: max_abs={max_abs:e} max_rel={max_rel:e} idx={idx}"
            );
        }
    }
}

#[test]
fn cuda_prefill_attention_f16kv_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");

    let seq_len = 3usize;
    let kv_len = 3usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let n_embd = 512usize;
    let n_ff = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let blocks = n_embd / 256;
    let o = make_test_q4k_weights(1, n_embd, blocks, 211).pop().unwrap();
    let gate = make_test_q4k_weights(1, n_ff, blocks, 223).pop().unwrap();
    let up = make_test_q4k_weights(1, n_ff, blocks, 227).pop().unwrap();
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 229)
        .pop()
        .unwrap();
    let q = (0..seq_len * num_heads * head_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.01)
        .collect::<Vec<_>>();
    let k = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.012).to_bits())
        .collect::<Vec<_>>();
    let v = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 37.0) - 18.0) * 0.009).to_bits())
        .collect::<Vec<_>>();
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let attn = attention_prefill_flash_hd512_f16kv(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    )
    .expect("CUDA f16KV attention");
    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate path");

    let mut actual = hidden;
    attention_prefill_flash_hd512_f16kv_dense_chain(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        &mut actual,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA f16KV attention dense chain");

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }
    assert_close_rows_abs_rel(
        "f16KV attention dense chain",
        &actual,
        &expected,
        0.02,
        0.01,
    );
}

#[test]
fn cuda_prefill_attention_f16kv_window_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");

    let seq_len = 5usize;
    let kv_len = 5usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let n_embd = 512usize;
    let n_ff = 512usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let blocks = n_embd / 256;
    let o = make_test_q4k_weights(1, n_embd, blocks, 231).pop().unwrap();
    let gate = make_test_q4k_weights(1, n_ff, blocks, 233).pop().unwrap();
    let up = make_test_q4k_weights(1, n_ff, blocks, 239).pop().unwrap();
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 241)
        .pop()
        .unwrap();
    let q = (0..seq_len * num_heads * head_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.01)
        .collect::<Vec<_>>();
    let k = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.012).to_bits())
        .collect::<Vec<_>>();
    let v = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 37.0) - 18.0) * 0.009).to_bits())
        .collect::<Vec<_>>();
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let attn = attention_prefill_flash_hd512_f16kv_window(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
    )
    .expect("CUDA f16KV window attention");
    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate window path");

    let mut actual = hidden;
    attention_prefill_flash_hd512_f16kv_window_dense_chain(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        &mut actual,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA f16KV window attention dense chain");

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }
    assert_close_rows_abs_rel(
        "f16KV window attention dense chain",
        &actual,
        &expected,
        0.02,
        0.01,
    );
}

#[test]
fn cuda_prefill_attention_hd256_f16kv_window_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");

    let seq_len = 5usize;
    let kv_len = 5usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let n_embd = 512usize;
    let n_ff = 512usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let o_blocks = head_dim / 256;
    let hidden_blocks = n_embd / 256;
    let o = make_test_q4k_weights(1, n_embd, o_blocks, 251)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, n_ff, hidden_blocks, 253)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, hidden_blocks, 257)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 263)
        .pop()
        .unwrap();
    let q = (0..seq_len * num_heads * head_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.01)
        .collect::<Vec<_>>();
    let k = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.012).to_bits())
        .collect::<Vec<_>>();
    let v = (0..kv_len * num_kv_heads * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 37.0) - 18.0) * 0.009).to_bits())
        .collect::<Vec<_>>();
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let attn = attention_prefill_flash_hd256_f16kv_window(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
    )
    .expect("CUDA hd256 f16KV window attention");
    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate hd256 window path");

    let mut actual = hidden;
    attention_prefill_flash_hd256_f16kv_window_dense_chain(
        &q,
        &k,
        &v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
        &o,
        &gate,
        &up,
        &down,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        &mut actual,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA hd256 f16KV window attention dense chain");

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }
    assert_close_rows_abs_rel(
        "hd256 f16KV window attention dense chain",
        &actual,
        &expected,
        0.02,
        0.01,
    );
}

#[test]
fn cuda_q4k_f16_q_cached_f16kv_hd256_window_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");

    let seq_len = 5usize;
    let kv_len = 5usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let n_embd = 512usize;
    let n_ff = 512usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_blocks = n_embd / 256;
    let o_blocks = head_dim / 256;
    let mut q_weights = make_test_q4k_weights(1, head_dim, hidden_blocks, 491)
        .pop()
        .unwrap();
    let mut k_weights = make_test_q4k_weights(1, head_dim, hidden_blocks, 499)
        .pop()
        .unwrap();
    let mut v_weights = make_test_q4k_weights(1, head_dim, hidden_blocks, 503)
        .pop()
        .unwrap();
    let mut o_weights = make_test_q4k_weights(1, n_embd, o_blocks, 509)
        .pop()
        .unwrap();
    let mut gate_weights = make_test_q4k_weights(1, n_ff, hidden_blocks, 521)
        .pop()
        .unwrap();
    let mut up_weights = make_test_q4k_weights(1, n_ff, hidden_blocks, 523)
        .pop()
        .unwrap();
    let mut down_weights = make_test_q6k_weights(1, n_embd, n_ff / 256, 541)
        .pop()
        .unwrap();
    let shrink_q4 = |weights: &mut [u8]| {
        for block in weights.chunks_exact_mut(144) {
            block[0..2].copy_from_slice(&half::f16::from_f32(0.000244140625).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(0.0001220703125).to_le_bytes());
        }
    };
    shrink_q4(&mut q_weights);
    shrink_q4(&mut k_weights);
    shrink_q4(&mut v_weights);
    shrink_q4(&mut o_weights);
    shrink_q4(&mut gate_weights);
    shrink_q4(&mut up_weights);
    for block in down_weights.chunks_exact_mut(210) {
        block[208..210].copy_from_slice(&half::f16::from_f32(0.000244140625).to_le_bytes());
    }
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.003)
        .collect::<Vec<_>>();
    let attn_norm = (0..n_embd)
        .map(|i| 0.64 + (i % 11) as f32 * 0.00275)
        .collect::<Vec<_>>();
    let mut normed_input = Vec::with_capacity(seq_len * n_embd);
    for row in hidden.chunks_exact(n_embd) {
        normed_input.extend(cpu_rms_norm(row, &attn_norm, 1.0e-5, true));
    }
    let q_norm = (0..head_dim)
        .map(|i| 0.72 + (i % 13) as f32 * 0.00625)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.83 + (i % 17) as f32 * 0.0046875)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q6_f16_limit = usize::MAX;
    let (q, k_bits, v_bits) = state
        .q4k_f16_qkv_postprocess_hd256(
            &q_weights,
            &k_weights,
            &v_weights,
            head_dim,
            head_dim,
            hidden_blocks,
            seq_len,
            &normed_input,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            10000.0,
            0,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 postprocess state API")
        .expect("Q4_K F16 cache admitted hd256 cached-KV source test weights");
    let attn = attention_prefill_flash_hd256_f16kv_window(
        &q,
        &k_bits,
        &v_bits,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        window,
    )
    .expect("CUDA hd256 f16KV window attention");

    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o_weights,
        &gate_weights,
        &up_weights,
        &down_weights,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        head_dim,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate cached-KV hd256 window path");
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "Q4_K F16 Q cached hd256 window dense chain reference must stay finite"
    );

    let mut actual = hidden;
    let hidden_input = actual.clone();
    let completed = state
        .q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
            &q_weights,
            head_dim,
            hidden_blocks,
            seq_len,
            kv_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_bits,
            &v_bits,
            num_heads,
            num_kv_heads,
            scale,
            10000.0,
            0,
            1.0e-5,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            head_dim,
            n_ff,
            n_embd,
            &mut actual,
            None,
            None,
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 Q cached hd256 window dense chain state API");

    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let mut device_hidden = hidden_input.clone();
    let device_completed = state
        .q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
            &q_weights,
            head_dim,
            hidden_blocks,
            seq_len,
            kv_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_bits,
            &v_bits,
            num_heads,
            num_kv_heads,
            scale,
            10000.0,
            0,
            1.0e-5,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            head_dim,
            n_ff,
            n_embd,
            &mut device_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 Q cached hd256 window dense chain device-output state API");
    let device_output_id = match device_completed {
        Some(Q4kF16DenseChainOutput::Device(id)) => id,
        other => panic!("expected hd256 device output, got {other:?}"),
    };
    let device_actual = state
        .download_device_tensor_f32(device_output_id)
        .expect("download Q4_K F16 Q cached hd256 window dense chain device output");
    assert!(state
        .release_device_tensor(device_output_id)
        .expect("release Q4_K F16 Q cached hd256 window dense chain device output"));

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }

    assert!(matches!(completed, Some(Q4kF16DenseChainOutput::Host)));
    assert_close_rows_abs_rel(
        "Q4_K F16 Q cached f16KV hd256 window dense chain",
        &actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 Q cached f16KV hd256 window dense chain device output",
        &device_actual,
        &expected,
        0.03,
        0.01,
    );
}

#[test]
fn cuda_q4k_f32_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let rows = 512usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 151)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual =
        q4k_f32_gemm_batch_for_test(&weights, rows, cols, &input).expect("CUDA Q4_K F32 GEMM");
    assert_close_rows("Q4_K F32 GEMM batch", &actual, &expected, 0.04);
}

#[test]
fn cuda_q4k_f32_cached_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let prev_enabled = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
    let prev_cache = std::env::var("RNB_CUDA_Q4_F32_CACHE_MB").ok();
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "1");
    std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    allow_default_cuda_q4_f32_cache_for_test();

    let rows = 512usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 163)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual = q4k_f32_gemm_batch_cached(&weights, rows, cols, &input)
        .expect("CUDA cached Q4_K F32 GEMM")
        .expect("Q4_K F32 cache admitted test weight");

    if let Some(prev) = prev_enabled {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
    }
    if let Some(prev) = prev_cache {
        std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4_F32_CACHE_MB");
    }

    assert_close_rows("cached Q4_K F32 GEMM batch", &actual, &expected, 0.04);
}

#[test]
fn cuda_q4k_f16_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let rows = 512usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 157)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.005)
        .collect::<Vec<_>>();
    let input_f16 = input
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let weights_dev = state
        .resident_q4k_f16_pair_ptrs(&weights, &weights, rows, blocks_per_row)
        .expect("Q4_K F16 resident cache")
        .expect("Q4_K F16 resident cache enabled")
        .0;
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input_f16.as_slice()))
        .expect("input buffer");
    let output_len = seq_len * rows;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let output_dev = state
        .compute_output_ptr(output_bytes)
        .expect("output buffer");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input_f16.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input_f16.as_slice()),
                state.stream,
            )
            .expect("input h2d");
    }
    state
        .hgemm_to_f32_device(weights_dev, rows, cols, input_dev, seq_len, output_dev)
        .expect("CUDA Q4_K F16 GEMM");
    let mut actual = vec![0.0f32; output_len];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )
            .expect("output dtoh");
    }
    state.stream_synchronize().expect("sync");

    assert_close_rows("Q4_K F16 GEMM batch", &actual, &expected, 0.08);
}

#[test]
fn cuda_q4k_transient_f16_upload_works_when_resident_cache_disabled() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let rows = 512usize;
    let blocks_per_row = 2usize;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 223)
        .pop()
        .unwrap();
    let mut state = CudaState::open().expect("open CUDA state");

    let ptr = state
        .transient_q4k_f16_ptr(&weights, rows, blocks_per_row)
        .expect("transient Q4_K F16 upload");

    assert_ne!(ptr, 0);
}

#[test]
fn cuda_q6_f16_down_policy_respects_activation_ceiling_and_force() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _mode = EnvVarGuard::set("RNB_CUDA_Q6K_BATCH_F16_DOWN", "1");
    let _cutoff = EnvVarGuard::remove("RNB_CUDA_Q6K_BATCH_F16_DOWN_MAX_ACTS");
    let _prewarm = EnvVarGuard::remove("RNB_CUDA_Q6K_BATCH_F16_PREWARM");

    assert!(dense_q6_batch_f16_down_enabled_for(false, 128, 10_752));
    assert!(dense_q6_batch_f16_down_enabled_for(false, 1024, 10_752));
    assert!(!dense_q6_batch_f16_down_enabled_for(false, 16_384, 10_752));
    assert!(!q6_f16_prewarm_enabled());

    std::env::set_var("RNB_CUDA_Q6K_BATCH_F16_DOWN", "force");
    assert!(dense_q6_batch_f16_down_enabled_for(false, 16_384, 10_752));
    assert!(q6_f16_prewarm_enabled());

    std::env::set_var("RNB_CUDA_Q6K_BATCH_F16_DOWN", "1");
    std::env::set_var("RNB_CUDA_Q6K_BATCH_F16_PREWARM", "1");
    assert!(q6_f16_prewarm_enabled());
}

#[test]
fn cuda_q4k_f16_gemm_batch_state_api_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f16 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    let rows = 1024usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 163)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let actual = state
        .q4k_f16_gemm_batch(&weights, rows, blocks_per_row, seq_len, &input)
        .expect("CUDA Q4_K F16 GEMM batch state API")
        .expect("Q4_K F16 cache admitted test weight");

    assert_close_rows_abs_rel(
        "Q4_K F16 GEMM batch state API",
        &actual,
        &expected,
        0.35,
        0.01,
    );
}

#[test]
fn cuda_q4k_f16_qkv_gemm_batch_state_api_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 167)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 173)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 179)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected_q = Vec::with_capacity(seq_len * q_rows);
    let mut expected_k = Vec::with_capacity(seq_len * kv_rows);
    let mut expected_v = Vec::with_capacity(seq_len * kv_rows);
    for token in 0..seq_len {
        let input_row = &input[token * cols..(token + 1) * cols];
        expected_q.extend(cpu_q4k_gemv_rows(
            &q_weights,
            q_rows,
            blocks_per_row,
            input_row,
        ));
        expected_k.extend(cpu_q4k_gemv_rows(
            &k_weights,
            kv_rows,
            blocks_per_row,
            input_row,
        ));
        expected_v.extend(cpu_q4k_gemv_rows(
            &v_weights,
            kv_rows,
            blocks_per_row,
            input_row,
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    let (actual_q, actual_k, actual_v) = state
        .q4k_f16_qkv_gemm_batch(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
        )
        .expect("CUDA Q4_K F16 QKV GEMM batch state API")
        .expect("Q4_K F16 cache admitted QKV test weights");

    assert_close_rows_abs_rel(
        "Q4_K F16 Q GEMM batch state API",
        &actual_q,
        &expected_q,
        0.35,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 K GEMM batch state API",
        &actual_k,
        &expected_k,
        0.35,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 V GEMM batch state API",
        &actual_v,
        &expected_v,
        0.35,
        0.01,
    );
}

#[test]
fn cuda_q4k_f16_qkv_prefill_attention_hd512_matches_cpu_postprocess_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rope_theta = 10000.0f32;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 197)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 199)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 211)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0035)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.75 + (i % 17) as f32 * 0.00625)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.80 + (i % 19) as f32 * 0.0046875)
        .collect::<Vec<_>>();
    let freq_factors = (0..head_dim / 2)
        .map(|i| 1.0 + (i % 11) as f32 * 0.015625)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    let (q_raw, k_raw, v_raw) = state
        .q4k_f16_qkv_gemm_batch(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
        )
        .expect("CUDA Q4_K F16 QKV GEMM batch state API")
        .expect("Q4_K F16 cache admitted QKV test weights");
    let q_post = cpu_qk_norm_rope_neox_hd512(
        &q_raw,
        &q_norm,
        seq_len,
        num_heads,
        head_dim,
        0,
        1.0e-5,
        rope_theta,
        Some(&freq_factors),
        false,
    );
    let k_post = cpu_qk_norm_rope_neox_hd512(
        &k_raw,
        &k_norm,
        seq_len,
        num_kv_heads,
        head_dim,
        0,
        1.0e-5,
        rope_theta,
        Some(&freq_factors),
        false,
    );
    let v_post = cpu_v_norm_pack_hd512(&v_raw, seq_len, num_kv_heads, head_dim, 1.0e-5);
    let k_bits = k_post
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let expected_v_bits = v_post
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let expected = cpu_prefill_attention_hd512_f16kv(
        &q_post,
        &k_bits,
        &expected_v_bits,
        seq_len,
        seq_len,
        num_heads,
        num_kv_heads,
        scale,
    );

    let (actual, actual_k_bits, actual_v_bits) = state
        .q4k_f16_qkv_prefill_attention_hd512(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            Some(&freq_factors),
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            false,
            false,
            true,
        )
        .expect("CUDA Q4_K F16 QKV prefill attention state API")
        .expect("Q4_K F16 cache admitted fused QKV attention test weights");

    assert_eq!(actual_k_bits, k_bits);
    assert_eq!(actual_v_bits, expected_v_bits);
    assert_close_rows(
        "Q4_K F16 QKV fused prefill attention",
        &actual,
        &expected,
        1.0e-4,
    );
}

#[test]
fn cuda_q4k_f16_qkv_prefill_attention_hd512_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    let prev_gate_f16 = std::env::var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    std::env::set_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");

    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let n_embd = cols;
    let n_ff = 1024usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rope_theta = 10000.0f32;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 401)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 409)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 419)
        .pop()
        .unwrap();
    let o_weights = make_test_q4k_weights(1, n_embd, q_rows / 256, 421)
        .pop()
        .unwrap();
    let gate_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 431)
        .pop()
        .unwrap();
    let up_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 433)
        .pop()
        .unwrap();
    let down_weights = make_test_q6k_weights(1, n_embd, n_ff / 256, 439)
        .pop()
        .unwrap();
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.0035)
        .collect::<Vec<_>>();
    let attn_norm = (0..n_embd)
        .map(|i| 0.66 + (i % 11) as f32 * 0.0025)
        .collect::<Vec<_>>();
    let mut input = Vec::with_capacity(seq_len * cols);
    for row in hidden.chunks_exact(n_embd) {
        input.extend(cpu_rms_norm(row, &attn_norm, 1.0e-5, true));
    }
    let q_norm = (0..head_dim)
        .map(|i| 0.70 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + (i % 17) as f32 * 0.005859375)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q6_f16_limit = usize::MAX;
    let (attn, k_bits, v_bits) = state
        .q4k_f16_qkv_prefill_attention_hd512(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV prefill attention state API")
        .expect("Q4_K F16 cache admitted hd512 prefill attention test weights");

    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o_weights,
        &gate_weights,
        &up_weights,
        &down_weights,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        q_rows,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate hd512 path");

    let mut actual = hidden;
    let hidden_input = actual.clone();
    let (actual_k_bits, actual_v_bits, completed) = state
        .q4k_f16_qkv_prefill_attention_hd512_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            true,
            true,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut actual,
            None,
            None,
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd512 dense chain state API")
        .expect("Q4_K F16 cache admitted hd512 dense chain test weights");

    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let mut device_hidden = hidden_input.clone();
    let (device_k_bits, device_v_bits, device_completed) = state
        .q4k_f16_qkv_prefill_attention_hd512_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            true,
            true,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut device_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd512 dense chain device-output state API")
        .expect("Q4_K F16 cache admitted hd512 dense chain device-output test weights");
    let device_output_id = match device_completed {
        Q4kF16DenseChainOutput::Device(id) => id,
        other => panic!("expected QKV hd512 device output, got {other:?}"),
    };
    let device_actual = state
        .download_device_tensor_f32(device_output_id)
        .expect("download Q4_K F16 QKV hd512 dense chain device output");
    assert!(state
        .release_device_tensor(device_output_id)
        .expect("release Q4_K F16 QKV hd512 dense chain device output"));

    let input_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let input_id = state
        .upload_device_tensor_f32(input_desc, &hidden_input)
        .expect("upload Q4_K F16 QKV hd512 dense chain device input");
    let mut device_input_hidden = hidden_input.clone();
    let (device_input_k_bits, device_input_v_bits, device_input_completed) = state
        .q4k_f16_qkv_prefill_attention_hd512_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &[],
            Some((input_id, input_desc)),
            &attn_norm,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            true,
            true,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut device_input_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd512 dense chain device-input state API")
        .expect("Q4_K F16 cache admitted hd512 dense chain device-input test weights");
    let device_input_output_id = match device_input_completed {
        Q4kF16DenseChainOutput::Device(id) => id,
        other => panic!("expected QKV hd512 device-input output, got {other:?}"),
    };
    let device_input_actual = state
        .download_device_tensor_f32(device_input_output_id)
        .expect("download Q4_K F16 QKV hd512 dense chain device-input output");
    assert!(state
        .release_device_tensor(device_input_output_id)
        .expect("release Q4_K F16 QKV hd512 dense chain device-input output"));
    assert!(state
        .release_device_tensor(input_id)
        .expect("release Q4_K F16 QKV hd512 dense chain device input"));

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }
    if let Some(prev) = prev_gate_f16 {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP");
    }

    assert!(matches!(completed, Q4kF16DenseChainOutput::Host));
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain K bits",
        &actual_k_bits,
        &k_bits,
        0.004,
    );
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain V bits",
        &actual_v_bits,
        &v_bits,
        0.004,
    );
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain device output K bits",
        &device_k_bits,
        &k_bits,
        0.004,
    );
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain device output V bits",
        &device_v_bits,
        &v_bits,
        0.004,
    );
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain device input K bits",
        &device_input_k_bits,
        &k_bits,
        0.004,
    );
    assert_f16_bits_close(
        "Q4_K F16 QKV hd512 dense chain device input V bits",
        &device_input_v_bits,
        &v_bits,
        0.004,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd512 dense chain",
        &actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd512 dense chain device output",
        &device_actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd512 dense chain device input",
        &device_input_actual,
        &expected,
        0.03,
        0.01,
    );
}

#[test]
fn cuda_q4k_f16_q_cached_f16kv_hd512_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");

    let q_rows = 1024usize;
    let kv_rows = 512usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 512usize;
    let n_embd = cols;
    let n_ff = 1024usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rope_theta = 10000.0f32;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 443)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 449)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 457)
        .pop()
        .unwrap();
    let o_weights = make_test_q4k_weights(1, n_embd, q_rows / 256, 461)
        .pop()
        .unwrap();
    let gate_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 463)
        .pop()
        .unwrap();
    let up_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 467)
        .pop()
        .unwrap();
    let down_weights = make_test_q6k_weights(1, n_embd, n_ff / 256, 479)
        .pop()
        .unwrap();
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.003)
        .collect::<Vec<_>>();
    let attn_norm = (0..n_embd)
        .map(|i| 0.64 + (i % 11) as f32 * 0.00275)
        .collect::<Vec<_>>();
    let mut input = Vec::with_capacity(seq_len * cols);
    for row in hidden.chunks_exact(n_embd) {
        input.extend(cpu_rms_norm(row, &attn_norm, 1.0e-5, true));
    }
    let q_norm = (0..head_dim)
        .map(|i| 0.72 + (i % 13) as f32 * 0.00625)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.83 + (i % 17) as f32 * 0.0046875)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q6_f16_limit = usize::MAX;
    let (attn, k_bits, v_bits) = state
        .q4k_f16_qkv_prefill_attention_hd512(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            None,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV prefill attention state API")
        .expect("Q4_K F16 cache admitted hd512 cached-KV source test weights");

    let mut expected = hidden.clone();
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &o_weights,
        &gate_weights,
        &up_weights,
        &down_weights,
        14,
        Some(&post_attn_norm),
        &ffn_norm,
        Some(&post_ffn_norm),
        q_rows,
        n_ff,
        n_embd,
        seq_len,
        &mut expected,
        &attn,
        1.0e-5,
        true,
        true,
        true,
    )
    .expect("CUDA dense chain separate cached-KV hd512 path");

    let mut actual = hidden;
    let hidden_input = actual.clone();
    let completed = state
        .q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
            &q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            None,
            &k_bits,
            &v_bits,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut actual,
            None,
            None,
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 Q cached hd512 dense chain state API");

    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let mut device_hidden = hidden_input.clone();
    let device_completed = state
        .q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
            &q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            None,
            &k_bits,
            &v_bits,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            0,
            1.0e-5,
            true,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut device_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 Q cached hd512 dense chain device-output state API");
    let device_output_id = match device_completed {
        Some(Q4kF16DenseChainOutput::Device(id)) => id,
        other => panic!("expected device output, got {other:?}"),
    };
    let device_actual = state
        .download_device_tensor_f32(device_output_id)
        .expect("download Q4_K F16 Q cached hd512 dense chain device output");
    assert!(state
        .release_device_tensor(device_output_id)
        .expect("release Q4_K F16 Q cached hd512 dense chain device output"));

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }

    assert!(matches!(completed, Some(Q4kF16DenseChainOutput::Host)));
    assert_close_rows_abs_rel(
        "Q4_K F16 Q cached f16KV hd512 dense chain",
        &actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 Q cached f16KV hd512 dense chain device output",
        &device_actual,
        &expected,
        0.03,
        0.01,
    );
}

#[test]
fn cuda_rope_table_cache_reuses_device_tables_by_shape_key() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let first = state
        .rope_table_ptrs(512, 4, 0, 10000.0)
        .expect("first rope table upload");
    let second = state
        .rope_table_ptrs(512, 4, 0, 10000.0)
        .expect("second rope table lookup");
    let different = state
        .rope_table_ptrs(256, 4, 0, 10000.0)
        .expect("different head_dim rope table upload");

    assert_eq!(first, second);
    assert_ne!(first, different);
}

#[test]
fn cuda_q4k_f16_qkv_postprocess_hd256_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _qkv = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "1");
    let q_rows = 512usize;
    let kv_rows = 256usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let rope_theta = 10000.0f32;
    let pos_start = 7usize;
    let q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 223)
        .pop()
        .unwrap();
    let k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 227)
        .pop()
        .unwrap();
    let v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 229)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.00325)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.70 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + (i % 17) as f32 * 0.005859375)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    let (q_raw, k_raw, v_raw) = state
        .q4k_f16_qkv_gemm_batch(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
        )
        .expect("CUDA Q4_K F16 QKV GEMM batch state API")
        .expect("Q4_K F16 cache admitted QKV test weights");
    let expected_q = cpu_qk_norm_rope_neox(
        &q_raw, &q_norm, seq_len, num_heads, head_dim, pos_start, 1.0e-5, rope_theta, None, true,
    );
    let expected_k = cpu_qk_norm_rope_neox(
        &k_raw,
        &k_norm,
        seq_len,
        num_kv_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        true,
    );
    let expected_v = cpu_v_norm_pack(&v_raw, seq_len, num_kv_heads, head_dim, 1.0e-5);
    let expected_k_bits = expected_k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let expected_v_bits = expected_v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();

    let (actual_q, actual_k_bits, actual_v_bits) = state
        .q4k_f16_qkv_postprocess_hd256(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV postprocess hd256 state API")
        .expect("Q4_K F16 cache admitted hd256 postprocess test weights");

    for (idx, (a, b)) in actual_k_bits.iter().zip(expected_k_bits.iter()).enumerate() {
        let diff = (*a as i32 - *b as i32).abs() as u32;
        assert!(
            diff <= 1,
            "Q4_K F16 QKV hd256 postprocess K bits idx {idx}: actual {a} vs expected {b} (ULP diff {diff})"
        );
    }
    for (idx, (a, b)) in actual_v_bits.iter().zip(expected_v_bits.iter()).enumerate() {
        let diff = (*a as i32 - *b as i32).abs() as u32;
        assert!(
            diff <= 1,
            "Q4_K F16 QKV hd256 postprocess V bits idx {idx}: actual {a} vs expected {b} (ULP diff {diff})"
        );
    }
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 postprocess Q",
        &actual_q,
        &expected_q,
        0.35,
        0.01,
    );
}

#[test]
fn cuda_q4k_f16_qkv_postprocess_hd256_window_dense_chain_matches_separate_path() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let prev_gate = std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP").ok();
    let prev_down = std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN").ok();
    let prev_prefill = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    let prev_gate_f16 = std::env::var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP").ok();
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", "0");
    std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", "0");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
    std::env::set_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", "1");

    let q_rows = 512usize;
    let kv_rows = 256usize;
    let cols = 512usize;
    let seq_len = 5usize;
    let blocks_per_row = cols / 256;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let n_embd = cols;
    let n_ff = 512usize;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rope_theta = 10000.0f32;
    let pos_start = 11usize;
    let mut q_weights = make_test_q4k_weights(1, q_rows, blocks_per_row, 331)
        .pop()
        .unwrap();
    let mut k_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 337)
        .pop()
        .unwrap();
    let mut v_weights = make_test_q4k_weights(1, kv_rows, blocks_per_row, 347)
        .pop()
        .unwrap();
    let mut o_weights = make_test_q4k_weights(1, n_embd, q_rows / 256, 349)
        .pop()
        .unwrap();
    let mut gate_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 353)
        .pop()
        .unwrap();
    let mut up_weights = make_test_q4k_weights(1, n_ff, n_embd / 256, 359)
        .pop()
        .unwrap();
    let mut down_weights = make_test_q6k_weights(1, n_embd, n_ff / 256, 367)
        .pop()
        .unwrap();
    let shrink_q4 = |weights: &mut [u8]| {
        for block in weights.chunks_exact_mut(144) {
            block[0..2].copy_from_slice(&half::f16::from_f32(0.000244140625).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(0.0001220703125).to_le_bytes());
        }
    };
    shrink_q4(&mut q_weights);
    shrink_q4(&mut k_weights);
    shrink_q4(&mut v_weights);
    shrink_q4(&mut o_weights);
    shrink_q4(&mut gate_weights);
    shrink_q4(&mut up_weights);
    for block in down_weights.chunks_exact_mut(210) {
        block[208..210].copy_from_slice(&half::f16::from_f32(0.000244140625).to_le_bytes());
    }
    let hidden = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let attn_norm = (0..n_embd)
        .map(|i| 0.65 + (i % 11) as f32 * 0.003)
        .collect::<Vec<_>>();
    let mut input = Vec::with_capacity(seq_len * cols);
    for row in hidden.chunks_exact(n_embd) {
        input.extend(cpu_rms_norm(row, &attn_norm, 1.0e-5, true));
    }
    let q_norm = (0..head_dim)
        .map(|i| 0.70 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + (i % 17) as f32 * 0.005859375)
        .collect::<Vec<_>>();
    let post_attn_norm = (0..n_embd)
        .map(|i| 0.7 + (i % 13) as f32 * 0.002)
        .collect::<Vec<_>>();
    let ffn_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 17) as f32 * 0.002)
        .collect::<Vec<_>>();
    let post_ffn_norm = (0..n_embd)
        .map(|i| 0.9 + (i % 19) as f32 * 0.001)
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q6_f16_limit = usize::MAX;
    let (q, k_bits, v_bits) = state
        .q4k_f16_qkv_postprocess_hd256(
            &q_weights,
            &k_weights,
            &v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &input,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV postprocess hd256 state API")
        .expect("Q4_K F16 cache admitted hd256 postprocess test weights");

    let mut expected = hidden.clone();
    state
        .attention_prefill_flash_hd256_f16kv_window_dense_chain(
            &q,
            &k_bits,
            &v_bits,
            seq_len,
            seq_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            q_rows,
            n_ff,
            n_embd,
            &mut expected,
            1.0e-5,
            true,
            true,
            true,
        )
        .expect("CUDA hd256 f16KV window attention dense chain separate path");

    let mut actual = hidden;
    let hidden_input = actual.clone();
    let (actual_k_bits, actual_v_bits, completed) = state
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut actual,
            None,
            None,
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 window dense chain state API")
        .expect("Q4_K F16 cache admitted hd256 window dense chain test weights");

    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let mut device_hidden = hidden_input.clone();
    let (device_k_bits, device_v_bits, device_completed) = state
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut device_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 window dense chain device-output state API")
        .expect("Q4_K F16 cache admitted hd256 window dense chain device-output test weights");
    let device_output_id = match device_completed {
        Q4kF16DenseChainOutput::Device(id) => id,
        other => panic!("expected QKV hd256 window device output, got {other:?}"),
    };
    let device_actual = state
        .download_device_tensor_f32(device_output_id)
        .expect("download Q4_K F16 QKV hd256 window dense chain device output");
    assert!(state
        .release_device_tensor(device_output_id)
        .expect("release Q4_K F16 QKV hd256 window dense chain device output"));

    let input_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    let input_id = state
        .upload_device_tensor_f32(input_desc, &hidden_input)
        .expect("upload Q4_K F16 QKV hd256 window dense chain device input");
    let mut device_input_hidden = hidden_input.clone();
    let (device_input_k_bits, device_input_v_bits, device_input_completed) = state
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &[],
            Some((input_id, input_desc)),
            &attn_norm,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            None,
            None,
            None,
            None,
            0,
            q_rows,
            n_ff,
            n_embd,
            &mut device_input_hidden,
            None,
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 window dense chain device-input state API")
        .expect("Q4_K F16 cache admitted hd256 window dense chain device-input test weights");
    let device_input_output_id = match device_input_completed {
        Q4kF16DenseChainOutput::Device(id) => id,
        other => panic!("expected QKV hd256 window device-input output, got {other:?}"),
    };
    let device_input_actual = state
        .download_device_tensor_f32(device_input_output_id)
        .expect("download Q4_K F16 QKV hd256 window dense chain device-input output");
    assert!(state
        .release_device_tensor(device_input_output_id)
        .expect("release Q4_K F16 QKV hd256 window dense chain device-input output"));
    assert!(state
        .release_device_tensor(input_id)
        .expect("release Q4_K F16 QKV hd256 window dense chain device input"));
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "Q4_K F16 QKV hd256 window dense chain baseline reference must stay finite"
    );

    let ple_dim = 256usize;
    let ple_gate = (0..ple_dim * n_embd)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.00007)
        .collect::<Vec<_>>();
    let ple_proj = (0..n_embd * ple_dim)
        .map(|i| ((i as f32 % 53.0) - 26.0) * 0.00006)
        .collect::<Vec<_>>();
    let ple_gate_bytes = f32_to_le_bytes(&ple_gate);
    let ple_proj_bytes = f32_to_le_bytes(&ple_proj);
    let ple_input = (0..seq_len * ple_dim)
        .map(|i| 0.6 + ((i as f32 * 0.017).sin() * 0.2))
        .collect::<Vec<_>>();
    let ple_post_norm = (0..n_embd)
        .map(|i| 0.8 + (i % 29) as f32 * 0.0015)
        .collect::<Vec<_>>();
    let layer_out_scale = [0.0625f32];
    let mut expected_fused = expected.clone();
    for token in 0..seq_len {
        let hidden_off = token * n_embd;
        let ple_off = token * ple_dim;
        let mut gate_out = cpu_f32_gemv_rows(
            &ple_gate,
            ple_dim,
            n_embd,
            &expected_fused[hidden_off..hidden_off + n_embd],
        );
        for (value, &scale) in gate_out
            .iter_mut()
            .zip(ple_input[ple_off..ple_off + ple_dim].iter())
        {
            let x = *value;
            let x3 = x * x * x;
            let c = 0.7978845608028654f32;
            let gelu = 0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh());
            *value = gelu * scale;
        }
        let projected = cpu_f32_gemv_rows(&ple_proj, n_embd, ple_dim, &gate_out);
        let normed = cpu_rms_norm(&projected, &ple_post_norm, 1.0e-5, false);
        for i in 0..n_embd {
            expected_fused[hidden_off + i] += normed[i];
        }
    }
    for value in &mut expected_fused {
        *value *= layer_out_scale[0];
    }
    assert!(
        expected_fused.iter().all(|value| value.is_finite()),
        "Q4_K F16 QKV hd256 F32 PLE reference fixture must stay finite"
    );
    let mut host_fused_hidden = hidden_input.clone();
    let (host_fused_k_bits, host_fused_v_bits, host_fused_completed) = state
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            Some(&ple_gate_bytes),
            Some(&ple_proj_bytes),
            Some(&ple_post_norm),
            Some(&ple_input),
            ple_dim,
            q_rows,
            n_ff,
            n_embd,
            &mut host_fused_hidden,
            Some(&layer_out_scale),
            None,
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 window dense chain F32 PLE host output")
        .expect("Q4_K F16 cache admitted hd256 window dense chain F32 PLE host output");
    let mut device_fused_hidden = hidden_input.clone();
    let (device_fused_k_bits, device_fused_v_bits, device_fused_completed) = state
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            &q_weights,
            &k_weights,
            &v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            &hidden_input,
            None,
            &attn_norm,
            &q_norm,
            &k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            1.0e-5,
            true,
            true,
            true,
            window,
            &o_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
            14,
            Some(&post_attn_norm),
            &ffn_norm,
            Some(&post_ffn_norm),
            Some(&ple_gate_bytes),
            Some(&ple_proj_bytes),
            Some(&ple_post_norm),
            Some(&ple_input),
            ple_dim,
            q_rows,
            n_ff,
            n_embd,
            &mut device_fused_hidden,
            Some(&layer_out_scale),
            Some(output_desc),
            true,
            true,
            true,
            true,
        )
        .expect("CUDA Q4_K F16 QKV hd256 window dense chain F32 PLE device output")
        .expect("Q4_K F16 cache admitted hd256 window dense chain F32 PLE device output");
    let device_fused_output_id = match device_fused_completed {
        Q4kF16DenseChainOutput::Device(id) => id,
        other => panic!("expected QKV hd256 window F32 PLE device output, got {other:?}"),
    };
    let device_fused_actual = state
        .download_device_tensor_f32(device_fused_output_id)
        .expect("download Q4_K F16 QKV hd256 window F32 PLE device output");
    assert!(state
        .release_device_tensor(device_fused_output_id)
        .expect("release Q4_K F16 QKV hd256 window F32 PLE device output"));

    if let Some(prev) = prev_gate {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_GATE_UP");
    }
    if let Some(prev) = prev_down {
        std::env::set_var("RNB_CUDA_DENSE_Q8DOT_DOWN", prev);
    } else {
        std::env::remove_var("RNB_CUDA_DENSE_Q8DOT_DOWN");
    }
    if let Some(prev) = prev_prefill {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
    }
    if let Some(prev) = prev_gate_f16 {
        std::env::set_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP");
    }

    assert!(matches!(completed, Q4kF16DenseChainOutput::Host));
    assert_eq!(actual_k_bits, k_bits);
    assert_eq!(actual_v_bits, v_bits);
    assert_eq!(device_k_bits, k_bits);
    assert_eq!(device_v_bits, v_bits);
    assert_eq!(device_input_k_bits, k_bits);
    assert_eq!(device_input_v_bits, v_bits);
    assert!(matches!(host_fused_completed, Q4kF16DenseChainOutput::Host));
    assert_eq!(host_fused_k_bits, device_fused_k_bits);
    assert_eq!(host_fused_v_bits, device_fused_v_bits);
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 window dense chain F32 PLE host output",
        &host_fused_hidden,
        &expected_fused,
        1.0e-3,
        5.0e-3,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 window dense chain F32 PLE device output",
        &device_fused_actual,
        &expected_fused,
        1.0e-3,
        5.0e-3,
    );
    assert_eq!(
        host_fused_hidden
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        device_fused_actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "F32 PLE host-output and device-output branches must produce identical hidden bits"
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 window dense chain",
        &actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 window dense chain device output",
        &device_actual,
        &expected,
        0.03,
        0.01,
    );
    assert_close_rows_abs_rel(
        "Q4_K F16 QKV hd256 window dense chain device input",
        &device_input_actual,
        &expected,
        0.03,
        0.01,
    );
}

#[test]
fn cuda_q4k_gemv_batch_f32_cache_path_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _f32 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "1");

    let rows = 1024usize;
    let cols = 1024usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q4k_weights(1, rows, blocks_per_row, 211)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q4k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4_f32_limit = usize::MAX;
    let actual = state
        .q4k_gemv_batch(&weights, rows, blocks_per_row, seq_len, &input)
        .expect("CUDA Q4_K F32 cache batch path");
    assert_close_rows("Q4_K F32 cache GEMV batch", &actual, &expected, 0.04);
}

#[test]
fn cuda_q6k_f32_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let rows = 512usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 157)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual =
        q6k_f32_gemm_batch_for_test(&weights, rows, cols, &input).expect("CUDA Q6_K F32 GEMM");
    assert_close_rows("Q6_K F32 GEMM batch", &actual, &expected, 0.08);
}

#[test]
fn cuda_f32_to_f16_kernel_matches_half_conversion() {
    let _guard = runtime_test_lock();
    let input = (0..257usize)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.03125)
        .collect::<Vec<_>>();
    let expected = input
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("input buffer");
    let output_dev = state
        .compute_output_ptr(std::mem::size_of_val(expected.as_slice()))
        .expect("output buffer");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("input h2d");
    }
    state
        .launch_f32_to_f16(input_dev, output_dev, input.len())
        .expect("f32 to f16 kernel");
    let mut actual = vec![0u16; input.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .expect("output dtoh");
    }
    state.stream_synchronize().expect("sync");

    assert_eq!(actual, expected);
}

#[test]
fn cuda_f16_to_f32_kernel_matches_half_conversion() {
    let _guard = runtime_test_lock();
    let input = (0..257usize)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.03125).to_bits())
        .collect::<Vec<_>>();
    let expected = input
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();

    let mut state = CudaState::open().expect("open CUDA state");
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("input buffer");
    let output_dev = state
        .compute_output_ptr(std::mem::size_of_val(expected.as_slice()))
        .expect("output buffer");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("input h2d");
    }
    state
        .launch_f16_to_f32(input_dev, output_dev, input.len())
        .expect("f16 to f32 kernel");
    let mut actual = vec![0.0f32; input.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .expect("output dtoh");
    }
    state.stream_synchronize().expect("sync");

    assert_eq!(actual, expected);
}

#[test]
fn cuda_q6k_f16_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let rows = 512usize;
    let cols = 512usize;
    let seq_len = 3usize;
    let blocks_per_row = cols / 256;
    let weights = make_test_q6k_weights(1, rows, blocks_per_row, 181)
        .pop()
        .unwrap();
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.004)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q6k_gemv_rows(
            &weights,
            rows,
            blocks_per_row,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q6_f16_limit = usize::MAX;
    let weights_dev = state
        .resident_q6k_f16_ptr(&weights, rows, blocks_per_row)
        .expect("Q6_K F16 resident cache")
        .expect("Q6_K F16 resident cache enabled");
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("input buffer");
    let input_f16_dev = state
        .compute_gate_ptrs_ptr(seq_len * cols * std::mem::size_of::<u16>())
        .expect("input f16 buffer");
    let output_len = seq_len * rows;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let output_dev = state
        .compute_output_ptr(output_bytes)
        .expect("output buffer");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("input h2d");
    }
    state
        .launch_f32_to_f16(input_dev, input_f16_dev, input.len())
        .expect("input f16 convert");
    state
        .hgemm_to_f32_device(weights_dev, rows, cols, input_f16_dev, seq_len, output_dev)
        .expect("CUDA Q6_K F16 GEMM");
    let mut actual = vec![0.0f32; output_len];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )
            .expect("output dtoh");
    }
    state.stream_synchronize().expect("sync");

    assert_close_rows("Q6_K F16 GEMM batch", &actual, &expected, 0.12);
}
#[test]
fn cuda_q6k_gemv_argmax_batched_single_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 17usize;
    let cols = 512usize;
    let weights = make_test_q6k_weights(1, rows, cols / 256, 97)
        .pop()
        .unwrap();
    let input = (0..cols)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.01953125)
        .collect::<Vec<_>>();
    let expected_rows = cpu_q6k_gemv_rows(&weights, rows, cols / 256, &input);
    let mut expected_idx = 0usize;
    let mut expected_value = f32::NEG_INFINITY;
    for (idx, &value) in expected_rows.iter().enumerate() {
        if value > expected_value || (value == expected_value && idx < expected_idx) {
            expected_value = value;
            expected_idx = idx;
        }
    }

    let (actual_idx, actual_value) =
        q6k_gemv_argmax(&weights, rows, cols, &input).expect("CUDA Q6_K argmax");
    assert_eq!(actual_idx as usize, expected_idx);
    assert!(
        (actual_value - expected_value).abs() < 0.3,
        "Q6_K argmax value mismatch: actual={actual_value} expected={expected_value}"
    );
}

#[test]
fn cuda_q8_0_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 7usize;
    let cols = 96usize;
    let weights = make_test_q8_0_weights(rows, cols, 101);
    let input = (0..cols)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let expected = cpu_q8_0_rows(&weights, rows, cols, &input);
    let actual = q8_0_gemv(&weights, rows, cols, &input).expect("CUDA Q8_0 GEMV");
    assert_close_rows("Q8_0", &actual, &expected, 0.01);
}

#[test]
fn cuda_q8_0_gemv_argmax_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 17usize;
    let cols = 96usize;
    let weights = make_test_q8_0_weights(rows, cols, 109);
    let input = (0..cols)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.03125)
        .collect::<Vec<_>>();
    let expected_rows = cpu_q8_0_rows(&weights, rows, cols, &input);
    let mut expected_idx = 0usize;
    let mut expected_value = f32::NEG_INFINITY;
    for (idx, &value) in expected_rows.iter().enumerate() {
        if value > expected_value || (value == expected_value && idx < expected_idx) {
            expected_value = value;
            expected_idx = idx;
        }
    }
    let (actual_idx, actual_value) =
        q8_0_gemv_argmax(&weights, rows, cols, &input).expect("CUDA Q8_0 argmax");
    assert_eq!(actual_idx as usize, expected_idx);
    assert!(
        (actual_value - expected_value).abs() < 0.01,
        "Q8_0 argmax value mismatch: actual={actual_value} expected={expected_value}"
    );
}

#[test]
fn cuda_q8_0_gemv_q8dot_argmax_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 17usize;
    let cols = 128usize;
    let weights = make_test_q8_0_weights(rows, cols, 151);
    let input = (0..cols)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.02734375)
        .collect::<Vec<_>>();
    let expected_rows = cpu_q8_0_q8dot_rows(&weights, rows, cols, &input);
    let mut expected_idx = 0usize;
    let mut expected_value = f32::NEG_INFINITY;
    for (idx, &value) in expected_rows.iter().enumerate() {
        if value > expected_value || (value == expected_value && idx < expected_idx) {
            expected_value = value;
            expected_idx = idx;
        }
    }
    let (actual_idx, actual_value) =
        q8_0_gemv_argmax_q8dot(&weights, rows, cols, &input).expect("CUDA Q8_0 Q8dot argmax");
    assert_eq!(actual_idx as usize, expected_idx);
    assert!(
        (actual_value - expected_value).abs() < 0.01,
        "Q8_0 Q8dot argmax value mismatch: actual={actual_value} expected={expected_value}"
    );
}

#[test]
fn cuda_q8_0_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 7usize;
    let cols = 96usize;
    let seq_len = 3usize;
    let weights = make_test_q8_0_weights(rows, cols, 137);
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q8_0_rows(
            &weights,
            rows,
            cols,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let actual = q8_0_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q8_0 batch GEMV");
    assert_close_rows("Q8_0 batch", &actual, &expected, 0.01);
}
#[test]
fn cuda_q8_0_mmq_tile32_matches_cpu_reference_with_tails() {
    let _guard = runtime_test_lock();
    let _mmq = EnvVarGuard::set("RNB_CUDA_Q8_0_MMQ_TILE32", "1");
    let rows = 513usize;
    let cols = 128usize;
    let seq_len = 37usize;
    let weights = make_test_q8_0_weights(rows, cols, 173);
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.00390625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q8_0_rows(
            &weights,
            rows,
            cols,
            &input[token * cols..(token + 1) * cols],
        ));
    }
    let mut first_actual: Option<Vec<f32>> = None;
    for run in 0..16 {
        let actual = q8_0_gemv_batch(&weights, rows, cols, &input).expect("CUDA Q8_0 tiled MMQ");
        assert_close_rows(
            &format!("Q8_0 tiled MMQ run {run}"),
            &actual,
            &expected,
            0.08,
        );
        if let Some(first) = first_actual.as_ref() {
            let mismatch = actual
                .iter()
                .zip(first)
                .position(|(actual, first)| actual.to_bits() != first.to_bits());
            assert!(
                mismatch.is_none(),
                "Q8_0 tiled MMQ run {run} is not bitwise deterministic at {mismatch:?}"
            );
        } else {
            first_actual = Some(actual);
        }
    }
    let device_actual = q8_0_gemv_batch_device_input_for_test(&weights, rows, cols, &input)
        .expect("CUDA Q8_0 device-input tiled MMQ");
    assert_close_rows(
        "Q8_0 device-input tiled MMQ",
        &device_actual,
        &expected,
        0.08,
    );
}

#[test]
fn cuda_q8_0_head_gemv_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let head_count = 3usize;
    let rows_per_head = 4usize;
    let cols = 96usize;
    let token_count = 3usize;
    let rows = head_count * rows_per_head;
    let weights = make_test_q8_0_weights(rows, cols, 181);
    let input = (0..token_count * head_count * cols)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();
    let row_bytes = (cols / 32) * 34;
    let head_bytes = rows_per_head * row_bytes;
    let mut expected = Vec::with_capacity(token_count * rows);
    for token in 0..token_count {
        for head in 0..head_count {
            let input_offset = (token * head_count + head) * cols;
            let weight_offset = head * head_bytes;
            expected.extend(cpu_q8_0_rows(
                &weights[weight_offset..weight_offset + head_bytes],
                rows_per_head,
                cols,
                &input[input_offset..input_offset + cols],
            ));
        }
    }
    let actual = q8_0_head_gemv_batch(
        &weights,
        head_count,
        rows_per_head,
        cols,
        token_count,
        &input,
    )
    .expect("CUDA Q8_0 head batch GEMV");
    assert_close_rows("Q8_0 head batch", &actual, &expected, 0.01);
}

#[test]
fn cuda_q8_0_gemv_batch_device_input_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 9usize;
    let cols = 128usize;
    let seq_len = 4usize;
    let weights = make_test_q8_0_weights(rows, cols, 173);
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q8_0_rows(
            &weights,
            rows,
            cols,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual = q8_0_gemv_batch_device_input_for_test(&weights, rows, cols, &input)
        .expect("CUDA Q8_0 device-input batch GEMV");

    assert_close_rows("Q8_0 device-input batch", &actual, &expected, 0.01);
}

#[test]
fn cuda_q8_0_gemv_batch_token2_irregular_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 17usize;
    let cols = 320usize;
    let seq_len = 2usize;
    let weights = make_test_q8_0_weights(rows, cols, 181);
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.01171875)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q8_0_rows(
            &weights,
            rows,
            cols,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual = q8_0_gemv_batch_device_input_for_test(&weights, rows, cols, &input)
        .expect("CUDA Q8_0 token2 batch GEMV");

    assert_close_rows("Q8_0 token2 batch", &actual, &expected, 0.01);
}

#[test]
fn cuda_q8_0_f32_gemm_batch_cached_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 8usize;
    let cols = 128usize;
    let seq_len = 5usize;
    let weights = make_test_q8_0_weights(rows, cols, 211);
    let input = (0..seq_len * cols)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.0125)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        expected.extend(cpu_q8_0_rows(
            &weights,
            rows,
            cols,
            &input[token * cols..(token + 1) * cols],
        ));
    }

    let actual = q8_0_f32_gemm_batch_cached_for_test(&weights, rows, cols, &input)
        .expect("CUDA Q8_0 F32 GEMM batch");

    assert_close_rows("Q8_0 F32 GEMM batch", &actual, &expected, 0.02);
}

#[test]
fn cuda_nemotron_q5_sparse_relu_sqr_by_token_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    unsafe {
        std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", "1");
        std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS", "1");
        std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS", "1");
    }
    let n_embd = 96usize;
    let n_ff = 64usize;
    let token_count = 2usize;
    let experts = 3usize;
    let up_all = (0..experts)
        .map(|expert| make_test_q5_basic_weights(n_ff, n_embd, 22, false, 17 + expert * 31))
        .collect::<Vec<_>>();
    let down_all = (0..experts)
        .map(|expert| make_test_q5_basic_weights(n_embd, n_ff, 24, true, 41 + expert * 29))
        .collect::<Vec<_>>();
    let up_refs = vec![
        up_all[0].as_slice(),
        up_all[2].as_slice(),
        up_all[1].as_slice(),
    ];
    let down_refs = vec![
        down_all[0].as_slice(),
        down_all[2].as_slice(),
        down_all[1].as_slice(),
    ];
    let route = vec![0.625f32, 0.375f32, 1.0f32];
    let expert_ids = vec![0u32, 2u32, 1u32];
    let token_ids = vec![0u32, 0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; token_count * n_embd];
    for slot in 0..up_refs.len() {
        let token = token_ids[slot] as usize;
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let mut mid = cpu_q5_basic_rows(up_refs[slot], n_ff, n_embd, 22, false, token_input);
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let down = cpu_q5_basic_rows(down_refs[slot], n_embd, n_ff, 24, true, &mid);
        for row in 0..n_embd {
            expected[token * n_embd + row] += down[row] * route[slot];
        }
    }

    let actual = nemotron_q5_sparse_relu_sqr_by_token(
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("CUDA Nemotron sparse Q5 batch");
    assert_close_rows("Nemotron sparse Q5", &actual, &expected, 0.2);

    let up_concat = up_all.concat();
    let down_concat = down_all.concat();
    let actual_full = nemotron_q5_sparse_relu_sqr_full_layer_by_token(
        &up_concat,
        &down_concat,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        experts,
        n_ff,
        n_embd,
        &input,
    )
    .expect("CUDA Nemotron full-layer sparse Q5 batch");
    assert_close_rows(
        "Nemotron full-layer sparse Q5",
        &actual_full,
        &expected,
        0.2,
    );

    let down_q8_all = (0..experts)
        .map(|expert| make_test_q8_0_weights(n_embd, n_ff, 211 + expert * 19))
        .collect::<Vec<_>>();
    let down_q8_refs = vec![
        down_q8_all[0].as_slice(),
        down_q8_all[2].as_slice(),
        down_q8_all[1].as_slice(),
    ];
    let mut expected_q8_down = vec![0.0f32; token_count * n_embd];
    for slot in 0..up_refs.len() {
        let token = token_ids[slot] as usize;
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let mut mid = cpu_q5_basic_rows(up_refs[slot], n_ff, n_embd, 22, false, token_input);
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let down = cpu_q8_0_rows(down_q8_refs[slot], n_embd, n_ff, &mid);
        for row in 0..n_embd {
            expected_q8_down[token * n_embd + row] += down[row] * route[slot];
        }
    }
    let actual_q8_down = nemotron_q5_q8_sparse_relu_sqr_by_token(
        &up_refs,
        &down_q8_refs,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
    )
    .expect("CUDA Nemotron sparse Q5/Q8 batch");
    assert_close_rows(
        "Nemotron sparse Q5/Q8",
        &actual_q8_down,
        &expected_q8_down,
        0.2,
    );

    let shared_up = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 113);
    let shared_down = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 157);
    let decode_up_refs = vec![up_all[0].as_slice(), up_all[2].as_slice()];
    let decode_down_refs = vec![down_all[0].as_slice(), down_all[2].as_slice()];
    let decode_route = vec![0.625f32, 0.375f32];
    let decode_input = &input[..n_embd];
    let mut decode_expected = vec![0.0f32; n_embd];
    let mut shared_mid = cpu_q5_basic_rows(&shared_up, n_ff, n_embd, 22, false, decode_input);
    for value in &mut shared_mid {
        *value = if *value > 0.0 { *value * *value } else { 0.0 };
    }
    let shared_out = cpu_q5_basic_rows(&shared_down, n_embd, n_ff, 24, true, &shared_mid);
    decode_expected.copy_from_slice(&shared_out);
    for slot in 0..decode_up_refs.len() {
        let mut mid =
            cpu_q5_basic_rows(decode_up_refs[slot], n_ff, n_embd, 22, false, decode_input);
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let down = cpu_q5_basic_rows(decode_down_refs[slot], n_embd, n_ff, 24, true, &mid);
        for row in 0..n_embd {
            decode_expected[row] += down[row] * decode_route[slot];
        }
    }
    let decode_actual = nemotron_q5_decode_moe_shared_sparse(
        &shared_up,
        &shared_down,
        &decode_up_refs,
        &decode_down_refs,
        &decode_route,
        n_ff,
        n_embd,
        decode_input,
    )
    .expect("CUDA Nemotron decode Q5 shared+sparse");
    assert_close_rows(
        "Nemotron decode Q5 shared+sparse",
        &decode_actual,
        &decode_expected,
        0.2,
    );

    let shared_ff = 96usize;
    let shared_up_q8 = make_test_q8_0_weights(shared_ff, n_embd, 191);
    let shared_down_q8 = make_test_q8_0_weights(n_embd, shared_ff, 223);
    let mut q8_decode_expected = vec![0.0f32; n_embd];
    let mut shared_mid = cpu_q8_0_rows(&shared_up_q8, shared_ff, n_embd, decode_input);
    for value in &mut shared_mid {
        *value = if *value > 0.0 { *value * *value } else { 0.0 };
    }
    let shared_out = cpu_q8_0_rows(&shared_down_q8, n_embd, shared_ff, &shared_mid);
    q8_decode_expected.copy_from_slice(&shared_out);
    for slot in 0..decode_up_refs.len() {
        let mut mid =
            cpu_q5_basic_rows(decode_up_refs[slot], n_ff, n_embd, 22, false, decode_input);
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let down = cpu_q5_basic_rows(decode_down_refs[slot], n_embd, n_ff, 24, true, &mid);
        for row in 0..n_embd {
            q8_decode_expected[row] += down[row] * decode_route[slot];
        }
    }
    let q8_decode_actual = nemotron_q8_shared_q5_sparse_decode_moe(
        &shared_up_q8,
        &shared_down_q8,
        &decode_up_refs,
        &decode_down_refs,
        &decode_route,
        shared_ff,
        n_ff,
        n_embd,
        decode_input,
    )
    .expect("CUDA Nemotron decode Q8 shared + Q5 sparse");
    assert_close_rows(
        "Nemotron decode Q8 shared + Q5 sparse",
        &q8_decode_actual,
        &q8_decode_expected,
        0.2,
    );
    unsafe {
        std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
        std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
        std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
    }
}

#[test]
fn cuda_nemotron_q8_shared_prefill_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _enabled = EnvVarGuard::set("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_FUSED", "1");
    let seq_len = 3usize;
    let n_embd = 64usize;
    let shared_ff = 96usize;
    let shared_up = make_test_q8_0_weights(shared_ff, n_embd, 419);
    let shared_down = make_test_q8_0_weights(n_embd, shared_ff, 467);
    let input = (0..seq_len * n_embd)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.03125)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(seq_len * n_embd);
    for token in 0..seq_len {
        let mut mid = cpu_q8_0_rows(
            &shared_up,
            shared_ff,
            n_embd,
            &input[token * n_embd..(token + 1) * n_embd],
        );
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        expected.extend(cpu_q8_0_rows(&shared_down, n_embd, shared_ff, &mid));
    }

    let actual =
        nemotron_q8_shared_prefill(&shared_up, &shared_down, shared_ff, n_embd, seq_len, &input)
            .expect("CUDA Nemotron Q8 shared prefill")
            .expect("Q8 shared prefill enabled");

    assert_close_rows("Nemotron Q8 shared prefill", &actual, &expected, 0.2);
}

#[test]
fn cuda_nemotron_q8_shared_sparse_prefill_moe_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _enabled = EnvVarGuard::set("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", "1");
    let token_count = 3usize;
    let n_embd = 64usize;
    let shared_ff = 96usize;
    let n_ff = 64usize;
    let shared_up = make_test_q8_0_weights(shared_ff, n_embd, 521);
    let shared_down = make_test_q8_0_weights(n_embd, shared_ff, 557);
    let up_a = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 601);
    let up_b = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 631);
    let down_a = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 661);
    let down_b = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 691);
    let up_refs = vec![up_a.as_slice(), up_b.as_slice(), up_a.as_slice()];
    let down_refs = vec![down_a.as_slice(), down_b.as_slice(), down_a.as_slice()];
    let route = vec![0.75f32, 0.5f32, 0.25f32];
    let token_ids = vec![0u32, 1u32, 2u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.02)
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.015)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let mut shared_mid = cpu_q8_0_rows(&shared_up, shared_ff, n_embd, token_input);
        for value in &mut shared_mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let mut token_out = cpu_q8_0_rows(&shared_down, n_embd, shared_ff, &shared_mid);
        for (slot, &token_id) in token_ids.iter().enumerate() {
            if token_id as usize != token {
                continue;
            }
            let mut mid = cpu_q5_basic_rows(up_refs[slot], n_ff, n_embd, 22, false, token_input);
            for value in &mut mid {
                *value = if *value > 0.0 { *value * *value } else { 0.0 };
            }
            let down = cpu_q5_basic_rows(down_refs[slot], n_embd, n_ff, 24, true, &mid);
            for row in 0..n_embd {
                token_out[row] += down[row] * route[slot];
            }
        }
        for row in 0..n_embd {
            token_out[row] += residual[token * n_embd + row];
        }
        expected.extend(token_out);
    }

    let actual = nemotron_q8_shared_q5_sparse_prefill_moe(
        &shared_up,
        &shared_down,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        &input,
        &residual,
    )
    .expect("CUDA Nemotron Q8 shared + sparse prefill MoE")
    .expect("Q8 shared + sparse fused prefill enabled");

    assert_close_rows(
        "Nemotron Q8 shared + sparse prefill MoE",
        &actual,
        &expected,
        0.3,
    );

    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Normalized,
    );
    let residual_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Residual,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Nemotron input");
    let residual_id =
        upload_device_tensor_f32(residual_desc, &residual).expect("upload Nemotron residual");
    let output_id = nemotron_q8_shared_q5_sparse_prefill_moe_device(
        &shared_up,
        &shared_down,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
    )
    .expect("CUDA Nemotron device Q8 shared + sparse prefill MoE")
    .expect("Q8 shared + sparse fused device prefill enabled");
    let device_actual =
        download_device_tensor_f32(output_id).expect("download Nemotron device MoE");
    assert_close_rows(
        "Nemotron device Q8 shared + sparse prefill MoE",
        &device_actual,
        &expected,
        0.3,
    );
    assert!(release_device_tensor(input_id).expect("release Nemotron input tensor"));
    assert!(release_device_tensor(residual_id).expect("release Nemotron residual tensor"));
    assert!(release_device_tensor(output_id).expect("release Nemotron output tensor"));
    let err = download_device_tensor_f32(output_id)
        .expect_err("released Nemotron output tensor must be missing");
    assert!(err.contains("missing CUDA device tensor id"));

    let mamba_residual_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MambaOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Nemotron input");
    let residual_id = upload_device_tensor_f32(mamba_residual_desc, &residual)
        .expect("upload Nemotron Mamba residual");
    let output_id = nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
        &shared_up,
        &shared_down,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
        mamba_residual_desc,
    )
    .expect("CUDA Nemotron device Q8 shared + sparse prefill MoE with Mamba residual")
    .expect("Q8 shared + sparse fused device prefill enabled");
    let device_actual = download_device_tensor_f32(output_id)
        .expect("download Nemotron device MoE with Mamba residual");
    assert_close_rows(
        "Nemotron device Q8 shared + sparse prefill MoE with Mamba residual",
        &device_actual,
        &expected,
        0.3,
    );
    assert!(release_device_tensor(input_id).expect("release Nemotron input tensor"));
    assert!(release_device_tensor(residual_id).expect("release Nemotron Mamba residual tensor"));
    assert!(release_device_tensor(output_id).expect("release Nemotron output tensor"));
}

#[test]
fn cuda_nemotron_q8_shared_sparse_prefill_moe_device_route_pack_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _enabled = EnvVarGuard::set("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", "1");
    let token_count = 3usize;
    let n_embd = 64usize;
    let shared_ff = 96usize;
    let n_expert = 2usize;
    let n_ff = 64usize;
    let shared_up = make_test_q8_0_weights(shared_ff, n_embd, 1521);
    let shared_down = make_test_q8_0_weights(n_embd, shared_ff, 1557);
    let up_a = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 1601);
    let up_b = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 1631);
    let down_a = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 1661);
    let down_b = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 1691);
    let logits = [4.0_f32, -4.0, -4.0, 4.0, 3.0, -3.0];
    let logits_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_expert,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::RouterLogits,
    );
    let logits_id = upload_device_tensor_f32(logits_desc, &logits).expect("upload route logits");
    let route_slots = token_count;
    let route_bytes = route_slots
        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
    let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
    let shared_mid_bytes = token_count * shared_ff * std::mem::size_of::<f32>();
    let sparse_mid_bytes = route_slots.max(1) * n_ff * std::mem::size_of::<f32>();
    begin_nemotron_prefill_workspace(NemotronPrefillWorkspaceConfig {
        hidden_bytes: output_bytes,
        normalized_bytes: 0,
        router_logits_bytes: 0,
        route_bytes,
        moe_shared_mid_bytes: shared_mid_bytes,
        moe_sparse_mid_bytes: sparse_mid_bytes,
        required_workspace_bytes: 4096,
        enabled: true,
    })
    .expect("begin workspace for route-pack output");
    let route_pack = nemotron_device_route_pack_from_logits(
        logits_id,
        logits_desc,
        None,
        token_count,
        n_expert,
        1,
        1.0,
    )
    .expect("route pack");
    let expert_ids =
        nemotron_device_route_pack_expert_ids(&route_pack).expect("download expert ids");
    assert_eq!(expert_ids, vec![0, 1, 0]);
    let up_refs = expert_ids
        .iter()
        .map(|&expert| {
            if expert == 0 {
                up_a.as_slice()
            } else {
                up_b.as_slice()
            }
        })
        .collect::<Vec<_>>();
    let down_refs = expert_ids
        .iter()
        .map(|&expert| {
            if expert == 0 {
                down_a.as_slice()
            } else {
                down_b.as_slice()
            }
        })
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.02)
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.015)
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let mut shared_mid = cpu_q8_0_rows(&shared_up, shared_ff, n_embd, token_input);
        for value in &mut shared_mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let mut token_out = cpu_q8_0_rows(&shared_down, n_embd, shared_ff, &shared_mid);
        let mut mid = cpu_q5_basic_rows(up_refs[token], n_ff, n_embd, 22, false, token_input);
        for value in &mut mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let down = cpu_q5_basic_rows(down_refs[token], n_embd, n_ff, 24, true, &mid);
        for row in 0..n_embd {
            token_out[row] += down[row] + residual[token * n_embd + row];
        }
        expected.extend(token_out);
    }

    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Normalized,
    );
    let residual_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MambaOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Nemotron input");
    let residual_id =
        upload_device_tensor_f32(residual_desc, &residual).expect("upload Nemotron residual");
    let output_id = nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
        &shared_up,
        &shared_down,
        &up_refs,
        &down_refs,
        &route_pack,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
        residual_desc,
    )
    .expect("CUDA Nemotron device route-pack Q8 shared + sparse prefill MoE")
    .expect("Q8 shared + sparse fused device route-pack prefill enabled");
    let summary = lock_default_cuda_compute_for_test()
        .as_deref()
        .and_then(|state| state.as_ref())
        .expect("cuda state")
        .nemotron_prefill_workspace_summary()
        .expect("workspace summary after MoE output");
    assert_eq!(summary.live_leases, 1);
    assert_eq!(summary.owned_alloc_count, 0);
    assert!(summary.hit_bytes >= output_bytes + route_bytes + shared_mid_bytes + sparse_mid_bytes);
    let actual =
        download_device_tensor_f32(output_id).expect("download Nemotron device route-pack MoE");
    assert_close_rows(
        "Nemotron device route-pack Q8 shared + sparse prefill MoE",
        &actual,
        &expected,
        0.3,
    );

    assert!(release_device_tensor(logits_id).expect("release route logits"));
    release_nemotron_device_route_pack(route_pack).expect("release route pack");
    assert!(release_device_tensor(input_id).expect("release Nemotron input tensor"));
    assert!(release_device_tensor(residual_id).expect("release Nemotron residual tensor"));
    assert!(release_device_tensor(output_id).expect("release Nemotron output tensor"));
    let summary = end_nemotron_prefill_workspace().expect("end workspace for route-pack output");
    assert_eq!(summary.live_leases, 0);
}

#[test]
fn cuda_nemotron_q8_shared_sparse_prefill_moe_rejects_residual_desc_mismatch() {
    let _guard = runtime_test_lock();
    let _enabled = EnvVarGuard::set("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", "1");
    let token_count = 3usize;
    let n_embd = 64usize;
    let shared_ff = 96usize;
    let n_ff = 64usize;
    let shared_up = make_test_q8_0_weights(shared_ff, n_embd, 521);
    let shared_down = make_test_q8_0_weights(n_embd, shared_ff, 557);
    let up_a = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 601);
    let up_b = make_test_q5_basic_weights(n_ff, n_embd, 22, false, 631);
    let down_a = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 661);
    let down_b = make_test_q5_basic_weights(n_embd, n_ff, 24, true, 691);
    let up_refs = vec![up_a.as_slice(), up_b.as_slice(), up_a.as_slice()];
    let down_refs = vec![down_a.as_slice(), down_b.as_slice(), down_a.as_slice()];
    let route = vec![0.75f32, 0.5f32, 0.25f32];
    let token_ids = vec![0u32, 1u32, 2u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.02)
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.015)
        .collect::<Vec<_>>();
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Normalized,
    );
    let mamba_residual_desc = rnb_backend_api::DeviceTensorDesc::new(
        token_count,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MambaOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Nemotron input");
    let residual_id = upload_device_tensor_f32(mamba_residual_desc, &residual)
        .expect("upload Nemotron Mamba residual");
    let assert_residual_desc_rejected = |residual_desc: rnb_backend_api::DeviceTensorDesc,
                                         expected: &str| {
        let err = nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
            &shared_up,
            &shared_down,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_id,
            residual_id,
            residual_desc,
        )
        .expect_err("mismatched residual desc should fail");
        assert!(
            err.contains(expected),
            "expected error containing {expected:?}, got {err:?}"
        );
    };

    assert_residual_desc_rejected(
        rnb_backend_api::DeviceTensorDesc::new(
            token_count - 1,
            n_embd,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MambaOutput,
        ),
        "residual shape mismatch",
    );
    assert_residual_desc_rejected(
        rnb_backend_api::DeviceTensorDesc::new(
            token_count,
            n_embd - 1,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MambaOutput,
        ),
        "residual shape mismatch",
    );
    assert_residual_desc_rejected(
        rnb_backend_api::DeviceTensorDesc::new(
            token_count,
            n_embd,
            rnb_backend_api::ScalarType::F16,
            rnb_backend_api::DeviceTensorRole::MambaOutput,
        ),
        "residual dtype mismatch",
    );
    assert!(release_device_tensor(input_id).expect("release Nemotron input tensor"));
    assert!(release_device_tensor(residual_id).expect("release Nemotron Mamba residual tensor"));
}

#[test]
fn cuda_nemotron_q8_shared_sparse_cached_prefill_moe_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _enabled = EnvVarGuard::set("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", "1");
    let token_count = 3usize;
    let n_embd = 64usize;
    let shared_ff = 96usize;
    let n_ff = 64usize;
    let n_expert = 2usize;
    let shared_up = make_test_q8_0_weights(shared_ff, n_embd, 727);
    let shared_down = make_test_q8_0_weights(n_embd, shared_ff, 757);
    let up_all = (0..n_expert)
        .map(|expert| make_test_q5_basic_weights(n_ff, n_embd, 22, false, 787 + expert * 31))
        .collect::<Vec<_>>()
        .concat();
    let down_all = (0..n_expert)
        .map(|expert| make_test_q5_basic_weights(n_embd, n_ff, 24, true, 853 + expert * 29))
        .collect::<Vec<_>>()
        .concat();
    let expert_ids = vec![0u32, 1u32, 0u32];
    let route = vec![0.625f32, 0.5f32, 0.375f32];
    let token_ids = vec![0u32, 1u32, 2u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.015625)
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.0125)
        .collect::<Vec<_>>();

    let up_expert_bytes = n_ff * (n_embd / 32) * 22;
    let down_expert_bytes = n_embd * (n_ff / 32) * 24;
    let mut expected = Vec::with_capacity(token_count * n_embd);
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let mut shared_mid = cpu_q8_0_rows(&shared_up, shared_ff, n_embd, token_input);
        for value in &mut shared_mid {
            *value = if *value > 0.0 { *value * *value } else { 0.0 };
        }
        let mut token_out = cpu_q8_0_rows(&shared_down, n_embd, shared_ff, &shared_mid);
        for (slot, &token_id) in token_ids.iter().enumerate() {
            if token_id as usize != token {
                continue;
            }
            let expert = expert_ids[slot] as usize;
            let up_start = expert * up_expert_bytes;
            let down_start = expert * down_expert_bytes;
            let mut mid = cpu_q5_basic_rows(
                &up_all[up_start..up_start + up_expert_bytes],
                n_ff,
                n_embd,
                22,
                false,
                token_input,
            );
            for value in &mut mid {
                *value = if *value > 0.0 { *value * *value } else { 0.0 };
            }
            let down = cpu_q5_basic_rows(
                &down_all[down_start..down_start + down_expert_bytes],
                n_embd,
                n_ff,
                24,
                true,
                &mid,
            );
            for row in 0..n_embd {
                token_out[row] += down[row] * route[slot];
            }
        }
        for row in 0..n_embd {
            token_out[row] += residual[token * n_embd + row];
        }
        expected.extend(token_out);
    }

    assert!(
        nemotron_q5_register_layer(&up_all, &down_all, n_expert, n_ff, n_embd)
            .expect("register Nemotron Q5 layer")
    );
    let actual = nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
        &shared_up,
        &shared_down,
        &up_all,
        &down_all,
        &expert_ids,
        &route,
        &token_ids,
        shared_ff,
        n_expert,
        n_ff,
        n_embd,
        token_count,
        &input,
        &residual,
    )
    .expect("CUDA cached Nemotron Q8 shared + sparse prefill MoE")
    .expect("cached Q8 shared + sparse fused prefill hit");

    assert_close_rows(
        "cached Nemotron Q8 shared + sparse prefill MoE",
        &actual,
        &expected,
        0.3,
    );
}

#[test]
fn cuda_nemotron_mamba2_split_projection_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let seq_len = 3usize;
    let d_inner = 4usize;
    let conv_channels = 6usize;
    let num_heads = 2usize;
    let rows = d_inner + conv_channels + num_heads;
    let projected = (0..seq_len * rows)
        .map(|i| (i as f32 - 11.0) * 0.125)
        .collect::<Vec<_>>();
    let dt_bias = vec![0.5f32, -0.25f32];

    let (z, conv, dt) = crate::runtime::test_support::nemotron_mamba2_split_projection_for_test(
        &projected,
        &dt_bias,
        seq_len,
        d_inner,
        conv_channels,
        num_heads,
    )
    .expect("split projection");

    let mut expected_z = vec![0.0f32; seq_len * d_inner];
    let mut expected_conv = vec![0.0f32; seq_len * conv_channels];
    let mut expected_dt = vec![0.0f32; seq_len * num_heads];
    for t in 0..seq_len {
        let src = &projected[t * rows..(t + 1) * rows];
        expected_z[t * d_inner..(t + 1) * d_inner].copy_from_slice(&src[..d_inner]);
        expected_conv[t * conv_channels..(t + 1) * conv_channels]
            .copy_from_slice(&src[d_inner..d_inner + conv_channels]);
        for h in 0..num_heads {
            expected_dt[t * num_heads + h] = src[d_inner + conv_channels + h] + dt_bias[h];
        }
    }

    assert_close_rows("Mamba2 split z", &z, &expected_z, 0.0);
    assert_close_rows("Mamba2 split conv", &conv, &expected_conv, 0.0);
    assert_close_rows("Mamba2 split dt", &dt, &expected_dt, 0.0);
}

#[test]
fn cuda_nemotron_mamba2_conv1d_bias_silu_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let seq_len = 3usize;
    let channels = 2usize;
    let kernel_size = 2usize;
    let input = (0..(seq_len + kernel_size) * channels)
        .map(|i| (i as f32 - 3.0) * 0.25)
        .collect::<Vec<_>>();
    let kernel = vec![0.5f32, -0.25, 0.125, 0.75];
    let bias = vec![0.1f32, -0.2];

    let actual = crate::runtime::test_support::nemotron_mamba2_conv1d_bias_silu_for_test(
        &input,
        &kernel,
        &bias,
        seq_len,
        channels,
        kernel_size,
    )
    .expect("Mamba2 conv bias SiLU");

    let mut expected = vec![0.0f32; seq_len * channels];
    for t in 0..seq_len {
        for c in 0..channels {
            let mut sum = bias[c];
            for k in 0..kernel_size {
                sum += input[(t + k) * channels + c] * kernel[k * channels + c];
            }
            expected[t * channels + c] = sum / (1.0 + (-sum).exp());
        }
    }

    assert_close_rows("Mamba2 conv bias SiLU", &actual, &expected, 1e-6);
}

#[test]
fn cuda_nemotron_mamba2_add_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let proj = vec![0.5f32, -1.0, 2.25, -3.5, 4.0];
    let residual = vec![-0.25f32, 0.75, -1.25, 3.5, -4.5];

    let actual =
        crate::runtime::test_support::nemotron_mamba2_add_residual_for_test(&proj, &residual)
            .expect("Mamba2 add residual");
    let expected = proj
        .iter()
        .zip(residual.iter())
        .map(|(proj, residual)| proj + residual)
        .collect::<Vec<_>>();

    assert_close_rows("Mamba2 add residual", &actual, &expected, 0.0);
}

#[test]
fn cuda_nemotron_mamba2_decode_scan_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let num_heads = 2usize;
    let head_dim = 3usize;
    let state_dim = 4usize;
    let n_group = 1usize;
    let mut cpu_state = (0..num_heads * head_dim * state_dim)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut gpu_state = cpu_state.clone();
    let x = (0..num_heads * head_dim)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.125)
        .collect::<Vec<_>>();
    let b = (0..n_group * state_dim)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.0625)
        .collect::<Vec<_>>();
    let c = (0..n_group * state_dim)
        .map(|i| ((i as f32 % 3.0) - 1.0) * 0.09375)
        .collect::<Vec<_>>();
    let dt = vec![0.5f32, 0.75f32];
    let a = vec![-0.25f32, -0.5f32];
    let d = vec![0.125f32, 0.25f32];

    let expected = cpu_nemotron_mamba2_decode_scan(
        &mut cpu_state,
        &x,
        &b,
        &c,
        &dt,
        &a,
        &d,
        num_heads,
        head_dim,
        state_dim,
        n_group,
    );
    let actual = nemotron_mamba2_decode_scan(
        &mut gpu_state,
        &x,
        &b,
        &c,
        &dt,
        &a,
        &d,
        num_heads,
        head_dim,
        state_dim,
        n_group,
    )
    .expect("CUDA Nemotron Mamba2 scan");
    assert_close_rows("Nemotron Mamba2 scan", &actual, &expected, 1e-5);
}

#[test]
fn cuda_nemotron_mamba2_decode_scan_rejects_state_dim_above_block_width() {
    let _guard = runtime_test_lock();
    let num_heads = 2usize;
    let head_dim = 16usize;
    let state_dim = 257usize;
    let n_group = 1usize;
    let d_inner = num_heads * head_dim;
    let bc_dim = n_group * state_dim;
    let mut state = vec![0.0f32; d_inner * state_dim];
    let x = vec![0.0f32; d_inner];
    let b = vec![0.0f32; bc_dim];
    let c = vec![0.0f32; bc_dim];
    let dt = vec![0.0f32; num_heads];
    let a = vec![0.0f32; num_heads];
    let d = vec![0.0f32; num_heads];

    let err = nemotron_mamba2_decode_scan(
        &mut state, &x, &b, &c, &dt, &a, &d, num_heads, head_dim, state_dim, n_group,
    )
    .expect_err("state_dim > 256 should fail before decode scan kernel");
    assert!(err.contains("state_dim"));
    assert!(err.contains("256"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_scan_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let seq_len = 3usize;
    let num_heads = 2usize;
    let head_dim = 3usize;
    let state_dim = 4usize;
    let n_group = 1usize;
    let d_inner = num_heads * head_dim;
    let bc_dim = n_group * state_dim;
    let conv_channels = d_inner + 2 * bc_dim;
    let mut cpu_state = (0..d_inner * state_dim)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut gpu_state = cpu_state.clone();
    let conv = (0..seq_len * conv_channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.0625)
        .collect::<Vec<_>>();
    let dt = (0..seq_len * num_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.25)
        .collect::<Vec<_>>();
    let a = vec![-0.25f32, -0.5f32];
    let d = vec![0.125f32, 0.25f32];

    let expected = cpu_nemotron_mamba2_prefill_scan(
        &mut cpu_state,
        &conv,
        &dt,
        &a,
        &d,
        seq_len,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        state_dim,
        n_group,
    );
    let actual = nemotron_mamba2_prefill_scan(
        &mut gpu_state,
        &conv,
        &dt,
        &a,
        &d,
        seq_len,
        d_inner,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        n_group,
        state_dim,
    )
    .expect("CUDA Nemotron Mamba2 prefill scan");
    assert_close_rows("Nemotron Mamba2 prefill scan", &actual, &expected, 1e-5);
    assert_close_rows(
        "Nemotron Mamba2 prefill state",
        &gpu_state,
        &cpu_state,
        1e-5,
    );
}

#[test]
fn cuda_nemotron_mamba2_prefill_scan_rejects_state_dim_above_block_width() {
    let _guard = runtime_test_lock();
    let seq_len = 1usize;
    let num_heads = 2usize;
    let head_dim = 16usize;
    let state_dim = 257usize;
    let n_group = 1usize;
    let d_inner = num_heads * head_dim;
    let bc_dim = n_group * state_dim;
    let conv_channels = d_inner + 2 * bc_dim;
    let mut state = vec![0.0f32; d_inner * state_dim];
    let conv = vec![0.0f32; seq_len * conv_channels];
    let dt = vec![0.0f32; seq_len * num_heads];
    let a = vec![0.0f32; num_heads];
    let d = vec![0.0f32; num_heads];

    let err = nemotron_mamba2_prefill_scan(
        &mut state,
        &conv,
        &dt,
        &a,
        &d,
        seq_len,
        d_inner,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        n_group,
        state_dim,
    )
    .expect_err("state_dim > 256 should fail before prefill scan kernel");
    assert!(err.contains("state_dim"));
    assert!(err.contains("256"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_rejects_non_q8_projection() {
    let _guard = runtime_test_lock();
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        2,
        64,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input = vec![0.0f32; 2 * 64];
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload input");
    let err = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q4_K,
        &[],
        4,
        64,
        GGML_Q8_0,
        &[],
        64,
        4,
        &[1.0; 64],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 2],
        &[0.0; 2],
        &[0.0; 2],
        &[1.0; 4],
        &mut vec![0.0; 4],
        &mut vec![0.0; 8],
        2,
        4,
        4,
        1,
        2,
        1,
        1,
        4,
        1,
        1,
        1.0e-5,
    )
    .expect_err("non-Q8 ssm_in should fail");
    assert!(err.contains("unsupported_ssm_in_quant"));
    assert!(release_device_tensor(input_id).expect("release input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_zero_projection_preserves_residual() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 2;
    const HIDDEN_DIM: usize = 64;
    const D_INNER: usize = 32;
    const D_STATE: usize = 4;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM;
    const CONV_KERNEL_SIZE: usize = 2;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    let input = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| (i as f32 - 31.0) * 0.01)
        .collect::<Vec<_>>();
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let zero_q8 = |rows: usize, cols: usize| vec![0u8; rows * (cols / 32) * 34];
    let mut conv_state = vec![0.0f32; (CONV_KERNEL_SIZE - 1) * CONV_CHANNELS];
    let mut delta_state = vec![0.0f32; D_INNER * D_STATE];

    let output = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &zero_q8(SSM_IN_ROWS, HIDDEN_DIM),
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &zero_q8(HIDDEN_DIM, D_INNER),
        HIDDEN_DIM,
        D_INNER,
        &[1.0; 64],
        &[0.0; 2 * CONV_CHANNELS],
        &[0.0; CONV_CHANNELS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[1.0; D_INNER],
        &mut conv_state,
        &mut delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        1.0e-5,
    )
    .expect("Mamba2 zero projection device path");
    let actual = download_device_tensor_f32(output.output_id).expect("download Mamba2 output");

    assert_close_rows("Mamba2 zero projection residual", &actual, &input, 0.001);
    assert!(release_device_tensor(output.output_id).expect("release Mamba2 output"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_handles_hidden_larger_than_projection() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 2;
    const HIDDEN_DIM: usize = 128;
    const D_INNER: usize = 32;
    const D_STATE: usize = 4;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM;
    const CONV_KERNEL_SIZE: usize = 2;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    let input = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| (i as f32 - 63.0) * 0.005)
        .collect::<Vec<_>>();
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let zero_q8 = |rows: usize, cols: usize| vec![0u8; rows * (cols / 32) * 34];
    let mut conv_state = vec![0.0f32; (CONV_KERNEL_SIZE - 1) * CONV_CHANNELS];
    let mut delta_state = vec![0.0f32; D_INNER * D_STATE];

    let output = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &zero_q8(SSM_IN_ROWS, HIDDEN_DIM),
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &zero_q8(HIDDEN_DIM, D_INNER),
        HIDDEN_DIM,
        D_INNER,
        &[1.0; HIDDEN_DIM],
        &[0.0; CONV_KERNEL_SIZE * CONV_CHANNELS],
        &[0.0; CONV_CHANNELS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[1.0; D_INNER],
        &mut conv_state,
        &mut delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        1.0e-5,
    )
    .expect("Mamba2 hidden larger than projection device path");
    let actual = download_device_tensor_f32(output.output_id).expect("download Mamba2 output");

    assert_close_rows("Mamba2 hidden larger residual", &actual, &input, 0.001);
    assert!(release_device_tensor(output.output_id).expect("release Mamba2 output"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_rejects_invalid_conv_channels() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 2;
    const HIDDEN_DIM: usize = 64;
    const D_INNER: usize = 32;
    const D_STATE: usize = 4;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM - 1;
    const CONV_KERNEL_SIZE: usize = 2;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input = vec![0.0f32; SEQ_LEN * HIDDEN_DIM];
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let zero_q8 = |rows: usize, cols: usize| vec![0u8; rows * (cols / 32) * 34];
    let mut conv_state = vec![0.0f32; (CONV_KERNEL_SIZE - 1) * CONV_CHANNELS];
    let mut delta_state = vec![0.0f32; D_INNER * D_STATE];

    let err = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &zero_q8(SSM_IN_ROWS, HIDDEN_DIM),
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &zero_q8(HIDDEN_DIM, D_INNER),
        HIDDEN_DIM,
        D_INNER,
        &[1.0; HIDDEN_DIM],
        &[0.0; CONV_KERNEL_SIZE * CONV_CHANNELS],
        &[0.0; CONV_CHANNELS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[1.0; D_INNER],
        &mut conv_state,
        &mut delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        1.0e-5,
    )
    .expect_err("invalid conv_channels should fail before kernels");
    assert!(err.contains("state_shape_mismatch"));
    assert!(err.contains("conv_channels"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_rejects_conv_kernel_size_one() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 2;
    const HIDDEN_DIM: usize = 64;
    const D_INNER: usize = 32;
    const D_STATE: usize = 4;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM;
    const CONV_KERNEL_SIZE: usize = 1;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input = vec![0.0f32; SEQ_LEN * HIDDEN_DIM];
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let zero_q8 = |rows: usize, cols: usize| vec![0u8; rows * (cols / 32) * 34];
    let mut conv_state = Vec::new();
    let mut delta_state = vec![0.0f32; D_INNER * D_STATE];

    let err = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &zero_q8(SSM_IN_ROWS, HIDDEN_DIM),
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &zero_q8(HIDDEN_DIM, D_INNER),
        HIDDEN_DIM,
        D_INNER,
        &[1.0; HIDDEN_DIM],
        &[0.0; CONV_KERNEL_SIZE * CONV_CHANNELS],
        &[0.0; CONV_CHANNELS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[1.0; D_INNER],
        &mut conv_state,
        &mut delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        1.0e-5,
    )
    .expect_err("conv_kernel_size=1 should fail");
    assert!(err.contains("unsupported_conv_shape"));
    assert!(err.contains("conv_kernel_size"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_rejects_state_dim_above_block_width() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 1;
    const HIDDEN_DIM: usize = 32;
    const D_INNER: usize = 32;
    const D_STATE: usize = 257;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM;
    const CONV_KERNEL_SIZE: usize = 2;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input = vec![0.0f32; SEQ_LEN * HIDDEN_DIM];
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let zero_q8 = |rows: usize, cols: usize| vec![0u8; rows * (cols / 32) * 34];
    let mut conv_state = vec![0.0f32; (CONV_KERNEL_SIZE - 1) * CONV_CHANNELS];
    let mut delta_state = vec![0.0f32; D_INNER * D_STATE];

    let err = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &zero_q8(SSM_IN_ROWS, HIDDEN_DIM),
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &zero_q8(HIDDEN_DIM, D_INNER),
        HIDDEN_DIM,
        D_INNER,
        &[1.0; HIDDEN_DIM],
        &[0.0; CONV_KERNEL_SIZE * CONV_CHANNELS],
        &[0.0; CONV_CHANNELS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[0.0; NUM_HEADS],
        &[1.0; D_INNER],
        &mut conv_state,
        &mut delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        1.0e-5,
    )
    .expect_err("d_state > 256 should fail before scan kernel");
    assert!(err.contains("state_dim"));
    assert!(err.contains("256"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_nemotron_mamba2_prefill_device_matches_nonzero_cpu_reference() {
    let _guard = runtime_test_lock();
    const SEQ_LEN: usize = 2;
    const HIDDEN_DIM: usize = 32;
    const D_INNER: usize = 32;
    const D_STATE: usize = 4;
    const N_GROUP: usize = 1;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 16;
    const BC_DIM: usize = N_GROUP * D_STATE;
    const CONV_CHANNELS: usize = D_INNER + 2 * BC_DIM;
    const CONV_KERNEL_SIZE: usize = 2;
    const SSM_IN_ROWS: usize = D_INNER + CONV_CHANNELS + NUM_HEADS;
    const EPS: f32 = 1.0e-5;
    let input = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.0375)
        .collect::<Vec<_>>();
    let input_norm = (0..HIDDEN_DIM)
        .map(|i| 0.75 + (i % 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let ssm_norm = (0..D_INNER)
        .map(|i| 0.625 + (i % 5) as f32 * 0.025)
        .collect::<Vec<_>>();
    let conv_kernel = (0..CONV_KERNEL_SIZE * CONV_CHANNELS)
        .map(|i| ((i as f32 % 9.0) - 4.0) * 0.0125)
        .collect::<Vec<_>>();
    let conv_bias = (0..CONV_CHANNELS)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.01)
        .collect::<Vec<_>>();
    let dt_bias = vec![0.125f32, -0.0625];
    let ssm_a = vec![-0.25f32, -0.5];
    let ssm_d = vec![0.2f32, -0.15];
    let ssm_in = make_test_q8_0_weights(SSM_IN_ROWS, HIDDEN_DIM, 313);
    let ssm_out = make_test_q8_0_weights(HIDDEN_DIM, D_INNER, 337);
    let input_desc = rnb_backend_api::DeviceTensorDesc::new(
        SEQ_LEN,
        HIDDEN_DIM,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::MoeOutput,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Mamba2 input");
    let mut gpu_conv_state = (0..(CONV_KERNEL_SIZE - 1) * CONV_CHANNELS)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.015)
        .collect::<Vec<_>>();
    let mut gpu_delta_state = (0..D_INNER * D_STATE)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.01)
        .collect::<Vec<_>>();
    let mut cpu_conv_state = gpu_conv_state.clone();
    let mut cpu_delta_state = gpu_delta_state.clone();
    let expected = cpu_nemotron_mamba2_prefill_device_reference(
        &input,
        &ssm_in,
        SSM_IN_ROWS,
        HIDDEN_DIM,
        &ssm_out,
        HIDDEN_DIM,
        D_INNER,
        &input_norm,
        &conv_kernel,
        &conv_bias,
        &dt_bias,
        &ssm_a,
        &ssm_d,
        &ssm_norm,
        &mut cpu_conv_state,
        &mut cpu_delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        EPS,
    );

    let output = nemotron_mamba2_prefill_device(
        input_id,
        input_desc,
        GGML_Q8_0,
        &ssm_in,
        SSM_IN_ROWS,
        HIDDEN_DIM,
        GGML_Q8_0,
        &ssm_out,
        HIDDEN_DIM,
        D_INNER,
        &input_norm,
        &conv_kernel,
        &conv_bias,
        &dt_bias,
        &ssm_a,
        &ssm_d,
        &ssm_norm,
        &mut gpu_conv_state,
        &mut gpu_delta_state,
        SEQ_LEN,
        HIDDEN_DIM,
        D_INNER,
        CONV_CHANNELS,
        BC_DIM,
        NUM_HEADS,
        HEAD_DIM,
        N_GROUP,
        D_STATE,
        CONV_KERNEL_SIZE,
        EPS,
    )
    .expect("Mamba2 nonzero device path");
    let actual = download_device_tensor_f32(output.output_id).expect("download Mamba2 output");

    assert_close_rows_abs_rel(
        "Mamba2 nonzero device output",
        &actual,
        &expected,
        0.03,
        0.02,
    );
    assert_close_rows(
        "Mamba2 nonzero conv_state",
        &gpu_conv_state,
        &cpu_conv_state,
        1.0e-6,
    );
    assert_close_rows_abs_rel(
        "Mamba2 nonzero delta_state",
        &gpu_delta_state,
        &cpu_delta_state,
        0.03,
        0.02,
    );
    assert!(actual
        .iter()
        .zip(input.iter())
        .any(|(&actual, &residual)| (actual - residual).abs() > 0.001));
    assert!(release_device_tensor(output.output_id).expect("release Mamba2 output"));
    assert!(release_device_tensor(input_id).expect("release Mamba2 input"));
}

#[test]
fn cuda_qwen35_sparse_experts_matches_cpu_q4k_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q4k_weights(selected, n_ff, n_embd / 256, 3);
    let up = make_test_q4k_weights(selected, n_ff, n_embd / 256, 17);
    let down = make_test_q4k_weights(selected, n_embd, n_ff / 256, 41);
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let route = vec![0.625f32, 0.375f32];
    let mut input = vec![0.0f32; n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 19.0) - 9.0) * 0.03125;
    }

    let expected = cpu_qwen35_sparse_q4k_reference(
        &gate_refs, &up_refs, &down_refs, &route, n_ff, n_embd, &input,
    );
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    let actual = match qwen35_sparse_experts(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        None,
        &[],
        &mut bundle_observation_receipt,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA sparse expert test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse expert failed: {err}"),
    };

    for row_idx in 0..n_embd {
        let diff = (actual[row_idx] - expected[row_idx]).abs();
        assert!(
            diff < 0.2,
            "CUDA sparse expert row {row_idx} mismatch: expected {}, actual {}, diff {}",
            expected[row_idx],
            actual[row_idx],
            diff
        );
    }
}

#[test]
fn cuda_qwen35_sparse_experts_into_matches_cpu_q2k_q3k_reference() {
    let _guard = runtime_test_lock();
    let _cache_trace = EnvVarGuard::set("RNB_CUDA_CACHE_TRACE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA sparse Q2_K/Q3_K expert test: {err}");
            return;
        }
    };
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;

    let selected = 1usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = (0..selected)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 101 + expert * 17))
        .collect::<Vec<_>>();
    let up = (0..selected)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 151 + expert * 19))
        .collect::<Vec<_>>();
    let down = (0..selected)
        .map(|expert| make_test_q3k_weights(n_embd, n_ff / 256, 211 + expert * 23))
        .collect::<Vec<_>>();
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let route = vec![1.0f32];
    let input = (0..n_embd)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();

    let expected = cpu_qwen35_sparse_q2k_q3k_reference(
        &gate_refs, &up_refs, &down_refs, &route, n_ff, n_embd, &input,
    );
    let before = cache_snapshot();
    let mut actual = vec![0.0f32; n_embd];
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    let prepared_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            Some(17),
            &[3],
            &mut bundle_observation_receipt,
            n_ff,
            n_embd,
        )
        .expect("prepare Q2_K/Q3_K expert bundle");
    assert_eq!(prepared_mask, [false]);
    assert_eq!(
        bundle_observation_receipt.pending_stats(),
        ExpertBundleCacheStats {
            bundle_lookups: 1,
            bundle_misses: 1,
            ..ExpertBundleCacheStats::default()
        }
    );
    state
        .qwen35_sparse_experts_into(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            Some(17),
            &[3],
            &mut bundle_observation_receipt,
            11,
            n_ff,
            n_embd,
            &input,
            &mut actual,
        )
        .expect("CUDA sparse Q2_K/Q3_K expert");

    assert!(bundle_observation_receipt.consumed());
    assert_eq!(
        bundle_observation_receipt.pending_stats(),
        ExpertBundleCacheStats::default(),
        "fallback trace must consume prepared bundle telemetry once"
    );
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (
            state.resident_q4k_limit / (gate[0].len() + up[0].len() + down[0].len()),
            1,
            1
        )
    );
    let bundle_delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(bundle_delta.bundle_lookups, 1);
    assert_eq!(bundle_delta.bundle_misses, 1);
    assert_close_rows_abs_rel("Q2_K/Q3_K sparse expert", &actual, &expected, 0.05, 0.02);
}

#[test]
fn cuda_qwen35_resident_per_slot_q2k_q3k_matches_cpu_layout() {
    let _guard = runtime_test_lock();
    let n_ff = 256usize;
    let n_embd = 256usize;
    let input = (0..n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    for selected in [1usize, 2, 8] {
        let mut state = match CudaState::open() {
            Ok(state) => state,
            Err(err) => {
                eprintln!("skipping CUDA resident per-slot Q2_K/Q3_K test: {err}");
                return;
            }
        };
        state.qwen35_target_decode_q4k_limit_checked = true;
        state.resident_q4k_limit = 16 * 1024 * 1024;
        let unique = match selected {
            1 | 2 => 1,
            8 => 4,
            _ => unreachable!(),
        };
        let gate = (0..unique)
            .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 501 + expert * 17))
            .collect::<Vec<_>>();
        let up = (0..unique)
            .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 601 + expert * 19))
            .collect::<Vec<_>>();
        let down = (0..unique)
            .map(|expert| make_test_q3k_weights(n_embd, n_ff / 256, 701 + expert * 23))
            .collect::<Vec<_>>();
        for weights in gate.iter().chain(up.iter()).chain(down.iter()) {
            assert!(state
                .preload_resident_q4k_weight_slice(weights)
                .expect("preload resident per-slot weight"));
        }
        let expert_for_slot = match selected {
            1 => vec![0],
            2 => vec![0, 0],
            8 => vec![0, 1, 2, 3, 1, 0, 3, 2],
            _ => unreachable!(),
        };
        let gate_refs = expert_for_slot
            .iter()
            .map(|&expert| gate[expert].as_slice())
            .collect::<Vec<_>>();
        let up_refs = expert_for_slot
            .iter()
            .map(|&expert| up[expert].as_slice())
            .collect::<Vec<_>>();
        let down_refs = expert_for_slot
            .iter()
            .map(|&expert| down[expert].as_slice())
            .collect::<Vec<_>>();

        let actual = state
            .qwen35_sparse_experts_per_slot_resident(
                &gate_refs, &up_refs, &down_refs, 11, n_ff, n_embd, &input,
            )
            .expect("resident per-slot Q2_K/Q3_K");
        assert_eq!(actual.len(), selected * n_embd);

        let mut expected = Vec::with_capacity(selected * n_embd);
        for slot in 0..selected {
            expected.extend(cpu_qwen35_sparse_q2k_q3k_reference(
                &[gate_refs[slot]],
                &[up_refs[slot]],
                &[down_refs[slot]],
                &[1.0],
                n_ff,
                n_embd,
                &input,
            ));
        }
        assert_close_rows_abs_rel(
            &format!("Q2_K/Q3_K resident per-slot selected={selected}"),
            &actual,
            &expected,
            0.05,
            0.02,
        );

        let route_weights = (0..selected)
            .map(|slot| 0.125 + slot as f32 * 0.0625)
            .collect::<Vec<_>>();
        let expected_weighted = cpu_qwen35_sparse_q2k_q3k_reference(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route_weights,
            n_ff,
            n_embd,
            &input,
        );
        let mut actual_weighted = vec![0.0f32; n_embd];
        for slot in 0..selected {
            for row in 0..n_embd {
                actual_weighted[row] += actual[slot * n_embd + row] * route_weights[slot];
            }
        }
        assert_close_rows_abs_rel(
            &format!("Q2_K/Q3_K host-weighted resident slots selected={selected}"),
            &actual_weighted,
            &expected_weighted,
            0.05,
            0.02,
        );
    }
}

#[test]
fn cuda_qwen35_resident_per_slot_rejects_miss_without_weight_upload() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA resident per-slot miss test: {err}");
            return;
        }
    };
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q2k_weights(n_ff, n_embd / 256, 801);
    let up = make_test_q2k_weights(n_ff, n_embd / 256, 811);
    let down = make_test_q3k_weights(n_embd, n_ff / 256, 821);
    assert!(state
        .preload_resident_q4k_weight_slice(&gate)
        .expect("preload resident gate"));
    assert!(state
        .preload_resident_q4k_weight_slice(&up)
        .expect("preload resident up"));
    let before = cache_snapshot();
    let history_before = state.qwen35_expert_bundle_history_state_for_test();

    let err = state
        .qwen35_sparse_experts_per_slot_resident(
            &[&gate],
            &[&up],
            &[&down],
            11,
            n_ff,
            n_embd,
            &vec![0.25f32; n_embd],
        )
        .expect_err("nonresident down must be rejected");

    assert!(err.contains("slot 0 is missing"));
    assert!(!state.q4k_weight_slice_is_resident(&down));
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        history_before
    );
    let delta = cache_snapshot().delta(before);
    assert_eq!(delta.temp_upload_bytes, 0);
    assert_eq!(delta.expert_bundles.h2d_bytes, 0);
    assert_eq!(delta.expert_bundles.temp_h2d_bytes, 0);
    assert_eq!(delta.expert_bundles.bundle_lookups, 0);
}

#[test]
fn qwen35_prepare_route_weights_aggregate_duplicate_experts() {
    let gate_a = [1u8];
    let up_a = [2u8];
    let down_a = [3u8];
    let gate_b = [4u8];
    let up_b = [5u8];
    let down_b = [6u8];

    let bundles = CudaState::qwen35_decode_expert_bundle_routes_for_test(
        &[7, 3, 7],
        &[&gate_a, &gate_b, &gate_a],
        &[&up_a, &up_b, &up_a],
        &[&down_a, &down_b, &down_a],
        &[0.25, 0.9, 0.5],
    )
    .expect("aggregate duplicate expert routes");

    assert_eq!(
        bundles,
        [(3, 1, f64::from(0.9f32)), (7, 0, f64::from(0.75f32))]
    );
}

#[test]
fn qwen35_prepare_route_weights_reject_invalid_length_and_values() {
    let gate = [1u8];
    let up = [2u8];
    let down = [3u8];
    let inputs = (&[&gate[..]][..], &[&up[..]][..], &[&down[..]][..]);

    let length_err = CudaState::qwen35_decode_expert_bundle_routes_for_test(
        &[5],
        inputs.0,
        inputs.1,
        inputs.2,
        &[],
    )
    .expect_err("route length mismatch must fail");
    assert!(length_err.contains("input length mismatch"));

    for route_weight in [f32::NAN, -0.25] {
        let value_err = CudaState::qwen35_decode_expert_bundle_routes_for_test(
            &[5],
            inputs.0,
            inputs.1,
            inputs.2,
            &[route_weight],
        )
        .expect_err("non-finite or negative route weight must fail");
        assert!(value_err.contains("finite and non-negative"));
    }
}

#[test]
fn cuda_qwen35_prepare_residency_prefers_larger_route_weight_at_equal_reuse() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA route-weight prepare residency test: {err}");
            return;
        }
    };
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = (0..2)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 1401 + expert * 17))
        .collect::<Vec<_>>();
    let up = (0..2)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 1501 + expert * 19))
        .collect::<Vec<_>>();
    let down = (0..2)
        .map(|expert| make_test_q3k_weights(n_embd, n_ff / 256, 1601 + expert * 23))
        .collect::<Vec<_>>();
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = gate[0].len() + up[0].len() + down[0].len();

    let mut receipt = ExpertBundleObservationReceipt::default();
    let mask = state
        .qwen35_with_decode_reuse_score_override_for_test(2, |state| {
            state.qwen35_prepare_selected_bundle_residency(
                &gate_refs,
                &up_refs,
                &down_refs,
                &[0.1, 0.9],
                Some(23),
                &[5, 7],
                &mut receipt,
                n_ff,
                n_embd,
            )
        })
        .expect("admit higher-weight bundle");
    assert!(receipt.consumed());

    assert_eq!(mask, [false, true]);
    assert!(!state.q4k_weight_slice_is_resident(&gate[0]));
    assert!(state.q4k_weight_slice_is_resident(&gate[1]));
    assert!(state.q4k_weight_slice_is_resident(&up[1]));
    assert!(state.q4k_weight_slice_is_resident(&down[1]));
}

#[test]
fn cuda_qwen35_prepare_residency_cold_repeated_and_receipt_once() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA prepare residency test: {err}");
            return;
        }
    };
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q2k_weights(n_ff, n_embd / 256, 901);
    let up = make_test_q2k_weights(n_ff, n_embd / 256, 911);
    let down = make_test_q3k_weights(n_embd, n_ff / 256, 921);
    let before = cache_snapshot();
    let mut receipt = ExpertBundleObservationReceipt::default();

    let cold_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(21),
            &[5],
            &mut receipt,
            n_ff,
            n_embd,
        )
        .expect("prepare cold bundle");
    assert_eq!(cold_mask, [false]);
    assert!(receipt.consumed());
    let same_token_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(21),
            &[5],
            &mut receipt,
            n_ff,
            n_embd,
        )
        .expect("prepare same-token bundle");
    assert_eq!(same_token_mask, [false]);
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        (
            state.resident_q4k_limit / (gate.len() + up.len() + down.len()),
            1,
            1
        )
    );

    let mut second_receipt = ExpertBundleObservationReceipt::default();
    let second_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(21),
            &[5],
            &mut second_receipt,
            n_ff,
            n_embd,
        )
        .expect("prepare first reused bundle");
    assert_eq!(second_mask, [false]);
    assert!(second_receipt.consumed());

    let mut third_receipt = ExpertBundleObservationReceipt::default();
    let third_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(21),
            &[5],
            &mut third_receipt,
            n_ff,
            n_embd,
        )
        .expect("prepare second reused bundle");
    assert_eq!(third_mask, [true]);
    assert!(third_receipt.consumed());
    assert!(state.q4k_weight_slice_is_resident(&gate));
    assert!(state.q4k_weight_slice_is_resident(&up));
    assert!(state.q4k_weight_slice_is_resident(&down));
    let delta = cache_snapshot().delta(before);
    assert_eq!(delta.expert_bundles.bundle_lookups, 3);
    assert_eq!(delta.expert_bundles.bundle_admissions, 1);
    assert_eq!(delta.expert_bundles.temp_h2d_bytes, 0);
}

#[test]
fn cuda_qwen35_prepare_residency_partial_uploads_missing_roles_only() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA partial prepare residency test: {err}");
            return;
        }
    };
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q2k_weights(n_ff, n_embd / 256, 1001);
    let up = make_test_q2k_weights(n_ff, n_embd / 256, 1011);
    let down = make_test_q3k_weights(n_embd, n_ff / 256, 1021);
    assert!(state
        .preload_resident_q4k_weight_slice(&gate)
        .expect("preload partial gate"));
    let gate_ptr = state.resident_q4k[&q4k_resident_key(&gate)].ptr;
    let mut cold_receipt = ExpertBundleObservationReceipt::default();
    assert_eq!(
        state
            .qwen35_prepare_selected_bundle_residency(
                &[&gate],
                &[&up],
                &[&down],
                &[1.0],
                Some(22),
                &[7],
                &mut cold_receipt,
                n_ff,
                n_embd,
            )
            .expect("seed partial bundle history"),
        [false]
    );
    assert!(cold_receipt.consumed());
    let before = cache_snapshot();
    let mut first_reuse_receipt = ExpertBundleObservationReceipt::default();
    let first_reuse_mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(22),
            &[7],
            &mut first_reuse_receipt,
            n_ff,
            n_embd,
        )
        .expect("observe first partial-bundle reuse");
    assert_eq!(first_reuse_mask, [false]);

    let mut second_reuse_receipt = ExpertBundleObservationReceipt::default();
    let mask = state
        .qwen35_prepare_selected_bundle_residency(
            &[&gate],
            &[&up],
            &[&down],
            &[1.0],
            Some(22),
            &[7],
            &mut second_reuse_receipt,
            n_ff,
            n_embd,
        )
        .expect("admit missing partial roles on second reuse");

    assert_eq!(mask, [true]);
    assert_eq!(state.resident_q4k[&q4k_resident_key(&gate)].ptr, gate_ptr);
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_admissions, 1);
    assert_eq!(delta.admitted_bytes, (up.len() + down.len()) as u64);
    assert_eq!(delta.h2d_bytes, (up.len() + down.len()) as u64);
    assert_eq!(delta.temp_h2d_bytes, 0);
    assert_eq!(
        second_reuse_receipt.pending_stats(),
        ExpertBundleCacheStats {
            bundle_lookups: 1,
            bundle_partial_hits: 1,
            bundle_admissions: 1,
            admitted_bytes: (up.len() + down.len()) as u64,
            h2d_bytes: (up.len() + down.len()) as u64,
            ..ExpertBundleCacheStats::default()
        }
    );
}

#[test]
fn cuda_qwen35_prepare_residency_without_layer_only_reports_current_mask() {
    let _guard = runtime_test_lock();
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) => {
            eprintln!("skipping CUDA no-layer prepare residency test: {err}");
            return;
        }
    };
    state.qwen35_target_decode_q4k_limit_checked = true;
    state.resident_q4k_limit = 8 * 1024 * 1024;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = (0..2)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 1101 + expert * 17))
        .collect::<Vec<_>>();
    let up = (0..2)
        .map(|expert| make_test_q2k_weights(n_ff, n_embd / 256, 1201 + expert * 19))
        .collect::<Vec<_>>();
    let down = (0..2)
        .map(|expert| make_test_q3k_weights(n_embd, n_ff / 256, 1301 + expert * 23))
        .collect::<Vec<_>>();
    for weights in [&gate[0], &up[0], &down[0]] {
        assert!(state
            .preload_resident_q4k_weight_slice(weights)
            .expect("preload no-layer resident role"));
    }
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let before = cache_snapshot();
    let history_before = state.qwen35_expert_bundle_history_state_for_test();
    let mut receipt = ExpertBundleObservationReceipt::default();

    let mask = state
        .qwen35_prepare_selected_bundle_residency(
            &gate_refs,
            &up_refs,
            &down_refs,
            &[1.0, 1.0],
            None,
            &[11, 13],
            &mut receipt,
            n_ff,
            n_embd,
        )
        .expect("report no-layer resident mask");

    assert_eq!(mask, [true, false]);
    assert!(receipt.consumed());
    assert_eq!(
        receipt.pending_stats(),
        ExpertBundleCacheStats::default(),
        "no-layer prepare does not mutate cache telemetry"
    );
    assert_eq!(
        state.qwen35_expert_bundle_history_state_for_test(),
        history_before
    );
    let delta = cache_snapshot().delta(before).expert_bundles;
    assert_eq!(delta.bundle_lookups, 0);
    assert_eq!(delta.bundle_admissions, 0);
    assert_eq!(delta.h2d_bytes, 0);
    assert_eq!(delta.temp_h2d_bytes, 0);
}

#[test]
fn cuda_qwen35_sparse_experts_iq4xs_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_iq4_xs_weights(selected, n_ff, n_embd / 256, 211);
    let up = make_test_iq4_xs_weights(selected, n_ff, n_embd / 256, 223);
    let down = make_test_iq4_xs_weights(selected, n_embd, n_ff / 256, 227);
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let route = vec![0.5625f32, 0.4375f32];
    let mut input = vec![0.0f32; n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 17.0) - 8.0) * 0.0234375;
    }

    let expected = cpu_qwen35_sparse_iq4xs_reference(
        &gate_refs, &up_refs, &down_refs, &route, n_ff, n_embd, &input,
    );
    let actual = match qwen35_sparse_experts_iq4xs(
        &gate_refs, &up_refs, &down_refs, &route, 23, n_ff, n_embd, &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA sparse IQ4_XS expert test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse IQ4_XS expert failed: {err}"),
    };

    for row_idx in 0..n_embd {
        let diff = (actual[row_idx] - expected[row_idx]).abs();
        assert!(
            diff < 0.2,
            "CUDA sparse IQ4_XS expert row {row_idx} mismatch: expected {}, actual {}, diff {}",
            expected[row_idx],
            actual[row_idx],
            diff
        );
    }
}

#[test]
fn cuda_qwen35_sparse_experts_device_roundtrip_matches_host_cuda() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let selected = 2usize;
    let n_ff = 256usize;
    let n_embd = 256usize;
    let gate = make_test_q4k_weights(selected, n_ff, n_embd / 256, 5);
    let up = make_test_q4k_weights(selected, n_ff, n_embd / 256, 19);
    let down = make_test_q4k_weights(selected, n_embd, n_ff / 256, 43);
    let gate_refs = gate.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = down.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let route = vec![0.625f32, 0.375f32];
    let mut input = vec![0.0f32; n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 23.0) - 11.0) * 0.03125;
    }

    let mut host_bundle_observation_consumed = ExpertBundleObservationReceipt::default();
    let host = match qwen35_sparse_experts(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        None,
        &[],
        &mut host_bundle_observation_consumed,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA device roundtrip test: {err}");
            return;
        }
        Err(err) => panic!("host CUDA sparse failed: {err}"),
    };
    let mut device_bundle_observation_consumed = ExpertBundleObservationReceipt::default();
    let device = qwen35_sparse_experts_device_roundtrip(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        None,
        &[],
        &mut device_bundle_observation_consumed,
        12,
        n_ff,
        n_embd,
        &input,
    )
    .expect("device roundtrip sparse MoE");

    assert_eq!(host.len(), device.len());
    for (row_idx, (&expected, &actual)) in host.iter().zip(device.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 1e-5,
                "CUDA device roundtrip row {row_idx} mismatch: expected {expected}, actual {actual}, diff {diff}",
            );
    }
}
#[test]
fn cuda_qwen35_sparse_experts_by_token_groups_repeated_expert() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 7);
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 23);
    let down = make_test_q4k_weights(1, n_embd, n_ff / 256, 47);
    let gate_refs = vec![gate[0].as_slice(), gate[0].as_slice()];
    let up_refs = vec![up[0].as_slice(), up[0].as_slice()];
    let down_refs = vec![down[0].as_slice(), down[0].as_slice()];
    let route = vec![0.625f32, 0.375f32];
    let token_ids = vec![0u32, 1u32];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 29.0) - 14.0) * 0.01953125;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_reference(
            &[gate[0].as_slice()],
            &[up[0].as_slice()],
            &[down[0].as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA sparse by-token grouped test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse by-token grouped failed: {err}"),
    };

    assert_close_rows_abs_rel(
        "CUDA sparse by-token grouped",
        &actual,
        &expected,
        0.5,
        0.01,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q4_group4_matches_default_exact_order() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _group4_down = EnvVarGuard::set("RNB_CUDA_GROUP4_DOWN", "1");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
    let _q4_down_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 401);
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 409);
    let down = make_test_q4k_weights(1, n_embd, n_ff / 256, 419);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
    ];
    let route = vec![0.125f32, 0.375, 0.25, 0.5];
    let token_ids = vec![0u32, 1, 0, 1];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.017578125)
        .collect::<Vec<_>>();

    let default = {
        let _q4_group4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            12,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q4 group4 exact-order test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q4 default selected down failed: {err}"),
        }
    };
    let group4 = {
        let _q4_group4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "1");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            12,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q4 group4 exact-order test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q4 group4 selected down failed: {err}"),
        }
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&group4, &default);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 group4 selected down changed default CUDA output order: row={max_idx} default={:.9} group4={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        default[max_idx],
        group4[max_idx]
    );
}

#[test]
fn cuda_q4k_selected_gate_up_pair2_matches_independent_slots() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _warp8 = EnvVarGuard::set("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8", "1");

    let rows = 259usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let gate = make_test_q4k_weights(6, rows, blocks_per_row, 421);
    let up = make_test_q4k_weights(6, rows, blocks_per_row, 433);
    let expert_ids = vec![0u32, 1, 2, 3, 2, 4, 0, 5];
    let gate_refs = expert_ids
        .iter()
        .map(|&expert| gate[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let up_refs = expert_ids
        .iter()
        .map(|&expert| up[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let input = (0..2 * cols)
        .map(|index| ((index as f32 % 67.0) - 33.0) * 0.01171875)
        .collect::<Vec<_>>();

    let independent = match q4k_selected_gate_up_pair2_for_test(
        &gate_refs,
        &up_refs,
        &expert_ids,
        rows,
        cols,
        &input,
        false,
        false,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA Q4 pair2 selected gate/up test: {err}");
            return;
        }
        Err(err) => panic!("CUDA Q4 independent selected gate/up failed: {err}"),
    };
    let paired = q4k_selected_gate_up_pair2_for_test(
        &gate_refs,
        &up_refs,
        &expert_ids,
        rows,
        cols,
        &input,
        true,
        false,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 pair2 selected gate/up failed: {err}"));

    for (name, actual, expected) in [
        ("gate", &paired.0, &independent.0),
        ("up", &paired.1, &independent.1),
    ] {
        let (max_idx, max_abs, max_rel) = max_abs_rel(actual, expected);
        assert!(
            max_abs <= 1.0e-6,
            "Q4 pair2 {name} changed independent CUDA output: idx={max_idx} independent={:.9} paired={:.9} abs={max_abs:.9} rel={max_rel:.9}",
            expected[max_idx],
            actual[max_idx]
        );
    }
    let separate_silu = q4k_selected_gate_up_pair2_for_test(
        &gate_refs,
        &up_refs,
        &expert_ids,
        rows,
        cols,
        &input,
        false,
        true,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 separate pair2 SiLU failed: {err}"));
    let fused_silu = q4k_selected_gate_up_pair2_for_test(
        &gate_refs,
        &up_refs,
        &expert_ids,
        rows,
        cols,
        &input,
        true,
        true,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 fused pair2 SiLU failed: {err}"));
    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused_silu.0, &separate_silu.0);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 pair2 fused SiLU changed separate CUDA output: idx={max_idx} separate={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        separate_silu.0[max_idx],
        fused_silu.0[max_idx]
    );
}

#[test]
fn cuda_q5k_selected_down_pair2_matches_independent_slots() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let rows = 259usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let down = (0..6usize)
        .map(|expert| make_test_q5k_weights(rows, blocks_per_row, 449 + expert * 13))
        .collect::<Vec<_>>();
    let expert_ids = vec![0u32, 1, 2, 3, 2, 4, 0, 5];
    let down_refs = expert_ids
        .iter()
        .map(|&expert| down[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let route = (1..=expert_ids.len())
        .map(|value| value as f32 / (expert_ids.len() + 1) as f32)
        .collect::<Vec<_>>();
    let input = (0..expert_ids.len() * cols)
        .map(|index| ((index as f32 % 71.0) - 35.0) * 0.0078125)
        .collect::<Vec<_>>();

    let independent = match q5k_selected_down_pair2_for_test(
        &down_refs,
        &expert_ids,
        &route,
        rows,
        cols,
        &input,
        false,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA Q5 pair2 selected down test: {err}");
            return;
        }
        Err(err) => panic!("CUDA Q5 independent selected down failed: {err}"),
    };
    let paired =
        q5k_selected_down_pair2_for_test(&down_refs, &expert_ids, &route, rows, cols, &input, true)
            .unwrap_or_else(|err| panic!("CUDA Q5 pair2 selected down failed: {err}"));
    let paired_repeat =
        q5k_selected_down_pair2_for_test(&down_refs, &expert_ids, &route, rows, cols, &input, true)
            .unwrap_or_else(|err| panic!("CUDA Q5 repeated pair2 selected down failed: {err}"));
    let (repeat_idx, repeat_abs, repeat_rel) = max_abs_rel(&paired_repeat, &paired);
    assert!(
        repeat_abs <= 1.0e-4,
        "Q5 pair2 selected down exceeded the existing atomic roundoff envelope across identical runs: idx={repeat_idx} first={:.9} repeated={:.9} abs={repeat_abs:.9} rel={repeat_rel:.9}",
        paired[repeat_idx],
        paired_repeat[repeat_idx]
    );

    let (max_idx, max_abs, max_rel) = max_abs_rel(&paired, &independent);
    assert!(
        max_abs <= 1.0e-4,
        "Q5 pair2 selected down changed independent CUDA output: idx={max_idx} independent={:.9} paired={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        independent[max_idx],
        paired[max_idx]
    );
}

#[test]
fn cuda_q4k_selected_gate_up_silu_group8_matches_separate_silu() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_GATE_UP_WARP4", "1");
    let _group16 = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");

    let rows = 256usize;
    let cols = 256usize;
    let blocks_per_row = cols / 256;
    let token_count = 2usize;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 431)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 439)
        .pop()
        .unwrap();
    let gate_refs = vec![
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
    ];
    let up_refs = vec![up.as_slice(), up.as_slice(), up.as_slice(), up.as_slice()];
    let token_ids = vec![0u32, 1, 0, 1];
    let group_meta = vec![0u32, 4];
    let input = (0..token_count * cols)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.01953125)
        .collect::<Vec<_>>();

    let separate = match q4k_selected_gate_up_silu_group8_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &group_meta,
        rows,
        cols,
        &input,
        false,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA Q4 gate/up fused-silu launcher test: {err}");
            return;
        }
        Err(err) => panic!("CUDA Q4 gate/up separate SiLU failed: {err}"),
    };
    let fused = q4k_selected_gate_up_silu_group8_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &group_meta,
        rows,
        cols,
        &input,
        true,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 gate/up fused SiLU failed: {err}"));

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &separate);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 group8 fused gate/up SiLU changed separate CUDA output: idx={max_idx} separate={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        separate[max_idx],
        fused[max_idx]
    );
}

#[test]
fn cuda_q4k_selected_gate_up_silu_pack4_matches_separate_pack4() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let rows = 256usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, rows, blocks_per_row, 461);
    let up = make_test_q4k_weights(2, rows, blocks_per_row, 463);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let token_ids = vec![0u32, 1, 2, 0, 2];
    let group_meta = vec![0u32, 3, 3, 2];
    let input = (0..token_count * cols)
        .map(|i| ((i as f32 % 59.0) - 29.0) * 0.0107421875)
        .collect::<Vec<_>>();

    let separate = match q4k_selected_gate_up_silu_pack4_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &group_meta,
        rows,
        cols,
        &input,
        false,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA Q4 gate/up fused pack4 launcher test: {err}");
            return;
        }
        Err(err) => panic!("CUDA Q4 gate/up separate pack4 failed: {err}"),
    };
    let fused = q4k_selected_gate_up_silu_pack4_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &group_meta,
        rows,
        cols,
        &input,
        true,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 gate/up fused pack4 failed: {err}"));

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &separate);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 fused gate/up pack4 changed separate CUDA pack output: idx={max_idx} separate={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        separate[max_idx],
        fused[max_idx]
    );
}

#[test]
fn cuda_q4k_selected_gate_up_silu_pack4_group8_matches_group8_then_pack4() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _group8_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP8_GATE_UP_WARP4", "1");
    let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");

    let rows = 256usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(1, rows, blocks_per_row, 491)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, rows, blocks_per_row, 499)
        .pop()
        .unwrap();
    let gate_refs = vec![
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
        gate.as_slice(),
    ];
    let up_refs = vec![
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
        up.as_slice(),
    ];
    let token_ids = vec![0u32, 1, 2, 0, 1, 2, 0, 1];
    let gate_up_group_meta = vec![0u32, 8];
    let down_group_meta = vec![0u32, 4, 4, 4];
    let input = (0..token_count * cols)
        .map(|i| ((i as f32 % 67.0) - 33.0) * 0.009765625)
        .collect::<Vec<_>>();

    let separate = match q4k_selected_gate_up_silu_pack4_group8_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &gate_up_group_meta,
        &down_group_meta,
        rows,
        cols,
        &input,
        false,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA Q4 group8 gate/up pack4 launcher test: {err}");
            return;
        }
        Err(err) => panic!("CUDA Q4 group8 gate/up separate pack4 failed: {err}"),
    };
    let fused = q4k_selected_gate_up_silu_pack4_group8_for_test(
        &gate_refs,
        &up_refs,
        &token_ids,
        &gate_up_group_meta,
        &down_group_meta,
        rows,
        cols,
        &input,
        true,
    )
    .unwrap_or_else(|err| panic!("CUDA Q4 group8 gate/up fused pack4 failed: {err}"));

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &separate);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 group8 fused gate/up pack4 changed group8+pack4 output: idx={max_idx} separate={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        separate[max_idx],
        fused[max_idx]
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q4_gate_up_silu_fused_matches_default_exact_order() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _group8_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP8_GATE_UP_WARP4", "1");
    let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");
    let _group2_down = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8_down = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _q4_down_group4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "1");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 443);
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 449);
    let down = make_test_q4k_weights(1, n_embd, n_ff / 256, 457);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
    ];
    let route = vec![0.125f32, 0.375, 0.25, 0.5];
    let token_ids = vec![0u32, 1, 0, 1];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.015625)
        .collect::<Vec<_>>();

    let default = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            12,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q4 gate/up fused-silu path test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q4 default gate/up path failed: {err}"),
        }
    };
    let fused = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "1");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            12,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q4 fused gate/up path failed: {err}"))
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &default);
    assert!(
        max_abs <= 1.0e-6,
        "Q4 fused gate/up SiLU changed default sparse output: row={max_idx} default={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        default[max_idx],
        fused[max_idx]
    );
}

#[test]
fn cuda_qwen35_cached_q4_shared_sparse_by_token_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let prev_enabled = std::env::var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE").ok();
    let prev_prefill_f32 = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
    let prev_short_window = std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ").ok();
    let prev_cache = std::env::var("RNB_CUDA_Q4_F32_CACHE_MB").ok();
    std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", "64");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 61)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 67)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 71)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 73);
    let sparse_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 79);
    let sparse_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 83);
    let gate_refs = vec![sparse_gate[0].as_slice(), sparse_gate[0].as_slice()];
    let up_refs = vec![sparse_up[0].as_slice(), sparse_up[0].as_slice()];
    let down_refs = vec![sparse_down[0].as_slice(), sparse_down[0].as_slice()];
    let expert_ids = vec![0u32, 0u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.5f32, 0.75f32];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &[sparse_gate[0].as_slice()],
            &[sparse_up[0].as_slice()],
            &[sparse_down[0].as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &gate_refs,
        &up_refs,
        &down_refs,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        12,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => panic!("cached Q4 shared MoE path was not admitted"),
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping cached Q4 shared MoE test: {err}");
            if let Some(prev) = prev_enabled {
                std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev);
            } else {
                std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
            }
            if let Some(prev) = prev_cache {
                std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", prev);
            } else {
                std::env::remove_var("RNB_CUDA_Q4_F32_CACHE_MB");
            }
            if let Some(prev) = prev_prefill_f32 {
                std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev);
            } else {
                std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
            }
            if let Some(prev) = prev_short_window {
                std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", prev);
            } else {
                std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
            }
            return;
        }
        Err(err) => panic!("cached Q4 shared MoE failed: {err}"),
    };

    if let Some(prev) = prev_enabled {
        std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev);
    } else {
        std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
    }
    if let Some(prev) = prev_cache {
        std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4_F32_CACHE_MB");
    }
    if let Some(prev) = prev_prefill_f32 {
        std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev);
    } else {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
    }
    if let Some(prev) = prev_short_window {
        std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", prev);
    } else {
        std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
    }

    assert_close_rows_abs_rel("cached Q4 shared sparse MoE", &actual, &expected, 0.2, 0.02);
}

#[test]
fn cuda_qwen35_cached_q4_shared_selected_base_temp_slab_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _enabled = EnvVarGuard::set("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    let _prefill_f32 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    let _short_window = EnvVarGuard::set("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    let _q4_cache = EnvVarGuard::set("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    let _temp_slab_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS", "1");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 61)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 67)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 71)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 73)
        .pop()
        .unwrap();
    let sparse_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 79)
        .pop()
        .unwrap();
    let sparse_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 83)
        .pop()
        .unwrap();
    let expert_ids = vec![0u32, 0u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.5f32, 0.75f32];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &[sparse_gate.as_slice()],
            &[sparse_up.as_slice()],
            &[sparse_down.as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        12,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => panic!("cached Q4 shared selected-base temp-slab MoE path was not admitted"),
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping cached Q4 shared selected-base temp-slab MoE test: {err}");
            return;
        }
        Err(err) => panic!("cached Q4 shared selected-base temp-slab MoE failed: {err}"),
    };

    assert_close_rows_abs_rel(
        "cached Q4 shared selected-base temp-slab sparse MoE",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_cached_q4_shared_selected_base_full_layer_resident_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _enabled = EnvVarGuard::set("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    let _prefill_f32 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    let _short_window = EnvVarGuard::set("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    let _q4_cache = EnvVarGuard::set("RNB_CUDA_Q4_F32_CACHE_MB", "64");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 181)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 191)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 193)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 197)
        .pop()
        .unwrap();
    let sparse_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 199)
        .pop()
        .unwrap();
    let sparse_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 211)
        .pop()
        .unwrap();
    let expert_ids = vec![0u32, 0u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.5f32, 0.75f32];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &[sparse_gate.as_slice()],
            &[sparse_up.as_slice()],
            &[sparse_down.as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping cached selected-base full-layer resident CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_moe_layer_limit = usize::MAX;
    assert!(state
        .register_qwen35_moe_layer_without_eviction(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            12,
            n_ff,
            n_embd,
        )
        .expect("register selected-base full-layer resident weights"));

    let actual = state
        .qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
            &shared_gate,
            &shared_up,
            &shared_down,
            &shared_route,
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route,
            &token_ids,
            token_count,
            12,
            12,
            n_ff,
            n_embd,
            &input,
        )
        .expect("cached selected-base full-layer resident execution")
        .expect("cached selected-base full-layer resident path admitted");

    assert_close_rows_abs_rel(
        "cached selected-base full-layer resident sparse MoE",
        &actual,
        &expected,
        0.2,
        0.02,
    );
    let stats = state
        .last_qwen35_selected_sparse_boundary_stats
        .expect("selected-base execution boundary stats");
    assert_eq!(stats.selected_upload_calls, 0);
    assert_eq!(stats.selected_upload_bytes, 0);
}

#[test]
fn cuda_qwen35_cached_q4_shared_q6_down_selected_base_full_layer_resident_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _enabled = EnvVarGuard::set("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    let _prefill_f32 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    let _short_window = EnvVarGuard::set("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    let _q4_cache = EnvVarGuard::set("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    let _q6_cache = EnvVarGuard::set("RNB_CUDA_Q6_F32_CACHE_MB", "64");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 223)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 227)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 229)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 233)
        .pop()
        .unwrap();
    let sparse_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 239)
        .pop()
        .unwrap();
    let sparse_down = make_test_q6k_weights(1, n_embd, n_ff / 256, 241)
        .pop()
        .unwrap();
    let expert_ids = vec![0u32, 0u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.0f32; token_count];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let sparse = cpu_qwen35_sparse_q4k_q6k_reference(
            &[sparse_gate.as_slice()],
            &[sparse_up.as_slice()],
            &[sparse_down.as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&sparse);
    }

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping cached Q6-down selected-base full-layer resident CUDA test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    state.resident_moe_layer_limit = usize::MAX;
    assert!(state
        .register_qwen35_moe_layer_without_eviction(
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            14,
            n_ff,
            n_embd,
        )
        .expect("register Q6-down selected-base full-layer resident weights"));

    let actual = state
        .qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
            &shared_gate,
            &shared_up,
            &shared_down,
            &shared_route,
            &sparse_gate,
            &sparse_up,
            &sparse_down,
            &expert_ids,
            &route,
            &token_ids,
            token_count,
            12,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .expect("cached Q6-down selected-base full-layer resident execution")
        .expect("cached Q6-down selected-base full-layer resident path admitted");

    assert_close_rows_abs_rel(
        "cached Q6-down selected-base full-layer resident sparse MoE",
        &actual,
        &expected,
        0.5,
        0.03,
    );
    let stats = state
        .last_qwen35_selected_sparse_boundary_stats
        .expect("Q6-down selected-base execution boundary stats");
    assert_eq!(stats.selected_upload_calls, 0);
    assert_eq!(stats.selected_upload_bytes, 0);
}

#[test]
fn cuda_qwen35_cached_q4_shared_q6_down_sparse_by_token_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let prev_enabled = std::env::var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE").ok();
    let prev_prefill_f32 = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
    let prev_short_window = std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ").ok();
    let prev_q4_cache = std::env::var("RNB_CUDA_Q4_F32_CACHE_MB").ok();
    let prev_q6_cache = std::env::var("RNB_CUDA_Q6_F32_CACHE_MB").ok();
    std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    std::env::set_var("RNB_CUDA_Q6_F32_CACHE_MB", "64");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 91)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 97)
        .pop()
        .unwrap();
    let shared_down = make_test_q6k_weights(1, n_embd, n_ff / 256, 101)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 103);
    let sparse_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 107);
    let sparse_down = make_test_q6k_weights(1, n_embd, n_ff / 256, 109);
    let gate_refs = vec![sparse_gate[0].as_slice(), sparse_gate[0].as_slice()];
    let up_refs = vec![sparse_up[0].as_slice(), sparse_up[0].as_slice()];
    let down_refs = vec![sparse_down[0].as_slice(), sparse_down[0].as_slice()];
    let expert_ids = vec![0u32, 0u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.5f32, 0.75f32];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q6k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let sparse = cpu_qwen35_sparse_q4k_q6k_reference(
            &[sparse_gate[0].as_slice()],
            &[sparse_up[0].as_slice()],
            &[sparse_down[0].as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &gate_refs,
        &up_refs,
        &down_refs,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        14,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => panic!("cached Q4 shared Q6 down MoE path was not admitted"),
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping cached Q4 shared Q6 down MoE test: {err}");
            restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
            restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
            restore_env_var(
                "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
                prev_short_window,
            );
            restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);
            restore_env_var("RNB_CUDA_Q6_F32_CACHE_MB", prev_q6_cache);
            return;
        }
        Err(err) => panic!("cached Q4 shared Q6 down MoE failed: {err}"),
    };

    restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
    restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
    restore_env_var(
        "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
        prev_short_window,
    );
    restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);
    restore_env_var("RNB_CUDA_Q6_F32_CACHE_MB", prev_q6_cache);

    assert_close_rows_abs_rel(
        "cached Q4 shared Q6 down sparse MoE",
        &actual,
        &expected,
        0.5,
        0.03,
    );
}

#[test]
fn cuda_qwen35_cached_q4_shared_sparse_full_layer_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let prev_enabled = std::env::var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE").ok();
    let prev_full_shared = std::env::var("RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE").ok();
    let prev_prefill_f32 = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
    let prev_short_window = std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ").ok();
    let prev_q4_cache = std::env::var("RNB_CUDA_Q4_F32_CACHE_MB").ok();
    let prev_full_layer = std::env::var("RNB_CUDA_PREFILL_MOE_FULL_LAYER").ok();
    let prev_full_layer_retry = std::env::var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY").ok();
    let prev_device_slot_ptrs = std::env::var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS").ok();
    std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE", "1");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "0");
    std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER", "1");
    std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", "1");
    std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 173)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 179)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 181)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 191).concat();
    let sparse_up = make_test_q4k_weights(2, n_ff, n_embd / 256, 193).concat();
    let sparse_down = make_test_q4k_weights(2, n_embd, n_ff / 256, 197).concat();
    let expert_ids = vec![0u32, 1u32];
    let route = vec![0.625f32, 0.375f32];
    let shared_route = vec![0.5f32, 0.75f32];
    let token_ids = vec![0u32, 1u32];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();

    let sparse_gate_split = sparse_gate
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_up_split = sparse_up
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_down_split = sparse_down
        .chunks_exact(n_embd * (n_ff / 256) * 144)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let expert = expert_ids[token] as usize;
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &[sparse_gate_split[expert]],
            &[sparse_up_split[expert]],
            &[sparse_down_split[expert]],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        12,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => panic!("cached Q4 shared full-layer MoE path was not admitted"),
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping cached Q4 shared full-layer MoE test: {err}");
            restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
            restore_env_var(
                "RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE",
                prev_full_shared,
            );
            restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
            restore_env_var(
                "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
                prev_short_window,
            );
            restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);
            restore_env_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER", prev_full_layer);
            restore_env_var(
                "RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY",
                prev_full_layer_retry,
            );
            restore_env_var(
                "RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS",
                prev_device_slot_ptrs,
            );
            return;
        }
        Err(err) => panic!("cached Q4 shared full-layer MoE failed: {err}"),
    };

    restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
    restore_env_var(
        "RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE",
        prev_full_shared,
    );
    restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
    restore_env_var(
        "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
        prev_short_window,
    );
    restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);
    restore_env_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER", prev_full_layer);
    restore_env_var(
        "RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY",
        prev_full_layer_retry,
    );
    restore_env_var(
        "RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS",
        prev_device_slot_ptrs,
    );

    assert_close_rows_abs_rel(
        "cached Q4 shared full-layer sparse MoE",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_cached_q4_shared_sparse_full_layer_q6_run_tiled4_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _shared_cache = EnvVarGuard::set("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    let _full_shared = EnvVarGuard::set("RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE", "1");
    let _prefill_f32 = EnvVarGuard::set("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    let _short_window = EnvVarGuard::set("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "0");
    let _q4_cache = EnvVarGuard::set("RNB_CUDA_Q4_F32_CACHE_MB", "64");
    let _q6_cache = EnvVarGuard::set("RNB_CUDA_Q6_F32_CACHE_MB", "64");
    let _full_layer = EnvVarGuard::set("RNB_CUDA_PREFILL_MOE_FULL_LAYER", "1");
    let _full_layer_retry = EnvVarGuard::set("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", "1");
    let _device_slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");
    let _run_tiled4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", "1");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "0");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _full4_fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 3usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 271)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 277)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 281)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 283).concat();
    let sparse_up = make_test_q4k_weights(2, n_ff, n_embd / 256, 293).concat();
    let sparse_down = make_test_q6k_weights(2, n_embd, n_ff / 256, 307).concat();
    let expert_ids = vec![0u32, 0, 0, 0, 0, 1, 1];
    let route = vec![0.10f32, 0.25, 0.15, 0.30, 0.20, 0.55, 0.45];
    let shared_route = vec![0.5f32, 0.75, 0.625];
    let token_ids = vec![0u32, 2, 1, 0, 2, 1, 0];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 59.0) - 29.0) * 0.00634765625)
        .collect::<Vec<_>>();

    let sparse_gate_split = sparse_gate
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_up_split = sparse_up
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_down_split = sparse_down
        .chunks_exact(n_embd * (n_ff / 256) * 210)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let slot_indices = token_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &token_id)| (token_id as usize == token).then_some(slot))
            .collect::<Vec<_>>();
        let token_gate = slot_indices
            .iter()
            .map(|&slot| sparse_gate_split[expert_ids[slot] as usize])
            .collect::<Vec<_>>();
        let token_up = slot_indices
            .iter()
            .map(|&slot| sparse_up_split[expert_ids[slot] as usize])
            .collect::<Vec<_>>();
        let token_down = slot_indices
            .iter()
            .map(|&slot| sparse_down_split[expert_ids[slot] as usize])
            .collect::<Vec<_>>();
        let token_route = slot_indices
            .iter()
            .map(|&slot| route[slot])
            .collect::<Vec<_>>();
        let sparse = cpu_qwen35_sparse_q4k_q6k_reference(
            &token_gate,
            &token_up,
            &token_down,
            &token_route,
            n_ff,
            n_embd,
            token_input,
        );
        for row in 0..n_embd {
            expected[token * n_embd + row] = sparse[row] + shared[row] * shared_route[token];
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &expert_ids,
        &route,
        &token_ids,
        token_count,
        12,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => panic!("cached Q4 shared Q6 run-tiled4 full-layer MoE path was not admitted"),
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping cached Q4 shared Q6 run-tiled4 full-layer MoE test: {err}");
            return;
        }
        Err(err) => panic!("cached Q4 shared Q6 run-tiled4 full-layer MoE failed: {err}"),
    };

    assert_close_rows_abs_rel(
        "cached Q4 shared Q6 run-tiled4 full-layer sparse MoE",
        &actual,
        &expected,
        0.5,
        0.03,
    );
}

#[test]
fn cuda_qwen35_cached_q4_shared_sparse_device_topk_selected_base_stream_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM", "1");
    let _stream_batches = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM_BATCHES", "2");
    let prev_enabled = std::env::var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE").ok();
    let prev_prefill_f32 = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
    let prev_short_window = std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ").ok();
    let prev_q4_cache = std::env::var("RNB_CUDA_Q4_F32_CACHE_MB").ok();
    std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
    std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "0");
    std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "5");
    std::env::set_var("RNB_CUDA_Q4_F32_CACHE_MB", "64");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let shared_gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 211)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, n_embd / 256, 223)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, n_embd, n_ff / 256, 227)
        .pop()
        .unwrap();
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 229).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 233).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 239).concat();
    let sparse_gate_split = sparse_gate
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_up_split = sparse_up
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_down_split = sparse_down
        .chunks_exact(n_embd * (n_ff / 256) * 144)
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| {
            let base = ((i as f32 % 31.0) - 15.0) * 0.015625;
            if i < n_embd {
                base.abs()
            } else {
                -base.abs()
            }
        })
        .collect::<Vec<_>>();
    let mut router = vec![0.0f32; n_expert * n_embd];
    router[0] = 1.0;
    router[n_embd] = -1.0;
    let shared_route = vec![0.5f32, 0.75f32];

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, n_embd / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, n_embd / 256, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_q4k_gemv_rows(&shared_down, n_embd, n_ff / 256, &shared_hidden);
        let logits = [token_input[0], -token_input[0]];
        let max_logit = logits[0].max(logits[1]);
        let exp0 = (logits[0] - max_logit).exp();
        let exp1 = (logits[1] - max_logit).exp();
        let sum = exp0 + exp1;
        let route = [exp0 / sum, exp1 / sum];
        for row in 0..n_embd {
            expected[token * n_embd + row] = shared[row] * shared_route[token];
        }
        for expert in 0..n_expert {
            let sparse = cpu_qwen35_sparse_q4k_reference(
                &[sparse_gate_split[expert]],
                &[sparse_up_split[expert]],
                &[sparse_down_split[expert]],
                &[route[expert]],
                n_ff,
                n_embd,
                token_input,
            );
            for row in 0..n_embd {
                expected[token * n_embd + row] += sparse[row];
            }
        }
    }

    let actual = match qwen35_prefill_moe_q4_shared_sparse_device_topk_selected_base_by_token_cached(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &router,
        n_expert,
        n_embd,
        &input,
        token_count,
        n_expert_used,
        12,
        12,
        n_ff,
        n_embd,
    ) {
        Ok(Some(actual)) => actual,
        Ok(None) => {
            panic!("cached Q4 shared device-topk selected-base MoE path was not admitted")
        }
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping cached Q4 shared device-topk selected-base MoE test: {err}");
            restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
            restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
            restore_env_var(
                "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
                prev_short_window,
            );
            restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);
            return;
        }
        Err(err) => panic!("cached Q4 shared device-topk selected-base MoE failed: {err}"),
    };

    restore_env_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", prev_enabled);
    restore_env_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_prefill_f32);
    restore_env_var(
        "RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ",
        prev_short_window,
    );
    restore_env_var("RNB_CUDA_Q4_F32_CACHE_MB", prev_q4_cache);

    assert_close_rows_abs_rel(
        "cached Q4 shared device-topk selected-base sparse MoE",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_f32_selected_base_mixed_admission_promotes_resident_pages() {
    let _guard = runtime_test_lock();
    reset_default_cuda_compute_for_test();
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    let _admission = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    let _future_hits = EnvVarGuard::set(
        "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS",
        "2",
    );
    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let n_expert = 2usize;
    let shared_gate = (0..n_ff * n_embd)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.003)
        .collect::<Vec<_>>();
    let shared_up = (0..n_ff * n_embd)
        .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
        .collect::<Vec<_>>();
    let shared_down = (0..n_embd * n_ff)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.0025)
        .collect::<Vec<_>>();
    let shared_route = vec![0.5f32, 0.75f32];
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let expert_ids = [0u32, 1, 0, 1];
    let route_weights = [0.6f32, 0.4, 0.55, 0.45];
    let token_ids = [0u32, 0, 1, 1];
    let input = (0..token_count * n_embd)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.01)
        .collect::<Vec<_>>();

    let output = match qwen35_prefill_moe_f32_shared_sparse_selected_base_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_route,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &expert_ids,
        &route_weights,
        &token_ids,
        token_count,
        12,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(output) => output,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping f32 selected-base mixed admission CUDA test: {err}");
            return;
        }
        Err(err) => panic!("f32 selected-base mixed admission failed: {err}"),
    };
    assert_eq!(output.len(), token_count * n_embd);

    let guard = lock_default_cuda_compute_for_test().expect("default CUDA state lock");
    let state = guard.as_ref().expect("default CUDA state initialized");
    assert_eq!(state.resident_q4k.len(), n_expert * 3);
}

#[test]
fn cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let shared_gate = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.00075)
        .collect::<Vec<_>>();
    let shared_up = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.0005)
        .collect::<Vec<_>>();
    let shared_down = (0..n_embd * n_ff)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.000625)
        .collect::<Vec<_>>();
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let sparse_gate_split = sparse_gate
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_up_split = sparse_up
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_down_split = sparse_down
        .chunks_exact(n_embd * (n_ff / 256) * 144)
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| {
            let base = ((i as f32 % 31.0) - 15.0) * 0.015625;
            if i < n_embd {
                base.abs()
            } else {
                -base.abs()
            }
        })
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.002)
        .collect::<Vec<_>>();
    let mut router = vec![0.0f32; n_expert * n_embd];
    router[0] = 1.0;
    router[n_embd] = -1.0;
    let mut shared_input_scale = vec![0.0f32; n_embd];
    shared_input_scale[0] = 1.0;

    let mut expected = residual.clone();
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_f32_gemv_rows(&shared_gate, n_ff, n_embd, token_input);
        let shared_up_out = cpu_f32_gemv_rows(&shared_up, n_ff, n_embd, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_f32_gemv_rows(&shared_down, n_embd, n_ff, &shared_hidden);
        let shared_route = 1.0 / (1.0 + (-token_input[0]).exp());
        let logits = [token_input[0], -token_input[0]];
        let max_logit = logits[0].max(logits[1]);
        let exp0 = (logits[0] - max_logit).exp();
        let exp1 = (logits[1] - max_logit).exp();
        let sum = exp0 + exp1;
        let routes = [exp0 / sum, exp1 / sum];
        for expert in 0..n_expert {
            let sparse = cpu_qwen35_sparse_q4k_reference(
                &[sparse_gate_split[expert]],
                &[sparse_up_split[expert]],
                &[sparse_down_split[expert]],
                &[routes[expert]],
                n_ff,
                n_embd,
                token_input,
            );
            for row in 0..n_embd {
                expected[token * n_embd + row] += sparse[row];
            }
        }
        for row in 0..n_embd {
            expected[token * n_embd + row] += shared[row] * shared_route;
        }
    }

    let input_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Normalized,
    );
    let residual_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Hidden,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Qwen input");
    let residual_id =
        upload_device_tensor_f32(residual_desc, &residual).expect("upload Qwen residual");

    let output = match qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_input_scale,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &router,
        n_expert,
        n_embd,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        token_count,
        n_expert_used,
        12,
        n_ff,
        n_embd,
    ) {
        Ok(Some(output)) => output,
        Ok(None) => panic!("Qwen35 device-input MoE path was not admitted"),
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping Qwen35 device-input MoE test: {err}");
            assert!(release_device_tensor(input_id).expect("release Qwen input"));
            assert!(release_device_tensor(residual_id).expect("release Qwen residual"));
            return;
        }
        Err(err) => panic!("Qwen35 device-input MoE failed: {err}"),
    };

    let actual =
        download_device_tensor_f32(output.output_id).expect("download Qwen device-input output");
    assert!(release_device_tensor(input_id).expect("release Qwen input"));
    assert!(release_device_tensor(residual_id).expect("release Qwen residual"));
    assert!(release_device_tensor(output.output_id).expect("release Qwen output"));
    assert_eq!(
        output.output_desc,
        DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::MoeOutput
        )
    );
    assert_close_rows_abs_rel(
        "Qwen35 device-input residual MoE",
        &actual,
        &expected,
        0.25,
        0.03,
    );
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_device_route_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _device_route = EnvVarGuard::set("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_device_route_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _device_route = EnvVarGuard::set("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_q4_down_group4_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _q4_group4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "1");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_copy_stream_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_copy_stream_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_pinned_staging_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    let _pinned = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_pinned_staging_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    let _pinned = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_overlap_staging_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    let _overlap = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_overlap_staging_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _copy_stream = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM", "1");
    let _overlap = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_mixed_resident_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_mixed_resident_device_slot_ptrs_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    let _mixed_device =
        EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_mixed_resident_admission_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    let _admission = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_mixed_resident_device_slot_ptrs_admission_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    let _mixed_device =
        EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS", "1");
    let _admission = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_stage_compute_fused_matches_cpu_reference() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_SPARSE_FUSED_BOUNDARY", "1");
    cuda_qwen35_device_input_moe_adds_residual_matches_cpu_reference();
}

#[test]
fn cuda_qwen35_device_input_moe_direct_sparse_stage_compute_fused_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _direct = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE", "1");
    let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_SPARSE_FUSED_BOUNDARY", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_fused_reuse_prepare_failure_preserves_residual() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_SPARSE_FUSED_BOUNDARY", "1");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 1usize;
    let n_expert = 2usize;
    let shared_gate = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.00075)
        .collect::<Vec<_>>();
    let shared_up = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.0005)
        .collect::<Vec<_>>();
    let shared_down = (0..n_embd * n_ff)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.000625)
        .collect::<Vec<_>>();
    let mut shared_input_scale = vec![0.0f32; n_embd];
    shared_input_scale[0] = 1.0;
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.002)
        .collect::<Vec<_>>();
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let invalid_expert_ids = [n_expert as u32];
    let route_weights = [1.0f32];
    let token_ids = [0u32];
    let empty_sparse_slots: [&[u8]; 0] = [];
    let input_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Normalized,
    );
    let residual_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Hidden,
    );

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping Qwen35 fused reuse failure atomicity test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    let input_id = state
        .upload_device_tensor_f32(input_desc, &input)
        .expect("upload Qwen input");
    let residual_id = state
        .upload_device_tensor_f32(residual_desc, &residual)
        .expect("upload Qwen residual");

    let err = state
        .qwen35_prefill_moe_f32_shared_sparse_by_token_device_input_reuse_residual(
            &shared_gate,
            &shared_up,
            &shared_down,
            &shared_input_scale,
            &empty_sparse_slots,
            &empty_sparse_slots,
            &empty_sparse_slots,
            &invalid_expert_ids,
            &route_weights,
            &token_ids,
            token_count,
            12,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            None,
            Some(DeferredQwen35SelectedBaseSparse {
                gate_all: &sparse_gate,
                up_all: &sparse_up,
                down_all: &sparse_down,
                expert_ids: &invalid_expert_ids,
                down_quant: 12,
                n_ff,
                n_embd,
            }),
        )
        .expect_err("invalid deferred selected-base expert must fail before residual reuse");
    assert!(err.contains("expert id out of range"), "{err}");

    let actual = state
        .download_device_tensor_f32(residual_id)
        .expect("download unchanged residual after failed fused reuse");
    assert_eq!(actual, residual);
    assert!(state
        .release_device_tensor(input_id)
        .expect("release Qwen input"));
    assert!(state
        .release_device_tensor(residual_id)
        .expect("release unchanged Qwen residual"));
}

#[test]
fn cuda_qwen35_device_input_moe_mixed_resident_reuses_residual_output() {
    let _slot_ptrs = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS", "1");
    let _mixed = EnvVarGuard::set("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT", "1");
    cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier();
}

#[test]
fn cuda_qwen35_device_input_moe_can_reuse_residual_output_carrier() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let n_ff = 256usize;
    let n_embd = 256usize;
    let token_count = 2usize;
    let n_expert = 2usize;
    let n_expert_used = 2usize;
    let shared_gate = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.00075)
        .collect::<Vec<_>>();
    let shared_up = (0..n_ff * n_embd)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.0005)
        .collect::<Vec<_>>();
    let shared_down = (0..n_embd * n_ff)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.000625)
        .collect::<Vec<_>>();
    let sparse_gate = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 331).concat();
    let sparse_up = make_test_q4k_weights(n_expert, n_ff, n_embd / 256, 337).concat();
    let sparse_down = make_test_q4k_weights(n_expert, n_embd, n_ff / 256, 347).concat();
    let sparse_gate_split = sparse_gate
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_up_split = sparse_up
        .chunks_exact(n_ff * (n_embd / 256) * 144)
        .collect::<Vec<_>>();
    let sparse_down_split = sparse_down
        .chunks_exact(n_embd * (n_ff / 256) * 144)
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| {
            let base = ((i as f32 % 31.0) - 15.0) * 0.015625;
            if i < n_embd {
                base.abs()
            } else {
                -base.abs()
            }
        })
        .collect::<Vec<_>>();
    let residual = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.002)
        .collect::<Vec<_>>();
    let mut router = vec![0.0f32; n_expert * n_embd];
    router[0] = 1.0;
    router[n_embd] = -1.0;
    let mut shared_input_scale = vec![0.0f32; n_embd];
    shared_input_scale[0] = 1.0;

    let mut expected = residual.clone();
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let shared_gate_out = cpu_f32_gemv_rows(&shared_gate, n_ff, n_embd, token_input);
        let shared_up_out = cpu_f32_gemv_rows(&shared_up, n_ff, n_embd, token_input);
        let mut shared_hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            shared_hidden[i] =
                (shared_gate_out[i] / (1.0 + (-shared_gate_out[i]).exp())) * shared_up_out[i];
        }
        let shared = cpu_f32_gemv_rows(&shared_down, n_embd, n_ff, &shared_hidden);
        let shared_route = 1.0 / (1.0 + (-token_input[0]).exp());
        let logits = [token_input[0], -token_input[0]];
        let max_logit = logits[0].max(logits[1]);
        let exp0 = (logits[0] - max_logit).exp();
        let exp1 = (logits[1] - max_logit).exp();
        let sum = exp0 + exp1;
        let routes = [exp0 / sum, exp1 / sum];
        for expert in 0..n_expert {
            let sparse = cpu_qwen35_sparse_q4k_reference(
                &[sparse_gate_split[expert]],
                &[sparse_up_split[expert]],
                &[sparse_down_split[expert]],
                &[routes[expert]],
                n_ff,
                n_embd,
                token_input,
            );
            for row in 0..n_embd {
                expected[token * n_embd + row] += sparse[row];
            }
        }
        for row in 0..n_embd {
            expected[token * n_embd + row] += shared[row] * shared_route;
        }
    }

    let input_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Normalized,
    );
    let residual_desc = DeviceTensorDesc::new(
        token_count,
        n_embd,
        ScalarType::F32,
        DeviceTensorRole::Hidden,
    );
    let input_id = upload_device_tensor_f32(input_desc, &input).expect("upload Qwen input");
    let residual_id =
        upload_device_tensor_f32(residual_desc, &residual).expect("upload Qwen residual");

    let output = match qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual(
        &shared_gate,
        &shared_up,
        &shared_down,
        &shared_input_scale,
        &sparse_gate,
        &sparse_up,
        &sparse_down,
        &router,
        n_expert,
        n_embd,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        token_count,
        n_expert_used,
        12,
        n_ff,
        n_embd,
    ) {
        Ok(Some(output)) => output,
        Ok(None) => panic!("Qwen35 device-input MoE reuse path was not admitted"),
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping Qwen35 device-input MoE residual reuse test: {err}");
            assert!(release_device_tensor(input_id).expect("release Qwen input"));
            assert!(release_device_tensor(residual_id).expect("release Qwen residual"));
            return;
        }
        Err(err) => panic!("Qwen35 device-input MoE residual reuse failed: {err}"),
    };

    assert_eq!(output.output_id, residual_id);
    assert_eq!(
        output.output_desc,
        DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::MoeOutput
        )
    );
    let actual =
        download_device_tensor_f32(output.output_id).expect("download reused Qwen MoE output");
    assert!(release_device_tensor(input_id).expect("release Qwen input"));
    assert!(release_device_tensor(output.output_id).expect("release reused Qwen output"));
    assert!(!release_device_tensor(residual_id).expect("residual carrier already consumed"));
    assert_close_rows_abs_rel(
        "Qwen35 device-input residual reuse MoE",
        &actual,
        &expected,
        0.25,
        0.03,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_row8_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    unsafe {
        std::env::set_var("RNB_CUDA_GROUP4_DOWN_ROW8", "1");
    }

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 2usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 11);
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 29);
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 53);
    let gate_refs = vec![gate[0].as_slice(), gate[0].as_slice()];
    let up_refs = vec![up[0].as_slice(), up[0].as_slice()];
    let down_refs = vec![down[0].as_slice(), down[0].as_slice()];
    let route = vec![0.5625f32, 0.4375f32];
    let token_ids = vec![0u32, 1u32];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 37.0) - 18.0) * 0.015625;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_q6k_reference(
            &[gate[0].as_slice()],
            &[up[0].as_slice()],
            &[down[0].as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA sparse by-token q6 grouped test: {err}");
            unsafe {
                std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
            }
            return;
        }
        Err(err) => panic!("CUDA sparse by-token q6 grouped failed: {err}"),
    };

    unsafe {
        std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
    }

    for row_idx in 0..token_count * n_embd {
        let diff = (actual[row_idx] - expected[row_idx]).abs();
        assert!(
                diff < 0.2,
                "CUDA sparse by-token q6 grouped row {row_idx} mismatch: expected {}, actual {}, diff {}",
                expected[row_idx],
                actual[row_idx],
                diff
            );
    }
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_group2_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "1");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 113);
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 127);
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 131);
    let gate_refs = vec![gate[0].as_slice(), gate[0].as_slice(), gate[0].as_slice()];
    let up_refs = vec![up[0].as_slice(), up[0].as_slice(), up[0].as_slice()];
    let down_refs = vec![down[0].as_slice(), down[0].as_slice(), down[0].as_slice()];
    let route = vec![0.5625f32, 0.3125f32, 0.125f32];
    let token_ids = vec![0u32, 1u32, 2u32];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 41.0) - 20.0) * 0.01171875;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_q6k_reference(
            &[gate[0].as_slice()],
            &[up[0].as_slice()],
            &[down[0].as_slice()],
            &[route[token]],
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA sparse by-token q6 group2 test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse by-token q6 group2 failed: {err}"),
    };

    assert_close_rows_abs_rel("Qwen35 q6 group2 down", &actual, &expected, 0.2, 0.02);
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_token_major_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "1");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 149);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 151);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 157);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[0].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[0].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[0].as_slice(),
    ];
    let route = vec![0.25f32, 0.5f32, 0.75f32, 1.0f32];
    let token_ids = vec![0u32, 1u32, 0u32, 2u32];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 43.0) - 21.0) * 0.009765625;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let slot_indices = token_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &token_id)| (token_id as usize == token).then_some(slot))
            .collect::<Vec<_>>();
        let token_gate = slot_indices
            .iter()
            .map(|&slot| gate_refs[slot])
            .collect::<Vec<_>>();
        let token_up = slot_indices
            .iter()
            .map(|&slot| up_refs[slot])
            .collect::<Vec<_>>();
        let token_down = slot_indices
            .iter()
            .map(|&slot| down_refs[slot])
            .collect::<Vec<_>>();
        let token_route = slot_indices
            .iter()
            .map(|&slot| route[slot])
            .collect::<Vec<_>>();
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_q6k_reference(
            &token_gate,
            &token_up,
            &token_down,
            &token_route,
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA sparse by-token q6 token-major test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse by-token q6 token-major failed: {err}"),
    };

    assert_close_rows_abs_rel("Qwen35 q6 token-major down", &actual, &expected, 0.5, 0.03);
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_pack4_f32_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _pack4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", "1");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _run_tiled4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", "0");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 181);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 191);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 193);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
    ];
    let route = vec![0.125f32, 0.375, 0.25, 0.50, 0.75];
    let token_ids = vec![0u32, 2, 1, 0, 2];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 47.0) - 23.0) * 0.0078125;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let slot_indices = token_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &token_id)| (token_id as usize == token).then_some(slot))
            .collect::<Vec<_>>();
        let token_gate = slot_indices
            .iter()
            .map(|&slot| gate_refs[slot])
            .collect::<Vec<_>>();
        let token_up = slot_indices
            .iter()
            .map(|&slot| up_refs[slot])
            .collect::<Vec<_>>();
        let token_down = slot_indices
            .iter()
            .map(|&slot| down_refs[slot])
            .collect::<Vec<_>>();
        let token_route = slot_indices
            .iter()
            .map(|&slot| route[slot])
            .collect::<Vec<_>>();
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_q6k_reference(
            &token_gate,
            &token_up,
            &token_down,
            &token_route,
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA sparse by-token q6 pack4-f32 down test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse by-token q6 pack4-f32 down failed: {err}"),
    };

    assert_close_rows_abs_rel("Qwen35 q6 pack4-f32 down", &actual, &expected, 0.5, 0.03);
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_pack4_gate_up_silu_fused_matches_pack4_path() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _pack4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", "1");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _run_tiled4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", "0");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 467);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 479);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 487);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
    ];
    let route = vec![0.125f32, 0.375, 0.25, 0.50, 0.75];
    let token_ids = vec![0u32, 2, 1, 0, 2];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 61.0) - 30.0) * 0.0068359375)
        .collect::<Vec<_>>();

    let baseline = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q6 pack4 gate/up fused path test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q6 pack4 baseline path failed: {err}"),
        }
    };
    let fused = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 gate/up fused path failed: {err}"))
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &baseline);
    assert!(
        max_abs <= 1.0e-4,
        "Q6 pack4 gate/up fused path changed baseline output: idx={max_idx} baseline={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        baseline[max_idx],
        fused[max_idx]
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_pack4_gate_up_q8dot_group16_matches_group8() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _pack4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", "1");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _run_tiled4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", "0");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8_down = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _group8_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP8_GATE_UP_WARP4", "1");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 8usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 503)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 509)
        .pop()
        .unwrap();
    let down = make_test_q6k_weights(1, n_embd, n_ff / 256, 521)
        .pop()
        .unwrap();
    let gate_refs = vec![gate.as_slice(); 32];
    let up_refs = vec![up.as_slice(); 32];
    let down_refs = vec![down.as_slice(); 32];
    let route = (1..=32)
        .map(|value| value as f32 / 32.0)
        .collect::<Vec<_>>();
    let token_ids = (0..32)
        .map(|slot| (slot % token_count) as u32)
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 71.0) - 35.0) * 0.005859375)
        .collect::<Vec<_>>();

    let baseline = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "0");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");
        let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q6 pack4 group8 len8 fused path test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q6 pack4 group8 len8 baseline path failed: {err}"),
        }
    };
    let fused = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");
        let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 group8 len8 gate/up fused path failed: {err}"))
    };
    let q8dot_group8 = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "1");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");
        let _mmq_group16 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "0");
        let _mmq_group32 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "0");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 group8 Q8-dot gate/up path failed: {err}"))
    };
    let q8dot_group16 = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "1");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "1");
        let _mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "1");
        let _mmq_group16 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "1");
        let _mmq_group32 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "0");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 group16 Q8-dot gate/up path failed: {err}"))
    };
    let q8dot_group32 = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "1");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");
        let _mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "1");
        let _mmq_group16 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "1");
        let _mmq_group32 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "1");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 group32 Q8-dot gate/up path failed: {err}"))
    };
    let q8dot_mmq_opt_out = {
        let _fused = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", "1");
        let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "1");
        let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "1");
        let _mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "0");
        let _mmq_group32 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "0");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4 Q8-dot MMQ opt-out path failed: {err}"))
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&fused, &baseline);
    assert!(
        max_abs <= 1.0e-4,
        "Q6 pack4 group8 len16 gate/up fused path changed baseline output: idx={max_idx} baseline={:.9} fused={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        baseline[max_idx],
        fused[max_idx]
    );
    let output_max = fused.iter().fold(0.0f32, |max, value| max.max(value.abs()));
    let (q8_max_idx, q8_max_abs, q8_max_rel) = max_abs_rel(&q8dot_group8, &fused);
    assert!(
        q8_max_abs <= output_max * 0.003,
        "Q6 pack4 group8 Q8-dot gate/up changed output beyond 0.3% of output scale: idx={q8_max_idx} abs={q8_max_abs:.6} rel={q8_max_rel:.6} output_max={output_max:.6}"
    );
    let (group16_idx, group16_abs, group16_rel) = max_abs_rel(&q8dot_group16, &q8dot_group8);
    assert!(
        group16_abs <= 1.0e-4,
        "Q6 pack4 group16 Q8-dot changed group8 output: idx={group16_idx} abs={group16_abs:.6} rel={group16_rel:.6}"
    );
    let (group32_idx, group32_abs, group32_rel) = max_abs_rel(&q8dot_group32, &q8dot_group16);
    assert!(
        group32_abs <= 1.0e-4,
        "Q6 pack4 group32 Q8-dot changed group16 output: idx={group32_idx} abs={group32_abs:.6} rel={group32_rel:.6}"
    );
    let (opt_out_idx, opt_out_abs, opt_out_rel) = max_abs_rel(&q8dot_mmq_opt_out, &q8dot_group8);
    assert!(
        opt_out_abs <= output_max * 0.003,
        "Q6 pack4 Q8-dot MMQ opt-out changed group8 output beyond 0.3% of output scale: idx={opt_out_idx} abs={opt_out_abs:.6} rel={opt_out_rel:.6} output_max={output_max:.6}"
    );
}

#[test]
fn cuda_qwen35_sparse_experts_q5_down_mmq_is_deterministic_and_matches_group4() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _group2_down = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8_down = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _group4_down = EnvVarGuard::set("RNB_CUDA_GROUP4_DOWN", "1");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
    let n_embd = 256usize;
    let n_ff = 256usize;
    let token_count = 32usize;
    let gate = make_test_q4k_weights(1, n_ff, n_embd / 256, 547)
        .pop()
        .unwrap();
    let up = make_test_q4k_weights(1, n_ff, n_embd / 256, 557)
        .pop()
        .unwrap();
    let down = make_test_q5k_weights(n_embd, n_ff / 256, 563);
    let slots = token_count * 2;
    let gate_refs = vec![gate.as_slice(); slots];
    let up_refs = vec![up.as_slice(); slots];
    let down_refs = vec![down.as_slice(); slots];
    let token_ids = (0..slots)
        .map(|slot| ((slot * 17 + 7) % token_count) as u32)
        .collect::<Vec<_>>();
    let route = (1..=slots)
        .map(|value| value as f32 / (slots + 1) as f32)
        .collect::<Vec<_>>();
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 67.0) - 33.0) * 0.00390625)
        .collect::<Vec<_>>();

    let baseline = {
        let _q5_mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            13,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA Q5_K deterministic down MMQ test: {err}");
                return;
            }
            Err(err) => panic!("CUDA Q5_K group4 baseline failed: {err}"),
        }
    };
    let run_mmq = || {
        let _q5_mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", "1");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            13,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA Q5_K deterministic down MMQ failed: {err}"))
    };
    let mmq_a = run_mmq();
    let mmq_b = run_mmq();
    let run_q8_gate_up = |handoff: &str| {
        let _q5_mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", "1");
        let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "1");
        let _gate_up_mmq = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "1");
        let _gate_up_group16 =
            EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "1");
        let _gate_up_group32 =
            EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "1");
        let _handoff = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF", handoff);
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            13,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA Q5_K Q8 gate/up handoff path failed: {err}"))
    };
    let separate_quantize = run_q8_gate_up("0");
    let handoff_a = run_q8_gate_up("1");
    let handoff_b = run_q8_gate_up("1");

    assert_eq!(
        handoff_a, handoff_b,
        "Q4_K gate/up Q8 handoff output changed across identical runs"
    );
    assert_eq!(
        handoff_a, separate_quantize,
        "Q4_K gate/up Q8 handoff changed the separate-quantize Q5_K output"
    );

    assert_eq!(
        mmq_a, mmq_b,
        "Q5_K down MMQ output changed across identical runs"
    );
    let output_max = baseline
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let (max_idx, max_abs, max_rel) = max_abs_rel(&mmq_a, &baseline);
    assert!(
        max_abs <= output_max * 0.005,
        "Q5_K down MMQ changed group4 output beyond 0.5% of output scale: idx={max_idx} baseline={:.9} mmq={:.9} abs={max_abs:.9} rel={max_rel:.9} output_max={output_max:.9}",
        baseline[max_idx],
        mmq_a[max_idx]
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_pack4_f32_vec4_matches_pack4_path() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");

    let _pack4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", "1");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _run_tiled4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", "0");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8_down = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _group8_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP8_GATE_UP_WARP4", "1");
    let _group16_gate_up = EnvVarGuard::set("RNB_CUDA_GROUP16_GATE_UP_WARP4", "0");

    let n_ff = 512usize;
    let n_embd = 512usize;
    let token_count = 4usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 541);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 547);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 557);
    let expert_ids = [0usize, 0, 0, 0, 1, 1, 1, 1];
    let gate_refs = expert_ids
        .iter()
        .map(|&expert| gate[expert].as_slice())
        .collect::<Vec<_>>();
    let up_refs = expert_ids
        .iter()
        .map(|&expert| up[expert].as_slice())
        .collect::<Vec<_>>();
    let down_refs = expert_ids
        .iter()
        .map(|&expert| down[expert].as_slice())
        .collect::<Vec<_>>();
    let route = vec![0.07f32, 0.17, 0.23, 0.31, 0.43, 0.53, 0.61, 0.73];
    let token_ids = vec![0u32, 1, 2, 3, 0, 2, 1, 3];
    let input = (0..token_count * n_embd)
        .map(|i| ((i as f32 % 83.0) - 41.0) * 0.0048828125)
        .collect::<Vec<_>>();

    let baseline = {
        let _vec4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32_VEC4", "0");
        match qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        ) {
            Ok(actual) => actual,
            Err(err) if cuda_driver_unavailable_for_test(&err) => {
                eprintln!("skipping CUDA q6 pack4-f32 vec4 down test: {err}");
                return;
            }
            Err(err) => panic!("CUDA q6 pack4-f32 baseline path failed: {err}"),
        }
    };
    let vec4 = {
        let _vec4 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32_VEC4", "1");
        qwen35_sparse_experts_by_token(
            &gate_refs,
            &up_refs,
            &down_refs,
            &route,
            &token_ids,
            token_count,
            14,
            n_ff,
            n_embd,
            &input,
        )
        .unwrap_or_else(|err| panic!("CUDA q6 pack4-f32 vec4 down path failed: {err}"))
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&vec4, &baseline);
    assert!(
        max_abs <= 1.0e-4,
        "Q6 pack4-f32 vec4 down path changed baseline output: idx={max_idx} baseline={:.9} vec4={:.9} abs={max_abs:.9} rel={max_rel:.9}",
        baseline[max_idx],
        vec4[max_idx]
    );
}

#[test]
fn qwen35_run_batched_q6_down_cpu_reference_matches_per_token_reference() {
    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 211);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 223);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 227);
    let expert_ids = vec![0u32, 0, 0, 1, 1, 1, 1, 1];
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
    ];
    let route = vec![0.125f32, 0.25, 0.375, 0.20, 0.15, 0.10, 0.30, 0.25];
    let token_ids = vec![0u32, 2, 1, 0, 2, 1, 0, 2];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 53.0) - 26.0) * 0.0068359375;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let slot_indices = token_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &token_id)| (token_id as usize == token).then_some(slot))
            .collect::<Vec<_>>();
        let token_gate = slot_indices
            .iter()
            .map(|&slot| gate_refs[slot])
            .collect::<Vec<_>>();
        let token_up = slot_indices
            .iter()
            .map(|&slot| up_refs[slot])
            .collect::<Vec<_>>();
        let token_down = slot_indices
            .iter()
            .map(|&slot| down_refs[slot])
            .collect::<Vec<_>>();
        let token_route = slot_indices
            .iter()
            .map(|&slot| route[slot])
            .collect::<Vec<_>>();
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = cpu_qwen35_sparse_q4k_q6k_reference(
            &token_gate,
            &token_up,
            &token_down,
            &token_route,
            n_ff,
            n_embd,
            token_input,
        );
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = cpu_qwen35_run_batched_q4k_q6k_by_token_reference(
        &expert_ids,
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
        4,
    )
    .expect("run-batched Q6 down CPU reference");

    assert_close_rows_abs_rel(
        "Qwen35 run-batched q6 down CPU reference",
        &actual,
        &expected,
        1.0e-4,
        1.0e-5,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_run_batched_ref_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _run_batched = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "1");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 231);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 233);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 239);
    let expert_ids = vec![0u32, 0, 0, 0, 0, 1, 1];
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
    ];
    let route = vec![0.10f32, 0.25, 0.15, 0.30, 0.20, 0.55, 0.45];
    let token_ids = vec![0u32, 2, 1, 0, 2, 1, 0];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 59.0) - 29.0) * 0.00634765625;
    }

    let expected = cpu_qwen35_run_batched_q4k_q6k_by_token_reference(
        &expert_ids,
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
        4,
    )
    .expect("run-batched Q6 down CPU reference");

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA sparse by-token q6 run-batched ref test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("compute input ptr");
    let output_len = token_count * n_embd;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let output_dev = state
        .compute_output_ptr(output_bytes)
        .expect("compute output ptr");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("upload input");
    }
    if let Err(err) = state.qwen35_sparse_experts_by_token_to_dev_prepared(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        input_dev,
        output_dev,
        true,
        false,
        Some(&expert_ids),
        None,
    ) {
        if cuda_driver_unavailable_for_test(&err) {
            eprintln!("skipping CUDA sparse by-token q6 run-batched ref test: {err}");
            return;
        }
        panic!("CUDA sparse by-token q6 run-batched ref failed: {err}");
    }

    let mut actual = vec![0.0f32; output_len];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )
            .expect("download output");
    }
    state.stream_synchronize().expect("sync output");

    assert_close_rows_abs_rel(
        "Qwen35 q6 run-batched ref down",
        &actual,
        &expected,
        0.5,
        0.03,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_run_batched8_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _run_batched8 = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", "1");
    let _run_batched_ref = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", "0");
    let _q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", "0");
    let _full4_split = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", "0");
    let _fastpath = EnvVarGuard::set("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", "0");
    let _token_major = EnvVarGuard::set("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", "0");
    let _group2 = EnvVarGuard::set("RNB_CUDA_GROUP2_DOWN_WARP4", "0");
    let _group8 = EnvVarGuard::set("RNB_CUDA_GROUP8_DOWN_WARP4", "0");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 4usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 241);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 251);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 257);
    let expert_ids = vec![0u32, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1];
    let gate_refs = expert_ids
        .iter()
        .map(|&expert| gate[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let up_refs = expert_ids
        .iter()
        .map(|&expert| up[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let down_refs = expert_ids
        .iter()
        .map(|&expert| down[expert as usize].as_slice())
        .collect::<Vec<_>>();
    let route = vec![
        0.07f32, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37, 0.41, 0.43, 0.47, 0.53, 0.59,
        0.61, 0.67,
    ];
    let token_ids = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3, 1, 0, 2, 1, 3];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 61.0) - 30.0) * 0.005859375;
    }

    let expected = cpu_qwen35_run_batched_q4k_q6k_by_token_reference(
        &expert_ids,
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        n_ff,
        n_embd,
        &input,
        8,
    )
    .expect("run-batched8 Q6 down CPU reference");

    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA sparse by-token q6 run-batched8 test: {err}");
            return;
        }
        Err(err) => panic!("open CUDA state failed: {err}"),
    };
    let input_dev = state
        .compute_input_ptr(std::mem::size_of_val(input.as_slice()))
        .expect("compute input ptr");
    let output_len = token_count * n_embd;
    let output_bytes = output_len * std::mem::size_of::<f32>();
    let output_dev = state
        .compute_output_ptr(output_bytes)
        .expect("compute output ptr");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input.as_slice()),
                state.stream,
            )
            .expect("upload input");
    }
    if let Err(err) = state.qwen35_sparse_experts_by_token_to_dev_prepared(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        input_dev,
        output_dev,
        true,
        false,
        Some(&expert_ids),
        None,
    ) {
        if cuda_driver_unavailable_for_test(&err) {
            eprintln!("skipping CUDA sparse by-token q6 run-batched8 test: {err}");
            return;
        }
        panic!("CUDA sparse by-token q6 run-batched8 failed: {err}");
    }

    let mut actual = vec![0.0f32; output_len];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )
            .expect("download output");
    }
    state.stream_synchronize().expect("sync output");

    assert_close_rows_abs_rel("Qwen35 q6 run-batched8 down", &actual, &expected, 0.5, 0.03);
}

#[derive(Clone, Copy)]
enum Qwen35Q6DownExpectation {
    F32,
    Q8Down,
}

fn assert_qwen35_q6_down_mixed_full4_tail_matches_cpu_reference(
    feature_env: &'static str,
    label: &str,
    expectation: Qwen35Q6DownExpectation,
    abs_tolerance: f32,
    rel_tolerance: f32,
) {
    let _guard = runtime_test_lock();
    let _feature = EnvVarGuard::set(feature_env, "1");
    reset_delta_state_cache().expect("reset CUDA runtime cache");
    let _gate_up_q8dot = EnvVarGuard::set("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");

    let n_ff = 256usize;
    let n_embd = 512usize;
    let token_count = 3usize;
    let gate = make_test_q4k_weights(2, n_ff, n_embd / 256, 163);
    let up = make_test_q4k_weights(2, n_ff, n_embd / 256, 167);
    let down = make_test_q6k_weights(2, n_embd, n_ff / 256, 173);
    let gate_refs = vec![
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[0].as_slice(),
        gate[1].as_slice(),
        gate[1].as_slice(),
    ];
    let up_refs = vec![
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[0].as_slice(),
        up[1].as_slice(),
        up[1].as_slice(),
    ];
    let down_refs = vec![
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[0].as_slice(),
        down[1].as_slice(),
        down[1].as_slice(),
    ];
    let route = vec![0.20f32, 0.30f32, 0.15f32, 0.35f32, 0.60f32, 0.40f32];
    let token_ids = vec![0u32, 1u32, 2u32, 0u32, 1u32, 2u32];
    let mut input = vec![0.0f32; token_count * n_embd];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 47.0) - 23.0) * 0.0078125;
    }

    let mut expected = vec![0.0f32; token_count * n_embd];
    for token in 0..token_count {
        let slot_indices = token_ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &token_id)| (token_id as usize == token).then_some(slot))
            .collect::<Vec<_>>();
        let token_gate = slot_indices
            .iter()
            .map(|&slot| gate_refs[slot])
            .collect::<Vec<_>>();
        let token_up = slot_indices
            .iter()
            .map(|&slot| up_refs[slot])
            .collect::<Vec<_>>();
        let token_down = slot_indices
            .iter()
            .map(|&slot| down_refs[slot])
            .collect::<Vec<_>>();
        let token_route = slot_indices
            .iter()
            .map(|&slot| route[slot])
            .collect::<Vec<_>>();
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let token_expected = match expectation {
            Qwen35Q6DownExpectation::F32 => cpu_qwen35_sparse_q4k_q6k_reference(
                &token_gate,
                &token_up,
                &token_down,
                &token_route,
                n_ff,
                n_embd,
                token_input,
            ),
            Qwen35Q6DownExpectation::Q8Down => cpu_qwen35_sparse_q4k_q6k_q8_down_reference(
                &token_gate,
                &token_up,
                &token_down,
                &token_route,
                n_ff,
                n_embd,
                token_input,
            ),
        };
        expected[token * n_embd..(token + 1) * n_embd].copy_from_slice(&token_expected);
    }

    let actual = match qwen35_sparse_experts_by_token(
        &gate_refs,
        &up_refs,
        &down_refs,
        &route,
        &token_ids,
        token_count,
        14,
        n_ff,
        n_embd,
        &input,
    ) {
        Ok(actual) => actual,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping CUDA sparse by-token {label} test: {err}");
            return;
        }
        Err(err) => panic!("CUDA sparse by-token {label} failed: {err}"),
    };

    let (max_idx, max_abs, max_rel) = max_abs_rel(&actual, &expected);
    eprintln!(
        "{label} max diff: row={max_idx} expected={:.6} actual={:.6} abs={max_abs:.6} rel={max_rel:.6}",
        expected[max_idx], actual[max_idx]
    );
    assert_close_rows_abs_rel(label, &actual, &expected, abs_tolerance, rel_tolerance);
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_full4_split_matches_cpu_reference() {
    assert_qwen35_q6_down_mixed_full4_tail_matches_cpu_reference(
        "RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT",
        "Qwen35 q6 full4 split down",
        Qwen35Q6DownExpectation::F32,
        0.5,
        0.03,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_full4_fastpath_matches_cpu_reference() {
    assert_qwen35_q6_down_mixed_full4_tail_matches_cpu_reference(
        "RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH",
        "Qwen35 q6 full4 fastpath down",
        Qwen35Q6DownExpectation::F32,
        0.5,
        0.03,
    );
}

#[test]
fn cuda_qwen35_sparse_experts_by_token_q6_down_q8dot_matches_cpu_reference() {
    assert_qwen35_q6_down_mixed_full4_tail_matches_cpu_reference(
        "RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT",
        "Qwen35 q6 q8dot down",
        Qwen35Q6DownExpectation::Q8Down,
        0.08,
        0.001,
    );
}

#[test]
fn cuda_delta_net_decode_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 23.0) - 11.0) * 0.00390625;
    }
    let mut expected_state = state.clone();
    let mut q = vec![0.0f32; num_heads * head_k_dim];
    let mut k = vec![0.0f32; num_heads * head_k_dim];
    let mut v = vec![0.0f32; num_heads * head_v_dim];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.015625;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.01171875;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.03125;
    }
    let gate = vec![-0.03125f32, -0.0625f32];
    let beta = vec![0.25f32, 0.5f32];

    let expected_first = cpu_delta_net_decode_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let expected_second = cpu_delta_net_decode_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let actual_first = match delta_net_decode(
        &mut state, &q, &k, &v, &gate, &beta, num_heads, head_k_dim, head_v_dim,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta net test: {err}");
            return;
        }
        Err(err) => panic!("CUDA delta net failed: {err}"),
    };
    let actual_second = delta_net_decode(
        &mut state, &q, &k, &v, &gate, &beta, num_heads, head_k_dim, head_v_dim,
    )
    .expect("second CUDA delta net call");

    for (i, (actual, expected)) in actual_first.iter().zip(&expected_first).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 0.001,
                "CUDA delta first output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
    }
    for (i, (actual, expected)) in actual_second.iter().zip(&expected_second).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 0.001,
                "CUDA delta second output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
    }
}

#[test]
fn cuda_delta_state_snapshot_restore_roundtrips_resident_state() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 23.0) - 11.0) * 0.00390625;
    }
    let mut expected_state = state.clone();
    let mut q = vec![0.0f32; num_heads * head_k_dim];
    let mut k = vec![0.0f32; num_heads * head_k_dim];
    let mut v = vec![0.0f32; num_heads * head_v_dim];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.015625;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.01171875;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.03125;
    }
    let gate = vec![-0.03125f32, -0.0625f32];
    let beta = vec![0.25f32, 0.5f32];

    let _ = cpu_delta_net_decode_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let expected_state_after_first = expected_state.clone();

    let actual_first = match delta_net_decode_resident(
        &mut state, &q, &k, &v, &gate, &beta, num_heads, head_k_dim, head_v_dim,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta snapshot test: {err}");
            return;
        }
        Err(err) => panic!("CUDA resident delta net failed: {err}"),
    };
    assert_eq!(actual_first.len(), num_heads * head_v_dim);

    let snapshot = snapshot_delta_state_cache(&mut state)
        .expect("snapshot resident delta state")
        .expect("resident delta state snapshot");

    let _ = delta_net_decode_resident(
        &mut state, &q, &k, &v, &gate, &beta, num_heads, head_k_dim, head_v_dim,
    )
    .expect("second resident delta net call");

    assert!(
        restore_delta_state_cache(&mut state, &snapshot).expect("restore resident delta state"),
        "restore should find resident delta state"
    );
    assert!(
        sync_delta_state_cache(&mut state).expect("sync restored resident delta state"),
        "sync should find resident delta state"
    );
    free_delta_state_snapshot(snapshot).expect("free resident delta state snapshot");

    for (i, (actual, expected)) in state.iter().zip(&expected_state_after_first).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "restored CUDA delta state {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

#[test]
fn cuda_delta_net_prefill_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let seq_len = 3usize;
    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 29.0) - 14.0) * 0.0029296875;
    }
    let mut expected_state = state.clone();
    let mut q = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut k = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut v = vec![0.0f32; seq_len * num_heads * head_v_dim];
    let mut gate = vec![0.0f32; seq_len * num_heads];
    let mut beta = vec![0.0f32; seq_len * num_heads];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.01171875;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.009765625;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.02734375;
    }
    for (i, value) in gate.iter_mut().enumerate() {
        *value = -0.015625 * (i as f32 + 1.0);
    }
    for (i, value) in beta.iter_mut().enumerate() {
        *value = 0.125 + 0.03125 * i as f32;
    }

    let expected_first = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let expected_state_after_first = expected_state.clone();
    let expected_second = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let actual_first = match delta_net_prefill(
        &mut state, &q, &k, &v, &gate, &beta, seq_len, num_heads, head_k_dim, head_v_dim,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta prefill test: {err}");
            return;
        }
        Err(err) => panic!("CUDA delta prefill failed: {err}"),
    };
    for (i, (actual, expected)) in state.iter().zip(&expected_state_after_first).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "CUDA delta prefill host state {i} mismatch after first call: expected {expected}, actual {actual}, diff {diff}"
        );
    }
    let actual_second = delta_net_prefill(
        &mut state, &q, &k, &v, &gate, &beta, seq_len, num_heads, head_k_dim, head_v_dim,
    )
    .expect("second CUDA delta prefill call");

    for (i, (actual, expected)) in actual_first.iter().zip(&expected_first).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 0.001,
                "CUDA delta prefill first output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
    }
    for (i, (actual, expected)) in actual_second.iter().zip(&expected_second).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 0.001,
                "CUDA delta prefill second output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
    }
}

#[test]
fn cuda_delta_net_prefill_snapshot_restores_prefix_state() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let seq_len = 3usize;
    let snapshot_after_tokens = 1usize;
    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 29.0) - 14.0) * 0.0029296875;
    }
    let mut expected_state = state.clone();
    let mut q = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut k = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut v = vec![0.0f32; seq_len * num_heads * head_v_dim];
    let mut gate = vec![0.0f32; seq_len * num_heads];
    let mut beta = vec![0.0f32; seq_len * num_heads];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.01171875;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.009765625;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.02734375;
    }
    for (i, value) in gate.iter_mut().enumerate() {
        *value = -0.015625 * (i as f32 + 1.0);
    }
    for (i, value) in beta.iter_mut().enumerate() {
        *value = 0.125 + 0.03125 * i as f32;
    }

    let _ = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q[..snapshot_after_tokens * num_heads * head_k_dim],
        &k[..snapshot_after_tokens * num_heads * head_k_dim],
        &v[..snapshot_after_tokens * num_heads * head_v_dim],
        &gate[..snapshot_after_tokens * num_heads],
        &beta[..snapshot_after_tokens * num_heads],
        snapshot_after_tokens,
        num_heads,
        head_k_dim,
        head_v_dim,
    );

    let (_actual, snapshot) = match delta_net_prefill_resident_snapshot(
        &mut state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        snapshot_after_tokens,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta prefill snapshot test: {err}");
            return;
        }
        Err(err) => panic!("CUDA delta prefill snapshot failed: {err}"),
    };
    let snapshot = snapshot.expect("prefix delta snapshot");

    assert!(
        restore_delta_state_cache(&mut state, &snapshot).expect("restore prefix snapshot"),
        "restore should find resident delta state"
    );
    assert!(
        sync_delta_state_cache(&mut state).expect("sync prefix snapshot"),
        "sync should find resident delta state"
    );
    free_delta_state_snapshot(snapshot).expect("free prefix snapshot");

    for (i, (actual, expected)) in state.iter().zip(&expected_state).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "CUDA delta prefix snapshot state {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

#[test]
fn cuda_delta_net_prefill_snapshot_restores_two_token_prefix_state() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let seq_len = 3usize;
    let snapshot_after_tokens = 2usize;
    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 29.0) - 14.0) * 0.0029296875;
    }
    let mut expected_state = state.clone();
    let mut q = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut k = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut v = vec![0.0f32; seq_len * num_heads * head_v_dim];
    let mut gate = vec![0.0f32; seq_len * num_heads];
    let mut beta = vec![0.0f32; seq_len * num_heads];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.01171875;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.009765625;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.02734375;
    }
    for (i, value) in gate.iter_mut().enumerate() {
        *value = -0.015625 * (i as f32 + 1.0);
    }
    for (i, value) in beta.iter_mut().enumerate() {
        *value = 0.125 + 0.03125 * i as f32;
    }

    let _ = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q[..snapshot_after_tokens * num_heads * head_k_dim],
        &k[..snapshot_after_tokens * num_heads * head_k_dim],
        &v[..snapshot_after_tokens * num_heads * head_v_dim],
        &gate[..snapshot_after_tokens * num_heads],
        &beta[..snapshot_after_tokens * num_heads],
        snapshot_after_tokens,
        num_heads,
        head_k_dim,
        head_v_dim,
    );

    let (_actual, snapshot) = match delta_net_prefill_resident_snapshot(
        &mut state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        snapshot_after_tokens,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta prefill snapshot test: {err}");
            return;
        }
        Err(err) => panic!("CUDA delta prefill snapshot failed: {err}"),
    };
    let snapshot = snapshot.expect("prefix delta snapshot");

    assert!(
        restore_delta_state_cache(&mut state, &snapshot).expect("restore prefix snapshot"),
        "restore should find resident delta state"
    );
    assert!(
        sync_delta_state_cache(&mut state).expect("sync prefix snapshot"),
        "sync should find resident delta state"
    );
    free_delta_state_snapshot(snapshot).expect("free prefix snapshot");

    for (i, (actual, expected)) in state.iter().zip(&expected_state).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "CUDA delta two-token prefix snapshot state {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

#[test]
fn cuda_delta_net_prefill_snapshots_restore_multiple_prefix_states() {
    let _guard = runtime_test_lock();
    reset_delta_state_cache().expect("reset CUDA delta state cache");

    let seq_len = 3usize;
    let snapshot_after_tokens = [1usize, 2usize];
    let num_heads = 2usize;
    let head_k_dim = 128usize;
    let head_v_dim = 4usize;
    let state_len = num_heads * head_k_dim * head_v_dim;
    let mut state = vec![0.0f32; state_len];
    for (i, value) in state.iter_mut().enumerate() {
        *value = ((i as f32 % 29.0) - 14.0) * 0.0029296875;
    }
    let initial_state = state.clone();
    let mut q = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut k = vec![0.0f32; seq_len * num_heads * head_k_dim];
    let mut v = vec![0.0f32; seq_len * num_heads * head_v_dim];
    let mut gate = vec![0.0f32; seq_len * num_heads];
    let mut beta = vec![0.0f32; seq_len * num_heads];
    for (i, value) in q.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.01171875;
    }
    for (i, value) in k.iter_mut().enumerate() {
        *value = ((i as f32 % 19.0) - 9.0) * 0.009765625;
    }
    for (i, value) in v.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.02734375;
    }
    for (i, value) in gate.iter_mut().enumerate() {
        *value = -0.015625 * (i as f32 + 1.0);
    }
    for (i, value) in beta.iter_mut().enumerate() {
        *value = 0.125 + 0.03125 * i as f32;
    }

    let expected_states = snapshot_after_tokens
        .iter()
        .map(|&tokens| {
            let mut expected_state = initial_state.clone();
            let _ = cpu_delta_net_prefill_reference(
                &mut expected_state,
                &q[..tokens * num_heads * head_k_dim],
                &k[..tokens * num_heads * head_k_dim],
                &v[..tokens * num_heads * head_v_dim],
                &gate[..tokens * num_heads],
                &beta[..tokens * num_heads],
                tokens,
                num_heads,
                head_k_dim,
                head_v_dim,
            );
            expected_state
        })
        .collect::<Vec<_>>();

    let (_actual, snapshots) = match delta_net_prefill_resident_snapshots(
        &mut state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        &snapshot_after_tokens,
    ) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") => {
            eprintln!("skipping CUDA delta prefill multi-snapshot test: {err}");
            return;
        }
        Err(err) => panic!("CUDA delta prefill multi-snapshot failed: {err}"),
    };

    assert_eq!(snapshots.len(), snapshot_after_tokens.len());
    for (snapshot, expected_state) in snapshots.iter().zip(expected_states.iter()) {
        assert!(
            restore_delta_state_cache(&mut state, snapshot).expect("restore prefix snapshot"),
            "restore should find resident delta state"
        );
        assert!(
            sync_delta_state_cache(&mut state).expect("sync prefix snapshot"),
            "sync should find resident delta state"
        );

        for (i, (actual, expected)) in state.iter().zip(expected_state).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.001,
                "CUDA delta multi-prefix snapshot state {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
        }
    }

    for snapshot in snapshots {
        free_delta_state_snapshot(snapshot).expect("free prefix snapshot");
    }
}

#[test]
fn cuda_f32_gemm_batch_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 7usize;
    let seq_len = 3usize;
    let mut weights = vec![0.0f32; rows * cols];
    let mut input = vec![0.0f32; seq_len * cols];
    for (i, value) in weights.iter_mut().enumerate() {
        *value = ((i as f32 % 11.0) - 5.0) * 0.125;
    }
    for (i, value) in input.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.0625;
    }
    let mut expected = vec![0.0f32; seq_len * rows];
    for s in 0..seq_len {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += weights[row * cols + col] * input[s * cols + col];
            }
            expected[s * rows + row] = acc;
        }
    }

    let actual = match f32_gemm_batch(&weights, rows, cols, &input) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") || err.contains("cuBLAS") => {
            eprintln!("skipping CUDA f32 GEMM test: {err}");
            return;
        }
        Err(err) => panic!("CUDA f32 GEMM failed: {err}"),
    };
    for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.0001,
            "CUDA f32 GEMM output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}
#[test]
fn cuda_f32_shared_expert_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let n_embd = 7usize;
    let n_ff = 5usize;
    let seq_len = 3usize;
    let mut gate = vec![0.0f32; n_ff * n_embd];
    let mut up = vec![0.0f32; n_ff * n_embd];
    let mut down = vec![0.0f32; n_embd * n_ff];
    let mut input = vec![0.0f32; seq_len * n_embd];
    let route = [0.25f32, 0.5, 0.75];
    for (i, value) in gate.iter_mut().enumerate() {
        *value = ((i as f32 % 11.0) - 5.0) * 0.125;
    }
    for (i, value) in up.iter_mut().enumerate() {
        *value = ((i as f32 % 13.0) - 6.0) * 0.0625;
    }
    for (i, value) in down.iter_mut().enumerate() {
        *value = ((i as f32 % 7.0) - 3.0) * 0.2;
    }
    for (i, value) in input.iter_mut().enumerate() {
        *value = ((i as f32 % 17.0) - 8.0) * 0.05;
    }

    let mut hidden = vec![0.0f32; seq_len * n_ff];
    for t in 0..seq_len {
        for row in 0..n_ff {
            let mut gate_acc = 0.0f32;
            let mut up_acc = 0.0f32;
            for col in 0..n_embd {
                let x = input[t * n_embd + col];
                gate_acc += gate[row * n_embd + col] * x;
                up_acc += up[row * n_embd + col] * x;
            }
            hidden[t * n_ff + row] = (gate_acc / (1.0 + (-gate_acc).exp())) * up_acc;
        }
    }
    let mut expected = vec![0.0f32; seq_len * n_embd];
    for t in 0..seq_len {
        for row in 0..n_embd {
            let mut acc = 0.0f32;
            for col in 0..n_ff {
                acc += down[row * n_ff + col] * hidden[t * n_ff + col];
            }
            expected[t * n_embd + row] = acc * route[t];
        }
    }

    let actual = match f32_shared_expert(&gate, &up, &down, &route, n_ff, n_embd, &input) {
        Ok(actual) => actual,
        Err(err) if err.contains("CUDA") || err.contains("cuda") || err.contains("cuBLAS") => {
            eprintln!("skipping CUDA f32 shared expert test: {err}");
            return;
        }
        Err(err) => panic!("CUDA f32 shared expert failed: {err}"),
    };
    for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
                diff < 0.0001,
                "CUDA f32 shared expert output {i} mismatch: expected {expected}, actual {actual}, diff {diff}"
            );
    }
}
#[test]
fn cuda_q6k_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let rows = 5usize;
    let cols = 512usize;
    let blocks_per_row = cols / 256;
    let mut weights = vec![0u8; rows * blocks_per_row * 210];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 210;
            let block = &mut weights[base..base + 210];
            for i in 0..128usize {
                block[i] = ((row_idx * 17 + block_idx * 23 + i * 29 + 3) & 0xff) as u8;
            }
            for i in 0..64usize {
                block[128 + i] = ((row_idx * 31 + block_idx * 11 + i * 7 + 5) & 0xff) as u8;
            }
            for i in 0..16usize {
                block[192 + i] = ((i as i8 % 7) - 3) as u8;
            }
            block[208..210].copy_from_slice(
                &half::f16::from_f32(
                    0.01171875 + row_idx as f32 * 0.001953125 + block_idx as f32 * 0.0009765625,
                )
                .to_le_bytes(),
            );
        }
    }
    let mut input = vec![0.0f32; cols];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32 % 31.0) - 15.0) * 0.025;
    }
    let mut expected = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * blocks_per_row * 210..(row_idx + 1) * blocks_per_row * 210];
        expected[row_idx] = row
            .chunks_exact(210)
            .enumerate()
            .map(|(block_idx, block)| {
                let block: &[u8; 210] = block.try_into().unwrap();
                let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                    .try_into()
                    .unwrap();
                cpu_q6k_block_dot(block, input)
            })
            .sum::<f32>();
    }
    let actual = q6k_gemv_for_test(&weights, rows, cols, &input).expect("CUDA Q6_K GEMV");
    for row_idx in 0..rows {
        let diff = (actual[row_idx] - expected[row_idx]).abs();
        assert!(
            diff < 0.05,
            "CUDA Q6_K GEMV row {row_idx} mismatch: expected {}, actual {}, diff {}",
            expected[row_idx],
            actual[row_idx],
            diff
        );
    }
}
fn cpu_q4k_block_dot(block: &[u8; 144], input: &[f32; 256]) -> f32 {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: [u8; 12] = block[4..16].try_into().unwrap();
    let (sc, mn) = cpu_extract_q4k_scales(&scales);
    let qs = &block[16..144];
    let mut acc = 0.0f32;
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut y_off = 0usize;
    for _ in 0..4 {
        let d1 = d * sc[is] as f32;
        let m1 = dmin * mn[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * mn[is + 1] as f32;
        for l in 0..32 {
            let y = d1 * (qs[q_off + l] & 0x0f) as f32 - m1;
            acc += y * input[y_off + l];
        }
        for l in 0..32 {
            let y = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            acc += y * input[y_off + 32 + l];
        }
        q_off += 32;
        is += 2;
        y_off += 64;
    }
    acc
}

fn cpu_q4k_dequant_row(row: &[u8], blocks_per_row: usize) -> Vec<f32> {
    assert_eq!(row.len(), blocks_per_row * 144);
    let mut output = Vec::with_capacity(blocks_per_row * 256);
    for block in row.chunks_exact(144) {
        let block: &[u8; 144] = block.try_into().unwrap();
        output.extend(cpu_q4k_dequant_block(block));
    }
    output
}

fn cpu_q4k_dequant_block(block: &[u8; 144]) -> [f32; 256] {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: [u8; 12] = block[4..16].try_into().unwrap();
    let (sc, mn) = cpu_extract_q4k_scales(&scales);
    let qs = &block[16..144];
    let mut output = [0.0f32; 256];
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut y_off = 0usize;
    for _ in 0..4 {
        let d1 = d * sc[is] as f32;
        let m1 = dmin * mn[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * mn[is + 1] as f32;
        for l in 0..32 {
            output[y_off + l] = d1 * (qs[q_off + l] & 0x0f) as f32 - m1;
        }
        for l in 0..32 {
            output[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
        }
        q_off += 32;
        is += 2;
        y_off += 64;
    }
    output
}

fn cpu_q5k_block_dot(block: &[u8; 176], input: &[f32; 256]) -> f32 {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: [u8; 12] = block[4..16].try_into().unwrap();
    let (sc, mn) = cpu_extract_q4k_scales(&scales);
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut acc = 0.0f32;
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut y_off = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for _ in 0..4 {
        let d1 = d * sc[is] as f32;
        let m1 = dmin * mn[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * mn[is + 1] as f32;
        for l in 0..32 {
            let high = if qh[l] & u1 != 0 { 16 } else { 0 };
            let y = d1 * ((qs[q_off + l] & 0x0f) + high) as f32 - m1;
            acc += y * input[y_off + l];
        }
        for l in 0..32 {
            let high = if qh[l] & u2 != 0 { 16 } else { 0 };
            let y = d2 * ((qs[q_off + l] >> 4) + high) as f32 - m2;
            acc += y * input[y_off + 32 + l];
        }
        q_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        y_off += 64;
    }
    acc
}

fn cpu_q2k_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    assert_eq!(std::mem::size_of::<rnb_cpu::quantize::BlockQ2_K>(), 84);
    let row_bytes = blocks_per_row * 84;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            row.chunks_exact(84)
                .enumerate()
                .map(|(block_idx, block_bytes)| {
                    let block = unsafe {
                        std::ptr::read_unaligned(
                            block_bytes.as_ptr().cast::<rnb_cpu::quantize::BlockQ2_K>(),
                        )
                    };
                    let mut values = [0.0f32; 256];
                    rnb_cpu::quantize::dequantize_q2_k(&block, &mut values);
                    values
                        .iter()
                        .zip(&input[block_idx * 256..(block_idx + 1) * 256])
                        .map(|(&weight, &x)| weight * x)
                        .sum::<f32>()
                })
                .sum::<f32>()
        })
        .collect()
}

fn cpu_q3k_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    assert_eq!(std::mem::size_of::<rnb_cpu::quantize::BlockQ3_K>(), 110);
    let row_bytes = blocks_per_row * 110;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            row.chunks_exact(110)
                .enumerate()
                .map(|(block_idx, block_bytes)| {
                    let block = unsafe {
                        std::ptr::read_unaligned(
                            block_bytes.as_ptr().cast::<rnb_cpu::quantize::BlockQ3_K>(),
                        )
                    };
                    let mut values = [0.0f32; 256];
                    rnb_cpu::quantize::dequantize_q3_k(&block, &mut values);
                    values
                        .iter()
                        .zip(&input[block_idx * 256..(block_idx + 1) * 256])
                        .map(|(&weight, &x)| weight * x)
                        .sum::<f32>()
                })
                .sum::<f32>()
        })
        .collect()
}

fn cpu_q5k_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    let row_bytes = blocks_per_row * 176;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            row.chunks_exact(176)
                .enumerate()
                .map(|(block_idx, block)| {
                    let block: &[u8; 176] = block.try_into().unwrap();
                    let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                        .try_into()
                        .unwrap();
                    cpu_q5k_block_dot(block, input)
                })
                .sum::<f32>()
        })
        .collect()
}

fn cpu_q5_basic_rows(
    weights: &[u8],
    rows: usize,
    cols: usize,
    block_bytes: usize,
    has_min: bool,
    input: &[f32],
) -> Vec<f32> {
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * block_bytes;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            let mut acc = 0.0f32;
            for block_idx in 0..blocks_per_row {
                let block = &row[block_idx * block_bytes..(block_idx + 1) * block_bytes];
                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let (m, qh_off, qs_off) = if has_min {
                    (
                        half::f16::from_le_bytes([block[2], block[3]]).to_f32(),
                        4usize,
                        8usize,
                    )
                } else {
                    (0.0f32, 2usize, 6usize)
                };
                let qh =
                    u32::from_le_bytes(block[qh_off..qh_off + 4].try_into().expect("Q5 qh bytes"));
                for lane in 0..32usize {
                    let byte = block[qs_off + (lane & 15)];
                    let low = if lane < 16 { byte & 0x0f } else { byte >> 4 };
                    let high = ((qh >> lane) & 1) as u8;
                    let q = (low | (high << 4)) as f32;
                    let y = if has_min { q * d + m } else { (q - 16.0) * d };
                    acc += y * input[block_idx * 32 + lane];
                }
            }
            acc
        })
        .collect()
}

fn assert_close_rows(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    for (row_idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{label} row {row_idx} mismatch: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

fn assert_close_rows_abs_rel(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    abs_tolerance: f32,
    rel_tolerance: f32,
) {
    for (row_idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
        assert!(
            diff <= tolerance,
            "{label} row {row_idx} mismatch: expected {expected}, actual {actual}, diff {diff}, tolerance {tolerance}"
        );
    }
}

fn max_abs_rel(actual: &[f32], expected: &[f32]) -> (usize, f32, f32) {
    assert_eq!(actual.len(), expected.len());
    actual.iter().zip(expected.iter()).enumerate().fold(
        (0usize, 0.0f32, 0.0f32),
        |(max_idx, max_abs, max_rel), (idx, (&actual, &expected))| {
            let diff = (actual - expected).abs();
            let rel = diff / expected.abs().max(1.0e-6);
            if diff > max_abs || rel > max_rel {
                (idx, max_abs.max(diff), max_rel.max(rel))
            } else {
                (max_idx, max_abs, max_rel)
            }
        },
    )
}

fn assert_f16_bits_close(label: &str, actual: &[u16], expected: &[u16], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let actual = half::f16::from_bits(actual).to_f32();
        let expected = half::f16::from_bits(expected).to_f32();
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{label} idx {idx} mismatch: expected {expected}, actual {actual}, diff {diff}, tolerance {tolerance}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_qk_norm_rope_neox_hd512(
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    pos_start: usize,
    eps: f32,
    theta: f32,
    freq_factors: Option<&[f32]>,
    unit_offset: bool,
) -> Vec<f32> {
    assert_eq!(head_dim, 512);
    cpu_qk_norm_rope_neox(
        input,
        weight,
        seq_len,
        heads,
        head_dim,
        pos_start,
        eps,
        theta,
        freq_factors,
        unit_offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn cpu_qk_norm_rope_neox(
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    pos_start: usize,
    eps: f32,
    theta: f32,
    freq_factors: Option<&[f32]>,
    unit_offset: bool,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = vec![0.0f32; input.len()];
    for t in 0..seq_len {
        let pos = pos_start + t;
        for h in 0..heads {
            let off = (t * heads + h) * head_dim;
            let src = &input[off..off + head_dim];
            let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv_rms = (mean_sq + eps).sqrt().recip();
            for i in 0..half {
                let factor = freq_factors.map(|f| f[i]).unwrap_or(1.0);
                let freq = 1.0 / (theta.powf((2 * i) as f32 / head_dim as f32) * factor);
                let angle = pos as f32 * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let scale0 = if unit_offset {
                    1.0 + weight[i]
                } else {
                    weight[i]
                };
                let scale1 = if unit_offset {
                    1.0 + weight[half + i]
                } else {
                    weight[half + i]
                };
                let x0 = src[i] * inv_rms * scale0;
                let x1 = src[half + i] * inv_rms * scale1;
                out[off + i] = x0 * cos_a - x1 * sin_a;
                out[off + half + i] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_qk_norm_rope_select(
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_neox: bool,
    pos_start: usize,
    eps: f32,
    theta: f32,
    freq_factors: Option<&[f32]>,
    unit_offset: bool,
) -> Vec<f32> {
    let rope_dim = if rope_dim == 0 {
        head_dim
    } else {
        rope_dim.min(head_dim)
    };
    assert_eq!(rope_dim % 2, 0);
    let mut out = vec![0.0f32; input.len()];
    for t in 0..seq_len {
        let pos = pos_start + t;
        for h in 0..heads {
            let off = (t * heads + h) * head_dim;
            let src = &input[off..off + head_dim];
            let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv_rms = (mean_sq + eps).sqrt().recip();
            for dim in 0..head_dim {
                if dim >= rope_dim {
                    let scale = if unit_offset {
                        1.0 + weight[dim]
                    } else {
                        weight[dim]
                    };
                    out[off + dim] = src[dim] * inv_rms * scale;
                    continue;
                }
                let (first_dim, second_dim, pair_idx, first_output) = if rope_neox {
                    let half_rot = rope_dim / 2;
                    if dim < half_rot {
                        (dim, half_rot + dim, dim, true)
                    } else {
                        let pair = dim - half_rot;
                        (pair, dim, pair, false)
                    }
                } else {
                    let first = dim & !1usize;
                    (first, first + 1, first / 2, dim == first)
                };
                let scale0 = if unit_offset {
                    1.0 + weight[first_dim]
                } else {
                    weight[first_dim]
                };
                let scale1 = if unit_offset {
                    1.0 + weight[second_dim]
                } else {
                    weight[second_dim]
                };
                let x0 = src[first_dim] * inv_rms * scale0;
                let x1 = src[second_dim] * inv_rms * scale1;
                let factor = freq_factors.map(|factors| factors[pair_idx]).unwrap_or(1.0);
                let freq = 1.0 / (theta.powf((2 * pair_idx) as f32 / rope_dim as f32) * factor);
                let angle = pos as f32 * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                out[off + dim] = if first_output {
                    x0 * cos_a - x1 * sin_a
                } else {
                    x0 * sin_a + x1 * cos_a
                };
            }
        }
    }
    out
}

fn cpu_v_norm_pack_hd512(
    input: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(head_dim, 512);
    cpu_v_norm_pack(input, seq_len, heads, head_dim, eps)
}

fn cpu_v_norm_pack(
    input: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for t in 0..seq_len {
        for h in 0..heads {
            let off = (t * heads + h) * head_dim;
            let src = &input[off..off + head_dim];
            let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv_rms = (mean_sq + eps).sqrt().recip();
            for i in 0..head_dim {
                out[off + i] = src[i] * inv_rms;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_prefill_attention_hd512_f16kv(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let head_dim = 512usize;
    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; seq_len * num_heads * head_dim];
    for t in 0..seq_len {
        let global_pos = kv_len - seq_len + t;
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1);
            for j in 0..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| *a * half::f16::from_bits(*b).to_f32())
                    .sum::<f32>()
                    * scale;
                scores.push(dot);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            for (j, score) in scores.iter().enumerate() {
                let p = (*score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * half::f16::from_bits(v[v_off + d]).to_f32();
                }
            }
        }
    }
    expected
}

fn cpu_q8_0_rows(weights: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            let mut acc = 0.0f32;
            for block_idx in 0..blocks_per_row {
                let block = &row[block_idx * 34..(block_idx + 1) * 34];
                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for lane in 0..32usize {
                    let q = block[2 + lane] as i8 as f32;
                    acc += q * d * input[block_idx * 32 + lane];
                }
            }
            acc
        })
        .collect()
}

fn cpu_q8_0_q8dot_rows(weights: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let blocks_per_row = cols / 32;
    let mut input_qs = vec![0i8; cols];
    let mut input_ds = vec![0.0f32; blocks_per_row];
    for block_idx in 0..blocks_per_row {
        let off = block_idx * 32;
        let chunk = &input[off..off + 32];
        let max_abs = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let d = max_abs / 127.0;
        let inv_d = 1.0 / d;
        input_ds[block_idx] = d;
        for (idx, &value) in chunk.iter().enumerate() {
            input_qs[off + idx] = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
        }
    }

    let row_bytes = blocks_per_row * 34;
    (0..rows)
        .map(|row_idx| {
            let row = &weights[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            let mut acc = 0.0f32;
            for block_idx in 0..blocks_per_row {
                let block = &row[block_idx * 34..(block_idx + 1) * 34];
                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let x_d = input_ds[block_idx];
                let mut dot = 0i32;
                for lane in 0..32usize {
                    let q = block[2 + lane] as i8 as i32;
                    let x = input_qs[block_idx * 32 + lane] as i32;
                    dot += q * x;
                }
                acc += d * x_d * dot as f32;
            }
            acc
        })
        .collect()
}

fn make_test_q8_0_weights(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    let blocks_per_row = cols / 32;
    let mut weights = vec![0u8; rows * blocks_per_row * 34];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 34;
            let block = &mut weights[base..base + 34];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(
                    0.015625 + ((seed + row_idx * 3 + block_idx * 5) % 13) as f32 * 0.001953125,
                )
                .to_le_bytes(),
            );
            for lane in 0..32 {
                block[2 + lane] =
                    (((seed + row_idx * 13 + block_idx * 17 + lane * 7) % 31) as i8 - 15) as u8;
            }
        }
    }
    weights
}

#[allow(clippy::too_many_arguments)]
fn cpu_nemotron_mamba2_decode_scan(
    state: &mut [f32],
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    a: &[f32],
    d: &[f32],
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    n_group: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let group = h / (num_heads / n_group);
        for p in 0..head_dim {
            let x_idx = h * head_dim + p;
            let x_dt = x[x_idx] * dt[h];
            let decay = (dt[h] * a[h]).exp();
            let mut y = d[h] * x[x_idx];
            for s in 0..state_dim {
                let bc_idx = group * state_dim + s;
                let state_idx = h * head_dim * state_dim + p * state_dim + s;
                state[state_idx] = state[state_idx] * decay + b[bc_idx] * x_dt;
                y += state[state_idx] * c[bc_idx];
            }
            out[x_idx] = y;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_nemotron_mamba2_prefill_scan(
    state: &mut [f32],
    conv: &[f32],
    dt_data: &[f32],
    a: &[f32],
    d: &[f32],
    seq_len: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    n_group: usize,
) -> Vec<f32> {
    let d_inner = num_heads * head_dim;
    let mut out = vec![0.0f32; seq_len * d_inner];
    for t in 0..seq_len {
        let token = &conv[t * conv_channels..(t + 1) * conv_channels];
        for h in 0..num_heads {
            let group = h / (num_heads / n_group);
            let dt_raw = dt_data[t * num_heads + h];
            let dt = if dt_raw > 20.0 {
                dt_raw
            } else {
                (1.0 + dt_raw.exp()).ln()
            };
            for p in 0..head_dim {
                let x_idx = h * head_dim + p;
                let x_dt = token[x_idx] * dt;
                let decay = (dt * a[h]).exp();
                let mut y = d[h] * token[x_idx];
                for s in 0..state_dim {
                    let bc_idx = group * state_dim + s;
                    let state_idx = h * head_dim * state_dim + p * state_dim + s;
                    state[state_idx] = state[state_idx] * decay + token[d_inner + bc_idx] * x_dt;
                    y += state[state_idx] * token[d_inner + bc_dim + bc_idx];
                }
                out[t * d_inner + x_idx] = y;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_nemotron_mamba2_prefill_device_reference(
    input: &[f32],
    ssm_in: &[u8],
    ssm_in_rows: usize,
    ssm_in_cols: usize,
    ssm_out: &[u8],
    ssm_out_rows: usize,
    ssm_out_cols: usize,
    input_norm: &[f32],
    conv_kernel: &[f32],
    conv_bias: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_d: &[f32],
    ssm_norm: &[f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    d_inner: usize,
    conv_channels: usize,
    bc_dim: usize,
    num_heads: usize,
    head_dim: usize,
    n_group: usize,
    d_state: usize,
    conv_kernel_size: usize,
    norm_eps: f32,
) -> Vec<f32> {
    let mut normed = Vec::with_capacity(seq_len * hidden_dim);
    for token in input.chunks_exact(hidden_dim) {
        normed.extend(cpu_rms_norm(token, input_norm, norm_eps, false));
    }

    let mut projected = Vec::with_capacity(seq_len * ssm_in_rows);
    for token in normed.chunks_exact(ssm_in_cols) {
        projected.extend(cpu_q8_0_rows(ssm_in, ssm_in_rows, ssm_in_cols, token));
    }

    let mut z = vec![0.0f32; seq_len * d_inner];
    let mut conv_seed = vec![0.0f32; seq_len * conv_channels];
    let mut dt = vec![0.0f32; seq_len * num_heads];
    for t in 0..seq_len {
        let src = &projected[t * ssm_in_rows..(t + 1) * ssm_in_rows];
        z[t * d_inner..(t + 1) * d_inner].copy_from_slice(&src[..d_inner]);
        conv_seed[t * conv_channels..(t + 1) * conv_channels]
            .copy_from_slice(&src[d_inner..d_inner + conv_channels]);
        for h in 0..num_heads {
            dt[t * num_heads + h] = src[d_inner + conv_channels + h] + dt_bias[h];
        }
    }

    let state_rows = conv_kernel_size - 1;
    let mut conv_input = vec![0.0f32; (seq_len + state_rows) * conv_channels];
    conv_input[..conv_state.len()].copy_from_slice(conv_state);
    conv_input[conv_state.len()..].copy_from_slice(&conv_seed);
    conv_state.copy_from_slice(&conv_input[seq_len * conv_channels..][..conv_state.len()]);

    let mut conv_out = vec![0.0f32; seq_len * conv_channels];
    for t in 0..seq_len {
        for c in 0..conv_channels {
            let mut sum = conv_bias[c];
            for k in 0..conv_kernel_size {
                sum += conv_input[(t + k) * conv_channels + c] * conv_kernel[k * conv_channels + c];
            }
            conv_out[t * conv_channels + c] = sum / (1.0 + (-sum).exp());
        }
    }

    let scan_out = cpu_nemotron_mamba2_prefill_scan(
        delta_state,
        &conv_out,
        &dt,
        ssm_a,
        ssm_d,
        seq_len,
        conv_channels,
        bc_dim,
        num_heads,
        head_dim,
        d_state,
        n_group,
    );

    let group_width = d_inner / n_group;
    let mut gated = vec![0.0f32; seq_len * d_inner];
    for row in 0..seq_len * n_group {
        let start = row * group_width;
        let normed_row = cpu_rms_norm(
            &scan_out[start..start + group_width],
            &ssm_norm[..group_width],
            norm_eps,
            false,
        );
        for i in 0..group_width {
            let zv = z[start + i];
            gated[start + i] = normed_row[i] * (zv / (1.0 + (-zv).exp()));
        }
    }

    let mut output = Vec::with_capacity(seq_len * ssm_out_rows);
    for (token_idx, token) in gated.chunks_exact(ssm_out_cols).enumerate() {
        let mut projected = cpu_q8_0_rows(ssm_out, ssm_out_rows, ssm_out_cols, token);
        let residual = &input[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
        for (value, &residual) in projected.iter_mut().zip(residual.iter()) {
            *value += residual;
        }
        output.extend(projected);
    }
    output
}

fn make_test_q5_basic_weights(
    rows: usize,
    cols: usize,
    block_bytes: usize,
    has_min: bool,
    seed: usize,
) -> Vec<u8> {
    let blocks_per_row = cols / 32;
    let mut weights = vec![0u8; rows * blocks_per_row * block_bytes];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * block_bytes;
            let block = &mut weights[base..base + block_bytes];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(
                    0.015625 + ((seed + row_idx * 3 + block_idx * 5) % 11) as f32 * 0.001953125,
                )
                .to_le_bytes(),
            );
            let (qh_off, qs_off) = if has_min {
                block[2..4].copy_from_slice(
                    &half::f16::from_f32(-0.1875 + ((seed + block_idx * 7) % 9) as f32 * 0.015625)
                        .to_le_bytes(),
                );
                (4usize, 8usize)
            } else {
                (2usize, 6usize)
            };
            for i in 0..4 {
                block[qh_off + i] = ((seed + row_idx * 13 + block_idx * 17 + i * 19) & 0xff) as u8;
            }
            for i in 0..16 {
                block[qs_off + i] = ((seed + row_idx * 23 + block_idx * 29 + i * 31) & 0xff) as u8;
            }
        }
    }
    weights
}

fn make_test_q2k_weights(rows: usize, blocks_per_row: usize, seed: usize) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks_per_row * 84];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 84;
            let block = &mut weights[base..base + 84];
            for i in 0..16 {
                block[i] = ((seed + row_idx * 11 + block_idx * 17 + i * 23) & 0xff) as u8;
            }
            for i in 0..64 {
                block[16 + i] = ((seed + row_idx * 29 + block_idx * 31 + i * 37) & 0xff) as u8;
            }
            block[80..82].copy_from_slice(
                &half::f16::from_f32(
                    0.00390625 + row_idx as f32 * 0.00024414063 + block_idx as f32 * 0.00012207031,
                )
                .to_le_bytes(),
            );
            block[82..84].copy_from_slice(
                &half::f16::from_f32(
                    0.001953125
                        + row_idx as f32 * 0.00012207031
                        + block_idx as f32 * 0.000061035156,
                )
                .to_le_bytes(),
            );
        }
    }
    weights
}

fn make_test_q3k_weights(rows: usize, blocks_per_row: usize, seed: usize) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks_per_row * 110];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 110;
            let block = &mut weights[base..base + 110];
            for i in 0..32 {
                block[i] = ((seed + row_idx * 13 + block_idx * 19 + i * 29) & 0xff) as u8;
            }
            for i in 0..64 {
                block[32 + i] = ((seed + row_idx * 31 + block_idx * 37 + i * 41) & 0xff) as u8;
            }
            for i in 0..12 {
                block[96 + i] = ((seed + row_idx * 43 + block_idx * 47 + i * 53) & 0xff) as u8;
            }
            block[108..110].copy_from_slice(
                &half::f16::from_f32(
                    0.00390625 + row_idx as f32 * 0.00024414063 + block_idx as f32 * 0.00012207031,
                )
                .to_le_bytes(),
            );
        }
    }
    weights
}

fn make_test_q4k_weights(
    experts: usize,
    rows: usize,
    blocks_per_row: usize,
    seed: usize,
) -> Vec<Vec<u8>> {
    let mut all = Vec::with_capacity(experts);
    for expert_idx in 0..experts {
        let mut weights = vec![0u8; rows * blocks_per_row * 144];
        for row_idx in 0..rows {
            for block_idx in 0..blocks_per_row {
                let base = (row_idx * blocks_per_row + block_idx) * 144;
                let block = &mut weights[base..base + 144];
                block[0..2].copy_from_slice(
                    &half::f16::from_f32(
                        0.00390625
                            + expert_idx as f32 * 0.00048828125
                            + row_idx as f32 * 0.000030517578
                            + block_idx as f32 * 0.000015258789,
                    )
                    .to_le_bytes(),
                );
                block[2..4].copy_from_slice(
                    &half::f16::from_f32(0.001953125 + expert_idx as f32 * 0.00024414063)
                        .to_le_bytes(),
                );
                for i in 0..12usize {
                    block[4 + i] = ((seed + expert_idx * 13 + row_idx * 7 + block_idx * 5 + i * 3)
                        & 0x3f) as u8;
                }
                for i in 0..128usize {
                    block[16 + i] =
                        ((seed + expert_idx * 31 + row_idx * 17 + block_idx * 11 + i * 19) & 0xff)
                            as u8;
                }
            }
        }
        all.push(weights);
    }
    all
}

fn make_test_q5k_weights(rows: usize, blocks_per_row: usize, seed: usize) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks_per_row * 176];
    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let base = (row_idx * blocks_per_row + block_idx) * 176;
            let block = &mut weights[base..base + 176];
            block[0..2].copy_from_slice(
                &half::f16::from_f32(
                    0.0029296875
                        + row_idx as f32 * 0.00024414063
                        + block_idx as f32 * 0.00012207031,
                )
                .to_le_bytes(),
            );
            block[2..4].copy_from_slice(
                &half::f16::from_f32(0.00146484375 + row_idx as f32 * 0.000061035156).to_le_bytes(),
            );
            for i in 0..12usize {
                block[4 + i] = ((seed + row_idx * 29 + block_idx * 31 + i * 17) & 0xff) as u8;
            }
            for i in 0..32usize {
                block[16 + i] = ((seed + row_idx * 37 + block_idx * 41 + i * 19) & 0xff) as u8;
            }
            for i in 0..128usize {
                block[48 + i] = ((seed + row_idx * 43 + block_idx * 47 + i * 23) & 0xff) as u8;
            }
        }
    }
    weights
}

fn make_test_iq4_xs_weights(
    experts: usize,
    rows: usize,
    blocks_per_row: usize,
    seed: usize,
) -> Vec<Vec<u8>> {
    let mut all = Vec::with_capacity(experts);
    for expert_idx in 0..experts {
        let mut weights = vec![0u8; rows * blocks_per_row * 136];
        for row_idx in 0..rows {
            for block_idx in 0..blocks_per_row {
                let base = (row_idx * blocks_per_row + block_idx) * 136;
                let block = &mut weights[base..base + 136];
                block[0..2].copy_from_slice(
                    &half::f16::from_f32(
                        0.000061035156
                            + expert_idx as f32 * 0.0000076293945
                            + row_idx as f32 * 0.0000009536743
                            + block_idx as f32 * 0.00000047683716,
                    )
                    .to_bits()
                    .to_le_bytes(),
                );
                block[2..4].copy_from_slice(
                    &(((seed + expert_idx * 17 + row_idx * 5 + block_idx * 3) & 0xffff) as u16)
                        .to_le_bytes(),
                );
                for i in 0..4usize {
                    block[4 + i] = ((seed + expert_idx * 11 + row_idx * 13 + block_idx * 7 + i * 5)
                        & 0xff) as u8;
                }
                for i in 0..128usize {
                    block[8 + i] =
                        ((seed + expert_idx * 29 + row_idx * 31 + block_idx * 37 + i * 41) & 0xff)
                            as u8;
                }
            }
        }
        all.push(weights);
    }
    all
}

fn make_test_q6k_weights(
    experts: usize,
    rows: usize,
    blocks_per_row: usize,
    seed: usize,
) -> Vec<Vec<u8>> {
    let mut all = Vec::with_capacity(experts);
    for expert_idx in 0..experts {
        let mut weights = vec![0u8; rows * blocks_per_row * 210];
        for row_idx in 0..rows {
            for block_idx in 0..blocks_per_row {
                let base = (row_idx * blocks_per_row + block_idx) * 210;
                let block = &mut weights[base..base + 210];
                for i in 0..128usize {
                    block[i] = ((seed + expert_idx * 19 + row_idx * 17 + block_idx * 23 + i * 29)
                        & 0xff) as u8;
                }
                for i in 0..64usize {
                    block[128 + i] =
                        ((seed + expert_idx * 13 + row_idx * 31 + block_idx * 11 + i * 7) & 0xff)
                            as u8;
                }
                for i in 0..16usize {
                    block[192 + i] =
                        (((seed + expert_idx + row_idx + block_idx + i) % 9) as i8 - 4) as u8;
                }
                block[208..210].copy_from_slice(
                    &half::f16::from_f32(
                        0.01171875
                            + expert_idx as f32 * 0.001953125
                            + row_idx as f32 * 0.00024414063
                            + block_idx as f32 * 0.00012207031,
                    )
                    .to_le_bytes(),
                );
            }
        }
        all.push(weights);
    }
    all
}

fn cpu_rms_norm(input: &[f32], weight: &[f32], eps: f32, unit_offset: bool) -> Vec<f32> {
    assert_eq!(input.len(), weight.len());
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_sq + eps).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| {
            let scale = if unit_offset { 1.0 + w } else { w };
            x * inv_rms * scale
        })
        .collect()
}

fn cpu_rms_norm_div(input: &[f32], weight: &[f32], eps: f32, unit_offset: bool) -> Vec<f32> {
    assert_eq!(input.len(), weight.len());
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| {
            let scale = if unit_offset { 1.0 + w } else { w };
            (x / rms) * scale
        })
        .collect()
}

fn cpu_q4k_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * blocks_per_row * 144..(row_idx + 1) * blocks_per_row * 144];
        output[row_idx] = row
            .chunks_exact(144)
            .enumerate()
            .map(|(block_idx, block)| {
                let block: &[u8; 144] = block.try_into().unwrap();
                let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                    .try_into()
                    .unwrap();
                cpu_q4k_block_dot(block, input)
            })
            .sum::<f32>();
    }
    output
}

fn cpu_f32_gemv_rows(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(input.len(), cols);
    let mut output = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * cols..(row_idx + 1) * cols];
        output[row_idx] = row
            .iter()
            .zip(input.iter())
            .map(|(&w, &x)| w * x)
            .sum::<f32>();
    }
    output
}

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_f32_bin(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| {
        panic!("read f32 bin {} failed: {err}", path.display());
    });
    assert!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "f32 bin {} byte length must be divisible by 4, got {}",
        path.display(),
        bytes.len()
    );
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_optional_f32_bin(path: &std::path::Path) -> Option<Vec<f32>> {
    path.exists().then(|| read_f32_bin(path))
}

fn cpu_iq4_xs_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * blocks_per_row * 136..(row_idx + 1) * blocks_per_row * 136];
        output[row_idx] = iq4_xs_reference_row_dot(row, input);
    }
    output
}

fn cpu_q6k_gemv_rows(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for row_idx in 0..rows {
        let row = &weights[row_idx * blocks_per_row * 210..(row_idx + 1) * blocks_per_row * 210];
        output[row_idx] = row
            .chunks_exact(210)
            .enumerate()
            .map(|(block_idx, block)| {
                let block: &[u8; 210] = block.try_into().unwrap();
                let input: &[f32; 256] = input[block_idx * 256..(block_idx + 1) * 256]
                    .try_into()
                    .unwrap();
                cpu_q6k_block_dot(block, input)
            })
            .sum::<f32>();
    }
    output
}

fn cpu_q6k_dequant_row(row: &[u8], blocks_per_row: usize) -> Vec<f32> {
    assert_eq!(row.len(), blocks_per_row * 210);
    let mut output = Vec::with_capacity(blocks_per_row * 256);
    for block in row.chunks_exact(210) {
        let block: &[u8; 210] = block.try_into().unwrap();
        output.extend(cpu_q6k_dequant_block(block));
    }
    output
}

fn cpu_q6k_dequant_block(block: &[u8; 210]) -> [f32; 256] {
    let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let mut output = [0.0f32; 256];
    for n in 0..2usize {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let sc_base = n * 8;
        let y_base = n * 128;
        for l in 0..32usize {
            let is = l / 16;
            let q1 = (ql[ql_base + l] & 0x0f) | (((qh[qh_base + l] >> 0) & 3) << 4);
            let q2 = (ql[ql_base + l + 32] & 0x0f) | (((qh[qh_base + l] >> 2) & 3) << 4);
            let q3 = (ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4);
            let q4 = (ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4);
            let s1 = scales[sc_base + is] as i8 as f32;
            let s2 = scales[sc_base + is + 2] as i8 as f32;
            let s3 = scales[sc_base + is + 4] as i8 as f32;
            let s4 = scales[sc_base + is + 6] as i8 as f32;
            output[y_base + l] = d * s1 * (q1 as i32 - 32) as f32;
            output[y_base + l + 32] = d * s2 * (q2 as i32 - 32) as f32;
            output[y_base + l + 64] = d * s3 * (q3 as i32 - 32) as f32;
            output[y_base + l + 96] = d * s4 * (q4 as i32 - 32) as f32;
        }
    }
    output
}

fn cpu_qwen35_sparse_q2k_q3k_reference(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_embd];
    for expert_idx in 0..gate_weights.len() {
        let gate = cpu_q2k_gemv_rows(gate_weights[expert_idx], n_ff, n_embd / 256, input);
        let up = cpu_q2k_gemv_rows(up_weights[expert_idx], n_ff, n_embd / 256, input);
        let hidden = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        let down = cpu_q3k_gemv_rows(down_weights[expert_idx], n_embd, n_ff / 256, &hidden);
        for i in 0..n_embd {
            output[i] += route[expert_idx] * down[i];
        }
    }
    output
}

fn cpu_qwen35_sparse_q4k_reference(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_embd];
    for expert_idx in 0..gate_weights.len() {
        let gate = cpu_q4k_gemv_rows(gate_weights[expert_idx], n_ff, n_embd / 256, input);
        let up = cpu_q4k_gemv_rows(up_weights[expert_idx], n_ff, n_embd / 256, input);
        let mut hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            hidden[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
        let down = cpu_q4k_gemv_rows(down_weights[expert_idx], n_embd, n_ff / 256, &hidden);
        for row_idx in 0..n_embd {
            output[row_idx] += down[row_idx] * route[expert_idx];
        }
    }
    output
}

fn cpu_qwen35_sparse_q4k_q6k_reference(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_embd];
    for expert_idx in 0..gate_weights.len() {
        let gate = cpu_q4k_gemv_rows(gate_weights[expert_idx], n_ff, n_embd / 256, input);
        let up = cpu_q4k_gemv_rows(up_weights[expert_idx], n_ff, n_embd / 256, input);
        let mut hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            hidden[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
        let down = cpu_q6k_gemv_rows(down_weights[expert_idx], n_embd, n_ff / 256, &hidden);
        for row_idx in 0..n_embd {
            output[row_idx] += down[row_idx] * route[expert_idx];
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn cpu_qwen35_run_batched_q4k_q6k_by_token_reference(
    expert_ids: &[u32],
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    max_tile_slots: usize,
) -> Result<Vec<f32>, String> {
    let slots = expert_ids.len();
    for (name, len) in [
        ("gate_weights", gate_weights.len()),
        ("up_weights", up_weights.len()),
        ("down_weights", down_weights.len()),
        ("route", route.len()),
        ("token_ids", token_ids.len()),
    ] {
        if len != slots {
            return Err(format!(
                "run-batched reference {name} length mismatch: got {len}, expected {slots}"
            ));
        }
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "run-batched reference input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }

    let mut hidden_by_slot = vec![0.0f32; slots * n_ff];
    for slot in 0..slots {
        let token = token_ids[slot] as usize;
        if token >= token_count {
            return Err(format!(
                "run-batched reference token id out of range: token_id={} token_count={}",
                token_ids[slot], token_count
            ));
        }
        let token_input = &input[token * n_embd..(token + 1) * n_embd];
        let gate = cpu_q4k_gemv_rows(gate_weights[slot], n_ff, n_embd / 256, token_input);
        let up = cpu_q4k_gemv_rows(up_weights[slot], n_ff, n_embd / 256, token_input);
        let hidden = &mut hidden_by_slot[slot * n_ff..(slot + 1) * n_ff];
        for i in 0..n_ff {
            hidden[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
    }

    let runs = qwen35_expert_run_batched_down_plan(expert_ids, max_tile_slots)?;
    let mut output = vec![0.0f32; token_count * n_embd];
    for run in runs {
        let run_end = run.slot_start + run.len;
        for slot in run.slot_start + 1..run_end {
            if down_weights[slot] != down_weights[run.slot_start] {
                return Err(format!(
                    "run-batched reference down weight mismatch inside expert run: expert_id={} run_start={} slot={}",
                    run.expert_id, run.slot_start, slot
                ));
            }
        }

        let mut tile_start = run.slot_start;
        for _ in 0..run.full_tiles {
            for slot in tile_start..tile_start + max_tile_slots {
                let token = token_ids[slot] as usize;
                let hidden = &hidden_by_slot[slot * n_ff..(slot + 1) * n_ff];
                let down =
                    cpu_q6k_gemv_rows(down_weights[run.slot_start], n_embd, n_ff / 256, hidden);
                for row_idx in 0..n_embd {
                    output[token * n_embd + row_idx] += down[row_idx] * route[slot];
                }
            }
            tile_start += max_tile_slots;
        }
        if run.tail > 0 {
            for slot in tile_start..tile_start + run.tail {
                let token = token_ids[slot] as usize;
                let hidden = &hidden_by_slot[slot * n_ff..(slot + 1) * n_ff];
                let down =
                    cpu_q6k_gemv_rows(down_weights[run.slot_start], n_embd, n_ff / 256, hidden);
                for row_idx in 0..n_embd {
                    output[token * n_embd + row_idx] += down[row_idx] * route[slot];
                }
            }
            tile_start += run.tail;
        }
        debug_assert_eq!(tile_start, run_end);
    }

    Ok(output)
}

fn cpu_qwen35_sparse_q4k_q6k_q8_down_reference(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_embd];
    for expert_idx in 0..gate_weights.len() {
        let gate = cpu_q4k_gemv_rows(gate_weights[expert_idx], n_ff, n_embd / 256, input);
        let up = cpu_q4k_gemv_rows(up_weights[expert_idx], n_ff, n_embd / 256, input);
        let mut hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            hidden[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
        let hidden_q8 = cpu_q8_1_dequant_by_32(&hidden);
        let down = cpu_q6k_gemv_rows(down_weights[expert_idx], n_embd, n_ff / 256, &hidden_q8);
        for row_idx in 0..n_embd {
            output[row_idx] += down[row_idx] * route[expert_idx];
        }
    }
    output
}

fn cpu_q8_1_dequant_by_32(input: &[f32]) -> Vec<f32> {
    assert_eq!(input.len() % 32, 0);
    let mut output = vec![0.0f32; input.len()];
    for (chunk_idx, chunk) in input.chunks_exact(32).enumerate() {
        let max_abs = chunk
            .iter()
            .fold(0.0f32, |acc, &value| acc.max(value.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let d = max_abs / 127.0;
        let inv_d = 1.0 / d;
        for (lane, &value) in chunk.iter().enumerate() {
            let q = (value * inv_d).round().clamp(-127.0, 127.0);
            output[chunk_idx * 32 + lane] = q * d;
        }
    }
    output
}

fn cpu_qwen35_sparse_iq4xs_reference(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_embd];
    for expert_idx in 0..gate_weights.len() {
        let gate = cpu_iq4_xs_gemv_rows(gate_weights[expert_idx], n_ff, n_embd / 256, input);
        let up = cpu_iq4_xs_gemv_rows(up_weights[expert_idx], n_ff, n_embd / 256, input);
        let mut hidden = vec![0.0f32; n_ff];
        for i in 0..n_ff {
            hidden[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
        let down = cpu_iq4_xs_gemv_rows(down_weights[expert_idx], n_embd, n_ff / 256, &hidden);
        for row_idx in 0..n_embd {
            output[row_idx] += down[row_idx] * route[expert_idx];
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn cpu_delta_net_decode_reference(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * head_v_dim];
    let state_size = head_k_dim * head_v_dim;
    for h in 0..num_heads {
        for vi in 0..head_v_dim {
            let state_off = h * state_size + vi * head_k_dim;
            let qk_off = h * head_k_dim;
            let v_off = h * head_v_dim + vi;
            let mut sk = 0.0f32;
            let decay = gate[h].exp();
            for ki in 0..head_k_dim {
                sk += decay * state[state_off + ki] * k[qk_off + ki];
            }
            let d = (v[v_off] - sk) * beta[h];
            for ki in 0..head_k_dim {
                let idx = state_off + ki;
                state[idx] = decay * state[idx] + k[qk_off + ki] * d;
            }
            for ki in 0..head_k_dim {
                output[v_off] += state[state_off + ki] * q[qk_off + ki];
            }
        }
    }
    output
}
#[allow(clippy::too_many_arguments)]
fn cpu_delta_net_prefill_reference(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * num_heads * head_v_dim];
    for t in 0..seq_len {
        let qk_start = t * num_heads * head_k_dim;
        let v_start = t * num_heads * head_v_dim;
        let gate_start = t * num_heads;
        let step = cpu_delta_net_decode_reference(
            state,
            &q[qk_start..qk_start + num_heads * head_k_dim],
            &k[qk_start..qk_start + num_heads * head_k_dim],
            &v[v_start..v_start + num_heads * head_v_dim],
            &gate[gate_start..gate_start + num_heads],
            &beta[gate_start..gate_start + num_heads],
            num_heads,
            head_k_dim,
            head_v_dim,
        );
        output[v_start..v_start + num_heads * head_v_dim].copy_from_slice(&step);
    }
    output
}
fn cpu_extract_q4k_scales(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8usize {
        let (s, m) = if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            let s = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (s, m)
        };
        sc[j] = s;
        mn[j] = m;
    }
    (sc, mn)
}
fn cpu_q6k_block_dot(block: &[u8; 210], input: &[f32; 256]) -> f32 {
    let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let mut acc = 0.0f32;
    for n in 0..2usize {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let sc_base = n * 8;
        let y_base = n * 128;
        for l in 0..32usize {
            let is = l / 16;
            let q1 = (ql[ql_base + l] & 0x0f) | (((qh[qh_base + l] >> 0) & 3) << 4);
            let q2 = (ql[ql_base + l + 32] & 0x0f) | (((qh[qh_base + l] >> 2) & 3) << 4);
            let q3 = (ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4);
            let q4 = (ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4);
            let s1 = scales[sc_base + is] as i8 as f32;
            let s2 = scales[sc_base + is + 2] as i8 as f32;
            let s3 = scales[sc_base + is + 4] as i8 as f32;
            let s4 = scales[sc_base + is + 6] as i8 as f32;
            acc += d * s1 * (q1 as i32 - 32) as f32 * input[y_base + l];
            acc += d * s2 * (q2 as i32 - 32) as f32 * input[y_base + l + 32];
            acc += d * s3 * (q3 as i32 - 32) as f32 * input[y_base + l + 64];
            acc += d * s4 * (q4 as i32 - 32) as f32 * input[y_base + l + 96];
        }
    }
    acc
}

// cu108: mma_flash.cuh 가 q4k_gemv 모듈에 합류했는지 + sm_86 mma 경로 활성인지 검증.
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_flash_probe -- --ignored --nocapture
fn mma_flash_probe_loads_from_q4k_module() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_flash_probe CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let out_dev = unsafe { state.api.mem_alloc(4).expect("alloc out") };
    let mut out_arg = out_dev;
    state
        .launch_cached_gemv(
            "rnb_mma_flash_probe",
            &[(&mut out_arg as *mut u64).cast::<libc::c_void>()],
            (1, 1, 1),
            (1, 1, 1),
        )
        .expect("probe launch");
    let mut out = vec![0i32; 1];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                out.as_mut_ptr().cast::<libc::c_void>(),
                out_dev,
                4,
                state.stream,
            )
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe { state.api.mem_free(out_dev).ok() };
    assert_eq!(out[0], 800, "sm_86 mma probe should return 800");
}

// Task1 PoC: m16n8k16 QK^T fragment 레이아웃을 scalar reference 와 비교 검증.
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_qkt -- --ignored --nocapture
fn mma_qkt_tile_matches_scalar_reference() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_qkt CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let q: Vec<f32> = (0..16 * 16).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let k: Vec<f32> = (0..8 * 16).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let mut expect = vec![0f32; 16 * 8];
    for i in 0..16 {
        for j in 0..8 {
            let mut acc = 0f32;
            for d in 0..16 {
                acc += q[i * 16 + d] * k[j * 16 + d];
            }
            expect[i * 8 + j] = acc;
        }
    }
    let q16: Vec<u16> = q
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let k16: Vec<u16> = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let q_dev = unsafe { state.api.mem_alloc(q16.len() * 2).expect("q alloc") };
    let k_dev = unsafe { state.api.mem_alloc(k16.len() * 2).expect("k alloc") };
    let s_dev = unsafe { state.api.mem_alloc(16 * 8 * 4).expect("s alloc") };
    unsafe {
        state
            .api
            .memcpy_htod_async(q_dev, q16.as_ptr().cast(), q16.len() * 2, state.stream)
            .expect("h2d q");
        state
            .api
            .memcpy_htod_async(k_dev, k16.as_ptr().cast(), k16.len() * 2, state.stream)
            .expect("h2d k");
    }
    let mut s_arg = s_dev;
    let mut q_arg = q_dev;
    let mut k_arg = k_dev;
    state
        .launch_cached_gemv(
            "rnb_mma_qkt_tile",
            &[
                (&mut s_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (32, 1, 1),
        )
        .expect("qkt launch");
    let mut got = vec![0f32; 16 * 8];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(got.as_mut_ptr().cast(), s_dev, 16 * 8 * 4, state.stream)
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(q_dev).ok();
        state.api.mem_free(k_dev).ok();
        state.api.mem_free(s_dev).ok();
    }
    for idx in 0..16 * 8 {
        assert!(
            (got[idx] - expect[idx]).abs() < 1e-2,
            "idx {idx}: got {} expect {}",
            got[idx],
            expect[idx]
        );
    }
}

// Task2 PoC: QK^T + online softmax row reduce 를 scalar softmax 와 비교.
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_softmax -- --ignored --nocapture
fn mma_softmax_tile_matches_scalar() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_softmax CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let q: Vec<f32> = (0..16 * 16).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let k: Vec<f32> = (0..8 * 16).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let scale = 1.0 / (16f32).sqrt();
    let mut expect = vec![0f32; 16 * 8];
    for i in 0..16 {
        let mut row = [0f32; 8];
        for j in 0..8 {
            let mut acc = 0f32;
            for d in 0..16 {
                acc += q[i * 16 + d] * k[j * 16 + d];
            }
            row[j] = acc * scale;
        }
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        for j in 0..8 {
            expect[i * 8 + j] = (row[j] - m).exp();
        }
    }
    let q16: Vec<u16> = q
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let k16: Vec<u16> = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let q_dev = unsafe { state.api.mem_alloc(q16.len() * 2).expect("q alloc") };
    let k_dev = unsafe { state.api.mem_alloc(k16.len() * 2).expect("k alloc") };
    let p_dev = unsafe { state.api.mem_alloc(16 * 8 * 4).expect("p alloc") };
    unsafe {
        state
            .api
            .memcpy_htod_async(q_dev, q16.as_ptr().cast(), q16.len() * 2, state.stream)
            .expect("h2d q");
        state
            .api
            .memcpy_htod_async(k_dev, k16.as_ptr().cast(), k16.len() * 2, state.stream)
            .expect("h2d k");
    }
    let mut p_arg = p_dev;
    let mut q_arg = q_dev;
    let mut k_arg = k_dev;
    state
        .launch_cached_gemv(
            "rnb_mma_qkt_softmax_tile",
            &[
                (&mut p_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (32, 1, 1),
        )
        .expect("softmax launch");
    let mut got = vec![0f32; 16 * 8];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(got.as_mut_ptr().cast(), p_dev, 16 * 8 * 4, state.stream)
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(q_dev).ok();
        state.api.mem_free(k_dev).ok();
        state.api.mem_free(p_dev).ok();
    }
    for idx in 0..16 * 8 {
        assert!(
            (got[idx] - expect[idx]).abs() < 2e-2,
            "idx {idx}: got {} expect {}",
            got[idx],
            expect[idx]
        );
    }
}

// Task3 PoC: P@V (m16n8k8) + V operand/kv_head stride 를 scalar 와 비교 (2단 mma 완성).
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_pv -- --ignored --nocapture
fn mma_pv_tile_matches_scalar() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_pv CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    // P: 16x8 (softmax 결과 흉내, 양수), V: 8x16
    let p: Vec<f32> = (0..16 * 8).map(|i| ((i % 5) as f32 + 1.0) * 0.05).collect();
    let v: Vec<f32> = (0..8 * 16).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let mut expect = vec![0f32; 16 * 16];
    for q in 0..16 {
        for hd in 0..16 {
            let mut acc = 0f32;
            for t in 0..8 {
                acc += p[q * 8 + t] * v[t * 16 + hd];
            }
            expect[q * 16 + hd] = acc;
        }
    }
    let v16: Vec<u16> = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let p_dev = unsafe { state.api.mem_alloc(p.len() * 4).expect("p alloc") };
    let v_dev = unsafe { state.api.mem_alloc(v16.len() * 2).expect("v alloc") };
    let o_dev = unsafe { state.api.mem_alloc(16 * 16 * 4).expect("o alloc") };
    unsafe {
        state
            .api
            .memcpy_htod_async(p_dev, p.as_ptr().cast(), p.len() * 4, state.stream)
            .expect("h2d p");
        state
            .api
            .memcpy_htod_async(v_dev, v16.as_ptr().cast(), v16.len() * 2, state.stream)
            .expect("h2d v");
    }
    let mut o_arg = o_dev;
    let mut p_arg = p_dev;
    let mut v_arg = v_dev;
    state
        .launch_cached_gemv(
            "rnb_mma_pv_tile",
            &[
                (&mut o_arg as *mut u64).cast::<libc::c_void>(),
                (&mut p_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (32, 1, 1),
        )
        .expect("pv launch");
    let mut got = vec![0f32; 16 * 16];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(got.as_mut_ptr().cast(), o_dev, 16 * 16 * 4, state.stream)
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(p_dev).ok();
        state.api.mem_free(v_dev).ok();
        state.api.mem_free(o_dev).ok();
    }
    for idx in 0..16 * 16 {
        assert!(
            (got[idx] - expect[idx]).abs() < 3e-2,
            "idx {idx}: got {} expect {}",
            got[idx],
            expect[idx]
        );
    }
}

// Task4 PoC: flash 루프(online rescale + row별 window mask) 를 scalar flash 와 비교.
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_flash_loop -- --ignored --nocapture
fn mma_flash_loop_matches_scalar_window() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_flash_loop CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let n = 16usize; // query=16, kv=16, head_dim=16
    let window = 8i32;
    let q: Vec<f32> = (0..n * 16).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let k: Vec<f32> = (0..n * 16).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let v: Vec<f32> = (0..n * 16).map(|i| ((i % 9) as f32 - 4.0) * 0.08).collect();
    let scale = 1.0 / (16f32).sqrt();
    let mut expect = vec![0f32; n * 16];
    for qi in 0..n {
        let pos = qi as i32;
        let start = if pos + 1 >= window {
            pos + 1 - window
        } else {
            0
        } as usize;
        let mut scores: Vec<(usize, f32)> = Vec::new();
        for j in start..=qi {
            let mut acc = 0f32;
            for d in 0..16 {
                acc += q[qi * 16 + d] * k[j * 16 + d];
            }
            scores.push((j, acc * scale));
        }
        let m = scores.iter().map(|&(_, s)| s).fold(f32::MIN, f32::max);
        let mut l = 0f32;
        let mut o = [0f32; 16];
        for &(j, s) in &scores {
            let e = (s - m).exp();
            l += e;
            for d in 0..16 {
                o[d] += e * v[j * 16 + d];
            }
        }
        for d in 0..16 {
            expect[qi * 16 + d] = o[d] / l;
        }
    }
    let q16: Vec<u16> = q
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let k16: Vec<u16> = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let v16: Vec<u16> = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let q_dev = unsafe { state.api.mem_alloc(q16.len() * 2).expect("q") };
    let k_dev = unsafe { state.api.mem_alloc(k16.len() * 2).expect("k") };
    let v_dev = unsafe { state.api.mem_alloc(v16.len() * 2).expect("v") };
    let o_dev = unsafe { state.api.mem_alloc(n * 16 * 4).expect("o") };
    unsafe {
        state
            .api
            .memcpy_htod_async(q_dev, q16.as_ptr().cast(), q16.len() * 2, state.stream)
            .expect("h2d q");
        state
            .api
            .memcpy_htod_async(k_dev, k16.as_ptr().cast(), k16.len() * 2, state.stream)
            .expect("h2d k");
        state
            .api
            .memcpy_htod_async(v_dev, v16.as_ptr().cast(), v16.len() * 2, state.stream)
            .expect("h2d v");
    }
    let mut o_arg = o_dev;
    let mut q_arg = q_dev;
    let mut k_arg = k_dev;
    let mut v_arg = v_dev;
    let mut kv_len_arg = n as i32;
    let mut window_arg = window;
    let mut q_start_arg = 0i32;
    state
        .launch_cached_gemv(
            "rnb_mma_flash_tile",
            &[
                (&mut o_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut i32).cast::<libc::c_void>(),
                (&mut window_arg as *mut i32).cast::<libc::c_void>(),
                (&mut q_start_arg as *mut i32).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (32, 1, 1),
        )
        .expect("flash launch");
    let mut got = vec![0f32; n * 16];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(got.as_mut_ptr().cast(), o_dev, n * 16 * 4, state.stream)
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(q_dev).ok();
        state.api.mem_free(k_dev).ok();
        state.api.mem_free(v_dev).ok();
        state.api.mem_free(o_dev).ok();
    }
    for idx in 0..n * 16 {
        assert!(
            (got[idx] - expect[idx]).abs() < 3e-2,
            "idx {idx} (q{} d{}): got {} expect {}",
            idx / 16,
            idx % 16,
            got[idx],
            expect[idx]
        );
    }
}

// Task5: production hd256 mma flash 를 scalar masked attention(GQA, row별 pos) 와 비교.
#[test]
#[ignore] // RTX 3080: cargo test -p rnb-backend-cuda mma_flash_hd256 -- --ignored --nocapture
fn mma_flash_hd256_matches_scalar() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_flash_hd256 CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let seq = 128usize;
    let hd = 256usize;
    let heads = 8usize;
    let window = 512i32;
    let scale = 1.0 / (hd as f32).sqrt();
    let q: Vec<f32> = (0..seq * heads * hd)
        .map(|i| (((i * 7) % 13) as f32 - 6.0) * 0.02)
        .collect();
    let k: Vec<f32> = (0..seq * hd)
        .map(|i| (((i * 5) % 11) as f32 - 5.0) * 0.02)
        .collect();
    let v: Vec<f32> = (0..seq * hd)
        .map(|i| (((i * 3) % 9) as f32 - 4.0) * 0.02)
        .collect();
    let mut expect = vec![0f32; seq * heads * hd];
    for h in 0..heads {
        for qi in 0..seq {
            let pos = qi as i32;
            let start = (pos + 1 - window).max(0) as usize;
            let mut scores: Vec<(usize, f32)> = Vec::new();
            for j in start..=qi {
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += q[qi * heads * hd + h * hd + d] * k[j * hd + d];
                }
                scores.push((j, acc * scale));
            }
            let m = scores.iter().map(|&(_, s)| s).fold(f32::MIN, f32::max);
            let mut l = 0f32;
            let mut o = vec![0f32; hd];
            for &(j, s) in &scores {
                let e = (s - m).exp();
                l += e;
                for d in 0..hd {
                    o[d] += e * v[j * hd + d];
                }
            }
            for d in 0..hd {
                expect[qi * heads * hd + h * hd + d] = o[d] / l;
            }
        }
    }
    // q_post_dev 는 f32 (production), k/v 는 f16 (k_bits_dev)
    let k16: Vec<u16> = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let v16: Vec<u16> = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let q_dev = unsafe { state.api.mem_alloc(q.len() * 4).expect("q") };
    let k_dev = unsafe { state.api.mem_alloc(k16.len() * 2).expect("k") };
    let v_dev = unsafe { state.api.mem_alloc(v16.len() * 2).expect("v") };
    let o_dev = unsafe { state.api.mem_alloc(seq * heads * hd * 4).expect("o") };
    unsafe {
        state
            .api
            .memcpy_htod_async(q_dev, q.as_ptr().cast(), q.len() * 4, state.stream)
            .expect("h2d q");
        state
            .api
            .memcpy_htod_async(k_dev, k16.as_ptr().cast(), k16.len() * 2, state.stream)
            .expect("h2d k");
        state
            .api
            .memcpy_htod_async(v_dev, v16.as_ptr().cast(), v16.len() * 2, state.stream)
            .expect("h2d v");
    }
    let mut o_arg = o_dev;
    let mut q_arg = q_dev;
    let mut k_arg = k_dev;
    let mut v_arg = v_dev;
    let mut seq_arg = seq as u32;
    let mut kv_len_arg = seq as u32;
    let mut heads_arg = heads as u32;
    let mut kv_heads_arg = 1u32;
    let mut scale_arg = scale;
    let mut window_arg = window as u32;
    state
        .launch_cached_gemv(
            "rnb_attention_prefill_flash_hd256_window_mma",
            &[
                (&mut o_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut window_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((seq as u32) + 63) / 64, heads as u32, 1),
            (128, 1, 1),
        )
        .expect("hd256 launch");
    let mut got = vec![0f32; seq * heads * hd];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                got.as_mut_ptr().cast(),
                o_dev,
                seq * heads * hd * 4,
                state.stream,
            )
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(q_dev).ok();
        state.api.mem_free(k_dev).ok();
        state.api.mem_free(v_dev).ok();
        state.api.mem_free(o_dev).ok();
    }
    let mut dot = 0f64;
    let mut ng = 0f64;
    let mut ne = 0f64;
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        dot += got[i] as f64 * expect[i] as f64;
        ng += got[i] as f64 * got[i] as f64;
        ne += expect[i] as f64 * expect[i] as f64;
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    let cos = dot / (ng.sqrt() * ne.sqrt());
    eprintln!("hd256 mma vs scalar: cosine={cos:.6} max_abs={max_abs:.5}");
    assert!(cos > 0.999, "cosine {cos} too low (max_abs {max_abs})");
}

// cu113: hd512 FULL attention mma 커널 vs scalar reference (causal, window 없음).
#[test]
#[ignore = "requires CUDA device (RTX 3080 sm_86)"]
fn mma_flash_hd512_matches_scalar() {
    let mut state = match CudaState::open() {
        Ok(state) => state,
        Err(err) if cuda_driver_unavailable_for_test(&err) => {
            eprintln!("skipping mma_flash_hd512 CUDA test: {err}");
            return;
        }
        Err(err) => panic!("cuda open: {err}"),
    };
    let seq = 128usize;
    let hd = 512usize;
    let heads = 8usize;
    let scale = 1.0 / (hd as f32).sqrt();
    let q: Vec<f32> = (0..seq * heads * hd)
        .map(|i| (((i * 7) % 13) as f32 - 6.0) * 0.02)
        .collect();
    let k: Vec<f32> = (0..seq * hd)
        .map(|i| (((i * 5) % 11) as f32 - 5.0) * 0.02)
        .collect();
    let v: Vec<f32> = (0..seq * hd)
        .map(|i| (((i * 3) % 9) as f32 - 4.0) * 0.02)
        .collect();
    let mut expect = vec![0f32; seq * heads * hd];
    for h in 0..heads {
        for qi in 0..seq {
            // FULL causal: j in 0..=qi (window 없음)
            let mut scores: Vec<(usize, f32)> = Vec::new();
            for j in 0..=qi {
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += q[qi * heads * hd + h * hd + d] * k[j * hd + d];
                }
                scores.push((j, acc * scale));
            }
            let m = scores.iter().map(|&(_, s)| s).fold(f32::MIN, f32::max);
            let mut l = 0f32;
            let mut o = vec![0f32; hd];
            for &(j, s) in &scores {
                let e = (s - m).exp();
                l += e;
                for d in 0..hd {
                    o[d] += e * v[j * hd + d];
                }
            }
            for d in 0..hd {
                expect[qi * heads * hd + h * hd + d] = o[d] / l;
            }
        }
    }
    let k16: Vec<u16> = k
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let v16: Vec<u16> = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let q_dev = unsafe { state.api.mem_alloc(q.len() * 4).expect("q") };
    let k_dev = unsafe { state.api.mem_alloc(k16.len() * 2).expect("k") };
    let v_dev = unsafe { state.api.mem_alloc(v16.len() * 2).expect("v") };
    let o_dev = unsafe { state.api.mem_alloc(seq * heads * hd * 4).expect("o") };
    unsafe {
        state
            .api
            .memcpy_htod_async(q_dev, q.as_ptr().cast(), q.len() * 4, state.stream)
            .expect("h2d q");
        state
            .api
            .memcpy_htod_async(k_dev, k16.as_ptr().cast(), k16.len() * 2, state.stream)
            .expect("h2d k");
        state
            .api
            .memcpy_htod_async(v_dev, v16.as_ptr().cast(), v16.len() * 2, state.stream)
            .expect("h2d v");
    }
    let mut o_arg = o_dev;
    let mut q_arg = q_dev;
    let mut k_arg = k_dev;
    let mut v_arg = v_dev;
    let mut seq_arg = seq as u32;
    let mut kv_len_arg = seq as u32;
    let mut heads_arg = heads as u32;
    let mut kv_heads_arg = 1u32;
    let mut scale_arg = scale;
    state
        .launch_cached_gemv(
            "rnb_attention_prefill_flash_hd512_mma",
            &[
                (&mut o_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (((seq as u32) + 63) / 64, heads as u32, 1),
            (128, 1, 1),
        )
        .expect("hd512 launch");
    let mut got = vec![0f32; seq * heads * hd];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                got.as_mut_ptr().cast(),
                o_dev,
                seq * heads * hd * 4,
                state.stream,
            )
            .expect("d2h");
    }
    state.stream_synchronize().expect("sync");
    unsafe {
        state.api.mem_free(q_dev).ok();
        state.api.mem_free(k_dev).ok();
        state.api.mem_free(v_dev).ok();
        state.api.mem_free(o_dev).ok();
    }
    let mut dot = 0f64;
    let mut ng = 0f64;
    let mut ne = 0f64;
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        dot += got[i] as f64 * expect[i] as f64;
        ng += got[i] as f64 * got[i] as f64;
        ne += expect[i] as f64 * expect[i] as f64;
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    let cos = dot / (ng.sqrt() * ne.sqrt());
    eprintln!("hd512 mma vs scalar: cosine={cos:.6} max_abs={max_abs:.5}");
    assert!(cos > 0.999, "cosine {cos} too low (max_abs {max_abs})");
}

#[test]
fn kvarn_attention_decode_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let head_dim = 128usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let config = rnb_cpu::quantize::kvarn::KvarnConfig::K4_V4_G64;
    let sink_len = config.sink_tokens;
    let tail_len = 5usize;
    let kv_len = sink_len + config.group + tail_len;
    let row_width = num_kv_heads * head_dim;
    let mut key = vec![0u16; kv_len * row_width];
    let mut value = vec![0u16; kv_len * row_width];
    for (index, bits) in key.iter_mut().enumerate() {
        let sample = ((index * 17 % 101) as f32 - 50.0) * 0.00625;
        *bits = half::f16::from_f32(sample).to_bits();
    }
    for (index, bits) in value.iter_mut().enumerate() {
        let sample = ((index * 29 % 113) as f32 - 56.0) * 0.0078125;
        *bits = half::f16::from_f32(sample).to_bits();
    }
    let block_start = sink_len * row_width;
    let block_end = block_start + config.group * row_width;
    let block = rnb_cpu::quantize::kvarn::KvarnBlock::quantize(
        config,
        num_kv_heads,
        head_dim,
        &key[block_start..block_end],
        &value[block_start..block_end],
    )
    .expect("quantize KVarN block");
    let layout =
        rnb_cpu::quantize::kvarn::KvarnDeviceRecordLayout::new(config, num_kv_heads, head_dim)
            .expect("KVarN device layout");
    let mut packed = Vec::new();
    block.append_device_record(&mut packed);
    let blocks = [block];
    let tail_start = sink_len + config.group;
    let view = rnb_cpu::quantize::kvarn::KvarnKvView {
        config,
        num_kv_heads,
        head_dim,
        sink_key: &key[..block_start],
        sink_value: &value[..block_start],
        blocks: &blocks,
        device_layout: layout,
        device_blocks: &packed,
        tail_start,
        tail_key: &key[block_end..],
        tail_value: &value[block_end..],
        len: kv_len,
    };
    let query = (0..num_heads * head_dim)
        .map(|index| ((index * 13 % 79) as f32 - 39.0) * 0.009)
        .collect::<Vec<_>>();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut expected = vec![0.0f32; query.len()];
    rnb_cpu::quantize::kvarn::attention_decode(
        &query,
        view,
        &mut expected,
        num_heads,
        scale,
        None,
        None,
    );

    let output_bytes = expected.len() * std::mem::size_of::<f32>();
    let output_dev = match acquire_decode_attn_out_carrier(output_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            eprintln!("skipping CUDA KVarN attention test: {err}");
            return;
        }
    };
    let request = rnb_backend_api::KvarnDecodeRequest::new(
        10_901,
        &query,
        &packed,
        view.sink_key,
        view.sink_value,
        view.tail_key,
        view.tail_value,
        kv_len,
        tail_start,
        num_heads,
        num_kv_heads,
        head_dim,
        config.key_bits,
        config.value_bits,
        config.group,
        config.sink_tokens,
        layout.block_bytes,
        scale,
        None,
        None,
    );
    if let Err(err) = attention_decode_kvarn_to_device(request, output_dev) {
        eprintln!("skipping CUDA KVarN attention test: {err}");
        return;
    }
    let mut actual = vec![0.0f32; expected.len()];
    if let Err(err) = download_from_decode_hidden_carrier(output_dev, &mut actual) {
        eprintln!("skipping CUDA KVarN attention download: {err}");
        return;
    }
    if let Err(err) = sync_decode_stream() {
        eprintln!("skipping CUDA KVarN attention sync: {err}");
        return;
    }
    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 2.0e-3,
        "CUDA KVarN attention diverged from CPU reference: max_diff={max_diff}"
    );
    let request = rnb_backend_api::KvarnDecodeRequest::new(
        10_902,
        &query,
        &packed,
        view.sink_key,
        view.sink_value,
        view.tail_key,
        view.tail_value,
        kv_len,
        tail_start,
        num_heads,
        num_kv_heads,
        head_dim,
        config.key_bits,
        config.value_bits,
        config.group,
        config.sink_tokens,
        layout.block_bytes,
        scale,
        None,
        None,
    );
    let host_actual = attention_decode_kvarn(request).expect("CUDA KVarN host output");
    let host_max_diff = host_actual
        .iter()
        .zip(expected.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        host_max_diff < 2.0e-3,
        "CUDA KVarN host output diverged from CPU reference: max_diff={host_max_diff}"
    );
}

#[test]
fn gdn_prepare_delta_gate_beta_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut alpha = vec![-80.0f32, -1.0, 0.0, 0.5, 10.0, 80.0];
    let mut beta = vec![-80.0f32, -1.5, 0.0, 0.75, 12.0, 80.0];
    let dt_bias = vec![0.25f32, -0.5];
    let ssm_a = vec![-0.75f32, -1.25];
    let expected_alpha = alpha
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let head = idx % dt_bias.len();
            (1.0 + (value + dt_bias[head]).exp()).ln() * ssm_a[head]
        })
        .collect::<Vec<_>>();
    let expected_beta = beta
        .iter()
        .map(|&value| 1.0 / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();

    if let Err(err) =
        crate::runtime::gdn_prepare_delta_gate_beta_f32(&mut alpha, &mut beta, &dt_bias, &ssm_a, 2)
    {
        eprintln!("skipping CUDA GDN delta gate test: {err}");
        return;
    }

    assert_close_rows("GDN delta gate", &alpha, &expected_alpha, 2.0e-5);
    assert_close_rows("GDN delta beta", &beta, &expected_beta, 2.0e-5);
}

#[test]
fn gdn_prepare_delta_gate_beta_rejects_shape_mismatch() {
    let mut alpha = vec![0.0f32; 3];
    let mut beta = vec![0.0f32; 3];
    let err = crate::runtime::gdn_prepare_delta_gate_beta_f32(
        &mut alpha,
        &mut beta,
        &[0.0, 0.0],
        &[0.0, 0.0],
        2,
    )
    .expect_err("non-divisible GDN gate rows must be rejected");
    assert!(err.contains("shape mismatch"), "unexpected error: {err}");
}
