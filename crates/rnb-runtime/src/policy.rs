mod base;
mod debug;
mod gemma;
mod general;
mod memory;
mod moe;
mod session_cache;

pub use base::*;
pub use debug::*;
pub use gemma::*;
pub use general::*;
pub use memory::*;
pub use moe::*;
pub use session_cache::*;

#[cfg(test)]
mod policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn layer_matches_spec_accepts_ranges_and_values() {
        assert!(layer_matches_spec("1,3-5,8", 1));
        assert!(layer_matches_spec("1,3-5,8", 4));
        assert!(layer_matches_spec("1,3-5,8", 8));
        assert!(!layer_matches_spec("1,3-5,8", 2));
        assert!(!layer_matches_spec("5-3", 4));
        assert!(!layer_matches_spec("bad", 4));
    }

    #[test]
    fn dump_bin_layer_filter_accepts_ranges_and_keeps_global_records() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_DUMP_BIN_LAYER_FILTER");
        assert!(dump_bin_layer_enabled(0));
        assert!(dump_bin_layer_enabled(usize::MAX));

        std::env::set_var("RNB_DUMP_BIN_LAYER_FILTER", "11-17,41");
        assert!(!dump_bin_layer_enabled(10));
        assert!(dump_bin_layer_enabled(11));
        assert!(dump_bin_layer_enabled(17));
        assert!(!dump_bin_layer_enabled(18));
        assert!(dump_bin_layer_enabled(41));
        assert!(dump_bin_layer_enabled(usize::MAX));
        std::env::remove_var("RNB_DUMP_BIN_LAYER_FILTER");
    }

    #[test]
    fn moe_section_decode_is_diagnostic_opt_in() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_MOE_DECODE");
        assert!(moe_section_decode_disabled());
        std::env::set_var("RNB_MOE_DECODE", "0");
        assert!(moe_section_decode_disabled());
        std::env::set_var("RNB_MOE_DECODE", "1");
        assert!(!moe_section_decode_disabled());
        std::env::remove_var("RNB_MOE_DECODE");
    }

    #[test]
    fn glm_dsa_batch_prefill_uses_expert_coverage_and_allows_override() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_GLM_DSA_BATCH_PREFILL");
        assert!(!glm_dsa_batch_prefill_enabled(19, 256, 8));
        assert!(glm_dsa_batch_prefill_enabled(32, 256, 8));
        assert!(glm_dsa_batch_prefill_enabled(61, 256, 8));

        std::env::set_var("RNB_GLM_DSA_BATCH_PREFILL", "1");
        assert!(glm_dsa_batch_prefill_enabled(19, 256, 8));
        std::env::set_var("RNB_GLM_DSA_BATCH_PREFILL", "0");
        assert!(!glm_dsa_batch_prefill_enabled(61, 256, 8));
        std::env::remove_var("RNB_GLM_DSA_BATCH_PREFILL");
    }

    #[test]
    fn shadow_weights_are_diagnostic_opt_in() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_SHADOW_WEIGHTS");
        assert!(!shadow_weights_requested());
        std::env::set_var("RNB_SHADOW_WEIGHTS", "1");
        assert!(shadow_weights_requested());
        std::env::remove_var("RNB_SHADOW_WEIGHTS");
    }

    #[test]
    fn gemma4_moe_expert_major_defaults_on_and_supports_opt_out() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_GEMMA4_MOE_EXPERT_MAJOR_OFF");
        assert!(gemma4_moe_expert_major_enabled());

        std::env::set_var("RNB_GEMMA4_MOE_EXPERT_MAJOR_OFF", "1");
        assert!(!gemma4_moe_expert_major_enabled());
        std::env::remove_var("RNB_GEMMA4_MOE_EXPERT_MAJOR_OFF");
    }

    #[test]
    fn moe_trace_policy_helpers_apply_existing_defaults() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_MOE_ROUTE_TRACE_FILE");
        std::env::remove_var("RNB_MOE_PREDICTOR_TRACE_FILE");
        std::env::remove_var("RNB_MOE_PREDICTOR_TRACE_TOP_N");
        assert!(!moe_route_trace_enabled());
        assert!(!moe_predictor_trace_enabled());
        assert_eq!(moe_predictor_trace_top_n_limit(), 16);

        std::env::set_var("RNB_MOE_ROUTE_TRACE_FILE", "route.jsonl");
        std::env::set_var("RNB_MOE_PREDICTOR_TRACE_FILE", "predictor.jsonl");
        std::env::set_var("RNB_MOE_PREDICTOR_TRACE_TOP_N", "0");
        assert!(moe_route_trace_enabled());
        assert!(moe_predictor_trace_enabled());
        assert_eq!(moe_predictor_trace_top_n_limit(), 16);

        std::env::set_var("RNB_MOE_PREDICTOR_TRACE_TOP_N", "7");
        assert_eq!(moe_predictor_trace_top_n_limit(), 7);

        std::env::remove_var("RNB_MOE_ROUTE_TRACE_FILE");
        std::env::remove_var("RNB_MOE_PREDICTOR_TRACE_FILE");
        std::env::remove_var("RNB_MOE_PREDICTOR_TRACE_TOP_N");
    }

    #[test]
    fn moe_adaptive_top_p_accepts_only_open_unit_interval() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_MOE_ADAPTIVE_TOP_P");
        assert_eq!(moe_adaptive_top_p(), None);
        for invalid in ["0", "1", "-0.1", "NaN", "inf"] {
            std::env::set_var("RNB_MOE_ADAPTIVE_TOP_P", invalid);
            assert_eq!(moe_adaptive_top_p(), None);
        }
        std::env::set_var("RNB_MOE_ADAPTIVE_TOP_P", "0.85");
        assert_eq!(moe_adaptive_top_p(), Some(0.85));
        std::env::remove_var("RNB_MOE_ADAPTIVE_TOP_P");
    }

    #[test]
    fn q2q3_sparse_cuda_uses_auto_policy_with_explicit_force_and_opt_out() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        const ENV: &str = "RNB_CUDA_Q2K_Q3K_SPARSE_MOE";

        std::env::remove_var(ENV);
        assert!(!cuda_q2k_q3k_sparse_moe_enabled(false));
        assert!(cuda_q2k_q3k_sparse_moe_enabled(true));

        for disabled in ["0", "false", "off", "no"] {
            std::env::set_var(ENV, disabled);
            assert!(!cuda_q2k_q3k_sparse_moe_enabled(true));
        }

        std::env::set_var(ENV, "1");
        assert!(cuda_q2k_q3k_sparse_moe_enabled(false));
        std::env::remove_var(ENV);
    }

    #[test]
    fn q2q3_mixed_resident_cpu_uses_auto_policy_with_explicit_force_and_opt_out() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        const ENV: &str = "RNB_CUDA_Q2Q3_MIXED_RESIDENT_CPU";

        std::env::remove_var(ENV);
        assert!(!cuda_q2k_q3k_mixed_resident_cpu_enabled(false));
        assert!(cuda_q2k_q3k_mixed_resident_cpu_enabled(true));

        for disabled in ["0", "false", "off", "no"] {
            std::env::set_var(ENV, disabled);
            assert!(!cuda_q2k_q3k_mixed_resident_cpu_enabled(true));
        }

        for enabled in ["1", "true", "on", "yes"] {
            std::env::set_var(ENV, enabled);
            assert!(cuda_q2k_q3k_mixed_resident_cpu_enabled(false));
        }
        std::env::remove_var(ENV);
    }

    #[test]
    fn cuda_moe_diagnostic_flags_require_exact_one() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        for (env, enabled) in [
            (
                "RNB_CUDA_CACHE_TRACE",
                cuda_cache_trace_enabled as fn() -> bool,
            ),
            (
                "RNB_CUDA_DECODE_MOE_COMBINED",
                cuda_decode_moe_combined_enabled as fn() -> bool,
            ),
        ] {
            std::env::remove_var(env);
            assert!(!enabled());
            for value in ["0", "true", "yes"] {
                std::env::set_var(env, value);
                assert!(!enabled());
            }
            std::env::set_var(env, "1");
            assert!(enabled());
            std::env::remove_var(env);
        }
    }

    #[test]
    fn spec_profile_policy_is_independent_from_global_profile() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_PROFILE");
        std::env::remove_var("RNB_SPEC_PROFILE");
        assert!(!profiling_enabled());
        assert!(!spec_profile_enabled());

        std::env::set_var("RNB_SPEC_PROFILE", "1");
        assert!(!profiling_enabled());
        assert!(spec_profile_enabled());

        std::env::remove_var("RNB_SPEC_PROFILE");
    }

    #[test]
    fn mtp_policy_flags_keep_expected_defaults() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_MTP_OUTPUT_ARGMAX");
        std::env::remove_var("RNB_MTP_DECODE_BLOCK");
        std::env::remove_var("RNB_MTP_BATCH_VERIFY");
        std::env::remove_var("RNB_MTP_FAST_RETAIN");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        std::env::remove_var("RNB_MTP_DRAFT_ONLY");
        std::env::remove_var("RNB_MTP_SHADOW_PRECOMPUTE");
        std::env::remove_var("RNB_MTP_RUNWAY_MAX_EXTRA");
        std::env::remove_var("RNB_MTP_DUMP_TOPK");
        std::env::remove_var("RNB_SPEC_MTP_SEQUENTIAL_MULTI");

        assert!(!mtp_output_argmax_enabled());
        assert!(mtp_decode_block_enabled());
        assert!(!mtp_batch_verify_enabled());
        assert!(!mtp_batch_verify_disabled());
        assert!(!mtp_fast_retain_enabled());
        assert!(!mtp_device_verify_enabled());
        assert!(!mtp_draft_only_enabled());
        assert!(!mtp_shadow_precompute_enabled());
        assert_eq!(mtp_runway_max_extra(), None);
        assert_eq!(mtp_dump_topk(), None);
        assert!(!spec_mtp_sequential_multi_enabled());

        std::env::set_var("RNB_MTP_OUTPUT_ARGMAX", "1");
        std::env::set_var("RNB_MTP_DECODE_BLOCK", "0");
        std::env::set_var("RNB_MTP_BATCH_VERIFY", "1");
        std::env::set_var("RNB_MTP_FAST_RETAIN", "1");
        std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
        std::env::set_var("RNB_MTP_DRAFT_ONLY", "1");
        std::env::set_var("RNB_MTP_SHADOW_PRECOMPUTE", "1");
        std::env::set_var("RNB_MTP_RUNWAY_MAX_EXTRA", "4");
        std::env::set_var("RNB_MTP_DUMP_TOPK", "3");
        std::env::set_var("RNB_SPEC_MTP_SEQUENTIAL_MULTI", "1");

        assert!(mtp_output_argmax_enabled());
        assert!(!mtp_decode_block_enabled());
        assert!(mtp_batch_verify_enabled());
        assert!(!mtp_batch_verify_disabled());
        assert!(mtp_fast_retain_enabled());
        assert!(mtp_device_verify_enabled());
        assert!(mtp_draft_only_enabled());
        assert!(mtp_shadow_precompute_enabled());
        assert_eq!(mtp_runway_max_extra(), Some(4));
        assert_eq!(mtp_dump_topk(), Some(3));
        assert!(spec_mtp_sequential_multi_enabled());

        std::env::set_var("RNB_MTP_OUTPUT_ARGMAX", "off");
        assert!(!mtp_output_argmax_enabled());
        std::env::set_var("RNB_MTP_BATCH_VERIFY", "0");
        assert!(!mtp_batch_verify_enabled());
        assert!(mtp_batch_verify_disabled());
        std::env::set_var("RNB_MTP_FAST_RETAIN", "off");
        assert!(!mtp_fast_retain_enabled());
        std::env::set_var("RNB_MTP_DEVICE_VERIFY", "false");
        assert!(!mtp_device_verify_enabled());
        std::env::set_var("RNB_MTP_DRAFT_ONLY", "false");
        assert!(!mtp_draft_only_enabled());
        std::env::set_var("RNB_MTP_SHADOW_PRECOMPUTE", "false");
        assert!(!mtp_shadow_precompute_enabled());
        std::env::set_var("RNB_MTP_DUMP_TOPK", "bad");
        assert_eq!(mtp_dump_topk(), Some(8));

        std::env::remove_var("RNB_MTP_OUTPUT_ARGMAX");
        std::env::remove_var("RNB_MTP_DECODE_BLOCK");
        std::env::remove_var("RNB_MTP_BATCH_VERIFY");
        std::env::remove_var("RNB_MTP_FAST_RETAIN");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        std::env::remove_var("RNB_MTP_DRAFT_ONLY");
        std::env::remove_var("RNB_MTP_SHADOW_PRECOMPUTE");
        std::env::remove_var("RNB_MTP_RUNWAY_MAX_EXTRA");
        std::env::remove_var("RNB_MTP_DUMP_TOPK");
        std::env::remove_var("RNB_SPEC_MTP_SEQUENTIAL_MULTI");
    }

    #[test]
    fn mtp_device_verify_defaults_draft_argmax_to_backend() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_MTP_OUTPUT_ARGMAX");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        assert!(!mtp_output_argmax_enabled());

        std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
        assert!(mtp_output_argmax_enabled());

        std::env::set_var("RNB_MTP_OUTPUT_ARGMAX", "0");
        assert!(!mtp_output_argmax_enabled());

        std::env::remove_var("RNB_MTP_OUTPUT_ARGMAX");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
    }

    #[test]
    fn cuda_nemotron_device_hidden_carrier_policy_is_explicit() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_CUDA_NEMOTRON_DEVICE_HIDDEN_CARRIER");
        assert!(!cuda_nemotron_device_hidden_carrier_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_HIDDEN_CARRIER", "1");
        assert!(cuda_nemotron_device_hidden_carrier_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_HIDDEN_CARRIER", "0");
        assert!(!cuda_nemotron_device_hidden_carrier_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_HIDDEN_CARRIER", "false");
        assert!(!cuda_nemotron_device_hidden_carrier_enabled());

        std::env::remove_var("RNB_CUDA_NEMOTRON_DEVICE_HIDDEN_CARRIER");
    }

    #[test]
    fn cuda_nemotron_device_route_pack_policy_is_explicit() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_CUDA_NEMOTRON_DEVICE_ROUTE_PACK");
        assert!(!cuda_nemotron_device_route_pack_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_ROUTE_PACK", "1");
        assert!(cuda_nemotron_device_route_pack_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_ROUTE_PACK", "0");
        assert!(!cuda_nemotron_device_route_pack_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_ROUTE_PACK", "false");
        assert!(!cuda_nemotron_device_route_pack_enabled());

        std::env::remove_var("RNB_CUDA_NEMOTRON_DEVICE_ROUTE_PACK");
    }

    #[test]
    fn cuda_nemotron_device_prefill_v2_policy_is_explicit_and_subflags_are_safe() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        for key in [
            "RNB_CUDA_NEMOTRON_DEVICE_PREFILL_V2",
            "RNB_CUDA_NEMOTRON_PREFILL_WORKSPACE",
            "RNB_CUDA_NEMOTRON_Q8_SPARSE_EXPERT",
            "RNB_CUDA_NEMOTRON_ATTENTION_DEVICE_INPUT",
        ] {
            std::env::remove_var(key);
        }

        assert!(!cuda_nemotron_device_prefill_v2_enabled());
        assert!(cuda_nemotron_prefill_workspace_enabled());
        assert!(cuda_nemotron_q8_sparse_expert_enabled());
        assert!(!cuda_nemotron_attention_device_input_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_PREFILL_V2", "1");
        assert!(cuda_nemotron_device_prefill_v2_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_DEVICE_PREFILL_V2", "0");
        assert!(!cuda_nemotron_device_prefill_v2_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_WORKSPACE", "0");
        assert!(!cuda_nemotron_prefill_workspace_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_WORKSPACE", "false");
        assert!(!cuda_nemotron_prefill_workspace_enabled());

        std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_WORKSPACE");
        assert!(cuda_nemotron_prefill_workspace_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_Q8_SPARSE_EXPERT", "0");
        assert!(!cuda_nemotron_q8_sparse_expert_enabled());

        std::env::remove_var("RNB_CUDA_NEMOTRON_Q8_SPARSE_EXPERT");
        assert!(cuda_nemotron_q8_sparse_expert_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_ATTENTION_DEVICE_INPUT", "1");
        assert!(cuda_nemotron_attention_device_input_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_ATTENTION_DEVICE_INPUT", "false");
        assert!(!cuda_nemotron_attention_device_input_enabled());

        for key in [
            "RNB_CUDA_NEMOTRON_DEVICE_PREFILL_V2",
            "RNB_CUDA_NEMOTRON_PREFILL_WORKSPACE",
            "RNB_CUDA_NEMOTRON_Q8_SPARSE_EXPERT",
            "RNB_CUDA_NEMOTRON_ATTENTION_DEVICE_INPUT",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn cuda_nemotron_carrier_drift_trace_policy_is_explicit() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_CUDA_NEMOTRON_CARRIER_ROUTE_TRACE");
        std::env::remove_var("RNB_CUDA_NEMOTRON_CARRIER_TENSOR_TRACE");
        assert!(!cuda_nemotron_carrier_route_trace_enabled());
        assert!(!cuda_nemotron_carrier_tensor_trace_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_CARRIER_ROUTE_TRACE", "1");
        std::env::set_var("RNB_CUDA_NEMOTRON_CARRIER_TENSOR_TRACE", "1");
        assert!(cuda_nemotron_carrier_route_trace_enabled());
        assert!(cuda_nemotron_carrier_tensor_trace_enabled());

        std::env::set_var("RNB_CUDA_NEMOTRON_CARRIER_ROUTE_TRACE", "0");
        std::env::set_var("RNB_CUDA_NEMOTRON_CARRIER_TENSOR_TRACE", "false");
        assert!(!cuda_nemotron_carrier_route_trace_enabled());
        assert!(!cuda_nemotron_carrier_tensor_trace_enabled());

        std::env::remove_var("RNB_CUDA_NEMOTRON_CARRIER_ROUTE_TRACE");
        std::env::remove_var("RNB_CUDA_NEMOTRON_CARRIER_TENSOR_TRACE");
    }

    #[test]
    fn speculative_policy_flags_require_one() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_SPEC_FORCE_BATCH_VERIFY");
        std::env::remove_var("RNB_SPEC_DECODE_FAST_WINDOW");
        std::env::remove_var("RNB_SPEC_BATCH_NO_BONUS");
        std::env::remove_var("RNB_SPEC_BATCH_PREFIX_SNAPSHOT");

        assert!(!spec_force_batch_verify_enabled());
        assert!(!spec_decode_fast_window_enabled());
        assert!(!spec_batch_no_bonus_enabled());
        assert_eq!(spec_batch_no_bonus_override(), None);
        assert!(!spec_batch_prefix_snapshot_enabled());

        std::env::set_var("RNB_SPEC_FORCE_BATCH_VERIFY", "1");
        std::env::set_var("RNB_SPEC_DECODE_FAST_WINDOW", "1");
        std::env::set_var("RNB_SPEC_BATCH_NO_BONUS", "1");
        std::env::set_var("RNB_SPEC_BATCH_PREFIX_SNAPSHOT", "1");

        assert!(spec_force_batch_verify_enabled());
        assert!(spec_decode_fast_window_enabled());
        assert!(spec_batch_no_bonus_enabled());
        assert_eq!(spec_batch_no_bonus_override(), Some(true));
        assert!(spec_batch_prefix_snapshot_enabled());

        std::env::set_var("RNB_SPEC_FORCE_BATCH_VERIFY", "0");
        assert!(!spec_force_batch_verify_enabled());
        std::env::set_var("RNB_SPEC_BATCH_NO_BONUS", "off");
        assert!(!spec_batch_no_bonus_enabled());
        assert_eq!(spec_batch_no_bonus_override(), Some(false));

        std::env::remove_var("RNB_SPEC_FORCE_BATCH_VERIFY");
        std::env::remove_var("RNB_SPEC_DECODE_FAST_WINDOW");
        std::env::remove_var("RNB_SPEC_BATCH_NO_BONUS");
        std::env::remove_var("RNB_SPEC_BATCH_PREFIX_SNAPSHOT");
    }

    #[test]
    fn moe_mixed_precision_policy_preserves_legacy_env_compatibility() {
        let _guard = env_lock().lock().expect("policy test env lock poisoned");
        std::env::remove_var("RNB_HOBBIT");
        std::env::remove_var("RNB_HOBBIT_T1");
        std::env::remove_var("RNB_HOBBIT_T2");
        std::env::remove_var("RNB_HOBBIT_LOW_PATH");

        assert!(!moe_mixed_precision_requested());
        assert!(!moe_mixed_precision_enabled(true));
        assert_eq!(moe_mixed_precision_thresholds(), (0.6, 0.9));
        assert_eq!(moe_mixed_precision_low_path(), None);

        std::env::set_var("RNB_HOBBIT", "1");
        std::env::set_var("RNB_HOBBIT_T1", "0.25");
        std::env::set_var("RNB_HOBBIT_T2", "0.75");
        std::env::set_var("RNB_HOBBIT_LOW_PATH", "tile");

        assert!(moe_mixed_precision_requested());
        assert!(!moe_mixed_precision_enabled(false));
        assert!(moe_mixed_precision_enabled(true));
        assert_eq!(moe_mixed_precision_thresholds(), (0.25, 0.75));
        assert_eq!(moe_mixed_precision_low_path().as_deref(), Some("tile"));

        std::env::remove_var("RNB_HOBBIT");
        std::env::remove_var("RNB_HOBBIT_T1");
        std::env::remove_var("RNB_HOBBIT_T2");
        std::env::remove_var("RNB_HOBBIT_LOW_PATH");
    }
}
