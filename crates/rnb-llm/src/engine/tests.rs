//! Unit tests for engine.rs internals.

#![cfg(test)]

#[cfg(feature = "vulkan")]
use super::backend_runtime::ggml_to_gpu_output_quant;
use super::cpu_runtime::quantize::BlockQ6_K;
use super::moe::tests::env_lock;
use super::moe_section::{attach_moe_section_decode, moe_section_decode_sidecar_requested};
use super::prefill::{new_empty_kv_cache, plan_slice1_boundary, run_prefill_layers_cpu_range};
use super::quantized_dispatch::{
    gemv_q8k_profile_method, q4k_kernel_backend_from_env, runtime_rawmeta_repack_enabled,
    Q4KKernelBackend,
};
#[cfg(target_arch = "aarch64")]
use super::quantized_dispatch::{
    pack_q4k_for_test, pack_q5k_for_test, pack_q6k_for_test, quantize_q8_for_test,
    quantize_q8k_for_test, QuantizedQ8Block,
};
use super::*;
use crate::tokenizer::{
    bpe::Tokenizer as BpeTokenizer,
    vocab::{SpecialTokens, Vocab},
};
use half::f16;

#[test]
fn qwen_gdn_packed_names_match_loaded_weight_names() {
    assert_eq!(gdn_qkv_weight_name(0), "blk.0.attn_qkv.weight");
    assert_eq!(gdn_gate_weight_name(0), "blk.0.attn_gate.weight");
}

#[test]
fn load_profile_env_gate_is_disabled_by_default() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let prev = std::env::var("RNB_LOAD_PROFILE").ok();
    unsafe {
        std::env::remove_var("RNB_LOAD_PROFILE");
    }

    assert!(!super::load_profile::load_profile_enabled_for_test());

    unsafe {
        if let Some(value) = prev {
            std::env::set_var("RNB_LOAD_PROFILE", value);
        }
    }
}

#[test]
fn load_profile_report_keeps_stage_ms_breakdown() {
    let mut profile = super::load_profile::LoadProfile::enabled_for_test();
    profile.record_for_test("load_model", std::time::Duration::from_micros(1250));
    profile.record_for_test("build_tokenizer", std::time::Duration::from_micros(500));

    let report = profile
        .finish_report_for_test(std::time::Duration::from_micros(2000))
        .expect("enabled profile should produce a report");

    assert!(report.contains("[load-profile] total_ms=2.000"));
    assert!(report.contains("load_model_ms=1.250"));
    assert!(report.contains("build_tokenizer_ms=0.500"));
}

fn make_mock_engine(vocab_size: usize) -> Engine {
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        vocab_size,
        max_seq_len: 64,
        hidden_dim: 16,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };
    Engine::mock(tokenizer, metadata)
}

#[test]
fn mock_engine_reports_no_mtp() {
    assert!(!make_mock_engine(8).has_mtp());
}

#[test]
fn engine_load_config_defaults_to_gguf_direct() {
    let config = EngineLoadConfig::default();
    assert!(config.diagnostic_sidecar.is_none());
}

#[test]
fn engine_init_does_not_auto_enable_q4k_prefill_f16_gemm() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let prev = std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").ok();
    let prev_allow = std::env::var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE").ok();
    unsafe {
        std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
        std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
    }

    super::init::maybe_enable_q4k_prefill_f16_gemm_for_test(true);

    assert!(
        std::env::var("RNB_CUDA_Q4K_PREFILL_F16_GEMM").is_err(),
        "rnb-llm must not auto-set CUDA expanded cache env"
    );

    unsafe {
        if let Some(value) = prev {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", value);
        }
        if let Some(value) = prev_allow {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", value);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_requests_exclude_expanded_quant_by_default() {
    let weights = minimal_cuda_product_prewarm_model_weights_for_test();
    let requests = backend_runtime::cuda_product_prewarm_request_kinds_for_test(&weights);
    let request_names: Vec<String> = requests
        .iter()
        .map(|request| format!("{request:?}"))
        .collect();

    assert!(
        requests.contains(&backend_runtime::CudaProductPrewarmRequestKindForTest::Q4PackedGateUp),
        "fixture must request Q4_K gate/up packed prewarm: {request_names:?}"
    );
    assert!(
        requests.contains(&backend_runtime::CudaProductPrewarmRequestKindForTest::Q4PackedSingle),
        "fixture must request Q4_K single/down packed prewarm: {request_names:?}"
    );
    assert!(
        requests.contains(&backend_runtime::CudaProductPrewarmRequestKindForTest::Q6PackedDown),
        "fixture must request Q6_K down packed prewarm: {request_names:?}"
    );
    assert!(
        request_names.iter().all(|name| {
            !name.contains("Expanded") && !name.contains("F16") && !name.contains("F32")
        }),
        "product prewarm requests must not include expanded quant cache kinds: {request_names:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_default_functions_do_not_call_expanded_prewarm_apis() {
    let forbidden = backend_runtime::cuda_product_prewarm_forbidden_expanded_calls_for_test();

    assert!(
        forbidden.is_empty(),
        "product default CUDA prewarm must not call expanded prewarm APIs: {forbidden:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_dense_wrappers_call_packed_executor() {
    let violations = backend_runtime::cuda_product_prewarm_wrapper_executor_violations_for_test();

    assert!(
        violations.is_empty(),
        "product dense CUDA prewarm wrappers must call packed executor families: {violations:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_does_not_set_quant_resident_policy() {
    let source = include_str!("backend_runtime/cuda_basic.rs");
    assert!(
        !source.contains("std::env::set_var(\"RNB_CUDA_QUANT_RESIDENT_MB\""),
        "rnb-llm must not set CUDA quant resident env"
    );
    assert!(
        !source.contains("std::env::var(\"RNB_CUDA_QUANT_RESIDENT_MB\""),
        "rnb-llm must not read CUDA quant resident policy env"
    );
    assert!(
        !source.contains("env::set_var(\"RNB_CUDA_QUANT_RESIDENT_MB\"")
            && !source.contains("set_var(\"RNB_CUDA_QUANT_RESIDENT_MB\""),
        "rnb-llm must not set CUDA quant resident env through an alias"
    );
    assert!(
        !source.contains("env::var(\"RNB_CUDA_QUANT_RESIDENT_MB\"")
            && !source.contains("var(\"RNB_CUDA_QUANT_RESIDENT_MB\""),
        "rnb-llm must not read CUDA quant resident policy env through an alias"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_collects_q4_raw_quant_candidates_without_policy() {
    let weights = minimal_cuda_product_prewarm_model_weights_for_test();
    let kinds = backend_runtime::cuda_product_prewarm_request_kinds_for_test(&weights);

    assert!(
        kinds.contains(&backend_runtime::CudaProductPrewarmRequestKindForTest::Q4RawQuant),
        "collector should expose raw Q4 quant candidates while backend owns opt-in policy"
    );
    assert_eq!(
        backend_runtime::cuda_product_prewarm_q4_raw_count_for_test(&weights),
        13,
        "collector should include Q4 raw quant gate/up/down/o_proj/qkv/ssm_out candidates"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_deduplicates_q4_raw_alias_candidates() {
    let weights = cuda_product_prewarm_alias_q4_raw_model_weights_for_test();

    assert_eq!(
        backend_runtime::cuda_product_prewarm_q4_raw_count_for_test(&weights),
        3,
        "collector should deduplicate aliased Q4 raw quant candidates by host slice"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_product_prewarm_q4_dense_executor_calls_quant_resident_backend_api() {
    assert!(
        !backend_runtime::cuda_product_prewarm_quant_resident_executor_missing_for_test(),
        "Q4 dense product prewarm executor must call backend-owned quant resident API"
    );
}

#[test]
fn device_mtp_verify_refuses_prefill_fallback_until_backend_exists() {
    let mut engine = make_mock_engine(8);
    let request =
        verify_window::MtpVerifyWindowRequest::new(1, &[2], verify_window::MtpVerifyBonus::Include);

    let err = engine
        .forward_mtp_device_verify_window_argmax_collect_mtp(&request)
        .unwrap_err();

    assert!(err.to_string().contains("우회하지 않음"));
    assert!(err.to_string().contains("pos_start=0"));
}

#[test]
fn device_verify_final_state_commit_updates_gdn_conv_state_and_len() {
    let mut engine = make_mock_engine(8);
    engine.kv_cache.init_ssm_state(1, 3, 2, 1, 2, 2);
    let result = verify_window::VerifyWindowResult {
        target_tokens: vec![10, 11],
        mtp_hidden_rows: vec![0.0; 2 * engine.metadata.hidden_dim],
        hidden_dim: engine.metadata.hidden_dim,
        prefix_state: None,
        prefix_states: Vec::new(),
        ssm_final_states: vec![verify_window::VerifyWindowSsmLayerFinalState {
            layer_idx: 1,
            conv_state: vec![0.25, 0.5, 0.75, 1.0],
        }],
        attention_kv_states: Vec::new(),
    };

    engine
        .commit_device_verify_window_final_states(4, &result)
        .expect("commit device verify final states");

    assert_eq!(engine.kv_cache.current_len(), 6);
    assert_eq!(
        engine.kv_cache.get_ssm_state(1).unwrap().conv_state,
        vec![0.25, 0.5, 0.75, 1.0]
    );
}

#[test]
fn device_verify_final_state_commit_writes_attention_kv_range() {
    let mut engine = make_mock_engine(8);
    let kv_rows = engine.metadata.num_kv_heads * engine.metadata.head_dim;
    let k_bits = (0..2 * kv_rows)
        .map(|i| half::f16::from_f32(10.0 + i as f32).to_bits())
        .collect::<Vec<_>>();
    let v_bits = (0..2 * kv_rows)
        .map(|i| half::f16::from_f32(100.0 + i as f32).to_bits())
        .collect::<Vec<_>>();
    let result = verify_window::VerifyWindowResult {
        target_tokens: vec![10, 11],
        mtp_hidden_rows: vec![0.0; 2 * engine.metadata.hidden_dim],
        hidden_dim: engine.metadata.hidden_dim,
        prefix_state: None,
        prefix_states: Vec::new(),
        ssm_final_states: Vec::new(),
        attention_kv_states: vec![verify_window::VerifyWindowAttentionKvState {
            layer_idx: 0,
            window_tokens: 2,
            kv_rows,
            k_bits: k_bits.clone(),
            v_bits: v_bits.clone(),
        }],
    };

    engine
        .commit_device_verify_window_final_states(4, &result)
        .expect("commit device verify attention K/V");

    let (actual_k, actual_v) = engine.kv_cache.get_up_to(0, 6);
    assert_eq!(&actual_k[4 * kv_rows..6 * kv_rows], k_bits.as_slice());
    assert_eq!(&actual_v[4 * kv_rows..6 * kv_rows], v_bits.as_slice());
}

fn make_f32_weight(rows: usize, cols: usize, data: Vec<f32>) -> QuantizedWeight {
    QuantizedWeight::new(
        Tensor::from_vec(data, &[rows, cols]),
        GGMLType::F32,
        rows,
        cols,
    )
}

#[cfg(feature = "cuda")]
fn make_q4k_weight_for_mtp_device_test(rows: usize, cols: usize) -> QuantizedWeight {
    assert_eq!(cols % 256, 0);
    let bytes = vec![0u8; rows * (cols / 256) * 144];
    QuantizedWeight::new(
        Tensor::from_vec(bytes.clone(), &[bytes.len()]),
        GGMLType::Q4_K,
        rows,
        cols,
    )
}

#[cfg(feature = "cuda")]
fn make_q4k_tensor_for_mtp_device_test(rows: usize, cols: usize) -> Tensor {
    assert_eq!(cols % 256, 0);
    let bytes = vec![0u8; rows * (cols / 256) * 144];
    Tensor::from_vec(bytes.clone(), &[bytes.len()])
}

#[cfg(feature = "cuda")]
fn make_qwen35_moe_weights_for_mtp_device_test(
    hidden: usize,
    n_ff: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> SharedExpertMoELayerWeights {
    SharedExpertMoELayerWeights {
        router_w: Tensor::from_slice(&vec![0.0f32; n_expert * hidden], &[n_expert, hidden]),
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        gate_exps: make_q4k_tensor_for_mtp_device_test(n_expert * n_ff, hidden),
        gate_quant: GGMLType::Q4_K,
        up_exps: make_q4k_tensor_for_mtp_device_test(n_expert * n_ff, hidden),
        up_quant: GGMLType::Q4_K,
        down_exps: make_q4k_tensor_for_mtp_device_test(n_expert * hidden, n_ff),
        down_quant: GGMLType::Q4_K,
        shared_input_scale: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        shared_expert_gated: true,
        shared_gate: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        shared_up: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        shared_down: make_q4k_weight_for_mtp_device_test(hidden, n_ff),
        n_embd: hidden,
        n_ff,
        n_expert,
        n_expert_used,
        prefer_sparse_moe_cuda: false,
        sparse_page_cache: None,
        packed_model: None,
        gate_exps_rnb_name: None,
        up_exps_rnb_name: None,
        down_exps_rnb_name: None,
        router_rnb_name: None,
        shared_gate_rnb_name: None,
        shared_up_rnb_name: None,
        shared_down_rnb_name: None,
        shared_scale_rnb_name: None,
        rank_to_original: None,
        shadow_model: None,
        shadow_gate_up_tile_rnb_name: None,
        shadow_gate_rnb_name: None,
        shadow_up_rnb_name: None,
        shadow_down_rnb_name: None,
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    }
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_device_verify_collects_qwen35_gdn_moe_layer_graph_from_engine_parts() {
    let hidden = 256usize;
    let d_inner = 256usize;
    let d_state = 128usize;
    let n_group = 1usize;
    let dt_rank = 1usize;
    let conv_kernel = 4usize;
    let conv_channels = d_inner + 2 * n_group * d_state;
    let n_ff = 256usize;
    let n_expert = 2usize;
    let n_expert_used = 1usize;
    let head_v_dim = d_inner / dt_rank;
    let layer = GdnLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        qkv_weight: make_q4k_weight_for_mtp_device_test(conv_channels, hidden),
        gate_weight: make_q4k_weight_for_mtp_device_test(d_inner, hidden),
        ssm_a: Tensor::from_slice(&vec![0.0f32; dt_rank], &[dt_rank]),
        ssm_alpha: make_q4k_weight_for_mtp_device_test(dt_rank, hidden),
        ssm_beta: make_q4k_weight_for_mtp_device_test(dt_rank, hidden),
        ssm_conv1d: Tensor::from_slice(
            &vec![0.0f32; conv_kernel * conv_channels],
            &[conv_kernel, conv_channels],
        ),
        ssm_dt_bias: Tensor::from_slice(&vec![0.0f32; dt_rank], &[dt_rank]),
        ssm_norm: Tensor::from_slice(&vec![1.0f32; head_v_dim], &[head_v_dim]),
        ssm_out: make_q4k_weight_for_mtp_device_test(hidden, d_inner),
        ffn_gate_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_up_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_down_weight: make_q4k_weight_for_mtp_device_test(hidden, n_ff),
        ffn_gate_up_fused: None,
        shared_expert_moe: Some(make_qwen35_moe_weights_for_mtp_device_test(
            hidden,
            n_ff,
            n_expert,
            n_expert_used,
        )),
    };
    let weights = ModelWeights {
        token_embd: make_q4k_weight_for_mtp_device_test(8, hidden),
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        output: make_q4k_weight_for_mtp_device_test(8, hidden),
        layers: vec![LayerType::GatedDeltaNet(layer)],
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: hidden,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: n_expert_used,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 0,
    };
    let mut kv_cache = KVCache::new(1, 16, 1, hidden);
    kv_cache.init_ssm_state(0, conv_kernel, conv_channels, dt_rank, head_v_dim, d_state);

    let layers =
        inference::build_mtp_device_verify_gdn_moe_layers(&weights, &metadata, &mut kv_cache)
            .unwrap();

    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].layer_index, 0);
    assert_eq!(layers[0].n_embd, hidden);
    assert_eq!(layers[0].n_ff, n_ff);
    assert_eq!(layers[0].n_expert, n_expert);
    assert_eq!(layers[0].n_expert_used, n_expert_used);
    assert_eq!(
        layers[0].conv_state.len(),
        (conv_kernel - 1) * conv_channels
    );
    assert_eq!(layers[0].delta_state.len(), dt_rank * head_v_dim * d_state);
    assert_eq!(layers[0].qkv_rows, conv_channels);
    assert_eq!(layers[0].ssm_out_rows, hidden);
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_device_verify_refuses_partial_graph_when_qwen35_attention_layer_is_present() {
    let hidden = 256usize;
    let d_inner = 256usize;
    let d_state = 128usize;
    let n_group = 1usize;
    let dt_rank = 1usize;
    let conv_kernel = 4usize;
    let n_ff = 256usize;
    let n_expert = 2usize;
    let n_expert_used = 1usize;
    let attention = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        q_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        k_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        v_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        o_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        k_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        post_attn_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        ffn_gate_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_up_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_down_weight: make_q4k_weight_for_mtp_device_test(hidden, n_ff),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: Some(make_qwen35_moe_weights_for_mtp_device_test(
            hidden,
            n_ff,
            n_expert,
            n_expert_used,
        )),
        v_proj_missing: false,
    };
    let weights = ModelWeights {
        token_embd: make_q4k_weight_for_mtp_device_test(8, hidden),
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        output: make_q4k_weight_for_mtp_device_test(8, hidden),
        layers: vec![LayerType::Attention(attention)],
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: hidden,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: n_expert_used,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 4,
    };
    let mut kv_cache = KVCache::new(1, 16, 1, hidden);

    let err = inference::build_mtp_device_verify_gdn_moe_layers(&weights, &metadata, &mut kv_cache)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("attention layer is not wired into MTP device verify graph"));
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_device_verify_collects_qwen35_attention_moe_layer_graph_from_engine_parts() {
    let hidden = 256usize;
    let n_ff = 256usize;
    let n_expert = 2usize;
    let n_expert_used = 1usize;
    let attention = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        q_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        k_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        v_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        o_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        k_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        post_attn_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        ffn_gate_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_up_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_down_weight: make_q4k_weight_for_mtp_device_test(hidden, n_ff),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: Some(make_qwen35_moe_weights_for_mtp_device_test(
            hidden,
            n_ff,
            n_expert,
            n_expert_used,
        )),
        v_proj_missing: false,
    };
    let weights = ModelWeights {
        token_embd: make_q4k_weight_for_mtp_device_test(8, hidden),
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        output: make_q4k_weight_for_mtp_device_test(8, hidden),
        layers: vec![LayerType::Attention(attention)],
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: hidden,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: n_expert_used,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 4,
    };
    let mut kv_cache = KVCache::new(1, 16, 1, hidden);
    let prior_tokens = 2usize;
    let prior_k_bits = (0..prior_tokens * hidden)
        .map(|i| half::f16::from_f32((i as f32 + 1.0) * 0.001).to_bits())
        .collect::<Vec<_>>();
    let prior_v_bits = (0..prior_tokens * hidden)
        .map(|i| half::f16::from_f32((i as f32 + 3.0) * 0.001).to_bits())
        .collect::<Vec<_>>();
    kv_cache.replace_layer_f16_range(0, 0, prior_tokens, &prior_k_bits, &prior_v_bits);
    kv_cache.set_len(prior_tokens);

    let graph =
        inference::build_mtp_device_verify_layer_graph(&weights, &metadata, &mut kv_cache).unwrap();

    assert_eq!(graph.attention_moe_layers.len(), 1);
    assert_eq!(graph.gdn_moe_layers.len(), 0);
    assert_eq!(
        graph.layer_order,
        vec![crate::engine::cuda_runtime::MtpDeviceVerifyLayerKind::AttentionMoe(0)]
    );
    let layer = &graph.attention_moe_layers[0];
    assert_eq!(layer.layer_index, 0);
    assert_eq!(layer.q_rows, hidden);
    assert_eq!(layer.k_rows, hidden);
    assert_eq!(layer.v_rows, hidden);
    assert_eq!(layer.o_rows, hidden);
    assert_eq!(layer.n_embd, hidden);
    assert_eq!(layer.n_ff, n_ff);
    assert_eq!(layer.n_expert, n_expert);
    assert_eq!(layer.n_expert_used, n_expert_used);
    assert_eq!(layer.q_norm.len(), hidden);
    assert_eq!(layer.k_norm.len(), hidden);
    assert_eq!(layer.post_attn_norm.len(), hidden);
    assert_eq!(layer.prior_tokens, prior_tokens);
    assert_eq!(layer.prior_k_bits, prior_k_bits);
    assert_eq!(layer.prior_v_bits, prior_v_bits);
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_device_verify_collects_qwen35_dense_attention_layer_graph_from_engine_parts() {
    let hidden = 256usize;
    let n_ff = 256usize;
    let attention = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        q_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        k_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        v_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        o_weight: make_q4k_weight_for_mtp_device_test(hidden, hidden),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        k_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        post_attn_norm: Some(Tensor::from_slice(&vec![1.0f32; hidden], &[hidden])),
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        ffn_gate_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_up_weight: make_q4k_weight_for_mtp_device_test(n_ff, hidden),
        ffn_down_weight: make_q4k_weight_for_mtp_device_test(hidden, n_ff),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };
    let weights = ModelWeights {
        token_embd: make_q4k_weight_for_mtp_device_test(8, hidden),
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        output: make_q4k_weight_for_mtp_device_test(8, hidden),
        layers: vec![LayerType::Attention(attention)],
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: hidden,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 4,
    };
    let mut kv_cache = KVCache::new(1, 16, 1, hidden);

    let graph =
        inference::build_mtp_device_verify_layer_graph(&weights, &metadata, &mut kv_cache).unwrap();

    assert_eq!(graph.attention_moe_layers.len(), 1);
    assert_eq!(graph.gdn_moe_layers.len(), 0);
    assert_eq!(
        graph.layer_order,
        vec![crate::engine::cuda_runtime::MtpDeviceVerifyLayerKind::AttentionMoe(0)]
    );
    let layer = &graph.attention_moe_layers[0];
    assert_eq!(layer.layer_index, 0);
    assert_eq!(layer.n_expert, 0);
    assert_eq!(layer.n_expert_used, 0);
    assert_eq!(layer.n_embd, hidden);
    assert_eq!(layer.n_ff, n_ff);
}

#[cfg(feature = "vulkan")]
fn make_q4k_weight_for_fullpath_test(rows: usize, cols: usize) -> QuantizedWeight {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let bytes = vec![0u8; rows * blocks_per_row * 144];
    QuantizedWeight::new(
        Tensor::from_vec(bytes.clone(), &[bytes.len()]),
        GGMLType::Q4_K,
        rows,
        cols,
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn fullpath_gdn_extraction_accepts_f32_alpha_beta() {
    let hidden = 256usize;
    let d_inner = 256usize;
    let conv_channels = 512usize;
    let ffn_inner = 256usize;
    let layer = GdnLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        qkv_weight: make_q4k_weight_for_fullpath_test(conv_channels, hidden),
        gate_weight: make_q4k_weight_for_fullpath_test(d_inner, hidden),
        ssm_a: Tensor::from_slice(&[0.0f32], &[1]),
        ssm_alpha: make_f32_weight(1, hidden, vec![0.0; hidden]),
        ssm_beta: make_f32_weight(1, hidden, vec![0.0; hidden]),
        ssm_conv1d: Tensor::from_slice(&vec![0.0f32; 4 * conv_channels], &[4, conv_channels]),
        ssm_dt_bias: Tensor::from_slice(&[0.0f32], &[1]),
        ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
        ssm_out: make_q4k_weight_for_fullpath_test(hidden, d_inner),
        ffn_gate_weight: make_q4k_weight_for_fullpath_test(ffn_inner, hidden),
        ffn_up_weight: make_q4k_weight_for_fullpath_test(ffn_inner, hidden),
        ffn_down_weight: make_q4k_weight_for_fullpath_test(hidden, ffn_inner),
        ffn_gate_up_fused: None,
        shared_expert_moe: None,
    };
    let weights = ModelWeights {
        token_embd: make_q4k_weight_for_fullpath_test(8, hidden),
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        output: make_q4k_weight_for_fullpath_test(8, hidden),
        layers: vec![LayerType::GatedDeltaNet(layer)],
        gemma_per_layer: None,
        rope_freqs: None,
    };

    assert_eq!(
        Engine::fullpath_ffn_inner_dim_or_error(&weights, "test").unwrap(),
        ffn_inner
    );
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: hidden,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: 128,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_conv_kernel: 4,
        full_attention_interval: 0,
    };
    let (raw, kinds) = inference::build_fullpath_layer_raw_weights(&weights, &metadata).unwrap();
    assert_eq!(kinds, vec![gpu_runtime::ModelLayerKind::Recurrent]);
    match &raw[0] {
        gpu_runtime::LayerRawWeights::Gdn(g) => {
            assert_eq!(g.ssm_alpha.1, 1);
            assert_eq!(g.ssm_alpha.2, hidden);
            assert_eq!(g.ssm_beta.1, 1);
            assert_eq!(g.ssm_beta.2, hidden);
            assert_eq!(g.num_k_heads, 1);
            assert_eq!(g.head_k_dim, 128);
        }
        gpu_runtime::LayerRawWeights::Attention(_) => panic!("expected GDN raw weights"),
    }
}

#[test]
fn output_weight_prewarm_without_weights_returns_false() {
    let engine = make_mock_engine(8);
    assert!(!engine.prewarm_output_weight_for_runtime());
}

#[cfg(target_arch = "aarch64")]
fn make_quant_weight(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
) -> QuantizedWeight {
    let byte_len = bytes.len();
    QuantizedWeight::new(Tensor::from_vec(bytes, &[byte_len]), ggml_type, rows, cols)
}

#[cfg(target_arch = "aarch64")]
fn make_q4k_block(d_val: f32, dmin_val: f32, scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
    block[4..16].copy_from_slice(&scales);
    block[16..144].copy_from_slice(&qs);
    block
}

#[cfg(target_arch = "aarch64")]
fn make_q5k_block(
    d_val: f32,
    dmin_val: f32,
    scales: [u8; 12],
    qh: [u8; 32],
    qs: [u8; 128],
) -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
    block[4..16].copy_from_slice(&scales);
    block[16..48].copy_from_slice(&qh);
    block[48..176].copy_from_slice(&qs);
    block
}

#[cfg(target_arch = "aarch64")]
fn make_q6k_block(d_val: f32, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    block[0..128].copy_from_slice(&ql);
    block[128..192].copy_from_slice(&qh);
    for (i, &s) in scales.iter().enumerate() {
        block[192 + i] = s as u8;
    }
    block[208..210].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
    block
}

fn make_gemma_per_layer_metadata() -> ModelMetadata {
    ModelMetadata {
        num_layers: 2,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 8,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 2,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    }
}

fn make_dummy_gemma_per_layer_weights() -> GemmaPerLayerWeights {
    GemmaPerLayerWeights {
        token_embd: make_f32_weight(
            8,
            4,
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        model_proj: make_f32_weight(4, 4, vec![0.0; 16]),
        proj_norm: Tensor::from_slice(&[0.0f32, 0.0], &[2]),
        layers: vec![
            GemmaPerLayerLayerWeights {
                inp_gate: make_f32_weight(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                proj: make_f32_weight(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                post_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
            },
            GemmaPerLayerLayerWeights {
                inp_gate: make_f32_weight(2, 4, vec![0.0; 8]),
                proj: make_f32_weight(4, 2, vec![0.0; 8]),
                post_norm: Tensor::from_slice(&[0.0f32; 4], &[4]),
            },
        ],
    }
}

fn make_decode_test_engine(vocab_size: usize) -> Engine {
    let hidden_dim = 4;
    let head_dim = 4;
    let ffn_inner_dim = 4;
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 % hidden_dim as f32) * 0.01)
            .collect(),
    );
    let eye = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let zero_ffn = vec![0.0; ffn_inner_dim * hidden_dim];
    let zero_down = vec![0.0; hidden_dim * ffn_inner_dim];
    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        q_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        k_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        v_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        o_weight: make_f32_weight(hidden_dim, hidden_dim, eye),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        output: make_f32_weight(vocab_size, hidden_dim, vec![0.0; vocab_size * hidden_dim]),
        layers: vec![LayerType::Attention(layer)],
        gemma_per_layer: None,
        glm_dsa_attention: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);

    Engine {
        tokenizer,
        kv_cache: KVCache::new(
            metadata.num_layers,
            metadata.max_seq_len,
            metadata.num_kv_heads,
            metadata.head_dim,
        ),
        metadata,
        architecture: ModelArchitecture::LLaMA,
        host_memory_plan: rnb_runtime::policy::HostMemoryPlan::default(),
        weights: Some(weights),
        scratch: Some(scratch),
        mtp: None,
        mtp_runtime: None,
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[test]
fn test_prepare_gemma_per_layer_base_uses_token_branch_when_model_proj_is_zero() {
    let metadata = make_gemma_per_layer_metadata();
    let gemma = make_dummy_gemma_per_layer_weights();
    let weights = ModelWeights {
        token_embd: make_f32_weight(8, 4, vec![0.0; 32]),
        output_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        output: make_f32_weight(8, 4, vec![0.0; 32]),
        layers: vec![],
        gemma_per_layer: Some(gemma),
        glm_dsa_attention: None,
        rope_freqs: None,
    };
    let hidden = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 4]);

    let base = prepare_gemma_per_layer_base(
        &weights,
        &hidden,
        &[1],
        &metadata,
        ModelArchitecture::Gemma,
        metadata.norm_eps,
    )
    .expect("prepare should succeed")
    .expect("gemma base should exist");

    let expected = vec![1.0, 2.0, 3.0, 4.0];
    for (got, want) in base.mixed.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
    }
}

#[test]
fn test_apply_gemma_per_layer_branch_adds_projected_branch_to_hidden() {
    let metadata = make_gemma_per_layer_metadata();
    let gemma = make_dummy_gemma_per_layer_weights();
    let hidden = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0], &[1, 4]);
    let base = GemmaPerLayerBase {
        mixed: vec![1.0f32, 2.0, 3.0, 4.0],
        token: vec![1.0f32, 2.0, 3.0, 4.0],
        model: vec![0.0f32, 0.0, 0.0, 0.0],
    };

    let updated = apply_gemma_per_layer_branch(
        hidden,
        &base,
        0,
        &gemma,
        &metadata,
        ModelArchitecture::Gemma,
        metadata.norm_eps,
    )
    .expect("branch apply should succeed");

    let data = kernels::tensor_as_f32_slice(&updated);
    assert!(data[0] > 1.3 && data[0] < 1.6, "got {}", data[0]);
    assert!(data[1] > 3.6 && data[1] < 4.2, "got {}", data[1]);
    assert!(data[2].abs() < 1e-6, "got {}", data[2]);
    assert!(data[3].abs() < 1e-6, "got {}", data[3]);
}

fn make_multi_layer_attention_engine(
    vocab_size: usize,
    num_layers: usize,
    full_attention_interval: usize,
) -> Engine {
    let hidden_dim = 4;
    let head_dim = 4;
    let ffn_inner_dim = 4;
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval,
    };

    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.05)
            .collect(),
    );
    let eye = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let zero_ffn = vec![0.0; ffn_inner_dim * hidden_dim];
    let zero_down = vec![0.0; hidden_dim * ffn_inner_dim];
    let layers = (0..num_layers)
        .map(|_| {
            LayerType::Attention(AttentionLayerWeights {
                attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
                q_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
                k_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
                v_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
                o_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
                post_attn_norm: None,
                out_scale: None,
                post_ffw_norm: None,
                ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
                ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
                ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
                ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down.clone()),
                ffn_gate_up_fused: None,
                moe: None,
                shared_expert_moe: None,
                v_proj_missing: false,
            })
        })
        .collect();
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        output: make_f32_weight(vocab_size, hidden_dim, vec![0.0; vocab_size * hidden_dim]),
        layers,
        gemma_per_layer: None,
        glm_dsa_attention: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);

    Engine {
        tokenizer,
        kv_cache: KVCache::new(
            metadata.num_layers,
            metadata.max_seq_len,
            metadata.num_kv_heads,
            metadata.head_dim,
        ),
        metadata,
        architecture: ModelArchitecture::LLaMA,
        host_memory_plan: rnb_runtime::policy::HostMemoryPlan::default(),
        weights: Some(weights),
        scratch: Some(scratch),
        mtp: None,
        mtp_runtime: None,
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[cfg(feature = "vulkan")]
fn make_hybrid_slice1_engine(vocab_size: usize) -> Engine {
    let hidden_dim = 4;
    let head_dim = 4;
    let ffn_inner_dim = 4;
    let d_inner = 4;
    let d_state = 1;
    let n_group = 1;
    let dt_rank = 1;
    let conv_kernel = 2;
    let conv_channels = d_inner + 2 * n_group * d_state;

    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 4,
    };

    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.05)
            .collect(),
    );
    let eye = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let zero_ssm_out = vec![0.0; hidden_dim * d_inner];
    let zero_ffn = vec![0.0; ffn_inner_dim * hidden_dim];
    let zero_down = vec![0.0; hidden_dim * ffn_inner_dim];
    let zero_conv = vec![0.0; conv_kernel * conv_channels];
    let zero_qkv = vec![0.0; conv_channels * hidden_dim];
    let zero_gate = vec![0.0; d_inner * hidden_dim];

    let mut layers = vec![];
    for _ in 0..3 {
        layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
            attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
            post_attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
            qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv.clone()),
            gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate.clone()),
            ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
            ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_norm: Tensor::from_slice(&[1.0f32; 4], &[d_inner]),
            ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out.clone()),
            ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
            ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
            ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down.clone()),
            ffn_gate_up_fused: None,
            shared_expert_moe: None,
        }));
    }
    layers.push(LayerType::Attention(AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        q_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        k_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        v_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        o_weight: make_f32_weight(hidden_dim, hidden_dim, eye.clone()),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    }));

    let output = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32 + 1.0) * 0.1)
            .collect(),
    );
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
        output,
        layers,
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);
    let mut kv_cache = KVCache::new(
        metadata.num_layers,
        metadata.max_seq_len,
        metadata.num_kv_heads,
        metadata.head_dim,
    );
    for layer_idx in 0..3 {
        kv_cache.init_ssm_state(
            layer_idx,
            conv_kernel,
            conv_channels,
            dt_rank,
            d_inner / dt_rank,
            d_state,
        );
    }

    Engine {
        tokenizer,
        kv_cache,
        metadata,
        architecture: ModelArchitecture::Qwen35,
        weights: Some(weights),
        scratch: Some(scratch),
        mtp: None,
        mtp_runtime: None,
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[cfg(feature = "vulkan")]
fn make_hybrid_slice1_engine_with_suffix(vocab_size: usize) -> Engine {
    let mut engine = make_hybrid_slice1_engine(vocab_size);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = 4usize;
    let d_inner = engine.metadata.ssm_d_inner;
    let d_state = engine.metadata.ssm_d_state;
    let n_group = engine.metadata.ssm_n_group;
    let dt_rank = engine.metadata.ssm_dt_rank;
    let conv_kernel = engine.metadata.ssm_conv_kernel;
    let conv_channels = d_inner + 2 * n_group * d_state;
    let zero_qkv = vec![0.0; conv_channels * hidden_dim];
    let zero_gate = vec![0.0; d_inner * hidden_dim];
    let zero_conv = vec![0.0; conv_kernel * conv_channels];
    let zero_ssm_out = vec![0.0; hidden_dim * d_inner];
    let zero_ffn = vec![0.0; ffn_inner_dim * hidden_dim];
    let zero_down = vec![0.0; hidden_dim * ffn_inner_dim];

    if let Some(weights) = engine.weights.as_mut() {
        weights
            .layers
            .push(LayerType::GatedDeltaNet(GdnLayerWeights {
                attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
                post_attn_norm: Tensor::from_slice(&[1.0f32; 4], &[hidden_dim]),
                qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv),
                gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate),
                ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
                ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
                ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
                ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
                ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
                ssm_norm: Tensor::from_slice(&[1.0f32; 4], &[d_inner]),
                ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out),
                ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
                ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
                ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down),
                ffn_gate_up_fused: None,
                shared_expert_moe: None,
            }));
    }
    engine.metadata.num_layers = 5;
    engine.kv_cache = KVCache::new(
        engine.metadata.num_layers,
        engine.metadata.max_seq_len,
        engine.metadata.num_kv_heads,
        engine.metadata.head_dim,
    );
    for layer_idx in [0usize, 1, 2, 4] {
        engine.kv_cache.init_ssm_state(
            layer_idx,
            conv_kernel,
            conv_channels,
            dt_rank,
            d_inner / dt_rank,
            d_state,
        );
    }
    engine
}

#[allow(dead_code)]
fn make_hybrid_slice1_gqa_engine_with_suffix(vocab_size: usize) -> Engine {
    let hidden_dim = 8;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 2;
    let ffn_inner_dim = 8;
    let d_inner = 4;
    let d_state = 1;
    let n_group = 1;
    let dt_rank = 1;
    let conv_kernel = 2;
    let conv_channels = d_inner + 2 * n_group * d_state;

    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: 5,
        num_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 4,
    };

    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.03)
            .collect(),
    );
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mut q_eye = vec![0.0f32; q_dim * hidden_dim];
    for i in 0..q_dim {
        q_eye[i * hidden_dim + i] = 1.0;
    }
    let mut kv_eye = vec![0.0f32; kv_dim * hidden_dim];
    for i in 0..kv_dim {
        kv_eye[i * hidden_dim + i] = 1.0;
    }
    let mut o_eye = vec![0.0f32; hidden_dim * q_dim];
    for i in 0..q_dim {
        o_eye[i * q_dim + i] = 1.0;
    }
    let zero_qkv = vec![0.0; conv_channels * hidden_dim];
    let zero_gate = vec![0.0; d_inner * hidden_dim];
    let zero_conv = vec![0.0; conv_kernel * conv_channels];
    let zero_ssm_out = vec![0.0; hidden_dim * d_inner];
    let zero_ffn = vec![0.0; ffn_inner_dim * hidden_dim];
    let zero_down = vec![0.0; hidden_dim * ffn_inner_dim];

    let mut layers = vec![];
    for _ in 0..3 {
        layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
            attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv.clone()),
            gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate.clone()),
            ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
            ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
            ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out.clone()),
            ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
            ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
            ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down.clone()),
            ffn_gate_up_fused: None,
            shared_expert_moe: None,
        }));
    }
    layers.push(LayerType::Attention(AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        q_weight: make_f32_weight(q_dim, hidden_dim, q_eye),
        k_weight: make_f32_weight(kv_dim, hidden_dim, kv_eye.clone()),
        v_weight: make_f32_weight(kv_dim, hidden_dim, kv_eye),
        o_weight: make_f32_weight(hidden_dim, q_dim, o_eye),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down.clone()),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    }));
    layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv),
        gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate),
        ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
        ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
        ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
        ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
        ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
        ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
        ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out),
        ffn_gate_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_up_weight: make_f32_weight(ffn_inner_dim, hidden_dim, zero_ffn.clone()),
        ffn_down_weight: make_f32_weight(hidden_dim, ffn_inner_dim, zero_down),
        ffn_gate_up_fused: None,
        shared_expert_moe: None,
    }));

    let output = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32 + 1.0) * 0.07)
            .collect(),
    );
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        output,
        layers,
        gemma_per_layer: None,
        glm_dsa_attention: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);
    let mut kv_cache = KVCache::new(
        metadata.num_layers,
        metadata.max_seq_len,
        metadata.num_kv_heads,
        metadata.head_dim,
    );
    for layer_idx in [0usize, 1, 2, 4] {
        kv_cache.init_ssm_state(
            layer_idx,
            conv_kernel,
            conv_channels,
            dt_rank,
            d_inner / dt_rank,
            d_state,
        );
    }

    Engine {
        tokenizer,
        kv_cache,
        metadata,
        architecture: ModelArchitecture::Qwen35,
        host_memory_plan: rnb_runtime::policy::HostMemoryPlan::default(),
        weights: Some(weights),
        scratch: Some(scratch),
        mtp: None,
        mtp_runtime: None,
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[cfg(any(target_arch = "aarch64", feature = "vulkan"))]
fn make_q8_0_weight(rows: usize, cols: usize, values: &[i8]) -> QuantizedWeight {
    assert_eq!(cols % 32, 0);
    assert_eq!(values.len(), rows * cols);
    let blocks_per_row = cols / 32;
    let mut bytes = Vec::with_capacity(rows * blocks_per_row * 34);
    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let start = row * cols + blk * 32;
            let d = f16::from_f32(1.0);
            bytes.extend_from_slice(&d.to_le_bytes());
            for &v in &values[start..start + 32] {
                bytes.push(v as u8);
            }
        }
    }
    QuantizedWeight::new(
        Tensor::from_slice(&bytes, &[bytes.len()]),
        GGMLType::Q8_0,
        rows,
        cols,
    )
}

#[cfg(feature = "vulkan")]
fn make_zero_q8_0_weight(rows: usize, cols: usize) -> QuantizedWeight {
    make_q8_0_weight(rows, cols, &vec![0i8; rows * cols])
}

#[cfg(feature = "vulkan")]
fn make_identity_q8_0_weight(rows: usize, cols: usize) -> QuantizedWeight {
    let mut values = vec![0i8; rows * cols];
    for i in 0..rows.min(cols) {
        values[i * cols + i] = 1;
    }
    make_q8_0_weight(rows, cols, &values)
}

#[cfg(feature = "vulkan")]
fn make_real_gpu_hybrid_slice1_gqa_engine_with_quantized_gdn_suffix(vocab_size: usize) -> Engine {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(vocab_size);
    let weights = engine.weights.as_mut().expect("weights");
    for layer in &mut weights.layers {
        if let LayerType::GatedDeltaNet(w) = layer {
            w.qkv_weight = make_zero_q8_0_weight(w.qkv_weight.rows, w.qkv_weight.cols);
        }
    }
    engine
}

#[cfg(feature = "vulkan")]
fn make_real_gpu_multi_layer_attention_engine(
    vocab_size: usize,
    num_layers: usize,
    full_attention_interval: usize,
) -> Engine {
    let hidden_dim = 32;
    let head_dim = 32;
    let ffn_inner_dim = 32;
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval,
    };

    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.01)
            .collect(),
    );
    let layers = (0..num_layers)
        .map(|_| {
            LayerType::Attention(AttentionLayerWeights {
                attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
                q_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
                k_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
                v_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
                o_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
                post_attn_norm: None,
                out_scale: None,
                post_ffw_norm: None,
                ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
                ffn_gate_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
                ffn_up_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
                ffn_down_weight: make_zero_q8_0_weight(hidden_dim, ffn_inner_dim),
                ffn_gate_up_fused: None,
                moe: None,
                shared_expert_moe: None,
                v_proj_missing: false,
            })
        })
        .collect();
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        output: make_f32_weight(vocab_size, hidden_dim, vec![0.0; vocab_size * hidden_dim]),
        layers,
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);
    Engine {
        tokenizer,
        kv_cache: KVCache::new(
            metadata.num_layers,
            metadata.max_seq_len,
            metadata.num_kv_heads,
            metadata.head_dim,
        ),
        metadata,
        architecture: ModelArchitecture::LLaMA,
        weights: Some(weights),
        scratch: Some(scratch),
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[cfg(feature = "vulkan")]
fn make_real_gpu_hybrid_slice1_engine(vocab_size: usize, with_suffix: bool) -> Engine {
    let hidden_dim = 32;
    let head_dim = 32;
    let ffn_inner_dim = 32;
    let d_inner = 32;
    let d_state = 1;
    let n_group = 1;
    let dt_rank = 1;
    let conv_kernel = 2;
    let conv_channels = d_inner + 2 * n_group * d_state;
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: if with_suffix { 5 } else { 4 },
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 4,
    };
    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.02)
            .collect(),
    );
    let zero_qkv = vec![0.0; conv_channels * hidden_dim];
    let zero_gate = vec![0.0; d_inner * hidden_dim];
    let zero_conv = vec![0.0; conv_kernel * conv_channels];
    let zero_ssm_out = vec![0.0; hidden_dim * d_inner];
    let mut layers = vec![];
    for _ in 0..3 {
        layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
            attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv.clone()),
            gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate.clone()),
            ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
            ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
            ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out.clone()),
            ffn_gate_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_up_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_down_weight: make_f32_weight(
                hidden_dim,
                ffn_inner_dim,
                vec![0.0; hidden_dim * ffn_inner_dim],
            ),
            ffn_gate_up_fused: None,
            shared_expert_moe: None,
        }));
    }
    layers.push(LayerType::Attention(AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        q_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
        k_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
        v_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
        o_weight: make_identity_q8_0_weight(hidden_dim, hidden_dim),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        ffn_gate_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
        ffn_up_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
        ffn_down_weight: make_zero_q8_0_weight(hidden_dim, ffn_inner_dim),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    }));
    if with_suffix {
        layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
            attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv),
            gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate),
            ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
            ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
            ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out),
            ffn_gate_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_up_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_down_weight: make_f32_weight(
                hidden_dim,
                ffn_inner_dim,
                vec![0.0; hidden_dim * ffn_inner_dim],
            ),
            ffn_gate_up_fused: None,
            shared_expert_moe: None,
        }));
    }
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        output: make_f32_weight(vocab_size, hidden_dim, vec![0.0; vocab_size * hidden_dim]),
        layers,
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);
    let mut kv_cache = KVCache::new(
        metadata.num_layers,
        metadata.max_seq_len,
        metadata.num_kv_heads,
        metadata.head_dim,
    );
    for layer_idx in if with_suffix {
        vec![0usize, 1, 2, 4]
    } else {
        vec![0usize, 1, 2]
    } {
        kv_cache.init_ssm_state(
            layer_idx,
            conv_kernel,
            conv_channels,
            dt_rank,
            d_inner / dt_rank,
            d_state,
        );
    }
    Engine {
        tokenizer,
        kv_cache,
        metadata,
        architecture: ModelArchitecture::Qwen35,
        weights: Some(weights),
        scratch: Some(scratch),
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[cfg(feature = "vulkan")]
fn make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(vocab_size: usize) -> Engine {
    let hidden_dim = 32;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let ffn_inner_dim = 32;
    let d_inner = 32;
    let d_state = 1;
    let n_group = 1;
    let dt_rank = 1;
    let conv_kernel = 2;
    let conv_channels = d_inner + 2 * n_group * d_state;
    let tokens: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
    let special = SpecialTokens {
        bos: 1,
        eos: 2,
        pad: None,
    };
    let vocab = Vocab::new(tokens, special);
    let tokenizer = BpeTokenizer::new(vocab, vec![]);
    let metadata = ModelMetadata {
        num_layers: 5,
        num_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len: 16,
        hidden_dim,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: d_inner,
        ssm_d_state: d_state,
        ssm_n_group: n_group,
        ssm_dt_rank: dt_rank,
        ssm_conv_kernel: conv_kernel,
        full_attention_interval: 4,
    };
    let token_embd = make_f32_weight(
        vocab_size,
        hidden_dim,
        (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.03)
            .collect(),
    );
    let zero_qkv = vec![0.0; conv_channels * hidden_dim];
    let zero_gate = vec![0.0; d_inner * hidden_dim];
    let zero_conv = vec![0.0; conv_kernel * conv_channels];
    let zero_ssm_out = vec![0.0; hidden_dim * d_inner];
    let mut layers = vec![];
    for _ in 0..3 {
        layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
            attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
            qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv.clone()),
            gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate.clone()),
            ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
            ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
            ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
            ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
            ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out.clone()),
            ffn_gate_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_up_weight: make_f32_weight(
                ffn_inner_dim,
                hidden_dim,
                vec![0.0; ffn_inner_dim * hidden_dim],
            ),
            ffn_down_weight: make_f32_weight(
                hidden_dim,
                ffn_inner_dim,
                vec![0.0; hidden_dim * ffn_inner_dim],
            ),
            ffn_gate_up_fused: None,
            shared_expert_moe: None,
        }));
    }
    layers.push(LayerType::Attention(AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        q_weight: make_identity_q8_0_weight(q_dim, hidden_dim),
        k_weight: make_identity_q8_0_weight(kv_dim, hidden_dim),
        v_weight: make_identity_q8_0_weight(kv_dim, hidden_dim),
        o_weight: make_identity_q8_0_weight(hidden_dim, q_dim),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        ffn_gate_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
        ffn_up_weight: make_zero_q8_0_weight(ffn_inner_dim, hidden_dim),
        ffn_down_weight: make_zero_q8_0_weight(hidden_dim, ffn_inner_dim),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    }));
    layers.push(LayerType::GatedDeltaNet(GdnLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        qkv_weight: make_f32_weight(conv_channels, hidden_dim, zero_qkv),
        gate_weight: make_f32_weight(d_inner, hidden_dim, zero_gate),
        ssm_a: Tensor::from_slice(&[0.0f32; 1], &[1]),
        ssm_alpha: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
        ssm_beta: make_f32_weight(1, hidden_dim, vec![0.0; hidden_dim]),
        ssm_conv1d: Tensor::from_slice(&zero_conv, &[conv_kernel, conv_channels]),
        ssm_dt_bias: Tensor::from_slice(&[0.0f32; 1], &[1]),
        ssm_norm: Tensor::from_slice(&vec![1.0f32; d_inner], &[d_inner]),
        ssm_out: make_f32_weight(hidden_dim, d_inner, zero_ssm_out),
        ffn_gate_weight: make_f32_weight(
            ffn_inner_dim,
            hidden_dim,
            vec![0.0; ffn_inner_dim * hidden_dim],
        ),
        ffn_up_weight: make_f32_weight(
            ffn_inner_dim,
            hidden_dim,
            vec![0.0; ffn_inner_dim * hidden_dim],
        ),
        ffn_down_weight: make_f32_weight(
            hidden_dim,
            ffn_inner_dim,
            vec![0.0; hidden_dim * ffn_inner_dim],
        ),
        ffn_gate_up_fused: None,
        shared_expert_moe: None,
    }));
    let weights = ModelWeights {
        token_embd,
        output_norm: Tensor::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]),
        output: make_f32_weight(vocab_size, hidden_dim, vec![0.0; vocab_size * hidden_dim]),
        layers,
        gemma_per_layer: None,
        rope_freqs: None,
    };
    let scratch = ScratchBuffers::new(&metadata, ffn_inner_dim);
    let mut kv_cache = KVCache::new(
        metadata.num_layers,
        metadata.max_seq_len,
        metadata.num_kv_heads,
        metadata.head_dim,
    );
    for layer_idx in [0usize, 1, 2, 4] {
        kv_cache.init_ssm_state(
            layer_idx,
            conv_kernel,
            conv_channels,
            dt_rank,
            d_inner / dt_rank,
            d_state,
        );
    }
    Engine {
        tokenizer,
        kv_cache,
        metadata,
        architecture: ModelArchitecture::Qwen35,
        weights: Some(weights),
        scratch: Some(scratch),
        backend_runtime: backend_runtime::EngineBackendRuntime::new(),
        packed_model: None,
        shadow_model: None,
        memtrace_step: std::sync::atomic::AtomicUsize::new(0),
        moe_section_decode_bytes: None,
        #[cfg(feature = "vulkan")]
        fullpath_token_embd_bound: false,
        last_layer_hidden_cached: Vec::new(),
    }
}

#[test]
fn test_engine_mock_forward_shape() {
    let mut engine = make_mock_engine(100);
    let logits = engine.forward(&[0, 1, 2]).expect("forward should succeed");
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_apply_model_norm_uses_plain_rms_for_gemma() {
    let input = Tensor::from_slice(&[1.0f32, 3.0, 10.0, 14.0], &[2, 2]);
    let weight = Tensor::from_slice(&[0.0f32, -0.5], &[2]);
    let normed = apply_model_norm(&input, &weight, 1e-5, ModelArchitecture::Gemma)
        .expect("gemma norm should succeed");
    let data = kernels::tensor_as_f32_slice(&normed);

    assert!(data[0].abs() < 1e-6);
    assert!(data[1] < -0.67 && data[1] > -0.68);
    assert!(data[2].abs() < 1e-6);
    assert!(data[3] < -0.57 && data[3] > -0.58);
}

#[test]
fn test_apply_model_gate_mul_inplace_uses_gelu_for_gemma() {
    let mut gemma_gate = vec![1.0f32];
    let mut llama_gate = vec![1.0f32];
    let up = vec![2.0f32];

    apply_model_gate_mul_inplace(&mut gemma_gate, &up, ModelArchitecture::Gemma);
    apply_model_gate_mul_inplace(&mut llama_gate, &up, ModelArchitecture::LLaMA);

    assert!((gemma_gate[0] - 1.6824).abs() < 1e-3);
    assert!((llama_gate[0] - 1.4621).abs() < 1e-3);
    assert!(gemma_gate[0] > llama_gate[0]);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_selector_defaults_to_layer_zero() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
    }

    assert!(super::mediatek_ffn::selected_layer_matches(0).expect("default layer 0"));
    assert!(!super::mediatek_ffn::selected_layer_matches(1).expect("layer 1 not selected"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_selector_rejects_all_until_cache_exists() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYER", "all");
    }

    let err = super::mediatek_ffn::selected_layer_matches(0)
        .expect_err("all must be rejected until compiled cache exists");

    assert_eq!(err.trace_reason(), "all_layers_require_compiled_cache");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_cache_cap_defaults_to_two_and_uses_valid_override() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
    assert_eq!(super::mediatek_ffn::cache_max_layers(), 2);

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "4");
    }
    assert_eq!(super::mediatek_ffn::cache_max_layers(), 4);

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "5");
    }
    assert_eq!(super::mediatek_ffn::cache_max_layers(), 2);

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_cache_cap_reports_invalid_override() {
    assert_eq!(
        super::mediatek_ffn::cache_max_layers_from_env_value(None),
        (2, false)
    );
    assert_eq!(
        super::mediatek_ffn::cache_max_layers_from_env_value(Some("4")),
        (4, false)
    );
    assert_eq!(
        super::mediatek_ffn::cache_max_layers_from_env_value(Some("5")),
        (2, true)
    );
    assert_eq!(
        super::mediatek_ffn::cache_max_layers_from_env_value(Some("foo")),
        (2, true)
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_cache_cap_uses_separate_hard_max() {
    assert_eq!(
        rnb_runtime::mediatek::prefill_cache_max_layers_from_env_value(None),
        (2, false)
    );
    assert_eq!(
        rnb_runtime::mediatek::prefill_cache_max_layers_from_env_value(Some("10")),
        (10, false)
    );
    assert_eq!(
        rnb_runtime::mediatek::prefill_cache_max_layers_from_env_value(Some("11")),
        (2, true)
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_layer_list_allows_ten_while_single_token_stays_capped() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,1,2,3,4,5,6,7,8,9");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "10");
    }

    let single_token = super::mediatek_ffn::selected_layer_decision(0, true)
        .expect_err("single-token path must keep hard cap 4");
    let prefill = super::mediatek_ffn::decide_prefill_npu(
        9,
        384,
        rnb_runtime::mediatek::MediaTekPrefillCacheState::InProcessThreadLocalHot,
        rnb_runtime::mediatek::MediaTekPrefillRequestMode::UserPath,
        false,
    );

    assert_eq!(single_token.trace_reason(), "layer_list_exceeds_cache_cap");
    assert_eq!(
        prefill,
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::UseWarmNpu {
            max_cache_entries: 10
        }
    );

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_batch_plan_uses_fixed_128_chunks() {
    assert_eq!(super::mediatek_ffn::prefill_batch_plan(59), None);
    assert_eq!(
        super::mediatek_ffn::prefill_batch_plan(128),
        Some(super::mediatek_ffn::MediaTekPrefillBatchPlan {
            batch_size: 128,
            full_chunks: 1,
            tail_tokens: 0,
        })
    );
    assert_eq!(
        super::mediatek_ffn::prefill_batch_plan(443),
        Some(super::mediatek_ffn::MediaTekPrefillBatchPlan {
            batch_size: 128,
            full_chunks: 3,
            tail_tokens: 59,
        })
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_tail_uses_quantized_cpu_without_host_materialization() {
    assert!(!super::mediatek_ffn::prefill_needs_host_weights(
        true, false
    ));
    assert!(super::mediatek_ffn::prefill_needs_host_weights(
        false, false
    ));
    assert!(super::mediatek_ffn::prefill_needs_host_weights(true, true));

    let hidden_dim = 2;
    let ffn_inner = 3;
    let tail_tokens = 2;
    let norm_data = vec![1.0, -0.5, 0.25, 2.0];
    let gate = make_f32_weight(ffn_inner, hidden_dim, vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.25]);
    let up = make_f32_weight(ffn_inner, hidden_dim, vec![0.25, 0.5, -1.0, 0.75, 0.5, 0.5]);
    let down = make_f32_weight(hidden_dim, ffn_inner, vec![1.0, 0.0, 0.5, -0.25, 0.75, 1.0]);

    let got = super::mediatek_ffn::prefill_tail_down_with_quantized_cpu(
        ModelArchitecture::Gemma,
        hidden_dim,
        &norm_data,
        &gate,
        &up,
        &down,
        tail_tokens,
    )
    .expect("tail should use production quantized CPU path");

    let (mut expected_gate, expected_up) = super::quantized_dispatch::prefill_gate_up_vectors(
        &gate,
        &up,
        None,
        &norm_data,
        tail_tokens,
    )
    .expect("production gate/up path");
    apply_model_gate_mul_inplace(&mut expected_gate, &expected_up, ModelArchitecture::Gemma);
    let expected = down.gemv_vec(&expected_gate).expect("production down path");

    assert_eq!(got, expected);
    assert_eq!(got.len(), tail_tokens * hidden_dim);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_list_requires_compiled_reuse() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,1");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }

    let err = super::mediatek_ffn::selected_layer_decision(0, false)
        .expect_err("layer list without compiled reuse must fail closed");

    assert_eq!(err.trace_reason(), "layer_list_requires_compiled_reuse");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_list_selects_only_listed_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,2");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "2");
    }

    let layer0 = super::mediatek_ffn::selected_layer_decision(0, true).expect("valid list");
    let layer1 = super::mediatek_ffn::selected_layer_decision(1, true).expect("valid list");
    let layer2 = super::mediatek_ffn::selected_layer_decision(2, true).expect("valid list");

    assert!(layer0.selected);
    assert!(!layer1.selected);
    assert!(layer2.selected);
    assert_eq!(layer0.max_cache_entries, 2);
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_list_rejects_duplicates_and_bad_tokens() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "4");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,0");
    }
    let duplicate = super::mediatek_ffn::selected_layer_decision(0, true)
        .expect_err("duplicates must be rejected");
    assert_eq!(duplicate.trace_reason(), "invalid_layer_list");

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,nope");
    }
    let bad_token = super::mediatek_ffn::selected_layer_decision(0, true)
        .expect_err("bad token must be rejected");
    assert_eq!(bad_token.trace_reason(), "invalid_layer_list");

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_list_rejects_list_longer_than_cache_cap() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "0,1,2");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "2");
    }

    let err = super::mediatek_ffn::selected_layer_decision(0, true)
        .expect_err("over-cap layer list must fail closed");

    assert_eq!(err.trace_reason(), "layer_list_exceeds_cache_cap");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_enable_env_requires_exact_one() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
    }
    assert!(!super::mediatek_ffn::opt_in_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "0");
    }
    assert!(!super::mediatek_ffn::opt_in_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "true");
    }
    assert!(!super::mediatek_ffn::opt_in_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "1");
    }
    assert!(super::mediatek_ffn::opt_in_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", " 1 ");
    }
    assert!(!super::mediatek_ffn::opt_in_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_compiled_reuse_defaults_on_with_exact_zero_opt_out() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
    }
    assert!(super::mediatek_ffn::compiled_reuse_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "0");
    }
    assert!(!super::mediatek_ffn::compiled_reuse_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "false");
    }
    assert!(super::mediatek_ffn::compiled_reuse_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quantized_env_is_exact_opt_in() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED");
    }
    assert!(!super::mediatek_ffn::quantized_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED", "0");
    }
    assert!(!super::mediatek_ffn::quantized_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED", "true");
    }
    assert!(!super::mediatek_ffn::quantized_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED", "1");
    }
    assert!(super::mediatek_ffn::quantized_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quantized_stage_probe_env_requires_exact_one() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE");
    }
    assert!(!super::mediatek_ffn::quantized_stage_probe_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE", "0");
    }
    assert!(!super::mediatek_ffn::quantized_stage_probe_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE", "true");
    }
    assert!(!super::mediatek_ffn::quantized_stage_probe_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE", "1");
    }
    assert!(super::mediatek_ffn::quantized_stage_probe_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_env_is_exact_opt_in() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_PREFILL");
    }
    assert!(!super::mediatek_ffn::prefill_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_PREFILL", "0");
    }
    assert!(!super::mediatek_ffn::prefill_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_PREFILL", "true");
    }
    assert!(!super::mediatek_ffn::prefill_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_PREFILL", "1");
    }
    assert!(super::mediatek_ffn::prefill_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_PREFILL");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_cache_presence_user_path_keeps_disk_aot_non_hot() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }

    let hot = super::mediatek_ffn::decide_prefill_npu_for_cache_presence(
        0,
        384,
        true,
        false,
        rnb_runtime::mediatek::MediaTekPrefillRequestMode::UserPath,
        false,
    );
    assert_eq!(
        hot,
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::UseWarmNpu {
            max_cache_entries: 2
        }
    );

    let decision = super::mediatek_ffn::decide_prefill_npu_for_cache_presence(
        0,
        384,
        false,
        true,
        rnb_runtime::mediatek::MediaTekPrefillRequestMode::UserPath,
        true,
    );

    assert_eq!(
        decision,
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::FallbackCpu {
            reason: rnb_runtime::mediatek::MediaTekPrefillFallbackReason::CacheMissUserPath
        }
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_prefill_bench_warmup_routes_to_compile_policy() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
    }

    let decision = super::mediatek_ffn::decide_prefill_npu_for_cache_presence(
        0,
        384,
        false,
        false,
        rnb_runtime::mediatek::MediaTekPrefillRequestMode::BenchWarmup,
        true,
    );

    assert_eq!(
        decision,
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::AllowPrewarmCompile {
            max_cache_entries: 2
        }
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_params_cover_finite_min_max_range() {
    let params =
        super::mediatek_ffn::quant_params_for_quantized_gated_gelu(&[&[-2.0, 0.0], &[3.0]]);

    assert_eq!(params.zero_point(), 102);
    assert!((params.scale() - (5.0 / 255.0)).abs() < 1e-7);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quantized_weight_params_use_asymmetric_min_max_range() {
    let params =
        super::mediatek_ffn::quant_params_for_quantized_gated_gelu_weight(&[-1.0, 2.0, 3.0]);

    assert_eq!(params.zero_point(), 64);
    assert!((params.scale() - (4.0 / 255.0)).abs() < 1e-7);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quantized_stage_oracle_tiny_shape_has_expected_lengths() {
    let admission = super::mediatek_ffn::build_quantized_gated_gelu_admission(
        rnb_loader::Architecture::Gemma,
        &[0.25, -0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        2,
        true,
    )
    .expect("tiny quantized admission");
    let stages = admission.stage_references.expect("stage references");
    assert_eq!(stages.len(), 7);
    for stage in &stages {
        let expected_len = match stage.stage {
            rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Output => 2,
            _ => 2,
        };
        assert_eq!(stage.f32_reference.len(), expected_len);
        assert_eq!(stage.cpu_quantized_reference.len(), expected_len);
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_has_gate_and_up_with_expected_lengths() {
    let admission = super::mediatek_ffn::build_quantized_gated_gelu_admission(
        rnb_loader::Architecture::Gemma,
        &[0.25, -0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        2,
        true,
    )
    .expect("tiny quantized admission");
    let splits = admission.splits.expect("splits");
    assert_eq!(splits.len(), 2);
    assert_eq!(
        splits[0].stage,
        rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::GateFc
    );
    assert_eq!(
        splits[1].stage,
        rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::UpFc
    );
    for split in &splits {
        assert_eq!(split.f32_reference.len(), 2);
        assert_eq!(split.full_w8a8.len(), 2);
        assert_eq!(split.act_only.len(), 2);
        assert_eq!(split.weight_only.len(), 2);
        assert_eq!(split.output_requant_only.len(), 2);
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_dominant_classifies_sources() {
    use super::mediatek_ffn::{quantized_split_dominant, MediaTekFfnParityStats};
    let pass = MediaTekFfnParityStats {
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        cosine_similarity: 1.0,
    };
    let fail = |abs: f32| MediaTekFfnParityStats {
        max_abs_error: abs,
        max_rel_error: 10.0,
        cosine_similarity: 0.99,
    };

    assert_eq!(
        quantized_split_dominant(fail(0.20), fail(0.18), pass, pass),
        "activation"
    );
    assert_eq!(
        quantized_split_dominant(fail(0.20), pass, fail(0.18), pass),
        "weight"
    );
    assert_eq!(
        quantized_split_dominant(fail(0.20), pass, pass, fail(0.18)),
        "output_requant"
    );
    // act barely fails but full error far exceeds any single source => residual dominates.
    assert_eq!(
        quantized_split_dominant(fail(1.0), fail(0.1), pass, pass),
        "accumulation_residual"
    );
    // all single sources pass but full fails => residual boundary.
    assert_eq!(
        quantized_split_dominant(fail(0.2), pass, pass, pass),
        "accumulation_residual"
    );
    assert_eq!(quantized_split_dominant(pass, pass, pass, pass), "none");
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_dominant_uses_parity_predicate_not_max_abs() {
    use super::mediatek_ffn::{quantized_split_dominant, MediaTekFfnParityStats};
    // weight has larger max_abs but still passes parity (tiny rel + high cosine);
    // activation has smaller max_abs but breaks cosine, so activation is dominant.
    let full = MediaTekFfnParityStats {
        max_abs_error: 0.05,
        max_rel_error: 10.0,
        cosine_similarity: 0.99,
    };
    let act = MediaTekFfnParityStats {
        max_abs_error: 0.05,
        max_rel_error: 10.0,
        cosine_similarity: 0.99,
    };
    let weight = MediaTekFfnParityStats {
        max_abs_error: 0.5,
        max_rel_error: 0.0,
        cosine_similarity: 1.0,
    };
    let output_requant = MediaTekFfnParityStats {
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        cosine_similarity: 1.0,
    };
    assert_eq!(
        quantized_split_dominant(full, act, weight, output_requant),
        "activation"
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_line_includes_all_terms() {
    use super::mediatek_ffn::{quantized_split_line, MediaTekFfnParityStats};
    let stats = MediaTekFfnParityStats {
        max_abs_error: 1.0,
        max_rel_error: 2.0,
        cosine_similarity: 0.9,
    };
    let line = quantized_split_line(0, "gate_fc", stats, stats, stats, stats, "activation");
    assert!(line.contains("stage=gate_fc"));
    assert!(line.contains("full_w8a8_max_abs=1.000000000"));
    assert!(line.contains("act_quant_max_abs=1.000000000"));
    assert!(line.contains("weight_quant_max_abs=1.000000000"));
    assert!(line.contains("output_requant_max_abs=1.000000000"));
    assert!(line.contains("dominant=activation"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_act_only_tracks_activation_conditioning() {
    use super::mediatek_ffn::{build_quantized_gated_gelu_admission, parity_stats};
    // Identity weights (near-lossless weight quant) so the act_only term isolates activation
    // quantization. A wide-range activation must produce a larger act_only error than a
    // well-conditioned one; if act_only accidentally quantized weights this would not hold.
    let act_err = |norm: &[f32]| {
        let admission = build_quantized_gated_gelu_admission(
            rnb_loader::Architecture::Gemma,
            norm,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            2,
            true,
        )
        .expect("admission");
        let splits = admission.splits.expect("splits");
        let gate = &splits[0];
        let act = parity_stats(&gate.f32_reference, &gate.act_only).expect("act");
        let weight = parity_stats(&gate.f32_reference, &gate.weight_only).expect("weight");
        (act.max_abs_error, weight.max_abs_error)
    };
    let (hard_act, hard_weight) = act_err(&[100.0, 0.05]);
    let (easy_act, _easy_weight) = act_err(&[1.0, 1.0]);
    assert!(
        hard_act > easy_act * 5.0,
        "wide-range activation must inflate act_only error: hard={hard_act} easy={easy_act}"
    );
    // identity weights are quantized losslessly, so weight_only stays negligible regardless.
    assert!(
        hard_weight < hard_act,
        "identity weight quant error should stay below activation error: weight={hard_weight} act={hard_act}"
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quant_split_weight_only_tracks_weight_conditioning() {
    use super::mediatek_ffn::{build_quantized_gated_gelu_admission, parity_stats};
    // Fixed well-conditioned activation so the weight_only term isolates weight quantization.
    // A wide-range weight matrix must produce a larger weight_only error than a well-conditioned
    // one; if weight_only accidentally quantized the activation this would not hold.
    let weight_err = |weight: Vec<f32>| {
        let admission = build_quantized_gated_gelu_admission(
            rnb_loader::Architecture::Gemma,
            &[1.0, 1.0],
            weight.clone(),
            weight,
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            2,
            true,
        )
        .expect("admission");
        let splits = admission.splits.expect("splits");
        let gate = &splits[0];
        let act = parity_stats(&gate.f32_reference, &gate.act_only).expect("act");
        let weight = parity_stats(&gate.f32_reference, &gate.weight_only).expect("weight");
        (weight.max_abs_error, act.max_abs_error)
    };
    let (hard_weight, hard_act) = weight_err(vec![100.0, 0.05, 0.05, 100.0]);
    let (easy_weight, _easy_act) = weight_err(vec![1.0, 1.0, 2.0, 2.0]);
    assert!(
        hard_weight > easy_weight * 5.0,
        "wide-range weights must inflate weight_only error: hard={hard_weight} easy={easy_weight}"
    );
    // well-conditioned activation is quantized losslessly, so act_only stays negligible.
    assert!(
        hard_act < hard_weight,
        "lossless activation quant error should stay below weight error: act={hard_act} weight={hard_weight}"
    );
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_quantized_fc_cpu_oracle_uses_integer_accumulator() {
    let output = super::mediatek_ffn::quantized_fc_output_reference(
        &[130, 126],
        rnb_runtime::mediatek::MediaTekQuantParams::new(0.5, 128),
        &[129, 127],
        rnb_runtime::mediatek::MediaTekQuantParams::new(0.25, 128),
        rnb_runtime::mediatek::MediaTekQuantParams::new(0.5, 128),
        1,
        2,
        "fc",
    )
    .expect("quantized fc oracle");

    assert_eq!(output.len(), 1);
    assert!((output[0] - 0.5).abs() < 1e-7);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_trace_env_is_exact_opt_in() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_TRACE");
    }
    assert!(!super::mediatek_ffn::trace_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_TRACE", "0");
    }
    assert!(!super::mediatek_ffn::trace_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_TRACE", "true");
    }
    assert!(!super::mediatek_ffn::trace_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_TRACE", " 1 ");
    }
    assert!(!super::mediatek_ffn::trace_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_TRACE", "1");
    }
    assert!(super::mediatek_ffn::trace_enabled());

    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_TRACE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_measure_timing_is_trace_or_parity() {
    assert!(!super::mediatek_ffn::mediatek_ffn_measure_timing(
        false, false
    ));
    assert!(super::mediatek_ffn::mediatek_ffn_measure_timing(
        true, false
    ));
    assert!(super::mediatek_ffn::mediatek_ffn_measure_timing(
        false, true
    ));
    assert!(super::mediatek_ffn::mediatek_ffn_measure_timing(true, true));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_stage_line_formats_both_comparisons() {
    let line = super::mediatek_ffn::stage_probe_line(
        0,
        rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Output,
        super::mediatek_ffn::MediaTekFfnParityStats {
            max_abs_error: 1.0,
            max_rel_error: 2.0,
            cosine_similarity: 0.9,
        },
        super::mediatek_ffn::MediaTekFfnParityStats {
            max_abs_error: 3.0,
            max_rel_error: 4.0,
            cosine_similarity: 0.8,
        },
        Some(super::mediatek_ffn::MediaTekFfnParityStats {
            max_abs_error: 5.0,
            max_rel_error: 6.0,
            cosine_similarity: 0.7,
        }),
    );

    assert!(line.contains("stage=output"));
    assert!(line.contains("nnapi_vs_cpu_quant_max_abs=1.000000000"));
    assert!(line.contains("cpu_quant_vs_f32_max_abs=3.000000000"));
    assert!(line.contains("nnapi_vs_f32_max_abs=5.000000000"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_device_defaults_to_mtk_neuron_and_rejects_empty_override() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }
    assert_eq!(
        super::mediatek_ffn::resolve_device_name().expect("default device"),
        "mtk-neuron"
    );

    unsafe {
        std::env::set_var("RNB_MEDIATEK_DEVICE", "  ");
    }
    let err = super::mediatek_ffn::resolve_device_name()
        .expect_err("blank explicit device must be rejected");
    assert_eq!(err.trace_reason(), "invalid_device");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_env_off_does_not_materialize_weights() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
    }
    let bad_gate = QuantizedWeight::new(Tensor::from_vec(vec![0.0], &[1]), GGMLType::F32, 3, 4);
    let bad_up = make_f32_weight(3, 4, vec![0.0; 12]);
    let bad_down = make_f32_weight(4, 3, vec![0.0; 12]);
    let norm = vec![0.5f32; 4];

    let result = super::mediatek_ffn::try_mediatek_gemma_ffn_down(
        ModelArchitecture::Gemma4,
        0,
        4,
        &norm,
        &bad_gate,
        &bad_up,
        &bad_down,
        false,
    )
    .expect("env off should return Ok(None) before materialization");

    assert!(result.is_none());
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_layer_all_falls_back_without_error() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYER", "all");
        std::env::remove_var("RNB_DUMP_BIN_DIR");
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }
    let gate = make_f32_weight(3, 4, vec![0.0; 12]);
    let up = make_f32_weight(3, 4, vec![0.0; 12]);
    let down = make_f32_weight(4, 3, vec![0.0; 12]);
    let norm = vec![0.5f32; 4];

    let result = super::mediatek_ffn::try_mediatek_gemma_ffn_down(
        ModelArchitecture::Gemma4,
        0,
        4,
        &norm,
        &gate,
        &up,
        &down,
        false,
    )
    .expect("unsupported all selector should fall back, not abort");

    assert!(result.is_none());
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_env_on_host_falls_back_without_error() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYER", "0");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_PARITY");
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
        std::env::remove_var("RNB_DUMP_BIN_DIR");
    }

    let hidden_dim = 4usize;
    let ffn_inner = 3usize;
    let gate = make_f32_weight(ffn_inner, hidden_dim, vec![0.01; ffn_inner * hidden_dim]);
    let up = make_f32_weight(ffn_inner, hidden_dim, vec![0.02; ffn_inner * hidden_dim]);
    let down = make_f32_weight(hidden_dim, ffn_inner, vec![0.03; hidden_dim * ffn_inner]);
    let norm = vec![0.5f32; hidden_dim];

    let result = super::mediatek_ffn::try_mediatek_gemma_ffn_down(
        ModelArchitecture::Gemma4,
        0,
        hidden_dim,
        &norm,
        &gate,
        &up,
        &down,
        false,
    )
    .expect("host unsupported should fall back through Ok(None)");

    #[cfg(not(target_os = "android"))]
    assert!(result.is_none());
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_dump_mode_disables_hook_before_runtime_execution() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let tmp = tempfile::TempDir::new().expect("temp dump dir");
    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN", "1");
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYER", "0");
        std::env::set_var("RNB_DUMP_BIN_DIR", tmp.path());
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_PARITY");
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }

    let gate = make_f32_weight(3, 4, vec![0.0; 12]);
    let up = make_f32_weight(3, 4, vec![0.0; 12]);
    let down = make_f32_weight(4, 3, vec![0.0; 12]);
    let norm = vec![0.5f32; 4];

    let result = super::mediatek_ffn::try_mediatek_gemma_ffn_down(
        ModelArchitecture::Gemma4,
        0,
        4,
        &norm,
        &gate,
        &up,
        &down,
        false,
    )
    .expect("dump mode should return Ok(None) before runtime execution");

    assert!(result.is_none());
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN");
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
        std::env::remove_var("RNB_DUMP_BIN_DIR");
        std::env::remove_var("RNB_MEDIATEK_DEVICE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_cpu_reference_uses_norm_input_not_raw_weights() {
    let norm = [2.0f32, -1.0];
    let gate = [1.0f32, 0.0, 0.0, 1.0];
    let up = [0.5f32, 0.0, 0.0, -2.0];
    let down = [1.0f32, 0.0, 0.0, 1.0];

    let output = super::mediatek_ffn::cpu_reference_down(
        ModelArchitecture::Gemma4,
        &norm,
        &gate,
        &up,
        &down,
        2,
        2,
    )
    .expect("valid dense reference");

    assert_eq!(output.len(), 2);
    assert!((output[0] - 1.9546).abs() < 1.0e-3);
    assert!((output[1] + 0.3176).abs() < 1.0e-3);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_rejects_nonfinite_runtime_output_values() {
    assert!(super::mediatek_ffn::runtime_output_is_finite(&[0.0, 1.0]));
    assert!(!super::mediatek_ffn::runtime_output_is_finite(&[
        0.0,
        f32::NAN
    ]));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_parity_pass_line_includes_acceptance_fields() {
    let line = super::mediatek_ffn::parity_pass_line(
        0,
        super::mediatek_ffn::MediaTekFfnParityStats {
            max_abs_error: 0.000000011,
            max_rel_error: 0.0000248591,
            cosine_similarity: 1.0,
        },
    );

    assert!(line.contains("[mediatek-ffn] parity=pass layer=0"));
    assert!(line.contains("max_abs_error="));
    assert!(line.contains("max_rel_error="));
    assert!(line.contains("cosine_similarity="));
}
#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_timing_line_includes_cost_breakdown_fields() {
    let line = super::mediatek_ffn::timing_line(
        0,
        super::mediatek_ffn::MediaTekFfnLocalTimings {
            materialize_ns: 10,
            runtime_total_ns: 20,
            parity_ns: Some(30),
            cache_hit: false,
            total_ns: 40,
        },
        rnb_runtime::mediatek::RunGatedGeluFfnF32Timings {
            model_build_ns: 1,
            supported_ops_query_ns: 2,
            compilation_ns: 3,
            execution_setup_ns: 4,
            execution_compute_ns: 5,
            token_hash_ns: 0,
        },
        Some(6),
        None,
    );

    assert!(line.contains("[mediatek-ffn] timing layer=0"));
    assert!(line.contains("materialize_ns=10"));
    assert!(line.contains("runtime_total_ns=20"));
    assert!(line.contains("cache_hit=false"));
    assert!(line.contains("backend_model_build_ns=1"));
    assert!(line.contains("backend_supported_ops_query_ns=2"));
    assert!(line.contains("backend_compilation_ns=3"));
    assert!(line.contains("backend_execution_setup_ns=4"));
    assert!(line.contains("backend_execution_compute_ns=5"));
    assert!(line.contains("nnapi_hw_ns=6"));
    assert!(line.contains("nnapi_driver_ns=none"));
    assert!(line.contains("parity_ns=30"));
    assert!(line.contains("total_ns=40"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_timing_line_includes_cache_hit_field() {
    let line = super::mediatek_ffn::timing_line(
        0,
        super::mediatek_ffn::MediaTekFfnLocalTimings {
            materialize_ns: 10,
            runtime_total_ns: 20,
            parity_ns: None,
            total_ns: 40,
            cache_hit: true,
        },
        rnb_runtime::mediatek::RunGatedGeluFfnF32Timings {
            model_build_ns: 0,
            supported_ops_query_ns: 0,
            compilation_ns: 0,
            execution_setup_ns: 4,
            execution_compute_ns: 5,
            token_hash_ns: 0,
        },
        Some(6),
        Some(7),
    );

    assert!(line.contains("cache_hit=true"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_compiled_reuse_env_defaults_on_with_zero_opt_out() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
    }
    assert!(super::mediatek_ffn::compiled_reuse_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "1");
    }
    assert!(super::mediatek_ffn::compiled_reuse_enabled());

    unsafe {
        std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE", "0");
    }
    assert!(!super::mediatek_ffn::compiled_reuse_enabled());
    unsafe {
        std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE");
    }
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_weight_cache_identity_is_unique_per_weight() {
    let first = make_f32_weight(3, 4, vec![0.1; 12]);
    let second = make_f32_weight(3, 4, vec![0.1; 12]);

    let first_key = first
        .mediatek_gated_gelu_cache_weight_key()
        .expect("first f32 tensor exposes raw key");
    let second_key = second
        .mediatek_gated_gelu_cache_weight_key()
        .expect("second f32 tensor exposes raw key");

    assert_eq!(first_key.rows, second_key.rows);
    assert_eq!(first_key.cols, second_key.cols);
    assert_eq!(first_key.ggml_type, second_key.ggml_type);
    assert_ne!(first_key.generation_id, second_key.generation_id);
}
#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_parity_stats_accepts_close_output() {
    let reference = [1.0f32, -2.0, 3.0, 4.0];
    let candidate = [1.0f32 + 5.0e-5, -2.0, 3.0, 4.0];

    let stats = super::mediatek_ffn::parity_stats(&reference, &candidate)
        .expect("same length parity stats");

    assert!(stats.passed());
    assert!(stats.max_abs_error <= 1.0e-4);
    assert!(stats.cosine_similarity >= 0.999);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_parity_stats_rejects_finite_wrong_output() {
    let reference = [1.0f32, -2.0, 3.0, 4.0];
    let candidate = [-1.0f32, 2.0, -3.0, -4.0];

    let stats = super::mediatek_ffn::parity_stats(&reference, &candidate)
        .expect("same length parity stats");

    assert!(!stats.passed());
    assert!(stats.cosine_similarity < 0.0);
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_materializes_f32_weight_and_rejects_bad_shape() {
    let weight = make_f32_weight(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        weight
            .materialize_f32_owned("test_weight")
            .expect("valid f32 weight materializes"),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let bad = QuantizedWeight::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), GGMLType::F32, 2, 2);
    let err = bad
        .materialize_f32_owned("bad_weight")
        .expect_err("declared rows*cols must match materialized values");
    assert!(err.to_string().contains("materialized len mismatch"));
}

#[cfg(feature = "mediatek")]
#[test]
fn mediatek_ffn_materialization_rejects_nonfinite_values() {
    let weight = make_f32_weight(1, 2, vec![0.0, f32::NAN]);
    let err = weight
        .materialize_f32_owned("nan_weight")
        .expect_err("non-finite f32 materialization must be rejected");
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn test_resolve_rope_params_uses_swa_theta_for_smaller_gemma_heads() {
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    assert_eq!(
        resolve_rope_params(&metadata, ModelArchitecture::Gemma, 0, 256),
        (256, 10_000.0, false)
    );
    assert_eq!(
        resolve_rope_params(&metadata, ModelArchitecture::Gemma, 5, 512),
        (128, 1_000_000.0, true)
    );
}

#[test]
fn test_qwen_text_mrope_dim_is_diagnostic_opt_in() {
    let metadata = ModelMetadata {
        num_layers: 40,
        num_heads: 16,
        num_kv_heads: 2,
        head_dim: 256,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 10_000_000.0,
        rope_theta_swa: 10_000_000.0,
        rope_dim: 64,
        rope_dim_swa: 64,
        rope_sections: [11, 11, 10, 0],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 8,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 4,
    };

    assert_eq!(
        qwen_text_mrope_dim(&metadata, ModelArchitecture::Qwen35MoE, 64, 256),
        None
    );
    assert_eq!(
        qwen_text_mrope_dim(&metadata, ModelArchitecture::Gemma, 64, 256),
        None
    );
}

#[test]
fn test_apply_logit_softcapping_clamps_large_logits() {
    let mut logits = vec![-100.0f32, 0.0, 100.0];
    apply_logit_softcapping(&mut logits, 30.0);

    assert!(logits[0] > -30.0 && logits[0] < -29.9);
    assert_eq!(logits[1], 0.0);
    assert!(logits[2] < 30.0 && logits[2] > 29.9);
}

#[test]
fn test_resolve_attention_scale_is_one_for_gemma() {
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 2560,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 512,
        shared_kv_layers: 18,
        sliding_window_pattern: vec![true],
        key_length_full: 0,
        key_length_swa: 256,
        value_length_swa: 256,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 256,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    let scale = resolve_attention_scale(&metadata, ModelArchitecture::Gemma);
    assert!((scale - 1.0).abs() < 1e-6, "got {scale}");
}

#[test]
fn test_apply_rms_norm_no_scale_into_normalizes_without_weight() {
    let input = [3.0f32, 4.0];
    let mut output = [0.0f32; 2];
    apply_rms_norm_no_scale_into(&input, 0.0, &mut output);
    assert!((output[0] - 0.8485281).abs() < 1e-6, "got {}", output[0]);
    assert!((output[1] - 1.1313709).abs() < 1e-6, "got {}", output[1]);
}

#[test]
fn test_shared_kv_source_layer_maps_last_layers_to_front_span() {
    let metadata = ModelMetadata {
        num_layers: 42,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 2560,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 512,
        shared_kv_layers: 18,
        sliding_window_pattern: vec![true; 42],
        key_length_full: 0,
        key_length_swa: 256,
        value_length_swa: 256,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 256,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma, 23),
        None
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma, 24),
        Some(0)
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma, 29),
        Some(5)
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma, 41),
        Some(17)
    );
}

#[test]
fn test_shared_kv_source_layer_gemma4_boundary_pair() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_GEMMA_SHARED_KV_SOURCE_SWA");
        std::env::remove_var("RNB_GEMMA_SHARED_KV_SOURCE_FULL");
        std::env::remove_var("RNB_DISABLE_GEMMA_SHARED_KV_SWA");
        std::env::remove_var("RNB_DISABLE_GEMMA_SHARED_KV_FULL");
    }
    // Gemma4 E2B-it layout: 35 layers, shared_kv_layers=20, ISWA pattern with every 5th
    // layer being full attention. Contract §5: reused layers (il >= 15) pull from
    // layer 13 (SWA source) or layer 14 (full-attention source) regardless of il.
    let metadata = ModelMetadata {
        num_layers: 35,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 1536,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 512,
        shared_kv_layers: 20,
        sliding_window_pattern: vec![
            true, true, true, true, false, true, true, true, true, false, true, true, true, true,
            false, true, true, true, true, false, true, true, true, true, false, true, true, true,
            true, false, true, true, true, true, false,
        ],
        key_length_full: 0,
        key_length_swa: 256,
        value_length_swa: 256,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 256,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    // Layer 14 is the last layer that owns its own KV (kv_from_start=15).
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 14),
        None
    );
    // HF Gemma4 `L15.kv_shared_layer_index = 13` (SWA source = kv_from_start - 2).
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 15),
        Some(13)
    );
    // Layer 29 is full-attn (29 % 5 == 4) → reuse from layer 14 (full source = kv_from_start - 1).
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 29),
        Some(14)
    );
    // Layer 30 is SWA → reuse from layer 13.
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 30),
        Some(13)
    );
    // Layer 34 is full-attn (34 % 5 == 4) → reuse from layer 14.
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 34),
        Some(14)
    );
}

#[test]
fn test_shared_kv_source_layer_gemma4_can_disable_full_or_swa_reuse_for_diagnostics() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let metadata = ModelMetadata {
        num_layers: 35,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 1536,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 512,
        shared_kv_layers: 20,
        sliding_window_pattern: vec![
            true, true, true, true, false, true, true, true, true, false, true, true, true, true,
            false, true, true, true, true, false, true, true, true, true, false, true, true, true,
            true, false, true, true, true, true, false,
        ],
        key_length_full: 0,
        key_length_swa: 256,
        value_length_swa: 256,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 256,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    unsafe {
        std::env::set_var("RNB_DISABLE_GEMMA_SHARED_KV_FULL", "1");
    }
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 34),
        None
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 30),
        Some(13)
    );
    unsafe {
        std::env::remove_var("RNB_DISABLE_GEMMA_SHARED_KV_FULL");
        std::env::set_var("RNB_DISABLE_GEMMA_SHARED_KV_SWA", "1");
    }
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 30),
        None
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 34),
        Some(14)
    );
    unsafe {
        std::env::remove_var("RNB_DISABLE_GEMMA_SHARED_KV_SWA");
    }
}

#[test]
fn test_shared_kv_source_layer_gemma4_can_override_diagnostic_source_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let metadata = ModelMetadata {
        num_layers: 35,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 512,
        vocab_size: 32,
        max_seq_len: 64,
        hidden_dim: 1536,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 512,
        rope_dim_swa: 256,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 512,
        shared_kv_layers: 20,
        sliding_window_pattern: vec![
            true, true, true, true, false, true, true, true, true, false, true, true, true, true,
            false, true, true, true, true, false, true, true, true, true, false, true, true, true,
            true, false, true, true, true, true, false,
        ],
        key_length_full: 0,
        key_length_swa: 256,
        value_length_swa: 256,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 256,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    unsafe {
        std::env::set_var("RNB_GEMMA_SHARED_KV_SOURCE_SWA", "12");
        std::env::set_var("RNB_GEMMA_SHARED_KV_SOURCE_FULL", "13");
    }
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 30),
        Some(12)
    );
    assert_eq!(
        shared_kv_source_layer(&metadata, ModelArchitecture::Gemma4, 34),
        Some(13)
    );
    unsafe {
        std::env::remove_var("RNB_GEMMA_SHARED_KV_SOURCE_SWA");
        std::env::remove_var("RNB_GEMMA_SHARED_KV_SOURCE_FULL");
    }
}

#[test]
fn test_gemma_post_attn_layer_overrides_match_selected_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER", "14,34-35");
        std::env::set_var("RNB_GEMMA_SKIP_POST_ATTN_LAYER", "13-14");
    }
    assert!(!gemma_unit_offset_post_attn_enabled(13));
    assert!(gemma_unit_offset_post_attn_enabled(14));
    assert!(gemma_unit_offset_post_attn_enabled(34));
    assert!(gemma_skip_post_attn_enabled(13));
    assert!(gemma_skip_post_attn_enabled(14));
    assert!(!gemma_skip_post_attn_enabled(15));
    unsafe {
        std::env::remove_var("RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER");
        std::env::remove_var("RNB_GEMMA_SKIP_POST_ATTN_LAYER");
    }
}

#[test]
fn test_gemma4_e2bit_no_default_unit_offset_post_attn() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_GEMMA_UNIT_OFFSET_POST_ATTN_ONLY");
        std::env::remove_var("RNB_GEMMA_UNIT_OFFSET_POST_ATTN_LAYER");
    }

    for l in [12usize, 13, 14, 24, 34] {
        assert!(
            !gemma_default_unit_offset_post_attn_enabled(
                ModelArchitecture::Gemma4,
                GemmaRuntimeFlavor::Gemma4E2BIt,
                l,
            ),
            "post_attn unit-offset must be env-only for layer {l}"
        );
    }
}

#[test]
fn test_gemma_skip_ffn_layer_matches_selected_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_GEMMA_SKIP_FFN_LAYER", "14,19-20");
    }
    assert!(!gemma_skip_ffn_enabled(13));
    assert!(gemma_skip_ffn_enabled(14));
    assert!(gemma_skip_ffn_enabled(19));
    assert!(gemma_skip_ffn_enabled(20));
    assert!(!gemma_skip_ffn_enabled(21));
    unsafe {
        std::env::remove_var("RNB_GEMMA_SKIP_FFN_LAYER");
    }
}

#[test]
fn test_gemma_decode_only_skip_layer_matches_selected_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_GEMMA_SKIP_POST_ATTN_DECODE_LAYER", "19,24-25");
        std::env::set_var("RNB_GEMMA_SKIP_FFN_DECODE_LAYER", "29,33-34");
        std::env::set_var("RNB_GEMMA_DISABLE_ATTN_DECODE_LAYER", "14,19-20");
        std::env::set_var("RNB_GEMMA_DISABLE_LAYER_DECODE", "24,29-30");
        std::env::set_var("RNB_GEMMA_REUSE_SOURCE_HIDDEN_DECODE_LAYER", "19,24-25");
    }
    assert!(gemma_skip_post_attn_decode_enabled(19));
    assert!(gemma_skip_post_attn_decode_enabled(24));
    assert!(gemma_skip_post_attn_decode_enabled(25));
    assert!(!gemma_skip_post_attn_decode_enabled(26));
    assert!(gemma_skip_ffn_decode_enabled(29));
    assert!(gemma_skip_ffn_decode_enabled(33));
    assert!(gemma_skip_ffn_decode_enabled(34));
    assert!(!gemma_skip_ffn_decode_enabled(32));
    assert!(gemma_disable_attn_decode_enabled(14));
    assert!(gemma_disable_attn_decode_enabled(19));
    assert!(gemma_disable_attn_decode_enabled(20));
    assert!(!gemma_disable_attn_decode_enabled(21));
    assert!(gemma_disable_layer_decode_enabled(24));
    assert!(gemma_disable_layer_decode_enabled(29));
    assert!(gemma_disable_layer_decode_enabled(30));
    assert!(!gemma_disable_layer_decode_enabled(31));
    assert!(gemma_reuse_source_hidden_decode_enabled(19));
    assert!(gemma_reuse_source_hidden_decode_enabled(24));
    assert!(gemma_reuse_source_hidden_decode_enabled(25));
    assert!(!gemma_reuse_source_hidden_decode_enabled(26));
    unsafe {
        std::env::remove_var("RNB_GEMMA_SKIP_POST_ATTN_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_SKIP_FFN_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_DISABLE_ATTN_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_DISABLE_LAYER_DECODE");
        std::env::remove_var("RNB_GEMMA_REUSE_SOURCE_HIDDEN_DECODE_LAYER");
    }
}

#[test]
fn test_gemma_layer14_post_attn_decode_seams_are_env_only() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_GEMMA_POST_ATTN_DECODE_PLAIN_LAYER");
        std::env::remove_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_ALPHA");
        std::env::remove_var("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_ALPHA");
    }
    assert!(!gemma_post_attn_decode_plain_enabled(14));
    assert!(!gemma_post_attn_blend_source_decode_enabled(14));
    assert!((gemma_post_attn_blend_source_decode_alpha() - 0.25).abs() < 1e-6);
    assert!(!gemma_pre_residual_blend_source_decode_enabled(14));
    assert!((gemma_pre_residual_blend_source_decode_alpha() - 0.25).abs() < 1e-6);

    unsafe {
        std::env::set_var("RNB_GEMMA_POST_ATTN_DECODE_PLAIN_LAYER", "14,19");
        std::env::set_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_LAYER", "14");
        std::env::set_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_ALPHA", "0.4");
        std::env::set_var(
            "RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_LAYER",
            "14,19-20",
        );
        std::env::set_var("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_ALPHA", "0.15");
    }
    assert!(gemma_post_attn_decode_plain_enabled(14));
    assert!(gemma_post_attn_decode_plain_enabled(19));
    assert!(!gemma_post_attn_decode_plain_enabled(24));
    assert!(gemma_post_attn_blend_source_decode_enabled(14));
    assert!(!gemma_post_attn_blend_source_decode_enabled(19));
    assert!((gemma_post_attn_blend_source_decode_alpha() - 0.4).abs() < 1e-6);
    assert!(gemma_pre_residual_blend_source_decode_enabled(14));
    assert!(gemma_pre_residual_blend_source_decode_enabled(19));
    assert!(gemma_pre_residual_blend_source_decode_enabled(20));
    assert!(!gemma_pre_residual_blend_source_decode_enabled(21));
    assert!((gemma_pre_residual_blend_source_decode_alpha() - 0.15).abs() < 1e-6);

    unsafe {
        std::env::remove_var("RNB_GEMMA_POST_ATTN_DECODE_PLAIN_LAYER");
        std::env::remove_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_POST_ATTN_BLEND_SOURCE_DECODE_ALPHA");
        std::env::remove_var("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_LAYER");
        std::env::remove_var("RNB_GEMMA_PRE_RESIDUAL_BLEND_SOURCE_DECODE_ALPHA");
    }
}

#[test]
fn test_gemma4_e2bit_skip_ffn_has_no_default_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_GEMMA_SKIP_FFN_LAYER");
        std::env::remove_var("RNB_GEMMA_SKIP_FFN_DECODE_LAYER");
    }

    for l in [13usize, 14, 19, 22, 23, 24, 29, 33] {
        assert!(
            !gemma_default_skip_ffn_enabled(
                ModelArchitecture::Gemma4,
                GemmaRuntimeFlavor::Gemma4E2BIt,
                l,
            ),
            "skip_ffn default should be off for layer {l}"
        );
    }
}

#[test]
fn test_gemma4_e2bit_defaults_unit_offset_output_norm() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_DISABLE_GEMMA_E2BIT_LOCAL_DEFAULTS");
    }

    assert!(gemma_default_unit_offset_output_norm(
        ModelArchitecture::Gemma4,
        GemmaRuntimeFlavor::Gemma4E2BIt
    ));
    assert!(!gemma_default_unit_offset_output_norm(
        ModelArchitecture::Gemma,
        GemmaRuntimeFlavor::Gemma4E2BIt
    ));
    assert!(!gemma_default_unit_offset_output_norm(
        ModelArchitecture::Gemma4,
        GemmaRuntimeFlavor::Generic
    ));
}

#[test]
fn test_gemma4_e2bit_decode_output_norm_unit_offset_is_default_on() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_GEMMA_UNIT_OFFSET_OUTPUT_NORM");
        std::env::remove_var("RNB_GEMMA_DISABLE_OUTPUT_NORM_DECODE_UNIT_OFFSET");
        std::env::remove_var("RNB_DISABLE_GEMMA_E2BIT_LOCAL_DEFAULTS");
    }
    assert!(gemma_effective_unit_offset_output_norm_decode(
        ModelArchitecture::Gemma4,
        GemmaRuntimeFlavor::Gemma4E2BIt
    ));
    unsafe {
        std::env::set_var("RNB_GEMMA_DISABLE_OUTPUT_NORM_DECODE_UNIT_OFFSET", "1");
    }
    assert!(!gemma_effective_unit_offset_output_norm_decode(
        ModelArchitecture::Gemma4,
        GemmaRuntimeFlavor::Gemma4E2BIt
    ));
    unsafe {
        std::env::remove_var("RNB_GEMMA_DISABLE_OUTPUT_NORM_DECODE_UNIT_OFFSET");
        std::env::set_var("RNB_DISABLE_GEMMA_E2BIT_LOCAL_DEFAULTS", "1");
    }
    assert!(!gemma_effective_unit_offset_output_norm_decode(
        ModelArchitecture::Gemma4,
        GemmaRuntimeFlavor::Gemma4E2BIt
    ));
    unsafe {
        std::env::remove_var("RNB_DISABLE_GEMMA_E2BIT_LOCAL_DEFAULTS");
    }
}

#[test]
fn test_gemma4_never_uses_legacy_layer34_ple_hard_fix() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var(
            "RNB_MODEL",
            "/data/local/tmp/rnb/gemma-4-E2B-it-Q4_K_M.gguf",
        );
        // Legacy Gemma family still uses the layer-34 fix, but only when the caller
        // opts in via RNB_GEMMA_PLE_LAYER34_FIX. Gemma4 never uses the legacy path.
        std::env::set_var("RNB_GEMMA_PLE_LAYER34_FIX", "1");
    }

    assert!(!gemma_ple_layer34_hard_fix_applies(
        ModelArchitecture::Gemma4,
        34,
        35
    ));
    assert!(gemma_ple_layer34_hard_fix_applies(
        ModelArchitecture::Gemma,
        34,
        35
    ));

    unsafe {
        std::env::remove_var("RNB_MODEL");
        std::env::remove_var("RNB_GEMMA_PLE_LAYER34_FIX");
    }
}

#[test]
fn test_gemma_skip_out_scale_layer_matches_selected_layers() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_GEMMA_SKIP_OUT_SCALE_LAYER", "14,19-20");
    }
    assert!(!gemma_skip_out_scale_enabled(13));
    assert!(gemma_skip_out_scale_enabled(14));
    assert!(gemma_skip_out_scale_enabled(19));
    assert!(gemma_skip_out_scale_enabled(20));
    assert!(!gemma_skip_out_scale_enabled(21));
    unsafe {
        std::env::remove_var("RNB_GEMMA_SKIP_OUT_SCALE_LAYER");
    }
}

#[test]
fn test_gemma4_prefill_uses_f16_cache_by_default() {
    assert!(gemma4_prefill_uses_f16_cache(ModelArchitecture::Gemma4));
    assert!(!gemma4_prefill_uses_f16_cache(ModelArchitecture::Gemma));
    assert!(!gemma4_prefill_uses_f16_cache(ModelArchitecture::LLaMA));
}

#[test]
fn test_forward_attention_layer_gemma4_reused_layer_skips_kv_projection() {
    let metadata = ModelMetadata {
        num_layers: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 10_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 0,
        shared_kv_layers: 2,
        sliding_window_pattern: vec![true, false, true, false],
        key_length_full: 0,
        key_length_swa: 4,
        value_length_swa: 4,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };
    let hidden = Tensor::from_slice(&[1.0f32, 0.5, -0.25, 0.75], &[1, 4]);
    let mut kv_cache = KVCache::new_per_layer(metadata.max_seq_len, &[1, 1, 1, 1], &[4, 4, 4, 4]);
    kv_cache.append(0, 0, &[0.2, -0.1, 0.3, 0.4], &[0.5, 0.25, -0.5, 0.75]);

    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        q_weight: make_f32_weight(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
        // Intentionally malformed for hidden_dim=4. Reused Gemma4 layers must not touch them.
        k_weight: make_f32_weight(4, 3, vec![0.0; 12]),
        v_weight: make_f32_weight(4, 3, vec![0.0; 12]),
        o_weight: make_f32_weight(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        ffn_gate_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_up_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_down_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };

    let output = forward_attention_layer(
        &mut kv_cache,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden,
        &layer,
        None,
        2,
        1,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("reused Gemma4 layer should read cached KV without projecting K/V");

    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_decode_attention_layer_gemma4_reused_layer_skips_kv_projection() {
    let metadata = ModelMetadata {
        num_layers: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 10_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 0,
        shared_kv_layers: 2,
        sliding_window_pattern: vec![true, false, true, false],
        key_length_full: 0,
        key_length_swa: 4,
        value_length_swa: 4,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };
    let mut kv_cache = KVCache::new_per_layer(metadata.max_seq_len, &[1, 1, 1, 1], &[4, 4, 4, 4]);
    kv_cache.append(0, 0, &[0.2, -0.1, 0.3, 0.4], &[0.5, 0.25, -0.5, 0.75]);
    kv_cache.set_len(1);

    let mut scratch = ScratchBuffers::new(&metadata, 4);
    scratch.hidden[..4].copy_from_slice(&[1.0, 0.5, -0.25, 0.75]);

    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        q_weight: make_f32_weight(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
        // Intentionally malformed for hidden_dim=4. Reused Gemma4 layers must not touch them.
        k_weight: make_f32_weight(2, 4, vec![0.0; 8]),
        v_weight: make_f32_weight(2, 4, vec![0.0; 8]),
        o_weight: make_f32_weight(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        ffn_gate_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_up_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_down_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };

    decode_attention_layer(
        &mut kv_cache,
        &metadata,
        ModelArchitecture::Gemma4,
        &mut scratch,
        &layer,
        None,
        2,
        0,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
    .expect("reused Gemma4 decode layer should read cached KV without projecting K/V");
}

#[test]
fn test_gemma4_reused_layer_prefill_decode_match() {
    let metadata = ModelMetadata {
        num_layers: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 10_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 0,
        shared_kv_layers: 2,
        sliding_window_pattern: vec![true, false, true, false],
        key_length_full: 0,
        key_length_swa: 4,
        value_length_swa: 4,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };
    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        q_weight: make_f32_weight(
            4,
            4,
            vec![
                0.7, 0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.9, 0.1, 0.1, 0.0, 0.0, 1.0,
            ],
        ),
        k_weight: make_f32_weight(2, 4, vec![0.0; 8]),
        v_weight: make_f32_weight(2, 4, vec![0.0; 8]),
        o_weight: make_f32_weight(
            4,
            4,
            vec![
                1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 1.0,
            ],
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        ffn_gate_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_up_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_down_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };

    let mut kv_prefill = KVCache::new_per_layer(16, &[1, 1, 1, 1], &[4, 4, 4, 4]);
    kv_prefill.append(0, 0, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.4, 0.3, 0.2]);
    kv_prefill.append(0, 1, &[0.2, 0.1, 0.4, 0.3], &[0.6, 0.1, 0.2, 0.7]);

    let hidden_prefill = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.3, 0.1, -0.2, 0.4], &[2, 4]);
    let out_prefill = forward_attention_layer(
        &mut kv_prefill,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden_prefill,
        &layer,
        None,
        2,
        2,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("prefill reused layer should succeed");
    let out_prefill_data = kernels::tensor_as_f32_slice(&out_prefill);
    let prefill_last = &out_prefill_data[4..8];

    let mut kv_decode = KVCache::new_per_layer(16, &[1, 1, 1, 1], &[4, 4, 4, 4]);
    kv_decode.append(0, 0, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.4, 0.3, 0.2]);
    kv_decode.append(0, 1, &[0.2, 0.1, 0.4, 0.3], &[0.6, 0.1, 0.2, 0.7]);
    kv_decode.set_len(1);

    let mut scratch = ScratchBuffers::new(&metadata, 4);
    scratch.hidden[..4].copy_from_slice(&[0.3, 0.1, -0.2, 0.4]);
    decode_attention_layer(
        &mut kv_decode,
        &metadata,
        ModelArchitecture::Gemma4,
        &mut scratch,
        &layer,
        None,
        2,
        1,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
    .expect("decode reused layer should succeed");

    for (a, b) in scratch.hidden[..4].iter().zip(prefill_last.iter()) {
        assert!((a - b).abs() < 1e-4, "decode={a} prefill={b}");
    }
}

#[test]
fn test_gemma4_owned_full_attention_prefill_decode_match_with_norms() {
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 8,
        vocab_size: 32,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 10_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 0,
        rope_dim_swa: 0,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![false],
        key_length_full: 0,
        key_length_swa: 8,
        value_length_swa: 8,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32, 0.8, 1.1, 0.9], &[4]),
        q_weight: make_f32_weight(
            8,
            4,
            vec![
                0.7, 0.1, -0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, -0.1, 0.9, 0.1, 0.1, 0.0, 0.0, 1.0,
                0.6, 0.0, 0.1, -0.1, -0.1, 0.7, 0.0, 0.2, 0.1, 0.0, 0.8, 0.0, 0.0, 0.2, -0.1, 0.9,
            ],
        ),
        k_weight: make_f32_weight(
            8,
            4,
            vec![
                0.5, 0.0, 0.0, 0.1, 0.1, 0.6, 0.0, 0.0, 0.0, 0.1, 0.7, 0.0, 0.0, 0.0, 0.2, 0.8,
                0.4, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.6, 0.1, 0.1, 0.0, 0.0, 0.7,
            ],
        ),
        v_weight: make_f32_weight(
            8,
            4,
            vec![
                0.4, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.6, 0.1, 0.1, 0.0, 0.0, 0.7,
                0.3, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0, 0.1, 0.1, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0, 0.6,
            ],
        ),
        o_weight: make_f32_weight(
            4,
            8,
            vec![
                0.9, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1,
                0.1, 0.0, 0.9, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.8, 0.0, 0.1, 0.0, 0.2,
            ],
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: Some(Tensor::from_slice(
            &[1.0f32, 0.9, 1.1, 0.8, 1.0, 0.95, 0.85, 1.05],
            &[8],
        )),
        k_norm: Some(Tensor::from_slice(
            &[0.9f32, 1.0, 0.85, 1.1, 1.05, 0.95, 1.0, 0.9],
            &[8],
        )),
        post_attn_norm: Some(Tensor::from_slice(&[1.1f32, 0.85, 0.9, 1.05], &[4])),
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        ffn_gate_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_up_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_down_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };

    let hidden_prefill =
        Tensor::from_slice(&[0.4f32, -0.1, 0.2, 0.5, 0.3, 0.1, -0.2, 0.4], &[2, 4]);
    let mut kv_prefill = KVCache::new_per_layer(16, &[1], &[8]);
    let out_prefill = forward_attention_layer(
        &mut kv_prefill,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden_prefill,
        &layer,
        None,
        0,
        2,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("owned Gemma4 full-attn prefill should succeed");
    let out_prefill_data = kernels::tensor_as_f32_slice(&out_prefill);
    let prefill_last = &out_prefill_data[4..8];

    let mut kv_decode = KVCache::new_per_layer(16, &[1], &[8]);
    let hidden_first = Tensor::from_slice(&[0.4f32, -0.1, 0.2, 0.5], &[1, 4]);
    forward_attention_layer(
        &mut kv_decode,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden_first,
        &layer,
        None,
        0,
        1,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("first-token prefill should populate cache");
    kv_decode.set_len(1);

    let mut scratch = ScratchBuffers::new(&metadata, 4);
    scratch.hidden[..4].copy_from_slice(&[0.3, 0.1, -0.2, 0.4]);
    decode_attention_layer(
        &mut kv_decode,
        &metadata,
        ModelArchitecture::Gemma4,
        &mut scratch,
        &layer,
        None,
        0,
        1,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
    .expect("owned Gemma4 full-attn decode should succeed");

    let max_abs_diff = scratch.hidden[..4]
        .iter()
        .zip(prefill_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 5e-4, "max_abs_diff={max_abs_diff}");
}

#[test]
fn test_gemma4_owned_swa_prefill_decode_match_with_norms() {
    let metadata = ModelMetadata {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 8,
        vocab_size: 32,
        max_seq_len: 16,
        hidden_dim: 4,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        rope_dim: 8,
        rope_dim_swa: 8,
        rope_sections: [0; 4],
        norm_eps: 1e-6,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 1.0,
        sliding_window: 2,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![true],
        key_length_full: 0,
        key_length_swa: 8,
        value_length_swa: 8,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 0,
        ssm_conv_kernel: 0,
        full_attention_interval: 0,
    };

    let layer = AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&[1.0f32, 0.8, 1.1, 0.9], &[4]),
        q_weight: make_f32_weight(
            8,
            4,
            vec![
                0.5, 0.1, -0.1, 0.0, 0.0, 0.6, 0.2, 0.0, 0.1, 0.0, 0.7, 0.1, 0.0, 0.2, 0.0, 0.8,
                0.4, 0.0, 0.1, -0.1, -0.1, 0.5, 0.0, 0.2, 0.1, 0.0, 0.6, 0.0, 0.0, 0.2, -0.1, 0.7,
            ],
        ),
        k_weight: make_f32_weight(
            8,
            4,
            vec![
                0.6, 0.0, 0.0, 0.1, 0.0, 0.7, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.9,
                0.5, 0.1, 0.0, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.7, 0.1, 0.1, 0.0, 0.0, 0.8,
            ],
        ),
        v_weight: make_f32_weight(
            8,
            4,
            vec![
                0.3, 0.1, 0.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.0, 0.0, 0.6,
                0.2, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.1, 0.1, 0.0, 0.4, 0.0, 0.0, 0.1, 0.0, 0.5,
            ],
        ),
        o_weight: make_f32_weight(
            4,
            8,
            vec![
                0.8, 0.0, 0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1,
                0.1, 0.0, 0.8, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.1, 0.0, 0.2,
            ],
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: Some(Tensor::from_slice(
            &[1.0f32, 0.9, 1.1, 0.8, 1.0, 0.95, 0.85, 1.05],
            &[8],
        )),
        k_norm: Some(Tensor::from_slice(
            &[0.9f32, 1.0, 0.85, 1.1, 1.05, 0.95, 1.0, 0.9],
            &[8],
        )),
        post_attn_norm: Some(Tensor::from_slice(&[1.1f32, 0.85, 0.9, 1.05], &[4])),
        out_scale: None,
        post_ffw_norm: None,
        ffn_norm: Tensor::from_slice(&[1.0f32; 4], &[4]),
        ffn_gate_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_up_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_down_weight: make_f32_weight(4, 4, vec![0.0; 16]),
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    };

    let hidden_prefill =
        Tensor::from_slice(&[0.2f32, -0.3, 0.4, 0.1, 0.5, 0.2, -0.1, 0.3], &[2, 4]);
    let mut kv_prefill = KVCache::new_per_layer(16, &[1], &[8]);
    let out_prefill = forward_attention_layer(
        &mut kv_prefill,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden_prefill,
        &layer,
        None,
        0,
        2,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("owned Gemma4 SWA prefill should succeed");
    let out_prefill_data = kernels::tensor_as_f32_slice(&out_prefill);
    let prefill_last = &out_prefill_data[4..8];

    let mut kv_decode = KVCache::new_per_layer(16, &[1], &[8]);
    let hidden_first = Tensor::from_slice(&[0.2f32, -0.3, 0.4, 0.1], &[1, 4]);
    forward_attention_layer(
        &mut kv_decode,
        &metadata,
        ModelArchitecture::Gemma4,
        hidden_first,
        &layer,
        None,
        0,
        1,
        0,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.rope_theta,
        metadata.norm_eps,
    )
    .expect("first-token SWA prefill should populate cache");
    kv_decode.set_len(1);

    let mut scratch = ScratchBuffers::new(&metadata, 4);
    scratch.hidden[..4].copy_from_slice(&[0.5, 0.2, -0.1, 0.3]);
    decode_attention_layer(
        &mut kv_decode,
        &metadata,
        ModelArchitecture::Gemma4,
        &mut scratch,
        &layer,
        None,
        0,
        1,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "vulkan")]
        None,
    )
    .expect("owned Gemma4 SWA decode should succeed");

    let max_abs_diff = scratch.hidden[..4]
        .iter()
        .zip(prefill_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 5e-4, "max_abs_diff={max_abs_diff}");
}

#[test]
fn test_apply_model_qk_norm_uses_plain_rms_for_gemma() {
    let input = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2]);
    let weight = Tensor::from_slice(&[0.0f32, -0.5], &[2]);

    let gemma = apply_model_qk_norm(&input, &weight, 1e-5, ModelArchitecture::Gemma)
        .expect("gemma qk norm should succeed");
    let llama = apply_model_qk_norm(&input, &weight, 1e-5, ModelArchitecture::LLaMA)
        .expect("llama qk norm should succeed");

    let gemma_data = kernels::tensor_as_f32_slice(&gemma);
    let llama_data = kernels::tensor_as_f32_slice(&llama);

    assert_eq!(gemma_data, llama_data);
    assert_eq!(llama_data[0], 0.0);
    assert!(llama_data[1] < -0.63 && llama_data[1] > -0.64);
}

#[test]
fn test_engine_kv_cache_initial() {
    let engine = make_mock_engine(100);
    assert_eq!(engine.kv_cache.current_len(), 0);
}

#[test]
fn test_engine_decode_with_profiling_enabled() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let prev = std::env::var("RNB_PROFILE").ok();
    unsafe {
        std::env::set_var("RNB_PROFILE", "2");
    }

    let mut engine = make_decode_test_engine(100);
    let _ = engine.forward(&[1, 2]).expect("prefill should succeed");
    let logits = engine.forward(&[3]).expect("decode should succeed");

    match prev {
        Some(v) => unsafe {
            std::env::set_var("RNB_PROFILE", v);
        },
        None => unsafe {
            std::env::remove_var("RNB_PROFILE");
        },
    }

    assert_eq!(logits.len(), 100);
    assert_eq!(engine.kv_cache.current_len(), 3);
    assert!(engine.scratch.is_some());
}

#[test]
fn test_prefill_with_q_norm_without_gated_q_layout_succeeds() {
    let mut engine = make_decode_test_engine(32);

    if let Some(ModelWeights { layers, .. }) = engine.weights.as_mut() {
        match &mut layers[0] {
            LayerType::Attention(w) => {
                w.q_norm = Some(Tensor::from_slice(&[1.0f32; 4], &[4]));
                w.k_norm = Some(Tensor::from_slice(&[1.0f32; 4], &[4]));
            }
            LayerType::GatedDeltaNet(_)
            | LayerType::NemotronMamba2(_)
            | LayerType::NemotronMoE(_) => panic!("expected attention layer"),
        }
    }

    let logits = engine
        .forward(&[1, 2])
        .expect("prefill with q_norm but non-gated q layout should succeed");

    assert_eq!(logits.len(), 32);
    assert_eq!(engine.kv_cache.current_len(), 2);
}

#[test]
fn test_slice1_boundary_plan_for_qwen35_hybrid_prefix_attention_suffix() {
    let metadata = ModelMetadata {
        num_layers: 24,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        vocab_size: 128,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 64,
        rope_dim_swa: 64,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 2048,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 16,
        ssm_conv_kernel: 4,
        full_attention_interval: 4,
    };

    let plan = plan_slice1_boundary(&metadata).expect("slice1 plan should exist");

    assert_eq!(plan.cpu_prefix_layer_range, 0..3);
    assert_eq!(plan.window_layer_range, 3..4);
    assert_eq!(plan.attention_layer_idx, 3);
    assert_eq!(plan.window_layer_range.end, 4);
}

#[test]
fn test_slice_window_handoff_requires_hidden_cursor_and_decode_ready_cache() {
    let mut kv_cache = KVCache::new(4, 16, 2, 8);
    kv_cache.set_len(7);
    let handoff = SliceWindowHandoff {
        hidden_after_window: vec![0.25; 16],
        next_layer_idx: 4,
        next_pos: 7,
        cpu_kv_cache: kv_cache,
    };

    assert_eq!(handoff.hidden_after_window.len(), 16);
    assert_eq!(handoff.next_layer_idx, 4);
    assert_eq!(handoff.next_pos, 7);
    assert_eq!(handoff.cpu_kv_cache.current_len(), 7);
}

#[test]
fn test_gpu_prefill_executor_is_available_only_for_hybrid_slice1_models() {
    let hybrid = ModelMetadata {
        num_layers: 24,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        vocab_size: 128,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 64,
        rope_dim_swa: 64,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 2048,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 16,
        ssm_conv_kernel: 4,
        full_attention_interval: 4,
    };
    let all_attention = ModelMetadata {
        full_attention_interval: 0,
        ..hybrid.clone()
    };

    let executor = GpuPrefillExecutor::for_slice1(&hybrid)
        .expect("hybrid model should support slice1 executor");

    assert_eq!(executor.boundary_plan().cpu_prefix_layer_range, 0..3);
    assert_eq!(executor.boundary_plan().window_layer_range, 3..4);
    assert_eq!(executor.boundary_plan().attention_layer_idx, 3);
    assert!(GpuPrefillExecutor::for_slice1(&all_attention).is_none());
}

#[test]
fn test_engine_can_materialize_slice_window_handoff_from_current_kv_state() {
    let mut engine = make_decode_test_engine(64);
    let _ = engine.forward(&[1, 2, 3]).expect("prefill should succeed");

    let handoff = engine.make_slice_window_handoff(vec![0.5; 4], 4, 3);

    assert_eq!(handoff.hidden_after_window, vec![0.5; 4]);
    assert_eq!(handoff.next_layer_idx, 4);
    assert_eq!(handoff.next_pos, 3);
    assert_eq!(handoff.cpu_kv_cache.current_len(), 3);
    assert_eq!(engine.kv_cache.current_len(), 3);
}

#[test]
fn test_debug_prefill_layer_logits_matches_final_forward_logits() {
    let mut engine = make_decode_test_engine(64);
    let tokens = [1u32, 2, 3];

    let per_layer = engine
        .debug_prefill_layer_logits(&tokens)
        .expect("debug prefill logits should succeed");
    let final_logits = engine.forward(&tokens).expect("forward should succeed");

    assert_eq!(per_layer.len(), engine.metadata.num_layers);
    assert_eq!(per_layer.last().unwrap(), &final_logits);
}

#[test]
fn test_engine_only_attempts_slice1_gpu_prefill_for_multi_token_hybrid_prefill() {
    let hybrid = ModelMetadata {
        num_layers: 24,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        vocab_size: 128,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 64,
        rope_dim_swa: 64,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 2048,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 16,
        ssm_conv_kernel: 4,
        full_attention_interval: 4,
    };
    let all_attention = ModelMetadata {
        full_attention_interval: 0,
        ..hybrid.clone()
    };

    let hybrid_engine = make_mock_engine(128);
    let hybrid_engine = Engine {
        metadata: hybrid,
        ..hybrid_engine
    };
    let all_attention_engine = Engine {
        metadata: all_attention,
        ..make_mock_engine(128)
    };

    assert!(hybrid_engine.should_attempt_slice1_gpu_prefill(8));
    assert!(!hybrid_engine.should_attempt_slice1_gpu_prefill(1));
    assert!(!all_attention_engine.should_attempt_slice1_gpu_prefill(8));
}

#[test]
fn test_engine_selects_slice1_candidate_for_hybrid_multi_token_prefill() {
    let hybrid = ModelMetadata {
        num_layers: 24,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        vocab_size: 128,
        max_seq_len: 64,
        hidden_dim: 2048,
        rope_theta: 10000.0,
        rope_theta_swa: 10000.0,
        rope_dim: 64,
        rope_dim_swa: 64,
        rope_sections: [0; 4],
        norm_eps: 1e-5,
        final_logit_softcapping: 0.0,
        query_pre_attn_scalar: 256.0,
        sliding_window: 0,
        shared_kv_layers: 0,
        sliding_window_pattern: vec![],
        key_length_full: 0,
        key_length_swa: 0,
        value_length_swa: 0,
        head_count_kv_per_layer: None,
        embedding_length_per_layer_input: 0,
        expert_used_count: 0,
        expert_weights_scale: 1.0,
        ssm_d_inner: 2048,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 16,
        ssm_conv_kernel: 4,
        full_attention_interval: 4,
    };
    let all_attention = ModelMetadata {
        full_attention_interval: 0,
        ..hybrid.clone()
    };

    let hybrid_engine = Engine {
        metadata: hybrid,
        ..make_mock_engine(128)
    };
    let all_attention_engine = Engine {
        metadata: all_attention,
        ..make_mock_engine(128)
    };

    assert!(matches!(
        hybrid_engine.select_prefill_path(8),
        PrefillExecutionPath::Cpu
    ));
    assert!(matches!(
        hybrid_engine.select_prefill_path(1),
        PrefillExecutionPath::Cpu
    ));
    assert!(matches!(
        all_attention_engine.select_prefill_path(8),
        PrefillExecutionPath::Cpu
    ));
}

/// mv27 task 10b-4a: `RNB_GPU_FULLPATH=1` 이 설정돼도 mock engine 은
/// active GPU prefill path 가 없으므로 결국 `Cpu` 를 돌려줘야 한다.
/// (Fullpath variant 진입은 vulkan runtime + slice1 plan 둘 다 있어야 함.)
/// scheduler 레벨의 Fullpath dispatch 자체는 `rnb-scheduler::tests::
/// plans_slice1_prefill_boundary_from_layer_counts` 에서 직접 검증.
#[test]
fn test_engine_select_prefill_path_falls_back_to_cpu_without_active_gpu() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let prev = std::env::var("RNB_GPU_FULLPATH").ok();
    // SAFETY: env_lock guards concurrent access across engine tests.
    unsafe { std::env::set_var("RNB_GPU_FULLPATH", "1") };

    let engine = make_mock_engine(128);
    // mock engine 은 backend_runtime 활성화 안 됐고 num_layers=1 이라
    // slice1 plan 도 없음 → fullpath gate 가 켜져도 Cpu.
    assert!(matches!(
        engine.select_prefill_path(8),
        PrefillExecutionPath::Cpu
    ));

    // SAFETY: restore previous env var state under env_lock guard.
    unsafe {
        match prev {
            Some(value) => std::env::set_var("RNB_GPU_FULLPATH", value),
            None => std::env::remove_var("RNB_GPU_FULLPATH"),
        }
    }
}

#[test]
fn test_slice1_candidate_prefill_matches_cpu_prefill_on_segmented_attention_stack() {
    let mut cpu_engine = make_multi_layer_attention_engine(32, 4, 0);
    let mut segmented_engine = make_multi_layer_attention_engine(32, 4, 4);
    let tokens = [1u32, 2, 3, 4];

    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu prefill should succeed");
    let segmented_logits = segmented_engine
        .forward_prefill_slice1_candidate(&tokens)
        .expect("segmented prefill should succeed");

    assert_eq!(cpu_logits, segmented_logits);
    assert_eq!(
        cpu_engine.kv_cache.current_len(),
        segmented_engine.kv_cache.current_len()
    );
}

#[test]
fn test_forward_uses_segmented_slice1_candidate_without_changing_prefill_result() {
    let mut cpu_engine = make_multi_layer_attention_engine(32, 4, 0);
    let mut segmented_engine = make_multi_layer_attention_engine(32, 4, 4);
    let tokens = [1u32, 2, 3, 4];

    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu prefill should succeed");
    let forward_logits = segmented_engine
        .forward(&tokens)
        .expect("forward should succeed");

    assert_eq!(cpu_logits, forward_logits);
    assert_eq!(segmented_engine.kv_cache.current_len(), tokens.len());
}

#[cfg(feature = "vulkan")]
#[test]
fn test_engine_exposes_prefill_runtime_counters() {
    let engine = make_mock_engine(32);
    let counters = engine.prefill_runtime_counters();
    assert!(counters.is_none());
}

#[test]
fn test_gpu_prefill_executor_runs_slice1_window_and_returns_handoff() {
    let mut engine = make_multi_layer_attention_engine(32, 4, 4);
    let tokens = [1u32, 2, 3, 4];
    let weights = engine.weights.as_ref().expect("weights");
    let metadata = engine.metadata.clone();
    let seq_len = tokens.len();
    let pos_start = engine.kv_cache.current_len();
    let num_heads = metadata.num_heads;
    let num_kv_heads = metadata.num_kv_heads;
    let head_dim = metadata.head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let rope_theta = metadata.rope_theta;
    let norm_eps = metadata.norm_eps;

    let hidden = weights
        .token_embd
        .gather(&tokens)
        .expect("embedding gather");
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");

    let hidden = run_prefill_layers_cpu_range(
        &mut engine.kv_cache,
        &metadata,
        engine.architecture,
        weights,
        None,
        hidden,
        executor.boundary_plan().cpu_prefix_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("prefix should succeed");

    let handoff = executor
        .run_attention_window_cpu_fallback(
            &mut engine.kv_cache,
            &metadata,
            weights,
            hidden,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )
        .expect("slice1 window should succeed");

    assert_eq!(handoff.next_layer_idx, 4);
    assert_eq!(handoff.next_pos, tokens.len());
    assert_eq!(handoff.cpu_kv_cache.current_len(), tokens.len());
    assert_eq!(engine.kv_cache.current_len(), 0);
    assert_eq!(
        handoff.hidden_after_window.len(),
        tokens.len() * metadata.hidden_dim
    );
}

#[test]
fn test_gpu_prefill_executor_run_returns_decode_ready_prefill_handoff() {
    let mut engine = make_multi_layer_attention_engine(32, 4, 4);
    let tokens = [1u32, 2, 3, 4];
    let weights = engine.weights.as_ref().expect("weights");
    let metadata = engine.metadata.clone();
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");
    let kv_cache = std::mem::replace(&mut engine.kv_cache, new_empty_kv_cache(&metadata));

    #[cfg(feature = "vulkan")]
    let handoff = executor
        .run(
            kv_cache,
            &metadata,
            weights,
            &tokens,
            metadata.norm_eps,
            None,
            None,
        )
        .expect("executor run should succeed");

    #[cfg(not(feature = "vulkan"))]
    let handoff = executor
        .run(kv_cache, &metadata, weights, &tokens, metadata.norm_eps)
        .expect("executor run should succeed");

    assert_eq!(handoff.logits.len(), metadata.vocab_size);
    assert_eq!(handoff.next_pos, tokens.len());
    assert_eq!(handoff.cpu_kv_cache.current_len(), tokens.len());
    assert_eq!(engine.kv_cache.current_len(), 0);
}

#[test]
fn test_prefill_handoff_preserves_next_decode_logits() {
    let mut baseline = make_multi_layer_attention_engine(32, 4, 4);
    let mut handoff_engine = make_multi_layer_attention_engine(32, 4, 4);
    let prompt = [1u32, 2, 3, 4];
    let next = 5u32;

    let _ = baseline
        .forward(&prompt)
        .expect("baseline prefill should succeed");
    let baseline_logits = baseline
        .forward_decode(next)
        .expect("baseline decode should succeed");

    let metadata = handoff_engine.metadata.clone();
    let weights = handoff_engine.weights.as_ref().expect("weights");
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");
    let kv_cache = std::mem::replace(&mut handoff_engine.kv_cache, new_empty_kv_cache(&metadata));

    #[cfg(feature = "vulkan")]
    let handoff = executor
        .run(
            kv_cache,
            &metadata,
            weights,
            &prompt,
            metadata.norm_eps,
            None,
            None,
        )
        .expect("handoff run should succeed");

    #[cfg(not(feature = "vulkan"))]
    let handoff = executor
        .run(kv_cache, &metadata, weights, &prompt, metadata.norm_eps)
        .expect("handoff run should succeed");

    handoff_engine.kv_cache = handoff.cpu_kv_cache;
    let handoff_logits = handoff_engine
        .forward_decode(next)
        .expect("handoff decode should succeed");

    let max_abs_diff = baseline_logits
        .iter()
        .zip(handoff_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
}

#[test]
fn test_slice_window_handoff_preserves_suffix_reentry_hidden() {
    let mut cpu_engine = make_multi_layer_attention_engine(32, 4, 4);
    let mut handoff_engine = make_multi_layer_attention_engine(32, 4, 4);
    let tokens = [1u32, 2, 3, 4];
    let metadata = cpu_engine.metadata.clone();
    let weights = cpu_engine.weights.as_ref().expect("weights");
    let seq_len = tokens.len();
    let pos_start = 0usize;
    let num_heads = metadata.num_heads;
    let num_kv_heads = metadata.num_kv_heads;
    let head_dim = metadata.head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let rope_theta = metadata.rope_theta;
    let norm_eps = metadata.norm_eps;
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");

    let hidden0 = apply_embedding_scale(
        weights.token_embd.gather(&tokens).expect("embed gather"),
        &metadata,
        cpu_engine.architecture,
    );
    let cpu_after_prefix = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        hidden0,
        executor.boundary_plan().cpu_prefix_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu prefix should succeed");
    let cpu_after_window = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        cpu_after_prefix,
        executor.boundary_plan().window_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu window should succeed");
    let cpu_after_suffix = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        cpu_after_window,
        executor.boundary_plan().window_layer_range.end..metadata.num_layers,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu suffix should succeed");

    let hidden1 = handoff_engine
        .weights
        .as_ref()
        .expect("weights")
        .token_embd
        .gather(&tokens)
        .expect("embed gather");
    let handoff_prefix = run_prefill_layers_cpu_range(
        &mut handoff_engine.kv_cache,
        &metadata,
        handoff_engine.architecture,
        handoff_engine.weights.as_ref().expect("weights"),
        None,
        hidden1,
        executor.boundary_plan().cpu_prefix_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("handoff prefix should succeed");
    let handoff = executor
        .run_attention_window_cpu_fallback(
            &mut handoff_engine.kv_cache,
            &metadata,
            handoff_engine.weights.as_ref().expect("weights"),
            handoff_prefix,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )
        .expect("window handoff should succeed");
    handoff_engine.kv_cache = handoff.cpu_kv_cache;
    let suffix_hidden =
        Tensor::from_vec(handoff.hidden_after_window, &[seq_len, metadata.hidden_dim]);
    let handoff_after_suffix = run_prefill_layers_cpu_range(
        &mut handoff_engine.kv_cache,
        &metadata,
        handoff_engine.architecture,
        handoff_engine.weights.as_ref().expect("weights"),
        None,
        suffix_hidden,
        handoff.next_layer_idx..metadata.num_layers,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("handoff suffix should succeed");

    let cpu_data = kernels::tensor_as_f32_slice(&cpu_after_suffix);
    let handoff_data = kernels::tensor_as_f32_slice(&handoff_after_suffix);
    let max_abs_diff = cpu_data
        .iter()
        .zip(handoff_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_prefill_handoff_preserves_next_decode_logits_with_real_gpu() {
    let mut baseline = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let mut handoff_engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = handoff_engine.metadata.hidden_dim;
    let ffn_inner_dim = handoff_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = handoff_engine.metadata.vocab_size.max(ffn_inner_dim);
    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for handoff test");
    handoff_engine.backend_runtime =
        backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let prompt = [1u32, 2, 3, 4];
    let next = 5u32;

    let _ = baseline
        .forward(&prompt)
        .expect("baseline prefill should succeed");
    let baseline_logits = baseline
        .forward_decode(next)
        .expect("baseline decode should succeed");

    let metadata = handoff_engine.metadata.clone();
    let weights = handoff_engine.weights.as_ref().expect("weights");
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");
    let mut scratch = handoff_engine.scratch.take();
    let mut gpu_runtime = handoff_engine.backend_runtime.take_gpu_runtime();
    let kv_cache = std::mem::replace(&mut handoff_engine.kv_cache, new_empty_kv_cache(&metadata));
    let handoff = executor
        .run(
            kv_cache,
            &metadata,
            weights,
            &prompt,
            metadata.norm_eps,
            scratch.as_mut(),
            gpu_runtime.as_mut(),
        )
        .expect("handoff run should succeed");
    handoff_engine.scratch = scratch;
    handoff_engine
        .backend_runtime
        .restore_gpu_runtime(gpu_runtime);

    handoff_engine.kv_cache = handoff.cpu_kv_cache;
    handoff_engine.backend_runtime = backend_runtime::EngineBackendRuntime::new();
    let handoff_logits = handoff_engine
        .forward_decode(next)
        .expect("handoff decode should succeed");

    let max_abs_diff = baseline_logits
        .iter()
        .zip(handoff_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
    let baseline_argmax = baseline_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let handoff_argmax = handoff_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(baseline_argmax, handoff_argmax);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader; quantized GPU path is validated by logits parity, not hidden exactness"]
fn test_slice_window_handoff_preserves_suffix_reentry_hidden_with_real_gpu() {
    let mut cpu_engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let mut handoff_engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = handoff_engine.metadata.hidden_dim;
    let ffn_inner_dim = handoff_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = handoff_engine.metadata.vocab_size.max(ffn_inner_dim);
    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for suffix reentry test");
    handoff_engine.backend_runtime =
        backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let tokens = [1u32, 2, 3, 4];
    let metadata = cpu_engine.metadata.clone();
    let weights = cpu_engine.weights.as_ref().expect("weights");
    let seq_len = tokens.len();
    let pos_start = 0usize;
    let num_heads = metadata.num_heads;
    let num_kv_heads = metadata.num_kv_heads;
    let head_dim = metadata.head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let rope_theta = metadata.rope_theta;
    let norm_eps = metadata.norm_eps;
    let executor = GpuPrefillExecutor::for_slice1(&metadata).expect("slice1 executor");

    let hidden0 = apply_embedding_scale(
        weights.token_embd.gather(&tokens).expect("embed gather"),
        &metadata,
        cpu_engine.architecture,
    );
    let cpu_after_prefix = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        hidden0,
        executor.boundary_plan().cpu_prefix_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu prefix should succeed");
    let cpu_after_window = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        cpu_after_prefix,
        executor.boundary_plan().window_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu window should succeed");
    let cpu_after_suffix = run_prefill_layers_cpu_range(
        &mut cpu_engine.kv_cache,
        &metadata,
        cpu_engine.architecture,
        weights,
        None,
        cpu_after_window,
        executor.boundary_plan().window_layer_range.end..metadata.num_layers,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("cpu suffix should succeed");

    let hidden1 = handoff_engine
        .weights
        .as_ref()
        .expect("weights")
        .token_embd
        .gather(&tokens)
        .expect("embed gather");
    let handoff_prefix = run_prefill_layers_cpu_range(
        &mut handoff_engine.kv_cache,
        &metadata,
        handoff_engine.architecture,
        handoff_engine.weights.as_ref().expect("weights"),
        None,
        hidden1,
        executor.boundary_plan().cpu_prefix_layer_range.clone(),
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("handoff prefix should succeed");
    let mut scratch = handoff_engine.scratch.take();
    let mut gpu_runtime = handoff_engine.backend_runtime.take_gpu_runtime();
    let handoff = executor
        .run_attention_window_gpu_tokenwise(
            &mut handoff_engine.kv_cache,
            &metadata,
            scratch.as_mut().expect("scratch"),
            match &handoff_engine.weights.as_ref().expect("weights").layers
                [executor.boundary_plan().attention_layer_idx]
            {
                LayerType::Attention(w) => w,
                LayerType::GatedDeltaNet(_) => panic!("expected attention layer"),
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => {
                    panic!("expected attention layer")
                }
            },
            handoff_prefix,
            seq_len,
            pos_start,
            gpu_runtime.as_mut(),
            None,
        )
        .expect("window handoff should succeed");
    handoff_engine.scratch = scratch;
    handoff_engine
        .backend_runtime
        .restore_gpu_runtime(gpu_runtime);

    let suffix_hidden =
        Tensor::from_vec(handoff.hidden_after_window, &[seq_len, metadata.hidden_dim]);
    let handoff_after_suffix = run_prefill_layers_cpu_range(
        &mut handoff_engine.kv_cache,
        &metadata,
        handoff_engine.architecture,
        handoff_engine.weights.as_ref().expect("weights"),
        None,
        suffix_hidden,
        handoff.next_layer_idx..metadata.num_layers,
        seq_len,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
    )
    .expect("handoff suffix should succeed");

    let cpu_data = kernels::tensor_as_f32_slice(&cpu_after_suffix);
    let handoff_data = kernels::tensor_as_f32_slice(&handoff_after_suffix);
    let max_abs_diff = cpu_data
        .iter()
        .zip(handoff_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_slice1_candidate_with_real_gpu_matches_cpu_baseline() {
    let mut cpu_engine = make_real_gpu_hybrid_slice1_engine(32, false);
    let mut gpu_engine = make_real_gpu_hybrid_slice1_engine(32, false);
    let hidden_dim = gpu_engine.metadata.hidden_dim;
    let ffn_inner_dim = gpu_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = gpu_engine.metadata.vocab_size.max(ffn_inner_dim);

    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for hybrid slice1 test");
    gpu_engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let tokens = [1u32, 2, 3, 4];
    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu hybrid prefill should succeed");
    let gpu_logits = gpu_engine
        .forward(&tokens)
        .expect("gpu hybrid prefill should succeed");

    let max_abs_diff = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(cpu_argmax, gpu_argmax);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_with_suffix_slice1_candidate_with_real_gpu_matches_cpu_baseline() {
    let mut cpu_engine = make_real_gpu_hybrid_slice1_engine(32, true);
    let mut gpu_engine = make_real_gpu_hybrid_slice1_engine(32, true);
    let hidden_dim = gpu_engine.metadata.hidden_dim;
    let ffn_inner_dim = gpu_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = gpu_engine.metadata.vocab_size.max(ffn_inner_dim);

    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for hybrid suffix test");
    gpu_engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let tokens = [1u32, 2, 3, 4];
    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu hybrid suffix prefill should succeed");
    let gpu_logits = gpu_engine
        .forward(&tokens)
        .expect("gpu hybrid suffix prefill should succeed");

    let max_abs_diff = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(cpu_argmax, gpu_argmax);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_with_real_gpu_matches_cpu_baseline() {
    let mut cpu_engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let mut gpu_engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = gpu_engine.metadata.hidden_dim;
    let ffn_inner_dim = gpu_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = gpu_engine.metadata.vocab_size.max(ffn_inner_dim);

    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for hybrid gqa test");
    gpu_engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let tokens = [1u32, 2, 3, 4];
    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu hybrid gqa prefill should succeed");
    let gpu_logits = gpu_engine
        .forward(&tokens)
        .expect("gpu hybrid gqa prefill should succeed");

    let max_abs_diff = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-3, "max_abs_diff={max_abs_diff}");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(cpu_argmax, gpu_argmax);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_exposes_gpu_runtime_counters() {
    let mut engine = make_real_gpu_hybrid_slice1_engine(32, false);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for runtime counter test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");
    assert!(counters.submits > 0, "expected submits > 0");
    assert!(counters.upload_bytes > 0, "expected upload_bytes > 0");
    assert!(counters.download_bytes > 0, "expected download_bytes > 0");
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_records_materialization_count() {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for materialization counter test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");
    assert!(
        counters.materializations > 0,
        "expected materializations > 0"
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_multichunk_slice1_candidate_materializes_once_per_prompt() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let prev = std::env::var("RNB_PREFILL_CHUNK_SIZE").ok();
    unsafe {
        std::env::set_var("RNB_PREFILL_CHUNK_SIZE", "2");
    }

    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for multichunk materialization test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu multichunk prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");

    match prev {
        Some(v) => unsafe {
            std::env::set_var("RNB_PREFILL_CHUNK_SIZE", v);
        },
        None => unsafe {
            std::env::remove_var("RNB_PREFILL_CHUNK_SIZE");
        },
    }

    assert_eq!(
        counters.materializations, 1,
        "expected one final CPU KV materialization per prompt, got {}",
        counters.materializations
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_quantized_gdn_slice1_candidate_batches_conv_state_materialization_at_handoff() {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_quantized_gdn_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let conv_channels =
        engine.metadata.ssm_d_inner + 2 * engine.metadata.ssm_n_group * engine.metadata.ssm_d_state;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim * 4, ffn_inner_dim);
    let max_output = std::cmp::max(
        engine.metadata.vocab_size.max(ffn_inner_dim),
        conv_channels * 4,
    );

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for quantized GDN materialization test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill with quantized GDN should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");

    assert_eq!(
        counters.materializations, 1,
        "expected one batched handoff materialization for attention KV + GDN conv_state, got {}",
        counters.materializations
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_eliminates_attention_fan_in_copies() {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for attention fan-in copy test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");

    assert_eq!(
        counters.attention_fan_in_copies, 0,
        "expected attention helper to eliminate per-token fan-in copies, got {}",
        counters.attention_fan_in_copies
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_submit_count_beats_tokenwise_upper_bound() {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for submit upper bound test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let seq_len = 4u64;
    let num_heads = engine.metadata.num_heads as u64;
    let naive_upper_bound = seq_len * (1 + 1 + num_heads + 1 + 1);

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");
    assert!(
        counters.submits < naive_upper_bound,
        "expected submits {} < naive upper bound {}",
        counters.submits,
        naive_upper_bound
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_hybrid_gqa_slice1_candidate_submit_count_drops_below_window_stage_count() {
    let mut engine = make_real_gpu_hybrid_slice1_gqa_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for exact submit count test");
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("gpu prefill should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");

    assert!(
        counters.submits < 10,
        "expected submits {} to drop below current 10-submit baseline",
        counters.submits
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn test_f32_hybrid_helper_falls_back_without_gpu_fast_path() {
    let mut engine = make_hybrid_slice1_engine_with_suffix(32);
    let hidden_dim = engine.metadata.hidden_dim;
    let ffn_inner_dim = engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = engine.metadata.vocab_size.max(ffn_inner_dim);

    let mut vk = match backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let _ = engine
        .forward(&[1u32, 2, 3, 4])
        .expect("forward should succeed");
    let counters = engine
        .prefill_runtime_counters()
        .expect("gpu counters should exist");
    assert_eq!(counters.submits, 0);
    assert_eq!(counters.upload_bytes, 0);
    assert_eq!(counters.download_bytes, 0);
}

#[cfg(feature = "vulkan")]
#[test]
fn test_ggml_to_gpu_quant_supports_q5_k() {
    let _guard = env_lock().lock().unwrap();
    std::env::set_var("RNB_GPU_Q5K", "1");
    assert_eq!(
        backend_runtime::ggml_to_quant_for_test(GGMLType::Q5_K),
        Some(backend_runtime::GpuQuant::Q5K)
    );
    std::env::remove_var("RNB_GPU_Q5K");
}

#[cfg(feature = "vulkan")]
#[test]
fn test_ggml_to_gpu_output_quant_rejects_q5_k() {
    assert_eq!(ggml_to_gpu_output_quant(GGMLType::Q5_K), None);
    assert_eq!(
        ggml_to_gpu_output_quant(GGMLType::Q4_K),
        Some(backend_runtime::GpuQuant::Q4K)
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn test_ggml_to_gpu_quant_rejects_q5_k_without_opt_in() {
    let _guard = env_lock().lock().unwrap();
    std::env::remove_var("RNB_GPU_Q5K");
    assert_eq!(
        backend_runtime::ggml_to_quant_for_test(GGMLType::Q5_K),
        None
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires runtime GPU loader"]
fn test_forward_segmented_slice1_candidate_with_real_gpu_matches_cpu_baseline() {
    let mut cpu_engine = make_real_gpu_multi_layer_attention_engine(32, 4, 0);
    let mut gpu_engine = make_real_gpu_multi_layer_attention_engine(32, 4, 4);
    let hidden_dim = gpu_engine.metadata.hidden_dim;
    let ffn_inner_dim = gpu_engine
        .weights
        .as_ref()
        .and_then(|w| w.layers.first())
        .map(|l| match l {
            LayerType::Attention(w) => w.ffn_gate_weight.rows,
            LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
            LayerType::NemotronMamba2(w) => w.ssm_out.rows,
            LayerType::NemotronMoE(w) => w.expert_up.rows,
        })
        .unwrap_or(hidden_dim);
    let max_input = std::cmp::max(hidden_dim, ffn_inner_dim);
    let max_output = gpu_engine.metadata.vocab_size.max(ffn_inner_dim);

    let vk = backend_runtime::init_layer_gemv_for_test(max_input, max_output, 64)
        .expect("requires runtime GPU loader for segmented slice1 test");
    gpu_engine.backend_runtime = backend_runtime::EngineBackendRuntime::from_gpu_runtime(Some(vk));

    let tokens = [1u32, 2, 3, 4];
    let cpu_logits = cpu_engine
        .forward_prefill_cpu(&tokens)
        .expect("cpu prefill should succeed");
    let gpu_logits = gpu_engine
        .forward(&tokens)
        .expect("gpu forward should succeed");

    assert_eq!(cpu_logits.len(), gpu_logits.len());
    let max_abs_diff = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 1e-4, "max_abs_diff={max_abs_diff}");
}

#[test]
fn test_dequantize_row_to_slice_q6k_matches_vec_path() {
    let block = BlockQ6_K {
        ql: [0x21; 128],
        qh: [0x54; 64],
        scales: [1, -2, 3, -4, 5, -6, 7, -8, 2, -3, 4, -5, 6, -7, 8, -1],
        d: f16::from_f32(0.125),
    };
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&block as *const BlockQ6_K).cast::<u8>(),
            std::mem::size_of::<BlockQ6_K>(),
        )
    };
    let expected = dequantize_bytes_to_f32(bytes, GGMLType::Q6_K);
    let mut actual = vec![0.0f32; 256];

    let ok = dequantize_row_to_slice_if_supported(bytes, GGMLType::Q6_K, &mut actual);

    assert!(ok);
    assert_eq!(actual, expected);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemv_into_q8_q8_0_matches_f32_path() {
    let values: Vec<i8> = (0..64).map(|i| (i as i8 % 7) - 3).collect();
    let weight = make_q8_0_weight(2, 32, &values);
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
    let mut q8 = [QuantizedQ8Block::default(); 1];
    quantize_q8_for_test(&input, &mut q8);

    let mut out_q8 = vec![0.0f32; 2];
    weight
        .gemv_into_q8(&q8, &mut out_q8)
        .expect("q8 path should succeed");

    let mut out_f32 = vec![0.0f32; 2];
    weight
        .gemv_into(&input, &mut out_f32)
        .expect("f32 path should succeed");

    for (a, b) in out_q8.iter().zip(out_f32.iter()) {
        assert!((a - b).abs() < 1e-4, "q8={a}, f32={b}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemv_into_q8k_uses_packed_q4k_data() {
    let _guard = env_lock().lock().unwrap();
    let old_packed_decode = std::env::var("RNB_PACKED_DECODE").ok();
    std::env::set_var("RNB_PACKED_DECODE", "1");

    let rows = 16;
    let cols = 2 * 256;
    let blocks_per_row = cols / 256;

    let make_raw = |offset: u16| {
        let mut raw = Vec::new();
        for row in 0..rows {
            for bi in 0..blocks_per_row {
                let seed = offset + (row * blocks_per_row + bi) as u16;
                let mut scales = [0u8; 12];
                for (i, scale) in scales.iter_mut().enumerate() {
                    *scale = ((seed + i as u16 * 7 + 11) % 256) as u8;
                }
                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
                }
                raw.extend_from_slice(&make_q4k_block(
                    0.01 + seed as f32 * 0.001,
                    0.005,
                    scales,
                    qs,
                ));
            }
        }
        raw
    };

    let base_raw = make_raw(0);
    let packed_raw = make_raw(97);
    let packed = pack_q4k_for_test(&packed_raw, rows, blocks_per_row);

    let base_weight = make_quant_weight(GGMLType::Q4_K, rows, cols, base_raw.clone());
    let expected_weight = make_quant_weight(GGMLType::Q4_K, rows, cols, packed_raw);
    let mut packed_weight = make_quant_weight(GGMLType::Q4_K, rows, cols, base_raw);
    packed_weight.packed_gemm_data = Some((packed.as_ptr(), packed.len()));

    let input: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.029).cos() * 0.2)
        .collect();
    let q8k = quantize_q8k_for_test(&input);

    let mut got = vec![0.0f32; rows];
    packed_weight.gemv_into_q8k(&q8k, &mut got).unwrap();
    let expected = expected_weight.gemv_vec_q8k(&q8k).unwrap();
    let base = base_weight.gemv_vec_q8k(&q8k).unwrap();

    let max_expected_diff = got
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_expected_diff < 1e-4,
        "packed into path did not match packed data: max_expected_diff={max_expected_diff}"
    );

    let max_base_diff = got
        .iter()
        .zip(base.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_base_diff > 1e-3,
        "test fixture failed to distinguish raw and packed paths"
    );

    match old_packed_decode {
        Some(v) => std::env::set_var("RNB_PACKED_DECODE", v),
        None => std::env::remove_var("RNB_PACKED_DECODE"),
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemv_vec_q8k_q4k_packed_matches_plain_path() {
    let rows = 16;
    let cols = 2 * 256;
    let blocks_per_row = cols / 256;

    let mut raw = Vec::new();
    for row in 0..rows {
        for bi in 0..blocks_per_row {
            let seed = (row * blocks_per_row + bi) as u16;
            let mut scales = [0u8; 12];
            for (i, scale) in scales.iter_mut().enumerate() {
                *scale = ((seed + i as u16 * 7 + 11) % 256) as u8;
            }
            let mut qs = [0u8; 128];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
            }
            raw.extend_from_slice(&make_q4k_block(
                0.01 + seed as f32 * 0.001,
                0.005,
                scales,
                qs,
            ));
        }
    }

    let packed = pack_q4k_for_test(&raw, rows, blocks_per_row);
    let plain = make_quant_weight(GGMLType::Q4_K, rows, cols, raw.clone());
    let mut packed_weight = make_quant_weight(GGMLType::Q4_K, rows, cols, raw);
    packed_weight.packed_gemm_data = Some((packed.as_ptr(), packed.len()));

    let input: Vec<f32> = (0..(cols * 17))
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.029).cos() * 0.2)
        .collect();
    let q8k = quantize_q8k_for_test(&input);

    let got_plain = plain.gemv_vec_q8k(&q8k).unwrap();
    let got_packed = packed_weight.gemv_vec_q8k(&q8k).unwrap();

    assert_eq!(got_plain.len(), got_packed.len());
    for (idx, (a, b)) in got_plain.iter().zip(got_packed.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "idx={idx} plain={a} packed={b}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemv_vec_q8k_q5k_packed_matches_plain_path() {
    let rows = 16;
    let cols = 2 * 256;
    let blocks_per_row = cols / 256;

    let mut raw = Vec::new();
    for row in 0..rows {
        for bi in 0..blocks_per_row {
            let seed = (row * blocks_per_row + bi) as u16;
            let mut scales = [0u8; 12];
            for (i, scale) in scales.iter_mut().enumerate() {
                *scale = ((seed + i as u16 * 7 + 11) % 256) as u8;
            }
            let mut qh = [0u8; 32];
            for (i, q) in qh.iter_mut().enumerate() {
                *q = ((seed * 3 + i as u16 * 5 + 17) % 256) as u8;
            }
            let mut qs = [0u8; 128];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
            }
            raw.extend_from_slice(&make_q5k_block(
                0.01 + seed as f32 * 0.001,
                0.005,
                scales,
                qh,
                qs,
            ));
        }
    }

    let packed = pack_q5k_for_test(&raw, rows, blocks_per_row);
    let plain = make_quant_weight(GGMLType::Q5_K, rows, cols, raw.clone());
    let mut packed_weight = make_quant_weight(GGMLType::Q5_K, rows, cols, raw);
    packed_weight.packed_gemm_data = Some((packed.as_ptr(), packed.len()));

    let input: Vec<f32> = (0..(cols * 17))
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.029).cos() * 0.2)
        .collect();
    let q8k = quantize_q8k_for_test(&input);

    let got_plain = plain.gemv_vec_q8k(&q8k).unwrap();
    let got_packed = packed_weight.gemv_vec_q8k(&q8k).unwrap();

    assert_eq!(got_plain.len(), got_packed.len());
    for (idx, (a, b)) in got_plain.iter().zip(got_packed.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "idx={idx} plain={a} packed={b}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemv_vec_q8k_q6k_packed_matches_plain_path() {
    let rows = 16;
    let cols = 2 * 256;
    let blocks_per_row = cols / 256;

    let mut raw = Vec::new();
    for row in 0..rows {
        for bi in 0..blocks_per_row {
            let seed = (row * blocks_per_row + bi) as u16;
            let mut scales = [0i8; 16];
            for (i, scale) in scales.iter_mut().enumerate() {
                *scale = ((((seed as i32) * 11 + i as i32 * 7) % 63) - 31) as i8;
            }
            let mut ql = [0u8; 128];
            for (i, q) in ql.iter_mut().enumerate() {
                *q = ((seed * 5 + i as u16 * 9 + 3) % 256) as u8;
            }
            let mut qh = [0u8; 64];
            for (i, q) in qh.iter_mut().enumerate() {
                *q = ((seed * 13 + i as u16 * 3 + 17) % 256) as u8;
            }
            raw.extend_from_slice(&make_q6k_block(0.01 + seed as f32 * 0.001, scales, ql, qh));
        }
    }

    let packed = pack_q6k_for_test(&raw, rows, blocks_per_row);
    let plain = make_quant_weight(GGMLType::Q6_K, rows, cols, raw.clone());
    let mut packed_weight = make_quant_weight(GGMLType::Q6_K, rows, cols, raw);
    packed_weight.packed_gemm_data = Some((packed.as_ptr(), packed.len()));

    let input: Vec<f32> = (0..(cols * 17))
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.029).cos() * 0.2)
        .collect();
    let q8k = quantize_q8k_for_test(&input);

    let got_plain = plain.gemv_vec_q8k(&q8k).unwrap();
    let got_packed = packed_weight.gemv_vec_q8k(&q8k).unwrap();

    assert_eq!(got_plain.len(), got_packed.len());
    for (idx, (a, b)) in got_plain.iter().zip(got_packed.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "idx={idx} plain={a} packed={b}");
    }
}

#[test]
fn test_packed_rnb_default_big_affinity_for_standalone_rnb() {
    let tmp = tempfile::TempDir::new().unwrap();
    let rnb_path = tmp.path().join("model.rnb");

    assert!(packed_rnb_default_big_affinity(
        &rnb_path, false, false, false
    ));
}

#[test]
fn test_packed_rnb_default_big_affinity_for_sidecar_rnb() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    std::fs::write(&rnb_path, b"RNBD").unwrap();

    assert!(packed_rnb_default_big_affinity(
        &gguf_path, false, false, false
    ));
}

#[test]
fn test_packed_rnb_default_big_affinity_respects_overrides() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    std::fs::write(&rnb_path, b"RNBD").unwrap();

    assert!(!packed_rnb_default_big_affinity(
        &gguf_path, true, false, false
    ));
    assert!(!packed_rnb_default_big_affinity(
        &gguf_path, false, true, false
    ));
    assert!(!packed_rnb_default_big_affinity(
        &gguf_path, false, false, true
    ));
}

#[test]
fn test_gemv_q8k_profile_method_distinguishes_packed_variants() {
    assert_eq!(
        gemv_q8k_profile_method(Some(super::packed_runtime::QuantType::Q4KCompact)),
        "gemv_vec_q8k_compact"
    );
    assert_eq!(
        gemv_q8k_profile_method(Some(super::packed_runtime::QuantType::Q4K)),
        "gemv_vec_q8k_packed"
    );
    assert_eq!(gemv_q8k_profile_method(None), "gemv_vec_q8k");
}

#[test]
fn test_runtime_rawmeta_repack_disabled_after_format_removal() {
    // Q4KRawMeta variant has been removed (standalone .rnb format deprecated).
    // The runtime-rawmeta-repack policy is now permanently disabled.
    let _lock = env_lock().lock().unwrap();
    let prev = std::env::var("RNB_RAWMETA_REPACK_CACHE").ok();
    std::env::remove_var("RNB_RAWMETA_REPACK_CACHE");

    assert!(!runtime_rawmeta_repack_enabled(
        Some(super::packed_runtime::QuantType::Q4K),
        rnb_loader::GGMLType::Q4_K,
        443,
        12_288,
        1_536,
    ));
    assert!(!runtime_rawmeta_repack_enabled(
        None,
        rnb_loader::GGMLType::Q4_K,
        443,
        12_288,
        1_536,
    ));

    match prev {
        Some(v) => std::env::set_var("RNB_RAWMETA_REPACK_CACHE", v),
        None => std::env::remove_var("RNB_RAWMETA_REPACK_CACHE"),
    }
}

#[test]
fn test_moe_decode_sidecar_requires_explicit_opt_in() {
    let _lock = env_lock().lock().unwrap();
    let prev = std::env::var("RNB_MOE_DECODE").ok();
    std::env::remove_var("RNB_MOE_DECODE");
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    std::fs::write(&gguf_path, b"").unwrap();
    std::fs::write(&rnb_path, b"RNBMxxxxxxxx").unwrap();
    assert!(!moe_section_decode_sidecar_requested(&gguf_path));
    std::env::set_var("RNB_MOE_DECODE", "1");
    assert!(moe_section_decode_sidecar_requested(&gguf_path));
    match prev {
        Some(v) => std::env::set_var("RNB_MOE_DECODE", v),
        None => std::env::remove_var("RNB_MOE_DECODE"),
    }
}

#[test]
fn test_moe_section_decode_sidecar_requested_respects_kill_switch() {
    let _lock = env_lock().lock().unwrap();
    let prev = std::env::var("RNB_MOE_DECODE").ok();
    std::env::set_var("RNB_MOE_DECODE", "0");

    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    std::fs::write(&gguf_path, b"").unwrap();
    std::fs::write(&rnb_path, b"RNBMxxxxxxxx").unwrap();

    assert!(!moe_section_decode_sidecar_requested(&gguf_path));

    match prev {
        Some(v) => std::env::set_var("RNB_MOE_DECODE", v),
        None => std::env::remove_var("RNB_MOE_DECODE"),
    }
}

#[test]
fn test_q4k_kernel_backend_defaults_to_none() {
    assert_eq!(q4k_kernel_backend_from_env(None), None);
}

#[test]
fn test_q4k_kernel_backend_parses_builtin_only() {
    assert_eq!(
        q4k_kernel_backend_from_env(Some("builtin")),
        Some(Q4KKernelBackend::Builtin)
    );
    assert_eq!(q4k_kernel_backend_from_env(Some("external")), None);
    assert_eq!(q4k_kernel_backend_from_env(Some("external-prefill")), None);
}

#[test]
fn test_q4k_kernel_backend_rejects_unknown_values() {
    assert_eq!(q4k_kernel_backend_from_env(Some("weird")), None);
}

// ---------------------------------------------------------------------------
// Session 79 Phase 1 Task 12: `.rnb` MoE section MOE_DECODE engine wiring tests.
// ---------------------------------------------------------------------------

/// Build the same synthetic `MOE_DECODE_SECTION` body as
/// `rnb_loader::rnb_moe_reader::tests::build_synthetic_body_qwen_min`,
/// duplicated here because that helper is private to the reader's test
/// module. Shape: 1 layer, 1 expert, n_embd=256, d_ff=256, no shared expert.
fn build_synthetic_moe_decode_body_min() -> Vec<u8> {
    use rnb_loader::rnb_moe_reader::{GU_PAIR_Q4K_BYTES, Q5K_INTSCALE_BYTES};

    let mut out = Vec::<u8>::new();
    // per_layer_count = 1
    out.extend_from_slice(&1u32.to_le_bytes());
    // layer header
    out.extend_from_slice(&1u32.to_le_bytes()); // n_experts
    out.extend_from_slice(&256u32.to_le_bytes()); // d_ff
    out.extend_from_slice(&256u32.to_le_bytes()); // n_embd
    out.push(0x12); // gate_up_quant (Q4_K in moe_section tag space)
    out.push(0x14); // down_quant   (Q5_K)
    out.push(0xFF); // shared_quant NONE
    while out.len() % 16 != 0 {
        out.push(0);
    }
    // Expert 0: d_ff=256 gate_up rows, each = 8B muls + 288B GUPairQ4K + 64B-aligned pad.
    for r in 0..256u32 {
        let gate_mul = (r as f32) + 0.125;
        let up_mul = (r as f32) + 0.250;
        out.extend_from_slice(&gate_mul.to_le_bytes());
        out.extend_from_slice(&up_mul.to_le_bytes());
        let pat = (r & 0xFF) as u8;
        for _ in 0..GU_PAIR_Q4K_BYTES {
            out.push(pat);
        }
        while out.len() % 64 != 0 {
            out.push(0);
        }
    }
    // Expert 0: n_embd=256 down rows, each = 4B down_mul + 176B Q5KIntScale + 64B-aligned pad.
    for r in 0..256u32 {
        let down_mul = (r as f32) * 2.0 + 1.0;
        out.extend_from_slice(&down_mul.to_le_bytes());
        let pat = ((r ^ 0xA5) & 0xFF) as u8;
        for _ in 0..Q5K_INTSCALE_BYTES {
            out.push(pat);
        }
        while out.len() % 64 != 0 {
            out.push(0);
        }
    }
    out
}

/// Build a complete MoE section `.rnb` file (header + section table + padded body)
/// around the given MOE_DECODE body.
fn build_moe_section_rnb_file(moe_body: &[u8]) -> Vec<u8> {
    use rnb_core::rnb_moe::{SectionId, MOE_HEADER_FIXED_LEN, MOE_MAGIC, MOE_SECTION_ENTRY_LEN};

    let header_size = MOE_HEADER_FIXED_LEN + MOE_SECTION_ENTRY_LEN;
    let body_start = (header_size + 15) & !15;

    let mut file = Vec::new();
    file.extend_from_slice(&MOE_MAGIC);
    file.extend_from_slice(&2u32.to_le_bytes());
    file.extend_from_slice(&1u32.to_le_bytes()); // section_count = 1
    file.push(SectionId::MoeDecode as u8);
    file.extend_from_slice(&(body_start as u64).to_le_bytes());
    file.extend_from_slice(&(moe_body.len() as u64).to_le_bytes());
    while file.len() < body_start {
        file.push(0);
    }
    file.extend_from_slice(moe_body);
    file
}

#[test]
fn offset_of_subslice_computes_byte_delta() {
    let parent: Vec<u8> = (0u8..32).collect();
    let sub = &parent[5..13];
    assert_eq!(super::offset_of_subslice(sub, &parent), 5);
    let sub2 = &parent[0..4];
    assert_eq!(super::offset_of_subslice(sub2, &parent), 0);
}

#[test]
#[should_panic(expected = "subslice pointer must be inside parent buffer")]
fn offset_of_subslice_panics_for_foreign_slice() {
    let parent: Vec<u8> = (0u8..32).collect();
    let other: Vec<u8> = (0u8..8).collect();
    let _ = super::offset_of_subslice(&other[..], &parent);
}

#[test]
fn convert_moe_section_decode_layer_preserves_rows_and_offsets() {
    use rnb_loader::rnb_moe_reader::{GU_PAIR_Q4K_BYTES, Q5K_INTSCALE_BYTES};
    use std::io::Write;

    let body = build_synthetic_moe_decode_body_min();
    let file = build_moe_section_rnb_file(&body);
    let mut tmp_file = tempfile::NamedTempFile::new().expect("tmpfile for MoE section bytes");
    tmp_file.write_all(&file).expect("write tmpfile bytes");
    tmp_file.flush().expect("flush tmpfile bytes");
    let bytes_arc = std::sync::Arc::new(unsafe {
        memmap2::MmapOptions::new()
            .len(file.len())
            .map(tmp_file.as_file())
            .expect("mmap tmpfile for MoE section bytes")
    });

    let view = rnb_loader::rnb_moe_reader::RnbMoeView::from_bytes(&bytes_arc)
        .expect("MoE section header parse");
    let parsed = view
        .parse_moe_decode()
        .expect("MOE_DECODE present")
        .expect("MOE_DECODE parse ok");
    assert_eq!(parsed.layers.len(), 1);

    let layer = parsed.layers.into_iter().next().unwrap();
    let moe_section = super::convert_moe_section_decode_layer(layer, bytes_arc.clone());

    assert_eq!(moe_section.n_experts, 1);
    assert_eq!(moe_section.d_ff, 256);
    assert_eq!(moe_section.n_embd, 256);
    assert_eq!(moe_section.gate_up_quant, 0x12);
    assert_eq!(moe_section.down_quant, 0x14);
    assert_eq!(moe_section.shared_quant, 0xFF);
    assert!(moe_section.shared_expert.is_none());
    assert_eq!(moe_section.experts.len(), 1);

    let expert = &moe_section.experts[0];
    assert_eq!(expert.gate_up_rows.len(), 256);
    assert_eq!(expert.down_rows.len(), 256);

    // Spot-check a few rows: multipliers preserved, offsets point into the
    // Arc buffer, and the slice reconstructed from (offset, len) contains
    // the expected pattern.
    for &r in &[0usize, 1, 42, 255] {
        let row = &expert.gate_up_rows[r];
        assert!((row.gate_mul - ((r as f32) + 0.125)).abs() < 1e-6);
        assert!((row.up_mul - ((r as f32) + 0.250)).abs() < 1e-6);
        assert_eq!(row.blocks_len, GU_PAIR_Q4K_BYTES);
        let recon = &bytes_arc[row.blocks_offset..row.blocks_offset + row.blocks_len];
        let pat = (r & 0xFF) as u8;
        assert!(recon.iter().all(|&b| b == pat), "row {r} gate_up mismatch");

        let drow = &expert.down_rows[r];
        assert!((drow.down_mul - ((r as f32) * 2.0 + 1.0)).abs() < 1e-6);
        assert_eq!(drow.blocks_len, Q5K_INTSCALE_BYTES);
        let recon = &bytes_arc[drow.blocks_offset..drow.blocks_offset + drow.blocks_len];
        let pat = ((r ^ 0xA5) & 0xFF) as u8;
        assert!(recon.iter().all(|&b| b == pat), "row {r} down mismatch");
    }

    // file_bytes Arc must be the same allocation we passed in (shared).
    assert!(std::sync::Arc::ptr_eq(&moe_section.file_bytes, &bytes_arc));
}

#[test]
fn attach_moe_section_decode_no_file_returns_none() {
    // Point at a `.gguf` path whose sibling `.rnb` does not exist.
    let tmp = tempfile::TempDir::new().unwrap();
    let fake_gguf = tmp.path().join("does_not_exist.gguf");
    let mut weights = make_empty_model_weights();
    let got = attach_moe_section_decode(&fake_gguf, &mut weights);
    assert!(got.is_none());
}

#[test]
fn attach_moe_section_decode_non_rnbm_magic_returns_none() {
    // Write a file at `<stem>.rnb` whose first 4 bytes are NOT `RNBM`.
    // `attach_moe_section_decode` must probe the magic and bail cleanly.
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    std::fs::write(&rnb_path, b"RNBDxxxxxxxx").unwrap(); // dense magic
    std::fs::write(&gguf_path, b"").unwrap();

    let mut weights = make_empty_model_weights();
    let got = attach_moe_section_decode(&gguf_path, &mut weights);
    assert!(got.is_none(), "non-RNBM magic must not attach MoE sections");
}

#[test]
fn attach_moe_section_decode_kill_switch_env_returns_none() {
    let _lock = env_lock().lock().unwrap();
    let prev = std::env::var("RNB_MOE_DECODE").ok();
    std::env::set_var("RNB_MOE_DECODE", "0");

    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    let rnb_path = tmp.path().join("model.rnb");
    // Valid MoE section file present — still must not attach because kill switch is on.
    let body = build_synthetic_moe_decode_body_min();
    let moe_section_file = build_moe_section_rnb_file(&body);
    std::fs::write(&rnb_path, &moe_section_file).unwrap();
    std::fs::write(&gguf_path, b"").unwrap();

    let mut weights = make_empty_model_weights();
    let got = attach_moe_section_decode(&gguf_path, &mut weights);
    assert!(got.is_none(), "RNB_MOE_DECODE=0 must disable detection");

    match prev {
        Some(v) => std::env::set_var("RNB_MOE_DECODE", v),
        None => std::env::remove_var("RNB_MOE_DECODE"),
    }
}

/// Minimal `ModelWeights` with zero layers. `attach_moe_section_decode` should
/// return `None` for any no-layers case because the MoE section detection short-
/// circuits on missing/non-MoE section files before touching `weights.layers`.
fn make_empty_model_weights() -> ModelWeights {
    use rnb_core::tensor::Tensor;
    ModelWeights {
        token_embd: make_zero_quantized_weight(),
        output_norm: Tensor::from_slice(&[0.0f32; 1], &[1]),
        output: make_zero_quantized_weight(),
        layers: Vec::new(),
        gemma_per_layer: None,
        glm_dsa_attention: None,
        rope_freqs: None,
    }
}

#[cfg(feature = "cuda")]
fn minimal_cuda_product_prewarm_model_weights_for_test() -> ModelWeights {
    use rnb_core::tensor::Tensor;

    let hidden = 256;
    let ffn = 256;
    let mut weights = make_empty_model_weights();
    weights.output_norm = Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]);
    weights.layers = vec![
        LayerType::Attention(cuda_product_prewarm_attention_layer_for_test(
            hidden,
            ffn,
            make_cuda_product_prewarm_quant_weight_for_test(GGMLType::Q4_K, hidden, ffn, 0x31),
        )),
        LayerType::Attention(cuda_product_prewarm_attention_layer_for_test(
            hidden,
            ffn,
            make_cuda_product_prewarm_quant_weight_for_test(GGMLType::Q6_K, hidden, ffn, 0x73),
        )),
        LayerType::GatedDeltaNet(cuda_product_prewarm_gdn_layer_for_test(hidden, ffn)),
    ];
    weights
}

#[cfg(feature = "cuda")]
fn cuda_product_prewarm_alias_q4_raw_model_weights_for_test() -> ModelWeights {
    use rnb_core::tensor::Tensor;

    let hidden = 256;
    let ffn = 256;
    let mut weights = make_empty_model_weights();
    weights.output_norm = Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]);
    let shared = make_cuda_product_prewarm_quant_tensor_for_test(GGMLType::Q4_K, ffn, hidden, 0x44);
    let mut layer = cuda_product_prewarm_attention_layer_for_test(
        hidden,
        ffn,
        make_cuda_product_prewarm_quant_weight_for_test(GGMLType::Q4_K, hidden, ffn, 0x45),
    );
    layer.ffn_gate_weight = QuantizedWeight::new(shared.clone(), GGMLType::Q4_K, ffn, hidden);
    layer.ffn_up_weight = QuantizedWeight::new(shared, GGMLType::Q4_K, ffn, hidden);
    weights.layers = vec![LayerType::Attention(layer)];
    weights
}

#[cfg(feature = "cuda")]
fn cuda_product_prewarm_attention_layer_for_test(
    hidden: usize,
    ffn: usize,
    down_weight: QuantizedWeight,
) -> AttentionLayerWeights {
    use rnb_core::tensor::Tensor;

    AttentionLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        q_weight: make_zero_quantized_weight(),
        k_weight: make_zero_quantized_weight(),
        v_weight: make_zero_quantized_weight(),
        o_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            hidden,
            hidden,
            0x0f,
        ),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        q_norm: None,
        k_norm: None,
        post_attn_norm: None,
        out_scale: None,
        ffn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        post_ffw_norm: None,
        ffn_gate_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            ffn,
            hidden,
            0x11,
        ),
        ffn_up_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            ffn,
            hidden,
            0x23,
        ),
        ffn_down_weight: down_weight,
        ffn_gate_up_fused: None,
        moe: None,
        shared_expert_moe: None,
        v_proj_missing: false,
    }
}

#[cfg(feature = "cuda")]
fn cuda_product_prewarm_gdn_layer_for_test(hidden: usize, ffn: usize) -> GdnLayerWeights {
    use rnb_core::tensor::Tensor;

    let num_heads = 16;
    GdnLayerWeights {
        attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        post_attn_norm: Tensor::from_slice(&vec![1.0f32; hidden], &[hidden]),
        qkv_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            hidden,
            hidden,
            0x41,
        ),
        gate_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            hidden,
            hidden,
            0x42,
        ),
        ssm_a: Tensor::from_slice(&vec![0.0f32; num_heads], &[num_heads]),
        ssm_alpha: make_zero_quantized_weight(),
        ssm_beta: make_zero_quantized_weight(),
        ssm_conv1d: Tensor::from_slice(&vec![0.0f32; hidden], &[1, hidden]),
        ssm_dt_bias: Tensor::from_slice(&vec![0.0f32; num_heads], &[num_heads]),
        ssm_norm: Tensor::from_slice(&vec![1.0f32; num_heads], &[num_heads]),
        ssm_out: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            hidden,
            hidden,
            0x43,
        ),
        ffn_gate_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            ffn,
            hidden,
            0x51,
        ),
        ffn_up_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            ffn,
            hidden,
            0x52,
        ),
        ffn_down_weight: make_cuda_product_prewarm_quant_weight_for_test(
            GGMLType::Q4_K,
            hidden,
            ffn,
            0x53,
        ),
        ffn_gate_up_fused: None,
        shared_expert_moe: None,
    }
}

#[cfg(feature = "cuda")]
fn make_cuda_product_prewarm_quant_weight_for_test(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    seed: u8,
) -> QuantizedWeight {
    let tensor = make_cuda_product_prewarm_quant_tensor_for_test(ggml_type, rows, cols, seed);
    QuantizedWeight::new(tensor, ggml_type, rows, cols)
}

#[cfg(feature = "cuda")]
fn make_cuda_product_prewarm_quant_tensor_for_test(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    seed: u8,
) -> rnb_core::tensor::Tensor {
    let block_size = match ggml_type {
        GGMLType::Q4_K => 144,
        GGMLType::Q6_K => 210,
        _ => panic!("unsupported CUDA product prewarm fixture type: {ggml_type:?}"),
    };
    assert_eq!(cols % 256, 0, "K-quant fixture cols must align to blocks");
    let bytes = rows * (cols / 256) * block_size;
    let raw: Vec<u8> = (0..bytes)
        .map(|idx| seed.wrapping_add((idx % 251) as u8))
        .collect();
    rnb_core::tensor::Tensor::from_vec(raw, &[bytes])
}

/// Minimal placeholder `QuantizedWeight` for test `ModelWeights`. The fields
/// mirror the `QuantizedWeight` struct definition in `engine.rs`.
fn make_zero_quantized_weight() -> QuantizedWeight {
    use rnb_core::tensor::Tensor;
    QuantizedWeight::new(Tensor::from_slice(&[0.0f32; 1], &[1]), GGMLType::F32, 1, 1)
}
