#[test]
fn decode_cuda_calls_stay_behind_runtime_facade() {
    let decode =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine/decode.rs"))
            .expect("read decode.rs");
    let forbidden = [
        "cuda_runtime",
        "runtime::cuda_inference::cuda",
        "layer_gemv_enabled",
        "q4k_gemv",
        "q6k_gemv",
        "decode_attention_enabled",
        "attention_decode_hd256",
        "delta_net_enabled",
        "delta_net_decode",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| decode.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "decode.rs must call CUDA through runtime facade only: {offenders:?}"
    );
}

#[test]
fn decode_vulkan_gemv_dispatch_stays_behind_backend_runtime() {
    let decode =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine/decode.rs"))
            .expect("read decode.rs");
    let forbidden = [
        "gpu::decode_ffn_chain",
        "gpu::DecodeGemvWeight",
        "gpu::DecodeWeightKind",
        "gpu::decode_gemv(",
        "gpu::decode_gemv_multi(",
        "gpu::decode_gemv_multi_same_quant(",
        "gpu::decode_gemv_multi_async(",
        "gpu::decode_wait_async(",
        "gpu::verify_attention_layer",
        "gpu::verify_attention_qkv_layer",
        "gpu::verify_gdn_layer",
        ".append_attention_kv_f32_for_layer(",
        ".attention_decode_gpu_kv_mirror_for_layer(",
        ".attention_decode_f16_cache(",
        ".materialize_attention_kv_f16_for_layer(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| decode.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "decode.rs Vulkan GEMV dispatch must stay behind engine/backend_runtime.rs: {offenders:?}"
    );
}

#[test]
fn gdn_ssm_prefill_cuda_calls_stay_behind_runtime_facade() {
    let forward = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/forward.rs"
    ))
    .expect("read forward.rs");
    let forbidden = [
        "cuda_runtime",
        "gdn_prefill_gemv_mode",
        "gdn_prefill_chunk_unimplemented_for_test",
        "prefill_conv_enabled",
        "gpu_compute::ssm_conv1d_silu(",
        "prefill_delta_enabled",
        "gpu_compute::delta_net_prefill(",
        "gdn_gated_norm_gemm_enabled",
        "gpu_compute::gdn_gated_norm_silu_f32_gemm(",
        "gdn_gated_norm_enabled",
        "gpu_compute::gdn_gated_norm_silu(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| forward.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "GDN/SSM prefill CUDA calls must go through runtime facade only: {offenders:?}"
    );
}

#[test]
fn attention_and_qwen_moe_prefill_cuda_calls_stay_behind_runtime_facade() {
    let forward = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/forward.rs"
    ))
    .expect("read forward.rs");
    let forbidden = [
        "cuda_runtime",
        "runtime::cuda_inference::cuda",
        "prefill_flash_attention_enabled",
        "gpu_compute::attention_prefill_flash_hd256(",
        "prefill_moe_enabled",
        "gpu_compute::f32_gemm_batch(",
        "moe_route_hist_enabled",
        "shared_f32_enabled",
        "gpu_compute::f32_shared_expert(",
        "gpu_compute::qwen35_sparse_experts_by_token(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| forward.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "attention/Qwen MoE prefill CUDA calls must go through runtime facade only: {offenders:?}"
    );
}

#[test]
fn qwen_moe_decode_cuda_calls_stay_behind_backend_runtime() {
    let moe = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine/moe.rs"))
        .expect("read moe.rs");
    let forbidden = [
        "cuda_runtime",
        "qwen_moe_cuda::",
        "runtime::cuda_inference::cuda",
        "q4k_gemv",
        "q5k_gemv",
        "q6k_gemv",
        "qwen35_expert",
        "qwen35_sparse_experts",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| moe.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "Qwen MoE decode CUDA calls must stay behind engine/backend_runtime.rs: {offenders:?}"
    );
}

#[test]
fn engine_vulkan_policy_and_output_logits_stay_behind_runtime_facade() {
    let engine = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine.rs"))
        .expect("read engine.rs");
    let forbidden = [
        "ggml_to_output_quant",
        "PrefillLayerRuntimeConfig::from_model_shape",
        "init_layer_gemv",
        "gpu::output_logits_enabled",
        "gpu::prefill_chunk_size_for_active_path",
        "gpu::decode_layers_allowed",
        "gpu::max_decode_layer",
        "gpu::output_logits_id",
        ".gemv(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| engine.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "engine.rs Vulkan policy/output logits calls must go through runtime facade only: {offenders:?}"
    );
}

#[test]
fn vulkan_runtime_bridge_does_not_reexport_backend_wildcard() {
    let runtime_gpu = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rnb-runtime/src/vulkan_inference.rs"
    ))
    .expect("read rnb-runtime vulkan_inference.rs");
    let forbidden = ["gpu::*", "pub use super::super::gpu::*"];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| runtime_gpu.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "rnb-runtime vulkan_inference.rs must expose selected Vulkan facade APIs only: {offenders:?}"
    );
}

#[test]
fn vulkan_layer_runtime_does_not_deref_to_backend_runtime() {
    let runtime_gpu = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rnb-runtime/src/vulkan_inference.rs"
    ))
    .expect("read rnb-runtime vulkan_inference.rs");
    let forbidden = [
        "Deref for LayerRuntime",
        "DerefMut for LayerRuntime",
        "std::ops::Deref",
        "std::ops::{Deref",
        "core::ops::Deref",
        "core::ops::{Deref",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| runtime_gpu.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "LayerRuntime must expose explicit facade methods instead of derefing to backend runtime: {offenders:?}"
    );
}

#[test]
fn prefill_does_not_name_vulkan_weight_ids_directly() {
    let prefill = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/prefill.rs"
    ))
    .expect("read prefill.rs");
    let forbidden = [
        "gpu::ffn_gate_id",
        "gpu::ffn_up_id",
        "gpu::ffn_down_id",
        "gpu::o_proj_id",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| prefill.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "prefill.rs must call Vulkan weight-id operations through runtime_gpu facade: {offenders:?}"
    );
}

#[test]
fn prefill_vulkan_window_dispatch_stays_behind_backend_runtime() {
    let prefill = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/prefill.rs"
    ))
    .expect("read prefill.rs");
    let forbidden = [
        "gpu::quantized_bytes_supported",
        "gpu::ggml_to_quant",
        "gpu::quant_or_error",
        "gpu::attention_block_window_for_layer",
        "gpu::o_proj_gemv_window",
        "gpu::ffn_chain_window_with_residual_from_resident_input",
        ".rms_norm_window(",
        ".append_attention_kv_f32_for_layer(",
        ".attention_decode_window_grouped_from_mirror_for_layer(",
        ".materialize_attention_kv_f16_for_layer(",
        ".materialize_attention_kv_f16_grouped_for_layer(",
        ".materialize_attention_kv_f16_range_for_layer_untracked(",
        ".materialize_attention_kv_f16_grouped_range_for_layer_untracked(",
        ".materialize_gdn_conv_state_f32_for_layer_untracked(",
        ".record_batched_materialization_download(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| prefill.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "prefill.rs Vulkan window dispatch must stay behind engine/backend_runtime.rs: {offenders:?}"
    );
}

#[test]
fn materialize_vulkan_kv_shapes_stay_behind_runtime_facade() {
    let materialize = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/backend_runtime/materialize.rs"
    ))
    .expect("read materialize.rs");
    let forbidden = [
        ".materialize_attention_kv_f16_for_layer(",
        ".materialize_attention_kv_f16_grouped_for_layer(",
        ".materialize_attention_kv_f16_range_for_layer_untracked(",
        ".materialize_attention_kv_f16_grouped_range_for_layer_untracked(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| materialize.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "attention KV materialization shape dispatch must go through runtime facade: {offenders:?}"
    );
}

#[test]
fn forward_gdn_prefill_gpu_attempt_errors_stay_behind_backend_runtime() {
    let forward = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/forward.rs"
    ))
    .expect("read forward.rs");
    let forbidden = ["backend_runtime::try_gdn_prefill_ffn_chain_window("];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| forward.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "forward.rs raw GDN prefill GPU attempts must stay behind engine/backend_runtime.rs wrappers: {offenders:?}"
    );
}

#[test]
fn decode_inference_restores_gpu_runtime_before_error_return() {
    let decode_inference = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/decode_inference.rs"
    ))
    .expect("read decode_inference.rs");
    let take = decode_inference
        .find("take_gpu_runtime")
        .expect("decode should take gpu runtime");
    let restore = decode_inference
        .find("restore_gpu_runtime(gpu_runtime)")
        .expect("decode should restore gpu runtime");
    let propagate = decode_inference
        .find("decode_result?;")
        .expect("decode should propagate result after restore");

    assert!(
        take < restore && restore < propagate,
        "decode GPU runtime must be restored before decode_result is propagated"
    );
}

#[test]
fn slice1_prefill_restores_gpu_runtime_before_error_return() {
    let inference = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/inference.rs"
    ))
    .expect("read inference.rs");
    let take = inference
        .find("take_gpu_runtime")
        .expect("slice1 prefill should take gpu runtime");
    let restore = inference
        .find("restore_gpu_runtime(gpu_runtime)")
        .expect("slice1 prefill should restore gpu runtime");
    let propagate = inference
        .find("result?")
        .expect("slice1 prefill should propagate result after restore");

    assert!(
        take < restore && restore < propagate,
        "slice1 prefill GPU runtime must be restored before result is propagated"
    );
}

#[test]
fn llm_engine_uses_backend_neutral_gpu_runtime_names() {
    let files = [
        "/src/engine.rs",
        "/src/engine/decode.rs",
        "/src/engine/forward.rs",
        "/src/engine/prefill.rs",
    ];
    let forbidden = [
        "gpu::PrefillLayerRuntime",
        "gpu::PrefillLayerRuntimeCounters",
        "gpu::QuantType",
        "gpu::Runtime",
    ];

    let mut offenders = Vec::new();
    for file in files {
        let source = std::fs::read_to_string(format!("{}{}", env!("CARGO_MANIFEST_DIR"), file))
            .unwrap_or_else(|error| panic!("read {file}: {error}"));
        for pattern in forbidden {
            if source.contains(pattern) {
                offenders.push(format!("{file}: {pattern}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "LLM engine must use backend-neutral gpu runtime names: {offenders:?}"
    );
}

#[test]
fn backend_runtime_state_uses_runtime_owned_gpu_facade() {
    let state = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/backend_runtime/state.rs"
    ))
    .expect("read backend_runtime/state.rs");
    let forbidden = [
        "type GpuRuntime = gpu::Runtime",
        "type GpuQuant = gpu::Quant",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| state.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "backend_runtime::state must name rnb-runtime-owned GPU facade types: {offenders:?}"
    );
}

#[test]
fn transformed_layout_metadata_stays_out_of_llm_runtime_parsing() {
    let forbidden = [
        "TransformedLayoutDescriptor",
        "TransformedLayoutKind",
        "transformed_layouts",
        "producer_options_hash",
        "source_fingerprint",
    ];

    fn collect_rust_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|error| panic!("read {dir:?}: {error}"))
        {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                collect_rust_files(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }

    let src_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let boundary_test = std::path::Path::new("runtime/boundary_tests.rs");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);

    let mut offenders = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(&src_dir)
            .unwrap_or_else(|error| panic!("strip src prefix for {file:?}: {error}"));
        if rel == boundary_test {
            continue;
        }
        let source =
            std::fs::read_to_string(&file).unwrap_or_else(|error| panic!("read {file:?}: {error}"));
        for pattern in forbidden {
            if source.contains(pattern) {
                offenders.push(format!("{}: {pattern}", rel.display()));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "rnb-llm must receive typed transformed views from runtime/backend-api, not parse transformed metadata directly: {offenders:?}"
    );
}
