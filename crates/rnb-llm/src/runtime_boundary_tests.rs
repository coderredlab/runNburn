use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read source directory") {
        let entry = entry.expect("read source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs")
            && !is_boundary_test_source(&path)
        {
            out.push(path);
        }
    }
}

fn is_boundary_test_source(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "runtime_boundary_tests.rs" || name == "boundary_tests.rs")
}

#[test]
fn llm_reads_compiled_backends_from_runtime_boundary() {
    let backends = crate::compiled_runtime_backends();

    assert_eq!(
        backends.len(),
        crate::runtime::BackendRegistry::compiled().backends().len()
    );
    #[cfg(feature = "cuda")]
    assert!(backends.contains(&"cuda"));
    #[cfg(feature = "vulkan")]
    assert!(backends.contains(&"vulkan"));
}

#[test]
fn llm_manifest_does_not_depend_on_concrete_gpu_backend_crates() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read rnb-llm Cargo.toml");

    assert!(!manifest.contains("rnb-backend-cuda"));
    assert!(!manifest.contains("rnb-backend-vulkan"));
    assert!(!manifest.contains("rnb-backend-opencl"));
    assert!(!manifest.contains("rnb-backend-api"));
    assert!(!manifest.contains("rnb-platform"));
    assert!(!manifest.contains("rnb-memory"));
    assert!(!manifest.contains("rnb-cpu"));
    assert!(!manifest.contains(&["rnb", "-", "gemm"].concat()));
}

#[test]
fn llm_does_not_own_arch_specific_kernel_modules() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    assert!(!manifest_dir.join("src/simd").exists());
    assert!(!manifest_dir.join("asm").exists());
    assert!(!manifest_dir.join("build.rs").exists());
}

#[test]
fn llm_does_not_probe_cpu_features_directly() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = ["engine/forward/attention_compute.rs"];
    let forbidden = [
        ["std", "::", "arch"].concat(),
        ["core", "::", "arch"].concat(),
        ["is_", "aarch64", "_feature_detected"].concat(),
        ["target", "_feature"].concat(),
    ];

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            forbidden
                .iter()
                .any(|pattern| text.contains(pattern.as_str()))
                .then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "CPU feature probes/declarations must go through runtime/platform boundary: {offenders:?}"
    );
}

#[test]
fn runtime_imports_stay_behind_llm_runtime_facade() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let runtime_file = src_dir.join("runtime.rs");
    let allowed = ["engine/models/qwen/prefill_moe.rs"];
    let forbidden = ["rnb_runtime", "::"].concat();

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter(|path| path != &runtime_file)
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            text.contains(&forbidden).then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "runtime imports must go through src/runtime.rs: {offenders:?}"
    );
}

#[test]
fn llm_does_not_name_moe_block_kernel_layouts() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let forbidden = ["moe", "_blocks"].concat();

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .flat_map(|path| {
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            text.contains(&forbidden)
                .then(|| {
                    path.strip_prefix(src_dir)
                        .unwrap_or(&path)
                        .display()
                        .to_string()
                })
                .into_iter()
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "MoE block kernel layouts must stay behind rnb-cpu gemm/runtime boundary: {offenders:?}"
    );
}

#[test]
fn llm_backend_runtime_names_are_isolated_to_known_migration_files() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = [
        "lib.rs",
        "runtime.rs",
        "engine/runtime.rs",
        "engine/cpu_runtime.rs",
        "engine/cuda_runtime.rs",
        "engine/gemm_runtime.rs",
        "engine/gpu_runtime.rs",
        "engine/memory_runtime.rs",
        "engine/packed_runtime.rs",
        "engine/platform_runtime.rs",
        "engine/policy.rs",
        "engine/inference.rs",
    ];
    let forbidden = [
        ["crate", "::", "runtime", "::", "cpu"].concat(),
        ["crate", "::", "runtime", "::", "cuda"].concat(),
        ["crate", "::", "runtime", "::", "gpu"].concat(),
        ["crate", "::", "runtime", "::", "compute"].concat(),
        ["crate", "::", "runtime", "::", "gemm"].concat(),
        ["crate", "::", "runtime", "::", "memory"].concat(),
        ["crate", "::", "runtime", "::", "packed_weights"].concat(),
        ["crate", "::", "runtime", "::", "platform"].concat(),
        ["crate", "::", "runtime", "::", "policy"].concat(),
        ["gpu", "::", "LayerRuntime"].concat(),
        "gpu_compute".to_string(),
    ];

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            forbidden
                .iter()
                .any(|pattern| text.contains(pattern.as_str()))
                .then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "backend runtime names must stay behind runtime facade: {offenders:?}"
    );
}

#[test]
fn llm_does_not_parse_runtime_policy_env_directly() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = [
        "lib.rs",
        "engine/tests.rs",
        "engine/backend_runtime/init.rs",
        "engine/backend_runtime/output.rs",
        "engine/backend_runtime/vulkan_attention_prefill.rs",
        "engine/backend_runtime/vulkan_gdn_prefill.rs",
        "engine/decode_inference/output.rs",
        "engine/forward.rs",
        "engine/forward/attention_compute.rs",
        "engine/forward/dense_chain.rs",
        "engine/forward/projection.rs",
        "engine/init.rs",
        "engine/models/nemotron/mamba.rs",
        "engine/models/nemotron/moe.rs",
        "engine/models/qwen/gdn_decode.rs",
        "engine/models/qwen/gdn_forward.rs",
        "engine/models/qwen/moe_view/fanout.rs",
        "engine/prefill/gpu_executor.rs",
        "engine/quantized_weight.rs",
    ];
    let forbidden = [
        ["std", "::", "env", "::", "var", "("].concat(),
        ["std", "::", "env", "::", "var_os", "("].concat(),
    ];

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            forbidden
                .iter()
                .any(|pattern| text.contains(pattern))
                .then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "runtime/gemm policy env parsing must stay behind runtime policy modules: {offenders:?}"
    );
}

#[test]
fn llm_gemm_policy_calls_stay_behind_quantized_dispatch() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = ["engine/quantized_dispatch.rs"];
    let forbidden = [["runtime", "::", "gemm", "::", "policy"].concat()];

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            forbidden
                .iter()
                .any(|pattern| text.contains(pattern))
                .then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "GEMM policy calls must stay behind engine/quantized_dispatch.rs: {offenders:?}"
    );
}

#[test]
fn llm_gemm_runtime_calls_stay_behind_dispatch_wrappers() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = [
        "lib.rs",
        "engine.rs",
        "engine/runtime.rs",
        "engine/dense_dispatch.rs",
        "engine/dequant.rs",
        "engine/moe_section_dispatch.rs",
        "engine/moe_shadow_dispatch.rs",
        "engine/quantized_dispatch.rs",
        "engine/quantized_dispatch/aarch64_gemv.rs",
        "engine/quantized_dispatch/decode.rs",
        "engine/quantized_dispatch/prefill.rs",
        "engine/quantized_packing.rs",
        "engine/scalar_gemv.rs",
    ];
    let forbidden = "gemm_runtime";

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            text.contains(forbidden).then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "GEMM runtime calls must stay behind dispatch wrappers: {offenders:?}"
    );
}

#[test]
fn gpu_runtime_state_stays_behind_backend_runtime() {
    let src_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let allowed = [
        "lib.rs",
        "engine/backend_runtime.rs",
        "engine/backend_runtime/state.rs",
    ];
    let forbidden = "gpu_layer_runtime";

    let mut files = Vec::new();
    collect_rs_files(src_dir, &mut files);

    let offenders: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(src_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            if allowed.contains(&rel.as_str()) {
                return None;
            }
            let text = std::fs::read_to_string(&path).expect("read rust source file");
            text.contains(forbidden).then_some(rel)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "GPU runtime state field must stay behind backend runtime state boundary: {offenders:?}"
    );
}

#[test]
fn forward_prefill_quantized_gemv_stays_behind_dispatch() {
    let forward = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/engine/forward.rs"
    ))
    .expect("read forward.rs");
    let forbidden = [
        "quantize_input_q8(",
        "quantize_input_q8k(",
        ".gemv_vec_q8(",
        ".gemv_vec_q8k(",
        "platform_runtime::aarch64_has_dotprod",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| forward.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "forward.rs prefill quantized GEMV dispatch must stay behind engine/quantized_dispatch.rs: {offenders:?}"
    );
}

#[test]
fn decode_quantized_gemv_stays_behind_dispatch() {
    let decode =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine/decode.rs"))
            .expect("read decode.rs");
    let forbidden = [
        "fast_dotprod_enabled(",
        "quantize_input_q8_into(",
        "quantize_input_q8k_into(",
        ".gemv_into_q8(",
        ".gemv_into_q8k(",
        "platform_runtime::aarch64_has_dotprod",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| decode.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "decode.rs quantized GEMV dispatch must stay behind engine/quantized_dispatch.rs: {offenders:?}"
    );
}

#[test]
fn decode_gpu_attempt_error_handling_stays_behind_backend_runtime() {
    let decode =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/engine/decode.rs"))
            .expect("read decode.rs");
    let forbidden = [
        "backend_runtime::try_decode_ffn_chain(",
        "backend_runtime::try_decode_ffn_gate_async(",
        "backend_runtime::wait_decode_async(",
        "backend_runtime::try_decode_attention_qkv(",
        "backend_runtime::try_decode_gdn_qkv_gate(",
        "backend_runtime::try_decode_gdn_alpha_beta(",
        "backend_runtime::try_decode_attention_single_head(",
        "backend_runtime::materialize_attention_kv_for_layer(",
    ];

    let offenders: Vec<&str> = forbidden
        .into_iter()
        .filter(|pattern| decode.contains(pattern))
        .collect();

    assert!(
        offenders.is_empty(),
        "decode.rs raw GPU attempt calls must stay behind engine/backend_runtime.rs wrappers: {offenders:?}"
    );
}
