use rnb_backend_vulkan::spirv::{
    emit_attention_decode, emit_elem_add, emit_elem_add_broadcast, emit_elem_add_out,
    emit_f32_gemv, emit_gdn_delta_precompute, emit_gdn_delta_sequence,
    emit_gdn_delta_sequence_d128, emit_gdn_delta_step, emit_gdn_gated_norm_silu,
    emit_q4k_block_reduce, emit_q4k_gate_up_batch4, emit_q4k_gemv, emit_q4k_gemv_batch4,
    emit_q4k_gemv_block_partial, emit_q4k_gemv_rowmajor_batched, emit_q4k_gemv_wg_reduce,
    emit_q4k_q8k_gemv, emit_q5k_gemv, emit_q5k_gemv_batch4, emit_q6k_gemv, emit_q6k_gemv_batch4,
    emit_q6k_gemv_batch4_f16, emit_q6k_q8k_gemv, emit_q8_0_gemv, emit_quantize_to_q8k,
    emit_rms_norm, emit_rope_apply, emit_silu_mul, SpirvModule,
};

use rnb_backend_vulkan::{GpuWeightMode, QuantType, VulkanLayerGemv, WeightId, WeightKind};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn vulkan_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("vulkan test lock poisoned")
}

fn new_vulkan_layer(
    hidden_size: usize,
    intermediate_size: usize,
    scratch_size: usize,
    weight_mode: GpuWeightMode,
) -> Result<(MutexGuard<'static, ()>, VulkanLayerGemv), String> {
    let guard = vulkan_test_lock();
    VulkanLayerGemv::new(hidden_size, intermediate_size, scratch_size, weight_mode)
        .map(|vk| (guard, vk))
}

#[test]
fn test_empty_module_header() {
    let module = SpirvModule::new();
    let words = module.encode();
    assert_eq!(words[0], 0x07230203);
    assert_eq!(words[1], 0x00010300);
    assert_eq!(words[2], 0);
    assert!(words[3] >= 1);
    assert_eq!(words[4], 0);
}

#[test]
fn test_minimal_compute_shader() {
    let mut m = SpirvModule::new();
    m.capability(1);
    m.memory_model(0, 1);
    let void = m.type_void();
    let fn_void = m.type_function(void, &[]);
    let func = m.alloc_id();
    m.entry_point(5, func, "main", &[]);
    m.execution_mode_local_size(func, 256, 1, 1);
    m.function(void, func, 0, fn_void);
    m.label();
    m.ret();
    m.function_end();
    let words = m.encode();
    assert_eq!(words[0], 0x07230203);
    assert!(words.len() > 20);
}

#[test]
fn test_elementwise_self_test_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(256, 256, 256, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };

    vk.self_test_elementwise().unwrap();
}

#[test]
fn test_q4k_block_parallel_self_test_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(1024, 256, 256, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };

    let max_diff = vk.self_test_q4k_block_parallel().unwrap();
    assert!(
        max_diff < 0.05,
        "Q4_K block-parallel self-test max_diff too high: {max_diff}"
    );
}

#[test]
fn test_quant_prompt_batch_matches_scalar_on_tail() {
    let (_guard, mut vk) = match new_vulkan_layer(4096, 4096, 4096, GpuWeightMode::RowMajor) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    for (quant, max_abs, max_rel) in vk.self_test_quant_prompt_batch().unwrap() {
        assert!(
            max_abs <= 0.125 && max_rel <= 1.0e-4,
            "{quant:?} prompt batch differs from scalar: max_abs={max_abs} max_rel={max_rel}"
        );
    }
}

#[test]
fn test_q4k_gate_up_batch4_matches_scalar_on_tail() {
    let (_guard, mut vk) = match new_vulkan_layer(4096, 4096, 4096, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    let (max_abs, max_rel) = vk.self_test_q4k_gate_up_batch4().unwrap();
    eprintln!("Q4_K gate/up batch4 parity: max_abs={max_abs} max_rel={max_rel}");
    assert!(
        max_abs <= 0.125 && max_rel <= 1.0e-4,
        "Q4_K gate/up prompt batch differs from scalar: max_abs={max_abs} max_rel={max_rel}"
    );
}

#[test]
fn test_q8_0_gemv_shader_generates() {
    let spirv = emit_q8_0_gemv(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_f32_gemv_shader_generates() {
    let spirv = emit_f32_gemv(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_gdn_gated_norm_silu_shader_generates() {
    let spirv = emit_gdn_gated_norm_silu(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_gdn_delta_step_shader_generates() {
    let spirv = emit_gdn_delta_step(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_gdn_delta_precompute_shader_generates() {
    let spirv = emit_gdn_delta_precompute(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/gdn_delta_precompute.spv", bytes).unwrap();
}

#[test]
fn test_gdn_delta_sequence_shader_generates() {
    let spirv = emit_gdn_delta_sequence(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/gdn_delta_sequence.spv", bytes).unwrap();
}

#[test]
fn test_elem_add_out_shader_generates() {
    let spirv = emit_elem_add_out(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_elem_add_broadcast_shader_generates() {
    let spirv = emit_elem_add_broadcast(256);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "too short: {} words", spirv.len());
    assert!(spirv[3] > 30, "bound too low: {}", spirv[3]);
}

#[test]
fn test_q8_0_gemv_shader_has_entry_point() {
    let spirv = emit_q8_0_gemv(256);
    let mut found = false;
    let mut i = 5;
    while i < spirv.len() {
        let word_count = (spirv[i] >> 16) as usize;
        let opcode = spirv[i] & 0xFFFF;
        if opcode == 15 {
            assert_eq!(spirv[i + 1], 5, "should be GLCompute");
            found = true;
            break;
        }
        i += word_count;
    }
    assert!(found, "OpEntryPoint not found");
}

#[test]
fn test_dump_spirv_for_validation() {
    let spirv = emit_q8_0_gemv(256);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q8_0_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q8_0_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_dump_q4k_spirv() {
    let spirv = emit_q4k_gemv(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q4k_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q4k_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_dump_q4k_batch4_spirv() {
    let spirv = emit_q4k_gemv_batch4(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q4k_gemv_batch4.spv", &bytes).unwrap();
}

#[test]
fn test_dump_q4k_gate_up_batch4_spirv() {
    let spirv = emit_q4k_gate_up_batch4(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write("/tmp/q4k_gate_up_batch4.spv", &bytes).unwrap();
}

#[test]
fn test_dump_q4k_rowmajor_batched_spirv() {
    let spirv = emit_q4k_gemv_rowmajor_batched(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write("/tmp/q4k_gemv_rowmajor_batched.spv", &bytes).unwrap();
}

#[test]
fn test_dump_q6k_batch4_spirv() {
    let spirv = emit_q6k_gemv_batch4(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q6k_gemv_batch4.spv", &bytes).unwrap();
}
#[test]
fn test_dump_q6k_batch4_f16_spirv() {
    let spirv = emit_q6k_gemv_batch4_f16(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q6k_gemv_batch4_f16.spv", &bytes).unwrap();
}

#[test]
fn test_dump_q5k_batch4_spirv() {
    let spirv = emit_q5k_gemv_batch4(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q5k_gemv_batch4.spv", &bytes).unwrap();
}
#[test]
fn test_dump_gdn_delta_sequence_d128_spirv() {
    let spirv = emit_gdn_delta_sequence_d128();
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/gdn_delta_sequence_d128.spv", &bytes).unwrap();
}

#[test]
fn test_q4k_block_partial_shader_generates() {
    let spirv = emit_q4k_gemv_block_partial(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert!(
        spirv.len() > 100,
        "Q4_K block partial shader should have a real body"
    );
}

#[test]
fn test_q4k_block_reduce_shader_generates() {
    let spirv = emit_q4k_block_reduce(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert!(
        spirv.len() > 50,
        "Q4_K block reduce shader should have a real body"
    );
}

#[test]
fn test_q4k_wg_reduce_shader_generates() {
    let spirv = emit_q4k_gemv_wg_reduce(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert!(
        spirv.len() > 100,
        "Q4_K workgroup reduce shader should have a real body"
    );
}

#[test]
fn test_dump_q6k_spirv() {
    let spirv = emit_q6k_gemv(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q6k_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q6k_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_q5k_gemv_shader_generates() {
    let spirv = emit_q5k_gemv(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 120, "too short: {} words", spirv.len());
}

#[test]
fn test_dump_q5k_spirv() {
    let spirv = emit_q5k_gemv(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q5k_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q5k_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_q5k_gemv_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(512, 4, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };

    let q5_block = [0u8; 176];
    let input_all = vec![1.0f32; 512];
    let mut output_all = vec![7.0f32; 2];

    vk.gemv_window(
        WeightId {
            layer: 77,
            kind: WeightKind::QProj,
        },
        &q5_block,
        1,
        256,
        QuantType::Q5K,
        &input_all,
        &mut output_all,
    )
    .expect("Q5_K gemv_window should execute");

    for &v in &output_all {
        assert!(v.abs() < 1e-6, "expected near-zero output, got {}", v);
    }
}

#[test]
fn test_q5k_gemv_multi_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(512, 4, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };

    let q5_block = [0u8; 176];
    let input_all = vec![0.25f32; 512];
    let weights = [(
        WeightId {
            layer: 78,
            kind: WeightKind::KProj,
        },
        &q5_block[..],
        1usize,
        256usize,
        QuantType::Q5K,
    )];
    let mut out0 = vec![3.0f32; 2];
    let mut outputs: [&mut [f32]; 1] = [&mut out0];

    vk.gemv_multi_window(&input_all, 256, &weights, &mut outputs)
        .expect("Q5_K gemv_multi_window should execute");

    for &v in &out0 {
        assert!(v.abs() < 1e-6, "expected near-zero output, got {}", v);
    }
}

#[test]
fn test_q5k_gemv_multi_async_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(256, 4, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };

    let q5_block = [0u8; 176];
    let input = vec![0.5f32; 256];
    let weights = [(
        WeightId {
            layer: 79,
            kind: WeightKind::VProj,
        },
        &q5_block[..],
        1usize,
        256usize,
        QuantType::Q5K,
    )];

    vk.gemv_multi_async(&input, &weights)
        .expect("Q5_K gemv_multi_async should submit");

    let mut out0 = vec![9.0f32; 1];
    let mut outputs: [&mut [f32]; 1] = [&mut out0];
    vk.wait_async(&mut outputs)
        .expect("Q5_K wait_async should complete");

    assert!(
        out0[0].abs() < 1e-6,
        "expected near-zero output, got {}",
        out0[0]
    );
}

#[test]
fn test_dump_silu_mul_spirv() {
    let spirv = emit_silu_mul(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/silu_mul.spv", &bytes).unwrap();
    eprintln!(
        "Dumped silu_mul: {} words ({} bytes)",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_dump_elem_add_spirv() {
    let spirv = emit_elem_add(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/elem_add.spv", &bytes).unwrap();
    eprintln!(
        "Dumped elem_add: {} words ({} bytes)",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_dump_rms_norm_spirv() {
    let spirv = emit_rms_norm(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/rms_norm.spv", &bytes).unwrap();
    eprintln!(
        "Dumped rms_norm: {} words ({} bytes)",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_attention_decode_shader_generates() {
    let spirv = emit_attention_decode(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 80, "too short: {} words", spirv.len());
}

#[test]
fn test_attention_decode_shader_has_entry_point_and_ext_inst() {
    let spirv = emit_attention_decode(64);
    let mut found_entry = false;
    let mut found_ext_inst = false;
    let mut i = 5;
    while i < spirv.len() {
        let word_count = (spirv[i] >> 16) as usize;
        let opcode = spirv[i] & 0xFFFF;
        if opcode == 15 {
            assert_eq!(spirv[i + 1], 5, "should be GLCompute");
            found_entry = true;
        }
        if opcode == 12 {
            found_ext_inst = true;
        }
        i += word_count;
    }
    assert!(found_entry, "OpEntryPoint not found");
    assert!(found_ext_inst, "OpExtInst not found");
}

#[test]
fn test_attention_decode_self_test_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode()
        .expect("attention self-test should pass");
}

#[test]
fn test_attention_decode_self_test_handles_large_scores_stably() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_large_scores()
        .expect("attention large-score self-test should pass");
}

#[test]
fn test_attention_decode_gpu_kv_mirror_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_gpu_kv_mirror()
        .expect("attention GPU KV mirror self-test should pass");
}

#[test]
fn test_attention_decode_gpu_kv_mirror_isolated_per_layer() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_gpu_kv_mirror_per_layer()
        .expect("attention GPU KV mirror per-layer self-test should pass");
}

#[test]
fn test_attention_decode_gpu_kv_mirror_can_materialize_f16_per_layer() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_gpu_kv_materialize_per_layer()
        .expect("attention GPU KV mirror materialize self-test should pass");
}

#[test]
fn test_attention_decode_gpu_kv_mirror_can_materialize_grouped_layer() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_gpu_kv_materialize_grouped_layer()
        .expect("attention GPU KV grouped materialize self-test should pass");
}

#[test]
fn test_grouped_materialize_increments_materialization_counter() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_attention_decode_gpu_kv_materialize_grouped_layer()
        .expect("grouped materialize self-test should pass");
    let stats = vk.runtime_counters();
    assert!(stats.materializations > 0, "expected materializations > 0");
}

#[test]
fn test_prefill_hidden_roundtrip_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_prefill_hidden_roundtrip()
        .expect("prefill hidden roundtrip self-test should pass");
}

#[test]
fn test_prefill_hidden_offset_writes_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_prefill_hidden_offset_writes()
        .expect("prefill hidden offset write self-test should pass");
}

#[test]
fn test_gdn_conv1d_silu_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_conv1d_silu_window()
        .expect("gdn conv1d silu self-test should pass");
}

#[test]
fn test_gdn_qkv_conv_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_qkv_conv_window()
        .expect("gdn qkv conv self-test should pass");
}

#[test]
fn test_gdn_qkv_conv_window_resident_conv_state_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_qkv_conv_window_resident_conv_state()
        .expect("gdn resident conv_state self-test should pass");
}

#[test]
fn test_gdn_qkv_conv_window_resident_conv_state_strided_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_qkv_conv_window_resident_conv_state_strided()
        .expect("gdn resident conv_state strided self-test should pass");
}

#[test]
fn test_q_window_into_kv_mirror_avoids_kv_host_roundtrip() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_q_window_into_kv_mirror_avoids_kv_host_roundtrip()
        .expect("q_window_into_kv_mirror should avoid kv host roundtrip");
}

#[test]
fn test_q_window_decode_project_avoids_attn_host_roundtrip() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_q_window_decode_project_avoids_attn_host_roundtrip()
        .expect("q_window decode project should avoid attn host roundtrip");
}

#[test]
fn test_q_window_decode_project_elides_q_host_download() {
    let (_guard, mut vk) = match new_vulkan_layer(128, 128, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_q_window_decode_project_elides_q_host_download()
        .expect("q_window decode project should elide q host download");
}

#[test]
fn test_attention_decode_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_attention_decode_window()
        .expect("attention window decode self-test should pass");
}

#[test]
fn test_ffn_chain_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_ffn_chain_window()
        .expect("ffn chain window self-test should pass");
}

#[test]
fn test_gemv_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gemv_window()
        .expect("gemv window self-test should pass");
}

#[test]
fn test_f32_gemv_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_f32_gemv_window()
        .expect("f32 gemv window self-test should pass");
}

#[test]
fn test_gdn_gated_norm_silu_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_gated_norm_silu()
        .expect("gdn gated norm silu self-test should pass");
}

#[test]
fn test_gdn_delta_step_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_gdn_delta_step()
        .expect("gdn delta step self-test should pass");
}
#[test]
fn test_gdn_delta_sequence_d128_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    let max_diff = vk
        .self_test_gdn_delta_sequence_d128()
        .expect("parallel d128 GDN delta sequence self-test should execute");
    assert!(
        max_diff < 2e-4,
        "parallel d128 GDN delta sequence max_diff={max_diff}"
    );
}

#[test]
fn test_qkv_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_qkv_window()
        .expect("qkv window self-test should pass");
}

#[test]
fn test_q_window_into_kv_mirror_and_decode_grouped_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped()
        .expect("mirror-direct grouped attention self-test should pass");
}

#[test]
fn test_q_window_into_kv_mirror_and_decode_grouped_combined_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped_combined()
        .expect("combined mirror-direct grouped attention self-test should pass");
}

#[test]
#[ignore = "tiny test fixture budget is insufficient for current mirror fast-path allocations"]
fn test_q_window_into_kv_mirror_and_decode_grouped_combined_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 128, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped_combined()
        .expect("combined mirror-direct grouped attention self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 2);
    assert_eq!(stats.upload_bytes, 128);
    assert_eq!(stats.download_bytes, 128);
    assert_eq!(stats.materializations, 0);
}

#[test]
#[ignore = "tiny test fixture budget is insufficient for current mirror fast-path allocations"]
fn test_q_window_into_kv_mirror_and_decode_grouped_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 128, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped()
        .expect("mirror-direct grouped attention self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 2);
    assert_eq!(stats.upload_bytes, 128);
    assert_eq!(stats.download_bytes, 128);
    assert_eq!(stats.materializations, 0);
}

#[test]
fn test_rms_norm_window_executes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_rms_norm_window()
        .expect("rms norm window self-test should pass");
}

#[test]
fn test_attention_decode_window_rejects_bad_grouped_shapes() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    let q_all = vec![0.0f32; 4];
    let k_all = vec![0.0f32; 4];
    let v_all = vec![0.0f32; 4];
    let mut out = vec![0.0f32; 8];

    let err = vk
        .attention_decode_window_grouped_for_layer(
            0, &q_all, &k_all, &v_all, 4, 2, 2, 1, 0, &mut out,
        )
        .expect_err("bad grouped shapes should return Err");
    assert!(err.contains("shape"), "unexpected error: {err}");
}

#[test]
fn test_attention_decode_window_rejects_partially_short_grouped_q() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    let q_all = vec![0.0f32; 7];
    let k_all = vec![0.0f32; 4];
    let v_all = vec![0.0f32; 4];
    let mut out = vec![0.0f32; 8];

    let err = vk
        .attention_decode_window_grouped_for_layer(
            0, &q_all, &k_all, &v_all, 4, 2, 2, 1, 0, &mut out,
        )
        .expect_err("partially short grouped q should return Err");
    assert!(err.contains("shape"), "unexpected error: {err}");
}

#[test]
fn test_vulkan_runtime_counters_track_attention_window_activity() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_attention_decode_window()
        .expect("attention window self-test should pass");
    let stats_after_attn = vk.runtime_counters();
    assert!(stats_after_attn.submits > 0, "expected submits > 0");
    assert!(
        stats_after_attn.upload_bytes > 0,
        "expected upload_bytes > 0"
    );
    assert!(
        stats_after_attn.download_bytes > 0,
        "expected download_bytes > 0"
    );

    vk.self_test_ffn_chain_window()
        .expect("ffn window self-test should pass");
    let stats_after_ffn = vk.runtime_counters();
    assert!(
        stats_after_ffn.submits > stats_after_attn.submits,
        "expected FFN to increase submit count"
    );
    assert!(
        stats_after_ffn.upload_bytes > stats_after_attn.upload_bytes,
        "expected FFN to increase upload bytes"
    );
    assert!(
        stats_after_ffn.download_bytes > stats_after_attn.download_bytes,
        "expected FFN to increase download bytes"
    );
}

#[test]
fn test_gemv_window_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_gemv_window()
        .expect("gemv window self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 256);
    assert_eq!(stats.download_bytes, 32);
}

#[test]
fn test_qkv_window_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_qkv_window()
        .expect("qkv window self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 256);
    assert_eq!(stats.download_bytes, 96);
}

#[test]
fn test_attention_decode_window_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_attention_decode_window()
        .expect("attention window self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 96);
    assert_eq!(stats.download_bytes, 32);
}

#[test]
fn test_attention_decode_window_grouped_runtime_counters_are_single_submit() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    let q_all = vec![1.0f32, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
    let k_all = vec![1.0f32, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.25];
    let v_all = vec![0.5f32, 0.0, 1.5, 0.0, 0.0, 0.5, 0.0, 1.5];
    let mut out_all = vec![0.0f32; 8];
    vk.attention_decode_window_grouped_for_layer(
        0,
        &q_all,
        &k_all,
        &v_all,
        2,
        2,
        2,
        2,
        0,
        &mut out_all,
    )
    .expect("grouped attention window should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 96);
    assert_eq!(stats.download_bytes, 32);
}

#[test]
fn test_rms_norm_window_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_rms_norm_window()
        .expect("rms norm window self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 48);
    assert_eq!(stats.download_bytes, 32);
}

#[test]
fn test_ffn_window_runtime_counters_are_exact() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.reset_runtime_counters();
    vk.self_test_ffn_chain_window()
        .expect("ffn window self-test should pass");
    let stats = vk.runtime_counters();
    assert_eq!(stats.submits, 1);
    assert_eq!(stats.upload_bytes, 384);
    assert_eq!(stats.download_bytes, 256);
}

/// mv24 Task 8c-impl-8: NEOX RoPE in-place rotation GPU vs CPU reference.
///
/// Parameters: head_dim=8, num_heads=2, seq_len=2, pos_offset=0, base_freq=10000.0.
/// Expected: GPU result matches CPU NEOX rotation formula within 1e-3.
#[test]
fn test_rope_apply_neox_matches_cpu_reference() {
    let (_guard, mut vk) = match new_vulkan_layer(64, 64, 64, GpuWeightMode::Soa) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    vk.self_test_rope_apply()
        .expect("rope apply self-test should pass: GPU NEOX RoPE must match CPU reference");
}

/// Dump emit_rope_apply SPIR-V to /tmp for external validation.
#[test]
fn test_dump_rope_apply_spirv() {
    let spirv = emit_rope_apply(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/rope_apply.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/rope_apply.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_q4k_q8k_gemv_shader_generates() {
    let spirv = emit_q4k_q8k_gemv(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(
        spirv.len() > 200,
        "Q4_K×Q8K shader too short: {} words",
        spirv.len()
    );
}

#[test]
fn test_dump_q4k_q8k_spirv() {
    let spirv = emit_q4k_q8k_gemv(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q4k_q8k_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q4k_q8k_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_q6k_q8k_gemv_shader_generates() {
    let spirv = emit_q6k_q8k_gemv(64);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(
        spirv.len() > 200,
        "Q6_K×Q8K shader too short: {} words",
        spirv.len()
    );
}

#[test]
fn test_dump_q6k_q8k_spirv() {
    let spirv = emit_q6k_q8k_gemv(64);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/q6k_q8k_gemv.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/q6k_q8k_gemv.spv",
        spirv.len(),
        bytes.len()
    );
}

#[test]
fn test_quantize_to_q8k_shader_generates() {
    let spirv = emit_quantize_to_q8k(1);
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(
        spirv.len() > 100,
        "quantize_to_q8k shader too short: {} words",
        spirv.len()
    );
}

#[test]
fn test_dump_quantize_to_q8k_spirv() {
    let spirv = emit_quantize_to_q8k(1);
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write("/tmp/quantize_to_q8k.spv", &bytes).unwrap();
    eprintln!(
        "Dumped {} words ({} bytes) to /tmp/quantize_to_q8k.spv",
        spirv.len(),
        bytes.len()
    );
}
