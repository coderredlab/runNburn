use super::*;
use crate::engine::moe_jit::{set_moe_jit_loader_for_test, MoeJitLoadRequest, MoeJitLoadSink};
use crate::engine::moe_section_dispatch::{down_q5k_unit_size, gate_up_unit_size};
use crate::engine::moe_section_layout::{moe_section_gate_up_layout, MoeSectionGateUpLayout};
use crate::engine::moe_shadow_dispatch::pack_q2k_gate_up_tile;

pub(in crate::engine) fn env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[test]
fn per_expert_byte_counts_match_gemma4_26b_a4b() {
    let view = MoeLayerView {
        router_w: &[],
        gate_up_bytes: &[],
        down_bytes: &[],
        down_scale: &[],
        down_quant: GGMLType::Q5_1,
        n_embd: 2816,
        n_ff: 704,
        n_expert: 128,
        n_expert_used: 8,
        layer_idx: None,
        rank_to_original: None,
        shadow_gate_up_bytes: None,
        gate_up_residency: None,
        down_residency: None,
    };
    assert_eq!(view.per_expert_gate_up_bytes(), 2_230_272);
    assert_eq!(view.per_expert_down_bytes(), 1_486_848);
}

#[test]
fn gemma_moe_down_byte_count_uses_tensor_quant_type() {
    assert_eq!(gemma_down_bytes_per_row(704, GGMLType::Q5_0), 484);
    assert_eq!(gemma_down_bytes_per_row(704, GGMLType::Q8_0), 748);
    assert_eq!(
        per_expert_down_bytes(2816, 704, GGMLType::Q5_0),
        174_456_832 / 128
    );
    assert_eq!(
        per_expert_down_bytes(2816, 704, GGMLType::Q8_0),
        269_615_104 / 128
    );
}

#[test]
fn zero_weights_produce_zero_output() {
    // n_embd must be Q4_K-compatible (divisible by 256) and Q5_1-compatible (divisible by 32).
    // n_ff must be Q5_1-compatible.
    let n_embd = 256;
    let n_ff = 64;
    let n_expert = 2;
    let n_expert_used = 2;

    let router_w = vec![0f32; n_expert * n_embd];
    let gate_up_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = q5_1_bytes_per_row(n_ff);
    let per_gu = n_ff * 2 * gate_up_bpr;
    let per_dn = n_embd * down_bpr;
    let gate_up_bytes = vec![0u8; n_expert * per_gu];
    let down_bytes = vec![0u8; n_expert * per_dn];
    let down_scale = vec![1.0f32; n_expert];

    let view = MoeLayerView {
        router_w: &router_w,
        gate_up_bytes: &gate_up_bytes,
        down_bytes: &down_bytes,
        down_scale: &down_scale,
        down_quant: GGMLType::Q5_1,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: None,
        rank_to_original: None,
        shadow_gate_up_bytes: None,
        gate_up_residency: None,
        down_residency: None,
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![f32::NAN; n_embd];
    view.forward(&h, &mut out);

    // All-zero weights → GEMV outputs zero (Q4_K/Q5_1 block with d=0, dmin=0, m=0).
    // gate=0, up=0, mid=gelu(0)*0=0, down GEMV = 0, weighted sum = 0.
    for v in out.iter() {
        assert!(v.abs() < 1e-6, "expected 0, got {}", v);
    }
}

#[test]
fn qwen35_moe_profile_records_high_path_summary() {
    let _profile_guard = crate::engine::moe_profile::test_lock()
        .lock()
        .expect("moe profile test lock poisoned");
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MOE_PROFILE", "1");
        std::env::remove_var("RNB_HOBBIT");
    }
    reset_moe_profile();

    let n_embd = 256;
    let n_ff = 256;
    let n_expert = 2;
    let n_expert_used = 2;

    let router_w = vec![0.0f32; n_expert * n_embd];
    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(n_ff, GGMLType::Q5_K);
    let sh_gate_bpr = q8_0_bytes_per_row(n_embd);
    let sh_down_bpr = q8_0_bytes_per_row(n_ff);

    let gate_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_up_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_down_bytes = vec![0u8; n_embd * sh_down_bpr];

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q5_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q8_0,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q8_0,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q8_0,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: Some(0),
        shadow_gate_bytes: None,
        shadow_up_bytes: None,
        shadow_gate_up_tile_bytes: None,
        shadow_down_bytes: None,
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![0.0f32; n_embd];
    view.forward(&h, &mut out);

    let report = moe_profile_report().expect("report should exist");
    assert!(report.contains("qwen35moe:decode:router"));
    assert!(report.contains("qwen35moe:decode:routing"));
    assert!(report.contains("qwen35moe:decode:high_compute"));
    assert!(report.contains("qwen35moe:decode:shared_expert"));
    assert!(report.contains("qwen35moe:decode counts high=2 low=0 skip=0"));

    unsafe {
        std::env::remove_var("RNB_MOE_PROFILE");
    }
    reset_moe_profile();
}

#[test]
fn qwen35_shared_expert_uses_tensor_quant_types() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MOE_PROFILE");
        std::env::remove_var("RNB_HOBBIT");
    }

    let n_embd = 256;
    let n_ff = 256;
    let n_expert = 1;
    let n_expert_used = 1;

    let router_w = vec![0.0f32; n_expert * n_embd];
    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(n_ff, GGMLType::Q6_K);
    let shared_gate_bpr = q4k_bytes_per_row(n_embd);
    let shared_down_bpr = q6k_bytes_per_row(n_ff);

    let gate_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; n_ff * shared_gate_bpr];
    let shared_up_bytes = vec![0u8; n_ff * shared_gate_bpr];
    let shared_down_bytes = vec![0u8; n_embd * shared_down_bpr];

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q6_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q4_K,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q4_K,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q6_K,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: None,
        shadow_gate_bytes: None,
        shadow_up_bytes: None,
        shadow_gate_up_tile_bytes: None,
        shadow_down_bytes: None,
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![f32::NAN; n_embd];
    view.forward(&h, &mut out);

    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn qwen35_moe_profile_records_low_fallback_summary() {
    let _profile_guard = crate::engine::moe_profile::test_lock()
        .lock()
        .expect("moe profile test lock poisoned");
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MOE_PROFILE", "1");
        std::env::set_var("RNB_HOBBIT", "1");
        std::env::set_var("RNB_HOBBIT_T1", "0.0");
        std::env::set_var("RNB_HOBBIT_T2", "1.0");
        std::env::set_var("RNB_HOBBIT_LOW_PATH", "auto");
    }
    reset_moe_profile();

    let n_embd = 256;
    let n_ff = 256;
    let n_expert = 2;
    let n_expert_used = 2;

    let mut router_w = vec![0.0f32; n_expert * n_embd];
    for i in 0..n_embd {
        router_w[i] = 1.0;
    }
    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(n_ff, GGMLType::Q5_K);
    let q2k_bpr = q2k_bytes_per_row(n_embd);
    let sh_gate_bpr = q8_0_bytes_per_row(n_embd);
    let sh_down_bpr = q8_0_bytes_per_row(n_ff);

    let gate_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shadow_gate_bytes = vec![0u8; n_expert * n_ff * q2k_bpr];
    let shadow_up_bytes = vec![0u8; n_expert * n_ff * q2k_bpr];
    let shadow_gate_up_tile_bytes = pack_q2k_gate_up_tile(
        &shadow_gate_bytes[..n_ff * q2k_bpr],
        &shadow_up_bytes[..n_ff * q2k_bpr],
        n_ff,
        n_embd,
    )
    .repeat(n_expert);
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_up_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_down_bytes = vec![0u8; n_embd * sh_down_bpr];

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q5_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q8_0,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q8_0,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q8_0,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: Some(0),
        shadow_gate_bytes: Some(&shadow_gate_bytes),
        shadow_up_bytes: Some(&shadow_up_bytes),
        shadow_gate_up_tile_bytes: Some(&shadow_gate_up_tile_bytes),
        shadow_down_bytes: None,
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![0.0f32; n_embd];
    view.forward(&h, &mut out);

    let report = moe_profile_report().expect("report should exist");
    assert!(report.contains("qwen35moe:decode:low_compute"));
    assert!(report.contains("qwen35moe:decode:low_gate_up_compute"));
    assert!(report.contains("qwen35moe:decode:low_gate_up_row_compute"));
    assert!(!report.contains("qwen35moe:decode:low_gate_up_tile_compute"));
    assert!(report.contains("qwen35moe:decode:low_base_down_compute"));
    assert!(report.contains("qwen35moe:decode counts high=1 low=1 skip=0"));

    unsafe {
        std::env::remove_var("RNB_MOE_PROFILE");
        std::env::remove_var("RNB_HOBBIT");
        std::env::remove_var("RNB_HOBBIT_T1");
        std::env::remove_var("RNB_HOBBIT_T2");
        std::env::remove_var("RNB_HOBBIT_LOW_PATH");
    }
    reset_moe_profile();
}

#[test]
fn qwen35_moe_profile_records_low_shadow_down_summary() {
    let _profile_guard = crate::engine::moe_profile::test_lock()
        .lock()
        .expect("moe profile test lock poisoned");
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MOE_PROFILE", "1");
        std::env::set_var("RNB_HOBBIT", "1");
        std::env::set_var("RNB_HOBBIT_T1", "-1.0");
        std::env::set_var("RNB_HOBBIT_T2", "1.0");
        std::env::set_var("RNB_HOBBIT_LOW_PATH", "auto");
    }
    reset_moe_profile();

    let n_embd = 256;
    let n_ff = 256;
    let n_expert = 2;
    let n_expert_used = 2;

    let mut router_w = vec![0.0f32; n_expert * n_embd];
    for i in 0..n_embd {
        router_w[i] = 1.0;
    }
    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(n_ff, GGMLType::Q5_K);
    let q2k_gu_bpr = q2k_bytes_per_row(n_embd);
    let q2k_dn_bpr = q2k_bytes_per_row(n_ff);
    let sh_gate_bpr = q8_0_bytes_per_row(n_embd);
    let sh_down_bpr = q8_0_bytes_per_row(n_ff);

    let gate_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shadow_gate_bytes = vec![0u8; n_expert * n_ff * q2k_gu_bpr];
    let shadow_up_bytes = vec![0u8; n_expert * n_ff * q2k_gu_bpr];
    let shadow_gate_up_tile_bytes = pack_q2k_gate_up_tile(
        &shadow_gate_bytes[..n_ff * q2k_gu_bpr],
        &shadow_up_bytes[..n_ff * q2k_gu_bpr],
        n_ff,
        n_embd,
    )
    .repeat(n_expert);
    let shadow_down_bytes = vec![0u8; n_expert * n_embd * q2k_dn_bpr];
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_up_bytes = vec![0u8; n_ff * sh_gate_bpr];
    let shared_down_bytes = vec![0u8; n_embd * sh_down_bpr];

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q5_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q8_0,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q8_0,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q8_0,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: Some(0),
        shadow_gate_bytes: Some(&shadow_gate_bytes),
        shadow_up_bytes: Some(&shadow_up_bytes),
        shadow_gate_up_tile_bytes: Some(&shadow_gate_up_tile_bytes),
        shadow_down_bytes: Some(&shadow_down_bytes),
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![0.0f32; n_embd];
    view.forward(&h, &mut out);

    let report = moe_profile_report().expect("report should exist");
    assert!(report.contains("qwen35moe:decode:low_compute"));
    assert!(report.contains("qwen35moe:decode:low_gate_up_compute"));
    assert!(report.contains("qwen35moe:decode:low_gate_up_tile_compute"));
    assert!(!report.contains("qwen35moe:decode:low_gate_up_row_compute"));
    assert!(report.contains("qwen35moe:decode:low_shadow_down_compute"));
    assert!(!report.contains("qwen35moe:decode:low_base_down_compute"));
    assert!(report.contains("qwen35moe:decode counts high=0 low=2 skip=0"));

    unsafe {
        std::env::remove_var("RNB_MOE_PROFILE");
        std::env::remove_var("RNB_HOBBIT");
        std::env::remove_var("RNB_HOBBIT_T1");
        std::env::remove_var("RNB_HOBBIT_T2");
        std::env::remove_var("RNB_HOBBIT_LOW_PATH");
    }
    reset_moe_profile();
}

/// Session 79 Phase 1 Task 13: synthetic zero-weight smoke for the moe_section
/// sdot dispatch. Builds a tiny `MoeSectionDecodeLayer` (1 expert, no shared
/// expert, all-zero file bytes) and confirms `forward` can select the moe_section
/// path without panicking and writes zeros
/// (the integer dot of all-zero weights × any activation = 0).
///
/// We intentionally allow x86 to compile the test (no `cfg` gate around
/// the test fn body) — on x86 the dispatch falls through to the legacy
/// path because `forward_moe_section_sdot` is `#[cfg(target_arch = "aarch64")]`,
/// and the legacy path also produces zeros for all-zero weights. The
/// test thus covers both paths' "zero ⇒ zero" invariant.
#[test]
fn forward_moe_section_sdot_zero_weights_produce_zero_output() {
    use crate::engine::{
        MoeSectionDecodeLayer, MoeSectionExpert, MoeSectionRowDown, MoeSectionRowGU,
    };
    use std::sync::Arc;

    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::set_var("RNB_MOE_DECODE", "1");
        std::env::remove_var("RNB_HOBBIT");
    }

    let n_embd = 256usize;
    let d_ff = 256usize;
    let n_expert = 1usize;
    let n_expert_used = 1usize;
    let n_embd_blocks = n_embd / 256;
    let d_ff_blocks = d_ff / 256;
    let gu_pair_size = gate_up_unit_size(MoeSectionGateUpLayout::Q4KPair);
    let q5k_size = down_q5k_unit_size();

    // Single contiguous buffer: gate_up rows then down rows for the lone
    // expert. Each row is exactly one block long for this geometry.
    let gate_up_row_bytes = n_embd_blocks * gu_pair_size;
    let down_row_bytes = d_ff_blocks * q5k_size;
    let total_bytes = d_ff * gate_up_row_bytes + n_embd * down_row_bytes;
    let tmp_file = tempfile::tempfile().expect("tmpfile for MoE section bytes");
    tmp_file
        .set_len(total_bytes as u64)
        .expect("size tmpfile for MoE section bytes");
    let file_bytes: Arc<memmap2::Mmap> = Arc::new(unsafe {
        memmap2::MmapOptions::new()
            .len(total_bytes)
            .map(&tmp_file)
            .expect("mmap tmpfile for MoE section bytes")
    });

    let mut gate_up_rows = Vec::with_capacity(d_ff);
    for r in 0..d_ff {
        gate_up_rows.push(MoeSectionRowGU {
            gate_mul: 1.0,
            up_mul: 1.0,
            blocks_offset: r * gate_up_row_bytes,
            blocks_len: gate_up_row_bytes,
            scale_offset: None,
            scale_len: 0,
        });
    }
    let down_base = d_ff * gate_up_row_bytes;
    let mut down_rows = Vec::with_capacity(n_embd);
    for r in 0..n_embd {
        down_rows.push(MoeSectionRowDown {
            down_mul: 1.0,
            blocks_offset: down_base + r * down_row_bytes,
            blocks_len: down_row_bytes,
        });
    }

    let moe_section = MoeSectionDecodeLayer {
        file_bytes,
        n_experts: n_expert as u32,
        d_ff: d_ff as u32,
        n_embd: n_embd as u32,
        gate_up_quant: 0,
        down_quant: 0,
        shared_quant: 0xFF, // SHARED_QUANT_NONE
        experts: vec![MoeSectionExpert {
            gate_up_rows,
            down_rows,
        }],
        shared_expert: None,
    };

    // Legacy slots are zero/empty — the MoE section path won't read them, but the
    // x86 fallback path needs valid backing memory.
    let router_w = vec![0.0f32; n_expert * n_embd];
    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(d_ff, GGMLType::Q5_K);
    let sh_gate_bpr = q8_0_bytes_per_row(n_embd);
    let sh_down_bpr = q8_0_bytes_per_row(d_ff);
    let gate_exps_bytes = vec![0u8; n_expert * d_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * d_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; d_ff * sh_gate_bpr];
    let shared_up_bytes = vec![0u8; d_ff * sh_gate_bpr];
    let shared_down_bytes = vec![0u8; n_embd * sh_down_bpr];

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q5_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q8_0,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q8_0,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q8_0,
        n_embd,
        n_ff: d_ff,
        n_expert,
        n_expert_used,
        layer_idx: Some(0),
        shadow_gate_bytes: None,
        shadow_up_bytes: None,
        shadow_gate_up_tile_bytes: None,
        shadow_down_bytes: None,
        moe_section_decode: Some(&moe_section),
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let h = vec![0.5f32; n_embd];
    let mut out = vec![f32::NAN; n_embd];
    view.forward(&h, &mut out);

    for (i, v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "expected zero at index {} (got {})", i, v);
    }

    unsafe {
        std::env::remove_var("RNB_MOE_DECODE");
    }
}

#[test]
fn moe_section_gate_up_unit_size_accepts_pair_and_unpacked_scale_tags() {
    use rnb_loader::rnb_moe_reader::{
        GATE_UP_QUANT_Q4K_PAIR, GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE,
        GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES,
    };

    assert_eq!(
        moe_section_gate_up_unit_size(GATE_UP_QUANT_Q4K_PAIR),
        Some(gate_up_unit_size(MoeSectionGateUpLayout::Q4KPair))
    );
    assert_eq!(
        moe_section_gate_up_unit_size(GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES),
        Some(gate_up_unit_size(MoeSectionGateUpLayout::UnpackedScales))
    );
    assert_eq!(
        moe_section_gate_up_unit_size(GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE),
        Some(gate_up_unit_size(MoeSectionGateUpLayout::ScalePlane))
    );
    assert_eq!(moe_section_gate_up_unit_size(0xFF), None);
}

#[test]
fn moe_section_gate_up_layout_classifies_tags_once() {
    use rnb_loader::rnb_moe_reader::{
        GATE_UP_QUANT_Q4K_PAIR, GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE,
        GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES,
    };

    let q4k_pair = moe_section_gate_up_layout(GATE_UP_QUANT_Q4K_PAIR).unwrap();
    assert_eq!(
        q4k_pair.unit_size(),
        gate_up_unit_size(MoeSectionGateUpLayout::Q4KPair)
    );
    assert!(!q4k_pair.uses_scale_plane());

    let unpacked = moe_section_gate_up_layout(GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES).unwrap();
    assert_eq!(
        unpacked.unit_size(),
        gate_up_unit_size(MoeSectionGateUpLayout::UnpackedScales)
    );
    assert!(!unpacked.uses_scale_plane());

    let scale_plane = moe_section_gate_up_layout(GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE).unwrap();
    assert_eq!(
        scale_plane.unit_size(),
        gate_up_unit_size(MoeSectionGateUpLayout::ScalePlane)
    );
    assert!(scale_plane.uses_scale_plane());

    assert!(moe_section_gate_up_layout(0xFF).is_none());
}

#[test]
fn qwen35_route_notifies_jit_loader_after_router_selection() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    unsafe {
        std::env::remove_var("RNB_MOE_PROFILE");
        std::env::remove_var("RNB_HOBBIT");
    }

    let n_embd = 256;
    let n_ff = 256;
    let n_expert = 4;
    let n_expert_used = 2;
    let mut router_w = vec![0.0f32; n_expert * n_embd];
    router_w[2 * n_embd] = 4.0;
    router_w[n_embd] = 3.0;
    router_w[3 * n_embd] = 2.0;
    router_w[0] = 1.0;

    let gate_bpr = q4k_bytes_per_row(n_embd);
    let down_bpr = down_bytes_per_row(n_ff, GGMLType::Q5_K);
    let shared_bpr = q8_0_bytes_per_row(n_embd);
    let shared_down_bpr = q8_0_bytes_per_row(n_ff);
    let gate_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let up_exps_bytes = vec![0u8; n_expert * n_ff * gate_bpr];
    let down_exps_bytes = vec![0u8; n_expert * n_embd * down_bpr];
    let shared_input_scale = vec![0.0f32; n_embd];
    let shared_gate_bytes = vec![0u8; n_ff * shared_bpr];
    let shared_up_bytes = vec![0u8; n_ff * shared_bpr];
    let shared_down_bytes = vec![0u8; n_embd * shared_down_bpr];

    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    set_moe_jit_loader_for_test(Some(std::sync::Arc::new(TestJitSink {
        captured: captured.clone(),
    })));

    let view = SharedExpertMoEView {
        router_selection_bias: None,
        expert_gating_func: 0,
        expert_weights_norm: false,
        expert_weights_scale: 1.0,
        shared_expert_gated: true,
        router_w: &router_w,
        gate_exps_bytes: &gate_exps_bytes,
        gate_quant: GGMLType::Q4_K,
        up_exps_bytes: &up_exps_bytes,
        up_quant: GGMLType::Q4_K,
        down_exps_bytes: &down_exps_bytes,
        down_quant: GGMLType::Q5_K,
        shared_input_scale: &shared_input_scale,
        shared_gate_bytes: &shared_gate_bytes,
        shared_gate_quant: GGMLType::Q8_0,
        shared_up_bytes: &shared_up_bytes,
        shared_up_quant: GGMLType::Q8_0,
        shared_down_bytes: &shared_down_bytes,
        shared_down_quant: GGMLType::Q8_0,
        n_embd,
        n_ff,
        n_expert,
        n_expert_used,
        layer_idx: Some(7),
        shadow_gate_bytes: None,
        shadow_up_bytes: None,
        shadow_gate_up_tile_bytes: None,
        shadow_down_bytes: None,
        moe_section_decode: None,
        gate_residency: None,
        up_residency: None,
        down_residency: None,
    };

    let mut h = vec![0.0f32; n_embd];
    h[0] = 1.0;
    let mut out = vec![0.0f32; n_embd];
    view.forward(&h, &mut out);
    set_moe_jit_loader_for_test(None);

    let got = captured.lock().expect("capture lock").clone();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].layer_idx, 7);
    assert_eq!(got[0].experts, vec![2, 1]);
    assert_eq!(got[0].gate_bytes_per_expert, n_ff * gate_bpr);
    assert_eq!(got[0].up_bytes_per_expert, n_ff * gate_bpr);
    assert_eq!(got[0].down_bytes_per_expert, n_embd * down_bpr);
    assert_eq!(got[0].expert_loads.len(), 2);
    assert_eq!(got[0].expert_loads[0].expert, 2);
    assert_eq!(
        got[0].expert_loads[0].gate.tensor_offset,
        2 * n_ff * gate_bpr
    );
    assert_eq!(got[0].expert_loads[0].up.tensor_offset, 2 * n_ff * gate_bpr);
    assert_eq!(
        got[0].expert_loads[0].down.tensor_offset,
        2 * n_embd * down_bpr
    );
    assert_eq!(got[0].expert_loads[1].expert, 1);
    assert_eq!(got[0].expert_loads[1].gate.tensor_offset, n_ff * gate_bpr);
}

#[derive(Clone)]
struct TestJitSink {
    captured: std::sync::Arc<std::sync::Mutex<Vec<MoeJitLoadRequest>>>,
}

impl MoeJitLoadSink for TestJitSink {
    fn request_load(&self, request: &MoeJitLoadRequest) {
        self.captured
            .lock()
            .expect("capture lock")
            .push(request.clone());
    }
}
