use super::*;
use crate::engine::moe_jit::{set_moe_jit_loader_for_test, MoeJitLoadRequest, MoeJitLoadSink};

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
    };

    let h = vec![1.0f32; n_embd];
    let mut out = vec![f32::NAN; n_embd];
    view.forward(&h, &mut out);

    assert!(out.iter().all(|v| v.is_finite()));
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
