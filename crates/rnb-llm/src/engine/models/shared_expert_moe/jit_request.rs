use crate::engine::moe_jit::{
    moe_jit_backend_from_env_cached, moe_jit_loader_registered, request_moe_jit_load,
    MoeJitByteRange, MoeJitExpertLoad, MoeJitLoadRequest,
};
use crate::runtime::scheduler::plan_moe_jit_load_order;
use crate::runtime::MoeRouteSlot;

#[inline]
pub(in crate::engine) fn qwen35_moe_jit_load_requested() -> bool {
    moe_jit_backend_from_env_cached().is_some() || moe_jit_loader_registered()
}

fn qwen35_moe_prefill_jit_top_experts_limit() -> usize {
    std::env::var("RNB_MOE_JIT_PREFILL_TOP_EXPERTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(8)
}

pub(in crate::engine) fn request_qwen35_moe_jit_load(
    layer_idx: usize,
    selected: &[usize],
    probs: &[f32],
    gate_exps_bytes: &[u8],
    up_exps_bytes: &[u8],
    down_exps_bytes: &[u8],
    gate_bytes_per_expert: usize,
    up_bytes_per_expert: usize,
    down_bytes_per_expert: usize,
) {
    let backend_hint = moe_jit_backend_from_env_cached();
    if backend_hint.is_none() && !moe_jit_loader_registered() {
        return;
    }
    request_qwen35_moe_jit_load_with_scores(
        backend_hint,
        layer_idx,
        selected,
        probs,
        gate_exps_bytes,
        up_exps_bytes,
        down_exps_bytes,
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
    );
}

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn request_qwen35_moe_jit_load_from_route_slots(
    layer_idx: usize,
    sparse_slots: &[MoeRouteSlot],
    n_expert: usize,
    gate_exps_bytes: &[u8],
    up_exps_bytes: &[u8],
    down_exps_bytes: &[u8],
    gate_bytes_per_expert: usize,
    up_bytes_per_expert: usize,
    down_bytes_per_expert: usize,
) {
    let backend_hint = moe_jit_backend_from_env_cached();
    if backend_hint.is_none() && !moe_jit_loader_registered() {
        return;
    }
    if sparse_slots.is_empty() || n_expert == 0 {
        return;
    }

    let mut scores = vec![0.0f32; n_expert];
    let mut seen = vec![false; n_expert];
    let mut selected = Vec::new();
    for slot in sparse_slots {
        assert!(
            slot.expert < n_expert,
            "qwen35moe prefill route expert {} out of bounds {}",
            slot.expert,
            n_expert
        );
        if !seen[slot.expert] {
            seen[slot.expert] = true;
            selected.push(slot.expert);
        }
        if slot.weight.is_finite() {
            scores[slot.expert] += slot.weight.max(0.0);
        }
    }
    let limit = qwen35_moe_prefill_jit_top_experts_limit();
    if limit == 0 {
        return;
    }
    if selected.len() > limit {
        selected = plan_moe_jit_load_order(&selected, &scores);
        selected.truncate(limit);
    }
    request_qwen35_moe_jit_load_with_scores(
        backend_hint,
        layer_idx,
        &selected,
        &scores,
        gate_exps_bytes,
        up_exps_bytes,
        down_exps_bytes,
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
    );
}

#[allow(clippy::too_many_arguments)]
fn request_qwen35_moe_jit_load_with_scores(
    backend_hint: Option<crate::runtime::BackendKind>,
    layer_idx: usize,
    selected: &[usize],
    probs: &[f32],
    gate_exps_bytes: &[u8],
    up_exps_bytes: &[u8],
    down_exps_bytes: &[u8],
    gate_bytes_per_expert: usize,
    up_bytes_per_expert: usize,
    down_bytes_per_expert: usize,
) {
    if selected.is_empty() {
        return;
    }
    let experts = plan_moe_jit_load_order(selected, probs);
    let expert_loads = experts
        .iter()
        .map(|&expert| MoeJitExpertLoad {
            expert,
            gate: MoeJitByteRange::from_tensor_slice(
                gate_exps_bytes,
                expert.saturating_mul(gate_bytes_per_expert),
                gate_bytes_per_expert,
            ),
            up: MoeJitByteRange::from_tensor_slice(
                up_exps_bytes,
                expert.saturating_mul(up_bytes_per_expert),
                up_bytes_per_expert,
            ),
            down: MoeJitByteRange::from_tensor_slice(
                down_exps_bytes,
                expert.saturating_mul(down_bytes_per_expert),
                down_bytes_per_expert,
            ),
        })
        .collect();
    request_moe_jit_load(&MoeJitLoadRequest {
        backend_hint,
        layer_idx,
        experts,
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
        expert_loads,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::moe_jit::{set_moe_jit_loader_for_test, MoeJitLoadSink};
    use crate::runtime::MoeRouteSlot;
    use std::sync::{Arc, Mutex};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn prefill_route_slots_notify_jit_loader_with_aggregated_expert_order() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        unsafe {
            std::env::remove_var("RNB_MOE_JIT_PREFILL_TOP_EXPERTS");
        }
        let gate = vec![0u8; 4 * 10];
        let up = vec![0u8; 4 * 20];
        let down = vec![0u8; 4 * 30];
        let slots = [
            MoeRouteSlot::new(1, 0, 0.25),
            MoeRouteSlot::new(3, 0, 0.70),
            MoeRouteSlot::new(1, 1, 0.50),
            MoeRouteSlot::new(2, 1, 0.60),
        ];
        let captured = Arc::new(Mutex::new(Vec::new()));
        set_moe_jit_loader_for_test(Some(Arc::new(TestJitSink {
            captured: captured.clone(),
        })));

        request_qwen35_moe_jit_load_from_route_slots(11, &slots, 4, &gate, &up, &down, 10, 20, 30);
        set_moe_jit_loader_for_test(None);

        let got = captured.lock().expect("capture lock").clone();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].layer_idx, 11);
        assert_eq!(got[0].experts, vec![1, 3, 2]);
        assert_eq!(got[0].expert_loads.len(), 3);
        assert_eq!(got[0].expert_loads[0].expert, 1);
        assert_eq!(got[0].expert_loads[0].gate.tensor_offset, 10);
        assert_eq!(got[0].expert_loads[0].up.tensor_offset, 20);
        assert_eq!(got[0].expert_loads[0].down.tensor_offset, 30);
        assert_eq!(got[0].expert_loads[1].expert, 3);
        assert_eq!(got[0].expert_loads[1].gate.tensor_offset, 30);
        assert_eq!(got[0].expert_loads[2].expert, 2);
    }

    #[test]
    fn prefill_route_slots_respect_top_expert_cap() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        unsafe {
            std::env::set_var("RNB_MOE_JIT_PREFILL_TOP_EXPERTS", "2");
        }
        let gate = vec![0u8; 4 * 10];
        let up = vec![0u8; 4 * 20];
        let down = vec![0u8; 4 * 30];
        let slots = [
            MoeRouteSlot::new(1, 0, 0.25),
            MoeRouteSlot::new(3, 0, 0.70),
            MoeRouteSlot::new(1, 1, 0.50),
            MoeRouteSlot::new(2, 1, 0.60),
        ];
        let captured = Arc::new(Mutex::new(Vec::new()));
        set_moe_jit_loader_for_test(Some(Arc::new(TestJitSink {
            captured: captured.clone(),
        })));

        request_qwen35_moe_jit_load_from_route_slots(12, &slots, 4, &gate, &up, &down, 10, 20, 30);
        set_moe_jit_loader_for_test(None);
        unsafe {
            std::env::remove_var("RNB_MOE_JIT_PREFILL_TOP_EXPERTS");
        }

        let got = captured.lock().expect("capture lock").clone();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].layer_idx, 12);
        assert_eq!(got[0].experts, vec![1, 3]);
        assert_eq!(got[0].expert_loads.len(), 2);
    }

    #[derive(Clone)]
    struct TestJitSink {
        captured: Arc<Mutex<Vec<MoeJitLoadRequest>>>,
    }

    impl MoeJitLoadSink for TestJitSink {
        fn request_load(&self, request: &MoeJitLoadRequest) {
            self.captured
                .lock()
                .expect("capture lock")
                .push(request.clone());
        }
    }
}
