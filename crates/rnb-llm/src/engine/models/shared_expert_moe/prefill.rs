//! Shared-expert MoE prefill helpers.

#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::policy;
use crate::engine::scalar_gemv::gemv_generic;
use crate::engine::*;
use crate::runtime::MoeRouteSlot;
use rayon::prelude::*;

use super::jit_request::{
    qwen35_moe_jit_load_requested, request_qwen35_moe_jit_load_from_route_slots,
};
use super::moe_types::{
    down_bytes_per_row, expert_bytes_per_row, sparse_expert_bytes, SharedExpertMoEView,
};
use super::routing::hy3_sigmoid_topk_route;

fn fill_qwen35_prefill_trace_selection(
    token_idx: usize,
    selected_count: usize,
    sparse_slots: &[MoeRouteSlot],
    selected: &mut [usize],
) -> usize {
    let Some(start) = token_idx.checked_mul(selected_count) else {
        return 0;
    };
    let Some(end) = start.checked_add(selected_count) else {
        return 0;
    };
    if selected_count == 0 || end > sparse_slots.len() || selected.len() < selected_count {
        return 0;
    }
    for (dst, slot) in selected.iter_mut().zip(&sparse_slots[start..end]) {
        *dst = slot.expert;
    }
    selected_count
}

fn fill_qwen35_prefill_trace_probs(logits: &[f32], probs: &mut [f32]) {
    if logits.is_empty() || probs.len() < logits.len() {
        return;
    }
    let probs = &mut probs[..logits.len()];
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for (dst, &logit) in probs.iter_mut().zip(logits.iter()) {
        *dst = (logit - max_l).exp();
    }
    let sum: f32 = probs.iter().sum();
    if sum != 0.0 {
        for value in probs.iter_mut() {
            *value /= sum;
        }
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn dump_task11_generic_moe_stage(layer_idx: usize, stage: &str, values: &[f32]) {
    if layer_idx > 3 {
        return;
    }
    let Some(dir) = crate::engine::policy::env_os_string("RNB_TASK11_MOE_STAGE_DUMP_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("create Task11 generic MoE stage dump directory");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    std::fs::write(
        std::path::Path::new(&dir).join(format!("L{layer_idx}_{stage}.bin")),
        bytes,
    )
    .expect("write Task11 generic MoE stage dump");
}

fn add_moe_output_to_hidden(
    mut moe_out: Vec<f32>,
    hidden: &Tensor,
    seq_len: usize,
    hidden_dim: usize,
) -> Tensor {
    let expected = seq_len
        .checked_mul(hidden_dim)
        .expect("shared-expert MoE output shape overflow");
    assert_eq!(
        moe_out.len(),
        expected,
        "shared-expert MoE output length mismatch"
    );
    let hidden_data = kernels::tensor_as_f32_slice(hidden);
    assert_eq!(
        hidden_data.len(),
        expected,
        "shared-expert MoE hidden length mismatch"
    );
    add_f32_inplace(&mut moe_out, hidden_data);
    Tensor::from_vec(moe_out, &[seq_len, hidden_dim])
}

pub(in crate::engine) fn qwen35_verify_tokens2_decode_equivalent_enabled(
    architecture: ModelArchitecture,
    seq_len: usize,
) -> bool {
    architecture == ModelArchitecture::Qwen35MoE
        && seq_len == 2
        && crate::engine::policy::env_string("RNB_QWEN35_VERIFY_TOKENS2_DECODE")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false)
}

fn trace_qwen35_prefill_moe_routes(
    layer_idx: usize,
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
    router_logits: &[f32],
    sparse_slots: &[MoeRouteSlot],
) {
    let route_trace = crate::engine::moe_trace::route_trace_is_active();
    let predictor_trace = crate::engine::moe_trace::predictor_trace_is_active();
    if !route_trace && !predictor_trace {
        return;
    }
    let selected_count = n_expert_used.min(n_expert);
    if selected_count == 0 {
        return;
    }
    let mut selected_stack = [0usize; 32];
    let mut selected_vec;
    let selected: &mut [usize] = if selected_count <= selected_stack.len() {
        &mut selected_stack[..selected_count]
    } else {
        selected_vec = vec![0usize; selected_count];
        &mut selected_vec
    };
    let mut probs = predictor_trace.then(|| vec![0.0f32; n_expert]);
    for token_idx in 0..seq_len {
        let count =
            fill_qwen35_prefill_trace_selection(token_idx, selected_count, sparse_slots, selected);
        if count == 0 {
            break;
        }
        let selected = &selected[..count];
        crate::engine::moe_trace::record_selection(layer_idx, selected);
        if let Some(probs) = probs.as_mut() {
            let logits_start = token_idx * n_expert;
            let logits_end = logits_start + n_expert;
            if logits_end > router_logits.len() {
                break;
            }
            fill_qwen35_prefill_trace_probs(&router_logits[logits_start..logits_end], probs);
            crate::engine::moe_trace::record_predictor(layer_idx, selected, probs);
        }
    }
}

fn qwen35_prefill_expert_major_slots_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_QWEN35_PREFILL_EXPERT_MAJOR_SLOTS")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(cfg!(all(feature = "metal", not(feature = "cuda"))))
}

const HY3_Q2Q3_BATCH_WIDTH: usize = 4;
const HY3_Q2Q3_MIN_BATCHES_PER_EXPERT: usize = 2;

fn hy3_q2q3_expert_major_prefill_enabled() -> bool {
    crate::engine::policy::env_string("RNB_HY3_Q2Q3_EXPERT_MAJOR_PREFILL")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn hy3_q2q3_expert_major_has_reuse(seq_len: usize, n_expert_used: usize, n_expert: usize) -> bool {
    let selected_count = n_expert_used.min(n_expert);
    if selected_count == 0 {
        return false;
    }
    let route_slots = seq_len.checked_mul(selected_count);
    let min_slots = n_expert.checked_mul(HY3_Q2Q3_BATCH_WIDTH * HY3_Q2Q3_MIN_BATCHES_PER_EXPERT);
    matches!((route_slots, min_slots), (Some(slots), Some(minimum)) if slots >= minimum)
}

fn hy3_q2q3_expert_major_chunk_tokens(view: &SharedExpertMoEView<'_>) -> usize {
    let selected_count = view.n_expert_used.min(view.n_expert);
    let rank_scratch_per_token = selected_count
        .saturating_mul(view.n_embd)
        .saturating_mul(std::mem::size_of::<f32>());
    let packed_sparse_layer_bytes = view
        .gate_exps_bytes
        .len()
        .saturating_add(view.up_exps_bytes.len())
        .saturating_add(view.down_exps_bytes.len());
    if rank_scratch_per_token == 0 {
        return 1;
    }
    (packed_sparse_layer_bytes / rank_scratch_per_token).max(1)
}

fn qwen35_prefill_device_topk_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_QWEN35_PREFILL_DEVICE_TOPK")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn qwen35_prefill_device_topk_combined_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_QWEN35_PREFILL_DEVICE_TOPK_COMBINED").as_deref()
        == Some("1")
}

fn qwen35_prefill_selected_base_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_QWEN35_PREFILL_SELECTED_BASE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenMoeSelectedBaseMetalFlow {
    Disabled,
    LlamaIdTokenMajor,
    V4TokenMajor,
    V3TokenMajor,
    GenericSelectedBase,
}

fn qwen_moe_selected_base_metal_flow(
    v4_requested: bool,
    v3_requested: bool,
    llama_id_requested: bool,
    use_selected_base: bool,
    trace_prefill_routes: bool,
    jit_load_requested: bool,
) -> QwenMoeSelectedBaseMetalFlow {
    if !use_selected_base || trace_prefill_routes || jit_load_requested {
        return QwenMoeSelectedBaseMetalFlow::Disabled;
    }
    if llama_id_requested {
        return QwenMoeSelectedBaseMetalFlow::LlamaIdTokenMajor;
    }
    if v4_requested {
        return QwenMoeSelectedBaseMetalFlow::V4TokenMajor;
    }
    if v3_requested {
        return QwenMoeSelectedBaseMetalFlow::V3TokenMajor;
    }
    QwenMoeSelectedBaseMetalFlow::GenericSelectedBase
}

fn qwen_moe_llama_id_product_tuple(
    gate_quant: GGMLType,
    up_quant: GGMLType,
    down_quant: GGMLType,
    shared_gate_quant: GGMLType,
    shared_up_quant: GGMLType,
    shared_down_quant: GGMLType,
) -> bool {
    let sparse = gate_quant == GGMLType::Q4_K
        && up_quant == GGMLType::Q4_K
        && matches!(down_quant, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K);
    let shared_q8 = shared_gate_quant == GGMLType::Q8_0
        && shared_up_quant == GGMLType::Q8_0
        && shared_down_quant == GGMLType::Q8_0;
    let shared_q4 = shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K
        && matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K);
    sparse && (shared_q8 || shared_q4)
}

fn qwen_moe_native_batch_product_tuple(
    gate_quant: GGMLType,
    up_quant: GGMLType,
    down_quant: GGMLType,
    shared_gate_quant: GGMLType,
    shared_up_quant: GGMLType,
    shared_down_quant: GGMLType,
    cuda_backend: bool,
) -> bool {
    let legacy_q4_shared = gate_quant == GGMLType::Q4_K
        && up_quant == GGMLType::Q4_K
        && matches!(down_quant, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K)
        && shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K;
    legacy_q4_shared
        || (cuda_backend
            && qwen_moe_llama_id_product_tuple(
                gate_quant,
                up_quant,
                down_quant,
                shared_gate_quant,
                shared_up_quant,
                shared_down_quant,
            ))
}

#[cfg(any(target_arch = "aarch64", test))]
fn qwen35_cpu_expert_major_should_run(
    cpu_expert_major_eligible: bool,
    llama_id_requested: bool,
    llama_id_product_tuple: bool,
) -> bool {
    cpu_expert_major_eligible && !(llama_id_requested && llama_id_product_tuple)
}

fn qwen_moe_prefill_outer_guard(
    legacy_policy_enabled: bool,
    legacy_q4_shared_tuple: bool,
    llama_id_requested: bool,
    llama_id_product_tuple: bool,
) -> bool {
    (legacy_policy_enabled && legacy_q4_shared_tuple)
        || (llama_id_requested && llama_id_product_tuple)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn qwen35moe_device_input_supported(
    moe_w: &SharedExpertMoELayerWeights,
    seq_len: usize,
) -> bool {
    let prefill_enabled = backend_runtime::qwen_moe_prefill_enabled_for_seq(seq_len);
    let supported = prefill_enabled
        && moe_w.gate_quant == GGMLType::Q4_K
        && moe_w.up_quant == GGMLType::Q4_K
        && matches!(
            moe_w.down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        && matches!(moe_w.shared_gate.ggml_type, GGMLType::Q4_K | GGMLType::Q8_0)
        && matches!(moe_w.shared_up.ggml_type, GGMLType::Q4_K | GGMLType::Q8_0)
        && matches!(
            moe_w.shared_down.ggml_type,
            GGMLType::F32 | GGMLType::Q4_K | GGMLType::Q6_K | GGMLType::Q8_0
        )
        && !crate::engine::moe_trace::route_trace_is_active()
        && !crate::engine::moe_trace::predictor_trace_is_active()
        && !crate::engine::moe_trace::is_active()
        && !qwen35_moe_jit_load_requested()
        && crate::engine::tuning_runtime::shared_f32_enabled();
    if crate::engine::policy::env_string("RNB_CUDA_DEVICE_PREFILL_TRACE").as_deref() == Some("1") {
        eprintln!(
            "[cuda:qwen-device-moe-support] supported={supported} seq={seq_len} prefill={prefill_enabled} gate={:?} up={:?} down={:?} shared_gate={:?} shared_up={:?} shared_down={:?} shared_f32={}",
            moe_w.gate_quant,
            moe_w.up_quant,
            moe_w.down_quant,
            moe_w.shared_gate.ggml_type,
            moe_w.shared_up.ggml_type,
            moe_w.shared_down.ggml_type,
            crate::engine::tuning_runtime::shared_f32_enabled(),
        );
    }
    supported
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn try_forward_ffn_qwen35moe_device_input_raw(
    moe_w: &SharedExpertMoELayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    normalized_id: cuda_runtime::DeviceTensorId,
    normalized_desc: cuda_runtime::DeviceTensorDesc,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
    reuse_residual_output: bool,
) -> crate::error::Result<Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>> {
    if !qwen35moe_device_input_supported(moe_w, seq_len) {
        return Ok(None);
    }

    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("qwen35moe router_f32 failed".into()))?;
    let gate_exps_bytes = moe_w.gate_exps_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("qwen35moe gate_exps_bytes failed".into())
    })?;
    let up_exps_bytes = moe_w
        .up_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("qwen35moe up_exps_bytes failed".into()))?;
    let down_exps_bytes = moe_w.down_exps_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("qwen35moe down_exps_bytes failed".into())
    })?;
    let shared_input_scale = kernels::tensor_as_f32_slice(&moe_w.shared_input_scale);
    let shared_gate_bytes = moe_w.shared_gate.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("qwen35moe shared_gate as_bytes failed".into())
    })?;
    let shared_up_bytes = moe_w.shared_up.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("qwen35moe shared_up as_bytes failed".into())
    })?;
    let shared_down_bytes = moe_w.shared_down.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward("qwen35moe shared_down as_bytes failed".into())
    })?;

    backend_runtime::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input(
        shared_gate_bytes,
        moe_w.shared_gate.ggml_type,
        shared_up_bytes,
        moe_w.shared_up.ggml_type,
        shared_down_bytes,
        moe_w.shared_down.ggml_type,
        shared_input_scale,
        gate_exps_bytes,
        up_exps_bytes,
        down_exps_bytes,
        router_w_data,
        moe_w.n_expert,
        hidden_dim,
        normalized_id,
        normalized_desc,
        residual_id,
        residual_desc,
        seq_len,
        moe_w.n_expert_used,
        moe_w.down_quant,
        moe_w.n_ff,
        moe_w.n_embd,
        reuse_residual_output,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn try_forward_ffn_qwen35moe_device_input_carrier(
    moe_w: &SharedExpertMoELayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    normalized_id: cuda_runtime::DeviceTensorId,
    normalized_desc: cuda_runtime::DeviceTensorDesc,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    let Some((output_id, output_desc)) = try_forward_ffn_qwen35moe_device_input_raw(
        moe_w,
        seq_len,
        hidden_dim,
        normalized_id,
        normalized_desc,
        residual_id,
        residual_desc,
        cuda_runtime::qwen35_device_moe_inplace_residual_enabled(),
    )?
    else {
        return Ok(None);
    };
    Ok(Some(backend_runtime::NemotronDeviceLayerOutput {
        output_id,
        output_desc,
    }))
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn try_forward_ffn_qwen35moe_device_input(
    moe_w: &SharedExpertMoELayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    normalized_id: cuda_runtime::DeviceTensorId,
    normalized_desc: cuda_runtime::DeviceTensorDesc,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
) -> crate::error::Result<Option<Tensor>> {
    let Some((output_id, _output_desc)) = try_forward_ffn_qwen35moe_device_input_raw(
        moe_w,
        seq_len,
        hidden_dim,
        normalized_id,
        normalized_desc,
        residual_id,
        residual_desc,
        false,
    )?
    else {
        return Ok(None);
    };

    let values = match backend_runtime::download_cuda_device_tensor_f32(output_id) {
        Ok(values) => values,
        Err(err) => {
            let _ = cuda_runtime::release_device_tensor(output_id);
            return Err(err);
        }
    };
    match cuda_runtime::release_device_tensor(output_id) {
        Ok(true) => {}
        Ok(false) => {
            return Err(crate::error::LlmError::Forward(
                "qwen35moe device-input output tensor was already missing".to_string(),
            ));
        }
        Err(err) => {
            return Err(crate::error::LlmError::Forward(format!(
                "qwen35moe device-input output release failed: {err}"
            )));
        }
    }
    Ok(Some(Tensor::from_vec(values, &[seq_len, hidden_dim])))
}

#[derive(Clone, Copy)]
struct Hy3PrefillRouteSlot {
    expert: usize,
    token: usize,
    rank: usize,
    weight: f32,
}

fn hy3_q2q3_expert_major_prefill_supported(
    architecture: ModelArchitecture,
    view: &SharedExpertMoEView<'_>,
    seq_len: usize,
) -> bool {
    hy3_q2q3_expert_major_prefill_enabled()
        && !cfg!(feature = "cuda")
        && architecture == ModelArchitecture::Hy3
        && hy3_q2q3_expert_major_has_reuse(seq_len, view.n_expert_used, view.n_expert)
        && view.expert_gating_func == 2
        && view.expert_weights_norm
        && view.n_expert_used > 0
        && view.gate_quant == GGMLType::Q2_K
        && view.up_quant == GGMLType::Q2_K
        && view.down_quant == GGMLType::Q3_K
        && view.shared_gate_quant == GGMLType::Q2_K
        && view.shared_up_quant == GGMLType::Q2_K
        && view.shared_down_quant == GGMLType::Q3_K
}

fn forward_hy3_q2q3_expert_major_moe(
    view: &SharedExpertMoEView<'_>,
    norm_all: &[f32],
    moe_outputs: &mut [f32],
    seq_len: usize,
    architecture: ModelArchitecture,
    layer_index: usize,
    page_cache: Option<&super::page_cache::SparseExpertPageCache>,
) {
    let n_embd = view.n_embd;
    let n_ff = view.n_ff;
    let n_expert = view.n_expert;
    let selected_count = view.n_expert_used.min(n_expert);
    assert_eq!(norm_all.len(), seq_len * n_embd);
    assert_eq!(moe_outputs.len(), seq_len * n_embd);

    // Route every token first. The packed slot table is then sorted by expert,
    // while `rank` retains the model's original top-k reduction order.
    let mut router_logits = vec![0.0f32; seq_len * n_expert];
    gemv_f32(
        view.router_w,
        norm_all,
        &mut router_logits,
        n_expert,
        n_embd,
        seq_len,
    );
    let selection_bias = view
        .router_selection_bias
        .expect("Hy3 expert-major prefill requires router selection bias");
    assert_eq!(selection_bias.len(), n_expert);

    let mut idx_all = vec![0usize; n_expert];
    let mut probs = vec![0.0f32; n_expert];
    let mut selected_weights = vec![0.0f32; selected_count];
    let mut route_slots = Vec::with_capacity(seq_len * selected_count);
    let mut shared_gate_scalars = vec![1.0f32; seq_len];
    let adaptive_top_p = crate::engine::policy::moe_adaptive_top_p();
    for token in 0..seq_len {
        let logits = &router_logits[token * n_expert..(token + 1) * n_expert];
        let selected_len = hy3_sigmoid_topk_route(
            logits,
            selection_bias,
            selected_count,
            view.expert_weights_norm,
            view.expert_weights_scale,
            adaptive_top_p,
            &mut idx_all,
            &mut probs,
            &mut selected_weights,
        );
        for rank in 0..selected_len {
            route_slots.push(Hy3PrefillRouteSlot {
                expert: idx_all[rank],
                token,
                rank,
                weight: selected_weights[rank],
            });
        }
        if view.shared_expert_gated {
            let shared_gate_dot: f32 = norm_all[token * n_embd..(token + 1) * n_embd]
                .iter()
                .zip(view.shared_input_scale.iter())
                .map(|(a, b)| a * b)
                .sum();
            shared_gate_scalars[token] = 1.0 / (1.0 + (-shared_gate_dot).exp());
        }
    }
    route_slots.sort_unstable_by_key(|slot| (slot.expert, slot.token, slot.rank));

    let gate_bpr = expert_bytes_per_row(n_embd, view.gate_quant, "gate_exps");
    let up_bpr = expert_bytes_per_row(n_embd, view.up_quant, "up_exps");
    let down_bpr = down_bytes_per_row(n_ff, view.down_quant);
    let gate_per_expert = n_ff * gate_bpr;
    let up_per_expert = n_ff * up_bpr;
    let down_per_expert = n_embd * down_bpr;

    // One output per routed slot preserves token-major rank reduction exactly.
    // This path is opt-in while its throughput and scratch footprint are measured.
    let mut rank_outputs = vec![0.0f32; seq_len * selected_count * n_embd];
    let mut expert_inputs = Vec::new();
    let mut gate_up = Vec::new();
    let mut expert_outputs = Vec::new();
    let mut group_start = 0;
    while group_start < route_slots.len() {
        let expert = route_slots[group_start].expert;
        let mut group_end = group_start + 1;
        while group_end < route_slots.len() && route_slots[group_end].expert == expert {
            group_end += 1;
        }
        let slots = &route_slots[group_start..group_end];
        let batch = slots.len();

        expert_inputs.resize(batch * n_embd, 0.0);
        for (batch_idx, slot) in slots.iter().enumerate() {
            expert_inputs[batch_idx * n_embd..(batch_idx + 1) * n_embd]
                .copy_from_slice(&norm_all[slot.token * n_embd..(slot.token + 1) * n_embd]);
        }

        gate_up.resize(2 * batch * n_ff, 0.0);
        let (gate, up) = gate_up.split_at_mut(batch * n_ff);
        let gate_start = expert * gate_per_expert;
        gemv_generic(
            &view.gate_exps_bytes[gate_start..gate_start + gate_per_expert],
            &expert_inputs,
            gate,
            n_ff,
            n_embd,
            batch,
            gate_bpr,
            view.gate_quant,
        );
        let up_start = expert * up_per_expert;
        gemv_generic(
            &view.up_exps_bytes[up_start..up_start + up_per_expert],
            &expert_inputs,
            up,
            n_ff,
            n_embd,
            batch,
            up_bpr,
            view.up_quant,
        );
        apply_model_gate_mul_inplace(gate, up, architecture);

        expert_outputs.resize(batch * n_embd, 0.0);
        let down_start = expert * down_per_expert;
        gemv_generic(
            &view.down_exps_bytes[down_start..down_start + down_per_expert],
            gate,
            &mut expert_outputs,
            n_embd,
            n_ff,
            batch,
            down_bpr,
            view.down_quant,
        );
        for (batch_idx, slot) in slots.iter().enumerate() {
            let src = &expert_outputs[batch_idx * n_embd..(batch_idx + 1) * n_embd];
            let rank_start = (slot.token * selected_count + slot.rank) * n_embd;
            let dst = &mut rank_outputs[rank_start..rank_start + n_embd];
            axpby_f32_inplace(dst, src, slot.weight, 0.0);
        }
        if let Some(page_cache) = page_cache {
            page_cache.touch(layer_index, &[expert]);
        }

        group_start = group_end;
    }

    let shared_gate_bpr = expert_bytes_per_row(n_embd, view.shared_gate_quant, "shared_gate");
    let shared_up_bpr = expert_bytes_per_row(n_embd, view.shared_up_quant, "shared_up");
    let shared_down_bpr = down_bytes_per_row(n_ff, view.shared_down_quant);
    let mut shared_gate_up = vec![0.0f32; 2 * seq_len * n_ff];
    let (shared_gate, shared_up) = shared_gate_up.split_at_mut(seq_len * n_ff);
    gemv_generic(
        view.shared_gate_bytes,
        norm_all,
        shared_gate,
        n_ff,
        n_embd,
        seq_len,
        shared_gate_bpr,
        view.shared_gate_quant,
    );
    gemv_generic(
        view.shared_up_bytes,
        norm_all,
        shared_up,
        n_ff,
        n_embd,
        seq_len,
        shared_up_bpr,
        view.shared_up_quant,
    );
    apply_model_gate_mul_inplace(shared_gate, shared_up, architecture);

    let mut shared_outputs = vec![0.0f32; seq_len * n_embd];
    gemv_generic(
        view.shared_down_bytes,
        shared_gate,
        &mut shared_outputs,
        n_embd,
        n_ff,
        seq_len,
        shared_down_bpr,
        view.shared_down_quant,
    );

    moe_outputs.fill(0.0);
    for token in 0..seq_len {
        let token_output = &mut moe_outputs[token * n_embd..(token + 1) * n_embd];
        for rank in 0..selected_count {
            let rank_start = (token * selected_count + rank) * n_embd;
            let rank_output = &rank_outputs[rank_start..rank_start + n_embd];
            add_f32_inplace(token_output, rank_output);
        }
        let shared = &shared_outputs[token * n_embd..(token + 1) * n_embd];
        let shared_scale = shared_gate_scalars[token];
        axpby_f32_inplace(token_output, shared, shared_scale, 1.0);
    }
}

/// Prefill FFN for split top-k sparse experts plus an always-on shared expert.
///
/// Qwen3.5 MoE and Hy3 share this execution shape; routing and quantization
/// remain metadata-driven.
///
/// Layout per token:
/// ```text
///   h_norm  = ffn_pre_norm(attn_out_t)
///   moe_out = SharedExpertMoEView::forward(h_norm)
///   out_t   = attn_out_t + moe_out              // residual (caller-free)
/// ```
///
/// `ffn_norm_weight` is the pre-FFN RMSNorm weight — for the attention variant
/// it's `ffn_norm` (or `post_attn_norm` fallback), for GDN layers it's
/// `post_attn_norm` directly.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn forward_shared_expert_moe(
    architecture: ModelArchitecture,
    hidden: Tensor,
    ffn_norm_weight: &Tensor,
    moe_w: &SharedExpertMoELayerWeights,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
) -> crate::error::Result<Tensor> {
    let attn_out_data = kernels::tensor_as_f32_slice(&hidden);
    let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm_weight);

    let router_w_data = moe_w
        .router_f32()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE router_f32 failed".into()))?;
    let gate_exps_bytes = moe_w
        .gate_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE gate_exps_bytes failed".into()))?;
    let up_exps_bytes = moe_w
        .up_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE up_exps_bytes failed".into()))?;
    let down_exps_bytes = moe_w
        .down_exps_bytes()
        .ok_or_else(|| crate::error::LlmError::Forward("MoE down_exps_bytes failed".into()))?;
    let shared_input_scale = kernels::tensor_as_f32_slice(&moe_w.shared_input_scale);
    let shared_gate_bytes =
        moe_w.shared_gate.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_gate as_bytes failed".into())
        })?;
    let shared_up_bytes =
        moe_w.shared_up.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_up as_bytes failed".into())
        })?;
    let shared_down_bytes =
        moe_w.shared_down.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward("MoE shared_down as_bytes failed".into())
        })?;

    let view = crate::engine::moe::SharedExpertMoEView {
        router_w: router_w_data,
        router_selection_bias: moe_w
            .router_selection_bias
            .as_ref()
            .map(kernels::tensor_as_f32_slice),
        expert_gating_func: moe_w.expert_gating_func,
        expert_weights_norm: moe_w.expert_weights_norm,
        expert_weights_scale: moe_w.expert_weights_scale,
        gate_exps_bytes,
        gate_quant: moe_w.gate_quant,
        up_exps_bytes,
        up_quant: moe_w.up_quant,
        down_exps_bytes,
        down_quant: moe_w.down_quant,
        shared_input_scale,
        shared_expert_gated: moe_w.shared_expert_gated,
        shared_gate_bytes,
        shared_gate_quant: moe_w.shared_gate.ggml_type,
        shared_up_bytes,
        shared_up_quant: moe_w.shared_up.ggml_type,
        shared_down_bytes,
        shared_down_quant: moe_w.shared_down.ggml_type,
        n_embd: moe_w.n_embd,
        n_ff: moe_w.n_ff,
        n_expert: moe_w.n_expert,
        n_expert_used: moe_w.n_expert_used,
        layer_idx: Some(layer_idx),
    };

    let trace_prefill_routes = crate::engine::moe_trace::route_trace_is_active()
        || crate::engine::moe_trace::predictor_trace_is_active();

    // pm123: Qwen3.6 IQ expert(IQ3_XXS gate/up + IQ4_XS down, Q8_0 shared) prefill 을
    // GLM IQ batch 커널로 Metal 라우팅(unsloth UD-Q3_K_M). 미지원/trace 시 Ok(None) fallback.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if architecture != ModelArchitecture::GlmDsa && seq_len > 1 && !trace_prefill_routes {
        if let Some(output) = super::qwen_iq_metal_prefill::forward(
            &view,
            attn_out_data,
            ffn_norm_data,
            norm_eps,
            seq_len,
            hidden_dim,
            architecture,
        )? {
            return Ok(Tensor::from_vec(output, &[seq_len, hidden_dim]));
        }
    }

    #[cfg(feature = "cuda")]
    if !trace_prefill_routes
        && architecture == ModelArchitecture::GlmDsa
        && view.gate_quant == view.up_quant
        && ((view.gate_quant == GGMLType::IQ2_XXS && view.down_quant == GGMLType::IQ3_XXS)
            || (super::glm_prefill::ud_iq_batch_enabled()
                && matches!(view.gate_quant, GGMLType::IQ2_XXS | GGMLType::IQ2_S)
                && matches!(view.down_quant, GGMLType::IQ3_XXS | GGMLType::IQ4_XS)))
    {
        let sparse_file_regions = moe_w.sparse_expert_file_regions();
        let direct_file = cuda_runtime::glm_moe_direct_file_prefill_enabled(
            moe_w.prefer_sparse_moe_cuda && sparse_file_regions.is_some(),
        );
        let mut output = attn_out_data.to_vec();
        if super::glm_prefill::enabled() {
            match super::glm_prefill::forward(
                &view,
                &mut output,
                ffn_norm_data,
                norm_eps,
                seq_len,
                hidden_dim,
                moe_w.sparse_page_cache.as_deref(),
                sparse_file_regions.as_ref(),
                direct_file,
            ) {
                Ok(()) => return Ok(Tensor::from_vec(output, &[seq_len, hidden_dim])),
                Err(error) => {
                    #[cfg(feature = "cuda")]
                    return Err(error);
                    #[cfg(not(feature = "cuda"))]
                    if direct_file {
                        return Err(error);
                    }
                    #[cfg(not(feature = "cuda"))]
                    eprintln!(
                        "[WARN] GLM batched sparse prefill failed, using token dispatch: {error}"
                    );
                }
            }
        }
        let mut norm = vec![0.0f32; hidden_dim];
        let mut moe_out = vec![0.0f32; hidden_dim];
        for token in 0..seq_len {
            let start = token * hidden_dim;
            apply_model_norm_into(
                &output[start..start + hidden_dim],
                ffn_norm_data,
                norm_eps,
                &mut norm,
                architecture,
            );
            let completed = view.forward_add_residual_with_policy(
                &norm,
                &mut moe_out,
                &mut output[start..start + hidden_dim],
                true,
                moe_w.sparse_page_cache.as_deref(),
            );
            if !completed {
                add_f32_inplace(&mut output[start..start + hidden_dim], &moe_out);
            }
        }
        return Ok(Tensor::from_vec(output, &[seq_len, hidden_dim]));
    }

    // pm113: GLM Metal token-batch prefill. route trace 기록 순서가 기존 per-token
    // 경로와 동일해서 trace 활성 시에도 배치 경로를 유지한다 (predictor trace 는
    // probs 기록이 필요해 제외).
    if architecture == ModelArchitecture::GlmDsa
        && seq_len > 1
        && !crate::engine::moe_trace::predictor_trace_is_active()
    {
        let metal_file_regions = moe_w.sparse_expert_file_regions();
        if let Some(output) = super::glm_metal_prefill::forward(
            &view,
            attn_out_data,
            ffn_norm_data,
            norm_eps,
            seq_len,
            hidden_dim,
            moe_w.sparse_page_cache.as_deref(),
            metal_file_regions.as_ref(),
        )? {
            return Ok(Tensor::from_vec(output, &[seq_len, hidden_dim]));
        }
    }

    if !cfg!(feature = "cuda")
        && !trace_prefill_routes
        && hy3_q2q3_expert_major_prefill_supported(architecture, &view, seq_len)
    {
        let mut norm_all = vec![0.0f32; seq_len * hidden_dim];
        for token in 0..seq_len {
            let start = token * hidden_dim;
            apply_model_norm_into(
                &attn_out_data[start..start + hidden_dim],
                ffn_norm_data,
                norm_eps,
                &mut norm_all[start..start + hidden_dim],
                architecture,
            );
        }
        let mut moe_outputs = vec![0.0f32; seq_len * hidden_dim];
        let chunk_tokens = hy3_q2q3_expert_major_chunk_tokens(&view);
        let mut token_start = 0;
        while token_start < seq_len {
            let chunk_len = (seq_len - token_start).min(chunk_tokens);
            let value_start = token_start * hidden_dim;
            let value_end = value_start + chunk_len * hidden_dim;
            forward_hy3_q2q3_expert_major_moe(
                &view,
                &norm_all[value_start..value_end],
                &mut moe_outputs[value_start..value_end],
                chunk_len,
                architecture,
                layer_idx,
                moe_w.sparse_page_cache.as_deref(),
            );
            token_start += chunk_len;
        }
        return Ok(add_moe_output_to_hidden(
            moe_outputs,
            &hidden,
            seq_len,
            hidden_dim,
        ));
    }

    let llama_id_requested = backend_runtime::qwen_moe_llama_id_requested();
    let llama_id_product_tuple = qwen_moe_llama_id_product_tuple(
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
        moe_w.shared_gate.ggml_type,
        moe_w.shared_up.ggml_type,
        moe_w.shared_down.ggml_type,
    );

    #[cfg(target_arch = "aarch64")]
    if qwen35_cpu_expert_major_should_run(
        super::prefill_cpu::qwen35_cpu_expert_major_enabled(architecture, &view, seq_len),
        llama_id_requested,
        llama_id_product_tuple,
    ) {
        return super::prefill_cpu::forward_qwen35_cpu_expert_major(
            architecture,
            hidden,
            ffn_norm_data,
            &view,
            seq_len,
            hidden_dim,
            norm_eps,
            layer_idx,
        );
    }

    let native_batch_product_tuple = qwen_moe_native_batch_product_tuple(
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
        moe_w.shared_gate.ggml_type,
        moe_w.shared_up.ggml_type,
        moe_w.shared_down.ggml_type,
        cfg!(feature = "cuda"),
    );

    if qwen_moe_prefill_outer_guard(
        backend_runtime::qwen_moe_prefill_enabled_for_seq(seq_len),
        native_batch_product_tuple,
        llama_id_requested,
        llama_id_product_tuple,
    ) {
        let profiling = policy::profiling_enabled();
        let profile_all_layers = policy::qwen_profile_all_layers_enabled();
        let profile_layer = profile_all_layers || layer_idx == 0 || layer_idx == 3;
        let profile_tag = if profile_all_layers {
            format!("QWEN L{layer_idx}")
        } else if layer_idx == 0 {
            "GDN L0".to_string()
        } else {
            "ATN L3".to_string()
        };
        let prof_moe = |label: &str, t: std::time::Instant| {
            if profiling && profile_layer {
                eprintln!(
                    "  [{}] moe_{:16} {:.1}ms",
                    profile_tag,
                    label,
                    t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
        let expert_bytes = sparse_expert_bytes(
            moe_w.n_embd,
            moe_w.n_ff,
            moe_w.gate_quant,
            moe_w.up_quant,
            moe_w.down_quant,
        )
        .ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "CUDA prefill MoE unsupported down quant {:?}",
                moe_w.down_quant
            ))
        })?;

        let selected_slot_capacity = seq_len * moe_w.n_expert_used;
        let mut norm_all = vec![0.0f32; seq_len * hidden_dim];
        let mut shared_route_weights = vec![0.0f32; seq_len];
        let mut expert_ids = Vec::with_capacity(selected_slot_capacity);
        let mut route_weights = Vec::with_capacity(selected_slot_capacity);
        let mut token_ids = Vec::with_capacity(selected_slot_capacity);

        let t_route = std::time::Instant::now();
        apply_model_norm_into(
            attn_out_data,
            ffn_norm_data,
            norm_eps,
            &mut norm_all,
            architecture,
        );
        norm_all
            .par_chunks(hidden_dim)
            .zip(shared_route_weights.par_iter_mut())
            .for_each(|(norm_t, shared_route_weight)| {
                let gate_dot: f32 = norm_t
                    .iter()
                    .zip(shared_input_scale.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                *shared_route_weight = 1.0 / (1.0 + (-gate_dot).exp());
            });

        let trace_prefill_routes = crate::engine::moe_trace::route_trace_is_active()
            || crate::engine::moe_trace::predictor_trace_is_active();
        let jit_load_requested = qwen35_moe_jit_load_requested();
        let use_expert_major_slots =
            !trace_prefill_routes && qwen35_prefill_expert_major_slots_enabled();
        let use_device_topk = qwen35_prefill_device_topk_enabled()
            && !trace_prefill_routes
            && !jit_load_requested
            && !use_expert_major_slots;
        let use_selected_base = !trace_prefill_routes && qwen35_prefill_selected_base_enabled();
        let v4_requested =
            backend_runtime::qwen_moe_prefill_shared_sparse_selected_base_metal_v4_requested();
        let v3_requested =
            backend_runtime::qwen_moe_prefill_shared_sparse_selected_base_metal_v3_requested();
        let selected_base_metal_flow = qwen_moe_selected_base_metal_flow(
            v4_requested,
            v3_requested,
            llama_id_requested,
            use_selected_base,
            trace_prefill_routes,
            jit_load_requested,
        );
        if crate::engine::policy::env_os_string("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TRACE")
            .is_some()
        {
            eprintln!(
                "[qwen-moe-flow] layer={layer_idx} seq={seq_len} flow={selected_base_metal_flow:?} llama_id={llama_id_requested} selected_base={use_selected_base} route_trace={trace_prefill_routes} jit={jit_load_requested}"
            );
        }

        #[cfg(feature = "cuda")]
        // The combined wrapper stays opt-in. The default minor stack uses the
        // route-array + selected-base path below, which measured better than
        // routing through this combined facade.
        if qwen35_prefill_device_topk_combined_enabled()
            && !trace_prefill_routes
            && !crate::engine::moe_trace::is_active()
            && !jit_load_requested
            && !use_expert_major_slots
            && crate::engine::tuning_runtime::shared_f32_enabled()
        {
            let t_combined = std::time::Instant::now();
            let out_buf = backend_runtime::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token(
                shared_gate_bytes,
                moe_w.shared_gate.ggml_type,
                shared_up_bytes,
                moe_w.shared_up.ggml_type,
                shared_down_bytes,
                moe_w.shared_down.ggml_type,
                gate_exps_bytes,
                up_exps_bytes,
                down_exps_bytes,
                router_w_data,
                moe_w.n_expert,
                hidden_dim,
                &norm_all,
                &shared_route_weights,
                seq_len,
                moe_w.n_expert_used,
                moe_w.down_quant,
                moe_w.n_ff,
                moe_w.n_embd,
            )?;
            prof_moe("combined_device_topk", t_combined);

            return Ok(add_moe_output_to_hidden(
                out_buf, &hidden, seq_len, hidden_dim,
            ));
        }

        let needs_token_major_metal_routes = matches!(
            selected_base_metal_flow,
            QwenMoeSelectedBaseMetalFlow::V4TokenMajor | QwenMoeSelectedBaseMetalFlow::V3TokenMajor
        ) || (selected_base_metal_flow
            == QwenMoeSelectedBaseMetalFlow::LlamaIdTokenMajor
            && llama_id_product_tuple);
        let mut token_major_route_arrays = if needs_token_major_metal_routes {
            Some(
                backend_runtime::qwen_moe_prefill_sparse_route_arrays_device_topk(
                    router_w_data,
                    moe_w.n_expert,
                    hidden_dim,
                    &norm_all,
                    seq_len,
                    moe_w.n_expert_used,
                )?,
            )
        } else {
            None
        };

        match selected_base_metal_flow {
            QwenMoeSelectedBaseMetalFlow::LlamaIdTokenMajor =>
            {
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                if llama_id_product_tuple {
                    let route_arrays_metal = token_major_route_arrays
                        .as_ref()
                        .expect("Llama ID flow requires token-major routes");
                    route_arrays_metal
                        .validate_token_rank(seq_len, moe_w.n_expert_used, moe_w.n_expert)
                        .map_err(crate::error::LlmError::Forward)?;
                    let t_metal = std::time::Instant::now();
                    if let Some(output) = backend_runtime::qwen_moe_llama_id_prefill(
                        gate_exps_bytes,
                        up_exps_bytes,
                        down_exps_bytes,
                        expert_bytes.gate,
                        expert_bytes.up,
                        expert_bytes.down,
                        route_arrays_metal.expert_ids(),
                        route_arrays_metal.route_weights(),
                        shared_gate_bytes,
                        shared_up_bytes,
                        shared_down_bytes,
                        &shared_route_weights,
                        moe_w.gate_quant,
                        moe_w.up_quant,
                        moe_w.down_quant,
                        moe_w.shared_gate.ggml_type,
                        moe_w.shared_up.ggml_type,
                        moe_w.shared_down.ggml_type,
                        &norm_all,
                        seq_len,
                        hidden_dim,
                        moe_w.n_ff,
                        layer_idx,
                    )
                    .map_err(crate::error::LlmError::Forward)?
                    {
                        prof_moe("llama_id", t_metal);
                        return Ok(add_moe_output_to_hidden(
                            output.values,
                            &hidden,
                            seq_len,
                            hidden_dim,
                        ));
                    }
                }
            }
            QwenMoeSelectedBaseMetalFlow::V4TokenMajor
            | QwenMoeSelectedBaseMetalFlow::V3TokenMajor => {
                let t_metal = std::time::Instant::now();
                let route_arrays_metal = token_major_route_arrays
                    .as_ref()
                    .expect("selected-base Metal flow requires token-major routes");
                let shared_token_ids_metal = (0..seq_len).map(|t| t as u32).collect::<Vec<_>>();
                let out_buf = match selected_base_metal_flow {
                    QwenMoeSelectedBaseMetalFlow::V4TokenMajor => {
                        backend_runtime::qwen_moe_prefill_shared_sparse_selected_base_metal_v4_token_major(
                            shared_gate_bytes,
                            moe_w.shared_gate.ggml_type,
                            shared_up_bytes,
                            moe_w.shared_up.ggml_type,
                            shared_down_bytes,
                            moe_w.shared_down.ggml_type,
                            gate_exps_bytes,
                            moe_w.gate_quant,
                            up_exps_bytes,
                            moe_w.up_quant,
                            down_exps_bytes,
                            route_arrays_metal,
                            &shared_route_weights,
                            &shared_token_ids_metal,
                            seq_len,
                            moe_w.down_quant,
                            moe_w.n_ff,
                            moe_w.n_embd,
                            &norm_all,
                        )?
                    }
                    QwenMoeSelectedBaseMetalFlow::V3TokenMajor => {
                        backend_runtime::qwen_moe_prefill_shared_sparse_selected_base_metal_v3_token_major(
                            shared_gate_bytes,
                            moe_w.shared_gate.ggml_type,
                            shared_up_bytes,
                            moe_w.shared_up.ggml_type,
                            shared_down_bytes,
                            moe_w.shared_down.ggml_type,
                            gate_exps_bytes,
                            moe_w.gate_quant,
                            up_exps_bytes,
                            moe_w.up_quant,
                            down_exps_bytes,
                            route_arrays_metal,
                            &shared_route_weights,
                            &shared_token_ids_metal,
                            seq_len,
                            moe_w.down_quant,
                            moe_w.n_ff,
                            moe_w.n_embd,
                            &norm_all,
                        )?
                    }
                    _ => unreachable!("selected-base Metal flow already filtered"),
                };
                if let Some(out_buf) = out_buf {
                    let label = match selected_base_metal_flow {
                        QwenMoeSelectedBaseMetalFlow::V4TokenMajor => "shared_sparse_metal_v4",
                        QwenMoeSelectedBaseMetalFlow::V3TokenMajor => "shared_sparse_metal_v3",
                        _ => unreachable!("selected-base Metal flow already filtered"),
                    };
                    prof_moe(label, t_metal);
                    return Ok(add_moe_output_to_hidden(
                        out_buf, &hidden, seq_len, hidden_dim,
                    ));
                }
            }
            QwenMoeSelectedBaseMetalFlow::Disabled
            | QwenMoeSelectedBaseMetalFlow::GenericSelectedBase => {}
        }

        let mut gate_slices = if use_selected_base {
            Vec::new()
        } else {
            Vec::with_capacity(selected_slot_capacity)
        };
        let mut up_slices = if use_selected_base {
            Vec::new()
        } else {
            Vec::with_capacity(selected_slot_capacity)
        };
        let mut down_slices = if use_selected_base {
            Vec::new()
        } else {
            Vec::with_capacity(selected_slot_capacity)
        };
        let mut route_arrays = if use_device_topk {
            if let Some(route_arrays) = token_major_route_arrays.take() {
                Some(route_arrays)
            } else {
                Some(
                    backend_runtime::qwen_moe_prefill_sparse_route_arrays_device_topk(
                        router_w_data,
                        moe_w.n_expert,
                        hidden_dim,
                        &norm_all,
                        seq_len,
                        moe_w.n_expert_used,
                    )?,
                )
            }
        } else {
            None
        };
        let mut sparse_slots = if route_arrays.is_some() {
            Vec::new()
        } else {
            let router_logits = backend_runtime::qwen_moe_prefill_router_logits(
                router_w_data,
                moe_w.n_expert,
                hidden_dim,
                &norm_all,
            )?;
            if use_expert_major_slots {
                backend_runtime::qwen_moe_prefill_sparse_slots_expert_major(
                    &router_logits,
                    seq_len,
                    moe_w.n_expert,
                    moe_w.n_expert_used,
                )
            } else {
                let slots = backend_runtime::qwen_moe_prefill_sparse_slots(
                    &router_logits,
                    seq_len,
                    moe_w.n_expert,
                    moe_w.n_expert_used,
                );
                trace_qwen35_prefill_moe_routes(
                    layer_idx,
                    seq_len,
                    moe_w.n_expert,
                    moe_w.n_expert_used,
                    &router_logits,
                    &slots,
                );
                slots
            }
        };
        if jit_load_requested {
            request_qwen35_moe_jit_load_from_route_slots(
                layer_idx,
                &sparse_slots,
                moe_w.n_expert,
                gate_exps_bytes,
                up_exps_bytes,
                down_exps_bytes,
                expert_bytes.gate,
                expert_bytes.up,
                expert_bytes.down,
            );
        }
        if let Some(route_arrays) = route_arrays.as_ref() {
            backend_runtime::log_qwen_moe_prefill_route_array_hist(
                layer_idx,
                seq_len,
                moe_w.n_expert,
                route_arrays,
            );
        } else {
            backend_runtime::log_qwen_moe_prefill_route_hist(
                layer_idx,
                seq_len,
                moe_w.n_expert,
                &sparse_slots,
            );
        }
        if !use_expert_major_slots {
            if let Some(route_arrays) = route_arrays.as_mut() {
                route_arrays.sort_by_expert_token();
            } else {
                sparse_slots.sort_unstable_by_key(|slot| (slot.expert, slot.token));
            }
        }
        #[cfg(feature = "cuda")]
        let mut unique_experts = 0usize;
        #[cfg(feature = "cuda")]
        let mut last_expert = None;
        if let Some(route_arrays) = route_arrays {
            for (expert, weight, token) in route_arrays.iter() {
                let e = expert as usize;
                #[cfg(feature = "cuda")]
                if last_expert != Some(e) {
                    unique_experts += 1;
                    last_expert = Some(e);
                }
                if !use_selected_base {
                    gate_slices
                        .push(&gate_exps_bytes[e * expert_bytes.gate..(e + 1) * expert_bytes.gate]);
                    up_slices.push(&up_exps_bytes[e * expert_bytes.up..(e + 1) * expert_bytes.up]);
                    down_slices
                        .push(&down_exps_bytes[e * expert_bytes.down..(e + 1) * expert_bytes.down]);
                }
                expert_ids.push(expert);
                route_weights.push(weight);
                token_ids.push(token);
            }
        } else {
            for slot in sparse_slots {
                let e = slot.expert;
                #[cfg(feature = "cuda")]
                if last_expert != Some(e) {
                    unique_experts += 1;
                    last_expert = Some(e);
                }
                if !use_selected_base {
                    gate_slices
                        .push(&gate_exps_bytes[e * expert_bytes.gate..(e + 1) * expert_bytes.gate]);
                    up_slices.push(&up_exps_bytes[e * expert_bytes.up..(e + 1) * expert_bytes.up]);
                    down_slices
                        .push(&down_exps_bytes[e * expert_bytes.down..(e + 1) * expert_bytes.down]);
                }
                expert_ids.push(e as u32);
                route_weights.push(slot.weight);
                token_ids.push(slot.token);
            }
        }
        prof_moe("route", t_route);

        let mut shared_gate_slices = Vec::with_capacity(seq_len);
        let mut shared_up_slices = Vec::with_capacity(seq_len);
        let mut shared_down_slices = Vec::with_capacity(seq_len);
        let mut shared_token_ids = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            shared_gate_slices.push(shared_gate_bytes);
            shared_up_slices.push(shared_up_bytes);
            shared_down_slices.push(shared_down_bytes);
            shared_token_ids.push(t as u32);
        }

        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if use_selected_base {
            let t_combined = std::time::Instant::now();
            if let Some(out_buf) =
                backend_runtime::qwen_moe_prefill_shared_sparse_selected_base_metal(
                    shared_gate_bytes,
                    moe_w.shared_gate.ggml_type,
                    shared_up_bytes,
                    moe_w.shared_up.ggml_type,
                    shared_down_bytes,
                    moe_w.shared_down.ggml_type,
                    gate_exps_bytes,
                    moe_w.gate_quant,
                    up_exps_bytes,
                    moe_w.up_quant,
                    down_exps_bytes,
                    &expert_ids,
                    &route_weights,
                    &token_ids,
                    &shared_route_weights,
                    &shared_token_ids,
                    seq_len,
                    moe_w.down_quant,
                    moe_w.n_ff,
                    moe_w.n_embd,
                    &norm_all,
                )?
            {
                prof_moe("shared_sparse_metal", t_combined);
                return Ok(add_moe_output_to_hidden(
                    out_buf, &hidden, seq_len, hidden_dim,
                ));
            }
        }

        #[cfg(feature = "cuda")]
        if crate::engine::tuning_runtime::shared_f32_enabled() {
            let t_combined = std::time::Instant::now();
            let full_layer_min_permille =
                crate::engine::tuning_runtime::prefill_moe_full_layer_min_expert_permille();
            let use_full_layer = !use_selected_base
                && (crate::engine::tuning_runtime::moe_layer_cache_enabled()
                    || crate::engine::tuning_runtime::prefill_moe_range_slab_enabled()
                    || (crate::engine::tuning_runtime::prefill_moe_full_layer_enabled()
                        && unique_experts.saturating_mul(1000)
                            >= moe_w.n_expert.saturating_mul(full_layer_min_permille)));
            let out_buf = if use_full_layer {
                backend_runtime::qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token(
                    shared_gate_bytes,
                    moe_w.shared_gate.ggml_type,
                    shared_up_bytes,
                    moe_w.shared_up.ggml_type,
                    shared_down_bytes,
                    moe_w.shared_down.ggml_type,
                    gate_exps_bytes,
                    up_exps_bytes,
                    down_exps_bytes,
                    &expert_ids,
                    &route_weights,
                    &token_ids,
                    &shared_route_weights,
                    seq_len,
                    moe_w.down_quant,
                    moe_w.n_ff,
                    moe_w.n_embd,
                    &norm_all,
                )?
            } else if use_selected_base {
                backend_runtime::qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token(
                    shared_gate_bytes,
                    moe_w.shared_gate.ggml_type,
                    shared_up_bytes,
                    moe_w.shared_up.ggml_type,
                    shared_down_bytes,
                    moe_w.shared_down.ggml_type,
                    gate_exps_bytes,
                    up_exps_bytes,
                    down_exps_bytes,
                    &expert_ids,
                    &route_weights,
                    &token_ids,
                    &shared_route_weights,
                    seq_len,
                    moe_w.down_quant,
                    moe_w.n_ff,
                    moe_w.n_embd,
                    &norm_all,
                )?
            } else {
                backend_runtime::qwen_moe_prefill_combined_f32_shared_sparse_by_token(
                    shared_gate_bytes,
                    moe_w.shared_gate.ggml_type,
                    shared_up_bytes,
                    moe_w.shared_up.ggml_type,
                    shared_down_bytes,
                    moe_w.shared_down.ggml_type,
                    &gate_slices,
                    &up_slices,
                    &down_slices,
                    &expert_ids,
                    &route_weights,
                    &token_ids,
                    &shared_route_weights,
                    seq_len,
                    moe_w.down_quant,
                    moe_w.n_ff,
                    moe_w.n_embd,
                    &norm_all,
                )?
            };
            prof_moe("combined", t_combined);

            return Ok(add_moe_output_to_hidden(
                out_buf, &hidden, seq_len, hidden_dim,
            ));
        }

        let t_shared = std::time::Instant::now();
        let shared_out = backend_runtime::qwen_moe_prefill_shared_expert_batch(
            &shared_gate_slices,
            &shared_up_slices,
            &shared_down_slices,
            &shared_route_weights,
            &shared_token_ids,
            seq_len,
            shared_gate_bytes,
            moe_w.shared_gate.ggml_type,
            shared_up_bytes,
            moe_w.shared_up.ggml_type,
            shared_down_bytes,
            moe_w.shared_down.ggml_type,
            moe_w.n_ff,
            moe_w.n_embd,
            &norm_all,
        )?;
        prof_moe("shared", t_shared);

        let t_sparse = std::time::Instant::now();
        let sparse_out = if use_selected_base {
            backend_runtime::qwen_moe_prefill_sparse_experts_selected_base_by_token(
                gate_exps_bytes,
                up_exps_bytes,
                down_exps_bytes,
                &expert_ids,
                &route_weights,
                &token_ids,
                seq_len,
                moe_w.down_quant,
                moe_w.n_ff,
                moe_w.n_embd,
                &norm_all,
            )?
        } else {
            backend_runtime::qwen_moe_prefill_sparse_experts_by_token(
                &gate_slices,
                &up_slices,
                &down_slices,
                &route_weights,
                &token_ids,
                seq_len,
                moe_w.down_quant,
                moe_w.n_ff,
                moe_w.n_embd,
                &norm_all,
            )?
        };
        prof_moe("sparse", t_sparse);

        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            dump_task11_generic_moe_stage(layer_idx, "hidden_in", attn_out_data);
            dump_task11_generic_moe_stage(layer_idx, "norm", &norm_all);
            dump_task11_generic_moe_stage(layer_idx, "shared", &shared_out);
            dump_task11_generic_moe_stage(layer_idx, "sparse", &sparse_out);
        }

        let mut out_buf = sparse_out;
        add_f32_inplace(&mut out_buf, &shared_out);

        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        dump_task11_generic_moe_stage(layer_idx, "final", &out_buf);

        return Ok(add_moe_output_to_hidden(
            out_buf, &hidden, seq_len, hidden_dim,
        ));
    }

    let mut out_buf = attn_out_data.to_vec();
    let mut norm_buf = vec![0f32; hidden_dim];
    let mut moe_out = vec![0f32; hidden_dim];
    for t in 0..seq_len {
        let attn_out_t = &out_buf[t * hidden_dim..(t + 1) * hidden_dim];
        apply_model_norm_into(
            attn_out_t,
            ffn_norm_data,
            norm_eps,
            &mut norm_buf,
            architecture,
        );
        view.forward_with_page_cache(&norm_buf, &mut moe_out, moe_w.sparse_page_cache.as_deref());
        add_f32_inplace(&mut out_buf[t * hidden_dim..(t + 1) * hidden_dim], &moe_out);
    }

    Ok(Tensor::from_vec(out_buf, &[seq_len, hidden_dim]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::MoeRouteSlot;
    use parking_lot::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_env(key: &str, previous: Option<String>) {
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen35_prefill_trace_selection_reads_token_major_slots() {
        let slots = [
            MoeRouteSlot::new(9, 0, 0.7),
            MoeRouteSlot::new(4, 0, 0.3),
            MoeRouteSlot::new(7, 1, 0.6),
            MoeRouteSlot::new(6, 1, 0.4),
        ];
        let mut selected = [usize::MAX; 4];

        let count = fill_qwen35_prefill_trace_selection(1, 2, &slots, &mut selected);

        assert_eq!(count, 2);
        assert_eq!(&selected[..count], &[7, 6]);
    }

    #[test]
    fn qwen35_prefill_trace_probs_are_full_softmax() {
        let logits = [1.0_f32, 2.0, 0.0];
        let mut probs = [0.0_f32; 3];

        fill_qwen35_prefill_trace_probs(&logits, &mut probs);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(probs[1] > probs[0]);
        assert!(probs[0] > probs[2]);
    }

    #[test]
    fn qwen35_prefill_adds_moe_output_without_extra_tensor() {
        let hidden = Tensor::from_vec(vec![1.0_f32, -2.0, 3.5, 4.0], &[2, 2]);
        let moe_out = vec![0.25_f32, 0.5, -1.5, 2.0];

        let out = add_moe_output_to_hidden(moe_out, &hidden, 2, 2);

        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(kernels::tensor_as_f32_slice(&out), &[1.25, -1.5, 2.0, 6.0]);
    }

    #[test]
    fn qwen35_prefill_expert_major_slots_defaults_for_backend_and_allows_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_CUDA_QWEN35_PREFILL_EXPERT_MAJOR_SLOTS";
        let previous = crate::engine::policy::env_string(key);

        std::env::remove_var(key);
        assert_eq!(
            qwen35_prefill_expert_major_slots_enabled(),
            cfg!(all(feature = "metal", not(feature = "cuda")))
        );

        std::env::set_var(key, "0");
        assert!(!qwen35_prefill_expert_major_slots_enabled());

        std::env::set_var(key, "1");
        assert!(qwen35_prefill_expert_major_slots_enabled());

        restore_env(key, previous);
    }

    #[test]
    fn qwen35_prefill_device_topk_defaults_on_and_allows_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_CUDA_QWEN35_PREFILL_DEVICE_TOPK";
        let previous = crate::engine::policy::env_string(key);

        std::env::remove_var(key);
        assert!(qwen35_prefill_device_topk_enabled());

        std::env::set_var(key, "0");
        assert!(!qwen35_prefill_device_topk_enabled());

        std::env::set_var(key, "1");
        assert!(qwen35_prefill_device_topk_enabled());

        restore_env(key, previous);
    }

    #[test]
    fn qwen35_prefill_device_topk_combined_stays_opt_in() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_CUDA_QWEN35_PREFILL_DEVICE_TOPK_COMBINED";
        let previous = crate::engine::policy::env_string(key);

        std::env::remove_var(key);
        assert!(!qwen35_prefill_device_topk_combined_enabled());

        std::env::set_var(key, "0");
        assert!(!qwen35_prefill_device_topk_combined_enabled());

        std::env::set_var(key, "1");
        assert!(qwen35_prefill_device_topk_combined_enabled());

        restore_env(key, previous);
    }

    #[test]
    fn qwen35_prefill_selected_base_defaults_on_and_allows_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_CUDA_QWEN35_PREFILL_SELECTED_BASE";
        let previous = crate::engine::policy::env_string(key);

        std::env::remove_var(key);
        assert!(qwen35_prefill_selected_base_enabled());

        std::env::set_var(key, "0");
        assert!(!qwen35_prefill_selected_base_enabled());

        std::env::set_var(key, "1");
        assert!(qwen35_prefill_selected_base_enabled());

        restore_env(key, previous);
    }

    #[test]
    fn qwen_moe_selected_base_metal_flow_llama_id_preempts_v4_and_v3() {
        assert_eq!(
            qwen_moe_selected_base_metal_flow(true, true, true, true, false, false),
            QwenMoeSelectedBaseMetalFlow::LlamaIdTokenMajor
        );
        assert_eq!(
            qwen_moe_selected_base_metal_flow(true, true, false, true, false, false),
            QwenMoeSelectedBaseMetalFlow::V4TokenMajor
        );
        assert_eq!(
            qwen_moe_selected_base_metal_flow(false, true, false, true, false, false),
            QwenMoeSelectedBaseMetalFlow::V3TokenMajor
        );
        assert_eq!(
            qwen_moe_selected_base_metal_flow(false, false, false, true, false, false),
            QwenMoeSelectedBaseMetalFlow::GenericSelectedBase
        );
    }

    #[test]
    fn qwen_moe_selected_base_metal_flow_llama_id_requires_all_gates() {
        assert_eq!(
            qwen_moe_selected_base_metal_flow(true, true, true, false, false, false),
            QwenMoeSelectedBaseMetalFlow::Disabled
        );
        assert_eq!(
            qwen_moe_selected_base_metal_flow(true, true, true, true, true, false),
            QwenMoeSelectedBaseMetalFlow::Disabled
        );
        assert_eq!(
            qwen_moe_selected_base_metal_flow(true, true, true, true, false, true),
            QwenMoeSelectedBaseMetalFlow::Disabled
        );
    }

    #[test]
    fn qwen_moe_selected_base_metal_flow_llama_id_product_tuple_is_exact() {
        for tuple in [
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q5_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
            ],
        ] {
            assert!(qwen_moe_llama_id_product_tuple(
                tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5],
            ));
        }

        for tuple in [
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q6_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q5_K,
            ],
            [
                GGMLType::Q5_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ],
        ] {
            assert!(!qwen_moe_llama_id_product_tuple(
                tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5],
            ));
        }
    }

    #[test]
    fn qwen_moe_native_batch_product_tuple_accepts_q8_shared_only_on_cuda() {
        let tuple = [
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q5_K,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
        ];
        assert!(qwen_moe_native_batch_product_tuple(
            tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5], true,
        ));
        assert!(!qwen_moe_native_batch_product_tuple(
            tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5], false,
        ));
    }

    #[test]
    fn qwen35_cpu_expert_major_yields_only_to_supported_llama_id() {
        assert!(!qwen35_cpu_expert_major_should_run(true, true, true));
        assert!(qwen35_cpu_expert_major_should_run(true, false, true));
        assert!(qwen35_cpu_expert_major_should_run(true, true, false));
        assert!(!qwen35_cpu_expert_major_should_run(false, false, false));
    }

    #[test]
    fn qwen_moe_prefill_outer_guard_keeps_legacy_policy() {
        assert!(!qwen_moe_prefill_outer_guard(false, false, false, false));
        assert!(qwen_moe_prefill_outer_guard(true, true, false, false));
    }

    #[test]
    fn qwen_moe_prefill_outer_guard_llama_product_bypasses_legacy_policy() {
        assert!(qwen_moe_prefill_outer_guard(false, false, true, true));
        assert!(!qwen_moe_prefill_outer_guard(false, false, false, true));
        assert!(!qwen_moe_prefill_outer_guard(false, false, true, false));
    }

    fn packed_k_rows(quant: GGMLType, rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        assert_eq!(cols % 256, 0);
        let blocks_per_row = cols / 256;
        let block_bytes = match quant {
            GGMLType::Q2_K => 84,
            GGMLType::Q3_K => 110,
            other => panic!("unexpected test quant {other:?}"),
        };
        let mut bytes = vec![0u8; rows * blocks_per_row * block_bytes];
        for row in 0..rows {
            for block_idx in 0..blocks_per_row {
                let start = (row * blocks_per_row + block_idx) * block_bytes;
                let block = &mut bytes[start..start + block_bytes];
                match quant {
                    GGMLType::Q2_K => {
                        for (i, scale) in block[..16].iter_mut().enumerate() {
                            let low = (1 + i * 7 + block_idx * 3 + row + salt) & 0x0f;
                            let high = (31 + block_idx + row + salt - i % 16) & 0x0f;
                            *scale = (low | (high << 4)) as u8;
                        }
                        for (i, q) in block[16..80].iter_mut().enumerate() {
                            *q = (i * 37 + block_idx * 19 + row * 11 + salt * 13 + 7) as u8;
                        }
                        block[80..82].copy_from_slice(
                            &half::f16::from_f32(0.015625 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                        block[82..84].copy_from_slice(
                            &half::f16::from_f32(0.0078125 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                    GGMLType::Q3_K => {
                        for (i, hmask) in block[..32].iter_mut().enumerate() {
                            *hmask = ((i * 29 + block_idx * 13 + row * 5 + salt) as u8) ^ 0xa5;
                        }
                        for (i, q) in block[32..96].iter_mut().enumerate() {
                            *q = (i * 41 + block_idx * 23 + row * 17 + salt * 11 + 3) as u8;
                        }
                        for (i, scale) in block[96..108].iter_mut().enumerate() {
                            *scale = (i * 53 + block_idx * 17 + row * 7 + salt * 5 + 3) as u8;
                        }
                        block[108..110].copy_from_slice(
                            &half::f16::from_f32(0.01171875 * (block_idx + 1) as f32)
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        bytes
    }

    #[test]
    fn hy3_q2q3_expert_major_prefill_defaults_on_and_allows_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_HY3_Q2Q3_EXPERT_MAJOR_PREFILL";
        let previous = crate::engine::policy::env_string(key);

        std::env::remove_var(key);
        assert!(hy3_q2q3_expert_major_prefill_enabled());

        std::env::set_var(key, "0");
        assert!(!hy3_q2q3_expert_major_prefill_enabled());

        std::env::set_var(key, "1");
        assert!(hy3_q2q3_expert_major_prefill_enabled());

        restore_env(key, previous);
    }

    #[test]
    fn hy3_q2q3_expert_major_requires_two_full_x4_batches_per_expert() {
        assert!(!hy3_q2q3_expert_major_has_reuse(191, 8, 192));
        assert!(hy3_q2q3_expert_major_has_reuse(192, 8, 192));
        assert!(!hy3_q2q3_expert_major_has_reuse(usize::MAX, 8, 192));
    }

    #[test]
    fn hy3_q2q3_expert_major_matches_token_major_reference() {
        let _guard = ENV_LOCK.lock();
        let batch_key = "RNB_CPU_Q2Q3_BATCH_X4";
        let previous_batch = crate::engine::policy::env_string(batch_key);
        let n_embd = 256;
        let n_ff = 256;
        let n_expert = 4;
        let n_expert_used = 2;
        let seq_len = 8;
        let router_w: Vec<f32> = (0..n_expert * n_embd)
            .map(|i| {
                let row = i / n_embd;
                let col = i % n_embd;
                ((row * 29 + col * 17 + 3) % 41) as f32 * 0.001 - 0.02
            })
            .collect();
        let selection_bias = [0.04f32, -0.03, 0.02, -0.01];
        let gate_exps = packed_k_rows(GGMLType::Q2_K, n_expert * n_ff, n_embd, 1);
        let up_exps = packed_k_rows(GGMLType::Q2_K, n_expert * n_ff, n_embd, 2);
        let down_exps = packed_k_rows(GGMLType::Q3_K, n_expert * n_embd, n_ff, 3);
        let shared_input_scale: Vec<f32> = (0..n_embd)
            .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.002)
            .collect();
        let shared_gate = packed_k_rows(GGMLType::Q2_K, n_ff, n_embd, 4);
        let shared_up = packed_k_rows(GGMLType::Q2_K, n_ff, n_embd, 5);
        let shared_down = packed_k_rows(GGMLType::Q3_K, n_embd, n_ff, 6);
        let norm_all: Vec<f32> = (0..seq_len * n_embd)
            .map(|i| ((i * 97 % 251) as f32 - 125.0) * 0.003125)
            .collect();
        let view = SharedExpertMoEView {
            router_w: &router_w,
            router_selection_bias: Some(&selection_bias),
            expert_gating_func: 2,
            expert_weights_norm: true,
            expert_weights_scale: 2.826,
            gate_exps_bytes: &gate_exps,
            gate_quant: GGMLType::Q2_K,
            up_exps_bytes: &up_exps,
            up_quant: GGMLType::Q2_K,
            down_exps_bytes: &down_exps,
            down_quant: GGMLType::Q3_K,
            shared_input_scale: &shared_input_scale,
            shared_expert_gated: true,
            shared_gate_bytes: &shared_gate,
            shared_gate_quant: GGMLType::Q2_K,
            shared_up_bytes: &shared_up,
            shared_up_quant: GGMLType::Q2_K,
            shared_down_bytes: &shared_down,
            shared_down_quant: GGMLType::Q3_K,
            n_embd,
            n_ff,
            n_expert,
            n_expert_used,
            layer_idx: None,
        };

        let mut expected = vec![0.0f32; seq_len * n_embd];
        for token in 0..seq_len {
            let start = token * n_embd;
            view.forward(
                &norm_all[start..start + n_embd],
                &mut expected[start..start + n_embd],
            );
        }
        std::env::set_var(batch_key, "1");
        let mut actual = vec![0.0f32; seq_len * n_embd];
        forward_hy3_q2q3_expert_major_moe(
            &view,
            &norm_all,
            &mut actual,
            seq_len,
            ModelArchitecture::Hy3,
            0,
            None,
        );
        restore_env(batch_key, previous_batch);

        assert_eq!(actual, expected);
    }
}
