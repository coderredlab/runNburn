#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(not(feature = "cuda"))]
use crate::engine::gemm_runtime::f32_gemv::gemv_f32;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use crate::engine::metal_runtime;
#[cfg(not(feature = "cuda"))]
use crate::engine::models::shared_expert_moe::moe_types::expert_bytes_per_row;
#[cfg(not(feature = "cuda"))]
use crate::engine::models::shared_expert_moe::routing::qwen35_softmax_topk_route;
#[cfg(not(feature = "cuda"))]
use crate::engine::norm::apply_model_gate_mul_inplace;
#[cfg(not(feature = "cuda"))]
use crate::engine::scalar_gemv::gemv_generic;
#[cfg(not(feature = "cuda"))]
use crate::engine::ModelArchitecture;
use crate::runtime::{ExpertBundleObservationReceipt, MoeRouteSlot};
#[cfg(not(feature = "cuda"))]
use rayon::prelude::*;
use rnb_loader::GGMLType;

#[derive(Debug)]
pub(in crate::engine) struct QwenPrefillRouteArrays {
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    token_ids: Vec<u32>,
}

impl QwenPrefillRouteArrays {
    pub(in crate::engine) fn new(
        expert_ids: Vec<u32>,
        route_weights: Vec<f32>,
        token_ids: Vec<u32>,
    ) -> std::result::Result<Self, String> {
        let slots = expert_ids.len();
        if route_weights.len() != slots || token_ids.len() != slots {
            return Err(format!(
                "Qwen prefill route array length mismatch: expert={} route={} token={}",
                slots,
                route_weights.len(),
                token_ids.len()
            ));
        }
        Ok(Self {
            expert_ids,
            route_weights,
            token_ids,
        })
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", not(feature = "cuda")), test))]
    pub(in crate::engine) fn len(&self) -> usize {
        self.expert_ids.len()
    }

    #[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
    pub(in crate::engine) fn validate_token_rank(
        &self,
        n_tokens: usize,
        n_expert_used: usize,
        n_expert: usize,
    ) -> std::result::Result<(), String> {
        if n_expert_used == 0 {
            return Err("Qwen token-rank route requires n_expert_used > 0".to_string());
        }
        let expected = n_tokens
            .checked_mul(n_expert_used)
            .ok_or_else(|| "Qwen token-rank route length overflow".to_string())?;
        if self.len() != expected {
            return Err(format!(
                "Qwen token-rank route length {} != {expected}",
                self.len()
            ));
        }

        for token in 0..n_tokens {
            let start = token * n_expert_used;
            for local_rank in 0..n_expert_used {
                let slot = start + local_rank;
                let expert = self.expert_ids[slot] as usize;
                if expert >= n_expert {
                    return Err(format!(
                        "invalid Qwen expert id {} at token {token} rank {local_rank}",
                        self.expert_ids[slot]
                    ));
                }
                if self.token_ids[slot] as usize != token {
                    return Err(format!(
                        "invalid Qwen token id {} at token {token} rank {local_rank}",
                        self.token_ids[slot]
                    ));
                }
                if !self.route_weights[slot].is_finite() {
                    return Err(format!(
                        "non-finite Qwen route weight at token {token} rank {local_rank}"
                    ));
                }
                if self.expert_ids[start..slot].contains(&self.expert_ids[slot]) {
                    return Err(format!(
                        "duplicate Qwen expert {} at token {token}",
                        self.expert_ids[slot]
                    ));
                }
            }
        }
        Ok(())
    }

    pub(in crate::engine) fn expert_ids(&self) -> &[u32] {
        &self.expert_ids
    }

    #[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
    pub(in crate::engine) fn route_weights(&self) -> &[f32] {
        &self.route_weights
    }

    #[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
    pub(in crate::engine) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub(in crate::engine) fn iter(&self) -> impl Iterator<Item = (u32, f32, u32)> + '_ {
        self.expert_ids
            .iter()
            .copied()
            .zip(self.route_weights.iter().copied())
            .zip(self.token_ids.iter().copied())
            .map(|((expert, weight), token)| (expert, weight, token))
    }

    pub(in crate::engine) fn sort_by_expert_token(&mut self) {
        if self.expert_ids.len() <= 1 {
            return;
        }
        if self.token_ids.windows(2).all(|pair| pair[0] <= pair[1]) {
            let max_expert = self.expert_ids.iter().copied().max().unwrap_or(0) as usize;
            let mut cursors = vec![0usize; max_expert + 2];
            for &expert in &self.expert_ids {
                cursors[expert as usize + 1] += 1;
            }
            for expert in 1..cursors.len() {
                cursors[expert] += cursors[expert - 1];
            }

            let mut sorted_experts = vec![0u32; self.expert_ids.len()];
            let mut sorted_weights = vec![0.0f32; self.route_weights.len()];
            let mut sorted_tokens = vec![0u32; self.token_ids.len()];
            for idx in 0..self.expert_ids.len() {
                let expert = self.expert_ids[idx];
                let dst = cursors[expert as usize];
                cursors[expert as usize] += 1;
                sorted_experts[dst] = expert;
                sorted_weights[dst] = self.route_weights[idx];
                sorted_tokens[dst] = self.token_ids[idx];
            }
            self.expert_ids = sorted_experts;
            self.route_weights = sorted_weights;
            self.token_ids = sorted_tokens;
            return;
        }

        let mut order = (0..self.expert_ids.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&idx| (self.expert_ids[idx], self.token_ids[idx]));
        self.expert_ids = order.iter().map(|&idx| self.expert_ids[idx]).collect();
        self.route_weights = order.iter().map(|&idx| self.route_weights[idx]).collect();
        self.token_ids = order.iter().map(|&idx| self.token_ids[idx]).collect();
    }
}

#[cfg(feature = "cuda")]
fn cuda_error(err: String) -> crate::error::LlmError {
    crate::error::LlmError::Forward(err)
}

#[cfg(not(feature = "cuda"))]
fn qwen_moe_cpu_error(message: impl Into<String>) -> crate::error::LlmError {
    crate::error::LlmError::Forward(message.into())
}

#[cfg(not(feature = "cuda"))]
fn env_falsey(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no"
    )
}

#[cfg(not(feature = "cuda"))]
fn qwen_moe_cpu_prefill_batch_enabled(seq_len: usize) -> bool {
    if seq_len <= 1 {
        return false;
    }
    match std::env::var("RNB_QWEN35_PREFILL_CPU_BATCH") {
        Ok(value) => !env_falsey(&value),
        Err(_) => true,
    }
}

#[cfg(not(feature = "cuda"))]
fn qwen35_bytes_per_row_result(
    cols: usize,
    quant: GGMLType,
    label: &str,
) -> crate::error::Result<usize> {
    match quant {
        GGMLType::F32 => Ok(cols * std::mem::size_of::<f32>()),
        GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_0 | GGMLType::IQ4_XS => {
            Ok(expert_bytes_per_row(cols, quant, label))
        }
        other => Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} unsupported quant {other:?}"
        ))),
    }
}

#[cfg(not(feature = "cuda"))]
fn validate_seq_matrix(
    label: &str,
    data_len: usize,
    rows: usize,
    cols: usize,
) -> crate::error::Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} has zero dim rows={rows} cols={cols}"
        )));
    }
    if data_len % cols != 0 {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} len {data_len} not divisible by cols {cols}"
        )));
    }
    let seq_len = data_len / cols;
    if seq_len == 0 {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} has zero seq"
        )));
    }
    Ok(seq_len)
}

#[cfg(not(feature = "cuda"))]
fn gemv_f32_bytes_seq_major(
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    label: &str,
) -> crate::error::Result<()> {
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| qwen_moe_cpu_error(format!("Qwen MoE CPU batch {label} size overflow")))?;
    if bytes.len() != expected {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} F32 byte len mismatch: got {}, expected {expected}",
            bytes.len()
        )));
    }
    if input.len() != seq_len * cols || output.len() != seq_len * rows {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} gemv shape mismatch input={} output={} rows={rows} cols={cols} seq={seq_len}",
            input.len(),
            output.len()
        )));
    }
    for s in 0..seq_len {
        let x = &input[s * cols..(s + 1) * cols];
        for r in 0..rows {
            let mut acc = 0.0f32;
            let row_off = r * cols * 4;
            for c in 0..cols {
                let off = row_off + c * 4;
                let w = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                acc += w * x[c];
            }
            output[s * rows + r] = acc;
        }
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gemv_bytes_seq_major(
    bytes: &[u8],
    quant: GGMLType,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    label: &str,
) -> crate::error::Result<()> {
    let bytes_per_row = qwen35_bytes_per_row_result(cols, quant, label)?;
    let expected = rows
        .checked_mul(bytes_per_row)
        .ok_or_else(|| qwen_moe_cpu_error(format!("Qwen MoE CPU batch {label} size overflow")))?;
    if bytes.len() != expected {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU batch {label} byte len mismatch: got {}, expected {expected}",
            bytes.len()
        )));
    }
    if quant == GGMLType::F32 {
        return gemv_f32_bytes_seq_major(bytes, input, output, rows, cols, seq_len, label);
    }
    gemv_generic(
        bytes,
        input,
        output,
        rows,
        cols,
        seq_len,
        bytes_per_row,
        quant,
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn qwen_moe_cpu_sparse_slots_from_logits(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    let selected_len = n_expert_used.min(n_expert);
    if selected_len == 0 {
        return Vec::new();
    }
    let mut slots = vec![MoeRouteSlot::new(0, 0, 0.0); seq_len * selected_len];
    slots
        .par_chunks_mut(selected_len)
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![0usize; n_expert],
                    vec![0.0f32; n_expert],
                    vec![0.0f32; selected_len],
                )
            },
            |(idx_all, probs, weights), (t, out)| {
                let logits = &router_logits[t * n_expert..(t + 1) * n_expert];
                let selected = qwen35_softmax_topk_route(
                    logits,
                    n_expert_used,
                    idx_all,
                    probs,
                    weights,
                    false,
                );
                debug_assert_eq!(selected, selected_len);
                for i in 0..selected {
                    out[i] = MoeRouteSlot::new(idx_all[i], t as u32, weights[i]);
                }
            },
        );
    slots
}

#[cfg(not(feature = "cuda"))]
fn qwen_moe_cpu_sparse_slots_expert_major_from_logits(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    let token_major =
        qwen_moe_cpu_sparse_slots_from_logits(router_logits, seq_len, n_expert, n_expert_used);
    if token_major.is_empty() {
        return token_major;
    }
    let mut counts = vec![0usize; n_expert];
    for slot in &token_major {
        counts[slot.expert] += 1;
    }
    let mut offsets = vec![0usize; n_expert];
    let mut offset = 0usize;
    for (expert, count) in counts.iter().copied().enumerate() {
        offsets[expert] = offset;
        offset += count;
    }
    let mut cursors = offsets;
    let mut expert_major = vec![MoeRouteSlot::new(0, 0, 0.0); token_major.len()];
    for slot in token_major {
        let dst = cursors[slot.expert];
        expert_major[dst] = slot;
        cursors[slot.expert] += 1;
    }
    expert_major
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn qwen_moe_prefill_shared_expert_batch_cpu(
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    if route_weights.len() != token_ids.len() || route_weights.len() != seq_len {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU shared route length mismatch weights={} tokens={} seq={seq_len}",
            route_weights.len(),
            token_ids.len()
        )));
    }
    if norm_all.len() != seq_len * n_embd {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU shared norm len mismatch: got {}, expected {}",
            norm_all.len(),
            seq_len * n_embd
        )));
    }

    let mut route_by_token = vec![0.0f32; seq_len];
    for (&token, &weight) in token_ids.iter().zip(route_weights.iter()) {
        let token = token as usize;
        if token >= seq_len {
            return Err(qwen_moe_cpu_error(format!(
                "Qwen MoE CPU shared token id {token} out of range {seq_len}"
            )));
        }
        route_by_token[token] = weight;
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let mut metal_down_out = vec![0.0f32; seq_len * n_embd];
        let used = metal_runtime::metal_qwen_moe_expert_ffn_into_if_supported(
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            norm_all,
            &mut metal_down_out,
            seq_len,
            n_embd,
            n_ff,
        )
        .map_err(qwen_moe_cpu_error)?;
        if used {
            for t in 0..seq_len {
                let route = route_by_token[t];
                for value in &mut metal_down_out[t * n_embd..(t + 1) * n_embd] {
                    *value *= route;
                }
            }
            return Ok(metal_down_out);
        }
    }

    let mut gate_out = vec![0.0f32; seq_len * n_ff];
    let mut up_out = vec![0.0f32; seq_len * n_ff];
    gemv_bytes_seq_major(
        shared_gate_bytes,
        shared_gate_quant,
        norm_all,
        &mut gate_out,
        n_ff,
        n_embd,
        seq_len,
        "shared_gate",
    )?;
    gemv_bytes_seq_major(
        shared_up_bytes,
        shared_up_quant,
        norm_all,
        &mut up_out,
        n_ff,
        n_embd,
        seq_len,
        "shared_up",
    )?;
    apply_model_gate_mul_inplace(&mut gate_out, &up_out, ModelArchitecture::Qwen35MoE);

    let mut down_out = vec![0.0f32; seq_len * n_embd];
    gemv_bytes_seq_major(
        shared_down_bytes,
        shared_down_quant,
        &gate_out,
        &mut down_out,
        n_embd,
        n_ff,
        seq_len,
        "shared_down",
    )?;

    for t in 0..seq_len {
        let route = route_by_token[t];
        for value in &mut down_out[t * n_embd..(t + 1) * n_embd] {
            *value *= route;
        }
    }
    Ok(down_out)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn qwen_moe_prefill_sparse_experts_by_token_cpu(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    let slots = route_weights.len();
    if gate.len() != slots || up.len() != slots || down.len() != slots || token_ids.len() != slots {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU sparse slice length mismatch gate={} up={} down={} weights={} tokens={}",
            gate.len(),
            up.len(),
            down.len(),
            route_weights.len(),
            token_ids.len()
        )));
    }
    if norm_all.len() != seq_len * n_embd {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU sparse norm len mismatch: got {}, expected {}",
            norm_all.len(),
            seq_len * n_embd
        )));
    }
    let mut out = vec![0.0f32; seq_len * n_embd];
    let mut gate_out = vec![0.0f32; n_ff];
    let mut up_out = vec![0.0f32; n_ff];
    let mut expert_out = vec![0.0f32; n_embd];
    for slot in 0..slots {
        let token = token_ids[slot] as usize;
        if token >= seq_len {
            return Err(qwen_moe_cpu_error(format!(
                "Qwen MoE CPU sparse token id {token} out of range {seq_len}"
            )));
        }
        let input = &norm_all[token * n_embd..(token + 1) * n_embd];
        gemv_bytes_seq_major(
            gate[slot],
            GGMLType::Q4_K,
            input,
            &mut gate_out,
            n_ff,
            n_embd,
            1,
            "sparse_gate",
        )?;
        gemv_bytes_seq_major(
            up[slot],
            GGMLType::Q4_K,
            input,
            &mut up_out,
            n_ff,
            n_embd,
            1,
            "sparse_up",
        )?;
        apply_model_gate_mul_inplace(&mut gate_out, &up_out, ModelArchitecture::Qwen35MoE);
        gemv_bytes_seq_major(
            down[slot],
            down_quant,
            &gate_out,
            &mut expert_out,
            n_embd,
            n_ff,
            1,
            "sparse_down",
        )?;
        let weight = route_weights[slot];
        for i in 0..n_embd {
            out[token * n_embd + i] += weight * expert_out[i];
        }
    }
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn qwen_moe_prefill_sparse_experts_selected_base_by_token_cpu(
    gate_all: &[u8],
    gate_quant: GGMLType,
    up_all: &[u8],
    up_quant: GGMLType,
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    let slots = expert_ids.len();
    if route_weights.len() != slots || token_ids.len() != slots {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU selected-base route length mismatch expert={} weights={} tokens={}",
            slots,
            route_weights.len(),
            token_ids.len()
        )));
    }
    if norm_all.len() != seq_len * n_embd {
        return Err(qwen_moe_cpu_error(format!(
            "Qwen MoE CPU selected-base norm len mismatch: got {}, expected {}",
            norm_all.len(),
            seq_len * n_embd
        )));
    }
    let gate_bpr = qwen35_bytes_per_row_result(n_embd, gate_quant, "sparse_gate")?;
    let up_bpr = qwen35_bytes_per_row_result(n_embd, up_quant, "sparse_up")?;
    let down_bpr = qwen35_bytes_per_row_result(n_ff, down_quant, "sparse_down")?;
    let gate_expert_bytes = n_ff
        .checked_mul(gate_bpr)
        .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU gate size overflow"))?;
    let up_expert_bytes = n_ff
        .checked_mul(up_bpr)
        .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU up size overflow"))?;
    let down_expert_bytes = n_embd
        .checked_mul(down_bpr)
        .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU down size overflow"))?;

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        if let Some(out) = metal_runtime::metal_qwen_moe_prefill_sparse_accum_if_supported(
            gate_quant,
            up_quant,
            down_quant,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            n_embd,
            n_ff,
            norm_all,
        )
        .map_err(qwen_moe_cpu_error)?
        {
            return Ok(out);
        }
    }

    let mut out = vec![0.0f32; seq_len * n_embd];
    let mut group_start = 0usize;
    while group_start < slots {
        let expert = expert_ids[group_start] as usize;
        let mut group_end = group_start + 1;
        while group_end < slots && expert_ids[group_end] as usize == expert {
            group_end += 1;
        }
        let group_len = group_end - group_start;
        let gate_start = expert
            .checked_mul(gate_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU gate offset overflow"))?;
        let up_start = expert
            .checked_mul(up_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU up offset overflow"))?;
        let down_start = expert
            .checked_mul(down_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU down offset overflow"))?;
        let gate_slice = gate_all
            .get(gate_start..gate_start + gate_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU gate expert slice out of range"))?;
        let up_slice = up_all
            .get(up_start..up_start + up_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU up expert slice out of range"))?;
        let down_slice = down_all
            .get(down_start..down_start + down_expert_bytes)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE CPU down expert slice out of range"))?;

        let mut input = vec![0.0f32; group_len * n_embd];
        for local in 0..group_len {
            let token = token_ids[group_start + local] as usize;
            if token >= seq_len {
                return Err(qwen_moe_cpu_error(format!(
                    "Qwen MoE CPU selected-base token id {token} out of range {seq_len}"
                )));
            }
            input[local * n_embd..(local + 1) * n_embd]
                .copy_from_slice(&norm_all[token * n_embd..(token + 1) * n_embd]);
        }

        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            let mut metal_down_out = vec![0.0f32; group_len * n_embd];
            let used = metal_runtime::metal_qwen_moe_expert_ffn_into_if_supported(
                gate_quant,
                up_quant,
                down_quant,
                gate_slice,
                up_slice,
                down_slice,
                &input,
                &mut metal_down_out,
                group_len,
                n_embd,
                n_ff,
            )
            .map_err(qwen_moe_cpu_error)?;
            if used {
                for local in 0..group_len {
                    let slot = group_start + local;
                    let token = token_ids[slot] as usize;
                    let weight = route_weights[slot];
                    for i in 0..n_embd {
                        out[token * n_embd + i] += weight * metal_down_out[local * n_embd + i];
                    }
                }
                group_start = group_end;
                continue;
            }
        }

        let mut gate_out = vec![0.0f32; group_len * n_ff];
        let mut up_out = vec![0.0f32; group_len * n_ff];
        gemv_bytes_seq_major(
            gate_slice,
            gate_quant,
            &input,
            &mut gate_out,
            n_ff,
            n_embd,
            group_len,
            "sparse_gate",
        )?;
        gemv_bytes_seq_major(
            up_slice,
            up_quant,
            &input,
            &mut up_out,
            n_ff,
            n_embd,
            group_len,
            "sparse_up",
        )?;
        apply_model_gate_mul_inplace(&mut gate_out, &up_out, ModelArchitecture::Qwen35MoE);

        let mut down_out = vec![0.0f32; group_len * n_embd];
        gemv_bytes_seq_major(
            down_slice,
            down_quant,
            &gate_out,
            &mut down_out,
            n_embd,
            n_ff,
            group_len,
            "sparse_down",
        )?;
        for local in 0..group_len {
            let slot = group_start + local;
            let token = token_ids[slot] as usize;
            let weight = route_weights[slot];
            for i in 0..n_embd {
                out[token * n_embd + i] += weight * down_out[local * n_embd + i];
            }
        }
        group_start = group_end;
    }
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
fn add_prefill_shared_sparse(mut sparse: Vec<f32>, shared: Vec<f32>) -> Vec<f32> {
    for (dst, value) in sparse.iter_mut().zip(shared.iter()) {
        *dst += *value;
    }
    sparse
}

pub(in crate::engine) fn qwen_moe_decode_sparse_batch_enabled(
    route_count: usize,
    all_high_precision: bool,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_batch_enabled(route_count, all_high_precision);
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "metal")]
        {
            return all_high_precision
                && route_count > 0
                && std::env::var("RNB_METAL_QWEN35_MOE_DECODE_BATCH").as_deref() == Ok("1");
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (route_count, all_high_precision);
            false
        }
    }
}

pub(in crate::engine) fn qwen_moe_prefill_enabled_for_seq(seq_len: usize) -> bool {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_enabled_for_seq(seq_len);
    }
    #[cfg(not(feature = "cuda"))]
    {
        qwen_moe_cpu_prefill_batch_enabled(seq_len)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_router_logits(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_router_logits(
            router_w, n_expert, hidden_dim, norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let seq_len = validate_seq_matrix("router_norm", norm_all.len(), n_expert, hidden_dim)?;
        if router_w.len() != n_expert * hidden_dim {
            return Err(qwen_moe_cpu_error(format!(
                "Qwen MoE CPU router weight len mismatch: got {}, expected {}",
                router_w.len(),
                n_expert * hidden_dim
            )));
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if let Some(logits) = metal_runtime::metal_qwen_moe_prefill_router_logits_if_supported(
            router_w, n_expert, hidden_dim, norm_all, seq_len,
        )
        .map_err(qwen_moe_cpu_error)?
        {
            return Ok(logits);
        }
        let mut logits = vec![0.0f32; seq_len * n_expert];
        gemv_f32(
            router_w,
            norm_all,
            &mut logits,
            n_expert,
            hidden_dim,
            seq_len,
        );
        Ok(logits)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_sparse_route_arrays_device_topk(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
    n_expert_used: usize,
) -> crate::error::Result<QwenPrefillRouteArrays> {
    #[cfg(feature = "cuda")]
    {
        let (expert_ids, route_weights, token_ids) =
            cuda_runtime::qwen_moe_prefill_sparse_route_arrays_device_topk(
                router_w,
                n_expert,
                hidden_dim,
                norm_all,
                seq_len,
                n_expert_used,
            )
            .map_err(cuda_error)?;
        return QwenPrefillRouteArrays::new(expert_ids, route_weights, token_ids)
            .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "metal")]
        if let Some((expert_ids, route_weights, token_ids)) =
            metal_runtime::metal_qwen_moe_prefill_route_arrays_if_supported(
                router_w,
                n_expert,
                hidden_dim,
                norm_all,
                seq_len,
                n_expert_used,
            )
            .map_err(qwen_moe_cpu_error)?
        {
            return QwenPrefillRouteArrays::new(expert_ids, route_weights, token_ids)
                .map_err(qwen_moe_cpu_error);
        }
        let logits = qwen_moe_prefill_router_logits(router_w, n_expert, hidden_dim, norm_all)?;
        if logits.len() != seq_len * n_expert {
            return Err(qwen_moe_cpu_error(format!(
                "Qwen MoE CPU route logits len mismatch: got {}, expected {}",
                logits.len(),
                seq_len * n_expert
            )));
        }
        let slots =
            qwen_moe_cpu_sparse_slots_from_logits(&logits, seq_len, n_expert, n_expert_used);
        let mut expert_ids = Vec::with_capacity(slots.len());
        let mut route_weights = Vec::with_capacity(slots.len());
        let mut token_ids = Vec::with_capacity(slots.len());
        for slot in slots {
            expert_ids.push(slot.expert as u32);
            route_weights.push(slot.weight);
            token_ids.push(slot.token);
        }
        QwenPrefillRouteArrays::new(expert_ids, route_weights, token_ids)
            .map_err(qwen_moe_cpu_error)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_sparse_slots(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_sparse_slots(
            router_logits,
            seq_len,
            n_expert,
            n_expert_used,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        qwen_moe_cpu_sparse_slots_from_logits(router_logits, seq_len, n_expert, n_expert_used)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_sparse_slots_expert_major(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_sparse_slots_expert_major(
            router_logits,
            seq_len,
            n_expert,
            n_expert_used,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        qwen_moe_cpu_sparse_slots_expert_major_from_logits(
            router_logits,
            seq_len,
            n_expert,
            n_expert_used,
        )
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn log_qwen_moe_prefill_route_hist(
    layer_idx: usize,
    seq_len: usize,
    n_expert: usize,
    sparse_slots: &[MoeRouteSlot],
) {
    #[cfg(feature = "cuda")]
    cuda_runtime::log_qwen_moe_prefill_route_hist(layer_idx, seq_len, n_expert, sparse_slots);
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn log_qwen_moe_prefill_route_array_hist(
    layer_idx: usize,
    seq_len: usize,
    n_expert: usize,
    route_arrays: &QwenPrefillRouteArrays,
) {
    #[cfg(feature = "cuda")]
    {
        if !crate::runtime::moe_route_hist_enabled() || !(layer_idx == 0 || layer_idx == 3) {
            return;
        }
        let mut counts = vec![0usize; n_expert];
        for &expert in route_arrays.expert_ids() {
            if let Some(count) = counts.get_mut(expert as usize) {
                *count += 1;
            }
        }
        let mut nonzero: Vec<usize> = counts.into_iter().filter(|&count| count > 0).collect();
        nonzero.sort_unstable();
        let unique = nonzero.len();
        let max = nonzero.last().copied().unwrap_or(0);
        let p50 = nonzero.get(unique / 2).copied().unwrap_or(0);
        let p90 = nonzero
            .get(unique.saturating_mul(9) / 10)
            .copied()
            .unwrap_or(0);
        eprintln!(
            "[cuda-route-hist] layer={} tokens={} slots={} unique={} max={} p50={} p90={}",
            layer_idx,
            seq_len,
            route_arrays.len(),
            unique,
            max,
            p50,
            p90
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_shared_expert_batch(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_shared_expert_batch(
            gate,
            up,
            down,
            route_weights,
            token_ids,
            seq_len,
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (gate, up, down);
        qwen_moe_prefill_shared_expert_batch_cpu(
            route_weights,
            token_ids,
            seq_len,
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_combined_f32_shared_sparse_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_by_token(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            gate,
            up,
            down,
            expert_ids,
            route_weights,
            token_ids,
            shared_route_weights,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let sparse = qwen_moe_prefill_sparse_experts_by_token_cpu(
            gate,
            up,
            down,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        let shared = qwen_moe_prefill_shared_expert_batch_cpu(
            shared_route_weights,
            token_ids,
            seq_len,
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        Ok(add_prefill_shared_sparse(sparse, shared))
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            shared_route_weights,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let sparse = qwen_moe_prefill_sparse_experts_selected_base_by_token_cpu(
            gate_all,
            GGMLType::Q4_K,
            up_all,
            GGMLType::Q4_K,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        let shared = qwen_moe_prefill_shared_expert_batch_cpu(
            shared_route_weights,
            token_ids,
            seq_len,
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        Ok(add_prefill_shared_sparse(sparse, shared))
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            shared_route_weights,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let sparse = qwen_moe_prefill_sparse_experts_selected_base_by_token_cpu(
            gate_all,
            GGMLType::Q4_K,
            up_all,
            GGMLType::Q4_K,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        let shared = qwen_moe_prefill_shared_expert_batch_cpu(
            shared_route_weights,
            token_ids,
            seq_len,
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            n_ff,
            n_embd,
            norm_all,
        )?;
        Ok(add_prefill_shared_sparse(sparse, shared))
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    shared_route_weights: &[f32],
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            norm_all,
            shared_route_weights,
            seq_len,
            n_expert_used,
            down_quant,
            n_ff,
            n_embd,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut routes = qwen_moe_prefill_sparse_route_arrays_device_topk(
            router_w,
            n_expert,
            hidden_dim,
            norm_all,
            seq_len,
            n_expert_used,
        )?;
        routes.sort_by_expert_token();
        qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            gate_all,
            up_all,
            down_all,
            routes.expert_ids(),
            &routes.route_weights,
            &routes.token_ids,
            shared_route_weights,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
#[cfg(feature = "cuda")]
pub(in crate::engine) fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: cuda_runtime::DeviceTensorId,
    input_desc: cuda_runtime::DeviceTensorDesc,
    residual_id: cuda_runtime::DeviceTensorId,
    residual_desc: cuda_runtime::DeviceTensorDesc,
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    reuse_residual_output: bool,
) -> crate::error::Result<Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>> {
    let output = if reuse_residual_output {
        cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            shared_input_scale,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            seq_len,
            n_expert_used,
            down_quant,
            n_ff,
            n_embd,
        )
    } else {
        cuda_runtime::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input(
            shared_gate_bytes,
            shared_gate_quant,
            shared_up_bytes,
            shared_up_quant,
            shared_down_bytes,
            shared_down_quant,
            shared_input_scale,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            seq_len,
            n_expert_used,
            down_quant,
            n_ff,
            n_embd,
        )
    }
    .map_err(cuda_error)?;
    Ok(output.map(|output| (output.output_id, output.output_desc)))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn qwen_moe_prefill_shared_sparse_selected_base_metal(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    gate_quant: GGMLType,
    up_all: &[u8],
    up_quant: GGMLType,
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let gate_bpr = qwen35_bytes_per_row_result(n_embd, gate_quant, "sparse_gate")?;
        let up_bpr = qwen35_bytes_per_row_result(n_embd, up_quant, "sparse_up")?;
        let down_bpr = qwen35_bytes_per_row_result(n_ff, down_quant, "sparse_down")?;
        let gate_expert_bytes = n_ff
            .checked_mul(gate_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal gate size overflow"))?;
        let up_expert_bytes = n_ff
            .checked_mul(up_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal up size overflow"))?;
        let down_expert_bytes = n_embd
            .checked_mul(down_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal down size overflow"))?;
        return metal_runtime::metal_qwen_moe_prefill_shared_sparse_accum_if_supported(
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            shared_token_ids,
            gate_quant,
            up_quant,
            down_quant,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            n_embd,
            n_ff,
            norm_all,
        )
        .map_err(qwen_moe_cpu_error);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

pub(in crate::engine) fn qwen_moe_prefill_shared_sparse_selected_base_metal_v3_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_qwen_moe_prefill_mulmmid_v3_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    false
}

pub(in crate::engine) fn qwen_moe_llama_id_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_qwen_moe_llama_id_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    false
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) fn qwen_moe_llama_id_prefill(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    selected_experts: &[u32],
    route_weights: &[f32],
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weights: &[f32],
    gate_quant: GGMLType,
    up_quant: GGMLType,
    down_quant: GGMLType,
    shared_gate_quant: GGMLType,
    shared_up_quant: GGMLType,
    shared_down_quant: GGMLType,
    norm_all: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    layer_idx: usize,
) -> std::result::Result<Option<metal_runtime::QwenMoeLlamaIdPrefillOutput>, String> {
    metal_runtime::metal_qwen_moe_llama_id_prefill(
        metal_runtime::MetalQwenMoeLlamaIdPrefillRequest {
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            selected_experts,
            route_weights,
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weights,
            gate_quant,
            up_quant,
            down_quant,
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            norm_all,
            seq_len,
            hidden_dim,
            ffn_dim,
            layer_idx,
        },
    )
}

pub(in crate::engine) fn qwen_moe_prefill_shared_sparse_selected_base_metal_v4_requested() -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_qwen_moe_prefill_mulmmid_v4_requested();
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    false
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn qwen_moe_prefill_shared_sparse_selected_base_metal_v4_token_major(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    gate_quant: GGMLType,
    up_all: &[u8],
    up_quant: GGMLType,
    down_all: &[u8],
    route_arrays: &QwenPrefillRouteArrays,
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let gate_bpr = qwen35_bytes_per_row_result(n_embd, gate_quant, "sparse_gate")?;
        let up_bpr = qwen35_bytes_per_row_result(n_embd, up_quant, "sparse_up")?;
        let down_bpr = qwen35_bytes_per_row_result(n_ff, down_quant, "sparse_down")?;
        let gate_expert_bytes = n_ff
            .checked_mul(gate_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v4 gate size overflow"))?;
        let up_expert_bytes = n_ff
            .checked_mul(up_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v4 up size overflow"))?;
        let down_expert_bytes = n_embd
            .checked_mul(down_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v4 down size overflow"))?;
        return metal_runtime::metal_qwen_moe_prefill_shared_sparse_selected_base_v4_if_supported(
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            shared_token_ids,
            gate_quant,
            up_quant,
            down_quant,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            route_arrays.expert_ids(),
            route_arrays.route_weights(),
            route_arrays.token_ids(),
            seq_len,
            n_embd,
            n_ff,
            norm_all,
        )
        .map_err(qwen_moe_cpu_error);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn qwen_moe_prefill_shared_sparse_selected_base_metal_v3_token_major(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    gate_quant: GGMLType,
    up_all: &[u8],
    up_quant: GGMLType,
    down_all: &[u8],
    route_arrays: &QwenPrefillRouteArrays,
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let gate_bpr = qwen35_bytes_per_row_result(n_embd, gate_quant, "sparse_gate")?;
        let up_bpr = qwen35_bytes_per_row_result(n_embd, up_quant, "sparse_up")?;
        let down_bpr = qwen35_bytes_per_row_result(n_ff, down_quant, "sparse_down")?;
        let gate_expert_bytes = n_ff
            .checked_mul(gate_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v3 gate size overflow"))?;
        let up_expert_bytes = n_ff
            .checked_mul(up_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v3 up size overflow"))?;
        let down_expert_bytes = n_embd
            .checked_mul(down_bpr)
            .ok_or_else(|| qwen_moe_cpu_error("Qwen MoE Metal v3 down size overflow"))?;
        return metal_runtime::metal_qwen_moe_prefill_shared_sparse_selected_base_v3_if_supported(
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            shared_token_ids,
            gate_quant,
            up_quant,
            down_quant,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            route_arrays.expert_ids(),
            route_arrays.route_weights(),
            route_arrays.token_ids(),
            seq_len,
            n_embd,
            n_ff,
            norm_all,
        )
        .map_err(qwen_moe_cpu_error);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    Ok(None)
}

#[allow(dead_code)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_register_layer(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> crate::error::Result<bool> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_register_layer(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_sparse_experts_by_token(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_sparse_experts_by_token(
            gate,
            up,
            down,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        qwen_moe_prefill_sparse_experts_by_token_cpu(
            gate,
            up,
            down,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prefill_sparse_experts_selected_base_by_token(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> crate::error::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prefill_sparse_experts_selected_base_by_token(
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
        .map_err(cuda_error);
    }
    #[cfg(not(feature = "cuda"))]
    {
        qwen_moe_prefill_sparse_experts_selected_base_by_token_cpu(
            gate_all,
            GGMLType::Q4_K,
            up_all,
            GGMLType::Q4_K,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            down_quant,
            n_ff,
            n_embd,
            norm_all,
        )
    }
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_expert(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_expert(
            gate, up, down, down_quant, n_ff, n_embd, input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_gate_up(
    gate: &[u8],
    up: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_gate_up(gate, up, n_ff, n_embd, input);
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_down(
    down_quant: GGMLType,
    down: &[u8],
    n_embd: usize,
    n_ff: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_down(down_quant, down, n_embd, n_ff, input);
    }
    #[cfg(not(feature = "cuda"))]
    None
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_prepare_selected_bundle_residency(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    n_ff: usize,
    n_embd: usize,
) -> std::result::Result<Vec<bool>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_prepare_selected_bundle_residency(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            n_ff,
            n_embd,
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_per_slot_resident(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_per_slot_resident(
            gate, up, down, down_quant, n_ff, n_embd, input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_into(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
            out,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "metal")]
        {
            let used = metal_runtime::metal_qwen_moe_decode_sparse_experts_into_if_supported(
                gate,
                up,
                down,
                route_weights,
                down_quant,
                n_ff,
                n_embd,
                input,
                out,
            )
            .map_err(|err| err.to_string())?;
            if used {
                return Ok(());
            }
        }
        Err("Metal Qwen MoE decode batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn glm_moe_decode_iq2xxs_iq3xxs_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weight: f32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
    gate_up_iq2s: bool,
    down_is_iq4xs: bool,
    shared_gate_up_q6k: bool,
    shared_down_q8_0: bool,
) -> std::result::Result<(), String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let used = metal_runtime::metal_glm_moe_decode_iq2xxs_iq3xxs_into_if_supported(
            gate,
            up,
            down,
            route_weights,
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weight,
            n_ff,
            n_embd,
            input,
            out,
            gate_up_iq2s,
            down_is_iq4xs,
            shared_gate_up_q6k,
            shared_down_q8_0,
        )
        .map_err(|err| err.to_string())?;
        if used {
            return Ok(());
        }
    }
    Err("Metal GLM MoE IQ decode batch is not available".to_string())
}

/// pm112: GLM MLA decode dense GEMV (q_a/q_b/kv_a/o) Metal 라우팅 glue.
/// 지원 안 되면 `Ok(false)` 로 CPU gemv fallback.
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn glm_mla_gemv_into(
    ggml_type: GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_mla_gemv_into_if_supported(
            ggml_type, raw, rows, cols, input, out,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

/// pm112: GLM MLA per-head Q8_0 GEMV 배치 (k_b/v_b) Metal glue.
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
pub(in crate::engine) fn glm_mla_head_gemv_into(
    ggml_type: GGMLType,
    raw: &[u8],
    heads: usize,
    rows_per_head: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_mla_head_gemv_into_if_supported(
            ggml_type,
            raw,
            heads,
            rows_per_head,
            cols,
            input,
            out,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

/// pm113: GLM MoE prefill token-batch Metal glue. slot 배열은 `[seq][sparse+1]`
/// flatten (마지막 slot 이 shared). 비지원 빌드는 Ok(false).
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn glm_moe_prefill_iq_batch_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    seq_len: usize,
    sparse_slots: usize,
    n_ff: usize,
    n_embd: usize,
    input_all: &[f32],
    out: &mut [f32],
    gate_up_iq2s: bool,
    down_is_iq4xs: bool,
    shared_gate_up_q6k: bool,
    shared_down_q8_0: bool,
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_moe_prefill_iq_batch_into_if_supported(
            gate,
            up,
            down,
            route_weights,
            seq_len,
            sparse_slots,
            n_ff,
            n_embd,
            input_all,
            out,
            gate_up_iq2s,
            down_is_iq4xs,
            shared_gate_up_q6k,
            shared_down_q8_0,
            file_regions,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

/// pm113: GLM MLA prefill slot-batch GEMV Metal glue.
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn glm_mla_head_slots_gemv_into(
    ggml_type: GGMLType,
    raw: &[u8],
    slots: usize,
    heads: usize,
    rows_per_head: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_mla_head_slots_gemv_into_if_supported(
            ggml_type,
            raw,
            slots,
            heads,
            rows_per_head,
            cols,
            input,
            out,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

/// pm112: GLM MLA front chain Metal glue.
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn glm_mla_front_into(
    qa_quant: GGMLType,
    qb_quant: GGMLType,
    kva_quant: GGMLType,
    kb_quant: GGMLType,
    qa: &[u8],
    qa_norm: &[u8],
    qb: &[u8],
    kva: &[u8],
    kb: &[u8],
    hidden: usize,
    q_rank: usize,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
    kv_rank: usize,
    q_nope: usize,
    value_dim: usize,
    eps: f32,
    input: &[f32],
    q_out: &mut [f32],
    kv_out: &mut [f32],
    qabs_out: &mut [f32],
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_mla_front_into_if_supported(
            qa_quant, qb_quant, kva_quant, kb_quant, qa, qa_norm, qb, kva, kb, hidden, q_rank,
            q_dim, kv_dim, heads, kv_rank, q_nope, value_dim, eps, input, q_out, kv_out, qabs_out,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

/// pm112: GLM MLA back chain Metal glue.
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(unused_variables)
)]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn glm_mla_back_into(
    vb_quant: GGMLType,
    o_quant: GGMLType,
    vb: &[u8],
    o: &[u8],
    hidden: usize,
    q_rank: usize,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
    kv_rank: usize,
    q_nope: usize,
    value_dim: usize,
    eps: f32,
    latent: &[f32],
    out: &mut [f32],
) -> std::result::Result<bool, String> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_glm_mla_back_into_if_supported(
            vb_quant, o_quant, vb, o, hidden, q_rank, q_dim, kv_dim, heads, kv_rank, q_nope,
            value_dim, eps, latent, out,
        )
        .map_err(|err| err.to_string());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_iq4xs_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_iq4xs_into(
            gate,
            up,
            down,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            out,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("Metal Qwen MoE IQ4_XS decode batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_add_residual_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_add_residual_into(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
            residual,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = residual;
        Err("Metal Qwen MoE decode residual batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_iq4xs_add_residual_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_iq4xs_add_residual_into(
            gate,
            up,
            down,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            residual,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = residual;
        Err("Metal Qwen MoE IQ4_XS decode residual batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_id_into(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    shared_expert_id: u32,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        let _ = (
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            shared_gate,
            shared_up,
            shared_down,
            expert_ids,
            route_weights,
            shared_expert_id,
            down_quant,
            n_ff,
            n_embd,
            input,
            out,
        );
        return Err("Metal Qwen MoE decode id batch is not available in CUDA build".to_string());
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "metal")]
        {
            let used = metal_runtime::metal_qwen_moe_decode_sparse_experts_id_into_if_supported(
                gate_all,
                up_all,
                down_all,
                gate_expert_bytes,
                up_expert_bytes,
                down_expert_bytes,
                shared_gate,
                shared_up,
                shared_down,
                expert_ids,
                route_weights,
                shared_expert_id,
                down_quant,
                n_ff,
                n_embd,
                input,
                out,
            )
            .map_err(|err| err.to_string())?;
            if used {
                return Ok(());
            }
        }
        Err("Metal Qwen MoE decode id batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_shared_sparse_experts_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: f32,
    shared_down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_shared_sparse_experts_into(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            shared_down_quant,
            n_ff,
            n_embd,
            input,
            out,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("Metal Qwen MoE shared+sparse decode batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "metal")]
        {
            let mut out = vec![0.0f32; n_embd];
            let used = metal_runtime::metal_qwen_moe_decode_sparse_experts_into_if_supported(
                gate,
                up,
                down,
                route_weights,
                down_quant,
                n_ff,
                n_embd,
                input,
                &mut out,
            )
            .map_err(|err| err.to_string())?;
            if used {
                return Ok(out);
            }
        }
        Err("Metal Qwen MoE decode batch is not available".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(in crate::engine) fn qwen_moe_decode_sparse_experts_iq4xs(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    #[cfg(feature = "cuda")]
    {
        return cuda_runtime::qwen_moe_decode_sparse_experts_iq4xs(
            gate,
            up,
            down,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
        );
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is not compiled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_env(key: &str, previous: Option<String>) {
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    fn qwen_moe_llama_id_test_call(
        gate_quant: GGMLType,
        up_quant: GGMLType,
        down_quant: GGMLType,
        shared_gate_quant: GGMLType,
        shared_up_quant: GGMLType,
        shared_down_quant: GGMLType,
    ) -> std::result::Result<Option<metal_runtime::QwenMoeLlamaIdPrefillOutput>, String> {
        qwen_moe_llama_id_prefill(
            &[],
            &[],
            &[],
            0,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[1.0],
            gate_quant,
            up_quant,
            down_quant,
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            &[0.0; 256],
            1,
            256,
            256,
            0,
        )
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_moe_llama_id_env_defaults_on_with_falsey_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let previous = std::env::var(key).ok();

        std::env::remove_var(key);
        assert!(qwen_moe_llama_id_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !qwen_moe_llama_id_requested(),
                "{value} should opt out of the llama ID carrier"
            );
        }
        for value in ["1", "true", "on", "yes", "typo"] {
            std::env::set_var(key, value);
            assert!(
                qwen_moe_llama_id_requested(),
                "{value} should keep the default carrier enabled"
            );
        }

        restore_env(key, previous);
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_moe_llama_id_unsupported_quant_preserves_none_fallback() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let previous = std::env::var(key).ok();
        std::env::set_var(key, "1");
        let unsupported = qwen_moe_llama_id_test_call(
            GGMLType::F32,
            GGMLType::Q4_K,
            GGMLType::Q5_K,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
        )
        .expect("unsupported quant should be a fallback");
        assert!(unsupported.is_none());

        let position_invalid = qwen_moe_llama_id_test_call(
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
        )
        .expect("position-invalid quant should be a fallback");
        assert!(position_invalid.is_none());
        restore_env(key, previous);
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_moe_llama_id_invalid_layout_propagates_error() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let previous = std::env::var(key).ok();
        std::env::set_var(key, "1");
        let error = match qwen_moe_llama_id_test_call(
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            GGMLType::Q5_K,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
            GGMLType::Q8_0,
        ) {
            Err(error) => error,
            Ok(_) => panic!("empty arenas and zero sparse rank must be an error"),
        };

        assert!(!error.is_empty());
        restore_env(key, previous);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn dense_moe(gate: &[f32], up: &[f32], down: &[f32], input: &[f32], n_ff: usize) -> Vec<f32> {
        let n_embd = input.len();
        let mut mid = vec![0.0f32; n_ff];
        for r in 0..n_ff {
            let gate_dot: f32 = gate[r * n_embd..(r + 1) * n_embd]
                .iter()
                .zip(input.iter())
                .map(|(a, b)| a * b)
                .sum();
            let up_dot: f32 = up[r * n_embd..(r + 1) * n_embd]
                .iter()
                .zip(input.iter())
                .map(|(a, b)| a * b)
                .sum();
            mid[r] = silu(gate_dot) * up_dot;
        }
        let mut out = vec![0.0f32; n_embd];
        for r in 0..n_embd {
            out[r] = down[r * n_ff..(r + 1) * n_ff]
                .iter()
                .zip(mid.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        out
    }

    #[test]
    fn qwen_prefill_route_arrays_reject_mismatched_lengths() {
        let err =
            QwenPrefillRouteArrays::new(vec![1], vec![0.5, 0.25], vec![0]).expect_err("bad arrays");

        assert!(err.contains("route array length mismatch"));
    }

    #[test]
    fn qwen_prefill_route_arrays_iterate_expert_weight_token_lockstep() {
        let arrays = QwenPrefillRouteArrays::new(vec![3, 1], vec![0.6, 0.4], vec![2, 0])
            .expect("route arrays");

        let rows = arrays.iter().collect::<Vec<_>>();

        assert_eq!(rows, vec![(3, 0.6, 2), (1, 0.4, 0)]);
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays.expert_ids(), &[3, 1]);
    }

    #[test]
    fn qwen_prefill_route_arrays_sort_by_expert_token_keeps_lockstep() {
        let mut arrays = QwenPrefillRouteArrays::new(
            vec![2, 1, 2, 1],
            vec![0.2, 0.7, 0.3, 0.6],
            vec![0, 1, 1, 0],
        )
        .expect("route arrays");

        arrays.sort_by_expert_token();

        assert_eq!(
            arrays.iter().collect::<Vec<_>>(),
            vec![(1, 0.6, 0), (1, 0.7, 1), (2, 0.2, 0), (2, 0.3, 1)]
        );
    }

    #[test]
    fn qwen_prefill_route_arrays_validate_token_rank_pair() {
        let routes = QwenPrefillRouteArrays::new(
            vec![2, 0, 3, 1],
            vec![0.75, 0.25, 0.6, 0.4],
            vec![0, 0, 1, 1],
        )
        .unwrap();

        assert_eq!(routes.validate_token_rank(2, 2, 4), Ok(()));
    }

    #[test]
    fn qwen_prefill_route_arrays_token_rank_borrow_preserves_none_fallback() {
        let mut generic_routes = QwenPrefillRouteArrays::new(
            vec![2, 0, 3, 1],
            vec![0.75, 0.25, 0.6, 0.4],
            vec![0, 0, 1, 1],
        )
        .unwrap();
        generic_routes.validate_token_rank(2, 2, 4).unwrap();
        let before_fallback = generic_routes.iter().collect::<Vec<_>>();

        {
            let llama_experts = generic_routes.expert_ids();
            let llama_weights = generic_routes.route_weights();
            assert_eq!(llama_experts, &[2, 0, 3, 1]);
            assert_eq!(llama_weights, &[0.75, 0.25, 0.6, 0.4]);
            let llama_output: Option<Vec<f32>> = None;
            assert!(llama_output.is_none());
        }

        assert_eq!(generic_routes.iter().collect::<Vec<_>>(), before_fallback);
        generic_routes.sort_by_expert_token();
        assert_eq!(
            generic_routes.iter().collect::<Vec<_>>(),
            vec![(0, 0.25, 0), (1, 0.4, 1), (2, 0.75, 0), (3, 0.6, 1)]
        );
    }

    #[test]
    fn qwen_prefill_route_arrays_reject_invalid_token_rank_routes() {
        let duplicate =
            QwenPrefillRouteArrays::new(vec![2, 2], vec![0.75, 0.25], vec![0, 0]).unwrap();
        assert_eq!(
            duplicate.validate_token_rank(1, 2, 4),
            Err("duplicate Qwen expert 2 at token 0".to_string())
        );

        let wrong_token =
            QwenPrefillRouteArrays::new(vec![2, 1], vec![0.75, 0.25], vec![0, 1]).unwrap();
        assert_eq!(
            wrong_token.validate_token_rank(1, 2, 4),
            Err("invalid Qwen token id 1 at token 0 rank 1".to_string())
        );

        let out_of_range =
            QwenPrefillRouteArrays::new(vec![2, 4], vec![0.75, 0.25], vec![0, 0]).unwrap();
        assert_eq!(
            out_of_range.validate_token_rank(1, 2, 4),
            Err("invalid Qwen expert id 4 at token 0 rank 1".to_string())
        );

        let non_finite =
            QwenPrefillRouteArrays::new(vec![2, 1], vec![0.75, f32::NAN], vec![0, 0]).unwrap();
        assert_eq!(
            non_finite.validate_token_rank(1, 2, 4),
            Err("non-finite Qwen route weight at token 0 rank 1".to_string())
        );
    }

    #[test]
    fn qwen_prefill_route_arrays_reject_invalid_token_rank_shape() {
        let routes = QwenPrefillRouteArrays::new(vec![2, 1], vec![0.75, 0.25], vec![0, 0]).unwrap();
        assert_eq!(
            routes.validate_token_rank(2, 2, 4),
            Err("Qwen token-rank route length 2 != 4".to_string())
        );
        assert_eq!(
            routes.validate_token_rank(1, 0, 4),
            Err("Qwen token-rank route requires n_expert_used > 0".to_string())
        );
        assert_eq!(
            routes.validate_token_rank(usize::MAX, 2, 4),
            Err("Qwen token-rank route length overflow".to_string())
        );
    }

    #[test]
    fn qwen_prefill_route_arrays_sort_uses_token_major_counting_path() {
        let mut arrays = QwenPrefillRouteArrays::new(
            vec![2, 1, 2, 0, 1, 0],
            vec![0.2, 0.7, 0.3, 0.4, 0.6, 0.5],
            vec![0, 0, 1, 1, 2, 2],
        )
        .expect("route arrays");

        arrays.sort_by_expert_token();

        assert_eq!(
            arrays.iter().collect::<Vec<_>>(),
            vec![
                (0, 0.4, 1),
                (0, 0.5, 2),
                (1, 0.7, 0),
                (1, 0.6, 2),
                (2, 0.2, 0),
                (2, 0.3, 1)
            ]
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn qwen_moe_prefill_cpu_batch_defaults_on_and_allows_opt_out() {
        let _guard = ENV_LOCK.lock();
        let key = "RNB_QWEN35_PREFILL_CPU_BATCH";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert!(!qwen_moe_prefill_enabled_for_seq(1));
        assert!(qwen_moe_prefill_enabled_for_seq(2));

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !qwen_moe_prefill_enabled_for_seq(8),
                "value {value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(qwen_moe_prefill_enabled_for_seq(8));

        restore_env(key, previous);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn qwen_moe_prefill_cpu_route_arrays_match_token_major_topk() {
        let router_w = [
            1.0f32, 0.0, // expert 0
            0.0, 1.0, // expert 1
            2.0, 0.0, // expert 2
            0.0, 2.0, // expert 3
        ];
        let norm_all = [
            1.0f32, 0.0, // token 0
            0.0, 1.0, // token 1
        ];

        let routes =
            qwen_moe_prefill_sparse_route_arrays_device_topk(&router_w, 4, 2, &norm_all, 2, 2)
                .unwrap();

        assert_eq!(routes.expert_ids(), &[2, 0, 3, 1]);
        assert_eq!(routes.token_ids, &[0, 0, 1, 1]);
        for chunk in routes.route_weights.chunks_exact(2) {
            assert!((chunk.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(chunk[0] > chunk[1]);
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn qwen_moe_prefill_cpu_shared_expert_batch_matches_dense_reference() {
        let n_ff = 2;
        let n_embd = 2;
        let seq_len = 2;
        let gate = [1.0f32, 0.0, 0.0, 1.0];
        let up = [2.0f32, 0.0, 0.0, 3.0];
        let down = [1.0f32, 1.0, 2.0, -1.0];
        let norm_all = [1.0f32, 2.0, 3.0, -1.0];
        let route_weights = [0.5f32, 0.25];
        let token_ids = [0u32, 1];

        let got = qwen_moe_prefill_shared_expert_batch(
            &[],
            &[],
            &[],
            &route_weights,
            &token_ids,
            seq_len,
            &f32_bytes(&gate),
            GGMLType::F32,
            &f32_bytes(&up),
            GGMLType::F32,
            &f32_bytes(&down),
            GGMLType::F32,
            n_ff,
            n_embd,
            &norm_all,
        )
        .unwrap();

        let mut expected = Vec::new();
        for t in 0..seq_len {
            let mut out = dense_moe(
                &gate,
                &up,
                &down,
                &norm_all[t * n_embd..(t + 1) * n_embd],
                n_ff,
            );
            for value in &mut out {
                *value *= route_weights[t];
            }
            expected.extend(out);
        }
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "shared idx={i} got={g} expected={e}");
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn qwen_moe_prefill_cpu_sparse_selected_base_batches_by_expert() {
        let n_ff = 2;
        let n_embd = 2;
        let seq_len = 2;
        let e0_gate = [1.0f32, 0.0, 0.0, 1.0];
        let e0_up = [2.0f32, 0.0, 0.0, 3.0];
        let e0_down = [1.0f32, 1.0, 2.0, -1.0];
        let e1_gate = [1.0f32, 1.0, -1.0, 2.0];
        let e1_up = [0.5f32, -1.0, 1.5, 0.25];
        let e1_down = [0.25f32, 2.0, -1.0, 0.5];
        let gate_all = [e0_gate, e1_gate].concat();
        let up_all = [e0_up, e1_up].concat();
        let down_all = [e0_down, e1_down].concat();
        let norm_all = [1.0f32, 2.0, 3.0, -1.0];
        let expert_ids = [0u32, 1, 1];
        let route_weights = [0.5f32, 0.25, 0.75];
        let token_ids = [0u32, 0, 1];

        let got = qwen_moe_prefill_sparse_experts_selected_base_by_token_cpu(
            &f32_bytes(&gate_all),
            GGMLType::F32,
            &f32_bytes(&up_all),
            GGMLType::F32,
            &f32_bytes(&down_all),
            &expert_ids,
            &route_weights,
            &token_ids,
            seq_len,
            GGMLType::F32,
            n_ff,
            n_embd,
            &norm_all,
        )
        .unwrap();

        let mut expected = vec![0.0f32; seq_len * n_embd];
        let expert_gate = [&e0_gate[..], &e1_gate[..]];
        let expert_up = [&e0_up[..], &e1_up[..]];
        let expert_down = [&e0_down[..], &e1_down[..]];
        for slot in 0..expert_ids.len() {
            let expert = expert_ids[slot] as usize;
            let token = token_ids[slot] as usize;
            let out = dense_moe(
                expert_gate[expert],
                expert_up[expert],
                expert_down[expert],
                &norm_all[token * n_embd..(token + 1) * n_embd],
                n_ff,
            );
            for i in 0..n_embd {
                expected[token * n_embd + i] += route_weights[slot] * out[i];
            }
        }
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "sparse idx={i} got={g} expected={e}");
        }
    }
}
