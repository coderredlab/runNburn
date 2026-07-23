use super::*;
use crate::runtime::{cuda_decode_moe_combined_enabled, ExpertBundleObservationReceipt};
#[inline]
fn gate_bytes_for<'a>(
    view: &'a SharedExpertMoEView<'a>,
    expert: usize,
    per_gate: usize,
) -> &'a [u8] {
    &view.gate_exps_bytes[expert * per_gate..(expert + 1) * per_gate]
}
#[inline]
fn up_bytes_for<'a>(view: &'a SharedExpertMoEView<'a>, expert: usize, per_up: usize) -> &'a [u8] {
    &view.up_exps_bytes[expert * per_up..(expert + 1) * per_up]
}
#[inline]
fn down_bytes_for<'a>(
    view: &'a SharedExpertMoEView<'a>,
    expert: usize,
    per_down: usize,
) -> &'a [u8] {
    &view.down_exps_bytes[expert * per_down..(expert + 1) * per_down]
}

#[cfg(target_arch = "aarch64")]
fn sparse_pair_batch_direct_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT").is_some_and(
        |value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        },
    )
}

#[cfg(target_arch = "aarch64")]
fn sparse_q6_down_q8k_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_DECODE_Q6_DOWN_Q8K")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(cfg!(target_os = "android"))
}

#[cfg(target_arch = "aarch64")]
fn compute_sparse_q6_down_q8k(
    down_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    bytes_per_row: usize,
) -> bool {
    if cols % 256 != 0
        || output.len() < rows
        || !crate::engine::quantized_dispatch::aarch64_dotprod_available()
    {
        return false;
    }

    let n_blocks = cols / 256;
    let mut inline_q8k = [crate::engine::quantized_dispatch::QuantizedQ8KBlock::default(); 2];
    let mut overflow_q8k = Vec::new();
    let input_q8k = if n_blocks <= inline_q8k.len() {
        &mut inline_q8k[..n_blocks]
    } else {
        overflow_q8k.resize(
            n_blocks,
            crate::engine::quantized_dispatch::QuantizedQ8KBlock::default(),
        );
        &mut overflow_q8k
    };
    crate::engine::quantized_dispatch::quantize_q8k_input_into(input, input_q8k);
    crate::engine::quantized_dispatch::dispatch_q6k_q8k_gemv(
        down_bytes,
        input_q8k,
        output,
        rows,
        cols,
        1,
        bytes_per_row,
    );
    true
}

#[cfg(target_arch = "aarch64")]
fn compute_sparse_pair_batch_direct(
    view: &SharedExpertMoEView<'_>,
    h_q8k: &[crate::engine::quantized_dispatch::QuantizedQ8KBlock],
    idx: &[usize],
    exps: &[f32],
    gate_bpr: usize,
    up_bpr: usize,
    down_bpr: usize,
    per_gate: usize,
    per_up: usize,
    per_down: usize,
    n_ff: usize,
    n_embd: usize,
    down_quant: GGMLType,
    profile_enabled: bool,
) -> Option<Vec<ExpertProfileAcc>> {
    if idx.is_empty()
        || idx.len() != exps.len()
        || !crate::engine::quantized_dispatch::aarch64_dotprod_available()
    {
        return None;
    }

    let total_start = Instant::now();
    let slot_count = idx.len();
    let total_gate_rows = slot_count * n_ff;
    let n_blocks = n_embd / 256;
    if h_q8k.len() != n_blocks {
        return None;
    }

    let mut gate_out = vec![0.0f32; total_gate_rows];
    let mut up_out = vec![0.0f32; total_gate_rows];
    let gate_up_start = profile_enabled.then(Instant::now);
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = if total_gate_rows <= 64 {
        total_gate_rows
    } else {
        ((total_gate_rows + n_threads * 4 - 1) / (n_threads * 4)).max(64)
    };
    gate_out
        .par_chunks_mut(chunk)
        .zip(up_out.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(chunk_idx, (gate_chunk, up_chunk))| {
            let row_base = chunk_idx * chunk;
            for i in 0..gate_chunk.len() {
                let global_row = row_base + i;
                let slot = global_row / n_ff;
                let row = global_row - slot * n_ff;
                let expert = idx[slot];
                let gate_offset = expert * per_gate + row * gate_bpr;
                let up_offset = expert * per_up + row * up_bpr;
                let gate_row = &view.gate_exps_bytes[gate_offset..gate_offset + gate_bpr];
                let up_row = &view.up_exps_bytes[up_offset..up_offset + up_bpr];
                gate_chunk[i] =
                    crate::engine::quantized_dispatch::dot_q4k_q8k(gate_row, h_q8k, n_blocks);
                up_chunk[i] =
                    crate::engine::quantized_dispatch::dot_q4k_q8k(up_row, h_q8k, n_blocks);
            }
        });
    gate_out
        .par_chunks_mut(n_ff)
        .zip(up_out.par_chunks(n_ff))
        .for_each(|(gate, up)| {
            apply_model_gate_mul_inplace(gate, up, ModelArchitecture::Qwen35MoE);
        });
    let high_gate_up_us = gate_up_start
        .map(|start| start.elapsed().as_micros())
        .unwrap_or(0);

    let down_start = profile_enabled.then(Instant::now);
    let mut per_expert: Vec<ExpertProfileAcc> = (0..slot_count)
        .into_par_iter()
        .map(|slot| {
            let expert = idx[slot];
            let down_offset = expert * per_down;
            let down_slice = &view.down_exps_bytes[down_offset..down_offset + per_down];
            let gate = &gate_out[slot * n_ff..(slot + 1) * n_ff];
            let mut expert_out = vec![0.0f32; n_embd];
            gemv_generic(
                down_slice,
                gate,
                &mut expert_out,
                n_embd,
                n_ff,
                1,
                down_bpr,
                down_quant,
            );
            let weight = exps[slot];
            for value in &mut expert_out {
                *value *= weight;
            }
            ExpertProfileAcc {
                out: expert_out,
                wall_us: 0,
                high_us: 0,
                high_gate_up_us: 0,
                high_down_us: 0,
                low_us: 0,
                low_gate_up_us: 0,
                low_gate_up_row_us: 0,
                low_gate_up_tile_us: 0,
                low_gate_up_post_us: 0,
                low_shadow_down_us: 0,
                low_base_down_us: 0,
                high: 1,
                low: 0,
                skip: 0,
            }
        })
        .collect();
    let high_down_us = down_start
        .map(|start| start.elapsed().as_micros())
        .unwrap_or(0);
    let elapsed_us = total_start.elapsed().as_micros();
    let profile_total = per_expert
        .first_mut()
        .expect("non-empty sparse batch must produce expert outputs");
    profile_total.wall_us = elapsed_us;
    profile_total.high_us = elapsed_us;
    profile_total.high_gate_up_us = high_gate_up_us;
    profile_total.high_down_us = high_down_us;

    Some(per_expert)
}

pub(super) enum SparseFanoutResult {
    Complete,
    ResidualComplete,
    Computed {
        per_expert: Vec<ExpertProfileAcc>,
        fanout_us: u128,
        shared_in_sparse_gpu: bool,
    },
}

pub(super) fn glm_iq_sparse_cuda_supported(
    view: &SharedExpertMoEView<'_>,
    prefer_sparse_moe_cuda: bool,
) -> bool {
    prefer_sparse_moe_cuda
        && view.gate_quant == GGMLType::IQ2_XXS
        && view.up_quant == GGMLType::IQ2_XXS
        && view.down_quant == GGMLType::IQ3_XXS
}

#[derive(Clone, Copy)]
struct SparseCpuGeometry {
    gate_bpr: usize,
    up_bpr: usize,
    down_bpr: usize,
    per_gate: usize,
    per_up: usize,
    per_down: usize,
    n_ff: usize,
    n_embd: usize,
    down_quant: GGMLType,
    q4k_sparse_cuda_supported: bool,
    high_q2q3_matrix: bool,
}

fn compute_sparse_expert_cpu(
    view: &SharedExpertMoEView<'_>,
    h: &[f32],
    expert: usize,
    route_weight: f32,
    geometry: SparseCpuGeometry,
    profile_enabled: bool,
    inner_gemv: bool,
    #[cfg(target_arch = "aarch64")] sparse_q6_down_q8k: bool,
    #[cfg(target_arch = "aarch64")] expert_local_rows: bool,
    #[cfg(target_arch = "aarch64")] gate_up_pair_h_q8k: Option<
        &[crate::engine::quantized_dispatch::QuantizedQ8KBlock],
    >,
) -> ExpertProfileAcc {
    let SparseCpuGeometry {
        gate_bpr,
        up_bpr,
        down_bpr,
        per_gate,
        per_up,
        per_down,
        n_ff,
        n_embd,
        down_quant,
        q4k_sparse_cuda_supported,
        high_q2q3_matrix,
    } = geometry;
    let compute_start = Instant::now();

    if q4k_sparse_cuda_supported {
        let gate_slice = gate_bytes_for(view, expert, per_gate);
        let up_slice = up_bytes_for(view, expert, per_up);
        let down_slice = down_bytes_for(view, expert, per_down);
        if let Some(mut expert_out) = qwen_moe_backend::qwen_moe_decode_expert(
            gate_slice, up_slice, down_slice, down_quant, n_ff, n_embd, h,
        ) {
            for value in &mut expert_out {
                *value *= route_weight;
            }
            let elapsed_us = compute_start.elapsed().as_micros();
            return ExpertProfileAcc {
                out: expert_out,
                wall_us: elapsed_us,
                high_us: elapsed_us,
                high_gate_up_us: 0,
                high_down_us: 0,
                low_us: 0,
                low_gate_up_us: 0,
                low_gate_up_row_us: 0,
                low_gate_up_tile_us: 0,
                low_gate_up_post_us: 0,
                low_shadow_down_us: 0,
                low_base_down_us: 0,
                high: 1,
                low: 0,
                skip: 0,
            };
        }
    }

    let mut gate_up_scratch = vec![0f32; n_ff * 2];
    let (gate_out, up_out) = gate_up_scratch.split_at_mut(n_ff);
    let high_gate_up_start = profile_enabled.then(Instant::now);
    let gate_slice = gate_bytes_for(view, expert, per_gate);
    let up_slice = up_bytes_for(view, expert, per_up);
    if high_q2q3_matrix {
        gemv_generic(
            gate_slice,
            h,
            gate_out,
            n_ff,
            n_embd,
            1,
            gate_bpr,
            view.gate_quant,
        );
        gemv_generic(up_slice, h, up_out, n_ff, n_embd, 1, up_bpr, view.up_quant);
    } else {
        let gpu_done = if q4k_sparse_cuda_supported {
            if let Some((gate_vec, up_vec)) =
                qwen_moe_backend::qwen_moe_decode_gate_up(gate_slice, up_slice, n_ff, n_embd, h)
            {
                gate_out.copy_from_slice(&gate_vec);
                up_out.copy_from_slice(&up_vec);
                true
            } else {
                false
            }
        } else {
            false
        };
        if !gpu_done {
            #[cfg(target_arch = "aarch64")]
            let pair_done = gate_up_pair_h_q8k.is_some_and(|h_q8k| {
                crate::engine::quantized_dispatch::dispatch_q4k_pair_q8k_prequantized(
                    gate_slice,
                    up_slice,
                    h_q8k,
                    gate_out,
                    up_out,
                    n_ff,
                    n_embd,
                    gate_bpr,
                    up_bpr,
                    expert_local_rows,
                )
            });
            #[cfg(not(target_arch = "aarch64"))]
            let pair_done = false;

            if !pair_done && inner_gemv {
                gemv_generic(
                    gate_slice,
                    h,
                    gate_out,
                    n_ff,
                    n_embd,
                    1,
                    gate_bpr,
                    view.gate_quant,
                );
                gemv_generic(up_slice, h, up_out, n_ff, n_embd, 1, up_bpr, view.up_quant);
            } else if !pair_done {
                for row in 0..n_ff {
                    let row_bytes = &gate_slice[row * gate_bpr..(row + 1) * gate_bpr];
                    gate_out[row] =
                        dot_k_block_row(row_bytes, h, n_embd, gate_bpr, view.gate_quant);
                }
                for row in 0..n_ff {
                    let row_bytes = &up_slice[row * up_bpr..(row + 1) * up_bpr];
                    up_out[row] = dot_k_block_row(row_bytes, h, n_embd, up_bpr, view.up_quant);
                }
            }
        }
    }
    apply_model_gate_mul_inplace(gate_out, up_out, ModelArchitecture::Qwen35MoE);
    let high_gate_up_us = high_gate_up_start
        .map(|start| start.elapsed().as_micros())
        .unwrap_or(0);

    let high_down_start = profile_enabled.then(Instant::now);
    let down_slice = down_bytes_for(view, expert, per_down);
    let mut expert_out = vec![0f32; n_embd];
    if high_q2q3_matrix {
        gemv_generic(
            down_slice,
            gate_out,
            &mut expert_out,
            n_embd,
            n_ff,
            1,
            down_bpr,
            down_quant,
        );
    } else {
        let gpu_done = if q4k_sparse_cuda_supported {
            if let Some(down_vec) = qwen_moe_backend::qwen_moe_decode_down(
                down_quant, down_slice, n_embd, n_ff, gate_out,
            ) {
                expert_out.copy_from_slice(&down_vec);
                true
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(target_arch = "aarch64")]
        let q6_down_done = !gpu_done
            && sparse_q6_down_q8k
            && down_quant == GGMLType::Q6_K
            && compute_sparse_q6_down_q8k(
                down_slice,
                gate_out,
                &mut expert_out,
                n_embd,
                n_ff,
                down_bpr,
            );
        #[cfg(not(target_arch = "aarch64"))]
        let q6_down_done = false;

        if !gpu_done && !q6_down_done {
            if inner_gemv {
                gemv_generic(
                    down_slice,
                    gate_out,
                    &mut expert_out,
                    n_embd,
                    n_ff,
                    1,
                    down_bpr,
                    down_quant,
                );
            } else {
                for row in 0..n_embd {
                    let row_bytes = &down_slice[row * down_bpr..(row + 1) * down_bpr];
                    expert_out[row] =
                        dot_k_block_row(row_bytes, gate_out, n_ff, down_bpr, down_quant);
                }
            }
        }
    }
    let high_down_us = high_down_start
        .map(|start| start.elapsed().as_micros())
        .unwrap_or(0);

    for value in &mut expert_out {
        *value *= route_weight;
    }
    let elapsed_us = compute_start.elapsed().as_micros();
    ExpertProfileAcc {
        out: expert_out,
        wall_us: elapsed_us,
        high_us: elapsed_us,
        high_gate_up_us,
        high_down_us,
        low_us: 0,
        low_gate_up_us: 0,
        low_gate_up_row_us: 0,
        low_gate_up_tile_us: 0,
        low_gate_up_post_us: 0,
        low_shadow_down_us: 0,
        low_base_down_us: 0,
        high: 1,
        low: 0,
        skip: 0,
    }
}

pub(super) fn glm_iq_metal_batch_eligible(
    view: &SharedExpertMoEView<'_>,
    selected_experts: usize,
) -> bool {
    let gate_up_ok = (view.gate_quant == GGMLType::IQ2_XXS && view.up_quant == GGMLType::IQ2_XXS)
        || (view.gate_quant == GGMLType::IQ2_S && view.up_quant == GGMLType::IQ2_S)
        || (view.gate_quant == GGMLType::IQ3_XXS && view.up_quant == GGMLType::IQ3_XXS);
    let down_ok = matches!(view.down_quant, GGMLType::IQ3_XXS | GGMLType::IQ4_XS);
    let shared_ok = (view.shared_gate_quant == GGMLType::Q5_K
        && view.shared_up_quant == GGMLType::Q5_K
        && view.shared_down_quant == GGMLType::Q6_K)
        || (view.shared_gate_quant == GGMLType::Q6_K
            && view.shared_up_quant == GGMLType::Q6_K
            && view.shared_down_quant == GGMLType::Q8_0)
        || (view.shared_gate_quant == GGMLType::Q8_0
            && view.shared_up_quant == GGMLType::Q8_0
            && view.shared_down_quant == GGMLType::Q8_0);
    gate_up_ok && down_ok && shared_ok && selected_experts <= 8
}

pub(super) fn compute_sparse_fanout(
    view: &SharedExpertMoEView<'_>,
    h: &[f32],
    out: &mut [f32],
    residual: Option<&mut [f32]>,
    idx: &[usize],
    exps: &[f32],
    gate_scalar: f32,
    profile_enabled: bool,
    prefer_sparse_moe_cuda: bool,
) -> SparseFanoutResult {
    // pm123: MTP nextn 레이어는 unrouted slot 을 usize::MAX sentinel 로 채운다.
    // 어떤 fanout 경로(IQ gather, batched gate_bytes_for)든 OOB 없이 valid expert 만
    // 보도록 진입부에서 sentinel 을 제거한다 (unrouted = 기여 0, 의미 동등).
    let (sentinel_idx_buf, sentinel_exps_buf);
    let (idx, exps): (&[usize], &[f32]) = if idx.iter().any(|&e| e >= view.n_expert) {
        sentinel_idx_buf = idx
            .iter()
            .copied()
            .filter(|&e| e < view.n_expert)
            .collect::<Vec<usize>>();
        sentinel_exps_buf = idx
            .iter()
            .zip(exps.iter())
            .filter(|(&e, _)| e < view.n_expert)
            .map(|(_, &w)| w)
            .collect::<Vec<f32>>();
        (sentinel_idx_buf.as_slice(), sentinel_exps_buf.as_slice())
    } else {
        (idx, exps)
    };
    // Per-expert: quant types come from the GGUF tensor metadata.
    // Parallelize across the selected sparse experts.
    let gate_bpr = expert_bytes_per_row(view.n_embd, view.gate_quant, "gate_exps");
    let up_bpr = expert_bytes_per_row(view.n_embd, view.up_quant, "up_exps");
    let down_bpr = down_bytes_per_row(view.n_ff, view.down_quant);
    let per_gate = view.n_ff * gate_bpr;
    let per_up = view.n_ff * up_bpr;
    let per_down = view.n_embd * down_bpr;
    let n_ff = view.n_ff;
    let n_embd = view.n_embd;
    let down_quant = view.down_quant;
    let inner_gemv = qwen_moe_decode_inner_gemv_enabled();
    #[cfg(target_arch = "aarch64")]
    let expert_local_rows = qwen_moe_decode_expert_local_rows_enabled();
    #[cfg(target_arch = "aarch64")]
    let sparse_q6_down_q8k =
        sparse_q6_down_q8k_enabled() && down_quant == GGMLType::Q6_K && n_ff % 256 == 0;
    #[cfg(target_arch = "aarch64")]
    let gate_up_pair_gemv = inner_gemv
        && qwen_moe_decode_gate_up_pair_gemv_enabled()
        && view.gate_quant == GGMLType::Q4_K
        && view.up_quant == GGMLType::Q4_K
        && view.n_embd % 256 == 0;
    let q4k_sparse_cuda_supported = cfg!(feature = "cuda")
        && view.gate_quant == GGMLType::Q4_K
        && view.up_quant == GGMLType::Q4_K
        && matches!(down_quant, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K);
    let iq4xs_sparse_cuda_supported = cfg!(feature = "cuda")
        && view.gate_quant == GGMLType::IQ4_XS
        && view.up_quant == GGMLType::IQ4_XS
        && matches!(
            down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::IQ4_XS
        );
    let glm_iq_sparse_cuda_supported = glm_iq_sparse_cuda_supported(view, prefer_sparse_moe_cuda);
    let high_q2q3_matrix = view.gate_quant == GGMLType::Q2_K
        && view.up_quant == GGMLType::Q2_K
        && down_quant == GGMLType::Q3_K;
    let q2q3_sparse_cuda_supported = prefer_sparse_moe_cuda && high_q2q3_matrix;
    let batched_sparse_cuda_supported = q4k_sparse_cuda_supported
        || iq4xs_sparse_cuda_supported
        || q2q3_sparse_cuda_supported
        || glm_iq_sparse_cuda_supported;
    let glm_iq_metal_supported = glm_iq_metal_batch_eligible(view, idx.len());
    let cpu_geometry = SparseCpuGeometry {
        gate_bpr,
        up_bpr,
        down_bpr,
        per_gate,
        per_up,
        per_down,
        n_ff,
        n_embd,
        down_quant,
        q4k_sparse_cuda_supported,
        high_q2q3_matrix,
    };

    #[cfg(target_arch = "aarch64")]
    let gate_up_pair_h_q8k = if gate_up_pair_gemv {
        let mut h_q8k = vec![
            crate::engine::quantized_dispatch::QuantizedQ8KBlock::default();
            view.n_embd / 256
        ];
        crate::engine::quantized_dispatch::quantize_q8k_input_into(h, &mut h_q8k);
        Some(h_q8k)
    } else {
        None
    };

    let fanout_start = Instant::now();
    let mut batched_sparse_out = None;

    let mut shared_in_sparse_gpu = false;
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    if glm_iq_metal_supported
        && idx.iter().all(|&e| e < view.n_expert)
        && view.gate_exps_bytes.len() >= view.n_expert * per_gate
        && view.up_exps_bytes.len() >= view.n_expert * per_up
        && view.down_exps_bytes.len() >= view.n_expert * per_down
    {
        let empty: &[u8] = &[];
        let mut gate_slices = [empty; 8];
        let mut up_slices = [empty; 8];
        let mut down_slices = [empty; 8];
        for (slot, &expert) in idx.iter().enumerate() {
            gate_slices[slot] = &view.gate_exps_bytes[expert * per_gate..(expert + 1) * per_gate];
            up_slices[slot] = &view.up_exps_bytes[expert * per_up..(expert + 1) * per_up];
            down_slices[slot] = &view.down_exps_bytes[expert * per_down..(expert + 1) * per_down];
        }
        if qwen_moe_backend::glm_moe_decode_iq2xxs_iq3xxs_into(
            &gate_slices[..idx.len()],
            &up_slices[..idx.len()],
            &down_slices[..idx.len()],
            exps,
            view.shared_gate_bytes,
            view.shared_up_bytes,
            view.shared_down_bytes,
            gate_scalar,
            n_ff,
            n_embd,
            h,
            out,
            view.gate_quant == GGMLType::IQ2_S,
            view.down_quant == GGMLType::IQ4_XS,
            view.shared_gate_quant == GGMLType::Q6_K,
            view.shared_down_quant == GGMLType::Q8_0,
            view.gate_quant == GGMLType::IQ3_XXS,
            view.shared_gate_quant == GGMLType::Q8_0,
        )
        .is_ok()
        {
            let fanout_us = fanout_start.elapsed().as_micros();
            if profile_enabled {
                record_moe_profile(
                    "qwen35moe:decode:high_compute",
                    std::time::Duration::from_micros(fanout_us.min(u64::MAX as u128) as u64),
                );
                record_moe_counts("qwen35moe:decode", idx.len() as u64, 0, 0);
            }
            if let Some(residual) = residual {
                for (dst, &value) in residual.iter_mut().zip(out.iter()) {
                    *dst += value;
                }
                return SparseFanoutResult::ResidualComplete;
            }
            return SparseFanoutResult::Complete;
        }
    }
    if batched_sparse_cuda_supported {
        if !qwen_moe_backend::qwen_moe_decode_sparse_batch_enabled(idx.len(), true) {
            panic!("CUDA sparse MoE is required for a CUDA-supported expert tuple");
        }
        let empty: &[u8] = &[];
        let mut gate_slices_stack = [empty; 32];
        let mut up_slices_stack = [empty; 32];
        let mut down_slices_stack = [empty; 32];
        let mut expert_ids_stack = [0u32; 32];
        let mut route_weights_stack = [0.0f32; 32];
        let mut slot_count = 0usize;
        for (&e, &w) in idx.iter().zip(exps.iter()) {
            gate_slices_stack[slot_count] = &view.gate_exps_bytes[e * per_gate..(e + 1) * per_gate];
            up_slices_stack[slot_count] = &view.up_exps_bytes[e * per_up..(e + 1) * per_up];
            down_slices_stack[slot_count] = &view.down_exps_bytes[e * per_down..(e + 1) * per_down];
            expert_ids_stack[slot_count] = e as u32;
            route_weights_stack[slot_count] = w;
            slot_count += 1;
        }
        let shared_sparse_quant_supported = if q4k_sparse_cuda_supported {
            view.shared_gate_quant == GGMLType::Q4_K
                && view.shared_up_quant == GGMLType::Q4_K
                && view.shared_down_quant == down_quant
        } else if iq4xs_sparse_cuda_supported {
            view.shared_gate_quant == GGMLType::IQ4_XS
                && view.shared_up_quant == GGMLType::IQ4_XS
                && view.shared_down_quant == down_quant
        } else {
            view.shared_gate_quant == GGMLType::Q2_K
                && view.shared_up_quant == GGMLType::Q2_K
                && view.shared_down_quant == GGMLType::Q3_K
        };
        if shared_sparse_quant_supported && slot_count < gate_slices_stack.len() {
            gate_slices_stack[slot_count] = view.shared_gate_bytes;
            up_slices_stack[slot_count] = view.shared_up_bytes;
            down_slices_stack[slot_count] = view.shared_down_bytes;
            expert_ids_stack[slot_count] = view.n_expert as u32;
            route_weights_stack[slot_count] = gate_scalar;
            slot_count += 1;
            shared_in_sparse_gpu = true;
        }
        if shared_in_sparse_gpu && residual.is_none() {
            let fanout_result = if qwen_moe_backend::qwen_moe_decode_sparse_experts_id_into(
                view.gate_exps_bytes,
                view.up_exps_bytes,
                view.down_exps_bytes,
                per_gate,
                per_up,
                per_down,
                view.shared_gate_bytes,
                view.shared_up_bytes,
                view.shared_down_bytes,
                &expert_ids_stack[..slot_count],
                &route_weights_stack[..slot_count],
                view.n_expert as u32,
                down_quant,
                n_ff,
                n_embd,
                h,
                out,
            )
            .is_ok()
            {
                Some(SparseFanoutResult::Complete)
            } else {
                None
            };
            if let Some(result) = fanout_result {
                let fanout_us = fanout_start.elapsed().as_micros();
                if profile_enabled {
                    record_moe_profile(
                        "qwen35moe:decode:high_compute",
                        std::time::Duration::from_micros(fanout_us.min(u64::MAX as u128) as u64),
                    );
                    record_moe_counts("qwen35moe:decode", idx.len() as u64, 0, 0);
                }
                return result;
            }
        }
        if shared_in_sparse_gpu {
            let fanout_result = if let Some(residual) = residual {
                if iq4xs_sparse_cuda_supported {
                    qwen_moe_backend::qwen_moe_decode_sparse_experts_iq4xs_add_residual_into(
                        &gate_slices_stack[..slot_count],
                        &up_slices_stack[..slot_count],
                        &down_slices_stack[..slot_count],
                        &route_weights_stack[..slot_count],
                        down_quant,
                        n_ff,
                        n_embd,
                        h,
                        residual,
                    )
                    .map(|_| SparseFanoutResult::ResidualComplete)
                } else {
                    qwen_moe_backend::qwen_moe_decode_sparse_experts_add_residual_into(
                        &gate_slices_stack[..slot_count],
                        &up_slices_stack[..slot_count],
                        &down_slices_stack[..slot_count],
                        &route_weights_stack[..slot_count],
                        view.layer_idx,
                        idx,
                        &mut bundle_observation_receipt,
                        down_quant,
                        n_ff,
                        n_embd,
                        h,
                        residual,
                    )
                    .map(|_| SparseFanoutResult::ResidualComplete)
                }
            } else {
                if iq4xs_sparse_cuda_supported {
                    qwen_moe_backend::qwen_moe_decode_sparse_experts_iq4xs_into(
                        &gate_slices_stack[..slot_count],
                        &up_slices_stack[..slot_count],
                        &down_slices_stack[..slot_count],
                        &route_weights_stack[..slot_count],
                        down_quant,
                        n_ff,
                        n_embd,
                        h,
                        out,
                    )
                    .map(|_| SparseFanoutResult::Complete)
                } else {
                    qwen_moe_backend::qwen_moe_decode_sparse_experts_into(
                        &gate_slices_stack[..slot_count],
                        &up_slices_stack[..slot_count],
                        &down_slices_stack[..slot_count],
                        &route_weights_stack[..slot_count],
                        view.layer_idx,
                        idx,
                        &mut bundle_observation_receipt,
                        down_quant,
                        n_ff,
                        n_embd,
                        h,
                        out,
                    )
                    .map(|_| SparseFanoutResult::Complete)
                }
            };
            if let Ok(result) = fanout_result {
                let fanout_us = fanout_start.elapsed().as_micros();
                if profile_enabled {
                    record_moe_profile(
                        "qwen35moe:decode:high_compute",
                        std::time::Duration::from_micros(fanout_us.min(u64::MAX as u128) as u64),
                    );
                    record_moe_counts("qwen35moe:decode", idx.len() as u64, 0, 0);
                }
                return result;
            }
        }
        if !shared_in_sparse_gpu
            && cuda_decode_moe_combined_enabled()
            && view.shared_gate_quant == GGMLType::Q4_K
            && view.shared_up_quant == GGMLType::Q4_K
            && qwen_moe_backend::qwen_moe_decode_shared_sparse_experts_into(
                &gate_slices_stack[..slot_count],
                &up_slices_stack[..slot_count],
                &down_slices_stack[..slot_count],
                &route_weights_stack[..slot_count],
                view.layer_idx,
                idx,
                &mut bundle_observation_receipt,
                down_quant,
                view.shared_gate_bytes,
                view.shared_up_bytes,
                view.shared_down_bytes,
                gate_scalar,
                view.shared_down_quant,
                n_ff,
                n_embd,
                h,
                out,
            )
            .is_ok()
        {
            let fanout_us = fanout_start.elapsed().as_micros();
            if profile_enabled {
                record_moe_profile(
                    "qwen35moe:decode:high_compute",
                    std::time::Duration::from_micros(fanout_us.min(u64::MAX as u128) as u64),
                );
                record_moe_counts("qwen35moe:decode", idx.len() as u64, 0, 0);
            }
            return SparseFanoutResult::Complete;
        }
        let sparse_result = if glm_iq_sparse_cuda_supported {
            qwen_moe_backend::glm_moe_decode_sparse_experts_iq2xxs_iq3xxs(
                &gate_slices_stack[..slot_count],
                &up_slices_stack[..slot_count],
                &down_slices_stack[..slot_count],
                &route_weights_stack[..slot_count],
                n_ff,
                n_embd,
                h,
            )
        } else if iq4xs_sparse_cuda_supported {
            qwen_moe_backend::qwen_moe_decode_sparse_experts_iq4xs(
                &gate_slices_stack[..slot_count],
                &up_slices_stack[..slot_count],
                &down_slices_stack[..slot_count],
                &route_weights_stack[..slot_count],
                down_quant,
                n_ff,
                n_embd,
                h,
            )
        } else {
            qwen_moe_backend::qwen_moe_decode_sparse_experts(
                &gate_slices_stack[..slot_count],
                &up_slices_stack[..slot_count],
                &down_slices_stack[..slot_count],
                &route_weights_stack[..slot_count],
                view.layer_idx,
                idx,
                &mut bundle_observation_receipt,
                down_quant,
                n_ff,
                n_embd,
                h,
            )
        };
        let gpu_sparse_out =
            sparse_result.unwrap_or_else(|err| panic!("CUDA sparse MoE execution failed: {err}"));
        batched_sparse_out = Some(gpu_sparse_out);
    }
    #[cfg(feature = "cuda")]
    if batched_sparse_out.is_none() {
        let run = |quant, raw: &[u8], rows, cols, input: &[f32], label: &str| {
            crate::engine::cuda_runtime::decode_gemv(quant, raw, rows, cols, input)
                .unwrap_or_else(|| {
                    panic!("CUDA {label} {quant:?} GEMV is unavailable; CPU fallback is disabled")
                })
                .unwrap_or_else(|err| {
                    panic!("CUDA {label} {quant:?} GEMV failed; CPU fallback is disabled: {err}")
                })
        };
        let mut sparse_out = vec![0.0f32; n_embd];
        for (slot, &expert) in idx.iter().enumerate() {
            let gate_bytes = gate_bytes_for(view, expert, per_gate);
            let up_bytes = up_bytes_for(view, expert, per_up);
            let down_bytes = down_bytes_for(view, expert, per_down);
            let mut gate = run(
                view.gate_quant,
                &gate_bytes,
                n_ff,
                n_embd,
                h,
                "sparse expert gate",
            );
            let up = run(
                view.up_quant,
                &up_bytes,
                n_ff,
                n_embd,
                h,
                "sparse expert up",
            );
            apply_model_gate_mul_inplace(&mut gate, &up, ModelArchitecture::Qwen35MoE);
            let mut expert_out = run(
                view.down_quant,
                &down_bytes,
                n_embd,
                n_ff,
                &gate,
                "sparse expert down",
            );
            scale_f32_inplace(&mut expert_out, exps[slot]);
            add_f32_inplace(&mut sparse_out, &expert_out);
        }
        batched_sparse_out = Some(sparse_out);
    }
    let per_expert: Vec<ExpertProfileAcc> = if let Some(out) = batched_sparse_out {
        let elapsed_us = fanout_start.elapsed().as_micros();
        vec![ExpertProfileAcc {
            out,
            wall_us: elapsed_us,
            high_us: elapsed_us,
            high_gate_up_us: 0,
            high_down_us: 0,
            low_us: 0,
            low_gate_up_us: 0,
            low_gate_up_row_us: 0,
            low_gate_up_tile_us: 0,
            low_gate_up_post_us: 0,
            low_shadow_down_us: 0,
            low_base_down_us: 0,
            high: idx.len() as u64,
            low: 0,
            skip: 0,
        }]
    } else if let Some(batch_accs) = {
        #[cfg(target_arch = "aarch64")]
        {
            if sparse_pair_batch_direct_enabled() {
                gate_up_pair_h_q8k.as_deref().and_then(|h_q8k| {
                    compute_sparse_pair_batch_direct(
                        view,
                        h_q8k,
                        idx,
                        exps,
                        gate_bpr,
                        up_bpr,
                        down_bpr,
                        per_gate,
                        per_up,
                        per_down,
                        n_ff,
                        n_embd,
                        down_quant,
                        profile_enabled,
                    )
                })
            } else {
                None
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            None
        }
    } {
        batch_accs
    } else {
        idx.par_iter()
            .enumerate()
            .map(|(slot, &expert)| {
                compute_sparse_expert_cpu(
                    view,
                    h,
                    expert,
                    exps[slot],
                    cpu_geometry,
                    profile_enabled,
                    inner_gemv,
                    #[cfg(target_arch = "aarch64")]
                    sparse_q6_down_q8k,
                    #[cfg(target_arch = "aarch64")]
                    expert_local_rows,
                    #[cfg(target_arch = "aarch64")]
                    gate_up_pair_h_q8k.as_deref(),
                )
            })
            .collect()
    };

    let fanout_us = fanout_start.elapsed().as_micros();
    SparseFanoutResult::Computed {
        per_expert,
        fanout_us,
        shared_in_sparse_gpu,
    }
}

#[cfg(test)]
#[cfg(all(test, target_arch = "aarch64"))]
mod aarch64_tests {
    use super::*;

    fn encode_q4_k_scales_mins(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
        let mut out = [0u8; 12];
        for i in 0..4 {
            out[i] = scales[i] & 63;
            out[i + 4] = mins[i] & 63;
        }
        for i in 4..8 {
            out[i + 4] = (scales[i] & 0x0f) | ((mins[i] & 0x0f) << 4);
            out[i - 4] |= ((scales[i] >> 4) & 0x03) << 6;
            out[i] |= ((mins[i] >> 4) & 0x03) << 6;
        }
        out
    }

    fn make_q4_k_row(seed: usize) -> Vec<u8> {
        let mut row = vec![0u8; 144];
        let d = half::f16::from_f32(0.00025 * ((seed % 5 + 1) as f32));
        row[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        row[2..4].copy_from_slice(&0u16.to_le_bytes());
        let scales = std::array::from_fn(|i| ((seed * 7 + i * 5) % 31 + 1) as u8);
        row[4..16].copy_from_slice(&encode_q4_k_scales_mins(scales, [0; 8]));
        for i in 0..128 {
            let lo = ((seed * 3 + i * 5 + 1) % 16) as u8;
            let hi = ((seed * 11 + i * 7 + 3) % 16) as u8;
            row[16 + i] = lo | (hi << 4);
        }
        row
    }

    fn make_q6_k_row(seed: usize) -> Vec<u8> {
        let mut row = vec![0u8; 210];
        for i in 0..128 {
            row[i] = ((seed * 13 + i * 7 + 5) % 256) as u8;
        }
        for i in 0..64 {
            row[128 + i] = ((seed * 11 + i * 5 + 17) % 256) as u8;
        }
        for i in 0..16 {
            row[192 + i] = (((seed + i * 3) % 7) as i8 - 3) as u8;
        }
        let d = half::f16::from_f32(0.0005 * ((seed % 3 + 1) as f32));
        row[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
        row
    }

    #[test]
    fn sparse_q6_down_q8k_matches_row_oracle() {
        if !crate::engine::quantized_dispatch::aarch64_dotprod_available() {
            return;
        }

        let rows = 17;
        let blocks = 2;
        let cols = blocks * 256;
        let bytes_per_row = blocks * 210;
        let mut down_bytes = Vec::with_capacity(rows * bytes_per_row);
        for row in 0..rows {
            for block in 0..blocks {
                down_bytes.extend_from_slice(&make_q6_k_row(row * blocks + block));
            }
        }
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.019).cos() * 0.125)
            .collect();
        let mut actual = vec![0.0f32; rows];

        assert!(compute_sparse_q6_down_q8k(
            &down_bytes,
            &input,
            &mut actual,
            rows,
            cols,
            bytes_per_row,
        ));

        let mut input_q8k =
            vec![crate::engine::quantized_dispatch::QuantizedQ8KBlock::default(); blocks];
        crate::engine::quantized_dispatch::quantize_q8k_input_into(&input, &mut input_q8k);
        for (row, &got) in actual.iter().enumerate() {
            let row_bytes = &down_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let expected =
                crate::engine::quantized_dispatch::dot_q6k_q8k(row_bytes, &input_q8k, blocks);
            assert_eq!(got.to_bits(), expected.to_bits(), "Q6_K down row {row}");
        }
    }

    fn computed_outputs(result: SparseFanoutResult) -> Vec<ExpertProfileAcc> {
        match result {
            SparseFanoutResult::Computed { per_expert, .. } => per_expert,
            SparseFanoutResult::Complete | SparseFanoutResult::ResidualComplete => {
                panic!("CPU direct fanout must return expert outputs")
            }
        }
    }

    #[test]
    fn sparse_batch_direct_preserves_route_slot_outputs() {
        if !crate::engine::quantized_dispatch::aarch64_dotprod_available() {
            return;
        }
        let _guard = crate::engine::moe::tests::env_lock()
            .lock()
            .expect("MoE env lock poisoned");
        let n_embd = 256;
        let n_ff = 256;
        let n_expert = 8;
        let mut gate_exps_bytes = Vec::with_capacity(n_expert * n_ff * 144);
        let mut up_exps_bytes = Vec::with_capacity(n_expert * n_ff * 144);
        let mut down_exps_bytes = Vec::with_capacity(n_expert * n_embd * 210);
        for expert in 0..n_expert {
            for row in 0..n_ff {
                gate_exps_bytes.extend_from_slice(&make_q4_k_row(expert * n_ff + row));
                up_exps_bytes.extend_from_slice(&make_q4_k_row(10_000 + expert * n_ff + row));
            }
            for row in 0..n_embd {
                down_exps_bytes.extend_from_slice(&make_q6_k_row(expert * n_embd + row));
            }
        }
        let view = SharedExpertMoEView {
            router_w: &[],
            router_selection_bias: None,
            expert_gating_func: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            gate_exps_bytes: &gate_exps_bytes,
            gate_quant: GGMLType::Q4_K,
            up_exps_bytes: &up_exps_bytes,
            up_quant: GGMLType::Q4_K,
            down_exps_bytes: &down_exps_bytes,
            down_quant: GGMLType::Q6_K,
            shared_input_scale: &[],
            shared_expert_gated: false,
            shared_gate_bytes: &[],
            shared_gate_quant: GGMLType::Q8_0,
            shared_up_bytes: &[],
            shared_up_quant: GGMLType::Q8_0,
            shared_down_bytes: &[],
            shared_down_quant: GGMLType::Q8_0,
            n_embd,
            n_ff,
            n_expert,
            n_expert_used: n_expert,
            layer_idx: None,
        };
        let h: Vec<f32> = (0..n_embd)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1)
            .collect();
        let idx = [7, 0, 5, 2, 6, 1, 4, 3];
        let exps = [0.31, 0.27, 0.19, 0.11, 0.07, 0.03, 0.015, 0.005];
        let mut scratch = vec![0.0f32; n_embd];
        let previous =
            crate::engine::policy::env_os_string("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT");

        unsafe {
            std::env::remove_var("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT");
        }
        let baseline = computed_outputs(compute_sparse_fanout(
            &view,
            &h,
            &mut scratch,
            None,
            &idx,
            &exps,
            1.0,
            false,
            false,
        ));
        unsafe {
            std::env::set_var("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT", "1");
        }
        let candidate = computed_outputs(compute_sparse_fanout(
            &view,
            &h,
            &mut scratch,
            None,
            &idx,
            &exps,
            1.0,
            false,
            false,
        ));
        unsafe {
            match previous {
                Some(value) => {
                    std::env::set_var("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT", value)
                }
                None => std::env::remove_var("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT"),
            }
        }

        assert_eq!(candidate.len(), baseline.len());
        for (slot, (expected, actual)) in baseline.iter().zip(&candidate).enumerate() {
            for (row, (&expected, &actual)) in expected.out.iter().zip(&actual.out).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "route slot {slot} output row {row}"
                );
            }
        }
    }
}
