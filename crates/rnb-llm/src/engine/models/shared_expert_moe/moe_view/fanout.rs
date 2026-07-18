use super::*;
use crate::runtime::{
    cuda_cache_trace_enabled, cuda_decode_moe_combined_enabled,
    cuda_q2k_q3k_mixed_resident_cpu_enabled, ExpertBundleCacheStats,
    ExpertBundleObservationReceipt,
};
use std::borrow::Cow;

/// in4 helpers: when the view carries a v3 sidecar residency view, expert
/// byte slices come from `expert_bytes(rank)` (hot/cold dispatch baked in);
/// otherwise we index the flat `*_exps_bytes` slice as before. Mixed-precision
/// shadow paths still use the legacy slice — they are not residency-wired
/// in this phase.
///
/// Returns `Cow` so disk-pread sources (`ColdReader`) can yield owned `Vec<u8>`
/// while mmap/RAM sources stay zero-copy. Callers deref the Cow to `&[u8]` for
/// the GEMV inner loop; the owned buffer (if any) lives until the end of the
/// per-expert closure.
#[inline]
fn gate_bytes_for<'a>(
    view: &'a SharedExpertMoEView<'a>,
    e: usize,
    per_gate: usize,
) -> Cow<'a, [u8]> {
    if let Some(r) = view.gate_residency {
        r.expert_bytes(e)
    } else {
        Cow::Borrowed(&view.gate_exps_bytes[e * per_gate..(e + 1) * per_gate])
    }
}
#[inline]
fn up_bytes_for<'a>(view: &'a SharedExpertMoEView<'a>, e: usize, per_up: usize) -> Cow<'a, [u8]> {
    if let Some(r) = view.up_residency {
        r.expert_bytes(e)
    } else {
        Cow::Borrowed(&view.up_exps_bytes[e * per_up..(e + 1) * per_up])
    }
}
#[inline]
fn down_bytes_for<'a>(
    view: &'a SharedExpertMoEView<'a>,
    e: usize,
    per_down: usize,
) -> Cow<'a, [u8]> {
    if let Some(r) = view.down_residency {
        r.expert_bytes(e)
    } else {
        Cow::Borrowed(&view.down_exps_bytes[e * per_down..(e + 1) * per_down])
    }
}

#[cfg(target_arch = "aarch64")]
fn sparse_pair_batch_direct_enabled() -> bool {
    std::env::var("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
}

#[cfg(target_arch = "aarch64")]
fn sparse_q6_down_q8k_enabled() -> bool {
    std::env::var("RNB_QWEN35_MOE_DECODE_Q6_DOWN_Q8K")
        .ok()
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
    if cols % 256 != 0 || output.len() < rows || !std::arch::is_aarch64_feature_detected!("dotprod")
    {
        return false;
    }

    let n_blocks = cols / 256;
    let mut inline_q8k = [crate::engine::gemm_runtime::Q8KBlock::default(); 2];
    let mut overflow_q8k = Vec::new();
    let input_q8k = if n_blocks <= inline_q8k.len() {
        &mut inline_q8k[..n_blocks]
    } else {
        overflow_q8k.resize(n_blocks, crate::engine::gemm_runtime::Q8KBlock::default());
        &mut overflow_q8k
    };
    crate::engine::gemm_runtime::quantize_input_q8k_into(input, input_q8k);
    crate::engine::gemm_runtime::neon_dot::gemv_q6_k_int8(
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
    h_q8k: &[crate::engine::gemm_runtime::Q8KBlock],
    idx: &[usize],
    exps: &[f32],
    precisions: &[MoePrecision],
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
        || idx.len() != precisions.len()
        || !precisions.iter().all(|&p| p == MoePrecision::High)
        || view.gate_residency.is_some()
        || view.up_residency.is_some()
        || view.down_residency.is_some()
        || !std::arch::is_aarch64_feature_detected!("dotprod")
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
                gate_chunk[i] = unsafe {
                    crate::engine::gemm_runtime::neon_dot::dot_q4_k_q8k_neon(
                        gate_row, h_q8k, n_blocks,
                    )
                };
                up_chunk[i] = unsafe {
                    crate::engine::gemm_runtime::neon_dot::dot_q4_k_q8k_neon(
                        up_row, h_q8k, n_blocks,
                    )
                };
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

fn should_try_q2q3_mixed_resident_cpu(
    enabled: bool,
    q2q3_sparse_cuda_supported: bool,
    layer_idx: Option<usize>,
    precisions: &[MoePrecision],
) -> bool {
    enabled
        && q2q3_sparse_cuda_supported
        && layer_idx.is_some()
        && precisions
            .iter()
            .all(|&precision| precision == MoePrecision::High)
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
    q2k_gu_bpr: usize,
    per_q2k_gu: usize,
    q2k_dn_bpr: usize,
    per_q2k_dn: usize,
}

fn skipped_expert_profile(n_embd: usize) -> ExpertProfileAcc {
    ExpertProfileAcc {
        out: vec![0.0f32; n_embd],
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
        high: 0,
        low: 0,
        skip: 1,
    }
}

fn compute_sparse_expert_cpu(
    view: &SharedExpertMoEView<'_>,
    h: &[f32],
    expert: usize,
    route_weight: f32,
    precision: MoePrecision,
    low_gate_up_path: LowGateUpPath,
    geometry: SparseCpuGeometry,
    profile_enabled: bool,
    inner_gemv: bool,
    #[cfg(target_arch = "aarch64")] sparse_q6_down_q8k: bool,
    #[cfg(target_arch = "aarch64")] expert_local_rows: bool,
    #[cfg(target_arch = "aarch64")] gate_up_pair_h_q8k: Option<
        &[crate::engine::gemm_runtime::Q8KBlock],
    >,
) -> ExpertProfileAcc {
    if precision == MoePrecision::Skip {
        return skipped_expert_profile(geometry.n_embd);
    }

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
        q2k_gu_bpr,
        per_q2k_gu,
        q2k_dn_bpr,
        per_q2k_dn,
    } = geometry;
    let use_shadow = precision == MoePrecision::Low;
    let compute_start = Instant::now();
    if !use_shadow && q4k_sparse_cuda_supported {
        let gate_cow = gate_bytes_for(view, expert, per_gate);
        let up_cow = up_bytes_for(view, expert, per_up);
        let down_cow = down_bytes_for(view, expert, per_down);
        let gate_slice: &[u8] = &gate_cow;
        let up_slice: &[u8] = &up_cow;
        let down_slice: &[u8] = &down_cow;
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
    let mut low_gate_up_us = 0u128;
    let mut low_gate_up_row_us = 0u128;
    let mut low_gate_up_tile_us = 0u128;
    let mut low_gate_up_post_us = 0u128;
    let mut low_shadow_down_us = 0u128;
    let mut low_base_down_us = 0u128;
    let high_gate_up_start = (!use_shadow && profile_enabled).then(Instant::now);
    let mut high_gate_up_us = 0u128;
    let mut high_down_us = 0u128;
    if use_shadow {
        let low_gate_up_compute_start = Instant::now();
        if low_gate_up_path == LowGateUpPath::TileMajor {
            let tile = view
                .shadow_gate_up_tile_bytes
                .expect("MoE low-precision tile path needs shadow_gate_up_tile_bytes");
            let per_tile = q2k_gate_up_tile_bytes_per_expert(n_ff, n_embd);
            let tile_slice = &tile[expert * per_tile..(expert + 1) * per_tile];
            let (gate_vec, up_vec) = gemv_q2k_gate_up_tile(tile_slice, h, n_ff, n_embd);
            gate_out.copy_from_slice(&gate_vec);
            up_out.copy_from_slice(&up_vec);
            low_gate_up_tile_us = low_gate_up_compute_start.elapsed().as_micros();
        } else {
            let shadow_gate = view
                .shadow_gate_bytes
                .expect("MoE low-precision row path needs shadow_gate_bytes");
            let shadow_up = view
                .shadow_up_bytes
                .expect("MoE low-precision row path needs shadow_up_bytes");
            let gate_slice = &shadow_gate[expert * per_q2k_gu..(expert + 1) * per_q2k_gu];
            let up_slice = &shadow_up[expert * per_q2k_gu..(expert + 1) * per_q2k_gu];
            for row in 0..n_ff {
                let row_bytes = &gate_slice[row * q2k_gu_bpr..(row + 1) * q2k_gu_bpr];
                gate_out[row] = dot_k_block_row(row_bytes, h, n_embd, q2k_gu_bpr, GGMLType::Q2_K);
            }
            for row in 0..n_ff {
                let row_bytes = &up_slice[row * q2k_gu_bpr..(row + 1) * q2k_gu_bpr];
                up_out[row] = dot_k_block_row(row_bytes, h, n_embd, q2k_gu_bpr, GGMLType::Q2_K);
            }
            low_gate_up_row_us = low_gate_up_compute_start.elapsed().as_micros();
        }
    } else {
        let gate_cow = gate_bytes_for(view, expert, per_gate);
        let up_cow = up_bytes_for(view, expert, per_up);
        let gate_slice: &[u8] = &gate_cow;
        let up_slice: &[u8] = &up_cow;
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
                    if expert_local_rows {
                        crate::engine::gemm_runtime::quant_gemv::gemv_q4k_pair_aarch64_q8k_prequantized_serial(
                            gate_slice, up_slice, h_q8k, gate_out, up_out, n_ff, n_embd, gate_bpr,
                            up_bpr,
                        )
                    } else {
                        crate::engine::gemm_runtime::quant_gemv::gemv_q4k_pair_aarch64_q8k_prequantized(
                            gate_slice, up_slice, h_q8k, gate_out, up_out, n_ff, n_embd, gate_bpr,
                            up_bpr,
                        )
                    }
                });
                #[cfg(not(target_arch = "aarch64"))]
                let pair_done = false;

                if pair_done {
                    // paired path filled gate_out and up_out
                } else if inner_gemv {
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
    }
    let low_gate_up_post_start = use_shadow.then(Instant::now);
    apply_model_gate_mul_inplace(gate_out, up_out, ModelArchitecture::Qwen35MoE);
    if let Some(start) = low_gate_up_post_start {
        low_gate_up_post_us = start.elapsed().as_micros();
        low_gate_up_us = low_gate_up_row_us + low_gate_up_tile_us + low_gate_up_post_us;
    }
    if let Some(start) = high_gate_up_start {
        high_gate_up_us = start.elapsed().as_micros();
    }

    let mut expert_out = vec![0f32; n_embd];
    if use_shadow {
        if let Some(shadow_down) = view.shadow_down_bytes {
            let low_shadow_down_start = Instant::now();
            let down_slice = &shadow_down[expert * per_q2k_dn..(expert + 1) * per_q2k_dn];
            for row in 0..n_embd {
                let row_bytes = &down_slice[row * q2k_dn_bpr..(row + 1) * q2k_dn_bpr];
                expert_out[row] =
                    dot_k_block_row(row_bytes, gate_out, n_ff, q2k_dn_bpr, GGMLType::Q2_K);
            }
            low_shadow_down_us = low_shadow_down_start.elapsed().as_micros();
        } else {
            let low_base_down_start = Instant::now();
            let down_slice = down_bytes_for(view, expert, per_down);
            for row in 0..n_embd {
                let row_bytes = &down_slice[row * down_bpr..(row + 1) * down_bpr];
                expert_out[row] = dot_k_block_row(row_bytes, gate_out, n_ff, down_bpr, down_quant);
            }
            low_base_down_us = low_base_down_start.elapsed().as_micros();
        }
    } else {
        let high_down_start = profile_enabled.then(Instant::now);
        let down_cow = down_bytes_for(view, expert, per_down);
        let down_slice: &[u8] = &down_cow;
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
        if let Some(start) = high_down_start {
            high_down_us = start.elapsed().as_micros();
        }
    }

    for value in &mut expert_out {
        *value *= route_weight;
    }
    let elapsed_us = compute_start.elapsed().as_micros();
    ExpertProfileAcc {
        out: expert_out,
        wall_us: elapsed_us,
        high_us: if use_shadow { 0 } else { elapsed_us },
        high_gate_up_us: if use_shadow { 0 } else { high_gate_up_us },
        high_down_us: if use_shadow { 0 } else { high_down_us },
        low_us: if use_shadow { elapsed_us } else { 0 },
        low_gate_up_us,
        low_gate_up_row_us,
        low_gate_up_tile_us,
        low_gate_up_post_us,
        low_shadow_down_us,
        low_base_down_us,
        high: if use_shadow { 0 } else { 1 },
        low: if use_shadow { 1 } else { 0 },
        skip: 0,
    }
}

fn partial_resident_slots(mask: &[bool]) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut hits = Vec::with_capacity(mask.len());
    let mut misses = Vec::with_capacity(mask.len());
    for (slot, &resident) in mask.iter().enumerate() {
        if resident {
            hits.push(slot);
        } else {
            misses.push(slot);
        }
    }
    (!hits.is_empty() && !misses.is_empty()).then_some((hits, misses))
}

struct PartialMixedFanoutResult {
    per_expert: Vec<ExpertProfileAcc>,
    gpu_slots: usize,
    cpu_slots: usize,
    gpu_us: u128,
    cpu_us: u128,
    wall_us: u128,
}

fn should_emit_partial_mixed_trace(
    trace_enabled: bool,
    mixed_enabled: bool,
    mixed_succeeded: bool,
    gpu_slots: usize,
    cpu_slots: usize,
) -> bool {
    trace_enabled && mixed_enabled && mixed_succeeded && gpu_slots > 0 && cpu_slots > 0
}

fn format_partial_mixed_trace(
    layer_idx: usize,
    mixed: &PartialMixedFanoutResult,
    bundle_stats: ExpertBundleCacheStats,
) -> String {
    format!(
        "[cuda-cache] qwen35_mixed_fanout layer={} gpu_slots={} cpu_slots={} gpu_us={} cpu_us={} wall_us={} overlap_hint={} q2q3_bundle_lookups={} q2q3_bundle_hits={} q2q3_bundle_partial_hits={} q2q3_bundle_misses={} q2q3_bundle_admissions={} q2q3_bundle_admitted_bytes={} q2q3_bundle_evictions={} q2q3_bundle_evicted_bytes={} q2q3_bundle_h2d_mb={:.2} q2q3_bundle_temp_h2d_mb={:.2} q2q3_bundle_h2d_bytes_per_token={:.1}",
        layer_idx,
        mixed.gpu_slots,
        mixed.cpu_slots,
        mixed.gpu_us,
        mixed.cpu_us,
        mixed.wall_us,
        mixed.wall_us <= mixed.gpu_us + mixed.cpu_us,
        bundle_stats.bundle_lookups,
        bundle_stats.bundle_hits,
        bundle_stats.bundle_partial_hits,
        bundle_stats.bundle_misses,
        bundle_stats.bundle_admissions,
        bundle_stats.admitted_bytes,
        bundle_stats.bundle_evictions,
        bundle_stats.evicted_bytes,
        bundle_stats.h2d_bytes as f64 / (1024.0 * 1024.0),
        bundle_stats.temp_h2d_bytes as f64 / (1024.0 * 1024.0),
        bundle_stats.h2d_bytes_per_token(1),
    )
}

fn reassemble_mixed_slots(
    slot_count: usize,
    n_embd: usize,
    route_weights: &[f32],
    hit_slots: &[usize],
    gpu_unweighted: Vec<f32>,
    gpu_elapsed_us: u128,
    cpu_slots: Vec<(usize, ExpertProfileAcc)>,
) -> Option<Vec<ExpertProfileAcc>> {
    if route_weights.len() != slot_count
        || gpu_unweighted.len() != hit_slots.len().checked_mul(n_embd)?
    {
        return None;
    }

    let mut slots: Vec<Option<ExpertProfileAcc>> = (0..slot_count).map(|_| None).collect();
    for (hit_index, (&slot, unweighted)) in hit_slots
        .iter()
        .zip(gpu_unweighted.chunks_exact(n_embd))
        .enumerate()
    {
        let target = slots.get_mut(slot)?;
        if target.is_some() {
            return None;
        }
        let mut weighted = unweighted.to_vec();
        for value in &mut weighted {
            *value *= route_weights[slot];
        }
        let elapsed_us = if hit_index == 0 { gpu_elapsed_us } else { 0 };
        *target = Some(ExpertProfileAcc {
            out: weighted,
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
        });
    }
    for (slot, expert) in cpu_slots {
        let target = slots.get_mut(slot)?;
        if target.is_some() {
            return None;
        }
        *target = Some(expert);
    }
    slots.into_iter().collect()
}

fn run_partial_mixed_fanout<G, C>(
    mask: &[bool],
    n_embd: usize,
    route_weights: &[f32],
    gpu: G,
    cpu: C,
) -> Option<PartialMixedFanoutResult>
where
    G: FnOnce(&[usize]) -> Result<Vec<f32>, String> + Send,
    C: FnOnce(&[usize]) -> Vec<(usize, ExpertProfileAcc)> + Send,
{
    if mask.len() != route_weights.len() {
        return None;
    }
    let (hit_slots, miss_slots) = partial_resident_slots(mask)?;
    let gpu_slots = hit_slots.len();
    let cpu_slots = miss_slots.len();
    let wall_start = Instant::now();
    let ((gpu_result, gpu_us), (cpu_result, cpu_us)) = rayon::join(
        || {
            let start = Instant::now();
            let result = gpu(&hit_slots);
            (result, start.elapsed().as_micros())
        },
        || {
            let start = Instant::now();
            let result = cpu(&miss_slots);
            (result, start.elapsed().as_micros())
        },
    );
    let wall_us = wall_start.elapsed().as_micros();
    let per_expert = reassemble_mixed_slots(
        mask.len(),
        n_embd,
        route_weights,
        &hit_slots,
        gpu_result.ok()?,
        gpu_us,
        cpu_result,
    )?;
    Some(PartialMixedFanoutResult {
        per_expert,
        gpu_slots,
        cpu_slots,
        gpu_us,
        cpu_us,
        wall_us,
    })
}

pub(super) fn glm_iq_metal_batch_eligible(
    view: &SharedExpertMoEView<'_>,
    selected_experts: usize,
    precisions: &[MoePrecision],
) -> bool {
    let gate_up_ok = (view.gate_quant == GGMLType::IQ2_XXS && view.up_quant == GGMLType::IQ2_XXS)
        || (view.gate_quant == GGMLType::IQ2_S && view.up_quant == GGMLType::IQ2_S);
    let down_ok = matches!(view.down_quant, GGMLType::IQ3_XXS | GGMLType::IQ4_XS);
    let shared_ok = (view.shared_gate_quant == GGMLType::Q5_K
        && view.shared_up_quant == GGMLType::Q5_K
        && view.shared_down_quant == GGMLType::Q6_K)
        || (view.shared_gate_quant == GGMLType::Q6_K
            && view.shared_up_quant == GGMLType::Q6_K
            && view.shared_down_quant == GGMLType::Q8_0);
    gate_up_ok
        && down_ok
        && shared_ok
        && selected_experts <= 8
        && precisions
            .iter()
            .all(|&precision| precision == MoePrecision::High)
}

pub(super) fn compute_sparse_fanout(
    view: &SharedExpertMoEView<'_>,
    h: &[f32],
    out: &mut [f32],
    residual: Option<&mut [f32]>,
    idx: &[usize],
    exps: &[f32],
    gate_scalar: f32,
    precisions: &[MoePrecision],
    low_gate_up_path: LowGateUpPath,
    profile_enabled: bool,
    prefer_sparse_moe_cuda: bool,
) -> SparseFanoutResult {
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
    let q4k_sparse_cuda_supported = view.gate_quant == GGMLType::Q4_K
        && view.up_quant == GGMLType::Q4_K
        && matches!(down_quant, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K);
    let iq4xs_sparse_cuda_supported = view.gate_quant == GGMLType::IQ4_XS
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
    let glm_iq_metal_supported = glm_iq_metal_batch_eligible(view, idx.len(), precisions);

    // Q2_K shadow geometry (only used when mixed_precision_on).
    let q2k_gu_bpr = q2k_bytes_per_row(view.n_embd);
    let per_q2k_gu = view.n_ff * q2k_gu_bpr;
    let q2k_dn_bpr = q2k_bytes_per_row(view.n_ff);
    let per_q2k_dn = view.n_embd * q2k_dn_bpr;
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
        q2k_gu_bpr,
        per_q2k_gu,
        q2k_dn_bpr,
        per_q2k_dn,
    };

    #[cfg(target_arch = "aarch64")]
    let gate_up_pair_h_q8k = if gate_up_pair_gemv {
        let mut h_q8k = vec![crate::engine::gemm_runtime::Q8KBlock::default(); view.n_embd / 256];
        crate::engine::gemm_runtime::quantize_input_q8k_into(h, &mut h_q8k);
        Some(h_q8k)
    } else {
        None
    };

    let fanout_start = Instant::now();
    let mut batched_sparse_out = None;

    let mut shared_in_sparse_gpu = false;
    let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
    if glm_iq_metal_supported {
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
    if batched_sparse_cuda_supported
        && qwen_moe_backend::qwen_moe_decode_sparse_batch_enabled(
            idx.len(),
            precisions.iter().all(|&p| p == MoePrecision::High),
        )
    {
        let empty: &[u8] = &[];
        let mut gate_slices_stack = [empty; 32];
        let mut up_slices_stack = [empty; 32];
        let mut down_slices_stack = [empty; 32];
        let mut expert_ids_stack = [0u32; 32];
        let mut route_weights_stack = [0.0f32; 32];
        let mut slot_count = 0usize;
        for (&e, &w) in idx.iter().zip(exps.iter()) {
            // Batched GPU path keeps the legacy flat-slice indexing because
            // the stack array stores `&[u8]` references; v3 residency Cow
            // (which may yield owned `Vec<u8>` for disk pread) cannot be
            // stashed there safely. v3 residency wiring still applies to the
            // single-expert CPU path below.
            gate_slices_stack[slot_count] = &view.gate_exps_bytes[e * per_gate..(e + 1) * per_gate];
            up_slices_stack[slot_count] = &view.up_exps_bytes[e * per_up..(e + 1) * per_up];
            down_slices_stack[slot_count] = &view.down_exps_bytes[e * per_down..(e + 1) * per_down];
            expert_ids_stack[slot_count] = e as u32;
            route_weights_stack[slot_count] = w;
            slot_count += 1;
        }
        let mixed_enabled = cuda_q2k_q3k_mixed_resident_cpu_enabled(q2q3_sparse_cuda_supported);
        if should_try_q2q3_mixed_resident_cpu(
            mixed_enabled,
            q2q3_sparse_cuda_supported,
            view.layer_idx,
            precisions,
        ) {
            let selected_count = slot_count;
            let residency = qwen_moe_backend::qwen_moe_prepare_selected_bundle_residency(
                &gate_slices_stack[..selected_count],
                &up_slices_stack[..selected_count],
                &down_slices_stack[..selected_count],
                exps,
                view.layer_idx,
                idx,
                &mut bundle_observation_receipt,
                n_ff,
                n_embd,
            );
            if let Ok(residency_mask) = residency {
                let mixed = run_partial_mixed_fanout(
                    &residency_mask,
                    n_embd,
                    exps,
                    |hit_slots| {
                        let mut hit_gate = [empty; 32];
                        let mut hit_up = [empty; 32];
                        let mut hit_down = [empty; 32];
                        for (hit_index, &slot) in hit_slots.iter().enumerate() {
                            hit_gate[hit_index] = gate_slices_stack[slot];
                            hit_up[hit_index] = up_slices_stack[slot];
                            hit_down[hit_index] = down_slices_stack[slot];
                        }
                        qwen_moe_backend::qwen_moe_decode_sparse_experts_per_slot_resident(
                            &hit_gate[..hit_slots.len()],
                            &hit_up[..hit_slots.len()],
                            &hit_down[..hit_slots.len()],
                            down_quant,
                            n_ff,
                            n_embd,
                            h,
                        )
                    },
                    |miss_slots| {
                        miss_slots
                            .par_iter()
                            .map(|&slot| {
                                (
                                    slot,
                                    compute_sparse_expert_cpu(
                                        view,
                                        h,
                                        idx[slot],
                                        exps[slot],
                                        precisions[slot],
                                        low_gate_up_path,
                                        cpu_geometry,
                                        profile_enabled,
                                        inner_gemv,
                                        #[cfg(target_arch = "aarch64")]
                                        sparse_q6_down_q8k,
                                        #[cfg(target_arch = "aarch64")]
                                        expert_local_rows,
                                        #[cfg(target_arch = "aarch64")]
                                        gate_up_pair_h_q8k.as_deref(),
                                    ),
                                )
                            })
                            .collect()
                    },
                );
                if let Some(mixed) = mixed {
                    if should_emit_partial_mixed_trace(
                        cuda_cache_trace_enabled(),
                        mixed_enabled,
                        true,
                        mixed.gpu_slots,
                        mixed.cpu_slots,
                    ) {
                        let bundle_stats = bundle_observation_receipt.pending_stats();
                        eprintln!(
                            "{}",
                            format_partial_mixed_trace(
                                view.layer_idx.expect("mixed path requires layer index"),
                                &mixed,
                                bundle_stats,
                            )
                        );
                        bundle_observation_receipt.clear_stats();
                    }
                    return SparseFanoutResult::Computed {
                        per_expert: mixed.per_expert,
                        fanout_us: fanout_start.elapsed().as_micros(),
                        shared_in_sparse_gpu: false,
                    };
                }
            }
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
        if let Ok(gpu_sparse_out) = sparse_result {
            batched_sparse_out = Some(gpu_sparse_out);
        }
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
                        precisions,
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
                    precisions[slot],
                    low_gate_up_path,
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
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn fake_cpu_profile(out: Vec<f32>) -> ExpertProfileAcc {
        ExpertProfileAcc {
            out,
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
    }

    #[test]
    fn q2q3_mixed_policy_respects_auto_gate() {
        assert!(!should_try_q2q3_mixed_resident_cpu(
            false,
            true,
            Some(3),
            &[MoePrecision::High],
        ));
        assert!(!should_try_q2q3_mixed_resident_cpu(
            true,
            false,
            Some(3),
            &[MoePrecision::High],
        ));
        assert!(should_try_q2q3_mixed_resident_cpu(
            true,
            true,
            Some(3),
            &[MoePrecision::High],
        ));
    }

    #[test]
    fn mixed_trace_requires_partial_success_and_formats_contract() {
        assert!(should_emit_partial_mixed_trace(true, true, true, 2, 3));
        assert!(!should_emit_partial_mixed_trace(false, true, true, 2, 3));
        assert!(!should_emit_partial_mixed_trace(true, false, true, 2, 3));
        assert!(!should_emit_partial_mixed_trace(true, true, false, 2, 3));
        assert!(!should_emit_partial_mixed_trace(true, true, true, 0, 3));
        assert!(!should_emit_partial_mixed_trace(true, true, true, 2, 0));

        let mixed = PartialMixedFanoutResult {
            per_expert: Vec::new(),
            gpu_slots: 2,
            cpu_slots: 3,
            gpu_us: 120,
            cpu_us: 80,
            wall_us: 150,
        };
        let bundle_stats = ExpertBundleCacheStats {
            bundle_lookups: 5,
            bundle_hits: 2,
            bundle_partial_hits: 1,
            bundle_misses: 2,
            bundle_admissions: 1,
            bundle_evictions: 1,
            admitted_bytes: 4096,
            evicted_bytes: 2048,
            h2d_bytes: 4096,
            temp_h2d_bytes: 0,
        };
        assert_eq!(
            format_partial_mixed_trace(7, &mixed, bundle_stats),
            "[cuda-cache] qwen35_mixed_fanout layer=7 gpu_slots=2 cpu_slots=3 gpu_us=120 cpu_us=80 wall_us=150 overlap_hint=true q2q3_bundle_lookups=5 q2q3_bundle_hits=2 q2q3_bundle_partial_hits=1 q2q3_bundle_misses=2 q2q3_bundle_admissions=1 q2q3_bundle_admitted_bytes=4096 q2q3_bundle_evictions=1 q2q3_bundle_evicted_bytes=2048 q2q3_bundle_h2d_mb=0.00 q2q3_bundle_temp_h2d_mb=0.00 q2q3_bundle_h2d_bytes_per_token=4096.0"
        );
    }

    #[test]
    fn q2q3_mixed_cannot_bypass_sparse_layer_or_precision_gates() {
        assert!(!should_try_q2q3_mixed_resident_cpu(
            true,
            false,
            Some(3),
            &[MoePrecision::High],
        ));
        assert!(!should_try_q2q3_mixed_resident_cpu(
            true,
            true,
            None,
            &[MoePrecision::High],
        ));
        assert!(!should_try_q2q3_mixed_resident_cpu(
            true,
            true,
            Some(3),
            &[MoePrecision::High, MoePrecision::Low],
        ));
    }

    #[test]
    fn partial_partition_keeps_current_slot_order_and_rejects_all_or_none() {
        assert_eq!(
            partial_resident_slots(&[false, true, false, true]),
            Some((vec![1, 3], vec![0, 2]))
        );
        assert_eq!(partial_resident_slots(&[true, true]), None);
        assert_eq!(partial_resident_slots(&[false, false]), None);
        assert_eq!(partial_resident_slots(&[]), None);
    }

    #[test]
    fn mixed_one_hit_many_misses_reassembles_duplicates_and_weights_once() {
        let experts = [7usize, 7, 3, 7];
        let weights = [0.5f32, 2.0, -1.0, 0.25];
        let mask = [false, true, false, false];
        let mixed = run_partial_mixed_fanout(
            &mask,
            2,
            &weights,
            |hit_slots| {
                assert_eq!(hit_slots, &[1]);
                Ok(hit_slots
                    .iter()
                    .flat_map(|&slot| [experts[slot] as f32, slot as f32])
                    .collect())
            },
            |miss_slots| {
                assert_eq!(miss_slots, &[0, 2, 3]);
                miss_slots
                    .iter()
                    .map(|&slot| {
                        (
                            slot,
                            fake_cpu_profile(vec![
                                experts[slot] as f32 * weights[slot],
                                slot as f32 * weights[slot],
                            ]),
                        )
                    })
                    .collect()
            },
        )
        .expect("partial mixed result");

        assert_eq!(mixed.gpu_slots, 1);
        assert_eq!(mixed.cpu_slots, 3);
        let outputs: Vec<Vec<f32>> = mixed
            .per_expert
            .into_iter()
            .map(|expert| expert.out)
            .collect();
        assert_eq!(
            outputs,
            vec![
                vec![3.5, 0.0],
                vec![14.0, 2.0],
                vec![-3.0, -2.0],
                vec![1.75, 0.75],
            ]
        );
    }

    #[test]
    fn mixed_branch_timings_do_not_change_result_semantics() {
        let mixed = run_partial_mixed_fanout(
            &[true, false],
            1,
            &[2.0, 3.0],
            |_| {
                std::thread::sleep(Duration::from_millis(2));
                Ok(vec![4.0])
            },
            |_| {
                std::thread::sleep(Duration::from_millis(2));
                vec![(1, fake_cpu_profile(vec![15.0]))]
            },
        )
        .expect("partial mixed result");

        assert_eq!(
            mixed
                .per_expert
                .iter()
                .map(|expert| expert.out[0])
                .collect::<Vec<_>>(),
            vec![8.0, 15.0]
        );
        assert!(mixed.gpu_us >= 1_000);
        assert!(mixed.cpu_us >= 1_000);
        assert!(mixed.wall_us >= mixed.gpu_us);
        assert!(mixed.wall_us >= mixed.cpu_us);
    }

    #[test]
    fn mixed_reduction_preserves_current_idx_order() {
        let mixed = run_partial_mixed_fanout(
            &[true, false, true],
            1,
            &[1.0, 1.0, 1.0],
            |hit_slots| {
                assert_eq!(hit_slots, &[0, 2]);
                Ok(vec![1.0e20, -1.0e20])
            },
            |miss_slots| {
                assert_eq!(miss_slots, &[1]);
                vec![(1, fake_cpu_profile(vec![1.0]))]
            },
        )
        .expect("partial mixed result");
        assert_eq!(
            mixed
                .per_expert
                .iter()
                .map(|expert| expert.out[0])
                .collect::<Vec<_>>(),
            vec![1.0e20, 1.0, -1.0e20]
        );
        assert_eq!(
            mixed
                .per_expert
                .iter()
                .fold(0.0f32, |sum, expert| sum + expert.out[0]),
            0.0
        );
    }

    #[test]
    fn gpu_subset_failure_returns_fallback_signal_after_cpu_join() {
        let cpu_ran = AtomicBool::new(false);
        let mixed = run_partial_mixed_fanout(
            &[true, false],
            1,
            &[1.0, 1.0],
            |_| Err("resident launch failed".to_owned()),
            |miss_slots| {
                cpu_ran.store(true, Ordering::SeqCst);
                vec![(miss_slots[0], fake_cpu_profile(vec![2.0]))]
            },
        );
        assert!(mixed.is_none());
        assert!(cpu_ran.load(Ordering::SeqCst));
    }

    #[test]
    fn all_hit_and_all_miss_skip_mixed_subset_execution() {
        for mask in [&[true, true][..], &[false, false][..]] {
            let mixed = run_partial_mixed_fanout(
                mask,
                1,
                &[1.0, 1.0],
                |_| panic!("GPU subset must not run"),
                |_| panic!("CPU subset must not run"),
            );
            assert!(mixed.is_none());
        }
    }
}

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
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
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

        let mut input_q8k = vec![crate::engine::gemm_runtime::Q8KBlock::default(); blocks];
        crate::engine::gemm_runtime::quantize_input_q8k_into(&input, &mut input_q8k);
        for (row, &got) in actual.iter().enumerate() {
            let row_bytes = &down_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let expected = unsafe {
                crate::engine::gemm_runtime::neon_dot::dot_q6_k_q8k_neon(
                    row_bytes, &input_q8k, blocks,
                )
            };
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
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
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
            shadow_gate_bytes: None,
            shadow_up_bytes: None,
            shadow_gate_up_tile_bytes: None,
            shadow_down_bytes: None,
            moe_section_decode: None,
            gate_residency: None,
            up_residency: None,
            down_residency: None,
        };
        let h: Vec<f32> = (0..n_embd)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1)
            .collect();
        let idx = [7, 0, 5, 2, 6, 1, 4, 3];
        let exps = [0.31, 0.27, 0.19, 0.11, 0.07, 0.03, 0.015, 0.005];
        let precisions = [MoePrecision::High; 8];
        let mut scratch = vec![0.0f32; n_embd];
        let previous = std::env::var_os("RNB_QWEN35_MOE_DECODE_SPARSE_BATCH_DIRECT");

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
            &precisions,
            LowGateUpPath::RowMajor,
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
            &precisions,
            LowGateUpPath::RowMajor,
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
