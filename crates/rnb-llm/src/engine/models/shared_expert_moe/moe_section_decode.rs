use rayon::prelude::*;
use std::time::Instant;

use super::routing::qwen35_softmax_topk_route;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::moe_profile::{
    is_enabled, record_moe_counts, record_moe_profile, record_moe_profile_by_layer,
};
use crate::engine::moe_section_dispatch::{
    dot_down_row_q5k, dot_gate_up_row_q4k, dot_shared_down_row_q80, dot_shared_gate_up_row_q80,
    down_q5k_unit_size, quantize_q8k, quantize_q8k_into, shared_down_q80_unit_size,
    shared_gate_up_q80_unit_size, MoeSectionQ8KBlock,
};
use crate::engine::moe_section_layout::moe_section_gate_up_layout;

use super::moe_types::SharedExpertMoEView;

impl<'a> SharedExpertMoEView<'a> {
    /// Session 79 Phase 1 Task 13: moe_section integer-scale sdot decode path.
    ///
    /// Consumes the `MOE_DECODE_SECTION` view attached at load time. The hot
    /// loop runs in i32/i64 sdot with a single f32 fold per (row, block):
    ///
    /// ```text
    ///   gate_f += (gate_int_block as f64) * (row.gate_mul as f64) * (h_q8k[b].d as f64)
    /// ```
    ///
    /// f64 accumulation across blocks is intentional: the per-block d varies,
    /// so we cannot defer the fold to a single multiply at row boundary like
    /// the legacy NEON gemv does. f64 keeps catastrophic cancellation off the
    /// table for adversarial weight/activation combinations (Q4_K block dot
    /// can reach ±3.9e9, multiplied by row_mul × d we land in mantissa-bleed
    /// territory for f32).
    ///
    /// Router and routing logic mirrors the legacy path verbatim — no
    /// re-derivation of softmax / norm_topk_prob.
    #[cfg(target_arch = "aarch64")]
    pub(super) fn forward_moe_section_sdot(
        &self,
        moe_section: &crate::engine::MoeSectionDecodeLayer,
        h: &[f32],
        out: &mut [f32],
    ) {
        // The MoE section path is attached only for ARM dotprod-capable devices in this
        // experimental branch; keep the large body in a target-feature function
        // so the tiny sdot helpers can inline.
        unsafe { self.forward_moe_section_sdot_dotprod(moe_section, h, out) }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn forward_moe_section_sdot_dotprod(
        &self,
        moe_section: &crate::engine::MoeSectionDecodeLayer,
        h: &[f32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(h.len(), self.n_embd);
        debug_assert_eq!(out.len(), self.n_embd);
        debug_assert_eq!(moe_section.n_embd as usize, self.n_embd);
        debug_assert_eq!(moe_section.n_experts as usize, self.n_expert);
        let profile_enabled = is_enabled();

        let n_embd = self.n_embd;
        let d_ff = moe_section.d_ff as usize;
        let n_expert = self.n_expert;
        let n_expert_used = self.n_expert_used;
        let gate_up_layout = moe_section_gate_up_layout(moe_section.gate_up_quant)
            .expect("unsupported moe_section gate_up_quant passed parser");
        let gate_up_unit_size = gate_up_layout.unit_size();

        // Router (legacy f32) — keep using the existing path. The MoE section layout
        // does not yet pack the router, so we route via the legacy slice.
        let router_start = profile_enabled.then(Instant::now);
        let mut logits_stack = [0.0f32; 256];
        let mut logits_vec;
        let logits: &mut [f32] = if n_expert <= logits_stack.len() {
            &mut logits_stack[..n_expert]
        } else {
            logits_vec = vec![0.0f32; n_expert];
            &mut logits_vec
        };
        gemv_f32(self.router_w, h, logits, n_expert, n_embd, 1);
        if let Some(start) = router_start {
            record_moe_profile("qwen35moe:moe_section:router", start.elapsed());
        }

        // Qwen3-Next routing: softmax(all) → top-k by prob → renormalize.
        let routing_start = profile_enabled.then(Instant::now);
        let mut probs_stack = [0.0f32; 256];
        let mut probs_vec;
        let probs: &mut [f32] = if n_expert <= probs_stack.len() {
            &mut probs_stack[..n_expert]
        } else {
            probs_vec = vec![0.0f32; n_expert];
            &mut probs_vec
        };
        let mut idx_stack = [0usize; 256];
        let mut idx_vec;
        let idx_all: &mut [usize] = if n_expert <= idx_stack.len() {
            &mut idx_stack[..n_expert]
        } else {
            idx_vec = vec![0usize; n_expert];
            &mut idx_vec
        };
        let selected_len = n_expert_used.min(n_expert);
        let mut weights_stack = [0.0f32; 32];
        let mut weights_vec;
        let weights: &mut [f32] = if selected_len <= weights_stack.len() {
            &mut weights_stack[..selected_len]
        } else {
            weights_vec = vec![0.0f32; selected_len];
            &mut weights_vec
        };
        let selected_len =
            qwen35_softmax_topk_route(logits, n_expert_used, idx_all, probs, weights, true);
        let idx = &idx_all[..selected_len];
        let weights = &weights[..selected_len];
        if let Some(layer) = self.layer_idx {
            crate::engine::moe_trace::record_selection(layer, idx);
            crate::engine::moe_trace::record_predictor(layer, idx, probs);
        }
        if let Some(start) = routing_start {
            record_moe_profile("qwen35moe:moe_section:routing", start.elapsed());
            record_moe_counts("qwen35moe:moe_section:decode", idx.len() as u64, 0, 0);
        }

        // Activation prequant (n_embd / 256 Q8K blocks).
        let h_prequant_start = profile_enabled.then(Instant::now);
        let n_embd_blocks_expected = n_embd / 256;
        let mut h_q8k_stack = [MoeSectionQ8KBlock::default(); 8];
        let h_q8k_vec;
        let h_q8k: &[MoeSectionQ8KBlock] = if n_embd_blocks_expected == h_q8k_stack.len() {
            quantize_q8k_into(h, &mut h_q8k_stack);
            &h_q8k_stack
        } else {
            h_q8k_vec = quantize_q8k(h);
            &h_q8k_vec
        };
        let n_embd_blocks = h_q8k.len();
        debug_assert_eq!(n_embd_blocks * 256, n_embd);
        if let Some(start) = h_prequant_start {
            record_moe_profile("qwen35moe:moe_section:h_prequant", start.elapsed());
        }

        let file_bytes: &[u8] = &moe_section.file_bytes;
        struct MoeSectionExpertAcc {
            sum_out: [f32; 2048],
            inter: [f32; 512],
            inter_q8k_scratch: [MoeSectionQ8KBlock; 2],
            wall_us: u128,
            gate_up_us: u128,
            inter_quant_us: u128,
            down_compute_us: u128,
        }
        let fanout_start = profile_enabled.then(Instant::now);
        let sparse_acc = idx
            .par_iter()
            .enumerate()
            .fold(
                || MoeSectionExpertAcc {
                    sum_out: [0.0f32; 2048],
                    inter: [0.0f32; 512],
                    inter_q8k_scratch: [MoeSectionQ8KBlock::default(); 2],
                    wall_us: 0,
                    gate_up_us: 0,
                    inter_quant_us: 0,
                    down_compute_us: 0,
                },
                |mut acc, (k, &expert_idx)| {
                    let expert_start = profile_enabled.then(Instant::now);
                    let gate_w = weights[k];
                    let expert = &moe_section.experts[expert_idx];
                    debug_assert_eq!(expert.gate_up_rows.len(), d_ff);
                    debug_assert_eq!(expert.down_rows.len(), n_embd);

                    let gate_up_start = profile_enabled.then(Instant::now);
                    for (r, row) in expert.gate_up_rows.iter().enumerate() {
                        debug_assert_eq!(row.blocks_len % gate_up_unit_size, 0);
                        let n_pairs = row.blocks_len / gate_up_unit_size;
                        debug_assert_eq!(n_pairs, n_embd_blocks);
                        debug_assert!(row.blocks_offset + row.blocks_len <= file_bytes.len());

                        let row_bytes =
                            &file_bytes[row.blocks_offset..row.blocks_offset + row.blocks_len];
                        let scale_bytes = if gate_up_layout.uses_scale_plane() {
                            let scale_offset = row
                                .scale_offset
                                .expect("scale-plane row missing scale bytes");
                            debug_assert!(scale_offset + row.scale_len <= file_bytes.len());
                            Some(&file_bytes[scale_offset..scale_offset + row.scale_len])
                        } else {
                            None
                        };
                        let (g_acc, u_acc) = unsafe {
                            dot_gate_up_row_q4k(gate_up_layout, row_bytes, scale_bytes, h_q8k)
                        };
                        acc.inter[r] = silu_f32(g_acc * row.gate_mul) * (u_acc * row.up_mul);
                    }
                    if let Some(start) = gate_up_start {
                        acc.gate_up_us += start.elapsed().as_micros();
                    }

                    let inter_q_start = profile_enabled.then(Instant::now);
                    quantize_q8k_into(&acc.inter, &mut acc.inter_q8k_scratch);
                    if let Some(start) = inter_q_start {
                        acc.inter_quant_us += start.elapsed().as_micros();
                    }
                    let inter_q8k: &[MoeSectionQ8KBlock] = &acc.inter_q8k_scratch;
                    let d_ff_blocks = inter_q8k.len();
                    debug_assert_eq!(d_ff_blocks * 256, d_ff);

                    let down_start = profile_enabled.then(Instant::now);
                    for (r, row) in expert.down_rows.iter().enumerate() {
                        let blk_size = down_q5k_unit_size();
                        debug_assert_eq!(row.blocks_len % blk_size, 0);
                        let n_blocks = row.blocks_len / blk_size;
                        debug_assert_eq!(n_blocks, d_ff_blocks);
                        debug_assert!(row.blocks_offset + row.blocks_len <= file_bytes.len());
                        let row_bytes =
                            &file_bytes[row.blocks_offset..row.blocks_offset + row.blocks_len];
                        let acc_f = unsafe { dot_down_row_q5k(row_bytes, inter_q8k) };
                        acc.sum_out[r] += gate_w * (acc_f * row.down_mul);
                    }
                    if let Some(start) = down_start {
                        acc.down_compute_us += start.elapsed().as_micros();
                    }

                    if let Some(start) = expert_start {
                        acc.wall_us = acc.wall_us.max(start.elapsed().as_micros());
                    }
                    acc
                },
            )
            .reduce(
                || MoeSectionExpertAcc {
                    sum_out: [0.0f32; 2048],
                    inter: [0.0f32; 512],
                    inter_q8k_scratch: [MoeSectionQ8KBlock::default(); 2],
                    wall_us: 0,
                    gate_up_us: 0,
                    inter_quant_us: 0,
                    down_compute_us: 0,
                },
                |mut a, b| {
                    for i in 0..n_embd {
                        a.sum_out[i] += b.sum_out[i];
                    }
                    a.wall_us = a.wall_us.max(b.wall_us);
                    a.gate_up_us += b.gate_up_us;
                    a.inter_quant_us += b.inter_quant_us;
                    a.down_compute_us += b.down_compute_us;
                    a
                },
            );
        if profile_enabled {
            let fanout_us = fanout_start
                .expect("profile fanout timer")
                .elapsed()
                .as_micros();
            let gate_up_us = sparse_acc.gate_up_us;
            let inter_quant_us = sparse_acc.inter_quant_us;
            let down_compute_us = sparse_acc.down_compute_us;
            let max_expert_us = sparse_acc.wall_us;
            let dispatch_us = fanout_us.saturating_sub(max_expert_us);
            if dispatch_us > 0 {
                let elapsed =
                    std::time::Duration::from_micros(dispatch_us.min(u64::MAX as u128) as u64);
                record_moe_profile("qwen35moe:moe_section:dispatch", elapsed);
            }
            if gate_up_us > 0 {
                let elapsed =
                    std::time::Duration::from_micros(gate_up_us.min(u64::MAX as u128) as u64);
                record_moe_profile("qwen35moe:moe_section:gate_up_compute", elapsed);
                record_moe_profile_by_layer(
                    "qwen35moe:moe_section",
                    self.layer_idx,
                    "gate_up_compute",
                    elapsed,
                );
            }
            if inter_quant_us > 0 {
                let elapsed =
                    std::time::Duration::from_micros(inter_quant_us.min(u64::MAX as u128) as u64);
                record_moe_profile("qwen35moe:moe_section:inter_quant", elapsed);
            }
            if down_compute_us > 0 {
                let elapsed =
                    std::time::Duration::from_micros(down_compute_us.min(u64::MAX as u128) as u64);
                record_moe_profile("qwen35moe:moe_section:down_compute", elapsed);
            }
        }

        let reduce_start = profile_enabled.then(Instant::now);
        out.fill(0.0);
        for i in 0..n_embd {
            out[i] += sparse_acc.sum_out[i];
        }
        if let Some(start) = reduce_start {
            record_moe_profile("qwen35moe:moe_section:reduce", start.elapsed());
        }

        // ----- Shared expert (Qwen3.6 only; absent on Gemma4 MoE). -----
        let shared_start = profile_enabled.then(Instant::now);
        if let Some(shared) = &moe_section.shared_expert {
            let gate_dot: f32 = h
                .iter()
                .zip(self.shared_input_scale.iter())
                .map(|(a, b)| a * b)
                .sum();
            let gate_scalar = 1.0 / (1.0 + (-gate_dot).exp());

            let d_ff_s = shared.d_ff_s as usize;
            let mut shared_inter_stack = [0.0f32; 512];
            let mut shared_inter_vec;
            let shared_inter: &mut [f32] = if d_ff_s == shared_inter_stack.len() {
                &mut shared_inter_stack
            } else {
                shared_inter_vec = vec![0.0f32; d_ff_s];
                &mut shared_inter_vec
            };
            let mut shared_inter_q8k_stack = [MoeSectionQ8KBlock::default(); 2];
            let mut shared_inter_q8k_vec;
            let shared_inter_q8k_scratch: &mut [MoeSectionQ8KBlock] =
                if d_ff_s / 256 == shared_inter_q8k_stack.len() {
                    &mut shared_inter_q8k_stack
                } else {
                    shared_inter_q8k_vec = vec![MoeSectionQ8KBlock::default(); d_ff_s / 256];
                    &mut shared_inter_q8k_vec
                };
            shared_inter
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, dst)| {
                    let row = &shared.shared_gate_up_rows[r];
                    let unit_size = shared_gate_up_q80_unit_size();
                    debug_assert_eq!(row.blocks_len % unit_size, 0);
                    let n_units = row.blocks_len / unit_size;
                    debug_assert_eq!(n_units, n_embd_blocks);
                    debug_assert!(row.blocks_offset + row.blocks_len <= file_bytes.len());
                    let row_bytes =
                        &file_bytes[row.blocks_offset..row.blocks_offset + row.blocks_len];
                    let (g_acc, u_acc) = unsafe { dot_shared_gate_up_row_q80(row_bytes, h_q8k) };
                    *dst = silu_f32(g_acc * row.gate_mul) * (u_acc * row.up_mul);
                });

            quantize_q8k_into(&shared_inter, shared_inter_q8k_scratch);
            let shared_inter_q8k: &[MoeSectionQ8KBlock] = &shared_inter_q8k_scratch;
            let shared_inter_blocks = shared_inter_q8k.len();
            debug_assert_eq!(shared_inter_blocks * 256, d_ff_s);

            out.par_iter_mut().enumerate().for_each(|(r, dst)| {
                let row = &shared.shared_down_rows[r];
                let blk_size = shared_down_q80_unit_size();
                debug_assert_eq!(row.blocks_len % blk_size, 0);
                let n_blocks = row.blocks_len / blk_size;
                debug_assert_eq!(n_blocks, shared_inter_blocks * 8);
                debug_assert!(row.blocks_offset + row.blocks_len <= file_bytes.len());
                let row_bytes = &file_bytes[row.blocks_offset..row.blocks_offset + row.blocks_len];
                let acc_f = unsafe { dot_shared_down_row_q80(row_bytes, shared_inter_q8k) };
                *dst += gate_scalar * (acc_f * row.down_mul);
            });
        }
        if let Some(start) = shared_start {
            record_moe_profile("qwen35moe:moe_section:shared_expert", start.elapsed());
        }
    }
}

/// Scalar SiLU for f32 — `x / (1 + exp(-x))`. Used by the MoE section sdot path where
/// each row's gate value is materialized one-at-a-time rather than as a
/// vectorized buffer (the legacy path uses `apply_model_gate_mul_inplace`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
