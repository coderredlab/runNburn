use rayon::prelude::*;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};

use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::memory_runtime::memtrace;
use crate::engine::moe_routing::{
    assign_moe_precisions, moe_mixed_precision_enabled, MoePrecision,
};
use crate::engine::moe_types::{q2k_bytes_per_row, q4k_bytes_per_row};
use crate::engine::norm::apply_model_gate_mul_inplace;
use crate::engine::scalar_gemv::dot_k_block_row;

use super::moe_types::{down_bytes_per_row, MoeLayerView};

pub(super) fn select_experts_from_logits(
    logits: &[f32],
    n_expert_used: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    if n_expert_used < logits.len() {
        idx.select_nth_unstable_by(n_expert_used, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(n_expert_used);
    }

    let max_l = idx
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = idx.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    (idx, weights)
}

impl<'a> MoeLayerView<'a> {
    #[inline]
    pub fn per_expert_gate_up_bytes(&self) -> usize {
        // rows = n_ff*2, cols = n_embd
        (self.n_ff * 2) * q4k_bytes_per_row(self.n_embd)
    }

    #[inline]
    pub fn per_expert_down_bytes(&self) -> usize {
        // rows = n_embd, cols = n_ff
        self.n_embd * down_bytes_per_row(self.n_ff, self.down_quant)
    }

    /// One MoE FFN forward: `out[n_embd] = Σ_k softmax(router @ h)[e_k] · expert_k(h)`.
    ///
    /// `out` is written (not accumulated). Callers who need to merge with a dense path
    /// should add `out` to their dense FFN result externally.
    pub fn forward(&self, h: &[f32], out: &mut [f32]) {
        // 1. Router logits: [n_expert] = router_w @ h
        let mut logits = vec![0f32; self.n_expert];
        gemv_f32(self.router_w, h, &mut logits, self.n_expert, self.n_embd, 1);
        self.forward_with_logits(h, &logits, out);
    }

    /// Same as `forward` but takes externally computed `logits[n_expert]`. Gemma4 26B-A4B
    /// calls this because its router input is not the MoE FFN input `h` — it's
    /// `rms_norm(attn_out) / sqrt(n_embd) * ffn_gate_inp_s` (pre-scale + learned scale).
    pub fn forward_with_logits(&self, h: &[f32], logits: &[f32], out: &mut [f32]) {
        assert_eq!(h.len(), self.n_embd);
        assert_eq!(out.len(), self.n_embd);
        assert_eq!(logits.len(), self.n_expert);

        // 2. top-k (k = n_expert_used) + softmax over picked
        let (mut idx, mut exps) = select_experts_from_logits(logits, self.n_expert_used);
        // Optional MoE expert selection trace (rnb-moe-profile).
        if let Some(layer) = self.layer_idx {
            crate::engine::moe_trace::record_selection(layer, &idx);
        }

        // Session 71 MoE mixed precision: when `RNB_HOBBIT=1` and a shadow `.rnb` is wired,
        // sort the picked experts by softmax weight descending and assign a
        // per-position precision tag using cumulative unimportance `s_{e_i}`.
        //   s_{e_0}     = 0
        //   s_{e_i}     = Σ_{j<i} exps[j]     (i > 0)
        //   s ≤ T1      → High (base Q4_K / Q5_1)
        //   T1 < s ≤ T2 → Low  (shadow Q2_K gate_up, base Q5_1 down)
        //   s > T2      → Skip (zero contribution, weight folded into renorm)
        // Default T1=0.6, T2=0.9 (from paper). Overrides via RNB_HOBBIT_T1 /
        // RNB_HOBBIT_T2. `e_0` is the highest-weight expert and always High.
        let mixed_precision_on = moe_mixed_precision_enabled(self.shadow_gate_up_bytes.is_some());
        let precisions = assign_moe_precisions(&mut idx, &mut exps, mixed_precision_on);

        // 3. per-expert GEMV and weighted accumulate.
        //
        // Parallelism moved to **expert level**: 8 picked experts fan out across the
        // thread pool via `par_iter`, each expert runs fully sequential inner GEMV
        // (direct scalar `dot_k_block_row`). Previous nested rayon (inner par_chunks
        // per gemv × 8 experts × 30 layers = 240 dispatches/decode step) was the
        // dominant overhead. This is bit-exact scalar — correctness matches the
        // original sequential-over-experts path.
        let gate_up_rows = self.n_ff * 2;
        let gate_up_bpr = q4k_bytes_per_row(self.n_embd);
        let down_bpr = down_bytes_per_row(self.n_ff, self.down_quant);
        let per_gu = self.per_expert_gate_up_bytes();
        let per_dn = self.per_expert_down_bytes();
        let n_ff = self.n_ff;
        let n_embd = self.n_embd;

        // Hot/cold split is owned by the residency view (`gate_up_residency` /
        // `down_residency`) when wired; otherwise the GGUF flat fallback is
        // used. The legacy `hot_count` / `runtime_hot_count` fields no longer
        // gate dispatch.

        let per_expert: Vec<Vec<f32>> = idx
            .par_iter()
            .enumerate()
            .map(|(k, &e)| {
                // Session 71 MoE mixed precision early branches.
                let prec = precisions[k];
                if prec == MoePrecision::Skip {
                    return vec![0.0f32; n_embd];
                }
                if prec == MoePrecision::Low {
                    // Mixed-precision Low path: assumes the GGUF-flat layout
                    // (`gate_up_residency` is `None`, no hot/cold split) so
                    // `self.down_bytes` spans every expert and
                    // `self.shadow_gate_up_bytes` is `[n_expert × per_gu_q2k]`.
                    // When residency is wired (split mode), shadow is disabled
                    // and we fall through to the High path.
                    if self.gate_up_residency.is_none() {
                        let gu_shadow = self
                            .shadow_gate_up_bytes
                            .expect("MoE low-precision requires shadow_gate_up_bytes");
                        let gu_bpr_q2k = q2k_bytes_per_row(n_embd);
                        let per_gu_q2k = gate_up_rows * gu_bpr_q2k;
                        let gu_slice = &gu_shadow[e * per_gu_q2k..(e + 1) * per_gu_q2k];
                        let dn_slice = &self.down_bytes[e * per_dn..(e + 1) * per_dn];

                        let mut gu_out = vec![0f32; gate_up_rows];
                        for r in 0..gate_up_rows {
                            let rb = &gu_slice[r * gu_bpr_q2k..(r + 1) * gu_bpr_q2k];
                            gu_out[r] = dot_k_block_row(rb, h, n_embd, gu_bpr_q2k, GGMLType::Q2_K);
                        }
                        let (gate, up) = gu_out.split_at_mut(n_ff);
                        apply_model_gate_mul_inplace(gate, up, ModelArchitecture::Gemma);
                        let mid = &gu_out[..n_ff];

                        let mut expert_out = vec![0f32; n_embd];
                        for r in 0..n_embd {
                            let rb = &dn_slice[r * down_bpr..(r + 1) * down_bpr];
                            expert_out[r] =
                                dot_k_block_row(rb, mid, n_ff, down_bpr, self.down_quant);
                        }

                        let weight = exps[k] * self.down_scale[e];
                        for v in expert_out.iter_mut() {
                            *v *= weight;
                        }
                        return expert_out;
                    }
                    // Fall through to High path when residency is wired
                    // (shadow disabled in split mode).
                }

                // Resolve `gu_slice` / `dn_slice` for expert rank `e` through
                // a single residency view dispatch. The view (composed in
                // `packed_wiring`) folds hot mmap, runtime hot pool, cold
                // mmap, cold pread, and unified-cold-pread into one
                // `expert_bytes(rank)` call. `Cow::Borrowed` for mmap/RAM,
                // `Cow::Owned(Vec<u8>)` for disk pread; both deref to `&[u8]`.
                //
                // GGUF-only path (no `.rnb` sidecar wired): residency is
                // `None`; the flat `gate_up_bytes` / `down_bytes` slices hold
                // every expert in a contiguous tensor, so we index them
                // directly. This path matches the unsplit (`hot_count == 0`)
                // case from the legacy 5-way matrix.
                // Resolve `gate_up` and `down` independently — Gemma4
                // sidecars commonly carry only `gate_up_exps` (Q4_K) while
                // `down_exps` (Q5_1) stays on the GGUF mmap. Each tensor
                // routes through its residency view when set, otherwise it
                // indexes the flat `gate_up_bytes` / `down_bytes` slice.
                let _gu_cow_holder;
                let gu_slice: &[u8] = if let Some(gur) = self.gate_up_residency {
                    _gu_cow_holder = gur.expert_bytes(e);
                    &*_gu_cow_holder
                } else {
                    &self.gate_up_bytes[e * per_gu..(e + 1) * per_gu]
                };
                let _dn_cow_holder;
                let dn_slice: &[u8] = if let Some(dnr) = self.down_residency {
                    _dn_cow_holder = dnr.expert_bytes(e);
                    &*_dn_cow_holder
                } else {
                    &self.down_bytes[e * per_dn..(e + 1) * per_dn]
                };
                let tier: &'static str = match self.gate_up_residency {
                    Some(gur) if e < gur.hot_count() => "hot",
                    Some(_) => "cold",
                    None => "hot",
                };

                // memtrace expert hook: capture per-slice mincore residency
                // BEFORE the GEMV reads the bytes, so the probe reflects the
                // pre-access state (what H3 wants to measure). `step_idx = 0`
                // is a placeholder — the offline analyzer attributes expert
                // rows to the nearest prior step_start row by timestamp.
                //
                // `original_id` inverts the `.rnb` hot-sort permutation so
                // analyzers can filter by offline popularity-space id. When
                // no permutation metadata is loaded, this is identity.
                if memtrace::is_active() {
                    if let Some(layer) = self.layer_idx {
                        let original_id = self
                            .rank_to_original
                            .and_then(|m| m.get(e).copied())
                            .unwrap_or(e as u32);
                        memtrace::record_expert_call(
                            0,
                            layer,
                            e,
                            original_id,
                            k,
                            gu_slice.as_ptr(),
                            gu_slice.len(),
                            "gu",
                            tier,
                        );
                        memtrace::record_expert_call(
                            0,
                            layer,
                            e,
                            original_id,
                            k,
                            dn_slice.as_ptr(),
                            dn_slice.len(),
                            "dn",
                            tier,
                        );
                    }
                }

                // Q4_K × f32 dot, sequential rows (no inner rayon).
                let mut gate_up_out = vec![0f32; gate_up_rows];
                for r in 0..gate_up_rows {
                    let rb = &gu_slice[r * gate_up_bpr..(r + 1) * gate_up_bpr];
                    gate_up_out[r] = dot_k_block_row(rb, h, n_embd, gate_up_bpr, GGMLType::Q4_K);
                }

                // split + GELU-tanh gate * up (Gemma 4)
                let (gate_slice, up_slice) = gate_up_out.split_at_mut(n_ff);
                apply_model_gate_mul_inplace(gate_slice, up_slice, ModelArchitecture::Gemma4);

                // Q5_1 × f32 dot, sequential rows.
                let mut expert_out = vec![0f32; n_embd];
                for r in 0..n_embd {
                    let rb = &dn_slice[r * down_bpr..(r + 1) * down_bpr];
                    expert_out[r] =
                        dot_k_block_row(rb, gate_slice, n_ff, down_bpr, self.down_quant);
                }

                // pre-multiply by router softmax weight so the final reduction is a plain add.
                let w = exps[k] * self.down_scale[e];
                for v in expert_out.iter_mut() {
                    *v *= w;
                }
                expert_out
            })
            .collect();

        out.fill(0.0);
        for eo in &per_expert {
            for i in 0..n_embd {
                out[i] += eo[i];
            }
        }
    }
}
