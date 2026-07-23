use rayon::prelude::*;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};

use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::memory_runtime::memtrace;
use crate::engine::moe_types::q4k_bytes_per_row;
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

#[cfg(target_arch = "aarch64")]
fn gemma4_moe_decode_q8k_enabled() -> bool {
    crate::engine::policy::env_string("RNB_GEMMA4_MOE_DECODE_Q8K")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(cfg!(target_os = "android"))
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

        let (idx, exps) = select_experts_from_logits(logits, self.n_expert_used);
        if let Some(layer) = self.layer_idx {
            crate::engine::moe_trace::record_selection(layer, &idx);
        }

        #[cfg(target_arch = "aarch64")]
        let h_q8k = gemma4_moe_decode_q8k_enabled()
            .then(|| crate::engine::quantized_dispatch::quantize_raw_q8k(h));

        let gate_up_rows = self.n_ff * 2;
        let gate_up_bpr = q4k_bytes_per_row(self.n_embd);
        let down_bpr = down_bytes_per_row(self.n_ff, self.down_quant);
        let per_gu = self.per_expert_gate_up_bytes();
        let per_dn = self.per_expert_down_bytes();
        let n_ff = self.n_ff;
        let n_embd = self.n_embd;

        let per_expert: Vec<Vec<f32>> = idx
            .par_iter()
            .enumerate()
            .map(|(k, &e)| {
                let gu_slice = &self.gate_up_bytes[e * per_gu..(e + 1) * per_gu];
                let dn_slice = &self.down_bytes[e * per_dn..(e + 1) * per_dn];

                if memtrace::is_active() {
                    if let Some(layer) = self.layer_idx {
                        memtrace::record_expert_call(
                            0,
                            layer,
                            e,
                            e as u32,
                            k,
                            gu_slice.as_ptr(),
                            gu_slice.len(),
                            "gu",
                            "hot",
                        );
                        memtrace::record_expert_call(
                            0,
                            layer,
                            e,
                            e as u32,
                            k,
                            dn_slice.as_ptr(),
                            dn_slice.len(),
                            "dn",
                            "hot",
                        );
                    }
                }

                let mut gate_up_out = vec![0f32; gate_up_rows];
                #[cfg(target_arch = "aarch64")]
                let pair_done = h_q8k.as_deref().is_some_and(|q8k| {
                    let projection_bytes = n_ff * gate_up_bpr;
                    let (gate_out, up_out) = gate_up_out.split_at_mut(n_ff);
                    crate::engine::quantized_dispatch::dispatch_q4k_pair_q8k_prequantized(
                        &gu_slice[..projection_bytes],
                        &gu_slice[projection_bytes..],
                        q8k,
                        gate_out,
                        up_out,
                        n_ff,
                        n_embd,
                        gate_up_bpr,
                        gate_up_bpr,
                        true,
                    )
                });
                #[cfg(not(target_arch = "aarch64"))]
                let pair_done = false;
                if !pair_done {
                    for r in 0..gate_up_rows {
                        let rb = &gu_slice[r * gate_up_bpr..(r + 1) * gate_up_bpr];
                        gate_up_out[r] =
                            dot_k_block_row(rb, h, n_embd, gate_up_bpr, GGMLType::Q4_K);
                    }
                }

                let (gate_slice, up_slice) = gate_up_out.split_at_mut(n_ff);
                apply_model_gate_mul_inplace(gate_slice, up_slice, ModelArchitecture::Gemma4);

                let mut expert_out = vec![0f32; n_embd];
                for r in 0..n_embd {
                    let rb = &dn_slice[r * down_bpr..(r + 1) * down_bpr];
                    expert_out[r] =
                        dot_k_block_row(rb, gate_slice, n_ff, down_bpr, self.down_quant);
                }

                let weight = exps[k] * self.down_scale[e];
                for value in &mut expert_out {
                    *value *= weight;
                }
                expert_out
            })
            .collect();

        out.fill(0.0);
        for expert_out in &per_expert {
            for i in 0..n_embd {
                out[i] += expert_out[i];
            }
        }
    }
}
