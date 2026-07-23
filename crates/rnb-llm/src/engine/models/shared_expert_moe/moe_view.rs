use super::jit_request::{qwen35_moe_jit_load_requested, request_qwen35_moe_jit_load};
use super::page_cache::SparseExpertPageCache;
use super::routing::{hy3_sigmoid_topk_route, qwen35_softmax_topk_route};
use crate::engine::backend_runtime as qwen_moe_backend;
use crate::engine::dense_dispatch::gemv_f32;
use crate::engine::moe_profile::*;
#[cfg(feature = "cuda")]
use crate::engine::norm::add_f32_inplace;
use crate::engine::norm::{apply_model_gate_mul_inplace, scale_f32_inplace};
use crate::engine::scalar_gemv::{dot_k_block_row, gemv_generic};
use crate::engine::ModelArchitecture;
use rayon::prelude::*;
use rnb_loader::GGMLType;
use std::time::Instant;

use super::moe_types::{down_bytes_per_row, expert_bytes_per_row, SharedExpertMoEView};

mod fanout;
mod profile;
pub(super) mod shared_expert;
#[cfg(not(feature = "cuda"))]
use fanout::glm_iq_metal_batch_eligible;
use fanout::{compute_sparse_fanout, SparseFanoutResult};
use profile::record_decode_fanout_profile;
use shared_expert::compute_shared_expert;

fn qwen_moe_decode_inner_gemv_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_DECODE_INNER_GEMV")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(any(target_arch = "aarch64", test))]
fn qwen_moe_decode_gate_up_pair_gemv_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_DECODE_GATE_UP_PAIR_GEMV")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[cfg(any(target_arch = "aarch64", test))]
fn qwen_moe_decode_expert_local_rows_enabled() -> bool {
    crate::engine::policy::env_string("RNB_QWEN35_MOE_DECODE_EXPERT_LOCAL_ROWS")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(cfg!(target_arch = "aarch64"))
}

struct ExpertProfileAcc {
    out: Vec<f32>,
    wall_us: u128,
    high_us: u128,
    high_gate_up_us: u128,
    high_down_us: u128,
    low_us: u128,
    low_gate_up_us: u128,
    low_gate_up_row_us: u128,
    low_gate_up_tile_us: u128,
    low_gate_up_post_us: u128,
    low_shadow_down_us: u128,
    low_base_down_us: u128,
    high: u64,
    low: u64,
    skip: u64,
}

// Split sparse experts use separate gate, up, and down tensors. Their
// quantization is model-local. The shared expert runs beside the selected
// sparse experts for every token.

impl<'a> SharedExpertMoEView<'a> {
    /// Writes the route-weighted sparse-expert sum plus the shared-expert
    /// output into `out`.
    pub fn forward(&self, h: &[f32], out: &mut [f32]) {
        self.forward_impl(h, out, None, false, None);
    }

    pub(in crate::engine) fn forward_with_page_cache(
        &self,
        h: &[f32],
        out: &mut [f32],
        page_cache: Option<&SparseExpertPageCache>,
    ) {
        self.forward_impl(h, out, None, false, page_cache);
    }

    /// Computes the decode MoE block and adds it directly into `residual` when
    /// the CUDA full sparse+shared path is available.
    pub fn forward_add_residual(&self, h: &[f32], out: &mut [f32], residual: &mut [f32]) -> bool {
        self.forward_impl(h, out, Some(residual), false, None)
    }

    pub(in crate::engine) fn forward_add_residual_with_policy(
        &self,
        h: &[f32],
        out: &mut [f32],
        residual: &mut [f32],
        prefer_sparse_moe_cuda: bool,
        page_cache: Option<&SparseExpertPageCache>,
    ) -> bool {
        self.forward_impl(h, out, Some(residual), prefer_sparse_moe_cuda, page_cache)
    }

    fn forward_impl(
        &self,
        h: &[f32],
        out: &mut [f32],
        residual: Option<&mut [f32]>,
        _prefer_sparse_moe_cuda: bool,
        page_cache: Option<&SparseExpertPageCache>,
    ) -> bool {
        #[cfg(feature = "cuda")]
        let prefer_sparse_moe_cuda = _prefer_sparse_moe_cuda;
        assert_eq!(h.len(), self.n_embd);
        assert_eq!(out.len(), self.n_embd);
        if let Some(residual) = residual.as_ref() {
            assert_eq!(residual.len(), self.n_embd);
        }

        let profile_enabled = is_enabled();

        // 1. Router logits.
        let router_start = Instant::now();
        let mut logits_stack = [0.0f32; 256];
        let mut logits_vec;
        let logits: &mut [f32] = if self.n_expert <= logits_stack.len() {
            &mut logits_stack[..self.n_expert]
        } else {
            logits_vec = vec![0.0f32; self.n_expert];
            &mut logits_vec
        };
        gemv_f32(self.router_w, h, logits, self.n_expert, self.n_embd, 1);
        if profile_enabled {
            record_moe_profile("qwen35moe:decode:router", router_start.elapsed());
        }

        // 2. Route according to the GGUF MoE contract. Qwen uses global softmax;
        // Hy3/DeepSeek-style gating uses sigmoid scores, a selection-only bias,
        // selected-weight normalization, then the model's route scale.
        let routing_start = Instant::now();
        let mut probs_stack = [0.0f32; 256];
        let mut probs_vec;
        let probs: &mut [f32] = if self.n_expert <= probs_stack.len() {
            &mut probs_stack[..self.n_expert]
        } else {
            probs_vec = vec![0.0f32; self.n_expert];
            &mut probs_vec
        };
        let mut idx_stack = [0usize; 256];
        let mut idx_vec;
        let idx_all: &mut [usize] = if self.n_expert <= idx_stack.len() {
            &mut idx_stack[..self.n_expert]
        } else {
            idx_vec = vec![0usize; self.n_expert];
            &mut idx_vec
        };
        let selected_len = self.n_expert_used.min(self.n_expert);
        let mut exps_stack = [0.0f32; 32];
        let mut exps_vec;
        let exps: &mut [f32] = if selected_len <= exps_stack.len() {
            &mut exps_stack[..selected_len]
        } else {
            exps_vec = vec![0.0f32; selected_len];
            &mut exps_vec
        };
        let predictor_trace = crate::engine::moe_trace::predictor_trace_is_active();
        let jit_load_requested = qwen35_moe_jit_load_requested();
        let selected_len = if predictor_trace || jit_load_requested {
            match self.expert_gating_func {
                0 | 1 => qwen35_softmax_topk_route(
                    logits,
                    self.n_expert_used,
                    idx_all,
                    probs,
                    exps,
                    true,
                ),
                2 => hy3_sigmoid_topk_route(
                    logits,
                    self.router_selection_bias
                        .expect("sigmoid MoE routing requires exp_probs_b"),
                    self.n_expert_used,
                    self.expert_weights_norm,
                    self.expert_weights_scale,
                    crate::engine::policy::moe_adaptive_top_p(),
                    idx_all,
                    probs,
                    exps,
                ),
                other => panic!("unsupported MoE expert_gating_func {other}"),
            }
        } else {
            #[cfg(feature = "cuda")]
            {
                let sigmoid_mode = match self.expert_gating_func {
                    0 | 1 => false,
                    2 => true,
                    other => panic!("unsupported MoE expert_gating_func {other}"),
                };
                let (expert_ids, route_weights, retained_counts) =
                    crate::engine::cuda_runtime::moe_route_topk_f32(
                        logits,
                        self.router_selection_bias,
                        1,
                        self.n_expert,
                        self.n_expert_used,
                        sigmoid_mode,
                        self.expert_weights_norm,
                        if sigmoid_mode {
                            self.expert_weights_scale
                        } else {
                            1.0
                        },
                        if sigmoid_mode {
                            crate::engine::policy::moe_adaptive_top_p()
                        } else {
                            None
                        },
                    )
                    .unwrap_or_else(|err| panic!("CUDA MoE routing failed: {err}"));
                let retained = retained_counts[0] as usize;
                for (dst, &expert) in idx_all.iter_mut().zip(expert_ids.iter()).take(retained) {
                    *dst = expert as usize;
                }
                exps[..retained].copy_from_slice(&route_weights[..retained]);
                retained
            }
            #[cfg(not(feature = "cuda"))]
            {
                match self.expert_gating_func {
                    0 | 1 => qwen35_softmax_topk_route(
                        logits,
                        self.n_expert_used,
                        idx_all,
                        probs,
                        exps,
                        false,
                    ),
                    2 => hy3_sigmoid_topk_route(
                        logits,
                        self.router_selection_bias
                            .expect("sigmoid MoE routing requires exp_probs_b"),
                        self.n_expert_used,
                        self.expert_weights_norm,
                        self.expert_weights_scale,
                        crate::engine::policy::moe_adaptive_top_p(),
                        idx_all,
                        probs,
                        exps,
                    ),
                    other => panic!("unsupported MoE expert_gating_func {other}"),
                }
            }
        };
        let idx = &mut idx_all[..selected_len];
        let exps = &mut exps[..selected_len];
        if let Some(layer) = self.layer_idx {
            crate::engine::moe_trace::record_selection(layer, idx);
            if predictor_trace {
                crate::engine::moe_trace::record_predictor(layer, idx, probs);
            }
            if jit_load_requested {
                request_qwen35_moe_jit_load(
                    layer,
                    idx,
                    probs,
                    self.gate_exps_bytes,
                    self.up_exps_bytes,
                    self.down_exps_bytes,
                    self.n_ff * expert_bytes_per_row(self.n_embd, self.gate_quant, "gate_exps"),
                    self.n_ff * expert_bytes_per_row(self.n_embd, self.up_quant, "up_exps"),
                    self.n_embd * down_bytes_per_row(self.n_ff, self.down_quant),
                );
            }
        }
        if profile_enabled {
            record_moe_profile("qwen35moe:decode:routing", routing_start.elapsed());
        }

        let gate_scalar = if self.shared_expert_gated {
            let shared_gate_dot: f32 = h
                .iter()
                .zip(self.shared_input_scale.iter())
                .map(|(a, b)| a * b)
                .sum();
            1.0 / (1.0 + (-shared_gate_dot).exp())
        } else {
            1.0
        };

        #[cfg(not(feature = "cuda"))]
        let (fanout, shared_out) = if glm_iq_metal_batch_eligible(self, idx.len())
            || qwen_moe_backend::qwen_moe_decode_sparse_batch_enabled(idx.len(), true)
        {
            let fanout = compute_sparse_fanout(
                self,
                h,
                out,
                residual,
                idx,
                exps,
                gate_scalar,
                profile_enabled,
                false,
            );
            let shared_out = if matches!(
                &fanout,
                SparseFanoutResult::Complete | SparseFanoutResult::ResidualComplete
            ) {
                Vec::new()
            } else {
                let shared_in_sparse = matches!(
                    &fanout,
                    SparseFanoutResult::Computed {
                        shared_in_sparse_gpu: true,
                        ..
                    }
                );
                compute_shared_expert(
                    self,
                    h,
                    gate_scalar,
                    shared_in_sparse,
                    profile_enabled,
                    false,
                )
            };
            (fanout, shared_out)
        } else {
            rayon::join(
                || {
                    compute_sparse_fanout(
                        self,
                        h,
                        out,
                        residual,
                        idx,
                        exps,
                        gate_scalar,
                        profile_enabled,
                        false,
                    )
                },
                || compute_shared_expert(self, h, gate_scalar, false, profile_enabled, false),
            )
        };

        #[cfg(feature = "cuda")]
        let fanout = compute_sparse_fanout(
            self,
            h,
            out,
            residual,
            idx,
            exps,
            gate_scalar,
            profile_enabled,
            prefer_sparse_moe_cuda,
        );
        if let (Some(page_cache), Some(layer_index)) = (page_cache, self.layer_idx) {
            page_cache.touch(layer_index, idx);
        }
        if matches!(fanout, SparseFanoutResult::ResidualComplete) {
            return true;
        }
        if matches!(fanout, SparseFanoutResult::Complete) {
            return false;
        }
        let SparseFanoutResult::Computed {
            per_expert,
            fanout_us,
            shared_in_sparse_gpu,
        } = fanout
        else {
            return false;
        };
        #[cfg(not(feature = "cuda"))]
        let _ = shared_in_sparse_gpu;
        record_decode_fanout_profile(&per_expert, fanout_us, profile_enabled);

        #[cfg(feature = "cuda")]
        let shared_out = compute_shared_expert(
            self,
            h,
            gate_scalar,
            shared_in_sparse_gpu,
            profile_enabled,
            prefer_sparse_moe_cuda,
        );

        // 5. out = sparse_sum + shared_out
        let reduce_start = Instant::now();
        out.fill(0.0);
        for eo in &per_expert {
            for i in 0..self.n_embd {
                out[i] += eo.out[i];
            }
        }
        for i in 0..self.n_embd {
            out[i] += shared_out[i];
        }
        if profile_enabled {
            record_moe_profile("qwen35moe:decode:reduce", reduce_start.elapsed());
        }
        false
    }
}

#[cfg(test)]
mod decode_policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn qwen_moe_decode_gemv_paths_default_on_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "RNB_QWEN35_MOE_DECODE_INNER_GEMV",
            "RNB_QWEN35_MOE_DECODE_GATE_UP_PAIR_GEMV",
        ];
        let previous: Vec<_> = keys
            .iter()
            .map(|key| crate::engine::policy::env_string(key))
            .collect();
        for key in keys {
            std::env::remove_var(key);
        }

        assert!(qwen_moe_decode_inner_gemv_enabled());
        assert!(qwen_moe_decode_gate_up_pair_gemv_enabled());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_QWEN35_MOE_DECODE_INNER_GEMV", value);
            std::env::set_var("RNB_QWEN35_MOE_DECODE_GATE_UP_PAIR_GEMV", value);
            assert!(
                !qwen_moe_decode_inner_gemv_enabled(),
                "{value} should opt out inner GEMV"
            );
            assert!(
                !qwen_moe_decode_gate_up_pair_gemv_enabled(),
                "{value} should opt out paired gate/up GEMV"
            );
        }

        std::env::set_var("RNB_QWEN35_MOE_DECODE_INNER_GEMV", "1");
        std::env::set_var("RNB_QWEN35_MOE_DECODE_GATE_UP_PAIR_GEMV", "1");
        assert!(qwen_moe_decode_inner_gemv_enabled());
        assert!(qwen_moe_decode_gate_up_pair_gemv_enabled());

        for (key, previous) in keys.iter().zip(previous) {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn qwen_moe_decode_expert_local_rows_defaults_on_for_aarch64_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_QWEN35_MOE_DECODE_EXPERT_LOCAL_ROWS";
        let previous = crate::engine::policy::env_string(key);
        std::env::remove_var(key);

        assert_eq!(
            qwen_moe_decode_expert_local_rows_enabled(),
            cfg!(target_arch = "aarch64")
        );
        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !qwen_moe_decode_expert_local_rows_enabled(),
                "{value} should opt out expert-local row scheduling"
            );
        }
        std::env::set_var(key, "1");
        assert!(qwen_moe_decode_expert_local_rows_enabled());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}
