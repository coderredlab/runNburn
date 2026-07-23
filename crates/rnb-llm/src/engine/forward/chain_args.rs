//! Architecture-agnostic helper that materializes the inputs to
//! `dense_q4k_attention_output_gelu_ffn_norm_residual_if_supported`
//! (the "chain function").
//!
//! cu58 milestone 1 — the caller in `decode.rs` used to hand-craft ~130 lines of
//! Gemma-only setup before calling the chain function. This module centralizes
//! that setup so other architectures (Llama dense, …) can share the same
//! entry point. The contract is also reused by `chain_function_active`, which
//! drives the cu57 `chain_emits_hidden_carrier` wire-readiness signal.
//!
//! Step 1 only ports the Gemma family path verbatim — `compute_chain_function_args`
//! returns `None` for other architectures, so existing behaviour is unchanged
//! (bit-exact with cu57 step 68).

#[cfg(feature = "cuda")]
use super::super::backend_runtime;
use super::super::cpu_runtime::kernels;
use super::super::layer_weights::AttentionLayerWeights;
use super::super::models::gemma::{
    active_layer_output_scale, gemma_effective_skip_ffn_decode_enabled,
    gemma_effective_unit_offset_post_attn_enabled, gemma_post_attn_blend_source_decode_enabled,
    gemma_post_attn_decode_plain_enabled, gemma_pre_residual_blend_source_decode_enabled,
    gemma_skip_post_attn_decode_enabled, gemma_skip_post_attn_enabled, select_ffn_pre_norm_weight,
    use_gemma_block_semantics, GemmaPerLayerBase, GemmaPerLayerWeights, GemmaRuntimeFlavor,
};
use super::super::quantized_weight_types::QuantizedWeight;
use super::super::types::ModelMetadata;
use rnb_loader::Architecture as ModelArchitecture;

/// Resolved arguments needed by
/// `dense_q4k_attention_output_gelu_ffn_norm_residual_if_supported`.
///
/// Lifetime `'a` is bound to the source `AttentionLayerWeights`, the PLE fusion
/// payload, and the `&'a ModelMetadata` (for `ple_input`).
pub(in crate::engine) struct ChainArgs<'a> {
    pub o_weight: &'a QuantizedWeight,
    pub gate_weight: &'a QuantizedWeight,
    pub up_weight: &'a QuantizedWeight,
    pub down_weight: &'a QuantizedWeight,

    pub post_attn_norm: Option<&'a [f32]>,
    pub ffn_norm: &'a [f32],
    pub post_ffn_norm: Option<&'a [f32]>,

    pub ple_gate_weight: Option<&'a QuantizedWeight>,
    pub ple_proj_weight: Option<&'a QuantizedWeight>,
    pub ple_post_norm_weight: Option<&'a [f32]>,
    pub ple_input: Option<&'a [f32]>,
    pub ple_input_device_offset: Option<usize>,
    pub ple_dim: usize,
    pub ple_fused: bool,

    pub unit_offset_post_attn_norm: bool,
    pub unit_offset_ffn_norm: bool,
    pub unit_offset_ple_norm: bool,

    pub layer_output_scale: Option<f32>,

    pub hidden_carrier_dev: Option<u64>,
    pub skip_h2d_hidden: bool,
    pub skip_d2h_hidden: bool,
    pub attn_out_dev_carrier: Option<u64>,
    pub dense_chain_graph_allowed: bool,
    pub layer_segment_graph_allowed: bool,
    pub layer_segment_graph_request:
        Option<super::super::decode_layer_graph::Cu71LayerSegmentGraphRequest>,

    // cu58 Task 6 fix: FFN activation 분기. true=gelu (Gemma), false=silu (Llama).
    // 이전엔 chain function 본문에서 true hardcoded — silu arch 진입 시 wrong
    // activation → garbage 의 root cause.
    pub ffn_uses_gelu: bool,
}

/// Helper inputs that the caller already has on hand. These are *not*
/// derivable from `architecture` alone (per-layer state, runtime flavor,
/// device-side carrier ownership, …).
pub(in crate::engine) struct ChainCallerCtx<'a> {
    pub architecture: ModelArchitecture,
    pub layer_idx: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub w: &'a AttentionLayerWeights,
    pub gemma_runtime_flavor: GemmaRuntimeFlavor,
    pub ple_fusion: Option<(&'a GemmaPerLayerBase, &'a GemmaPerLayerWeights)>,
    pub ple_input_device_offset: Option<usize>,
    pub metadata: &'a ModelMetadata,
    /// True only after `decode_attention_compute` reported the attention output
    /// already lives in `attn_out_carrier_dev`.
    pub attn_on_device: bool,
    /// Pre-acquired `attn_out` device carrier. `None` if not active.
    pub attn_out_carrier_dev: Option<u64>,
    pub has_gated_attn: bool,
    pub gemma4_reuse_q_only: bool,
    pub gemma4_attn_rot_active: bool,
    pub has_sliding_window: bool,
    pub long_kv_split_preferred: bool,
}

/// `true` iff `compute_chain_function_args` would return `Some` — i.e. the
/// chain function will be called for this `(arch, layer)` and the resulting
/// `hidden_carrier_dev` will be written. cu57 `chain_emits_hidden_carrier`
/// single source.
///
/// **Light-weight** — guard 만 평가. carrier acquire / weight tensor resolve /
/// norm slice 변환은 skip. cu58 step 3c — chain_function_active 가 매 layer 의
/// try_rms_norm cuda 호출 전 평가 + 또 chain function call 직전 평가 = double
/// evaluation. 본래 compute_chain_function_args 호출이면 매 layer carrier
/// acquire / weight resolve cost 가 두 번 (Gemma4 +22.8% decode 회귀의 원인).
/// invariant — chain_function_active(ctx) == compute_chain_function_args(ctx).is_some().
pub(in crate::engine) fn chain_function_active(ctx: &ChainCallerCtx<'_>) -> bool {
    #[cfg(feature = "cuda")]
    {
        // cu59 axis A — A path 회귀용 same-binary 토글.
        // RNB_CU58_HELPER_DISABLE=1 ⇒ helper bypass, cu57 의 chain env OFF baseline 동등.
        if crate::engine::policy::env_string("RNB_CU58_HELPER_DISABLE").as_deref() == Some("1") {
            return false;
        }
        // Common guards — MoE / Nemotron / chain env OFF ⇒ false.
        if ctx.w.moe.is_some() || ctx.w.shared_expert_moe.is_some() {
            return false;
        }
        if matches!(ctx.architecture, ModelArchitecture::NemotronHMoE) {
            return false;
        }
        let chain_env_on = crate::engine::policy::cuda_decode_device_chain_enabled()
            && crate::engine::policy::cuda_decode_device_out_scale_enabled();
        if !chain_env_on {
            return false;
        }
        match ctx.architecture {
            ModelArchitecture::Gemma | ModelArchitecture::Gemma4 => {
                // Gemma 의 추가 skip guard (build_gemma_args 의 None 반환 조건).
                if gemma_effective_skip_ffn_decode_enabled(
                    ctx.architecture,
                    ctx.gemma_runtime_flavor,
                    ctx.layer_idx,
                ) {
                    return false;
                }
                if gemma_post_attn_blend_source_decode_enabled(ctx.layer_idx) {
                    return false;
                }
                if gemma_pre_residual_blend_source_decode_enabled(ctx.layer_idx) {
                    return false;
                }
                true
            }
            ModelArchitecture::LLaMA => true,
            _ => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ctx;
        false
    }
}

/// Build the chain function arguments for this `(arch, layer)`, or `None` if
/// the chain function should not be entered (MoE, Nemotron, env-gated off,
/// Gemma-specific layer-skip guards, …).
pub(in crate::engine) fn compute_chain_function_args<'a>(
    ctx: &ChainCallerCtx<'a>,
) -> Option<ChainArgs<'a>> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ctx;
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        // cu59 axis A — A path 회귀용 same-binary 토글.
        // RNB_CU58_HELPER_DISABLE=1 ⇒ helper bypass, cu57 의 chain env OFF baseline 동등.
        if crate::engine::policy::env_string("RNB_CU58_HELPER_DISABLE").as_deref() == Some("1") {
            return None;
        }

        // cu59 axis A — helper_args sub-phase timing 시작 (early bypass 이후).
        let diag_active = super::chain_diag::is_active();
        let t_helper_args_start = if diag_active {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Common guards — MoE / Nemotron / chain env OFF ⇒ None.
        if ctx.w.moe.is_some() || ctx.w.shared_expert_moe.is_some() {
            return None;
        }
        if matches!(ctx.architecture, ModelArchitecture::NemotronHMoE) {
            return None;
        }
        let chain_env_on = crate::engine::policy::cuda_decode_device_chain_enabled()
            && crate::engine::policy::cuda_decode_device_out_scale_enabled();
        if !chain_env_on {
            return None;
        }

        let result = match ctx.architecture {
            ModelArchitecture::Gemma | ModelArchitecture::Gemma4 => build_gemma_args(ctx),
            ModelArchitecture::LLaMA => build_llama_args(ctx),
            // Qwen / Phi dense paths land in a later step.
            _ => None,
        };

        if let (Some(start), Some(_)) = (t_helper_args_start, result.as_ref()) {
            let helper_args_us = start.elapsed().as_micros() as u64;
            super::chain_diag::stash_phase_us(super::chain_diag::Phase::HelperArgs, helper_args_us);
        }

        result
    }
}

#[cfg(feature = "cuda")]
fn build_llama_args<'a>(ctx: &ChainCallerCtx<'a>) -> Option<ChainArgs<'a>> {
    let w = ctx.w;
    let layer_idx = ctx.layer_idx;

    // cu59 axis A — sub-phase timing.
    let diag_active = super::chain_diag::is_active();

    // Llama dense only — MoE 는 위 compute_chain_function_args 의 공통 guard 가
    // 이미 None 처리. PLE / post_attn_norm / post_ffn_norm / layer_output_scale
    // 모두 없음 — Gemma 특화 인자 = None / false 의 단순 path.

    // WeightResolve — w.o_weight / ffn_gate / ffn_up / ffn_down reference (단순 borrow).
    let t_weight = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let o_weight = &w.o_weight;
    let gate_weight = &w.ffn_gate_weight;
    let up_weight = &w.ffn_up_weight;
    let down_weight = &w.ffn_down_weight;
    if let Some(t) = t_weight {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::WeightResolve,
            t.elapsed().as_micros() as u64,
        );
    }

    // NormSlice
    let t_norm = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let ffn_norm_data = kernels::tensor_as_f32_slice(&w.ffn_norm);
    if let Some(t) = t_norm {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::NormSlice,
            t.elapsed().as_micros() as u64,
        );
    }

    let skip_h2d_hidden = layer_idx > 0;
    let skip_d2h_hidden = layer_idx + 1 < ctx.num_layers;

    // Acquire
    let t_acquire = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let hidden_carrier_dev = {
        let bytes = ctx.hidden_dim * std::mem::size_of::<f32>();
        backend_runtime::acquire_decode_hidden_carrier(bytes).ok()
    };
    if let Some(t) = t_acquire {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::Acquire,
            t.elapsed().as_micros() as u64,
        );
    }

    let attn_out_dev_carrier = if ctx.attn_on_device {
        ctx.attn_out_carrier_dev
    } else {
        None
    };

    Some(ChainArgs {
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        post_attn_norm: None, // Llama 없음
        ffn_norm: ffn_norm_data,
        post_ffn_norm: None, // Llama 없음
        ple_gate_weight: None,
        ple_proj_weight: None,
        ple_post_norm_weight: None,
        ple_input: None,
        ple_input_device_offset: None,
        ple_dim: 0,
        ple_fused: false,
        unit_offset_post_attn_norm: false, // Llama RMS 는 unit offset 없음
        unit_offset_ffn_norm: false,
        unit_offset_ple_norm: false,
        layer_output_scale: None, // Llama 없음
        hidden_carrier_dev,
        skip_h2d_hidden,
        skip_d2h_hidden,
        attn_out_dev_carrier,
        dense_chain_graph_allowed: false,
        layer_segment_graph_allowed: false,
        layer_segment_graph_request: None,
        ffn_uses_gelu: false, // Llama FFN = silu
    })
}

#[cfg(feature = "cuda")]
fn build_gemma_args<'a>(ctx: &ChainCallerCtx<'a>) -> Option<ChainArgs<'a>> {
    let w = ctx.w;
    let layer_idx = ctx.layer_idx;
    let architecture = ctx.architecture;

    // cu59 axis A — sub-phase timing.
    let diag_active = super::chain_diag::is_active();

    // Gemma-specific entry guards (cu57 decode.rs:454-462).
    if !use_gemma_block_semantics(architecture) {
        return None;
    }
    if gemma_effective_skip_ffn_decode_enabled(architecture, ctx.gemma_runtime_flavor, layer_idx) {
        return None;
    }
    if gemma_post_attn_blend_source_decode_enabled(layer_idx) {
        return None;
    }
    if gemma_pre_residual_blend_source_decode_enabled(layer_idx) {
        return None;
    }

    // WeightResolve — Gemma 의 4개 dense weight reference (Llama 와 동일 패턴).
    let t_weight = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let o_weight = &w.o_weight;
    let gate_weight = &w.ffn_gate_weight;
    let up_weight = &w.ffn_up_weight;
    let down_weight = &w.ffn_down_weight;
    if let Some(t) = t_weight {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::WeightResolve,
            t.elapsed().as_micros() as u64,
        );
    }

    let skip_post_attn =
        gemma_skip_post_attn_enabled(layer_idx) || gemma_skip_post_attn_decode_enabled(layer_idx);
    let post_attn_norm = if skip_post_attn {
        None
    } else {
        w.post_attn_norm.as_ref().map(kernels::tensor_as_f32_slice)
    };
    let unit_offset_post_attn_norm = post_attn_norm.is_some()
        && !gemma_post_attn_decode_plain_enabled(layer_idx)
        && gemma_effective_unit_offset_post_attn_enabled(
            architecture,
            ctx.gemma_runtime_flavor,
            layer_idx,
        );

    // NormSlice — ffn_norm 만 측정 (post_attn / post_ffn 추가 slice 는 skip).
    let ffn_norm_tensor = select_ffn_pre_norm_weight(w, architecture);
    let t_norm = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm_tensor);
    if let Some(t) = t_norm {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::NormSlice,
            t.elapsed().as_micros() as u64,
        );
    }
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);

    let unit_offset_ffn_norm = super::super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
        || super::super::policy::gemma_unit_offset_norm_enabled()
        || super::super::policy::gemma_unit_offset_main_norm_enabled();

    // PLE fusion — mirrors cu57 decode.rs:483-501.
    let ple_args = ctx.ple_fusion.and_then(|(base, gemma)| {
        let ple_dim = ctx.metadata.embedding_length_per_layer_input;
        let base_off = layer_idx.checked_mul(ple_dim)?;
        let ple_input = base.mixed.get(base_off..base_off + ple_dim)?;
        let layer = gemma.layers.get(layer_idx)?;
        Some((
            &layer.inp_gate,
            &layer.proj,
            kernels::tensor_as_f32_slice(&layer.post_norm),
            ple_input,
            ple_dim,
        ))
    });
    let ple_fused = ple_args.is_some();
    let (ple_gate_weight, ple_proj_weight, ple_post_norm_weight, ple_input, ple_dim) = ple_args
        .map(|(gate, proj, norm, input, dim)| {
            (Some(gate), Some(proj), Some(norm), Some(input), dim)
        })
        .unwrap_or((None, None, None, None, 0));

    let unit_offset_ple_norm = super::super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
        || super::super::policy::gemma_unit_offset_norm_enabled()
        || super::super::policy::gemma_unit_offset_main_norm_enabled();

    // Carrier policy — cu46 step 27-29 + cu57 contract.
    // (chain_env_on is already true at this point.)
    let skip_h2d_hidden = layer_idx > 0;
    let skip_d2h_hidden = layer_idx + 1 < ctx.num_layers;

    // Acquire — cu59 axis A timing.
    let t_acquire = if diag_active {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let hidden_carrier_dev = {
        let bytes = ctx.hidden_dim * std::mem::size_of::<f32>();
        backend_runtime::acquire_decode_hidden_carrier(bytes).ok()
    };
    if let Some(t) = t_acquire {
        super::chain_diag::stash_phase_us(
            super::chain_diag::Phase::Acquire,
            t.elapsed().as_micros() as u64,
        );
    }

    // cu44 step 20: Gemma4 layer_output_scale device-side apply.
    // out_scale env is implicit in chain_env_on above.
    let layer_output_scale =
        active_layer_output_scale(w.out_scale.as_ref(), layer_idx).map(|s| s[0]);

    // cu47 step 33: attention forward already wrote to device — pass carrier.
    let attn_out_dev_carrier = if ctx.attn_on_device {
        ctx.attn_out_carrier_dev
    } else {
        None
    };
    let dense_chain_graph_supported =
        super::super::decode_layer_graph::cu69_dense_chain_graph_decision(
            super::super::decode_layer_graph::Cu69LayerGraphRequest {
                dense_chain_graph_enabled: true,
                architecture_is_gemma4: matches!(architecture, ModelArchitecture::Gemma4),
                device_qkv_enabled: crate::engine::tuning_runtime::cu65_device_qkv_enabled(),
                chain_emits_hidden_carrier: true,
                rms_used_cuda: true,
                attn_out_on_device: ctx.attn_on_device && attn_out_dev_carrier.is_some(),
                hidden_carrier_available: hidden_carrier_dev.is_some(),
                skip_h2d_hidden,
                skip_d2h_hidden,
                has_gated_attn: ctx.has_gated_attn,
                gemma4_reuse_q_only: ctx.gemma4_reuse_q_only,
                gemma4_attn_rot_active: ctx.gemma4_attn_rot_active,
                has_sliding_window: ctx.has_sliding_window,
            },
        ) == super::super::decode_layer_graph::Cu69DenseChainGraphDecision::Eligible;
    let dense_chain_graph_allowed = crate::engine::tuning_runtime::cu69_dense_chain_graph_enabled()
        && dense_chain_graph_supported;
    let layer_segment_graph_request =
        super::super::decode_layer_graph::Cu71LayerSegmentGraphRequest {
            layer_segment_graph_enabled:
                crate::engine::tuning_runtime::cu71_layer_segment_graph_enabled(),
            architecture_is_gemma4: matches!(architecture, ModelArchitecture::Gemma4),
            device_qkv_enabled: crate::engine::tuning_runtime::cu65_device_qkv_enabled(),
            chain_emits_hidden_carrier: true,
            rms_used_cuda: true,
            attn_out_on_device: ctx.attn_on_device && attn_out_dev_carrier.is_some(),
            hidden_carrier_available: hidden_carrier_dev.is_some(),
            skip_h2d_hidden,
            skip_d2h_hidden,
            has_gated_attn: ctx.has_gated_attn,
            gemma4_reuse_q_only: ctx.gemma4_reuse_q_only,
            gemma4_attn_rot_active: ctx.gemma4_attn_rot_active,
            has_sliding_window: ctx.has_sliding_window,
            long_kv_split_preferred: ctx.long_kv_split_preferred,
            dense_chain_graph_supported,
        };
    let layer_segment_graph_allowed =
        super::super::decode_layer_graph::cu71_layer_segment_graph_decision(
            layer_segment_graph_request,
        ) == super::super::decode_layer_graph::Cu71LayerSegmentGraphDecision::Eligible;

    Some(ChainArgs {
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        post_attn_norm,
        ffn_norm: ffn_norm_data,
        post_ffn_norm,
        ple_gate_weight,
        ple_proj_weight,
        ple_post_norm_weight,
        ple_input,
        ple_input_device_offset: ctx.ple_input_device_offset,
        ple_dim,
        ple_fused,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_ple_norm,
        layer_output_scale,
        hidden_carrier_dev,
        skip_h2d_hidden,
        skip_d2h_hidden,
        attn_out_dev_carrier,
        dense_chain_graph_allowed,
        layer_segment_graph_allowed,
        layer_segment_graph_request: Some(layer_segment_graph_request),
        ffn_uses_gelu: true, // Gemma FFN = gelu
    })
}
