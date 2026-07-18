//! Gemma prefill QKV + attention + dense-chain CUDA fast paths.

use super::super::*;

pub(in crate::engine) struct PrefillFusedQkvDenseChain {
    pub(in crate::engine) hidden: Tensor,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
    pub(in crate::engine) gemma4_ple_fused: bool,
    pub(in crate::engine) gemma4_output_scale_fused: bool,
    #[cfg(feature = "cuda")]
    pub(in crate::engine) device_output: Option<backend_runtime::NemotronDeviceLayerOutput>,
}

pub(super) struct PrefillFusedHidden {
    pub(super) hidden: Tensor,
    pub(super) gemma4_ple_fused: bool,
    pub(super) gemma4_output_scale_fused: bool,
    #[cfg(feature = "cuda")]
    pub(super) device_output: Option<backend_runtime::NemotronDeviceLayerOutput>,
}

#[cfg(feature = "cuda")]
pub(in crate::engine) struct Gemma4PrefillPleFusion<'a> {
    pub(in crate::engine) base: &'a GemmaPerLayerBase,
    pub(in crate::engine) weights: &'a GemmaPerLayerWeights,
}

#[cfg(not(feature = "cuda"))]
pub(in crate::engine) struct Gemma4PrefillPleFusion<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

struct PreparedGemma4PrefillPleFusion<'a> {
    gate_weight: &'a QuantizedWeight,
    proj_weight: &'a QuantizedWeight,
    post_norm_weight: &'a [f32],
    input: Vec<f32>,
    dim: usize,
}

fn gemma_reuse_q_dense_chain_requested() -> bool {
    let parse = |name: &str| {
        std::env::var(name).ok().map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
    };

    parse("RNB_CUDA_GEMMA_REUSE_Q_DENSE_CHAIN")
        .or_else(|| parse("RNB_CUDA_GEMMA_REUSE_Q_HD512_DENSE_CHAIN"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn gemma4_prefill_ple_dense_chain_requested() -> bool {
    std::env::var("RNB_CUDA_GEMMA_PLE_DENSE_CHAIN")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn gemma4_f32_ple_device_fusion_requested() -> bool {
    // cu110(2026-06-06): default ON 승격. Gemma4 device hidden carrier 가 chain 을
    // 끊지 않으려면 F32 PLE layer 도 device fusion 으로 처리돼야 한다(host reject 시
    // 매 PLE layer 마다 host materialize 로 carrier 단절). E2B 측정에서 chat
    // token-identical + prefill -14.7%(ABAB median, 겹침0). cu28 의 hard gate 는
    // raw-prompt token parity 기준이었고 cu26 정확도 정책(chat 의미 동등성)으로
    // 무효화됨. 자세히 docs/perf-journal/pc-cuda-gemma4.md cu110.
    std::env::var("RNB_CUDA_GEMMA_F32_PLE_DEVICE_FUSION")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn gemma4_partial_ple_host_carrier_requested() -> bool {
    std::env::var("RNB_CUDA_GEMMA_PARTIAL_PLE_CARRIER")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(any(feature = "cuda", test))]
fn gemma4_device_hidden_carrier_requested() -> bool {
    // cu110(2026-06-06): default ON 승격. layer-to-layer device hidden carrier 로
    // 매 layer host materialize(H2D/D2H 왕복)를 제거해 prefill dispatch gap 을 줄인다.
    // cu24 에서 한 번 ON 승격됐다가 cu25-26 의 K/V corruption(NaN, 32767)·OOM 으로
    // OFF 회귀했으나, cu27-28 fix 후 E2B 에서 chat token-identical + prefill -14.7%
    // (ABAB median, 겹침0, 100 decode OOM 없음) 재확인. 두 env 모두 0/false/off/no 로
    // opt-out 가능. 자세히 docs/perf-journal/pc-cuda-gemma4.md cu110.
    let carrier = std::env::var("RNB_CUDA_GEMMA_DEVICE_HIDDEN_CARRIER")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true);
    let qkv_carrier = std::env::var("RNB_CUDA_GEMMA_QKV_DEVICE_HIDDEN_CARRIER")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true);
    carrier && qkv_carrier
}

#[cfg(any(feature = "cuda", test))]
const GEMMA4_QKV_DEVICE_INPUT_DEFAULT: bool = true;
#[cfg(any(feature = "cuda", test))]
const GEMMA4_HD512_DEVICE_HIDDEN_CARRIER_DEFAULT: bool = true;
#[cfg(any(feature = "cuda", test))]
const GEMMA4_REUSE_Q_DEVICE_HIDDEN_CARRIER_DEFAULT: bool = true;

#[cfg(any(feature = "cuda", test))]
fn parse_env_bool_or(value: Option<&str>, default: bool) -> bool {
    value
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(default)
}

#[cfg(any(feature = "cuda", test))]
fn env_bool_or(name: &str, default: bool) -> bool {
    parse_env_bool_or(std::env::var(name).ok().as_deref(), default)
}

#[cfg(feature = "cuda")]
fn gemma4_qkv_device_input_requested() -> bool {
    // cu115(2026-06-08): default ON after product chat semantic-equivalence
    // smoke. This removes the remaining activation H2D hop. Raw-prompt token
    // drift is diagnostic only; product promotion follows chat meaning.
    env_bool_or(
        "RNB_CUDA_GEMMA_QKV_DEVICE_INPUT",
        GEMMA4_QKV_DEVICE_INPUT_DEFAULT,
    )
}

#[cfg(feature = "cuda")]
fn gemma4_hd512_device_hidden_carrier_requested() -> bool {
    // cu115(2026-06-08): hd512 full-attention layers must keep the carrier alive;
    // otherwise E2B falls back to host materialization between full/SWA blocks.
    env_bool_or(
        "RNB_CUDA_GEMMA_HD512_DEVICE_HIDDEN_CARRIER",
        GEMMA4_HD512_DEVICE_HIDDEN_CARRIER_DEFAULT,
    )
}

#[cfg(feature = "cuda")]
fn gemma4_reuse_q_device_hidden_carrier_requested() -> bool {
    // cu115(2026-06-08): shared-KV reuse-Q attention is part of the same
    // layer-to-layer activation chain, with explicit env opt-out retained.
    env_bool_or(
        "RNB_CUDA_GEMMA_REUSE_Q_DEVICE_HIDDEN_CARRIER",
        GEMMA4_REUSE_Q_DEVICE_HIDDEN_CARRIER_DEFAULT,
    )
}

#[cfg(feature = "cuda")]
fn gemma4_prefill_ple_fusion_trace_enabled() -> bool {
    std::env::var("RNB_CUDA_GEMMA_PLE_FUSION_TRACE")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn trace_gemma4_prefill_ple_fusion(layer_idx: usize, message: impl std::fmt::Display) {
    if gemma4_prefill_ple_fusion_trace_enabled() {
        eprintln!("[gemma4-ple-fusion] layer={} {}", layer_idx, message);
    }
}

#[cfg(feature = "cuda")]
fn prepare_gemma4_prefill_ple_fusion<'a>(
    fusion: Option<&'a Gemma4PrefillPleFusion<'a>>,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    w: &AttentionLayerWeights,
    layer_idx: usize,
    seq_len: usize,
) -> Option<PreparedGemma4PrefillPleFusion<'a>> {
    let Some(fusion) = fusion else {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=no_fusion");
        return None;
    };
    if !gemma4_prefill_ple_dense_chain_requested() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=disabled");
        return None;
    }
    if !matches!(architecture, ModelArchitecture::Gemma4) {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=arch");
        return None;
    }
    if gemma_ple_before_layer() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=before_layer");
        return None;
    }
    if gemma_ple_after_out_scale() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=after_out_scale");
        return None;
    }
    if gemma_ple_use_layer_input() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=layer_input");
        return None;
    }
    if gemma_ple_pre_norm_input() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=pre_norm_input");
        return None;
    }
    if !gemma_ple_layer_enabled(layer_idx) {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=layer_disabled");
        return None;
    }
    if gemma_ple_global_only(metadata, w) {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=global_only");
        return None;
    }
    if dump_bin_dir().is_some() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=dump_bin");
        return None;
    }
    let dim = metadata.embedding_length_per_layer_input;
    let Some(total_dim) = metadata.num_layers.checked_mul(dim) else {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=total_dim_overflow");
        return None;
    };
    if dim == 0 || layer_idx >= fusion.weights.layers.len() {
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "reject=layer_bounds dim={} layers={}",
                dim,
                fusion.weights.layers.len()
            ),
        );
        return None;
    }
    let Some(expected_base_len) = seq_len.checked_mul(total_dim) else {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=base_len_overflow");
        return None;
    };
    if fusion.base.mixed.len() < expected_base_len {
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "reject=base_len got={} expected={} seq_len={} total_dim={}",
                fusion.base.mixed.len(),
                expected_base_len,
                seq_len,
                total_dim
            ),
        );
        return None;
    }
    let layer = &fusion.weights.layers[layer_idx];
    if layer.inp_gate.rows != dim
        || layer.inp_gate.cols != metadata.hidden_dim
        || layer.proj.rows != metadata.hidden_dim
        || layer.proj.cols != dim
        || layer.post_norm.numel() != metadata.hidden_dim
    {
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "reject=shape gate={}x{} proj={}x{} post_norm={} expected_gate={}x{} expected_proj={}x{}",
                layer.inp_gate.rows,
                layer.inp_gate.cols,
                layer.proj.rows,
                layer.proj.cols,
                layer.post_norm.numel(),
                dim,
                metadata.hidden_dim,
                metadata.hidden_dim,
                dim
            ),
        );
        return None;
    }

    let mut input = vec![0.0f32; seq_len * dim];
    for token_idx in 0..seq_len {
        let src = token_idx * total_dim + layer_idx * dim;
        let dst = token_idx * dim;
        input[dst..dst + dim].copy_from_slice(&fusion.base.mixed[src..src + dim]);
    }
    trace_gemma4_prefill_ple_fusion(
        layer_idx,
        format!(
            "prepared seq_len={} dim={} hidden_dim={}",
            seq_len, dim, metadata.hidden_dim
        ),
    );
    Some(PreparedGemma4PrefillPleFusion {
        gate_weight: &layer.inp_gate,
        proj_weight: &layer.proj,
        post_norm_weight: kernels::tensor_as_f32_slice(&layer.post_norm),
        input,
        dim,
    })
}

#[cfg(feature = "cuda")]
fn gemma4_prepared_ple_device_fusion_allowed(
    layer_idx: usize,
    prepared_ple: Option<&PreparedGemma4PrefillPleFusion<'_>>,
) -> bool {
    let Some(ple) = prepared_ple else {
        return true;
    };
    let has_f32 = matches!(ple.gate_weight.ggml_type, GGMLType::F32)
        || matches!(ple.proj_weight.ggml_type, GGMLType::F32);
    if has_f32 && !gemma4_f32_ple_device_fusion_requested() {
        trace_gemma4_prefill_ple_fusion(layer_idx, "reject=f32_ple_device_fusion_disabled");
        return false;
    }
    true
}

#[cfg(feature = "cuda")]
fn gemma4_partial_ple_host_carrier_active(
    prepared_ple: Option<&PreparedGemma4PrefillPleFusion<'_>>,
    device_ple_allowed: bool,
) -> bool {
    gemma4_partial_ple_host_carrier_requested() && prepared_ple.is_some() && !device_ple_allowed
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_q4k_f16_reuse_q_hd512_dense_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    pos_start: usize,
    norm_eps: f32,
    _ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedHidden>> {
    #[cfg(feature = "cuda")]
    let ple_fusion = _ple_fusion;
    if !gemma_reuse_q_dense_chain_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !gemma4_reuse_q_only
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 512
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || super::super::policy::gemma_reused_reapply_k_norm_enabled()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let Some(q_norm) = &w.q_norm else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim != layout.head_dim {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let q_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    let prepared_ple: Option<PreparedGemma4PrefillPleFusion<'_>> = None;
    let mut hidden_out = kernels::tensor_as_f32_slice(hidden).to_vec();
    #[cfg(feature = "cuda")]
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    #[cfg(feature = "cuda")]
    let host_post_ple_pending = ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled();
    #[cfg(feature = "cuda")]
    if gemma4_device_hidden_carrier_requested()
        && gemma4_reuse_q_device_hidden_carrier_requested()
        && gemma4_hd512_device_hidden_carrier_requested()
        && !host_post_ple_pending
        && !layer_trace_enabled()
    {
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "device_output_try ple_prepared={} out_scale={}",
                prepared_ple.is_some(),
                layer_out_scale.is_some()
            ),
        );
        let device_output =
            backend_runtime::prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output_if_supported(
                    &w.q_weight,
                    kernels::tensor_as_f32_slice(hidden),
                    None,
                    kernels::tensor_as_f32_slice(&w.attn_norm),
                    kernels::tensor_as_f32_slice(q_norm),
                    freq_factors,
                    cached_k_f16,
                    cached_v_f16,
                    seq_len,
                    kv_len,
                    layout.num_heads,
                    layout.num_kv_heads,
                    resolve_attention_scale(metadata, architecture),
                    rope_theta,
                    pos_start,
                    norm_eps,
                    q_unit_offset,
                    &w.o_weight,
                    &w.ffn_gate_weight,
                    &w.ffn_up_weight,
                    &w.ffn_down_weight,
                    post_attn_norm,
                    kernels::tensor_as_f32_slice(ffn_norm),
                    post_ffn_norm,
                    prepared_ple.as_ref().map(|ple| ple.gate_weight),
                    prepared_ple.as_ref().map(|ple| ple.proj_weight),
                    prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
                    prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
                    prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
                    w.o_weight.cols,
                    w.ffn_gate_weight.rows,
                    metadata.hidden_dim,
                    &mut hidden_out,
                    layer_out_scale,
                    unit_offset_attn_norm,
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                    unit_offset_post_ffn_norm,
                )?;
        if let Some(device_output) = device_output {
            trace_gemma4_prefill_ple_fusion(layer_idx, "device_output_ok");
            return Ok(Some(PrefillFusedHidden {
                hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
                gemma4_ple_fused: prepared_ple.is_some(),
                gemma4_output_scale_fused: layer_out_scale.is_some(),
                device_output: Some(device_output),
            }));
        }
        trace_gemma4_prefill_ple_fusion(layer_idx, "device_output_none");
    }
    let completed =
        backend_runtime::prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_if_supported(
            &w.q_weight,
            kernels::tensor_as_f32_slice(hidden),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            freq_factors,
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            prepared_ple.as_ref().map(|ple| ple.gate_weight),
            prepared_ple.as_ref().map(|ple| ple.proj_weight),
            prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
            prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
            prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    if completed {
        Ok(Some(PrefillFusedHidden {
            hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
            gemma4_ple_fused: prepared_ple.is_some(),
            gemma4_output_scale_fused: false,
            #[cfg(feature = "cuda")]
            device_output: None,
        }))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_prefill_q4k_f16_reuse_q_hd512_dense_chain_from_device(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    device_hidden: &crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    pos_start: usize,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    if !gemma4_device_hidden_carrier_requested()
        || !gemma4_reuse_q_device_hidden_carrier_requested()
        || !gemma4_hd512_device_hidden_carrier_requested()
        || !gemma_reuse_q_dense_chain_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !gemma4_reuse_q_only
        || gemma_ple_before_layer()
        || gemma_ple_after_out_scale()
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 512
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || super::super::policy::gemma_reused_reapply_k_norm_enabled()
        || dump_bin_dir().is_some()
        || layer_trace_enabled()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let input_desc = device_hidden.output.output_desc;
    if input_desc.rows() != seq_len
        || input_desc.cols() != metadata.hidden_dim
        || input_desc.dtype() != crate::engine::cuda_runtime::ScalarType::F32
    {
        return Ok(None);
    }
    let Some(q_norm) = &w.q_norm else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim != layout.head_dim {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let q_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    let prepared_ple: Option<PreparedGemma4PrefillPleFusion<'_>> = None;
    if ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled()
    {
        return Ok(None);
    }
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    let mut hidden_out = vec![0.0f32; seq_len.saturating_mul(metadata.hidden_dim)];
    backend_runtime::prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output_if_supported(
        &w.q_weight,
        &[],
        Some((device_hidden.output.output_id, input_desc)),
        kernels::tensor_as_f32_slice(&w.attn_norm),
        kernels::tensor_as_f32_slice(q_norm),
        freq_factors,
        cached_k_f16,
        cached_v_f16,
        seq_len,
        kv_len,
        layout.num_heads,
        layout.num_kv_heads,
        resolve_attention_scale(metadata, architecture),
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        &w.o_weight,
        &w.ffn_gate_weight,
        &w.ffn_up_weight,
        &w.ffn_down_weight,
        post_attn_norm,
        kernels::tensor_as_f32_slice(ffn_norm),
        post_ffn_norm,
        prepared_ple.as_ref().map(|ple| ple.gate_weight),
        prepared_ple.as_ref().map(|ple| ple.proj_weight),
        prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
        prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
        prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
        w.o_weight.cols,
        w.ffn_gate_weight.rows,
        metadata.hidden_dim,
        &mut hidden_out,
        layer_out_scale,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    pos_start: usize,
    norm_eps: f32,
    _ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedHidden>> {
    #[cfg(feature = "cuda")]
    let ple_fusion = _ple_fusion;
    let Some(window) = active_sliding_window(metadata, architecture, layer_idx) else {
        return Ok(None);
    };
    if !gemma_reuse_q_dense_chain_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !gemma4_reuse_q_only
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 256
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || super::super::policy::gemma_reused_reapply_k_norm_enabled()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let Some(q_norm) = &w.q_norm else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    if proportional_rope || rope_dim != layout.head_dim || freq_factors.is_some() {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let q_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let prepared_ple: Option<PreparedGemma4PrefillPleFusion<'_>> = None;
    let mut hidden_out = kernels::tensor_as_f32_slice(hidden).to_vec();
    #[cfg(feature = "cuda")]
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    #[cfg(feature = "cuda")]
    let host_post_ple_pending = ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled();
    #[cfg(feature = "cuda")]
    if gemma4_device_hidden_carrier_requested()
        && !host_post_ple_pending
        && !layer_trace_enabled()
        && gemma4_reuse_q_device_hidden_carrier_requested()
    {
        if let Some(device_output) =
            backend_runtime::prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output_if_supported(
                &w.q_weight,
                kernels::tensor_as_f32_slice(hidden),
                None,
                kernels::tensor_as_f32_slice(&w.attn_norm),
                kernels::tensor_as_f32_slice(q_norm),
                cached_k_f16,
                cached_v_f16,
                seq_len,
                kv_len,
                layout.num_heads,
                layout.num_kv_heads,
                resolve_attention_scale(metadata, architecture),
                rope_theta,
                pos_start,
                norm_eps,
                q_unit_offset,
                window,
                &w.o_weight,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
                post_attn_norm,
                kernels::tensor_as_f32_slice(ffn_norm),
                post_ffn_norm,
                prepared_ple.as_ref().map(|ple| ple.gate_weight),
                prepared_ple.as_ref().map(|ple| ple.proj_weight),
                prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
                prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
                prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
                w.o_weight.cols,
                w.ffn_gate_weight.rows,
                metadata.hidden_dim,
                &mut hidden_out,
                layer_out_scale,
                unit_offset_attn_norm,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?
        {
            return Ok(Some(PrefillFusedHidden {
                hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
                gemma4_ple_fused: prepared_ple.is_some(),
                gemma4_output_scale_fused: layer_out_scale.is_some(),
                device_output: Some(device_output),
            }));
        }
    }
    let completed =
        backend_runtime::prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_if_supported(
            &w.q_weight,
            kernels::tensor_as_f32_slice(hidden),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            window,
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            prepared_ple.as_ref().map(|ple| ple.gate_weight),
            prepared_ple.as_ref().map(|ple| ple.proj_weight),
            prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
            prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
            prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    if completed {
        Ok(Some(PrefillFusedHidden {
            hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
            gemma4_ple_fused: prepared_ple.is_some(),
            gemma4_output_scale_fused: false,
            #[cfg(feature = "cuda")]
            device_output: None,
        }))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain_from_device(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    device_hidden: &crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    pos_start: usize,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    let Some(window) = active_sliding_window(metadata, architecture, layer_idx) else {
        return Ok(None);
    };
    if !gemma4_device_hidden_carrier_requested()
        || !gemma4_reuse_q_device_hidden_carrier_requested()
        || !gemma_reuse_q_dense_chain_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !gemma4_reuse_q_only
        || gemma_ple_before_layer()
        || gemma_ple_after_out_scale()
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 256
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || super::super::policy::gemma_reused_reapply_k_norm_enabled()
        || dump_bin_dir().is_some()
        || layer_trace_enabled()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let input_desc = device_hidden.output.output_desc;
    if input_desc.rows() != seq_len
        || input_desc.cols() != metadata.hidden_dim
        || input_desc.dtype() != crate::engine::cuda_runtime::ScalarType::F32
    {
        return Ok(None);
    }
    let Some(q_norm) = &w.q_norm else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    if proportional_rope || rope_dim != layout.head_dim || freq_factors.is_some() {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let q_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let prepared_ple: Option<PreparedGemma4PrefillPleFusion<'_>> = None;
    if ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled()
    {
        return Ok(None);
    }
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    let mut hidden_out = vec![0.0f32; seq_len.saturating_mul(metadata.hidden_dim)];
    backend_runtime::prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output_if_supported(
        &w.q_weight,
        &[],
        Some((device_hidden.output.output_id, input_desc)),
        kernels::tensor_as_f32_slice(&w.attn_norm),
        kernels::tensor_as_f32_slice(q_norm),
        cached_k_f16,
        cached_v_f16,
        seq_len,
        kv_len,
        layout.num_heads,
        layout.num_kv_heads,
        resolve_attention_scale(metadata, architecture),
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        window,
        &w.o_weight,
        &w.ffn_gate_weight,
        &w.ffn_up_weight,
        &w.ffn_down_weight,
        post_attn_norm,
        kernels::tensor_as_f32_slice(ffn_norm),
        post_ffn_norm,
        prepared_ple.as_ref().map(|ple| ple.gate_weight),
        prepared_ple.as_ref().map(|ple| ple.proj_weight),
        prepared_ple.as_ref().map(|ple| ple.post_norm_weight),
        prepared_ple.as_ref().map(|ple| ple.input.as_slice()),
        prepared_ple.as_ref().map(|ple| ple.dim).unwrap_or(0),
        w.o_weight.cols,
        w.ffn_gate_weight.rows,
        metadata.hidden_dim,
        &mut hidden_out,
        layer_out_scale,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_prefill_q4k_f16_qkv_hd256_window_dense_chain_from_device(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    device_hidden: &crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedQkvDenseChain>> {
    let Some(window) = active_sliding_window(metadata, architecture, layer_idx) else {
        return Ok(None);
    };
    if !gemma4_device_hidden_carrier_requested()
        || !gemma4_qkv_device_input_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || gemma4_reuse_q_only
        || pos_start != 0
        || w.v_proj_missing
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 256
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || dump_bin_dir().is_some()
        || layer_trace_enabled()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let input_desc = device_hidden.output.output_desc;
    if input_desc.rows() != seq_len
        || input_desc.cols() != metadata.hidden_dim
        || input_desc.dtype() != crate::engine::cuda_runtime::ScalarType::F32
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    if proportional_rope || rope_dim != layout.head_dim || freq_factors.is_some() {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let gemma_block = use_gemma_block_semantics(architecture);
    let unit_offset_attn_norm = gemma_block
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let qk_unit_offset = gemma_block && super::super::policy::gemma_unit_offset_norm_enabled();
    let prepared_ple = prepare_gemma4_prefill_ple_fusion(
        ple_fusion,
        metadata,
        architecture,
        w,
        layer_idx,
        seq_len,
    );
    let device_ple_allowed =
        gemma4_prepared_ple_device_fusion_allowed(layer_idx, prepared_ple.as_ref());
    let partial_ple_host_carrier =
        gemma4_partial_ple_host_carrier_active(prepared_ple.as_ref(), device_ple_allowed);
    if !device_ple_allowed && !partial_ple_host_carrier {
        return Ok(None);
    }
    if ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled()
    {
        return Ok(None);
    }
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    let fused_ple = if device_ple_allowed {
        prepared_ple.as_ref()
    } else {
        None
    };
    let fused_layer_out_scale = if device_ple_allowed || prepared_ple.is_none() {
        layer_out_scale
    } else {
        None
    };
    let mut hidden_out = vec![0.0f32; seq_len.saturating_mul(metadata.hidden_dim)];
    let Some((k_bits, v_bits, device_output)) =
        backend_runtime::prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            &[],
            Some((device_hidden.output.output_id, input_desc)),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            kernels::tensor_as_f32_slice(k_norm),
            seq_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            qk_unit_offset,
            qk_unit_offset,
            gemma_block && super::super::policy::gemma_v_norm_enabled(),
            window,
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            fused_ple.map(|ple| ple.gate_weight),
            fused_ple.map(|ple| ple.proj_weight),
            fused_ple.map(|ple| ple.post_norm_weight),
            fused_ple.map(|ple| ple.input.as_slice()),
            fused_ple.map(|ple| ple.dim).unwrap_or(0),
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            fused_layer_out_scale,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    else {
        return Ok(None);
    };
    Ok(Some(PrefillFusedQkvDenseChain {
        hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
        k_bits,
        v_bits,
        gemma4_ple_fused: fused_ple.is_some(),
        gemma4_output_scale_fused: fused_layer_out_scale.is_some(),
        device_output: Some(device_output),
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_q4k_f16_qkv_hd256_window_dense_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    _ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedQkvDenseChain>> {
    #[cfg(feature = "cuda")]
    let ple_fusion = _ple_fusion;
    let Some(window) = active_sliding_window(metadata, architecture, layer_idx) else {
        return Ok(None);
    };
    if !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || gemma4_reuse_q_only
        || pos_start != 0
        || w.v_proj_missing
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 256
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    if proportional_rope || rope_dim != layout.head_dim || freq_factors.is_some() {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let gemma_block = use_gemma_block_semantics(architecture);
    let unit_offset_attn_norm = gemma_block
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let qk_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    #[cfg(feature = "cuda")]
    let prepared_ple = prepare_gemma4_prefill_ple_fusion(
        ple_fusion,
        metadata,
        architecture,
        w,
        layer_idx,
        seq_len,
    );
    let mut hidden_out = kernels::tensor_as_f32_slice(hidden).to_vec();
    #[cfg(feature = "cuda")]
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    #[cfg(feature = "cuda")]
    let host_post_ple_pending = ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled();
    #[cfg(feature = "cuda")]
    let device_ple_allowed =
        gemma4_prepared_ple_device_fusion_allowed(layer_idx, prepared_ple.as_ref());
    #[cfg(feature = "cuda")]
    let partial_ple_host_carrier =
        gemma4_partial_ple_host_carrier_active(prepared_ple.as_ref(), device_ple_allowed);
    #[cfg(feature = "cuda")]
    if gemma4_device_hidden_carrier_requested()
        && (device_ple_allowed || partial_ple_host_carrier)
        && !host_post_ple_pending
        && !layer_trace_enabled()
    {
        let fused_ple = if device_ple_allowed {
            prepared_ple.as_ref()
        } else {
            None
        };
        let fused_layer_out_scale = if device_ple_allowed || prepared_ple.is_none() {
            layer_out_scale
        } else {
            None
        };
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "qkv_hd256_device_output_try ple_prepared={} out_scale={} partial_ple_host_carrier={}",
                prepared_ple.is_some(),
                fused_layer_out_scale.is_some(),
                partial_ple_host_carrier
            ),
        );
        if let Some((k_bits, v_bits, device_output)) =
            backend_runtime::prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output_if_supported(
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                kernels::tensor_as_f32_slice(hidden),
                None,
                kernels::tensor_as_f32_slice(&w.attn_norm),
                kernels::tensor_as_f32_slice(q_norm),
                kernels::tensor_as_f32_slice(k_norm),
                seq_len,
                layout.num_heads,
                layout.num_kv_heads,
                resolve_attention_scale(metadata, architecture),
                rope_theta,
                pos_start,
                norm_eps,
                qk_unit_offset,
                qk_unit_offset,
                use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
                window,
                &w.o_weight,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
                post_attn_norm,
                kernels::tensor_as_f32_slice(ffn_norm),
                post_ffn_norm,
                fused_ple.map(|ple| ple.gate_weight),
                fused_ple.map(|ple| ple.proj_weight),
                fused_ple.map(|ple| ple.post_norm_weight),
                fused_ple.map(|ple| ple.input.as_slice()),
                fused_ple.map(|ple| ple.dim).unwrap_or(0),
                w.o_weight.cols,
                w.ffn_gate_weight.rows,
                metadata.hidden_dim,
                &mut hidden_out,
                fused_layer_out_scale,
                unit_offset_attn_norm,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?
        {
            trace_gemma4_prefill_ple_fusion(layer_idx, "qkv_hd256_device_output_ok");
            return Ok(Some(PrefillFusedQkvDenseChain {
                hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
                k_bits,
                v_bits,
                gemma4_ple_fused: fused_ple.is_some(),
                gemma4_output_scale_fused: fused_layer_out_scale.is_some(),
                device_output: Some(device_output),
            }));
        }
        trace_gemma4_prefill_ple_fusion(layer_idx, "qkv_hd256_device_output_none");
    }
    let Some((k_bits, v_bits)) =
        backend_runtime::prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            kernels::tensor_as_f32_slice(hidden),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            kernels::tensor_as_f32_slice(k_norm),
            seq_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            qk_unit_offset,
            qk_unit_offset,
            use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
            window,
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    else {
        return Ok(None);
    };

    Ok(Some(PrefillFusedQkvDenseChain {
        hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
        k_bits,
        v_bits,
        gemma4_ple_fused: false,
        gemma4_output_scale_fused: false,
        #[cfg(feature = "cuda")]
        device_output: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_device_input_chain_defaults_on_for_chat_semantics() {
        assert!(parse_env_bool_or(None, GEMMA4_QKV_DEVICE_INPUT_DEFAULT));
        assert!(parse_env_bool_or(
            None,
            GEMMA4_HD512_DEVICE_HIDDEN_CARRIER_DEFAULT
        ));
        assert!(parse_env_bool_or(
            None,
            GEMMA4_REUSE_Q_DEVICE_HIDDEN_CARRIER_DEFAULT
        ));
    }

    #[test]
    fn gemma4_device_input_chain_honors_opt_out_values() {
        for value in ["0", "false", "off", "no"] {
            assert!(!parse_env_bool_or(Some(value), true));
        }
    }

    #[test]
    fn gemma4_device_input_chain_honors_opt_in_values() {
        for value in ["1", "true", "on", "yes"] {
            assert!(parse_env_bool_or(Some(value), false));
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_prefill_q4k_f16_qkv_hd512_dense_chain_from_device(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    device_hidden: &crate::engine::prefill::hidden_carrier::DevicePrefillHidden,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedQkvDenseChain>> {
    if !gemma4_device_hidden_carrier_requested()
        || !gemma4_hd512_device_hidden_carrier_requested()
        || !gemma4_qkv_device_input_requested()
        || !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !policy::q4k_fused_prefill_attention_requested()
        || gemma4_reuse_q_only
        || pos_start != 0
        || w.v_proj_missing
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 512
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || dump_bin_dir().is_some()
        || layer_trace_enabled()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let input_desc = device_hidden.output.output_desc;
    if input_desc.rows() != seq_len
        || input_desc.cols() != metadata.hidden_dim
        || input_desc.dtype() != crate::engine::cuda_runtime::ScalarType::F32
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim != layout.head_dim {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let qk_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    let prepared_ple = prepare_gemma4_prefill_ple_fusion(
        ple_fusion,
        metadata,
        architecture,
        w,
        layer_idx,
        seq_len,
    );
    let device_ple_allowed =
        gemma4_prepared_ple_device_fusion_allowed(layer_idx, prepared_ple.as_ref());
    let partial_ple_host_carrier =
        gemma4_partial_ple_host_carrier_active(prepared_ple.as_ref(), device_ple_allowed);
    if !device_ple_allowed && !partial_ple_host_carrier {
        return Ok(None);
    }
    let host_post_ple_pending = ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled();
    if host_post_ple_pending {
        return Ok(None);
    }
    let mut hidden_out = vec![0.0f32; seq_len * metadata.hidden_dim];
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    let fused_ple = if device_ple_allowed {
        prepared_ple.as_ref()
    } else {
        None
    };
    let fused_layer_out_scale = if device_ple_allowed || prepared_ple.is_none() {
        layer_out_scale
    } else {
        None
    };
    let Some((k_bits, v_bits, device_output)) =
        backend_runtime::prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_device_output_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            &[],
            Some((device_hidden.output.output_id, input_desc)),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            kernels::tensor_as_f32_slice(k_norm),
            freq_factors,
            seq_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            qk_unit_offset,
            qk_unit_offset,
            use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            fused_ple.map(|ple| ple.gate_weight),
            fused_ple.map(|ple| ple.proj_weight),
            fused_ple.map(|ple| ple.post_norm_weight),
            fused_ple.map(|ple| ple.input.as_slice()),
            fused_ple.map(|ple| ple.dim).unwrap_or(0),
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            fused_layer_out_scale,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    else {
        return Ok(None);
    };

    Ok(Some(PrefillFusedQkvDenseChain {
        hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
        k_bits,
        v_bits,
        gemma4_ple_fused: fused_ple.is_some(),
        gemma4_output_scale_fused: fused_layer_out_scale.is_some(),
        device_output: Some(device_output),
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_q4k_f16_qkv_hd512_dense_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma4_reuse_q_only: bool,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    norm_eps: f32,
    _ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<Option<PrefillFusedQkvDenseChain>> {
    #[cfg(feature = "cuda")]
    let ple_fusion = _ple_fusion;
    if !super::dense_chain::prefill_dense_chain_enabled()
        || !matches!(architecture, ModelArchitecture::Gemma4)
        || !use_gemma_block_semantics(architecture)
        || !policy::q4k_fused_prefill_attention_requested()
        || gemma4_reuse_q_only
        || pos_start != 0
        || w.v_proj_missing
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || layout.has_gated_attn
        || layout.head_dim != 512
        || active_sliding_window(metadata, architecture, layer_idx).is_some()
        || resolve_attention_softcap(architecture).is_some()
        || !super::super::policy::gemma_neox_rope_enabled()
        || super::super::policy::gemma_qk_norm_disabled()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || w.q_bias.is_some()
        || w.k_bias.is_some()
        || w.v_bias.is_some()
        || gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, layout.head_dim)
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(None);
    }
    let (Some(q_norm), Some(k_norm)) = (&w.q_norm, &w.k_norm) else {
        return Ok(None);
    };
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, layout.head_dim);
    if proportional_rope || rope_dim != layout.head_dim {
        return Ok(None);
    }

    let post_attn_norm = w.post_attn_norm.as_ref().and_then(|post_attn_norm| {
        if gemma_skip_post_attn_enabled(layer_idx)
            || gemma_effective_skip_post_attn_prefill_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            )
        {
            None
        } else {
            Some(kernels::tensor_as_f32_slice(post_attn_norm))
        }
    });
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let post_ffn_norm = w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
    let unit_offset_post_attn_norm = gemma_effective_unit_offset_post_attn_enabled(
        architecture,
        gemma_runtime_flavor,
        layer_idx,
    ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx);
    let unit_offset_ffn_norm = policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx);
    let unit_offset_post_ffn_norm = policy::gemma_unit_offset_ffn_post_norm_enabled();
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (policy::gemma_unit_offset_attn_norm_enabled(layer_idx)
            || policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || policy::gemma_unit_offset_norm_enabled()
            || policy::gemma_unit_offset_main_norm_enabled());
    let qk_unit_offset = use_gemma_block_semantics(architecture)
        && super::super::policy::gemma_unit_offset_norm_enabled();
    let freq_factors = gemma_rope_freq_factors(
        rope_freqs,
        metadata,
        architecture,
        layer_idx,
        layout.head_dim,
    );
    #[cfg(feature = "cuda")]
    let prepared_ple = prepare_gemma4_prefill_ple_fusion(
        ple_fusion,
        metadata,
        architecture,
        w,
        layer_idx,
        seq_len,
    );
    let mut hidden_out = vec![0.0f32; seq_len * metadata.hidden_dim];
    #[cfg(feature = "cuda")]
    let layer_out_scale = active_layer_output_scale(w.out_scale.as_ref(), layer_idx);
    #[cfg(feature = "cuda")]
    let host_post_ple_pending = ple_fusion.is_some()
        && prepared_ple.is_none()
        && !gemma_ple_global_only(metadata, w)
        && !super::super::policy::gemma_ple_global_only_enabled();
    #[cfg(feature = "cuda")]
    let device_ple_allowed =
        gemma4_prepared_ple_device_fusion_allowed(layer_idx, prepared_ple.as_ref());
    #[cfg(feature = "cuda")]
    let partial_ple_host_carrier =
        gemma4_partial_ple_host_carrier_active(prepared_ple.as_ref(), device_ple_allowed);
    #[cfg(feature = "cuda")]
    if gemma4_device_hidden_carrier_requested()
        && gemma4_hd512_device_hidden_carrier_requested()
        && (device_ple_allowed || partial_ple_host_carrier)
        && !host_post_ple_pending
        && !layer_trace_enabled()
    {
        let fused_ple = if device_ple_allowed {
            prepared_ple.as_ref()
        } else {
            None
        };
        let fused_layer_out_scale = if device_ple_allowed || prepared_ple.is_none() {
            layer_out_scale
        } else {
            None
        };
        trace_gemma4_prefill_ple_fusion(
            layer_idx,
            format!(
                "qkv_hd512_device_output_try ple_prepared={} out_scale={} partial_ple_host_carrier={}",
                prepared_ple.is_some(),
                fused_layer_out_scale.is_some(),
                partial_ple_host_carrier
            ),
        );
        if let Some((k_bits, v_bits, device_output)) =
            backend_runtime::prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_device_output_if_supported(
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                kernels::tensor_as_f32_slice(hidden),
                None,
                kernels::tensor_as_f32_slice(&w.attn_norm),
                kernels::tensor_as_f32_slice(q_norm),
                kernels::tensor_as_f32_slice(k_norm),
                freq_factors,
                seq_len,
                layout.num_heads,
                layout.num_kv_heads,
                resolve_attention_scale(metadata, architecture),
                rope_theta,
                pos_start,
                norm_eps,
                qk_unit_offset,
                qk_unit_offset,
                use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
                &w.o_weight,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
                post_attn_norm,
                kernels::tensor_as_f32_slice(ffn_norm),
                post_ffn_norm,
                fused_ple.map(|ple| ple.gate_weight),
                fused_ple.map(|ple| ple.proj_weight),
                fused_ple.map(|ple| ple.post_norm_weight),
                fused_ple.map(|ple| ple.input.as_slice()),
                fused_ple.map(|ple| ple.dim).unwrap_or(0),
                w.o_weight.cols,
                w.ffn_gate_weight.rows,
                metadata.hidden_dim,
                &mut hidden_out,
                fused_layer_out_scale,
                unit_offset_attn_norm,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?
        {
            trace_gemma4_prefill_ple_fusion(layer_idx, "qkv_hd512_device_output_ok");
            return Ok(Some(PrefillFusedQkvDenseChain {
                hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
                k_bits,
                v_bits,
                gemma4_ple_fused: fused_ple.is_some(),
                gemma4_output_scale_fused: fused_layer_out_scale.is_some(),
                device_output: Some(device_output),
            }));
        }
        trace_gemma4_prefill_ple_fusion(layer_idx, "qkv_hd512_device_output_none");
    }
    let Some((k_bits, v_bits)) =
        backend_runtime::prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            kernels::tensor_as_f32_slice(hidden),
            kernels::tensor_as_f32_slice(&w.attn_norm),
            kernels::tensor_as_f32_slice(q_norm),
            kernels::tensor_as_f32_slice(k_norm),
            freq_factors,
            seq_len,
            layout.num_heads,
            layout.num_kv_heads,
            resolve_attention_scale(metadata, architecture),
            rope_theta,
            pos_start,
            norm_eps,
            qk_unit_offset,
            qk_unit_offset,
            use_gemma_block_semantics(architecture) && super::super::policy::gemma_v_norm_enabled(),
            &w.o_weight,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            post_attn_norm,
            kernels::tensor_as_f32_slice(ffn_norm),
            post_ffn_norm,
            w.o_weight.cols,
            w.ffn_gate_weight.rows,
            metadata.hidden_dim,
            &mut hidden_out,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    else {
        return Ok(None);
    };

    Ok(Some(PrefillFusedQkvDenseChain {
        hidden: Tensor::from_vec(hidden_out, &[seq_len, metadata.hidden_dim]),
        k_bits,
        v_bits,
        gemma4_ple_fused: false,
        gemma4_output_scale_fused: false,
        #[cfg(feature = "cuda")]
        device_output: None,
    }))
}
