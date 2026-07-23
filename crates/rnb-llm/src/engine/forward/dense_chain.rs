//! Prefill dense attention-output + FFN CUDA chain.

use super::super::*;

pub(super) fn prefill_dense_chain_enabled() -> bool {
    crate::engine::policy::env_string("RNB_CUDA_PREFILL_DENSE_CHAIN")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_attention_output_ffn_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    attn_out: &Tensor,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<Tensor>> {
    if !prefill_dense_chain_enabled()
        || !use_gemma_block_semantics(architecture)
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
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
    let mut hidden_out = kernels::tensor_as_f32_slice(hidden).to_vec();
    let attn_out_data = kernels::tensor_as_f32_slice(attn_out);
    if backend_runtime::dense_q4k_attention_output_gelu_ffn_batch_norm_residual_if_supported(
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
        seq_len,
        &mut hidden_out,
        attn_out_data,
        norm_eps,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )? {
        Ok(Some(Tensor::from_vec(
            hidden_out,
            &[seq_len, metadata.hidden_dim],
        )))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_prefill_f16kv_attention_output_ffn_chain(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    q: &Tensor,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    layout: AttentionLayout,
    layer_idx: usize,
    seq_len: usize,
    kv_len: usize,
    norm_eps: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> crate::error::Result<Option<Tensor>> {
    if !prefill_dense_chain_enabled()
        || !use_gemma_block_semantics(architecture)
        || w.moe.is_some()
        || w.shared_expert_moe.is_some()
        || dump_bin_dir().is_some()
        || attn_trace_enabled()
        || targeted_attn_trace_enabled(layer_idx)
        || layout.has_gated_attn
        || gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, layout.head_dim)
        || matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
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
    let mut hidden_out = kernels::tensor_as_f32_slice(hidden).to_vec();
    let completed = if sliding_window.is_some() {
        backend_runtime::prefill_attention_f16kv_window_dense_chain_if_supported(
            kernels::tensor_as_f32_slice(q),
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            layout.num_heads,
            layout.num_kv_heads,
            layout.head_dim,
            resolve_attention_scale(metadata, architecture),
            sliding_window,
            has_softcap,
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
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    } else {
        backend_runtime::prefill_attention_f16kv_dense_chain_if_supported(
            kernels::tensor_as_f32_slice(q),
            cached_k_f16,
            cached_v_f16,
            seq_len,
            kv_len,
            layout.num_heads,
            layout.num_kv_heads,
            layout.head_dim,
            resolve_attention_scale(metadata, architecture),
            false,
            has_softcap,
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
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?
    };
    if completed {
        Ok(Some(Tensor::from_vec(
            hidden_out,
            &[seq_len, metadata.hidden_dim],
        )))
    } else {
        Ok(None)
    }
}
