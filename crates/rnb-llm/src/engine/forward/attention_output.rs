//! Prefill attention output projection and residual helpers.

use super::super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn apply_prefill_attention_output(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    attn_out: &Tensor,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    // pm37: o_proj seam. K=q_dim(attn_out 입력), N=hidden. residual(아래 :add)은 seam 밖 CPU.
    let attn_proj = if let Some(t) =
        super::projection::atn_proj_metal("o_proj", layer_idx, &w.o_weight, attn_out, seq_len)?
    {
        t
    } else {
        w.o_weight.gemv(attn_out)?
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "o_proj",
            kernels::tensor_as_f32_slice(&attn_proj),
        );
    }
    if targeted_attn_trace_enabled(layer_idx) {
        let proj_data = kernels::tensor_as_f32_slice(&attn_proj);
        let proj_last =
            &proj_data[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
        emit_vec_trace("prefill-l34", layer_idx, "attn_proj", proj_last);
    }
    let attn_proj = if use_gemma_block_semantics(architecture) {
        if let Some(post_attn_norm) = &w.post_attn_norm {
            if gemma_skip_post_attn_enabled(layer_idx)
                || gemma_effective_skip_post_attn_prefill_enabled(
                    architecture,
                    gemma_runtime_flavor,
                    layer_idx,
                )
            {
                attn_proj
            } else if gemma_effective_unit_offset_post_attn_enabled(
                architecture,
                gemma_runtime_flavor,
                layer_idx,
            ) || gemma_unit_offset_post_attn_prefill_enabled(layer_idx)
            {
                apply_model_norm_unit_offset(&attn_proj, post_attn_norm, norm_eps).map_err(fwd)?
            } else {
                apply_model_norm(&attn_proj, post_attn_norm, norm_eps, architecture).map_err(fwd)?
            }
        } else {
            attn_proj
        }
    } else {
        attn_proj
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "post_attn_norm",
            kernels::tensor_as_f32_slice(&attn_proj),
        );
    }
    if targeted_attn_trace_enabled(layer_idx) {
        let normed = kernels::tensor_as_f32_slice(&attn_proj);
        let normed_last =
            &normed[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
        emit_vec_trace("prefill-l34", layer_idx, "post_attn_norm", normed_last);
    }
    let hidden = kernels::elementwise::add(&hidden, &attn_proj).map_err(fwd)?;
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "residual_1",
            kernels::tensor_as_f32_slice(&hidden),
        );
    }

    Ok(hidden)
}
