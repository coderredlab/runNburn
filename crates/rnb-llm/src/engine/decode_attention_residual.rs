//! Decode-time attention post-projection normalization and residual handling.

use super::*;

pub(super) fn apply_decode_attention_residual<F>(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    w: &AttentionLayerWeights,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
    source_hidden: Option<&[f32]>,
    prev_layer_hidden: Option<&[f32]>,
    verbose: bool,
    output_weight_start: std::time::Instant,
    residual_block_start: std::time::Instant,
    mut profile: F,
) where
    F: FnMut(&str, std::time::Instant),
{
    if use_gemma_block_semantics(architecture) {
        if let Some(post_attn_norm) = &w.post_attn_norm {
            if !(gemma_skip_post_attn_enabled(layer_idx)
                || gemma_skip_post_attn_decode_enabled(layer_idx))
            {
                let force_plain = gemma_post_attn_decode_plain_enabled(layer_idx);
                if !force_plain
                    && gemma_effective_unit_offset_post_attn_enabled(
                        architecture,
                        gemma_runtime_flavor,
                        layer_idx,
                    )
                {
                    apply_model_norm_unit_offset_into(
                        &scratch.proj_buf[..hidden_dim],
                        kernels::tensor_as_f32_slice(post_attn_norm),
                        norm_eps,
                        &mut scratch.norm_buf2[..hidden_dim],
                    );
                } else {
                    let post_attn_norm_data = kernels::tensor_as_f32_slice(post_attn_norm);
                    apply_model_norm_into(
                        &scratch.proj_buf[..hidden_dim],
                        post_attn_norm_data,
                        norm_eps,
                        &mut scratch.norm_buf2[..hidden_dim],
                        architecture,
                    );
                }
                scratch.proj_buf[..hidden_dim].copy_from_slice(&scratch.norm_buf2[..hidden_dim]);
            }
            if gemma_post_attn_blend_source_decode_enabled(layer_idx) {
                let blend_src = source_hidden.or(prev_layer_hidden);
                if let Some(src) = blend_src {
                    let alpha = gemma_post_attn_blend_source_decode_alpha();
                    let beta = 1.0 - alpha;
                    let n = hidden_dim.min(src.len());
                    axpby_f32_inplace(&mut scratch.proj_buf[..n], &src[..n], alpha, beta);
                }
            }
        }
    }
    if verbose {
        profile("o_weight", output_weight_start);
    }

    if gemma_pre_residual_blend_source_decode_enabled(layer_idx) {
        let blend_src = source_hidden.or(prev_layer_hidden);
        if let Some(src) = blend_src {
            let alpha = gemma_pre_residual_blend_source_decode_alpha();
            let beta = 1.0 - alpha;
            let n = hidden_dim.min(src.len());
            axpby_f32_inplace(&mut scratch.hidden[..n], &src[..n], alpha, beta);
        }
    }

    let residual_start = std::time::Instant::now();
    add_f32_inplace(
        &mut scratch.hidden[..hidden_dim],
        &scratch.proj_buf[..hidden_dim],
    );

    if verbose {
        profile("attn_residual", residual_start);
    } else {
        profile("o_proj+residual", residual_block_start);
    }
}
