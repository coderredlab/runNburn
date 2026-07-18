//! Decode-time QKV projection post-processing.

use super::*;

pub(super) fn apply_decode_attention_qkv_postprocess(
    scratch: &mut ScratchBuffers,
    w: &AttentionLayerWeights,
    architecture: ModelArchitecture,
    layout: AttentionLayout,
    q_out_dim: usize,
    norm_eps: f32,
    gemma4_reuse_q_only: bool,
) {
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;
    let head_dim = layout.head_dim;

    if let Some(bias) = &w.q_bias {
        let bias_data = kernels::tensor_as_f32_slice(bias);
        kernels::elementwise::add_inplace(&mut scratch.q_buf[..q_out_dim], bias_data);
    }
    if !gemma4_reuse_q_only {
        if let Some(bias) = &w.k_bias {
            let bias_data = kernels::tensor_as_f32_slice(bias);
            kernels::elementwise::add_inplace(&mut scratch.k_buf[..kv_dim], bias_data);
        }
        if let Some(bias) = &w.v_bias {
            let bias_data = kernels::tensor_as_f32_slice(bias);
            kernels::elementwise::add_inplace(&mut scratch.v_buf[..kv_dim], bias_data);
        }
    }

    if layout.has_gated_attn {
        for h in 0..layout.num_heads {
            let src_off = h * head_dim * 2;
            let dst_off = h * head_dim;
            scratch.q_split[dst_off..dst_off + head_dim]
                .copy_from_slice(&scratch.q_buf[src_off..src_off + head_dim]);
            scratch.gate_split[dst_off..dst_off + head_dim]
                .copy_from_slice(&scratch.q_buf[src_off + head_dim..src_off + head_dim * 2]);
        }
        if let Some(q_norm) = &w.q_norm {
            let q_norm_data = kernels::tensor_as_f32_slice(q_norm);
            for h in 0..layout.num_heads {
                let off = h * head_dim;
                apply_model_qk_norm_into(
                    &scratch.q_split[off..off + head_dim],
                    q_norm_data,
                    norm_eps,
                    &mut scratch.norm_buf2[off..off + head_dim],
                    architecture,
                );
            }
            scratch.q_split[..q_dim].copy_from_slice(&scratch.norm_buf2[..q_dim]);
        }
    } else if let Some(q_norm) = &w.q_norm {
        let q_norm_data = kernels::tensor_as_f32_slice(q_norm);
        for h in 0..layout.num_heads {
            let off = h * head_dim;
            apply_model_qk_norm_into(
                &scratch.q_buf[off..off + head_dim],
                q_norm_data,
                norm_eps,
                &mut scratch.norm_buf2[off..off + head_dim],
                architecture,
            );
        }
        scratch.q_buf[..q_dim].copy_from_slice(&scratch.norm_buf2[..q_dim]);
    }

    if !gemma4_reuse_q_only {
        if let Some(k_norm) = &w.k_norm {
            let k_norm_data = kernels::tensor_as_f32_slice(k_norm);
            for h in 0..layout.num_kv_heads {
                let off = h * head_dim;
                apply_model_qk_norm_into(
                    &scratch.k_buf[off..off + head_dim],
                    k_norm_data,
                    norm_eps,
                    &mut scratch.norm_buf2[off..off + head_dim],
                    architecture,
                );
            }
            scratch.k_buf[..kv_dim].copy_from_slice(&scratch.norm_buf2[..kv_dim]);
        }
    }

    if !gemma4_reuse_q_only
        && gemma4_should_apply_attn_rotation(architecture, w.k_weight.ggml_type, head_dim)
    {
        let nrot = gemma4_k_rot_size(head_dim);
        let q_slice = if layout.has_gated_attn {
            &mut scratch.q_split[..q_dim]
        } else {
            &mut scratch.q_buf[..q_dim]
        };
        gemma4_apply_attn_rot_inplace(q_slice, head_dim, q_dim, nrot);
        gemma4_apply_attn_rot_inplace(&mut scratch.k_buf[..kv_dim], head_dim, kv_dim, nrot);
    } else if gemma4_reuse_q_only
        && gemma4_should_apply_attn_rotation(architecture, w.k_weight.ggml_type, head_dim)
    {
        let nrot = gemma4_k_rot_size(head_dim);
        let q_slice = if layout.has_gated_attn {
            &mut scratch.q_split[..q_dim]
        } else {
            &mut scratch.q_buf[..q_dim]
        };
        gemma4_apply_attn_rot_inplace(q_slice, head_dim, q_dim, nrot);
    }

    if !gemma4_reuse_q_only && use_gemma_block_semantics(architecture) {
        for h in 0..layout.num_kv_heads {
            let off = h * head_dim;
            apply_rms_norm_no_scale_into(
                &scratch.v_buf[off..off + head_dim],
                norm_eps,
                &mut scratch.norm_buf2[off..off + head_dim],
            );
        }
        scratch.v_buf[..kv_dim].copy_from_slice(&scratch.norm_buf2[..kv_dim]);
    }
    if !gemma4_reuse_q_only
        && gemma4_should_apply_attn_rotation(architecture, w.v_weight.ggml_type, head_dim)
    {
        gemma4_apply_attn_rot_inplace(&mut scratch.v_buf[..kv_dim], head_dim, kv_dim, 64);
    }
}
