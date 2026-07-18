//! Decode-time RoPE application for attention Q/K vectors.

use super::*;

pub(in crate::engine) fn apply_decode_rope(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    pos: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    gemma4_reuse_q_only: bool,
    q_slice: &mut [f32],
    k_slice: &mut [f32],
) {
    if matches!(architecture, ModelArchitecture::NemotronHMoE) {
        return;
    }

    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, head_dim);
    let freq_factors =
        gemma_rope_freq_factors(rope_freqs, metadata, architecture, layer_idx, head_dim);
    let qwen_mrope_dim = qwen_text_mrope_dim(metadata, architecture, rope_dim, head_dim);
    let trace_k_pre_rope = should_capture_decode_k_pre_rope(layer_idx, attn_trace_enabled());
    let k_pre_rope = trace_k_pre_rope.then(|| k_slice.to_vec());
    let use_neox_rope = matches!(architecture, ModelArchitecture::Hy3)
        || (matches!(architecture, ModelArchitecture::Gemma4)
            && super::policy::gemma_neox_rope_enabled());
    if let Some(mrope_dim) = qwen_mrope_dim {
        // M2: text M-RoPE device 커널. RNB_METAL_ROPE=1 일 때만 device.
        let metal_done = backend_runtime::metal_rope_mrope_into_if_supported(
            q_slice,
            k_slice,
            head_dim,
            q_dim,
            kv_dim,
            mrope_dim,
            rope_theta,
            pos,
            !gemma4_reuse_q_only,
        )
        .unwrap_or(false);
        if !metal_done {
            kernels::rope::rope_mrope_text_inplace(
                q_slice, pos, head_dim, q_dim, mrope_dim, rope_theta,
            );
            if !gemma4_reuse_q_only {
                kernels::rope::rope_mrope_text_inplace(
                    k_slice, pos, head_dim, kv_dim, mrope_dim, rope_theta,
                );
            }
        }
    } else if use_neox_rope {
        if let Some(freq_factors) = freq_factors {
            kernels::rope::rope_neox_inplace_with_factors(
                q_slice,
                pos,
                head_dim,
                q_dim,
                rope_theta,
                freq_factors,
            );
            if !gemma4_reuse_q_only {
                kernels::rope::rope_neox_inplace_with_factors(
                    k_slice,
                    pos,
                    head_dim,
                    kv_dim,
                    rope_theta,
                    freq_factors,
                );
            }
        } else {
            kernels::rope::rope_neox_inplace(q_slice, pos, head_dim, q_dim, rope_theta);
            if !gemma4_reuse_q_only {
                kernels::rope::rope_neox_inplace(k_slice, pos, head_dim, kv_dim, rope_theta);
            }
        }
    } else if let Some(freq_factors) = freq_factors {
        kernels::rope::rope_inplace_with_factors(
            q_slice,
            pos,
            head_dim,
            q_dim,
            rope_theta,
            freq_factors,
        );
        kernels::rope::rope_inplace_with_factors(
            k_slice,
            pos,
            head_dim,
            kv_dim,
            rope_theta,
            freq_factors,
        );
    } else if rope_dim > 0 && rope_dim < head_dim {
        if proportional_rope {
            kernels::rope::rope_partial_proportional_inplace(
                q_slice, pos, head_dim, q_dim, rope_dim, rope_theta,
            );
            kernels::rope::rope_partial_proportional_inplace(
                k_slice, pos, head_dim, kv_dim, rope_dim, rope_theta,
            );
        } else {
            kernels::rope::rope_partial_inplace(
                q_slice, pos, head_dim, q_dim, rope_dim, rope_theta,
            );
            kernels::rope::rope_partial_inplace(
                k_slice, pos, head_dim, kv_dim, rope_dim, rope_theta,
            );
        }
    } else {
        kernels::rope::rope_inplace(q_slice, pos, head_dim, q_dim, rope_theta);
        kernels::rope::rope_inplace(k_slice, pos, head_dim, kv_dim, rope_theta);
    }
    if let Some(k_pre_rope) = k_pre_rope.as_deref() {
        emit_vec_trace("decode", layer_idx, "q", q_slice);
        emit_vec_trace("decode", layer_idx, "k_pre", k_pre_rope);
        emit_vec_trace("decode", layer_idx, "k", k_slice);
    }
}

fn should_capture_decode_k_pre_rope(layer_idx: usize, trace_enabled: bool) -> bool {
    layer_idx == 0 && trace_enabled
}

#[cfg(test)]
mod tests {
    #[test]
    fn captures_decode_k_pre_rope_only_for_layer_zero_trace() {
        assert!(super::should_capture_decode_k_pre_rope(0, true));
        assert!(!super::should_capture_decode_k_pre_rope(1, true));
        assert!(!super::should_capture_decode_k_pre_rope(0, false));
    }
}
