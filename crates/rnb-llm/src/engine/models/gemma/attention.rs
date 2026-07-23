use super::*;

pub(in crate::engine) fn gemma4_apply_attn_rot_inplace(
    data: &mut [f32],
    head_dim: usize,
    dim: usize,
    nrot: usize,
) {
    if nrot == 0 || head_dim == 0 || dim == 0 {
        return;
    }
    assert!(head_dim % nrot == 0, "head_dim must be divisible by nrot");
    let seq_len = data.len() / dim;
    for t in 0..seq_len {
        let token = &mut data[t * dim..(t + 1) * dim];
        for head in token.chunks_mut(head_dim) {
            for block in head.chunks_mut(nrot) {
                hadamard_inplace(block);
            }
        }
    }
}

pub(in crate::engine) fn gemma4_apply_attn_rot_tensor(
    tensor: &Tensor,
    head_dim: usize,
    dim: usize,
    nrot: usize,
) -> Tensor {
    let mut data = kernels::tensor_as_f32_slice(tensor).to_vec();
    gemma4_apply_attn_rot_inplace(&mut data, head_dim, dim, nrot);
    Tensor::from_vec(data, tensor.shape())
}

pub(in crate::engine) fn gemma4_k_rot_size(head_dim: usize) -> usize {
    let mut nrot = 64usize;
    while head_dim.is_multiple_of(nrot) {
        nrot *= 2;
    }
    (nrot / 2).max(64)
}

pub(in crate::engine) fn gemma4_should_apply_attn_rotation(
    architecture: ModelArchitecture,
    weight_type: GGMLType,
    head_dim: usize,
) -> bool {
    matches!(architecture, ModelArchitecture::Gemma4)
        && head_dim % 64 == 0
        && env_flag("RNB_ENABLE_GEMMA_ATTN_ROT")
        && !matches!(weight_type, GGMLType::F32 | GGMLType::F16 | GGMLType::BF16)
}

pub(in crate::engine) fn gemma_attn_rot_mode() -> Option<String> {
    env_string("RNB_GEMMA_ATTN_ROT_MODE")
}

pub(in crate::engine) fn gemma4_should_apply_k_rotation(
    architecture: ModelArchitecture,
    weight_type: GGMLType,
    head_dim: usize,
) -> bool {
    if let Some(mode) = gemma_attn_rot_mode() {
        return matches!(mode.as_str(), "all" | "qk")
            && !matches!(weight_type, GGMLType::F32 | GGMLType::F16 | GGMLType::BF16)
            && matches!(architecture, ModelArchitecture::Gemma4)
            && head_dim % 64 == 0;
    }
    gemma4_should_apply_attn_rotation(architecture, weight_type, head_dim)
}

pub(in crate::engine) fn gemma4_should_apply_v_rotation(
    architecture: ModelArchitecture,
    weight_type: GGMLType,
    head_dim: usize,
) -> bool {
    if let Some(mode) = gemma_attn_rot_mode() {
        return matches!(mode.as_str(), "all" | "v")
            && !matches!(weight_type, GGMLType::F32 | GGMLType::F16 | GGMLType::BF16)
            && matches!(architecture, ModelArchitecture::Gemma4)
            && head_dim % 64 == 0;
    }
    gemma4_should_apply_attn_rotation(architecture, weight_type, head_dim)
}

pub(in crate::engine) fn uses_neox_rope(architecture: ModelArchitecture) -> bool {
    matches!(
        architecture,
        ModelArchitecture::Qwen2 | ModelArchitecture::Hy3
    ) || (matches!(architecture, ModelArchitecture::Gemma4)
        && crate::engine::policy::gemma_neox_rope_enabled())
}

pub(in crate::engine) fn resolve_rope_params(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layer_idx: usize,
    head_dim: usize,
) -> (usize, f32, bool) {
    if use_gemma_block_semantics(architecture)
        && metadata.rope_theta_swa != metadata.rope_theta
        && (is_gemma_sliding_window_layer(metadata, layer_idx) || head_dim < metadata.head_dim)
    {
        let rope_dim = if metadata.rope_dim_swa == 0 {
            head_dim
        } else {
            metadata.rope_dim_swa.min(head_dim)
        };
        (rope_dim, metadata.rope_theta_swa, false)
    } else {
        // Gemma4 full-attention layers rotate the full head_dim (rope.dimension_count == head_dim,
        // freq_factors come from `rope_freqs` tensor, not from a 1/4-proportional trick). See
        // docs/gemma4-e2b-it-runtime-contract.md §3 and llama.cpp src/models/gemma4-iswa.cpp
        // (freq_factors = rope_freqs for non-SWA, scalar RoPE otherwise).
        let is_gemma4 = matches!(architecture, ModelArchitecture::Gemma4);
        let proportional = !is_gemma4
            && use_gemma_block_semantics(architecture)
            && metadata.rope_theta_swa != metadata.rope_theta
            && !is_gemma_sliding_window_layer(metadata, layer_idx)
            && head_dim == metadata.head_dim
            && !env_flag("RNB_DISABLE_GEMMA_PROP_ROPE");
        let rope_dim = if proportional {
            (head_dim / 4).max(2)
        } else {
            metadata.rope_dim.min(head_dim)
        };
        (rope_dim, metadata.rope_theta, proportional)
    }
}

pub(in crate::engine) fn qwen_text_mrope_dim(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    rope_dim: usize,
    head_dim: usize,
) -> Option<usize> {
    if !matches!(
        architecture,
        ModelArchitecture::Qwen35 | ModelArchitecture::Qwen35MoE
    ) || !metadata.rope_sections.iter().any(|&s| s > 0)
        || !env_flag("RNB_QWEN_TEXT_IMROPE")
    {
        return None;
    }

    let section_dim = metadata.rope_sections.iter().sum::<usize>() * 2;
    let n_rot = if rope_dim > 0 { rope_dim } else { section_dim };
    debug_assert!(
        section_dim == 0 || section_dim == n_rot,
        "Qwen M-RoPE section sum {} does not match rope_dim {}",
        section_dim,
        n_rot
    );
    Some(n_rot.min(head_dim))
}

pub(in crate::engine) fn gemma_rope_freq_factors<'a>(
    rope_freqs: Option<&'a Tensor>,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layer_idx: usize,
    head_dim: usize,
) -> Option<&'a [f32]> {
    if !matches!(architecture, ModelArchitecture::Gemma4)
        || is_gemma_sliding_window_layer(metadata, layer_idx)
        || head_dim != metadata.head_dim
    {
        return None;
    }
    let tensor = rope_freqs?;
    let data = kernels::tensor_as_f32_slice(tensor);
    let pair_count = head_dim / 2;
    (data.len() >= pair_count).then_some(&data[..pair_count])
}

pub(in crate::engine) fn resolve_attention_scale(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
) -> f32 {
    if use_gemma_block_semantics(architecture) {
        if env_flag("RNB_GEMMA_USE_QUERY_PRE_ATTN_SCALE") {
            metadata.query_pre_attn_scalar.powf(-0.5)
        } else {
            1.0
        }
    } else {
        (metadata.head_dim as f32).powf(-0.5)
    }
}

pub(in crate::engine) fn resolve_attention_softcap(architecture: ModelArchitecture) -> Option<f32> {
    if use_gemma_block_semantics(architecture) && env_flag("RNB_GEMMA_ATTN_SOFTCAP") {
        Some(50.0)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_uses_neox_rope_layout() {
        assert!(uses_neox_rope(ModelArchitecture::Qwen2));
        assert!(!uses_neox_rope(ModelArchitecture::LLaMA));
    }
}
