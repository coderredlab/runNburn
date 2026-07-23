//! Prefill RoPE and Gemma attention rotation helpers.

use super::super::*;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_apply_rope_tensor(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    rope_dim: usize,
    rope_theta: f32,
    proportional_rope: bool,
    mrope_dim: Option<usize>,
    use_neox_rope: bool,
    freq_factors: Option<&[f32]>,
) -> crate::error::Result<Tensor> {
    let (mode, n_rot, factors) = if let Some(mrope_dim) = mrope_dim {
        (
            backend_runtime::CudaForwardRopeMode::MropeText,
            mrope_dim,
            None,
        )
    } else if use_neox_rope {
        (
            backend_runtime::CudaForwardRopeMode::Neox,
            head_dim,
            freq_factors,
        )
    } else if freq_factors.is_some() {
        (
            backend_runtime::CudaForwardRopeMode::Adjacent,
            head_dim,
            freq_factors,
        )
    } else if rope_dim > 0 && rope_dim < head_dim {
        (
            if proportional_rope {
                backend_runtime::CudaForwardRopeMode::PartialProportional
            } else {
                backend_runtime::CudaForwardRopeMode::PartialAdjacent
            },
            rope_dim,
            None,
        )
    } else {
        (
            backend_runtime::CudaForwardRopeMode::Adjacent,
            head_dim,
            None,
        )
    };
    let mut output = kernels::tensor_as_f32_slice(input).to_vec();
    backend_runtime::cuda_rope_f32_inplace(
        &mut output,
        dim,
        head_dim,
        n_rot,
        pos_start,
        rope_theta,
        mode,
        factors,
    )?;
    Ok(Tensor::from_vec(output, input.shape()))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_prefill_rope(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    q: Tensor,
    k: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    head_dim: usize,
    kv_dim: usize,
) -> crate::error::Result<(Tensor, Option<Tensor>)> {
    if matches!(architecture, ModelArchitecture::NemotronHMoE) {
        return Ok((q, k.cloned()));
    }

    #[cfg(not(feature = "cuda"))]
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let (rope_dim, rope_theta, proportional_rope) =
        resolve_rope_params(metadata, architecture, layer_idx, head_dim);
    let freq_factors =
        gemma_rope_freq_factors(rope_freqs, metadata, architecture, layer_idx, head_dim);
    let qwen_mrope_dim = qwen_text_mrope_dim(metadata, architecture, rope_dim, head_dim);
    let use_neox_rope = uses_neox_rope(architecture);
    #[cfg(feature = "cuda")]
    let q = cuda_apply_rope_tensor(
        &q,
        pos_start,
        head_dim,
        layout.q_dim,
        rope_dim,
        rope_theta,
        proportional_rope,
        qwen_mrope_dim,
        use_neox_rope,
        freq_factors,
    )?;
    #[cfg(not(feature = "cuda"))]
    let q = if let Some(mrope_dim) = qwen_mrope_dim {
        kernels::rope::rope_mrope_text(&q, pos_start, head_dim, mrope_dim, rope_theta)
            .map_err(fwd)?
    } else if use_neox_rope {
        if let Some(freq_factors) = freq_factors {
            kernels::rope::rope_neox_with_factors(&q, pos_start, head_dim, rope_theta, freq_factors)
                .map_err(fwd)?
        } else {
            kernels::rope::rope_neox(&q, pos_start, head_dim, rope_theta).map_err(fwd)?
        }
    } else if let Some(freq_factors) = freq_factors {
        kernels::rope::rope_with_factors(&q, pos_start, head_dim, rope_theta, freq_factors)
            .map_err(fwd)?
    } else if rope_dim > 0 && rope_dim < head_dim {
        if proportional_rope {
            kernels::rope::rope_partial_proportional(&q, pos_start, head_dim, rope_dim, rope_theta)
        } else {
            kernels::rope::rope_partial(&q, pos_start, head_dim, rope_dim, rope_theta)
        }
        .map_err(fwd)?
    } else {
        kernels::rope::rope(&q, pos_start, head_dim, rope_theta).map_err(fwd)?
    };
    #[cfg(feature = "cuda")]
    let k_rope = if let Some(k_tensor) = k {
        Some(cuda_apply_rope_tensor(
            k_tensor,
            pos_start,
            head_dim,
            kv_dim,
            rope_dim,
            rope_theta,
            proportional_rope,
            qwen_mrope_dim,
            use_neox_rope,
            freq_factors,
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let k_rope = if let Some(k_tensor) = k {
        if let Some(mrope_dim) = qwen_mrope_dim {
            Some(
                kernels::rope::rope_mrope_text(
                    k_tensor, pos_start, head_dim, mrope_dim, rope_theta,
                )
                .map_err(fwd)?,
            )
        } else if use_neox_rope {
            Some(if let Some(freq_factors) = freq_factors {
                kernels::rope::rope_neox_with_factors(
                    k_tensor,
                    pos_start,
                    head_dim,
                    rope_theta,
                    freq_factors,
                )
                .map_err(fwd)?
            } else {
                kernels::rope::rope_neox(k_tensor, pos_start, head_dim, rope_theta).map_err(fwd)?
            })
        } else if let Some(freq_factors) = freq_factors {
            Some(
                kernels::rope::rope_with_factors(
                    k_tensor,
                    pos_start,
                    head_dim,
                    rope_theta,
                    freq_factors,
                )
                .map_err(fwd)?,
            )
        } else if rope_dim > 0 && rope_dim < head_dim {
            Some(
                if proportional_rope {
                    kernels::rope::rope_partial_proportional(
                        k_tensor, pos_start, head_dim, rope_dim, rope_theta,
                    )
                } else {
                    kernels::rope::rope_partial(k_tensor, pos_start, head_dim, rope_dim, rope_theta)
                }
                .map_err(fwd)?,
            )
        } else {
            Some(kernels::rope::rope(k_tensor, pos_start, head_dim, rope_theta).map_err(fwd)?)
        }
    } else {
        None
    };
    let q = if gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, head_dim) {
        gemma4_apply_attn_rot_tensor(&q, head_dim, layout.q_dim, gemma4_k_rot_size(head_dim))
    } else {
        q
    };
    let k_rope = k_rope.map(|k_rope| {
        if gemma4_should_apply_k_rotation(architecture, w.k_weight.ggml_type, head_dim) {
            gemma4_apply_attn_rot_tensor(&k_rope, head_dim, kv_dim, gemma4_k_rot_size(head_dim))
        } else {
            k_rope
        }
    });
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "q_rope",
            kernels::tensor_as_f32_slice(&q),
        );
        if let Some(ref k_rope_tensor) = k_rope {
            dump_bin(
                "prefill",
                layer_idx,
                "k_rope",
                kernels::tensor_as_f32_slice(k_rope_tensor),
            );
        }
    }
    if layer_idx == 0 && attn_trace_enabled() {
        let q_data = kernels::tensor_as_f32_slice(&q);
        let q_last = &q_data[(seq_len - 1) * layout.q_dim..seq_len * layout.q_dim];
        emit_vec_trace("prefill", layer_idx, "q_last", q_last);
        if let Some(k_tensor) = k {
            let k_data_pre = kernels::tensor_as_f32_slice(k_tensor);
            let k_last_pre = &k_data_pre[(seq_len - 1) * kv_dim..seq_len * kv_dim];
            emit_vec_trace("prefill", layer_idx, "k_last_pre", k_last_pre);
        }
        if let Some(k_rope) = &k_rope {
            let k_data_post = kernels::tensor_as_f32_slice(k_rope);
            let k_last = &k_data_post[(seq_len - 1) * kv_dim..seq_len * kv_dim];
            emit_vec_trace("prefill", layer_idx, "k_last", k_last);
        }
    }
    if targeted_attn_trace_enabled(layer_idx) {
        let q_data = kernels::tensor_as_f32_slice(&q);
        let q_last = &q_data[(seq_len - 1) * layout.q_dim..seq_len * layout.q_dim];
        emit_vec_trace("prefill-l34", layer_idx, "q_last", q_last);
    }
    Ok((q, k_rope))
}
