//! Prefill FFN helpers.

use super::super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn forward_prefill_ffn(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());

    if let Some(moe_w) = &w.moe {
        return forward_ffn_gemma4_moe_hybrid(
            architecture,
            hidden,
            w,
            moe_w,
            seq_len,
            metadata.hidden_dim,
            norm_eps,
            layer_idx,
        );
    }

    if let Some(moe_w) = &w.shared_expert_moe {
        let ffn_norm_w = select_ffn_pre_norm_weight(w, architecture);
        return forward_shared_expert_moe(
            architecture,
            hidden,
            ffn_norm_w,
            moe_w,
            seq_len,
            metadata.hidden_dim,
            norm_eps,
            layer_idx,
        );
    }

    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    let normed = if use_gemma_block_semantics(architecture)
        && policy::gemma_unit_offset_ffn_pre_norm_enabled(layer_idx)
    {
        apply_model_norm_unit_offset(&hidden, ffn_norm, norm_eps).map_err(fwd)?
    } else {
        apply_model_norm(&hidden, ffn_norm, norm_eps, architecture).map_err(fwd)?
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "pre_ffn_norm",
            kernels::tensor_as_f32_slice(&normed),
        );
    }
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    #[cfg(feature = "cuda")]
    let cuda_down = if use_gemma_block_semantics(architecture) {
        if dump_bin_dir().is_none() {
            if let Some(down) = backend_runtime::dense_q4k_gelu_ffn_batch_if_supported(
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
                normed_data,
                seq_len,
            )? {
                Some(Tensor::from_vec(down, &[seq_len, metadata.hidden_dim]))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_down: Option<Tensor> = None;

    // pm33: Metal prefill FFN batch GEMM chain. non-Gemma(SiLU) + gate/up 분리(fused 제외) 한정,
    // RNB_METAL_PREFILL_FFN env opt-in. shim 이 quant/shape 미지원 시 used=false → CPU fallback.
    #[cfg(feature = "metal")]
    let metal_down: Option<Tensor> = if !use_gemma_block_semantics(architecture)
        && w.ffn_gate_up_fused.is_none()
        && dump_bin_dir().is_none()
    {
        let mut out = vec![0f32; seq_len * metadata.hidden_dim];
        let used = backend_runtime::metal_prefill_ffn_chain_into_if_supported(
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            normed_data,
            &mut out,
            seq_len,
            metadata.hidden_dim,
        )?;
        if used {
            Some(Tensor::from_vec(out, &[seq_len, metadata.hidden_dim]))
        } else {
            None
        }
    } else {
        None
    };
    #[cfg(not(feature = "metal"))]
    let metal_down: Option<Tensor> = None;

    #[cfg(feature = "mediatek")]
    let mediatek_down: Option<Tensor> = if super::super::mediatek_ffn::prefill_enabled()
        && use_gemma_block_semantics(architecture)
        && w.ffn_gate_up_fused.is_none()
        && dump_bin_dir().is_none()
    {
        super::super::mediatek_ffn::try_mediatek_gemma_prefill_ffn_down(
            architecture,
            layer_idx,
            metadata.hidden_dim,
            seq_len,
            &normed_data,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            false,
        )?
        .map(|down| Tensor::from_vec(down, &[seq_len, metadata.hidden_dim]))
    } else {
        None
    };
    #[cfg(not(feature = "mediatek"))]
    let mediatek_down: Option<Tensor> = None;

    let down = if let Some(down) = cuda_down.or(metal_down).or(mediatek_down) {
        down
    } else {
        let (mut gate_vec, up_vec) = prefill_gate_up_vectors(
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            w.ffn_gate_up_fused.as_ref(),
            normed_data,
            seq_len,
        )?;
        if dump_bin_dir().is_some() {
            dump_bin("prefill", layer_idx, "mlp_gate", &gate_vec);
            dump_bin("prefill", layer_idx, "mlp_up", &up_vec);
        }
        apply_model_gate_mul_inplace(&mut gate_vec, &up_vec, architecture);
        if dump_bin_dir().is_some() {
            dump_bin("prefill", layer_idx, "mlp_gated", &gate_vec);
        }
        let ffn_inner_dim = gate_vec.len() / seq_len;
        let gate_up_tensor = Tensor::from_vec(gate_vec, &[seq_len, ffn_inner_dim]);
        w.ffn_down_weight.gemv(&gate_up_tensor)?
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "mlp_down",
            kernels::tensor_as_f32_slice(&down),
        );
    }
    let down = if use_gemma_block_semantics(architecture) {
        if let Some(post_ffw_norm) = &w.post_ffw_norm {
            if use_gemma_block_semantics(architecture)
                && policy::gemma_unit_offset_ffn_post_norm_enabled()
            {
                apply_model_norm_unit_offset(&down, post_ffw_norm, norm_eps).map_err(fwd)?
            } else {
                apply_model_norm(&down, post_ffw_norm, norm_eps, architecture).map_err(fwd)?
            }
        } else {
            down
        }
    } else {
        down
    };
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "post_ffn_norm",
            kernels::tensor_as_f32_slice(&down),
        );
    }
    let hidden = add_tensors(&hidden, &down).map_err(fwd)?;
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "residual_2",
            kernels::tensor_as_f32_slice(&hidden),
        );
    }

    Ok(hidden)
}
