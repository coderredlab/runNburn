//! Q/K/V projection step for attention decode.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_qkv_projection<F>(
    scratch: &mut ScratchBuffers,
    w: &AttentionLayerWeights,
    layer_idx: usize,
    hidden_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
    gemma4_reuse_q_only: bool,
    metal_chain_default: bool,
    verbose: bool,
    rms_used_cuda: bool,
    #[cfg(feature = "vulkan")] gpu_runtime: &mut Option<&mut backend_runtime::GpuRuntime>,
    mut profile: F,
) -> crate::error::Result<()>
where
    F: FnMut(&str, std::time::Instant),
{
    if backend_runtime::verify_attention_layer(layer_idx) {
        #[cfg(feature = "vulkan")]
        let gpu_runtime_present = gpu_runtime.is_some();
        #[cfg(not(feature = "vulkan"))]
        let gpu_runtime_present = false;
        log_decode_attention_gpu_debug(layer_idx, gpu_runtime_present, w);
    }
    let gpu_qkv_ok = backend_runtime::try_decode_attention_qkv_if_supported(
        #[cfg(feature = "vulkan")]
        gpu_runtime.as_deref_mut(),
        layer_idx,
        &scratch.norm_buf[..hidden_dim],
        &w.q_weight,
        &w.k_weight,
        &w.v_weight,
        q_out_dim,
        kv_dim,
        &mut scratch.q_buf[..q_out_dim],
        &mut scratch.k_buf[..kv_dim],
        &mut scratch.v_buf[..kv_dim],
    );
    if gpu_qkv_ok && backend_runtime::verify_attention_qkv_layer(layer_idx) {
        verify_decode_attention_qkv(layer_idx, scratch, w, hidden_dim, q_out_dim, kv_dim);
    }
    // Metal QKV device-resident chain. Gemma Q4_K는 기본 활성이고 다른 모델은
    // RNB_METAL_QKV_CHAIN=1 opt-in이다. V projection 부재 시 raw K를 그대로 재사용한다.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let gpu_qkv_ok = if !gpu_qkv_ok && !gemma4_reuse_q_only {
        backend_runtime::metal_attention_qkv_chain_into_if_supported(
            &scratch.norm_buf[..hidden_dim],
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            &mut scratch.q_buf[..q_out_dim],
            &mut scratch.k_buf[..kv_dim],
            &mut scratch.v_buf[..kv_dim],
            hidden_dim,
            q_out_dim,
            kv_dim,
            metal_chain_default,
            w.v_proj_missing,
        )?
    } else {
        gpu_qkv_ok
    };
    #[cfg(feature = "cuda")]
    let gpu_qkv_ok = if !gpu_qkv_ok && !gemma4_reuse_q_only {
        backend_runtime::dense_q4k_attention_qkv_if_supported(
            &w.q_weight,
            &w.k_weight,
            &w.v_weight,
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.q_buf[..q_out_dim],
            &mut scratch.k_buf[..kv_dim],
            &mut scratch.v_buf[..kv_dim],
            rms_used_cuda,
        )?
    } else {
        gpu_qkv_ok
    };
    let gpu_qkv_ok = if !gpu_qkv_ok {
        if gemma4_reuse_q_only {
            gpu_gemv_into_if_supported(
                &w.q_weight,
                &scratch.norm_buf[..hidden_dim],
                &mut scratch.q_buf[..q_out_dim],
                "attention q",
                rms_used_cuda,
            )?
        } else {
            let q_ok = gpu_gemv_into_if_supported(
                &w.q_weight,
                &scratch.norm_buf[..hidden_dim],
                &mut scratch.q_buf[..q_out_dim],
                "attention q",
                rms_used_cuda,
            )?;
            let k_ok = gpu_gemv_into_if_supported(
                &w.k_weight,
                &scratch.norm_buf[..hidden_dim],
                &mut scratch.k_buf[..kv_dim],
                "attention k",
                rms_used_cuda,
            )?;
            let v_ok = if w.v_proj_missing {
                true
            } else {
                gpu_gemv_into_if_supported(
                    &w.v_weight,
                    &scratch.norm_buf[..hidden_dim],
                    &mut scratch.v_buf[..kv_dim],
                    "attention v",
                    rms_used_cuda,
                )?
            };
            q_ok && k_ok && v_ok
        }
    } else {
        gpu_qkv_ok
    };

    if !gpu_qkv_ok {
        if gemma4_reuse_q_only {
            let norm = &scratch.norm_buf[..hidden_dim];
            w.q_weight
                .gemv_into(norm, &mut scratch.q_buf[..q_out_dim])?;
        } else if force_generic_attn_qkv_layer(layer_idx) {
            let norm = &scratch.norm_buf[..hidden_dim];
            w.q_weight
                .gemv_into_generic(norm, &mut scratch.q_buf[..q_out_dim])?;
            w.k_weight
                .gemv_into_generic(norm, &mut scratch.k_buf[..kv_dim])?;
            w.v_weight
                .gemv_into_generic(norm, &mut scratch.v_buf[..kv_dim])?;
        } else {
            decode_attention_qkv_cpu_into(
                scratch,
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                hidden_dim,
                q_out_dim,
                kv_dim,
                verbose,
                |label, t| profile(label, t),
            )?;
        }
    }

    if w.v_proj_missing && !gemma4_reuse_q_only {
        scratch.v_buf[..kv_dim].copy_from_slice(&scratch.k_buf[..kv_dim]);
    }

    Ok(())
}
