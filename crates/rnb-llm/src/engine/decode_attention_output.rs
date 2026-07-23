//! Output projection step for attention decode.

use super::*;

pub(super) fn decode_attention_output_projection(
    scratch: &mut ScratchBuffers,
    w: &AttentionLayerWeights,
    layer_idx: usize,
    q_dim: usize,
    hidden_dim: usize,
    #[cfg(feature = "vulkan")] gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    let gpu_oproj_ok = backend_runtime::try_decode_gemv_if_supported(
        #[cfg(feature = "vulkan")]
        gpu_runtime,
        layer_idx,
        backend_runtime::DecodeProjectionKind::OProj,
        &w.o_weight,
        hidden_dim,
        &scratch.attn_out[..q_dim],
        &mut scratch.proj_buf[..hidden_dim],
        "o_proj",
    );
    let gpu_oproj_ok = if !gpu_oproj_ok {
        gpu_gemv_into_if_supported(
            &w.o_weight,
            &scratch.attn_out[..q_dim],
            &mut scratch.proj_buf[..hidden_dim],
            "attention o",
            false,
        )?
    } else {
        gpu_oproj_ok
    };

    if !gpu_oproj_ok {
        if force_generic_attn_oproj_layer(layer_idx) {
            w.o_weight.gemv_into_generic(
                &scratch.attn_out[..q_dim],
                &mut scratch.proj_buf[..hidden_dim],
            )?;
        } else {
            w.o_weight.gemv_into(
                &scratch.attn_out[..q_dim],
                &mut scratch.proj_buf[..hidden_dim],
            )?;
        }
    }

    Ok(())
}
