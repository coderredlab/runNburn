//! Decode-time FFN path selection.

use super::*;

pub(super) fn decode_ffn_layer(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    w: &AttentionLayerWeights,
    hidden_dim: usize,
    norm_eps: f32,
    layer_idx: usize,
    #[cfg(feature = "vulkan")] gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    let ffn_norm = select_ffn_pre_norm_weight(w, architecture);
    if let Some(moe_w) = &w.shared_expert_moe {
        decode_shared_expert_moe(
            scratch,
            architecture,
            ffn_norm,
            moe_w,
            hidden_dim,
            norm_eps,
            layer_idx,
        )?;
    } else if let Some(moe_w) = &w.moe {
        decode_ffn_gemma4_moe_hybrid(
            scratch,
            architecture,
            ffn_norm,
            &w.post_ffw_norm,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            moe_w,
            hidden_dim,
            norm_eps,
            layer_idx,
        )?;
    } else {
        decode_ffn(
            scratch,
            architecture,
            ffn_norm,
            &w.post_ffw_norm,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            &w.ffn_gate_up_fused,
            hidden_dim,
            norm_eps,
            layer_idx,
            #[cfg(feature = "vulkan")]
            gpu_runtime,
        )?;
    }
    Ok(())
}
