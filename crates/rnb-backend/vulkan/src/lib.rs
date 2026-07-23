use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOp, BackendOutput,
    BackendRequest, BackendResult,
};

pub mod context;
mod ffi;
pub use context::GpuBuffer;
pub mod full_path;
mod gemv;
pub mod kv_resident;
mod layer_gemv;
pub mod per_kv_shard;
mod pipeline;
pub mod runtime;
pub mod spirv;
pub mod staging;
mod weight_cache;

pub use gemv::VulkanGemv;
pub use layer_gemv::{RuntimeCounters, VulkanLayerGemv};
pub use per_kv_shard::kv_head_shard_byte_range;
pub use weight_cache::{GpuWeightMode, QuantType, WeightId, WeightKind};

pub use runtime::{
    attn_norm_id, decode_layers_allowed, decode_layers_enabled, ffn_down_id, ffn_gate_id,
    ffn_norm_id, ffn_up_id, gdn_alpha_id, gdn_attn_norm_id, gdn_beta_id, gdn_gate_id,
    gdn_post_attn_norm_id, gdn_qkv_id, gdn_qkv_prefill_window_chunk, gdn_ssm_a_id,
    gdn_ssm_conv1d_id, gdn_ssm_dt_bias_id, gdn_ssm_norm_id, gdn_ssm_out_id, ggml_to_fullpath_quant,
    ggml_to_output_quant, ggml_to_quant, gpu_disabled_by_env, init_layer_gemv,
    initial_fullpath_staging_tokens, k_bias_id, k_norm_id, k_proj_id, k_proj_shard_id,
    max_decode_layer, o_proj_id, output_logits_enabled, output_logits_id,
    prefill_chunk_size_for_active_path, q_bias_id, q_norm_id, q_proj_id, v_bias_id, v_proj_id,
    v_proj_shard_id, verify_attention_layer, verify_attention_qkv_layer, verify_enabled,
    verify_gdn_layer, LayerGemv, LayerGemvConfig,
};

pub use runtime::init_layer_gemv_for_test;

#[derive(Debug, Default)]
pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::Vulkan).with_op(BackendOp::MatMul)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        Err(BackendError::unsupported(self.kind(), request.op()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_backend_adapter_is_explicit_about_unsupported_execution() {
        let mut backend = VulkanBackend::new();

        assert!(backend.capabilities().supports(BackendOp::MatMul));
        assert!(matches!(
            backend.execute(BackendRequest::new(BackendOp::Attention)),
            Err(err) if err.backend() == BackendKind::Vulkan && err.op() == Some(BackendOp::Attention)
        ));
    }
}
