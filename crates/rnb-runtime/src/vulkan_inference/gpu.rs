use super::super::vulkan_backend::init_layer_gemv_for_test as backend_init_layer_gemv_for_test;
use super::super::vulkan_backend::{
    decode_layers_allowed, ffn_down_id as backend_ffn_down_id, ffn_gate_id as backend_ffn_gate_id,
    ffn_up_id as backend_ffn_up_id, gdn_alpha_id as backend_gdn_alpha_id,
    gdn_beta_id as backend_gdn_beta_id, gdn_gate_id as backend_gdn_gate_id,
    gdn_qkv_id as backend_gdn_qkv_id,
    gdn_qkv_prefill_window_chunk as backend_gdn_qkv_prefill_window_chunk,
    gdn_ssm_out_id as backend_gdn_ssm_out_id, ggml_to_output_quant,
    ggml_to_quant as backend_ggml_to_quant, init_layer_gemv, k_proj_id, max_decode_layer,
    o_proj_id as backend_o_proj_id, output_logits_enabled, output_logits_id,
    prefill_chunk_size_for_active_path, q_proj_id, v_proj_id,
    verify_attention_layer as backend_verify_attention_layer,
    verify_attention_qkv_layer as backend_verify_attention_qkv_layer,
    verify_gdn_layer as backend_verify_gdn_layer, PrefillLayerRuntimeConfig, WeightId,
};
pub use rnb_backend_api::{
    AttentionKvMaterializeRangeRequest, AttentionKvMaterializeRequest, DecodeWeightKind,
};
use rnb_loader::GGMLType;

mod decode;
mod gdn_window;
mod layer_runtime;
mod materialize;
mod output;
mod window;

pub use decode::{
    decode_ffn_chain, decode_gemv, decode_gemv_multi, decode_gemv_multi_async,
    decode_gemv_multi_same_quant, decode_wait_async, DecodeGemvWeight,
};
pub use gdn_window::{gdn_prefill_ffn_chain_window, gdn_qkv_conv_window_from_resident_state};
pub use layer_runtime::{
    AttentionRawWeights, FullPathDecodeStepOutput, FullPathPrefillOutput, GdnRawWeights,
    KvResidentLayout, LayerRawWeights, LayerRuntime, ModelLayerKind, StagingPolicy,
};
pub use materialize::{
    materialize_attention_kv, materialize_attention_kv_range_untracked,
    materialize_gdn_conv_state_f32, write_gdn_conv_state_f32,
};
pub use output::{output_logits_quant_for_test, try_output_logits};
pub use window::{
    attention_block_window_for_layer, ffn_chain_window_with_residual_from_resident_input,
    o_proj_gemv_window,
};

pub type Runtime = LayerRuntime;
pub type RuntimeCounters = super::super::vulkan_backend::PrefillLayerRuntimeCounters;
pub type Quant = super::super::vulkan_backend::QuantType;

fn ffn_gate_id(layer_idx: usize) -> WeightId {
    backend_ffn_gate_id(layer_idx)
}

fn ffn_up_id(layer_idx: usize) -> WeightId {
    backend_ffn_up_id(layer_idx)
}

fn ffn_down_id(layer_idx: usize) -> WeightId {
    backend_ffn_down_id(layer_idx)
}

pub fn verify_attention_layer(layer_idx: usize) -> bool {
    backend_verify_attention_layer(layer_idx)
}

pub fn verify_attention_qkv_layer(layer_idx: usize) -> bool {
    backend_verify_attention_qkv_layer(layer_idx)
}

pub fn verify_gdn_layer(layer_idx: usize) -> bool {
    backend_verify_gdn_layer(layer_idx)
}

pub fn ggml_to_quant(ggml_type: GGMLType) -> Option<Quant> {
    backend_ggml_to_quant(ggml_type)
}

pub fn quantized_bytes_supported(ggml_type: GGMLType, bytes: Option<&[u8]>) -> bool {
    backend_ggml_to_quant(ggml_type).is_some() && bytes.is_some()
}

pub fn quant_or_error(ggml_type: GGMLType, label: &str) -> Result<Quant, String> {
    backend_ggml_to_quant(ggml_type).ok_or_else(|| format!("unsupported {label} quant"))
}

pub fn init_layer_gemv_for_test(
    max_input: usize,
    max_output: usize,
    chunk_size: usize,
) -> Result<Runtime, String> {
    backend_init_layer_gemv_for_test(max_input, max_output, chunk_size)
        .map(LayerRuntime::from_backend)
}

pub fn init_prefill_layer_runtime(
    hidden_dim: usize,
    ffn_inner_dim: usize,
    max_layer_rows: usize,
    output_rows: usize,
) -> Option<Runtime> {
    let config = PrefillLayerRuntimeConfig::from_model_shape(
        hidden_dim,
        ffn_inner_dim,
        max_layer_rows,
        output_rows,
    );
    init_layer_gemv(config).map(LayerRuntime::from_backend)
}

pub fn gpu_output_logits_enabled() -> bool {
    output_logits_enabled()
}

pub fn prefill_runtime_counters(runtime: Option<&Runtime>) -> Option<RuntimeCounters> {
    runtime.map(|vk| vk.runtime_counters())
}

pub fn active_prefill_chunk_size(active_gpu_prefill_path: bool, hidden_dim: usize) -> usize {
    prefill_chunk_size_for_active_path(active_gpu_prefill_path, hidden_dim)
}

pub fn gdn_qkv_prefill_window_chunk(hidden_dim: usize) -> usize {
    backend_gdn_qkv_prefill_window_chunk(hidden_dim)
}

pub fn decode_layers_policy_allows() -> bool {
    decode_layers_allowed()
}

pub fn decode_layers_policy_max_layer() -> usize {
    max_decode_layer()
}

pub fn layer_quant_debug(ggml_type: GGMLType) -> String {
    format!("{:?}", ggml_to_quant(ggml_type))
}

pub fn layer_quant_supported(ggml_type: GGMLType) -> bool {
    ggml_to_quant(ggml_type).is_some()
}
