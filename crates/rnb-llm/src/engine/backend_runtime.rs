mod cuda_basic;
mod engine_ext;
mod glm_moe;
mod gpu_gdn;
mod init;
mod materialize;
mod output;
mod qwen_moe;
mod state;
mod verify;
mod vulkan_attention_prefill;
mod vulkan_decode;
mod vulkan_gdn_prefill;

#[cfg(feature = "cuda")]
pub(super) use super::cuda_runtime::{
    acquire_decode_attn_out_carrier, acquire_decode_k_carrier, acquire_decode_k_f16_carrier,
    acquire_decode_norm_buf_carrier, acquire_decode_v_carrier, acquire_decode_v_f16_carrier,
    decode_device_qkv_rope_kv, decode_device_qkv_rope_kv_graph, f32_to_f16_pack_device,
    upload_to_decode_hidden_carrier, Cu71LayerSegmentGraphRuntimeContext,
};

// cu29/cu30 의 fused QKV+RoPE 함수는 CPU 빌드에서도 stub (`Ok(false)` / `Ok(None)`)
// 가 동작하도록 본문이 분기되어 있다. decode.rs / projection.rs callsite 가 cfg
// 분기 없이 호출하므로 re-export 는 unconditional 이어야 한다. cu29-39 mega-cycle
// 의 회귀 fix (mc78 사이드).
#[cfg(all(test, feature = "cuda"))]
pub(super) use cuda_basic::{
    cuda_product_prewarm_forbidden_expanded_calls_for_test,
    cuda_product_prewarm_q4_raw_count_for_test,
    cuda_product_prewarm_quant_resident_executor_missing_for_test,
    cuda_product_prewarm_request_kinds_for_test,
    cuda_product_prewarm_wrapper_executor_violations_for_test,
    CudaProductPrewarmRequestKindForTest,
};
#[allow(unused_imports)]
pub(super) use cuda_basic::{
    dense_q4k_attention_qkv_rope_hd128_if_supported,
    dense_q4k_attention_qkv_rope_hd128_prefill_if_supported,
    try_rms_norm_into_decode_carrier_if_supported,
};

#[cfg(feature = "cuda")]
pub(super) use cuda_basic::decode_attention_cached_to_device_if_supported;
#[cfg(feature = "cuda")]
pub(super) use cuda_basic::decode_attention_hd256_if_supported;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(super) use cuda_basic::{
    bf16_gemv, dense_q4k_attention_output_gelu_ffn_norm_residual_if_supported,
    dense_q4k_attention_qkv_if_supported, dense_q4k_gelu_ffn_batch_if_supported,
    dense_q4k_gelu_ffn_if_supported, dense_q4k_gelu_ffn_norm_residual_if_supported,
    download_cuda_device_tensor_f32, download_nemotron_device_layer_output,
    download_nemotron_device_layer_output_row, download_nemotron_device_prefill_handoff,
    nemotron_device_route_pack_expert_ids, nemotron_device_route_pack_from_logits,
    nemotron_mamba2_prefill_device, nemotron_q8_shared_q5_sparse_prefill_moe_device_ids,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_output,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack_ids,
    nemotron_reorder_device_route_pack, nemotron_router_logits_from_device_f32,
    prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output_if_supported,
    prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output_if_supported,
    prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_device_output_if_supported,
    prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output_if_supported,
    release_nemotron_device_route_pack, upload_hidden_device_output_f32, NemotronDeviceLayerOutput,
    NemotronDeviceRoutePack, NemotronDeviceRouterLogitsOutput, NemotronMamba2DeviceTrace,
};
#[cfg(feature = "cuda")]
pub(super) use cuda_basic::{
    cuda_cache_snapshot, decode_gemv_cuda, f16_gemv as cuda_f16_gemv, prefill_gemv_cuda,
    prefill_output_argmax_token_cuda, prefill_output_logits_cuda,
};
pub(super) use cuda_basic::{
    decode_gemv_into_if_supported,
    dense_q4k_attention_output_gelu_ffn_batch_norm_residual_if_supported,
    gdn_prefill_quantized_projection, gdn_prefill_quantized_projection_q,
    gemma4_ple_q4k_batch_norm_residual_if_supported, metal_attn_layer_into_if_supported,
    metal_decode_parity_counters_report, metal_decode_parity_counters_reset,
    metal_prefill_atn_full_counters_report, metal_prefill_atn_full_counters_reset,
    metal_prefill_atn_full_expected_dense_layer, metal_prefill_atn_full_record_adapter_reject,
    metal_prefill_atn_full_record_backend_err, metal_prefill_atn_full_record_core_hit,
    metal_prefill_atn_full_record_full_layer_hit, metal_prefill_atn_full_record_skip,
    metal_prefill_atn_full_timing_enabled, metal_prefill_atn_o_tail_counters_report,
    metal_prefill_atn_o_tail_counters_reset, metal_prefill_atn_o_tail_expected_dense_layer,
    metal_prefill_atn_o_tail_requested, metal_prefill_attn_chain_if_supported,
    metal_prefill_delta_net_scan_into_if_supported, metal_prefill_gdn_f32_dual_proj_if_supported,
    metal_prefill_gdn_proj_into_if_supported, metal_rope_mrope_into_if_supported,
    nemotron_prefill_sparse_copy_prefetch, nemotron_q5_decode_moe_shared_sparse,
    nemotron_q5_q8_sparse_relu_sqr_by_token, nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token,
    nemotron_q5_sparse_relu_sqr_by_token, nemotron_q5_sparse_relu_sqr_full_layer_by_token,
    nemotron_q8_shared_prefill, nemotron_q8_shared_q5_sparse_decode_enabled,
    nemotron_q8_shared_q5_sparse_decode_moe, nemotron_q8_shared_q5_sparse_decode_moe_cached_layer,
    nemotron_q8_shared_q5_sparse_prefill_moe,
    nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_probe,
    prefill_attention_f16kv_dense_chain_if_supported, prefill_attention_f16kv_if_supported,
    prefill_attention_f16kv_window_dense_chain_if_supported,
    prefill_attention_f16kv_window_if_supported, prefill_attention_hd256_if_supported,
    prefill_attention_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_if_supported,
    prefill_attention_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_if_supported,
    prefill_attention_q4k_f16_qkv_attention_hd512_dense_chain_if_supported,
    prefill_attention_q4k_f16_qkv_attention_hd512_if_supported,
    prefill_attention_q4k_f16_qkv_if_supported,
    prefill_attention_q4k_f16_qkv_postprocess_hd256_if_supported,
    prefill_attention_q4k_f16_qkv_postprocess_hd256_window_dense_chain_if_supported,
    prewarm_dense_q4_packed_gate_up_weights, prewarm_dense_q6_packed_down_weights,
    prewarm_prefill_q4_f32_projection_weights, prewarm_q4k_weight_slices, q5_basic_gemv,
    q5_basic_gemv_raw, try_delta_restore_step_if_supported, try_delta_step_if_supported,
    upload_gemma_ple_base,
};
#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
pub(super) use cuda_basic::{
    metal_attention_o_chain_into_if_supported, metal_attention_qkv_chain_into_if_supported,
    metal_decode_attn_carrier_kv_filled, metal_decode_chain_run,
    metal_decode_kv_int8_requires_carrier_error, metal_decode_legacy_attn_layer_enabled_by_policy,
    metal_decode_legacy_carrier_enabled_by_policy, metal_decode_parity_record_expected_token,
    metal_ffn_chain_into_if_supported, metal_gdn_core_into_if_supported,
    metal_gdn_inproj_chain_into_if_supported, metal_gdn_layer_into_if_supported,
    metal_gdn_moe_layer_into_if_supported, metal_prefill_atn_o_tail_record_backend_err,
    metal_prefill_atn_o_tail_record_hit, metal_prefill_atn_o_tail_record_skip,
    metal_prefill_atn_o_tail_timing_enabled, metal_prefill_ffn_chain_into_if_supported,
    metal_qwen_moe_decode_chain_enabled_by_policy, ChainAttnShape, ChainLayerInput,
    MetalDecodeChainRunReport, MetalDecodeOutputArgmax,
};
#[cfg(not(feature = "cuda"))]
pub(super) use cuda_basic::{
    metal_attn_decode_into_if_supported, metal_attn_decode_kv_resident_into_if_supported,
};
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(super) use cuda_basic::{
    metal_prefill_atn_core_if_supported, metal_prefill_atn_full_layer_if_supported,
    metal_prefill_atn_o_tail_if_supported, metal_prefill_atn_o_tail_record_adapter_reject,
    metal_prefill_gdn_proj_into_if_supported_with_trace, MetalPrefillAtnCoreShape, MetalProjTrace,
};
#[cfg(feature = "cuda")]
pub(super) use cuda_basic::{nemotron_q5_q8_register_layer, nemotron_q5_register_layer};
#[cfg(feature = "cuda")]
pub(super) use glm_moe::glm_moe_prefill_sparse_experts_iq2xxs_iq3xxs_by_token;
pub(super) use glm_moe::{
    glm_moe_decode_shared_expert_q5k_q6k, glm_moe_decode_sparse_experts_iq2xxs_iq3xxs,
};
#[cfg(feature = "cuda")]
pub(super) use gpu_gdn::ssm_prefill_delta_net_snapshots;
pub(super) use gpu_gdn::{
    ensure_gdn_prefill_chunk_supported, gdn_prefill_chain_q4k, gdn_prefill_conv_delta_chain,
    gdn_prefill_full_chain, gdn_prefill_full_ffn_chain, gdn_prefill_gated_norm_silu,
    gdn_prefill_gated_norm_silu_project, nemotron_mamba2_decode_scan, nemotron_mamba2_prefill_scan,
    ssm_prefill_conv1d_silu, ssm_prefill_delta_net, try_gdn_prefill_chain_if_supported,
    GdnPrefillChainShape,
};
#[cfg(feature = "vulkan")]
pub(super) use init::init_prefill_layer_runtime;
pub(super) use init::{
    clear_decode_attention_kv_cache_before_prefill, clear_host_registered_ranges_before_prefill,
    release_prefill_residency_after_prefill, reset_backend_state_for_engine_init,
};
pub(super) use materialize::materialize_attention_kv_for_layer_if_supported;
#[cfg(feature = "vulkan")]
pub(super) use materialize::{
    materialize_attention_kv_for_layer, materialize_attention_kv_range_untracked,
    materialize_gdn_conv_state_untracked, record_batched_materialization_download,
};
#[cfg(all(test, feature = "vulkan"))]
pub(super) use output::ggml_to_gpu_output_quant;
#[cfg(feature = "cuda")]
pub(super) use output::prewarm_backend_output_weight_for_runtime;
pub(super) use output::{
    backend_output_argmax_supported_for_runtime, force_prewarm_backend_output_weight_for_runtime,
};
pub(super) use output::{output_logits_enabled_for_runtime, try_backend_output_logits_for_runtime};
#[allow(unused_imports)]
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(super) use qwen_moe::qwen_moe_llama_id_prefill;
#[cfg(feature = "cuda")]
pub(super) use qwen_moe::qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(super) use qwen_moe::qwen_moe_prefill_shared_sparse_selected_base_metal;
#[allow(unused_imports)]
pub(super) use qwen_moe::{
    glm_mla_back_into, glm_mla_front_into, glm_mla_gemv_into, glm_mla_head_gemv_into,
    glm_mla_head_slots_gemv_into, glm_moe_decode_iq2xxs_iq3xxs_into, glm_moe_prefill_iq_batch_into,
    log_qwen_moe_prefill_route_array_hist, log_qwen_moe_prefill_route_hist, qwen_moe_decode_down,
    qwen_moe_decode_expert, qwen_moe_decode_gate_up, qwen_moe_decode_shared_sparse_experts_into,
    qwen_moe_decode_sparse_batch_enabled, qwen_moe_decode_sparse_experts,
    qwen_moe_decode_sparse_experts_add_residual_into, qwen_moe_decode_sparse_experts_id_into,
    qwen_moe_decode_sparse_experts_into, qwen_moe_decode_sparse_experts_iq4xs,
    qwen_moe_decode_sparse_experts_iq4xs_add_residual_into,
    qwen_moe_decode_sparse_experts_iq4xs_into, qwen_moe_decode_sparse_experts_per_slot_resident,
    qwen_moe_llama_id_requested, qwen_moe_prefill_combined_f32_shared_sparse_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token,
    qwen_moe_prefill_enabled_for_seq, qwen_moe_prefill_router_logits,
    qwen_moe_prefill_shared_expert_batch,
    qwen_moe_prefill_shared_sparse_selected_base_metal_v3_requested,
    qwen_moe_prefill_shared_sparse_selected_base_metal_v3_token_major,
    qwen_moe_prefill_shared_sparse_selected_base_metal_v4_requested,
    qwen_moe_prefill_shared_sparse_selected_base_metal_v4_token_major,
    qwen_moe_prefill_sparse_experts_by_token,
    qwen_moe_prefill_sparse_experts_selected_base_by_token,
    qwen_moe_prefill_sparse_route_arrays_device_topk, qwen_moe_prefill_sparse_slots,
    qwen_moe_prefill_sparse_slots_expert_major, qwen_moe_prepare_selected_bundle_residency,
    qwen_moe_register_layer,
};
#[cfg(feature = "vulkan")]
pub(super) use state::GpuRuntime;
#[cfg(all(feature = "vulkan", test))]
pub(super) use state::{ggml_to_quant_for_test, init_layer_gemv_for_test, GpuQuant};
pub(super) use state::{init_engine_backend_runtime, is_attention_layer, EngineBackendRuntime};
pub(super) use verify::{verify_attention_layer, verify_attention_qkv_layer, verify_gdn_layer};
#[cfg(feature = "vulkan")]
pub(super) use vulkan_attention_prefill::{
    attention_window_fast_path_supported, attention_window_needs_qk_norm_path,
    run_attention_window_attention_only_for_layer, try_attention_ffn_window_fast_path,
};
#[cfg(feature = "vulkan")]
pub(super) use vulkan_decode::wait_decode_async_if_supported;
pub(super) use vulkan_decode::{
    try_decode_attention_qkv_if_supported, try_decode_attention_single_head_if_supported,
    try_decode_ffn_chain_if_supported, try_decode_ffn_gate_async_if_supported,
    try_decode_gdn_alpha_beta_if_supported, try_decode_gdn_qkv_gate_if_supported,
    try_decode_gemv_if_supported, DecodeProjectionKind,
};
pub(super) use vulkan_gdn_prefill::try_gdn_prefill_ffn_chain_window_if_supported;
#[cfg(feature = "vulkan")]
pub(super) use vulkan_gdn_prefill::try_gdn_qkv_conv_prefill_window;
