use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOp, BackendOutput,
    BackendRequest, BackendResult,
};

pub mod runtime;
pub mod tuning;

mod dynlib;

// cu19: expose cache hit/miss stats for prefill profile instrumentation.
pub use runtime::{
    cuda_cache_snapshot, cuda_weight_residency_counters, CudaCacheSnapshot,
    CudaWeightResidencyCounters,
};

// cu41 Phase 1: decode device-resident hidden carrier API.
// cu42 step 9: q4k_gemv_with_device_input 추가 (reuse_q_only path).
// cu42 step 11: rms_norm_f32_dev_input_to_carrier (carrier→carrier chain).
// cu42 step 12: acquire_decode_norm_buf_carrier (norm_buf 전용 carrier).
pub use runtime::{
    acquire_decode_attn_out_carrier, acquire_decode_hidden_carrier, acquire_decode_k_carrier,
    acquire_decode_k_f16_carrier, acquire_decode_norm_buf_carrier, acquire_decode_v_carrier,
    acquire_decode_v_f16_carrier, dense_q4k_attention_qkv_with_device_input,
    download_from_decode_hidden_carrier, f32_to_f16_pack_device, q4k_gemv_with_device_input,
    q6k_gemv_with_device_input, rms_norm_f32_dev_input_to_carrier, rms_norm_f32_to_carrier,
    rms_norm_rows_f32, sync_decode_stream, upload_to_decode_hidden_carrier,
};
// cu63: device-resident decode full-layer API.
pub use runtime::{
    cu63_device_decode_enabled, cu65_device_qkv_enabled, cu68_layer_graph_enabled,
    cu69_dense_chain_graph_enabled, cu71_layer_segment_graph_enabled,
    cu71_layer_segment_graph_trace_enabled, decode_device_qkv_rope_kv,
    decode_device_qkv_rope_kv_graph, decode_full_layer_device_resident,
    device_kv_cache_f16_matches, dispatch_persistent_decode, launch_attention_decode_device,
    launch_attention_decode_device_len_device, persistent_decode_enabled,
    populate_device_kv_cache_f16, sync_device_kv_cache_f16_to_host,
    Cu71LayerSegmentGraphRuntimeContext,
};

pub use runtime::{
    activation_mul_f32_inplace, add_f32_inplace, add_rows_f32_inplace, attention_decode_cached,
    attention_decode_cached_to_device, attention_decode_cached_to_device_len_device,
    attention_decode_cached_to_device_len_device_graph, attention_decode_cached_window,
    attention_decode_hd128, attention_decode_hd256, attention_decode_hd512,
    attention_decode_hd512_len_device, attention_decode_kvarn, attention_decode_kvarn_to_device,
    attention_prefill_flash_f32, attention_prefill_flash_hd128, attention_prefill_flash_hd256,
    attention_prefill_flash_hd256_f16kv_window,
    attention_prefill_flash_hd256_f16kv_window_dense_chain, attention_prefill_flash_hd512,
    attention_prefill_flash_hd512_f16kv, attention_prefill_flash_hd512_f16kv_dense_chain,
    attention_prefill_flash_hd512_f16kv_window,
    attention_prefill_flash_hd512_f16kv_window_dense_chain, axpby_f32_inplace,
    begin_nemotron_prefill_workspace, bf16_gemv, bf16_gemv_batch,
    build_gdn_prefill_chain_conv_input_for_test, clear_decode_attention_kv_cache,
    clear_host_registered_ranges, clear_moe_layer_cache, clear_q4_f32_cache, clear_q4k_cache,
    clear_sequence_state_cache, cuda_memory_info, delta_net_decode, delta_net_decode_resident,
    delta_net_prefill, delta_net_prefill_resident, delta_net_prefill_resident_snapshot,
    delta_net_prefill_resident_snapshots, dense_q4k_attention_output_gelu_ffn_batch_norm_residual,
    dense_q4k_attention_output_gelu_ffn_norm_residual, dense_q4k_attention_qkv,
    dense_q4k_attention_qkv_rope_hd128_decode, dense_q4k_attention_qkv_rope_hd128_prefill,
    dense_q4k_gelu_ffn, dense_q4k_gelu_ffn_batch, dense_q4k_gelu_ffn_norm_residual,
    dense_q4k_silu_ffn_batch, derive_gdn_prefill_chain_dims, end_nemotron_prefill_workspace,
    f16_gemv, f16_gemv_batch, f32_gemm_batch, f32_shared_expert, free_delta_state_snapshot,
    gdn_gated_norm_silu, gdn_gated_norm_silu_f32_gemm,
    gdn_prefill_chain_conv_state_after_prefix_for_test, gdn_prefill_chain_q4k,
    gdn_prepare_delta_gate_beta_f32, gemma4_ple_q4k_batch_norm_residual,
    glm_mla_prefill_attention_f16, glm_shared_expert_iq, glm_sparse_experts_iq2xxs_iq3xxs,
    glm_sparse_experts_iq_by_token, hadamard_f32_inplace, iq2_s_gemv, iq2_s_gemv_batch,
    iq2_xxs_gemv, iq2_xxs_gemv_batch, iq3_xxs_gemv, iq3_xxs_gemv_batch, iq4_xs_gemv,
    iq4_xs_gemv_batch, l2_norm_rows_f32, moe_route_topk_f32, mul_rows_f32_inplace,
    nemotron_device_route_pack_expert_ids, nemotron_device_route_pack_from_logits,
    nemotron_mamba2_decode_scan, nemotron_mamba2_prefill_device, nemotron_mamba2_prefill_scan,
    nemotron_prefill_sparse_copy_prefetch, nemotron_q5_decode_moe_shared_sparse,
    nemotron_q5_q8_register_layer, nemotron_q5_q8_sparse_relu_sqr_by_token,
    nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token, nemotron_q5_register_layer,
    nemotron_q5_sparse_relu_sqr_by_token, nemotron_q5_sparse_relu_sqr_cached_layer_by_token,
    nemotron_q5_sparse_relu_sqr_full_layer_by_token, nemotron_q8_shared_prefill,
    nemotron_q8_shared_q5_sparse_decode_moe, nemotron_q8_shared_q5_sparse_decode_moe_cached_layer,
    nemotron_q8_shared_q5_sparse_prefill_moe,
    nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc,
    nemotron_reorder_device_route_pack, nemotron_router_logits_from_device_f32,
    plan_gdn_prefill_chain, plan_gdn_prefill_chain_for_test, prewarm_q4k_f16_weights,
    prewarm_q4k_f32_gate_up_weights, prewarm_q4k_f32_weights, prewarm_q4k_packed_gate_up_weights,
    prewarm_q4k_packed_weights, prewarm_q4k_prefill_f16_weights, prewarm_q4k_weights,
    prewarm_q4k_weights_pinned, prewarm_q6k_f32_weights, prewarm_q6k_packed_weights,
    prewarm_q8_0_weight, prewarm_quant_resident_q4k_weights, q2k_gemv, q2k_gemv_batch, q3k_gemv,
    q3k_gemv_batch, q4_0_gemv, q4_0_gemv_batch, q4_1_gemv, q4_1_gemv_batch, q4k_f16_gemm_batch,
    q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain,
    q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain_device_output,
    q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain,
    q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain_device_output,
    q4k_f16_qkv_gemm_batch, q4k_f16_qkv_postprocess_hd256,
    q4k_f16_qkv_postprocess_hd256_window_dense_chain,
    q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output,
    q4k_f16_qkv_prefill_attention_hd512, q4k_f16_qkv_prefill_attention_hd512_dense_chain,
    q4k_f16_qkv_prefill_attention_hd512_dense_chain_device_output, q4k_f32_gemm_batch_cached,
    q4k_gemv, q4k_gemv_batch, q4k_gemv_into, q4k_gemv_into_touch_hit, q5_0_gemv, q5_0_gemv_batch,
    q5_1_gemv, q5_1_gemv_batch, q5k_gemv, q5k_gemv_batch, q5k_gemv_into, q6k_gemv, q6k_gemv_argmax,
    q6k_gemv_batch, q6k_gemv_into, q6k_gemv_into_touch_hit, q8_0_f32_gemm_batch_cached, q8_0_gemv,
    q8_0_gemv_argmax, q8_0_gemv_argmax_q8dot, q8_0_gemv_batch, q8_0_head_gemv_batch, q8_1_gemv,
    q8_1_gemv_batch, quant_embedding_gather, qwen35_configure_moe_layer_cache,
    qwen35_decode_moe_shared_sparse_into, qwen35_expert, qwen35_mtp_device_draft_step,
    qwen35_mtp_device_verify_window, qwen35_mtp_verify_buffer_plan,
    qwen35_prefill_attention_device_input, qwen35_prefill_device_topk_route_slots,
    qwen35_prefill_moe_f32_shared_sparse_by_token,
    qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_by_token,
    qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input,
    qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual,
    qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token,
    qwen35_prefill_moe_f32_shared_sparse_selected_base_by_token,
    qwen35_prefill_moe_q4_shared_sparse_by_token_cached,
    qwen35_prefill_moe_q4_shared_sparse_device_topk_selected_base_by_token_cached,
    qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached,
    qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached,
    qwen35_prefill_moe_quant_shared_sparse_device_topk_selected_base_device_input,
    qwen35_prepare_selected_bundle_residency, qwen35_register_moe_layer, qwen35_sparse_experts,
    qwen35_sparse_experts_add_residual_into, qwen35_sparse_experts_by_token,
    qwen35_sparse_experts_device_roundtrip, qwen35_sparse_experts_into,
    qwen35_sparse_experts_iq4xs, qwen35_sparse_experts_iq4xs_add_residual_into,
    qwen35_sparse_experts_iq4xs_into, qwen35_sparse_experts_per_slot_resident,
    qwen35_sparse_experts_selected_base_by_token, release_nemotron_device_route_pack,
    release_q4_f32_after_prefill, release_q8_0_prefill_f32_after_prefill, relu_sqr_f32_inplace,
    reset_delta_state_cache, restore_delta_state_cache, rope_f32_inplace, scale_f32_inplace,
    sigmoid_f32_inplace, sigmoid_mul_f32_inplace, snapshot_delta_state_cache, ssm_conv1d_silu,
    sync_delta_state_cache, upload_gemma_ple_base, validate_gdn_prefill_chain_shape,
    CudaMemoryInfo, CudaMoeJitLoader, DeltaStateSnapshot, GdnPrefillChainDims, GdnPrefillChainPlan,
    GdnPrefillChainQ4KOutput, GdnPrefillChainQ4KRequest, GdnPrefillChainShape, MtpVerifyBufferPlan,
    NemotronDeviceRoutePack, NemotronDeviceRouterLogitsOutput, NemotronMamba2DeviceOutput,
    NemotronPrefillWorkspaceConfig, NemotronPrefillWorkspaceSummary, Qwen35MtpDeviceDraftRequest,
    Qwen35MtpDeviceDraftResult, Qwen35MtpDeviceVerifyAttentionKvState,
    Qwen35MtpDeviceVerifyAttentionMoeLayer, Qwen35MtpDeviceVerifyGdnMoeLayer,
    Qwen35MtpDeviceVerifyLayerKind, Qwen35MtpDeviceVerifyPrefixState, Qwen35MtpDeviceVerifyRequest,
    Qwen35MtpDeviceVerifyResult, Qwen35MtpDeviceVerifySsmLayerFinalState,
    Qwen35MtpDeviceVerifySsmLayerPrefixState, Qwen35PrefillAttentionDeviceOutput,
};

#[cfg(test)]
pub use runtime::{
    iq4_xs_gemv_batch_for_test, iq4_xs_gemv_for_test, launch_smoke_add_one_for_test,
    launch_smoke_graph_add_one_for_test, q4k_block_dot_for_test, q4k_gemv_for_test,
    q4k_row_dot_for_test, q6k_gemv_for_test,
};

pub fn gdn_prefill_chunk_unimplemented_for_test(
    seq_len: usize,
    hidden_dim: usize,
) -> Result<(), String> {
    Err(format!(
        "CUDA GDN prefill chunk path is not implemented yet: seq_len={seq_len} hidden_dim={hidden_dim}"
    ))
}

#[derive(Debug, Default)]
pub struct CudaBackend;

impl CudaBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::Cuda)
            .with_op(BackendOp::MatMul)
            .with_op(BackendOp::Attention)
            .with_op(BackendOp::Gdn)
            .with_op(BackendOp::MoE)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        Err(BackendError::unsupported(self.kind(), request.op()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_declares_gpu_ops_but_does_not_fake_execution() {
        let mut backend = CudaBackend::new();
        let caps = backend.capabilities();

        assert!(caps.supports(BackendOp::MatMul));
        assert!(matches!(
            backend.execute(BackendRequest::new(BackendOp::MatMul)),
            Err(err) if err.backend() == BackendKind::Cuda && err.op() == Some(BackendOp::MatMul)
        ));
    }
}
