pub mod backend;
pub mod device_prefill;

#[cfg(feature = "cuda")]
pub mod compute {
    pub use rnb_backend_cuda::runtime::{
        download_device_tensor_f32, download_device_tensor_f32_row,
        nemotron_q8_shared_q5_sparse_prefill_moe_device,
        nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc, release_device_tensor,
        upload_device_tensor_f32,
    };
    pub use rnb_backend_cuda::{cuda_cache_snapshot, CudaCacheSnapshot};
    // cu74: persistent decode (Gemma4 E2B) cooperative kernel dispatch entry.
    pub use rnb_backend_api::{
        PersistentDecodeLayerInput, PersistentDecodeRequest, PERSISTENT_DECODE_FLAG_ATTN_ROT,
        PERSISTENT_DECODE_FLAG_DOWN_Q6K, PERSISTENT_DECODE_FLAG_GATED_ATTN,
        PERSISTENT_DECODE_FLAG_K_Q6K, PERSISTENT_DECODE_FLAG_O_Q6K, PERSISTENT_DECODE_FLAG_PLE_F32,
        PERSISTENT_DECODE_FLAG_REUSE_Q, PERSISTENT_DECODE_FLAG_V_Q6K,
    };
    pub use rnb_backend_cuda::persistent_decode_enabled;
    pub use rnb_backend_cuda::runtime::dispatch_persistent_decode;
    // cu59 axis A — chain function sub-phase timing bridge re-export.
    pub use rnb_backend_cuda::runtime::chain_diag_bridge;
    // cu41 Phase 1: decode device-resident hidden carrier API re-export.
    // cu42 step 9+11+12: q4k_gemv_with_device_input, rms_norm_dev_input, norm_buf carrier.
    pub use rnb_backend_cuda::{
        acquire_decode_attn_out_carrier, acquire_decode_hidden_carrier, acquire_decode_k_carrier,
        acquire_decode_k_f16_carrier, acquire_decode_norm_buf_carrier, acquire_decode_v_carrier,
        acquire_decode_v_f16_carrier, dense_q4k_attention_qkv_with_device_input,
        download_from_decode_hidden_carrier, f32_to_f16_pack_device, q4k_gemv_with_device_input,
        q6k_gemv_with_device_input, rms_norm_f32_dev_input_to_carrier, rms_norm_f32_to_carrier,
        sync_decode_stream, upload_to_decode_hidden_carrier,
    };
    // cu63: device-resident decode full-layer API.
    pub use rnb_backend_cuda::{
        attention_decode_cached, attention_decode_cached_to_device,
        attention_decode_cached_to_device_len_device,
        attention_decode_cached_to_device_len_device_graph, attention_decode_cached_window,
        attention_decode_hd128, attention_decode_hd256, attention_decode_hd512,
        attention_decode_hd512_len_device, attention_prefill_flash_hd128,
        attention_prefill_flash_hd256, attention_prefill_flash_hd256_f16kv_window,
        attention_prefill_flash_hd256_f16kv_window_dense_chain, attention_prefill_flash_hd512,
        attention_prefill_flash_hd512_f16kv, attention_prefill_flash_hd512_f16kv_dense_chain,
        attention_prefill_flash_hd512_f16kv_window,
        attention_prefill_flash_hd512_f16kv_window_dense_chain, bf16_gemv,
        clear_decode_attention_kv_cache, clear_host_registered_ranges, clear_moe_layer_cache,
        clear_q4_f32_cache, clear_q4k_cache, clear_sequence_state_cache, cuda_memory_info,
        delta_net_decode, delta_net_decode_resident, delta_net_prefill, delta_net_prefill_resident,
        delta_net_prefill_resident_snapshot, delta_net_prefill_resident_snapshots,
        dense_q4k_attention_output_gelu_ffn_batch_norm_residual,
        dense_q4k_attention_output_gelu_ffn_norm_residual, dense_q4k_attention_qkv,
        dense_q4k_attention_qkv_rope_hd128_decode, dense_q4k_attention_qkv_rope_hd128_prefill,
        dense_q4k_gelu_ffn, dense_q4k_gelu_ffn_batch, dense_q4k_gelu_ffn_norm_residual,
        dense_q4k_silu_ffn_batch, f16_gemv, f32_gemm_batch, f32_shared_expert,
        free_delta_state_snapshot, gdn_gated_norm_silu, gdn_gated_norm_silu_f32_gemm,
        gdn_prefill_chain_q4k, gdn_prefill_chunk_unimplemented_for_test,
        gemma4_ple_q4k_batch_norm_residual, glm_shared_expert_q5k_q6k,
        glm_sparse_experts_iq2xxs_iq3xxs, glm_sparse_experts_iq2xxs_iq3xxs_by_token, iq4_xs_gemv,
        iq4_xs_gemv_batch, nemotron_device_route_pack_expert_ids,
        nemotron_device_route_pack_from_logits, nemotron_mamba2_decode_scan,
        nemotron_mamba2_prefill_device, nemotron_mamba2_prefill_scan,
        nemotron_prefill_sparse_copy_prefetch, nemotron_q5_decode_moe_shared_sparse,
        nemotron_q5_q8_register_layer, nemotron_q5_q8_sparse_relu_sqr_by_token,
        nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token, nemotron_q5_register_layer,
        nemotron_q5_sparse_relu_sqr_by_token, nemotron_q5_sparse_relu_sqr_cached_layer_by_token,
        nemotron_q5_sparse_relu_sqr_full_layer_by_token, nemotron_q8_shared_prefill,
        nemotron_q8_shared_q5_sparse_decode_moe,
        nemotron_q8_shared_q5_sparse_decode_moe_cached_layer,
        nemotron_q8_shared_q5_sparse_prefill_moe,
        nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer,
        nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack,
        nemotron_reorder_device_route_pack, nemotron_router_logits_from_device_f32,
        prewarm_q4k_f16_weights, prewarm_q4k_f32_gate_up_weights, prewarm_q4k_f32_weights,
        prewarm_q4k_packed_gate_up_weights, prewarm_q4k_packed_weights,
        prewarm_q4k_prefill_f16_weights, prewarm_q4k_sidecar_packed_weights, prewarm_q4k_weights,
        prewarm_q4k_weights_pinned, prewarm_q6k_f32_weights, prewarm_q6k_packed_weights,
        prewarm_q6k_sidecar_packed_weights, prewarm_q8_0_weight,
        prewarm_quant_resident_q4k_weights, q2k_gemv, q3k_gemv, q4k_f16_gemm_batch,
        q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain,
        q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain_device_output,
        q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain,
        q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain_device_output,
        q4k_f16_qkv_gemm_batch, q4k_f16_qkv_postprocess_hd256,
        q4k_f16_qkv_postprocess_hd256_window_dense_chain,
        q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output,
        q4k_f16_qkv_prefill_attention_hd512, q4k_f16_qkv_prefill_attention_hd512_dense_chain,
        q4k_f16_qkv_prefill_attention_hd512_dense_chain_device_output, q4k_f32_gemm_batch_cached,
        q4k_gemv, q4k_gemv_batch, q4k_gemv_into, q4k_gemv_into_touch_hit, q5_0_gemv,
        q5_0_gemv_batch, q5_1_gemv, q5_1_gemv_batch, q5k_gemv, q5k_gemv_batch, q6k_gemv,
        q6k_gemv_argmax, q6k_gemv_batch, q6k_gemv_into, q6k_gemv_into_touch_hit,
        q8_0_f32_gemm_batch_cached, q8_0_gemv, q8_0_gemv_argmax, q8_0_gemv_argmax_q8dot,
        q8_0_gemv_batch, qwen35_decode_moe_shared_sparse_into, qwen35_expert,
        qwen35_mtp_device_verify_window, qwen35_mtp_verify_buffer_plan,
        qwen35_prefill_device_topk_route_slots, qwen35_prefill_moe_f32_shared_sparse_by_token,
        qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_by_token,
        qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input,
        qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual,
        qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token,
        qwen35_prefill_moe_f32_shared_sparse_selected_base_by_token,
        qwen35_prefill_moe_q4_shared_sparse_by_token_cached,
        qwen35_prefill_moe_q4_shared_sparse_device_topk_selected_base_by_token_cached,
        qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached,
        qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached,
        qwen35_prepare_selected_bundle_residency, qwen35_register_moe_layer, qwen35_sparse_experts,
        qwen35_sparse_experts_add_residual_into, qwen35_sparse_experts_by_token,
        qwen35_sparse_experts_device_roundtrip, qwen35_sparse_experts_into,
        qwen35_sparse_experts_iq4xs, qwen35_sparse_experts_iq4xs_add_residual_into,
        qwen35_sparse_experts_iq4xs_into, qwen35_sparse_experts_per_slot_resident,
        qwen35_sparse_experts_selected_base_by_token, release_nemotron_device_route_pack,
        release_q4_f32_after_prefill, release_q8_0_prefill_f32_after_prefill,
        reset_delta_state_cache, restore_delta_state_cache, snapshot_delta_state_cache,
        ssm_conv1d_silu, sync_delta_state_cache, upload_gemma_ple_base, CudaMemoryInfo,
        DeltaStateSnapshot, GdnPrefillChainPlan, GdnPrefillChainQ4KOutput,
        GdnPrefillChainQ4KRequest, GdnPrefillChainShape, MtpVerifyBufferPlan,
        NemotronDeviceRoutePack as BackendNemotronDeviceRoutePack,
        Qwen35MtpDeviceVerifyAttentionKvState, Qwen35MtpDeviceVerifyAttentionMoeLayer,
        Qwen35MtpDeviceVerifyGdnMoeLayer, Qwen35MtpDeviceVerifyLayerKind,
        Qwen35MtpDeviceVerifyPrefixState, Qwen35MtpDeviceVerifyRequest,
        Qwen35MtpDeviceVerifyResult, Qwen35MtpDeviceVerifySsmLayerFinalState,
        Qwen35MtpDeviceVerifySsmLayerPrefixState,
    };
    pub use rnb_backend_cuda::{
        cu63_device_decode_enabled, cu65_device_qkv_enabled, cu68_layer_graph_enabled,
        cu69_dense_chain_graph_enabled, cu71_layer_segment_graph_enabled,
        cu71_layer_segment_graph_trace_enabled, decode_device_qkv_rope_kv,
        decode_device_qkv_rope_kv_graph, decode_full_layer_device_resident,
        launch_attention_decode_device, launch_attention_decode_device_len_device,
        plan_gdn_prefill_chain, populate_device_kv_cache_f16, Cu71LayerSegmentGraphRuntimeContext,
    };

    pub mod tuning {
        pub use rnb_backend_cuda::tuning::{
            decode_attention_enabled, decode_attention_hd512_enabled,
            decode_attention_hd512_split_enabled, decode_attention_kv_cache_enabled,
            decode_attention_sliding_window_enabled, delta_net_enabled,
            delta_state_sync_each_step_enabled, gdn_gated_norm_enabled,
            gdn_gated_norm_gemm_enabled, gdn_gated_norm_gemm_enabled_for_seq,
            gdn_prefill_chain_device_output_enabled, gdn_prefill_chain_enabled,
            gdn_prefill_chain_moe_input_device_enabled,
            gdn_prefill_chain_moe_output_device_enabled,
            gdn_prefill_chain_skip_host_projection_enabled, gdn_prefill_enabled,
            gdn_prefill_gemv_mode, gdn_prefill_gemv_mode_for_seq, layer_gemv_enabled,
            moe_layer_cache_enabled, moe_route_hist_enabled, nemotron_q5_full_layer_enabled,
            nemotron_q5_layer_cache_enabled, nemotron_q8_shared_q5_sparse_decode_enabled,
            output_argmax_enabled, output_logits_enabled, prefill_conv_enabled,
            prefill_delta_enabled, prefill_f32_gemm_allowed, prefill_f32_gemm_trace_enabled,
            prefill_flash_attention_enabled, prefill_flash_attention_min_seq, prefill_gemv_enabled,
            prefill_moe_enabled, prefill_moe_enabled_for_seq, prefill_moe_full_layer_enabled,
            prefill_moe_full_layer_min_expert_permille, prefill_moe_range_slab_enabled,
            prefill_output_logits_requested, prefill_q4k_f16_gemm_enabled,
            prefill_q4k_f16_o_proj_enabled, prefill_q4k_f16_qkv_gemm_enabled,
            prefill_q8_0_batch_enabled, q4k_gemv_warp8_enabled, q8_0_output_q8dot_argmax_enabled,
            qwen35_device_moe_inplace_residual_enabled, qwen_moe_batch_enabled,
            qwen_moe_device_decode_enabled, qwen_moe_gate_up_enabled, shared_f32_enabled,
        };
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_inference;

#[cfg(feature = "metal")]
pub mod metal_inference;

#[cfg(feature = "mediatek")]
pub mod mediatek;

#[cfg(feature = "vulkan")]
pub mod vulkan_inference;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_backend {
    pub use rnb_backend_vulkan::{
        decode_layers_allowed, ffn_down_id, ffn_gate_id, ffn_up_id, gdn_alpha_id, gdn_beta_id,
        gdn_gate_id, gdn_qkv_id, gdn_qkv_prefill_window_chunk, gdn_ssm_out_id,
        ggml_to_output_quant, ggml_to_quant, init_layer_gemv, init_layer_gemv_for_test, k_proj_id,
        max_decode_layer, o_proj_id, output_logits_enabled, output_logits_id,
        prefill_chunk_size_for_active_path, q_proj_id, v_proj_id, verify_attention_layer,
        verify_attention_qkv_layer, verify_gdn_layer, QuantType, WeightId,
    };

    pub type PrefillLayerRuntime = rnb_backend_vulkan::LayerGemv;
    pub type PrefillLayerRuntimeConfig = rnb_backend_vulkan::LayerGemvConfig;
    pub type PrefillLayerRuntimeCounters = rnb_backend_vulkan::RuntimeCounters;
}

pub mod android {
    pub use rnb_platform::android::*;
}

pub mod platform {
    pub use rnb_platform::android::*;
    pub use rnb_platform::*;
}

pub mod memory {
    pub use rnb_memory::*;
}

pub mod packed_weights;
pub mod policy;
pub mod session;

pub mod scheduler {
    pub use rnb_scheduler::request_queue::{
        fair_execution_queue, FairExecutionReceiver, FairExecutionSender, FairExecutionSubmitError,
    };
    pub use rnb_scheduler::{
        force_mobile_vulkan_requested, fullpath_gpu_prefill_requested, plan_moe_jit_load_order,
        plan_slice1_boundary, select_prefill_path, select_prefill_path_for_profile,
        select_runtime_execution_profile, should_attempt_slice1_gpu_prefill,
        ExecutionProfileRequest, PrefillExecutionPath, RuntimeExecutionProfile, Slice1BoundaryPlan,
    };
}

pub mod cpu {
    pub use rnb_cpu::*;
}

pub mod gemm {
    pub use rnb_cpu::gemm::*;
}

pub use backend::{
    compiled_accelerators, default_moe_jit_loader, execute_backend_request, moe_jit_report,
    AttentionRequest, BackendError, BackendErrorKind, BackendKind, BackendOp, BackendOutput,
    BackendRegistry, BackendRequest, BackendResult, BackendWorkload, DecodeWeightKind, GdnRequest,
    GpuBackend, GpuBackendSupport, KvBucketView, MatMulRequest, MoeJitByteRange, MoeJitExpertLoad,
    MoeJitLoadRequest, MoeJitLoadSink, MoeRequest, MoeRouteSlot, QuantFormat, QuantizedWeightView,
    ScalarType, TensorShape, TransformedSourceQuant, TransformedWeightLayout,
    TransformedWeightView, ACCELERATOR_BACKENDS,
};
pub use device_prefill::{
    plan_nemotron_prefill_workspace, DevicePrefillProfile, DevicePrefillSession,
    DevicePrefillSessionConfig, DevicePrefillTransferBreakdown, NemotronPrefillWorkspaceDecision,
    NemotronPrefillWorkspacePlan, NemotronPrefillWorkspaceRequest,
};
pub use policy::moe_section_decode_enabled;
pub use rnb_backend_api::{
    AttentionKvMaterializeRangeRequest, AttentionKvMaterializeRequest, DeviceOpStatus,
    DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, DeviceTransferCounters,
};
pub use rnb_scheduler::{ExecutionProfileRequest, RuntimeExecutionProfile};
pub use session::{
    target_supports_backend, RuntimeConfig, RuntimeError, RuntimeResult, RuntimeSession,
};
