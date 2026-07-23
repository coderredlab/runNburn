mod attention;
mod gdn;
mod glm_moe;
mod output;
mod prefill;
mod qwen_moe;

pub use attention::{
    decode_attention_hd256_if_supported, prefill_attention_f16kv_dense_chain_if_supported,
    prefill_attention_f16kv_if_supported, prefill_attention_f16kv_window_dense_chain_if_supported,
    prefill_attention_f16kv_window_if_supported, prefill_attention_hd256_if_supported,
    try_delta_step_if_supported,
};
pub use gdn::{
    begin_nemotron_prefill_workspace, end_nemotron_prefill_workspace,
    ensure_gdn_prefill_chunk_supported, free_delta_state_snapshot, gdn_prefill_chain,
    gdn_prefill_chain_q4k, gdn_prefill_gated_norm_silu, gdn_prefill_gated_norm_silu_project,
    gdn_prefill_quantized_projection, gdn_prefill_quantized_projection_q,
    nemotron_device_route_pack_expert_ids, nemotron_device_route_pack_from_logits,
    nemotron_mamba2_decode_scan, nemotron_mamba2_prefill_device, nemotron_mamba2_prefill_scan,
    nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack, nemotron_reorder_device_route_pack,
    nemotron_router_logits_from_device_f32, release_nemotron_device_route_pack,
    restore_delta_state_cache, snapshot_delta_state_cache, ssm_prefill_conv1d_silu,
    ssm_prefill_delta_net, ssm_prefill_delta_net_resident, ssm_prefill_delta_net_resident_snapshot,
    ssm_prefill_delta_net_resident_snapshots, sync_delta_state_cache,
    try_delta_step_resident_if_supported, DeltaStateSnapshot, GdnPrefillChainOutput,
    GdnPrefillChainQ4KOutput, GdnPrefillChainQ4KRequest, GdnPrefillChainShape,
    NemotronDeviceRoutePack, NemotronDeviceRouterLogitsOutput, NemotronMamba2DeviceOutput,
    NemotronPrefillWorkspaceSummary,
};
pub use glm_moe::{
    glm_moe_decode_shared_expert_q5k_q6k, glm_moe_decode_sparse_experts_iq2xxs_iq3xxs,
    glm_moe_direct_file_prefill_enabled, glm_moe_prefill_shared_expert_iq,
    glm_moe_prefill_sparse_experts_iq_by_token,
};
pub use output::{
    output_logits_enabled, prefill_output_logits, prewarm_output_weight, try_output_argmax_token,
    try_output_logits, try_output_logits_if_enabled, try_output_logits_into_if_enabled,
};
pub use prefill::{
    decode_gemv, decode_gemv_into_if_supported, embedding_gather, prefill_gemv,
    prefill_gemv_enabled, prefill_q4k_f16_gemm_allowed,
    prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain,
    prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output,
    prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain,
    prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output,
    prefill_q4k_f16_qkv_attention_hd512, prefill_q4k_f16_qkv_attention_hd512_dense_chain,
    prefill_q4k_f16_qkv_attention_hd512_dense_chain_device_output, prefill_q4k_f16_qkv_gemm,
    prefill_q4k_f16_qkv_postprocess_hd256,
    prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain,
    prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output,
    PrefillQ4kF16QDenseChainDeviceOutput, PrefillQ4kF16QkvDenseChainDeviceOutput,
};
pub use qwen_moe::{
    log_qwen_moe_prefill_route_hist, qwen35_device_moe_inplace_residual_enabled,
    qwen35_mtp_device_draft_step, qwen35_mtp_device_verify_window,
    qwen35_prefill_attention_device_input, qwen_moe_configure_layer_cache, qwen_moe_decode_down,
    qwen_moe_decode_expert, qwen_moe_decode_gate_up, qwen_moe_decode_shared_sparse_experts_into,
    qwen_moe_decode_sparse_batch_enabled, qwen_moe_decode_sparse_experts,
    qwen_moe_decode_sparse_experts_add_residual_into, qwen_moe_decode_sparse_experts_into,
    qwen_moe_decode_sparse_experts_iq4xs, qwen_moe_decode_sparse_experts_iq4xs_add_residual_into,
    qwen_moe_decode_sparse_experts_iq4xs_into,
    qwen_moe_prefill_combined_f32_shared_sparse_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input,
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual,
    qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token,
    qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token, qwen_moe_prefill_enabled,
    qwen_moe_prefill_enabled_for_seq, qwen_moe_prefill_router_logits,
    qwen_moe_prefill_shared_expert_batch, qwen_moe_prefill_sparse_experts_by_token,
    qwen_moe_prefill_sparse_experts_selected_base_by_token,
    qwen_moe_prefill_sparse_route_arrays_device_topk, qwen_moe_prefill_sparse_slots,
    qwen_moe_prefill_sparse_slots_device_topk, qwen_moe_prefill_sparse_slots_expert_major,
    qwen_moe_register_layer, MtpDeviceDraftRequest, MtpDeviceDraftResult,
    MtpDeviceVerifyAttentionKvState, MtpDeviceVerifyAttentionMoeLayer, MtpDeviceVerifyGdnMoeLayer,
    MtpDeviceVerifyLayerKind, MtpDeviceVerifyPrefixState, MtpDeviceVerifySsmLayerFinalState,
    MtpDeviceVerifySsmLayerPrefixState, MtpDeviceVerifyWindowRequest, MtpDeviceVerifyWindowResult,
    QwenAttentionDeviceInputOutput, QwenMoeDeviceInputOutput,
};

use rnb_loader::GGMLType;

type Result<T> = std::result::Result<T, String>;
use crate::compute as backend;
use crate::gemm::dequant::{self, DequantType};

pub fn q8_0_head_gemv_batch(
    weights: &[u8],
    head_count: usize,
    rows_per_head: usize,
    cols: usize,
    token_count: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::q8_0_head_gemv_batch(weights, head_count, rows_per_head, cols, token_count, input)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_mla_prefill_attention_f16(
    q_absorbed: &[f32],
    q_pe: &[f32],
    cache: &[u16],
    pos_start: usize,
    seq_len: usize,
    num_heads: usize,
    kv_len: usize,
    kv_rank: usize,
    rope_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    backend::glm_mla_prefill_attention_f16(
        q_absorbed, q_pe, cache, pos_start, seq_len, num_heads, kv_len, kv_rank, rope_dim, scale,
    )
}
pub use backend::CudaMemoryInfo;
pub use rnb_backend_api::{
    DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, KvarnDecodeRequest, ScalarType,
};

// cu19: expose CUDA cache hit/miss stats for prefill profiling instrumentation.
pub use backend::Cu71LayerSegmentGraphRuntimeContext;
pub use backend::{cuda_cache_snapshot, CudaCacheSnapshot};

// cu59 axis A — chain function sub-phase timing bridge re-export.
pub use backend::chain_diag_bridge;

// cu41 Phase 1: decode device-resident hidden carrier API wrappers.
pub fn acquire_decode_attn_out_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_attn_out_carrier(bytes)
        .map_err(|err| format!("CUDA acquire_decode_attn_out_carrier failed: {err}"))
}

pub fn acquire_decode_k_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_k_carrier(bytes)
        .map_err(|err| format!("CUDA acquire_decode_k_carrier failed: {err}"))
}

pub fn acquire_decode_v_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_v_carrier(bytes)
        .map_err(|err| format!("CUDA acquire_decode_v_carrier failed: {err}"))
}

pub fn acquire_decode_k_f16_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_k_f16_carrier(bytes)
        .map_err(|err| format!("CUDA acquire_decode_k_f16_carrier failed: {err}"))
}

pub fn acquire_decode_v_f16_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_v_f16_carrier(bytes)
        .map_err(|err| format!("CUDA acquire_decode_v_f16_carrier failed: {err}"))
}

pub fn f32_to_f16_pack_device(src_dev: u64, dst_dev: u64, len: usize) -> Result<()> {
    backend::f32_to_f16_pack_device(src_dev, dst_dev, len)
        .map_err(|err| format!("CUDA f32_to_f16_pack_device failed: {err}"))
}

pub fn acquire_decode_hidden_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_hidden_carrier(bytes)
        .map_err(|err| format!("CUDA acquire decode hidden carrier failed: {err}"))
}

pub fn acquire_decode_norm_buf_carrier(bytes: usize) -> Result<u64> {
    backend::acquire_decode_norm_buf_carrier(bytes)
        .map_err(|err| format!("CUDA acquire decode norm_buf carrier failed: {err}"))
}

pub fn upload_to_decode_hidden_carrier(host: &[f32], dev: u64) -> Result<()> {
    backend::upload_to_decode_hidden_carrier(host, dev)
        .map_err(|err| format!("CUDA upload to decode hidden carrier failed: {err}"))
}

pub fn download_from_decode_hidden_carrier(dev: u64, host: &mut [f32]) -> Result<()> {
    backend::download_from_decode_hidden_carrier(dev, host)
        .map_err(|err| format!("CUDA download from decode hidden carrier failed: {err}"))
}

pub fn sync_decode_stream() -> Result<()> {
    backend::sync_decode_stream().map_err(|err| format!("CUDA sync decode stream failed: {err}"))
}

pub fn cu65_device_qkv_enabled() -> bool {
    backend::cu65_device_qkv_enabled()
}

pub fn cu68_layer_graph_enabled() -> bool {
    backend::cu68_layer_graph_enabled()
}

pub fn cu69_dense_chain_graph_enabled() -> bool {
    backend::cu69_dense_chain_graph_enabled()
}

pub fn cu71_layer_segment_graph_enabled() -> bool {
    backend::cu71_layer_segment_graph_enabled()
}

pub fn cu71_layer_segment_graph_trace_enabled() -> bool {
    backend::cu71_layer_segment_graph_trace_enabled()
}

pub fn persistent_decode_enabled() -> bool {
    backend::persistent_decode_enabled()
}

pub fn dispatch_persistent_decode(
    request: &mut crate::compute::PersistentDecodeRequest<'_>,
) -> Result<()> {
    backend::dispatch_persistent_decode(request)
}

pub fn cu63_device_decode_enabled() -> bool {
    backend::cu63_device_decode_enabled()
}

#[allow(clippy::too_many_arguments)]
pub fn decode_full_layer_device_resident(
    layer_idx: usize,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    n_embd: usize,
    n_ff: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    q_rows: usize,
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    out_scale: f32,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    hidden_dev: u64,
) -> Result<()> {
    backend::decode_full_layer_device_resident(
        layer_idx,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        gate_weights,
        up_weights,
        down_weights,
        attn_norm,
        ffn_norm,
        n_embd,
        n_ff,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        q_rows,
        q_norm_weight,
        k_norm_weight,
        out_scale,
        rope_theta,
        rope_pos,
        kv_len,
        norm_eps,
        hidden_dev,
    )
    .map_err(|err| format!("CUDA decode_full_layer_device_resident failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_device_qkv_rope_kv(
    layer_idx: usize,
    norm_carrier_dev: u64,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    q_rows: usize,
    kv_dim: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    q_host_out: &mut [f32],
    k_host_out: &mut [f32],
    v_host_out: &mut [f32],
) -> Result<u64> {
    backend::decode_device_qkv_rope_kv(
        layer_idx,
        norm_carrier_dev,
        q_weights,
        k_weights,
        v_weights,
        q_norm_weight,
        k_norm_weight,
        q_rows,
        kv_dim,
        n_embd,
        num_heads,
        num_kv_heads,
        rope_theta,
        rope_pos,
        kv_len,
        norm_eps,
        q_host_out,
        k_host_out,
        v_host_out,
    )
    .map_err(|err| format!("CUDA decode_device_qkv_rope_kv failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_device_qkv_rope_kv_graph(
    layer_idx: usize,
    norm_carrier_dev: u64,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    q_rows: usize,
    kv_dim: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    q_host_out: &mut [f32],
    k_host_out: &mut [f32],
    v_host_out: &mut [f32],
) -> Result<u64> {
    backend::decode_device_qkv_rope_kv_graph(
        layer_idx,
        norm_carrier_dev,
        q_weights,
        k_weights,
        v_weights,
        q_norm_weight,
        k_norm_weight,
        q_rows,
        kv_dim,
        n_embd,
        num_heads,
        num_kv_heads,
        rope_theta,
        rope_pos,
        kv_len,
        norm_eps,
        q_host_out,
        k_host_out,
        v_host_out,
    )
    .map_err(|err| format!("CUDA graph QKV failed: {err}"))
}

pub fn launch_attention_decode_device(
    layer_idx: usize,
    q_dev: u64,
    output_dev: u64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<()> {
    backend::launch_attention_decode_device(
        layer_idx,
        q_dev,
        output_dev,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    )
    .map_err(|err| format!("CUDA launch_attention_decode_device failed: {err}"))
}

pub fn launch_attention_decode_device_len_device(
    layer_idx: usize,
    q_dev: u64,
    output_dev: u64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<()> {
    backend::launch_attention_decode_device_len_device(
        layer_idx,
        q_dev,
        output_dev,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    )
    .map_err(|err| format!("CUDA launch_attention_decode_device_len_device failed: {err}"))
}

pub fn populate_device_kv_cache_f16(
    layer_idx: usize,
    k_bits: &[u16],
    v_bits: &[u16],
    kv_dim: usize,
    num_tokens: usize,
) -> Result<()> {
    backend::populate_device_kv_cache_f16(layer_idx, k_bits, v_bits, kv_dim, num_tokens)
        .map_err(|err| format!("CUDA populate_device_kv_cache_f16 failed: {err}"))
}

pub fn device_kv_cache_f16_matches(
    layer_idx: usize,
    kv_dim: usize,
    num_tokens: usize,
) -> Result<bool> {
    backend::device_kv_cache_f16_matches(layer_idx, kv_dim, num_tokens)
        .map_err(|err| format!("CUDA device KV cache query failed: {err}"))
}

pub fn sync_device_kv_cache_f16_to_host(
    layer_idx: usize,
    k_bits: &mut [u16],
    v_bits: &mut [u16],
    kv_dim: usize,
    num_tokens: usize,
) -> Result<bool> {
    backend::sync_device_kv_cache_f16_to_host(layer_idx, k_bits, v_bits, kv_dim, num_tokens)
        .map_err(|err| format!("CUDA device KV cache materialization failed: {err}"))
}

pub fn rms_norm_f32_to_carrier(
    input: &[f32],
    weight: &[f32],
    output_carrier: u64,
    eps: f32,
    unit_offset: bool,
) -> Result<()> {
    backend::rms_norm_f32_to_carrier(input, weight, output_carrier, eps, unit_offset)
        .map_err(|err| format!("CUDA rms_norm_f32_to_carrier failed: {err}"))
}
pub fn rms_norm_rows_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
    unit_offset: bool,
) -> Result<()> {
    backend::rms_norm_rows_f32(input, weight, output, eps, unit_offset)
        .map_err(|err| format!("CUDA batched RMS norm failed: {err}"))
}

pub fn activation_mul_f32_inplace(gate: &mut [f32], up: &[f32], gelu: bool) -> Result<()> {
    backend::activation_mul_f32_inplace(gate, up, gelu)
        .map_err(|err| format!("CUDA activation multiply failed: {err}"))
}

pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) -> Result<()> {
    backend::add_f32_inplace(dst, src).map_err(|err| format!("CUDA residual add failed: {err}"))
}
pub fn add_rows_f32_inplace(dst: &mut [f32], src: &[f32]) -> Result<()> {
    backend::add_rows_f32_inplace(dst, src)
        .map_err(|err| format!("CUDA broadcast add failed: {err}"))
}

pub fn mul_rows_f32_inplace(dst: &mut [f32], src: &[f32]) -> Result<()> {
    backend::mul_rows_f32_inplace(dst, src)
        .map_err(|err| format!("CUDA broadcast multiply failed: {err}"))
}

pub fn scale_f32_inplace(values: &mut [f32], scale: f32) -> Result<()> {
    backend::scale_f32_inplace(values, scale).map_err(|err| format!("CUDA scale failed: {err}"))
}

pub fn sigmoid_f32_inplace(values: &mut [f32]) -> Result<()> {
    backend::sigmoid_f32_inplace(values).map_err(|err| format!("CUDA sigmoid failed: {err}"))
}

pub fn gdn_prepare_delta_gate_beta_f32(
    alpha: &mut [f32],
    beta: &mut [f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    num_heads: usize,
) -> Result<()> {
    backend::gdn_prepare_delta_gate_beta_f32(alpha, beta, dt_bias, ssm_a, num_heads)
        .map_err(|err| format!("CUDA GDN delta gate preparation failed: {err}"))
}

pub fn l2_norm_rows_f32(
    input: &[f32],
    output: &mut [f32],
    row_width: usize,
    eps: f32,
) -> Result<()> {
    backend::l2_norm_rows_f32(input, output, row_width, eps)
        .map_err(|err| format!("CUDA L2 norm failed: {err}"))
}

pub fn axpby_f32_inplace(dst: &mut [f32], src: &[f32], alpha: f32, beta: f32) -> Result<()> {
    backend::axpby_f32_inplace(dst, src, alpha, beta)
        .map_err(|err| format!("CUDA axpby failed: {err}"))
}

pub fn sigmoid_mul_f32_inplace(values: &mut [f32], gate: &[f32]) -> Result<()> {
    backend::sigmoid_mul_f32_inplace(values, gate)
        .map_err(|err| format!("CUDA sigmoid multiply failed: {err}"))
}

pub fn relu_sqr_f32_inplace(values: &mut [f32]) -> Result<()> {
    backend::relu_sqr_f32_inplace(values).map_err(|err| format!("CUDA relu-squared failed: {err}"))
}

pub fn hadamard_f32_inplace(values: &mut [f32], chunk_len: usize) -> Result<()> {
    backend::hadamard_f32_inplace(values, chunk_len)
        .map_err(|err| format!("CUDA Hadamard transform failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_route_topk_f32(
    logits: &[f32],
    selection_bias: Option<&[f32]>,
    seq_len: usize,
    n_expert: usize,
    top_k: usize,
    sigmoid_mode: bool,
    normalize_selected: bool,
    scale: f32,
    adaptive_top_p: Option<f32>,
) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>)> {
    backend::moe_route_topk_f32(
        logits,
        selection_bias,
        seq_len,
        n_expert,
        top_k,
        sigmoid_mode,
        normalize_selected,
        scale,
        adaptive_top_p,
    )
    .map_err(|err| format!("CUDA MoE top-k routing failed: {err}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CudaRopeMode {
    Adjacent = 0,
    Neox = 1,
    PartialAdjacent = 2,
    PartialProportional = 3,
    MropeText = 4,
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32_inplace(
    values: &mut [f32],

    dim: usize,
    head_dim: usize,
    n_rot: usize,
    pos_start: usize,
    theta: f32,
    mode: CudaRopeMode,
    factors: Option<&[f32]>,
) -> Result<()> {
    backend::rope_f32_inplace(
        values,
        dim,
        head_dim,
        n_rot,
        pos_start,
        theta,
        mode as u32,
        factors,
    )
    .map_err(|err| format!("CUDA RoPE failed: {err}"))
}

pub fn rms_norm_f32_dev_input_to_carrier(
    input_dev: u64,
    weight: &[f32],
    output_carrier: u64,
    len: usize,
    eps: f32,
    unit_offset: bool,
) -> Result<()> {
    backend::rms_norm_f32_dev_input_to_carrier(
        input_dev,
        weight,
        output_carrier,
        len,
        eps,
        unit_offset,
    )
    .map_err(|err| format!("CUDA rms_norm_f32_dev_input_to_carrier failed: {err}"))
}

pub fn q4k_gemv_with_device_input(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input_dev: u64,
    output: &mut [f32],
) -> Result<()> {
    backend::q4k_gemv_with_device_input(weights, rows, cols, input_dev, output)
        .map_err(|err| format!("CUDA q4k_gemv_with_device_input failed: {err}"))
}

pub fn q6k_gemv_with_device_input(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input_dev: u64,
    output: &mut [f32],
) -> Result<()> {
    backend::q6k_gemv_with_device_input(weights, rows, cols, input_dev, output)
        .map_err(|err| format!("CUDA q6k_gemv_with_device_input failed: {err}"))
}

// cu47 step 32: attention_decode_cached 의 device output variant. caller 가
// attn_out carrier 제공 시 D2H 없이 device buffer 에 결과 유지.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_to_device(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
) -> Result<()> {
    backend::attention_decode_cached_to_device(
        layer_index,
        q,
        k,
        v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev_target,
        last_token_k_dev,
        last_token_v_dev,
        q_dev_override,
    )
    .map_err(|err| format!("CUDA attention_decode_cached_to_device failed: {err}"))
}

pub fn kvarn_attention_enabled(
    device_output: bool,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> bool {
    crate::policy::kvarn_attention_enabled(
        crate::policy::KvarnAttentionBackend::Cuda,
        device_output,
        kv_len,
        num_heads,
        head_dim,
    )
}

pub fn attention_decode_kvarn(request: KvarnDecodeRequest<'_>) -> Result<Vec<f32>> {
    backend::attention_decode_kvarn(request)
        .map_err(|err| format!("CUDA attention_decode_kvarn failed: {err}"))
}

pub fn attention_decode_kvarn_to_device(
    request: KvarnDecodeRequest<'_>,
    output_dev_target: u64,
) -> Result<()> {
    backend::attention_decode_kvarn_to_device(request, output_dev_target)
        .map_err(|err| format!("CUDA attention_decode_kvarn_to_device failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_to_device_len_device(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
) -> Result<()> {
    backend::attention_decode_cached_to_device_len_device(
        layer_index,
        q,
        k,
        v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev_target,
        last_token_k_dev,
        last_token_v_dev,
        q_dev_override,
    )
    .map_err(|err| format!("CUDA attention_decode_cached_to_device_len_device failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_to_device_len_device_graph(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
) -> Result<()> {
    backend::attention_decode_cached_to_device_len_device_graph(
        layer_index,
        q,
        k,
        v,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        output_dev_target,
        last_token_k_dev,
        last_token_v_dev,
        q_dev_override,
    )
    .map_err(|err| format!("CUDA attention_decode_cached_to_device_len_device_graph failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv_with_device_input(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    input_dev: u64,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> Result<()> {
    backend::dense_q4k_attention_qkv_with_device_input(
        q_weights, k_weights, v_weights, q_rows, kv_rows, n_embd, input_dev, q, k, v,
    )
    .map_err(|err| format!("CUDA dense_q4k_attention_qkv_with_device_input failed: {err}"))
}

fn dequant_type(ggml_type: GGMLType) -> DequantType {
    match ggml_type {
        GGMLType::F32 => DequantType::F32,
        GGMLType::F16 => DequantType::F16,
        GGMLType::BF16 => DequantType::BF16,
        GGMLType::Q4_0 => DequantType::Q4_0,
        GGMLType::Q4_1 => DequantType::Q4_1,
        GGMLType::Q5_0 => DequantType::Q5_0,
        GGMLType::Q5_1 => DequantType::Q5_1,
        GGMLType::Q8_0 => DequantType::Q8_0,
        GGMLType::Q8_1 => DequantType::Q8_1,
        GGMLType::Q2_K => DequantType::Q2K,
        GGMLType::Q3_K => DequantType::Q3K,
        GGMLType::Q4_K => DequantType::Q4K,
        GGMLType::Q5_K => DequantType::Q5K,
        GGMLType::Q6_K => DequantType::Q6K,
        GGMLType::Q8_K => DequantType::Q8K,
        GGMLType::IQ2_XXS => DequantType::IQ2XXS,
        GGMLType::IQ2_XS => DequantType::IQ2XS,
        GGMLType::IQ1_S => DequantType::IQ1S,
        GGMLType::IQ4_NL => DequantType::IQ4NL,
        GGMLType::IQ3_S => DequantType::IQ3S,
        GGMLType::IQ3_XXS => DequantType::IQ3XXS,
        GGMLType::IQ2_S => DequantType::IQ2S,
        GGMLType::IQ4_XS => DequantType::IQ4XS,
        GGMLType::IQ1_M => DequantType::IQ1M,
        GGMLType::TQ1_0 => DequantType::TQ1_0,
        GGMLType::TQ2_0 => DequantType::TQ2_0,
        GGMLType::MXFP4 => DequantType::MXFP4,
        GGMLType::NVFP4 => DequantType::NVFP4,
        GGMLType::Q1_0 => DequantType::Q1_0,
        GGMLType::Q2_0 => DequantType::Q2_0,
        GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 | GGMLType::F64 => {
            panic!("{ggml_type:?} GGUF tensors cannot be dequantized as model weights")
        }
    }
}

pub fn reset_state_for_engine_init() -> Result<()> {
    backend::clear_moe_layer_cache().map_err(|err| err)?;
    backend::reset_delta_state_cache().map_err(|err| err)
}

pub fn cuda_memory_info() -> Result<CudaMemoryInfo> {
    backend::cuda_memory_info()
}

pub fn clear_moe_layer_cache() -> Result<()> {
    backend::clear_moe_layer_cache().map_err(|err| err)
}

pub fn clear_q4k_cache() -> Result<()> {
    backend::clear_q4k_cache().map_err(|err| err)
}

pub fn clear_q4_f32_cache() -> Result<()> {
    backend::clear_q4_f32_cache().map_err(|err| err)
}

pub fn clear_decode_attention_kv_cache() -> Result<()> {
    backend::clear_decode_attention_kv_cache().map_err(|err| err)
}

pub fn clear_host_registered_ranges() -> Result<()> {
    backend::clear_host_registered_ranges().map_err(|err| err)
}

pub fn clear_sequence_state_cache() -> Result<()> {
    backend::clear_sequence_state_cache().map_err(|err| err)
}

pub fn release_q4_f32_after_prefill() -> Result<()> {
    backend::release_q4_f32_after_prefill().map_err(|err| err)
}

pub fn release_q8_0_prefill_f32_after_prefill() -> Result<()> {
    backend::release_q8_0_prefill_f32_after_prefill().map_err(|err| err)
}

pub fn upload_device_tensor_f32(
    desc: rnb_backend_api::DeviceTensorDesc,
    input: &[f32],
) -> Result<rnb_backend_api::DeviceTensorId> {
    backend::upload_device_tensor_f32(desc, input)
        .map_err(|err| format!("CUDA device tensor upload failed: {err}"))
}

pub fn download_device_tensor_f32(id: rnb_backend_api::DeviceTensorId) -> Result<Vec<f32>> {
    backend::download_device_tensor_f32(id)
        .map_err(|err| format!("CUDA device tensor download failed: {err}"))
}

pub fn download_device_tensor_f32_row(
    id: rnb_backend_api::DeviceTensorId,
    row: usize,
) -> Result<Vec<f32>> {
    backend::download_device_tensor_f32_row(id, row)
        .map_err(|err| format!("CUDA device tensor row download failed: {err}"))
}

pub fn release_device_tensor(id: rnb_backend_api::DeviceTensorId) -> Result<bool> {
    backend::release_device_tensor(id)
        .map_err(|err| format!("CUDA device tensor release failed: {err}"))
}

pub fn nemotron_q5_full_layer_enabled() -> bool {
    backend::tuning::nemotron_q5_full_layer_enabled()
}

pub fn nemotron_q5_layer_cache_enabled() -> bool {
    backend::tuning::nemotron_q5_layer_cache_enabled()
}

pub fn nemotron_q5_q8_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool> {
    backend::nemotron_q5_q8_register_layer(up_all, down_all, n_expert, n_ff, n_embd)
        .map_err(|err| format!("CUDA Nemotron Q5/Q8 layer registration failed: {err}"))
}

pub fn nemotron_q8_shared_q5_sparse_decode_enabled() -> bool {
    backend::tuning::nemotron_q8_shared_q5_sparse_decode_enabled()
}

pub fn nemotron_prefill_sparse_copy_prefetch(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    n_ff: usize,
    n_embd: usize,
    down_q8: bool,
) -> Result<bool> {
    backend::nemotron_prefill_sparse_copy_prefetch(up_weights, down_weights, n_ff, n_embd, down_q8)
        .map_err(|err| format!("CUDA Nemotron prefill sparse copy prefetch failed: {err}"))
}

pub fn prewarm_q4k_weight_slices(weights: &[&[u8]]) -> Result<usize> {
    backend::prewarm_q4k_weights(weights)
}

pub fn prewarm_q4k_weight_slices_pinned(weights: &[&[u8]]) -> Result<usize> {
    backend::prewarm_q4k_weights_pinned(weights)
}

pub fn prewarm_quant_resident_q4k_weights(weights: &[&[u8]]) -> Result<usize> {
    backend::prewarm_quant_resident_q4k_weights(weights)
}

pub fn prewarm_q4k_packed_gate_up_weights(
    weights: &[(&[u8], &[u8], usize, usize)],
) -> Result<usize> {
    backend::prewarm_q4k_packed_gate_up_weights(weights)
}

pub fn prewarm_q4k_f32_gate_up_weights(weights: &[(&[u8], &[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q4k_f32_gate_up_weights(weights)
}

pub fn prewarm_q4k_f32_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q4k_f32_weights(weights)
}

pub fn prewarm_q4k_f16_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q4k_f16_weights(weights)
}

pub fn prewarm_q4k_prefill_f16_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q4k_prefill_f16_weights(weights)
}

pub fn prewarm_q4k_packed_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q4k_packed_weights(weights)
}

pub fn prewarm_q6k_packed_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q6k_packed_weights(weights)
}

pub fn prewarm_q6k_f32_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize> {
    backend::prewarm_q6k_f32_weights(weights)
}

pub fn upload_gemma_ple_base(data: &[f32]) -> Result<()> {
    backend::upload_gemma_ple_base(data)
        .map_err(|err| format!("CUDA Gemma PLE base upload failed: {err}"))
}

pub fn q2k_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q2k_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA Q2_K GEMV failed: {err}"))
}

pub fn q3k_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q3k_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA Q3_K GEMV failed: {err}"))
}

pub fn q4k_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q4k_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA Q4_K GEMV failed: {err}"))
}

pub fn q5_0_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q5_0_gemv(raw, rows, cols, input)
        .map_err(|err| format!("CUDA Q5_0 GEMV failed: {err}"))
}

pub fn bf16_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::bf16_gemv(raw, rows, cols, input)
        .map_err(|err| format!("CUDA BF16 GEMV failed: {err}"))
}

pub fn f16_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::f16_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA F16 GEMV failed: {err}"))
}

pub fn q5_1_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q5_1_gemv(raw, rows, cols, input)
        .map_err(|err| format!("CUDA Q5_1 GEMV failed: {err}"))
}

pub fn q5k_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q5k_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA Q5_K GEMV failed: {err}"))
}

pub fn q6k_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q6k_gemv(raw, rows, cols, input).map_err(|err| format!("CUDA Q6_K GEMV failed: {err}"))
}

pub fn q8_0_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::q8_0_gemv(raw, rows, cols, input)
        .map_err(|err| format!("CUDA Q8_0 GEMV failed: {err}"))
}

pub fn iq4_xs_gemv(raw: &[u8], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    backend::iq4_xs_gemv(raw, rows, cols, input)
        .map_err(|err| format!("CUDA IQ4_XS GEMV failed: {err}"))
}

pub fn dense_q4k_gelu_ffn(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::dense_q4k_gelu_ffn(gate, up, down, down_quant as u32, n_ff, n_embd, input)
        .map_err(|err| format!("CUDA dense Q4_K GELU FFN failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_gelu_ffn_batch(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::dense_q4k_gelu_ffn_batch(
        gate,
        up,
        down,
        down_quant as u32,
        n_ff,
        n_embd,
        seq_len,
        input,
    )
    .map_err(|err| format!("CUDA dense Q4_K GELU FFN batch failed: {err}"))
}

// cu46: drafter SwiGLU FFN port — silu_mul activation 의 batch variant.
// drafter forward = single-token (seq_len=1) 라 batch=1 의 silu_ffn_batch.
#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_silu_ffn_batch(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::dense_q4k_silu_ffn_batch(
        gate,
        up,
        down,
        down_quant as u32,
        n_ff,
        n_embd,
        seq_len,
        input,
    )
    .map_err(|err| format!("CUDA dense Q4_K SiLU FFN batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_gelu_ffn_norm_residual(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    norm_weight: &[f32],
    post_norm_weight: Option<&[f32]>,
    n_ff: usize,
    n_embd: usize,
    hidden: &[f32],
    norm_eps: f32,
    unit_offset_norm: bool,
) -> Result<Vec<f32>> {
    backend::dense_q4k_gelu_ffn_norm_residual(
        gate,
        up,
        down,
        down_quant as u32,
        norm_weight,
        post_norm_weight,
        n_ff,
        n_embd,
        hidden,
        norm_eps,
        unit_offset_norm,
    )
    .map_err(|err| format!("CUDA dense Q4_K GELU FFN norm residual failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_output_gelu_ffn_norm_residual(
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_input_device_offset: Option<usize>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_ple_norm: bool,
    hidden_carrier_dev: Option<u64>,
    skip_h2d_hidden: bool,
    skip_d2h_hidden: bool,
    layer_output_scale: Option<f32>,
    attn_out_dev_carrier: Option<u64>,
    ffn_uses_gelu: bool,
    dense_chain_graph_allowed: bool,
    layer_segment_graph_context: Option<Cu71LayerSegmentGraphRuntimeContext>,
) -> Result<()> {
    backend::dense_q4k_attention_output_gelu_ffn_norm_residual(
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_input_device_offset,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        attn_out,
        norm_eps,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_ple_norm,
        hidden_carrier_dev,
        skip_h2d_hidden,
        skip_d2h_hidden,
        layer_output_scale,
        attn_out_dev_carrier,
        ffn_uses_gelu,
        dense_chain_graph_allowed,
        layer_segment_graph_context,
    )
    .map_err(|err| format!("CUDA dense Q4_K attention+FFN failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<()> {
    backend::dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        o_cols,
        n_ff,
        n_embd,
        seq_len,
        hidden,
        attn_out,
        norm_eps,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map_err(|err| format!("CUDA dense Q4_K attention+FFN batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_ple_q4k_batch_norm_residual(
    gate: &[u8],
    proj: &[u8],
    post_norm_weight: &[f32],
    out_scale: Option<&[f32]>,
    ple_input: &[f32],
    ple_dim: usize,
    n_embd: usize,
    seq_len: usize,
    hidden: &mut [f32],
    norm_eps: f32,
) -> Result<()> {
    backend::gemma4_ple_q4k_batch_norm_residual(
        gate,
        proj,
        post_norm_weight,
        out_scale,
        ple_input,
        ple_dim,
        n_embd,
        seq_len,
        hidden,
        norm_eps,
    )
    .map_err(|err| format!("CUDA Gemma4 PLE Q4_K batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    input: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<()> {
    backend::dense_q4k_attention_qkv(q, k, v, q_rows, kv_rows, n_embd, input, q_out, k_out, v_out)
        .map_err(|err| format!("CUDA dense Q4_K QKV failed: {err}"))
}

// cu29 Phase 2: Llama / Mistral hd=128 decode path 용 fused Q4K QKV + GPU RoPE +
// f16 K/V pack. host RoPE round-trip 제거.
#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv_rope_hd128_decode(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    input: &[f32],
    q_rope_out: &mut [f32],
    k_bits_out: &mut [u16],
    v_bits_out: &mut [u16],
) -> Result<()> {
    backend::dense_q4k_attention_qkv_rope_hd128_decode(
        q,
        k,
        v,
        q_rows,
        kv_rows,
        n_embd,
        num_heads,
        num_kv_heads,
        rope_theta,
        pos_start,
        input,
        q_rope_out,
        k_bits_out,
        v_bits_out,
    )
    .map_err(|err| format!("CUDA dense Q4_K QKV+RoPE hd128 failed: {err}"))
}

// cu30 Phase 2c: multi-token (prefill).
#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv_rope_hd128_prefill(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    seq_len: usize,
    input: &[f32],
    q_rope_out: &mut [f32],
    k_bits_out: &mut [u16],
    v_bits_out: &mut [u16],
) -> Result<()> {
    backend::dense_q4k_attention_qkv_rope_hd128_prefill(
        q,
        k,
        v,
        q_rows,
        kv_rows,
        n_embd,
        num_heads,
        num_kv_heads,
        rope_theta,
        pos_start,
        seq_len,
        input,
        q_rope_out,
        k_bits_out,
        v_bits_out,
    )
    .map_err(|err| format!("CUDA dense Q4_K QKV+RoPE hd128 prefill failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::nemotron_q5_sparse_relu_sqr_by_token(
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q5 sparse batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_q8_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::nemotron_q5_q8_sparse_relu_sqr_by_token(
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        token_count,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q5/Q8 sparse batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q5/Q8 cached-layer sparse batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_decode_moe_shared_sparse(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::nemotron_q5_decode_moe_shared_sparse(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q5 decode shared+sparse failed: {err}"))
}

pub fn nemotron_q8_shared_prefill(
    shared_up: &[u8],
    shared_down: &[u8],
    shared_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q8_shared_prefill(
        shared_up,
        shared_down,
        shared_ff,
        n_embd,
        token_count,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q8 shared prefill failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q8_shared_q5_sparse_prefill_moe(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input,
        residual,
    )
    .map_err(|err| format!("CUDA Nemotron Q8 shared + Q5 sparse prefill MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
) -> Result<Option<rnb_backend_api::DeviceTensorId>> {
    backend::nemotron_q8_shared_q5_sparse_prefill_moe_device(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
    )
    .map_err(|err| format!("CUDA Nemotron device Q8 shared + Q5 sparse prefill MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
) -> Result<Option<rnb_backend_api::DeviceTensorId>> {
    backend::nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
        residual_desc,
    )
    .map_err(|err| format!("CUDA Nemotron device Q8 shared + Q5 sparse prefill MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device_output(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
) -> Result<Option<rnb_backend_api::DeviceTensorId>> {
    nemotron_q8_shared_q5_sparse_prefill_moe_device(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        token_ids,
        shared_ff,
        n_ff,
        n_embd,
        token_count,
        input_id,
        residual_id,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
        shared_up,
        shared_down,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        shared_ff,
        n_expert,
        n_ff,
        n_embd,
        token_count,
        input,
        residual,
    )
    .map_err(|err| format!("CUDA cached Nemotron Q8 shared + Q5 sparse prefill MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_decode_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::nemotron_q8_shared_q5_sparse_decode_moe(
        shared_up,
        shared_down,
        up_weights,
        down_weights,
        route_weights,
        shared_ff,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron Q8 shared + Q5 sparse decode MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
        shared_up,
        shared_down,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        shared_ff,
        n_expert,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron cached Q8 shared + Q5 sparse decode MoE failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_full_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::nemotron_q5_sparse_relu_sqr_full_layer_by_token(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron full-layer Q5 sparse batch failed: {err}"))
}

pub fn nemotron_q5_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool> {
    backend::nemotron_q5_register_layer(up_all, down_all, n_expert, n_ff, n_embd)
        .map_err(|err| format!("CUDA Nemotron Q5 layer registration failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>> {
    backend::nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )
    .map_err(|err| format!("CUDA Nemotron cached-layer Q5 sparse batch failed: {err}"))
}
