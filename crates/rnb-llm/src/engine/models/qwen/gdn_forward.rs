//! Qwen GDN prefill forward path.

#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
use crate::engine::models::shared_expert_moe::forward_shared_expert_moe;
#[cfg(feature = "cuda")]
use crate::engine::models::shared_expert_moe::{
    qwen35moe_device_input_supported, try_forward_ffn_qwen35moe_device_input,
    try_forward_ffn_qwen35moe_device_input_carrier,
};
#[cfg(feature = "vulkan")]
use crate::engine::DeferredGdnConvStateFlush;
use crate::engine::{
    backend_runtime, kernels, policy, prefill_dual_gemv_q8_or_f32, prefill_gate_up_vectors,
    prefill_gemv_vec, prefill_quantized_input_for_weight, GdnLayerWeights, ModelMetadata,
    QuantizedWeight,
};
use crate::kv_cache::KVCache;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;

fn gpu_quantized_prefill_gemv(
    weight: &QuantizedWeight,
    input: &[f32],
) -> crate::error::Result<Option<Vec<f32>>> {
    backend_runtime::gdn_prefill_quantized_projection(weight, input)
}

pub(in crate::engine) fn debug_gdn_stage_trace_enabled(layer_idx: usize) -> bool {
    if let Ok(selector) = std::env::var("RNB_DEBUG_GDN_STAGE_TRACE_LAYER") {
        return selector.split(',').any(|raw| {
            let item = raw.trim();
            item.eq_ignore_ascii_case("all") || item.parse::<usize>().ok() == Some(layer_idx)
        });
    }
    std::env::var_os("RNB_DEBUG_GDN_STAGE_TRACE").is_some() && layer_idx == 0
}

fn debug_f32_bit_hash(data: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in data {
        hash ^= value.to_bits() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn emit_gdn_stage_trace(tag: &str, layer_idx: usize, data: &[f32], seq_len: usize, width: usize) {
    // pm35 M2: full-f32 dump(scan 누적 correctness 비교용). early-return 앞 — width 무관 전체 write.
    if let Ok(dir) = std::env::var("RNB_DEBUG_GDN_STAGE_DUMP_DIR") {
        let path = format!("{dir}/{tag}_L{layer_idx}.bin");
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let _ = std::fs::write(&path, bytes);
    }
    if seq_len == 0 || width == 0 || data.len() < seq_len * width {
        eprintln!(
            "[gdn-stage-trace][{}] layer={} invalid len={} seq_len={} width={}",
            tag,
            layer_idx,
            data.len(),
            seq_len,
            width
        );
        return;
    }
    let start = (seq_len - 1) * width;
    let row = &data[start..start + width];
    let n = row.len().max(1) as f32;
    let mean = row.iter().sum::<f32>() / n;
    let l2 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let first = &data[..width];
    let mid_start = (seq_len / 2) * width;
    let mid = &data[mid_start..mid_start + width];
    let prev_start = seq_len.saturating_sub(2) * width;
    let prev = &data[prev_start..prev_start + width];
    let first_head = first.iter().take(4).copied().collect::<Vec<_>>();
    let mid_head = mid.iter().take(4).copied().collect::<Vec<_>>();
    let prev_head = prev.iter().take(4).copied().collect::<Vec<_>>();
    let last_head = row.iter().take(4).copied().collect::<Vec<_>>();
    let hash = debug_f32_bit_hash(data);
    eprintln!(
        "[gdn-stage-trace][{}] layer={} hash={:#018x} mean={:.6} l2={:.6} first={:?} mid={:?} prev={:?} last={:?}",
        tag, layer_idx, hash, mean, l2, first_head, mid_head, prev_head, last_head
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_forward_gdn_layer_to_device_output(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: &[f32],
    hidden_device: Option<(cuda_runtime::DeviceTensorId, cuda_runtime::DeviceTensorDesc)>,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    if !rnb_runtime::compute::tuning::gdn_prefill_chain_moe_input_device_enabled()
        || !rnb_runtime::compute::tuning::gdn_prefill_chain_moe_output_device_enabled()
    {
        return Ok(None);
    }
    let Some(moe_w) = w.shared_expert_moe.as_ref() else {
        return Ok(None);
    };
    if !qwen35moe_device_input_supported(moe_w, seq_len) {
        return Ok(None);
    }

    let d_inner = metadata.ssm_d_inner;
    let d_state = metadata.ssm_d_state;
    let n_group = metadata.ssm_n_group;
    let dt_rank = metadata.ssm_dt_rank;
    let conv_kernel = metadata.ssm_conv_kernel;

    backend_runtime::ensure_gdn_prefill_chunk_supported(seq_len, metadata.hidden_dim)?;

    let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
        crate::error::LlmError::Forward(format!("SSM state not initialized for layer {layer_idx}"))
    })?;
    let chain_shape = super::make_gdn_prefill_chain_shape(
        seq_len,
        metadata.hidden_dim,
        d_inner,
        d_state,
        n_group,
        dt_rank,
        conv_kernel,
        ssm_state.conv_state.len(),
        ssm_state.delta_state.len(),
    );
    let attn_norm_data = kernels::tensor_as_f32_slice(&w.attn_norm);
    let conv_kernel_data = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
    let dt_bias_data = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
    let ssm_a_data = kernels::tensor_as_f32_slice(&w.ssm_a);
    let ssm_norm_data = kernels::tensor_as_f32_slice(&w.ssm_norm);

    let Some(mut chain_output) = backend_runtime::gdn_prefill_chain_q4k(
        &chain_shape,
        hidden,
        hidden_device,
        attn_norm_data,
        &w.qkv_weight,
        &w.gate_weight,
        &w.ssm_alpha,
        &w.ssm_beta,
        &mut ssm_state.conv_state,
        conv_kernel_data,
        dt_bias_data,
        ssm_a_data,
        &mut ssm_state.delta_state,
        ssm_norm_data,
        &w.ssm_out,
        kernels::tensor_as_f32_slice(&w.post_attn_norm),
        !rnb_runtime::compute::tuning::gdn_prefill_chain_skip_host_projection_enabled(),
        norm_eps,
    )?
    else {
        return Ok(None);
    };
    chain_output.release_device_output_if_present()?;

    let (Some((residual_id, residual_desc)), Some((moe_input_id, moe_input_desc))) = (
        chain_output.device_residual.as_ref().copied(),
        chain_output.device_moe_input.as_ref().copied(),
    ) else {
        let cleanup = chain_output.release_device_carriers_if_present();
        return match cleanup {
            Ok(()) => Err(crate::error::LlmError::Forward(
                "CUDA Qwen GDN device-input chain did not return MoE carriers".to_string(),
            )),
            Err(cleanup_err) => Err(crate::error::LlmError::Forward(format!(
                "CUDA Qwen GDN device-input chain did not return MoE carriers; cleanup failed: {cleanup_err}"
            ))),
        };
    };

    match try_forward_ffn_qwen35moe_device_input_carrier(
        moe_w,
        seq_len,
        metadata.hidden_dim,
        moe_input_id,
        moe_input_desc,
        residual_id,
        residual_desc,
    ) {
        Ok(Some(output)) => {
            if output.output_id == residual_id {
                chain_output.device_residual = None;
            }
            match chain_output.release_device_carriers_if_present() {
                Ok(()) => Ok(Some(output)),
                Err(cleanup_err) => {
                    let output_cleanup = output.release();
                    Err(match output_cleanup {
                        Ok(true) => crate::error::LlmError::Forward(format!(
                            "CUDA Qwen GDN device carrier cleanup failed: {cleanup_err}"
                        )),
                        Ok(false) => crate::error::LlmError::Forward(format!(
                            "CUDA Qwen GDN device carrier cleanup failed: {cleanup_err}; output cleanup failed: tensor was already missing"
                        )),
                        Err(output_cleanup_err) => crate::error::LlmError::Forward(format!(
                            "CUDA Qwen GDN device carrier cleanup failed: {cleanup_err}; output cleanup failed: {output_cleanup_err}"
                        )),
                    })
                }
            }
        }
        Ok(None) => {
            let cleanup = chain_output.release_device_carriers_if_present();
            match cleanup {
                Ok(()) => Err(crate::error::LlmError::Forward(
                    "CUDA Qwen device-input MoE returned no output after support check"
                        .to_string(),
                )),
                Err(cleanup_err) => Err(crate::error::LlmError::Forward(format!(
                    "CUDA Qwen device-input MoE returned no output after support check; cleanup failed: {cleanup_err}"
                ))),
            }
        }
        Err(err) => {
            let cleanup = chain_output.release_device_carriers_if_present();
            Err(match cleanup {
                Ok(()) => err,
                Err(cleanup_err) => crate::error::LlmError::Forward(format!(
                    "{err}; CUDA Qwen GDN device carriers cleanup failed: {cleanup_err}"
                )),
            })
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_forward_gdn_layer_from_host_to_device_output(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: &Tensor,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    try_forward_gdn_layer_to_device_output(
        kv_cache,
        metadata,
        kernels::tensor_as_f32_slice(hidden),
        None,
        w,
        layer_idx,
        seq_len,
        norm_eps,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn try_forward_gdn_layer_from_device_input(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    input: &backend_runtime::NemotronDeviceLayerOutput,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
) -> crate::error::Result<Option<backend_runtime::NemotronDeviceLayerOutput>> {
    try_forward_gdn_layer_to_device_output(
        kv_cache,
        metadata,
        &[],
        Some((input.output_id, input.output_desc)),
        w,
        layer_idx,
        seq_len,
        norm_eps,
    )
}

pub(in crate::engine) fn forward_gdn_layer_impl(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    mut hidden: Tensor,
    w: &GdnLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    norm_eps: f32,
    mut prefix_collector: Option<&mut crate::engine::verify_window::GdnPrefixStateCollector>,
    #[cfg(feature = "vulkan")] mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
    #[cfg(feature = "vulkan")] mut deferred_gdn_flush: Option<&mut DeferredGdnConvStateFlush>,
) -> crate::error::Result<Tensor> {
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let profiling = policy::profiling_enabled();
    let profile_all_layers = std::env::var_os("RNB_QWEN_PROFILE_ALL_LAYERS").is_some();
    macro_rules! prof {
        ($label:expr, $t:expr) => {
            if profiling && (profile_all_layers || layer_idx == 0) {
                eprintln!(
                    "  [GDN L{}] {:20} {:.1}ms",
                    layer_idx,
                    $label,
                    $t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
    }

    let d_inner = metadata.ssm_d_inner;
    let d_state = metadata.ssm_d_state;
    let n_group = metadata.ssm_n_group;
    let dt_rank = metadata.ssm_dt_rank;
    let conv_kernel = metadata.ssm_conv_kernel;
    let head_v_dim = d_inner / dt_rank;
    let head_k_dim = d_state;
    let num_v_heads = dt_rank;
    let num_k_heads = n_group;
    let conv_channels = d_inner + 2 * n_group * d_state;

    backend_runtime::ensure_gdn_prefill_chunk_supported(seq_len, metadata.hidden_dim)?;

    let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
        crate::error::LlmError::Forward(format!("SSM state not initialized for layer {layer_idx}"))
    })?;
    let chain_shape = super::make_gdn_prefill_chain_shape(
        seq_len,
        metadata.hidden_dim,
        d_inner,
        d_state,
        n_group,
        dt_rank,
        conv_kernel,
        ssm_state.conv_state.len(),
        ssm_state.delta_state.len(),
    );
    let attn_norm_data = kernels::tensor_as_f32_slice(&w.attn_norm);
    let conv_kernel_data = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
    let dt_bias_data = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
    let ssm_a_data = kernels::tensor_as_f32_slice(&w.ssm_a);
    let ssm_norm_data = kernels::tensor_as_f32_slice(&w.ssm_norm);
    let wants_prefix_state = prefix_collector
        .as_ref()
        .is_some_and(|collector| collector.wants_snapshot_for(seq_len));
    if wants_prefix_state && backend_runtime::try_gdn_prefill_chain_if_supported(&chain_shape)? {
        return Err(crate::error::LlmError::Forward(
            "CUDA GDN prefill chain does not yet capture prefix SSM states".to_string(),
        ));
    }

    let t_chain = std::time::Instant::now();
    let chain_output = if wants_prefix_state {
        None
    } else {
        backend_runtime::gdn_prefill_chain_q4k(
            &chain_shape,
            kernels::tensor_as_f32_slice(&hidden),
            #[cfg(feature = "cuda")]
            None,
            attn_norm_data,
            &w.qkv_weight,
            &w.gate_weight,
            &w.ssm_alpha,
            &w.ssm_beta,
            &mut ssm_state.conv_state,
            conv_kernel_data,
            dt_bias_data,
            ssm_a_data,
            &mut ssm_state.delta_state,
            ssm_norm_data,
            &w.ssm_out,
            kernels::tensor_as_f32_slice(&w.post_attn_norm),
            true,
            norm_eps,
        )?
    };
    if chain_output.is_some() {
        prof!("gdn_chain_q4k", t_chain);
    }

    let trace_gdn_stages = debug_gdn_stage_trace_enabled(layer_idx);
    let mut chain_device_carriers = None;
    let proj_vec = if let Some(mut chain_output) = chain_output {
        chain_output.release_device_output_if_present()?;
        let proj_vec = std::mem::take(&mut chain_output.ssm_projection);
        chain_device_carriers = Some(chain_output);
        proj_vec
    } else {
        let t0 = std::time::Instant::now();
        let normed = kernels::norm::rms_norm(&hidden, &w.attn_norm, norm_eps).map_err(fwd)?;
        prof!("rms_norm", t0);

        let normed_data = kernels::tensor_as_f32_slice(&normed);
        if trace_gdn_stages {
            emit_gdn_stage_trace(
                "cpu-norm-attn",
                layer_idx,
                normed_data,
                seq_len,
                metadata.hidden_dim,
            );
        }
        let normed_quantized = prefill_quantized_input_for_weight(&w.qkv_weight, normed_data);

        let t0 = std::time::Instant::now();
        let z_data_vec = if let Some(out) = gpu_quantized_prefill_gemv(&w.gate_weight, normed_data)?
        {
            out
        } else if let Some(out) = backend_runtime::metal_prefill_gdn_proj_into_if_supported(
            &w.gate_weight,
            normed_data,
            seq_len,
            metadata.hidden_dim,
        )? {
            // pm35 M2: Metal tensorops batch GEMM (Q4_K|Q6_K, RNB_METAL_PREFILL_GDN_INPROJ opt-in)
            out
        } else {
            prefill_gemv_vec(&w.gate_weight, normed_data, &normed_quantized)?
        };
        prof!("gate_gemv", t0);
        let t0 = std::time::Instant::now();
        let (alpha_vec, mut beta_vec) = if let Some(out) =
            backend_runtime::metal_prefill_gdn_f32_dual_proj_if_supported(
                &w.ssm_alpha,
                &w.ssm_beta,
                normed_data,
                seq_len,
                metadata.hidden_dim,
            )? {
            out
        } else {
            prefill_dual_gemv_q8_or_f32(&w.ssm_alpha, &w.ssm_beta, normed_data, &normed_quantized)?
        };
        prof!("alpha_beta_gemv", t0);
        if trace_gdn_stages {
            emit_gdn_stage_trace("cpu-z-raw", layer_idx, &z_data_vec, seq_len, d_inner);
            emit_gdn_stage_trace("cpu-alpha-raw", layer_idx, &alpha_vec, seq_len, num_v_heads);
            emit_gdn_stage_trace("cpu-beta-raw", layer_idx, &beta_vec, seq_len, num_v_heads);
        }

        for b in beta_vec.iter_mut() {
            *b = 1.0 / (1.0 + (-*b).exp());
        }
        if trace_gdn_stages {
            emit_gdn_stage_trace(
                "cpu-beta-sigmoid",
                layer_idx,
                &beta_vec,
                seq_len,
                num_v_heads,
            );
        }

        if trace_gdn_stages {
            emit_gdn_stage_trace(
                "cpu-conv-kernel",
                layer_idx,
                conv_kernel_data,
                conv_kernel,
                conv_channels,
            );
        }
        let mut gate_data = alpha_vec;
        super::apply_dt_gate_inplace(
            &mut gate_data,
            dt_bias_data,
            ssm_a_data,
            seq_len,
            num_v_heads,
        );
        if trace_gdn_stages {
            emit_gdn_stage_trace(
                "cpu-alpha-gate",
                layer_idx,
                &gate_data,
                seq_len,
                num_v_heads,
            );
        }

        let gpu_conv_data: Option<Vec<f32>> = {
            #[cfg(feature = "vulkan")]
            {
                let mut gpu_conv_data = None;
                if seq_len > 1 {
                    if let Some(ref mut vk) = gpu_runtime {
                        let t_gpu = std::time::Instant::now();
                        let defer_state_materialization = deferred_gdn_flush.is_some();
                        if let Some(conv_out) = backend_runtime::try_gdn_qkv_conv_prefill_window(
                            vk,
                            layer_idx,
                            &w.qkv_weight,
                            conv_kernel_data,
                            normed_data,
                            seq_len,
                            metadata.hidden_dim,
                            conv_channels,
                            conv_kernel,
                            &mut ssm_state.conv_state,
                            defer_state_materialization,
                            profiling,
                        )? {
                            if let Some(tracker) = deferred_gdn_flush.as_deref_mut() {
                                tracker.mark_touched(layer_idx);
                            }
                            prof!("qkv+conv_gpu", t_gpu);
                            gpu_conv_data = Some(conv_out);
                        }
                    }
                }
                gpu_conv_data
            }
            #[cfg(not(feature = "vulkan"))]
            {
                None
            }
        };

        let mut prefix_conv_states: Vec<(usize, Vec<f32>)> = Vec::new();
        // pm45 M2: metal conv→delta chain output(있으면 split/l2/repeat/scale/delta 전부 우회).
        // delta 이후(gated_norm) 합류는 `output` 변수가 받는다. chain 은 중간 conv/delta snapshot 을
        // 만들지 않으므로 verify-window prefix snapshot 이 필요하면(=`wants_prefix_state` 또는
        // prefix_conv_states 채움) 진입하지 않고 기존 CPU/seam 경로 그대로 둔다.
        let mut chain_conv_delta_output: Option<Vec<f32>> = None;
        // pm45 M3-1: full chain(conv→delta→gated→ssm_out)이 proj 까지 만들면 여기 담아 line 794
        // gated_norm_silu_project 호출까지 skip. M2 chain 과 같은 진입 조건에서만 시도.
        let mut chain_full_proj: Option<Vec<f32>> = None;
        let conv_data_vec = if let Some(conv_data) = gpu_conv_data {
            if let Some(collector) = prefix_collector.as_deref_mut() {
                if collector.wants_snapshot_for(seq_len) {
                    collector.mark_incomplete(layer_idx);
                }
            }
            conv_data
        } else {
            let t0 = std::time::Instant::now();
            let qkv_data =
                if let Some(out) = gpu_quantized_prefill_gemv(&w.qkv_weight, normed_data)? {
                    out
                } else if let Some(out) = backend_runtime::metal_prefill_gdn_proj_into_if_supported(
                    &w.qkv_weight,
                    normed_data,
                    seq_len,
                    metadata.hidden_dim,
                )? {
                    // pm35 M2: Metal tensorops batch GEMM (Q4_K|Q6_K, RNB_METAL_PREFILL_GDN_INPROJ)
                    out
                } else {
                    prefill_gemv_vec(&w.qkv_weight, normed_data, &normed_quantized)?
                };
            prof!("qkv_gemv", t0);
            if trace_gdn_stages {
                emit_gdn_stage_trace("cpu-qkv-raw", layer_idx, &qkv_data, seq_len, conv_channels);
            }

            let conv_input = super::build_conv_input_and_advance_state(
                &mut ssm_state.conv_state,
                &qkv_data,
                seq_len,
                conv_channels,
                conv_kernel,
            );
            if let Some(collector) = prefix_collector.as_ref() {
                for prefix_tokens in collector.snapshot_prefix_tokens(seq_len) {
                    prefix_conv_states.push((
                        prefix_tokens,
                        super::conv_state_after_prefix_tokens(
                            &conv_input,
                            prefix_tokens,
                            conv_channels,
                            conv_kernel,
                        ),
                    ));
                }
            }

            // pm45 M2: conv→delta 단일 GPU chain(metal). prefix snapshot 불요일 때만 — chain 은
            // 중간 conv/delta state 를 캡처하지 않으므로 verify-window 가 prefix 를 원하면 기존 경로.
            // gate_data/beta_vec/delta_state 는 delta_net_scan_prefill 에 넘기던 그대로,
            // conv_input/conv_kernel_data 는 ssm_prefill_conv1d_silu 에 넘기던 raw 그대로.
            if !wants_prefix_state && prefix_conv_states.is_empty() {
                if !trace_gdn_stages
                    && w.shared_expert_moe.is_none()
                    && w.ffn_gate_up_fused.is_none()
                {
                    let t_full_ffn = std::time::Instant::now();
                    if let Some((hidden_after, state_after)) =
                        backend_runtime::gdn_prefill_full_ffn_chain(
                            kernels::tensor_as_f32_slice(&hidden),
                            &conv_input,
                            conv_kernel_data,
                            &gate_data,
                            &beta_vec,
                            &ssm_state.delta_state,
                            &z_data_vec,
                            ssm_norm_data,
                            &w.ssm_out,
                            kernels::tensor_as_f32_slice(&w.post_attn_norm),
                            &w.ffn_gate_weight,
                            &w.ffn_up_weight,
                            &w.ffn_down_weight,
                            seq_len,
                            conv_channels,
                            conv_kernel,
                            num_k_heads,
                            num_v_heads,
                            head_k_dim,
                            head_v_dim,
                            metadata.hidden_dim,
                            norm_eps,
                        )?
                    {
                        ssm_state.delta_state.copy_from_slice(&state_after);
                        prof!("gdn_full_ffn_chain", t_full_ffn);
                        return Ok(Tensor::from_vec(
                            hidden_after,
                            &[seq_len, metadata.hidden_dim],
                        ));
                    }
                }

                // pm45 M3-1: full chain(conv→delta→gated→ssm_out) 먼저 시도. Some 이면 proj 까지
                // 한 command buffer 로 얻어 delta output readback + gated input upload 제거.
                // delta_state 만 갱신하고 proj 는 chain_full_proj 에 담아 line 794 gated proj skip.
                // chain_conv_delta_output 도 Some(빈 vec)로 둬서 line 617 의 output 블록(split~delta+
                // gated proj)을 전부 우회한다(empty output 은 chain_full_proj hit 시 미사용).
                let t_full = std::time::Instant::now();
                if let Some((proj, state_after)) = backend_runtime::gdn_prefill_full_chain(
                    &conv_input,
                    conv_kernel_data,
                    &gate_data,
                    &beta_vec,
                    &ssm_state.delta_state,
                    &z_data_vec,
                    ssm_norm_data,
                    &w.ssm_out,
                    seq_len,
                    conv_channels,
                    conv_kernel,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    norm_eps,
                )? {
                    ssm_state.delta_state.copy_from_slice(&state_after);
                    chain_full_proj = Some(proj);
                    chain_conv_delta_output = Some(Vec::new());
                    prof!("gdn_full_chain", t_full);
                    Vec::new()
                } else {
                    let t_chain = std::time::Instant::now();
                    if let Some((output, state_after)) =
                        backend_runtime::gdn_prefill_conv_delta_chain(
                            &conv_input,
                            conv_kernel_data,
                            &gate_data,
                            &beta_vec,
                            &ssm_state.delta_state,
                            seq_len,
                            conv_channels,
                            conv_kernel,
                            num_k_heads,
                            num_v_heads,
                            head_k_dim,
                            head_v_dim,
                            norm_eps,
                        )?
                    {
                        ssm_state.delta_state.copy_from_slice(&state_after);
                        chain_conv_delta_output = Some(output);
                        prof!("gdn_conv_delta_chain", t_chain);
                        // chain 이 conv→delta 를 흡수 → conv_data 미사용(아래 split~delta 우회).
                        Vec::new()
                    } else {
                        let t0 = std::time::Instant::now();
                        let conv_data = conv_delta_chain_cpu_conv(
                            &conv_input,
                            conv_kernel_data,
                            &w.ssm_conv1d,
                            seq_len,
                            conv_channels,
                            conv_kernel,
                        )?;
                        prof!("conv1d+silu", t0);
                        conv_data
                    }
                }
            } else {
                let t0 = std::time::Instant::now();
                let conv_data = conv_delta_chain_cpu_conv(
                    &conv_input,
                    conv_kernel_data,
                    &w.ssm_conv1d,
                    seq_len,
                    conv_channels,
                    conv_kernel,
                )?;
                prof!("conv1d+silu", t0);
                conv_data
            }
        };

        // pm45 M2: metal conv→delta chain 이 output 을 이미 만들었으면 split/l2/repeat/scale/delta
        // 전부 우회(chain Some 조건상 wants_prefix_state=false → prefix collector 블록도 무관).
        // None 이면 기존 CPU/conv-seam/delta-seam 경로 그대로.
        let output = if let Some(chain_output) = chain_conv_delta_output.take() {
            chain_output
        } else {
            let conv_data = conv_data_vec.as_slice();
            if trace_gdn_stages {
                emit_gdn_stage_trace("cpu-conv", layer_idx, conv_data, seq_len, conv_channels);
            }
            let q_dim = head_k_dim * num_k_heads;
            let k_dim = head_k_dim * num_k_heads;
            let v_dim = head_v_dim * num_v_heads;
            let (q_data, k_data, v_data) =
                super::split_conv_qkv(conv_data, seq_len, conv_channels, q_dim, k_dim, v_dim);

            let t0 = std::time::Instant::now();
            let q_tensor = Tensor::from_slice(&q_data, &[seq_len * num_k_heads, head_k_dim]);
            let k_tensor = Tensor::from_slice(&k_data, &[seq_len * num_k_heads, head_k_dim]);
            let q_normed = kernels::norm::l2_norm(&q_tensor, norm_eps).map_err(fwd)?;
            let k_normed = kernels::norm::l2_norm(&k_tensor, norm_eps).map_err(fwd)?;
            prof!("l2_norm", t0);

            let q_raw = kernels::tensor_as_f32_slice(&q_normed);
            let k_raw = kernels::tensor_as_f32_slice(&k_normed);
            let (q_final_vec, k_final_vec) = super::repeat_qk_for_value_heads(
                q_raw,
                k_raw,
                seq_len,
                num_k_heads,
                num_v_heads,
                head_k_dim,
            );
            let q_final = q_final_vec.as_slice();
            let k_final = k_final_vec.as_slice();
            // pm35 M2: k_final(scan state read+write 양면, spec 4.2 1차 hard surface). width = per-token
            // = num_v_heads*head_k_dim → k_final.len()/seq_len(정확, 추정 불요). half drift scan 누적 검증.
            if trace_gdn_stages && seq_len > 0 {
                emit_gdn_stage_trace(
                    "cpu-kfinal",
                    layer_idx,
                    k_final,
                    seq_len,
                    k_final.len() / seq_len,
                );
            }

            let scale = 1.0 / (head_k_dim as f32).sqrt();
            let mut q_scaled = q_final.to_vec();
            for x in q_scaled.iter_mut() {
                *x *= scale;
            }

            let t0 = std::time::Instant::now();
            #[cfg(feature = "cuda")]
            let prefix_snapshot_tokens = prefix_collector
                .as_ref()
                .map(|collector| collector.snapshot_prefix_tokens(seq_len))
                .unwrap_or_default();
            #[cfg(feature = "cuda")]
            let wants_prefix_state = prefix_collector
                .as_ref()
                .is_some_and(|collector| collector.wants_snapshot_for(seq_len));
            #[cfg(feature = "cuda")]
            let mut resident_delta_snapshots = Vec::new();
            #[cfg(feature = "cuda")]
            let output_with_snapshots = if wants_prefix_state && !prefix_conv_states.is_empty() {
                backend_runtime::ssm_prefill_delta_net_snapshots(
                    &mut ssm_state.delta_state,
                    &q_scaled,
                    k_final,
                    &v_data,
                    &gate_data,
                    &beta_vec,
                    seq_len,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    &prefix_snapshot_tokens,
                )?
            } else {
                None
            };
            #[cfg(feature = "cuda")]
            let output = if let Some((output, snapshots)) = output_with_snapshots {
                resident_delta_snapshots = snapshots;
                output
            } else {
                delta_net_scan_prefill(
                    &q_scaled,
                    k_final,
                    &v_data,
                    &gate_data,
                    &beta_vec,
                    &mut ssm_state.delta_state,
                    seq_len,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                )?
            };
            #[cfg(not(feature = "cuda"))]
            let output = delta_net_scan_prefill(
                &q_scaled,
                k_final,
                &v_data,
                &gate_data,
                &beta_vec,
                &mut ssm_state.delta_state,
                seq_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )?;
            if let Some(collector) = prefix_collector.as_deref_mut() {
                if collector.wants_snapshot_for(seq_len) {
                    if prefix_conv_states.is_empty() {
                        collector.mark_incomplete(layer_idx);
                    }
                    #[cfg(feature = "cuda")]
                    let mut resident_delta_by_prefix = prefix_snapshot_tokens
                        .iter()
                        .copied()
                        .zip(resident_delta_snapshots.into_iter())
                        .collect::<std::collections::HashMap<_, _>>();
                    for (prefix_tokens, conv_state) in prefix_conv_states {
                        #[cfg(feature = "cuda")]
                        let has_resident_snapshot =
                            resident_delta_by_prefix.contains_key(&prefix_tokens);
                        #[cfg(not(feature = "cuda"))]
                        let has_resident_snapshot = false;
                        match crate::engine::verify_window::prefix_delta_restore_kind(
                        prefix_tokens,
                        has_resident_snapshot,
                    ) {
                        crate::engine::verify_window::PrefixDeltaRestoreKind::ResidentSnapshot => {
                            #[cfg(feature = "cuda")]
                            collector.record_layer_with_resident_delta_snapshot_for_prefix(
                                prefix_tokens,
                                layer_idx,
                                conv_state,
                                resident_delta_by_prefix
                                    .remove(&prefix_tokens)
                                    .expect("resident snapshot availability checked"),
                            );
                            #[cfg(not(feature = "cuda"))]
                            collector.mark_incomplete_for_prefix(prefix_tokens, layer_idx);
                        }
                        crate::engine::verify_window::PrefixDeltaRestoreKind::OneStepDeltaInput => {
                            let qk_len = num_v_heads * head_k_dim;
                            let v_len = num_v_heads * head_v_dim;
                            let gate_len = num_v_heads;
                            collector.record_layer_for_prefix(
                                prefix_tokens,
                                layer_idx,
                                conv_state,
                                crate::engine::verify_window::VerifyWindowSsmDeltaInput {
                                    q: q_scaled[..qk_len].to_vec(),
                                    k: k_final[..qk_len].to_vec(),
                                    v: v_data[..v_len].to_vec(),
                                    gate: gate_data[..gate_len].to_vec(),
                                    beta: beta_vec[..gate_len].to_vec(),
                                    num_heads: num_v_heads,
                                    head_k_dim,
                                    head_v_dim,
                                },
                            );
                        }
                        crate::engine::verify_window::PrefixDeltaRestoreKind::Unsupported => {
                            collector.mark_incomplete_for_prefix(prefix_tokens, layer_idx);
                        }
                    }
                    }
                }
            }
            prof!("delta_net_scan", t0);
            output
        };
        if trace_gdn_stages {
            emit_gdn_stage_trace("cpu-delta", layer_idx, &output, seq_len, d_inner);
        }

        // pm45 M3-1: full chain 이 proj 까지 만들었으면 gated_norm_silu_project 호출 skip.
        if let Some(full_proj) = chain_full_proj.take() {
            full_proj
        } else {
            let proj_vec = if let Some(proj_vec) = {
                let t0 = std::time::Instant::now();
                let projected = backend_runtime::gdn_prefill_gated_norm_silu_project(
                    &output,
                    &z_data_vec,
                    ssm_norm_data,
                    &w.ssm_out,
                    seq_len,
                    head_v_dim,
                    norm_eps,
                )?;
                if projected.is_some() {
                    prof!("gated_norm+ssm_out", t0);
                }
                projected
            } {
                proj_vec
            } else {
                let t0 = std::time::Instant::now();
                let gated_output = if let Some(gated_output) =
                    backend_runtime::gdn_prefill_gated_norm_silu(
                        &output,
                        &z_data_vec,
                        ssm_norm_data,
                        seq_len,
                        seq_len * num_v_heads,
                        head_v_dim,
                        norm_eps,
                    )? {
                    gated_output
                } else {
                    let out_tensor = Tensor::from_vec(output, &[seq_len * num_v_heads, head_v_dim]);
                    let out_normed =
                        kernels::norm::rms_norm(&out_tensor, &w.ssm_norm, norm_eps).map_err(fwd)?;
                    let out_normed_data = kernels::tensor_as_f32_slice(&out_normed);
                    let gated_len = seq_len * d_inner;
                    let mut gated_output = vec![0.0f32; gated_len];
                    for i in 0..gated_len {
                        let z = z_data_vec[i];
                        gated_output[i] = out_normed_data[i] * (z / (1.0 + (-z).exp()));
                    }
                    gated_output
                };
                prof!("gated_norm+silu", t0);
                let t0 = std::time::Instant::now();
                // pm36: ssm_out seam. 입력 gated_output[seq*d_inner](scan 출력) → K=d_inner.
                // scan 의존이라 in_proj/gate seam 과 파이프라인 분리(별도 commit/wait).
                let proj_vec = if let Some(out) =
                    backend_runtime::metal_prefill_gdn_proj_into_if_supported(
                        &w.ssm_out,
                        &gated_output,
                        seq_len,
                        d_inner,
                    )? {
                    // pm36: Metal tensorops batch GEMM (Q5_K ssm_out, RNB_METAL_PREFILL_GDN_INPROJ opt-in)
                    out
                } else {
                    w.ssm_out.gemv_vec(&gated_output)?
                };
                prof!("ssm_out_gemv", t0);
                proj_vec
            };
            proj_vec
        }
    };
    let proj_cols = proj_vec.len() / seq_len;
    if trace_gdn_stages {
        emit_gdn_stage_trace("cpu-ssm-out", layer_idx, &proj_vec, seq_len, proj_cols);
    }
    #[cfg(feature = "cuda")]
    if let (Some(chain_output), Some(moe_w)) =
        (chain_device_carriers.as_mut(), w.shared_expert_moe.as_ref())
    {
        if let (Some(&(residual_id, residual_desc)), Some(&(moe_input_id, moe_input_desc))) = (
            chain_output.device_residual.as_ref(),
            chain_output.device_moe_input.as_ref(),
        ) {
            let t_device_moe = std::time::Instant::now();
            match try_forward_ffn_qwen35moe_device_input(
                moe_w,
                seq_len,
                metadata.hidden_dim,
                moe_input_id,
                moe_input_desc,
                residual_id,
                residual_desc,
            ) {
                Ok(Some(hidden)) => {
                    chain_output.release_device_carriers_if_present()?;
                    prof!("ffn_device_input_total", t_device_moe);
                    return Ok(hidden);
                }
                Ok(None) => {}
                Err(err) => {
                    let cleanup = chain_output.release_device_carriers_if_present();
                    return match cleanup {
                        Ok(()) => Err(err),
                        Err(cleanup_err) => Err(crate::error::LlmError::Forward(format!(
                            "{err}; CUDA GDN device carrier cleanup failed: {cleanup_err}"
                        ))),
                    };
                }
            }
        }
    }
    if let Some(chain_output) = chain_device_carriers.as_mut() {
        chain_output.release_device_carriers_if_present()?;
    }
    let proj_tensor = Tensor::from_vec(proj_vec, &[seq_len, proj_cols]);
    hidden = kernels::elementwise::add(&hidden, &proj_tensor).map_err(fwd)?;
    if trace_gdn_stages {
        emit_gdn_stage_trace(
            "cpu-after-ssm-add",
            layer_idx,
            kernels::tensor_as_f32_slice(&hidden),
            seq_len,
            metadata.hidden_dim,
        );
    }

    let t0 = std::time::Instant::now();
    if let Some(moe_w) = &w.shared_expert_moe {
        let hidden = forward_shared_expert_moe(
            ModelArchitecture::Qwen35MoE,
            hidden,
            &w.post_attn_norm,
            moe_w,
            seq_len,
            metadata.hidden_dim,
            norm_eps,
            layer_idx,
        )?;
        prof!("ffn_total", t0);
        return Ok(hidden);
    }

    let normed = kernels::norm::rms_norm(&hidden, &w.post_attn_norm, norm_eps).map_err(fwd)?;
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    if trace_gdn_stages {
        emit_gdn_stage_trace(
            "cpu-norm-ffn",
            layer_idx,
            normed_data,
            seq_len,
            metadata.hidden_dim,
        );
    }
    if seq_len <= 4 {
        let mut hidden_data = kernels::tensor_as_f32_slice(&hidden).to_vec();
        let norm_weight_data = kernels::tensor_as_f32_slice(&w.post_attn_norm);
        let gpu_ffn_done = backend_runtime::try_gdn_prefill_ffn_chain_window_if_supported(
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_deref_mut(),
            layer_idx,
            &mut hidden_data,
            metadata.hidden_dim,
            norm_weight_data,
            norm_eps,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
        );
        if gpu_ffn_done {
            prof!("ffn_total", t0);
            return Ok(Tensor::from_vec(
                hidden_data,
                &[seq_len, metadata.hidden_dim],
            ));
        }
    }
    // pm33: Metal prefill FFN batch GEMM chain (GDN inline FFN, fused 제외 + trace off, env opt-in).
    // GDN silu(g/(1+exp(-g))*up)는 Metal silu_mul 과 일치. 미지원 quant/shape 시 used=false → CPU.
    #[cfg(feature = "metal")]
    let metal_down: Option<Tensor> = if w.ffn_gate_up_fused.is_none() && !trace_gdn_stages {
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

    let down = if let Some(d) = metal_down {
        d
    } else {
        let (mut gate_vec, up_vec) = prefill_gate_up_vectors(
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            w.ffn_gate_up_fused.as_ref(),
            normed_data,
            seq_len,
        )?;
        for i in 0..gate_vec.len() {
            let g = gate_vec[i];
            gate_vec[i] = (g / (1.0 + (-g).exp())) * up_vec[i];
        }
        let gdn_ffn_inner_dim = gate_vec.len() / seq_len;
        let gate_up_tensor = Tensor::from_vec(gate_vec, &[seq_len, gdn_ffn_inner_dim]);
        w.ffn_down_weight.gemv(&gate_up_tensor)?
    };
    if trace_gdn_stages {
        emit_gdn_stage_trace(
            "cpu-ffn-down",
            layer_idx,
            kernels::tensor_as_f32_slice(&down),
            seq_len,
            metadata.hidden_dim,
        );
    }
    hidden = kernels::elementwise::add(&hidden, &down).map_err(fwd)?;
    if trace_gdn_stages {
        emit_gdn_stage_trace(
            "cpu-final",
            layer_idx,
            kernels::tensor_as_f32_slice(&hidden),
            seq_len,
            metadata.hidden_dim,
        );
    }
    prof!("ffn_total", t0);

    Ok(hidden)
}

/// pm45 M2: conv1d+silu 단독 CPU/seam 경로(metal conv→delta chain miss / prefix snapshot 경로용).
/// 기존 conv_data_vec else 가지의 conv1d 로직과 1:1 — backend seam(cuda/metal conv1d) 시도 후
/// CPU `ssm_conv1d_silu` fallback. chain Some 일 때만 이 conv 결과가 미사용된다.
fn conv_delta_chain_cpu_conv(
    conv_input: &[f32],
    conv_kernel_data: &[f32],
    ssm_conv1d: &Tensor,
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> crate::error::Result<Vec<f32>> {
    let fwd = |e: rnb_core::error::RnbError| crate::error::LlmError::Forward(e.to_string());
    let conv_data = if let Some(conv_data) = backend_runtime::ssm_prefill_conv1d_silu(
        conv_input,
        conv_kernel_data,
        seq_len,
        conv_channels,
        conv_kernel,
    )? {
        conv_data
    } else {
        let total_conv_len = (conv_kernel - 1) + seq_len;
        let conv_input_tensor = Tensor::from_slice(conv_input, &[total_conv_len, conv_channels]);
        let conv_out =
            kernels::conv::ssm_conv1d_silu(&conv_input_tensor, ssm_conv1d).map_err(fwd)?;
        kernels::tensor_as_f32_slice(&conv_out).to_vec()
    };
    Ok(conv_data)
}

#[allow(clippy::too_many_arguments)]
fn delta_net_scan_prefill(
    q_scaled: &[f32],
    k_final: &[f32],
    v_data: &[f32],
    gate_data: &[f32],
    beta_vec: &[f32],
    delta_state: &mut [f32],
    seq_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> crate::error::Result<Vec<f32>> {
    if let Some(output) = backend_runtime::ssm_prefill_delta_net(
        delta_state,
        q_scaled,
        k_final,
        v_data,
        gate_data,
        beta_vec,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )? {
        return Ok(output);
    }
    // pm39 M3: Metal chunkwise GPU delta scan (cuda 아닐 때). opt-in RNB_METAL_PREFILL_GDN_SCAN=1.
    // GQA 는 위에서 q_scaled/k_final 이 이미 num_v_heads 로 repeat 푼 상태(repeat_qk_for_value_heads).
    if let Some(output) = backend_runtime::metal_prefill_delta_net_scan_into_if_supported(
        q_scaled,
        k_final,
        v_data,
        gate_data,
        beta_vec,
        delta_state,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )? {
        return Ok(output);
    }
    Ok(kernels::delta_net::delta_net_scan(
        q_scaled,
        k_final,
        v_data,
        gate_data,
        beta_vec,
        delta_state,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvVarGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                old: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.old {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn debug_gdn_stage_trace_selects_requested_layer() {
        let _trace_guard = EnvVarGuard::new("RNB_DEBUG_GDN_STAGE_TRACE");
        let _layer_guard = EnvVarGuard::new("RNB_DEBUG_GDN_STAGE_TRACE_LAYER");

        unsafe {
            std::env::remove_var("RNB_DEBUG_GDN_STAGE_TRACE");
            std::env::remove_var("RNB_DEBUG_GDN_STAGE_TRACE_LAYER");
        }
        assert!(!debug_gdn_stage_trace_enabled(0));

        unsafe {
            std::env::set_var("RNB_DEBUG_GDN_STAGE_TRACE", "1");
        }
        assert!(debug_gdn_stage_trace_enabled(0));
        assert!(!debug_gdn_stage_trace_enabled(4));

        unsafe {
            std::env::set_var("RNB_DEBUG_GDN_STAGE_TRACE_LAYER", "4");
        }
        assert!(!debug_gdn_stage_trace_enabled(0));
        assert!(debug_gdn_stage_trace_enabled(4));

        unsafe {
            std::env::set_var("RNB_DEBUG_GDN_STAGE_TRACE_LAYER", "all");
        }
        assert!(debug_gdn_stage_trace_enabled(0));
        assert!(debug_gdn_stage_trace_enabled(4));
    }

    #[test]
    fn debug_f32_bit_hash_tracks_exact_bit_changes() {
        let values = [0.0f32, -0.0, 1.0];
        let same = [0.0f32, -0.0, 1.0];
        let changed = [0.0f32, 0.0, 1.0];

        assert_eq!(debug_f32_bit_hash(&values), debug_f32_bit_hash(&same));
        assert_ne!(debug_f32_bit_hash(&values), debug_f32_bit_hash(&changed));
    }
}
