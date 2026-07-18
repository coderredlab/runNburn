//! Qwen GDN decode layer path.

use crate::engine::models::shared_expert_moe::decode_shared_expert_moe;
use crate::engine::*;

fn debug_gdn_decode_stage_trace_enabled(layer_idx: usize) -> bool {
    std::env::var_os("RNB_DEBUG_GDN_STAGE_TRACE").is_some() && layer_idx == 0
}

fn emit_gdn_decode_stage_trace(tag: &str, layer_idx: usize, data: &[f32], width: usize) {
    if width == 0 || data.len() < width {
        eprintln!(
            "[gdn-stage-trace][{}] layer={} invalid len={} width={}",
            tag,
            layer_idx,
            data.len(),
            width
        );
        return;
    }
    let row = &data[..width];
    let n = row.len().max(1) as f32;
    let mean = row.iter().sum::<f32>() / n;
    let l2 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head = row.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[gdn-stage-trace][{}] layer={} mean={:.6} l2={:.6} first={:?} mid={:?} prev={:?} last={:?}",
        tag, layer_idx, mean, l2, head, head, head, head
    );
}

// gdn carrier 진입 조건(weight shape). decode loop 가 호출 전 판정 가능하게 추출.
// env/quant/shape 판정은 seam(metal_inference.rs)이 소유 — 여기엔 넣지 않는다.
// caller(아래 가드 + 1.4 decode loop)가 모두 metal cfg 안이라 함수도 동일 cfg.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) fn gdn_carrier_eligible(w: &GdnLayerWeights) -> bool {
    w.shared_expert_moe.is_none() && w.ffn_gate_up_fused.is_none()
}

/// GDN (Gated Delta Net) layer decode (seq_len=1). Operates in-place on scratch buffers.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn decode_gdn_layer_qwen(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    w: &GdnLayerWeights,
    layer_idx: usize,
    #[cfg(feature = "vulkan")] mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    let prof_level = policy::profiling_level();
    let profiling = prof_level >= 1;
    let verbose = prof_level >= 2;
    macro_rules! prof {
        ($label:expr, $t:expr) => {
            if profiling && (verbose || layer_idx == 0) {
                eprintln!(
                    "  [DEC-GDN L{}] {:20} {:.1}ms",
                    layer_idx,
                    $label,
                    $t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
    }

    let hidden_dim = metadata.hidden_dim;
    let norm_eps = metadata.norm_eps;
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
    let q_dim = head_k_dim * num_k_heads;
    let k_dim = head_k_dim * num_k_heads;
    let v_dim = head_v_dim * num_v_heads;
    let trace_gdn_stages = debug_gdn_decode_stage_trace_enabled(layer_idx);

    // pm16: GDN layer 전체 device-resident carrier (RNB_METAL_GDN_LAYER=1, dense FFN 만).
    // 성공 시 hidden + conv_state + delta_state 를 단일 command buffer chain 으로 갱신하고
    // 즉시 반환(FFN 포함, host 경로 skip). qkv/gate/alpha/beta/ssm_out/ffn Q4_K 한정.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if gdn_carrier_eligible(w) {
        let attn_norm_w = kernels::tensor_as_f32_slice(&w.attn_norm);
        let dt_bias_w = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
        let ssm_a_w = kernels::tensor_as_f32_slice(&w.ssm_a);
        let conv1d_w = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
        let ssm_norm_w = kernels::tensor_as_f32_slice(&w.ssm_norm);
        let ffn_norm_w = kernels::tensor_as_f32_slice(&w.post_attn_norm);
        let ffn_dim = w.ffn_gate_weight.rows;
        let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "SSM state not initialized for layer {layer_idx}"
            ))
        })?;
        let did = backend_runtime::metal_gdn_layer_into_if_supported(
            layer_idx,
            &mut scratch.hidden[..hidden_dim],
            &mut ssm_state.conv_state,
            &mut ssm_state.delta_state,
            attn_norm_w,
            &w.qkv_weight,
            &w.gate_weight,
            &w.ssm_alpha,
            &w.ssm_beta,
            dt_bias_w,
            ssm_a_w,
            conv1d_w,
            ssm_norm_w,
            &w.ssm_out,
            ffn_norm_w,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            hidden_dim,
            conv_channels,
            conv_kernel,
            d_inner,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            ffn_dim,
            norm_eps,
        )?;
        if did {
            return Ok(());
        }
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if let Some(moe_w) = &w.shared_expert_moe {
        let t0 = std::time::Instant::now();
        let attn_norm_w = kernels::tensor_as_f32_slice(&w.attn_norm);
        let dt_bias_w = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
        let ssm_a_w = kernels::tensor_as_f32_slice(&w.ssm_a);
        let conv1d_w = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
        let ssm_norm_w = kernels::tensor_as_f32_slice(&w.ssm_norm);
        let ffn_norm_w = kernels::tensor_as_f32_slice(&w.post_attn_norm);
        let did_full = {
            let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "SSM state not initialized for layer {layer_idx}"
                ))
            })?;
            backend_runtime::metal_gdn_moe_layer_into_if_supported(
                layer_idx,
                &mut scratch.hidden[..hidden_dim],
                &mut ssm_state.conv_state,
                &mut ssm_state.delta_state,
                attn_norm_w,
                &w.qkv_weight,
                &w.gate_weight,
                &w.ssm_alpha,
                &w.ssm_beta,
                dt_bias_w,
                ssm_a_w,
                conv1d_w,
                ssm_norm_w,
                &w.ssm_out,
                ffn_norm_w,
                moe_w,
                hidden_dim,
                conv_channels,
                conv_kernel,
                d_inner,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                norm_eps,
            )?
        };
        if did_full {
            prof!("gdn_moe_full_carrier", t0);
            return Ok(());
        }
        let did = {
            let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "SSM state not initialized for layer {layer_idx}"
                ))
            })?;
            backend_runtime::metal_gdn_core_into_if_supported(
                layer_idx,
                &mut scratch.hidden[..hidden_dim],
                &mut ssm_state.conv_state,
                &mut ssm_state.delta_state,
                attn_norm_w,
                &w.qkv_weight,
                &w.gate_weight,
                &w.ssm_alpha,
                &w.ssm_beta,
                dt_bias_w,
                ssm_a_w,
                conv1d_w,
                ssm_norm_w,
                &w.ssm_out,
                hidden_dim,
                conv_channels,
                conv_kernel,
                d_inner,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                norm_eps,
            )?
        };
        if did {
            prof!("gdn_core_carrier", t0);
            if trace_gdn_stages {
                emit_gdn_decode_stage_trace(
                    "metal-dec-core-after-ssm-add",
                    layer_idx,
                    &scratch.hidden[..hidden_dim],
                    hidden_dim,
                );
            }
            let t0 = std::time::Instant::now();
            decode_shared_expert_moe(
                scratch,
                ModelArchitecture::Qwen35MoE,
                &w.post_attn_norm,
                moe_w,
                hidden_dim,
                norm_eps,
                layer_idx,
            )?;
            prof!("ffn_total", t0);
            return Ok(());
        }
    }

    let t0 = std::time::Instant::now();
    let attn_norm_data = kernels::tensor_as_f32_slice(&w.attn_norm);
    kernels::norm::rms_norm_into(
        &scratch.hidden[..hidden_dim],
        attn_norm_data,
        norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
    );
    prof!("rms_norm", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-norm-attn",
            layer_idx,
            &scratch.norm_buf[..hidden_dim],
            hidden_dim,
        );
    }

    if backend_runtime::verify_gdn_layer(layer_idx) {
        #[cfg(feature = "vulkan")]
        let gpu_runtime_present = gpu_runtime.is_some();
        #[cfg(not(feature = "vulkan"))]
        let gpu_runtime_present = false;
        eprintln!(
            "[GPU DEBUG] GDN L0: gpu_runtime={}, qkv_type={:?}, conv_ch={}, cols={}",
            gpu_runtime_present, w.qkv_weight.ggml_type, conv_channels, w.qkv_weight.cols
        );
    }
    let gpu_qkv_gate_ok = backend_runtime::try_decode_gdn_qkv_gate_if_supported(
        #[cfg(feature = "vulkan")]
        gpu_runtime.as_deref_mut(),
        layer_idx,
        &scratch.norm_buf[..hidden_dim],
        &w.qkv_weight,
        &w.gate_weight,
        conv_channels,
        d_inner,
        &mut scratch.qkv_buf[..conv_channels],
        &mut scratch.z_buf[..d_inner],
    );
    if gpu_qkv_gate_ok && backend_runtime::verify_gdn_layer(layer_idx) {
        let mut cpu_qkv = vec![0.0f32; conv_channels];
        w.qkv_weight
            .gemv_into(&scratch.norm_buf[..hidden_dim], &mut cpu_qkv)
            .ok();
        let max_diff = scratch.qkv_buf[..conv_channels]
            .iter()
            .zip(&cpu_qkv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let gpu_first8: Vec<f32> = scratch.qkv_buf[..8].to_vec();
        let cpu_first8: Vec<f32> = cpu_qkv[..8].to_vec();
        eprintln!(
            "[GPU VERIFY] L{} GDN qkv: max_diff={:.6}, rows={}, cols={}",
            layer_idx, max_diff, conv_channels, w.qkv_weight.cols
        );
        eprintln!("[GPU VERIFY]   gpu[0..8]={:.4?}", gpu_first8);
        eprintln!("[GPU VERIFY]   cpu[0..8]={:.4?}", cpu_first8);
    }
    // Metal GDN qkv+gate device-resident chain (RNB_METAL_GDN_CHAIN=1) — 2 GEMV 를
    // 단일 command buffer 로(per-op 2 commit/wait → 1). Q4_K qkv/gate 한정.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let gpu_qkv_gate_ok = if !gpu_qkv_gate_ok {
        backend_runtime::metal_gdn_inproj_chain_into_if_supported(
            &scratch.norm_buf[..hidden_dim],
            &w.qkv_weight,
            &w.gate_weight,
            &mut scratch.qkv_buf[..conv_channels],
            &mut scratch.z_buf[..d_inner],
            hidden_dim,
            conv_channels,
            d_inner,
        )?
    } else {
        gpu_qkv_gate_ok
    };
    let gpu_qkv_gate_ok = if !gpu_qkv_gate_ok {
        let qkv_ok = gpu_gemv_into_if_supported(
            &w.qkv_weight,
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.qkv_buf[..conv_channels],
            "gdn qkv",
            false,
        )?;
        let gate_ok = gpu_gemv_into_if_supported(
            &w.gate_weight,
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.z_buf[..d_inner],
            "gdn gate",
            false,
        )?;
        qkv_ok && gate_ok
    } else {
        gpu_qkv_gate_ok
    };

    let t0 = std::time::Instant::now();
    if !gpu_qkv_gate_ok {
        let (qkv_us, gate_us) = decode_gdn_qkv_gate_cpu_into(
            scratch,
            &w.qkv_weight,
            &w.gate_weight,
            hidden_dim,
            conv_channels,
            d_inner,
        )?;
        if verbose {
            eprintln!(
                "  [DEC-GDN L{}] {:20} {:.1}ms",
                layer_idx,
                "qkv_gemv",
                qkv_us as f64 / 1000.0
            );
            eprintln!(
                "  [DEC-GDN L{}] {:20} {:.1}ms",
                layer_idx,
                "gate_gemv",
                gate_us as f64 / 1000.0
            );
        }
    }
    prof!("qkv+gate_gemv", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-qkv-raw",
            layer_idx,
            &scratch.qkv_buf[..conv_channels],
            conv_channels,
        );
        emit_gdn_decode_stage_trace(
            "cpu-dec-z-raw",
            layer_idx,
            &scratch.z_buf[..d_inner],
            d_inner,
        );
    }

    let t0 = std::time::Instant::now();
    let gpu_alpha_beta_ok = backend_runtime::try_decode_gdn_alpha_beta_if_supported(
        #[cfg(feature = "vulkan")]
        gpu_runtime.as_deref_mut(),
        layer_idx,
        &scratch.norm_buf[..hidden_dim],
        &w.ssm_alpha,
        &w.ssm_beta,
        num_v_heads,
        &mut scratch.alpha_buf[..num_v_heads],
        &mut scratch.beta_buf[..num_v_heads],
    );

    if !gpu_alpha_beta_ok {
        w.ssm_alpha.gemv_into(
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.alpha_buf[..num_v_heads],
        )?;
        w.ssm_beta.gemv_into(
            &scratch.norm_buf[..hidden_dim],
            &mut scratch.beta_buf[..num_v_heads],
        )?;
    }
    prof!("alpha_beta_gemv", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-alpha-raw",
            layer_idx,
            &scratch.alpha_buf[..num_v_heads],
            num_v_heads,
        );
        emit_gdn_decode_stage_trace(
            "cpu-dec-beta-raw",
            layer_idx,
            &scratch.beta_buf[..num_v_heads],
            num_v_heads,
        );
    }

    kernels::activation::sigmoid_inplace(&mut scratch.beta_buf[..num_v_heads]);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-beta-sigmoid",
            layer_idx,
            &scratch.beta_buf[..num_v_heads],
            num_v_heads,
        );
    }

    let dt_bias_data = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
    let ssm_a_data = kernels::tensor_as_f32_slice(&w.ssm_a);
    for h in 0..num_v_heads {
        let a_biased = scratch.alpha_buf[h] + dt_bias_data[h];
        let sp = (1.0 + a_biased.exp()).ln();
        scratch.alpha_buf[h] = sp * ssm_a_data[h];
    }
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-alpha-gate",
            layer_idx,
            &scratch.alpha_buf[..num_v_heads],
            num_v_heads,
        );
    }

    let ssm_state = kv_cache.get_ssm_state_mut(layer_idx).ok_or_else(|| {
        crate::error::LlmError::Forward(format!("SSM state not initialized for layer {layer_idx}"))
    })?;

    let conv_state_len = (conv_kernel - 1) * conv_channels;
    let total_conv_len = conv_kernel;
    scratch.conv_input[..conv_state_len].copy_from_slice(&ssm_state.conv_state);
    scratch.conv_input[conv_state_len..conv_state_len + conv_channels]
        .copy_from_slice(&scratch.qkv_buf[..conv_channels]);

    let shift_len = (conv_kernel - 2) * conv_channels;
    if shift_len > 0 {
        ssm_state
            .conv_state
            .copy_within(conv_channels..conv_state_len, 0);
    }
    ssm_state.conv_state[shift_len..conv_state_len]
        .copy_from_slice(&scratch.qkv_buf[..conv_channels]);

    let t0 = std::time::Instant::now();
    let conv_kernel_data = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
    kernels::conv::ssm_conv1d_silu_into(
        &scratch.conv_input[..total_conv_len * conv_channels],
        conv_kernel_data,
        &mut scratch.conv_out[..conv_channels],
        1,
        conv_channels,
        conv_kernel,
    );
    prof!("conv1d+silu", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-conv",
            layer_idx,
            &scratch.conv_out[..conv_channels],
            conv_channels,
        );
    }

    scratch.gdn_q[..q_dim].copy_from_slice(&scratch.conv_out[..q_dim]);
    scratch.gdn_k[..k_dim].copy_from_slice(&scratch.conv_out[q_dim..q_dim + k_dim]);
    scratch.gdn_v[..v_dim].copy_from_slice(&scratch.conv_out[q_dim + k_dim..q_dim + k_dim + v_dim]);

    let t0 = std::time::Instant::now();
    kernels::norm::l2_norm_into(
        &scratch.gdn_q[..q_dim],
        norm_eps,
        &mut scratch.gdn_q_norm[..q_dim],
        head_k_dim,
    );
    kernels::norm::l2_norm_into(
        &scratch.gdn_k[..k_dim],
        norm_eps,
        &mut scratch.gdn_k_norm[..k_dim],
        head_k_dim,
    );
    prof!("l2_norm", t0);

    let scale = 1.0 / (head_k_dim as f32).sqrt();
    for x in scratch.gdn_q_norm[..q_dim].iter_mut() {
        *x *= scale;
    }

    let (q_input, k_input): (&[f32], &[f32]) = if num_v_heads != num_k_heads {
        assert!(
            num_v_heads % num_k_heads == 0,
            "gdn GQA: num_v_heads ({}) must be multiple of num_k_heads ({})",
            num_v_heads,
            num_k_heads
        );
        for vh in 0..num_v_heads {
            let kh = vh % num_k_heads;
            let src = kh * head_k_dim;
            let dst = vh * head_k_dim;
            scratch.gdn_q_rep[dst..dst + head_k_dim]
                .copy_from_slice(&scratch.gdn_q_norm[src..src + head_k_dim]);
            scratch.gdn_k_rep[dst..dst + head_k_dim]
                .copy_from_slice(&scratch.gdn_k_norm[src..src + head_k_dim]);
        }
        (
            &scratch.gdn_q_rep[..num_v_heads * head_k_dim],
            &scratch.gdn_k_rep[..num_v_heads * head_k_dim],
        )
    } else {
        (&scratch.gdn_q_norm[..q_dim], &scratch.gdn_k_norm[..k_dim])
    };
    let t0 = std::time::Instant::now();
    let gpu_delta_ok = if let Some(out) = backend_runtime::try_delta_step_if_supported(
        &mut ssm_state.delta_state,
        q_input,
        k_input,
        &scratch.gdn_v[..v_dim],
        &scratch.alpha_buf[..num_v_heads],
        &scratch.beta_buf[..num_v_heads],
        num_v_heads,
        head_k_dim,
        head_v_dim,
    ) {
        match out {
            Ok(out) => {
                scratch.delta_out[..d_inner].copy_from_slice(&out[..d_inner]);
                true
            }
            Err(err) => {
                eprintln!("[cuda] delta_net failed, CPU fallback: {err}");
                false
            }
        }
    } else {
        false
    };
    if !gpu_delta_ok {
        kernels::delta_net::delta_net_scan_into(
            q_input,
            k_input,
            &scratch.gdn_v[..v_dim],
            &scratch.alpha_buf[..num_v_heads],
            &scratch.beta_buf[..num_v_heads],
            &mut ssm_state.delta_state,
            &mut scratch.delta_out[..d_inner],
            1,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
    }
    prof!("delta_net_scan", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-delta",
            layer_idx,
            &scratch.delta_out[..d_inner],
            d_inner,
        );
    }

    let ssm_norm_data = kernels::tensor_as_f32_slice(&w.ssm_norm);
    for h in 0..num_v_heads {
        let off = h * head_v_dim;
        kernels::norm::rms_norm_into(
            &scratch.delta_out[off..off + head_v_dim],
            ssm_norm_data,
            norm_eps,
            &mut scratch.gated_out[off..off + head_v_dim],
        );
    }

    for x in scratch.z_buf[..d_inner].iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
    kernels::elementwise::mul_inplace(&mut scratch.gated_out[..d_inner], &scratch.z_buf[..d_inner]);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-gated",
            layer_idx,
            &scratch.gated_out[..d_inner],
            d_inner,
        );
    }

    let t0 = std::time::Instant::now();
    let gpu_ssm_out_ok = backend_runtime::try_decode_gemv_if_supported(
        #[cfg(feature = "vulkan")]
        gpu_runtime.as_deref_mut(),
        layer_idx,
        backend_runtime::DecodeProjectionKind::GdnSsmOut,
        &w.ssm_out,
        hidden_dim,
        &scratch.gated_out[..d_inner],
        &mut scratch.ssm_proj[..hidden_dim],
        "GDN ssm_out",
    );
    let gpu_ssm_out_ok = if !gpu_ssm_out_ok {
        gpu_gemv_into_if_supported(
            &w.ssm_out,
            &scratch.gated_out[..d_inner],
            &mut scratch.ssm_proj[..hidden_dim],
            "gdn ssm_out",
            false,
        )?
    } else {
        gpu_ssm_out_ok
    };

    if !gpu_ssm_out_ok {
        w.ssm_out.gemv_into(
            &scratch.gated_out[..d_inner],
            &mut scratch.ssm_proj[..hidden_dim],
        )?;
    }
    kernels::elementwise::add_inplace(
        &mut scratch.hidden[..hidden_dim],
        &scratch.ssm_proj[..hidden_dim],
    );
    prof!("ssm_out_gemv", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-ssm-out",
            layer_idx,
            &scratch.ssm_proj[..hidden_dim],
            hidden_dim,
        );
        emit_gdn_decode_stage_trace(
            "cpu-dec-after-ssm-add",
            layer_idx,
            &scratch.hidden[..hidden_dim],
            hidden_dim,
        );
    }

    let t0 = std::time::Instant::now();
    if let Some(moe_w) = &w.shared_expert_moe {
        decode_shared_expert_moe(
            scratch,
            ModelArchitecture::Qwen35MoE,
            &w.post_attn_norm,
            moe_w,
            hidden_dim,
            norm_eps,
            layer_idx,
        )?;
    } else {
        decode_ffn(
            scratch,
            ModelArchitecture::Qwen35,
            &w.post_attn_norm,
            &None,
            &w.ffn_gate_weight,
            &w.ffn_up_weight,
            &w.ffn_down_weight,
            &w.ffn_gate_up_fused,
            hidden_dim,
            norm_eps,
            layer_idx,
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_mut().map(|v| &mut **v),
        )?;
    }
    prof!("ffn_total", t0);
    if trace_gdn_stages {
        emit_gdn_decode_stage_trace(
            "cpu-dec-norm-ffn",
            layer_idx,
            &scratch.norm_buf[..hidden_dim],
            hidden_dim,
        );
        emit_gdn_decode_stage_trace(
            "cpu-dec-ffn-down",
            layer_idx,
            &scratch.ffn_down[..hidden_dim],
            hidden_dim,
        );
        emit_gdn_decode_stage_trace(
            "cpu-dec-final",
            layer_idx,
            &scratch.hidden[..hidden_dim],
            hidden_dim,
        );
    }

    Ok(())
}
