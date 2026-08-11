//! Decode FFN sub-block shared by attention and GDN decode layers.

use super::*;

/// FFN sub-block for decode path. Reads from scratch.norm_buf, writes result via add_inplace to scratch.hidden.
/// Called by both decode_attention_layer and decode_gdn_layer.
#[allow(unused_variables)]
pub(super) fn decode_ffn(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    ffn_norm_weight: &Tensor,
    post_ffw_norm_weight: &Option<Tensor>,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    ffn_gate_up_fused: &Option<QuantizedWeight>,
    hidden_dim: usize,
    norm_eps: f32,
    post_norm_eps: f32,
    layer_idx: usize,
    #[cfg(feature = "vulkan")] gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<()> {
    let prof_level = super::policy::profiling_level();
    let profiling = prof_level >= 1;
    let verbose = prof_level >= 2;
    macro_rules! prof {
        ($label:expr, $t:expr) => {
            if profiling && (verbose || layer_idx == 0 || layer_idx == 3) {
                eprintln!(
                    "  [DEC-FFN L{}] {:20} {:.1}ms",
                    layer_idx,
                    $label,
                    $t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
    }

    #[cfg(feature = "vulkan")]
    let mut gpu_runtime = gpu_runtime;
    let needs_post_ffw_norm = post_ffw_norm_weight.is_some()
        && (use_gemma_block_semantics(architecture)
            || super::models::muse_glimmer::uses_muse_glimmer_semantics(architecture));

    #[cfg(feature = "cuda")]
    if use_gemma_block_semantics(architecture) {
        let norm_weight_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
        let post_norm_weight_data = post_ffw_norm_weight
            .as_ref()
            .map(kernels::tensor_as_f32_slice);
        let unit_offset_norm = super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled();
        let t_chain = std::time::Instant::now();
        if let Some(output) = backend_runtime::dense_q4k_gelu_ffn_norm_residual_if_supported(
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            norm_weight_data,
            post_norm_weight_data,
            &scratch.hidden[..hidden_dim],
            norm_eps,
            unit_offset_norm,
            true,
        )? {
            scratch.hidden[..hidden_dim].copy_from_slice(&output[..hidden_dim]);
            prof!("ffn_resident_chain", t_chain);
            return Ok(());
        }
    }

    // cu203: Qwen3.6 dense(27B 등) decode FFN device chain — norm→gate/up(silu)→down→residual
    // 를 device 1왕복으로 실행한다. 기존 경로는 GEMV마다 host↔device 왕복(층당 H2D/D2H
    // 다발)이라 토큰당 수천 memcpy가 GPU idle 44%의 주범이었다. gate/up Q4_K + down
    // Q4/Q5/Q6_K 만 진입하고, 그 밖은 기존 개별 GEMV 경로로 폴백한다.
    #[cfg(feature = "cuda")]
    if matches!(architecture, ModelArchitecture::Qwen35)
        && post_ffw_norm_weight.is_none()
        && ffn_gate_up_fused.is_none()
        && super::policy::qwen35_dense_ffn_chain_enabled()
    {
        let norm_weight_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
        let t_chain = std::time::Instant::now();
        if let Some(output) = backend_runtime::dense_q4k_gelu_ffn_norm_residual_if_supported(
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            norm_weight_data,
            None,
            &scratch.hidden[..hidden_dim],
            norm_eps,
            false,
            false,
        )? {
            scratch.hidden[..hidden_dim].copy_from_slice(&output[..hidden_dim]);
            prof!("ffn_resident_chain", t_chain);
            return Ok(());
        }
    }

    // Dense Metal FFN carrier. Gemma GeGLU + post-FFW norm is default-on; diagnostic
    // unit-offset norm modes retain the host path because this carrier consumes raw weights.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let gemma = use_gemma_block_semantics(architecture);
        let unit_offset_norm = gemma
            && (super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
                || super::policy::gemma_unit_offset_norm_enabled()
                || super::policy::gemma_unit_offset_main_norm_enabled());
        if ffn_gate_up_fused.is_none() && !unit_offset_norm {
            let norm_w = kernels::tensor_as_f32_slice(ffn_norm_weight);
            let post_norm_w = post_ffw_norm_weight
                .as_ref()
                .map(kernels::tensor_as_f32_slice);
            let ffn_dim = ffn_gate_weight.rows;
            let t_chain = std::time::Instant::now();
            let done = backend_runtime::metal_ffn_chain_into_if_supported(
                norm_w,
                post_norm_w,
                ffn_gate_weight,
                ffn_up_weight,
                ffn_down_weight,
                &mut scratch.hidden[..hidden_dim],
                hidden_dim,
                ffn_dim,
                norm_eps,
                gemma,
            )?;
            if done {
                prof!("ffn_chain_metal", t_chain);
                return Ok(());
            }
        }
    }

    if !needs_post_ffw_norm {
        let norm_weight_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
        let gpu_chain_ok = backend_runtime::try_decode_ffn_chain_if_supported(
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_deref_mut(),
            layer_idx,
            &mut scratch.hidden[..hidden_dim],
            norm_weight_data,
            norm_eps,
            hidden_dim,
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
        );
        if gpu_chain_ok {
            if profiling && (verbose || layer_idx == 0 || layer_idx == 3) {
                eprintln!("  [DEC-FFN L{}] {:20} gpu", layer_idx, "ffn_chain");
            }
            return Ok(());
        }
    }

    let t_norm = std::time::Instant::now();
    let norm_weight_data = kernels::tensor_as_f32_slice(ffn_norm_weight);
    apply_model_norm_into(
        &scratch.hidden[..hidden_dim],
        norm_weight_data,
        norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
        architecture,
    );
    prof!("rms_norm", t_norm);

    let norm_data = &scratch.norm_buf[..hidden_dim];
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut gpu_down_done = false;
    #[cfg_attr(
        not(any(feature = "cuda", feature = "vulkan", feature = "mediatek")),
        allow(unused_mut)
    )]
    let mut gpu_ffn_ok = false;

    #[cfg(feature = "mediatek")]
    if let Some(output) = super::mediatek_ffn::try_mediatek_gemma_ffn_down(
        architecture,
        layer_idx,
        hidden_dim,
        norm_data,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_down_weight,
        ffn_gate_up_fused.is_some(),
    )? {
        scratch.ffn_down[..hidden_dim].copy_from_slice(&output[..hidden_dim]);
        gpu_ffn_ok = true;
        gpu_down_done = true;
    }

    if !gpu_ffn_ok && !needs_post_ffw_norm {
        let gate_rows = ffn_gate_weight.rows;
        let gpu_gate_dispatched = backend_runtime::try_decode_ffn_gate_async_if_supported(
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_deref_mut(),
            layer_idx,
            norm_data,
            ffn_gate_weight,
            gate_rows,
        );

        #[cfg(feature = "vulkan")]
        {
            if gpu_gate_dispatched {
                decode_ffn_up_cpu_best_effort(scratch, ffn_up_weight, hidden_dim, |label, t| {
                    prof!(label, t)
                });

                let t_gate = std::time::Instant::now();
                let mut gate_out = [&mut scratch.ffn_gate[..gate_rows]];
                let gate_wait_ok = backend_runtime::wait_decode_async_if_supported(
                    gpu_runtime.as_deref_mut(),
                    &mut gate_out,
                    "FFN gate",
                );
                if gate_wait_ok {
                    prof!("gate_gemv", t_gate);
                    let t_act = std::time::Instant::now();
                    apply_model_gate_mul_inplace(
                        &mut scratch.ffn_gate[..gate_rows],
                        &scratch.ffn_up[..gate_rows],
                        architecture,
                    );
                    prof!("silu_mul", t_act);
                    gpu_ffn_ok = true;
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    if !gpu_ffn_ok {
        if use_gemma_block_semantics(architecture) {
            let t_chain = std::time::Instant::now();
            if let Some(output) = backend_runtime::dense_q4k_gelu_ffn_if_supported(
                ffn_gate_weight,
                ffn_up_weight,
                ffn_down_weight,
                norm_data,
            )? {
                scratch.ffn_down[..hidden_dim].copy_from_slice(&output[..hidden_dim]);
                prof!("ffn_chain", t_chain);
                gpu_ffn_ok = true;
                gpu_down_done = true;
            }
        }
    }

    #[cfg(feature = "cuda")]
    if !gpu_ffn_ok {
        let gate_rows = ffn_gate_weight.rows;
        let t_gate = std::time::Instant::now();
        let gate_ok = backend_runtime::decode_gemv_into_if_supported(
            ffn_gate_weight,
            norm_data,
            &mut scratch.ffn_gate[..gate_rows],
            "FFN gate",
            false,
        )?;
        if gate_ok {
            prof!("gate_gemv", t_gate);
            let t_up = std::time::Instant::now();
            let up_ok = backend_runtime::decode_gemv_into_if_supported(
                ffn_up_weight,
                norm_data,
                &mut scratch.ffn_up[..gate_rows],
                "FFN up",
                false,
            )?;
            if up_ok {
                prof!("up_gemv", t_up);
                let t_act = std::time::Instant::now();
                apply_model_gate_mul_inplace(
                    &mut scratch.ffn_gate[..gate_rows],
                    &scratch.ffn_up[..gate_rows],
                    architecture,
                );
                prof!("silu_mul", t_act);
                gpu_ffn_ok = true;
            }
        }
    }

    if !gpu_ffn_ok {
        decode_ffn_gate_up_cpu_into(
            scratch,
            architecture,
            ffn_gate_weight,
            ffn_up_weight,
            ffn_gate_up_fused.as_ref(),
            hidden_dim,
            |label, t| prof!(label, t),
        )?;
    }

    let gate_rows = ffn_gate_weight.rows;
    let gpu_down_ok = gpu_down_done
        || backend_runtime::try_decode_gemv_if_supported(
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_deref_mut(),
            layer_idx,
            backend_runtime::DecodeProjectionKind::FfnDown,
            ffn_down_weight,
            hidden_dim,
            &scratch.ffn_gate[..gate_rows],
            &mut scratch.ffn_down[..hidden_dim],
            "FFN down",
        );

    let t_down = std::time::Instant::now();
    if !gpu_down_ok {
        ffn_down_weight.gemv_into(
            &scratch.ffn_gate[..gate_rows],
            &mut scratch.ffn_down[..hidden_dim],
        )?;
    }
    prof!("down_gemv", t_down);

    if let Some(post_ffw_norm) = post_ffw_norm_weight {
        let post_ffw_norm_data = kernels::tensor_as_f32_slice(post_ffw_norm);
        if super::models::muse_glimmer::uses_muse_glimmer_semantics(architecture) {
            apply_model_norm_into(
                &scratch.ffn_down[..hidden_dim],
                post_ffw_norm_data,
                post_norm_eps,
                &mut scratch.norm_buf2[..hidden_dim],
                architecture,
            );
            scratch.ffn_down[..hidden_dim].copy_from_slice(&scratch.norm_buf2[..hidden_dim]);
        } else if use_gemma_block_semantics(architecture) {
            apply_model_norm_into(
                &scratch.ffn_down[..hidden_dim],
                post_ffw_norm_data,
                norm_eps,
                &mut scratch.norm_buf2[..hidden_dim],
                architecture,
            );
            scratch.ffn_down[..hidden_dim].copy_from_slice(&scratch.norm_buf2[..hidden_dim]);
        }
    }

    let t_res = std::time::Instant::now();
    add_f32_inplace(
        &mut scratch.hidden[..hidden_dim],
        &scratch.ffn_down[..hidden_dim],
    );
    if verbose {
        prof!("residual_add", t_res);
    }

    Ok(())
}
