use super::super::mtp_verify::{
    MtpVerifyDeviceBuffers, Qwen35MtpDeviceVerifyAttentionKvState,
    Qwen35MtpDeviceVerifySsmLayerFinalState, Qwen35MtpGdnMoeLayerRequest,
    Qwen35MtpGdnProjectionRequest, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0,
};
use super::super::*;
use std::time::Instant;

#[derive(Default)]
struct Qwen35MtpDeviceVerifyStateCapture {
    prefix_states: Vec<Qwen35MtpDeviceVerifyPrefixState>,
    ssm_final_states: Vec<Qwen35MtpDeviceVerifySsmLayerFinalState>,
    attention_kv_states: Vec<Qwen35MtpDeviceVerifyAttentionKvState>,
}

fn mtp_verify_trace_enabled() -> bool {
    std::env::var("RNB_MTP_VERIFY_TRACE").ok().as_deref() == Some("1")
}

fn trace_elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn trace_mtp_verify_stage(
    enabled: bool,
    state: &mut CudaState,
    label: &str,
    start: Instant,
) -> Result<(), String> {
    if enabled {
        state.stream_synchronize()?;
        eprintln!(
            "[mtp-verify-trace] {label} {:.3}ms",
            trace_elapsed_ms(start)
        );
    }
    Ok(())
}

pub fn qwen35_mtp_device_verify_window(
    request: Qwen35MtpDeviceVerifyRequest<'_>,
) -> Result<Qwen35MtpDeviceVerifyResult, String> {
    validate_qwen35_mtp_device_verify_attention_layers(
        request.hidden_dim,
        request.attention_moe_layers,
    )?;
    validate_qwen35_mtp_device_verify_gdn_layers(request.hidden_dim, request.gdn_moe_layers)?;
    validate_qwen35_mtp_device_verify_layer_order(
        request.layer_order,
        request.attention_moe_layers,
        request.gdn_moe_layers,
    )?;
    let plan = qwen35_mtp_verify_buffer_plan(
        request.verify_tokens.len(),
        request.hidden_dim,
        request.prefix_tokens.len(),
    )?;
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let trace = mtp_verify_trace_enabled();
    let total_start = Instant::now();
    let buffers =
        state.stage_mtp_verify_window(&plan, request.verify_tokens, request.prefix_tokens)?;
    trace_mtp_verify_stage(trace, state, "stage_window", total_start)?;
    let stage_start = Instant::now();
    match request.token_embd_quant {
        GGML_Q4_K => state.stage_mtp_verify_token_embeddings_q4k(
            &buffers,
            request.token_embd_q4k,
            request.token_embd_rows,
            request.token_embd_cols,
            request.verify_tokens,
        )?,
        GGML_Q6_K => state.stage_mtp_verify_token_embeddings_q6k(
            &buffers,
            request.token_embd_q4k,
            request.token_embd_rows,
            request.token_embd_cols,
            request.verify_tokens,
        )?,
        other => {
            return Err(format!(
                "MTP verify token_embd quant must be Q4_K or Q6_K, got {other}"
            ));
        }
    }
    trace_mtp_verify_stage(trace, state, "token_embeddings", stage_start)?;
    let prefix_capture_tokens = request.prefix_tokens;
    let stage_start = Instant::now();
    let mut state_capture = stage_qwen35_mtp_device_verify_ordered_layers(
        state,
        &buffers,
        request.layer_order,
        request.attention_moe_layers,
        request.gdn_moe_layers,
        prefix_capture_tokens,
        request.rope_dim,
        request.rope_neox,
        request.rope_theta,
        request.pos_start,
        request.norm_eps,
    )?;
    trace_mtp_verify_stage(trace, state, "ordered_layers", stage_start)?;
    if request.layer_order.is_empty() {
        let stage_start = Instant::now();
        let gdn_state_capture = stage_qwen35_mtp_device_verify_gdn_moe_layers(
            state,
            &buffers,
            request.gdn_moe_layers,
            prefix_capture_tokens,
            request.norm_eps,
        )?;
        merge_qwen35_mtp_state_capture(&mut state_capture, gdn_state_capture);
        trace_mtp_verify_stage(trace, state, "gdn_layers", stage_start)?;
    }
    ensure_qwen35_mtp_prefix_placeholders(&mut state_capture, request.prefix_tokens);
    let stage_start = Instant::now();
    let output_argmax = match request.output_quant {
        GGML_Q4_K => state.stage_mtp_verify_output_argmax_q4k(
            &buffers,
            request.output_q6k,
            request.output_rows,
            request.output_cols,
            request.output_norm,
            request.norm_eps,
        ),
        GGML_Q6_K => state.stage_mtp_verify_output_argmax_q6k(
            &buffers,
            request.output_q6k,
            request.output_rows,
            request.output_cols,
            request.output_norm,
            request.norm_eps,
        ),
        GGML_Q8_0 => state.stage_mtp_verify_output_argmax_q8_0(
            &buffers,
            request.output_q6k,
            request.output_rows,
            request.output_cols,
            request.output_norm,
            request.norm_eps,
        ),
        other => Err(format!(
            "MTP verify output quant must be Q4_K, Q6_K or Q8_0, got {other}"
        )),
    };
    if let Err(err) = output_argmax {
        if let Err(free_err) =
            state.free_mtp_verify_prefix_state_snapshots(state_capture.prefix_states)
        {
            return Err(format!(
                "{err}; failed to free prefix snapshots: {free_err}"
            ));
        }
        return Err(err);
    }
    trace_mtp_verify_stage(trace, state, "output_argmax", stage_start)?;

    if request.attention_moe_layers.is_empty() && request.gdn_moe_layers.is_empty() {
        return Err(format!(
            "Qwen35 MTP device verify graph is not implemented: tokens={}, prefix_states={}, pos_start={}, include_bonus={}, gdn_moe_layers={}, gdn_moe_layers_staged={}, bytes={}, staged=true, embeddings_staged=true, output_argmax_staged=true",
            request.verify_tokens.len(),
            request.prefix_tokens.len(),
            request.pos_start,
            request.include_bonus,
            request.gdn_moe_layers.len(),
            request.gdn_moe_layers.len(),
            plan.total_device_bytes()
        ));
    }
    let stage_start = Instant::now();
    let mut result = match state.collect_mtp_verify_result(&plan) {
        Ok(result) => result,
        Err(err) => {
            if let Err(free_err) =
                state.free_mtp_verify_prefix_state_snapshots(state_capture.prefix_states)
            {
                return Err(format!(
                    "{err}; failed to free prefix snapshots: {free_err}"
                ));
            }
            return Err(err);
        }
    };
    trace_mtp_verify_stage(trace, state, "collect_result", stage_start)?;
    result.prefix_states = state_capture.prefix_states;
    result.ssm_final_states = state_capture.ssm_final_states;
    result.attention_kv_states = state_capture.attention_kv_states;
    trace_mtp_verify_stage(trace, state, "total", total_start)?;
    Ok(result)
}

fn validate_qwen35_mtp_device_verify_attention_layers(
    hidden_dim: usize,
    layers: &[Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>],
) -> Result<(), String> {
    for (idx, layer) in layers.iter().enumerate() {
        if layer.n_embd != hidden_dim {
            return Err(format!(
                "attention_moe_layers[{idx}] n_embd {} != hidden_dim {hidden_dim}",
                layer.n_embd
            ));
        }
        if layer.prior_tokens == 0 {
            if !layer.prior_k_bits.is_empty() || !layer.prior_v_bits.is_empty() {
                return Err(format!(
                    "attention_moe_layers[{idx}] prior K/V bits must be empty when prior_tokens=0"
                ));
            }
            continue;
        }
        if layer.k_rows != layer.v_rows {
            return Err(format!(
                "attention_moe_layers[{idx}] prior K/V rows mismatch: k_rows={} v_rows={}",
                layer.k_rows, layer.v_rows
            ));
        }
        let expected = layer
            .prior_tokens
            .checked_mul(layer.k_rows)
            .ok_or_else(|| {
                format!(
                    "attention_moe_layers[{idx}] prior K/V len overflow: tokens={} rows={}",
                    layer.prior_tokens, layer.k_rows
                )
            })?;
        if layer.prior_k_bits.len() != expected || layer.prior_v_bits.len() != expected {
            return Err(format!(
                "attention_moe_layers[{idx}] prior K/V len mismatch: k={} v={} expected={expected}",
                layer.prior_k_bits.len(),
                layer.prior_v_bits.len()
            ));
        }
    }
    Ok(())
}

fn validate_qwen35_mtp_device_verify_gdn_layers(
    hidden_dim: usize,
    layers: &[Qwen35MtpDeviceVerifyGdnMoeLayer<'_>],
) -> Result<(), String> {
    for (idx, layer) in layers.iter().enumerate() {
        if layer.n_embd != hidden_dim {
            return Err(format!(
                "gdn_moe_layers[{idx}] n_embd {} != hidden_dim {hidden_dim}",
                layer.n_embd
            ));
        }
    }
    Ok(())
}

fn validate_qwen35_mtp_device_verify_layer_order(
    layer_order: &[Qwen35MtpDeviceVerifyLayerKind],
    attention_moe_layers: &[Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>],
    gdn_moe_layers: &[Qwen35MtpDeviceVerifyGdnMoeLayer<'_>],
) -> Result<(), String> {
    for (idx, kind) in layer_order.iter().enumerate() {
        match *kind {
            Qwen35MtpDeviceVerifyLayerKind::AttentionMoe(layer_index) => {
                if layer_index >= attention_moe_layers.len() {
                    return Err(format!(
                        "layer_order[{idx}] attention index {layer_index} out of range {}",
                        attention_moe_layers.len()
                    ));
                }
            }
            Qwen35MtpDeviceVerifyLayerKind::GdnMoe(layer_index) => {
                if layer_index >= gdn_moe_layers.len() {
                    return Err(format!(
                        "layer_order[{idx}] GDN index {layer_index} out of range {}",
                        gdn_moe_layers.len()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn stage_qwen35_mtp_device_verify_ordered_layers(
    state: &mut CudaState,
    buffers: &MtpVerifyDeviceBuffers,
    layer_order: &[Qwen35MtpDeviceVerifyLayerKind],
    layers: &[Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>],
    gdn_layers: &mut [Qwen35MtpDeviceVerifyGdnMoeLayer<'_>],
    prefix_tokens: &[usize],
    rope_dim: usize,
    rope_neox: bool,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
) -> Result<Qwen35MtpDeviceVerifyStateCapture, String> {
    let mut state_capture = Qwen35MtpDeviceVerifyStateCapture::default();
    let trace = mtp_verify_trace_enabled();
    if layer_order.is_empty() {
        for layer in layers {
            let stage_start = Instant::now();
            let attention_kv = state
                .stage_mtp_verify_qwen35_attention_moe_layer_q4k_with_kv_state(
                    buffers, layer, rope_dim, rope_neox, rope_theta, pos_start, norm_eps,
                )?;
            trace_mtp_verify_stage(
                trace,
                state,
                &format!("layer attention:{}", layer.layer_index),
                stage_start,
            )?;
            state_capture.attention_kv_states.push(attention_kv);
        }
        return Ok(state_capture);
    }
    for kind in layer_order {
        match *kind {
            Qwen35MtpDeviceVerifyLayerKind::AttentionMoe(index) => {
                let stage_start = Instant::now();
                let attention_kv = state
                    .stage_mtp_verify_qwen35_attention_moe_layer_q4k_with_kv_state(
                        buffers,
                        &layers[index],
                        rope_dim,
                        rope_neox,
                        rope_theta,
                        pos_start,
                        norm_eps,
                    )?;
                trace_mtp_verify_stage(
                    trace,
                    state,
                    &format!("layer attention:{}", layers[index].layer_index),
                    stage_start,
                )?;
                state_capture.attention_kv_states.push(attention_kv);
            }
            Qwen35MtpDeviceVerifyLayerKind::GdnMoe(index) => {
                let stage_start = Instant::now();
                merge_qwen35_mtp_state_capture(
                    &mut state_capture,
                    stage_qwen35_mtp_device_verify_gdn_moe_layer(
                        state,
                        buffers,
                        &mut gdn_layers[index],
                        prefix_tokens,
                        norm_eps,
                    )?,
                );
                trace_mtp_verify_stage(
                    trace,
                    state,
                    &format!("layer gdn:{}", gdn_layers[index].layer_index),
                    stage_start,
                )?;
            }
        }
    }
    Ok(state_capture)
}

fn ensure_qwen35_mtp_prefix_placeholders(
    state_capture: &mut Qwen35MtpDeviceVerifyStateCapture,
    prefix_tokens: &[usize],
) {
    for &prefix_tokens in prefix_tokens {
        if !state_capture
            .prefix_states
            .iter()
            .any(|state| state.prefix_tokens == prefix_tokens)
        {
            state_capture
                .prefix_states
                .push(Qwen35MtpDeviceVerifyPrefixState {
                    prefix_tokens,
                    layers: Vec::new(),
                });
        }
    }
    state_capture
        .prefix_states
        .sort_by_key(|state| state.prefix_tokens);
}

fn merge_qwen35_mtp_state_capture(
    merged: &mut Qwen35MtpDeviceVerifyStateCapture,
    mut layer_capture: Qwen35MtpDeviceVerifyStateCapture,
) {
    for mut layer_prefix in layer_capture.prefix_states.drain(..) {
        if let Some(existing) = merged
            .prefix_states
            .iter_mut()
            .find(|state| state.prefix_tokens == layer_prefix.prefix_tokens)
        {
            existing.layers.append(&mut layer_prefix.layers);
        } else {
            merged.prefix_states.push(layer_prefix);
        }
    }
    merged
        .ssm_final_states
        .append(&mut layer_capture.ssm_final_states);
    merged
        .attention_kv_states
        .append(&mut layer_capture.attention_kv_states);
}

fn stage_qwen35_mtp_device_verify_gdn_moe_layers(
    state: &mut CudaState,
    buffers: &MtpVerifyDeviceBuffers,
    layers: &mut [Qwen35MtpDeviceVerifyGdnMoeLayer<'_>],
    prefix_tokens: &[usize],
    norm_eps: f32,
) -> Result<Qwen35MtpDeviceVerifyStateCapture, String> {
    let mut state_capture = Qwen35MtpDeviceVerifyStateCapture::default();
    for layer in layers {
        merge_qwen35_mtp_state_capture(
            &mut state_capture,
            stage_qwen35_mtp_device_verify_gdn_moe_layer(
                state,
                buffers,
                layer,
                prefix_tokens,
                norm_eps,
            )?,
        );
    }
    Ok(state_capture)
}

fn stage_qwen35_mtp_device_verify_gdn_moe_layer(
    state: &mut CudaState,
    buffers: &MtpVerifyDeviceBuffers,
    layer: &mut Qwen35MtpDeviceVerifyGdnMoeLayer<'_>,
    prefix_tokens: &[usize],
    norm_eps: f32,
) -> Result<Qwen35MtpDeviceVerifyStateCapture, String> {
    let capture = state.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_capture_states(
        buffers,
        layer.layer_index,
        Qwen35MtpGdnMoeLayerRequest {
            projection: Qwen35MtpGdnProjectionRequest {
                attn_norm: layer.attn_norm,
                qkv_q4k: layer.qkv_q4k,
                qkv_quant: layer.qkv_quant,
                qkv_rows: layer.qkv_rows,
                qkv_cols: layer.qkv_cols,
                gate_q4k: layer.gate_q4k,
                gate_rows: layer.gate_rows,
                gate_cols: layer.gate_cols,
                alpha_q4k: layer.alpha_q4k,
                alpha_f32: layer.alpha_f32,
                alpha_quant: layer.alpha_quant,
                alpha_rows: layer.alpha_rows,
                alpha_cols: layer.alpha_cols,
                beta_q4k: layer.beta_q4k,
                beta_f32: layer.beta_f32,
                beta_quant: layer.beta_quant,
                beta_rows: layer.beta_rows,
                beta_cols: layer.beta_cols,
                norm_eps,
            },
            conv_state: layer.conv_state,
            conv_kernel: layer.conv_kernel,
            kernel_size: layer.kernel_size,
            dt_bias: layer.dt_bias,
            ssm_a: layer.ssm_a,
            num_k_heads: layer.num_k_heads,
            num_v_heads: layer.num_v_heads,
            head_k_dim: layer.head_k_dim,
            head_v_dim: layer.head_v_dim,
            delta_state: layer.delta_state,
            sync_delta_state_to_host: layer.sync_delta_state_to_host,
            ssm_norm: layer.ssm_norm,
            ssm_out_q4k: layer.ssm_out_q4k,
            ssm_out_quant: layer.ssm_out_quant,
            ssm_out_rows: layer.ssm_out_rows,
            ssm_out_cols: layer.ssm_out_cols,
            post_attn_norm: layer.post_attn_norm,
            router_w: layer.router_w,
            n_expert: layer.n_expert,
            n_expert_used: layer.n_expert_used,
            gate_all: layer.gate_all,
            up_all: layer.up_all,
            down_all: layer.down_all,
            down_quant: layer.down_quant,
            shared_input_scale: layer.shared_input_scale,
            shared_gate: layer.shared_gate,
            shared_up: layer.shared_up,
            shared_down: layer.shared_down,
            shared_down_quant: layer.shared_down_quant,
            n_ff: layer.n_ff,
            n_embd: layer.n_embd,
            ffn_gate_q4k: layer.ffn_gate_q4k,
            ffn_gate_rows: layer.ffn_gate_rows,
            ffn_gate_cols: layer.ffn_gate_cols,
            ffn_up_q4k: layer.ffn_up_q4k,
            ffn_up_rows: layer.ffn_up_rows,
            ffn_up_cols: layer.ffn_up_cols,
            ffn_down: layer.ffn_down,
            ffn_down_quant: layer.ffn_down_quant,
            ffn_down_rows: layer.ffn_down_rows,
            ffn_down_cols: layer.ffn_down_cols,
            norm_eps,
        },
        prefix_tokens,
    )?;
    Ok(Qwen35MtpDeviceVerifyStateCapture {
        prefix_states: capture.prefix_states,
        ssm_final_states: vec![capture.final_state],
        attention_kv_states: Vec::new(),
    })
}
