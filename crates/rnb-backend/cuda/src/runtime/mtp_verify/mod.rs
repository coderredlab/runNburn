mod expert_cache;
mod profile;
mod types;
mod validation;

pub use types::{
    qwen35_mtp_verify_buffer_plan, MtpVerifyBufferPlan, Qwen35MtpDeviceDraftRequest,
    Qwen35MtpDeviceDraftResult, Qwen35MtpDeviceVerifyAttentionKvState,
    Qwen35MtpDeviceVerifyAttentionMoeLayer, Qwen35MtpDeviceVerifyGdnMoeLayer,
    Qwen35MtpDeviceVerifyLayerKind, Qwen35MtpDeviceVerifyPrefixState, Qwen35MtpDeviceVerifyRequest,
    Qwen35MtpDeviceVerifyResult, Qwen35MtpDeviceVerifySsmLayerFinalState,
    Qwen35MtpDeviceVerifySsmLayerPrefixState,
};
pub(in crate::runtime) use types::{
    MtpVerifyAttentionOutputBuffers, MtpVerifyAttentionQkNormRopeBuffers,
    MtpVerifyAttentionQkvProjectionBuffers, MtpVerifyDeviceBuffers, MtpVerifyGdnConvBuffers,
    MtpVerifyGdnDeltaInputBuffers, MtpVerifyGdnDeltaScanBuffers, MtpVerifyGdnProjectionBuffers,
    MtpVerifyGdnSsmOutBuffers, MtpVerifyQwen35RouterBuffers, Qwen35MtpAttentionOutputRequest,
    Qwen35MtpAttentionOutputWithPriorRequest, Qwen35MtpAttentionQkNormRopeRequest,
    Qwen35MtpAttentionQkvProjectionRequest, Qwen35MtpGdnMoeLayerRequest,
    Qwen35MtpGdnMoeLayerStateCapture, Qwen35MtpGdnProjectionRequest, GGML_F32, GGML_Q4_K,
    GGML_Q6_K, GGML_Q8_0,
};
use validation::{validate_mtp_verify_f32_matrix, validate_mtp_verify_q4k_matrix};
pub(in crate::runtime) use validation::{
    validate_mtp_verify_k_quant_matrix, validate_mtp_verify_prefix_tokens,
};

fn ensure_mtp_verify_prior_cache_buffer(
    api: &super::CudaApi,
    stream: usize,
    ptr: &mut Option<u64>,
    capacity: &mut usize,
    bytes: usize,
) -> Result<bool, String> {
    if ptr.is_some() && *capacity >= bytes {
        return Ok(false);
    }
    let new_capacity = bytes
        .checked_next_power_of_two()
        .ok_or_else(|| format!("MTP verify prior cache capacity overflow: bytes={bytes}"))?;
    let allocated = unsafe { api.mem_alloc(new_capacity) }?;
    if let Some(existing) = *ptr {
        unsafe {
            api.memcpy_dtod_async(allocated, existing, *capacity, stream)?;
            api.stream_synchronize(stream)?;
            api.mem_free(existing)?;
        }
    }
    *ptr = Some(allocated);
    *capacity = new_capacity;
    Ok(true)
}

fn mtp_verify_attention_window_kernel(head_dim: usize) -> Result<&'static str, String> {
    match head_dim {
        256 if std::env::var("RNB_CUDA_MTP_ATTN_FLASH_HD256_JBATCH8")
            .ok()
            .as_deref()
            != Some("0") =>
        {
            Ok("rnb_attention_prefill_flash_hd256_window_jbatch8")
        }
        256 if std::env::var("RNB_CUDA_ATTN_FLASH_HD256_JBATCH4")
            .ok()
            .as_deref()
            != Some("0") =>
        {
            Ok("rnb_attention_prefill_flash_hd256_window_jbatch4")
        }
        256 => Ok("rnb_attention_prefill_flash_hd256_window"),
        512 => Ok("rnb_attention_prefill_flash_hd512_window_w256"),
        other => Err(format!(
            "MTP verify attention output unsupported head_dim: {other}"
        )),
    }
}

impl super::CudaState {
    fn mtp_verify_subtrace_enabled() -> bool {
        std::env::var("RNB_MTP_VERIFY_SUBTRACE").ok().as_deref() == Some("1")
    }

    fn trace_mtp_verify_subphase(
        &mut self,
        enabled: bool,
        layer_kind: &str,
        layer_index: Option<usize>,
        phase: &str,
        start: std::time::Instant,
    ) -> Result<(), String> {
        if enabled {
            self.stream_synchronize()?;
            let layer = layer_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "?".to_string());
            eprintln!(
                "[mtp-verify-subtrace] layer {layer_kind}:{layer} {phase} {:.3}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    pub(in crate::runtime) fn ensure_mtp_verify_buffers(
        &mut self,
        plan: &MtpVerifyBufferPlan,
    ) -> Result<MtpVerifyDeviceBuffers, String> {
        self.set_current()?;
        let token_ids_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_token_ids,
            &mut self.mtp_verify_token_ids_capacity,
            plan.token_id_bytes,
        )?;
        let target_tokens_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_target_tokens,
            &mut self.mtp_verify_target_tokens_capacity,
            plan.target_token_bytes,
        )?;
        let hidden_rows_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_hidden_rows,
            &mut self.mtp_verify_hidden_rows_capacity,
            plan.hidden_row_bytes,
        )?;
        let scratch_hidden_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_scratch_hidden,
            &mut self.mtp_verify_scratch_hidden_capacity,
            plan.scratch_hidden_bytes,
        )?;
        let prefix_indices_dev = if plan.prefix_index_bytes == 0 {
            0
        } else {
            super::ensure_device_buffer(
                &self.api,
                &mut self.mtp_verify_prefix_indices,
                &mut self.mtp_verify_prefix_indices_capacity,
                plan.prefix_index_bytes,
            )?
        };

        Ok(MtpVerifyDeviceBuffers {
            token_ids_dev,
            target_tokens_dev,
            hidden_rows_dev,
            scratch_hidden_dev,
            prefix_indices_dev,
            plan: *plan,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_qwen35_router_buffers(
        &mut self,
        plan: &MtpVerifyBufferPlan,
        n_expert: usize,
        n_expert_used: usize,
    ) -> Result<MtpVerifyQwen35RouterBuffers, String> {
        self.set_current()?;
        if n_expert == 0 {
            return Err("MTP verify Qwen35 router requires at least one expert".to_string());
        }
        if n_expert_used == 0 {
            return Err(
                "MTP verify Qwen35 router requires at least one selected expert".to_string(),
            );
        }
        if n_expert_used > n_expert {
            return Err(format!(
                "MTP verify Qwen35 router selected expert count exceeds experts: used={n_expert_used}, experts={n_expert}"
            ));
        }
        if n_expert_used > 32 {
            return Err(format!(
                "MTP verify Qwen35 router top-k kernel supports at most 32 selected experts, got {n_expert_used}"
            ));
        }
        let logits_bytes = plan
            .window_tokens
            .checked_mul(n_expert)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify Qwen35 router logits buffer overflow: tokens={} experts={n_expert}",
                    plan.window_tokens
                )
            })?;
        let slots = plan
            .window_tokens
            .checked_mul(n_expert_used)
            .ok_or_else(|| {
                format!(
                    "MTP verify Qwen35 router slot count overflow: tokens={} used={n_expert_used}",
                    plan.window_tokens
                )
            })?;
        let slot_u32_bytes = slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                format!("MTP verify Qwen35 router slot id bytes overflow: slots={slots}")
            })?;
        let slot_f32_bytes = slots
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("MTP verify Qwen35 router route bytes overflow: slots={slots}")
            })?;
        let logits_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_router_logits,
            &mut self.mtp_verify_router_logits_capacity,
            logits_bytes,
        )?;
        let expert_ids_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_router_expert_ids,
            &mut self.mtp_verify_router_expert_ids_capacity,
            slot_u32_bytes,
        )?;
        let route_weights_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_router_route_weights,
            &mut self.mtp_verify_router_route_weights_capacity,
            slot_f32_bytes,
        )?;
        let token_ids_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_router_token_ids,
            &mut self.mtp_verify_router_token_ids_capacity,
            slot_u32_bytes,
        )?;

        Ok(MtpVerifyQwen35RouterBuffers {
            logits_dev,
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            window_tokens: plan.window_tokens,
            n_expert,
            n_expert_used,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_gdn_projection_buffers(
        &mut self,
        plan: &MtpVerifyBufferPlan,
        qkv_rows: usize,
        gate_rows: usize,
        alpha_rows: usize,
        beta_rows: usize,
    ) -> Result<MtpVerifyGdnProjectionBuffers, String> {
        self.set_current()?;
        let row_bytes = |label: &str, rows: usize| -> Result<usize, String> {
            if rows == 0 {
                return Err(format!("MTP verify GDN {label} rows must be non-zero"));
            }
            plan.window_tokens
                .checked_mul(rows)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!(
                        "MTP verify GDN {label} buffer size overflow: tokens={} rows={rows}",
                        plan.window_tokens
                    )
                })
        };
        let qkv_bytes = row_bytes("qkv", qkv_rows)?;
        let gate_bytes = row_bytes("gate", gate_rows)?;
        let alpha_bytes = row_bytes("alpha", alpha_rows)?;
        let beta_bytes = row_bytes("beta", beta_rows)?;
        let qkv_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_qkv,
            &mut self.mtp_verify_gdn_qkv_capacity,
            qkv_bytes,
        )?;
        let gate_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_gate,
            &mut self.mtp_verify_gdn_gate_capacity,
            gate_bytes,
        )?;
        let alpha_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_alpha,
            &mut self.mtp_verify_gdn_alpha_capacity,
            alpha_bytes,
        )?;
        let beta_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_beta,
            &mut self.mtp_verify_gdn_beta_capacity,
            beta_bytes,
        )?;

        Ok(MtpVerifyGdnProjectionBuffers {
            qkv_dev,
            gate_dev,
            alpha_dev,
            beta_dev,
            window_tokens: plan.window_tokens,
            qkv_rows,
            gate_rows,
            alpha_rows,
            beta_rows,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_attention_qkv_projection_buffers(
        &mut self,
        plan: &MtpVerifyBufferPlan,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
    ) -> Result<MtpVerifyAttentionQkvProjectionBuffers, String> {
        self.set_current()?;
        let row_bytes = |label: &str, rows: usize| -> Result<usize, String> {
            if rows == 0 {
                return Err(format!(
                    "MTP verify attention {label} rows must be non-zero"
                ));
            }
            plan.window_tokens
                .checked_mul(rows)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!(
                        "MTP verify attention {label} buffer size overflow: tokens={} rows={rows}",
                        plan.window_tokens
                    )
                })
        };
        let q_bytes = row_bytes("q", q_rows)?;
        let k_bytes = row_bytes("k", k_rows)?;
        let v_bytes = row_bytes("v", v_rows)?;
        let q_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_qkv,
            &mut self.mtp_verify_gdn_qkv_capacity,
            q_bytes,
        )?;
        let k_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_gate,
            &mut self.mtp_verify_gdn_gate_capacity,
            k_bytes,
        )?;
        let v_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_alpha,
            &mut self.mtp_verify_gdn_alpha_capacity,
            v_bytes,
        )?;

        Ok(MtpVerifyAttentionQkvProjectionBuffers {
            q_dev,
            k_dev,
            v_dev,
            window_tokens: plan.window_tokens,
            q_rows,
            k_rows,
            v_rows,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_attention_qk_norm_rope_buffers(
        &mut self,
        projection_buffers: &MtpVerifyAttentionQkvProjectionBuffers,
        q_rows: usize,
        head_dim: usize,
    ) -> Result<MtpVerifyAttentionQkNormRopeBuffers, String> {
        self.set_current()?;
        let q_bytes = projection_buffers
            .window_tokens
            .checked_mul(q_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention q postprocess buffer overflow: tokens={} rows={}",
                    projection_buffers.window_tokens, q_rows
                )
            })?;
        let kv_rows = projection_buffers.k_rows;
        if projection_buffers.v_rows != kv_rows {
            return Err(format!(
                "MTP verify attention k/v rows mismatch: k={} v={}",
                projection_buffers.k_rows, projection_buffers.v_rows
            ));
        }
        let kv_bits_bytes = projection_buffers
            .window_tokens
            .checked_mul(kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention kv postprocess buffer overflow: tokens={} rows={kv_rows}",
                    projection_buffers.window_tokens
                )
            })?;
        let q_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_q,
            &mut self.mtp_verify_gdn_delta_q_capacity,
            q_bytes,
        )?;
        let k_bits_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_k,
            &mut self.mtp_verify_gdn_delta_k_capacity,
            kv_bits_bytes,
        )?;
        let v_bits_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_v,
            &mut self.mtp_verify_gdn_delta_v_capacity,
            kv_bits_bytes,
        )?;

        Ok(MtpVerifyAttentionQkNormRopeBuffers {
            q_dev,
            gate_dev: None,
            k_bits_dev,
            v_bits_dev,
            window_tokens: projection_buffers.window_tokens,
            q_rows,
            kv_rows,
            head_dim,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_attention_gated_q_buffers(
        &mut self,
        window_tokens: usize,
        q_rows: usize,
    ) -> Result<(u64, u64), String> {
        let bytes = window_tokens
            .checked_mul(q_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention gated q buffer overflow: tokens={window_tokens} rows={q_rows}"
                )
            })?;
        let q_compact_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_attention_q_compact,
            &mut self.mtp_verify_attention_q_compact_capacity,
            bytes,
        )?;
        let gate_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_attention_gate,
            &mut self.mtp_verify_attention_gate_capacity,
            bytes,
        )?;
        Ok((q_compact_dev, gate_dev))
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_attention_output_buffers(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
    ) -> Result<MtpVerifyAttentionOutputBuffers, String> {
        self.ensure_mtp_verify_attention_output_buffers_for_kv_tokens(
            post_buffers,
            post_buffers.window_tokens,
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_attention_output_buffers_for_kv_tokens(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        kv_tokens: usize,
    ) -> Result<MtpVerifyAttentionOutputBuffers, String> {
        self.set_current()?;
        let output_bytes = post_buffers
            .window_tokens
            .checked_mul(post_buffers.q_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output buffer overflow: tokens={} rows={}",
                    post_buffers.window_tokens, post_buffers.q_rows
                )
            })?;
        let kv_f32_bytes = kv_tokens
            .checked_mul(post_buffers.kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention kv f32 buffer overflow: tokens={} rows={}",
                    kv_tokens, post_buffers.kv_rows
                )
            })?;
        let k_f32_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_attention_k_f32,
            &mut self.mtp_verify_attention_k_f32_capacity,
            kv_f32_bytes,
        )?;
        let v_f32_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_attention_v_f32,
            &mut self.mtp_verify_attention_v_f32_capacity,
            kv_f32_bytes,
        )?;
        let attn_out_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_attention_out,
            &mut self.mtp_verify_attention_out_capacity,
            output_bytes,
        )?;

        Ok(MtpVerifyAttentionOutputBuffers {
            k_f32_dev,
            v_f32_dev,
            attn_out_dev,
            window_tokens: post_buffers.window_tokens,
            q_rows: post_buffers.q_rows,
            kv_rows: post_buffers.kv_rows,
            head_dim: post_buffers.head_dim,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_prior_kv_bits(
        &mut self,
        prior_k_bits: &[u16],
        prior_v_bits: &[u16],
        prior_tokens: usize,
        kv_rows: usize,
    ) -> Result<Option<(u64, u64)>, String> {
        self.stage_mtp_verify_attention_prior_kv_bits_for_layer(
            usize::MAX,
            0,
            prior_k_bits,
            prior_v_bits,
            prior_tokens,
            kv_rows,
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_prior_kv_bits_for_layer(
        &mut self,
        layer_index: usize,
        sequence_epoch: u64,
        prior_k_bits: &[u16],
        prior_v_bits: &[u16],
        prior_tokens: usize,
        kv_rows: usize,
    ) -> Result<Option<(u64, u64)>, String> {
        self.set_current()?;
        if prior_tokens == 0 {
            if !prior_k_bits.is_empty() || !prior_v_bits.is_empty() {
                return Err(format!(
                    "MTP verify attention prior K/V bits must be empty when prior_tokens=0: k={} v={}",
                    prior_k_bits.len(),
                    prior_v_bits.len()
                ));
            }
            return Ok(None);
        }
        if kv_rows == 0 {
            return Err("MTP verify attention prior K/V kv_rows must be non-zero".to_string());
        }
        let prior_values = prior_tokens.checked_mul(kv_rows).ok_or_else(|| {
            format!(
                "MTP verify attention prior K/V value overflow: tokens={prior_tokens} rows={kv_rows}"
            )
        })?;
        let prior_bytes = prior_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("MTP verify attention prior K/V byte overflow: values={prior_values}")
            })?;
        let persistent_prior_source = if prior_k_bits.is_empty()
            && prior_v_bits.is_empty()
            && crate::tuning::mtp_verify_resident_attn_kv_enabled()
        {
            self.persistent_decode_ctx.as_ref().and_then(|ctx| {
                let k_dev = *ctx.k_cache_devs.get(layer_index)?;
                let v_dev = *ctx.v_cache_devs.get(layer_index)?;
                (k_dev != 0
                    && v_dev != 0
                    && prior_tokens <= ctx.max_seq_len as usize
                    && prior_tokens <= ctx.resident_kv_tokens)
                    .then_some((k_dev, v_dev))
            })
        } else {
            None
        };
        let cache_index = match self
            .mtp_verify_attention_prior_kv
            .iter()
            .position(|cache| cache.layer_index == layer_index)
        {
            Some(index) => index,
            None => {
                self.mtp_verify_attention_prior_kv
                    .push(super::MtpVerifyAttentionPriorKvCache {
                        layer_index,
                        kv_rows,
                        ..Default::default()
                    });
                self.mtp_verify_attention_prior_kv.len() - 1
            }
        };

        let cache = &mut self.mtp_verify_attention_prior_kv[cache_index];
        if cache.kv_rows != kv_rows {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr) }
                    .map_err(|err| format!("MTP verify prior K cache free failed: {err}"))?;
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr) }
                    .map_err(|err| format!("MTP verify prior V cache free failed: {err}"))?;
            }
            cache.k_bits_capacity = 0;
            cache.v_bits_capacity = 0;
            cache.cached_tokens = 0;
            cache.host_k_bits.clear();
            cache.host_v_bits.clear();
            cache.kv_rows = kv_rows;
        }
        if cache.sequence_epoch != sequence_epoch {
            cache.cached_tokens = 0;
            cache.host_k_bits.clear();
            cache.host_v_bits.clear();
            cache.sequence_epoch = sequence_epoch;
        }
        let host_kv_complete =
            prior_k_bits.len() == prior_values && prior_v_bits.len() == prior_values;
        if !host_kv_complete {
            if prior_k_bits.is_empty()
                && prior_v_bits.is_empty()
                && crate::tuning::mtp_verify_resident_attn_kv_enabled()
            {
                if cache.cached_tokens >= prior_tokens {
                    let prior_k_dev = cache
                        .k_bits_dev
                        .ok_or_else(|| "MTP verify resident prior K buffer missing".to_string())?;
                    let prior_v_dev = cache
                        .v_bits_dev
                        .ok_or_else(|| "MTP verify resident prior V buffer missing".to_string())?;
                    cache.cached_tokens = prior_tokens;
                    return Ok(Some((prior_k_dev, prior_v_dev)));
                }
                if let Some((source_k_dev, source_v_dev)) = persistent_prior_source {
                    ensure_mtp_verify_prior_cache_buffer(
                        &self.api,
                        self.stream,
                        &mut cache.k_bits_dev,
                        &mut cache.k_bits_capacity,
                        prior_bytes,
                    )?;
                    ensure_mtp_verify_prior_cache_buffer(
                        &self.api,
                        self.stream,
                        &mut cache.v_bits_dev,
                        &mut cache.v_bits_capacity,
                        prior_bytes,
                    )?;
                    let prior_k_dev = cache
                        .k_bits_dev
                        .ok_or_else(|| "MTP verify prior K device buffer missing".to_string())?;
                    let prior_v_dev = cache
                        .v_bits_dev
                        .ok_or_else(|| "MTP verify prior V device buffer missing".to_string())?;
                    unsafe {
                        self.api.memcpy_dtod_async(
                            prior_k_dev,
                            source_k_dev,
                            prior_bytes,
                            self.stream,
                        )?;
                        self.api.memcpy_dtod_async(
                            prior_v_dev,
                            source_v_dev,
                            prior_bytes,
                            self.stream,
                        )?;
                    }
                    cache.cached_tokens = prior_tokens;
                    cache.host_k_bits.clear();
                    cache.host_v_bits.clear();
                    return Ok(Some((prior_k_dev, prior_v_dev)));
                }
            }
            return Err(format!(
                "MTP verify attention layer {layer_index} prior K/V len mismatch without a resident source: k={} v={} expected={prior_values}",
                prior_k_bits.len(),
                prior_v_bits.len()
            ));
        }

        let k_reallocated = ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.k_bits_dev,
            &mut cache.k_bits_capacity,
            prior_bytes,
        )?;
        let v_reallocated = ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.v_bits_dev,
            &mut cache.v_bits_capacity,
            prior_bytes,
        )?;
        let prior_k_dev = cache
            .k_bits_dev
            .ok_or_else(|| "MTP verify prior K device buffer missing".to_string())?;
        let prior_v_dev = cache
            .v_bits_dev
            .ok_or_else(|| "MTP verify prior V device buffer missing".to_string())?;
        if crate::tuning::mtp_verify_resident_attn_kv_enabled()
            && cache.cached_tokens >= prior_tokens
        {
            cache.cached_tokens = prior_tokens;
            return Ok(Some((prior_k_dev, prior_v_dev)));
        }
        let common_values = cache.host_k_bits.len().min(prior_values);
        let prefix_matches = !k_reallocated
            && !v_reallocated
            && cache.host_k_bits.len() >= common_values
            && cache.host_v_bits.len() >= common_values
            && cache.host_k_bits[..common_values] == prior_k_bits[..common_values]
            && cache.host_v_bits[..common_values] == prior_v_bits[..common_values];
        let upload_start_values = if prefix_matches { common_values } else { 0 };
        if upload_start_values < prior_values {
            let upload_offset_bytes = upload_start_values
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| {
                    format!(
                        "MTP verify prior K/V upload offset overflow: values={upload_start_values}"
                    )
                })?;
            let upload_bytes = prior_bytes.checked_sub(upload_offset_bytes).ok_or_else(|| {
                format!(
                    "MTP verify prior K/V upload byte underflow: total={prior_bytes} offset={upload_offset_bytes}"
                )
            })?;
            let upload_offset_bytes_u64 = u64::try_from(upload_offset_bytes).map_err(|_| {
                format!("MTP verify prior K/V upload offset exceeds u64: {upload_offset_bytes}")
            })?;
            let prior_k_dst = prior_k_dev
                .checked_add(upload_offset_bytes_u64)
                .ok_or_else(|| "MTP verify prior K upload pointer overflow".to_string())?;
            let prior_v_dst = prior_v_dev
                .checked_add(upload_offset_bytes_u64)
                .ok_or_else(|| "MTP verify prior V upload pointer overflow".to_string())?;
            unsafe {
                self.api
                    .memcpy_htod_async(
                        prior_k_dst,
                        prior_k_bits[upload_start_values..]
                            .as_ptr()
                            .cast::<libc::c_void>(),
                        upload_bytes,
                        self.stream,
                    )
                    .map_err(|err| format!("MTP verify prior K upload failed: {err}"))?;
                self.api
                    .memcpy_htod_async(
                        prior_v_dst,
                        prior_v_bits[upload_start_values..]
                            .as_ptr()
                            .cast::<libc::c_void>(),
                        upload_bytes,
                        self.stream,
                    )
                    .map_err(|err| format!("MTP verify prior V upload failed: {err}"))?;
            }
        }
        cache.host_k_bits.clear();
        cache.host_k_bits.extend_from_slice(prior_k_bits);
        cache.host_v_bits.clear();
        cache.host_v_bits.extend_from_slice(prior_v_bits);
        cache.cached_tokens = prior_tokens;
        Ok(Some((prior_k_dev, prior_v_dev)))
    }

    fn retain_mtp_verify_attention_window_kv_for_layer(
        &mut self,
        layer_index: usize,
        sequence_epoch: u64,
        prior_tokens: usize,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
    ) -> Result<(), String> {
        let cache_index = match self
            .mtp_verify_attention_prior_kv
            .iter()
            .position(|cache| cache.layer_index == layer_index)
        {
            Some(index) => index,
            None => {
                self.mtp_verify_attention_prior_kv
                    .push(super::MtpVerifyAttentionPriorKvCache {
                        layer_index,
                        sequence_epoch,
                        kv_rows: post_buffers.kv_rows,
                        ..Default::default()
                    });
                self.mtp_verify_attention_prior_kv.len() - 1
            }
        };
        let cache = &mut self.mtp_verify_attention_prior_kv[cache_index];
        if cache.sequence_epoch != sequence_epoch {
            cache.cached_tokens = 0;
            cache.host_k_bits.clear();
            cache.host_v_bits.clear();
            cache.sequence_epoch = sequence_epoch;
        }
        if cache.kv_rows != post_buffers.kv_rows {
            return Err(format!(
                "MTP verify resident attention K/V row mismatch for layer {layer_index}: cache={} current={}",
                cache.kv_rows, post_buffers.kv_rows
            ));
        }
        let total_tokens = prior_tokens
            .checked_add(post_buffers.window_tokens)
            .ok_or_else(|| "MTP verify resident attention token overflow".to_string())?;
        let total_bytes = total_tokens
            .checked_mul(post_buffers.kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "MTP verify resident attention byte overflow".to_string())?;
        ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.k_bits_dev,
            &mut cache.k_bits_capacity,
            total_bytes,
        )?;
        ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.v_bits_dev,
            &mut cache.v_bits_capacity,
            total_bytes,
        )?;
        let offset_bytes = prior_tokens
            .checked_mul(post_buffers.kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "MTP verify resident attention offset overflow".to_string())?;
        let window_bytes = post_buffers
            .window_tokens
            .checked_mul(post_buffers.kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "MTP verify resident attention window byte overflow".to_string())?;
        let k_dst = cache
            .k_bits_dev
            .and_then(|ptr| ptr.checked_add(offset_bytes as u64))
            .ok_or_else(|| "MTP verify resident K pointer overflow".to_string())?;
        let v_dst = cache
            .v_bits_dev
            .and_then(|ptr| ptr.checked_add(offset_bytes as u64))
            .ok_or_else(|| "MTP verify resident V pointer overflow".to_string())?;
        unsafe {
            self.api.memcpy_dtod_async(
                k_dst,
                post_buffers.k_bits_dev,
                window_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtod_async(
                v_dst,
                post_buffers.v_bits_dev,
                window_bytes,
                self.stream,
            )?;
        }
        cache.cached_tokens = total_tokens;
        Ok(())
    }

    fn retain_mtp_verify_attention_shared_window_kv(
        &mut self,
        layer_index: usize,
        sequence_epoch: u64,
        pos_start: usize,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
    ) -> Result<(), String> {
        let cache_index = match self
            .mtp_verify_attention_shared_window_kv
            .iter()
            .position(|cache| cache.layer_index == layer_index)
        {
            Some(index) => index,
            None => {
                self.mtp_verify_attention_shared_window_kv.push(
                    super::MtpVerifyAttentionSharedWindowKvCache {
                        layer_index,
                        ..Default::default()
                    },
                );
                self.mtp_verify_attention_shared_window_kv.len() - 1
            }
        };
        let cache = &mut self.mtp_verify_attention_shared_window_kv[cache_index];
        let window_bytes = post_buffers
            .window_tokens
            .checked_mul(post_buffers.kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "MTP verify shared attention window byte overflow".to_string())?;
        ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.k_bits_dev,
            &mut cache.k_bits_capacity,
            window_bytes,
        )?;
        ensure_mtp_verify_prior_cache_buffer(
            &self.api,
            self.stream,
            &mut cache.v_bits_dev,
            &mut cache.v_bits_capacity,
            window_bytes,
        )?;
        let k_bits_dev = cache
            .k_bits_dev
            .ok_or_else(|| "MTP verify shared attention K buffer missing".to_string())?;
        let v_bits_dev = cache
            .v_bits_dev
            .ok_or_else(|| "MTP verify shared attention V buffer missing".to_string())?;
        unsafe {
            self.api.memcpy_dtod_async(
                k_bits_dev,
                post_buffers.k_bits_dev,
                window_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtod_async(
                v_bits_dev,
                post_buffers.v_bits_dev,
                window_bytes,
                self.stream,
            )?;
        }
        cache.kv_rows = post_buffers.kv_rows;
        cache.pos_start = pos_start;
        cache.window_tokens = post_buffers.window_tokens;
        cache.sequence_epoch = sequence_epoch;
        Ok(())
    }

    fn mtp_verify_attention_shared_window_kv(
        &self,
        layer_index: usize,
        sequence_epoch: u64,
        pos_start: usize,
        window_tokens: usize,
        kv_rows: usize,
    ) -> Result<(u64, u64), String> {
        let cache = self
            .mtp_verify_attention_shared_window_kv
            .iter()
            .find(|cache| cache.layer_index == layer_index)
            .ok_or_else(|| {
                format!(
                    "MTP verify shared attention source layer {layer_index} window is unavailable"
                )
            })?;
        if cache.sequence_epoch != sequence_epoch
            || cache.pos_start != pos_start
            || cache.window_tokens != window_tokens
            || cache.kv_rows != kv_rows
        {
            return Err(format!(
                "MTP verify shared attention source layer {layer_index} window mismatch: epoch={}/{} pos={}/{} tokens={}/{} rows={}/{}",
                cache.sequence_epoch,
                sequence_epoch,
                cache.pos_start,
                pos_start,
                cache.window_tokens,
                window_tokens,
                cache.kv_rows,
                kv_rows
            ));
        }
        Ok((
            cache
                .k_bits_dev
                .ok_or_else(|| "MTP verify shared attention K buffer missing".to_string())?,
            cache
                .v_bits_dev
                .ok_or_else(|| "MTP verify shared attention V buffer missing".to_string())?,
        ))
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_gdn_conv_buffers(
        &mut self,
        window_tokens: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<MtpVerifyGdnConvBuffers, String> {
        self.set_current()?;
        if window_tokens == 0 {
            return Err("MTP verify GDN conv window_tokens must be non-zero".to_string());
        }
        if channels == 0 {
            return Err("MTP verify GDN conv channels must be non-zero".to_string());
        }
        let state_rows = kernel_size
            .checked_sub(1)
            .ok_or_else(|| "MTP verify GDN conv kernel_size must be non-zero".to_string())?;
        let conv_state_bytes = state_rows
            .checked_mul(channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN conv state size overflow: channels={channels}, kernel_size={kernel_size}"
                )
            })?;
        let conv_input_bytes = window_tokens
            .checked_add(state_rows)
            .and_then(|rows| rows.checked_mul(channels))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN conv input size overflow: tokens={window_tokens}, channels={channels}, kernel_size={kernel_size}"
                )
            })?;
        let conv_out_bytes = window_tokens
            .checked_mul(channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN conv output size overflow: tokens={window_tokens}, channels={channels}"
                )
            })?;
        let conv_state_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_conv_state,
            &mut self.mtp_verify_gdn_conv_state_capacity,
            conv_state_bytes,
        )?;
        let conv_input_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_conv_input,
            &mut self.mtp_verify_gdn_conv_input_capacity,
            conv_input_bytes,
        )?;
        let conv_out_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_conv_out,
            &mut self.mtp_verify_gdn_conv_out_capacity,
            conv_out_bytes,
        )?;

        Ok(MtpVerifyGdnConvBuffers {
            conv_state_dev,
            device_resident_state: false,
            conv_input_dev,
            conv_out_dev,
            window_tokens,
            channels,
            kernel_size,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_gdn_delta_input_buffers(
        &mut self,
        window_tokens: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<MtpVerifyGdnDeltaInputBuffers, String> {
        self.set_current()?;
        if window_tokens == 0 {
            return Err("MTP verify GDN delta window_tokens must be non-zero".to_string());
        }
        if num_heads == 0 {
            return Err("MTP verify GDN delta num_heads must be non-zero".to_string());
        }
        if head_k_dim == 0 || head_v_dim == 0 {
            return Err(format!(
                "MTP verify GDN delta head dims must be non-zero: head_k_dim={head_k_dim}, head_v_dim={head_v_dim}"
            ));
        }
        let qk_bytes = window_tokens
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(head_k_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta q/k size overflow: tokens={window_tokens}, heads={num_heads}, head_k_dim={head_k_dim}"
                )
            })?;
        let v_bytes = window_tokens
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(head_v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta v size overflow: tokens={window_tokens}, heads={num_heads}, head_v_dim={head_v_dim}"
                )
            })?;
        let gate_bytes = window_tokens
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta gate size overflow: tokens={window_tokens}, heads={num_heads}"
                )
            })?;
        let q_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_q,
            &mut self.mtp_verify_gdn_delta_q_capacity,
            qk_bytes,
        )?;
        let k_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_k,
            &mut self.mtp_verify_gdn_delta_k_capacity,
            qk_bytes,
        )?;
        let v_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_v,
            &mut self.mtp_verify_gdn_delta_v_capacity,
            v_bytes,
        )?;
        let gate_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_gate,
            &mut self.mtp_verify_gdn_delta_gate_capacity,
            gate_bytes,
        )?;
        let beta_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_beta,
            &mut self.mtp_verify_gdn_delta_beta_capacity,
            gate_bytes,
        )?;

        Ok(MtpVerifyGdnDeltaInputBuffers {
            q_dev,
            k_dev,
            v_dev,
            gate_dev,
            beta_dev,
            window_tokens,
            num_heads,
            head_k_dim,
            head_v_dim,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_gdn_delta_scan_buffers(
        &mut self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
    ) -> Result<MtpVerifyGdnDeltaScanBuffers, String> {
        self.set_current()?;
        let output_bytes = delta_buffers
            .window_tokens
            .checked_mul(delta_buffers.num_heads)
            .and_then(|values| values.checked_mul(delta_buffers.head_v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta output size overflow: tokens={}, heads={}, head_v_dim={}",
                    delta_buffers.window_tokens, delta_buffers.num_heads, delta_buffers.head_v_dim
                )
            })?;
        let output_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_delta_out,
            &mut self.mtp_verify_gdn_delta_out_capacity,
            output_bytes,
        )?;

        Ok(MtpVerifyGdnDeltaScanBuffers {
            output_dev,
            window_tokens: delta_buffers.window_tokens,
            num_heads: delta_buffers.num_heads,
            head_v_dim: delta_buffers.head_v_dim,
        })
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn ensure_mtp_verify_gdn_ssm_out_buffers(
        &mut self,
        window_tokens: usize,
        d_inner: usize,
        hidden_dim: usize,
    ) -> Result<MtpVerifyGdnSsmOutBuffers, String> {
        self.set_current()?;
        if window_tokens == 0 || d_inner == 0 || hidden_dim == 0 {
            return Err(format!(
                "MTP verify GDN ssm_out dimensions must be non-zero: tokens={window_tokens}, d_inner={d_inner}, hidden_dim={hidden_dim}"
            ));
        }
        let gated_bytes = window_tokens
            .checked_mul(d_inner)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN gated buffer size overflow: tokens={window_tokens}, d_inner={d_inner}"
                )
            })?;
        let ssm_out_bytes = window_tokens
            .checked_mul(hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN ssm_out buffer size overflow: tokens={window_tokens}, hidden_dim={hidden_dim}"
                )
            })?;
        let gated_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_gated,
            &mut self.mtp_verify_gdn_gated_capacity,
            gated_bytes,
        )?;
        let ssm_out_dev = super::ensure_device_buffer(
            &self.api,
            &mut self.mtp_verify_gdn_ssm_out,
            &mut self.mtp_verify_gdn_ssm_out_capacity,
            ssm_out_bytes,
        )?;

        Ok(MtpVerifyGdnSsmOutBuffers {
            gated_dev,
            ssm_out_dev,
            window_tokens,
            hidden_dim,
            d_inner,
        })
    }

    pub(in crate::runtime) fn stage_mtp_verify_window(
        &mut self,
        plan: &MtpVerifyBufferPlan,
        verify_tokens: &[u32],
        prefix_tokens: &[usize],
    ) -> Result<MtpVerifyDeviceBuffers, String> {
        if verify_tokens.len() != plan.window_tokens {
            return Err(format!(
                "MTP verify token count mismatch: got {}, expected {}",
                verify_tokens.len(),
                plan.window_tokens
            ));
        }
        if prefix_tokens.len() != plan.prefix_count {
            return Err(format!(
                "MTP verify prefix count mismatch: got {}, expected {}",
                prefix_tokens.len(),
                plan.prefix_count
            ));
        }
        validate_mtp_verify_prefix_tokens(plan.window_tokens, prefix_tokens)?;

        let prefix_indices = prefix_tokens
            .iter()
            .copied()
            .map(|token| {
                u32::try_from(token).map_err(|_| {
                    format!("MTP verify prefix token index exceeds CUDA u32 limit: {token}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = self.ensure_mtp_verify_buffers(plan)?;
        self.set_current()?;
        unsafe {
            self.api.memcpy_htod_async(
                buffers.token_ids_dev,
                verify_tokens.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(verify_tokens),
                self.stream,
            )?;
            if !prefix_indices.is_empty() {
                self.api.memcpy_htod_async(
                    buffers.prefix_indices_dev,
                    prefix_indices.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(prefix_indices.as_slice()),
                    self.stream,
                )?;
            }
        }
        Ok(buffers)
    }

    pub(in crate::runtime) fn stage_mtp_verify_token_embeddings_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        token_embd_q4k: &[u8],
        token_embd_rows: usize,
        token_embd_cols: usize,
        verify_tokens: &[u32],
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if token_embd_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify token_embd cols must match hidden_dim: cols={}, hidden_dim={}",
                token_embd_cols, plan.hidden_dim
            ));
        }
        if token_embd_cols == 0 || token_embd_cols % 256 != 0 {
            return Err(format!(
                "MTP verify Q4_K token_embd cols must be non-zero and divisible by 256, got {token_embd_cols}"
            ));
        }
        if verify_tokens.len() != plan.window_tokens {
            return Err(format!(
                "MTP verify embedding token count mismatch: got {}, expected {}",
                verify_tokens.len(),
                plan.window_tokens
            ));
        }
        for &token_id in verify_tokens {
            let token_idx = usize::try_from(token_id)
                .map_err(|_| format!("MTP verify token id exceeds usize: {token_id}"))?;
            if token_idx >= token_embd_rows {
                return Err(format!(
                    "MTP verify token id out of token_embd range: token={token_idx}, rows={token_embd_rows}"
                ));
            }
        }
        let blocks_per_row = token_embd_cols / 256;
        let expected = token_embd_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q4_K token_embd byte size overflow: rows={token_embd_rows} cols={token_embd_cols}"
                )
            })?;
        if token_embd_q4k.len() != expected {
            return Err(format!(
                "MTP verify Q4_K token_embd byte mismatch: got {}, expected {expected}",
                token_embd_q4k.len()
            ));
        }

        self.launch_q4k_embedding_gather_to_dev_pinned(
            token_embd_q4k,
            token_embd_rows,
            blocks_per_row,
            buffers.token_ids_dev,
            plan.window_tokens,
            buffers.hidden_rows_dev,
        )
    }

    pub(in crate::runtime) fn stage_mtp_verify_token_embeddings_q6k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        token_embd_q6k: &[u8],
        token_embd_rows: usize,
        token_embd_cols: usize,
        verify_tokens: &[u32],
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if token_embd_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify token_embd cols must match hidden_dim: cols={}, hidden_dim={}",
                token_embd_cols, plan.hidden_dim
            ));
        }
        if token_embd_cols == 0 || token_embd_cols % 256 != 0 {
            return Err(format!(
                "MTP verify Q6_K token_embd cols must be non-zero and divisible by 256, got {token_embd_cols}"
            ));
        }
        if verify_tokens.len() != plan.window_tokens {
            return Err(format!(
                "MTP verify embedding token count mismatch: got {}, expected {}",
                verify_tokens.len(),
                plan.window_tokens
            ));
        }
        for &token_id in verify_tokens {
            let token_idx = usize::try_from(token_id)
                .map_err(|_| format!("MTP verify token id exceeds usize: {token_id}"))?;
            if token_idx >= token_embd_rows {
                return Err(format!(
                    "MTP verify token id out of token_embd range: token={token_idx}, rows={token_embd_rows}"
                ));
            }
        }
        let blocks_per_row = token_embd_cols / 256;
        let expected = token_embd_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(210))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q6_K token_embd byte size overflow: rows={token_embd_rows} cols={token_embd_cols}"
                )
            })?;
        if token_embd_q6k.len() != expected {
            return Err(format!(
                "MTP verify Q6_K token_embd byte mismatch: got {}, expected {expected}",
                token_embd_q6k.len()
            ));
        }

        self.launch_q6k_embedding_gather_to_dev_pinned(
            token_embd_q6k,
            token_embd_rows,
            blocks_per_row,
            buffers.token_ids_dev,
            plan.window_tokens,
            buffers.hidden_rows_dev,
        )
    }

    pub(in crate::runtime) fn stage_mtp_verify_token_embeddings_q8_0(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        token_embd_q8_0: &[u8],
        token_embd_rows: usize,
        token_embd_cols: usize,
        verify_tokens: &[u32],
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if token_embd_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify token_embd cols must match hidden_dim: cols={}, hidden_dim={}",
                token_embd_cols, plan.hidden_dim
            ));
        }
        if token_embd_cols == 0 || token_embd_cols % 32 != 0 {
            return Err(format!(
                "MTP verify Q8_0 token_embd cols must be non-zero and divisible by 32, got {token_embd_cols}"
            ));
        }
        if verify_tokens.len() != plan.window_tokens {
            return Err(format!(
                "MTP verify embedding token count mismatch: got {}, expected {}",
                verify_tokens.len(),
                plan.window_tokens
            ));
        }
        for &token_id in verify_tokens {
            let token_idx = usize::try_from(token_id)
                .map_err(|_| format!("MTP verify token id exceeds usize: {token_id}"))?;
            if token_idx >= token_embd_rows {
                return Err(format!(
                    "MTP verify token id out of token_embd range: token={token_idx}, rows={token_embd_rows}"
                ));
            }
        }
        let blocks_per_row = token_embd_cols / 32;
        let expected = token_embd_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(34))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q8_0 token_embd byte size overflow: rows={token_embd_rows} cols={token_embd_cols}"
                )
            })?;
        if token_embd_q8_0.len() != expected {
            return Err(format!(
                "MTP verify Q8_0 token_embd byte mismatch: got {}, expected {expected}",
                token_embd_q8_0.len()
            ));
        }

        self.launch_q8_0_embedding_gather_to_dev_pinned(
            token_embd_q8_0,
            token_embd_rows,
            blocks_per_row,
            buffers.token_ids_dev,
            plan.window_tokens,
            buffers.hidden_rows_dev,
        )
    }

    pub(in crate::runtime) fn stage_mtp_verify_output_argmax_q6k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        output_q6k: &[u8],
        output_rows: usize,
        output_cols: usize,
        output_norm: &[f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if output_rows == 0 {
            return Err("MTP verify output rows must be non-zero".to_string());
        }
        if output_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify output cols must match hidden_dim: cols={}, hidden_dim={}",
                output_cols, plan.hidden_dim
            ));
        }
        if output_cols == 0 || output_cols % 256 != 0 {
            return Err(format!(
                "MTP verify Q6_K output cols must be non-zero and divisible by 256, got {output_cols}"
            ));
        }
        if output_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify output_norm length mismatch: got {}, expected {}",
                output_norm.len(),
                plan.hidden_dim
            ));
        }
        let blocks_per_row = output_cols / 256;
        let expected = output_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(210))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q6_K output byte size overflow: rows={output_rows} cols={output_cols}"
                )
            })?;
        if output_q6k.len() != expected {
            return Err(format!(
                "MTP verify Q6_K output byte mismatch: got {}, expected {expected}",
                output_q6k.len()
            ));
        }

        self.stage_mtp_verify_hidden_rows_rms_norm(buffers, output_norm, norm_eps, false)?;
        self.write_q6k_argmax_tokens_batched_from_dev_input(
            output_q6k,
            output_rows,
            blocks_per_row,
            buffers.scratch_hidden_dev,
            plan.hidden_dim,
            plan.window_tokens,
            buffers.target_tokens_dev,
        )
    }

    pub(in crate::runtime) fn stage_mtp_verify_output_argmax_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        output_q4k: &[u8],
        output_rows: usize,
        output_cols: usize,
        output_norm: &[f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if output_rows == 0 {
            return Err("MTP verify output rows must be non-zero".to_string());
        }
        if output_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify output cols must match hidden_dim: cols={}, hidden_dim={}",
                output_cols, plan.hidden_dim
            ));
        }
        if output_cols == 0 || output_cols % 256 != 0 {
            return Err(format!(
                "MTP verify Q4_K output cols must be non-zero and divisible by 256, got {output_cols}"
            ));
        }
        if output_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify output_norm length mismatch: got {}, expected {}",
                output_norm.len(),
                plan.hidden_dim
            ));
        }
        let blocks_per_row = output_cols / 256;
        let expected = output_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q4_K output byte size overflow: rows={output_rows} cols={output_cols}"
                )
            })?;
        if output_q4k.len() != expected {
            return Err(format!(
                "MTP verify Q4_K output byte mismatch: got {}, expected {expected}",
                output_q4k.len()
            ));
        }

        self.stage_mtp_verify_hidden_rows_rms_norm(buffers, output_norm, norm_eps, false)?;
        for token_idx in 0..plan.window_tokens {
            let row_byte_offset = token_idx
                .checked_mul(plan.hidden_dim)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!(
                        "MTP verify hidden row byte offset overflow: token_idx={token_idx}, hidden_dim={}",
                        plan.hidden_dim
                    )
                })?;
            let normed_row_dev = buffers
                .scratch_hidden_dev
                .checked_add(row_byte_offset as u64)
                .ok_or_else(|| "MTP verify normed row device pointer overflow".to_string())?;
            let target_token_dev = buffers
                .target_tokens_dev
                .checked_add((token_idx * std::mem::size_of::<u32>()) as u64)
                .ok_or_else(|| "MTP verify target token device pointer overflow".to_string())?;
            self.write_q4k_argmax_token_from_dev_input(
                output_q4k,
                output_rows,
                blocks_per_row,
                normed_row_dev,
                target_token_dev,
            )?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn stage_mtp_verify_output_argmax_q8_0(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        output_q8_0: &[u8],
        output_rows: usize,
        output_cols: usize,
        output_norm: &[f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if output_rows == 0 {
            return Err("MTP verify output rows must be non-zero".to_string());
        }
        if output_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify output cols must match hidden_dim: cols={}, hidden_dim={}",
                output_cols, plan.hidden_dim
            ));
        }
        if output_cols == 0 || output_cols % 32 != 0 {
            return Err(format!(
                "MTP verify Q8_0 output cols must be non-zero and divisible by 32, got {output_cols}"
            ));
        }
        if output_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify output_norm length mismatch: got {}, expected {}",
                output_norm.len(),
                plan.hidden_dim
            ));
        }
        let blocks_per_row = output_cols / 32;
        let expected = output_rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(34))
            .ok_or_else(|| {
                format!(
                    "MTP verify Q8_0 output byte size overflow: rows={output_rows} cols={output_cols}"
                )
            })?;
        if output_q8_0.len() != expected {
            return Err(format!(
                "MTP verify Q8_0 output byte mismatch: got {}, expected {expected}",
                output_q8_0.len()
            ));
        }

        self.stage_mtp_verify_hidden_rows_rms_norm(buffers, output_norm, norm_eps, false)?;
        for token_idx in 0..plan.window_tokens {
            let row_byte_offset = token_idx
                .checked_mul(plan.hidden_dim)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!(
                        "MTP verify hidden row byte offset overflow: token_idx={token_idx}, hidden_dim={}",
                        plan.hidden_dim
                    )
                })?;
            let normed_row_dev = buffers
                .scratch_hidden_dev
                .checked_add(row_byte_offset as u64)
                .ok_or_else(|| "MTP verify normed row device pointer overflow".to_string())?;
            let target_token_dev = buffers
                .target_tokens_dev
                .checked_add((token_idx * std::mem::size_of::<u32>()) as u64)
                .ok_or_else(|| "MTP verify target token device pointer overflow".to_string())?;
            self.write_q8_0_argmax_token_from_dev_input(
                output_q8_0,
                output_rows,
                blocks_per_row,
                normed_row_dev,
                target_token_dev,
            )?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn stage_mtp_verify_hidden_rows_rms_norm(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        norm_weight: &[f32],
        norm_eps: f32,
        unit_offset: bool,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if norm_weight.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify RMSNorm weight length mismatch: got {}, expected {}",
                norm_weight.len(),
                plan.hidden_dim
            ));
        }
        let norm_weight_dev = self.resident_f32_ptr(norm_weight)?;
        self.launch_rms_norm_rows_f32(
            buffers.hidden_rows_dev,
            norm_weight_dev,
            buffers.scratch_hidden_dev,
            norm_eps,
            plan.window_tokens,
            plan.hidden_dim,
            unit_offset,
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_input_projections_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        request: Qwen35MtpGdnProjectionRequest<'_>,
    ) -> Result<MtpVerifyGdnProjectionBuffers, String> {
        let plan = &buffers.plan;
        if request.attn_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify GDN attn_norm length mismatch: got {}, expected {}",
                request.attn_norm.len(),
                plan.hidden_dim
            ));
        }
        let qkv_blocks = validate_mtp_verify_k_quant_matrix(
            "GDN qkv",
            request.qkv_quant,
            request.qkv_q4k,
            request.qkv_rows,
            request.qkv_cols,
            plan.hidden_dim,
        )?;
        let gate_blocks = validate_mtp_verify_k_quant_matrix(
            "GDN gate",
            request.gate_quant,
            request.gate_q4k,
            request.gate_rows,
            request.gate_cols,
            plan.hidden_dim,
        )?;
        let alpha_blocks = match request.alpha_quant {
            GGML_Q4_K => Some(validate_mtp_verify_q4k_matrix(
                "GDN alpha",
                request.alpha_q4k,
                request.alpha_rows,
                request.alpha_cols,
                plan.hidden_dim,
            )?),
            GGML_F32 => {
                validate_mtp_verify_f32_matrix(
                    "GDN alpha",
                    request.alpha_f32,
                    request.alpha_rows,
                    request.alpha_cols,
                    plan.hidden_dim,
                )?;
                None
            }
            other => {
                return Err(format!(
                    "MTP verify GDN alpha quant must be F32 or Q4_K, got {other}"
                ));
            }
        };
        let beta_blocks = match request.beta_quant {
            GGML_Q4_K => Some(validate_mtp_verify_q4k_matrix(
                "GDN beta",
                request.beta_q4k,
                request.beta_rows,
                request.beta_cols,
                plan.hidden_dim,
            )?),
            GGML_F32 => {
                validate_mtp_verify_f32_matrix(
                    "GDN beta",
                    request.beta_f32,
                    request.beta_rows,
                    request.beta_cols,
                    plan.hidden_dim,
                )?;
                None
            }
            other => {
                return Err(format!(
                    "MTP verify GDN beta quant must be F32 or Q4_K, got {other}"
                ));
            }
        };

        self.stage_mtp_verify_hidden_rows_rms_norm(
            buffers,
            request.attn_norm,
            request.norm_eps,
            false,
        )?;
        let projection_buffers = self.ensure_mtp_verify_gdn_projection_buffers(
            plan,
            request.qkv_rows,
            request.gate_rows,
            request.alpha_rows,
            request.beta_rows,
        )?;
        if plan.window_tokens == 2
            && request.qkv_quant == GGML_Q8_0
            && request.gate_quant == GGML_Q8_0
            && qkv_blocks == gate_blocks
            && crate::tuning::mtp_verify_q8_multi_projection_enabled()
        {
            self.launch_q8_0_gemv_batch_token2_multi3_to_dev(
                [request.qkv_q4k, request.gate_q4k, request.qkv_q4k],
                [request.qkv_rows, request.gate_rows, 0],
                qkv_blocks,
                buffers.scratch_hidden_dev,
                [
                    projection_buffers.qkv_dev,
                    projection_buffers.gate_dev,
                    projection_buffers.qkv_dev,
                ],
            )?;
        } else {
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "GDN qkv",
                request.qkv_quant,
                request.qkv_q4k,
                request.qkv_rows,
                qkv_blocks,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                projection_buffers.qkv_dev,
            )?;
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "GDN gate",
                request.gate_quant,
                request.gate_q4k,
                request.gate_rows,
                gate_blocks,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                projection_buffers.gate_dev,
            )?;
        }
        if alpha_blocks.is_none()
            && beta_blocks.is_none()
            && plan.window_tokens == 2
            && request.alpha_cols == request.beta_cols
            && crate::tuning::mtp_verify_f32_multi_projection_enabled()
        {
            self.launch_f32_gemv_batch_token2_multi2_to_dev(
                [request.alpha_f32, request.beta_f32],
                [request.alpha_rows, request.beta_rows],
                request.alpha_cols,
                buffers.scratch_hidden_dev,
                [projection_buffers.alpha_dev, projection_buffers.beta_dev],
            )?;
        } else {
            match alpha_blocks {
                Some(blocks) => self.stage_mtp_verify_k_quant_projection_to_dev(
                    "GDN alpha",
                    GGML_Q4_K,
                    request.alpha_q4k,
                    request.alpha_rows,
                    blocks,
                    plan.window_tokens,
                    buffers.scratch_hidden_dev,
                    projection_buffers.alpha_dev,
                )?,
                None => {
                    let alpha_dev = self.resident_f32_ptr(request.alpha_f32)?;
                    self.sgemm_device(
                        alpha_dev,
                        request.alpha_rows,
                        request.alpha_cols,
                        buffers.scratch_hidden_dev,
                        plan.window_tokens,
                        projection_buffers.alpha_dev,
                    )?;
                }
            }
            match beta_blocks {
                Some(blocks) => self.stage_mtp_verify_k_quant_projection_to_dev(
                    "GDN beta",
                    GGML_Q4_K,
                    request.beta_q4k,
                    request.beta_rows,
                    blocks,
                    plan.window_tokens,
                    buffers.scratch_hidden_dev,
                    projection_buffers.beta_dev,
                )?,
                None => {
                    let beta_dev = self.resident_f32_ptr(request.beta_f32)?;
                    self.sgemm_device(
                        beta_dev,
                        request.beta_rows,
                        request.beta_cols,
                        buffers.scratch_hidden_dev,
                        plan.window_tokens,
                        projection_buffers.beta_dev,
                    )?;
                }
            }
        }

        Ok(projection_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_qkv_projections_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        request: Qwen35MtpAttentionQkvProjectionRequest<'_>,
    ) -> Result<MtpVerifyAttentionQkvProjectionBuffers, String> {
        let plan = &buffers.plan;
        if request.attn_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify attention attn_norm length mismatch: got {}, expected {}",
                request.attn_norm.len(),
                plan.hidden_dim
            ));
        }
        let q_blocks = validate_mtp_verify_k_quant_matrix(
            "attention q",
            request.q_quant,
            request.q_q4k,
            request.q_rows,
            request.q_cols,
            plan.hidden_dim,
        )?;
        let k_blocks = validate_mtp_verify_k_quant_matrix(
            "attention k",
            request.k_quant,
            request.k_q4k,
            request.k_rows,
            request.k_cols,
            plan.hidden_dim,
        )?;
        let v_blocks = validate_mtp_verify_k_quant_matrix(
            "attention v",
            request.v_quant,
            request.v_q4k,
            request.v_rows,
            request.v_cols,
            plan.hidden_dim,
        )?;

        self.stage_mtp_verify_hidden_rows_rms_norm(
            buffers,
            request.attn_norm,
            request.norm_eps,
            false,
        )?;
        let projection_buffers = self.ensure_mtp_verify_attention_qkv_projection_buffers(
            plan,
            request.q_rows,
            request.k_rows,
            request.v_rows,
        )?;
        if plan.window_tokens == 2
            && request.q_quant == GGML_Q8_0
            && request.k_quant == GGML_Q8_0
            && request.v_quant == GGML_Q8_0
            && q_blocks == k_blocks
            && q_blocks == v_blocks
            && crate::tuning::mtp_verify_q8_multi_projection_enabled()
        {
            self.launch_q8_0_gemv_batch_token2_multi3_to_dev(
                [request.q_q4k, request.k_q4k, request.v_q4k],
                [request.q_rows, request.k_rows, request.v_rows],
                q_blocks,
                buffers.scratch_hidden_dev,
                [
                    projection_buffers.q_dev,
                    projection_buffers.k_dev,
                    projection_buffers.v_dev,
                ],
            )?;
        } else {
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "attention q",
                request.q_quant,
                request.q_q4k,
                request.q_rows,
                q_blocks,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                projection_buffers.q_dev,
            )?;
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "attention k",
                request.k_quant,
                request.k_q4k,
                request.k_rows,
                k_blocks,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                projection_buffers.k_dev,
            )?;
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "attention v",
                request.v_quant,
                request.v_q4k,
                request.v_rows,
                v_blocks,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                projection_buffers.v_dev,
            )?;
        }

        Ok(projection_buffers)
    }

    fn stage_mtp_verify_attention_qkv_projections_q4k_graph(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_idx: usize,
        request: Qwen35MtpAttentionQkvProjectionRequest<'_>,
    ) -> Result<MtpVerifyAttentionQkvProjectionBuffers, String> {
        let graph_enabled = crate::tuning::mtp_verify_attention_graph_enabled()
            && (buffers.plan.window_tokens == 1
                || (buffers.plan.window_tokens == 2
                    && crate::tuning::mtp_verify_window2_graphs_enabled()));
        if !graph_enabled {
            return self.stage_mtp_verify_attention_qkv_projections_q4k(buffers, request);
        }
        let key = super::MtpVerifyAttentionGraphKey {
            layer_idx,
            model_weight_ptr: request.q_q4k.as_ptr() as usize,
            hidden_dev: buffers.hidden_rows_dev,
            segment: 0,
            q8_selected_gate_up: false,
            pair2_selected_gate_up: false,
            pair2_selected_gate_up_silu: false,
            pair2_selected_down: false,
            pair2_selected_map: false,
        };
        if let Some(graph) = self.mtp_verify_attention_graphs.get(&key) {
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return self.ensure_mtp_verify_attention_qkv_projection_buffers(
                &buffers.plan,
                request.q_rows,
                request.k_rows,
                request.v_rows,
            );
        }
        if !self.mtp_verify_attention_graph_warmed.insert(key) {
            self.ensure_q4k_gemv_module()?;
            unsafe {
                self.api.stream_begin_capture(self.stream)?;
            }
            let result = self.stage_mtp_verify_attention_qkv_projections_q4k(buffers, request);
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    unsafe {
                        let _ = self.api.stream_end_capture(self.stream);
                    }
                    return Err(err);
                }
            };
            let graph = unsafe { self.api.stream_end_capture(self.stream)? };
            let exec = unsafe { self.api.graph_instantiate(graph)? };
            self.mtp_verify_attention_graphs.insert(
                key,
                super::SparseMoeGraph {
                    graph: graph as usize,
                    exec: exec as usize,
                },
            );
            let graph = self
                .mtp_verify_attention_graphs
                .get(&key)
                .ok_or_else(|| "missing MTP verify attention QKV CUDA graph".to_string())?;
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok(result);
        }
        self.stage_mtp_verify_attention_qkv_projections_q4k(buffers, request)
    }

    pub(in crate::runtime) fn stage_mtp_verify_k_quant_projection_to_dev(
        &mut self,
        label: &str,
        quant: u32,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        match (quant, seq_len) {
            (GGML_Q4_K, 1) => {
                self.q4k_dev_input_to_dev(weights, rows, blocks_per_row, input_dev, output_dev)
            }
            (GGML_Q6_K, 1) => {
                self.q6k_gemv_device_to_device(weights, rows, blocks_per_row, input_dev, output_dev)
            }
            (GGML_Q4_K, _) => self.q4k_batch_dev_input_to_dev(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_dev,
                output_dev,
            ),
            (GGML_Q6_K, _) => self.q6k_batch_dev_input_to_dev(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_dev,
                output_dev,
            ),
            (GGML_Q8_0, _) => self.launch_q8_0_gemv_batch_to_dev(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_dev,
                output_dev,
            ),
            (other, _) => Err(format!("MTP verify {label} unsupported quant code {other}")),
        }
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_qk_norm_rope(
        &mut self,
        projection_buffers: &MtpVerifyAttentionQkvProjectionBuffers,
        request: Qwen35MtpAttentionQkNormRopeRequest<'_>,
    ) -> Result<MtpVerifyAttentionQkNormRopeBuffers, String> {
        if request.q_norm.len() != request.head_dim {
            return Err(format!(
                "MTP verify attention q_norm length mismatch: got {}, expected {}",
                request.q_norm.len(),
                request.head_dim
            ));
        }
        if request.k_norm.len() != request.head_dim {
            return Err(format!(
                "MTP verify attention k_norm length mismatch: got {}, expected {}",
                request.k_norm.len(),
                request.head_dim
            ));
        }
        let expected_q_rows = request.num_heads.saturating_mul(request.head_dim);
        let has_gated_q = projection_buffers.q_rows == expected_q_rows.saturating_mul(2);
        if projection_buffers.q_rows != expected_q_rows && !has_gated_q {
            return Err(format!(
                "MTP verify attention q rows mismatch: got {}, expected {} or gated {}",
                projection_buffers.q_rows,
                expected_q_rows,
                expected_q_rows.saturating_mul(2)
            ));
        }
        let expected_kv_rows = request.num_kv_heads.saturating_mul(request.head_dim);
        if projection_buffers.k_rows != expected_kv_rows
            || projection_buffers.v_rows != expected_kv_rows
        {
            return Err(format!(
                "MTP verify attention kv rows mismatch: k={} v={} expected {expected_kv_rows}",
                projection_buffers.k_rows, projection_buffers.v_rows
            ));
        }
        if request.head_dim != 256 && request.head_dim != 512 {
            return Err(format!(
                "MTP verify attention head_dim unsupported for qk norm rope: {}",
                request.head_dim
            ));
        }
        let rope_dim = if request.rope_dim == 0 {
            request.head_dim
        } else {
            request.rope_dim.min(request.head_dim)
        };
        if rope_dim == 0 || rope_dim % 2 != 0 {
            return Err(format!(
                "MTP verify attention rope_dim must be non-zero and even: rope_dim={}",
                request.rope_dim
            ));
        }

        let post_buffers = self.ensure_mtp_verify_attention_qk_norm_rope_buffers(
            projection_buffers,
            expected_q_rows,
            request.head_dim,
        )?;
        let (q_input_dev, gate_dev) = if has_gated_q {
            let (q_compact_dev, gate_dev) = self.ensure_mtp_verify_attention_gated_q_buffers(
                projection_buffers.window_tokens,
                expected_q_rows,
            )?;
            self.launch_split_gated_attention_q_f32(
                projection_buffers.q_dev,
                q_compact_dev,
                gate_dev,
                projection_buffers.window_tokens,
                request.num_heads,
                request.head_dim,
            )?;
            (q_compact_dev, Some(gate_dev))
        } else {
            (projection_buffers.q_dev, None)
        };
        let post_buffers = MtpVerifyAttentionQkNormRopeBuffers {
            gate_dev,
            ..post_buffers
        };
        let q_norm_dev = self.resident_f32_ptr(request.q_norm)?;
        let k_norm_dev = self.resident_f32_ptr(request.k_norm)?;
        let rope_table_dim = if request.rope_neox && rope_dim == request.head_dim {
            request.head_dim
        } else {
            rope_dim
        };
        let (rope_sin_dev, rope_cos_dev) = if request.rope_freq_factors.is_empty() {
            let table = self.rope_table_ptrs(
                rope_table_dim,
                projection_buffers.window_tokens,
                request.pos_start,
                request.rope_theta,
            )?;
            (table.sin_ptr, table.cos_ptr)
        } else {
            let pair_count = rope_table_dim / 2;
            if request.rope_freq_factors.len() < pair_count {
                return Err(format!(
                    "MTP verify attention RoPE factor len mismatch: got {}, expected at least {pair_count}",
                    request.rope_freq_factors.len()
                ));
            }
            let table_len = projection_buffers
                .window_tokens
                .checked_mul(pair_count)
                .ok_or_else(|| "MTP verify custom RoPE table length overflow".to_string())?;
            let mut sin = Vec::with_capacity(table_len);
            let mut cos = Vec::with_capacity(table_len);
            for token in 0..projection_buffers.window_tokens {
                let pos = (request.pos_start + token) as f32;
                for i in 0..pair_count {
                    let freq = 1.0
                        / (request
                            .rope_theta
                            .powf((2 * i) as f32 / rope_table_dim as f32)
                            * request.rope_freq_factors[i]);
                    let (sin_value, cos_value) = (pos * freq).sin_cos();
                    sin.push(sin_value);
                    cos.push(cos_value);
                }
            }
            let table_bytes = std::mem::size_of_val(sin.as_slice());
            let slab_bytes = table_bytes
                .checked_mul(2)
                .ok_or_else(|| "MTP verify custom RoPE slab byte overflow".to_string())?;
            let sin_dev = self.compute_temp_slab_ptr(slab_bytes)?;
            let cos_dev = sin_dev + table_bytes as u64;
            unsafe {
                self.api.memcpy_htod_async(
                    sin_dev,
                    sin.as_ptr().cast::<libc::c_void>(),
                    table_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    cos_dev,
                    cos.as_ptr().cast::<libc::c_void>(),
                    table_bytes,
                    self.stream,
                )?;
            }
            (sin_dev, cos_dev)
        };

        match request.head_dim {
            256 if request.rope_neox && rope_dim == request.head_dim => self
                .launch_qk_norm_rope_neox_hd256_f16kv(
                    q_input_dev,
                    projection_buffers.k_dev,
                    projection_buffers.v_dev,
                    q_norm_dev,
                    k_norm_dev,
                    rope_sin_dev,
                    rope_cos_dev,
                    post_buffers.q_dev,
                    post_buffers.k_bits_dev,
                    post_buffers.v_bits_dev,
                    request.norm_eps,
                    projection_buffers.window_tokens,
                    request.num_heads,
                    request.num_kv_heads,
                    request.pos_start,
                    request.q_unit_offset,
                    request.k_unit_offset,
                    request.v_no_scale_norm,
                )?,
            256 => self.launch_qk_norm_rope_select_hd256_f16kv(
                q_input_dev,
                projection_buffers.k_dev,
                projection_buffers.v_dev,
                q_norm_dev,
                k_norm_dev,
                rope_sin_dev,
                rope_cos_dev,
                post_buffers.q_dev,
                post_buffers.k_bits_dev,
                post_buffers.v_bits_dev,
                request.norm_eps,
                projection_buffers.window_tokens,
                request.num_heads,
                request.num_kv_heads,
                request.pos_start,
                request.q_unit_offset,
                request.k_unit_offset,
                request.v_no_scale_norm,
                rope_dim,
                request.rope_neox,
            )?,
            512 => self.launch_qk_norm_rope_select_hd512_f16kv(
                q_input_dev,
                projection_buffers.k_dev,
                projection_buffers.v_dev,
                q_norm_dev,
                k_norm_dev,
                rope_sin_dev,
                rope_cos_dev,
                post_buffers.q_dev,
                post_buffers.k_bits_dev,
                post_buffers.v_bits_dev,
                request.norm_eps,
                projection_buffers.window_tokens,
                request.num_heads,
                request.num_kv_heads,
                request.pos_start,
                request.q_unit_offset,
                request.k_unit_offset,
                request.v_no_scale_norm,
                rope_dim,
                request.rope_neox,
            )?,
            _ => unreachable!("head_dim validated above"),
        }

        Ok(post_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn collect_mtp_verify_attention_window_kv_bits(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        self.collect_mtp_verify_attention_window_kv_bits_inner(post_buffers, true)
    }

    pub(in crate::runtime) fn collect_mtp_verify_attention_window_kv_bits_deferred(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        self.collect_mtp_verify_attention_window_kv_bits_inner(post_buffers, false)
    }

    fn collect_mtp_verify_attention_window_kv_bits_inner(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        sync_after_copy: bool,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        let kv_values = post_buffers
            .window_tokens
            .checked_mul(post_buffers.kv_rows)
            .ok_or_else(|| {
                format!(
                    "MTP verify attention current K/V value count overflow: tokens={} rows={}",
                    post_buffers.window_tokens, post_buffers.kv_rows
                )
            })?;
        let kv_bytes = kv_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("MTP verify attention current K/V byte count overflow: values={kv_values}")
            })?;
        let mut k_bits = vec![0u16; kv_values];
        let mut v_bits = vec![0u16; kv_values];
        unsafe {
            if let Err(err) = self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                post_buffers.k_bits_dev,
                kv_bytes,
                self.stream,
            ) {
                return Err(err);
            }
            if let Err(err) = self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                post_buffers.v_bits_dev,
                kv_bytes,
                self.stream,
            ) {
                if !sync_after_copy {
                    let _ = self.stream_synchronize();
                }
                return Err(err);
            }
        }
        if sync_after_copy {
            self.stream_synchronize()?;
        }
        Ok((k_bits, v_bits))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mtp_verify_attention_hd256_split(
        &mut self,
        attention_buffers: &MtpVerifyAttentionOutputBuffers,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        f16_kv: Option<(u64, u64)>,
        kv_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: usize,
    ) -> Result<bool, String> {
        if post_buffers.head_dim != 256
            || !crate::tuning::mtp_verify_attention_hd256_split_enabled()
        {
            return Ok(false);
        }
        let active_window = window.min(kv_tokens);
        let chunk_size = crate::tuning::mtp_verify_attention_hd256_split_chunk_size();
        if active_window <= chunk_size {
            return Ok(false);
        }
        let num_chunks = active_window.div_ceil(chunk_size);
        let partial_values = post_buffers
            .window_tokens
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(num_chunks))
            .and_then(|values| values.checked_mul(post_buffers.head_dim))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention split partial overflow: tokens={} heads={num_heads} chunks={num_chunks} head_dim={}",
                    post_buffers.window_tokens, post_buffers.head_dim
                )
            })?;
        let partial_bytes = partial_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("MTP verify attention split partial byte overflow: values={partial_values}")
            })?;
        let meta_values = post_buffers
            .window_tokens
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(num_chunks))
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| {
                format!(
                    "MTP verify attention split meta overflow: tokens={} heads={num_heads} chunks={num_chunks}",
                    post_buffers.window_tokens
                )
            })?;
        let meta_bytes = meta_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("MTP verify attention split meta byte overflow: values={meta_values}")
            })?;
        let partial_dev = self.compute_mid_a_ptr(partial_bytes)?;
        let meta_dev = self.compute_mid_b_ptr(meta_bytes)?;

        let mut partial_arg = partial_dev;
        let mut meta_arg = meta_dev;
        let mut output_arg = attention_buffers.attn_out_dev;
        let mut q_arg = post_buffers.q_dev;
        let mut k_arg = attention_buffers.k_f32_dev;
        let mut v_arg = attention_buffers.v_f32_dev;
        let mut seq_arg = u32::try_from(post_buffers.window_tokens).map_err(|_| {
            format!(
                "MTP verify attention split seq_len exceeds u32: {}",
                post_buffers.window_tokens
            )
        })?;
        let mut kv_len_arg = u32::try_from(kv_tokens)
            .map_err(|_| format!("MTP verify attention split kv_len exceeds u32: {kv_tokens}"))?;
        let mut heads_arg = u32::try_from(num_heads).map_err(|_| {
            format!("MTP verify attention split num_heads exceeds u32: {num_heads}")
        })?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads).map_err(|_| {
            format!("MTP verify attention split num_kv_heads exceeds u32: {num_kv_heads}")
        })?;
        let mut scale_arg = scale;
        let mut window_arg = u32::try_from(window)
            .map_err(|_| format!("MTP verify attention split window exceeds u32: {window}"))?;
        let mut chunk_size_arg = u32::try_from(chunk_size).map_err(|_| {
            format!("MTP verify attention split chunk size exceeds u32: {chunk_size}")
        })?;
        let mut num_chunks_arg = u32::try_from(num_chunks).map_err(|_| {
            format!("MTP verify attention split chunk count exceeds u32: {num_chunks}")
        })?;
        let use_mma_stream_k = f16_kv.is_some()
            && kv_tokens == post_buffers.window_tokens
            && post_buffers.window_tokens >= 64
            && window >= kv_tokens
            && crate::tuning::mtp_verify_attention_hd256_mma_stream_k_enabled();
        if use_mma_stream_k {
            if let Some((k_f16_dev, v_f16_dev)) = f16_kv {
                k_arg = k_f16_dev;
                v_arg = v_f16_dev;
            }
        }
        // Window-2 decode keeps the single-query jbatch8 path; four-query tiling only
        // amortizes K/V reads once a full query tile is available.
        let query_tile = post_buffers.window_tokens >= 4
            && window >= kv_tokens
            && crate::tuning::mtp_verify_attention_hd256_query_tile_enabled();
        let partial_kernel = if use_mma_stream_k {
            "rnb_attention_prefill_flash_hd256_window_mma_stream_k_partials"
        } else if query_tile {
            "rnb_attention_prefill_flash_hd256_window_split_partials_qtile4_jbatch4"
        } else {
            "rnb_attention_prefill_flash_hd256_window_split_partials_jbatch8"
        };
        let partial_grid_x = if use_mma_stream_k {
            seq_arg.div_ceil(64)
        } else if query_tile {
            seq_arg.div_ceil(4)
        } else {
            seq_arg
        };
        let partial_block_x = if use_mma_stream_k { 128 } else { 256 };
        self.launch_cached_gemv(
            partial_kernel,
            &[
                (&mut partial_arg as *mut u64).cast::<libc::c_void>(),
                (&mut meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                (&mut chunk_size_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_chunks_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (partial_grid_x, heads_arg, num_chunks_arg),
            (partial_block_x, 1, 1),
        )?;
        self.launch_cached_gemv(
            "rnb_attention_prefill_flash_hd256_window_split_reduce",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut partial_arg as *mut u64).cast::<libc::c_void>(),
                (&mut meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_chunks_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg, 1),
            (256, 1, 1),
        )?;
        Ok(true)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_output_window(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        request: Qwen35MtpAttentionOutputRequest,
    ) -> Result<MtpVerifyAttentionOutputBuffers, String> {
        let attention_kernel = mtp_verify_attention_window_kernel(post_buffers.head_dim)?;
        if post_buffers.q_rows != request.num_heads.saturating_mul(post_buffers.head_dim) {
            return Err(format!(
                "MTP verify attention output q rows mismatch: got {}, expected {}",
                post_buffers.q_rows,
                request.num_heads.saturating_mul(post_buffers.head_dim)
            ));
        }
        let expected_kv_rows = request.num_kv_heads.saturating_mul(post_buffers.head_dim);
        if post_buffers.kv_rows != expected_kv_rows {
            return Err(format!(
                "MTP verify attention output kv rows mismatch: got {}, expected {expected_kv_rows}",
                post_buffers.kv_rows
            ));
        }
        if request.window == 0 {
            return Err("MTP verify attention output window must be non-zero".to_string());
        }
        let attention_buffers = self.ensure_mtp_verify_attention_output_buffers(post_buffers)?;
        let kv_len = post_buffers
            .window_tokens
            .checked_mul(post_buffers.kv_rows)
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output kv len overflow: tokens={} rows={}",
                    post_buffers.window_tokens, post_buffers.kv_rows
                )
            })?;
        self.launch_f16_to_f32(post_buffers.k_bits_dev, attention_buffers.k_f32_dev, kv_len)?;
        self.launch_f16_to_f32(post_buffers.v_bits_dev, attention_buffers.v_f32_dev, kv_len)?;

        let mut output_arg = attention_buffers.attn_out_dev;
        let mut q_arg = post_buffers.q_dev;
        let mut k_arg = attention_buffers.k_f32_dev;
        let mut v_arg = attention_buffers.v_f32_dev;
        let mut seq_arg = u32::try_from(post_buffers.window_tokens).map_err(|_| {
            format!(
                "MTP verify attention output seq_len exceeds u32: {}",
                post_buffers.window_tokens
            )
        })?;
        let mut kv_len_arg = seq_arg;
        let mut heads_arg = u32::try_from(request.num_heads).map_err(|_| {
            format!(
                "MTP verify attention output num_heads exceeds u32: {}",
                request.num_heads
            )
        })?;
        let mut kv_heads_arg = u32::try_from(request.num_kv_heads).map_err(|_| {
            format!(
                "MTP verify attention output num_kv_heads exceeds u32: {}",
                request.num_kv_heads
            )
        })?;
        let mut scale_arg = request.scale;
        let mut window_arg = u32::try_from(request.window).map_err(|_| {
            format!(
                "MTP verify attention output window exceeds u32: {}",
                request.window
            )
        })?;
        if !self.launch_mtp_verify_attention_hd256_split(
            &attention_buffers,
            post_buffers,
            Some((post_buffers.k_bits_dev, post_buffers.v_bits_dev)),
            post_buffers.window_tokens,
            request.num_heads,
            request.num_kv_heads,
            request.scale,
            request.window,
        )? {
            self.launch_cached_gemv(
                attention_kernel,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_arg, heads_arg, 1),
                (256, 1, 1),
            )?;
        }

        Ok(attention_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_output_prior_window(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        request: Qwen35MtpAttentionOutputWithPriorRequest,
    ) -> Result<MtpVerifyAttentionOutputBuffers, String> {
        let attention_kernel = mtp_verify_attention_window_kernel(post_buffers.head_dim)?;
        if post_buffers.q_rows != request.num_heads.saturating_mul(post_buffers.head_dim) {
            return Err(format!(
                "MTP verify attention output with prior q rows mismatch: got {}, expected {}",
                post_buffers.q_rows,
                request.num_heads.saturating_mul(post_buffers.head_dim)
            ));
        }
        let expected_kv_rows = request.num_kv_heads.saturating_mul(post_buffers.head_dim);
        if post_buffers.kv_rows != expected_kv_rows {
            return Err(format!(
                "MTP verify attention output with prior kv rows mismatch: got {}, expected {expected_kv_rows}",
                post_buffers.kv_rows
            ));
        }
        if request.window == 0 {
            return Err(
                "MTP verify attention output with prior window must be non-zero".to_string(),
            );
        }
        if request.prior_tokens > 0
            && (request.prior_k_bits_dev == 0 || request.prior_v_bits_dev == 0)
        {
            return Err(
                "MTP verify attention output with prior requires non-null prior K/V device pointers"
                    .to_string(),
            );
        }
        let kv_tokens = request
            .prior_tokens
            .checked_add(post_buffers.window_tokens)
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output with prior kv token overflow: prior={} window={}",
                    request.prior_tokens, post_buffers.window_tokens
                )
            })?;
        let attention_buffers =
            self.ensure_mtp_verify_attention_output_buffers_for_kv_tokens(post_buffers, kv_tokens)?;
        let prior_values = request
            .prior_tokens
            .checked_mul(post_buffers.kv_rows)
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output with prior value overflow: prior={} rows={}",
                    request.prior_tokens, post_buffers.kv_rows
                )
            })?;
        let current_values = post_buffers
            .window_tokens
            .checked_mul(post_buffers.kv_rows)
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output current value overflow: tokens={} rows={}",
                    post_buffers.window_tokens, post_buffers.kv_rows
                )
            })?;
        if prior_values > 0 {
            self.launch_f16_to_f32(
                request.prior_k_bits_dev,
                attention_buffers.k_f32_dev,
                prior_values,
            )?;
            self.launch_f16_to_f32(
                request.prior_v_bits_dev,
                attention_buffers.v_f32_dev,
                prior_values,
            )?;
        }
        let prior_f32_bytes = prior_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "MTP verify attention output prior f32 byte overflow: values={prior_values}"
                )
            })?;
        let prior_f32_bytes = u64::try_from(prior_f32_bytes).map_err(|_| {
            format!(
                "MTP verify attention output prior f32 byte offset exceeds u64: {prior_f32_bytes}"
            )
        })?;
        let current_k_f32_dev = attention_buffers
            .k_f32_dev
            .checked_add(prior_f32_bytes)
            .ok_or_else(|| {
                "MTP verify attention output prior K device pointer overflow".to_string()
            })?;
        let current_v_f32_dev = attention_buffers
            .v_f32_dev
            .checked_add(prior_f32_bytes)
            .ok_or_else(|| {
                "MTP verify attention output prior V device pointer overflow".to_string()
            })?;
        self.launch_f16_to_f32(post_buffers.k_bits_dev, current_k_f32_dev, current_values)?;
        self.launch_f16_to_f32(post_buffers.v_bits_dev, current_v_f32_dev, current_values)?;

        let mut output_arg = attention_buffers.attn_out_dev;
        let mut q_arg = post_buffers.q_dev;
        let mut k_arg = attention_buffers.k_f32_dev;
        let mut v_arg = attention_buffers.v_f32_dev;
        let mut seq_arg = u32::try_from(post_buffers.window_tokens).map_err(|_| {
            format!(
                "MTP verify attention output with prior seq_len exceeds u32: {}",
                post_buffers.window_tokens
            )
        })?;
        let mut kv_len_arg = u32::try_from(kv_tokens).map_err(|_| {
            format!("MTP verify attention output with prior kv_len exceeds u32: {kv_tokens}")
        })?;
        let mut heads_arg = u32::try_from(request.num_heads).map_err(|_| {
            format!(
                "MTP verify attention output with prior num_heads exceeds u32: {}",
                request.num_heads
            )
        })?;
        let mut kv_heads_arg = u32::try_from(request.num_kv_heads).map_err(|_| {
            format!(
                "MTP verify attention output with prior num_kv_heads exceeds u32: {}",
                request.num_kv_heads
            )
        })?;
        let mut scale_arg = request.scale;
        let mut window_arg = u32::try_from(request.window).map_err(|_| {
            format!(
                "MTP verify attention output with prior window exceeds u32: {}",
                request.window
            )
        })?;
        if !self.launch_mtp_verify_attention_hd256_split(
            &attention_buffers,
            post_buffers,
            None,
            kv_tokens,
            request.num_heads,
            request.num_kv_heads,
            request.scale,
            request.window,
        )? {
            self.launch_cached_gemv(
                attention_kernel,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_arg, heads_arg, 1),
                (256, 1, 1),
            )?;
        }

        Ok(attention_buffers)
    }

    fn stage_mtp_verify_attention_output_kvarn_window(
        &mut self,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
    ) -> Result<MtpVerifyAttentionOutputBuffers, String> {
        if post_buffers.q_rows != request.num_heads().saturating_mul(post_buffers.head_dim) {
            return Err(format!(
                "MTP verify KVarN attention q rows mismatch: got {}, expected {}",
                post_buffers.q_rows,
                request.num_heads().saturating_mul(post_buffers.head_dim)
            ));
        }
        let expected_kv_rows = request.num_kv_heads().saturating_mul(post_buffers.head_dim);
        if post_buffers.kv_rows != expected_kv_rows {
            return Err(format!(
                "MTP verify KVarN attention kv rows mismatch: got {}, expected {expected_kv_rows}",
                post_buffers.kv_rows
            ));
        }
        if request.head_dim() != post_buffers.head_dim {
            return Err(format!(
                "MTP verify KVarN attention head dim mismatch: cache={} current={}",
                request.head_dim(),
                post_buffers.head_dim
            ));
        }
        let attention_buffers = self.ensure_mtp_verify_attention_output_buffers(post_buffers)?;
        self.attention_verify_kvarn_to_device(
            request,
            post_buffers.q_dev,
            post_buffers.k_bits_dev,
            post_buffers.v_bits_dev,
            post_buffers.window_tokens,
            attention_buffers.attn_out_dev,
        )?;
        Ok(attention_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_o_projection_residual_ffn_norm_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        attention_buffers: &MtpVerifyAttentionOutputBuffers,
        o_q4k: &[u8],
        o_quant: u32,
        o_rows: usize,
        o_cols: usize,
        post_attn_norm: &[f32],
        post_attn_norm_unit_offset: bool,
        ffn_norm: &[f32],
        ffn_norm_unit_offset: bool,
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if attention_buffers.window_tokens != plan.window_tokens {
            return Err(format!(
                "MTP verify attention residual window mismatch: attention={}, plan={}",
                attention_buffers.window_tokens, plan.window_tokens
            ));
        }
        if attention_buffers.q_rows != o_cols {
            return Err(format!(
                "MTP verify attention output rows mismatch: got {}, expected o_proj cols {o_cols}",
                attention_buffers.q_rows
            ));
        }
        if o_rows != plan.hidden_dim {
            return Err(format!(
                "MTP verify attention o_proj rows mismatch: got {o_rows}, expected {}",
                plan.hidden_dim
            ));
        }
        let blocks_per_row = validate_mtp_verify_k_quant_matrix(
            "attention o_proj",
            o_quant,
            o_q4k,
            o_rows,
            o_cols,
            attention_buffers.q_rows,
        )?;
        if ffn_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify attention ffn_norm length mismatch: got {}, expected {}",
                ffn_norm.len(),
                plan.hidden_dim
            ));
        }
        if !post_attn_norm.is_empty() && post_attn_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify attention post_attn_norm length mismatch: got {}, expected {}",
                post_attn_norm.len(),
                plan.hidden_dim
            ));
        }
        self.stage_mtp_verify_k_quant_projection_to_dev(
            "attention o_proj",
            o_quant,
            o_q4k,
            o_rows,
            blocks_per_row,
            plan.window_tokens,
            attention_buffers.attn_out_dev,
            buffers.scratch_hidden_dev,
        )?;
        if post_attn_norm.is_empty() {
            self.launch_add_f32_inplace(
                buffers.hidden_rows_dev,
                buffers.scratch_hidden_dev,
                plan.window_tokens * plan.hidden_dim,
            )?;
        } else {
            let post_attn_norm_dev = self.resident_f32_ptr(post_attn_norm)?;
            self.launch_rms_norm_add_rows_f32_inplace(
                buffers.scratch_hidden_dev,
                post_attn_norm_dev,
                buffers.hidden_rows_dev,
                norm_eps,
                plan.window_tokens,
                plan.hidden_dim,
                post_attn_norm_unit_offset,
            )?;
        }
        self.stage_mtp_verify_hidden_rows_rms_norm(
            buffers,
            ffn_norm,
            norm_eps,
            ffn_norm_unit_offset,
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_attention_dense_ffn_residual_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        gate_q4k: &[u8],
        gate_rows: usize,
        gate_cols: usize,
        up_q4k: &[u8],
        up_rows: usize,
        up_cols: usize,
        down: &[u8],
        down_quant: u32,
        down_rows: usize,
        down_cols: usize,
        ffn_uses_gelu: bool,
        post_ffw_norm: &[f32],
        post_ffw_norm_unit_offset: bool,
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if gate_rows == 0 {
            return Err("MTP verify attention dense FFN gate rows must be non-zero".to_string());
        }
        if gate_rows != up_rows {
            return Err(format!(
                "MTP verify attention dense FFN gate/up row mismatch: gate={gate_rows} up={up_rows}"
            ));
        }
        if gate_cols != plan.hidden_dim || up_cols != plan.hidden_dim {
            return Err(format!(
                "MTP verify attention dense FFN gate/up cols must match hidden_dim {}: gate={} up={}",
                plan.hidden_dim, gate_cols, up_cols
            ));
        }
        if down_rows != plan.hidden_dim || down_cols != gate_rows {
            return Err(format!(
                "MTP verify attention dense FFN down shape mismatch: got [{down_rows}x{down_cols}], expected [{}x{}]",
                plan.hidden_dim, gate_rows
            ));
        }
        let gate_blocks = validate_mtp_verify_q4k_matrix(
            "attention dense FFN gate",
            gate_q4k,
            gate_rows,
            gate_cols,
            plan.hidden_dim,
        )?;
        let up_blocks = validate_mtp_verify_q4k_matrix(
            "attention dense FFN up",
            up_q4k,
            up_rows,
            up_cols,
            plan.hidden_dim,
        )?;
        let down_blocks = validate_mtp_verify_k_quant_matrix(
            "attention dense FFN down",
            down_quant,
            down,
            down_rows,
            down_cols,
            gate_rows,
        )?;
        let _ = (gate_blocks, up_blocks, down_blocks);
        let mut trace_stage = std::time::Instant::now();
        if ffn_uses_gelu {
            self.dense_q4k_gelu_ffn_batch_dev_input_to_dev(
                gate_q4k,
                up_q4k,
                down,
                down_quant,
                gate_rows,
                plan.hidden_dim,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                buffers.scratch_hidden_dev,
                None,
                None,
                &mut trace_stage,
            )?;
        } else {
            self.dense_q4k_silu_ffn_batch_dev_input_to_dev(
                gate_q4k,
                up_q4k,
                down,
                down_quant,
                gate_rows,
                plan.hidden_dim,
                plan.window_tokens,
                buffers.scratch_hidden_dev,
                buffers.scratch_hidden_dev,
                None,
                None,
                &mut trace_stage,
            )?;
        }
        if post_ffw_norm.is_empty() {
            self.launch_add_f32_inplace(
                buffers.hidden_rows_dev,
                buffers.scratch_hidden_dev,
                plan.window_tokens * plan.hidden_dim,
            )
        } else {
            if post_ffw_norm.len() != plan.hidden_dim {
                return Err(format!(
                    "MTP verify attention post_ffw_norm length mismatch: got {}, expected {}",
                    post_ffw_norm.len(),
                    plan.hidden_dim
                ));
            }
            let post_ffw_norm_dev = self.resident_f32_ptr(post_ffw_norm)?;
            self.launch_rms_norm_add_rows_f32_inplace(
                buffers.scratch_hidden_dev,
                post_ffw_norm_dev,
                buffers.hidden_rows_dev,
                norm_eps,
                plan.window_tokens,
                plan.hidden_dim,
                post_ffw_norm_unit_offset,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn stage_mtp_verify_gemma4_ple(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        gate_weights: &[u8],
        gate_quant: u32,
        gate_rows: usize,
        gate_cols: usize,
        proj_weights: &[u8],
        proj_quant: u32,
        proj_rows: usize,
        proj_cols: usize,
        post_norm: &[f32],
        post_norm_unit_offset: bool,
        ple_input: &[f32],
        out_scale: &[f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        let seq_len = plan.window_tokens;
        let hidden_dim = plan.hidden_dim;
        if gate_cols != hidden_dim || proj_rows != hidden_dim || proj_cols != gate_rows {
            return Err(format!(
                "MTP verify Gemma PLE shape mismatch: gate=[{gate_rows}x{gate_cols}] proj=[{proj_rows}x{proj_cols}] hidden={hidden_dim}"
            ));
        }
        let ple_dim = gate_rows;
        let expected_ple = seq_len
            .checked_mul(ple_dim)
            .ok_or_else(|| "MTP verify Gemma PLE input size overflow".to_string())?;
        if ple_input.len() != expected_ple {
            return Err(format!(
                "MTP verify Gemma PLE input len mismatch: got {}, expected {expected_ple}",
                ple_input.len()
            ));
        }
        if post_norm.len() != hidden_dim {
            return Err(format!(
                "MTP verify Gemma PLE post_norm len mismatch: got {}, expected {hidden_dim}",
                post_norm.len()
            ));
        }
        if !out_scale.is_empty() && out_scale.len() != 1 {
            return Err(format!(
                "MTP verify Gemma PLE out_scale len mismatch: got {}, expected 1",
                out_scale.len()
            ));
        }
        let ple_bytes = std::mem::size_of_val(ple_input);
        let ple_input_dev = self.compute_full_down_ptr(ple_bytes)?;
        let gate_dev = self.compute_mid_a_ptr(ple_bytes)?;
        let proj_dev = self.compute_output_ptr(
            seq_len
                .checked_mul(hidden_dim)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "MTP verify Gemma PLE output byte overflow".to_string())?,
        )?;
        unsafe {
            self.api.memcpy_htod_async(
                ple_input_dev,
                ple_input.as_ptr().cast::<libc::c_void>(),
                ple_bytes,
                self.stream,
            )?;
        }
        match (gate_quant, proj_quant) {
            (GGML_Q4_K, GGML_Q4_K) => {
                let gate_blocks = validate_mtp_verify_q4k_matrix(
                    "Gemma PLE gate",
                    gate_weights,
                    gate_rows,
                    gate_cols,
                    hidden_dim,
                )?;
                let proj_blocks = validate_mtp_verify_q4k_matrix(
                    "Gemma PLE proj",
                    proj_weights,
                    proj_rows,
                    proj_cols,
                    ple_dim,
                )?;
                self.q4k_batch_dev_input_to_dev(
                    gate_weights,
                    gate_rows,
                    gate_blocks,
                    seq_len,
                    buffers.hidden_rows_dev,
                    gate_dev,
                )?;
                self.launch_gelu_mul(gate_dev, ple_input_dev, expected_ple)?;
                self.q4k_batch_dev_input_to_dev(
                    proj_weights,
                    proj_rows,
                    proj_blocks,
                    seq_len,
                    gate_dev,
                    proj_dev,
                )?;
            }
            (GGML_F32, GGML_F32) => {
                let expected_gate_bytes = gate_rows
                    .checked_mul(gate_cols)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| "MTP verify Gemma PLE F32 gate byte overflow".to_string())?;
                let expected_proj_bytes = proj_rows
                    .checked_mul(proj_cols)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| "MTP verify Gemma PLE F32 proj byte overflow".to_string())?;
                if gate_weights.len() != expected_gate_bytes
                    || proj_weights.len() != expected_proj_bytes
                {
                    return Err(format!(
                        "MTP verify Gemma PLE F32 byte mismatch: gate={} expected={} proj={} expected={}",
                        gate_weights.len(),
                        expected_gate_bytes,
                        proj_weights.len(),
                        expected_proj_bytes
                    ));
                }
                let gate_weights_dev = self.resident_f32_weights_ptr_from_le_bytes(
                    gate_weights,
                    "MTP verify Gemma PLE gate",
                )?;
                self.sgemm_device(
                    gate_weights_dev,
                    gate_rows,
                    gate_cols,
                    buffers.hidden_rows_dev,
                    seq_len,
                    gate_dev,
                )?;
                self.launch_gelu_mul(gate_dev, ple_input_dev, expected_ple)?;
                let proj_weights_dev = self.resident_f32_weights_ptr_from_le_bytes(
                    proj_weights,
                    "MTP verify Gemma PLE proj",
                )?;
                self.sgemm_device(
                    proj_weights_dev,
                    proj_rows,
                    proj_cols,
                    gate_dev,
                    seq_len,
                    proj_dev,
                )?;
            }
            _ => {
                return Err(format!(
                    "MTP verify Gemma PLE requires matching Q4_K or F32 weights, got gate={gate_quant} proj={proj_quant}"
                ));
            }
        }
        let post_norm_dev = self.resident_f32_ptr(post_norm)?;
        self.launch_rms_norm_add_rows_f32_inplace(
            proj_dev,
            post_norm_dev,
            buffers.hidden_rows_dev,
            norm_eps,
            seq_len,
            hidden_dim,
            post_norm_unit_offset,
        )?;
        if let Some(scale) = out_scale.first().copied() {
            self.launch_scale_f32_inplace(buffers.hidden_rows_dev, scale, seq_len * hidden_dim)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_conv1d_silu(
        &mut self,
        projection_buffers: &MtpVerifyGdnProjectionBuffers,
        conv_state: &[f32],
        conv_kernel: &[f32],
        kernel_size: usize,
        keep_state_resident: bool,
    ) -> Result<MtpVerifyGdnConvBuffers, String> {
        if kernel_size < 2 {
            return Err(format!(
                "MTP verify GDN conv kernel_size must be >= 2, got {kernel_size}"
            ));
        }
        let channels = projection_buffers.qkv_rows;
        let state_values = (kernel_size - 1).checked_mul(channels).ok_or_else(|| {
            format!(
                "MTP verify GDN conv state length overflow: channels={channels}, kernel_size={kernel_size}"
            )
        })?;
        if conv_state.len() != state_values {
            return Err(format!(
                "MTP verify GDN conv_state length mismatch: got {}, expected {state_values}",
                conv_state.len()
            ));
        }
        let kernel_values = kernel_size.checked_mul(channels).ok_or_else(|| {
            format!(
                "MTP verify GDN conv kernel length overflow: channels={channels}, kernel_size={kernel_size}"
            )
        })?;
        if conv_kernel.len() != kernel_values {
            return Err(format!(
                "MTP verify GDN conv_kernel length mismatch: got {}, expected {kernel_values}",
                conv_kernel.len()
            ));
        }

        let mut conv_buffers = self.ensure_mtp_verify_gdn_conv_buffers(
            projection_buffers.window_tokens,
            channels,
            kernel_size,
        )?;
        let keep_state_resident =
            keep_state_resident && crate::tuning::mtp_verify_resident_conv_enabled();
        if keep_state_resident {
            conv_buffers.conv_state_dev = self.resident_delta_state_ptr(conv_state)?;
            conv_buffers.device_resident_state = true;
        } else {
            unsafe {
                self.api.memcpy_htod_async(
                    conv_buffers.conv_state_dev,
                    conv_state.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(conv_state),
                    self.stream,
                )?;
            }
        }
        let conv_kernel_dev = self.resident_f32_ptr(conv_kernel)?;
        self.launch_gdn_build_conv_input_f32(
            conv_buffers.conv_input_dev,
            conv_buffers.conv_state_dev,
            projection_buffers.qkv_dev,
            projection_buffers.window_tokens,
            channels,
            kernel_size,
        )?;
        self.launch_ssm_conv1d_silu_dev(
            conv_buffers.conv_input_dev,
            conv_kernel_dev,
            conv_buffers.conv_out_dev,
            projection_buffers.window_tokens,
            channels,
            kernel_size,
        )?;
        Ok(conv_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_conv_prefix_state(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        prefix_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        self.stage_mtp_verify_gdn_conv_prefix_state_inner(conv_buffers, prefix_tokens, true)
    }

    pub(in crate::runtime) fn stage_mtp_verify_gdn_conv_prefix_state_deferred(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        prefix_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        self.stage_mtp_verify_gdn_conv_prefix_state_inner(conv_buffers, prefix_tokens, false)
    }

    fn stage_mtp_verify_gdn_conv_prefix_state_inner(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        prefix_tokens: usize,
        sync_after_copy: bool,
    ) -> Result<Vec<f32>, String> {
        if prefix_tokens == 0 {
            return Err("MTP verify GDN conv prefix state requires prefix_tokens > 0".to_string());
        }
        if prefix_tokens > conv_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN conv prefix state requires prefix_tokens <= window_tokens: prefix={prefix_tokens}, window={}",
                conv_buffers.window_tokens
            ));
        }
        self.collect_mtp_verify_gdn_conv_state_at_tokens(
            conv_buffers,
            prefix_tokens,
            sync_after_copy,
        )
    }

    fn stage_mtp_verify_gdn_conv_prefix_snapshot(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        prefix_tokens: usize,
    ) -> Result<super::DeltaStateSnapshot, String> {
        if prefix_tokens == 0 || prefix_tokens > conv_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN resident conv prefix out of range: prefix={prefix_tokens}, window={}",
                conv_buffers.window_tokens
            ));
        }
        let state_rows = conv_buffers.kernel_size.checked_sub(1).ok_or_else(|| {
            "MTP verify GDN resident conv kernel_size must be non-zero".to_string()
        })?;
        let state_bytes = state_rows
            .checked_mul(conv_buffers.channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "MTP verify GDN resident conv snapshot size overflow".to_string())?;
        let offset_bytes = prefix_tokens
            .checked_mul(conv_buffers.channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "MTP verify GDN resident conv snapshot offset overflow".to_string())?;
        let src_dev = conv_buffers
            .conv_input_dev
            .checked_add(offset_bytes as u64)
            .ok_or_else(|| "MTP verify GDN resident conv snapshot pointer overflow".to_string())?;
        let snapshot = self.allocate_delta_state_snapshot(state_bytes)?;
        if let Err(err) = unsafe {
            self.api
                .memcpy_dtod_async(snapshot.ptr, src_dev, state_bytes, self.stream)
        } {
            let _ = self.free_delta_state_snapshot(snapshot);
            return Err(err);
        }
        Ok(snapshot)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_conv_final_state(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
    ) -> Result<Vec<f32>, String> {
        self.stage_mtp_verify_gdn_conv_final_state_inner(conv_buffers, true)
    }

    pub(in crate::runtime) fn stage_mtp_verify_gdn_conv_final_state_deferred(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
    ) -> Result<Vec<f32>, String> {
        self.stage_mtp_verify_gdn_conv_final_state_inner(conv_buffers, false)
    }

    fn stage_mtp_verify_gdn_conv_final_state_inner(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        sync_after_copy: bool,
    ) -> Result<Vec<f32>, String> {
        if conv_buffers.device_resident_state {
            let state_rows = conv_buffers.kernel_size.checked_sub(1).ok_or_else(|| {
                "MTP verify GDN resident conv kernel_size must be non-zero".to_string()
            })?;
            let state_bytes = state_rows
                .checked_mul(conv_buffers.channels)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "MTP verify GDN resident conv state size overflow".to_string())?;
            let offset_bytes = conv_buffers
                .window_tokens
                .checked_mul(conv_buffers.channels)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "MTP verify GDN resident conv state offset overflow".to_string())?;
            let src_dev = conv_buffers
                .conv_input_dev
                .checked_add(offset_bytes as u64)
                .ok_or_else(|| "MTP verify GDN resident conv state pointer overflow".to_string())?;
            unsafe {
                self.api.memcpy_dtod_async(
                    conv_buffers.conv_state_dev,
                    src_dev,
                    state_bytes,
                    self.stream,
                )?;
            }
            return Ok(Vec::new());
        }
        self.collect_mtp_verify_gdn_conv_state_at_tokens(
            conv_buffers,
            conv_buffers.window_tokens,
            sync_after_copy,
        )
    }

    fn collect_mtp_verify_gdn_conv_state_at_tokens(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        tokens: usize,
        sync_after_copy: bool,
    ) -> Result<Vec<f32>, String> {
        if tokens == 0 {
            return Err("MTP verify GDN conv state capture requires tokens > 0".to_string());
        }
        if tokens > conv_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN conv state capture requires tokens <= window_tokens: tokens={tokens}, window={}",
                conv_buffers.window_tokens
            ));
        }
        let state_rows = conv_buffers
            .kernel_size
            .checked_sub(1)
            .ok_or_else(|| "MTP verify GDN conv prefix kernel_size must be non-zero".to_string())?;
        let state_values = state_rows
            .checked_mul(conv_buffers.channels)
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN conv prefix state value overflow: rows={state_rows} channels={}",
                    conv_buffers.channels
                )
            })?;
        let offset_bytes = tokens
            .checked_mul(conv_buffers.channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN conv state byte offset overflow: tokens={tokens} channels={}",
                    conv_buffers.channels
                )
            })?;
        let offset_bytes = u64::try_from(offset_bytes).map_err(|_| {
            format!("MTP verify GDN conv prefix state byte offset exceeds u64: {offset_bytes}")
        })?;
        let src_dev = conv_buffers
            .conv_input_dev
            .checked_add(offset_bytes)
            .ok_or_else(|| "MTP verify GDN conv prefix state pointer overflow".to_string())?;
        let mut conv_state = vec![0.0f32; state_values];
        unsafe {
            self.api.memcpy_dtoh_async(
                conv_state.as_mut_ptr().cast::<libc::c_void>(),
                src_dev,
                std::mem::size_of_val(conv_state.as_slice()),
                self.stream,
            )?;
        }
        if sync_after_copy {
            self.stream_synchronize()?;
        }
        Ok(conv_state)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_delta_inputs(
        &mut self,
        conv_buffers: &MtpVerifyGdnConvBuffers,
        projection_buffers: &MtpVerifyGdnProjectionBuffers,
        dt_bias: &[f32],
        ssm_a: &[f32],
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        norm_eps: f32,
    ) -> Result<MtpVerifyGdnDeltaInputBuffers, String> {
        if conv_buffers.window_tokens != projection_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN delta window mismatch: conv={}, projections={}",
                conv_buffers.window_tokens, projection_buffers.window_tokens
            ));
        }
        if num_k_heads == 0 || num_v_heads == 0 {
            return Err(format!(
                "MTP verify GDN delta heads must be non-zero: k={num_k_heads}, v={num_v_heads}"
            ));
        }
        if num_v_heads % num_k_heads != 0 {
            return Err(format!(
                "MTP verify GDN delta num_v_heads must be multiple of num_k_heads: v={num_v_heads}, k={num_k_heads}"
            ));
        }
        if dt_bias.len() != num_v_heads {
            return Err(format!(
                "MTP verify GDN delta dt_bias length mismatch: got {}, expected {num_v_heads}",
                dt_bias.len()
            ));
        }
        if ssm_a.len() != num_v_heads {
            return Err(format!(
                "MTP verify GDN delta ssm_a length mismatch: got {}, expected {num_v_heads}",
                ssm_a.len()
            ));
        }
        let q_dim = num_k_heads.checked_mul(head_k_dim).ok_or_else(|| {
            format!(
                "MTP verify GDN delta q dim overflow: heads={num_k_heads}, head_k_dim={head_k_dim}"
            )
        })?;
        let k_dim = q_dim;
        let v_dim = num_v_heads.checked_mul(head_v_dim).ok_or_else(|| {
            format!(
                "MTP verify GDN delta v dim overflow: heads={num_v_heads}, head_v_dim={head_v_dim}"
            )
        })?;
        let expected_conv_channels = q_dim
            .checked_add(k_dim)
            .and_then(|sum| sum.checked_add(v_dim))
            .ok_or_else(|| "MTP verify GDN delta conv channel overflow".to_string())?;
        if conv_buffers.channels != expected_conv_channels {
            return Err(format!(
                "MTP verify GDN delta conv channel mismatch: got {}, expected {expected_conv_channels}",
                conv_buffers.channels
            ));
        }
        if projection_buffers.alpha_rows != num_v_heads {
            return Err(format!(
                "MTP verify GDN delta alpha rows mismatch: got {}, expected {num_v_heads}",
                projection_buffers.alpha_rows
            ));
        }
        if projection_buffers.beta_rows != num_v_heads {
            return Err(format!(
                "MTP verify GDN delta beta rows mismatch: got {}, expected {num_v_heads}",
                projection_buffers.beta_rows
            ));
        }

        let delta_buffers = self.ensure_mtp_verify_gdn_delta_input_buffers(
            conv_buffers.window_tokens,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )?;
        let dt_bias_dev = self.resident_f32_ptr(dt_bias)?;
        let ssm_a_dev = self.resident_f32_ptr(ssm_a)?;
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        self.launch_gdn_prepare_delta_qkv_f32(
            delta_buffers.q_dev,
            delta_buffers.k_dev,
            delta_buffers.v_dev,
            conv_buffers.conv_out_dev,
            conv_buffers.window_tokens,
            conv_buffers.channels,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            norm_eps,
            q_scale,
        )?;
        self.launch_gdn_prepare_delta_gate_beta_f32(
            delta_buffers.gate_dev,
            delta_buffers.beta_dev,
            projection_buffers.alpha_dev,
            projection_buffers.beta_dev,
            dt_bias_dev,
            ssm_a_dev,
            conv_buffers.window_tokens * num_v_heads,
            num_v_heads,
        )?;
        Ok(delta_buffers)
    }

    fn mtp_verify_gdn_delta_decode_tokenwise_enabled(
        &self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
    ) -> bool {
        delta_buffers.window_tokens <= 4
    }

    fn launch_mtp_verify_gdn_delta_decode_token(
        &mut self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
        scan_buffers: &MtpVerifyGdnDeltaScanBuffers,
        state_dev: u64,
        token_idx: usize,
    ) -> Result<(), String> {
        if token_idx >= delta_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN delta token index out of range: token_idx={token_idx}, window={}",
                delta_buffers.window_tokens
            ));
        }
        let qk_values = delta_buffers
            .num_heads
            .checked_mul(delta_buffers.head_k_dim)
            .ok_or_else(|| "MTP verify GDN delta q/k token stride overflow".to_string())?;
        let v_values = delta_buffers
            .num_heads
            .checked_mul(delta_buffers.head_v_dim)
            .ok_or_else(|| "MTP verify GDN delta v token stride overflow".to_string())?;
        let head_values = delta_buffers.num_heads;
        let qk_offset = token_idx
            .checked_mul(qk_values)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "MTP verify GDN delta q/k byte offset overflow".to_string())?;
        let v_offset = token_idx
            .checked_mul(v_values)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "MTP verify GDN delta v byte offset overflow".to_string())?;
        let head_offset = token_idx
            .checked_mul(head_values)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "MTP verify GDN delta gate byte offset overflow".to_string())?;
        let q_dev = delta_buffers.q_dev + qk_offset as u64;
        let k_dev = delta_buffers.k_dev + qk_offset as u64;
        let v_dev = delta_buffers.v_dev + v_offset as u64;
        let gate_dev = delta_buffers.gate_dev + head_offset as u64;
        let beta_dev = delta_buffers.beta_dev + head_offset as u64;
        let output_dev = scan_buffers.output_dev + v_offset as u64;

        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut gate_arg = gate_dev;
        let mut beta_arg = beta_dev;
        let mut heads_arg = u32::try_from(delta_buffers.num_heads).map_err(|_| {
            format!(
                "MTP verify GDN delta tokenwise num_heads exceeds u32: {}",
                delta_buffers.num_heads
            )
        })?;
        let mut head_k_arg = u32::try_from(delta_buffers.head_k_dim).map_err(|_| {
            format!(
                "MTP verify GDN delta tokenwise head_k_dim exceeds u32: {}",
                delta_buffers.head_k_dim
            )
        })?;
        let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
            format!(
                "MTP verify GDN delta tokenwise head_v_dim exceeds u32: {}",
                delta_buffers.head_v_dim
            )
        })?;
        let (kernel, block) = if delta_buffers.head_k_dim == 128 {
            ("rnb_delta_net_decode_predecay_hd128", (128, 1, 1))
        } else {
            ("rnb_delta_net_decode_predecay", (256, 1, 1))
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                delta_buffers.head_v_dim as u32,
                delta_buffers.num_heads as u32,
                1,
            ),
            block,
        )
    }

    fn stage_mtp_verify_gdn_delta_scan_decode_tokenwise(
        &mut self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
        scan_buffers: &MtpVerifyGdnDeltaScanBuffers,
        state_dev: u64,
    ) -> Result<(), String> {
        for token_idx in 0..delta_buffers.window_tokens {
            self.launch_mtp_verify_gdn_delta_decode_token(
                delta_buffers,
                scan_buffers,
                state_dev,
                token_idx,
            )?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_delta_scan(
        &mut self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
        state: &mut [f32],
        sync_state_to_host: bool,
    ) -> Result<MtpVerifyGdnDeltaScanBuffers, String> {
        let expected_state_values = delta_buffers
            .num_heads
            .checked_mul(delta_buffers.head_v_dim)
            .and_then(|values| values.checked_mul(delta_buffers.head_k_dim))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta state size overflow: heads={}, head_v_dim={}, head_k_dim={}",
                    delta_buffers.num_heads, delta_buffers.head_v_dim, delta_buffers.head_k_dim
                )
            })?;
        if state.len() != expected_state_values {
            return Err(format!(
                "MTP verify GDN delta state length mismatch: got {}, expected {expected_state_values}",
                state.len()
            ));
        }

        let scan_buffers = self.ensure_mtp_verify_gdn_delta_scan_buffers(delta_buffers)?;
        let state_dev = self.resident_delta_state_ptr(state)?;
        if self.mtp_verify_gdn_delta_decode_tokenwise_enabled(delta_buffers) {
            self.stage_mtp_verify_gdn_delta_scan_decode_tokenwise(
                delta_buffers,
                &scan_buffers,
                state_dev,
            )?;
            if sync_state_to_host {
                unsafe {
                    self.api.memcpy_dtoh_async(
                        state.as_mut_ptr().cast::<libc::c_void>(),
                        state_dev,
                        std::mem::size_of_val(state),
                        self.stream,
                    )?;
                }
            }
            return Ok(scan_buffers);
        }
        let mut output_arg = scan_buffers.output_dev;
        let mut state_arg = state_dev;
        let mut q_arg = delta_buffers.q_dev;
        let mut k_arg = delta_buffers.k_dev;
        let mut v_arg = delta_buffers.v_dev;
        let mut gate_arg = delta_buffers.gate_dev;
        let mut beta_arg = delta_buffers.beta_dev;
        let mut snapshot_arg = 0_u64;
        let mut seq_arg = u32::try_from(delta_buffers.window_tokens).map_err(|_| {
            format!(
                "MTP verify GDN delta seq_len exceeds CUDA u32 limit: {}",
                delta_buffers.window_tokens
            )
        })?;
        let mut heads_arg = u32::try_from(delta_buffers.num_heads).map_err(|_| {
            format!(
                "MTP verify GDN delta num_heads exceeds CUDA u32 limit: {}",
                delta_buffers.num_heads
            )
        })?;
        let mut snapshot_after_arg = 0_u32;
        if delta_buffers.head_k_dim == 128 && crate::tuning::prefill_delta_k128_warp4_enabled() {
            let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                format!(
                    "MTP verify GDN delta head_v_dim exceeds CUDA u32 limit: {}",
                    delta_buffers.head_v_dim
                )
            })?;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill_k128_warp4",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (
                    ((delta_buffers.head_v_dim as u32).saturating_add(3)) / 4,
                    delta_buffers.num_heads as u32,
                    1,
                ),
                (32, 4, 1),
            )?;
        } else if delta_buffers.head_k_dim == 128 {
            let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                format!(
                    "MTP verify GDN delta head_v_dim exceeds CUDA u32 limit: {}",
                    delta_buffers.head_v_dim
                )
            })?;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill_k128",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (
                    delta_buffers.head_v_dim as u32,
                    delta_buffers.num_heads as u32,
                    1,
                ),
                (128, 1, 1),
            )?;
        } else {
            let mut head_k_arg = u32::try_from(delta_buffers.head_k_dim).map_err(|_| {
                format!(
                    "MTP verify GDN delta head_k_dim exceeds CUDA u32 limit: {}",
                    delta_buffers.head_k_dim
                )
            })?;
            let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                format!(
                    "MTP verify GDN delta head_v_dim exceeds CUDA u32 limit: {}",
                    delta_buffers.head_v_dim
                )
            })?;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (
                    delta_buffers.head_v_dim as u32,
                    delta_buffers.num_heads as u32,
                    1,
                ),
                (256, 1, 1),
            )?;
        }

        if sync_state_to_host {
            unsafe {
                self.api.memcpy_dtoh_async(
                    state.as_mut_ptr().cast::<libc::c_void>(),
                    state_dev,
                    std::mem::size_of_val(state),
                    self.stream,
                )?;
            }
        }
        Ok(scan_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_delta_scan_snapshots(
        &mut self,
        delta_buffers: &MtpVerifyGdnDeltaInputBuffers,
        state: &mut [f32],
        sync_state_to_host: bool,
        snapshot_after_tokens: &[usize],
    ) -> Result<(MtpVerifyGdnDeltaScanBuffers, Vec<super::DeltaStateSnapshot>), String> {
        if snapshot_after_tokens.is_empty() {
            return Err("MTP verify GDN delta snapshots require at least one prefix".to_string());
        }
        for &tokens in snapshot_after_tokens {
            if tokens == 0 || tokens > delta_buffers.window_tokens {
                return Err(format!(
                    "MTP verify GDN delta snapshot token count {tokens} out of range for seq_len {}",
                    delta_buffers.window_tokens
                ));
            }
        }
        let expected_state_values = delta_buffers
            .num_heads
            .checked_mul(delta_buffers.head_v_dim)
            .and_then(|values| values.checked_mul(delta_buffers.head_k_dim))
            .ok_or_else(|| {
                format!(
                    "MTP verify GDN delta snapshot state size overflow: heads={}, head_v_dim={}, head_k_dim={}",
                    delta_buffers.num_heads, delta_buffers.head_v_dim, delta_buffers.head_k_dim
                )
            })?;
        if state.len() != expected_state_values {
            return Err(format!(
                "MTP verify GDN delta snapshot state length mismatch: got {}, expected {expected_state_values}",
                state.len()
            ));
        }

        let scan_buffers = self.ensure_mtp_verify_gdn_delta_scan_buffers(delta_buffers)?;
        let state_dev = self.resident_delta_state_ptr(state)?;
        let state_bytes = std::mem::size_of_val(state);
        if self.mtp_verify_gdn_delta_decode_tokenwise_enabled(delta_buffers) {
            let mut snapshots = Vec::with_capacity(snapshot_after_tokens.len());
            let result = (|| {
                for _ in snapshot_after_tokens {
                    snapshots.push(self.allocate_delta_state_snapshot(state_bytes)?);
                }
                let mut snapshot_cursor = 0usize;
                for token_idx in 0..delta_buffers.window_tokens {
                    self.launch_mtp_verify_gdn_delta_decode_token(
                        delta_buffers,
                        &scan_buffers,
                        state_dev,
                        token_idx,
                    )?;
                    let prefix = token_idx + 1;
                    while snapshot_cursor < snapshot_after_tokens.len()
                        && snapshot_after_tokens[snapshot_cursor] == prefix
                    {
                        unsafe {
                            self.api.memcpy_dtod_async(
                                snapshots[snapshot_cursor].ptr,
                                state_dev,
                                state_bytes,
                                self.stream,
                            )?;
                        }
                        snapshot_cursor += 1;
                    }
                }
                if snapshot_cursor != snapshot_after_tokens.len() {
                    return Err(format!(
                        "MTP verify GDN delta tokenwise snapshots missed prefixes: captured={} requested={}",
                        snapshot_cursor,
                        snapshot_after_tokens.len()
                    ));
                }
                if sync_state_to_host {
                    unsafe {
                        self.api.memcpy_dtoh_async(
                            state.as_mut_ptr().cast::<libc::c_void>(),
                            state_dev,
                            state_bytes,
                            self.stream,
                        )?;
                    }
                }
                Ok(())
            })();
            if let Err(err) = result {
                for snapshot in snapshots {
                    self.free_delta_state_snapshot(snapshot)?;
                }
                return Err(err);
            }
            return Ok((scan_buffers, snapshots));
        }
        let mut snapshots = Vec::with_capacity(snapshot_after_tokens.len());
        let result = (|| {
            for _ in snapshot_after_tokens {
                snapshots.push(self.allocate_delta_state_snapshot(state_bytes)?);
            }
            let snapshot_ptrs = snapshots
                .iter()
                .map(|snapshot| snapshot.ptr)
                .collect::<Vec<_>>();
            let snapshot_tokens = snapshot_after_tokens
                .iter()
                .map(|&tokens| {
                    u32::try_from(tokens).map_err(|_| {
                        format!("MTP verify GDN delta snapshot token count exceeds u32: {tokens}")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let snapshot_ptrs_dev =
                self.compute_down_ptrs_ptr(std::mem::size_of_val(snapshot_ptrs.as_slice()))?;
            let snapshot_tokens_dev =
                self.compute_token_ids_ptr(std::mem::size_of_val(snapshot_tokens.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    snapshot_ptrs_dev,
                    snapshot_ptrs.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(snapshot_ptrs.as_slice()),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    snapshot_tokens_dev,
                    snapshot_tokens.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(snapshot_tokens.as_slice()),
                    self.stream,
                )?;
            }

            let mut output_arg = scan_buffers.output_dev;
            let mut state_arg = state_dev;
            let mut q_arg = delta_buffers.q_dev;
            let mut k_arg = delta_buffers.k_dev;
            let mut v_arg = delta_buffers.v_dev;
            let mut gate_arg = delta_buffers.gate_dev;
            let mut beta_arg = delta_buffers.beta_dev;
            let mut snapshot_ptrs_arg = snapshot_ptrs_dev;
            let mut snapshot_tokens_arg = snapshot_tokens_dev;
            let mut snapshot_count_arg =
                u32::try_from(snapshot_after_tokens.len()).map_err(|_| {
                    format!(
                        "MTP verify GDN delta snapshot count exceeds u32: {}",
                        snapshot_after_tokens.len()
                    )
                })?;
            let mut seq_arg = u32::try_from(delta_buffers.window_tokens).map_err(|_| {
                format!(
                    "MTP verify GDN delta snapshot seq_len exceeds u32: {}",
                    delta_buffers.window_tokens
                )
            })?;
            let mut heads_arg = u32::try_from(delta_buffers.num_heads).map_err(|_| {
                format!(
                    "MTP verify GDN delta snapshot num_heads exceeds u32: {}",
                    delta_buffers.num_heads
                )
            })?;
            if delta_buffers.head_k_dim == 128 && crate::tuning::prefill_delta_k128_warp4_enabled()
            {
                let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                    format!(
                        "MTP verify GDN delta snapshot head_v_dim exceeds u32: {}",
                        delta_buffers.head_v_dim
                    )
                })?;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_k128_warp4_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (
                        ((delta_buffers.head_v_dim as u32).saturating_add(3)) / 4,
                        delta_buffers.num_heads as u32,
                        1,
                    ),
                    (32, 4, 1),
                )?;
            } else if delta_buffers.head_k_dim == 128 {
                let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                    format!(
                        "MTP verify GDN delta snapshot head_v_dim exceeds u32: {}",
                        delta_buffers.head_v_dim
                    )
                })?;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_k128_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (
                        delta_buffers.head_v_dim as u32,
                        delta_buffers.num_heads as u32,
                        1,
                    ),
                    (128, 1, 1),
                )?;
            } else {
                let mut head_k_arg = u32::try_from(delta_buffers.head_k_dim).map_err(|_| {
                    format!(
                        "MTP verify GDN delta snapshot head_k_dim exceeds u32: {}",
                        delta_buffers.head_k_dim
                    )
                })?;
                let mut head_v_arg = u32::try_from(delta_buffers.head_v_dim).map_err(|_| {
                    format!(
                        "MTP verify GDN delta snapshot head_v_dim exceeds u32: {}",
                        delta_buffers.head_v_dim
                    )
                })?;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (
                        delta_buffers.head_v_dim as u32,
                        delta_buffers.num_heads as u32,
                        1,
                    ),
                    (256, 1, 1),
                )?;
            }

            if sync_state_to_host {
                unsafe {
                    self.api.memcpy_dtoh_async(
                        state.as_mut_ptr().cast::<libc::c_void>(),
                        state_dev,
                        state_bytes,
                        self.stream,
                    )?;
                }
                self.stream_synchronize()?;
            }
            Ok(scan_buffers)
        })();

        match result {
            Ok(scan_buffers) => Ok((scan_buffers, snapshots)),
            Err(err) => {
                self.stream_synchronize().ok();
                for snapshot in snapshots {
                    unsafe {
                        let _ = self.api.mem_free(snapshot.ptr);
                    }
                }
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_ssm_out_q4k(
        &mut self,
        scan_buffers: &MtpVerifyGdnDeltaScanBuffers,
        projection_buffers: &MtpVerifyGdnProjectionBuffers,
        ssm_norm: &[f32],
        ssm_out_q4k: &[u8],
        ssm_out_quant: u32,
        ssm_out_rows: usize,
        ssm_out_cols: usize,
        norm_eps: f32,
    ) -> Result<MtpVerifyGdnSsmOutBuffers, String> {
        if scan_buffers.window_tokens != projection_buffers.window_tokens {
            return Err(format!(
                "MTP verify GDN ssm_out window mismatch: scan={}, projections={}",
                scan_buffers.window_tokens, projection_buffers.window_tokens
            ));
        }
        if ssm_norm.len() != scan_buffers.head_v_dim {
            return Err(format!(
                "MTP verify GDN ssm_norm length mismatch: got {}, expected {}",
                ssm_norm.len(),
                scan_buffers.head_v_dim
            ));
        }
        let d_inner = scan_buffers
            .num_heads
            .checked_mul(scan_buffers.head_v_dim)
            .ok_or_else(|| "MTP verify GDN ssm_out d_inner overflow".to_string())?;
        if projection_buffers.gate_rows != d_inner {
            return Err(format!(
                "MTP verify GDN gate rows mismatch for ssm_out: got {}, expected {d_inner}",
                projection_buffers.gate_rows
            ));
        }
        let blocks_per_row = validate_mtp_verify_k_quant_matrix(
            "GDN ssm_out",
            ssm_out_quant,
            ssm_out_q4k,
            ssm_out_rows,
            ssm_out_cols,
            d_inner,
        )?;

        let ssm_buffers = self.ensure_mtp_verify_gdn_ssm_out_buffers(
            scan_buffers.window_tokens,
            d_inner,
            ssm_out_rows,
        )?;
        let norm_dev = self.resident_f32_ptr(ssm_norm)?;
        self.launch_gdn_gated_norm_silu_dev(
            ssm_buffers.gated_dev,
            scan_buffers.output_dev,
            projection_buffers.gate_dev,
            norm_dev,
            scan_buffers.window_tokens * scan_buffers.num_heads,
            scan_buffers.head_v_dim,
            norm_eps,
        )?;
        self.stage_mtp_verify_k_quant_projection_to_dev(
            "GDN ssm_out",
            ssm_out_quant,
            ssm_out_q4k,
            ssm_out_rows,
            blocks_per_row,
            scan_buffers.window_tokens,
            ssm_buffers.gated_dev,
            ssm_buffers.ssm_out_dev,
        )?;
        Ok(ssm_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_residual_post_norm(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        ssm_buffers: &MtpVerifyGdnSsmOutBuffers,
        post_attn_norm: &[f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if ssm_buffers.window_tokens != plan.window_tokens {
            return Err(format!(
                "MTP verify GDN residual window mismatch: ssm={}, plan={}",
                ssm_buffers.window_tokens, plan.window_tokens
            ));
        }
        if ssm_buffers.hidden_dim != plan.hidden_dim {
            return Err(format!(
                "MTP verify GDN residual hidden mismatch: ssm={}, plan={}",
                ssm_buffers.hidden_dim, plan.hidden_dim
            ));
        }
        if post_attn_norm.len() != plan.hidden_dim {
            return Err(format!(
                "MTP verify GDN post_attn_norm length mismatch: got {}, expected {}",
                post_attn_norm.len(),
                plan.hidden_dim
            ));
        }
        self.launch_add_f32_inplace(
            buffers.hidden_rows_dev,
            ssm_buffers.ssm_out_dev,
            plan.window_tokens * plan.hidden_dim,
        )?;
        self.stage_mtp_verify_hidden_rows_rms_norm(buffers, post_attn_norm, norm_eps, false)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_router_topk(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        router_w: &[f32],
        n_expert: usize,
        n_expert_used: usize,
    ) -> Result<MtpVerifyQwen35RouterBuffers, String> {
        let plan = &buffers.plan;
        let expected_router = n_expert.checked_mul(plan.hidden_dim).ok_or_else(|| {
            format!(
                "MTP verify Qwen35 router weight shape overflow: experts={n_expert}, hidden_dim={}",
                plan.hidden_dim
            )
        })?;
        if router_w.len() != expected_router {
            return Err(format!(
                "MTP verify Qwen35 router weight len mismatch: got {}, expected {expected_router}",
                router_w.len()
            ));
        }
        let router_buffers =
            self.ensure_mtp_verify_qwen35_router_buffers(plan, n_expert, n_expert_used)?;
        let router_dev = if crate::tuning::mtp_verify_router_stable_key_enabled() {
            self.resident_f32_ptr_stable_source(router_w)?
        } else {
            self.resident_f32_ptr(router_w)?
        };
        self.sgemm_device(
            router_dev,
            n_expert,
            plan.hidden_dim,
            buffers.scratch_hidden_dev,
            plan.window_tokens,
            router_buffers.logits_dev,
        )?;
        self.launch_qwen35_router_topk_softmax_f32(
            router_buffers.logits_dev,
            router_buffers.expert_ids_dev,
            router_buffers.route_weights_dev,
            router_buffers.token_ids_dev,
            plan.window_tokens,
            n_expert,
            n_expert_used,
        )?;
        Ok(router_buffers)
    }

    fn collect_mtp_verify_qwen35_router_slots(
        &mut self,
        router_buffers: &MtpVerifyQwen35RouterBuffers,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>), String> {
        let slots = router_buffers
            .window_tokens
            .checked_mul(router_buffers.n_expert_used)
            .ok_or_else(|| {
                format!(
                    "MTP verify Qwen35 router slot overflow: tokens={} used={}",
                    router_buffers.window_tokens, router_buffers.n_expert_used
                )
            })?;
        let mut expert_ids = vec![0_u32; slots];
        let mut route_weights = vec![0.0_f32; slots];
        let mut token_ids = vec![0_u32; slots];
        unsafe {
            self.api.memcpy_dtoh_async(
                expert_ids.as_mut_ptr().cast::<libc::c_void>(),
                router_buffers.expert_ids_dev,
                std::mem::size_of_val(expert_ids.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                route_weights.as_mut_ptr().cast::<libc::c_void>(),
                router_buffers.route_weights_dev,
                std::mem::size_of_val(route_weights.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                token_ids.as_mut_ptr().cast::<libc::c_void>(),
                router_buffers.token_ids_dev,
                std::mem::size_of_val(token_ids.as_slice()),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok((expert_ids, route_weights, token_ids))
    }

    fn qwen35_mtp_weight_residency_stats(
        &self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> (usize, usize, usize, usize, usize) {
        let mut seen = std::collections::HashSet::new();
        let mut unique_weights = 0usize;
        let mut active_bytes = 0usize;
        let mut resident_hits = 0usize;
        let mut resident_misses = 0usize;
        let mut miss_bytes = 0usize;
        for &weights in gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter())
        {
            let key = super::q4k_resident_key(weights);
            if !seen.insert(key) {
                continue;
            }
            unique_weights += 1;
            active_bytes = active_bytes.saturating_add(weights.len());
            if self.resident_q4k.contains_key(&key) {
                resident_hits += 1;
            } else {
                resident_misses += 1;
                miss_bytes = miss_bytes.saturating_add(weights.len());
            }
        }
        (
            unique_weights,
            active_bytes,
            resident_hits,
            resident_misses,
            miss_bytes,
        )
    }

    fn trace_and_promote_mtp_verify_experts(
        &mut self,
        layer_index: Option<usize>,
        expert_ids: &[u32],
        route_weights: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<(), String> {
        let Some(layer_index) = layer_index else {
            return Ok(());
        };
        let trace = crate::tuning::mtp_expert_trace_enabled();
        let hot_resident = crate::tuning::mtp_expert_hot_resident_enabled();
        if !trace && !hot_resident {
            return Ok(());
        }

        let previous = self.qwen35_mtp_expert_history.get(&layer_index).cloned();
        let layer_observations = self
            .qwen35_mtp_expert_observations
            .get(&layer_index)
            .copied()
            .unwrap_or(0);
        let current_unique = expert_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let current_unique_len = current_unique.len();
        let (
            unique_weights,
            active_bytes,
            resident_hits_before,
            resident_misses_before,
            _miss_bytes_before,
        ) = self.qwen35_mtp_weight_residency_stats(gate_weights, up_weights, down_weights);

        let pred_hit_slots = previous
            .as_ref()
            .map(|prev| {
                expert_ids
                    .iter()
                    .filter(|&&expert| prev.contains(&expert))
                    .count()
            })
            .unwrap_or(0);
        let pred_hit_unique = previous
            .as_ref()
            .map(|prev| {
                current_unique
                    .iter()
                    .filter(|&&expert| prev.contains(&expert))
                    .count()
            })
            .unwrap_or(0);

        let mut promoted_weights = 0usize;
        let mut promoted_bytes = 0usize;
        let mut extra_experts = 0usize;
        let mut extra_weights = 0usize;
        let mut extra_bytes = 0usize;
        let extra_budget_bytes = crate::tuning::mtp_expert_extra_resident_budget_bytes_for_layer(
            self.resident_q4k_limit,
            layer_observations,
        )
        .min(
            self.resident_q4k_limit
                .saturating_sub(self.resident_q4k_bytes),
        );
        if hot_resident {
            let mut promoted_keys = std::collections::HashSet::new();
            if let Some(previous) = previous.as_ref() {
                for (slot, expert_id) in expert_ids.iter().enumerate() {
                    if !previous.contains(expert_id) {
                        continue;
                    }
                    for weights in [gate_weights[slot], up_weights[slot], down_weights[slot]] {
                        let key = super::q4k_resident_key(weights);
                        if !promoted_keys.insert(key) || self.resident_q4k.contains_key(&key) {
                            continue;
                        }
                        if self.preload_resident_q4k_weight_slice(weights)? {
                            promoted_weights += 1;
                            promoted_bytes = promoted_bytes.saturating_add(weights.len());
                        }
                    }
                }
            }

            let mut extra_budget_left = extra_budget_bytes.min(
                self.resident_q4k_limit
                    .saturating_sub(self.resident_q4k_bytes),
            );
            if extra_budget_left > 0 {
                for candidate in expert_cache::extra_expert_candidates(
                    expert_ids,
                    route_weights,
                    previous.as_ref(),
                ) {
                    let slot = candidate.first_slot;
                    let mut missing = Vec::new();
                    let mut missing_bytes = 0usize;
                    for weights in [gate_weights[slot], up_weights[slot], down_weights[slot]] {
                        let key = super::q4k_resident_key(weights);
                        if promoted_keys.contains(&key) || self.resident_q4k.contains_key(&key) {
                            continue;
                        }
                        missing_bytes = missing_bytes.saturating_add(weights.len());
                        missing.push((key, weights));
                    }
                    if missing.is_empty() || missing_bytes > extra_budget_left {
                        continue;
                    }

                    let mut slot_promoted_weights = 0usize;
                    let mut slot_promoted_bytes = 0usize;
                    for (key, weights) in missing {
                        promoted_keys.insert(key);
                        if self.preload_resident_q4k_weight_slice(weights)? {
                            slot_promoted_weights += 1;
                            slot_promoted_bytes = slot_promoted_bytes.saturating_add(weights.len());
                        }
                    }
                    if slot_promoted_weights > 0 {
                        promoted_weights += slot_promoted_weights;
                        promoted_bytes = promoted_bytes.saturating_add(slot_promoted_bytes);
                        extra_weights += slot_promoted_weights;
                        extra_bytes = extra_bytes.saturating_add(slot_promoted_bytes);
                        extra_experts += 1;
                        extra_budget_left = extra_budget_left.saturating_sub(slot_promoted_bytes);
                    }
                }
            }
        }

        let (
            _unique_weights_after,
            _active_bytes_after,
            resident_hits_after,
            resident_misses_after,
            miss_bytes_after,
        ) = self.qwen35_mtp_weight_residency_stats(gate_weights, up_weights, down_weights);
        self.qwen35_mtp_expert_history
            .insert(layer_index, current_unique);
        *self
            .qwen35_mtp_expert_observations
            .entry(layer_index)
            .or_insert(0) += 1;

        if trace {
            let profile = profile::MtpVerifyExpertResidencyProfile {
                layer_index,
                slots: expert_ids.len(),
                predicted_hits: pred_hit_slots,
                resident_hits_before,
                resident_misses_before,
                resident_hits_after,
                resident_misses_after,
                temp_h2d_bytes: miss_bytes_after,
                promoted_bytes,
                weight_ptr_ms: 0.0,
                setup_h2d_ms: 0.0,
                kernels_ms: 0.0,
            };
            let pred_hit_rate = profile.predicted_hit_rate();
            eprintln!(
                "[cuda-mtp-expert] layer={} slots={} unique_experts={} pred_hits={}/{} pred_unique_hits={}/{} pred_hit_rate={:.1}% resident_before_hit={} resident_before_miss={} resident_after_hit={} resident_after_miss={} active_mb={:.2} temp_h2d_bytes={} temp_h2d_mb={:.2} promoted_weights={} promoted_mb={:.2}",
                layer_index,
                expert_ids.len(),
                current_unique_len,
                pred_hit_slots,
                expert_ids.len(),
                pred_hit_unique,
                current_unique_len,
                pred_hit_rate,
                resident_hits_before,
                resident_misses_before,
                resident_hits_after,
                resident_misses_after,
                active_bytes as f64 / (1024.0 * 1024.0),
                miss_bytes_after,
                miss_bytes_after as f64 / (1024.0 * 1024.0),
                promoted_weights,
                promoted_bytes as f64 / (1024.0 * 1024.0)
            );
            if extra_budget_bytes > 0 || extra_weights > 0 {
                eprintln!(
                    "[cuda-mtp-expert-extra] layer={} observations={} extra_experts={} extra_weights={} extra_mb={:.2} budget_mb={:.2}",
                    layer_index,
                    layer_observations,
                    extra_experts,
                    extra_weights,
                    extra_bytes as f64 / (1024.0 * 1024.0),
                    extra_budget_bytes as f64 / (1024.0 * 1024.0)
                );
            }
            if unique_weights == 0 {
                eprintln!(
                    "[cuda-mtp-expert] layer={} selected no weights",
                    layer_index
                );
            }
        }
        Ok(())
    }

    fn qwen35_mtp_verify_expert_byte_sizes(
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<(usize, usize), String> {
        if n_ff == 0 || n_embd == 0 {
            return Err(format!(
                "MTP verify Qwen35 selected MoE dims must be non-zero: n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if n_ff % 256 != 0 || n_embd % 256 != 0 {
            return Err(format!(
                "MTP verify Qwen35 selected MoE dims must be divisible by 256: n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        let gate_row_bytes = (n_embd / 256) * 144;
        let gate_expert_bytes = n_ff.checked_mul(gate_row_bytes).ok_or_else(|| {
            format!(
                "MTP verify Qwen35 selected gate expert byte overflow: n_ff={n_ff} n_embd={n_embd}"
            )
        })?;
        let down_row_bytes = match down_quant {
            12 => (n_ff / 256) * 144,
            13 => (n_ff / 256) * 176,
            14 => (n_ff / 256) * 210,
            other => {
                return Err(format!(
                    "MTP verify Qwen35 selected MoE unsupported down quant {other}"
                ))
            }
        };
        let down_expert_bytes = n_embd.checked_mul(down_row_bytes).ok_or_else(|| {
            format!(
                "MTP verify Qwen35 selected down expert byte overflow: n_ff={n_ff} n_embd={n_embd}"
            )
        })?;
        Ok((gate_expert_bytes, down_expert_bytes))
    }

    fn ensure_qwen35_mtp_verify_resident_moe_layer(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<bool, String> {
        let key = super::qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        if self.resident_moe_layers.contains_key(&key) {
            self.touch_resident_moe_layer(key);
            return Ok(true);
        }
        if !crate::tuning::mtp_verify_missing_moe_layer_promotion_enabled() {
            return Ok(false);
        }
        self.register_qwen35_moe_layer(gate_all, up_all, down_all, down_quant, n_ff, n_embd)?;
        Ok(self.resident_moe_layers.contains_key(&key))
    }

    pub(in crate::runtime) fn has_qwen35_mtp_verify_resident_moe_layer(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> bool {
        let key = super::qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        self.resident_moe_layers.contains_key(&key)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mtp_verify_selected_graph(
        &mut self,
        q8_gate_up: bool,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        selected: usize,
        input_dev: u64,
        hidden_dev: u64,
        gate_dev: u64,
        up_dev: u64,
        output_dev: u64,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        route_dev: u64,
        input_qs_dev: u64,
        input_ds_dev: u64,
    ) -> Result<(), String> {
        let key = super::MtpVerifySelectedGraphKey {
            q8_gate_up,
            down_quant,
            n_ff,
            n_embd,
            selected,
            input_dev,
            hidden_dev,
            gate_dev,
            up_dev,
            output_dev,
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            route_dev,
            input_qs_dev,
            input_ds_dev,
        };
        if let Some(graph) = self.mtp_verify_selected_graphs.get(&key) {
            return unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)
            };
        }
        self.ensure_q4k_gemv_module()?;
        unsafe {
            self.api.stream_begin_capture(self.stream)?;
        }
        let capture_result = (|| {
            if q8_gate_up {
                self.launch_quantize_q8_1_by_32(input_dev, input_qs_dev, input_ds_dev, n_embd)?;
                self.launch_selected_q4k_gate_up_q8dot_by_token_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    n_ff,
                    selected,
                    n_embd / 256,
                    input_qs_dev,
                    input_ds_dev,
                    gate_dev,
                    up_dev,
                )?;
            } else {
                self.launch_selected_q4k_gate_up_gemv_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    n_ff,
                    selected,
                    n_embd / 256,
                    input_dev,
                    gate_dev,
                    up_dev,
                )?;
            }
            let down_kernel = match down_quant {
                12 => "rnb_q4k_selected_down_silu_rowreduce",
                13 => "rnb_q5k_selected_down_silu_rowreduce",
                14 => "rnb_q6k_selected_down_silu_rowreduce",
                other => {
                    return Err(format!(
                        "MTP verify selected graph unsupported down quant {other}"
                    ))
                }
            };
            self.launch_selected_down_silu_rowreduce(
                down_kernel,
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            )?;
            self.launch_add_f32_inplace(hidden_dev, output_dev, n_embd)
        })();
        if let Err(err) = capture_result {
            unsafe {
                let _ = self.api.stream_end_capture(self.stream);
            }
            return Err(err);
        }
        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
        let exec = unsafe { self.api.graph_instantiate(graph)? };
        self.mtp_verify_selected_graphs.insert(
            key,
            super::SparseMoeGraph {
                graph: graph as usize,
                exec: exec as usize,
            },
        );
        let graph = self
            .mtp_verify_selected_graphs
            .get(&key)
            .ok_or_else(|| "missing MTP verify selected CUDA graph".to_string())?;
        unsafe {
            self.api
                .graph_launch(graph.exec as *mut libc::c_void, self.stream)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_sparse_moe_selected_from_router_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        router_buffers: &MtpVerifyQwen35RouterBuffers,
        layer_index: Option<usize>,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_expert: usize,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if n_embd != plan.hidden_dim {
            return Err(format!(
                "MTP verify Qwen35 selected MoE n_embd mismatch: n_embd={n_embd}, hidden_dim={}",
                plan.hidden_dim
            ));
        }
        if router_buffers.window_tokens != plan.window_tokens {
            return Err(format!(
                "MTP verify Qwen35 selected MoE router token mismatch: router={}, window={}",
                router_buffers.window_tokens, plan.window_tokens
            ));
        }
        if router_buffers.n_expert != n_expert {
            return Err(format!(
                "MTP verify Qwen35 selected MoE expert mismatch: router={}, weights={n_expert}",
                router_buffers.n_expert
            ));
        }
        let (gate_expert_bytes, down_expert_bytes) =
            Self::qwen35_mtp_verify_expert_byte_sizes(down_quant, n_ff, n_embd)?;
        let expected_gate = n_expert.checked_mul(gate_expert_bytes).ok_or_else(|| {
            format!("MTP verify Qwen35 selected gate byte overflow: experts={n_expert}")
        })?;
        let expected_down = n_expert.checked_mul(down_expert_bytes).ok_or_else(|| {
            format!("MTP verify Qwen35 selected down byte overflow: experts={n_expert}")
        })?;
        if gate_all.len() != expected_gate || up_all.len() != expected_gate {
            return Err(format!(
                "MTP verify Qwen35 selected gate/up byte mismatch: gate={} up={} expected={expected_gate}",
                gate_all.len(),
                up_all.len()
            ));
        }
        if down_all.len() != expected_down {
            return Err(format!(
                "MTP verify Qwen35 selected down byte mismatch: got {}, expected={expected_down}",
                down_all.len()
            ));
        }

        let (expert_ids, route_weights, token_ids) =
            self.collect_mtp_verify_qwen35_router_slots(router_buffers)?;
        let mut gate_weights = Vec::with_capacity(expert_ids.len());
        let mut up_weights = Vec::with_capacity(expert_ids.len());
        let mut down_weights = Vec::with_capacity(expert_ids.len());
        for &expert_id in &expert_ids {
            let expert = usize::try_from(expert_id).map_err(|_| {
                format!("MTP verify Qwen35 selected expert id exceeds usize: {expert_id}")
            })?;
            if expert >= n_expert {
                return Err(format!(
                    "MTP verify Qwen35 selected expert id out of range: got {expert}, n_expert={n_expert}"
                ));
            }
            let gate_start = expert.checked_mul(gate_expert_bytes).ok_or_else(|| {
                format!("MTP verify Qwen35 selected gate offset overflow: expert={expert}")
            })?;
            let down_start = expert.checked_mul(down_expert_bytes).ok_or_else(|| {
                format!("MTP verify Qwen35 selected down offset overflow: expert={expert}")
            })?;
            gate_weights.push(&gate_all[gate_start..gate_start + gate_expert_bytes]);
            up_weights.push(&up_all[gate_start..gate_start + gate_expert_bytes]);
            down_weights.push(&down_all[down_start..down_start + down_expert_bytes]);
        }
        self.trace_and_promote_mtp_verify_experts(
            layer_index,
            &expert_ids,
            &route_weights,
            &gate_weights,
            &up_weights,
            &down_weights,
        )?;
        self.stage_mtp_verify_gdn_sparse_moe_by_token_q4k(
            buffers,
            &gate_weights,
            &up_weights,
            &down_weights,
            &route_weights,
            &token_ids,
            down_quant,
            n_ff,
            n_embd,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_sparse_moe_full_layer_from_router_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        router_buffers: &MtpVerifyQwen35RouterBuffers,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_expert: usize,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if n_embd != plan.hidden_dim {
            return Err(format!(
                "MTP verify Qwen35 full-layer MoE n_embd mismatch: n_embd={n_embd}, hidden_dim={}",
                plan.hidden_dim
            ));
        }
        if router_buffers.window_tokens != plan.window_tokens {
            return Err(format!(
                "MTP verify Qwen35 full-layer MoE router token mismatch: router={}, window={}",
                router_buffers.window_tokens, plan.window_tokens
            ));
        }
        if router_buffers.n_expert != n_expert {
            return Err(format!(
                "MTP verify Qwen35 full-layer MoE expert mismatch: router={}, weights={n_expert}",
                router_buffers.n_expert
            ));
        }
        if n_expert == 0 || n_ff == 0 || n_embd == 0 {
            return Err(format!(
                "MTP verify Qwen35 full-layer MoE dims must be non-zero: experts={n_expert} n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if n_ff % 256 != 0 || n_embd % 256 != 0 {
            return Err(format!(
                "MTP verify Qwen35 full-layer MoE dims must be divisible by 256: n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        let gate_row_bytes = (n_embd / 256) * 144;
        let gate_expert_bytes = n_ff.checked_mul(gate_row_bytes).ok_or_else(|| {
            format!(
                "MTP verify Qwen35 full-layer gate expert byte overflow: n_ff={n_ff} n_embd={n_embd}"
            )
        })?;
        let down_row_bytes = match down_quant {
            12 => (n_ff / 256) * 144,
            13 => (n_ff / 256) * 176,
            14 => (n_ff / 256) * 210,
            other => {
                return Err(format!(
                    "MTP verify Qwen35 full-layer MoE unsupported down quant {other}"
                ))
            }
        };
        let down_expert_bytes = n_embd.checked_mul(down_row_bytes).ok_or_else(|| {
            format!(
                "MTP verify Qwen35 full-layer down expert byte overflow: n_ff={n_ff} n_embd={n_embd}"
            )
        })?;
        let expected_gate = n_expert.checked_mul(gate_expert_bytes).ok_or_else(|| {
            format!("MTP verify Qwen35 full-layer gate byte overflow: experts={n_expert}")
        })?;
        let expected_down = n_expert.checked_mul(down_expert_bytes).ok_or_else(|| {
            format!("MTP verify Qwen35 full-layer down byte overflow: experts={n_expert}")
        })?;
        if gate_all.len() != expected_gate || up_all.len() != expected_gate {
            return Err(format!(
                "MTP verify Qwen35 full-layer gate/up byte mismatch: gate={} up={} expected={expected_gate}",
                gate_all.len(),
                up_all.len()
            ));
        }
        if down_all.len() != expected_down {
            return Err(format!(
                "MTP verify Qwen35 full-layer down byte mismatch: got {}, expected={expected_down}",
                down_all.len()
            ));
        }
        let key = super::qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        let Some((gate_base, up_base, down_base)) = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.gate_base, entry.up_base, entry.down_base))
        else {
            return self.stage_mtp_verify_gdn_sparse_moe_selected_from_router_q4k(
                buffers,
                router_buffers,
                None,
                gate_all,
                up_all,
                down_all,
                down_quant,
                n_expert,
                n_ff,
                n_embd,
            );
        };
        self.touch_resident_moe_layer(key);

        let slots = router_buffers
            .window_tokens
            .checked_mul(router_buffers.n_expert_used)
            .ok_or_else(|| {
                format!(
                    "MTP verify Qwen35 full-layer MoE slot overflow: tokens={} used={}",
                    router_buffers.window_tokens, router_buffers.n_expert_used
                )
            })?;
        let gate_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let pair2_selected_map = router_buffers.window_tokens == 2
            && crate::tuning::mtp_verify_selected_pair_map_enabled();
        let gate_ptrs_bytes = slots
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "MTP verify Qwen35 gate pointer byte overflow".to_string())?;
        let pair_slots_bytes = if pair2_selected_map {
            slots
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| "MTP verify Qwen35 pair slot byte overflow".to_string())?
        } else {
            0
        };
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(
            gate_ptrs_bytes
                .checked_add(pair_slots_bytes)
                .ok_or_else(|| "MTP verify Qwen35 pair map buffer overflow".to_string())?,
        )?;
        let pair_slots_dev = if pair2_selected_map {
            gate_ptrs_dev + gate_ptrs_bytes as u64
        } else {
            0
        };
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        self.launch_qwen35_build_q4k_full_layer_slot_ptrs(
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            router_buffers.expert_ids_dev,
            pair_slots_dev,
            if pair2_selected_map {
                router_buffers.n_expert_used
            } else {
                0
            },
            gate_base,
            up_base,
            down_base,
            gate_expert_bytes,
            down_expert_bytes,
            slots,
        )?;
        let selected_q8dot = std::env::var("RNB_CUDA_MTP_VERIFY_SELECTED_Q8DOT").ok();
        let single_token_q8dot = router_buffers.window_tokens == 1
            && selected_q8dot.as_deref().is_some_and(|value| {
                value == "1"
                    || (value == "q4" && down_quant == 12)
                    || (value == "q6" && down_quant == 14)
            })
            && matches!(down_quant, 12 | 14)
            && slots > 0
            && slots <= 16
            && n_ff % 256 == 0
            && n_embd % 256 == 0;
        if single_token_q8dot {
            let mut group_meta = [0u32; 32];
            let mut pack_group_offsets = [0u32; 17];
            for slot in 0..slots {
                group_meta[slot * 2] = slot as u32;
                group_meta[slot * 2 + 1] = 1;
                pack_group_offsets[slot] = slot as u32;
            }
            pack_group_offsets[slots] = slots as u32;
            let group_meta = &group_meta[..slots * 2];
            let pack_group_offsets = &pack_group_offsets[..=slots];
            let group_meta_bytes = std::mem::size_of_val(group_meta);
            let pack_group_offsets_bytes = std::mem::size_of_val(pack_group_offsets);
            let group_meta_dev =
                self.compute_group_meta_ptr(group_meta_bytes + pack_group_offsets_bytes)?;
            let pack_group_offsets_dev = group_meta_dev + group_meta_bytes as u64;
            unsafe {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    group_meta.as_ptr().cast::<libc::c_void>(),
                    group_meta_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    pack_group_offsets_dev,
                    pack_group_offsets.as_ptr().cast::<libc::c_void>(),
                    pack_group_offsets_bytes,
                    self.stream,
                )?;
            }
            let q8_capacity = slots * n_ff;
            let qs_dev = self.compute_full_gate_ptr(q8_capacity)?;
            let ds_dev =
                self.compute_full_up_ptr((q8_capacity / 32) * std::mem::size_of::<f32>())?;
            match down_quant {
                12 => {
                    self.launch_quantize_q8_1_by_32(
                        buffers.scratch_hidden_dev,
                        qs_dev,
                        ds_dev,
                        n_embd,
                    )?;
                    self.launch_selected_q4k_gate_up_silu_q8dot_by_token_group8_to_dev(
                        gate_ptrs_dev,
                        up_ptrs_dev,
                        router_buffers.token_ids_dev,
                        group_meta_dev,
                        pack_group_offsets_dev,
                        n_ff,
                        slots,
                        n_embd / 256,
                        n_ff / 256,
                        qs_dev,
                        ds_dev,
                        gate_dev,
                        None,
                    )?;
                }
                14 => {
                    self.launch_selected_q4k_gate_up_silu_by_token_group8_to_dev(
                        gate_ptrs_dev,
                        up_ptrs_dev,
                        router_buffers.token_ids_dev,
                        group_meta_dev,
                        n_ff,
                        slots,
                        n_embd / 256,
                        buffers.scratch_hidden_dev,
                        gate_dev,
                        up_dev,
                    )?;
                }
                _ => unreachable!(),
            }
            self.launch_quantize_q8_1_by_32(gate_dev, qs_dev, ds_dev, slots * n_ff)?;
            let selected_output_dev =
                self.compute_output_ptr(n_embd * std::mem::size_of::<f32>())?;
            self.launch_zero_f32(selected_output_dev, n_embd)?;
            match down_quant {
                12 => self.launch_selected_q4k_down_accum_by_token_group4_q8dot(
                    down_ptrs_dev,
                    router_buffers.token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    qs_dev,
                    ds_dev,
                    router_buffers.route_weights_dev,
                    selected_output_dev,
                )?,
                14 => self.launch_selected_q6k_down_accum_by_token_group4_q8dot(
                    down_ptrs_dev,
                    router_buffers.token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    qs_dev,
                    ds_dev,
                    router_buffers.route_weights_dev,
                    selected_output_dev,
                )?,
                _ => unreachable!(),
            }
            return self.launch_add_f32_inplace(
                buffers.hidden_rows_dev,
                selected_output_dev,
                n_embd,
            );
        }
        let selected_graph = std::env::var("RNB_CUDA_MTP_VERIFY_SELECTED_GRAPH")
            .ok()
            .as_deref()
            == Some("1");
        let selected_graph_q8 = std::env::var("RNB_CUDA_MTP_VERIFY_SELECTED_GRAPH_Q8")
            .ok()
            .as_deref()
            == Some("1");
        if router_buffers.window_tokens == 1 && (selected_graph || selected_graph_q8) {
            let selected_output_dev =
                self.compute_output_ptr(n_embd * std::mem::size_of::<f32>())?;
            let (input_qs_dev, input_ds_dev) = if selected_graph_q8 {
                (
                    self.compute_full_gate_ptr(n_embd)?,
                    self.compute_full_up_ptr((n_embd / 32) * std::mem::size_of::<f32>())?,
                )
            } else {
                (0, 0)
            };
            return self.launch_mtp_verify_selected_graph(
                selected_graph_q8,
                down_quant,
                n_ff,
                n_embd,
                slots,
                buffers.scratch_hidden_dev,
                buffers.hidden_rows_dev,
                gate_dev,
                up_dev,
                selected_output_dev,
                gate_ptrs_dev,
                up_ptrs_dev,
                down_ptrs_dev,
                router_buffers.route_weights_dev,
                input_qs_dev,
                input_ds_dev,
            );
        }
        let pair2_gate_up = router_buffers.window_tokens == 2
            && crate::tuning::mtp_verify_selected_gate_pair2_enabled();
        let pair2_gate_up_silu =
            pair2_gate_up && crate::tuning::mtp_verify_selected_gate_pair2_silu_enabled();
        if router_buffers.window_tokens == 1
            && crate::tuning::mtp_verify_selected_q8_gate_up_enabled()
        {
            let input_qs_dev = self.compute_full_gate_ptr(n_embd)?;
            let input_ds_dev =
                self.compute_full_up_ptr((n_embd / 32) * std::mem::size_of::<f32>())?;
            self.launch_quantize_q8_1_by_32(
                buffers.scratch_hidden_dev,
                input_qs_dev,
                input_ds_dev,
                n_embd,
            )?;
            self.launch_selected_q4k_gate_up_q8dot_by_token_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                n_ff,
                slots,
                n_embd / 256,
                input_qs_dev,
                input_ds_dev,
                gate_dev,
                up_dev,
            )?;
        } else if pair2_gate_up {
            self.launch_selected_q4k_gate_up_gemv_pair2_by_token_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                router_buffers.expert_ids_dev,
                pair_slots_dev,
                router_buffers.token_ids_dev,
                n_ff,
                router_buffers.n_expert_used,
                n_embd / 256,
                buffers.scratch_hidden_dev,
                gate_dev,
                up_dev,
                pair2_gate_up_silu,
            )?;
        } else {
            self.launch_selected_q4k_gate_up_gemv_by_token_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                router_buffers.token_ids_dev,
                n_ff,
                slots,
                n_embd / 256,
                buffers.scratch_hidden_dev,
                gate_dev,
                up_dev,
            )?;
        }
        if router_buffers.window_tokens == 1 {
            let selected_output_dev =
                self.compute_output_ptr(n_embd * std::mem::size_of::<f32>())?;
            match down_quant {
                12 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q4k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    router_buffers.route_weights_dev,
                    selected_output_dev,
                )?,
                13 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q5k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    router_buffers.route_weights_dev,
                    selected_output_dev,
                )?,
                14 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q6k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    router_buffers.route_weights_dev,
                    selected_output_dev,
                )?,
                other => {
                    return Err(format!(
                    "MTP verify Qwen35 single-token full-layer MoE unsupported down quant {other}"
                ))
                }
            }
            return self.launch_add_f32_inplace(
                buffers.hidden_rows_dev,
                selected_output_dev,
                n_embd,
            );
        }
        if !pair2_gate_up_silu {
            self.launch_silu_mul(gate_dev, up_dev, slots * n_ff)?;
        }
        let warp_down = std::env::var("RNB_CUDA_WARP_DOWN").ok().as_deref() != Some("0");
        let pair2_down = router_buffers.window_tokens == 2
            && down_quant == 13
            && warp_down
            && crate::tuning::mtp_verify_selected_down_pair2_enabled();
        match (down_quant, warp_down) {
            (12, true) => self.launch_selected_down_accum_by_token_warp4(
                "rnb_q4k_selected_down_accum_by_token_warp4",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (13, true) if pair2_down => self.launch_selected_q5k_down_accum_by_token_pair2_warp4(
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                router_buffers.expert_ids_dev,
                pair_slots_dev,
                n_embd,
                router_buffers.n_expert_used,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (13, true) => self.launch_selected_down_accum_by_token_warp4(
                "rnb_q5k_selected_down_accum_by_token_warp4",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (14, true) => self.launch_selected_down_accum_by_token_warp4(
                "rnb_q6k_selected_down_accum_by_token_warp4",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (12, false) => self.launch_selected_down_accum_by_token(
                "rnb_q4k_selected_down_accum_by_token",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (13, false) => self.launch_selected_down_accum_by_token(
                "rnb_q5k_selected_down_accum_by_token",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (14, false) => self.launch_selected_down_accum_by_token(
                "rnb_q6k_selected_down_accum_by_token",
                down_ptrs_dev,
                router_buffers.token_ids_dev,
                n_embd,
                slots,
                n_ff / 256,
                gate_dev,
                router_buffers.route_weights_dev,
                buffers.hidden_rows_dev,
            )?,
            (other, _) => {
                return Err(format!(
                    "MTP verify Qwen35 full-layer MoE unsupported down quant {other}"
                ))
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_shared_expert_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        shared_input_scale: &[f32],
        shared_gate: &[u8],
        shared_gate_quant: u32,
        shared_up: &[u8],
        shared_up_quant: u32,
        shared_down: &[u8],
        shared_down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if n_embd != plan.hidden_dim {
            return Err(format!(
                "MTP verify Qwen35 shared expert n_embd mismatch: n_embd={n_embd}, hidden_dim={}",
                plan.hidden_dim
            ));
        }
        if n_ff == 0 || n_embd == 0 {
            return Err(format!(
                "MTP verify Qwen35 shared expert dims must be non-zero: n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if n_ff % 256 != 0 || n_embd % 256 != 0 {
            return Err(format!(
                "MTP verify Qwen35 shared expert dims must be divisible by 256: n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if shared_input_scale.len() != n_embd {
            return Err(format!(
                "MTP verify Qwen35 shared input scale len mismatch: got {}, expected {n_embd}",
                shared_input_scale.len()
            ));
        }
        let shared_gate_blocks = validate_mtp_verify_k_quant_matrix(
            "Qwen35 shared gate",
            shared_gate_quant,
            shared_gate,
            n_ff,
            n_embd,
            n_embd,
        )?;
        let shared_up_blocks = validate_mtp_verify_k_quant_matrix(
            "Qwen35 shared up",
            shared_up_quant,
            shared_up,
            n_ff,
            n_embd,
            n_embd,
        )?;
        let shared_down_blocks = validate_mtp_verify_k_quant_matrix(
            "Qwen35 shared down",
            shared_down_quant,
            shared_down,
            n_embd,
            n_ff,
            n_ff,
        )?;

        let expanded_q4k_path = crate::tuning::expanded_weight_cache_allowed()
            && shared_gate_quant == GGML_Q4_K
            && shared_up_quant == GGML_Q4_K
            && matches!(shared_down_quant, GGML_Q4_K | GGML_Q6_K);
        let (shared_gate_f32_dev, shared_up_f32_dev, shared_down_f32_dev) = if expanded_q4k_path {
            (
                self.resident_q4k_f32_ptr(shared_gate, n_ff, n_embd / 256)?,
                self.resident_q4k_f32_ptr(shared_up, n_ff, n_embd / 256)?,
                match shared_down_quant {
                    12 => self.resident_q4k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
                    14 => self.resident_q6k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
                    other => {
                        return Err(format!(
                            "MTP verify Qwen35 shared expert unsupported down quant {other}"
                        ))
                    }
                },
            )
        } else {
            (None, None, None)
        };
        let shared_scale_dev = self.resident_f32_ptr(shared_input_scale)?;
        let shared_route_dev =
            self.compute_route_ptr(plan.window_tokens * std::mem::size_of::<f32>())?;
        let shared_gate_out_dev =
            self.compute_mid_a_ptr(plan.window_tokens * n_ff * std::mem::size_of::<f32>())?;
        let shared_up_out_dev =
            self.compute_mid_b_ptr(plan.window_tokens * n_ff * std::mem::size_of::<f32>())?;
        let shared_output_dev =
            self.compute_output_ptr(plan.window_tokens * n_embd * std::mem::size_of::<f32>())?;

        self.launch_qwen35_shared_route_sigmoid_f32(
            shared_route_dev,
            buffers.scratch_hidden_dev,
            shared_scale_dev,
            plan.window_tokens,
            n_embd,
        )?;
        if let (Some(shared_gate_dev), Some(shared_up_dev), Some(shared_down_dev)) =
            (shared_gate_f32_dev, shared_up_f32_dev, shared_down_f32_dev)
        {
            self.sgemm_device(
                shared_gate_dev,
                n_ff,
                n_embd,
                buffers.scratch_hidden_dev,
                plan.window_tokens,
                shared_gate_out_dev,
            )?;
            self.sgemm_device(
                shared_up_dev,
                n_ff,
                n_embd,
                buffers.scratch_hidden_dev,
                plan.window_tokens,
                shared_up_out_dev,
            )?;
            self.launch_silu_mul(
                shared_gate_out_dev,
                shared_up_out_dev,
                plan.window_tokens * n_ff,
            )?;
            self.sgemm_device(
                shared_down_dev,
                n_embd,
                n_ff,
                shared_gate_out_dev,
                plan.window_tokens,
                shared_output_dev,
            )?;
        } else {
            if plan.window_tokens == 2
                && shared_gate_quant == GGML_Q8_0
                && shared_up_quant == GGML_Q8_0
                && shared_gate_blocks == shared_up_blocks
                && crate::tuning::mtp_verify_q8_multi_projection_enabled()
            {
                self.launch_q8_0_gemv_batch_token2_multi3_to_dev(
                    [shared_gate, shared_up, shared_gate],
                    [n_ff, n_ff, 0],
                    shared_gate_blocks,
                    buffers.scratch_hidden_dev,
                    [shared_gate_out_dev, shared_up_out_dev, shared_gate_out_dev],
                )?;
            } else {
                self.stage_mtp_verify_k_quant_projection_to_dev(
                    "Qwen35 shared gate",
                    shared_gate_quant,
                    shared_gate,
                    n_ff,
                    shared_gate_blocks,
                    plan.window_tokens,
                    buffers.scratch_hidden_dev,
                    shared_gate_out_dev,
                )?;
                self.stage_mtp_verify_k_quant_projection_to_dev(
                    "Qwen35 shared up",
                    shared_up_quant,
                    shared_up,
                    n_ff,
                    shared_up_blocks,
                    plan.window_tokens,
                    buffers.scratch_hidden_dev,
                    shared_up_out_dev,
                )?;
            }
            self.launch_silu_mul(
                shared_gate_out_dev,
                shared_up_out_dev,
                plan.window_tokens * n_ff,
            )?;
            self.stage_mtp_verify_k_quant_projection_to_dev(
                "Qwen35 shared down",
                shared_down_quant,
                shared_down,
                n_embd,
                shared_down_blocks,
                plan.window_tokens,
                shared_gate_out_dev,
                shared_output_dev,
            )?;
        }
        if crate::tuning::mtp_verify_shared_scale_add_enabled() {
            self.launch_scale_rows_add_f32_inplace(
                buffers.hidden_rows_dev,
                shared_output_dev,
                shared_route_dev,
                n_embd,
                plan.window_tokens,
            )
        } else {
            self.launch_scale_rows_inplace(
                shared_output_dev,
                shared_route_dev,
                n_embd,
                plan.window_tokens,
            )?;
            self.launch_add_f32_inplace(
                buffers.hidden_rows_dev,
                shared_output_dev,
                plan.window_tokens * n_embd,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_moe_residual_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_index: Option<usize>,
        router_w: &[f32],
        n_expert: usize,
        n_expert_used: usize,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        shared_input_scale: &[f32],
        shared_gate: &[u8],
        shared_gate_quant: u32,
        shared_up: &[u8],
        shared_up_quant: u32,
        shared_down: &[u8],
        shared_down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<MtpVerifyQwen35RouterBuffers, String> {
        let router_buffers =
            self.stage_mtp_verify_qwen35_router_topk(buffers, router_w, n_expert, n_expert_used)?;
        if crate::tuning::mtp_verify_resident_moe_layer_enabled()
            && self.ensure_qwen35_mtp_verify_resident_moe_layer(
                gate_all, up_all, down_all, down_quant, n_ff, n_embd,
            )?
        {
            self.stage_mtp_verify_gdn_sparse_moe_full_layer_from_router_q4k(
                buffers,
                &router_buffers,
                gate_all,
                up_all,
                down_all,
                down_quant,
                n_expert,
                n_ff,
                n_embd,
            )?;
        } else {
            self.stage_mtp_verify_gdn_sparse_moe_selected_from_router_q4k(
                buffers,
                &router_buffers,
                layer_index,
                gate_all,
                up_all,
                down_all,
                down_quant,
                n_expert,
                n_ff,
                n_embd,
            )?;
        }
        self.stage_mtp_verify_gdn_shared_expert_q4k(
            buffers,
            shared_input_scale,
            shared_gate,
            shared_gate_quant,
            shared_up,
            shared_up_quant,
            shared_down,
            shared_down_quant,
            n_ff,
            n_embd,
        )?;
        Ok(router_buffers)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_attention_moe_layer_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        rope_dim: usize,
        rope_neox: bool,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
    ) -> Result<MtpVerifyQwen35RouterBuffers, String> {
        let (router_buffers, _) = self.stage_mtp_verify_qwen35_attention_moe_layer_q4k_inner(
            buffers, layer, rope_dim, rope_neox, rope_theta, pos_start, norm_eps, false, true,
            true, false,
        )?;
        router_buffers
            .ok_or_else(|| "MTP verify attention dense layer has no router buffers".to_string())
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_attention_moe_layer_q4k_with_kv_state(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        rope_dim: usize,
        rope_neox: bool,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
    ) -> Result<Qwen35MtpDeviceVerifyAttentionKvState, String> {
        let (_, attention_kv) = self.stage_mtp_verify_qwen35_attention_moe_layer_q4k_inner(
            buffers, layer, rope_dim, rope_neox, rope_theta, pos_start, norm_eps, true, true, true,
            false,
        )?;
        attention_kv.ok_or_else(|| {
            "MTP verify attention K/V state was not collected after request".to_string()
        })
    }

    pub(in crate::runtime) fn stage_mtp_verify_qwen35_attention_moe_layer_q4k_with_host_kv_state(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        rope_dim: usize,
        rope_neox: bool,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
    ) -> Result<Qwen35MtpDeviceVerifyAttentionKvState, String> {
        let (_, attention_kv) = self.stage_mtp_verify_qwen35_attention_moe_layer_q4k_inner(
            buffers, layer, rope_dim, rope_neox, rope_theta, pos_start, norm_eps, true, true,
            false, false,
        )?;
        attention_kv.ok_or_else(|| {
            "MTP device draft attention K/V state was not collected after request".to_string()
        })
    }

    pub(in crate::runtime) fn stage_qwen35_prefill_attention_layer_q4k_with_kv_state(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        rope_dim: usize,
        rope_neox: bool,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        collect_host_kv_when_resident: bool,
    ) -> Result<Qwen35MtpDeviceVerifyAttentionKvState, String> {
        let (_, attention_kv) = self.stage_mtp_verify_qwen35_attention_moe_layer_q4k_inner(
            buffers,
            layer,
            rope_dim,
            rope_neox,
            rope_theta,
            pos_start,
            norm_eps,
            true,
            false,
            true,
            collect_host_kv_when_resident,
        )?;
        attention_kv.ok_or_else(|| {
            "Qwen prefill attention K/V state was not collected after request".to_string()
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_mtp_verify_attention_post_ffn_graph(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        attention_buffers: &MtpVerifyAttentionOutputBuffers,
        norm_eps: f32,
        run_ffn: bool,
        collect_kv_state: bool,
    ) -> Result<Option<MtpVerifyQwen35RouterBuffers>, String> {
        let graph_candidate = crate::tuning::mtp_verify_attention_graph_enabled()
            && (buffers.plan.window_tokens == 1
                || (buffers.plan.window_tokens == 2
                    && crate::tuning::mtp_verify_window2_graphs_enabled()))
            && run_ffn
            && collect_kv_state
            && layer.post_ffw_norm.is_empty()
            && layer.ple_gate.is_empty()
            && layer.out_scale.is_empty()
            && std::env::var("RNB_CUDA_MTP_VERIFY_SELECTED_GRAPH")
                .ok()
                .as_deref()
                != Some("1")
            && std::env::var("RNB_CUDA_MTP_VERIFY_SELECTED_GRAPH_Q8")
                .ok()
                .as_deref()
                != Some("1");
        let graph_enabled = graph_candidate
            && crate::tuning::mtp_verify_resident_moe_layer_enabled()
            && self.ensure_qwen35_mtp_verify_resident_moe_layer(
                layer.gate_all,
                layer.up_all,
                layer.down_all,
                layer.down_quant,
                layer.n_ff,
                layer.n_embd,
            )?;
        if !graph_enabled {
            return self.stage_mtp_verify_attention_post_ffn_uncaptured(
                buffers,
                layer,
                post_buffers,
                attention_buffers,
                norm_eps,
                run_ffn,
            );
        }
        let key = super::MtpVerifyAttentionGraphKey {
            layer_idx: layer.layer_index,
            model_weight_ptr: layer.o_q4k.as_ptr() as usize,
            hidden_dev: buffers.hidden_rows_dev,
            segment: 1,
            q8_selected_gate_up: crate::tuning::mtp_verify_selected_q8_gate_up_enabled(),
            pair2_selected_gate_up: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_gate_pair2_enabled(),
            pair2_selected_gate_up_silu: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_gate_pair2_enabled()
                && crate::tuning::mtp_verify_selected_gate_pair2_silu_enabled(),
            pair2_selected_down: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_down_pair2_enabled()
                && std::env::var("RNB_CUDA_WARP_DOWN").ok().as_deref() != Some("0"),
            pair2_selected_map: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_pair_map_enabled(),
        };
        if let Some(graph) = self.mtp_verify_attention_graphs.get(&key) {
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok(None);
        }
        if !self.mtp_verify_attention_graph_warmed.insert(key) {
            self.ensure_q4k_gemv_module()?;
            unsafe {
                self.api.stream_begin_capture(self.stream)?;
            }
            let result = self.stage_mtp_verify_attention_post_ffn_uncaptured(
                buffers,
                layer,
                post_buffers,
                attention_buffers,
                norm_eps,
                run_ffn,
            );
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    unsafe {
                        let _ = self.api.stream_end_capture(self.stream);
                    }
                    return Err(err);
                }
            };
            let graph = unsafe { self.api.stream_end_capture(self.stream)? };
            let exec = unsafe { self.api.graph_instantiate(graph)? };
            self.mtp_verify_attention_graphs.insert(
                key,
                super::SparseMoeGraph {
                    graph: graph as usize,
                    exec: exec as usize,
                },
            );
            let graph = self
                .mtp_verify_attention_graphs
                .get(&key)
                .ok_or_else(|| "missing MTP verify attention post CUDA graph".to_string())?;
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok(result);
        }
        self.stage_mtp_verify_attention_post_ffn_uncaptured(
            buffers,
            layer,
            post_buffers,
            attention_buffers,
            norm_eps,
            run_ffn,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_mtp_verify_attention_post_ffn_uncaptured(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        post_buffers: &MtpVerifyAttentionQkNormRopeBuffers,
        attention_buffers: &MtpVerifyAttentionOutputBuffers,
        norm_eps: f32,
        run_ffn: bool,
    ) -> Result<Option<MtpVerifyQwen35RouterBuffers>, String> {
        let subtrace = Self::mtp_verify_subtrace_enabled();
        if let Some(gate_dev) = post_buffers.gate_dev {
            let stage_start = std::time::Instant::now();
            let gated_values = buffers
                .plan
                .window_tokens
                .checked_mul(post_buffers.q_rows)
                .ok_or_else(|| {
                    format!(
                        "MTP verify attention gated output value overflow: tokens={} rows={}",
                        buffers.plan.window_tokens, post_buffers.q_rows
                    )
                })?;
            self.launch_sigmoid_mul_inplace(
                attention_buffers.attn_out_dev,
                gate_dev,
                gated_values,
            )?;
            self.trace_mtp_verify_subphase(
                subtrace,
                "attention",
                Some(layer.layer_index),
                "gate_attention",
                stage_start,
            )?;
        }
        let stage_start = std::time::Instant::now();
        self.stage_mtp_verify_attention_o_projection_residual_ffn_norm_q4k(
            buffers,
            attention_buffers,
            layer.o_q4k,
            layer.o_quant,
            layer.o_rows,
            layer.o_cols,
            layer.post_attn_norm,
            layer.post_attn_norm_unit_offset,
            layer.ffn_norm,
            layer.ffn_norm_unit_offset,
            norm_eps,
        )?;
        self.trace_mtp_verify_subphase(
            subtrace,
            "attention",
            Some(layer.layer_index),
            "o_proj_residual_norm",
            stage_start,
        )?;
        if !run_ffn {
            return Ok(None);
        }
        let stage_start = std::time::Instant::now();
        let router_buffers = if layer.n_expert == 0 {
            self.stage_mtp_verify_attention_dense_ffn_residual_q4k(
                buffers,
                layer.ffn_gate_q4k,
                layer.ffn_gate_rows,
                layer.ffn_gate_cols,
                layer.ffn_up_q4k,
                layer.ffn_up_rows,
                layer.ffn_up_cols,
                layer.ffn_down,
                layer.ffn_down_quant,
                layer.ffn_down_rows,
                layer.ffn_down_cols,
                layer.ffn_uses_gelu,
                layer.post_ffw_norm,
                layer.post_ffw_norm_unit_offset,
                norm_eps,
            )?;
            None
        } else {
            Some(self.stage_mtp_verify_qwen35_moe_residual_q4k(
                buffers,
                Some(layer.layer_index),
                layer.router_w,
                layer.n_expert,
                layer.n_expert_used,
                layer.gate_all,
                layer.up_all,
                layer.down_all,
                layer.down_quant,
                layer.shared_input_scale,
                layer.shared_gate,
                layer.shared_gate_quant,
                layer.shared_up,
                layer.shared_up_quant,
                layer.shared_down,
                layer.shared_down_quant,
                layer.n_ff,
                layer.n_embd,
            )?)
        };
        if !layer.ple_gate.is_empty() {
            self.stage_mtp_verify_gemma4_ple(
                buffers,
                layer.ple_gate,
                layer.ple_gate_quant,
                layer.ple_gate_rows,
                layer.ple_gate_cols,
                layer.ple_proj,
                layer.ple_proj_quant,
                layer.ple_proj_rows,
                layer.ple_proj_cols,
                layer.ple_post_norm,
                layer.ple_post_norm_unit_offset,
                layer.ple_input,
                layer.out_scale,
                norm_eps,
            )?;
        } else if let Some(scale) = layer.out_scale.first().copied() {
            self.launch_scale_f32_inplace(
                buffers.hidden_rows_dev,
                scale,
                buffers.plan.window_tokens * buffers.plan.hidden_dim,
            )?;
        }
        self.trace_mtp_verify_subphase(
            subtrace,
            "attention",
            Some(layer.layer_index),
            "ffn_residual",
            stage_start,
        )?;
        Ok(router_buffers)
    }

    fn stage_mtp_verify_qwen35_attention_moe_layer_q4k_inner(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer: &Qwen35MtpDeviceVerifyAttentionMoeLayer<'_>,
        rope_dim: usize,
        rope_neox: bool,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        collect_kv_state: bool,
        run_ffn: bool,
        allow_resident_kv: bool,
        collect_host_kv_when_resident: bool,
    ) -> Result<
        (
            Option<MtpVerifyQwen35RouterBuffers>,
            Option<Qwen35MtpDeviceVerifyAttentionKvState>,
        ),
        String,
    > {
        if run_ffn
            && layer.n_expert != 0
            && (!matches!(layer.expert_gating_func, 0 | 1) || !layer.shared_expert_gated)
        {
            return Err(format!(
                "MTP verify attention MoE requires softmax top-k routing and gated shared expert: expert_gating_func={} shared_expert_gated={}",
                layer.expert_gating_func, layer.shared_expert_gated
            ));
        }
        let subtrace = Self::mtp_verify_subtrace_enabled();
        let stage_start = std::time::Instant::now();
        let projection_buffers = self.stage_mtp_verify_attention_qkv_projections_q4k_graph(
            buffers,
            layer.layer_index,
            Qwen35MtpAttentionQkvProjectionRequest {
                attn_norm: layer.attn_norm,
                q_q4k: layer.q_q4k,
                q_quant: layer.q_quant,
                q_rows: layer.q_rows,
                q_cols: layer.q_cols,
                k_q4k: layer.k_q4k,
                k_quant: layer.k_quant,
                k_rows: layer.k_rows,
                k_cols: layer.k_cols,
                v_q4k: layer.v_q4k,
                v_quant: layer.v_quant,
                v_rows: layer.v_rows,
                v_cols: layer.v_cols,
                norm_eps,
            },
        )?;
        self.trace_mtp_verify_subphase(
            subtrace,
            "attention",
            Some(layer.layer_index),
            "qkv_projections",
            stage_start,
        )?;
        let head_dim = if layer.q_norm.is_empty() {
            layer.k_norm.len()
        } else {
            layer.q_norm.len()
        };
        let q_attention_rows = if layer.q_rows == layer.o_cols.saturating_mul(2) {
            layer.o_cols
        } else {
            layer.q_rows
        };
        let num_heads = if head_dim == 0 {
            0
        } else {
            q_attention_rows / head_dim
        };
        let num_kv_heads = if head_dim == 0 {
            0
        } else {
            layer.k_rows / head_dim
        };
        let stage_start = std::time::Instant::now();
        let mut post_buffers = self.stage_mtp_verify_attention_qk_norm_rope(
            &projection_buffers,
            Qwen35MtpAttentionQkNormRopeRequest {
                q_norm: layer.q_norm,
                k_norm: layer.k_norm,
                num_heads,
                num_kv_heads,
                rope_freq_factors: layer.rope_freq_factors,
                head_dim,
                rope_dim,
                rope_neox,
                rope_theta,
                pos_start,
                norm_eps,
                q_unit_offset: layer.qk_norm_unit_offset,
                k_unit_offset: layer.qk_norm_unit_offset,
                v_no_scale_norm: layer.v_no_scale_norm,
            },
        )?;
        if let Some(source_layer) = layer.kv_source_layer {
            let (source_k_bits_dev, source_v_bits_dev) = self
                .mtp_verify_attention_shared_window_kv(
                    source_layer,
                    layer.prior_sequence_epoch,
                    pos_start,
                    post_buffers.window_tokens,
                    post_buffers.kv_rows,
                )?;
            post_buffers.k_bits_dev = source_k_bits_dev;
            post_buffers.v_bits_dev = source_v_bits_dev;
        }
        self.trace_mtp_verify_subphase(
            subtrace,
            "attention",
            Some(layer.layer_index),
            "qk_norm_rope",
            stage_start,
        )?;
        let scale = layer.attention_scale;
        let stage_start = std::time::Instant::now();
        let kv_cache_layer = layer.kv_source_layer.unwrap_or(layer.layer_index);
        let attention_buffers = if let Some(kvarn_prior) = layer.kvarn_prior {
            if kvarn_prior.kv_len() != layer.prior_tokens {
                return Err(format!(
                    "MTP verify KVarN prior token mismatch: cache={} layer={}",
                    kvarn_prior.kv_len(),
                    layer.prior_tokens
                ));
            }
            self.stage_mtp_verify_attention_output_kvarn_window(&post_buffers, kvarn_prior)?
        } else if let Some((prior_k_bits_dev, prior_v_bits_dev)) = self
            .stage_mtp_verify_attention_prior_kv_bits_for_layer(
                kv_cache_layer,
                layer.prior_sequence_epoch,
                layer.prior_k_bits,
                layer.prior_v_bits,
                layer.prior_tokens,
                post_buffers.kv_rows,
            )?
        {
            let attention_window = layer
                .prior_tokens
                .checked_add(buffers.plan.window_tokens)
                .ok_or_else(|| {
                    format!(
                        "MTP verify attention prior window overflow: prior={} window={}",
                        layer.prior_tokens, buffers.plan.window_tokens
                    )
                })?;
            self.stage_mtp_verify_attention_output_prior_window(
                &post_buffers,
                Qwen35MtpAttentionOutputWithPriorRequest {
                    prior_k_bits_dev,
                    prior_v_bits_dev,
                    prior_tokens: layer.prior_tokens,
                    num_heads,
                    num_kv_heads,
                    scale,
                    window: attention_window,
                },
            )?
        } else {
            self.stage_mtp_verify_attention_output_window(
                &post_buffers,
                Qwen35MtpAttentionOutputRequest {
                    num_heads,
                    num_kv_heads,
                    scale,
                    window: buffers.plan.window_tokens,
                },
            )?
        };
        self.trace_mtp_verify_subphase(
            subtrace,
            "attention",
            Some(layer.layer_index),
            "attention_output",
            stage_start,
        )?;
        if layer.kv_source_layer.is_none() {
            self.retain_mtp_verify_attention_shared_window_kv(
                layer.layer_index,
                layer.prior_sequence_epoch,
                pos_start,
                &post_buffers,
            )?;
        }
        let router_buffers = self.stage_mtp_verify_attention_post_ffn_graph(
            buffers,
            layer,
            &post_buffers,
            &attention_buffers,
            norm_eps,
            run_ffn,
            collect_kv_state,
        )?;
        let attention_kv = if collect_kv_state {
            let stage_start = std::time::Instant::now();
            let shared_kv_reuse = layer.kv_source_layer.is_some();
            let keep_resident = shared_kv_reuse
                || (!collect_host_kv_when_resident
                    && layer.kvarn_prior.is_none()
                    && allow_resident_kv
                    && crate::tuning::mtp_verify_resident_attn_kv_enabled());
            let (k_bits, v_bits) = if shared_kv_reuse {
                (Vec::new(), Vec::new())
            } else if keep_resident {
                self.retain_mtp_verify_attention_window_kv_for_layer(
                    layer.layer_index,
                    layer.prior_sequence_epoch,
                    layer.prior_tokens,
                    &post_buffers,
                )?;
                if collect_host_kv_when_resident {
                    self.collect_mtp_verify_attention_window_kv_bits_deferred(&post_buffers)?
                } else {
                    (Vec::new(), Vec::new())
                }
            } else {
                self.collect_mtp_verify_attention_window_kv_bits_deferred(&post_buffers)?
            };
            let state = Some(Qwen35MtpDeviceVerifyAttentionKvState {
                layer_idx: layer.layer_index,
                window_tokens: post_buffers.window_tokens,
                kv_rows: post_buffers.kv_rows,
                k_bits,
                v_bits,
                device_resident: keep_resident && !collect_host_kv_when_resident,
            });
            self.trace_mtp_verify_subphase(
                subtrace,
                "attention",
                Some(layer.layer_index),
                "kv_capture",
                stage_start,
            )?;
            state
        } else {
            None
        };
        Ok((router_buffers, attention_kv))
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_gdn_moe_layer_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        request: Qwen35MtpGdnMoeLayerRequest<'_>,
    ) -> Result<MtpVerifyQwen35RouterBuffers, String> {
        let (router_buffers, _, _) =
            self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_inner(buffers, None, request, &[])?;
        router_buffers.ok_or_else(|| "MTP verify GDN dense layer has no router buffers".to_string())
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_qwen35_gdn_moe_layer_q4k_with_prefix_states(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_index: usize,
        request: Qwen35MtpGdnMoeLayerRequest<'_>,
        prefix_tokens: &[usize],
    ) -> Result<Vec<Qwen35MtpDeviceVerifyPrefixState>, String> {
        let (_, prefix_states, _) = self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_inner(
            buffers,
            Some(layer_index),
            request,
            prefix_tokens,
        )?;
        Ok(prefix_states)
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn begin_mtp_verify_segment_graph(
        &mut self,
        key: super::MtpVerifySegmentGraphKey,
    ) -> Result<super::MtpVerifySegmentGraphStep, String> {
        if !crate::tuning::mtp_verify_segment_graph_enabled() {
            return Ok(super::MtpVerifySegmentGraphStep::Disabled);
        }
        if let Some(graph) = self.mtp_verify_segment_graphs.get(&key) {
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok(super::MtpVerifySegmentGraphStep::Replay);
        }
        if self.mtp_verify_segment_graph_warmed.insert(key) {
            return Ok(super::MtpVerifySegmentGraphStep::Warm);
        }
        self.ensure_q4k_gemv_module()?;
        unsafe {
            self.api.stream_begin_capture(self.stream)?;
        }
        self.mtp_verify_segment_capture_active = true;
        Ok(super::MtpVerifySegmentGraphStep::Capture)
    }

    pub(in crate::runtime) fn finish_mtp_verify_segment_graph(
        &mut self,
        key: super::MtpVerifySegmentGraphKey,
    ) -> Result<(), String> {
        self.mtp_verify_segment_capture_active = false;
        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
        let exec = unsafe { self.api.graph_instantiate(graph)? };
        self.mtp_verify_segment_graphs.insert(
            key,
            super::SparseMoeGraph {
                graph: graph as usize,
                exec: exec as usize,
            },
        );
        let graph = self
            .mtp_verify_segment_graphs
            .get(&key)
            .ok_or_else(|| "missing MTP verify segment CUDA graph".to_string())?;
        unsafe {
            self.api
                .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn abort_mtp_verify_segment_graph(&mut self) {
        self.mtp_verify_segment_capture_active = false;
        unsafe {
            let _ = self.api.stream_end_capture(self.stream);
        }
    }

    pub(in crate::runtime) fn stage_mtp_verify_qwen35_gdn_moe_layer_q4k_capture_states(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_index: usize,
        request: Qwen35MtpGdnMoeLayerRequest<'_>,
        prefix_tokens: &[usize],
    ) -> Result<Qwen35MtpGdnMoeLayerStateCapture, String> {
        let (_, prefix_states, final_state) = self
            .stage_mtp_verify_qwen35_gdn_moe_layer_q4k_inner(
                buffers,
                Some(layer_index),
                request,
                prefix_tokens,
            )?;
        let final_state = final_state.ok_or_else(|| {
            "MTP verify GDN layer final state was not captured after request".to_string()
        })?;
        Ok(Qwen35MtpGdnMoeLayerStateCapture {
            prefix_states,
            final_state,
        })
    }

    fn stage_mtp_verify_qwen35_gdn_moe_layer_q4k_inner(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_index: Option<usize>,
        request: Qwen35MtpGdnMoeLayerRequest<'_>,
        prefix_tokens: &[usize],
    ) -> Result<
        (
            Option<MtpVerifyQwen35RouterBuffers>,
            Vec<Qwen35MtpDeviceVerifyPrefixState>,
            Option<Qwen35MtpDeviceVerifySsmLayerFinalState>,
        ),
        String,
    > {
        if self.mtp_verify_segment_capture_active {
            return self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
                buffers,
                layer_index,
                request,
                prefix_tokens,
            );
        }
        let graph_enabled = crate::tuning::mtp_verify_gdn_graph_enabled()
            && (buffers.plan.window_tokens == 1
                || (buffers.plan.window_tokens == 2
                    && crate::tuning::mtp_verify_window2_graphs_enabled()));
        let Some(layer_idx) = layer_index else {
            return self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
                buffers,
                layer_index,
                request,
                prefix_tokens,
            );
        };
        if !graph_enabled
            || !prefix_tokens.is_empty()
            || request.sync_delta_state_to_host
            || !crate::tuning::mtp_verify_resident_conv_enabled()
        {
            return self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
                buffers,
                layer_index,
                request,
                prefix_tokens,
            );
        }
        if !crate::tuning::mtp_verify_resident_moe_layer_enabled()
            || !self.ensure_qwen35_mtp_verify_resident_moe_layer(
                request.gate_all,
                request.up_all,
                request.down_all,
                request.down_quant,
                request.n_ff,
                request.n_embd,
            )?
        {
            return self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
                buffers,
                layer_index,
                request,
                prefix_tokens,
            );
        }
        let key = super::MtpVerifyGdnGraphKey {
            layer_idx,
            model_weight_ptr: request.projection.qkv_q4k.as_ptr() as usize,
            hidden_dev: buffers.hidden_rows_dev,
            conv_state_ptr: request.conv_state.as_ptr() as usize,
            delta_state_ptr: request.delta_state.as_ptr() as usize,
            q8_selected_gate_up: crate::tuning::mtp_verify_selected_q8_gate_up_enabled(),
            pair2_selected_gate_up: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_gate_pair2_enabled(),
            pair2_selected_gate_up_silu: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_gate_pair2_enabled()
                && crate::tuning::mtp_verify_selected_gate_pair2_silu_enabled(),
            pair2_selected_down: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_down_pair2_enabled()
                && std::env::var("RNB_CUDA_WARP_DOWN").ok().as_deref() != Some("0"),
            pair2_selected_map: buffers.plan.window_tokens == 2
                && crate::tuning::mtp_verify_selected_pair_map_enabled(),
        };
        if let Some(graph) = self.mtp_verify_gdn_graphs.get(&key) {
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok((
                None,
                Vec::new(),
                Some(Qwen35MtpDeviceVerifySsmLayerFinalState {
                    layer_idx,
                    conv_state: Vec::new(),
                    device_resident: true,
                }),
            ));
        }
        if !self.mtp_verify_gdn_graph_warmed.insert(key) {
            self.ensure_q4k_gemv_module()?;
            unsafe {
                self.api.stream_begin_capture(self.stream)?;
            }
            let result = self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
                buffers,
                layer_index,
                request,
                prefix_tokens,
            );
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    unsafe {
                        let _ = self.api.stream_end_capture(self.stream);
                    }
                    return Err(err);
                }
            };
            let graph = unsafe { self.api.stream_end_capture(self.stream)? };
            let exec = unsafe { self.api.graph_instantiate(graph)? };
            self.mtp_verify_gdn_graphs.insert(
                key,
                super::SparseMoeGraph {
                    graph: graph as usize,
                    exec: exec as usize,
                },
            );
            let graph = self
                .mtp_verify_gdn_graphs
                .get(&key)
                .ok_or_else(|| "missing MTP verify GDN CUDA graph".to_string())?;
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
            }
            return Ok(result);
        }
        self.stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
            buffers,
            layer_index,
            request,
            prefix_tokens,
        )
    }

    fn stage_mtp_verify_qwen35_gdn_moe_layer_q4k_uncaptured(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        layer_index: Option<usize>,
        request: Qwen35MtpGdnMoeLayerRequest<'_>,
        prefix_tokens: &[usize],
    ) -> Result<
        (
            Option<MtpVerifyQwen35RouterBuffers>,
            Vec<Qwen35MtpDeviceVerifyPrefixState>,
            Option<Qwen35MtpDeviceVerifySsmLayerFinalState>,
        ),
        String,
    > {
        let subtrace = Self::mtp_verify_subtrace_enabled();
        let stage_start = std::time::Instant::now();
        let projection_buffers =
            self.stage_mtp_verify_gdn_input_projections_q4k(buffers, request.projection)?;
        self.trace_mtp_verify_subphase(
            subtrace,
            "gdn",
            layer_index,
            "input_projections",
            stage_start,
        )?;
        let stage_start = std::time::Instant::now();
        let conv_buffers = self.stage_mtp_verify_gdn_conv1d_silu(
            &projection_buffers,
            request.conv_state,
            request.conv_kernel,
            request.kernel_size,
            true,
        )?;
        self.trace_mtp_verify_subphase(subtrace, "gdn", layer_index, "conv1d_silu", stage_start)?;
        let stage_start = std::time::Instant::now();
        let delta_buffers = self.stage_mtp_verify_gdn_delta_inputs(
            &conv_buffers,
            &projection_buffers,
            request.dt_bias,
            request.ssm_a,
            request.num_k_heads,
            request.num_v_heads,
            request.head_k_dim,
            request.head_v_dim,
            request.norm_eps,
        )?;
        self.trace_mtp_verify_subphase(subtrace, "gdn", layer_index, "delta_inputs", stage_start)?;
        let stage_start = std::time::Instant::now();
        let (scan_buffers, prefix_snapshots) = if prefix_tokens.is_empty() {
            let scan_buffers = self.stage_mtp_verify_gdn_delta_scan(
                &delta_buffers,
                request.delta_state,
                request.sync_delta_state_to_host,
            )?;
            (scan_buffers, Vec::new())
        } else {
            layer_index.ok_or_else(|| {
                "MTP verify GDN prefix state capture requires a layer index".to_string()
            })?;
            let (scan_buffers, snapshots) = self.stage_mtp_verify_gdn_delta_scan_snapshots(
                &delta_buffers,
                request.delta_state,
                request.sync_delta_state_to_host,
                prefix_tokens,
            )?;
            (scan_buffers, snapshots)
        };
        self.trace_mtp_verify_subphase(subtrace, "gdn", layer_index, "delta_scan", stage_start)?;

        let result = (|| {
            let stage_start = std::time::Instant::now();
            let ssm_buffers = self.stage_mtp_verify_gdn_ssm_out_q4k(
                &scan_buffers,
                &projection_buffers,
                request.ssm_norm,
                request.ssm_out_q4k,
                request.ssm_out_quant,
                request.ssm_out_rows,
                request.ssm_out_cols,
                request.norm_eps,
            )?;
            self.trace_mtp_verify_subphase(subtrace, "gdn", layer_index, "ssm_out", stage_start)?;
            let stage_start = std::time::Instant::now();
            self.stage_mtp_verify_gdn_residual_post_norm(
                buffers,
                &ssm_buffers,
                request.post_attn_norm,
                request.norm_eps,
            )?;
            self.trace_mtp_verify_subphase(
                subtrace,
                "gdn",
                layer_index,
                "residual_post_norm",
                stage_start,
            )?;
            let stage_start = std::time::Instant::now();
            if request.n_expert == 0 {
                self.stage_mtp_verify_attention_dense_ffn_residual_q4k(
                    buffers,
                    request.ffn_gate_q4k,
                    request.ffn_gate_rows,
                    request.ffn_gate_cols,
                    request.ffn_up_q4k,
                    request.ffn_up_rows,
                    request.ffn_up_cols,
                    request.ffn_down,
                    request.ffn_down_quant,
                    request.ffn_down_rows,
                    request.ffn_down_cols,
                    false,
                    &[],
                    false,
                    request.norm_eps,
                )?;
                self.trace_mtp_verify_subphase(
                    subtrace,
                    "gdn",
                    layer_index,
                    "ffn_residual",
                    stage_start,
                )?;
                Ok(None)
            } else {
                let router_buffers = self.stage_mtp_verify_qwen35_moe_residual_q4k(
                    buffers,
                    layer_index,
                    request.router_w,
                    request.n_expert,
                    request.n_expert_used,
                    request.gate_all,
                    request.up_all,
                    request.down_all,
                    request.down_quant,
                    request.shared_input_scale,
                    request.shared_gate,
                    request.shared_gate_quant,
                    request.shared_up,
                    request.shared_up_quant,
                    request.shared_down,
                    request.shared_down_quant,
                    request.n_ff,
                    request.n_embd,
                )?;
                self.trace_mtp_verify_subphase(
                    subtrace,
                    "gdn",
                    layer_index,
                    "moe_residual",
                    stage_start,
                )?;
                Ok(Some(router_buffers))
            }
        })();
        match result {
            Ok(router_buffers) => {
                let stage_start = std::time::Instant::now();
                let final_state = layer_index
                    .map(|layer_idx| {
                        self.stage_mtp_verify_gdn_conv_final_state_deferred(&conv_buffers)
                            .map(|conv_state| Qwen35MtpDeviceVerifySsmLayerFinalState {
                                layer_idx,
                                conv_state,
                                device_resident: conv_buffers.device_resident_state,
                            })
                    })
                    .transpose();
                let final_state = match final_state {
                    Ok(final_state) => final_state,
                    Err(err) => {
                        let _ = self.stream_synchronize();
                        for snapshot in prefix_snapshots {
                            self.free_delta_state_snapshot(snapshot)?;
                        }
                        return Err(err);
                    }
                };
                self.trace_mtp_verify_subphase(
                    subtrace,
                    "gdn",
                    layer_index,
                    "final_state_capture",
                    stage_start,
                )?;
                let layer_index = layer_index.unwrap_or(0);
                let stage_start = std::time::Instant::now();
                let mut prefix_conv_states = Vec::with_capacity(prefix_tokens.len());
                for &prefix in prefix_tokens {
                    let captured = if conv_buffers.device_resident_state {
                        self.stage_mtp_verify_gdn_conv_prefix_snapshot(&conv_buffers, prefix)
                            .map(|snapshot| (Vec::new(), Some(snapshot)))
                    } else {
                        self.stage_mtp_verify_gdn_conv_prefix_state_deferred(&conv_buffers, prefix)
                            .map(|conv_state| (conv_state, None))
                    };
                    match captured {
                        Ok((conv_state, resident_conv_snapshot)) => {
                            prefix_conv_states.push((prefix, conv_state, resident_conv_snapshot));
                        }
                        Err(err) => {
                            let _ = self.stream_synchronize();
                            for (_, _, snapshot) in prefix_conv_states {
                                if let Some(snapshot) = snapshot {
                                    self.free_delta_state_snapshot(snapshot)?;
                                }
                            }
                            for snapshot in prefix_snapshots {
                                self.free_delta_state_snapshot(snapshot)?;
                            }
                            return Err(err);
                        }
                    }
                }
                self.trace_mtp_verify_subphase(
                    subtrace,
                    "gdn",
                    Some(layer_index),
                    "prefix_state_capture",
                    stage_start,
                )?;
                let prefix_states = prefix_conv_states
                    .into_iter()
                    .zip(prefix_snapshots)
                    .map(
                        |(
                            (prefix_tokens, conv_state, resident_conv_snapshot),
                            resident_delta_snapshot,
                        )| {
                            Qwen35MtpDeviceVerifyPrefixState {
                                prefix_tokens,
                                layers: vec![Qwen35MtpDeviceVerifySsmLayerPrefixState {
                                    layer_idx: layer_index,
                                    conv_state,
                                    resident_conv_snapshot,
                                    resident_delta_snapshot: Some(resident_delta_snapshot),
                                }],
                            }
                        },
                    )
                    .collect::<Vec<_>>();
                Ok((router_buffers, prefix_states, final_state))
            }
            Err(err) => {
                for snapshot in prefix_snapshots {
                    if let Err(free_err) = self.free_delta_state_snapshot(snapshot) {
                        return Err(format!(
                            "{err}; failed to free prefix snapshots: {free_err}"
                        ));
                    }
                }
                Err(err)
            }
        }
    }

    pub(in crate::runtime) fn free_mtp_verify_prefix_state_snapshots(
        &mut self,
        prefix_states: Vec<Qwen35MtpDeviceVerifyPrefixState>,
    ) -> Result<(), String> {
        for prefix_state in prefix_states {
            for layer in prefix_state.layers {
                if let Some(snapshot) = layer.resident_conv_snapshot {
                    self.free_delta_state_snapshot(snapshot)?;
                }
                if let Some(snapshot) = layer.resident_delta_snapshot {
                    self.free_delta_state_snapshot(snapshot)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(in crate::runtime) fn stage_mtp_verify_gdn_sparse_moe_by_token_q4k(
        &mut self,
        buffers: &MtpVerifyDeviceBuffers,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let plan = &buffers.plan;
        if n_embd != plan.hidden_dim {
            return Err(format!(
                "MTP verify GDN MoE n_embd mismatch: n_embd={n_embd}, hidden_dim={}",
                plan.hidden_dim
            ));
        }
        if gate_weights.is_empty() {
            return Err("MTP verify GDN MoE requires at least one selected slot".to_string());
        }
        if gate_weights.len() != up_weights.len()
            || gate_weights.len() != down_weights.len()
            || gate_weights.len() != route_weights.len()
            || gate_weights.len() != token_ids.len()
        {
            return Err(format!(
                "MTP verify GDN MoE slot mismatch: gate={} up={} down={} route={} token_ids={}",
                gate_weights.len(),
                up_weights.len(),
                down_weights.len(),
                route_weights.len(),
                token_ids.len()
            ));
        }
        for &token_id in token_ids {
            let idx = usize::try_from(token_id)
                .map_err(|_| format!("MTP verify GDN MoE token id exceeds usize: {token_id}"))?;
            if idx >= plan.window_tokens {
                return Err(format!(
                    "MTP verify GDN MoE token id out of window: token={idx}, window_tokens={}",
                    plan.window_tokens
                ));
            }
        }
        if n_embd == 0 || n_embd % 256 != 0 {
            return Err(format!(
                "MTP verify GDN MoE n_embd must be non-zero and divisible by 256, got {n_embd}"
            ));
        }
        if n_ff == 0 || n_ff % 256 != 0 {
            return Err(format!(
                "MTP verify GDN MoE n_ff must be non-zero and divisible by 256, got {n_ff}"
            ));
        }
        self.qwen35_sparse_experts_by_token_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            plan.window_tokens,
            down_quant,
            n_ff,
            n_embd,
            buffers.scratch_hidden_dev,
            buffers.hidden_rows_dev,
            false,
            crate::tuning::mtp_verify_group2_down_warp4_enabled(),
        )
    }

    pub(in crate::runtime) fn write_q6k_argmax_tokens_batched_from_dev_input(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_base_dev: u64,
        input_stride_values: usize,
        token_count: usize,
        target_tokens_dev: u64,
    ) -> Result<(), String> {
        if rows == 0 {
            return Err("MTP verify batched Q6_K argmax requires non-empty rows".to_string());
        }
        if token_count == 0 {
            return Err("MTP verify batched Q6_K argmax requires at least one token".to_string());
        }
        if input_stride_values == 0 {
            return Err(
                "MTP verify batched Q6_K argmax requires non-zero input stride".to_string(),
            );
        }
        let weights_dev = self.resident_q4k_weights_ptr_pinned(weights)?;
        let q8dot = std::env::var("RNB_CUDA_MTP_VERIFY_OUTPUT_Q8DOT")
            .ok()
            .as_deref()
            == Some("1")
            && input_stride_values % 32 == 0;
        let block_count = rows.div_ceil(8);
        let total_blocks = token_count
            .checked_mul(block_count)
            .ok_or_else(|| "MTP verify batched Q6_K argmax block count overflow".to_string())?;
        let values_dev = self.compute_mid_a_ptr(total_blocks * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(total_blocks * std::mem::size_of::<u32>())?;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_base_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut rows_arg = u32::try_from(rows)
            .map_err(|_| format!("MTP verify output rows exceeds CUDA u32 limit: {rows}"))?;
        let mut blocks_per_row_arg = u32::try_from(blocks_per_row).map_err(|_| {
            format!("MTP verify output blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let mut input_stride_arg = u32::try_from(input_stride_values).map_err(|_| {
            format!("MTP verify output input stride exceeds CUDA u32 limit: {input_stride_values}")
        })?;
        let mut token_count_arg = u32::try_from(token_count).map_err(|_| {
            format!("MTP verify output token count exceeds CUDA u32 limit: {token_count}")
        })?;
        let mut block_count_arg = u32::try_from(block_count).map_err(|_| {
            format!("MTP verify output block count exceeds CUDA u32 limit: {block_count}")
        })?;
        if q8dot {
            let input_values = token_count
                .checked_mul(input_stride_values)
                .ok_or_else(|| "MTP verify output Q8 input length overflow".to_string())?;
            let mut input_qs_arg = self.compute_full_gate_ptr(input_values)?;
            let mut input_ds_arg =
                self.compute_full_up_ptr((input_values / 32) * std::mem::size_of::<f32>())?;
            self.launch_quantize_q8_1_by_32(
                input_base_dev,
                input_qs_arg,
                input_ds_arg,
                input_values,
            )?;
            let mut input_ds_stride_arg = u32::try_from(input_stride_values / 32)
                .map_err(|_| "MTP verify output Q8 scale stride exceeds CUDA u32".to_string())?;
            self.launch_cached_gemv(
                "rnb_q6k_gemv_argmax_q8dot_warp8_batched",
                &[
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut input_stride_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut input_ds_stride_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut block_count_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (block_count as u32, token_count as u32, 1),
                (256, 1, 1),
            )?;
        } else if crate::tuning::mtp_verify_output_q6k_token2_enabled(token_count) {
            self.launch_cached_gemv(
                "rnb_q6k_gemv_argmax_warp8_token2",
                &[
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut input_stride_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut block_count_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (block_count as u32, 1, 1),
                (256, 1, 1),
            )?;
        } else {
            self.launch_cached_gemv(
                "rnb_q6k_gemv_argmax_warp8_batched",
                &[
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut input_stride_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut block_count_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (block_count as u32, token_count as u32, 1),
                (256, 1, 1),
            )?;
        }

        let (_, final_indices_dev) = self.reduce_argmax_pairs_to_single_batched(
            values_dev,
            indices_dev,
            block_count,
            token_count,
        )?;
        unsafe {
            self.api.memcpy_dtod_async(
                target_tokens_dev,
                final_indices_dev,
                token_count * std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        Ok(())
    }

    fn write_q4k_argmax_token_from_dev_input(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        target_token_dev: u64,
    ) -> Result<(), String> {
        let output_dev = self.compute_output_ptr(rows * std::mem::size_of::<f32>())?;
        self.launch_q4k_gemv_to_dev(weights, rows, blocks_per_row, input_dev, output_dev)?;

        let block_count = 256usize.min(rows.max(1).div_ceil(256));
        let values_dev = self.compute_mid_a_ptr(block_count * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(block_count * std::mem::size_of::<u32>())?;
        let mut values_arg = output_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut len_arg = u32::try_from(rows)
            .map_err(|_| format!("MTP verify output rows exceeds CUDA u32 limit: {rows}"))?;
        self.launch_cached_gemv(
            "rnb_argmax_f32",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (block_count as u32, 1, 1),
            (256, 1, 1),
        )?;

        let (_, final_indices_dev) =
            self.reduce_argmax_pairs_to_single(values_dev, indices_dev, block_count)?;
        unsafe {
            self.api.memcpy_dtod_async(
                target_token_dev,
                final_indices_dev,
                std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        Ok(())
    }

    fn write_q8_0_argmax_token_from_dev_input(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        target_token_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr_pinned(weights)?;
        let block_count = rows.div_ceil(8);
        let values_dev = self.compute_mid_a_ptr(block_count * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(block_count * std::mem::size_of::<u32>())?;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut rows_arg = u32::try_from(rows)
            .map_err(|_| format!("MTP verify output rows exceeds CUDA u32 limit: {rows}"))?;
        let mut blocks_per_row_arg = u32::try_from(blocks_per_row).map_err(|_| {
            format!("MTP verify output blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        self.launch_cached_gemv(
            "rnb_q8_0_gemv_argmax_warp8",
            &[
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (block_count as u32, 1, 1),
            (256, 1, 1),
        )?;

        let (_, final_indices_dev) =
            self.reduce_argmax_pairs_to_single(values_dev, indices_dev, block_count)?;
        unsafe {
            self.api.memcpy_dtod_async(
                target_token_dev,
                final_indices_dev,
                std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        Ok(())
    }

    fn reduce_argmax_pairs_to_single(
        &mut self,
        mut values_dev: u64,
        mut indices_dev: u64,
        mut len: usize,
    ) -> Result<(u64, u64), String> {
        if len == 0 {
            return Err("MTP verify argmax reduction requires non-empty input".to_string());
        }
        let mut use_full_buffers = true;
        while len > 1 {
            let out_len = 256usize.min(len.div_ceil(256));
            let (out_values_dev, out_indices_dev) = if use_full_buffers {
                (
                    self.compute_full_up_ptr(out_len * std::mem::size_of::<f32>())?,
                    self.compute_full_down_ptr(out_len * std::mem::size_of::<u32>())?,
                )
            } else {
                (
                    self.compute_mid_a_ptr(out_len * std::mem::size_of::<f32>())?,
                    self.compute_mid_b_ptr(out_len * std::mem::size_of::<u32>())?,
                )
            };
            let mut values_arg = values_dev;
            let mut indices_arg = indices_dev;
            let mut block_values_arg = out_values_dev;
            let mut block_indices_arg = out_indices_dev;
            let mut len_arg = u32::try_from(len)
                .map_err(|_| format!("MTP verify argmax len exceeds CUDA u32 limit: {len}"))?;
            self.launch_cached_gemv(
                "rnb_argmax_pairs_f32",
                &[
                    (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (out_len as u32, 1, 1),
                (256, 1, 1),
            )?;
            values_dev = out_values_dev;
            indices_dev = out_indices_dev;
            len = out_len;
            use_full_buffers = !use_full_buffers;
        }
        Ok((values_dev, indices_dev))
    }

    fn reduce_argmax_pairs_to_single_batched(
        &mut self,
        mut values_dev: u64,
        mut indices_dev: u64,
        mut len: usize,
        token_count: usize,
    ) -> Result<(u64, u64), String> {
        if len == 0 {
            return Err("MTP verify batched argmax reduction requires non-empty input".to_string());
        }
        if token_count == 0 {
            return Err(
                "MTP verify batched argmax reduction requires at least one token".to_string(),
            );
        }
        let mut input_stride = len;
        let mut use_full_buffers = true;
        while len > 1 {
            let out_len = 256usize.min(len.div_ceil(256));
            let total_out = token_count
                .checked_mul(out_len)
                .ok_or_else(|| "MTP verify batched argmax reduction size overflow".to_string())?;
            let (out_values_dev, out_indices_dev) = if use_full_buffers {
                (
                    self.compute_full_up_ptr(total_out * std::mem::size_of::<f32>())?,
                    self.compute_full_down_ptr(total_out * std::mem::size_of::<u32>())?,
                )
            } else {
                (
                    self.compute_mid_a_ptr(total_out * std::mem::size_of::<f32>())?,
                    self.compute_mid_b_ptr(total_out * std::mem::size_of::<u32>())?,
                )
            };
            let mut values_arg = values_dev;
            let mut indices_arg = indices_dev;
            let mut block_values_arg = out_values_dev;
            let mut block_indices_arg = out_indices_dev;
            let mut len_arg = u32::try_from(len).map_err(|_| {
                format!("MTP verify batched argmax len exceeds CUDA u32 limit: {len}")
            })?;
            let mut token_count_arg = u32::try_from(token_count).map_err(|_| {
                format!(
                    "MTP verify batched argmax token count exceeds CUDA u32 limit: {token_count}"
                )
            })?;
            let mut input_stride_arg = u32::try_from(input_stride).map_err(|_| {
                format!(
                    "MTP verify batched argmax input stride exceeds CUDA u32 limit: {input_stride}"
                )
            })?;
            let mut output_stride_arg = u32::try_from(out_len).map_err(|_| {
                format!("MTP verify batched argmax output stride exceeds CUDA u32 limit: {out_len}")
            })?;
            self.launch_cached_gemv(
                "rnb_argmax_pairs_f32_batched",
                &[
                    (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut input_stride_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut output_stride_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (out_len as u32, token_count as u32, 1),
                (256, 1, 1),
            )?;
            values_dev = out_values_dev;
            indices_dev = out_indices_dev;
            len = out_len;
            input_stride = out_len;
            use_full_buffers = !use_full_buffers;
        }
        Ok((values_dev, indices_dev))
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn collect_mtp_verify_result(
        &mut self,
        plan: &MtpVerifyBufferPlan,
    ) -> Result<Qwen35MtpDeviceVerifyResult, String> {
        let buffers = self.ensure_mtp_verify_buffers(plan)?;
        let mut target_tokens = vec![0_u32; plan.window_tokens];
        let hidden_values = plan
            .window_tokens
            .checked_mul(plan.hidden_dim)
            .ok_or_else(|| "MTP verify result hidden rows length overflow".to_string())?;
        let mut mtp_hidden_rows = vec![0.0_f32; hidden_values];
        unsafe {
            self.api.memcpy_dtoh_async(
                target_tokens.as_mut_ptr().cast::<libc::c_void>(),
                buffers.target_tokens_dev,
                plan.target_token_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                mtp_hidden_rows.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                plan.hidden_row_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Qwen35MtpDeviceVerifyResult {
            target_tokens,
            mtp_hidden_rows,
            hidden_dim: plan.hidden_dim,
            prefix_states: Vec::new(),
            ssm_final_states: Vec::new(),
            attention_kv_states: Vec::new(),
        })
    }
}
