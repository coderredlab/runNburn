use super::*;

mod draft;
mod output;
mod profile;
use output::finalize_decode_logits;
use profile::report_decode_layer_profile;
impl Engine {
    pub(crate) fn scratch_checkpoint(&self) -> Option<crate::engine::ScratchBuffers> {
        self.scratch.clone()
    }

    pub(crate) fn restore_scratch_checkpoint(
        &mut self,
        scratch: &Option<crate::engine::ScratchBuffers>,
    ) {
        self.scratch = scratch.clone();
    }

    /// Zero-alloc decode path for seq_len=1 (single token generation).
    /// Uses pre-allocated ScratchBuffers instead of Tensor heap allocation.
    pub(super) fn forward_decode(&mut self, token: u32) -> crate::error::Result<Vec<f32>> {
        let (logits, _, _) = self.forward_decode_impl(token, false, true)?;
        Ok(logits.expect("decode logits missing"))
    }

    /// mv27 task 10b-4c-3: full GPU offload decode entry point.
    ///
    /// **Greedy-only path (Option A)**. Same shape as
    /// [`Engine::forward_prefill_fullpath`]: GPU-resident `logit_argmax` shader
    /// returns the next token id only; vocab × f32 logits are never downloaded.
    /// The token id is recorded in `scratch.backend_argmax_token` and the
    /// function returns `Ok(Vec::new())` as a sentinel.
    ///
    /// **Caller contract** (same as prefill): callers using a sampler chain
    /// must NOT enable `RNB_GPU_FULLPATH=1`. `generate_stream_impl` is the
    /// canonical caller and honors the `last_backend_argmax_token` bypass.
    #[cfg(feature = "vulkan")]
    pub(super) fn forward_decode_fullpath(&mut self, token: u32) -> crate::error::Result<Vec<f32>> {
        let mut gpu_runtime = self.backend_runtime.take_gpu_runtime();
        let result = self.fullpath_run_decode_step_impl(token, gpu_runtime.as_mut());
        self.backend_runtime.restore_gpu_runtime(gpu_runtime);
        // KV cache cursor advancement: forward_decode_impl does
        // `kv_cache.set_len(pos + 1)` at the bottom (line ~624). Mirror that
        // here on success so subsequent decode tokens see the right pos_start.
        // GPU-side cursor is managed by KvResidentLayout independently.
        if result.is_ok() {
            let new_len = self.kv_cache.current_len() + 1;
            self.kv_cache.set_len(new_len);
        }
        result
    }

    #[cfg(not(feature = "vulkan"))]
    pub(super) fn forward_decode_fullpath(
        &mut self,
        _token: u32,
    ) -> crate::error::Result<Vec<f32>> {
        Err(crate::error::LlmError::Forward(
            "mv27-task10b-4c: fullpath body wiring pending (vulkan feature disabled)".into(),
        ))
    }

    #[cfg(feature = "vulkan")]
    fn fullpath_run_decode_step_impl(
        &mut self,
        token: u32,
        gpu_runtime: Option<&mut super::backend_runtime::GpuRuntime>,
    ) -> crate::error::Result<Vec<f32>> {
        let gpu = gpu_runtime.ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_decode_step: gpu_runtime is None — fullpath path requires an active GPU runtime".into(),
            )
        })?;
        let weights = self.weights.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_decode_step: engine.weights is None".into(),
            )
        })?;

        let (layer_raw, layer_kinds) =
            super::inference::build_fullpath_layer_raw_weights(weights, &self.metadata)?;

        if weights.token_embd.ggml_type != rnb_loader::GGMLType::Q6_K {
            return Err(crate::error::LlmError::Forward(format!(
                "fullpath_run_decode_step: token_embd ggml_type {:?} — wrapper requires Q6_K",
                weights.token_embd.ggml_type,
            )));
        }
        let output_quant = super::Engine::fullpath_output_quant_or_error(
            weights.output.ggml_type,
            "fullpath_run_decode_step",
        )?;
        let token_embd_q6k = weights.token_embd.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_decode_step: token_embd.data has no contiguous host bytes".into(),
            )
        })?;
        let output_quantized = weights.output.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_decode_step: output.data has no contiguous host bytes".into(),
            )
        })?;
        let output_norm = kernels::tensor_as_f32_slice(&weights.output_norm);

        let metadata = &self.metadata;
        let ffn_inner_dim =
            super::Engine::fullpath_ffn_inner_dim_or_error(weights, "fullpath_run_decode_step")?;
        let fullpath_vocab =
            super::Engine::fullpath_vocab_rows_or_error(weights, "fullpath_run_decode_step")?;
        let max_ctx = self.kv_cache.max_seq_len;
        let kv_cursor = self.kv_cache.current_len();
        let (rope_dim, rope_neox) = self.fullpath_rope_config();

        let staging = super::gpu_runtime::StagingPolicy::default();
        let output = gpu
            .run_fullpath_decode_step(
                token,
                kv_cursor,
                metadata.num_layers,
                metadata.hidden_dim,
                metadata.num_heads,
                metadata.num_kv_heads,
                metadata.head_dim,
                ffn_inner_dim,
                metadata.norm_eps,
                metadata.rope_theta,
                rope_dim,
                rope_neox,
                fullpath_vocab,
                max_ctx,
                output_quantized,
                output_quant,
                output_norm,
                token_embd_q6k,
                &layer_raw,
                &layer_kinds,
                staging,
            )
            .map_err(crate::error::LlmError::Forward)?;

        // mv27-task10b-4c-3: greedy-only return path.
        // Write GPU-side argmax token id into scratch.backend_argmax_token
        // (precedent: forward_decode_backend_argmax_only +
        // backend_runtime/output.rs:67). caller (generate_stream_impl) skips
        // the sampler chain when last_backend_argmax_token() is Some.
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.backend_argmax_token = Some(output.next_token_id);
        }
        let _ = output.kv_cursor_after; // GPU-side cursor; CPU side advanced in caller.
        Ok(Vec::new())
    }

    pub(super) fn forward_decode_backend_argmax_only_inner(
        &mut self,
        token: u32,
    ) -> crate::error::Result<Option<u32>> {
        #[cfg(feature = "cuda")]
        if qwen35_device_verify_decode_enabled() {
            return self.forward_decode_backend_argmax_only_device_verify(token);
        }
        let (_, backend_argmax_token, _) = self.forward_decode_impl(token, true, true)?;
        Ok(backend_argmax_token)
    }

    #[cfg(feature = "cuda")]
    fn forward_decode_backend_argmax_only_device_verify(
        &mut self,
        token: u32,
    ) -> crate::error::Result<Option<u32>> {
        let request = crate::engine::verify_window::MtpVerifyWindowRequest::new(
            token,
            &[],
            crate::engine::verify_window::MtpVerifyBonus::Include,
        );
        let result = self.forward_mtp_device_verify_window_argmax_collect_mtp(&request)?;
        let target = result.target_tokens.first().copied().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "device verify decode produced no target token".to_string(),
            )
        })?;
        if self.mtp_spec_requested() && !result.mtp_hidden_rows.is_empty() {
            self.mtp_observe_target_batch(&[token], &result.mtp_hidden_rows)?;
        }
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.backend_argmax_token = Some(target);
        }
        Ok(Some(target))
    }

    pub(crate) fn forward_verify_all_logits_sequential(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<Vec<f32>>> {
        self.forward_verify_all_logits_sequential_collect_mtp(tokens, true)
            .map(|(logits, _)| logits)
    }

    pub(crate) fn forward_verify_all_logits_sequential_collect_mtp(
        &mut self,
        tokens: &[u32],
        observe_mtp: bool,
    ) -> crate::error::Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut all_logits = Vec::with_capacity(tokens.len());
        let mut mtp_hidden_rows = Vec::new();
        let collect_mtp_hidden = self.mtp_spec_requested();

        for &token in tokens {
            let (logits, _, hidden_row) = self.forward_decode_impl(token, false, false)?;
            let logits = logits.ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "sequential verify requires logits but decode returned argmax-only output"
                        .to_string(),
                )
            })?;
            all_logits.push(logits);
            if collect_mtp_hidden {
                if let Some(row) = hidden_row {
                    mtp_hidden_rows.extend_from_slice(&row);
                }
            }
        }

        if observe_mtp && collect_mtp_hidden && !tokens.is_empty() {
            self.mtp_observe_target_batch(tokens, &mtp_hidden_rows)?;
        }

        Ok((all_logits, mtp_hidden_rows))
    }

    pub(crate) fn forward_verify_argmax_sequential_collect_mtp(
        &mut self,
        token: u32,
    ) -> crate::error::Result<(u32, Vec<f32>)> {
        let (_, backend_argmax_token, hidden_row) = self.forward_decode_impl(token, true, false)?;
        let target_token = if let Some(token) = backend_argmax_token {
            token
        } else {
            let scratch = self.scratch.as_ref().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "argmax verify missing decode scratch logits".to_string(),
                )
            })?;
            crate::sampler::greedy::greedy_sample(&scratch.logits)
        };

        let mut mtp_hidden_rows = Vec::new();
        if self.mtp_spec_requested() {
            if let Some(row) = hidden_row {
                mtp_hidden_rows.extend_from_slice(&row);
            }
        }
        Ok((target_token, mtp_hidden_rows))
    }

    pub(crate) fn forward_verify_window_argmax_collect_mtp(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        let mut target_tokens = Vec::with_capacity(tokens.len());
        let mut mtp_hidden_rows = Vec::new();
        let hidden_dim = self.metadata.hidden_dim;

        for &token in tokens {
            let (target_token, hidden_rows) =
                self.forward_verify_argmax_sequential_collect_mtp(token)?;
            target_tokens.push(target_token);
            mtp_hidden_rows.extend_from_slice(&hidden_rows);
        }

        Ok(crate::engine::verify_window::VerifyWindowResult {
            target_tokens,
            mtp_hidden_rows,
            hidden_dim,
            prefix_state: None,
            prefix_states: Vec::new(),
            #[cfg(any(feature = "cuda", test))]
            ssm_final_states: Vec::new(),
            #[cfg(any(feature = "cuda", test))]
            attention_kv_states: Vec::new(),
        })
    }

    fn forward_decode_impl(
        &mut self,
        token: u32,
        backend_argmax_only: bool,
        observe_mtp: bool,
    ) -> crate::error::Result<(Option<Vec<f32>>, Option<u32>, Option<Vec<f32>>)> {
        let _moe_jit_preload_guard = moe_jit::suppress_preload_requests();

        // memtrace step hook: bump the per-engine counter and emit a
        // `step\t<ts>\t<idx>\tstart` row. The matching `end` row is emitted
        // by `MemtraceStepGuard::drop` below, which fires on all exit paths
        // (Ok, Err, panic unwind) so ts_end is always paired.
        struct MemtraceStepGuard(usize);
        impl Drop for MemtraceStepGuard {
            fn drop(&mut self) {
                memtrace::record_step_end(self.0);
            }
        }
        let _memtrace_guard = {
            let step_idx = self
                .memtrace_step
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            memtrace::record_step_start(step_idx);
            MemtraceStepGuard(step_idx)
        };

        let prof_level = super::policy::profiling_level();
        let profiling = prof_level >= 1;
        let verbose = prof_level >= 2;
        // Take scratch out of self to avoid borrow issues with kv_cache/metadata
        let mut scratch = self.scratch.take().expect("ScratchBuffers not initialized");
        scratch.backend_argmax_only = backend_argmax_only;
        let metadata = &self.metadata;
        let architecture = self.architecture;
        let tokenizer = &self.tokenizer;

        // 1. Embedding lookup → scratch.hidden (on-the-fly dequant)
        let weights = self.weights.as_ref().unwrap();
        let hidden_dim = metadata.hidden_dim;
        {
            let embd = &weights.token_embd;
            let embd_bytes = embd.data.as_bytes().expect("token_embd bytes");
            let row_bytes = embd_bytes.len() / embd.rows;
            let row_start = token as usize * row_bytes;
            let row_data = &embd_bytes[row_start..row_start + row_bytes];
            let f32_row = dequantize_bytes_to_f32(row_data, embd.ggml_type);
            scratch.hidden[..hidden_dim].copy_from_slice(&f32_row[..hidden_dim]);
        }
        let gemma_ple_active = gemma_per_layer_enabled_for_model(weights, metadata, architecture);
        let raw_hidden_for_base = if gemma_ple_active && gemma_ple_pre_emb_scale_base() {
            Some(scratch.hidden[..hidden_dim].to_vec())
        } else {
            None
        };
        apply_embedding_scale_inplace(&mut scratch.hidden[..hidden_dim], metadata, architecture);
        let gemma_per_layer_base = if gemma_ple_active {
            prepare_gemma_per_layer_base(
                weights,
                &Tensor::from_slice(
                    raw_hidden_for_base
                        .as_deref()
                        .unwrap_or(&scratch.hidden[..hidden_dim]),
                    &[1, hidden_dim],
                ),
                &[token],
                metadata,
                architecture,
                metadata.norm_eps,
            )?
        } else {
            None
        };
        let gemma_ple_base_on_device = if gemma_ple_active && !gemma_ple_dynamic_base() {
            if let Some(base) = gemma_per_layer_base.as_ref() {
                backend_runtime::upload_gemma_ple_base(&base.mixed)?;
                true
            } else {
                false
            }
        } else {
            false
        };
        emit_layer_trace("decode-input", usize::MAX, &scratch.hidden[..hidden_dim]);

        let pos = self.kv_cache.current_len();

        // 2. Transformer layers — with per-layer timing for verbose mode
        let t_layers = std::time::Instant::now();
        let mut layer_times = if verbose {
            Vec::with_capacity(metadata.num_layers)
        } else {
            Vec::new()
        };
        let keep_layer_hidden_snapshots =
            gemma_ple_active && gemma_decode_hidden_snapshots_needed();
        let mut layer_hidden_snapshots = if keep_layer_hidden_snapshots {
            vec![None::<Vec<f32>>; metadata.num_layers]
        } else {
            Vec::new()
        };

        // Take gpu_runtime out temporarily to avoid borrow issues
        #[cfg(feature = "vulkan")]
        let mut gpu_runtime = self.backend_runtime.take_gpu_runtime();
        #[cfg(feature = "vulkan")]
        let backend_layers = self.decode_backend_layers_allowed();
        #[cfg(feature = "vulkan")]
        let backend_max_layer = self.decode_backend_max_layer();
        let use_backend_output_logits = backend_runtime::output_logits_enabled_for_runtime();
        let mtp_collect_hidden = self.mtp_spec_requested();
        let kv_cache = &mut self.kv_cache;

        let decode_result: crate::error::Result<Option<Vec<f32>>> = (|| {
            // cu65: pre-populate device KV cache ONCE (first decode token only).
            // Subsequent tokens' KV are written by launch_kv_f16_write in the device QKV path.
            #[cfg(feature = "cuda")]
            if cuda_runtime::cu65_device_qkv_enabled() {
                use std::sync::atomic::{AtomicBool, Ordering};
                static CU65_KV_POPULATED: AtomicBool = AtomicBool::new(false);
                if !CU65_KV_POPULATED.load(Ordering::Relaxed) {
                    let kv_dim_for_populate = if pos > 0 {
                        let (k_bits, _) = kv_cache.get_up_to(0, pos);
                        k_bits.len() / pos
                    } else {
                        0
                    };
                    if kv_dim_for_populate > 0 {
                        for li in 0..metadata.num_layers {
                            let (k_bits, v_bits) = kv_cache.get_up_to(li, pos);
                            let layer_kv = k_bits.len() / pos;
                            let _ = cuda_runtime::populate_device_kv_cache_f16(
                                li, k_bits, v_bits, layer_kv, pos,
                            );
                        }
                        CU65_KV_POPULATED.store(true, Ordering::Relaxed);
                    }
                }
            }

            // cu63: device-resident decode — all 35 layers on GPU with 1 sync total.
            #[cfg(feature = "cuda")]
            let cu63_device_decode_done = {
                let cu63_enabled = cuda_runtime::cu63_device_decode_enabled();
                let is_gemma4_e2b = matches!(architecture, ModelArchitecture::Gemma4)
                    && metadata.hidden_dim == 1536
                    && metadata.head_dim == 512
                    && metadata.num_heads == 8
                    && metadata.num_kv_heads == 1;
                eprintln!("[cu63-gate] enabled={cu63_enabled} is_e2b={is_gemma4_e2b}");
                if is_gemma4_e2b && cuda_runtime::cu63_device_decode_enabled() {
                    let hidden_bytes = hidden_dim * std::mem::size_of::<f32>();
                    let hidden_dev = cuda_runtime::acquire_decode_hidden_carrier(hidden_bytes)
                        .map_err(crate::error::LlmError::Forward)?;
                    cuda_runtime::upload_to_decode_hidden_carrier(
                        &scratch.hidden[..hidden_dim],
                        hidden_dev,
                    )
                    .map_err(crate::error::LlmError::Forward)?;

                    for layer_idx in 0..metadata.num_layers {
                        let (k_bits, v_bits) = kv_cache.get_up_to(layer_idx, pos);
                        let kv_dim = if pos > 0 { k_bits.len() / pos } else { 0 };
                        if kv_dim > 0 {
                            cuda_runtime::populate_device_kv_cache_f16(
                                layer_idx, k_bits, v_bits, kv_dim, pos,
                            )
                            .map_err(crate::error::LlmError::Forward)?;
                        }
                    }

                    for layer_idx in 0..metadata.num_layers {
                        let LayerType::Attention(w) = &weights.layers[layer_idx] else {
                            return Err(crate::error::LlmError::Forward(format!(
                                "cu63 device decode: layer {layer_idx} is not Attention"
                            )));
                        };
                        let q_raw = w.q_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} q_weight bytes unavailable"
                            ))
                        })?;
                        let k_raw = w.k_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} k_weight bytes unavailable"
                            ))
                        })?;
                        let v_raw = w.v_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} v_weight bytes unavailable"
                            ))
                        })?;
                        let o_raw = w.o_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} o_weight bytes unavailable"
                            ))
                        })?;
                        let gate_raw = w.ffn_gate_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} ffn_gate_weight bytes unavailable"
                            ))
                        })?;
                        let up_raw = w.ffn_up_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} ffn_up_weight bytes unavailable"
                            ))
                        })?;
                        let down_raw = w.ffn_down_weight.data.as_bytes().ok_or_else(|| {
                            crate::error::LlmError::Forward(format!(
                                "cu63: layer {layer_idx} ffn_down_weight bytes unavailable"
                            ))
                        })?;
                        let attn_norm_data = kernels::tensor_as_f32_slice(&w.attn_norm);
                        let ffn_norm_data = kernels::tensor_as_f32_slice(&w.ffn_norm);
                        let n_ff = w.ffn_gate_weight.rows;

                        let layer_kv_dim = w.k_weight.rows;
                        let layer_q_rows = w.q_weight.rows;
                        let q_norm_data =
                            w.q_norm.as_ref().map(|t| kernels::tensor_as_f32_slice(t));
                        let k_norm_data =
                            w.k_norm.as_ref().map(|t| kernels::tensor_as_f32_slice(t));
                        let layer_out_scale = w
                            .out_scale
                            .as_ref()
                            .map(|t| kernels::tensor_as_f32_slice(t))
                            .and_then(|s| s.first().copied())
                            .unwrap_or(1.0);
                        cuda_runtime::decode_full_layer_device_resident(
                            layer_idx,
                            q_raw,
                            k_raw,
                            v_raw,
                            o_raw,
                            gate_raw,
                            up_raw,
                            down_raw,
                            attn_norm_data,
                            ffn_norm_data,
                            metadata.hidden_dim,
                            n_ff,
                            metadata.num_heads,
                            metadata.num_kv_heads,
                            metadata.head_dim,
                            layer_kv_dim,
                            layer_q_rows,
                            q_norm_data,
                            k_norm_data,
                            layer_out_scale,
                            metadata.rope_theta,
                            pos,
                            pos,
                            metadata.norm_eps,
                            hidden_dev,
                        )
                        .map_err(crate::error::LlmError::Forward)?;
                    }

                    cuda_runtime::download_from_decode_hidden_carrier(
                        hidden_dev,
                        &mut scratch.hidden[..hidden_dim],
                    )
                    .map_err(crate::error::LlmError::Forward)?;
                    cuda_runtime::sync_decode_stream().map_err(crate::error::LlmError::Forward)?;
                    eprintln!("[cu63-final] hidden[0..8]: {:?}", &scratch.hidden[..8]);
                    true
                } else {
                    false
                }
            };
            #[cfg(not(feature = "cuda"))]
            let cu63_device_decode_done = false;

            // cu75: persistent decode가 성공하면 layer loop + finalize_decode_logits
            // 둘 다 skip 해야 한다. cu74 까진 argmax 만 저장하고 eager 가 그대로
            // 돌아서 net-negative ROI (persistent kernel 198ms + eager 200ms+).
            // mtp_collect_hidden 활성 시엔 post-layer hidden 이 필요하니 persistent
            // 비활성.
            #[cfg(feature = "cuda")]
            let mut persistent_decode_done = false;
            #[cfg(not(feature = "cuda"))]
            let persistent_decode_done = false;
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            let mut backend_output_done = false;
            #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
            let backend_output_done = false;
            #[cfg(feature = "cuda")]
            if !cu63_device_decode_done && !mtp_collect_hidden {
                // cu94 Milestone 0: pass scratch.logits as out buffer so the
                // dispatch path writes full vocab logits back, not just the
                // argmax token. Required for sampler-chain callers and the
                // persistent prefill token-loop wrapper.
                // cu100: skip the 1 MB logits D2H copy when the caller only
                // needs the argmax token (forward_decode_backend_argmax_only).
                // nsys (cu99) showed D2H = 44% of prefill+decode wall, and
                // ~52 MB of that 58 MB total D2H came from per-dispatch logits
                // download. Conditional skip recovers most of it.
                let logits_dst: Option<&mut [f32]> = if backend_argmax_only {
                    None
                } else {
                    Some(&mut scratch.logits[..])
                };
                if let Ok(Some(argmax_token)) =
                    super::persistent_decode_dispatch::try_persistent_decode_dispatch(
                        metadata,
                        architecture,
                        weights,
                        kv_cache,
                        &scratch.hidden[..hidden_dim],
                        pos,
                        token,
                        logits_dst,
                    )
                {
                    scratch.backend_argmax_token = Some(argmax_token as u32);
                    scratch.backend_argmax_only = backend_argmax_only;
                    persistent_decode_done = true;
                }
            }

            if !cu63_device_decode_done && !persistent_decode_done {
                #[cfg(feature = "cuda")]
                let mut cu72_hidden_persistence_trace =
                    if cuda_runtime::cu71_layer_segment_graph_trace_enabled() {
                        Some(decode_layer_graph::Cu72HiddenPersistenceTrace::new())
                    } else {
                        None
                    };
                #[cfg(not(feature = "cuda"))]
                let mut cu72_hidden_persistence_trace: Option<
                    decode_layer_graph::Cu72HiddenPersistenceTrace,
                > = None;
                // cu76 diag: optional eager layer cap for layer-by-layer
                // divergence diff vs persistent decode.  Set
                // RNB_CUDA_EAGER_DECODE_LAYERS=N (1..=num_layers) to break out
                // after layer N's full pipeline.  Combined with
                // RNB_CUDA_EAGER_DECODE_HIDDEN_OUT=path, scratch.hidden is
                // dumped to file as little-endian f32 bytes.
                let eager_layer_cap: Option<usize> = std::env::var("RNB_CUDA_EAGER_DECODE_LAYERS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| n > 0 && n <= metadata.num_layers);
                // M5-B(1.4): 연속 GDN carrier layer 를 단일 command buffer chain 으로 묶는다.
                // pm51: Metal decode env policy 는 rnb-runtime 의 parity policy facade 가 소유한다.
                // chain 진입 precondition:
                //   - legacy carrier policy ON
                //   - eager_layer_cap None (chain 이 여러 layer 처리 → cap 무력화 방지)
                //   - !gemma_ple_active && !keep_layer_hidden_snapshots
                //     (chain 은 hidden 을 device 에 가두므로 host hidden 경로가 OFF 여야).
                //     gemma 등은 자연 OFF — 안전 가드.
                // carrier-ineligible(gemma/non-Q4K)은 try_run_decode_chain 의 layer 별
                // eligible 검사(attn_carrier_eligible/gdn_carrier_eligible)가 걸러 host fallback.
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let decode_chain_enabled =
                    backend_runtime::metal_decode_legacy_carrier_enabled_by_policy()
                        && eager_layer_cap.is_none()
                        && !gemma_ple_active
                        && !keep_layer_hidden_snapshots
                        && architecture != ModelArchitecture::GlmDsa;
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let decode_chain_trace =
                    std::env::var("RNB_METAL_DECODE_CHAIN_TRACE").as_deref() == Ok("1");
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                if decode_chain_trace {
                    eprintln!(
                        "[metal-decode-chain] precondition enabled={} carrier_policy={} eager_none={} gemma_ple={} keep_snapshots={}",
                        decode_chain_enabled,
                        backend_runtime::metal_decode_legacy_carrier_enabled_by_policy(),
                        eager_layer_cap.is_none(),
                        gemma_ple_active,
                        keep_layer_hidden_snapshots
                    );
                }
                // Dense attention은 기존 attn carrier policy, Qwen Attention+MoE는
                // qwen_moe_decode_chain policy가 켜져야 chain에 합류한다.
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let attn_layer_env =
                    backend_runtime::metal_decode_legacy_attn_layer_enabled_by_policy();
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let qwen_moe_decode_chain_env =
                    backend_runtime::metal_qwen_moe_decode_chain_enabled_by_policy();
                // footgun 가드: int8 KV 는 carrier(attn) 경로 전용이다. 사용자가 명시적으로
                // RNB_METAL_KV_INT8=1 을 켜고 carrier(chain/attn) 를 끈(=0) 모순만 startup 에서
                // 죽인다. default(미설정) int8 은 carrier 종속(compute.rs build_metal_context)이라
                // carrier off 시 자동 f16 으로 후퇴하므로 panic 대상이 아니다.
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                if let Some(err) = backend_runtime::metal_decode_kv_int8_requires_carrier_error(
                    decode_chain_enabled,
                    attn_layer_env,
                ) {
                    panic!("{err}; int8 KV is carrier-path only");
                }
                // chain 이 이미 처리한 GDN layer 들을 skip(run 의 마지막 layer index).
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let mut skip_until: Option<usize> = None;
                for layer_idx in 0..metadata.num_layers {
                    #[cfg(all(feature = "metal", not(feature = "cuda")))]
                    if let Some(last) = skip_until {
                        if layer_idx <= last {
                            continue;
                        }
                        skip_until = None;
                    }
                    if let Some(cap) = eager_layer_cap {
                        if layer_idx >= cap {
                            break;
                        }
                    }
                    let ple_hidden_storage = if gemma_ple_active && gemma_ple_use_layer_input() {
                        Some(scratch.hidden[..hidden_dim].to_vec())
                    } else {
                        None
                    };
                    let ple_post_hidden_storage = if gemma_ple_active && gemma_ple_pre_norm_input()
                    {
                        Some(scratch.hidden[..hidden_dim].to_vec())
                    } else {
                        None
                    };
                    let dynamic_base_storage = if gemma_ple_active && gemma_ple_dynamic_base() {
                        prepare_gemma_per_layer_base(
                            weights,
                            &Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim]),
                            &[token],
                            metadata,
                            architecture,
                            metadata.norm_eps,
                        )?
                    } else {
                        None
                    };
                    let ple_base = if gemma_ple_active && gemma_ple_dynamic_base() {
                        dynamic_base_storage.as_ref()
                    } else {
                        gemma_per_layer_base.as_ref()
                    };
                    let ple_base = if gemma_ple_layer_enabled(layer_idx) {
                        ple_base
                    } else {
                        None
                    };
                    let ple_after_out_scale = gemma_ple_after_out_scale()
                        || gemma_ple_layer34_hard_fix_applies(
                            architecture,
                            layer_idx,
                            metadata.num_layers,
                        );
                    if gemma_ple_before_layer() {
                        if let (Some(base), Some(gemma)) =
                            (ple_base, weights.gemma_per_layer.as_ref())
                        {
                            let ple_hidden_tensor = ple_hidden_storage
                                .as_ref()
                                .map(|v| Tensor::from_slice(v, &[1, hidden_dim]));
                            let updated = apply_gemma_per_layer_branch(
                                ple_hidden_tensor.unwrap_or_else(|| {
                                    Tensor::from_slice(
                                        &scratch.hidden[..hidden_dim],
                                        &[1, hidden_dim],
                                    )
                                }),
                                base,
                                layer_idx,
                                gemma,
                                metadata,
                                architecture,
                                metadata.norm_eps,
                            )?;
                            let updated_data = kernels::tensor_as_f32_slice(&updated);
                            scratch.hidden[..hidden_dim]
                                .copy_from_slice(&updated_data[..hidden_dim]);
                        }
                    }
                    let t_layer = std::time::Instant::now();
                    if keep_layer_hidden_snapshots
                        && gemma_reuse_source_hidden_decode_enabled(layer_idx)
                    {
                        if let Some(src_layer) =
                            shared_kv_source_layer(metadata, architecture, layer_idx)
                        {
                            if let Some(src_hidden) = layer_hidden_snapshots
                                .get(src_layer)
                                .and_then(|v| v.as_ref())
                            {
                                let alpha = gemma_reuse_source_hidden_decode_blend_alpha();
                                for (dst, src) in scratch.hidden[..hidden_dim]
                                    .iter_mut()
                                    .zip(src_hidden.iter())
                                {
                                    *dst = alpha * *src + (1.0 - alpha) * *dst;
                                }
                                emit_layer_trace(
                                    "decode",
                                    layer_idx,
                                    &scratch.hidden[..hidden_dim],
                                );
                                emit_decode_layer_target_trace(
                                    &tokenizer,
                                    metadata,
                                    architecture,
                                    weights,
                                    &scratch.hidden[..hidden_dim],
                                    metadata.norm_eps,
                                    layer_idx,
                                )?;
                                layer_hidden_snapshots[layer_idx] =
                                    Some(scratch.hidden[..hidden_dim].to_vec());
                                if verbose {
                                    layer_times.push(t_layer.elapsed().as_micros() as f64 / 1000.0);
                                }
                                continue;
                            }
                        }
                    }
                    if gemma_disable_layer_decode_enabled(layer_idx) {
                        emit_layer_trace("decode", layer_idx, &scratch.hidden[..hidden_dim]);
                        emit_decode_layer_target_trace(
                            &tokenizer,
                            metadata,
                            architecture,
                            weights,
                            &scratch.hidden[..hidden_dim],
                            metadata.norm_eps,
                            layer_idx,
                        )?;
                        if verbose {
                            layer_times.push(t_layer.elapsed().as_micros() as f64 / 1000.0);
                        }
                        continue;
                    }
                    // M5-B(2단계): attn carrier + GDN carrier 연속 run 을 단일 command buffer
                    // chain 으로. layer_idx 부터 연속 eligible run 을 수집해 묶고 skip_until 로
                    // 나머지 skip. attn↔gdn 어느 쪽으로 시작하든 진입(match 전 시도). handle 시
                    // hidden 갱신 + trace 후 다음 layer 로 continue(post-processing skip — 9B 는
                    // PLE/out_scale 자연 OFF 라 attn 분기 post-processing 이 no-op).
                    #[cfg(all(feature = "metal", not(feature = "cuda")))]
                    if decode_chain_enabled {
                        let handled = match &weights.layers[layer_idx] {
                            LayerType::Attention(w) => {
                                if !attn_layer_env && !qwen_moe_decode_chain_env {
                                    false
                                } else {
                                    let kv_source_layer =
                                        shared_kv_source_layer(metadata, architecture, layer_idx);
                                    let owns_kv = kv_source_layer.is_none();
                                    let gemma4_reuse_q_only =
                                        matches!(architecture, ModelArchitecture::Gemma4)
                                            && !owns_kv;
                                    let layer_kv_override = metadata
                                        .head_count_kv_per_layer
                                        .as_ref()
                                        .and_then(|v| v.get(layer_idx).copied());
                                    let layout = if gemma4_reuse_q_only {
                                        resolve_attention_layout_gemma4_reuse(
                                            metadata,
                                            w,
                                            layer_kv_override,
                                        )?
                                    } else {
                                        resolve_attention_layout(metadata, w, layer_kv_override)?
                                    };
                                    (attn_layer_env
                                        && attn_carrier_eligible(
                                            w,
                                            layout.has_gated_attn,
                                            owns_kv,
                                            pos,
                                            pos,
                                            gemma4_reuse_q_only,
                                        ))
                                        || qwen_attn_moe_chain_eligible(
                                            w,
                                            layout.has_gated_attn,
                                            owns_kv,
                                            pos,
                                            pos,
                                            gemma4_reuse_q_only,
                                            qwen_moe_decode_chain_env,
                                        )
                                }
                            }
                            LayerType::GatedDeltaNet(w) => {
                                models::qwen::gdn_carrier_eligible(w)
                                    || qwen_moe_decode_chain_candidate(
                                        w.shared_expert_moe.is_some(),
                                        w.ffn_gate_up_fused.is_some(),
                                        qwen_moe_decode_chain_env,
                                    )
                            }
                            _ => false,
                        };
                        if decode_chain_trace && layer_idx == 0 {
                            eprintln!(
                                "[metal-decode-chain] layer0 handled={} kind={}",
                                handled,
                                match &weights.layers[layer_idx] {
                                    LayerType::Attention(_) => "attention",
                                    LayerType::GatedDeltaNet(_) => "gdn",
                                    LayerType::NemotronMamba2(_) => "nemotron_mamba2",
                                    LayerType::NemotronMoE(_) => "nemotron_moe",
                                }
                            );
                        }
                        if handled {
                            let chain_report = try_run_decode_chain(
                                kv_cache,
                                metadata,
                                architecture,
                                &mut scratch,
                                weights,
                                layer_idx,
                                pos,
                                attn_layer_env,
                                qwen_moe_decode_chain_env,
                                &mut skip_until,
                            )?;
                            if chain_report.did_run {
                                if let Some(token) = chain_report.output_argmax_token {
                                    scratch.backend_argmax_token = Some(token);
                                    apply_model_norm_into(
                                        &scratch.hidden[..hidden_dim],
                                        kernels::tensor_as_f32_slice(&weights.output_norm),
                                        metadata.norm_eps,
                                        &mut scratch.norm_buf[..hidden_dim],
                                        architecture,
                                    );
                                    backend_output_done = true;
                                }
                                let last = skip_until.unwrap_or(layer_idx);
                                emit_layer_trace("decode", last, &scratch.hidden[..hidden_dim]);
                                if verbose {
                                    layer_times.push(t_layer.elapsed().as_micros() as f64 / 1000.0);
                                }
                                continue;
                            }
                        }
                    }
                    let mut ple_fused_in_attention = false;
                    match &weights.layers[layer_idx] {
                        LayerType::Attention(w) => {
                            if architecture == ModelArchitecture::GlmDsa {
                                let glm_layers =
                                    weights.glm_dsa_attention.as_ref().ok_or_else(|| {
                                        crate::error::LlmError::Forward(
                                            "GLM DSA attention weights are not loaded".into(),
                                        )
                                    })?;
                                let glm = glm_layers.get(layer_idx).ok_or_else(|| {
                                    crate::error::LlmError::Forward(format!(
                                        "GLM DSA attention layer {layer_idx} is missing"
                                    ))
                                })?;
                                models::glm_dsa::decode_layer(
                                    kv_cache,
                                    metadata,
                                    &mut scratch,
                                    w,
                                    glm,
                                    layer_idx,
                                    pos,
                                )?;
                            } else if !gemma_ple_disable_attention_layer(layer_idx)
                                && !gemma_disable_attn_decode_enabled(layer_idx)
                            {
                                let src_hidden_vec = if keep_layer_hidden_snapshots {
                                    shared_kv_source_layer(metadata, architecture, layer_idx)
                                        .and_then(|src| layer_hidden_snapshots.get(src))
                                        .and_then(|opt| opt.as_ref())
                                        .cloned()
                                } else {
                                    None
                                };
                                let src_hidden_slice = src_hidden_vec.as_deref();
                                let blend_src_layer = gemma_blend_source_decode_src_layer();
                                let prev_hidden_vec =
                                    if keep_layer_hidden_snapshots && blend_src_layer < layer_idx {
                                        layer_hidden_snapshots
                                            .get(blend_src_layer)
                                            .and_then(|opt| opt.as_ref())
                                            .cloned()
                                    } else {
                                        None
                                    };
                                let prev_hidden_slice = prev_hidden_vec.as_deref();
                                let ple_fusion_for_attention = if !gemma_ple_before_layer()
                                    && !ple_after_out_scale
                                    && !gemma_ple_use_layer_input()
                                    && !gemma_ple_pre_norm_input()
                                    && !gemma_ple_dynamic_base()
                                {
                                    ple_base.and_then(|base| {
                                        weights.gemma_per_layer.as_ref().map(|gemma| (base, gemma))
                                    })
                                } else {
                                    None
                                };
                                let ple_input_device_offset = if gemma_ple_base_on_device
                                    && ple_fusion_for_attention.is_some()
                                {
                                    Some(layer_idx * metadata.embedding_length_per_layer_input)
                                } else {
                                    None
                                };
                                ple_fused_in_attention = decode_attention_layer(
                                    kv_cache,
                                    metadata,
                                    architecture,
                                    &mut scratch,
                                    w,
                                    weights.rope_freqs.as_ref(),
                                    layer_idx,
                                    pos,
                                    src_hidden_slice,
                                    prev_hidden_slice,
                                    ple_fusion_for_attention,
                                    ple_input_device_offset,
                                    cu72_hidden_persistence_trace.as_mut(),
                                    #[cfg(feature = "vulkan")]
                                    if backend_layers && layer_idx < backend_max_layer {
                                        gpu_runtime.as_mut()
                                    } else {
                                        None
                                    },
                                )?;
                            }
                        }
                        LayerType::GatedDeltaNet(w) => {
                            // M5-B(2단계): chain 은 match 전에 시도했으므로 여기 도달하면 chain
                            // 미handle(eligible 아님 또는 chain disabled) — per-layer 경로로 처리.
                            decode_gdn_layer(
                                kv_cache,
                                metadata,
                                &mut scratch,
                                w,
                                layer_idx,
                                #[cfg(feature = "vulkan")]
                                if backend_layers && layer_idx < backend_max_layer {
                                    gpu_runtime.as_mut()
                                } else {
                                    None
                                },
                            )?;
                        }
                        LayerType::NemotronMamba2(w) => {
                            models::nemotron::mamba::decode_mamba2_layer(
                                kv_cache,
                                metadata,
                                &mut scratch,
                                w,
                                layer_idx,
                                metadata.norm_eps,
                            )?;
                        }
                        LayerType::NemotronMoE(w) => {
                            models::nemotron::moe::decode_moe_layer(
                                metadata,
                                &mut scratch,
                                w,
                                metadata.norm_eps,
                            )?;
                        }
                    }
                    if gemma_ple_before_layer() {
                        if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                            apply_layer_output_scale_inplace(
                                &mut scratch.hidden[..hidden_dim],
                                w.out_scale.as_ref(),
                                layer_idx,
                            );
                        }
                        emit_layer_trace("decode", layer_idx, &scratch.hidden[..hidden_dim]);
                        if verbose {
                            layer_times.push(t_layer.elapsed().as_micros() as f64 / 1000.0);
                        }
                        continue;
                    }
                    if let LayerType::Attention(w) = &weights.layers[layer_idx] {
                        if !ple_after_out_scale && !ple_fused_in_attention {
                            if let (Some(base), Some(gemma)) =
                                (ple_base, weights.gemma_per_layer.as_ref())
                            {
                                let ple_hidden_tensor = if let Some(v) = ple_hidden_storage.as_ref()
                                {
                                    Some(Tensor::from_slice(v, &[1, hidden_dim]))
                                } else {
                                    ple_post_hidden_storage
                                        .as_ref()
                                        .map(|v| Tensor::from_slice(v, &[1, hidden_dim]))
                                };
                                let updated = apply_gemma_per_layer_branch(
                                    ple_hidden_tensor.unwrap_or_else(|| {
                                        Tensor::from_slice(
                                            &scratch.hidden[..hidden_dim],
                                            &[1, hidden_dim],
                                        )
                                    }),
                                    base,
                                    layer_idx,
                                    gemma,
                                    metadata,
                                    architecture,
                                    metadata.norm_eps,
                                )?;
                                let updated_data = kernels::tensor_as_f32_slice(&updated);
                                scratch.hidden[..hidden_dim]
                                    .copy_from_slice(&updated_data[..hidden_dim]);
                            }
                        }
                        // cu44 step 20: chain path (ple_fused_in_attention=true) +
                        // env opt-in 시 chain function 내부에서 device-side out_scale
                        // 이미 apply 됨 — host apply skip (double 방지).
                        let device_out_scale_applied = ple_fused_in_attention
                            && cuda_decode_device_out_scale_in_chain_active();
                        if !device_out_scale_applied {
                            apply_layer_output_scale_inplace(
                                &mut scratch.hidden[..hidden_dim],
                                w.out_scale.as_ref(),
                                layer_idx,
                            );
                        }
                        if ple_after_out_scale {
                            if let (Some(base), Some(gemma)) =
                                (ple_base, weights.gemma_per_layer.as_ref())
                            {
                                let ple_hidden_tensor = if let Some(v) = ple_hidden_storage.as_ref()
                                {
                                    Some(Tensor::from_slice(v, &[1, hidden_dim]))
                                } else {
                                    ple_post_hidden_storage
                                        .as_ref()
                                        .map(|v| Tensor::from_slice(v, &[1, hidden_dim]))
                                };
                                let updated = apply_gemma_per_layer_branch(
                                    ple_hidden_tensor.unwrap_or_else(|| {
                                        Tensor::from_slice(
                                            &scratch.hidden[..hidden_dim],
                                            &[1, hidden_dim],
                                        )
                                    }),
                                    base,
                                    layer_idx,
                                    gemma,
                                    metadata,
                                    architecture,
                                    metadata.norm_eps,
                                )?;
                                let updated_data = kernels::tensor_as_f32_slice(&updated);
                                scratch.hidden[..hidden_dim]
                                    .copy_from_slice(&updated_data[..hidden_dim]);
                            }
                        }
                    } else if let (Some(base), Some(gemma)) =
                        (ple_base, weights.gemma_per_layer.as_ref())
                    {
                        let ple_hidden_tensor = if let Some(v) = ple_hidden_storage.as_ref() {
                            Some(Tensor::from_slice(v, &[1, hidden_dim]))
                        } else {
                            ple_post_hidden_storage
                                .as_ref()
                                .map(|v| Tensor::from_slice(v, &[1, hidden_dim]))
                        };
                        let updated = apply_gemma_per_layer_branch(
                            ple_hidden_tensor.unwrap_or_else(|| {
                                Tensor::from_slice(&scratch.hidden[..hidden_dim], &[1, hidden_dim])
                            }),
                            base,
                            layer_idx,
                            gemma,
                            metadata,
                            architecture,
                            metadata.norm_eps,
                        )?;
                        let updated_data = kernels::tensor_as_f32_slice(&updated);
                        scratch.hidden[..hidden_dim].copy_from_slice(&updated_data[..hidden_dim]);
                    }
                    emit_layer_trace("decode", layer_idx, &scratch.hidden[..hidden_dim]);
                    emit_decode_layer_target_trace(
                        &tokenizer,
                        metadata,
                        architecture,
                        weights,
                        &scratch.hidden[..hidden_dim],
                        metadata.norm_eps,
                        layer_idx,
                    )?;
                    if keep_layer_hidden_snapshots {
                        layer_hidden_snapshots[layer_idx] =
                            Some(scratch.hidden[..hidden_dim].to_vec());
                    }
                    if verbose {
                        layer_times.push(t_layer.elapsed().as_micros() as f64 / 1000.0);
                    }
                }
                if let Some(trace) = cu72_hidden_persistence_trace.as_ref() {
                    trace.emit_trace(pos);
                }
                // cu76 diag: dump eager hidden after capped layer loop.
                if let Ok(path) = std::env::var("RNB_CUDA_EAGER_DECODE_HIDDEN_OUT") {
                    let bytes: Vec<u8> = scratch.hidden[..hidden_dim]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    if let Err(e) = std::fs::write(&path, bytes) {
                        eprintln!("[cu76 eager dump] failed to write {path}: {e}");
                    } else {
                        let mean =
                            scratch.hidden[..hidden_dim].iter().sum::<f32>() / hidden_dim as f32;
                        let max_abs = scratch.hidden[..hidden_dim]
                            .iter()
                            .map(|v| v.abs())
                            .fold(0.0f32, f32::max);
                        eprintln!(
                            "[cu76 eager dump] wrote {path} (cap={:?}) mean={:.6} max_abs={:.6} first8={:?}",
                            eager_layer_cap,
                            mean,
                            max_abs,
                            &scratch.hidden[..8.min(hidden_dim)],
                        );
                    }
                }
            } // end if !cu63_device_decode_done

            report_decode_layer_profile(
                weights,
                &layer_times,
                t_layers.elapsed().as_micros() as f64 / 1000.0,
                profiling,
                verbose,
            );

            let mtp_hidden_row = mtp_collect_hidden.then(|| scratch.hidden[..hidden_dim].to_vec());
            if !persistent_decode_done && !backend_output_done {
                finalize_decode_logits(
                    weights,
                    &mut scratch,
                    metadata,
                    architecture,
                    hidden_dim,
                    profiling,
                    verbose,
                    use_backend_output_logits,
                    #[cfg(feature = "vulkan")]
                    gpu_runtime.as_mut(),
                )?;
            }
            Ok(mtp_hidden_row)
        })();

        // Put gpu_runtime back
        #[cfg(feature = "vulkan")]
        {
            self.backend_runtime.restore_gpu_runtime(gpu_runtime);
        }
        let mtp_hidden_row = decode_result?;

        // 4. KV cache update
        kv_cache.set_len(pos + 1);

        // 5. Return logits when requested, then return scratch
        let backend_argmax_token = scratch.backend_argmax_token;
        let result = if backend_argmax_only {
            None
        } else {
            Some(scratch.logits.clone())
        };
        scratch.backend_argmax_only = false;
        // mt83 Stage B: cache normed last-layer hidden for KvBorrow accessor.
        // `finalize_decode_logits` 가 `scratch.norm_buf[..hidden_dim]` 에 output_norm
        // 적용 직후의 hidden 을 써놓는다 (lm_head 입력 형태).
        self.last_layer_hidden_cached.clear();
        self.last_layer_hidden_cached
            .extend_from_slice(&scratch.norm_buf[..hidden_dim]);
        self.scratch = Some(scratch);
        if observe_mtp {
            if let Some(hidden_row) = mtp_hidden_row.as_ref() {
                self.mtp_observe_target_batch(&[token], &hidden_row)?;
            }
        }
        Ok((result, backend_argmax_token, mtp_hidden_row))
    }
}

#[cfg(feature = "cuda")]
fn qwen35_device_verify_decode_enabled() -> bool {
    policy::cuda_device_verify_decode_enabled()
}

fn cuda_decode_device_out_scale_in_chain_active() -> bool {
    #[cfg(feature = "cuda")]
    {
        rnb_runtime::policy::cuda_decode_device_out_scale_enabled()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// `start_layer`부터 연속된 dense attention, Qwen Attention+MoE, GDN carrier layer를
/// 수집해 단일 command buffer chain으로 실행한다(종류가 섞여도 연속이면 한 run).
/// 성공 시 `*skip_until = Some(마지막 layer index)` 로 caller(decode loop)가 run 의 나머지
/// layer 를 skip 하게 하고 `Ok(true)` 반환. chain 이 불가(quant 미지원 등)면 `Ok(false)`
/// → caller 가 per-layer 경로로 fallback. 입력 hidden = `scratch.hidden`, run 종료 후 마지막
/// layer 출력으로 갱신. hidden 은 공유 device buffer 라 attn↔gdn 경계도 host 안 거친다.
///
/// state: GDN conv/delta 는 chain 진입 전 host ssm_state 에서 **clone**(read), 종료 후
/// **write**(read/write 분리 → borrow 충돌 회피). attn prior KV 는 첫 token device init 용으로
/// owned clone(이후 token 은 `filled!=0` 가드로 backend 가 무시 → device 누적 KV 사용).
///
/// Dense attention은 `attn_layer_env`, Attention+MoE는 `qwen_moe_decode_chain_env`를 따른다.
/// 둘 다 꺼진 attention layer에서 run이 종료된다.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn qwen_moe_decode_chain_candidate(
    has_shared_expert_moe: bool,
    has_ffn_gate_up_fused: bool,
    env_enabled: bool,
) -> bool {
    env_enabled && has_shared_expert_moe && !has_ffn_gate_up_fused
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
fn try_run_decode_chain(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    scratch: &mut ScratchBuffers,
    weights: &ModelWeights,
    start_layer: usize,
    pos: usize,
    attn_layer_env: bool,
    qwen_moe_decode_chain_env: bool,
    skip_until: &mut Option<usize>,
) -> crate::error::Result<backend_runtime::MetalDecodeChainRunReport> {
    let num_layers = metadata.num_layers;
    let hidden_dim = metadata.hidden_dim;

    // GDN shape — 9B 의 모든 GDN layer 동일(decode_gdn_layer_qwen 과 같은 식).
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
    let z_dim = d_inner;

    // 1) 연속 carrier-eligible run 수집(attn/gdn 섞임). 미충족 layer 만나면 종료.
    //    각 layer 의 입력 state(owned)와 attn shape 를 같이 모은다(borrow 충돌 회피).
    let mut run: Vec<(usize, &LayerType)> = Vec::new();
    let mut inputs: Vec<backend_runtime::ChainLayerInput> = Vec::new();
    let mut attn_shapes: Vec<Option<backend_runtime::ChainAttnShape>> = Vec::new();
    let mut li = start_layer;
    while li < num_layers {
        match &weights.layers[li] {
            LayerType::GatedDeltaNet(w)
                if models::qwen::gdn_carrier_eligible(w)
                    || qwen_moe_decode_chain_candidate(
                        w.shared_expert_moe.is_some(),
                        w.ffn_gate_up_fused.is_some(),
                        qwen_moe_decode_chain_env,
                    ) =>
            {
                // GDN state 미초기화면 이 layer 부터 run 종료(per-layer 가 init+처리).
                let Some(st) = kv_cache.get_ssm_state(li) else {
                    break;
                };
                inputs.push(backend_runtime::ChainLayerInput::Gdn {
                    conv_state: st.conv_state.clone(),
                    delta_state: st.delta_state.clone(),
                });
                attn_shapes.push(None);
                run.push((li, &weights.layers[li]));
                li += 1;
            }
            LayerType::Attention(w) if attn_layer_env || qwen_moe_decode_chain_env => {
                // Dense attention은 기존 carrier env를, Attention+MoE는 Qwen MoE chain env를
                // 각각 따른다. KV/input/shape 수집은 두 경로가 동일하다.
                let kv_source_layer = shared_kv_source_layer(metadata, architecture, li);
                let kv_cache_layer = kv_source_layer.unwrap_or(li);
                let owns_kv = kv_source_layer.is_none();
                let gemma4_reuse_q_only =
                    matches!(architecture, ModelArchitecture::Gemma4) && !owns_kv;
                let layer_kv_override = metadata
                    .head_count_kv_per_layer
                    .as_ref()
                    .and_then(|v| v.get(li).copied());
                // decode 는 cache_pos == rope_pos == pos.
                let layout = if gemma4_reuse_q_only {
                    resolve_attention_layout_gemma4_reuse(metadata, w, layer_kv_override)?
                } else {
                    resolve_attention_layout(metadata, w, layer_kv_override)?
                };
                if !((attn_layer_env
                    && attn_carrier_eligible(
                        w,
                        layout.has_gated_attn,
                        owns_kv,
                        pos,
                        pos,
                        gemma4_reuse_q_only,
                    ))
                    || qwen_attn_moe_chain_eligible(
                        w,
                        layout.has_gated_attn,
                        owns_kv,
                        pos,
                        pos,
                        gemma4_reuse_q_only,
                        qwen_moe_decode_chain_env,
                    ))
                {
                    break;
                }
                let (carrier_rope_dim, carrier_rope_theta, _) =
                    resolve_rope_params(metadata, architecture, li, layout.head_dim);
                let carrier_filled = backend_runtime::metal_decode_attn_carrier_kv_filled(li);
                let (prior_k, prior_v) = if decode_chain_prior_kv_required(carrier_filled, pos) {
                    let (prior_k, prior_v) = kv_cache.get_up_to(kv_cache_layer, pos);
                    (prior_k.to_vec(), prior_v.to_vec())
                } else {
                    (Vec::new(), Vec::new())
                };
                inputs.push(backend_runtime::ChainLayerInput::Attn { prior_k, prior_v });
                attn_shapes.push(Some(backend_runtime::ChainAttnShape {
                    q_dim: layout.q_dim,
                    q_out_dim: w.q_weight.rows, // gated: q_dim*2 ([query|gate] 인터리브)
                    kv_dim: layout.kv_dim,
                    head_dim: layout.head_dim,
                    num_heads: layout.num_heads,
                    num_kv_heads: layout.num_kv_heads,
                    n_rot: carrier_rope_dim,
                    pos,
                    theta: carrier_rope_theta,
                    scale: (layout.head_dim as f32).sqrt().recip(),
                }));
                run.push((li, &weights.layers[li]));
                li += 1;
            }
            _ => break,
        }
    }
    // run 길이 1 도 chain(specs 길이 1)으로 통일. run 이 비면 per-layer fallback.
    if run.is_empty() {
        let report = backend_runtime::MetalDecodeChainRunReport {
            fallback_reason: Some("no chain candidate"),
            ..backend_runtime::MetalDecodeChainRunReport::default()
        };
        if std::env::var("RNB_METAL_DECODE_CHAIN_TRACE").as_deref() == Ok("1") {
            eprintln!(
                "[metal-decode-chain] run start={} end={} layers=0 did_run=false qwen_moe_layers=0 fallback_reason={}",
                start_layer,
                start_layer,
                report.fallback_reason.unwrap_or("-")
            );
        }
        return Ok(report);
    }
    let last_layer = run[run.len() - 1].0;
    let capacity = kv_cache.max_seq_len;

    // 2) chain 실행. out_states: gdn 만 Some((conv_new, delta_new)), attn 은 None. hidden 갱신.
    let mut out_states: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; run.len()];
    backend_runtime::metal_decode_parity_record_expected_token();
    let output_argmax =
        decode_chain_output_argmax_tail(scratch, weights, metadata, architecture, hidden_dim);
    let report = backend_runtime::metal_decode_chain_run(
        &mut scratch.hidden[..hidden_dim],
        &run,
        &inputs,
        &attn_shapes,
        &mut out_states,
        capacity,
        hidden_dim,
        conv_channels,
        conv_kernel,
        z_dim,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        metadata.norm_eps,
        output_argmax,
    )?;
    if std::env::var("RNB_METAL_DECODE_CHAIN_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "[metal-decode-chain] run start={} end={} layers={} did_run={} qwen_moe_layers={} fallback_reason={}",
            start_layer,
            last_layer,
            run.len(),
            report.did_run,
            report.qwen_moe_layers,
            report.fallback_reason.unwrap_or("-")
        );
    }
    if !report.did_run {
        return Ok(report);
    }

    // 3) GDN state write — 새 conv/delta 를 host ssm_state 에 반영(read 와 분리된 가변 borrow).
    //    attn 은 out=None(KV device 소유 → carrier 가 누적, host write 불필요).
    for ((layer_idx, _), out) in run.iter().zip(out_states.iter()) {
        if let Some((conv_new, delta_new)) = out {
            if let Some(st) = kv_cache.get_ssm_state_mut(*layer_idx) {
                st.conv_state.copy_from_slice(conv_new);
                // pm31: delta residency 시 backend 가 delta 를 빈 Vec 로 반환(device 잔류).
                // host delta_state 는 stale 로 두되 다음 토큰 upload 도 skip 되므로 무해.
                // materialize(speculative/clear)가 요청할 때만 device→host sync.
                if !delta_new.is_empty() {
                    st.delta_state.copy_from_slice(delta_new);
                }
            }
        }
    }

    *skip_until = Some(last_layer);
    Ok(report)
}

#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
fn decode_chain_prior_kv_required(carrier_filled: Option<usize>, pos: usize) -> bool {
    carrier_filled.map_or(true, |filled| filled < pos)
}

#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
fn decode_chain_output_argmax_tail<'a>(
    scratch: &ScratchBuffers,
    weights: &'a ModelWeights,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden_dim: usize,
) -> Option<backend_runtime::MetalDecodeOutputArgmax<'a>> {
    if !scratch.backend_argmax_only
        || use_token_embedding_as_output()
        || !matches!(architecture, ModelArchitecture::Qwen35)
        || metadata.final_logit_softcapping != 0.0
        || weights.output.cols != hidden_dim
    {
        return None;
    }
    let norm_weight = kernels::tensor_as_f32_slice(&weights.output_norm);
    if norm_weight.len() != hidden_dim {
        return None;
    }
    Some(backend_runtime::MetalDecodeOutputArgmax {
        norm_weight,
        output_weight: &weights.output,
        rows: weights.output.rows,
        cols: weights.output.cols,
        eps: metadata.norm_eps,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn decode_chain_prior_kv_required_only_until_carrier_catches_up() {
        assert!(super::decode_chain_prior_kv_required(None, 4556));
        assert!(super::decode_chain_prior_kv_required(Some(0), 4556));
        assert!(super::decode_chain_prior_kv_required(Some(4555), 4556));
        assert!(!super::decode_chain_prior_kv_required(Some(4556), 4556));
        assert!(!super::decode_chain_prior_kv_required(Some(4557), 4556));
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn qwen_moe_old_env_preserves_gdn_candidate() {
        assert!(super::qwen_moe_decode_chain_candidate(true, false, true));
        assert!(!super::qwen_moe_decode_chain_candidate(false, false, true));
        assert!(!super::qwen_moe_decode_chain_candidate(true, true, true));
        assert!(!super::qwen_moe_decode_chain_candidate(true, false, false));
    }
}
