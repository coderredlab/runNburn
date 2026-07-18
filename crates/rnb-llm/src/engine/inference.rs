use super::cpu_runtime::kernels;
use super::layer_weights::LayerType;
use super::logits::{
    finalize_prefill_all_logits, finalize_prefill_argmax_tokens, finalize_prefill_logits,
};
#[cfg(feature = "cuda")]
use super::logits::{
    finalize_prefill_argmax_token_cuda_only, finalize_prefill_argmax_token_cuda_only_carrier,
};
use super::models::gemma::gemma_ple_pre_emb_scale_base;
use super::models::gemma::{apply_embedding_scale, prepare_gemma_per_layer_base};
#[cfg(feature = "cuda")]
use super::models::gemma::{qwen_text_mrope_dim, resolve_rope_params};
use super::policy;
#[cfg(feature = "cuda")]
use super::prefill::run_prefill_layers_cpu_range_carrier;
use super::prefill::{
    new_empty_kv_cache, run_prefill_layers_cpu_range,
    run_prefill_layers_cpu_range_collect_prefix_state,
};
#[cfg(test)]
use super::prefill_handoff::SliceWindowHandoff;
use super::prefill_handoff::{GpuPrefillExecutor, PrefillExecutionPath};
use super::state::Engine;
use super::trace::{dump_bin, dump_bin_dir, emit_layer_trace};
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use super::types::ModelMetadata;

impl Engine {
    pub fn architecture(&self) -> rnb_loader::Architecture {
        self.architecture
    }

    pub fn tool_call_format(&self) -> crate::tool_call::ToolCallFormat {
        match self.architecture {
            rnb_loader::Architecture::Gemma4 | rnb_loader::Architecture::Gemma4Assistant => {
                crate::tool_call::ToolCallFormat::Gemma
            }
            _ => crate::tool_call::ToolCallFormat::Json,
        }
    }

    #[cfg(test)]
    pub(super) fn make_slice_window_handoff(
        &self,
        hidden_after_window: Vec<f32>,
        next_layer_idx: usize,
        next_pos: usize,
    ) -> SliceWindowHandoff {
        SliceWindowHandoff {
            hidden_after_window,
            #[cfg(test)]
            next_layer_idx,
            #[cfg(test)]
            next_pos,
            cpu_kv_cache: self.kv_cache.clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn should_attempt_slice1_gpu_prefill(&self, token_count: usize) -> bool {
        crate::runtime::scheduler::should_attempt_slice1_gpu_prefill(
            token_count,
            GpuPrefillExecutor::for_slice1(&self.metadata).is_some(),
        )
    }

    pub(super) fn select_prefill_path(&self, token_count: usize) -> PrefillExecutionPath {
        let target = crate::runtime::platform::RuntimeTarget::current();
        let profile = crate::runtime::scheduler::select_runtime_execution_profile(
            crate::runtime::scheduler::ExecutionProfileRequest {
                is_android_target: target.is_android(),
                is_mobile_target: target.is_mobile(),
                is_desktop_target: target.is_desktop(),
                cpu_available: true,
                vulkan_available: self.has_active_gpu_prefill_path(),
                cuda_available: false,
                fullpath_requested: crate::runtime::scheduler::fullpath_gpu_prefill_requested(),
                force_mobile_vulkan_requested:
                    crate::runtime::scheduler::force_mobile_vulkan_requested(),
                requested_profile: None,
            },
        );
        crate::runtime::scheduler::select_prefill_path_for_profile(
            profile,
            token_count,
            GpuPrefillExecutor::for_slice1(&self.metadata).is_some(),
            self.has_active_gpu_prefill_path(),
        )
    }

    pub(super) fn forward_prefill_cpu(&mut self, tokens: &[u32]) -> crate::error::Result<Vec<f32>> {
        let vocab_size = self.metadata.vocab_size;

        let weights = match &self.weights {
            Some(w) => w,
            None => return Ok(vec![0.0f32; vocab_size]),
        };

        let seq_len = tokens.len();
        let pos_start = self.kv_cache.current_len();
        let num_heads = self.metadata.num_heads;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rope_theta = self.metadata.rope_theta;
        let norm_eps = self.metadata.norm_eps;

        let profiling = policy::profiling_enabled();
        let t_prefill_setup = profiling.then(std::time::Instant::now);
        let t_embed = profiling.then(std::time::Instant::now);
        let raw_hidden = weights.token_embd.gather(tokens)?;
        if let Some(t_embed) = t_embed {
            eprintln!(
                "  [FWD] token_embd      {:.1}ms",
                t_embed.elapsed().as_micros() as f64 / 1000.0
            );
        }
        let t_embed_scale = profiling.then(std::time::Instant::now);
        let mut hidden =
            apply_embedding_scale(raw_hidden.clone(), &self.metadata, self.architecture);
        if let Some(t_embed_scale) = t_embed_scale {
            eprintln!(
                "  [FWD] embed_scale     {:.1}ms",
                t_embed_scale.elapsed().as_micros() as f64 / 1000.0
            );
        }
        let t_ple_base = profiling.then(std::time::Instant::now);
        let gemma_per_layer_base = prepare_gemma_per_layer_base(
            weights,
            if gemma_ple_pre_emb_scale_base() {
                &raw_hidden
            } else {
                &hidden
            },
            tokens,
            &self.metadata,
            self.architecture,
            norm_eps,
        )?;
        if let Some(t_ple_base) = t_ple_base {
            eprintln!(
                "  [FWD] ple_base        {:.1}ms",
                t_ple_base.elapsed().as_micros() as f64 / 1000.0
            );
        }
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        if dump_bin_dir().is_some() {
            let raw_hidden_data = kernels::tensor_as_f32_slice(&raw_hidden);
            dump_bin("prefill", usize::MAX, "embed_raw", raw_hidden_data);
            dump_bin("prefill", usize::MAX, "embed_scaled", hidden_data);
        }
        let last_row = &hidden_data
            [(seq_len - 1) * self.metadata.hidden_dim..seq_len * self.metadata.hidden_dim];
        emit_layer_trace("prefill-input", usize::MAX, last_row);
        if let Some(t_prefill_setup) = t_prefill_setup {
            eprintln!(
                "  [FWD] setup_total     {:.1}ms",
                t_prefill_setup.elapsed().as_micros() as f64 / 1000.0
            );
        }

        let t_layers = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let use_output_last_row_carrier = policy::cuda_prefill_argmax_only_enabled()
            && !self.mtp_spec_requested()
            && self.architecture == rnb_loader::Architecture::Gemma4
            && !policy::use_token_embedding_as_output()
            && matches!(
                weights.output.ggml_type,
                rnb_loader::GGMLType::Q8_0 | rnb_loader::GGMLType::Q6_K
            );
        #[cfg(feature = "cuda")]
        if use_output_last_row_carrier {
            let hidden_carrier = run_prefill_layers_cpu_range_carrier(
                &mut self.kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                gemma_per_layer_base.as_ref(),
                hidden,
                0..self.metadata.num_layers,
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
            )?;

            if profiling {
                eprintln!(
                    "  [FWD] layers_total     {:.1}ms",
                    t_layers.elapsed().as_micros() as f64 / 1000.0
                );
            }

            if let Some(token) = finalize_prefill_argmax_token_cuda_only_carrier(
                &mut self.kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                hidden_carrier,
                seq_len,
                pos_start,
                norm_eps,
            )? {
                if let Some(scratch) = self.scratch.as_mut() {
                    scratch.backend_argmax_token = Some(token);
                }
                return Ok(Vec::new());
            }
            return Err(crate::error::LlmError::Forward(
                "CUDA output last-row carrier did not produce argmax token".to_string(),
            ));
        }
        hidden = run_prefill_layers_cpu_range(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            gemma_per_layer_base.as_ref(),
            hidden,
            0..self.metadata.num_layers,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )?;

        if profiling {
            eprintln!(
                "  [FWD] layers_total     {:.1}ms",
                t_layers.elapsed().as_micros() as f64 / 1000.0
            );
        }

        let mtp_hidden_rows = self
            .mtp_spec_requested()
            .then(|| kernels::tensor_as_f32_slice(&hidden).to_vec());
        #[cfg(feature = "cuda")]
        if policy::cuda_prefill_argmax_only_enabled() {
            if let Some(token) = finalize_prefill_argmax_token_cuda_only(
                &mut self.kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                hidden.clone(),
                seq_len,
                pos_start,
                norm_eps,
            )? {
                if let Some(scratch) = self.scratch.as_mut() {
                    scratch.backend_argmax_token = Some(token);
                }
                if let Some(hidden_rows) = mtp_hidden_rows {
                    self.mtp_observe_prompt_batch(tokens, &hidden_rows)?;
                }
                return Ok(Vec::new());
            }
        }
        let logits = finalize_prefill_logits(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            hidden,
            seq_len,
            pos_start,
            norm_eps,
            Some(&mut self.last_layer_hidden_cached),
        )?;
        if let Some(hidden_rows) = mtp_hidden_rows {
            self.mtp_observe_prompt_batch(tokens, &hidden_rows)?;
        }
        Ok(logits)
    }

    pub fn forward_prefill_all_logits(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<Vec<f32>>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => {
                // Mock path: 각 위치마다 zero logits
                let vocab_size = self.metadata.vocab_size;
                return Ok(vec![vec![0.0f32; vocab_size]; tokens.len()]);
            }
        };

        let seq_len = tokens.len();
        let pos_start = self.kv_cache.current_len();
        let num_heads = self.metadata.num_heads;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rope_theta = self.metadata.rope_theta;
        let norm_eps = self.metadata.norm_eps;

        let raw_hidden = weights.token_embd.gather(tokens)?;
        let hidden = apply_embedding_scale(raw_hidden.clone(), &self.metadata, self.architecture);
        let gemma_per_layer_base = prepare_gemma_per_layer_base(
            weights,
            if gemma_ple_pre_emb_scale_base() {
                &raw_hidden
            } else {
                &hidden
            },
            tokens,
            &self.metadata,
            self.architecture,
            norm_eps,
        )?;

        let hidden = run_prefill_layers_cpu_range(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            gemma_per_layer_base.as_ref(),
            hidden,
            0..self.metadata.num_layers,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )?;

        let mtp_hidden_rows = self
            .mtp_spec_requested()
            .then(|| kernels::tensor_as_f32_slice(&hidden).to_vec());
        let logits = finalize_prefill_all_logits(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            hidden,
            seq_len,
            pos_start,
            norm_eps,
        )?;
        if let Some(hidden_rows) = mtp_hidden_rows {
            self.mtp_observe_target_batch(tokens, &hidden_rows)?;
        }
        Ok(logits)
    }

    pub(crate) fn forward_prefill_argmax_tokens_collect_mtp(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_prefill_argmax_tokens_collect_mtp_impl(tokens, None, true)
    }

    pub(crate) fn forward_prefill_argmax_tokens_collect_mtp_prefix_state(
        &mut self,
        tokens: &[u32],
        prefix_tokens: usize,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_prefill_argmax_tokens_collect_mtp_impl(tokens, Some(vec![prefix_tokens]), true)
    }

    pub(crate) fn forward_prefill_argmax_tokens_collect_mtp_deferred_observe(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_prefill_argmax_tokens_collect_mtp_impl(tokens, None, false)
    }

    pub(crate) fn forward_prefill_argmax_tokens_collect_mtp_prefix_state_deferred_observe(
        &mut self,
        tokens: &[u32],
        prefix_tokens: usize,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_prefill_argmax_tokens_collect_mtp_impl(
            tokens,
            Some(vec![prefix_tokens]),
            false,
        )
    }

    pub(crate) fn forward_prefill_argmax_tokens_collect_mtp_prefix_states_deferred_observe(
        &mut self,
        tokens: &[u32],
        prefix_tokens: &[usize],
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_prefill_argmax_tokens_collect_mtp_impl(
            tokens,
            Some(prefix_tokens.to_vec()),
            false,
        )
    }

    pub(crate) fn forward_mtp_device_verify_window_argmax_collect_mtp(
        &mut self,
        request: &crate::engine::verify_window::MtpVerifyWindowRequest,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_mtp_device_verify_window_argmax_collect_mtp_impl(
            request,
            request.prefix_tokens(),
            true,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn forward_mtp_device_verify_window_argmax_collect_mtp_shadow(
        &mut self,
        request: &crate::engine::verify_window::MtpVerifyWindowRequest,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        self.forward_mtp_device_verify_window_argmax_collect_mtp_impl(
            request,
            request.shadow_commit_prefix_tokens(),
            false,
        )
    }

    fn forward_mtp_device_verify_window_argmax_collect_mtp_impl(
        &mut self,
        _request: &crate::engine::verify_window::MtpVerifyWindowRequest,
        _prefix_tokens: Vec<usize>,
        _commit_final_states: bool,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        let pos_start = self.kv_cache.current_len();
        #[cfg(feature = "cuda")]
        {
            let request = _request;
            let prefix_tokens = _prefix_tokens;
            let commit_final_states = _commit_final_states;
            let verify_tokens = request.verify_tokens();
            let weights = self.weights.as_ref().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 요청됐지만 {:?} 모델 weights가 로드되지 않음: pos_start={pos_start}. prefill verify로 우회하지 않음",
                    self.architecture
                ))
            })?;
            let token_embd = &weights.token_embd;
            if !matches!(
                token_embd.ggml_type,
                rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K
            ) {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 요청됐지만 token_embd가 Q4_K/Q6_K가 아님: {:?}. prefill verify로 우회하지 않음",
                    token_embd.ggml_type
                )));
            }
            if token_embd.cols != self.metadata.hidden_dim {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 token_embd cols {} != hidden_dim {}. prefill verify로 우회하지 않음",
                    token_embd.cols, self.metadata.hidden_dim
                )));
            }
            let token_embd_q4k = token_embd.data.as_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "RNB_MTP_DEVICE_VERIFY=1 token_embd raw bytes가 없음. prefill verify로 우회하지 않음"
                        .to_string(),
                )
            })?;
            let output = &weights.output;
            if !matches!(
                output.ggml_type,
                rnb_loader::GGMLType::Q4_K
                    | rnb_loader::GGMLType::Q6_K
                    | rnb_loader::GGMLType::Q8_0
            ) {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 요청됐지만 output.weight가 Q4_K/Q6_K/Q8_0가 아님: {:?}. prefill verify로 우회하지 않음",
                    output.ggml_type
                )));
            }
            if output.cols != self.metadata.hidden_dim {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 output cols {} != hidden_dim {}. prefill verify로 우회하지 않음",
                    output.cols, self.metadata.hidden_dim
                )));
            }
            let output_q6k = output.data.as_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "RNB_MTP_DEVICE_VERIFY=1 output.weight raw bytes가 없음. prefill verify로 우회하지 않음"
                        .to_string(),
                )
            })?;
            let output_norm = kernels::tensor_as_f32_slice(&weights.output_norm);
            if output_norm.len() != self.metadata.hidden_dim {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 output_norm len {} != hidden_dim {}. prefill verify로 우회하지 않음",
                    output_norm.len(),
                    self.metadata.hidden_dim
                )));
            }
            let mut layer_graph =
                build_mtp_device_verify_layer_graph(weights, &self.metadata, &mut self.kv_cache)?;
            let (rope_dim, rope_theta, proportional_rope) =
                resolve_rope_params(&self.metadata, self.architecture, 0, self.metadata.head_dim);
            if proportional_rope {
                return Err(crate::error::LlmError::Forward(format!(
                    "RNB_MTP_DEVICE_VERIFY=1 {:?} proportional RoPE는 device verifier에서 아직 지원하지 않음: rope_dim={}, head_dim={}. prefill verify로 우회하지 않음",
                    self.architecture, rope_dim, self.metadata.head_dim
                )));
            }
            let qwen_mrope_dim = qwen_text_mrope_dim(
                &self.metadata,
                self.architecture,
                rope_dim,
                self.metadata.head_dim,
            );
            let device_rope_dim = qwen_mrope_dim.unwrap_or(rope_dim);
            let device_request = crate::engine::cuda_runtime::MtpDeviceVerifyWindowRequest {
                verify_tokens: &verify_tokens,
                prefix_tokens: &prefix_tokens,
                pos_start,
                hidden_dim: self.metadata.hidden_dim,
                rope_dim: device_rope_dim,
                rope_neox: qwen_mrope_dim.is_some(),
                rope_theta,
                include_bonus: matches!(
                    request.bonus,
                    crate::engine::verify_window::MtpVerifyBonus::Include
                ),
                token_embd_q4k,
                token_embd_quant: token_embd.ggml_type as u32,
                token_embd_rows: token_embd.rows,
                token_embd_cols: token_embd.cols,
                layer_order: &layer_graph.layer_order,
                attention_moe_layers: &layer_graph.attention_moe_layers,
                gdn_moe_layers: &mut layer_graph.gdn_moe_layers,
                output_q6k,
                output_quant: output.ggml_type as u32,
                output_rows: output.rows,
                output_cols: output.cols,
                output_norm,
                norm_eps: self.metadata.norm_eps,
            };
            let result =
                crate::engine::cuda_runtime::qwen35_mtp_device_verify_window(device_request)
                    .map_err(|err| {
                        crate::error::LlmError::Forward(format!(
                            "RNB_MTP_DEVICE_VERIFY=1 요청됐지만 {:?} device-resident MTP verify graph 실행 실패: {err}. prefill verify로 우회하지 않음",
                            self.architecture
                        ))
                    })?;
            let result = crate::engine::verify_window::VerifyWindowResult::from_device_result_with_state_payload(
                result.target_tokens,
                result.mtp_hidden_rows,
                result.hidden_dim,
                result.prefix_states,
                result.ssm_final_states,
                result.attention_kv_states,
            )?;
            if commit_final_states {
                self.commit_device_verify_window_final_states(pos_start, &result)?;
            }
            return Ok(result);
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(crate::error::LlmError::Forward(format!(
                "RNB_MTP_DEVICE_VERIFY=1 요청됐지만 {:?} CUDA backend가 빌드되지 않음: pos_start={pos_start}. prefill verify로 우회하지 않음",
                self.architecture,
            )))
        }
    }

    fn forward_prefill_argmax_tokens_collect_mtp_impl(
        &mut self,
        tokens: &[u32],
        prefix_tokens: Option<Vec<usize>>,
        observe_mtp: bool,
    ) -> crate::error::Result<crate::engine::verify_window::VerifyWindowResult> {
        let weights = match &self.weights {
            Some(w) => w,
            None => {
                let token = self.metadata.vocab_size.saturating_sub(1) as u32;
                return Ok(crate::engine::verify_window::VerifyWindowResult {
                    target_tokens: vec![token; tokens.len()],
                    mtp_hidden_rows: Vec::new(),
                    hidden_dim: self.metadata.hidden_dim,
                    prefix_state: None,
                    prefix_states: Vec::new(),
                    #[cfg(any(feature = "cuda", test))]
                    ssm_final_states: Vec::new(),
                    #[cfg(any(feature = "cuda", test))]
                    attention_kv_states: Vec::new(),
                });
            }
        };

        let seq_len = tokens.len();
        let pos_start = self.kv_cache.current_len();
        let num_heads = self.metadata.num_heads;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rope_theta = self.metadata.rope_theta;
        let norm_eps = self.metadata.norm_eps;

        let raw_hidden = weights.token_embd.gather(tokens)?;
        let hidden = apply_embedding_scale(raw_hidden.clone(), &self.metadata, self.architecture);
        let gemma_per_layer_base = prepare_gemma_per_layer_base(
            weights,
            if gemma_ple_pre_emb_scale_base() {
                &raw_hidden
            } else {
                &hidden
            },
            tokens,
            &self.metadata,
            self.architecture,
            norm_eps,
        )?;

        let mut prefix_collector =
            prefix_tokens.map(crate::engine::verify_window::GdnPrefixStateCollector::new_many);
        let hidden = if prefix_collector.is_some() {
            run_prefill_layers_cpu_range_collect_prefix_state(
                &mut self.kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                gemma_per_layer_base.as_ref(),
                hidden,
                0..self.metadata.num_layers,
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
                prefix_collector.as_mut(),
            )?
        } else {
            run_prefill_layers_cpu_range(
                &mut self.kv_cache,
                &self.metadata,
                self.architecture,
                weights,
                gemma_per_layer_base.as_ref(),
                hidden,
                0..self.metadata.num_layers,
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
            )?
        };

        let mtp_hidden_rows = self
            .mtp_spec_requested()
            .then(|| kernels::tensor_as_f32_slice(&hidden).to_vec());
        let target_tokens = finalize_prefill_argmax_tokens(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            hidden,
            seq_len,
            pos_start,
            norm_eps,
        )?;
        if observe_mtp {
            if let Some(hidden_rows) = mtp_hidden_rows.as_ref() {
                self.mtp_observe_target_batch(tokens, hidden_rows)?;
            }
        }
        let mut prefix_states = prefix_collector
            .map(|collector| collector.finish_many_required())
            .transpose()?
            .unwrap_or_default();
        let prefix_state = if prefix_states.len() == 1 {
            Some(prefix_states.remove(0))
        } else {
            None
        };
        Ok(crate::engine::verify_window::VerifyWindowResult {
            target_tokens,
            mtp_hidden_rows: mtp_hidden_rows.unwrap_or_default(),
            hidden_dim: self.metadata.hidden_dim,
            prefix_state,
            prefix_states,
            #[cfg(any(feature = "cuda", test))]
            ssm_final_states: Vec::new(),
            #[cfg(any(feature = "cuda", test))]
            attention_kv_states: Vec::new(),
        })
    }

    pub(crate) fn restore_verify_window_prefix_state(
        &mut self,
        base_kv_len: usize,
        prefix_state: &crate::engine::verify_window::VerifyWindowPrefixState,
    ) -> crate::error::Result<()> {
        self.kv_cache
            .set_len((base_kv_len + prefix_state.prefix_tokens).min(self.kv_cache.max_seq_len));
        #[cfg(feature = "cuda")]
        let mut restored_resident = false;

        for layer in &prefix_state.layers {
            let state = self
                .kv_cache
                .get_ssm_state_mut(layer.layer_idx)
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "missing SSM state for verify prefix layer {}",
                        layer.layer_idx
                    ))
                })?;
            if state.conv_state.len() != layer.conv_state.len() {
                return Err(crate::error::LlmError::Forward(format!(
                    "verify prefix conv_state mismatch for layer {}: got {}, expected {}",
                    layer.layer_idx,
                    layer.conv_state.len(),
                    state.conv_state.len()
                )));
            }
            state.conv_state.copy_from_slice(&layer.conv_state);

            #[cfg(feature = "cuda")]
            {
                if let Some(snapshot) = layer.resident_delta_snapshot.as_ref() {
                    let restored = crate::engine::cuda_runtime::restore_delta_state_cache(
                        &mut state.delta_state,
                        snapshot,
                    )
                    .map_err(crate::error::LlmError::Forward)?;
                    if !restored {
                        return Err(crate::error::LlmError::Forward(format!(
                            "missing resident delta state for verify prefix layer {}",
                            layer.layer_idx
                        )));
                    }
                    restored_resident = true;
                    continue;
                }
            }

            let delta_input = layer.delta_input.as_ref().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "verify prefix layer {} has no delta restore payload",
                    layer.layer_idx
                ))
            })?;
            super::backend_runtime::try_delta_restore_step_if_supported(
                &mut state.delta_state,
                &delta_input.q,
                &delta_input.k,
                &delta_input.v,
                &delta_input.gate,
                &delta_input.beta,
                delta_input.num_heads,
                delta_input.head_k_dim,
                delta_input.head_v_dim,
            )
            .ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "verify prefix restore requires resident delta step support".to_string(),
                )
            })?
            .map_err(crate::error::LlmError::Forward)?;

            #[cfg(feature = "cuda")]
            {
                crate::engine::cuda_runtime::sync_delta_state_cache(&mut state.delta_state)
                    .map_err(crate::error::LlmError::Forward)?;
                restored_resident = true;
            }
        }

        #[cfg(feature = "cuda")]
        {
            self.finalize_resident_sequence_state_after_restore(restored_resident)?;
        }
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn commit_device_verify_window_final_states(
        &mut self,
        base_kv_len: usize,
        result: &crate::engine::verify_window::VerifyWindowResult,
    ) -> crate::error::Result<()> {
        self.kv_cache
            .set_len((base_kv_len + result.len()).min(self.kv_cache.max_seq_len));
        for layer in &result.attention_kv_states {
            if layer.window_tokens != result.len() {
                return Err(crate::error::LlmError::Forward(format!(
                    "device verify attention K/V window mismatch for layer {}: got {}, expected {}",
                    layer.layer_idx,
                    layer.window_tokens,
                    result.len()
                )));
            }
            let expected_values =
                layer
                    .window_tokens
                    .checked_mul(layer.kv_rows)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "device verify attention K/V length overflow for layer {}",
                            layer.layer_idx
                        ))
                    })?;
            if layer.k_bits.len() != expected_values || layer.v_bits.len() != expected_values {
                return Err(crate::error::LlmError::Forward(format!(
                    "device verify attention K/V length mismatch for layer {}: k={} v={} expected {}",
                    layer.layer_idx,
                    layer.k_bits.len(),
                    layer.v_bits.len(),
                    expected_values
                )));
            }
            self.kv_cache.replace_layer_f16_range(
                layer.layer_idx,
                base_kv_len,
                layer.window_tokens,
                &layer.k_bits,
                &layer.v_bits,
            );
        }
        for layer in &result.ssm_final_states {
            let state = self
                .kv_cache
                .get_ssm_state_mut(layer.layer_idx)
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "missing SSM state for device verify final layer {}",
                        layer.layer_idx
                    ))
                })?;
            if state.conv_state.len() != layer.conv_state.len() {
                return Err(crate::error::LlmError::Forward(format!(
                    "device verify final conv_state mismatch for layer {}: got {}, expected {}",
                    layer.layer_idx,
                    layer.conv_state.len(),
                    state.conv_state.len()
                )));
            }
            state.conv_state.copy_from_slice(&layer.conv_state);
        }
        Ok(())
    }

    pub(super) fn forward_prefill_slice1_candidate(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<f32>> {
        let weights = match &self.weights {
            Some(w) => w,
            None => return Ok(vec![0.0f32; self.metadata.vocab_size]),
        };
        let executor = match GpuPrefillExecutor::for_slice1(&self.metadata) {
            Some(executor) => executor,
            None => return self.forward_prefill_cpu(tokens),
        };
        let norm_eps = self.metadata.norm_eps;
        let kv_cache = std::mem::replace(&mut self.kv_cache, new_empty_kv_cache(&self.metadata));

        #[cfg(feature = "vulkan")]
        let handoff = {
            let mut scratch = self.scratch.take();
            let mut gpu_runtime = self.backend_runtime.take_gpu_runtime();
            let result = executor.run(
                kv_cache,
                &self.metadata,
                weights,
                tokens,
                norm_eps,
                scratch.as_mut(),
                gpu_runtime.as_mut(),
            );
            self.scratch = scratch;
            self.backend_runtime.restore_gpu_runtime(gpu_runtime);
            result?
        };

        #[cfg(not(feature = "vulkan"))]
        let handoff = executor.run(kv_cache, &self.metadata, weights, tokens, norm_eps)?;

        self.kv_cache = handoff.cpu_kv_cache;
        Ok(handoff.logits)
    }

    /// mv27 task 10b-4c-3: full GPU offload prefill entry point.
    ///
    /// **Greedy-only path (Option A)**. The fullpath wrapper runs the GPU-resident
    /// `logit_argmax` shader inside the single submit, so only the argmax `u32`
    /// token id is downloaded — the vocab × f32 logits Vec never leaves the GPU.
    ///
    /// Returns `Ok(Vec::new())` as a sentinel meaning "no logits, check
    /// `Engine::last_backend_argmax_token()`". This matches the channel that
    /// [`Self::forward_decode_backend_argmax_only`] already exposes (see
    /// `engine/backend_runtime/output.rs:67` for the precedent on the CUDA side).
    ///
    /// **Caller contract**: callers using a sampler chain (`temperature > 0`,
    /// `top_k`, `top_p`, `min_p`, mirostat) must NOT enable `RNB_GPU_FULLPATH=1`
    /// — fullpath skips logits entirely, so the sampler has nothing to operate
    /// on. `generate_stream_impl` is the canonical caller and honors the
    /// `last_backend_argmax_token` bypass before invoking the sampler.
    ///
    /// take/restore 패턴은 `forward_prefill_slice1_candidate` 와 동일.
    /// Body 는 [`Self::fullpath_run_prefill`] 로 위임한다.
    pub(super) fn forward_prefill_fullpath(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<f32>> {
        let _weights = match &self.weights {
            Some(w) => w,
            None => return Ok(vec![0.0f32; self.metadata.vocab_size]),
        };
        let _norm_eps = self.metadata.norm_eps;
        let kv_cache = std::mem::replace(&mut self.kv_cache, new_empty_kv_cache(&self.metadata));

        #[cfg(feature = "vulkan")]
        let result: crate::error::Result<Vec<f32>> = {
            let mut scratch = self.scratch.take();
            let mut gpu_runtime = self.backend_runtime.take_gpu_runtime();
            let outcome = self.fullpath_run_prefill(tokens, scratch.as_mut(), gpu_runtime.as_mut());
            self.scratch = scratch;
            self.backend_runtime.restore_gpu_runtime(gpu_runtime);
            outcome
        };

        #[cfg(not(feature = "vulkan"))]
        let result: crate::error::Result<Vec<f32>> = {
            let _ = tokens;
            Err(crate::error::LlmError::Forward(
                "mv27-task10b-4c: fullpath body wiring pending (vulkan feature disabled)".into(),
            ))
        };

        // kv_cache 는 fullpath 가 자체적으로 KvResidentLayout 으로 관리하므로 복원.
        // CPU-side cursor advancement: fullpath 성공 시에만 prompt_len 만큼 전진해서
        // 후속 decode 가 올바른 pos_start 를 본다. take/restore 만으로는 cursor 가
        // 0 에 머물러서 다음 decode 가 prefill 처음부터 시작하는 버그가 났었다.
        self.kv_cache = kv_cache;
        if result.is_ok() {
            let new_len = self.kv_cache.current_len() + tokens.len();
            self.kv_cache.set_len(new_len);
        }
        result
    }

    /// mv27 task 10b-4c-3: 4c-2a 의 `LayerRuntime::run_fullpath_prefill`
    /// wrapper 에 raw byte slice 와 dims 를 넘겨 GPU 단일 submit prefill 을 실행한다.
    ///
    /// **Greedy-only design (Option A)**: backend 가 GPU-resident `logit_argmax`
    /// shader 로 reduce 까지 마쳐서 `FullPathPrefillOutput::last_token_id: u32`
    /// 만 돌려준다 (vocab × f32 logits 은 host 로 내려오지 않음). 이 token id
    /// 를 `scratch.backend_argmax_token` 에 기록한 뒤 sentinel 로 빈 `Vec::new()`
    /// 를 반환한다. caller (`generate_stream_impl`) 가
    /// [`Engine::last_backend_argmax_token`] 으로 sampler chain 을 우회한다.
    ///
    /// 이 채널 자체는 새로 만든 게 아니라
    /// [`Engine::forward_decode_backend_argmax_only`] 가 이미 사용하는 mechanism
    /// 을 그대로 재사용한 것 (precedent: `backend_runtime/output.rs:67`).
    ///
    /// `Vec::new()` sentinel: 빈 Vec 는 "logits 없음, backend_argmax_token 확인"
    /// 을 의미한다. `vec![0.0; vocab]` fake one-hot 은 명시적으로 금지 (땜빵).
    /// downstream 에서 `logits.len() == vocab_size` 를 가정하는 코드는
    /// `if let Some(t) = engine.last_backend_argmax_token()` branch 를 먼저
    /// 처리한 뒤에만 logits 에 접근해야 한다.
    #[cfg(feature = "vulkan")]
    fn fullpath_run_prefill(
        &mut self,
        tokens: &[u32],
        scratch: Option<&mut super::types::ScratchBuffers>,
        gpu_runtime: Option<&mut super::backend_runtime::GpuRuntime>,
    ) -> crate::error::Result<Vec<f32>> {
        let gpu = gpu_runtime.ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_prefill: gpu_runtime is None — fullpath path requires an active GPU runtime".into(),
            )
        })?;
        let weights = self.weights.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_prefill: engine.weights is None (mock engine cannot run fullpath)"
                    .into(),
            )
        })?;

        // 1. Build per-layer LayerRawWeights<'_> + ModelLayerKind from
        //    engine.weights. GDN / MoE layers return Err — those are
        //    10b-5 / future task scope.
        let (layer_raw, layer_kinds) = build_fullpath_layer_raw_weights(weights, &self.metadata)?;

        // 2. Pull token_embd / quantized output bytes for the wrapper.
        if weights.token_embd.ggml_type != rnb_loader::GGMLType::Q6_K {
            return Err(crate::error::LlmError::Forward(format!(
                "fullpath_run_prefill: token_embd ggml_type {:?} — wrapper requires Q6_K",
                weights.token_embd.ggml_type,
            )));
        }
        let output_quant =
            Self::fullpath_output_quant_or_error(weights.output.ggml_type, "fullpath_run_prefill")?;
        let token_embd_q6k = weights.token_embd.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_prefill: token_embd.data has no contiguous host bytes".into(),
            )
        })?;
        let output_quantized = weights.output.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "fullpath_run_prefill: output.data has no contiguous host bytes".into(),
            )
        })?;
        let output_norm = kernels::tensor_as_f32_slice(&weights.output_norm);

        // 3. Pull metadata. max_ctx = KVCache::max_seq_len capacity.
        let metadata = &self.metadata;
        let ffn_inner_dim = Self::fullpath_ffn_inner_dim_or_error(weights, "fullpath_run_prefill")?;
        let fullpath_vocab = Self::fullpath_vocab_rows_or_error(weights, "fullpath_run_prefill")?;
        let max_ctx = self.kv_cache.max_seq_len;
        let (rope_dim, rope_neox) = self.fullpath_rope_config();

        // 4. Call wrapper. Output's last_token_id is the GPU-side argmax;
        //    counters are dropped here (caller observes via runtime_counters()).
        let staging = super::gpu_runtime::StagingPolicy::default();
        let output = gpu
            .run_fullpath_prefill(
                tokens,
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

        // 6. mv27-task10b-4c-3: greedy-only return path.
        //    Write GPU-side argmax token id into scratch.backend_argmax_token
        //    (precedent: forward_decode_backend_argmax_only +
        //    backend_runtime/output.rs:67) and return an empty Vec<f32>
        //    sentinel meaning "no logits, check last_backend_argmax_token()".
        //    Caller (generate_stream_impl) bypasses the sampler chain when
        //    last_backend_argmax_token() is Some.
        if let Some(scratch) = scratch {
            scratch.backend_argmax_token = Some(output.last_token_id);
        }
        // KV cache cursor advancement happens in `forward_prefill_fullpath`
        // (after the take/restore restore), since `kv_cache` is moved out
        // before this body runs.
        let _ = output.kv_cursor_after; // GPU-side cursor; CPU side advanced in caller.
        Ok(Vec::new())
    }

    /// Legacy full-table binding for the GPU fullpath `token_embd` Q6_K table.
    ///
    /// Mobile Vulkan devices can cap one storage-buffer descriptor around
    /// 256MiB, while 2B token embeddings are larger than that. Production
    /// fullpath now stages only the requested Q6_K rows into a compact table,
    /// so this is kept only for older smoke paths.
    ///
    /// Standalone `fn` (not a method) so both prefill and decode entry points
    /// can call it without method-borrow conflicts (`gpu_runtime` is borrowed
    /// `&mut` while `&mut self` is also held).
    #[cfg(feature = "vulkan")]
    #[allow(dead_code)]
    pub(super) fn ensure_fullpath_token_embd_bound_inner(
        bound_flag: &mut bool,
        weights: &super::layer_weights::ModelWeights,
        gpu_runtime: &mut super::backend_runtime::GpuRuntime,
    ) -> crate::error::Result<()> {
        if *bound_flag {
            return Ok(());
        }
        if weights.token_embd.ggml_type != rnb_loader::GGMLType::Q6_K {
            return Err(crate::error::LlmError::Forward(format!(
                "ensure_fullpath_token_embd_bound: token_embd ggml_type {:?} — wrapper requires Q6_K",
                weights.token_embd.ggml_type,
            )));
        }
        let bytes = weights.token_embd.data.as_bytes().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "ensure_fullpath_token_embd_bound: token_embd has no contiguous host bytes".into(),
            )
        })?;
        gpu_runtime
            .bind_token_embd(bytes)
            .map_err(crate::error::LlmError::Forward)?;
        *bound_flag = true;
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn fullpath_output_quant_or_error(
        ggml_type: rnb_loader::GGMLType,
        caller: &'static str,
    ) -> crate::error::Result<super::gpu_runtime::Quant> {
        match ggml_type {
            rnb_loader::GGMLType::Q6_K => Ok(super::gpu_runtime::Quant::Q6K),
            rnb_loader::GGMLType::Q8_0 => Ok(super::gpu_runtime::Quant::Q8_0),
            other => Err(crate::error::LlmError::Forward(format!(
                "{caller}: output ggml_type {other:?} — wrapper requires Q6_K or Q8_0",
            ))),
        }
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn fullpath_vocab_rows_or_error(
        weights: &super::layer_weights::ModelWeights,
        caller: &'static str,
    ) -> crate::error::Result<usize> {
        if weights.token_embd.rows != weights.output.rows {
            return Err(crate::error::LlmError::Forward(format!(
                "{caller}: token_embd rows {} != output rows {}",
                weights.token_embd.rows, weights.output.rows
            )));
        }
        Ok(weights.token_embd.rows)
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn fullpath_rope_config(&self) -> (usize, bool) {
        let rope_dim = if self.metadata.rope_dim == 0 {
            self.metadata.head_dim
        } else {
            self.metadata.rope_dim.min(self.metadata.head_dim)
        };
        let rope_neox = matches!(self.architecture, rnb_loader::Architecture::Gemma4)
            && policy::gemma_neox_rope_enabled();
        (rope_dim, rope_neox)
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn fullpath_ffn_inner_dim_or_error(
        weights: &super::layer_weights::ModelWeights,
        caller: &'static str,
    ) -> crate::error::Result<usize> {
        let mut max_ffn_inner = 0usize;
        for (idx, layer) in weights.layers.iter().enumerate() {
            let rows = match layer {
                super::layer_weights::LayerType::Attention(w) => w.ffn_gate_weight.rows,
                super::layer_weights::LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
                other => {
                    return Err(crate::error::LlmError::Forward(format!(
                        "{caller}: layer {idx} is {}, which is not supported by fullpath ffn_inner sizing",
                        layer_type_name(other),
                    )));
                }
            };
            max_ffn_inner = max_ffn_inner.max(rows);
        }
        if max_ffn_inner == 0 {
            return Err(crate::error::LlmError::Forward(format!(
                "{caller}: no supported layers for fullpath ffn_inner sizing",
            )));
        }
        Ok(max_ffn_inner)
    }

    fn forward_prefill(&mut self, tokens: &[u32]) -> crate::error::Result<Vec<f32>> {
        crate::generate::check_generation_cancellation()?;
        super::backend_runtime::clear_host_registered_ranges_before_prefill()?;
        super::backend_runtime::clear_decode_attention_kv_cache_before_prefill()?;
        let result = (|| {
            let chunk_size = self.prefill_chunk_size();
            let full_path = self.select_prefill_path(tokens.len());
            let logits = if matches!(full_path, PrefillExecutionPath::Fullpath) {
                self.forward_prefill_fullpath(tokens)?
            } else if matches!(full_path, PrefillExecutionPath::Slice1GpuCandidate) {
                self.forward_prefill_slice1_candidate(tokens)?
            } else if tokens.len() > chunk_size {
                let mut last_logits = vec![0.0f32; self.metadata.vocab_size];
                for chunk in tokens.chunks(chunk_size) {
                    crate::generate::check_generation_cancellation()?;
                    last_logits = match self.select_prefill_path(chunk.len()) {
                        PrefillExecutionPath::Cpu => self.forward_prefill_cpu(chunk),
                        PrefillExecutionPath::Slice1GpuCandidate => {
                            self.forward_prefill_slice1_candidate(chunk)
                        }
                        PrefillExecutionPath::Fullpath => self.forward_prefill_fullpath(chunk),
                    }?;
                }
                last_logits
            } else {
                match self.select_prefill_path(tokens.len()) {
                    PrefillExecutionPath::Cpu => self.forward_prefill_cpu(tokens),
                    PrefillExecutionPath::Slice1GpuCandidate => {
                        self.forward_prefill_slice1_candidate(tokens)
                    }
                    PrefillExecutionPath::Fullpath => self.forward_prefill_fullpath(tokens),
                }?
            };
            Ok(logits)
        })();
        let cleanup =
            super::backend_runtime::release_prefill_residency_after_prefill(self.architecture);
        match (result, cleanup) {
            (Ok(logits), Ok(())) => Ok(logits),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Err(error), Err(cleanup)) => Err(crate::error::LlmError::Forward(format!(
                "{error}; prefill residency cleanup failed: {cleanup}"
            ))),
        }
    }

    pub fn forward(&mut self, tokens: &[u32]) -> crate::error::Result<Vec<f32>> {
        crate::generate::check_generation_cancellation()?;
        // Zero-alloc fast path for single-token decode
        if tokens.len() == 1 && self.scratch.is_some() && self.weights.is_some() {
            // mv27 task 10b-4a skeleton: RNB_GPU_FULLPATH=1 + slice1 eligibility +
            // 활성 GPU prefill path 가 있으면 fullpath decode skeleton 으로 분기.
            // 본체 wiring 은 4c 에서 채움. unset 이면 기존 zero-alloc decode 그대로.
            if crate::runtime::scheduler::fullpath_gpu_prefill_requested()
                && GpuPrefillExecutor::for_slice1(&self.metadata).is_some()
                && self.has_active_gpu_prefill_path()
            {
                return self.forward_decode_fullpath(tokens[0]);
            }
            return self.forward_decode(tokens[0]);
        }

        // GLM DSA uses a compressed MLA KV cache. For prompts with enough
        // routing slots to cover the expert set, the layer-major path batches
        // MLA attention and deduplicates sparse-expert uploads. Short prompts
        // retain tokenwise decode to avoid displacing the hot decode cache.
        if tokens.len() > 1
            && self.architecture == rnb_loader::Architecture::GlmDsa
            && self.scratch.is_some()
            && self.weights.is_some()
        {
            let expert_count = self
                .weights
                .as_ref()
                .and_then(|weights| {
                    weights.layers.iter().find_map(|layer| match layer {
                        LayerType::Attention(layer) => {
                            layer.shared_expert_moe.as_ref().map(|moe| moe.n_expert)
                        }
                        _ => None,
                    })
                })
                .unwrap_or(0);
            if policy::glm_dsa_batch_prefill_enabled(
                tokens.len(),
                expert_count,
                self.metadata.expert_used_count,
            ) {
                return self.forward_prefill_cpu(tokens);
            }
            let mut last_logits = Vec::new();
            for &token in tokens {
                last_logits = self.forward_decode(token)?;
            }
            return Ok(last_logits);
        }

        // cu101 Milestone 2 — true batch prefill: single cooperative kernel
        // dispatch processes `tokens.len()` tokens via the in-kernel outer
        // token loop (cu100 plumbing). Pre-gathers embeddings + per-token PLE
        // bases on the host, then calls try_persistent_decode_dispatch_batch.
        // Env-gated `RNB_CUDA_PERSISTENT_PREFILL_BATCH=1` so the cu94 token
        // loop fallback stays the default until batch path is fully verified.
        #[cfg(feature = "cuda")]
        if tokens.len() > 1
            && self.scratch.is_some()
            && self.weights.is_some()
            && std::env::var("RNB_CUDA_PERSISTENT_PREFILL_BATCH").is_ok()
        {
            return self.forward_persistent_prefill_batch(tokens);
        }

        // cu94 Milestone 1: persistent prefill via token-by-token decode loop.
        // Bypass eager forward_prefill (broken at layer 4 with long prompts —
        // hidden state NaN at layer 5 entry) by feeding each prompt token
        // through forward_decode sequentially. KV cache accumulates one token
        // at a time, identical to natural decode. Last token's logits returned.
        // Correctness-first; perf is intentionally bad (N cooperative launches).
        if tokens.len() > 1
            && self.scratch.is_some()
            && self.weights.is_some()
            && std::env::var("RNB_CUDA_PERSISTENT_PREFILL").is_ok()
        {
            let trace = std::env::var("RNB_CUDA_PERSISTENT_PREFILL_TRACE").is_ok();
            let mut last_logits: Vec<f32> = Vec::new();
            let last_idx = tokens.len() - 1;
            for (step, &tok) in tokens.iter().enumerate() {
                // cu100: skip per-token logits D2H for all but the last prefill
                // token. The caller only needs the last token's logits (to
                // sample the first decode token). nsys (cu99) showed logits
                // D2H = ~52 MB / 1212 ms across the 47-token loop.
                if step < last_idx {
                    let _ = self.forward_decode_backend_argmax_only(tok)?;
                    last_logits = Vec::new();
                } else {
                    last_logits = self.forward_decode(tok)?;
                }
                if trace {
                    let mut nan = 0usize;
                    let mut inf = 0usize;
                    let mut max_abs: f32 = 0.0;
                    let mut argmax: (usize, f32) = (0, f32::NEG_INFINITY);
                    for (i, &v) in last_logits.iter().enumerate() {
                        if v.is_nan() {
                            nan += 1;
                        } else if v.is_infinite() {
                            inf += 1;
                        } else {
                            if v.abs() > max_abs {
                                max_abs = v.abs();
                            }
                            if v > argmax.1 {
                                argmax = (i, v);
                            }
                        }
                    }
                    eprintln!(
                        "[cu94-pp] step={step} tok_in={tok} logits_len={} nan={nan} inf={inf} max_abs={max_abs:.3} argmax=({},{:.3})",
                        last_logits.len(),
                        argmax.0,
                        argmax.1
                    );
                }
            }
            if std::env::var("RNB_KV_DUMP_AFTER_FORWARD").is_ok() {
                dump_kv_layer0_token0(&self.kv_cache, "persistent_prefill_loop");
            }
            return Ok(last_logits);
        }

        let logits = self.forward_prefill(tokens)?;
        // cu95: KV dump for cross-path comparison (RNB_KV_DUMP_AFTER_FORWARD=1).
        if std::env::var("RNB_KV_DUMP_AFTER_FORWARD").is_ok() {
            dump_kv_layer0_token0(&self.kv_cache, "eager_prefill");
        }
        Ok(logits)
    }

    pub(crate) fn forward_with_logits(&mut self, tokens: &[u32]) -> crate::error::Result<Vec<f32>> {
        if tokens.len() == 1 && self.scratch.is_some() && self.weights.is_some() {
            self.forward_decode(tokens[0])
        } else {
            self.forward(tokens)
        }
    }

    /// cu101 Milestone 2 — true batch prefill via single cooperative kernel
    /// dispatch. Pre-gathers per-token embeddings on the host (dequant +
    /// embedding_scale), then calls the batch dispatch entry which packs the
    /// per-layer PLE bases and walks `seq_len` token iterations inside the
    /// kernel's outer loop (cu100 plumbing). Returns the last token's logits.
    ///
    /// Env-gated by `RNB_CUDA_PERSISTENT_PREFILL_BATCH=1` until correctness is
    /// verified across all prompts in the cu97 suite.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_persistent_prefill_batch(
        &mut self,
        tokens: &[u32],
    ) -> crate::error::Result<Vec<f32>> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "forward_persistent_prefill_batch: weights missing".into(),
            )
        })?;
        let metadata = &self.metadata;
        let architecture = self.architecture;
        let hidden_dim = metadata.hidden_dim;
        let seq_len = tokens.len();

        // 1) Per-token embedding gather + embedding_scale (host CPU).
        //
        // cu102: eager (decode_inference.rs:331) uses raw embedding (pre-scale)
        // for PLE base prep when `gemma_ple_pre_emb_scale_base()` is true,
        // otherwise the scaled hidden. Without this batch path was sending
        // scaled embeddings into prepare_gemma_per_layer_base and producing
        // NaN-contaminated mixed at layer 1+ entry.
        let mut input_hidden: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
        let mut raw_hidden_for_base: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
        let use_raw_for_ple_base = super::models::gemma::gemma_ple_pre_emb_scale_base();
        {
            let embd = &weights.token_embd;
            let embd_bytes = embd.data.as_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(
                    "forward_persistent_prefill_batch: token_embd has no host bytes".into(),
                )
            })?;
            let row_bytes = embd_bytes.len() / embd.rows;
            for &tok in tokens {
                let row_start = tok as usize * row_bytes;
                let row_data = &embd_bytes[row_start..row_start + row_bytes];
                let f32_row = super::dequant::dequantize_bytes_to_f32(row_data, embd.ggml_type);
                let mut scaled: Vec<f32> = f32_row[..hidden_dim].to_vec();
                if use_raw_for_ple_base {
                    raw_hidden_for_base.extend_from_slice(&scaled);
                }
                super::models::gemma::apply_embedding_scale_inplace(
                    &mut scaled,
                    metadata,
                    architecture,
                );
                input_hidden.extend_from_slice(&scaled);
            }
        }
        // PLE base 계산 시 caller 가 raw 또는 scaled 선택. dispatch 가 받는
        // input_hidden 는 항상 scaled (kernel 의 layer 0 input). PLE base 만
        // raw 가 필요한 경우 따로 prepare 후 dispatch 에 PLE buffer 전달.
        let _ = use_raw_for_ple_base;
        let _ = raw_hidden_for_base;

        // 2) Dispatch. Batch entry runs PLE base for all tokens internally.
        let scratch = self.scratch.as_mut().ok_or_else(|| {
            crate::error::LlmError::Forward(
                "forward_persistent_prefill_batch: scratch missing".into(),
            )
        })?;
        let logits_dst: &mut [f32] = &mut scratch.logits[..];
        let pos_start = self.kv_cache.current_len();
        let weights_ref = self.weights.as_ref().unwrap();
        let argmax = super::persistent_decode_dispatch::try_persistent_decode_dispatch_batch(
            metadata,
            architecture,
            weights_ref,
            &mut self.kv_cache,
            &input_hidden,
            pos_start,
            tokens,
            Some(logits_dst),
        )
        .map_err(crate::error::LlmError::Forward)?;
        if argmax.is_none() {
            return Err(crate::error::LlmError::Forward(
                "forward_persistent_prefill_batch: dispatch returned None (not eligible?)".into(),
            ));
        }
        // 3) Advance KV cursor by seq_len.
        let new_len = pos_start + seq_len;
        self.kv_cache.set_len(new_len);

        // cu104: KV dump for batch path verification.
        if std::env::var("RNB_KV_DUMP_AFTER_FORWARD").is_ok() {
            dump_kv_layer0_token0(&self.kv_cache, "persistent_prefill_batch");
        }
        // 4) Return scratch.logits (last token only — kernel skips per-token
        //    output projection except __t + 1 == seq_len).
        Ok(scratch.logits.clone())
    }

    pub fn last_backend_argmax_token(&self) -> Option<u32> {
        self.scratch
            .as_ref()
            .and_then(|scratch| scratch.backend_argmax_token)
    }

    /// Test-only: install a minimal `ScratchBuffers` in a mock engine and seed
    /// `backend_argmax_token`, so the `generate_stream_impl` fullpath bypass can
    /// be exercised end-to-end without a real GPU. Production code paths set
    /// the field via the GPU runtime; tests need a way to mimic that.
    #[cfg(test)]
    pub fn force_backend_argmax_token_for_test(&mut self, token: Option<u32>) {
        // Minimal scratch: only `backend_argmax_token` is read by the bypass
        // path; the other Vec<f32> fields are sized small to keep the test
        // cheap. ffn_inner_dim=1 picks a tiny FFN scratch — never executed
        // by mock engine (weights=None short-circuits forward_prefill_cpu).
        if self.scratch.is_none() {
            self.scratch = Some(super::types::ScratchBuffers::new(&self.metadata, 1));
        }
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.backend_argmax_token = token;
        }
    }

    pub fn forward_decode_backend_argmax_only(
        &mut self,
        token: u32,
    ) -> crate::error::Result<Option<u32>> {
        self.forward_decode_backend_argmax_only_inner(token)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        params: &crate::generate::GenerateParams,
    ) -> crate::error::Result<crate::generate::GenerateResult> {
        crate::generate::generate(self, prompt, params)
    }

    pub fn generate_stream(
        &mut self,
        prompt: &str,
        params: &crate::generate::GenerateParams,
        callback: impl FnMut(&str) -> bool,
    ) -> crate::error::Result<crate::generate::GenerateResult> {
        crate::generate::generate_stream(self, prompt, params, callback)
    }

    pub fn generate_stream_cancellable(
        &mut self,
        prompt: &str,
        params: &crate::generate::GenerateParams,
        cancellation: &crate::generate::GenerationCancellation,
        callback: impl FnMut(&str) -> bool,
    ) -> crate::error::Result<crate::generate::GenerateResult> {
        crate::generate::generate_stream_cancellable(self, prompt, params, cancellation, callback)
    }

    pub fn generate_stream_resuming(
        &mut self,
        prompt: &str,
        params: &crate::generate::GenerateParams,
        state: &super::EngineSequenceState,
        callback: impl FnMut(&str) -> bool,
    ) -> crate::error::Result<crate::generate::GenerateResult> {
        crate::generate::generate_stream_resuming(self, prompt, params, state, callback)
    }

    pub fn generate_stream_resuming_cancellable(
        &mut self,
        prompt: &str,
        params: &crate::generate::GenerateParams,
        state: &super::EngineSequenceState,
        cancellation: &crate::generate::GenerationCancellation,
        callback: impl FnMut(&str) -> bool,
    ) -> crate::error::Result<crate::generate::GenerateResult> {
        crate::generate::generate_stream_resuming_cancellable(
            self,
            prompt,
            params,
            state,
            cancellation,
            callback,
        )
    }
}

#[cfg(feature = "cuda")]
pub(super) struct MtpDeviceVerifyLayerGraph<'a> {
    pub(super) layer_order: Vec<super::cuda_runtime::MtpDeviceVerifyLayerKind>,
    pub(super) attention_moe_layers: Vec<super::cuda_runtime::MtpDeviceVerifyAttentionMoeLayer<'a>>,
    pub(super) gdn_moe_layers: Vec<super::cuda_runtime::MtpDeviceVerifyGdnMoeLayer<'a>>,
}

#[cfg(feature = "cuda")]
pub(super) fn build_mtp_device_verify_layer_graph<'a>(
    weights: &'a super::layer_weights::ModelWeights,
    metadata: &ModelMetadata,
    kv_cache: &'a mut crate::kv_cache::KVCache,
) -> crate::error::Result<MtpDeviceVerifyLayerGraph<'a>> {
    let attention_moe_layers =
        build_mtp_device_verify_attention_moe_layers(weights, metadata, kv_cache)?;
    let gdn_moe_layers =
        build_mtp_device_verify_gdn_moe_layers_inner(weights, metadata, kv_cache, true)?;
    let layer_order =
        build_mtp_device_verify_layer_order(weights, &attention_moe_layers, &gdn_moe_layers)?;
    Ok(MtpDeviceVerifyLayerGraph {
        layer_order,
        attention_moe_layers,
        gdn_moe_layers,
    })
}

/// cu95: dump layer 0 token 0 K (first 16 fp16 → f32) to compare KV state
/// between eager batch prefill and persistent token-by-token loop.
fn dump_kv_layer0_token0(kv_cache: &crate::kv_cache::KVCache, tag: &str) {
    let dump_layer = |layer: usize| {
        let key = kv_cache.get_key(layer);
        let val = kv_cache.get_value(layer);
        let current_len = kv_cache.current_len;
        if key.is_empty() {
            eprintln!("[cu95-kv L{layer} {tag}] empty (current_len={current_len})");
            return;
        }
        let stride = key.len() / current_len.max(1);
        let n = 8.min(stride);
        let k_vals: Vec<f32> = key[..n]
            .iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect();
        let v_vals: Vec<f32> = val[..n]
            .iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect();
        let mut per_tok_k: Vec<f32> = Vec::with_capacity(current_len);
        for t in 0..current_len {
            let v = key.get(t * stride).copied().unwrap_or(0);
            per_tok_k.push(half::f16::from_bits(v).to_f32());
        }
        eprintln!("[cu95-kv L{layer} {tag}] cl={current_len} stride={stride} k_t0={k_vals:?} v_t0={v_vals:?}");
        eprintln!("[cu95-kv L{layer} {tag}] per_tok_k0={per_tok_k:?}");
    };
    for layer in [0usize, 4, 5, 9, 13, 14] {
        if layer < kv_cache.num_layers() {
            dump_layer(layer);
        }
    }
}

#[cfg(feature = "cuda")]
fn build_mtp_device_verify_layer_order(
    weights: &super::layer_weights::ModelWeights,
    attention_moe_layers: &[super::cuda_runtime::MtpDeviceVerifyAttentionMoeLayer<'_>],
    gdn_moe_layers: &[super::cuda_runtime::MtpDeviceVerifyGdnMoeLayer<'_>],
) -> crate::error::Result<Vec<super::cuda_runtime::MtpDeviceVerifyLayerKind>> {
    let mut order = Vec::new();
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        match layer {
            super::layer_weights::LayerType::Attention(_) => {
                let pos = attention_moe_layers
                    .iter()
                    .position(|candidate| candidate.layer_index == layer_idx)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "MTP device verify layer order missing attention layer {layer_idx}"
                        ))
                    })?;
                order.push(super::cuda_runtime::MtpDeviceVerifyLayerKind::AttentionMoe(
                    pos,
                ));
            }
            super::layer_weights::LayerType::GatedDeltaNet(_) => {
                let pos = gdn_moe_layers
                    .iter()
                    .position(|candidate| candidate.layer_index == layer_idx)
                    .ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "MTP device verify layer order missing GDN layer {layer_idx}"
                        ))
                    })?;
                order.push(super::cuda_runtime::MtpDeviceVerifyLayerKind::GdnMoe(pos));
            }
            other => {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} type {} is not wired into MTP device verify graph",
                    layer_type_name_for_mtp_device(other)
                )));
            }
        }
    }
    Ok(order)
}

#[cfg(feature = "cuda")]
fn build_mtp_device_verify_attention_moe_layers<'a>(
    weights: &'a super::layer_weights::ModelWeights,
    metadata: &ModelMetadata,
    kv_cache: &crate::kv_cache::KVCache,
) -> crate::error::Result<Vec<super::cuda_runtime::MtpDeviceVerifyAttentionMoeLayer<'a>>> {
    let mut layers = Vec::new();
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let attn = match layer {
            super::layer_weights::LayerType::Attention(attn) => attn,
            super::layer_weights::LayerType::GatedDeltaNet(_) => continue,
            other => {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} type {} is not wired into MTP device verify graph",
                    layer_type_name_for_mtp_device(other)
                )))
            }
        };
        let (q_q4k, q_quant) =
            k_quant_weight_bytes_for_mtp_device(&attn.q_weight, layer_idx, "attn_q")?;
        let (k_q4k, k_quant) =
            k_quant_weight_bytes_for_mtp_device(&attn.k_weight, layer_idx, "attn_k")?;
        let (v_q4k, v_quant) =
            k_quant_weight_bytes_for_mtp_device(&attn.v_weight, layer_idx, "attn_v")?;
        let (o_q4k, o_quant) =
            k_quant_weight_bytes_for_mtp_device(&attn.o_weight, layer_idx, "attn_o")?;
        let q_norm = attn.q_norm.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} attention q_norm missing"
            ))
        })?;
        let k_norm = attn.k_norm.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} attention k_norm missing"
            ))
        })?;
        let post_attn_norm = attn.post_attn_norm.as_ref().ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} attention post_attn_norm missing"
            ))
        })?;
        let (
            ffn_gate_q4k,
            ffn_gate_rows,
            ffn_gate_cols,
            ffn_up_q4k,
            ffn_up_rows,
            ffn_up_cols,
            ffn_down,
            ffn_down_quant,
            ffn_down_rows,
            ffn_down_cols,
            router_w,
            n_expert,
            n_expert_used,
            gate_all,
            up_all,
            down_all,
            down_quant,
            shared_input_scale,
            shared_gate,
            shared_up,
            shared_down,
            shared_down_quant,
            n_ff,
            n_embd,
        ) = if let Some(moe) = attn.shared_expert_moe.as_ref() {
            if moe.n_embd != metadata.hidden_dim {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} attention MoE n_embd {} != hidden_dim {}",
                    moe.n_embd, metadata.hidden_dim
                )));
            }
            let gate_all = moe.gate_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} gate_exps raw bytes missing"
                ))
            })?;
            let up_all = moe.up_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} up_exps raw bytes missing"
                ))
            })?;
            let down_all = moe.down_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} down_exps raw bytes missing"
                ))
            })?;
            if moe.gate_quant != rnb_loader::GGMLType::Q4_K
                || moe.up_quant != rnb_loader::GGMLType::Q4_K
            {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} sparse gate/up must be Q4_K: gate={:?} up={:?}",
                    moe.gate_quant, moe.up_quant
                )));
            }
            if !matches!(
                moe.down_quant,
                rnb_loader::GGMLType::Q4_K
                    | rnb_loader::GGMLType::Q5_K
                    | rnb_loader::GGMLType::Q6_K
            ) {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} sparse down unsupported quant {:?}",
                    moe.down_quant
                )));
            }
            let shared_gate =
                q4k_weight_bytes_for_mtp_device(&moe.shared_gate, layer_idx, "shared_gate")?;
            let shared_up =
                q4k_weight_bytes_for_mtp_device(&moe.shared_up, layer_idx, "shared_up")?;
            let shared_down = match moe.shared_down.ggml_type {
                rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K => {
                    moe.shared_down.data.as_bytes().ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "MTP device verify layer {layer_idx} shared_down raw bytes missing"
                        ))
                    })?
                }
                other => {
                    return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} shared_down unsupported quant {other:?}"
                )))
                }
            };
            let router_w = moe.router_f32().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} router raw f32 missing"
                ))
            })?;
            let shared_input_scale =
                super::cpu_runtime::kernels::tensor_as_f32_slice(&moe.shared_input_scale);
            (
                &[][..],
                0,
                0,
                &[][..],
                0,
                0,
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                0,
                0,
                router_w,
                moe.n_expert,
                moe.n_expert_used,
                gate_all,
                up_all,
                down_all,
                moe.down_quant,
                shared_input_scale,
                shared_gate,
                shared_up,
                shared_down,
                moe.shared_down.ggml_type,
                moe.n_ff,
                moe.n_embd,
            )
        } else {
            let ffn_gate_q4k =
                q4k_weight_bytes_for_mtp_device(&attn.ffn_gate_weight, layer_idx, "ffn_gate")?;
            let ffn_up_q4k =
                q4k_weight_bytes_for_mtp_device(&attn.ffn_up_weight, layer_idx, "ffn_up")?;
            let (ffn_down, ffn_down_quant) =
                k_quant_weight_bytes_for_mtp_device(&attn.ffn_down_weight, layer_idx, "ffn_down")?;
            (
                ffn_gate_q4k,
                attn.ffn_gate_weight.rows,
                attn.ffn_gate_weight.cols,
                ffn_up_q4k,
                attn.ffn_up_weight.rows,
                attn.ffn_up_weight.cols,
                ffn_down,
                ffn_down_quant,
                attn.ffn_down_weight.rows,
                attn.ffn_down_weight.cols,
                &[][..],
                0,
                0,
                &[][..],
                &[][..],
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                &[][..],
                &[][..],
                &[][..],
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                attn.ffn_gate_weight.rows,
                metadata.hidden_dim,
            )
        };
        let prior_tokens = kv_cache.current_len();
        let (prior_k_bits, prior_v_bits) = if prior_tokens == 0 {
            (Vec::new(), Vec::new())
        } else {
            let (k_bits, v_bits) = kv_cache.get_up_to(layer_idx, prior_tokens);
            (k_bits.to_vec(), v_bits.to_vec())
        };
        let expected_prior_values = prior_tokens.checked_mul(attn.k_weight.rows).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} prior KV size overflow: tokens={prior_tokens} rows={}",
                attn.k_weight.rows
            ))
        })?;
        if prior_k_bits.len() != expected_prior_values
            || prior_v_bits.len() != expected_prior_values
        {
            return Err(crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} prior KV len mismatch: k={} v={} expected={expected_prior_values}",
                prior_k_bits.len(),
                prior_v_bits.len()
            )));
        }
        layers.push(super::cuda_runtime::MtpDeviceVerifyAttentionMoeLayer {
            layer_index: layer_idx,
            attn_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(&attn.attn_norm),
            q_q4k,
            q_quant,
            q_rows: attn.q_weight.rows,
            q_cols: attn.q_weight.cols,
            k_q4k,
            k_quant,
            k_rows: attn.k_weight.rows,
            k_cols: attn.k_weight.cols,
            v_q4k,
            v_quant,
            v_rows: attn.v_weight.rows,
            v_cols: attn.v_weight.cols,
            prior_k_bits,
            prior_v_bits,
            prior_tokens,
            o_q4k,
            o_quant,
            o_rows: attn.o_weight.rows,
            o_cols: attn.o_weight.cols,
            q_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(q_norm),
            k_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(k_norm),
            post_attn_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(post_attn_norm),
            ffn_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(&attn.ffn_norm),
            ffn_gate_q4k,
            ffn_gate_rows,
            ffn_gate_cols,
            ffn_up_q4k,
            ffn_up_rows,
            ffn_up_cols,
            ffn_down,
            ffn_down_quant,
            ffn_down_rows,
            ffn_down_cols,
            router_w,
            n_expert,
            n_expert_used,
            gate_all,
            up_all,
            down_all,
            down_quant,
            shared_input_scale,
            shared_gate,
            shared_up,
            shared_down,
            shared_down_quant,
            n_ff,
            n_embd,
        });
    }
    Ok(layers)
}

#[cfg(all(feature = "cuda", test))]
pub(super) fn build_mtp_device_verify_gdn_moe_layers<'a>(
    weights: &'a super::layer_weights::ModelWeights,
    metadata: &ModelMetadata,
    kv_cache: &'a mut crate::kv_cache::KVCache,
) -> crate::error::Result<Vec<super::cuda_runtime::MtpDeviceVerifyGdnMoeLayer<'a>>> {
    build_mtp_device_verify_gdn_moe_layers_inner(weights, metadata, kv_cache, false)
}

#[cfg(feature = "cuda")]
fn device_verify_sync_delta_state_to_host() -> bool {
    policy::cuda_device_verify_sync_delta_enabled()
}

#[cfg(feature = "cuda")]
fn build_mtp_device_verify_gdn_moe_layers_inner<'a>(
    weights: &'a super::layer_weights::ModelWeights,
    metadata: &ModelMetadata,
    kv_cache: &'a mut crate::kv_cache::KVCache,
    allow_attention_moe_skip: bool,
) -> crate::error::Result<Vec<super::cuda_runtime::MtpDeviceVerifyGdnMoeLayer<'a>>> {
    if allow_attention_moe_skip
        && !weights
            .layers
            .iter()
            .any(|layer| matches!(layer, super::layer_weights::LayerType::GatedDeltaNet(_)))
    {
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            match layer {
                super::layer_weights::LayerType::Attention(_) => {}
                other => {
                    return Err(crate::error::LlmError::Forward(format!(
                        "MTP device verify layer {layer_idx} type {} is not wired into MTP device verify graph",
                        layer_type_name_for_mtp_device(other)
                    )))
                }
            }
        }
        return Ok(Vec::new());
    }

    let d_inner = metadata.ssm_d_inner;
    let d_state = metadata.ssm_d_state;
    let n_group = metadata.ssm_n_group;
    let dt_rank = metadata.ssm_dt_rank;
    let conv_kernel = metadata.ssm_conv_kernel;
    if d_inner == 0 || d_state == 0 || n_group == 0 || dt_rank == 0 || conv_kernel == 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "MTP device verify GDN metadata invalid: d_inner={d_inner} d_state={d_state} n_group={n_group} dt_rank={dt_rank} conv_kernel={conv_kernel}"
        )));
    }
    if d_inner % dt_rank != 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "MTP device verify GDN d_inner must be divisible by dt_rank: d_inner={d_inner} dt_rank={dt_rank}"
        )));
    }
    if kv_cache.ssm_states.len() < weights.layers.len() {
        return Err(crate::error::LlmError::Forward(format!(
            "MTP device verify SSM state table too short: states={} layers={}",
            kv_cache.ssm_states.len(),
            weights.layers.len()
        )));
    }

    let head_v_dim = d_inner / dt_rank;
    let head_k_dim = d_state;
    let num_v_heads = dt_rank;
    let num_k_heads = n_group;
    let conv_channels = d_inner
        .checked_add(
            n_group
                .checked_mul(d_state)
                .and_then(|x| x.checked_mul(2))
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(
                        "MTP device verify GDN conv channel overflow".to_string(),
                    )
                })?,
        )
        .ok_or_else(|| {
            crate::error::LlmError::Forward(
                "MTP device verify GDN conv channel overflow".to_string(),
            )
        })?;
    let mut layers = Vec::new();
    for (layer_idx, (layer, ssm_slot)) in weights
        .layers
        .iter()
        .zip(kv_cache.ssm_states.iter_mut())
        .enumerate()
    {
        let gdn = match layer {
            super::layer_weights::LayerType::GatedDeltaNet(gdn) => gdn,
            super::layer_weights::LayerType::Attention(_) if allow_attention_moe_skip => {
                continue;
            }
            super::layer_weights::LayerType::Attention(attn)
                if attn.shared_expert_moe.is_some() =>
            {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} attention layer is not wired into MTP device verify graph"
                )));
            }
            other => {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} type {} is not wired into MTP device verify graph",
                    layer_type_name_for_mtp_device(other)
                )));
            }
        };
        let state = ssm_slot.as_mut().ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} has Qwen35 GDN weights but no SSM state"
            ))
        })?;
        if state.conv_kernel != conv_kernel || state.conv_channels != conv_channels {
            return Err(crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} conv state shape mismatch: state kernel={} channels={}, metadata kernel={conv_kernel} channels={conv_channels}",
                state.conv_kernel, state.conv_channels
            )));
        }
        if state.delta_state.len() != num_v_heads * head_v_dim * head_k_dim {
            return Err(crate::error::LlmError::Forward(format!(
                "MTP device verify layer {layer_idx} delta state len {} != expected {}",
                state.delta_state.len(),
                num_v_heads * head_v_dim * head_k_dim
            )));
        }

        let (qkv_q4k, qkv_quant) =
            k_quant_weight_bytes_for_mtp_device(&gdn.qkv_weight, layer_idx, "attn_qkv")?;
        let gate_q4k = q4k_weight_bytes_for_mtp_device(&gdn.gate_weight, layer_idx, "attn_gate")?;
        let (alpha_q4k, alpha_f32, alpha_quant) =
            q4k_or_f32_weight_for_mtp_device(&gdn.ssm_alpha, layer_idx, "ssm_alpha")?;
        let (beta_q4k, beta_f32, beta_quant) =
            q4k_or_f32_weight_for_mtp_device(&gdn.ssm_beta, layer_idx, "ssm_beta")?;
        let (ssm_out_q4k, ssm_out_quant) =
            k_quant_weight_bytes_for_mtp_device(&gdn.ssm_out, layer_idx, "ssm_out")?;
        let (
            router_w,
            n_expert,
            n_expert_used,
            gate_all,
            up_all,
            down_all,
            down_quant,
            shared_input_scale,
            shared_gate,
            shared_up,
            shared_down,
            shared_down_quant,
            n_ff,
            n_embd,
            ffn_gate_q4k,
            ffn_gate_rows,
            ffn_gate_cols,
            ffn_up_q4k,
            ffn_up_rows,
            ffn_up_cols,
            ffn_down,
            ffn_down_quant,
            ffn_down_rows,
            ffn_down_cols,
        ) = if let Some(moe) = gdn.shared_expert_moe.as_ref() {
            let gate_all = moe.gate_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} gate_exps raw bytes missing"
                ))
            })?;
            let up_all = moe.up_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} up_exps raw bytes missing"
                ))
            })?;
            let down_all = moe.down_exps_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} down_exps raw bytes missing"
                ))
            })?;
            if moe.gate_quant != rnb_loader::GGMLType::Q4_K
                || moe.up_quant != rnb_loader::GGMLType::Q4_K
            {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} sparse gate/up must be Q4_K: gate={:?} up={:?}",
                    moe.gate_quant, moe.up_quant
                )));
            }
            if !matches!(
                moe.down_quant,
                rnb_loader::GGMLType::Q4_K
                    | rnb_loader::GGMLType::Q5_K
                    | rnb_loader::GGMLType::Q6_K
            ) {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} sparse down unsupported quant {:?}",
                    moe.down_quant
                )));
            }
            let shared_gate =
                q4k_weight_bytes_for_mtp_device(&moe.shared_gate, layer_idx, "shared_gate")?;
            let shared_up =
                q4k_weight_bytes_for_mtp_device(&moe.shared_up, layer_idx, "shared_up")?;
            let shared_down = match moe.shared_down.ggml_type {
                rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K => {
                    moe.shared_down.data.as_bytes().ok_or_else(|| {
                        crate::error::LlmError::Forward(format!(
                            "MTP device verify layer {layer_idx} shared_down raw bytes missing"
                        ))
                    })?
                }
                other => {
                    return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} shared_down unsupported quant {other:?}"
                )))
                }
            };
            let router_w = moe.router_f32().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} router raw f32 missing"
                ))
            })?;
            let shared_input_scale =
                super::cpu_runtime::kernels::tensor_as_f32_slice(&moe.shared_input_scale);
            (
                router_w,
                moe.n_expert,
                moe.n_expert_used,
                gate_all,
                up_all,
                down_all,
                moe.down_quant,
                shared_input_scale,
                shared_gate,
                shared_up,
                shared_down,
                moe.shared_down.ggml_type,
                moe.n_ff,
                moe.n_embd,
                &[][..],
                0,
                0,
                &[][..],
                0,
                0,
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                0,
                0,
            )
        } else {
            let ffn_gate_q4k =
                q4k_weight_bytes_for_mtp_device(&gdn.ffn_gate_weight, layer_idx, "ffn_gate")?;
            let ffn_up_q4k =
                q4k_weight_bytes_for_mtp_device(&gdn.ffn_up_weight, layer_idx, "ffn_up")?;
            let (ffn_down, ffn_down_quant) =
                k_quant_weight_bytes_for_mtp_device(&gdn.ffn_down_weight, layer_idx, "ffn_down")?;
            (
                &[][..],
                0,
                0,
                &[][..],
                &[][..],
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                &[][..],
                &[][..],
                &[][..],
                &[][..],
                rnb_loader::GGMLType::Q4_K,
                gdn.ffn_gate_weight.rows,
                metadata.hidden_dim,
                ffn_gate_q4k,
                gdn.ffn_gate_weight.rows,
                gdn.ffn_gate_weight.cols,
                ffn_up_q4k,
                gdn.ffn_up_weight.rows,
                gdn.ffn_up_weight.cols,
                ffn_down,
                ffn_down_quant,
                gdn.ffn_down_weight.rows,
                gdn.ffn_down_weight.cols,
            )
        };
        let (conv_state, delta_state) = (&state.conv_state, &mut state.delta_state);
        layers.push(super::cuda_runtime::MtpDeviceVerifyGdnMoeLayer {
            layer_index: layer_idx,
            attn_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.attn_norm),
            qkv_q4k,
            qkv_quant,
            qkv_rows: gdn.qkv_weight.rows,
            qkv_cols: gdn.qkv_weight.cols,
            gate_q4k,
            gate_rows: gdn.gate_weight.rows,
            gate_cols: gdn.gate_weight.cols,
            alpha_q4k,
            alpha_f32,
            alpha_quant,
            alpha_rows: gdn.ssm_alpha.rows,
            alpha_cols: gdn.ssm_alpha.cols,
            beta_q4k,
            beta_f32,
            beta_quant,
            beta_rows: gdn.ssm_beta.rows,
            beta_cols: gdn.ssm_beta.cols,
            conv_state,
            conv_kernel: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.ssm_conv1d),
            kernel_size: conv_kernel,
            dt_bias: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.ssm_dt_bias),
            ssm_a: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.ssm_a),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            delta_state,
            sync_delta_state_to_host: device_verify_sync_delta_state_to_host(),
            ssm_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.ssm_norm),
            ssm_out_q4k,
            ssm_out_quant,
            ssm_out_rows: gdn.ssm_out.rows,
            ssm_out_cols: gdn.ssm_out.cols,
            post_attn_norm: super::cpu_runtime::kernels::tensor_as_f32_slice(&gdn.post_attn_norm),
            router_w,
            n_expert,
            n_expert_used,
            gate_all,
            up_all,
            down_all,
            down_quant,
            shared_input_scale,
            shared_gate,
            shared_up,
            shared_down,
            shared_down_quant,
            n_ff,
            n_embd,
            ffn_gate_q4k,
            ffn_gate_rows,
            ffn_gate_cols,
            ffn_up_q4k,
            ffn_up_rows,
            ffn_up_cols,
            ffn_down,
            ffn_down_quant,
            ffn_down_rows,
            ffn_down_cols,
        });
    }
    Ok(layers)
}

#[cfg(feature = "cuda")]
fn q4k_weight_bytes_for_mtp_device<'a>(
    weight: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<&'a [u8]> {
    if weight.ggml_type != rnb_loader::GGMLType::Q4_K {
        return Err(crate::error::LlmError::Forward(format!(
            "MTP device verify layer {layer_idx} {label} must be Q4_K, got {:?}",
            weight.ggml_type
        )));
    }
    weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "MTP device verify layer {layer_idx} {label} raw bytes missing"
        ))
    })
}

#[cfg(feature = "cuda")]
fn k_quant_weight_bytes_for_mtp_device<'a>(
    weight: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], rnb_loader::GGMLType)> {
    if !matches!(
        weight.ggml_type,
        rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K | rnb_loader::GGMLType::Q8_0
    ) {
        return Err(crate::error::LlmError::Forward(format!(
            "MTP device verify layer {layer_idx} {label} must be Q4_K, Q6_K or Q8_0, got {:?}",
            weight.ggml_type
        )));
    }
    let bytes = weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "MTP device verify layer {layer_idx} {label} raw bytes missing"
        ))
    })?;
    Ok((bytes, weight.ggml_type))
}

#[cfg(feature = "cuda")]
fn q4k_or_f32_weight_for_mtp_device<'a>(
    weight: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], &'a [f32], rnb_loader::GGMLType)> {
    match weight.ggml_type {
        rnb_loader::GGMLType::Q4_K => {
            let bytes = weight.data.as_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} {label} raw bytes missing"
                ))
            })?;
            Ok((bytes, &[], weight.ggml_type))
        }
        rnb_loader::GGMLType::F32 => {
            if weight.data.dtype() != rnb_core::tensor::DType::F32 {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} {label} tensor dtype {:?} does not match ggml_type F32",
                    weight.data.dtype()
                )));
            }
            let values = super::cpu_runtime::kernels::tensor_as_f32_slice(&weight.data);
            let expected = weight.rows.checked_mul(weight.cols).ok_or_else(|| {
                crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} {label} shape overflow: rows={} cols={}",
                    weight.rows, weight.cols
                ))
            })?;
            if values.len() != expected {
                return Err(crate::error::LlmError::Forward(format!(
                    "MTP device verify layer {layer_idx} {label} F32 len {} != rows*cols {expected}",
                    values.len()
                )));
            }
            Ok((&[], values, weight.ggml_type))
        }
        other => Err(crate::error::LlmError::Forward(format!(
            "MTP device verify layer {layer_idx} {label} must be F32 or Q4_K, got {other:?}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn layer_type_name_for_mtp_device(t: &super::layer_weights::LayerType) -> &'static str {
    match t {
        super::layer_weights::LayerType::Attention(_) => "Attention",
        super::layer_weights::LayerType::GatedDeltaNet(_) => "GatedDeltaNet",
        super::layer_weights::LayerType::NemotronMamba2(_) => "NemotronMamba2",
        super::layer_weights::LayerType::NemotronMoE(_) => "NemotronMoE",
    }
}

// ---------------------------------------------------------------------------
// mv27-task10b-4c-2b: fullpath layer-raw-weight builder helpers
// ---------------------------------------------------------------------------

/// Convert engine [`LayerType`] -> short label for diagnostics.
#[cfg(feature = "vulkan")]
pub(super) fn layer_type_name(t: &super::layer_weights::LayerType) -> &'static str {
    match t {
        super::layer_weights::LayerType::Attention(_) => "Attention",
        super::layer_weights::LayerType::GatedDeltaNet(_) => "GatedDeltaNet",
        super::layer_weights::LayerType::NemotronMamba2(_) => "NemotronMamba2",
        super::layer_weights::LayerType::NemotronMoE(_) => "NemotronMoE",
    }
}

/// Build per-layer `LayerRawWeights<'_>` + `ModelLayerKind` arrays from
/// `engine.weights` for the GPU fullpath wrapper.
///
/// Lifetimes: result borrows from `&'a ModelWeights`. Caller must keep the
/// Vec alive across the wrapper call (the wrapper consumes it within a single
/// call, so this is automatic for direct callers).
///
/// **Constraints (post 10b-5c)**:
/// - Hybrid `Attention` + `GatedDeltaNet` layer mixes are supported (Qwen3.5
///   0.8B). `NemotronMamba2` and `NemotronMoE` still return Err — those are
///   future-task scope.
/// - `ffn_gate_up_fused` (both Attention and GDN) must be `None`. Fused-FFN
///   models stay unsupported until backend adds a fused-gate-up shader path.
/// - GDN layers must have `shared_expert_moe` = `None`. Qwen3.5 0.8B is dense; the
///   Qwen3.5 35B-A3B GDN-MoE path is out of scope for the dense fullpath PoC.
/// - All quantized weights must map to a backend `QuantType` via
///   `ggml_to_quant`. Unsupported quants (e.g. Q4_0, Q3_K) return Err.
#[cfg(feature = "vulkan")]
pub(super) fn build_fullpath_layer_raw_weights<'a>(
    weights: &'a super::layer_weights::ModelWeights,
    metadata: &ModelMetadata,
) -> crate::error::Result<(
    Vec<super::gpu_runtime::LayerRawWeights<'a>>,
    Vec<super::gpu_runtime::ModelLayerKind>,
)> {
    use super::gpu_runtime;

    let mut layer_raw: Vec<gpu_runtime::LayerRawWeights<'a>> =
        Vec::with_capacity(weights.layers.len());
    let mut layer_kinds: Vec<gpu_runtime::ModelLayerKind> =
        Vec::with_capacity(weights.layers.len());

    for (idx, layer) in weights.layers.iter().enumerate() {
        match layer {
            super::layer_weights::LayerType::Attention(attn) => {
                if attn.ffn_gate_up_fused.is_some() {
                    return Err(crate::error::LlmError::Forward(format!(
                        "build_fullpath_layer_raw_weights: layer {idx} has fused gate_up — \
                         fullpath wrapper requires separate gate + up projections",
                    )));
                }

                // f32 norm slices via cpu_runtime kernels helper.
                let attn_norm = super::cpu_runtime::kernels::tensor_as_f32_slice(&attn.attn_norm);
                let ffn_norm = super::cpu_runtime::kernels::tensor_as_f32_slice(&attn.ffn_norm);
                let q_norm = attn
                    .q_norm
                    .as_ref()
                    .map(super::cpu_runtime::kernels::tensor_as_f32_slice);
                let k_norm = attn
                    .k_norm
                    .as_ref()
                    .map(super::cpu_runtime::kernels::tensor_as_f32_slice);

                let q_proj = quant_weight_to_raw_tuple(&attn.q_weight, idx, "q")?;
                let k_combined = quant_weight_to_raw_tuple_combined(&attn.k_weight, idx, "k")?;
                let v_combined = quant_weight_to_raw_tuple_combined(&attn.v_weight, idx, "v")?;
                let o_proj = quant_weight_to_raw_tuple(&attn.o_weight, idx, "o")?;
                let gate_proj = quant_weight_to_raw_tuple(&attn.ffn_gate_weight, idx, "ffn_gate")?;
                let up_proj = quant_weight_to_raw_tuple(&attn.ffn_up_weight, idx, "ffn_up")?;
                let down_proj = quant_weight_to_raw_tuple(&attn.ffn_down_weight, idx, "ffn_down")?;

                layer_raw.push(gpu_runtime::LayerRawWeights::Attention(
                    gpu_runtime::AttentionRawWeights {
                        attn_norm,
                        q_proj,
                        q_norm,
                        k_proj_combined: k_combined,
                        k_norm,
                        v_proj_combined: v_combined,
                        o_proj,
                        ffn_norm,
                        gate_proj,
                        up_proj,
                        down_proj,
                    },
                ));
                layer_kinds.push(gpu_runtime::ModelLayerKind::Attention);
            }
            super::layer_weights::LayerType::GatedDeltaNet(g) => {
                let gdn_raw = extract_gdn_raw_weights(g, metadata, idx)?;
                layer_raw.push(gpu_runtime::LayerRawWeights::Gdn(gdn_raw));
                layer_kinds.push(gpu_runtime::ModelLayerKind::Recurrent);
            }
            other => {
                return Err(crate::error::LlmError::Forward(format!(
                    "build_fullpath_layer_raw_weights: layer {idx} is LayerType::{} — \
                     fullpath wrapper currently only supports Attention + GDN",
                    layer_type_name(other),
                )));
            }
        }
    }

    Ok((layer_raw, layer_kinds))
}

/// Extract `GdnRawWeights<'a>` from one engine `GdnLayerWeights`.
///
/// Mirrors the Attention branch in `build_fullpath_layer_raw_weights`: f32
/// fields ride `tensor_as_f32_slice` / `f32_weight_to_raw_tuple` (Raw32 upload
/// path), quantized fields go through `quant_weight_to_raw_tuple` (standard
/// Soa cache).
///
/// Rejects:
/// - `shared_expert_moe.is_some()` — Qwen3.5 35B-A3B GDN-MoE is out of scope for the
///   dense fullpath PoC.
/// - `ffn_gate_up_fused.is_some()` — fused gate+up not supported by the
///   wrapper (no fused-gate-up shader path yet).
#[cfg(feature = "vulkan")]
fn extract_gdn_raw_weights<'a>(
    g: &'a super::layer_weights::GdnLayerWeights,
    metadata: &ModelMetadata,
    layer_idx: usize,
) -> crate::error::Result<super::gpu_runtime::GdnRawWeights<'a>> {
    use super::gpu_runtime;

    if g.shared_expert_moe.is_some() {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} GDN layer has \
             shared_expert_moe set — only dense GDN layers supported in fullpath wrapper",
        )));
    }
    if g.ffn_gate_up_fused.is_some() {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} GDN layer has fused \
             gate_up — fullpath wrapper requires separate gate + up projections",
        )));
    }
    if metadata.ssm_n_group == 0 || metadata.ssm_d_state == 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} GDN metadata has invalid \
             ssm_n_group={} ssm_d_state={}",
            metadata.ssm_n_group, metadata.ssm_d_state
        )));
    }

    // f32 raw fields (Raw32 upload path).
    let attn_norm = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.attn_norm);
    let post_attn_norm = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.post_attn_norm);
    let ssm_a = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.ssm_a);
    let ssm_conv1d = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.ssm_conv1d);
    let ssm_dt_bias = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.ssm_dt_bias);
    let ssm_norm = super::cpu_runtime::kernels::tensor_as_f32_slice(&g.ssm_norm);

    // Quantized fields (standard Soa cache path).
    let qkv = quant_weight_to_raw_tuple(&g.qkv_weight, layer_idx, "gdn_qkv")?;
    let gate = quant_weight_to_raw_tuple(&g.gate_weight, layer_idx, "gdn_gate")?;
    let ssm_out = quant_weight_to_raw_tuple(&g.ssm_out, layer_idx, "gdn_ssm_out")?;
    let ffn_gate = quant_weight_to_raw_tuple(&g.ffn_gate_weight, layer_idx, "gdn_ffn_gate")?;
    let ffn_up = quant_weight_to_raw_tuple(&g.ffn_up_weight, layer_idx, "gdn_ffn_up")?;
    let ffn_down = quant_weight_to_raw_tuple(&g.ffn_down_weight, layer_idx, "gdn_ffn_down")?;

    // Qwen3.5 GDN alpha/beta are GGML F32 tensors, not quantized blocks.
    let ssm_alpha = f32_weight_to_raw_tuple(&g.ssm_alpha, layer_idx, "gdn_ssm_alpha")?;
    let ssm_beta = f32_weight_to_raw_tuple(&g.ssm_beta, layer_idx, "gdn_ssm_beta")?;
    let expected_qkv_rows = metadata
        .ssm_d_inner
        .checked_add(
            metadata
                .ssm_n_group
                .checked_mul(metadata.ssm_d_state)
                .and_then(|x| x.checked_mul(2))
                .ok_or_else(|| {
                    crate::error::LlmError::Forward(format!(
                        "build_fullpath_layer_raw_weights: layer {layer_idx} GDN qkv row shape overflow"
                    ))
                })?,
        )
        .ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "build_fullpath_layer_raw_weights: layer {layer_idx} GDN qkv row shape overflow"
            ))
        })?;
    if qkv.1 != expected_qkv_rows || gate.1 != metadata.ssm_d_inner {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} GDN shape mismatch \
             (qkv_rows={} expected={} gate_rows={} ssm_d_inner={})",
            qkv.1, expected_qkv_rows, gate.1, metadata.ssm_d_inner
        )));
    }

    // Field order matches `GdnRawWeights<'a>` declaration in
    // crates/rnb-runtime/.../layer_runtime/fullpath.rs:335-364.
    Ok(gpu_runtime::GdnRawWeights {
        attn_norm,
        qkv,
        gate,
        ssm_alpha,
        ssm_beta,
        ssm_a,
        ssm_conv1d,
        ssm_dt_bias,
        ssm_norm,
        num_k_heads: metadata.ssm_n_group,
        head_k_dim: metadata.ssm_d_state,
        ssm_out,
        post_attn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}

/// Extract `(raw_bytes, rows, cols, QuantType)` from a [`QuantizedWeight`].
#[cfg(feature = "vulkan")]
fn quant_weight_to_raw_tuple<'a>(
    w: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], usize, usize, super::gpu_runtime::Quant)> {
    let bytes = w.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} has no contiguous host bytes",
        ))
    })?;
    let quant = super::gpu_runtime::ggml_to_quant(w.ggml_type).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} ggml_type {:?} \
             not supported by GPU wrapper (must be Q4_K / Q5_K / Q6_K / Q8_0)",
            w.ggml_type,
        ))
    })?;
    Ok((bytes, w.rows, w.cols, quant))
}

/// Extract `(f32_values, rows, cols)` from a GGML F32 [`QuantizedWeight`].
#[cfg(feature = "vulkan")]
fn f32_weight_to_raw_tuple<'a>(
    w: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<(&'a [f32], usize, usize)> {
    if w.ggml_type != rnb_loader::GGMLType::F32 {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} ggml_type {:?} \
             not supported by GDN Raw32 path (must be F32)",
            w.ggml_type,
        )));
    }
    if w.data.dtype() != rnb_core::tensor::DType::F32 {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} tensor dtype {:?} \
             does not match ggml_type F32",
            w.data.dtype(),
        )));
    }
    if w.data.as_bytes().is_none() {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} has no contiguous host f32 values",
        )));
    }
    let values = super::cpu_runtime::kernels::tensor_as_f32_slice(&w.data);
    let expected_len = w.rows.checked_mul(w.cols).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} shape overflow rows={} cols={}",
            w.rows, w.cols,
        ))
    })?;
    if values.len() != expected_len {
        return Err(crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} F32 len {} != rows*cols {}",
            values.len(),
            expected_len,
        )));
    }
    Ok((values, w.rows, w.cols))
}

/// Extract `(raw_bytes, QuantType)` for combined K/V tensors. The wrapper
/// shards them per-kv-head internally using `kv_head_shard_byte_range`.
#[cfg(feature = "vulkan")]
fn quant_weight_to_raw_tuple_combined<'a>(
    w: &'a super::quantized_weight_types::QuantizedWeight,
    layer_idx: usize,
    label: &'static str,
) -> crate::error::Result<(&'a [u8], super::gpu_runtime::Quant)> {
    let bytes = w.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} has no contiguous host bytes",
        ))
    })?;
    let quant = super::gpu_runtime::ggml_to_quant(w.ggml_type).ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "build_fullpath_layer_raw_weights: layer {layer_idx} {label} ggml_type {:?} \
             not supported by GPU wrapper",
            w.ggml_type,
        ))
    })?;
    Ok((bytes, quant))
}
