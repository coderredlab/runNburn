//! GPU prefill executor and slice-window handoff paths.

use super::handoff_plan::take_kv_cache_for_handoff;
use super::*;

/// mv33: Apply per-head q_norm/k_norm RMSNorm if the layer supplies one.
///
/// Returns `Some(buffer)` when the norm tensor exists (caller borrows the
/// new buffer); `None` keeps the existing q/k tensor passthrough so the
/// post_ffw_norm path stays a single allocation when the model has no
/// q_norm/k_norm (Gemma 4 etc.).
#[cfg(feature = "vulkan")]
fn apply_optional_qk_norm_per_head(
    input: &[f32],
    norm_weight: Option<&Tensor>,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    _architecture: ModelArchitecture,
) -> Option<Vec<f32>> {
    let weight = norm_weight?;
    let weight_data = kernels::tensor_as_f32_slice(weight);
    let total = seq_len * num_heads * head_dim;
    debug_assert_eq!(input.len(), total);
    debug_assert_eq!(weight_data.len(), head_dim);
    let mut out = vec![0.0f32; total];
    for t in 0..seq_len {
        for h in 0..num_heads {
            let off = t * num_heads * head_dim + h * head_dim;
            kernels::norm::rms_norm_into(
                &input[off..off + head_dim],
                weight_data,
                eps,
                &mut out[off..off + head_dim],
            );
        }
    }
    Some(out)
}

impl GpuPrefillExecutor {
    pub(in crate::engine) fn for_slice1(metadata: &ModelMetadata) -> Option<Self> {
        let boundary_plan = plan_slice1_boundary(metadata)?;
        Some(Self { boundary_plan })
    }

    #[cfg(test)]
    pub(in crate::engine) fn boundary_plan(&self) -> &Slice1BoundaryPlan {
        &self.boundary_plan
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "vulkan")]
    fn run_window_gpu(
        &self,
        kv_cache: &mut KVCache,
        metadata: &ModelMetadata,
        weights: &ModelWeights,
        mut hidden: Tensor,
        seq_len: usize,
        pos_start: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        rope_theta: f32,
        norm_eps: f32,
        mut scratch: Option<&mut ScratchBuffers>,
        mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
        mut deferred_gdn_flush: Option<&mut DeferredGdnConvStateFlush>,
        mut deferred_attention_kv: Option<&mut DeferredAttentionKvMaterialization>,
    ) -> crate::error::Result<SliceWindowHandoff> {
        for layer_idx in self.boundary_plan.window_layer_range.clone() {
            match &weights.layers[layer_idx] {
                LayerType::GatedDeltaNet(w) => {
                    hidden = forward_gdn_layer_with_gpu(
                        kv_cache,
                        metadata,
                        hidden,
                        w,
                        layer_idx,
                        seq_len,
                        norm_eps,
                        gpu_runtime.as_deref_mut(),
                        deferred_gdn_flush.as_deref_mut(),
                    )?;
                }
                LayerType::Attention(w) => {
                    return if let Some(ref mut scratch) = scratch {
                        self.run_attention_window_gpu_tokenwise(
                            kv_cache,
                            metadata,
                            scratch,
                            w,
                            hidden,
                            seq_len,
                            pos_start,
                            gpu_runtime.as_deref_mut(),
                            deferred_attention_kv.as_deref_mut(),
                        )
                    } else {
                        self.run_attention_window_cpu_fallback(
                            kv_cache,
                            metadata,
                            weights,
                            hidden,
                            seq_len,
                            pos_start,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            kv_dim,
                            rope_theta,
                            norm_eps,
                        )
                    };
                }
                LayerType::NemotronMamba2(w) => {
                    hidden = models::nemotron::mamba::forward_mamba2_layer(
                        kv_cache, metadata, hidden, w, layer_idx, seq_len, norm_eps,
                    )?;
                }
                LayerType::NemotronMoE(w) => {
                    hidden = models::nemotron::moe::forward_moe_layer_for_prefill(
                        metadata,
                        hidden,
                        w,
                        norm_eps,
                        Some(layer_idx),
                    )?;
                }
            }
        }

        Err(crate::error::LlmError::Forward(
            "slice1 window expected attention layer within window range".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn run(
        &self,
        mut kv_cache: KVCache,
        metadata: &ModelMetadata,
        weights: &ModelWeights,
        tokens: &[u32],
        norm_eps: f32,
        #[cfg(feature = "vulkan")] scratch: Option<&mut ScratchBuffers>,
        #[cfg(feature = "vulkan")] mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
    ) -> crate::error::Result<PrefillHandoff> {
        let seq_len = tokens.len();
        let pos_start = kv_cache.current_len();
        // NOTE: kv_cache를 in-place로 수정. clone하면 0.8B 기본 ctx에서 ~786MB 복사 → OOM.
        let num_heads = metadata.num_heads;
        let num_kv_heads = metadata.num_kv_heads;
        let head_dim = metadata.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rope_theta = metadata.rope_theta;
        #[cfg(feature = "vulkan")]
        let mut deferred_gdn_flush = DeferredGdnConvStateFlush::default();
        #[cfg(feature = "vulkan")]
        let mut deferred_attention_kv = DeferredAttentionKvMaterialization::default();

        let mut hidden = apply_embedding_scale(
            weights.token_embd.gather(tokens)?,
            metadata,
            ModelArchitecture::Qwen35,
        );
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        let last_row =
            &hidden_data[(seq_len - 1) * metadata.hidden_dim..seq_len * metadata.hidden_dim];
        emit_layer_trace("prefill-input", usize::MAX, last_row);
        #[cfg(feature = "vulkan")]
        {
            hidden = run_prefill_layers_cpu_range_with_gpu(
                &mut kv_cache,
                metadata,
                weights,
                None,
                hidden,
                self.boundary_plan.cpu_prefix_layer_range.clone(),
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
                gpu_runtime.as_deref_mut(),
                Some(&mut deferred_gdn_flush),
            )?;
        }
        #[cfg(not(feature = "vulkan"))]
        {
            hidden = run_prefill_layers_cpu_range(
                &mut kv_cache,
                metadata,
                ModelArchitecture::Qwen35,
                weights,
                None,
                hidden,
                self.boundary_plan.cpu_prefix_layer_range.clone(),
                seq_len,
                pos_start,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                rope_theta,
                norm_eps,
            )?;
        }

        let handoff = {
            #[cfg(feature = "vulkan")]
            {
                self.run_window_gpu(
                    &mut kv_cache,
                    metadata,
                    weights,
                    hidden,
                    seq_len,
                    pos_start,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                    rope_theta,
                    norm_eps,
                    scratch,
                    gpu_runtime.as_deref_mut(),
                    Some(&mut deferred_gdn_flush),
                    Some(&mut deferred_attention_kv),
                )?
            }
            #[cfg(not(feature = "vulkan"))]
            {
                self.run_attention_window_cpu_fallback(
                    &mut kv_cache,
                    metadata,
                    weights,
                    hidden,
                    seq_len,
                    pos_start,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                    rope_theta,
                    norm_eps,
                )?
            }
        };

        let SliceWindowHandoff {
            hidden_after_window,
            cpu_kv_cache,
            ..
        } = handoff;
        kv_cache = cpu_kv_cache;

        let hidden = Tensor::from_vec(hidden_after_window, &[seq_len, metadata.hidden_dim]);
        #[cfg(feature = "vulkan")]
        let hidden = run_prefill_layers_cpu_range_with_gpu(
            &mut kv_cache,
            metadata,
            weights,
            None,
            hidden,
            self.boundary_plan.window_layer_range.end..metadata.num_layers,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
            gpu_runtime.as_deref_mut(),
            Some(&mut deferred_gdn_flush),
        )?;
        #[cfg(feature = "vulkan")]
        if let Some(vk) = gpu_runtime.as_deref_mut() {
            let mut total_bytes =
                deferred_attention_kv.flush_into_kv_cache_untracked(&mut kv_cache, vk)?;
            total_bytes += deferred_gdn_flush.flush_into_kv_cache_untracked(&mut kv_cache, vk)?;
            backend_runtime::record_batched_materialization_download(vk, total_bytes);
        }
        #[cfg(not(feature = "vulkan"))]
        let hidden = run_prefill_layers_cpu_range(
            &mut kv_cache,
            metadata,
            ModelArchitecture::Qwen35,
            weights,
            None,
            hidden,
            self.boundary_plan.window_layer_range.end..metadata.num_layers,
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )?;

        let logits = finalize_prefill_logits(
            &mut kv_cache,
            metadata,
            ModelArchitecture::Qwen35,
            weights,
            hidden,
            seq_len,
            pos_start,
            norm_eps,
            None,
        )?;
        kv_cache.current_len = pos_start + seq_len;
        Ok(PrefillHandoff {
            logits,
            #[cfg(test)]
            next_pos: pos_start + seq_len,
            cpu_kv_cache: kv_cache,
        })
    }

    #[cfg(feature = "vulkan")]
    pub(in crate::engine) fn run_attention_window_gpu_tokenwise(
        &self,
        kv_cache: &mut KVCache,
        metadata: &ModelMetadata,
        _scratch: &mut ScratchBuffers,
        weights: &AttentionLayerWeights,
        hidden: Tensor,
        seq_len: usize,
        pos_start: usize,
        mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
        mut deferred_attention_kv: Option<&mut DeferredAttentionKvMaterialization>,
    ) -> crate::error::Result<SliceWindowHandoff> {
        let hidden_dim = metadata.hidden_dim;
        let hidden_data = kernels::tensor_as_f32_slice(&hidden);
        let mut hidden_after_window = vec![0.0f32; seq_len * hidden_dim];

        let supported_fast_path = backend_runtime::attention_window_fast_path_supported(weights);

        if supported_fast_path && gpu_runtime.is_some() {
            let attn_norm_data = kernels::tensor_as_f32_slice(&weights.attn_norm);
            let q_dim = metadata.num_heads * metadata.head_dim;
            let kv_dim = metadata.num_kv_heads * metadata.head_dim;
            let ffn_norm = if let Some(ref pan) = weights.post_attn_norm {
                pan
            } else {
                &weights.ffn_norm
            };

            let vk = gpu_runtime.as_deref_mut().unwrap();
            let layer_idx = self.boundary_plan.attention_layer_idx;
            let ffn_norm_data = kernels::tensor_as_f32_slice(ffn_norm);
            let gemma_needs_post_ffw_norm = weights.post_ffw_norm.is_some();
            // mv33: Qwen3+ 류 모델은 q_norm/k_norm 을 attention 진입 전에 head-별
            // rmsnorm 적용해야 한다. fully-vulkan fast path 에는 그 단계가 없으니
            // post_ffw_norm path 를 재사용하면서 hybrid (CPU q/k gemv → CPU
            // qk_norm → vulkan attention) 로 우회한다.
            let needs_qk_norm_path = backend_runtime::attention_window_needs_qk_norm_path(weights);
            let needs_hybrid_path = gemma_needs_post_ffw_norm || needs_qk_norm_path;

            if !needs_hybrid_path {
                let completed = backend_runtime::try_attention_ffn_window_fast_path(
                    vk,
                    layer_idx,
                    hidden_data,
                    attn_norm_data,
                    ffn_norm_data,
                    &weights.q_weight,
                    &weights.k_weight,
                    &weights.v_weight,
                    &weights.o_weight,
                    &weights.ffn_gate_weight,
                    &weights.ffn_up_weight,
                    &weights.ffn_down_weight,
                    seq_len,
                    q_dim,
                    kv_dim,
                    hidden_dim,
                    pos_start,
                    metadata.num_heads,
                    metadata.num_kv_heads,
                    metadata.head_dim,
                    metadata.norm_eps,
                    &mut hidden_after_window,
                )?;
                if !completed {
                    return Err(crate::error::LlmError::Forward(
                        "slice1 attention window unexpectedly unsupported".into(),
                    ));
                }
            } else {
                // mv36: pre-attention RMSNorm 도 CPU strict 로 swap. vulkan
                // rms_norm_window vs CPU strict apply_model_norm fp accumulation
                // 차이 → 입구 normed input 이 drift 시작점이 될 수 있어 차단.
                let _ = attn_norm_data;
                let hidden_tensor_local = Tensor::from_slice(hidden_data, &[seq_len, hidden_dim]);
                let normed_tensor = crate::engine::norm::apply_model_norm(
                    &hidden_tensor_local,
                    &weights.attn_norm,
                    metadata.norm_eps,
                    ModelArchitecture::Qwen35,
                )
                .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?;
                // mv38 D: hybrid hidden + normed (layer 3, t=last 첫 8 elem).
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                    && self.boundary_plan.attention_layer_idx == 3
                {
                    let nd = kernels::tensor_as_f32_slice(&normed_tensor);
                    let last_off = (seq_len - 1) * hidden_dim;
                    let preview_n = hidden_dim.min(8);
                    let h_preview: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", hidden_data[last_off + d]))
                        .collect();
                    let n_preview: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", nd[last_off + d]))
                        .collect();
                    eprintln!(
                        "[mv38:hidden_in] hybrid layer={} hidden[0..{}] = [{}]",
                        self.boundary_plan.attention_layer_idx,
                        preview_n,
                        h_preview.join(", ")
                    );
                    eprintln!(
                        "[mv38:normed] hybrid layer={} normed[0..{}] = [{}]",
                        self.boundary_plan.attention_layer_idx,
                        preview_n,
                        n_preview.join(", ")
                    );
                }
                let q_all_full = weights.q_weight.gemv(&normed_tensor)?;
                let k_all_cpu = weights.k_weight.gemv(&normed_tensor)?;
                let v_all_cpu = weights.v_weight.gemv(&normed_tensor)?;
                // mv38 E: gated attention 분리 — Qwen3.5 0.8B 의 q_weight.rows
                // 가 num_heads * head_dim * 2 인 경우 (has_gated_attn=true), 첫
                // 절반은 query, 둘째 절반은 attention gate. baseline 의
                // project_prefill_attention 이 같은 분리. mv33 hybrid path 가
                // 누락해서 layer 3 hidden_after 가 garbage.
                let q_out_dim = q_all_full.shape().last().copied().unwrap_or(0);
                let q_dim_split = metadata.num_heads * metadata.head_dim;
                let has_gated = q_out_dim == q_dim_split * 2;
                let (q_all_cpu, attn_gate_vec) = if has_gated {
                    let q_full_data = kernels::tensor_as_f32_slice(&q_all_full);
                    let head_dim = metadata.head_dim;
                    let mut q_vec = vec![0.0f32; seq_len * q_dim_split];
                    let mut gate_vec = vec![0.0f32; seq_len * q_dim_split];
                    for t in 0..seq_len {
                        for h in 0..metadata.num_heads {
                            let src_off = t * q_out_dim + h * head_dim * 2;
                            let dst_off = t * q_dim_split + h * head_dim;
                            q_vec[dst_off..dst_off + head_dim]
                                .copy_from_slice(&q_full_data[src_off..src_off + head_dim]);
                            gate_vec[dst_off..dst_off + head_dim].copy_from_slice(
                                &q_full_data[src_off + head_dim..src_off + head_dim * 2],
                            );
                        }
                    }
                    (
                        Tensor::from_vec(q_vec, &[seq_len, q_dim_split]),
                        Some(gate_vec),
                    )
                } else {
                    (q_all_full, None)
                };
                // mv33: q_norm/k_norm head-별 rmsnorm 을 CPU 에서 적용해서
                // vulkan attention 에 token-identical 입력을 넘긴다. 미적용 모델은
                // 그대로 통과.
                // mv38 D / mv39: hybrid q before q_norm (after gemv). sum/max_abs 강화.
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                    && self.boundary_plan.attention_layer_idx == 3
                {
                    let q_data_pre = kernels::tensor_as_f32_slice(&q_all_cpu);
                    let q_dim_actual = q_data_pre.len() / seq_len;
                    let q_last_off = (seq_len - 1) * q_dim_actual;
                    let preview_n = q_dim_actual.min(8);
                    let q_t0: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_data_pre[d]))
                        .collect();
                    let q_last: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_data_pre[q_last_off + d]))
                        .collect();
                    let q_sum: f64 = q_data_pre.iter().map(|&v| v as f64).sum();
                    let q_max_abs = q_data_pre.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let q_xor: u64 = q_data_pre
                        .iter()
                        .map(|&v| v.to_bits() as u64)
                        .reduce(|a, b| a ^ b)
                        .unwrap_or(0);
                    eprintln!(
                        "[mv39:q_pre_norm] hybrid layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                        self.boundary_plan.attention_layer_idx, preview_n, q_t0.join(", "), preview_n, q_last.join(", "),
                        q_sum, q_max_abs, q_xor, q_data_pre.len(),
                    );
                }
                let q_all_normed = apply_optional_qk_norm_per_head(
                    kernels::tensor_as_f32_slice(&q_all_cpu),
                    weights.q_norm.as_ref(),
                    seq_len,
                    metadata.num_heads,
                    metadata.head_dim,
                    metadata.norm_eps,
                    ModelArchitecture::Qwen35,
                );
                let k_all_normed = apply_optional_qk_norm_per_head(
                    kernels::tensor_as_f32_slice(&k_all_cpu),
                    weights.k_norm.as_ref(),
                    seq_len,
                    metadata.num_kv_heads,
                    metadata.head_dim,
                    metadata.norm_eps,
                    ModelArchitecture::Qwen35,
                );
                // mv33: q_norm 후 RoPE 적용 (Qwen3+ 표준 순서). vulkan
                // attention 은 RoPE 처리 없으니 token-identical 위해 CPU 에서
                // 적용. apply_decode_rope 가 metadata/arch 기반 자동 dispatch
                // (partial / proportional / NEOX / freq_factors).
                let q_dim = metadata.num_heads * metadata.head_dim;
                let mut q_with_rope: Vec<f32> = q_all_normed
                    .as_deref()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| kernels::tensor_as_f32_slice(&q_all_cpu).to_vec());
                // mv38 D / mv39: hybrid q before RoPE (after q_norm). sum/max_abs 강화.
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                    && self.boundary_plan.attention_layer_idx == 3
                {
                    let q_last_off = (seq_len - 1) * q_dim;
                    let preview_n = q_dim.min(8);
                    let q_t0: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_with_rope[d]))
                        .collect();
                    let q_last: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_with_rope[q_last_off + d]))
                        .collect();
                    let q_sum: f64 = q_with_rope.iter().map(|&v| v as f64).sum();
                    let q_max_abs = q_with_rope.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let q_xor: u64 = q_with_rope
                        .iter()
                        .map(|&v| v.to_bits() as u64)
                        .reduce(|a, b| a ^ b)
                        .unwrap_or(0);
                    eprintln!(
                        "[mv39:q_pre_rope] hybrid layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                        self.boundary_plan.attention_layer_idx, preview_n, q_t0.join(", "), preview_n, q_last.join(", "),
                        q_sum, q_max_abs, q_xor, q_with_rope.len(),
                    );
                }
                let mut k_with_rope: Vec<f32> = k_all_normed
                    .as_deref()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| kernels::tensor_as_f32_slice(&k_all_cpu).to_vec());
                for t in 0..seq_len {
                    let q_off = t * q_dim;
                    let k_off = t * kv_dim;
                    super::super::decode_attention_rope::apply_decode_rope(
                        metadata,
                        ModelArchitecture::Qwen35,
                        None,
                        self.boundary_plan.attention_layer_idx,
                        pos_start + t,
                        metadata.head_dim,
                        q_dim,
                        kv_dim,
                        false,
                        &mut q_with_rope[q_off..q_off + q_dim],
                        &mut k_with_rope[k_off..k_off + kv_dim],
                    );
                }
                let q_all_cpu_data: &[f32] = &q_with_rope;
                let k_all_cpu_data: &[f32] = &k_with_rope;
                let v_all_cpu_data = kernels::tensor_as_f32_slice(&v_all_cpu);
                // mv38 D / mv39: hybrid q trace (sum/max_abs 강화).
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                    && self.boundary_plan.attention_layer_idx == 3
                {
                    let q_last_off = (seq_len - 1) * q_dim;
                    let preview_n = q_dim.min(8);
                    let q_preview_t0: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_all_cpu_data[d]))
                        .collect();
                    let q_preview_tl: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", q_all_cpu_data[q_last_off + d]))
                        .collect();
                    let q_sum: f64 = q_all_cpu_data.iter().map(|&v| v as f64).sum();
                    let q_max_abs = q_all_cpu_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let q_xor: u64 = q_all_cpu_data
                        .iter()
                        .map(|&v| v.to_bits() as u64)
                        .reduce(|a, b| a ^ b)
                        .unwrap_or(0);
                    eprintln!(
                        "[mv39:q_trace] hybrid layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                        self.boundary_plan.attention_layer_idx, preview_n, q_preview_t0.join(", "), preview_n, q_preview_tl.join(", "),
                        q_sum, q_max_abs, q_xor, q_all_cpu_data.len(),
                    );
                }
                // mv36: vulkan attention dispatch 만 keep (mv35 측정 1-ULP),
                // o_proj + FFN 은 CPU strict path (apply_prefill_attention_output
                // + forward_prefill_ffn) 로 처리해 token-identical 시도. hybrid
                // path 의 vulkan o_proj/FFN 단계가 drift origin 인지 fix 로 검증.
                let _ = ffn_norm_data;
                let mut attention_output =
                    backend_runtime::run_attention_window_attention_only_for_layer(
                        vk,
                        layer_idx,
                        q_all_cpu_data,
                        k_all_cpu_data,
                        v_all_cpu_data,
                        seq_len,
                        pos_start,
                        metadata.num_heads,
                        metadata.num_kv_heads,
                        metadata.head_dim,
                        kv_dim,
                    )?;
                let q_dim_local = metadata.num_heads * metadata.head_dim;
                // mv38 E: gated attention 의 sigmoid(gate) elementwise 적용.
                // baseline 의 attention_compute.rs line 307-310 와 동일.
                if let Some(ref gate_vec) = attn_gate_vec {
                    debug_assert_eq!(gate_vec.len(), attention_output.len());
                    for i in 0..attention_output.len() {
                        let g = gate_vec[i];
                        let sig = 1.0f32 / (1.0f32 + (-g).exp());
                        attention_output[i] *= sig;
                    }
                }
                // mv38 D / mv39: hybrid attn_out trace (sum/max_abs 강화).
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                    && self.boundary_plan.attention_layer_idx == 3
                {
                    let last_off = (seq_len - 1) * q_dim_local;
                    let preview_n = q_dim_local.min(8);
                    let preview_t0: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", attention_output[d]))
                        .collect();
                    let preview_tl: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", attention_output[last_off + d]))
                        .collect();
                    let attn_sum: f64 = attention_output.iter().map(|&v| v as f64).sum();
                    let attn_max_abs = attention_output.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let attn_xor: u64 = attention_output
                        .iter()
                        .map(|&v| v.to_bits() as u64)
                        .reduce(|a, b| a ^ b)
                        .unwrap_or(0);
                    eprintln!(
                        "[mv39:attn_out_trace] hybrid layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                        self.boundary_plan.attention_layer_idx, preview_n, preview_t0.join(", "), preview_n, preview_tl.join(", "),
                        attn_sum, attn_max_abs, attn_xor, attention_output.len(),
                    );

                    // mv39: 같은 q/k/v 으로 CPU strict attention 호출 (첫 chunk 만,
                    // pos_start=0 이라 KV cache prefix 없음). vulkan output 과
                    // element-wise diff 측정 → attention shader 자체의 fp drift isolate.
                    if pos_start == 0 {
                        let q_tensor = Tensor::from_slice(q_all_cpu_data, &[seq_len, q_dim_local]);
                        let k_tensor = Tensor::from_slice(k_all_cpu_data, &[seq_len, kv_dim]);
                        let v_tensor = Tensor::from_slice(v_all_cpu_data, &[seq_len, kv_dim]);
                        let scale = 1.0_f32 / (metadata.head_dim as f32).sqrt();
                        if let Ok(cpu_attn) =
                            kernels::attention::attention_with_scale_window_and_softcap(
                                &q_tensor,
                                &k_tensor,
                                &v_tensor,
                                metadata.num_heads,
                                metadata.num_kv_heads,
                                metadata.head_dim,
                                scale,
                                None,
                                None,
                            )
                        {
                            let cpu_data = kernels::tensor_as_f32_slice(&cpu_attn);
                            // gated attention: CPU 결과에도 sigmoid(gate) 적용해야 vulkan과 같은 비교
                            let cpu_gated: Vec<f32> = if let Some(ref gate_vec) = attn_gate_vec {
                                cpu_data
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &x)| {
                                        let g = gate_vec[i];
                                        let sig = 1.0_f32 / (1.0_f32 + (-g).exp());
                                        x * sig
                                    })
                                    .collect()
                            } else {
                                cpu_data.to_vec()
                            };
                            let mut max_abs_diff: f32 = 0.0;
                            let mut max_ulp_diff: u32 = 0;
                            let mut max_diff_idx: usize = 0;
                            let mut total_abs_diff: f64 = 0.0;
                            let mut nonzero: usize = 0;
                            for i in 0..attention_output.len() {
                                let v_diff = (attention_output[i] - cpu_gated[i]).abs();
                                if v_diff > 0.0 {
                                    nonzero += 1;
                                }
                                total_abs_diff += v_diff as f64;
                                if v_diff > max_abs_diff {
                                    max_abs_diff = v_diff;
                                    max_diff_idx = i;
                                }
                                let a_bits = attention_output[i].to_bits();
                                let b_bits = cpu_gated[i].to_bits();
                                let bit_diff = a_bits.max(b_bits) - a_bits.min(b_bits);
                                if bit_diff > max_ulp_diff {
                                    max_ulp_diff = bit_diff;
                                }
                            }
                            let max_diff_t = max_diff_idx / q_dim_local;
                            let max_diff_h = (max_diff_idx % q_dim_local) / metadata.head_dim;
                            let max_diff_d = (max_diff_idx % q_dim_local) % metadata.head_dim;
                            eprintln!(
                                "[mv39:attn_vulkan_vs_cpu] layer={} max_abs_diff={:.6e} max_ulp={} total_abs_diff={:.6e} nonzero={}/{} worst_at(t={},h={},d={})=vulkan={:.6e},cpu={:.6e}",
                                self.boundary_plan.attention_layer_idx, max_abs_diff, max_ulp_diff,
                                total_abs_diff, nonzero, attention_output.len(),
                                max_diff_t, max_diff_h, max_diff_d,
                                attention_output[max_diff_idx], cpu_gated[max_diff_idx],
                            );
                        }
                    }
                }
                let attn_out_tensor = Tensor::from_vec(attention_output, &[seq_len, q_dim_local]);
                let hidden_residual = Tensor::from_slice(hidden_data, &[seq_len, hidden_dim]);
                let after_attn =
                    crate::engine::forward::attention_output::apply_prefill_attention_output(
                        metadata,
                        ModelArchitecture::Qwen35,
                        crate::engine::models::gemma::GemmaRuntimeFlavor::Generic,
                        hidden_residual,
                        weights,
                        &attn_out_tensor,
                        self.boundary_plan.attention_layer_idx,
                        seq_len,
                        metadata.norm_eps,
                    )?;
                let after_ffn = crate::engine::forward::ffn::forward_prefill_ffn(
                    metadata,
                    ModelArchitecture::Qwen35,
                    after_attn,
                    weights,
                    self.boundary_plan.attention_layer_idx,
                    seq_len,
                    metadata.norm_eps,
                )?;
                hidden_after_window.copy_from_slice(kernels::tensor_as_f32_slice(&after_ffn));
                // mv38 D / mv39: hybrid path layer trace.
                if std::env::var("RNB_VULKAN_LAYER_TRACE")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                {
                    let last_off = (seq_len - 1) * hidden_dim;
                    let preview_n = hidden_dim.min(8);
                    let preview_t0: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", hidden_after_window[d]))
                        .collect();
                    let preview_tl: Vec<String> = (0..preview_n)
                        .map(|d| format!("{:.6}", hidden_after_window[last_off + d]))
                        .collect();
                    let total = hidden_after_window.len();
                    let sum: f64 = hidden_after_window.iter().map(|&v| v as f64).sum();
                    let max_abs = hidden_after_window
                        .iter()
                        .fold(0.0f32, |m, &v| m.max(v.abs()));
                    let bits_xor: u64 = hidden_after_window
                        .iter()
                        .map(|&v| v.to_bits() as u64)
                        .reduce(|a, b| a ^ b)
                        .unwrap_or(0);
                    eprintln!(
                        "[mv39:layer_trace] hybrid_attn_layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                        self.boundary_plan.attention_layer_idx,
                        preview_n,
                        preview_t0.join(", "),
                        preview_n,
                        preview_tl.join(", "),
                        sum,
                        max_abs,
                        bits_xor,
                        total,
                    );
                }
            }
            if let Some(tracker) = deferred_attention_kv.as_deref_mut() {
                tracker.mark_touched(
                    self.boundary_plan.attention_layer_idx,
                    pos_start,
                    seq_len,
                    metadata.num_kv_heads,
                    metadata.head_dim,
                );
            } else {
                materialize_attention_kv_into_cache_untracked(
                    kv_cache,
                    vk,
                    self.boundary_plan.attention_layer_idx,
                    metadata.num_kv_heads,
                    pos_start + seq_len,
                    metadata.head_dim,
                    kv_dim,
                )?;
            }

            return Ok(SliceWindowHandoff {
                hidden_after_window,
                #[cfg(test)]
                next_layer_idx: self.boundary_plan.window_layer_range.end,
                #[cfg(test)]
                next_pos: pos_start + seq_len,
                cpu_kv_cache: take_kv_cache_for_handoff(metadata, kv_cache),
            });
        }

        let hidden_tensor = Tensor::from_slice(hidden_data, &[seq_len, hidden_dim]);
        let hidden_tensor = forward_attention_layer(
            kv_cache,
            metadata,
            ModelArchitecture::Qwen35,
            hidden_tensor,
            weights,
            None,
            self.boundary_plan.attention_layer_idx,
            seq_len,
            pos_start,
            metadata.num_heads,
            metadata.num_kv_heads,
            metadata.head_dim,
            metadata.num_kv_heads * metadata.head_dim,
            metadata.rope_theta,
            metadata.norm_eps,
        )?;
        hidden_after_window.copy_from_slice(kernels::tensor_as_f32_slice(&hidden_tensor));

        Ok(SliceWindowHandoff {
            hidden_after_window,
            #[cfg(test)]
            next_layer_idx: self.boundary_plan.window_layer_range.end,
            #[cfg(test)]
            next_pos: pos_start + seq_len,
            cpu_kv_cache: take_kv_cache_for_handoff(metadata, kv_cache),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn run_attention_window_cpu_fallback(
        &self,
        kv_cache: &mut KVCache,
        metadata: &ModelMetadata,
        weights: &ModelWeights,
        hidden: Tensor,
        seq_len: usize,
        pos_start: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        rope_theta: f32,
        norm_eps: f32,
    ) -> crate::error::Result<SliceWindowHandoff> {
        let hidden = run_prefill_layers_cpu_range(
            kv_cache,
            metadata,
            ModelArchitecture::Qwen35,
            weights,
            None,
            hidden,
            self.boundary_plan.window_layer_range.clone(),
            seq_len,
            pos_start,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            rope_theta,
            norm_eps,
        )?;

        let hidden_after_window = kernels::tensor_as_f32_slice(&hidden).to_vec();
        Ok(SliceWindowHandoff {
            hidden_after_window,
            #[cfg(test)]
            next_layer_idx: self.boundary_plan.window_layer_range.end,
            #[cfg(test)]
            next_pos: pos_start + seq_len,
            cpu_kv_cache: take_kv_cache_for_handoff(metadata, kv_cache),
        })
    }
}
