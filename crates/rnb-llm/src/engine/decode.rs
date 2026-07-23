//! Zero-alloc decode helpers (seq_len=1).
//!
//! Moved from `engine.rs`. Re-uses the engine-module-private types and helpers.

use super::*;
// =============================================================================
// Zero-alloc decode helpers (seq_len=1)
// =============================================================================

#[cfg(feature = "cuda")]
fn cu71_f32_bit_hash(data: &[f32]) -> u64 {
    let mut bit_hash = 0xcbf29ce484222325_u64;
    for value in data {
        bit_hash ^= value.to_bits() as u64;
        bit_hash = bit_hash.wrapping_mul(0x100000001b3);
    }
    bit_hash
}

#[cfg(feature = "cuda")]
fn cu71_quantized_weight_identity(weight: &QuantizedWeight) -> u64 {
    weight
        .data
        .as_bytes()
        .map(|bytes| bytes.as_ptr() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn cu71_long_kv_split_preferred(head_dim: usize, kv_len: usize) -> bool {
    head_dim == 512
        && kv_len >= 256
        && crate::engine::tuning_runtime::decode_attention_hd512_split_enabled()
}

/// Attention layer decode (seq_len=1). Operates in-place on scratch buffers.
pub(super) fn decode_attention_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    scratch: &mut ScratchBuffers,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    pos: usize,
    source_hidden: Option<&[f32]>,
    prev_layer_hidden: Option<&[f32]>,
    ple_fusion: Option<(&GemmaPerLayerBase, &GemmaPerLayerWeights)>,
    ple_input_device_offset: Option<usize>,
    cu72_trace: Option<&mut super::decode_layer_graph::Cu72HiddenPersistenceTrace>,
    #[cfg(feature = "vulkan")] mut vulkan_backend: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<bool> {
    decode_attention_layer_with_rope_pos(
        kv_cache,
        metadata,
        architecture,
        scratch,
        w,
        rope_freqs,
        layer_idx,
        pos,
        pos,
        source_hidden,
        prev_layer_hidden,
        ple_fusion,
        ple_input_device_offset,
        cu72_trace,
        #[cfg(feature = "vulkan")]
        vulkan_backend.as_deref_mut(),
    )
}

fn attn_chain_core_eligible(
    w: &AttentionLayerWeights,
    has_gated_attn: bool,
    owns_kv: bool,
    rope_pos: usize,
    cache_pos: usize,
    gemma4_reuse_q_only: bool,
) -> bool {
    has_gated_attn
        && !gemma4_reuse_q_only
        && owns_kv
        && rope_pos == cache_pos
        && w.q_norm.is_some()
        && w.k_norm.is_some()
        && w.q_bias.is_none()
        && w.k_bias.is_none()
        && w.v_bias.is_none()
}

#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
pub(in crate::engine) fn qwen_attn_moe_chain_eligible(
    w: &AttentionLayerWeights,
    has_gated_attn: bool,
    owns_kv: bool,
    rope_pos: usize,
    cache_pos: usize,
    gemma4_reuse_q_only: bool,
    qwen_moe_chain_env: bool,
) -> bool {
    qwen_moe_chain_env
        && attn_chain_core_eligible(
            w,
            has_gated_attn,
            owns_kv,
            rope_pos,
            cache_pos,
            gemma4_reuse_q_only,
        )
        && w.shared_expert_moe.is_some()
        && w.ffn_gate_up_fused.is_none()
}

// attention carrier 진입 조건. decode loop 가 호출 전 판정 가능하게 추출.
// pm17 carrier 의 12조건(weight shape + KV ownership + rope/cache pos 정렬)만
// 그대로 옮긴 동작 불변 함수. env/quant/PLE precondition 은 여기 넣지 않는다
// (PLE precondition 은 1.4 에서 loop 레벨에서 AND). caller(아래 가드 + 1.4 loop)가
// cfg 없는 attn carrier 블록과 동일하게 모든 빌드에서 호출되므로 함수도 cfg 없음
// (metal_attn_layer_into_if_supported 가 non-metal 에서도 컴파일됨).
pub(in crate::engine) fn attn_carrier_eligible(
    w: &AttentionLayerWeights,
    has_gated_attn: bool,
    owns_kv: bool,
    rope_pos: usize,
    cache_pos: usize,
    gemma4_reuse_q_only: bool,
) -> bool {
    attn_chain_core_eligible(
        w,
        has_gated_attn,
        owns_kv,
        rope_pos,
        cache_pos,
        gemma4_reuse_q_only,
    ) && w.moe.is_none()
        && w.shared_expert_moe.is_none()
        && w.ffn_gate_up_fused.is_none()
        && w.post_ffw_norm.is_none()
}

/// Attention layer decode with separate cache and RoPE positions.
///
/// MTP NextN draft blocks can intentionally skip retaining some MTP KV rows
/// while still advancing the absolute target position. In that case the KV
/// write index is dense (`cache_pos`) but the newly encoded Q/K must use the
/// target-side absolute RoPE position (`rope_pos`).
pub(super) fn decode_attention_layer_with_rope_pos(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    scratch: &mut ScratchBuffers,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    cache_pos: usize,
    rope_pos: usize,
    source_hidden: Option<&[f32]>,
    prev_layer_hidden: Option<&[f32]>,
    ple_fusion: Option<(&GemmaPerLayerBase, &GemmaPerLayerWeights)>,
    ple_input_device_offset: Option<usize>,
    cu72_trace: Option<&mut super::decode_layer_graph::Cu72HiddenPersistenceTrace>,
    #[cfg(feature = "vulkan")] mut vulkan_backend: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<bool> {
    let prof_level = super::policy::profiling_level();
    let profiling = prof_level >= 1;
    let verbose = prof_level >= 2;
    macro_rules! prof {
        ($label:expr, $t:expr) => {
            if profiling && (verbose || layer_idx == 3) {
                eprintln!(
                    "  [DEC-ATN L{}] {:20} {:.1}ms",
                    layer_idx,
                    $label,
                    $t.elapsed().as_micros() as f64 / 1000.0
                );
            }
        };
    }

    let hidden_dim = metadata.hidden_dim;
    #[cfg(not(feature = "cuda"))]
    let _ = (ple_fusion, ple_input_device_offset, cu72_trace);
    let gemma_runtime_flavor = if matches!(architecture, ModelArchitecture::Gemma4)
        && metadata.num_layers == 35
        && metadata.hidden_dim == 1536
        && metadata.num_heads == 8
        && metadata.num_kv_heads == 1
        && metadata.head_dim == 512
        && metadata.embedding_length_per_layer_input == 256
    {
        GemmaRuntimeFlavor::Gemma4E2BIt
    } else {
        GemmaRuntimeFlavor::Generic
    };
    let kv_source_layer = shared_kv_source_layer(metadata, architecture, layer_idx);
    let kv_cache_layer = kv_source_layer.unwrap_or(layer_idx);
    let owns_kv = kv_source_layer.is_none();
    let gemma4_reuse_q_only = matches!(architecture, ModelArchitecture::Gemma4) && !owns_kv;
    let layer_kv_override = metadata
        .head_count_kv_per_layer
        .as_ref()
        .and_then(|v| v.get(layer_idx).copied());
    let layout = if gemma4_reuse_q_only {
        resolve_attention_layout_gemma4_reuse(metadata, w, layer_kv_override)?
    } else {
        resolve_attention_layout(metadata, w, layer_kv_override)?
    };
    let head_dim = layout.head_dim;
    let kv_dim = layout.kv_dim;
    let q_dim = layout.q_dim;
    let has_gated_attn = layout.has_gated_attn;
    let num_heads_layout = layout.num_heads;
    let num_kv_heads_layout = layout.num_kv_heads;
    let norm_eps = metadata.norm_eps;

    // pm14: attention layer 전체(attention + FFN) device-resident carrier
    // (RNB_METAL_ATTN_LAYER=1). 표준 Qwen3 한정 — norm→q/k/v GEMV→q/k norm→rope→
    // kv_append→attn→o→residual→ffn_norm→gate/up→silu→down→residual 을 device
    // 단일 command buffer 로. 미지원(gemma/gated/bias/non-mrope/MTP/shared-kv/
    // non-Q4K/MoE/fused-ffn/post-ffw-norm)은 false → 기존 host 경로.
    let attn_carrier_done = {
        // pm17: carrier 는 9B gated attention(q=[query|gate] 인터리브) + partial RoPE
        // (인접페어, production) 전용. carrier_rope_dim(=resolve_rope_params, env 무관)을
        // n_rot 으로 전달. has_gated_attn 요구(non-gated 는 split 미지원이라 host fallback).
        let (carrier_rope_dim, carrier_rope_theta, _) =
            resolve_rope_params(metadata, architecture, layer_idx, head_dim);
        if !kv_cache.layer_uses_kvarn(kv_cache_layer)
            && attn_carrier_eligible(
                w,
                has_gated_attn,
                owns_kv,
                rope_pos,
                cache_pos,
                gemma4_reuse_q_only,
            )
        {
            let norm_weight = kernels::tensor_as_f32_slice(&w.attn_norm);
            let q_norm_weight = kernels::tensor_as_f32_slice(w.q_norm.as_ref().unwrap());
            let k_norm_weight = kernels::tensor_as_f32_slice(w.k_norm.as_ref().unwrap());
            let ffn_norm_weight = kernels::tensor_as_f32_slice(&w.ffn_norm);
            let ffn_dim = w.ffn_gate_weight.rows;
            let carrier_q_out_dim = w.q_weight.rows; // gated: q_dim*2 ([query|gate] 인터리브)
            let (prior_k, prior_v) = kv_cache.get_up_to(kv_cache_layer, cache_pos);
            let capacity = kv_cache.max_seq_len;
            let scale = (head_dim as f32).sqrt().recip();
            backend_runtime::metal_attn_layer_into_if_supported(
                layer_idx,
                &mut scratch.hidden[..hidden_dim],
                norm_weight,
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                q_norm_weight,
                k_norm_weight,
                &w.o_weight,
                ffn_norm_weight,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
                prior_k,
                prior_v,
                cache_pos,
                hidden_dim,
                q_dim,
                carrier_q_out_dim,
                kv_dim,
                head_dim,
                num_heads_layout,
                num_kv_heads_layout,
                carrier_rope_dim,
                capacity,
                ffn_dim,
                norm_eps,
                carrier_rope_theta,
                scale,
            )?
        } else {
            false
        }
    };
    if attn_carrier_done {
        // carrier 가 attention + FFN 전체 완료 — host FFN skip.
        emit_mtp_finite_trace(
            "decode-attn",
            layer_idx,
            "after_ffn",
            &scratch.hidden[..hidden_dim],
        );
        return Ok(false);
    }

    // 1. Attention norm
    let t0 = std::time::Instant::now();
    let attn_norm_data = kernels::tensor_as_f32_slice(&w.attn_norm);
    // cu41 Phase 1 step 5: device-resident hidden chain 의 첫 op — cuda RMS norm
    // carrier (env opt-in). 다음 step 에서 device input QKV 와 chain.
    let unit_offset_attn_norm = use_gemma_block_semantics(architecture)
        && (super::policy::gemma_unit_offset_attn_ffn_norm_enabled()
            || super::policy::gemma_unit_offset_norm_enabled()
            || super::policy::gemma_unit_offset_main_norm_enabled());
    // cu58 step 3 (Task 3): chain_emits_hidden_carrier = chain_args helper 의
    // single source. compute_chain_function_args 가 Some 반환 = chain function
    // 진입 가능 + hidden_carrier 가 layer 끝마다 chain end 로 갱신.
    // 이전 cu57 정의 (use_gemma_block_semantics && CHAIN_env && OUT_SCALE_env)
    // 를 helper invariant 로 일반화 — Llama 도 chain function 진입하면 자동으로
    // wire 활성 (W1 try_rms_norm / W2 K/V/Q dev input / W3 attn_out / W4 K/V f16).
    //
    // cu59 axis A Task 7: SignalEval phase timing — signal_ctx 생성 +
    // chain_function_active 호출 cost (host-side, chain function 진입 *전*).
    // 기존 6 phase 가 측정 밖이던 구간. cu58 step 3c 의 "double evaluation"
    // 영향 측정 = 매 layer × decode token 호출 cost.
    //
    // cu59 step 7.5 revert experiment 결과 (post-revert ABAB):
    // - Gemma B median: pre 2969.5ms → post 2997.7ms (회복 안 됨, 회귀 잔존)
    // - Llama 정확성 깨짐 (chain_emits_hidden_carrier=false 인데
    //   compute_chain_function_args 가 Llama 활성 → wire mismatch → garbage)
    // - smoke signal_eval_us_avg=0 (host call cost 0us)
    // → step 3b 는 wall-clock 회귀 단독 원인 아님. revert 본문 원복.
    //   진짜 root cause = compute_chain_function_args 의 chain function body
    //   자체 GPU compute (Task 6 의 kernel_us=480us/call 가설 재확인).
    #[cfg(feature = "cuda")]
    let signal_eval_t = if super::forward::chain_diag::is_active() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    #[cfg(feature = "cuda")]
    let chain_emits_hidden_carrier: bool = {
        let signal_ctx = super::forward::chain_args::ChainCallerCtx {
            architecture,
            layer_idx,
            num_layers: metadata.num_layers,
            hidden_dim,
            w,
            gemma_runtime_flavor,
            ple_fusion,
            ple_input_device_offset,
            metadata,
            // 신호 계산만 — carrier 활성 자체와 무관
            attn_on_device: false,
            attn_out_carrier_dev: None,
            has_gated_attn,
            gemma4_reuse_q_only,
            gemma4_attn_rot_active: gemma4_should_apply_attn_rotation(
                architecture,
                w.v_weight.ggml_type,
                head_dim,
            ),
            has_sliding_window: active_sliding_window(metadata, architecture, layer_idx).is_some(),
            long_kv_split_preferred: cu71_long_kv_split_preferred(head_dim, cache_pos + 1),
        };
        super::forward::chain_args::chain_function_active(&signal_ctx)
    };
    #[cfg(feature = "cuda")]
    if let Some(t) = signal_eval_t {
        super::forward::chain_diag::stash_phase_us(
            super::forward::chain_diag::Phase::SignalEval,
            t.elapsed().as_micros() as u64,
        );
    }
    #[cfg(not(feature = "cuda"))]
    let chain_emits_hidden_carrier: bool = false;
    #[cfg(feature = "cuda")]
    let qkv_consumes_device_norm = {
        let supports_device_input = |weight: &QuantizedWeight| {
            matches!(
                weight.ggml_type,
                rnb_loader::GGMLType::Q4_K | rnb_loader::GGMLType::Q6_K
            )
        };
        supports_device_input(&w.q_weight)
            && (gemma4_reuse_q_only
                || (supports_device_input(&w.k_weight) && supports_device_input(&w.v_weight)))
    };
    #[cfg(not(feature = "cuda"))]
    let qkv_consumes_device_norm = false;

    // cu57 step 67d: chain_emits_hidden_carrier=false 시 carrier path 전체 skip.
    // 이전엔 effective_rms_layer=0 로 강제해서 carrier 에 layer 0 norm 만 채우고
    // rms_used_cuda=true 반환했는데, 후속 attention QKV 의 carrier path 가 그
    // stale (layer 0) norm 을 모든 layer 에서 input 으로 사용 → garbage.
    // Nemotron / Llama / Qwen 처럼 chain function 미호출 arch 는 항상 host RMS.
    let rms_used_cuda = if chain_emits_hidden_carrier && qkv_consumes_device_norm {
        backend_runtime::try_rms_norm_into_decode_carrier_if_supported(
            layer_idx,
            &scratch.hidden[..hidden_dim],
            attn_norm_data,
            norm_eps,
            &mut scratch.norm_buf[..hidden_dim],
            unit_offset_attn_norm,
        )?
    } else {
        false
    };
    if !rms_used_cuda {
        apply_model_norm_into(
            &scratch.hidden[..hidden_dim],
            attn_norm_data,
            norm_eps,
            &mut scratch.norm_buf[..hidden_dim],
            architecture,
        );
    }
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "attn_norm",
        &scratch.norm_buf[..hidden_dim],
    );
    // cu76: eager attn_norm output probe for layer-0 divergence diff.
    // Even if rms_used_cuda=true (device computed it), reconstruct host RMSNorm
    // here for diagnostic comparison vs persistent kernel.
    if layer_idx == 0
        && crate::engine::policy::env_string("RNB_CUDA_EAGER_ATTN_NORM_PROBE").as_deref()
            == Some("1")
    {
        let hidden_in = &scratch.hidden[..hidden_dim];
        let sumsq: f32 = hidden_in.iter().map(|v| v * v).sum();
        eprintln!(
            "[cu76 eager input] layer 0 input first8={:?} sumsq={:.6} mean_sq={:.6} first4_weight={:?}",
            &hidden_in[..8.min(hidden_dim)],
            sumsq,
            sumsq / hidden_dim as f32,
            &attn_norm_data[..4.min(attn_norm_data.len())],
        );
        let mut diag = vec![0.0f32; hidden_dim];
        apply_model_norm_into(
            &scratch.hidden[..hidden_dim],
            attn_norm_data,
            norm_eps,
            &mut diag,
            architecture,
        );
        let mean = diag.iter().sum::<f32>() / hidden_dim as f32;
        let max_abs = diag.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "[cu76 eager attn_norm] layer 0 (host recompute) mean={:.6} max_abs={:.6} first8={:?}",
            mean,
            max_abs,
            &diag[..8.min(hidden_dim)],
        );
    }

    // 2. QKV GEMV — try GPU batch first, then CPU fallback.
    // cu29 Phase 2: hd=128 fused QKV+GPU RoPE+f16 pack path 우선 시도. Llama /
    // Mistral 처럼 qk-norm 없는 dense hd128 모델에서 host RoPE round-trip 제거.
    let q_out_dim = w.q_weight.rows; // num_heads * head_dim * 2 (gated) or num_heads * head_dim

    let used_fused_hd128 = {
        let (rope_dim, rope_theta, proportional_rope) =
            resolve_rope_params(metadata, architecture, layer_idx, head_dim);
        if !rms_used_cuda
            && head_dim == 128
            && !has_gated_attn
            && !gemma4_reuse_q_only
            && w.q_norm.is_none()
            && w.k_norm.is_none()
            && !proportional_rope
            && rope_dim == head_dim
            && scratch.k_bits_buf.len() >= kv_dim
            && scratch.v_bits_buf.len() >= kv_dim
            && q_out_dim == num_heads_layout * head_dim
        {
            backend_runtime::dense_q4k_attention_qkv_rope_hd128_if_supported(
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                num_heads_layout,
                num_kv_heads_layout,
                rope_theta,
                rope_pos,
                &scratch.norm_buf[..hidden_dim],
                &mut scratch.q_buf[..q_out_dim],
                &mut scratch.k_bits_buf[..kv_dim],
                &mut scratch.v_bits_buf[..kv_dim],
            )?
        } else {
            false
        }
    };

    // cu66: device QKV + QK norm + RoPE + f16 K/V pack + Q dedicated carrier.
    // No D2H, no sync. Gated to non-sliding-window layers (sliding_window
    // layers use CPU attention fallback which needs host K/V).
    #[cfg(feature = "cuda")]
    let cu65_device_qkv_enabled = crate::engine::tuning_runtime::cu65_device_qkv_enabled();
    #[cfg(feature = "cuda")]
    let cu68_layer_graph_enabled = crate::engine::tuning_runtime::cu68_layer_graph_enabled();
    #[cfg(feature = "cuda")]
    let cu68_layer_graph_decision = super::decode_layer_graph::cu68_layer_graph_decision(
        super::decode_layer_graph::Cu68LayerGraphRequest {
            layer_graph_enabled: cu68_layer_graph_enabled,
            device_qkv_enabled: cu65_device_qkv_enabled,
            used_fused_hd128,
            chain_emits_hidden_carrier,
            rms_used_cuda,
            has_gated_attn,
            gemma4_reuse_q_only,
            gemma4_attn_rot_active: gemma4_should_apply_attn_rotation(
                architecture,
                w.v_weight.ggml_type,
                head_dim,
            ),
            has_sliding_window: active_sliding_window(metadata, architecture, layer_idx).is_some(),
        },
    );
    #[cfg(feature = "cuda")]
    let cu68_attention_device_len = matches!(
        cu68_layer_graph_decision,
        super::decode_layer_graph::Cu68LayerGraphDecision::Eligible
    ) && head_dim == 512;
    #[cfg(not(feature = "cuda"))]
    let cu68_attention_device_len = false;

    #[cfg(feature = "cuda")]
    let (used_device_qkv, cu66_q_dev): (bool, Option<u64>) = if !used_fused_hd128
        && chain_emits_hidden_carrier
        && rms_used_cuda
        && !has_gated_attn
        && !gemma4_reuse_q_only
        && active_sliding_window(metadata, architecture, layer_idx).is_none()
        && cu65_device_qkv_enabled
    {
        let norm_carrier_bytes = hidden_dim * std::mem::size_of::<f32>();
        if let Ok(norm_carrier_dev) =
            backend_runtime::acquire_decode_norm_buf_carrier(norm_carrier_bytes)
        {
            let q_norm_data = w
                .q_norm
                .as_ref()
                .map(|t| super::cpu_runtime::kernels::tensor_as_f32_slice(t));
            let k_norm_data = w
                .k_norm
                .as_ref()
                .map(|t| super::cpu_runtime::kernels::tensor_as_f32_slice(t));
            let layer_q_rows = w.q_weight.rows;
            let layer_kv_dim = w.k_weight.rows;
            let (_, rope_theta_cu65, _) =
                resolve_rope_params(metadata, architecture, layer_idx, head_dim);

            let use_graph = super::decode_layer_graph::cu68_qkv_graph_enabled(
                crate::engine::policy::env_string("RNB_CU65_GRAPH").as_deref() == Some("1"),
                cu68_attention_device_len,
            );
            let qkv_result = if use_graph {
                backend_runtime::decode_device_qkv_rope_kv_graph(
                    layer_idx,
                    norm_carrier_dev,
                    w.q_weight.data.as_bytes().unwrap_or(&[]),
                    w.k_weight.data.as_bytes().unwrap_or(&[]),
                    w.v_weight.data.as_bytes().unwrap_or(&[]),
                    q_norm_data,
                    k_norm_data,
                    layer_q_rows,
                    layer_kv_dim,
                    hidden_dim,
                    num_heads_layout,
                    num_kv_heads_layout,
                    rope_theta_cu65,
                    rope_pos,
                    cache_pos,
                    metadata.norm_eps,
                    &mut scratch.q_buf[..layer_q_rows],
                    &mut scratch.k_buf[..layer_kv_dim],
                    &mut scratch.v_buf[..layer_kv_dim],
                )
            } else {
                backend_runtime::decode_device_qkv_rope_kv(
                    layer_idx,
                    norm_carrier_dev,
                    w.q_weight.data.as_bytes().unwrap_or(&[]),
                    w.k_weight.data.as_bytes().unwrap_or(&[]),
                    w.v_weight.data.as_bytes().unwrap_or(&[]),
                    q_norm_data,
                    k_norm_data,
                    layer_q_rows,
                    layer_kv_dim,
                    hidden_dim,
                    num_heads_layout,
                    num_kv_heads_layout,
                    rope_theta_cu65,
                    rope_pos,
                    cache_pos,
                    metadata.norm_eps,
                    &mut scratch.q_buf[..layer_q_rows],
                    &mut scratch.k_buf[..layer_kv_dim],
                    &mut scratch.v_buf[..layer_kv_dim],
                )
            };
            match qkv_result {
                Ok(q_dev) => (true, Some(q_dev)),
                Err(_) => (false, None),
            }
        } else {
            (false, None)
        }
    } else {
        (false, None)
    };
    #[cfg(not(feature = "cuda"))]
    let (used_device_qkv, cu66_q_dev): (bool, Option<u64>) = (false, None);

    if !used_fused_hd128 && !used_device_qkv {
        decode_attention_qkv_projection(
            scratch,
            w,
            layer_idx,
            hidden_dim,
            q_out_dim,
            kv_dim,
            gemma4_reuse_q_only,
            verbose,
            rms_used_cuda,
            #[cfg(feature = "vulkan")]
            &mut vulkan_backend,
            |label, t| prof!(label, t),
        )?;
    }
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "q_raw",
        &scratch.q_buf[..q_out_dim],
    );
    if !used_fused_hd128 {
        emit_mtp_finite_trace("decode-attn", layer_idx, "k_raw", &scratch.k_buf[..kv_dim]);
        emit_mtp_finite_trace("decode-attn", layer_idx, "v_raw", &scratch.v_buf[..kv_dim]);
    }

    if !used_fused_hd128 && !used_device_qkv {
        apply_decode_attention_qkv_postprocess(
            scratch,
            w,
            architecture,
            layout,
            q_out_dim,
            norm_eps,
            gemma4_reuse_q_only,
        );
    }
    let q_after_post = if has_gated_attn {
        &scratch.q_split[..q_dim]
    } else {
        &scratch.q_buf[..q_dim]
    };
    emit_mtp_finite_trace("decode-attn", layer_idx, "q_post", q_after_post);
    if !used_fused_hd128 {
        emit_mtp_finite_trace("decode-attn", layer_idx, "k_post", &scratch.k_buf[..kv_dim]);
        emit_mtp_finite_trace("decode-attn", layer_idx, "v_post", &scratch.v_buf[..kv_dim]);
    }
    prof!("qkv_gemv+norm", t0);

    // 3. RoPE — fused path 또는 device QKV path 에서 GPU RoPE 가 이미 적용된 상태라 skip.
    let t0 = std::time::Instant::now();
    if !used_fused_hd128 && !used_device_qkv {
        let q_slice = if has_gated_attn {
            &mut scratch.q_split[..q_dim]
        } else {
            &mut scratch.q_buf[..q_dim]
        };
        apply_decode_rope(
            metadata,
            architecture,
            rope_freqs,
            layer_idx,
            rope_pos,
            head_dim,
            q_dim,
            kv_dim,
            gemma4_reuse_q_only,
            q_slice,
            &mut scratch.k_buf[..kv_dim],
        );
    }
    let q_after_rope = if has_gated_attn {
        &scratch.q_split[..q_dim]
    } else {
        &scratch.q_buf[..q_dim]
    };
    emit_mtp_finite_trace("decode-attn", layer_idx, "q_rope", q_after_rope);
    emit_mtp_finite_trace("decode-attn", layer_idx, "k_rope", &scratch.k_buf[..kv_dim]);
    if layer_idx == 0 && attn_trace_enabled() {
        emit_vec_trace("decode", layer_idx, "v", &scratch.v_buf[..kv_dim]);
    }
    if verbose {
        prof!("rope", t0);
    }

    let t_kv_cache = std::time::Instant::now();
    if verbose {
        prof!("kv_cache", t_kv_cache);
    } else {
        prof!("rope+kv_cache", t0);
    }

    // 6. Attention decode
    let t0 = std::time::Instant::now();
    let q_slice = if has_gated_attn {
        &scratch.q_split[..q_dim]
    } else {
        &scratch.q_buf[..q_dim]
    };
    // cu29 Phase 2: fused hd128 path 면 K/V bits 를 KvCache 에 직접 append
    // (host f32→f16 변환 skip).
    if used_fused_hd128 && owns_kv {
        kv_cache.append_bits_range(
            kv_cache_layer,
            cache_pos,
            1,
            &scratch.k_bits_buf[..kv_dim],
            &scratch.v_bits_buf[..kv_dim],
        );
    }
    // cu47 step 33 + cu57: attn_out carrier 는 chain function 안에서 device
    // input 으로 사용되는 경로에서만 의미 있음. chain 미호출 arch (Llama / Qwen /
    // Nemotron) 는 carrier 가 dead device write → cu56 부터 Llama gated-ON
    // broken 의 root cause 후보. chain_emits_hidden_carrier (= use_gemma_block_semantics
    // + CHAIN + OUT_SCALE) 와 신호 동일하므로 재사용해서 acquire 자체 skip.
    // host gated_attn 또는 gemma4_rot 적용 layer 는 carrier 사용 안 함 (host
    // post-processing 이 attn_out 변경 → device 와 mismatch).
    #[cfg(feature = "cuda")]
    let gemma4_attn_rot_active =
        gemma4_should_apply_attn_rotation(architecture, w.v_weight.ggml_type, head_dim);
    #[cfg(feature = "cuda")]
    let attn_out_carrier_dev: Option<u64> = {
        if chain_emits_hidden_carrier && !has_gated_attn && !gemma4_attn_rot_active {
            let bytes = q_dim * std::mem::size_of::<f32>();
            backend_runtime::acquire_decode_attn_out_carrier(bytes).ok()
        } else {
            None
        }
    };
    #[cfg(not(feature = "cuda"))]
    let attn_out_carrier_dev: Option<u64> = None;
    // cu53 step 53: K/V projection device gemv 시도 — broken (token garbage).
    // 진단: norm_carrier (device) 정확, host scratch.norm_buf=0 (cu45 D2H 제거),
    // host scratch.k_buf 정확 (cu42 wire 결과), 단 K f32 carrier device gemv 결과
    // 는 host_k 의 400x. 같은 kernel + 같은 input/weight 인데 결과 다름. cu54
    // root cause 추적 약속.
    // cu53 fallback = cu52 stepping stone (host scratch → carrier H2D + f16 pack).
    // cu52 step 50 + cu57 step 67c: K/V f16 carrier 는 attention forward 의
    // device source 로 들어감. chain function 호출 arch 외 (Nemotron 등) 에서
    // host K/V scratch 와 device source f16 의 numerical 차이가 layer 누적
    // drift → cu57 step 67b 후 Nemotron 의 step 2+ drift root cause.
    // chain_emits_hidden_carrier (= use_gemma_block_semantics + CHAIN + OUT_SCALE)
    // 와 동일 신호로 W4 도 gating. Gemma4 동작 동등, 그 외 arch wire 비활성.
    #[cfg(feature = "cuda")]
    let (last_token_k_dev, last_token_v_dev): (Option<u64>, Option<u64>) = {
        if used_device_qkv
            && chain_emits_hidden_carrier
            && crate::engine::policy::cuda_decode_device_kv_cache_enabled()
            && !has_gated_attn
            && !gemma4_reuse_q_only
        {
            // cu66: f16 K/V already packed on device by decode_device_qkv_rope_kv.
            // Reuse the same carrier ptrs — skips cu52's host H2D + f16 pack.
            let f16_bytes = kv_dim * std::mem::size_of::<u16>();
            let k_f16 = backend_runtime::acquire_decode_k_f16_carrier(f16_bytes).ok();
            let v_f16 = backend_runtime::acquire_decode_v_f16_carrier(f16_bytes).ok();
            match (k_f16, v_f16) {
                (Some(k), Some(v)) => (Some(k), Some(v)),
                _ => (None, None),
            }
        } else if !used_device_qkv
            && chain_emits_hidden_carrier
            && crate::engine::policy::cuda_decode_device_kv_cache_enabled()
            && !has_gated_attn
            && !gemma4_reuse_q_only
        {
            // cu52 stepping stone: host K/V → H2D → f32→f16 pack → f16 carrier.
            let f32_bytes = kv_dim * std::mem::size_of::<f32>();
            let f16_bytes = kv_dim * std::mem::size_of::<u16>();
            let k_f32 = backend_runtime::acquire_decode_k_carrier(f32_bytes).ok();
            let v_f32 = backend_runtime::acquire_decode_v_carrier(f32_bytes).ok();
            let k_f16 = backend_runtime::acquire_decode_k_f16_carrier(f16_bytes).ok();
            let v_f16 = backend_runtime::acquire_decode_v_f16_carrier(f16_bytes).ok();
            if let (Some(kf), Some(vf), Some(kh), Some(vh)) = (k_f32, v_f32, k_f16, v_f16) {
                let upload_k =
                    backend_runtime::upload_to_decode_hidden_carrier(&scratch.k_buf[..kv_dim], kf);
                let upload_v =
                    backend_runtime::upload_to_decode_hidden_carrier(&scratch.v_buf[..kv_dim], vf);
                let pack_k = backend_runtime::f32_to_f16_pack_device(kf, kh, kv_dim);
                let pack_v = backend_runtime::f32_to_f16_pack_device(vf, vh, kv_dim);
                if upload_k.is_ok() && upload_v.is_ok() && pack_k.is_ok() && pack_v.is_ok() {
                    (Some(kh), Some(vh))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        }
    };
    #[cfg(not(feature = "cuda"))]
    let (last_token_k_dev, last_token_v_dev): (Option<u64>, Option<u64>) = (None, None);
    let attn_on_device = decode_attention_compute(
        kv_cache,
        metadata,
        architecture,
        layer_idx,
        kv_cache_layer,
        owns_kv,
        cache_pos,
        layout,
        q_slice,
        &scratch.k_buf[..kv_dim],
        &scratch.v_buf[..kv_dim],
        &mut scratch.attn_out[..q_dim],
        used_fused_hd128,
        attn_out_carrier_dev,
        last_token_k_dev,
        last_token_v_dev,
        cu66_q_dev, // cu66: dedicated Q carrier, safe across layers
        cu68_attention_device_len,
        #[cfg(feature = "vulkan")]
        vulkan_backend.as_deref_mut(),
    )?;
    #[cfg(not(feature = "cuda"))]
    let _ = attn_on_device;
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "attn_out",
        &scratch.attn_out[..q_dim],
    );
    if verbose {
        prof!("attention_decode_into", t0);
    }

    // Gated attention: output *= sigmoid(gate)
    let t_gate = std::time::Instant::now();
    if has_gated_attn {
        sigmoid_mul_f32_inplace(&mut scratch.attn_out[..q_dim], &scratch.gate_split[..q_dim]);
    }
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "attn_after_gate",
        &scratch.attn_out[..q_dim],
    );

    if gemma4_should_apply_attn_rotation(architecture, w.v_weight.ggml_type, head_dim) {
        gemma4_apply_attn_rot_inplace(&mut scratch.attn_out[..q_dim], head_dim, q_dim, 64);
    }

    if verbose {
        prof!("attn_gate", t_gate);
    } else {
        prof!("attention", t0);
    }

    // cu58 step 1: chain function 호출을 arch-agnostic helper (chain_args) 로
    // 일반화. Gemma family path 만 helper 안에 이동 (산술 변경 0). 다음 step 에서
    // Llama / Qwen / Phi dense path 도 helper 의 분기로 진입.
    #[cfg(feature = "cuda")]
    {
        let ctx = super::forward::chain_args::ChainCallerCtx {
            architecture,
            layer_idx,
            hidden_dim,
            num_layers: metadata.num_layers,
            w,
            gemma_runtime_flavor,
            ple_fusion,
            ple_input_device_offset,
            metadata,
            attn_on_device,
            attn_out_carrier_dev,
            has_gated_attn,
            gemma4_reuse_q_only,
            gemma4_attn_rot_active,
            has_sliding_window: active_sliding_window(metadata, architecture, layer_idx).is_some(),
            long_kv_split_preferred: cu71_long_kv_split_preferred(head_dim, cache_pos + 1),
        };
        // cu59 axis A — chain function 호출 직전 call context stash.
        // token index 로 cache_pos 사용 (decode 의 절대 token 위치).
        super::forward::chain_diag::stash_call_context(layer_idx, cache_pos);
        if let Some(args) = super::forward::chain_args::compute_chain_function_args(&ctx) {
            // cu71: eligibility is now computed on the product path; execution
            // stays on the existing dense chain until segment replay is wired.
            let cu71_layer_segment_graph_allowed = args.layer_segment_graph_allowed;
            if let (Some(trace), Some(request)) = (cu72_trace, args.layer_segment_graph_request) {
                trace.record_layer(super::decode_layer_graph::Cu72HiddenPersistenceLayer {
                    layer_idx,
                    request,
                });
            }
            if cu71_layer_segment_graph_allowed
                && crate::engine::tuning_runtime::cu71_layer_segment_graph_trace_enabled()
            {
                eprintln!(
                    "[cu71 layer-segment-graph] state=eligible_pending_replay layer={layer_idx}"
                );
            }
            let cu71_layer_segment_graph_context = if cu71_layer_segment_graph_allowed {
                let kv_bucket = kv_cache
                    .layer_bucket_view(kv_cache_layer, cache_pos + 1)
                    .ok();
                let q_weight_identity = cu71_quantized_weight_identity(&w.q_weight);
                let k_weight_identity = cu71_quantized_weight_identity(&w.k_weight);
                let v_weight_identity = cu71_quantized_weight_identity(&w.v_weight);
                match (
                    cu66_q_dev,
                    last_token_k_dev,
                    last_token_v_dev,
                    kv_bucket,
                    q_weight_identity,
                    k_weight_identity,
                    v_weight_identity,
                ) {
                    (
                        Some(q_carrier_dev),
                        Some(k_f16_dev),
                        Some(v_f16_dev),
                        Some(kv_bucket),
                        q_weight_identity,
                        k_weight_identity,
                        v_weight_identity,
                    ) if q_weight_identity != 0
                        && k_weight_identity != 0
                        && v_weight_identity != 0 =>
                    {
                        let q_norm_hash = w
                            .q_norm
                            .as_ref()
                            .map(kernels::tensor_as_f32_slice)
                            .map(cu71_f32_bit_hash)
                            .unwrap_or(0);
                        let k_norm_hash = w
                            .k_norm
                            .as_ref()
                            .map(kernels::tensor_as_f32_slice)
                            .map(cu71_f32_bit_hash)
                            .unwrap_or(0);
                        let (_, rope_theta_cu71, _) =
                            resolve_rope_params(metadata, architecture, layer_idx, head_dim);
                        Some(backend_runtime::Cu71LayerSegmentGraphRuntimeContext {
                            layer_idx,
                            q_rows: w.q_weight.rows,
                            kv_dim,
                            num_heads: num_heads_layout,
                            num_kv_heads: num_kv_heads_layout,
                            head_dim,
                            rope_theta: rope_theta_cu71,
                            attention_scale: (head_dim as f32).sqrt().recip(),
                            q_quant: w.q_weight.ggml_type as u32,
                            k_quant: w.k_weight.ggml_type as u32,
                            v_quant: w.v_weight.ggml_type as u32,
                            q_weight_identity,
                            k_weight_identity,
                            v_weight_identity,
                            q_norm_hash,
                            k_norm_hash,
                            q_carrier_dev,
                            k_f16_dev,
                            v_f16_dev,
                            kv_bucket,
                            long_kv_split_preferred: cu71_long_kv_split_preferred(
                                head_dim,
                                cache_pos + 1,
                            ),
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };
            let t_chain = std::time::Instant::now();
            if backend_runtime::dense_q4k_attention_output_gelu_ffn_norm_residual_if_supported(
                args.o_weight,
                args.gate_weight,
                args.up_weight,
                args.down_weight,
                args.post_attn_norm,
                args.ffn_norm,
                args.post_ffn_norm,
                args.ple_gate_weight,
                args.ple_proj_weight,
                args.ple_post_norm_weight,
                args.ple_input,
                args.ple_input_device_offset,
                args.ple_dim,
                &mut scratch.hidden[..hidden_dim],
                &scratch.attn_out[..q_dim],
                norm_eps,
                args.unit_offset_post_attn_norm,
                args.unit_offset_ffn_norm,
                args.unit_offset_ple_norm,
                args.hidden_carrier_dev,
                args.skip_h2d_hidden,
                args.skip_d2h_hidden,
                args.layer_output_scale,
                args.attn_out_dev_carrier,
                args.ffn_uses_gelu,
                args.dense_chain_graph_allowed,
                cu71_layer_segment_graph_context,
            )? {
                prof!("o_proj+ffn_chain", t_chain);
                // cu59 axis A — chain call 완료 후 sub-phase timing flush.
                super::forward::chain_diag::flush_call();
                return Ok(args.ple_fused);
            }
        }
    }

    // 7. Output projection + residual — try GPU first
    let t0 = std::time::Instant::now();
    let t_o_weight = std::time::Instant::now();
    // Metal attention O chain (RNB_METAL_O_CHAIN=1) — o_proj + residual 단일 command
    // buffer. non-gemma(qwen) 한정(gemma post-attn-norm 제외). 성공 시 두 함수 skip.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let o_chain_done = if !use_gemma_block_semantics(architecture) {
        backend_runtime::metal_attention_o_chain_into_if_supported(
            &scratch.attn_out[..q_dim],
            &w.o_weight,
            &mut scratch.hidden[..hidden_dim],
            hidden_dim,
            q_dim,
        )?
    } else {
        false
    };
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    let o_chain_done = false;

    if !o_chain_done {
        decode_attention_output_projection(
            scratch,
            w,
            layer_idx,
            q_dim,
            hidden_dim,
            #[cfg(feature = "vulkan")]
            vulkan_backend.as_deref_mut(),
        )?;
        emit_mtp_finite_trace(
            "decode-attn",
            layer_idx,
            "o_projected",
            &scratch.proj_buf[..hidden_dim],
        );
        apply_decode_attention_residual(
            scratch,
            architecture,
            gemma_runtime_flavor,
            w,
            hidden_dim,
            norm_eps,
            layer_idx,
            source_hidden,
            prev_layer_hidden,
            verbose,
            t_o_weight,
            t0,
            |label, t| prof!(label, t),
        );
    }
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "after_attn_residual",
        &scratch.hidden[..hidden_dim],
    );

    if matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_decode_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(false);
    }

    // 8. FFN.
    let t0 = std::time::Instant::now();
    decode_ffn_layer(
        scratch,
        architecture,
        w,
        hidden_dim,
        norm_eps,
        layer_idx,
        #[cfg(feature = "vulkan")]
        vulkan_backend.as_mut().map(|v| &mut **v),
    )?;
    emit_mtp_finite_trace(
        "decode-attn",
        layer_idx,
        "after_ffn",
        &scratch.hidden[..hidden_dim],
    );
    prof!("ffn_total", t0);

    Ok(false)
}
