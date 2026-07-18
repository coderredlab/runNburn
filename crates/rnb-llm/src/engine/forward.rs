//! Full-prefill attention + GDN forward helpers.
//!
//! Moved from `engine.rs`.

use super::*;

mod attention_compute;
pub(in crate::engine) mod attention_output;
#[cfg(feature = "cuda")]
pub(in crate::engine) mod chain_args;
#[cfg(any(feature = "cuda", test))]
pub(in crate::engine) mod chain_diag;
mod dense_chain;
pub(in crate::engine) mod ffn;
mod fused_qkv_chain;
mod projection;
mod rope;

use attention_compute::compute_prefill_attention;
use attention_output::apply_prefill_attention_output;
use dense_chain::{
    try_prefill_attention_output_ffn_chain, try_prefill_f16kv_attention_output_ffn_chain,
};
use ffn::forward_prefill_ffn;
pub(in crate::engine) use fused_qkv_chain::Gemma4PrefillPleFusion;
use fused_qkv_chain::{
    try_prefill_q4k_f16_qkv_hd256_window_dense_chain, try_prefill_q4k_f16_qkv_hd512_dense_chain,
    try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain,
    try_prefill_q4k_f16_reuse_q_hd512_dense_chain,
};
#[cfg(feature = "cuda")]
pub(in crate::engine) use fused_qkv_chain::{
    try_prefill_q4k_f16_qkv_hd256_window_dense_chain_from_device,
    try_prefill_q4k_f16_qkv_hd512_dense_chain_from_device,
    try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain_from_device,
    try_prefill_q4k_f16_reuse_q_hd512_dense_chain_from_device,
};
use projection::{
    project_prefill_attention, try_prefill_atn_core_metal, try_prefill_atn_full_layer_metal,
    try_prefill_atn_o_tail_metal, try_prefill_attention_q4k_f16_qkv_attention_hd512,
    try_prefill_attn_chain_metal,
};
use rope::apply_prefill_rope;

pub(super) struct PrefillAttentionLayerOutput {
    pub(super) hidden: Tensor,
    pub(super) gemma4_ple_fused: bool,
    pub(super) gemma4_output_scale_fused: bool,
    #[cfg(feature = "cuda")]
    pub(super) device_output: Option<backend_runtime::NemotronDeviceLayerOutput>,
}

impl PrefillAttentionLayerOutput {
    fn host(hidden: Tensor) -> Self {
        Self {
            hidden,
            gemma4_ple_fused: false,
            gemma4_output_scale_fused: false,
            #[cfg(feature = "cuda")]
            device_output: None,
        }
    }
}

/// Attention 레이어 forward (기존 LLaMA/Qwen2 + Qwen3.5 attention 지원)
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    _kv_dim: usize,
    _rope_theta: f32,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    Ok(forward_attention_layer_impl(
        kv_cache,
        metadata,
        architecture,
        hidden,
        w,
        rope_freqs,
        layer_idx,
        seq_len,
        pos_start,
        pos_start,
        _num_heads,
        _num_kv_heads,
        _head_dim,
        _kv_dim,
        _rope_theta,
        norm_eps,
        None,
    )?
    .hidden)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention_layer_with_rope_pos(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    cache_pos_start: usize,
    rope_pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    Ok(forward_attention_layer_impl(
        kv_cache,
        metadata,
        architecture,
        hidden,
        w,
        rope_freqs,
        layer_idx,
        seq_len,
        cache_pos_start,
        rope_pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        None,
    )?
    .hidden)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention_layer_with_gemma4_ple_fusion(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<PrefillAttentionLayerOutput> {
    forward_attention_layer_impl(
        kv_cache,
        metadata,
        architecture,
        hidden,
        w,
        rope_freqs,
        layer_idx,
        seq_len,
        pos_start,
        pos_start,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_dim,
        rope_theta,
        norm_eps,
        ple_fusion,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_impl(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    mut hidden: Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    rope_pos_start: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    _kv_dim: usize,
    _rope_theta: f32,
    norm_eps: f32,
    ple_fusion: Option<&Gemma4PrefillPleFusion<'_>>,
) -> crate::error::Result<PrefillAttentionLayerOutput> {
    let profiling = super::policy::profiling_enabled();
    let positions_aligned = pos_start == rope_pos_start;
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
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let kv_dim = layout.kv_dim;
    if super::policy::debug_gemma_layout_enabled() {
        eprintln!(
            "[prefill-attn] layer={} q_dim={} kv_dim={} head_dim={} gated={}",
            layer_idx, layout.q_dim, layout.kv_dim, layout.head_dim, layout.has_gated_attn
        );
    }
    if kv_trace_enabled() {
        eprintln!(
            "[kv-trace][prefill] layer={} cache_layer={} owns_kv={} seq_len={} pos_start={} kv_dim={} kv_heads={} head_dim={}",
            layer_idx, kv_cache_layer, owns_kv, seq_len, pos_start, kv_dim, num_kv_heads, head_dim
        );
    }
    let profile_all_layers = std::env::var_os("RNB_QWEN_PROFILE_ALL_LAYERS").is_some();
    let prof = |label: &str, t: std::time::Instant| {
        if profiling && (profile_all_layers || layer_idx == 3) {
            eprintln!(
                "  [ATN L{}] {:20} {:.1}ms",
                layer_idx,
                label,
                t.elapsed().as_micros() as f64 / 1000.0
            );
        }
    };

    // --- Attention sub-block ---
    let fused_t0 = std::time::Instant::now();
    if positions_aligned && gemma4_reuse_q_only {
        let kv_len = pos_start + seq_len;
        let (cached_k_f16, cached_v_f16) = kv_cache.get_up_to(kv_cache_layer, kv_len);
        if let Some(fused) = try_prefill_q4k_f16_reuse_q_hd256_window_dense_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            cached_k_f16,
            cached_v_f16,
            layer_idx,
            seq_len,
            kv_len,
            pos_start,
            norm_eps,
            ple_fusion,
        )? {
            prof("q+cachedkv+attn+ffn_cuda", fused_t0);
            return Ok(PrefillAttentionLayerOutput {
                hidden: fused.hidden,
                gemma4_ple_fused: fused.gemma4_ple_fused,
                gemma4_output_scale_fused: fused.gemma4_output_scale_fused,
                #[cfg(feature = "cuda")]
                device_output: fused.device_output,
            });
        }
        if let Some(fused) = try_prefill_q4k_f16_reuse_q_hd512_dense_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            cached_k_f16,
            cached_v_f16,
            layer_idx,
            seq_len,
            kv_len,
            pos_start,
            norm_eps,
            ple_fusion,
        )? {
            prof("q+cachedkv+attn+ffn_cuda", fused_t0);
            return Ok(PrefillAttentionLayerOutput {
                hidden: fused.hidden,
                gemma4_ple_fused: fused.gemma4_ple_fused,
                gemma4_output_scale_fused: fused.gemma4_output_scale_fused,
                #[cfg(feature = "cuda")]
                device_output: fused.device_output,
            });
        }
    }

    if positions_aligned {
        if let Some(fused) = try_prefill_q4k_f16_qkv_hd256_window_dense_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
            ple_fusion,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+attn+ffn_cuda", fused_t0);
            return Ok(PrefillAttentionLayerOutput {
                hidden: fused.hidden,
                gemma4_ple_fused: fused.gemma4_ple_fused,
                gemma4_output_scale_fused: fused.gemma4_output_scale_fused,
                #[cfg(feature = "cuda")]
                device_output: fused.device_output,
            });
        }
    }

    if positions_aligned {
        if let Some(fused) = try_prefill_q4k_f16_qkv_hd512_dense_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
            ple_fusion,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+attn+ffn_cuda", fused_t0);
            return Ok(PrefillAttentionLayerOutput {
                hidden: fused.hidden,
                gemma4_ple_fused: fused.gemma4_ple_fused,
                gemma4_output_scale_fused: fused.gemma4_output_scale_fused,
                #[cfg(feature = "cuda")]
                device_output: fused.device_output,
            });
        }
    }

    let attn_step = if positions_aligned {
        if let Some(fused) = try_prefill_atn_full_layer_metal(
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+qknorm+attn_o_ffn_full_metal", fused_t0);
            PrefillAttentionStep::FinalHidden(fused.hidden)
        } else if let Some(fused) = {
            if backend_runtime::metal_prefill_atn_o_tail_requested() {
                backend_runtime::metal_prefill_atn_o_tail_expected_dense_layer();
                try_prefill_atn_o_tail_metal(
                    metadata,
                    architecture,
                    &hidden,
                    w,
                    rope_freqs,
                    layout,
                    gemma4_reuse_q_only,
                    layer_idx,
                    seq_len,
                    pos_start,
                    norm_eps,
                )?
            } else {
                None
            }
        } {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+qknorm+attn_o_tail_metal", fused_t0);
            PrefillAttentionStep::PostAttentionHidden(fused.hidden)
        } else if let Some(fused) = try_prefill_atn_core_metal(
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+qknorm+attn_core_metal", fused_t0);
            PrefillAttentionStep::AttnOut(fused.attn_out)
        } else if let Some(fused) = try_prefill_attn_chain_metal(
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+qknorm+attn_chain_metal", fused_t0);
            PrefillAttentionStep::AttnOut(fused.attn_out)
        } else if let Some(fused) = try_prefill_attention_q4k_f16_qkv_attention_hd512(
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            pos_start,
            norm_eps,
        )? {
            if owns_kv {
                kv_cache.replace_layer_f16_range(
                    kv_cache_layer,
                    pos_start,
                    seq_len,
                    &fused.k_bits,
                    &fused.v_bits,
                );
            }
            prof("qkv+rope+attn_cuda", fused_t0);
            PrefillAttentionStep::AttnOut(fused.attn_out)
        } else {
            let t0 = std::time::Instant::now();
            let projection = project_prefill_attention(
                metadata,
                architecture,
                &hidden,
                w,
                rope_freqs,
                layout,
                gemma4_reuse_q_only,
                layer_idx,
                seq_len,
                rope_pos_start,
                norm_eps,
            )?;
            compute_prefill_attention_from_projection(
                kv_cache,
                metadata,
                architecture,
                &hidden,
                w,
                rope_freqs,
                layout,
                gemma_runtime_flavor,
                gemma4_reuse_q_only,
                owns_kv,
                kv_cache_layer,
                layer_idx,
                seq_len,
                pos_start,
                rope_pos_start,
                head_dim,
                kv_dim,
                norm_eps,
                projection,
                &prof,
                t0,
            )?
        }
    } else {
        let t0 = std::time::Instant::now();
        let projection = project_prefill_attention(
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma4_reuse_q_only,
            layer_idx,
            seq_len,
            rope_pos_start,
            norm_eps,
        )?;
        compute_prefill_attention_from_projection(
            kv_cache,
            metadata,
            architecture,
            &hidden,
            w,
            rope_freqs,
            layout,
            gemma_runtime_flavor,
            gemma4_reuse_q_only,
            owns_kv,
            kv_cache_layer,
            layer_idx,
            seq_len,
            pos_start,
            rope_pos_start,
            head_dim,
            kv_dim,
            norm_eps,
            projection,
            &prof,
            t0,
        )?
    };
    let attn_out = match attn_step {
        PrefillAttentionStep::AttnOut(attn_out) => Some(attn_out),
        PrefillAttentionStep::PostAttentionHidden(post_attention_hidden) => {
            hidden = post_attention_hidden;
            None
        }
        PrefillAttentionStep::FinalHidden(hidden) => {
            return Ok(PrefillAttentionLayerOutput::host(hidden));
        }
    };

    if let Some(attn_out) = attn_out {
        let chain_t0 = std::time::Instant::now();
        if let Some(chained_hidden) = try_prefill_attention_output_ffn_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            &attn_out,
            layer_idx,
            seq_len,
            norm_eps,
        )? {
            prof("o_proj+ffn_chain", chain_t0);
            return Ok(PrefillAttentionLayerOutput::host(chained_hidden));
        }

        // Output projection + residual
        let t0 = std::time::Instant::now();
        hidden = apply_prefill_attention_output(
            metadata,
            architecture,
            gemma_runtime_flavor,
            hidden,
            w,
            &attn_out,
            layer_idx,
            seq_len,
            norm_eps,
        )?;
        prof("o_proj+residual", t0);
    }

    if matches!(architecture, ModelArchitecture::NemotronHMoE)
        || gemma_effective_skip_ffn_enabled(architecture, gemma_runtime_flavor, layer_idx)
    {
        return Ok(PrefillAttentionLayerOutput::host(hidden));
    }

    // --- FFN sub-block ---
    let t0 = std::time::Instant::now();
    hidden = forward_prefill_ffn(
        metadata,
        architecture,
        hidden,
        w,
        layer_idx,
        seq_len,
        norm_eps,
    )?;
    prof("ffn_total", t0);

    // mv38 D / mv39: layer-by-layer hidden state trace. RNB_VULKAN_LAYER_TRACE=1 시
    // attention layer 의 hidden_after_window stat 출력. t=0/t=last × first 8 dims +
    // 전체 sum/max_abs. baseline vs hybrid path 의 누적 drift 추적.
    if std::env::var("RNB_VULKAN_LAYER_TRACE")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let data = kernels::tensor_as_f32_slice(&hidden);
        let h_dim = metadata.hidden_dim;
        let last_off = (seq_len - 1) * h_dim;
        let preview_n = h_dim.min(8);
        let preview_t0: Vec<String> = (0..preview_n).map(|d| format!("{:.6}", data[d])).collect();
        let preview_tl: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", data[last_off + d]))
            .collect();
        let total = data.len();
        let sum: f64 = data.iter().map(|&v| v as f64).sum();
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let bits_xor: u64 = data
            .iter()
            .map(|&v| v.to_bits() as u64)
            .reduce(|a, b| a ^ b)
            .unwrap_or(0);
        eprintln!(
            "[mv39:layer_trace] attn_layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
            layer_idx,
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

    Ok(PrefillAttentionLayerOutput::host(hidden))
}

enum PrefillAttentionStep {
    AttnOut(Tensor),
    PostAttentionHidden(Tensor),
    FinalHidden(Tensor),
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum AttentionStepDisposition {
    NeedsOutputProjection,
    RunFfn,
    FinishLayer,
}

#[cfg(test)]
fn attention_step_disposition(step: &PrefillAttentionStep) -> AttentionStepDisposition {
    match step {
        PrefillAttentionStep::AttnOut(_) => AttentionStepDisposition::NeedsOutputProjection,
        PrefillAttentionStep::PostAttentionHidden(_) => AttentionStepDisposition::RunFfn,
        PrefillAttentionStep::FinalHidden(_) => AttentionStepDisposition::FinishLayer,
    }
}

#[cfg(test)]
mod atn_o_tail_flow_tests {
    use super::*;

    fn tiny_tensor() -> Tensor {
        Tensor::from_vec(vec![0.0], &[1, 1])
    }

    #[test]
    fn post_attention_hidden_continues_to_ffn() {
        let step = PrefillAttentionStep::PostAttentionHidden(tiny_tensor());

        assert_eq!(
            attention_step_disposition(&step),
            AttentionStepDisposition::RunFfn
        );
    }

    #[test]
    fn final_hidden_finishes_layer() {
        let step = PrefillAttentionStep::FinalHidden(tiny_tensor());

        assert_eq!(
            attention_step_disposition(&step),
            AttentionStepDisposition::FinishLayer
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_prefill_attention_from_projection<F>(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    hidden: &Tensor,
    w: &AttentionLayerWeights,
    rope_freqs: Option<&Tensor>,
    layout: AttentionLayout,
    gemma_runtime_flavor: GemmaRuntimeFlavor,
    gemma4_reuse_q_only: bool,
    owns_kv: bool,
    kv_cache_layer: usize,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
    rope_pos_start: usize,
    head_dim: usize,
    kv_dim: usize,
    norm_eps: f32,
    projection: projection::PrefillAttentionProjection,
    prof: &F,
    t_projection: std::time::Instant,
) -> crate::error::Result<PrefillAttentionStep>
where
    F: Fn(&str, std::time::Instant),
{
    let q = projection.q;
    let k = projection.k;
    let v = projection.v;
    let attn_gate = projection.attn_gate;
    let cached_projection_kv_f16 = projection.cached_kv_f16;
    prof("qkv_gemv+norm", t_projection);

    // mv38 D / mv39: baseline q before RoPE (after q_norm). sum/max_abs 강화.
    if std::env::var("RNB_VULKAN_LAYER_TRACE")
        .map(|v| v != "0")
        .unwrap_or(false)
        && layer_idx == 3
    {
        let q_data = kernels::tensor_as_f32_slice(&q);
        let q_last_off = (seq_len - 1) * layout.q_dim;
        let preview_n = layout.q_dim.min(8);
        let q_preview_t0: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data[d]))
            .collect();
        let q_preview_tl: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data[q_last_off + d]))
            .collect();
        let q_sum: f64 = q_data.iter().map(|&v| v as f64).sum();
        let q_max_abs = q_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let q_xor: u64 = q_data
            .iter()
            .map(|&v| v.to_bits() as u64)
            .reduce(|a, b| a ^ b)
            .unwrap_or(0);
        eprintln!(
                "[mv39:q_pre_rope] baseline layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                layer_idx, preview_n, q_preview_t0.join(", "), preview_n, q_preview_tl.join(", "),
                q_sum, q_max_abs, q_xor, q_data.len(),
            );
    }

    let t0 = std::time::Instant::now();
    let kv_len = pos_start + seq_len;
    let (q, cached_kv_f16_storage, mut cached_k_tensor, mut cached_v_tensor) = if let Some((
        k_bits,
        v_bits,
    )) =
        cached_projection_kv_f16
    {
        debug_assert!(owns_kv);
        if owns_kv {
            kv_cache.replace_layer_f16_range(kv_cache_layer, pos_start, seq_len, &k_bits, &v_bits);
        }
        let cached_kv_f16_storage = if owns_kv && pos_start == 0 {
            Some((k_bits, v_bits))
        } else {
            let (k, v) = kv_cache.get_up_to(kv_cache_layer, kv_len);
            Some((k.to_vec(), v.to_vec()))
        };
        if kv_trace_enabled() {
            if let Some((cached_k_f16, cached_v_f16)) = cached_kv_f16_storage.as_ref() {
                eprintln!(
                            "[kv-trace][prefill-read] layer={} cache_layer={} kv_len={} cached_k={} cached_v={} expected={}",
                            layer_idx,
                            kv_cache_layer,
                            kv_len,
                            cached_k_f16.len(),
                            cached_v_f16.len(),
                            kv_len * kv_dim
                        );
            }
        }
        (q, cached_kv_f16_storage, None, None)
    } else {
        let (q, k_rope) = apply_prefill_rope(
            metadata,
            architecture,
            w,
            rope_freqs,
            layout,
            q,
            k.as_ref(),
            layer_idx,
            seq_len,
            rope_pos_start,
            head_dim,
            kv_dim,
        )?;

        let k_data = k_rope.as_ref().map(|t| kernels::tensor_as_f32_slice(t));
        let v_storage = if !gemma4_reuse_q_only
            && use_gemma_block_semantics(architecture)
            && super::policy::gemma_v_norm_enabled()
        {
            let raw = kernels::tensor_as_f32_slice(v.as_ref().unwrap());
            let mut normed = vec![0.0f32; raw.len()];
            for t in 0..seq_len {
                for h in 0..layout.num_kv_heads {
                    let off = t * kv_dim + h * head_dim;
                    apply_rms_norm_no_scale_into(
                        &raw[off..off + head_dim],
                        norm_eps,
                        &mut normed[off..off + head_dim],
                    );
                }
            }
            Some(normed)
        } else {
            None
        };
        let v_data_base = if gemma4_reuse_q_only {
            None
        } else {
            Some(
                v_storage
                    .as_deref()
                    .unwrap_or_else(|| kernels::tensor_as_f32_slice(v.as_ref().unwrap())),
            )
        };
        let gemma_v_rot_storage = if !gemma4_reuse_q_only
            && gemma4_should_apply_v_rotation(architecture, w.v_weight.ggml_type, head_dim)
        {
            let mut rotated = v_data_base.unwrap().to_vec();
            gemma4_apply_attn_rot_inplace(&mut rotated, head_dim, kv_dim, 64);
            Some(rotated)
        } else {
            None
        };
        let v_data = gemma_v_rot_storage
            .as_deref()
            .unwrap_or(v_data_base.unwrap_or(&[]));
        if dump_bin_dir().is_some() && !v_data.is_empty() {
            dump_bin("prefill", layer_idx, "v_normed", v_data);
        }
        if layer_idx == 0 && attn_trace_enabled() {
            let v_last = &v_data[(seq_len - 1) * kv_dim..seq_len * kv_dim];
            emit_vec_trace("prefill", layer_idx, "v_last", v_last);
        }
        // cu19: collapse the per-token `kv_cache.append` loop into a single
        // batched f32→f16 conversion + `write_bits_range` for prefill
        // (`pos_start == 0`, multi-token). Decode (`seq_len == 1`) keeps the
        // legacy per-token path because the prefill batch path also reuses the
        // converted bits for `cached_kv_f16_storage` below, which is only
        // populated when `pos_start == 0`. Reusing bits in the decode path
        // changes the bits visible to attention compute and caused a regression.
        let prefill_batch_bits: Option<(Vec<u16>, Vec<u16>)> =
            if owns_kv && pos_start == 0 && seq_len > 1 {
                let k_bits: Vec<u16> = k_data
                    .unwrap()
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_bits: Vec<u16> = v_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                kv_cache.append_bits_range(kv_cache_layer, pos_start, seq_len, &k_bits, &v_bits);
                Some((k_bits, v_bits))
            } else {
                if owns_kv {
                    for t in 0..seq_len {
                        let k_slice = &k_data.unwrap()[t * kv_dim..(t + 1) * kv_dim];
                        let v_slice = &v_data[t * kv_dim..(t + 1) * kv_dim];
                        kv_cache.append(kv_cache_layer, pos_start + t, k_slice, v_slice);
                    }
                }
                None
            };

        // mt93 step5: when RNB_ATTN_FP32 is set, suppress the arch-default
        // f16 KV cache round-trip so attention compute receives K/V as fp32.
        // Production default OFF; flag is wired only for the drift probe.
        let attn_fp32_override = super::policy::attn_fp32_enabled();
        let prefill_force_f16 = !attn_fp32_override
            && (super::policy::prefill_f16_cache_requested()
                || gemma4_prefill_uses_f16_cache(architecture));
        let cached_kv_f16_storage = if owns_kv && pos_start == 0 && !prefill_force_f16 {
            None
        } else {
            let (cached_k_f16, cached_v_f16) = if owns_kv && pos_start == 0 {
                if let Some((k_bits, v_bits)) = prefill_batch_bits {
                    (k_bits, v_bits)
                } else {
                    let cached_k_owned: Vec<u16> = k_data
                        .unwrap()
                        .iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect();
                    let cached_v_owned: Vec<u16> = v_data
                        .iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect();
                    (cached_k_owned, cached_v_owned)
                }
            } else {
                let (k, v) = kv_cache.get_up_to(kv_cache_layer, kv_len);
                (k.to_vec(), v.to_vec())
            };
            if kv_trace_enabled() {
                eprintln!(
                            "[kv-trace][prefill-read] layer={} cache_layer={} kv_len={} cached_k={} cached_v={} expected={}",
                            layer_idx,
                            kv_cache_layer,
                            kv_len,
                            cached_k_f16.len(),
                            cached_v_f16.len(),
                            kv_len * kv_dim
                        );
            }
            Some((cached_k_f16, cached_v_f16))
        };
        let cached_k_tensor = if cached_kv_f16_storage.is_none() {
            Some(Tensor::from_slice(k_data.unwrap(), &[kv_len, kv_dim]))
        } else {
            None
        };
        let cached_v_tensor = if cached_kv_f16_storage.is_none() {
            Some(Tensor::from_slice(v_data, &[kv_len, kv_dim]))
        } else {
            None
        };
        (q, cached_kv_f16_storage, cached_k_tensor, cached_v_tensor)
    };
    let must_materialize_cached_kv = (layer_idx == 0 && attn_trace_enabled())
        || targeted_attn_trace_enabled(layer_idx)
        || (gemma4_reuse_q_only && super::policy::gemma_reused_reapply_k_norm_enabled());
    if must_materialize_cached_kv && cached_k_tensor.is_none() {
        let (cached_k_f16, cached_v_f16) = cached_kv_f16_storage
            .as_ref()
            .expect("cached f16 KV storage available");
        let (k_tensor, v_tensor) =
            materialize_f16_kv_tensors(cached_k_f16, cached_v_f16, kv_len, kv_dim);
        cached_k_tensor = Some(k_tensor);
        cached_v_tensor = Some(v_tensor);
    }
    if layer_idx == 0 && attn_trace_enabled() {
        let cached_k_tensor = cached_k_tensor
            .as_ref()
            .expect("cached K tensor materialized for trace");
        let cached_v_tensor = cached_v_tensor
            .as_ref()
            .expect("cached V tensor materialized for trace");
        emit_vec_trace(
            "prefill",
            layer_idx,
            "cached_k",
            kernels::tensor_as_f32_slice(&cached_k_tensor),
        );
        emit_vec_trace(
            "prefill",
            layer_idx,
            "cached_v",
            kernels::tensor_as_f32_slice(&cached_v_tensor),
        );
    }
    if targeted_attn_trace_enabled(layer_idx) {
        let cached_k_tensor = cached_k_tensor
            .as_ref()
            .expect("cached K tensor materialized for trace");
        emit_vec_trace(
            "prefill-l34",
            layer_idx,
            "cached_k",
            kernels::tensor_as_f32_slice(&cached_k_tensor),
        );
    }

    if gemma4_reuse_q_only && super::policy::gemma_reused_reapply_k_norm_enabled() {
        let cached_k_tensor_ref = cached_k_tensor
            .as_ref()
            .expect("cached K tensor materialized for Gemma4 reused K norm");
        cached_k_tensor = Some(if let Some(k_norm) = &w.k_norm {
            let k_norm_data = kernels::tensor_as_f32_slice(k_norm);
            let cached = kernels::tensor_as_f32_slice(cached_k_tensor_ref);
            let mut out = vec![0.0f32; cached.len()];
            let total_heads = kv_len * layout.num_kv_heads;
            for h in 0..total_heads {
                let off = h * head_dim;
                apply_model_qk_norm_into(
                    &cached[off..off + head_dim],
                    k_norm_data,
                    norm_eps,
                    &mut out[off..off + head_dim],
                    architecture,
                );
            }
            Tensor::from_vec(out, &[kv_len, kv_dim])
        } else {
            cached_k_tensor
                .take()
                .expect("cached K tensor materialized")
        });
    }
    let cached_kv_f16 = if cached_k_tensor.is_none() && cached_v_tensor.is_none() {
        cached_kv_f16_storage
            .as_ref()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
    } else {
        None
    };

    prof("rope+kv_cache", t0);

    let t0 = std::time::Instant::now();
    let sliding_window = active_sliding_window(metadata, architecture, layer_idx);
    let has_softcap = resolve_attention_softcap(architecture).is_some();
    if let Some((cached_k_f16, cached_v_f16)) = cached_kv_f16 {
        if let Some(chained_hidden) = try_prefill_f16kv_attention_output_ffn_chain(
            metadata,
            architecture,
            gemma_runtime_flavor,
            &hidden,
            w,
            &q,
            cached_k_f16,
            cached_v_f16,
            layout,
            layer_idx,
            seq_len,
            kv_len,
            norm_eps,
            sliding_window,
            has_softcap,
        )? {
            prof("attention+o_proj+ffn_chain", t0);
            return Ok(PrefillAttentionStep::FinalHidden(chained_hidden));
        }
    }

    let attn_out = compute_prefill_attention(
        metadata,
        architecture,
        w,
        layout,
        &q,
        cached_k_tensor.as_ref(),
        cached_v_tensor.as_ref(),
        cached_kv_f16,
        attn_gate.as_ref(),
        layer_idx,
        seq_len,
        pos_start,
        kv_len,
    )?;
    prof("attention", t0);
    // mv38 D / mv39: baseline attn_out + q trace (layer 3 만, sum/max_abs 강화).
    if std::env::var("RNB_VULKAN_LAYER_TRACE")
        .map(|v| v != "0")
        .unwrap_or(false)
        && layer_idx == 3
    {
        let data = kernels::tensor_as_f32_slice(&attn_out);
        let last_off = (seq_len - 1) * layout.q_dim;
        let preview_n = layout.q_dim.min(8);
        let preview_t0: Vec<String> = (0..preview_n).map(|d| format!("{:.6}", data[d])).collect();
        let preview_tl: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", data[last_off + d]))
            .collect();
        let attn_sum: f64 = data.iter().map(|&v| v as f64).sum();
        let attn_max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let attn_xor: u64 = data
            .iter()
            .map(|&v| v.to_bits() as u64)
            .reduce(|a, b| a ^ b)
            .unwrap_or(0);
        eprintln!(
                "[mv39:attn_out_trace] baseline layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                layer_idx, preview_n, preview_t0.join(", "), preview_n, preview_tl.join(", "),
                attn_sum, attn_max_abs, attn_xor, data.len(),
            );
        let q_data = kernels::tensor_as_f32_slice(&q);
        let q_last_off = (seq_len - 1) * layout.q_dim;
        let q_preview_t0: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data[d]))
            .collect();
        let q_preview_tl: Vec<String> = (0..preview_n)
            .map(|d| format!("{:.6}", q_data[q_last_off + d]))
            .collect();
        let q_sum: f64 = q_data.iter().map(|&v| v as f64).sum();
        let q_max_abs = q_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let q_xor: u64 = q_data
            .iter()
            .map(|&v| v.to_bits() as u64)
            .reduce(|a, b| a ^ b)
            .unwrap_or(0);
        eprintln!(
                "[mv39:q_trace] baseline layer={} t=0[0..{}]=[{}] t=last[0..{}]=[{}] sum={:.6} max_abs={:.6e} xor=0x{:016x} elem={}",
                layer_idx, preview_n, q_preview_t0.join(", "), preview_n, q_preview_tl.join(", "),
                q_sum, q_max_abs, q_xor, q_data.len(),
            );
    }
    Ok(PrefillAttentionStep::AttnOut(attn_out))
}

fn materialize_f16_kv_tensors(
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    kv_len: usize,
    kv_dim: usize,
) -> (Tensor, Tensor) {
    let cached_k_f32 = cached_k_f16
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect::<Vec<_>>();
    let cached_v_f32 = cached_v_f16
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect::<Vec<_>>();
    (
        Tensor::from_vec(cached_k_f32, &[kv_len, kv_dim]),
        Tensor::from_vec(cached_v_f32, &[kv_len, kv_dim]),
    )
}
