#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(feature = "metal")]
use crate::engine::quantized_weight_types::backend_ggml_type;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use crate::engine::quantized_weight_types::QuantizedWeight;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct GemmaPrefillLayerRangeSpec<'a> {
    pub layer_idx: usize,
    pub attn_norm_w: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub rope_freq_factors: Option<Vec<f32>>,
    pub v_from_k: bool,
    pub q_weight: &'a QuantizedWeight,
    pub k_weight: &'a QuantizedWeight,
    pub v_weight: &'a QuantizedWeight,
    pub o_weight: &'a QuantizedWeight,
    pub post_attn_norm_w: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub post_ffn_norm_w: Vec<f32>,
    pub out_scale: Option<f32>,
    pub ffn_gate_weight: &'a QuantizedWeight,
    pub ffn_up_weight: &'a QuantizedWeight,
    pub ffn_down_weight: &'a QuantizedWeight,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub sliding_window: Option<usize>,
    pub softcap: Option<f32>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[derive(Clone, Copy)]
pub(in crate::engine) struct MusePrefillLayerRangeTarget {
    pub sequence_epoch: u64,
    pub cache_layer: usize,
    pub pos_start: usize,
    pub kv_len: usize,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MusePrefillLayerRangeSpec<'a> {
    pub layer_idx: usize,
    pub target: Option<MusePrefillLayerRangeTarget>,
    pub hidden: &'a [f32],
    pub attn_norm_w: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub post_attn_norm_w: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub post_ffn_norm_w: Vec<f32>,
    pub q_weight: &'a QuantizedWeight,
    pub k_weight: &'a QuantizedWeight,
    pub v_weight: &'a QuantizedWeight,
    pub attention_gate_weight: &'a QuantizedWeight,
    pub o_weight: &'a QuantizedWeight,
    pub ffn_gate_weight: &'a QuantizedWeight,
    pub ffn_up_weight: &'a QuantizedWeight,
    pub ffn_down_weight: &'a QuantizedWeight,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub post_norm_eps: f32,
    pub apply_rope: bool,
    pub sliding_window: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_dflash_attention_if_supported(
    query: &[f32],
    context_key: &[u16],
    context_value: &[u16],
    block_key: &[f32],
    block_value: &[f32],
    seq_len: usize,
    position: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
) -> crate::error::Result<Option<Vec<f32>>> {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_dflash_attention_if_supported(
            query,
            context_key,
            context_value,
            block_key,
            block_value,
            seq_len,
            position,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (
            query,
            context_key,
            context_value,
            block_key,
            block_value,
            seq_len,
            position,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_gemma_prefill_qkv_o_tail_if_supported(
    normed: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    rope_freq_factors: Option<&[f32]>,
    v_from_k: bool,
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    scale: f32,
    norm_eps: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<Option<metal_runtime::MetalPrefillAtnOTailOut>> {
    let (Some(q_view), Some(k_view), Some(v_view), Some(o_view)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    metal_runtime::metal_gemma_prefill_qkv_o_tail_if_supported(
        metal_runtime::MetalGemmaPrefillQkvOTailRequest {
            normed,
            q_norm_w,
            k_norm_w,
            rope_freq_factors,
            v_from_k,
            q_weight_ggml: backend_ggml_type(q_view.quant()),
            q_weight_raw: q_view.raw(),
            q_weight_rows: q_view.rows(),
            q_weight_cols: q_view.cols(),
            k_weight_ggml: backend_ggml_type(k_view.quant()),
            k_weight_raw: k_view.raw(),
            k_weight_rows: k_view.rows(),
            k_weight_cols: k_view.cols(),
            v_weight_ggml: backend_ggml_type(v_view.quant()),
            v_weight_raw: v_view.raw(),
            v_weight_rows: v_view.rows(),
            v_weight_cols: v_view.cols(),
            o_weight_ggml: backend_ggml_type(o_view.quant()),
            o_weight_raw: o_view.raw(),
            o_weight_rows: o_view.rows(),
            o_weight_cols: o_view.cols(),
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            rope_theta,
            scale,
            norm_eps,
            sliding_window,
            softcap,
        },
    )
    .map_err(crate::error::LlmError::Forward)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_gemma_prefill_qkv_o_resident_if_supported(
    normed: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    rope_freq_factors: Option<&[f32]>,
    v_from_k: bool,
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    sequence_epoch: u64,
    cache_layer: usize,
    owns_kv: bool,
    seq_len: usize,
    pos_start: usize,
    kv_len: usize,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    scale: f32,
    norm_eps: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<Option<metal_runtime::MetalPrefillAtnOTailOut>> {
    let (Some(q_view), Some(o_view)) = (q_weight.backend_view(), o_weight.backend_view()) else {
        return Ok(None);
    };
    let (k_view, v_view) = if owns_kv {
        let (Some(k_view), Some(v_view)) = (k_weight.backend_view(), v_weight.backend_view())
        else {
            return Ok(None);
        };
        (k_view, v_view)
    } else {
        (q_view, q_view)
    };
    metal_runtime::metal_gemma_prefill_qkv_o_resident_if_supported(
        metal_runtime::MetalGemmaPrefillQkvOResidentRequest {
            attention: metal_runtime::MetalGemmaPrefillQkvOTailRequest {
                normed,
                q_norm_w,
                k_norm_w,
                rope_freq_factors,
                v_from_k,
                q_weight_ggml: backend_ggml_type(q_view.quant()),
                q_weight_raw: q_view.raw(),
                q_weight_rows: q_view.rows(),
                q_weight_cols: q_view.cols(),
                k_weight_ggml: backend_ggml_type(k_view.quant()),
                k_weight_raw: k_view.raw(),
                k_weight_rows: k_view.rows(),
                k_weight_cols: k_view.cols(),
                v_weight_ggml: backend_ggml_type(v_view.quant()),
                v_weight_raw: v_view.raw(),
                v_weight_rows: v_view.rows(),
                v_weight_cols: v_view.cols(),
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_dim,
                q_dim,
                kv_dim,
                rope_theta,
                scale,
                norm_eps,
                sliding_window,
                softcap,
            },
            sequence_epoch,
            cache_layer,
            owns_kv,
            pos_start,
            kv_len,
            cached_k_f16,
            cached_v_f16,
        },
    )
    .map_err(crate::error::LlmError::Forward)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_gemma_prefill_full_layer_if_supported(
    hidden: &[f32],
    normed: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    rope_freq_factors: Option<&[f32]>,
    v_from_k: bool,
    post_attn_norm_w: &[f32],
    ffn_norm_w: &[f32],
    post_ffn_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    rope_theta: f32,
    scale: f32,
    norm_eps: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> crate::error::Result<Option<metal_runtime::MetalPrefillAtnFullLayerOut>> {
    let (
        Some(q_view),
        Some(k_view),
        Some(v_view),
        Some(o_view),
        Some(ffn_gate_view),
        Some(ffn_up_view),
        Some(ffn_down_view),
    ) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    )
    else {
        return Ok(None);
    };
    metal_runtime::metal_gemma_prefill_full_layer_if_supported(
        metal_runtime::MetalGemmaPrefillFullLayerRequest {
            attention: metal_runtime::MetalGemmaPrefillQkvOTailRequest {
                normed,
                q_norm_w,
                k_norm_w,
                rope_freq_factors,
                v_from_k,
                q_weight_ggml: backend_ggml_type(q_view.quant()),
                q_weight_raw: q_view.raw(),
                q_weight_rows: q_view.rows(),
                q_weight_cols: q_view.cols(),
                k_weight_ggml: backend_ggml_type(k_view.quant()),
                k_weight_raw: k_view.raw(),
                k_weight_rows: k_view.rows(),
                k_weight_cols: k_view.cols(),
                v_weight_ggml: backend_ggml_type(v_view.quant()),
                v_weight_raw: v_view.raw(),
                v_weight_rows: v_view.rows(),
                v_weight_cols: v_view.cols(),
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_dim,
                q_dim,
                kv_dim,
                rope_theta,
                scale,
                norm_eps,
                sliding_window,
                softcap,
            },
            hidden,
            post_attn_norm_w,
            ffn_norm_w,
            post_ffn_norm_w,
            ffn_gate_weight: metal_runtime::MetalQuantWeightRef {
                ggml_type: backend_ggml_type(ffn_gate_view.quant()),
                raw: ffn_gate_view.raw(),
                rows: ffn_gate_view.rows(),
                cols: ffn_gate_view.cols(),
            },
            ffn_up_weight: metal_runtime::MetalQuantWeightRef {
                ggml_type: backend_ggml_type(ffn_up_view.quant()),
                raw: ffn_up_view.raw(),
                rows: ffn_up_view.rows(),
                cols: ffn_up_view.cols(),
            },
            ffn_down_weight: metal_runtime::MetalQuantWeightRef {
                ggml_type: backend_ggml_type(ffn_down_view.quant()),
                raw: ffn_down_view.raw(),
                rows: ffn_down_view.rows(),
                cols: ffn_down_view.cols(),
            },
            ffn_dim: ffn_gate_view.rows(),
        },
    )
    .map_err(crate::error::LlmError::Forward)
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_muse_prepare_target_kv_resident_if_supported(
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    sequence_epoch: u64,
    cache_layer: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
    capacity_len: usize,
) -> crate::error::Result<bool> {
    metal_runtime::metal_muse_prepare_target_kv_resident_if_supported(
        cached_k_f16,
        cached_v_f16,
        sequence_epoch,
        cache_layer,
        num_kv_heads,
        head_dim,
        pos_start,
        capacity_len,
    )
    .map_err(crate::error::LlmError::Forward)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) fn metal_muse_prefill_layer_range_if_supported(
    hidden: &[f32],
    layers: &[MusePrefillLayerRangeSpec<'_>],
    feature_layers: &[usize],
    output_argmax: Option<super::metal_decode_chain::MetalDecodeOutputArgmax<'_>>,
    mut on_kv: impl FnMut(usize, &[u16], &[u16]) -> Result<(), String>,
) -> crate::error::Result<Option<metal_runtime::MetalMusePrefillLayerRangeOut>> {
    fn view(weight: &QuantizedWeight) -> Option<(rnb_loader::GGMLType, &[u8], usize, usize)> {
        let backend = weight.backend_view()?;
        Some((
            backend_ggml_type(backend.quant()),
            backend.raw(),
            backend.rows(),
            backend.cols(),
        ))
    }

    let mut runtime_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let (q_ggml, q_raw, q_rows, q_cols) = match view(layer.q_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (k_ggml, k_raw, k_rows, k_cols) = match view(layer.k_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (v_ggml, v_raw, v_rows, v_cols) = match view(layer.v_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (ag_ggml, ag_raw, ag_rows, ag_cols) = match view(layer.attention_gate_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (o_ggml, o_raw, o_rows, o_cols) = match view(layer.o_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (fg_ggml, fg_raw, fg_rows, fg_cols) = match view(layer.ffn_gate_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (fu_ggml, fu_raw, fu_rows, fu_cols) = match view(layer.ffn_up_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        let (fd_ggml, fd_raw, fd_rows, fd_cols) = match view(layer.ffn_down_weight) {
            Some(value) => value,
            None => return Ok(None),
        };
        runtime_layers.push(metal_runtime::MetalMusePrefillLayerRangeLayer {
            layer_idx: layer.layer_idx,
            target: layer
                .target
                .map(|target| metal_runtime::MetalMusePrefillLayerRangeTarget {
                    sequence_epoch: target.sequence_epoch,
                    cache_layer: target.cache_layer,
                    pos_start: target.pos_start,
                    kv_len: target.kv_len,
                }),
            request: metal_runtime::MetalMusePrefillFullLayerRequest {
                hidden: layer.hidden,
                attn_norm_w: &layer.attn_norm_w,
                q_norm_w: &layer.q_norm_w,
                k_norm_w: &layer.k_norm_w,
                post_attn_norm_w: &layer.post_attn_norm_w,
                ffn_norm_w: &layer.ffn_norm_w,
                post_ffn_norm_w: &layer.post_ffn_norm_w,
                q_weight_ggml: q_ggml,
                q_weight_raw: q_raw,
                q_weight_rows: q_rows,
                q_weight_cols: q_cols,
                k_weight_ggml: k_ggml,
                k_weight_raw: k_raw,
                k_weight_rows: k_rows,
                k_weight_cols: k_cols,
                v_weight_ggml: v_ggml,
                v_weight_raw: v_raw,
                v_weight_rows: v_rows,
                v_weight_cols: v_cols,
                attention_gate_weight_ggml: ag_ggml,
                attention_gate_weight_raw: ag_raw,
                attention_gate_weight_rows: ag_rows,
                attention_gate_weight_cols: ag_cols,
                o_weight_ggml: o_ggml,
                o_weight_raw: o_raw,
                o_weight_rows: o_rows,
                o_weight_cols: o_cols,
                ffn_gate_weight_ggml: fg_ggml,
                ffn_gate_weight_raw: fg_raw,
                ffn_gate_weight_rows: fg_rows,
                ffn_gate_weight_cols: fg_cols,
                ffn_up_weight_ggml: fu_ggml,
                ffn_up_weight_raw: fu_raw,
                ffn_up_weight_rows: fu_rows,
                ffn_up_weight_cols: fu_cols,
                ffn_down_weight_ggml: fd_ggml,
                ffn_down_weight_raw: fd_raw,
                ffn_down_weight_rows: fd_rows,
                ffn_down_weight_cols: fd_cols,
                seq_len: layer.seq_len,
                num_heads: layer.num_heads,
                num_kv_heads: layer.num_kv_heads,
                head_dim: layer.head_dim,
                hidden_dim: layer.hidden_dim,
                q_dim: layer.q_dim,
                kv_dim: layer.kv_dim,
                ffn_dim: layer.ffn_dim,
                rope_theta: layer.rope_theta,
                scale: layer.scale,
                norm_eps: layer.norm_eps,
                post_norm_eps: layer.post_norm_eps,
                apply_rope: layer.apply_rope,
                sliding_window: layer.sliding_window,
            },
        });
    }
    let output_argmax = match output_argmax {
        Some(tail) => {
            let Some(view) = tail.output_weight.backend_view() else {
                return Ok(None);
            };
            Some(metal_runtime::MetalDecodeOutputArgmax {
                norm_weight: tail.norm_weight,
                output_ggml: backend_ggml_type(view.quant()),
                output_raw: view.raw(),
                rows: tail.rows,
                cols: tail.cols,
                eps: tail.eps,
                excluded_token: tail.excluded_token,
            })
        }
        None => None,
    };
    metal_runtime::metal_muse_prefill_layer_range_if_supported(
        hidden,
        &runtime_layers,
        feature_layers,
        output_argmax,
        |layer, k, v| on_kv(layer, k, v),
    )
    .map_err(crate::error::LlmError::Forward)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) fn metal_gemma_prefill_layer_range_if_supported(
    hidden: &[f32],
    layers: &[GemmaPrefillLayerRangeSpec<'_>],
    mut on_kv: impl FnMut(usize, &[u16], &[u16]) -> Result<(), String>,
) -> crate::error::Result<Option<metal_runtime::MetalGemmaPrefillLayerRangeOut>> {
    fn runtime_view(weight: &QuantizedWeight) -> Option<metal_runtime::MetalQuantWeightRef<'_>> {
        let view = weight.backend_view()?;
        Some(metal_runtime::MetalQuantWeightRef {
            ggml_type: backend_ggml_type(view.quant()),
            raw: view.raw(),
            rows: view.rows(),
            cols: view.cols(),
        })
    }

    let mut runtime_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let (
            Some(q_weight),
            Some(k_weight),
            Some(v_weight),
            Some(o_weight),
            Some(ffn_gate_weight),
            Some(ffn_up_weight),
            Some(ffn_down_weight),
        ) = (
            runtime_view(layer.q_weight),
            runtime_view(layer.k_weight),
            runtime_view(layer.v_weight),
            runtime_view(layer.o_weight),
            runtime_view(layer.ffn_gate_weight),
            runtime_view(layer.ffn_up_weight),
            runtime_view(layer.ffn_down_weight),
        )
        else {
            return Ok(None);
        };
        runtime_layers.push(metal_runtime::MetalGemmaPrefillLayerRangeLayer {
            layer_idx: layer.layer_idx,
            attn_norm_w: &layer.attn_norm_w,
            q_norm_w: &layer.q_norm_w,
            k_norm_w: &layer.k_norm_w,
            rope_freq_factors: layer.rope_freq_factors.as_deref(),
            v_from_k: layer.v_from_k,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            post_attn_norm_w: &layer.post_attn_norm_w,
            ffn_norm_w: &layer.ffn_norm_w,
            post_ffn_norm_w: &layer.post_ffn_norm_w,
            out_scale: layer.out_scale,
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            seq_len: layer.seq_len,
            num_heads: layer.num_heads,
            num_kv_heads: layer.num_kv_heads,
            head_dim: layer.head_dim,
            hidden_dim: layer.hidden_dim,
            q_dim: layer.q_dim,
            kv_dim: layer.kv_dim,
            ffn_dim: layer.ffn_dim,
            rope_theta: layer.rope_theta,
            scale: layer.scale,
            norm_eps: layer.norm_eps,
            sliding_window: layer.sliding_window,
            softcap: layer.softcap,
        });
    }
    metal_runtime::metal_gemma_prefill_layer_range_if_supported(
        hidden,
        &runtime_layers,
        |layer_idx, k_bits, v_bits| on_kv(layer_idx, k_bits, v_bits),
    )
    .map_err(crate::error::LlmError::Forward)
}

/// pm48 ②: Metal prefill attention 2차 device-resident chain seam(rope/qk_norm→cast→flash 단일
/// command buffer). 입력(host): q_proj(gate split 후, norm 전), k_proj(norm 전), v(f32),
/// q_norm/k_norm weight. 반환 `(attn_out, k_f16, v_f16)`. Metal 전용(CUDA 미지원 → None).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_attn_chain_if_supported(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    eps: f32,
    pos_start: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
    layer_idx: usize,
    timing_enabled: bool,
) -> Option<(Vec<f32>, Vec<u16>, Vec<u16>)> {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_attn_chain_if_supported(
            q_in,
            k_in,
            v_in,
            q_norm_w,
            k_norm_w,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            n_rot,
            theta,
            eps,
            pos_start,
            scale,
            has_sliding_window,
            has_softcap,
            layer_idx,
            timing_enabled,
        );
    }
    #[cfg(not(feature = "metal"))]
    None
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_timing_enabled() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_full_timing_enabled();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_expected_dense_layer() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_expected_dense_layer();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_core_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_core_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_full_layer_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_full_layer_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_skip() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_skip();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_adapter_reject() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_adapter_reject();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_record_backend_err() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_record_backend_err();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_full_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_full_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_full_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_timing_enabled() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_o_tail_timing_enabled();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_requested() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_prefill_atn_o_tail_requested();
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_expected_dense_layer() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_expected_dense_layer();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_hit() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_hit();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_skip() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_skip();
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_adapter_reject() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_adapter_reject();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_record_backend_err() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_record_backend_err();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_prefill_atn_o_tail_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct MetalPrefillAtnCoreShape {
    pub(in crate::engine) seq_len: usize,
    pub(in crate::engine) num_heads: usize,
    pub(in crate::engine) num_kv_heads: usize,
    pub(in crate::engine) head_dim: usize,
    pub(in crate::engine) hidden_dim: usize,
    pub(in crate::engine) q_dim: usize,
    pub(in crate::engine) kv_dim: usize,
    pub(in crate::engine) n_rot: usize,
    pub(in crate::engine) rope_theta: f32,
    pub(in crate::engine) scale: f32,
    pub(in crate::engine) norm_eps: f32,
    pub(in crate::engine) pos_start: usize,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnCoreAdapterOut {
    pub(in crate::engine) attn_out: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnFullLayerAdapterOut {
    pub(in crate::engine) hidden: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) struct MetalPrefillAtnOTailAdapterOut {
    pub(in crate::engine) hidden: Vec<f32>,
    pub(in crate::engine) k_bits: Vec<u16>,
    pub(in crate::engine) v_bits: Vec<u16>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_core_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnCoreAdapterOut>> {
    let (Some(q_view), Some(k_view), Some(v_view)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
    ) else {
        metal_prefill_atn_full_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_core_if_supported(
            metal_runtime::MetalPrefillAtnCoreRequest {
                hidden,
                attn_norm_w,
                q_norm_w,
                k_norm_w,
                q_weight_ggml: backend_ggml_type(q_view.quant()),
                q_weight_raw: q_view.raw(),
                q_weight_rows: q_view.rows(),
                q_weight_cols: q_view.cols(),
                k_weight_ggml: backend_ggml_type(k_view.quant()),
                k_weight_raw: k_view.raw(),
                k_weight_rows: k_view.rows(),
                k_weight_cols: k_view.cols(),
                v_weight_ggml: backend_ggml_type(v_view.quant()),
                v_weight_raw: v_view.raw(),
                v_weight_rows: v_view.rows(),
                v_weight_cols: v_view.cols(),
                seq_len: shape.seq_len,
                num_heads: shape.num_heads,
                num_kv_heads: shape.num_kv_heads,
                head_dim: shape.head_dim,
                hidden_dim: shape.hidden_dim,
                q_dim: shape.q_dim,
                kv_dim: shape.kv_dim,
                n_rot: shape.n_rot,
                rope_theta: shape.rope_theta,
                scale: shape.scale,
                norm_eps: shape.norm_eps,
                pos_start: shape.pos_start,
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnCoreAdapterOut {
            attn_out: out.attn_out,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            shape,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_o_tail_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnOTailAdapterOut>> {
    let (Some(q_view), Some(k_view), Some(v_view), Some(o_view)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
    ) else {
        metal_prefill_atn_o_tail_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_o_tail_if_supported(
            metal_runtime::MetalPrefillAtnOTailRequest {
                core: metal_runtime::MetalPrefillAtnCoreRequest {
                    hidden,
                    attn_norm_w,
                    q_norm_w,
                    k_norm_w,
                    q_weight_ggml: backend_ggml_type(q_view.quant()),
                    q_weight_raw: q_view.raw(),
                    q_weight_rows: q_view.rows(),
                    q_weight_cols: q_view.cols(),
                    k_weight_ggml: backend_ggml_type(k_view.quant()),
                    k_weight_raw: k_view.raw(),
                    k_weight_rows: k_view.rows(),
                    k_weight_cols: k_view.cols(),
                    v_weight_ggml: backend_ggml_type(v_view.quant()),
                    v_weight_raw: v_view.raw(),
                    v_weight_rows: v_view.rows(),
                    v_weight_cols: v_view.cols(),
                    seq_len: shape.seq_len,
                    num_heads: shape.num_heads,
                    num_kv_heads: shape.num_kv_heads,
                    head_dim: shape.head_dim,
                    hidden_dim: shape.hidden_dim,
                    q_dim: shape.q_dim,
                    kv_dim: shape.kv_dim,
                    n_rot: shape.n_rot,
                    rope_theta: shape.rope_theta,
                    scale: shape.scale,
                    norm_eps: shape.norm_eps,
                    pos_start: shape.pos_start,
                },
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnOTailAdapterOut {
            hidden: out.hidden,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            o_view,
            shape,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_prefill_atn_full_layer_if_supported(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    o_weight: &QuantizedWeight,
    ffn_norm_w: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    shape: MetalPrefillAtnCoreShape,
) -> crate::error::Result<Option<MetalPrefillAtnFullLayerAdapterOut>> {
    let (
        Some(q_view),
        Some(k_view),
        Some(v_view),
        Some(o_view),
        Some(ffn_gate_view),
        Some(ffn_up_view),
        Some(ffn_down_view),
    ) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    )
    else {
        metal_prefill_atn_full_record_adapter_reject();
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let out = metal_runtime::metal_prefill_atn_full_layer_if_supported(
            metal_runtime::MetalPrefillAtnFullLayerRequest {
                core: metal_runtime::MetalPrefillAtnCoreRequest {
                    hidden,
                    attn_norm_w,
                    q_norm_w,
                    k_norm_w,
                    q_weight_ggml: backend_ggml_type(q_view.quant()),
                    q_weight_raw: q_view.raw(),
                    q_weight_rows: q_view.rows(),
                    q_weight_cols: q_view.cols(),
                    k_weight_ggml: backend_ggml_type(k_view.quant()),
                    k_weight_raw: k_view.raw(),
                    k_weight_rows: k_view.rows(),
                    k_weight_cols: k_view.cols(),
                    v_weight_ggml: backend_ggml_type(v_view.quant()),
                    v_weight_raw: v_view.raw(),
                    v_weight_rows: v_view.rows(),
                    v_weight_cols: v_view.cols(),
                    seq_len: shape.seq_len,
                    num_heads: shape.num_heads,
                    num_kv_heads: shape.num_kv_heads,
                    head_dim: shape.head_dim,
                    hidden_dim: shape.hidden_dim,
                    q_dim: shape.q_dim,
                    kv_dim: shape.kv_dim,
                    n_rot: shape.n_rot,
                    rope_theta: shape.rope_theta,
                    scale: shape.scale,
                    norm_eps: shape.norm_eps,
                    pos_start: shape.pos_start,
                },
                o_weight_ggml: backend_ggml_type(o_view.quant()),
                o_weight_raw: o_view.raw(),
                o_weight_rows: o_view.rows(),
                o_weight_cols: o_view.cols(),
                ffn_norm_w,
                ffn_gate_weight_ggml: backend_ggml_type(ffn_gate_view.quant()),
                ffn_gate_weight_raw: ffn_gate_view.raw(),
                ffn_gate_weight_rows: ffn_gate_view.rows(),
                ffn_gate_weight_cols: ffn_gate_view.cols(),
                ffn_up_weight_ggml: backend_ggml_type(ffn_up_view.quant()),
                ffn_up_weight_raw: ffn_up_view.raw(),
                ffn_up_weight_rows: ffn_up_view.rows(),
                ffn_up_weight_cols: ffn_up_view.cols(),
                ffn_down_weight_ggml: backend_ggml_type(ffn_down_view.quant()),
                ffn_down_weight_raw: ffn_down_view.raw(),
                ffn_down_weight_rows: ffn_down_view.rows(),
                ffn_down_weight_cols: ffn_down_view.cols(),
                ffn_dim: ffn_gate_view.rows(),
            },
        )
        .map_err(crate::error::LlmError::Forward)?;
        return Ok(out.map(|out| MetalPrefillAtnFullLayerAdapterOut {
            hidden: out.hidden,
            k_bits: out.k_bits,
            v_bits: out.v_bits,
        }));
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            attn_norm_w,
            q_norm_w,
            k_norm_w,
            q_view,
            k_view,
            v_view,
            o_view,
            ffn_norm_w,
            ffn_gate_view,
            ffn_up_view,
            ffn_down_view,
            shape,
        );
        Ok(None)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_muse_prefill_o_tail_ffn_if_supported(
    hidden: &[f32],
    attn_out: &[f32],
    post_attn_norm_w: &[f32],
    ffn_norm_w: &[f32],
    post_ffn_norm_w: &[f32],
    o_weight: &QuantizedWeight,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    post_norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    metal_muse_prefill_o_tail_ffn_input_if_supported(
        metal_runtime::MetalMusePrefillOTailInput::Attention(attn_out),
        hidden,
        post_attn_norm_w,
        ffn_norm_w,
        post_ffn_norm_w,
        o_weight,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_down_weight,
        seq_len,
        hidden_dim,
        norm_eps,
        post_norm_eps,
    )
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_muse_target_attention_o_tail_ffn_if_supported(
    query: &[f32],
    cached_k_f16: &[u16],
    attention_gate: &[f32],
    cached_v_f16: &[u16],
    sequence_epoch: u64,
    cache_layer: usize,
    pos_start: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    hidden: &[f32],
    post_attn_norm_w: &[f32],
    ffn_norm_w: &[f32],
    post_ffn_norm_w: &[f32],
    o_weight: &QuantizedWeight,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    post_norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    metal_muse_prefill_o_tail_ffn_input_if_supported(
        metal_runtime::MetalMusePrefillOTailInput::TargetAttention {
            query,
            cached_k_f16,
            cached_v_f16,
            sequence_epoch,
            cache_layer,
            pos_start,
            kv_len,
            num_heads,
            num_kv_heads,
            attention_gate,
            head_dim,
            scale,
            sliding_window,
        },
        hidden,
        post_attn_norm_w,
        ffn_norm_w,
        post_ffn_norm_w,
        o_weight,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_down_weight,
        seq_len,
        hidden_dim,
        norm_eps,
        post_norm_eps,
    )
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
fn metal_muse_prefill_o_tail_ffn_input_if_supported(
    input: metal_runtime::MetalMusePrefillOTailInput<'_>,
    hidden: &[f32],
    post_attn_norm_w: &[f32],
    ffn_norm_w: &[f32],
    post_ffn_norm_w: &[f32],
    o_weight: &QuantizedWeight,
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    seq_len: usize,
    hidden_dim: usize,
    norm_eps: f32,
    post_norm_eps: f32,
) -> crate::error::Result<Option<Vec<f32>>> {
    let (Some(o_view), Some(ffn_gate_view), Some(ffn_up_view), Some(ffn_down_view)) = (
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    ) else {
        return Ok(None);
    };
    metal_runtime::metal_muse_prefill_o_tail_ffn_if_supported(
        metal_runtime::MetalMusePrefillOTailFfnRequest {
            input,
            hidden,
            post_attn_norm_w,
            ffn_norm_w,
            post_ffn_norm_w,
            o_weight_ggml: backend_ggml_type(o_view.quant()),
            o_weight_raw: o_view.raw(),
            o_weight_rows: o_view.rows(),
            o_weight_cols: o_view.cols(),
            ffn_gate_weight_ggml: backend_ggml_type(ffn_gate_view.quant()),
            ffn_gate_weight_raw: ffn_gate_view.raw(),
            ffn_gate_weight_rows: ffn_gate_view.rows(),
            ffn_gate_weight_cols: ffn_gate_view.cols(),
            ffn_up_weight_ggml: backend_ggml_type(ffn_up_view.quant()),
            ffn_up_weight_raw: ffn_up_view.raw(),
            ffn_up_weight_rows: ffn_up_view.rows(),
            ffn_up_weight_cols: ffn_up_view.cols(),
            ffn_down_weight_ggml: backend_ggml_type(ffn_down_view.quant()),
            ffn_down_weight_raw: ffn_down_view.raw(),
            ffn_down_weight_rows: ffn_down_view.rows(),
            ffn_down_weight_cols: ffn_down_view.cols(),
            seq_len,
            q_dim: o_view.cols(),
            hidden_dim,
            ffn_dim: ffn_gate_view.rows(),
            norm_eps,
            post_norm_eps,
        },
    )
    .map(|result| result.map(|output| output.hidden))
    .map_err(crate::error::LlmError::Forward)
}
