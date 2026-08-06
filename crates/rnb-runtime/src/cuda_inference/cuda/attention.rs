use rnb_loader::GGMLType;

use super::{backend, Result};

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_hd256_if_supported(
    layer_index: Option<usize>,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> Option<Result<Vec<f32>>> {
    if has_softcap || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return None;
    }
    let window_range = sliding_window
        .filter(|window| *window > 0 && *window < kv_len)
        .map(|window| (kv_len - window, window));
    if window_range.is_some() && !backend::tuning::decode_attention_sliding_window_enabled() {
        return None;
    }
    if head_dim == 512 && !backend::tuning::decode_attention_hd512_enabled() {
        return None;
    }
    if backend::tuning::decode_attention_kv_cache_enabled() && matches!(head_dim, 128 | 256 | 512) {
        if let Some(layer_index) = layer_index {
            let result = if let Some((window_start, window_len)) = window_range {
                backend::attention_decode_cached_window(
                    layer_index,
                    q,
                    k,
                    v,
                    kv_len,
                    window_start,
                    window_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                )
            } else {
                backend::attention_decode_cached(
                    layer_index,
                    q,
                    k,
                    v,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                )
            };
            return Some(
                result.map_err(|err| format!("CUDA cached decode attention failed: {err}")),
            );
        }
    }
    if !backend::tuning::decode_attention_enabled() {
        return None;
    }
    let (k, v, kv_len) = if let Some((window_start, window_len)) = window_range {
        let kv_rows = num_kv_heads.checked_mul(head_dim)?;
        let start = window_start.checked_mul(kv_rows)?;
        let end = start.checked_add(window_len.checked_mul(kv_rows)?)?;
        (k.get(start..end)?, v.get(start..end)?, window_len)
    } else {
        (k, v, kv_len)
    };
    let result = match head_dim {
        128 => backend::attention_decode_hd128(q, k, v, kv_len, num_heads, num_kv_heads, scale),
        256 => backend::attention_decode_hd256(q, k, v, kv_len, num_heads, num_kv_heads, scale),
        512 => backend::attention_decode_hd512(q, k, v, kv_len, num_heads, num_kv_heads, scale),
        _ => return None,
    };
    Some(result.map_err(|err| format!("CUDA decode attention failed: {err}")))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_if_supported(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> Result<Option<Vec<f32>>> {
    // Terminal CUDA prefill provider: since the strict CUDA cutover removed the
    // CPU prefill fallback, returning None here is fatal for the caller. The
    // min-seq tuning threshold only selects between earlier fused CUDA paths;
    // it must not gate the last supported path. `prefill_flash_attention_enabled`
    // stays as an explicit diagnostic opt-out.
    if !backend::tuning::prefill_flash_attention_enabled() {
        return Ok(None);
    }
    let result = if sliding_window.is_some() || softcap.is_some() {
        backend::attention_prefill_flash_f32(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            softcap,
        )
    } else {
        match head_dim {
            128 => backend::attention_prefill_flash_hd128(
                q,
                k,
                v,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
            ),
            256 => backend::attention_prefill_flash_hd256(
                q,
                k,
                v,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
            ),
            512 => backend::attention_prefill_flash_hd512(
                q,
                k,
                v,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
            ),
            _ => backend::attention_prefill_flash_f32(
                q,
                k,
                v,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                None,
                None,
            ),
        }
    };
    result
        .map(Some)
        .map_err(|err| format!("CUDA prefill flash attention failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_non_causal_if_supported(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Option<Vec<f32>>> {
    backend::attention_prefill_flash_f32_non_causal(
        q,
        k,
        v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
    .map(Some)
    .map_err(|err| format!("CUDA non-causal prefill attention failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f16kv_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::prefill_flash_attention_enabled()
        || has_sliding_window
        || has_softcap
        || num_heads % num_kv_heads != 0
        || head_dim != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(head_dim)
    {
        return Ok(None);
    }
    backend::attention_prefill_flash_hd512_f16kv(
        q,
        k,
        v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
    )
    .map(Some)
    .map_err(|err| format!("CUDA prefill flash attention f16kv failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f16kv_window_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> Result<Option<Vec<f32>>> {
    let Some(window) = sliding_window.filter(|window| *window > 0) else {
        return Ok(None);
    };
    if !backend::tuning::prefill_flash_attention_enabled()
        || has_softcap
        || num_heads % num_kv_heads != 0
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(head_dim)
    {
        return Ok(None);
    }
    let result = match head_dim {
        256 => backend::attention_prefill_flash_hd256_f16kv_window(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
        ),
        512 => backend::attention_prefill_flash_hd512_f16kv_window(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
        ),
        _ => return Ok(None),
    };
    result
        .map(Some)
        .map_err(|err| format!("CUDA prefill flash attention f16kv window failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f16kv_dense_chain_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool> {
    if !backend::tuning::prefill_flash_attention_enabled()
        || has_sliding_window
        || has_softcap
        || num_heads % num_kv_heads != 0
        || head_dim != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(head_dim)
    {
        return Ok(false);
    }
    backend::attention_prefill_flash_hd512_f16kv_dense_chain(
        q,
        k,
        v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        scale,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        norm_eps,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|()| true)
    .map_err(|err| format!("CUDA prefill f16KV attention dense chain failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f16kv_window_dense_chain_if_supported(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool> {
    let Some(window) = sliding_window.filter(|window| *window > 0) else {
        return Ok(false);
    };
    if !backend::tuning::prefill_flash_attention_enabled()
        || has_softcap
        || num_heads % num_kv_heads != 0
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(head_dim)
    {
        return Ok(false);
    }
    let result = match head_dim {
        256 => backend::attention_prefill_flash_hd256_f16kv_window_dense_chain(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
            o,
            gate,
            up,
            down,
            down_quant as u32,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        ),
        512 => backend::attention_prefill_flash_hd512_f16kv_window_dense_chain(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
            o,
            gate,
            up,
            down,
            down_quant as u32,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        ),
        _ => return Ok(false),
    };
    result
        .map(|()| true)
        .map_err(|err| format!("CUDA prefill f16KV window attention dense chain failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn try_delta_step_if_supported(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Option<std::result::Result<Vec<f32>, String>> {
    backend::tuning::delta_net_enabled().then(|| {
        backend::delta_net_decode(
            state, q, k, v, gate, beta, num_heads, head_k_dim, head_v_dim,
        )
    })
}

/// cu203: Qwen GDN decode 층 core device chain. GGMLType 을 backend quant code 로
/// 변환해 넘긴다. conv/delta host 사본은 갱신되지 않는다 (resident 계약).
#[allow(clippy::too_many_arguments)]
pub struct QwenGdnDecodeChainCall<'a> {
    pub hidden: &'a mut [f32],
    pub conv_state: &'a mut [f32],
    pub delta_state: &'a mut [f32],
    pub attn_norm: &'a [f32],
    pub qkv_weights: &'a [u8],
    pub qkv_quant: GGMLType,
    pub gate_weights: &'a [u8],
    pub alpha_weights: &'a [f32],
    pub beta_weights: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub conv_kernel_weights: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_weights: &'a [u8],
    pub ssm_out_quant: GGMLType,
    pub n_embd: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub d_inner: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub norm_eps: f32,
}

pub fn qwen35_gdn_decode_core_chain(call: QwenGdnDecodeChainCall<'_>) -> Result<()> {
    backend::qwen35_gdn_decode_core_chain(backend::QwenGdnDecodeChainArgs {
        hidden: call.hidden,
        conv_state: call.conv_state,
        delta_state: call.delta_state,
        attn_norm: call.attn_norm,
        qkv_weights: call.qkv_weights,
        qkv_quant: call.qkv_quant as u32,
        gate_weights: call.gate_weights,
        alpha_weights: call.alpha_weights,
        beta_weights: call.beta_weights,
        dt_bias: call.dt_bias,
        ssm_a: call.ssm_a,
        conv_kernel_weights: call.conv_kernel_weights,
        ssm_norm: call.ssm_norm,
        ssm_out_weights: call.ssm_out_weights,
        ssm_out_quant: call.ssm_out_quant as u32,
        n_embd: call.n_embd,
        conv_channels: call.conv_channels,
        conv_kernel: call.conv_kernel,
        d_inner: call.d_inner,
        num_k_heads: call.num_k_heads,
        num_v_heads: call.num_v_heads,
        head_k_dim: call.head_k_dim,
        head_v_dim: call.head_v_dim,
        norm_eps: call.norm_eps,
    })
    .map_err(|err| format!("CUDA Qwen GDN decode chain failed: {err}"))
}
