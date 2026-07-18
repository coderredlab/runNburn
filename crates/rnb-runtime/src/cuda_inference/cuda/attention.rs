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
    has_sliding_window: bool,
    has_softcap: bool,
) -> Result<Option<Vec<f32>>> {
    if !backend::tuning::prefill_flash_attention_enabled()
        || has_sliding_window
        || has_softcap
        || num_heads % num_kv_heads != 0
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(head_dim)
    {
        return Ok(None);
    }
    let result = match head_dim {
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
        _ => return Ok(None),
    };
    result
        .map(Some)
        .map_err(|err| format!("CUDA prefill flash attention failed: {err}"))
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
