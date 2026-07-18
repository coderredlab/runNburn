use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, ScalarType};
use rnb_loader::GGMLType;

use super::{backend, dequant, dequant_type, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillQ4kF16QDenseChainDeviceOutput {
    pub output_id: DeviceTensorId,
    pub output_desc: DeviceTensorDesc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillQ4kF16QkvDenseChainDeviceOutput {
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
    pub output_id: DeviceTensorId,
    pub output_desc: DeviceTensorDesc,
}

pub fn prefill_gemv_enabled(seq_len: usize) -> bool {
    backend::tuning::prefill_gemv_enabled() && seq_len > 1
}

fn prefill_f32_gemm_allowed(
    quant_supported: bool,
    seq_len: usize,
    rows: usize,
    cols: usize,
) -> bool {
    backend::tuning::prefill_f32_gemm_allowed(quant_supported, seq_len, rows, cols)
}

fn prefill_float16_f32_gemm_allowed(seq_len: usize, rows: usize, cols: usize) -> bool {
    if !prefill_f32_gemm_allowed(true, seq_len, rows.min(8192), cols) {
        return false;
    }
    let max_rows = std::env::var("RNB_CUDA_F16_PREFILL_F32_GEMM_MAX_ROWS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("RNB_CUDA_BF16_PREFILL_F32_GEMM_MAX_ROWS")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
        })
        .unwrap_or(16_384);
    rows <= max_rows
}

fn prefill_bf16_f32_gemm_allowed(seq_len: usize, rows: usize, cols: usize) -> bool {
    if !prefill_f32_gemm_allowed(true, seq_len, rows.min(8192), cols) {
        return false;
    }
    let max_rows = std::env::var("RNB_CUDA_BF16_PREFILL_F32_GEMM_MAX_ROWS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(16_384);
    rows <= max_rows
}

fn prefill_f32_gemm_trace_enabled() -> bool {
    backend::tuning::prefill_f32_gemm_trace_enabled()
}

pub fn prefill_q4k_f16_gemm_allowed(seq_len: usize, rows: usize, cols: usize) -> bool {
    backend::tuning::prefill_q4k_f16_gemm_enabled()
        && prefill_f32_gemm_allowed(true, seq_len, rows, cols)
}

fn prefill_q4k_f16_qkv_gemm_allowed(seq_len: usize, rows: usize, cols: usize) -> bool {
    backend::tuning::prefill_q4k_f16_qkv_gemm_enabled()
        && prefill_f32_gemm_allowed(true, seq_len, rows, cols)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_gemm(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
    seq_len: usize,
) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols) {
        return Ok(None);
    }
    backend::q4k_f16_qkv_gemm_batch(
        q_weights, k_weights, v_weights, q_rows, kv_rows, cols, input,
    )
    .map_err(|err| format!("CUDA prefill Q4_K F16 QKV GEMM failed: {err}"))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_attention_hd512(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>> {
    if !prefill_q4k_f16_qkv_gemm_allowed(seq_len_from_input(input, cols)?, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(512)
        || kv_rows != num_kv_heads.saturating_mul(512)
        || num_heads % num_kv_heads != 0
        || q_norm.len() != 512
        || k_norm.len() != 512
        || seq_len_from_input(input, cols)? < backend::tuning::prefill_flash_attention_min_seq(512)
    {
        return Ok(None);
    }
    backend::q4k_f16_qkv_prefill_attention_hd512(
        q_weights,
        k_weights,
        v_weights,
        q_rows,
        kv_rows,
        cols,
        input,
        q_norm,
        k_norm,
        freq_factors,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
    )
    .map_err(|err| format!("CUDA prefill Q4_K F16 QKV attention hd512 failed: {err}"))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_attention_hd512_dense_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>)>> {
    let seq_len = seq_len_from_input(hidden_input, cols)?;
    if pos_start != 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(512)
        || kv_rows != num_kv_heads.saturating_mul(512)
        || num_heads % num_kv_heads != 0
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 512
        || k_norm.len() != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(512)
    {
        return Ok(None);
    }
    backend::q4k_f16_qkv_prefill_attention_hd512_dense_chain(
        q_weights,
        k_weights,
        v_weights,
        q_rows,
        kv_rows,
        cols,
        hidden_input,
        attn_norm_weight,
        q_norm,
        k_norm,
        freq_factors,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
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
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map_err(|err| format!("CUDA prefill Q4_K F16 QKV attention hd512 dense chain failed: {err}"))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_attention_hd512_dense_chain_device_output(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(DeviceTensorId, DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<PrefillQ4kF16QkvDenseChainDeviceOutput>> {
    let Some(seq_len) =
        seq_len_from_input_or_device(hidden_input, hidden_input_device.as_ref(), cols)?
    else {
        return Ok(None);
    };
    if pos_start != 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(512)
        || kv_rows != num_kv_heads.saturating_mul(512)
        || num_heads % num_kv_heads != 0
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 512
        || k_norm.len() != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(512)
    {
        return Ok(None);
    }
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    backend::q4k_f16_qkv_prefill_attention_hd512_dense_chain_device_output(
        q_weights,
        k_weights,
        v_weights,
        q_rows,
        kv_rows,
        cols,
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        k_norm,
        freq_factors,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        output_desc,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output| {
        output.map(
            |(k_bits, v_bits, output_id)| PrefillQ4kF16QkvDenseChainDeviceOutput {
                k_bits,
                v_bits,
                output_id,
                output_desc,
            },
        )
    })
    .map_err(|err| {
        format!("CUDA prefill Q4_K F16 QKV attention hd512 dense chain device output failed: {err}")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool> {
    if !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(512)
        || num_heads % num_kv_heads != 0
        || hidden_input.len() != seq_len.saturating_mul(cols)
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(512)
    {
        return Ok(false);
    }
    backend::q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
        q_weights,
        q_rows,
        cols,
        seq_len,
        kv_len,
        hidden_input,
        attn_norm_weight,
        q_norm,
        freq_factors,
        cached_k_f16,
        cached_v_f16,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map_err(|err| format!("CUDA prefill Q4_K F16 Q cached f16KV hd512 dense chain failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_q4k_f16_q_attention_hd512_cached_f16kv_dense_chain_device_output(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(DeviceTensorId, DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<PrefillQ4kF16QDenseChainDeviceOutput>> {
    if !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(512)
        || num_heads % num_kv_heads != 0
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 512
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(512)
    {
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len || desc.cols() != cols || desc.dtype() != ScalarType::F32 {
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Ok(None);
    }
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    backend::q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain_device_output(
        q_weights,
        q_rows,
        cols,
        seq_len,
        kv_len,
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        freq_factors,
        cached_k_f16,
        cached_v_f16,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        output_desc,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output_id| {
        output_id.map(|output_id| PrefillQ4kF16QDenseChainDeviceOutput {
            output_id,
            output_desc,
        })
    })
    .map_err(|err| {
        format!(
            "CUDA prefill Q4_K F16 Q cached f16KV hd512 dense chain device output failed: {err}"
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    window: usize,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool> {
    if window == 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(256)
        || num_heads % num_kv_heads != 0
        || hidden_input.len() != seq_len.saturating_mul(cols)
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 256
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(256)
    {
        return Ok(false);
    }
    backend::q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
        q_weights,
        q_rows,
        cols,
        seq_len,
        kv_len,
        hidden_input,
        attn_norm_weight,
        q_norm,
        cached_k_f16,
        cached_v_f16,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        window,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map_err(|err| {
        format!("CUDA prefill Q4_K F16 Q cached f16KV hd256 window dense chain failed: {err}")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_q4k_f16_q_attention_hd256_cached_f16kv_window_dense_chain_device_output(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(DeviceTensorId, DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    window: usize,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<PrefillQ4kF16QDenseChainDeviceOutput>> {
    if window == 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(256)
        || num_heads % num_kv_heads != 0
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || q_norm.len() != 256
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(256)
    {
        return Ok(None);
    }
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len || desc.cols() != cols || desc.dtype() != ScalarType::F32 {
            return Ok(None);
        }
    } else if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Ok(None);
    }
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    backend::q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain_device_output(
        q_weights,
        q_rows,
        cols,
        seq_len,
        kv_len,
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        cached_k_f16,
        cached_v_f16,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        window,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        output_desc,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output_id| {
        output_id.map(|output_id| PrefillQ4kF16QDenseChainDeviceOutput {
            output_id,
            output_desc,
        })
    })
    .map_err(|err| {
        format!(
            "CUDA prefill Q4_K F16 Q cached f16KV hd256 window dense chain device output failed: {err}"
        )
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_postprocess_hd256(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>> {
    let seq_len = seq_len_from_input(input, cols)?;
    if !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(256)
        || kv_rows != num_kv_heads.saturating_mul(256)
        || num_heads % num_kv_heads != 0
        || q_norm.len() != 256
        || k_norm.len() != 256
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(256)
    {
        return Ok(None);
    }
    backend::q4k_f16_qkv_postprocess_hd256(
        q_weights,
        k_weights,
        v_weights,
        q_rows,
        kv_rows,
        cols,
        input,
        q_norm,
        k_norm,
        num_heads,
        num_kv_heads,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
    )
    .map_err(|err| format!("CUDA prefill Q4_K F16 QKV postprocess hd256 failed: {err}"))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: GGMLType,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    window: usize,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>)>> {
    let seq_len = seq_len_from_input(hidden_input, cols)?;
    if pos_start != 0
        || window == 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(256)
        || kv_rows != num_kv_heads.saturating_mul(256)
        || o_cols != q_rows
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || num_heads % num_kv_heads != 0
        || q_norm.len() != 256
        || k_norm.len() != 256
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(256)
    {
        return Ok(None);
    }
    backend::q4k_f16_qkv_postprocess_hd256_window_dense_chain(
        q_weights,
        k_weights,
        v_weights,
        v_quant as u32,
        q_rows,
        kv_rows,
        cols,
        hidden_input,
        attn_norm_weight,
        q_norm,
        k_norm,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
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
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map_err(|err| {
        format!("CUDA prefill Q4_K F16 QKV postprocess hd256 window dense chain failed: {err}")
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prefill_q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: GGMLType,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(DeviceTensorId, DeviceTensorDesc)>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
    window: usize,
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    ple_gate: Option<&[u8]>,
    ple_proj: Option<&[u8]>,
    ple_post_norm_weight: Option<&[f32]>,
    ple_input: Option<&[f32]>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    layer_out_scale: Option<&[f32]>,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<PrefillQ4kF16QkvDenseChainDeviceOutput>> {
    let Some(seq_len) =
        seq_len_from_input_or_device(hidden_input, hidden_input_device.as_ref(), cols)?
    else {
        return Ok(None);
    };
    if pos_start != 0
        || window == 0
        || !prefill_q4k_f16_qkv_gemm_allowed(seq_len, q_rows, cols)
        || !backend::tuning::prefill_flash_attention_enabled()
        || q_rows != num_heads.saturating_mul(256)
        || kv_rows != num_kv_heads.saturating_mul(256)
        || o_cols != q_rows
        || attn_norm_weight.len() != cols
        || hidden.len() != seq_len.saturating_mul(n_embd)
        || num_heads % num_kv_heads != 0
        || q_norm.len() != 256
        || k_norm.len() != 256
        || seq_len < backend::tuning::prefill_flash_attention_min_seq(256)
    {
        return Ok(None);
    }
    let output_desc =
        DeviceTensorDesc::new(seq_len, n_embd, ScalarType::F32, DeviceTensorRole::Hidden);
    backend::q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output(
        q_weights,
        k_weights,
        v_weights,
        v_quant as u32,
        q_rows,
        kv_rows,
        cols,
        hidden_input,
        hidden_input_device,
        attn_norm_weight,
        q_norm,
        k_norm,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        norm_eps,
        q_unit_offset,
        k_unit_offset,
        v_no_scale_norm,
        window,
        o,
        gate,
        up,
        down,
        down_quant as u32,
        post_attn_norm_weight,
        ffn_norm_weight,
        post_ffn_norm_weight,
        ple_gate,
        ple_proj,
        ple_post_norm_weight,
        ple_input,
        ple_dim,
        o_cols,
        n_ff,
        n_embd,
        hidden,
        layer_out_scale,
        output_desc,
        unit_offset_attn_norm,
        unit_offset_post_attn_norm,
        unit_offset_ffn_norm,
        unit_offset_post_ffn_norm,
    )
    .map(|output| {
        output.map(|(k_bits, v_bits, output_id)| PrefillQ4kF16QkvDenseChainDeviceOutput {
            k_bits,
            v_bits,
            output_id,
            output_desc,
        })
    })
    .map_err(|err| {
        format!(
            "CUDA prefill Q4_K F16 QKV postprocess hd256 window dense chain device output failed: {err}"
        )
    })
}

fn seq_len_from_input(input: &[f32], cols: usize) -> Result<usize> {
    if cols == 0 {
        return Err("CUDA prefill input cols must be non-zero".to_string());
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "CUDA prefill input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    Ok(input.len() / cols)
}

fn seq_len_from_input_or_device(
    input: &[f32],
    input_device: Option<&(DeviceTensorId, DeviceTensorDesc)>,
    cols: usize,
) -> Result<Option<usize>> {
    if cols == 0 {
        return Err("CUDA prefill input cols must be non-zero".to_string());
    }
    if let Some((_, desc)) = input_device {
        if desc.cols() != cols || desc.dtype() != ScalarType::F32 {
            return Ok(None);
        }
        return Ok(Some(desc.rows()));
    }
    seq_len_from_input(input, cols).map(Some)
}

pub fn prefill_gemv(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    seq_len: usize,
) -> Option<Result<Vec<f32>>> {
    if !prefill_gemv_enabled(seq_len) {
        return None;
    }
    let quant_supported = matches!(ggml_type, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K);
    let f32_gemm_allowed = match ggml_type {
        GGMLType::F16 => prefill_float16_f32_gemm_allowed(seq_len, rows, cols),
        GGMLType::BF16 => prefill_bf16_f32_gemm_allowed(seq_len, rows, cols),
        _ => prefill_f32_gemm_allowed(quant_supported, seq_len, rows, cols),
    };
    let trace_route = std::env::var("RNB_CUDA_PREFILL_GEMV_TRACE").ok().as_deref() == Some("1");
    if ggml_type == GGMLType::Q4_K && prefill_q4k_f16_gemm_allowed(seq_len, rows, cols) {
        match backend::q4k_f16_gemm_batch(bytes, rows, cols, input) {
            Ok(Some(output)) => {
                if trace_route {
                    eprintln!("[prefill-gemv] route=q4k_f16 rows={rows} cols={cols} seq={seq_len}");
                }
                return Some(Ok(output));
            }
            Ok(None) => {
                if trace_route {
                    eprintln!(
                        "[prefill-gemv] route=q4k_f16_NONE rows={rows} cols={cols} seq={seq_len}"
                    );
                }
            }
            Err(err) => return Some(Err(format!("CUDA prefill Q4_K F16 GEMM failed: {err}"))),
        }
    }
    if f32_gemm_allowed {
        if ggml_type == GGMLType::Q4_K {
            match backend::q4k_f32_gemm_batch_cached(bytes, rows, cols, input) {
                Ok(Some(output)) => {
                    if trace_route {
                        eprintln!("[prefill-gemv] route=q4k_f32_cached rows={rows} cols={cols} seq={seq_len}");
                    }
                    return Some(Ok(output));
                }
                Ok(None) => {
                    if trace_route {
                        eprintln!("[prefill-gemv] route=q4k_f32_NONE_fallback rows={rows} cols={cols} seq={seq_len}");
                    }
                }
                Err(err) => return Some(Err(format!("CUDA prefill Q4_K F32 GEMM failed: {err}"))),
            }
        }
        if trace_route {
            eprintln!("[prefill-gemv] route=host_f32_dequant+gemm type={ggml_type:?} rows={rows} cols={cols} seq={seq_len}");
        }
        let trace_f32_gemm = prefill_f32_gemm_trace_enabled();
        let trace_t0 = trace_f32_gemm.then(std::time::Instant::now);
        let dequant_t0 = trace_f32_gemm.then(std::time::Instant::now);
        let weights_f32 = dequant::dequantize_bytes_to_f32(bytes, dequant_type(ggml_type));
        let dequant_ms = dequant_t0.map(|t| t.elapsed().as_micros() as f64 / 1000.0);
        if weights_f32.len() == rows * cols {
            let gpu_t0 = trace_f32_gemm.then(std::time::Instant::now);
            let result = f32_gemm_batch(&weights_f32, rows, cols, input);
            if let (Some(total_t0), Some(dequant_ms), Some(gpu_t0)) = (trace_t0, dequant_ms, gpu_t0)
            {
                eprintln!(
                    "[gpu-f32-gemm] type={:?} rows={} cols={} seq={} dequant_ms={:.1} gpu_ms={:.1} total_ms={:.1}",
                    ggml_type,
                    rows,
                    cols,
                    seq_len,
                    dequant_ms,
                    gpu_t0.elapsed().as_micros() as f64 / 1000.0,
                    total_t0.elapsed().as_micros() as f64 / 1000.0
                );
            }
            return Some(result);
        }
    }
    gemv_batch(ggml_type, bytes, rows, cols, input)
}

fn f32_gemm_batch(
    weights_f32: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    backend::f32_gemm_batch(weights_f32, rows, cols, input)
        .map_err(|err| format!("CUDA prefill f32 GEMM failed: {err}"))
}

fn gemv_batch(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Option<Result<Vec<f32>>> {
    match ggml_type {
        GGMLType::Q4_K => Some(
            backend::q4k_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q4_K GEMV failed: {err}")),
        ),
        GGMLType::Q5_K => Some(
            backend::q5k_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q5_K GEMV failed: {err}")),
        ),
        GGMLType::Q6_K => Some(
            backend::q6k_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q6_K GEMV failed: {err}")),
        ),
        GGMLType::IQ4_XS => Some(
            backend::iq4_xs_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill IQ4_XS GEMV failed: {err}")),
        ),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_gemv_into_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    label: &str,
) -> Result<bool> {
    if !backend::tuning::layer_gemv_enabled() {
        return Ok(false);
    }
    let touch_resident_hit = label.starts_with("nemotron_");
    let result = match ggml_type {
        GGMLType::Q4_K if touch_resident_hit => {
            backend::q4k_gemv_into_touch_hit(raw, rows, cols, input, output)
        }
        GGMLType::Q4_K => backend::q4k_gemv_into(raw, rows, cols, input, output),
        GGMLType::Q6_K if touch_resident_hit => {
            backend::q6k_gemv_into_touch_hit(raw, rows, cols, input, output)
        }
        GGMLType::Q6_K => backend::q6k_gemv_into(raw, rows, cols, input, output),
        _ => return Ok(false),
    };
    match result {
        Ok(()) => Ok(true),
        Err(err) => {
            eprintln!("[cuda] {label} GEMV failed, CPU fallback: {err}");
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn restore_env(key: &str, value: Option<String>) {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn float16_prefill_f32_gemm_allows_gemma4_ple_model_proj_shape() {
        let _guard = env_lock().lock().unwrap();
        let keys = [
            "RNB_CUDA_PREFILL_F32_GEMM",
            "RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ",
            "RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS",
            "RNB_CUDA_F16_PREFILL_F32_GEMM_MAX_ROWS",
            "RNB_CUDA_BF16_PREFILL_F32_GEMM_MAX_ROWS",
        ];
        let saved = keys.map(|key| (key, std::env::var(key).ok()));
        std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM", "1");
        std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ", "128");
        std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS", "4096");
        std::env::set_var("RNB_CUDA_F16_PREFILL_F32_GEMM_MAX_ROWS", "16384");
        std::env::set_var("RNB_CUDA_BF16_PREFILL_F32_GEMM_MAX_ROWS", "16384");

        let gemma4_f16_shape_allowed = prefill_float16_f32_gemm_allowed(1024, 10_752, 2560);
        let short_f16_prefill_allowed = prefill_float16_f32_gemm_allowed(64, 10_752, 2560);
        let too_tall_f16_allowed = prefill_float16_f32_gemm_allowed(1024, 20_000, 2560);
        let gemma4_shape_allowed = prefill_bf16_f32_gemm_allowed(1024, 10_752, 2560);
        let short_prefill_allowed = prefill_bf16_f32_gemm_allowed(64, 10_752, 2560);
        let too_tall_allowed = prefill_bf16_f32_gemm_allowed(1024, 20_000, 2560);

        for (key, value) in saved {
            restore_env(key, value);
        }

        assert!(gemma4_f16_shape_allowed);
        assert!(!short_f16_prefill_allowed);
        assert!(!too_tall_f16_allowed);
        assert!(gemma4_shape_allowed);
        assert!(!short_prefill_allowed);
        assert!(!too_tall_allowed);
    }
}
