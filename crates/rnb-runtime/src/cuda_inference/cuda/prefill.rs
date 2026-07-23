use rnb_backend_api::{
    DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, QuantFormat, ScalarType,
};
use rnb_loader::GGMLType;

use super::{backend, Result};

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
    seq_len > 1
}

fn prefill_f32_gemm_allowed(
    quant_supported: bool,
    seq_len: usize,
    rows: usize,
    cols: usize,
) -> bool {
    backend::tuning::prefill_f32_gemm_allowed(quant_supported, seq_len, rows, cols)
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
    if ggml_type == GGMLType::F32 {
        let expected_bytes = rows
            .checked_mul(cols)?
            .checked_mul(std::mem::size_of::<f32>())?;
        if bytes.len() != expected_bytes
            || (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() != 0
        {
            return Some(Err(format!(
                "CUDA prefill F32 weight layout mismatch: bytes={} expected={expected_bytes} alignment={}",
                bytes.len(),
                (bytes.as_ptr() as usize) % std::mem::align_of::<f32>()
            )));
        }
        let weights = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), rows.saturating_mul(cols))
        };
        return Some(f32_gemm_batch(weights, rows, cols, input));
    }
    if ggml_type == GGMLType::F16 {
        return Some(
            backend::f16_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill F16 GEMV failed: {err}")),
        );
    }
    if ggml_type == GGMLType::BF16 {
        return Some(
            backend::bf16_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill BF16 GEMV failed: {err}")),
        );
    }

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
        GGMLType::F32 => {
            let expected_bytes = rows
                .checked_mul(cols)?
                .checked_mul(std::mem::size_of::<f32>())?;
            if bytes.len() != expected_bytes
                || (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() != 0
            {
                return Some(Err(format!(
                    "CUDA prefill F32 weight layout mismatch: bytes={} expected={expected_bytes} alignment={}",
                    bytes.len(),
                    (bytes.as_ptr() as usize) % std::mem::align_of::<f32>()
                )));
            }
            let weights = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), rows.saturating_mul(cols))
            };
            Some(f32_gemm_batch(weights, rows, cols, input))
        }
        GGMLType::F16 => Some(
            backend::f16_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill F16 GEMV failed: {err}")),
        ),
        GGMLType::BF16 => Some(
            backend::bf16_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill BF16 GEMV failed: {err}")),
        ),
        GGMLType::Q4_0 => Some(
            backend::q4_0_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q4_0 GEMV failed: {err}")),
        ),
        GGMLType::Q4_1 => Some(
            backend::q4_1_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q4_1 GEMV failed: {err}")),
        ),
        GGMLType::Q5_0 => Some(
            backend::q5_0_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q5_0 GEMV failed: {err}")),
        ),
        GGMLType::Q5_1 => Some(
            backend::q5_1_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q5_1 GEMV failed: {err}")),
        ),
        GGMLType::Q8_0 => Some(
            backend::q8_0_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q8_0 GEMV failed: {err}")),
        ),
        GGMLType::Q8_1 => Some(
            backend::q8_1_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q8_1 GEMV failed: {err}")),
        ),
        GGMLType::Q2_K => Some(
            backend::q2k_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q2_K GEMV failed: {err}")),
        ),
        GGMLType::Q3_K => Some(
            backend::q3k_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill Q3_K GEMV failed: {err}")),
        ),
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
        GGMLType::IQ2_XXS => Some(
            backend::iq2_xxs_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill IQ2_XXS GEMV failed: {err}")),
        ),
        GGMLType::IQ2_S => Some(
            backend::iq2_s_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill IQ2_S GEMV failed: {err}")),
        ),
        GGMLType::IQ3_XXS => Some(
            backend::iq3_xxs_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill IQ3_XXS GEMV failed: {err}")),
        ),
        GGMLType::IQ4_XS => Some(
            backend::iq4_xs_gemv_batch(bytes, rows, cols, input)
                .map_err(|err| format!("CUDA prefill IQ4_XS GEMV failed: {err}")),
        ),
        GGMLType::Q8_K
        | GGMLType::IQ2_XS
        | GGMLType::IQ1_S
        | GGMLType::IQ4_NL
        | GGMLType::IQ3_S
        | GGMLType::IQ1_M
        | GGMLType::TQ1_0
        | GGMLType::TQ2_0
        | GGMLType::MXFP4
        | GGMLType::NVFP4
        | GGMLType::Q1_0
        | GGMLType::Q2_0
        | GGMLType::I8
        | GGMLType::I16
        | GGMLType::I32
        | GGMLType::I64
        | GGMLType::F64 => None,
    }
}

pub fn embedding_gather(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    token_ids: &[u32],
) -> Option<Result<Vec<f32>>> {
    let quant = match ggml_type {
        GGMLType::F32 => QuantFormat::F32,
        GGMLType::F16 => QuantFormat::F16,
        GGMLType::BF16 => QuantFormat::BF16,
        GGMLType::Q4_0 => QuantFormat::Q40,
        GGMLType::Q4_1 => QuantFormat::Q41,
        GGMLType::Q5_0 => QuantFormat::Q50,
        GGMLType::Q5_1 => QuantFormat::Q51,
        GGMLType::Q8_0 => QuantFormat::Q80,
        GGMLType::Q8_1 => QuantFormat::Q81,
        GGMLType::Q2_K => QuantFormat::Q2K,
        GGMLType::Q3_K => QuantFormat::Q3K,
        GGMLType::Q4_K => QuantFormat::Q4K,
        GGMLType::Q5_K => QuantFormat::Q5K,
        GGMLType::Q6_K => QuantFormat::Q6K,
        GGMLType::IQ2_XXS => QuantFormat::IQ2XXS,
        GGMLType::IQ2_S => QuantFormat::IQ2S,
        GGMLType::IQ3_XXS => QuantFormat::IQ3XXS,
        GGMLType::IQ4_XS => QuantFormat::IQ4XS,
        GGMLType::Q8_K
        | GGMLType::IQ2_XS
        | GGMLType::IQ1_S
        | GGMLType::IQ4_NL
        | GGMLType::IQ3_S
        | GGMLType::IQ1_M
        | GGMLType::TQ1_0
        | GGMLType::TQ2_0
        | GGMLType::MXFP4
        | GGMLType::NVFP4
        | GGMLType::Q1_0
        | GGMLType::Q2_0
        | GGMLType::I8
        | GGMLType::I16
        | GGMLType::I32
        | GGMLType::I64
        | GGMLType::F64 => return None,
    };
    Some(
        backend::quant_embedding_gather(quant, bytes, rows, cols, token_ids)
            .map_err(|err| format!("CUDA {ggml_type:?} embedding gather failed: {err}")),
    )
}

pub fn decode_gemv(
    ggml_type: GGMLType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Option<Result<Vec<f32>>> {
    let result = match ggml_type {
        GGMLType::F32 => {
            let expected_bytes = rows
                .checked_mul(cols)?
                .checked_mul(std::mem::size_of::<f32>())?;
            if bytes.len() != expected_bytes
                || (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() != 0
            {
                return Some(Err(format!(
                    "CUDA decode F32 weight layout mismatch: bytes={} expected={expected_bytes} alignment={}",
                    bytes.len(),
                    (bytes.as_ptr() as usize) % std::mem::align_of::<f32>()
                )));
            }
            let weights = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), rows.saturating_mul(cols))
            };
            backend::f32_gemm_batch(weights, rows, cols, input)
        }
        GGMLType::F16 => backend::f16_gemv(bytes, rows, cols, input),
        GGMLType::BF16 => backend::bf16_gemv(bytes, rows, cols, input),
        GGMLType::Q4_0 => backend::q4_0_gemv(bytes, rows, cols, input),
        GGMLType::Q4_1 => backend::q4_1_gemv(bytes, rows, cols, input),
        GGMLType::Q5_0 => backend::q5_0_gemv(bytes, rows, cols, input),
        GGMLType::Q5_1 => backend::q5_1_gemv(bytes, rows, cols, input),
        GGMLType::Q8_0 => backend::q8_0_gemv(bytes, rows, cols, input),
        GGMLType::Q8_1 => backend::q8_1_gemv(bytes, rows, cols, input),
        GGMLType::Q2_K => backend::q2k_gemv(bytes, rows, cols, input),
        GGMLType::Q3_K => backend::q3k_gemv(bytes, rows, cols, input),
        GGMLType::Q4_K => backend::q4k_gemv(bytes, rows, cols, input),
        GGMLType::Q5_K => backend::q5k_gemv(bytes, rows, cols, input),
        GGMLType::Q6_K => backend::q6k_gemv(bytes, rows, cols, input),
        GGMLType::IQ2_XXS => backend::iq2_xxs_gemv(bytes, rows, cols, input),
        GGMLType::IQ2_S => backend::iq2_s_gemv(bytes, rows, cols, input),
        GGMLType::IQ3_XXS => backend::iq3_xxs_gemv(bytes, rows, cols, input),
        GGMLType::IQ4_XS => backend::iq4_xs_gemv(bytes, rows, cols, input),
        GGMLType::Q8_K
        | GGMLType::IQ2_XS
        | GGMLType::IQ1_S
        | GGMLType::IQ4_NL
        | GGMLType::IQ3_S
        | GGMLType::IQ1_M
        | GGMLType::TQ1_0
        | GGMLType::TQ2_0
        | GGMLType::MXFP4
        | GGMLType::NVFP4
        | GGMLType::Q1_0
        | GGMLType::Q2_0
        | GGMLType::I8
        | GGMLType::I16
        | GGMLType::I32
        | GGMLType::I64
        | GGMLType::F64 => return None,
    };
    Some(result.map_err(|err| format!("CUDA decode {ggml_type:?} GEMV failed: {err}")))
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
    let touch_resident_hit = label.starts_with("nemotron_");
    match ggml_type {
        GGMLType::Q4_K if touch_resident_hit => {
            backend::q4k_gemv_into_touch_hit(raw, rows, cols, input, output)
                .map_err(|err| format!("CUDA {label} Q4_K GEMV failed: {err}"))?;
        }
        GGMLType::Q4_K => {
            backend::q4k_gemv_into(raw, rows, cols, input, output)
                .map_err(|err| format!("CUDA {label} Q4_K GEMV failed: {err}"))?;
        }
        GGMLType::Q6_K if touch_resident_hit => {
            backend::q6k_gemv_into_touch_hit(raw, rows, cols, input, output)
                .map_err(|err| format!("CUDA {label} Q6_K GEMV failed: {err}"))?;
        }
        GGMLType::Q6_K => {
            backend::q6k_gemv_into(raw, rows, cols, input, output)
                .map_err(|err| format!("CUDA {label} Q6_K GEMV failed: {err}"))?;
        }
        GGMLType::I32 => return Ok(false),
        _ => {
            let values = decode_gemv(ggml_type, raw, rows, cols, input)
                .ok_or_else(|| format!("CUDA {label} {ggml_type:?} GEMV is unsupported"))??;
            if values.len() != rows || output.len() < rows {
                return Err(format!(
                    "CUDA {label} {ggml_type:?} GEMV output mismatch: got {} expected {rows}",
                    values.len()
                ));
            }
            output[..rows].copy_from_slice(&values);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};

    const ALL_WEIGHT_TYPES: [(GGMLType, DequantType, usize, usize); 18] = [
        (GGMLType::F32, DequantType::F32, 4, 32),
        (GGMLType::F16, DequantType::F16, 2, 32),
        (GGMLType::BF16, DequantType::BF16, 2, 32),
        (GGMLType::Q4_0, DequantType::Q4_0, 18, 32),
        (GGMLType::Q4_1, DequantType::Q4_1, 20, 32),
        (GGMLType::Q5_0, DequantType::Q5_0, 22, 32),
        (GGMLType::Q5_1, DequantType::Q5_1, 24, 32),
        (GGMLType::Q8_0, DequantType::Q8_0, 34, 32),
        (GGMLType::Q8_1, DequantType::Q8_1, 36, 32),
        (GGMLType::Q2_K, DequantType::Q2K, 84, 256),
        (GGMLType::Q3_K, DequantType::Q3K, 110, 256),
        (GGMLType::Q4_K, DequantType::Q4K, 144, 256),
        (GGMLType::Q5_K, DequantType::Q5K, 176, 256),
        (GGMLType::Q6_K, DequantType::Q6K, 210, 256),
        (GGMLType::IQ2_XXS, DequantType::IQ2XXS, 66, 256),
        (GGMLType::IQ2_S, DequantType::IQ2S, 82, 256),
        (GGMLType::IQ3_XXS, DequantType::IQ3XXS, 98, 256),
        (GGMLType::IQ4_XS, DequantType::IQ4XS, 136, 256),
    ];

    fn half_bytes(value: f32) -> [u8; 2] {
        let bits = match value {
            0.03125 => 0x2800u16,
            0.015625 => 0x2400u16,
            -0.125 => 0xb000u16,
            _ => panic!("test fixture has no f16 encoding for {value}"),
        };
        bits.to_le_bytes()
    }

    fn make_weight_bytes(
        ggml_type: GGMLType,
        rows: usize,
        cols: usize,
        block_bytes: usize,
        block_elems: usize,
    ) -> Vec<u8> {
        if ggml_type == GGMLType::F32 {
            return (0..rows * cols)
                .flat_map(|index| {
                    (((index * 17 + 5) % 31) as f32 - 15.0)
                        .mul_add(0.03125, 0.0)
                        .to_le_bytes()
                })
                .collect();
        }
        if matches!(ggml_type, GGMLType::F16 | GGMLType::BF16) {
            const VALUES: [f32; 5] = [1.0, -1.0, 0.5, -0.5, 0.0];
            const F16_BITS: [u16; 5] = [0x3c00, 0xbc00, 0x3800, 0xb800, 0x0000];
            return (0..rows * cols)
                .flat_map(|index| {
                    let value_index = (index * 17 + 5) % VALUES.len();
                    if ggml_type == GGMLType::F16 {
                        F16_BITS[value_index].to_le_bytes()
                    } else {
                        ((VALUES[value_index].to_bits() >> 16) as u16).to_le_bytes()
                    }
                })
                .collect();
        }

        let blocks_per_row = cols / block_elems;
        let mut bytes = (0..rows * blocks_per_row * block_bytes)
            .map(|index| ((index * 29 + 7) % 251) as u8)
            .collect::<Vec<_>>();
        for block in bytes.chunks_exact_mut(block_bytes) {
            let d = half_bytes(0.03125);
            let dmin = half_bytes(0.015625);
            match ggml_type {
                GGMLType::Q4_0 | GGMLType::Q5_0 | GGMLType::Q8_0 => {
                    block[..2].copy_from_slice(&d);
                }
                GGMLType::Q4_1 | GGMLType::Q5_1 => {
                    block[..2].copy_from_slice(&d);
                    block[2..4].copy_from_slice(&half_bytes(-0.125));
                }
                GGMLType::Q8_1 => {
                    block[..2].copy_from_slice(&d);
                    block[2..4].fill(0);
                }
                GGMLType::Q2_K => {
                    block[80..82].copy_from_slice(&d);
                    block[82..84].copy_from_slice(&dmin);
                }
                GGMLType::Q3_K => block[108..110].copy_from_slice(&d),
                GGMLType::Q4_K | GGMLType::Q5_K => {
                    block[..2].copy_from_slice(&d);
                    block[2..4].copy_from_slice(&dmin);
                }
                GGMLType::Q6_K => block[208..210].copy_from_slice(&d),
                GGMLType::IQ2_XXS | GGMLType::IQ2_S | GGMLType::IQ3_XXS | GGMLType::IQ4_XS => {
                    block[..2].copy_from_slice(&d);
                }
                _ => unreachable!("non-quantized type handled above"),
            }
        }
        bytes
    }

    fn cpu_matmul(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        input
            .chunks_exact(cols)
            .flat_map(|token| {
                weights
                    .chunks_exact(cols)
                    .take(rows)
                    .map(move |row| row.iter().zip(token).map(|(w, x)| w * x).sum::<f32>())
            })
            .collect()
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} output length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.02f32.max(expected.abs() * 0.003);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label} mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn every_model_weight_type_runs_decode_and_prefill_on_cuda() {
        let rows = 5usize;
        for (ggml_type, dequant_type, block_bytes, block_elems) in ALL_WEIGHT_TYPES {
            let cols = block_elems;
            let weights = make_weight_bytes(ggml_type, rows, cols, block_bytes, block_elems);
            let dequantized = dequantize_bytes_to_f32(&weights, dequant_type);

            let decode_input = (0..cols)
                .map(|index| (((index * 13 + 3) % 23) as f32 - 11.0) * 0.015625)
                .collect::<Vec<_>>();
            let expected_decode = cpu_matmul(&dequantized, rows, cols, &decode_input);
            let actual_decode = decode_gemv(ggml_type, &weights, rows, cols, &decode_input)
                .expect("model weight type must have a CUDA decode route")
                .unwrap_or_else(|err| panic!("{ggml_type:?} CUDA decode failed: {err}"));
            assert_close(
                &format!("{ggml_type:?} decode"),
                &actual_decode,
                &expected_decode,
            );

            let seq_len = 3usize;
            let prefill_input = (0..seq_len * cols)
                .map(|index| (((index * 11 + 7) % 29) as f32 - 14.0) * 0.01171875)
                .collect::<Vec<_>>();
            let expected_prefill = cpu_matmul(&dequantized, rows, cols, &prefill_input);
            let actual_prefill = gemv_batch(ggml_type, &weights, rows, cols, &prefill_input)
                .unwrap_or_else(|| panic!("{ggml_type:?} has no CUDA prefill route"))
                .unwrap_or_else(|err| panic!("{ggml_type:?} CUDA prefill failed: {err}"));
            assert_close(
                &format!("{ggml_type:?} prefill"),
                &actual_prefill,
                &expected_prefill,
            );
        }
    }
}
