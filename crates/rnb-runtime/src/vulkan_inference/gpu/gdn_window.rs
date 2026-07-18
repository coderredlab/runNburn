use super::*;

#[allow(clippy::too_many_arguments)]
pub fn gdn_qkv_conv_window_from_resident_state(
    runtime: &mut Runtime,
    layer_idx: usize,
    qkv_ggml_type: GGMLType,
    qkv_raw: &[u8],
    qkv_rows: usize,
    qkv_cols: usize,
    kernel: &[f32],
    input_all: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    conv_channels: usize,
    kernel_size: usize,
    conv_out: &mut [f32],
) -> Result<bool, String> {
    let Some(qkv_quant) = ggml_to_quant(qkv_ggml_type) else {
        return Ok(false);
    };
    runtime.gdn_qkv_conv_window_from_resident_state(
        layer_idx,
        backend_gdn_qkv_id(layer_idx),
        qkv_raw,
        qkv_rows,
        qkv_cols,
        qkv_quant,
        kernel,
        input_all,
        seq_len,
        hidden_dim,
        conv_channels,
        kernel_size,
        conv_out,
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_ffn_chain_window(
    runtime: &mut Runtime,
    layer_idx: usize,
    hidden_all: &mut [f32],
    hidden_dim: usize,
    norm_weight: &[f32],
    norm_eps: f32,
    gate_ggml_type: GGMLType,
    gate_raw: &[u8],
    gate_rows: usize,
    gate_cols: usize,
    up_ggml_type: GGMLType,
    up_raw: &[u8],
    up_rows: usize,
    up_cols: usize,
    down_ggml_type: GGMLType,
    down_raw: &[u8],
    down_rows: usize,
    down_cols: usize,
) -> Result<bool, String> {
    let Some(gate_quant) = ggml_to_quant(gate_ggml_type) else {
        return Ok(false);
    };
    let Some(up_quant) = ggml_to_quant(up_ggml_type) else {
        return Ok(false);
    };
    let Some(down_quant) = ggml_to_quant(down_ggml_type) else {
        return Ok(false);
    };
    runtime.ffn_chain_window(
        hidden_all,
        hidden_dim,
        norm_weight,
        norm_eps,
        ffn_gate_id(layer_idx),
        gate_raw,
        gate_rows,
        gate_cols,
        gate_quant,
        ffn_up_id(layer_idx),
        up_raw,
        up_rows,
        up_cols,
        up_quant,
        ffn_down_id(layer_idx),
        down_raw,
        down_rows,
        down_cols,
        down_quant,
    )?;
    Ok(true)
}
