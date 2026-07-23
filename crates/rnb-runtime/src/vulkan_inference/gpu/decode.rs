use super::*;

pub struct DecodeGemvWeight<'a> {
    pub kind: DecodeWeightKind,
    pub raw: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub ggml_type: GGMLType,
}

fn decode_weight_id(layer_idx: usize, kind: DecodeWeightKind) -> WeightId {
    match kind {
        DecodeWeightKind::QProj => q_proj_id(layer_idx),
        DecodeWeightKind::KProj => k_proj_id(layer_idx),
        DecodeWeightKind::VProj => v_proj_id(layer_idx),
        DecodeWeightKind::OProj => backend_o_proj_id(layer_idx),
        DecodeWeightKind::FfnGate => ffn_gate_id(layer_idx),
        DecodeWeightKind::FfnDown => ffn_down_id(layer_idx),
        DecodeWeightKind::GdnQkv => backend_gdn_qkv_id(layer_idx),
        DecodeWeightKind::GdnGate => backend_gdn_gate_id(layer_idx),
        DecodeWeightKind::GdnAlpha => backend_gdn_alpha_id(layer_idx),
        DecodeWeightKind::GdnBeta => backend_gdn_beta_id(layer_idx),
        DecodeWeightKind::GdnSsmOut => backend_gdn_ssm_out_id(layer_idx),
    }
}

fn decode_gemv_batch<'a>(
    layer_idx: usize,
    weights: &'a [DecodeGemvWeight<'a>],
) -> Option<Vec<(WeightId, &'a [u8], usize, usize, Quant)>> {
    weights
        .iter()
        .map(|weight| {
            Some((
                decode_weight_id(layer_idx, weight.kind),
                weight.raw,
                weight.rows,
                weight.cols,
                ggml_to_quant(weight.ggml_type)?,
            ))
        })
        .collect()
}

pub fn decode_gemv(
    runtime: &mut Runtime,
    layer_idx: usize,
    weight: DecodeGemvWeight<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<bool, String> {
    let Some(quant) = ggml_to_quant(weight.ggml_type) else {
        return Ok(false);
    };
    runtime
        .gemv(
            decode_weight_id(layer_idx, weight.kind),
            weight.raw,
            weight.rows,
            weight.cols,
            quant,
            input,
            output,
        )
        .map(|()| true)
}

pub fn decode_gemv_multi(
    runtime: &mut Runtime,
    layer_idx: usize,
    input: &[f32],
    weights: &[DecodeGemvWeight<'_>],
    outputs: &mut [&mut [f32]],
) -> Result<bool, String> {
    let Some(weights) = decode_gemv_batch(layer_idx, weights) else {
        return Ok(false);
    };
    runtime.gemv_multi(input, &weights, outputs).map(|()| true)
}

pub fn decode_gemv_multi_same_quant(
    runtime: &mut Runtime,
    layer_idx: usize,
    input: &[f32],
    weights: &[DecodeGemvWeight<'_>],
    outputs: &mut [&mut [f32]],
) -> Result<bool, String> {
    let Some(weights) = decode_gemv_batch(layer_idx, weights) else {
        return Ok(false);
    };
    if !weights.windows(2).all(|pair| pair[0].4 == pair[1].4) {
        return Ok(false);
    }
    runtime.gemv_multi(input, &weights, outputs).map(|()| true)
}

pub fn decode_gemv_multi_async(
    runtime: &mut Runtime,
    layer_idx: usize,
    input: &[f32],
    weights: &[DecodeGemvWeight<'_>],
) -> Result<bool, String> {
    let Some(weights) = decode_gemv_batch(layer_idx, weights) else {
        return Ok(false);
    };
    runtime.gemv_multi_async(input, &weights).map(|()| true)
}

pub fn decode_wait_async(runtime: &mut Runtime, outputs: &mut [&mut [f32]]) -> Result<(), String> {
    runtime.wait_async(outputs)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_ffn_chain(
    runtime: &mut Runtime,
    layer_idx: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    norm_eps: f32,
    hidden_dim: usize,
    gate_raw: &[u8],
    gate_rows: usize,
    gate_cols: usize,
    gate_ggml_type: GGMLType,
    up_raw: &[u8],
    up_rows: usize,
    up_cols: usize,
    up_ggml_type: GGMLType,
    down_raw: &[u8],
    down_rows: usize,
    down_cols: usize,
    down_ggml_type: GGMLType,
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
    runtime
        .ffn_chain(
            hidden,
            norm_weight,
            norm_eps,
            hidden_dim,
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
        )
        .map(|()| true)
}
