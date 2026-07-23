//! `mk4` MediaTek NPU diagnostic GGUF projection extractor helpers.
//!
//! This module is dev tooling only. It converts selected GGUF tensor rows into
//! the temporary `.rnbmtk-mlp` smoke-test format used by the mobile-mtk-npu
//! research track.

use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};
use rnb_loader::{GGMLType, LoadedModel};
use sha2::{Digest, Sha256};

const RNBMTK_MAGIC: &[u8; 8] = b"RNBMTK3\0";
const RNBMTK_QUANT_MAGIC: &[u8; 8] = b"RNBMTKQ1";
const RNBMTK_VERSION: u32 = 1;
const DIAGNOSTIC_DIM_CAP: usize = 32768;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    pub block_elems: usize,
    pub block_bytes: usize,
    pub dequant_type: DequantType,
}

#[derive(Debug, Clone)]
pub struct MtkMlpConfig {
    pub w1_tensor: String,
    pub w2_tensor: String,
    pub input_tensor: String,
    pub input_row: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub input_scale: f32,
}

impl Default for MtkMlpConfig {
    fn default() -> Self {
        Self {
            w1_tensor: "blk.0.ffn_up.weight".to_string(),
            w2_tensor: "blk.0.ffn_down.weight".to_string(),
            input_tensor: "token_embd.weight".to_string(),
            input_row: 0,
            input_size: 256,
            hidden_size: 128,
            output_size: 64,
            input_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MtkMlpPayload {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MtkGatedGeluFfnConfig {
    pub layer: usize,
    pub input_tensor: String,
    pub input_row: usize,
    pub input_scale: f32,
}

impl Default for MtkGatedGeluFfnConfig {
    fn default() -> Self {
        Self {
            layer: 0,
            input_tensor: "token_embd.weight".to_string(),
            input_row: 0,
            input_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MtkGatedGeluFfnPayload {
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub gate_weight: Vec<f32>,
    pub up_weight: Vec<f32>,
    pub down_weight: Vec<f32>,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

#[derive(Debug, Clone)]
pub struct QuantizedMtkMlpPayload {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub input_params: QuantParams,
    pub w1_params: QuantParams,
    pub hidden_params: QuantParams,
    pub w2_params: QuantParams,
    pub output_params: QuantParams,
    pub w1: Vec<u8>,
    pub b1: Vec<i32>,
    pub w2: Vec<u8>,
    pub b2: Vec<i32>,
    pub input: Vec<u8>,
    pub expected: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizedOutputParity {
    pub max_byte_delta: u32,
    pub max_byte_delta_index: usize,
    pub max_abs_error: f32,
}

impl QuantizedOutputParity {
    pub fn passes(self, tolerance: f32) -> bool {
        self.max_byte_delta <= 1 || self.max_abs_error <= tolerance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MlpComputation<'a> {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: &'a [f32],
    pub input: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct TensorRowMetadata {
    pub name: String,
    pub ggml_type: GGMLType,
    pub shape: Vec<usize>,
    pub rows_selected: usize,
    pub row_start: usize,
    pub cols_selected: usize,
    pub bytes_per_row: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ValueRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone)]
pub struct MtkMlpMetadata {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub input_scale: f32,
    pub tensors: Vec<TensorRowMetadata>,
    pub w1_range: ValueRange,
    pub w2_range: ValueRange,
    pub input_range: ValueRange,
    pub hidden_range: ValueRange,
    pub output_range: ValueRange,
    pub payload_bytes: usize,
    pub payload_sha256: String,
}

#[derive(Debug, Clone)]
pub struct MtkGatedGeluFfnMetadata {
    pub layer: usize,
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub input_scale: f32,
    pub tensors: Vec<TensorRowMetadata>,
    pub gate_range: ValueRange,
    pub up_range: ValueRange,
    pub down_range: ValueRange,
    pub input_range: ValueRange,
    pub gated_range: ValueRange,
    pub output_range: ValueRange,
}

#[derive(Debug, Clone)]
pub struct ExtractedMtkMlp {
    pub payload: MtkMlpPayload,
    pub metadata: MtkMlpMetadata,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ExtractedMtkGatedGeluFfn {
    pub payload: MtkGatedGeluFfnPayload,
    pub metadata: MtkGatedGeluFfnMetadata,
}

pub fn row_layout_for_type(ggml_type: GGMLType) -> Result<RowLayout, String> {
    let layout = match ggml_type {
        GGMLType::F32 => RowLayout {
            block_elems: 1,
            block_bytes: 4,
            dequant_type: DequantType::F32,
        },
        GGMLType::F16 => RowLayout {
            block_elems: 1,
            block_bytes: 2,
            dequant_type: DequantType::F16,
        },
        GGMLType::BF16 => RowLayout {
            block_elems: 1,
            block_bytes: 2,
            dequant_type: DequantType::BF16,
        },
        GGMLType::Q8_0 => RowLayout {
            block_elems: 32,
            block_bytes: 34,
            dequant_type: DequantType::Q8_0,
        },
        GGMLType::Q8_1 => RowLayout {
            block_elems: 32,
            block_bytes: 36,
            dequant_type: DequantType::Q8_1,
        },
        GGMLType::Q4_K => RowLayout {
            block_elems: 256,
            block_bytes: 144,
            dequant_type: DequantType::Q4K,
        },
        GGMLType::Q5_K => RowLayout {
            block_elems: 256,
            block_bytes: 176,
            dequant_type: DequantType::Q5K,
        },
        GGMLType::Q6_K => RowLayout {
            block_elems: 256,
            block_bytes: 210,
            dequant_type: DequantType::Q6K,
        },
        other => {
            return Err(format!(
                "unsupported GGML type for mk4 extractor: {other:?}"
            ))
        }
    };
    Ok(layout)
}

pub fn bytes_per_row_checked(
    ggml_type: GGMLType,
    cols: usize,
    actual_bytes_per_row: usize,
) -> Result<usize, String> {
    if cols == 0 {
        return Err("row layout requires non-zero cols".to_string());
    }
    let layout = row_layout_for_type(ggml_type)?;
    if !cols.is_multiple_of(layout.block_elems) {
        return Err(format!(
            "{ggml_type:?} cols {cols} not divisible by block_elems {}",
            layout.block_elems
        ));
    }
    let expected = cols
        .checked_div(layout.block_elems)
        .and_then(|blocks| blocks.checked_mul(layout.block_bytes))
        .ok_or_else(|| format!("{ggml_type:?} bytes_per_row overflow for cols={cols}"))?;
    if expected != actual_bytes_per_row {
        return Err(format!(
            "{ggml_type:?} bytes_per_row mismatch: expected {expected}, got {actual_bytes_per_row}"
        ));
    }
    Ok(expected)
}

pub fn validate_matrix_dims(
    name: &str,
    shape: &[usize],
    min_rows: usize,
    min_cols: usize,
) -> Result<(usize, usize), String> {
    if shape.len() != 2 {
        return Err(format!("{name} must be 2D, got shape {shape:?}"));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows < min_rows || cols < min_cols {
        return Err(format!(
            "{name} shape {shape:?} is smaller than required rows={min_rows} cols={min_cols}"
        ));
    }
    Ok((rows, cols))
}

pub fn slice_f32_row_prefix(
    data: &[f32],
    rows: usize,
    cols: usize,
    row: usize,
    prefix_cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 {
        return Err("slice_f32_row_prefix requires non-zero shape".to_string());
    }
    if data.len()
        != rows
            .checked_mul(cols)
            .ok_or_else(|| "slice_f32_row_prefix shape overflow".to_string())?
    {
        return Err(format!(
            "slice_f32_row_prefix data length {} does not match shape {}x{}",
            data.len(),
            rows,
            cols
        ));
    }
    if row >= rows {
        return Err(format!("row {row} out of bounds for rows {rows}"));
    }
    if prefix_cols > cols {
        return Err(format!("prefix cols {prefix_cols} exceeds cols {cols}"));
    }
    let start = row * cols;
    Ok(data[start..start + prefix_cols].to_vec())
}

pub fn compute_expected(computation: MlpComputation<'_>) -> Result<Vec<f32>, String> {
    let (_, output) = compute_hidden_and_output(computation)?;
    Ok(output)
}

pub fn compute_gated_gelu_ffn_expected(
    input_size: usize,
    ffn_inner_size: usize,
    output_size: usize,
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let (_, output) = compute_gated_gelu_hidden_and_output(
        input_size,
        ffn_inner_size,
        output_size,
        gate_weight,
        up_weight,
        down_weight,
        input,
    )?;
    Ok(output)
}

pub fn write_rnbmtk_mlp(model: &MtkMlpPayload) -> Result<Vec<u8>, String> {
    validate_payload(model)?;
    let float_count = model
        .w1
        .len()
        .checked_add(model.b1.len())
        .and_then(|v| v.checked_add(model.w2.len()))
        .and_then(|v| v.checked_add(model.b2.len()))
        .and_then(|v| v.checked_add(model.input.len()))
        .and_then(|v| v.checked_add(model.expected.len()))
        .ok_or_else(|| "RNBMTK payload length overflow".to_string())?;
    let mut bytes = Vec::with_capacity(24 + float_count * 4);
    bytes.extend_from_slice(RNBMTK_MAGIC);
    push_u32(&mut bytes, RNBMTK_VERSION);
    push_usize_as_u32(&mut bytes, model.input_size)?;
    push_usize_as_u32(&mut bytes, model.hidden_size)?;
    push_usize_as_u32(&mut bytes, model.output_size)?;
    for values in [
        model.w1.as_slice(),
        model.b1.as_slice(),
        model.w2.as_slice(),
        model.b2.as_slice(),
        model.input.as_slice(),
        model.expected.as_slice(),
    ] {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

pub fn quantize_mlp_payload(model: &MtkMlpPayload) -> Result<QuantizedMtkMlpPayload, String> {
    validate_payload(model)?;
    let (hidden, fp32_output) = compute_hidden_and_output(MlpComputation {
        input_size: model.input_size,
        hidden_size: model.hidden_size,
        output_size: model.output_size,
        w1: &model.w1,
        b1: &model.b1,
        w2: &model.w2,
        b2: &model.b2,
        input: &model.input,
    })?;

    let input_params = symmetric_u8_params(&model.input, "input")?;
    let w1_params = symmetric_u8_params(&model.w1, "w1")?;
    let hidden_params = relu_u8_params(&hidden)?;
    let w2_params = symmetric_u8_params(&model.w2, "w2")?;
    let output_params = symmetric_u8_params(&fp32_output, "output")?;

    let input = quantize_slice(&model.input, input_params);
    let w1 = quantize_slice(&model.w1, w1_params);
    let w2 = quantize_slice(&model.w2, w2_params);
    let b1 = vec![0; model.hidden_size];
    let b2 = vec![0; model.output_size];
    let expected = quantized_expected(QuantizedComputation {
        input_size: model.input_size,
        hidden_size: model.hidden_size,
        output_size: model.output_size,
        input: &input,
        input_params,
        w1: &w1,
        w1_params,
        b1: &b1,
        hidden_params,
        w2: &w2,
        w2_params,
        b2: &b2,
        output_params,
    })?;

    let quantized = QuantizedMtkMlpPayload {
        input_size: model.input_size,
        hidden_size: model.hidden_size,
        output_size: model.output_size,
        input_params,
        w1_params,
        hidden_params,
        w2_params,
        output_params,
        w1,
        b1,
        w2,
        b2,
        input,
        expected,
    };
    validate_quantized_payload(&quantized)?;
    Ok(quantized)
}

pub fn write_rnbmtk_quant_mlp(model: &QuantizedMtkMlpPayload) -> Result<Vec<u8>, String> {
    validate_quantized_payload(model)?;
    let byte_count = model
        .w1
        .len()
        .checked_add(
            model
                .b1
                .len()
                .checked_mul(4)
                .ok_or("b1 byte length overflow")?,
        )
        .and_then(|v| v.checked_add(model.w2.len()))
        .and_then(|v| v.checked_add(model.b2.len().checked_mul(4)?))
        .and_then(|v| v.checked_add(model.input.len()))
        .and_then(|v| v.checked_add(model.expected.len().checked_mul(4)?))
        .ok_or_else(|| "RNBMTKQ payload length overflow".to_string())?;
    let mut bytes = Vec::with_capacity(64 + byte_count);
    bytes.extend_from_slice(RNBMTK_QUANT_MAGIC);
    push_u32(&mut bytes, RNBMTK_VERSION);
    push_usize_as_u32(&mut bytes, model.input_size)?;
    push_usize_as_u32(&mut bytes, model.hidden_size)?;
    push_usize_as_u32(&mut bytes, model.output_size)?;
    for params in [
        model.input_params,
        model.w1_params,
        model.hidden_params,
        model.w2_params,
        model.output_params,
    ] {
        push_f32(&mut bytes, params.scale);
        push_i32(&mut bytes, params.zero_point);
    }
    bytes.extend_from_slice(&model.w1);
    push_i32_slice(&mut bytes, &model.b1);
    bytes.extend_from_slice(&model.w2);
    push_i32_slice(&mut bytes, &model.b2);
    bytes.extend_from_slice(&model.input);
    for value in &model.expected {
        push_f32(&mut bytes, *value);
    }
    Ok(bytes)
}

pub fn quantized_metadata_json(model: &QuantizedMtkMlpPayload, bytes: &[u8]) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"input_size\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"output_size\": {},\n",
            "  \"payload_bytes\": {},\n",
            "  \"payload_sha256\": \"{}\",\n",
            "  \"quant_params\": {{\n",
            "    \"input\": {},\n",
            "    \"w1\": {},\n",
            "    \"hidden\": {},\n",
            "    \"w2\": {},\n",
            "    \"output\": {}\n",
            "  }},\n",
            "  \"expected_range\": {}\n",
            "}}\n"
        ),
        model.input_size,
        model.hidden_size,
        model.output_size,
        bytes.len(),
        sha256_hex(bytes),
        quant_params_json(model.input_params),
        quant_params_json(model.w1_params),
        quant_params_json(model.hidden_params),
        quant_params_json(model.w2_params),
        quant_params_json(model.output_params),
        value_range(&model.expected)
            .map(value_range_json)
            .unwrap_or_else(|_| "{\"min\":0,\"max\":0}".to_string())
    )
}

#[cfg(feature = "mediatek")]
pub fn mediatek_quantized_mlp_tensor_view(
    model: &QuantizedMtkMlpPayload,
) -> Result<rnb_backend_mediatek::MediaTekQuantizedMlpTensorView<'_>, String> {
    validate_quantized_payload(model)?;
    rnb_backend_mediatek::MediaTekQuantizedMlpTensorView::new(
        rnb_backend_mediatek::MediaTekQuantizedMlpShape::new(
            model.input_size,
            model.hidden_size,
            model.output_size,
        ),
        mediatek_quant_params(model.input_params),
        mediatek_quant_params(model.w1_params),
        mediatek_quant_params(model.hidden_params),
        mediatek_quant_params(model.w2_params),
        mediatek_quant_params(model.output_params),
        &model.w1,
        &model.b1,
        &model.w2,
        &model.b2,
        &model.input,
    )
    .map_err(|err| err.to_string())
}

pub fn extract_from_loaded_model(
    model: &LoadedModel,
    config: &MtkMlpConfig,
) -> Result<ExtractedMtkMlp, String> {
    validate_config(config)?;

    let (w1, w1_meta) = extract_rows_prefix(
        model,
        &config.w1_tensor,
        0,
        config.hidden_size,
        config.input_size,
    )?;
    let (w2, w2_meta) = extract_rows_prefix(
        model,
        &config.w2_tensor,
        0,
        config.output_size,
        config.hidden_size,
    )?;
    let (mut input, input_meta) = extract_rows_prefix(
        model,
        &config.input_tensor,
        config.input_row,
        1,
        config.input_size,
    )?;
    input
        .iter_mut()
        .for_each(|value| *value *= config.input_scale);

    ensure_finite("w1", &w1)?;
    ensure_finite("w2", &w2)?;
    ensure_finite("input", &input)?;

    let b1 = vec![0.0; config.hidden_size];
    let b2 = vec![0.0; config.output_size];
    let (hidden, expected) = compute_hidden_and_output(MlpComputation {
        input_size: config.input_size,
        hidden_size: config.hidden_size,
        output_size: config.output_size,
        w1: &w1,
        b1: &b1,
        w2: &w2,
        b2: &b2,
        input: &input,
    })?;
    let payload = MtkMlpPayload {
        input_size: config.input_size,
        hidden_size: config.hidden_size,
        output_size: config.output_size,
        w1,
        b1,
        w2,
        b2,
        input,
        expected,
    };
    let bytes = write_rnbmtk_mlp(&payload)?;
    let hash = sha256_hex(&bytes);
    let metadata = MtkMlpMetadata {
        input_size: config.input_size,
        hidden_size: config.hidden_size,
        output_size: config.output_size,
        input_scale: config.input_scale,
        tensors: vec![w1_meta, w2_meta, input_meta],
        w1_range: value_range(&payload.w1)?,
        w2_range: value_range(&payload.w2)?,
        input_range: value_range(&payload.input)?,
        hidden_range: value_range(&hidden)?,
        output_range: value_range(&payload.expected)?,
        payload_bytes: bytes.len(),
        payload_sha256: hash,
    };
    Ok(ExtractedMtkMlp {
        payload,
        metadata,
        bytes,
    })
}

pub fn extract_gated_gelu_ffn_from_loaded_model(
    model: &LoadedModel,
    config: &MtkGatedGeluFfnConfig,
) -> Result<ExtractedMtkGatedGeluFfn, String> {
    validate_gated_gelu_config(model, config)?;
    let input_size = model.metadata.hidden_size;
    let gate_tensor = format!("blk.{}.ffn_gate.weight", config.layer);
    let up_tensor = format!("blk.{}.ffn_up.weight", config.layer);
    let down_tensor = format!("blk.{}.ffn_down.weight", config.layer);
    let (gate_rows, gate_cols) = loaded_tensor_matrix_shape(model, &gate_tensor)?;
    let (up_rows, up_cols) = loaded_tensor_matrix_shape(model, &up_tensor)?;
    let (down_rows, down_cols) = loaded_tensor_matrix_shape(model, &down_tensor)?;
    if gate_cols != input_size {
        return Err(format!(
            "{gate_tensor} cols {gate_cols} != metadata hidden_size {input_size}"
        ));
    }
    if up_rows != gate_rows || up_cols != gate_cols {
        return Err(format!(
            "{up_tensor} shape {up_rows}x{up_cols} does not match gate shape {gate_rows}x{gate_cols}"
        ));
    }
    if down_cols != gate_rows {
        return Err(format!(
            "{down_tensor} cols {down_cols} != FFN inner size {gate_rows}"
        ));
    }
    if down_rows != input_size {
        return Err(format!(
            "{down_tensor} rows {down_rows} != metadata hidden_size {input_size}"
        ));
    }
    let ffn_inner_size = gate_rows;
    let output_size = down_rows;

    let (gate_weight, gate_meta) =
        extract_rows_prefix(model, &gate_tensor, 0, ffn_inner_size, input_size)?;
    let (up_weight, up_meta) =
        extract_rows_prefix(model, &up_tensor, 0, ffn_inner_size, input_size)?;
    let (down_weight, down_meta) =
        extract_rows_prefix(model, &down_tensor, 0, output_size, ffn_inner_size)?;
    let (mut input, input_meta) =
        extract_rows_prefix(model, &config.input_tensor, config.input_row, 1, input_size)?;
    input
        .iter_mut()
        .for_each(|value| *value *= config.input_scale);

    ensure_finite("gate_weight", &gate_weight)?;
    ensure_finite("up_weight", &up_weight)?;
    ensure_finite("down_weight", &down_weight)?;
    ensure_finite("input", &input)?;

    let (gated, expected) = compute_gated_gelu_hidden_and_output(
        input_size,
        ffn_inner_size,
        output_size,
        &gate_weight,
        &up_weight,
        &down_weight,
        &input,
    )?;
    let payload = MtkGatedGeluFfnPayload {
        input_size,
        ffn_inner_size,
        output_size,
        gate_weight,
        up_weight,
        down_weight,
        input,
        expected,
    };
    let metadata = MtkGatedGeluFfnMetadata {
        layer: config.layer,
        input_size,
        ffn_inner_size,
        output_size,
        input_scale: config.input_scale,
        tensors: vec![gate_meta, up_meta, down_meta, input_meta],
        gate_range: value_range(&payload.gate_weight)?,
        up_range: value_range(&payload.up_weight)?,
        down_range: value_range(&payload.down_weight)?,
        input_range: value_range(&payload.input)?,
        gated_range: value_range(&gated)?,
        output_range: value_range(&payload.expected)?,
    };
    Ok(ExtractedMtkGatedGeluFfn { payload, metadata })
}

pub fn metadata_json(metadata: &MtkMlpMetadata) -> String {
    let tensors = metadata
        .tensors
        .iter()
        .map(tensor_metadata_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    format!(
        concat!(
            "{{\n",
            "  \"input_size\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"output_size\": {},\n",
            "  \"input_scale\": {},\n",
            "  \"payload_bytes\": {},\n",
            "  \"payload_sha256\": \"{}\",\n",
            "  \"ranges\": {{\n",
            "    \"w1\": {},\n",
            "    \"w2\": {},\n",
            "    \"input\": {},\n",
            "    \"hidden\": {},\n",
            "    \"output\": {}\n",
            "  }},\n",
            "  \"tensors\": [\n",
            "    {}\n",
            "  ]\n",
            "}}\n"
        ),
        metadata.input_size,
        metadata.hidden_size,
        metadata.output_size,
        fmt_f32(metadata.input_scale),
        metadata.payload_bytes,
        json_escape(&metadata.payload_sha256),
        value_range_json(metadata.w1_range),
        value_range_json(metadata.w2_range),
        value_range_json(metadata.input_range),
        value_range_json(metadata.hidden_range),
        value_range_json(metadata.output_range),
        tensors
    )
}

fn compute_hidden_and_output(
    computation: MlpComputation<'_>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    check_len(
        "w1",
        computation.w1.len(),
        computation.hidden_size,
        computation.input_size,
    )?;
    check_len("b1", computation.b1.len(), computation.hidden_size, 1)?;
    check_len(
        "w2",
        computation.w2.len(),
        computation.output_size,
        computation.hidden_size,
    )?;
    check_len("b2", computation.b2.len(), computation.output_size, 1)?;
    check_len("input", computation.input.len(), computation.input_size, 1)?;
    ensure_finite("w1", computation.w1)?;
    ensure_finite("b1", computation.b1)?;
    ensure_finite("w2", computation.w2)?;
    ensure_finite("b2", computation.b2)?;
    ensure_finite("input", computation.input)?;

    let mut hidden = Vec::with_capacity(computation.hidden_size);
    for h in 0..computation.hidden_size {
        let mut acc = computation.b1[h];
        for i in 0..computation.input_size {
            acc += computation.input[i] * computation.w1[h * computation.input_size + i];
        }
        hidden.push(acc.max(0.0));
    }

    let mut output = Vec::with_capacity(computation.output_size);
    for o in 0..computation.output_size {
        let mut acc = computation.b2[o];
        for (h, hidden_value) in hidden.iter().enumerate() {
            acc += *hidden_value * computation.w2[o * computation.hidden_size + h];
        }
        output.push(acc);
    }
    ensure_finite("hidden", &hidden)?;
    ensure_finite("output", &output)?;
    Ok((hidden, output))
}

fn compute_gated_gelu_hidden_and_output(
    input_size: usize,
    ffn_inner_size: usize,
    output_size: usize,
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    for (name, value) in [
        ("input_size", input_size),
        ("ffn_inner_size", ffn_inner_size),
        ("output_size", output_size),
    ] {
        if value == 0 || value > DIAGNOSTIC_DIM_CAP {
            return Err(format!(
                "{name}={value} outside diagnostic cap 1..={DIAGNOSTIC_DIM_CAP}"
            ));
        }
    }
    check_len("gate_weight", gate_weight.len(), ffn_inner_size, input_size)?;
    check_len("up_weight", up_weight.len(), ffn_inner_size, input_size)?;
    check_len(
        "down_weight",
        down_weight.len(),
        output_size,
        ffn_inner_size,
    )?;
    check_len("input", input.len(), input_size, 1)?;
    ensure_finite("gate_weight", gate_weight)?;
    ensure_finite("up_weight", up_weight)?;
    ensure_finite("down_weight", down_weight)?;
    ensure_finite("input", input)?;

    let mut gated = Vec::with_capacity(ffn_inner_size);
    for row in 0..ffn_inner_size {
        let row_offset = row * input_size;
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        for col in 0..input_size {
            gate_acc += gate_weight[row_offset + col] * input[col];
            up_acc += up_weight[row_offset + col] * input[col];
        }
        gated.push(gelu_tanh_approx(gate_acc) * up_acc);
    }

    let mut output = Vec::with_capacity(output_size);
    for row in 0..output_size {
        let row_offset = row * ffn_inner_size;
        let mut acc = 0.0f32;
        for col in 0..ffn_inner_size {
            acc += down_weight[row_offset + col] * gated[col];
        }
        output.push(acc);
    }
    ensure_finite("gated_gelu_hidden", &gated)?;
    ensure_finite("gated_gelu_output", &output)?;
    Ok((gated, output))
}

fn gelu_tanh_approx(value: f32) -> f32 {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * value * (1.0 + (sqrt_2_over_pi * (value + 0.044715 * value.powi(3))).tanh())
}

#[derive(Clone, Copy)]
struct QuantizedComputation<'a> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    input: &'a [u8],
    input_params: QuantParams,
    w1: &'a [u8],
    w1_params: QuantParams,
    b1: &'a [i32],
    hidden_params: QuantParams,
    w2: &'a [u8],
    w2_params: QuantParams,
    b2: &'a [i32],
    output_params: QuantParams,
}

fn quantized_expected(computation: QuantizedComputation<'_>) -> Result<Vec<f32>, String> {
    check_len(
        "quant input",
        computation.input.len(),
        computation.input_size,
        1,
    )?;
    check_len(
        "quant w1",
        computation.w1.len(),
        computation.hidden_size,
        computation.input_size,
    )?;
    check_len("quant b1", computation.b1.len(), computation.hidden_size, 1)?;
    check_len(
        "quant w2",
        computation.w2.len(),
        computation.output_size,
        computation.hidden_size,
    )?;
    check_len("quant b2", computation.b2.len(), computation.output_size, 1)?;

    let mut hidden = Vec::with_capacity(computation.hidden_size);
    for h in 0..computation.hidden_size {
        let mut acc = i64::from(computation.b1[h]);
        for i in 0..computation.input_size {
            let lhs = i32::from(computation.input[i]) - computation.input_params.zero_point;
            let rhs = i32::from(computation.w1[h * computation.input_size + i])
                - computation.w1_params.zero_point;
            acc += i64::from(lhs) * i64::from(rhs);
        }
        let real = (acc as f32) * computation.input_params.scale * computation.w1_params.scale;
        hidden.push(quantize_u8(real.max(0.0), computation.hidden_params));
    }

    let mut expected = Vec::with_capacity(computation.output_size);
    for o in 0..computation.output_size {
        let mut acc = i64::from(computation.b2[o]);
        for (h, hidden_value) in hidden.iter().enumerate() {
            let lhs = i32::from(*hidden_value) - computation.hidden_params.zero_point;
            let rhs = i32::from(computation.w2[o * computation.hidden_size + h])
                - computation.w2_params.zero_point;
            acc += i64::from(lhs) * i64::from(rhs);
        }
        let real = (acc as f32) * computation.hidden_params.scale * computation.w2_params.scale;
        let quantized = quantize_u8(real, computation.output_params);
        expected.push(dequantize_u8(quantized, computation.output_params));
    }
    ensure_finite("quant expected", &expected)?;
    Ok(expected)
}

fn extract_rows_prefix(
    model: &LoadedModel,
    name: &str,
    row_start: usize,
    row_count: usize,
    prefix_cols: usize,
) -> Result<(Vec<f32>, TensorRowMetadata), String> {
    if row_count == 0 {
        return Err(format!("{name} row_count must be non-zero"));
    }
    if prefix_cols == 0 {
        return Err(format!("{name} prefix_cols must be non-zero"));
    }
    let tensor = model
        .weights
        .get(name)
        .ok_or_else(|| format!("tensor not found: {name}"))?;
    let ggml_type = *model
        .tensor_ggml_types
        .get(name)
        .ok_or_else(|| format!("missing GGML type for tensor: {name}"))?;
    let shape = tensor_shape(model, name, ggml_type, tensor.shape())?;
    let (rows, cols) = validate_matrix_dims(name, &shape, row_start + row_count, prefix_cols)?;
    let bytes = tensor
        .as_bytes()
        .ok_or_else(|| format!("tensor bytes unavailable or non-contiguous: {name}"))?;
    if bytes.len() % rows != 0 {
        return Err(format!(
            "{name} byte length {} is not divisible by rows {rows}",
            bytes.len()
        ));
    }
    let actual_bytes_per_row = bytes.len() / rows;
    let bytes_per_row = bytes_per_row_checked(ggml_type, cols, actual_bytes_per_row)?;
    let layout = row_layout_for_type(ggml_type)?;
    let mut out = Vec::with_capacity(row_count * prefix_cols);
    for row in row_start..row_start + row_count {
        let start = row * bytes_per_row;
        let row_bytes = &bytes[start..start + bytes_per_row];
        let dequant = dequantize_bytes_to_f32(row_bytes, layout.dequant_type);
        if dequant.len() != cols {
            return Err(format!(
                "{name} row {row} dequant length {} != cols {cols}",
                dequant.len()
            ));
        }
        out.extend_from_slice(&dequant[..prefix_cols]);
    }
    Ok((
        out,
        TensorRowMetadata {
            name: name.to_string(),
            ggml_type,
            shape,
            rows_selected: row_count,
            row_start,
            cols_selected: prefix_cols,
            bytes_per_row,
        },
    ))
}

fn loaded_tensor_matrix_shape(model: &LoadedModel, name: &str) -> Result<(usize, usize), String> {
    let tensor = model
        .weights
        .get(name)
        .ok_or_else(|| format!("tensor not found: {name}"))?;
    let ggml_type = *model
        .tensor_ggml_types
        .get(name)
        .ok_or_else(|| format!("missing GGML type for tensor: {name}"))?;
    let shape = tensor_shape(model, name, ggml_type, tensor.shape())?;
    validate_matrix_dims(name, &shape, 1, 1)
}

fn tensor_shape(
    model: &LoadedModel,
    name: &str,
    ggml_type: GGMLType,
    tensor_shape: &[usize],
) -> Result<Vec<usize>, String> {
    match ggml_type {
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => Ok(tensor_shape.to_vec()),
        _ => model
            .float_shapes
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing float shape for quantized tensor: {name}")),
    }
}

fn validate_config(config: &MtkMlpConfig) -> Result<(), String> {
    for (name, value) in [
        ("input_size", config.input_size),
        ("hidden_size", config.hidden_size),
        ("output_size", config.output_size),
    ] {
        if value == 0 || value > DIAGNOSTIC_DIM_CAP {
            return Err(format!(
                "{name}={value} outside diagnostic cap 1..={DIAGNOSTIC_DIM_CAP}"
            ));
        }
    }
    if !config.input_scale.is_finite() {
        return Err(format!(
            "input_scale must be finite, got {}",
            config.input_scale
        ));
    }
    Ok(())
}

fn validate_gated_gelu_config(
    model: &LoadedModel,
    config: &MtkGatedGeluFfnConfig,
) -> Result<(), String> {
    if config.layer >= model.metadata.num_layers {
        return Err(format!(
            "layer {} out of range for num_layers {}",
            config.layer, model.metadata.num_layers
        ));
    }
    for (name, value) in [
        ("hidden_size", model.metadata.hidden_size),
        ("intermediate_size", model.metadata.intermediate_size),
    ] {
        if value == 0 || value > DIAGNOSTIC_DIM_CAP {
            return Err(format!(
                "{name}={value} outside diagnostic cap 1..={DIAGNOSTIC_DIM_CAP}"
            ));
        }
    }
    if !config.input_scale.is_finite() {
        return Err(format!(
            "input_scale must be finite, got {}",
            config.input_scale
        ));
    }
    if config.input_tensor.trim().is_empty() {
        return Err("input_tensor must not be empty".to_string());
    }
    Ok(())
}

fn validate_payload(model: &MtkMlpPayload) -> Result<(), String> {
    for (name, value) in [
        ("input_size", model.input_size),
        ("hidden_size", model.hidden_size),
        ("output_size", model.output_size),
    ] {
        if value == 0 || value > DIAGNOSTIC_DIM_CAP {
            return Err(format!(
                "{name}={value} outside diagnostic cap 1..={DIAGNOSTIC_DIM_CAP}"
            ));
        }
    }
    check_len("w1", model.w1.len(), model.hidden_size, model.input_size)?;
    check_len("b1", model.b1.len(), model.hidden_size, 1)?;
    check_len("w2", model.w2.len(), model.output_size, model.hidden_size)?;
    check_len("b2", model.b2.len(), model.output_size, 1)?;
    check_len("input", model.input.len(), model.input_size, 1)?;
    check_len("expected", model.expected.len(), model.output_size, 1)?;
    ensure_finite("w1", &model.w1)?;
    ensure_finite("b1", &model.b1)?;
    ensure_finite("w2", &model.w2)?;
    ensure_finite("b2", &model.b2)?;
    ensure_finite("input", &model.input)?;
    ensure_finite("expected", &model.expected)?;
    Ok(())
}

fn validate_quantized_payload(model: &QuantizedMtkMlpPayload) -> Result<(), String> {
    for (name, value) in [
        ("input_size", model.input_size),
        ("hidden_size", model.hidden_size),
        ("output_size", model.output_size),
    ] {
        if value == 0 || value > DIAGNOSTIC_DIM_CAP {
            return Err(format!(
                "{name}={value} outside diagnostic cap 1..={DIAGNOSTIC_DIM_CAP}"
            ));
        }
    }
    for (name, params) in [
        ("input_params", model.input_params),
        ("w1_params", model.w1_params),
        ("hidden_params", model.hidden_params),
        ("w2_params", model.w2_params),
        ("output_params", model.output_params),
    ] {
        validate_quant_params(name, params)?;
    }
    check_len(
        "quant w1",
        model.w1.len(),
        model.hidden_size,
        model.input_size,
    )?;
    check_len("quant b1", model.b1.len(), model.hidden_size, 1)?;
    check_len(
        "quant w2",
        model.w2.len(),
        model.output_size,
        model.hidden_size,
    )?;
    check_len("quant b2", model.b2.len(), model.output_size, 1)?;
    check_len("quant input", model.input.len(), model.input_size, 1)?;
    check_len("quant expected", model.expected.len(), model.output_size, 1)?;
    ensure_finite("quant expected", &model.expected)?;
    Ok(())
}

fn check_len(name: &str, actual: usize, rows: usize, cols: usize) -> Result<(), String> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{name} length check overflow for shape {rows}x{cols}"))?;
    if actual != expected {
        return Err(format!("{name} len {actual} != expected {expected}"));
    }
    Ok(())
}

fn validate_quant_params(name: &str, params: QuantParams) -> Result<(), String> {
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(format!("{name} scale must be finite and positive"));
    }
    if !(0..=255).contains(&params.zero_point) {
        return Err(format!(
            "{name} zero_point {} outside 0..=255",
            params.zero_point
        ));
    }
    Ok(())
}

#[cfg(feature = "mediatek")]
fn mediatek_quant_params(params: QuantParams) -> rnb_backend_mediatek::MediaTekQuantParams {
    rnb_backend_mediatek::MediaTekQuantParams::new(params.scale, params.zero_point)
}

fn symmetric_u8_params(values: &[f32], name: &str) -> Result<QuantParams, String> {
    ensure_finite(name, values)?;
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let scale = if max_abs > 0.0 {
        max_abs / 127.0
    } else {
        1.0 / 255.0
    };
    Ok(QuantParams {
        scale,
        zero_point: 128,
    })
}

fn relu_u8_params(values: &[f32]) -> Result<QuantParams, String> {
    ensure_finite("hidden", values)?;
    let max = values.iter().copied().fold(0.0_f32, f32::max);
    let scale = if max > 0.0 { max / 255.0 } else { 1.0 / 255.0 };
    Ok(QuantParams {
        scale,
        zero_point: 0,
    })
}

fn quantize_slice(values: &[f32], params: QuantParams) -> Vec<u8> {
    values
        .iter()
        .map(|value| quantize_u8(*value, params))
        .collect()
}

fn quantize_u8(value: f32, params: QuantParams) -> u8 {
    let raw = (value / params.scale + params.zero_point as f32).round();
    raw.clamp(0.0, 255.0) as u8
}

fn dequantize_u8(value: u8, params: QuantParams) -> f32 {
    (i32::from(value) - params.zero_point) as f32 * params.scale
}

pub fn quantized_output_parity(
    model: &QuantizedMtkMlpPayload,
    output: &[u8],
) -> Result<QuantizedOutputParity, String> {
    validate_quantized_payload(model)?;
    if output.len() != model.output_size {
        return Err(format!(
            "quantized output len {} != expected {}",
            output.len(),
            model.output_size
        ));
    }

    let mut parity = QuantizedOutputParity {
        max_byte_delta: 0,
        max_byte_delta_index: 0,
        max_abs_error: 0.0,
    };
    for (idx, (&actual, &expected)) in output.iter().zip(model.expected.iter()).enumerate() {
        let expected_byte = quantize_u8(expected, model.output_params);
        let byte_delta = u32::from(actual.abs_diff(expected_byte));
        if byte_delta > parity.max_byte_delta {
            parity.max_byte_delta = byte_delta;
            parity.max_byte_delta_index = idx;
        }

        let abs_error = (dequantize_u8(actual, model.output_params) - expected).abs();
        if abs_error > parity.max_abs_error {
            parity.max_abs_error = abs_error;
        }
    }
    Ok(parity)
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{name} contains non-finite value at {idx}: {value}"
        ));
    }
    Ok(())
}

fn value_range(values: &[f32]) -> Result<ValueRange, String> {
    ensure_finite("value_range", values)?;
    let Some(first) = values.first().copied() else {
        return Err("value_range requires at least one value".to_string());
    };
    let mut min = first;
    let mut max = first;
    for value in &values[1..] {
        min = min.min(*value);
        max = max.max(*value);
    }
    Ok(ValueRange { min, max })
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32_slice(bytes: &mut Vec<u8>, values: &[i32]) {
    for value in values {
        push_i32(bytes, *value);
    }
}

fn push_usize_as_u32(bytes: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u32::try_from(value).map_err(|_| format!("value {value} does not fit in u32"))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn tensor_metadata_json(metadata: &TensorRowMetadata) -> String {
    format!(
        concat!(
            "{{",
            "\"name\":\"{}\",",
            "\"ggml_type\":\"{:?}\",",
            "\"shape\":{},",
            "\"row_start\":{},",
            "\"rows_selected\":{},",
            "\"cols_selected\":{},",
            "\"bytes_per_row\":{}",
            "}}"
        ),
        json_escape(&metadata.name),
        metadata.ggml_type,
        usize_array_json(&metadata.shape),
        metadata.row_start,
        metadata.rows_selected,
        metadata.cols_selected,
        metadata.bytes_per_row
    )
}

fn value_range_json(range: ValueRange) -> String {
    format!(
        "{{\"min\":{},\"max\":{}}}",
        fmt_f32(range.min),
        fmt_f32(range.max)
    )
}

fn quant_params_json(params: QuantParams) -> String {
    format!(
        "{{\"scale\":{},\"zero_point\":{}}}",
        fmt_f32(params.scale),
        params.zero_point
    )
}

fn usize_array_json(values: &[usize]) -> String {
    let body = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn fmt_f32(value: f32) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.9}")
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_loader::GGMLType;

    #[test]
    fn maps_supported_ggml_types_to_row_layouts() {
        assert_eq!(row_layout_for_type(GGMLType::F32).unwrap().block_elems, 1);
        assert_eq!(row_layout_for_type(GGMLType::F16).unwrap().block_bytes, 2);
        assert_eq!(row_layout_for_type(GGMLType::BF16).unwrap().block_bytes, 2);
        assert_eq!(row_layout_for_type(GGMLType::Q8_0).unwrap().block_elems, 32);
        assert_eq!(row_layout_for_type(GGMLType::Q8_1).unwrap().block_bytes, 36);
        assert_eq!(
            row_layout_for_type(GGMLType::Q4_K).unwrap().block_elems,
            256
        );
        assert_eq!(
            row_layout_for_type(GGMLType::Q5_K).unwrap().block_bytes,
            176
        );
        assert_eq!(
            row_layout_for_type(GGMLType::Q6_K).unwrap().block_bytes,
            210
        );
        assert!(row_layout_for_type(GGMLType::I32).is_err());
    }

    #[test]
    fn validates_row_layout_boundaries() {
        assert_eq!(
            bytes_per_row_checked(GGMLType::Q4_K, 512, 288).unwrap(),
            288
        );
        assert_eq!(
            bytes_per_row_checked(GGMLType::Q5_K, 256, 176).unwrap(),
            176
        );
        assert_eq!(
            bytes_per_row_checked(GGMLType::Q6_K, 768, 630).unwrap(),
            630
        );
        assert_eq!(bytes_per_row_checked(GGMLType::Q8_0, 64, 68).unwrap(), 68);
        assert_eq!(bytes_per_row_checked(GGMLType::Q8_1, 64, 72).unwrap(), 72);
        assert_eq!(bytes_per_row_checked(GGMLType::F32, 7, 28).unwrap(), 28);
        assert_eq!(bytes_per_row_checked(GGMLType::F16, 7, 14).unwrap(), 14);
        assert_eq!(bytes_per_row_checked(GGMLType::BF16, 7, 14).unwrap(), 14);

        assert!(bytes_per_row_checked(GGMLType::Q4_K, 384, 288).is_err());
        assert!(bytes_per_row_checked(GGMLType::Q8_0, 33, 68).is_err());
        assert!(bytes_per_row_checked(GGMLType::F32, 7, 24).is_err());
    }

    #[test]
    fn slices_f32_rows_and_builds_expected_payload() {
        let rows = vec![
            1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 0.5, 0.25, 0.125, 0.0625,
        ];
        let row = slice_f32_row_prefix(&rows, 3, 4, 2, 3).unwrap();
        assert_eq!(row, vec![0.5, 0.25, 0.125]);

        let model = MtkMlpPayload {
            input_size: 2,
            hidden_size: 2,
            output_size: 1,
            w1: vec![1.0, -2.0, 0.5, 0.25],
            b1: vec![0.0, 0.0],
            w2: vec![2.0, -1.0],
            b2: vec![0.25],
            input: vec![3.0, 1.0],
            expected: compute_expected(MlpComputation {
                input_size: 2,
                hidden_size: 2,
                output_size: 1,
                w1: &[1.0, -2.0, 0.5, 0.25],
                b1: &[0.0, 0.0],
                w2: &[2.0, -1.0],
                b2: &[0.25],
                input: &[3.0, 1.0],
            })
            .unwrap(),
        };
        assert_eq!(model.expected, vec![0.5]);
        let bytes = write_rnbmtk_mlp(&model).unwrap();
        assert_eq!(&bytes[..8], b"RNBMTK3\0");
        assert_eq!(bytes.len(), 24 + 4 * (4 + 2 + 2 + 1 + 2 + 1));
    }

    #[test]
    fn gated_gelu_ffn_oracle_uses_gate_up_and_down_projection() {
        let output = compute_gated_gelu_ffn_expected(
            2,
            2,
            1,
            &[1.0, -0.5, 0.25, 0.75],
            &[0.5, 0.25, -0.25, 0.5],
            &[0.75, -0.5],
            &[0.4, -0.2],
        )
        .unwrap();

        let gate0 = gelu_tanh_approx(1.0 * 0.4 + -0.5 * -0.2) * (0.5 * 0.4 + 0.25 * -0.2);
        let gate1 = gelu_tanh_approx(0.25 * 0.4 + 0.75 * -0.2) * (-0.25 * 0.4 + 0.5 * -0.2);
        let expected = 0.75 * gate0 + -0.5 * gate1;

        assert_eq!(output.len(), 1);
        assert!((output[0] - expected).abs() <= 1e-6);
        assert_ne!(output[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn gated_gelu_ffn_rejects_mismatched_lengths() {
        let err = compute_gated_gelu_ffn_expected(
            2,
            2,
            1,
            &[1.0, -0.5, 0.25],
            &[0.5, 0.25, -0.25, 0.5],
            &[0.75, -0.5],
            &[0.4, -0.2],
        )
        .unwrap_err();

        assert!(err.contains("gate_weight len 3 != expected 4"));
    }

    #[test]
    fn rejects_bad_dimensions_and_rows() {
        assert!(validate_matrix_dims("w1", &[4], 1, 1).is_err());
        assert!(validate_matrix_dims("w1", &[1, 1], 2, 1).is_err());
        assert!(slice_f32_row_prefix(&[1.0, 2.0], 1, 2, 1, 1).is_err());
        assert!(compute_expected(MlpComputation {
            input_size: 2,
            hidden_size: 1,
            output_size: 1,
            w1: &[1.0],
            b1: &[0.0],
            w2: &[1.0],
            b2: &[0.0],
            input: &[1.0, 2.0],
        })
        .is_err());
    }

    #[test]
    fn quantized_payload_writer_uses_separate_magic_and_header() {
        let model = MtkMlpPayload {
            input_size: 2,
            hidden_size: 2,
            output_size: 1,
            w1: vec![0.25, -0.5, 0.125, 0.375],
            b1: vec![0.0, 0.0],
            w2: vec![0.5, -0.25],
            b2: vec![0.0],
            input: vec![0.5, -0.25],
            expected: vec![0.0],
        };

        let quantized = quantize_mlp_payload(&model).unwrap();
        assert_eq!(quantized.input.len(), 2);
        assert_eq!(quantized.w1.len(), 4);
        assert_eq!(quantized.b1, vec![0, 0]);
        assert_eq!(quantized.w2.len(), 2);
        assert_eq!(quantized.b2, vec![0]);
        assert_eq!(quantized.expected.len(), 1);

        let bytes = write_rnbmtk_quant_mlp(&quantized).unwrap();
        assert_eq!(&bytes[..8], b"RNBMTKQ1");
        assert_eq!(bytes.len(), 64 + 4 + 8 + 2 + 4 + 2 + 4);
    }

    #[test]
    fn quantized_payload_accepts_full_gemma_e2b_ffn_layer_shape() {
        let input_size = 1536;
        let hidden_size = 6144;
        let output_size = 1536;
        let payload = QuantizedMtkMlpPayload {
            input_size,
            hidden_size,
            output_size,
            input_params: QuantParams {
                scale: 1.0 / 255.0,
                zero_point: 128,
            },
            w1_params: QuantParams {
                scale: 1.0 / 255.0,
                zero_point: 128,
            },
            hidden_params: QuantParams {
                scale: 1.0 / 255.0,
                zero_point: 0,
            },
            w2_params: QuantParams {
                scale: 1.0 / 255.0,
                zero_point: 128,
            },
            output_params: QuantParams {
                scale: 1.0 / 255.0,
                zero_point: 128,
            },
            w1: vec![128; hidden_size * input_size],
            b1: vec![0; hidden_size],
            w2: vec![128; output_size * hidden_size],
            b2: vec![0; output_size],
            input: vec![128; input_size],
            expected: vec![0.0; output_size],
        };

        validate_quantized_payload(&payload).expect("full Gemma E2B FFN layer shape");
    }

    #[test]
    #[cfg(feature = "mediatek")]
    fn quantized_payload_borrows_into_mediatek_tensor_view_without_artifact_path() {
        let payload = QuantizedMtkMlpPayload {
            input_size: 4,
            hidden_size: 3,
            output_size: 2,
            input_params: QuantParams {
                scale: 0.25,
                zero_point: 128,
            },
            w1_params: QuantParams {
                scale: 0.5,
                zero_point: 128,
            },
            hidden_params: QuantParams {
                scale: 0.125,
                zero_point: 0,
            },
            w2_params: QuantParams {
                scale: 0.75,
                zero_point: 128,
            },
            output_params: QuantParams {
                scale: 1.0,
                zero_point: 128,
            },
            w1: vec![128; 12],
            b1: vec![0; 3],
            w2: vec![128; 6],
            b2: vec![0; 2],
            input: vec![128; 4],
            expected: vec![0.0; 2],
        };

        let tensors = mediatek_quantized_mlp_tensor_view(&payload)
            .expect("quantized GGUF-derived payload should become in-memory MediaTek tensor view");

        assert_eq!(tensors.shape().input_size(), 4);
        assert_eq!(tensors.shape().hidden_size(), 3);
        assert_eq!(tensors.shape().output_size(), 2);
        assert_eq!(tensors.w1().as_ptr(), payload.w1.as_ptr());
        assert_eq!(tensors.w2().as_ptr(), payload.w2.as_ptr());
        assert_eq!(tensors.input().as_ptr(), payload.input.as_ptr());
        assert_eq!(tensors.output_len(), payload.output_size);
    }

    #[test]
    fn quantized_oracle_matches_quantized_pipeline_not_fp32_pipeline() {
        let model = MtkMlpPayload {
            input_size: 2,
            hidden_size: 1,
            output_size: 2,
            w1: vec![0.33, -0.21],
            b1: vec![0.0],
            w2: vec![0.69, 0.37],
            b2: vec![0.0, 0.0],
            input: vec![0.4, -0.1],
            expected: compute_expected(MlpComputation {
                input_size: 2,
                hidden_size: 1,
                output_size: 2,
                w1: &[0.33, -0.21],
                b1: &[0.0],
                w2: &[0.69, 0.37],
                b2: &[0.0, 0.0],
                input: &[0.4, -0.1],
            })
            .unwrap(),
        };

        let quantized = quantize_mlp_payload(&model).unwrap();
        let fp32_expected = model.expected[1];
        let quantized_expected = quantized.expected[1];
        assert!(quantized_expected.is_finite());
        assert!(fp32_expected.is_finite());
        assert_ne!(quantized_expected.to_bits(), fp32_expected.to_bits());
    }

    #[test]
    fn mtk_gguf_mlp_parity_rejects_large_output_delta() {
        let payload = QuantizedMtkMlpPayload {
            input_size: 1,
            hidden_size: 1,
            output_size: 2,
            input_params: QuantParams {
                scale: 1.0,
                zero_point: 128,
            },
            w1_params: QuantParams {
                scale: 1.0,
                zero_point: 128,
            },
            hidden_params: QuantParams {
                scale: 1.0,
                zero_point: 0,
            },
            w2_params: QuantParams {
                scale: 1.0,
                zero_point: 128,
            },
            output_params: QuantParams {
                scale: 0.5,
                zero_point: 128,
            },
            w1: vec![128],
            b1: vec![0],
            w2: vec![128, 128],
            b2: vec![0, 0],
            input: vec![128],
            expected: vec![0.0, 0.0],
        };

        let parity = quantized_output_parity(&payload, &[128, 140])
            .expect("matching output length should compute parity");

        assert_eq!(parity.max_byte_delta, 12);
        assert_eq!(parity.max_byte_delta_index, 1);
        assert!(!parity.passes(0.003));
    }
}
