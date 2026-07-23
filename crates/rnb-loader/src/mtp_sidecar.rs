use crate::arch::{Architecture, ModelMetadata, MtpMetadata};
use crate::convert::compute_tensor_size;
use crate::error::LoaderError;
use crate::gguf::types::GGMLType;
use rnb_core::tensor::Tensor;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MtpSidecarTensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) ggml_type: GGMLType,
    pub(crate) data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MtpSidecarLoadOptions {
    materialize_f16_matmul_as_f32: bool,
}

impl MtpSidecarLoadOptions {
    fn from_env() -> Self {
        Self {
            materialize_f16_matmul_as_f32: mtp_sidecar_f16_to_f32_enabled(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_materialize_f16_matmul_as_f32(enabled: bool) -> Self {
        Self {
            materialize_f16_matmul_as_f32: enabled,
        }
    }
}

pub(crate) fn attach_adjacent_qwen35_mtp1_sidecar(
    model_path: &Path,
    metadata: &mut ModelMetadata,
    weights: &mut HashMap<String, Tensor>,
    float_shapes: &mut HashMap<String, Vec<usize>>,
    tensor_ggml_types: &mut HashMap<String, GGMLType>,
) -> Result<bool, LoaderError> {
    let Some(model_dir) = model_path.parent() else {
        return Ok(false);
    };
    let sidecar_path = model_dir.join("mtp_weights.bin");
    if !sidecar_path.exists() {
        return Ok(false);
    }
    let bytes = fs::read(&sidecar_path)?;
    inject_qwen35_mtp1_sidecar_bytes(&bytes, metadata, weights, float_shapes, tensor_ggml_types)?;
    eprintln!("[INFO] MTP sidecar: {}", sidecar_path.display());
    Ok(true)
}

pub(crate) fn parse_mtp1_sidecar_bytes(bytes: &[u8]) -> Result<Vec<MtpSidecarTensor>, LoaderError> {
    let mut pos = 0usize;
    let magic = read_exact(bytes, &mut pos, 4)?;
    if magic != b"MTP1" {
        return Err(parse_error(0, "invalid MTP sidecar magic"));
    }
    let tensor_count = read_u32(bytes, &mut pos)? as usize;
    let mut headers = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let record_offset = pos;
        let name_len = read_u32(bytes, &mut pos)? as usize;
        let name = std::str::from_utf8(read_exact(bytes, &mut pos, name_len)?)
            .map_err(|_| parse_error(record_offset, "invalid UTF-8 tensor name"))?
            .to_string();
        let dim_count = read_u32(bytes, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            shape.push(read_u32(bytes, &mut pos)? as usize);
        }
        let ggml_type_raw = read_u32(bytes, &mut pos)?;
        let ggml_type = GGMLType::try_from(ggml_type_raw)
            .map_err(|raw| LoaderError::UnsupportedGGMLType(raw))?;
        let data_offset = read_u64(bytes, &mut pos)? as usize;
        let data_size = read_u64(bytes, &mut pos)? as usize;
        let expected_size = compute_tensor_size(&shape, ggml_type);
        if data_size != expected_size {
            return Err(parse_error(
                record_offset,
                &format!("MTP tensor '{name}' byte size {data_size} != expected {expected_size}"),
            ));
        }
        headers.push((name, shape, ggml_type, data_offset, data_size));
    }

    let mut tensors = Vec::with_capacity(headers.len());
    for (name, shape, ggml_type, data_offset, data_size) in headers {
        let data_end = data_offset.checked_add(data_size).ok_or_else(|| {
            parse_error(
                data_offset,
                &format!("MTP tensor '{name}' byte range overflows"),
            )
        })?;
        if data_end > bytes.len() {
            return Err(parse_error(
                data_offset,
                &format!("MTP tensor '{name}' byte range exceeds file size"),
            ));
        }
        tensors.push(MtpSidecarTensor {
            name,
            shape,
            ggml_type,
            data: bytes[data_offset..data_end].to_vec(),
        });
    }
    Ok(tensors)
}

pub(crate) fn inject_qwen35_mtp1_sidecar_bytes(
    bytes: &[u8],
    metadata: &mut ModelMetadata,
    weights: &mut HashMap<String, Tensor>,
    float_shapes: &mut HashMap<String, Vec<usize>>,
    tensor_ggml_types: &mut HashMap<String, GGMLType>,
) -> Result<(), LoaderError> {
    let tensors = parse_mtp1_sidecar_bytes(bytes)?;
    inject_qwen35_mtp_tensors(
        tensors,
        metadata,
        weights,
        float_shapes,
        tensor_ggml_types,
        MtpSidecarLoadOptions::from_env(),
    )
}

#[cfg(test)]
pub(crate) fn inject_qwen35_mtp1_sidecar_bytes_with_options(
    bytes: &[u8],
    metadata: &mut ModelMetadata,
    weights: &mut HashMap<String, Tensor>,
    float_shapes: &mut HashMap<String, Vec<usize>>,
    tensor_ggml_types: &mut HashMap<String, GGMLType>,
    options: MtpSidecarLoadOptions,
) -> Result<(), LoaderError> {
    let tensors = parse_mtp1_sidecar_bytes(bytes)?;
    inject_qwen35_mtp_tensors(
        tensors,
        metadata,
        weights,
        float_shapes,
        tensor_ggml_types,
        options,
    )
}

fn inject_qwen35_mtp_tensors(
    tensors: Vec<MtpSidecarTensor>,
    metadata: &mut ModelMetadata,
    weights: &mut HashMap<String, Tensor>,
    float_shapes: &mut HashMap<String, Vec<usize>>,
    tensor_ggml_types: &mut HashMap<String, GGMLType>,
    options: MtpSidecarLoadOptions,
) -> Result<(), LoaderError> {
    if metadata.architecture != Architecture::Qwen35 {
        return Err(parse_error(
            0,
            "MTP1 sidecar is currently supported only for Qwen3.5 dense models",
        ));
    }
    if metadata.mtp.is_some() {
        return Err(parse_error(
            0,
            "MTP1 sidecar cannot be attached to a model that already has MTP metadata",
        ));
    }

    let layer_index = metadata.num_layers;
    let source_by_name = tensors
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<HashMap<_, _>>();

    for (source, target) in qwen35_mtp_name_mapping(layer_index) {
        let tensor = source_by_name
            .get(source)
            .ok_or_else(|| parse_error(0, &format!("missing MTP sidecar tensor '{source}'")))?;
        if is_qwen35_mtp_norm_tensor(source) {
            let norm = decode_f16_norm_with_unit_offset(tensor, source)?;
            weights.insert(target.clone(), Tensor::from_slice(&norm, &tensor.shape));
            float_shapes.remove(&target);
            tensor_ggml_types.insert(target, GGMLType::F32);
        } else if options.materialize_f16_matmul_as_f32 && tensor.ggml_type == GGMLType::F16 {
            let values = decode_f16_tensor(tensor, source)?;
            weights.insert(target.clone(), Tensor::from_slice(&values, &tensor.shape));
            float_shapes.remove(&target);
            tensor_ggml_types.insert(target, GGMLType::F32);
        } else {
            weights.insert(
                target.clone(),
                Tensor::from_slice(&tensor.data, &[tensor.data.len()]),
            );
            float_shapes.insert(target.clone(), tensor.shape.clone());
            tensor_ggml_types.insert(target, tensor.ggml_type);
        }
    }

    metadata.mtp = Some(MtpMetadata {
        total_block_count: metadata.num_layers + 1,
        trunk_layers: metadata.num_layers,
        first_mtp_layer: metadata.num_layers,
        nextn_predict_layers: 1,
    });
    Ok(())
}

fn mtp_sidecar_f16_to_f32_enabled() -> bool {
    std::env::var("RNB_MTP_SIDECAR_F16_TO_F32")
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES"
            )
        })
        .unwrap_or(false)
}

fn is_qwen35_mtp_norm_tensor(name: &str) -> bool {
    name.contains("norm") && name.ends_with(".weight")
}

fn decode_f16_norm_with_unit_offset(
    tensor: &MtpSidecarTensor,
    source_name: &str,
) -> Result<Vec<f32>, LoaderError> {
    if tensor.ggml_type != GGMLType::F16 {
        return Err(parse_error(
            0,
            &format!("MTP norm tensor '{source_name}' must be F16"),
        ));
    }
    let expected_values: usize = tensor.shape.iter().product();
    if tensor.data.len() != expected_values * 2 {
        return Err(parse_error(
            0,
            &format!("MTP norm tensor '{source_name}' byte length mismatch"),
        ));
    }
    Ok(tensor
        .data
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_bits_to_f32(bits) + 1.0
        })
        .collect())
}

fn decode_f16_tensor(
    tensor: &MtpSidecarTensor,
    source_name: &str,
) -> Result<Vec<f32>, LoaderError> {
    if tensor.ggml_type != GGMLType::F16 {
        return Err(parse_error(
            0,
            &format!("MTP tensor '{source_name}' must be F16"),
        ));
    }
    let expected_values: usize = tensor.shape.iter().product();
    if tensor.data.len() != expected_values * 2 {
        return Err(parse_error(
            0,
            &format!("MTP tensor '{source_name}' byte length mismatch"),
        ));
    }
    Ok(tensor
        .data
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            let exp32 = ((e + 127) as u32) << 23;
            sign | exp32 | (mant << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = ((exp - 15 + 127) as u32) << 23;
        sign | exp32 | (frac << 13)
    };
    f32::from_bits(out)
}

fn qwen35_mtp_name_mapping(layer_index: usize) -> Vec<(&'static str, String)> {
    vec![
        (
            "mtp.fc.weight",
            format!("blk.{layer_index}.nextn.eh_proj.weight"),
        ),
        (
            "mtp.pre_fc_norm_embedding.weight",
            format!("blk.{layer_index}.nextn.enorm.weight"),
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            format!("blk.{layer_index}.nextn.hnorm.weight"),
        ),
        (
            "mtp.norm.weight",
            format!("blk.{layer_index}.nextn.shared_head_norm.weight"),
        ),
        (
            "mtp.layers.0.input_layernorm.weight",
            format!("blk.{layer_index}.attn_norm.weight"),
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            format!("blk.{layer_index}.attn_q.weight"),
        ),
        (
            "mtp.layers.0.self_attn.k_proj.weight",
            format!("blk.{layer_index}.attn_k.weight"),
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight",
            format!("blk.{layer_index}.attn_v.weight"),
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            format!("blk.{layer_index}.attn_output.weight"),
        ),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            format!("blk.{layer_index}.attn_q_norm.weight"),
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            format!("blk.{layer_index}.attn_k_norm.weight"),
        ),
        (
            "mtp.layers.0.post_attention_layernorm.weight",
            format!("blk.{layer_index}.post_attention_norm.weight"),
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            format!("blk.{layer_index}.ffn_gate.weight"),
        ),
        (
            "mtp.layers.0.mlp.up_proj.weight",
            format!("blk.{layer_index}.ffn_up.weight"),
        ),
        (
            "mtp.layers.0.mlp.down_proj.weight",
            format!("blk.{layer_index}.ffn_down.weight"),
        ),
    ]
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, LoaderError> {
    let offset = *pos;
    let raw = read_exact(bytes, pos, 4)?;
    Ok(u32::from_le_bytes(raw.try_into().map_err(|_| {
        parse_error(offset, "failed to read u32 from MTP sidecar")
    })?))
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, LoaderError> {
    let offset = *pos;
    let raw = read_exact(bytes, pos, 8)?;
    Ok(u64::from_le_bytes(raw.try_into().map_err(|_| {
        parse_error(offset, "failed to read u64 from MTP sidecar")
    })?))
}

fn read_exact<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8], LoaderError> {
    let start = *pos;
    let end = start
        .checked_add(len)
        .ok_or_else(|| parse_error(start, "MTP sidecar read offset overflows"))?;
    if end > bytes.len() {
        return Err(parse_error(start, "unexpected end of MTP sidecar"));
    }
    *pos = end;
    Ok(&bytes[start..end])
}

fn parse_error(offset: usize, msg: &str) -> LoaderError {
    LoaderError::ParseError {
        offset,
        msg: msg.to_string(),
    }
}
