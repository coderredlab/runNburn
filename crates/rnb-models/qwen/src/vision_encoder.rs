use std::mem;

use rayon::prelude::*;
use rnb_cpu::gemm::f32_gemv::gemv_bf16;
use rnb_loader::{GGMLType, LoadedVisionProjector};

use super::vision::{inspect_qwen36_vision_projector, Qwen36VisionError};
use super::vision_math::{
    add_in_place, apply_vision_mrope, ensure_finite, full_attention as full_attention_cpu,
    gelu_in_place, layer_norm_affine,
};
use super::vision_preprocess::{
    tensor_f32, tensor_stats, Qwen36TensorStats, Qwen36VisionIntermediate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36VisionLayerSummary {
    pub layer_index: usize,
    pub stats: Qwen36TensorStats,
    pub first_values: [f32; 8],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36VisionOutput {
    pub target_width: usize,
    pub target_height: usize,
    pub patch_grid_width: usize,
    pub patch_grid_height: usize,
    pub merged_grid_width: usize,
    pub merged_grid_height: usize,
    pub projection_dim: usize,
    pub layer_summaries: Vec<Qwen36VisionLayerSummary>,
    pub post_layer_norm_stats: Qwen36TensorStats,
    pub embedding_stats: Qwen36TensorStats,
    pub embeddings: Vec<f32>,
}

pub trait Qwen36VisionExecutor {
    fn linear_bf16(
        &mut self,
        weight: &[u16],
        input: &[f32],
        bias: &[f32],
        rows: usize,
        cols: usize,
        sequence_length: usize,
    ) -> Result<Option<Vec<f32>>, String>;

    fn full_attention(
        &mut self,
        qkv: &[f32],
        embedding_length: usize,
        head_count: usize,
        sequence_length: usize,
    ) -> Result<Option<Vec<f32>>, String>;
}

pub fn encode_qwen36_vision_intermediate(
    projector: &LoadedVisionProjector,
    intermediate: Qwen36VisionIntermediate,
) -> Result<Qwen36VisionOutput, Qwen36VisionError> {
    let mut executor = CpuVisionExecutor;
    encode_qwen36_vision_intermediate_with_executor(projector, intermediate, &mut executor)
}

pub fn encode_qwen36_vision_intermediate_with_executor(
    projector: &LoadedVisionProjector,
    intermediate: Qwen36VisionIntermediate,
    executor: &mut dyn Qwen36VisionExecutor,
) -> Result<Qwen36VisionOutput, Qwen36VisionError> {
    let capability = inspect_qwen36_vision_projector(&projector.descriptor)?;
    validate_intermediate(&intermediate, &capability)?;

    let patch_count = intermediate
        .patch_grid_width
        .checked_mul(intermediate.patch_grid_height)
        .ok_or_else(|| error("vision patch count overflows usize"))?;
    let mut hidden = intermediate.patch_embeddings;
    let mut layer_summaries = Vec::with_capacity(capability.block_count);

    for layer in 0..capability.block_count {
        let prefix = format!("v.blk.{layer}");
        let ln1_weight = tensor_f32(projector, &format!("{prefix}.ln1.weight"))?;
        let ln1_bias = tensor_f32(projector, &format!("{prefix}.ln1.bias"))?;
        let normalized = layer_norm_affine(
            &hidden,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &ln1_weight,
            &ln1_bias,
        )?;

        let qkv_bias = tensor_f32(projector, &format!("{prefix}.attn_qkv.bias"))?;
        let qkv_width = capability
            .embedding_length
            .checked_mul(3)
            .ok_or_else(|| error("vision QKV width overflows usize"))?;
        let mut qkv = linear_bf16(
            executor,
            projector,
            &format!("{prefix}.attn_qkv.weight"),
            &normalized,
            &qkv_bias,
            qkv_width,
            capability.embedding_length,
            patch_count,
        )?;
        apply_vision_mrope(
            &mut qkv,
            intermediate.patch_grid_width,
            intermediate.patch_grid_height,
            capability.spatial_merge_size,
            capability.embedding_length,
            capability.head_count,
        )?;
        let attended = full_attention_with_executor(
            executor,
            &qkv,
            capability.embedding_length,
            capability.head_count,
            patch_count,
        )?;
        let attention_bias = tensor_f32(projector, &format!("{prefix}.attn_out.bias"))?;
        let attention_output = linear_bf16(
            executor,
            projector,
            &format!("{prefix}.attn_out.weight"),
            &attended,
            &attention_bias,
            capability.embedding_length,
            capability.embedding_length,
            patch_count,
        )?;
        add_in_place(&mut hidden, &attention_output)?;

        let ln2_weight = tensor_f32(projector, &format!("{prefix}.ln2.weight"))?;
        let ln2_bias = tensor_f32(projector, &format!("{prefix}.ln2.bias"))?;
        let normalized = layer_norm_affine(
            &hidden,
            capability.embedding_length,
            capability.layer_norm_epsilon,
            &ln2_weight,
            &ln2_bias,
        )?;
        let up_bias = tensor_f32(projector, &format!("{prefix}.ffn_up.bias"))?;
        let mut feed_forward = linear_bf16(
            executor,
            projector,
            &format!("{prefix}.ffn_up.weight"),
            &normalized,
            &up_bias,
            capability.feed_forward_length,
            capability.embedding_length,
            patch_count,
        )?;
        gelu_in_place(&mut feed_forward);
        let down_bias = tensor_f32(projector, &format!("{prefix}.ffn_down.bias"))?;
        let feed_forward = linear_bf16(
            executor,
            projector,
            &format!("{prefix}.ffn_down.weight"),
            &feed_forward,
            &down_bias,
            capability.embedding_length,
            capability.feed_forward_length,
            patch_count,
        )?;
        add_in_place(&mut hidden, &feed_forward)?;
        ensure_finite(&hidden, &format!("vision block {layer} output"))?;

        let mut first_values = [0.0f32; 8];
        first_values.copy_from_slice(&hidden[..8]);
        layer_summaries.push(Qwen36VisionLayerSummary {
            layer_index: layer,
            stats: tensor_stats(&hidden)?,
            first_values,
        });
    }

    let post_weight = tensor_f32(projector, "v.post_ln.weight")?;
    let post_bias = tensor_f32(projector, "v.post_ln.bias")?;
    let post_normalized = layer_norm_affine(
        &hidden,
        capability.embedding_length,
        capability.layer_norm_epsilon,
        &post_weight,
        &post_bias,
    )?;
    let post_layer_norm_stats = tensor_stats(&post_normalized)?;

    let merge_area = capability
        .spatial_merge_size
        .checked_mul(capability.spatial_merge_size)
        .ok_or_else(|| error("vision merge area overflows usize"))?;
    let merged_count = patch_count / merge_area;
    let merger_width = capability
        .embedding_length
        .checked_mul(merge_area)
        .ok_or_else(|| error("vision merger width overflows usize"))?;
    let merger_bias = tensor_f32(projector, "mm.0.bias")?;
    let mut merged = linear_bf16(
        executor,
        projector,
        "mm.0.weight",
        &post_normalized,
        &merger_bias,
        merger_width,
        merger_width,
        merged_count,
    )?;
    gelu_in_place(&mut merged);
    let projection_bias = tensor_f32(projector, "mm.2.bias")?;
    let embeddings = linear_bf16(
        executor,
        projector,
        "mm.2.weight",
        &merged,
        &projection_bias,
        capability.projection_dim,
        merger_width,
        merged_count,
    )?;
    ensure_finite(&embeddings, "vision merger output")?;
    let embedding_stats = tensor_stats(&embeddings)?;

    Ok(Qwen36VisionOutput {
        target_width: intermediate.target_width,
        target_height: intermediate.target_height,
        patch_grid_width: intermediate.patch_grid_width,
        patch_grid_height: intermediate.patch_grid_height,
        merged_grid_width: intermediate.merged_grid_width,
        merged_grid_height: intermediate.merged_grid_height,
        projection_dim: capability.projection_dim,
        layer_summaries,
        post_layer_norm_stats,
        embedding_stats,
        embeddings,
    })
}

fn validate_intermediate(
    intermediate: &Qwen36VisionIntermediate,
    capability: &super::vision::Qwen36VisionCapability,
) -> Result<(), Qwen36VisionError> {
    if intermediate.embedding_length != capability.embedding_length {
        return Err(error(format!(
            "vision intermediate width is {}, expected {}",
            intermediate.embedding_length, capability.embedding_length
        )));
    }
    if intermediate.target_width / capability.patch_size != intermediate.patch_grid_width
        || intermediate.target_height / capability.patch_size != intermediate.patch_grid_height
    {
        return Err(error("vision intermediate target and patch grids disagree"));
    }
    if intermediate.patch_grid_width % capability.spatial_merge_size != 0
        || intermediate.patch_grid_height % capability.spatial_merge_size != 0
    {
        return Err(error(
            "vision patch grid is not divisible by spatial merge size",
        ));
    }
    if intermediate.merged_grid_width
        != intermediate.patch_grid_width / capability.spatial_merge_size
        || intermediate.merged_grid_height
            != intermediate.patch_grid_height / capability.spatial_merge_size
    {
        return Err(error("vision intermediate merged grid is inconsistent"));
    }
    let expected = intermediate
        .patch_grid_width
        .checked_mul(intermediate.patch_grid_height)
        .and_then(|count| count.checked_mul(capability.embedding_length))
        .ok_or_else(|| error("vision intermediate element count overflows usize"))?;
    if intermediate.patch_embeddings.len() != expected {
        return Err(error(format!(
            "vision intermediate has {} values, expected {expected}",
            intermediate.patch_embeddings.len()
        )));
    }
    Ok(())
}

fn linear_bf16(
    executor: &mut dyn Qwen36VisionExecutor,
    projector: &LoadedVisionProjector,
    weight_name: &str,
    input: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Qwen36VisionError> {
    let expected_input = cols
        .checked_mul(sequence_length)
        .ok_or_else(|| error(format!("tensor '{weight_name}' input size overflows usize")))?;
    if input.len() != expected_input {
        return Err(error(format!(
            "tensor '{weight_name}' input has {} values, expected {expected_input}",
            input.len()
        )));
    }
    if bias.len() != rows {
        return Err(error(format!(
            "tensor '{weight_name}' bias has {} values, expected {rows}",
            bias.len()
        )));
    }
    let weight = tensor_bf16_words(projector, weight_name, rows, cols)?;
    let output_len = rows.checked_mul(sequence_length).ok_or_else(|| {
        error(format!(
            "tensor '{weight_name}' output size overflows usize"
        ))
    })?;
    match executor
        .linear_bf16(weight, input, bias, rows, cols, sequence_length)
        .map_err(|message| error(format!("tensor '{weight_name}' executor failed: {message}")))?
    {
        Some(output) if output.len() == output_len => Ok(output),
        Some(output) => Err(error(format!(
            "tensor '{weight_name}' executor returned {} values, expected {output_len}",
            output.len()
        ))),
        None => Ok(linear_bf16_cpu(
            weight,
            input,
            bias,
            rows,
            cols,
            sequence_length,
        )),
    }
}

fn full_attention_with_executor(
    executor: &mut dyn Qwen36VisionExecutor,
    qkv: &[f32],
    embedding_length: usize,
    head_count: usize,
    sequence_length: usize,
) -> Result<Vec<f32>, Qwen36VisionError> {
    let output_len = sequence_length
        .checked_mul(embedding_length)
        .ok_or_else(|| error("vision attention output size overflows usize"))?;
    match executor
        .full_attention(qkv, embedding_length, head_count, sequence_length)
        .map_err(|message| error(format!("vision attention executor failed: {message}")))?
    {
        Some(output) if output.len() == output_len => Ok(output),
        Some(output) => Err(error(format!(
            "vision attention executor returned {} values, expected {output_len}",
            output.len(),
        ))),
        None => full_attention_cpu(qkv, embedding_length, head_count, sequence_length),
    }
}
struct CpuVisionExecutor;

impl Qwen36VisionExecutor for CpuVisionExecutor {
    fn linear_bf16(
        &mut self,
        weight: &[u16],
        input: &[f32],
        bias: &[f32],
        rows: usize,
        cols: usize,
        sequence_length: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        Ok(Some(linear_bf16_cpu(
            weight,
            input,
            bias,
            rows,
            cols,
            sequence_length,
        )))
    }

    fn full_attention(
        &mut self,
        qkv: &[f32],
        embedding_length: usize,
        head_count: usize,
        sequence_length: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        full_attention_cpu(qkv, embedding_length, head_count, sequence_length)
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

fn linear_bf16_cpu(
    weight: &[u16],
    input: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    sequence_length: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * sequence_length];
    gemv_bf16(weight, input, &mut output, rows, cols, sequence_length);
    output.par_chunks_mut(rows).for_each(|row| {
        for index in 0..rows {
            row[index] += bias[index];
        }
    });
    output
}

fn tensor_bf16_words<'a>(
    projector: &'a LoadedVisionProjector,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<&'a [u16], Qwen36VisionError> {
    if cfg!(target_endian = "big") {
        return Err(error(
            "BF16 vision execution requires a little-endian target",
        ));
    }
    let descriptor = projector
        .descriptor
        .tensors
        .get(name)
        .ok_or_else(|| error(format!("missing projector tensor '{name}'")))?;
    if descriptor.ggml_type != GGMLType::BF16 {
        return Err(error(format!(
            "tensor '{name}' must be BF16 for CPU vision execution"
        )));
    }
    let expected_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| error(format!("tensor '{name}' element count overflows usize")))?;
    let descriptor_elements = descriptor
        .shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| error(format!("tensor '{name}' descriptor size overflows usize")))?;
    if descriptor_elements != expected_elements {
        return Err(error(format!(
            "tensor '{name}' has {descriptor_elements} elements, expected {expected_elements}"
        )));
    }
    let tensor = projector
        .weights
        .get(name)
        .ok_or_else(|| error(format!("missing mapped projector tensor '{name}'")))?;
    let bytes = tensor
        .as_bytes()
        .ok_or_else(|| error(format!("tensor '{name}' has no contiguous host bytes")))?;
    let expected_bytes = expected_elements
        .checked_mul(mem::size_of::<u16>())
        .ok_or_else(|| error(format!("tensor '{name}' byte count overflows usize")))?;
    if bytes.len() != expected_bytes {
        return Err(error(format!(
            "tensor '{name}' has {} mapped bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }
    if bytes.as_ptr().align_offset(mem::align_of::<u16>()) != 0 {
        return Err(error(format!(
            "tensor '{name}' BF16 bytes are not u16-aligned"
        )));
    }
    // SAFETY: the mapped slice has exact u16 length and alignment, checked above. GGUF and
    // supported targets are little-endian, so each u16 preserves the BF16 bit pattern.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), expected_elements) })
}

fn error(message: impl Into<String>) -> Qwen36VisionError {
    Qwen36VisionError::new(message)
}
