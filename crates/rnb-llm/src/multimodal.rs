use rnb_model_qwen::{
    plan_qwen36_multimodal_positions, Qwen36PositionSpan, Qwen36RgbImage, Qwen36VisionOutput,
};
use sha2::{Digest, Sha256};

use crate::error::{LlmError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum PromptSpan {
    Tokens {
        ids: Vec<u32>,
    },
    Embeddings {
        rows: usize,
        width: usize,
        values: Vec<f32>,
        grid_width: usize,
        grid_height: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPrompt {
    pub spans: Vec<PromptSpan>,
    pub positions: Vec<[u32; 4]>,
    pub executed_rows: usize,
    pub logical_position_end: u32,
    pub sampler_token_ids: Vec<u32>,
    pub(crate) image_fingerprint: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SequenceCursor {
    pub physical_rows: usize,
    pub logical_position: u32,
    pub token_count: usize,
    pub image_fingerprint: [u8; 32],
}

impl SequenceCursor {
    pub(crate) fn qwen_multimodal(prompt: &CompiledPrompt) -> Self {
        Self {
            physical_rows: prompt.executed_rows,
            logical_position: prompt.logical_position_end,
            token_count: prompt.sampler_token_ids.len(),
            image_fingerprint: prompt.image_fingerprint,
        }
    }
}

pub(crate) fn qwen36_image_fingerprint(image: &Qwen36RgbImage) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((image.width() as u64).to_le_bytes());
    hasher.update((image.height() as u64).to_le_bytes());
    hasher.update(image.pixels());
    hasher.finalize().into()
}

pub(crate) fn compile_qwen36_prompt(
    token_ids: Vec<u32>,
    image_pad_token_id: u32,
    vision: Qwen36VisionOutput,
    image_fingerprint: [u8; 32],
) -> Result<CompiledPrompt> {
    let placeholder_indices = token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image_pad_token_id).then_some(index))
        .collect::<Vec<_>>();
    if placeholder_indices.len() != 1 {
        return Err(LlmError::InvalidChatRequest(format!(
            "Qwen3.6 multimodal prompt must contain exactly one <|image_pad|> token, got {}",
            placeholder_indices.len()
        )));
    }
    if vision.projection_dim == 0 {
        return Err(LlmError::Forward(
            "Qwen3.6 vision projection width must be positive".into(),
        ));
    }
    let image_rows = vision
        .merged_grid_width
        .checked_mul(vision.merged_grid_height)
        .ok_or_else(|| {
            LlmError::Forward("Qwen3.6 image embedding row count overflows usize".into())
        })?;
    let expected_values = image_rows
        .checked_mul(vision.projection_dim)
        .ok_or_else(|| {
            LlmError::Forward("Qwen3.6 image embedding value count overflows usize".into())
        })?;
    if vision.embeddings.len() != expected_values {
        return Err(LlmError::Forward(format!(
            "Qwen3.6 image embedding has {} values, expected {expected_values}",
            vision.embeddings.len()
        )));
    }

    let placeholder = placeholder_indices[0];
    let mut spans = Vec::with_capacity(3);
    if placeholder > 0 {
        spans.push(PromptSpan::Tokens {
            ids: token_ids[..placeholder].to_vec(),
        });
    }
    spans.push(PromptSpan::Embeddings {
        rows: image_rows,
        width: vision.projection_dim,
        values: vision.embeddings,
        grid_width: vision.merged_grid_width,
        grid_height: vision.merged_grid_height,
    });
    if placeholder + 1 < token_ids.len() {
        spans.push(PromptSpan::Tokens {
            ids: token_ids[placeholder + 1..].to_vec(),
        });
    }

    let position_spans = spans
        .iter()
        .map(|span| match span {
            PromptSpan::Tokens { ids } => Qwen36PositionSpan::Text { rows: ids.len() },
            PromptSpan::Embeddings {
                grid_width,
                grid_height,
                ..
            } => Qwen36PositionSpan::Image {
                grid_width: *grid_width,
                grid_height: *grid_height,
            },
        })
        .collect::<Vec<_>>();
    let position_plan = plan_qwen36_multimodal_positions(&position_spans, 0)
        .map_err(|error| LlmError::Forward(error.to_string()))?;
    let sampler_token_ids = spans
        .iter()
        .filter_map(|span| match span {
            PromptSpan::Tokens { ids } => Some(ids.as_slice()),
            PromptSpan::Embeddings { .. } => None,
        })
        .flatten()
        .copied()
        .collect();

    Ok(CompiledPrompt {
        spans,
        positions: position_plan.positions,
        executed_rows: position_plan.physical_rows,
        logical_position_end: position_plan.logical_position_end,
        sampler_token_ids,
        image_fingerprint,
    })
}

pub(crate) fn assemble_prompt_hidden(
    prompt: &CompiledPrompt,
    hidden_width: usize,
    mut gather_tokens: impl FnMut(&[u32]) -> Result<Vec<f32>>,
    mut scale_token_embeddings: impl FnMut(&mut [f32]),
) -> Result<Vec<f32>> {
    let expected_values = prompt
        .executed_rows
        .checked_mul(hidden_width)
        .ok_or_else(|| LlmError::Forward("mixed prompt hidden size overflows usize".into()))?;
    let mut hidden = Vec::with_capacity(expected_values);

    for span in &prompt.spans {
        match span {
            PromptSpan::Tokens { ids } => {
                let mut values = gather_tokens(ids)?;
                let expected = ids.len().checked_mul(hidden_width).ok_or_else(|| {
                    LlmError::Forward("token embedding span size overflows usize".into())
                })?;
                if values.len() != expected {
                    return Err(LlmError::Forward(format!(
                        "token embedding span has {} values, expected {expected}",
                        values.len()
                    )));
                }
                scale_token_embeddings(&mut values);
                hidden.extend_from_slice(&values);
            }
            PromptSpan::Embeddings {
                rows,
                width,
                values,
                ..
            } => {
                if *width != hidden_width {
                    return Err(LlmError::Forward(format!(
                        "image embedding width {width} does not match model hidden width {hidden_width}"
                    )));
                }
                let expected = rows.checked_mul(*width).ok_or_else(|| {
                    LlmError::Forward("image embedding span size overflows usize".into())
                })?;
                if values.len() != expected {
                    return Err(LlmError::Forward(format!(
                        "image embedding span has {} values, expected {expected}",
                        values.len()
                    )));
                }
                hidden.extend_from_slice(values);
            }
        }
    }

    if hidden.len() != expected_values {
        return Err(LlmError::Forward(format!(
            "mixed prompt has {} hidden values, expected {expected_values}",
            hidden.len()
        )));
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_model_qwen::Qwen36TensorStats;

    fn vision_output(values: Vec<f32>) -> Qwen36VisionOutput {
        let stats = Qwen36TensorStats {
            count: values.len(),
            mean: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
        };
        Qwen36VisionOutput {
            target_width: 32,
            target_height: 32,
            patch_grid_width: 2,
            patch_grid_height: 2,
            merged_grid_width: 2,
            merged_grid_height: 1,
            projection_dim: 2,
            layer_summaries: Vec::new(),
            post_layer_norm_stats: stats,
            embedding_stats: stats,
            embeddings: values,
        }
    }

    #[test]
    fn compile_replaces_only_image_placeholder_with_physical_rows() {
        let prompt =
            compile_qwen36_prompt(vec![10, 99, 11], 99, vision_output(vec![0.1; 4]), [7; 32])
                .unwrap();

        assert_eq!(prompt.executed_rows, 4);
        assert_eq!(prompt.logical_position_end, 4);
        assert_eq!(prompt.sampler_token_ids, vec![10, 11]);
        assert_eq!(prompt.positions[0], [0, 0, 0, 0]);
        assert_eq!(prompt.positions[1], [1, 1, 1, 0]);
        assert_eq!(prompt.positions[2], [1, 1, 2, 0]);
        assert_eq!(prompt.positions[3], [3, 3, 3, 3]);
    }

    #[test]
    fn assembly_scales_token_rows_but_not_image_rows() {
        let prompt = compile_qwen36_prompt(
            vec![10, 99, 11],
            99,
            vision_output(vec![0.25, -0.5, 0.75, -1.0]),
            [0; 32],
        )
        .unwrap();
        let hidden = assemble_prompt_hidden(
            &prompt,
            2,
            |ids| {
                Ok(ids
                    .iter()
                    .flat_map(|&id| [id as f32, id as f32 + 0.5])
                    .collect())
            },
            |values| values.iter_mut().for_each(|value| *value *= 2.0),
        )
        .unwrap();

        assert_eq!(hidden, vec![20.0, 21.0, 0.25, -0.5, 0.75, -1.0, 22.0, 23.0]);
    }

    #[test]
    fn compile_rejects_missing_or_duplicate_placeholders() {
        let missing = compile_qwen36_prompt(vec![1, 2], 99, vision_output(vec![0.0; 4]), [0; 32])
            .unwrap_err();
        assert!(missing.to_string().contains("got 0"));

        let duplicate =
            compile_qwen36_prompt(vec![99, 99], 99, vision_output(vec![0.0; 4]), [0; 32])
                .unwrap_err();
        assert!(duplicate.to_string().contains("got 2"));
    }
}
