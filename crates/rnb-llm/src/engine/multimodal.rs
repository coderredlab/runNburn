use rnb_core::image::RgbImage;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;
use rnb_model_gemma::{encode_gemma4_vision_intermediate, prepare_gemma4_vision_intermediate};
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
use rnb_model_qwen::encode_qwen36_vision_intermediate;
use rnb_model_qwen::prepare_qwen36_vision_intermediate;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use rnb_model_qwen::{encode_qwen36_vision_intermediate_with_executor, Qwen36VisionExecutor};

use super::models::gemma::{
    apply_embedding_scale, apply_embedding_scale_inplace, gemma_ple_pre_emb_scale_base,
    prepare_gemma_per_layer_base,
};
use super::prefill::{
    run_prefill_layers_cpu_range, run_prefill_layers_cpu_range_non_causal,
    run_prefill_layers_cpu_range_with_positions,
};
use super::{finalize_prefill_logits, kernels, Engine};
use crate::error::{LlmError, Result};
use crate::multimodal::{
    assemble_prompt_hidden, compile_gemma4_prompt, compile_qwen36_prompt, image_fingerprint,
    CompiledPrompt, PromptPositions, PromptSpan, SequenceCursor,
};

#[cfg(all(feature = "metal", not(feature = "cuda")))]
struct MetalVisionExecutor {
    enabled: bool,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
impl Qwen36VisionExecutor for MetalVisionExecutor {
    fn linear_bf16(
        &mut self,
        weight: &[u16],
        input: &[f32],
        bias: &[f32],
        rows: usize,
        cols: usize,
        sequence_length: usize,
    ) -> std::result::Result<Option<Vec<f32>>, String> {
        if !self.enabled {
            return Ok(None);
        }
        crate::runtime::metal::metal_qwen36_vision_linear_bf16(
            weight,
            input,
            bias,
            rows,
            cols,
            sequence_length,
        )
        .map_err(|error| error.to_string())
    }

    fn full_attention(
        &mut self,
        qkv: &[f32],
        embedding_length: usize,
        head_count: usize,
        sequence_length: usize,
    ) -> std::result::Result<Option<Vec<f32>>, String> {
        if !self.enabled {
            return Ok(None);
        }
        crate::runtime::metal::metal_qwen36_vision_full_attention(
            qkv,
            embedding_length,
            head_count,
            sequence_length,
        )
        .map_err(|error| error.to_string())
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_vision_enabled() -> bool {
    super::policy::env_string("RNB_METAL_VISION")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

impl Engine {
    pub fn has_vision_projector(&self) -> bool {
        self.vision_projector.is_some()
    }

    pub(crate) fn compile_multimodal_prompt(
        &self,
        rendered_prompt: &str,
        image: &RgbImage,
    ) -> Result<CompiledPrompt> {
        let projector = self.vision_projector.as_ref().ok_or_else(|| {
            LlmError::InvalidChatRequest(
                "image input requires an explicitly configured vision projector".into(),
            )
        })?;

        let mut token_ids = Vec::new();
        if self.tokenizer.should_add_bos() {
            token_ids.push(self.tokenizer.vocab.special.bos);
        }
        token_ids.extend(self.tokenizer.encode(rendered_prompt));

        let compiled = match self.architecture {
            ModelArchitecture::Qwen35MoE => {
                let image_pad_token_id =
                    self.tokenizer.token_id("<|image_pad|>").ok_or_else(|| {
                        LlmError::Tokenizer("Qwen3.6 tokenizer is missing <|image_pad|>".into())
                    })?;
                let intermediate = prepare_qwen36_vision_intermediate(projector, image)
                    .map_err(|error| LlmError::Forward(error.to_string()))?;
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                let vision = {
                    let mut executor = MetalVisionExecutor {
                        enabled: metal_vision_enabled(),
                    };
                    encode_qwen36_vision_intermediate_with_executor(
                        projector,
                        intermediate,
                        &mut executor,
                    )
                    .map_err(|error| LlmError::Forward(error.to_string()))?
                };
                #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
                let vision = encode_qwen36_vision_intermediate(projector, intermediate)
                    .map_err(|error| LlmError::Forward(error.to_string()))?;
                compile_qwen36_prompt(
                    token_ids,
                    image_pad_token_id,
                    vision,
                    image_fingerprint(image),
                )?
            }
            ModelArchitecture::Gemma4 => {
                let image_placeholder_id =
                    self.tokenizer.token_id("<|image|>").ok_or_else(|| {
                        LlmError::Tokenizer("Gemma 4 tokenizer is missing <|image|>".into())
                    })?;
                let image_begin_id = self.tokenizer.token_id("<|image>").ok_or_else(|| {
                    LlmError::Tokenizer("Gemma 4 tokenizer is missing <|image>".into())
                })?;
                let image_end_id = self.tokenizer.token_id("<image|>").ok_or_else(|| {
                    LlmError::Tokenizer("Gemma 4 tokenizer is missing <image|>".into())
                })?;
                let intermediate = prepare_gemma4_vision_intermediate(projector, image)
                    .map_err(|error| LlmError::Forward(error.to_string()))?;
                let vision = encode_gemma4_vision_intermediate(projector, intermediate)
                    .map_err(|error| LlmError::Forward(error.to_string()))?;
                compile_gemma4_prompt(
                    token_ids,
                    image_placeholder_id,
                    image_begin_id,
                    image_end_id,
                    vision,
                    image_fingerprint(image),
                )?
            }
            architecture => {
                return Err(LlmError::InvalidChatRequest(format!(
                    "image input is unsupported for {architecture:?}"
                )));
            }
        };
        if compiled.executed_rows > self.metadata.max_seq_len {
            return Err(LlmError::InvalidChatRequest(format!(
                "multimodal prompt executes {} rows, exceeding context limit {}",
                compiled.executed_rows, self.metadata.max_seq_len
            )));
        }
        Ok(compiled)
    }

    pub fn compile_qwen36_multimodal_prompt(
        &self,
        rendered_prompt: &str,
        image: &RgbImage,
    ) -> Result<CompiledPrompt> {
        if self.architecture != ModelArchitecture::Qwen35MoE {
            return Err(LlmError::InvalidChatRequest(format!(
                "Qwen3.6 image input requires Qwen35MoE, got {:?}",
                self.architecture
            )));
        }
        self.compile_multimodal_prompt(rendered_prompt, image)
    }

    /// Runs a fresh Qwen3.6 multimodal prefill and returns the first decode-step logits.
    ///
    /// This is a diagnostic seam for numerical comparison with reference runtimes.
    pub fn debug_qwen36_multimodal_prefill_logits(
        &mut self,
        rendered_prompt: &str,
        image: &RgbImage,
    ) -> Result<Vec<f32>> {
        if self.architecture != ModelArchitecture::Qwen35MoE {
            return Err(LlmError::InvalidChatRequest(format!(
                "Qwen3.6 image input requires Qwen35MoE, got {:?}",
                self.architecture
            )));
        }
        self.clear_sequence_state()?;
        let prompt = self.compile_multimodal_prompt(rendered_prompt, image)?;
        self.forward_compiled_prompt(&prompt)
    }

    /// Runs a fresh Gemma 4 multimodal prefill and returns the first decode-step logits.
    pub fn debug_gemma4_multimodal_prefill_logits(
        &mut self,
        rendered_prompt: &str,
        image: &RgbImage,
    ) -> Result<Vec<f32>> {
        if self.architecture != ModelArchitecture::Gemma4 {
            return Err(LlmError::InvalidChatRequest(format!(
                "Gemma 4 image input requires Gemma4, got {:?}",
                self.architecture
            )));
        }
        self.clear_sequence_state()?;
        let prompt = self.compile_multimodal_prompt(rendered_prompt, image)?;
        self.forward_compiled_prompt(&prompt)
    }

    pub(crate) fn forward_compiled_prompt(&mut self, prompt: &CompiledPrompt) -> Result<Vec<f32>> {
        crate::generate::check_generation_cancellation()?;
        if prompt.executed_rows == 0 {
            return Err(LlmError::InvalidChatRequest(
                "multimodal prompt must execute at least one row".into(),
            ));
        }
        if prompt.physical_token_ids.len() != prompt.executed_rows {
            return Err(LlmError::Forward(format!(
                "compiled prompt has {} physical token IDs for {} rows",
                prompt.physical_token_ids.len(),
                prompt.executed_rows
            )));
        }
        if self.kv_cache.current_len() != 0 || self.sequence_cursor.is_some() {
            return Err(LlmError::Forward(
                "multimodal prefill requires a fresh sequence; resume and prefix reuse are unsupported"
                    .into(),
            ));
        }

        match self.architecture {
            ModelArchitecture::Qwen35MoE => self.forward_qwen36_compiled_prompt(prompt),
            ModelArchitecture::Gemma4 => self.forward_gemma4_compiled_prompt(prompt),
            architecture => Err(LlmError::Forward(format!(
                "compiled multimodal prompt cannot run on {architecture:?}"
            ))),
        }
    }

    fn forward_qwen36_compiled_prompt(&mut self, prompt: &CompiledPrompt) -> Result<Vec<f32>> {
        let positions = match &prompt.positions {
            PromptPositions::QwenImrope(positions) if positions.len() == prompt.executed_rows => {
                positions
            }
            PromptPositions::QwenImrope(positions) => {
                return Err(LlmError::Forward(format!(
                    "compiled prompt has {} positions for {} physical rows",
                    positions.len(),
                    prompt.executed_rows
                )));
            }
            PromptPositions::Linear => {
                return Err(LlmError::Forward(
                    "Qwen3.6 multimodal prompt requires IMRoPE positions".into(),
                ));
            }
        };
        let weights = self.weights.as_ref().ok_or_else(|| {
            LlmError::Forward("multimodal prefill requires loaded model weights".into())
        })?;
        let hidden_dim = self.metadata.hidden_dim;
        let hidden = assemble_prompt_hidden(
            prompt,
            hidden_dim,
            |ids| {
                let gathered = weights.token_embd.gather(ids)?;
                Ok(kernels::tensor_as_f32_slice(&gathered).to_vec())
            },
            |values| {
                apply_embedding_scale_inplace(values, &self.metadata, self.architecture);
            },
        )?;
        let hidden = Tensor::from_vec(hidden, &[prompt.executed_rows, hidden_dim]);
        let rope_sections: [usize; 4] =
            self.metadata
                .rope_sections
                .as_slice()
                .try_into()
                .map_err(|_| {
                    LlmError::Forward(format!(
                        "Qwen3.6 IMRoPE requires four sections, got {:?}",
                        self.metadata.rope_sections
                    ))
                })?;
        let seq_len = prompt.executed_rows;
        let output = run_prefill_layers_cpu_range_with_positions(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            hidden,
            0..self.metadata.num_layers,
            seq_len,
            0,
            positions,
            rope_sections,
            self.metadata.num_heads,
            self.metadata.num_kv_heads,
            self.metadata.head_dim,
            self.metadata.num_kv_heads * self.metadata.head_dim,
            self.metadata.rope_theta,
            self.metadata.norm_eps,
        )?;
        let logits = finalize_prefill_logits(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            output,
            seq_len,
            0,
            self.metadata.norm_eps,
            Some(&mut self.last_layer_hidden_cached),
        )?;
        self.sequence_cursor = Some(SequenceCursor::multimodal(prompt));
        Ok(logits)
    }

    fn forward_gemma4_compiled_prompt(&mut self, prompt: &CompiledPrompt) -> Result<Vec<f32>> {
        if prompt.positions != PromptPositions::Linear {
            return Err(LlmError::Forward(
                "Gemma 4 multimodal prompt requires linear positions".into(),
            ));
        }
        let weights = self.weights.as_ref().ok_or_else(|| {
            LlmError::Forward("multimodal prefill requires loaded model weights".into())
        })?;
        let hidden_dim = self.metadata.hidden_dim;
        let kv_dim = self.metadata.num_kv_heads * self.metadata.head_dim;
        let mut last_output = None;

        for span in &prompt.spans {
            crate::generate::check_generation_cancellation()?;
            let pos_start = self.kv_cache.current_len();
            let (raw_hidden, hidden, token_ids, non_causal) = match span {
                PromptSpan::Tokens { ids } => {
                    let raw_hidden = weights.token_embd.gather(ids)?;
                    let hidden = apply_embedding_scale(
                        raw_hidden.clone(),
                        &self.metadata,
                        self.architecture,
                    );
                    (raw_hidden, hidden, ids.clone(), false)
                }
                PromptSpan::Embeddings {
                    rows,
                    width,
                    values,
                    ..
                } => {
                    if *width != hidden_dim || values.len() != rows * width {
                        return Err(LlmError::Forward(format!(
                            "Gemma 4 image embedding shape [{rows}, {width}] does not match hidden width {hidden_dim}"
                        )));
                    }
                    let hidden = Tensor::from_slice(values, &[*rows, *width]);
                    (hidden.clone(), hidden, vec![0; *rows], true)
                }
            };
            let seq_len = token_ids.len();
            if seq_len == 0 {
                continue;
            }
            let ple_base = prepare_gemma_per_layer_base(
                weights,
                if gemma_ple_pre_emb_scale_base() {
                    &raw_hidden
                } else {
                    &hidden
                },
                &token_ids,
                &self.metadata,
                self.architecture,
                self.metadata.norm_eps,
            )?;
            let output = if non_causal {
                run_prefill_layers_cpu_range_non_causal(
                    &mut self.kv_cache,
                    &self.metadata,
                    self.architecture,
                    weights,
                    ple_base.as_ref(),
                    hidden,
                    0..self.metadata.num_layers,
                    seq_len,
                    pos_start,
                    self.metadata.num_heads,
                    self.metadata.num_kv_heads,
                    self.metadata.head_dim,
                    kv_dim,
                    self.metadata.rope_theta,
                    self.metadata.norm_eps,
                )?
            } else {
                run_prefill_layers_cpu_range(
                    &mut self.kv_cache,
                    &self.metadata,
                    self.architecture,
                    weights,
                    ple_base.as_ref(),
                    hidden,
                    0..self.metadata.num_layers,
                    seq_len,
                    pos_start,
                    self.metadata.num_heads,
                    self.metadata.num_kv_heads,
                    self.metadata.head_dim,
                    kv_dim,
                    self.metadata.rope_theta,
                    self.metadata.norm_eps,
                )?
            };
            last_output = Some((output, seq_len, pos_start));
        }

        if self.kv_cache.current_len() != prompt.executed_rows {
            return Err(LlmError::Forward(format!(
                "Gemma 4 multimodal prefill cached {} rows, expected {}",
                self.kv_cache.current_len(),
                prompt.executed_rows
            )));
        }
        let (output, seq_len, pos_start) = last_output.ok_or_else(|| {
            LlmError::Forward("Gemma 4 multimodal prompt produced no hidden rows".into())
        })?;
        let logits = finalize_prefill_logits(
            &mut self.kv_cache,
            &self.metadata,
            self.architecture,
            weights,
            output,
            seq_len,
            pos_start,
            self.metadata.norm_eps,
            Some(&mut self.last_layer_hidden_cached),
        )?;
        self.sequence_cursor = Some(SequenceCursor::multimodal(prompt));
        Ok(logits)
    }
}
