use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
use rnb_model_qwen::encode_qwen36_vision_intermediate;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use rnb_model_qwen::{encode_qwen36_vision_intermediate_with_executor, Qwen36VisionExecutor};
use rnb_model_qwen::{prepare_qwen36_vision_intermediate, Qwen36RgbImage};

use super::models::gemma::apply_embedding_scale_inplace;
use super::prefill::run_prefill_layers_cpu_range_with_positions;
use super::{finalize_prefill_logits, kernels, Engine};
use crate::error::{LlmError, Result};
use crate::multimodal::{
    assemble_prompt_hidden, compile_qwen36_prompt, qwen36_image_fingerprint, CompiledPrompt,
    SequenceCursor,
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

    pub fn compile_qwen36_multimodal_prompt(
        &self,
        rendered_prompt: &str,
        image: &Qwen36RgbImage,
    ) -> Result<CompiledPrompt> {
        if self.architecture != ModelArchitecture::Qwen35MoE {
            return Err(LlmError::InvalidChatRequest(format!(
                "image input requires Qwen35MoE, got {:?}",
                self.architecture
            )));
        }
        let projector = self.vision_projector.as_ref().ok_or_else(|| {
            LlmError::InvalidChatRequest(
                "image input requires an explicitly configured vision projector".into(),
            )
        })?;
        let image_pad_token_id = self.tokenizer.token_id("<|image_pad|>").ok_or_else(|| {
            LlmError::Tokenizer("Qwen3.6 tokenizer is missing <|image_pad|>".into())
        })?;

        let intermediate = prepare_qwen36_vision_intermediate(projector, image)
            .map_err(|error| LlmError::Forward(error.to_string()))?;
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        let vision = {
            let mut executor = MetalVisionExecutor {
                enabled: metal_vision_enabled(),
            };
            encode_qwen36_vision_intermediate_with_executor(projector, intermediate, &mut executor)
                .map_err(|error| LlmError::Forward(error.to_string()))?
        };
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        let vision = encode_qwen36_vision_intermediate(projector, intermediate)
            .map_err(|error| LlmError::Forward(error.to_string()))?;
        let mut token_ids = Vec::new();
        if self.tokenizer.should_add_bos() {
            token_ids.push(self.tokenizer.vocab.special.bos);
        }
        token_ids.extend(self.tokenizer.encode(rendered_prompt));
        let compiled = compile_qwen36_prompt(
            token_ids,
            image_pad_token_id,
            vision,
            qwen36_image_fingerprint(image),
        )?;
        if compiled.executed_rows > self.metadata.max_seq_len {
            return Err(LlmError::InvalidChatRequest(format!(
                "multimodal prompt executes {} rows, exceeding context limit {}",
                compiled.executed_rows, self.metadata.max_seq_len
            )));
        }
        Ok(compiled)
    }

    /// Runs a fresh Qwen3.6 multimodal prefill and returns the first decode-step logits.
    ///
    /// This is a diagnostic seam for numerical comparison with reference runtimes.
    pub fn debug_qwen36_multimodal_prefill_logits(
        &mut self,
        rendered_prompt: &str,
        image: &Qwen36RgbImage,
    ) -> Result<Vec<f32>> {
        self.clear_sequence_state()?;
        let prompt = self.compile_qwen36_multimodal_prompt(rendered_prompt, image)?;
        self.forward_compiled_prompt(&prompt)
    }

    pub(crate) fn forward_compiled_prompt(&mut self, prompt: &CompiledPrompt) -> Result<Vec<f32>> {
        crate::generate::check_generation_cancellation()?;
        if self.architecture != ModelArchitecture::Qwen35MoE {
            return Err(LlmError::Forward(format!(
                "compiled Qwen3.6 prompt cannot run on {:?}",
                self.architecture
            )));
        }
        if prompt.executed_rows == 0 {
            return Err(LlmError::InvalidChatRequest(
                "multimodal prompt must execute at least one row".into(),
            ));
        }
        if prompt.positions.len() != prompt.executed_rows {
            return Err(LlmError::Forward(format!(
                "compiled prompt has {} positions for {} physical rows",
                prompt.positions.len(),
                prompt.executed_rows
            )));
        }
        if self.kv_cache.current_len() != 0 || self.sequence_cursor.is_some() {
            return Err(LlmError::Forward(
                "multimodal prefill requires a fresh sequence; resume and prefix reuse are unsupported"
                    .into(),
            ));
        }

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
            &prompt.positions,
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
        self.sequence_cursor = Some(SequenceCursor::qwen_multimodal(prompt));
        Ok(logits)
    }
}
