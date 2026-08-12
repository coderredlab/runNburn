use super::http::ApiError;
use super::types::PreparedGenerationRequest;
use rnb_llm::{
    generate_stream_multimodal, generate_stream_multimodal_cancellable,
    generate_stream_multimodal_resuming, generate_stream_multimodal_resuming_cancellable,
    parse_assistant_output_with_format, AssistantTurnStreamFilter, Engine, EngineSequenceState,
    GenerationCancellation, ParsedAssistantOutput, TextStopFilter, ToolCallFormat,
};

pub(super) struct GeneratedCompletion {
    pub output: ParsedAssistantOutput,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub max_tokens: usize,
    pub matched_stop: bool,
    pub callback_stopped: bool,
    pub generated_token_ids: Vec<u32>,
    pub prompt_token_ids: Vec<u32>,
    pub cached_prompt_tokens: usize,
}

impl GeneratedCompletion {
    pub fn incomplete(&self) -> bool {
        self.output.tool_calls.is_empty()
            && !self.matched_stop
            && self.output_tokens >= self.max_tokens
    }
}

pub(super) fn run_generation(
    engine: &mut Engine,
    prepared: &PreparedGenerationRequest,
    resume_state: Option<&EngineSequenceState>,
    cancellation: Option<&GenerationCancellation>,
    mut on_text: Option<&mut dyn FnMut(&str) -> bool>,
) -> Result<GeneratedCompletion, ApiError> {
    let tool_mode = !prepared.tool_names.is_empty();
    let stop_on_tool_call = tool_mode && !prepared.parallel_tool_calls;
    let mut content = String::new();
    let mut filter = TextStopFilter::new(prepared.stop_sequences.clone());
    let mut turn_filter = AssistantTurnStreamFilter::new(prepared.muse_protocol);
    let mut callback_stopped = false;
    let mut callback = |piece: &str| {
        filter.push(piece, |text| {
            let should_continue = append_generated_text(&mut content, text, stop_on_tool_call);
            if !tool_mode && !text.is_empty() {
                let callback_continue = turn_filter.push(text, |visible| {
                    if on_text
                        .as_deref_mut()
                        .is_some_and(|callback| !callback(visible))
                    {
                        callback_stopped = true;
                        false
                    } else {
                        true
                    }
                });
                return should_continue && callback_continue;
            }
            should_continue
        })
    };
    let result = if let Some(image) = prepared.image.as_ref() {
        match (resume_state, cancellation) {
            (Some(state), Some(cancellation)) => generate_stream_multimodal_resuming_cancellable(
                engine,
                &prepared.prompt,
                image,
                &prepared.params,
                state,
                cancellation,
                &mut callback,
            ),
            (Some(state), None) => generate_stream_multimodal_resuming(
                engine,
                &prepared.prompt,
                image,
                &prepared.params,
                state,
                &mut callback,
            ),
            (None, Some(cancellation)) => generate_stream_multimodal_cancellable(
                engine,
                &prepared.prompt,
                image,
                &prepared.params,
                cancellation,
                &mut callback,
            ),
            (None, None) => generate_stream_multimodal(
                engine,
                &prepared.prompt,
                image,
                &prepared.params,
                &mut callback,
            ),
        }
    } else {
        match (resume_state, cancellation) {
            (Some(state), Some(cancellation)) => engine.generate_stream_resuming_cancellable(
                &prepared.prompt,
                &prepared.params,
                state,
                cancellation,
                &mut callback,
            ),
            (Some(state), None) => engine.generate_stream_resuming(
                &prepared.prompt,
                &prepared.params,
                state,
                &mut callback,
            ),
            (None, Some(cancellation)) => engine.generate_stream_cancellable(
                &prepared.prompt,
                &prepared.params,
                cancellation,
                &mut callback,
            ),
            (None, None) => {
                engine.generate_stream(&prepared.prompt, &prepared.params, &mut callback)
            }
        }
    }
    .map_err(|error| match error {
        rnb_llm::error::LlmError::Cancelled => ApiError::cancelled(),
        error => ApiError::internal(format!("generation failed: {error}")),
    })?;
    if !callback_stopped {
        filter.finish(|text| {
            let should_continue = append_generated_text(&mut content, text, stop_on_tool_call);
            if !tool_mode && !text.is_empty() {
                let callback_continue = turn_filter.push(text, |visible| {
                    if on_text
                        .as_deref_mut()
                        .is_some_and(|callback| !callback(visible))
                    {
                        callback_stopped = true;
                        false
                    } else {
                        true
                    }
                });
                return should_continue && callback_continue;
            }
            should_continue
        });
        if !callback_stopped && !tool_mode {
            turn_filter.finish(|visible| {
                if on_text
                    .as_deref_mut()
                    .is_some_and(|callback| !callback(visible))
                {
                    callback_stopped = true;
                    false
                } else {
                    true
                }
            });
        }
    }
    let output = parse_generated_output(
        &content,
        &prepared.tool_names,
        prepared.parallel_tool_calls,
        prepared.muse_protocol,
    )?;

    Ok(GeneratedCompletion {
        output,
        prompt_tokens: result.prompt_tokens,
        output_tokens: result.tokens_generated,
        max_tokens: prepared.params.max_tokens,
        matched_stop: filter.matched(),
        callback_stopped,
        generated_token_ids: result.generated_token_ids,
        prompt_token_ids: result.prompt_token_ids,
        cached_prompt_tokens: result.cached_prompt_tokens,
    })
}

pub(super) fn append_generated_text(
    content: &mut String,
    text: &str,
    stop_on_tool_call: bool,
) -> bool {
    content.push_str(text);
    if !stop_on_tool_call {
        return true;
    }
    let end = [
        ("<tool_call|>", content.find("<tool_call|>")),
        ("</tool_call>", content.find("</tool_call>")),
    ]
    .into_iter()
    .filter_map(|(marker, index)| index.map(|index| index + marker.len()))
    .min();
    if let Some(end) = end {
        content.truncate(end);
        false
    } else {
        true
    }
}

pub(super) fn parse_generated_output(
    content: &str,
    tool_names: &[String],
    parallel_tool_calls: bool,
    muse_protocol: bool,
) -> Result<ParsedAssistantOutput, ApiError> {
    let format = if muse_protocol {
        ToolCallFormat::Muse
    } else {
        ToolCallFormat::Json
    };
    let parsed = parse_assistant_output_with_format(content, tool_names, format)
        .map_err(|error| ApiError::internal(format!("invalid assistant output: {error}")))?;
    if !parallel_tool_calls && parsed.tool_calls.len() > 1 {
        return Err(ApiError::internal(
            "model generated parallel tool calls when parallel_tool_calls is false",
        ));
    }
    Ok(parsed)
}
