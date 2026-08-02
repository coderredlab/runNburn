use super::http::ApiError;
use super::structured::{prepare_generation_constraint, prepare_tools};
use base64::prelude::{Engine as _, BASE64_STANDARD};
use image::{ImageFormat, ImageReader, Limits};
use rnb_llm::{
    ChatContent, ChatContentPart, ChatMessage, ChatTemplateOptions, Engine, GenerateParams,
    Qwen36RgbImage,
};
use serde::Deserialize;
use serde_json::Value;
use std::io::Cursor;

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopSequences>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub n: Option<usize>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub modalities: Option<Value>,
    pub audio: Option<Value>,
    pub functions: Option<Value>,
    pub function_call: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ApiMessage {
    role: String,
    content: Option<MessageContent>,
    tool_calls: Option<Value>,
    function_call: Option<Value>,
    tool_call_id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    image_url: Option<ImageUrl>,
}

#[derive(Debug, Deserialize)]
struct ImageUrl {
    url: String,
    detail: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamOptions {
    pub include_usage: Option<bool>,
}

pub(super) struct PreparedGenerationRequest {
    pub prompt: String,
    pub params: GenerateParams,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub tool_names: Vec<String>,
    pub parallel_tool_calls: bool,
    pub response_history_affixes: Option<(String, String)>,
    pub image: Option<Qwen36RgbImage>,
}

pub(super) struct GenerationRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub image: Option<Qwen36RgbImage>,
    pub max_tokens: Option<usize>,
    pub max_tokens_param: &'static str,
    pub input_param: &'static str,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub capture_response_history: bool,
}

impl ChatCompletionRequest {
    fn validate_supported_features(&self) -> Result<(), ApiError> {
        if self.modalities.as_ref().is_some_and(|modalities| {
            !matches!(modalities, Value::Null)
                && modalities
                    .as_array()
                    .is_none_or(|values| values.len() != 1 || values[0].as_str() != Some("text"))
        }) {
            return Err(unsupported(
                "modalities",
                "only text response modality is supported",
            ));
        }
        if self
            .audio
            .as_ref()
            .is_some_and(|audio| !matches!(audio, Value::Null))
        {
            return Err(unsupported("audio", "audio output is not supported"));
        }
        if self.functions.as_ref().is_some_and(|functions| {
            !matches!(functions, Value::Null)
                && functions.as_array().is_none_or(|values| !values.is_empty())
        }) {
            return Err(unsupported(
                "functions",
                "legacy function calling is not supported",
            ));
        }
        if self.function_call.as_ref().is_some_and(|choice| {
            !matches!(choice, Value::Null) && choice.as_str().is_none_or(|value| value != "none")
        }) {
            return Err(unsupported(
                "function_call",
                "legacy function calling is not supported",
            ));
        }
        Ok(())
    }

    pub fn prepare(
        self,
        served_model: &str,
        engine: &Engine,
    ) -> Result<PreparedGenerationRequest, ApiError> {
        self.validate_supported_features()?;
        if self.messages.is_empty() {
            return Err(ApiError::invalid(
                "messages must contain at least one message",
                Some("messages"),
                Some("invalid_value"),
            ));
        }
        if self.n.unwrap_or(1) != 1 {
            return Err(unsupported("n", "only n=1 is supported"));
        }
        if self.logprobs.unwrap_or(false) || self.top_logprobs.is_some() {
            return Err(unsupported(
                "logprobs",
                "log probabilities are not supported",
            ));
        }

        let mut messages = Vec::with_capacity(self.messages.len());
        let mut image = None;
        for (index, message) in self.messages.into_iter().enumerate() {
            let (message, message_image) = message.into_chat_message(index)?;
            if let Some(message_image) = message_image {
                if image.replace(message_image).is_some() {
                    return Err(ApiError::invalid(
                        "only one image is supported per chat completion",
                        Some("messages"),
                        Some("unsupported_value"),
                    ));
                }
            }
            messages.push(message);
        }
        let stop_sequences = match self.stop {
            None => Vec::new(),
            Some(StopSequences::One(value)) => vec![value],
            Some(StopSequences::Many(values)) => values,
        };

        GenerationRequest {
            model: self.model,
            messages,
            image,
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            max_tokens_param: "max_completion_tokens",
            input_param: "messages",
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            stop_sequences,
            stream: self.stream.unwrap_or(false),
            include_usage: self
                .stream_options
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
            parallel_tool_calls: self.parallel_tool_calls,
            capture_response_history: false,
        }
        .prepare(served_model, engine)
    }
}

impl GenerationRequest {
    pub fn prepare(
        self,
        served_model: &str,
        engine: &Engine,
    ) -> Result<PreparedGenerationRequest, ApiError> {
        if self.model != served_model {
            return Err(ApiError::model_not_found(&self.model));
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid(
                format!("{} must contain at least one message", self.input_param),
                Some(self.input_param),
                Some("invalid_value"),
            ));
        }

        let tools = prepare_tools(self.tools.as_ref(), self.tool_choice.as_ref())?;
        let parallel_tool_calls = self.parallel_tool_calls.unwrap_or(true);
        let constraint = prepare_generation_constraint(
            self.response_format.as_ref(),
            &tools,
            parallel_tool_calls,
            engine.tool_call_format(),
        )?;
        if let Some(constraint) = constraint.as_ref() {
            constraint.validate(&engine.tokenizer).map_err(|error| {
                let param = if self.response_format.is_some() {
                    "response_format"
                } else {
                    "tools"
                };
                ApiError::invalid(error.to_string(), Some(param), Some("invalid_value"))
            })?;
        }

        let prompt = engine
            .tokenizer
            .render_chat_prompt_with_tools(
                &self.messages,
                ChatTemplateOptions::default(),
                &tools.prompt_definitions,
            )
            .map_err(|error| match error {
                rnb_llm::error::LlmError::InvalidChatRequest(message) => {
                    ApiError::invalid(message, Some(self.input_param), Some("invalid_value"))
                }
                error => ApiError::internal(error.to_string()),
            })?;
        let response_history_affixes = if self.capture_response_history {
            let mut assistant_sentinel = "__RNB_RESPONSE_ASSISTANT_CONTENT_7F43A9C2__".to_string();
            let mut user_sentinel = "__RNB_RESPONSE_NEXT_USER_CONTENT_51D8E604__".to_string();
            while prompt.contains(&assistant_sentinel) || prompt.contains(&user_sentinel) {
                assistant_sentinel.push('_');
                user_sentinel.push('_');
            }
            let mut history_messages = self.messages.clone();
            history_messages.push(ChatMessage::new("assistant", assistant_sentinel.clone()));
            history_messages.push(ChatMessage::new("user", user_sentinel.clone()));
            let rendered = engine
                .tokenizer
                .render_chat_prompt_with_tools(
                    &history_messages,
                    ChatTemplateOptions {
                        add_generation_prompt: false,
                        enable_thinking: false,
                    },
                    &tools.prompt_definitions,
                )
                .map_err(|error| match error {
                    rnb_llm::error::LlmError::InvalidChatRequest(message) => {
                        ApiError::invalid(message, Some(self.input_param), Some("invalid_value"))
                    }
                    error => ApiError::internal(error.to_string()),
                })?;
            let (prefix, tail) = rendered.split_once(&assistant_sentinel).ok_or_else(|| {
                ApiError::internal("chat template omitted the assistant response content")
            })?;
            let (bridge, _) = tail
                .split_once(&user_sentinel)
                .ok_or_else(|| ApiError::internal("chat template omitted the next user content"))?;
            Some((prefix.to_string(), bridge.to_string()))
        } else {
            None
        };

        let mut params = GenerateParams::default();
        params.max_tokens = self.max_tokens.unwrap_or(params.max_tokens);
        if params.max_tokens == 0 {
            return Err(ApiError::invalid(
                format!("{} must be greater than zero", self.max_tokens_param),
                Some(self.max_tokens_param),
                Some("invalid_value"),
            ));
        }
        if let Some(value) = self.temperature {
            validate_f32(value, 0.0, 2.0, "temperature")?;
            params.temperature = value;
        }
        if let Some(value) = self.top_p {
            validate_f32(value, 0.0, 1.0, "top_p")?;
            params.top_p = value;
        }
        if let Some(value) = self.min_p {
            validate_f32(value, 0.0, 1.0, "min_p")?;
            params.min_p = value;
        }
        if let Some(value) = self.repetition_penalty {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid_number(
                    "repetition_penalty",
                    "must be finite and greater than zero",
                ));
            }
            params.repetition_penalty = value;
        }
        if let Some(value) = self.presence_penalty {
            validate_f32(value, -2.0, 2.0, "presence_penalty")?;
            params.presence_penalty = value;
        }
        if let Some(value) = self.frequency_penalty {
            validate_f32(value, -2.0, 2.0, "frequency_penalty")?;
            params.frequency_penalty = value;
        }
        if let Some(value) = self.top_k {
            params.top_k = value;
        }
        params.seed = self.seed;
        params.constraint = constraint;

        if self.image.is_none() {
            let prompt_tokens = engine.tokenizer.encode(&prompt).len()
                + usize::from(engine.tokenizer.should_add_bos());
            let available_tokens = engine.metadata.max_seq_len.saturating_sub(prompt_tokens);
            if prompt_tokens >= engine.metadata.max_seq_len || params.max_tokens > available_tokens
            {
                return Err(ApiError::invalid(
                    format!(
                        "This model's maximum context length is {} tokens, but the request uses {} prompt tokens and allows {} completion tokens",
                        engine.metadata.max_seq_len, prompt_tokens, params.max_tokens
                    ),
                    Some(self.input_param),
                    Some("context_length_exceeded"),
                ));
            }
        } else if !engine.has_vision_projector() {
            return Err(ApiError::invalid(
                "image input requires the server to start with --mmproj",
                Some(self.input_param),
                Some("invalid_value"),
            ));
        } else if params.max_tokens >= engine.metadata.max_seq_len {
            return Err(ApiError::invalid(
                format!(
                    "This model's maximum context length is {} tokens, but the request allows {} completion tokens",
                    engine.metadata.max_seq_len, params.max_tokens
                ),
                Some(self.max_tokens_param),
                Some("context_length_exceeded"),
            ));
        }

        validate_stop_sequences(&self.stop_sequences)?;

        Ok(PreparedGenerationRequest {
            prompt,
            params,
            stop_sequences: self.stop_sequences,
            stream: self.stream,
            include_usage: self.include_usage,
            tool_names: tools.names,
            parallel_tool_calls,
            response_history_affixes,
            image: self.image,
        })
    }
}

impl ApiMessage {
    fn into_chat_message(
        self,
        index: usize,
    ) -> Result<(ChatMessage, Option<Qwen36RgbImage>), ApiError> {
        if !matches!(
            self.role.as_str(),
            "system" | "developer" | "user" | "assistant" | "tool"
        ) {
            return Err(ApiError::invalid(
                format!("unsupported messages[{index}].role '{}'", self.role),
                Some("messages"),
                Some("unsupported_value"),
            ));
        }
        if self
            .function_call
            .as_ref()
            .is_some_and(|call| !matches!(call, Value::Null))
        {
            return Err(unsupported(
                "messages",
                "legacy function calls are not supported",
            ));
        }

        let tool_calls = match self.tool_calls {
            None | Some(Value::Null) => None,
            Some(value) => {
                if self.role != "assistant" {
                    return Err(ApiError::invalid(
                        format!(
                            "messages[{index}].tool_calls is only valid for assistant messages"
                        ),
                        Some("messages"),
                        Some("invalid_value"),
                    ));
                }
                validate_message_tool_calls(&value, index)?;
                Some(value)
            }
        };
        let tool_call_id = match self.tool_call_id {
            Some(id) if self.role == "tool" && !id.is_empty() => Some(id),
            Some(_) => {
                return Err(ApiError::invalid(
                    format!(
                        "messages[{index}].tool_call_id is only valid for tool messages and must not be empty"
                    ),
                    Some("messages"),
                    Some("invalid_value"),
                ));
            }
            None if self.role == "tool" => {
                return Err(ApiError::invalid(
                    format!("messages[{index}].tool_call_id is required for tool messages"),
                    Some("messages"),
                    Some("invalid_value"),
                ));
            }
            None => None,
        };
        let (content, image) = match self.content {
            Some(content) => {
                let (content, image) = content.into_chat_content(index, &self.role)?;
                (Some(content), image)
            }
            None if self.role == "assistant" && tool_calls.is_some() => (None, None),
            None => {
                return Err(ApiError::invalid(
                    format!("messages[{index}].content must contain text"),
                    Some("messages"),
                    Some("invalid_value"),
                ));
            }
        };

        Ok((
            ChatMessage {
                role: self.role,
                content,
                tool_calls,
                tool_call_id,
                name: self.name,
            },
            image,
        ))
    }
}

impl MessageContent {
    fn into_chat_content(
        self,
        message_index: usize,
        role: &str,
    ) -> Result<(ChatContent, Option<Qwen36RgbImage>), ApiError> {
        match self {
            Self::Text(text) => Ok((ChatContent::Text(text), None)),
            Self::Parts(parts) => {
                let mut content = Vec::with_capacity(parts.len());
                let mut image = None;
                for (part_index, part) in parts.into_iter().enumerate() {
                    match part.kind.as_str() {
                        "text" => {
                            let text = part.text.ok_or_else(|| {
                                invalid_content_part(
                                    message_index,
                                    part_index,
                                    "text is required",
                                    "invalid_value",
                                )
                            })?;
                            content.push(ChatContentPart::Text { text });
                        }
                        "image_url" => {
                            if role != "user" {
                                return Err(invalid_content_part(
                                    message_index,
                                    part_index,
                                    "image_url is only valid for user messages",
                                    "invalid_value",
                                ));
                            }
                            if part
                                .image_url
                                .as_ref()
                                .and_then(|image_url| image_url.detail.as_deref())
                                .is_some_and(|detail| !matches!(detail, "auto" | "low" | "high"))
                            {
                                return Err(invalid_content_part(
                                    message_index,
                                    part_index,
                                    "image_url.detail must be auto, low, or high",
                                    "invalid_value",
                                ));
                            }
                            let image_url = part.image_url.ok_or_else(|| {
                                invalid_content_part(
                                    message_index,
                                    part_index,
                                    "image_url is required",
                                    "invalid_value",
                                )
                            })?;
                            if image.replace(decode_data_image(&image_url.url)?).is_some() {
                                return Err(invalid_content_part(
                                    message_index,
                                    part_index,
                                    "only one image is supported per message",
                                    "unsupported_value",
                                ));
                            }
                            content.push(ChatContentPart::Image);
                        }
                        kind => {
                            return Err(invalid_content_part(
                                message_index,
                                part_index,
                                &format!("type '{kind}' is not supported"),
                                "unsupported_value",
                            ));
                        }
                    }
                }
                Ok((ChatContent::Parts(content), image))
            }
        }
    }
}

fn invalid_content_part(
    message_index: usize,
    part_index: usize,
    detail: &str,
    code: &'static str,
) -> ApiError {
    ApiError::invalid(
        format!("messages[{message_index}].content[{part_index}] {detail}"),
        Some("messages"),
        Some(code),
    )
}

fn decode_data_image(url: &str) -> Result<Qwen36RgbImage, ApiError> {
    const MAX_ENCODED_BYTES: usize = 28 * 1024 * 1024;
    const MAX_DECODED_ALLOC: u64 = 256 * 1024 * 1024;
    const MAX_DIMENSION: u32 = 8192;

    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(unsupported(
            "messages",
            "remote image URLs are not supported; use a PNG or JPEG data URL",
        ));
    }
    let (header, payload) = url.split_once(',').ok_or_else(|| {
        ApiError::invalid(
            "image_url.url must be a PNG or JPEG data URL",
            Some("messages"),
            Some("invalid_image"),
        )
    })?;
    let claimed_format = match header {
        "data:image/png;base64" => ImageFormat::Png,
        "data:image/jpeg;base64" => ImageFormat::Jpeg,
        _ => {
            return Err(ApiError::invalid(
                "image_url.url must use data:image/png;base64 or data:image/jpeg;base64",
                Some("messages"),
                Some("invalid_image"),
            ));
        }
    };
    if payload.len() > MAX_ENCODED_BYTES {
        return Err(ApiError::invalid(
            "image data URL exceeds the 28 MiB encoded limit",
            Some("messages"),
            Some("invalid_image"),
        ));
    }
    let bytes = BASE64_STANDARD.decode(payload).map_err(|error| {
        ApiError::invalid(
            format!("image data URL contains invalid base64: {error}"),
            Some("messages"),
            Some("invalid_image"),
        )
    })?;
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| {
            ApiError::invalid(
                format!("could not identify image data: {error}"),
                Some("messages"),
                Some("invalid_image"),
            )
        })?;
    if reader.format() != Some(claimed_format) {
        return Err(ApiError::invalid(
            "image MIME type does not match the encoded image",
            Some("messages"),
            Some("invalid_image"),
        ));
    }
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    limits.max_alloc = Some(MAX_DECODED_ALLOC);
    reader.limits(limits);
    let decoded = reader.decode().map_err(|error| {
        ApiError::invalid(
            format!("could not decode image: {error}"),
            Some("messages"),
            Some("invalid_image"),
        )
    })?;
    let rgb = decoded.into_rgb8();
    Qwen36RgbImage::new(rgb.width() as usize, rgb.height() as usize, rgb.into_raw()).map_err(
        |error| {
            ApiError::invalid(
                format!("invalid image: {error}"),
                Some("messages"),
                Some("invalid_image"),
            )
        },
    )
}

fn validate_message_tool_calls(value: &Value, message_index: usize) -> Result<(), ApiError> {
    let calls = value
        .as_array()
        .filter(|calls| !calls.is_empty())
        .ok_or_else(|| {
            ApiError::invalid(
                format!("messages[{message_index}].tool_calls must be a non-empty array"),
                Some("messages"),
                Some("invalid_value"),
            )
        })?;
    for (call_index, call) in calls.iter().enumerate() {
        let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
        if id.is_empty() || call.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::invalid(
                format!(
                    "messages[{message_index}].tool_calls[{call_index}] requires a non-empty id and type 'function'"
                ),
                Some("messages"),
                Some("invalid_value"),
            ));
        }
        let function = call
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ApiError::invalid(
                    format!(
                        "messages[{message_index}].tool_calls[{call_index}].function must be an object"
                    ),
                    Some("messages"),
                    Some("invalid_value"),
                )
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::invalid(
                    format!(
                    "messages[{message_index}].tool_calls[{call_index}].function.name is required"
                ),
                    Some("messages"),
                    Some("invalid_value"),
                )
            })?;
        validate_tool_name(name, "messages")?;
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::invalid(
                    format!(
                        "messages[{message_index}].tool_calls[{call_index}].function.arguments must be a JSON string"
                    ),
                    Some("messages"),
                    Some("invalid_value"),
                )
            })?;
        if serde_json::from_str::<Value>(arguments).is_err() {
            return Err(ApiError::invalid(
                format!(
                    "messages[{message_index}].tool_calls[{call_index}].function.arguments must contain valid JSON"
                ),
                Some("messages"),
                Some("invalid_value"),
            ));
        }
    }
    Ok(())
}

fn validate_tool_name(name: &str, param: &'static str) -> Result<(), ApiError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(ApiError::invalid(
            format!("tool name '{name}' is invalid"),
            Some(param),
            Some("invalid_value"),
        ));
    }
    Ok(())
}

fn validate_f32(value: f32, min: f32, max: f32, param: &'static str) -> Result<(), ApiError> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(invalid_number(
            param,
            &format!("must be between {min} and {max}"),
        ));
    }
    Ok(())
}

fn invalid_number(param: &'static str, requirement: &str) -> ApiError {
    ApiError::invalid(
        format!("{param} {requirement}"),
        Some(param),
        Some("invalid_value"),
    )
}

fn unsupported(param: &'static str, message: &str) -> ApiError {
    ApiError::invalid(message, Some(param), Some("unsupported_value"))
}

fn validate_stop_sequences(stops: &[String]) -> Result<(), ApiError> {
    if stops.len() > 4 {
        return Err(ApiError::invalid(
            "stop supports at most 4 sequences",
            Some("stop"),
            Some("invalid_value"),
        ));
    }
    if stops.iter().any(|stop| stop.is_empty()) {
        return Err(ApiError::invalid(
            "stop sequences must not be empty",
            Some("stop"),
            Some("invalid_value"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_text_and_data_url_image_parts_but_rejects_remote_urls() {
        let content: MessageContent = serde_json::from_str(
            r#"[{"type":"text","text":"hello "},{"type":"text","text":"world"}]"#,
        )
        .unwrap();
        let (content, image) = content.into_chat_content(0, "user").unwrap();
        assert_eq!(
            content,
            ChatContent::Parts(vec![
                ChatContentPart::Text {
                    text: "hello ".to_string(),
                },
                ChatContentPart::Text {
                    text: "world".to_string(),
                },
            ])
        );
        assert!(image.is_none());

        let content: MessageContent = serde_json::from_str(
            r#"[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}},{"type":"text","text":"describe"}]"#,
        )
        .unwrap();
        let (content, image) = content.into_chat_content(0, "user").unwrap();
        assert_eq!(
            content,
            ChatContent::Parts(vec![
                ChatContentPart::Image,
                ChatContentPart::Text {
                    text: "describe".to_string(),
                },
            ])
        );
        let image = image.unwrap();
        assert_eq!((image.width(), image.height()), (1, 1));

        let remote: MessageContent = serde_json::from_str(
            r#"[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]"#,
        )
        .unwrap();
        assert_eq!(
            remote.into_chat_content(0, "user").unwrap_err().code,
            Some("unsupported_value")
        );
    }

    #[test]
    fn rejects_malformed_image_data_urls() {
        for url in [
            "data:image/gif;base64,AAAA",
            "data:image/png;base64,not-base64",
        ] {
            let content: MessageContent = serde_json::from_value(serde_json::json!([{
                "type": "image_url",
                "image_url": {"url": url}
            }]))
            .unwrap();
            assert_eq!(
                content.into_chat_content(0, "user").unwrap_err().code,
                Some("invalid_image")
            );
        }
    }

    #[test]
    fn rejects_unsupported_response_modalities_before_generation() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":["audio"]}"#,
        )
        .unwrap();

        let error = request.validate_supported_features().unwrap_err();
        assert_eq!(error.param, Some("modalities"));
        assert_eq!(error.code, Some("unsupported_value"));
    }

    #[test]
    fn validates_function_tools_and_honors_none_choice() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);

        let enabled = prepare_tools(Some(&tools), Some(&Value::String("auto".into()))).unwrap();
        assert_eq!(enabled.prompt_definitions.len(), 1);
        assert_eq!(enabled.names, vec!["get_weather"]);

        let disabled = prepare_tools(Some(&tools), Some(&Value::String("none".into()))).unwrap();
        assert!(disabled.prompt_definitions.is_empty());
        assert!(disabled.names.is_empty());
    }

    #[test]
    fn preserves_assistant_tool_calls_and_tool_responses() {
        let assistant: ApiMessage = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Seoul\"}"}}]}"#,
        )
        .unwrap();
        let (assistant, image) = assistant.into_chat_message(0).unwrap();
        assert!(image.is_none());
        assert!(assistant.content.is_none());
        assert_eq!(
            assistant.tool_calls.unwrap()[0]["function"]["name"],
            "get_weather"
        );

        let response: ApiMessage =
            serde_json::from_str(r#"{"role":"tool","tool_call_id":"call_1","content":"sunny"}"#)
                .unwrap();
        let (response, image) = response.into_chat_message(1).unwrap();
        assert!(image.is_none());
        assert_eq!(response.role, "tool");
        assert_eq!(response.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(
            response.content,
            Some(rnb_llm::ChatContent::Text("sunny".to_string()))
        );
    }
}
