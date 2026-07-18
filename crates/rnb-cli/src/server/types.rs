use super::http::ApiError;
use super::structured::{prepare_generation_constraint, prepare_tools};
use rnb_llm::{ChatMessage, ChatTemplateOptions, Engine, GenerateParams};
use serde::Deserialize;
use serde_json::Value;

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
}

pub(super) struct GenerationRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
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

        let messages = self
            .messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| message.into_chat_message(index))
            .collect::<Result<Vec<_>, _>>()?;
        let stop_sequences = match self.stop {
            None => Vec::new(),
            Some(StopSequences::One(value)) => vec![value],
            Some(StopSequences::Many(values)) => values,
        };

        GenerationRequest {
            model: self.model,
            messages,
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

        let prompt_tokens =
            engine.tokenizer.encode(&prompt).len() + usize::from(engine.tokenizer.should_add_bos());
        let available_tokens = engine.metadata.max_seq_len.saturating_sub(prompt_tokens);
        if prompt_tokens >= engine.metadata.max_seq_len || params.max_tokens > available_tokens {
            return Err(ApiError::invalid(
                format!(
                    "This model's maximum context length is {} tokens, but the request uses {} prompt tokens and allows {} completion tokens",
                    engine.metadata.max_seq_len, prompt_tokens, params.max_tokens
                ),
                Some(self.input_param),
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
        })
    }
}

impl ApiMessage {
    fn into_chat_message(self, index: usize) -> Result<ChatMessage, ApiError> {
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
        let content = match self.content {
            Some(content) => Some(content.into_text(index)?),
            None if self.role == "assistant" && tool_calls.is_some() => None,
            None => {
                return Err(ApiError::invalid(
                    format!("messages[{index}].content must contain text"),
                    Some("messages"),
                    Some("invalid_value"),
                ));
            }
        };

        Ok(ChatMessage {
            role: self.role,
            content,
            tool_calls,
            tool_call_id,
            name: self.name,
        })
    }
}

impl MessageContent {
    fn into_text(self, message_index: usize) -> Result<String, ApiError> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Parts(parts) => {
                let mut text = String::new();
                for (part_index, part) in parts.into_iter().enumerate() {
                    if part.kind != "text" {
                        return Err(ApiError::invalid(
                            format!(
                                "messages[{message_index}].content[{part_index}] type '{}' is not supported",
                                part.kind
                            ),
                            Some("messages"),
                            Some("unsupported_value"),
                        ));
                    }
                    text.push_str(part.text.as_deref().ok_or_else(|| {
                        ApiError::invalid(
                            format!(
                                "messages[{message_index}].content[{part_index}].text is required"
                            ),
                            Some("messages"),
                            Some("invalid_value"),
                        )
                    })?);
                }
                Ok(text)
            }
        }
    }
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
    fn text_parts_are_joined_and_non_text_parts_are_rejected() {
        let content: MessageContent = serde_json::from_str(
            r#"[{"type":"text","text":"hello "},{"type":"text","text":"world"}]"#,
        )
        .unwrap();
        assert_eq!(content.into_text(0).unwrap(), "hello world");

        let image: MessageContent =
            serde_json::from_str(r#"[{"type":"image_url","image_url":{"url":"x"}}]"#).unwrap();
        assert_eq!(
            image.into_text(0).unwrap_err().code,
            Some("unsupported_value")
        );
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
        let assistant = assistant.into_chat_message(0).unwrap();
        assert!(assistant.content.is_none());
        assert_eq!(
            assistant.tool_calls.unwrap()[0]["function"]["name"],
            "get_weather"
        );

        let response: ApiMessage =
            serde_json::from_str(r#"{"role":"tool","tool_call_id":"call_1","content":"sunny"}"#)
                .unwrap();
        let response = response.into_chat_message(1).unwrap();
        assert_eq!(response.role, "tool");
        assert_eq!(response.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(response.content.as_deref(), Some("sunny"));
    }
}
