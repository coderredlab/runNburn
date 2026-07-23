use super::http::ApiError;
use super::response_config::{
    normalize_metadata, normalize_stream_options, normalize_text, normalize_tool_choice,
    normalize_tools,
};
use super::response_input::{normalize_input, ResponseInput};
use super::session_store::ResolvedResponseContext;
use super::types::{GenerationRequest, PreparedGenerationRequest};
use rnb_llm::Engine;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Deserialize)]
pub(super) struct ResponseRequest {
    model: String,
    pub(super) input: Option<ResponseInput>,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    text: Option<Value>,
    stream: Option<bool>,
    stream_options: Option<Value>,
    metadata: Option<Value>,
    store: Option<bool>,
    pub(super) previous_response_id: Option<Value>,
    background: Option<bool>,
    include: Option<Value>,
    pub(super) conversation: Option<Value>,
    prompt: Option<Value>,
    reasoning: Option<Value>,
    truncation: Option<String>,
    top_logprobs: Option<usize>,
    max_tool_calls: Option<usize>,
    service_tier: Option<String>,
    user: Option<String>,
    safety_identifier: Option<String>,
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<Value>,
    prompt_cache_options: Option<Value>,
    context_management: Option<Value>,
    moderation: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

pub(super) struct PreparedResponseRequest {
    pub generation: PreparedGenerationRequest,
    pub instructions: Value,
    pub metadata: Value,
    pub tools: Value,
    pub tool_choice: Value,
    pub text: Value,
    pub service_tier: String,
    pub store: bool,
    pub user: Option<String>,
    pub safety_identifier: Option<String>,
    pub history_items: Vec<Value>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub resume_state: Option<std::sync::Arc<rnb_llm::EngineSequenceState>>,
}

impl ResponseRequest {
    pub fn prepare(
        self,
        served_model: &str,
        engine: &Engine,
        context: ResolvedResponseContext,
    ) -> Result<PreparedResponseRequest, ApiError> {
        self.validate_supported_features()?;
        let messages = normalize_input(
            ResponseInput::Items(context.history_items.clone()),
            self.instructions.as_deref(),
        )?;
        let (chat_tools, response_tools) = normalize_tools(self.tools)?;
        let (chat_tool_choice, response_tool_choice) = normalize_tool_choice(self.tool_choice)?;
        let (response_format, response_text) = normalize_text(self.text)?;
        let metadata = normalize_metadata(self.metadata)?;
        let include_usage = normalize_stream_options(self.stream_options)?;
        let service_tier = match self.service_tier.as_deref() {
            None | Some("auto") | Some("default") => "default".to_string(),
            Some(value) => {
                return Err(unsupported(
                    "service_tier",
                    format!("service tier '{value}' is not supported"),
                ));
            }
        };

        let generation = GenerationRequest {
            model: self.model,
            messages,
            max_tokens: self.max_output_tokens,
            max_tokens_param: "max_output_tokens",
            input_param: "input",
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            stop_sequences: Vec::new(),
            stream: self.stream.unwrap_or(false),
            include_usage,
            tools: chat_tools,
            tool_choice: chat_tool_choice,
            response_format,
            parallel_tool_calls: self.parallel_tool_calls,
            capture_response_history: true,
        }
        .prepare(served_model, engine)
        .map_err(remap_response_error)?;

        Ok(PreparedResponseRequest {
            generation,
            instructions: self.instructions.map_or(Value::Null, Value::String),
            metadata,
            tools: response_tools,
            tool_choice: response_tool_choice,
            text: response_text,
            service_tier,
            store: self.store.unwrap_or(true),
            user: self.user,
            safety_identifier: self.safety_identifier,
            history_items: context.history_items,
            previous_response_id: context.previous_response_id,
            conversation_id: context.conversation_id,
            resume_state: context.resume_state,
        })
    }

    fn validate_supported_features(&self) -> Result<(), ApiError> {
        if let Some((name, _)) = self.extra.iter().next() {
            return Err(unsupported(
                "request",
                format!("unsupported request field '{name}'"),
            ));
        }
        if self.background.unwrap_or(false) {
            return Err(unsupported(
                "background",
                "background responses are not supported",
            ));
        }
        reject_non_null(
            self.prompt.as_ref(),
            "prompt",
            "stored prompt templates are not supported",
        )?;
        reject_non_null(
            self.reasoning.as_ref(),
            "reasoning",
            "reasoning configuration is not supported",
        )?;
        reject_non_null(
            self.context_management.as_ref(),
            "context_management",
            "context management is not supported",
        )?;
        reject_non_null(
            self.moderation.as_ref(),
            "moderation",
            "response moderation is not supported",
        )?;
        if self.include.as_ref().is_some_and(|value| {
            !value.is_null() && value.as_array().is_none_or(|items| !items.is_empty())
        }) {
            return Err(unsupported(
                "include",
                "additional response data is not supported",
            ));
        }
        if self.max_tool_calls.is_some() {
            return Err(unsupported(
                "max_tool_calls",
                "built-in tool call limits are not supported",
            ));
        }
        if self.top_logprobs.unwrap_or(0) != 0 {
            return Err(unsupported(
                "top_logprobs",
                "log probabilities are not supported",
            ));
        }
        if self
            .truncation
            .as_deref()
            .is_some_and(|value| value != "disabled")
        {
            return Err(unsupported(
                "truncation",
                "only truncation='disabled' is supported",
            ));
        }
        validate_optional_string(&self.user, "user", 64)?;
        validate_optional_string(&self.safety_identifier, "safety_identifier", 64)?;
        validate_optional_string(&self.prompt_cache_key, "prompt_cache_key", 64)?;
        validate_optional_null(
            self.prompt_cache_retention.as_ref(),
            "prompt_cache_retention",
        )?;
        validate_optional_null(self.prompt_cache_options.as_ref(), "prompt_cache_options")?;
        Ok(())
    }
}

pub(super) fn validate_name(name: &str, param: &'static str) -> Result<(), ApiError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid(
            param,
            "function names must contain 1-64 alphanumeric, underscore, or hyphen characters",
        ));
    }
    Ok(())
}

fn validate_optional_string(
    value: &Option<String>,
    param: &'static str,
    max_len: usize,
) -> Result<(), ApiError> {
    if value.as_ref().is_some_and(|value| value.len() > max_len) {
        return Err(invalid(
            param,
            format!("{param} must be at most {max_len} characters"),
        ));
    }
    Ok(())
}

fn validate_optional_null(value: Option<&Value>, param: &'static str) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_null()) {
        return Err(unsupported(param, format!("{param} is not supported")));
    }
    Ok(())
}

fn reject_non_null(
    value: Option<&Value>,
    param: &'static str,
    message: &'static str,
) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_null()) {
        return Err(unsupported(param, message));
    }
    Ok(())
}

fn remap_response_error(mut error: ApiError) -> ApiError {
    if error.param == Some("response_format") {
        error.param = Some("text.format");
        error.message = error.message.replace("response_format", "text.format");
    }
    error
}

pub(super) fn invalid(param: &'static str, message: impl Into<String>) -> ApiError {
    ApiError::invalid(message, Some(param), Some("invalid_value"))
}

pub(super) fn unsupported(param: &'static str, message: impl Into<String>) -> ApiError {
    ApiError::invalid(message, Some(param), Some("unsupported_value"))
}
