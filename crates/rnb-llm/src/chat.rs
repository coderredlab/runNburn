use crate::error::{LlmError, Result};
use crate::tokenizer::Tokenizer;
use minijinja::{context, Environment, Error, ErrorKind};
use serde::Serialize;
use serde_json::Value;

const REQUEST_REJECTED_PREFIX: &str = "__RNB_CHAT_REQUEST_REJECTED__:";
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
        }
    }
}

impl Tokenizer {
    pub fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        self.render_chat_prompt_with_tools(messages, options, &[])
    }

    pub fn render_chat_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
        tools: &[Value],
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(LlmError::Tokenizer(
                "chat completion requires at least one message".to_string(),
            ));
        }
        let source = self.chat_template().ok_or_else(|| {
            LlmError::Tokenizer("GGUF does not contain tokenizer.chat_template".to_string())
        })?;

        let mut environment = Environment::new();
        minijinja_contrib::add_to_environment(&mut environment);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("{REQUEST_REJECTED_PREFIX}{message}"),
                ))
            },
        );
        environment
            .add_template("chat", source)
            .map_err(|error| LlmError::Tokenizer(format!("invalid GGUF chat template: {error}")))?;

        let bos_token = if self.should_add_bos() {
            ""
        } else {
            self.vocab
                .token_str(self.vocab.special.bos)
                .unwrap_or_default()
        };
        let eos_token = self
            .vocab
            .token_str(self.vocab.special.eos)
            .unwrap_or_default();
        let rendered = environment
            .get_template("chat")
            .expect("template was added above")
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => options.add_generation_prompt,
                enable_thinking => options.enable_thinking,
                bos_token => bos_token,
                eos_token => eos_token,
            });
        match rendered {
            Ok(prompt) => Ok(prompt),
            Err(error) => {
                let message = error.to_string();
                if let Some(detail) = message
                    .split_once(REQUEST_REJECTED_PREFIX)
                    .map(|(_, detail)| detail)
                {
                    let detail = detail.split(" (in chat:").next().unwrap_or(detail);
                    Err(LlmError::InvalidChatRequest(detail.to_string()))
                } else {
                    Err(LlmError::Tokenizer(format!(
                        "render GGUF chat template: {error}"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::vocab::{SpecialTokens, Vocab};

    fn tokenizer(add_bos_token: bool, template: Option<&str>) -> Tokenizer {
        let vocab = Vocab::new(
            vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()],
            SpecialTokens {
                bos: 1,
                eos: 2,
                pad: None,
            },
        );
        let mut tokenizer = Tokenizer::new_sentencepiece_with_config(
            vocab,
            Vec::new(),
            Vec::new(),
            add_bos_token,
            true,
        );
        tokenizer.set_chat_template(template.map(str::to_owned));
        tokenizer
    }

    #[test]
    fn renders_roles_and_generation_prompt_without_duplicate_bos() {
        let tokenizer = tokenizer(
            true,
            Some(
                "{{ bos_token }}{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
            ),
        );
        let messages = [
            ChatMessage::new("system", "Be concise."),
            ChatMessage::new("user", "Hello"),
        ];

        let rendered = tokenizer
            .render_chat_prompt(&messages, ChatTemplateOptions::default())
            .unwrap();

        assert_eq!(rendered, "<|system|>Be concise.<|user|>Hello<|assistant|>");
    }

    #[test]
    fn includes_template_bos_when_tokenizer_does_not_prepend_it() {
        let tokenizer = tokenizer(false, Some("{{ bos_token }}{{ messages[0].content }}"));

        let rendered = tokenizer
            .render_chat_prompt(
                &[ChatMessage::new("user", "Hello")],
                ChatTemplateOptions::default(),
            )
            .unwrap();

        assert_eq!(rendered, "<s>Hello");
    }

    #[test]
    fn reports_missing_chat_template() {
        let error = tokenizer(true, None)
            .render_chat_prompt(
                &[ChatMessage::new("user", "Hello")],
                ChatTemplateOptions::default(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("tokenizer.chat_template"));
    }

    #[test]
    fn maps_template_raise_exception_to_invalid_chat_request() {
        let tokenizer = tokenizer(true, Some("{{ raise_exception('roles must alternate') }}"));

        let error = tokenizer
            .render_chat_prompt(
                &[ChatMessage::new("user", "Hello")],
                ChatTemplateOptions::default(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            LlmError::InvalidChatRequest(message) if message == "roles must alternate"
        ));
    }

    #[test]
    fn renders_tools_and_tool_history_into_template_context() {
        let tokenizer = tokenizer(
            true,
            Some(
                "{% for tool in tools %}{{ tool.function.name }}{% endfor %}|{{ messages[0].tool_calls[0].function.name }}|{{ messages[1].tool_call_id }}",
            ),
        );
        let messages = [
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(serde_json::json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\":\"Seoul\"}"}
                }])),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some("sunny".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                name: None,
            },
        ];
        let tools = serde_json::json!([{
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}}
        }]);

        let rendered = tokenizer
            .render_chat_prompt_with_tools(
                &messages,
                ChatTemplateOptions::default(),
                tools.as_array().unwrap(),
            )
            .unwrap();

        assert_eq!(rendered, "weather|weather|call_1");
    }
}
