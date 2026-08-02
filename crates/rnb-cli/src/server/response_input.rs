use super::http::ApiError;
use super::image_input::decode_data_image;
use super::responses::{invalid, unsupported, validate_name};
use rnb_llm::{ChatContent, ChatContentPart, ChatMessage, Qwen36RgbImage};
use serde::Deserialize;
use serde_json::{json, Map, Value};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum ResponseInput {
    Text(String),
    Items(Vec<Value>),
}

#[derive(Debug)]
pub(super) struct NormalizedResponseInput {
    pub messages: Vec<ChatMessage>,
    pub image: Option<Qwen36RgbImage>,
}

impl ResponseInput {
    pub(super) fn into_items(self) -> Vec<Value> {
        match self {
            Self::Text(text) => vec![json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            })],
            Self::Items(items) => items,
        }
    }
}

pub(super) fn normalize_input(
    input: ResponseInput,
    instructions: Option<&str>,
) -> Result<NormalizedResponseInput, ApiError> {
    let mut messages = Vec::new();
    let mut image = None;
    if let Some(instructions) = instructions {
        messages.push(ChatMessage::new("system", instructions));
    }
    match input {
        ResponseInput::Text(text) => messages.push(ChatMessage::new("user", text)),
        ResponseInput::Items(items) => {
            if items.is_empty() {
                return Err(invalid("input", "input must not be empty"));
            }
            let mut pending_calls = Vec::new();
            for (index, item) in items.into_iter().enumerate() {
                let object = item
                    .as_object()
                    .ok_or_else(|| invalid("input", format!("input[{index}] must be an object")))?;
                let kind = object.get("type").and_then(Value::as_str);
                if kind.is_none() || kind == Some("message") {
                    flush_function_calls(&mut messages, &mut pending_calls);
                    let (message, message_image) = normalize_message(object, index)?;
                    if let Some(message_image) = message_image {
                        if image.replace(message_image).is_some() {
                            return Err(unsupported(
                                "input",
                                "only one image is supported per response input",
                            ));
                        }
                    }
                    messages.push(message);
                } else if kind == Some("function_call") {
                    pending_calls.push(normalize_function_call(object, index)?);
                } else if kind == Some("function_call_output") {
                    flush_function_calls(&mut messages, &mut pending_calls);
                    messages.push(normalize_function_output(object, index)?);
                } else {
                    return Err(unsupported(
                        "input",
                        format!(
                            "input[{index}] type '{}' is not supported",
                            kind.unwrap_or_default()
                        ),
                    ));
                }
            }
            flush_function_calls(&mut messages, &mut pending_calls);
        }
    }
    Ok(NormalizedResponseInput { messages, image })
}

fn normalize_message(
    object: &Map<String, Value>,
    index: usize,
) -> Result<(ChatMessage, Option<Qwen36RgbImage>), ApiError> {
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("input", format!("input[{index}].role is required")))?;
    if !matches!(role, "system" | "developer" | "user" | "assistant") {
        return Err(unsupported(
            "input",
            format!("input[{index}].role '{role}' is not supported"),
        ));
    }
    let content = object
        .get("content")
        .ok_or_else(|| invalid("input", format!("input[{index}].content is required")))?;
    let (content, image) = normalize_message_content(content, index, role)?;
    Ok((
        ChatMessage {
            role: role.to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        image,
    ))
}

fn normalize_message_content(
    content: &Value,
    item_index: usize,
    role: &str,
) -> Result<(ChatContent, Option<Qwen36RgbImage>), ApiError> {
    if let Some(text) = content.as_str() {
        return Ok((ChatContent::Text(text.to_string()), None));
    }
    let parts = content.as_array().ok_or_else(|| {
        invalid(
            "input",
            format!("input[{item_index}].content must be text or an array"),
        )
    })?;
    let mut normalized = Vec::with_capacity(parts.len());
    let mut image = None;
    for (part_index, part) in parts.iter().enumerate() {
        let kind = part.get("type").and_then(Value::as_str);
        match kind {
            Some("input_text" | "output_text") => {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    invalid(
                        "input",
                        format!("input[{item_index}].content[{part_index}].text is required"),
                    )
                })?;
                normalized.push(ChatContentPart::Text {
                    text: text.to_string(),
                });
            }
            Some("input_image") => {
                if role != "user" {
                    return Err(invalid(
                        "input",
                        format!(
                            "input[{item_index}].content[{part_index}] input_image is only valid for user messages"
                        ),
                    ));
                }
                let url = part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        invalid(
                            "input",
                            format!(
                                "input[{item_index}].content[{part_index}].image_url is required"
                            ),
                        )
                    })?;
                if image
                    .replace(decode_data_image(url, "input", "input_image.image_url")?)
                    .is_some()
                {
                    return Err(unsupported(
                        "input",
                        "only one image is supported per response message",
                    ));
                }
                normalized.push(ChatContentPart::Image);
            }
            _ => {
                return Err(unsupported(
                    "input",
                    format!(
                        "input[{item_index}].content[{part_index}] type '{}' is not supported",
                        kind.unwrap_or_default()
                    ),
                ));
            }
        }
    }
    Ok((ChatContent::Parts(normalized), image))
}

fn normalize_function_call(object: &Map<String, Value>, index: usize) -> Result<Value, ApiError> {
    let call_id = required_nonempty_string(object, "call_id", index)?;
    let name = required_nonempty_string(object, "name", index)?;
    validate_name(name, "input")?;
    let arguments = required_nonempty_string(object, "arguments", index)?;
    let parsed: Value = serde_json::from_str(arguments).map_err(|error| {
        invalid(
            "input",
            format!("input[{index}].arguments must contain valid JSON: {error}"),
        )
    })?;
    if !parsed.is_object() {
        return Err(invalid(
            "input",
            format!("input[{index}].arguments must encode a JSON object"),
        ));
    }
    Ok(json!({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    }))
}

fn normalize_function_output(
    object: &Map<String, Value>,
    index: usize,
) -> Result<ChatMessage, ApiError> {
    let call_id = required_nonempty_string(object, "call_id", index)?;
    let output = object
        .get("output")
        .ok_or_else(|| invalid("input", format!("input[{index}].output is required")))?;
    let content = normalize_text_content(output, index, "output")?;
    Ok(ChatMessage {
        role: "tool".to_string(),
        content: Some(rnb_llm::ChatContent::Text(content)),
        tool_calls: None,
        tool_call_id: Some(call_id.to_string()),
        name: None,
    })
}

fn flush_function_calls(messages: &mut Vec<ChatMessage>, calls: &mut Vec<Value>) {
    if calls.is_empty() {
        return;
    }
    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(Value::Array(std::mem::take(calls))),
        tool_call_id: None,
        name: None,
    });
}

fn normalize_text_content(
    content: &Value,
    item_index: usize,
    field: &str,
) -> Result<String, ApiError> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_string());
    }
    let parts = content.as_array().ok_or_else(|| {
        invalid(
            "input",
            format!("input[{item_index}].{field} must be text or an array"),
        )
    })?;
    let mut text = String::new();
    for (part_index, part) in parts.iter().enumerate() {
        let kind = part.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("input_text" | "output_text")) {
            return Err(unsupported(
                "input",
                format!(
                    "input[{item_index}].{field}[{part_index}] type '{}' is not supported",
                    kind.unwrap_or_default()
                ),
            ));
        }
        text.push_str(part.get("text").and_then(Value::as_str).ok_or_else(|| {
            invalid(
                "input",
                format!("input[{item_index}].{field}[{part_index}].text is required"),
            )
        })?);
    }
    Ok(text)
}

fn required_nonempty_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    index: usize,
) -> Result<&'a str, ApiError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            invalid(
                "input",
                format!("input[{index}].{field} must be a non-empty string"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_string_and_function_round_trip_input() {
        let input = ResponseInput::Items(vec![
            json!({"role": "user", "content": "weather"}),
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Seoul\"}"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny"
            }),
        ]);
        let normalized = normalize_input(input, Some("be concise")).unwrap();

        assert!(normalized.image.is_none());
        assert_eq!(normalized.messages.len(), 4);
        assert_eq!(normalized.messages[0].role, "system");
        assert_eq!(normalized.messages[2].role, "assistant");
        assert_eq!(normalized.messages[3].role, "tool");
        assert_eq!(
            normalized.messages[3].tool_call_id.as_deref(),
            Some("call_1")
        );
    }

    #[test]
    fn normalizes_input_image_data_url_and_rejects_remote_image() {
        let data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let input = ResponseInput::Items(vec![json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": "describe"}
            ]
        })]);
        let normalized = normalize_input(input, None).unwrap();

        let image = normalized.image.unwrap();
        assert_eq!((image.width(), image.height()), (1, 1));
        assert_eq!(
            normalized.messages[0].content,
            Some(ChatContent::Parts(vec![
                ChatContentPart::Image,
                ChatContentPart::Text {
                    text: "describe".to_string(),
                },
            ]))
        );

        let remote = ResponseInput::Items(vec![json!({
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": "https://example.com/image.png"
            }]
        })]);
        let error = normalize_input(remote, None).unwrap_err();
        assert_eq!(error.param, Some("input"));
        assert_eq!(error.code, Some("unsupported_value"));
    }

    #[test]
    fn rejects_input_image_outside_user_messages() {
        let input = ResponseInput::Items(vec![json!({
            "role": "assistant",
            "content": [{
                "type": "input_image",
                "image_url": "data:image/png;base64,AAAA"
            }]
        })]);

        let error = normalize_input(input, None).unwrap_err();
        assert_eq!(error.param, Some("input"));
        assert!(error.message.contains("only valid for user messages"));
    }
}
