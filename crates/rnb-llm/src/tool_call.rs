use serde::Serialize;
use serde_json::{Map, Value};

const GEMMA_CALL_OPEN: &str = "<|tool_call>call:";
const GEMMA_CALL_CLOSE: &str = "<tool_call|>";
const JSON_CALL_OPEN: &str = "<tool_call>";
const JSON_CALL_CLOSE: &str = "</tool_call>";
const GEMMA_QUOTE: &str = "<|\"|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Gemma,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

pub fn parse_assistant_output(
    text: &str,
    allowed_tools: &[String],
) -> Result<ParsedAssistantOutput, String> {
    let text = strip_reasoning(text);
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text.as_str();

    loop {
        let next = [
            remaining.find(GEMMA_CALL_OPEN).map(|index| (index, true)),
            remaining.find(JSON_CALL_OPEN).map(|index| (index, false)),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|(index, _)| *index);
        let Some((index, gemma)) = next else {
            content.push_str(remaining);
            break;
        };
        content.push_str(&remaining[..index]);

        let (open, close) = if gemma {
            (GEMMA_CALL_OPEN, GEMMA_CALL_CLOSE)
        } else {
            (JSON_CALL_OPEN, JSON_CALL_CLOSE)
        };
        let body_start = index + open.len();
        let Some(relative_end) = remaining[body_start..].find(close) else {
            break;
        };
        let body_end = body_start + relative_end;
        let body = remaining[body_start..body_end].trim();
        let tool_call = if gemma {
            parse_gemma_call(body)?
        } else {
            parse_json_call(body)?
        };
        if !allowed_tools.iter().any(|name| name == &tool_call.name) {
            return Err(format!(
                "model requested undeclared tool '{}'",
                tool_call.name
            ));
        }
        tool_calls.push(tool_call);
        remaining = &remaining[body_end + close.len()..];
    }

    Ok(ParsedAssistantOutput {
        content: content.trim().to_string(),
        tool_calls,
    })
}

fn parse_gemma_call(body: &str) -> Result<ParsedToolCall, String> {
    let object_start = body
        .find('{')
        .ok_or_else(|| "Gemma tool call is missing arguments".to_string())?;
    let name = body[..object_start].trim();
    validate_name(name)?;
    let mut arguments = match serde_json::from_str::<Value>(&body[object_start..]) {
        Ok(arguments) => arguments,
        Err(_) => GemmaValueParser::new(&body[object_start..]).parse_complete()?,
    };
    if !arguments.is_object() {
        return Err("Gemma tool arguments must be an object".to_string());
    }
    sort_object_keys(&mut arguments);
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&arguments)
            .map_err(|error| format!("serialize Gemma tool arguments: {error}"))?,
    })
}

fn parse_json_call(body: &str) -> Result<ParsedToolCall, String> {
    let value: Value =
        serde_json::from_str(body).map_err(|error| format!("invalid JSON tool call: {error}"))?;
    let function = value.get("function").unwrap_or(&value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "JSON tool call is missing function name".to_string())?;
    validate_name(name)?;
    let arguments = function
        .get("arguments")
        .ok_or_else(|| "JSON tool call is missing arguments".to_string())?;
    let mut arguments = match arguments {
        Value::String(raw) => serde_json::from_str(raw)
            .map_err(|error| format!("invalid JSON tool arguments: {error}"))?,
        value => value.clone(),
    };
    if !arguments.is_object() {
        return Err("tool arguments must be a JSON object".to_string());
    }
    sort_object_keys(&mut arguments);
    let arguments = serde_json::to_string(&arguments)
        .map_err(|error| format!("serialize tool arguments: {error}"))?;
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

fn sort_object_keys(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(sort_object_keys),
        Value::Object(object) => {
            object.values_mut().for_each(sort_object_keys);
            object.sort_keys();
        }
        _ => {}
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!("invalid tool name '{name}'"));
    }
    Ok(())
}

fn strip_reasoning(text: &str) -> String {
    let without_gemma = strip_blocks(text, "<|channel>thought", "<channel|>");
    strip_blocks(&without_gemma, "<think>", "</think>")
}

fn strip_blocks(text: &str, open: &str, close: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find(open) {
        output.push_str(&remaining[..start]);
        let after_open = &remaining[start + open.len()..];
        let Some(end) = after_open.find(close) else {
            return output;
        };
        remaining = &after_open[end + close.len()..];
    }
    output.push_str(remaining);
    output
}

struct GemmaValueParser<'a> {
    input: &'a str,
    offset: usize,
}

impl<'a> GemmaValueParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
    }

    fn parse_complete(mut self) -> Result<Value, String> {
        let value = self.parse_value()?;
        self.skip_whitespace();
        if self.offset != self.input.len() {
            return Err(format!(
                "unexpected Gemma tool argument suffix '{}'",
                self.rest()
            ));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        if self.rest().starts_with(GEMMA_QUOTE) {
            return self.parse_string();
        }
        match self.peek_byte() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(_) => self.parse_scalar(),
            None => Err("unexpected end of Gemma tool arguments".to_string()),
        }
    }

    fn parse_string(&mut self) -> Result<Value, String> {
        self.offset += GEMMA_QUOTE.len();
        let end = self
            .rest()
            .find(GEMMA_QUOTE)
            .ok_or_else(|| "unterminated Gemma string argument".to_string())?;
        let value = self.rest()[..end].to_string();
        self.offset += end + GEMMA_QUOTE.len();
        Ok(Value::String(value))
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.expect_byte(b'{')?;
        self.skip_whitespace();
        let mut object = Map::new();
        if self.consume_byte(b'}') {
            return Ok(Value::Object(object));
        }
        loop {
            let key_end = self
                .rest()
                .find(':')
                .ok_or_else(|| "Gemma object key is missing ':'".to_string())?;
            let key = self.rest()[..key_end].trim().to_string();
            validate_name(&key)?;
            self.offset += key_end + 1;
            let value = self.parse_value()?;
            if object.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate Gemma tool argument '{key}'"));
            }
            self.skip_whitespace();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',')?;
            self.skip_whitespace();
        }
        Ok(Value::Object(object))
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.expect_byte(b'[')?;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(Value::Array(values))
    }

    fn parse_scalar(&mut self) -> Result<Value, String> {
        let end = self
            .rest()
            .find([',', '}', ']'])
            .unwrap_or_else(|| self.rest().len());
        let raw = self.rest()[..end].trim();
        if raw.is_empty() {
            return Err("empty Gemma tool argument".to_string());
        }
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| format!("invalid Gemma scalar '{raw}': {error}"))?;
        if value.is_string() || value.is_array() || value.is_object() {
            return Err(format!("invalid Gemma scalar '{raw}'"));
        }
        self.offset += end;
        Ok(value)
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(format!(
                "expected '{}' in Gemma tool arguments",
                expected as char
            ))
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .peek_byte()
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.offset += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.offset).copied()
    }

    fn rest(&self) -> &'a str {
        &self.input[self.offset..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<String> {
        vec!["get_weather".to_string(), "set_config".to_string()]
    }

    #[test]
    fn parses_gemma_tool_calls_and_converts_arguments_to_json() {
        let parsed = parse_assistant_output(
            "<|tool_call>call:get_weather{city:<|\"|>Seoul<|\"|>,days:[1,2],metric:true}<tool_call|>",
            &tools(),
        )
        .unwrap();

        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"city":"Seoul","days":[1,2],"metric":true}"#
        );
    }

    #[test]
    fn parses_gemma_wrapper_with_standard_json_arguments() {
        let parsed = parse_assistant_output(
            r#"<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>"#,
            &tools(),
        )
        .unwrap();

        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"city":"Seoul"}"#);
    }

    #[test]
    fn parses_nested_gemma_arguments_and_content_before_call() {
        let parsed = parse_assistant_output(
            "Checking now. <|tool_call>call:set_config{config:{theme:<|\"|>dark<|\"|>,count:3},value:null}<tool_call|>",
            &tools(),
        )
        .unwrap();

        assert_eq!(parsed.content, "Checking now.");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"config":{"count":3,"theme":"dark"},"value":null}"#
        );
    }

    #[test]
    fn parses_standard_json_tool_call_and_strips_reasoning() {
        let parsed = parse_assistant_output(
            "<think>choose tool</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Seoul\"}}</tool_call>",
            &tools(),
        )
        .unwrap();

        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"city":"Seoul"}"#);
    }

    #[test]
    fn rejects_calls_to_undeclared_tools() {
        let error =
            parse_assistant_output("<|tool_call>call:delete_everything{}<tool_call|>", &tools())
                .unwrap_err();

        assert!(error.contains("undeclared tool"));
    }
}
