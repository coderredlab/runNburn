use serde::Serialize;
use serde_json::{Map, Value};

const GEMMA_CALL_OPEN: &str = "<|tool_call>call:";
const GEMMA_CALL_CLOSE: &str = "<tool_call|>";
const JSON_CALL_OPEN: &str = "<tool_call>";
const JSON_CALL_CLOSE: &str = "</tool_call>";
const GEMMA_QUOTE: &str = "<|\"|>";

const MUSE_END_MESSAGE: &str = "<|eom|>";
const MUSE_END_TURN: &str = "<|eot|>";
const MUSE_MESSAGE: &str = "<|message|>";
const MUSE_NEXT_ASSISTANT: &str = "<|eom|><|start|>assistant";
const MUSE_NEXT_USER: &str = "<|eom|><|start|>assistant to=user<|message|>";
const ATEM_CALLS_OPEN: &str = "<atem:function_calls>";
const ATEM_CALLS_CLOSE: &str = "</atem:function_calls>";
const ATEM_INVOKE_OPEN: &str = "<atem:invoke name=\"";
const ATEM_INVOKE_CLOSE: &str = "</atem:invoke>";
const ATEM_PARAMETER_OPEN: &str = "<atem:parameter name=\"";
const ATEM_PARAMETER_CLOSE: &str = "</atem:parameter>";

#[derive(Debug, Default)]
pub struct AssistantTurnStreamFilter {
    muse_protocol: bool,
    state: AssistantTurnStreamState,
    pending: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum AssistantTurnStreamState {
    #[default]
    Detecting,
    MuseProtocol,
    Passthrough,
}

impl AssistantTurnStreamFilter {
    pub fn new(muse_protocol: bool) -> Self {
        Self {
            muse_protocol,
            ..Self::default()
        }
    }

    pub fn push<F>(&mut self, text: &str, mut emit: F) -> bool
    where
        F: FnMut(&str) -> bool,
    {
        if !self.muse_protocol {
            return emit(text);
        }
        match self.state {
            AssistantTurnStreamState::Detecting => {
                self.pending.push_str(text);
                let candidate = self.pending.trim_start();
                if "to=".starts_with(candidate) {
                    return true;
                }
                if !candidate.starts_with("to=") {
                    self.state = AssistantTurnStreamState::Passthrough;
                    let pending = std::mem::take(&mut self.pending);
                    return emit(&pending);
                }
                let Some(message_start) = candidate.find(MUSE_MESSAGE) else {
                    return true;
                };
                let recipient = candidate[3..message_start].trim();
                if recipient == "user" {
                    let body = candidate[message_start + MUSE_MESSAGE.len()..].to_string();
                    self.pending.clear();
                    self.state = AssistantTurnStreamState::Passthrough;
                    return body.is_empty() || emit(&body);
                }
                self.state = AssistantTurnStreamState::MuseProtocol;
                self.emit_muse_user_turn(emit)
            }
            AssistantTurnStreamState::MuseProtocol => {
                self.pending.push_str(text);
                self.emit_muse_user_turn(emit)
            }
            AssistantTurnStreamState::Passthrough => emit(text),
        }
    }

    pub fn finish<F>(&mut self, mut emit: F) -> bool
    where
        F: FnMut(&str) -> bool,
    {
        match self.state {
            AssistantTurnStreamState::Detecting => {
                let pending = std::mem::take(&mut self.pending);
                self.state = AssistantTurnStreamState::Passthrough;
                let candidate = pending.trim_start();
                if candidate.starts_with("to=") && candidate != "to=" {
                    true
                } else {
                    pending.is_empty() || emit(&pending)
                }
            }
            AssistantTurnStreamState::MuseProtocol => {
                self.pending.clear();
                true
            }
            AssistantTurnStreamState::Passthrough => true,
        }
    }

    fn emit_muse_user_turn<F>(&mut self, mut emit: F) -> bool
    where
        F: FnMut(&str) -> bool,
    {
        let Some(user_start) = self.pending.find(MUSE_NEXT_USER) else {
            return true;
        };
        let body_start = user_start + MUSE_NEXT_USER.len();
        let body = self.pending[body_start..].to_string();
        self.pending.clear();
        self.state = AssistantTurnStreamState::Passthrough;
        body.is_empty() || emit(&body)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Gemma,
    Muse,
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
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

pub fn parse_assistant_output(
    text: &str,
    allowed_tools: &[String],
) -> Result<ParsedAssistantOutput, String> {
    parse_assistant_output_with_format(text, allowed_tools, ToolCallFormat::Json)
}

pub fn parse_assistant_output_with_format(
    text: &str,
    allowed_tools: &[String],
    format: ToolCallFormat,
) -> Result<ParsedAssistantOutput, String> {
    let muse_output = if format == ToolCallFormat::Muse {
        parse_muse_output(text, allowed_tools)?
    } else {
        None
    };
    let is_muse = muse_output.is_some();
    let (reasoning_content, visible_text, mut tool_calls) = match muse_output {
        Some(output) => (output.reasoning_content, output.content, output.tool_calls),
        None => (None, text.to_string(), Vec::new()),
    };
    let text = strip_reasoning(&visible_text);
    let mut content = String::new();
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

    let content = if is_muse || !allowed_tools.is_empty() {
        content.trim().to_string()
    } else {
        content
    };
    Ok(ParsedAssistantOutput {
        content,
        reasoning_content,
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

struct MuseOutput {
    reasoning_content: Option<String>,
    content: String,
    tool_calls: Vec<ParsedToolCall>,
}

fn parse_muse_output(text: &str, allowed_tools: &[String]) -> Result<Option<MuseOutput>, String> {
    let mut remaining = text.trim_start();
    if !remaining.starts_with("to=") || remaining == "to=" {
        return Ok(None);
    }

    let mut reasoning = Vec::new();
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    loop {
        let Some(message_start) = remaining.find(MUSE_MESSAGE) else {
            return Ok(Some(MuseOutput {
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning.join("\n\n"))
                },
                content,
                tool_calls,
            }));
        };
        let recipient = remaining[3..message_start].trim();
        if recipient.is_empty() {
            return Ok(None);
        }
        let body_and_rest = &remaining[message_start + MUSE_MESSAGE.len()..];
        let (body, next) = match body_and_rest.find(MUSE_NEXT_ASSISTANT) {
            Some(next_start) => (
                &body_and_rest[..next_start],
                Some(&body_and_rest[next_start + MUSE_NEXT_ASSISTANT.len()..]),
            ),
            None => (body_and_rest, None),
        };
        let body = body
            .strip_suffix(MUSE_END_TURN)
            .or_else(|| body.strip_suffix(MUSE_END_MESSAGE))
            .unwrap_or(body);

        match recipient {
            "self" => {
                let body = body.trim();
                if !body.is_empty() {
                    reasoning.push(body.to_string());
                }
            }
            "user" => content.push_str(body),
            tool_name => {
                if !allowed_tools.iter().any(|name| name == tool_name) {
                    return Err(format!("model requested undeclared tool '{tool_name}'"));
                }
                tool_calls.extend(parse_atem_calls(body, tool_name, allowed_tools)?);
            }
        }

        let Some(next) = next else {
            break;
        };
        remaining = next.trim_start();
        if !remaining.starts_with("to=") {
            return Err("Muse assistant turn is missing a recipient".to_string());
        }
    }

    Ok(Some(MuseOutput {
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning.join("\n\n"))
        },
        content,
        tool_calls,
    }))
}

fn parse_atem_calls(
    text: &str,
    recipient: &str,
    allowed_tools: &[String],
) -> Result<Vec<ParsedToolCall>, String> {
    let calls_start = text
        .find(ATEM_CALLS_OPEN)
        .ok_or_else(|| format!("Muse tool turn '{recipient}' is missing ATEM calls"))?;
    let calls_body = &text[calls_start + ATEM_CALLS_OPEN.len()..];
    let calls_end = calls_body
        .find(ATEM_CALLS_CLOSE)
        .ok_or_else(|| "Muse ATEM calls are not terminated".to_string())?;
    let mut remaining = &calls_body[..calls_end];
    let mut calls = Vec::new();

    while let Some(invoke_start) = remaining.find(ATEM_INVOKE_OPEN) {
        let invoke = &remaining[invoke_start + ATEM_INVOKE_OPEN.len()..];
        let name_end = invoke
            .find("\">")
            .ok_or_else(|| "Muse ATEM invoke is missing a name terminator".to_string())?;
        let name = &invoke[..name_end];
        if name != recipient {
            return Err(format!(
                "Muse tool recipient '{recipient}' does not match invoke '{name}'"
            ));
        }
        if !allowed_tools.iter().any(|allowed| allowed == name) {
            return Err(format!("model requested undeclared tool '{name}'"));
        }
        let invoke_body = &invoke[name_end + 2..];
        let invoke_end = invoke_body
            .find(ATEM_INVOKE_CLOSE)
            .ok_or_else(|| "Muse ATEM invoke is not terminated".to_string())?;
        let mut parameters = &invoke_body[..invoke_end];
        let mut arguments = Map::new();

        while let Some(parameter_start) = parameters.find(ATEM_PARAMETER_OPEN) {
            let parameter = &parameters[parameter_start + ATEM_PARAMETER_OPEN.len()..];
            let name_end = parameter
                .find("\">")
                .ok_or_else(|| "Muse ATEM parameter is missing a name terminator".to_string())?;
            let parameter_name = &parameter[..name_end];
            validate_name(parameter_name)?;
            let value_and_rest = &parameter[name_end + 2..];
            let value_end = value_and_rest
                .find(ATEM_PARAMETER_CLOSE)
                .ok_or_else(|| "Muse ATEM parameter is not terminated".to_string())?;
            let raw_value = &value_and_rest[..value_end];
            let value = serde_json::from_str(raw_value)
                .unwrap_or_else(|_| Value::String(raw_value.to_string()));
            if arguments
                .insert(parameter_name.to_string(), value)
                .is_some()
            {
                return Err(format!(
                    "duplicate Muse ATEM tool argument '{parameter_name}'"
                ));
            }
            parameters = &value_and_rest[value_end + ATEM_PARAMETER_CLOSE.len()..];
        }

        let mut arguments = Value::Object(arguments);
        sort_object_keys(&mut arguments);
        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments: serde_json::to_string(&arguments)
                .map_err(|error| format!("serialize Muse ATEM arguments: {error}"))?,
        });
        remaining = &invoke_body[invoke_end + ATEM_INVOKE_CLOSE.len()..];
    }

    if calls.is_empty() {
        return Err("Muse ATEM block contains no invokes".to_string());
    }
    Ok(calls)
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

    fn parse_muse(text: &str, allowed_tools: &[String]) -> Result<ParsedAssistantOutput, String> {
        parse_assistant_output_with_format(text, allowed_tools, ToolCallFormat::Muse)
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
    fn separates_muse_reasoning_from_user_facing_content() {
        let parsed = parse_muse(
            r#" to=self<|message|>Think privately.<|eom|><|start|>assistant to=user<|message|>Final answer."#,
            &[],
        )
        .unwrap();

        assert_eq!(
            parsed.reasoning_content.as_deref(),
            Some("Think privately.")
        );
        assert_eq!(parsed.content, "Final answer.");
    }

    #[test]
    fn parses_muse_direct_user_turn_without_reasoning() {
        let parsed = parse_muse(r#" to=user<|message|>Direct answer.<|eot|>"#, &[]).unwrap();

        assert_eq!(parsed.reasoning_content, None);
        assert_eq!(parsed.content, "Direct answer.");
    }

    #[test]
    fn accumulates_multiple_muse_self_turns() {
        let parsed = parse_muse(
            r#"to=self<|message|>first<|eom|><|start|>assistant to=self<|message|>second<|eom|><|start|>assistant to=user<|message|>answer"#,
            &[],
        )
        .unwrap();

        assert_eq!(parsed.reasoning_content.as_deref(), Some("first\n\nsecond"));
        assert_eq!(parsed.content, "answer");
    }

    #[test]
    fn parses_muse_atem_tool_call() {
        let parsed = parse_muse(
            r#"to=self<|message|>use weather<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">Seoul</atem:parameter>
<atem:parameter name="days">[1,2]</atem:parameter>
</atem:invoke>
</atem:function_calls><|eot|>"#,
            &tools(),
        )
        .unwrap();

        assert_eq!(parsed.reasoning_content.as_deref(), Some("use weather"));
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"city":"Seoul","days":[1,2]}"#
        );
    }

    #[test]
    fn partial_muse_header_is_not_exposed_as_content() {
        let parsed = parse_muse("to=self<|mess", &[]).unwrap();

        assert_eq!(parsed.content, "");
        assert_eq!(parsed.reasoning_content, None);
    }

    #[test]
    fn preserves_outer_whitespace_for_plain_output() {
        let parsed = parse_assistant_output(" answer ", &[]).unwrap();

        assert_eq!(parsed.content, " answer ");
    }

    #[test]
    fn preserves_plain_outputs_that_are_muse_prefixes() {
        for text in ["t", "to", "to=", "to=foo"] {
            let parsed = parse_assistant_output(text, &[]).unwrap();
            assert_eq!(parsed.content, text);
        }
    }

    #[test]
    fn muse_stream_filter_hides_reasoning_across_chunk_boundaries() {
        let mut filter = AssistantTurnStreamFilter::new(true);
        let mut visible = String::new();
        for chunk in [
            " to=se",
            r#"lf<|message|>private"#,
            r#"<|eom|><|start|>assis"#,
            r#"tant to=user<|message|>public"#,
        ] {
            assert!(filter.push(chunk, |text| {
                visible.push_str(text);
                true
            }));
        }
        assert!(filter.finish(|text| {
            visible.push_str(text);
            true
        }));

        assert_eq!(visible, "public");
    }

    #[test]
    fn muse_stream_filter_handles_direct_user_turn() {
        let mut filter = AssistantTurnStreamFilter::new(true);
        let mut visible = String::new();
        for chunk in [r#" to=user<|mes"#, r#"sage|>direct"#] {
            assert!(filter.push(chunk, |text| {
                visible.push_str(text);
                true
            }));
        }

        assert_eq!(visible, "direct");
    }

    #[test]
    fn muse_stream_filter_hides_multiple_self_turns() {
        let mut filter = AssistantTurnStreamFilter::new(true);
        let mut visible = String::new();
        for chunk in [
            r#"to=self<|message|>one<|eom|><|start|>assistant "#,
            r#"to=self<|message|>two<|eom|><|start|>assistant to=user"#,
            r#"<|message|>answer"#,
        ] {
            assert!(filter.push(chunk, |text| {
                visible.push_str(text);
                true
            }));
        }

        assert_eq!(visible, "answer");
    }

    #[test]
    fn partial_muse_stream_header_is_not_emitted_on_finish() {
        let mut filter = AssistantTurnStreamFilter::new(true);
        let mut visible = String::new();
        assert!(filter.push("to=self<|mess", |text| {
            visible.push_str(text);
            true
        }));
        assert!(filter.finish(|text| {
            visible.push_str(text);
            true
        }));

        assert_eq!(visible, "");
    }

    #[test]
    fn stream_filter_preserves_non_muse_text() {
        let mut filter = AssistantTurnStreamFilter::default();
        let mut visible = String::new();
        assert!(filter.push(" normal", |text| {
            visible.push_str(text);
            true
        }));
        assert!(filter.push(" output", |text| {
            visible.push_str(text);
            true
        }));
        assert_eq!(visible, " normal output");
    }

    #[test]
    fn stream_filter_preserves_plain_muse_prefixes_on_finish() {
        for text in ["t", "to", "to=", "to=foo"] {
            let mut filter = AssistantTurnStreamFilter::default();
            let mut visible = String::new();
            assert!(filter.push(text, |piece| {
                visible.push_str(piece);
                true
            }));
            assert!(filter.finish(|piece| {
                visible.push_str(piece);
                true
            }));
            assert_eq!(visible, text);
        }
    }

    #[test]
    fn rejects_calls_to_undeclared_tools() {
        let error =
            parse_assistant_output("<|tool_call>call:delete_everything{}<tool_call|>", &tools())
                .unwrap_err();

        assert!(error.contains("undeclared tool"));
    }
}
