use super::http::ApiError;
use rnb_llm::{GenerationConstraint, ToolCallFormat};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
}

#[derive(Debug, Clone)]
struct FunctionTool {
    name: String,
    parameters: Value,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedTools {
    pub prompt_definitions: Vec<Value>,
    pub names: Vec<String>,
    definitions: Vec<FunctionTool>,
    choice: ToolChoice,
}

pub(super) fn prepare_tools(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<PreparedTools, ApiError> {
    let definitions = match tools {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(definitions)) => definitions.as_slice(),
        Some(_) => return Err(invalid("tools", "tools must be an array")),
    };
    if definitions.len() > 128 {
        return Err(invalid(
            "tools",
            "tools may contain at most 128 definitions",
        ));
    }

    let mut prompt_definitions = Vec::with_capacity(definitions.len());
    let mut functions = Vec::with_capacity(definitions.len());
    for (index, tool) in definitions.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(invalid(
                "tools",
                format!("tools[{index}].type must be 'function'"),
            ));
        }
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                invalid(
                    "tools",
                    format!("tools[{index}].function must be an object"),
                )
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tools", format!("tools[{index}].function.name is required")))?;
        validate_name(name, "tools")?;
        if functions
            .iter()
            .any(|definition: &FunctionTool| definition.name == name)
        {
            return Err(invalid("tools", format!("duplicate tool name '{name}'")));
        }
        if function
            .get("description")
            .is_some_and(|description| !description.is_string())
        {
            return Err(invalid(
                "tools",
                format!("tools[{index}].function.description must be a string"),
            ));
        }
        if function
            .get("strict")
            .is_some_and(|strict| !strict.is_boolean())
        {
            return Err(invalid(
                "tools",
                format!("tools[{index}].function.strict must be a boolean"),
            ));
        }
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        let mut parameters = parameters.as_object().cloned().ok_or_else(|| {
            invalid(
                "tools",
                format!("tools[{index}].function.parameters must be an object"),
            )
        })?;
        parameters
            .entry("type".to_string())
            .or_insert_with(|| Value::String("object".to_string()));
        if parameters.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid(
                "tools",
                format!("tools[{index}].function.parameters must describe an object"),
            ));
        }

        prompt_definitions.push(tool.clone());
        functions.push(FunctionTool {
            name: name.to_string(),
            parameters: Value::Object(parameters),
        });
    }

    let choice = parse_tool_choice(tool_choice)?;
    match &choice {
        ToolChoice::Required if functions.is_empty() => {
            return Err(invalid(
                "tool_choice",
                "tool_choice 'required' requires at least one tool",
            ));
        }
        ToolChoice::Function(name) if !functions.iter().any(|tool| &tool.name == name) => {
            return Err(invalid(
                "tool_choice",
                format!("tool_choice references unknown function '{name}'"),
            ));
        }
        _ => {}
    }

    if choice == ToolChoice::None {
        Ok(PreparedTools {
            prompt_definitions: Vec::new(),
            names: Vec::new(),
            definitions: Vec::new(),
            choice,
        })
    } else {
        Ok(PreparedTools {
            names: functions.iter().map(|tool| tool.name.clone()).collect(),
            prompt_definitions,
            definitions: functions,
            choice,
        })
    }
}

pub(super) fn prepare_generation_constraint(
    response_format: Option<&Value>,
    tools: &PreparedTools,
    parallel_tool_calls: bool,
    tool_format: ToolCallFormat,
) -> Result<Option<GenerationConstraint>, ApiError> {
    let response_schema = parse_response_format(response_format)?;
    if tools.definitions.is_empty() {
        return Ok(response_schema.map(GenerationConstraint::JsonSchema));
    }

    let selected = match &tools.choice {
        ToolChoice::Function(name) => tools
            .definitions
            .iter()
            .filter(|tool| &tool.name == name)
            .cloned()
            .collect::<Vec<_>>(),
        _ => tools.definitions.clone(),
    };
    let grammar = build_tool_grammar(
        &selected,
        &tools.choice,
        parallel_tool_calls,
        tool_format,
        response_schema.as_ref(),
    );
    Ok(Some(GenerationConstraint::Lark(grammar)))
}

fn parse_tool_choice(choice: Option<&Value>) -> Result<ToolChoice, ApiError> {
    match choice {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(choice)) if choice == "none" => Ok(ToolChoice::None),
        Some(Value::String(choice)) if choice == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(choice)) if choice == "required" => Ok(ToolChoice::Required),
        Some(Value::Object(choice)) => {
            if choice.get("type").and_then(Value::as_str) != Some("function") {
                return Err(invalid(
                    "tool_choice",
                    "tool_choice.type must be 'function'",
                ));
            }
            let name = choice
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice", "tool_choice.function.name is required"))?;
            validate_name(name, "tool_choice")?;
            Ok(ToolChoice::Function(name.to_string()))
        }
        Some(_) => Err(invalid(
            "tool_choice",
            "tool_choice must be 'auto', 'none', 'required', or a function object",
        )),
    }
}

fn parse_response_format(format: Option<&Value>) -> Result<Option<Value>, ApiError> {
    let Some(format) = format.filter(|format| !format.is_null()) else {
        return Ok(None);
    };
    let object = format
        .as_object()
        .ok_or_else(|| invalid("response_format", "response_format must be an object"))?;
    match object.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type": "object"}))),
        Some("json_schema") => {
            let definition = object
                .get("json_schema")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    invalid(
                        "response_format",
                        "response_format.json_schema must be an object",
                    )
                })?;
            let name = definition
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid(
                        "response_format",
                        "response_format.json_schema.name is required",
                    )
                })?;
            validate_name(name, "response_format")?;
            if definition
                .get("description")
                .is_some_and(|description| !description.is_string())
            {
                return Err(invalid(
                    "response_format",
                    "response_format.json_schema.description must be a string",
                ));
            }
            if definition
                .get("strict")
                .is_some_and(|strict| !strict.is_boolean())
            {
                return Err(invalid(
                    "response_format",
                    "response_format.json_schema.strict must be a boolean",
                ));
            }
            let schema = definition
                .get("schema")
                .filter(|schema| schema.is_object())
                .cloned()
                .ok_or_else(|| {
                    invalid(
                        "response_format",
                        "response_format.json_schema.schema must be an object",
                    )
                })?;
            Ok(Some(schema))
        }
        Some(kind) => Err(invalid(
            "response_format",
            format!("unsupported response_format type '{kind}'"),
        )),
        None => Err(invalid(
            "response_format",
            "response_format.type is required",
        )),
    }
}

fn build_tool_grammar(
    tools: &[FunctionTool],
    choice: &ToolChoice,
    parallel: bool,
    format: ToolCallFormat,
    response_schema: Option<&Value>,
) -> String {
    let required = matches!(choice, ToolChoice::Required | ToolChoice::Function(_));
    let mut grammar = String::new();
    match (required, response_schema) {
        (true, _) => grammar.push_str(if parallel {
            "start: tool_call+\n"
        } else {
            "start: tool_call\n"
        }),
        (false, Some(schema)) => {
            grammar.push_str(if parallel {
                "start: response | tool_call+\n"
            } else {
                "start: response | tool_call\n"
            });
            grammar.push_str(&format!("response: %json {}\n", compact(schema)));
        }
        (false, None) => {
            grammar.push_str(if parallel {
                "start: no_call | first_tool_call tool_call*\n"
            } else {
                "start: no_call | first_tool_call\n"
            });
            grammar.push_str(match format {
                ToolCallFormat::Gemma => "no_call: NO_GEMMA_CALL\n",
                ToolCallFormat::Json => "no_call: NO_JSON_CALL\n",
            });
        }
    }

    match format {
        ToolCallFormat::Gemma => {
            if required || response_schema.is_some() {
                grammar.push_str("tool_call: \"<|tool_call>call:\" tool_body \"<tool_call|>\"\n");
            } else {
                grammar.push_str("first_tool_call: tool_header tool_body \"<tool_call|>\"\n");
                grammar.push_str("tool_call: \"<|tool_call>call:\" tool_body \"<tool_call|>\"\n");
                grammar.push_str("tool_header[lazy]: TEXT \"<|tool_call>call:\"\n");
            }
            grammar.push_str("tool_body: ");
            grammar.push_str(
                &(0..tools.len())
                    .map(|index| format!("tool_{index}"))
                    .collect::<Vec<_>>()
                    .join(" | "),
            );
            grammar.push('\n');
            for (index, tool) in tools.iter().enumerate() {
                grammar.push_str(&format!(
                    "tool_{index}: {} args_{index}\nargs_{index}: %json {}\n",
                    compact_string(&tool.name),
                    compact(&tool.parameters)
                ));
            }
        }
        ToolCallFormat::Json => {
            if required || response_schema.is_some() {
                grammar.push_str("tool_call: \"<tool_call>\" call_body \"</tool_call>\"\n");
            } else {
                grammar.push_str("first_tool_call: tool_header call_body \"</tool_call>\"\n");
                grammar.push_str("tool_call: \"<tool_call>\" call_body \"</tool_call>\"\n");
                grammar.push_str("tool_header[lazy]: TEXT \"<tool_call>\"\n");
            }
            let choices = tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "object",
                        "properties": {
                            "name": {"const": tool.name},
                            "arguments": tool.parameters
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": false
                    })
                })
                .collect::<Vec<_>>();
            grammar.push_str(&format!(
                "call_body: %json {}\n",
                compact(&json!({"anyOf": choices}))
            ));
        }
    }

    if !required && response_schema.is_none() {
        grammar.push_str("TEXT: /(?s:.*)/\n");
        match format {
            ToolCallFormat::Gemma => grammar
                .push_str("NO_GEMMA_CALL: /(?s:.*)/ & ~/(?s:.*)<\\|tool_call>call:(?s:.*)/\n"),
            ToolCallFormat::Json => {
                grammar.push_str("NO_JSON_CALL: /(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/\n")
            }
        }
    }
    grammar
}

fn compact(value: &Value) -> String {
    serde_json::to_string(value).expect("JSON values always serialize")
}

fn compact_string(value: &str) -> String {
    serde_json::to_string(value).expect("strings always serialize")
}

fn validate_name(name: &str, param: &'static str) -> Result<(), ApiError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid(
            param,
            "name must contain 1-64 letters, digits, underscores, or hyphens",
        ));
    }
    Ok(())
}

fn invalid(param: &'static str, message: impl Into<String>) -> ApiError {
    ApiError::invalid(message, Some(param), Some("invalid_value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_llm::tokenizer::{SpecialTokens, Tokenizer, Vocab};

    fn ascii_tokenizer() -> Tokenizer {
        let mut tokens = (32u8..=126)
            .map(|byte| (byte as char).to_string())
            .collect::<Vec<_>>();
        let bos = tokens.len() as u32;
        tokens.push("<bos>".to_string());
        let eos = tokens.len() as u32;
        tokens.push("<eos>".to_string());
        Tokenizer::new_sentencepiece_with_config(
            Vocab::new(
                tokens,
                SpecialTokens {
                    bos,
                    eos,
                    pad: None,
                },
            ),
            Vec::new(),
            Vec::new(),
            false,
            false,
        )
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            }
        }])
    }

    #[test]
    fn required_and_named_choices_select_constrained_tools() {
        let definitions = tools();
        let required = prepare_tools(
            Some(&definitions),
            Some(&Value::String("required".to_string())),
        )
        .unwrap();
        let constraint =
            prepare_generation_constraint(None, &required, false, ToolCallFormat::Gemma)
                .unwrap()
                .unwrap();
        let GenerationConstraint::Lark(grammar) = &constraint else {
            panic!("expected Lark grammar");
        };
        assert!(grammar.contains("start: tool_call"));
        assert!(grammar.contains("\"get_weather\" args_0"));
        constraint.validate(&ascii_tokenizer()).unwrap();

        let selected = prepare_tools(
            Some(&definitions),
            Some(&json!({"type":"function","function":{"name":"get_weather"}})),
        )
        .unwrap();
        assert_eq!(selected.choice, ToolChoice::Function("get_weather".into()));
    }

    #[test]
    fn response_format_json_schema_becomes_decoder_constraint() {
        let prepared = prepare_tools(None, None).unwrap();
        let constraint = prepare_generation_constraint(
            Some(&json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": false
                    }
                }
            })),
            &prepared,
            true,
            ToolCallFormat::Json,
        )
        .unwrap();
        assert!(matches!(
            constraint,
            Some(GenerationConstraint::JsonSchema(_))
        ));
    }

    #[test]
    fn auto_grammar_routes_tool_markers_through_constrained_calls() {
        let definitions = tools();
        let prepared = prepare_tools(Some(&definitions), None).unwrap();
        let constraint =
            prepare_generation_constraint(None, &prepared, false, ToolCallFormat::Json)
                .unwrap()
                .unwrap();
        let GenerationConstraint::Lark(grammar) = &constraint else {
            panic!("expected Lark grammar");
        };
        assert!(grammar.contains("start: no_call | first_tool_call"));
        assert!(grammar.contains("NO_JSON_CALL"));
        constraint.validate(&ascii_tokenizer()).unwrap();

        let parallel = prepare_generation_constraint(None, &prepared, true, ToolCallFormat::Json)
            .unwrap()
            .unwrap();
        parallel.validate(&ascii_tokenizer()).unwrap();
    }
}
