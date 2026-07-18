use super::http::ApiError;
use super::responses::{invalid, unsupported, validate_name};
use serde_json::{json, Value};

pub(super) fn normalize_tools(tools: Option<Value>) -> Result<(Option<Value>, Value), ApiError> {
    let Some(tools) = tools.filter(|value| !value.is_null()) else {
        return Ok((None, json!([])));
    };
    let definitions = tools
        .as_array()
        .ok_or_else(|| invalid("tools", "tools must be an array"))?;
    let mut chat_tools = Vec::with_capacity(definitions.len());
    let mut response_tools = Vec::with_capacity(definitions.len());
    for (index, definition) in definitions.iter().enumerate() {
        let object = definition
            .as_object()
            .ok_or_else(|| invalid("tools", format!("tools[{index}] must be an object")))?;
        if object.get("type").and_then(Value::as_str) != Some("function") {
            return Err(unsupported(
                "tools",
                format!("tools[{index}] must be a function tool"),
            ));
        }
        for field in ["allowed_callers", "defer_loading", "output_schema"] {
            if object.get(field).is_some_and(|value| !value.is_null()) {
                return Err(unsupported(
                    "tools",
                    format!("tools[{index}].{field} is not supported"),
                ));
            }
        }
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tools", format!("tools[{index}].name is required")))?;
        validate_name(name, "tools")?;
        let description = match object.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(invalid(
                    "tools",
                    format!("tools[{index}].description must be a string"),
                ));
            }
        };
        let parameters = match object.get("parameters") {
            None | Some(Value::Null) => json!({"type": "object"}),
            Some(Value::Object(value)) => Value::Object(value.clone()),
            Some(_) => {
                return Err(invalid(
                    "tools",
                    format!("tools[{index}].parameters must be an object"),
                ));
            }
        };
        let strict = match object.get("strict") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(invalid(
                    "tools",
                    format!("tools[{index}].strict must be a boolean"),
                ));
            }
        };
        let mut function =
            json!({"name": name, "parameters": parameters.clone(), "strict": strict});
        let mut response = json!({
            "type": "function",
            "name": name,
            "parameters": parameters,
            "strict": strict
        });
        if let Some(description) = description {
            function["description"] = json!(description);
            response["description"] = json!(description);
        }
        chat_tools.push(json!({"type": "function", "function": function}));
        response_tools.push(response);
    }
    Ok((Some(Value::Array(chat_tools)), Value::Array(response_tools)))
}

pub(super) fn normalize_tool_choice(
    choice: Option<Value>,
) -> Result<(Option<Value>, Value), ApiError> {
    let Some(choice) = choice.filter(|value| !value.is_null()) else {
        return Ok((None, json!("auto")));
    };
    if let Some(choice) = choice.as_str() {
        if matches!(choice, "none" | "auto" | "required") {
            return Ok((Some(json!(choice)), json!(choice)));
        }
        return Err(invalid(
            "tool_choice",
            format!("unsupported tool_choice '{choice}'"),
        ));
    }
    let object = choice
        .as_object()
        .ok_or_else(|| invalid("tool_choice", "tool_choice must be a string or object"))?;
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(unsupported(
            "tool_choice",
            "only function tool choice objects are supported",
        ));
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("tool_choice", "tool_choice.name is required"))?;
    validate_name(name, "tool_choice")?;
    Ok((
        Some(json!({"type": "function", "function": {"name": name}})),
        json!({"type": "function", "name": name}),
    ))
}

pub(super) fn normalize_text(text: Option<Value>) -> Result<(Option<Value>, Value), ApiError> {
    let default = json!({"format": {"type": "text"}});
    let Some(text) = text.filter(|value| !value.is_null()) else {
        return Ok((None, default));
    };
    let object = text
        .as_object()
        .ok_or_else(|| invalid("text", "text must be an object"))?;
    if object
        .get("verbosity")
        .is_some_and(|value| !value.is_null())
    {
        return Err(unsupported("text.verbosity", "verbosity is not supported"));
    }
    let Some(format) = object.get("format").filter(|value| !value.is_null()) else {
        return Ok((None, default));
    };
    let format = format
        .as_object()
        .ok_or_else(|| invalid("text.format", "text.format must be an object"))?;
    match format.get("type").and_then(Value::as_str) {
        Some("text") => Ok((None, json!({"format": {"type": "text"}}))),
        Some("json_object") => Ok((
            Some(json!({"type": "json_object"})),
            json!({"format": {"type": "json_object"}}),
        )),
        Some("json_schema") => {
            let name = format
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("text.format", "text.format.name is required"))?;
            validate_name(name, "text.format")?;
            let schema = format
                .get("schema")
                .filter(|value| value.is_object())
                .cloned()
                .ok_or_else(|| invalid("text.format", "text.format.schema must be an object"))?;
            let strict = match format.get("strict") {
                None | Some(Value::Null) => None,
                Some(Value::Bool(value)) => Some(*value),
                Some(_) => {
                    return Err(invalid(
                        "text.format",
                        "text.format.strict must be a boolean",
                    ));
                }
            };
            let description = match format.get("description") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) => Some(value.clone()),
                Some(_) => {
                    return Err(invalid(
                        "text.format",
                        "text.format.description must be a string",
                    ));
                }
            };
            let mut flat = json!({"type": "json_schema", "name": name, "schema": schema.clone()});
            let mut nested = json!({"name": name, "schema": schema});
            if let Some(strict) = strict {
                flat["strict"] = json!(strict);
                nested["strict"] = json!(strict);
            }
            if let Some(description) = description {
                flat["description"] = json!(description);
                nested["description"] = json!(description);
            }
            Ok((
                Some(json!({"type": "json_schema", "json_schema": nested})),
                json!({"format": flat}),
            ))
        }
        Some(kind) => Err(unsupported(
            "text.format",
            format!("text format '{kind}' is not supported"),
        )),
        None => Err(invalid("text.format", "text.format.type is required")),
    }
}

pub(super) fn normalize_metadata(metadata: Option<Value>) -> Result<Value, ApiError> {
    let Some(metadata) = metadata.filter(|value| !value.is_null()) else {
        return Ok(Value::Null);
    };
    let object = metadata
        .as_object()
        .ok_or_else(|| invalid("metadata", "metadata must be an object"))?;
    if object.len() > 16 {
        return Err(invalid(
            "metadata",
            "metadata may contain at most 16 entries",
        ));
    }
    for (key, value) in object {
        if key.len() > 64 || value.as_str().is_none_or(|value| value.len() > 512) {
            return Err(invalid(
                "metadata",
                "metadata keys must be at most 64 characters and values must be strings of at most 512 characters",
            ));
        }
    }
    Ok(metadata)
}

pub(super) fn normalize_stream_options(options: Option<Value>) -> Result<bool, ApiError> {
    let Some(options) = options.filter(|value| !value.is_null()) else {
        return Ok(false);
    };
    let object = options
        .as_object()
        .ok_or_else(|| invalid("stream_options", "stream_options must be an object"))?;
    if object
        .keys()
        .any(|name| name.as_str() != "include_obfuscation")
    {
        return Err(unsupported(
            "stream_options",
            "unsupported response stream option",
        ));
    }
    if object
        .get("include_obfuscation")
        .is_some_and(|value| !value.is_null() && !value.is_boolean())
    {
        return Err(invalid(
            "stream_options.include_obfuscation",
            "stream_options.include_obfuscation must be a boolean",
        ));
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_flat_response_tools_and_specific_choice() {
        let (chat, response) = normalize_tools(Some(json!([{
            "type": "function",
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object"},
            "strict": true
        }])))
        .unwrap();
        let (choice, echoed) = normalize_tool_choice(Some(json!({
            "type": "function",
            "name": "get_weather"
        })))
        .unwrap();

        assert_eq!(chat.unwrap()[0]["function"]["name"], "get_weather");
        assert_eq!(response[0]["name"], "get_weather");
        assert_eq!(choice.unwrap()["function"]["name"], "get_weather");
        assert_eq!(echoed["name"], "get_weather");
    }

    #[test]
    fn converts_responses_json_schema_to_chat_constraint_shape() {
        let (constraint, echoed) = normalize_text(Some(json!({
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": true
            }
        })))
        .unwrap();

        assert_eq!(constraint.unwrap()["json_schema"]["name"], "answer");
        assert_eq!(echoed["format"]["type"], "json_schema");
    }
    #[test]
    fn accepts_only_responses_stream_options() {
        assert!(!normalize_stream_options(Some(json!({
            "include_obfuscation": true
        })))
        .unwrap());
        assert!(normalize_stream_options(Some(json!({
            "include_usage": true
        })))
        .is_err());
        assert!(normalize_stream_options(Some(json!({
            "include_obfuscation": "true"
        })))
        .is_err());
    }
}
