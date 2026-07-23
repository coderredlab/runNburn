use super::generation::{run_generation, GeneratedCompletion};
use super::http::{write_json_response, write_sse_event, write_sse_headers, ApiError};
use super::responses::PreparedResponseRequest;
use super::session_store::ResponseStore;
use rnb_llm::{
    Engine, EngineSequenceState, GenerationCancellation, ParsedAssistantOutput, ParsedToolCall,
};
use serde_json::{json, Value};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_RESPONSE_ID: AtomicU64 = AtomicU64::new(1);

struct ResponseIdentity {
    id: String,
    suffix: String,
    created_at: u64,
}

pub(super) fn complete_response(
    stream: &mut TcpStream,
    engine: &mut Engine,
    store: &mut ResponseStore,
    cancellation: &GenerationCancellation,
    model_name: &str,
    prepared: PreparedResponseRequest,
) -> Result<(), ApiError> {
    let identity = ResponseIdentity::new();
    let result = run_generation(
        engine,
        &prepared.generation,
        prepared.resume_state.as_deref(),
        Some(cancellation),
        None,
    )?;
    let status = final_status(&result);
    let output = output_items(&identity, &result.output, status);
    let body = response_value(
        &identity,
        model_name,
        &prepared,
        status,
        output,
        usage_value(&result),
    );
    let stored_output = output_items(&identity, &result.output, status);
    let sequence_state =
        capture_sequence_state(store, engine, &prepared, &result, &body, &stored_output);
    if cancellation.is_cancelled() {
        return Err(ApiError::cancelled());
    }
    store.commit(
        &prepared.history_items,
        prepared.conversation_id.as_deref(),
        prepared.store,
        body.clone(),
        &stored_output,
        sequence_state,
        unix_timestamp(),
    )?;
    write_json_response(stream, 200, &body)
        .map_err(|error| ApiError::internal(format!("write response: {error}")))
}

pub(super) fn stream_response(
    stream: &mut TcpStream,
    engine: &mut Engine,
    store: &mut ResponseStore,
    cancellation: &GenerationCancellation,
    model_name: &str,
    prepared: PreparedResponseRequest,
) -> Result<(), ApiError> {
    let identity = ResponseIdentity::new();
    write_sse_headers(stream)
        .map_err(|error| ApiError::internal(format!("write stream headers: {error}")))?;
    let mut sequence = 0_u64;
    let initial_response = response_value(
        &identity,
        model_name,
        &prepared,
        "in_progress",
        Vec::new(),
        Value::Null,
    );
    if !write_event(
        stream,
        &mut sequence,
        json!({"type": "response.created", "response": initial_response}),
    ) {
        return Ok(());
    }
    let in_progress = response_value(
        &identity,
        model_name,
        &prepared,
        "in_progress",
        Vec::new(),
        Value::Null,
    );
    if !write_event(
        stream,
        &mut sequence,
        json!({"type": "response.in_progress", "response": in_progress}),
    ) {
        return Ok(());
    }

    let tool_mode = !prepared.generation.tool_names.is_empty();
    let result = if tool_mode {
        match run_generation(
            engine,
            &prepared.generation,
            prepared.resume_state.as_deref(),
            Some(cancellation),
            None,
        ) {
            Ok(result) => result,
            Err(error) => {
                write_failure_events(
                    stream,
                    &mut sequence,
                    &identity,
                    model_name,
                    &prepared,
                    &error,
                    Vec::new(),
                    Value::Null,
                );
                return Ok(());
            }
        }
    } else {
        let output_index = 0_usize;
        let item_id = identity.message_id(output_index);
        if !write_event(
            stream,
            &mut sequence,
            json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": message_started_item(&item_id)
            }),
        ) || !write_event(
            stream,
            &mut sequence,
            json!({
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": output_text_part("")
            }),
        ) {
            return Ok(());
        }
        let result = {
            let mut write_text = |text: &str| {
                write_event(
                    stream,
                    &mut sequence,
                    json!({
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": []
                    }),
                )
            };
            run_generation(
                engine,
                &prepared.generation,
                prepared.resume_state.as_deref(),
                Some(cancellation),
                Some(&mut write_text),
            )
        };
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                write_failure_events(
                    stream,
                    &mut sequence,
                    &identity,
                    model_name,
                    &prepared,
                    &error,
                    Vec::new(),
                    Value::Null,
                );
                return Ok(());
            }
        };
        if result.callback_stopped {
            return Ok(());
        }
        let text = &result.output.content;
        let status = final_status(&result);
        if !write_event(
            stream,
            &mut sequence,
            json!({
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
                "logprobs": []
            }),
        ) || !write_event(
            stream,
            &mut sequence,
            json!({
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": output_text_part(text)
            }),
        ) || !write_event(
            stream,
            &mut sequence,
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": message_item(&item_id, status, text)
            }),
        ) {
            return Ok(());
        }
        result
    };

    let status = final_status(&result);
    let output = output_items(&identity, &result.output, status);
    if tool_mode {
        for event in buffered_output_events(&identity, &result.output, status) {
            if !write_event(stream, &mut sequence, event) {
                return Ok(());
            }
        }
    }
    let response = response_value(
        &identity,
        model_name,
        &prepared,
        status,
        output,
        usage_value(&result),
    );
    let stored_output = output_items(&identity, &result.output, status);
    let sequence_state =
        capture_sequence_state(store, engine, &prepared, &result, &response, &stored_output);
    if cancellation.is_cancelled() {
        return Err(ApiError::cancelled());
    }
    if let Err(error) = store.commit(
        &prepared.history_items,
        prepared.conversation_id.as_deref(),
        prepared.store,
        response.clone(),
        &stored_output,
        sequence_state,
        unix_timestamp(),
    ) {
        write_failure_events(
            stream,
            &mut sequence,
            &identity,
            model_name,
            &prepared,
            &error,
            stored_output.clone(),
            usage_value(&result),
        );
        return Ok(());
    }
    let event_type = if status == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    let _ = write_event(
        stream,
        &mut sequence,
        json!({"type": event_type, "response": response}),
    );
    Ok(())
}

fn capture_sequence_state(
    store: &ResponseStore,
    engine: &mut Engine,
    prepared: &PreparedResponseRequest,
    result: &GeneratedCompletion,
    response: &Value,
    output_items: &[Value],
) -> Option<EngineSequenceState> {
    if !engine.durable_sequence_state_supported() {
        return None;
    }
    let snapshot_copies =
        usize::from(prepared.store) + usize::from(prepared.conversation_id.is_some());
    if snapshot_copies == 0 {
        return None;
    }
    let estimated_bytes = engine.sequence_state_byte_size_estimate();
    if !store.snapshot_fits(
        &prepared.history_items,
        prepared.conversation_id.as_deref(),
        prepared.store,
        response,
        output_items,
        estimated_bytes,
    ) {
        eprintln!("[WARN] response KV snapshot skipped: estimated state exceeds session budget");
        return None;
    }
    let mut token_ids = if result.prompt_token_ids.is_empty() {
        let mut token_ids = Vec::new();
        if engine.tokenizer.should_add_bos() {
            token_ids.push(engine.tokenizer.vocab.special.bos);
        }
        token_ids.extend(engine.tokenizer.encode(&prepared.generation.prompt));
        token_ids
    } else {
        result.prompt_token_ids.clone()
    };
    token_ids.extend_from_slice(&result.generated_token_ids);
    let prompt_alignment = prepared
        .generation
        .response_history_affixes
        .as_ref()
        .filter(|_| result.output.tool_calls.is_empty() && !result.output.content.is_empty())
        .map(|(prefix, suffix)| {
            let mut prompt_prefix =
                String::with_capacity(prefix.len() + result.output.content.len() + suffix.len());
            prompt_prefix.push_str(prefix);
            prompt_prefix.push_str(&result.output.content);
            prompt_prefix.push_str(suffix);
            (prompt_prefix, suffix.clone())
        });
    let captured = match prompt_alignment {
        Some((prompt_prefix, append_text)) => engine.capture_sequence_state_with_prompt_alignment(
            token_ids,
            prompt_prefix,
            append_text,
        ),
        None => engine.capture_sequence_state(token_ids),
    };
    match captured {
        Ok(state) => Some(state),
        Err(error) => {
            eprintln!("[WARN] response KV snapshot skipped: {error}");
            None
        }
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn response_value(
    identity: &ResponseIdentity,
    model_name: &str,
    prepared: &PreparedResponseRequest,
    status: &str,
    output: Vec<Value>,
    usage: Value,
) -> Value {
    let completed_at = if status == "completed" {
        json!(unix_timestamp())
    } else {
        Value::Null
    };
    let incomplete_details = if status == "incomplete" {
        json!({"reason": "max_output_tokens"})
    } else {
        Value::Null
    };
    json!({
        "id": identity.id,
        "object": "response",
        "created_at": identity.created_at,
        "completed_at": completed_at,
        "status": status,
        "background": false,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": prepared.instructions,
        "max_output_tokens": prepared.generation.params.max_tokens,
        "max_tool_calls": null,
        "metadata": prepared.metadata,
        "model": model_name,
        "output": output,
        "parallel_tool_calls": prepared.generation.parallel_tool_calls,
        "previous_response_id": prepared.previous_response_id,
        "conversation": prepared
            .conversation_id
            .as_ref()
            .map(|id| json!({"id": id})),
        "prompt": null,
        "reasoning": null,
        "service_tier": prepared.service_tier,
        "store": prepared.store,
        "temperature": prepared.generation.params.temperature,
        "text": prepared.text,
        "tool_choice": prepared.tool_choice,
        "tools": prepared.tools,
        "top_p": prepared.generation.params.top_p,
        "truncation": "disabled",
        "usage": usage,
        "user": prepared.user,
        "safety_identifier": prepared.safety_identifier
    })
}

fn failed_response_value(
    identity: &ResponseIdentity,
    model_name: &str,
    prepared: &PreparedResponseRequest,
    error: &ApiError,
    output: Vec<Value>,
    usage: Value,
) -> Value {
    let mut response = response_value(identity, model_name, prepared, "failed", output, usage);
    response["error"] = json!({
        "code": "server_error",
        "message": error.message
    });
    response
}

fn output_items(
    identity: &ResponseIdentity,
    output: &ParsedAssistantOutput,
    status: &str,
) -> Vec<Value> {
    let mut items = Vec::new();
    if !output.content.is_empty() || output.tool_calls.is_empty() {
        let item_id = identity.message_id(items.len());
        items.push(message_item(&item_id, status, &output.content));
    }
    for call in &output.tool_calls {
        let index = items.len();
        items.push(function_item(identity, index, call, status));
    }
    items
}

fn buffered_output_events(
    identity: &ResponseIdentity,
    output: &ParsedAssistantOutput,
    status: &str,
) -> Vec<Value> {
    let items = output_items(identity, output, status);
    let mut events = Vec::new();
    for (output_index, item) in items.into_iter().enumerate() {
        if item["type"] == "message" {
            let item_id = item["id"].as_str().unwrap_or_default();
            let text = item["content"][0]["text"].as_str().unwrap_or_default();
            events.push(json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": message_started_item(item_id)
            }));
            events.push(json!({
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": output_text_part("")
            }));
            if !text.is_empty() {
                events.push(json!({
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": []
                }));
            }
            events.push(json!({
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
                "logprobs": []
            }));
            events.push(json!({
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": output_text_part(text)
            }));
            events.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
            }));
        } else {
            let item_id = item["id"].as_str().unwrap_or_default();
            let name = item["name"].as_str().unwrap_or_default();
            let arguments = item["arguments"].as_str().unwrap_or_default();
            let mut started = item.clone();
            started["arguments"] = json!("");
            started["status"] = json!("in_progress");
            events.push(json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": started
            }));
            if !arguments.is_empty() {
                events.push(json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": arguments
                }));
            }
            events.push(json!({
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": output_index,
                "name": name,
                "arguments": arguments
            }));
            events.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
            }));
        }
    }
    events
}

fn message_started_item(id: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": []
    })
}

fn message_item(id: &str, status: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [output_text_part(text)]
    })
}

fn output_text_part(text: &str) -> Value {
    json!({"type": "output_text", "annotations": [], "text": text, "logprobs": []})
}

fn function_item(
    identity: &ResponseIdentity,
    index: usize,
    call: &ParsedToolCall,
    status: &str,
) -> Value {
    json!({
        "id": identity.function_id(index),
        "type": "function_call",
        "status": status,
        "call_id": identity.call_id(index),
        "name": call.name,
        "arguments": call.arguments
    })
}

fn usage_value(result: &GeneratedCompletion) -> Value {
    json!({
        "input_tokens": result.prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": result.cached_prompt_tokens,
            "cache_write_tokens": 0
        },
        "output_tokens": result.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": result.prompt_tokens + result.output_tokens
    })
}

fn final_status(result: &GeneratedCompletion) -> &'static str {
    if result.incomplete() {
        "incomplete"
    } else {
        "completed"
    }
}

fn write_event(stream: &mut TcpStream, sequence: &mut u64, mut event: Value) -> bool {
    event["sequence_number"] = json!(*sequence);
    *sequence += 1;
    let event_type = event["type"].as_str().unwrap_or("error");
    write_sse_event(stream, event_type, &event).is_ok()
}

fn write_failure_events(
    stream: &mut TcpStream,
    sequence: &mut u64,
    identity: &ResponseIdentity,
    model_name: &str,
    prepared: &PreparedResponseRequest,
    error: &ApiError,
    output: Vec<Value>,
    usage: Value,
) {
    if !write_event(
        stream,
        sequence,
        json!({
            "type": "error",
            "code": error.code,
            "message": error.message,
            "param": error.param
        }),
    ) {
        return;
    }
    let response = failed_response_value(identity, model_name, prepared, error, output, usage);
    let _ = write_event(
        stream,
        sequence,
        json!({
            "type": "response.failed",
            "response": response
        }),
    );
}

impl ResponseIdentity {
    fn new() -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let sequence = NEXT_RESPONSE_ID.fetch_add(1, Ordering::Relaxed);
        let suffix = format!("{created_at}_{sequence}");
        Self {
            id: format!("resp_{suffix}"),
            suffix,
            created_at,
        }
    }

    fn message_id(&self, index: usize) -> String {
        format!("msg_{}_{}", self.suffix, index + 1)
    }

    fn function_id(&self, index: usize) -> String {
        format!("fc_{}_{}", self.suffix, index + 1)
    }

    fn call_id(&self, index: usize) -> String {
        format!("call_{}_{}", self.suffix, index + 1)
    }
}

#[cfg(test)]
#[path = "response_output_tests.rs"]
mod tests;
