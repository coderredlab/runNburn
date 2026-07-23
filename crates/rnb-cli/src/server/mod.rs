mod generation;
mod http;
mod lifecycle;
mod response_config;
mod response_input;
mod response_output;
mod responses;
mod session_store;
mod structured;
mod types;
mod worker;

#[cfg(test)]
use generation::append_generated_text;
use generation::run_generation;
use http::{
    read_request, write_json_response, write_sse_done, write_sse_headers, write_sse_json, ApiError,
};
use response_output::{complete_response, stream_response};
use responses::ResponseRequest;
use rnb_llm::{GenerationCancellation, ParsedAssistantOutput, ParsedToolCall};
use rnb_runtime::scheduler::FairExecutionSubmitError;
use serde_json::{json, Value};
use session_store::{ConversationCreateRequest, ResponseStore};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use types::{ChatCompletionRequest, PreparedGenerationRequest};
use worker::{RequestCancellation, WorkerRequest, WorkerSender};

static NEXT_COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct ServeOptions {
    host: String,
    port: u16,
    model_path: PathBuf,
    model_name: String,
    api_key: Option<Arc<str>>,
    ram_budget_bytes: Option<u64>,
    response_cache_bytes: Option<u64>,
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_usage();
        return Ok(());
    }
    let options = ServeOptions::parse(args)?;
    let shutdown = lifecycle::install_shutdown_handler()?;
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .map_err(|error| format!("bind {}:{}: {error}", options.host, options.port))?;
    let bound_address = listener
        .local_addr()
        .map_err(|error| format!("read bound server address: {error}"))?;
    validate_bind_auth(bound_address, options.api_key.as_deref())?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("configure nonblocking server listener: {error}"))?;

    eprintln!("Loading model from {}...", options.model_path.display());
    eprintln!(
        "Runtime backends: {}",
        super::runtime_boundary::compiled_runtime_backends().join(",")
    );
    let mut load_config = rnb_llm::EngineLoadConfig::default();
    if let Some(bytes) = options.ram_budget_bytes {
        load_config = load_config.with_host_ram_budget_bytes(bytes);
    }
    let worker = worker::start(
        options.model_path.clone(),
        load_config,
        options.model_name.clone(),
        options.response_cache_bytes,
    )?;
    eprintln!("Model loaded: {}", options.model_name);
    eprintln!("OpenAI-compatible API listening on http://{bound_address}/v1");

    let mut connections = lifecycle::ConnectionThreads::new();
    eprintln!("HTTP connection limit: {}", connections.limit());
    let mut run_result = Ok(());
    while !shutdown.is_cancelled() {
        if let Err(error) = connections.reap_finished() {
            run_result = Err(error);
            break;
        }
        let mut stream = match lifecycle::poll_accept(&listener) {
            Ok(Some(stream)) => stream,
            Ok(None) => continue,
            Err(error) => {
                run_result = Err(format!("accept HTTP connection: {error}"));
                break;
            }
        };
        if connections.is_full() {
            let error = ApiError::overloaded();
            let _ = write_json_response(&mut stream, error.status, &error.body());
            continue;
        }
        let worker = worker.sender();
        let api_key = options.api_key.clone();
        let request_shutdown = shutdown.clone();
        if let Err(error) = connections
            .spawn(move || handle_connection(stream, &worker, api_key.as_deref(), request_shutdown))
        {
            run_result = Err(error);
            break;
        }
    }

    shutdown.cancel();
    let connections_result = connections.join_all();
    let worker_result = worker.shutdown();
    run_result?;
    connections_result?;
    worker_result?;
    eprintln!("OpenAI-compatible API stopped");
    Ok(())
}

impl ServeOptions {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut host = "127.0.0.1".to_string();
        let mut port = 8000_u16;
        let mut model_path = None;
        let mut model_name = None;
        let mut ram_budget_bytes = None;
        let mut response_cache_bytes = None;
        let mut api_key_file = None;
        let mut index = 0;

        while index < args.len() {
            let arg = &args[index];
            if let Some(value) = arg.strip_prefix("--host=") {
                host = nonempty(value, "--host")?.to_string();
            } else if arg == "--host" {
                index += 1;
                host = nonempty(required_value(args, index, "--host")?, "--host")?.to_string();
            } else if let Some(value) = arg.strip_prefix("--port=") {
                port = parse_port(value)?;
            } else if arg == "--port" {
                index += 1;
                port = parse_port(required_value(args, index, "--port")?)?;
            } else if let Some(value) = arg.strip_prefix("--model-name=") {
                model_name = Some(nonempty(value, "--model-name")?.to_string());
            } else if arg == "--model-name" {
                index += 1;
                model_name = Some(
                    nonempty(required_value(args, index, "--model-name")?, "--model-name")?
                        .to_string(),
                );
            } else if let Some(value) = arg.strip_prefix("--api-key-file=") {
                api_key_file = Some(PathBuf::from(nonempty(value, "--api-key-file")?));
            } else if arg == "--api-key-file" {
                index += 1;
                api_key_file = Some(PathBuf::from(nonempty(
                    required_value(args, index, "--api-key-file")?,
                    "--api-key-file",
                )?));
            } else if let Some(value) = arg.strip_prefix("--ram-budget=") {
                ram_budget_bytes = Some(super::parse_byte_size(value)?);
            } else if arg == "--ram-budget" {
                index += 1;
                ram_budget_bytes = Some(super::parse_byte_size(required_value(
                    args,
                    index,
                    "--ram-budget",
                )?)?);
            } else if let Some(value) = arg.strip_prefix("--response-cache-budget=") {
                response_cache_bytes = Some(super::parse_byte_size(value)?);
            } else if arg == "--response-cache-budget" {
                index += 1;
                response_cache_bytes = Some(super::parse_byte_size(required_value(
                    args,
                    index,
                    "--response-cache-budget",
                )?)?);
            } else if arg.starts_with('-') {
                return Err(format!("unknown serve option: {arg}"));
            } else if model_path.is_some() {
                return Err(format!("unexpected serve argument: {arg}"));
            } else {
                model_path = Some(PathBuf::from(arg));
            }
            index += 1;
        }

        let model_path = model_path.ok_or_else(|| {
            "missing GGUF model path; run `runNburn serve --help` for usage".to_string()
        })?;
        if !model_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return Err("serve requires a GGUF model path".to_string());
        }
        let model_name = model_name.unwrap_or_else(|| {
            model_path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("runNburn-model")
                .to_string()
        });
        let api_key = if let Some(path) = api_key_file {
            Some(Arc::<str>::from(read_api_key(&path)?))
        } else {
            std::env::var("RNB_API_KEY")
                .ok()
                .filter(|value| !value.is_empty())
                .map(Arc::<str>::from)
        };

        Ok(Self {
            host,
            port,
            model_path,
            model_name,
            api_key,
            ram_budget_bytes,
            response_cache_bytes,
        })
    }
}

fn required_value<'a>(args: &'a [String], index: usize, option: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{option} requires a value"))
}

fn nonempty<'a>(value: &'a str, option: &str) -> Result<&'a str, String> {
    if value.is_empty() {
        Err(format!("{option} requires a non-empty value"))
    } else {
        Ok(value)
    }
}

fn parse_port(value: &str) -> Result<u16, String> {
    value
        .parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
        .ok_or_else(|| format!("invalid --port: {value}"))
}

fn read_api_key(path: &PathBuf) -> Result<String, String> {
    let value = std::fs::read_to_string(path)
        .map_err(|error| format!("read API key file {}: {error}", path.display()))?;
    let key = value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(&value);
    if key.is_empty() {
        return Err(format!("API key file is empty: {}", path.display()));
    }
    if key.contains('\r') || key.contains('\n') {
        return Err(format!(
            "API key file must contain exactly one line: {}",
            path.display()
        ));
    }
    Ok(key.to_string())
}

fn validate_bind_auth(address: SocketAddr, api_key: Option<&str>) -> Result<(), String> {
    if !address.ip().is_loopback() && api_key.is_none() {
        return Err(format!(
            "non-loopback bind {address} requires --api-key-file or RNB_API_KEY"
        ));
    }
    Ok(())
}

fn print_usage() {
    eprintln!("Usage: runNburn serve [options] <model.gguf>");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --host HOST          Bind host (default: 127.0.0.1)");
    eprintln!("  --port PORT          Bind port (default: 8000)");
    eprintln!("  --model-name NAME    Model ID exposed through /v1/models");
    eprintln!("  --ram-budget SIZE    Host RAM budget, for example 8GiB");
    eprintln!("  --response-cache-budget SIZE  Stored response/KV cache cap");
    eprintln!("  --api-key-file PATH  Read the bearer API key from a one-line file");
    eprintln!("  RNB_API_KEY=KEY      Read the bearer API key from the environment");
    eprintln!("  Non-loopback binds require an API key.");
}

fn handle_connection(
    mut stream: TcpStream,
    worker: &WorkerSender,
    api_key: Option<&str>,
    shutdown: GenerationCancellation,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let request = match read_request(&mut stream, |headers| {
        authorize(headers.get("authorization").map(String::as_str), api_key)
    }) {
        Ok(request) => request,
        Err(error) => {
            let _ = write_json_response(&mut stream, error.status, &error.body());
            return;
        }
    };
    let cancellation = RequestCancellation::monitor(&stream, shutdown);
    let work = WorkerRequest {
        stream,
        request,
        cancellation,
    };
    if let Err(error) = worker.submit(work) {
        let (mut work, api_error) = match error {
            FairExecutionSubmitError::Full(work) => (work, ApiError::overloaded()),
            FairExecutionSubmitError::Disconnected(work) => {
                (work, ApiError::internal("engine worker is not available"))
            }
        };
        let _ = write_json_response(&mut work.stream, api_error.status, &api_error.body());
    }
}

fn handle_worker_request(
    stream: &mut TcpStream,
    engine: &mut rnb_llm::Engine,
    store: &mut ResponseStore,
    cancellation: &GenerationCancellation,
    model_name: &str,
    request: http::HttpRequest,
) -> Result<(), ApiError> {
    let input_items_query =
        request.method == "GET" && response_input_items_id(&request.path).is_some();
    if !input_items_query
        && request
            .query
            .as_deref()
            .is_some_and(|query| !query.is_empty())
    {
        return Err(ApiError::invalid(
            "query parameters are not supported for this endpoint",
            Some("query"),
            Some("unsupported_parameter"),
        ));
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => write_json_response(stream, 200, &model_list(model_name))
            .map_err(|error| ApiError::internal(format!("write response: {error}"))),
        ("POST", "/v1/chat/completions") => {
            let request: ChatCompletionRequest =
                serde_json::from_slice(&request.body).map_err(|error| {
                    ApiError::invalid(
                        format!("Invalid JSON request body: {error}"),
                        None,
                        Some("invalid_json"),
                    )
                })?;
            let prepared = request.prepare(model_name, engine)?;
            if prepared.stream {
                stream_chat_completion(stream, engine, cancellation, model_name, prepared)
            } else {
                complete_chat(stream, engine, cancellation, model_name, prepared)
            }
        }
        ("POST", "/v1/responses") => {
            let mut request: ResponseRequest =
                serde_json::from_slice(&request.body).map_err(|error| {
                    ApiError::invalid(
                        format!("Invalid JSON request body: {error}"),
                        None,
                        Some("invalid_json"),
                    )
                })?;
            let context = store.resolve(&mut request, unix_timestamp())?;
            let prepared = request.prepare(model_name, engine, context)?;
            if prepared.generation.stream {
                stream_response(stream, engine, store, cancellation, model_name, prepared)
            } else {
                complete_response(stream, engine, store, cancellation, model_name, prepared)
            }
        }
        ("POST", "/v1/conversations") => {
            let request: ConversationCreateRequest = serde_json::from_slice(&request.body)
                .map_err(|error| {
                    ApiError::invalid(
                        format!("Invalid JSON request body: {error}"),
                        None,
                        Some("invalid_json"),
                    )
                })?;
            let body = store.create_conversation(request, unix_timestamp())?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("GET", path) if response_input_items_id(path).is_some() => {
            let id = response_input_items_id(path).unwrap();
            let page = parse_input_items_query(request.query.as_deref())?;
            let body = store.get_input_items(
                id,
                unix_timestamp(),
                page.descending,
                page.after.as_deref(),
                page.limit,
            )?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("GET", path) if response_id(path).is_some() => {
            let body = store.get_response(response_id(path).unwrap(), unix_timestamp())?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("DELETE", path) if response_id(path).is_some() => {
            let body = store.delete_response(response_id(path).unwrap())?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("GET", path) if conversation_id(path).is_some() => {
            let body = store.get_conversation(conversation_id(path).unwrap())?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("DELETE", path) if conversation_id(path).is_some() => {
            let body = store.delete_conversation(conversation_id(path).unwrap())?;
            write_json_response(stream, 200, &body)
                .map_err(|error| ApiError::internal(format!("write response: {error}")))
        }
        ("GET", "/v1/chat/completions")
        | ("GET", "/v1/responses")
        | ("GET", "/v1/conversations")
        | ("POST", "/v1/models") => Err(ApiError::method_not_allowed()),
        _ => Err(ApiError::route_not_found()),
    }
}

fn response_id(path: &str) -> Option<&str> {
    path.strip_prefix("/v1/responses/")
        .filter(|id| !id.is_empty() && !id.contains('/'))
}

fn response_input_items_id(path: &str) -> Option<&str> {
    path.strip_prefix("/v1/responses/")
        .and_then(|path| path.strip_suffix("/input_items"))
        .filter(|id| !id.is_empty() && !id.contains('/'))
}

fn conversation_id(path: &str) -> Option<&str> {
    path.strip_prefix("/v1/conversations/")
        .filter(|id| !id.is_empty() && !id.contains('/'))
}

struct InputItemsPage {
    descending: bool,
    after: Option<String>,
    limit: usize,
}

fn parse_input_items_query(query: Option<&str>) -> Result<InputItemsPage, ApiError> {
    let mut page = InputItemsPage {
        descending: true,
        after: None,
        limit: 20,
    };
    for parameter in query.into_iter().flat_map(|query| query.split('&')) {
        if parameter.is_empty() {
            continue;
        }
        let (name, value) = parameter.split_once('=').unwrap_or((parameter, ""));
        match name {
            "order" => match value {
                "asc" => page.descending = false,
                "desc" => page.descending = true,
                _ => {
                    return Err(ApiError::invalid(
                        "order must be 'asc' or 'desc'",
                        Some("order"),
                        Some("invalid_value"),
                    ));
                }
            },
            "after" if !value.is_empty() => page.after = Some(value.to_string()),
            "after" => {
                return Err(ApiError::invalid(
                    "after must be a non-empty item ID",
                    Some("after"),
                    Some("invalid_value"),
                ));
            }
            "limit" => {
                page.limit = value
                    .parse::<usize>()
                    .ok()
                    .filter(|limit| (1..=100).contains(limit))
                    .ok_or_else(|| {
                        ApiError::invalid(
                            "limit must be an integer between 1 and 100",
                            Some("limit"),
                            Some("invalid_value"),
                        )
                    })?;
            }
            "include" | "include[]" | "include%5B%5D" => {
                return Err(ApiError::invalid(
                    "include is not supported by this server",
                    Some("include"),
                    Some("unsupported_parameter"),
                ));
            }
            _ => {
                return Err(ApiError::invalid(
                    format!("unsupported query parameter '{name}'"),
                    Some("query"),
                    Some("invalid_value"),
                ));
            }
        }
    }
    Ok(page)
}

fn authorize(authorization: Option<&str>, api_key: Option<&str>) -> Result<(), ApiError> {
    let Some(expected) = api_key else {
        return Ok(());
    };
    let provided = authorization
        .and_then(|header| header.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, token)| token);
    if provided.is_some_and(|token| constant_time_eq(token.as_bytes(), expected.as_bytes())) {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn model_list(model_name: &str) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": unix_timestamp(),
            "owned_by": "runNburn"
        }]
    })
}

fn complete_chat(
    stream: &mut TcpStream,
    engine: &mut rnb_llm::Engine,
    cancellation: &GenerationCancellation,
    model_name: &str,
    prepared: PreparedGenerationRequest,
) -> Result<(), ApiError> {
    let (id, created) = completion_identity();
    let result = run_generation(engine, &prepared, None, Some(cancellation), None)?;
    let finish_reason = if result.output.tool_calls.is_empty() {
        finish_reason(result.output_tokens, result.max_tokens, result.matched_stop)
    } else {
        "tool_calls"
    };
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": assistant_message(&id, &result.output),
            "finish_reason": finish_reason,
            "logprobs": null
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.prompt_tokens + result.output_tokens
        }
    });
    write_json_response(stream, 200, &body)
        .map_err(|error| ApiError::internal(format!("write response: {error}")))
}

fn stream_chat_completion(
    stream: &mut TcpStream,
    engine: &mut rnb_llm::Engine,
    cancellation: &GenerationCancellation,
    model_name: &str,
    prepared: PreparedGenerationRequest,
) -> Result<(), ApiError> {
    let (id, created) = completion_identity();
    write_sse_headers(stream)
        .map_err(|error| ApiError::internal(format!("write stream headers: {error}")))?;
    let tool_mode = !prepared.tool_names.is_empty();
    if write_sse_json(
        stream,
        &stream_chunk(
            &id,
            created,
            model_name,
            if tool_mode {
                json!({"role": "assistant", "content": null})
            } else {
                json!({"role": "assistant", "content": ""})
            },
            Value::Null,
        ),
    )
    .is_err()
    {
        return Ok(());
    }
    if tool_mode {
        return stream_tool_completion(
            stream,
            engine,
            cancellation,
            model_name,
            prepared,
            &id,
            created,
        );
    }

    let mut write_text = |text: &str| {
        let chunk = stream_chunk(
            &id,
            created,
            model_name,
            json!({"content": text}),
            Value::Null,
        );
        write_sse_json(stream, &chunk).is_ok()
    };
    let result = match run_generation(
        engine,
        &prepared,
        None,
        Some(cancellation),
        Some(&mut write_text),
    ) {
        Ok(result) => result,
        Err(error) => {
            let _ = write_sse_json(stream, &error.body());
            let _ = write_sse_done(stream);
            return Ok(());
        }
    };
    if result.callback_stopped {
        return Ok(());
    }

    let finish_reason = finish_reason(result.output_tokens, result.max_tokens, result.matched_stop);
    let final_chunk = stream_chunk(&id, created, model_name, json!({}), json!(finish_reason));
    if write_sse_json(stream, &final_chunk).is_err() {
        return Ok(());
    }
    if prepared.include_usage {
        let usage = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.prompt_tokens + result.output_tokens
            }
        });
        if write_sse_json(stream, &usage).is_err() {
            return Ok(());
        }
    }
    let _ = write_sse_done(stream);
    Ok(())
}

fn stream_tool_completion(
    stream: &mut TcpStream,
    engine: &mut rnb_llm::Engine,
    cancellation: &GenerationCancellation,
    model_name: &str,
    prepared: PreparedGenerationRequest,
    id: &str,
    created: u64,
) -> Result<(), ApiError> {
    let result = match run_generation(engine, &prepared, None, Some(cancellation), None) {
        Ok(result) => result,
        Err(error) => {
            let _ = write_sse_json(stream, &error.body());
            let _ = write_sse_done(stream);
            return Ok(());
        }
    };

    if !result.output.content.is_empty() {
        let chunk = stream_chunk(
            id,
            created,
            model_name,
            json!({"content": result.output.content}),
            Value::Null,
        );
        if write_sse_json(stream, &chunk).is_err() {
            return Ok(());
        }
    }
    for (index, call) in result.output.tool_calls.iter().enumerate() {
        let chunk = stream_chunk(
            id,
            created,
            model_name,
            tool_call_delta(id, index, call),
            Value::Null,
        );
        if write_sse_json(stream, &chunk).is_err() {
            return Ok(());
        }
    }

    let finish_reason = if result.output.tool_calls.is_empty() {
        finish_reason(result.output_tokens, result.max_tokens, result.matched_stop)
    } else {
        "tool_calls"
    };
    let final_chunk = stream_chunk(id, created, model_name, json!({}), json!(finish_reason));
    if write_sse_json(stream, &final_chunk).is_err() {
        return Ok(());
    }
    if prepared.include_usage {
        let usage = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.prompt_tokens + result.output_tokens
            }
        });
        if write_sse_json(stream, &usage).is_err() {
            return Ok(());
        }
    }
    let _ = write_sse_done(stream);
    Ok(())
}

fn stream_chunk(
    id: &str,
    created: u64,
    model_name: &str,
    delta: Value,
    finish_reason: Value,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": null
        }],
        "usage": null
    })
}

fn assistant_message(completion_id: &str, output: &ParsedAssistantOutput) -> Value {
    if output.tool_calls.is_empty() {
        return json!({
            "role": "assistant",
            "content": output.content
        });
    }
    let content = if output.content.is_empty() {
        Value::Null
    } else {
        json!(output.content)
    };
    let tool_calls = output
        .tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| tool_call_value(completion_id, index, call))
        .collect::<Vec<_>>();
    json!({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
    })
}

fn tool_call_value(completion_id: &str, index: usize, call: &ParsedToolCall) -> Value {
    json!({
        "id": tool_call_id(completion_id, index),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments
        }
    })
}

fn tool_call_delta(completion_id: &str, index: usize, call: &ParsedToolCall) -> Value {
    json!({
        "tool_calls": [{
            "index": index,
            "id": tool_call_id(completion_id, index),
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments
            }
        }]
    })
}

fn tool_call_id(completion_id: &str, index: usize) -> String {
    let suffix = completion_id
        .strip_prefix("chatcmpl-")
        .unwrap_or(completion_id)
        .replace('-', "_");
    format!("call_{suffix}_{}", index + 1)
}

fn finish_reason(tokens_generated: usize, max_tokens: usize, matched_stop: bool) -> &'static str {
    if !matched_stop && tokens_generated >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

fn completion_identity() -> (String, u64) {
    let created = unix_timestamp();
    let sequence = NEXT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed);
    (format!("chatcmpl-{created}-{sequence}"), created)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::thread;

    #[test]
    fn nonblocking_listener_accepts_delayed_request_body() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n",
                )
                .unwrap();
            thread::sleep(Duration::from_millis(250));
            let _ = stream.write_all(b"{}");
        });
        let stream = loop {
            if let Some(stream) = lifecycle::poll_accept(&listener).unwrap() {
                break stream;
            }
        };
        let (sender, receiver) = rnb_runtime::scheduler::fair_execution_queue(1);

        handle_connection(stream, &sender, None, GenerationCancellation::new());
        drop(sender);
        let work = receiver
            .receive()
            .expect("delayed request body should reach the worker queue");

        assert_eq!(work.request.body, b"{}");
        drop(work);
        client.join().unwrap();
    }

    #[test]
    fn parses_serve_options_and_defaults_model_name_to_file_stem() {
        let options = ServeOptions::parse(&[
            "--host=0.0.0.0".to_string(),
            "--port".to_string(),
            "9000".to_string(),
            "--ram-budget=4GiB".to_string(),
            "--response-cache-budget=512MiB".to_string(),
            "/models/gemma.gguf".to_string(),
        ])
        .unwrap();

        assert_eq!(options.host, "0.0.0.0");
        assert_eq!(options.port, 9000);
        assert_eq!(options.model_name, "gemma");
        assert_eq!(options.ram_budget_bytes, Some(4_u64 << 30));
        assert_eq!(options.response_cache_bytes, Some(512_u64 << 20));
    }

    #[test]
    fn rejects_non_gguf_product_input() {
        let error = ServeOptions::parse(&["model.rnb".to_string()]).unwrap_err();
        assert!(error.contains("GGUF"));
    }

    #[test]
    fn bearer_secret_comparison_is_exact() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"Secret"));
        assert!(!constant_time_eq(b"secret", b"secret-long"));
    }

    #[test]
    fn non_loopback_bind_requires_authentication() {
        let loopback = SocketAddr::from(([127, 0, 0, 1], 8000));
        let wildcard = SocketAddr::from(([0, 0, 0, 0], 8000));

        assert!(validate_bind_auth(loopback, None).is_ok());
        assert!(validate_bind_auth(wildcard, Some("secret")).is_ok());
        assert!(validate_bind_auth(wildcard, None).is_err());
    }

    #[test]
    fn api_key_file_option_loads_one_line_secret() {
        let path = std::env::temp_dir().join(format!(
            "rnb-api-key-{}-{}.txt",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"packaged-secret\r\n").unwrap();

        let options = ServeOptions::parse(&[
            "--api-key-file".to_string(),
            path.display().to_string(),
            "/models/gemma.gguf".to_string(),
        ])
        .unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(options.api_key.as_deref(), Some("packaged-secret"));
    }

    #[test]
    fn api_key_file_rejects_multiple_lines() {
        let path = std::env::temp_dir().join(format!(
            "rnb-api-key-multiline-{}-{}.txt",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"first\nsecond\n").unwrap();

        let error = read_api_key(&path).unwrap_err();
        let _ = std::fs::remove_file(path);

        assert!(error.contains("one line"));
    }

    #[test]
    fn parses_input_item_pagination_query() {
        let page = parse_input_items_query(Some("order=asc&limit=2&after=msg_1")).unwrap();
        assert!(!page.descending);
        assert_eq!(page.limit, 2);
        assert_eq!(page.after.as_deref(), Some("msg_1"));
        assert!(parse_input_items_query(Some("order=newest")).is_err());
        assert!(parse_input_items_query(Some("limit=0")).is_err());
        assert!(parse_input_items_query(Some("include=message.input_image.image_url")).is_err());
    }

    #[test]
    fn max_token_exhaustion_maps_to_length_finish_reason() {
        assert_eq!(finish_reason(8, 8, false), "length");
        assert_eq!(finish_reason(3, 8, false), "stop");
        assert_eq!(finish_reason(8, 8, true), "stop");
    }

    #[test]
    fn builds_openai_tool_call_message() {
        let output = ParsedAssistantOutput {
            content: String::new(),
            tool_calls: vec![ParsedToolCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Seoul"}"#.to_string(),
            }],
        };

        let message = assistant_message("chatcmpl-100-2", &output);

        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"][0]["id"], "call_100_2_1");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Seoul"}"#
        );
    }

    #[test]
    fn builds_openai_streaming_tool_call_delta() {
        let call = ParsedToolCall {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"Seoul"}"#.to_string(),
        };

        let delta = tool_call_delta("chatcmpl-100-2", 0, &call);

        assert_eq!(delta["tool_calls"][0]["index"], 0);
        assert_eq!(delta["tool_calls"][0]["id"], "call_100_2_1");
        assert_eq!(delta["tool_calls"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn stops_generation_after_complete_tool_call_marker() {
        let mut content = String::new();

        assert!(!append_generated_text(
            &mut content,
            "<|tool_call>call:get_weather{city:<|\"|>Seoul<|\"|>}<tool_call|><eos>",
            true,
        ));
        assert_eq!(
            content,
            "<|tool_call>call:get_weather{city:<|\"|>Seoul<|\"|>}<tool_call|>"
        );
    }

    #[test]
    fn parallel_tool_mode_keeps_multiple_complete_calls() {
        let mut content = String::new();
        let calls = concat!(
            "<tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"second\",\"arguments\":{}}</tool_call>"
        );

        assert!(append_generated_text(&mut content, calls, false));
        assert_eq!(content, calls);
    }
}
