use rnb_llm::{
    generate_stream_multimodal, generate_stream_multimodal_resuming,
    parse_assistant_output_with_format, AssistantTurnStreamFilter, ChatMessage,
    ChatTemplateOptions, Engine, EngineLoadConfig, EngineSequenceState, GenerateParams,
    GenerateResult, RgbImage, ToolCallFormat,
};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct ChatConfig {
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    image_path: Option<PathBuf>,
    ram_budget_bytes: Option<u64>,
    system_prompt: Option<String>,
    params: GenerateParams,
    enable_thinking: bool,
}

enum ParsedArgs {
    Help,
    Run(ChatConfig),
}

#[derive(Debug, PartialEq, Eq)]
enum InputAction {
    Prompt(String),
    Exit,
    Clear,
    Help,
    ShowSystem,
    SetSystem(String),
}

#[derive(Debug)]
struct ChatHistory {
    system_prompt: Option<String>,
    messages: Vec<ChatMessage>,
    has_image_message: bool,
}

impl ChatHistory {
    fn new(system_prompt: Option<String>) -> Self {
        let mut history = Self {
            system_prompt,
            messages: Vec::new(),
            has_image_message: false,
        };
        history.clear();
        history
    }

    fn clear(&mut self) {
        self.messages.clear();
        self.has_image_message = false;
        if let Some(system_prompt) = self
            .system_prompt
            .as_deref()
            .filter(|prompt| !prompt.is_empty())
        {
            self.messages
                .push(ChatMessage::new("system", system_prompt));
        }
    }

    fn set_system(&mut self, system_prompt: String) {
        self.system_prompt = (!system_prompt.is_empty()).then_some(system_prompt);
        self.clear();
    }

    fn push(&mut self, role: &str, content: String) {
        self.messages.push(ChatMessage::new(role, content));
    }

    fn push_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    fn push_user(&mut self, content: String, attach_image: bool) {
        if attach_image {
            self.messages.push(ChatMessage::with_image("user", content));
            self.has_image_message = true;
        } else {
            self.push("user", content);
        }
    }
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    let config = match parse_args(args)? {
        ParsedArgs::Help => {
            print_help(io::stdout()).map_err(|error| error.to_string())?;
            return Ok(());
        }
        ParsedArgs::Run(config) => config,
    };

    eprintln!("Loading model from {}...", config.model_path.display());
    eprintln!(
        "Runtime backends: {}",
        super::runtime_boundary::compiled_runtime_backends().join(",")
    );

    let mut load_config = EngineLoadConfig::default();
    if let Some(bytes) = config.ram_budget_bytes {
        load_config = load_config.with_host_ram_budget_bytes(bytes);
    }
    if let Some(path) = config.mmproj_path.as_ref() {
        load_config = load_config.with_vision_projector(path);
    }
    let mut engine = Engine::from_gguf_with_config(&config.model_path, load_config)
        .map_err(|error| format!("failed to load model: {error}"))?;
    let image = config
        .image_path
        .as_ref()
        .map(|path| load_image(path.as_path()))
        .transpose()?;

    eprintln!("Model loaded. Type /help for commands.");
    let stdin = io::stdin();
    let interactive = stdin.is_terminal();
    let mut stdout = io::stdout().lock();
    run_session(
        &mut engine,
        &config,
        image.as_ref(),
        stdin.lock(),
        &mut stdout,
        interactive,
    )
}

fn run_session(
    engine: &mut Engine,
    config: &ChatConfig,
    image: Option<&RgbImage>,
    mut input: impl BufRead,
    output: &mut impl Write,
    interactive: bool,
) -> Result<(), String> {
    let mut history = ChatHistory::new(config.system_prompt.clone());
    let mut line = String::new();
    let mut sequence_state: Option<EngineSequenceState> = None;

    loop {
        if interactive {
            write!(output, ">>> ").map_err(|error| error.to_string())?;
            output.flush().map_err(|error| error.to_string())?;
        }

        line.clear();
        if input
            .read_line(&mut line)
            .map_err(|error| format!("failed to read input: {error}"))?
            == 0
        {
            if interactive {
                writeln!(output).map_err(|error| error.to_string())?;
            }
            return Ok(());
        }

        let action = match parse_input(&line) {
            Ok(action) => action,
            Err(message) => {
                writeln!(output, "{message}").map_err(|error| error.to_string())?;
                continue;
            }
        };
        match action {
            InputAction::Exit => return Ok(()),
            InputAction::Clear => {
                history.clear();
                sequence_state = None;
                engine
                    .clear_sequence_state()
                    .map_err(|error| format!("failed to clear sequence state: {error}"))?;
                writeln!(output, "Conversation cleared.").map_err(|error| error.to_string())?;
            }
            InputAction::Help => {
                print_session_help(&mut *output).map_err(|error| error.to_string())?;
            }
            InputAction::ShowSystem => {
                writeln!(
                    output,
                    "System: {}",
                    history.system_prompt.as_deref().unwrap_or("(not set)")
                )
                .map_err(|error| error.to_string())?;
            }
            InputAction::SetSystem(system_prompt) => {
                history.set_system(system_prompt);
                sequence_state = None;
                engine
                    .clear_sequence_state()
                    .map_err(|error| format!("failed to clear sequence state: {error}"))?;
                writeln!(output, "System prompt updated; conversation cleared.")
                    .map_err(|error| error.to_string())?;
            }
            InputAction::Prompt(prompt) => {
                history.push_user(prompt, image.is_some() && !history.has_image_message);
                let rendered = engine
                    .tokenizer
                    .render_chat_prompt(
                        &history.messages,
                        ChatTemplateOptions {
                            add_generation_prompt: true,
                            enable_thinking: config.enable_thinking,
                        },
                    )
                    .map_err(|error| format!("failed to render chat prompt: {error}"))?;

                let muse_protocol = engine.tool_call_format() == ToolCallFormat::Muse;
                let mut turn_filter = AssistantTurnStreamFilter::new(muse_protocol);
                let mut on_piece = |piece: &str| {
                    turn_filter.push(piece, |visible| {
                        if write!(output, "{visible}").is_err() || output.flush().is_err() {
                            return false;
                        }
                        true
                    })
                };
                let result = match (image, sequence_state.as_ref()) {
                    (Some(image), Some(state)) => generate_stream_multimodal_resuming(
                        engine,
                        &rendered,
                        image,
                        &config.params,
                        state,
                        &mut on_piece,
                    ),
                    (Some(image), None) => generate_stream_multimodal(
                        engine,
                        &rendered,
                        image,
                        &config.params,
                        &mut on_piece,
                    ),
                    (None, Some(state)) => engine.generate_stream_resuming(
                        &rendered,
                        &config.params,
                        state,
                        &mut on_piece,
                    ),
                    (None, None) => {
                        engine.generate_stream(&rendered, &config.params, &mut on_piece)
                    }
                }
                .map_err(|error| format!("generation failed: {error}"))?;
                turn_filter.finish(|visible| {
                    if write!(output, "{visible}").is_err() || output.flush().is_err() {
                        return false;
                    }
                    true
                });
                writeln!(output).map_err(|error| error.to_string())?;
                let parsed = parse_assistant_output_with_format(
                    &result.text,
                    &[],
                    engine.tool_call_format(),
                )
                .map_err(|error| format!("failed to parse assistant output: {error}"))?;
                let assistant = ChatMessage::assistant(parsed.content, parsed.reasoning_content);
                sequence_state = capture_chat_sequence_state(
                    engine,
                    &history.messages,
                    &rendered,
                    &result,
                    assistant.clone(),
                    config.enable_thinking,
                )?;
                history.push_message(assistant);
            }
        }
    }
}

fn capture_chat_sequence_state(
    engine: &mut Engine,
    messages_before_assistant: &[ChatMessage],
    rendered_prompt: &str,
    result: &GenerateResult,
    assistant_message: ChatMessage,
    enable_thinking: bool,
) -> Result<Option<EngineSequenceState>, String> {
    if !engine.durable_sequence_state_supported() {
        return Ok(None);
    }
    let (prompt_prefix, append_text) = crate::chat_alignment::render_chat_resume_alignment_message(
        &engine.tokenizer,
        messages_before_assistant,
        assistant_message,
        ChatTemplateOptions {
            add_generation_prompt: false,
            enable_thinking,
        },
        &[],
    )
    .map_err(|error| format!("failed to align chat continuation: {error}"))?;
    let mut token_ids = if result.prompt_token_ids.is_empty() {
        let mut token_ids = Vec::new();
        if engine.tokenizer.should_add_bos() {
            token_ids.push(engine.tokenizer.vocab.special.bos);
        }
        token_ids.extend(engine.tokenizer.encode(rendered_prompt));
        token_ids
    } else {
        result.prompt_token_ids.clone()
    };
    token_ids.extend_from_slice(&result.generated_token_ids);
    engine
        .capture_sequence_state_with_prompt_alignment(token_ids, prompt_prefix, append_text)
        .map(Some)
        .map_err(|error| format!("failed to capture chat sequence state: {error}"))
}

fn load_image(path: &std::path::Path) -> Result<RgbImage, String> {
    let decoded = image::ImageReader::open(path)
        .map_err(|error| format!("failed to open image {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("failed to decode image {}: {error}", path.display()))?
        .into_rgb8();
    RgbImage::new(
        decoded.width() as usize,
        decoded.height() as usize,
        decoded.into_raw(),
    )
    .map_err(|error| format!("invalid image {}: {error}", path.display()))
}
fn parse_input(line: &str) -> Result<InputAction, String> {
    let input = line.trim();
    if input.is_empty() {
        return Err("Enter a message or type /help.".to_string());
    }
    match input {
        "/bye" | "/exit" | "/quit" => Ok(InputAction::Exit),
        "/clear" => Ok(InputAction::Clear),
        "/help" | "/?" => Ok(InputAction::Help),
        "/show" | "/show system" => Ok(InputAction::ShowSystem),
        _ => {
            if let Some(system_prompt) = input.strip_prefix("/set system ") {
                let system_prompt = system_prompt.trim();
                if system_prompt.is_empty() {
                    return Err("Usage: /set system <prompt>".to_string());
                }
                return Ok(InputAction::SetSystem(system_prompt.to_string()));
            }
            if input == "/set system" {
                return Err("Usage: /set system <prompt>".to_string());
            }
            if input.starts_with('/') {
                return Err(format!("Unknown command: {input}. Type /help."));
            }
            Ok(InputAction::Prompt(input.to_string()))
        }
    }
}

fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut model_path = None;
    let mut mmproj_path = None;
    let mut image_path = None;
    let mut ram_budget_bytes = None;
    let mut system_prompt = None;
    let mut params = GenerateParams::default();
    let mut enable_thinking = false;
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];
        if matches!(argument.as_str(), "-h" | "--help") {
            return Ok(ParsedArgs::Help);
        }
        if argument == "--thinking" {
            enable_thinking = true;
            index += 1;
            continue;
        }

        let (name, inline_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(name, value)| {
                (name, Some(value))
            });
        let option = matches!(
            name,
            "--ram-budget"
                | "--mmproj"
                | "--image"
                | "--system"
                | "--max-tokens"
                | "--temperature"
                | "--top-p"
                | "--top-k"
                | "--seed"
        );
        if option {
            let value = if let Some(value) = inline_value {
                value
            } else {
                index += 1;
                args.get(index)
                    .map(String::as_str)
                    .ok_or_else(|| format!("missing value for {name}"))?
            };
            match name {
                "--ram-budget" => {
                    ram_budget_bytes = Some(
                        super::parse_byte_size(value)
                            .map_err(|message| format!("invalid --ram-budget: {message}"))?,
                    );
                }
                "--mmproj" => mmproj_path = Some(PathBuf::from(value)),
                "--image" => image_path = Some(PathBuf::from(value)),
                "--system" => system_prompt = Some(value.to_string()),
                "--max-tokens" => {
                    params.max_tokens = value
                        .parse::<usize>()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| "--max-tokens must be greater than zero".to_string())?;
                }
                "--temperature" => {
                    params.temperature = parse_f32_range(value, "--temperature", 0.0, 2.0)?;
                }
                "--top-p" => {
                    params.top_p = parse_f32_range(value, "--top-p", 0.0, 1.0)?;
                }
                "--top-k" => {
                    params.top_k = value
                        .parse::<usize>()
                        .map_err(|_| "--top-k must be a non-negative integer".to_string())?;
                }
                "--seed" => {
                    params.seed = Some(
                        value
                            .parse::<u64>()
                            .map_err(|_| "--seed must be a non-negative integer".to_string())?,
                    );
                }
                _ => unreachable!(),
            }
            index += 1;
            continue;
        }
        if argument.starts_with('-') {
            return Err(format!("unknown chat option: {argument}"));
        }
        if model_path.replace(PathBuf::from(argument)).is_some() {
            return Err(format!("unexpected chat argument: {argument}"));
        }
        index += 1;
    }

    let model_path = model_path.ok_or_else(|| "missing GGUF model path".to_string())?;
    if model_path.extension().and_then(|value| value.to_str()) != Some("gguf") {
        return Err("chat requires a GGUF model path".to_string());
    }
    if image_path.is_some() != mmproj_path.is_some() {
        return Err("--image and --mmproj must be specified together".to_string());
    }
    if mmproj_path
        .as_ref()
        .is_some_and(|path| path.extension().and_then(|value| value.to_str()) != Some("gguf"))
    {
        return Err("--mmproj requires a GGUF projector path".to_string());
    }
    Ok(ParsedArgs::Run(ChatConfig {
        model_path,
        mmproj_path,
        image_path,
        ram_budget_bytes,
        system_prompt,
        params,
        enable_thinking,
    }))
}

fn parse_f32_range(raw: &str, option: &str, min: f32, max: f32) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| format!("{option} must be a number between {min} and {max}"))?;
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{option} must be between {min} and {max}"));
    }
    Ok(value)
}

fn print_help(mut output: impl Write) -> io::Result<()> {
    writeln!(output, "Usage:")?;
    writeln!(output, "  runNburn chat [options] <model.gguf>")?;
    writeln!(output)?;
    writeln!(output, "Options:")?;
    writeln!(
        output,
        "  --ram-budget <size>   Host RAM budget, for example 8GiB"
    )?;
    writeln!(
        output,
        "  --mmproj <path>       Vision projector GGUF used with --image"
    )?;
    writeln!(
        output,
        "  --image <path>        Local PNG or JPEG included in the conversation"
    )?;
    writeln!(
        output,
        "  --system <prompt>     Set the initial system prompt"
    )?;
    writeln!(
        output,
        "  --max-tokens <count>  Maximum tokens per response (default: 256)"
    )?;
    writeln!(
        output,
        "  --temperature <n>     Sampling temperature from 0 to 2"
    )?;
    writeln!(
        output,
        "  --top-p <n>           Nucleus sampling probability from 0 to 1"
    )?;
    writeln!(
        output,
        "  --top-k <count>       Top-k sampling; 0 disables it"
    )?;
    writeln!(
        output,
        "  --seed <n>            Deterministic sampling seed"
    )?;
    writeln!(
        output,
        "  --thinking            Enable model thinking when the template supports it"
    )?;
    writeln!(output, "  -h, --help            Show this help")?;
    writeln!(output)?;
    print_session_help(output)
}

fn print_session_help(mut output: impl Write) -> io::Result<()> {
    writeln!(output, "Chat commands:")?;
    writeln!(output, "  /clear                Clear conversation history")?;
    writeln!(
        output,
        "  /set system <prompt>  Replace the system prompt and clear history"
    )?;
    writeln!(
        output,
        "  /show system          Show the current system prompt"
    )?;
    writeln!(output, "  /bye                  Exit chat")?;
    writeln!(output, "  /help                 Show chat commands")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_llm::engine::ModelMetadata;
    use rnb_llm::tokenizer::vocab::{SpecialTokens, Vocab};

    fn mock_engine() -> Engine {
        let vocab = Vocab::new(
            (0..16).map(|index| format!("t{index}")).collect(),
            SpecialTokens {
                bos: 1,
                eos: 2,
                pad: None,
            },
        );
        let mut tokenizer = rnb_llm::Tokenizer::new_sentencepiece_with_config(
            vocab,
            Vec::new(),
            Vec::new(),
            false,
            true,
        );
        tokenizer.set_chat_template(Some(
            "{% for message in messages %}<{{ message.role }}>{{ message.content | trim }}</{{ message.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
                .to_string(),
        ));
        Engine::mock(
            tokenizer,
            ModelMetadata {
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                vocab_size: 16,
                max_seq_len: 64,
                hidden_dim: 8,
                rope_theta: 10_000.0,
                rope_theta_swa: 10_000.0,
                rope_dim: 0,
                rope_dim_swa: 0,
                rope_sections: [0; 4],
                norm_eps: 1e-5,
                post_norm_eps: 1e-5,
                logit_scale: 1.0,
                final_logit_softcapping: 0.0,
                query_pre_attn_scalar: 256.0,
                sliding_window: 0,
                shared_kv_layers: 0,
                sliding_window_pattern: vec![],
                key_length_full: 0,
                key_length_swa: 0,
                value_length_swa: 0,
                head_count_kv_per_layer: None,
                embedding_length_per_layer_input: 0,
                expert_used_count: 0,
                expert_weights_scale: 1.0,
                ssm_d_inner: 0,
                ssm_d_state: 0,
                ssm_n_group: 0,
                ssm_dt_rank: 0,
                ssm_conv_kernel: 0,
                full_attention_interval: 0,
            },
        )
    }

    fn chat_config(max_tokens: usize) -> ChatConfig {
        ChatConfig {
            model_path: PathBuf::from("model.gguf"),
            mmproj_path: None,
            image_path: None,
            ram_budget_bytes: None,
            system_prompt: None,
            params: GenerateParams {
                max_tokens,
                temperature: 0.0,
                ..GenerateParams::default()
            },
            enable_thinking: false,
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parses_chat_options_and_requires_gguf() {
        let ParsedArgs::Run(config) = parse_args(&strings(&[
            "--ram-budget=8GiB",
            "--system",
            "Be concise.",
            "--max-tokens",
            "64",
            "--temperature=0.25",
            "--top-p",
            "0.9",
            "--top-k",
            "20",
            "--seed",
            "7",
            "--thinking",
            "--mmproj",
            "projector.gguf",
            "--image",
            "picture.png",
            "model.gguf",
        ]))
        .unwrap() else {
            panic!("expected runnable chat config");
        };
        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert_eq!(
            config.mmproj_path.as_deref(),
            Some(std::path::Path::new("projector.gguf"))
        );
        assert_eq!(
            config.image_path.as_deref(),
            Some(std::path::Path::new("picture.png"))
        );
        assert_eq!(config.ram_budget_bytes, Some(8_u64 << 30));
        assert_eq!(config.system_prompt.as_deref(), Some("Be concise."));
        assert_eq!(config.params.max_tokens, 64);
        assert_eq!(config.params.temperature, 0.25);
        assert_eq!(config.params.top_p, 0.9);
        assert_eq!(config.params.top_k, 20);
        assert_eq!(config.params.seed, Some(7));
        assert!(config.enable_thinking);
        assert!(parse_args(&strings(&["model.rnb"])).is_err());
        assert!(parse_args(&strings(&["--image", "picture.png", "model.gguf"])).is_err());
        assert!(parse_args(&strings(&["--mmproj", "projector.gguf", "model.gguf"])).is_err());
    }

    #[test]
    fn parses_ollama_style_session_commands() {
        assert_eq!(parse_input("/bye\n").unwrap(), InputAction::Exit);
        assert_eq!(parse_input("/clear").unwrap(), InputAction::Clear);
        assert_eq!(
            parse_input("/show system").unwrap(),
            InputAction::ShowSystem
        );
        assert_eq!(
            parse_input("/set system Answer briefly.").unwrap(),
            InputAction::SetSystem("Answer briefly.".to_string())
        );
        assert_eq!(
            parse_input("What is Rust?").unwrap(),
            InputAction::Prompt("What is Rust?".to_string())
        );
        assert!(parse_input("/unknown").is_err());
    }

    #[test]
    fn clearing_history_preserves_the_system_prompt() {
        let mut history = ChatHistory::new(Some("Be concise.".to_string()));
        history.push("user", "Hello".to_string());
        history.push("assistant", "Hi".to_string());
        history.push_user("Look".to_string(), true);
        assert!(history.has_image_message);
        assert!(matches!(
            history.messages[3].content,
            Some(rnb_llm::ChatContent::Parts(_))
        ));
        history.clear();
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0].role, "system");
        assert_eq!(
            history.messages[0].content,
            Some(rnb_llm::ChatContent::Text("Be concise.".to_string()))
        );

        history.set_system("Use Korean.".to_string());
        assert_eq!(history.messages.len(), 1);
        assert_eq!(
            history.messages[0].content,
            Some(rnb_llm::ChatContent::Text("Use Korean.".to_string()))
        );
    }

    #[test]
    fn captured_chat_state_resumes_only_the_appended_turn() {
        let mut engine = mock_engine();
        let params = chat_config(1).params;
        let first_messages = vec![ChatMessage::new("user", "first")];
        let first_prompt = engine
            .tokenizer
            .render_chat_prompt(&first_messages, ChatTemplateOptions::default())
            .unwrap();
        let first = engine
            .generate_stream(&first_prompt, &params, |_| true)
            .unwrap();
        engine.kv_cache.append(0, 0, &[0.0, 0.0], &[0.0, 0.0]);
        let first_assistant = ChatMessage::assistant(first.text.clone(), None);
        let state = capture_chat_sequence_state(
            &mut engine,
            &first_messages,
            &first_prompt,
            &first,
            first_assistant,
            false,
        )
        .unwrap()
        .unwrap();

        let second_messages = vec![
            ChatMessage::new("user", "first"),
            ChatMessage::new("assistant", first.text.clone()),
            ChatMessage::new("user", "second"),
        ];
        let second_prompt = engine
            .tokenizer
            .render_chat_prompt(&second_messages, ChatTemplateOptions::default())
            .unwrap();
        let second = engine
            .generate_stream_resuming(&second_prompt, &params, &state, |_| true)
            .unwrap();

        assert_eq!(second.cached_prompt_tokens, state.token_len());
        assert!(second.cached_prompt_tokens > 0);
    }

    #[test]
    fn clear_command_releases_cached_sequence_state() {
        let mut engine = mock_engine();
        engine.kv_cache.append(0, 0, &[0.0, 0.0], &[0.0, 0.0]);
        let config = chat_config(1);
        let mut output = Vec::new();

        run_session(
            &mut engine,
            &config,
            None,
            std::io::Cursor::new(b"/clear\n/bye\n"),
            &mut output,
            false,
        )
        .unwrap();

        assert_eq!(engine.kv_cache.current_len(), 0);
        assert!(String::from_utf8(output)
            .unwrap()
            .contains("Conversation cleared."));
    }
}
