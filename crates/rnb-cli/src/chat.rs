use rnb_llm::{ChatMessage, ChatTemplateOptions, Engine, EngineLoadConfig, GenerateParams};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct ChatConfig {
    model_path: PathBuf,
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
}

impl ChatHistory {
    fn new(system_prompt: Option<String>) -> Self {
        let mut history = Self {
            system_prompt,
            messages: Vec::new(),
        };
        history.clear();
        history
    }

    fn clear(&mut self) {
        self.messages.clear();
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
    let mut engine = Engine::from_gguf_with_config(&config.model_path, load_config)
        .map_err(|error| format!("failed to load model: {error}"))?;

    eprintln!("Model loaded. Type /help for commands.");
    let stdin = io::stdin();
    let interactive = stdin.is_terminal();
    let mut stdout = io::stdout().lock();
    run_session(&mut engine, &config, stdin.lock(), &mut stdout, interactive)
}

fn run_session(
    engine: &mut Engine,
    config: &ChatConfig,
    mut input: impl BufRead,
    output: &mut impl Write,
    interactive: bool,
) -> Result<(), String> {
    let mut history = ChatHistory::new(config.system_prompt.clone());
    let mut line = String::new();

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
                writeln!(output, "System prompt updated; conversation cleared.")
                    .map_err(|error| error.to_string())?;
            }
            InputAction::Prompt(prompt) => {
                history.push("user", prompt);
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

                let result = engine
                    .generate_stream(&rendered, &config.params, |piece| {
                        if write!(output, "{piece}").is_err() || output.flush().is_err() {
                            return false;
                        }
                        true
                    })
                    .map_err(|error| format!("generation failed: {error}"))?;
                writeln!(output).map_err(|error| error.to_string())?;
                history.push("assistant", result.text);
            }
        }
    }
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
    Ok(ParsedArgs::Run(ChatConfig {
        model_path,
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
            "model.gguf",
        ]))
        .unwrap() else {
            panic!("expected runnable chat config");
        };
        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert_eq!(config.ram_budget_bytes, Some(8_u64 << 30));
        assert_eq!(config.system_prompt.as_deref(), Some("Be concise."));
        assert_eq!(config.params.max_tokens, 64);
        assert_eq!(config.params.temperature, 0.25);
        assert_eq!(config.params.top_p, 0.9);
        assert_eq!(config.params.top_k, 20);
        assert_eq!(config.params.seed, Some(7));
        assert!(config.enable_thinking);
        assert!(parse_args(&strings(&["model.rnb"])).is_err());
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
        history.clear();
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0].role, "system");
        assert_eq!(history.messages[0].content.as_deref(), Some("Be concise."));

        history.set_system("Use Korean.".to_string());
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0].content.as_deref(), Some("Use Korean."));
    }
}
