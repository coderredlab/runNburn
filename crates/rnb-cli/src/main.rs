use std::io::{self, Write};
use std::path::PathBuf;

mod runtime_boundary;
mod server;

fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let split = raw
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(raw.len());
    if split == 0 {
        return Err(format!("invalid size: {raw}"));
    }
    let value = raw[..split]
        .parse::<u64>()
        .map_err(|_| format!("invalid size: {raw}"))?;
    if value == 0 {
        return Err("size must be greater than zero".to_string());
    }
    let multiplier = match raw[split..].to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => 1_000,
        "kib" => 1_u64 << 10,
        "mb" => 1_000_000,
        "mib" => 1_u64 << 20,
        "gb" => 1_000_000_000,
        "gib" => 1_u64 << 30,
        "tb" => 1_000_000_000_000,
        "tib" => 1_u64 << 40,
        _ => return Err(format!("unsupported size suffix: {raw}")),
    };
    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("size is too large: {raw}"))
}

fn print_usage(mut output: impl Write) -> io::Result<()> {
    writeln!(output, "runNburn {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(
        output,
        "Quantized GGUF inference for memory-constrained systems"
    )?;
    writeln!(output)?;
    writeln!(output, "Usage:")?;
    writeln!(
        output,
        "  runNburn [--ram-budget <size>] <model.gguf> [prompt]"
    )?;
    writeln!(output, "  runNburn serve [options] <model.gguf>")?;
    writeln!(output)?;
    writeln!(output, "Options:")?;
    writeln!(
        output,
        "  --ram-budget <size>  Host RAM budget, for example 8GiB"
    )?;
    writeln!(output, "  -h, --help            Show this help")?;
    writeln!(output, "  -V, --version         Show the version")?;
    writeln!(output)?;
    writeln!(
        output,
        "Run `runNburn serve --help` for OpenAI-compatible server options."
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("serve") => {
            if let Err(error) = server::run(&args[2..]) {
                eprintln!("Server error: {error}");
                std::process::exit(1);
            }
            return;
        }
        Some("-h" | "--help" | "help") => {
            print_usage(io::stdout()).expect("failed to write help");
            return;
        }
        Some("-V" | "--version" | "version") => {
            println!("runNburn {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        _ => {}
    }

    let mut model_arg = 1;
    let mut ram_budget_bytes = None;
    while model_arg < args.len() {
        let raw_budget = if args[model_arg] == "--ram-budget" {
            model_arg += 1;
            args.get(model_arg).map(String::as_str)
        } else {
            args[model_arg].strip_prefix("--ram-budget=")
        };
        let Some(raw_budget) = raw_budget else {
            break;
        };
        ram_budget_bytes = Some(parse_byte_size(raw_budget).unwrap_or_else(|message| {
            eprintln!("Invalid --ram-budget: {message}");
            std::process::exit(2);
        }));
        model_arg += 1;
    }

    if model_arg >= args.len() {
        print_usage(io::stderr()).expect("failed to write usage");
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[model_arg]);
    let prompt = if args.len() > model_arg + 1 {
        args[model_arg + 1..].join(" ")
    } else {
        // Interactive mode
        eprint!("prompt> ");
        io::stderr().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input.trim().to_string()
    };

    eprintln!("Loading model from {}...", model_path.display());
    eprintln!(
        "Runtime backends: {}",
        runtime_boundary::compiled_runtime_backends().join(",")
    );

    let mut load_config = rnb_llm::EngineLoadConfig::default();
    if let Some(bytes) = ram_budget_bytes {
        load_config = load_config.with_host_ram_budget_bytes(bytes);
    }
    let mut engine = match rnb_llm::Engine::from_gguf_with_config(&model_path, load_config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("Model loaded. Generating...");
    eprintln!("---");

    let params = rnb_llm::GenerateParams::default();

    match engine.generate_stream(&prompt, &params, |token_text| {
        print!("{token_text}");
        io::stdout().flush().unwrap();
        true
    }) {
        Ok(result) => {
            println!();
            eprintln!("---");
            eprintln!(
                "Generated {} tokens ({:.1} tokens/sec)",
                result.tokens_generated, result.tokens_per_second
            );
        }
        Err(e) => {
            eprintln!("\nGeneration error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_byte_size;

    #[test]
    fn parses_binary_and_decimal_ram_budget_sizes() {
        assert_eq!(parse_byte_size("32GiB").unwrap(), 32_u64 << 30);
        assert_eq!(parse_byte_size("64GB").unwrap(), 64_000_000_000);
        assert!(parse_byte_size("32G").is_err());
        assert!(parse_byte_size("0GiB").is_err());
        assert!(parse_byte_size("GiB").is_err());
    }
}
