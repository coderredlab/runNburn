//! Extract selected GGUF rows for MediaTek MLP diagnostics.

use std::fmt::Write as FmtWrite;
use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::process::ExitCode;

use rnb_dev_tools::mtk_gguf_mlp::{
    extract_from_loaded_model, metadata_json, quantize_mlp_payload, quantized_metadata_json,
    write_rnbmtk_quant_mlp, MtkMlpConfig,
};
#[cfg(feature = "mediatek")]
use rnb_dev_tools::mtk_gguf_mlp::{
    mediatek_quantized_mlp_tensor_view, quantized_output_parity, QuantizedMtkMlpPayload,
};

#[derive(Debug)]
struct Args {
    gguf: PathBuf,
    out: Option<PathBuf>,
    metadata_out: Option<PathBuf>,
    quantized_out: Option<PathBuf>,
    quantized_stdout: bool,
    quantized_metadata_out: Option<PathBuf>,
    mediatek_nnapi_run: bool,
    #[cfg_attr(not(feature = "mediatek"), allow(dead_code))]
    mediatek_device_name_substring: String,
    #[cfg_attr(not(feature = "mediatek"), allow(dead_code))]
    mediatek_tolerance: f32,
    config: MtkMlpConfig,
}

fn usage() {
    eprintln!(concat!(
        "usage: rnb-mtk-gguf-mlp-extract --gguf <model.gguf> (--out <model.rnbmtk-mlp> | --quantized-out <model.rnbmtkq> | --quantized-stdout | --mediatek-nnapi-run) [options]\n",
        "\n",
        "options:\n",
        "  --out PATH             optional FP32 RNBMTK3 diagnostic output\n",
        "  --metadata-out PATH    optional FP32 manual JSON metadata\n",
        "  --quantized-out PATH   optional UINT8 RNBMTKQ1 diagnostic output\n",
        "  --quantized-stdout     write UINT8 RNBMTKQ1 diagnostic bytes to stdout\n",
        "  --quantized-metadata-out PATH  optional quantized manual JSON metadata\n",
        "  --mediatek-nnapi-run   run quantized MLP through Android NNAPI without writing a model payload file\n",
        "  --mediatek-device-name-substring TEXT  default: mtk-neuron\n",
        "  --mediatek-tolerance F32  default: 0.003, max: 0.01\n",
        "  --w1-tensor NAME       default: blk.0.ffn_up.weight\n",
        "  --w2-tensor NAME       default: blk.0.ffn_down.weight\n",
        "  --input-tensor NAME    default: token_embd.weight\n",
        "  --input-row N          default: 0\n",
        "  --input-size N         default: 256\n",
        "  --hidden-size N        default: 128\n",
        "  --output-size N        default: 64\n",
        "  --input-scale F32      default: 1.0\n"
    ));
}

fn parse_args() -> Result<Args, String> {
    let mut gguf: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut metadata_out: Option<PathBuf> = None;
    let mut quantized_out: Option<PathBuf> = None;
    let mut quantized_stdout = false;
    let mut quantized_metadata_out: Option<PathBuf> = None;
    let mut mediatek_nnapi_run = false;
    let mut mediatek_device_name_substring = "mtk-neuron".to_string();
    let mut mediatek_tolerance = 0.003;
    let mut config = MtkMlpConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gguf" => gguf = Some(PathBuf::from(take_value(&mut args, "--gguf")?)),
            "--out" => out = Some(PathBuf::from(take_value(&mut args, "--out")?)),
            "--metadata-out" => {
                metadata_out = Some(PathBuf::from(take_value(&mut args, "--metadata-out")?));
            }
            "--quantized-out" => {
                quantized_out = Some(PathBuf::from(take_value(&mut args, "--quantized-out")?));
            }
            "--quantized-stdout" => quantized_stdout = true,
            "--quantized-metadata-out" => {
                quantized_metadata_out = Some(PathBuf::from(take_value(
                    &mut args,
                    "--quantized-metadata-out",
                )?));
            }
            "--mediatek-nnapi-run" => mediatek_nnapi_run = true,
            "--mediatek-device-name-substring" => {
                mediatek_device_name_substring =
                    take_value(&mut args, "--mediatek-device-name-substring")?;
            }
            "--mediatek-tolerance" => {
                mediatek_tolerance = parse_mediatek_tolerance(&mut args, "--mediatek-tolerance")?;
            }
            "--w1-tensor" => config.w1_tensor = take_value(&mut args, "--w1-tensor")?,
            "--w2-tensor" => config.w2_tensor = take_value(&mut args, "--w2-tensor")?,
            "--input-tensor" => config.input_tensor = take_value(&mut args, "--input-tensor")?,
            "--input-row" => config.input_row = parse_usize(&mut args, "--input-row")?,
            "--input-size" => config.input_size = parse_usize(&mut args, "--input-size")?,
            "--hidden-size" => config.hidden_size = parse_usize(&mut args, "--hidden-size")?,
            "--output-size" => config.output_size = parse_usize(&mut args, "--output-size")?,
            "--input-scale" => config.input_scale = parse_f32(&mut args, "--input-scale")?,
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if out.is_none() && quantized_out.is_none() && !quantized_stdout && !mediatek_nnapi_run {
        return Err(
            "--out, --quantized-out, --quantized-stdout, or --mediatek-nnapi-run required"
                .to_string(),
        );
    }
    if mediatek_nnapi_run
        && (out.is_some()
            || metadata_out.is_some()
            || quantized_out.is_some()
            || quantized_stdout
            || quantized_metadata_out.is_some())
    {
        return Err(
            "--mediatek-nnapi-run cannot be combined with file or stdout output options"
                .to_string(),
        );
    }
    if mediatek_nnapi_run && mediatek_device_name_substring.trim().is_empty() {
        return Err("--mediatek-device-name-substring must not be empty".to_string());
    }
    if metadata_out.is_some() && out.is_none() {
        return Err("--metadata-out requires --out".to_string());
    }
    if quantized_metadata_out.is_some() && quantized_out.is_none() && !quantized_stdout {
        return Err(
            "--quantized-metadata-out requires --quantized-out or --quantized-stdout".to_string(),
        );
    }
    Ok(Args {
        gguf: gguf.ok_or("--gguf required")?,
        out,
        metadata_out,
        quantized_out,
        quantized_stdout,
        quantized_metadata_out,
        mediatek_nnapi_run,
        mediatek_device_name_substring,
        mediatek_tolerance,
        config,
    })
}

fn take_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn parse_usize(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize, String> {
    take_value(args, name)?
        .parse::<usize>()
        .map_err(|err| format!("{name} parse failed: {err}"))
}

fn parse_f32(args: &mut impl Iterator<Item = String>, name: &str) -> Result<f32, String> {
    let value = take_value(args, name)?
        .parse::<f32>()
        .map_err(|err| format!("{name} parse failed: {err}"))?;
    if !value.is_finite() {
        return Err(format!("{name} must be finite, got {value}"));
    }
    Ok(value)
}

fn parse_mediatek_tolerance(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<f32, String> {
    let value = parse_f32(args, name)?;
    if value <= 0.0 || value > 0.01 {
        return Err(format!("{name} must be > 0 and <= 0.01, got {value}"));
    }
    Ok(value)
}

fn emit_status(to_stderr: bool, args: std::fmt::Arguments<'_>) {
    if to_stderr {
        eprintln!("{args}");
    } else {
        println!("{args}");
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("arg error: {err}");
            usage();
            return ExitCode::from(2);
        }
    };
    #[cfg(not(feature = "mediatek"))]
    if args.mediatek_nnapi_run {
        eprintln!("arg error: --mediatek-nnapi-run requires building with --features mediatek");
        return ExitCode::from(2);
    }

    eprintln!("[load] {}", args.gguf.display());
    let model = match rnb_loader::load_model(&args.gguf) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("load_model failed: {err:?}");
            return ExitCode::from(1);
        }
    };

    let extracted = match extract_from_loaded_model(&model, &args.config) {
        Ok(extracted) => extracted,
        Err(err) => {
            eprintln!("extract failed: {err}");
            return ExitCode::from(1);
        }
    };

    if let Some(path) = &args.out {
        if let Err(err) = write_file(path, &extracted.bytes, "output") {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
        let json = metadata_json(&extracted.metadata);
        if let Some(path) = &args.metadata_out {
            if let Err(err) = write_file(path, json.as_bytes(), "metadata") {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
        emit_status(
            args.quantized_stdout,
            format_args!("output={}", path.display()),
        );
        if let Some(path) = &args.metadata_out {
            emit_status(
                args.quantized_stdout,
                format_args!("metadata={}", path.display()),
            );
        }
    }

    let quantized =
        if args.quantized_out.is_some() || args.quantized_stdout || args.mediatek_nnapi_run {
            let quantized = match quantize_mlp_payload(&extracted.payload) {
                Ok(quantized) => quantized,
                Err(err) => {
                    eprintln!("quantize failed: {err}");
                    return ExitCode::from(1);
                }
            };
            Some(quantized)
        } else {
            None
        };
    if args.quantized_out.is_some()
        || args.quantized_stdout
        || args.quantized_metadata_out.is_some()
    {
        let quantized = quantized
            .as_ref()
            .expect("quantized payload must exist when quantized output is requested");
        let quantized_bytes = match write_rnbmtk_quant_mlp(quantized) {
            Ok(bytes) => bytes,
            Err(err) => {
                eprintln!("quantized write failed: {err}");
                return ExitCode::from(1);
            }
        };
        if let Some(path) = &args.quantized_out {
            if let Err(err) = write_file(path, &quantized_bytes, "quantized output") {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
        if let Some(path) = &args.quantized_metadata_out {
            let json = quantized_metadata_json(quantized, &quantized_bytes);
            if let Err(err) = write_file(path, json.as_bytes(), "quantized metadata") {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
        if args.quantized_stdout {
            if let Err(err) = io::stdout().write_all(&quantized_bytes) {
                eprintln!("write quantized stdout failed: {err}");
                return ExitCode::from(1);
            }
            emit_status(
                true,
                format_args!("quantized_stdout_bytes={}", quantized_bytes.len()),
            );
            if let Some(path) = &args.quantized_out {
                emit_status(true, format_args!("quantized_output={}", path.display()));
            }
            if let Some(path) = &args.quantized_metadata_out {
                emit_status(true, format_args!("quantized_metadata={}", path.display()));
            }
        } else {
            if let Some(path) = &args.quantized_out {
                emit_status(false, format_args!("quantized_output={}", path.display()));
            }
            if let Some(path) = &args.quantized_metadata_out {
                emit_status(false, format_args!("quantized_metadata={}", path.display()));
            }
        }
    }
    if args.mediatek_nnapi_run {
        #[cfg(feature = "mediatek")]
        {
            let quantized = quantized
                .as_ref()
                .expect("quantized payload must exist when MediaTek NNAPI run is requested");
            if let Err(failure) = run_mediatek_nnapi(quantized, &args) {
                eprintln!("{}", failure.message);
                return ExitCode::from(failure.code);
            }
        }
    }
    let mut metadata = String::new();
    writeln!(
        &mut metadata,
        "dims input={} hidden={} output={}",
        extracted.metadata.input_size,
        extracted.metadata.hidden_size,
        extracted.metadata.output_size
    )
    .expect("format metadata");
    writeln!(
        &mut metadata,
        "input_scale={:.9}",
        extracted.metadata.input_scale
    )
    .expect("format metadata");
    writeln!(
        &mut metadata,
        "payload_bytes={}",
        extracted.metadata.payload_bytes
    )
    .expect("format metadata");
    writeln!(
        &mut metadata,
        "payload_sha256={}",
        extracted.metadata.payload_sha256
    )
    .expect("format metadata");
    for tensor in &extracted.metadata.tensors {
        writeln!(
            &mut metadata,
            "tensor={} type={:?} shape={:?} row_start={} rows_selected={} cols_selected={} bytes_per_row={}",
            tensor.name,
            tensor.ggml_type,
            tensor.shape,
            tensor.row_start,
            tensor.rows_selected,
            tensor.cols_selected,
            tensor.bytes_per_row
        )
        .expect("format metadata");
    }
    write!(
        &mut metadata,
        "range.w1=[{:.9},{:.9}] range.w2=[{:.9},{:.9}] range.input=[{:.9},{:.9}] range.hidden=[{:.9},{:.9}] range.output=[{:.9},{:.9}]",
        extracted.metadata.w1_range.min,
        extracted.metadata.w1_range.max,
        extracted.metadata.w2_range.min,
        extracted.metadata.w2_range.max,
        extracted.metadata.input_range.min,
        extracted.metadata.input_range.max,
        extracted.metadata.hidden_range.min,
        extracted.metadata.hidden_range.max,
        extracted.metadata.output_range.min,
        extracted.metadata.output_range.max,
    )
    .expect("format metadata");
    if args.quantized_stdout {
        emit_status(true, format_args!("{metadata}"));
    } else {
        emit_status(false, format_args!("{metadata}"));
    }
    ExitCode::SUCCESS
}

fn write_file(path: &PathBuf, bytes: &[u8], label: &str) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("create {label} dir {} failed: {err}", parent.display()))?;
    }
    std::fs::write(path, bytes)
        .map_err(|err| format!("write {label} {} failed: {err}", path.display()))
}

#[cfg(feature = "mediatek")]
#[derive(Debug)]
struct CliFailure {
    code: u8,
    message: String,
}

#[cfg(feature = "mediatek")]
impl CliFailure {
    fn new(code: u8, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[cfg(feature = "mediatek")]
fn run_mediatek_nnapi(quantized: &QuantizedMtkMlpPayload, args: &Args) -> Result<(), CliFailure> {
    let tensors = mediatek_quantized_mlp_tensor_view(quantized)
        .map_err(|err| CliFailure::new(1, format!("mediatek tensor view failed: {err}")))?;
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let output = backend
        .run_quantized_mlp(
            &tensors,
            rnb_backend_mediatek::MediaTekNnapiOptions::new(
                args.mediatek_device_name_substring.clone(),
            ),
        )
        .map_err(|err| CliFailure::new(1, format!("mediatek nnapi run failed: {err}")))?;
    let parity = quantized_output_parity(quantized, output.output())
        .map_err(|err| CliFailure::new(1, format!("mediatek parity failed: {err}")))?;

    println!("mediatek_nnapi_run=true");
    println!("chosen_device.name={}", output.chosen_device().name());
    println!(
        "chosen_device.type={}",
        mediatek_device_type_label(output.chosen_device().device_type())
    );
    println!(
        "chosen_device.type_code={}",
        output.chosen_device().device_type()
    );
    println!(
        "chosen_device.feature_level={}",
        output.chosen_device().feature_level()
    );
    println!("chosen_device.version={}", output.chosen_device().version());
    println!("supported_ops.fc1={}", output.supported_ops().fc1());
    println!("supported_ops.fc2={}", output.supported_ops().fc2());
    println!(
        "duration_hardware_ns={}",
        optional_u64_label(output.duration_hardware_ns())
    );
    println!(
        "duration_driver_ns={}",
        optional_u64_label(output.duration_driver_ns())
    );
    println!("output_len={}", output.output().len());
    println!("max_byte_delta={}", parity.max_byte_delta);
    println!("max_byte_delta_index={}", parity.max_byte_delta_index);
    println!("max_abs_error={:.9}", parity.max_abs_error);
    if !parity.passes(args.mediatek_tolerance) {
        return Err(CliFailure::new(
            5,
            format!(
                "MTK_NNAPI_INMEMORY_MLP_FAIL max_byte_delta={} max_abs_error={:.9} tolerance={:.9}",
                parity.max_byte_delta, parity.max_abs_error, args.mediatek_tolerance
            ),
        ));
    }
    println!("MTK_NNAPI_INMEMORY_MLP_OK");
    Ok(())
}

#[cfg(feature = "mediatek")]
fn mediatek_device_type_label(device_type: i32) -> &'static str {
    if device_type == rnb_backend_mediatek::MEDIATEK_NNAPI_DEVICE_ACCELERATOR {
        "ACCELERATOR"
    } else {
        "OTHER"
    }
}

#[cfg(feature = "mediatek")]
fn optional_u64_label(value: Option<u64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "unavailable".to_string(),
    }
}
