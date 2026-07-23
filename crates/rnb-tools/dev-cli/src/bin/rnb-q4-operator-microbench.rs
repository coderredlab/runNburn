//! mt91 — CLI entry point for the Q4 operator-level GEMV microbench.
//!
//! Runs the three Rust GEMV paths (`prod`, `generic`, `f64`) against a single
//! `(GGUF tensor, bf16-reference input.bin)` pair and dumps one little-endian
//! `f32` `.bin` per path. The bf16 reference comes from the Python mt90 stage
//! dumper; comparison is done by `docs/sessions/mtp/mt91_operator_compare.py`.
//!
//! Example:
//! ```text
//! rnb-q4-operator-microbench \
//!     --gguf models/gemma-4-E4B/Gemma-4-E4B-Q4_K_M.gguf \
//!     --layer 17 --operator o_proj \
//!     --input-bin /tmp/mt90_ref_stage/layer_017_attn_out.bin \
//!     --output-dir /tmp/mt91_microbench
//! ```

use rnb_dev_tools::q4_operator_microbench::{run_four_paths, OperatorKind, OperatorMicrobenchArgs};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_and_run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("rnb-q4-operator-microbench: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn parse_and_run() -> Result<(), String> {
    let mut args_iter = std::env::args().skip(1);
    let mut gguf: Option<PathBuf> = None;
    let mut layer: Option<usize> = None;
    let mut operator: Option<OperatorKind> = None;
    let mut input_bin: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;

    while let Some(flag) = args_iter.next() {
        match flag.as_str() {
            "--gguf" => {
                gguf = Some(PathBuf::from(
                    args_iter.next().ok_or("--gguf needs a value")?,
                ));
            }
            "--layer" => {
                let v = args_iter.next().ok_or("--layer needs a value")?;
                layer = Some(v.parse().map_err(|e| format!("--layer parse: {e}"))?);
            }
            "--operator" => {
                let v = args_iter.next().ok_or("--operator needs a value")?;
                operator = Some(OperatorKind::parse(&v)?);
            }
            "--input-bin" => {
                input_bin = Some(PathBuf::from(
                    args_iter.next().ok_or("--input-bin needs a value")?,
                ));
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    args_iter.next().ok_or("--output-dir needs a value")?,
                ));
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let args = OperatorMicrobenchArgs {
        gguf_path: gguf.ok_or("--gguf required")?,
        layer: layer.ok_or("--layer required")?,
        operator: operator
            .ok_or("--operator required (q_proj|k_proj|v_proj|o_proj|mlp_gate|mlp_up|mlp_down)")?,
        input_bin: input_bin.ok_or("--input-bin required")?,
        output_dir: output_dir.ok_or("--output-dir required")?,
    };
    run_four_paths(&args)
}

fn print_help() {
    eprintln!(
        "rnb-q4-operator-microbench — mt91 Q4 operator drift probe\n\
         \n\
         Flags (all required unless noted):\n\
           --gguf <path>           GGUF model file\n\
           --layer <usize>         Decoder block index (e.g. 17)\n\
           --operator <name>       q_proj|k_proj|v_proj|o_proj|mlp_gate|mlp_up|mlp_down\n\
           --input-bin <path>      Little-endian f32 row, length == cols\n\
           --output-dir <path>     Directory for {{prod,generic,f64}}.bin dumps"
    );
}
