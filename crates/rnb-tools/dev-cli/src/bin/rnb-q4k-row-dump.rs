//! mt91 debug: dump Rust's dequantized row of a Q4_K tensor to a binary file.
//!
//! Calls `rnb_cpu::gemm::dequant::dequantize_bytes_to_f32` (the actual entry
//! point shared by `scalar_gemv::dot_k_block_row`, `gemv_generic`, and
//! `gemv_output_f64_logit` in `crates/rnb-llm/src/engine/scalar_gemv.rs`).
//!
//! Output is a little-endian f32 buffer of length `cols`.
//!
//! Usage:
//!   rnb-q4k-row-dump --gguf <gguf path> --tensor <name> --row <usize> --out <bin path>

use std::path::PathBuf;
use std::process::ExitCode;

use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};
use rnb_loader::GGMLType;

fn parse_args() -> Result<(PathBuf, String, usize, PathBuf), String> {
    let mut gguf: Option<PathBuf> = None;
    let mut tensor: Option<String> = None;
    let mut row: Option<usize> = None;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => {
                gguf = Some(PathBuf::from(args.next().ok_or("--gguf needs value")?));
            }
            "--tensor" => {
                tensor = Some(args.next().ok_or("--tensor needs value")?);
            }
            "--row" => {
                row = Some(
                    args.next()
                        .ok_or("--row needs value")?
                        .parse::<usize>()
                        .map_err(|e| format!("--row parse: {e}"))?,
                );
            }
            "--out" => {
                out = Some(PathBuf::from(args.next().ok_or("--out needs value")?));
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok((
        gguf.ok_or("--gguf required")?,
        tensor.ok_or("--tensor required")?,
        row.ok_or("--row required")?,
        out.ok_or("--out required")?,
    ))
}

fn main() -> ExitCode {
    let (gguf_path, tensor_name, row, out_path) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("arg error: {e}");
            eprintln!(
                "usage: rnb-q4k-row-dump --gguf <path> --tensor <name> --row <usize> --out <bin>"
            );
            return ExitCode::from(2);
        }
    };

    eprintln!("[load] {}", gguf_path.display());
    let model = match rnb_loader::load_model(&gguf_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load_model failed: {e:?}");
            return ExitCode::from(1);
        }
    };

    let tensor = match model.weights.get(&tensor_name) {
        Some(t) => t,
        None => {
            eprintln!("tensor not found: {tensor_name}");
            return ExitCode::from(1);
        }
    };
    let ggml_type = match model.tensor_ggml_types.get(&tensor_name).copied() {
        Some(t) => t,
        None => {
            eprintln!("no ggml_type for tensor: {tensor_name}");
            return ExitCode::from(1);
        }
    };
    let float_shape = match model.float_shapes.get(&tensor_name) {
        Some(s) => s.clone(),
        None => {
            eprintln!("no float_shape for tensor: {tensor_name}");
            return ExitCode::from(1);
        }
    };

    if ggml_type != GGMLType::Q4_K {
        eprintln!("expected Q4_K, got {:?}", ggml_type);
        return ExitCode::from(1);
    }

    // float_shape convention used elsewhere: [rows, cols] = [out, in].
    // tensor bytes are stored row-major over the leading axis.
    if float_shape.len() < 2 {
        eprintln!("unexpected shape: {:?}", float_shape);
        return ExitCode::from(1);
    }
    let rows = float_shape[0];
    let cols = float_shape[1];

    let bytes = match tensor.as_bytes() {
        Some(b) => b,
        None => {
            eprintln!("tensor bytes unavailable");
            return ExitCode::from(1);
        }
    };

    if row >= rows {
        eprintln!("row {row} >= rows {rows}");
        return ExitCode::from(1);
    }

    let total = bytes.len();
    if total % rows != 0 {
        eprintln!("total bytes {total} not divisible by rows {rows}");
        return ExitCode::from(1);
    }
    let bytes_per_row = total / rows;

    // Q4_K block: 144 bytes / 256 elements.
    let expected_bpr = (cols / 256) * 144;
    if bytes_per_row != expected_bpr {
        eprintln!("bytes_per_row {bytes_per_row} != expected {expected_bpr} (cols={cols})",);
        return ExitCode::from(1);
    }

    eprintln!(
        "tensor={} type={:?} rows={} cols={} bpr={} row={}",
        tensor_name, ggml_type, rows, cols, bytes_per_row, row
    );

    let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];

    // Call THE SHARED ENTRY POINT used by scalar_gemv.rs paths.
    let dequant = dequantize_bytes_to_f32(row_bytes, DequantType::Q4K);

    if dequant.len() != cols {
        eprintln!("dequant len {} != cols {}", dequant.len(), cols);
        return ExitCode::from(1);
    }

    let mut le_bytes = Vec::with_capacity(dequant.len() * 4);
    for v in &dequant {
        le_bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&out_path, &le_bytes) {
        eprintln!("write {} failed: {e}", out_path.display());
        return ExitCode::from(1);
    }

    eprintln!(
        "[write] {} ({} f32 values, {} bytes)",
        out_path.display(),
        dequant.len(),
        le_bytes.len()
    );
    eprintln!(
        "first5={:?} last5={:?}",
        &dequant[..5.min(dequant.len())],
        &dequant[dequant.len().saturating_sub(5)..]
    );
    ExitCode::SUCCESS
}
