//! GGUF metadata + tensor 표 dump 유틸. inference 안 함, parser 만 호출.
use std::env;
use std::path::Path;
use std::process::ExitCode;

use rnb_core::memory::mmap::MmapLoader;
use rnb_loader::convert::ggml_quant_params;
use rnb_loader::gguf::parser::GGUFFile;
use rnb_loader::gguf::types::{GGUFValue, TensorInfo};

fn tensor_size_bytes(t: &TensorInfo) -> u64 {
    let nel: u64 = t.shape.iter().product::<usize>() as u64;
    let (bs, bb) = ggml_quant_params(t.ggml_type);
    let nblocks = nel / bs as u64;
    nblocks * bb as u64
}

fn fmt_value(v: &GGUFValue) -> String {
    match v {
        GGUFValue::U8(x) => format!("u8={}", x),
        GGUFValue::I8(x) => format!("i8={}", x),
        GGUFValue::U16(x) => format!("u16={}", x),
        GGUFValue::I16(x) => format!("i16={}", x),
        GGUFValue::U32(x) => format!("u32={}", x),
        GGUFValue::I32(x) => format!("i32={}", x),
        GGUFValue::U64(x) => format!("u64={}", x),
        GGUFValue::I64(x) => format!("i64={}", x),
        GGUFValue::F32(x) => format!("f32={}", x),
        GGUFValue::F64(x) => format!("f64={}", x),
        GGUFValue::Bool(x) => format!("bool={}", x),
        GGUFValue::String(s) => {
            if s.len() > 120 {
                format!("string=({} chars) {:?}…", s.len(), &s[..120])
            } else {
                format!("string={:?}", s)
            }
        }
        GGUFValue::Array(arr) => {
            let elem_kind = arr
                .first()
                .map(|v| match v {
                    GGUFValue::U8(_) => "u8",
                    GGUFValue::I8(_) => "i8",
                    GGUFValue::U16(_) => "u16",
                    GGUFValue::I16(_) => "i16",
                    GGUFValue::U32(_) => "u32",
                    GGUFValue::I32(_) => "i32",
                    GGUFValue::U64(_) => "u64",
                    GGUFValue::I64(_) => "i64",
                    GGUFValue::F32(_) => "f32",
                    GGUFValue::F64(_) => "f64",
                    GGUFValue::Bool(_) => "bool",
                    GGUFValue::String(_) => "string",
                    GGUFValue::Array(_) => "array",
                })
                .unwrap_or("?");
            format!("array<{}, len={}>", elem_kind, arr.len())
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: rnb-gguf-dump <path-to-gguf>");
        return ExitCode::from(2);
    }
    let path = Path::new(&args[1]);

    let mmap = match MmapLoader::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mmap failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    let gguf = match GGUFFile::parse(&mmap[..]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gguf parse failed: {:?}", e);
            return ExitCode::from(1);
        }
    };

    println!("# GGUF dump: {}", path.display());
    println!("file_size_bytes = {}", mmap.len());
    println!("version = {}", gguf.version);
    println!("data_start = {}", gguf.data_start);
    println!("alignment = {}", gguf.alignment);
    println!();

    println!("## Metadata ({} entries)", gguf.metadata.len());
    for (k, v) in gguf.metadata.iter() {
        println!("- `{}` = {}", k, fmt_value(v));
    }
    println!();

    println!("## Tensors ({} total)", gguf.tensor_infos.len());
    println!();
    println!("| name | quant | shape | offset | size_bytes |");
    println!("|---|---|---|---:|---:|");
    let mut total_bytes: u64 = 0;
    for t in &gguf.tensor_infos {
        let size = tensor_size_bytes(t);
        total_bytes += size;
        let shape_str = format!("{:?}", t.shape);
        println!(
            "| `{}` | {:?} | {} | {} | {} |",
            t.name, t.ggml_type, shape_str, t.offset, size
        );
    }
    println!();
    println!(
        "Total tensor bytes: {} ({:.2} GiB)",
        total_bytes,
        total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

    // Expert / router pattern hint
    println!();
    println!("## Expert / router pattern matches");
    let mut expert_total: u64 = 0;
    let mut expert_count = 0usize;
    for t in &gguf.tensor_infos {
        let n = &t.name;
        if n.contains("exps")
            || n.contains("router")
            || n.contains("ffn_gate_inp")
            || n.contains("expert")
        {
            let size = tensor_size_bytes(t);
            expert_total += size;
            expert_count += 1;
            println!(
                "- `{}` quant={:?} shape={:?} size={} ({:.2} MiB)",
                n,
                t.ggml_type,
                t.shape,
                size,
                size as f64 / 1024.0 / 1024.0
            );
        }
    }
    println!();
    println!("Expert-pattern tensor count: {}", expert_count);
    println!(
        "Expert-pattern total bytes: {} ({:.2} GiB)",
        expert_total,
        expert_total as f64 / 1024.0 / 1024.0 / 1024.0
    );

    ExitCode::SUCCESS
}
