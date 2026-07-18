//! mt91 — Q4 operator-level microbench library module.
//!
//! Drives the three Rust GEMV paths exposed by `rnb_llm::engine::q4_microbench`
//! over a single (`gguf`, `layer`, `operator`, `input.bin`) triple and dumps
//! one little-endian `f32` `.bin` per path into an output directory. The bf16
//! reference comes from the mt90 Python dump and is compared in a separate
//! Python comparator (`mt91_operator_compare.py`).
//!
//! Output naming: `layer_{NNN}_{operator}_path_{prod|generic|f64}.bin`.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorKind {
    QProj,
    KProj,
    VProj,
    OProj,
    MlpGate,
    MlpUp,
    MlpDown,
}

impl OperatorKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "q_proj" => Ok(Self::QProj),
            "k_proj" => Ok(Self::KProj),
            "v_proj" => Ok(Self::VProj),
            "o_proj" => Ok(Self::OProj),
            "mlp_gate" => Ok(Self::MlpGate),
            "mlp_up" => Ok(Self::MlpUp),
            "mlp_down" => Ok(Self::MlpDown),
            other => Err(format!("unknown operator: {other}")),
        }
    }

    /// GGUF tensor name template — `{layer}` is replaced with the layer index.
    pub fn gguf_tensor_template(self) -> &'static str {
        match self {
            Self::QProj => "blk.{layer}.attn_q.weight",
            Self::KProj => "blk.{layer}.attn_k.weight",
            Self::VProj => "blk.{layer}.attn_v.weight",
            Self::OProj => "blk.{layer}.attn_output.weight",
            Self::MlpGate => "blk.{layer}.ffn_gate.weight",
            Self::MlpUp => "blk.{layer}.ffn_up.weight",
            Self::MlpDown => "blk.{layer}.ffn_down.weight",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::MlpGate => "mlp_gate",
            Self::MlpUp => "mlp_up",
            Self::MlpDown => "mlp_down",
        }
    }
}

#[derive(Debug)]
pub struct OperatorMicrobenchArgs {
    pub gguf_path: PathBuf,
    pub layer: usize,
    pub operator: OperatorKind,
    /// `f32` little-endian binary, exactly `cols` elements (single row input).
    pub input_bin: PathBuf,
    pub output_dir: PathBuf,
}

/// Run all three Rust GEMV paths (prod / generic / f64_logit) and dump
/// little-endian `f32` outputs into `args.output_dir`.
pub fn run_four_paths(args: &OperatorMicrobenchArgs) -> Result<(), String> {
    use rnb_llm::engine::q4_microbench;
    use std::fs;

    if !args.gguf_path.exists() {
        return Err(format!("gguf missing: {:?}", args.gguf_path));
    }
    if !args.input_bin.exists() {
        return Err(format!("input.bin missing: {:?}", args.input_bin));
    }
    fs::create_dir_all(&args.output_dir).map_err(|e| e.to_string())?;

    // 1. GGUF load + locate the tensor for (layer, operator).
    let tensor_name = args
        .operator
        .gguf_tensor_template()
        .replace("{layer}", &args.layer.to_string());
    let (bytes, ggml_type, rows, cols) = load_weight_from_gguf(&args.gguf_path, &tensor_name)?;

    // 2. Input.bin → f32 (must equal `cols`).
    let input = read_f32_bin(&args.input_bin)?;
    if input.len() != cols {
        return Err(format!(
            "input dim mismatch for {tensor_name}: expected {cols}, got {}",
            input.len()
        ));
    }

    // 3. Run the three paths.
    let mut prod_out = vec![0.0f32; rows];
    let mut generic_out = vec![0.0f32; rows];
    let mut f64_out = vec![0.0f32; rows];

    q4_microbench::run_gemv_prod(&bytes, ggml_type, rows, cols, &input, &mut prod_out)?;
    q4_microbench::run_gemv_generic(&bytes, ggml_type, rows, cols, &input, &mut generic_out)?;
    q4_microbench::run_gemv_f64_logit(&bytes, ggml_type, rows, cols, &input, &mut f64_out)?;

    // 4. Dump each path's output as little-endian f32 (mt90 dump format).
    let op = args.operator.as_str();
    let stem = format!("layer_{:03}_{op}", args.layer);
    write_f32_bin(
        &args.output_dir.join(format!("{stem}_path_prod.bin")),
        &prod_out,
    )?;
    write_f32_bin(
        &args.output_dir.join(format!("{stem}_path_generic.bin")),
        &generic_out,
    )?;
    write_f32_bin(
        &args.output_dir.join(format!("{stem}_path_f64.bin")),
        &f64_out,
    )?;

    Ok(())
}

fn read_f32_bin(path: &std::path::Path) -> Result<Vec<f32>, String> {
    use std::fs::File;
    use std::io::Read;
    let mut f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .map_err(|e| format!("read {path:?}: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "input.bin size {} for {path:?} not multiple of 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn write_f32_bin(path: &std::path::Path, data: &[f32]) -> Result<(), String> {
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| format!("write {path:?}: {e}"))?;
    Ok(())
}

/// Load `tensor_name` from a GGUF file as `(bytes, ggml_type, rows, cols)`.
///
/// Uses `rnb_loader::load_model`; `rows` = out_features, `cols` = in_features
/// per the engine convention (`weight @ x` with `x: [cols]` → `out: [rows]`).
/// `float_shapes` returned by the loader is already in engine convention,
/// i.e. `[rows, cols]` = `[out_features, in_features]`. This matches the
/// standard loader at `rnb-llm/src/engine/weight_loading/quantized.rs:25-37`
/// which reads `(rows, cols) = (float_shape[0], float_shape[1])`.
fn load_weight_from_gguf(
    gguf_path: &std::path::Path,
    tensor_name: &str,
) -> Result<(Vec<u8>, rnb_loader::GGMLType, usize, usize), String> {
    let model = rnb_loader::load_model(gguf_path).map_err(|e| format!("load_gguf: {e:?}"))?;
    let tensor = model
        .weights
        .get(tensor_name)
        .ok_or_else(|| format!("tensor not found: {tensor_name}"))?;
    let ggml_type = *model
        .tensor_ggml_types
        .get(tensor_name)
        .ok_or_else(|| format!("ggml type missing for {tensor_name}"))?;
    let float_shape = model
        .float_shapes
        .get(tensor_name)
        .ok_or_else(|| format!("float shape missing for {tensor_name}"))?;
    if float_shape.len() < 2 {
        return Err(format!(
            "tensor {tensor_name} has shape {float_shape:?}; expected at least 2D weight"
        ));
    }
    // float_shapes is in engine convention: [rows, cols] = [out_features, in_features].
    let rows = float_shape[0];
    let cols = float_shape[1];

    // Copy the raw quant bytes out of the mmap-backed Tensor. Quant tensors
    // are stored as a flat U8 buffer (`[byte_count]` 1-D in `weights`); copy
    // it into an owned `Vec<u8>` so the microbench wrapper owns the bytes.
    let bytes = tensor
        .as_bytes()
        .ok_or_else(|| format!("tensor {tensor_name} has no byte view"))?
        .to_vec();
    Ok((bytes, ggml_type, rows, cols))
}
