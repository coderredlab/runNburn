//! mt91 — Q4 operator-level GEMV microbench API.
//!
//! Exposes the three Rust GEMV paths that live inside `engine::quantized_weight`
//! (`pub(super)`) as a small `pub` surface so external crates (`rnb-dev-tools`,
//! integration tests) can drive them with the same `(bytes, ggml_type, rows, cols)`
//! triple. The bf16 reference comes from Python (mt90 dump), not this module.
//!
//! ## Paths
//! - `prod`     — `QuantizedWeight::gemv_into` (production dispatch, including
//!                NEON SIMD on aarch64, CUDA when feature enabled, etc.).
//! - `generic`  — `QuantizedWeight::gemv_into_generic` (scalar dequant + f32 reduce,
//!                no SIMD shortcuts).
//! - `f64_logit` — `QuantizedWeight::gemv_into_f64_logit` (full dequant + f64
//!                accumulator; single-row only).
//!
//! ## Stability
//! This API is **instrumentation only**. It is not part of the engine's public
//! semantics and may change/disappear when the mt9x drift probe series wraps up.

use super::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::tensor::Tensor;
use rnb_loader::GGMLType;

/// Build an in-memory [`QuantizedWeight`] from raw quant bytes for microbench use.
///
/// `bytes` must be exactly the row-major quant payload (no GGUF header). `rows`
/// is `out_features` and `cols` is `in_features`. The function copies `bytes`
/// into an owned [`Tensor`] (via `Tensor::from_slice`) so the caller does not
/// need to keep the input buffer alive — convenient when bytes come from a
/// short-lived `LoadedModel` view.
fn make_weight(bytes: &[u8], ggml_type: GGMLType, rows: usize, cols: usize) -> QuantizedWeight {
    let tensor = Tensor::from_slice::<u8>(bytes, &[bytes.len()]);
    QuantizedWeight::new(tensor, ggml_type, rows, cols)
}

/// Run the production GEMV path on a single row input (`seq_len = 1`).
///
/// On success, writes exactly `rows` `f32` values into the start of `output`.
/// `input.len()` must equal `cols`; `output.len()` must be `>= rows`.
pub fn run_gemv_prod(
    bytes: &[u8],
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    validate_dims(rows, cols, input, output)?;
    let weight = make_weight(bytes, ggml_type, rows, cols);
    weight
        .gemv_into_for_microbench(input, output)
        .map_err(|e| format!("gemv_into (prod) failed: {e:?}"))
}

/// Run the scalar-dequant + f32-reduce path.
pub fn run_gemv_generic(
    bytes: &[u8],
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    validate_dims(rows, cols, input, output)?;
    let weight = make_weight(bytes, ggml_type, rows, cols);
    weight
        .gemv_into_generic_for_microbench(input, output)
        .map_err(|e| format!("gemv_into_generic failed: {e:?}"))
}

/// Run the full-dequant + f64-accumulator path. Single-row only.
pub fn run_gemv_f64_logit(
    bytes: &[u8],
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    validate_dims(rows, cols, input, output)?;
    let weight = make_weight(bytes, ggml_type, rows, cols);
    weight
        .gemv_into_f64_logit_for_microbench(input, output)
        .map_err(|e| format!("gemv_into_f64_logit failed: {e:?}"))
}

fn validate_dims(rows: usize, cols: usize, input: &[f32], output: &[f32]) -> Result<(), String> {
    if input.len() != cols {
        return Err(format!(
            "input length {} does not match cols {cols}",
            input.len()
        ));
    }
    if output.len() < rows {
        return Err(format!(
            "output length {} smaller than rows {rows}",
            output.len()
        ));
    }
    Ok(())
}
