//! AArch64 quantized GEMV dispatch paths.

use super::*;
use crate::engine::policy;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
struct PackedDispatchCounters {
    no_data: AtomicU64,
    unsupported_type: AtomicU64,
    early_exit_decode_off: AtomicU64,
    q4k_compact_seq1: AtomicU64,
    q4k_compact_len_seq1: AtomicU64,
    q4k_generic_tile_gemv_seq1: AtomicU64,
    q4k_generic_tile_gemm_batch: AtomicU64,
    q5k_seq1: AtomicU64,
    q5k_batch: AtomicU64,
    q6k_seq1: AtomicU64,
    q6k_batch: AtomicU64,
}

static PACKED_COUNTERS: PackedDispatchCounters = PackedDispatchCounters {
    no_data: AtomicU64::new(0),
    unsupported_type: AtomicU64::new(0),
    early_exit_decode_off: AtomicU64::new(0),
    q4k_compact_seq1: AtomicU64::new(0),
    q4k_compact_len_seq1: AtomicU64::new(0),
    q4k_generic_tile_gemv_seq1: AtomicU64::new(0),
    q4k_generic_tile_gemm_batch: AtomicU64::new(0),
    q5k_seq1: AtomicU64::new(0),
    q5k_batch: AtomicU64::new(0),
    q6k_seq1: AtomicU64::new(0),
    q6k_batch: AtomicU64::new(0),
};

pub fn packed_dispatch_report() -> Option<String> {
    let c = &PACKED_COUNTERS;
    let total: u64 = c.no_data.load(Ordering::Relaxed)
        + c.unsupported_type.load(Ordering::Relaxed)
        + c.early_exit_decode_off.load(Ordering::Relaxed)
        + c.q4k_compact_seq1.load(Ordering::Relaxed)
        + c.q4k_compact_len_seq1.load(Ordering::Relaxed)
        + c.q4k_generic_tile_gemv_seq1.load(Ordering::Relaxed)
        + c.q4k_generic_tile_gemm_batch.load(Ordering::Relaxed)
        + c.q5k_seq1.load(Ordering::Relaxed)
        + c.q5k_batch.load(Ordering::Relaxed)
        + c.q6k_seq1.load(Ordering::Relaxed)
        + c.q6k_batch.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let mut out = String::from("=== packed dispatch counters ===\n");
    out.push_str(&format!(
        "no_data={} unsupported_type={} early_exit_decode_off={}\n",
        c.no_data.load(Ordering::Relaxed),
        c.unsupported_type.load(Ordering::Relaxed),
        c.early_exit_decode_off.load(Ordering::Relaxed),
    ));
    out.push_str(&format!(
        "q4k_compact_seq1={} q4k_compact_len_seq1={} q4k_generic_tile_gemv_seq1={} q4k_generic_tile_gemm_batch={}\n",
        c.q4k_compact_seq1.load(Ordering::Relaxed),
        c.q4k_compact_len_seq1.load(Ordering::Relaxed),
        c.q4k_generic_tile_gemv_seq1.load(Ordering::Relaxed),
        c.q4k_generic_tile_gemm_batch.load(Ordering::Relaxed),
    ));
    out.push_str(&format!(
        "q5k_seq1={} q5k_batch={} q6k_seq1={} q6k_batch={}\n",
        c.q5k_seq1.load(Ordering::Relaxed),
        c.q5k_batch.load(Ordering::Relaxed),
        c.q6k_seq1.load(Ordering::Relaxed),
        c.q6k_batch.load(Ordering::Relaxed),
    ));
    Some(out)
}

pub fn reset_packed_dispatch() {
    let c = &PACKED_COUNTERS;
    c.no_data.store(0, Ordering::Relaxed);
    c.unsupported_type.store(0, Ordering::Relaxed);
    c.early_exit_decode_off.store(0, Ordering::Relaxed);
    c.q4k_compact_seq1.store(0, Ordering::Relaxed);
    c.q4k_compact_len_seq1.store(0, Ordering::Relaxed);
    c.q4k_generic_tile_gemv_seq1.store(0, Ordering::Relaxed);
    c.q4k_generic_tile_gemm_batch.store(0, Ordering::Relaxed);
    c.q5k_seq1.store(0, Ordering::Relaxed);
    c.q5k_batch.store(0, Ordering::Relaxed);
    c.q6k_seq1.store(0, Ordering::Relaxed);
    c.q6k_batch.store(0, Ordering::Relaxed);
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q80_pair_packed_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> bool {
    if weight.ggml_type != GGMLType::Q8_0 || seq_len != 1 {
        return false;
    }
    if !q80_pair_i8mm_supported(weight.cols) {
        return false;
    }
    if let Some((ptr, len)) = weight.arch.q80_pair_packed_data {
        let packed = unsafe { std::slice::from_raw_parts(ptr, len) };
        gemv_q8_0_packed_i8mm(packed, q8, output, weight.rows, weight.cols);
        return true;
    }

    if bytes_per_row != (weight.cols / 32) * 34 {
        return false;
    }
    let Some(packed) = weight
        .arch
        .q80_pair_packed
        .get_or_init(|| Some(pack_q8_0_row_pairs(bytes, weight.rows, bytes_per_row)))
        .as_ref()
    else {
        return false;
    };
    gemv_q8_0_packed_i8mm(packed, q8, output, weight.rows, weight.cols);
    true
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q80_f32_scale_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> bool {
    let Some(scales) = q80_f32_scales(weight, bytes, bytes_per_row) else {
        return false;
    };
    gemv_q8_0_int8_f32_scales(
        bytes,
        scales,
        q8,
        output,
        weight.rows,
        weight.cols,
        seq_len,
        bytes_per_row,
    );
    true
}

#[cfg(target_arch = "aarch64")]
fn q80_f32_scales<'a>(
    weight: &'a QuantizedWeight,
    bytes: &[u8],
    bytes_per_row: usize,
) -> Option<&'a [f32]> {
    if weight.ggml_type != GGMLType::Q8_0 || bytes_per_row != (weight.cols / 32) * 34 {
        return None;
    }
    weight
        .arch
        .q80_f32_scales
        .get_or_init(|| {
            if !q80_f32_scales_requested() {
                return None;
            }
            let n_blocks = weight.cols / 32;
            let mut scales = vec![0.0f32; weight.rows * n_blocks];
            for row in 0..weight.rows {
                let row_off = row * bytes_per_row;
                for bi in 0..n_blocks {
                    let off = row_off + bi * 34;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    scales[row * n_blocks + bi] = half::f16::from_bits(bits).to_f32();
                }
            }
            Some(scales)
        })
        .as_deref()
}

#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_packed_q8k_gemv(
    weight: &QuantizedWeight,
    q8k: &[Q8KBlock],
    output: &mut [f32],
) -> crate::error::Result<bool> {
    let Some((packed_ptr, packed_len)) = weight.packed_gemm_data else {
        PACKED_COUNTERS.no_data.fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    };
    if !matches!(
        weight.ggml_type,
        GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
    ) {
        PACKED_COUNTERS
            .unsupported_type
            .fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }

    let n_blocks = weight.cols / 256;
    let seq_len = q8k.len() / n_blocks;
    let required = seq_len * weight.rows;
    assert!(
        output.len() >= required,
        "gemv_packed_q8k_into: output buffer size mismatch"
    );
    let output = &mut output[..required];
    output.fill(0.0);

    if seq_len == 1 && !policy::packed_decode_enabled() && has_full_raw_quantized_bytes(weight) {
        PACKED_COUNTERS
            .early_exit_decode_off
            .fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }

    let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) };
    if weight.ggml_type == GGMLType::Q4_K {
        if weight.has_packed_q4k_compact_format() && seq_len == 1 {
            PACKED_COUNTERS
                .q4k_compact_seq1
                .fetch_add(1, Ordering::Relaxed);
            gemm_runtime::neon_repacked::gemv_q4k_compact(
                packed,
                q8k,
                output,
                weight.rows,
                weight.cols,
            );
            return Ok(true);
        }
        if weight.has_packed_q4k_raw_meta_format() {
            if seq_len == 1 {
                gemv_q4_k_int8_raw_meta_interleaved(
                    packed,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                );
            } else if runtime_rawmeta_repack_enabled(
                weight.packed_gemm_quant_type,
                weight.ggml_type,
                seq_len,
                weight.rows,
                weight.cols,
            ) {
                let Some(runtime_packed) = weight
                    .arch
                    .rawmeta_runtime_packed
                    .get_or_init(|| {
                        Some(gemm_runtime::pack_q4k::pack_q4k_from_raw_meta(
                            packed,
                            weight.rows,
                            n_blocks,
                        ))
                    })
                    .as_ref()
                else {
                    return Ok(false);
                };
                let (qs_flat, d_flat, bsums_flat) = flatten_q8k_blocks(q8k);
                gemm_runtime::tile_q4k::gemm_q4k_packed(
                    runtime_packed,
                    &qs_flat,
                    &d_flat,
                    &bsums_flat,
                    output,
                    weight.rows,
                    n_blocks,
                    seq_len,
                );
            } else {
                let (qs_flat, d_flat, bsums_flat) = flatten_q8k_blocks(q8k);
                gemm_runtime::tile_q4k::gemm_q4k_raw_meta(
                    packed,
                    &qs_flat,
                    &d_flat,
                    &bsums_flat,
                    output,
                    weight.rows,
                    n_blocks,
                    seq_len,
                );
            }
            return Ok(true);
        }
        let raw_meta_len =
            weight.rows * n_blocks * gemm_runtime::pack_q4k::Q4K_RAW_META_BLOCK_BYTES;
        if weight.packed_gemm_quant_type.is_none() && packed_len == raw_meta_len {
            if seq_len == 1 {
                gemv_q4_k_int8_raw_meta_interleaved(
                    packed,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                );
            } else {
                let (qs_flat, d_flat, bsums_flat) = flatten_q8k_blocks(q8k);
                gemm_runtime::tile_q4k::gemm_q4k_raw_meta(
                    packed,
                    &qs_flat,
                    &d_flat,
                    &bsums_flat,
                    output,
                    weight.rows,
                    n_blocks,
                    seq_len,
                );
            }
            return Ok(true);
        }
    }

    if weight.ggml_type == GGMLType::Q4_K
        && seq_len == 1
        && weight.packed_gemm_quant_type.is_none()
        && packed_len
            == weight.rows.div_ceil(8) * n_blocks * gemm_runtime::pack_q4k::Q4K_COMPACT_BLOCK_BYTES
    {
        PACKED_COUNTERS
            .q4k_compact_len_seq1
            .fetch_add(1, Ordering::Relaxed);
        gemm_runtime::neon_repacked::gemv_q4k_compact(
            packed,
            q8k,
            output,
            weight.rows,
            weight.cols,
        );
        return Ok(true);
    }

    let (qs_flat, d_flat, bsums_flat) = flatten_q8k_blocks(q8k);
    match weight.ggml_type {
        GGMLType::Q4_K if seq_len == 1 => {
            PACKED_COUNTERS
                .q4k_generic_tile_gemv_seq1
                .fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q4k::gemv_q4k_packed(
                packed,
                &qs_flat,
                &d_flat,
                &bsums_flat,
                output,
                weight.rows,
                n_blocks,
            );
        }
        GGMLType::Q4_K => {
            PACKED_COUNTERS
                .q4k_generic_tile_gemm_batch
                .fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q4k::gemm_q4k_packed(
                packed,
                &qs_flat,
                &d_flat,
                &bsums_flat,
                output,
                weight.rows,
                n_blocks,
                seq_len,
            );
        }
        GGMLType::Q5_K if seq_len == 1 => {
            PACKED_COUNTERS.q5k_seq1.fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q5k::gemv_q5k_packed(
                packed,
                &qs_flat,
                &d_flat,
                &bsums_flat,
                output,
                weight.rows,
                n_blocks,
            );
        }
        GGMLType::Q5_K => {
            PACKED_COUNTERS.q5k_batch.fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q5k::gemm_q5k_packed(
                packed,
                &qs_flat,
                &d_flat,
                &bsums_flat,
                output,
                weight.rows,
                n_blocks,
                seq_len,
            );
        }
        GGMLType::Q6_K if seq_len == 1 => {
            PACKED_COUNTERS.q6k_seq1.fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q6k::gemv_q6k_packed(
                packed,
                &qs_flat,
                &d_flat,
                output,
                weight.rows,
                n_blocks,
            );
        }
        GGMLType::Q6_K => {
            PACKED_COUNTERS.q6k_batch.fetch_add(1, Ordering::Relaxed);
            gemm_runtime::tile_q6k::gemm_q6k_packed(
                packed,
                &qs_flat,
                &d_flat,
                output,
                weight.rows,
                n_blocks,
                seq_len,
            );
        }
        _ => unreachable!(),
    }

    Ok(true)
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_q8k_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8k: &[Q8KBlock],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<()> {
    if weight.ggml_type == GGMLType::Q4_K {
        let q4k_backend = q4k_kernel_backend_from_env(policy::q4k_kernel_backend().as_deref());
        if let Some(backend) = q4k_backend.filter(|_| has_full_raw_quantized_bytes(weight)) {
            dispatch_q4k_kernel_backend(
                backend,
                bytes,
                weight.arch.repacked.as_ref(),
                q8k,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            return Ok(());
        }
    }

    if dispatch_packed_q8k_gemv(weight, q8k, output)? {
        return Ok(());
    }

    match weight.ggml_type {
        GGMLType::Q4_K => {
            if let Some(ref rpk) = weight.arch.repacked {
                gemm_runtime::neon_repacked::gemv_q4k_repacked(
                    rpk,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                    bytes,
                    bytes_per_row,
                );
            } else {
                gemv_q4_k_int8(
                    bytes,
                    q8k,
                    output,
                    weight.rows,
                    weight.cols,
                    seq_len,
                    bytes_per_row,
                );
            }
        }
        GGMLType::Q5_K => gemv_q5_k_int8(
            bytes,
            q8k,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            bytes_per_row,
        ),
        GGMLType::Q6_K => gemv_q6_k_int8(
            bytes,
            q8k,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            bytes_per_row,
        ),
        _ => {
            return Err(crate::error::LlmError::Forward(format!(
                "dispatch_q8k_gemv: unsupported type {:?}",
                weight.ggml_type
            )))
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_q8_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    q8: &[Q8Block],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<()> {
    if let Some(ref q4_0) = weight.q4_0_data {
        let q4_0_bytes_per_row = (weight.cols / 32) * 18;
        gemv_q4_0_int8(
            q4_0,
            q8,
            output,
            weight.rows,
            weight.cols,
            seq_len,
            q4_0_bytes_per_row,
        );
        return Ok(());
    }

    match weight.ggml_type {
        GGMLType::Q4_0 => {
            gemv_q4_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        GGMLType::Q5_0 => {
            gemv_q5_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        GGMLType::Q8_0 => {
            if dispatch_q80_pair_packed_gemv(weight, bytes, q8, output, seq_len, bytes_per_row) {
                return Ok(());
            }
            if dispatch_q80_f32_scale_gemv(weight, bytes, q8, output, seq_len, bytes_per_row) {
                return Ok(());
            }
            gemv_q8_0_int8(
                bytes,
                q8,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            Ok(())
        }
        _ => Err(crate::error::LlmError::Forward(format!(
            "dispatch_q8_gemv: unsupported type {:?}",
            weight.ggml_type
        ))),
    }
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_vec_fast_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if weight.ggml_type == GGMLType::Q4_K {
        let q4k_backend = q4k_kernel_backend_from_env(policy::q4k_kernel_backend().as_deref());
        if let Some(backend) = q4k_backend.filter(|_| has_full_raw_quantized_bytes(weight)) {
            let q8k = quantize_input_q8k(input);
            dispatch_q4k_kernel_backend(
                backend,
                bytes,
                weight.arch.repacked.as_ref(),
                &q8k,
                output,
                weight.rows,
                weight.cols,
                seq_len,
                bytes_per_row,
            );
            return Ok(true);
        }
    }

    if weight.q4_0_data.is_some() && fast_dotprod_enabled() {
        let q8 = quantize_input_q8(input);
        dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
        return Ok(true);
    }

    if weight.packed_gemm_data.is_some()
        && matches!(
            weight.ggml_type,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        && weight.cols % 256 == 0
    {
        let q8k = quantize_input_q8k(input);
        if dispatch_packed_q8k_gemv(weight, &q8k, output)? {
            return Ok(true);
        }
    }

    dispatch_dotprod_f32_input_gemv(weight, bytes, input, output, seq_len, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
pub(in crate::engine) fn dispatch_into_fast_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if weight.packed_gemm_data.is_some()
        && matches!(
            weight.ggml_type,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        && weight.cols % 256 == 0
    {
        let q8k = quantize_input_q8k(input);
        if dispatch_packed_q8k_gemv(weight, &q8k, output)? {
            return Ok(true);
        }
    }

    if weight.q4_0_data.is_some() && fast_dotprod_enabled() {
        let q8 = quantize_input_q8(input);
        dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
        return Ok(true);
    }

    dispatch_dotprod_f32_input_gemv(weight, bytes, input, output, seq_len, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_dotprod_f32_input_gemv(
    weight: &QuantizedWeight,
    bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    bytes_per_row: usize,
) -> crate::error::Result<bool> {
    if !fast_dotprod_enabled() {
        return Ok(false);
    }

    match weight.ggml_type {
        GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K => {
            let q8k = quantize_input_q8k(input);
            dispatch_q8k_gemv(weight, bytes, &q8k, output, seq_len, bytes_per_row)?;
            Ok(true)
        }
        GGMLType::Q4_0 | GGMLType::Q5_0 | GGMLType::Q8_0 => {
            let q8 = quantize_input_q8(input);
            dispatch_q8_gemv(weight, bytes, &q8, output, seq_len, bytes_per_row)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn has_full_raw_quantized_bytes(weight: &QuantizedWeight) -> bool {
    weight.data.as_bytes().is_some_and(|bytes| {
        expected_quantized_byte_len(weight.ggml_type, weight.rows, weight.cols) == Some(bytes.len())
    })
}
#[cfg(target_arch = "aarch64")]
pub(super) fn dispatch_q4k_kernel_backend(
    backend: Q4KKernelBackend,
    bytes: &[u8],
    repacked: Option<&memmap2::Mmap>,
    q8k: &[Q8KBlock],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
    bytes_per_row: usize,
) {
    match backend {
        Q4KKernelBackend::Builtin => {
            if seq_len > 1 {
                if let Some(rpk) = repacked {
                    gemm_runtime::neon_repacked::gemm_q4k_repacked(
                        rpk,
                        q8k,
                        output,
                        rows,
                        cols,
                        seq_len,
                        bytes,
                        bytes_per_row,
                    );
                } else {
                    gemv_q4_k_int8(bytes, q8k, output, rows, cols, seq_len, bytes_per_row);
                }
            } else {
                gemv_q4_k_int8(bytes, q8k, output, rows, cols, seq_len, bytes_per_row);
            }
        }
    }
}
