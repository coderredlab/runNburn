use super::super::dequant::dequantize_bytes_to_f32;
use super::super::packed_runtime::PackedModel;
use super::super::policy;
use super::super::quantized_weight_types::QuantizedWeight;
use rnb_core::tensor::Tensor;

pub(super) fn build_tied_output_weight(token_embd: &QuantizedWeight) -> QuantizedWeight {
    // weight tying: quantized token_embd -> Q8_0 conversion (chunk streaming)
    let rows = token_embd.rows;
    let cols = token_embd.cols;

    if policy::tied_output_q8_disabled() {
        eprintln!(
            "[INFO] output: reusing token_embd {:?} without Q8_0 conversion",
            token_embd.ggml_type
        );
        return QuantizedWeight::new(token_embd.data.clone(), token_embd.ggml_type, rows, cols);
    }

    let blocks_per_row = cols / 32;
    let bytes_per_block = 34usize;
    let total_bytes = rows * blocks_per_row * bytes_per_block;
    let embd_ggml = token_embd.ggml_type;
    let embd_bytes = token_embd
        .data
        .as_bytes()
        .expect("token_embd must have bytes");
    let embd_row_bytes = embd_bytes.len() / rows;

    let tmp_file = tempfile::tempfile().expect("failed to create tempfile for Q8_0");
    tmp_file
        .set_len(total_bytes as u64)
        .expect("failed to set tempfile size");
    let mut q8_mmap = unsafe { memmap2::MmapMut::map_mut(&tmp_file).expect("mmap failed") };

    let chunk_rows = 1024usize;
    {
        let q8_bytes = &mut q8_mmap[..];
        let mut row = 0usize;
        while row < rows {
            let end = (row + chunk_rows).min(rows);
            let chunk_data = &embd_bytes[row * embd_row_bytes..end * embd_row_bytes];
            let f32_chunk = dequantize_bytes_to_f32(chunk_data, embd_ggml);

            for r in 0..(end - row) {
                let abs_row = row + r;
                for blk in 0..blocks_per_row {
                    let src_off = r * cols + blk * 32;
                    let dst_off = (abs_row * blocks_per_row + blk) * bytes_per_block;
                    let chunk = &f32_chunk[src_off..src_off + 32];

                    let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

                    let d_f16 = half::f16::from_f32(d);
                    q8_bytes[dst_off..dst_off + 2].copy_from_slice(&d_f16.to_le_bytes());

                    for i in 0..32 {
                        let q = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
                        q8_bytes[dst_off + 2 + i] = q as u8;
                    }
                }
            }
            row = end;
        }
    }

    eprintln!(
        "[INFO] output: token_embd {:?} → Q8_0 ({:.1}MB → {:.1}MB, chunk streaming)",
        embd_ggml,
        (rows * cols * 4) as f64 / 1e6,
        total_bytes as f64 / 1e6
    );

    let q8_mmap_ro = q8_mmap
        .make_read_only()
        .expect("failed to make mmap read-only");
    let q8_storage = std::sync::Arc::new(rnb_core::tensor::storage::Storage::Mmap(q8_mmap_ro));
    let q8_tensor = Tensor::from_mmap(q8_storage, 0, &[total_bytes], rnb_core::tensor::DType::U8)
        .expect("failed to create mmap tensor");
    QuantizedWeight::new_q80_with_load_time_packs(q8_tensor, rows, cols, total_bytes)
}

pub(super) fn load_packed_output_weight(
    _packed_model: Option<&PackedModel>,
    token_embd: &QuantizedWeight,
) -> QuantizedWeight {
    // Q80Tile8 synthetic tied-output mmap encoder was removed when standalone
    // .rnb was deprecated. The cache-side `output.weight` is preserved as the
    // original quantized tensor by the v3 sidecar, so packed-model dispatch
    // falls back to runtime tied-output conversion.
    build_tied_output_weight(token_embd)
}
