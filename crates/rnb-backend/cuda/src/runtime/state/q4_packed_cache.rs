use super::super::*;

impl CudaState {
    // Descriptor-backed admission lands before product dispatch changes.
    #[allow(dead_code)]
    pub(in crate::runtime) fn resident_q4k_transformed_view_ptr(
        &mut self,
        view: rnb_backend_api::TransformedWeightView<'_>,
    ) -> Result<Option<u64>, String> {
        if view.layout() != rnb_backend_api::TransformedWeightLayout::Q4kCompactMetadata
            || view.source_quant() != rnb_backend_api::TransformedSourceQuant::DenseQ4kRowPair
        {
            return Err(format!(
                "Q4 packed cache requires Q4kCompactMetadata/DenseQ4kRowPair view, got {:?}/{:?}",
                view.layout(),
                view.source_quant()
            ));
        }
        self.resident_q4k_packed_ptrs(view.source_bytes(), view.rows(), view.blocks_per_row())
    }

    pub(in crate::runtime) fn resident_q4k_packed_ptrs(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        let key = Q4PackedKey {
            ptr: weights.as_ptr() as usize,
            len: weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q4_packed.get(&key) {
            return Ok(Some(entry.ptr));
        }

        let packed = pack_q4k_for_q8dot(weights, rows, blocks_per_row)?;
        self.resident_q4k_packed_payload_ptr(weights, &packed, rows, blocks_per_row)
    }

    pub(in crate::runtime) fn resident_q4k_packed_payload_ptr(
        &mut self,
        source_weights: &[u8],
        packed_payload: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        let source_expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!("Q4 source size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if source_weights.len() != source_expected {
            return Err(format!(
                "Q4 source byte mismatch: got {}, expected {source_expected}",
                source_weights.len()
            ));
        }
        let key = Q4PackedKey {
            ptr: source_weights.as_ptr() as usize,
            len: source_weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q4_packed.get(&key) {
            return Ok(Some(entry.ptr));
        }

        super::weight_residency::validate_q4k_packed_payload_bytes_per_block(
            super::weight_residency::Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK,
        )?;
        let bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(super::weight_residency::Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK))
            .ok_or_else(|| {
                format!("Q4 packed size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if packed_payload.len() != bytes {
            return Err(format!(
                "Q4 packed payload byte mismatch: got {}, expected {bytes}",
                packed_payload.len()
            ));
        }
        let effective_limit = self.resident_class_effective_limit(
            self.resident_q4_packed_bytes,
            self.resident_q4_packed_limit,
        );
        if bytes > effective_limit
            || self.resident_q4_packed_bytes.saturating_add(bytes) > effective_limit
            || !self.resident_admission_allowed(bytes)?
        {
            return Ok(None);
        }

        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let upload = unsafe {
            self.api.memcpy_htod_async(
                ptr,
                packed_payload.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )
        };
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }

        self.resident_q4_packed
            .insert(key, ResidentQ4Packed { ptr });
        self.resident_q4_packed_bytes = self.resident_q4_packed_bytes.saturating_add(bytes);
        self.record_packed_q8dot_residency("Q4_K", bytes);
        Ok(Some(ptr))
    }
}

pub(in crate::runtime) fn pack_q4k_for_q8dot(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<Vec<u8>, String> {
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| format!("Q4 input size overflow: rows={rows} blocks={blocks_per_row}"))?;
    if weights.len() != expected {
        return Err(format!(
            "Q4 packed input byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }

    let mut packed = vec![
        0u8;
        rows * blocks_per_row
            * super::weight_residency::Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK
    ];
    for row in 0..rows {
        for block_idx in 0..blocks_per_row {
            let src_base = (row * blocks_per_row + block_idx) * 144;
            let block = &weights[src_base..src_base + 144];
            let dst_base = (row * blocks_per_row + block_idx) * 148;
            let dst = &mut packed[dst_base..dst_base + 148];
            dst[0..4].copy_from_slice(&block[0..4]);
            for j in 0..8usize {
                let (sc, mn) = if j < 4 {
                    (block[4 + j] & 63, block[4 + j + 4] & 63)
                } else {
                    let sc = (block[4 + j + 4] & 0x0f) | ((block[4 + j - 4] >> 6) << 4);
                    let mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
                    (sc, mn)
                };
                let scmn = (sc as u16) | ((mn as u16) << 8);
                dst[4 + j * 2..6 + j * 2].copy_from_slice(&scmn.to_le_bytes());
            }
            dst[20..148].copy_from_slice(&block[16..144]);
        }
    }
    Ok(packed)
}
