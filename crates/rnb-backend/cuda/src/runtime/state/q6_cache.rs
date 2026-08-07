use super::super::*;

impl CudaState {
    // Descriptor-backed admission lands before product dispatch changes.
    #[allow(dead_code)]
    pub(in crate::runtime) fn resident_q6k_transformed_view_ptrs(
        &mut self,
        view: rnb_backend_api::TransformedWeightView<'_>,
    ) -> Result<Option<(u64, u64, u64)>, String> {
        if view.layout() != rnb_backend_api::TransformedWeightLayout::Q6kPackedQ8dot
            || view.source_quant() != rnb_backend_api::TransformedSourceQuant::DenseQ6k
        {
            return Err(format!(
                "Q6 packed cache requires Q6kPackedQ8dot/DenseQ6k view, got {:?}/{:?}",
                view.layout(),
                view.source_quant()
            ));
        }
        self.resident_q6k_packed_ptrs(view.source_bytes(), view.rows(), view.blocks_per_row())
    }

    pub(in crate::runtime) fn resident_q6k_packed_ptrs(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<(u64, u64, u64)>, String> {
        let key = Q6PackedKey {
            ptr: weights.as_ptr() as usize,
            len: weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q6_packed.get(&key) {
            return Ok(Some((entry.qs_ptr, entry.d_super_ptr, entry.sub_scale_ptr)));
        }

        // cu231: payload 레이아웃으로 직접 pack — triple Vec + GB 급 concat
        // 복사를 제거한다.
        let packed_payload = pack_q6k_payload_for_q8dot(weights, rows, blocks_per_row)?;
        self.resident_q6k_packed_payload_ptrs(weights, &packed_payload, rows, blocks_per_row)
    }

    pub(in crate::runtime) fn resident_q6k_packed_payload_ptrs(
        &mut self,
        source_weights: &[u8],
        packed_payload: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<(u64, u64, u64)>, String> {
        let source_expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(210))
            .ok_or_else(|| {
                format!("Q6 source size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if source_weights.len() != source_expected {
            return Err(format!(
                "Q6 source byte mismatch: got {}, expected {source_expected}",
                source_weights.len()
            ));
        }
        let key = Q6PackedKey {
            ptr: source_weights.as_ptr() as usize,
            len: source_weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q6_packed.get(&key) {
            return Ok(Some((entry.qs_ptr, entry.d_super_ptr, entry.sub_scale_ptr)));
        }

        let q_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(256))
            .ok_or_else(|| {
                format!("Q6 packed q size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        let block_count = rows.checked_mul(blocks_per_row).ok_or_else(|| {
            format!("Q6 packed block count overflow: rows={rows} blocks={blocks_per_row}")
        })?;
        let d_super_bytes = block_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("Q6 packed d_super byte overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        let sub_scale_bytes = block_count.checked_mul(16).ok_or_else(|| {
            format!("Q6 packed sub_scale byte overflow: rows={rows} blocks={blocks_per_row}")
        })?;
        let bytes = q_bytes
            .checked_add(d_super_bytes)
            .and_then(|v| v.checked_add(sub_scale_bytes))
            .ok_or_else(|| "Q6 packed total size overflow".to_string())?;
        super::weight_residency::validate_q6k_packed_payload_bytes_per_block(
            super::weight_residency::Q6K_PACKED_Q8DOT_BYTES_PER_BLOCK,
        )?;
        if packed_payload.len() != bytes {
            return Err(format!(
                "Q6 packed payload byte mismatch: got {}, expected {bytes}",
                packed_payload.len()
            ));
        }
        let effective_limit = self.resident_class_effective_limit(
            self.resident_q6_packed_bytes,
            self.resident_q6_packed_limit,
        );
        if bytes > effective_limit
            || self.resident_q6_packed_bytes.saturating_add(bytes) > effective_limit
            || !self.resident_admission_allowed(bytes)?
        {
            return Ok(None);
        }

        let qs = &packed_payload[..q_bytes];
        let d_super = &packed_payload[q_bytes..q_bytes + d_super_bytes];
        let sub_scale = &packed_payload[q_bytes + d_super_bytes..];
        let qs_ptr = unsafe { self.api.mem_alloc(q_bytes) }?;
        let d_super_ptr = match unsafe { self.api.mem_alloc(d_super_bytes) } {
            Ok(ptr) => ptr,
            Err(err) => {
                let _ = unsafe { self.api.mem_free(qs_ptr) };
                return Err(err);
            }
        };
        let sub_scale_ptr = match unsafe { self.api.mem_alloc(sub_scale_bytes) } {
            Ok(ptr) => ptr,
            Err(err) => {
                let _ = unsafe { self.api.mem_free(qs_ptr) };
                let _ = unsafe { self.api.mem_free(d_super_ptr) };
                return Err(err);
            }
        };
        let upload = unsafe {
            self.api.memcpy_htod_async(
                qs_ptr,
                qs.as_ptr().cast::<libc::c_void>(),
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                d_super_ptr,
                d_super.as_ptr().cast::<libc::c_void>(),
                d_super_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                sub_scale_ptr,
                sub_scale.as_ptr().cast::<libc::c_void>(),
                sub_scale_bytes,
                self.stream,
            )
        };
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(qs_ptr) };
            let _ = unsafe { self.api.mem_free(d_super_ptr) };
            let _ = unsafe { self.api.mem_free(sub_scale_ptr) };
            return Err(err);
        }

        self.resident_q6_packed.insert(
            key,
            ResidentQ6Packed {
                qs_ptr,
                d_super_ptr,
                sub_scale_ptr,
            },
        );
        self.resident_q6_packed_bytes = self.resident_q6_packed_bytes.saturating_add(bytes);
        self.record_packed_q8dot_residency("Q6_K", bytes);
        Ok(Some((qs_ptr, d_super_ptr, sub_scale_ptr)))
    }
}

/// cu230/cu231: packs the canonical 210B Q6_K blocks into the resident
/// payload layout `[qs | d_super(le16) | sub_scale]` in one pass — per-row
/// scoped threads over disjoint output slices (identical bytes regardless of
/// the split — deterministic), no intermediate triple-Vec or concat copy.
pub(in crate::runtime) fn pack_q6k_payload_for_q8dot(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<Vec<u8>, String> {
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(210))
        .ok_or_else(|| format!("Q6 input size overflow: rows={rows} blocks={blocks_per_row}"))?;
    if weights.len() != expected {
        return Err(format!(
            "Q6 packed input byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }

    let block_count = rows * blocks_per_row;
    let q_bytes = block_count * 256;
    let d_bytes = block_count * 2;
    let ss_bytes = block_count * 16;
    let mut payload = vec![0u8; q_bytes + d_bytes + ss_bytes];
    let (qs_all, tail) = payload.split_at_mut(q_bytes);
    let (d_all, ss_all) = tail.split_at_mut(d_bytes);

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    let rows_per = rows.div_ceil(threads.max(1)).max(1);
    let qs_row = blocks_per_row * 256;
    let d_row = blocks_per_row * 2;
    let ss_row = blocks_per_row * 16;
    std::thread::scope(|scope| {
        let mut qs_rest = qs_all;
        let mut d_rest = d_all;
        let mut ss_rest = ss_all;
        let mut row_start = 0usize;
        while row_start < rows {
            let chunk_rows = rows_per.min(rows - row_start);
            let (qs_chunk, qs_next) = qs_rest.split_at_mut(chunk_rows * qs_row);
            let (d_chunk, d_next) = d_rest.split_at_mut(chunk_rows * d_row);
            let (ss_chunk, ss_next) = ss_rest.split_at_mut(chunk_rows * ss_row);
            qs_rest = qs_next;
            d_rest = d_next;
            ss_rest = ss_next;
            let base_row = row_start;
            scope.spawn(move || {
                for local_row in 0..chunk_rows {
                    let row = base_row + local_row;
                    for block_idx in 0..blocks_per_row {
                        let src_base = (row * blocks_per_row + block_idx) * 210;
                        let block = &weights[src_base..src_base + 210];
                        let d_base = (local_row * blocks_per_row + block_idx) * 2;
                        d_chunk[d_base] = block[208];
                        d_chunk[d_base + 1] = block[209];

                        let sub_scale_base = (local_row * blocks_per_row + block_idx) * 16;
                        ss_chunk[sub_scale_base..sub_scale_base + 16]
                            .copy_from_slice(&block[192..208]);

                        let q_base = (local_row * blocks_per_row + block_idx) * 256;
                        for tid in 0..256usize {
                            let n = tid >> 7;
                            let rem = tid & 127;
                            let l = rem & 31;
                            let ql_base = n * 64;
                            let qh_base = 128 + n * 32;
                            let qh = block[qh_base + l];
                            let q = if rem < 32 {
                                (block[ql_base + l] & 0x0f) | (((qh >> 0) & 3) << 4)
                            } else if rem < 64 {
                                (block[ql_base + l + 32] & 0x0f) | (((qh >> 2) & 3) << 4)
                            } else if rem < 96 {
                                (block[ql_base + l] >> 4) | (((qh >> 4) & 3) << 4)
                            } else {
                                (block[ql_base + l + 32] >> 4) | (((qh >> 6) & 3) << 4)
                            };
                            qs_chunk[q_base + tid] = ((q as i16 - 32) as i8) as u8;
                        }
                    }
                }
            });
            row_start += chunk_rows;
        }
    });
    Ok(payload)
}

/// Test-only tuple view of the payload pack (the payload layout is the
/// production contract; tests that want the split buffers slice it apart).
#[cfg(test)]
pub(in crate::runtime) fn pack_q6k_for_q8dot(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<(Vec<i8>, Vec<u16>, Vec<i8>), String> {
    let payload = pack_q6k_payload_for_q8dot(weights, rows, blocks_per_row)?;
    let block_count = rows * blocks_per_row;
    let q_bytes = block_count * 256;
    let d_bytes = block_count * 2;
    let qs = payload[..q_bytes]
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let d_super = payload[q_bytes..q_bytes + d_bytes]
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    let sub_scale = payload[q_bytes + d_bytes..]
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    Ok((qs, d_super, sub_scale))
}
