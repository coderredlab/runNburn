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

        let (qs, d_super, sub_scale) = pack_q6k_for_q8dot(weights, rows, blocks_per_row)?;
        let mut packed_payload = Vec::with_capacity(
            qs.len()
                + d_super.len() * std::mem::size_of::<u16>()
                + sub_scale.len() * std::mem::size_of::<i8>(),
        );
        // cu230: byte-map extend 는 GB 급 payload 에서 직렬 byte 루프가 된다 —
        // i8→u8 은 layout 동일 reinterpret, u16 은 지원 타깃(x86_64/aarch64)
        // 전부 little-endian 이라 native byte order == LE 로 한 번에 복사한다.
        packed_payload.extend_from_slice(unsafe {
            std::slice::from_raw_parts(qs.as_ptr().cast::<u8>(), qs.len())
        });
        packed_payload.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                d_super.as_ptr().cast::<u8>(),
                d_super.len() * std::mem::size_of::<u16>(),
            )
        });
        packed_payload.extend_from_slice(unsafe {
            std::slice::from_raw_parts(sub_scale.as_ptr().cast::<u8>(), sub_scale.len())
        });
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

pub(in crate::runtime) fn pack_q6k_for_q8dot(
    weights: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<(Vec<i8>, Vec<u16>, Vec<i8>), String> {
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

    let mut qs = vec![0i8; rows * blocks_per_row * 256];
    let mut d_super = vec![0u16; rows * blocks_per_row];
    let mut sub_scale = vec![0i8; rows * blocks_per_row * 16];
    // cu230: per-element 분기 루프라 GB 급 텐서에서 직렬로 수 초를 먹는다 —
    // row 단위로 출력이 disjoint 하므로 scoped thread 로 분할한다 (출력 바이트
    // 는 분할과 무관하게 동일 — 결정적).
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    let rows_per = rows.div_ceil(threads.max(1)).max(1);
    let qs_row = blocks_per_row * 256;
    let ds_row = blocks_per_row;
    let ss_row = blocks_per_row * 16;
    std::thread::scope(|scope| {
        let mut qs_rest = qs.as_mut_slice();
        let mut d_rest = d_super.as_mut_slice();
        let mut ss_rest = sub_scale.as_mut_slice();
        let mut row_start = 0usize;
        while row_start < rows {
            let chunk_rows = rows_per.min(rows - row_start);
            let (qs_chunk, qs_next) = qs_rest.split_at_mut(chunk_rows * qs_row);
            let (d_chunk, d_next) = d_rest.split_at_mut(chunk_rows * ds_row);
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
                        let raw_d = u16::from_le_bytes([block[208], block[209]]);
                        d_chunk[local_row * ds_row + block_idx] = raw_d;

                        let sub_scale_base = (local_row * blocks_per_row + block_idx) * 16;
                        for i in 0..16 {
                            ss_chunk[sub_scale_base + i] = block[192 + i] as i8;
                        }

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
                            qs_chunk[q_base + tid] = (q as i16 - 32) as i8;
                        }
                    }
                }
            });
            row_start += chunk_rows;
        }
    });
    Ok((qs, d_super, sub_scale))
}
