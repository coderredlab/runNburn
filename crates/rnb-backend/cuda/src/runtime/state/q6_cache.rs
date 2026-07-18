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
        packed_payload.extend(qs.iter().map(|value| *value as u8));
        for value in d_super {
            packed_payload.extend_from_slice(&value.to_le_bytes());
        }
        packed_payload.extend(sub_scale.iter().map(|value| *value as u8));
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
        if bytes > self.resident_q6_packed_limit
            || self.resident_q6_packed_bytes.saturating_add(bytes) > self.resident_q6_packed_limit
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

    pub(in crate::runtime) fn resident_q6k_sidecar_packed_ptrs(
        &mut self,
        source_weights: &[u8],
        sidecar_packed: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<(u64, u64, u64)>, String> {
        let payload = sidecar_q6k_row_pair_to_q8dot_payload(sidecar_packed, rows, blocks_per_row)?;
        self.resident_q6k_packed_payload_ptrs(source_weights, &payload, rows, blocks_per_row)
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
    for row in 0..rows {
        for block_idx in 0..blocks_per_row {
            let src_base = (row * blocks_per_row + block_idx) * 210;
            let block = &weights[src_base..src_base + 210];
            let raw_d = u16::from_le_bytes([block[208], block[209]]);
            d_super[row * blocks_per_row + block_idx] = raw_d;

            let sub_scale_base = (row * blocks_per_row + block_idx) * 16;
            for i in 0..16 {
                sub_scale[sub_scale_base + i] = block[192 + i] as i8;
            }

            let q_base = (row * blocks_per_row + block_idx) * 256;
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
                qs[q_base + tid] = (q as i16 - 32) as i8;
            }
        }
    }
    Ok((qs, d_super, sub_scale))
}

const SIDECAR_Q6K_QS_OFF: usize = 0;
const SIDECAR_Q6K_SC_RAW_OFF: usize = 2048;
const SIDECAR_Q6K_D_OFF: usize = SIDECAR_Q6K_SC_RAW_OFF + 128;
const SIDECAR_Q6K_BLOCK_BYTES: usize = SIDECAR_Q6K_D_OFF + 32;

fn read_sidecar_q6k_row_qs(block: &[u8], nr: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    let pair = nr / 2;
    let odd = nr % 2;
    let pair_base = SIDECAR_Q6K_QS_OFF + pair * 512;
    for chunk in 0..32usize {
        let chunk_off = pair_base + chunk * 16 + odd * 8;
        out[chunk * 8..chunk * 8 + 8].copy_from_slice(&block[chunk_off..chunk_off + 8]);
    }
    out
}

pub(in crate::runtime) fn sidecar_q6k_row_pair_to_q8dot_payload(
    sidecar_packed: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<Vec<u8>, String> {
    let row_groups = rows.div_ceil(8);
    let expected = row_groups
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(SIDECAR_Q6K_BLOCK_BYTES))
        .ok_or_else(|| {
            format!("Q6 sidecar packed size overflow: rows={rows} blocks={blocks_per_row}")
        })?;
    if sidecar_packed.len() != expected {
        return Err(format!(
            "Q6 sidecar packed byte mismatch: got {}, expected {expected}",
            sidecar_packed.len()
        ));
    }

    let q_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(256))
        .ok_or_else(|| format!("Q6 q8dot q size overflow: rows={rows} blocks={blocks_per_row}"))?;
    let block_count = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        format!("Q6 q8dot block count overflow: rows={rows} blocks={blocks_per_row}")
    })?;
    let d_super_bytes = block_count
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| {
            format!("Q6 q8dot d_super byte overflow: rows={rows} blocks={blocks_per_row}")
        })?;
    let sub_scale_bytes = block_count.checked_mul(16).ok_or_else(|| {
        format!("Q6 q8dot sub_scale byte overflow: rows={rows} blocks={blocks_per_row}")
    })?;
    let mut payload = vec![0u8; q_bytes + d_super_bytes + sub_scale_bytes];
    let (qs_payload, rest) = payload.split_at_mut(q_bytes);
    let (d_super_payload, sub_scale_payload) = rest.split_at_mut(d_super_bytes);

    for row in 0..rows {
        let group = row / 8;
        let nr = row % 8;
        for block_idx in 0..blocks_per_row {
            let sidecar_base = (group * blocks_per_row + block_idx) * SIDECAR_Q6K_BLOCK_BYTES;
            let sidecar_block =
                &sidecar_packed[sidecar_base..sidecar_base + SIDECAR_Q6K_BLOCK_BYTES];
            let block_linear = row * blocks_per_row + block_idx;

            let qs = read_sidecar_q6k_row_qs(sidecar_block, nr);
            let q_base = block_linear * 256;
            qs_payload[q_base..q_base + 256].copy_from_slice(&qs);

            let d = f32::from_le_bytes(
                sidecar_block[SIDECAR_Q6K_D_OFF + nr * 4..SIDECAR_Q6K_D_OFF + nr * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            d_super_payload[block_linear * 2..block_linear * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(d).to_le_bytes());

            let scale = &sidecar_block
                [SIDECAR_Q6K_SC_RAW_OFF + nr * 16..SIDECAR_Q6K_SC_RAW_OFF + nr * 16 + 16];
            sub_scale_payload[block_linear * 16..block_linear * 16 + 16].copy_from_slice(scale);
        }
    }
    Ok(payload)
}
