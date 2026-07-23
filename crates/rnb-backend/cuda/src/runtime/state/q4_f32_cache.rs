use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn clear_resident_q4_f32_cache(&mut self) -> Result<(), String> {
        self.set_current()?;
        for (_, entry) in self.resident_q4_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q4_f32_bytes = 0;
        Ok(())
    }

    pub(in crate::runtime) fn resident_q4k_f32_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        let key = q4_f32_key_for(weights, rows, blocks_per_row);
        if let Some(entry) = self.resident_q4_f32.get(&key) {
            return Ok(Some(entry.ptr));
        }
        if !crate::tuning::expanded_weight_cache_allowed() {
            return Err(
                "Q4_K F32 expanded weight cache requires RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE=1"
                    .to_string(),
            );
        }

        let bytes = q4_f32_cache_bytes(rows, blocks_per_row)?;
        let effective_limit = self
            .resident_class_effective_limit(self.resident_q4_f32_bytes, self.resident_q4_f32_limit);
        if bytes > effective_limit
            || self.resident_q4_f32_bytes.saturating_add(bytes) > effective_limit
            || !self.resident_admission_allowed(bytes)?
        {
            return Ok(None);
        }

        // cu22: route cache enrollment through the GPU dequant kernel so the
        // cache-hit path is bit-identical to the cu19 cache-miss fallback path
        // (`q4k_f32_gemm_batch_cached` calls the same `launch_q4k_dequant_f32_to_dev`
        // when the cache misses). The previous CPU `dequant_q4k_to_f32` produced
        // 1-ULP differences vs the GPU kernel — likely from nvcc's default
        // `-fmad=true` fusing `scale * q - min` while Rust release keeps mul/sub
        // separate. On Gemma4 the drift stayed below sampling margin, but on
        // Qwen3.6 35B (GDN+MoE hybrid) it compounded into a token divergence
        // around index 51 once the cache was populated.
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        if let Err(err) = self.launch_q4k_dequant_f32_to_dev(weights, rows, blocks_per_row, ptr) {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;

        self.resident_q4_f32.insert(key, ResidentQ4F32 { ptr });
        self.resident_q4_f32_bytes = self.resident_q4_f32_bytes.saturating_add(bytes);
        self.record_q4_expanded_f32(bytes);
        Ok(Some(ptr))
    }

    pub(in crate::runtime) fn resident_q4k_f32_pair_ptrs(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<(u64, u64)>, String> {
        let gate_key = q4_f32_key_for(gate_weights, rows, blocks_per_row);
        let up_key = q4_f32_key_for(up_weights, rows, blocks_per_row);
        let gate_cached = self.resident_q4_f32.get(&gate_key).map(|entry| entry.ptr);
        let up_cached = self.resident_q4_f32.get(&up_key).map(|entry| entry.ptr);
        if let (Some(gate), Some(up)) = (gate_cached, up_cached) {
            return Ok(Some((gate, up)));
        }

        let bytes = q4_f32_cache_bytes(rows, blocks_per_row)?;
        let missing = if gate_key == up_key {
            usize::from(gate_cached.is_none() && up_cached.is_none())
        } else {
            usize::from(gate_cached.is_none()) + usize::from(up_cached.is_none())
        };
        let needed = bytes.saturating_mul(missing);
        let effective_limit = self
            .resident_class_effective_limit(self.resident_q4_f32_bytes, self.resident_q4_f32_limit);
        if bytes > effective_limit
            || self.resident_q4_f32_bytes.saturating_add(needed) > effective_limit
            || !self.resident_admission_allowed(needed)?
        {
            return Ok(None);
        }

        let gate = match gate_cached {
            Some(ptr) => ptr,
            None => self
                .resident_q4k_f32_ptr(gate_weights, rows, blocks_per_row)?
                .ok_or_else(|| {
                    "Q4 F32 pair gate admission failed after capacity check".to_string()
                })?,
        };
        let up = match up_cached {
            Some(ptr) => ptr,
            None => self
                .resident_q4k_f32_ptr(up_weights, rows, blocks_per_row)?
                .ok_or_else(|| {
                    "Q4 F32 pair up admission failed after capacity check".to_string()
                })?,
        };
        Ok(Some((gate, up)))
    }
}

fn q4_f32_key_for(weights: &[u8], rows: usize, blocks_per_row: usize) -> Q4F32Key {
    Q4F32Key {
        ptr: weights.as_ptr() as usize,
        len: weights.len(),
        rows,
        blocks_per_row,
    }
}

pub(in crate::runtime) fn q4_f32_cache_bytes(
    rows: usize,
    blocks_per_row: usize,
) -> Result<usize, String> {
    let values = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(256))
        .ok_or_else(|| format!("Q4 F32 size overflow: rows={rows} blocks={blocks_per_row}"))?;
    values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("Q4 F32 byte overflow: rows={rows} blocks={blocks_per_row}"))
}
