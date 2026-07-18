use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn resident_q6k_f32_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        let key = Q6F32Key {
            ptr: weights.as_ptr() as usize,
            len: weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q6_f32.get(&key) {
            return Ok(Some(entry.ptr));
        }
        if !crate::tuning::expanded_weight_cache_allowed() {
            return Err(
                "Q6_K F32 expanded weight cache requires RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE=1"
                    .to_string(),
            );
        }

        let values = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(256))
            .ok_or_else(|| format!("Q6 F32 size overflow: rows={rows} blocks={blocks_per_row}"))?;
        let bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Q6 F32 byte overflow: rows={rows} blocks={blocks_per_row}"))?;
        if bytes > self.resident_q6_f32_limit
            || self.resident_q6_f32_bytes.saturating_add(bytes) > self.resident_q6_f32_limit
        {
            return Ok(None);
        }

        // cu26 phase A: GPU dequant for cache enrollment (mirrors cu22
        // q4_f32_cache fix). Guarantees bit-identity with the GPU kernel used
        // by other Q6_K paths, removing a potential CPU-vs-GPU drift source
        // for models that activate the q6_f32 cache.
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        if let Err(err) = self.launch_q6k_dequant_f32_to_dev(weights, rows, blocks_per_row, ptr) {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;

        self.resident_q6_f32.insert(key, ResidentQ6F32 { ptr });
        self.resident_q6_f32_bytes = self.resident_q6_f32_bytes.saturating_add(bytes);
        self.record_q6_expanded_f32(bytes);
        Ok(Some(ptr))
    }
}
