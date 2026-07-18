use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn resident_q6k_f16_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        let key = Q6F16Key {
            ptr: weights.as_ptr() as usize,
            len: weights.len(),
            rows,
            blocks_per_row,
        };
        if let Some(entry) = self.resident_q6_f16.get(&key) {
            return Ok(Some(entry.ptr));
        }
        if !crate::tuning::expanded_weight_cache_allowed() {
            return Err(
                "Q6_K F16 expanded weight cache requires RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE=1"
                    .to_string(),
            );
        }

        let bytes = q6_f16_cache_bytes(rows, blocks_per_row)?;
        if bytes > self.resident_q6_f16_limit
            || self.resident_q6_f16_bytes.saturating_add(bytes) > self.resident_q6_f16_limit
        {
            // cu19: cache full / weight too large — fall back to GPU dequant
            // into a transient buffer. Reuse the cu17 q4 f16 transient pool so
            // we don't add a second 8-slot pool. Slot capacity grows to fit
            // Q6 weight (max ~50 MiB) and stays sized; pool rotates round-robin.
            let scratch = self.acquire_transient_q4_f16_slot(bytes)?;
            self.launch_q6k_dequant_f16_to_dev(weights, rows, blocks_per_row, scratch)?;
            self.record_q6_expanded_f16(bytes);
            return Ok(Some(scratch));
        }

        // cu26 phase A: route cache enrollment through the GPU dequant kernel
        // so the cache-hit path is bit-identical to the cu19 cache-full
        // fallback path (transient pool above). Mirrors the cu22 fix applied
        // to q4_f32_cache. Eliminates a class of drift bugs where CPU host
        // dequant produced slightly different f16 conversions than the GPU
        // kernel for the same Q6_K block.
        // cu111: Q6K→F16 dequant resident enrollment(예: E4B FFN down 50MiB) 도 OOM
        // retry 경로로. 이전엔 retry 없는 직접 mem_alloc 이라 VRAM 천장(E4B 10GB)에서
        // 즉시 panic — dense 모델 prefill 이 offload 없이 죽던 핵심 지점(backtrace 확정).
        // attention.rs cu26 generic OOM retry 와 동일: q4k resident offload → MoE clear.
        let ptr = match unsafe { self.api.mem_alloc(bytes) } {
            Ok(p) => p,
            Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                let _ = self.offload_non_pinned_resident_q4k();
                match unsafe { self.api.mem_alloc(bytes) } {
                    Ok(p) => p,
                    Err(err2) if cuda_mem_alloc_oom(&err2) => {
                        self.clear_resident_moe_layer_cache()?;
                        unsafe { self.api.mem_alloc(bytes)? }
                    }
                    Err(err2) => return Err(err2),
                }
            }
            Err(err) => return Err(err),
        };
        if let Err(err) = self.launch_q6k_dequant_f16_to_dev(weights, rows, blocks_per_row, ptr) {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;

        self.resident_q6_f16.insert(key, ResidentQ6F16 { ptr });
        self.resident_q6_f16_bytes = self.resident_q6_f16_bytes.saturating_add(bytes);
        self.record_q6_expanded_f16(bytes);
        Ok(Some(ptr))
    }
}

fn q6_f16_cache_bytes(rows: usize, blocks_per_row: usize) -> Result<usize, String> {
    let values = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(256))
        .ok_or_else(|| format!("Q6 F16 size overflow: rows={rows} blocks={blocks_per_row}"))?;
    values
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| format!("Q6 F16 byte overflow: rows={rows} blocks={blocks_per_row}"))
}
