use super::super::*;

// mc78 NaN root cause 추적: pool size 8 이 mtp_spec_requested=true + 35 layer ×
// long-context (1115 token) prefill 의 cursor round-robin 빈도 + 추가 async work
// 와 race-prone. Layer 수 보다 큰 pool 으로 race window 차단.
const TRANSIENT_Q4_F16_POOL_SIZE: usize = 64;

impl CudaState {
    pub(in crate::runtime) fn resident_q4k_f16_pair_ptrs(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<(u64, u64)>, String> {
        let gate = self.transient_q4k_f16_ptr(gate_weights, rows, blocks_per_row)?;
        let up = self.transient_q4k_f16_ptr(up_weights, rows, blocks_per_row)?;
        Ok(Some((gate, up)))
    }

    pub(in crate::runtime) fn resident_q4k_f16_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<Option<u64>, String> {
        Ok(Some(self.transient_q4k_f16_ptr(
            weights,
            rows,
            blocks_per_row,
        )?))
    }

    pub(in crate::runtime) fn transient_q4k_f16_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<u64, String> {
        if !crate::tuning::expanded_weight_cache_allowed() {
            return Err(
                "Q4_K F16 expanded weight cache requires RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE=1"
                    .to_string(),
            );
        }
        let upload_bytes = q4_f16_cache_bytes(rows, blocks_per_row)?;
        let ptr = self.acquire_transient_q4_f16_slot(upload_bytes)?;
        self.launch_q4k_dequant_f16_to_dev(weights, rows, blocks_per_row, ptr)?;
        self.record_q4_expanded_f16(upload_bytes);
        Ok(ptr)
    }

    pub(in crate::runtime) fn acquire_transient_q4_f16_slot(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        self.set_current()?;
        while self.transient_q4_f16_pool.len() < TRANSIENT_Q4_F16_POOL_SIZE {
            self.transient_q4_f16_pool.push(TransientQ4F16Slot {
                buffer: None,
                capacity: 0,
            });
        }
        let slot_idx = self.transient_q4_f16_pool_cursor;
        self.transient_q4_f16_pool_cursor =
            (self.transient_q4_f16_pool_cursor + 1) % TRANSIENT_Q4_F16_POOL_SIZE;

        // 인덱스 접근(참조 미보유)으로 self 의 다중 mutable borrow 회피 — OOM retry 가
        // self.offload_non_pinned_resident_q4k() 등 self 메서드를 호출해야 하기 때문.
        if self.transient_q4_f16_pool[slot_idx].capacity < bytes {
            if let Some(old_ptr) = self.transient_q4_f16_pool[slot_idx].buffer.take() {
                let _ = unsafe { self.api.mem_free(old_ptr) };
                self.transient_q4_f16_pool[slot_idx].capacity = 0;
            }
            // cu111: transient Q4→F16 dequant slot(예: E4B FFN gate/up 50MiB) 도 OOM
            // retry 경로로. 이전엔 retry 없는 직접 mem_alloc 이라 VRAM 천장(E4B 10GB)
            // 에서 즉시 panic — dense 모델 prefill 이 offload 없이 죽던 핵심 지점.
            // attention.rs cu26 generic OOM retry 와 동일: q4k resident offload → MoE clear.
            self.reclaim_residency_for_transient(bytes)?;
            let new_ptr = match unsafe { self.api.mem_alloc(bytes) } {
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
            self.transient_q4_f16_pool[slot_idx].buffer = Some(new_ptr);
            self.transient_q4_f16_pool[slot_idx].capacity = bytes;
        }
        Ok(self.transient_q4_f16_pool[slot_idx]
            .buffer
            .expect("transient Q4 F16 slot buffer allocated"))
    }
}

fn q4_f16_cache_bytes(rows: usize, blocks_per_row: usize) -> Result<usize, String> {
    let values = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(256))
        .ok_or_else(|| format!("Q4 F16 size overflow: rows={rows} blocks={blocks_per_row}"))?;
    values
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| format!("Q4 F16 byte overflow: rows={rows} blocks={blocks_per_row}"))
}
