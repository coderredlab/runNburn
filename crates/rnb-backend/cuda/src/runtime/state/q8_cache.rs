use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn clear_resident_q8_prefill_projection_cache(
        &mut self,
    ) -> Result<(), String> {
        self.set_current()?;
        for (_, entry) in self.resident_q8_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        for (_, entry) in self.resident_q8_quant.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q8_f32_lru.clear();
        self.resident_q8_f32_bytes = 0;
        Ok(())
    }

    pub(in crate::runtime) fn resident_q8_0_f32_ptr(
        &mut self,
        weights: &[u8],
        quant_dev: u64,
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<u64, String> {
        let cols = blocks_per_row * 32;
        let key = Q8F32Key {
            ptr: weights.as_ptr() as usize,
            len: weights.len(),
            rows,
            cols,
        };
        if let Some(ptr) = self.resident_q8_f32.get(&key).map(|entry| entry.ptr) {
            let epoch = self.next_resident_q8_f32_epoch();
            if let Some(entry) = self.resident_q8_f32.get_mut(&key) {
                entry.epoch = epoch;
            }
            self.resident_q8_f32_lru.push_back((key, epoch));
            return Ok(ptr);
        }

        let bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| format!("Q8_0 F32 cache size overflow: rows={rows} cols={cols}"))?;
        let effective_limit = self
            .resident_class_effective_limit(self.resident_q8_f32_bytes, self.resident_q8_f32_limit);
        if effective_limit < bytes {
            let ptr = if rows >= cols {
                self.compute_full_down_ptr(bytes)?
            } else {
                self.compute_full_up_ptr(bytes)?
            };
            self.launch_q8_0_dequant_f32_to_dev(quant_dev, ptr, rows, blocks_per_row)?;
            return Ok(ptr);
        }

        self.evict_resident_q8_f32_until(bytes)?;
        let effective_limit = self
            .resident_class_effective_limit(self.resident_q8_f32_bytes, self.resident_q8_f32_limit);
        if self.resident_q8_f32_bytes.saturating_add(bytes) > effective_limit
            || !self.resident_admission_allowed(bytes)?
        {
            let ptr = if rows >= cols {
                self.compute_full_down_ptr(bytes)?
            } else {
                self.compute_full_up_ptr(bytes)?
            };
            self.launch_q8_0_dequant_f32_to_dev(quant_dev, ptr, rows, blocks_per_row)?;
            return Ok(ptr);
        }
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        self.launch_q8_0_dequant_f32_to_dev(quant_dev, ptr, rows, blocks_per_row)?;
        let epoch = self.next_resident_q8_f32_epoch();
        self.resident_q8_f32
            .insert(key, ResidentQ8F32 { ptr, bytes, epoch });
        self.resident_q8_f32_lru.push_back((key, epoch));
        self.resident_q8_f32_bytes = self.resident_q8_f32_bytes.saturating_add(bytes);
        Ok(ptr)
    }

    pub(in crate::runtime) fn resident_q8_quant_ptr(
        &mut self,
        weights: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<u64, String> {
        let key = q8_f32_key(weights, rows, cols);
        if let Some(ptr) = self.resident_q8_quant.get(&key).map(|entry| entry.ptr) {
            return Ok(ptr);
        }
        self.reclaim_residency_for_transient(weights.len())?;
        let ptr = unsafe { self.api.mem_alloc(weights.len()) }?;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )?;
        }
        self.resident_q8_quant.insert(
            key,
            ResidentQ8F32 {
                ptr,
                bytes: weights.len(),
                epoch: 0,
            },
        );
        Ok(ptr)
    }

    pub(in crate::runtime) fn next_resident_q8_f32_epoch(&mut self) -> u64 {
        self.resident_q8_f32_epoch = self.resident_q8_f32_epoch.wrapping_add(1);
        if self.resident_q8_f32_epoch == 0 {
            self.resident_q8_f32_epoch = 1;
        }
        self.resident_q8_f32_epoch
    }

    pub(in crate::runtime) fn evict_resident_q8_f32_until(
        &mut self,
        incoming: usize,
    ) -> Result<(), String> {
        let effective_limit = self
            .resident_class_effective_limit(self.resident_q8_f32_bytes, self.resident_q8_f32_limit);
        while self.resident_q8_f32_bytes.saturating_add(incoming) > effective_limit {
            let Some((key, epoch)) = self.resident_q8_f32_lru.pop_front() else {
                break;
            };
            if self
                .resident_q8_f32
                .get(&key)
                .is_some_and(|entry| entry.epoch != epoch)
            {
                continue;
            }
            if let Some(entry) = self.resident_q8_f32.remove(&key) {
                unsafe { self.api.mem_free(entry.ptr)? };
                self.resident_q8_f32_bytes = self.resident_q8_f32_bytes.saturating_sub(entry.bytes);
            }
        }
        Ok(())
    }
}
