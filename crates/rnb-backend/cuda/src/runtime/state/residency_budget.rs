use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn resident_cache_bytes(&self) -> usize {
        let q4k_physical_bytes = self
            .resident_q4k_non_arena_bytes()
            .saturating_add(self.resident_q4k_arena_capacity);
        let q8_quant_bytes = self
            .resident_q8_quant
            .values()
            .fold(0usize, |acc, entry| acc.saturating_add(entry.bytes));
        let native_f32_bytes = self.resident_f32.keys().fold(0usize, |acc, key| {
            acc.saturating_add(key.len.saturating_mul(std::mem::size_of::<f32>()))
        });
        let rope_bytes = self
            .resident_rope_tables
            .values()
            .fold(0usize, |acc, entry| {
                acc.saturating_add(entry.bytes.saturating_mul(2))
            });

        q4k_physical_bytes
            .saturating_add(self.resident_q8_f32_bytes)
            .saturating_add(q8_quant_bytes)
            .saturating_add(self.resident_q4_packed_bytes)
            .saturating_add(self.resident_q4_f32_bytes)
            .saturating_add(self.resident_q6_packed_bytes)
            .saturating_add(self.resident_q6_f32_bytes)
            .saturating_add(self.resident_q6_f16_bytes)
            .saturating_add(self.resident_moe_layer_bytes)
            .saturating_add(native_f32_bytes)
            .saturating_add(rope_bytes)
    }

    pub(in crate::runtime) fn resident_class_effective_limit(
        &self,
        class_bytes: usize,
        local_limit: usize,
    ) -> usize {
        let other_resident_bytes = self.resident_cache_bytes().saturating_sub(class_bytes);
        local_limit.min(
            self.device_residency_plan
                .resident_limit_for_class(class_bytes, other_resident_bytes),
        )
    }

    pub(in crate::runtime) fn resident_admission_allowed(
        &self,
        incoming_bytes: usize,
    ) -> Result<bool, String> {
        let (free_bytes, _) = unsafe { self.api.mem_get_info() }?;
        Ok(self.device_residency_plan.allows_resident_admission(
            self.resident_cache_bytes(),
            incoming_bytes,
            free_bytes,
        ))
    }

    pub(in crate::runtime) fn prepare_quant_resident_admission(
        &mut self,
        incoming_bytes: usize,
    ) -> Result<bool, String> {
        if self.resident_admission_allowed(incoming_bytes)? {
            return Ok(true);
        }

        self.set_current()?;
        self.stream_synchronize()?;
        unsafe { self.api.stream_synchronize(self.copy_stream)? };
        self.clear_low_priority_resident_caches()?;
        self.resident_admission_allowed(incoming_bytes)
    }

    pub(in crate::runtime) fn reclaim_residency_for_transient(
        &mut self,
        requested_bytes: usize,
    ) -> Result<(), String> {
        let transient_reserve_bytes = self.transient_residency_reserve_bytes();
        let reclaim_bytes = |free_bytes: usize| {
            requested_bytes
                .saturating_add(transient_reserve_bytes)
                .saturating_sub(free_bytes)
        };
        let (free_bytes, _) = unsafe { self.api.mem_get_info() }?;
        if reclaim_bytes(free_bytes) == 0 {
            return Ok(());
        }

        self.set_current()?;
        self.stream_synchronize()?;
        unsafe { self.api.stream_synchronize(self.copy_stream)? };

        let released_low_priority = self.clear_low_priority_resident_caches()?;
        let (free_after_low_priority, _) = unsafe { self.api.mem_get_info() }?;
        let moe_reclaim_bytes = reclaim_bytes(free_after_low_priority);
        if moe_reclaim_bytes > 0 {
            let resident_bytes_before = self.resident_moe_layer_bytes;
            self.evict_resident_moe_layers_until(moe_reclaim_bytes, resident_bytes_before)?;
        }
        let (free_after_moe, _) = unsafe { self.api.mem_get_info() }?;
        if reclaim_bytes(free_after_moe) > 0 {
            let _ = self.offload_non_pinned_resident_q4k()?;
        }

        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            let (final_free_bytes, _) = unsafe { self.api.mem_get_info() }?;
            eprintln!(
                "[cuda] unified residency reclaim: request={}MiB low_priority_released={}MiB free={}MiB reserve={}MiB",
                requested_bytes / (1024 * 1024),
                released_low_priority / (1024 * 1024),
                final_free_bytes / (1024 * 1024),
                transient_reserve_bytes / (1024 * 1024),
            );
        }
        Ok(())
    }

    fn clear_low_priority_resident_caches(&mut self) -> Result<usize, String> {
        let before = self.resident_cache_bytes();

        for (_, entry) in self.resident_q8_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q8_f32_lru.clear();
        self.resident_q8_f32_bytes = 0;

        for (_, entry) in self.resident_q4_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q4_f32_bytes = 0;

        for (_, entry) in self.resident_q6_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q6_f32_bytes = 0;

        for (_, entry) in self.resident_q6_f16.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q6_f16_bytes = 0;

        for (_, entry) in self.resident_q4_packed.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q4_packed_bytes = 0;

        for (_, entry) in self.resident_q6_packed.drain() {
            unsafe {
                self.api.mem_free(entry.qs_ptr)?;
                self.api.mem_free(entry.d_super_ptr)?;
                self.api.mem_free(entry.sub_scale_ptr)?;
            }
        }
        self.resident_q6_packed_bytes = 0;

        Ok(before.saturating_sub(self.resident_cache_bytes()))
    }
}
