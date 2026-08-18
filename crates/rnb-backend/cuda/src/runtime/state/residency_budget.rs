use super::super::*;

impl CudaState {
    /// Mid-prefill resident-admission clamp for chunked CUDA prefill.
    ///
    /// Chunked prefill keeps its chain temps and per-chunk KV growth in device
    /// memory outside the resident caches. Without this clamp, hot-slab
    /// admissions during a long multi-chunk prompt fill every byte the plan
    /// allows and the later chunks die on cuMemAlloc with nothing evictable
    /// (in-use weights are protected). `clamp` lowers the admission limit for
    /// the duration of the prefill; `release` restores the saved limit.
    pub(in crate::runtime) fn clamp_resident_limit_for_prefill_scratch(
        &mut self,
        scratch_bytes: usize,
    ) {
        if scratch_bytes == 0 {
            return;
        }
        if self.prefill_scratch_saved_limit.is_none() {
            self.prefill_scratch_saved_limit = Some(self.resident_q4k_limit);
        }
        let base = self
            .prefill_scratch_saved_limit
            .unwrap_or(self.resident_q4k_limit);
        self.resident_q4k_limit = base.saturating_sub(scratch_bytes);
    }

    pub(in crate::runtime) fn release_prefill_scratch_clamp(&mut self) {
        if let Some(saved) = self.prefill_scratch_saved_limit.take() {
            self.resident_q4k_limit = saved;
        }
    }

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

    fn transient_reclaimable_resident_bytes(&self) -> usize {
        self.resident_q8_f32_bytes
            .saturating_add(self.resident_q4_f32_bytes)
            .saturating_add(self.resident_q6_f32_bytes)
            .saturating_add(self.resident_q6_f16_bytes)
            .saturating_add(self.resident_q4_packed_bytes)
            .saturating_add(self.resident_q6_packed_bytes)
            .saturating_add(self.resident_moe_layer_bytes)
            .saturating_add(self.moe_slice_cache_held_bytes())
    }

    pub(in crate::runtime) fn selected_moe_transient_admission_allowed(
        &self,
        required_bytes: usize,
    ) -> Result<bool, String> {
        let (current_free_bytes, total_bytes) = unsafe { self.api.mem_get_info() }?;
        Ok(rnb_memory::DeviceTransientAdmissionPlan {
            total_bytes,
            current_free_bytes,
            reclaimable_resident_bytes: self.transient_reclaimable_resident_bytes(),
            protected_reserve_bytes: self.device_residency_plan.dynamic_reserve_bytes,
        }
        .allows(required_bytes))
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
        let slice_reclaim_bytes = reclaim_bytes(free_after_moe);
        if slice_reclaim_bytes > 0 {
            self.shrink_moe_slice_cache_for_reclaim(slice_reclaim_bytes)?;
        }
        let (free_after_slice, _) = unsafe { self.api.mem_get_info() }?;
        if reclaim_bytes(free_after_slice) > 0 {
            let _ = self.offload_non_pinned_resident_q4k()?;
        }

        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            let (final_free_bytes, _) = unsafe { self.api.mem_get_info() }?;
            eprintln!(
                "[cuda] unified residency reclaim: request={} bytes low_priority_released={}MiB free={}MiB reserve={}MiB",
                requested_bytes,
                released_low_priority / (1024 * 1024),
                final_free_bytes / (1024 * 1024),
                transient_reserve_bytes / (1024 * 1024),
            );
        }
        Ok(())
    }

    pub(in crate::runtime) fn configure_decode_residency_reserve(&mut self, reserve_bytes: usize) {
        let current = self.device_residency_plan;
        if reserve_bytes >= current.dynamic_reserve_bytes {
            return;
        }
        self.device_residency_plan = rnb_memory::DeviceResidencyPlan::from_snapshot(
            current.total_bytes,
            current.initial_free_bytes,
            reserve_bytes,
        );
    }

    pub(in crate::runtime) fn begin_muse_decode_tail_residency(&mut self) {
        self.muse_decode_tail_base_residency
            .get_or_insert((self.device_residency_plan, self.resident_q4k_limit));
    }

    pub(in crate::runtime) fn prepare_muse_decode_layer_streaming_admission(
        &mut self,
    ) -> Result<(), String> {
        if !self.muse_decode_tail_streaming {
            return Ok(());
        }
        let (free_bytes, _) = unsafe { self.api.mem_get_info() }?;
        self.configure_decode_residency_reserve(0);
        self.resident_q4k_limit = self.resident_q4k_bytes.saturating_add(free_bytes);
        Ok(())
    }

    pub(in crate::runtime) fn restore_configured_device_residency_plan(&mut self) {
        self.device_residency_plan = self.configured_device_residency_plan;
        self.resident_q4k_limit = self.configured_resident_q4k_limit;
    }

    pub(in crate::runtime) fn clear_low_priority_resident_caches(
        &mut self,
    ) -> Result<usize, String> {
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
