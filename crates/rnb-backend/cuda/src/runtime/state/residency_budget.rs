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
        // cu287: 한도만 조이면 이미 상주한 weight이 그대로라 KV가 자라는
        // 중반에 eviction thrash(내린 weight 재업로드 반복)가 시작된다.
        // clamp 시점에 초과분을 즉시 LRU(모델 수명 pinned 포함)로 내려
        // prefill 전 구간의 자리를 시작부터 확보한다.
        let resident_before_clamp_evict = self.resident_q4k_bytes;
        // cu287: MoE layer cache(q4k와 별도 한도)가 prefill 중 KV 성장분을
        // 희생하지 않으면 q4k eviction → evicted weight의 매 청크 temporary
        // upload(H2D 수백 GB)로 이어진다. clamp 때 MoE cache 한도도 같이
        // 조이고 초과분을 즉시 내린다.
        if self.prefill_scratch_saved_moe_limit.is_none() {
            self.prefill_scratch_saved_moe_limit = Some(self.resident_moe_layer_limit);
        }
        let moe_base = self
            .prefill_scratch_saved_moe_limit
            .unwrap_or(self.resident_moe_layer_limit);
        let moe_target = moe_base.saturating_sub(scratch_bytes);
        self.resident_moe_layer_limit = moe_target;
        if self.resident_moe_layer_bytes > moe_target {
            let moe_excess = self.resident_moe_layer_bytes - moe_target;
            let moe_before = self.resident_moe_layer_bytes;
            let _ = self.evict_resident_moe_layers_until(moe_excess, moe_before);
        }
        if self.resident_q4k_bytes > self.resident_q4k_limit {
            let excess = self.resident_q4k_bytes - self.resident_q4k_limit;
            let _ = self.clear_weight_referencing_graphs();
            if let Some(plan) = self.resident_q4k_transient_reclaim_eviction_plan(excess) {
                let _ = self.execute_resident_q4k_eviction_plan(plan);
            }
        }
        // cu293: 원장 한도 정합만으로는 부족하다. packed cache·staging·graph
        // 처럼 q4k/MoE 원장 밖의 할당이 VRAM을 점유하면 resident < limit인데도
        // 실측 free가 scratch+reserve보다 부족해져 prefill 중반이 cuMemAlloc
        // error 2로 죽는다(47k verify-on: resident 16,293 < limit 18,080이라
        // eviction 0이었지만 실측 free 3,884MiB < 필요 5,763MiB). 실측 free
        // 기준으로 부족분을 시작 전에 확보한다.
        //
        // 단 packed 등 파생 캐시는 MMQ가 prefill 내내 다시 올리므로(cu289)
        // 선제 해제하면 곧 돌아올 바이트만큼 유령 여유가 생겨 아래 cap이
        // over-commit된다. shortfall은 q4k(pinned 포함 LRU) 선해제로 먼저
        // 충당하고, 그래도 부족할 때만 generic reclaim으로 내려간다.
        let target_free = scratch_bytes.saturating_add(self.transient_residency_reserve_bytes());
        let mut deep_reclaim_used = false;
        if let Ok((free_bytes, _)) = unsafe { self.api.mem_get_info() } {
            if (free_bytes as usize) < target_free {
                let shortfall = target_free - (free_bytes as usize);
                const RECLAIM_HYSTERESIS_BYTES: usize = 1024 * 1024 * 1024;
                let _ = self.clear_weight_referencing_graphs();
                let plan = self
                    .resident_q4k_transient_reclaim_eviction_plan(
                        shortfall.saturating_add(RECLAIM_HYSTERESIS_BYTES),
                    )
                    .or_else(|| self.resident_q4k_transient_reclaim_eviction_plan(shortfall));
                if let Some(plan) = plan {
                    let _ = self.execute_resident_q4k_eviction_plan(plan);
                }
                if let Ok((free_after_q4k, _)) = unsafe { self.api.mem_get_info() } {
                    if (free_after_q4k as usize) < target_free {
                        let _ = self.reclaim_residency_for_transient(scratch_bytes);
                        deep_reclaim_used = true;
                    }
                }
            }
        }
        // re-admission이 확보된 scratch 여유를 다시 먹지 못하게 한도를 실측
        // slack으로 조인다. admission 자체의 free 게이트는 plan reserve만 보기
        // 때문에 이 cap이 없으면 prefill 중 admission이 free를 reserve까지
        // 채워 scratch 예산을 잠식한다. generic reclaim이 파생 캐시를 해제한
        // 경우 그 바이트는 prefill 중 돌아오므로 slack에서 제외한다.
        if let Ok((free_bytes, _)) = unsafe { self.api.mem_get_info() } {
            let slack = if deep_reclaim_used {
                0
            } else {
                (free_bytes as usize).saturating_sub(target_free)
            };
            let headroom_cap = self.resident_q4k_bytes.saturating_add(slack);
            if self.resident_q4k_limit > headroom_cap {
                self.resident_q4k_limit = headroom_cap;
            }
        }
        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            let released = resident_before_clamp_evict.saturating_sub(self.resident_q4k_bytes);
            let free_mib = unsafe { self.api.mem_get_info() }
                .map(|(free, _)| free / (1024 * 1024))
                .unwrap_or(0);
            eprintln!(
                "[cuda] prefill scratch clamp armed: scratch={}MiB limit={}MiB resident_before={}MiB evicted={}MiB resident_now={}MiB free={}MiB",
                scratch_bytes / (1024 * 1024),
                self.resident_q4k_limit / (1024 * 1024),
                resident_before_clamp_evict / (1024 * 1024),
                released / (1024 * 1024),
                self.resident_q4k_bytes / (1024 * 1024),
                free_mib,
            );
        }
    }

    pub(in crate::runtime) fn release_prefill_scratch_clamp(&mut self) {
        if let Some(saved) = self.prefill_scratch_saved_limit.take() {
            self.resident_q4k_limit = saved;
        }
        if let Some(saved_moe) = self.prefill_scratch_saved_moe_limit.take() {
            self.resident_moe_layer_limit = saved_moe;
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
            let _ = self.offload_non_pinned_resident_q4k_releasing(requested_bytes)?;
        }

        // cu287: 마지막 계단 — non-pinned까지 풀어도 부족하면 prewarm이 올린
        // model-lifetime pinned q4k까지 LRU로 내려 transient(KV device 상주·
        // 청크 스크래치)에 예산을 이전한다. 긴 prefill에서 KV가 자랄 때 이
        // 경로가 없으면 free가 단조 감소하다 cuMemAlloc error 2로 사망한다.
        // weight ptr을 쥔 CUDA graph를 먼저 drain해 stale launch를 막고,
        // stream 캡처 진행 중에는 graph drain이 부적절하므로 건너뛴다.
        let (free_after_offload, _) = unsafe { self.api.mem_get_info() }?;
        let still_needed = reclaim_bytes(free_after_offload);
        if still_needed > 0 && !self.mtp_verify_segment_capture_active {
            self.clear_weight_referencing_graphs()?;
            let resident_before = self.resident_q4k_bytes;
            // hysteresis: 필요분만 내리면 free가 매번 reserve 근처에
            // 붙어 다음 할당에서 즉시 재발동한다(release 41~95MiB가
            // 반복되다 96MiB 요청 하나에 바닥나는 패턴). 여유 1GiB까지
            // 함께 내리고, 내릴 수 있는 q4k가 그만 못하면 필요분으로
            // 폴백한다.
            const RECLAIM_HYSTERESIS_BYTES: usize = 1024 * 1024 * 1024;
            let plan = self
                .resident_q4k_transient_reclaim_eviction_plan(
                    still_needed.saturating_add(RECLAIM_HYSTERESIS_BYTES),
                )
                .or_else(|| self.resident_q4k_transient_reclaim_eviction_plan(still_needed));
            if let Some(plan) = plan {
                let _ = self.execute_resident_q4k_eviction_plan(plan)?;
            }
            if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                let released = resident_before.saturating_sub(self.resident_q4k_bytes);
                let (free_after_evict, _) = unsafe { self.api.mem_get_info() }?;
                eprintln!(
                    "[cuda] transient reclaim evicted resident q4k (pinned incl.): needed={}MiB released={}MiB free={}MiB",
                    still_needed / (1024 * 1024),
                    released / (1024 * 1024),
                    free_after_evict / (1024 * 1024),
                );
            }
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
