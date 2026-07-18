use super::super::*;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum ResidentQ4kLruCandidate {
    OwnedBundle(rnb_memory::SparseExpertCacheKey),
    UnownedRole((usize, usize)),
}

impl CudaState {
    fn resident_q4k_non_arena_bytes(&self) -> usize {
        let owned_bytes = self
            .resident_q4k
            .values()
            .filter(|entry| entry.owned_alloc)
            .fold(0usize, |acc, entry| acc.saturating_add(entry.bytes));
        let slab_bytes = self
            .resident_q4k_slabs
            .values()
            .fold(0usize, |acc, slab| acc.saturating_add(slab.bytes));
        owned_bytes.saturating_add(slab_bytes)
    }

    fn refresh_resident_q4k_bytes(&mut self) {
        self.resident_q4k_bytes = self
            .resident_q4k_non_arena_bytes()
            .saturating_add(self.resident_q4k_arena_offset);
    }

    pub(in crate::runtime) fn register_qwen35_q2q3_bundle_ownership(
        &mut self,
        bundle_key: rnb_memory::SparseExpertCacheKey,
        roles: &HashSet<(usize, usize)>,
    ) -> Option<u64> {
        if self
            .qwen35_q2q3_bundle_ownership
            .bundle_roles
            .contains_key(&bundle_key)
        {
            self.touch_qwen35_q2q3_bundle(bundle_key);
            return None;
        }
        if roles
            .iter()
            .any(|role| !self.resident_q4k.contains_key(role))
        {
            return None;
        }

        let mut newly_owned_payload_bytes = 0u64;
        for &role in roles {
            let owners = self
                .qwen35_q2q3_bundle_ownership
                .role_owners
                .entry(role)
                .or_default();
            if owners.is_empty() {
                newly_owned_payload_bytes =
                    newly_owned_payload_bytes.saturating_add(self.resident_q4k[&role].bytes as u64);
            }
            owners.insert(bundle_key);
        }
        self.qwen35_q2q3_bundle_ownership
            .bundle_roles
            .insert(bundle_key, roles.clone());
        self.touch_qwen35_q2q3_bundle(bundle_key);
        self.qwen35_q2q3_resident_payload_bytes = self
            .qwen35_q2q3_resident_payload_bytes
            .saturating_add(newly_owned_payload_bytes);
        cache_stats().add_expert_bundle_resident_payload(newly_owned_payload_bytes);
        Some(newly_owned_payload_bytes)
    }

    fn touch_qwen35_q2q3_bundle(&mut self, bundle_key: rnb_memory::SparseExpertCacheKey) {
        if !self
            .qwen35_q2q3_bundle_ownership
            .bundle_roles
            .contains_key(&bundle_key)
        {
            return;
        }
        let epoch = self.next_resident_q4k_epoch();
        self.qwen35_q2q3_bundle_ownership
            .bundle_epochs
            .insert(bundle_key, epoch);
        self.qwen35_q2q3_bundle_ownership
            .bundle_lru
            .push_back((bundle_key, epoch));
    }

    fn qwen35_q2q3_owned_closure(
        &self,
        root: rnb_memory::SparseExpertCacheKey,
    ) -> ResidentQ4kEvictionUnit {
        let mut bundles = HashSet::new();
        let mut roles = HashSet::new();
        let mut pending = vec![root];
        while let Some(bundle) = pending.pop() {
            if !bundles.insert(bundle) {
                continue;
            }
            let Some(bundle_roles) = self.qwen35_q2q3_bundle_ownership.bundle_roles.get(&bundle)
            else {
                continue;
            };
            for &role in bundle_roles {
                if roles.insert(role) {
                    if let Some(owners) = self.qwen35_q2q3_bundle_ownership.role_owners.get(&role) {
                        pending.extend(owners.iter().copied());
                    }
                }
            }
        }
        let mut bundles = bundles.into_iter().collect::<Vec<_>>();
        let mut roles = roles.into_iter().collect::<Vec<_>>();
        bundles.sort_unstable();
        roles.sort_unstable();
        ResidentQ4kEvictionUnit::OwnedClosure { roles, bundles }
    }

    fn record_qwen35_q2q3_cache_clear(&mut self) {
        let bundle_evictions = self.qwen35_q2q3_bundle_ownership.bundle_roles.len() as u64;
        if bundle_evictions == 0 {
            return;
        }
        let evicted_bytes = self
            .qwen35_q2q3_bundle_ownership
            .role_owners
            .keys()
            .filter_map(|role| self.resident_q4k.get(role))
            .fold(0u64, |bytes, entry| {
                bytes.saturating_add(entry.bytes as u64)
            });
        cache_stats()
            .remove_expert_bundle_resident_payload(self.qwen35_q2q3_resident_payload_bytes);
        self.qwen35_q2q3_resident_payload_bytes = 0;
        self.qwen35_q2q3_bundle_ownership = ResidentQ4kBundleOwnership::default();
        let mut delta = rnb_memory::ExpertBundleCacheStats::default();
        delta.bundle_evictions = bundle_evictions;
        delta.evicted_bytes = evicted_bytes;
        cache_stats().record_expert_bundles(delta);
    }

    pub(in crate::runtime) fn raise_resident_q4k_limit_for_qwen35_target_decode(
        &mut self,
    ) -> Result<(), String> {
        if self.qwen35_target_decode_q4k_limit_checked {
            return Ok(());
        }
        if mtp_device_verify_env_enabled() {
            return Ok(());
        }
        if std::env::var("RNB_CUDA_Q4K_CACHE_MB")
            .ok()
            .map(|raw| {
                let raw = raw.trim();
                !raw.is_empty() && !raw.eq_ignore_ascii_case("auto")
            })
            .unwrap_or(false)
        {
            self.qwen35_target_decode_q4k_limit_checked = true;
            return Ok(());
        }

        let (free_bytes, total_bytes) = unsafe { self.api.mem_get_info() }?;
        let mib = 1024 * 1024;
        let total_mib = total_bytes / mib;
        if total_mib > 16 * 1024 {
            self.qwen35_target_decode_q4k_limit_checked = true;
            return Ok(());
        }
        let free_mib = free_bytes / mib;
        let current_mib = self.resident_q4k_bytes / mib;
        let reserve_mib = q4k_resident_configured_reserve_mib(total_mib, false)?;
        let available_mib = free_mib
            .saturating_add(current_mib)
            .saturating_sub(reserve_mib);
        let target_mib = q4k_resident_target_decode_cache_cap_mib(total_mib)
            .min(total_mib.saturating_sub(reserve_mib))
            .min(available_mib);
        let target_limit = target_mib.saturating_mul(mib);
        if target_limit > self.resident_q4k_limit {
            self.resident_q4k_limit = target_limit;
            if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cuda] q4k resident cache qwen35 target decode raised: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
                    total_mib, free_mib, reserve_mib, target_mib
                );
            }
        }
        self.qwen35_target_decode_q4k_limit_checked = true;
        Ok(())
    }

    pub(in crate::runtime) fn raise_resident_q4k_limit_for_nemotron_decode(
        &mut self,
    ) -> Result<(), String> {
        self.resident_q4k_touch_hits_auto = true;
        if self.nemotron_decode_q4k_limit_checked {
            return Ok(());
        }
        if mtp_device_verify_env_enabled() {
            return Ok(());
        }
        if std::env::var("RNB_CUDA_Q4K_CACHE_MB")
            .ok()
            .map(|raw| {
                let raw = raw.trim();
                !raw.is_empty() && !raw.eq_ignore_ascii_case("auto")
            })
            .unwrap_or(false)
        {
            self.nemotron_decode_q4k_limit_checked = true;
            return Ok(());
        }

        let (free_bytes, total_bytes) = unsafe { self.api.mem_get_info() }?;
        let mib = 1024 * 1024;
        let total_mib = total_bytes / mib;
        let free_mib = free_bytes / mib;
        let current_mib = self.resident_q4k_bytes / mib;
        let reserve_mib = q4k_resident_configured_reserve_mib(total_mib, false)?;
        let available_mib = free_mib
            .saturating_add(current_mib)
            .saturating_sub(reserve_mib);
        let target_mib = q4k_resident_nemotron_decode_cache_cap_mib(total_mib)
            .min(total_mib.saturating_sub(reserve_mib))
            .min(available_mib);
        let target_limit = target_mib.saturating_mul(mib);
        if target_limit > self.resident_q4k_limit {
            self.resident_q4k_limit = target_limit;
            if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cuda] q4k resident cache nemotron decode raised: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
                    total_mib, free_mib, reserve_mib, target_mib
                );
            }
        }
        self.nemotron_decode_q4k_limit_checked = true;
        Ok(())
    }

    fn resident_q4k_entry_is_in_arena(&self, entry: &ResidentQ4k) -> bool {
        let Some(arena) = self.resident_q4k_arena else {
            return false;
        };
        let arena_end = arena.saturating_add(self.resident_q4k_arena_capacity as u64);
        entry.ptr >= arena && entry.ptr < arena_end
    }

    fn release_resident_q4k_arena_if_unreferenced(&mut self) -> Result<usize, String> {
        let arena_can_be_released = self.resident_q4k_arena.is_some()
            && !self
                .resident_q4k
                .values()
                .any(|entry| self.resident_q4k_entry_is_in_arena(entry));
        if !arena_can_be_released {
            return Ok(0);
        }
        let released = self.resident_q4k_arena_capacity;
        if let Some(arena) = self.resident_q4k_arena.take() {
            unsafe { self.api.mem_free(arena)? };
        }
        self.resident_q4k_arena_capacity = 0;
        self.resident_q4k_arena_offset = 0;
        Ok(released)
    }

    pub(in crate::runtime) fn offload_non_pinned_resident_q4k(&mut self) -> Result<usize, String> {
        self.set_current()?;
        if self.resident_q4k.is_empty() && self.resident_q4k_arena.is_none() {
            return Ok(0);
        }
        let before = self.resident_q4k_bytes;
        let plan = self
            .plan_resident_q4k_evictions(None, &HashSet::new(), None, false)
            .expect("unbounded resident Q4K eviction plan");
        let _ = self.execute_resident_q4k_eviction_plan(plan)?;
        let released = before.saturating_sub(self.resident_q4k_bytes);
        if released > 0 && std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            eprintln!(
                "[cuda] q4k resident offloaded released={}MiB remaining={}MiB",
                released / (1024 * 1024),
                self.resident_q4k_bytes / (1024 * 1024)
            );
        }
        Ok(released)
    }

    fn upload_temp_q4k_weights_current(&mut self, weights: &[u8]) -> Result<u64, String> {
        cache_stats()
            .temp_upload_bytes
            .fetch_add(weights.len() as u64, Ordering::Relaxed);
        let ptr = self.compute_weights_ptr(weights.len())?;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )
        }?;
        self.record_transient_quant_upload("Q4_K", weights.len());
        Ok(ptr)
    }

    pub(in crate::runtime) fn resident_q4k_weights_ptr(
        &mut self,
        weights: &[u8],
    ) -> Result<u64, String> {
        self.set_current()?;
        self.resident_q4k_weights_ptr_current(weights)
    }

    pub(in crate::runtime) fn resident_q4k_weights_ptr_touch_hit(
        &mut self,
        weights: &[u8],
    ) -> Result<u64, String> {
        let key = q4k_resident_key(weights);
        let was_resident = self.resident_q4k.contains_key(&key);
        let touch_after_lookup = was_resident
            && !self.resident_q4k_touch_hits_auto
            && !tuning::resident_q4k_touch_hits_enabled();
        let ptr = self.resident_q4k_weights_ptr(weights)?;
        if touch_after_lookup {
            self.touch_resident_q4k(key);
        }
        Ok(ptr)
    }

    pub(in crate::runtime) fn resident_q4k_weights_ptr_current(
        &mut self,
        weights: &[u8],
    ) -> Result<u64, String> {
        self.resident_q4k_weights_ptr_current_with_arena(
            weights,
            tuning::resident_q4k_arena_enabled(),
            false,
        )
    }

    pub(in crate::runtime) fn resident_q4k_weights_ptr_pinned(
        &mut self,
        weights: &[u8],
    ) -> Result<u64, String> {
        self.set_current()?;
        self.resident_q4k_weights_ptr_current_with_arena(weights, false, true)
    }

    // cu27: launch 시퀀스 안에서 q→k→v 등 다중 register시, 직전 register가
    // 후속 register의 OOM offload에 휩쓸려 free되는 race 차단용 임시 pin/unpin.
    pub(in crate::runtime) fn unpin_resident_q4k(&mut self, weights: &[u8]) {
        let key = q4k_resident_key(weights);
        if let Some(entry) = self.resident_q4k.get_mut(&key) {
            entry.pinned = false;
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::runtime) fn resident_q4k_weights_ptr_current_arena(
        &mut self,
        weights: &[u8],
    ) -> Result<u64, String> {
        self.resident_q4k_weights_ptr_current_with_arena(weights, true, false)
    }

    fn resident_q4k_weights_ptr_current_with_arena(
        &mut self,
        weights: &[u8],
        use_arena: bool,
        pinned: bool,
    ) -> Result<u64, String> {
        cache_stats().lookups.fetch_add(1, Ordering::Relaxed);
        let key = q4k_resident_key(weights);
        if let Some(ptr) = self.resident_q4k.get(&key).map(|entry| entry.ptr) {
            cache_stats().hits.fetch_add(1, Ordering::Relaxed);
            if pinned {
                if let Some(entry) = self.resident_q4k.get_mut(&key) {
                    entry.pinned = true;
                }
            }
            if pinned
                || self.resident_q4k_touch_hits_auto
                || tuning::resident_q4k_touch_hits_enabled()
            {
                let epoch = self.next_resident_q4k_epoch();
                if let Some(entry) = self.resident_q4k.get_mut(&key) {
                    entry.epoch = epoch;
                    self.resident_q4k_lru.push_back((key, epoch));
                }
            }
            return Ok(ptr);
        }

        cache_stats().misses.fetch_add(1, Ordering::Relaxed);
        let epoch = self.next_resident_q4k_epoch();
        if use_arena {
            return self.resident_q4k_weights_ptr_arena(weights, key, epoch);
        }
        if self.resident_q4k_limit < weights.len() {
            return self.upload_temp_q4k_weights_current(weights);
        }

        self.evict_resident_q4k_until(weights.len())?;
        if self.resident_q4k_bytes.saturating_add(weights.len()) > self.resident_q4k_limit {
            return self.upload_temp_q4k_weights_current(weights);
        }
        let ptr = match self.resident_q4k_mem_alloc(weights.len()) {
            Ok(ptr) => ptr,
            Err(err) if cuda_mem_alloc_oom(&err) => {
                match self.resident_q4k_mem_alloc_after_oom_with_bundle_eviction_retry(
                    err,
                    weights.len(),
                    &HashSet::new(),
                ) {
                    Ok((ptr, _)) => ptr,
                    Err(retry_err)
                        if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&retry_err) =>
                    {
                        let _ = self.offload_non_pinned_resident_q4k()?;
                        if self.resident_q4k_bytes.saturating_add(weights.len())
                            > self.resident_q4k_limit
                        {
                            return self.upload_temp_q4k_weights_current(weights);
                        }
                        match self.resident_q4k_mem_alloc(weights.len()) {
                            Ok(ptr) => ptr,
                            Err(second) if cuda_mem_alloc_oom(&second) => {
                                return self.upload_temp_q4k_weights_current(weights);
                            }
                            Err(second) => return Err(second),
                        }
                    }
                    Err(retry_err) => return Err(retry_err),
                }
            }
            Err(err) => return Err(err),
        };
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )
        }?;
        cache_stats()
            .resident_upload_bytes
            .fetch_add(weights.len() as u64, Ordering::Relaxed);
        self.resident_q4k.insert(
            key,
            ResidentQ4k {
                ptr,
                bytes: weights.len(),
                epoch,
                owned_alloc: true,
                slab_base: None,
                pinned,
            },
        );
        self.resident_q4k_lru.push_back((key, epoch));
        self.record_raw_quant_residency("Q4_K", weights.len());
        self.resident_q4k_bytes = self.resident_q4k_bytes.saturating_add(weights.len());
        Ok(ptr)
    }

    pub(in crate::runtime) fn resident_q4k_weights_ptr_arena(
        &mut self,
        weights: &[u8],
        key: (usize, usize),
        epoch: u64,
    ) -> Result<u64, String> {
        let non_arena_bytes = self.resident_q4k_non_arena_bytes();
        let arena_limit = self.resident_q4k_limit.saturating_sub(non_arena_bytes);
        if self.resident_q4k_limit < weights.len()
            || self.resident_q4k_arena_offset.saturating_add(weights.len()) > arena_limit
        {
            return self.upload_temp_q4k_weights_current(weights);
        }

        if self.resident_q4k_arena.is_none() {
            let ptr = match unsafe { self.api.mem_alloc(arena_limit) } {
                Ok(ptr) => ptr,
                Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                    let _ = self.offload_non_pinned_resident_q4k()?;
                    self.resident_q4k_limit = self.resident_q4k_non_arena_bytes();
                    return self.upload_temp_q4k_weights_current(weights);
                }
                Err(err) => return Err(err),
            };
            self.resident_q4k_arena = Some(ptr);
            self.resident_q4k_arena_capacity = arena_limit;
            self.resident_q4k_arena_offset = 0;
            if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cuda] q4k resident arena allocated limit={}MiB",
                    arena_limit / (1024 * 1024)
                );
            }
        }

        let arena = self
            .resident_q4k_arena
            .ok_or_else(|| "missing Q4K resident arena".to_string())?;
        let aligned = align_up(self.resident_q4k_arena_offset, 256);
        if aligned.saturating_add(weights.len()) > self.resident_q4k_arena_capacity {
            return self.upload_temp_q4k_weights_current(weights);
        }
        let ptr = arena + aligned as u64;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )
        }?;
        cache_stats()
            .resident_upload_bytes
            .fetch_add(weights.len() as u64, Ordering::Relaxed);
        self.resident_q4k.insert(
            key,
            ResidentQ4k {
                ptr,
                bytes: weights.len(),
                epoch,
                owned_alloc: false,
                slab_base: None,
                pinned: false,
            },
        );
        self.resident_q4k_lru.push_back((key, epoch));
        self.record_raw_quant_residency("Q4_K", weights.len());
        self.resident_q4k_arena_offset = aligned.saturating_add(weights.len());
        self.refresh_resident_q4k_bytes();
        Ok(ptr)
    }

    pub(in crate::runtime) fn preload_resident_q4k_range(
        &mut self,
        range: MoeJitByteRange,
    ) -> Result<bool, String> {
        self.set_current()?;
        let weights = unsafe { std::slice::from_raw_parts(range.ptr_addr as *const u8, range.len) };
        cache_stats().lookups.fetch_add(1, Ordering::Relaxed);
        let key = q4k_resident_key(weights);
        if self.resident_q4k.contains_key(&key) {
            cache_stats().hits.fetch_add(1, Ordering::Relaxed);
            self.touch_resident_q4k(key);
            return Ok(false);
        }
        cache_stats().misses.fetch_add(1, Ordering::Relaxed);
        if self.resident_q4k_limit < weights.len() {
            cache_stats()
                .temp_upload_bytes
                .fetch_add(weights.len() as u64, Ordering::Relaxed);
            return Ok(false);
        }

        self.evict_resident_q4k_until(weights.len())?;
        if self.resident_q4k_bytes.saturating_add(weights.len()) > self.resident_q4k_limit {
            cache_stats()
                .temp_upload_bytes
                .fetch_add(weights.len() as u64, Ordering::Relaxed);
            return Ok(false);
        }
        let ptr = match unsafe { self.api.mem_alloc(weights.len()) } {
            Ok(ptr) => ptr,
            Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                let _ = self.offload_non_pinned_resident_q4k()?;
                return Ok(false);
            }
            Err(err) => return Err(err),
        };
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )
        }?;
        cache_stats()
            .resident_upload_bytes
            .fetch_add(weights.len() as u64, Ordering::Relaxed);
        let epoch = self.next_resident_q4k_epoch();
        self.resident_q4k.insert(
            key,
            ResidentQ4k {
                ptr,
                bytes: weights.len(),
                epoch,
                owned_alloc: true,
                slab_base: None,
                pinned: false,
            },
        );
        self.resident_q4k_lru.push_back((key, epoch));
        self.record_raw_quant_residency("Q4_K", weights.len());
        self.resident_q4k_bytes = self.resident_q4k_bytes.saturating_add(weights.len());
        Ok(true)
    }

    pub(in crate::runtime) fn preload_resident_q4k_weight_slice(
        &mut self,
        weights: &[u8],
    ) -> Result<bool, String> {
        self.preload_resident_q4k_weight_slice_with_protection(weights, &HashSet::new(), true, None)
            .map(|result| result.uploaded)
    }

    pub(in crate::runtime) fn preload_resident_q4k_weight_slice_protecting(
        &mut self,
        weights: &[u8],
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Result<bool, String> {
        self.preload_resident_q4k_weight_slice_with_protection(weights, protected_keys, false, None)
            .map(|result| result.uploaded)
    }

    pub(in crate::runtime) fn preload_resident_q4k_weight_slice_for_profitable_bundle(
        &mut self,
        weights: &[u8],
        protected_keys: &HashSet<(usize, usize)>,
        additional_oom_reload_budget: u64,
    ) -> Result<ResidentQ4kAdmissionResult, String> {
        self.preload_resident_q4k_weight_slice_with_protection(
            weights,
            protected_keys,
            false,
            Some(additional_oom_reload_budget),
        )
    }

    pub(in crate::runtime) fn resident_q4k_mem_alloc(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        #[cfg(test)]
        if self.qwen35_resident_alloc_ooms_remaining > 0 {
            self.qwen35_resident_alloc_ooms_remaining -= 1;
            return Err(
                "cuMemAlloc failed with CUDA error 2 (injected resident allocation OOM)"
                    .to_string(),
            );
        }
        unsafe { self.api.mem_alloc(bytes) }
    }

    #[cfg(test)]
    pub(in crate::runtime) fn inject_qwen35_resident_alloc_ooms_for_test(&mut self, count: usize) {
        self.qwen35_resident_alloc_ooms_remaining = count;
    }

    pub(in crate::runtime) fn resident_q4k_mem_alloc_after_oom_with_bundle_eviction_retry(
        &mut self,
        original_oom: String,
        bytes: usize,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Result<(u64, rnb_memory::ExpertBundleCacheStats), String> {
        let mut evictions = rnb_memory::ExpertBundleCacheStats::default();
        loop {
            let Some(plan) =
                self.resident_q4k_additional_oom_eviction_plan(protected_keys, u64::MAX)
            else {
                return Err(original_oom);
            };
            let delta = self.execute_resident_q4k_eviction_plan(plan)?;
            evictions.bundle_evictions = evictions
                .bundle_evictions
                .saturating_add(delta.bundle_evictions);
            evictions.evicted_bytes = evictions.evicted_bytes.saturating_add(delta.evicted_bytes);
            match self.resident_q4k_mem_alloc(bytes) {
                Ok(ptr) => return Ok((ptr, evictions)),
                Err(err) if cuda_mem_alloc_oom(&err) => {}
                Err(err) => return Err(err),
            }
        }
    }

    pub(in crate::runtime) fn resident_q4k_mem_alloc_with_profitable_oom_retry(
        &mut self,
        bytes: usize,
        protected_keys: &HashSet<(usize, usize)>,
        additional_oom_reload_budget: u64,
    ) -> Result<(u64, rnb_memory::ExpertBundleCacheStats), String> {
        let original_oom = match self.resident_q4k_mem_alloc(bytes) {
            Ok(ptr) => {
                return Ok((ptr, rnb_memory::ExpertBundleCacheStats::default()));
            }
            Err(err) if cuda_mem_alloc_oom(&err) => err,
            Err(err) => return Err(err),
        };

        let mut remaining_budget = additional_oom_reload_budget;
        let mut evictions = rnb_memory::ExpertBundleCacheStats::default();
        loop {
            let Some(plan) =
                self.resident_q4k_additional_oom_eviction_plan(protected_keys, remaining_budget)
            else {
                return Err(original_oom);
            };
            remaining_budget = remaining_budget.saturating_sub(plan.reload_payload_bytes);
            let delta = self.execute_resident_q4k_eviction_plan(plan)?;
            evictions.bundle_evictions = evictions
                .bundle_evictions
                .saturating_add(delta.bundle_evictions);
            evictions.evicted_bytes = evictions.evicted_bytes.saturating_add(delta.evicted_bytes);
            match self.resident_q4k_mem_alloc(bytes) {
                Ok(ptr) => return Ok((ptr, evictions)),
                Err(err) if cuda_mem_alloc_oom(&err) => {}
                Err(err) => return Err(err),
            }
        }
    }

    fn preload_resident_q4k_weight_slice_with_protection(
        &mut self,
        weights: &[u8],
        protected_keys: &HashSet<(usize, usize)>,
        allow_global_oom_offload: bool,
        additional_oom_reload_budget: Option<u64>,
    ) -> Result<ResidentQ4kAdmissionResult, String> {
        self.set_current()?;
        cache_stats().lookups.fetch_add(1, Ordering::Relaxed);
        let key = q4k_resident_key(weights);
        if self.resident_q4k.contains_key(&key) {
            cache_stats().hits.fetch_add(1, Ordering::Relaxed);
            self.touch_resident_q4k(key);
            return Ok(ResidentQ4kAdmissionResult::default());
        }
        cache_stats().misses.fetch_add(1, Ordering::Relaxed);
        if self.resident_q4k_limit < weights.len() {
            return Ok(ResidentQ4kAdmissionResult::default());
        }

        self.evict_resident_q4k_until_protecting(weights.len(), protected_keys)?;
        if self.resident_q4k_bytes.saturating_add(weights.len()) > self.resident_q4k_limit {
            return Ok(ResidentQ4kAdmissionResult::default());
        }
        let (ptr, evictions) = if let Some(oom_budget) = additional_oom_reload_budget {
            match self.resident_q4k_mem_alloc_with_profitable_oom_retry(
                weights.len(),
                protected_keys,
                oom_budget,
            ) {
                Ok(result) => result,
                Err(err) if cuda_mem_alloc_oom(&err) => {
                    return Ok(ResidentQ4kAdmissionResult::default());
                }
                Err(err) => return Err(err),
            }
        } else {
            match self.resident_q4k_mem_alloc(weights.len()) {
                Ok(ptr) => (ptr, rnb_memory::ExpertBundleCacheStats::default()),
                Err(err) if cuda_mem_alloc_oom(&err) => {
                    match self.resident_q4k_mem_alloc_after_oom_with_bundle_eviction_retry(
                        err,
                        weights.len(),
                        protected_keys,
                    ) {
                        Ok(result) => result,
                        Err(retry_err)
                            if allow_global_oom_offload
                                && cuda_offload_on_oom_enabled()
                                && cuda_mem_alloc_oom(&retry_err) =>
                        {
                            let _ = self.offload_non_pinned_resident_q4k()?;
                            self.evict_resident_q4k_until(weights.len())?;
                            if self.resident_q4k_bytes.saturating_add(weights.len())
                                > self.resident_q4k_limit
                            {
                                return Ok(ResidentQ4kAdmissionResult::default());
                            }
                            match self.resident_q4k_mem_alloc(weights.len()) {
                                Ok(ptr) => (ptr, rnb_memory::ExpertBundleCacheStats::default()),
                                Err(second) if cuda_mem_alloc_oom(&second) => {
                                    return Ok(ResidentQ4kAdmissionResult::default());
                                }
                                Err(second) => return Err(second),
                            }
                        }
                        Err(retry_err) if cuda_mem_alloc_oom(&retry_err) => {
                            return Ok(ResidentQ4kAdmissionResult::default());
                        }
                        Err(retry_err) => return Err(retry_err),
                    }
                }
                Err(err) => return Err(err),
            }
        };
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                weights.as_ptr().cast::<libc::c_void>(),
                weights.len(),
                self.stream,
            )
        }?;
        cache_stats()
            .resident_upload_bytes
            .fetch_add(weights.len() as u64, Ordering::Relaxed);
        let epoch = self.next_resident_q4k_epoch();
        self.resident_q4k.insert(
            key,
            ResidentQ4k {
                ptr,
                bytes: weights.len(),
                epoch,
                owned_alloc: true,
                slab_base: None,
                pinned: false,
            },
        );
        self.resident_q4k_lru.push_back((key, epoch));
        self.record_raw_quant_residency("Q4_K", weights.len());
        self.resident_q4k_bytes = self.resident_q4k_bytes.saturating_add(weights.len());
        Ok(ResidentQ4kAdmissionResult {
            uploaded: true,
            evictions,
        })
    }

    pub(in crate::runtime) fn q4k_weight_slice_is_resident(&self, weights: &[u8]) -> bool {
        self.resident_q4k.contains_key(&q4k_resident_key(weights))
    }

    pub(in crate::runtime) fn touch_resident_q4k(&mut self, key: (usize, usize)) {
        if let Some(owners) = self
            .qwen35_q2q3_bundle_ownership
            .role_owners
            .get(&key)
            .cloned()
        {
            for owner in owners {
                self.touch_qwen35_q2q3_bundle(owner);
            }
            return;
        }
        let epoch = self.next_resident_q4k_epoch();
        if let Some(entry) = self.resident_q4k.get_mut(&key) {
            entry.epoch = epoch;
            self.resident_q4k_lru.push_back((key, epoch));
        }
    }

    pub(in crate::runtime) fn next_resident_q4k_epoch(&mut self) -> u64 {
        self.resident_q4k_epoch = self.resident_q4k_epoch.wrapping_add(1);
        if self.resident_q4k_epoch == 0 {
            self.resident_q4k_epoch = 1;
        }
        self.resident_q4k_epoch
    }

    fn resident_q4k_eviction_candidates(
        &self,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Vec<ResidentQ4kEvictionUnit> {
        let mut ordered = Vec::new();
        for &(bundle, epoch) in &self.qwen35_q2q3_bundle_ownership.bundle_lru {
            if self.qwen35_q2q3_bundle_ownership.bundle_epochs.get(&bundle) == Some(&epoch) {
                ordered.push((epoch, ResidentQ4kLruCandidate::OwnedBundle(bundle)));
            }
        }
        for &(role, epoch) in &self.resident_q4k_lru {
            let Some(entry) = self.resident_q4k.get(&role) else {
                continue;
            };
            if entry.epoch == epoch
                && !self
                    .qwen35_q2q3_bundle_ownership
                    .role_owners
                    .contains_key(&role)
            {
                ordered.push((epoch, ResidentQ4kLruCandidate::UnownedRole(role)));
            }
        }
        ordered.sort_unstable_by(|(left_epoch, left), (right_epoch, right)| {
            left_epoch
                .cmp(right_epoch)
                .then_with(|| match (left, right) {
                    (
                        ResidentQ4kLruCandidate::OwnedBundle(left),
                        ResidentQ4kLruCandidate::OwnedBundle(right),
                    ) => left.cmp(right),
                    (
                        ResidentQ4kLruCandidate::UnownedRole(left),
                        ResidentQ4kLruCandidate::UnownedRole(right),
                    ) => left.cmp(right),
                    (ResidentQ4kLruCandidate::OwnedBundle(_), _) => std::cmp::Ordering::Less,
                    _ => std::cmp::Ordering::Greater,
                })
        });

        let mut considered_roles = HashSet::new();
        let mut considered_bundles = HashSet::new();
        let mut candidates = Vec::new();
        for (_, candidate) in ordered {
            let unit = match candidate {
                ResidentQ4kLruCandidate::OwnedBundle(bundle) => {
                    if considered_bundles.contains(&bundle) {
                        continue;
                    }
                    self.qwen35_q2q3_owned_closure(bundle)
                }
                ResidentQ4kLruCandidate::UnownedRole(role) => {
                    if considered_roles.contains(&role) {
                        continue;
                    }
                    ResidentQ4kEvictionUnit::UnownedRole { role }
                }
            };
            considered_roles.extend(unit.roles().iter().copied());
            considered_bundles.extend(unit.bundles().iter().copied());
            if unit.roles().iter().any(|role| {
                protected_keys.contains(role)
                    || self.resident_q4k.get(role).is_none_or(|entry| entry.pinned)
            }) {
                continue;
            }
            candidates.push(unit);
        }
        candidates
    }

    fn plan_resident_q4k_evictions(
        &self,
        need_release: Option<usize>,
        protected_keys: &HashSet<(usize, usize)>,
        max_reload_payload_bytes: Option<u64>,
        bundles_only: bool,
    ) -> Option<ResidentQ4kEvictionPlan> {
        if need_release == Some(0) {
            return Some(ResidentQ4kEvictionPlan::default());
        }
        let mut plan = ResidentQ4kEvictionPlan::default();
        let mut slab_live_entries = self
            .resident_q4k_slabs
            .iter()
            .map(|(&base, slab)| (base, slab.live_entries))
            .collect::<HashMap<_, _>>();
        let mut arena_live_entries = self
            .resident_q4k
            .values()
            .filter(|entry| self.resident_q4k_entry_is_in_arena(entry))
            .count();
        let mut arena_released = false;

        for unit in self.resident_q4k_eviction_candidates(protected_keys) {
            if bundles_only && unit.bundles().is_empty() {
                continue;
            }
            let unit_reload_payload_bytes = unit
                .roles()
                .iter()
                .filter_map(|role| self.resident_q4k.get(role))
                .fold(0u64, |bytes, entry| {
                    bytes.saturating_add(entry.bytes as u64)
                });
            if max_reload_payload_bytes.is_some_and(|budget| {
                plan.reload_payload_bytes
                    .saturating_add(unit_reload_payload_bytes)
                    > budget
            }) {
                continue;
            }

            for role in unit.roles() {
                let entry = &self.resident_q4k[role];
                if let Some(base) = entry.slab_base {
                    if let Some(live_entries) = slab_live_entries.get_mut(&base) {
                        *live_entries = live_entries.saturating_sub(1);
                        if *live_entries == 0 {
                            plan.releasable_bytes = plan
                                .releasable_bytes
                                .saturating_add(self.resident_q4k_slabs[&base].bytes);
                        }
                    }
                } else if entry.owned_alloc {
                    plan.releasable_bytes = plan.releasable_bytes.saturating_add(entry.bytes);
                } else if self.resident_q4k_entry_is_in_arena(entry) {
                    arena_live_entries = arena_live_entries.saturating_sub(1);
                    if arena_live_entries == 0 && !arena_released {
                        plan.releasable_bytes = plan
                            .releasable_bytes
                            .saturating_add(self.resident_q4k_arena_offset);
                        arena_released = true;
                    }
                }
            }
            plan.reload_payload_bytes = plan
                .reload_payload_bytes
                .saturating_add(unit_reload_payload_bytes);
            plan.units.push(unit);
            if need_release.is_some_and(|needed| plan.releasable_bytes >= needed) {
                return Some(plan);
            }
        }

        match need_release {
            Some(needed) if plan.releasable_bytes < needed => None,
            _ => Some(plan),
        }
    }

    pub(in crate::runtime) fn resident_q4k_eviction_plan_for_incoming(
        &self,
        incoming: usize,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Option<ResidentQ4kEvictionPlan> {
        if incoming > self.resident_q4k_limit {
            return None;
        }
        let need_release = self
            .resident_q4k_bytes
            .saturating_add(incoming)
            .saturating_sub(self.resident_q4k_limit);
        self.plan_resident_q4k_evictions(Some(need_release), protected_keys, None, false)
    }

    pub(in crate::runtime) fn resident_q4k_eviction_cost_bytes_for_incoming(
        &self,
        incoming: usize,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Option<u64> {
        self.resident_q4k_eviction_plan_for_incoming(incoming, protected_keys)
            .map(|plan| plan.reload_payload_bytes)
    }

    pub(in crate::runtime) fn resident_q4k_additional_oom_eviction_plan(
        &self,
        protected_keys: &HashSet<(usize, usize)>,
        max_reload_payload_bytes: u64,
    ) -> Option<ResidentQ4kEvictionPlan> {
        if max_reload_payload_bytes == 0 {
            return None;
        }
        self.plan_resident_q4k_evictions(
            Some(1),
            protected_keys,
            Some(max_reload_payload_bytes),
            true,
        )
    }

    pub(in crate::runtime) fn execute_resident_q4k_eviction_plan(
        &mut self,
        plan: ResidentQ4kEvictionPlan,
    ) -> Result<rnb_memory::ExpertBundleCacheStats, String> {
        let mut delta = rnb_memory::ExpertBundleCacheStats::default();
        if plan.units.is_empty() {
            return Ok(delta);
        }

        // cu27: 다른 layer의 in-flight kernel이 entry/slab ptr를 들고
        // 있을 수 있어서 실제 free 직전 한 번 동기화해.
        self.stream_synchronize()?;
        let mut owned_evicted_payload_bytes = 0u64;
        for unit in plan.units {
            if !unit.bundles().is_empty() {
                delta.bundle_evictions = delta
                    .bundle_evictions
                    .saturating_add(unit.bundles().len() as u64);
                for &role in unit.roles() {
                    if let Some(entry) = self.resident_q4k.get(&role) {
                        owned_evicted_payload_bytes =
                            owned_evicted_payload_bytes.saturating_add(entry.bytes as u64);
                    }
                    self.qwen35_q2q3_bundle_ownership.role_owners.remove(&role);
                }
                for &bundle in unit.bundles() {
                    self.qwen35_q2q3_bundle_ownership
                        .bundle_roles
                        .remove(&bundle);
                    self.qwen35_q2q3_bundle_ownership
                        .bundle_epochs
                        .remove(&bundle);
                }
            }
            for &role in unit.roles() {
                if let Some(entry) = self.resident_q4k.remove(&role) {
                    let _ = self.free_resident_q4k_entry(&entry)?;
                    cache_stats().evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        delta.evicted_bytes = owned_evicted_payload_bytes;
        if owned_evicted_payload_bytes > 0 {
            self.qwen35_q2q3_resident_payload_bytes = self
                .qwen35_q2q3_resident_payload_bytes
                .saturating_sub(owned_evicted_payload_bytes);
            cache_stats().remove_expert_bundle_resident_payload(owned_evicted_payload_bytes);
        }
        if delta.bundle_evictions > 0 {
            cache_stats().record_expert_bundles(delta);
        }
        let released_arena = self.release_resident_q4k_arena_if_unreferenced()?;
        if released_arena > 0 {
            cache_stats().evictions.fetch_add(1, Ordering::Relaxed);
        }
        self.refresh_resident_q4k_bytes();
        let role_owners = &self.qwen35_q2q3_bundle_ownership.role_owners;
        let resident_q4k = &self.resident_q4k;
        self.resident_q4k_lru.retain(|(role, epoch)| {
            !role_owners.contains_key(role)
                && resident_q4k
                    .get(role)
                    .is_some_and(|entry| entry.epoch == *epoch)
        });
        let ResidentQ4kBundleOwnership {
            bundle_epochs,
            bundle_lru,
            ..
        } = &mut self.qwen35_q2q3_bundle_ownership;
        bundle_lru.retain(|(bundle, epoch)| bundle_epochs.get(bundle) == Some(epoch));
        Ok(delta)
    }

    pub(in crate::runtime) fn evict_resident_q4k_until(
        &mut self,
        incoming: usize,
    ) -> Result<(), String> {
        self.evict_resident_q4k_until_protecting(incoming, &HashSet::new())
    }

    pub(in crate::runtime) fn evict_resident_q4k_until_protecting(
        &mut self,
        incoming: usize,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Result<(), String> {
        if let Some(plan) = self.resident_q4k_eviction_plan_for_incoming(incoming, protected_keys) {
            let _ = self.execute_resident_q4k_eviction_plan(plan)?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn free_resident_q4k_entry(
        &mut self,
        entry: &ResidentQ4k,
    ) -> Result<usize, String> {
        if let Some(base) = entry.slab_base {
            if let Some(slab) = self.resident_q4k_slabs.get_mut(&base) {
                slab.live_entries = slab.live_entries.saturating_sub(1);
                if slab.live_entries == 0 {
                    let bytes = slab.bytes;
                    self.resident_q4k_slabs.remove(&base);
                    unsafe { self.api.mem_free(base)? };
                    return Ok(bytes);
                }
            }
        } else if entry.owned_alloc {
            unsafe { self.api.mem_free(entry.ptr)? };
            return Ok(entry.bytes);
        }
        Ok(0)
    }

    pub(in crate::runtime) fn clear_resident_q4k_cache(&mut self) -> Result<(), String> {
        self.set_current()?;
        // cu27: in-flight kernel safety — clear_*_cache 류는 launch param으로
        // 잡혀있는 ptr까지 free하므로 stream sync 필수.
        if !self.resident_q4k.is_empty()
            || !self.resident_q4k_slabs.is_empty()
            || self.resident_q4k_arena.is_some()
        {
            self.stream_synchronize()?;
        }
        self.record_qwen35_q2q3_cache_clear();
        for (_, entry) in self.resident_q4k.drain() {
            if entry.slab_base.is_none() && entry.owned_alloc {
                unsafe { self.api.mem_free(entry.ptr)? };
            }
        }
        for (ptr, _) in self.resident_q4k_slabs.drain() {
            unsafe { self.api.mem_free(ptr)? };
        }
        if let Some(ptr) = self.resident_q4k_arena.take() {
            unsafe { self.api.mem_free(ptr)? };
        }
        for (_, entry) in self.resident_q8_f32.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        for (_, entry) in self.resident_q8_quant.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_q8_f32_lru.clear();
        self.resident_q8_f32_bytes = 0;
        self.resident_q4k_lru.clear();
        self.resident_q4k_bytes = 0;
        self.qwen35_target_decode_q4k_limit_checked = false;
        self.nemotron_decode_q4k_limit_checked = false;
        self.resident_q4k_touch_hits_auto = false;
        self.qwen35_selected_base_admission_history.clear();
        self.qwen35_expert_bundle_reuse_history = rnb_memory::ExpertBundleReuseHistory::new(0);
        self.resident_q4k_arena_capacity = 0;
        self.resident_q4k_arena_offset = 0;
        self.qwen35_mtp_expert_history.clear();
        self.qwen35_mtp_expert_observations.clear();
        Ok(())
    }
}
