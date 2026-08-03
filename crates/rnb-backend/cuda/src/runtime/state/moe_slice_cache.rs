//! Device-resident MoE expert weight slice cache.
//!
//! Keyed by the same `q4k_resident_key` identity as temp uploads, this LRU
//! keeps original quantized expert slices (any GGML layout — the bytes are
//! opaque) resident in VRAM so hot experts skip host reads and H2D copies.
//! The cache never changes arithmetic: hits and misses feed the exact same
//! kernels with identical bytes, so outputs are independent of cache state.

use super::super::*;

#[derive(Default)]
pub(in crate::runtime) struct MoeSliceCache {
    entries: HashMap<(usize, usize), MoeSliceEntry>,
    /// Evicted buffers bucketed by byte size for stream-ordered reuse.
    /// All cache traffic runs on the compute stream, so re-uploading into a
    /// reused buffer is ordered after every queued kernel that read it —
    /// no synchronization or cuMemFree is needed on eviction.
    free_lists: HashMap<usize, Vec<u64>>,
    epoch: u64,
    resident_bytes: usize,
    /// Resolved once from env + device memory info; `Some(0)` = disabled.
    budget_bytes: Option<usize>,
    /// Deferred shrink target when an OOM clamp could not evict enough
    /// because every entry was in use by the clamping call; applied at the
    /// start of the next call before any entry is marked used.
    pending_shrink: Option<usize>,
    lookups: u64,
    hits: u64,
    admissions: u64,
    evictions: u64,
}

struct MoeSliceEntry {
    ptr: u64,
    len: usize,
    last_use: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoeSliceCacheEnv {
    Auto,
    Disabled,
    FixedMib(usize),
}

fn parse_moe_slice_cache_env() -> MoeSliceCacheEnv {
    let Ok(raw) = std::env::var("RNB_CUDA_MOE_EXPERT_CACHE_MB") else {
        return MoeSliceCacheEnv::Auto;
    };
    let raw = raw.trim();
    if raw.is_empty()
        || matches!(
            raw.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    {
        return MoeSliceCacheEnv::Disabled;
    }
    if raw.eq_ignore_ascii_case("auto") {
        return MoeSliceCacheEnv::Auto;
    }
    match raw.parse::<usize>() {
        Ok(mib) if mib > 0 => MoeSliceCacheEnv::FixedMib(mib),
        _ => MoeSliceCacheEnv::Disabled,
    }
}

/// Reserve kept free for other tenants and transient per-run workspaces
/// (prefill temp slabs, logits uploads, verify scratch). Delegates to the
/// shared quant-resident reserve policy so the device keeps one
/// capacity-proportional reserve definition instead of a second constant.
fn moe_slice_cache_reserve_bytes(total_bytes: usize) -> usize {
    super::quant_resident::quant_resident_reserve_mib(total_bytes / (1024 * 1024))
        .saturating_mul(1024 * 1024)
}

fn moe_slice_cache_budget_bytes(env: MoeSliceCacheEnv, free: usize, total: usize) -> usize {
    match env {
        MoeSliceCacheEnv::Disabled => 0,
        MoeSliceCacheEnv::FixedMib(mib) => mib.saturating_mul(1024 * 1024),
        MoeSliceCacheEnv::Auto => free.saturating_sub(moe_slice_cache_reserve_bytes(total)),
    }
}

fn moe_slice_cache_trace_enabled() -> bool {
    std::env::var("RNB_CUDA_MOE_EXPERT_CACHE_TRACE")
        .ok()
        .as_deref()
        == Some("1")
}

impl CudaState {
    /// Resolve per-slot device pointers for gate/up/down expert slices,
    /// serving hits from the resident cache and uploading misses into
    /// newly admitted resident buffers. Returns `Ok(None)` when the cache
    /// is disabled or the request cannot fit its budget.
    pub(in crate::runtime) fn moe_slice_resident_ptrs_3(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<Option<(Vec<u64>, Vec<u64>, Vec<u64>)>, String> {
        let budget = match self.moe_slice_cache.budget_bytes {
            Some(budget) => budget,
            None => {
                let env = parse_moe_slice_cache_env();
                let (free, total) = unsafe { self.api.mem_get_info()? };
                let budget = moe_slice_cache_budget_bytes(env, free, total);
                self.moe_slice_cache.budget_bytes = Some(budget);
                if moe_slice_cache_trace_enabled() {
                    eprintln!(
                        "[cuda-moe-slice-cache] budget={:.1}MiB free={:.1}MiB total={:.1}MiB env={env:?}",
                        budget as f64 / (1024.0 * 1024.0),
                        free as f64 / (1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0),
                    );
                }
                budget
            }
        };
        if budget == 0 {
            return Ok(None);
        }
        // Apply a shrink deferred from an OOM clamp; no entry carries the
        // upcoming epoch yet, so the whole cache is evictable here.
        if let Some(target) = self.moe_slice_cache.pending_shrink.take() {
            if self.moe_slice_cache.resident_bytes > target {
                let shrink_epoch = self.moe_slice_cache.epoch.wrapping_add(1);
                self.shrink_moe_slice_cache_to(target, shrink_epoch)?;
            }
        }

        // Unique slices in first-appearance order.
        let mut order: Vec<((usize, usize), &[u8])> = Vec::new();
        let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
        for &weights in gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter())
        {
            let key = q4k_resident_key(weights);
            if let std::collections::hash_map::Entry::Vacant(entry) = seen.entry(key) {
                entry.insert(order.len());
                order.push((key, weights));
            }
        }
        if order.is_empty() {
            return Ok(Some((Vec::new(), Vec::new(), Vec::new())));
        }

        self.moe_slice_cache.epoch = self.moe_slice_cache.epoch.wrapping_add(1);
        let epoch = self.moe_slice_cache.epoch;

        for (key, _) in &order {
            self.moe_slice_cache.lookups += 1;
            if let Some(entry) = self.moe_slice_cache.entries.get_mut(key) {
                entry.last_use = epoch;
                self.moe_slice_cache.hits += 1;
            }
        }

        // Admit misses. Priority per miss of size L:
        //   1. reuse a same-size buffer from the free list
        //   2. fresh cuMemAlloc while under budget (OOM clamps the budget)
        //   3. steal the LRU resident entry of the same size
        //   4. overflow into the shared temp slab for this call only
        // Every path feeds the exact same kernels, so cache pressure never
        // changes arithmetic — only transfer cost. Reused/stolen buffers
        // are safe without synchronization because uploads ride the same
        // stream as every kernel that could still read them.
        let mut temp_overflow: Vec<(usize, usize)> = Vec::new(); // (order idx, bytes)
        for (index, (key, weights)) in order.iter().enumerate() {
            if self.moe_slice_cache.entries.contains_key(key) {
                continue;
            }
            let len = weights.len();
            let mut ptr = self
                .moe_slice_cache
                .free_lists
                .get_mut(&len)
                .and_then(Vec::pop);
            if ptr.is_none() {
                let budget = self
                    .moe_slice_cache
                    .budget_bytes
                    .expect("moe slice cache budget resolved");
                if self.moe_slice_cache.resident_bytes.saturating_add(len) <= budget {
                    match unsafe { self.api.mem_alloc(len) } {
                        Ok(fresh) => {
                            self.moe_slice_cache.resident_bytes =
                                self.moe_slice_cache.resident_bytes.saturating_add(len);
                            ptr = Some(fresh);
                        }
                        Err(_) => {
                            // Device is tighter than the resolved budget.
                            // Back off to 3/4 of what we actually hold so
                            // the other tenants stop OOM-retrying. Entries
                            // used by this call are protected, so defer the
                            // remainder to the next call when the whole
                            // cache is evictable again.
                            let target = self.moe_slice_cache.resident_bytes.saturating_mul(3) / 4;
                            self.moe_slice_cache.budget_bytes = Some(target);
                            self.shrink_moe_slice_cache_to(target, epoch)?;
                            if self.moe_slice_cache.resident_bytes > target {
                                self.moe_slice_cache.pending_shrink = Some(target);
                            }
                        }
                    }
                }
            }
            if ptr.is_none() {
                let victim = self
                    .moe_slice_cache
                    .entries
                    .iter()
                    .filter(|(_, entry)| entry.len == len && entry.last_use != epoch)
                    .min_by_key(|(_, entry)| entry.last_use)
                    .map(|(&key, _)| key);
                if let Some(victim) = victim {
                    let entry = self
                        .moe_slice_cache
                        .entries
                        .remove(&victim)
                        .expect("victim entry present");
                    self.moe_slice_cache.evictions += 1;
                    ptr = Some(entry.ptr);
                }
            }
            let Some(ptr) = ptr else {
                temp_overflow.push((index, len));
                continue;
            };
            unsafe {
                self.api.memcpy_htod_async(
                    ptr,
                    weights.as_ptr().cast::<libc::c_void>(),
                    len,
                    self.stream,
                )?;
            }
            self.moe_slice_cache.entries.insert(
                *key,
                MoeSliceEntry {
                    ptr,
                    len,
                    last_use: epoch,
                },
            );
            self.moe_slice_cache.admissions += 1;
        }

        // Overflow slices share one temp slab upload for this call.
        let mut temp_ptrs: HashMap<(usize, usize), u64> = HashMap::new();
        if !temp_overflow.is_empty() {
            let total: usize = temp_overflow.iter().map(|(_, len)| len).sum();
            // Even the shared temp slab can fail under extreme pressure;
            // the cache is then unavailable and the caller must take its
            // host path instead of failing the whole forward.
            let Ok(slab) = self.compute_temp_slab_ptr(total) else {
                return Ok(None);
            };
            let mut offset = 0usize;
            for (index, len) in temp_overflow {
                let (key, weights) = order[index];
                unsafe {
                    self.api.memcpy_htod_async(
                        slab + offset as u64,
                        weights.as_ptr().cast::<libc::c_void>(),
                        len,
                        self.stream,
                    )?;
                }
                temp_ptrs.insert(key, slab + offset as u64);
                offset += len;
            }
        }

        if moe_slice_cache_trace_enabled()
            && self.moe_slice_cache.lookups % 100_000 < order.len() as u64
        {
            eprintln!(
                "[cuda-moe-slice-cache] lookups={} hits={} hit_rate={:.1}% admissions={} evictions={} resident={:.1}MiB entries={} temp_overflow={}",
                self.moe_slice_cache.lookups,
                self.moe_slice_cache.hits,
                self.moe_slice_cache.hits as f64 * 100.0 / self.moe_slice_cache.lookups.max(1) as f64,
                self.moe_slice_cache.admissions,
                self.moe_slice_cache.evictions,
                self.moe_slice_cache.resident_bytes as f64 / (1024.0 * 1024.0),
                self.moe_slice_cache.entries.len(),
                temp_ptrs.len(),
            );
        }

        let make_ptrs = |slot_weights: &[&[u8]]| {
            slot_weights
                .iter()
                .map(|weights| {
                    let key = q4k_resident_key(weights);
                    self.moe_slice_cache
                        .entries
                        .get(&key)
                        .map(|entry| entry.ptr)
                        .or_else(|| temp_ptrs.get(&key).copied())
                        .expect("moe slice ptr resolved")
                })
                .collect::<Vec<_>>()
        };
        Ok(Some((
            make_ptrs(gate_weights),
            make_ptrs(up_weights),
            make_ptrs(down_weights),
        )))
    }

    pub(in crate::runtime) fn clear_moe_slice_cache(&mut self) -> Result<(), String> {
        if self.moe_slice_cache.entries.is_empty() && self.moe_slice_cache.free_lists.is_empty() {
            self.moe_slice_cache.budget_bytes = None;
            return Ok(());
        }
        self.set_current()?;
        unsafe { self.api.stream_synchronize(self.stream)? };
        let entries = std::mem::take(&mut self.moe_slice_cache.entries);
        for (_, entry) in entries {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        let free_lists = std::mem::take(&mut self.moe_slice_cache.free_lists);
        for (_, ptrs) in free_lists {
            for ptr in ptrs {
                unsafe { self.api.mem_free(ptr)? };
            }
        }
        self.moe_slice_cache = MoeSliceCache::default();
        Ok(())
    }

    /// Hard-shrink the cache to `target` bytes: frees LRU entries (except
    /// ones touched in the current call epoch) and every free-list buffer.
    /// One-time stream sync — only runs on an OOM clamp event.
    fn shrink_moe_slice_cache_to(&mut self, target: usize, epoch: u64) -> Result<(), String> {
        unsafe { self.api.stream_synchronize(self.stream)? };
        let free_lists = std::mem::take(&mut self.moe_slice_cache.free_lists);
        for (_, ptrs) in free_lists {
            for ptr in ptrs {
                unsafe { self.api.mem_free(ptr)? };
            }
        }
        while self.moe_slice_cache.resident_bytes > target {
            let victim = self
                .moe_slice_cache
                .entries
                .iter()
                .filter(|(_, entry)| entry.last_use != epoch)
                .min_by_key(|(_, entry)| entry.last_use)
                .map(|(&key, _)| key);
            let Some(victim) = victim else {
                break;
            };
            let entry = self
                .moe_slice_cache
                .entries
                .remove(&victim)
                .expect("shrink victim present");
            unsafe { self.api.mem_free(entry.ptr)? };
            self.moe_slice_cache.resident_bytes = self
                .moe_slice_cache
                .resident_bytes
                .saturating_sub(entry.len);
            self.moe_slice_cache.evictions += 1;
        }
        Ok(())
    }

    /// Total device bytes held by the cache including recycled buffers on
    /// the free lists — this is what unified reclaim accounting sees.
    pub(in crate::runtime) fn moe_slice_cache_held_bytes(&self) -> usize {
        let free_list_bytes: usize = self
            .moe_slice_cache
            .free_lists
            .iter()
            .map(|(len, ptrs)| len.saturating_mul(ptrs.len()))
            .sum();
        self.moe_slice_cache
            .resident_bytes
            .saturating_add(free_list_bytes)
    }

    /// Release at least `bytes` from the cache for transient allocations.
    /// Entries handed out in the current call epoch stay protected; the
    /// lowered budget prevents immediate regrowth into the reclaimed room.
    pub(in crate::runtime) fn shrink_moe_slice_cache_for_reclaim(
        &mut self,
        bytes: usize,
    ) -> Result<(), String> {
        if self.moe_slice_cache.entries.is_empty() && self.moe_slice_cache.free_lists.is_empty() {
            return Ok(());
        }
        let target = self.moe_slice_cache.resident_bytes.saturating_sub(bytes);
        self.moe_slice_cache.budget_bytes = Some(target);
        let epoch = self.moe_slice_cache.epoch;
        self.shrink_moe_slice_cache_to(target, epoch)?;
        if self.moe_slice_cache.resident_bytes > target {
            self.moe_slice_cache.pending_shrink = Some(target);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_uses_env_override_and_disables_on_zero() {
        assert_eq!(
            moe_slice_cache_budget_bytes(MoeSliceCacheEnv::FixedMib(512), 0, 0),
            512 * 1024 * 1024
        );
        assert_eq!(
            moe_slice_cache_budget_bytes(MoeSliceCacheEnv::Disabled, usize::MAX, usize::MAX),
            0
        );
    }

    #[test]
    fn auto_budget_scales_with_free_vram_and_keeps_reserve() {
        let total = 24 * 1024 * 1024 * 1024usize;
        let free = 23 * 1024 * 1024 * 1024usize;
        let budget = moe_slice_cache_budget_bytes(MoeSliceCacheEnv::Auto, free, total);
        let reserve = moe_slice_cache_reserve_bytes(total);
        assert_eq!(budget, free - reserve);
        assert!(budget > 10 * 1024 * 1024 * 1024);
        // Small device: reserve floor dominates, budget shrinks to zero.
        let small_total = 2 * 1024 * 1024 * 1024usize;
        let small_free = 1024 * 1024 * 1024usize;
        assert_eq!(
            moe_slice_cache_budget_bytes(MoeSliceCacheEnv::Auto, small_free, small_total),
            0
        );
    }
}
