use super::*;
use rnb_backend_api::{MoeJitLoadRequest, MoeJitLoadSink};
use rnb_memory::ExpertBundleCacheStats;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static CUDA_CACHE_STATS: OnceLock<CudaCacheStats> = OnceLock::new();

#[derive(Default)]
pub(super) struct CudaCacheStats {
    pub(super) lookups: AtomicU64,
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
    pub(super) evictions: AtomicU64,
    pub(super) resident_upload_bytes: AtomicU64,
    pub(super) temp_upload_bytes: AtomicU64,
    expert_bundles: Mutex<ExpertBundleCacheStats>,
    expert_bundle_resident_payload_bytes: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaCacheSnapshot {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_upload_bytes: u64,
    pub temp_upload_bytes: u64,
    pub expert_bundles: ExpertBundleCacheStats,
    /// Logical Q2_K/Q3_K bundle payload currently resident. Slab padding is excluded.
    pub resident_payload_bytes: u64,
}

impl CudaCacheStats {
    fn lock_expert_bundles(&self) -> std::sync::MutexGuard<'_, ExpertBundleCacheStats> {
        self.expert_bundles
            .lock()
            .expect("cuda expert bundle cache stats lock poisoned")
    }

    pub(super) fn record_expert_bundles(&self, delta: ExpertBundleCacheStats) {
        let mut stats = self.lock_expert_bundles();
        stats.bundle_lookups = stats.bundle_lookups.wrapping_add(delta.bundle_lookups);
        stats.bundle_hits = stats.bundle_hits.wrapping_add(delta.bundle_hits);
        stats.bundle_partial_hits = stats
            .bundle_partial_hits
            .wrapping_add(delta.bundle_partial_hits);
        stats.bundle_misses = stats.bundle_misses.wrapping_add(delta.bundle_misses);
        stats.bundle_admissions = stats
            .bundle_admissions
            .wrapping_add(delta.bundle_admissions);
        stats.bundle_evictions = stats.bundle_evictions.wrapping_add(delta.bundle_evictions);
        stats.admitted_bytes = stats.admitted_bytes.wrapping_add(delta.admitted_bytes);
        stats.evicted_bytes = stats.evicted_bytes.wrapping_add(delta.evicted_bytes);
        stats.h2d_bytes = stats.h2d_bytes.wrapping_add(delta.h2d_bytes);
        stats.temp_h2d_bytes = stats.temp_h2d_bytes.wrapping_add(delta.temp_h2d_bytes);
    }

    pub(super) fn record_expert_bundle_h2d(&self, payload_bytes: u64, temporary: bool) {
        let mut stats = self.lock_expert_bundles();
        stats.h2d_bytes = stats.h2d_bytes.wrapping_add(payload_bytes);
        if temporary {
            stats.temp_h2d_bytes = stats.temp_h2d_bytes.wrapping_add(payload_bytes);
        }
    }

    pub(super) fn add_expert_bundle_resident_payload(&self, payload_bytes: u64) {
        self.expert_bundle_resident_payload_bytes
            .fetch_add(payload_bytes, Ordering::Relaxed);
    }

    pub(super) fn remove_expert_bundle_resident_payload(&self, payload_bytes: u64) {
        let _ = self.expert_bundle_resident_payload_bytes.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |resident| Some(resident.saturating_sub(payload_bytes)),
        );
    }

    pub(super) fn expert_bundles(&self) -> ExpertBundleCacheStats {
        *self.lock_expert_bundles()
    }
}
pub(super) fn cache_stats() -> &'static CudaCacheStats {
    CUDA_CACHE_STATS.get_or_init(CudaCacheStats::default)
}

pub fn cache_snapshot() -> CudaCacheSnapshot {
    let stats = cache_stats();
    CudaCacheSnapshot {
        lookups: stats.lookups.load(Ordering::Relaxed),
        hits: stats.hits.load(Ordering::Relaxed),
        misses: stats.misses.load(Ordering::Relaxed),
        evictions: stats.evictions.load(Ordering::Relaxed),
        resident_upload_bytes: stats.resident_upload_bytes.load(Ordering::Relaxed),
        temp_upload_bytes: stats.temp_upload_bytes.load(Ordering::Relaxed),
        expert_bundles: stats.expert_bundles(),
        resident_payload_bytes: stats
            .expert_bundle_resident_payload_bytes
            .load(Ordering::Relaxed),
    }
}

impl CudaCacheSnapshot {
    pub(super) fn delta(self, before: Self) -> Self {
        Self {
            lookups: self.lookups.saturating_sub(before.lookups),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            evictions: self.evictions.saturating_sub(before.evictions),
            resident_upload_bytes: self
                .resident_upload_bytes
                .saturating_sub(before.resident_upload_bytes),
            temp_upload_bytes: self
                .temp_upload_bytes
                .saturating_sub(before.temp_upload_bytes),
            expert_bundles: self.expert_bundles.delta(before.expert_bundles),
            resident_payload_bytes: self.resident_payload_bytes,
        }
    }
}

#[derive(Default)]
pub struct CudaMoeJitLoader {
    requests: AtomicU64,
    requested_experts: AtomicU64,
    requested_bytes: AtomicU64,
    copied_bytes: AtomicU64,
    resident_bytes: AtomicU64,
    cuda_failures: AtomicU64,
}

impl CudaMoeJitLoader {
    pub fn stats(&self) -> CudaMoeJitStats {
        CudaMoeJitStats {
            requests: self.requests.load(Ordering::Relaxed),
            requested_experts: self.requested_experts.load(Ordering::Relaxed),
            requested_bytes: self.requested_bytes.load(Ordering::Relaxed),
            copied_bytes: self.copied_bytes.load(Ordering::Relaxed),
            resident_bytes: self.resident_bytes.load(Ordering::Relaxed),
            cuda_failures: self.cuda_failures.load(Ordering::Relaxed),
            cache: cache_snapshot(),
        }
    }

    fn preload_request_to_cuda(&self, request: &MoeJitLoadRequest) -> Result<(), String> {
        let total_bytes = request.requested_bytes();
        if total_bytes == 0 {
            return Ok(());
        }
        let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
        let mut guard = compute
            .lock()
            .map_err(|_| "cuda compute state lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(CudaState::open()?);
        }
        let state = guard.as_mut().expect("cuda compute state initialized");
        let mut copied = 0usize;
        for expert in &request.expert_loads {
            if state.preload_resident_q4k_range(expert.gate)? {
                copied = copied.saturating_add(expert.gate.len);
            }
            if state.preload_resident_q4k_range(expert.up)? {
                copied = copied.saturating_add(expert.up.len);
            }
            if state.preload_resident_q4k_range(expert.down)? {
                copied = copied.saturating_add(expert.down.len);
            }
        }
        self.copied_bytes
            .fetch_add(copied as u64, Ordering::Relaxed);
        self.resident_bytes
            .store(state.resident_q4k_bytes as u64, Ordering::Relaxed);
        Ok(())
    }
}

impl MoeJitLoadSink for CudaMoeJitLoader {
    fn request_load(&self, request: &MoeJitLoadRequest) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.requested_experts
            .fetch_add(request.experts.len() as u64, Ordering::Relaxed);
        self.requested_bytes
            .fetch_add(request.requested_bytes() as u64, Ordering::Relaxed);
        if let Err(err) = self.preload_request_to_cuda(request) {
            self.cuda_failures.fetch_add(1, Ordering::Relaxed);
            if std::env::var("RNB_MOE_JIT_LOG").is_ok() {
                eprintln!("[moe-jit:cuda] request failed: {err}");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMoeJitStats {
    pub requests: u64,
    pub requested_experts: u64,
    pub requested_bytes: u64,
    pub copied_bytes: u64,
    pub resident_bytes: u64,
    pub cuda_failures: u64,
    pub cache: CudaCacheSnapshot,
}
