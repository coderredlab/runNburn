use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use rnb_backend_api::{BackendKind, MoeJitLoadSink};

#[cfg(feature = "cuda")]
static DEFAULT_CUDA_MOE_JIT_LOADER: OnceLock<Arc<rnb_backend_cuda::CudaMoeJitLoader>> =
    OnceLock::new();

pub fn default_moe_jit_loader(backend: Option<BackendKind>) -> Option<Arc<dyn MoeJitLoadSink>> {
    match backend {
        #[cfg(feature = "cuda")]
        Some(BackendKind::Cuda) => Some(
            DEFAULT_CUDA_MOE_JIT_LOADER
                .get_or_init(|| Arc::new(rnb_backend_cuda::CudaMoeJitLoader::default()))
                .clone(),
        ),
        _ => None,
    }
}

pub fn moe_jit_report() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        let loader = DEFAULT_CUDA_MOE_JIT_LOADER.get()?;
        let stats = loader.stats();
        let hit_rate = if stats.cache.lookups == 0 {
            0.0
        } else {
            (stats.cache.hits as f64 * 100.0) / stats.cache.lookups as f64
        };
        return Some(format!(
            "moe_jit_cuda requests={} experts={} requested_bytes={} copied_bytes={} resident_bytes={} failures={} cache_lookups={} cache_hits={} cache_misses={} cache_hit_rate={:.1}% cache_evictions={} resident_upload_bytes={} temp_upload_bytes={}",
            stats.requests,
            stats.requested_experts,
            stats.requested_bytes,
            stats.copied_bytes,
            stats.resident_bytes,
            stats.cuda_failures,
            stats.cache.lookups,
            stats.cache.hits,
            stats.cache.misses,
            hit_rate,
            stats.cache.evictions,
            stats.cache.resident_upload_bytes,
            stats.cache.temp_upload_bytes,
        ));
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}
