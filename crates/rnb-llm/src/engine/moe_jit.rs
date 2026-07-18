//! MoE JIT expert loader contract.
//!
//! This module is intentionally backend-neutral. The MoE router can hand the
//! selected experts to this sink immediately after routing, and concrete CUDA /
//! Vulkan / OpenCL loaders can decide how to stage or cache the bytes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

pub use crate::runtime::{MoeJitByteRange, MoeJitExpertLoad, MoeJitLoadRequest, MoeJitLoadSink};

static PRELOAD_SUPPRESS_DEPTH: AtomicUsize = AtomicUsize::new(0);

pub struct MoeJitPreloadSuppressGuard;

impl Drop for MoeJitPreloadSuppressGuard {
    fn drop(&mut self) {
        PRELOAD_SUPPRESS_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn suppress_preload_requests() -> MoeJitPreloadSuppressGuard {
    PRELOAD_SUPPRESS_DEPTH.fetch_add(1, Ordering::Relaxed);
    MoeJitPreloadSuppressGuard
}

fn preload_requests_suppressed() -> bool {
    PRELOAD_SUPPRESS_DEPTH.load(Ordering::Relaxed) > 0
}

pub use crate::runtime::BackendKind as MoeJitBackendKind;

pub fn moe_jit_backend_from_env_cached() -> Option<MoeJitBackendKind> {
    static BACKEND: OnceLock<Option<MoeJitBackendKind>> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        crate::runtime::GpuBackend::from_moe_jit_env().and_then(|backend| backend.backend_kind())
    })
}

static GLOBAL_JIT_LOADER: OnceLock<RwLock<Option<Arc<dyn MoeJitLoadSink>>>> = OnceLock::new();

fn global_loader() -> &'static RwLock<Option<Arc<dyn MoeJitLoadSink>>> {
    GLOBAL_JIT_LOADER.get_or_init(|| RwLock::new(None))
}

pub fn moe_jit_loader_registered() -> bool {
    let Some(lock) = GLOBAL_JIT_LOADER.get() else {
        return false;
    };
    lock.read().expect("moe jit loader lock poisoned").is_some()
}

pub fn request_moe_jit_load(request: &MoeJitLoadRequest) {
    if preload_requests_suppressed() {
        return;
    }
    let loader = global_loader()
        .read()
        .expect("moe jit loader lock poisoned")
        .clone();
    let loader = match loader {
        Some(loader) => loader,
        None => {
            let Some(loader) = crate::runtime::default_moe_jit_loader(request.backend_hint) else {
                return;
            };
            *global_loader()
                .write()
                .expect("moe jit loader lock poisoned") = Some(loader.clone());
            loader
        }
    };
    loader.request_load(request);
}

pub fn moe_jit_report() -> Option<String> {
    crate::runtime::moe_jit_report()
}

#[cfg(test)]
pub fn set_moe_jit_loader_for_test(loader: Option<Arc<dyn MoeJitLoadSink>>) {
    *global_loader()
        .write()
        .expect("moe jit loader lock poisoned") = loader;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_backend_defaults_to_off() {
        let _guard = ENV_LOCK.get_or_init(Default::default).lock().unwrap();
        unsafe {
            std::env::remove_var("RNB_MOE_JIT_BACKEND");
        }

        assert_eq!(moe_jit_backend_from_env_cached(), None);
    }

    #[test]
    fn jit_backend_accepts_explicit_off() {
        let _guard = ENV_LOCK.get_or_init(Default::default).lock().unwrap();
        unsafe {
            std::env::set_var("RNB_MOE_JIT_BACKEND", "off");
        }

        assert_eq!(moe_jit_backend_from_env_cached(), None);
        unsafe {
            std::env::remove_var("RNB_MOE_JIT_BACKEND");
        }
    }

    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
}
