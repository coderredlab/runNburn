use std::path::Path;
use std::sync::Arc;

pub use rnb_loader::packed::{PackedModel, PackedWeight};
pub use rnb_loader::QuantType;

/// Open an explicitly selected v3 RNBC sidecar for diagnostics.
///
/// Product inference never calls this path automatically; GGUF remains the
/// sole default model source.
pub fn open_sidecar_v3_packed_model(sidecar_path: &Path) -> Option<Arc<PackedModel>> {
    if crate::policy::force_gguf_enabled() {
        eprintln!("[INFO] RNB_FORCE_GGUF set, ignoring v3 sidecar cache");
        return None;
    }
    match PackedModel::from_v3_sidecar(sidecar_path) {
        Ok(pm) => {
            eprintln!(
                "[INFO] v3 sidecar cache loaded: {} tensors from {}",
                pm.weights.len(),
                sidecar_path.display()
            );
            Some(Arc::new(pm))
        }
        Err(e) => {
            eprintln!(
                "[WARN] failed to load v3 sidecar {}: {}",
                sidecar_path.display(),
                e
            );
            None
        }
    }
}

pub fn open_shadow_model(path: &Path) -> Option<Arc<PackedModel>> {
    if !crate::policy::shadow_weights_requested() {
        return None;
    }

    let shadow_path = path.with_extension("shadow.rnb");
    if shadow_path.exists() {
        match PackedModel::open(&shadow_path) {
            Ok(pm) => {
                eprintln!(
                    "[INFO] MoE mixed-precision shadow loaded: {} tensors from {:?}",
                    pm.weights.len(),
                    shadow_path.file_name().unwrap_or_default()
                );
                Some(Arc::new(pm))
            }
            Err(e) => {
                eprintln!("[WARN] Failed to open shadow.rnb {:?}: {}", shadow_path, e);
                None
            }
        }
    } else {
        None
    }
}
