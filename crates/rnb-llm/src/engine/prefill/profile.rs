use super::*;

#[cfg(feature = "cuda")]
use crate::engine::backend_runtime::cuda_cache_snapshot;
#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime::CudaCacheSnapshot;

pub(super) struct PrefillLayerProfiler {
    enabled: bool,
    times_ms: Vec<(usize, f64)>,
    #[cfg(feature = "cuda")]
    cache_before: Option<CudaCacheSnapshot>,
}

impl PrefillLayerProfiler {
    pub(super) fn new(n_layers: usize) -> Self {
        let enabled = policy::prefill_layer_profile_enabled();
        Self {
            enabled,
            times_ms: Vec::with_capacity(n_layers),
            #[cfg(feature = "cuda")]
            cache_before: if enabled {
                Some(cuda_cache_snapshot())
            } else {
                None
            },
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn record(&mut self, layer_idx: usize, elapsed_ms: f64) {
        if self.enabled {
            self.times_ms.push((layer_idx, elapsed_ms));
        }
    }

    pub(super) fn report(&self, weights: &ModelWeights) {
        if !self.enabled {
            return;
        }

        #[cfg(feature = "cuda")]
        if let Some(before) = self.cache_before {
            let after = cuda_cache_snapshot();
            let lookups = after.lookups.saturating_sub(before.lookups);
            let hits = after.hits.saturating_sub(before.hits);
            let misses = after.misses.saturating_sub(before.misses);
            let resident_mb = (after
                .resident_upload_bytes
                .saturating_sub(before.resident_upload_bytes)) as f64
                / (1024.0 * 1024.0);
            let temp_mb = (after
                .temp_upload_bytes
                .saturating_sub(before.temp_upload_bytes)) as f64
                / (1024.0 * 1024.0);
            let hit_rate = if lookups > 0 {
                100.0 * hits as f64 / lookups as f64
            } else {
                0.0
            };
            eprintln!(
                "  [PREFILL] cache: lookups={} hits={} miss={} hit_rate={:.1}% resident_upload={:.1}MiB temp_upload={:.1}MiB",
                lookups, hits, misses, hit_rate, resident_mb, temp_mb
            );
        }

        let mut gdn_times = Vec::new();
        let mut atn_times = Vec::new();
        for &(layer_idx, elapsed_ms) in &self.times_ms {
            match &weights.layers[layer_idx] {
                LayerType::GatedDeltaNet(_) => gdn_times.push(elapsed_ms),
                LayerType::Attention(_) => atn_times.push(elapsed_ms),
                LayerType::NemotronMamba2(_) => gdn_times.push(elapsed_ms),
                LayerType::NemotronMoE(_) => gdn_times.push(elapsed_ms),
            }
        }

        let total_ms: f64 = self
            .times_ms
            .iter()
            .map(|&(_, elapsed_ms)| elapsed_ms)
            .sum();
        eprintln!("  [PREFILL] layers_total {:.1}ms", total_ms);
        report_kind("GDN", &gdn_times);
        report_kind("ATN", &atn_times);

        for &(layer_idx, elapsed_ms) in &self.times_ms {
            let kind = match &weights.layers[layer_idx] {
                LayerType::GatedDeltaNet(_) => "GDN",
                LayerType::Attention(_) => "ATN",
                LayerType::NemotronMamba2(_) => "NMT-M",
                LayerType::NemotronMoE(_) => "NMT-E",
            };
            eprintln!(
                "  [PREFILL]   L{:2} ({}) {:.2}ms",
                layer_idx, kind, elapsed_ms
            );
        }
    }
}

fn report_kind(label: &str, times_ms: &[f64]) {
    if times_ms.is_empty() {
        eprintln!("  [PREFILL]   {label} x0: 0.0ms total");
        return;
    }

    let total_ms: f64 = times_ms.iter().sum();
    let min_ms = times_ms.iter().copied().fold(f64::MAX, f64::min);
    let max_ms = times_ms.iter().copied().fold(0.0f64, f64::max);
    eprintln!(
        "  [PREFILL]   {label} x{}: {:.1}ms total (avg {:.2}ms, min {:.2}ms, max {:.2}ms)",
        times_ms.len(),
        total_ms,
        total_ms / times_ms.len() as f64,
        min_ms,
        max_ms
    );
}
