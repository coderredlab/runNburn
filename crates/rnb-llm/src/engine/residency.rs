use super::layer_weights::LayerType;
use super::memory_runtime::mlock;
use super::Engine;

impl Engine {
    /// in4 cleanup: legacy `cold_reader` direct field is gone. The trait-based
    /// residency view (`MoeExpertResidencyView`) does not expose pread stats
    /// because not every implementation backs onto a single `ColdReader` —
    /// the API surface returns no stats. Kept as a no-op so existing FFI
    /// callers compile.
    pub fn cold_reader_stats(&self) -> Option<(u64, u64, u64)> {
        None
    }

    pub fn cold_reader_reset_stats(&self) {}

    /// Session 67 axis P: pin router (and optionally per-layer top-N hottest
    /// experts) into RAM via `mlock(2)`. See `super::memory_runtime::mlock` for env schema.
    /// Every failure is logged and non-fatal — on Android the 64 MB
    /// `RLIMIT_MEMLOCK` makes most expert pins return `ENOMEM`, which is
    /// expected and handled.
    pub(super) fn apply_axis_p_mlock(&self) {
        let cfg = mlock::AxisPConfig::from_env();
        if !cfg.is_active() {
            return;
        }
        let Some(weights) = self.weights.as_ref() else {
            return;
        };

        let popularity = mlock::load_popularity_for_config(&cfg);
        let mut report = mlock::AxisPMlockReport::default();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let LayerType::Attention(w) = layer else {
                continue;
            };
            let Some(moe) = w.moe.as_ref() else {
                continue;
            };

            if cfg.mlock_router {
                if let Some(rf) = moe.router_f32() {
                    let bytes = rf.len() * std::mem::size_of::<f32>();
                    report.lock_router(layer_idx, rf.as_ptr() as *const u8, bytes);
                }
            }

            if let (Some(pop), n) = (popularity.as_ref(), cfg.mlock_top_n) {
                if n == 0 {
                    continue;
                }
                let Some(gu_all) = moe.gate_up_bytes() else {
                    continue;
                };
                let Some(dn_all) = moe.down_bytes() else {
                    continue;
                };
                let per_gu = crate::engine::moe::per_expert_gate_up_bytes(moe.n_embd, moe.n_ff);
                let per_dn =
                    crate::engine::moe::per_expert_down_bytes(moe.n_embd, moe.n_ff, moe.down_quant);
                let max_slots = gu_all.len() / per_gu.max(1);
                let pin_ids = mlock::expert_pin_ids(
                    pop,
                    layer_idx,
                    n,
                    max_slots,
                    moe.gate_up_rnb_name.is_some(),
                );
                for &eid in &pin_ids {
                    report.lock_expert_pair_by_id(layer_idx, eid, gu_all, per_gu, dn_all, per_dn);
                }
            }
        }

        report.log_summary(&cfg);
    }
}
