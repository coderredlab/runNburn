use super::layer_weights::LayerType;
use super::memory_runtime::mlock;
use super::Engine;

impl Engine {
    /// pm117: output head weight 는 매 decode/verify 라운드 필수 접근인데 expert
    /// pread 가 page cache 를 공유하면서 간헐 evict → argmax 스파이크 (5ms 평시,
    /// 40~563ms 스파이크 실측). vocab×hidden 은 모델 대비 소량이라 `mlock(2)` 로
    /// 고정한다. `RNB_MLOCK_OUTPUT=0` opt-out. 실패(RLIMIT 등)는 로그만 하고
    /// 비치명 — 그 경우 기존 page cache 동작 그대로.
    pub(super) fn apply_output_weight_mlock(&self) {
        if crate::engine::policy::env_string("RNB_MLOCK_OUTPUT").as_deref() == Some("0") {
            return;
        }
        let Some(weights) = self.weights.as_ref() else {
            return;
        };
        let Some(raw) = weights.output.data.as_bytes() else {
            return;
        };
        match mlock::mlock_region(raw.as_ptr(), raw.len()) {
            Ok(bytes) => {
                eprintln!(
                    "[INFO] output weight mlock: {:.1} MiB pinned",
                    bytes as f64 / (1024.0 * 1024.0)
                );
            }
            Err(err) => {
                eprintln!("[WARN] output weight mlock failed (non-fatal): {err}");
            }
        }
    }

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
                let pin_ids = mlock::expert_pin_ids(pop, layer_idx, n, max_slots, false);
                for &eid in &pin_ids {
                    report.lock_expert_pair_by_id(layer_idx, eid, gu_all, per_gu, dn_all, per_dn);
                }
            }
        }

        report.log_summary(&cfg);
    }
}
