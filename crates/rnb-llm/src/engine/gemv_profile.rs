//! GEMV profiling helpers (enable via `RNB_GEMV_PROFILE=1`).

use rnb_loader::GGMLType;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Default, Clone)]
struct GemvProfileStat {
    calls: u64,
    micros: u128,
}

static GEMV_PROFILE: OnceLock<Mutex<HashMap<String, GemvProfileStat>>> = OnceLock::new();

fn gemv_profile_enabled() -> bool {
    super::policy::gemv_profile_enabled()
}

fn record_gemv_profile(
    method: &'static str,
    ggml_type: GGMLType,
    seq_len: usize,
    rows: usize,
    cols: usize,
    elapsed: std::time::Duration,
) {
    if !gemv_profile_enabled() {
        return;
    }
    let bucket = if seq_len > 1 { "prefill" } else { "decode" };
    let key = format!("{method}:{bucket}:{ggml_type:?}:{rows}x{cols}");
    let lock = GEMV_PROFILE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = lock.lock().expect("gemv profile lock poisoned");
    let stat = guard.entry(key).or_default();
    stat.calls += 1;
    stat.micros += elapsed.as_micros();
}

pub fn reset_gemv_profile() {
    if let Some(lock) = GEMV_PROFILE.get() {
        lock.lock().expect("gemv profile lock poisoned").clear();
    }
}

pub fn gemv_profile_report() -> Option<String> {
    let lock = GEMV_PROFILE.get()?;
    let guard = lock.lock().ok()?;
    if guard.is_empty() {
        return None;
    }
    let mut entries = guard
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| b.1.micros.cmp(&a.1.micros));
    let mut out = String::from("=== GEMV profile ===\n");
    for (key, stat) in entries {
        let avg = stat.micros as f64 / stat.calls as f64;
        out.push_str(&format!(
            "{key} calls={} total_ms={:.1} avg_us={:.1}\n",
            stat.calls,
            stat.micros as f64 / 1000.0,
            avg
        ));
    }
    Some(out)
}

pub(super) struct GemvProfileGuard {
    enabled: bool,
    method: &'static str,
    ggml_type: GGMLType,
    seq_len: usize,
    rows: usize,
    cols: usize,
    start: Instant,
}

impl GemvProfileGuard {
    pub(super) fn new(
        method: &'static str,
        ggml_type: GGMLType,
        seq_len: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            enabled: gemv_profile_enabled(),
            method,
            ggml_type,
            seq_len,
            rows,
            cols,
            start: Instant::now(),
        }
    }
}

impl Drop for GemvProfileGuard {
    fn drop(&mut self) {
        if self.enabled {
            record_gemv_profile(
                self.method,
                self.ggml_type,
                self.seq_len,
                self.rows,
                self.cols,
                self.start.elapsed(),
            );
        }
    }
}
