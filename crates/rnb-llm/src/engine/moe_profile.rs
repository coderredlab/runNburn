use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Default, Clone)]
struct MoeProfileStat {
    calls: u64,
    micros: u128,
    max_micros: u128,
}

#[derive(Default, Clone)]
struct MoeProfileCounts {
    high: u64,
    low: u64,
    skip: u64,
}

static MOE_PROFILE: OnceLock<Mutex<HashMap<String, MoeProfileStat>>> = OnceLock::new();
static MOE_COUNTS: OnceLock<Mutex<HashMap<String, MoeProfileCounts>>> = OnceLock::new();

#[cfg(test)]
static TEST_ENABLED: OnceLock<Mutex<Option<bool>>> = OnceLock::new();

#[cfg(test)]
pub(super) fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
fn test_enabled_override() -> Option<bool> {
    TEST_ENABLED
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("moe profile test gate lock poisoned")
        .as_ref()
        .copied()
}

#[cfg(test)]
pub(super) fn set_moe_profile_enabled_for_tests(enabled: Option<bool>) {
    *TEST_ENABLED
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("moe profile test gate lock poisoned") = enabled;
}

#[inline]
pub(super) fn is_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = test_enabled_override() {
        return enabled;
    }

    super::policy::moe_profile_enabled()
}

#[inline]
pub(super) fn record_moe_profile(key: &'static str, elapsed: Duration) {
    if !is_enabled() {
        return;
    }
    record_moe_profile_key(key.to_string(), elapsed);
}

#[inline]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn record_moe_profile_by_layer(
    prefix: &'static str,
    layer_idx: Option<usize>,
    stage: &'static str,
    elapsed: Duration,
) {
    if !is_enabled() || !super::policy::moe_profile_by_layer_enabled() {
        return;
    }
    let Some(layer_idx) = layer_idx else {
        return;
    };
    record_moe_profile_key(format!("{prefix}:layer{layer_idx:02}:{stage}"), elapsed);
}

#[inline]
fn record_moe_profile_key(key: String, elapsed: Duration) {
    let lock = MOE_PROFILE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = lock.lock().expect("moe profile lock poisoned");
    let stat = guard.entry(key).or_default();
    let micros = elapsed.as_micros();
    stat.calls += 1;
    stat.micros += micros;
    stat.max_micros = stat.max_micros.max(micros);
}

#[inline]
pub(super) fn record_moe_counts(key: &'static str, high: u64, low: u64, skip: u64) {
    if !is_enabled() {
        return;
    }
    let lock = MOE_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = lock.lock().expect("moe count lock poisoned");
    let counts = guard.entry(key.to_string()).or_default();
    counts.high += high;
    counts.low += low;
    counts.skip += skip;
}

pub fn reset_moe_profile() {
    if let Some(lock) = MOE_PROFILE.get() {
        lock.lock().expect("moe profile lock poisoned").clear();
    }
    if let Some(lock) = MOE_COUNTS.get() {
        lock.lock().expect("moe count lock poisoned").clear();
    }
}

pub fn moe_profile_report() -> Option<String> {
    let profile_guard = MOE_PROFILE
        .get()
        .map(|lock| lock.lock().expect("moe profile lock poisoned"));
    let count_guard = MOE_COUNTS
        .get()
        .map(|lock| lock.lock().expect("moe count lock poisoned"));

    if profile_guard.as_ref().is_none_or(|g| g.is_empty())
        && count_guard.as_ref().is_none_or(|g| g.is_empty())
    {
        return None;
    }

    let mut out = String::from("=== MoE profile ===\n");

    if let Some(profile_guard) = profile_guard {
        let mut entries = profile_guard
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| b.1.micros.cmp(&a.1.micros));
        for (key, stat) in entries {
            let avg = stat.micros as f64 / stat.calls as f64;
            out.push_str(&format!(
                "{key} calls={} total_ms={:.1} avg_us={:.1} max_us={}\n",
                stat.calls,
                stat.micros as f64 / 1000.0,
                avg,
                stat.max_micros,
            ));
        }
    }

    if let Some(count_guard) = count_guard {
        let mut count_entries = count_guard.iter().collect::<Vec<_>>();
        count_entries.sort_by(|a, b| a.0.cmp(b.0));
        for (key, counts) in count_entries {
            out.push_str(&format!(
                "{key} counts high={} low={} skip={}\n",
                counts.high, counts.low, counts.skip,
            ));
        }
    }

    Some(out)
}

#[allow(dead_code)]
pub(super) struct MoeProfileGuard {
    enabled: bool,
    key: &'static str,
    start: Instant,
}

#[allow(dead_code)]
impl MoeProfileGuard {
    pub(super) fn new(key: &'static str) -> Self {
        Self {
            enabled: is_enabled(),
            key,
            start: Instant::now(),
        }
    }
}

impl Drop for MoeProfileGuard {
    fn drop(&mut self) {
        if self.enabled {
            record_moe_profile(self.key, self.start.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_profile_report_is_none_when_disabled() {
        let _guard = test_lock().lock().expect("moe profile test lock poisoned");
        set_moe_profile_enabled_for_tests(Some(false));
        reset_moe_profile();
        record_moe_profile("qwen35moe:decode:router", Duration::from_micros(123));
        assert!(moe_profile_report().is_none());
        set_moe_profile_enabled_for_tests(None);
    }

    #[test]
    fn moe_profile_report_contains_bucket_and_counts() {
        let _guard = test_lock().lock().expect("moe profile test lock poisoned");
        set_moe_profile_enabled_for_tests(Some(true));
        reset_moe_profile();
        record_moe_profile("qwen35moe:decode:router", Duration::from_micros(1200));
        record_moe_profile("qwen35moe:decode:router", Duration::from_micros(800));
        record_moe_counts("qwen35moe:decode", 3, 2, 1);

        let report = moe_profile_report().expect("report should exist");
        assert!(report.contains("=== MoE profile ==="));
        assert!(report.contains("qwen35moe:decode:router"));
        assert!(report.contains("calls=2"));
        assert!(report.contains("counts high=3 low=2 skip=1"));

        reset_moe_profile();
        set_moe_profile_enabled_for_tests(None);
    }

    #[test]
    fn moe_profile_guard_records_on_drop() {
        let _guard = test_lock().lock().expect("moe profile test lock poisoned");
        set_moe_profile_enabled_for_tests(Some(true));
        reset_moe_profile();

        {
            let _guard = MoeProfileGuard::new("qwen35moe:decode:router");
        }

        let report = moe_profile_report().expect("report should exist");
        assert!(report.contains("qwen35moe:decode:router calls=1"));

        reset_moe_profile();
        set_moe_profile_enabled_for_tests(None);
    }

    #[test]
    fn moe_profile_by_layer_requires_layer_env_and_records_dynamic_key() {
        let _guard = test_lock().lock().expect("moe profile test lock poisoned");
        set_moe_profile_enabled_for_tests(Some(true));
        unsafe {
            std::env::remove_var("RNB_MOE_PROFILE_BY_LAYER");
        }
        reset_moe_profile();

        record_moe_profile_by_layer(
            "qwen35moe:moe_section",
            Some(3),
            "gate_up_compute",
            Duration::from_micros(1200),
        );
        assert!(moe_profile_report().is_none());

        unsafe {
            std::env::set_var("RNB_MOE_PROFILE_BY_LAYER", "1");
        }
        record_moe_profile_by_layer(
            "qwen35moe:moe_section",
            Some(3),
            "gate_up_compute",
            Duration::from_micros(1200),
        );
        record_moe_profile_by_layer(
            "qwen35moe:moe_section",
            Some(3),
            "gate_up_compute",
            Duration::from_micros(800),
        );
        record_moe_profile_by_layer(
            "qwen35moe:moe_section",
            None,
            "gate_up_compute",
            Duration::from_micros(999),
        );

        let report = moe_profile_report().expect("report should exist");
        assert!(report.contains("qwen35moe:moe_section:layer03:gate_up_compute calls=2"));
        assert!(report.contains("total_ms=2.0"));
        assert!(!report.contains("layerNone"));

        unsafe {
            std::env::remove_var("RNB_MOE_PROFILE_BY_LAYER");
        }
        reset_moe_profile();
        set_moe_profile_enabled_for_tests(None);
    }
}
