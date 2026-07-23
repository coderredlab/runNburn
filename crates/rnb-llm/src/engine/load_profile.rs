use std::time::{Duration, Instant};

pub(in crate::engine) struct LoadProfile {
    enabled: bool,
    stages: Vec<LoadProfileStage>,
}

struct LoadProfileStage {
    name: &'static str,
    elapsed: Duration,
}

impl LoadProfile {
    pub(in crate::engine) fn from_env() -> Self {
        Self {
            enabled: load_profile_enabled(),
            stages: Vec::new(),
        }
    }

    pub(in crate::engine) fn begin(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    pub(in crate::engine) fn record_since(&mut self, name: &'static str, start: Option<Instant>) {
        if let Some(start) = start.filter(|_| self.enabled) {
            self.record(name, start.elapsed());
        }
    }

    pub(in crate::engine) fn finish_and_emit(self, total_start: Option<Instant>) {
        let Some(total_start) = total_start.filter(|_| self.enabled) else {
            return;
        };
        if let Some(report) = self.report(total_start.elapsed()) {
            eprintln!("{report}");
        }
    }

    fn record(&mut self, name: &'static str, elapsed: Duration) {
        self.stages.push(LoadProfileStage { name, elapsed });
    }

    fn report(&self, total: Duration) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let mut report = format!("[load-profile] total_ms={:.3}", duration_ms(total));
        for stage in &self.stages {
            report.push(' ');
            report.push_str(stage.name);
            report.push_str("_ms=");
            report.push_str(&format!("{:.3}", duration_ms(stage.elapsed)));
        }
        Some(report)
    }

    #[cfg(test)]
    pub(in crate::engine) fn enabled_for_test() -> Self {
        Self {
            enabled: true,
            stages: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(in crate::engine) fn record_for_test(&mut self, name: &'static str, elapsed: Duration) {
        self.record(name, elapsed);
    }

    #[cfg(test)]
    pub(in crate::engine) fn finish_report_for_test(&self, total: Duration) -> Option<String> {
        self.report(total)
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn load_profile_enabled() -> bool {
    crate::engine::policy::env_string("RNB_LOAD_PROFILE").is_some_and(|value| {
        !matches!(
            value.as_str(),
            "0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO"
        )
    })
}

#[cfg(test)]
pub(in crate::engine) fn load_profile_enabled_for_test() -> bool {
    load_profile_enabled()
}
