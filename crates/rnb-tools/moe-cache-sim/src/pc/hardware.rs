#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwareMeta {
    pub name: String,
    pub h2d_bandwidth_gbps: f64,
    pub h2d_latency_us: f64,
    pub gpu_expert_compute_us: f64,
    pub cpu_expert_compute_us: f64,
}

impl HardwareMeta {
    pub fn copy_ms_for_bytes(&self, bytes: f64, misses: u64) -> f64 {
        if bytes <= 0.0 || self.h2d_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        let bandwidth_bytes_per_ms = self.h2d_bandwidth_gbps * 1_000_000.0;
        let transfer_ms = bytes / bandwidth_bytes_per_ms;
        let latency_ms = misses as f64 * self.h2d_latency_us / 1000.0;
        transfer_ms + latency_ms
    }

    pub fn break_even_hit_rate(&self, h2d_copy_us: f64) -> Option<f64> {
        if h2d_copy_us <= 0.0 || self.cpu_expert_compute_us <= self.gpu_expert_compute_us {
            return None;
        }
        let miss_rate = (self.cpu_expert_compute_us - self.gpu_expert_compute_us) / h2d_copy_us;
        Some((1.0 - miss_rate).clamp(0.0, 1.0))
    }
}
