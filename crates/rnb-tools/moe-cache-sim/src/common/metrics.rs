#[derive(Debug, Clone, serde::Serialize)]
pub struct SimulationMetrics {
    pub events: u64,
    pub hits: u64,
    pub misses: u64,
    pub miss_bytes: u64,
    pub resident_entries: usize,
}

impl SimulationMetrics {
    pub fn hit_rate(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.hits as f64 / self.events as f64
        }
    }

    pub fn miss_rate(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.misses as f64 / self.events as f64
        }
    }

    pub fn miss_bytes_per_token(&self, steps: usize) -> f64 {
        if steps == 0 {
            0.0
        } else {
            self.miss_bytes as f64 / steps as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_compute_rates() {
        let metrics = SimulationMetrics {
            events: 10,
            hits: 7,
            misses: 3,
            miss_bytes: 900,
            resident_entries: 4,
        };

        assert_eq!(metrics.hit_rate(), 0.7);
        assert_eq!(metrics.miss_rate(), 0.3);
        assert_eq!(metrics.miss_bytes_per_token(3), 300.0);
    }
}
