use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub name: String,
    pub num_layers: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub default_expert_bytes: u64,
    #[serde(default)]
    pub default_gate_up_bytes: u64,
    #[serde(default)]
    pub default_down_bytes: u64,
    #[serde(default)]
    pub dense_resident_bytes: u64,
    #[serde(default)]
    pub kv_cache_bytes: u64,
    #[serde(default)]
    pub entries: Vec<ExpertBytes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertBytes {
    pub layer: usize,
    pub expert_id: usize,
    pub bytes: u64,
}

impl ModelMeta {
    pub fn expert_bytes(&self, layer: usize, expert_id: usize) -> u64 {
        self.entries
            .iter()
            .find(|entry| entry.layer == layer && entry.expert_id == expert_id)
            .map(|entry| entry.bytes)
            .unwrap_or(self.default_expert_bytes)
    }

    pub fn expert_gate_up_bytes(&self, _layer: usize, _expert_id: usize) -> u64 {
        if self.default_gate_up_bytes > 0 {
            self.default_gate_up_bytes
        } else {
            self.default_expert_bytes
        }
    }

    pub fn expert_down_bytes(&self, _layer: usize, _expert_id: usize) -> u64 {
        self.default_down_bytes
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.dense_resident_bytes
            .saturating_add(self.kv_cache_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_meta_uses_explicit_entry_before_default() {
        let meta = ModelMeta {
            name: "demo".to_string(),
            num_layers: 2,
            num_experts: 4,
            experts_per_token: 2,
            default_expert_bytes: 100,
            default_gate_up_bytes: 60,
            default_down_bytes: 40,
            dense_resident_bytes: 10,
            kv_cache_bytes: 20,
            entries: vec![ExpertBytes {
                layer: 1,
                expert_id: 3,
                bytes: 777,
            }],
        };

        assert_eq!(meta.expert_bytes(1, 3), 777);
        assert_eq!(meta.expert_bytes(1, 2), 100);
        assert_eq!(meta.expert_gate_up_bytes(1, 2), 60);
        assert_eq!(meta.expert_down_bytes(1, 2), 40);
        assert_eq!(meta.reserved_bytes(), 30);
    }
}
