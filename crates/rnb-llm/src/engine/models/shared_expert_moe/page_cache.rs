use super::weights::SharedExpertMoELayerWeights;
use crate::engine::layer_weights::{LayerType, ModelWeights};
use crate::engine::memory_runtime::sparse_expert_cache::{
    SparseExpertCacheBudget, SparseExpertCacheKey,
};
use rnb_core::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct MappedProjection(Tensor);

impl MappedProjection {
    fn from_weight(direct: &Tensor) -> Option<Self> {
        direct.as_bytes()?;
        Some(Self(direct.clone()))
    }

    fn byte_len(&self) -> usize {
        self.0.as_bytes().map_or(0, <[u8]>::len)
    }

    fn reclaim(&self, relative_byte_offset: usize, byte_len: usize) -> std::io::Result<()> {
        self.0
            .reclaim_file_mmap_range(relative_byte_offset, byte_len)
    }
}

#[derive(Clone)]
struct SparseExpertMappedLayer {
    gate: MappedProjection,
    up: MappedProjection,
    down: MappedProjection,
    per_gate: usize,
    per_up: usize,
    per_down: usize,
    n_expert: usize,
}

impl SparseExpertMappedLayer {
    fn from_weights(weights: &SharedExpertMoELayerWeights) -> Option<Self> {
        if weights.n_expert == 0 {
            return None;
        }
        let gate = MappedProjection::from_weight(&weights.gate_exps)?;
        let up = MappedProjection::from_weight(&weights.up_exps)?;
        let down = MappedProjection::from_weight(&weights.down_exps)?;
        let per_gate = gate.byte_len() / weights.n_expert;
        let per_up = up.byte_len() / weights.n_expert;
        let per_down = down.byte_len() / weights.n_expert;
        Some(Self {
            gate,
            up,
            down,
            per_gate,
            per_up,
            per_down,
            n_expert: weights.n_expert,
        })
    }

    fn bytes_per_expert(&self) -> u64 {
        self.per_gate
            .saturating_add(self.per_up)
            .saturating_add(self.per_down) as u64
    }

    fn reclaim(&self, expert_index: usize) -> std::io::Result<()> {
        if expert_index >= self.n_expert {
            return Ok(());
        }
        self.gate
            .reclaim(expert_index * self.per_gate, self.per_gate)?;
        self.up.reclaim(expert_index * self.per_up, self.per_up)?;
        self.down
            .reclaim(expert_index * self.per_down, self.per_down)
    }
}

pub(in crate::engine) struct SparseExpertPageCache {
    policy: Mutex<SparseExpertCacheBudget>,
    layers: Vec<Option<SparseExpertMappedLayer>>,
    reclaim_warning_emitted: AtomicBool,
}

impl SparseExpertPageCache {
    pub fn touch(&self, layer_index: usize, expert_indices: &[usize]) {
        let Some(Some(layer)) = self.layers.get(layer_index) else {
            return;
        };
        let bytes = layer.bytes_per_expert();
        let mut evicted = Vec::new();
        {
            let mut policy = self.policy.lock().expect("sparse expert page-cache lock");
            for &expert_index in expert_indices {
                evicted.extend(
                    policy.touch(SparseExpertCacheKey::new(layer_index, expert_index), bytes),
                );
            }
        }
        for key in evicted {
            if let Some(Some(layer)) = self.layers.get(key.layer_index()) {
                if let Err(error) = layer.reclaim(key.expert_index()) {
                    if !self.reclaim_warning_emitted.swap(true, Ordering::Relaxed) {
                        eprintln!("[WARN] sparse expert page reclaim failed: {error}");
                    }
                }
            }
        }
    }
}

/// Wires a byte-budgeted LRU over file-backed sparse-expert mappings.
/// Returns the expert-cache byte budget when at least one reclaimable GGUF layer was found.
pub(in crate::engine) fn wire_sparse_expert_page_cache(
    weights: &mut ModelWeights,
    host_budget_bytes: u64,
    gguf_file_bytes: u64,
) -> Option<u64> {
    if !cfg!(unix) {
        return None;
    }
    let layers: Vec<Option<SparseExpertMappedLayer>> = weights
        .layers
        .iter()
        .map(|layer| match layer {
            LayerType::Attention(weights) => weights
                .shared_expert_moe
                .as_ref()
                .and_then(SparseExpertMappedLayer::from_weights),
            LayerType::GatedDeltaNet(weights) => weights
                .shared_expert_moe
                .as_ref()
                .and_then(SparseExpertMappedLayer::from_weights),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
        })
        .collect();
    let reclaimable_sparse_bytes = layers
        .iter()
        .flatten()
        .map(|layer| {
            layer
                .bytes_per_expert()
                .saturating_mul(layer.n_expert as u64)
        })
        .sum::<u64>();
    if reclaimable_sparse_bytes == 0 {
        return None;
    }

    let non_sparse_bytes = gguf_file_bytes.saturating_sub(reclaimable_sparse_bytes);
    let expert_budget_bytes = host_budget_bytes.saturating_sub(non_sparse_bytes);
    let cache = Arc::new(SparseExpertPageCache {
        policy: Mutex::new(SparseExpertCacheBudget::new(expert_budget_bytes)),
        layers,
        reclaim_warning_emitted: AtomicBool::new(false),
    });

    for layer in &mut weights.layers {
        let moe = match layer {
            LayerType::Attention(weights) => weights.shared_expert_moe.as_mut(),
            LayerType::GatedDeltaNet(weights) => weights.shared_expert_moe.as_mut(),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
        };
        if let Some(moe) = moe {
            moe.sparse_page_cache = Some(Arc::clone(&cache));
        }
    }
    Some(expert_budget_bytes)
}
