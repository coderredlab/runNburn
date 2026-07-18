//! MoE expert residency view: thin abstraction over the byte source for a
//! single MoE weight tensor's experts.
//!
//! Implementations live in concrete weight stores (`rnb-loader::PackedModel`
//! for v3 sidecar mmap; `ColdReader` / `HotPool` for the legacy hot/cold
//! split in `rnb-memory`). Returned slices follow the same per-expert
//! layout: `pack_q4k_compact` output, repeated `n_experts()` times.
//!
//! `expert_bytes(rank)` returns a `Cow<'_, [u8]>` so mmap-backed sources can
//! return `Cow::Borrowed` (zero-copy) while disk-pread sources return
//! `Cow::Owned(Vec<u8>)`. Callers that need a `&[u8]` simply deref the Cow.

use std::borrow::Cow;
use std::sync::Arc;

/// Abstract view of a single MoE weight tensor's expert byte slices, indexed
/// by **rank** in popularity order. Engine forward paths translate
/// `selected_idx -> popularity_order[rank] = original_id`, then call
/// `expert_bytes(rank)` for the GEMV input row block.
pub trait MoeExpertResidencyView: Send + Sync {
    /// Total number of experts represented by this tensor.
    fn n_experts(&self) -> usize;

    /// Number of experts whose bytes are hot-resident (`MADV_WILLNEED`-backed
    /// or RAM-resident). `0..hot_count` are hot ranks.
    fn hot_count(&self) -> usize;

    /// `rank -> original_expert_id` mapping. `None` means identity
    /// (rank == original_id), used by the simple "all hot, no popularity" path.
    fn popularity_order(&self) -> Option<&[u32]>;

    /// Per-expert byte stride (bytes per expert payload). All experts in a
    /// tensor share the same byte length.
    fn per_expert_bytes(&self) -> usize;

    /// Bytes for the expert at the given rank. `Cow::Borrowed` for mmap or
    /// in-RAM sources; `Cow::Owned(Vec<u8>)` for disk-pread sources.
    fn expert_bytes(&self, rank: usize) -> Cow<'_, [u8]>;
}

/// Source for a single hot rank's bytes. Backs the `0..hot_count` range of
/// a `ComposedResidency`.
pub trait HotByteSource: Send + Sync {
    fn hot(&self, rank: usize, per_expert: usize) -> Cow<'_, [u8]>;
}

/// Source for a single cold rank's bytes (`hot_count..n_experts`).
pub trait ColdByteSource: Send + Sync {
    fn cold(&self, cold_rank: usize, per_expert: usize) -> Cow<'_, [u8]>;
}

/// Composes a `HotByteSource` + optional `ColdByteSource` into a single
/// residency view. Engine `MoeLayerView::forward` uses one composed view
/// per MoE tensor instead of branching across the legacy 5-way
/// `ExpertResidencySource` matrix.
pub struct ComposedResidency {
    pub hot_source: Arc<dyn HotByteSource>,
    pub cold_source: Option<Arc<dyn ColdByteSource>>,
    pub n_experts: usize,
    pub hot_count: usize,
    pub per_expert_bytes: usize,
    pub popularity_order: Option<Vec<u32>>,
}

impl MoeExpertResidencyView for ComposedResidency {
    fn n_experts(&self) -> usize {
        self.n_experts
    }
    fn hot_count(&self) -> usize {
        self.hot_count
    }
    fn popularity_order(&self) -> Option<&[u32]> {
        self.popularity_order.as_deref()
    }
    fn per_expert_bytes(&self) -> usize {
        self.per_expert_bytes
    }
    fn expert_bytes(&self, rank: usize) -> Cow<'_, [u8]> {
        assert!(
            rank < self.n_experts,
            "rank {rank} out of bounds for n_experts {}",
            self.n_experts
        );
        if rank < self.hot_count {
            self.hot_source.hot(rank, self.per_expert_bytes)
        } else {
            let cs = self
                .cold_source
                .as_ref()
                .expect("rank >= hot_count requires a cold source");
            cs.cold(rank - self.hot_count, self.per_expert_bytes)
        }
    }
}

/// Hot source backed by a contiguous `&[u8]` (mmap or RAM pool), each rank
/// occupying `per_expert` bytes at offset `rank * per_expert`.
pub struct ContiguousHot {
    bytes: ContiguousBacking,
}

enum ContiguousBacking {
    Mmap(Arc<dyn AsRef<[u8]> + Send + Sync>),
    Owned(Vec<u8>),
}

impl ContiguousHot {
    pub fn from_owned(bytes: Vec<u8>) -> Self {
        Self {
            bytes: ContiguousBacking::Owned(bytes),
        }
    }
    pub fn from_arc(bytes: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        Self {
            bytes: ContiguousBacking::Mmap(bytes),
        }
    }
    fn slice(&self) -> &[u8] {
        match &self.bytes {
            ContiguousBacking::Mmap(a) => (**a).as_ref(),
            ContiguousBacking::Owned(v) => v.as_slice(),
        }
    }
}

impl HotByteSource for ContiguousHot {
    fn hot(&self, rank: usize, per_expert: usize) -> Cow<'_, [u8]> {
        let s = self.slice();
        Cow::Borrowed(&s[rank * per_expert..(rank + 1) * per_expert])
    }
}

/// Cold source backed by a contiguous `&[u8]` (mmap), each cold rank at
/// offset `cold_rank * per_expert`.
pub struct ContiguousCold {
    bytes: ContiguousBacking,
}

impl ContiguousCold {
    pub fn from_arc(bytes: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        Self {
            bytes: ContiguousBacking::Mmap(bytes),
        }
    }
    fn slice(&self) -> &[u8] {
        match &self.bytes {
            ContiguousBacking::Mmap(a) => (**a).as_ref(),
            ContiguousBacking::Owned(v) => v.as_slice(),
        }
    }
}

impl ColdByteSource for ContiguousCold {
    fn cold(&self, cold_rank: usize, per_expert: usize) -> Cow<'_, [u8]> {
        let s = self.slice();
        Cow::Borrowed(&s[cold_rank * per_expert..(cold_rank + 1) * per_expert])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestView {
        hot: Vec<u8>,
        cold: Vec<u8>,
        hot_count: usize,
        n_experts: usize,
        per_expert: usize,
        order: Option<Vec<u32>>,
    }

    impl MoeExpertResidencyView for TestView {
        fn n_experts(&self) -> usize {
            self.n_experts
        }
        fn hot_count(&self) -> usize {
            self.hot_count
        }
        fn popularity_order(&self) -> Option<&[u32]> {
            self.order.as_deref()
        }
        fn per_expert_bytes(&self) -> usize {
            self.per_expert
        }
        fn expert_bytes(&self, rank: usize) -> Cow<'_, [u8]> {
            assert!(rank < self.n_experts);
            if rank < self.hot_count {
                Cow::Borrowed(&self.hot[rank * self.per_expert..(rank + 1) * self.per_expert])
            } else {
                let cold_rank = rank - self.hot_count;
                Cow::Borrowed(
                    &self.cold[cold_rank * self.per_expert..(cold_rank + 1) * self.per_expert],
                )
            }
        }
    }

    #[test]
    fn view_routes_hot_and_cold_ranks_to_correct_byte_blocks() {
        let view = TestView {
            hot: vec![1, 1, 1, 1, 2, 2, 2, 2],
            cold: vec![3, 3, 3, 3, 4, 4, 4, 4],
            hot_count: 2,
            n_experts: 4,
            per_expert: 4,
            order: Some(vec![10, 11, 12, 13]),
        };
        assert_eq!(view.n_experts(), 4);
        assert_eq!(view.hot_count(), 2);
        assert_eq!(view.per_expert_bytes(), 4);
        assert_eq!(view.popularity_order(), Some(&[10u32, 11, 12, 13][..]));
        assert_eq!(&*view.expert_bytes(0), &[1, 1, 1, 1]);
        assert_eq!(&*view.expert_bytes(1), &[2, 2, 2, 2]);
        assert_eq!(&*view.expert_bytes(2), &[3, 3, 3, 3]);
        assert_eq!(&*view.expert_bytes(3), &[4, 4, 4, 4]);
    }

    #[test]
    fn composed_residency_routes_through_hot_and_cold_sources() {
        let hot = Arc::new(ContiguousHot::from_owned(vec![
            10, 10, 10, 10, 20, 20, 20, 20,
        ]));
        // Cold source needs Mmap-style backing (Arc<dyn AsRef<[u8]>>).
        let cold_bytes: Arc<dyn AsRef<[u8]> + Send + Sync> =
            Arc::new(vec![30u8, 30, 30, 30, 40, 40, 40, 40]);
        let cold = Arc::new(ContiguousCold::from_arc(cold_bytes));
        let composed = ComposedResidency {
            hot_source: hot,
            cold_source: Some(cold),
            n_experts: 4,
            hot_count: 2,
            per_expert_bytes: 4,
            popularity_order: None,
        };
        assert_eq!(&*composed.expert_bytes(0), &[10, 10, 10, 10]);
        assert_eq!(&*composed.expert_bytes(1), &[20, 20, 20, 20]);
        assert_eq!(&*composed.expert_bytes(2), &[30, 30, 30, 30]);
        assert_eq!(&*composed.expert_bytes(3), &[40, 40, 40, 40]);
    }

    #[test]
    fn composed_residency_no_cold_when_hot_count_equals_n_experts() {
        let hot = Arc::new(ContiguousHot::from_owned(vec![5u8; 8]));
        let composed = ComposedResidency {
            hot_source: hot,
            cold_source: None,
            n_experts: 2,
            hot_count: 2,
            per_expert_bytes: 4,
            popularity_order: None,
        };
        assert_eq!(&*composed.expert_bytes(0), &[5; 4]);
        assert_eq!(&*composed.expert_bytes(1), &[5; 4]);
    }

    #[test]
    #[should_panic(expected = "rank >= hot_count requires a cold source")]
    fn composed_residency_panics_when_cold_rank_has_no_cold_source() {
        let hot = Arc::new(ContiguousHot::from_owned(vec![0u8; 4]));
        let composed = ComposedResidency {
            hot_source: hot,
            cold_source: None,
            n_experts: 4,
            hot_count: 1,
            per_expert_bytes: 4,
            popularity_order: None,
        };
        let _ = composed.expert_bytes(2);
    }
}
