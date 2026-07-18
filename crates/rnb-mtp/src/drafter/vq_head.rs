//! Drafter VQ logit head.
//!
//! 두 단계로 분리:
//!
//! 1. [`vq_head_forward`] — `cluster_logits = x_norm @ centroids.T`. 매 forward
//!    호출마다 무조건 돌린다. shape = `[n_centroids]` (Gemma 4 assistant 는
//!    2048). 다음 단계 (top-K vocab) 의 input.
//!
//! 2. [`vocab_logits_in_top_k_clusters`] — cluster_logits 의 top-K cluster 안에
//!    속한 token 만 vocab logit 을 실제로 계산한다. 나머지 token 은
//!    `f32::NEG_INFINITY`. cluster ↔ token 매핑은 [`ClusterTokenTable`] 이
//!    소유.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-cross-attention-design.md`
//! §"VQ logit head".

use super::dequant::dequant_to_f32;
use super::types::Drafter;

/// VQ head 의 결과.
///
/// - `cluster_logits.len() == drafter.n_centroids` — 항상 채워짐.
/// - `vocab_logits` — caller 가 [`vocab_logits_in_top_k_clusters`] 를 호출했을
///   때만 채워진다. `None` 이면 cluster level 까지만 평가된 상태.
#[derive(Debug, Clone)]
pub struct VqHeadOutput {
    pub cluster_logits: Vec<f32>,
    pub vocab_logits: Option<Vec<f32>>,
}

/// `cluster_logits = x_norm @ centroids.T`.
///
/// `x_norm` 은 drafter 의 `output_norm` 까지 적용된 상태여야 한다.
/// `cross_attention::drafter_cross_attention_forward` 의 마지막 step 이
/// 이 함수를 호출한다.
pub fn vq_head_forward(drafter: &Drafter, x_norm: &[f32]) -> VqHeadOutput {
    assert_eq!(
        x_norm.len(),
        drafter.hidden,
        "vq_head_forward: x_norm len {} != drafter.hidden {}",
        x_norm.len(),
        drafter.hidden
    );

    let n_centroids = drafter.n_centroids as usize;
    assert_eq!(
        drafter.centroids.shape.len(),
        2,
        "centroids shape rank != 2: {:?}",
        drafter.centroids.shape
    );
    assert_eq!(
        drafter.centroids.shape[0], n_centroids,
        "centroids shape[0]={} != n_centroids={}",
        drafter.centroids.shape[0], n_centroids
    );
    let centroid_dim = drafter.centroids.shape[1];
    assert_eq!(
        centroid_dim, drafter.hidden,
        "centroid_dim {} != drafter.hidden {} (post_projection 전 단계 가정)",
        centroid_dim, drafter.hidden
    );

    let mut cluster_logits = vec![0.0f32; n_centroids];

    // cu67: vq_head centroids GEMV cuda port (env opt-in).
    let cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && super::cuda::drafter_vq_head_cuda(
            &drafter.centroids,
            x_norm,
            &mut cluster_logits,
            n_centroids,
            centroid_dim,
        )
        .unwrap_or(false);
    if !cuda_ok {
        let centroids = dequant_to_f32(&drafter.centroids);
        for c in 0..n_centroids {
            let row = &centroids[c * centroid_dim..(c + 1) * centroid_dim];
            let mut acc = 0.0f32;
            for j in 0..centroid_dim {
                acc += row[j] * x_norm[j];
            }
            cluster_logits[c] = acc;
        }
    }

    VqHeadOutput {
        cluster_logits,
        vocab_logits: None,
    }
}

/// Cluster index → vocab token id 매핑.
///
/// 두 가지 형태가 있다:
///
/// - **Ordered** (Stage C 가정): token_embd 가 cluster-block 순서로 정렬돼
///   있다는 가정 아래, cluster `c` 의 token range 를 균등 분할로 계산.
///   `[c * vocab/n_centroids, (c+1) * vocab/n_centroids)`. 마지막 cluster 는
///   잔여 token 모두 흡수.
/// - **Permutation** (transformers source 정의): `mtp.token_ordering.weight`
///   I32 buffer 가 token_id → cluster slot position 의 explicit permutation 을
///   준다. transformers `Gemma4AssistantMaskedEmbedder.forward`:
///   ```python
///   canonical_positions_per_cluster = token_ordering.view(num_centroids, vocab_size_per_centroid)
///   selected_canonical = canonical_positions_per_cluster[top_k_indices]
///   ```
///   즉 cluster `c` 의 token 집합 = `token_ordering[c*vocab_per_centroid ..
///   (c+1)*vocab_per_centroid]`.
#[derive(Debug, Clone)]
pub enum ClusterTokenTable {
    Ordered {
        vocab_size: usize,
        n_centroids: usize,
    },
    Permutation {
        /// `mtp.token_ordering.weight` 의 owned 사본. canonical_positions_per_cluster
        /// = `token_ordering.view(n_centroids, vocab_per_centroid)`.
        token_ordering: Vec<u32>,
        n_centroids: usize,
        vocab_per_centroid: usize,
    },
}

impl ClusterTokenTable {
    /// Ordered-embedding 가정. cluster `c` 의 token range =
    /// `[c * vocab/n_centroids, (c+1) * vocab/n_centroids)`. 마지막 cluster 는
    /// 잔여 token 모두 흡수.
    pub fn ordered(vocab_size: usize, n_centroids: usize) -> Self {
        assert!(
            n_centroids > 0 && vocab_size >= n_centroids,
            "ClusterTokenTable::ordered: invalid sizes vocab={vocab_size} centroids={n_centroids}"
        );
        Self::Ordered {
            vocab_size,
            n_centroids,
        }
    }

    /// transformers source 의 explicit permutation. `token_ordering` 은
    /// `mtp.token_ordering.weight` 의 owned 사본. shape `[vocab_size]`,
    /// `vocab_per_centroid = vocab_size / n_centroids` 로 reshape.
    pub fn permutation(token_ordering: Vec<u32>, n_centroids: usize) -> Self {
        assert!(
            n_centroids > 0,
            "ClusterTokenTable::permutation: n_centroids must be > 0"
        );
        assert_eq!(
            token_ordering.len() % n_centroids,
            0,
            "ClusterTokenTable::permutation: token_ordering.len() {} not divisible by n_centroids {}",
            token_ordering.len(),
            n_centroids
        );
        let vocab_per_centroid = token_ordering.len() / n_centroids;
        Self::Permutation {
            token_ordering,
            n_centroids,
            vocab_per_centroid,
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Ordered { vocab_size, .. } => *vocab_size,
            Self::Permutation { token_ordering, .. } => token_ordering.len(),
        }
    }

    pub fn n_centroids(&self) -> usize {
        match self {
            Self::Ordered { n_centroids, .. } => *n_centroids,
            Self::Permutation { n_centroids, .. } => *n_centroids,
        }
    }

    /// cluster `c` 의 token id 들. Ordered 는 연속 range, Permutation 은
    /// token_ordering buffer 의 slice.
    pub fn tokens_in_cluster(&self, c: usize) -> Vec<u32> {
        assert!(
            c < self.n_centroids(),
            "tokens_in_cluster: cluster idx {c} out of range (n_centroids={})",
            self.n_centroids()
        );
        match self {
            Self::Ordered {
                vocab_size,
                n_centroids,
            } => {
                let stride = vocab_size / n_centroids;
                let start = c * stride;
                let end = if c + 1 == *n_centroids {
                    *vocab_size
                } else {
                    (c + 1) * stride
                };
                (start..end).map(|i| i as u32).collect()
            }
            Self::Permutation {
                token_ordering,
                vocab_per_centroid,
                n_centroids,
            } => {
                let start = c * vocab_per_centroid;
                let end = if c + 1 == *n_centroids {
                    token_ordering.len()
                } else {
                    (c + 1) * vocab_per_centroid
                };
                token_ordering[start..end].to_vec()
            }
        }
    }
}

/// Top-K cluster 의 token 만 vocab logit 계산. 나머지 token 은 `NEG_INFINITY`.
///
/// transformers `Gemma4AssistantMaskedEmbedder.forward` 에 따르면 lm_head 는
/// `model.embed_tokens.weight` 와 **tied** — drafter 자체의
/// `token_embd.weight` (Q6_K [vocab, hidden=256]) 이 lm_head 로 그대로 쓰인다.
/// 즉 vocab_logits = `x_norm · drafter.token_embd[tok]` (256-dim dot).
/// `post_projection` (256 → 2560 backbone) 은 다른 backbone-space 용도이고
/// vocab logit 계산엔 쓰이지 않는다.
///
/// 단계:
/// 1. `cluster_logits` 의 top-K cluster 를 고른다.
/// 2. 각 top-K cluster 의 token 에 대해 `vocab_logits[tok] = x_norm ·
///    drafter.token_embd[tok]` (drafter 의 256-dim row).
pub fn vocab_logits_in_top_k_clusters(
    drafter: &Drafter,
    cluster_logits: &[f32],
    x_norm: &[f32],
    top_k_clusters: usize,
    cluster_token_table: &ClusterTokenTable,
) -> Vec<f32> {
    assert_eq!(
        x_norm.len(),
        drafter.hidden,
        "vocab_logits: x_norm len {} != drafter.hidden {}",
        x_norm.len(),
        drafter.hidden
    );
    assert_eq!(
        cluster_logits.len() as u32,
        drafter.n_centroids,
        "vocab_logits: cluster_logits len {} != n_centroids {}",
        cluster_logits.len(),
        drafter.n_centroids
    );
    assert_eq!(
        cluster_token_table.n_centroids(),
        drafter.n_centroids as usize,
        "vocab_logits: table n_centroids mismatch"
    );

    // 1. Top-K cluster (cluster_logits 의 desc 정렬).
    let k = top_k_clusters.min(cluster_logits.len()).max(1);
    let mut idx: Vec<usize> = (0..cluster_logits.len()).collect();
    idx.sort_by(|&a, &b| {
        cluster_logits[b]
            .partial_cmp(&cluster_logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);

    // 2. Vocab logits — tied lm_head (drafter.token_embd_row).
    let vocab_size = cluster_token_table.vocab_size();
    let mut vocab_logits = vec![f32::NEG_INFINITY; vocab_size];
    for c in idx {
        for tok in cluster_token_table.tokens_in_cluster(c) {
            let e = drafter.token_embd_row(tok);
            assert_eq!(
                e.len(),
                drafter.hidden,
                "drafter.token_embd_row({tok}) returned len {} != drafter.hidden {}",
                e.len(),
                drafter.hidden
            );
            let mut acc = 0.0f32;
            for j in 0..drafter.hidden {
                acc += x_norm[j] * e[j];
            }
            vocab_logits[tok as usize] = acc;
        }
    }
    // post_projection 은 vocab logit 경로에서 쓰이지 않음 (tied lm_head 라 hidden=256
    // 공간에서 바로 dot product). dequant 호출도 제거되어 함수가 가벼워진다.
    vocab_logits
}
