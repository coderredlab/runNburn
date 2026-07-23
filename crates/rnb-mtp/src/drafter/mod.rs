//! Drafter (small assistant model) weight loader + forward orchestration.
//!
//! Stage B (mt-Task-2) — `gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant` GGUF
//! sidecars 를 zero-copy mmap 으로 로드한다. attention layout 의 SWA vs full
//! 분기, 누락된 `attn_k` / `attn_v` (drafter 는 target 의 KV 를 공유) 도 함께
//! 처리한다.
//!
//! Stage γ (mt84) — mt83 의 4-layer custom cross-attention (`cross_attention.rs`)
//! 폐기. transformers source verbatim 으로 `Gemma4TextModel` backbone reuse 방식
//! 의 `backbone.rs` 채택. drafter 의 모든 4 layer 가 `is_kv_shared_layer=True`
//! 라 K/V projection 없이 `SharedKvStates` 에서 K/V 가져옴. layer_scalar 위치
//! 정정 (decoder_layer 마지막 multiplication), Q-only RoPE.
//!
//! - `backbone` — pre_projection (5120 → 256) + Gemma4TextModel backbone (4 layer)
//!   + output_norm + post_projection (256 → 2560) + VQ masked embedding.
//! - `vq_head` — cluster_logits 와 (top-K) vocab_logits 두 단계. mt83 verbatim.
//!
//! Stage δ (mt84) — calibrate 가 single-variant 로 재설계됨. mt83 의 9-variant
//! grid (`KvShareMap` × `ClusterTokenStrategy`) 폐기, calibration test 는
//! `tests/drafter_backbone_calibrate_test.rs` 로 이전. `calibrate` 모듈은
//! diagnostic helper 가 필요해질 때 까지 placeholder.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`.

pub mod backbone;
pub mod calibrate;
pub(crate) mod cuda;
pub(crate) mod dequant;
pub mod loader;
pub mod shared_kv;
pub mod types;
pub mod vq_head;

pub use backbone::{drafter_forward, DrafterForwardOutput};
pub use cuda::{drafter_prewarm_weights_cuda, drafter_prewarm_weights_cuda_full};
pub use loader::{load_drafter, DrafterLoadError};
pub use shared_kv::{SharedKvLayer, SharedKvStates};
pub use types::{Drafter, DrafterLayer, TensorView, VQCodebook};
pub use vq_head::{
    vocab_logits_in_top_k_clusters, vq_head_forward, ClusterTokenTable, VqHeadOutput,
};
