//! mt84 Stage γ acceptance test — drafter backbone forward dim flow.
//!
//! transformers source verbatim 의 `Gemma4TextModel` backbone reuse architecture
//! 검증. drafter weight (51 tensor) 가 정상 로드된 상태에서:
//!
//! - `drafter_forward` 입력 `inputs_embeds.len() == 2 * backbone_hidden`
//!   (= 5120) → 정상 처리
//! - 출력 `logits.len() == vocab_size` (= 262144), `projected_hidden.len() ==
//!   backbone_hidden` (= 2560)
//! - 두 출력 모두 finite (no NaN/Inf)
//! - top-K cluster 안 token 의 vocab logit 은 `> -inf` (mask 가 일부 token 만
//!   masking)
//!
//! Fixture:
//! - `RNB_DRAFTER_MODEL` — drafter (`*-it-assistant.Q4_K_M.gguf`).
//!
//! 미설정 시 `#[ignore]` 로 cargo test 가 자동 skip.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §1-§5.

use rnb_mtp::drafter::{drafter_forward, load_drafter};
use rnb_mtp::{SharedKvLayer, SharedKvStates};
use std::path::PathBuf;

const DRAFTER_ENV: &str = "RNB_DRAFTER_MODEL";

/// Mock `SharedKvStates` — drafter forward 가 expected layout 로 K/V 를 받는지만
/// 검증. layer head_dim: sliding=256, full=512, n_kv_heads=2.
fn mock_shared_kv_states(seq_len: usize) -> SharedKvStates {
    let sliding_head_dim = 256;
    let full_head_dim = 512;
    let n_kv_heads = 2;
    let sliding_len = n_kv_heads * seq_len * sliding_head_dim;
    let full_len = n_kv_heads * seq_len * full_head_dim;
    SharedKvStates {
        sliding_attention: SharedKvLayer {
            k: vec![0.01f32; sliding_len],
            v: vec![0.01f32; sliding_len],
            n_kv_heads,
            seq_len,
            head_dim: sliding_head_dim,
        },
        full_attention: SharedKvLayer {
            k: vec![0.01f32; full_len],
            v: vec![0.01f32; full_len],
            n_kv_heads,
            seq_len,
            head_dim: full_head_dim,
        },
    }
}

#[test]
#[ignore = "needs RNB_DRAFTER_MODEL (drafter GGUF path)"]
fn drafter_backbone_dim_flow() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    assert!(
        drafter_path.exists(),
        "${DRAFTER_ENV} points to missing file: {drafter_path:?}"
    );

    let drafter = load_drafter(&drafter_path).expect("load drafter");
    assert_eq!(drafter.block_count, 4, "drafter block_count != 4");
    assert_eq!(drafter.hidden, 256, "drafter hidden != 256");
    assert_eq!(
        drafter.backbone_hidden, 2560,
        "drafter backbone_hidden != 2560"
    );

    let inputs_embeds = vec![0.1f32; 2 * drafter.backbone_hidden]; // 5120
    let seq_len = 10usize;
    let shared_kv = mock_shared_kv_states(seq_len);
    let position_id = (seq_len - 1) as u32; // last validated token's position

    let out = drafter_forward(&drafter, &inputs_embeds, &shared_kv, position_id);

    // Dim flow assertions.
    let vocab_size = drafter.token_ordering.len();
    assert_eq!(
        out.logits.len(),
        vocab_size,
        "logits.len() != vocab_size {vocab_size}"
    );
    assert_eq!(
        out.projected_hidden.len(),
        drafter.backbone_hidden,
        "projected_hidden.len() != backbone_hidden"
    );

    // projected_hidden 은 모두 finite.
    for (i, v) in out.projected_hidden.iter().enumerate() {
        assert!(v.is_finite(), "non-finite projected_hidden at idx {i}: {v}");
    }

    // logits 는 `NEG_INFINITY` (mask) 또는 finite. NaN 은 금지.
    let mut max_logit = f32::NEG_INFINITY;
    let mut some_finite = false;
    for (i, &l) in out.logits.iter().enumerate() {
        assert!(!l.is_nan(), "NaN logit at idx {i}");
        if l.is_finite() {
            some_finite = true;
            if l > max_logit {
                max_logit = l;
            }
        }
    }
    assert!(some_finite, "no finite logit produced — all -inf");
    assert!(
        max_logit > f32::MIN,
        "max logit {max_logit} not above f32::MIN (no top-K cluster activated?)"
    );
}
