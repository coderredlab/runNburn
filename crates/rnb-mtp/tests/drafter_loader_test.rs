//! Stage B acceptance test — load the gemma4_assistant drafter GGUF and
//! verify 51-tensor layout (SWA vs full split, head_dim per layer, global
//! VQ projection shapes).
//!
//! Fixture 위치는 `RNB_GEMMA4_ASSISTANT_FIXTURE` 환경변수로 지정 (Task 1 패턴).
//! - env 미설정: skip 메시지 출력 후 PASS (로컬 quick run).
//! - env 설정 + 파일 없음: panic.
//! - env 설정 + 파일 있음: 전체 assert 수행.

use rnb_mtp::drafter::load_drafter;
use rnb_mtp::drafter::types::TensorView;

const FIXTURE_ENV: &str = "RNB_GEMMA4_ASSISTANT_FIXTURE";

#[test]
fn loads_e4b_assistant_drafter() {
    let Some(fixture) = std::env::var_os(FIXTURE_ENV) else {
        eprintln!(
            "skip: ${FIXTURE_ENV} not set — local quick run. \
             CI must export this to a real .gguf path."
        );
        return;
    };
    let path = std::path::PathBuf::from(&fixture);
    assert!(
        path.exists(),
        "${FIXTURE_ENV} points to missing file: {path:?}"
    );

    let drafter = load_drafter(&path).expect("load");

    // --- top-level ----------------------------------------------------------
    assert_eq!(drafter.block_count, 4);
    assert_eq!(drafter.hidden, 256);
    assert_eq!(drafter.backbone_hidden, 2560);
    assert_eq!(drafter.n_centroids, 2048);
    assert_eq!(drafter.centroid_top_k, 32);
    assert_eq!(drafter.sliding_window, 512);
    assert_eq!(drafter.layers.len(), 4);

    // --- global tensors -----------------------------------------------------
    assert_shape(&drafter.token_embd, &[262144, 256], "token_embd");
    assert_eq!(drafter.output_norm.len(), 256);
    // rope_freqs holds inv freq table = head_dim_full / 2 = 256.
    assert_eq!(drafter.rope_freqs.len(), 256);
    assert_shape(&drafter.pre_projection, &[256, 5120], "mtp.pre_projection");
    assert_shape(
        &drafter.post_projection,
        &[2560, 256],
        "mtp.post_projection",
    );
    assert_shape(&drafter.centroids, &[2048, 256], "mtp.centroids");

    // --- SWA pattern: layers 0,1,2 sliding window; layer 3 full -------------
    assert!(drafter.layers[0].is_sliding_window);
    assert!(drafter.layers[1].is_sliding_window);
    assert!(drafter.layers[2].is_sliding_window);
    assert!(!drafter.layers[3].is_sliding_window);

    assert_eq!(drafter.layers[0].head_dim, 256);
    assert_eq!(drafter.layers[3].head_dim, 512);

    // GQA: n_heads=4, n_kv_heads=2 for every drafter layer.
    for (i, layer) in drafter.layers.iter().enumerate() {
        assert_eq!(layer.layer_idx, i, "layer_idx mismatch");
        assert_eq!(layer.n_heads, 4, "layer {i} n_heads");
        assert_eq!(layer.n_kv_heads, 2, "layer {i} n_kv_heads");
        assert_eq!(layer.attn_norm.len(), 256);
        assert_eq!(layer.ffn_norm.len(), 256);
        assert_eq!(layer.post_attention_norm.len(), 256);
        assert_eq!(layer.post_ffw_norm.len(), 256);
        // attn_q_norm length tracks head_dim (256 for SWA, 512 for full).
        assert_eq!(layer.attn_q_norm.len(), layer.head_dim);
    }

    // --- per-layer quantized tensor shapes ----------------------------------
    // SWA layers (0,1,2): attn_q [1024,256], attn_output [256,1024].
    for &i in &[0usize, 1, 2] {
        let layer = &drafter.layers[i];
        assert_shape(&layer.attn_q, &[1024, 256], &format!("blk.{i}.attn_q"));
        assert_shape(
            &layer.attn_output,
            &[256, 1024],
            &format!("blk.{i}.attn_output"),
        );
    }
    // Full layer (3): attn_q [2048,256], attn_output [256,2048].
    let l3 = &drafter.layers[3];
    assert_shape(&l3.attn_q, &[2048, 256], "blk.3.attn_q");
    assert_shape(&l3.attn_output, &[256, 2048], "blk.3.attn_output");

    // FFN shapes identical across layers; ffn_down dtype 분기 (Q4_K layers 0-1,
    // Q6_K layers 2-3) 만 따로 점검.
    for layer in &drafter.layers {
        assert_shape(
            &layer.ffn_gate,
            &[2048, 256],
            &format!("blk.{}.ffn_gate", layer.layer_idx),
        );
        assert_shape(
            &layer.ffn_up,
            &[2048, 256],
            &format!("blk.{}.ffn_up", layer.layer_idx),
        );
        assert_shape(
            &layer.ffn_down,
            &[256, 2048],
            &format!("blk.{}.ffn_down", layer.layer_idx),
        );
    }
    use rnb_loader::gguf::types::GGMLType;
    assert_eq!(drafter.layers[0].ffn_down.ggml_type, GGMLType::Q4_K);
    assert_eq!(drafter.layers[1].ffn_down.ggml_type, GGMLType::Q4_K);
    assert_eq!(drafter.layers[2].ffn_down.ggml_type, GGMLType::Q6_K);
    assert_eq!(drafter.layers[3].ffn_down.ggml_type, GGMLType::Q6_K);

    // --- zero-copy mmap sanity check ----------------------------------------
    // 모든 TensorView 의 byte range 가 mmap 안에 들어와야 한다.
    let mmap_len = drafter.token_embd.mmap.len();
    let probe = |v: &TensorView, label: &str| {
        assert!(
            v.offset + v.len <= mmap_len,
            "{label}: offset+len {} > mmap_len {mmap_len}",
            v.offset + v.len
        );
    };
    probe(&drafter.token_embd, "token_embd");
    probe(&drafter.pre_projection, "pre_projection");
    probe(&drafter.post_projection, "post_projection");
    probe(&drafter.centroids, "centroids");
    for layer in &drafter.layers {
        probe(&layer.attn_q, &format!("blk.{}.attn_q", layer.layer_idx));
        probe(
            &layer.attn_output,
            &format!("blk.{}.attn_output", layer.layer_idx),
        );
        probe(
            &layer.ffn_gate,
            &format!("blk.{}.ffn_gate", layer.layer_idx),
        );
        probe(&layer.ffn_up, &format!("blk.{}.ffn_up", layer.layer_idx));
        probe(
            &layer.ffn_down,
            &format!("blk.{}.ffn_down", layer.layer_idx),
        );
    }
}

fn assert_shape(view: &TensorView, expected: &[usize], label: &str) {
    assert_eq!(view.shape, expected, "{label} shape");
}
