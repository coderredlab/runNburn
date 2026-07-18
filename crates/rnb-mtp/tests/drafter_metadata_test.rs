//! Task 1 / Stage A acceptance test.
//!
//! GGUF metadata key `general.architecture = "gemma4_assistant"` 를 panic 없이
//! 파싱하고, drafter 전용 키 (`n_centroids`, `centroid_top_k`, `n_embd_backbone`,
//! `shared_kv_layers`, `use_ordered_embeddings`, `sliding_window_pattern`,
//! `requires_target_arch`, `attention.key_length`, `attention.key_length_swa`,
//! `rope.freq_base`, `rope.freq_base_swa`) 가 `AssistantMetadata` 로 추출되는지
//! fixture round-trip 으로 검증한다.
//!
//! Fixture 위치: 환경변수 `RNB_GEMMA4_ASSISTANT_FIXTURE` 에 절대경로로 지정.
//!
//! - env 미설정: 메시지 출력 후 skip (로컬 quick run, CI secret-free 환경 대비)
//! - env 설정됐는데 파일 없음: panic (CI 검증 — fixture 경로 오타/누락 감지)
//! - env 설정됐고 파일 있음: 정상 assert
//!
//! Spec: `docs/superpowers/specs/2026-05-13-gemma4-assistant-drafter-design.md`.

use rnb_loader::arch::Architecture;

const FIXTURE_ENV: &str = "RNB_GEMMA4_ASSISTANT_FIXTURE";

#[test]
fn gemma4_assistant_metadata_round_trip() {
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

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let gguf = rnb_loader::gguf::parser::GGUFFile::parse(&mmap[..]).unwrap();
    let metadata = rnb_loader::arch::extract_metadata(&gguf.metadata).unwrap();

    assert_eq!(metadata.architecture, Architecture::Gemma4Assistant);
    let assistant = metadata.assistant.expect("assistant block present");
    assert_eq!(assistant.n_centroids, 2048);
    assert_eq!(assistant.centroid_top_k, 32);
    assert_eq!(assistant.n_embd_backbone, 2560);
    assert!(assistant.use_ordered_embeddings);
    assert_eq!(assistant.requires_target_arch, "gemma4");
    assert_eq!(assistant.shared_kv_layers, 4);
    assert_eq!(
        assistant.sliding_window_pattern,
        vec![true, true, true, false]
    );
    assert_eq!(assistant.key_length_full, 512);
    assert_eq!(assistant.key_length_swa, 256);
    assert!((assistant.rope_freq_base_full - 1_000_000.0).abs() < 0.1);
    assert!((assistant.rope_freq_base_swa - 10_000.0).abs() < 0.1);
}
