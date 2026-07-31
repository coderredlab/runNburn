use rnb_llm::auto_drafter::find_sibling_drafter;
use std::fs;
use tempfile::tempdir;

#[test]
fn finds_direct_sibling_assistant_file() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("gemma-4-E4B-it.gguf");
    let assistant = dir.path().join("gemma-4-E4B-it-assistant.Q4_K_M.gguf");
    fs::write(&target, b"placeholder").unwrap();
    fs::write(&assistant, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), Some(assistant));
}

#[test]
fn finds_assistant_in_mtp_subdir() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("gemma-4-E4B-it.gguf");
    let sub = dir.path().join("gemma-4-E4B-it-mtp");
    fs::create_dir(&sub).unwrap();
    let assistant = sub.join("gemma-4-E4B-it-assistant.Q4_K_M.gguf");
    fs::write(&target, b"placeholder").unwrap();
    fs::write(&assistant, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), Some(assistant));
}

#[test]
fn finds_q8_mtp_model_in_standard_subdir_for_quantized_target() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("gemma-4-E2B-it-Q4_K_M.gguf");
    let sub = dir.path().join("MTP");
    fs::create_dir(&sub).unwrap();
    let assistant = sub.join("mtp-gemma-4-E2B-it-Q8_0.gguf");
    fs::write(&target, b"placeholder").unwrap();
    fs::write(&assistant, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), Some(assistant));
}

#[test]
fn returns_none_when_no_sibling() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("qwen3.5-0.8B.gguf");
    fs::write(&target, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), None);
}

#[test]
fn finds_assistant_in_sibling_of_parent_mtp_dir() {
    // Layout used in this repo: models/gemma-4-E4B/<target>.gguf alongside
    // models/gemma-4-E4B-mtp/<target>-assistant.Q4_K_M.gguf.
    let dir = tempdir().unwrap();
    let target_dir = dir.path().join("gemma-4-E4B");
    let mtp_dir = dir.path().join("gemma-4-E4B-mtp");
    fs::create_dir(&target_dir).unwrap();
    fs::create_dir(&mtp_dir).unwrap();
    let target = target_dir.join("gemma-4-E4B-it.gguf");
    let assistant = mtp_dir.join("gemma-4-E4B-it-assistant.Q4_K_M.gguf");
    fs::write(&target, b"placeholder").unwrap();
    fs::write(&assistant, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), Some(assistant));
}

#[test]
fn ignores_unrelated_assistant_named_files() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("gemma-4-E4B-it.gguf");
    let helper = dir.path().join("gemma-4-E4B-it-helper.gguf");
    fs::write(&target, b"placeholder").unwrap();
    fs::write(&helper, b"placeholder").unwrap();
    assert_eq!(find_sibling_drafter(&target), None);
}
