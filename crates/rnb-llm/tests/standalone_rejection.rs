//! Phase 1 task 18: standalone `.rnb` files are no longer accepted by
//! `Engine::from_gguf`. The new policy is: callers pass a GGUF model path,
//! and any sidecar/cache `.rnb` is auto-resolved by the runNburn cache layer.
//!
//! This integration test guards the rejection path. It does not need a real
//! file on disk because the rejection is based on the path extension and
//! must happen before any I/O.

use std::path::PathBuf;

use rnb_llm::Engine;

#[test]
fn from_gguf_rejects_standalone_rnb_input() {
    let path = PathBuf::from("/tmp/some-model.rnb");
    let err = Engine::from_gguf(&path).err().expect("must reject .rnb");
    let msg = format!("{err}");
    assert!(
        msg.contains(".rnb") || msg.contains("standalone") || msg.contains("GGUF"),
        "expected rejection mentioning .rnb / standalone / GGUF, got: {msg}"
    );
}
