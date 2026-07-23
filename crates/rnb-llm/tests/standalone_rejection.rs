//! Standalone `.rnb` files are not accepted by `Engine::from_gguf`.
//!
//! The product input contract is GGUF-only. Rejection is based on the path
//! extension and therefore happens before any file I/O.

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
