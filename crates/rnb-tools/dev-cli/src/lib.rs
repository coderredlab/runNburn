//! `rnb-dev-tools` library surface.
//!
//! This crate primarily ships standalone debug/probe/bench binaries (see
//! `src/bin/*.rs`). The library facade exists so integration tests and select
//! binaries can share reusable instrumentation modules without each `#[path]`-
//! importing the same source file.
//!
//! New modules added here must be **dev-tooling only** — production engine
//! semantics live in `rnb-llm`. Anything we put behind `pub mod` should be
//! documented as instrumentation/research API with the owning session prefix
//! (e.g. `mt91`) in the module doc.

pub mod mtk_gguf_mlp;
pub mod q4_operator_microbench;
