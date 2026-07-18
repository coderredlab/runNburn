//! Gemma family runtime helpers (attention rot, rope, scale, PLE branch, ...).
//!
//! Most of these take module-private types from `engine.rs` and are used on
//! both prefill and decode paths.

#![allow(unused_imports)]

use crate::engine::*;

mod attention;
mod decode_moe;
mod env;
mod loading;
pub(in crate::engine) mod moe_types;
mod moe_view;
mod output;
pub(in crate::engine) mod packed_wiring;
mod per_layer;
mod prefill_moe;
#[cfg(target_arch = "aarch64")]
mod prefill_moe_expert_group;
mod prefill_moe_expert_major;
mod runtime;
mod weights;

pub(in crate::engine) use attention::*;
pub(in crate::engine) use decode_moe::decode_ffn_gemma4_moe_hybrid;
pub(in crate::engine) use env::*;
pub(in crate::engine) use loading::*;
pub(in crate::engine) use output::*;
pub(in crate::engine) use per_layer::*;
pub(in crate::engine) use prefill_moe::forward_ffn_gemma4_moe_hybrid;
pub(in crate::engine) use runtime::*;
pub(in crate::engine) use weights::*;
