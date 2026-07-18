//! MoE (Mixture of Experts) facade.
//!
//! Model-specific execution lives under `engine::models::*`. This module keeps
//! the legacy `engine::moe::*` import surface stable for callers and tests.

pub use super::models::gemma::moe_types::{
    down_bytes_per_row as gemma_down_bytes_per_row, per_expert_down_bytes,
    per_expert_gate_up_bytes, per_expert_shadow_gate_up_bytes, MoeLayerView,
};
#[cfg(test)]
use super::models::shared_expert_moe::moe_types::down_bytes_per_row;
pub use super::models::shared_expert_moe::moe_types::SharedExpertMoEView;
#[cfg(test)]
use super::moe_profile::*;
#[cfg(test)]
use super::moe_section_layout::moe_section_gate_up_unit_size;
pub use super::moe_types::{
    q2k_bytes_per_row, q4k_bytes_per_row, q5_1_bytes_per_row, q5k_bytes_per_row, q6k_bytes_per_row,
    q8_0_bytes_per_row,
};
#[cfg(test)]
use rnb_loader::GGMLType;

#[cfg(test)]
pub(in crate::engine) mod tests;
