//! Nemotron-H text-backbone support.
//!
//! Keep Nemotron-H code here instead of extending Qwen/Gemma modules. The text
//! backbone has separate Mamba2, MoE-only, and attention-only blocks, so sharing
//! Qwen35 GDN/MoE structs would hide the real runtime boundary.

pub(super) mod attention;
pub(in crate::engine) mod mamba;
pub(in crate::engine) mod moe;
pub(in crate::engine) mod policy;
pub(super) mod reasoning;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum NemotronTextLayerKind {
    Mamba2,
    MoE,
    Attention,
}

#[allow(dead_code)]
pub(super) fn layer_kind_from_pattern_byte(byte: u8) -> Option<NemotronTextLayerKind> {
    match byte {
        b'M' => Some(NemotronTextLayerKind::Mamba2),
        b'E' => Some(NemotronTextLayerKind::MoE),
        b'*' | b'A' => Some(NemotronTextLayerKind::Attention),
        _ => None,
    }
}
